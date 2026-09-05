"""OpenSearch backend: one shared k-NN index per collection, not per video.

Replaces the earlier per-video on-disk FAISS + SQLite pair (one `.faiss` /
`.sqlite` file per `data/<video_id>/`) with two archive-wide OpenSearch
indexes -- `vvs_visual` and `vvs_text` -- so every video's vectors live in the
same place, distinguished by a `video_id` field on each document. This is the
storage shape `docs/CONFLUENCE_ONPREM.md` describes (one shared index, not one
per asset), scaled down to a single local node instead of a sharded cluster.

Two collections, not one, because visual and text embeddings are different
widths (the visual encoder's dimension vs. BGE's fixed 384) and OpenSearch's
`knn_vector` field has one fixed dimension per index -- mixing widths in one
index is not possible. A consequence worth knowing: unlike the old per-video
FAISS files, this shared index locks in whichever visual encoder wrote it
first. Switching visual encoders (e.g. CLIP 512-d -> Qwen-VL 2048-d) once the
archive is non-empty requires a fresh index, not a mixed one -- `create()`
raises rather than silently corrupting the mapping.

`search()` intentionally does not use OpenSearch's server-side k-NN filter
(`knn.filter`). That feature's availability and exact query shape varies by
OpenSearch version and k-NN engine, and this store only ever holds a few
hundred to a few thousand vectors per video across a handful of videos --
POC scale, per the same tradeoff the old FAISS thread-pinning comment made.
So: over-fetch unfiltered, then filter to one `video_id` in Python. Simple,
version-independent, and correct at this scale. The real on-prem design in
CONFLUENCE_ONPREM.md uses native filtered k-NN, correctly, at cluster scale --
this is the local-demo tradeoff, not the production one.

The `hybrid` search stays out of scope here too: OpenSearch's `hybrid` query
type fuses k-NN and BM25 scores server-side via a search pipeline, but doing
so would discard the separate per-channel ranks (visual vs. transcript vs.
OCR vs. scene) that `retrieve/search.py`'s RRF fusion, temporal boosts, and
motion/flash heuristics depend on. Both collections here are searched with
k-NN only (the text collection embeds OCR/transcript/scene text with BGE, same
as before), which is why `retrieve/search.py` needed no changes at all --
this module is a drop-in for `FaissStore` against the same `VectorStore`
surface.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import numpy as np

from vvs.config import Settings, load_settings
from vvs.retrieve.store import StoreHit, VectorPayload

_INDEX_FOR = {"visual": "vvs_visual", "text": "vvs_text"}

# Over-fetch factor for the unfiltered-then-filter-in-Python search above.
# Generous because a POC archive is tens of videos, not millions.
_OVERFETCH_MIN = 500


@lru_cache(maxsize=4)
def _client(url: str, timeout_s: float):
    from opensearchpy import OpenSearch

    return OpenSearch(
        hosts=[url],
        http_compress=True,
        use_ssl=url.startswith("https://"),
        verify_certs=False,
        timeout=timeout_s,
        max_retries=2,
        retry_on_timeout=True,
    )


def _connect(settings: Settings | None):
    cfg = settings or load_settings()
    return _client(cfg.opensearch_url, cfg.opensearch_timeout_s), cfg


def ping(settings: Settings | None = None) -> bool:
    client, _ = _connect(settings)
    try:
        return bool(client.ping())
    except Exception:
        return False


class OpenSearchStore:
    def __init__(
        self,
        video_id: str,
        name: str,
        *,
        settings: Settings | None = None,
        index_override: str | None = None,
    ) -> None:
        if name not in _INDEX_FOR:
            raise ValueError(f"unknown store name {name!r}; expected one of {sorted(_INDEX_FOR)}")
        self.video_id = video_id
        self.name = name
        self.dim = 0
        self._client, self._settings = _connect(settings)
        # index_override exists only so tests can point at a disposable index
        # instead of the real shared archive (which already has real vectors
        # at a fixed dimension and would trip the mismatch guard below).
        self.index = index_override or _INDEX_FOR[name]

    # -- lifecycle -----------------------------------------------------

    def create(self, dim: int) -> None:
        """Ensure the shared index exists, then drop any prior rows for this video.

        Mirrors FaissStore.create()'s truncate-on-rebuild semantics, scoped to
        this video_id within the shared index rather than to a whole file.
        """
        self.dim = int(dim)
        self._ensure_index(self.dim)
        self._client.delete_by_query(
            index=self.index,
            body={"query": {"term": {"video_id": self.video_id}}},
            conflicts="proceed",
            refresh=True,
            ignore=[404],
        )

    def open(self, dim: int) -> None:
        if not self._client.indices.exists(index=self.index):
            self.dim = int(dim)
            return
        mapped = self._mapped_dim()
        self.dim = mapped if mapped else int(dim)

    def _ensure_index(self, dim: int) -> None:
        if self._client.indices.exists(index=self.index):
            mapped = self._mapped_dim()
            if mapped and mapped != dim:
                raise SystemExit(
                    f"{self.index} is already a {mapped}-d index (from an earlier video's "
                    f"visual encoder); this video's encoder produced {dim}-d vectors. One "
                    "shared index has one fixed vector width -- see the module docstring in "
                    "retrieve/opensearch_store.py. Use a fresh VVS_OPENSEARCH index (or drop "
                    f"`{self.index}`) to switch encoders."
                )
            return
        self._client.indices.create(
            index=self.index,
            body={
                "settings": {"index": {"knn": True}},
                "mappings": {
                    "properties": {
                        "video_id": {"type": "keyword"},
                        "modality": {"type": "keyword"},
                        "vector": {
                            "type": "knn_vector",
                            "dimension": int(dim),
                            "method": {
                                "name": "hnsw",
                                "space_type": "innerproduct",
                                "engine": "lucene",
                            },
                        },
                    }
                },
            },
        )

    def _mapped_dim(self) -> int:
        mapping = self._client.indices.get_mapping(index=self.index)
        props = next(iter(mapping.values()))["mappings"]["properties"]
        return int(props.get("vector", {}).get("dimension", 0) or 0)

    # -- writes ----------------------------------------------------------

    def add(self, vectors: np.ndarray, payloads: list[VectorPayload]) -> None:
        if vectors.ndim != 2:
            raise ValueError("vectors must be (n, dim)")
        if len(payloads) != vectors.shape[0]:
            raise ValueError("payloads length must match vectors")
        if vectors.shape[0] == 0:
            return
        if self.dim == 0:
            self.dim = int(vectors.shape[1])
            self._ensure_index(self.dim)
        mat = np.ascontiguousarray(vectors.astype(np.float32))
        actions = []
        for vec, payload in zip(mat, payloads, strict=True):
            payload.asset_id = self.video_id
            doc = payload.model_dump()
            doc["video_id"] = self.video_id
            doc["vector"] = vec.tolist()
            actions.append({"index": {"_index": self.index}})
            actions.append(doc)
        self._client.bulk(body=actions, refresh=True)

    # -- reads -------------------------------------------------------------

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[StoreHit]:
        if top_k <= 0 or not self._client.indices.exists(index=self.index):
            return []
        vec = np.ascontiguousarray(np.asarray(query_vec, dtype=np.float32).reshape(-1))
        k = max(top_k, _OVERFETCH_MIN)
        resp = self._client.search(
            index=self.index,
            body={
                "size": k,
                "query": {"knn": {"vector": {"vector": vec.tolist(), "k": k}}},
            },
        )
        hits: list[StoreHit] = []
        for row in resp.get("hits", {}).get("hits", []):
            source = row["_source"]
            if source.get("video_id") != self.video_id:
                continue
            if filters and not _payload_matches(source, filters):
                continue
            hits.append(
                StoreHit(vector_id=0, score=float(row["_score"]), payload=_payload_from_source(source))
            )
            if len(hits) >= top_k:
                break
        return hits

    def all_payloads(self, modality: str | None = None) -> list[VectorPayload]:
        """Every payload for this video, in insertion order. Used to re-embed in place."""
        if not self._client.indices.exists(index=self.index):
            return []
        must: list[dict[str, Any]] = [{"term": {"video_id": self.video_id}}]
        if modality:
            must.append({"term": {"modality": modality}})
        rows: list[VectorPayload] = []
        resp = self._client.search(
            index=self.index,
            body={"size": 1000, "query": {"bool": {"must": must}}, "sort": [{"t_s": "asc"}]},
            scroll="1m",
        )
        scroll_id = resp.get("_scroll_id")
        try:
            while True:
                batch = resp.get("hits", {}).get("hits", [])
                if not batch:
                    break
                for row in batch:
                    rows.append(_payload_from_source(row["_source"]))
                resp = self._client.scroll(scroll_id=scroll_id, scroll="1m")
                scroll_id = resp.get("_scroll_id")
        finally:
            if scroll_id:
                self._client.clear_scroll(scroll_id=scroll_id, ignore=[404])
        return rows

    def ntotal(self) -> int:
        if not self._client.indices.exists(index=self.index):
            return 0
        resp = self._client.count(
            index=self.index, body={"query": {"term": {"video_id": self.video_id}}}
        )
        return int(resp.get("count", 0))

    def close(self) -> None:
        # A shared HTTP client outlives any one store; nothing to release per video.
        pass


def clone_video(
    name: str, source_id: str, dest_id: str, *, settings: Settings | None = None
) -> int:
    """Copy every document for source_id into dest_id, vectors included.

    Used by `retrieve/reembed.py` to A/B a visual encoder: the text channel
    (transcript/OCR/scene, still BGE-embedded) is identical between the two
    ids, so it is copied rather than re-embedded. Returns the number of
    documents copied.
    """
    client, _ = _connect(settings)
    index = _INDEX_FOR[name]
    if not client.indices.exists(index=index):
        return 0
    copied = 0
    resp = client.search(
        index=index,
        body={"size": 1000, "query": {"term": {"video_id": source_id}}},
        scroll="1m",
    )
    scroll_id = resp.get("_scroll_id")
    try:
        while True:
            batch = resp.get("hits", {}).get("hits", [])
            if not batch:
                break
            actions = []
            for row in batch:
                doc = dict(row["_source"])
                doc["video_id"] = dest_id
                doc["asset_id"] = dest_id
                actions.append({"index": {"_index": index}})
                actions.append(doc)
            client.bulk(body=actions, refresh=True)
            copied += len(batch)
            resp = client.scroll(scroll_id=scroll_id, scroll="1m")
            scroll_id = resp.get("_scroll_id")
    finally:
        if scroll_id:
            client.clear_scroll(scroll_id=scroll_id, ignore=[404])
    return copied


def video_exists(video_id: str, *, settings: Settings | None = None) -> bool:
    """Any document at all for this video, in either collection."""
    client, _ = _connect(settings)
    for index in _INDEX_FOR.values():
        try:
            if not client.indices.exists(index=index):
                continue
            resp = client.count(index=index, body={"query": {"term": {"video_id": video_id}}})
            if int(resp.get("count", 0)) > 0:
                return True
        except Exception:
            continue
    return False


def delete_video(video_id: str, *, settings: Settings | None = None) -> None:
    client, _ = _connect(settings)
    for index in _INDEX_FOR.values():
        if not client.indices.exists(index=index):
            continue
        client.delete_by_query(
            index=index,
            body={"query": {"term": {"video_id": video_id}}},
            conflicts="proceed",
            refresh=True,
            ignore=[404],
        )


def distinct_video_ids(*, settings: Settings | None = None) -> list[str]:
    """Every video_id that has at least one document, across both collections."""
    client, _ = _connect(settings)
    found: set[str] = set()
    for index in _INDEX_FOR.values():
        if not client.indices.exists(index=index):
            continue
        resp = client.search(
            index=index,
            body={
                "size": 0,
                "aggs": {"videos": {"terms": {"field": "video_id", "size": 10_000}}},
            },
        )
        for bucket in resp.get("aggregations", {}).get("videos", {}).get("buckets", []):
            found.add(str(bucket["key"]))
    return sorted(found)


def _payload_from_source(source: dict[str, Any]) -> VectorPayload:
    return VectorPayload(**{k: v for k, v in source.items() if k not in ("vector", "video_id")})


def _payload_matches(source: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, value in filters.items():
        if value is None:
            continue
        got = source.get(key)
        if isinstance(value, (list, tuple, set)):
            if got not in value:
                return False
        elif got != value:
            return False
    return True


# Local dev convenience: fail fast with a clear message instead of a raw
# connection-refused traceback the first time something touches OpenSearch.
if os.environ.get("VVS_OPENSEARCH_STRICT_PING"):  # pragma: no cover - opt-in diagnostic
    if not ping():
        raise SystemExit(
            "Cannot reach OpenSearch. Run `docker compose up -d` "
            "(see docker-compose.yml) or set VVS_OPENSEARCH_URL."
        )
