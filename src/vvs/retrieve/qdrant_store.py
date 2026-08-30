"""Qdrant backend stub: same VectorStore surface as FAISS.

Prod notes (not implemented in this POC):
- Collection name: `{name}_shard_{shard_key(asset_id, N)}` — visual and text stay separate.
- Shard key is hash(asset_id) % N so one asset never spans shards.
- Query router embeds once, fans out search, merges scores; it must not pull vectors back.
- Payload indexes: asset_id, modality, site_id, t_s.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vvs.config import Settings, load_settings
from vvs.retrieve.store import StoreHit, VectorPayload, shard_key


class QdrantStore:
    def __init__(self, video_id: str, name: str, *, settings: Settings | None = None) -> None:
        self.video_id = video_id
        self.name = name
        self.dim = 0
        cfg = settings or load_settings()
        url = (cfg.qdrant_url or "").strip()
        if not url:
            raise SystemExit(
                "VVS_VECTOR_BACKEND=qdrant requires VVS_QDRANT_URL "
                "(e.g. http://127.0.0.1:6333). FAISS is the local POC default."
            )
        self._url = url
        self._api_key = cfg.qdrant_api_key
        self._n_shards = max(1, int(cfg.qdrant_shards))
        self._client = None

    @property
    def collection(self) -> str:
        shard = shard_key(self.video_id, self._n_shards)
        return f"vvs_{self.name}_shard_{shard}"

    def create(self, dim: int) -> None:
        self.dim = int(dim)
        self._connect()

    def open(self, dim: int) -> None:
        self.dim = int(dim)
        self._connect()

    def add(self, vectors: np.ndarray, payloads: list[VectorPayload]) -> None:
        self._connect()
        raise SystemExit(
            f"Qdrant upsert is not wired in this POC (collection={self.collection}). "
            "Use VVS_VECTOR_BACKEND=faiss for local indexing."
        )

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[StoreHit]:
        self._connect()
        raise SystemExit(
            f"Qdrant search is not wired in this POC (collection={self.collection}). "
            "Use VVS_VECTOR_BACKEND=faiss for local query."
        )

    def ntotal(self) -> int:
        return 0

    def close(self) -> None:
        self._client = None

    def _connect(self) -> None:
        if self._client is not None:
            return
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise SystemExit(
                "qdrant-client is not installed. "
                "pip/uv install the optional extra: uv sync --extra qdrant"
            ) from exc
        self._client = QdrantClient(url=self._url, api_key=self._api_key or None)
