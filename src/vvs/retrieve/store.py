"""VectorStore protocol: ANN search without loading all embeddings into app RAM.

POC: FaissStore (on-disk, mmap). Prod: QdrantStore (sharded collections).
Visual embeddings and BGE text live in separate indexes — they are different spaces.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel

from vvs.config import Settings, load_settings
from vvs.paths import video_dir


class VectorPayload(BaseModel):
    vector_id: int
    asset_id: str
    t_s: float = 0.0
    modality: str = "video_frame"
    path: str = ""
    text: str | None = None
    frame_index: int | None = None
    shot_id: int | None = None
    motion_energy: float = 0.0
    flash_energy: float = 0.0
    capture_ts: str | None = None
    site_id: str | None = None


class StoreHit(BaseModel):
    vector_id: int
    score: float
    payload: VectorPayload


@runtime_checkable
class VectorStore(Protocol):
    dim: int

    def create(self, dim: int) -> None: ...

    def add(self, vectors: np.ndarray, payloads: list[VectorPayload]) -> None: ...

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[StoreHit]: ...

    def ntotal(self) -> int: ...

    def close(self) -> None: ...


def shard_key(asset_id: str, n_shards: int) -> int:
    """Stable prod routing: hash(asset_id) % N. Same asset never splits across shards."""
    import hashlib

    if n_shards <= 0:
        raise ValueError("n_shards must be positive")
    digest = hashlib.sha256(asset_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % n_shards


def open_store(
    video_id: str,
    name: str,
    *,
    dim: int,
    settings: Settings | None = None,
    create: bool = False,
) -> VectorStore:
    """Open the visual or text collection for one asset (POC) / shard (prod)."""
    cfg = settings or load_settings()
    backend = (cfg.vector_backend or "faiss").strip().lower()
    if backend == "qdrant":
        from vvs.retrieve.qdrant_store import QdrantStore

        store = QdrantStore(video_id, name, settings=cfg)
        if create:
            store.create(dim)
        else:
            store.open(dim)
        return store
    if backend not in {"faiss", ""}:
        raise SystemExit(f"Unknown VVS_VECTOR_BACKEND={backend!r}. Use faiss or qdrant.")
    from vvs.retrieve.faiss_store import FaissStore

    store = FaissStore(video_id, name, index_type=cfg.faiss_index_type)
    if create:
        store.create(dim)
    else:
        store.open(dim)
    return store


def store_dir(video_id: str) -> Any:
    return video_dir(video_id)


def visual_store_exists(video_id: str) -> bool:
    root = video_dir(video_id)
    return (root / "visual.faiss").exists() and (root / "visual.sqlite").exists()
