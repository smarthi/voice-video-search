"""VectorStore protocol: ANN search without loading all embeddings into app RAM.

Backend: OpenSearch, one shared k-NN index per collection (`vvs_visual`,
`vvs_text`) across the whole archive -- see `retrieve/opensearch_store.py` for
why, and for the "one index per video" model this replaced.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel

from vvs.config import Settings


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

    def all_payloads(self, modality: str | None = None) -> list[VectorPayload]: ...

    def close(self) -> None: ...


def open_store(
    video_id: str,
    name: str,
    *,
    dim: int,
    settings: Settings | None = None,
    create: bool = False,
    generation: str | None = None,
) -> VectorStore:
    """Open the visual or text collection, scoped to one video within the shared index."""
    from vvs.retrieve.opensearch_store import OpenSearchStore

    store = OpenSearchStore(video_id, name, settings=settings, generation=generation)
    if create:
        store.create(dim)
    else:
        store.open(dim)
    return store


def visual_store_exists(video_id: str, *, generation: str | None = None) -> bool:
    from vvs.retrieve.opensearch_store import OpenSearchStore

    return OpenSearchStore(video_id, "visual", generation=generation).ntotal() > 0
