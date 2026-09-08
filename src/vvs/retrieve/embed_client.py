"""HTTP client for the visual-embed service (`services/visual_embed_server.py`).

`embed_clip_images`/`embed_clip_texts` in `embed.py` call into this module
when `Settings.visual_embed_url` is set, instead of loading the encoder
in-process -- so `build_index` and `search()` need no changes at all to run
against a remote encoder. Only the default Qwen3-VL-Embedding-2B path is
proxied; a `wemm` or `clip` index still embeds locally regardless of this
setting (see `embed.py`'s dispatch).

Kept import-light: `httpx` and `Path.read_bytes` only, no torch. This module
is meant to also be importable from a lightweight client process that never
wants torch on its own machine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class EmbedServiceError(RuntimeError):
    """The visual-embed service was unreachable, errored, or returned a bad shape."""


def remote_embed_images(
    paths: list[str] | list[Path],
    base_url: str,
    *,
    timeout_s: float = 60.0,
) -> np.ndarray:
    """POST a batch of local image files to `/embed/images`, return their vectors in order."""
    import httpx

    if not paths:
        return np.zeros((0, 0), dtype=np.float32)
    files = [
        ("files", (Path(p).name, Path(p).read_bytes(), "image/jpeg")) for p in paths
    ]
    try:
        resp = httpx.post(
            f"{base_url.rstrip('/')}/embed/images", files=files, timeout=timeout_s
        )
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPError as exc:
        raise EmbedServiceError(f"visual-embed service at {base_url} failed: {exc}") from exc
    vectors = body.get("vectors")
    if vectors is None or len(vectors) != len(paths):
        raise EmbedServiceError(
            f"visual-embed service returned {len(vectors or [])} vectors for "
            f"{len(paths)} images"
        )
    return np.asarray(vectors, dtype=np.float32)


def remote_embed_texts(
    texts: list[str],
    base_url: str,
    *,
    timeout_s: float = 60.0,
) -> np.ndarray:
    """POST query strings to `/embed/texts`, return their vectors in order."""
    import httpx

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    try:
        resp = httpx.post(
            f"{base_url.rstrip('/')}/embed/texts",
            json={"texts": texts},
            timeout=timeout_s,
        )
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPError as exc:
        raise EmbedServiceError(f"visual-embed service at {base_url} failed: {exc}") from exc
    vectors = body.get("vectors")
    if vectors is None or len(vectors) != len(texts):
        raise EmbedServiceError(
            f"visual-embed service returned {len(vectors or [])} vectors for "
            f"{len(texts)} texts"
        )
    return np.asarray(vectors, dtype=np.float32)
