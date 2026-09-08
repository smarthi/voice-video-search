"""FastAPI service wrapping the shared Qwen3-VL-Embedding-2B visual encoder.

Both the ingest worker and the retrieval side need to embed into the *same*
vector space the index was built in, so this is one resident copy of the
model, called over HTTP, rather than two copies that could quietly drift.
Running it as its own process also means it never shares a card with the
captioner: `vvs.device.should_evict()` and the whole eviction dance in
`retrieve/index.py` exist only because one process used to hold both 2B
models at once. A process that only ever loads this one never needs it.

Scope: serves the default Qwen3-VL-Embedding-2B encoder only, at its native
width. The `wemm` and `clip` alternate encoders (`VVS_VISUAL_ENCODER=wemm|clip`)
and `VVS_VISUAL_TRUNCATE_DIM` stay local-only; `embed.py`'s dispatch only
proxies here when a call would have resolved to plain Qwen3-VL-Embedding-2B
locally too (see `_remote_qwen_ok`).

Run directly:
    uv run --extra services uvicorn vvs.services.visual_embed_server:app \
        --host 0.0.0.0 --port 8001

Or via Docker: see docker/visual-embed.Dockerfile. `GET /health` reports
readiness; `GET /info` reports the model, device, and dimension actually
loaded, so a client can sanity-check compatibility instead of trusting it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

app = FastAPI(title="vvs-visual-embed", description=__doc__)


class TextEmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    dim: int
    model: str


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness only -- does not load the model. See /info for readiness."""
    return {"status": "ok"}


@app.get("/info")
def info() -> dict[str, Any]:
    """Loads the model if it has not been already, so a slow first call
    happens here rather than on the first real request."""
    from vvs.device import resolve_device, resolve_dtype
    from vvs.retrieve.embed import QWEN_VL_2B, visual_dim

    device = resolve_device()
    return {
        "model": QWEN_VL_2B,
        "dim": visual_dim(),
        "device": device,
        "dtype": resolve_dtype(device),
    }


@app.post("/embed/images", response_model=EmbedResponse)
async def embed_images(files: list[UploadFile] = File(...)) -> EmbedResponse:
    """Encode a batch of images. `vectors[i]` corresponds to `files[i]`."""
    if not files:
        raise HTTPException(400, "no files given")
    from vvs.retrieve.embed import QWEN_VL_2B, _embed_qwen_images, default_image_batch

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            paths: list[Path] = []
            for i, upload in enumerate(files):
                data = await upload.read()
                dest = tmp_dir / f"{i:04d}_{upload.filename or 'frame.jpg'}"
                dest.write_bytes(data)
                paths.append(dest)
            vecs = _embed_qwen_images(paths, batch_size=default_image_batch())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"embed failed: {exc}") from exc
    return EmbedResponse(
        vectors=vecs.tolist(), dim=int(vecs.shape[1]) if vecs.size else 0, model=QWEN_VL_2B
    )


@app.post("/embed/texts", response_model=EmbedResponse)
def embed_texts(req: TextEmbedRequest) -> EmbedResponse:
    """Encode a batch of query strings into the same space as /embed/images."""
    if not req.texts:
        raise HTTPException(400, "no texts given")
    from vvs.retrieve.embed import QWEN_VL_2B, _embed_qwen_texts

    try:
        vecs = _embed_qwen_texts(req.texts)
    except Exception as exc:
        raise HTTPException(500, f"embed failed: {exc}") from exc
    return EmbedResponse(
        vectors=vecs.tolist(), dim=int(vecs.shape[1]) if vecs.size else 0, model=QWEN_VL_2B
    )
