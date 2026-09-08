"""FastAPI service wrapping the shared Qwen3-VL-2B-Instruct captioner.

Both index-time scene captioning and query-time LLM query expansion use
this exact model. One resident copy, called over HTTP, replaces the
`release_captioner()`/`release_visual_encoder()` eviction dance this repo
carries today (see `device.should_evict`) -- this process never shares a
card with the visual encoder, so it never needs to give it up.

Scope: Qwen3-VL-2B-Instruct only. The BLIP fallback path (`caption.py`'s
`_blip_caption`) stays local-only; there is no text-only BLIP path to expand
a query with anyway.

Run directly:
    uv run --extra services uvicorn vvs.services.caption_server:app \
        --host 0.0.0.0 --port 8002

Or via Docker: see docker/caption.Dockerfile.
"""

from __future__ import annotations

import io
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

app = FastAPI(title="vvs-caption", description=__doc__)


class CaptionResponse(BaseModel):
    caption: str
    objects: list[str]
    model: str


class ExpandRequest(BaseModel):
    query: str
    max_variants: int | None = None


class ExpandResponse(BaseModel):
    variants: list[str]


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness only -- does not load the model. See /info for readiness."""
    return {"status": "ok"}


@app.get("/info")
def info() -> dict[str, Any]:
    """Loads the model if it has not been already, so a slow first call
    happens here rather than on the first real request."""
    from vvs.device import resolve_device, resolve_dtype
    from vvs.media.caption import QWEN_CAPTION_ID

    device = resolve_device()
    return {"model": QWEN_CAPTION_ID, "device": device, "dtype": resolve_dtype(device)}


@app.post("/caption", response_model=CaptionResponse)
async def caption(file: UploadFile = File(...)) -> CaptionResponse:
    """Caption one still. Send it already resized -- see caption_client.py."""
    from PIL import Image

    from vvs.media.caption import _qwen_caption

    data = await file.read()
    try:
        with Image.open(io.BytesIO(data)) as im:
            image = im.convert("RGB")
        cap, objects, model = _qwen_caption(image)
    except Exception as exc:
        raise HTTPException(500, f"caption failed: {exc}") from exc
    return CaptionResponse(caption=cap, objects=objects, model=model)


@app.post("/expand_query", response_model=ExpandResponse)
def expand_query(req: ExpandRequest) -> ExpandResponse:
    """Text-only rewrite: alternate phrasings of the query's visual subject."""
    from vvs.media.caption import _EXPAND_PROMPT, _parse_expansion_lines, _qwen_rewrite

    query = (req.query or "").strip()
    if not query:
        raise HTTPException(400, "empty query")
    n = req.max_variants if req.max_variants is not None else 2
    if n <= 0:
        return ExpandResponse(variants=[])
    try:
        decoded = _qwen_rewrite(_EXPAND_PROMPT.format(query=query, n=n))
    except Exception as exc:
        raise HTTPException(500, f"expansion failed: {exc}") from exc
    return ExpandResponse(variants=_parse_expansion_lines(decoded, query, n))
