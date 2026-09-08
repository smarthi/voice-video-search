"""Standalone FastAPI services wrapping the two 5 GB models.

`visual_embed_server.py` (Qwen3-VL-Embedding-2B) and `caption_server.py`
(Qwen3-VL-2B-Instruct) are the two models that both index-time and query-time
code need identical weights for, and the two models responsible for every
GPU-contention/eviction bug this repo has carried (see `vvs.device.should_evict`
and the `release_captioner`/`release_visual_encoder` calls in `retrieve/index.py`).
Running each as its own process removes that whole problem: a process that
only ever holds one model never needs to evict it for the other.

Neither module here is imported by the rest of `vvs` at import time -- they
pull in `fastapi`, which is an optional extra (`pip install vvs[services]`),
not a core dependency of the CLI or the UI. Run one directly:

    uv run --extra services uvicorn vvs.services.visual_embed_server:app --port 8001
    uv run --extra services uvicorn vvs.services.caption_server:app --port 8002

or build the Dockerfiles under `docker/`. Point the rest of the app at them
with `VVS_VISUAL_EMBED_URL` / `VVS_CAPTION_SERVICE_URL`; unset (the default),
everything embeds and captions in-process exactly as before.
"""
