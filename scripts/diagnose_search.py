#!/usr/bin/env python
"""Walk the search path one step at a time to find where a native crash happens.

A crash inside torch, onnxruntime, or the OpenSearch client's native TLS stack
kills the process with a signal (or hangs on a connection that never resolves),
so Python raises nothing useful. This script prints each step before
performing it and flushes immediately, so the last line on screen is the step
that died or hung.

    uv run python scripts/diagnose_search.py "your query"

Useful variations, to bisect device involvement:

    VVS_DEVICE=cpu           uv run python scripts/diagnose_search.py   # rule out Metal
    VVS_VISUAL_DTYPE=float32 uv run python scripts/diagnose_search.py   # rule out fp16

The dual-OpenMP-runtime segfault this script originally targeted (faiss-cpu
and torch each bundling their own libomp.dylib on macOS) no longer applies:
OpenSearch's k-NN search runs server-side, in the OpenSearch JVM process, not
in this Python process, so there is no second libomp to collide with torch's.
See git history for the retired faiss-specific diagnostic steps.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import time

faulthandler.enable()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


def step(label: str) -> float:
    print(f"[diag] --> {label}", flush=True)
    return time.perf_counter()


def ok(started: float) -> None:
    print(f"[diag]     ok ({time.perf_counter() - started:.2f}s)", flush=True)


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else "robotics"

    t = step("import vvs (sets MPS fallback, OpenMP and faulthandler flags)")
    import vvs

    print(f"[diag]     vvs {vvs.__version__}", flush=True)
    print(f"[diag]     KMP_DUPLICATE_LIB_OK={os.environ.get('KMP_DUPLICATE_LIB_OK')!r}", flush=True)
    ok(t)

    t = step("import torch")
    import torch

    print(f"[diag]     torch {torch.__version__} mps={torch.backends.mps.is_available()}", flush=True)
    ok(t)

    t = step("ping OpenSearch")
    from vvs.config import load_settings
    from vvs.retrieve.opensearch_store import ping

    settings = load_settings()
    reachable = ping(settings)
    print(f"[diag]     {settings.opensearch_url} reachable={reachable}", flush=True)
    if not reachable:
        print("[diag] OpenSearch unreachable; run `podman compose up -d`", flush=True)
        return 1
    ok(t)

    t = step("resolve device")
    from vvs.device import resolve_device, resolve_dtype

    device = resolve_device()
    print(f"[diag]     device={device} dtype={resolve_dtype(device)}", flush=True)
    ok(t)

    t = step("list indexed videos")
    from vvs.retrieve.index import list_indexed

    ids = list_indexed()
    print(f"[diag]     {len(ids)} indexed: {', '.join(ids[:6])}", flush=True)
    ok(t)
    if not ids:
        print("[diag] nothing indexed; analyze a video first", flush=True)
        return 1

    t = step("torch matmul on the chosen device")
    a = torch.randn(256, 256, device=device)
    print(f"[diag]     result finite: {bool(torch.isfinite(a @ a).all())}", flush=True)
    ok(t)

    t = step("embed the query in the visual space (loads the VL encoder)")
    from vvs.retrieve.embed import embed_clip_texts

    qv = embed_clip_texts([query])
    print(f"[diag]     shape={qv.shape}", flush=True)
    ok(t)

    t = step("embed the query with EmbeddingGemma")
    from vvs.retrieve.embed import embed_text_query

    print(f"[diag]     shape={embed_text_query(query).shape}", flush=True)
    ok(t)

    for vid in ids:
        t = step(f"load index {vid}")
        from vvs.retrieve.index import load_index

        bundle = load_index(vid)
        ok(t)

        t = step(f"OpenSearch k-NN search {vid}")
        from vvs.retrieve.search import search

        hits = search(vid, query, top_k=3, loaded=bundle)
        print(f"[diag]     {len(hits)} hits", flush=True)
        ok(t)

    t = step("full corpus search")
    from vvs.retrieve.search import search_corpus

    hits = search_corpus(query, top_k=5)
    ok(t)

    print(f"\n[diag] SURVIVED. {len(hits)} hits for {query!r}:", flush=True)
    for h in hits:
        print(f"   {getattr(h, 'video_id', '?')} t={h.t_s:.1f}s score={h.score:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
