"""One dedicated thread for every torch / whisper call in the Gradio UI.

Originally written because FAISS's OpenMP search segfaulted on macOS the
moment it was entered from a *second* distinct OS thread in the process's
lifetime -- reliably, even with the thread count pinned to 1, and even after
the same call had already succeeded once on another thread. FAISS is gone now
(search runs server-side in OpenSearch; see retrieve/opensearch_store.py), so
that specific crash is retired.

This module is still needed for a related, independent crash: transformers'
parallel weight-materialization path (`core_model_loading.py`, used the first
time a model like Qwen3-VL-Embedding loads) segfaults when that first load
happens on a background thread rather than the process's main thread --
confirmed by direct repro, same as the FAISS case, different library. Since
`vvs/ui/app.py` is a long-lived server handling requests on a different
Gradio/anyio worker thread per request, "run the first load on main" is not
available to it; the fix is the same shape as before: force every heavy call,
including the model warm-up itself, through one dedicated thread, so it is
always the same thread that touches torch, no matter which request triggered
the first load.

The one-shot CLI (`vvs/cli.py`) does not need this: a single process running
on its own main thread never introduces a second thread, so it never hits
either bug in the first place.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vvs-compute")


def run(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Run `fn` on the dedicated compute thread and block for its result.

    Exceptions raised in `fn` propagate to the caller unchanged.
    """
    return _EXECUTOR.submit(fn, *args, **kwargs).result()
