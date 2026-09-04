"""One dedicated thread for every torch / faiss / whisper call in the Gradio UI.

FAISS's OpenMP search (`swigfaiss.search` -> `__kmpc_fork_call`) segfaults on
macOS the moment it is entered from a *second* distinct OS thread in the
process's lifetime -- reliably, even with `omp_set_num_threads(1)`, and even
after the same call has already succeeded once on another thread. This is not
about main-vs-background: warming everything on the main thread first and
then searching from a background thread still crashes. What survives is
calling in from as many different Python threads as you like, as long as the
actual torch/faiss work always lands on the *same* one.

`vvs/ui/app.py` is a long-lived server handling requests on Gradio/anyio
worker threads, a different one per request, so the app cannot rely on "the
calling thread" being consistent -- it has to force consistency itself. This
module is that: a single-worker executor that every heavy call is funneled
through, created once and warmed on itself (see `vvs.retrieve.embed.warm_models`
called via `run` in `launch()`), so the same thread that loads the models is
the same thread that ever touches faiss.

The one-shot CLI (`vvs/cli.py`) does not need this: a single process running
on its own main thread never introduces a second thread, so it never hits the
bug in the first place.
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
