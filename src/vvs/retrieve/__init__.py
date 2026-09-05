"""Embedding, index build/load, and dual-channel search."""

from vvs.retrieve.embed import warm_models
from vvs.retrieve.index import (
    build_index,
    embed_counts,
    ensure_index,
    format_embed_compare,
    format_embed_counts,
    format_timings,
    index_exists,
    load_index,
    load_timings,
    should_run_asr,
)
from vvs.retrieve.search import (
    format_hits,
    fmt_clock,
    fmt_confidence,
    fmt_time,
    merge_corpus_hits,
    search,
    search_corpus,
)

__all__ = [
    "build_index",
    "embed_counts",
    "ensure_index",
    "format_embed_compare",
    "format_embed_counts",
    "format_hits",
    "format_timings",
    "fmt_clock",
    "fmt_confidence",
    "fmt_time",
    "index_exists",
    "load_index",
    "load_timings",
    "merge_corpus_hits",
    "search",
    "search_corpus",
    "should_run_asr",
    "warm_models",
]
