#!/usr/bin/env python
"""A/B two visual encoders over one video's keyframes.

Usage
-----
Build the challenger index from an existing baseline index (same keyframes):

    uv run python scripts/compare_encoders.py build \
        --source-id fBEGOk3ZHw4 --encoder wemm --overwrite

Score baseline against challenger:

    uv run python scripts/compare_encoders.py score \
        --a fBEGOk3ZHw4 --b fBEGOk3ZHw4__wemm --report docs/encoder_ab.md

Metrics
-------
Two independent readings, because the labeled set here is small:

* **Labeled** — each query carries an expected timestamp window, so a hit is
  relevant when its frame falls inside the window. Yields recall@k, MRR and
  nDCG@k. Trustworthy only in proportion to how many judgments exist.
* **Unlabeled agreement** — overlap@k and Spearman rank correlation between the
  two rankings. Needs no ground truth and answers a different question: whether
  the encoders disagree enough to be worth judging at all. Near-identical
  rankings mean any metric difference is noise.

Latency is reported per query and per frame, since a 3B challenger on CPU can
win on quality and still lose on cost.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from vvs.config import load_settings
from vvs.models import IndexMeta
from vvs.paths import demo_questions_path, meta_path
from vvs.retrieve.embed import (
    embed_clip_texts,
    encoder_kind,
    normalize_encoder_kind,
    visual_dim,
)
from vvs.retrieve.store import open_store

DEFAULT_K = 10
# Queries that exercise the visual channel specifically. Timestamp windows are
# only meaningful for the video they were written against, so an unknown video
# falls back to agreement-only scoring.
FALLBACK_QUERIES = [
    "a person speaking to camera",
    "a slide with a bulleted list",
    "a line chart or graph",
    "a wide shot of a stage",
    "text on a dark background",
    "two people side by side",
    "a close-up of a face",
    "a logo or title card",
]


@dataclass
class Judgment:
    query: str
    windows: list[tuple[float, float]] = field(default_factory=list)

    def relevant(self, t_s: float) -> bool:
        return any(lo <= t_s <= hi for lo, hi in self.windows)

    @property
    def labeled(self) -> bool:
        return bool(self.windows)


@dataclass
class Ranking:
    query: str
    frame_keys: list[str]
    timestamps: list[float]
    scores: list[float]
    encode_ms: float
    search_ms: float


def parse_window(spec: str) -> tuple[float, float] | None:
    """Accept "0-55", "12.5-30", or "90" (a point, padded by 2s)."""
    text = str(spec).strip()
    if not text:
        return None
    if "-" in text:
        lo, _, hi = text.partition("-")
        try:
            return float(lo), float(hi)
        except ValueError:
            return None
    try:
        point = float(text)
    except ValueError:
        return None
    return max(0.0, point - 2.0), point + 2.0


def load_judgments(video_id: str, path: Path | None) -> list[Judgment]:
    if path is not None:
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        default = demo_questions_path(video_id)
        if not default.exists():
            print(f"[compare] no judgments at {default}; using unlabeled fallback queries")
            return [Judgment(query=q) for q in FALLBACK_QUERIES]
        raw = json.loads(default.read_text(encoding="utf-8"))

    rows = raw.get("questions", raw) if isinstance(raw, dict) else raw
    out: list[Judgment] = []
    for row in rows:
        if isinstance(row, str):
            out.append(Judgment(query=row))
            continue
        query = row.get("query") or row.get("question")
        if not query:
            continue
        specs = row.get("windows") or row.get("expect") or row.get("expected") or []
        if isinstance(specs, (str, int, float)):
            specs = [specs]
        windows = [w for w in (parse_window(s) for s in specs) if w]
        out.append(Judgment(query=query, windows=windows))
    return out


def frame_key(payload) -> str:
    """Identity of a keyframe that survives being re-embedded into a sibling index.

    `vector_id` is assigned per store and `asset_id` differs between the two
    indexes, so neither is stable. The JPEG filename is.
    """
    if payload.path:
        return Path(payload.path).name
    return f"t={round(float(payload.t_s), 3)}"


def rank(video_id: str, queries: list[str], top_k: int) -> tuple[list[Ranking], dict]:
    cfg = load_settings()
    meta = IndexMeta.model_validate_json(meta_path(video_id).read_text(encoding="utf-8"))
    dim = visual_dim(meta=meta)
    kind = encoder_kind(meta=meta)
    store = open_store(video_id, "visual", dim=dim, settings=cfg, create=False)
    info = {
        "video_id": video_id,
        "encoder": kind,
        "model": meta.models.clip,
        "dim": dim,
        "vectors": store.ntotal(),
    }
    print(f"[compare] {video_id}: {kind} / {meta.models.clip} / {dim}-d / {info['vectors']} vectors")
    rankings: list[Ranking] = []
    try:
        for query in queries:
            t0 = perf_counter()
            vec = embed_clip_texts([query], meta=meta)[0]
            encode_ms = (perf_counter() - t0) * 1000.0
            t0 = perf_counter()
            hits = store.search(vec, top_k)
            search_ms = (perf_counter() - t0) * 1000.0
            rankings.append(
                Ranking(
                    query=query,
                    frame_keys=[frame_key(h.payload) for h in hits],
                    timestamps=[float(h.payload.t_s) for h in hits],
                    scores=[float(h.score) for h in hits],
                    encode_ms=encode_ms,
                    search_ms=search_ms,
                )
            )
    finally:
        store.close()
    return rankings, info


def recall_at_k(judgment: Judgment, ranking: Ranking, k: int) -> float:
    return 1.0 if any(judgment.relevant(t) for t in ranking.timestamps[:k]) else 0.0


def reciprocal_rank(judgment: Judgment, ranking: Ranking, k: int) -> float:
    for i, t in enumerate(ranking.timestamps[:k], start=1):
        if judgment.relevant(t):
            return 1.0 / i
    return 0.0


def ndcg_at_k(judgment: Judgment, ranking: Ranking, k: int) -> float:
    gains = [1.0 if judgment.relevant(t) else 0.0 for t in ranking.timestamps[:k]]
    dcg = sum(g / np.log2(i + 1) for i, g in enumerate(gains, start=1))
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, int(sum(gains)) + 1))
    return float(dcg / ideal) if ideal else 0.0


def precision_at_k(judgment: Judgment, ranking: Ranking, k: int) -> float:
    window = ranking.timestamps[:k]
    if not window:
        return 0.0
    return sum(1.0 for t in window if judgment.relevant(t)) / len(window)


def overlap_at_k(a: Ranking, b: Ranking, k: int) -> float:
    top_a, top_b = set(a.frame_keys[:k]), set(b.frame_keys[:k])
    if not top_a or not top_b:
        return 0.0
    return len(top_a & top_b) / float(max(len(top_a), len(top_b)))


def spearman(a: Ranking, b: Ranking, k: int) -> float | None:
    """Rank correlation over frames both rankings placed in their top-k."""
    rank_a = {key: i for i, key in enumerate(a.frame_keys[:k])}
    rank_b = {key: i for i, key in enumerate(b.frame_keys[:k])}
    shared = sorted(set(rank_a) & set(rank_b))
    if len(shared) < 3:
        return None
    xs = np.array([rank_a[key] for key in shared], dtype=float)
    ys = np.array([rank_b[key] for key in shared], dtype=float)
    if xs.std() == 0 or ys.std() == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def cmd_build(args: argparse.Namespace) -> int:
    import os

    os.environ["VVS_VISUAL_ENCODER"] = normalize_encoder_kind(args.encoder)
    if args.model:
        os.environ["VVS_VISUAL_MODEL"] = args.model
    elif normalize_encoder_kind(args.encoder) == "wemm":
        os.environ["VVS_VISUAL_MODEL"] = "wemm-2b"
    if args.truncate_dim:
        os.environ["VVS_VISUAL_TRUNCATE_DIM"] = str(args.truncate_dim)

    from vvs.retrieve.reembed import reembed_visual

    dest = args.dest_id or f"{args.source_id}__{normalize_encoder_kind(args.encoder)}"
    result = reembed_visual(
        args.source_id,
        dest,
        settings=load_settings(),
        overwrite=args.overwrite,
        limit=args.limit,
    )
    print(
        f"[compare] built {result.video_id}: {result.frames} frames, {result.dim}-d, "
        f"{result.embed_s:.1f}s total, {result.per_frame_ms:.0f} ms/frame"
    )
    print(f"[compare] next: uv run python scripts/compare_encoders.py score --a {args.source_id} --b {dest}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    judgments = load_judgments(args.a, Path(args.judgments) if args.judgments else None)
    queries = [j.query for j in judgments]
    if not queries:
        raise SystemExit("no queries to score")

    rank_a, info_a = rank(args.a, queries, args.top_k)
    rank_b, info_b = rank(args.b, queries, args.top_k)

    labeled = [j for j in judgments if j.labeled]
    report = _render(
        judgments, rank_a, rank_b, info_a, info_b, top_k=args.top_k, ks=args.at
    )
    print()
    print(report)
    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"[compare] wrote {out}")
    if not labeled:
        print(
            "[compare] no timestamp judgments found: agreement numbers are valid, "
            "quality numbers are not."
        )
    return 0


def _render(
    judgments: list[Judgment],
    rank_a: list[Ranking],
    rank_b: list[Ranking],
    info_a: dict,
    info_b: dict,
    *,
    top_k: int,
    ks: list[int],
) -> str:
    labeled = [
        (j, a, b)
        for j, a, b in zip(judgments, rank_a, rank_b, strict=True)
        if j.labeled
    ]
    lines: list[str] = []
    lines.append("# Visual encoder A/B")
    lines.append("")
    lines.append(f"- **A** `{info_a['video_id']}` — {info_a['encoder']}, `{info_a['model']}`, {info_a['dim']}-d, {info_a['vectors']} vectors")
    lines.append(f"- **B** `{info_b['video_id']}` — {info_b['encoder']}, `{info_b['model']}`, {info_b['dim']}-d, {info_b['vectors']} vectors")
    lines.append(f"- {len(judgments)} queries, {len(labeled)} with timestamp judgments, top_k={top_k}")
    lines.append("")

    if info_a["vectors"] != info_b["vectors"]:
        lines.append(
            f"> Vector counts differ ({info_a['vectors']} vs {info_b['vectors']}). "
            "The two indexes do not cover the same keyframes, so quality differences "
            "are confounded by coverage."
        )
        lines.append("")

    lines.append("## Cost")
    lines.append("")
    lines.append("| Side | Query encode (ms, mean) | ANN search (ms, mean) |")
    lines.append("| --- | --- | --- |")
    for label, rankings in (("A", rank_a), ("B", rank_b)):
        lines.append(
            f"| {label} | {mean([r.encode_ms for r in rankings]):.0f} | "
            f"{mean([r.search_ms for r in rankings]):.1f} |"
        )
    lines.append("")

    lines.append("## Agreement (no labels needed)")
    lines.append("")
    lines.append("| k | overlap@k | Spearman (shared frames) |")
    lines.append("| --- | --- | --- |")
    for k in ks:
        overlaps = [overlap_at_k(a, b, k) for a, b in zip(rank_a, rank_b, strict=True)]
        rhos = [
            r
            for r in (spearman(a, b, k) for a, b in zip(rank_a, rank_b, strict=True))
            if r is not None
        ]
        rho = f"{mean(rhos):+.2f}" if rhos else "n/a"
        lines.append(f"| {k} | {mean(overlaps):.2f} | {rho} |")
    lines.append("")

    if labeled:
        lines.append("## Quality against timestamp judgments")
        lines.append("")
        lines.append("| Metric | A | B | Delta |")
        lines.append("| --- | --- | --- | --- |")
        for k in ks:
            for name, fn in (
                (f"recall@{k}", recall_at_k),
                (f"precision@{k}", precision_at_k),
                (f"nDCG@{k}", ndcg_at_k),
                (f"MRR@{k}", reciprocal_rank),
            ):
                va = mean([fn(j, a, k) for j, a, _ in labeled])
                vb = mean([fn(j, b, k) for j, _, b in labeled])
                lines.append(f"| {name} | {va:.3f} | {vb:.3f} | {vb - va:+.3f} |")
        lines.append("")

        lines.append("## Per-query first relevant rank")
        lines.append("")
        lines.append("| Query | A | B |")
        lines.append("| --- | --- | --- |")
        for j, a, b in labeled:
            lines.append(
                f"| {j.query} | {_first_hit(j, a)} | {_first_hit(j, b)} |"
            )
        lines.append("")

    lines.append("## Top-3 side by side")
    lines.append("")
    for j, a, b in zip(judgments, rank_a, rank_b, strict=True):
        lines.append(f"**{j.query}**")
        lines.append("")
        lines.append("| Rank | A frame @ t | A score | B frame @ t | B score |")
        lines.append("| --- | --- | --- | --- | --- |")
        for i in range(3):
            ak = f"{a.frame_keys[i]} @ {a.timestamps[i]:.1f}s" if i < len(a.frame_keys) else "-"
            av = f"{a.scores[i]:.3f}" if i < len(a.scores) else "-"
            bk = f"{b.frame_keys[i]} @ {b.timestamps[i]:.1f}s" if i < len(b.frame_keys) else "-"
            bv = f"{b.scores[i]:.3f}" if i < len(b.scores) else "-"
            lines.append(f"| {i + 1} | {ak} | {av} | {bk} | {bv} |")
        lines.append("")
    return "\n".join(lines)


def _first_hit(judgment: Judgment, ranking: Ranking) -> str:
    for i, t in enumerate(ranking.timestamps, start=1):
        if judgment.relevant(t):
            return str(i)
    return "miss"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Re-embed an existing index with another encoder")
    b.add_argument("--source-id", required=True)
    b.add_argument("--dest-id", default=None)
    b.add_argument("--encoder", default="wemm", help="wemm | qwen_vl | clip")
    b.add_argument("--model", default=None, help="Explicit HF id or alias")
    b.add_argument("--truncate-dim", type=int, default=None, help="WeMM Matryoshka width")
    b.add_argument("--limit", type=int, default=None, help="Only the first N frames")
    b.add_argument("--overwrite", action="store_true")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("score", help="Score two indexes against each other")
    s.add_argument("--a", required=True, help="Baseline video id")
    s.add_argument("--b", required=True, help="Challenger video id")
    s.add_argument("--judgments", default=None, help="JSON judgment file")
    s.add_argument("--top-k", type=int, default=DEFAULT_K)
    s.add_argument("--at", type=int, nargs="+", default=[1, 3, 5])
    s.add_argument("--report", default=None, help="Write the markdown report here")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
