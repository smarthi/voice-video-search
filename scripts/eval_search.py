#!/usr/bin/env python3
"""Automated relevance eval: recall@k / MRR against a labeled eval set.

Every fix in retrieve/search.py so far has been verified by hand, one live
query at a time (see git log). That does not scale past a handful of videos
and it is exactly how the "robotics" / "woman with bindi" ranking bugs went
unnoticed for as long as they did -- nothing re-ran the earlier, already-fixed
queries automatically when later changes landed. This script is that
automatic re-run.

Each line of the eval file is one JSON case:
    {
      "id":            unique label, free text
      "query":         the search string
      "video_ids":     [video_id, ...] -- any ONE counts as a hit
      "must_not_top1": [video_id, ...] -- optional; these must never be rank 1
      "channel":       "scene" | "transcript" | "ocr" | "motion" | "visual" | "mixed"
      "genre":         free text, for the by-category breakdown
      "note":          free text, why this case exists
    }

Usage:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        uv run python scripts/eval_search.py [--eval-file eval/eval_set.jsonl] [--top-k 5]

Exit code is nonzero if any must_not_top1 is violated, or (with --fail-under)
if recall@k drops below the given fraction -- wire this into CI once the eval
set is large enough to trust for that.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


def load_cases(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[eval] no eval file at {path}", file=sys.stderr)
        return []
    cases = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"[eval] {path}:{lineno}: skipping malformed line ({exc})", file=sys.stderr)
    return cases


def evaluate(cases: list[dict], top_k: int) -> list[dict]:
    from vvs.retrieve.search import search_corpus

    results = []
    for case in cases:
        query = case["query"]
        hits = search_corpus(query, top_k=top_k)
        video_ids = set(case.get("video_ids", []))
        rank = None
        for i, hit in enumerate(hits, start=1):
            if hit.video_id in video_ids:
                rank = i
                break
        top1 = hits[0].video_id if hits else None
        must_not = set(case.get("must_not_top1", []))
        results.append(
            {
                "id": case.get("id", query),
                "query": query,
                "rank": rank,
                "hit_at_k": rank is not None,
                "top1": top1,
                "must_not_top1_violated": bool(must_not) and top1 in must_not,
                "channel": case.get("channel", "unspecified"),
                "genre": case.get("genre", "unspecified"),
                "note": case.get("note", ""),
            }
        )
        status = "OK " if results[-1]["hit_at_k"] else "MISS"
        flag = " !!! must_not_top1 violated" if results[-1]["must_not_top1_violated"] else ""
        print(f"[eval] {status} rank={rank!s:<4} top1={top1!s:<40} {query!r}{flag}", flush=True)
    return results


def summarize(results: list[dict], top_k: int) -> tuple[float, bool]:
    n = len(results)
    if n == 0:
        print("[eval] no cases evaluated")
        return 0.0, False
    hits = sum(1 for r in results if r["hit_at_k"])
    recall = hits / n
    mrr = sum((1.0 / r["rank"]) if r["rank"] else 0.0 for r in results) / n
    violations = [r for r in results if r["must_not_top1_violated"]]

    print(f"\n=== Eval summary ({n} cases, top_k={top_k}) ===")
    print(f"recall@{top_k}: {hits}/{n} ({100 * recall:.1f}%)")
    print(f"MRR:       {mrr:.3f}")

    by_channel: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_channel[r["channel"]].append(r)
    print("\nBy channel:")
    for channel, rs in sorted(by_channel.items()):
        h = sum(1 for r in rs if r["hit_at_k"])
        print(f"  {channel:12s} {h}/{len(rs)}")

    by_genre: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_genre[r["genre"]].append(r)
    if len(by_genre) > 1:
        print("\nBy genre:")
        for genre, rs in sorted(by_genre.items()):
            h = sum(1 for r in rs if r["hit_at_k"])
            print(f"  {genre:12s} {h}/{len(rs)}")

    failures = [r for r in results if not r["hit_at_k"] or r["must_not_top1_violated"]]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for r in failures:
            reason = "miss" if not r["hit_at_k"] else "must_not_top1 violated"
            print(f"  [{r['id']}] {reason}: query={r['query']!r} rank={r['rank']} top1={r['top1']}")
            if r["note"]:
                print(f"      note: {r['note']}")
    else:
        print("\nNo failures.")

    return recall, bool(violations)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-file", default="eval/eval_set.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Exit nonzero if recall@k drops below this fraction (e.g. 0.9)",
    )
    args = parser.parse_args()

    cases = load_cases(Path(args.eval_file))
    if not cases:
        return 1

    results = evaluate(cases, args.top_k)
    recall, any_violation = summarize(results, args.top_k)

    if any_violation:
        return 1
    if args.fail_under is not None and recall < args.fail_under:
        print(f"\n[eval] recall {recall:.3f} < --fail-under {args.fail_under:.3f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
