#!/usr/bin/env python3
"""Tune retrieval: run rehearsed queries and check top-3 timestamps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vvs.config import load_settings
from vvs.models import DemoQuestion
from vvs.paths import demo_questions_path
from vvs.retrieve.index import ensure_index, load_index
from vvs.retrieve.search import fmt_time, search


DEFAULT_CHECKS = [
    DemoQuestion(
        query="what is hybrid AI",
        why="Opening definition of hybrid AI",
        expect="0-55",
    ),
    DemoQuestion(
        query="show me the Motorola products and Kira",
        why="Product lineup and Kira ambient AI",
        expect="110-140",
    ),
    DemoQuestion(
        query="personalized knowledge base on phone tablet and PC",
        why="Cross-device UI and knowledge base",
        expect="130-180",
    ),
    DemoQuestion(
        query="robotics and cybersecurity research",
        why="Emerging tech beyond AI PCs",
        expect="175-200",
    ),
    DemoQuestion(
        query="agent-native devices voice and gesture interfaces",
        why="Horizon / new interaction modalities",
        expect="196-260",
    ),
]


def _window(spec: str) -> tuple[float, float]:
    start, end = spec.split("-", 1)
    return float(start), float(end)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--write-demo", action="store_true")
    args = parser.parse_args()
    settings = load_settings(source=args.source, video_id=args.video_id)
    vid = ensure_index(settings.source, settings.video_id)
    bundle = load_index(vid)
    if "frame_vecs" in bundle or "transcript_vecs" in bundle:
        raise SystemExit("load_index still returns dense embedding matrices")
    nvis = bundle["visual"].ntotal()
    print(f"[store] backend=OpenSearch visual_ntotal={nvis} text={bundle['text'] is not None}")
    ok = 0
    for q in DEFAULT_CHECKS:
        hits = search(vid, q.query, top_k=3, loaded=bundle)
        lo, hi = _window(q.expect)
        in_window = any(lo <= h.t_s <= hi for h in hits)
        ok += int(in_window)
        mark = "PASS" if in_window else "MISS"
        print(f"\n[{mark}] {q.query}")
        for h in hits:
            print(f"  #{h.rank} {fmt_time(h.t_s)} score={h.score:.4f} {','.join(h.channels)}")
    print(f"\n{ok}/{len(DEFAULT_CHECKS)} queries had a top-3 hit in the expected window")

    mute_meta = bundle["meta"].model_copy(deep=True)
    mute_meta.channels.asr = False
    mute_meta.channels.transcript = False
    mute_bundle = {**bundle, "meta": mute_meta}
    mute_hits = search(vid, DEFAULT_CHECKS[0].query, top_k=3, loaded=mute_bundle)
    print(f"\n[mute] visual-only search returned {len(mute_hits)} hits (ASR boosts off)")
    if not mute_hits:
        raise SystemExit("mute/visual-only search returned no hits")
    if args.write_demo:
        dest = demo_questions_path(vid)
        dest.write_text(json.dumps([q.model_dump() for q in DEFAULT_CHECKS], indent=2), encoding="utf-8")
        print(f"wrote {dest}")


if __name__ == "__main__":
    main()
