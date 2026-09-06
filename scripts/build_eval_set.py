#!/usr/bin/env python3
"""Bootstrap DRAFT eval-set candidates from indexed video metadata.

For a sample of each indexed video's scene captions, asks the same
text-only Qwen3-VL-Instruct bundle media/caption.py uses for query
expansion to paraphrase the caption into a short, natural search query --
explicitly told not to reuse the caption's own wording. Using the caption
verbatim as the query would be circular (it just tests whether an
embedding matches itself); a paraphrase tests actual semantic retrieval,
which is the thing that broke in the "robotics" / "woman with bindi" bugs.

The caption's own (video_id, t_s) becomes the candidate's expected answer.

This script only PROPOSES candidates -- it does not verify anything. A
paraphrase can be too close to verbatim, too vague to be answerable, wrong,
or trivially easy. Always read scripts/build_eval_set.py's output file and
hand-pick good cases into eval/eval_set.jsonl; do not merge the draft
directly.

Usage:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/build_eval_set.py \
        --samples-per-video 2 --out eval/eval_set.draft.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

_PARAPHRASE_PROMPT = (
    'A video search caption reads: "{caption}"\n'
    "Write ONE short, natural search query (5-8 words) a person might type "
    "to find this exact video clip. Use different wording than the caption "
    "where you can -- a synonym or a shorter description, not the same "
    "phrases. Reply with only the query text, nothing else."
)


def paraphrase(caption: str) -> str | None:
    from vvs.media.caption import _qwen_rewrite

    try:
        decoded = _qwen_rewrite(_PARAPHRASE_PROMPT.format(caption=caption))
    except Exception as exc:
        print(f"[build_eval_set] paraphrase failed: {exc}", file=sys.stderr)
        return None
    lines = [ln.strip().strip("\"'").strip() for ln in (decoded or "").splitlines() if ln.strip()]
    return lines[0] if lines else None


def build(samples_per_video: int, seed: int, min_caption_words: int) -> list[dict]:
    from vvs.retrieve.index import list_indexed, load_index

    rng = random.Random(seed)
    cases: list[dict] = []
    for vid in list_indexed():
        try:
            bundle = load_index(vid)
        except Exception as exc:
            print(f"[build_eval_set] skip {vid}: {exc}", file=sys.stderr)
            continue
        meta = bundle["meta"]
        scenes = [s for s in meta.scenes if len((s.caption or "").split()) >= min_caption_words]
        if not scenes:
            continue
        sample = rng.sample(scenes, k=min(samples_per_video, len(scenes)))
        for scene in sample:
            query = paraphrase(scene.caption)
            if not query:
                continue
            case = {
                "id": f"{vid}-scene-{scene.scene_id}",
                "query": query,
                "video_ids": [vid],
                "channel": "scene",
                "genre": "auto",
                "note": f"auto-generated (unreviewed) from caption: {scene.caption!r}",
                "review": "pending",
            }
            cases.append(case)
            print(f"[{vid}] {scene.caption[:70]!r} -> {query!r}", flush=True)
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples-per-video", type=int, default=2)
    parser.add_argument("--min-caption-words", type=int, default=6, help="skip terse/uninformative captions")
    parser.add_argument("--out", default="eval/eval_set.draft.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cases = build(args.samples_per_video, args.seed, args.min_caption_words)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")

    print(f"\n[build_eval_set] wrote {len(cases)} DRAFT cases to {out_path}")
    print("[build_eval_set] review before merging into eval/eval_set.jsonl -- these are unverified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
