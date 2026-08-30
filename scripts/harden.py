#!/usr/bin/env python3
"""Demo-day checklist: confirm index, write fallback tape, pre-warm models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vvs.config import load_settings
from vvs.paths import demo_questions_path, fallback_html_path, index_path, timings_path
from vvs.retrieve.embed import warm_models
from vvs.retrieve.index import ensure_index, index_exists, load_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None)
    parser.add_argument("--video-id", default=None)
    args = parser.parse_args()
    settings = load_settings(source=args.source, video_id=args.video_id)
    vid = ensure_index(settings.source, settings.video_id)
    assert index_exists(vid), vid
    meta = load_index(vid)["meta"]
    print(f"index  {index_path(vid)}")
    print(f"title  {meta.title}")
    print(f"frames {len(meta.frames)}  asr {len(meta.transcript)}  ocr {len(meta.ocr)}  slide_heavy={meta.slide_heavy}")
    print(f"timings {timings_path(vid)}")
    print(f"demo questions {demo_questions_path(vid)} exists={demo_questions_path(vid).exists()}")
    print(f"fallback {fallback_html_path(vid)} exists={fallback_html_path(vid).exists()}")
    print("warming models…")
    warm_models()
    print("ready. launch with: uv run vvs app --video-id", vid)


if __name__ == "__main__":
    main()
