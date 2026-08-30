#!/usr/bin/env python3
"""Build a retrieval index. Source is required via --source or VVS_SOURCE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vvs.cli import main as cli_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a video retrieval index")
    parser.add_argument("--source", default=None, help="YouTube URL or local path (or VVS_SOURCE)")
    parser.add_argument("--video-id", default=None, help="Override video id (or VVS_VIDEO_ID)")
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--scene-caption", action="store_true")
    parser.add_argument("--shot-detect", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    argv = ["index"]
    if args.source:
        argv += ["--source", args.source]
    if args.video_id:
        argv += ["--video-id", args.video_id]
    if args.no_ocr:
        argv.append("--no-ocr")
    if args.scene_caption:
        argv.append("--scene-caption")
    if args.shot_detect:
        argv.append("--shot-detect")
    if args.rebuild:
        argv.append("--rebuild")
    cli_main(argv)


if __name__ == "__main__":
    main()
