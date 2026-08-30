#!/usr/bin/env python3
"""Typed query against an indexed video. Source/id via flags or env."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vvs.cli import main as cli_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Query an indexed video")
    parser.add_argument("text", nargs="+")
    parser.add_argument("--source", default=None)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    argv = ["query"]
    if args.source:
        argv += ["--source", args.source]
    if args.video_id:
        argv += ["--video-id", args.video_id]
    argv += ["--top-k", str(args.top_k)]
    if args.json:
        argv.append("--json")
    argv += args.text
    cli_main(argv)


if __name__ == "__main__":
    main()
