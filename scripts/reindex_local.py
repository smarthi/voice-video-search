#!/usr/bin/env python
"""Re-index the videos already sitting in the data directory, keeping their IDs.

Needed after the store is replaced or emptied -- a fresh OpenSearch volume, or
a switch of visual encoder -- when the source videos are still on disk and only
the vectors are gone.

`vvs index --sources-file` cannot do this. Its IDs come from
`local_video_id()`, which hashes the resolved path so that two unrelated
`demo.mp4` files cannot collide. That is right for new ingests and wrong here:
a directory already named `fBEGOk3ZHw4` would be re-indexed as
`video-a1b2c3d4e5f6`, orphaning the keyframes, captions and preview beside it.
This script takes each ID from its own directory name instead.

Everything runs in one process so the encoder and captioner are loaded once
rather than per video, and sequentially because there is one GPU to share.

    uv run python scripts/reindex_local.py            # skip anything indexed
    uv run python scripts/reindex_local.py --dry-run  # just show the plan
    uv run python scripts/reindex_local.py --rebuild  # redo even if present

Already-indexed videos are skipped, `index_exists()` being a question about
the store rather than the disk, so an interrupted run resumes where it stopped.

OCR, scene captions and TransNetV2 shot detection are all **on** here, and
each has a `--no-` opt-out. That deliberately departs from `Settings`, where
`scene_caption` and `shot_detect` both default to False: those defaults suit
`vvs index`, whose caller is choosing what to spend on a single new video,
whereas the job of this script is to put the archive back the way it was, and a
channel omitted by default is a channel silently missing from the store. The
enabled stages are printed with the plan for the same reason -- the first
version of this script left `shot_detect` unset, which quietly meant no
TransNetV2 and a 2 fps index, visible only as `shots=0` in `vvs list`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vvs.paths import DATA_ROOT
from vvs.retrieve.index import build_index, index_exists


def discover(only: list[str] | None) -> list[tuple[str, Path]]:
    """Every data directory holding a source video, as (video_id, path) pairs."""
    found: list[tuple[str, Path]] = []
    for child in sorted(DATA_ROOT.iterdir()) if DATA_ROOT.exists() else []:
        if not child.is_dir():
            continue
        video = child / "video.mp4"
        if not video.exists():
            continue
        if only and child.name not in only:
            continue
        found.append((child.name, video))
    return found


def human(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true", help="re-index even if the store has it")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    ap.add_argument("--only", default=None, help="comma-separated video ids to limit to")
    ap.add_argument("--no-scene-caption", action="store_true", help="skip captioning (much faster)")
    ap.add_argument("--no-shot-detect", action="store_true", help="skip TransNetV2 shot grouping")
    ap.add_argument("--no-ocr", action="store_true", help="skip RapidOCR")
    args = ap.parse_args()

    only = [s.strip() for s in args.only.split(",") if s.strip()] if args.only else None
    videos = discover(only)
    if not videos:
        print(f"No video.mp4 found under {DATA_ROOT}.")
        return 1

    todo = [(vid, p) for vid, p in videos if args.rebuild or not index_exists(vid)]
    queued = {vid for vid, _ in todo}
    done_already = [vid for vid, _ in videos if vid not in queued]
    stages = [
        ("ocr", not args.no_ocr),
        ("scene captions", not args.no_scene_caption),
        ("TransNetV2 shots", not args.no_shot_detect),
    ]
    on = [name for name, want in stages if want]
    off = [name for name, want in stages if not want]
    print(f"stages on: {', '.join(on) if on else 'none'}"
          + (f" | off: {', '.join(off)}" if off else ""))
    print(f"{len(videos)} videos on disk, {len(todo)} to index, "
          f"{len(done_already)} already in the store")
    for vid, path in todo:
        print(f"  {vid:<16} {path.stat().st_size / 1e6:>6.1f} MB")
    if done_already and not args.rebuild:
        print(f"  (skipping: {', '.join(sorted(done_already))})")
    if args.dry_run:
        return 0

    started = time.perf_counter()
    ok: list[tuple[str, float]] = []
    failed: list[tuple[str, str]] = []
    for n, (vid, path) in enumerate(todo, 1):
        print(f"\n=== [{n}/{len(todo)}] {vid} ===", flush=True)
        each = time.perf_counter()
        try:
            build_index(
                str(path),
                vid,
                enable_ocr=not args.no_ocr,
                scene_caption=not args.no_scene_caption,
                shot_detect=not args.no_shot_detect,
            )
        except KeyboardInterrupt:
            print("\nInterrupted. Re-run to resume; finished videos are skipped.")
            return 130
        except Exception as exc:  # noqa: BLE001 - one bad video must not stop the run
            reason = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED {reason}", flush=True)
            failed.append((vid, reason))
            continue
        elapsed = time.perf_counter() - each
        ok.append((vid, elapsed))
        print(f"  done in {human(elapsed)}", flush=True)

    total = time.perf_counter() - started
    print(f"\n{len(ok)} indexed, {len(failed)} failed, in {human(total)}")
    for vid, elapsed in ok:
        print(f"  {vid:<16} {human(elapsed)}")
    if failed:
        print("\nFailed:")
        for vid, reason in failed:
            print(f"  {vid:<16} {reason}")
        # Re-running picks these up again, since nothing was written to the store.
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
