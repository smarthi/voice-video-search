#!/usr/bin/env python
"""Sweep the visual-embedding batch size and report throughput per batch.

`default_image_batch()` returns a flat 8 for every non-CPU device. That number
was picked to be safe on unknown hardware, not measured, so on a machine with
real headroom -- 128 GB of unified memory, or a discrete card -- a wider batch
may be most of the remaining win. The right value is hardware-specific, so
measure it where you actually run.

    uv run python scripts/bench_batch.py                      # 1,4,8,16,32
    uv run python scripts/bench_batch.py --batches 8,32,64 --frames 64

Alongside the timings this checks that batching does not change the vectors.
Attention over a padded batch is not bit-identical to one frame at a time, and
a batch wide enough to shift results would trade recall for speed without
saying so. Every trial is compared against the batch-1 vectors.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vvs.device import resolve_device, resolve_dtype
from vvs.paths import DATA_ROOT
from vvs.retrieve.embed import default_image_batch, embed_clip_images


def collect_frames(count: int, video: str | None, seed: int = 0) -> list[Path]:
    """Sample real keyframes. Content does not affect cost, but real sizes do."""
    root = DATA_ROOT / video / "frames" if video else DATA_ROOT
    frames = sorted(root.rglob("*.jpg")) if root.exists() else []
    if not frames:
        raise SystemExit(
            f"no keyframes under {root}. Index a video first, or pass --video ID."
        )
    if len(frames) < count:
        print(f"[bench] only {len(frames)} frames available; using all of them")
        return frames
    # Sample rather than take a prefix: consecutive keyframes come from one
    # scene and can share a resolution or a letterbox that is not typical.
    return random.Random(seed).sample(frames, count)


# Batching pads sequences together and changes the order of the attention
# reductions, so in fp16 the vectors differ in about the fourth decimal no
# matter what. That is arithmetic, not a quality regression: retrieval score
# gaps between neighbouring hits are orders of magnitude larger, so a cosine
# this close cannot reorder results. Only a bigger gap means the batch is
# genuinely computing something else.
_NOISE_FLOOR = 0.999


def classify_drift(cos: float) -> str:
    if cos > 0.99999:
        return "identical"
    if cos >= _NOISE_FLOOR:
        return f"fp16 noise ({cos:.5f})"
    return f"DRIFT ({cos:.5f})"


def peak_memory_mb(device: str) -> float | None:
    """Peak allocation since the last reset, in MB. None where torch cannot say."""
    import torch

    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    if device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / 1e6
    return None


def reset_memory(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def sync(device: str) -> None:
    """Block until the GPU is actually finished, or the timings are fiction."""
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def trial(frames: list[Path], batch: int, device: str) -> tuple[float, np.ndarray, float | None]:
    reset_memory(device)
    sync(device)
    start = time.perf_counter()
    vectors = embed_clip_images([str(p) for p in frames], batch_size=batch)
    sync(device)
    return time.perf_counter() - start, vectors, peak_memory_mb(device)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", default="1,4,8,16,32", help="comma-separated sizes")
    ap.add_argument("--frames", type=int, default=32, help="frames embedded per trial")
    ap.add_argument("--video", default=None, help="restrict frames to one video id")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    if not batches:
        raise SystemExit("--batches parsed to nothing")

    device = resolve_device()
    print(f"[bench] device={device} dtype={resolve_dtype(device)} "
          f"current default batch={default_image_batch(device)}")
    if device == "cpu":
        print("[bench] WARNING: on CPU the batch size barely matters and every "
              "trial will be slow. This sweep is meant for a GPU.")

    frames = collect_frames(args.frames, args.video, args.seed)
    print(f"[bench] {len(frames)} frames, batches {batches}\n")

    # Warm up so the first trial is not charged for the model load, the lazy
    # weight materialization, or the first-kernel compile.
    print("[bench] warming up (model load is not timed) ...", flush=True)
    embed_clip_images([str(frames[0])], batch_size=1)
    sync(device)

    rows: list[tuple[int, float, float, float | None, str]] = []
    reference: np.ndarray | None = None
    for batch in batches:
        try:
            elapsed, vectors, peak = trial(frames, batch, device)
        except (RuntimeError, MemoryError) as exc:
            first = str(exc).strip().splitlines()[0][:70]
            print(f"  batch {batch:>3}  FAILED  {first}")
            rows.append((batch, float("nan"), float("nan"), None, "oom/error"))
            continue
        if reference is None:
            reference = vectors
            drift = "reference"
        else:
            # Vectors are L2-normalized, so a row-wise dot product is cosine.
            cos = float(np.min(np.sum(reference * vectors, axis=1)))
            drift = classify_drift(cos)
        per_frame = elapsed / len(frames)
        rows.append((batch, elapsed, per_frame, peak, drift))
        mem = f"{peak:>7.0f} MB" if peak is not None else "      n/a"
        print(f"  batch {batch:>3}  {elapsed:>7.2f} s  {per_frame:>6.3f} s/frame  "
              f"{len(frames) / elapsed:>6.2f} frame/s  {mem}  {drift}")

    ok = [r for r in rows if r[1] == r[1]]  # drop the NaN (failed) rows
    if not ok:
        print("\nEvery trial failed.")
        return 1

    print(f"\n{'batch':>6} {'s/frame':>9} {'frame/s':>9} {'vs batch ' + str(ok[0][0]):>12} {'peak':>10}")
    base = ok[0][2]
    for batch, _elapsed, per_frame, peak, drift in ok:
        mem = f"{peak:.0f} MB" if peak is not None else "n/a"
        flag = f"  <- {drift}" if drift.startswith("DRIFT") else ""
        print(f"{batch:>6} {per_frame:>9.3f} {1 / per_frame:>9.2f} "
              f"{base / per_frame:>11.2f}x {mem:>10}{flag}")

    best = min(ok, key=lambda r: r[2])
    current = default_image_batch(device)
    print(f"\nFastest: batch {best[0]} at {best[2]:.3f} s/frame "
          f"({base / best[2]:.2f}x the batch-{ok[0][0]} rate).")

    measured = {r[0]: r for r in ok}
    if best[0] == current:
        print(f"That is already the default ({current}). Nothing to change.")
    elif current in measured:
        margin = measured[current][2] / best[2] - 1
        head = f"Batch {best[0]} beats the default ({current}) by {margin:.0%}"
        if margin < 0.15:
            extra = ""
            if measured[current][3] and best[3]:
                extra = (f", for {best[3] - measured[current][3]:.0f} MB more "
                         "memory and less room under the cliff")
            print(f"{head}{extra}. That is inside the noise of a "
                  f"{len(frames)}-frame trial -- re-run with several times more "
                  "frames before treating it as real.")
        else:
            print(f"{head}. To use it:\n    export VVS_VISUAL_BATCH={best[0]}")
    else:
        print(f"The default is {current}, which this sweep did not measure. "
              f"To use the winner:\n    export VVS_VISUAL_BATCH={best[0]}")
    # A collapse means the batch no longer fits and the allocator is thrashing.
    # Worth calling out, because the cliff is sudden rather than gradual: the
    # size just under it looks fine and the next one up is many times worse.
    cliff = [r for r in ok if r[2] > ok[0][2]]
    if cliff:
        print("\nSlower than batch 1, so these no longer fit in memory:")
        for batch, _e, per_frame, peak, _d in cliff:
            mem = f", peaked at {peak:.0f} MB" if peak is not None else ""
            print(f"    batch {batch}: {per_frame:.3f} s/frame{mem}")
        print("Leave a margin below the cliff; a frame set at a higher "
              "resolution will reach it at a smaller batch.")

    drifted = [r for r in ok if r[4].startswith("DRIFT")]
    if drifted:
        print(f"\nWARNING: these fell below cosine {_NOISE_FLOOR} against "
              "batch 1, which is more than fp16 reduction order explains, so "
              "check retrieval quality before adopting them:")
        for batch, _e, _p, _m, drift in drifted:
            print(f"    batch {batch}: {drift}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
