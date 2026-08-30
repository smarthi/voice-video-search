"""Decode a video at 2 fps and keep a compact keyframe set.

Also writes motion/flash scalars from consecutive 2 fps samples and force-keeps
luma spikes (lightning) that dHash would otherwise drop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from vvs.media.motion import FLASH_KEEP, pair_energy, should_keep_sample, tiny_gray
from vvs.models import Keyframe
from vvs.paths import frames_dir


def extract_keyframes(
    video_file: Path,
    video_id: str,
    *,
    sample_fps: float = 2.0,
    hash_size: int = 16,
    similarity_threshold: float = 0.92,
    uniform_interval_s: float = 4.0,
    jpeg_quality: int = 85,
    max_width: int = 960,
) -> list[Keyframe]:
    """Sample frames, drop near-duplicates via dHash, keep a uniform floor."""
    import av

    out_dir = frames_dir(video_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jpg"):
        stale.unlink()

    container = av.open(str(video_file))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    duration_s = float(stream.duration * stream.time_base) if stream.duration else 0.0

    interval = 1.0 / sample_fps
    next_sample = 0.0
    last_hash: int | None = None
    last_keep_t = -1e9
    last_tiny: np.ndarray | None = None
    kept: list[Keyframe] = []
    bits = hash_size * hash_size
    flash_kept = 0

    for frame in container.decode(stream):
        t = float(frame.time or 0.0)
        if t + 1e-6 < next_sample:
            continue
        next_sample = t + interval

        rgb = frame.to_ndarray(format="rgb24")
        image = Image.fromarray(rgb)
        gray = tiny_gray(image)
        motion_energy, flash_energy = pair_energy(last_tiny, gray)
        last_tiny = gray
        digest = dhash64(image, hash_size=hash_size)
        similar = False
        if last_hash is not None:
            dist = (last_hash ^ digest).bit_count()
            similar = (1.0 - dist / bits) >= similarity_threshold

        force_uniform = (t - last_keep_t) >= uniform_interval_s
        if not should_keep_sample(
            similar=similar,
            force_uniform=force_uniform,
            motion_energy=motion_energy,
            flash_energy=flash_energy,
        ):
            continue
        if similar and flash_energy >= FLASH_KEEP:
            flash_kept += 1

        saved = _save_jpeg(image, out_dir / f"{len(kept):04d}_{t:08.2f}s.jpg", max_width, jpeg_quality)
        kept.append(
            Keyframe(
                index=len(kept),
                t_s=round(t, 3),
                path=str(saved),
                hash_hex=f"{digest:0{bits // 4}x}",
                motion_energy=round(motion_energy, 4),
                flash_energy=round(flash_energy, 4),
            )
        )
        last_hash = digest
        last_keep_t = t

    container.close()
    if not kept:
        raise RuntimeError(f"No keyframes extracted from {video_file}")

    print(
        f"[frames] kept {len(kept)} keyframes from {duration_s:.1f}s "
        f"@ {sample_fps} fps (uniform floor {uniform_interval_s}s"
        f"{f', flash-kept {flash_kept}' if flash_kept else ''})"
    )
    return kept


def dhash64(image: Image.Image, hash_size: int = 16) -> int:
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def _save_jpeg(image: Image.Image, dest: Path, max_width: int, quality: int) -> Path:
    out = image
    if out.width > max_width:
        ratio = max_width / out.width
        out = out.resize((max_width, max(1, int(out.height * ratio))), Image.Resampling.BILINEAR)
    out.save(dest, format="JPEG", quality=quality, optimize=True)
    return dest
