"""Frame-to-frame motion and flash from the decode we already do.

The visual encoder is still-only. These scalars are the temporal sidecar:
L1 / luma jump on a tiny gray image, stored on the keyframe and vector store payload.
Query time only boosts ANN candidates when the question asks for motion or a flash.
"""

from __future__ import annotations

import re

import numpy as np
from PIL import Image

from vvs.models import MotionSample

TINY_SIZE = (64, 36)  # width, height
FLASH_KEEP = 0.22
MOTION_KEEP = 0.18
ENERGY_NEAR_S = 0.75
FLASH_PEAK_GAP_S = 0.4

_MOTION_TERMS = frozenset(
    {
        "moving",
        "movement",
        "motion",
        "flying",
        "spinning",
        "rotating",
        "rushing",
        "running",
        "driving",
        "swirling",
        "blowing",
        "waving",
        "shaking",
        "speeding",
        "fast",
        "racing",
        "soaring",
    }
)
_FLASH_TERMS = frozenset(
    {
        "lightning",
        "flash",
        "thunderbolt",
        "thunder",
        "strobe",
        "strike",
        "bolt",
    }
)
_TOKEN = re.compile(r"[a-z0-9]+")


def tiny_gray(image: Image.Image | np.ndarray) -> np.ndarray:
    """64×36 grayscale uint8 for L1 / luma. Accepts a PIL image or HxWxC array."""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            pil = Image.fromarray(image.astype(np.uint8), mode="L")
        else:
            pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    else:
        pil = image
    return np.asarray(pil.convert("L").resize(TINY_SIZE, Image.Resampling.BILINEAR), dtype=np.uint8)


def pair_energy(
    prev: np.ndarray | None, curr: np.ndarray
) -> tuple[float, float]:
    """Return (motion_energy, flash_energy) in [0, 1] vs the previous tiny gray."""
    now = np.asarray(curr, dtype=np.float32)
    if prev is None or prev.shape != now.shape:
        return 0.0, 0.0
    before = np.asarray(prev, dtype=np.float32)
    motion = float(np.mean(np.abs(now - before)) / 255.0)
    flash = float(abs(float(now.mean()) - float(before.mean())) / 255.0)
    return min(1.0, motion), min(1.0, flash)


def should_keep_sample(
    *,
    similar: bool,
    force_uniform: bool,
    motion_energy: float,
    flash_energy: float,
    flash_keep: float = FLASH_KEEP,
    motion_keep: float = MOTION_KEEP,
) -> bool:
    """Drop near-duplicates unless the clock floor, a flash, or strong motion fired."""
    if not similar or force_uniform:
        return True
    return flash_energy >= flash_keep or motion_energy >= motion_keep


def energies_from_rgb_stack(
    frames: np.ndarray, times: np.ndarray
) -> list[MotionSample]:
    """Per-frame motion/flash from a TransNet (or any) RGB stack."""
    if frames.size == 0:
        return []
    prev: np.ndarray | None = None
    out: list[MotionSample] = []
    for rgb, t in zip(frames, times, strict=False):
        gray = tiny_gray(rgb)
        motion, flash = pair_energy(prev, gray)
        out.append(
            MotionSample(
                t_s=round(float(t), 3),
                motion_energy=round(motion, 4),
                flash_energy=round(flash, 4),
            )
        )
        prev = gray
    return out


def energy_at(
    track: list[MotionSample], t_s: float, *, near_s: float = ENERGY_NEAR_S
) -> tuple[float, float]:
    if not track:
        return 0.0, 0.0
    nearest = min(track, key=lambda sample: abs(sample.t_s - float(t_s)))
    if abs(nearest.t_s - float(t_s)) > near_s:
        return 0.0, 0.0
    return float(nearest.motion_energy), float(nearest.flash_energy)


def flash_peak_times(
    track: list[MotionSample],
    *,
    threshold: float = FLASH_KEEP,
    min_gap_s: float = FLASH_PEAK_GAP_S,
) -> list[float]:
    """Local flash maxima to force-keep (lightning)."""
    peaks: list[float] = []
    for i, sample in enumerate(track):
        if sample.flash_energy < threshold:
            continue
        prev_e = track[i - 1].flash_energy if i else -1.0
        next_e = track[i + 1].flash_energy if i + 1 < len(track) else -1.0
        if sample.flash_energy < prev_e or sample.flash_energy < next_e:
            continue
        if peaks and sample.t_s - peaks[-1] < min_gap_s:
            if sample.flash_energy > next(
                (s.flash_energy for s in track if abs(s.t_s - peaks[-1]) < 1e-6),
                0.0,
            ):
                peaks[-1] = sample.t_s
            continue
        peaks.append(sample.t_s)
    return peaks


def query_motion_flags(query: str) -> tuple[bool, bool]:
    """(wants_motion, wants_flash) from query tokens. Not a title keep/drop."""
    tokens = set(_TOKEN.findall((query or "").lower()))
    return bool(tokens & _MOTION_TERMS), bool(tokens & _FLASH_TERMS)


def motion_query_score(
    query: str, motion_energy: float = 0.0, flash_energy: float = 0.0
) -> float:
    want_motion, want_flash = query_motion_flags(query)
    score = 0.0
    if want_motion:
        score += max(0.0, float(motion_energy))
    if want_flash:
        score += max(0.0, float(flash_energy))
    return score
