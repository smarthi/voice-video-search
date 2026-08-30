"""Motion / flash sidecar (no VLM, no video file)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from vvs.media.motion import (
    FLASH_KEEP,
    energies_from_rgb_stack,
    energy_at,
    flash_peak_times,
    motion_query_score,
    pair_energy,
    query_motion_flags,
    should_keep_sample,
    tiny_gray,
)
from vvs.media.shots import flash_reps_for_shots, merge_rep_times
from vvs.models import MotionSample, ShotSpan


def test_pair_energy_still_vs_flash() -> None:
    dark = np.zeros((36, 64), dtype=np.uint8)
    bright = np.full((36, 64), 220, dtype=np.uint8)
    motion, flash = pair_energy(dark, bright)
    assert flash > 0.7
    assert motion > 0.7
    still_m, still_f = pair_energy(dark, dark)
    assert still_m < 0.01
    assert still_f < 0.01


def test_tiny_gray_from_rgb() -> None:
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200
    gray = tiny_gray(rgb)
    assert gray.shape == (36, 64)
    assert gray.mean() > 50


def test_should_keep_flash_and_motion() -> None:
    assert should_keep_sample(
        similar=True, force_uniform=False, motion_energy=0.02, flash_energy=0.03
    ) is False
    assert should_keep_sample(
        similar=True, force_uniform=False, motion_energy=0.02, flash_energy=FLASH_KEEP
    ) is True
    assert should_keep_sample(
        similar=True, force_uniform=False, motion_energy=0.25, flash_energy=0.01
    ) is True
    assert should_keep_sample(
        similar=False, force_uniform=False, motion_energy=0.0, flash_energy=0.0
    ) is True


def test_query_motion_flags() -> None:
    assert query_motion_flags("clips with a trumpet") == (False, False)
    assert query_motion_flags("moving tornado") == (True, False)
    assert query_motion_flags("lightning over the field") == (False, True)
    assert motion_query_score("tornado", 0.4, 0.5) == 0.0
    assert motion_query_score("moving tornado", 0.4, 0.0) == 0.4
    assert motion_query_score("lightning", 0.0, 0.5) == 0.5


def test_flash_peaks_and_energy_at() -> None:
    track = [
        MotionSample(t_s=0.0, motion_energy=0.02, flash_energy=0.01),
        MotionSample(t_s=0.5, motion_energy=0.03, flash_energy=0.40),
        MotionSample(t_s=1.0, motion_energy=0.03, flash_energy=0.08),
        MotionSample(t_s=4.0, motion_energy=0.20, flash_energy=0.02),
    ]
    peaks = flash_peak_times(track)
    assert peaks == [0.5]
    motion, flash = energy_at(track, 0.52)
    assert flash == 0.40
    assert energy_at(track, 9.0) == (0.0, 0.0)


def test_energies_from_rgb_stack() -> None:
    dark = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    dark[1] = 200
    times = np.array([0.0, 0.5])
    samples = energies_from_rgb_stack(dark, times)
    assert len(samples) == 2
    assert samples[0].flash_energy == 0.0
    assert samples[1].flash_energy > 0.5


def test_merge_flash_reps() -> None:
    shots = [ShotSpan(shot_id=0, start_s=0.0, end_s=10.0)]
    track = [
        MotionSample(t_s=3.1, motion_energy=0.1, flash_energy=0.5),
        MotionSample(t_s=3.2, motion_energy=0.1, flash_energy=0.3),
    ]
    extras = flash_reps_for_shots(shots, track)
    assert extras
    merged = merge_rep_times([(0, 0.5), (0, 8.5)], extras)
    assert any(abs(t - 3.1) < 0.05 for _, t in merged)


def test_tiny_gray_from_pil() -> None:
    image = Image.new("RGB", (200, 100), (10, 10, 10))
    assert tiny_gray(image).shape == (36, 64)


if __name__ == "__main__":
    test_pair_energy_still_vs_flash()
    test_tiny_gray_from_rgb()
    test_should_keep_flash_and_motion()
    test_query_motion_flags()
    test_flash_peaks_and_energy_at()
    test_energies_from_rgb_stack()
    test_merge_flash_reps()
    test_tiny_gray_from_pil()
    print("ok")
