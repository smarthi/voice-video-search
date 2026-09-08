"""TransNetV2 shot grouping without downloading weights."""

from __future__ import annotations

import numpy as np

from vvs.media.scenes import drafts_from_shots
from vvs.media.shots import pick_rep_times, predictions_to_scenes, scenes_to_shots, shot_covering
from vvs.models import Keyframe, ShotSpan


def test_predictions_to_scenes_finds_cuts() -> None:
    # 0 = in-shot, 1 = transition frame (official TransNet encoding)
    pred = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0], dtype=np.float32)
    scenes = predictions_to_scenes(pred, threshold=0.5)
    assert len(scenes) >= 2
    assert scenes[0, 0] == 0


def test_predictions_all_one_shot() -> None:
    pred = np.zeros(20, dtype=np.float32)
    scenes = predictions_to_scenes(pred, threshold=0.5)
    assert len(scenes) == 1
    assert scenes[0, 0] == 0
    assert scenes[0, 1] == 19


def test_pick_reps_caps_long_shot() -> None:
    shots = [ShotSpan(shot_id=0, start_s=0.0, end_s=40.0, start_frame=0, end_frame=100)]
    reps = pick_rep_times(shots, max_gap_s=8.0)
    assert 4 <= len(reps) <= 6
    assert all(sid == 0 for sid, _ in reps)
    times = [t for _, t in reps]
    assert times[0] < 8.0
    assert times[-1] < 40.0


def test_pick_reps_one_per_short_shot() -> None:
    shots = [
        ShotSpan(shot_id=0, start_s=0.0, end_s=3.0),
        ShotSpan(shot_id=1, start_s=3.0, end_s=6.0),
        ShotSpan(shot_id=2, start_s=6.0, end_s=9.0),
    ]
    reps = pick_rep_times(shots, max_gap_s=8.0)
    assert len(reps) == 3
    assert [sid for sid, _ in reps] == [0, 1, 2]


def test_scenes_to_shots_and_covering() -> None:
    times = np.linspace(0.0, 9.0, 10)
    scenes = np.array([[0, 4], [5, 9]], dtype=np.int32)
    shots = scenes_to_shots(scenes, times, duration_s=10.0)
    assert len(shots) == 2
    assert shots[0].start_s == 0.0
    assert shot_covering(2.0, shots) is shots[0]
    assert shot_covering(7.0, shots) is shots[1]


def test_drafts_from_shots() -> None:
    frames = [
        Keyframe(index=0, t_s=1.0, path="a.jpg", hash_hex="aa", shot_id=0),
        Keyframe(index=1, t_s=4.0, path="b.jpg", hash_hex="bb", shot_id=1),
    ]
    shots = [
        ShotSpan(shot_id=0, start_s=0.0, end_s=3.0),
        ShotSpan(shot_id=1, start_s=3.0, end_s=6.0),
    ]
    drafts = drafts_from_shots(frames, shots)
    assert len(drafts) == 2
    assert drafts[0].frame_index == 0
    assert drafts[1].frame_index == 1


def test_shot_device_follows_the_shared_choice_and_honours_its_override() -> None:
    """Inference is ~87% of shot detection, so this must not silently sit on CPU."""
    import os

    from vvs.device import resolve_device
    from vvs.media.shots import shot_device

    os.environ.pop("VVS_SHOT_DEVICE", None)
    assert shot_device() == resolve_device()
    try:
        # The int32 scatter_add_ in the histogram path is the reason an escape
        # hatch exists at all.
        os.environ["VVS_SHOT_DEVICE"] = "cpu"
        assert shot_device() == "cpu"
    finally:
        os.environ.pop("VVS_SHOT_DEVICE", None)


def test_inference_inputs_are_moved_to_wherever_the_weights_landed() -> None:
    """A requested device can be refused at load time; inputs must still match."""
    import inspect

    from vvs.media import shots

    for fn in (shots.predict_shot_scores_stream, shots.predict_shot_scores):
        src = inspect.getsource(fn)
        assert "_model_device(model)" in src, f"{fn.__name__} assumes a device"
        assert src.count(".to(device)") >= 1, f"{fn.__name__} feeds an unmoved tensor"
        # Scores come back to numpy, so the caller never sees a device tensor.
        assert ".cpu().numpy()" in src


if __name__ == "__main__":
    test_predictions_to_scenes_finds_cuts()
    test_predictions_all_one_shot()
    test_pick_reps_caps_long_shot()
    test_pick_reps_one_per_short_shot()
    test_scenes_to_shots_and_covering()
    test_drafts_from_shots()
    print("ok")
