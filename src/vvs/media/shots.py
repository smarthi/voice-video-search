"""TransNetV2 shot grouping: cut the video, then embed a few stills per shot.

A take with no hard cut is still one shot. Long shots are sampled on a
time floor so a 3-minute fly-through does not collapse to a single vector.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from vvs.media.frames import dhash64, _save_jpeg
from vvs.media.motion import energy_at, flash_peak_times
from vvs.models import Keyframe, MotionSample, ShotSpan
from vvs.paths import frames_dir

TRANSNET_H, TRANSNET_W = 27, 48
WINDOW = 100
STRIDE = 50
CONTEXT = 25
DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_GAP_S = 8.0
WEIGHTS_REPO = "MiaoshouAI/transnetv2-pytorch-weights"
WEIGHTS_FILE = "transnetv2-pytorch-weights.pth"


def predictions_to_scenes(predictions: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Official TransNetV2 scene intervals: inclusive [start, end] frame indices."""
    flags = (np.asarray(predictions) > threshold).astype(np.uint8)
    scenes: list[list[int]] = []
    t_prev, start = 0, 0
    last_i = 0
    t = 0
    for i, t in enumerate(flags.tolist()):
        last_i = i
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, last_i])
    if not scenes:
        return np.array([[0, max(0, len(flags) - 1)]], dtype=np.int32)
    return np.array(scenes, dtype=np.int32)


def scenes_to_shots(
    scenes: np.ndarray,
    times: np.ndarray,
    *,
    duration_s: float = 0.0,
) -> list[ShotSpan]:
    """Map TransNet frame intervals onto clock time."""
    times = np.asarray(times, dtype=np.float64)
    n = int(times.shape[0])
    shots: list[ShotSpan] = []
    for shot_id, (a, b) in enumerate(np.asarray(scenes).tolist()):
        start_i = int(max(0, min(a, n - 1)))
        end_i = int(max(start_i, min(b, n - 1)))
        end_s = float(times[end_i])
        if shot_id + 1 < len(scenes):
            nxt = int(scenes[shot_id + 1][0])
            if 0 <= nxt < n:
                end_s = float(times[nxt])
        elif duration_s > end_s:
            end_s = float(duration_s)
        start_s = float(times[start_i])
        if end_s <= start_s:
            end_s = start_s + max(1.0 / 30.0, 0.04)
        shots.append(
            ShotSpan(
                shot_id=shot_id,
                start_s=round(start_s, 3),
                end_s=round(end_s, 3),
                start_frame=start_i,
                end_frame=end_i,
            )
        )
    return shots


def pick_rep_times(shots: list[ShotSpan], *, max_gap_s: float = DEFAULT_MAX_GAP_S) -> list[tuple[int, float]]:
    """One still near the start of each gap inside the shot. Returns (shot_id, t_s)."""
    gap = max(0.5, float(max_gap_s))
    picked: list[tuple[int, float]] = []
    for shot in shots:
        span = max(0.0, shot.end_s - shot.start_s)
        if span <= 0:
            picked.append((shot.shot_id, shot.start_s))
            continue
        first = shot.start_s + min(gap / 2.0, span / 2.0)
        t = first
        while t < shot.end_s - 1e-3:
            picked.append((shot.shot_id, round(float(t), 3)))
            t += gap
        if not any(sid == shot.shot_id for sid, _ in picked):
            picked.append((shot.shot_id, round(shot.start_s + span / 2.0, 3)))
    return picked


def merge_rep_times(
    reps: list[tuple[int, float]], extra: list[tuple[int, float]], *, min_gap_s: float = 0.35
) -> list[tuple[int, float]]:
    """Union of shot reps and flash peaks, dropping times that sit on an existing still."""
    merged = list(reps)
    for shot_id, t_s in extra:
        if any(abs(t_s - existing) < min_gap_s for _, existing in merged):
            continue
        merged.append((shot_id, t_s))
    merged.sort(key=lambda item: item[1])
    return merged


def flash_reps_for_shots(
    shots: list[ShotSpan], track: list[MotionSample]
) -> list[tuple[int, float]]:
    extras: list[tuple[int, float]] = []
    for t_s in flash_peak_times(track):
        shot = shot_covering(t_s, shots)
        extras.append((shot.shot_id if shot else (shots[0].shot_id if shots else 0), t_s))
    return extras


def shot_covering(t_s: float, shots: list[ShotSpan]) -> ShotSpan | None:
    t = float(t_s)
    for shot in shots:
        if shot.start_s <= t < shot.end_s:
            return shot
    if shots and t >= shots[-1].start_s and t <= shots[-1].end_s + 0.05:
        return shots[-1]
    return None


def detect_shots(
    video_file: Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    duration_s: float = 0.0,
) -> tuple[list[ShotSpan], list[MotionSample]]:
    """Run TransNetV2 on 48×27 RGB frames decoded with PyAV (no ffmpeg CLI)."""
    preds, times, track = predict_shot_scores_stream(video_file)
    if preds.size == 0:
        raise RuntimeError(f"No frames decoded from {video_file}")
    scenes = predictions_to_scenes(preds, threshold=threshold)
    shots = scenes_to_shots(scenes, times, duration_s=duration_s or float(times[-1]))
    flashes = flash_peak_times(track)
    print(
        f"[shots] TransNetV2 {len(shots)} shots from {len(times)} frames @ 48x27"
        f"{f', flash peaks {len(flashes)}' if flashes else ''}"
    )
    return shots, track


def predict_shot_scores_stream(video_file: Path) -> tuple[np.ndarray, np.ndarray, list[MotionSample]]:
    """Decode and infer in a bounded 100-frame window, never retaining video RGB.

    Only timestamps, transition scores, and tiny motion samples grow with
    duration. The previous implementation held every 48×27 RGB frame before
    inference, which needlessly made long videos memory-bound.
    """
    import av
    import torch

    container = av.open(str(video_file))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    rate = float(stream.average_rate) if stream.average_rate else 0.0
    model = _transnet()
    buffer: list[np.ndarray] = []
    times: list[float] = []
    track: list[MotionSample] = []
    preds: list[np.ndarray] = []
    first: np.ndarray | None = None
    last: np.ndarray | None = None
    previous: np.ndarray | None = None
    idx = 0

    def infer_ready() -> None:
        while len(buffer) >= WINDOW:
            window = np.stack(buffer[:WINDOW], axis=0)
            with torch.no_grad():
                single, _many = model(torch.from_numpy(window).unsqueeze(0))
                preds.append(torch.sigmoid(single)[0, CONTEXT : CONTEXT + STRIDE, 0].cpu().numpy())
            del buffer[:STRIDE]

    try:
        for frame in container.decode(stream):
            t = float(frame.time if frame.time is not None else (idx / rate if rate else 0.0))
            rgb = frame.to_ndarray(format="rgb24")
            small = np.asarray(Image.fromarray(rgb).resize((TRANSNET_W, TRANSNET_H), Image.Resampling.BILINEAR), dtype=np.uint8)
            if first is None:
                first = small
                buffer.extend([small] * CONTEXT)
            last = small
            times.append(t)
            gray = np.asarray(Image.fromarray(small).convert("L").resize((64, 36), Image.Resampling.BILINEAR), dtype=np.uint8)
            motion, flash = (0.0, 0.0) if previous is None else (float(np.mean(np.abs(gray.astype(np.float32) - previous.astype(np.float32))) / 255.0), float(abs(float(gray.mean()) - float(previous.mean())) / 255.0))
            track.append(MotionSample(t_s=round(t, 3), motion_energy=round(min(1.0, motion), 4), flash_energy=round(min(1.0, flash), 4)))
            previous = gray
            buffer.append(small)
            infer_ready()
            idx += 1
            if idx % 500 == 0:
                print(f"[shots] decoded {idx} frames for TransNetV2")
    finally:
        container.close()
    if not times or first is None or last is None:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float64), []
    pad_end = CONTEXT + STRIDE - (len(times) % STRIDE if len(times) % STRIDE else STRIDE)
    buffer.extend([last] * pad_end)
    infer_ready()
    scores = np.concatenate(preds)[: len(times)] if preds else np.zeros((len(times),), dtype=np.float32)
    return scores.astype(np.float32), np.asarray(times, dtype=np.float64), track


def extract_shot_keyframes(
    video_file: Path,
    video_id: str,
    shots: list[ShotSpan],
    *,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    jpeg_quality: int = 85,
    max_width: int = 960,
    motion_track: list[MotionSample] | None = None,
) -> list[Keyframe]:
    """Second pass: save only the representative stills (the ones we embed)."""
    import av

    track = list(motion_track or [])
    targets = merge_rep_times(
        pick_rep_times(shots, max_gap_s=max_gap_s),
        flash_reps_for_shots(shots, track),
    )
    if not targets:
        raise RuntimeError("No shot representatives to extract")
    dests = list(targets)
    next_i = 0
    next_t = dests[0][1]
    slop = max(0.04, (dests[1][1] - dests[0][1]) * 0.25) if len(dests) > 1 else 0.2

    out_dir = frames_dir(video_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jpg"):
        stale.unlink()

    kept: list[Keyframe] = []
    container = av.open(str(video_file))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    for frame in container.decode(stream):
        if next_i >= len(dests):
            break
        t = float(frame.time or 0.0)
        if t + slop < next_t:
            continue
        rgb = frame.to_ndarray(format="rgb24")
        image = Image.fromarray(rgb)
        digest = dhash64(image)
        shot_id, _want = dests[next_i]
        motion_energy, flash_energy = energy_at(track, t)
        saved = _save_jpeg(
            image,
            out_dir / f"{len(kept):04d}_{t:08.2f}s_shot{shot_id:03d}.jpg",
            max_width,
            jpeg_quality,
        )
        kept.append(
            Keyframe(
                index=len(kept),
                t_s=round(t, 3),
                path=str(saved),
                hash_hex=f"{digest:064x}",
                shot_id=shot_id,
                motion_energy=round(motion_energy, 4),
                flash_energy=round(flash_energy, 4),
            )
        )
        next_i += 1
        if next_i < len(dests):
            next_t = dests[next_i][1]
    container.close()
    if not kept:
        raise RuntimeError(f"No shot keyframes extracted from {video_file}")
    print(f"[shots] kept {len(kept)} representatives from {len(shots)} shots (gap {max_gap_s:.1f}s)")
    return kept


def predict_shot_scores(frames: np.ndarray) -> np.ndarray:
    """Per-frame transition probability (official 100-frame window, stride 50)."""
    import torch

    model = _transnet()
    n = int(frames.shape[0])
    pad_start = CONTEXT
    pad_end = CONTEXT + STRIDE - (n % STRIDE if n % STRIDE != 0 else STRIDE)
    start = np.repeat(frames[0:1], pad_start, axis=0)
    end = np.repeat(frames[-1:], pad_end, axis=0)
    padded = np.concatenate([start, frames, end], axis=0)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        ptr = 0
        while ptr + WINDOW <= len(padded):
            window = padded[ptr : ptr + WINDOW]
            tensor = torch.from_numpy(window).unsqueeze(0)
            single, _many = model(tensor)
            chunk = torch.sigmoid(single)[0, CONTEXT : CONTEXT + STRIDE, 0].cpu().numpy()
            preds.append(chunk)
            ptr += STRIDE
            done = min(len(preds) * STRIDE, n)
            if done % 500 == 0 or done == n:
                print(f"[shots] TransNetV2 {done}/{n}")
    scores = np.concatenate(preds)[:n] if preds else np.zeros((n,), dtype=np.float32)
    return scores.astype(np.float32)


@lru_cache(maxsize=1)
def _transnet():
    import torch

    from vvs.media.transnetv2_model import TransNetV2

    weights = _weights_path()
    print(f"[shots] loading TransNetV2 ({weights})")
    model = TransNetV2()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _weights_path() -> Path:
    names = (WEIGHTS_FILE, "transnetv2-pytorch.pth", "transnetv2.pth")
    last: Exception | None = None
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required to fetch TransNetV2 weights") from exc
    for name in names:
        try:
            return Path(hf_hub_download(repo_id=WEIGHTS_REPO, filename=name))
        except Exception as exc:
            last = exc
    raise RuntimeError(
        f"Could not download TransNetV2 weights from {WEIGHTS_REPO}: {last}"
    ) from last
