"""Group keyframes into visual scenes for X-Ray cards.

Uses visual-encoder cosine when vectors are available, otherwise dHash from the
keyframe extractor. Consecutive similar frames stay in one scene until
the picture changes or the scene hits a maximum length.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vvs.models import Keyframe, SceneCard, ShotSpan

COSINE_JOIN = 0.86
HASH_JOIN = 0.78
MAX_SCENE_S = 24.0


@dataclass(frozen=True)
class SceneDraft:
    start_i: int
    end_i: int  # inclusive frame index into `frames`
    frame_index: int


def cluster_scenes(
    frames: list[Keyframe],
    frame_vecs: np.ndarray | None = None,
    *,
    duration_s: float = 0.0,
    cosine_join: float = COSINE_JOIN,
    hash_join: float = HASH_JOIN,
    max_scene_s: float = MAX_SCENE_S,
) -> list[SceneDraft]:
    """Split a keyframe run into scenes. Does not call a captioner."""
    if not frames:
        return []
    vecs = None
    if frame_vecs is not None and len(frame_vecs) == len(frames):
        vecs = np.asarray(frame_vecs, dtype=np.float32)

    starts = [0]
    for i in range(1, len(frames)):
        prev = starts[-1]
        too_long = (frames[i].t_s - frames[prev].t_s) >= max_scene_s
        if too_long or not _similar(frames, i, vecs, cosine_join, hash_join):
            starts.append(i)
    starts.append(len(frames))

    drafts: list[SceneDraft] = []
    for a, b in zip(starts, starts[1:]):
        last = b - 1
        mid_t = 0.5 * (frames[a].t_s + frames[last].t_s)
        rep = min(range(a, b), key=lambda j: abs(frames[j].t_s - mid_t))
        drafts.append(SceneDraft(start_i=a, end_i=last, frame_index=frames[rep].index))
    return drafts


def drafts_from_shots(frames: list[Keyframe], shots: list[ShotSpan]) -> list[SceneDraft]:
    """One caption draft per TransNet shot that has at least one still."""
    if not frames or not shots:
        return []
    drafts: list[SceneDraft] = []
    for shot in shots:
        members = [i for i, frame in enumerate(frames) if frame.shot_id == shot.shot_id]
        if not members:
            continue
        mid_t = 0.5 * (shot.start_s + shot.end_s)
        rep = min(members, key=lambda i: abs(frames[i].t_s - mid_t))
        drafts.append(SceneDraft(start_i=members[0], end_i=members[-1], frame_index=frames[rep].index))
    return drafts


def drafts_to_cards(
    frames: list[Keyframe],
    drafts: list[SceneDraft],
    *,
    duration_s: float,
    captions: list[tuple[str, list[str], str]],
) -> list[SceneCard]:
    """Attach captions to clustered scenes and close each interval."""
    cards: list[SceneCard] = []
    n = len(frames)
    for scene_id, draft in enumerate(drafts):
        caption, objects, model = (
            captions[scene_id] if scene_id < len(captions) else ("", [], "")
        )
        next_start = frames[drafts[scene_id + 1].start_i].t_s if scene_id + 1 < len(drafts) else 0.0
        end_s = next_start if next_start > frames[draft.end_i].t_s else _scene_end(
            frames, draft.end_i, duration_s, n
        )
        if end_s <= frames[draft.start_i].t_s:
            end_s = frames[draft.end_i].t_s + 1.0
        cards.append(
            SceneCard(
                scene_id=scene_id,
                start_s=round(frames[draft.start_i].t_s, 3),
                end_s=round(float(end_s), 3),
                frame_index=draft.frame_index,
                path=_path_for_index(frames, draft.frame_index),
                caption=(caption or "").strip(),
                objects=list(objects),
                model=model,
            )
        )
    return cards


def scene_covering(t_s: float, scenes: list[SceneCard]) -> SceneCard | None:
    """X-Ray lookup: the card whose [start, end) covers the playhead."""
    t = float(t_s)
    for scene in scenes:
        if scene.start_s <= t < scene.end_s:
            return scene
    if scenes and t >= scenes[-1].start_s:
        last = scenes[-1]
        if t <= last.end_s + 0.05:
            return last
    return None


def _similar(
    frames: list[Keyframe],
    i: int,
    vecs: np.ndarray | None,
    cosine_join: float,
    hash_join: float,
) -> bool:
    if vecs is not None:
        return float(np.dot(vecs[i], vecs[i - 1])) >= cosine_join
    return _hash_similarity(frames[i].hash_hex, frames[i - 1].hash_hex) >= hash_join


def _hash_similarity(a: str, b: str) -> float:
    try:
        x, y = int(a, 16), int(b, 16)
    except ValueError:
        return 0.0
    bits = max(len(a), len(b), 1) * 4
    return 1.0 - (x ^ y).bit_count() / bits


def _scene_end(frames: list[Keyframe], last_i: int, duration_s: float, n: int) -> float:
    if last_i + 1 < n:
        return frames[last_i + 1].t_s
    if duration_s > frames[last_i].t_s:
        return duration_s
    return frames[last_i].t_s + 2.0


def _path_for_index(frames: list[Keyframe], frame_index: int) -> str:
    for frame in frames:
        if frame.index == frame_index:
            return frame.path
    return frames[0].path
