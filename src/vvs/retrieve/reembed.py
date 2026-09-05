"""Clone an index and rebuild only its visual channel with a different encoder.

Re-running `vvs index` to test a second encoder would redo decode, ASR, OCR and
captioning — minutes of work unrelated to the thing under test — and any
nondeterminism in keyframe selection would leak into the comparison as if it
were an encoder difference. Instead this reuses the finished artifacts and
replaces only the visual collection's rows for `dest_id`, so the two indexes
are guaranteed to describe the same keyframes and the same text channel.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from vvs.config import Settings, load_settings
from vvs.models import IndexMeta
from vvs.paths import meta_path, video_dir, write_json_atomic
from vvs.retrieve.embed import (
    embed_clip_images,
    encoder_kind,
    visual_dim,
    visual_model_label,
)
from vvs.retrieve.opensearch_store import clone_video
from vvs.retrieve.store import VectorPayload, open_store, visual_store_exists

_LINKED = ("video.mp4",)
_COPIED = ("timings.json", "demo_questions.json")


@dataclass
class ReembedResult:
    video_id: str
    encoder: str
    model: str
    dim: int
    frames: int
    embed_s: float

    @property
    def per_frame_ms(self) -> float:
        return (self.embed_s / self.frames * 1000.0) if self.frames else 0.0


def reembed_visual(
    source_id: str,
    dest_id: str,
    *,
    settings: Settings | None = None,
    overwrite: bool = False,
    limit: int | None = None,
) -> ReembedResult:
    """Write `dest_id` as `source_id` with a freshly encoded visual store."""
    cfg = settings or load_settings()
    if source_id == dest_id:
        raise SystemExit("source and destination ids must differ")
    if not visual_store_exists(source_id):
        raise SystemExit(f"No visual store for {source_id!r}. Index it first.")

    src_dir = video_dir(source_id)
    dst_dir = video_dir(dest_id)
    if dst_dir.exists():
        if not overwrite:
            raise SystemExit(f"{dst_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    frames = _read_frame_payloads(source_id, cfg, limit=limit)
    if not frames:
        raise SystemExit(f"{source_id!r} has no video_frame vectors to re-embed.")

    kind = encoder_kind(settings=cfg)
    model = visual_model_label(settings=cfg)
    print(f"[reembed] {source_id} -> {dest_id} with {kind} ({model}), {len(frames)} frames")

    started = perf_counter()
    vecs = embed_clip_images([p.path for p in frames], settings=cfg)
    embed_s = perf_counter() - started
    if vecs.shape[0] != len(frames):
        raise SystemExit(
            f"encoder returned {vecs.shape[0]} vectors for {len(frames)} frames"
        )

    _clone_artifacts(src_dir, dst_dir)
    n_text = clone_video("text", source_id, dest_id, settings=cfg)
    if n_text:
        print(f"[reembed] cloned {n_text} text-channel rows (transcript/OCR/scene) unchanged")
    _write_visual(dest_id, cfg, frames, vecs)
    _write_meta(source_id, dest_id, cfg, dim=int(vecs.shape[1]))

    return ReembedResult(
        video_id=dest_id,
        encoder=kind,
        model=model,
        dim=int(vecs.shape[1]),
        frames=len(frames),
        embed_s=embed_s,
    )


def _read_frame_payloads(
    video_id: str, cfg: Settings, *, limit: int | None
) -> list[VectorPayload]:
    meta = IndexMeta.model_validate_json(meta_path(video_id).read_text(encoding="utf-8"))
    store = open_store(
        video_id, "visual", dim=visual_dim(meta=meta), settings=cfg, create=False
    )
    try:
        rows = store.all_payloads(modality="video_frame")
    finally:
        store.close()
    usable = [row for row in rows if row.path and Path(row.path).exists()]
    missing = len(rows) - len(usable)
    if missing:
        print(f"[reembed] skipping {missing} frames whose JPEG is gone")
    return usable[:limit] if limit else usable


def _clone_artifacts(src_dir: Path, dst_dir: Path) -> None:
    """Share the heavy inputs, copy the small local state.

    Frame JPEGs stay where they are: payload paths keep pointing at the source
    directory, so both indexes read the identical bytes and we avoid duplicating
    a frame set per encoder. The video is symlinked so clip extraction in the UI
    still works. The text channel (transcript/OCR/scene) lives in the shared
    OpenSearch index, not on disk, so it is cloned separately via
    `clone_video("text", ...)` in `reembed_visual`, not copied here.
    """
    for name in _LINKED:
        src = src_dir / name
        if not src.exists():
            continue
        link = dst_dir / name
        try:
            link.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, link)
    for name in _COPIED:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


def _write_visual(
    dest_id: str, cfg: Settings, frames: list[VectorPayload], vecs: np.ndarray
) -> None:
    store = open_store(
        dest_id, "visual", dim=int(vecs.shape[1]), settings=cfg, create=True
    )
    try:
        store.add(
            vecs,
            [
                VectorPayload(
                    vector_id=0,
                    asset_id=dest_id,
                    t_s=row.t_s,
                    modality="video_frame",
                    path=row.path,
                    frame_index=row.frame_index,
                    shot_id=row.shot_id,
                    motion_energy=row.motion_energy,
                    flash_energy=row.flash_energy,
                )
                for row in frames
            ],
        )
    finally:
        store.close()


def _write_meta(source_id: str, dest_id: str, cfg: Settings, *, dim: int) -> None:
    meta = IndexMeta.model_validate_json(meta_path(source_id).read_text(encoding="utf-8"))
    card = meta.models.model_copy(
        update={
            "clip": visual_model_label(settings=cfg),
            "visual_dim": dim,
            "visual_encoder": encoder_kind(settings=cfg),
        }
    )
    updates: dict[str, object] = {"video_id": dest_id, "models": card}
    video = video_dir(dest_id) / "video.mp4"
    if video.exists():
        updates["video_path"] = str(video.resolve())
    clone = meta.model_copy(update=updates)
    write_json_atomic(meta_path(dest_id), clone.model_dump_json(indent=2))
