"""Build and load a visual-first ANN index (OpenSearch, one shared index per collection)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np

from vvs.config import load_settings
from vvs.device import should_evict
from vvs.ingest import extract_video_id, ingest, validate_video_id
from vvs.media.caption import caption_scenes, release_captioner
from vvs.media.frames import extract_keyframes
from vvs.media.ocr import is_slide_heavy, ocr_frames
from vvs.media.probe import has_audio_stream
from vvs.media.scenes import cluster_scenes, drafts_from_shots
from vvs.media.shots import detect_shots, extract_shot_keyframes
from vvs.media.transcribe import transcribe
from vvs.models import (
    BuildResult,
    ChannelFlags,
    IndexMeta,
    ModelCard,
    SceneCard,
    ShotSpan,
    StageTimings,
)
from vvs.paths import (
    DATA_ROOT,
    meta_path,
    read_json_text,
    timings_path,
    video_dir,
    video_path,
    write_json_atomic,
)
from vvs.retrieve.embed import (
    embed_clip_images,
    embed_text_passages,
    encoder_kind,
    release_visual_encoder,
    text_dim,
    text_model_label,
    visual_dim,
    visual_model_label,
)
from vvs.retrieve.opensearch_store import delete_other_generations, delete_video
from vvs.retrieve.store import VectorPayload, open_store, visual_store_exists

StageCallback = Callable[[str], None]


def should_run_asr(media: Path, *, force_mute: bool | None = None) -> bool:
    """ASR only when the file has audio and mute is not forced."""
    if force_mute is None:
        force_mute = load_settings().force_mute
    if force_mute:
        print("[index] ASR skipped: VVS_FORCE_MUTE / force_mute")
        return False
    try:
        if not has_audio_stream(media):
            print("[index] ASR skipped: no audio track")
            return False
    except Exception as exc:
        print(f"[index] audio probe failed ({exc}); skipping ASR")
        return False
    return True


def build_index(
    source: str,
    video_id: str | None = None,
    *,
    enable_ocr: bool = True,
    force_mute: bool | None = None,
    scene_caption: bool | None = None,
    shot_detect: bool | None = None,
    on_stage: StageCallback | None = None,
) -> BuildResult:
    def report(stage: str) -> None:
        print(f"[index] stage={stage}")
        if on_stage is not None:
            on_stage(stage)

    settings = load_settings()
    # A rebuild stays unpublished until every vector channel is present and
    # this metadata file is atomically replaced.
    generation = f"g{uuid4().hex}"
    want_scenes = settings.scene_caption if scene_caption is None else bool(scene_caption)
    want_shots = settings.shot_detect if shot_detect is None else bool(shot_detect)
    timings = StageTimings()
    started = perf_counter()

    report("ingest")
    t0 = perf_counter()
    ingest_info = ingest(source, dest_id=video_id)
    timings.ingest_s = perf_counter() - t0
    vid = ingest_info.video_id
    media = Path(ingest_info.path)
    print(f"[index] video_id={vid} path={media} backend=opensearch")

    shots: list[ShotSpan] = []
    frames = []
    if want_shots:
        report("shots")
        if should_evict():
            # TransNetV2 is 7.6M params but still wants activation memory and a
            # cuDNN workspace per window. A process that has served a search is
            # holding both 2B models -- the captioner for query expansion, the
            # encoder for the query vector -- and neither is needed again until
            # the visual stage, so on a small card they only starve this one.
            release_captioner()
            release_visual_encoder()
        t0 = perf_counter()
        try:
            shots, motion_track = detect_shots(
                media,
                threshold=settings.shot_threshold,
                duration_s=float(ingest_info.duration_s or 0.0),
            )
            frames = extract_shot_keyframes(
                media,
                vid,
                shots,
                max_gap_s=settings.shot_max_gap_s,
                motion_track=motion_track,
            )
        except Exception as exc:
            print(f"[shots] TransNetV2 failed ({exc}); falling back to 2 fps keyframes")
            shots = []
            frames = []
        timings.shots_s = perf_counter() - t0
    if not frames:
        report("frames")
        t0 = perf_counter()
        frames = extract_keyframes(media, vid)
        timings.frames_s = perf_counter() - t0

    run_asr = should_run_asr(media, force_mute=force_mute)
    segments = []
    if run_asr:
        report("asr")
        t0 = perf_counter()
        segments = transcribe(media)
        timings.asr_s = perf_counter() - t0
    else:
        report("asr_skip")

    ocr_obs = []
    slide_heavy = False
    if enable_ocr:
        report("ocr")
        t0 = perf_counter()
        try:
            ocr_obs = ocr_frames(frames)
            slide_heavy = is_slide_heavy(ocr_obs, frames)
        except Exception as exc:
            print(f"[ocr] skipped: {exc}")
        timings.ocr_s = perf_counter() - t0

    report("visual")
    t0 = perf_counter()
    if should_evict():
        # The previous video's caption stage, or a query expansion in this
        # process if it is also serving the UI, may still be holding the
        # captioner on a card too small for both 2B models.
        release_captioner()
    frame_vecs = embed_clip_images([frame.path for frame in frames], settings=settings)
    timings.clip_embed_s = perf_counter() - t0

    tdim = text_dim(settings=settings)
    transcript_vecs = np.zeros((0, tdim), dtype=np.float32)
    if segments:
        report("bge_transcript")
        t0 = perf_counter()
        transcript_vecs = embed_text_passages([seg.text for seg in segments], settings=settings)
        timings.bge_transcript_s = perf_counter() - t0

    ocr_vecs = np.zeros((0, tdim), dtype=np.float32)
    if ocr_obs:
        report("bge_ocr")
        t0 = perf_counter()
        ocr_vecs = embed_text_passages([row.text for row in ocr_obs], settings=settings)
        timings.bge_ocr_s = perf_counter() - t0

    duration_s = float(ingest_info.duration_s or _duration(media, frames, segments))
    scenes: list[SceneCard] = []
    scene_vecs = np.zeros((0, tdim), dtype=np.float32)
    caption_model: str | None = None
    if want_scenes:
        report("scenes")
        t0 = perf_counter()
        if shots:
            drafts = drafts_from_shots(frames, shots)
            if not drafts:
                drafts = cluster_scenes(frames, frame_vecs, duration_s=duration_s)
        else:
            drafts = cluster_scenes(frames, frame_vecs, duration_s=duration_s)
        timings.scenes_s = perf_counter() - t0
        report("caption")
        t0 = perf_counter()
        if should_evict():
            # Scene drafting is done with frame_vecs (plain numpy), so the
            # encoder itself is no longer needed for this video.
            release_visual_encoder()
        try:
            scenes = caption_scenes(
                frames,
                drafts,
                duration_s=duration_s,
                model_name=settings.scene_caption_model,
            )
            caption_model = next((s.model for s in scenes if s.model), settings.scene_caption_model)
        except Exception as exc:
            print(f"[caption] skipped: {exc}")
            scenes = []
        timings.caption_s = perf_counter() - t0
        if scenes:
            report("bge_scene")
            t0 = perf_counter()
            scene_vecs = embed_text_passages(
                [scene.passage_text() for scene in scenes], settings=settings
            )
            timings.bge_scene_s = perf_counter() - t0

    report("save")
    vdim = int(frame_vecs.shape[1]) if frame_vecs.size else visual_dim(settings=settings)
    _write_stores(
        vid,
        generation=generation,
        settings=settings,
        frames=frames,
        frame_vecs=frame_vecs,
        segments=segments,
        transcript_vecs=transcript_vecs,
        ocr_obs=ocr_obs,
        ocr_vecs=ocr_vecs,
        scenes=scenes,
        scene_vecs=scene_vecs,
    )
    # Do not persist dense matrices. Drop local copies after the store write.
    del frame_vecs, transcript_vecs, ocr_vecs, scene_vecs

    meta = IndexMeta(
        video_id=vid,
        source=source,
        title=ingest_info.title,
        duration_s=duration_s,
        video_path=str(media),
        frames=frames,
        transcript=segments,
        ocr=ocr_obs,
        scenes=scenes,
        shots=shots,
        slide_heavy=slide_heavy,
        channels=ChannelFlags(
            visual=True,
            transcript=bool(segments),
            ocr=bool(ocr_obs),
            asr=bool(segments),
            scene=bool(scenes),
        ),
        models=ModelCard(
            clip=visual_model_label(settings=settings),
            asr="faster-whisper small int8" if segments else "skipped (no audio track)",
            text=text_model_label(),
            text_dim=text_dim(settings=settings),
            ocr="rapidocr-onnxruntime" if ocr_obs else None,
            caption=caption_model,
            shot="TransNetV2" if shots else None,
            visual_dim=vdim,
            visual_encoder=encoder_kind(settings=settings),
        ),
        index_generation=generation,
    )
    write_json_atomic(meta_path(vid), meta.model_dump_json(indent=2))
    try:
        delete_other_generations(vid, generation, settings=settings)
    except Exception as exc:
        # Publishing succeeded; deferred cleanup cannot make the new index
        # visible incorrectly, so leave it for a later rebuild.
        print(f"[index] old-generation cleanup deferred: {exc}")
    timings.total_s = perf_counter() - started
    write_json_atomic(timings_path(vid), timings.model_dump_json(indent=2))
    print(f"[index] wrote OpenSearch collections and {meta_path(vid)}")
    for name, value in timings.model_dump().items():
        print(f"[timing] {name}: {value:.2f}s")
    report("done")
    return BuildResult(video_id=vid, meta=meta, timings=timings)


def load_timings(video_id: str) -> StageTimings | None:
    path = timings_path(video_id)
    if not path.exists():
        return None
    return StageTimings.model_validate_json(path.read_text(encoding="utf-8"))


_TIMING_LABELS = (
    ("ingest_s", "ingest"),
    ("frames_s", "keyframes"),
    ("shots_s", "TransNetV2 shots"),
    ("asr_s", "whisper"),
    ("ocr_s", "ocr"),
    ("scenes_s", "scenes"),
    ("caption_s", "caption"),
    ("clip_embed_s", "visual embed"),
    ("bge_transcript_s", "text transcript"),
    ("bge_ocr_s", "text OCR"),
    ("bge_scene_s", "text scene"),
    ("total_s", "total"),
)


def format_timings(timings: StageTimings) -> str:
    parts: list[str] = []
    for field, label in _TIMING_LABELS:
        value = getattr(timings, field)
        if value or field == "total_s":
            parts.append(f"{label} {value:.1f}s")
    return " · ".join(parts)


def load_index(video_id: str) -> dict[str, Any]:
    """Open the shared OpenSearch collections, scoped to this video, plus local metadata."""
    meta = IndexMeta.model_validate_json(read_json_text(meta_path(video_id)))
    settings = load_settings()
    visual = open_store(
        video_id,
        "visual",
        dim=visual_dim(meta=meta),
        settings=settings,
        create=False,
        generation=meta.index_generation,
    )
    # Store dim is source of truth (meta can lag after a crashed Qwen write).
    if getattr(visual, "dim", 0) and meta.models.visual_dim != visual.dim:
        meta.models.visual_dim = int(visual.dim)
        stale_clip_label = visual.dim != 512 and (
            "open_clip" in (meta.models.clip or "").lower()
            or "vit-b-32" in (meta.models.clip or "").lower()
        )
        # Only adopt the configured label when it actually produces this width;
        # otherwise we would stamp (say) a WeMM name onto a Qwen-written store.
        if stale_clip_label and visual_dim(settings=settings) == int(visual.dim):
            meta.models.clip = visual_model_label(settings=settings)
        if not meta.models.visual_encoder:
            meta.models.visual_encoder = encoder_kind(meta=meta)
        write_json_atomic(meta_path(video_id), meta.model_dump_json(indent=2))
        print(f"[index] healed {video_id} meta visual_dim={visual.dim}")
    text = None
    text_probe = open_store(
        video_id,
        "text",
        dim=text_dim(meta=meta),
        settings=settings,
        create=False,
        generation=meta.index_generation,
    )
    if text_probe.ntotal() > 0:
        text = text_probe
        mapped = int(text.dim or 0)
        if mapped and meta.models.text_dim != mapped:
            meta.models.text_dim = mapped
            write_json_atomic(meta_path(video_id), meta.model_dump_json(indent=2))
            print(f"[index] healed {video_id} meta text_dim={mapped}")
    return {
        "video_id": video_id,
        "meta": meta,
        "visual": visual,
        "text": text,
    }


def _write_stores(
    vid: str,
    *,
    generation: str,
    settings,
    frames,
    frame_vecs: np.ndarray,
    segments,
    transcript_vecs: np.ndarray,
    ocr_obs,
    ocr_vecs: np.ndarray,
    scenes=(),
    scene_vecs: np.ndarray | None = None,
) -> None:
    vdim = int(frame_vecs.shape[1]) if frame_vecs.size else visual_dim(settings=settings)
    visual = open_store(
        vid, "visual", dim=vdim, settings=settings, create=True, generation=generation
    )
    try:
        visual.add(
            frame_vecs,
            [
                VectorPayload(
                    vector_id=0,
                    asset_id=vid,
                    t_s=frame.t_s,
                    modality="video_frame",
                    path=frame.path,
                    frame_index=frame.index,
                    shot_id=frame.shot_id,
                    motion_energy=frame.motion_energy,
                    flash_energy=frame.flash_energy,
                )
                for frame in frames
            ],
        )
    finally:
        visual.close()

    text_payloads: list[VectorPayload] = []
    text_rows: list[np.ndarray] = []
    if segments and transcript_vecs.shape[0]:
        for seg, vec in zip(segments, transcript_vecs, strict=False):
            text_payloads.append(
                VectorPayload(
                    vector_id=0,
                    asset_id=vid,
                    t_s=seg.start_s,
                    modality="transcript",
                    path="",
                    text=seg.text,
                    frame_index=None,
                )
            )
            text_rows.append(vec)
    if ocr_obs and ocr_vecs.shape[0]:
        for row, vec in zip(ocr_obs, ocr_vecs, strict=False):
            text_payloads.append(
                VectorPayload(
                    vector_id=0,
                    asset_id=vid,
                    t_s=row.t_s,
                    modality="ocr",
                    path="",
                    text=row.text,
                    frame_index=row.frame_index,
                )
            )
            text_rows.append(vec)
    if scenes and scene_vecs is not None and scene_vecs.shape[0]:
        for scene, vec in zip(scenes, scene_vecs, strict=False):
            text_payloads.append(
                VectorPayload(
                    vector_id=0,
                    asset_id=vid,
                    t_s=scene.start_s,
                    modality="scene",
                    path=scene.path,
                    text=scene.passage_text(),
                    frame_index=scene.frame_index,
                )
            )
            text_rows.append(vec)
    if not text_payloads:
        return
    tdim = int(text_rows[0].shape[0]) if text_rows else text_dim(settings=settings)
    text = open_store(
        vid, "text", dim=tdim, settings=settings, create=True, generation=generation
    )
    try:
        text.add(np.vstack(text_rows), text_payloads)
    finally:
        text.close()


def resolve_video_id(source: str | None, video_id: str | None) -> str:
    if video_id:
        try:
            return validate_video_id(video_id)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    if source:
        return extract_video_id(source)
    raise SystemExit("Pass --video-id, --source, or set VVS_VIDEO_ID / VVS_SOURCE.")


def index_exists(video_id: str) -> bool:
    if not meta_path(video_id).exists():
        return False
    try:
        meta = IndexMeta.model_validate_json(read_json_text(meta_path(video_id)))
    except (OSError, ValueError):
        return False
    return visual_store_exists(video_id, generation=meta.index_generation)


def ensure_index(source: str | None, video_id: str | None, *, rebuild: bool = False) -> str:
    vid = resolve_video_id(source, video_id)
    if rebuild or not index_exists(vid):
        if not source:
            local = video_path(vid)
            if not local.exists():
                raise SystemExit(f"No index for {vid} and no --source to build from.")
            source = str(local)
        build_index(source, video_id=vid)
    return vid


def _duration(media: Path, frames, segments) -> float:
    try:
        import av

        container = av.open(str(media))
        stream = container.streams.video[0]
        if stream.duration:
            value = float(stream.duration * stream.time_base)
            container.close()
            return value
        container.close()
    except Exception:
        pass
    ends = [float(f.t_s) for f in frames] + [float(s.end_s) for s in segments]
    return max(ends) if ends else 0.0


def delete_index(video_id: str) -> bool:
    """Remove one asset's rows from the shared OpenSearch index, plus its local directory.

    Guarded so a stray id can never escape the data root: the target must be a
    direct child of it. Returns False when there was nothing to delete.
    """
    import shutil

    root = DATA_ROOT.resolve()
    target = video_dir(video_id).resolve()
    if target == root or target.parent != root:
        raise SystemExit(f"refusing to delete {target}: not a direct child of {root}")
    delete_video(video_id)
    if not target.exists():
        return False
    shutil.rmtree(target)
    print(f"[index] deleted {target}")
    return True


def list_indexed() -> list[str]:
    if not DATA_ROOT.exists():
        return []
    found: list[str] = []
    for path in DATA_ROOT.iterdir():
        if not path.is_dir():
            continue
        if index_exists(path.name):
            found.append(path.name)
    return sorted(found)


def embed_counts(video_id: str) -> dict[str, Any]:
    """How many vectors this index actually stored (visual = stills)."""
    meta = IndexMeta.model_validate_json(meta_path(video_id).read_text(encoding="utf-8"))
    visual = len(meta.frames)
    text = len(meta.transcript) + len(meta.ocr) + len(meta.scenes)
    return {
        "video_id": video_id,
        "duration_s": meta.duration_s,
        "visual_embeddings": visual,
        "shots": len(meta.shots),
        "text_embeddings": text,
        "transcript": len(meta.transcript),
        "ocr": len(meta.ocr),
        "scenes": len(meta.scenes),
        "shot_detect": bool(meta.shots),
    }


def format_embed_counts(row: dict[str, Any]) -> str:
    mode = "TransNetV2" if row["shot_detect"] else "2fps"
    return (
        f"{row['video_id']:24}  {mode:10}  "
        f"visual={row['visual_embeddings']:<5}  shots={row['shots']:<4}  "
        f"text={row['text_embeddings']:<5}  {row['duration_s']:.0f}s"
    )


def format_embed_compare(left: dict[str, Any], right: dict[str, Any]) -> str:
    """left = baseline (no shots), right = TransNetV2."""
    a, b = left["visual_embeddings"], right["visual_embeddings"]
    if a <= 0:
        saved = "n/a"
    else:
        saved = f"{a - b} fewer visual embeddings ({(a - b) / a * 100:.0f}% less)"
    return (
        f"{format_embed_counts(left)}\n"
        f"{format_embed_counts(right)}\n"
        f"compare visual: {a} → {b}  ·  {saved}"
    )
