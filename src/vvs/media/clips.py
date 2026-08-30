"""Cut browser-playable MP4 clips covering the supporting transcript window."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from vvs.models import ClipWindow, ExtractedClip, Hit, IndexMeta, TranscriptSegment
from vvs.paths import clips_dir, video_path


# Prefer the full Whisper span. Cap only extremely long segments, always
# from the transcript start (never mid-sentence / around the keyframe).
MAX_CLIP_S = 60.0
FALLBACK_PAD_S = 2.0
CACHE_TAG = "asr"
MUTE_CACHE_TAG = "vis"


def supporting_segments(
    transcript: list[TranscriptSegment],
    hit: Hit,
) -> list[TranscriptSegment]:
    """Whisper segments that justify this hit, in time order.

    Uses the search-assigned quote window when present, otherwise the
    segment(s) that contain the keyframe. Adjacent segments are joined
    only when they actually overlap that window (not merely touch).
    """
    if not transcript:
        return []

    window: tuple[float, float] | None = None
    if hit.quote_start_s is not None and hit.quote_end_s is not None:
        if float(hit.quote_end_s) > float(hit.quote_start_s):
            window = (float(hit.quote_start_s), float(hit.quote_end_s))
    if window is None:
        t = float(hit.t_s)
        for seg in transcript:
            if seg.start_s <= t < seg.end_s:
                window = (seg.start_s, seg.end_s)
                break
        else:
            for seg in transcript:
                if seg.start_s <= t <= seg.end_s:
                    window = (seg.start_s, seg.end_s)
                    break
    if window is None:
        return []

    w0, w1 = window
    picked = [seg for seg in transcript if _overlaps(seg.start_s, seg.end_s, w0, w1)]
    picked.sort(key=lambda seg: (seg.start_s, seg.end_s, seg.index))
    return picked


def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def clip_window(
    hit: Hit,
    duration_s: float,
    transcript: list[TranscriptSegment] | None = None,
    *,
    max_s: float = MAX_CLIP_S,
) -> ClipWindow:
    """Play the supporting transcript from its first word through the segment end."""
    duration_s = max(0.0, float(duration_s))
    segs = supporting_segments(transcript or [], hit)
    if segs:
        start = min(seg.start_s for seg in segs)
        end = max(seg.end_s for seg in segs)
    elif hit.quote_start_s is not None and hit.quote_end_s is not None and hit.quote_end_s > hit.quote_start_s:
        start = float(hit.quote_start_s)
        end = float(hit.quote_end_s)
    else:
        t = float(hit.t_s)
        start = t - FALLBACK_PAD_S
        end = t + FALLBACK_PAD_S

    if end - start > max_s:
        end = start + max_s

    start = max(0.0, start)
    end = min(duration_s, end) if duration_s else end
    if end <= start:
        leftover = duration_s - start if duration_s else 0.5
        end = start + min(0.5, leftover if leftover > 0 else 0.5)
        if duration_s:
            end = min(duration_s, end)
            start = max(0.0, min(start, end))
    return ClipWindow(start_s=round(start, 3), end_s=round(end, 3))


def quote_for_window(
    transcript: list[TranscriptSegment],
    start_s: float,
    end_s: float,
) -> tuple[str, float | None, float | None]:
    """Join every spoken segment that overlaps [start_s, end_s)."""
    parts: list[str] = []
    q_start: float | None = None
    q_end: float | None = None
    for seg in transcript:
        if seg.end_s <= start_s or seg.start_s >= end_s:
            continue
        text = seg.text.strip()
        if not text:
            continue
        parts.append(text)
        q_start = seg.start_s if q_start is None else min(q_start, seg.start_s)
        q_end = seg.end_s if q_end is None else max(q_end, seg.end_s)
    return " ".join(parts), q_start, q_end


def clip_filename(window: ClipWindow, tag: str = CACHE_TAG) -> str:
    return f"clip_{tag}_{_centis(window.start_s)}_{_centis(window.end_s)}.mp4"


def attach_clips(
    video_id: str,
    hits: list[Hit],
    *,
    duration_s: float,
    source_video: Path | None = None,
    top_k: int = 3,
    transcript: list[TranscriptSegment] | None = None,
) -> list[Hit]:
    """Extract (or reuse) a clip for each of the top hits and return updated models."""
    source = Path(source_video) if source_video else video_path(video_id)
    out_dir = clips_dir(video_id)
    _purge_invalid_clips(out_dir)
    tag = CACHE_TAG if transcript else MUTE_CACHE_TAG
    attached: list[Hit] = []
    for hit in hits[:top_k]:
        window = clip_window(hit, duration_s, transcript)
        dest = out_dir / clip_filename(window, tag)
        extracted = extract_clip(source, dest, window.start_s, window.end_s)
        actual = ClipWindow(start_s=extracted.start_s, end_s=extracted.end_s)
        final_path = Path(extracted.path)
        keyed = out_dir / clip_filename(actual, tag)
        if final_path.resolve() != keyed.resolve():
            if keyed.exists() and _usable_clip(keyed, expect_s=actual.end_s - actual.start_s):
                if final_path.exists() and final_path.resolve() != keyed.resolve():
                    final_path.unlink()
                final_path = keyed
            else:
                final_path.replace(keyed)
                final_path = keyed
        # Quote the requested transcript window so the caption matches the
        # full supporting speech, not a few words around the keyframe.
        quote, q_start, q_end = quote_for_window(
            transcript or [],
            window.start_s,
            window.end_s,
        )
        if not quote and hit.quote:
            quote, q_start, q_end = hit.quote, window.start_s, window.end_s
        attached.append(
            hit.model_copy(
                update={
                    "clip_path": str(final_path.resolve()),
                    "clip_start_s": actual.start_s,
                    "clip_end_s": actual.end_s,
                    "quote": quote,
                    "quote_start_s": q_start,
                    "quote_end_s": q_end,
                }
            )
        )
    return attached


def attach_corpus_clips(
    hits: list[Hit],
    *,
    top_k: int = 3,
    metas: dict[str, IndexMeta] | None = None,
) -> list[Hit]:
    """Cut playable clips from each hit's own source video."""
    chosen = [hit for hit in hits[:top_k] if hit.video_id]
    if not chosen:
        return hits
    by_vid: dict[str, list[Hit]] = {}
    for hit in chosen:
        by_vid.setdefault(hit.video_id, []).append(hit)
    updated: dict[tuple[str, int, float], Hit] = {}
    for vid, group in by_vid.items():
        meta = (metas or {}).get(vid)
        if meta is None:
            from vvs.retrieve.index import load_index

            meta = load_index(vid)["meta"]
        done = attach_clips(
            vid,
            group,
            duration_s=meta.duration_s,
            source_video=Path(meta.video_path),
            top_k=len(group),
            transcript=meta.transcript,
        )
        for old, new in zip(group, done, strict=False):
            updated[(vid, old.frame_index, float(old.t_s))] = new
    return [
        updated.get((hit.video_id, hit.frame_index, float(hit.t_s)), hit) for hit in hits
    ]


def extract_clip(source: Path, dest: Path, start_s: float, end_s: float) -> ExtractedClip:
    """Write `dest` as an H.264 MP4 covering [start_s, end_s). Reuse a complete file."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    expect = max(0.0, end_s - start_s)
    if _usable_clip(dest, expect_s=expect):
        actual_start, actual_end = _probe_written_span(dest, start_s, end_s)
        return ExtractedClip(path=str(dest.resolve()), start_s=actual_start, end_s=actual_end)

    if end_s <= start_s:
        raise ValueError(f"Invalid clip window {start_s:.3f}–{end_s:.3f}")
    if not source.exists():
        raise FileNotFoundError(f"Source video not found: {source}")

    tmp = dest.with_name(dest.stem + ".partial.mp4")
    if tmp.exists():
        tmp.unlink()
    try:
        written = _reencode_clip(source, tmp, start_s, end_s)
        if not _usable_clip(tmp, expect_s=written.end_s - written.start_s):
            raise RuntimeError(f"Clip write produced an unusable file: {tmp}")
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return ExtractedClip(path=str(dest.resolve()), start_s=written.start_s, end_s=written.end_s)


def _centis(t: float) -> str:
    return f"{max(0, int(round(float(t) * 100))):06d}"


def _purge_invalid_clips(directory: Path) -> None:
    if not directory.exists():
        return
    for path in directory.glob("clip_*.mp4"):
        stale = _is_stale_cache_name(path)
        if stale or not _usable_clip(path):
            try:
                path.unlink()
            except OSError:
                pass


def _is_stale_cache_name(path: Path) -> bool:
    """Old ~4s caches were clip_SSSSSS_EEEEEE.mp4 without the asr tag."""
    parts = path.stem.split("_")
    return not (len(parts) >= 4 and parts[0] == "clip" and parts[1] == CACHE_TAG)


def _usable_clip(path: Path, *, expect_s: float | None = None) -> bool:
    if not (path.exists() and path.is_file() and path.stat().st_size > 1024):
        return False
    try:
        import av

        container = av.open(str(path))
        try:
            if not container.streams.video:
                return False
            video = container.streams.video[0]
            if video.codec_context.name != "h264":
                return False
            times: list[float] = []
            for i, frame in enumerate(container.decode(video)):
                t = _frame_time(frame)
                if t is not None:
                    times.append(t)
                if i >= 12:
                    break
            if len(times) < 2 or times[-1] <= times[0] + 1e-3:
                return False
            if expect_s is not None and container.duration:
                dur = container.duration / 1_000_000
                if abs(dur - expect_s) > 0.75:
                    return False
            return True
        finally:
            container.close()
    except Exception:
        return False


def _probe_written_span(path: Path, start_s: float, end_s: float) -> tuple[float, float]:
    """Map the file's first/last timestamps back onto the source timeline."""
    try:
        import av

        container = av.open(str(path))
        try:
            first: float | None = None
            video = container.streams.video[0]
            for frame in container.decode(video):
                t = _frame_time(frame)
                if t is None:
                    continue
                first = t
                break
            if first is None:
                return round(start_s, 3), round(end_s, 3)
            actual_start = start_s
            if first > 0.5:
                actual_start = start_s + first
            return round(actual_start, 3), round(end_s, 3)
        finally:
            container.close()
    except Exception:
        return round(start_s, 3), round(end_s, 3)


def _frame_time(frame) -> float | None:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is not None and frame.time_base is not None:
        return float(frame.pts * frame.time_base)
    return None


def _output_fps(stream) -> Fraction:
    for candidate in (getattr(stream, "guessed_rate", None), getattr(stream, "base_rate", None)):
        if candidate is None:
            continue
        value = float(candidate)
        if 1.0 <= value <= 120.0:
            return Fraction(candidate).limit_denominator(1001)
    avg = getattr(stream, "average_rate", None)
    if avg is not None and 1.0 <= float(avg) <= 120.0:
        return Fraction(avg).limit_denominator(1001)
    return Fraction(30, 1)


def _reencode_clip(source: Path, dest: Path, start_s: float, end_s: float) -> ExtractedClip:
    """Decode/encode a precise H.264 (+ AAC) window so A/V share the same [start, end)."""
    import av
    from av.audio.resampler import AudioResampler

    video_tb = Fraction(1, 90_000)
    fps = Fraction(30, 1)
    inp = av.open(str(source))
    out = av.open(str(dest), mode="w", options={"movflags": "+faststart"})
    first_video_t: float | None = None
    last_video_t: float | None = None
    last_audio_end: float | None = None
    try:
        in_video = inp.streams.video[0]
        in_video.thread_type = "AUTO"
        fps = _output_fps(in_video)
        out_video = out.add_stream("libx264", rate=fps)
        out_video.width = in_video.width
        out_video.height = in_video.height
        out_video.pix_fmt = "yuv420p"
        out_video.time_base = video_tb
        out_video.options = {"preset": "ultrafast", "crf": "28"}

        in_audio = inp.streams.audio[0] if inp.streams.audio else None
        out_audio = None
        resampler = None
        audio_tb = None
        if in_audio is not None:
            rate = int(in_audio.rate or 44100)
            out_audio = out.add_stream("aac", rate=rate)
            audio_tb = Fraction(1, rate)
            out_audio.time_base = audio_tb
            resampler = AudioResampler(format="fltp", layout="stereo", rate=rate)

        seek_s = max(0.0, start_s - 1.0)
        inp.seek(int(seek_s * av.time_base))

        streams = [in_video] + ([in_audio] if in_audio is not None else [])
        video_done = False
        audio_done = in_audio is None
        for frame in inp.decode(*streams):
            t = _frame_time(frame)
            if t is None:
                continue
            if isinstance(frame, av.VideoFrame):
                if t + 1e-4 < start_s:
                    continue
                if t >= end_s:
                    video_done = True
                    if audio_done:
                        break
                    continue
                if first_video_t is None:
                    first_video_t = t
                last_video_t = t
                rel = t - start_s
                frame.pts = max(0, int(round(rel / float(video_tb))))
                frame.time_base = video_tb
                for packet in out_video.encode(frame):
                    out.mux(packet)
            elif isinstance(frame, av.AudioFrame) and out_audio is not None and resampler is not None:
                rate = int(frame.sample_rate or out_audio.rate or 44100)
                samples = int(frame.samples)
                dur = samples / rate if rate else 0.0
                if t + dur <= start_s:
                    continue
                if t >= end_s:
                    audio_done = True
                    if video_done:
                        break
                    continue
                sliced, slice_t = _slice_audio(frame, t, start_s, end_s)
                if sliced is None:
                    continue
                last_audio_end = max(last_audio_end or 0.0, slice_t[1])
                for resampled in resampler.resample(sliced):
                    rel = slice_t[0] - start_s
                    resampled.pts = max(0, int(round(rel / float(audio_tb))))
                    resampled.time_base = audio_tb
                    for packet in out_audio.encode(resampled):
                        out.mux(packet)
                    slice_t = (slice_t[0] + resampled.samples / rate, slice_t[1])

        if resampler is not None and out_audio is not None and audio_tb is not None:
            for resampled in resampler.resample(None):
                rel = (last_audio_end - start_s) if last_audio_end is not None else 0.0
                resampled.pts = max(0, int(round(rel / float(audio_tb))))
                resampled.time_base = audio_tb
                for packet in out_audio.encode(resampled):
                    out.mux(packet)
        for packet in out_video.encode():
            out.mux(packet)
        if out_audio is not None:
            for packet in out_audio.encode():
                out.mux(packet)
    finally:
        out.close()
        inp.close()

    # File timeline origin is start_s (PTS 0). Use that as the label unless the
    # first picture landed far from the requested start (decoder/keyframe miss).
    actual_start = start_s
    if first_video_t is not None and first_video_t - start_s > 0.5:
        actual_start = first_video_t
    frame_dt = 1.0 / float(fps) if float(fps) else 1.0 / 30.0
    actual_end = end_s
    tails = [t for t in (last_video_t + frame_dt if last_video_t is not None else None, last_audio_end) if t]
    if tails:
        actual_end = min(end_s, max(tails))
    if actual_end <= actual_start:
        actual_end = min(end_s, actual_start + frame_dt)
    return ExtractedClip(
        path=str(dest.resolve()),
        start_s=round(actual_start, 3),
        end_s=round(actual_end, 3),
    )


def _slice_audio(frame, t: float, start_s: float, end_s: float):
    """Trim an audio frame so only samples overlapping [start_s, end_s) remain."""
    import av

    rate = int(frame.sample_rate or 0)
    samples = int(frame.samples)
    if rate <= 0 or samples <= 0:
        return None, (t, t)
    dur = samples / rate
    frame_end = t + dur
    if frame_end <= start_s or t >= end_s:
        return None, (t, t)
    skip = 0
    if t < start_s:
        skip = min(samples, max(0, int(round((start_s - t) * rate))))
    take_end = samples
    if frame_end > end_s:
        take_end = min(samples, max(skip, int(round((end_s - t) * rate))))
    if take_end <= skip:
        return None, (t, t)
    new_t = t + skip / rate
    new_end = t + take_end / rate
    if skip == 0 and take_end == samples:
        return frame, (new_t, new_end)
    array = frame.to_ndarray()
    sliced = array[:, skip:take_end] if array.ndim == 2 else array[skip:take_end]
    layout = frame.layout.name if frame.layout is not None else "stereo"
    fmt = frame.format.name if frame.format is not None else "fltp"
    trimmed = av.AudioFrame.from_ndarray(sliced, format=fmt, layout=layout)
    trimmed.sample_rate = rate
    trimmed.pts = None
    trimmed.time_base = frame.time_base
    return trimmed, (new_t, new_end)
