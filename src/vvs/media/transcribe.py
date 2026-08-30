"""Timestamped speech-to-text with faster-whisper on CPU."""

from __future__ import annotations

from pathlib import Path

from vvs.models import TranscriptSegment


_WHISPER = None


def get_whisper(
    model_size: str = "small",
    compute_type: str = "int8",
    cpu_threads: int = 16,
):
    global _WHISPER
    if _WHISPER is None:
        from faster_whisper import WhisperModel

        print(f"[asr] loading faster-whisper {model_size} ({compute_type}) on CPU")
        _WHISPER = WhisperModel(
            model_size,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
    return _WHISPER


def transcribe_video(
    video_file: Path,
    *,
    language: str = "en",
    vad_filter: bool = True,
) -> list[TranscriptSegment]:
    model = get_whisper()
    segments, _info = model.transcribe(
        str(video_file),
        language=language,
        vad_filter=vad_filter,
        vad_parameters={"min_silence_duration_ms": 400},
        beam_size=5,
        word_timestamps=False,
        condition_on_previous_text=False,
    )
    out: list[TranscriptSegment] = []
    for i, seg in enumerate(segments):
        text = (seg.text or "").strip()
        if not text:
            continue
        out.append(
            TranscriptSegment(
                index=len(out),
                start_s=round(float(seg.start), 3),
                end_s=round(float(seg.end), 3),
                text=text,
            )
        )
        if (i + 1) % 25 == 0:
            print(f"[asr] {i + 1} segments… last @ {seg.end:.1f}s")
    print(f"[asr] {len(out)} transcript segments")
    return out


def transcribe_audio_file(audio_file: Path, *, language: str = "en") -> str:
    model = get_whisper()
    segments, _info = model.transcribe(
        str(audio_file),
        language=language,
        vad_filter=True,
        beam_size=1,
        condition_on_previous_text=False,
    )
    parts = [seg.text.strip() for seg in segments if (seg.text or "").strip()]
    return " ".join(parts).strip()


def try_batched_transcribe(
    video_file: Path,
    *,
    language: str = "en",
    batch_size: int = 8,
) -> list[TranscriptSegment] | None:
    try:
        from faster_whisper import BatchedInferencePipeline
    except Exception:
        return None
    model = get_whisper()
    pipeline = BatchedInferencePipeline(model=model)
    segments, _info = pipeline.transcribe(
        str(video_file),
        language=language,
        batch_size=batch_size,
        vad_filter=True,
        beam_size=5,
    )
    out: list[TranscriptSegment] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        out.append(
            TranscriptSegment(
                index=len(out),
                start_s=round(float(seg.start), 3),
                end_s=round(float(seg.end), 3),
                text=text,
            )
        )
    print(f"[asr] batched {len(out)} transcript segments")
    return out


def transcribe(video_file: Path, language: str = "en", batch_size: int = 8) -> list[TranscriptSegment]:
    batched = try_batched_transcribe(video_file, language=language, batch_size=batch_size)
    if batched is not None:
        return batched
    return transcribe_video(video_file, language=language)
