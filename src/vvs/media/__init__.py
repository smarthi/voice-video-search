"""Keyframe sampling, clip extraction, ASR, OCR, and optional scene captions."""

from vvs.media.caption import caption_scenes
from vvs.media.clips import attach_clips, extract_clip, quote_for_window
from vvs.media.frames import extract_keyframes
from vvs.media.ocr import is_slide_heavy, ocr_frames
from vvs.media.probe import has_audio_stream
from vvs.media.scenes import cluster_scenes, scene_covering
from vvs.media.shots import detect_shots, extract_shot_keyframes, shot_covering
from vvs.media.transcribe import transcribe, transcribe_audio_file

__all__ = [
    "attach_clips",
    "caption_scenes",
    "cluster_scenes",
    "detect_shots",
    "extract_clip",
    "extract_keyframes",
    "extract_shot_keyframes",
    "has_audio_stream",
    "is_slide_heavy",
    "ocr_frames",
    "quote_for_window",
    "scene_covering",
    "shot_covering",
    "transcribe",
    "transcribe_audio_file",
]
