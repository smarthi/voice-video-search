"""Ingest a YouTube URL or a local video file into data/<video_id>/."""

from vvs.ingest.service import extract_video_id, ingest

__all__ = ["extract_video_id", "ingest"]
