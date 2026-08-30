"""Lightweight media probes. Sphere assets are expected to have no audio track."""

from __future__ import annotations

from pathlib import Path


def has_audio_stream(path: Path | str) -> bool:
    """True if the container exposes at least one audio stream."""
    import av

    container = av.open(str(path))
    try:
        return any(stream.type == "audio" for stream in container.streams)
    finally:
        container.close()
