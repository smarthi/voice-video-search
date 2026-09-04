"""Resolve a source string and place video.mp4 under data/<id>/."""

from __future__ import annotations

from pathlib import Path

from vvs.ingest.local import ingest_local, local_slug
from vvs.ingest.youtube import download_youtube, parse_youtube_id
from vvs.models import IngestInfo
from vvs.paths import video_dir, video_path, write_json_atomic


def extract_video_id(source: str) -> str:
    """Return a stable 11-char YouTube id, or a filesystem-safe local slug."""
    youtube_id = parse_youtube_id(source)
    if youtube_id:
        return youtube_id

    local = Path(source).expanduser().resolve()
    if local.exists():
        return local_slug(local.stem)
    raise ValueError(f"Cannot determine video id from source: {source}")


def ingest(source: str, dest_id: str | None = None) -> IngestInfo:
    """Place `video.mp4` under data/<id>/ and return ingest metadata."""
    video_id = dest_id or extract_video_id(source)
    out_dir = video_dir(video_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = video_path(video_id)
    cached = out_dir / "ingest.json"

    local = Path(source).expanduser()
    if local.exists() and local.is_file():
        info = ingest_local(local, dest, video_id)
        write_json_atomic(cached, info.model_dump_json(indent=2))
        return info

    if dest.exists() and dest.stat().st_size > 0 and cached.exists():
        return IngestInfo.model_validate_json(cached.read_text(encoding="utf-8"))

    info = download_youtube(source, dest).model_copy(update={"video_id": video_id, "path": str(dest)})
    write_json_atomic(cached, info.model_dump_json(indent=2))
    return info
