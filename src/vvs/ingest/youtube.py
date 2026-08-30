"""YouTube id parsing and yt-dlp download."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from vvs.models import IngestInfo


YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def parse_youtube_id(source: str) -> str | None:
    """Return an 11-char YouTube id from a bare id or URL, else None."""
    if YOUTUBE_ID_RE.match(source):
        return source

    parsed = urlparse(source)
    host = (parsed.hostname or "").lower()
    if host in {"youtu.be"}:
        slug = parsed.path.lstrip("/").split("/")[0]
        if YOUTUBE_ID_RE.match(slug):
            return slug
    if "youtube.com" in host:
        qs = parse_qs(parsed.query)
        if "v" in qs and YOUTUBE_ID_RE.match(qs["v"][0]):
            return qs["v"][0]
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"} and YOUTUBE_ID_RE.match(parts[1]):
            return parts[1]
    return None


def ffmpeg_location() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def download_youtube(source: str, dest: Path) -> IngestInfo:
    from yt_dlp import YoutubeDL

    tmp_base = dest.with_suffix("")
    ffmpeg = ffmpeg_location()
    opts = {
        # Prefer a single progressive file. Fall back to split A/V if we have ffmpeg.
        "format": "b[height<=720][ext=mp4]/b[ext=mp4][height<=720]/b[height<=720]/bv*[height<=720]+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(tmp_base) + ".%(ext)s",
        "noplaylist": True,
        "quiet": False,
        "noprogress": False,
        "writesubtitles": False,
        "writeinfojson": True,
        "overwrites": True,
    }
    if ffmpeg:
        opts["ffmpeg_location"] = ffmpeg
    with YoutubeDL(opts) as ydl:
        extracted = ydl.extract_info(source, download=True) or {}

    if not dest.exists():
        candidates = list(dest.parent.glob(tmp_base.name + ".*"))
        videos = [p for p in candidates if p.suffix.lower() in {".mp4", ".webm", ".mkv"}]
        if not videos:
            raise FileNotFoundError(f"yt-dlp finished but no video file was written near {dest}")
        if videos[0] != dest:
            videos[0].replace(dest)

    return IngestInfo(
        video_id="",
        source=source,
        title=str(extracted.get("title") or dest.stem),
        path=str(dest),
        duration_s=extracted.get("duration"),
        uploader=extracted.get("uploader"),
        description=(extracted.get("description") or "")[:2000],
        webpage_url=extracted.get("webpage_url") or source,
    )
