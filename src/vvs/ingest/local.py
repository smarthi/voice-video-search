"""Copy a local video into the data directory."""

from __future__ import annotations

import re
import shutil
from hashlib import sha256
from pathlib import Path

from vvs.models import IngestInfo


def local_slug(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")
    return cleaned[:48] or "local-video"


def local_video_id(source: Path) -> str:
    """Stable, collision-resistant ID for a local source path.

    Filename-only IDs made two unrelated ``demo.mp4`` files silently share one
    archive directory.  Keep a readable stem, but make the resolved pathname
    part of the identity.
    """
    resolved = source.expanduser().resolve()
    digest = sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    return f"{local_slug(resolved.stem)}-{digest}"


def ingest_local(source: Path, dest: Path, video_id: str) -> IngestInfo:
    if dest.resolve() != source.resolve():
        shutil.copy2(source, dest)
    return IngestInfo(
        video_id=video_id,
        source=str(source.resolve()),
        title=source.stem,
        path=str(dest),
    )
