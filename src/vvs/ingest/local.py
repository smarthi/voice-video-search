"""Copy a local video into the data directory."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from vvs.models import IngestInfo


def local_slug(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")
    return cleaned[:48] or "local-video"


def ingest_local(source: Path, dest: Path, video_id: str) -> IngestInfo:
    if dest.resolve() != source.resolve():
        shutil.copy2(source, dest)
    return IngestInfo(
        video_id=video_id,
        source=str(source.resolve()),
        title=source.stem,
        path=str(dest),
    )
