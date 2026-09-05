"""Data-directory helpers."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("VVS_DATA_DIR") or (PROJECT_ROOT / "data"))


def write_json_atomic(path: Path, payload: str) -> None:
    """Write JSON via a sibling temp file and rename, keeping the old copy.

    `os.replace` is atomic on POSIX, so an abrupt death (an OpenMP abort, say)
    leaves either the previous file or the complete new one. A plain write_text
    can leave a truncated file, and a truncated meta.json is worse than a missing
    one: `index_exists` only tests for presence, so the video stays listed while
    failing to parse, and then quietly disappears from the library.

    The previous copy is rotated to `.bak` rather than overwritten, because
    meta.json holds the transcript, scene captions and OCR text, none of which
    can be reconstructed from the OpenSearch collections. Losing it means re-running the
    captioner over the whole video. `read_json_text` reads the backup when the
    primary is unusable.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        if path.exists():
            os.replace(path, path.with_name(path.name + ".bak"))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_json_text(path: Path) -> str:
    """Read JSON text, falling back to the `.bak` copy if the primary is unusable.

    Returns the primary whenever it parses. Only a missing or corrupt primary
    reaches for the backup, and doing so is announced, because silently serving
    stale metadata would be worse than a visible warning.
    """
    import json

    backup = path.with_name(path.name + ".bak")
    for candidate, is_backup in ((path, False), (backup, True)):
        if not candidate.exists():
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
            json.loads(text)
        except (OSError, ValueError) as exc:
            print(f"[paths] {candidate.name} is unusable ({exc})")
            continue
        if is_backup:
            print(f"[paths] recovered {path.name} from {backup.name}; primary was unusable")
        return text
    raise FileNotFoundError(f"no usable JSON at {path} or {backup}")


def video_dir(video_id: str) -> Path:
    return DATA_ROOT / video_id


def video_path(video_id: str) -> Path:
    return video_dir(video_id) / "video.mp4"


def frames_dir(video_id: str) -> Path:
    return video_dir(video_id) / "frames"


def clips_dir(video_id: str) -> Path:
    return video_dir(video_id) / "clips"


def preview_path(video_id: str) -> Path:
    """Short silent preview for the library grid.

    Deliberately outside `clips/`: `attach_clips` garbage-collects anything in
    that directory whose name does not match its own `clip_<tag>_...` pattern.
    """
    return video_dir(video_id) / "preview.mp4"


def meta_path(video_id: str) -> Path:
    return video_dir(video_id) / "meta.json"


def timings_path(video_id: str) -> Path:
    return video_dir(video_id) / "timings.json"


def demo_questions_path(video_id: str) -> Path:
    return video_dir(video_id) / "demo_questions.json"


def fallback_html_path(video_id: str) -> Path:
    return video_dir(video_id) / "fallback.html"
