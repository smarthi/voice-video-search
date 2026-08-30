"""Data-directory helpers."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("VVS_DATA_DIR") or (PROJECT_ROOT / "data"))


def video_dir(video_id: str) -> Path:
    return DATA_ROOT / video_id


def video_path(video_id: str) -> Path:
    return video_dir(video_id) / "video.mp4"


def frames_dir(video_id: str) -> Path:
    return video_dir(video_id) / "frames"


def clips_dir(video_id: str) -> Path:
    return video_dir(video_id) / "clips"


def index_path(video_id: str) -> Path:
    """Legacy NumPy archive. New indexes use FAISS files; this is migrate-only."""
    return video_dir(video_id) / "index.npz"


def visual_faiss_path(video_id: str) -> Path:
    return video_dir(video_id) / "visual.faiss"


def visual_sqlite_path(video_id: str) -> Path:
    return video_dir(video_id) / "visual.sqlite"


def meta_path(video_id: str) -> Path:
    return video_dir(video_id) / "meta.json"


def timings_path(video_id: str) -> Path:
    return video_dir(video_id) / "timings.json"


def demo_questions_path(video_id: str) -> Path:
    return video_dir(video_id) / "demo_questions.json"


def fallback_html_path(video_id: str) -> Path:
    return video_dir(video_id) / "fallback.html"
