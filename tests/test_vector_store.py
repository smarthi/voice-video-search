"""Lightweight checks for the VectorStore path (no model download)."""

from __future__ import annotations

from pathlib import Path

from vvs.config import Settings
from vvs.media.probe import has_audio_stream
from vvs.retrieve.index import should_run_asr
from vvs.retrieve.qdrant_store import QdrantStore
from vvs.retrieve.store import shard_key


def test_shard_key_is_stable() -> None:
    assert shard_key("asset-a", 8) == shard_key("asset-a", 8)
    assert 0 <= shard_key("asset-a", 8) < 8


def test_qdrant_requires_url() -> None:
    try:
        QdrantStore(
            "demo",
            "visual",
            settings=Settings(vector_backend="qdrant", qdrant_url=None),
        )
    except SystemExit as exc:
        assert "VVS_QDRANT_URL" in str(exc)
        return
    raise AssertionError("expected SystemExit")


def test_force_mute_skips_asr() -> None:
    video = Path("data/fBEGOk3ZHw4/video.mp4")
    if not video.exists():
        return
    assert has_audio_stream(video) is True
    assert should_run_asr(video, force_mute=True) is False
    assert should_run_asr(video, force_mute=False) is True


if __name__ == "__main__":
    test_shard_key_is_stable()
    test_qdrant_requires_url()
    test_force_mute_skips_asr()
    print("ok")
