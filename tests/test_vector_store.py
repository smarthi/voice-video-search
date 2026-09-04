"""Lightweight checks for the VectorStore path (no model download)."""

from __future__ import annotations

import os
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


def test_faiss_is_single_threaded_on_macos_only() -> None:
    """A FAISS OpenMP region on a worker thread segfaults when two libomp load.

    macOS is where faiss-cpu and torch each bring their own libomp, so it is the
    only platform that needs the parallelism turned off. Linux keeps the default.
    """
    import os
    import sys
    from unittest import mock

    from vvs.retrieve.faiss_store import faiss_thread_default

    os.environ.pop("VVS_FAISS_THREADS", None)
    with mock.patch.object(sys, "platform", "darwin"):
        assert faiss_thread_default() == 1
    with mock.patch.object(sys, "platform", "linux"):
        assert faiss_thread_default() == 0


def test_faiss_thread_override() -> None:
    import os
    import sys
    from unittest import mock

    from vvs.retrieve.faiss_store import faiss_thread_default

    try:
        os.environ["VVS_FAISS_THREADS"] = "4"
        with mock.patch.object(sys, "platform", "darwin"):
            assert faiss_thread_default() == 4
        os.environ["VVS_FAISS_THREADS"] = "0"  # 0 means "leave faiss alone"
        with mock.patch.object(sys, "platform", "darwin"):
            assert faiss_thread_default() == 0
        for bogus in ("", "lots", "-2"):
            os.environ["VVS_FAISS_THREADS"] = bogus
            with mock.patch.object(sys, "platform", "darwin"):
                assert faiss_thread_default() == 1, bogus
    finally:
        os.environ.pop("VVS_FAISS_THREADS", None)


def test_every_faiss_call_site_goes_through_the_pinned_loader() -> None:
    """A direct `import faiss` would skip the thread limit and reintroduce the crash."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src/vvs/retrieve/faiss_store.py"
    lines = src.read_text().splitlines()
    raw = [i for i, line in enumerate(lines, 1) if line.strip() == "import faiss"]
    assert len(raw) == 1, f"expected only the loader to import faiss, found lines {raw}"
    assert "def _faiss" in "\n".join(lines[max(0, raw[0] - 8) : raw[0]])


def test_search_still_returns_hits_with_threads_pinned() -> None:
    import importlib
    import tempfile

    import numpy as np

    import vvs.paths
    import vvs.retrieve.faiss_store as fs
    from vvs.retrieve.faiss_store import FaissStore
    from vvs.retrieve.store import VectorPayload

    previous_root = os.environ.get("VVS_DATA_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["VVS_DATA_DIR"] = tmp
        importlib.reload(vvs.paths)
        # faiss_store captured video_dir at import, so it needs the new root too.
        fs.video_dir = vvs.paths.video_dir
        store = FaissStore("probe", "visual", index_type="hnsw")
        store.create(8)
        vecs = np.eye(8, dtype=np.float32)
        store.add(
            vecs,
            [
                VectorPayload(
                    vector_id=i, modality="video_frame", path=f"f{i}.jpg", t_s=float(i)
                )
                for i in range(8)
            ],
        )
        hits = store.search(vecs[3], top_k=2)
        assert hits, "single-threaded FAISS must still return hits"
        assert hits[0].payload.t_s == 3.0
        store.close()

    # Leave the data root as we found it; later tests read real indexes.
    if previous_root is None:
        os.environ.pop("VVS_DATA_DIR", None)
    else:
        os.environ["VVS_DATA_DIR"] = previous_root
    importlib.reload(vvs.paths)
    fs.video_dir = vvs.paths.video_dir


if __name__ == "__main__":
    test_shard_key_is_stable()
    test_qdrant_requires_url()
    test_force_mute_skips_asr()
    print("ok")
