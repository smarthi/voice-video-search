"""Checks for the OpenSearch-backed VectorStore (no model download).

These need a reachable OpenSearch (`docker compose up -d`); each test skips
itself with a printed reason when one isn't there, the same way
`test_force_mute_skips_asr` skips when the demo video isn't present.

Each test gets its own disposable index name (not the real archive's
vvs_visual/vvs_text, which already hold real vectors at a fixed dimension)
so tests can freely pick whatever dimension is convenient and clean up by
just dropping the whole index afterward.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np

from vvs.media.probe import has_audio_stream
from vvs.retrieve.index import should_run_asr
from vvs.retrieve.opensearch_store import OpenSearchStore, ping
from vvs.retrieve.store import VectorPayload


def _skip_without_opensearch() -> bool:
    if ping():
        return False
    print("[test_vector_store] OpenSearch unreachable; run `docker compose up -d`. Skipping.")
    return True


def _fresh_index() -> str:
    return f"vvs_test_{uuid.uuid4().hex[:8]}"


def _drop_index(index: str) -> None:
    from opensearchpy import OpenSearch

    OpenSearch(hosts=["http://localhost:9200"]).indices.delete(index=index, ignore=[404])


def test_force_mute_skips_asr() -> None:
    video = Path("data/fBEGOk3ZHw4/video.mp4")
    if not video.exists():
        return
    assert has_audio_stream(video) is True
    assert should_run_asr(video, force_mute=True) is False
    assert should_run_asr(video, force_mute=False) is True


def test_opensearch_roundtrip() -> None:
    if _skip_without_opensearch():
        return
    index = _fresh_index()
    vid = f"probe-{uuid.uuid4().hex[:8]}"
    try:
        store = OpenSearchStore(vid, "visual", index_override=index)
        store.create(8)
        vecs = np.eye(8, dtype=np.float32)
        store.add(
            vecs,
            [
                VectorPayload(
                    vector_id=i, asset_id=vid, modality="video_frame", path=f"f{i}.jpg", t_s=float(i)
                )
                for i in range(8)
            ],
        )
        hits = store.search(vecs[3], top_k=2)
        assert hits, "OpenSearch k-NN search must return hits"
        assert hits[0].payload.t_s == 3.0
        assert store.ntotal() == 8
        store.close()
    finally:
        _drop_index(index)


def test_all_payloads_filters_by_video_and_modality() -> None:
    if _skip_without_opensearch():
        return
    index = _fresh_index()
    vid_a = f"probe-a-{uuid.uuid4().hex[:8]}"
    vid_b = f"probe-b-{uuid.uuid4().hex[:8]}"
    try:
        store_a = OpenSearchStore(vid_a, "visual", index_override=index)
        store_b = OpenSearchStore(vid_b, "visual", index_override=index)
        store_a.create(4)
        store_a.add(
            np.eye(4, dtype=np.float32),
            [
                VectorPayload(vector_id=0, asset_id=vid_a, modality="video_frame", t_s=0.0),
                VectorPayload(vector_id=1, asset_id=vid_a, modality="ocr", text="hi", t_s=1.0),
                VectorPayload(vector_id=2, asset_id=vid_a, modality="video_frame", t_s=2.0),
                VectorPayload(vector_id=3, asset_id=vid_a, modality="ocr", text="bye", t_s=3.0),
            ],
        )
        store_b.create(4)
        store_b.add(
            np.eye(4, dtype=np.float32)[:1],
            [VectorPayload(vector_id=0, asset_id=vid_b, modality="video_frame", t_s=9.0)],
        )

        all_a = store_a.all_payloads()
        assert {p.t_s for p in all_a} == {0.0, 1.0, 2.0, 3.0}
        ocr_a = store_a.all_payloads(modality="ocr")
        assert {p.text for p in ocr_a} == {"hi", "bye"}
        # vid_b's row must never leak into vid_a's payload list.
        assert all(p.asset_id == vid_a for p in all_a)
        store_a.close()
        store_b.close()
    finally:
        _drop_index(index)


def test_create_wipes_prior_rows_for_the_same_video_only() -> None:
    """Rebuild semantics: create() truncates this video's rows, not the whole shared index."""
    if _skip_without_opensearch():
        return
    index = _fresh_index()
    vid_rebuilt = f"probe-rebuild-{uuid.uuid4().hex[:8]}"
    vid_other = f"probe-other-{uuid.uuid4().hex[:8]}"
    try:
        rebuilt = OpenSearchStore(vid_rebuilt, "visual", index_override=index)
        other = OpenSearchStore(vid_other, "visual", index_override=index)
        rebuilt.create(4)
        rebuilt.add(
            np.eye(4, dtype=np.float32)[:2],
            [
                VectorPayload(vector_id=0, asset_id=vid_rebuilt, t_s=0.0),
                VectorPayload(vector_id=1, asset_id=vid_rebuilt, t_s=1.0),
            ],
        )
        other.create(4)
        other.add(
            np.eye(4, dtype=np.float32)[:1],
            [VectorPayload(vector_id=0, asset_id=vid_other, t_s=5.0)],
        )
        assert rebuilt.ntotal() == 2
        assert other.ntotal() == 1

        # Re-analyze vid_rebuilt: create() again, then add fewer rows.
        rebuilt.create(4)
        assert rebuilt.ntotal() == 0, "create() must wipe this video's prior rows"
        assert other.ntotal() == 1, "a sibling video's rows must survive another video's rebuild"
        rebuilt.add(
            np.eye(4, dtype=np.float32)[:1],
            [VectorPayload(vector_id=0, asset_id=vid_rebuilt, t_s=2.0)],
        )
        assert rebuilt.ntotal() == 1
        rebuilt.close()
        other.close()
    finally:
        _drop_index(index)


def test_dim_mismatch_across_videos_is_rejected() -> None:
    """One shared index has one fixed vector width; a later, different-dim video must fail loudly."""
    if _skip_without_opensearch():
        return
    index = _fresh_index()
    vid_first = f"probe-dim-a-{uuid.uuid4().hex[:8]}"
    vid_second = f"probe-dim-b-{uuid.uuid4().hex[:8]}"
    try:
        first = OpenSearchStore(vid_first, "visual", index_override=index)
        first.create(6)
        first.add(
            np.eye(6, dtype=np.float32)[:1], [VectorPayload(vector_id=0, asset_id=vid_first, t_s=0.0)]
        )
        second = OpenSearchStore(vid_second, "visual", index_override=index)
        try:
            second.create(3)
        except SystemExit as exc:
            assert "6-d" in str(exc) and "3-d" in str(exc)
        else:
            raise AssertionError("expected SystemExit on a conflicting vector width")
        first.close()
    finally:
        _drop_index(index)


def test_generation_scopes_reads_and_rebuild_cleanup() -> None:
    if _skip_without_opensearch():
        return
    index = _fresh_index()
    vid = f"probe-generation-{uuid.uuid4().hex[:8]}"
    try:
        old = OpenSearchStore(vid, "visual", index_override=index, generation="gold")
        old.create(2)
        old.add(np.array([[1.0, 0.0]], dtype=np.float32), [VectorPayload(vector_id=0, asset_id=vid)])
        new = OpenSearchStore(vid, "visual", index_override=index, generation="gnew")
        new.create(2)
        new.add(np.array([[0.0, 1.0]], dtype=np.float32), [VectorPayload(vector_id=0, asset_id=vid)])
        assert old.search(np.array([1.0, 0.0]), top_k=1)
        assert new.search(np.array([0.0, 1.0]), top_k=1)
        assert old.search(np.array([0.0, 1.0]), top_k=1) == []
        old.close()
        new.close()
    finally:
        _drop_index(index)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ok")
