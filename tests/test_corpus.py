"""Corpus merge (no model download)."""

from __future__ import annotations

import importlib

from vvs.models import Hit
from vvs.retrieve.store import StoreHit, VectorPayload
from vvs.retrieve.search import corpus_relevance, merge_corpus_hits, search_corpus

# `import vvs.retrieve.search as search_module` would silently bind the
# *function* vvs.retrieve.search (re-exported by vvs/retrieve/__init__.py's
# `from vvs.retrieve.search import search`), not the submodule -- because
# importing that function overwrites the package attribute Python originally
# set to the submodule. importlib.import_module goes through sys.modules
# directly and is immune to that shadowing.
search_module = importlib.import_module("vvs.retrieve.search")


def _hit(
    video_id: str,
    rank: int,
    confidence: float,
    *,
    t_s: float | None = None,
    visual_score: float | None = None,
    scene_caption: str = "",
    quote: str = "",
    scene_score: float = 0.0,
    transcript_score: float = 0.0,
    ocr_score: float = 0.0,
    motion_energy: float = 0.0,
    flash_energy: float = 0.0,
) -> Hit:
    return Hit(
        rank=rank,
        frame_index=rank,
        t_s=t_s if t_s is not None else float(rank),
        path=f"{video_id}/{rank}.jpg",
        video_id=video_id,
        video_title=video_id,
        score=confidence,
        confidence=confidence,
        visual_score=confidence if visual_score is None else visual_score,
        transcript_score=transcript_score,
        ocr_score=ocr_score,
        scene_score=scene_score,
        scene_caption=scene_caption,
        quote=quote,
        motion_energy=motion_energy,
        flash_energy=flash_energy,
    )


def test_merge_prefers_stronger_ann() -> None:
    strong = [_hit("a", i, 0.99 - i * 0.01) for i in range(1, 7)]
    weak = [_hit("b", 1, 0.5), _hit("b", 2, 0.4)]
    merged = merge_corpus_hits([strong, weak], top_k=6)
    assert [hit.video_id for hit in merged] == ["a"] * 6
    assert [hit.rank for hit in merged] == list(range(1, 7))


def test_merge_single_video_passthrough() -> None:
    only = [_hit("solo", i, 0.9 - i * 0.05) for i in range(1, 5)]
    merged = merge_corpus_hits([only], top_k=3)
    assert [hit.video_id for hit in merged] == ["solo", "solo", "solo"]
    assert [hit.rank for hit in merged] == [1, 2, 3]


def test_empty_query_is_empty() -> None:
    assert search_corpus("   ", ["a", "b"]) == []


def test_corpus_reuses_query_embeddings_per_encoder_family(monkeypatch) -> None:
    from types import SimpleNamespace

    bundles = {
        "a": {"meta": SimpleNamespace(models=SimpleNamespace(clip="same")), "text": object()},
        "b": {"meta": SimpleNamespace(models=SimpleNamespace(clip="same")), "text": object()},
    }
    calls = {"visual": 0, "text": 0, "search": 0, "expand": 0}
    monkeypatch.setattr(search_module, "encoder_kind", lambda **_kw: "qwen_vl")
    monkeypatch.setattr(search_module, "visual_dim", lambda **_kw: 2048)
    # Stub expansion so this stays a fast, offline unit test -- expand_query_terms
    # would otherwise load the real Qwen3-VL-Instruct model (query_expand defaults on).
    monkeypatch.setattr(
        search_module,
        "expand_query_terms",
        lambda *_a, **_kw: calls.__setitem__("expand", calls["expand"] + 1) or [],
    )
    monkeypatch.setattr(
        search_module,
        "embed_clip_texts",
        lambda *_a, **_kw: calls.__setitem__("visual", calls["visual"] + 1) or [[1.0]],
    )
    monkeypatch.setattr(
        search_module,
        "embed_bge_query",
        lambda *_a, **_kw: calls.__setitem__("text", calls["text"] + 1) or [[1.0]],
    )
    monkeypatch.setattr(
        search_module,
        "search",
        lambda *_a, **_kw: calls.__setitem__("search", calls["search"] + 1) or [],
    )
    assert search_module.search_corpus("bird", ["a", "b"], loaded=bundles) == []
    # Expansion is looked up once for the whole archive call (it's a rewrite of
    # the query text, independent of which encoder embeds it), and visual
    # embedding is still once per encoder family, not once per video.
    assert calls == {"visual": 1, "text": 1, "search": 2, "expand": 1}


def test_merge_max_score_hits_keeps_best_across_phrasings() -> None:
    def store_hit(score: float, frame_index: int, t_s: float) -> StoreHit:
        return StoreHit(
            vector_id=0,
            score=score,
            payload=VectorPayload(vector_id=0, asset_id="a", frame_index=frame_index, t_s=t_s),
        )

    original = [store_hit(0.4, 1, 1.0), store_hit(0.9, 2, 2.0)]
    expansion = [store_hit(0.7, 1, 1.0), store_hit(0.2, 3, 3.0)]
    merged = search_module._merge_max_score_hits([original, expansion])
    by_frame = {h.payload.frame_index: h.score for h in merged}
    assert by_frame == {1: 0.7, 2: 0.9, 3: 0.2}
    # Sorted best-first: downstream _channel_from_visual assigns rank by list order.
    assert [h.payload.frame_index for h in merged] == [2, 1, 3]


def test_merge_max_score_hits_single_list_is_passthrough() -> None:
    hits = [
        StoreHit(vector_id=0, score=0.5, payload=VectorPayload(vector_id=0, asset_id="a"))
    ]
    assert search_module._merge_max_score_hits([hits]) is hits


def test_bm25_candidate_is_added_without_using_its_raw_score() -> None:
    row = StoreHit(
        vector_id=0,
        score=42.0,
        payload=VectorPayload(vector_id=0, asset_id="a", modality="transcript", text="exact term"),
    )
    merged = search_module._merge_lexical_candidates([], [row])
    assert len(merged) == 1
    assert merged[0].score == search_module.LEXICAL_CANDIDATE_SCORE - 0.001


def test_uniform_text_noise_does_not_outrank_a_real_visual_match() -> None:
    """Regression: a video whose text channels all sit near BGE's noise floor
    for every query (not a watermark -- real, distinct marketing copy that
    just happens to score similarly against almost anything) must not
    outrank a video with a genuinely elevated, single-channel match, just
    because it has more channels to sum. Summing let 0.55*1.7 (ocr) push a
    "kissing" query's top hit to an unrelated corporate video ahead of the
    actual kissing scene, which had no ocr contribution at all.
    """
    noisy_but_irrelevant = [
        _hit(
            "corporate",
            1,
            0.95,
            visual_score=1.30,
            transcript_score=1.66,
            ocr_score=1.67,
            scene_score=1.75,
            scene_caption="Our platform delivers peer-reviewed research to clinicians.",
        )
    ]
    genuine_match = [
        _hit(
            "kissing_scene",
            1,
            0.98,
            visual_score=1.46,
            scene_score=2.30,
            scene_caption="A man and a woman are kissing passionately in a dimly lit room.",
        )
    ]
    merged = merge_corpus_hits(
        [noisy_but_irrelevant, genuine_match], top_k=2, query="show me clips of kissing"
    )
    assert merged[0].video_id == "kissing_scene"


def test_ann_scores_beat_rrf_confidence() -> None:
    """Interview can look 92% confident from ASR extras and still lose."""
    interview = [
        _hit(
            "interview",
            1,
            0.92,
            visual_score=0.27,
            scene_caption="a man in a suit and tie is talking to the camera",
        )
    ]
    mountains = [
        _hit(
            "roja",
            1,
            0.38,
            visual_score=0.31,
            scene_score=0.67,
            scene_caption="A vast, snow-covered mountain range with sharp peaks.",
        )
    ]
    merged = merge_corpus_hits(
        [interview, mountains],
        top_k=3,
        query="show me clips with snow-capped mountains",
    )
    assert merged[0].video_id == "roja"


def test_caption_ann_wins_without_query_words() -> None:
    """Synonyms: BGE scene score is high, caption never says 'alps'."""
    interview = [_hit("interview", 1, 0.9, visual_score=0.28, scene_caption="a man in a suit")]
    alps = [
        _hit(
            "doc",
            1,
            0.4,
            visual_score=0.30,
            scene_score=0.62,
            scene_caption="A high ridge under winter light.",
        )
    ]
    query = "snow-capped alps"
    assert "alps" not in alps[0].scene_caption.lower()
    assert corpus_relevance(alps[0], query) > corpus_relevance(interview[0], query)
    merged = merge_corpus_hits([interview, alps], top_k=2, query=query)
    assert merged[0].video_id == "doc"


def test_spoken_robotics_beats_music_video_visual() -> None:
    """Only the interview says robotics; a song still can have a high visual score."""
    interview = [
        _hit(
            "interview",
            1,
            0.4,
            visual_score=0.22,
            transcript_score=0.58,
            quote="I'm leading up research groups in robotics.",
            scene_caption="a man in a suit talking to the camera",
        )
    ]
    song = [
        _hit(
            "roja",
            1,
            0.7,
            visual_score=0.33,
            scene_score=0.18,
            scene_caption="A snow-covered mountain range framed by bare trees.",
        )
    ]
    merged = merge_corpus_hits([song, interview], top_k=3, query="robotics")
    assert merged[0].video_id == "interview"


def test_instrument_still_beats_unrelated_interview_asr() -> None:
    """BGE ~0.66 on Kira-talk is not a trumpet match; the sax close-up is."""
    interview = [
        _hit(
            "interview",
            1,
            0.62,
            visual_score=0.17,
            transcript_score=0.66,
            ocr_score=0.69,
            scene_score=0.69,
            quote="We're excited about the entire product lineup from our AI PCs.",
            scene_caption="a man in a suit talking to the camera",
        )
    ]
    connelly = [
        _hit(
            "careless",
            1,
            0.94,
            visual_score=0.50,
            ocr_score=0.71,
            scene_score=0.82,
            scene_caption="A close-up of a person playing a saxophone.",
        )
    ]
    merged = merge_corpus_hits(
        [interview, connelly], top_k=3, query="clips with a trumpet"
    )
    assert merged[0].video_id == "careless"


def test_moving_query_prefers_high_motion_still() -> None:
    still = [
        _hit(
            "parked",
            1,
            0.8,
            visual_score=0.48,
            scene_score=0.70,
            scene_caption="A tornado on the horizon under a dark sky.",
            motion_energy=0.04,
        )
    ]
    moving = [
        _hit(
            "spinning",
            1,
            0.8,
            visual_score=0.46,
            scene_score=0.68,
            scene_caption="A tornado on the horizon under a dark sky.",
            motion_energy=0.41,
        )
    ]
    parked_first = merge_corpus_hits([still, moving], top_k=2, query="tornado")
    assert {hit.video_id for hit in parked_first[:2]} == {"parked", "spinning"}
    moving_first = merge_corpus_hits([still, moving], top_k=2, query="moving tornado")
    assert moving_first[0].video_id == "spinning"


def test_flash_query_prefers_lightning_still() -> None:
    landscape = [_hit("field", 1, 0.7, visual_score=0.40, scene_score=0.55, flash_energy=0.03)]
    bolt = [_hit("storm", 1, 0.7, visual_score=0.38, scene_score=0.52, flash_energy=0.48)]
    merged = merge_corpus_hits([landscape, bolt], top_k=2, query="lightning over the field")
    assert merged[0].video_id == "storm"


if __name__ == "__main__":
    import types

    # No pytest here -- this repo's tests run as plain scripts (`uv run python
    # tests/test_x.py`). Without this shim, any test taking a `monkeypatch`
    # parameter (like test_corpus_reuses_query_embeddings_per_encoder_family
    # below) would silently never execute under that runner.
    class _Monkeypatch:
        def __init__(self) -> None:
            self._saved: list[tuple[object, str, object]] = []

        def setattr(self, target, name, value) -> None:
            self._saved.append((target, name, getattr(target, name)))
            setattr(target, name, value)

        def undo(self) -> None:
            for target, name, value in reversed(self._saved):
                setattr(target, name, value)

    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and isinstance(_fn, types.FunctionType):
            if "monkeypatch" in _fn.__code__.co_varnames[: _fn.__code__.co_argcount]:
                _mp = _Monkeypatch()
                try:
                    _fn(_mp)
                finally:
                    _mp.undo()
            else:
                _fn()
    print("ok")
