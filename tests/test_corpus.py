"""Corpus merge (no model download)."""

from __future__ import annotations

import importlib

import numpy as np

from vvs.models import (
    ChannelFlags,
    Hit,
    IndexMeta,
    Keyframe,
    ModelCard,
    SceneCard,
    TranscriptSegment,
)
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


def test_bm25_candidate_already_found_by_dense_keeps_its_dense_score() -> None:
    """Regression: a candidate found by BOTH dense ANN and BM25 -- the
    strongest kind of match there is -- must keep its real dense score, not
    get overwritten with the ~0.3 placeholder meant only for lexical-only
    extras dense search missed entirely.
    """
    payload = VectorPayload(vector_id=0, asset_id="a", modality="scene", t_s=2.52, text="a woman with a bindi")
    dense_hit = StoreHit(vector_id=0, score=1.82, payload=payload)
    lexical_hit = StoreHit(vector_id=0, score=8.35, payload=payload)
    merged = search_module._merge_lexical_candidates([dense_hit], [lexical_hit])
    assert len(merged) == 1
    assert merged[0].score == 1.82


def test_lexical_rerank_idf_boost_does_not_reward_a_bigger_video_more() -> None:
    """Regression: "greenery" mentioned once in an 86-scene conference video
    used to outscore "water"+"greenery" both genuinely, repeatedly present
    in an 8-scene waterfall video -- idf computed against each video's own
    corpus size, so the same word is worth wildly different amounts
    depending on how many *other*, unrelated scenes that video happens to
    have. Capping idf means the single incidental mention in the big corpus
    can no longer out-boost the real, repeated match in the small one.
    """
    query = "clips of running water with greenery"
    # Small, on-topic corpus (mirrors the real 8-doc waterfall video): both
    # "water" and "greenery" genuinely recur.
    small_corpus = [
        "a waterfall in a green valley",
        "water cascades over green rocks",
        "a lush greenery covered hillside",
        "another waterfall scene",
        "water flowing through a garden",
        "greenery and stone bridge",
        "a river with water",
        "final shot of the waterfall",
    ]
    waterfall_hit = StoreHit(
        vector_id=0,
        score=1.83,
        payload=VectorPayload(
            vector_id=0,
            asset_id="waterfall",
            modality="scene",
            text="a waterfall in a green valley. Objects: waterfall, greenery",
        ),
    )
    # Large, mostly-unrelated corpus (mirrors the real 86-doc conference
    # video): "greenery" appears exactly once, by coincidence, in a scene
    # that isn't really about water at all.
    big_corpus = [f"scene about topic number {i}" for i in range(85)] + [
        "a donkey under a roof with greenery in the background"
    ]
    donkey_hit = StoreHit(
        vector_id=0,
        score=1.72,
        payload=VectorPayload(
            vector_id=0,
            asset_id="donkey",
            modality="scene",
            text="a donkey under a roof with greenery in the background",
        ),
    )
    waterfall_reranked = search_module._lexical_rerank(
        [waterfall_hit], query, small_corpus
    )[0]
    donkey_reranked = search_module._lexical_rerank([donkey_hit], query, big_corpus)[0]
    assert waterfall_reranked.score > donkey_reranked.score


def test_search_uses_the_provided_lexical_corpus_not_the_local_one() -> None:
    """Regression: a per-video-only corpus made ordinary words ("man",
    "woman", "suit") look rare in a video that just doesn't happen to
    describe people much, capped-idf-boosting an unrelated scene that
    happened to share several such words over a caption genuinely about
    the query. search_corpus() now builds one archive-wide corpus and
    passes it through -- this checks search() actually uses it instead of
    silently falling back to building its own local one.
    """
    frames = [Keyframe(index=0, t_s=1.0, path="f0.jpg", hash_hex="a")]
    scenes = [
        SceneCard(
            scene_id=0,
            start_s=1.0,
            end_s=2.0,
            frame_index=0,
            path="f0.jpg",
            caption="a common word appears here",
        )
    ]
    meta = _fake_meta(frames=frames, scenes=scenes)
    scene_hit = StoreHit(
        vector_id=0,
        score=1.7,
        payload=VectorPayload(
            vector_id=0, asset_id="fake", modality="scene", t_s=1.0, text="a common word appears here"
        ),
    )
    bundle = {
        "meta": meta,
        "visual": _FakeVisualStore([]),
        "text": _FakeTextStore(dense=[scene_hit], lexical=[]),
    }
    seen_corpora: list[list[str]] = []
    original = search_module._lexical_rerank

    def spy(hits, query, documents):
        seen_corpora.append(list(documents))
        return original(hits, query, documents)

    real = search_module._lexical_rerank
    search_module._lexical_rerank = spy
    try:
        search_module.search(
            "fake",
            "common word",
            top_k=1,
            loaded=bundle,
            visual_queries=[np.zeros(1)],
            text_query=np.zeros(1),
            lexical_corpus=["an archive-wide document", "another one"],
        )
    finally:
        search_module._lexical_rerank = real

    assert seen_corpora == [["an archive-wide document", "another one"]]


def test_search_corpus_builds_lexical_corpus_from_every_bundle() -> None:
    """search_corpus() must hand every video's search() call the SAME
    archive-wide corpus (all bundles' text combined), not each video's own
    slice -- otherwise the idf-inflation bug is back per-video in
    disguise.
    """
    meta_a = _fake_meta(
        video_id="a",
        scenes=[
            SceneCard(scene_id=0, start_s=0.0, end_s=1.0, frame_index=0, path="a.jpg", caption="alpha scene text")
        ],
    )
    meta_b = _fake_meta(
        video_id="b",
        scenes=[
            SceneCard(scene_id=0, start_s=0.0, end_s=1.0, frame_index=0, path="b.jpg", caption="beta scene text")
        ],
    )
    bundles = {
        "a": {"meta": meta_a, "text": object()},
        "b": {"meta": meta_b, "text": object()},
    }
    seen_corpora: list[list[str]] = []
    real_search = search_module.search
    orig_encoder_kind = search_module.encoder_kind
    orig_visual_dim = search_module.visual_dim
    orig_embed_clip_texts = search_module.embed_clip_texts
    orig_embed_bge_query = search_module.embed_bge_query
    orig_expand = search_module.expand_query_terms

    def fake_search(vid, query, *, top_k, loaded, visual_queries, text_query, lexical_corpus=None):
        seen_corpora.append(list(lexical_corpus or []))
        return []

    search_module.encoder_kind = lambda **_kw: "fake_encoder"
    search_module.visual_dim = lambda **_kw: 8
    search_module.embed_clip_texts = lambda *_a, **_kw: [[1.0]]
    search_module.embed_bge_query = lambda *_a, **_kw: [[1.0]]
    search_module.expand_query_terms = lambda *_a, **_kw: []
    search_module.search = fake_search
    try:
        search_module.search_corpus("query", ["a", "b"], top_k=1, loaded=bundles)
    finally:
        search_module.encoder_kind = orig_encoder_kind
        search_module.visual_dim = orig_visual_dim
        search_module.embed_clip_texts = orig_embed_clip_texts
        search_module.embed_bge_query = orig_embed_bge_query
        search_module.expand_query_terms = orig_expand
        search_module.search = real_search

    assert len(seen_corpora) == 2
    for corpus in seen_corpora:
        assert "alpha scene text" in corpus
        assert "beta scene text" in corpus


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


class _FakeVisualStore:
    def __init__(self, hits: list[StoreHit]) -> None:
        self._hits = hits

    def search(self, query_vec, top_k, filters=None):  # noqa: ANN001
        return self._hits


class _FakeTextStore:
    def __init__(self, dense: list[StoreHit], lexical: list[StoreHit]) -> None:
        self._dense = dense
        self._lexical = lexical

    def search(self, query_vec, top_k, filters=None):  # noqa: ANN001
        return self._dense

    def lexical_search(self, query, top_k):  # noqa: ANN001
        return self._lexical


def _fake_meta(**kwargs) -> IndexMeta:
    defaults = dict(
        video_id="fake",
        source="unit-test",
        video_path="fake.mp4",
        models=ModelCard(clip="fake-clip", asr="fake-asr", text="fake-bge"),
        channels=ChannelFlags(visual=True, transcript=True, asr=True, scene=True),
    )
    defaults.update(kwargs)
    return IndexMeta(**defaults)


def test_search_does_not_let_a_noisy_asr_window_bury_a_real_scene_match() -> None:
    """Regression: real bug reproduced end-to-end through search() itself.

    "show me clips of a woman with a bindi" returned frames from one long
    ASR segment ("thank you all for joining us today", 0-20s) that never
    mentions "bindi" at all, and never surfaced the frame at 25.5s whose
    scene caption literally says "a woman with a red bindi on her
    forehead" -- because (a) the noisy ASR hit's dense score (noise floor,
    unrelated to the query) drove TEMPORAL_BOOST/anchors and the primary
    transcript RRF term regardless of relevance, and (b) the real scene
    hit was *also* found by BM25 lexical search (it contains "bindi"
    verbatim) which used to clobber its genuine dense score down to the
    ~0.3 placeholder meant only for lexical-only extras. Both are fixed;
    this proves neither regresses.
    """
    frames = [
        Keyframe(index=0, t_s=10.0, path="f0.jpg", hash_hex="a"),
        Keyframe(index=1, t_s=25.5, path="f1.jpg", hash_hex="b"),
    ]
    transcript = [
        TranscriptSegment(index=0, start_s=0.0, end_s=20.0, text="thank you all for joining us today")
    ]
    scenes = [
        SceneCard(
            scene_id=0,
            start_s=25.0,
            end_s=26.0,
            frame_index=1,
            path="f1.jpg",
            caption="A woman with a red bindi on her forehead",
        )
    ]
    meta = _fake_meta(frames=frames, transcript=transcript, scenes=scenes)

    visual_hits = [
        StoreHit(
            vector_id=0,
            score=1.30,
            payload=VectorPayload(vector_id=0, asset_id="fake", t_s=25.5, frame_index=1),
        )
    ]
    asr_hit = StoreHit(
        vector_id=0,
        score=1.75,  # noise-floor "high" score, but the text never says "bindi"
        payload=VectorPayload(
            vector_id=0,
            asset_id="fake",
            modality="transcript",
            t_s=10.0,
            text="thank you all for joining us today",
        ),
    )
    scene_hit = StoreHit(
        vector_id=0,
        score=1.6,  # the real match -- lower dense score than the noise above
        payload=VectorPayload(
            vector_id=0,
            asset_id="fake",
            modality="scene",
            t_s=25.0,
            text="A woman with a red bindi on her forehead",
        ),
    )
    # BM25 also finds the scene hit (it contains "bindi" verbatim) -- the
    # exact overlap that used to make _merge_lexical_candidates clobber it.
    lexical_hit = scene_hit.model_copy(update={"score": 8.0})

    bundle = {
        "meta": meta,
        "visual": _FakeVisualStore(visual_hits),
        "text": _FakeTextStore(dense=[asr_hit, scene_hit], lexical=[lexical_hit]),
    }

    hits = search_module.search(
        "fake",
        "woman with bindi",
        top_k=2,
        loaded=bundle,
        visual_queries=[np.zeros(1)],
        text_query=np.zeros(1),
    )

    assert hits[0].frame_index == 1
    # Not clobbered down to the ~0.3 lexical-only placeholder (a tiny
    # in-test corpus also earns _lexical_rerank's own small IDF boost on
    # top of the 1.6 dense score, so assert well above the placeholder
    # rather than pin the exact post-boost value).
    assert hits[0].scene_score > 1.0


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


def test_transcript_and_ocr_need_lexical_overlap_to_compete() -> None:
    """Regression: "robotics" never spoken in this video's 3 ASR segments,
    but one of them still scores ~1.7 (noise floor) against the query.
    Weighted by TEXT_ANN_WEIGHT, that alone used to beat a real, moderate
    scene match (a genuine "two humanoid robots" caption). Without a query
    term actually in the hit's own quote/caption, transcript and ocr must
    not enter corpus_relevance's max() at all -- scene still can, by design
    (see test_caption_ann_wins_without_query_words), since it has real
    dynamic range that short transcript/ocr text doesn't.
    """
    noisy_video = [
        _hit(
            "conference",
            1,
            0.5,
            visual_score=1.28,
            transcript_score=1.706,
            quote="how everything is just a gesture away",
            scene_caption="a man talking to the camera",
        )
    ]
    robot_video = [
        _hit(
            "same-conference-different-scene",
            1,
            0.5,
            visual_score=1.39,
            scene_score=1.80,
            scene_caption="A man stands in a factory surrounded by two humanoid robots.",
        )
    ]
    merged = merge_corpus_hits([noisy_video, robot_video], top_k=2, query="robotics")
    assert merged[0].video_id == "same-conference-different-scene"


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
