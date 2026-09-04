"""Corpus merge (no model download)."""

from __future__ import annotations

from vvs.models import Hit
from vvs.retrieve.search import corpus_relevance, merge_corpus_hits, search_corpus


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
    test_merge_prefers_stronger_ann()
    test_merge_single_video_passthrough()
    test_empty_query_is_empty()
    test_ann_scores_beat_rrf_confidence()
    test_caption_ann_wins_without_query_words()
    test_spoken_robotics_beats_music_video_visual()
    test_instrument_still_beats_unrelated_interview_asr()
    test_moving_query_prefers_high_motion_still()
    test_flash_query_prefers_lightning_still()
    print("ok")
