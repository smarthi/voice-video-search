"""Scene clustering and X-Ray lookup (no VLM download)."""

from __future__ import annotations

import numpy as np

from vvs.media.caption import (
    _parse_expansion_lines,
    _want_gemma,
    caption_scenes,
    expand_query_terms,
    parse_qwen_caption,
)
from vvs.media.scenes import cluster_scenes, drafts_to_cards, scene_covering
from vvs.models import Keyframe
from vvs.retrieve.search import rrf_ceiling
from vvs.ui.xray import render_xray


def _frame(i: int, t: float, digest: str, path: str = "f.jpg") -> Keyframe:
    return Keyframe(index=i, t_s=t, path=path, hash_hex=digest)


def test_cluster_merges_similar_hashes() -> None:
    frames = [
        _frame(0, 0.0, "ffff"),
        _frame(1, 4.0, "fffe"),
        _frame(2, 8.0, "0000"),
        _frame(3, 12.0, "0001"),
    ]
    drafts = cluster_scenes(frames)
    assert len(drafts) == 2
    assert drafts[0].start_i == 0 and drafts[0].end_i == 1
    assert drafts[1].start_i == 2 and drafts[1].end_i == 3


def test_cluster_splits_on_max_length() -> None:
    same = "aaaaaaaa"
    frames = [_frame(i, float(i * 4), same) for i in range(8)]
    drafts = cluster_scenes(frames, max_scene_s=12.0)
    assert len(drafts) >= 3
    for draft in drafts:
        span = frames[draft.end_i].t_s - frames[draft.start_i].t_s
        assert span < 12.0 or draft.start_i == draft.end_i


def test_cluster_uses_clip_cosine() -> None:
    frames = [_frame(i, float(i), "ffff") for i in range(3)]
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    vecs = np.stack([a, a, b])
    drafts = cluster_scenes(frames, vecs, cosine_join=0.9)
    assert len(drafts) == 2
    assert drafts[0].end_i == 1
    assert drafts[1].start_i == 2


def test_scene_covering_and_caption_inject() -> None:
    frames = [_frame(0, 0.0, "aa", "a.jpg"), _frame(1, 10.0, "bb", "b.jpg")]
    drafts = cluster_scenes(frames, max_scene_s=60.0)
    cards = caption_scenes(
        frames,
        drafts,
        duration_s=20.0,
        caption_fn=lambda path: (f"caption for {path}", ["person"], "fake"),
    )
    assert len(cards) == 2
    assert cards[0].caption.startswith("caption")
    assert cards[0].objects == ["person"]
    assert scene_covering(2.0, cards) is cards[0]
    assert scene_covering(15.0, cards) is cards[1]
    assert scene_covering(-1.0, cards) is None


def test_drafts_close_intervals() -> None:
    frames = [_frame(0, 1.0, "aa"), _frame(1, 5.0, "ff")]
    drafts = cluster_scenes(frames)
    cards = drafts_to_cards(
        frames,
        drafts,
        duration_s=12.0,
        captions=[("one", [], "m"), ("two", ["laptop"], "m")],
    )
    assert cards[0].end_s == cards[1].start_s
    assert cards[-1].end_s == 12.0
    assert "laptop" in cards[1].passage_text()


def test_rrf_ceiling_includes_scene() -> None:
    base = rrf_ceiling(asr=False, ocr=False, scene=False)
    with_scene = rrf_ceiling(asr=False, ocr=False, scene=True)
    assert with_scene > base


def test_xray_html_highlights_hit() -> None:
    frames = [_frame(0, 0.0, "aa"), _frame(1, 8.0, "ff")]
    drafts = cluster_scenes(frames)
    cards = drafts_to_cards(
        frames, drafts, duration_s=16.0, captions=[("stage", [], "m"), ("slide", [], "m")]
    )
    from vvs.models import ChannelFlags, IndexMeta, ModelCard

    meta = IndexMeta(
        video_id="demo",
        source="local",
        duration_s=16.0,
        video_path="x.mp4",
        frames=frames,
        scenes=cards,
        channels=ChannelFlags(scene=True),
        models=ModelCard(clip="c", asr="a", text="t"),
    )
    from vvs.models import Hit

    html = render_xray(
        meta,
        [
            Hit(
                rank=1,
                frame_index=1,
                t_s=8.0,
                path="b.jpg",
                score=1.0,
                visual_score=1.0,
                transcript_score=0.0,
                ocr_score=0.0,
                quote="",
            )
        ],
    )
    assert "X-Ray" in html
    assert "sl-xray-on" in html
    assert "slide" in html


def test_parse_qwen_caption_splits_objects() -> None:
    caption, objects = parse_qwen_caption(
        "A red car in a tunnel.\nObjects: car, tunnel, headlights"
    )
    assert caption == "A red car in a tunnel."
    assert objects == ["car", "tunnel", "headlights"]
    assert parse_qwen_caption("")[0] == "A video frame."
    assert _want_gemma("gemma4")
    assert _want_gemma("google/gemma-4-E4B-it")
    assert _want_gemma("qwen_vl")
    assert not _want_gemma("blip")


def test_parse_expansion_lines_dedupes_and_strips_markers() -> None:
    text = "1. Running water\n- flowing stream\nrunning water\n\n   \nRIVER"
    out = _parse_expansion_lines(text, original="river", max_variants=5)
    # "running water" repeats line 1's key and "RIVER" repeats the original;
    # both are deduped, not just an exact-string match against the original.
    assert out == ["Running water", "flowing stream"]


def test_parse_expansion_lines_respects_max_variants() -> None:
    assert _parse_expansion_lines("a\nb\nc\nd", original="orig", max_variants=2) == ["a", "b"]


def test_expand_query_terms_empty_query_returns_empty() -> None:
    assert expand_query_terms("   ") == []


def test_expand_query_terms_zero_variants_returns_empty() -> None:
    # Short-circuits before ever touching the model, regardless of VVS_QUERY_EXPAND.
    assert expand_query_terms("river with greenery", max_variants=0) == []


def test_expand_query_terms_disabled_by_settings(monkeypatch) -> None:
    from types import SimpleNamespace

    import vvs.media.caption as caption_module

    monkeypatch.setattr(
        caption_module,
        "load_settings",
        lambda: SimpleNamespace(query_expand=False, query_expand_variants=2),
    )
    assert caption_module.expand_query_terms("river with greenery") == []


if __name__ == "__main__":
    import types

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
