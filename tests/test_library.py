"""Library grid helpers: filter, sort, preview windows, markup (no model download)."""

from __future__ import annotations

from pathlib import Path

from vvs.ui.library import (
    PREVIEW_S,
    LibraryEntry,
    card_html,
    empty_html,
    file_url,
    filter_entries,
    library_summary,
    preview_window,
    render_library,
    sort_entries,
)


def _entry(
    video_id: str,
    *,
    title: str = "",
    duration_s: float = 60.0,
    keyframes: int = 10,
    transcript: int = 0,
    ocr: int = 0,
    scenes: int = 0,
    shots: int = 0,
    has_audio: bool = False,
    thumb: Path | None = None,
    preview: Path | None = None,
    has_video: bool = True,
) -> LibraryEntry:
    return LibraryEntry(
        video_id=video_id,
        title=title,
        duration_s=duration_s,
        keyframes=keyframes,
        transcript=transcript,
        ocr=ocr,
        scenes=scenes,
        shots=shots,
        has_audio=has_audio,
        model="Qwen/Qwen3-VL-Embedding-2B",
        encoder="qwen",
        backend="faiss",
        thumb=thumb,
        preview=preview,
        preview_start_s=0.0,
        has_video=has_video,
    )


def test_filter_matches_title_and_id() -> None:
    items = [
        _entry("abc123", title="Lenovo Tech World"),
        _entry("zzz999", title="Skyfall"),
    ]
    assert [i.video_id for i in filter_entries(items, "lenovo")] == ["abc123"]
    assert [i.video_id for i in filter_entries(items, "ZZZ")] == ["zzz999"]
    assert len(filter_entries(items, "  ")) == 2
    assert filter_entries(items, "nothing here") == []


def test_sorts_are_descending_where_expected() -> None:
    items = [
        _entry("a", title="Beta", duration_s=10.0, keyframes=5),
        _entry("b", title="alpha", duration_s=300.0, keyframes=1),
        _entry("c", title="Gamma", duration_s=100.0, keyframes=99),
    ]
    assert [i.video_id for i in sort_entries(items, "title")] == ["b", "a", "c"]
    assert [i.video_id for i in sort_entries(items, "duration")] == ["b", "c", "a"]
    assert [i.video_id for i in sort_entries(items, "keyframes")] == ["c", "a", "b"]
    # Unknown keys fall back to "recent" rather than raising.
    assert len(sort_entries(items, "bogus")) == 3


def test_untitled_entry_falls_back_to_id() -> None:
    assert _entry("rawid").label == "rawid"


def test_preview_window_prefers_first_speech_past_the_intro() -> None:
    meta = {
        "duration_s": 200.0,
        "transcript": [
            {"start_s": 1.0, "text": "cold open"},  # inside the skipped intro
            {"start_s": 40.0, "text": "the real thing"},
        ],
    }
    start, end = preview_window(meta)
    assert start == 40.0
    assert end == 40.0 + PREVIEW_S


def test_preview_window_skips_blank_transcript_text() -> None:
    meta = {"duration_s": 200.0, "transcript": [{"start_s": 90.0, "text": "   "}]}
    start, _ = preview_window(meta)
    assert start == 200.0 * 0.08


def test_preview_window_for_mute_video_uses_offset() -> None:
    start, end = preview_window({"duration_s": 100.0})
    assert start == 8.0
    assert end == 8.0 + PREVIEW_S


def test_preview_window_never_runs_past_a_short_video() -> None:
    start, end = preview_window({"duration_s": 3.0})
    assert start == 0.0
    assert end == 3.0


def test_preview_window_handles_missing_duration() -> None:
    start, end = preview_window({})
    assert start == 0.0
    assert end == PREVIEW_S


def test_summary_reports_subsets_and_missing_previews() -> None:
    items = [_entry("a", duration_s=60.0), _entry("b", duration_s=60.0)]
    assert "2 videos" in library_summary(items, 2)
    assert "2 of 5" in library_summary(items, 5)
    assert "2 without a preview" in library_summary(items, 2)


def test_card_marks_channels_and_selection() -> None:
    mute = card_html(_entry("m", keyframes=7))
    assert "7 keyframes" in mute
    assert "sl-chip-mute" in mute
    assert "sl-lib-on" not in mute

    rich = card_html(
        _entry("r", has_audio=True, transcript=12, ocr=3, scenes=4, shots=5),
        selected=True,
    )
    assert "12 spoken" in rich
    assert "3 OCR" in rich and "4 X-Ray" in rich and "5 shots" in rich
    assert "sl-lib-on" in rich


def test_card_warns_when_source_video_is_gone() -> None:
    assert "video file missing" in card_html(_entry("x", has_video=False))


def test_card_escapes_markup_in_titles() -> None:
    hostile = card_html(_entry("x", title='<script>alert("x")</script>'))
    assert "<script>" not in hostile
    assert "&lt;script&gt;" in hostile


def test_card_without_media_degrades_instead_of_breaking() -> None:
    html = card_html(_entry("x"))
    assert "sl-lib-blank" in html
    assert "<video" not in html


def test_card_uses_video_when_preview_exists() -> None:
    html = card_html(_entry("x", preview=Path("/tmp/p.mp4"), thumb=Path("/tmp/t.jpg")))
    assert "<video" in html and 'preload="none"' in html
    assert "poster=" in html
    # Native controls would cover the poster in a dense grid.
    assert "controls" not in html


def test_file_url_is_the_gradio_file_route() -> None:
    assert file_url(None) is None
    url = file_url(Path("/tmp/a b.mp4"))
    assert url.startswith("/gradio_api/file=")
    assert "%20" in url  # spaces escaped, slashes left intact
    assert "/tmp/" in url


def test_empty_states_differ_for_filtered_and_bare() -> None:
    assert "No videos indexed yet" in empty_html()
    assert "matches that filter" in empty_html(filtered=True)
    assert "No videos indexed yet" in render_library([])


def test_render_library_emits_one_card_per_entry() -> None:
    html = render_library([_entry("a"), _entry("b")])
    assert html.count("sl-lib-card") == 2


def test_delete_index_refuses_to_escape_the_data_root() -> None:
    """The id reaches this from the UI, so traversal must not resolve outside."""
    from vvs.retrieve.index import delete_index

    for hostile in ("../../etc", "..", "a/b", "/etc"):
        try:
            delete_index(hostile)
        except SystemExit as exc:
            assert "refusing to delete" in str(exc)
        else:
            raise AssertionError(f"{hostile!r} was not refused")


def test_delete_index_is_a_noop_for_unknown_ids() -> None:
    from vvs.retrieve.index import delete_index

    assert delete_index("definitely-not-indexed-9f3a") is False


def test_atomic_write_leaves_no_partial_file_and_keeps_a_backup() -> None:
    import json
    import tempfile
    from pathlib import Path

    from vvs.paths import read_json_text, write_json_atomic

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "meta.json"
        write_json_atomic(target, json.dumps({"v": 1}))
        assert json.loads(target.read_text())["v"] == 1
        assert not (Path(tmp) / "meta.json.tmp").exists()

        write_json_atomic(target, json.dumps({"v": 2}))
        assert json.loads(target.read_text())["v"] == 2
        # The previous copy is rotated, not overwritten: meta.json holds the
        # transcript and captions, which cannot be rebuilt from the FAISS store.
        assert json.loads(target.with_suffix(".json.bak").read_text())["v"] == 1
        assert read_json_text(target) == target.read_text()


def test_truncated_meta_is_recovered_from_the_backup() -> None:
    """The exact macOS failure: an abort mid-write leaves invalid JSON behind."""
    import json
    import tempfile
    from pathlib import Path

    from vvs.paths import read_json_text, write_json_atomic

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "meta.json"
        write_json_atomic(target, json.dumps({"good": True}))
        write_json_atomic(target, json.dumps({"good": True, "n": 2}))
        target.write_text('{"good": tr')  # truncated, as an abort would leave it

        recovered = json.loads(read_json_text(target))
        assert recovered["good"] is True

        target.unlink()  # a missing primary must also fall back
        assert json.loads(read_json_text(target))["good"] is True


def test_read_json_text_raises_when_nothing_is_usable() -> None:
    import tempfile
    from pathlib import Path

    from vvs.paths import read_json_text

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "meta.json"
        try:
            read_json_text(target)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected FileNotFoundError")

        target.write_text("{not json")
        target.with_suffix(".json.bak").write_text("also not json")
        try:
            read_json_text(target)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected FileNotFoundError when both are corrupt")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ok")
