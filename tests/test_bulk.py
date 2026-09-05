"""Bulk ingest: parsing and per-item error isolation (no model download)."""

from __future__ import annotations

import vvs.retrieve.bulk as bulk


def test_parse_sources_skips_blank_and_comment_lines() -> None:
    text = "\n".join(
        [
            "# a manifest",
            "",
            "https://www.youtube.com/watch?v=aaaaaaaaaaa",
            "  ",
            "/path/to/video.mp4  ",
            "# trailing comment",
        ]
    )
    assert bulk.parse_sources(text) == [
        "https://www.youtube.com/watch?v=aaaaaaaaaaa",
        "/path/to/video.mp4",
    ]


def test_bulk_index_skips_already_indexed(monkeypatch) -> None:
    monkeypatch.setattr(bulk, "index_exists", lambda vid: vid == "aaaaaaaaaaa")
    monkeypatch.setattr(
        bulk,
        "build_index",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not build an indexed video")),
    )
    result = bulk.bulk_index(["https://www.youtube.com/watch?v=aaaaaaaaaaa"])
    assert [i.status for i in result.items] == ["skipped"]
    assert result.summary() == "0 indexed, 1 skipped, 0 failed (of 1)"


def test_bulk_index_isolates_one_bad_source() -> None:
    result = bulk.bulk_index(["/definitely/does/not/exist.mp4"])
    assert len(result.items) == 1
    item = result.items[0]
    assert item.status == "failed"
    assert item.video_id is None
    assert "Cannot determine video id" in item.detail


def test_bulk_index_continues_after_a_build_failure(monkeypatch) -> None:
    monkeypatch.setattr(bulk, "index_exists", lambda vid: False)
    calls = []

    def fake_build_index(source, video_id=None, **kwargs):
        calls.append(video_id)
        if video_id == "bad":
            raise RuntimeError("decode failed")
        return None

    monkeypatch.setattr(bulk, "build_index", fake_build_index)
    monkeypatch.setattr(
        bulk,
        "extract_video_id",
        lambda source: "bad" if "bad" in source else "good",
    )
    result = bulk.bulk_index(["/videos/bad.mp4", "/videos/good.mp4"])
    assert calls == ["bad", "good"]
    assert [i.status for i in result.items] == ["failed", "indexed"]
    assert "decode failed" in result.failed[0].detail


def test_on_item_start_reports_progress() -> None:
    monkeypatch_calls = []
    bulk.bulk_index(
        ["#comment ignored by callers, not by bulk_index"],
        on_item_start=lambda i, total, source: monkeypatch_calls.append((i, total, source)),
    )
    assert monkeypatch_calls == [(1, 1, "#comment ignored by callers, not by bulk_index")]


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

    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and isinstance(fn, types.FunctionType):
            if "monkeypatch" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                mp = _Monkeypatch()
                try:
                    fn(mp)
                finally:
                    mp.undo()
            else:
                fn()
    print("ok")
