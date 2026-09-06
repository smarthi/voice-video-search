"""RapidOCR post-processing: watermark detection (no model download)."""

from __future__ import annotations

from vvs.media.ocr import drop_watermark_rows, usable_ocr_text
from vvs.models import OcrObservation


def _row(i: int, text: str) -> OcrObservation:
    return OcrObservation(frame_index=i, t_s=float(i), text=text, score=0.9)


def test_drop_watermark_rows_removes_persistent_overlay() -> None:
    # Real-world shape: the same upload watermark, read slightly differently
    # frame to frame by RapidOCR.
    variants = [
        "JenniferConnelly80s 4KULTRAHD·",
        "JenniferConnelly80s 4KUUTRAHD·",
        "JenniferConnalty80 4KULTRAHD·",
        "JenniferConnellyBos 4KULTRAHD",
        "JenniferConnelly&0s -4KULTRAHD",
        "JenniferConmelly80s 4KULTRAHD",
    ]
    rows = [_row(i, v) for i, v in enumerate(variants)]
    assert drop_watermark_rows(rows) == []


def test_drop_watermark_rows_keeps_distinct_content() -> None:
    rows = [
        _row(0, "EXIT 14A"),
        _row(1, "Refinery Digital Twin Control Room"),
        _row(2, "Donation Thanks for your donation $60.00"),
    ]
    assert drop_watermark_rows(rows) == rows


def test_drop_watermark_rows_below_ratio_threshold_is_kept() -> None:
    # 4/16 (25%) recurring text -- plausibly a company's own logo appearing
    # in a few frames of its own video, not a persistent overlay. Below the
    # 35% ratio (and this is also a case real branding legitimately produces,
    # unlike a fan-upload watermark that covers nearly every frame).
    distinct = [
        "Refinery Digital Twin Control Room",
        "Donation Thanks for your donation 60 dollars",
        "EXIT 14A next right",
        "Slide 3 of 12 quarterly results",
        "Speaker Jane Doe VP of Engineering",
        "Q and A starts in five minutes",
        "Booth 42 visit us after the talk",
        "Wifi password conference2024",
        "Session ends at four PM sharp",
        "Please silence your phones now",
        "Thank you for attending today",
        "See you next year same place",
    ]
    rows = [_row(i, "WoltersKluwer logo") for i in range(4)]
    rows += [_row(i + 4, text) for i, text in enumerate(distinct)]
    assert drop_watermark_rows(rows) == rows


def test_drop_watermark_rows_below_min_count_never_triggers() -> None:
    # Two rows is 100% "recurrence" by ratio, but min_count guards against
    # flagging a video with barely any OCR data at all.
    rows = [_row(0, "same text"), _row(1, "same text")]
    assert drop_watermark_rows(rows) == rows


def test_drop_watermark_rows_mixed_keeps_the_non_watermark_rows() -> None:
    watermark = [_row(i, "ChannelBrand 2024 subscribe") for i in range(5)]
    real = [_row(i + 5, f"unique sign {i}") for i in range(2)]
    kept = drop_watermark_rows(watermark + real)
    assert kept == real


def test_usable_ocr_text_rejects_garbage() -> None:
    assert not usable_ocr_text("::--..")
    assert not usable_ocr_text("xyz")
    assert usable_ocr_text("EXIT HERE NOW")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ok")
