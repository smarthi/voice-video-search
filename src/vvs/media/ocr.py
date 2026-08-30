"""Optional RapidOCR channel for slide-heavy video."""

from __future__ import annotations

from vvs.models import Keyframe, OcrObservation


_OCR = None


def get_ocr():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR

        print("[ocr] loading RapidOCR")
        _OCR = RapidOCR()
    return _OCR


def ocr_frames(frames: list[Keyframe], min_score: float = 0.5) -> list[OcrObservation]:
    engine = get_ocr()
    observations: list[OcrObservation] = []
    for i, frame in enumerate(frames):
        result, _elapse = engine(str(frame.path))
        if not result:
            continue
        texts: list[str] = []
        scores: list[float] = []
        for item in result:
            if len(item) < 3:
                continue
            text = str(item[1]).strip()
            try:
                score = float(item[2])
            except (TypeError, ValueError):
                score = 0.0
            if text and score >= min_score:
                texts.append(text)
                scores.append(score)
        if not texts:
            continue
        joined = " ".join(texts)
        if not usable_ocr_text(joined):
            continue
        observations.append(
            OcrObservation(
                frame_index=frame.index,
                t_s=frame.t_s,
                text=" ".join(texts),
                score=float(sum(scores) / len(scores)),
            )
        )
        if (i + 1) % 25 == 0:
            print(f"[ocr] {i + 1}/{len(frames)} frames")
    print(f"[ocr] {len(observations)} frames with text")
    return observations


def is_slide_heavy(
    observations: list[OcrObservation],
    frames: list[Keyframe],
    min_ratio: float = 0.35,
) -> bool:
    if not frames:
        return False
    long_hits = sum(1 for o in observations if len(o.text) >= 24)
    return (long_hits / len(frames)) >= min_ratio


def usable_ocr_text(text: str) -> bool:
    tokens = [tok for tok in text.replace(":", " ").replace("/", " ").split() if len(tok) >= 3]
    if len(tokens) < 2:
        return False
    alpha = sum(ch.isalpha() for ch in text)
    if alpha < 8:
        return False
    vowels = sum(ch in "aeiouAEIOU" for ch in text)
    return vowels >= 2
