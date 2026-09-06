"""Optional RapidOCR channel for slide-heavy video."""

from __future__ import annotations

from difflib import SequenceMatcher

from vvs.models import Keyframe, OcrObservation

# A persistent upload watermark ("JenniferConnelly80s 4KULTRAHD", a channel
# tag, a timestamp overlay) recurs near-identically across most or all of a
# video's kept frames -- unlike real on-screen text (a sign, a slide, a
# caption), which is localized to the handful of frames it is actually
# visible in. BGE rates these short, repetitive strings as surprisingly
# similar to almost any query (the same "BGE-small cosine sits around 0.65
# even on unrelated talk" noise search.py's corpus_relevance already accounts
# for on the transcript channel), and because the same text repeats across
# dozens of frames, every one of those frames gets that spurious OCR boost --
# enough, in practice, to outrank a genuinely relevant clip in another video's
# archive-wide corpus_relevance. Drop rows recognized as this kind of overlay
# before they are ever embedded or searched, rather than trying to out-tune
# the ranking formula around them.
_WATERMARK_MIN_RATIO = 0.35
_WATERMARK_MIN_COUNT = 3
_WATERMARK_SIMILARITY = 0.75


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
    observations = drop_watermark_rows(observations)
    print(f"[ocr] {len(observations)} frames with text")
    return observations


def drop_watermark_rows(
    observations: list[OcrObservation],
    *,
    min_ratio: float = _WATERMARK_MIN_RATIO,
    min_count: int = _WATERMARK_MIN_COUNT,
    similarity: float = _WATERMARK_SIMILARITY,
) -> list[OcrObservation]:
    """Drop rows whose text is a persistent overlay, not localized content.

    Clusters rows by fuzzy text similarity (RapidOCR reads the same watermark
    slightly differently frame to frame -- "Connelly" vs "Connalty" -- so an
    exact-string count would undercount it) and drops every row in any
    cluster that covers at least `min_ratio` of all rows (and at least
    `min_count` of them): recurring at that rate across a video's kept frames
    is what a watermark or channel tag looks like, not a sign or a slide.
    """
    if len(observations) < min_count:
        return observations
    norms = ["".join(ch.lower() for ch in o.text if ch.isalnum()) for o in observations]
    cluster_of: list[int] = [-1] * len(observations)
    reps: list[str] = []
    for i, norm in enumerate(norms):
        for c, rep in enumerate(reps):
            if SequenceMatcher(None, norm, rep).ratio() >= similarity:
                cluster_of[i] = c
                break
        else:
            cluster_of[i] = len(reps)
            reps.append(norm)
    total = len(observations)
    counts: dict[int, int] = {}
    for c in cluster_of:
        counts[c] = counts.get(c, 0) + 1
    watermark_clusters = {
        c for c, n in counts.items() if n >= min_count and n / total >= min_ratio
    }
    if not watermark_clusters:
        return observations
    dropped = sum(1 for c in cluster_of if c in watermark_clusters)
    print(f"[ocr] dropped {dropped}/{total} rows as a persistent watermark/overlay")
    return [o for o, c in zip(observations, cluster_of, strict=True) if c not in watermark_clusters]


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
