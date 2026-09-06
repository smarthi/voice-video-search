"""ANN retrieval: visual encoder + BGE text (OCR / optional transcript), fused with RRF."""

from __future__ import annotations

import math
import re
from typing import Any

from vvs.media.caption import expand_query_terms
from vvs.media.motion import motion_query_score, query_motion_flags
from vvs.media.ocr import usable_ocr_text
from vvs.media.scenes import scene_covering
from vvs.media.shots import shot_covering
from vvs.models import Hit, IndexMeta, Keyframe, OcrObservation, SceneCard, TranscriptSegment
from vvs.retrieve.embed import embed_bge_query, embed_clip_texts, encoder_kind, visual_dim
from vvs.retrieve.index import list_indexed, load_index
from vvs.retrieve.store import StoreHit


RRF_K = 60
TEMPORAL_BOOST = 0.85
TEXT_WEIGHT = 1.6
OCR_WEIGHT = 0.6
OCR_MIN_SCORE = 0.28
SCENE_WEIGHT = 1.0
SCENE_MIN_SCORE = 0.22
SCENE_ANN_WEIGHT = 1.15
TEXT_ANN_WEIGHT = 1.4
OCR_ANN_WEIGHT = 0.55
LEXICAL_BLEND = 0.2
MOTION_BLEND = 0.45
MOTION_RRF_WEIGHT = 2.5
ANN_OVERFETCH = 48
LEXICAL_WEIGHT = 0.14
LEXICAL_CANDIDATE_SCORE = 0.30
_STOP = frozenset(
    {
        "what",
        "whats",
        "which",
        "when",
        "where",
        "who",
        "how",
        "does",
        "did",
        "are",
        "was",
        "were",
        "the",
        "this",
        "that",
        "with",
        "from",
        "about",
        "into",
        "your",
        "their",
        "them",
        "show",
        "tell",
        "please",
        "approach",
        "regarding",
        "clip",
        "clips",
        "video",
        "videos",
        "moment",
        "moments",
        "scene",
        "scenes",
        "footage",
        "some",
        "any",
    }
)


def search(
    video_id: str,
    query: str,
    *,
    top_k: int = 6,
    loaded: dict[str, Any] | None = None,
    visual_queries: list | None = None,
    text_query=None,
) -> list[Hit]:
    query = (query or "").strip()
    if not query:
        return []
    bundle = loaded or load_index(video_id)
    meta: IndexMeta = bundle["meta"]
    frames = meta.frames
    transcript = meta.transcript
    ocr_rows = meta.ocr
    scenes = meta.scenes
    asr_on = bool(meta.channels.asr and transcript)
    ocr_on = bool(meta.channels.ocr and ocr_rows)
    scene_on = bool(getattr(meta.channels, "scene", False) and scenes)
    by_index = {frame.index: i for i, frame in enumerate(frames)}

    clip_qs = (
        visual_queries
        if visual_queries is not None
        else embed_clip_texts(_query_texts(query), meta=meta)
    )
    visual_hits = _merge_max_score_hits(
        [bundle["visual"].search(vec, top_k=max(ANN_OVERFETCH, top_k * 8)) for vec in clip_qs]
    )

    text_hits: list[StoreHit] = []
    if bundle.get("text") is not None:
        bge_q = text_query if text_query is not None else embed_bge_query(query)[0]
        text_hits = bundle["text"].search(bge_q, top_k=max(ANN_OVERFETCH, top_k * 8))
        corpus = (
            [seg.text for seg in transcript]
            + [row.text for row in ocr_rows]
            + [scene.passage_text() for scene in scenes]
        )
        text_hits = _lexical_rerank(text_hits, query, corpus)
        # BM25 scores are not comparable to cosine similarity. Convert their
        # rank into a conservative score solely to add exact-match candidates
        # to the existing RRF pipeline without corrupting corpus ANN ranking.
        lexical = bundle["text"].lexical_search(query, top_k=max(ANN_OVERFETCH, top_k * 8))
        text_hits = _merge_lexical_candidates(text_hits, lexical)

    visual_score, visual_rank = _channel_from_visual(visual_hits, frames, by_index)
    transcript_score, transcript_rank, frame_quote = _channel_from_transcript(
        text_hits, frames, transcript, asr_on
    )
    ocr_score, ocr_rank = _channel_from_ocr(text_hits, frames, by_index, ocr_rows, ocr_on)
    scene_score, scene_rank = _channel_from_scene(text_hits, frames, scenes, scene_on)

    candidates = sorted(
        set(visual_rank) | set(transcript_rank) | set(ocr_rank) | set(scene_rank),
        key=lambda i: (
            visual_rank.get(i, 10_000),
            transcript_rank.get(i, 10_000),
            ocr_rank.get(i, 10_000),
            scene_rank.get(i, 10_000),
        ),
    )
    if not candidates:
        return []

    asr_hits = [h for h in text_hits if h.payload.modality == "transcript"]
    top_text_windows = _windows_from_asr_hits(asr_hits, transcript, n=1) if asr_on else []
    anchors = _anchors_from_asr_hits(asr_hits, frames, transcript, n=1) if asr_on else set()

    fused: list[tuple[float, int]] = []
    for i in candidates:
        score = 0.0
        if i in visual_rank:
            score += 1.0 / (RRF_K + visual_rank[i])
        if asr_on and i in transcript_rank and transcript_score.get(i, 0.0) > 0.05:
            score += TEXT_WEIGHT / (RRF_K + transcript_rank[i])
        if ocr_on and i in ocr_rank and ocr_score.get(i, 0.0) >= OCR_MIN_SCORE:
            score += OCR_WEIGHT / (RRF_K + ocr_rank[i])
        if scene_on and i in scene_rank and scene_score.get(i, 0.0) >= SCENE_MIN_SCORE:
            score += SCENE_WEIGHT / (RRF_K + scene_rank[i])
        if asr_on and _in_windows(frames[i].t_s, top_text_windows, pad=2.5):
            score += TEMPORAL_BOOST / (RRF_K + 1)
        if asr_on and i in anchors:
            score += 1.25 / (RRF_K + 1)
        score += (
            MOTION_RRF_WEIGHT
            * motion_query_score(query, frames[i].motion_energy, frames[i].flash_energy)
            / (RRF_K + 1)
        )
        fused.append((score, i))

    fused.sort(key=lambda item: item[0], reverse=True)
    picked = _diverse_picks(fused, frames, top_k, min_gap_s=6.0)
    hits: list[Hit] = []
    for rank, (score, i) in enumerate(picked, start=1):
        quote, q_start, q_end = frame_quote.get(i, ("", None, None))
        scene = scene_covering(frames[i].t_s, scenes) if scenes else None
        scene_text = scene.caption if scene else ""
        shot = shot_covering(frames[i].t_s, meta.shots) if meta.shots else None
        if not quote and scene_text:
            quote, q_start, q_end = scene.passage_text(), scene.start_s, scene.end_s
        if not quote:
            quote, q_start, q_end = _ocr_quote_for_frame(frames[i], ocr_rows)
        if shot and q_start is None:
            q_start, q_end = shot.start_s, shot.end_s
        if not quote and shot:
            quote = f"shot {shot.shot_id}"
        channels: list[str] = []
        if i in visual_rank:
            channels.append("visual")
        if asr_on and transcript_score.get(i, 0.0) > 0:
            channels.append("transcript")
        if ocr_on and ocr_score.get(i, 0.0) >= OCR_MIN_SCORE:
            channels.append("ocr")
        if scene_on and scene_score.get(i, 0.0) >= SCENE_MIN_SCORE:
            channels.append("scene")
        want_motion, want_flash = query_motion_flags(query)
        if want_motion and frames[i].motion_energy >= 0.12:
            channels.append("motion")
        if want_flash and frames[i].flash_energy >= 0.18:
            channels.append("flash")
        raw = float(score)
        hits.append(
            Hit(
                rank=rank,
                frame_index=frames[i].index,
                t_s=frames[i].t_s,
                path=frames[i].path,
                video_id=meta.video_id,
                video_title=meta.title or meta.video_id,
                score=raw,
                confidence=confidence_from_score(raw, asr=asr_on, ocr=ocr_on, scene=scene_on),
                visual_score=float(visual_score.get(i, 0.0)),
                transcript_score=float(transcript_score.get(i, 0.0)),
                ocr_score=float(ocr_score.get(i, 0.0)),
                scene_score=float(scene_score.get(i, 0.0)),
                scene_caption=scene_text,
                shot_id=shot.shot_id if shot else frames[i].shot_id,
                motion_energy=float(frames[i].motion_energy),
                flash_energy=float(frames[i].flash_energy),
                quote=quote,
                quote_start_s=q_start,
                quote_end_s=q_end,
                channels=channels,
            )
        )
    return hits


def search_corpus(
    query: str,
    video_ids: list[str] | None = None,
    *,
    top_k: int = 6,
    loaded: dict[str, dict[str, Any]] | None = None,
) -> list[Hit]:
    """Search an archive while embedding once per visual encoder family."""
    query = (query or "").strip()
    if not query:
        return []
    ids = list(video_ids) if video_ids is not None else list_indexed()
    if not ids:
        return []
    bundles: dict[str, dict[str, Any]] = {}
    for vid in ids:
        try:
            bundles[vid] = (loaded or {}).get(vid) or load_index(vid)
        except Exception as exc:
            print(f"[search] skip {vid}: {exc}")
    if not bundles:
        return []
    # BGE is shared by every text collection. Visual indexes can legally have
    # different model families at the same width, so cache by writer identity.
    text_q = embed_bge_query(query)[0] if any(b.get("text") is not None for b in bundles.values()) else None
    # Expand once for the whole archive call, not once per encoder family --
    # it's a rewrite of the query text, independent of which model embeds it.
    texts = _query_texts(query)
    visual_queries: dict[tuple[str, str, int], list] = {}
    groups: list[list[Hit]] = []
    for vid, bundle in bundles.items():
        meta: IndexMeta = bundle["meta"]
        key = (encoder_kind(meta=meta), str(meta.models.clip or ""), visual_dim(meta=meta))
        if key not in visual_queries:
            visual_queries[key] = list(embed_clip_texts(texts, meta=meta))
        try:
            groups.append(
                search(
                    vid,
                    query,
                    top_k=top_k,
                    loaded=bundle,
                    visual_queries=visual_queries[key],
                    text_query=text_q,
                )
            )
        except Exception as exc:
            print(f"[search] skip {vid}: {exc}")
    return merge_corpus_hits(groups, top_k=top_k, query=query)


def corpus_relevance(hit: Hit, query: str = "") -> float:
    """Cross-asset score from absolute ANN similarities, not per-video RRF confidence.

    Picture path (visual + caption + OCR) and spoken path (transcript) compete
    with max(), they are not summed. BGE-small cosine sits around 0.65 even on
    unrelated talk, so adding 1.4×transcript to every interview beat a saxophone
    still. A real spoken hit (robotics, score ~1.4 after lexical) still wins the
    max() against a music-video picture. Lexical overlap is a small hybrid boost.
    Motion / flash scalars boost only when the query asks for movement or a strike.
    """
    visual_path = (
        float(hit.visual_score)
        + SCENE_ANN_WEIGHT * float(hit.scene_score)
        + OCR_ANN_WEIGHT * float(hit.ocr_score)
    )
    spoken_path = TEXT_ANN_WEIGHT * float(hit.transcript_score)
    return (
        max(visual_path, spoken_path)
        + LEXICAL_BLEND * _hit_overlap(query, hit)
        + MOTION_BLEND * motion_query_score(query, hit.motion_energy, hit.flash_energy)
    )


def merge_corpus_hits(
    groups: list[list[Hit]], *, top_k: int, query: str = ""
) -> list[Hit]:
    """Global top-k by corpus_relevance. One strong title may take every slot."""
    nonempty = [group for group in groups if group]
    if not nonempty:
        return []
    if len(nonempty) == 1:
        return [_with_rank(hit, rank) for rank, hit in enumerate(nonempty[0][:top_k], start=1)]
    pool = [hit for group in nonempty for hit in group]
    pool.sort(key=lambda hit: corpus_relevance(hit, query), reverse=True)
    picked: list[Hit] = []
    seen: set[tuple[str, int, float]] = set()
    for hit in pool:
        key = (hit.video_id, hit.frame_index, float(hit.t_s))
        if key in seen:
            continue
        picked.append(hit)
        seen.add(key)
        if len(picked) >= top_k:
            break
    return [_with_rank(hit, rank) for rank, hit in enumerate(picked, start=1)]


def _with_rank(hit: Hit, rank: int) -> Hit:
    return hit if hit.rank == rank else hit.model_copy(update={"rank": rank})


def format_hits(hits: list[Hit]) -> str:
    lines = []
    for hit in hits:
        if hit.clip_start_s is not None and hit.clip_end_s is not None:
            ts = f"{fmt_clock(hit.clip_start_s)}–{fmt_clock(hit.clip_end_s)}"
        else:
            ts = fmt_time(hit.t_s)
        quote = hit.quote.replace("\n", " ")
        if len(quote) > 160:
            quote = quote[:157] + "..."
        clip = f" clip={hit.clip_path}" if hit.clip_path else ""
        asset = f" {hit.video_id}" if hit.video_id else ""
        lines.append(
            f"{hit.rank}.{asset} [{ts}] confidence={fmt_confidence(hit.confidence)} "
            f"(raw={hit.score:.4f}) vis={hit.visual_score:.3f} "
            f"asr={hit.transcript_score:.3f} ocr={hit.ocr_score:.3f} "
            f"scene={hit.scene_score:.3f} mot={hit.motion_energy:.3f} "
            f"flash={hit.flash_energy:.3f} "
            f"{','.join(hit.channels)}{clip}\n    {quote}"
        )
    return "\n".join(lines)


def fmt_time(t: float) -> str:
    m, s = divmod(int(round(t)), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def rrf_ceiling(*, asr: bool, ocr: bool, scene: bool = False) -> float:
    """Largest RRF sum given the channels that can actually fire."""
    total = 1.0
    if asr:
        total += TEXT_WEIGHT + TEMPORAL_BOOST + 1.25
    if ocr:
        total += OCR_WEIGHT
    if scene:
        total += SCENE_WEIGHT
    return total / (RRF_K + 1)


def confidence_from_score(
    raw: float, *, asr: bool = True, ocr: bool = True, scene: bool = False
) -> float:
    ceiling = rrf_ceiling(asr=asr, ocr=ocr, scene=scene)
    if ceiling <= 0:
        return 0.0
    return round(min(1.0, max(0.0, float(raw) / ceiling)), 3)


def fmt_confidence(value: float) -> str:
    return f"{max(0.0, min(1.0, float(value))) * 100:.0f}%"


def fmt_clock(t: float) -> str:
    total_c = max(0, int(round(float(t) * 100)))
    s, c = divmod(total_c, 100)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}.{c:02d}"
    return f"{m:d}:{s:02d}.{c:02d}"


def _channel_from_visual(
    hits: list[StoreHit],
    frames: list[Keyframe],
    by_index: dict[int, int],
) -> tuple[dict[int, float], dict[int, int]]:
    scores: dict[int, float] = {}
    ranks: dict[int, int] = {}
    rank = 0
    for hit in hits:
        payload = hit.payload
        i = _frame_pos(payload.frame_index, payload.t_s, frames, by_index)
        if i is None or i in ranks:
            continue
        rank += 1
        ranks[i] = rank
        scores[i] = float(hit.score)
    return scores, ranks


def _channel_from_transcript(
    hits: list[StoreHit],
    frames: list[Keyframe],
    transcript: list[TranscriptSegment],
    asr_on: bool,
) -> tuple[dict[int, float], dict[int, int], dict[int, tuple[str, float | None, float | None]]]:
    scores: dict[int, float] = {}
    quotes: dict[int, tuple[str, float | None, float | None]] = {}
    if not asr_on or not frames:
        return scores, {}, quotes
    asr_hits = [h for h in hits if h.payload.modality == "transcript"]
    if not asr_hits and transcript:
        return scores, {}, quotes
    # Map each ANN transcript hit onto the nearest frame(s) in its spoken window.
    frame_best: dict[int, tuple[float, StoreHit]] = {}
    for hit in asr_hits:
        start = float(hit.payload.t_s)
        end = start
        text = (hit.payload.text or "").strip()
        for seg in transcript:
            if text and seg.text.strip() == text:
                start, end = seg.start_s, seg.end_s
                break
        for i, frame in enumerate(frames):
            if frame.t_s < start - 0.5 or frame.t_s > end + 0.5:
                continue
            prev = frame_best.get(i)
            if prev is None or hit.score > prev[0]:
                frame_best[i] = (float(hit.score), hit)
                quotes[i] = (text or seg_text(transcript, start), start, end)
    ordered = sorted(frame_best.items(), key=lambda kv: -kv[1][0])
    ranks = {i: rank for rank, (i, _) in enumerate(ordered, start=1)}
    scores = {i: pair[0] for i, pair in frame_best.items()}
    return scores, ranks, quotes


def seg_text(transcript: list[TranscriptSegment], start: float) -> str:
    for seg in transcript:
        if abs(seg.start_s - start) < 0.05:
            return seg.text
    return ""


def _channel_from_ocr(
    hits: list[StoreHit],
    frames: list[Keyframe],
    by_index: dict[int, int],
    ocr_rows: list[OcrObservation],
    ocr_on: bool,
) -> tuple[dict[int, float], dict[int, int]]:
    scores: dict[int, float] = {}
    ranks: dict[int, int] = {}
    if not ocr_on:
        return scores, ranks
    rank = 0
    for hit in hits:
        if hit.payload.modality != "ocr":
            continue
        text = hit.payload.text or ""
        if not usable_ocr_text(text):
            continue
        i = _frame_pos(hit.payload.frame_index, hit.payload.t_s, frames, by_index)
        if i is None or i in ranks:
            continue
        if float(hit.score) < OCR_MIN_SCORE:
            continue
        rank += 1
        ranks[i] = rank
        scores[i] = float(hit.score)
    if not ranks and ocr_rows:
        # Hydrate from payload-less meta only when ANN returned nothing usable.
        return scores, ranks
    return scores, ranks


def _channel_from_scene(
    hits: list[StoreHit],
    frames: list[Keyframe],
    scenes: list[SceneCard],
    scene_on: bool,
) -> tuple[dict[int, float], dict[int, int]]:
    scores: dict[int, float] = {}
    if not scene_on or not scenes or not frames:
        return scores, {}
    scene_hits = [h for h in hits if h.payload.modality == "scene"]
    frame_best: dict[int, float] = {}
    for hit in scene_hits:
        if float(hit.score) < SCENE_MIN_SCORE:
            continue
        scene = _scene_for_hit(hit, scenes)
        if scene is None:
            continue
        for i, frame in enumerate(frames):
            if frame.t_s < scene.start_s - 0.25 or frame.t_s >= scene.end_s + 0.05:
                continue
            prev = frame_best.get(i)
            if prev is None or hit.score > prev:
                frame_best[i] = float(hit.score)
    ordered = sorted(frame_best.items(), key=lambda kv: -kv[1])
    ranks = {i: rank for rank, (i, _) in enumerate(ordered, start=1)}
    return frame_best, ranks


def _scene_for_hit(hit: StoreHit, scenes: list[SceneCard]) -> SceneCard | None:
    text = (hit.payload.text or "").strip()
    for scene in scenes:
        if text and scene.passage_text().strip() == text:
            return scene
        if abs(scene.start_s - float(hit.payload.t_s)) < 0.08:
            return scene
        if hit.payload.frame_index is not None and scene.frame_index == hit.payload.frame_index:
            return scene
    return scene_covering(hit.payload.t_s, scenes)


def _ocr_quote_for_frame(
    frame: Keyframe, ocr_rows: list[OcrObservation]
) -> tuple[str, float | None, float | None]:
    for row in ocr_rows:
        if row.frame_index == frame.index and usable_ocr_text(row.text):
            return row.text, frame.t_s, frame.t_s
    return "", None, None


def _frame_pos(
    frame_index: int | None,
    t_s: float,
    frames: list[Keyframe],
    by_index: dict[int, int],
) -> int | None:
    if frame_index is not None and frame_index in by_index:
        return by_index[frame_index]
    if not frames:
        return None
    return min(range(len(frames)), key=lambda i: abs(frames[i].t_s - t_s))


def _query_terms(text: str) -> list[str]:
    return [
        tok
        for tok in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(tok) >= 4 and tok not in _STOP
    ]


def _term_in_text(term: str, blob: str) -> bool:
    if term in blob:
        return True
    if len(term) >= 5 and term.endswith("s") and term[:-1] in blob:
        return True
    return f"{term}s" in blob


def _term_overlap(query: str, text: str) -> float:
    terms = _query_terms(query)
    if not terms:
        return 0.0
    blob = (text or "").lower()
    hits = sum(1 for term in terms if _term_in_text(term, blob))
    return hits / len(terms)


def _hit_overlap(query: str, hit: Hit) -> float:
    return _term_overlap(query, f"{hit.scene_caption} {hit.quote}")


def _lexical_rerank(hits: list[StoreHit], query: str, documents: list[str]) -> list[StoreHit]:
    """Boost ANN hits whose payload contains rare query terms (hybrid dense + lexical)."""
    terms = _query_terms(query)
    if not terms or not hits:
        return hits
    docs = [(doc or "").lower() for doc in documents if doc]
    n = max(1, len(docs))
    idf = {
        term: math.log((n + 1) / (sum(1 for doc in docs if term in doc) + 1)) + 1.0
        for term in terms
    }
    adjusted: list[StoreHit] = []
    for hit in hits:
        blob = (hit.payload.text or "").lower()
        boost = sum(LEXICAL_WEIGHT * idf[term] for term in terms if term in blob)
        adjusted.append(hit.model_copy(update={"score": float(hit.score) + boost}) if boost else hit)
    adjusted.sort(key=lambda item: item.score, reverse=True)
    return adjusted


def _query_texts(query: str) -> list[str]:
    """The query plus its LLM-expanded phrasings (media/caption.py), original first.

    Called once per archive-wide search (search_corpus caches the embedding
    per encoder family, not this) and once per direct single-video search()
    call, so an expansion failure or a disabled setting costs one skipped
    call, not a retry storm.
    """
    return [query, *expand_query_terms(query)]


def _merge_max_score_hits(hit_lists: list[list[StoreHit]]) -> list[StoreHit]:
    """Union ANN candidates across query-expansion phrasings, keeping each one's best score.

    Unlike `_merge_lexical_candidates`, every list here comes from the same
    ANN space (cosine/inner-product against the same visual index), so their
    scores are directly comparable -- max is the right merge, not a capped
    placeholder score.
    """
    if len(hit_lists) == 1:
        return hit_lists[0]
    best: dict[tuple[int | None, float], StoreHit] = {}
    for hits in hit_lists:
        for hit in hits:
            key = (hit.payload.frame_index, hit.payload.t_s)
            prev = best.get(key)
            if prev is None or hit.score > prev.score:
                best[key] = hit
    return sorted(best.values(), key=lambda h: h.score, reverse=True)


def _merge_lexical_candidates(
    dense: list[StoreHit], lexical: list[StoreHit]
) -> list[StoreHit]:
    """Put exact matches first while preserving dense similarity semantics."""
    out: list[StoreHit] = []
    seen: set[tuple[str, str, float, int | None]] = set()
    for rank, hit in enumerate(lexical, start=1):
        key = (hit.payload.modality, hit.payload.text or "", hit.payload.t_s, hit.payload.frame_index)
        seen.add(key)
        out.append(hit.model_copy(update={"score": max(0.05, LEXICAL_CANDIDATE_SCORE - rank * 0.001)}))
    for hit in dense:
        key = (hit.payload.modality, hit.payload.text or "", hit.payload.t_s, hit.payload.frame_index)
        if key not in seen:
            out.append(hit)
    return out


def _resolve_seg_span(hit: StoreHit, transcript: list[TranscriptSegment]) -> tuple[float, float]:
    start = float(hit.payload.t_s)
    text = (hit.payload.text or "").strip()
    for seg in transcript:
        if text and seg.text.strip() == text:
            return seg.start_s, seg.end_s
        if abs(seg.start_s - start) < 0.05:
            return seg.start_s, seg.end_s
    return start, start + 8.0


def _windows_from_asr_hits(
    hits: list[StoreHit],
    transcript: list[TranscriptSegment],
    n: int,
) -> list[tuple[float, float]]:
    return [_resolve_seg_span(hit, transcript) for hit in hits[:n]]


def _anchors_from_asr_hits(
    hits: list[StoreHit],
    frames,
    transcript: list[TranscriptSegment],
    n: int,
) -> set[int]:
    if not frames or not hits:
        return set()
    anchors: set[int] = set()
    for hit in hits[:n]:
        start, end = _resolve_seg_span(hit, transcript)
        mid = 0.5 * (start + end)
        nearest = min(range(len(frames)), key=lambda i: abs(frames[i].t_s - mid))
        anchors.add(nearest)
    return anchors


def _in_windows(t: float, windows: list[tuple[float, float]], pad: float = 1.0) -> bool:
    return any(start - pad <= t <= end + pad for start, end in windows)


def _diverse_picks(
    fused: list[tuple[float, int]],
    frames,
    top_k: int,
    min_gap_s: float,
) -> list[tuple[float, int]]:
    picked: list[tuple[float, int]] = []
    for score, i in fused:
        t = frames[i].t_s
        if any(abs(t - frames[j].t_s) < min_gap_s for _s, j in picked):
            continue
        picked.append((score, i))
        if len(picked) >= top_k:
            break
    return picked
