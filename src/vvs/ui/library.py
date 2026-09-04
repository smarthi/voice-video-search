"""Library view: every indexed video as a card with a short preview clip.

Metadata is read straight from `meta.json` with plain json rather than through
`IndexMeta`, so a half-written or legacy directory degrades to a partial card
instead of breaking the whole grid. Previews are generated once and cached.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote

from vvs.paths import meta_path, preview_path, video_path
from vvs.retrieve.index import list_indexed
from vvs.retrieve.search import fmt_time

PREVIEW_S = 6.0
PREVIEW_WIDTH = 480
PREVIEW_FPS = 15
# Skip cold opens (black frames, title cards) when choosing where to start.
PREVIEW_SKIP_FRACTION = 0.08

try:  # pragma: no cover - depends on the installed Gradio
    from gradio.routes import API_PREFIX as _API_PREFIX
except Exception:  # pragma: no cover
    _API_PREFIX = "/gradio_api"


@dataclass(frozen=True)
class LibraryEntry:
    video_id: str
    title: str
    duration_s: float
    keyframes: int
    transcript: int
    ocr: int
    scenes: int
    shots: int
    has_audio: bool
    model: str
    encoder: str
    backend: str
    thumb: Path | None
    preview: Path | None
    preview_start_s: float
    has_video: bool

    @property
    def label(self) -> str:
        return self.title or self.video_id


def entries(video_ids: list[str] | None = None) -> list[LibraryEntry]:
    ids = list(video_ids) if video_ids is not None else list_indexed()
    out: list[LibraryEntry] = []
    for vid in ids:
        entry = _entry(vid)
        if entry is not None:
            out.append(entry)
    return out


def _entry(video_id: str) -> LibraryEntry | None:
    path = meta_path(video_id)
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[library] unreadable meta for {video_id}: {exc}")
        return None

    frames = meta.get("frames") or []
    duration = float(meta.get("duration_s") or 0.0)
    channels = meta.get("channels") or {}
    models = meta.get("models") or {}
    start, _ = preview_window(meta)
    preview = preview_path(video_id)
    return LibraryEntry(
        video_id=video_id,
        title=str(meta.get("title") or "").strip(),
        duration_s=duration,
        keyframes=len(frames),
        transcript=len(meta.get("transcript") or []),
        ocr=len(meta.get("ocr") or []),
        scenes=len(meta.get("scenes") or []),
        shots=len(meta.get("shots") or []),
        has_audio=bool(channels.get("asr")),
        model=str(models.get("clip") or "unknown"),
        encoder=str(models.get("visual_encoder") or ""),
        backend=str(meta.get("vector_backend") or "faiss"),
        thumb=_pick_thumb(frames, start),
        preview=preview if _usable(preview) else None,
        preview_start_s=start,
        has_video=video_path(video_id).exists(),
    )


def _pick_thumb(frames: list[dict], near_s: float) -> Path | None:
    """Keyframe closest to where the preview starts, so poster and clip agree."""
    best: Path | None = None
    best_delta = float("inf")
    for frame in frames:
        raw = frame.get("path")
        if not raw:
            continue
        candidate = Path(raw)
        if not candidate.exists():
            continue
        delta = abs(float(frame.get("t_s") or 0.0) - near_s)
        if delta < best_delta:
            best, best_delta = candidate, delta
    return best


def preview_window(meta: dict) -> tuple[float, float]:
    """Where to cut the preview from.

    Prefer the first spoken segment past the intro so a preview of a talk opens
    on speech rather than a title card; fall back to a fixed offset for mute
    assets. Short clips just start at zero.
    """
    duration = float(meta.get("duration_s") or 0.0)
    if duration <= 0:
        return 0.0, PREVIEW_S
    floor = duration * PREVIEW_SKIP_FRACTION
    start = floor
    for segment in meta.get("transcript") or []:
        text = str(segment.get("text") or "").strip()
        begin = float(segment.get("start_s") or 0.0)
        if text and begin >= floor:
            start = begin
            break
    length = min(PREVIEW_S, max(1.0, duration))
    start = max(0.0, min(start, max(0.0, duration - length)))
    return round(start, 3), round(min(duration, start + length), 3)


def ensure_preview(video_id: str, *, force: bool = False) -> Path | None:
    """Build the preview if missing. Returns None when it cannot be made."""
    dest = preview_path(video_id)
    if not force and _usable(dest):
        return dest
    source = video_path(video_id)
    if not source.exists():
        return None
    path = meta_path(video_id)
    try:
        meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        meta = {}
    start, end = preview_window(meta)
    if end <= start:
        return None
    tmp = dest.with_name("preview.partial.mp4")
    try:
        if tmp.exists():
            tmp.unlink()
        _encode_preview(source, tmp, start, end)
        if not _usable(tmp):
            raise RuntimeError("preview encode produced an unusable file")
        tmp.replace(dest)
    except Exception as exc:
        print(f"[library] preview failed for {video_id}: {exc}")
        if tmp.exists():
            tmp.unlink()
        return None
    return dest


def ensure_previews(
    video_ids: list[str] | None = None,
    *,
    force: bool = False,
    on_progress=None,
) -> dict[str, Path | None]:
    ids = list(video_ids) if video_ids is not None else list_indexed()
    built: dict[str, Path | None] = {}
    for i, vid in enumerate(ids, start=1):
        if on_progress is not None:
            on_progress(i, len(ids), vid)
        built[vid] = ensure_preview(vid, force=force)
    return built


def _encode_preview(source: Path, dest: Path, start_s: float, end_s: float) -> None:
    """Silent, downscaled, decimated H.264 so a grid of cards stays cheap.

    Audio is dropped on purpose: a wall of preview tiles that can autoplay on
    hover must be silent, and skipping the resampler keeps this fast.
    """
    import av

    dest.parent.mkdir(parents=True, exist_ok=True)
    time_base = Fraction(1, 90_000)
    container = av.open(str(source))
    out = av.open(str(dest), mode="w", options={"movflags": "+faststart"})
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        width, height = _fit(stream.width, stream.height, PREVIEW_WIDTH)
        encoder = out.add_stream("libx264", rate=PREVIEW_FPS)
        encoder.width = width
        encoder.height = height
        encoder.pix_fmt = "yuv420p"
        encoder.time_base = time_base
        encoder.options = {"preset": "veryfast", "crf": "30"}

        container.seek(int(max(0.0, start_s - 1.0) * av.time_base))
        step = 1.0 / PREVIEW_FPS
        next_t = start_s
        wrote = 0
        for frame in container.decode(stream):
            t = frame.time
            if t is None:
                continue
            if t + 1e-4 < start_s:
                continue
            if t >= end_s:
                break
            if t + 1e-6 < next_t:
                continue
            next_t = t + step
            scaled = frame.reformat(width=width, height=height, format="yuv420p")
            scaled.pts = max(0, round((t - start_s) / float(time_base)))
            scaled.time_base = time_base
            for packet in encoder.encode(scaled):
                out.mux(packet)
            wrote += 1
        for packet in encoder.encode():
            out.mux(packet)
        if not wrote:
            raise RuntimeError(f"no frames decoded in {start_s:.2f}-{end_s:.2f}s")
    finally:
        out.close()
        container.close()


def _fit(width: int | None, height: int | None, target: int) -> tuple[int, int]:
    """Scale to `target` width, keeping aspect and even dimensions for yuv420p."""
    if not width or not height:
        return target, int(target * 9 / 16) // 2 * 2
    if width <= target:
        return width // 2 * 2, height // 2 * 2
    scaled_h = round(height * (target / width))
    return target // 2 * 2, max(2, scaled_h // 2 * 2)


def _usable(path: Path | None) -> bool:
    return bool(path and path.exists() and path.is_file() and path.stat().st_size > 1024)


def file_url(path: Path | None) -> str | None:
    if path is None:
        return None
    return f"{_API_PREFIX}/file={quote(str(Path(path).resolve()))}"


SORT_CHOICES: tuple[tuple[str, str], ...] = (
    ("Recently indexed", "recent"),
    ("Title (A-Z)", "title"),
    ("Longest first", "duration"),
    ("Most keyframes", "keyframes"),
)


def filter_entries(items: list[LibraryEntry], text: str | None) -> list[LibraryEntry]:
    """Case-insensitive substring match over title and id."""
    needle = (text or "").strip().lower()
    if not needle:
        return list(items)
    return [
        item
        for item in items
        if needle in item.title.lower() or needle in item.video_id.lower()
    ]


def sort_entries(items: list[LibraryEntry], key: str | None) -> list[LibraryEntry]:
    if key == "title":
        return sorted(items, key=lambda i: (i.label.lower(), i.video_id))
    if key == "duration":
        return sorted(items, key=lambda i: (-i.duration_s, i.label.lower()))
    if key == "keyframes":
        return sorted(items, key=lambda i: (-i.keyframes, i.label.lower()))
    # "recent": newest write first, which is what you want right after an
    # analyze. meta.json's mtime is the closest thing to an indexed-at stamp.
    return sorted(items, key=lambda i: (-_meta_mtime(i.video_id), i.label.lower()))


def _meta_mtime(video_id: str) -> float:
    try:
        return meta_path(video_id).stat().st_mtime
    except OSError:
        return 0.0


def library_summary(items: list[LibraryEntry], total: int) -> str:
    total_s = sum(item.duration_s for item in items)
    shown = f"{len(items)} videos" if len(items) == total else f"{len(items)} of {total}"
    missing = sum(1 for item in items if item.preview is None)
    tail = f" · {missing} without a preview" if missing else ""
    return f"{shown} · {fmt_time(total_s)} total{tail}"


def card_html(item: LibraryEntry, *, selected: bool = False) -> str:
    """Inner markup for one card: media plus metadata, no outer container.

    The container is a real Gradio column so the card can own buttons, while the
    media stays hand-written HTML to keep hover-play and `preload="none"`.
    """
    klass = "sl-lib-inner sl-lib-on" if selected else "sl-lib-inner"
    return (
        f'<div class="{klass}">'
        f'<div class="sl-lib-media">{_media_html(item)}</div>'
        f'<div class="sl-lib-body">{_body_html(item)}</div>'
        "</div>"
    )


def render_library(items: list[LibraryEntry]) -> str:
    """Static, non-interactive grid. Kept for tests and read-only surfaces."""
    if not items:
        return empty_html()
    cards = "\n".join(
        f'<figure class="sl-lib-card">{card_html(item)}</figure>' for item in items
    )
    return f'<div class="sl-lib"><div class="sl-lib-grid">{cards}</div></div>'


def empty_html(filtered: bool = False) -> str:
    if filtered:
        return (
            '<div class="sl-lib-empty"><p>No video matches that filter.</p>'
            "<p>Clear it to see the whole archive.</p></div>"
        )
    return (
        '<div class="sl-lib-empty"><p>No videos indexed yet.</p>'
        "<p>Analyze one below and it will appear here with a preview.</p></div>"
    )


def _media_html(item: LibraryEntry) -> str:
    title = html.escape(item.label)
    vid = html.escape(item.video_id)
    poster = file_url(item.thumb)
    preview = file_url(item.preview)
    if preview:
        # No native controls: they cover the poster in a dense grid, and a 6s
        # silent loop has nothing to scrub. Hover plays instead.
        return (
            f'<video class="sl-lib-vid" src="{preview}"'
            + (f' poster="{poster}"' if poster else "")
            + ' muted loop playsinline preload="none"'
            f' aria-label="Preview of {title}" data-vid="{vid}"></video>'
            '<span class="sl-lib-play" aria-hidden="true"></span>'
        )
    if poster:
        return (
            f'<img class="sl-lib-still" src="{poster}" alt="{title}">'
            '<span class="sl-lib-badge">no preview</span>'
        )
    return '<div class="sl-lib-blank">no frames</div>'


def _body_html(item: LibraryEntry) -> str:
    title = html.escape(item.label)
    vid = html.escape(item.video_id)
    chips = [f'<span class="sl-chip">{item.keyframes} keyframes</span>']
    if item.has_audio:
        chips.append(f'<span class="sl-chip">{item.transcript} spoken</span>')
    else:
        chips.append('<span class="sl-chip sl-chip-mute">mute</span>')
    if item.ocr:
        chips.append(f'<span class="sl-chip">{item.ocr} OCR</span>')
    if item.scenes:
        chips.append(f'<span class="sl-chip">{item.scenes} X-Ray</span>')
    if item.shots:
        chips.append(f'<span class="sl-chip">{item.shots} shots</span>')
    if not item.has_video:
        chips.append('<span class="sl-chip sl-chip-warn">video file missing</span>')
    model = html.escape(item.model.split("/")[-1])
    return (
        f'<div class="sl-lib-title" title="{title}">{title}</div>'
        f'<div class="sl-lib-meta"><code>{vid}</code> · {fmt_time(item.duration_s)}</div>'
        f'<div class="sl-lib-chips">{"".join(chips)}</div>'
        f'<div class="sl-lib-model">{model}</div>'
    )
