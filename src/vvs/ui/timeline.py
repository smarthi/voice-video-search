"""Render a simple match-density timeline as a PNG for the Gradio UI."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw

from vvs.models import Hit, IndexMeta
from vvs.retrieve.search import fmt_time

_INK = (5, 7, 12)
_LINE = (26, 35, 49)
_DIM = (139, 152, 169)
_TEXT = (234, 238, 244)
_MARKERS = (
    (226, 35, 26),    # red
    (255, 106, 69),   # coral
    (127, 212, 255),  # ice
    (240, 180, 41),   # amber
)


def render_corpus_timeline(
    metas: dict[str, IndexMeta],
    hits: list[Hit],
    width: int = 920,
) -> Image.Image:
    """One rail per video that contributed a hit. Single-video falls back to the old bar."""
    ids = []
    for hit in hits:
        if hit.video_id and hit.video_id not in ids:
            ids.append(hit.video_id)
    if len(ids) <= 1:
        vid = ids[0] if ids else next(iter(metas), "")
        meta = metas.get(vid)
        if meta is None:
            image = Image.new("RGB", (width, 88), _INK)
            return image
        return render_timeline(meta, hits, width=width)
    row_h = 56
    height = 28 + row_h * len(ids)
    image = Image.new("RGB", (width, height), _INK)
    draw = ImageDraw.Draw(image)
    pad = 16
    draw.text((pad, 8), f"Archive  ·  {len(ids)} videos", fill=_DIM)
    by_vid: dict[str, list[Hit]] = {}
    for hit in hits:
        by_vid.setdefault(hit.video_id, []).append(hit)
    for row, vid in enumerate(ids):
        meta = metas.get(vid)
        duration = max(getattr(meta, "duration_s", 1.0) or 1.0, 1.0)
        y0 = 28 + row * row_h
        bar_y0, bar_y1 = y0 + 18, y0 + 32
        usable = width - 2 * pad
        draw.rounded_rectangle((pad, bar_y0, width - pad, bar_y1), radius=2, fill=_LINE)
        label = (meta.title or vid) if meta else vid
        if len(label) > 48:
            label = label[:45] + "..."
        draw.text((pad, y0), label, fill=_DIM)
        for i, hit in enumerate(by_vid.get(vid, [])):
            x = pad + int((hit.t_s / duration) * usable)
            color = _MARKERS[min(i, len(_MARKERS) - 1)]
            draw.rectangle((x - 1, bar_y0 - 4, x + 1, bar_y1 + 4), fill=color)
            draw.ellipse((x - 4, bar_y0 - 2, x + 4, bar_y1 + 2), fill=color, outline=_INK)
    return image


def render_timeline(meta: IndexMeta, hits: list[Hit], width: int = 920, height: int = 88) -> Image.Image:
    image = Image.new("RGB", (width, height), _INK)
    draw = ImageDraw.Draw(image)
    pad = 16
    bar_y0, bar_y1 = 38, 54
    draw.rounded_rectangle((pad, bar_y0, width - pad, bar_y1), radius=2, fill=_LINE)
    # Sightline coverage bar: red → coral wash along the rail
    wash_w = max(8, (width - 2 * pad) // 5)
    draw.rectangle((pad, bar_y0, pad + wash_w, bar_y1), fill=(88, 18, 16))
    duration = max(meta.duration_s, 1.0)
    usable = width - 2 * pad

    for i, hit in enumerate(hits):
        x = pad + int((hit.t_s / duration) * usable)
        color = _MARKERS[min(i, len(_MARKERS) - 1)]
        draw.rectangle((x - 1, bar_y0 - 6, x + 1, bar_y1 + 6), fill=color)
        draw.ellipse((x - 5, 36, x + 5, 56), fill=color, outline=_INK)
        label = f"{hit.rank} {fmt_time(hit.t_s)}"
        draw.text((max(pad, x - 22), 60), label, fill=_TEXT)

    title = f"{meta.title or meta.video_id}  ·  {fmt_time(duration)}"
    draw.text((pad, 10), title, fill=_DIM)
    draw.rectangle((pad, height - 3, pad + max(24, usable // 4), height - 1), fill=(226, 35, 26))
    return image


def timeline_png_bytes(meta: IndexMeta, hits: list[Hit]) -> bytes:
    image = render_timeline(meta, hits)
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
