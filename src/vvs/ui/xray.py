"""X-Ray rail: timed scene cards for the Gradio UI."""

from __future__ import annotations

from html import escape

from vvs.models import Hit, IndexMeta
from vvs.retrieve.search import fmt_clock


def render_xray_corpus(metas: dict[str, IndexMeta], hits: list[Hit] | None = None) -> str:
    """X-Ray cards for every video that has scenes, hits first."""
    hit_ids = []
    for hit in hits or []:
        if hit.video_id and hit.video_id not in hit_ids:
            hit_ids.append(hit.video_id)
    order = hit_ids + [vid for vid in metas if vid not in hit_ids]
    sections = []
    for vid in order:
        meta = metas.get(vid)
        if meta is None or not meta.scenes:
            continue
        own = [hit for hit in (hits or []) if hit.video_id == vid]
        block = render_xray(meta, own)
        title = escape(meta.title or vid)
        sections.append(
            f'<div class="sl-xray-asset"><div class="sl-xray-asset-h">{title}</div>{block}</div>'
        )
    if not sections:
        return render_xray(None)
    return '<div class="sl-xray-wrap-corpus">' + "".join(sections) + "</div>"


def render_xray(meta: IndexMeta | None, hits: list[Hit] | None = None) -> str:
    if meta is None or not meta.scenes:
        return (
            '<div class="sl-xray sl-xray-empty">'
            '<div class="sl-xray-h">X-Ray</div>'
            "<p>No scene cards on this index. Re-analyze with "
            "<code>VVS_SCENE_CAPTION=1</code> or the scene-caption checkbox.</p>"
            "</div>"
        )
    hit_times = [float(hit.t_s) for hit in (hits or [])]
    rows: list[str] = []
    for scene in meta.scenes:
        active = any(scene.start_s <= t < scene.end_s for t in hit_times)
        objects = ", ".join(escape(obj) for obj in scene.objects[:8])
        caption = escape(scene.caption or "(no caption)")
        cls = "sl-xray-row sl-xray-on" if active else "sl-xray-row"
        obj_html = f'<span class="sl-xray-o">{objects}</span>' if objects else ""
        rows.append(
            f'<div class="{cls}">'
            f'<span class="sl-xray-t">{escape(fmt_clock(scene.start_s))}–'
            f"{escape(fmt_clock(scene.end_s))}</span>"
            f'<span class="sl-xray-c">{caption}</span>'
            f"{obj_html}"
            "</div>"
        )
    n = len(meta.scenes)
    model = escape(next((s.model for s in meta.scenes if s.model), "vlm"))
    return (
        '<div class="sl-xray">'
        f'<div class="sl-xray-h">X-Ray · {n} scene{"s" if n != 1 else ""}'
        f'<span class="sl-xray-m">{model}</span></div>'
        f"{''.join(rows)}"
        "</div>"
    )
