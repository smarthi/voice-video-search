#!/usr/bin/env python3
"""Write a static HTML fallback of rehearsed queries so the demo can survive a live-mic miss."""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vvs.config import load_settings
from vvs.models import DemoQuestion
from vvs.paths import demo_questions_path, fallback_html_path
from vvs.retrieve.index import ensure_index, load_index
from vvs.retrieve.search import fmt_time, search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None)
    parser.add_argument("--video-id", default=None)
    args = parser.parse_args()
    settings = load_settings(source=args.source, video_id=args.video_id)
    vid = ensure_index(settings.source, settings.video_id)
    bundle = load_index(vid)
    questions = _load_questions(vid, bundle["meta"].title or vid)
    blocks = []
    for q in questions:
        hits = search(vid, q.query, top_k=3, loaded=bundle)
        cards = []
        for hit in hits:
            rel = Path(hit.path).name
            cards.append(
                f'<figure><img src="frames/{html.escape(rel)}" alt="">'
                f"<figcaption>#{hit.rank} {fmt_time(hit.t_s)} · {html.escape(hit.quote[:180])}</figcaption></figure>"
            )
        blocks.append(
            f"<section><h2>{html.escape(q.query)}</h2>"
            f"<p class='why'>{html.escape(q.why)}</p>"
            f"<div class='row'>{''.join(cards)}</div></section>"
        )
    title = bundle["meta"].title or vid
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Fallback · {html.escape(title)}</title>
<style>
body {{ font-family: sans-serif; margin: 24px; background: #111; color: #eee; }}
.row {{ display: flex; gap: 12px; flex-wrap: wrap; }}
figure {{ margin: 0; width: 280px; }}
img {{ width: 100%; border-radius: 8px; }}
figcaption {{ font-size: 12px; color: #bbb; }}
.why {{ color: #9ad; }}
</style></head><body>
<h1>Fallback tape · {html.escape(title)}</h1>
<p>Use this if the live microphone path fails on stage.</p>
{''.join(blocks)}
</body></html>
"""
    dest = fallback_html_path(vid)
    dest.write_text(page, encoding="utf-8")
    print(f"wrote {dest}")


def _load_questions(video_id: str, title: str) -> list[DemoQuestion]:
    path = demo_questions_path(video_id)
    if path.exists():
        import json

        raw = json.loads(path.read_text(encoding="utf-8"))
        return [DemoQuestion.model_validate(item) for item in raw]
    return [
        DemoQuestion(query=f"what is this video about", why="overview", expect=title),
    ]


if __name__ == "__main__":
    main()
