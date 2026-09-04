"""Render Mermaid diagrams to PNG so Confluence needs no Mermaid plugin.

Two modes:

    extract  Pull ```mermaid blocks out of a Markdown file, save each as a
             .mmd under docs/diagrams/, render it, and replace the fence with
             an image reference. Run once per document.

    render   Re-render every docs/diagrams/*.mmd after editing the source.
             Safe to run repeatedly; does not touch Markdown.

Rendering uses mermaid.ink (PNG) with a kroki.io fallback. Both are public
services, so this needs outbound HTTPS. Nothing else in the repo does.
"""

from __future__ import annotations

import argparse
import base64
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MMD_DIR = REPO / "docs" / "diagrams"
PNG_DIR = REPO / "docs" / "images" / "diagrams"

FENCE = re.compile(r"```mermaid\n(.*?)```\n", re.S)
HEADING = re.compile(r"^#{2,4} +(.*)$", re.M)

WIDTH = 1800
PREFIXES = {"CONFLUENCE": "proto", "CONFLUENCE_ONPREM": "onprem"}

# Both services reject the default urllib agent.
UA = "Mozilla/5.0 (X11; Linux x86_64) render_diagrams.py"


def slugify(text: str) -> str:
    text = text.replace("`", "").replace(".", "-")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return re.sub(r"-{2,}", "-", text)


def heading_before(md: str, pos: int) -> str:
    matches = [m for m in HEADING.finditer(md) if m.start() < pos]
    return matches[-1].group(1).strip() if matches else "diagram"


def render(mermaid: str, out: Path) -> None:
    payload = base64.urlsafe_b64encode(mermaid.encode()).decode()
    url = f"https://mermaid.ink/img/{payload}?type=png&bgColor=FFFFFF&width={WIDTH}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        print(f"    mermaid.ink failed ({exc}); trying kroki")
        req = urllib.request.Request(
            "https://kroki.io/mermaid/png",
            data=mermaid.encode(),
            headers={"Content-Type": "text/plain", "User-Agent": UA},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    if not data.startswith(b"\x89PNG"):
        raise SystemExit(f"{out.name}: response was not a PNG ({len(data)} bytes)")
    out.write_bytes(compress(data))
    print(f"    {out.relative_to(REPO)}  {out.stat().st_size // 1024} KB")


def compress(data: bytes) -> bytes:
    """Palette-quantize flat diagram PNGs. Roughly 60% smaller, visually identical.

    Matters because these get base64-inlined into the paste-in docs.
    """
    from io import BytesIO

    from PIL import Image

    img = Image.open(BytesIO(data)).convert("RGB")
    buf = BytesIO()
    img.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(
        buf, format="PNG", optimize=True
    )
    return buf.getvalue() if buf.tell() < len(data) else data


def extract(md_path: Path) -> None:
    md = md_path.read_text()
    prefix = PREFIXES.get(md_path.stem, slugify(md_path.stem))
    MMD_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    blocks: list[tuple[re.Match[str], str, str]] = []
    for m in FENCE.finditer(md):
        title = heading_before(md, m.start())
        blocks.append((m, title, f"{prefix}-{slugify(title)}"))

    if not blocks:
        print(f"{md_path.name}: no mermaid blocks")
        return

    print(f"{md_path.name}: {len(blocks)} diagrams")
    for m, title, slug in blocks:
        (MMD_DIR / f"{slug}.mmd").write_text(m.group(1))
        print(f"  {title}")
        render(m.group(1), PNG_DIR / f"{slug}.png")
        time.sleep(1.0)

    # Replace back to front so earlier spans stay valid.
    for m, title, slug in reversed(blocks):
        ref = f"![{title}](images/diagrams/{slug}.png)\n"
        md = md[: m.start()] + ref + md[m.end() :]
    md_path.write_text(md)
    print(f"  rewrote {md_path.relative_to(REPO)}")


def render_all(only: list[str] | None = None) -> None:
    sources = sorted(MMD_DIR.glob("*.mmd"))
    if only:
        sources = [s for s in sources if any(frag in s.stem for frag in only)]
    if not sources:
        raise SystemExit(f"no matching .mmd files in {MMD_DIR}")
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    for src in sources:
        print(f"  {src.stem}")
        render(src.read_text(), PNG_DIR / f"{src.stem}.png")
        time.sleep(1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["extract", "render"])
    ap.add_argument(
        "paths",
        nargs="*",
        help="Markdown files for extract mode, or slug fragments for render mode",
    )
    args = ap.parse_args()

    if args.mode == "render":
        render_all(args.paths or None)
        return 0

    if not args.paths:
        raise SystemExit("extract mode needs at least one Markdown path")
    for path in args.paths:
        extract(Path(path) if Path(path).is_absolute() else REPO / path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
