"""Produce self-contained Markdown: swap image paths for base64 data URIs.

The target Confluence has no Mermaid macro and no outbound network access, and
pasted Markdown cannot resolve relative image paths. Embedding every PNG as a
data URI makes a file that carries its own figures.

Edit the prose under docs/src/. These outputs are generated; re-run after
changing a source doc or re-rendering a diagram.

    python3 scripts/inline_docs.py
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "docs" / "src"

# editable source -> generated self-contained doc
TARGETS = [
    (SRC / "CONFLUENCE.md", REPO / "docs" / "CONFLUENCE.md"),
    (SRC / "CONFLUENCE_ONPREM.md", REPO / "docs" / "CONFLUENCE_ONPREM.md"),
    # Long-standing filename for the prototype paste-in; same content as
    # docs/CONFLUENCE.md.
    (SRC / "CONFLUENCE.md", REPO / "VOICE_VIDEO_SEARCH.md"),
]

IMG = re.compile(r"!\[([^\]]*)\]\((?!data:)([^)]+\.png)\)")


def resolve(rel: str) -> Path | None:
    """Image refs are written relative to docs/, where the generated docs land."""
    for base in (REPO / "docs", SRC, REPO):
        candidate = (base / rel.lstrip("./")).resolve()
        if candidate.exists():
            return candidate
    return None


def inline(src: Path, dst: Path) -> None:
    md = src.read_text()
    used = 0

    def sub(m: re.Match[str]) -> str:
        nonlocal used
        alt, rel = m.group(1), m.group(2)
        png = resolve(rel)
        if png is None:
            print(f"  MISSING {rel}")
            return m.group(0)
        payload = base64.b64encode(png.read_bytes()).decode()
        used += 1
        return f"![{alt}](data:image/png;base64,{payload})"

    out = IMG.sub(sub, md)

    banner = (
        "> Generated file — every figure is embedded as base64, so this page "
        "needs no attachments, no Mermaid macro, and no outbound network "
        f"access. Do not edit it; edit `docs/src/{src.name}` and re-run "
        "`python3 scripts/inline_docs.py`.\n\n"
    )
    lines = out.split("\n")
    # Keep the H1 first so Confluence picks up the title.
    head = 1 if lines and lines[0].startswith("# ") else 0
    out = "\n".join(lines[:head]) + ("\n\n" if head else "") + banner + "\n".join(lines[head:])

    dst.write_text(out)
    print(
        f"  {dst.relative_to(REPO)}  {used} images inlined, "
        f"{dst.stat().st_size // 1024} KB"
    )


def main() -> None:
    for src, dst in TARGETS:
        if not src.exists():
            print(f"  skip, missing {src.relative_to(REPO)}")
            continue
        print(f"{src.relative_to(REPO)} ->")
        inline(src, dst)


if __name__ == "__main__":
    main()
