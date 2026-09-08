"""Produce paste-ready Markdown for Confluence, in two flavours per document.

The target Confluence has no Mermaid macro and no outbound network access, and
pasted Markdown cannot resolve relative image paths. So each source doc yields:

    docs/<NAME>.md          every PNG embedded as a base64 data URI, so the
                            file carries its own figures and needs nothing else

    docs/<NAME>_ATTACH.md   every image reference reduced to a bare filename,
                            for Confluence instances that strip `data:` URIs
                            (common, and the usual reason figures come out
                            blank). Attach the listed PNGs to the page first;
                            Confluence resolves a bare filename to an
                            attachment on the same page.

Try the inlined file first — it is one paste with no attachments. If the
figures do not render, use the _ATTACH file.

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
    (SRC / "CONFLUENCE_ONPREM.md", REPO / "docs" / "CONFLUENCE_ONPREM.md"),
]

IMG = re.compile(r"!\[([^\]]*)\]\((?!data:)([^)]+\.png)\)")


def resolve(rel: str) -> Path | None:
    """Image refs are written relative to docs/, where the generated docs land."""
    for base in (REPO / "docs", SRC, REPO):
        candidate = (base / rel.lstrip("./")).resolve()
        if candidate.exists():
            return candidate
    return None


def _with_banner(out: str, banner: str) -> str:
    lines = out.split("\n")
    # Keep the H1 first so Confluence picks up the title.
    head = 1 if lines and lines[0].startswith("# ") else 0
    return "\n".join(lines[:head]) + ("\n\n" if head else "") + banner + "\n".join(lines[head:])


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
        "`python3 scripts/inline_docs.py`.\n>\n"
        "> If the figures render blank, this Confluence strips `data:` image "
        f"URIs — use `docs/{dst.stem}_ATTACH.md` instead.\n\n"
    )
    dst.write_text(_with_banner(out, banner))
    print(
        f"  {dst.relative_to(REPO)}  {used} images inlined, "
        f"{dst.stat().st_size // 1024} KB"
    )


def attach(src: Path, dst: Path) -> None:
    """Bare-filename image refs, for a Confluence that strips data: URIs."""
    md = src.read_text()
    files: list[Path] = []

    def sub(m: re.Match[str]) -> str:
        alt, rel = m.group(1), m.group(2)
        png = resolve(rel)
        if png is None:
            print(f"  MISSING {rel}")
            return m.group(0)
        files.append(png)
        return f"![{alt}]({png.name})"

    out = IMG.sub(sub, md)

    listing = "\n".join(f">   - `{p.relative_to(REPO)}`" for p in files)
    banner = (
        "> Generated file — **attach these PNGs to the Confluence page before "
        "pasting**, so each bare filename below resolves to an attachment:\n>\n"
        f"{listing}\n>\n"
        f"> Do not edit it; edit `docs/src/{src.name}` and re-run "
        "`python3 scripts/inline_docs.py`.\n\n"
    )
    dst.write_text(_with_banner(out, banner))
    print(
        f"  {dst.relative_to(REPO)}  {len(files)} images by filename, "
        f"{dst.stat().st_size // 1024} KB"
    )


def main() -> None:
    for src, dst in TARGETS:
        if not src.exists():
            print(f"  skip, missing {src.relative_to(REPO)}")
            continue
        print(f"{src.relative_to(REPO)} ->")
        inline(src, dst)
        attach(src, dst.with_name(f"{dst.stem}_ATTACH.md"))


if __name__ == "__main__":
    main()
