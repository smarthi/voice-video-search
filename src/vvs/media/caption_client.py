"""HTTP client for the caption service (`services/caption_server.py`).

`caption_image` and `expand_query_terms` in `caption.py` call into this
module when `Settings.caption_service_url` is set, instead of loading
Qwen3-VL-2B-Instruct in-process -- so `caption_scenes` and the search-time
query expansion path need no changes to run against a remote captioner.
Only the Qwen path is proxied; the BLIP fallback stays local regardless of
this setting.

Kept import-light: no torch, no PIL model calls, just HTTP.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


class CaptionServiceError(RuntimeError):
    """The caption service was unreachable, errored, or returned a bad shape."""


def remote_caption_image(
    path: str | Path,
    base_url: str,
    *,
    timeout_s: float = 60.0,
    max_side: int = 1024,
) -> tuple[str, list[str], str]:
    """POST one still to `/caption`. Returns (caption, objects, model_id).

    Resizes the same way `_open_fit` does locally before sending, both to
    match what a local caption call would have seen and to keep the
    request small -- a video frame at full resolution is needless bytes
    over the wire for a caption call.
    """
    import io

    import httpx

    with Image.open(path) as im:
        image = im.convert("RGB")
    w, h = image.size
    side = max(w, h)
    if side > max_side:
        scale = max_side / side
        image = image.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR
        )
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    try:
        resp = httpx.post(
            f"{base_url.rstrip('/')}/caption",
            files={"file": (Path(path).name, buf.getvalue(), "image/jpeg")},
            timeout=timeout_s,
        )
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPError as exc:
        raise CaptionServiceError(f"caption service at {base_url} failed: {exc}") from exc
    caption = body.get("caption")
    if caption is None:
        raise CaptionServiceError(f"caption service at {base_url} returned no caption")
    return caption, body.get("objects") or [], body.get("model") or ""


def remote_expand_query_terms(
    query: str,
    base_url: str,
    *,
    max_variants: int | None = None,
    timeout_s: float = 30.0,
) -> list[str]:
    """POST to `/expand_query`. Returns alternate phrasings, original excluded."""
    import httpx

    try:
        resp = httpx.post(
            f"{base_url.rstrip('/')}/expand_query",
            json={"query": query, "max_variants": max_variants},
            timeout=timeout_s,
        )
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPError as exc:
        raise CaptionServiceError(f"caption service at {base_url} failed: {exc}") from exc
    return body.get("variants") or []
