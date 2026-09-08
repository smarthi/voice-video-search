"""Scene captioner (ingest) and query expansion (search), sharing one model.

Primary: Qwen3-VL-2B-Instruct (caption + object list at ingest; text-only
phrasing rewrite at query time). Fallback for captioning only: BLIP-base --
BLIP cannot do text-only generation, so a query-expansion failure has no
fallback and simply skips expansion (see `expand_query_terms`).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from PIL import Image

from vvs.config import load_settings
from vvs.device import (
    empty_cache,
    force_cpu,
    resolve_device,
    resolve_dtype,
    to_device,
    torch_dtype,
)
from vvs.media.scenes import SceneDraft, drafts_to_cards
from vvs.models import Keyframe, SceneCard

QWEN_CAPTION_ID = "Qwen/Qwen3-VL-2B-Instruct"
BLIP_ID = "Salesforce/blip-image-captioning-base"
MAX_SIDE = 768
CAPTION_MAX_TOKENS = 32
QWEN_MAX_TOKENS = 80
EXPAND_MAX_TOKENS = 60
_CAPTION_PROMPT = (
    "Describe this video frame in one short English sentence. "
    "Then on a new line write Objects: comma-separated notable objects."
)
_EXPAND_PROMPT = (
    'A user is searching a video archive for: "{query}"\n'
    "List {n} short alternative phrasings of the visual subject in this query, "
    "using different words or synonyms that a video frame's content might match "
    '(for example, "river" and "running water" describe the same thing). '
    "One phrasing per line. No numbering, no quotes, no commentary, "
    "do not repeat the original wording."
)
_LEADING_MARKER = re.compile(r"^[\s\-*\d.)]+")
_qwen_disabled: str | None = None
# Separate from _qwen_disabled: a text-only rewrite call can fail in a way an
# image+text caption call would not (or vice versa), so one usage failing
# should not silently disable the other for the rest of the process.
_qwen_expand_disabled: str | None = None


CaptionFn = Callable[[str], tuple[str, list[str], str]]


def release_captioner() -> None:
    """Drop the cached captioner and give its device memory back.

    Called at a stage boundary during indexing, where the visual encoder is
    about to want the same card. Also covers the case where this process is a
    long-lived UI that captioned or expanded a query earlier.
    """
    _qwen_bundle.cache_clear()
    _blip_bundle.cache_clear()
    empty_cache()
    print("[caption] released the captioner")


def caption_scenes(
    frames: list[Keyframe],
    drafts: list[SceneDraft],
    *,
    duration_s: float,
    caption_fn: CaptionFn | None = None,
    model_name: str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[SceneCard]:
    """Caption the representative frame of each scene draft."""
    if not frames or not drafts:
        return []
    run = caption_fn or (lambda path: caption_image(path, model_name=model_name))
    by_index = {frame.index: frame for frame in frames}
    rows: list[tuple[str, list[str], str]] = []
    for i, draft in enumerate(drafts):
        frame = by_index.get(draft.frame_index) or frames[draft.start_i]
        caption, objects, model = run(frame.path)
        rows.append((caption, objects, model))
        if on_progress is not None:
            on_progress(i + 1, len(drafts))
        else:
            print(f"[caption] scene {i + 1}/{len(drafts)} t={frame.t_s:.1f}s {caption[:80]}")
    return drafts_to_cards(frames, drafts, duration_s=duration_s, captions=rows)


def caption_image(path: str | Path, *, model_name: str | None = None) -> tuple[str, list[str], str]:
    """Return (caption, objects, model_id) for one still."""
    global _qwen_disabled
    preferred = (model_name or load_settings().scene_caption_model or "qwen_vl").strip()
    image = _open_fit(path)
    key = preferred.lower().replace("_", "-")
    if _want_qwen(key) and _qwen_disabled is None:
        try:
            return _qwen_caption(image)
        except Exception as exc:
            _qwen_disabled = str(exc)
            print(f"[caption] Qwen-VL Instruct failed ({exc}); falling back to BLIP for this run")
    return _blip_caption(image)


def _want_qwen(name: str) -> bool:
    key = (name or "").lower().replace("_", "-")
    if key in {"qwen", "qwen-vl", "qwen3", "qwen3-vl", "qwen-vl-instruct"}:
        return True
    return "qwen3-vl" in key and "instruct" in key


def parse_qwen_caption(text: str) -> tuple[str, list[str]]:
    """Split Instruct output into (caption, objects). Used by tests without a VLM."""
    raw = (text or "").strip()
    if not raw:
        return "A video frame.", []
    lower = raw.lower()
    caption = raw
    objects: list[str] = []
    for marker in ("\nobjects:", "\nobject:"):
        idx = lower.find(marker)
        if idx >= 0:
            caption = raw[:idx].strip()
            rest = raw[idx + len(marker) :].strip()
            objects = _unique_labels(part.strip(" .") for part in rest.replace(";", ",").split(","))
            break
    else:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if len(lines) >= 2 and lines[-1].lower().startswith("object"):
            caption = " ".join(lines[:-1])
            rest = lines[-1].split(":", 1)[-1]
            objects = _unique_labels(part.strip(" .") for part in rest.replace(";", ",").split(","))
    if caption.lower().startswith("caption:"):
        caption = caption.split(":", 1)[1].strip()
    return caption or "A video frame.", objects


@lru_cache(maxsize=1)
def _qwen_bundle():
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(QWEN_CAPTION_ID)

    def load(device: str, dtype: str):
        print(f"[caption] loading {QWEN_CAPTION_ID} ({device}, {dtype}, Qwen3-VL Instruct)")
        model = AutoModelForImageTextToText.from_pretrained(
            QWEN_CAPTION_ID,
            dtype=torch_dtype(dtype),
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        return model

    device = resolve_device()
    try:
        return load(device, resolve_dtype(device)), processor
    except Exception as exc:
        if device == "cpu":
            raise
        print(f"[caption] {device} load failed ({type(exc).__name__}: {exc}); using cpu")
        force_cpu()
        return load("cpu", resolve_dtype("cpu")), processor


def _qwen_caption(image: Image.Image) -> tuple[str, list[str], str]:
    import torch

    model, processor = _qwen_bundle()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _CAPTION_PROMPT},
            ],
        }
    ]
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception:
        from qwen_vl_utils import process_vision_info

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    inputs = to_device(inputs, resolve_device())
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=QWEN_MAX_TOKENS,
            do_sample=False,
            use_cache=True,
        )
    in_len = int(inputs["input_ids"].shape[1])
    trimmed = generated[:, in_len:]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    caption, objects = parse_qwen_caption(decoded)
    return caption, objects, QWEN_CAPTION_ID


def expand_query_terms(query: str, *, max_variants: int | None = None) -> list[str]:
    """Alternate phrasings of `query`'s visual subject, for a wider ANN net.

    Dual encoders like Qwen-VL are lexically brittle: "river" and "running
    water" can land in noticeably different places in embedding space even
    though they mean the same thing to a person. `retrieve/search.py` embeds
    each returned phrasing too and merges the candidates by max score, so a
    weak original phrasing does not cost the whole query -- the same shape as
    `media/motion.py`'s query-time heuristics, just backed by a model instead
    of a keyword set.

    Never raises and never blocks a search on a slow/broken model for more
    than one attempt: returns `[]` (search proceeds on the literal query
    alone) if expansion is off, the model is unavailable, or generation
    produced nothing usable. There is no fallback model here the way BLIP
    backs up captioning -- BLIP cannot do text-only generation.
    """
    global _qwen_expand_disabled
    query = (query or "").strip()
    if not query or _qwen_expand_disabled is not None:
        return []
    settings = load_settings()
    if not settings.query_expand:
        return []
    n = max_variants if max_variants is not None else settings.query_expand_variants
    if n <= 0:
        return []
    try:
        decoded = _qwen_rewrite(_EXPAND_PROMPT.format(query=query, n=n))
    except Exception as exc:
        _qwen_expand_disabled = str(exc)
        print(f"[query] expansion failed ({exc}); searching the literal query only")
        return []
    return _parse_expansion_lines(decoded, query, n)


def _qwen_rewrite(prompt: str) -> str:
    """Text-only generation through the same Qwen3-VL-Instruct bundle captioning uses."""
    import torch

    model, processor = _qwen_bundle()
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = to_device(inputs, resolve_device())
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=EXPAND_MAX_TOKENS,
            do_sample=False,
            use_cache=True,
        )
    in_len = int(inputs["input_ids"].shape[1])
    trimmed = generated[:, in_len:]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def _parse_expansion_lines(text: str, original: str, max_variants: int) -> list[str]:
    """Split model output into deduped alternate phrasings. Used by tests without a VLM."""
    seen = {original.strip().lower()}
    out: list[str] = []
    for raw_line in (text or "").splitlines():
        line = _LEADING_MARKER.sub("", raw_line).strip().strip("\"'").strip()
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
        if len(out) >= max_variants:
            break
    return out


@lru_cache(maxsize=1)
def _blip_bundle():
    from transformers import BlipForConditionalGeneration, BlipProcessor

    device = resolve_device()
    print(f"[caption] loading {BLIP_ID} ({device})")
    processor = BlipProcessor.from_pretrained(BLIP_ID)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_ID)
    model.to(device)
    model.eval()
    return model, processor


def _blip_caption(image: Image.Image) -> tuple[str, list[str], str]:
    import torch

    model, processor = _blip_bundle()
    inputs = to_device(processor(images=image, return_tensors="pt"), resolve_device())
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=CAPTION_MAX_TOKENS, num_beams=1)
    text = processor.decode(out[0], skip_special_tokens=True).strip()
    return text or "A video frame.", [], BLIP_ID


def _open_fit(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        image = im.convert("RGB")
        w, h = image.size
        side = max(w, h)
        if side <= MAX_SIDE:
            return image.copy()
        scale = MAX_SIDE / side
        return image.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.BILINEAR,
        )


def _unique_labels(labels) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in labels:
        label = str(raw).strip().lower()
        if not label or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out[:12]
