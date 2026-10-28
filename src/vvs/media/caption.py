"""Scene captioner (ingest) and query expansion (search), sharing one model.

Primary: Gemma 4 E4B Instruct (caption + object list at ingest; text-only
phrasing rewrite at query time). Thinking is off: these calls are short.
Fallback for captioning only: BLIP-base. BLIP cannot do text-only generation,
so a query-expansion failure has no fallback and simply skips expansion
(see `expand_query_terms`).
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
    to_device,
    torch_dtype,
)
from vvs.media.scenes import SceneDraft, drafts_to_cards
from vvs.models import Keyframe, SceneCard

GEMMA_CAPTION_ID = "google/gemma-4-E4B-it"
BLIP_ID = "Salesforce/blip-image-captioning-base"
MAX_SIDE = 768
CAPTION_MAX_TOKENS = 32
GEMMA_MAX_TOKENS = 80
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
_gemma_disabled: str | None = None
# Separate from _gemma_disabled: a text-only rewrite call can fail in a way an
# image+text caption call would not (or vice versa), so one usage failing
# should not silently disable the other for the rest of the process.
_gemma_expand_disabled: str | None = None


CaptionFn = Callable[[str], tuple[str, list[str], str]]


def release_captioner() -> None:
    """Drop the cached captioner and give its device memory back.

    Called at a stage boundary during indexing, where the visual encoder is
    about to want the same card. Also covers the case where this process is a
    long-lived UI that captioned or expanded a query earlier.
    """
    _gemma_bundle.cache_clear()
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
    global _gemma_disabled
    preferred = (model_name or load_settings().scene_caption_model or "gemma4").strip()
    image = _open_fit(path)
    if _want_gemma(preferred) and _gemma_disabled is None:
        try:
            return _gemma_caption(image)
        except Exception as exc:
            _gemma_disabled = str(exc)
            print(f"[caption] Gemma 4 failed ({exc}); falling back to BLIP for this run")
    return _blip_caption(image)


def _want_gemma(name: str) -> bool:
    """Anything other than an explicit BLIP request uses Gemma 4.

    `qwen_vl` is included so an old `VVS_SCENE_CAPTION_MODEL` value does not
    silently drop to BLIP.
    """
    key = (name or "").lower().replace("_", "-")
    return key not in {"blip", "blip-base"} and "blip" not in key


def parse_scene_caption(text: str) -> tuple[str, list[str]]:
    """Split captioner output into (caption, objects). Used by tests without a VLM."""
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


parse_qwen_caption = parse_scene_caption


def _caption_dtype(device: str) -> str:
    """bfloat16 on CUDA, float32 elsewhere. Gemma 4 is not run in float16."""
    dev = (device or "cpu").strip().lower()
    if dev == "cuda":
        import torch

        if torch.cuda.is_bf16_supported():
            return "bfloat16"
    return "float32"


@lru_cache(maxsize=1)
def _gemma_bundle():
    from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(GEMMA_CAPTION_ID)
    config = AutoConfig.from_pretrained(GEMMA_CAPTION_ID)
    # Captions and query rewrites never use audio. Dropping the tower keeps
    # the 8 GB card for the text and vision weights.
    config.audio_config = None

    def load(device: str, dtype: str):
        print(f"[caption] loading {GEMMA_CAPTION_ID} ({device}, {dtype})")
        model = AutoModelForMultimodalLM.from_pretrained(
            GEMMA_CAPTION_ID,
            config=config,
            dtype=torch_dtype(dtype),
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        return model

    device = resolve_device()
    try:
        return load(device, _caption_dtype(device)), processor
    except Exception as exc:
        if device == "cpu":
            raise
        print(f"[caption] {device} load failed ({type(exc).__name__}: {exc}); using cpu")
        force_cpu()
        return load("cpu", _caption_dtype("cpu")), processor


def _gemma_generate(messages: list[dict], max_new_tokens: int) -> str:
    import torch

    model, processor = _gemma_bundle()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    inputs = to_device(inputs, resolve_device())
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    in_len = int(inputs["input_ids"].shape[1])
    trimmed = generated[:, in_len:]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    parsed = processor.parse_response(decoded)
    if isinstance(parsed, dict):
        content = parsed.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return decoded.strip()


def _gemma_caption(image: Image.Image) -> tuple[str, list[str], str]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _CAPTION_PROMPT},
            ],
        }
    ]
    decoded = _gemma_generate(messages, GEMMA_MAX_TOKENS)
    caption, objects = parse_scene_caption(decoded)
    return caption, objects, GEMMA_CAPTION_ID


def expand_query_terms(query: str, *, max_variants: int | None = None) -> list[str]:
    """Alternate phrasings of `query`'s visual subject, for a wider ANN net.

    Embedding models are lexically brittle: "river" and "running
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
    global _gemma_expand_disabled
    query = (query or "").strip()
    if not query or _gemma_expand_disabled is not None:
        return []
    settings = load_settings()
    if not settings.query_expand:
        return []
    n = max_variants if max_variants is not None else settings.query_expand_variants
    if n <= 0:
        return []
    try:
        decoded = _gemma_rewrite(_EXPAND_PROMPT.format(query=query, n=n))
    except Exception as exc:
        _gemma_expand_disabled = str(exc)
        print(f"[query] expansion failed ({exc}); searching the literal query only")
        return []
    return _parse_expansion_lines(decoded, query, n)


def _gemma_rewrite(prompt: str) -> str:
    """Text-only generation through the same Gemma 4 bundle captioning uses."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return _gemma_generate(messages, EXPAND_MAX_TOKENS)


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
