"""Ingest-only scene captioner. Query time never loads this model.

Primary: Qwen3-VL-2B-Instruct (caption + object list). Fallback: BLIP-base.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from PIL import Image

from vvs.config import load_settings
from vvs.device import force_cpu, resolve_device, resolve_dtype, to_device, torch_dtype
from vvs.media.scenes import SceneDraft, drafts_to_cards
from vvs.models import Keyframe, SceneCard

QWEN_CAPTION_ID = "Qwen/Qwen3-VL-2B-Instruct"
BLIP_ID = "Salesforce/blip-image-captioning-base"
MAX_SIDE = 768
CAPTION_MAX_TOKENS = 32
QWEN_MAX_TOKENS = 80
_CAPTION_PROMPT = (
    "Describe this video frame in one short English sentence. "
    "Then on a new line write Objects: comma-separated notable objects."
)
_qwen_disabled: str | None = None


CaptionFn = Callable[[str], tuple[str, list[str], str]]


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
