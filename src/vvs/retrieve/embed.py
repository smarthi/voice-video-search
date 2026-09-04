"""Visual encoders (Qwen3-VL-Embedding, WeMM-Embedding, open_clip) plus BGE text.

Device is auto-detected: Metal (`mps`) on Apple Silicon, then CUDA, then CPU.
`VVS_DEVICE` overrides it. This matters a lot — a 2B VL encoder on CPU costs tens
of seconds per frame, and the same model on an M-series GPU is an order of
magnitude quicker.

Official Qwen VL embedding sizes are 2B (2048-d) and 8B (4096-d). There is no
4B VL embedder — Qwen3-Embedding-4B is text-only. `qwen-vl-emb-4B` aliases to 2B.

WeMM-Embedding (Tencent, Apache-2.0) is a drop-in alternative built on Qwen3.5.
It is Matryoshka-trained, so its output may be truncated to any dimension listed
in `_WEMM_MRL_DIMS` and renormalized. Note that WeMM-2B and Qwen-VL-2B are both
2048-d: vector width alone cannot identify which model wrote an index, which is
why `ModelCard.visual_encoder` is recorded and `encoder_kind` prefers it.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from vvs.config import load_settings
from vvs.device import force_cpu, resolve_device, resolve_dtype, torch_dtype

QWEN_VL_2B = "Qwen/Qwen3-VL-Embedding-2B"
QWEN_VL_8B = "Qwen/Qwen3-VL-Embedding-8B"
WEMM_2B = "tencent/WeMM-Embedding-2B"
WEMM_4B = "tencent/WeMM-Embedding-4B"
WEMM_9B = "tencent/WeMM-Embedding-9B"
CLIP_NAME = "open_clip ViT-B-32 laion2b_s34b_b79k"

_QUERY_PROMPT = "Retrieve video frames that match this query."
_FRAME_PROMPT = "Represent the video frame for retrieval."
# WeMM takes the instruction as a `text` item alongside the image, and supplies
# the query-side instruction itself via `encode_query`. Wording follows the model
# card so we stay inside the distribution it was tuned on.
_WEMM_FRAME_PROMPT = "Represent this image."
_WEMM_VIDEO_PROMPT = "Represent this video."

ENCODER_KINDS = ("qwen_vl", "wemm", "clip")

_WEMM_NATIVE_DIM = {WEMM_2B: 2048, WEMM_4B: 2560, WEMM_9B: 4096}
_WEMM_MRL_DIMS = {
    WEMM_2B: (64, 128, 256, 512, 1024, 2048),
    WEMM_4B: (64, 128, 256, 512, 1024, 2560),
    WEMM_9B: (64, 128, 256, 512, 1024, 2048, 4096),
}

_MODEL_ALIASES = {
    "qwen-vl-emb-4b": QWEN_VL_2B,
    "qwen-vl-emb-4B": QWEN_VL_2B,
    "qwen-vl-emb-2b": QWEN_VL_2B,
    "qwen-vl-emb-8b": QWEN_VL_8B,
    "2b": QWEN_VL_2B,
    "8b": QWEN_VL_8B,
    "qwen": QWEN_VL_2B,
    "qwen_vl": QWEN_VL_2B,
    "wemm": WEMM_2B,
    "wemm_2b": WEMM_2B,
    "wemm-2b": WEMM_2B,
    "wemm-embedding-2b": WEMM_2B,
    "wemm_4b": WEMM_4B,
    "wemm-4b": WEMM_4B,
    "wemm-embedding-4b": WEMM_4B,
    "wemm_9b": WEMM_9B,
    "wemm-9b": WEMM_9B,
    "wemm-embedding-9b": WEMM_9B,
}


def resolve_visual_model(name: str | None) -> str:
    raw = (name or QWEN_VL_2B).strip()
    mapped = _MODEL_ALIASES.get(raw.lower())
    if mapped:
        if "4b" in raw.lower() and mapped in (QWEN_VL_2B, QWEN_VL_8B):
            print(
                "[embed] no official Qwen-VL-Embedding-4B (VL family is 2B / 8B); "
                f"using {mapped}"
            )
        return mapped
    if "wemm" in raw.lower():
        return raw
    if "vl" in raw.lower() and "4b" in raw.lower():
        print(f"[embed] {raw!r} is not a published VL embedder; using {QWEN_VL_2B}")
        return QWEN_VL_2B
    return raw


def kind_from_model(name: str | None) -> str | None:
    """Encoder family implied by a model id, or None when unrecognizable.

    WeMM is checked before Qwen because WeMM is built on a Qwen backbone and its
    description mentions Qwen even though its repo id does not.
    """
    lower = (name or "").strip().lower()
    if not lower:
        return None
    if "wemm" in lower:
        return "wemm"
    if "qwen" in lower:
        return "qwen_vl"
    if "open_clip" in lower or "vit-b-32" in lower or lower == "clip":
        return "clip"
    return None


def _dims_for_kind(kind: str, name: str) -> set[int]:
    """Widths an index written by `kind` may legitimately have."""
    if kind == "clip":
        return {512}
    if kind == "wemm":
        model = resolve_visual_model(name)
        return set(_WEMM_MRL_DIMS.get(model, ())) or {_dim_from_name(name)}
    return {2048, 4096}


def normalize_encoder_kind(value: str | None) -> str:
    raw = (value or "qwen_vl").strip().lower()
    if raw in ENCODER_KINDS:
        return raw
    if raw in ("qwen", "qwenvl", "qwen-vl"):
        return "qwen_vl"
    if raw.startswith("wemm"):
        return "wemm"
    if raw.startswith(("clip", "open_clip")):
        return "clip"
    return "qwen_vl"


def encoder_kind(*, meta=None, settings=None) -> str:
    """Which visual encoder wrote (or will write) this index.

    Resolution order matters. Vector width is *not* a reliable discriminator
    because WeMM-2B and Qwen-VL-2B are both 2048-d, so trusting the width would
    silently query a WeMM index with the Qwen encoder. But width is still needed
    to heal old indexes whose `meta.json` records a stale CLIP label after a Qwen
    rewrite. The compromise: trust the explicit field, then the model name, and
    only fall back to width when the name disagrees with the stored width.
    """
    if meta is not None:
        card = getattr(meta, "models", None)
        explicit = normalize_encoder_kind(getattr(card, "visual_encoder", None) or "") \
            if (getattr(card, "visual_encoder", None) or "").strip() else None
        if explicit:
            return explicit
        name = getattr(card, "clip", "") or ""
        dim = getattr(card, "visual_dim", None)
        named = kind_from_model(name)
        if named and (not dim or int(dim) in _dims_for_kind(named, name)):
            return named
        if dim:
            return "clip" if int(dim) == 512 else "qwen_vl"
        if named:
            return named
    cfg = settings or load_settings()
    return normalize_encoder_kind(cfg.visual_encoder)


def visual_dim(*, settings=None, meta=None) -> int:
    """FAISS dim for the visual collection. Must match the encoder that wrote it."""
    if meta is not None:
        stored = getattr(getattr(meta, "models", None), "visual_dim", None)
        if stored:
            return int(stored)
        return _dim_from_name(getattr(meta.models, "clip", "") or "")
    cfg = settings or load_settings()
    kind = normalize_encoder_kind(cfg.visual_encoder)
    if kind == "clip":
        return 512
    model = resolve_visual_model(cfg.visual_model)
    if kind == "wemm":
        return _wemm_dim(model, cfg.visual_truncate_dim)
    return _dim_from_name(model)


def _wemm_dim(model: str, truncate_dim: int | None) -> int:
    native = _WEMM_NATIVE_DIM.get(model) or _dim_from_name(model)
    if not truncate_dim:
        return native
    allowed = _WEMM_MRL_DIMS.get(model)
    if allowed and int(truncate_dim) not in allowed:
        raise SystemExit(
            f"VVS_VISUAL_TRUNCATE_DIM={truncate_dim} is not a Matryoshka dimension for "
            f"{model}. Choose one of {', '.join(str(d) for d in allowed)}."
        )
    return int(truncate_dim)


def visual_model_label(*, settings=None) -> str:
    cfg = settings or load_settings()
    if normalize_encoder_kind(cfg.visual_encoder) == "clip":
        return CLIP_NAME
    return resolve_visual_model(cfg.visual_model)


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return (mat / np.clip(norms, eps, None)).astype(np.float32)


def _dim_from_name(name: str) -> int:
    lower = (name or "").lower()
    if "wemm" in lower:
        native = _WEMM_NATIVE_DIM.get(_MODEL_ALIASES.get(lower, name))
        if native:
            return native
        if "9b" in lower:
            return 4096
        if "4b" in lower:
            return 2560
        return 2048
    if "8b" in lower or "4096" in lower:
        return 4096
    if "vit-b-32" in lower or "open_clip" in lower or lower == "clip":
        return 512
    return 2048


def default_image_batch(device: str | None = None) -> int:
    """Frames per forward pass.

    Batch 1 keeps peak memory tiny on CPU, where each frame is slow anyway. On a
    GPU the per-call overhead dominates, so a wider batch is most of the win.
    """
    override = (os.environ.get("VVS_VISUAL_BATCH") or "").strip()
    if override.isdigit() and int(override) > 0:
        return int(override)
    return 1 if (device or resolve_device()) == "cpu" else 8


@lru_cache(maxsize=1)
def _clip_bundle():
    import open_clip
    import torch

    torch.set_num_threads(max(1, min(16, os.cpu_count() or 8)))
    device = resolve_device()
    print(f"[embed] loading {CLIP_NAME} ({device})")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=device,
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    return model, preprocess, tokenizer


# maxsize=2 so an A/B run can hold Qwen and WeMM at once instead of reloading
# a multi-GB model on every alternating call.
@lru_cache(maxsize=2)
def _sentence_transformer(model_id: str) -> object:
    # The Metal CPU-fallback switch is armed in vvs/__init__.py; it has to be set
    # before torch is imported, so it cannot live here.
    device = resolve_device()
    try:
        return _load_st(model_id, device, resolve_dtype(device))
    except Exception as exc:
        if device == "cpu":
            raise
        # A GPU that cannot load the model is worth abandoning for the whole
        # session, not once per call: CPU is slow but it always works. Pinning
        # the choice also keeps hand-placed tensors (the CLIP path) on the same
        # device as the weights.
        print(f"[embed] {device} load failed ({type(exc).__name__}: {exc}); using cpu")
        force_cpu()
        return _load_st(model_id, "cpu", resolve_dtype("cpu"))


def _load_st(model_id: str, device: str, dtype: str) -> object:
    from sentence_transformers import SentenceTransformer

    print(f"[embed] loading {model_id} ({device}, {dtype})")
    model = SentenceTransformer(
        model_id,
        device=device,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch_dtype(dtype)},
    )
    if dtype != "float32" and not _probe_finite(model):
        # Half precision overflows in some VL stacks and yields NaN vectors,
        # which would silently poison an index. Rebuild in fp32 instead.
        print(f"[embed] {dtype} produced non-finite vectors; reloading in float32")
        model = _load_st(model_id, device, "float32")
    return model


def _probe_finite(model) -> bool:
    """Encode one throwaway string and check the result is usable."""
    try:
        encode = getattr(model, "encode_query", None) or model.encode
        out = np.asarray(
            encode(["probe"], convert_to_numpy=True, show_progress_bar=False),
            dtype=np.float32,
        )
    except Exception as exc:
        print(f"[embed] dtype probe failed ({type(exc).__name__}: {exc}); assuming ok")
        return True
    return bool(out.size) and bool(np.isfinite(out).all()) and float(np.abs(out).max()) > 0


def _qwen_vl(model_id: str):
    return _sentence_transformer(model_id)


def _wemm(model_id: str):
    return _sentence_transformer(model_id)


@lru_cache(maxsize=1)
def _bge():
    from sentence_transformers import SentenceTransformer

    # Deliberately CPU. This is a 33M-parameter 384-d model that was never the
    # bottleneck, so a GPU buys it nothing measurable, while loading a third
    # model into the Metal context alongside the 2B embedder and the 2B
    # captioner is pure added risk. `VVS_BGE_DEVICE` exists for benchmarking.
    device = (os.environ.get("VVS_BGE_DEVICE") or "cpu").strip().lower()
    print(f"[embed] loading BAAI/bge-small-en-v1.5 ({device})")
    return SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)


def _model_id_for(*, meta=None, settings=None) -> str:
    if meta is not None:
        name = getattr(getattr(meta, "models", None), "clip", "") or ""
        if name and encoder_kind(meta=meta) in ("qwen_vl", "wemm"):
            return resolve_visual_model(name)
    cfg = settings or load_settings()
    return resolve_visual_model(cfg.visual_model)


def _truncate_dim_for(*, meta=None, settings=None) -> int | None:
    """Matryoshka width to request. A stored index's own width always wins."""
    if meta is not None:
        stored = getattr(getattr(meta, "models", None), "visual_dim", None)
        if stored:
            model = _model_id_for(meta=meta, settings=settings)
            native = _WEMM_NATIVE_DIM.get(model)
            return None if native and int(stored) == native else int(stored)
    cfg = settings or load_settings()
    return cfg.visual_truncate_dim or None


def warm_models() -> None:
    """Force-load visual encoder, BGE, and Whisper so the live demo pays no first-query tax."""
    embed_clip_texts(["a frame from a video"])
    _bge()
    from vvs.media.transcribe import get_whisper

    get_whisper()
    embed_bge_texts(["a spoken sentence from a video"])


def embed_clip_images(
    paths: list[str] | list[Path],
    batch_size: int | None = None,
    *,
    meta=None,
    settings=None,
) -> np.ndarray:
    """Index-time visual embeddings. Name kept for callers; backend is configurable."""
    cfg = settings or load_settings()
    dim = visual_dim(settings=cfg, meta=meta)
    if not paths:
        return np.zeros((0, dim), dtype=np.float32)
    kind = encoder_kind(meta=meta, settings=cfg)
    if kind == "clip":
        return _embed_clip_images(paths, batch_size=batch_size or 16)
    vl_batch = batch_size or default_image_batch()
    if kind == "wemm":
        return _embed_wemm_images(paths, batch_size=vl_batch, meta=meta, settings=cfg)
    return _embed_qwen_images(paths, batch_size=vl_batch, meta=meta, settings=cfg)


def embed_clip_texts(texts: list[str], *, meta=None, settings=None) -> np.ndarray:
    """Query-time visual-space text embeddings. Pass meta so old CLIP indexes still search."""
    cfg = settings or load_settings()
    dim = visual_dim(settings=cfg, meta=meta)
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    kind = encoder_kind(meta=meta, settings=cfg)
    if kind == "clip":
        return _embed_clip_texts(texts)
    if kind == "wemm":
        return _embed_wemm_texts(texts, meta=meta, settings=cfg)
    return _embed_qwen_texts(texts, meta=meta, settings=cfg)


def _embed_clip_images(paths: list[str] | list[Path], batch_size: int) -> np.ndarray:
    import torch

    model, preprocess, _tokenizer = _clip_bundle()
    vecs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            images = []
            for p in batch_paths:
                with Image.open(p) as im:
                    images.append(preprocess(im.convert("RGB")))
            tensor = torch.stack(images, dim=0).to(resolve_device())
            feats = model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            vecs.append(feats.cpu().numpy().astype(np.float32))
            print(f"[embed] clip images {min(i + batch_size, len(paths))}/{len(paths)}")
    return np.vstack(vecs) if vecs else np.zeros((0, 512), dtype=np.float32)


def _embed_clip_texts(texts: list[str]) -> np.ndarray:
    import torch

    model, _preprocess, tokenizer = _clip_bundle()
    tokens = tokenizer(texts).to(resolve_device())
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)


def _embed_qwen_images(
    paths: list[str] | list[Path],
    batch_size: int,
    *,
    meta=None,
    settings=None,
) -> np.ndarray:
    model_id = _model_id_for(meta=meta, settings=settings)
    model = _qwen_vl(model_id)
    dim = visual_dim(settings=settings, meta=meta)
    vecs: list[np.ndarray] = []
    for i in range(0, len(paths), batch_size):
        batch = [{"image": str(p)} for p in paths[i : i + batch_size]]
        out = model.encode(
            batch,
            prompt=_FRAME_PROMPT,
            batch_size=len(batch),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vecs.append(np.asarray(out, dtype=np.float32))
        print(f"[embed] qwen-vl images {min(i + batch_size, len(paths))}/{len(paths)}")
    return np.vstack(vecs) if vecs else np.zeros((0, dim), dtype=np.float32)


def _embed_qwen_texts(texts: list[str], *, meta=None, settings=None) -> np.ndarray:
    model_id = _model_id_for(meta=meta, settings=settings)
    model = _qwen_vl(model_id)
    out = model.encode(
        list(texts),
        prompt=_QUERY_PROMPT,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(out, dtype=np.float32)


def _embed_wemm_images(
    paths: list[str] | list[Path],
    batch_size: int,
    *,
    meta=None,
    settings=None,
) -> np.ndarray:
    model_id = _model_id_for(meta=meta, settings=settings)
    model = _wemm(model_id)
    dim = visual_dim(settings=settings, meta=meta)
    truncate = _truncate_dim_for(meta=meta, settings=settings)
    vecs: list[np.ndarray] = []
    for i in range(0, len(paths), batch_size):
        # Image before text: WeMM's chat template expects that order.
        batch = [
            {"image": str(p), "text": _WEMM_FRAME_PROMPT} for p in paths[i : i + batch_size]
        ]
        out = model.encode_document(
            batch,
            batch_size=len(batch),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            **({"truncate_dim": truncate} if truncate else {}),
        )
        vecs.append(np.asarray(out, dtype=np.float32))
        print(f"[embed] wemm images {min(i + batch_size, len(paths))}/{len(paths)}")
    return np.vstack(vecs) if vecs else np.zeros((0, dim), dtype=np.float32)


def embed_wemm_videos(
    paths: list[str] | list[Path],
    *,
    batch_size: int = 1,
    meta=None,
    settings=None,
) -> np.ndarray:
    """Embed whole clips rather than stills. WeMM accepts native video input.

    This is the channel WeMM's MMEB-v2 video score is measured on; the still-frame
    path above only exercises its image tower. Kept separate from
    `embed_clip_images` because callers must supply clip paths, not keyframes.
    """
    cfg = settings or load_settings()
    if encoder_kind(meta=meta, settings=cfg) != "wemm":
        raise SystemExit("embed_wemm_videos requires VVS_VISUAL_ENCODER=wemm")
    model_id = _model_id_for(meta=meta, settings=cfg)
    model = _wemm(model_id)
    dim = visual_dim(settings=cfg, meta=meta)
    truncate = _truncate_dim_for(meta=meta, settings=cfg)
    vecs: list[np.ndarray] = []
    for i in range(0, len(paths), batch_size):
        batch = [
            {"video": str(p), "text": _WEMM_VIDEO_PROMPT} for p in paths[i : i + batch_size]
        ]
        out = model.encode_document(
            batch,
            batch_size=len(batch),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            **({"truncate_dim": truncate} if truncate else {}),
        )
        vecs.append(np.asarray(out, dtype=np.float32))
        print(f"[embed] wemm videos {min(i + batch_size, len(paths))}/{len(paths)}")
    return np.vstack(vecs) if vecs else np.zeros((0, dim), dtype=np.float32)


def _embed_wemm_texts(texts: list[str], *, meta=None, settings=None) -> np.ndarray:
    model_id = _model_id_for(meta=meta, settings=settings)
    model = _wemm(model_id)
    truncate = _truncate_dim_for(meta=meta, settings=settings)
    # encode_query applies WeMM's own query-side instruction, so no prompt here.
    out = model.encode_query(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        **({"truncate_dim": truncate} if truncate else {}),
    )
    return np.asarray(out, dtype=np.float32)


def embed_bge_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    prefixed = [f"query: {t}" if _looks_like_query(t) else f"passage: {t}" for t in texts]
    model = _bge()
    vecs = model.encode(
        prefixed,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 16,
    )
    return np.asarray(vecs, dtype=np.float32)


def embed_bge_query(text: str) -> np.ndarray:
    model = _bge()
    vec = model.encode(
        [f"query: {text}"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vec, dtype=np.float32)


def _looks_like_query(text: str) -> bool:
    # Index-time texts are passages; live search uses embed_bge_query.
    return False
