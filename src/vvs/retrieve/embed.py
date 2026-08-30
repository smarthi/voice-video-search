"""Visual encoder (Qwen3-VL-Embedding by default) plus BGE text. CPU only.

Official Qwen VL embedding sizes are 2B (2048-d) and 8B (4096-d). There is no
4B VL embedder — Qwen3-Embedding-4B is text-only. `qwen-vl-emb-4B` aliases to 2B.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from vvs.config import load_settings

QWEN_VL_2B = "Qwen/Qwen3-VL-Embedding-2B"
QWEN_VL_8B = "Qwen/Qwen3-VL-Embedding-8B"
CLIP_NAME = "open_clip ViT-B-32 laion2b_s34b_b79k"
_QUERY_PROMPT = "Retrieve video frames that match this query."
_FRAME_PROMPT = "Represent the video frame for retrieval."

_MODEL_ALIASES = {
    "qwen-vl-emb-4b": QWEN_VL_2B,
    "qwen-vl-emb-4B": QWEN_VL_2B,
    "qwen-vl-emb-2b": QWEN_VL_2B,
    "qwen-vl-emb-8b": QWEN_VL_8B,
    "2b": QWEN_VL_2B,
    "8b": QWEN_VL_8B,
    "qwen": QWEN_VL_2B,
    "qwen_vl": QWEN_VL_2B,
}


def resolve_visual_model(name: str | None) -> str:
    raw = (name or QWEN_VL_2B).strip()
    mapped = _MODEL_ALIASES.get(raw.lower())
    if mapped:
        if "4b" in raw.lower():
            print(
                "[embed] no official Qwen-VL-Embedding-4B (VL family is 2B / 8B); "
                f"using {mapped}"
            )
        return mapped
    if "vl" in raw.lower() and "4b" in raw.lower():
        print(f"[embed] {raw!r} is not a published VL embedder; using {QWEN_VL_2B}")
        return QWEN_VL_2B
    return raw


def encoder_kind(*, meta=None, settings=None) -> str:
    """Which visual encoder wrote (or will write) this index."""
    if meta is not None:
        dim = getattr(getattr(meta, "models", None), "visual_dim", None)
        name = (getattr(getattr(meta, "models", None), "clip", "") or "").lower()
        if dim in (2048, 4096):
            return "qwen_vl"
        if dim == 512:
            return "clip"
        if "qwen" in name:
            return "qwen_vl"
        if "open_clip" in name or "vit-b-32" in name:
            return "clip"
        if dim:
            return "clip" if int(dim) == 512 else "qwen_vl"
    cfg = settings or load_settings()
    return "clip" if (cfg.visual_encoder or "qwen_vl").strip().lower() == "clip" else "qwen_vl"


def visual_dim(*, settings=None, meta=None) -> int:
    """FAISS dim for the visual collection. Must match the encoder that wrote it."""
    if meta is not None:
        stored = getattr(getattr(meta, "models", None), "visual_dim", None)
        if stored:
            return int(stored)
        return _dim_from_name(getattr(meta.models, "clip", "") or "")
    cfg = settings or load_settings()
    if (cfg.visual_encoder or "qwen_vl").strip().lower() == "clip":
        return 512
    return _dim_from_name(resolve_visual_model(cfg.visual_model))


def visual_model_label(*, settings=None) -> str:
    cfg = settings or load_settings()
    if (cfg.visual_encoder or "qwen_vl").strip().lower() == "clip":
        return CLIP_NAME
    return resolve_visual_model(cfg.visual_model)


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return (mat / np.clip(norms, eps, None)).astype(np.float32)


def _dim_from_name(name: str) -> int:
    lower = (name or "").lower()
    if "8b" in lower or "4096" in lower:
        return 4096
    if "vit-b-32" in lower or "open_clip" in lower or lower == "clip":
        return 512
    return 2048


@lru_cache(maxsize=1)
def _clip_bundle():
    import open_clip
    import torch

    torch.set_num_threads(max(1, min(16, __import__("os").cpu_count() or 8)))
    print(f"[embed] loading {CLIP_NAME}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device="cpu",
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    return model, preprocess, tokenizer


@lru_cache(maxsize=1)
def _qwen_vl(model_id: str):
    from sentence_transformers import SentenceTransformer

    print(f"[embed] loading {model_id} (CPU, Qwen3-VL-Embedding)")
    return SentenceTransformer(model_id, device="cpu", trust_remote_code=True)


@lru_cache(maxsize=1)
def _bge():
    from sentence_transformers import SentenceTransformer

    print("[embed] loading BAAI/bge-small-en-v1.5")
    return SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")


def _model_id_for(*, meta=None, settings=None) -> str:
    if meta is not None:
        name = getattr(getattr(meta, "models", None), "clip", "") or ""
        if name and encoder_kind(meta=meta) == "qwen_vl":
            return resolve_visual_model(name)
    cfg = settings or load_settings()
    return resolve_visual_model(cfg.visual_model)


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
    if encoder_kind(meta=meta, settings=cfg) == "clip":
        return _embed_clip_images(paths, batch_size=batch_size or 16)
    return _embed_qwen_images(paths, batch_size=batch_size or 1, meta=meta, settings=cfg)


def embed_clip_texts(texts: list[str], *, meta=None, settings=None) -> np.ndarray:
    """Query-time visual-space text embeddings. Pass meta so old CLIP indexes still search."""
    cfg = settings or load_settings()
    dim = visual_dim(settings=cfg, meta=meta)
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    if encoder_kind(meta=meta, settings=cfg) == "clip":
        return _embed_clip_texts(texts)
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
            tensor = torch.stack(images, dim=0)
            feats = model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            vecs.append(feats.cpu().numpy().astype(np.float32))
            print(f"[embed] clip images {min(i + batch_size, len(paths))}/{len(paths)}")
    return np.vstack(vecs) if vecs else np.zeros((0, 512), dtype=np.float32)


def _embed_clip_texts(texts: list[str]) -> np.ndarray:
    import torch

    model, _preprocess, tokenizer = _clip_bundle()
    tokens = tokenizer(texts)
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
