"""Torch device and precision selection, shared by the encoders and the captioner.

Kept out of `config.py` so that importing settings does not drag in torch, and
out of `retrieve/` so that `media/` can use it without depending on retrieval.

The Metal CPU-fallback switch that makes `mps` safe is set in `vvs/__init__.py`;
it has to be armed before torch is imported, so it cannot live here.
"""

from __future__ import annotations

import os
from functools import lru_cache

_FORCED_CPU = False


@lru_cache(maxsize=1)
def resolve_device() -> str:
    """Best available torch device. `VVS_DEVICE` wins if it is actually usable."""
    import torch

    if _FORCED_CPU:
        return "cpu"
    want = (os.environ.get("VVS_DEVICE") or "").strip().lower()
    if want:
        if want == "mps" and not torch.backends.mps.is_available():
            print("[device] VVS_DEVICE=mps but Metal is unavailable; using cpu")
            return "cpu"
        if want == "cuda" and not torch.cuda.is_available():
            print("[device] VVS_DEVICE=cuda but no CUDA device; using cpu")
            return "cpu"
        return want
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dtype(device: str | None = None) -> str:
    """Half precision on a GPU, full precision on CPU.

    CPU fp16 is emulated and slower than fp32, so it is never chosen there.
    `VVS_VISUAL_DTYPE` accepts float16 / bfloat16 / float32.
    """
    want = (os.environ.get("VVS_VISUAL_DTYPE") or "").strip().lower()
    if want in {"float16", "fp16", "half"}:
        return "float16"
    if want in {"bfloat16", "bf16"}:
        return "bfloat16"
    if want in {"float32", "fp32", "full"}:
        return "float32"
    dev = device or resolve_device()
    return "float32" if dev == "cpu" else "float16"


def torch_dtype(name: str):
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def force_cpu() -> None:
    """Abandon the GPU for the rest of the session after it fails to load a model.

    Pinning the choice keeps hand-placed input tensors on the same device as the
    weights, and avoids paying a doomed GPU load once per call.
    """
    global _FORCED_CPU

    _FORCED_CPU = True
    resolve_device.cache_clear()
    should_evict.cache_clear()


# The 2B visual encoder and the 2B captioner each want roughly 5 GB in fp16
# once activations are counted, so they need ~12 GB to sit together with any
# headroom. Below this, holding both resident does not fail cleanly -- the
# allocator simply thrashes, flushing and re-reserving on almost every call,
# which measured ~28x slower per frame than reloading one model per stage.
EVICT_UNDER_GB = 16.0


def device_memory_gb() -> float:
    """Total memory the chosen device can use, or 0.0 when it will not say."""
    import torch

    device = resolve_device()
    try:
        if device == "cuda" and torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
        if device == "mps" and torch.backends.mps.is_available():
            # Unified memory: a 128 GB Mac reports far more than EVICT_UNDER_GB
            # and correctly keeps both models loaded.
            return torch.mps.recommended_max_memory() / 1024**3
    except Exception:
        return 0.0
    return 0.0


@lru_cache(maxsize=1)
def should_evict() -> bool:
    """Whether to unload one large model before loading the next.

    Indexing touches the visual encoder and the captioner in separate stages,
    so on a small GPU only one of them ever needs to be resident. Reloading
    costs seconds per video; sharing a too-small card costs minutes.

    `VVS_EVICT_MODELS` forces the choice either way.
    """
    override = (os.environ.get("VVS_EVICT_MODELS") or "").strip().lower()
    if override:
        return override not in {"0", "false", "no", "off"}
    total = device_memory_gb()
    if total <= 0.0:
        # CPU, or a device with no memory query. Reloading would be pure loss.
        return False
    evict = total < EVICT_UNDER_GB
    print(
        f"[device] {resolve_device()} has {total:.1f} GB; "
        f"model eviction {'on' if evict else 'off'}"
    )
    return evict


def empty_cache() -> None:
    """Hand cached device memory back to the driver after dropping a model.

    Clearing an lru_cache only drops Python's reference; the allocator keeps the
    freed blocks reserved until told otherwise, so the next model would still
    see a full card.
    """
    import gc

    import torch

    gc.collect()
    device = resolve_device()
    try:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as exc:
        print(f"[device] empty_cache failed ({type(exc).__name__}: {exc})")


def to_device(inputs, device: str):
    """Move every tensor in a processor's output mapping, leaving the rest alone."""
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
