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


def to_device(inputs, device: str):
    """Move every tensor in a processor's output mapping, leaving the rest alone."""
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
