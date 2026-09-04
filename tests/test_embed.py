"""Visual-encoder routing (no model download)."""

from __future__ import annotations

import os

import numpy as np

from vvs.config import Settings
from vvs.models import ModelCard
from vvs.retrieve.embed import (
    QWEN_VL_2B,
    QWEN_VL_8B,
    WEMM_2B,
    WEMM_4B,
    WEMM_9B,
    encoder_kind,
    kind_from_model,
    normalize_encoder_kind,
    resolve_visual_model,
    visual_dim,
)


class _Meta:
    def __init__(
        self,
        clip: str,
        visual_dim: int | None = None,
        visual_encoder: str | None = None,
    ) -> None:
        self.models = ModelCard(
            clip=clip,
            asr="a",
            text="t",
            visual_dim=visual_dim,
            visual_encoder=visual_encoder,
        )


def test_qwen_vl_emb_4b_aliases_to_official_2b() -> None:
    assert resolve_visual_model("qwen-vl-emb-4B") == QWEN_VL_2B
    assert resolve_visual_model("qwen-vl-emb-4b") == QWEN_VL_2B
    assert resolve_visual_model("8b") == QWEN_VL_8B
    assert resolve_visual_model(QWEN_VL_2B) == QWEN_VL_2B


def test_visual_dim_follows_encoder() -> None:
    assert visual_dim(settings=Settings(visual_encoder="clip")) == 512
    assert visual_dim(settings=Settings(visual_encoder="qwen_vl", visual_model="2b")) == 2048
    assert visual_dim(settings=Settings(visual_encoder="qwen_vl", visual_model="8b")) == 4096


def test_old_clip_index_stays_on_clip() -> None:
    meta = _Meta("open_clip ViT-B-32 laion2b_s34b_b79k")
    assert encoder_kind(meta=meta) == "clip"
    assert visual_dim(meta=meta) == 512


def test_qwen_index_uses_stored_dim() -> None:
    meta = _Meta(QWEN_VL_2B, visual_dim=2048)
    assert encoder_kind(meta=meta) == "qwen_vl"
    assert visual_dim(meta=meta) == 2048


def test_store_dim_wins_over_stale_clip_meta() -> None:
    """Qwen wrote 2048-d FAISS; old meta.json still says CLIP."""
    meta = _Meta("open_clip ViT-B-32 laion2b_s34b_b79k", visual_dim=2048)
    assert encoder_kind(meta=meta) == "qwen_vl"
    assert visual_dim(meta=meta) == 2048


def test_wemm_aliases() -> None:
    assert resolve_visual_model("wemm") == WEMM_2B
    assert resolve_visual_model("wemm-2b") == WEMM_2B
    assert resolve_visual_model("wemm-4b") == WEMM_4B
    assert resolve_visual_model("wemm-9b") == WEMM_9B
    assert resolve_visual_model(WEMM_2B) == WEMM_2B


def test_wemm_4b_is_not_rerouted_like_qwen_4b() -> None:
    """The 4B guard exists for the VL family only; WeMM really ships a 4B."""
    assert resolve_visual_model("wemm-4b") == WEMM_4B
    assert resolve_visual_model("qwen-vl-emb-4b") == QWEN_VL_2B


def test_wemm_native_dims() -> None:
    assert visual_dim(settings=Settings(visual_encoder="wemm", visual_model="wemm")) == 2048
    assert visual_dim(settings=Settings(visual_encoder="wemm", visual_model="wemm-4b")) == 2560
    assert visual_dim(settings=Settings(visual_encoder="wemm", visual_model="wemm-9b")) == 4096


def test_wemm_matryoshka_truncation() -> None:
    cfg = Settings(visual_encoder="wemm", visual_model="wemm", visual_truncate_dim=256)
    assert visual_dim(settings=cfg) == 256


def test_wemm_rejects_non_matryoshka_width() -> None:
    cfg = Settings(visual_encoder="wemm", visual_model="wemm", visual_truncate_dim=300)
    try:
        visual_dim(settings=cfg)
    except SystemExit:
        return
    raise AssertionError("expected SystemExit for a width WeMM cannot emit")


def test_wemm_index_is_not_mistaken_for_qwen() -> None:
    """Regression: both are 2048-d, so width alone must not decide the encoder.

    Getting this wrong queries a WeMM index with the Qwen text tower, which
    returns plausible-looking but meaningless rankings and no error.
    """
    meta = _Meta(WEMM_2B, visual_dim=2048)
    assert encoder_kind(meta=meta) == "wemm"
    meta_qwen = _Meta(QWEN_VL_2B, visual_dim=2048)
    assert encoder_kind(meta=meta_qwen) == "qwen_vl"


def test_explicit_encoder_field_beats_name_and_dim() -> None:
    meta = _Meta("some-private-mirror/checkpoint", visual_dim=2048, visual_encoder="wemm")
    assert encoder_kind(meta=meta) == "wemm"


def test_truncated_wemm_index_keeps_its_encoder() -> None:
    """A 256-d store is still WeMM, not CLIP-like, despite the unusual width."""
    meta = _Meta(WEMM_2B, visual_dim=256)
    assert encoder_kind(meta=meta) == "wemm"
    assert visual_dim(meta=meta) == 256


def test_kind_from_model() -> None:
    assert kind_from_model(WEMM_2B) == "wemm"
    assert kind_from_model(QWEN_VL_2B) == "qwen_vl"
    assert kind_from_model("open_clip ViT-B-32 laion2b_s34b_b79k") == "clip"
    assert kind_from_model("") is None


def test_normalize_encoder_kind() -> None:
    assert normalize_encoder_kind("wemm") == "wemm"
    assert normalize_encoder_kind("WeMM-2B") == "wemm"
    assert normalize_encoder_kind("qwen") == "qwen_vl"
    assert normalize_encoder_kind("clip") == "clip"
    assert normalize_encoder_kind(None) == "qwen_vl"
    assert normalize_encoder_kind("nonsense") == "qwen_vl"


def test_dtype_is_half_on_gpu_and_full_on_cpu() -> None:
    """CPU fp16 is emulated and slower, so it must never be chosen there."""
    from vvs.retrieve.embed import resolve_dtype

    _clear_device_env()
    assert resolve_dtype("cpu") == "float32"
    assert resolve_dtype("mps") == "float16"
    assert resolve_dtype("cuda") == "float16"


def test_dtype_env_override_wins_for_every_device() -> None:
    from vvs.retrieve.embed import resolve_dtype

    _clear_device_env()
    for value, expected in (
        ("float32", "float32"),
        ("fp32", "float32"),
        ("bfloat16", "bfloat16"),
        ("bf16", "bfloat16"),
        ("fp16", "float16"),
        ("half", "float16"),
    ):
        os.environ["VVS_VISUAL_DTYPE"] = value
        assert resolve_dtype("mps") == expected, value
        assert resolve_dtype("cpu") == expected, value
    _clear_device_env()


def test_unknown_dtype_falls_back_to_the_device_default() -> None:
    from vvs.retrieve.embed import resolve_dtype

    _clear_device_env()
    os.environ["VVS_VISUAL_DTYPE"] = "quantum"
    assert resolve_dtype("cpu") == "float32"
    assert resolve_dtype("mps") == "float16"
    _clear_device_env()


def test_image_batch_defaults_wider_on_gpu() -> None:
    from vvs.retrieve.embed import default_image_batch

    _clear_device_env()
    assert default_image_batch("cpu") == 1
    assert default_image_batch("mps") == 8
    assert default_image_batch("cuda") == 8


def test_image_batch_env_override() -> None:
    from vvs.retrieve.embed import default_image_batch

    _clear_device_env()
    os.environ["VVS_VISUAL_BATCH"] = "32"
    assert default_image_batch("cpu") == 32
    for bogus in ("0", "-4", "many", ""):
        os.environ["VVS_VISUAL_BATCH"] = bogus
        assert default_image_batch("cpu") == 1, bogus
    _clear_device_env()


def test_dtype_names_map_to_real_torch_dtypes() -> None:
    import torch

    from vvs.retrieve.embed import _torch_dtype

    assert _torch_dtype("float16") is torch.float16
    assert _torch_dtype("bfloat16") is torch.bfloat16
    assert _torch_dtype("float32") is torch.float32


def test_probe_rejects_nan_and_accepts_real_vectors() -> None:
    """Guards the fp16 fallback: NaN vectors would silently poison an index."""
    from vvs.retrieve.embed import _probe_finite

    class _Model:
        def __init__(self, value):
            self.value = value

        def encode(self, texts, **_kw):
            return np.full((len(texts), 4), self.value, dtype=np.float32)

    assert _probe_finite(_Model(0.5)) is True
    assert _probe_finite(_Model(float("nan"))) is False
    assert _probe_finite(_Model(float("inf"))) is False
    assert _probe_finite(_Model(0.0)) is False


def test_probe_tolerates_a_model_that_raises() -> None:
    """A probe failure must not block loading; assume the model is fine."""
    from vvs.retrieve.embed import _probe_finite

    class _Broken:
        def encode(self, texts, **_kw):
            raise RuntimeError("no kernel")

    assert _probe_finite(_Broken()) is True


def test_mps_cpu_fallback_is_armed_before_torch_is_imported() -> None:
    """PyTorch reads this at MPS init; setting it after `import torch` is a no-op.

    Regression: it was set inside the model loader, which runs long after torch
    is imported, so unsupported Metal ops aborted the process instead of
    falling back to CPU.
    """
    import subprocess
    import sys

    probe = (
        "import sys, os;"
        "assert 'torch' not in sys.modules;"
        "import vvs;"
        "assert os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] == '1';"
        "assert 'torch' not in sys.modules, 'vvs imported torch at package import';"
        "print('ok')"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_forced_cpu_pins_the_device_for_the_rest_of_the_session() -> None:
    from vvs.retrieve import embed

    _clear_device_env()
    embed.resolve_device.cache_clear()
    original = embed._FORCED_CPU
    try:
        embed._force_cpu()
        assert embed.resolve_device() == "cpu"
        # Even an explicit request must not resurrect a device that just failed.
        os.environ["VVS_DEVICE"] = "mps"
        embed.resolve_device.cache_clear()
        assert embed.resolve_device() == "cpu"
    finally:
        embed._FORCED_CPU = original
        embed.resolve_device.cache_clear()
        _clear_device_env()


def _clear_device_env() -> None:
    for key in ("VVS_DEVICE", "VVS_VISUAL_DTYPE", "VVS_VISUAL_BATCH"):
        os.environ.pop(key, None)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ok")
