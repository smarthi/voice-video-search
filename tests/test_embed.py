"""Visual-encoder routing (no model download)."""

from __future__ import annotations

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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ok")
