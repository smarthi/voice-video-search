"""Visual-encoder routing (no model download)."""

from __future__ import annotations

from vvs.config import Settings
from vvs.models import ModelCard
from vvs.retrieve.embed import (
    QWEN_VL_2B,
    QWEN_VL_8B,
    encoder_kind,
    resolve_visual_model,
    visual_dim,
)


class _Meta:
    def __init__(self, clip: str, visual_dim: int | None = None) -> None:
        self.models = ModelCard(clip=clip, asr="a", text="t", visual_dim=visual_dim)


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


if __name__ == "__main__":
    test_qwen_vl_emb_4b_aliases_to_official_2b()
    test_visual_dim_follows_encoder()
    test_old_clip_index_stays_on_clip()
    test_qwen_index_uses_stored_dim()
    test_store_dim_wins_over_stale_clip_meta()
    print("ok")
