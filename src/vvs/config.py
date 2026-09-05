"""Runtime configuration via CLI flags and VVS_* environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """No baked-in YouTube URL. Source is supplied at runtime or in the UI."""

    model_config = SettingsConfigDict(
        env_prefix="VVS_",
        extra="ignore",
        case_sensitive=False,
    )

    source: str | None = Field(default=None, description="YouTube URL or local video path")
    video_id: str | None = Field(default=None, description="Override derived video id")
    data_dir: str | None = Field(default=None, description="Optional data root override")
    host: str = "0.0.0.0"
    port: int = 7860
    opensearch_url: str = Field(
        default="http://localhost:9200",
        description="OpenSearch endpoint. `docker compose up -d` starts one locally.",
    )
    opensearch_timeout_s: float = Field(
        default=10.0,
        description="Client timeout for OpenSearch requests.",
    )
    force_mute: bool = Field(
        default=False,
        description="Ignore audio and skip Whisper (mute video path).",
    )
    scene_caption: bool = Field(
        default=False,
        description="Caption visual scenes with a VLM and store X-Ray metadata.",
    )
    scene_caption_model: str = Field(
        default="qwen_vl",
        description="qwen_vl (Qwen3-VL-2B-Instruct) or blip.",
    )
    shot_detect: bool = Field(
        default=False,
        description="TransNetV2 shot grouping; embed representatives only.",
    )
    shot_threshold: float = Field(default=0.5, description="TransNetV2 cut threshold.")
    shot_max_gap_s: float = Field(
        default=8.0,
        description="Inside a long shot, keep a still every N seconds.",
    )
    visual_encoder: str = Field(
        default="qwen_vl",
        description=(
            "qwen_vl (Qwen3-VL-Embedding), wemm (tencent/WeMM-Embedding), "
            "or clip (open_clip ViT-B-32)."
        ),
    )
    visual_model: str = Field(
        default="Qwen/Qwen3-VL-Embedding-2B",
        description="HF id. qwen-vl-emb-4B aliases to the official 2B VL embedder.",
    )
    visual_truncate_dim: int | None = Field(
        default=None,
        description=(
            "Matryoshka output width for WeMM (64/128/256/512/1024/native). "
            "256 keeps ~98.7% of MMEB-v2 image and video score at 8x less index. "
            "Ignored by encoders that are not Matryoshka-trained."
        ),
    )
    # Read from the environment inside retrieve.embed (which has torch) rather
    # than here, so config stays import-light. Declared for discoverability.
    device: str | None = Field(
        default=None,
        description="Force a torch device: mps, cuda, or cpu. Default auto-detects.",
    )
    visual_dtype: str | None = Field(
        default=None,
        description="float16, bfloat16 or float32. Default: fp16 on GPU, fp32 on CPU.",
    )
    visual_batch: int | None = Field(
        default=None,
        description="Frames per forward pass. Default 8 on GPU, 1 on CPU.",
    )


def load_settings(
    source: str | None = None,
    video_id: str | None = None,
    data_dir: str | None = None,
) -> Settings:
    env = Settings()
    return env.model_copy(
        update={
            "source": _first(source, env.source),
            "video_id": _first(video_id, env.video_id),
            "data_dir": _first(data_dir, env.data_dir),
        }
    )


def require_source(settings: Settings) -> str:
    if settings.source:
        return settings.source
    raise SystemExit(
        "No video source given. Pass --source <youtube-url-or-local-path> or set VVS_SOURCE."
    )


def _first(*values: str | None) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None
