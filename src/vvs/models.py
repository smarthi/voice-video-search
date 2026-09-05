"""Pydantic models for ingest, index, search, and demo artifacts."""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field, model_validator


class IngestInfo(BaseModel):
    video_id: str
    source: str
    title: str
    path: str
    duration_s: float | None = None
    uploader: str | None = None
    description: str = ""
    webpage_url: str | None = None


class MotionSample(BaseModel):
    """Decode-time L1 / luma jump at one clock time. Not an embedding."""

    t_s: float
    motion_energy: float = 0.0
    flash_energy: float = 0.0


class Keyframe(BaseModel):
    index: int
    t_s: float
    path: str
    hash_hex: str
    shot_id: int | None = None
    motion_energy: float = 0.0
    flash_energy: float = 0.0


class ShotSpan(BaseModel):
    """TransNetV2 shot: inclusive frame indices, half-open clock interval [start_s, end_s)."""

    shot_id: int
    start_s: float
    end_s: float
    start_frame: int = 0
    end_frame: int = 0


class TranscriptSegment(BaseModel):
    index: int
    start_s: float
    end_s: float
    text: str


class OcrObservation(BaseModel):
    frame_index: int
    t_s: float
    text: str
    score: float


class SceneCard(BaseModel):
    """Timed visual description, X-Ray style. Written at analyze, not at query."""

    scene_id: int
    start_s: float
    end_s: float
    frame_index: int
    path: str
    caption: str
    objects: list[str] = Field(default_factory=list)
    model: str = ""

    def passage_text(self) -> str:
        parts = [self.caption.strip()]
        if self.objects:
            parts.append("Objects: " + ", ".join(self.objects))
        return "\n".join(part for part in parts if part)


class ChannelFlags(BaseModel):
    visual: bool = True
    transcript: bool = False
    ocr: bool = False
    asr: bool = False
    scene: bool = False
    vector_backend: str = "opensearch"


class ModelCard(BaseModel):
    clip: str
    asr: str
    text: str
    ocr: str | None = None
    caption: str | None = None
    shot: str | None = None
    visual_dim: int | None = None
    # Which encoder family wrote the visual store. Width cannot answer this:
    # WeMM-2B and Qwen-VL-2B are both 2048-d. Optional so pre-existing indexes
    # still validate; they fall back to name/width inference in `encoder_kind`.
    visual_encoder: str | None = None


class IndexMeta(BaseModel):
    video_id: str
    source: str
    title: str | None = None
    duration_s: float = 0.0
    video_path: str
    frames: list[Keyframe] = Field(default_factory=list)
    transcript: list[TranscriptSegment] = Field(default_factory=list)
    ocr: list[OcrObservation] = Field(default_factory=list)
    scenes: list[SceneCard] = Field(default_factory=list)
    shots: list[ShotSpan] = Field(default_factory=list)
    slide_heavy: bool = False
    channels: ChannelFlags = Field(default_factory=ChannelFlags)
    models: ModelCard
    vector_backend: str = "opensearch"

    @model_validator(mode="after")
    def _legacy_asr_channel(self) -> IndexMeta:
        if self.transcript and not self.channels.asr:
            self.channels.asr = True
            self.channels.transcript = True
        return self


class StageTimings(BaseModel):
    ingest_s: float = 0.0
    frames_s: float = 0.0
    shots_s: float = 0.0
    asr_s: float = 0.0
    ocr_s: float = 0.0
    scenes_s: float = 0.0
    caption_s: float = 0.0
    clip_embed_s: float = 0.0
    bge_transcript_s: float = 0.0
    bge_ocr_s: float = 0.0
    bge_scene_s: float = 0.0
    total_s: float = 0.0


class Hit(BaseModel):
    rank: int
    frame_index: int
    t_s: float
    path: str
    video_id: str = ""
    video_title: str = ""
    score: float
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="CTO-facing 0–1 confidence (RRF score divided by the theoretical max).",
    )
    visual_score: float
    transcript_score: float
    ocr_score: float
    scene_score: float = 0.0
    scene_caption: str = ""
    shot_id: int | None = None
    motion_energy: float = 0.0
    flash_energy: float = 0.0
    quote: str
    quote_start_s: float | None = None
    quote_end_s: float | None = None
    channels: list[str] = Field(default_factory=list)
    clip_path: str | None = None
    clip_start_s: float | None = None
    clip_end_s: float | None = None

    @computed_field
    @property
    def frame_path(self) -> str:
        return self.path

    @computed_field
    @property
    def clip_start_sec(self) -> float | None:
        return self.clip_start_s

    @computed_field
    @property
    def clip_end_sec(self) -> float | None:
        return self.clip_end_s


class ClipWindow(BaseModel):
    start_s: float
    end_s: float


class ExtractedClip(BaseModel):
    path: str
    start_s: float
    end_s: float


class DemoQuestion(BaseModel):
    query: str
    why: str
    expect: str


class BuildResult(BaseModel):
    video_id: str
    meta: IndexMeta
    timings: StageTimings
