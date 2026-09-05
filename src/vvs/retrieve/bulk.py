"""Bulk ingest: index many videos in one call, sequentially.

Each video's rows are independent within the shared OpenSearch index (tagged
by `video_id`, see `retrieve/opensearch_store.py`), so bulk ingest is just a
loop over `build_index`, with per-item error isolation: one bad URL or
unreadable file must not abort the rest of the batch. Already-indexed videos
are skipped unless `rebuild=True`, so re-running the same manifest is cheap.

This stays a plain sequential loop, not a thread/process pool. On CPU the VLM
embedding and captioning stages already saturate the machine for one video,
so concurrent videos would only contend for the same cores. It is also the
only safe option in the Gradio UI: torch model loading must run on a single
thread for the whole process (see `vvs/compute.py`), so callers there should
route this entire function through `compute.run`, not call it per-video from
several threads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from vvs.ingest import extract_video_id
from vvs.retrieve.index import StageCallback, build_index, index_exists

ItemStatus = Literal["indexed", "skipped", "failed"]

# (item index (1-based), total, source) -> None. Fires before each item starts.
OnItemStart = Callable[[int, int, str], None]


@dataclass
class BulkItemResult:
    source: str
    video_id: str | None
    status: ItemStatus
    detail: str = ""


@dataclass
class BulkResult:
    items: list[BulkItemResult] = field(default_factory=list)

    @property
    def indexed(self) -> list[BulkItemResult]:
        return [i for i in self.items if i.status == "indexed"]

    @property
    def skipped(self) -> list[BulkItemResult]:
        return [i for i in self.items if i.status == "skipped"]

    @property
    def failed(self) -> list[BulkItemResult]:
        return [i for i in self.items if i.status == "failed"]

    def summary(self) -> str:
        return (
            f"{len(self.indexed)} indexed, {len(self.skipped)} skipped, "
            f"{len(self.failed)} failed (of {len(self.items)})"
        )


def parse_sources(text: str) -> list[str]:
    """One YouTube URL or local path per line. Blank lines and `#` comments are skipped."""
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            out.append(stripped)
    return out


def bulk_index(
    sources: list[str],
    *,
    enable_ocr: bool = True,
    scene_caption: bool | None = None,
    shot_detect: bool | None = None,
    rebuild: bool = False,
    on_item_start: OnItemStart | None = None,
    on_stage: StageCallback | None = None,
) -> BulkResult:
    result = BulkResult()
    total = len(sources)
    for i, source in enumerate(sources, start=1):
        if on_item_start is not None:
            on_item_start(i, total, source)
        try:
            vid = extract_video_id(source)
        except ValueError as exc:
            result.items.append(BulkItemResult(source, None, "failed", str(exc)))
            continue
        if not rebuild and index_exists(vid):
            result.items.append(BulkItemResult(source, vid, "skipped", "already indexed"))
            continue
        try:
            build_index(
                source,
                video_id=vid,
                enable_ocr=enable_ocr,
                scene_caption=scene_caption,
                shot_detect=shot_detect,
                on_stage=on_stage,
            )
            result.items.append(BulkItemResult(source, vid, "indexed"))
        except Exception as exc:  # noqa: BLE001 - one bad video must not kill the batch
            result.items.append(
                BulkItemResult(source, vid, "failed", f"{type(exc).__name__}: {exc}")
            )
    return result
