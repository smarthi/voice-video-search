"""Gradio UI: pick a video, analyze it, then voice or typed query of matching moments."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image, ImageDraw

from vvs import compute
from vvs.config import load_settings
from vvs.ingest import extract_video_id
from vvs.media.clips import attach_corpus_clips
from vvs.media.transcribe import transcribe_audio_file
from vvs.models import Hit, IndexMeta, StageTimings
from vvs.paths import DATA_ROOT
from vvs.retrieve.bulk import bulk_index, parse_sources
from vvs.retrieve.embed import warm_models
from vvs.retrieve.index import (
    build_index,
    delete_index,
    format_timings,
    index_exists,
    list_indexed,
    load_index,
    load_timings,
)
from vvs.retrieve.search import fmt_clock, fmt_confidence, fmt_time, search_corpus
from vvs.ui.library import (
    SORT_CHOICES,
    card_html,
    empty_html,
    ensure_previews,
    entries as library_entries,
    filter_entries,
    library_summary,
    sort_entries,
)
from vvs.ui.theme import FORCE_DARK_JS, HEAD, RESULTS_CHROME, TOPBAR_HTML, build_theme, load_css
from vvs.ui.timeline import render_corpus_timeline, render_timeline
from vvs.ui.xray import render_xray, render_xray_corpus


PLAYABLE_K = 3
_BUNDLES: dict[str, dict[str, Any]] = {}

_STAGE_UI: dict[str, tuple[float, str]] = {
    "ingest": (0.08, "Ingesting video (download or copy)…"),
    "frames": (0.22, "Extracting keyframes…"),
    "shots": (0.22, "Detecting shots (TransNetV2)…"),
    "asr": (0.42, "Transcribing with Whisper…"),
    "asr_skip": (0.42, "No audio track — skipping Whisper (visual + OCR)…"),
    "ocr": (0.58, "Running OCR…"),
    "clip": (0.68, "Embedding frames (visual encoder)…"),
    "visual": (0.68, "Embedding frames (Qwen-VL)…"),
    "bge_transcript": (0.78, "Embedding transcript (BGE)…"),
    "bge_ocr": (0.82, "Embedding OCR text…"),
    "scenes": (0.84, "Clustering visual scenes…"),
    "caption": (0.90, "Captioning scenes (Qwen3-VL Instruct)…"),
    "bge_scene": (0.96, "Embedding scene captions (BGE)…"),
    "save": (0.98, "Writing index…"),
    "done": (1.0, "Analyze complete."),
}

_DEFAULT_HEADER = (
    "### Ask for a moment, not a file\n"
    "Analyze one or more videos, then ask by text (voice is optional if a file has audio). "
    "One query searches the whole archive. Mute video uses Qwen-VL + OCR."
)


def launch(
    source: str | None = None,
    video_id: str | None = None,
    *,
    rebuild: bool = False,
    warm: bool = True,
    server_name: str | None = None,
    server_port: int | None = None,
) -> None:
    settings = load_settings(source=source, video_id=video_id)
    prefill = settings.source or settings.video_id or ""
    video_id_override = settings.video_id
    analyze_rebuild = rebuild

    if warm:
        print("[app] warming models…")
        # Warm on the dedicated compute thread, not this (main) thread: every
        # request lands on a different Gradio worker thread, and torch/faiss
        # segfault the moment a *second* distinct thread touches them (see
        # vvs/compute.py). Warming here on main would just be the first of
        # those crashes waiting to happen.
        compute.run(warm_models)
        print("[app] models ready")

    archive_n = len(list_indexed())
    have_archive = archive_n > 0
    idle = _idle_outputs(
        "Paste a YouTube URL or upload a file, then click **Analyze video**."
        if not have_archive
        else f"{archive_n} video(s) already indexed. Ask a question — search covers the whole archive."
    )
    start_header = _archive_header(archive_n)

    def run_query(query: str, vid: str | None, scope_id: str | None = None) -> tuple:
        # Every request is served on its own Gradio worker thread; funnel the
        # actual torch/faiss work onto the one dedicated compute thread (see
        # vvs/compute.py) so it never runs on two different threads.
        return compute.run(_run_query, query, vid, scope_id)

    def _run_query(query: str, vid: str | None, scope_id: str | None) -> tuple:
        ids = _ensure_corpus()
        if vid and vid not in ids and index_exists(vid):
            _BUNDLES[vid] = load_index(vid)
            ids = _ensure_corpus()
        if not ids:
            return _idle_outputs("No indexed videos. Analyze at least one.")
        archive_size = len(ids)
        if scope_id:
            if scope_id in ids:
                ids = [scope_id]
            else:
                print(f"[app] scope {scope_id} is not indexed; searching everything")
                scope_id = None
        started = time.perf_counter()
        hits = search_corpus(query, ids, top_k=6, loaded=_BUNDLES)
        search_ms = (time.perf_counter() - started) * 1000
        metas = _corpus_metas()
        clip_started = time.perf_counter()
        try:
            playable = attach_corpus_clips(hits, top_k=PLAYABLE_K, metas=metas)
        except Exception as exc:
            print(f"[app] clip extract failed: {exc}")
            playable = hits[:PLAYABLE_K]
        clip_ms = (time.perf_counter() - clip_started) * 1000
        gallery = [(hit.frame_path, _still_caption(hit)) for hit in playable if hit.frame_path]
        timeline = render_corpus_timeline(metas, hits)
        xray = render_xray_corpus(metas, hits)
        sources: list[str] = []
        for hit in hits:
            if hit.video_id and hit.video_id not in sources:
                sources.append(hit.video_id)
        reach = (
            f"scoped to {scope_id}"
            if scope_id
            else f"searched {archive_size} video{'s' if archive_size != 1 else ''}"
        )
        status = (
            f"Query: “{query}”  ·  {len(playable)} playable clips  ·  {reach}"
            + (f" · hits from {', '.join(sources)}" if sources else "")
            + f"  ·  search {search_ms:.0f} ms  ·  clips {clip_ms:.0f} ms"
        )
        return (*_hit_outputs(playable), xray, gallery, timeline, status)

    def from_text(query: str, vid: str | None, scope_id: str | None):
        if not list_indexed() and not vid:
            return _idle_outputs("Analyze a video first.")
        query = (query or "").strip()
        if not query:
            meta = compute.run(_bundle, vid)["meta"] if vid else None
            return _idle_outputs("Type or speak a query.", meta)
        return run_query(query, vid, scope_id)

    def from_audio(audio_path: str | None, typed: str, vid: str | None, scope_id: str | None):
        if not list_indexed() and not vid:
            return "", *_idle_outputs("Analyze a video first.")
        meta = compute.run(_bundle, vid)["meta"] if vid else None
        if audio_path:
            # Whisper (faster-whisper/ctranslate2) is another native, threaded
            # backend; keep it on the same dedicated compute thread as
            # everything else (see vvs/compute.py).
            spoken = compute.run(transcribe_audio_file, Path(audio_path))
            if spoken:
                return spoken, *run_query(spoken, vid, scope_id)
        if typed and typed.strip():
            cleaned = typed.strip()
            return cleaned, *run_query(cleaned, vid, scope_id)
        return "", *_idle_outputs("No speech detected.", meta)

    def analyze(
        url_or_path: str | None,
        upload: Any,
        current_vid: str | None,
        scene_on: bool,
        shot_on: bool,
        progress: gr.Progress = gr.Progress(),
    ):
        return compute.run(
            _analyze,
            url_or_path,
            upload,
            current_vid,
            rebuild=analyze_rebuild,
            prefill=prefill,
            video_id_override=video_id_override,
            scene_caption=bool(scene_on),
            shot_detect=bool(shot_on),
            progress=progress,
        )

    def reanalyze(
        url_or_path: str | None,
        upload: Any,
        current_vid: str | None,
        scene_on: bool,
        shot_on: bool,
        progress: gr.Progress = gr.Progress(),
    ):
        return compute.run(
            _analyze,
            url_or_path,
            upload,
            current_vid,
            rebuild=True,
            prefill=prefill,
            video_id_override=video_id_override,
            scene_caption=bool(scene_on),
            shot_detect=bool(shot_on),
            progress=progress,
        )

    def bulk_analyze(
        sources_text: str,
        files: list[str] | None,
        scene_on: bool,
        shot_on: bool,
        progress: gr.Progress = gr.Progress(),
    ):
        sources = parse_sources(sources_text or "")
        for f in files or []:
            path = _as_path(f)
            if path:
                sources.append(path)
        if not sources:
            n = len(list_indexed())
            return (
                _archive_header(n),
                "Paste one or more YouTube URLs / local paths, or upload files, then click **Bulk analyze**.",
                *_bulk_query_unlock(n > 0),
            )
        return compute.run(_bulk_analyze, sources, scene_on, shot_on, progress)

    def build_previews(progress: gr.Progress = gr.Progress()) -> None:
        ids = list_indexed()

        def on_progress(done: int, total: int, vid: str) -> None:
            progress(done / max(1, total), desc=f"Preview {done}/{total} · {vid}")

        made = compute.run(ensure_previews, ids, on_progress=on_progress)
        failed = [vid for vid, path in made.items() if path is None]
        if failed:
            print(f"[app] no preview for: {', '.join(failed)}")

    def drop_video(video_id: str, current_scope: str | None):
        """Delete an index and stop searching it. Second click of a two-step confirm."""
        bundle = _BUNDLES.pop(video_id, None)
        for key in ("visual", "text"):
            store = (bundle or {}).get(key)
            if store is not None:
                try:
                    store.close()
                except Exception as exc:
                    print(f"[app] close {key} for {video_id}: {exc}")
        try:
            delete_index(video_id)
        except SystemExit as exc:
            print(f"[app] delete refused: {exc}")
        return None, (None if current_scope == video_id else current_scope)

    with gr.Blocks(title="Voice Video Search", fill_width=True) as demo:
        gr.HTML(TOPBAR_HTML)
        header = gr.Markdown(start_header, elem_classes=["sl-hero"])
        active_vid = gr.State(None)
        scope = gr.State(None)
        armed = gr.State(None)
        # Monotonic counter, because writing None over None does not fire
        # `.change` and so would not retrigger the grid after an analyze.
        tick = gr.State(0)
        with gr.Accordion("00  ·  Library", open=have_archive):
            with gr.Row():
                lib_filter = gr.Textbox(
                    label="Filter",
                    placeholder="title or video id",
                    scale=3,
                )
                # Radio, not Dropdown: four fixed choices need no popup, and a
                # Gradio dropdown accepts arbitrary typed text for a closed set.
                lib_sort = gr.Radio(
                    label="Sort",
                    choices=list(SORT_CHOICES),
                    value=SORT_CHOICES[0][1],
                    scale=4,
                )
            with gr.Row():
                refresh_btn = gr.Button("Refresh", size="sm")
                preview_btn = gr.Button("Build previews", size="sm", variant="primary")
                all_btn = gr.Button("Search whole archive", size="sm")

            # Cards are real components, not one HTML blob, so each can own its
            # own click handlers. Only the media stays hand-written HTML, which
            # is what keeps hover-play and preload="none".
            @gr.render(
                inputs=[lib_filter, lib_sort, scope, armed],
                triggers=[
                    demo.load,
                    lib_filter.change,
                    lib_sort.change,
                    scope.change,
                    armed.change,
                    refresh_btn.click,
                    tick.change,
                ],
            )
            def draw_library(needle, sort_key, current, armed_id):
                found = library_entries()
                items = sort_entries(filter_entries(found, needle), sort_key)
                gr.Markdown(
                    f"{library_summary(items, len(found))}  ·  {_scope_line(current)}",
                    elem_classes=["sl-lib-hint"],
                )
                if not items:
                    gr.HTML(empty_html(filtered=bool(found)))
                    return
                with gr.Column(elem_classes=["sl-lib-grid-native"]):
                    for item in items:
                        vid = item.video_id
                        selected = vid == current
                        with gr.Column(elem_classes=["sl-lib-card"]):
                            gr.HTML(card_html(item, selected=selected))
                            with gr.Row(elem_classes=["sl-lib-actions"]):
                                if armed_id == vid:
                                    # Armed: offer a way out before the way through.
                                    cancel = gr.Button("Cancel", size="sm")
                                    cancel.click(
                                        lambda: None,
                                        outputs=armed,
                                        show_progress="hidden",
                                    )
                                    confirm = gr.Button(
                                        "Confirm delete", size="sm", variant="stop"
                                    )
                                    confirm.click(
                                        lambda s, v=vid: drop_video(v, s),
                                        inputs=scope,
                                        outputs=[armed, scope],
                                    )
                                else:
                                    pick = gr.Button(
                                        "Searching this" if selected else "Search this",
                                        size="sm",
                                        variant="primary" if selected else "secondary",
                                    )
                                    pick.click(
                                        lambda v=vid: v,
                                        outputs=scope,
                                        show_progress="hidden",
                                    )
                                    ask = gr.Button("Delete", size="sm")
                                    ask.click(
                                        lambda v=vid: v,
                                        outputs=armed,
                                        show_progress="hidden",
                                    )
            gr.Markdown(
                "Hover a tile to play its preview. **Search this** narrows every query "
                "to one video. Previews are cached as `preview.mp4` beside each index.",
                elem_classes=["sl-lib-hint"],
            )
        with gr.Accordion("01  ·  Pick and analyze a video", open=True):
            source_box = gr.Textbox(
                label="YouTube URL or local path",
                value=prefill,
                placeholder="https://www.youtube.com/watch?v=… or /path/to/video.mp4",
            )
            upload = gr.File(
                label="Or upload a local video",
                file_count="single",
                type="filepath",
                file_types=[".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v"],
            )
            gr.Markdown(
                "Independent options — check **both** to group shots and caption those scenes in one Analyze."
            )
            with gr.Row():
                shot_box = gr.Checkbox(
                    label="Group shots with TransNetV2",
                    value=settings.shot_detect,
                    info="Cuts first, then embed representatives (8s floor in long takes).",
                )
                scene_box = gr.Checkbox(
                    label="Describe scenes (X-Ray captions)",
                    value=settings.scene_caption,
                    info="Qwen3-VL-2B-Instruct captions those scenes (BLIP fallback).",
                )
            with gr.Row():
                analyze_btn = gr.Button("Analyze video", variant="primary")
                reanalyze_btn = gr.Button("Re-analyze", interactive=False)
            analyze_status = gr.Markdown(
                f"{archive_n} video(s) indexed. Search queries the whole archive."
                if have_archive
                else "Waiting for a video. Query unlocks after the first analyze, or if indexes already exist."
            )
        with gr.Accordion("01b  ·  Bulk analyze (many videos)", open=False):
            bulk_sources_box = gr.Textbox(
                label="YouTube URLs or local paths, one per line",
                placeholder=(
                    "https://www.youtube.com/watch?v=…\n"
                    "/path/to/video.mp4\n"
                    "# lines starting with # are ignored"
                ),
                lines=5,
            )
            bulk_upload = gr.File(
                label="Or upload local videos",
                file_count="multiple",
                type="filepath",
                file_types=[".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v"],
            )
            bulk_btn = gr.Button("Bulk analyze", variant="primary")
            bulk_status = gr.Markdown(
                "Already-indexed videos are skipped automatically; one bad entry does not stop the rest. "
                "Shot grouping / X-Ray captions above apply to every video in the batch."
            )
        with gr.Accordion("02  ·  Ask the archive", open=True, elem_classes=["sl-ask"]):
            with gr.Row():
                query_box = gr.Textbox(
                    label="Typed query (searches every indexed video)",
                    placeholder="e.g. show me the product on stage",
                    interactive=have_archive,
                )
                mic = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Voice query",
                    interactive=have_archive,
                )
            with gr.Row():
                ask_btn = gr.Button("Search", variant="primary", interactive=have_archive)
                voice_btn = gr.Button("Transcribe + search", interactive=have_archive)
        status = gr.Markdown(idle[-1], elem_classes=["sl-answer"])
        gr.HTML(RESULTS_CHROME)
        players: list[gr.Video] = []
        stills: list[gr.Image] = []
        notes: list[gr.Markdown] = []
        with gr.Row(elem_classes=["sl-results"]):
            for i in range(PLAYABLE_K):
                with gr.Column(elem_classes=["sl-card"]):
                    notes.append(gr.Markdown(_placeholder_note(i)))
                    players.append(
                        gr.Video(
                            label=f"Clip #{i + 1}",
                            interactive=False,
                            include_audio=True,
                            autoplay=False,
                            height=280,
                            format="mp4",
                        )
                    )
                    stills.append(
                        gr.Image(
                            label=f"Original frame #{i + 1}",
                            height=140,
                        )
                    )
        xray = gr.HTML(render_xray(None), elem_classes=["sl-xray-wrap"])
        gallery = gr.Gallery(
            label="Matching keyframes",
            columns=3,
            height=180,
            object_fit="contain",
        )
        timeline = gr.Image(label="Match timeline", value=_blank_timeline())

        result_outputs = [*players, *stills, *notes, xray, gallery, timeline, status]
        query_unlock = [query_box, mic, ask_btn, voice_btn, reanalyze_btn]
        analyze_outputs = [header, active_vid, analyze_status, *result_outputs, *query_unlock]

        bump = {"fn": lambda n: (n or 0) + 1, "inputs": tick, "outputs": tick}
        # Redraw only after the work finishes, so the grid never renders against
        # half-built previews or a half-written index.
        preview_btn.click(build_previews).then(**bump, show_progress="hidden")
        all_btn.click(lambda: None, outputs=scope, show_progress="hidden")
        analyze_btn.click(
            analyze,
            inputs=[source_box, upload, active_vid, scene_box, shot_box],
            outputs=analyze_outputs,
        ).then(**bump, show_progress="hidden")
        reanalyze_btn.click(
            reanalyze,
            inputs=[source_box, upload, active_vid, scene_box, shot_box],
            outputs=analyze_outputs,
        ).then(**bump, show_progress="hidden")
        bulk_btn.click(
            bulk_analyze,
            inputs=[bulk_sources_box, bulk_upload, scene_box, shot_box],
            outputs=[header, bulk_status, query_box, mic, ask_btn, voice_btn],
        ).then(**bump, show_progress="hidden")
        ask_btn.click(
            from_text, inputs=[query_box, active_vid, scope], outputs=result_outputs
        )
        query_box.submit(
            from_text, inputs=[query_box, active_vid, scope], outputs=result_outputs
        )
        voice_btn.click(
            from_audio,
            inputs=[mic, query_box, active_vid, scope],
            outputs=[query_box, *result_outputs],
        )

    demo.queue()
    demo.launch(
        server_name=server_name or settings.host,
        server_port=server_port or settings.port,
        show_error=True,
        allowed_paths=[str(DATA_ROOT.resolve())],
        theme=build_theme(),
        css=load_css(),
        head=HEAD,
        js=FORCE_DARK_JS,
    )


def _analyze(
    url_or_path: str | None,
    upload: Any,
    current_vid: str | None,
    *,
    rebuild: bool,
    prefill: str,
    video_id_override: str | None,
    scene_caption: bool,
    shot_detect: bool,
    progress: gr.Progress,
) -> tuple:
    keep_unlocked = bool(current_vid)
    try:
        from_upload, source = _resolve_user_source(url_or_path, upload)
        dest_id = _dest_id(source, from_upload, video_id_override, prefill)
        try:
            vid = dest_id or extract_video_id(source)
        except ValueError as exc:
            raise ValueError(
                "Could not determine a video id. Paste a YouTube URL / id, "
                "a local file path, or upload a video."
            ) from exc

        reused = False
        timings: StageTimings | None = None
        if not rebuild and index_exists(vid):
            progress(1.0, desc=f"Loading existing index for {vid}…")
            _BUNDLES[vid] = load_index(vid)
            timings = load_timings(vid)
            reused = True
        else:
            def on_stage(stage: str) -> None:
                fraction, label = _STAGE_UI.get(stage, (None, stage))
                if fraction is None:
                    progress(desc=label)
                else:
                    progress(fraction, desc=label)

            _BUNDLES.pop(vid, None)
            result = build_index(
                source,
                video_id=dest_id,
                scene_caption=scene_caption,
                shot_detect=shot_detect,
                on_stage=on_stage,
            )
            vid = result.video_id
            _BUNDLES[vid] = load_index(vid)
            timings = result.timings

        meta: IndexMeta = _BUNDLES[vid]["meta"]
        header = _ready_header(meta)
        timing_line = format_timings(timings) if timings else ""
        if reused:
            analyze_status = (
                f"**Using existing index** for `{vid}`. Skipped rebuild.  \n"
                + (f"Previous stage timings: {timing_line}  \n" if timing_line else "")
                + "Query is unlocked. One search covers every indexed video. "
                "Click **Re-analyze** to rebuild this title from scratch."
            )
        else:
            analyze_status = (
                f"**Analyze complete** for `{vid}`.  \n"
                + (f"Stage timings: {timing_line}  \n" if timing_line else "")
                + f"Archive has {len(list_indexed())} video(s). One query searches all of them."
            )
        status = (
            f"Ready · **{meta.title or vid}** · `{vid}` · {fmt_time(meta.duration_s)} · "
            f"{len(meta.frames)} keyframes"
            + (f" · {len(meta.transcript)} spoken segments" if meta.channels.asr else " · mute / no ASR")
            + (" · OCR on" if meta.channels.ocr else "")
            + (f" · {len(meta.scenes)} scenes" if meta.scenes else "")
            + (f" · {len(meta.shots)} shots" if meta.shots else "")
            + f" · {meta.vector_backend}"
        )
        return (
            header,
            vid,
            analyze_status,
            *_idle_outputs(status, meta),
            *_query_unlock(True),
        )
    except Exception as exc:
        print(f"[app] analyze failed: {exc}")
        header = _ready_header(_bundle(current_vid)["meta"]) if current_vid and current_vid in _BUNDLES else _DEFAULT_HEADER
        meta = _bundle(current_vid)["meta"] if current_vid and current_vid in _BUNDLES else None
        return (
            header,
            current_vid,
            f"**Analyze failed:** {exc}",
            *_idle_outputs(f"Analyze failed: {exc}", meta),
            *_query_unlock(keep_unlocked),
        )


def _scope_line(scope_id: str | None) -> str:
    if scope_id:
        return f"scoped to **{scope_id}** — use *Search whole archive* to widen"
    return "searching the whole archive"


def _resolve_user_source(url_or_path: str | None, upload: Any) -> tuple[bool, str]:
    uploaded = _as_path(upload)
    if uploaded:
        return True, uploaded
    text = (url_or_path or "").strip()
    if text:
        return False, text
    raise ValueError("Paste a YouTube URL or local path, or upload a video file.")


def _dest_id(
    source: str,
    from_upload: bool,
    override: str | None,
    prefill: str,
) -> str | None:
    if from_upload or not override:
        return None
    if source.strip() in {prefill.strip(), override}:
        return override
    return None


def _as_path(upload: Any) -> str | None:
    if upload is None:
        return None
    if isinstance(upload, str) and upload.strip():
        return upload.strip()
    name = getattr(upload, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _bundle(video_id: str) -> dict[str, Any]:
    if video_id not in _BUNDLES:
        _BUNDLES[video_id] = load_index(video_id)
    return _BUNDLES[video_id]


def _ensure_corpus() -> list[str]:
    ready: list[str] = []
    for vid in list_indexed():
        try:
            _bundle(vid)
            ready.append(vid)
        except Exception as exc:
            print(f"[app] skip {vid}: {exc}")
    return ready


def _corpus_metas() -> dict[str, IndexMeta]:
    return {vid: bundle["meta"] for vid, bundle in _BUNDLES.items()}


def _ready_header(meta: IndexMeta) -> str:
    extra = " · OCR on" if meta.channels.ocr else ""
    extra += f" · {len(meta.scenes)} X-Ray scenes" if meta.scenes else ""
    extra += f" · {len(meta.shots)} shots" if meta.shots else ""
    speech = f"{len(meta.transcript)} spoken segments" if meta.channels.asr else "visual + OCR (no audio)"
    n = len(list_indexed())
    archive = f" · archive {n} videos" if n > 1 else ""
    return (
        f"### {meta.title or meta.video_id}\n"
        f"`{meta.video_id}` · {fmt_time(meta.duration_s)} · "
        f"{len(meta.frames)} keyframes · {speech}{extra} · {meta.vector_backend}{archive}"
    )


def _query_unlock(enabled: bool) -> tuple:
    return tuple(gr.update(interactive=enabled) for _ in range(5))


def _bulk_query_unlock(enabled: bool) -> tuple:
    return tuple(gr.update(interactive=enabled) for _ in range(4))


def _archive_header(n: int) -> str:
    if n <= 0:
        return _DEFAULT_HEADER
    return (
        f"### Archive ready\n`{n}` indexed video(s). Analyze another, or ask — "
        "one query searches all of them."
    )


def _bulk_analyze(
    sources: list[str],
    scene_on: bool,
    shot_on: bool,
    progress: gr.Progress,
) -> tuple:
    """Runs on the dedicated compute thread (see vvs/compute.py). Never call directly
    from a Gradio callback thread — go through bulk_analyze(), which dispatches here.
    """

    def on_item_start(i: int, total: int, source: str) -> None:
        progress(max(0.0, (i - 1) / max(1, total)), desc=f"({i}/{total}) {source}")

    result = bulk_index(
        sources,
        scene_caption=bool(scene_on),
        shot_detect=bool(shot_on),
        on_item_start=on_item_start,
    )
    progress(1.0, desc="done")
    lines = [f"**Bulk analyze**: {result.summary()}."]
    for item in result.failed:
        lines.append(f"- FAILED `{item.source}`: {item.detail}")
    n = len(list_indexed())
    return _archive_header(n), "  \n".join(lines), *_bulk_query_unlock(n > 0)


def _hit_outputs(hits: list[Hit]) -> tuple:
    videos: list[str | None] = []
    frames: list[str | None] = []
    captions: list[str] = []
    for i in range(PLAYABLE_K):
        if i < len(hits) and hits[i].clip_path:
            videos.append(str(Path(hits[i].clip_path).resolve()))
            frames.append(str(Path(hits[i].frame_path).resolve()))
            captions.append(_clip_caption(hits[i]))
        else:
            videos.append(None)
            frames.append(None)
            captions.append(_placeholder_note(i))
    return (*videos, *frames, *captions)


def _idle_outputs(status: str, meta: IndexMeta | None = None) -> tuple:
    return (
        *([None] * PLAYABLE_K),
        *([None] * PLAYABLE_K),
        *[_placeholder_note(i) for i in range(PLAYABLE_K)],
        render_xray(meta),
        [],
        render_timeline(meta, []) if meta else _blank_timeline(),
        status,
    )


def _clip_caption(hit: Hit) -> str:
    start = hit.clip_start_s if hit.clip_start_s is not None else hit.t_s
    if hit.clip_end_s is not None:
        span = f"{fmt_clock(start)}–{fmt_clock(hit.clip_end_s)}"
    else:
        span = fmt_clock(start)
    quote = hit.quote.strip() if hit.quote else "*no nearby speech*"
    channels = _channel_scores(hit)
    xray = f"  \nX-Ray: {hit.scene_caption}" if hit.scene_caption else ""
    shot = f"  ·  shot {hit.shot_id}" if hit.shot_id is not None else ""
    asset = hit.video_title or hit.video_id
    from_line = f"  \n**{asset}**" if asset else ""
    return (
        f"**Clip {span}**  ·  confidence {fmt_confidence(hit.confidence)}{from_line}  \n"
        f"#{hit.rank}  ·  {channels}{shot}  ·  frame {fmt_clock(hit.t_s)}  \n{quote}{xray}"
    )


def _still_caption(hit: Hit) -> str:
    start = hit.clip_start_s if hit.clip_start_s is not None else hit.t_s
    end = f"–{fmt_clock(hit.clip_end_s)}" if hit.clip_end_s is not None else ""
    asset = f"  ·  {hit.video_id}" if hit.video_id else ""
    return (
        f"#{hit.rank}  {fmt_clock(start)}{end}  ·  "
        f"confidence {fmt_confidence(hit.confidence)}{asset}"
    )


def _channel_scores(hit: Hit) -> str:
    parts = [f"visual {hit.visual_score:.2f}", f"transcript {hit.transcript_score:.2f}"]
    if hit.ocr_score > 0:
        parts.append(f"ocr {hit.ocr_score:.2f}")
    if hit.scene_score > 0:
        parts.append(f"scene {hit.scene_score:.2f}")
    if hit.motion_energy > 0:
        parts.append(f"motion {hit.motion_energy:.2f}")
    if hit.flash_energy > 0:
        parts.append(f"flash {hit.flash_energy:.2f}")
    if hit.channels:
        parts.append("[" + "+".join(hit.channels) + "]")
    return " · ".join(parts)


def _placeholder_note(index: int) -> str:
    return f"*Match #{index + 1} will appear here.*"


def _blank_timeline() -> Image.Image:
    image = Image.new("RGB", (920, 88), (5, 7, 12))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((16, 40, 904, 56), radius=2, fill=(26, 35, 49))
    draw.text((16, 14), "Analyze a video to see the match timeline.", fill=(90, 102, 117))
    return image
