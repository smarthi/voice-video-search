"""Command-line interface. Video source is always passed in, never hardcoded."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vvs.config import load_settings, require_source
from vvs.media.clips import attach_clips, attach_corpus_clips
from vvs.retrieve.bulk import bulk_index, parse_sources
from vvs.retrieve.embed import warm_models
from vvs.retrieve.index import (
    build_index,
    embed_counts,
    ensure_index,
    format_embed_compare,
    format_embed_counts,
    list_indexed,
    load_index,
)
from vvs.retrieve.search import drop_weak_hits, format_hits, search, search_corpus


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vvs", description="Voice-queried video frame retrieval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Download/copy a video and build the retrieval index")
    _add_source_flags(p_index)
    p_index.add_argument(
        "--sources-file",
        default=None,
        help=(
            "Bulk mode: index every video listed in this file, one YouTube URL or "
            "local path per line (blank lines and '#' comments are skipped). "
            "Ignores --source / --video-id; each video id is derived from its own "
            "line. Already-indexed videos are skipped unless --rebuild."
        ),
    )
    p_index.add_argument("--no-ocr", action="store_true", help="Skip RapidOCR")
    p_index.add_argument(
        "--scene-caption",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Caption visual scenes (X-Ray). Default follows VVS_SCENE_CAPTION.",
    )
    p_index.add_argument(
        "--shot-detect",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="TransNetV2 shot grouping (embed reps only). Default follows VVS_SHOT_DETECT.",
    )
    p_index.add_argument("--rebuild", action="store_true")

    p_query = sub.add_parser(
        "query",
        help="Typed search (one video, or the whole archive if --video-id is omitted)",
    )
    _add_source_flags(p_query)
    p_query.add_argument("text", nargs="+", help="Query text")
    p_query.add_argument("--top-k", type=int, default=6)
    p_query.add_argument("--json", action="store_true")
    p_query.add_argument(
        "--all",
        action="store_true",
        help="Search every indexed video (default when --video-id / --source are omitted)",
    )

    p_app = sub.add_parser(
        "app",
        help="Launch the Gradio UI (pick a video, analyze, then query)",
    )
    _add_source_flags(p_app, optional_prefill=True)
    p_app.add_argument("--rebuild", action="store_true")
    p_app.add_argument("--no-warm", action="store_true")
    p_app.add_argument("--host", default=None)
    p_app.add_argument("--port", type=int, default=None)

    p_list = sub.add_parser("list", help="List indexed videos and embedding counts")
    p_list.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE_ID", "SHOTS_ID"),
        help="Compare visual embedding counts (2fps id vs TransNetV2 id)",
    )
    p_warm = sub.add_parser("warm", help="Preload EmbeddingGemma, the captioner, and Whisper")
    _ = p_list
    _ = p_warm

    args = parser.parse_args(argv)
    if args.cmd == "index":
        if args.sources_file:
            _bulk_index_cmd(args)
            return
        settings = load_settings(source=args.source, video_id=args.video_id)
        source = require_source(settings)
        result = build_index(
            source,
            video_id=settings.video_id,
            enable_ocr=not args.no_ocr,
            scene_caption=args.scene_caption,
            shot_detect=args.shot_detect,
        )
        print(result.meta.model_dump_json(indent=2)[:800])
        print(
            f"indexed {result.video_id}: {len(result.meta.frames)} frames, "
            f"{len(result.meta.transcript)} segments, {len(result.meta.scenes)} scenes, "
            f"{len(result.meta.shots)} shots"
        )
        return
    if args.cmd == "query":
        settings = load_settings(source=args.source, video_id=args.video_id)
        query = " ".join(args.text)
        pin = bool(settings.video_id or settings.source) and not args.all
        if pin:
            vid = ensure_index(settings.source, settings.video_id)
            # search_corpus applies the floor itself; this branch calls the
            # scorer directly, so it has to ask.
            hits = drop_weak_hits(search(vid, query, top_k=args.top_k), query)
            meta = load_index(vid)["meta"]
            playable = attach_clips(
                vid,
                hits,
                duration_s=meta.duration_s,
                source_video=Path(meta.video_path),
                top_k=min(3, len(hits)),
                transcript=meta.transcript,
            )
        else:
            ids = list_indexed()
            if not ids:
                raise SystemExit("No indexes. Run `vvs index` or pass --video-id / --source.")
            hits = search_corpus(query, ids, top_k=args.top_k)
            playable = attach_corpus_clips(hits, top_k=min(3, len(hits)))
        by_key = {(hit.video_id, hit.rank): hit for hit in playable}
        hits = [by_key.get((hit.video_id, hit.rank), hit) for hit in hits]
        if args.json:
            print(json.dumps([h.model_dump() for h in hits], indent=2))
        else:
            print(format_hits(hits))
        return
    if args.cmd == "app":
        from vvs.ui.app import launch

        settings = load_settings(source=args.source, video_id=args.video_id)
        launch(
            source=settings.source,
            video_id=settings.video_id,
            rebuild=args.rebuild,
            warm=not args.no_warm,
            server_name=args.host,
            server_port=args.port,
        )
        return
    if args.cmd == "list":
        if args.compare:
            left, right = args.compare
            print(format_embed_compare(embed_counts(left), embed_counts(right)))
            return
        videos = list_indexed()
        if not videos:
            print("(none)")
            return
        for vid in videos:
            try:
                print(format_embed_counts(embed_counts(vid)))
            except Exception as exc:
                print(f"{vid:24}  (unreadable: {exc})")
        return
    if args.cmd == "warm":
        warm_models()
        print("models warmed")
        return


def _bulk_index_cmd(args: argparse.Namespace) -> None:
    path = Path(args.sources_file).expanduser()
    try:
        sources = parse_sources(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc
    if not sources:
        raise SystemExit(f"No sources found in {path}")

    def on_item_start(i: int, total: int, source: str) -> None:
        print(f"[bulk] ({i}/{total}) {source}")

    result = bulk_index(
        sources,
        enable_ocr=not args.no_ocr,
        scene_caption=args.scene_caption,
        shot_detect=args.shot_detect,
        rebuild=args.rebuild,
        on_item_start=on_item_start,
    )
    for item in result.items:
        label = item.video_id or "?"
        if item.status == "indexed":
            print(f"[bulk]   indexed {label}")
        elif item.status == "skipped":
            print(f"[bulk]   skipped {label} ({item.detail})")
        else:
            print(f"[bulk]   FAILED  {item.source}: {item.detail}")
    print(f"[bulk] {result.summary()}")
    if result.failed:
        raise SystemExit(1)


def _add_source_flags(
    parser: argparse.ArgumentParser,
    *,
    optional_prefill: bool = False,
) -> None:
    source_help = "YouTube URL or local video path. Also reads VVS_SOURCE."
    video_help = "Override the derived video id. Also reads VVS_VIDEO_ID."
    if optional_prefill:
        source_help = (
            "Optional UI prefill: YouTube URL or local path. "
            "The app works with no args; the user can paste or upload in the UI. "
            "Also reads VVS_SOURCE."
        )
        video_help = (
            "Optional UI prefill / id override. Also reads VVS_VIDEO_ID."
        )
    parser.add_argument("--source", default=None, help=source_help)
    parser.add_argument("--video-id", default=None, help=video_help)


if __name__ == "__main__":
    main()
