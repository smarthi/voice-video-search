# Voice Video Search

CPU-only prototype: analyze one or more YouTube URLs or local files, then ask by text (or voice) and play matching moments from the whole archive.

The source is never hardcoded. The app starts empty unless `data/` already has indexes — then Search is unlocked and one query covers every title.

This is a visual-first retrieval demo shaped like a production video search system: every video's vectors live in the same two OpenSearch indexes (`vvs_visual`, `vvs_text`), not a file per video — see [`docs/CONFLUENCE_ONPREM.md`](docs/CONFLUENCE_ONPREM.md) for the sharded-cluster shape this scales to. Mute video is first-class — Whisper runs only when an audio track exists.

## Workflow

Two loops: **analyze once**, then **ask many times**.

```text
1. Pick a video (URL, path, or upload)
2. Optional (check both): Group shots (TransNetV2) and Describe scenes (X-Ray captions)
3. Analyze video  →  stills + OpenSearch rows + meta.json  (repeat for more titles)
4. Type a question (mic = your voice, not the video)
5. One query searches every indexed video · play top-3 clips labeled by title
```

Analyze is the expensive pass (download, decode, optional VLM). Search is a k-NN query against OpenSearch. An existing video's rows are **reused** on Analyze. **Re-analyze** rebuilds from scratch.

### Mute video vs talking-head

| | Mute / no audio | Interview with speech |
| --- | --- | --- |
| Whisper | Skipped (no track, or `VVS_FORCE_MUTE=1`) | Runs |
| Stills | TransNetV2 shots (opt-in) or 2 fps keyframes | Same |
| Text index | OCR + optional scene captions | Those plus transcript |
| Clip | Silent shot window (`clip_vis_*`) | Full Whisper paragraph (`clip_asr_*`) |
| Citation | Still, `shot_id`, caption / OCR | Transcript quote + still |

A mute query is pictures and on-screen/captioned words only. “What did they say about robotics” has nothing to hit. “Red car in the tunnel” is Qwen-VL (+ caption/OCR if those words were written down).

## Design

```text
source → video.mp4
      → optional TransNetV2 shots → 1 still / shot (+ 8s floor in long takes)
      → else keyframes (2 fps + dHash + 4s floor)
      → L1 motion + luma flash on the same decode (force-keep lightning)
      → Qwen3-VL-Embedding-2B → vvs_visual (video_id-tagged rows in the shared index)
      → RapidOCR       → BGE-small   → vvs_text (video_id-tagged rows in the shared index)
      → Whisper ASR    → BGE-small   ↗   (only if the file has audio)
      → scene cluster + Qwen3-VL-2B-Instruct captions → BGE ↗   (opt-in X-Ray)

query  → Qwen-VL text + BGE query
      → k-NN each collection, filtered to this video's rows (or every video for an archive search)
      → lexical boost when a rare query word appears in transcript / OCR / caption
      → reciprocal rank fusion onto a still (per video; UI confidence)
      → merge across videos by absolute ANN scores (visual + caption + transcript + OCR)
      → motion / flash hybrid boost when the query asks for movement or a strike
      → hydrate payload (asset_id, path, timestamp, shot_id, motion, quote) from the same OpenSearch row
      → play clip from that video + original still + confidence
```

**Why two indexes.** Qwen-VL (2048-d for the 2B model) and BGE (384-d) are different embedding spaces, and OpenSearch's `knn_vector` field has one fixed width per index — visual frames and text are never mixed in one collection. Both collections are shared across the whole archive, distinguished by a `video_id` field on every row, not one file per video. See [`retrieve/opensearch_store.py`](src/vvs/retrieve/opensearch_store.py) for why (and its one real limitation: a shared index can't mix visual encoders of different widths).

**Visual encoder.** Default is `Qwen/Qwen3-VL-Embedding-2B` (shared image/text space, 2048-d). Official VL embedding sizes are **2B** and **8B** only — there is no Qwen-VL-Embedding-4B (`Qwen3-Embedding-4B` is text-only). `VVS_VISUAL_MODEL=qwen-vl-emb-4B` aliases to the 2B VL model and logs that. CLIP ViT-B-32 remains available via `VVS_VISUAL_ENCODER=clip` so existing 512-d indexes still search. Switching encoder requires **Re-analyze**.

**WeMM as an alternative encoder.** `VVS_VISUAL_ENCODER=wemm` selects the WeMM
embedding family (`wemm-2b` 2048-d, `wemm-4b` 2560-d, `wemm-9b` 4096-d), which
uses `encode_query` / `encode_document` instead of Qwen's prompt argument. WeMM
is Matryoshka-trained, so `VVS_VISUAL_TRUNCATE_DIM` can shorten vectors to a
supported width to trade recall for index size.

Note that WeMM-2B and Qwen-VL-Embedding-2B are **both 2048-d**, so dimension
alone cannot identify an index. Indexes therefore record `models.visual_encoder`
in `meta.json`, and encoder detection reads that field first, falling back to the
model name and only then to dimension. Do not reintroduce dimension-first
inference: it silently mismatches the two 2048-d families.

To compare encoders on identical inputs, `retrieve/reembed.py` clones an existing
index and re-encodes only the visual channel, reusing the same keyframes,
transcript and OCR. `scripts/compare_encoders.py` then scores the two indexes
against each other (recall@k, MRR, nDCG when timestamp judgments exist; rank
overlap and Spearman agreement when they do not, plus encode and search latency).

**Device.** The visual encoder is the slowest stage by far, so it picks the best
device automatically: Metal (`mps`) on Apple Silicon, then CUDA, then CPU.
`VVS_DEVICE=mps|cuda|cpu` overrides, and an unavailable choice falls back to CPU
with a log line rather than crashing. Precision and batch width follow the
device — fp16 and 8 frames per pass on a GPU, fp32 and 1 frame on CPU, since CPU
fp16 is emulated and slower than fp32. Override with `VVS_VISUAL_DTYPE` and
`VVS_VISUAL_BATCH`.

Three guards keep the GPU path from being worse than the CPU one. `vvs/__init__.py`
sets `PYTORCH_ENABLE_MPS_FALLBACK=1` **before anything imports torch** — PyTorch
reads it when the MPS backend initializes, so setting it later does nothing and a
VL op with no Metal kernel takes the process down. If the model still fails to
load on the GPU, the encoder pins itself to CPU for the rest of the session
rather than retrying a broken device once per call. And a one-shot probe after
load checks the vectors are finite: if fp16 overflowed to NaN the model is
rebuilt in fp32, since NaN vectors would poison the index instead of failing
loudly.

**Troubleshooting.** The startup line prints `[embed] loading <model> (<device>,
<dtype>)`. If it says anything but `mps` on Apple Silicon, that is why embedding
is slow, and it usually means a CPU-only torch build. If Python dies during
search with `leaked semaphore objects to clean up at shutdown`, that is a native
crash rather than a Python error — the message is the resource tracker's parting
note, not the cause. Set `VVS_DEVICE=cpu` to confirm the GPU is implicated, then
`VVS_VISUAL_DTYPE=float32` to test whether it is specifically half precision.

**Duplicate native libraries (historical — the fatal one is retired).** This
stack used to carry `faiss-cpu` for on-disk ANN, and it collided with torch on
macOS in a way worth recording even though the dependency is gone: FAISS
search now runs server-side in OpenSearch (`retrieve/opensearch_store.py`),
not as an in-process library, so there is no second copy of anything for
torch to collide with. Left here because the *mechanism* that fixed it
(`vvs/compute.py`) is still load-bearing for an unrelated crash — see below.

*OpenMP (fatal, retired with FAISS).* `faiss-cpu` and `torch` each bundled
their own LLVM OpenMP runtime — on macOS arm64, `faiss/.dylibs/libomp.dylib`
and `torch/lib/libomp.dylib`. Two consequences, and both had to be handled:

1. Whichever runtime initializes second makes OpenMP abort the process.
   `vvs/__init__.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` before torch or faiss can
   be imported, which is the documented escape hatch for that.
2. Past that abort, `swigfaiss.search`'s OpenMP parallel region
   (`__kmpc_fork_call` → `__kmp_join_barrier` → `__kmp_suspend_initialize_thread`)
   segfaults the moment it is entered from a **second distinct OS thread**, warm
   or not — confirmed by direct repro, not just inference from crash logs.
   Pinning FAISS to one thread (`faiss_thread_default()`, in the now-removed
   `retrieve/faiss_store.py`) does **not** avoid
   this: `__kmpc_fork_call` still runs the full fork/join barrier machinery
   regardless of team size. Priming the runtime on the main thread first doesn't
   help either — the crash isn't main-vs-background, it's "first thread ever to
   call in" vs "any other thread calling in later." Every Gradio request is
   served on a different worker thread, so search crashed in the UI while the
   same query from the CLI (a single process, one thread, done) was fine.
   `faulthandler` pinned it to `swigfaiss.search` called from `ui/app.py`.

The fix for (2) was `vvs/compute.py`: a single dedicated background thread
that every heavy call in the UI is funneled through via
`compute.run(fn, *args, **kwargs)`. That module is still very much in use —
it turned out transformers' parallel weight-loading path hits a *different*
same-shape crash (a background-thread-only segfault) the first time a model
loads, independent of FAISS. See the docstring in `vvs/compute.py` for that
still-active bug. `KMP_DUPLICATE_LIB_OK=TRUE` is kept in `vvs/__init__.py` as
a harmless defensive default (see its updated comment) even though FAISS no
longer needs it, since onnxruntime (via rapidocr) is still a native library
in this process.

*Diagnosing a silent death.* A native crash in torch, onnxruntime or the
bundled FFmpeg kills the process with a signal, so Python raises nothing and the
only evidence is `resource_tracker: There appear to be 1 leaked semaphore
objects` at shutdown — which names neither the step nor the library. Two things
help. `vvs/__init__.py` enables `faulthandler`, which prints a C-level traceback
on a fatal signal (`VVS_FAULTHANDLER=0` opts out). And `scripts/diagnose_search.py`
walks the search path one step at a time, flushing before each, so the last line
on screen is the step that died:

```bash
uv run python scripts/diagnose_search.py "your query"
VVS_DEVICE=cpu           uv run python scripts/diagnose_search.py   # rule out Metal
VVS_VISUAL_DTYPE=float32 uv run python scripts/diagnose_search.py   # rule out fp16
```

macOS also writes a full report per crash; the top frames name the guilty dylib:

```bash
ls -t ~/Library/Logs/DiagnosticReports/*.ips | head -1 | xargs head -60
```

*Surviving a crash.* An abort can kill the process mid-write, and `meta.json`
carries the transcript, scene captions and OCR text — none of which can be
rebuilt from the OpenSearch collections, so losing it means re-captioning the whole video.
It is written atomically (temp file, `fsync`, `os.replace`) with the previous copy
rotated to `meta.json.bak`, and readers fall back to that backup when the primary
will not parse. This matters because a truncated `meta.json` is worse than a
missing one: `index_exists()` only tests for presence, so before this change a
half-written file left the video listed but unloadable, and it simply disappeared
from the library with no explanation. To find already-damaged indexes from an
earlier crash:

```bash
for m in data/*/meta.json; do python -c "import json,sys;json.load(open(sys.argv[1]))" "$m" \
  2>/dev/null || echo "CORRUPT: $m"; done
```

Anything listed there predates the atomic write and needs **Re-analyze**, since
there is no backup to recover from yet.

*Surviving a crash mid-rebuild.* The same atomicity now extends to the OpenSearch
rows, not just `meta.json`. Every `build_index()` call gets a fresh
`index_generation` token; new rows are written tagged with it while the
*previous* generation's rows stay fully intact and queryable throughout the
rebuild. Only after every channel has been written and `meta.json` is atomically
replaced with the new token do readers switch over — then the old generation's
rows are garbage-collected. A crash mid-rebuild (or a search that lands mid-rebuild)
sees the complete previous generation, never a half-written one. Indexes from
before this existed have no token (`index_generation: null`) and are read as
their own implicit "legacy" generation — no migration needed.

*FFmpeg / AVFoundation (noisy, not fatal).* PyAV and OpenCV each bundle
`libavdevice`, which on macOS registers the ObjC classes `AVFFrameReceiver` and
`AVFAudioReceiver`. Two copies means two registrations, and the runtime says so at
`stage=ocr` where RapidOCR first imports cv2. It warns about "mysterious crashes",
but in practice captioning and indexing run fine with the warning present, so
treat it as noise until proven otherwise. Note that `opencv-python-headless` drops
`libavdevice` **only on manylinux wheels** — the macOS wheels of both the full and
headless builds still ship it, so the swap does not silence this on a Mac. We use
headless anyway: nothing here opens an OpenCV window, RapidOCR only uses cv2 for
array work, and it removes the duplicate on Linux. Eliminating it on macOS would
mean running OCR out-of-process or dropping cv2 altogether.

**Measured CPU cost (this laptop, 4 frames after warmup).** Qwen-VL-Embedding-2B
28.3 s/frame; WeMM-Embedding-2B 72.8 s/frame — **2.6× slower** for the same
2048-d output. Text queries are comparable (9.8 s vs 10.2 s). At that rate,
re-embedding the 1,332-frame sample archive with WeMM is ~27 hours, so the
**quality A/B is deferred to a GPU box**; the re-embed and comparison harness are
ready to run there unchanged. Both models emit 2048-d L2-normalized vectors,
which is exactly why `models.visual_encoder` has to be recorded per index.

**Archive search.** Every video's rows live in the same `vvs_visual` index, filtered by `video_id`. A query embeds once, searches each video's rows in turn, then **merges by absolute ANN scores**:

```text
corpus_relevance = max( visual + 1.15×caption + 0.55×OCR,
                        1.4×transcript )
                 + 0.2×lexical overlap
                 + 0.45×motion/flash   # only if the query asks for that
```

Picture path and spoken path **compete**, they are not summed. BGE-small cosine is ~0.65 even on unrelated talk, so adding transcript onto every interview made “trumpet” return the interview clip over Jennifer Connelly’s saxophone still. A real spoken hit (`robotics`, lexical-boosted ANN) still wins the max() against a random music-video picture. **Motion is a sidecar, not a VLM clip:** the encoder still sees one JPEG. Decode computes L1 / luma-jump on a 64×36 gray; lightning stills are force-kept. “Moving tornado” boosts high `motion_energy`; “lightning” boosts `flash_energy`; “tornado” does not. Old indexes default both scalars to 0 — **Re-analyze** to populate them. Per-video RRF **confidence** is only the UI badge. We do **not** keep-or-drop titles because a caption contains the query words. One strong title may take every slot. Pin one file with `vvs query --video-id ID`. Production is one visual collection + one text collection with `asset_id` in the payload — same ranker, no per-movie file walk.

**OpenSearch k-NN is the only ANN here, not BM25.** Every payload field (`path`, `t_s`, `modality`, `text`, `shot_id`, `motion_energy`, `flash_energy`) lives on the same document as its `knn_vector`, so there is no separate payload store to keep in sync. Lexical ranking and the motion/flash boost still run in Python on the ANN candidates only — OpenSearch's BM25 engine is available on these text fields but intentionally unused (see the `opensearch_store.py` module docstring for why: adopting it would fuse visual and text scores server-side, discarding the per-channel ranks the RRF fusion below depends on). If k-NN never retrieved the row, those boosts never see it.

**`VectorStore` and OpenSearch.** The app talks only to a `VectorStore` (`create`, `add`, `search`, `close`) — see `retrieve/store.py`. `OpenSearchStore` (`retrieve/opensearch_store.py`) is the only implementation: two shared indexes, `vvs_visual` and `vvs_text`, across the whole archive, each row tagged with `video_id`. `docker compose up -d` runs a single local node; `docs/CONFLUENCE_ONPREM.md` §6–8 covers the sharded-cluster version of the same document shape.

**Query expansion.** Qwen-VL's dual encoder is lexically brittle: “river” and
“running water” can land in noticeably different places in embedding space
despite meaning the same thing to a person. Before embedding, `expand_query_terms()`
(`media/caption.py`) asks the already-loaded Qwen3-VL-2B-Instruct model — same
bundle X-Ray captions use, a text-only prompt this time, no second model
download — for `VVS_QUERY_EXPAND_VARIANTS` (default 2) alternate phrasings.
The original query plus its expansions are embedded in one batched call and
k-NN searched individually, then merged by **max score per candidate** — every
list here is the same ANN space, so max is the right merge, unlike the
BM25-vs-cosine case below. Shaped like `media/motion.py`'s `query_motion_flags`:
a small, directly testable function called inline from `search()`, just backed
by a model instead of a keyword set. Never blocks a search: a disabled setting,
a missing/broken model, or an unusable generation all fall back to the literal
query alone. `VVS_QUERY_EXPAND=0` disables it outright; `warm_models()` warms
this model too when it is on, so a live search does not pay its load time.

**Hybrid ranking.** Dense ANN alone can prefer generic “corporate / strategy” talk over a paragraph that actually says “robotics”. After ANN, hits that contain rare query terms get a lexical IDF boost (`0.14 × idf`), then RRF (`k=60`):

| Channel | Weight | When |
| --- | --- | --- |
| visual | 1.0 | always |
| transcript | 1.6 | audio exists |
| OCR | 0.6 | score ≥ 0.28 |
| scene caption | 1.0 | captions were built |
| ASR window / anchor | 0.85 / 1.25 | ASR only |

Mute video: visual + OCR + scene. No time-anchor boosts.

That table is **inside one video**. The UI confidence badge is RRF / ceiling. Archive order uses `corpus_relevance` above, not that badge. Examples: `robotics` → interview clip ~3:01; `clips with a trumpet` → Careless Whisper / Jennifer Connelly saxophone still; `snow-covered mountains` → Roja; `moving tornado` prefers the high-motion still of the same funnel.

**X-Ray scene cards.** The visual embedder does not write English. When captions are on, stills are grouped (TransNet shots if present, else embedding cosine / dHash, 24 s cap). **Qwen3-VL-2B-Instruct** captions the representative still and lists objects (`VVS_SCENE_CAPTION_MODEL=qwen_vl`). That is a second checkpoint from the embedder — Instruct generates text; Embedding only writes vectors. **BLIP-base** is the fallback if Instruct fails, and the run falls back **once** rather than once per scene. Cards live in `meta.json` and are BGE-embedded as `modality="scene"`. Query time only does a playhead lookup plus ANN — the captioner does not run again.

Captioning uses the same device and precision as the embedder (see **Device** above), which matters more here than anywhere else: captioning is autoregressive, so each scene costs up to `QWEN_MAX_TOKENS` sequential forward passes rather than the single pass an embedding needs. If a caption model fails to load on the GPU, the session pins itself to CPU exactly as the encoder does.

**TransNetV2 shots.** Mute video does not need a visual vector every 2 fps. TransNetV2 finds hard cuts on 48×27 frames (PyAV, no ffmpeg CLI), decoded and scored in a bounded sliding window rather than held as one full-video RGB array — only timestamps, transition scores and the tiny motion samples grow with duration, not a per-frame image buffer, so a long video no longer costs memory proportional to its frame count. Each shot keeps one still, plus another every `VVS_SHOT_MAX_GAP_S` (default 8 s) so a long take is not a single embedding. The same decode yields `motion_energy` / `flash_energy` inline as it streams; luma spikes add extra stills so lightning is not dropped. `shot_id` is stored on the keyframe and in the OpenSearch payload for citations. Mute clips use the shot span, not ±2 s around a keyframe. If TransNetV2 fails, analyze falls back to the 2 fps path.

TransNetV2 is **shot-boundary** detection (cuts / fades), not semantic “this is a concert.” Object crops (`segment_id`) are a later pass on these stills, not on every raw frame.

**Later, on a larger server: N-frame VLM windows.** `Qwen3-VL-Embedding` already accepts video (`{"video": frames, "fps": 4, "max_frames": 8}`), not only a still. This laptop path does **not** call that — one JPEG per vector, plus the L1/luma sidecar. When a GPU or a bigger CPU box is available, encode a short window around each kept still (about 1 s, 6–8 frames) into the **same** 2048-d `vvs_visual` index. Then `moving tornado` can match motion in the pixels. Do not embed the whole `video.mp4` as one vector (the playhead disappears). Lightning still needs the force-keep so the strike is inside the window. That pass is opt-in and needs **Re-analyze**; it is not on this CPU prototype.

## How to run

### Setup

```bash
cd video-voice-search
uv sync
docker compose up -d   # starts a local single-node OpenSearch on :9200
```

Needs Python 3.12+, `uv`, Docker, ~8 GB RAM (Qwen-VL embedding 2B is heavier than CLIP; Instruct captioning loads a second 2B VL). First run needs network for Qwen-VL Embedding / BGE / RapidOCR, plus Whisper (if audio), TransNetV2, and Qwen3-VL-Instruct (or BLIP) if captions are on. After that, query is offline (OpenSearch itself needs no network once its image is pulled). CLIP indexes built before the Qwen-VL swap still search; new analyzes use Qwen-VL. `VVS_OPENSEARCH_URL` points at a different instance if you're not running the bundled compose file.

### Live demo UI

```bash
uv run vvs app
```

Open the printed `http://127.0.0.1:7860` URL from Windows. Then:

1. Paste a YouTube URL or local path, or upload `.mp4` / `.webm` / `.mkv` / `.mov` / `.avi` / `.m4v`.
2. Optional analyze options (independent — check **both** to run in one pass): **Group shots with TransNetV2**, then **Describe scenes** (Qwen3-VL-2B-Instruct captions those shot-grouped scenes).
3. Click **Analyze video**. Repeat for a second title if you want archive hits from more than one file — or use **01b · Bulk analyze** to paste/upload several at once (see below).
4. Type a question → **Search**. That query searches **every** indexed video. Clips are labeled by title. Mic is optional and only useful when a file has speech.

#### 00 · Library

The top section lists every indexed video as a card: a preview clip, title, id,
duration, and chips for keyframes, spoken segments (or `mute`), OCR, X-Ray scenes
and shots, plus the encoder that built the index. Cards flag `video file missing`
when `video.mp4` is gone, which is what a half-cleaned directory looks like.

- **Hover** a tile to play its preview; leaving rewinds it. Tap toggles on touch.
- **Search this** narrows every subsequent query to that one video, and the card
  gets a red ring. **Search whole archive** widens it again.
- **Filter** matches title or id. **Sort** by recency, title, duration or keyframes.
- **Delete** is two-step: the card swaps to **Cancel** / **Confirm delete**, and
  confirming removes that `data/<id>/` directory for good.

Previews are cut once and cached as `data/<id>/preview.mp4`: 6 s, 480px wide,
15 fps, silent, roughly 20–350 KB each. For videos with speech the cut starts at
the first transcript segment past the first 8% so a talk opens on someone
talking rather than a title card; mute assets use a fixed offset. Tiles are
`preload="none"`, so a full grid downloads no video until you hover one. Build
them for a whole archive up front with **Build previews** (14 videos took ~9 s).

Prefill without auto-analyze:

```bash
uv run vvs app --source 'https://www.youtube.com/watch?v=fBEGOk3ZHw4'
uv run vvs app --video-id fBEGOk3ZHw4
```

On stage, warm weights first:

```bash
uv run vvs warm
uv run vvs app
```

### Index (CLI)

Talking-head (Whisper if there is audio, 2 fps keyframes):

```bash
uv run vvs index --source 'https://www.youtube.com/watch?v=fBEGOk3ZHw4'
uv run vvs index --source /path/to/talk.mp4 --video-id my-talk
```

A local source's derived id is a slug of the filename plus a hash of its resolved
path, not the filename alone — two different `demo.mp4` files in different
folders no longer silently collide into one archive entry. An explicit
`--video-id` (here, from a bulk manifest, or from `--dest-id`) is validated
against `^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$` before it ever touches a filesystem
path, so `--video-id '../../etc'` fails loudly instead of writing outside `data/`.

Mute video (shots + no Whisper + optional captions):

```bash
VVS_FORCE_MUTE=1 uv run vvs index --source /path/to/mute.mp4 --shot-detect --scene-caption
# or
export VVS_FORCE_MUTE=1 VVS_SHOT_DETECT=1 VVS_SCENE_CAPTION=1
uv run vvs index --source /path/to/mute.mp4
```

Flags match the UI checkboxes: `--shot-detect` / `--no-shot-detect`, `--scene-caption` / `--no-scene-caption`. Defaults follow the `VVS_*` env vars.

### Bulk index (many videos)

`--sources-file` indexes a whole manifest in one call — one YouTube URL or local path per line, `#` comments and blank lines ignored:

```bash
cat > videos.txt <<'EOF'
# a batch of titles
https://www.youtube.com/watch?v=fBEGOk3ZHw4
/path/to/talk.mp4
/path/to/mute-clip.mp4
EOF

uv run vvs index --sources-file videos.txt --shot-detect --scene-caption
```

Each line is processed in turn (this stays a plain sequential loop — see `retrieve/bulk.py` for why). Already-indexed videos are skipped so re-running the same manifest is cheap; pass `--rebuild` to force every line to reprocess. One bad URL or unreadable file is reported and skipped, not fatal to the batch — the command exits non-zero only if something failed, after every line has had a chance to run. `--no-ocr`, `--scene-caption` / `--shot-detect` apply to every video in the file.

The **01b · Bulk analyze** section in the live UI does the same thing: paste one URL/path per line and/or drop multiple files, click **Bulk analyze**, and the archive (library grid + query) picks up everything that succeeded.

### Query

One query searches **every** indexed video (whole archive). Pass `--video-id` to pin one title.

```bash
uv run vvs query 'red car in the tunnel'
uv run vvs query --all 'product on stage'
uv run vvs query --video-id fBEGOk3ZHw4 'what is the approach to robotics'
```

### List indexes

```bash
uv run vvs list
```

## Compare embeddings: TransNetV2 vs 2 fps

**Visual embeddings = number of stills the visual encoder wrote** (`len(meta.frames)`, same as this video's row count in `vvs_visual`). That is the TransNetV2 vs 2 fps number. Text vectors (OCR / captions / ASR) are a separate count.

Index the **same file twice** under two ids so they do not overwrite each other:

```bash
uv run vvs index --source /path/to/video.mp4 --video-id cmp-2fps
uv run vvs index --source /path/to/video.mp4 --video-id cmp-shots --shot-detect

uv run vvs list --compare cmp-2fps cmp-shots
```

Example:

```text
cmp-2fps                  2fps        visual=149    shots=0     text=42     263s
cmp-shots                 TransNetV2  visual=41     shots=38    text=12     263s
compare visual: 149 → 41  ·  108 fewer visual embeddings (72% less)
```

`vvs list` prints those columns for every index under `data/`. After a single analyze, the last log line already has `frames` / `shots`. In `meta.json`, `frames` is the embedding list; `shots` is non-empty only when TransNetV2 ran.

For a mute clip, add `VVS_FORCE_MUTE=1` on both index commands so Whisper does not change the text side of the comparison.

## Example demo video

Optional paste-in (not a library default):

```text
https://www.youtube.com/watch?v=fBEGOk3ZHw4
```

Rehearsed questions (top-3 window):

1. *what is hybrid AI*
2. *show me the Motorola products and Kira*
3. *personalized knowledge base on phone tablet and PC*
4. *robotics and cybersecurity research*
5. *agent-native devices voice and gesture interfaces*

```bash
uv run python scripts/validate.py --video-id fBEGOk3ZHw4 --write-demo
```

This interview is a talking-head check. Real archive content is mute and multi-title: index two files, then `vvs query 'red car'` with no `--video-id`. Use `--shot-detect` and visual / caption queries.

## On-disk layout

Outputs in `data/<video_id>/`:

| Path | What |
| --- | --- |
| `video.mp4` | Ingested source |
| `frames/` | Representative JPEGs (2 fps set, or TransNet stills) |
| `clips/` | On-demand H.264 hits |
| `preview.mp4` | Cached 6 s library preview (silent, 480px). Outside `clips/` on purpose — `attach_clips` prunes foreign names there |
| `meta.json` | Title, duration, frames, shots, transcript, OCR, scene cards, channels |
| `meta.json.bak` | Previous `meta.json`, rotated on every write. Read only when the primary will not parse |
| `timings.json` | Stage seconds |

The vectors and their payload (`asset_id`, `shot_id`, path, `t_s`, `text`, `motion_energy`, `flash_energy`) are **not** under `data/`: they live in OpenSearch, in the shared `vvs_visual` (Qwen-VL) and `vvs_text` (BGE: OCR / transcript / scene) indexes, one row per video per still or text observation, filtered by `video_id`. Deleting `data/<video_id>/` alone leaves those rows behind — use `vvs`'s delete flow (UI **Delete**, or `delete_index()`), which purges both.

## Package layout

```text
src/vvs/
  cli.py config.py models.py paths.py
  ingest/     YouTube + local ingest
  media/      frames, shots (TransNetV2), motion/flash, clips, transcribe, ocr, scenes, caption
  retrieve/   embed, index, search, reembed, VectorStore (OpenSearch)
  ui/         Gradio + custom theme + library grid
tests/        store / mute / shots / scenes / corpus / motion / library (no VLM download)
```

Commands: `uv run vvs index|query|app|warm|list`.

## Environment

| Variable | Meaning |
| --- | --- |
| `VVS_SOURCE` / `VVS_VIDEO_ID` | Optional CLI / UI prefill |
| `VVS_DATA_DIR` | Index root (default `data/`) |
| `VVS_HOST` / `VVS_PORT` | Gradio bind (default `0.0.0.0:7860`) |
| `VVS_OPENSEARCH_URL` | OpenSearch endpoint (default `http://localhost:9200`; `docker compose up -d` starts one) |
| `VVS_OPENSEARCH_TIMEOUT_S` | Client request timeout (default `10`) |
| `VVS_FORCE_MUTE` | Skip Whisper even if the file has audio |
| `VVS_SCENE_CAPTION` | Caption visual scenes at analyze (X-Ray metadata) |
| `VVS_SCENE_CAPTION_MODEL` | `qwen_vl` (default, Qwen3-VL-2B-Instruct) or `blip` |
| `VVS_QUERY_EXPAND` | Query-time LLM phrasing expansion (default on); `0` disables it |
| `VVS_QUERY_EXPAND_VARIANTS` | Alternate phrasings to generate per query (default `2`) |
| `VVS_SHOT_DETECT` | TransNetV2 shot grouping; embed representatives only |
| `VVS_SHOT_THRESHOLD` | Cut threshold (default `0.5`) |
| `VVS_SHOT_MAX_GAP_S` | Extra still every N seconds inside a long shot (default `8`) |
| `VVS_VISUAL_ENCODER` | `qwen_vl` (default), `wemm`, or `clip` (old 512-d indexes) |
| `VVS_VISUAL_MODEL` | HF id or alias (`qwen-vl-emb-4B` → official 2B VL; `8b` → 8B; `wemm-2b` / `wemm-4b` / `wemm-9b`) |
| `VVS_VISUAL_TRUNCATE_DIM` | Matryoshka width for WeMM only; must be a supported dim |
| `VVS_DEVICE` | `mps`, `cuda`, or `cpu`. Default auto-detects (mps → cuda → cpu) |
| `VVS_VISUAL_DTYPE` | `float16`, `bfloat16`, `float32`. Default fp16 on GPU, fp32 on CPU |
| `VVS_VISUAL_BATCH` | Frames per forward pass. Default 8 on GPU, 1 on CPU |
| `VVS_FAULTHANDLER` | `0` disables the C-level traceback on fatal signals |

Search weights live in `retrieve/search.py` (not env): per-video RRF (`TEXT_WEIGHT` 1.6, `OCR_WEIGHT` 0.6, `SCENE_WEIGHT` 1.0) and archive merge (`SCENE_ANN_WEIGHT` 1.15, `TEXT_ANN_WEIGHT` 1.4, `OCR_ANN_WEIGHT` 0.55, `LEXICAL_BLEND` 0.2, `MOTION_BLEND` 0.45). Motion/flash keep thresholds are in `media/motion.py` (`FLASH_KEEP` 0.22, `MOTION_KEEP` 0.18).

Docs for Confluence: [`docs/CONFLUENCE.md`](docs/CONFLUENCE.md) (diagrams in [`docs/images/`](docs/images/)). Self-contained paste-in with inlined figures: [`VOICE_VIDEO_SEARCH.md`](VOICE_VIDEO_SEARCH.md).
