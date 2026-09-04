# Voice Video Search

CPU-only prototype: analyze one or more YouTube URLs or local files, then ask by text (or voice) and play matching moments from the whole archive.

The source is never hardcoded. The app starts empty unless `data/` already has indexes — then Search is unlocked and one query covers every title.

This is a visual-first retrieval demo shaped like production Sphere search. Each video keeps its own on-disk FAISS store (Qdrant-shaped for later). A query fans out across those stores and merges; it does not require one shared index. Sphere-style mute video is first-class — Whisper runs only when an audio track exists.

## Workflow

Two loops: **analyze once**, then **ask many times**.

```text
1. Pick a video (URL, path, or upload)
2. Optional (check both): Group shots (TransNetV2) and Describe scenes (X-Ray captions)
3. Analyze video  →  stills + FAISS + SQLite + meta.json  (repeat for more titles)
4. Type a question (mic = your voice, not the video)
5. One query searches every indexed video · play top-3 clips labeled by title
```

Analyze is the expensive pass (download, decode, optional VLM). Search is local ANN + payload lookup. Existing `data/<id>/visual.faiss` is **reused** on Analyze. **Re-analyze** rebuilds from scratch.

### Mute Sphere vs talking-head

| | Sphere / no audio | Interview with speech |
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
      → Qwen3-VL-Embedding-2B → visual.faiss + visual.sqlite
      → RapidOCR       → BGE-small   → text.faiss  + text.sqlite
      → Whisper ASR    → BGE-small   ↗   (only if the file has audio)
      → scene cluster + Qwen3-VL-2B-Instruct captions → BGE ↗   (opt-in X-Ray)

query  → Qwen-VL text + BGE query
      → ANN each video's stores (mmap, top-k only) — fan-out, not one shared FAISS file
      → lexical boost when a rare query word appears in transcript / OCR / caption
      → reciprocal rank fusion onto a still (per video; UI confidence)
      → merge across videos by absolute ANN scores (visual + caption + transcript + OCR)
      → motion / flash hybrid boost when the query asks for movement or a strike
      → hydrate SQLite payload (asset_id, path, timestamp, shot_id, motion, quote)
      → play clip from that video + original still + confidence
```

**Why two indexes.** Qwen-VL (2048-d for the 2B model) and BGE (384-d) are different embedding spaces. Visual frames and text are never mixed in one FAISS file.

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
`VVS_VISUAL_BATCH`. On Metal, `PYTORCH_ENABLE_MPS_FALLBACK=1` is set before load
so the few VL ops lacking a Metal kernel run on CPU instead of aborting, and a
one-shot probe after load verifies the vectors are finite — if fp16 overflowed to
NaN the model is silently rebuilt in fp32, because NaN vectors would otherwise
poison the index instead of failing loudly.

If embedding is slow on an Apple machine, check the startup line: it prints
`[embed] loading <model> (<device>, <dtype>)`. Anything other than `mps` there is
the problem, and it usually means a CPU-only torch build.

**Measured CPU cost (this laptop, 4 frames after warmup).** Qwen-VL-Embedding-2B
28.3 s/frame; WeMM-Embedding-2B 72.8 s/frame — **2.6× slower** for the same
2048-d output. Text queries are comparable (9.8 s vs 10.2 s). At that rate,
re-embedding the 1,332-frame sample archive with WeMM is ~27 hours, so the
**quality A/B is deferred to a GPU box**; the re-embed and comparison harness are
ready to run there unchanged. Both models emit 2048-d L2-normalized vectors,
which is exactly why `models.visual_encoder` has to be recorded per index.

**Archive search.** Each video keeps its own `visual.faiss` (POC fan-out). A query embeds once, searches every store, then **merges by absolute ANN scores**:

```text
corpus_relevance = max( visual + 1.15×caption + 0.55×OCR,
                        1.4×transcript )
                 + 0.2×lexical overlap
                 + 0.45×motion/flash   # only if the query asks for that
```

Picture path and spoken path **compete**, they are not summed. BGE-small cosine is ~0.65 even on unrelated talk, so adding transcript onto every interview made “trumpet” return Tolga over Jennifer Connelly’s saxophone still. A real spoken hit (`robotics`, lexical-boosted ANN) still wins the max() against a random music-video picture. **Motion is a sidecar, not a VLM clip:** the encoder still sees one JPEG. Decode computes L1 / luma-jump on a 64×36 gray; lightning stills are force-kept. “Moving tornado” boosts high `motion_energy`; “lightning” boosts `flash_energy`; “tornado” does not. Old indexes default both scalars to 0 — **Re-analyze** to populate them. Per-video RRF **confidence** is only the UI badge. We do **not** keep-or-drop titles because a caption contains the query words. One strong title may take every slot. Pin one file with `vvs query --video-id ID`. Production is one visual collection + one text collection with `asset_id` in the payload — same ranker, no per-movie file walk.

**FAISS + SQLite is not “SQLite does keywords.”** FAISS is the only ANN. SQLite is the payload sidecar (`vector_id` → path, `t_s`, `modality`, `text`, `shot_id`, `motion_energy`, `flash_energy`). Lexical ranking and the motion/flash boost run in Python on the ANN candidates only. There is no FTS5 / BM25 table. If FAISS never retrieved the row, those boosts never see it.

**Why FAISS here / Qdrant later.** The app talks only to a `VectorStore` (`create`, `add`, `search`, `close`). `FaissStore` mmaps `*.faiss` and reads payload from SQLite. `QdrantStore` uses the same interface. A production cluster would use auto-sharding (`shard_number ≈ nodes`) in a single DC; custom `hash(asset_id) % N` only when every query is already scoped to one asset. Set `VVS_VECTOR_BACKEND=qdrant` and `VVS_QDRANT_URL` when a cluster exists.

**Hybrid ranking.** Dense ANN alone can prefer generic “Lenovo / strategy” talk over a paragraph that actually says “robotics”. After ANN, hits that contain rare query terms get a lexical IDF boost (`0.14 × idf`), then RRF (`k=60`):

| Channel | Weight | When |
| --- | --- | --- |
| visual | 1.0 | always |
| transcript | 1.6 | audio exists |
| OCR | 0.6 | score ≥ 0.28 |
| scene caption | 1.0 | captions were built |
| ASR window / anchor | 0.85 / 1.25 | ASR only |

Mute video: visual + OCR + scene. No time-anchor boosts.

That table is **inside one video**. The UI confidence badge is RRF / ceiling. Archive order uses `corpus_relevance` above, not that badge. Examples: `robotics` → Tolga ~3:01; `clips with a trumpet` → Careless Whisper / Jennifer Connelly saxophone still; `snow-covered mountains` → Roja; `moving tornado` prefers the high-motion still of the same funnel.

**X-Ray scene cards.** The visual embedder does not write English. When captions are on, stills are grouped (TransNet shots if present, else embedding cosine / dHash, 24 s cap). **Qwen3-VL-2B-Instruct** captions the representative still and lists objects (`VVS_SCENE_CAPTION_MODEL=qwen_vl`). That is a second checkpoint from the embedder — Instruct generates text; Embedding only writes vectors. **BLIP-base** is the CPU fallback if Instruct fails. Florence-2 is still available via `VVS_SCENE_CAPTION_MODEL=florence2`. Cards live in `meta.json` and are BGE-embedded as `modality="scene"`. Query time only does a playhead lookup plus ANN — the captioner does not run again.

Florence (if selected) uses the **native** `Florence2ForConditionalGeneration` in transformers 5 (`florence-community/Florence-2-base`). Do not use Microsoft’s old `trust_remote_code` config: transformers 5.x raises `'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'`. If the chosen captioner cannot load, the run falls back to BLIP **once** (not once per scene).

**TransNetV2 shots.** Mute Sphere video does not need a visual vector every 2 fps. TransNetV2 finds hard cuts on 48×27 frames (PyAV, no ffmpeg CLI). Each shot keeps one still, plus another every `VVS_SHOT_MAX_GAP_S` (default 8 s) so a long take is not a single embedding. The same 48×27 stack yields `motion_energy` / `flash_energy`; luma spikes add extra stills so lightning is not dropped. `shot_id` is stored on the keyframe and in the FAISS payload for citations. Mute clips use the shot span, not ±2 s around a keyframe. If TransNetV2 fails, analyze falls back to the 2 fps path.

TransNetV2 is **shot-boundary** detection (cuts / fades), not semantic “this is a concert.” Object crops (`segment_id`) are a later pass on these stills, not on every raw frame.

**Later, on a larger server: N-frame VLM windows.** `Qwen3-VL-Embedding` already accepts video (`{"video": frames, "fps": 4, "max_frames": 8}`), not only a still. This laptop path does **not** call that — one JPEG per vector, plus the L1/luma sidecar. When a GPU or a bigger CPU box is available, encode a short window around each kept still (about 1 s, 6–8 frames) into the **same** 2048-d visual FAISS. Then `moving tornado` can match motion in the pixels. Do not embed the whole `video.mp4` as one vector (the playhead disappears). Lightning still needs the force-keep so the strike is inside the window. That pass is opt-in and needs **Re-analyze**; it is not on this CPU prototype.

## How to run

### Setup

```bash
cd video-voice-search
uv sync
# optional Qdrant client:
# uv sync --extra qdrant
```

Needs Python 3.12+, `uv`, ~8 GB RAM (Qwen-VL embedding 2B is heavier than CLIP; Instruct captioning loads a second 2B VL). First run needs network for Qwen-VL Embedding / BGE / RapidOCR, plus Whisper (if audio), TransNetV2, and Qwen3-VL-Instruct (or Florence/BLIP) if captions are on. After that, query is offline. CLIP indexes built before this swap still search; new analyzes use Qwen-VL.

### Live demo UI

```bash
uv run vvs app
```

Open the printed `http://127.0.0.1:7860` URL from Windows. Then:

1. Paste a YouTube URL or local path, or upload `.mp4` / `.webm` / `.mkv` / `.mov` / `.avi` / `.m4v`.
2. Optional analyze options (independent — check **both** to run in one pass): **Group shots with TransNetV2**, then **Describe scenes** (Qwen3-VL-2B-Instruct captions those shot-grouped scenes).
3. Click **Analyze video**. Repeat for a second title if you want archive hits from more than one file.
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

Mute Sphere-style (shots + no Whisper + optional captions):

```bash
VVS_FORCE_MUTE=1 uv run vvs index --source /path/to/mute.mp4 --shot-detect --scene-caption
# or
export VVS_FORCE_MUTE=1 VVS_SHOT_DETECT=1 VVS_SCENE_CAPTION=1
uv run vvs index --source /path/to/mute.mp4
```

Flags match the UI checkboxes: `--shot-detect` / `--no-shot-detect`, `--scene-caption` / `--no-scene-caption`. Defaults follow the `VVS_*` env vars.

### Query

One query searches **every** indexed video (Sphere archive). Pass `--video-id` to pin one title.

```bash
uv run vvs query 'red car in the tunnel'
uv run vvs query --all 'product on stage'
uv run vvs query --video-id fBEGOk3ZHw4 'what is lenovo approach to robotics'
```

### List indexes

```bash
uv run vvs list
```

## Compare embeddings: TransNetV2 vs 2 fps

**Visual embeddings = number of stills the visual encoder wrote** (`len(meta.frames)`, same as `visual.faiss` ntotal). That is the TransNetV2 vs 2 fps number. Text vectors (OCR / captions / ASR) are a separate count.

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

For a mute Sphere clip, add `VVS_FORCE_MUTE=1` on both index commands so Whisper does not change the text side of the comparison.

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

This interview is a talking-head check. Real Sphere content is mute and multi-title: index two files, then `vvs query 'red car'` with no `--video-id`. Use `--shot-detect` and visual / caption queries.

## On-disk layout

Outputs in `data/<video_id>/`:

| Path | What |
| --- | --- |
| `video.mp4` | Ingested source |
| `frames/` | Representative JPEGs (2 fps set, or TransNet stills) |
| `clips/` | On-demand H.264 hits |
| `preview.mp4` | Cached 6 s library preview (silent, 480px). Outside `clips/` on purpose — `attach_clips` prunes foreign names there |
| `visual.faiss` / `visual.sqlite` | Qwen-VL ANN + payload (`asset_id`, `shot_id`, path, `t_s`, `motion_energy`, `flash_energy`) |
| `text.faiss` / `text.sqlite` | BGE ANN + payload (OCR / transcript / scene) |
| `meta.json` | Title, duration, frames, shots, transcript, OCR, scene cards, channels |
| `timings.json` | Stage seconds |

`index.npz` is migrate-only leftover. Query does not load it.

## Package layout

```text
src/vvs/
  cli.py config.py models.py paths.py
  ingest/     YouTube + local ingest
  media/      frames, shots (TransNetV2), motion/flash, clips, transcribe, ocr, scenes, caption
  retrieve/   embed, index, search, reembed, VectorStore (FAISS / Qdrant)
  ui/         Gradio + Sightline theme + library grid
tests/        store / mute / shots / scenes / corpus / motion / library (no VLM download)
```

Commands: `uv run vvs index|query|app|warm|list`.

## Environment

| Variable | Meaning |
| --- | --- |
| `VVS_SOURCE` / `VVS_VIDEO_ID` | Optional CLI / UI prefill |
| `VVS_DATA_DIR` | Index root (default `data/`) |
| `VVS_HOST` / `VVS_PORT` | Gradio bind (default `0.0.0.0:7860`) |
| `VVS_VECTOR_BACKEND` | `faiss` (default) or `qdrant` |
| `VVS_FAISS_INDEX_TYPE` | `hnsw` (small N) or `ivf_pq` |
| `VVS_QDRANT_URL` / `VVS_QDRANT_API_KEY` | Required if backend is `qdrant` |
| `VVS_QDRANT_SHARDS` | Prod shard count for `hash(asset_id) % N` (default 8) |
| `VVS_FORCE_MUTE` | Skip Whisper even if the file has audio |
| `VVS_SCENE_CAPTION` | Caption visual scenes at analyze (X-Ray metadata) |
| `VVS_SCENE_CAPTION_MODEL` | `qwen_vl` (default, Qwen3-VL-2B-Instruct), `florence2`, or `blip` |
| `VVS_SHOT_DETECT` | TransNetV2 shot grouping; embed representatives only |
| `VVS_SHOT_THRESHOLD` | Cut threshold (default `0.5`) |
| `VVS_SHOT_MAX_GAP_S` | Extra still every N seconds inside a long shot (default `8`) |
| `VVS_VISUAL_ENCODER` | `qwen_vl` (default), `wemm`, or `clip` (old 512-d indexes) |
| `VVS_VISUAL_MODEL` | HF id or alias (`qwen-vl-emb-4B` → official 2B VL; `8b` → 8B; `wemm-2b` / `wemm-4b` / `wemm-9b`) |
| `VVS_VISUAL_TRUNCATE_DIM` | Matryoshka width for WeMM only; must be a supported dim |
| `VVS_DEVICE` | `mps`, `cuda`, or `cpu`. Default auto-detects (mps → cuda → cpu) |
| `VVS_VISUAL_DTYPE` | `float16`, `bfloat16`, `float32`. Default fp16 on GPU, fp32 on CPU |
| `VVS_VISUAL_BATCH` | Frames per forward pass. Default 8 on GPU, 1 on CPU |

Search weights live in `retrieve/search.py` (not env): per-video RRF (`TEXT_WEIGHT` 1.6, `OCR_WEIGHT` 0.6, `SCENE_WEIGHT` 1.0) and archive merge (`SCENE_ANN_WEIGHT` 1.15, `TEXT_ANN_WEIGHT` 1.4, `OCR_ANN_WEIGHT` 0.55, `LEXICAL_BLEND` 0.2, `MOTION_BLEND` 0.45). Motion/flash keep thresholds are in `media/motion.py` (`FLASH_KEEP` 0.22, `MOTION_KEEP` 0.18).

Docs for Confluence: [`docs/CONFLUENCE.md`](docs/CONFLUENCE.md) (diagrams in [`docs/images/`](docs/images/)). Self-contained paste-in with inlined figures: [`VOICE_VIDEO_SEARCH.md`](VOICE_VIDEO_SEARCH.md).
