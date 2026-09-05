# Voice Video Search — CTO Demo Prototype

Repo: `video-voice-search` (`vvs` 0.1.0)

Audience: C-level live demo and the engineers who run or explain it.

Runtime: CPU-only local Gradio app. Source is chosen at runtime. No video is baked into the product.

**Confluence import.** Paste `docs/CONFLUENCE.md` (or the identical [`VOICE_VIDEO_SEARCH.md`](../VOICE_VIDEO_SEARCH.md) at the repo root). All five figures are embedded in the file as base64 PNGs, so the page needs no Mermaid macro, no attachments, and no outbound network access.

If your Confluence strips `data:` image URIs, fall back to attaching the five PNGs this page references from `docs/images/` — that route is also fully offline.

Editing: the prose lives in `docs/src/CONFLUENCE.md` and the diagram sources in `docs/diagrams/proto-*.mmd`. `docs/CONFLUENCE.md` is generated — do not edit it directly.

On-prem scale-out — OpenSearch hybrid k-NN + BM25, sharding for a 50 PB archive, vLLM, Kafka consumer groups — is a separate page: [`CONFLUENCE_ONPREM.md`](CONFLUENCE_ONPREM.md).

---

## 1. Purpose

Turn one or more long-form videos into a searchable archive on a laptop.

1. Pick any YouTube URL or local file (including mute video with no audio track). Repeat for more titles.
2. Analyze once on CPU: stills, optional speech, Qwen-VL + BGE embeddings into the shared OpenSearch collections.
3. Ask a question by typed text (voice is optional if a file has audio). **One query searches every indexed title.**
4. Play the matching moments as clips, labeled by title, with the original still, real start–end timestamps, and a confidence score.

This is a C-level demo, not a product. A rehearsed CTO interview (`fBEGOk3ZHw4`) is an optional paste-in example. Index a second title (music video, mute clip) and the same Search box covers the whole archive.

Production-shaped retrieval: visual-first ANN plus a text channel, ranked by comparable scores. This POC already runs on one shared OpenSearch index per collection (§6.1), a single local node rather than a sharded cluster. The on-prem design replaces the current per-channel k-NN calls with one OpenSearch `hybrid` query and scales the same index out to a cluster — see §6.3.

---

## 2. What it is / is not

| It is | It is not |
| --- | --- |
| A local prototype: ingest, index, query the **archive**, play clips | A full 50 PB deployment |
| Visual-first semantic search: Qwen-VL frames + BGE (OCR / captions / ASR if audio) | A keyword / filename scan |
| OpenSearch k-NN ANN + payload (`vvs_visual`, `vvs_text`, shared across the archive) | Embeddings loaded as a dense matrix in app RAM |
| Archive merge by **absolute ANN scores** (picture + caption + transcript + OCR) | Ranking by per-video RRF “confidence” or query-word keep/drop |
| A `VectorStore` interface with one real implementation (`OpenSearchStore`) | A sharded, replicated cluster with Kafka ingest (the on-prem target, §6.3) |
| Gradio UI with a custom visual language | Any third-party retrieval service or backend |
| CPU PyTorch (`pytorch-cpu` wheel index) | A GPU or cloud inference path |

After Qwen-VL Embedding, BGE, and (if needed) Whisper / Instruct captioner weights are cached, query is fully local. Network is only required on first run (weights plus optional YouTube download).

---

## 3. End-user flow

The app starts empty unless `data/` already has indexes — then Search is unlocked and one query covers every title.

1. Paste a YouTube URL or local path, or upload `.mp4` / `.webm` / `.mkv` / `.mov` / `.avi` / `.m4v`.
2. Optional analyze options (independent — check **both** for one pass): **Group shots with TransNetV2**, then **Describe scenes** (Qwen3-VL-2B-Instruct captions those scenes).
3. Click Analyze video. Pipeline: ingest, stills (shots or 2 fps), Whisper only if an audio track exists, OCR, optional scene captions, Qwen-VL + BGE, write to the shared OpenSearch collections.
4. Repeat Analyze for a second title if you want archive hits from more than one file, or use **01b · Bulk analyze** to do several at once.
5. If this video already has rows in the shared index and `meta.json` exists, the index is reused. Re-analyze rebuilds from scratch.
6. Type a question and Search. That query searches **every** indexed video. Clips are labeled by title. Mic / Transcribe + search is optional and only useful when a source has speech.
7. Top 3 hits become playable H.264 clips. Mute assets get a silent shot window. Gallery and match timeline show the rest of the rank list.

### 3.1 Library (section 00 in the UI)

The app opens on a grid of every indexed video, one card each: a preview clip,
title, id, duration, and chips for keyframes, spoken segments (or `mute`), OCR,
X-Ray scenes and shots, plus the encoder that built the index. That last chip is
how a legacy 512-d CLIP index is visibly distinguished from the Qwen ones. A card
reads `video file missing` when `video.mp4` is gone.

| Action | Effect |
| --- | --- |
| Hover a tile | Plays its preview muted; leaving rewinds. Tap toggles on touch |
| **Search this** | Narrows every later query to that one video; the card gets a red ring |
| **Search whole archive** | Clears the scope and fans out across all indexes again |
| **Filter** | Substring match on title or id |
| **Sort** | Recently indexed, title, longest first, or most keyframes |
| **Delete** | Two-step: card swaps to **Cancel** / **Confirm delete**, then removes `data/<id>/` |
| **Build previews** | Cuts and caches any missing previews for the whole archive |

Previews are cut once and cached as `data/<video_id>/preview.mp4`: 6 s, 480px
wide, 15 fps, silent, roughly 20–350 KB each. A 14-video archive took about 9
seconds to generate in full. For videos with speech the cut starts at the first
transcript segment past the first 8% of the runtime, so a talk opens on someone
talking rather than on a title card; mute assets fall back to a fixed offset, and
the poster is the keyframe nearest that same timestamp. Tiles are
`preload="none"`, so a full grid of cards downloads no video bytes until a tile
is actually hovered.

Two implementation notes that are easy to get wrong. Previews live beside the
index rather than inside `clips/`, because `attach_clips` garbage-collects
anything in that directory that does not match its own `clip_<tag>_*` naming — a
preview stored there disappears on the next search. And the hover behaviour is
registered from Gradio's `head` parameter, not its `js=` hook: that hook
evaluates its payload as statements, so passing a function expression there
silently defines and discards it.

Optional CLI prefill (does not auto-analyze):

```
uv run vvs app --source 'https://www.youtube.com/watch?v=fBEGOk3ZHw4'
uv run vvs app --video-id fBEGOk3ZHw4
```

On stage, pre-warm weights so the first query is not a cold start:

```
uv run vvs warm
uv run vvs app
```

Force the mute path (skip Whisper even if the file has audio):

```
VVS_FORCE_MUTE=1 uv run vvs index --source /path/to/video.mp4 --shot-detect --scene-caption
```

WSL note: open the printed localhost URL from Windows. The browser owns the microphone.

---

## 4. Architecture

![Architecture: ingest, OpenSearch visual and text indexes, archive query](images/architecture.png)

### 4.1 Analyze pipeline

![4.1 Analyze pipeline](images/diagrams/proto-4-1-analyze-pipeline.png)

Numbered pipeline:

1. Ingest — `yt-dlp` (YouTube, up to 720p progressive MP4) or copy a local file to `data/<id>/video.mp4`. Title / uploader / description come from yt-dlp; only **title** is persisted on `IndexMeta`.
2. Probe — PyAV checks for an audio stream. No track (or `VVS_FORCE_MUTE=1`) skips ASR.
3. Stills — default 2 fps + dHash + 4 s floor, **or** TransNetV2 shots (one still per cut, plus an 8 s floor in long takes). Same decode writes `motion_energy` / `flash_energy` (64×36 L1 and luma jump). A luma spike force-keeps the frame so lightning is not dropped. The VLM still embeds one JPEG — it does not see N frames.
4. Optional ASR — faster-whisper `small` int8 only when audio exists. Native Whisper segments, not fixed tiles.
5. OCR — RapidOCR on kept JPEGs. This is the on-screen text channel.
6. Optional X-Ray — stills clustered into scenes; **Qwen3-VL-2B-Instruct** captions the representative still (BLIP fallback). Embedded with BGE as `modality="scene"`.
7. Embed — **Qwen3-VL-Embedding-2B** on **one still** (2048-d, shared image/text space). BGE-small on OCR / transcript / captions (384-d). L2-normalized. CLIP ViT-B-32 (512-d) remains available for old indexes via `VVS_VISUAL_ENCODER=clip`. The official encode API also takes `{"video": frames, "fps", "max_frames"}` — **not used on this CPU box**. Later, on a GPU / larger server: one short window (~1 s, 6–8 frames) per kept still, same 2048-d collection. Do not smash the whole title into one vector.
8. Vector store — two OpenSearch indexes, `vvs_visual` and `vvs_text`, **shared across the whole archive** and filtered by `video_id`, not one file per video. Visual and text are separate spaces (different vector widths) and are never mixed in one index. Do not smash CLIP 512 and Qwen 2048 into one collection — a shared index has one fixed vector width; see `retrieve/opensearch_store.py`.
9. Search — Qwen-VL (or CLIP) query vs `vvs_visual` filtered to the video; BGE query vs `vvs_text` filtered the same way; lexical IDF on ANN candidates; RRF **inside** a video; **archive merge by absolute ANN scores**; hydrate path / timestamp / quote / `asset_id` from the same OpenSearch row.
10. Clip extract — H.264 + AAC from **that** asset’s `video.mp4`. With speech: play the full supporting transcript (cap 60 s). Mute: the shot span, not ±2 s around a keyframe.
11. UI — Gradio “Voice Video Search”: three clip cards labeled by title, original stills, captions, gallery, match-density timeline.

### 4.2 Package and component view

![4.2 Package and component view](images/diagrams/proto-4-2-package-and-component-view.png)

Object model (one row per embedding):

| Field | Meaning |
| --- | --- |
| `vector_id` | Legacy per-video id; OpenSearch documents are addressed by `_id`, not this field |
| `asset_id` | Video or still id (the shared index's `video_id` filter field) |
| `t_s` | Timestamp (0 for stills) |
| `modality` | `video_frame`, `still`, `ocr`, `transcript`, or `scene` |
| `path` | Frame or media URI |
| `text` | OCR / transcript / caption snippet when present |
| `shot_id` | TransNet shot when shots were built |
| `motion_energy` | Frame-to-frame L1 on a 64×36 gray (0–1). Default 0 on old indexes. |
| `flash_energy` | Absolute luma jump vs the previous sample (0–1). |
| `site_id`, `capture_ts` | Optional prod filters |

---

## 5. Models

Recorded on each index in `meta.json` under `models`. Search uses the encoder that **wrote** the index (`models.clip` / `models.visual_encoder` / `visual_dim`). `load_index` heals `visual_dim` from the OpenSearch store if meta is stale.

WeMM-Embedding-2B and Qwen3-VL-Embedding-2B are **both 2048-d**, so dimension
alone cannot identify an index. Every index therefore records
`models.visual_encoder`, and detection reads that field first, then the model
name, and only then dimension. Dimension-first inference silently mismatches the
two 2048-d families and must not be reintroduced.

### 5.1 Device selection

The visual encoder dominates index time, so the device it lands on is the single
biggest performance lever in the POC. `vvs/device.py` auto-detects: Metal
(`mps`) on Apple Silicon, then CUDA, then CPU, with `VVS_DEVICE` as an override.
Precision follows the device — fp16 on a GPU, fp32 on CPU, because CPU fp16 is
emulated and slower than fp32. Batch width follows too: 8 frames per forward pass
on a GPU where per-call overhead dominates, 1 on CPU where it does not and where
a narrow batch keeps peak memory small.

Three guards make that safe. `vvs/__init__.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1`
before anything imports torch, so the handful of VL ops without a Metal kernel run
on CPU instead of aborting. Placement matters: PyTorch reads that variable when
the MPS backend initializes, so setting it inside the model loader — after torch
is already imported — has no effect at all, and the symptom is a hard process
crash reported as `leaked semaphore objects to clean up at shutdown` rather than a
Python traceback. Second, a GPU that fails to load the model pins the session to
CPU instead of retrying a broken device on every call. Third, a one-shot probe
after load encodes a throwaway string and checks the result is finite and
non-zero; if fp16 has overflowed to NaN the model is rebuilt in fp32. Without
that probe a half-precision overflow would write NaN vectors into an index and
silently break retrieval rather than fail loudly.

The captioner in `media/caption.py` shares that single decision rather than
making its own, so the weights and the hand-placed input tensors cannot end up on
different devices. The GPU matters more for captioning than for embedding:
captioning is autoregressive, so a scene costs up to `QWEN_MAX_TOKENS` sequential
forward passes where an embedding costs one.

### 5.2 Duplicate native runtimes

Two independent library duplications affected this stack. They are worth
keeping apart, because one was fatal and the other only looks alarming — and
the fatal one is now retired along with the dependency that caused it.

**OpenMP — fatal, retired with FAISS.** This POC used to carry `faiss-cpu` for
on-disk ANN; it collided with torch on macOS the way described below. FAISS
search now runs server-side in OpenSearch (§6.1), not as an in-process
library, so there is no second copy of anything for torch to collide with.
Kept here because the *mechanism* that fixed it, a single dedicated compute
thread (`vvs/compute.py`), is still load-bearing for an unrelated crash: model
loading in transformers hits the same non-main-thread segfault shape,
independent of FAISS. `faiss-cpu` and `torch` each bundled their own copy of
the LLVM OpenMP runtime; on macOS arm64 these are `faiss/.dylibs/libomp.dylib`
and `torch/lib/libomp.dylib`. This produces two distinct failures.

First, whichever runtime initializes second causes OpenMP to abort the process.
`vvs/__init__.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` before torch or faiss can be
imported, which is the upstream-documented escape hatch.

Second, and this is the one that mattered, allowing the duplicate exposes exactly
the crash that flag warns about: a FAISS OpenMP parallel region entered from a
**non-main thread** segfaults. Every Gradio request is served on an anyio worker
thread, so search died in the UI while the identical query from the CLI
succeeded — which is why the failure looked capricious and resisted diagnosis
from symptoms. What settled it was `faulthandler`, which converted the silent
signal death into a C traceback naming `swigfaiss.search` beneath
`ui/app.py:from_text`. Before that, the only evidence was the resource tracker's
`leaked semaphore objects` warning, which is emitted merely because the process
was killed and names neither step nor library.

Pinning FAISS to one thread turned out not to fix this either — the fork/join
barrier machinery runs regardless of team size, and the real condition was
thread identity, not thread count: the first thread to call in was fine, any
different thread calling in later reliably crashed. The actual fix was the
same dedicated-thread approach used today (`vvs/compute.py`), just aimed at a
FAISS-specific symptom that no longer exists. Now that OpenSearch does this
work server-side, the whole class of "which OS thread touches the native ANN
library" question is moot for search — it still matters for torch model
loading, which is why `vvs/compute.py` remains.

**Surviving the abort.** Because an OpenMP abort can land mid-write, `meta.json`
is written atomically: temp file, `fsync`, then `os.replace`, with the previous
copy rotated to `meta.json.bak` and readers falling back to it when the primary
will not parse. This is not incidental hardening. `meta.json` holds the
transcript, the scene captions and the OCR text, none of which are recoverable
from the OpenSearch collections, so losing it means re-running the captioner across the
whole asset. And a truncated file is worse than a missing one, because
`index_exists()` only tests for presence: a half-written file left the video
listed but unparseable, so it vanished from the library with no explanation.
Indexes damaged before this change have no backup to recover from and must be
re-analyzed.

**FFmpeg / AVFoundation — noisy, not fatal.** PyAV and OpenCV each bundle
`libavdevice`, which on macOS registers the Objective-C classes
`AVFFrameReceiver` and `AVFAudioReceiver`. Two copies in one process register the
same class names twice, and the ObjC runtime reports it at `stage=ocr`, the point
where RapidOCR first imports `cv2`. It warns of "spurious casting failures and
mysterious crashes", but captioning, OCR and indexing all complete with the
warning present, so it is treated as noise pending evidence otherwise. Note that
`opencv-python-headless` omits `libavdevice` **only in the manylinux wheels**;
both the full and headless macOS wheels still ship it, so the substitution does
not silence this on a Mac. The POC uses the headless build regardless — nothing
opens an OpenCV window, RapidOCR needs cv2 only for array manipulation, and it
does remove the duplicate on Linux, which is what the on-prem containers run.
Removing it on macOS would require running OCR out-of-process or replacing cv2.

Measured on the prototype laptop (CPU, 4 frames after warmup), Qwen-VL-Embedding-2B
costs **28.3 s/frame** against WeMM-Embedding-2B's **72.8 s/frame** — 2.6× slower
for the same 2048-d output, with comparable text-query cost (9.8 s vs 10.2 s).
Re-embedding the 1,332-frame sample archive with WeMM would take roughly 27
hours, so the **quality comparison is deferred to a GPU host**. The tooling is in
place for it: `retrieve/reembed.py` clones an index and re-encodes only the
visual channel (same keyframes, transcript and OCR), and
`scripts/compare_encoders.py` scores the pair on recall@k / MRR / nDCG where
timestamp judgments exist, plus rank overlap, Spearman agreement and latency.

| Role | Model | Device / quant | Dim | Notes |
| --- | --- | --- | --- | --- |
| Visual (default) | `Qwen/Qwen3-VL-Embedding-2B` | auto (mps/cuda fp16, cpu fp32) | 2048 | Shared image/text space. Official VL embed sizes are **2B / 8B only** — there is no 4B VL embedder (`Qwen3-Embedding-4B` is text-only). `VVS_VISUAL_MODEL=qwen-vl-emb-4B` aliases to 2B and logs that. |
| Visual (alternative) | `WeMM-Embedding-2B` (also 4B / 9B) | auto (mps/cuda fp16, cpu fp32) | 2048 / 2560 / 4096 | `VVS_VISUAL_ENCODER=wemm`. Uses `encode_query` / `encode_document` rather than Qwen's prompt argument. Matryoshka-trained, so `VVS_VISUAL_TRUNCATE_DIM` can shorten vectors to a supported width. |
| Visual (legacy) | open_clip ViT-B-32 `laion2b_s34b_b79k` | auto | 512 | `VVS_VISUAL_ENCODER=clip`. Existing 512-d indexes still search. Switching encoder requires **Re-analyze**. |
| Scene caption | `Qwen/Qwen3-VL-2B-Instruct` | auto (shared with the embedder) | — | Opt-in X-Ray. Different checkpoint from the embedder: Instruct writes English; Embedding only writes vectors. BLIP-base is the fallback, taken once per run rather than once per scene. |
| ASR | faster-whisper `small` int8 | CPU | — | Skipped when there is no audio track |
| Text | BAAI/bge-small-en-v1.5 | CPU | 384 | `passage:` at index, `query:` at search (OCR + optional transcript + scene) |
| OCR | rapidocr-onnxruntime | CPU | — | Off with `vvs index --no-ocr` |
| Shots | TransNetV2 | CPU | — | Opt-in cut detection on 48×27 frames (PyAV, no ffmpeg CLI) |

`uv run vvs warm` loads the visual encoder, BGE, and Whisper so the live demo pays no first-query tax.

---

## 6. Vector store: POC, and the on-prem target

### 6.1 POC (this demo)

OpenSearch, one shared node (`docker compose up -d`), `VVS_OPENSEARCH_URL` to point elsewhere.

- `OpenSearchStore` (`retrieve/opensearch_store.py`) writes to two indexes shared across the **whole archive**: `vvs_visual` and `vvs_text`, each row tagged with `video_id`. Not one file per video.
- Both collections use HNSW k-NN (inner product on L2-normalized vectors), one fixed vector width per index — a real limitation versus the old per-video files: mixing CLIP-512 and Qwen-2048 videos now needs two separate indexes, not one archive.
- Payload (path, timestamp, `text`, `shot_id`, motion/flash) lives on the same OpenSearch document as its vector, not in a separate store.
- Query returns ids + scores; the app hydrates path, timestamp, quote, and `shot_id` straight from that document's `_source`.
- Archive search still loops per video (embed once, k-NN each video's rows, merge) rather than one global query — see the `opensearch_store.py` module docstring for why: a true archive-wide `hybrid` query would fuse visual and text scores server-side, discarding the per-channel ranks §8's RRF fusion depends on.

**OpenSearch k-NN is the only ANN here, not BM25.** Lexical ranking still runs in Python on ANN candidates only, the same as before. OpenSearch's own BM25 engine is available on the text fields but deliberately unused for the reason above.

### 6.2 What changed from the original FAISS + SQLite design

This section documented `FaissStore` (one `.faiss` + `.sqlite` pair per video, mmap'd) and a `QdrantStore` stub proving the `VectorStore` interface wasn't FAISS-shaped. Both are gone: `OpenSearchStore` is the only implementation now, and it already is the shape §6.2 used to gesture at wanting. The interface itself (`create`, `add`, `search`, `close`) did not change — only what implements it.

### 6.3 On-prem target: OpenSearch hybrid

Full design, capacity math, and sharding plan: [`CONFLUENCE_ONPREM.md`](CONFLUENCE_ONPREM.md).

One index, **one document per ~20 s segment**, `_id = segment_id`:

- `visual_vector` — `knn_vector` 2048-d, `engine: faiss`, `space_type: innerproduct`, `mode: on_disk`
- `caption`, `ocr`, `transcript` — `text` fields scored by **BM25**
- `asset_id`, `source_class`, `rights`, `wrap`, `pack_ids` — `keyword` filters
- `still_uri`, `t0`, `t1`, `motion_energy`, `flash_energy` — payload

Retrieval is one `hybrid` query: a `knn` clause on `visual_vector` plus a `multi_match` on `caption^2` / `ocr` / `transcript`, fused by a search pipeline (RRF first, weighted `min_max` once a judgment list exists). That replaces the per-video k-NN fan-out and the Python RRF/merge in §8 with one server-side call — the next step past what §6.1 runs today.

**Sharding for a 50 PB archive, in one table.** The sizing input is segment count, not petabytes — pixels never enter the index:

| Input | Value |
| --- | --- |
| Archive | 50 PB in object storage |
| Assumed average mezzanine bitrate | ~100 Mbps → ~1.11 M hours |
| Retrieval unit | ~20 s segment → ~200 M vectors |
| Vector memory, fp32 | 9,152 B/vector → **3.66 TB** with 1 replica. Not viable |
| Vector memory, `on_disk` 32x | 422 B/vector → **169 GB** with 1 replica |
| Shard target | ~6–8 M vectors → ~3 GB resident graph, ~62 GB on disk |
| Primaries | **32** across `vvs_captured` (24), `vvs_produced` (6), `vvs_wraps` (2) |
| Data nodes | **6** × 256 GB RAM / 2 TB NVMe, 32 GB heap, ~28 GB resident graph each |
| p95 latency target | under 300 ms end to end, excluding pixel delivery |

Three facts worth carrying into any scale conversation:

1. **`mode: on_disk` is mandatory.** fp32 HNSW at 2048-d is 3.66 TB of RAM; binary quantization with fp32 rescoring from NVMe is 169 GB. A 22x difference.
2. **The Lucene k-NN engine caps at 1024 dimensions**, so 2048-d requires FAISS — and `compression_level: 4x` is therefore unavailable, because 4x routes to Lucene.
3. **The 20 s segment is a ~40x cost lever.** Indexing every 2 fps still instead would be ~8 B vectors and an 80-to-120 node cluster.

50 PB of pixels is a modest vector workload. The bytes live in object storage; the index holds one vector and a few hundred bytes of text per 20 s of content.

---

## 7. Example video: 149 frames / 12 segments

Example (optional): a Bloomberg interview with a tech-company CTO on AI product strategy and vision

`https://www.youtube.com/watch?v=fBEGOk3ZHw4` maps to `data/fBEGOk3ZHw4/`

Duration: 263.0 s. Not hardcoded.

**149 keyframes** — `extract_keyframes` in `media/frames.py` (2 fps path; TransNetV2 would keep fewer stills).

| Knob | Default | Effect on this 263 s file |
| --- | --- | --- |
| `sample_fps` | 2.0 | About 526 candidate pictures |
| dHash | 16×16 (256-bit) | Near-duplicate if similarity is at least 0.92 |
| `uniform_interval_s` | 4.0 | Force-keep even if visually similar |

About 66 frames from the 4 s floor plus about 83 scene-change extras. Those 149 visual vectors live in `vvs_visual`, tagged with this video's id (512-d if this tape was indexed with CLIP; 2048-d after a Qwen re-analyze — see §6.1 on why a shared index can't hold both at once).

**12 transcript segments** — Whisper native turns (this file has audio). Typical 10–30 s Q&A chunks, 0.00–262.8 s. They are BGE-indexed in `vvs_text` together with OCR (and scene) rows, this video's id filtering them out of every other video's rows. A mute file would have zero transcript segments and still be searchable via Qwen-VL + OCR / captions.

| # | Start–end | Topic (abridged) |
| --- | --- | --- |
| 0 | 0.00–22.30 | Hybrid AI / distributed future |
| 1 | 22.30–51.04 | Ecosystem of models, devices, agents |
| 2 | 51.04–61.82 | Edge / on-device |
| 3 | 61.82–84.22 | Cloud + edge coexist |
| 4 | 84.22–113.02 | Ambient / personalized experience |
| 5 | 113.02–132.93 | Motorola lineup + Kira |
| 6 | 132.93–159.84 | Same UI on phone, tablet, PC |
| 7 | 159.84–180.85 | Personalized knowledge base |
| 8 | 180.85–196.24 | Robotics + cybersecurity |
| 9 | 196.24–223.95 | Agent-native devices |
| 10 | 223.95–252.58 | Voice, gesture, sensing |
| 11 | 252.58–262.80 | Always-on AI |

Only this interview’s ASR contains the word **robotics** (`I'm leading up research groups in robotics`, ~181–196 s). A music-video still can still have a decent visual cosine for a random query — archive merge must use the transcript channel, or Roja wins that search.

---

## 8. Semantic search

![Semantic search: visual and text ANN, per-video RRF, archive merge by ANN scores](images/search.png)

Code: `retrieve/search.py`. `vvs_visual` k-NN is always on. `vvs_text` k-NN holds OCR plus optional transcript and scene captions.

### 8.1 Query flow

![8.1 Query flow](images/diagrams/proto-8-1-query-flow.png)

### 8.2 Inside one video

1. Embed the query with the visual encoder that wrote the index (Qwen-VL text or CLIP text) and BGE (`query: …`).
2. ANN search each store for a shortlist (not a full scan of every vector in RAM). Over-fetch ~48, then lexical-IDF rerank text hits.
3. Map text hits onto nearby frames (OCR by `frame_index`; transcript by spoken window; scene by covering interval).
4. Fuse with reciprocal rank fusion (`k = 60`):
   - visual: `1 / (60 + rank)`
   - transcript: `1.6 / (60 + rank)` only if ASR ran
   - OCR: `0.6 / (60 + rank)` if score >= 0.28
   - scene caption: `1.0 / (60 + rank)` if captions were built and score >= 0.22
5. Temporal / anchor boosts apply only when `channels.asr` is true.
6. Diversify: keep top-k frames at least 6 s apart (search `top_k=6`; UI plays 3).

Confidence (0–100%) is CTO-facing, not a calibrated probability. It is the fused RRF score divided by the ceiling of the channels that actually ran. **Do not sort the archive by this number** — extra ASR extras inflate an interview’s ceiling, so the interview would win mountain queries for the wrong reason.

### 8.3 Across the archive

The UI and `vvs query` (no `--video-id`) fan out to every `data/<id>/`, then merge by **absolute ANN similarities**:

```
corpus_relevance = max( visual + 1.15×caption + 0.55×OCR,
                        1.4×transcript )
                 + 0.2×lexical overlap
                 + 0.45×motion/flash   # only if the query asks for that
```

Picture path and spoken path compete (`max`), they are not summed. BGE-small cosine is ~0.65 on unrelated talk; adding transcript onto every interview made “trumpet” return the interview clip over a saxophone still. A real spoken hit still wins max() against a random picture. Motion/flash are decode-time scalars, not a video transformer. “Moving tornado” boosts high `motion_energy`; “lightning” boosts `flash_energy`; a plain “tornado” query does not. **Re-analyze** to populate those fields. On a larger server, replace (or add) the still vector with an N-frame Qwen-VL window so “moving” lives in the embedding — same `vvs_visual` dim, opt-in Re-analyze. We do **not** keep-or-drop a title because a caption contains the query words. One strong title may take every slot.

Worked examples on a mixed archive (interview clip + Roja + Careless Whisper / Jennifer Connelly):

| Query | First hit | Why |
| --- | --- | --- |
| `robotics` | Interview clip ~3:01 | Spoken ANN (lexical-boosted) beats a music-video still |
| `clips with a trumpet` | Jennifer Connelly ~1:30 | Saxophone still + caption; interview ASR is unrelated |
| `moving tornado` | High-motion funnel still | Same picture ANN; motion sidecar breaks the tie vs a parked tornado |
| `lightning` | Force-kept luma-spike still | 2 fps / TransNet would otherwise skip a sub-second flash |
| `snow-covered mountains` | Roja | Visual + caption ANN; the interview never said that |
| `what is hybrid AI` | Interview clip 0–55 s | Transcript on the talking-head tape |

Pin one file with `vvs query --video-id ID` (or `--source`). Mixed CLIP 512-d and Qwen 2048-d indexes in one archive still search, but their visual cosines are not perfectly comparable — re-analyze old CLIP titles on Qwen if you care about picture-vs-picture ties.

**On-prem equivalent.** The `max(...)` blend above is Python over two OpenSearch k-NN queries (visual vs. text), the same shared archive-wide indexes this POC already writes to (§6.1). The on-prem step still ahead of this repo is folding those two k-NN calls plus BM25 into one `hybrid` query fused by a search pipeline. The design rule survives that port either way: relevance comes from comparable cross-channel scores, never from query-word keep/drop. See [`CONFLUENCE_ONPREM.md`](CONFLUENCE_ONPREM.md) §6.4.

---

## 9. Clip playback

With speech: play the supporting Whisper segment from its first word through the segment end (cap 60 s from the start). Files are named `clip_asr_<start>_<end>.mp4`.

Mute / no ASR: play the TransNet shot window when shots exist (`clip_vis_*.mp4`), otherwise a short silent window around the retrieved frame. Caption uses scene text or OCR when available.

UI card shows:

- Video title (`video_title`) — which archive asset this clip came from
- Actual clip start–end (probed after write)
- Original keyframe still
- Supporting quote (transcript window, scene caption, or OCR)
- Confidence (per-video RRF) and per-channel ANN scores (visual / transcript / OCR / scene)

---

## 10. Storage

Per video, under `data/<video_id>/` (`VVS_DATA_DIR` overrides the root):

| Path | What |
| --- | --- |
| `video.mp4` | Ingested source |
| `frames/` | Representative JPEGs (2 fps set, or TransNet stills) |
| `clips/` | On-demand H.264 hits |
| `preview.mp4` | Cached 6 s library preview (silent, 480px). Kept outside `clips/` so `attach_clips` does not prune it |
| `meta.json` | Title, duration, frames, shots, transcript, OCR, scene cards, channels, models |
| `timings.json` | Stage seconds |
| `ingest.json` | Source / title / duration / uploader / description |

Not under `data/`: vectors and their payload (`asset_id`, `shot_id`, path, `t_s`, motion/flash) live in OpenSearch, in the shared `vvs_visual` (Qwen-VL / WeMM / CLIP) and `vvs_text` (BGE: OCR / transcript / scene) indexes, filtered by `video_id`. Deleting a video's `data/<video_id>/` directory alone leaves those rows behind — use the app's delete flow, which purges both.

---

## 11. Package layout

```
src/vvs/
  cli.py config.py models.py paths.py
  ingest/     YouTube + local file ingest
  media/      frames, shots (TransNetV2), motion/flash, clips, transcribe, ocr, scenes, caption
  retrieve/   embed, index, search, VectorStore (OpenSearch)
  ui/         Gradio app, custom theme, timeline
scripts/      build_index.py, query.py, validate.py
tests/        store / mute / shots / scenes / corpus (no VLM download)
```

| Package | Responsibility |
| --- | --- |
| `ingest/` | Resolve YouTube id or local slug; download or copy to `video.mp4` |
| `media/` | Keyframes, TransNet shots, optional Whisper, RapidOCR, scene cards, clip cut |
| `retrieve/` | Qwen-VL or CLIP + BGE, OpenSearch write/k-NN, per-video RRF, archive ANN merge, motion/flash boost |
| `ui/` | Gradio + custom CSS (style only); Search calls `search_corpus` |

Public commands: `uv run vvs index`, `query`, `app`, `warm`, `list`.

---

## 12. How to run

Requirements: Python 3.12+, `uv`, Docker, about 8 GB RAM (Qwen-VL Embedding 2B is heavier than CLIP; Instruct captioning loads a second 2B VL). Network on first model/video download.

```
cd video-voice-search
uv sync
docker compose up -d   # local single-node OpenSearch on :9200
uv run vvs app
```

| Action | Command |
| --- | --- |
| Live UI | `uv run vvs app` |
| Bind | `VVS_HOST` / `VVS_PORT` (default `0.0.0.0:7860`) |
| Index talking-head | `uv run vvs index --source 'https://www.youtube.com/watch?v=…'` |
| Mute video index | `VVS_FORCE_MUTE=1 uv run vvs index --source /path/to/mute.mp4 --shot-detect --scene-caption` |
| Archive query | `uv run vvs query 'robotics'` |
| Pin one title | `uv run vvs query --video-id fBEGOk3ZHw4 'what is hybrid AI'` |
| Compare still counts | `uv run vvs list --compare cmp-2fps cmp-shots` |
| Preload weights | `uv run vvs warm` |
| List indexes | `uv run vvs list` |

Env: `VVS_SOURCE`, `VVS_VIDEO_ID` (optional prefill), `VVS_DATA_DIR`, `VVS_OPENSEARCH_URL` (default `http://localhost:9200`), `VVS_OPENSEARCH_TIMEOUT_S`, `VVS_FORCE_MUTE`, `VVS_SHOT_DETECT`, `VVS_SCENE_CAPTION`, `VVS_SCENE_CAPTION_MODEL` (`qwen_vl` / `blip`), `VVS_VISUAL_ENCODER` (`qwen_vl` or `clip`), `VVS_VISUAL_MODEL`, `VVS_DEVICE`, `VVS_VISUAL_DTYPE`, `VVS_VISUAL_BATCH`.

`docker compose up -d` starts a local single-node OpenSearch on `:9200`; see `docker-compose.yml`.

---

## 13. Constraints (say these out loud on stage)

| Constraint | Reality |
| --- | --- |
| CPU-only | PyTorch CPU wheels; no CUDA path in this package |
| First-run downloads | Qwen-VL Embedding + BGE + Whisper (if audio) + RapidOCR + optional Instruct / TransNetV2 + the video |
| Analyze is minutes | OCR or VLM captioning is usually the slowest stage |
| Visual is pictures | OCR / captions / transcript carry the words; Qwen-VL matches what the frame looks like |
| Rehearse the questions | Five tuned queries hit the expected window in the top 3 on the example tape; `robotics` must beat a music-video still in an archive |
| This box indexes one (or a few) videos | The code path is add / k-NN search / hydrate against a single-node OpenSearch — the same shape as a sharded cluster |
| Mixed encoders | CLIP 512-d and Qwen 2048-d indexes can coexist; visual scores are not perfectly comparable until you re-analyze |
| Mic / WSL | Use the Windows browser against localhost |

Rehearsed queries (CTO example only):

1. what is hybrid AI — 0–55 s
2. show me the Motorola products and Kira — 110–140 s
3. personalized knowledge base on phone tablet and PC — 130–180 s
4. robotics and cybersecurity research — 175–200 s
5. agent-native devices voice and gesture interfaces — 196–260 s

Archive check (needs a second indexed title, e.g. Roja): type `robotics` with no pin — the interview clip at ~3:01 should be first.

If someone asks “does this scale to the real archive?”, the answer is §6.3 and [`CONFLUENCE_ONPREM.md`](CONFLUENCE_ONPREM.md): 50 PB of pixels is ~200 M segment vectors, which is a 6-node OpenSearch cluster once `mode: on_disk` quantization is used, because the pixels stay in object storage.

---

## 14. Example video is optional

`Settings.source` and `Settings.video_id` default to None. There is no baked-in YouTube URL.

The same app indexes any other URL or upload, including mute content. `--source`, `--video-id`, and `VVS_*` only prefill the text box. Omit `--video-id` on `vvs query` to search the archive.

---

## 15. Custom UI (visual language only)

`ui/theme.py` and `ui/theme.css` define a custom dark look: accent red `#E2231A`, ink neutrals, sphere/arcs mark, dark canvas `#05070C`. Window title: Voice Video Search.

Style only. Retrieval, models, and storage are this repo. No external UI service, index, or API is called.

---

## 16. Tunables

| Area | Name | Default |
| --- | --- | --- |
| Frames | `sample_fps` | 2.0 |
| Frames | `similarity_threshold` | 0.92 |
| Frames | `uniform_interval_s` | 4.0 |
| Shots | `VVS_SHOT_THRESHOLD` | 0.5 |
| Shots | `VVS_SHOT_MAX_GAP_S` | 8 |
| Visual | `VVS_VISUAL_ENCODER` | qwen_vl |
| Visual | `VVS_VISUAL_MODEL` | Qwen3-VL-Embedding-2B |
| Captions | `VVS_SCENE_CAPTION_MODEL` | qwen_vl |
| Store | `VVS_OPENSEARCH_URL` | http://localhost:9200 |
| Store | `VVS_OPENSEARCH_TIMEOUT_S` | 10 |
| Per-video RRF | `RRF_K` | 60 |
| Per-video RRF | `TEXT_WEIGHT` | 1.6 |
| Per-video RRF | `OCR_WEIGHT` / `OCR_MIN_SCORE` | 0.6 / 0.28 |
| Per-video RRF | `SCENE_WEIGHT` / `SCENE_MIN_SCORE` | 1.0 / 0.22 |
| Per-video RRF | `TEMPORAL_BOOST` (ASR only) | 0.85 |
| Archive merge | `SCENE_ANN_WEIGHT` | 1.15 |
| Archive merge | `TEXT_ANN_WEIGHT` | 1.4 |
| Archive merge | `OCR_ANN_WEIGHT` | 0.55 |
| Archive merge | `LEXICAL_BLEND` | 0.2 |
| Archive merge | `MOTION_BLEND` | 0.45 |
| Capture | `FLASH_KEEP` / `MOTION_KEEP` | 0.22 / 0.18 |
| Search | diversity `min_gap_s` | 6.0 |
| Clips | `MAX_CLIP_S` | 60 |
| UI | `PLAYABLE_K` | 3 |
