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

Getting onto the GPU at all is what matters; the batch width is a rounding error
next to it. Measured on one Linux box (RTX PRO 2000, 8 GB, Qwen3-VL-Embedding-2B,
96 keyframes):

| device | batch | s/frame | peak |
| --- | --- | --- | --- |
| CPU, fp32 | 1 | 49.4 | — |
| CUDA, fp16 | 1 | 0.342 | 4.4 GB |
| CUDA, fp16 | 8 (default) | 0.270 | 5.2 GB |
| CUDA, fp16 | 16 | 0.254 | 6.2 GB |
| CUDA, fp16 | 32 | 2.362 | 8.0 GB |

The GPU is worth ~180×. Widening the batch from 1 to 16 is worth 1.35×, and from
the default 8 to 16 only 6% for another ~940 MB — not worth the lost headroom,
so the default stays 8. Batch 32 does not fit in 8 GB and collapses to 9× *slower*
than batch 8; the cliff is sudden rather than gradual, which is the reason to keep
a margin. Measure your own hardware with `scripts/bench_batch.py`, which also
verifies that a wider batch has not moved the vectors (padded batched attention
reorders fp16 reductions, so expect agreement around cosine 0.9996 — that is
arithmetic, not a quality change).

**Apple Silicon does not tell the same batch-width story.** Measured on an M3
Max (128 GB unified memory, Qwen3-VL-Embedding-2B, 128 keyframes):

| device | batch | s/frame | vs batch 1 |
| --- | --- | --- | --- |
| MPS, fp16 | 1 | 0.718 | 1.00× |
| MPS, fp16 | 4 | 0.489 | 1.47× |
| MPS, fp16 | 8 (default) | 0.516 | 1.39× |
| MPS, fp16 | 16 | 0.563 | 1.28× |
| MPS, fp16 | 32 | 0.579 | 1.24× |
| MPS, fp16 | 64 | 0.569 | 1.26× |

On CUDA, wider kept helping until the memory cliff. On MPS the win from
batching is real but caps out well before that: batch 4 measured fastest here,
8 is a close second, and 16+ is *slower* than 8, not just flatter — Apple's GPU
does not have the raw throughput headroom to keep profiting from a wider batch
the way a discrete card does, so past a small sweet spot the extra padding is
pure overhead. Batch 4 beating the default by 6% is inside this script's own
noise threshold for a 128-frame trial, so that alone is not a reason to change
`VVS_VISUAL_BATCH` from 8 — but pushing it up to 32 or 64 expecting a CUDA-style
continued win will make things slightly worse, not better. Peak memory reported
flat at ~4.26 GB across every batch size here, unlike CUDA's climbing figures —
`torch.mps` has no `max_memory_allocated`-style peak-since-reset counter the
way CUDA does, so that column is really "resident size after the trial," not a
true peak; do not read a memory-driven ceiling into it the way the CUDA numbers
support one.

Two processes sharing the GPU cost more here too, for a different reason than
the memory story below: this sweep's own model load took over 8 minutes with
a long-lived `vvs app` also holding the GPU, on a machine with far more
unified memory than either process needs. It is not an OOM story on a big
Mac — the single Metal command queue serializes across processes even when
there is plenty of memory for both, so a slow *load*, not just a slow
*forward pass*, is the symptom to watch for. Stop the UI before a benchmark
or a batch reindex.

**Read the CUDA table above as an upper bound: it loads the encoder alone.** A
real index run also needs the 2B captioner, and each model wants ~5 GB in
fp16, so on this 8 GB card they cannot both be resident. Holding both does
not fail cleanly — the allocator thrashes, flushing and re-reserving on
nearly every call. Measured
across one 16-video reindex, the first video embedded at 0.33 s/frame and every
video after its captioning stage ran at 7.8–9.9 s/frame, which is *slower than
CPU* (4–6 s/frame). Two processes are worse still: a long-lived `vvs app` holding
~5 GB leaves a separate reindex nothing to work with.

So `build_index` drops the model it has finished with at each stage boundary —
the captioner before embedding, the encoder before captioning — whenever the
device reports less than `EVICT_UNDER_GB` (16 GB). Reloading costs seconds per
video; sharing a too-small card costs minutes. `VVS_EVICT_MODELS=0` or `=1`
forces the choice. A 128 GB unified-memory Mac reports well above the threshold
and keeps both models loaded, exactly as before.

Re-running one identical video (47 keyframes, 29 scenes) before and after:

| stage | both resident | one at a time | gain |
| --- | --- | --- | --- |
| visual embed | 9.451 s/frame | 0.280 s/frame | 33.7× |
| caption | 14.96 s/scene | 2.43 s/scene | 6.2× |
| whole index | 1452.6 s | 559.4 s | 2.6× |

0.280 s/frame is the 0.270 the isolated benchmark predicted, so memory pressure
was the whole story.

That left TransNetV2 as the dominant cost at 69% of the run, and profiling it
showed the forward pass — not the video decode — was 87% of the stage (271.9 s
of 310.8 s, at 1732 ms per 100-frame window). It was on the CPU only because the
model was never moved to a device. On the GPU it runs at 25.3 ms/window, **68×
faster**, taking shot detection from 385 s to 47 s. `VVS_SHOT_DEVICE` overrides,
since the color-histogram path uses an int32 `scatter_add_` that a given Metal
build may not implement natively. CPU and GPU were verified to produce identical
cuts on a 29-shot video — zero frames changed side of the 0.5 threshold, every
boundary matched to 0.0000 s.

The two fixes together, on the same 263-second video:

| stage | before | after | device |
| --- | --- | --- | --- |
| TransNetV2 shots | 489.9 s | 46.9 s | GPU |
| caption (29 scenes) | 433.7 s | 66.1 s | GPU |
| OCR | 55.3 s | 49.2 s | CPU |
| ASR | 27.8 s | 31.1 s | CPU |
| visual embed (47 frames) | 444.2 s | 16.5 s | GPU |
| **total** | **1452.6 s** | **211.8 s** | |

**6.9× end to end**, with no change to any model or hyperparameter. Nothing
dominates now; decode and the PIL resize are ~90% of what is left in shot
detection, and ASR is the last stage stuck on the CPU by design.

This is per process. If you run the UI and a separate CLI reindex at once on a
small card they still contend, because each holds its own copy — index through
the UI, or stop it for the duration of a batch run.

On Linux, `pyproject.toml` installs a CUDA build of torch (`cu130`; Blackwell
`sm_120` needs 12.8 or newer and torch publishes no `cu128` wheels). macOS stays
on the plain index, which costs it nothing — `cpu` there names only the non-CUDA
variant and those wheels carry Metal support.

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

**`VectorStore` and OpenSearch.** The app talks only to a `VectorStore` (`create`, `add`, `search`, `close`) — see `retrieve/store.py`. `OpenSearchStore` (`retrieve/opensearch_store.py`) is the only implementation: two shared indexes, `vvs_visual` and `vvs_text`, across the whole archive, each row tagged with `video_id`. `podman compose up -d` runs a single local node; `docs/CONFLUENCE_ONPREM.md` §6–8 covers the sharded-cluster version of the same document shape.

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

**Saying there is no match.** k-NN always returns its k nearest rows, so ranking alone can never report an absent subject — a query for something simply not in the archive used to come back with confident-looking stills of whatever was least unlike it, indistinguishable on screen from a real answer. `drop_weak_hits` applies an absolute floor on `corpus_relevance`, and the UI shows *No matching content found* when nothing clears it.

The floor is measured, not picked. Over 21 videos, ten queries written from what is actually in the archive against ten for things plainly absent from it:

| | min | mean | max |
|---|---|---|---|
| genuine match | 2.940 | 3.339 | 4.133 |
| nothing to match | 2.014 | 2.290 | 2.545 |

`MIN_RELEVANCE = 2.7` sits in that gap with roughly 0.2 of headroom either side, and separates all twenty. Worth noting *why* the gap exists: on a nonsense query the visual channel still scores 1.25–1.42 and the caption channel 1.75–2.13, so neither channel alone would separate anything — it is their combination through `corpus_relevance` that does. The floor also trims weak tail hits inside an otherwise good result set, so a strong query can legitimately return fewer than `top_k` (`waterfall in a mossy forest` returns one). `VVS_MIN_RELEVANCE=0` restores the old always-answer behaviour. The number is calibrated on this archive's shape and wants re-measuring on a corpus of a different one.

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
podman compose up -d   # starts a local single-node OpenSearch on :9200
```

`docker compose up -d` runs the same file unchanged; the image is fully
qualified as `docker.io/...` precisely so both resolve it.

`podman compose` does not implement Compose itself, it hands off to a
provider, and which one you get decides what else you need installed. Its
default order tries `docker-compose` first — fine if Docker is around, but on
a machine where that binary belongs to Docker Desktop it vanishes with it.
Pin the provider instead, in `~/.config/containers/containers.conf`:

```toml
[engine]
compose_providers = ["/home/you/.local/bin/podman-compose"]
```

Install it without root via `uv tool install podman-compose`. Use an absolute
path: podman resolves providers through `PATH`, and `~/.local/bin` is not on
it for every shell. With this provider nothing Docker-related is needed —
`podman-compose` drives the `podman` CLI directly, so podman's
Docker-compatible API socket can stay stopped. Via the `docker-compose`
provider that socket *is* required (`systemctl --user enable --now
podman.socket`), or you get "failed to connect to the docker API at
unix:///run/user/1000/podman/podman.sock". Note that `podman-compose` also
groups the service into a pod (`pod_<project>`), which `docker-compose` does
not; harmless, but it is why `podman pod ls` is non-empty.

Containers do not come back by themselves after a reboot, and the compose
`restart:` key alone does not change that under rootless podman — nothing is
watching to act on it. A stopped node shows up as the app failing to start,
since `vvs app` refuses to run without OpenSearch. Three things have to line
up, and it is worth doing all three at once because any one missing looks
identical:

```bash
systemctl --user enable podman-restart.service   # the thing that acts on restart:
loginctl enable-linger "$USER"                   # lets user units run at boot, no login
podman compose down && podman compose up -d      # recreate so the policy sticks
```

The compose file asks for `restart: always`, not the more usual
`unless-stopped`, and that detail matters: `podman-restart.service` runs
`podman start --all --filter restart-policy=always`, which `unless-stopped`
does not match. Set it and the container is quietly skipped at boot, with the
policy looking correct in `podman inspect` the whole time. The tradeoff is
that `always` also restarts a container you deliberately stopped, which for a
dev database is what you want anyway. Check it without rebooting:

```bash
podman ps -a --filter restart-policy=always     # must list vvs-opensearch
podman stop vvs-opensearch
systemctl --user restart podman-restart.service # what fires at boot
podman ps                                       # back up
```

One rootless behaviour is worth knowing and is not fatal. A rootless
container cannot raise `memlock` above your own hard limit, and it caps it
silently rather than erroring, so OpenSearch logs `Unable to lock JVM Memory`
and `memory locking requested ... but memory is not locked` and then starts
regardless — `single-node` does not enforce that bootstrap check. Harmless
for a dev node holding a few thousand vectors; set
`bootstrap.memory_lock=false` in the compose file to quieten it.

Volumes are per-runtime, so podman will not see data written under Docker.
Either move it across once (`tar` the volume out through a throwaway
container and back in under the name `<project>_vvs-opensearch-data`) or
re-run `scripts/reindex_local.py`.

Needs Python 3.12+, `uv`, podman or Docker, ~8 GB RAM (Qwen-VL embedding 2B is heavier than CLIP; Instruct captioning loads a second 2B VL). First run needs network for Qwen-VL Embedding / BGE / RapidOCR, plus Whisper (if audio), TransNetV2, and Qwen3-VL-Instruct (or BLIP) if captions are on. After that, query is offline (OpenSearch itself needs no network once its image is pulled). CLIP indexes built before the Qwen-VL swap still search; new analyzes use Qwen-VL. `VVS_OPENSEARCH_URL` points at a different instance if you're not running the bundled compose file.

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
  services/   optional FastAPI wrappers for the two 2B models -- see below
  ui/         Gradio + custom theme + library grid
tests/        store / mute / shots / scenes / corpus / motion / library (no VLM download)
```

Commands: `uv run vvs index|query|app|warm|list`.

## Embedding services (optional)

Everything above runs the visual encoder (Qwen3-VL-Embedding-2B) and the
captioner (Qwen3-VL-2B-Instruct) in-process, loaded on first use exactly as
`retrieve/embed.py` and `media/caption.py` always have. `src/vvs/services/`
wraps each of those two models in its own FastAPI app instead, so they can be
deployed as independent, separately-scaled services rather than living inside
whichever process happens to call them.

**Why these two, and not the others.** These are the only two models a
process serving search needs to be *identical* to the process that built the
index -- an old query embedded with different weights than the frames it is
searching lands in the wrong part of the vector space. They are also the two
models responsible for every GPU-contention bug this repo has actually hit:
`vvs.device.should_evict()` and the `release_captioner()` /
`release_visual_encoder()` calls in `retrieve/index.py` exist only because
one process used to hold both 5 GB models on one card at once. A process
that only ever loads one of them never needs to evict it for the other, so
running them as separate services removes that whole class of bug rather
than working around it. TransNetV2, RapidOCR, and Whisper stay in-process:
they're index-time-only and single-consumer, so neither problem applies to
them yet.

**Run them directly** (needs the `services` extra: `fastapi`, `uvicorn`,
`python-multipart` -- not installed by default, since the CLI and the UI
never need a web framework):

```bash
uv sync --extra services
uv run --extra services uvicorn vvs.services.visual_embed_server:app --port 8001 &
uv run --extra services uvicorn vvs.services.caption_server:app --port 8002 &
```

**Or via Docker** (`docker/visual-embed.Dockerfile`, `docker/caption.Dockerfile`,
wired into `docker-compose.yml` behind the `services` profile so plain
`docker compose up -d` is unaffected):

```bash
docker compose --profile services up -d --build
```

These images are Linux-only by construction (Metal does not pass through
Docker) -- CUDA on a box with the NVIDIA Container Toolkit, or CPU. Run the
services directly with `uv run uvicorn` instead to use Apple Silicon's GPU.

**Point the rest of the app at them:**

```bash
export VVS_VISUAL_EMBED_URL=http://localhost:8001
export VVS_CAPTION_SERVICE_URL=http://localhost:8002
uv run vvs index --source <video> --scene-caption
uv run vvs query "a query" --video-id <id>
```

Leave both unset (the default) and nothing changes -- every call embeds and
captions in-process exactly as before. Each endpoint's own module docstring
(`retrieve/embed.py`'s `_remote_qwen_ok`, `media/caption.py`'s `caption_image`
and `expand_query_terms`) explains the one safety rule: a call only proxies
to the service when it would have resolved to the *default* Qwen weights
locally too. An index built with `VVS_VISUAL_ENCODER=wemm`, `clip`, or a
non-default `VVS_VISUAL_MODEL` (e.g. the 8B VL embedder) always embeds
locally regardless of these settings, since the service only ever loads the
one default model -- proxying a call that wanted different weights would
silently land vectors in the wrong space rather than searching it correctly.
A service failure never fails a search or an index run outright: query
expansion degrades to the literal query, and captioning falls back to the
same local Qwen → BLIP chain it already had.

## Environment

| Variable | Meaning |
| --- | --- |
| `VVS_SOURCE` / `VVS_VIDEO_ID` | Optional CLI / UI prefill |
| `VVS_DATA_DIR` | Index root (default `data/`) |
| `VVS_HOST` / `VVS_PORT` | Gradio bind (default `0.0.0.0:7860`) |
| `VVS_OPENSEARCH_URL` | OpenSearch endpoint (default `http://localhost:9200`; `podman compose up -d` starts one) |
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
| `VVS_VISUAL_BATCH` | Frames per forward pass. Default 8 on GPU, 1 on CPU. Measure with `scripts/bench_batch.py` |
| `VVS_EVICT_MODELS` | `1` / `0` to force unloading one 2B model before loading the other. Default: on below 16 GB of device memory |
| `VVS_SHOT_DEVICE` | Device for TransNetV2. Default: the shared device choice |
| `VVS_MIN_RELEVANCE` | Absolute `corpus_relevance` floor below which a search reports no match. `0` always answers. Default: `2.7` |
| `VVS_FAULTHANDLER` | `0` disables the C-level traceback on fatal signals |
| `VVS_VISUAL_EMBED_URL` | Base URL of the visual-embed service (e.g. `http://localhost:8001`). Default unset: embed in-process. See [Embedding services](#embedding-services-optional) |
| `VVS_CAPTION_SERVICE_URL` | Base URL of the caption service (e.g. `http://localhost:8002`). Default unset: caption/expand in-process. See [Embedding services](#embedding-services-optional) |
| `VVS_SERVICE_TIMEOUT_S` | HTTP timeout for calls to either service above (default `60`) |

Search weights live in `retrieve/search.py` (not env): per-video RRF (`TEXT_WEIGHT` 1.6, `OCR_WEIGHT` 0.6, `SCENE_WEIGHT` 1.0) and archive merge (`SCENE_ANN_WEIGHT` 1.15, `TEXT_ANN_WEIGHT` 1.4, `OCR_ANN_WEIGHT` 0.55, `LEXICAL_BLEND` 0.2, `MOTION_BLEND` 0.45). Motion/flash keep thresholds are in `media/motion.py` (`FLASH_KEEP` 0.22, `MOTION_KEEP` 0.18).

Docs for Confluence, a self-contained paste-in with figures inlined as base64 PNGs (no Mermaid macro or attachments needed): [`docs/CONFLUENCE_ONPREM.md`](docs/CONFLUENCE_ONPREM.md), the design for scaling this pipeline to a 50 PB on-prem archive. If your Confluence strips `data:` image URIs and the figures render blank, paste [`docs/CONFLUENCE_ONPREM_ATTACH.md`](docs/CONFLUENCE_ONPREM_ATTACH.md) instead, having first attached the matching `docs/images/diagrams/*.png` to the page. Both are generated from `docs/src/` by `scripts/inline_docs.py`; edit the source, not the generated files. This README is the reference for the single-node system that actually runs here.
