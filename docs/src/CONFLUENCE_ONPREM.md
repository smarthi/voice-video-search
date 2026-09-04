# On-prem ingest and retrieval — 50 PB design

Status: design, not implemented in this repo.  
Audience: platform / ML / Sightline engineers standing up Sphere-scale ingest and search.  
Companion: the CPU prototype is [`CONFLUENCE.md`](CONFLUENCE.md). This page is the on-prem shape of the same object model.

**Confluence import.** Paste `docs/CONFLUENCE_ONPREM.md`. Every diagram is embedded in the file as a base64 PNG, so the page needs no Mermaid macro, no attachments, and no outbound network access.

If your Confluence strips `data:` image URIs, fall back to attaching the ten PNGs from `docs/images/diagrams/onprem-*.png` to the page — that route is also fully offline.

Editing: the prose lives in `docs/src/CONFLUENCE_ONPREM.md` and the diagram sources in `docs/diagrams/*.mmd`. `docs/CONFLUENCE_ONPREM.md` is generated — do not edit it directly.

```
python3 scripts/render_diagrams.py render onprem-4-2   # re-render one diagram
python3 scripts/inline_docs.py                         # regenerate the paste-in docs
```

---

## 1. Purpose and scope

Ingest a 50 PB Sphere archive, cut each asset into ~20 s retrieval segments, embed and caption those segments concurrently, and write **one OpenSearch index** that serves **hybrid retrieval**: Qwen-VL k-NN on the segment still **and** BM25 on captions / OCR / transcript, in the same engine, in one query.

Two hard requirements drive every choice on this page:

1. **Hybrid in one engine.** Vector k-NN and BM25 over caption/OCR must be one `hybrid` query against one document — not two stores with an application-side merge.
2. **Low retrieval latency at 50 PB.** Target p95 under 300 ms end to end, including query embedding, excluding pixel delivery.

Scope covers ingest through index write, plus the sharding and latency plan that the write path must satisfy.

---

## 2. Verdict (short)

| Decision | Choice | Why |
| --- | --- | --- |
| Search engine | **OpenSearch 2.19+** | One engine holds `knn_vector` on video embeddings **and** BM25 on caption/OCR. `hybrid` query + search pipeline fuses them in-cluster. Qdrant has no BM25. |
| Vector field | `knn_vector` 2048-d, **engine `faiss`**, `space_type: innerproduct`, `mode: on_disk` | The Lucene engine caps at 1024 dimensions, so FAISS is mandatory at 2048. `on_disk` cuts RAM ~22x with rescoring. |
| Document model | **One doc per segment**, `_id = segment_id` | Hybrid scores one object. Embed and caption **partial-update** the same id. |
| Retrieval unit | **~20 s segment**, one vector | The single biggest cost lever on this page. 2 fps stills would be ~40x the vectors. |
| Shards | **32 primaries + 1 replica** across three class indexes at the 200 M-vector plan | ~6–7 M vectors per shard. Derivation in §8.1. |
| Qwen serving | **vLLM**, one deployment per checkpoint, behind a thin FastAPI façade | Embedding-2B and Instruct-2B are different weights. Workers write vectors; OpenSearch never sees a JPEG. |
| TransNetV2 | **FastAPI process pool, not vLLM** | Shot-boundary CNN on 48×27 frames. Not a generative model. |
| Text encoder (BGE) | **Omit on day one** | BM25 is the lexical channel. Add a 384-d clause later only if synonym recall fails. |
| Queue | **Apache Kafka**, or **Redpanda** if this is the first broker | True consumer groups: one topic, every segment delivered to **both** embed and caption. |
| Object store | Existing Sphere / S3-compatible bucket | 50 PB of pixels stays put. The pipeline writes **proxy stills**, not 4K clones. |
| Asset ledger | Postgres | `M of N` index-completeness counters per asset. Kafka is not the ledger, and Kafka offsets cannot answer this question (§13). |

**Two headline findings.**

1. **Search is small.** 50 PB of pixels is roughly **200 M segment vectors**, which is a **6-to-8 data-node** OpenSearch cluster — not a hundred-node build. The archive is enormous in bytes and modest in vectors, because the retrieval unit is a 20 s segment and pixels never enter the index.
2. **Decode is the choke point, not the GPUs.** Backfilling 1.11 M hours costs ~278,000 CPU process-hours of video decode against ~7,000 GPU-hours of captioning and ~560 GPU-hours of embedding. Decode outweighs all inference by 50–500x, and captioning outweighs embedding by ~12x. Size the shot tier first, and consider GPU NVDEC for it (§15).

---

## 3. What we are not doing

| Avoid | Do this instead |
| --- | --- |
| Two competing consumers on one Kafka group | Two **consumer groups** on the same topic |
| A vector DB plus a separate BM25 store | One OpenSearch index, `hybrid` query in-cluster |
| Materialize a temp 4K clip per cut | Keep 4K in object storage. Message = ids + times + still URIs |
| Captions only on a file share | Caption worker updates `caption` on the same document |
| Index every 2 fps still | One vector per ~20 s segment. See §7.4 for the 40x blow-up |
| fp32 HNSW at 2048-d | `mode: on_disk`. fp32 is 3.7 TB of RAM at plan scale; on_disk is 169 GB |
| `compression_level: 4x` | 4x routes to the Lucene engine, which caps at 1024-d. Use 32x or 16x on FAISS |
| Lucene k-NN engine | 2048-d exceeds the Lucene 1024-d limit |
| Smash Qwen 2048-d and a text embedding into one field | One visual field. BM25 for text. Optional second k-NN field later, never mixed dims |
| An OpenSearch ingest pipeline calling Qwen-VL | Workers call vLLM, then `_update` the float vector |
| One giant index for everything | Split by `source_class` so wrap suppression removes shards from the hot query (§8.2) |
| Retry by republishing a temp path | Stable `segment_id = hash(asset_id, t0, t1)`. Idempotent `_update` |
| TransNet on the 4K/16K mezzanine | TransNet decode at 48×27 |
| Threads inside one asset for TransNet | Process parallelism **across assets** |
| Many threads sharing one GPU | One vLLM process per device. Workers are HTTP clients |
| Deep pagination on hybrid results | Bounded top-k, then `search_after` if producers need paging |

---

## 4. Architecture diagrams

### 4.1 System context

![4.1 System context](images/diagrams/onprem-4-1-system-context.png)

### 4.2 Component and data-flow view

Every service and store in the system. Which hardware tier each one runs on is in §15.3.

![4.2 Component and data-flow view](images/diagrams/onprem-4-2-full-component-and-deployment-view.png)

### 4.3 Fan-out: one topic, two consumer groups

![4.3 Fan-out: one topic, two consumer groups](images/diagrams/onprem-4-3-fan-out-one-topic-two-consumer-groups.png)

Members **inside** a group compete. The two groups do **not** compete. One group id shared by both process types would give embed XOR caption — the failure mode this diagram exists to prevent.

### 4.4 Shot worker internals

![4.4 Shot worker internals](images/diagrams/onprem-4-4-shot-worker-internals.png)

### 4.5 Embed and caption worker internals

![4.5 Embed and caption worker internals](images/diagrams/onprem-4-5-embed-and-caption-worker-internals.png)

### 4.6 One segment, both consumers, same document

![4.6 One segment, both consumers, same document](images/diagrams/onprem-4-6-one-segment-both-consumers-same-document.png)

Either worker may win the create race. `doc_as_upsert` means whoever arrives first creates the document and the other adds its fields. The **asset** stays unsearchable until visual reaches N/N, so a caption-only document is never served.

### 4.7 Query path and latency budget

![4.7 Query path and latency budget](images/diagrams/onprem-4-7-query-path-and-latency-budget.png)

### 4.8 Index and shard layout

![4.8 Index and shard layout](images/diagrams/onprem-4-8-index-and-shard-layout.png)

### 4.9 Asset lifecycle

![4.9 Asset lifecycle](images/diagrams/onprem-4-9-asset-lifecycle.png)

### 4.10 Message failure handling

![4.10 Message failure handling](images/diagrams/onprem-4-10-message-failure-handling.png)

---

## 5. Pipeline, step by step

1. **Ingest API** accepts an asset from Sphere: media URI, `source_class` (captured / produced / wrap), pack ids, rights, duration. Writes an asset row in Postgres as `pending`. Publishes one message to `vvs.assets`. Classification happens **here**, before the queue, so both consumers see the same label space.
2. **Shot workers** (group `vvs.shots`, one in-flight decode per process) call `transnet-api`. Segment = **hard cut boundary + ~20 s time floor**. Extract one proxy still per segment plus the motion/flash sidecar from a mid-res decode — never from the 4K mezzanine. PUT stills to the durable bucket. Publish one `vvs.segments` message per segment. Record `segment_count = N`.
3. **Embed group** and **caption group** each receive every segment.
4. **Embed worker** calls `vllm-embed` on the still, runs RapidOCR on the same JPEG, `_update`s `visual_vector` + `ocr` + payload, increments `visual_done`, commits.
5. **Caption worker** calls `vllm-instruct`, `_update`s `caption` on the same `_id`, writes the caption sidecar, increments `caption_done`, commits. Caption failure never blocks visual searchability.
6. When `visual_done == segment_count`, Postgres marks the asset **searchable**. OpenSearch near-real-time refresh is ~1–5 s; use `refresh=wait_for` on the final upsert of an asset only if the UI must show it immediately.

**Segment length is a design decision, not a TransNet output.** TransNet returns hard cuts. A Sphere fly-through with no cut is one shot until you floor it. The taxonomy retrieval unit is ~20 s; the laptop prototype uses an 8 s floor. §7.4 prices the difference.

---

## 6. OpenSearch document and hybrid query

### 6.1 One document per segment

Index family: `vvs_captured_v1`, `vvs_produced_v1`, `vvs_wraps_v1` (see §8.2). `_id = segment_id`, a stable hash. Embed and caption never create a second document for the same cut.

| Field | Type | Written by | Role |
| --- | --- | --- | --- |
| `visual_vector` | `knn_vector` 2048-d, faiss, hnsw, innerproduct, `mode: on_disk` | Embed | Qwen3-VL-Embedding-2B, L2-normalized |
| `caption` | `text`, english analyzer | Caption | BM25 and X-Ray |
| `ocr` | `text` | Embed | BM25. On-screen text; the Sphere mute path |
| `transcript` | `text` | Later ASR group | BM25. Produced and wraps with a bed only |
| `asset_id`, `source_class`, `rights`, `pack_ids` | `keyword` | Payload on first upsert | Filters, not vectors |
| `wrap` | `boolean` | Payload | Suppressed for find-the-shot |
| `t0`, `t1` | `float` | Shot payload | Clip window |
| `shot_id` | `integer` | Shot payload | Provenance |
| `still_uri` | `keyword`, `index: false` | Shot payload | Citation. Pixels stay in object storage |
| `motion_energy`, `flash_energy` | `float` | Shot / embed | Query-time boost when the query asks for motion or flash |
| `visual_model`, `caption_model` | `keyword` | Each worker | Reindex planning |
| `capture_ts`, `site_id` | `date`, `keyword` | Payload | Operational filters |

L2-normalize before upsert so `innerproduct` equals cosine, exactly as the prototype does.

### 6.2 Mapping

```json
PUT /vvs_captured_v1
{
  "settings": {
    "index": {
      "knn": true,
      "number_of_shards": 24,
      "number_of_replicas": 1,
      "refresh_interval": "5s"
    }
  },
  "mappings": {
    "properties": {
      "visual_vector": {
        "type": "knn_vector",
        "dimension": 2048,
        "space_type": "innerproduct",
        "data_type": "float",
        "mode": "on_disk",
        "compression_level": "32x",
        "method": {
          "name": "hnsw",
          "engine": "faiss",
          "parameters": { "ef_construction": 256, "m": 16 }
        }
      },
      "caption":    { "type": "text", "analyzer": "english" },
      "ocr":        { "type": "text", "analyzer": "english" },
      "transcript": { "type": "text", "analyzer": "english" },
      "asset_id":     { "type": "keyword" },
      "source_class": { "type": "keyword" },
      "rights":       { "type": "keyword" },
      "pack_ids":     { "type": "keyword" },
      "wrap":         { "type": "boolean" },
      "site_id":      { "type": "keyword" },
      "capture_ts":   { "type": "date" },
      "t0":      { "type": "float" },
      "t1":      { "type": "float" },
      "shot_id": { "type": "integer" },
      "motion_energy": { "type": "float" },
      "flash_energy":  { "type": "float" },
      "still_uri":     { "type": "keyword", "index": false },
      "visual_model":  { "type": "keyword" },
      "caption_model": { "type": "keyword" }
    }
  }
}
```

Two constraints decide the engine:

- **Lucene k-NN caps at 1024 dimensions.** 2048-d requires `engine: faiss`.
- **`compression_level: 4x` routes to Lucene**, so it is unavailable at 2048-d. The usable levels here are `32x`, `16x`, `8x`, `2x`, `1x`.

### 6.3 Concurrent writes from two groups

Both groups update the same `_id`. A plain `_index` from either side would erase the other's fields. Use a partial update with conflict retry:

```json
POST /vvs_captured_v1/_update/a1b2c3d4e5f6?retry_on_conflict=5
{
  "doc": {
    "visual_vector": [ "... 2048 floats ..." ],
    "ocr": "EXIT 14A",
    "asset_id": "sph-2026-08-30-4412",
    "source_class": "captured",
    "wrap": false,
    "rights": "internal",
    "still_uri": "s3://sphere-proxies/sph-2026-08-30-4412/a1b2c3d4e5f6.jpg",
    "t0": 340.0,
    "t1": 358.2,
    "shot_id": 17,
    "motion_energy": 0.31,
    "flash_energy": 0.04,
    "visual_model": "Qwen3-VL-Embedding-2B"
  },
  "doc_as_upsert": true
}
```

The caption worker sends only `caption` and `caption_model` on the same `_id`.

Batch with `_bulk` (100–500 docs) rather than per-segment `_update`, and back off on 429. A 2048-d vector is roughly 24 KB as JSON text, so size bulk bodies at 5–15 MB rather than by document count.

### 6.4 The hybrid query

Qwen-VL is **not** registered as an OpenSearch ML model. The Query API embeds the text once through `vllm-embed` (the same 2048-d image/text space), then sends that vector in a `knn` clause alongside BM25.

Search pipeline — start with **RRF**, because visual cosine and BM25 live on different scales and RRF needs no weight hunt to be safe:

```json
PUT /_search/pipeline/vvs-hybrid-rrf
{
  "description": "RRF over visual kNN and BM25 caption/OCR/transcript",
  "phase_results_processors": [
    { "score-ranker-processor": { "combination": { "technique": "rrf", "rank_constant": 60 } } }
  ]
}
```

```json
GET /vvs_search/_search?search_pipeline=vvs-hybrid-rrf
{
  "size": 50,
  "_source": { "excludes": ["visual_vector"] },
  "query": {
    "hybrid": {
      "queries": [
        {
          "knn": {
            "visual_vector": {
              "vector": [ "... query 2048-d ..." ],
              "k": 100,
              "method_parameters": { "ef_search": 100 },
              "rescore": { "oversample_factor": 2.0 }
            }
          }
        },
        {
          "multi_match": {
            "query": "clips with a trumpet",
            "fields": ["caption^2", "ocr", "transcript"]
          }
        }
      ],
      "filter": {
        "bool": {
          "must":     [ { "terms": { "rights": ["internal", "cleared"] } } ],
          "must_not": [ { "term": { "wrap": true } } ]
        }
      }
    }
  }
}
```

Use the top-level `filter` on the `hybrid` query so suppressed wraps and un-cleared rights never enter **either** clause. Reserve `post_filter` for facet aggregations that must reflect the unfiltered set.

Once a judgment list exists, compare RRF against `normalization-processor` with `min_max` and a weighted `arithmetic_mean`. Score normalization is what reproduces the prototype's explicit "picture path versus spoken path compete" behavior; RRF is the safer default before measurement.

Motion and flash stay a conditional `should` / `function_score` on `motion_energy` and `flash_energy`, applied only when the query lexicon asks (moving, lightning, and so on). Never baked into the vector.

### 6.5 The lexical gap this design accepts

The prototype scored caption text with BGE cosine, a dense semantic space. This design scores it with BM25, which matches only tokens that literally appear. Text that is semantically close but lexically different scores **zero** — the document gets no candidacy from the BM25 clause at all, not merely a weak score.

Three cases, in increasing difficulty:

| Query term vs caption text | BM25 | What fixes it |
| --- | --- | --- |
| "saxophone" vs "saxophone" | matches | nothing needed |
| "sax" vs "saxophone", "twister" vs "tornado" | zero | synonym and stemming token filters |
| "trumpet" vs "saxophone" | zero | a dense text vector, nothing cheaper |

The third row is the actual gap. Trumpet and saxophone are not synonyms, so no synonym list connects them — yet the prototype's BGE cosine ranked that saxophone still first for "clips with a trumpet", and that result was judged correct.

**Consequence.** On such a query the segment is ranked by visual k-NN alone. That is usually enough, since the picture channel is the primary channel by design. What is lost is the *redundancy*: when the frame is visually ambiguous, the prototype had a second channel voting for the right segment and this design does not.

**Measure it instead of guessing.** Log, per query, whether the BM25 clause contributed any document to the final top-k. If removing the BM25 clause leaves the top-k unchanged for a material share of real Sphere queries, the text channel is not earning its keep and the gap is real.

**Fixes, in cost order:**

1. **Synonym and stemming token filters** on `caption`, `ocr`, `transcript`. Near-zero cost, no reindex of vectors, covers row 2.
2. **Add `text_vector`** (BGE-small, 384-d — under the Lucene 1024-d cap, so the `lucene` engine is fine here) as a **third** hybrid clause. Still one engine, one document, one query. Costs a text encoder in the embed worker plus roughly 77–99 GB of resident graph at plan scale (32x or 16x compression), on top of the 169 GB visual budget from §7.3. Low-dimension fields tolerate aggressive binary quantization less well than 2048-d, so budget for 16x.

Ship the two-clause hybrid first. Measure, then decide.

---

## 7. Scale: 50 PB to vector count

### 7.1 Bytes are not the sizing input

OpenSearch never stores a pixel. Sizing depends on **segment count**, which depends on archive **duration** and the **segment floor**. Duration follows from the average mezzanine bitrate.

Hours in the archive: `hours = 50 PB / (bitrate_Mbps × 0.45 GB per hour)`.  
Segments at a 20 s floor: `segments = hours × 180`.

| Average bitrate | Interpretation | Hours in 50 PB | Segments at 20 s |
| --- | --- | --- | --- |
| 1000 Mbps | 16K masters only | ~111 K | ~20 M |
| 500 Mbps | mezzanine-heavy | ~222 K | ~40 M |
| 200 Mbps | mixed 16K + 4K | ~556 K | ~100 M |
| 100 Mbps | 4K mezzanine average | ~1.11 M | ~200 M |
| 50 Mbps | delivery-proxy heavy | ~2.22 M | ~400 M |

**Plan for 200 M vectors. Validate headroom at 400 M.** Confirm the archive's real average bitrate before ordering hardware — it moves the answer by 20x, and it is the one input this design cannot derive.

### 7.2 Memory per vector

OpenSearch documents these HNSW estimates, where `m` is graph connectivity and the 1.1 factor covers filesystem overhead:

- fp32: `1.1 × (4 × d + 8 × m)` bytes/vector
- byte / scalar-quantized: `1.1 × (d + 8 × m)` bytes/vector
- binary quantized: `1.1 × (d / 8 + 8 × m)` bytes/vector

At `d = 2048`, `m = 16`:

| Mode | Bytes per dim | Formula | Bytes/vector |
| --- | --- | --- | --- |
| fp32, no compression | 4 | 1.1 × (8192 + 128) | **9,152** |
| `on_disk` 2x | 2 | 1.1 × (4096 + 128) | 4,646 |
| `on_disk` 8x, 4 bits | 0.5 | 1.1 × (1024 + 128) | 1,267 |
| `on_disk` 16x, 2 bits | 0.25 | 1.1 × (512 + 128) | 704 |
| `on_disk` 32x, 1 bit (default) | 0.125 | 1.1 × (256 + 128) | **422** |

### 7.3 Cluster memory at plan scale

`total = bytes_per_vector × num_vectors × (1 + replicas)`, with `replicas = 1`:

| Mode | 100 M vectors | 200 M (plan) | 400 M (ceiling) |
| --- | --- | --- | --- |
| fp32 | 1.83 TB | **3.66 TB** | 7.32 TB |
| 16x | 141 GB | **282 GB** | 563 GB |
| 32x | 84 GB | **169 GB** | 338 GB |

fp32 at 2048-d is not economically viable at this scale. **`mode: on_disk` is mandatory, not an optimization.** It moves the plan from 3.66 TB of RAM to 169 GB — a 22x reduction — because the compressed graph stays resident while full-precision vectors are read from NVMe only to rescore the top candidates.

Disk at 200 M vectors, both copies:

| Component | Per vector | 200 M × 2 copies |
| --- | --- | --- |
| fp32 vectors for rescoring | 8,192 B | 3.28 TB |
| Binary quantized vectors | 256 B | 102 GB |
| `_source`, BM25 postings, doc values | ~800 B | 320 GB |
| **Total NVMe** | | **~3.7 TB** |

Provision **~6 TB of NVMe across the data tier**, which leaves room for merges (force-merge needs transient space up to the size of the segments being merged).

### 7.4 The 20 s segment is worth ~40x

If the retrieval unit were every 2 fps still instead of one still per 20 s segment:

| Unit | Vectors at 1.11 M hours | 32x memory, 1 replica | Data nodes |
| --- | --- | --- | --- |
| One per 20 s segment | 200 M | 169 GB | ~6 |
| One per TransNet cut, ~4 s average | ~1.0 B | 845 GB | ~12–16 |
| Every 2 fps still | ~8.0 B | 6.8 TB | ~80–120 |

This is the strongest argument for the taxonomy's ~20 s retrieval unit, and the reason the shot stage applies a time floor rather than emitting raw TransNet cuts. Segment granularity is a **cluster-cost decision**, so changing the floor is a capacity change, not a tuning knob.

### 7.5 Dimension reduction, if recall allows

The Qwen3 embedding family supports Matryoshka-style output truncation. If `Qwen3-VL-Embedding-2B` truncates cleanly to 1024 or 512 dimensions, every number above scales roughly linearly: 1024-d at 32x is `1.1 × (128 + 128) = 282` bytes/vector, and 1024-d also becomes eligible for the Lucene engine.

Treat this as an **experiment gated on a judgment list**, not a plan assumption. Truncation changes the space, so it requires a full reindex. Measure recall@10 on Sphere queries before committing.

---

## 8. Sharding and index layout

### 8.1 Shard count derivation

Two constraints bound shard size:

**Resident graph per shard.** A shard's compressed graph should be a few GB so a node can hold many shards and relocation is quick. At 32x, `8 M vectors × 422 B ≈ 3.4 GB`.

**Total bytes per shard.** OpenSearch guidance is 10–50 GB for text shards. With `on_disk`, most shard bytes are fp32 vectors that are *intentionally* not resident, so the practical ceiling is relocation time rather than page cache. At 8 M vectors: `8 M × 9,248 B ≈ 74 GB` on disk. A 74 GB relocation at 1 GB/s is about 75 s — acceptable. Do not exceed ~100 GB per shard, or node replacement becomes an hours-long event.

Target **~6–8 M vectors per shard**:

| Index | Share | Vectors | Primaries | Vectors/shard | Graph/shard | Disk/shard |
| --- | --- | --- | --- | --- | --- | --- |
| `vvs_captured_v1` | ~80% | 160 M | **24** | 6.7 M | 2.8 GB | 62 GB |
| `vvs_produced_v1` | ~18% | 36 M | **6** | 6.0 M | 2.5 GB | 56 GB |
| `vvs_wraps_v1` | ~2% | 4 M | **2** | 2.0 M | 0.8 GB | 19 GB |
| **Total** | | 200 M | **32** primaries (64 shards with 1 replica) | | 169 GB | 3.7 TB |

Set primary counts to a **multiple of the data-node count** so shards distribute evenly. With 6 data nodes, 64 total shards is about 11 per node and roughly 28 GB of resident graph per node.

**Do not overshard.** Every shard runs its own HNSW traversal and returns its own top-k, so both the coordinator's merge work and the tail latency grow with shard count. Shard count is sized for capacity and even placement, not for parallelism beyond the node count.

### 8.2 Why split by `source_class`

Splitting by class is the only split that genuinely reduces **shards searched** on the dominant query, because the taxonomy suppresses dome wraps for "find the shot":

- `vvs_search` alias → captured + produced (30 primaries). The hot path.
- `vvs_search_all` alias → all three, for host, zone, and sync queries that legitimately want wraps.

A `wrap: false` filter inside one big index still touches every shard. Two aliases over two index sets do not. The taxonomy says class **biases rank and does not filter**, so keep class scoring in the ranker; this split is purely about not paying for wrap shards on every query.

### 8.3 Routing

Use **default routing**. Custom `_routing = asset_id` would confine a query to one shard, which only helps queries already scoped to a single title — and archive search is global by definition. Custom routing at this scale also risks hot shards, since Sphere asset durations vary by orders of magnitude.

The exception is a "more from this asset" panel. Add `"routing": asset_id` to **that** request only if profiling shows the fan-out matters. Correctness does not depend on it when documents are not routed at write time, so do not set write-time routing.

### 8.4 Time-based rollover, for operations not latency

Roll `vvs_captured_v1` monthly by `capture_ts` (`vvs_captured_v1-2026.08`) behind the same alias. This buys:

- Force-merge a sealed month to one Lucene segment, making the graph one contiguous file
- Reindex a model bump one month at a time instead of as one 200 M-document job
- ISM transitions, and searchable snapshots for rights-expired content

It does **not** reduce shards searched, because a find-the-shot query has no date predicate. Do not present rollover as a latency win.

### 8.5 Model version in the index name

A new embedding checkpoint is a new vector space and cannot share a field with the old one. Version the index (`_v1`, `_v2`), reindex month by month, then swap the alias atomically. `visual_model` on each document tells you what still needs re-embedding. Budget the GPU cost: re-embedding 200 M segments is the same work as the original ingest.

### 8.6 Node roles

| Role | Count | Sizing | Notes |
| --- | --- | --- | --- |
| Cluster manager | 3 | 8 vCPU, 32 GB | Dedicated. Never data. |
| Coordinator | 2–3 | 16 vCPU, 64 GB | Terminate hybrid queries and run the RRF pipeline off the data nodes. |
| Data (vector) | 6 | 32 vCPU, 256 GB RAM, 2 TB NVMe | 32 GB heap. The rest is OS page cache and the k-NN native cache. |

Heap is capped at 32 GB so pointers stay compressed; the remaining ~224 GB serves the k-NN native cache and page cache. The k-NN circuit breaker defaults to **50% of available off-heap**, so usable graph capacity per node is about `(256 − 32) × 0.5 = 112 GB`. Plan usage is ~28 GB per node, so there is roughly 4x headroom — and that headroom is what makes `compression_level: 16x` affordable if 32x recall proves short.

---

## 9. Latency: budget and levers

### 9.1 Budget, p95, 200 M vectors

| Stage | p50 | p95 | Notes |
| --- | --- | --- | --- |
| Query text → 2048-d via vLLM | 15 ms | 40 ms | Single short text on a warm 2B model. Cache repeat queries. |
| Coordinator fan-out to 30 shards | 2 ms | 5 ms | |
| Per-shard quantized graph traversal, `ef_search` 100 | 8 ms | 25 ms | Compressed graph resident |
| Rescore top candidates from NVMe, oversample 2.0 | 6 ms | 20 ms | The main `on_disk` cost |
| BM25 clause | 3 ms | 10 ms | Text is tiny next to vectors |
| RRF combine and fetch top 50 `_source` | 5 ms | 15 ms | `visual_vector` excluded |
| API shaping and signed URLs | 5 ms | 15 ms | |
| **End to end, excluding pixel delivery** | **~45 ms** | **~130 ms** | |

That leaves substantial margin against the 300 ms target. Pixel delivery — stills and clips from object storage — is the real user-perceived long pole. Put proxies behind a CDN and return signed URLs rather than proxying bytes through the API.

### 9.2 Levers, in the order to reach for them

1. **Force-merge sealed indexes.** One Lucene segment per shard means one contiguous graph file. A shard with 40 small segments runs 40 graph traversals. Only force-merge read-only indexes.
2. **Warm the cache** with the k-NN warmup API after a restart, a merge, or a shard relocation. Otherwise the first queries pay cold NVMe reads.
3. **Tune `ef_search`** from 100. Higher improves recall and costs latency roughly linearly.
4. **Tune `oversample_factor`** down from the 2.0 default if rescore I/O dominates; raise it if 32x recall is short.
5. **Move to `16x`** before adding nodes. 282 GB versus 169 GB at plan scale, well inside the 112 GB-per-node budget.
6. **Add replicas for QPS**, not for capacity. Each replica multiplies resident graph memory.
7. **Pre-filter, do not post-filter.** FAISS efficient filtering evaluates the filter during traversal.
8. **Exclude `visual_vector` from `_source`** on every query. Fetching 50 × 24 KB of vector JSON is pure waste.
9. **Cap `k` and `size`.** Hybrid pulls candidates per clause per shard, and deep paging multiplies all of it.
10. **Dedicated coordinator nodes** so RRF and fetch never compete with graph traversal.

### 9.3 Watch these

`GET _plugins/_knn/stats` — `graph_memory_usage`, `cache_capacity_reached`, `hit_count`, `miss_count`, `eviction_count`. Non-zero evictions mean the graphs no longer fit and latency has already degraded: raise compression, add nodes, or raise the circuit breaker. Native memory pressure does **not** appear as JVM heap pressure, so heap dashboards will look healthy while search is thrashing.

---

## 10. Model serving: vLLM and FastAPI

### 10.1 On vLLM

| Service | Checkpoint | API | Called by |
| --- | --- | --- | --- |
| `vllm-embed` | `Qwen/Qwen3-VL-Embedding-2B` | Embeddings — image at ingest, text at query, one space | Embed workers **and** the Query API |
| `vllm-instruct` | `Qwen/Qwen3-VL-2B-Instruct` | Chat completions — one still to one caption | Caption workers |

Different weights. Separate deployments so ingest batches never queue behind interactive query embedding. If GPUs are scarce, give the Query API its own small `vllm-embed` replica with a strict latency SLO and let ingest use the rest.

Thin FastAPI façade for auth, request ids, and a stable contract:

```
POST /v1/embed/visual
{ "still_uri": "s3://.../seg.jpg", "asset_id": "...", "segment_id": "..." }
→ { "vector": [ 2048 floats ], "dim": 2048, "model": "Qwen3-VL-Embedding-2B" }

POST /v1/embed/query
{ "text": "clips with a trumpet" }
→ { "vector": [ 2048 floats ], "dim": 2048, "model": "Qwen3-VL-Embedding-2B" }

POST /v1/caption
{ "still_uri": "...", "source_class": "captured" }
→ { "text": "a saxophone player in blue light", "model": "Qwen3-VL-2B-Instruct" }
```

The façade L2-normalizes so no worker can forget.

**Later, on GPU:** the official Qwen-VL embed API accepts `{"video": frames, "fps", "max_frames"}`. Use one short window of 6–8 frames per segment into the **same** `visual_vector` field, so "moving tornado" lives in the embedding rather than in the motion sidecar. Do not embed a whole asset as one vector. Lightning still needs the force-kept flash still from the shot stage. This is a reindex, so it is a `_v2` index.

### 10.2 Not on vLLM

| Workload | Host | Why |
| --- | --- | --- |
| TransNetV2 | `transnet-api`, FastAPI + process pool | Shot-boundary CNN on 48×27 frames, sequential decode |
| RapidOCR | In-process in the embed worker, ONNX | Same JPEG as the visual embed; a network hop buys nothing |
| Motion and flash | Shot worker, already decoding | L1 and luma jump on 64×36 gray |
| BGE-small | Omit until §6.5 says otherwise | BM25 is the lexical channel |

### 10.3 `transnet-api` contract

```
POST /v1/shots
{ "media_uri": "s3://.../asset.mp4", "threshold": 0.5, "max_gap_s": 20 }
→ { "shots": [ { "t0": 0.0, "t1": 12.4, "shot_id": 0 } ],
    "motion_track": [ "..." ] }
```

- Input is a readable URI. The service decodes a **48×27** proxy. It does not transcode 4K or 16K.
- Output is time spans, never pixels. The shot worker extracts citation JPEGs.
- Scale by process count, one decode each. Intra-asset threads do not help a sequential decode.
- `max_gap_s` is the taxonomy floor. This is where 20 s is enforced.

---

## 11. Queue and consumer groups

### 11.1 Kafka, or Redpanda

| Need | Kafka / Redpanda | RabbitMQ | Redis Streams |
| --- | --- | --- | --- |
| Every segment handled by **both** embed and caption | Two consumer groups on one topic | Fan-out exchange to two queues | Two groups, weaker as system of record |
| Scale workers of one type | Add members up to partition count | Competing consumers | Same |
| Replay after a model bump | Reset offsets, or a new group id | Must republish | Awkward |
| Partition locality per asset | Key = `asset_id` | No log or offsets | Weak |
| On-prem operations | Kafka if you already run it; Redpanda if not | Simpler at low volume | Cache, not a bus |
| Dead-letter | Application-owned DLQ topics | Native dead-letter exchange | Do it yourself |

**Kafka if the DC already runs it. Redpanda if this would be the first broker.** Do not use Redis as the ingest bus. Do not add Pulsar unless the site already operates it, despite its subscription model being the closest native match.

### 11.2 Topics

| Topic | Key | Producer | Consumers | Purpose |
| --- | --- | --- | --- | --- |
| `vvs.assets` | `asset_id` | Ingest API | group `vvs.shots` | One message per asset. Competing consumers: exactly one decode per asset. |
| `vvs.segments` | `asset_id` | Shot workers | groups `vvs.embed` **and** `vvs.caption` | Fan-out. Every segment reaches both. |
| `vvs.embed.dlq` | `asset_id` | Embed workers | Ops / replay job | Poison, or five failed attempts |
| `vvs.caption.dlq` | `asset_id` | Caption workers | Ops / replay job | Same, non-blocking |
| `vvs.shots.dlq` | `asset_id` | Shot workers | Ops | Unreadable media, TransNet crash |

Partitions on `vvs.segments`: `max(32, embed_gpu_replicas)`. Partition count caps embed concurrency and is expensive to raise later, so start high. Retention on `vvs.segments`: 7 days, long enough to replay a bad deploy without a full re-cut.

**Never share a group id between embed and caption.**

### 11.3 Groups

```
vvs.segments
  ├── group vvs.embed    → N workers, competing with each other
  └── group vvs.caption  → M workers, competing with each other
```

Within a group, one worker owns a partition at a time. Concurrency on a single hot title therefore comes from partition count, not worker count. Prefer more partitions and more assets in flight over splitting one asset across keys — keeping `asset_id` as the key is what keeps the completeness counter updates contention-free, since one asset's segments all land on one partition and are processed by one worker at a time.

### 11.4 Delivery, retry, DLQ

Kafka is at-least-once with explicit commits, not SQS-style visibility timeouts.

| Rule | Setting or behavior |
| --- | --- |
| Commit only after success | `enable.auto.commit=false`. Commit after the OpenSearch bulk response **and** the completeness counter increment. |
| Long GPU calls | `max.poll.interval.ms` above worst-case vLLM + OCR for a whole batch. Start at 600,000 ms. |
| Transient failure | Do not commit. In-process exponential backoff, then throw so the partition redelivers. |
| Poison record | At `attempt >= 5`, produce to the group DLQ, **then** commit the original so the partition unblocks. Flag it on the asset row. |
| Idempotency | `_update` keyed by `segment_id`. Redelivery overwrites the same fields. |
| Independence | Separate group offsets. A caption in the DLQ never rolls back a visual upsert. |

An optional delayed `vvs.segments.retry` topic avoids head-of-line blocking when a downstream service is briefly down. In-process retry plus DLQ is enough for a first cut.

### 11.5 Message bodies, no pixels

`vvs.assets`:

```json
{
  "asset_id": "sph-2026-08-30-4412",
  "media_uri": "s3://sphere-mezz/2026/08/30/4412.mp4",
  "source_class": "captured",
  "pack_ids": ["nature"],
  "rights": "internal",
  "duration_s": 1842.0,
  "capture_ts": "2026-08-30T14:02:11Z",
  "site_id": "sphere-lv",
  "trace_id": "0af7651916cd43dd8448eb211c80319c"
}
```

`vvs.segments`:

```json
{
  "asset_id": "sph-2026-08-30-4412",
  "segment_id": "a1b2c3d4e5f6",
  "shot_id": 17,
  "t0": 340.0,
  "t1": 358.2,
  "still_uris": ["s3://sphere-proxies/sph-2026-08-30-4412/a1b2c3d4e5f6.jpg"],
  "motion_energy": 0.31,
  "flash_energy": 0.04,
  "source_class": "captured",
  "pack_ids": ["nature"],
  "rights": "internal",
  "target_index": "vvs_captured_v1",
  "attempt": 0,
  "trace_id": "0af7651916cd43dd8448eb211c80319c"
}
```

`segment_id = sha256(asset_id + ":" + f"{t0:.3f}" + ":" + f"{t1:.3f}")[:12]`. Paths change on re-extract; the id does not. `target_index` travels on the message so a worker never has to re-derive class routing.

---

## 12. Workers

| Worker | Parallelism | Calls | Writes |
| --- | --- | --- | --- |
| Shot, `vvs.shots` | One asset decode per process | `transnet-api`; object-store PUT | `vvs.segments`; Postgres `segment_count` |
| Embed, `vvs.embed` | One in-flight bulk per process | `vllm-embed`; in-process RapidOCR | `_update` `visual_vector` + `ocr` + payload; `visual_done` |
| Caption, `vvs.caption` | Same, usually fewer GPUs | `vllm-instruct` | `_update` `caption`; caption sidecar; `caption_done` |

Neither consumer re-opens the mezzanine. Both read the same proxy still URIs.

OCR belongs in the embed worker. Captured Sphere content is mute, so if embed writes only `visual_vector`, BM25 has nothing for on-screen text and half the hybrid query is dead on the largest class.

ASR is a later group that `_update`s `transcript` on the same `_id`, for produced and wraps with a bed. It must never be required for the captured class.

---

## 13. Index completeness and failure policy

> **Not Kafka's watermark.** Kafka's high and low watermarks are broker-internal log offsets — the high watermark is the offset replicated across the in-sync replica set, which bounds what consumers may read. They describe the *log*, not whether a segment reached OpenSearch. Everything below is application bookkeeping in Postgres: explicit `M of N` counters per asset, incremented only after a successful upsert.
>
> Kafka offsets cannot serve this purpose, for four reasons:
>
> 1. **A committed offset does not mean an indexed segment.** §11.4 commits the offset *after* routing a poison record to the DLQ, precisely so the partition unblocks. Offset progress therefore over-reports success.
> 2. **Two consumer groups advance independently.** Visual completeness and caption completeness are separate gates with different policies.
> 3. **Offsets are per partition, and a partition interleaves many assets.** Deriving one asset's completeness would mean recording its first and last message offsets and reasoning across rebalances.
> 4. **Offsets expire with topic retention** (7 days here). The asset ledger has to outlive the log.
>
> Consumer lag is still the right *operational* metric — how far behind is ingest overall — but it cannot answer "is this asset safe to serve".

| Event | Asset state | Query visibility |
| --- | --- | --- |
| Ingest accepted | `pending` | Not visible |
| All segments published | `segments_ready`, `segment_count = N` | Not visible |
| `visual_done < N` | `visual_partial` | Not visible |
| `visual_done == N` | `searchable` | Hybrid queries may hit this `asset_id` |
| `caption_done == K < N` | `caption_partial` | Visible. BM25 uses whatever `caption` and `ocr` exist |
| `caption_done == N` | `caption_complete` | Visible, full X-Ray |
| Required segment in `vvs.embed.dlq` | `blocked` | Not visible until operator replay |
| Segment in `vvs.caption.dlq` | `searchable` | Visible. Captions have a hole |

Gate the asset all-or-nothing rather than serving partially indexed assets: a half-indexed 3-hour fly-through produces confidently wrong "best match" results, which is worse than the asset being absent.

Proxy stills are **durable**, not scratch. They are the retrieval citation. Garbage-collect the TransNet decode cache, never the JPEG the UI plays against.

---

## 14. Mapping from the laptop prototype

| Prototype (`vvs`) | On-prem |
| --- | --- |
| `data/<id>/video.mp4` | Sphere mezzanine URI, never copied |
| TransNet in-process, 48×27 | `transnet-api`, same decode size |
| 8 s floor in long takes | ~20 s floor, the taxonomy segment |
| Qwen-VL embed in-process | `vllm-embed` |
| Qwen-VL-Instruct in-process | `vllm-instruct` |
| BGE on caption / OCR / ASR | **BM25** on `caption` / `ocr` / `transcript` |
| RapidOCR in-process | Unchanged, inside the embed worker |
| `visual.faiss` + `visual.sqlite` | `visual_vector` + `_source` in one document |
| `text.faiss` + `text.sqlite` | `caption` / `ocr` / `transcript` on the same document |
| Archive fan-out across directories | One `hybrid` query behind one alias |
| `max(visual + caption + OCR, transcript)` in Python | RRF search pipeline, then weighted normalization once judged |
| Motion/flash boost when the query asks | Same rule, as a conditional `should` clause |
| Gradio | Out of scope |

The retrieval philosophy is unchanged: relevance comes from comparable scores across channels, never from keeping or dropping a title because its caption contains the query words.

---

## 15. Throughput and cluster sizing

Two fleets, sized independently: the **steady-state** cluster that serves search and incremental ingest, and the **backfill** fleet that gets 1.11 M hours of history indexed once.

### 15.1 Total work in the backfill

Assumptions, both of which must be benchmarked before procurement (§16):

- **Decode**: a 4K stream at ~4x realtime per 8-thread CPU process, or ~20x realtime per GPU NVDEC engine. TransNet needs *consecutive* frames to see a cut, so frame skipping is not available — every frame must be decoded, even though it is immediately downscaled to 48×27.
- **Embed**: ~100 stills/s per GPU. One forward pass, batchable.
- **Caption**: ~8 captions/s per GPU. Autoregressive generation of 20–30 tokens behind a vision encoder.

| Stage | Total work for 1.11 M hours / 200 M segments |
| --- | --- |
| Decode, CPU at 4x realtime | **277,800 process-hours** |
| Decode, GPU NVDEC at 20x realtime | **55,600 GPU-hours** |
| Caption, 8/s/GPU | 6,940 GPU-hours |
| Embed, 100/s/GPU | 556 GPU-hours |

**Decode dominates by 50 to 500 times.** Embedding is nearly free; captioning costs about 12x embedding. This inverts the intuition that a VLM pipeline is GPU-embedding-bound.

### 15.2 Backfill fleet by target window

| Window | Decode, CPU procs at 4x | Decode, GPUs at 20x | Caption GPUs | Embed GPUs |
| --- | --- | --- | --- | --- |
| 12 months | 32 (~256 cores) | 6 | 1 | <1 |
| 6 months | 63 (~500 cores) | 13 | 2 | <1 |
| **3 months** | **127 (~1,000 cores)** | **26** | **4** | **1** |
| 1 month | 381 (~3,000 cores) | 76 | 10 | 1 |
| 1 week | 1,653 (~13,000 cores) | 331 | 41 | 4 |

A 3-month backfill on CPU decode means roughly **1,000 cores** — about 25–30 servers doing nothing but decoding. If that is unacceptable, move the shot tier to **GPU NVDEC**: 26 decode GPUs replace 1,000 CPU cores for the same window. Benchmark both on ten real Sphere assets before committing, because the answer changes the hardware order by an order of magnitude.

Run the backfill as its own consumer groups on the same topics (`vvs.embed.backfill`, `vvs.caption.backfill`) so it can be rate-limited and scaled to zero independently of live ingest, and so a bad backfill deploy cannot stall new content.

### 15.3 Steady-state cluster

| Pool | Count | Spec | Role |
| --- | --- | --- | --- |
| Shot / TransNet | 8–16 CPU procs, or 2 NVDEC GPUs | Decode-bound | `transnet-api` plus shot workers. Sized for daily new content, not history. |
| vLLM instruct | 2–4 GPU | 1 GPU each | Captions. The larger GPU pool. X-Ray may lag. |
| vLLM embed | 2 GPU | 1 GPU each | One for ingest, **one reserved for query embedding** with a strict latency SLO. |
| Embed / caption workers | Scales with partitions | 4 vCPU, 8 GB each | HTTP clients plus in-process OCR |
| Kafka / Redpanda | 3 brokers | 8 vCPU, 32 GB, 2 TB NVMe | RF=3 |
| OpenSearch manager | 3 | 8 vCPU, 32 GB | Dedicated, never data |
| OpenSearch coordinator | 2–3 | 16 vCPU, 64 GB | Hybrid queries and the RRF pipeline |
| OpenSearch data | 6 | 32 vCPU, 256 GB, 2 TB NVMe | 32 GB heap, 64 shards, ~28 GB resident graph per node |
| Postgres | 1 primary + standby | 8 vCPU, 64 GB | Assets, segments, completeness counters |
| Object store | Existing | 50 PB + proxies | Mezzanine, proxy stills, caption sidecars |

Note the GPU ratio: **more caption GPUs than embed GPUs**, the opposite of what the pipeline diagram suggests. Reserving one embed GPU purely for query embedding is what keeps the 15–40 ms query-embed budget in §9.1 from queueing behind an ingest batch.

---

## 16. Open items

- **Benchmark two numbers on ten real Sphere assets before ordering hardware.** They dominate everything else on this page:
  1. **Average mezzanine bitrate** — swings the vector count and the whole OpenSearch cluster by 20x (§7.1).
  2. **Decode throughput per process, CPU versus NVDEC** — swings the backfill fleet between 26 GPUs and 1,000 CPU cores (§15.2).
- Judgment list on real Sphere queries: RRF versus `min_max` with weights, and whether 32x recall holds or needs 16x.
- Whether Matryoshka truncation to 1024-d is lossless enough to halve the cluster (§7.5).
- Whether the shot tier runs on CPU or GPU NVDEC, and whether NVDEC decode can share the caption GPUs.
- `source_class`, rights, and wrap fields versus Sightline taxonomy inheritance — inherit, never copy.
- Produced-class ASR as a third consumer group writing `transcript`.
- N-frame VLM windows as `_v2` (§10.1).
- A tag ANN as a possible third hybrid clause, if controlled-vocabulary retrieval is required.

---

## 17. Confluence import checklist

- Title: *Voice Video Search — on-prem ingest and retrieval, 50 PB*
- Same space as the CTO demo page; link [`CONFLUENCE.md`](CONFLUENCE.md) as "what runs on the demo box"
- Paste [`CONFLUENCE_ONPREM_INLINE.md`](CONFLUENCE_ONPREM_INLINE.md) if you do not want to attach files. Otherwise attach the ten PNGs from `docs/images/diagrams/onprem-*.png`
- No Mermaid macro is needed for either route
- Flag §7.1 and §15.1 to the storage and platform teams — bitrate and decode throughput are the two numbers they must confirm
