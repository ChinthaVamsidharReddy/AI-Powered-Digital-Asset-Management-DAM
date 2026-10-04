# Architecture and data flow

```
 MEDIA_DIR ──scan──► files (job queue, state machine) ──► worker threads ──► extractors ──► embedders ──► SQLite
                                                                                                          │
 browser UI ◄── FastAPI ◄── Searcher (in-memory matrices + FTS5 + SQL filters, rank fusion) ◄─────────────┘
```

## Why these choices

| Decision | Reason |
|---|---|
| **Python + FastAPI** | The AI ecosystem (PyTorch, sentence-transformers, PyMuPDF, OpenCV) is Python-native; FastAPI gives a typed REST API with little code. |
| **SQLite (WAL) for everything** | One file, zero setup, transactional, FTS5 built in. Metadata, job queue, vectors and keyword index live together so a file is committed atomically. |
| **CLIP (ViT-B/32) for visuals** | Image and text share one embedding space → "a woman with a cat" matches pixels, not captions or filenames. Runs on CPU. |
| **MiniLM for text** | Fast, strong sentence embeddings for PDF chunks / transcripts. |
| **FTS5 BM25 alongside embeddings** | Exact words, product names, numbers that embeddings blur. |
| **Rank fusion (RRF) instead of score mixing** | CLIP, MiniLM and BM25 scores live on unrelated scales; rank fusion needs no calibration. Absolute cosine floors then suppress junk. |
| **Exact in-memory search (NumPy)** | 10⁵ vectors × 512 dims ≈ 200 MB, searched in a few ms. ANN adds complexity with no gain at this size. Replace with pgvector/Qdrant beyond ~10⁶. |

## Data model

* `assets` - one row per unique **content** (sha256): type, size, dimensions/duration/pages, metadata JSON, status, warnings, thumbnail, `pipeline` (model identity).
* `files` - one row per **path**: size, mtime, state, error, `asset_id`. Duplicate files in different folders are separate rows pointing at the same asset. This table doubles as the persistent work queue.
* `vectors` - embeddings (`clip` space for images / video frames / PDF page renders; `text` space for PDF chunks / transcripts) with `kind`, `ref` (timestamp or page), and a thumbnail of the matched frame/page.
* `vtext` - FTS5 index over the same text chunks (rowid = `vectors.id`).
* `runs` - history of indexing runs.

## Indexing flow

1. **Scan** (`os.walk` + `stat`, batched transactions of 500). Per path:
   new → `pending`; unsupported extension → `unsupported` (reason stored); size/mtime unchanged and `done` → **skipped**; changed → `pending`; previously failed → only with *retry failed*.
   Paths no longer seen are marked `present=0` (kept, so remounting a drive restores them). If a scan finds **zero** files while the index has many, nothing is marked missing (protects against an unmounted drive).
2. **Process** with `WORKERS` threads (decoding/hashing parallel; model inference serialised by a lock). Smallest/cheapest types first (images → PDFs → videos) so searchable content appears early.
   For each file: stream sha256 → if that content is already indexed with the current models, just link the path (**no AI work**: duplicates, touched files) → else extract → embed → write asset + vectors + FTS rows + file state in **one transaction**.
3. **GC** removes assets no path references any more (content replaced) with their vectors and thumbnails.

### Reliability

| Concern | Handling |
|---|---|
| Long runs / huge folders | Streaming hash, bounded memory per file (≤24 frames, ≤8 page renders), progress + ETA, cancel at any time. |
| Crash / kill / power loss | State lives in SQLite; per-file atomic commit; `processing` rows reset to `pending` on startup; just re-run. |
| Corrupt files | Decoders raise → file `failed` with reason (`truncated image`, `cannot open PDF`, `no decodable frames`, empty file). Other files unaffected. |
| Unsupported / encrypted | `unsupported` with reason; listed in UI. |
| AI / model failure | 3 attempts with backoff; then `failed: AI processing failed…`, asset metadata kept; **Retry failed** re-runs only those. Optional steps (ASR, single page render) degrade to *warnings*, not failures. |
| Duplicates | sha256 identity; one analysis, N locations shown. Per-hash lock prevents two workers analysing the same bytes simultaneously. |
| Model/config change | `pipeline` string stored per asset; changing models re-queues affected files automatically and old-pipeline vectors are never mixed into results. |
| Large PDFs / long videos | Text indexed for first 300 pages, 8 evenly spaced page renders; videos have a wall-clock budget (`VIDEO_TIME_BUDGET_SEC`) and fall back to sequential sampling when seeking fails. |

## Per-type understanding

* **Images** - decode (EXIF-rotated, alpha flattened), downscale, CLIP embedding. EXIF camera/date kept as metadata.
* **Videos** - not every frame: up to `VIDEO_MAX_FRAMES` (24) frames evenly spread over the duration (≥ `VIDEO_MIN_INTERVAL` seconds apart), blank frames dropped, then near-duplicate consecutive frames (cosine ≥ 0.97) dropped after embedding. Each kept frame is a vector with its timestamp, so a hit says *where* in the video it matched and the player seeks there. Optional Whisper transcript chunks are indexed as text.
* **PDFs** - per-page text → ~900-character overlapping chunks (text embedding + BM25), document title/keywords, plus up to 8 rendered pages through CLIP so brochures dominated by imagery (or scanned) are still findable.

## Search flow

1. SQL filters (type, extension, size, folder, index time, `present=1`, current pipeline) → allowed asset ids.
2. Query is embedded by CLIP-text and MiniLM; BM25 runs on non-stopword terms.
3. Per signal, candidates must pass an **absolute cosine floor** and lie within a **relative margin** of the best hit for that query. The floor makes unrelated (negative) queries return nothing; the margin cuts the weakly-related tail adaptively instead of returning the whole corpus. Best frame/page/chunk per asset wins.
4. Weighted RRF per signal (`clip 1.0, text 1.0, keyword 0.5`), combined as `max + 0.25 × (others)`. A plain sum favoured PDFs (three signals) over images/videos (one signal); the max-based combination removes most of that bias. Type words in the query ("videos", "brochures", "photos") give a soft ×1.3 boost when no explicit type filter is set.
5. Response carries **match strength** (absolute 0–100 % mapped from the raw similarity - not normalised to the top hit, so a poor best match shows a low number), per-signal evidence, the matching frame timestamp / PDF page, the extracted matching text, per-type facet counts and all file locations.

Why this design: the first test round showed that normalising to the top hit made every query's best result "100 %", that every video/PDF query returned nearly the whole video/PDF set, and that PDFs crowded out images/videos for generic queries. Steps 3-5 are the fix.

## Scaling a larger collection (what changes at 100 GB – 10 TB)

* Scan is O(files) of `stat` calls - fine to millions; replace with FS watchers for near-real-time.
* Throughput is bounded by model inference: use GPU, larger batches, or multiple worker processes pulling from a Postgres/Redis queue (the `files.state` machine ports directly; claim rows with `SELECT … FOR UPDATE SKIP LOCKED`).
* Vectors → pgvector HNSW / Qdrant; quantise (fp16/int8) to cut memory 2–4×; shard by folder or tenant.
* Store thumbnails in object storage; serve originals through signed URLs instead of local paths.
