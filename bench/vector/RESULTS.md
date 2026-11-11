# Tier 1 — vector search: dqaDB vs LanceDB

Run 2026-11-10. Conventions from VectorDBBench (load / search / filtered cases, p99,
concurrent QPS), ANN-Benchmarks (recall@10 vs QPS over a parameter sweep) and Big-ANN
NeurIPS'23 (filtered recall against the filtered ground truth).

## Setup

| | |
|---|---|
| Dataset | VectorDBBench **Cohere 1M** (1,000,000 × 768, cosine), official `neighbors.parquet` ground truth |
| Queries | 1,000 for the sweep, 200 per filter level |
| Filters | integer `ord < X` keeping 1% / 10% / 50% / 99%; ground truth by brute force (checked: keep-99% vs official truth = 0.987) |
| Machine | Intel i5-4288U (2 cores / 4 threads, 2013), 7 GB RAM, SSD, Linux 6.8 |
| dqaDB | dqadb-core 0.1.0: HNSW base usearch 2.26.6 int8 + exact f16 rerank; adaptive post-filter |
| LanceDB | lancedb 0.41.0 / lance 14.0.2, `IVF_HNSW_SQ` and `IVF_PQ` (cosine), BTree index on `ord` |
| Method | library calls in-process; one process per system; search runs reuse the built index and do not hold the 3 GB of base vectors; 50 warm-up queries per point; 4-thread QPS = 4 independent clients |

## Load

| | Insert | Index build | Disk | 
|---|---|---|---|
| dqaDB | 153 s (validated events + graph) | 766 s (vector store 340 s + HNSW 427 s) | 5.5 GB (incl. 0.47 GB event files, 1.6 GB embedding cache) |
| LanceDB IVF_HNSW_SQ | 15 s | 1,123 s | 4.1 GB |
| LanceDB IVF_PQ | 14 s | 106 s | 3.1 GB |

## Search (unfiltered), recall@10 vs speed

| System | Param | Recall@10 | p50 ms | p99 ms | QPS 1 thread | QPS 4 threads |
|---|---|---|---|---|---|---|
| dqaDB | ef=64 | 0.938 | 3.66 | 6.86 | 254 | 457 |
| dqaDB | ef=128 (default) | 0.958 | 4.11 | 7.83 | 227 | 446 |
| dqaDB | ef=256 | 0.973 | 5.20 | 8.93 | 183 | 374 |
| dqaDB | ef=512 | **0.981** | 7.16 | 12.60 | 133 | 302 |
| LanceDB HNSW_SQ | ef=32 | 0.921 | 2.96 | 4.24 | 332 | 853 |
| LanceDB HNSW_SQ | ef=64 | 0.952 | 3.16 | 5.10 | 306 | 762 |
| LanceDB HNSW_SQ | ef=128 | 0.968 | 5.23 | 10.90 | 163 | 604 |
| LanceDB HNSW_SQ | ef=512 | 0.976 | 6.95 | 10.91 | 139 | 314 |
| LanceDB IVF_PQ | nprobes=20 | 0.576 | 8.47 | 10.68 | 117 | 237 |
| LanceDB IVF_PQ | nprobes=20, refine=10 | 0.915 | 13.94 | 73.59 | 53 | 123 |
| LanceDB IVF_PQ | nprobes=50, refine=10 | 0.925 | 23.31 | 46.48 | 40 | 74 |

## Filtered search (default search parameters)

| Keep | dqaDB recall / p50 / p99 | LanceDB HNSW_SQ | LanceDB IVF_PQ |
|---|---|---|---|
| 1% | **0.996** / 37 / 139 ms | 0.966 / 10.7 / 12.8 ms | 0.411 / 8.7 / 34.3 ms |
| 10% | **0.982** / 35 / 86 ms | 0.931 / 13.5 / 21.6 ms | 0.480 / 11.6 / 45.2 ms |
| 50% | **0.979** / 16.8 / 47.8 ms | 0.878 / 23.6 / 30.1 ms | 0.537 / 25.7 / 81.5 ms |
| 99% | **0.986** / 14.0 / 21.6 ms | 0.860 / 36.1 / 46.0 ms | 0.559 / 40.7 / 132.8 ms |

## Streaming (Big-ANN streaming track / VectorDBBench streaming, scaled)

Sliding window: start with 100k vectors, then 10 steps of +20k new / −10k oldest
(live set 110k → 200k). Each step: write, then the maintenance each system needs
(dqaDB `embed`, LanceDB `optimize`), with a reader searching non-stop throughout.
Recall is against the moving ground truth (brute force over the live set). "Fresh" =
right after the write, before maintenance.

| | dqaDB | LanceDB IVF_HNSW_SQ |
|---|---|---|
| Initial 100k load + index | 65 s | 82 s |
| Write per step until searchable (+20k / −10k) | 6–8 s typical, spikes 24 / 55 s (validated events + vector projection) | 0.2–0.3 s |
| Maintenance per step | 10–17 s (incremental HNSW update); 128 s once (full rebuild at 25% tombstones) | 16–98 s |
| Recall@10 settled (mean) | **0.964** (0.962–0.968, no downward trend) | 0.902 (0.860–0.918) |
| Recall@10 fresh, right after write (mean) | **0.971** | 0.927 |
| Hits pointing at deleted vectors | 0 | 0 |
| Search p50 / p99 settled (200k) | 7.1 / 11.2 ms | 5.9 / 9.9 ms |
| Search p50 / p99 during write + maintenance | **3–9 / 94–173 ms** | 91–137 / ~195 ms |

*This section measured the previous design (one HNSW base + tombstones); it was replaced
by segments after the Big-ANN runbook below.* Index maintenance followed the published "logical deletion + periodic rebuild" scheme
(Yamashita et al., NeurIPS'25 workshop; also what Qdrant does inside a segment):
changed vectors are tombstoned in the HNSW graph (kept for routing, never returned),
new ones are added to a copy of the base, and the base is rebuilt once tombstones pass
25%. History of this design, measured on this benchmark:

| Variant | Settled recall over 10 steps | Maintenance per step |
|---|---|---|
| Full rebuild every time the delta passes 10% | 0.955–0.963 | 54 → 94 s (grows with data) |
| Incremental, physical removal (usearch reuses freed slots) | 0.954 → 0.935 (declining) | 10–16 s |
| **Incremental, tombstones + rebuild at 25%** (then current) | **0.962–0.968 (stable)** | 10–17 s |

## Big-ANN NeurIPS'23 streaming runbook (wikipedia-1M, replace-delete)

The official streaming-track runbook `wikipedia-1M_expiration_time_replace_delete_runbook`
(Cohere 768-d Wikipedia embeddings, inserts + deletes + in-place replaces, live set up to
~415k, 100 search steps). Ground truth recomputed for cosine; recall@10 is tie-aware.
4 search threads, one 2-core / 4-thread laptop (i5-4288U, 7.8 GB). dqaDB goes through its
full write path (validated events, embedding projection, `embed` maintenance after each
step) and is searched through `kb.search`; Qdrant 1.20 runs in Docker with its defaults
and is searched over HTTP; DiskANN is Microsoft's Rust crate run by its own benchmark tool.

| System | Search width | Recall@10 mean | Min step | QPS | Notes |
|---|---|---|---|---|---|
| DiskANN (in-place deletes) | L 100 / 200 | 0.978 / 0.992 | 0.970 / – | ~1,670 at L 100 | in-memory index only; first ~80 steps |
| Qdrant 1.20 (segments) | ef 128 / 256 | 0.976 / 0.990 | 0.971 / – | ~350 at ef 128 (HTTP) | first 80 steps |
| dqaDB, previous design (base + tombstones) | ef 128 / 256 | 0.956 / 0.971 | 0.948 / – | 290 → 65–135 (decaying) | |
| **dqaDB, segments** | ef 64 / **128** / 256 | 0.972 / **0.989** / 0.996 | 0.963 / **0.985** / 0.994 | 390 / **332** / 247, stable | all steps, 1000 queries, every 2nd step |

Recall at a given parameter is not comparable across systems (ef and L mean different
amounts of work); compare at similar QPS. dqaDB's acceptance bar for the segment design
was recall ≥ 0.97 at the default ef 128, min step ≥ 0.965, QPS not decaying.

The segment design, each part taken from a shipping system:
Lucene/Qdrant-style immutable segments with per-segment deletion lists; Qdrant's merge
cap (one segment per core below full size, merge the smallest); merges that insert into
the largest graph instead of rebuilding (Lucene `IncrementalHnswGraphMerger`, Qdrant 1.14
incremental HNSW); vacuum at 20% deleted (Qdrant's default); replaced files deleted at
once (Lucene `IndexFileDeleter`); unflushed changes scanned exactly from memory (Qdrant's
appendable segment). Measured effect of each step on this runbook:

| Change | Effect |
|---|---|
| Single-threaded segment builds (usearch concurrent inserts degrade the graph) | recall at ef 512 on 135k: 0.982 (4 threads) → 0.995 |
| Qdrant merge cap instead of tiered factor 8 (fewer segments to search) | QPS stable instead of decaying |
| Append merges (simulated: rebuilt vectors 7.6M → 3.4M) | maintenance 5,483 s → 2,293 s; recall 0.990 → 0.989 |
| Append into a clean graph vs fresh build (200k, isolated) | recall 0.9729 vs 0.9730 at ef 128, 40% less time |
| Delete replaced files at once (was a 10-minute grace) | vector directory 3.3 GB → 0.7 GB |
| In-memory delta + vectorized dot (steps with 1–2k unflushed changes) | 228 → 329 QPS (no-delta steps ~380) |

The last two were measured by resuming from saved checkpoints (`RB_RESUME`), not over
the full run.

## Reading

- Unfiltered: dqaDB and LanceDB HNSW_SQ trace about the same single-thread curve; dqaDB
  reaches the highest recall (0.981). LanceDB scales better with 4 clients (604–853 vs
  446–457 QPS around 0.95–0.97 recall).
- Filtered: dqaDB keeps recall ≥ 0.979 at every selectivity; LanceDB HNSW_SQ drops to
  0.86–0.88 for loose filters. For tight filters (1–10%) LanceDB is 3× faster at lower recall.
- Big-ANN runbook: with segments dqaDB meets its bar (0.989 at ef 128, min 0.985) with
  stable QPS, in the same range as Qdrant at the same ef; DiskANN reaches the same recall
  at ~5× the QPS but is an in-memory library without persistence or multi-process reads.
  Maintenance (merges, vacuums) is the largest cost: ~38 min of the 52-min run.
- Streaming (previous design): dqaDB keeps recall 0.96–0.97 as data churns (LanceDB 0.86–0.92), both
  fresh and settled, and keeps searches fast while writing (p50 3–9 ms vs ~100 ms).
  Its weak point: writes are ~25× slower than LanceDB's appends (validation, durable
  event files, graph index), with occasional unexplained spikes.
- Load: dqaDB indexes faster than IVF_HNSW_SQ, much slower than IVF_PQ; it also writes
  durable, validated event files. Disk is 1.3–1.8× LanceDB, of which 2.1 GB is the
  source of truth and the embedding cache.

## Known limits of this run

- One run per point (no repetition variance yet); one machine.
- LanceDB index-build parameters (partitions, sub-vectors) left at defaults; search
  parameters swept. Filtered cases use each system's default search parameters.

Reproduce: see `README.md` in this directory.
