# SPARQL benchmark (tier 2): BSBM

Berlin SPARQL Benchmark, explore use case, on dqaDB and Oxigraph. Standalone
crate like `bench/vector`.

Data: the pre-generated BSBM files published with Oxigraph's own benchmarks
(github.com/Tpt/bsbm-tools release v0.2, no Java needed). Note their naming:

| file | content |
|---|---|
| `explore-2000.nt.zst` | the full 1000-product dataset: 377,241 triples (1000 products, 19,916 offers, 9,977 reviews) |
| `explore-1000.nt.zst` | base data only (types, features, producers, vendors, people; 27,886 triples) for the explore-*and-update* mix |
| `mix-exploreAndUpdate-1000.tsv.zst` | 13,750 queries + 1,000 `INSERT DATA` (adds the 1000 products) + 1,500 `DELETE WHERE` |

Schema: `schema/bsbm.linkml.yaml` is the BSBM vocabulary written by hand (checked against
the data); the product-type hierarchy is data, so `bsbm-schema` generates one LinkML class
per product type (`schema/bsbm_product_types.linkml.yaml`, 151 types at this scale).

```bash
D=~/.cache/dqadb-bench/bsbm; mkdir -p $D
curl -L -o $D/explore-2000.nt.zst https://github.com/Tpt/bsbm-tools/releases/download/v0.2/explore-2000.nt.zst
zstd -d $D/explore-2000.nt.zst
# Oxigraph's RocksDB build needs libclang (the official PyPI `libclang` wheel works) and
# compiler headers for bindgen:
export LIBCLANG_PATH=<dir with libclang.so> BINDGEN_EXTRA_CLANG_ARGS=-I/usr/lib/gcc/x86_64-linux-gnu/12/include
cd bench/sparql && CARGO_TARGET_DIR=~/.cache/dqadb-bench/target-sparql cargo build --release
B=~/.cache/dqadb-bench/target-sparql/release/dqadb-bench-sparql W=~/.cache/dqadb-bench/work-sparql
$B dq-load $D/explore-2000.nt schema $W/dq-1000 && $B ox-load $D/explore-2000.nt $W/ox-1000
$B run dqadb $W/dq-1000 $D/mix-exploreAndUpdate-1000.tsv.zst $W/dq.jsonl
$B run oxigraph $W/ox-1000 $D/mix-exploreAndUpdate-1000.tsv.zst $W/ox.jsonl && $B compare $W/ox.jsonl $W/dq.jsonl
~/.cache/dqadb-bench/target-sparql/release/dqadb-bench-sparql bsbm-schema $D/explore-2000.nt schema/bsbm_product_types.linkml.yaml

dqadb init kb && cd kb && cp ../schema/bsbm.linkml.yaml ontology/schema.linkml.yaml
cp ../schema/bsbm_product_types.linkml.yaml ontology/ && dqadb ontology build --lock-only
dqadb import $D/explore-2000.nt --dry-run && dqadb import $D/explore-2000.nt
```

## Load (dqaDB `import`, 1000 products)

i5-4288U (2 cores / 4 threads), 7.8 GB RAM, SSD; release build.

| | |
|---|---|
| Triples | 377,241 (N-Triples, 96 MB) |
| Validate only (`--dry-run`) | 3.7 s |
| Import (validate + write 8 events + index) | 15.0 s (~25k triples/s), peak RSS 312 MB |
| On disk | events 105 MB, source copy 92 MB, index 109 MB |

## dqaDB vs Oxigraph 0.5.13 (1000 products, 377,241 triples)

Both evaluate SPARQL with the same engine (spareval 0.2.9), in process; what differs is
storage, indexes and the dataset layer. Queries: the 13,750 queries of
`mix-exploreAndUpdate-1000` (updates skipped), in order, one client, results fully read.
Results are compared by value (dqaDB stores canonical literals, e.g. `1200.5` for
`1200.50`).

| | Oxigraph (RocksDB) | dqaDB (SQLite) |
|---|---|---|
| Load | 3.0 s (bulk loader) | 14.8 s (`import`: validation, event files, source copy, index) |
| Disk | 45 MB | 156 MB index (+ 105 MB events, 92 MB source copy) |
| Query mix | 810 queries/s (17.0 s) | **1,408 queries/s** (9.8 s) |
| Results | | **13,750 / 13,750 identical** |
| Peak RSS | 83 MB | 135 MB |

Mean latency per BSBM query template (ms):

| | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | T11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Oxigraph | 0.28 | 1.07 | 0.48 | 0.51 | 7.05 | 1.30 | 0.92 | 0.09 | 0.80 | 0.14 | 0.28 |
| dqaDB | 0.30 | 0.71 | 0.43 | 0.43 | 2.83 | 0.78 | 0.67 | 0.17 | 0.52 | 0.23 | 0.36 |

T1 product type + features + numeric filter · T2 product details · T3 type + feature
negation · T4 UNION · T5 similar products (numeric range joins) · T6 product + offers +
reviews · T7 reviews · T8 DESCRIBE reviewer · T9 offers of a product · T10 all properties
of an entity · T11 CONSTRUCT offer. Empty-result shares are the same on both systems
(about 11% of the mix's parameters refer to offers/reviews that only the update stream
creates).

Reading: at this scale (the whole index fits in memory on both) dqaDB answers the mix
1.7x faster with the same results; Oxigraph is faster on the smallest queries (T8, T10:
dqaDB's fixed cost per query is ~0.1 ms) and loads 5x faster into a 3.5x smaller index.
Larger scales are not measured yet.

How dqaDB got there (each step measured with the diagnostics below, not guessed):

| Step | Queries/s |
|---|---|
| first run | 168 (first 1000 queries) |
| ontology parsed once per schema change, not twice per query (like SQLite's schema cookie) | 289 |
| `events/HEAD` pointer instead of listing event files; RDFS rules computed once | 343 |
| per-query cache of decoded terms (like Jena TDB's node cache) | 389 |
| SQLite query planner stability guarantee (QPSG) | **1,408** |

The last one was the big one: the bundled SQLite is built with STAT4, which marks
statements whose plan could depend on a bound value for re-prepare, and rusqlite clears a
statement's bindings when it returns to the cache, which expires it. Every reuse of a
cached statement was therefore recompiled: 14.5 us per pattern lookup instead of 2.1 us,
and T5 does about 950 lookups. dqaDB never runs ANALYZE, so plans do not depend on bound
values; `SQLITE_DBCONFIG_ENABLE_QPSG` keeps the prepared plan.

Diagnostics: `overhead <kb>` (fixed per-query steps), `profile <kb> <mix> <template>`
(engine calls into the dataset and their time), `lookup <kb> <oxigraph dir>` (one
pattern lookup on both, and the raw SQLite floor).

`overhead <kb dir>` times the fixed per-query steps; `compare a.jsonl b.jsonl` checks
results query by query.
