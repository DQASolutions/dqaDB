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
| Query mix | **802 queries/s** (17.1 s) | **289 queries/s** (47.6 s) |
| Results | | **13,750 / 13,750 identical** |
| Peak RSS | 83 MB | 136 MB |

Mean latency per BSBM query template (ms):

| | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | T11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Oxigraph | 0.28 | 1.07 | 0.48 | 0.51 | 7.05 | 1.30 | 0.92 | 0.09 | 0.80 | 0.14 | 0.28 |
| dqaDB | 1.08 | 2.30 | 1.55 | 1.29 | 21.04 | 3.10 | 2.41 | 0.90 | 1.93 | 0.94 | 1.13 |

T1 product type + features + numeric filter · T2 product details · T3 type + feature
negation · T4 UNION · T5 similar products (numeric range joins) · T6 product + offers +
reviews · T7 reviews · T8 DESCRIBE reviewer · T9 offers of a product · T10 all properties
of an entity · T11 CONSTRUCT offer. Empty-result shares are the same on both systems
(about 11% of the mix's parameters refer to offers/reviews that only the update stream
creates).

Reading:
- dqaDB returns the same answers; it is 2.8x slower on the mix. About 0.7 ms per query
  is fixed cost of `Kb::query` (freshness check against the event files), the rest is
  the dataset layer: T5, a join with numeric range filters, is 3x slower.
- Before this run, `Kb::query` re-parsed the schema files twice per query (3.6 ms fixed
  cost; 168 queries/s on the first 1000 queries). The parsed ontology is now cached until
  a schema file changes, as SQLite caches its schema until the schema cookie changes.

`overhead <kb dir>` times the fixed per-query steps; `compare a.jsonl b.jsonl` checks
results query by query.
