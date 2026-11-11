# SPARQL benchmark (tier 2): BSBM

Berlin SPARQL Benchmark, explore use case, on dqaDB (and, next, Oxigraph). Standalone
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
cd bench/sparql && CARGO_TARGET_DIR=~/.cache/dqadb-bench/target-sparql cargo build --release
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

Comparison with Oxigraph (load and the query mix) is the next step.
