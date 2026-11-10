# dqaDB

Ontology-driven graph database used as a CLI harness for AI agents.
Event files are the source of truth; the index in `.dqadb/` is derived and can be deleted and rebuilt at any time.

Spec: [dqaDB — Spec v0.1](https://claude.ai/artifact/FxKedPmEgCtMp3Jj6xUtCU)

## Layout

```
crates/
  dqadb-core/   library: events, ontology (LinkML), validation, SQLite index, RDFS, SPARQL
  dqadb-cli/    the `dqadb` binary
```

## Build and test

```bash
cargo build --release        # binary at target/release/dqadb
cargo test
```

Requires Rust 1.89+ (uses `std::fs::File::lock`).

## Quick start

```bash
dqadb init my-kb && cd my-kb
dqadb ontology build              # needs LinkML: pip install linkml
dqadb ontology build --lock-only  # or: skip TTL/SHACL generation (no Python)
dqadb ontology show

cat > facts.json <<'EOF'
[{"op":"add","s":"_:acme","p":"rdf:type","o":"ex:Company"},
 {"op":"add","s":"_:acme","p":"rdfs:label","o":{"@value":"ACME Logistics","@language":"vi"}},
 {"op":"add","s":"_:acme","p":"ex:country","o":"VN"}]
EOF
dqadb assert -f facts.json --actor agent:extractor --source contract.pdf
dqadb get dqadb:<id printed above>
dqadb events ls
rm -rf .dqadb && dqadb rebuild   # same fingerprint as before
```

Exit codes: 0 ok · 1 error · 2 validation · 3 not found · 4 ontology not built/stale · 5 index stale · 6 version conflict or lock timeout. `--json` gives machine-readable output (errors go to stderr as JSON).

## Status (milestone M1)

Done: `init`, `status`, `rebuild`, `verify`, `new-id`, `ontology build|show`, `assert`, `validate`, `retract --event`, `get`, `query` (SPARQL 1.1), `search` (`--text` full-text, `--type`, `--where` filters), `traverse`, `path`, `events show|ls`; RDFS subclass inference.
Validation enforces the LinkML schema directly (known predicates/classes, ranges, enums, required, single-valued).

Not yet: web UI (M3). SHACL: validation runs from LinkML directly; `shapes.ttl` is an export (no runtime SHACL engine until hand-written shapes need one).

## Index store

The derived index is a single SQLite file, `.dqadb/index.sqlite` (WAL mode). SQLite was chosen over LMDB after a benchmark on this workload (5M triples): LMDB reads were 2-3x faster, but SQLite files were 2.7x smaller, small commits ~15x faster, its file format is guaranteed stable to 2050, many processes can read while one writes, and the index can be inspected with `sqlite3 .dqadb/index.sqlite`. redb, fjall, sled, RocksDB and DuckDB were ruled out because they do not allow a second process to read while another holds the database.

## Full-text search

```bash
dqadb search --text "hai phong"                       # accents and đ/d ignored
dqadb search --text "vận tải" --type ex:Company --where 'ex:country = "VN"'
```

Full-text runs on tantivy (pinned `=0.27.0`) in `.dqadb/fts/`, as a derived projection with its own watermark stored in the tantivy commit payload. `assert` never touches it; the next text search catches up on the entities touched since that watermark, in one commit. A crash at any point leaves an older watermark, so the next search repairs it; a change to which predicates have `dqadb_index: fts` triggers a re-index. Benchmark at 1M documents: p50 0.5-0.8 ms, p95 under 2 ms, ~100 MB (SQLite FTS5: 9-45 ms, p95 up to ~150 ms).

## Semantic search

```bash
dqadb model install        # official bge-m3 ONNX (Hugging Face) + official ONNX Runtime (PyPI) -> ~/.cache/dqadb, sha256-verified
dqadb serve &              # resident: keeps the model loaded, embeds changes in the background, answers query embedding
dqadb search --semantic "công ty vận chuyển đường biển"
dqadb search --semantic "vận tải hàng hóa" --text "container"               # text required, ranked by meaning
dqadb search --semantic "vận tải hàng hóa" --text "container" --text-mode fuse   # RRF of both rankings
dqadb embed                # one-shot catch-up when serve is not running
```

Without `dqadb serve`, a semantic search loads the model in-process (~4 s on a 2013 laptop CPU); with it, ~0.3 s. Paths can be overridden with `embedding.model_dir` / `embedding.onnxruntime` in `dqadb.yaml` or `DQADB_MODEL_DIR` / `DQADB_ORT_LIB`.

Embeddings: official BAAI bge-m3 ONNX (fp32) on the official ONNX Runtime, loaded dynamically (glibc 2.27+). Each entity's vector is built from `rdfs:label` and its `dqadb_index: vector` literals; vectors live in the SQLite index (f16) and in `emb-cache/<model>.sqlite` (f32, by content hash), so a rebuild never calls the model again. `assert` never embeds; `dqadb embed` catches up by watermark, filling the cache without holding the write lock. Searches report `pending` events not yet embedded.

Vector index: from 20k vectors on, immutable HNSW segments (usearch `=2.26.6`, f16, memory-mapped, `.dqadb/vec/`), laid out like Lucene / Qdrant. Deletes and updates never touch a graph: each segment has a versioned deletion list (deleted nodes still route searches but are never returned), and changes since the last flush are scanned exactly. `dqadb embed` flushes the delta into a new segment once 2,048 entities changed, merges segments of similar size 8 at a time (tiered, up to 262,144 vectors per segment), and rebuilds a segment from its live vectors once 20% of it is deleted (Qdrant's default); `--compact` rebuilds everything. Only the segments involved are rebuilt, each file is written aside and renamed into place, and `segments.json` names the current set, so updates never block writers or readers. Each segment is built from one thread (usearch graphs from concurrent inserts were measurably worse: recall@10 at ef 512 on 135k Wikipedia vectors 0.995 with 1 thread, 0.982 with 4); different segments are built in parallel. The design was chosen on the Big-ANN NeurIPS'23 streaming runbook (wikipedia-1M replace-delete) against DiskANN and Qdrant, see `bench/vector`. Filtered searches score the candidates exactly when they are few, and use the segments restricted to the candidates when they are many; the switch point is sqrt(250 x vectors) (about 5k candidates at 100k vectors, measured), overridable with DQADB_EXACT_MAX.

## SPARQL

```bash
dqadb query 'SELECT ?c ?v WHERE { ?c a ex:Contract ; ex:value ?v FILTER(?v > 10000) }'
dqadb query 'SELECT ?g WHERE { GRAPH ?g { ?s ex:country "VN" } }'   # which event said it
```

The default graph is the current state (asserted + inferred facts). Named graph `dqadb:event:<ULID>` (`urn:dqadb:event:<ULID>`) holds what that event asserted and is still in effect. KB prefixes are predeclared; `--limit` and `--timeout` bound the work.
