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

Not yet: vector search and bge-m3 embeddings (M2), web UI (M3). SHACL: validation runs from LinkML directly; `shapes.ttl` is an export (no runtime SHACL engine until hand-written shapes need one).

## Index store

The derived index is a single SQLite file, `.dqadb/index.sqlite` (WAL mode). SQLite was chosen over LMDB after a benchmark on this workload (5M triples): LMDB reads were 2-3x faster, but SQLite files were 2.7x smaller, small commits ~15x faster, its file format is guaranteed stable to 2050, many processes can read while one writes, and the index can be inspected with `sqlite3 .dqadb/index.sqlite`. redb, fjall, sled, RocksDB and DuckDB were ruled out because they do not allow a second process to read while another holds the database.

## Full-text search

```bash
dqadb search --text "hai phong"                       # accents and đ/d ignored
dqadb search --text "vận tải" --type ex:Company --where 'ex:country = "VN"'
```

Full-text runs on tantivy (pinned `=0.27.0`) in `.dqadb/fts/`, as a derived projection with its own watermark stored in the tantivy commit payload. `assert` never touches it; the next text search catches up on the entities touched since that watermark, in one commit. A crash at any point leaves an older watermark, so the next search repairs it; a change to which predicates have `dqadb_index: fts` triggers a re-index. Benchmark at 1M documents: p50 0.5-0.8 ms, p95 under 2 ms, ~100 MB (SQLite FTS5: 9-45 ms, p95 up to ~150 ms).

## SPARQL

```bash
dqadb query 'SELECT ?c ?v WHERE { ?c a ex:Contract ; ex:value ?v FILTER(?v > 10000) }'
dqadb query 'SELECT ?g WHERE { GRAPH ?g { ?s ex:country "VN" } }'   # which event said it
```

The default graph is the current state (asserted + inferred facts). Named graph `dqadb:event:<ULID>` (`urn:dqadb:event:<ULID>`) holds what that event asserted and is still in effect. KB prefixes are predeclared; `--limit` and `--timeout` bound the work.
