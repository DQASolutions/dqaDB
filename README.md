# dqaDB

Ontology-driven graph database used as a CLI harness for AI agents.
Event files are the source of truth; the index in `.dqadb/` is derived and can be deleted and rebuilt at any time.

Spec: [dqaDB — Spec v0.1](https://claude.ai/artifact/FxKedPmEgCtMp3Jj6xUtCU)

## Layout

```
crates/
  dqadb-core/   library: events, ontology (LinkML), validation, LMDB index, rebuild
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

Done: `init`, `status`, `rebuild`, `verify`, `new-id`, `ontology build|show`, `assert`, `validate`, `retract --event`, `get`, `events show|ls`.
Validation enforces the LinkML schema directly (known predicates/classes, ranges, enums, required, single-valued).

Not yet: SPARQL (`query`), SHACL engine on `shapes.ttl`, RDFS materialisation, search (M2), web UI (M3).
