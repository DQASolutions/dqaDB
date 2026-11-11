# Vector benchmark (tier 1)

Standalone crate (LanceDB's dependency tree takes ~45 min to build on a laptop CPU).

```bash
# data: VectorDBBench Cohere 1M (3.1 GB)
mkdir -p ~/.cache/dqadb-bench/cohere-1m && cd ~/.cache/dqadb-bench/cohere-1m
for f in test neighbors shuffle_train; do curl -LO https://assets.zilliz.com/benchmark/cohere_medium_1m/$f.parquet; done

# build (needs protoc for lance)
cd bench/vector
export PROTOC=/path/to/protoc CARGO_TARGET_DIR=~/.cache/dqadb-bench/target RUSTFLAGS="-C target-cpu=native"
cargo build --release
B=$CARGO_TARGET_DIR/release/dqadb-bench-vector D=~/.cache/dqadb-bench/cohere-1m W=~/.cache/dqadb-bench/work

$B $D $W gt                                   # filtered ground truth (once)
$B $D $W dqadb; $B $D $W lance-hnsw; $B $D $W lance-pq            # load (+ search)
BENCH_REUSE=1 $B $D $W dqadb                  # search only, on the built index
BENCH_REUSE=1 $B $D $W lance-hnsw; BENCH_REUSE=1 $B $D $W lance-pq
$B $D $W stream-dqadb; $B $D $W stream-lance  # streaming case (first 300k vectors)
```

Big-ANN NeurIPS'23 streaming runbook (wikipedia-1M files and runbook YAML from
big-ann-benchmarks; Qdrant in Docker on :6333 for the comparison):

```bash
D=~/.cache/dqadb-bench/wikipedia-1M R=~/.cache/dqadb-bench/runbooks
RB=$R/wikipedia-1M_expiration_time_replace_delete_runbook.yaml
$B $D $W rb-gt $RB wikipedia-1M cos $D/cos-replace-delete       # cosine ground truth (once)
$B $D $W rb-dqadb $RB wikipedia-1M $D/cos-replace-delete full 4  # dqaDB, 4 search threads
$B $D $W rb-qdrant $RB wikipedia-1M $D/cos-replace-delete full 4
# quicker: RB_QUERIES=1000 RB_EVERY=2; checkpoints RB_CKPT=150,300, resume RB_RESUME=$W/ckpt-<label>-150,
# stop RB_STOP=<step>, per-step segment state RB_SEGLOG=1, widths RB_EFS=64,128,256
# diagnostics on the state left in $W/dqadb-runbook: rb-diag, rb-diag-build, rb-diag-append,
# rb-diag-filter, rb-diag-dot
```

Results: `RESULTS.md`; raw JSON in the work directory.
