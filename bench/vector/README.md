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
```

Results: `RESULTS.md`; raw JSON in the work directory.
