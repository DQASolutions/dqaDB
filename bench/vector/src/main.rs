//! Tier-1 vector benchmark for dqaDB vs LanceDB, following VectorDBBench / ANN-Benchmarks /
//! Big-ANN conventions:
//! - dataset: VectorDBBench Cohere 1M (768 dims, cosine) with its official ground truth;
//! - load: insert time, index build time, disk size, peak RSS;
//! - search: recall@10 vs QPS curve over the search-breadth parameter, latency p50/p95/p99,
//!   serial QPS and 4-thread QPS;
//! - filtered search: integer filter keeping 1% / 10% / 50% / 99% of the data, recall
//!   against the filtered ground truth.
//! Each system runs in its own process. Usage:
//!   dqadb-bench-vector <data dir> <work dir> gt|dqadb|lance-hnsw|lance-pq [max N]

use arrow_array::types::Float32Type;
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Int64Array, LargeListArray, RecordBatch,
    RecordBatchIterator, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use dqadb_core::embed::Embedder;
use dqadb_core::ontology::InputOp;
use dqadb_core::search::SearchQuery;
use dqadb_core::{Kb, Result as DqResult, WriteOpts};
use futures::TryStreamExt;
use lancedb::DistanceType;
use lancedb::index::Index;
use lancedb::index::scalar::BTreeIndexBuilder;
use lancedb::index::vector::{IvfHnswSqIndexBuilder, IvfPqIndexBuilder};
use lancedb::query::{ExecutableQuery, QueryBase};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

mod runbook;
mod stream;

const DIM: usize = 768;
const K: usize = 10;
const SWEEP_QUERIES: usize = 1000;
const FILTER_QUERIES: usize = 200;
const SELECTIVITIES: [f64; 4] = [0.01, 0.10, 0.50, 0.99];

// ---------------- data ----------------

struct Data {
    n: usize,
    /// row-major, L2-normalised, indexed by dataset id
    train: Arc<Vec<f32>>,
    queries: Arc<Vec<f32>>,
    /// official top-K neighbours per query (unfiltered)
    truth: Vec<Vec<u64>>,
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// Streams record batches (never holds the whole file in memory).
fn read_parquet(path: &Path) -> impl Iterator<Item = RecordBatch> {
    let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    ParquetRecordBatchReaderBuilder::try_new(f)
        .unwrap()
        .with_batch_size(10_000)
        .build()
        .unwrap()
        .map(|b| b.unwrap())
}

/// (id, vector) rows from a VectorDBBench parquet file with columns `id`, `emb`.
fn read_vectors(path: &Path, max: usize, out: &mut Vec<f32>) -> usize {
    let mut n = 0;
    for b in read_parquet(path) {
        let ids = b
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .clone();
        let emb = b
            .column_by_name("emb")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeListArray>()
            .unwrap()
            .clone();
        for i in 0..b.num_rows() {
            let id = ids.value(i) as usize;
            if id >= max {
                continue;
            }
            let row = emb.value(i);
            let vals = row.as_any().downcast_ref::<Float32Array>().unwrap();
            let dst = &mut out[id * DIM..(id + 1) * DIM];
            dst.copy_from_slice(vals.values());
            normalize(dst);
            n += 1;
        }
    }
    n
}

fn load(data_dir: &Path, max: usize) -> Data {
    let t = Instant::now();
    // search-only runs reuse built indexes and never hold the 3 GB of base vectors,
    // which would otherwise evict the indexes from the page cache on a 7 GB machine
    let search_only = std::env::var_os("BENCH_REUSE").is_some();
    let (n, train) = if search_only {
        (max, Vec::new())
    } else {
        let mut train = vec![0f32; max * DIM];
        (
            read_vectors(&data_dir.join("shuffle_train.parquet"), max, &mut train),
            train,
        )
    };
    let mut queries = vec![0f32; SWEEP_QUERIES * DIM];
    let q = read_vectors(&data_dir.join("test.parquet"), SWEEP_QUERIES, &mut queries);
    assert_eq!(q, SWEEP_QUERIES, "expected {SWEEP_QUERIES} test queries");
    let mut truth = vec![Vec::new(); SWEEP_QUERIES];
    for b in read_parquet(&data_dir.join("neighbors.parquet")) {
        let ids = b
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .clone();
        let nb = b
            .column_by_name("neighbors_id")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeListArray>()
            .unwrap()
            .clone();
        for i in 0..b.num_rows() {
            let qid = ids.value(i) as usize;
            if qid < SWEEP_QUERIES {
                let row = nb.value(i);
                let v = row.as_any().downcast_ref::<Int64Array>().unwrap();
                truth[qid] = v.values().iter().take(K).map(|x| *x as u64).collect();
            }
        }
    }
    eprintln!(
        "loaded {n} vectors x {DIM} + {SWEEP_QUERIES} queries in {:.0}s",
        t.elapsed().as_secs_f64()
    );
    Data {
        n,
        train: Arc::new(train),
        queries: Arc::new(queries),
        truth,
    }
}

fn query(d: &Data, i: usize) -> &[f32] {
    &d.queries[i * DIM..(i + 1) * DIM]
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    for (x, y) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    acc.iter().sum()
}

// ---------------- ground truth for filters ----------------

#[derive(Serialize, Deserialize)]
struct FilteredTruth {
    n: usize,
    /// selectivity -> per-query top-K ids among ids < selectivity * n
    by_selectivity: Vec<(f64, Vec<Vec<u64>>)>,
}

fn threshold(n: usize, sel: f64) -> usize {
    ((n as f64) * sel).round() as usize
}

fn filtered_truth(d: &Data, path: &Path) -> FilteredTruth {
    if let Ok(t) = std::fs::read(path).map(|b| serde_json::from_slice::<FilteredTruth>(&b)) {
        if let Ok(t) = t {
            if t.n == d.n {
                return t;
            }
        }
    }
    let t0 = Instant::now();
    let per_query: Vec<Vec<Vec<u64>>> = (0..FILTER_QUERIES)
        .into_par_iter()
        .map(|qi| {
            let q = query(d, qi);
            let scores: Vec<f32> = (0..d.n)
                .map(|i| dot(q, &d.train[i * DIM..(i + 1) * DIM]))
                .collect();
            SELECTIVITIES
                .iter()
                .map(|s| {
                    let lim = threshold(d.n, *s);
                    let mut idx: Vec<u64> = (0..lim as u64).collect();
                    idx.sort_by(|a, b| {
                        scores[*b as usize]
                            .partial_cmp(&scores[*a as usize])
                            .unwrap()
                    });
                    idx.truncate(K);
                    idx
                })
                .collect()
        })
        .collect();
    let by_selectivity = SELECTIVITIES
        .iter()
        .enumerate()
        .map(|(si, s)| (*s, per_query.iter().map(|q| q[si].clone()).collect()))
        .collect();
    let t = FilteredTruth {
        n: d.n,
        by_selectivity,
    };
    std::fs::write(path, serde_json::to_vec(&t).unwrap()).unwrap();
    eprintln!(
        "filtered ground truth computed in {:.0}s",
        t0.elapsed().as_secs_f64()
    );
    t
}

// ---------------- reporting ----------------

#[derive(Serialize, Default, Clone)]
struct Point {
    system: String,
    case: String,
    param: String,
    recall_at_10: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    qps_1_thread: f64,
    qps_4_threads: f64,
}

#[derive(Serialize, Default)]
struct Report {
    system: String,
    version: String,
    n: usize,
    insert_s: f64,
    index_s: f64,
    disk_mb: f64,
    peak_rss_mb: f64,
    points: Vec<Point>,
}

fn recall(got: &[u64], truth: &[u64]) -> f64 {
    let t: std::collections::HashSet<&u64> = truth.iter().collect();
    got.iter().take(K).filter(|g| t.contains(g)).count() as f64 / truth.len().clamp(1, K) as f64
}

fn percentiles(mut v: Vec<Duration>) -> (f64, f64, f64) {
    v.sort();
    let at = |p: f64| v[((v.len() as f64 * p) as usize).min(v.len() - 1)].as_secs_f64() * 1e3;
    (at(0.50), at(0.95), at(0.99))
}

fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with("VmHWM")).map(|l| {
                l.split_whitespace()
                    .nth(1)
                    .unwrap_or("0")
                    .parse::<f64>()
                    .unwrap_or(0.0)
            })
        })
        .unwrap_or(0.0)
        / 1024.0
}

fn dir_mb(p: &Path) -> f64 {
    fn size(p: &Path) -> u64 {
        if p.is_file() {
            return p.metadata().map(|m| m.len()).unwrap_or(0);
        }
        std::fs::read_dir(p)
            .map(|d| d.map(|e| size(&e.unwrap().path())).sum())
            .unwrap_or(0)
    }
    size(p) as f64 / 1e6
}

fn print_point(p: &Point) {
    println!(
        "{:<22} {:<14} {:<10} recall@10 {:.3}  p50 {:>7.2}  p95 {:>7.2}  p99 {:>7.2} ms  QPS 1t {:>7.1}  4t {:>7.1}",
        p.system,
        p.case,
        p.param,
        p.recall_at_10,
        p.p50_ms,
        p.p95_ms,
        p.p99_ms,
        p.qps_1_thread,
        p.qps_4_threads
    );
}

// ---------------- dqaDB ----------------

/// Maps entity text `v<id>` to dataset vector `id` and `q<i>` to query `i`.
struct Precomputed {
    train: Arc<Vec<f32>>,
    queries: Arc<Vec<f32>>,
}

impl Embedder for Precomputed {
    fn id(&self) -> &str {
        "cohere-1m-precomputed"
    }
    fn dim(&self) -> usize {
        DIM
    }
    fn embed(&self, texts: &[String]) -> DqResult<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let (src, i) = match t.split_at(1) {
                    ("v", n) => (&self.train, n.trim().parse::<usize>().unwrap()),
                    ("q", n) => (&self.queries, n.trim().parse::<usize>().unwrap()),
                    _ => panic!("unexpected text {t}"),
                };
                src[i * DIM..(i + 1) * DIM].to_vec()
            })
            .collect())
    }
}

const SCHEMA: &str = "id: https://example.org/bench\nname: bench\nprefixes:\n  ex: https://example.org/bench#\n  linkml: https://w3id.org/linkml/\ndefault_prefix: ex\ndefault_range: string\nimports:\n  - linkml:types\nclasses:\n  Doc:\n    slots: [label, ord]\nslots:\n  label:\n    slot_uri: rdfs:label\n    required: true\n  ord:\n    range: integer\n    annotations:\n      dqadb_index: filter\n";

fn open_dqadb(root: &Path, d: &Data) -> Kb {
    let kb = Kb::open(root).unwrap();
    kb.set_embedder(Box::new(Precomputed {
        train: d.train.clone(),
        queries: d.queries.clone(),
    }))
    .unwrap();
    kb
}

fn dqadb_ids(kb: &Kb, q: &SearchQuery) -> Vec<u64> {
    kb.search(q)
        .unwrap()
        .hits
        .iter()
        .filter_map(|h| h.label.as_deref())
        .map(|l| l[1..].parse().unwrap())
        .collect()
}

fn dqadb_run(d: &Data, work: &Path, ft: &FilteredTruth) -> Report {
    let root = work.join("dqadb");
    let reuse = std::env::var_os("BENCH_REUSE").is_some() && root.join("dqadb.yaml").exists();
    if !reuse {
        let _ = std::fs::remove_dir_all(&root);
        Kb::init(&root, "urn:dqadb:").unwrap();
    }
    let cfg = root.join("dqadb.yaml");
    let y = std::fs::read_to_string(&cfg).unwrap().replace(
        "model: bge-m3\n  dim: 1024",
        &format!("model: cohere-1m-precomputed\n  dim: {DIM}"),
    );
    std::fs::write(&cfg, y).unwrap();
    std::fs::write(root.join("ontology/schema.linkml.yaml"), SCHEMA).unwrap();
    let kb = open_dqadb(&root, d);
    kb.build_ontology(true).unwrap();
    let (mut insert_s, mut index_s) = (0.0, 0.0);
    if !reuse {
        let opts = WriteOpts {
            actor: "agent:bench".into(),
            ..Default::default()
        };

        let t = Instant::now();
        for start in (0..d.n).step_by(20_000) {
            let mut ops = Vec::with_capacity(60_000);
            for id in start..(start + 20_000).min(d.n) {
                let s = format!("_:e{id}");
                ops.push(serde_json::json!({"op":"add","s":s,"p":"rdf:type","o":"ex:Doc"}));
                ops.push(
                    serde_json::json!({"op":"add","s":s,"p":"rdfs:label","o":format!("v{id}")}),
                );
                ops.push(serde_json::json!({"op":"add","s":s,"p":"ex:ord","o":id}));
            }
            let ops: Vec<InputOp> = serde_json::from_value(serde_json::Value::Array(ops)).unwrap();
            kb.assert(&ops, &opts).unwrap();
            eprint!("\r  inserted {}", (start + 20_000).min(d.n));
        }
        eprintln!();
        insert_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let rep = kb.embed(true).unwrap();
        index_s = t.elapsed().as_secs_f64();
        eprintln!(
            "  vectors stored + HNSW in {index_s:.0}s ({:?})",
            rep.compaction.as_ref().map(|c| c.seconds)
        );
    }
    let mut report = Report {
        system: "dqaDB".into(),
        version: format!(
            "dqadb-core {} (HNSW i8 + f16 rerank)",
            env!("CARGO_PKG_VERSION")
        ),
        n: d.n,
        insert_s,
        index_s,
        ..Default::default()
    };

    let measure = |case: &str,
                   param: String,
                   qs: &(dyn Fn(usize) -> SearchQuery + Sync),
                   n_q: usize,
                   truth: &dyn Fn(usize) -> Vec<u64>|
     -> Point {
        let mut lat = Vec::with_capacity(n_q);
        let mut rc = 0.0;
        for i in 0..50.min(n_q) {
            kb.search(&qs(i)).unwrap(); // warm-up (not measured)
        }
        let t = Instant::now();
        for i in 0..n_q {
            let t1 = Instant::now();
            let got = dqadb_ids(&kb, &qs(i));
            lat.push(t1.elapsed());
            rc += recall(&got, &truth(i));
        }
        let qps1 = n_q as f64 / t.elapsed().as_secs_f64();
        // 4 threads, one Kb (own SQLite connection) each, as separate agents would
        let t = Instant::now();
        std::thread::scope(|s| {
            for th in 0..4 {
                let root = &root;
                s.spawn(move || {
                    let kb = open_dqadb(root, d);
                    for i in (th..n_q).step_by(4) {
                        dqadb_ids(&kb, &qs(i));
                    }
                });
            }
        });
        let qps4 = n_q as f64 / t.elapsed().as_secs_f64();
        let (p50, p95, p99) = percentiles(lat);
        let p = Point {
            system: "dqaDB".into(),
            case: case.into(),
            param,
            recall_at_10: rc / n_q as f64,
            p50_ms: p50,
            p95_ms: p95,
            p99_ms: p99,
            qps_1_thread: qps1,
            qps_4_threads: qps4,
        };
        print_point(&p);
        p
    };

    for ef in [32, 64, 128, 256, 512] {
        unsafe { std::env::set_var("DQADB_HNSW_EF", ef.to_string()) };
        let p = measure(
            "search",
            format!("ef={ef}"),
            &|i| SearchQuery {
                semantic: Some(format!("q{i}")),
                limit: K,
                ..Default::default()
            },
            SWEEP_QUERIES,
            &|i| d.truth[i].clone(),
        );
        report.points.push(p);
    }
    unsafe { std::env::remove_var("DQADB_HNSW_EF") };
    for (sel, truth) in &ft.by_selectivity {
        let lim = threshold(d.n, *sel);
        let p = measure(
            "filter",
            format!("keep {:.0}%", sel * 100.0),
            &|i| SearchQuery {
                semantic: Some(format!("q{i}")),
                wheres: vec![format!("ex:ord < {lim}")],
                limit: K,
                ..Default::default()
            },
            FILTER_QUERIES,
            &|i| truth[i].clone(),
        );
        report.points.push(p);
    }
    report.peak_rss_mb = peak_rss_mb();
    drop(kb);
    report.disk_mb = dir_mb(&root); // after close: WAL checkpointed
    report
}

// ---------------- LanceDB ----------------

fn lance_batch(d: &Data, from: usize, to: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("ord", DataType::Int64, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            true,
        ),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from_iter_values((from..to).map(|i| i as u64))),
            Arc::new(Int64Array::from_iter_values((from..to).map(|i| i as i64))),
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                    (from..to).map(|i| {
                        Some(
                            d.train[i * DIM..(i + 1) * DIM]
                                .iter()
                                .map(|x| Some(*x))
                                .collect::<Vec<_>>(),
                        )
                    }),
                    DIM as i32,
                ),
            ),
        ],
    )
    .unwrap()
}

async fn lance_ids(tbl: &lancedb::Table, q: &[f32], param: &str, filter: Option<&str>) -> Vec<u64> {
    let mut vq = tbl
        .vector_search(q)
        .unwrap()
        .distance_type(DistanceType::Cosine)
        .limit(K);
    for part in param.split(',') {
        if let Some((k, v)) = part.split_once('=') {
            let v: usize = v.parse().unwrap();
            vq = match k {
                "ef" => vq.ef(v),
                "nprobes" => vq.nprobes(v).unwrap(),
                "refine" => vq.refine_factor(v as u32),
                _ => vq,
            };
        }
    }
    if let Some(f) = filter {
        vq = vq.only_if(f);
    }
    let batches: Vec<RecordBatch> = vq.execute().await.unwrap().try_collect().await.unwrap();
    let mut out = Vec::new();
    for b in batches {
        let col = b
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        out.extend(col.values().iter().copied());
    }
    out
}

async fn lance_run(d: &Data, work: &Path, ft: &FilteredTruth, hnsw: bool) -> Report {
    let name = if hnsw {
        "LanceDB IVF_HNSW_SQ"
    } else {
        "LanceDB IVF_PQ"
    };
    let path = work.join(if hnsw { "lance-hnsw" } else { "lance-pq" });
    let reuse = std::env::var_os("BENCH_REUSE").is_some() && path.exists();
    if !reuse {
        let _ = std::fs::remove_dir_all(&path);
    }
    let db = lancedb::connect(path.to_str().unwrap())
        .execute()
        .await
        .unwrap();
    let (tbl, insert_s, index_s) = if reuse {
        (db.open_table("t").execute().await.unwrap(), 0.0, 0.0)
    } else {
        let t = Instant::now();
        let batches: Vec<_> = (0..d.n)
            .step_by(50_000)
            .map(|s| Ok(lance_batch(d, s, (s + 50_000).min(d.n))))
            .collect();
        let schema = lance_batch(d, 0, 1).schema();
        let reader = RecordBatchIterator::new(batches, schema);
        let tbl = db
            .create_table(
                "t",
                Box::new(reader) as Box<dyn arrow_array::RecordBatchReader + Send>,
            )
            .execute()
            .await
            .unwrap();
        let insert_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let idx = if hnsw {
            Index::IvfHnswSq(IvfHnswSqIndexBuilder::default().distance_type(DistanceType::Cosine))
        } else {
            Index::IvfPq(IvfPqIndexBuilder::default().distance_type(DistanceType::Cosine))
        };
        tbl.create_index(&["vector"], idx).execute().await.unwrap();
        tbl.create_index(&["ord"], Index::BTree(BTreeIndexBuilder::default()))
            .execute()
            .await
            .unwrap();
        let index_s = t.elapsed().as_secs_f64();
        eprintln!("  {name}: insert {insert_s:.0}s, index {index_s:.0}s");
        (tbl, insert_s, index_s)
    };
    let mut report = Report {
        system: name.into(),
        version: "lancedb 0.41.0 (lance 14.0.2)".into(),
        n: d.n,
        insert_s,
        index_s,
        disk_mb: dir_mb(&path),
        ..Default::default()
    };
    let tbl = Arc::new(tbl);
    let params: Vec<String> = if hnsw {
        [32, 64, 128, 256, 512]
            .iter()
            .map(|v| format!("ef={v}"))
            .collect()
    } else {
        // without re-ranking PQ caps recall; LanceDB recommends refine_factor for that
        let mut p: Vec<String> = [5, 10, 20, 50, 100]
            .iter()
            .map(|v| format!("nprobes={v}"))
            .collect();
        p.extend(
            [10, 20, 50]
                .iter()
                .map(|v| format!("nprobes={v},refine=10")),
        );
        p
    };
    let measure = async |case: &str,
                         param: &str,
                         n_q: usize,
                         filter: Option<String>,
                         truth: &dyn Fn(usize) -> Vec<u64>|
           -> Point {
        for i in 0..50.min(n_q) {
            lance_ids(&tbl, query(d, i), param, filter.as_deref()).await; // warm-up (not measured)
        }
        let mut lat = Vec::with_capacity(n_q);
        let mut rc = 0.0;
        let t = Instant::now();
        for i in 0..n_q {
            let t1 = Instant::now();
            let got = lance_ids(&tbl, query(d, i), param, filter.as_deref()).await;
            lat.push(t1.elapsed());
            rc += recall(&got, &truth(i));
        }
        let qps1 = n_q as f64 / t.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut tasks = Vec::new();
        for th in 0..4 {
            let (tbl, queries, param, filter) = (
                tbl.clone(),
                d.queries.clone(),
                param.to_string(),
                filter.clone(),
            );
            tasks.push(tokio::spawn(async move {
                for i in (th..n_q).step_by(4) {
                    lance_ids(
                        &tbl,
                        &queries[i * DIM..(i + 1) * DIM],
                        &param,
                        filter.as_deref(),
                    )
                    .await;
                }
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }
        let qps4 = n_q as f64 / t.elapsed().as_secs_f64();
        let (p50, p95, p99) = percentiles(lat);
        let p = Point {
            system: name.into(),
            case: case.into(),
            param: param.into(),
            recall_at_10: rc / n_q as f64,
            p50_ms: p50,
            p95_ms: p95,
            p99_ms: p99,
            qps_1_thread: qps1,
            qps_4_threads: qps4,
        };
        print_point(&p);
        p
    };
    for param in &params {
        let p = measure("search", param, SWEEP_QUERIES, None, &|i| {
            d.truth[i].clone()
        })
        .await;
        report.points.push(p);
    }
    // filtered: the default search breadth (what users get without tuning)
    for (sel, truth) in &ft.by_selectivity {
        let lim = threshold(d.n, *sel);
        let p = measure(
            &format!("filter keep {:.0}%", sel * 100.0),
            "default",
            FILTER_QUERIES,
            Some(format!("ord < {lim}")),
            &|i| truth[i].clone(),
        )
        .await;
        report.points.push(p);
    }
    report.peak_rss_mb = peak_rss_mb();
    report
}

#[tokio::main]
async fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (data_dir, work, which) = (PathBuf::from(&a[1]), PathBuf::from(&a[2]), a[3].as_str());
    let max: usize = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(1_000_000);
    std::fs::create_dir_all(&work).unwrap();
    if let Some(cmd) = which.strip_prefix("rb-") {
        // runbook modes: <data> <work> rb-<cmd> <runbook.yaml> <dataset> ...
        let (rb, ds) = (PathBuf::from(&a[4]), a[5].as_str());
        match cmd {
            "gt" => {
                // ... cos|ip <out dir> [comma-separated steps]
                let s = runbook::setup(&data_dir, &rb, ds);
                eprintln!("base {} x {}, {} queries, {} ops", s.base.n, s.base.d, s.queries.len(), s.ops.len());
                let steps: Option<Vec<usize>> = a.get(8).map(|v| v.split(',').map(|x| x.parse().unwrap()).collect());
                runbook::compute_truth(&s, &PathBuf::from(&a[7]), a[6] == "cos", steps.as_deref());
            }
            "compare" => {
                // ... <official dir> <our dir> <steps>
                let steps: Vec<usize> = a[8].split(',').map(|x| x.parse().unwrap()).collect();
                runbook::compare_truth(&PathBuf::from(&a[6]), &PathBuf::from(&a[7]), &steps);
            }
            "dqadb" => {
                // ... <truth dir> <label> [threads]
                let s = runbook::setup(&data_dir, &rb, ds);
                let threads = a.get(8).map(|v| v.parse().unwrap()).unwrap_or(4);
                let rep = tokio::task::block_in_place(|| runbook::run_dqadb(s, &PathBuf::from(&a[6]), &work, &a[7], threads));
                rep.print();
                let out = work.join(format!("runbook-dqadb-{}.json", a[7]));
                std::fs::write(&out, serde_json::to_vec_pretty(&rep).unwrap()).unwrap();
            }
            "qdrant" => {
                // ... <truth dir> <label> [threads] [url]
                let s = runbook::setup(&data_dir, &rb, ds);
                let threads = a.get(8).map(|v| v.parse().unwrap()).unwrap_or(4);
                let url = a.get(9).cloned().unwrap_or_else(|| "http://127.0.0.1:6333".into());
                let rep = tokio::task::block_in_place(|| runbook::run_qdrant(s, &PathBuf::from(&a[6]), &a[7], &url, threads));
                rep.print();
                let out = work.join(format!("runbook-qdrant-{}.json", a[7]));
                std::fs::write(&out, serde_json::to_vec_pretty(&rep).unwrap()).unwrap();
            }
            "diag" => {
                // ... <nq> <ef,ef,...>   (on the state left in <work>/dqadb-runbook)
                let s = runbook::setup(&data_dir, &rb, ds);
                let nq: usize = a[6].parse().unwrap();
                for ef in a[7].split(',') {
                    runbook::diag(&work, &s.queries, nq, ef.parse().unwrap());
                }
            }
            "diag-build" => {
                // ... <nq> conn:add:quant:threads[,...]
                let s = runbook::setup(&data_dir, &rb, ds);
                let nq: usize = a[6].parse().unwrap();
                for cfg in a[7].split(',') {
                    let p: Vec<&str> = cfg.split(':').collect();
                    runbook::diag_build(&work, &s.queries, nq, p[0].parse().unwrap(), p[1].parse().unwrap(), p[2], p[3].parse().unwrap());
                }
            }
            other => panic!("unknown runbook command {other}"),
        }
        return;
    }
    if which.starts_with("stream-") {
        let d = load(&data_dir, stream::MAX_ID);
        let rep = match which {
            "stream-dqadb" => tokio::task::block_in_place(|| stream::dqadb(&d, &work)),
            "stream-lance" => stream::lance(&d, &work).await,
            other => panic!("unknown case {other}"),
        };
        let out = work.join(format!("result-{which}.json"));
        std::fs::write(&out, serde_json::to_vec_pretty(&rep).unwrap()).unwrap();
        eprintln!("wrote {}", out.display());
        return;
    }
    let d = load(&data_dir, max);
    let ft_path = data_dir.join(format!("filtered_truth_{}.json", d.n));
    let ft = if d.train.is_empty() {
        serde_json::from_slice(&std::fs::read(&ft_path).expect("run `gt` first")).unwrap()
    } else {
        filtered_truth(&d, &ft_path)
    };
    if which == "gt" {
        // sanity: brute force keeping 99% should match the official (unfiltered) truth
        let (_, t99) = ft.by_selectivity.iter().find(|(s, _)| *s > 0.98).unwrap();
        let overlap: f64 = (0..FILTER_QUERIES)
            .map(|i| recall(&t99[i], &d.truth[i]))
            .sum::<f64>()
            / FILTER_QUERIES as f64;
        println!(
            "ground truth ready for {} vectors; brute force (99%) vs official neighbors: {:.3}",
            d.n, overlap
        );
        return;
    }
    let report = match which {
        "dqadb" => tokio::task::block_in_place(|| dqadb_run(&d, &work, &ft)),
        "lance-hnsw" => lance_run(&d, &work, &ft, true).await,
        "lance-pq" => lance_run(&d, &work, &ft, false).await,
        other => panic!("unknown system {other}"),
    };
    println!(
        "{}: n={} insert {:.0}s, index {:.0}s, disk {:.0} MB, peak RSS {:.0} MB",
        report.system,
        report.n,
        report.insert_s,
        report.index_s,
        report.disk_mb,
        report.peak_rss_mb
    );
    let out = work.join(format!("result-{which}-{}.json", d.n));
    std::fs::write(&out, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    eprintln!("wrote {}", out.display());
}
