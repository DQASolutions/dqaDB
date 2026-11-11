//! big-ann-benchmarks streaming runbooks (NeurIPS'23 format) for dqaDB.
//!
//! - Runbook YAML: steps `insert {start,end}`, `delete {start,end}` (tags),
//!   `replace {tags_start,tags_end,ids_start,ids_end}`, `search`.
//! - Data: `.bin` files (u32 n, u32 d, f32 rows); ground truth `.gt100` files
//!   (u32 nq, u32 k, i32 ids, f32 dists) named `step{N}.gt100`.
//! - Recall: the official definition (benchmark/plotting/metrics.py,
//!   `compute_recall_with_distance_ties`), averaged over queries, then over search steps.
//!
//! Metric: dqaDB ranks by cosine. When the base vectors are not unit length, the official
//! inner-product ground truth answers a different question, so `gt-cos` computes a cosine
//! ground truth (brute force over the live set at each search step) in the same format;
//! every system is then scored against it. `gt-ip` recomputes the inner-product ground
//! truth for a few steps to validate this harness against the official files.

use super::*;

pub const K: usize = 10;

#[derive(Debug, Clone)]
pub enum Op {
    Insert { start: usize, end: usize },
    Delete { start: usize, end: usize },
    Replace { tags_start: usize, ids_start: usize, n: usize },
    Search,
}

pub fn load_runbook(path: &Path, dataset: &str) -> (usize, Vec<Op>) {
    let y: serde_yaml::Value = serde_yaml::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let r = &y[dataset];
    let max_pts = r["max_pts"].as_u64().unwrap() as usize;
    let mut ops = Vec::new();
    let mut i = 1u64;
    let m = r.as_mapping().expect("runbook mapping");
    while let Some(e) = m.get(serde_yaml::Value::from(i)) {
        let g = |k: &str| e[k].as_u64().unwrap_or_else(|| panic!("step {i}: {k}")) as usize;
        ops.push(match e["operation"].as_str().unwrap() {
            "insert" => Op::Insert { start: g("start"), end: g("end") },
            "delete" => Op::Delete { start: g("start"), end: g("end") },
            "replace" => {
                let n = g("tags_end") - g("tags_start");
                assert_eq!(n, g("ids_end") - g("ids_start"));
                Op::Replace { tags_start: g("tags_start"), ids_start: g("ids_start"), n }
            }
            "search" => Op::Search,
            o => panic!("unknown op {o}"),
        });
        i += 1;
    }
    (max_pts, ops)
}

/// A `.bin` vector file, memory-mapped (the header may claim more rows than a crop holds).
pub struct Vectors {
    map: memmap2::Mmap,
    pub n: usize,
    pub d: usize,
}

impl Vectors {
    pub fn open(path: &Path) -> Vectors {
        let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let map = unsafe { memmap2::Mmap::map(&f).unwrap() };
        let d = u32::from_le_bytes(map[4..8].try_into().unwrap()) as usize;
        let header_n = u32::from_le_bytes(map[0..4].try_into().unwrap()) as usize;
        let n = header_n.min((map.len() - 8) / (4 * d));
        Vectors { map, n, d }
    }
    pub fn row(&self, i: usize) -> &[f32] {
        assert!(i < self.n, "vector {i} beyond the {} available", self.n);
        let b = &self.map[8 + i * self.d * 4..8 + (i + 1) * self.d * 4];
        // the file is 4-byte aligned after the 8-byte header
        unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, self.d) }
    }
    pub fn unit(&self, i: usize) -> Vec<f32> {
        let mut v = self.row(i).to_vec();
        normalize(&mut v);
        v
    }
}

pub struct Truth {
    pub ids: Vec<i32>,
    pub dists: Vec<f32>,
    pub nq: usize,
    pub k: usize,
}

pub fn read_truth(path: &Path) -> Truth {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let nq = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let k = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    assert_eq!(b.len(), 8 + nq * k * 8, "{}", path.display());
    let ids = b[8..8 + nq * k * 4].chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    let dists = b[8 + nq * k * 4..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    Truth { ids, dists, nq, k }
}

fn write_truth(path: &Path, t: &Truth) {
    let mut b = Vec::with_capacity(8 + t.ids.len() * 8);
    b.extend((t.nq as u32).to_le_bytes());
    b.extend((t.k as u32).to_le_bytes());
    t.ids.iter().for_each(|x| b.extend(x.to_le_bytes()));
    t.dists.iter().for_each(|x| b.extend(x.to_le_bytes()));
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, b).unwrap();
    std::fs::rename(tmp, path).unwrap();
}

/// Official recall: ties at the k-th distance count as hits.
pub fn recall_with_ties(t: &Truth, q: usize, got: &[u64]) -> f64 {
    let ids = &t.ids[q * t.k..(q + 1) * t.k];
    let d = &t.dists[q * t.k..(q + 1) * t.k];
    let set_end = if t.k == K {
        K
    } else {
        let anchor = d[K - 1];
        (K..t.k).find(|i| (anchor - d[*i]).abs() >= 1e-6).unwrap_or(t.k)
    };
    let truth: std::collections::HashSet<i64> = ids[..set_end].iter().map(|x| *x as i64).collect();
    let got: std::collections::HashSet<i64> = got.iter().take(K).map(|x| *x as i64).collect();
    got.intersection(&truth).count() as f64 / K as f64
}

/// Which vector each tag currently holds, replaying the runbook up to (and including)
/// each step.
pub struct Live {
    pub tag_vec: Vec<Option<u32>>,
}

impl Live {
    pub fn new(max_tag: usize) -> Live {
        Live { tag_vec: vec![None; max_tag] }
    }
    pub fn apply(&mut self, op: &Op) {
        match *op {
            Op::Insert { start, end } => (start..end).for_each(|i| self.tag_vec[i] = Some(i as u32)),
            Op::Delete { start, end } => (start..end).for_each(|i| self.tag_vec[i] = None),
            Op::Replace { tags_start, ids_start, n } => {
                (0..n).for_each(|j| self.tag_vec[tags_start + j] = Some((ids_start + j) as u32))
            }
            Op::Search => {}
        }
    }
    pub fn live(&self) -> Vec<(u32, u32)> {
        self.tag_vec.iter().enumerate().filter_map(|(t, v)| v.map(|v| (t as u32, v))).collect()
    }
}

fn max_tag(ops: &[Op]) -> usize {
    ops.iter()
        .map(|o| match *o {
            Op::Insert { end, .. } | Op::Delete { end, .. } => end,
            Op::Replace { tags_start, n, .. } => tags_start + n,
            Op::Search => 0,
        })
        .max()
        .unwrap_or(0)
}

/// Brute-force top-K by `score` over the live set for all queries, blocked so each base
/// vector is read once per block of queries.
fn brute_force(base: &Vectors, queries: &[Vec<f32>], live: &[(u32, u32)], unit_base: bool, inv_norm: &[f32]) -> Truth {
    const BLOCK: usize = 64;
    let nq = queries.len();
    let results: Vec<Vec<(f32, i32)>> = queries
        .par_chunks(BLOCK)
        .flat_map_iter(|qs| {
            let mut heaps: Vec<Vec<(f32, i32)>> = vec![Vec::with_capacity(4 * K); qs.len()];
            let mut floor = vec![f32::NEG_INFINITY; qs.len()];
            for (tag, vid) in live {
                let row = base.row(*vid as usize);
                let scale = if unit_base { inv_norm[*vid as usize] } else { 1.0 };
                for (qi, q) in qs.iter().enumerate() {
                    let s = dot(q, row) * scale;
                    if s > floor[qi] {
                        let h = &mut heaps[qi];
                        h.push((s, *tag as i32));
                        if h.len() >= 4 * K {
                            h.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                            h.truncate(K);
                            floor[qi] = h[K - 1].0;
                        }
                    }
                }
            }
            heaps.into_iter().map(|mut h| {
                h.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
                h.truncate(K);
                h
            })
        })
        .collect();
    let mut ids = Vec::with_capacity(nq * K);
    let mut dists = Vec::with_capacity(nq * K);
    for r in results {
        for j in 0..K {
            let (s, id) = r.get(j).copied().unwrap_or((f32::NEG_INFINITY, -1));
            ids.push(id);
            // stored like the official files for inner product: negated similarity
            dists.push(-s);
        }
    }
    Truth { ids, dists, nq, k: K }
}

pub struct Setup {
    pub base: Vectors,
    pub queries: Vec<Vec<f32>>,
    pub ops: Vec<Op>,
    pub max_tag: usize,
}

pub fn setup(data: &Path, runbook: &Path, dataset: &str) -> Setup {
    let base = Vectors::open(&data.join("wikipedia_base.bin"));
    let qv = Vectors::open(&data.join("wikipedia_query.bin"));
    let queries = (0..qv.n).map(|i| qv.row(i).to_vec()).collect();
    let (_max_pts, ops) = load_runbook(runbook, dataset);
    let max_tag = max_tag(&ops);
    Setup { base, queries, ops, max_tag }
}

/// Compute (and cache) ground truth for the runbook's search steps.
/// `cosine`: rank by cosine; otherwise inner product (for validating against official files).
pub fn compute_truth(s: &Setup, out_dir: &Path, cosine: bool, only_steps: Option<&[usize]>) {
    std::fs::create_dir_all(out_dir).unwrap();
    let inv_norm: Vec<f32> = if cosine {
        (0..s.base.n).into_par_iter().map(|i| 1.0 / s.base.row(i).iter().map(|x| x * x).sum::<f32>().sqrt()).collect()
    } else {
        Vec::new()
    };
    let queries: Vec<Vec<f32>> = s
        .queries
        .iter()
        .map(|q| {
            let mut q = q.clone();
            if cosine {
                normalize(&mut q);
            }
            q
        })
        .collect();
    let mut live = Live::new(s.max_tag);
    for (i, op) in s.ops.iter().enumerate() {
        live.apply(op);
        let step = i + 1;
        if !matches!(op, Op::Search) || only_steps.is_some_and(|o| !o.contains(&step)) {
            continue;
        }
        let path = out_dir.join(format!("step{step}.gt100"));
        if path.exists() {
            continue;
        }
        let t = Instant::now();
        let set = live.live();
        let truth = brute_force(&s.base, &queries, &set, cosine, &inv_norm);
        write_truth(&path, &truth);
        eprintln!("  gt step {step}: {} live, {:.0}s", set.len(), t.elapsed().as_secs_f64());
    }
}

/// Compare two ground-truth directories on the given steps (fraction of official top-10
/// found in ours, with ties).
pub fn compare_truth(official: &Path, ours: &Path, steps: &[usize]) {
    for step in steps {
        let o = read_truth(&official.join(format!("step{step}.gt100")));
        let m = read_truth(&ours.join(format!("step{step}.gt100")));
        let r: f64 = (0..o.nq)
            .map(|q| recall_with_ties(&o, q, &m.ids[q * m.k..q * m.k + K].iter().map(|x| *x as u64).collect::<Vec<_>>()))
            .sum::<f64>()
            / o.nq as f64;
        println!("step {step}: our brute force vs official ground truth = {r:.4}");
    }
}

// ---------------- dqaDB under a runbook ----------------

/// Entity text `v<id>` -> base vector `id` (unit length), `q<i>` -> query `i`.
struct RunbookEmbedder {
    base: Arc<Vectors>,
    queries: Arc<Vec<Vec<f32>>>,
}

impl Embedder for RunbookEmbedder {
    fn id(&self) -> &str {
        "runbook-precomputed"
    }
    fn dim(&self) -> usize {
        self.base.d
    }
    fn embed(&self, texts: &[String]) -> DqResult<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| match t.split_at(1) {
                ("v", n) => self.base.unit(n.parse().unwrap()),
                ("q", n) => {
                    let mut q = self.queries[n.parse::<usize>().unwrap()].clone();
                    normalize(&mut q);
                    q
                }
                _ => panic!("unexpected text {t}"),
            })
            .collect())
    }
}

const RB_SCHEMA: &str = "id: https://example.org/bench\nname: bench\nprefixes:\n  ex: https://example.org/bench#\n  linkml: https://w3id.org/linkml/\ndefault_prefix: ex\ndefault_range: string\nimports:\n  - linkml:types\nclasses:\n  Doc:\n    slots: [label]\nslots:\n  label:\n    slot_uri: rdfs:label\n    required: true\n";

#[derive(Serialize, Default, Clone)]
pub struct StepResult {
    pub step: usize,
    pub live: usize,
    /// (search parameter, recall@10, QPS)
    pub by_param: Vec<(String, f64, f64)>,
}

#[derive(Serialize, Default)]
pub struct RunbookReport {
    pub system: String,
    pub runbook: String,
    pub truth: String,
    /// per search parameter: (param, mean recall@10 over steps, min step recall, mean QPS)
    pub summary: Vec<(String, f64, f64, f64)>,
    pub steps: Vec<StepResult>,
    pub write_s: f64,
    pub maintain_s: f64,
    pub search_s: f64,
    pub total_s: f64,
    pub disk_mb: f64,
    pub peak_rss_mb: f64,
    pub notes: Vec<String>,
}

impl RunbookReport {
    fn finish(&mut self) {
        let params: Vec<String> = self.steps.first().map(|s| s.by_param.iter().map(|p| p.0.clone()).collect()).unwrap_or_default();
        let n = self.steps.len().max(1) as f64;
        self.summary = params
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let r: Vec<f64> = self.steps.iter().map(|s| s.by_param[i].1).collect();
                let q: f64 = self.steps.iter().map(|s| s.by_param[i].2).sum::<f64>() / n;
                (p.clone(), r.iter().sum::<f64>() / n, r.iter().copied().fold(1.0, f64::min), q)
            })
            .collect();
    }

    pub fn print(&self) {
        println!("{} on {}: write {:.0}s, maintain {:.0}s, search {:.0}s, total {:.0}s, disk {:.0} MB, peak RSS {:.0} MB", self.system, self.runbook, self.write_s, self.maintain_s, self.search_s, self.total_s, self.disk_mb, self.peak_rss_mb);
        for (p, r, m, q) in &self.summary {
            println!("  {p:<14} mean recall@10 {r:.4}  min step {m:.4}  mean QPS {q:.0}");
        }
    }
}

/// Run every query of a search step once per parameter, on `threads` threads.
fn sweep(
    params: &[String],
    nq: usize,
    threads: usize,
    truth: &Truth,
    before: &dyn Fn(&str),
    search: &(dyn Fn(&str, usize, usize) -> Vec<u64> + Sync),
) -> (Vec<(String, f64, f64)>, f64) {
    let mut out = Vec::new();
    let mut total = 0.0;
    for p in params {
        before(p);
        let t = Instant::now();
        let results: Vec<(usize, Vec<u64>)> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads)
                .map(|th| sc.spawn(move || (th..nq).step_by(threads).map(|q| (q, search(p, th, q))).collect::<Vec<_>>()))
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        let secs = t.elapsed().as_secs_f64();
        total += secs;
        let r = results.iter().map(|(q, got)| recall_with_ties(truth, *q, got)).sum::<f64>() / nq as f64;
        out.push((p.clone(), r, nq as f64 / secs));
    }
    (out, total)
}

fn log_step(step: usize, live: usize, by: &[(String, f64, f64)]) {
    let parts: Vec<String> = by.iter().map(|(p, r, q)| format!("{p}: {r:.4} @ {q:.0} QPS")).collect();
    eprintln!("  step {step:>3}: live {live:>7}  {}", parts.join("  |  "));
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn copy_dir(from: &Path, to: &Path) {
    let _ = std::fs::remove_dir_all(to);
    if let Some(parent) = to.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let ok = std::process::Command::new("cp").arg("-a").arg(from).arg(to).status().unwrap().success();
    assert!(ok, "copy {} -> {}", from.display(), to.display());
}

/// Development knobs (env), so iterations do not need the full 3-hour run:
/// - RB_QUERIES: queries per search step (default all 5000; 1000 gives recall +-0.0005)
/// - RB_EVERY: measure every Nth search step only (default 1)
/// - RB_EFS: search widths to sweep (default 64,128,256)
/// - RB_CKPT: runbook steps after which to save a checkpoint (work/ckpt-<label>-<step>)
/// - RB_RESUME: checkpoint dir to continue from (skips the steps it covers)
/// - RB_STOP: stop after this runbook step
pub fn run_dqadb(s: Setup, truth_dir: &Path, work: &Path, label: &str, threads: usize) -> RunbookReport {
    let root = work.join("dqadb-runbook");
    let nq_used = env_usize("RB_QUERIES", s.queries.len()).min(s.queries.len());
    let every = env_usize("RB_EVERY", 1).max(1);
    let stop = env_usize("RB_STOP", usize::MAX);
    let efs: Vec<String> = std::env::var("RB_EFS").unwrap_or_else(|_| "64,128,256".into()).split(',').map(|e| format!("ef={e}")).collect();
    let ckpt: Vec<usize> = std::env::var("RB_CKPT").ok().map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
    let mut live = Live::new(s.max_tag);
    let mut first = 1usize;
    if let Ok(from) = std::env::var("RB_RESUME") {
        let from = PathBuf::from(from);
        copy_dir(&from.join("kb"), &root);
        live.tag_vec = serde_json::from_slice(&std::fs::read(from.join("live.json")).unwrap()).unwrap();
        first = std::fs::read_to_string(from.join("step")).unwrap().trim().parse::<usize>().unwrap() + 1;
        eprintln!("resumed from {} at step {first}", from.display());
    } else {
        let _ = std::fs::remove_dir_all(&root);
        Kb::init(&root, "urn:dqadb:").unwrap();
        let cfg = root.join("dqadb.yaml");
        let y = std::fs::read_to_string(&cfg)
            .unwrap()
            .replace("model: bge-m3\n  dim: 1024", &format!("model: runbook-precomputed\n  dim: {}", s.base.d));
        std::fs::write(&cfg, y).unwrap();
        std::fs::write(root.join("ontology/schema.linkml.yaml"), RB_SCHEMA).unwrap();
    }
    let base = Arc::new(s.base);
    let queries = Arc::new(s.queries);
    let open = || {
        let kb = Kb::open(&root).unwrap();
        kb.set_embedder(Box::new(RunbookEmbedder { base: base.clone(), queries: queries.clone() })).unwrap();
        kb
    };
    let kb = open();
    if first == 1 {
        kb.build_ontology(true).unwrap();
    }
    let opts = WriteOpts { actor: "agent:bench".into(), ..Default::default() };
    let mut rep = RunbookReport { system: "dqaDB".into(), runbook: label.into(), ..Default::default() };
    let started = Instant::now();
    let readers: Vec<Kb> = (0..threads).map(|_| open()).collect();
    let nq = queries.len();
    let mut searches = 0usize;
    for (i, op) in s.ops.iter().enumerate() {
        let step = i + 1;
        if step < first {
            continue;
        }
        if step > stop {
            break;
        }
        if ckpt.contains(&(step - 1)) && step - 1 >= first {
            let dir = work.join(format!("ckpt-{label}-{}", step - 1));
            copy_dir(&root, &dir.join("kb"));
            std::fs::write(dir.join("live.json"), serde_json::to_vec(&live.tag_vec).unwrap()).unwrap();
            std::fs::write(dir.join("step"), (step - 1).to_string()).unwrap();
            eprintln!("  checkpoint {}", dir.display());
        }
        let t = Instant::now();
        let mut ops: Vec<serde_json::Value> = Vec::new();
        let mut ops_retract: Vec<serde_json::Value> = Vec::new();
        let ent = |tag: usize| format!("dqadb:t{tag}");
        match *op {
            Op::Insert { start, end } => {
                for tag in start..end {
                    ops.push(serde_json::json!({"op":"add","s":ent(tag),"p":"rdf:type","o":"ex:Doc"}));
                    ops.push(serde_json::json!({"op":"add","s":ent(tag),"p":"rdfs:label","o":format!("v{tag}")}));
                }
            }
            Op::Delete { start, end } => {
                for tag in start..end {
                    if let Some(v) = live.tag_vec[tag] {
                        ops_retract.push(serde_json::json!({"op":"retract","s":ent(tag),"p":"rdf:type","o":"ex:Doc"}));
                        ops_retract.push(serde_json::json!({"op":"retract","s":ent(tag),"p":"rdfs:label","o":format!("v{v}")}));
                    }
                }
            }
            Op::Replace { tags_start, ids_start, n } => {
                for j in 0..n {
                    let tag = tags_start + j;
                    if let Some(v) = live.tag_vec[tag] {
                        ops_retract.push(serde_json::json!({"op":"retract","s":ent(tag),"p":"rdfs:label","o":format!("v{v}")}));
                    }
                    ops_retract.push(serde_json::json!({"op":"add","s":ent(tag),"p":"rdfs:label","o":format!("v{}", ids_start + j)}));
                }
            }
            Op::Search => {}
        }
        live.apply(op);
        if !matches!(op, Op::Search) {
            // one event per runbook batch, at most 50k ops each (events stay readable)
            for chunk in ops.chunks(50_000).chain(ops_retract.chunks(50_000)) {
                let list: Vec<InputOp> = serde_json::from_value(serde_json::Value::Array(chunk.to_vec())).unwrap();
                kb.assert(&list, &opts).unwrap();
            }
            kb.embed_sync().unwrap();
            rep.write_s += t.elapsed().as_secs_f64();
            let t = Instant::now();
            kb.embed(false).unwrap();
            rep.maintain_s += t.elapsed().as_secs_f64();
            continue;
        }
        searches += 1;
        if (searches - 1) % every != 0 {
            continue;
        }
        let truth = read_truth(&truth_dir.join(format!("step{step}.gt100")));
        assert_eq!(truth.nq, nq);
        let params = efs.clone();
        let (by, secs) = sweep(
            &params,
            nq_used,
            threads,
            &truth,
            &|p| unsafe { std::env::set_var("DQADB_HNSW_EF", &p[3..]) },
            &|_, th, q| {
                let page = readers[th].search(&SearchQuery { semantic: Some(format!("q{q}")), limit: K, ..Default::default() }).unwrap();
                page.hits.iter().map(|h| h.iri.rsplit(":t").next().unwrap().parse::<u64>().unwrap()).collect()
            },
        );
        rep.search_s += secs;
        let n_live = live.tag_vec.iter().filter(|v| v.is_some()).count();
        log_step(step, n_live, &by);
        if std::env::var("RB_SEGLOG").is_ok() {
            // segment state behind this step's numbers: (nodes, deleted%, deletion list?)
            let vdir = dqadb_core::ann::dir(&root.join(".dqadb"));
            if let Some(m) = dqadb_core::ann::load_manifest(&vdir) {
                let store = dqadb_core::store::Store::open(&root.join(".dqadb/index.sqlite")).unwrap();
                let delta = store.read().unwrap().vec_changed_count(m.seq).unwrap();
                let segs: Vec<String> = m.segments.iter().map(|s| format!("{}k/{:.0}%{}", s.count / 1000, 100.0 * s.deleted as f64 / s.count as f64, if s.del.is_some() { "L" } else { "" })).collect();
                eprintln!("        segments {} [{}] delta {delta}", m.segments.len(), segs.join(" "));
            }
        }
        rep.steps.push(StepResult { step, live: n_live, by_param: by });
    }
    unsafe { std::env::remove_var("DQADB_HNSW_EF") };
    rep.total_s = started.elapsed().as_secs_f64();
    rep.finish();
    drop(readers);
    drop(kb);
    rep.disk_mb = dir_mb(&root);
    rep.peak_rss_mb = peak_rss_mb();
    rep.truth = truth_dir.display().to_string();
    rep
}

// ---------------- Qdrant (REST, default configuration) ----------------

pub fn run_qdrant(s: Setup, truth_dir: &Path, label: &str, url: &str, threads: usize) -> RunbookReport {
    let coll = format!("{url}/collections/bench");
    let agent = || -> ureq::Agent { ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(600))).build().into() };
    let a = agent();
    let _ = a.delete(&coll).call();
    a.put(&coll).send_json(serde_json::json!({"vectors": {"size": s.base.d, "distance": "Cosine"}})).unwrap();
    let info = |a: &ureq::Agent| -> serde_json::Value { a.get(&coll).call().unwrap().body_mut().read_json::<serde_json::Value>().unwrap()["result"].clone() };
    let mut rep = RunbookReport { system: "Qdrant 1.20.0 (defaults)".into(), runbook: label.into(), truth: truth_dir.display().to_string(), ..Default::default() };
    let mut live = Live::new(s.max_tag);
    let nq = s.queries.len();
    let agents: Vec<ureq::Agent> = (0..threads).map(|_| agent()).collect();
    let started = Instant::now();
    let upsert = |a: &ureq::Agent, pts: Vec<(usize, u32)>| {
        for chunk in pts.chunks(1000) {
            let points: Vec<serde_json::Value> = chunk.iter().map(|(tag, vid)| serde_json::json!({"id": tag, "vector": s.base.row(*vid as usize)})).collect();
            a.put(&format!("{coll}/points?wait=true")).send_json(serde_json::json!({"points": points})).unwrap();
        }
    };
    for (i, op) in s.ops.iter().enumerate() {
        let step = i + 1;
        let t = Instant::now();
        match *op {
            Op::Insert { start, end } => upsert(&a, (start..end).map(|t| (t, t as u32)).collect()),
            Op::Replace { tags_start, ids_start, n } => upsert(&a, (0..n).map(|j| (tags_start + j, (ids_start + j) as u32)).collect()),
            Op::Delete { start, end } => {
                let ids: Vec<usize> = (start..end).filter(|t| live.tag_vec[*t].is_some()).collect();
                for chunk in ids.chunks(10_000) {
                    a.post(&format!("{coll}/points/delete?wait=true")).send_json(serde_json::json!({"points": chunk})).unwrap();
                }
            }
            Op::Search => {}
        }
        live.apply(op);
        if !matches!(op, Op::Search) {
            rep.write_s += t.elapsed().as_secs_f64();
            continue;
        }
        let truth = read_truth(&truth_dir.join(format!("step{step}.gt100")));
        let params: Vec<String> = [64, 128, 256].iter().map(|e| format!("hnsw_ef={e}")).collect();
        let (by, secs) = sweep(&params, nq, threads, &truth, &|_| {}, &|p, th, q| {
            let ef: usize = p[8..].parse().unwrap();
            let body = serde_json::json!({"query": s.queries[q], "limit": K, "params": {"hnsw_ef": ef}});
            let v: serde_json::Value = agents[th].post(&format!("{coll}/points/query")).send_json(body).unwrap().body_mut().read_json().unwrap();
            v["result"]["points"].as_array().unwrap().iter().map(|p| p["id"].as_u64().unwrap()).collect()
        });
        rep.search_s += secs;
        let n_live = live.tag_vec.iter().filter(|v| v.is_some()).count();
        let inf = info(&a);
        let note = format!("step {step}: segments {}, indexed {}, points {}, optimizer {}", inf["segments_count"], inf["indexed_vectors_count"], inf["points_count"], inf["optimizer_status"]);
        eprintln!("    {note}");
        rep.notes.push(note);
        log_step(step, n_live, &by);
        rep.steps.push(StepResult { step, live: n_live, by_param: by });
    }
    rep.total_s = started.elapsed().as_secs_f64();
    rep.finish();
    rep
}

// ---------------- diagnosis of a stopped dqaDB runbook state ----------------

/// For `nq` queries: exact top-10 over the stored vectors vs `ann::search`, and where each
/// missed neighbour lives (live in a segment, deleted there, in the delta, nowhere).
pub fn diag(work: &Path, queries: &[Vec<f32>], nq: usize, ef: usize) {
    use dqadb_core::store::Store;
    use std::collections::{HashMap, HashSet};
    let root = work.join("dqadb-runbook/.dqadb");
    let dir = dqadb_core::ann::dir(&root);
    let store = Store::open(&root.join("index.sqlite")).unwrap();
    let r = store.read().unwrap();
    let mut all: Vec<(u64, Vec<f32>)> = Vec::new();
    r.vec_each(|id, b| all.push((id, dqadb_core::embed::from_f16_bytes(b)))).unwrap();
    let m = dqadb_core::ann::usable(&dir, &r, r.vec_state().unwrap().as_ref()).unwrap().expect("usable segments");
    let delta: HashSet<u64> = {
        let mut d = HashSet::new();
        r.vec_since(m.seq, |id, _| { d.insert(id); }).unwrap();
        d
    };
    let read = |p: std::path::PathBuf| -> Vec<u64> {
        std::fs::read(p).unwrap().chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect()
    };
    let mut segs: Vec<(String, HashSet<u64>, HashSet<u64>)> = Vec::new();
    for s in &m.segments {
        let ids: HashSet<u64> = read(dir.join(format!("{}.ids", s.name))).into_iter().collect();
        let del: HashSet<u64> = s.del.as_ref().map(|d| read(dir.join(d)).into_iter().collect()).unwrap_or_default();
        segs.push((s.name.clone(), ids, del));
    }
    let live: HashSet<u64> = all.iter().map(|x| x.0).collect();
    eprintln!("vectors {}, delta {}, segments {:?}", all.len(), delta.len(),
        m.segments.iter().map(|s| (s.count, s.deleted)).collect::<Vec<_>>());
    // consistency: every live id is live in exactly one segment or in the delta
    let mut where_live: HashMap<u64, usize> = HashMap::new();
    for (_, ids, del) in &segs { for id in ids { if !del.contains(id) { *where_live.entry(*id).or_default() += 1; } } }
    let missing = live.iter().filter(|id| !delta.contains(id) && !where_live.contains_key(id)).count();
    let doubled = where_live.values().filter(|c| **c > 1).count();
    let dead_live = where_live.keys().filter(|id| !live.contains(id)).count();
    eprintln!("live ids in no segment/delta: {missing}; live in >1 segment: {doubled}; segment-live but not in store: {dead_live}");
    unsafe { std::env::set_var("DQADB_HNSW_EF", ef.to_string()) };
    let (mut hit, mut tot, mut rhit) = (0usize, 0usize, 0usize);
    let mut maxerr = 0f32;
    let byid: HashMap<u64, Vec<f32>> = all.iter().cloned().collect();
    let mut why: HashMap<String, usize> = HashMap::new();
    for q in queries.iter().take(nq) {
        let norm = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let qv: Vec<f32> = q.iter().map(|x| x / norm).collect();
        let mut ex: Vec<(f32, u64)> = all.iter().map(|(id, v)| (v.iter().zip(&qv).map(|(a, b)| a * b).sum::<f32>(), *id)).collect();
        ex.select_nth_unstable_by(10, |a, b| b.0.partial_cmp(&a.0).unwrap());
        ex.truncate(10);
        let (got, _) = dqadb_core::ann::search(&dir, &m, &r, &qv, 10, None).unwrap();
        let got: HashSet<u64> = got.into_iter().map(|x| x.0).collect();
        // same candidates (the whole beam), re-scored exactly with the stored vectors
        let (beam, _) = dqadb_core::ann::search(&dir, &m, &r, &qv, ef, None).unwrap();
        let mut resc: Vec<(f32, u64)> = beam.iter().map(|(id, _)| (byid[id].iter().zip(&qv).map(|(a, b)| a * b).sum::<f32>(), *id)).collect();
        resc.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (_, id) in &ex { if resc.iter().take(10).any(|x| x.1 == *id) { rhit += 1; } }
        // score gap at the top-10 boundary, and usearch vs exact score difference
        let mut s = ex.clone(); s.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (id, sc) in beam.iter().take(10) { maxerr = maxerr.max((sc - byid[id].iter().zip(&qv).map(|(a, b)| a * b).sum::<f32>()).abs()); }
        let _ = s;
        for (_, id) in &ex {
            tot += 1;
            if got.contains(id) { hit += 1; continue; }
            let place = if delta.contains(id) { "delta".to_string() } else {
                segs.iter().find(|(_, ids, del)| ids.contains(id) && !del.contains(id))
                    .map(|(n, ids, del)| format!("live in {n} ({} nodes, {} del)", ids.len(), del.len()))
                    .unwrap_or_else(|| "nowhere".into())
            };
            *why.entry(place).or_default() += 1;
        }
    }
    eprintln!("ef {ef}: recall {:.4} over {nq} queries; misses by place: {why:?}", hit as f64 / tot as f64);
    eprintln!("ef {ef}: beam re-scored exactly: recall {:.4}; max |usearch score - exact| {maxerr:.5}", rhit as f64 / tot as f64);
}

/// Graph quality of a fresh usearch index over the stored vectors, per build setting.
pub fn diag_build(work: &Path, queries: &[Vec<f32>], nq: usize, conn: usize, add: usize, quant: &str, threads: usize) {
    use dqadb_core::store::Store;
    use usearch::{Index, IndexOptions, MetricKind, ScalarKind};
    let root = work.join("dqadb-runbook/.dqadb");
    let store = Store::open(&root.join("index.sqlite")).unwrap();
    let r = store.read().unwrap();
    let mut all: Vec<(u64, Vec<f32>)> = Vec::new();
    r.vec_each(|id, b| all.push((id, dqadb_core::embed::from_f16_bytes(b)))).unwrap();
    let dim = all[0].1.len();
    let index = Index::new(&IndexOptions {
        dimensions: dim, metric: MetricKind::Cos,
        quantization: match quant { "f32" => ScalarKind::F32, "i8" => ScalarKind::I8, _ => ScalarKind::F16 },
        connectivity: conn, expansion_add: add, ..Default::default()
    }).unwrap();
    index.reserve_capacity_and_threads(all.len(), threads.max(1)).unwrap();
    let t = Instant::now();
    std::thread::scope(|s| {
        for th in 0..threads {
            let (index, all) = (&index, &all);
            s.spawn(move || for (id, v) in all.iter().skip(th).step_by(threads) { index.add(*id, v).unwrap(); });
        }
    });
    let build = t.elapsed().as_secs_f64();
    let qs: Vec<Vec<f32>> = queries.iter().take(nq).map(|q| { let n = q.iter().map(|x| x * x).sum::<f32>().sqrt(); q.iter().map(|x| x / n).collect() }).collect();
    let exact: Vec<Vec<u64>> = qs.iter().map(|qv| {
        let mut ex: Vec<(f32, u64)> = all.iter().map(|(id, v)| (v.iter().zip(qv).map(|(a, b)| a * b).sum::<f32>(), *id)).collect();
        ex.select_nth_unstable_by(10, |a, b| b.0.partial_cmp(&a.0).unwrap());
        ex.truncate(10); ex.into_iter().map(|x| x.1).collect()
    }).collect();
    let mut out = format!("conn {conn} add {add} {quant} threads {threads}: build {build:.0}s");
    for ef in [64usize, 128, 256, 512] {
        index.change_expansion_search(ef);
        let mut hit = 0;
        for (qv, ex) in qs.iter().zip(&exact) {
            let m = index.search(qv, 10).unwrap();
            hit += ex.iter().filter(|id| m.keys.contains(id)).count();
        }
        out += &format!("  ef{ef} {:.4}", hit as f64 / (10 * qs.len()) as f64);
    }
    eprintln!("{out}");
}

/// Graph quality when a segment grows by appending into an existing graph (the merge
/// strategy), against a fresh single-threaded build of the same live set. Variants:
/// `fresh`; `append-clean` (base without deletions: the Lucene/Qdrant rule); `append-tomb`
/// (base with 1/7 tombstoned + 1/10 re-added keys). Env DIAG_N limits the vectors.
pub fn diag_append(work: &Path, queries: &[Vec<f32>], nq: usize) {
    use dqadb_core::store::Store;
    use usearch::{Index, IndexOptions, MetricKind, ScalarKind};
    let root = work.join("dqadb-runbook/.dqadb");
    let store = Store::open(&root.join("index.sqlite")).unwrap();
    let r = store.read().unwrap();
    let mut all: Vec<(u64, Vec<f32>)> = Vec::new();
    r.vec_each(|id, b| all.push((id, dqadb_core::embed::from_f16_bytes(b)))).unwrap();
    all.truncate(env_usize("DIAG_N", all.len()));
    let dim = all[0].1.len();
    let opts = IndexOptions { dimensions: dim, metric: MetricKind::Cos, quantization: ScalarKind::F16, connectivity: 16, expansion_add: 128, ..Default::default() };
    let half = all.len() / 2;
    let gone: std::collections::HashSet<u64> = all[..half].iter().enumerate().filter(|(i, _)| i % 7 == 0).map(|(_, (id, _))| *id).collect();
    let qs: Vec<Vec<f32>> = queries.iter().take(nq).map(|q| { let n = q.iter().map(|x| x * x).sum::<f32>().sqrt(); q.iter().map(|x| x / n).collect() }).collect();
    let truth = |skip: &std::collections::HashSet<u64>| -> Vec<Vec<u64>> {
        qs.iter().map(|qv| {
            let mut ex: Vec<(f32, u64)> = all.iter().filter(|(id, _)| !skip.contains(id)).map(|(id, v)| (v.iter().zip(qv).map(|(a, b)| a * b).sum::<f32>(), *id)).collect();
            ex.select_nth_unstable_by(10, |a, b| b.0.partial_cmp(&a.0).unwrap());
            ex.truncate(10); ex.into_iter().map(|x| x.1).collect()
        }).collect()
    };
    let none = std::collections::HashSet::new();
    let (t_all, t_live) = (truth(&none), truth(&gone));
    let f = work.join("append-test.usearch");
    let eval = |name: &str, idx: &Index, secs: f64, exact: &[Vec<u64>]| {
        let mut out = format!("{name:13} {secs:4.0}s");
        let mut leaked = 0;
        for ef in [64usize, 128, 256, 512] {
            idx.change_expansion_search(ef);
            let mut hit = 0;
            for (qv, ex) in qs.iter().zip(exact) {
                let m = idx.search(qv, 10).unwrap();
                leaked += m.keys.iter().filter(|k| gone.contains(k) && exact.as_ptr() == t_live.as_ptr()).count();
                hit += ex.iter().filter(|id| m.keys.contains(id)).count();
            }
            out += &format!("  ef{ef} {:.4}", hit as f64 / (10 * qs.len()) as f64);
        }
        eprintln!("{out}  removed returned {leaked}  (n {})", all.len());
    };
    // fresh build of the live set
    let t = Instant::now();
    let a = Index::new(&opts).unwrap();
    a.reserve_capacity_and_threads(all.len(), 1).unwrap();
    for (id, v) in all.iter().filter(|(id, _)| !gone.contains(id)) { a.add(*id, v).unwrap(); }
    eval("fresh", &a, t.elapsed().as_secs_f64(), &t_live);
    drop(a);
    // base = first half, saved
    let a = Index::new(&opts).unwrap();
    a.reserve_capacity_and_threads(half, 1).unwrap();
    for (id, v) in &all[..half] { a.add(*id, v).unwrap(); }
    a.save(f.to_str().unwrap()).unwrap();
    drop(a);
    for tomb in [false, true] {
        let t = Instant::now();
        let b = Index::new(&opts).unwrap();
        b.load(f.to_str().unwrap()).unwrap();
        b.reserve_capacity_and_threads(all.len() + half / 10, 1).unwrap();
        if tomb {
            for (i, (id, v)) in all[..half].iter().enumerate() {
                if i % 7 == 0 { b.remove(*id).unwrap(); }
                else if i % 10 == 0 { b.remove(*id).unwrap(); b.add(*id, v).unwrap(); }
            }
        }
        for (id, v) in &all[half..] { b.add(*id, v).unwrap(); }
        let secs = t.elapsed().as_secs_f64();
        let g = work.join("append-test2.usearch");
        b.save(g.to_str().unwrap()).unwrap();
        drop(b);
        let c = Index::new(&opts).unwrap();
        c.view(g.to_str().unwrap()).unwrap();
        if tomb { eval("append-tomb", &c, secs, &t_live) } else { eval("append-clean", &c, secs, &t_all) }
        drop(c);
        let _ = std::fs::remove_file(&g);
    }
    let _ = std::fs::remove_file(&f);
}

/// Cost of the in-graph filter: plain search vs `filtered_search` with a std HashSet
/// (SipHash) vs a multiplicative-hash set, `stale` random live ids, on the current segments.
pub fn diag_filter(work: &Path, queries: &[Vec<f32>], nq: usize, stale: usize) {
    use std::collections::HashSet;
    use usearch::{Index, IndexOptions, MetricKind, ScalarKind};
    #[derive(Default, Clone, Copy)]
    struct Mul(u64);
    impl std::hash::Hasher for Mul {
        fn finish(&self) -> u64 { self.0 }
        fn write(&mut self, _: &[u8]) { unreachable!() }
        fn write_u64(&mut self, x: u64) { let m = (x as u128) * 0x9E37_79B9_7F4A_7C15u128; self.0 = (m as u64) ^ ((m >> 64) as u64); }
    }
    type Fast = HashSet<u64, std::hash::BuildHasherDefault<Mul>>;
    let root = work.join("dqadb-runbook/.dqadb");
    let dir = dqadb_core::ann::dir(&root);
    let m = dqadb_core::ann::load_manifest(&dir).unwrap();
    let qs: Vec<Vec<f32>> = queries.iter().take(nq).map(|q| { let n = q.iter().map(|x| x * x).sum::<f32>().sqrt(); q.iter().map(|x| x / n).collect() }).collect();
    let mut ids: Vec<u64> = Vec::new();
    let segs: Vec<Index> = m.segments.iter().map(|s| {
        let i = Index::new(&IndexOptions { dimensions: m.dim, metric: MetricKind::Cos, quantization: ScalarKind::F16, connectivity: 16, expansion_add: 128, ..Default::default() }).unwrap();
        i.view(dir.join(format!("{}.usearch", s.name)).to_str().unwrap()).unwrap();
        i.change_expansion_search(128);
        let raw = std::fs::read(dir.join(format!("{}.ids", s.name))).unwrap();
        ids.extend(raw.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())));
        i
    }).collect();
    let pick: Vec<u64> = ids.iter().copied().step_by((ids.len() / stale.max(1)).max(1)).take(stale).collect();
    let sip: HashSet<u64> = pick.iter().copied().collect();
    let fast: Fast = pick.iter().copied().collect();
    let run = |name: &str, f: &dyn Fn(&Index, &[f32]) -> usize| {
        let t = Instant::now();
        let mut n = 0;
        for q in &qs { for s in &segs { n += f(s, q); } }
        eprintln!("{name:22} {:.2} ms/query  ({} segments, {n} hits)", t.elapsed().as_secs_f64() * 1e3 / qs.len() as f64, segs.len());
    };
    for _ in 0..2 {
        run("plain", &|s, q| s.search(q, 128).unwrap().keys.len());
        run("filtered SipHash", &|s, q| s.filtered_search(q, 128, |k| !sip.contains(&k)).unwrap().keys.len());
        run("filtered mul-hash", &|s, q| s.filtered_search(q, 128, |k| !fast.contains(&k)).unwrap().keys.len());
        run("filtered always-true", &|s, q| s.filtered_search(q, 128, |_| true).unwrap().keys.len());
    }
}

/// Cost of the exact delta scan per query: decode f16 + scalar dot (old) vs cached f32
/// with the SIMD-friendly dot (new), `n` vectors of `dim`.
pub fn diag_dot(n: usize, dim: usize) {
    let mut seed = 1u64;
    let mut rnd = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; (seed % 2000) as f32 / 1000.0 - 1.0 };
    let rows: Vec<Vec<f32>> = (0..n).map(|_| (0..dim).map(|_| rnd()).collect()).collect();
    let blobs: Vec<Vec<u8>> = rows.iter().map(|v| v.iter().flat_map(|x| half::f16::from_f32(*x).to_le_bytes()).collect()).collect();
    let q: Vec<f32> = (0..dim).map(|_| rnd()).collect();
    let reps = 200;
    let t = Instant::now();
    let mut s = 0f32;
    for _ in 0..reps { for b in &blobs { let v = dqadb_core::embed::from_f16_bytes(b); s += v.iter().zip(&q).map(|(x, y)| x * y).sum::<f32>(); } }
    eprintln!("old: decode f16 + scalar dot  {:.3} ms/query  ({s:.0})", t.elapsed().as_secs_f64() * 1e3 / reps as f64);
    let t = Instant::now();
    let mut s = 0f32;
    for _ in 0..reps { for v in &rows { s += dqadb_core::embed::dot(&q, v); } }
    eprintln!("new: cached f32 + SIMD dot    {:.3} ms/query  ({s:.0})", t.elapsed().as_secs_f64() * 1e3 / reps as f64);
}
