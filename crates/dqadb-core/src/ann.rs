//! Approximate nearest neighbours: an HNSW "base" (usearch, f16, memory-mapped) over the
//! vectors up to some change sequence, plus an exact scan of the "delta" changed since.
//! (Spec §8/§11, decision 2026-11-09.)
//!
//! - Each base file is immutable; readers memory-map it. Producing a new one never blocks
//!   writers or readers: it reads a snapshot through its own connection, writes a temp
//!   file, and renames it into place (processes still using the old file keep their
//!   mapping).
//! - A new base is normally made incrementally: load the current one, retire the nodes
//!   of entities changed since, add their new vectors. A retired node is not removed
//!   (usearch would reuse its slot and cut the paths through it, losing recall): its key
//!   moves to the tombstone range (>= 2^63), so it still routes searches but is never
//!   returned. Once tombstones exceed `FULL_REBUILD_CHURN` of the base, or on
//!   `--compact`, the base is rebuilt from scratch.
//! - `base.json` records the vector-table generation (`epoch`) and the sequence the base
//!   covers. If the table was cleared since (config change, index rebuild), the base is
//!   ignored and search falls back to an exact scan until the next compaction.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::embed;
use crate::lock::WriteLock;
use crate::store::{Id, Reader, Store, VecState};
use crate::{Error, Result};

/// Filtered search: an exact scan over the candidates costs ~c, the HNSW base with a
/// candidate filter ~N/c (sparser filters make it walk more nodes). They cross at
/// c* = sqrt(K * N); K = 250 was measured at 100k x 1024 (c* ~ 5k; exact won below,
/// HNSW above, both with the same results). Never below 1k. Env DQADB_EXACT_MAX overrides.
const CROSSOVER_K: f64 = 250.0;

pub fn exact_max_candidates(total_vectors: u64) -> usize {
    if let Some(v) = std::env::var("DQADB_EXACT_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        return v;
    }
    ((CROSSOVER_K * total_vectors as f64).sqrt() as usize).max(1_000)
}

/// Below this many vectors an exact scan is fast enough; no base is built.
pub const BASE_MIN_VECTORS: u64 = 20_000;
const BUILD_PAGE: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaseMeta {
    /// vector-table generation the base was built from
    pub epoch: String,
    /// highest change sequence included
    pub seq: u64,
    /// vector configuration (model + predicates) it was built with
    pub config: String,
    pub count: usize,
    pub dim: usize,
    /// vector encoding inside the HNSW file: "f16" or "i8"
    #[serde(default = "f16_name")]
    pub quant: String,
    /// tombstoned nodes (retired by incremental updates since the last full build)
    #[serde(default)]
    pub churn: usize,
}

/// Keys at or above this are tombstones (entity ids never get this large).
const TOMBSTONE: u64 = 1 << 63;

/// Rebuild from scratch once incremental removals exceed this share of the base.
const FULL_REBUILD_CHURN: f64 = 0.25;

fn f16_name() -> String {
    "f16".into()
}

/// Encoding for new bases: `i8` stores vectors at 1 byte/dim and re-ranks the
/// candidates exactly with the f16 vectors kept in the index; `f16` is exact enough
/// on its own. Override with DQADB_HNSW_QUANT.
pub const BASE_QUANT: &str = "i8";

pub fn base_quant() -> String {
    match std::env::var("DQADB_HNSW_QUANT").as_deref() {
        Ok("i8") => "i8".into(),
        Ok("f16") => "f16".into(),
        _ => BASE_QUANT.into(),
    }
}

fn scalar(quant: &str) -> ScalarKind {
    if quant == "i8" {
        ScalarKind::I8
    } else {
        ScalarKind::F16
    }
}

#[derive(Debug, Default, Serialize)]
pub struct CompactReport {
    pub built: bool,
    /// updated from the previous base rather than rebuilt from scratch
    pub incremental: bool,
    pub vectors: usize,
    pub seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skipped: Option<String>,
}

/// HNSW search breadth (usearch default is 64). Override with DQADB_HNSW_EF.
pub const EXPANSION_SEARCH: usize = 128;

type OpenBase = (PathBuf, BaseMeta, Option<std::time::SystemTime>, Arc<Index>);

/// The memory-mapped base, opened once per process and reused until the file changes
/// (a long-running `dqadb serve` then searches in ~1 ms instead of re-mapping each time).
fn open_cached(file: &Path, meta: &BaseMeta) -> Result<Arc<Index>> {
    static OPEN: Mutex<Option<OpenBase>> = Mutex::new(None);
    let mtime = std::fs::metadata(file).and_then(|m| m.modified()).ok();
    let mut slot = OPEN.lock().unwrap_or_else(|p| p.into_inner());
    if let Some((p, m, t, idx)) = slot.as_ref()
        && p == file
        && m == meta
        && *t == mtime
    {
        return Ok(idx.clone());
    }
    let index = Index::new(&options(meta.dim, &meta.quant)).map_err(ann_err)?;
    index
        .view(file.to_str().ok_or_else(|| ann_err("non-UTF-8 path"))?)
        .map_err(ann_err)?;
    let ef = std::env::var("DQADB_HNSW_EF")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(EXPANSION_SEARCH);
    index.change_expansion_search(ef);
    let index = Arc::new(index);
    *slot = Some((file.to_path_buf(), meta.clone(), mtime, index.clone()));
    Ok(index)
}

pub fn dir(index_dir: &Path) -> PathBuf {
    index_dir.join("vec")
}

fn files(dir: &Path) -> (PathBuf, PathBuf) {
    (dir.join("base.usearch"), dir.join("base.json"))
}

fn ann_err(e: impl std::fmt::Display) -> Error {
    Error::other(format!("vector index: {e}"))
}

fn options(dim: usize, quant: &str) -> IndexOptions {
    IndexOptions {
        dimensions: dim,
        metric: MetricKind::Cos,
        quantization: scalar(quant),
        ..Default::default()
    }
}

pub fn load_meta(dir: &Path) -> Option<BaseMeta> {
    let (_, meta) = files(dir);
    serde_json::from_slice(&std::fs::read(meta).ok()?).ok()
}

/// The base, if it belongs to the current vector-table generation and configuration.
pub fn usable(dir: &Path, r: &Reader, state: Option<&VecState>) -> Result<Option<BaseMeta>> {
    let (Some(meta), Some(state)) = (load_meta(dir), state) else {
        return Ok(None);
    };
    let fresh = Some(&meta.epoch) == r.vec_epoch()?.as_ref()
        && meta.config == state.config
        && meta.quant == base_quant()
        && meta.seq <= r.vec_seq()?;
    Ok(fresh.then_some(meta))
}

/// Whether a (re)build is worthwhile: enough vectors, and the delta has grown.
pub fn should_build(dir: &Path, r: &Reader, state: Option<&VecState>) -> Result<bool> {
    let total = r.vec_count()?;
    if total < BASE_MIN_VECTORS {
        return Ok(false);
    }
    Ok(match usable(dir, r, state)? {
        None => true,
        Some(meta) => {
            // incremental updates cost about the delta, so they can run often
            let delta = r.vec_changed_since(meta.seq)?.len() as u64;
            delta > (meta.count as u64 / 20).max(2_000)
        }
    })
}

/// Add vectors from all cores.
fn add_parallel(index: &Index, decoded: &[(Id, Vec<f32>)]) -> Result<()> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    std::thread::scope(|s| -> Result<()> {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || -> Result<()> {
                    for (id, v) in decoded.iter().skip(t).step_by(threads) {
                        index.add(*id, v).map_err(ann_err)?;
                    }
                    Ok(())
                })
            })
            .collect();
        for h in handles {
            h.join()
                .map_err(|_| Error::other("vector index build thread panicked"))??;
        }
        Ok(())
    })
}

fn decode(rows: Vec<(Id, Vec<u8>)>, dim: usize) -> Result<Vec<(Id, Vec<f32>)>> {
    let decoded: Vec<(Id, Vec<f32>)> = rows
        .into_iter()
        .map(|(id, b)| (id, embed::from_f16_bytes(&b)))
        .collect();
    if decoded.iter().any(|(_, v)| v.len() != dim) {
        return Err(Error::other(format!(
            "stored vectors do not have {dim} dimensions"
        )));
    }
    Ok(decoded)
}

/// Write `index` and its metadata into place (temp file + rename each).
fn install(dir: &Path, index: &Index, meta: &BaseMeta) -> Result<()> {
    let (file, meta_path) = files(dir);
    let tmp = dir.join("base.usearch.tmp");
    index
        .save(tmp.to_str().ok_or_else(|| ann_err("non-UTF-8 path"))?)
        .map_err(ann_err)?;
    std::fs::rename(&tmp, &file)?;
    let tmp_meta = dir.join("base.json.tmp");
    std::fs::write(&tmp_meta, serde_json::to_vec_pretty(meta)?)?;
    std::fs::rename(&tmp_meta, &meta_path)?;
    Ok(())
}

/// Bring the base up to date from a snapshot of `store_path` (own connection, own
/// lock): incrementally when the current base allows it, else (or with `full`) from
/// scratch.
pub fn build(store_path: &Path, dir: &Path, dim: usize, full: bool) -> Result<CompactReport> {
    let started = Instant::now();
    std::fs::create_dir_all(dir)?;
    let _lock = match WriteLock::acquire(&dir.join(".compact.lock"), Duration::ZERO) {
        Ok(l) => l,
        Err(Error::LockTimeout(_)) => {
            return Ok(CompactReport {
                skipped: Some("another compaction is running".into()),
                ..Default::default()
            });
        }
        Err(e) => return Err(e),
    };
    let store = Store::open(store_path)?;
    let r = store.read()?;
    let (Some(state), Some(epoch)) = (r.vec_state()?, r.vec_epoch()?) else {
        return Ok(CompactReport {
            skipped: Some("no vectors yet (run: dqadb embed)".into()),
            ..Default::default()
        });
    };
    let seq = r.vec_seq()?;

    // incremental: current base minus what changed, plus the new vectors
    if let Some(old) = usable(dir, &r, Some(&state))?.filter(|m| {
        !full && m.dim == dim && (m.churn as f64) < FULL_REBUILD_CHURN * m.count.max(1) as f64
    }) {
        let mut changed = r.vec_changed_since(old.seq)?;
        changed.sort_unstable();
        changed.dedup();
        if changed.is_empty() && old.seq == seq {
            return Ok(CompactReport {
                skipped: Some("vector base is up to date".into()),
                ..Default::default()
            });
        }
        let mut rows = Vec::new();
        r.vec_of(&changed, |id, b| rows.push((id, b.to_vec())))?;
        drop(r);
        let fresh = decode(rows, dim)?;
        let (file, _) = files(dir);
        let index = Index::new(&options(dim, &old.quant)).map_err(ann_err)?;
        index
            .load(file.to_str().ok_or_else(|| ann_err("non-UTF-8 path"))?)
            .map_err(ann_err)?;
        let mut removed = 0usize;
        for id in &changed {
            if index.contains(*id) {
                let tomb = TOMBSTONE + (old.churn + removed) as u64;
                index.rename(*id, tomb).map_err(ann_err)?;
                removed += 1;
            }
        }
        index.reserve(index.size() + fresh.len()).map_err(ann_err)?;
        add_parallel(&index, &fresh)?;
        let meta = BaseMeta {
            epoch,
            seq,
            config: state.config,
            count: old.count + fresh.len() - removed,
            dim,
            quant: old.quant,
            churn: old.churn + removed,
        };
        install(dir, &index, &meta)?;
        return Ok(CompactReport {
            built: true,
            incremental: true,
            vectors: meta.count,
            seconds: started.elapsed().as_secs_f64(),
            skipped: None,
        });
    }

    let count = r.vec_count()? as usize;
    let quant = base_quant();
    let index = Index::new(&options(dim, &quant)).map_err(ann_err)?;
    index.reserve(count.max(1)).map_err(ann_err)?;
    let mut added = 0usize;
    r.vec_upto_pages(seq, BUILD_PAGE, |rows| {
        let decoded = decode(rows, dim)?;
        add_parallel(&index, &decoded)?;
        added += decoded.len();
        Ok(())
    })?;
    drop(r);
    let meta = BaseMeta {
        epoch,
        seq,
        config: state.config,
        count: added,
        dim,
        quant,
        churn: 0,
    };
    install(dir, &index, &meta)?;
    Ok(CompactReport {
        built: true,
        incremental: false,
        vectors: added,
        seconds: started.elapsed().as_secs_f64(),
        skipped: None,
    })
}

/// Top `n` by cosine: HNSW over the base (skipping entities changed since it was built)
/// merged with an exact scan of the delta, optionally restricted to `allow`.
/// Returns (hits, vectors considered).
pub fn search(
    dir: &Path,
    meta: &BaseMeta,
    r: &Reader,
    query: &[f32],
    n: usize,
    allow: Option<&HashSet<u64>>,
) -> Result<(Vec<(Id, f32)>, usize)> {
    let stale: HashSet<u64> = r.vec_changed_since(meta.seq)?.into_iter().collect();
    let (file, _) = files(dir);
    let index = open_cached(&file, meta)?;
    let ef = std::env::var("DQADB_HNSW_EF")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(EXPANSION_SEARCH);
    if index.expansion_search() != ef {
        index.change_expansion_search(ef);
    }
    // an i8 base only proposes candidates; their f16 vectors give the exact order
    let pool = if meta.quant == "i8" {
        (n * 8).max(64)
    } else {
        n
    };
    let tombs = meta.churn > 0;
    let found = match (stale.is_empty() && !tombs, allow) {
        (true, None) => index.search(query, pool).map_err(ann_err)?,
        (_, None) => index
            .filtered_search(query, pool, |k| k < TOMBSTONE && !stale.contains(&k))
            .map_err(ann_err)?,
        (_, Some(a)) => index
            .filtered_search(query, pool, |k| a.contains(&k) && !stale.contains(&k))
            .map_err(ann_err)?,
    };
    let mut hits: Vec<(Id, f32)> = if meta.quant == "i8" {
        let mut exact = Vec::with_capacity(found.keys.len());
        let (mut bits, mut buf) = (Vec::new(), Vec::new());
        r.vec_of(&found.keys, |id, blob| {
            exact.push((id, embed::dot_f16(query, blob, &mut bits, &mut buf)))
        })?;
        exact
    } else {
        found
            .keys
            .iter()
            .zip(&found.distances)
            .map(|(k, d)| (*k, 1.0 - *d))
            .collect()
    };
    r.vec_since(meta.seq, |id, blob| {
        if allow.is_none_or(|a| a.contains(&id)) {
            hits.push((id, embed::dot(query, &embed::from_f16_bytes(blob))));
        }
    })?;
    hits.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    hits.truncate(n);
    let considered = allow.map(|a| a.len()).unwrap_or(r.vec_count()? as usize);
    Ok((hits, considered))
}
