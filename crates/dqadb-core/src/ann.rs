//! Approximate nearest neighbours: an HNSW "base" (usearch, f16, memory-mapped) over the
//! vectors up to some change sequence, plus an exact scan of the "delta" changed since.
//! (Spec §8/§11, decision 2026-11-09.)
//!
//! - The base is immutable; readers memory-map it. Building a new one never blocks writers
//!   or readers: it reads a snapshot through its own connection, writes a temp file, and
//!   renames it into place (processes still using the old file keep their mapping).
//! - `base.json` records the vector-table generation (`epoch`) and the sequence the base
//!   covers. If the table was cleared since (config change, index rebuild), the base is
//!   ignored and search falls back to an exact scan until the next compaction.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::embed;
use crate::lock::WriteLock;
use crate::store::{Id, Reader, Store, VecState};
use crate::{Error, Result};

/// Filtered searches with more candidates than this use the base (approximate) instead
/// of an exact scan over the candidates.
pub const EXACT_MAX_CANDIDATES: usize = 20_000;

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
}

#[derive(Debug, Default, Serialize)]
pub struct CompactReport {
    pub built: bool,
    pub vectors: usize,
    pub seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skipped: Option<String>,
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

fn options(dim: usize) -> IndexOptions {
    IndexOptions {
        dimensions: dim,
        metric: MetricKind::Cos,
        quantization: ScalarKind::F16,
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
            let delta = r.vec_changed_since(meta.seq)?.len() as u64;
            delta > (meta.count as u64 / 10).max(5_000)
        }
    })
}

/// Build a new base from a snapshot of `store_path` (own connection, own lock).
pub fn build(store_path: &Path, dir: &Path, dim: usize) -> Result<CompactReport> {
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
    let count = r.vec_count()? as usize;
    let index = Index::new(&options(dim)).map_err(ann_err)?;
    index.reserve(count.max(1)).map_err(ann_err)?;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let mut added = 0usize;
    r.vec_upto_pages(seq, BUILD_PAGE, |rows| {
        let decoded: Vec<(Id, Vec<f32>)> = rows
            .into_iter()
            .map(|(id, b)| (id, embed::from_f16_bytes(&b)))
            .collect();
        if decoded.iter().any(|(_, v)| v.len() != dim) {
            return Err(Error::other(format!(
                "stored vectors do not have {dim} dimensions"
            )));
        }
        std::thread::scope(|s| -> Result<()> {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let (index, decoded) = (&index, &decoded);
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
        })?;
        added += decoded.len();
        Ok(())
    })?;
    drop(r);
    let (file, meta_path) = files(dir);
    let tmp = dir.join("base.usearch.tmp");
    index
        .save(tmp.to_str().ok_or_else(|| ann_err("non-UTF-8 path"))?)
        .map_err(ann_err)?;
    std::fs::rename(&tmp, &file)?;
    let meta = BaseMeta {
        epoch,
        seq,
        config: state.config,
        count: added,
        dim,
    };
    let tmp_meta = dir.join("base.json.tmp");
    std::fs::write(&tmp_meta, serde_json::to_vec_pretty(&meta)?)?;
    std::fs::rename(&tmp_meta, &meta_path)?;
    Ok(CompactReport {
        built: true,
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
    let index = Index::new(&options(meta.dim)).map_err(ann_err)?;
    index
        .view(file.to_str().ok_or_else(|| ann_err("non-UTF-8 path"))?)
        .map_err(ann_err)?;
    let found = match (stale.is_empty(), allow) {
        (true, None) => index.search(query, n).map_err(ann_err)?,
        (_, None) => index
            .filtered_search(query, n, |k| !stale.contains(&k))
            .map_err(ann_err)?,
        (_, Some(a)) => index
            .filtered_search(query, n, |k| a.contains(&k) && !stale.contains(&k))
            .map_err(ann_err)?,
    };
    let mut hits: Vec<(Id, f32)> = found
        .keys
        .iter()
        .zip(&found.distances)
        .map(|(k, d)| (*k, 1.0 - *d))
        .collect();
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
