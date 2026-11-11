//! Approximate nearest neighbours over immutable HNSW segments (usearch, memory-mapped),
//! plus an exact scan of the small "delta" changed since the last flush.
//! The layout follows Lucene / Qdrant (decision 2026-11-10, evaluated on the Big-ANN
//! NeurIPS'23 streaming runbook against DiskANN and Qdrant; see bench/vector/RESULTS.md).
//!
//! - A segment is a usearch file plus the sorted list of its keys. It is never modified.
//!   Readers (any process) memory-map it.
//! - Deletes and updates do not touch the graph: each segment has a deletion list
//!   (an immutable, versioned file) naming its nodes that are no longer current. A
//!   deleted node still routes searches, so the graph keeps its connectivity.
//! - `build` ("flush") turns the delta into a new small segment and extends the deletion
//!   lists, then merges like Qdrant's optimizer (decision 2026-11-11): once there are
//!   more segments below full size (`MAX_SEGMENT`) than CPU cores, the smallest (at
//!   least `MERGE_MIN`) are merged; a segment whose deleted share passes
//!   `VACUUM_DELETED` is rebuilt from its live vectors. Merging rebuilds only the
//!   segments involved, never the whole index. Vector-specific systems keep few, large
//!   segments (Qdrant: one per core; Milvus: ~1 GB; LanceDB: one index plus deltas)
//!   because every segment costs a graph search per query; Lucene's tiered factor of
//!   10 suits text, where a segment is a cheap lookup.
//! - A query searches the segments in parallel (one per core) and merges the results.
//! - `segments.json` (the manifest) names the live files and the change sequence they
//!   cover; it is replaced atomically, so readers see either the old or the new set.
//!   Unreferenced files are removed after a grace period (readers still holding an old
//!   manifest keep working).
//! - The manifest records the vector-table generation (`epoch`). If the table was
//!   cleared since (config change, index rebuild), the segments are ignored and search
//!   falls back to an exact scan until the next build.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::embed;
use crate::lock::WriteLock;
use crate::store::{Id, Reader, Store, VecState};
use crate::{Error, Result};

/// Filtered search: an exact scan over the candidates costs ~c, the HNSW segments with
/// a candidate filter ~N/c (sparser filters make it walk more nodes). They cross at
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

/// Below this many vectors an exact scan is fast enough; no segments are built.
pub const BASE_MIN_VECTORS: u64 = 20_000;
/// Flush the delta into a new segment once this many entities changed.
pub const FLUSH_CHANGES: u64 = 2_048;
/// Merge at least this many segments at once (Qdrant merges >= 3 of the smallest).
pub const MERGE_MIN: usize = 3;

/// Segments below full size to keep before merging: the number of CPU cores (Qdrant's
/// `default_segment_number`), at least 2. Env DQADB_MAX_SEGMENTS overrides.
pub fn max_segments() -> usize {
    std::env::var("DQADB_MAX_SEGMENTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(cores)
        .max(2)
}

fn cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
}
/// Segments are not merged beyond this size (bounds the cost of any one rebuild).
pub const MAX_SEGMENT: usize = 262_144;
/// Rebuild a segment once this share of its nodes is deleted (Qdrant's default 0.2).
pub const VACUUM_DELETED: f64 = 0.2;
const BUILD_PAGE: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Segment {
    /// file stem: `<name>.usearch`, `<name>.ids`
    pub name: String,
    /// nodes in the graph (live + deleted)
    pub count: usize,
    pub deleted: usize,
    /// deletion list file, when `deleted > 0`
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub del: Option<String>,
}

impl Segment {
    fn live(&self) -> usize {
        self.count - self.deleted
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// vector-table generation the segments were built from
    pub epoch: String,
    /// highest change sequence reflected in the segments and deletion lists
    pub seq: u64,
    /// vector configuration (model + predicates) they were built with
    pub config: String,
    pub dim: usize,
    /// vector encoding inside the HNSW files: "f16" or "i8"
    pub quant: String,
    /// counter for new file names
    pub next: u64,
    pub segments: Vec<Segment>,
}

impl Manifest {
    pub fn live(&self) -> usize {
        self.segments.iter().map(Segment::live).sum()
    }
}

/// Encoding for new segments: `f16` (default) is exact enough to rank directly; `i8`
/// stores 1 byte/dim and re-ranks the candidates with the f16 vectors from the store.
/// Override with DQADB_HNSW_QUANT.
pub const BASE_QUANT: &str = "f16";

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
    /// updated from the existing segments rather than rebuilt from scratch
    pub incremental: bool,
    /// live vectors in the segments afterwards
    pub vectors: usize,
    pub segments: usize,
    /// vectors written into new segments by this build (flush + merges)
    pub written: usize,
    pub seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skipped: Option<String>,
}

/// HNSW search breadth (usearch default is 64). Override with DQADB_HNSW_EF.
pub const EXPANSION_SEARCH: usize = 128;

fn ef() -> usize {
    std::env::var("DQADB_HNSW_EF")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(EXPANSION_SEARCH)
}

pub fn dir(index_dir: &Path) -> PathBuf {
    index_dir.join("vec")
}

fn manifest_path(dir: &Path) -> PathBuf {
    dir.join("segments.json")
}

fn ann_err(e: impl std::fmt::Display) -> Error {
    Error::other(format!("vector index: {e}"))
}

fn path_str(p: &Path) -> Result<&str> {
    p.to_str().ok_or_else(|| ann_err("non-UTF-8 path"))
}

fn options(dim: usize, quant: &str) -> IndexOptions {
    IndexOptions {
        dimensions: dim,
        metric: MetricKind::Cos,
        quantization: scalar(quant),
        ..Default::default()
    }
}

pub fn load_manifest(dir: &Path) -> Option<Manifest> {
    serde_json::from_slice(&std::fs::read(manifest_path(dir)).ok()?).ok()
}

/// The segments, if they belong to the current vector-table generation and configuration.
pub fn usable(dir: &Path, r: &Reader, state: Option<&VecState>) -> Result<Option<Manifest>> {
    let (Some(m), Some(state)) = (load_manifest(dir), state) else {
        return Ok(None);
    };
    let fresh = Some(&m.epoch) == r.vec_epoch()?.as_ref()
        && m.config == state.config
        && m.quant == base_quant()
        && m.seq <= r.vec_seq()?;
    Ok(fresh.then_some(m))
}

/// Whether a build is worthwhile: enough vectors and no usable segments yet, or a delta
/// large enough to flush.
pub fn should_build(dir: &Path, r: &Reader, state: Option<&VecState>) -> Result<bool> {
    Ok(match usable(dir, r, state)? {
        None => r.vec_count()? >= BASE_MIN_VECTORS,
        Some(m) => r.vec_changed_count(m.seq)? >= FLUSH_CHANGES,
    })
}

// ---- files ----

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, bytes)?;
    std::fs::File::open(&tmp)?.sync_all()?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn write_keys(path: &Path, keys: &[Id]) -> Result<()> {
    let mut b = Vec::with_capacity(keys.len() * 8);
    for k in keys {
        b.extend_from_slice(&k.to_le_bytes());
    }
    write_atomic(path, &b)
}

fn read_keys(path: &Path) -> Result<Vec<Id>> {
    let b = std::fs::read(path)?;
    if b.len() % 8 != 0 {
        return Err(ann_err(format!("{} is truncated", path.display())));
    }
    Ok(b.as_chunks::<8>()
        .0
        .iter()
        .map(|c| u64::from_le_bytes(*c))
        .collect())
}

// ---- reader-side cache (one per process; files are immutable, keyed by name) ----

/// A segment graph viewed in place over a mapping this process owns. usearch's own
/// `view(path)` is not used: when its open fails (file just removed by a compaction) it
/// still closes descriptor 0 on cleanup (2.26.6 `memory_mapped_file_t::close`), which
/// can close a file another component of the process holds.
struct Viewed {
    // declared first: dropped before the mapping it reads
    index: Index,
    _map: memmap2::Mmap,
}

impl std::ops::Deref for Viewed {
    type Target = Index;
    fn deref(&self) -> &Index {
        &self.index
    }
}

fn view_file(path: &Path, dim: usize, quant: &str) -> Result<Viewed> {
    let file = std::fs::File::open(path)?;
    // SAFETY: segment files are immutable once renamed into place and are replaced only
    // by new names; a removed file stays readable while mapped
    let map = unsafe { memmap2::Mmap::map(&file)? };
    let index = Index::new(&options(dim, quant)).map_err(ann_err)?;
    // SAFETY: `map` lives in the same struct and is dropped after `index`
    unsafe { index.view_from_buffer(&map) }.map_err(ann_err)?;
    Ok(Viewed { index, _map: map })
}

struct Loaded {
    index: Arc<Viewed>,
    deleted: Arc<HashSet<Id>>,
}

/// A file on disk: path plus modification time and length (a rebuilt index directory
/// can reuse names, and a cached mapping must not outlive the file it came from).
type FileKey = (PathBuf, Option<SystemTime>, u64);

fn file_key(path: &Path) -> FileKey {
    let md = std::fs::metadata(path).ok();
    (
        path.to_path_buf(),
        md.as_ref().and_then(|m| m.modified().ok()),
        md.map(|m| m.len()).unwrap_or(0),
    )
}

#[derive(Default)]
struct Cache {
    indexes: HashMap<FileKey, Arc<Viewed>>,
    deletions: HashMap<FileKey, Arc<HashSet<Id>>>,
}

static CACHE: Mutex<Option<Cache>> = Mutex::new(None);

fn open_segments(dir: &Path, m: &Manifest) -> Result<Vec<Loaded>> {
    let mut guard = CACHE.lock().unwrap_or_else(|p| p.into_inner());
    let cache = guard.get_or_insert_with(Cache::default);
    let mut out = Vec::with_capacity(m.segments.len());
    let mut used_i = HashSet::new();
    let mut used_d = HashSet::new();
    for s in &m.segments {
        let path = dir.join(format!("{}.usearch", s.name));
        let file = file_key(&path);
        let index = match cache.indexes.get(&file) {
            Some(i) => i.clone(),
            None => {
                let index = Arc::new(view_file(&path, m.dim, &m.quant)?);
                cache.indexes.insert(file.clone(), index.clone());
                index
            }
        };
        let deleted = match &s.del {
            None => Arc::new(HashSet::new()),
            Some(d) => {
                let path = dir.join(d);
                let file = file_key(&path);
                let set = match cache.deletions.get(&file) {
                    Some(set) => set.clone(),
                    None => {
                        let set = Arc::new(read_keys(&path)?.into_iter().collect::<HashSet<_>>());
                        cache.deletions.insert(file.clone(), set.clone());
                        set
                    }
                };
                used_d.insert(file);
                set
            }
        };
        used_i.insert(file);
        out.push(Loaded { index, deleted });
    }
    // drop mappings of segments no longer in the manifest
    cache.indexes.retain(|k, _| used_i.contains(k));
    cache.deletions.retain(|k, _| used_d.contains(k));
    Ok(out)
}

/// Open the segments of `m`; when a compaction replaced them meanwhile (their files are
/// removed as soon as a newer `segments.json` is written), retry with the newer manifest,
/// like Lucene's `SegmentInfos.FindSegmentsFile`. Returns the manifest used if not `m`.
fn open_current(dir: &Path, m: &Manifest) -> Result<(Option<Manifest>, Vec<Loaded>)> {
    let mut err = match open_segments(dir, m) {
        Ok(s) => return Ok((None, s)),
        Err(e) => e,
    };
    let mut seen = m.clone();
    for _ in 0..5 {
        match load_manifest(dir) {
            Some(newer) if newer != seen => match open_segments(dir, &newer) {
                Ok(s) => return Ok((Some(newer), s)),
                Err(e) => {
                    err = e;
                    seen = newer;
                }
            },
            _ => break,
        }
    }
    Err(err)
}

/// Top `n` by cosine across the segments (skipping deleted nodes and entities changed
/// since the last flush) merged with an exact scan of the delta, optionally restricted
/// to `allow`. Returns (hits, vectors considered).
pub fn search(
    dir: &Path,
    m: &Manifest,
    r: &Reader,
    query: &[f32],
    n: usize,
    allow: Option<&HashSet<u64>>,
) -> Result<(Vec<(Id, f32)>, usize)> {
    let (newer, segments) = open_current(dir, m)?;
    let m = newer.as_ref().unwrap_or(m);
    let stale: HashSet<u64> = r.vec_changed_since(m.seq)?.into_iter().collect();
    let ef = ef();
    // Ask each segment for its whole search beam (>= ef): the top `n` of the same beam
    // is then a prefix of the top `n + k`, so pages of one query stay consistent.
    // An i8 segment only proposes candidates; their f16 vectors give the exact order.
    let pool = if m.quant == "i8" { (n * 8).max(64) } else { n }.max(ef);
    let one = |seg: &Loaded| -> Result<(Vec<Id>, Vec<f32>)> {
        if seg.index.expansion_search() != ef {
            seg.index.change_expansion_search(ef);
        }
        let plain = stale.is_empty() && seg.deleted.is_empty() && allow.is_none();
        let found = if plain {
            seg.index.search(query, pool)
        } else {
            seg.index.filtered_search(query, pool, |k| {
                !seg.deleted.contains(&k)
                    && !stale.contains(&k)
                    && allow.is_none_or(|a| a.contains(&k))
            })
        }
        .map_err(ann_err)?;
        Ok((found.keys, found.distances))
    };
    // segments in parallel, one per core (a query on one small segment stays inline)
    let width = cores().min(segments.len());
    let per_segment: Vec<Result<(Vec<Id>, Vec<f32>)>> = if width <= 1 {
        segments.iter().map(one).collect()
    } else {
        std::thread::scope(|sc| {
            let handles: Vec<_> = (0..width)
                .map(|t| {
                    let (segments, one) = (&segments, &one);
                    sc.spawn(move || {
                        segments
                            .iter()
                            .skip(t)
                            .step_by(width)
                            .map(one)
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| {
                    h.join().unwrap_or_else(|_| {
                        vec![Err(Error::other("vector segment search panicked"))]
                    })
                })
                .collect()
        })
    };
    let mut hits: Vec<(Id, f32)> = Vec::new();
    let mut proposed: Vec<Id> = Vec::new();
    for r in per_segment {
        let (keys, distances) = r?;
        if m.quant == "i8" {
            proposed.extend_from_slice(&keys);
        } else {
            hits.extend(keys.iter().zip(&distances).map(|(k, d)| (*k, 1.0 - *d)));
        }
    }
    if !proposed.is_empty() {
        let (mut bits, mut buf) = (Vec::new(), Vec::new());
        r.vec_of(&proposed, |id, blob| {
            hits.push((id, embed::dot_f16(query, blob, &mut bits, &mut buf)))
        })?;
    }
    r.vec_since(m.seq, |id, blob| {
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

// ---- building ----

/// One segment to write: its file stem and its entities (sorted).
struct Job {
    name: String,
    ids: Vec<Id>,
}

fn new_job(m: &mut Manifest, ids: Vec<Id>) -> Job {
    let name = format!("s{:08}", m.next);
    m.next += 1;
    Job { name, ids }
}

/// Build one segment from (id, f16 bytes) rows, inserting from a single thread:
/// usearch graphs built by concurrent inserts are measurably worse (135k Wikipedia
/// vectors, recall@10 at ef 512: 0.995 with 1 thread, 0.989 with 2, 0.982 with 4).
fn build_one(
    dir: &Path,
    dim: usize,
    quant: &str,
    name: &str,
    rows: &[(Id, Vec<u8>)],
) -> Result<Segment> {
    let index = Index::new(&options(dim, quant)).map_err(ann_err)?;
    index
        .reserve_capacity_and_threads(rows.len().max(1), 1)
        .map_err(ann_err)?;
    let mut keys = Vec::with_capacity(rows.len());
    for (id, b) in rows {
        let v = embed::from_f16_bytes(b);
        if v.len() != dim {
            return Err(Error::other(format!(
                "stored vectors do not have {dim} dimensions"
            )));
        }
        index.add(*id, &v).map_err(ann_err)?;
        keys.push(*id);
    }
    keys.sort_unstable();
    let file = dir.join(format!("{name}.usearch"));
    let tmp = dir.join(format!("{name}.usearch.tmp"));
    index.save(path_str(&tmp)?).map_err(ann_err)?;
    std::fs::File::open(&tmp)?.sync_all()?;
    std::fs::rename(&tmp, &file)?;
    write_keys(&dir.join(format!("{name}.ids")), &keys)?;
    Ok(Segment {
        name: name.to_string(),
        count: keys.len(),
        deleted: 0,
        del: None,
    })
}

/// Write the segments for `jobs`: vectors are read here (the reader is not shared
/// across threads), then up to one segment per core is built at a time.
fn write_segments(dir: &Path, r: &Reader, m: &Manifest, jobs: Vec<Job>) -> Result<Vec<Segment>> {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    // hyper-threads add little to graph building and double the memory held
    let width = (cores / 2).max(1);
    let mut out = Vec::with_capacity(jobs.len());
    for batch in jobs.chunks(width) {
        let mut loaded = Vec::with_capacity(batch.len());
        for job in batch {
            let mut rows = Vec::with_capacity(job.ids.len());
            r.vec_of(&job.ids, |id, b| rows.push((id, b.to_vec())))?;
            loaded.push((job.name.as_str(), rows));
        }
        let built: Vec<Result<Segment>> = std::thread::scope(|s| {
            let handles: Vec<_> = loaded
                .iter()
                .map(|(name, rows)| s.spawn(move || build_one(dir, m.dim, &m.quant, name, rows)))
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err(Error::other("vector segment build panicked")))
                })
                .collect()
        });
        for seg in built {
            out.push(seg?);
        }
    }
    Ok(out)
}

/// Merge by appending: copy `dest`'s graph, tombstone its deleted nodes in the graph
/// itself (usearch skips them in results but still routes through them), and insert
/// `ids` into it from a single thread, the same sequential inserts a fresh build does.
/// Costs the appended vectors instead of a rebuild of all of them. The new segment has
/// no deletion list; its tombstones stay counted in `deleted` until a vacuum.
fn append_into(
    dir: &Path,
    r: &Reader,
    m: &mut Manifest,
    dest: &Segment,
    ids: &[Id],
) -> Result<Segment> {
    let index = Index::new(&options(m.dim, &m.quant)).map_err(ann_err)?;
    index
        .load(path_str(&dir.join(format!("{}.usearch", dest.name)))?)
        .map_err(ann_err)?;
    index
        .reserve_capacity_and_threads(dest.count + ids.len(), 1)
        .map_err(ann_err)?;
    let mut keys = live_keys(dir, dest)?;
    if let Some(d) = &dest.del {
        for k in read_keys(&dir.join(d))? {
            index.remove(k).map_err(ann_err)?;
        }
    }
    let mut added = 0usize;
    r.vec_of(ids, |id, b| {
        let v = embed::from_f16_bytes(b);
        if v.len() != m.dim {
            return;
        }
        if index.add(id, &v).is_ok() {
            keys.push(id);
            added += 1;
        }
    })?;
    if added != ids.len() {
        return Err(Error::other("vector segment append failed"));
    }
    keys.sort_unstable();
    let name = format!("s{:08}", m.next);
    m.next += 1;
    let file = dir.join(format!("{name}.usearch"));
    let tmp = dir.join(format!("{name}.usearch.tmp"));
    index.save(path_str(&tmp)?).map_err(ann_err)?;
    std::fs::File::open(&tmp)?.sync_all()?;
    std::fs::rename(&tmp, &file)?;
    write_keys(&dir.join(format!("{name}.ids")), &keys)?;
    Ok(Segment {
        name,
        count: dest.count + added,
        deleted: dest.deleted,
        del: None,
    })
}

/// Live keys of a segment.
fn live_keys(dir: &Path, s: &Segment) -> Result<Vec<Id>> {
    let mut keys = read_keys(&dir.join(format!("{}.ids", s.name)))?;
    if let Some(d) = &s.del {
        let dead: HashSet<Id> = read_keys(&dir.join(d))?.into_iter().collect();
        keys.retain(|k| !dead.contains(k));
    }
    Ok(keys)
}

/// The next merge to do, as indexes into `segs` ((count, deleted) per segment): a
/// segment past the deletion threshold alone, or, when more than `cap` segments are
/// below full size, the smallest of them (at least `MERGE_MIN`, enough to get back to
/// `cap`).
pub fn plan_merge(segs: &[(usize, usize)]) -> Option<Vec<usize>> {
    plan_merge_with(segs, max_segments())
}

fn plan_merge_with(segs: &[(usize, usize)], cap: usize) -> Option<Vec<usize>> {
    if let Some(i) = segs
        .iter()
        .position(|&(c, d)| c > 0 && d as f64 > VACUUM_DELETED * c as f64)
    {
        return Some(vec![i]);
    }
    let mut small: Vec<usize> = (0..segs.len())
        .filter(|&i| segs[i].0 - segs[i].1 < MAX_SEGMENT)
        .collect();
    if small.len() <= cap {
        return None;
    }
    small.sort_by_key(|&i| (segs[i].0 - segs[i].1, i));
    small.truncate((small.len() - cap + 1).max(MERGE_MIN).min(small.len()));
    Some(small)
}

/// Remove the files the manifest no longer refers to, right after it is written (as
/// Lucene's IndexFileDeleter does): processes that still map them keep reading them
/// (POSIX unlink semantics) and readers that have not opened them yet retry with the new
/// manifest. A file that cannot be removed (open on Windows) is retried next time.
fn collect_garbage(dir: &Path, m: &Manifest) {
    let mut keep: HashSet<String> = HashSet::new();
    for s in &m.segments {
        keep.insert(format!("{}.usearch", s.name));
        keep.insert(format!("{}.ids", s.name));
        if let Some(d) = &s.del {
            keep.insert(d.clone());
        }
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let ours = (name.starts_with('s') && name.len() > 9 && name[1..9].bytes().all(|b| b.is_ascii_digit()))
            // files of the previous single-base layout
            || name.starts_with("base.");
        if !ours || keep.contains(&name) {
            continue;
        }
        let _ = std::fs::remove_file(e.path());
    }
}

/// Bring the segments up to date from a snapshot of `store_path` (own connection, own
/// lock): flush the delta and merge as needed, or (with `full`, or when the segments
/// are unusable) rebuild them all.
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
    let mut written = 0usize;

    let current = usable(dir, &r, Some(&state))?.filter(|m| !full && m.dim == dim);
    let incremental = current.is_some();
    let mut m = match current {
        Some(mut m) => {
            // flush: mark replaced/removed nodes deleted, put the delta in a new segment
            let mut changed = r.vec_changed_since(m.seq)?;
            changed.sort_unstable();
            changed.dedup();
            if changed.is_empty() && m.seq == seq && plan_merge(&pairs(&m)).is_none() {
                return Ok(CompactReport {
                    skipped: Some("vector segments are up to date".into()),
                    vectors: m.live(),
                    segments: m.segments.len(),
                    ..Default::default()
                });
            }
            let loaded = open_segments(dir, &m)?;
            for (s, l) in m.segments.iter_mut().zip(&loaded) {
                let newly: Vec<Id> = changed
                    .iter()
                    .copied()
                    .filter(|k| l.index.contains(*k) && !l.deleted.contains(k))
                    .collect();
                if newly.is_empty() {
                    continue;
                }
                // nodes tombstoned inside the graph by an append stay counted
                s.deleted += newly.len();
                let mut all: Vec<Id> = l.deleted.iter().copied().chain(newly).collect();
                all.sort_unstable();
                let name = format!("{}.d{:08}", s.name, m.next);
                m.next += 1;
                write_keys(&dir.join(&name), &all)?;
                s.del = Some(name);
            }
            drop(loaded);
            let mut present = Vec::new();
            r.vec_of(&changed, |id, _| present.push(id))?;
            present.sort_unstable();
            if !present.is_empty() {
                let job = new_job(&mut m, present);
                for seg in write_segments(dir, &r, &m, vec![job])? {
                    written += seg.count;
                    m.segments.push(seg);
                }
            }
            m.segments.retain(|s| s.live() > 0);
            m.seq = seq;
            m
        }
        None => {
            // from scratch: every vector, in id order, cut into MAX_SEGMENT pieces
            let mut m = Manifest {
                epoch,
                seq,
                config: state.config.clone(),
                dim,
                quant: base_quant(),
                next: load_manifest(dir).map(|o| o.next).unwrap_or(1),
                segments: Vec::new(),
            };
            let mut ids = Vec::new();
            r.vec_upto_pages(seq, BUILD_PAGE, |rows| {
                ids.extend(rows.into_iter().map(|(id, _)| id));
                Ok(())
            })?;
            let jobs: Vec<Job> = ids
                .chunks(MAX_SEGMENT)
                .map(|c| c.to_vec())
                .collect::<Vec<_>>()
                .into_iter()
                .map(|c| new_job(&mut m, c))
                .collect();
            for seg in write_segments(dir, &r, &m, jobs)? {
                written += seg.count;
                m.segments.push(seg);
            }
            m
        }
    };

    // merges (consistent with the same snapshot: every live node is current at `seq`)
    while let Some(pick) = plan_merge(&pairs(&m)) {
        // Append into the picked graph with the most live vectors when the rest fits next
        // to it (Lucene's IncrementalHnswGraphMerger, Qdrant's incremental HNSW indexing);
        // a vacuum (one segment) or an oversized merge is a fresh build. Lucene reuses a
        // base graph up to 40% deleted; here every segment is under VACUUM_DELETED (20%)
        // because vacuums are planned first.
        let dest = *pick
            .iter()
            .max_by_key(|&&i| m.segments[i].live())
            .expect("a merge picks segments");
        let mut ids = Vec::new();
        for &i in pick.iter().filter(|&&i| i != dest) {
            ids.extend(live_keys(dir, &m.segments[i])?);
        }
        let mut pick = pick;
        pick.sort_unstable_by(|a, b| b.cmp(a));
        if pick.len() > 1 && m.segments[dest].count + ids.len() <= MAX_SEGMENT {
            ids.sort_unstable();
            let target = m.segments[dest].clone();
            let seg = append_into(dir, &r, &mut m, &target, &ids)?;
            written += ids.len();
            for i in pick {
                m.segments.remove(i);
            }
            m.segments.push(seg);
            continue;
        }
        ids.extend(live_keys(dir, &m.segments[dest])?);
        ids.sort_unstable();
        for i in pick {
            m.segments.remove(i);
        }
        let jobs: Vec<Job> = ids
            .chunks(MAX_SEGMENT)
            .map(|c| c.to_vec())
            .collect::<Vec<_>>()
            .into_iter()
            .map(|c| new_job(&mut m, c))
            .collect();
        for seg in write_segments(dir, &r, &m, jobs)? {
            written += seg.count;
            m.segments.push(seg);
        }
    }
    drop(r);

    write_atomic(&manifest_path(dir), &serde_json::to_vec_pretty(&m)?)?;
    collect_garbage(dir, &m);
    Ok(CompactReport {
        built: true,
        incremental,
        vectors: m.live(),
        segments: m.segments.len(),
        written,
        seconds: started.elapsed().as_secs_f64(),
        skipped: None,
    })
}

fn pairs(m: &Manifest) -> Vec<(usize, usize)> {
    m.segments.iter().map(|s| (s.count, s.deleted)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_plan() {
        let f = FLUSH_CHANGES as usize;
        // up to `cap` segments below full size: nothing to do
        assert_eq!(plan_merge_with(&[(f, 0); 4], 4), None);
        // one more: merge the smallest three
        let segs = [(5 * f, 0), (f, 0), (3 * f, 0), (2 * f, 0), (4 * f, 0)];
        assert_eq!(plan_merge_with(&segs, 4), Some(vec![1, 3, 2]));
        // far over the cap: merge enough to get back to it
        assert_eq!(plan_merge_with(&[(f, 0); 10], 4).unwrap().len(), 7);
        // a segment past the deletion threshold is vacuumed first
        assert_eq!(
            plan_merge_with(&[(f, 0), (100_000, 25_000)], 4),
            Some(vec![1])
        );
        // full-size segments do not count and are never merged
        let mut segs = vec![(MAX_SEGMENT, 0); 6];
        segs.extend([(f, 0); 4]);
        assert_eq!(plan_merge_with(&segs, 4), None);
    }
}
