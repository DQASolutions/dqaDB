//! Vector projection (spec §14): one embedding per entity, built from its `rdfs:label`
//! and the literals of `dqadb_index: vector` predicates.
//!
//! Like the text index, it lags the graph and catches up by watermark, but embedding is
//! slow, so it never runs inside `assert`. `dqadb embed` (and later `dqadb serve`) runs it:
//! 1. without the write lock: find touched entities and fill the embedding cache
//!    (the slow part; writers are never blocked);
//! 2. with the write lock, briefly: re-read the texts at the current watermark, embed any
//!    stragglers, and store the vectors plus the new watermark in one transaction.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashSet};
use std::time::Instant;
use ulid::Ulid;

use crate::embed::{self, Cache};
use crate::kb::Kb;
use crate::ontology::Ontology;
use crate::rdf::{self, Term};
use crate::store::{Id, Reader, VecRow, VecState};
use crate::{Error, Result, event, fts};

/// Bump when the text recipe changes (forces re-embedding from cache).
const VEC_FORMAT: u32 = 1;

#[derive(Debug, Default, Serialize)]
pub struct EmbedReport {
    pub entities_checked: usize,
    /// HNSW base (re)built after embedding
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compaction: Option<crate::ann::CompactReport>,
    pub vectors_written: usize,
    pub vectors_removed: usize,
    /// texts sent to the model (cache misses)
    pub model_calls: usize,
    pub full_rebuild: bool,
    pub seconds: f64,
}

/// Predicates whose literals are embedded: `rdfs:label` plus `dqadb_index: vector` slots.
pub fn vector_predicates(onto: &Ontology) -> Vec<String> {
    let mut v: Vec<String> = onto
        .slots
        .values()
        .filter(|s| s.index.iter().any(|i| i == "vector"))
        .map(|s| s.iri.clone())
        .collect();
    v.push(rdf::RDFS_LABEL.to_string());
    v.sort();
    v.dedup();
    v
}

fn config_hash(model_id: &str, preds: &[String]) -> String {
    let mut h = Sha256::new();
    h.update(format!("vec{VEC_FORMAT}\n{model_id}\n"));
    for p in preds {
        h.update(format!("{p}\n"));
    }
    format!("sha256:{}", hex::encode(h.finalize()))
}

/// The text embedded for an entity (empty = no vector).
fn entity_text(r: &Reader, s: Id, pred_ids: &[Id]) -> Result<String> {
    let mut parts = Vec::new();
    for p in pred_ids {
        for o in r.objects(s, *p)? {
            if let Term::Literal(l) = r.term(o)? {
                parts.push(l.value);
            }
        }
    }
    Ok(parts.join("\n"))
}

fn pred_ids(r: &Reader, preds: &[String]) -> Result<Vec<Id>> {
    let mut out = Vec::new();
    for p in preds {
        if let Some(id) = r.term_id(&Term::Iri(p.clone()))? {
            out.push(id);
        }
    }
    Ok(out)
}

/// Entities to (re)consider: everything on a config change, else those touched since `from`.
fn touched(
    kb: &Kb,
    r: &Reader,
    preds: &[String],
    from: Option<&VecState>,
    config: &str,
    upto: Option<Ulid>,
) -> Result<(BTreeSet<Id>, bool)> {
    let from_wm = from
        .and_then(|s| s.watermark.as_deref())
        .map(event::parse_ulid)
        .transpose()?;
    let incremental = from.is_some_and(|s| s.config == config) && from_wm <= upto;
    if incremental {
        let names: HashSet<String> = preds.iter().cloned().collect();
        Ok((
            fts::touched_subjects(r, &kb.events_dir(), from_wm, upto, &names)?,
            false,
        ))
    } else {
        let mut all = BTreeSet::new();
        for p in pred_ids(r, preds)? {
            for t in r.scan(None, Some(p), None) {
                all.insert(t?.0);
            }
        }
        Ok((all, true))
    }
}

impl Kb {
    fn cache(&self, model_id: &str) -> Result<Cache> {
        Cache::open(&self.root.join("emb-cache"), model_id)
    }

    /// Bring the vector projection up to date, then rebuild the HNSW base if the delta
    /// has grown (or `force_compact`).
    pub fn embed(&self, force_compact: bool) -> Result<EmbedReport> {
        let mut report = self.embed_sync()?;
        let store = self.store_handle()?;
        let dir = crate::ann::dir(&self.index_dir());
        let wanted = {
            let r = store.read()?;
            let state = r.vec_state()?;
            (force_compact && r.vec_count()? > 0)
                || crate::ann::should_build(&dir, &r, state.as_ref())?
        };
        if wanted {
            report.compaction = Some(crate::ann::build(
                store.path(),
                &dir,
                self.config.embedding.dim as usize,
            )?);
        }
        Ok(report)
    }

    /// Bring the vector projection up to date with the index (see module docs).
    pub fn embed_sync(&self) -> Result<EmbedReport> {
        let started = Instant::now();
        self.sync()?;
        let onto = self.ontology()?;
        let preds = vector_predicates(&onto);
        let model = &self.embedder;
        let model_id = model.id();
        let config = config_hash(&model_id, &preds);
        let batch = self.config.embedding.batch_size;
        let mut cache = self.cache(&model_id)?;
        let store = self.store_handle()?;
        let mut report = EmbedReport::default();

        // phase 1: no lock, fill the cache (slow)
        {
            let r = store.read()?;
            let state = r.vec_state()?;
            let (ids, _) = touched(self, &r, &preds, state.as_ref(), &config, r.watermark()?)?;
            let pids = pred_ids(&r, &preds)?;
            let texts: Vec<String> = ids
                .iter()
                .map(|s| entity_text(&r, *s, &pids))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .filter(|t| !t.is_empty())
                .collect();
            drop(r);
            report.model_calls += count_misses(&cache, &model_id, &texts)?;
            embed::embed_cached(&mut cache, model, &texts, batch)?;
        }

        // phase 2: write lock, catch up and store (fast: nearly everything is cached)
        let _lock = self.lock()?;
        let inference = onto.inference();
        self.sync_locked(&store, &inference)?;
        let r = store.read()?;
        let wm = r.watermark()?;
        let state = r.vec_state()?;
        let want = VecState {
            watermark: wm.map(|u| u.to_string()),
            config: config.clone(),
        };
        if state.as_ref() == Some(&want) {
            report.seconds = started.elapsed().as_secs_f64();
            return Ok(report);
        }
        let (ids, full) = touched(self, &r, &preds, state.as_ref(), &config, wm)?;
        report.full_rebuild = full;
        report.entities_checked = ids.len();
        let pids = pred_ids(&r, &preds)?;
        let mut texts = Vec::new();
        let mut keyed = Vec::new();
        for s in &ids {
            let text = entity_text(&r, *s, &pids)?;
            keyed.push((*s, text.clone()));
            if !text.is_empty() {
                texts.push(text);
            }
        }
        report.model_calls += count_misses(&cache, &model_id, &texts)?;
        let vecs = embed::embed_cached(&mut cache, model, &texts, batch)?;
        let mut vecs = vecs.into_iter();
        let mut updates: Vec<(Id, Option<VecRow>)> = Vec::new();
        for (s, text) in keyed {
            if text.is_empty() {
                if full || r.vec_hash(s)?.is_some() {
                    updates.push((s, None));
                    report.vectors_removed += 1;
                }
                continue;
            }
            let v = vecs.next().expect("one vector per non-empty text");
            let hash = embed::text_key(&model_id, &text);
            if !full && r.vec_hash(s)?.as_deref() == Some(&hash[..]) {
                continue;
            }
            updates.push((
                s,
                Some(VecRow {
                    hash,
                    vec: embed::to_f16_bytes(&v),
                }),
            ));
            report.vectors_written += 1;
        }
        drop(r);
        store.vec_write(&updates, &want, full)?;
        report.seconds = started.elapsed().as_secs_f64();
        Ok(report)
    }

    /// Events not yet reflected in the vector projection.
    pub fn vec_pending(&self) -> Result<usize> {
        let store = self.store_handle()?;
        let wm = match store.read()?.vec_state()? {
            Some(s) => s.watermark.as_deref().map(event::parse_ulid).transpose()?,
            None => None,
        };
        Ok(event::list_after(&self.events_dir(), wm)?.len())
    }

    /// Embed a query text: through a running `dqadb serve` when there is one (model
    /// already in memory), otherwise with the model loaded in this process.
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let texts = [text.to_string()];
        if let Some(remote) =
            crate::serve::embed_remote(&self.index_dir(), &self.embedder.id(), &texts)
        {
            return remote?
                .into_iter()
                .next()
                .ok_or_else(|| Error::other("empty embedding"));
        }
        self.embed_texts_local(&texts)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::other("empty embedding"))
    }

    /// Embed texts with the model in this process (used by `dqadb serve` itself).
    pub fn embed_texts_local(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embedder.get()?.embed(texts)
    }

    /// Model id of the configured embedder, and whether it is loaded in this process.
    pub fn embedder_status(&self) -> (String, bool) {
        (self.embedder.id(), self.embedder.loaded())
    }
}

fn count_misses(cache: &Cache, model_id: &str, texts: &[String]) -> Result<usize> {
    let mut n = 0;
    for t in texts {
        if cache.get(&embed::text_key(model_id, t))?.is_none() {
            n += 1;
        }
    }
    Ok(n)
}

/// Cosine ranking. With `candidates` (structured filters) it is exact over those; without,
/// it uses the HNSW base + delta when one is valid, else an exact scan.
/// Returns the top `n` and how many vectors were considered.
pub fn rank(
    r: &Reader,
    index_dir: &std::path::Path,
    query: &[f32],
    candidates: Option<&[Id]>,
    n: usize,
) -> Result<(Vec<(Id, f32)>, usize)> {
    let large = candidates.is_none_or(|c| c.len() > crate::ann::EXACT_MAX_CANDIDATES);
    if large {
        let dir = crate::ann::dir(index_dir);
        if let Some(meta) = crate::ann::usable(&dir, r, r.vec_state()?.as_ref())? {
            let allow: Option<std::collections::HashSet<u64>> =
                candidates.map(|c| c.iter().copied().collect());
            return crate::ann::search(&dir, &meta, r, query, n, allow.as_ref());
        }
    }
    let mut scored: Vec<(Id, f32)> = Vec::new();
    let (mut bits, mut buf) = (Vec::new(), Vec::new());
    let mut visit =
        |id: Id, blob: &[u8]| scored.push((id, embed::dot_f16(query, blob, &mut bits, &mut buf)));
    match candidates {
        Some(ids) => r.vec_of(ids, &mut visit)?,
        None => r.vec_each(&mut visit)?,
    }
    let total = scored.len();
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    scored.truncate(n);
    Ok((scored, total))
}

/// Reciprocal Rank Fusion (k = 60) of several ranked lists.
pub fn rrf(lists: &[Vec<Id>], n: usize) -> Vec<(Id, f32)> {
    let mut score: std::collections::HashMap<Id, f32> = std::collections::HashMap::new();
    for list in lists {
        for (rank, id) in list.iter().enumerate() {
            *score.entry(*id).or_default() += 1.0 / (60.0 + rank as f32 + 1.0);
        }
    }
    let mut v: Vec<(Id, f32)> = score.into_iter().collect();
    v.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    v.truncate(n);
    v
}
