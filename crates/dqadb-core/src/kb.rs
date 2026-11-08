//! A knowledge base directory: ties the source of truth (events, ontology, sources)
//! to the derived index in `.dqadb/`.

use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use ulid::Ulid;

use crate::config::{CONFIG_FILE, Config};
use crate::event::{self, EVENT_SCHEMA, Event, Op, OpKind, Source};
use crate::lock::WriteLock;
use crate::ontology::{self, Inference, InputOp, Ontology};
use crate::rdf::{self, Term};
use crate::sparql;
use crate::store::{Counts, STORE_FORMAT, Store};
use crate::{Error, Result};

const REPLAY_CHUNK: usize = 2048;
const SAMPLE_SCHEMA: &str = include_str!("sample_schema.linkml.yaml");
const GITIGNORE: &str = "# Derived index: rebuild with `dqadb rebuild`\n.dqadb/\n\n# Source of truth / caches: too many files for git, back them up with rsync or restic\nevents/\nsources/\nemb-cache/\n";

pub struct Kb {
    pub root: PathBuf,
    pub config: Config,
    store: Mutex<Option<Arc<Store>>>,
    pub lock_timeout: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct WriteOpts {
    pub actor: String,
    pub source: Option<PathBuf>,
    pub note: Option<String>,
    pub expected_version: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct WriteResult {
    pub event_id: String,
    pub path: PathBuf,
    pub ops: usize,
    pub added: u64,
    pub retracted: u64,
    pub id_map: std::collections::BTreeMap<String, String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct Manifest {
    pub dqadb: u32,
    pub watermark: Option<String>,
    pub ontology: Option<String>,
    pub embedding: crate::config::EmbeddingConfig,
    pub indexes: serde_json::Value,
    pub counts: Counts,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fingerprint: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RebuildReport {
    pub events: usize,
    pub counts: Counts,
    pub fingerprint: String,
    pub seconds: f64,
}

#[derive(Debug, Serialize)]
pub struct Status {
    pub root: PathBuf,
    pub watermark: Option<String>,
    pub latest_event: Option<String>,
    pub in_sync: bool,
    pub ontology: String,
    pub counts: Counts,
}

#[derive(Debug, Serialize)]
pub struct VerifyReport {
    pub events: usize,
    pub indexed_events: u64,
    pub problems: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct EventSummary {
    pub event_id: String,
    pub ts: String,
    pub actor: String,
    pub ops: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl From<&Event> for EventSummary {
    fn from(e: &Event) -> Self {
        Self {
            event_id: e.event_id.clone(),
            ts: e.ts.clone(),
            actor: e.actor.clone(),
            ops: e.ops.len(),
            source: e.source.as_ref().map(|s| s.path.clone()),
            note: e.note.clone(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct PropValue {
    pub p: String,
    pub value: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub is_iri: bool,
}

/// A type of an entity; `inferred` = entailed by RDFS only, never asserted by an event.
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct TypeRef {
    pub iri: String,
    pub inferred: bool,
}

#[derive(Debug, Serialize)]
pub struct Card {
    pub iri: String,
    pub label: Option<String>,
    pub types: Vec<TypeRef>,
    pub props: Vec<PropValue>,
    pub incoming: Vec<PropValue>,
    pub incoming_total: usize,
    pub last_event: Option<EventSummary>,
}

impl Kb {
    // ---------- layout ----------

    pub fn events_dir(&self) -> PathBuf {
        self.root.join("events")
    }
    pub fn ontology_dir(&self) -> PathBuf {
        self.root.join("ontology")
    }
    pub fn generated_dir(&self) -> PathBuf {
        self.ontology_dir().join("_generated")
    }
    pub fn sources_dir(&self) -> PathBuf {
        self.root.join("sources")
    }
    pub fn index_dir(&self) -> PathBuf {
        self.root.join(".dqadb")
    }
    fn store_dir(&self) -> PathBuf {
        self.index_dir().join("store")
    }
    fn main_schema(&self) -> PathBuf {
        self.root.join(&self.config.ontology_main)
    }

    // ---------- open / init ----------

    pub fn init(root: &Path, base_iri: &str) -> Result<Kb> {
        if root.join(CONFIG_FILE).exists() {
            return Err(Error::other(format!(
                "{} already contains a knowledge base",
                root.display()
            )));
        }
        for d in ["events", "ontology", "sources", "emb-cache"] {
            std::fs::create_dir_all(root.join(d))?;
        }
        let config = Config::new(base_iri);
        config.save(root)?;
        let main = root.join(&config.ontology_main);
        if !main.exists() {
            std::fs::write(main, SAMPLE_SCHEMA)?;
        }
        let gi = root.join(".gitignore");
        if !gi.exists() {
            std::fs::write(gi, GITIGNORE)?;
        }
        Self::at(root.to_path_buf(), config)
    }

    /// Open the KB containing `start` (searching parent directories).
    pub fn open(start: &Path) -> Result<Kb> {
        let start = std::path::absolute(start)?;
        for dir in start.ancestors() {
            if dir.join(CONFIG_FILE).exists() {
                return Self::at(dir.to_path_buf(), Config::load(dir)?);
            }
        }
        Err(Error::NotAKb(start))
    }

    fn at(root: PathBuf, config: Config) -> Result<Kb> {
        if config.dqadb != 1 {
            return Err(Error::other(format!(
                "unsupported dqadb.yaml version {}",
                config.dqadb
            )));
        }
        Ok(Kb {
            root,
            config,
            store: Mutex::new(None),
            lock_timeout: Duration::from_secs(10),
        })
    }

    fn lock(&self) -> Result<WriteLock> {
        std::fs::create_dir_all(self.events_dir())?;
        WriteLock::acquire(&self.events_dir().join(".lock"), self.lock_timeout)
    }

    fn store(&self) -> Result<Arc<Store>> {
        let mut guard = self.store.lock().unwrap();
        if let Some(s) = guard.as_ref() {
            return Ok(s.clone());
        }
        let s = Arc::new(Store::open(&self.store_dir())?);
        *guard = Some(s.clone());
        Ok(s)
    }

    fn close_store(&self) {
        if let Some(s) = self.store.lock().unwrap().take()
            && let Ok(s) = Arc::try_unwrap(s)
        {
            s.close();
        }
    }

    pub fn ontology(&self) -> Result<Ontology> {
        Ontology::load(
            &self.ontology_dir(),
            &self.main_schema(),
            &self.config.base_iri,
        )
    }

    pub fn build_ontology(&self, lock_only: bool) -> Result<ontology::BuildLock> {
        ontology::build(
            &self.ontology_dir(),
            &self.main_schema(),
            &self.config.base_iri,
            lock_only,
        )
    }

    // ---------- sync / replay ----------

    /// Apply events newer than the index watermark (e.g. written by another machine
    /// and copied in). Called before every read so answers never lag the files.
    pub fn sync(&self) -> Result<usize> {
        let store = self.store()?;
        let inference = self.ontology()?.inference();
        let (wm, rules) = {
            let r = store.read_txn()?;
            (store.watermark(&r)?, store.inference_hash(&r)?)
        };
        let latest = event::latest(&self.events_dir())?;
        if latest <= wm && rules.as_deref() == Some(inference.hash.as_str()) {
            return Ok(0);
        }
        let _lock = self.lock()?;
        let n = self.sync_locked(&store, &inference)?;
        self.write_manifest(&store, None)?;
        Ok(n)
    }

    fn sync_locked(&self, store: &Store, inference: &Inference) -> Result<usize> {
        let (wm, rules) = {
            let r = store.read_txn()?;
            (store.watermark(&r)?, store.inference_hash(&r)?)
        };
        let files = event::list_after(&self.events_dir(), wm)?;
        if files.is_empty() && rules.as_deref() != Some(inference.hash.as_str()) {
            // ontology hierarchy changed: opening a batch recomputes inferences
            store.batch(inference)?.commit()?;
        }
        replay(store, &files, inference)
    }

    /// Rebuild `.dqadb/` from scratch into a temp dir, then swap it in.
    pub fn rebuild(&self) -> Result<RebuildReport> {
        let started = Instant::now();
        let _lock = self.lock()?;
        self.close_store();
        let tmp_root = self.index_dir().join("tmp");
        let tmp = tmp_root.join("store");
        if tmp_root.exists() {
            std::fs::remove_dir_all(&tmp_root)?;
        }
        let files = event::list_after(&self.events_dir(), None)?;
        let inference = self.ontology()?.inference();
        let (counts, fingerprint) = {
            let store = Store::open(&tmp)?;
            replay(&store, &files, &inference)?;
            store.batch(&inference)?.commit()?;
            let counts = store.counts(&*store.read_txn()?)?;
            let fp = store.fingerprint()?;
            store.close();
            (counts, fp)
        };
        let live = self.store_dir();
        let old = self.index_dir().join("store.old");
        if old.exists() {
            std::fs::remove_dir_all(&old)?;
        }
        if live.exists() {
            std::fs::rename(&live, &old)?;
        }
        std::fs::rename(&tmp, &live)?;
        std::fs::remove_dir_all(&tmp_root)?;
        if old.exists() {
            std::fs::remove_dir_all(&old)?;
        }
        let store = self.store()?;
        self.write_manifest(&store, Some(fingerprint.clone()))?;
        Ok(RebuildReport {
            events: files.len(),
            counts,
            fingerprint,
            seconds: started.elapsed().as_secs_f64(),
        })
    }

    fn write_manifest(&self, store: &Store, fingerprint: Option<String>) -> Result<()> {
        let r = store.read_txn()?;
        let m = Manifest {
            dqadb: 1,
            watermark: store.watermark(&r)?.map(|u| u.to_string()),
            ontology: ontology::read_lock(&self.generated_dir())?.map(|l| l.hash),
            embedding: self.config.embedding.clone(),
            indexes: serde_json::json!({ "store": { "format": "lmdb", "version": STORE_FORMAT } }),
            counts: store.counts(&r)?,
            fingerprint,
        };
        std::fs::write(
            self.index_dir().join("manifest.json"),
            serde_json::to_vec_pretty(&m)?,
        )?;
        Ok(())
    }

    // ---------- writes ----------

    /// Validate input ops against the ontology and current state without writing.
    pub fn validate(&self, input: &[InputOp]) -> Result<ontology::Normalized> {
        ontology::check_fresh(&self.ontology_dir(), &self.generated_dir())?;
        let onto = self.ontology()?;
        self.sync()?;
        let store = self.store()?;
        let r = store.read_txn()?;
        let base = self.config.base_iri.clone();
        let mut mint = || format!("{base}{}", Ulid::generate());
        let existing = |s: &str, p: &str| values(&store, &r, s, p).unwrap_or_default();
        onto.normalize(input, &mut mint, &existing)
            .map_err(Error::Validation)
    }

    pub fn assert(&self, input: &[InputOp], opts: &WriteOpts) -> Result<WriteResult> {
        let _lock = self.lock()?;
        let hash = ontology::check_fresh(&self.ontology_dir(), &self.generated_dir())?;
        let onto = self.ontology()?;
        let store = self.store()?;
        let inference = onto.inference();
        self.sync_locked(&store, &inference)?;
        let latest = self.check_expected(opts)?;

        let batch = store.batch(&inference)?;
        let base = self.config.base_iri.clone();
        let mut mint = || format!("{base}{}", Ulid::generate());
        let existing = |s: &str, p: &str| batch.values(s, p).unwrap_or_default();
        let norm = onto
            .normalize(input, &mut mint, &existing)
            .map_err(Error::Validation)?;
        self.commit(
            &store,
            batch,
            latest,
            hash,
            norm.ops,
            norm.id_map,
            norm.warnings,
            opts,
        )
    }

    /// Write an event that retracts every triple added by `event_id`.
    pub fn retract_event(
        &self,
        event_id: &str,
        why: &str,
        opts: &WriteOpts,
    ) -> Result<WriteResult> {
        let _lock = self.lock()?;
        let hash = ontology::check_fresh(&self.ontology_dir(), &self.generated_dir())?;
        let target = event::read_event(&event::find_event(
            &self.events_dir(),
            event::parse_ulid(event_id)?,
        )?)?;
        let ops: Vec<Op> = target
            .ops
            .iter()
            .filter(|o| o.op == OpKind::Add)
            .map(|o| Op {
                op: OpKind::Retract,
                ..o.clone()
            })
            .collect();
        if ops.is_empty() {
            return Err(Error::other(format!(
                "event {event_id} has no add ops to retract"
            )));
        }
        let store = self.store()?;
        let inference = self.ontology()?.inference();
        self.sync_locked(&store, &inference)?;
        let latest = self.check_expected(opts)?;
        let batch = store.batch(&inference)?;
        let mut opts = opts.clone();
        opts.note = Some(format!("retract event {event_id}: {why}"));
        self.commit(
            &store,
            batch,
            latest,
            hash,
            ops,
            Default::default(),
            Vec::new(),
            &opts,
        )
    }

    fn check_expected(&self, opts: &WriteOpts) -> Result<Option<Ulid>> {
        let latest = event::latest(&self.events_dir())?;
        if let Some(exp) = &opts.expected_version {
            let actual = latest.map(|u| u.to_string()).unwrap_or_default();
            if *exp != actual {
                return Err(Error::VersionConflict {
                    expected: exp.clone(),
                    actual,
                });
            }
        }
        Ok(latest)
    }

    #[allow(clippy::too_many_arguments)]
    fn commit(
        &self,
        store: &Store,
        mut batch: crate::store::Batch<'_>,
        latest: Option<Ulid>,
        ontology_hash: String,
        ops: Vec<Op>,
        id_map: std::collections::BTreeMap<String, String>,
        mut warnings: Vec<String>,
        opts: &WriteOpts,
    ) -> Result<WriteResult> {
        let source = match &opts.source {
            Some(p) => Some(self.store_source(p, &mut warnings)?),
            None => None,
        };
        let id = next_ulid(latest);
        let ev = Event {
            dqadb: EVENT_SCHEMA,
            event_id: id.to_string(),
            ts: event::ulid_ts(id),
            actor: opts.actor.clone(),
            source,
            ontology: ontology_hash,
            note: opts.note.clone(),
            ops,
        };
        let path = event::write_atomic(&self.events_dir(), &ev)?;
        // The event is durable now; if indexing fails, the next command catches up.
        let stats = batch.apply(&ev)?;
        batch.commit()?;
        self.write_manifest(store, None)?;
        Ok(WriteResult {
            event_id: ev.event_id,
            path,
            ops: ev.ops.len(),
            added: stats.added,
            retracted: stats.retracted,
            id_map,
            warnings,
        })
    }

    /// Copy a source file into `sources/sha256/ab/cd/<hash>.<ext>` (content-addressed).
    fn store_source(&self, path: &Path, warnings: &mut Vec<String>) -> Result<Source> {
        let shown = path.to_string_lossy().to_string();
        if !path.is_file() {
            warnings.push(format!(
                "source file {shown} not found; recorded by name only"
            ));
            return Ok(Source {
                path: shown,
                sha256: None,
            });
        }
        let bytes = std::fs::read(path)?;
        let h = hex::encode(Sha256::digest(&bytes));
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| format!(".{e}"))
            .unwrap_or_default();
        let dest = self
            .sources_dir()
            .join("sha256")
            .join(&h[..2])
            .join(&h[2..4])
            .join(format!("{h}{ext}"));
        if !dest.exists() {
            std::fs::create_dir_all(dest.parent().unwrap())?;
            std::fs::write(&dest, &bytes)?;
        }
        Ok(Source {
            path: shown,
            sha256: Some(h),
        })
    }

    pub fn new_ids(&self, n: usize) -> Vec<String> {
        (0..n)
            .map(|_| format!("dqadb:{}", Ulid::generate()))
            .collect()
    }

    // ---------- reads ----------

    pub fn status(&self) -> Result<Status> {
        let store = self.store()?;
        let r = store.read_txn()?;
        let wm = store.watermark(&r)?;
        let latest = event::latest(&self.events_dir())?;
        let ontology = match ontology::check_fresh(&self.ontology_dir(), &self.generated_dir()) {
            Ok(h) => format!("ok ({h})"),
            Err(Error::OntologyMissing) => "not built".into(),
            Err(Error::OntologyStale) => "stale (run: dqadb ontology build)".into(),
            Err(e) => return Err(e),
        };
        Ok(Status {
            root: self.root.clone(),
            watermark: wm.map(|u| u.to_string()),
            latest_event: latest.map(|u| u.to_string()),
            in_sync: latest <= wm,
            ontology,
            counts: store.counts(&r)?,
        })
    }

    /// Check every event file: parses, name matches id, path matches sharding rule.
    pub fn verify(&self) -> Result<VerifyReport> {
        let files = event::list_after(&self.events_dir(), None)?;
        let problems: Vec<String> = files
            .par_iter()
            .filter_map(|(id, path)| {
                let ev = match event::read_event(path) {
                    Ok(ev) => ev,
                    Err(e) => return Some(e.to_string()),
                };
                let shown = path.display();
                if ev.event_id != id.to_string() {
                    return Some(format!(
                        "{shown}: event_id {} does not match file name",
                        ev.event_id
                    ));
                }
                if !path.ends_with(event::rel_path(*id)) {
                    return Some(format!(
                        "{shown}: not at its sharded path {}",
                        event::rel_path(*id).display()
                    ));
                }
                if ev.dqadb > EVENT_SCHEMA {
                    return Some(format!(
                        "{shown}: event schema {} is newer than this binary",
                        ev.dqadb
                    ));
                }
                if ev.ops.is_empty() {
                    return Some(format!("{shown}: no ops"));
                }
                None
            })
            .collect();
        let store = self.store()?;
        let indexed = store.counts(&*store.read_txn()?)?.events;
        Ok(VerifyReport {
            events: files.len(),
            indexed_events: indexed,
            problems,
        })
    }

    pub fn event(&self, id: &str) -> Result<Event> {
        event::read_event(&event::find_event(
            &self.events_dir(),
            event::parse_ulid(id)?,
        )?)
    }

    pub fn events_ls(
        &self,
        limit: usize,
        actor: Option<&str>,
        since: Option<&str>,
    ) -> Result<Vec<EventSummary>> {
        let since = since.map(event::parse_day).transpose()?;
        let mut out = Vec::new();
        for item in event::list_recent(&self.events_dir(), since)? {
            let (_, path) = item?;
            let ev = event::read_event(&path)?;
            if actor.is_some_and(|a| ev.actor != a) {
                continue;
            }
            out.push(EventSummary::from(&ev));
            if out.len() >= limit {
                break;
            }
        }
        Ok(out)
    }

    pub fn get(&self, iri: &str) -> Result<Card> {
        self.sync()?;
        let onto = self.ontology()?;
        let full = onto.prefixes.expand(iri).map_err(Error::Other)?;
        let store = self.store()?;
        let r = store.read_txn()?;
        let not_found = || Error::NotFound(format!("entity {iri}"));
        let sid = store
            .term_id(&r, &Term::Iri(full.clone()))?
            .ok_or_else(not_found)?;
        let out = store.outgoing(&r, sid)?;
        let inc = store.incoming(&r, sid)?;
        if out.is_empty() && inc.is_empty() {
            return Err(not_found());
        }
        let label_pid = store.term_id(&r, &Term::Iri(rdf::RDFS_LABEL.into()))?;
        let label_of = |id: u64| -> Result<Option<String>> {
            let Some(lp) = label_pid else { return Ok(None) };
            Ok(match store.objects(&r, id, lp)?.first() {
                Some(o) => Some(match store.term(&r, *o)? {
                    Term::Literal(l) => l.value,
                    t => onto.prefixes.display(&t),
                }),
                None => None,
            })
        };
        let mut card = Card {
            iri: onto.compact(&full),
            label: label_of(sid)?,
            types: Vec::new(),
            props: Vec::new(),
            incoming: Vec::new(),
            incoming_total: inc.len(),
            last_event: None,
        };
        let mut last: Option<Ulid> = None;
        for (p, o) in &out {
            let p_term = store.term(&r, *p)?;
            let o_term = store.term(&r, *o)?;
            let prov = store.provenance(&r, sid, *p, *o)?;
            last = last.max(prov.iter().copied().max());
            let p_iri = p_term.as_iri().unwrap_or_default();
            if p_iri == rdf::RDF_TYPE {
                card.types.push(TypeRef {
                    iri: onto.prefixes.display(&o_term),
                    inferred: prov.is_empty(),
                });
                continue;
            }
            if p_iri == rdf::RDFS_LABEL {
                continue;
            }
            card.props.push(PropValue {
                p: onto.compact(p_iri),
                value: onto.prefixes.display(&o_term),
                label: if o_term.as_iri().is_some() {
                    label_of(*o)?
                } else {
                    None
                },
                is_iri: o_term.as_iri().is_some(),
            });
        }
        for (s, p) in inc.iter().take(20) {
            card.incoming.push(PropValue {
                p: onto.compact(store.term(&r, *p)?.as_iri().unwrap_or_default()),
                value: onto.prefixes.display(&store.term(&r, *s)?),
                label: label_of(*s)?,
                is_iri: true,
            });
        }
        if let Some(id) = last {
            card.last_event = Some(EventSummary::from(&self.event(&id.to_string())?));
        }
        Ok(card)
    }

    /// Fingerprint of the current index (for determinism checks).
    pub fn fingerprint(&self) -> Result<String> {
        self.store()?.fingerprint()
    }

    /// Run a SPARQL 1.1 query. The KB's prefixes are predeclared.
    pub fn query(&self, query: &str, limits: &sparql::Limits) -> Result<sparql::Answer> {
        self.sync()?;
        let onto = self.ontology()?;
        let store = self.store()?;
        let txn = store.read_txn()?;
        let event_ns = self.event_namespace();
        let events_dir = self.events_dir();
        let dataset = sparql::KbDataset::new(&store, &txn, &event_ns, &events_dir);
        sparql::run(dataset, &onto.prefixes, query, limits)
    }

    /// Named graph IRI prefix for events: `<base_iri>event:`.
    pub fn event_namespace(&self) -> String {
        format!("{}event:", self.config.base_iri)
    }
}

fn values(store: &Store, r: &heed::RoTxn, s: &str, p: &str) -> Result<Vec<Term>> {
    let (Some(s), Some(p)) = (
        store.term_id(r, &Term::Iri(s.into()))?,
        store.term_id(r, &Term::Iri(p.into()))?,
    ) else {
        return Ok(Vec::new());
    };
    store
        .objects(r, s, p)?
        .into_iter()
        .map(|o| store.term(r, o))
        .collect()
}

/// Monotonic ULID: never at or before the newest event on disk, even if the clock went back.
fn next_ulid(latest: Option<Ulid>) -> Ulid {
    let now = Ulid::generate();
    match latest {
        Some(l) if now <= l => l.increment().expect("ULID space exhausted"),
        _ => now,
    }
}

/// Parse files in parallel, apply strictly in ULID order, commit per chunk.
fn replay(store: &Store, files: &[(Ulid, PathBuf)], inference: &Inference) -> Result<usize> {
    for chunk in files.chunks(REPLAY_CHUNK) {
        let events: Vec<Event> = chunk
            .par_iter()
            .map(|(_, p)| event::read_event(p))
            .collect::<Result<_>>()?;
        let mut batch = store.batch(inference)?;
        for ((id, path), ev) in chunk.iter().zip(&events) {
            if ev.event_id != id.to_string() {
                return Err(Error::other(format!(
                    "{}: event_id does not match file name",
                    path.display()
                )));
            }
            batch.apply(ev)?;
        }
        batch.commit()?;
    }
    Ok(files.len())
}
