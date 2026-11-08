//! MVP index store on LMDB (via heed): dictionary, SPO/POS/OSP permutations, provenance,
//! and materialised RDFS inferences. LMDB allows many reader processes alongside one
//! writer, which a CLI-driven harness (plus `dqadb serve`) needs.
//! Everything here is derived and can be rebuilt from events.
//!
//! A triple is in `spo`/`pos`/`osp` when it is asserted (has at least one `prov` entry)
//! or inferred (has an `inf` entry), or both.

use heed::byteorder::BigEndian;
use heed::types::{Bytes, Str, U64, Unit};
use heed::{Database, Env, EnvOpenOptions, RoTxn, RwTxn, WithTls};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::Path;
use ulid::Ulid;

use crate::event::{Event, OpKind};
use crate::ontology::Inference;
use crate::rdf::{self, Term};
use crate::{Error, Result};

/// Bump when the on-disk layout changes; a mismatch forces `dqadb rebuild`.
/// 2: added the `inf` table (RDFS inferences).
pub const STORE_FORMAT: u32 = 2;
const MAP_SIZE: usize = 1 << 37; // 128 GiB of address space; disk use grows with data

pub type Id = u64;
pub type Txn<'a> = RoTxn<'a, WithTls>;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Counts {
    pub events: u64,
    /// distinct triples in the default graph (asserted or inferred)
    pub triples: u64,
    /// inferred triple entries (some may also be asserted)
    #[serde(default)]
    pub inferred: u64,
    pub terms: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ApplyStats {
    pub added: u64,
    pub retracted: u64,
    pub noop: u64,
}

pub struct Store {
    env: Env,
    meta: Database<Str, Bytes>,
    term2id: Database<Bytes, U64<BigEndian>>,
    id2term: Database<U64<BigEndian>, Bytes>,
    pub(crate) spo: Database<Bytes, Unit>,
    pub(crate) pos: Database<Bytes, Unit>,
    pub(crate) osp: Database<Bytes, Unit>,
    pub(crate) prov: Database<Bytes, Unit>,
    pub(crate) inf: Database<Bytes, Unit>,
}

pub fn key3(a: Id, b: Id, c: Id) -> [u8; 24] {
    let mut k = [0u8; 24];
    k[..8].copy_from_slice(&a.to_be_bytes());
    k[8..16].copy_from_slice(&b.to_be_bytes());
    k[16..].copy_from_slice(&c.to_be_bytes());
    k
}

pub fn key2(a: Id, b: Id) -> [u8; 16] {
    let mut k = [0u8; 16];
    k[..8].copy_from_slice(&a.to_be_bytes());
    k[8..].copy_from_slice(&b.to_be_bytes());
    k
}

pub fn split3(k: &[u8]) -> (Id, Id, Id) {
    let n = |i: usize| u64::from_be_bytes(k[i..i + 8].try_into().unwrap());
    (n(0), n(8), n(16))
}

/// Event id stored after the 24-byte triple key in `prov`.
pub fn prov_event(k: &[u8]) -> Ulid {
    Ulid(u128::from_be_bytes(k[24..40].try_into().unwrap()))
}

fn term_hash(t: &Term) -> [u8; 32] {
    Sha256::digest(t.ntriples().as_bytes()).into()
}

impl Store {
    pub fn open(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        // SAFETY: the env directory is only ever opened through this function, and dqaDB
        // never truncates or rewrites LMDB files outside of LMDB itself.
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(MAP_SIZE)
                .max_dbs(16)
                .open(dir)?
        };
        let mut w = env.write_txn()?;
        let meta: Database<Str, Bytes> = env.create_database(&mut w, Some("meta"))?;
        let s = Self {
            meta,
            term2id: env.create_database(&mut w, Some("term2id"))?,
            id2term: env.create_database(&mut w, Some("id2term"))?,
            spo: env.create_database(&mut w, Some("spo"))?,
            pos: env.create_database(&mut w, Some("pos"))?,
            osp: env.create_database(&mut w, Some("osp"))?,
            prov: env.create_database(&mut w, Some("prov"))?,
            inf: env.create_database(&mut w, Some("inf"))?,
            env: env.clone(),
        };
        match s.meta.get(&w, "format")? {
            None => s.meta.put(&mut w, "format", &STORE_FORMAT.to_be_bytes())?,
            Some(b) => {
                let v = u32::from_be_bytes(
                    b.try_into()
                        .map_err(|_| Error::IndexStale("bad format marker".into()))?,
                );
                if v != STORE_FORMAT {
                    return Err(Error::IndexStale(format!(
                        "store format {v}, this binary expects {STORE_FORMAT}"
                    )));
                }
            }
        }
        w.commit()?;
        Ok(s)
    }

    /// Close the environment so its directory can be replaced (used by rebuild).
    pub fn close(self) {
        let Self { env, .. } = self;
        env.prepare_for_closing().wait();
    }

    pub fn read_txn(&self) -> Result<Txn<'_>> {
        Ok(self.env.read_txn()?)
    }

    pub fn watermark(&self, r: &RoTxn) -> Result<Option<Ulid>> {
        Ok(self
            .meta
            .get(r, "watermark")?
            .map(|b| Ulid(u128::from_be_bytes(b.try_into().unwrap()))))
    }

    pub fn inference_hash(&self, r: &RoTxn) -> Result<Option<String>> {
        Ok(self
            .meta
            .get(r, "inference_hash")?
            .map(|b| String::from_utf8_lossy(b).into_owned()))
    }

    pub fn counts(&self, r: &RoTxn) -> Result<Counts> {
        Ok(match self.meta.get(r, "counts")? {
            Some(b) => serde_json::from_slice(b)?,
            None => Counts::default(),
        })
    }

    pub fn term_id(&self, r: &RoTxn, t: &Term) -> Result<Option<Id>> {
        Ok(self.term2id.get(r, &term_hash(t))?)
    }

    pub fn term(&self, r: &RoTxn, id: Id) -> Result<Term> {
        let b = self
            .id2term
            .get(r, &id)?
            .ok_or_else(|| Error::IndexStale(format!("dangling term id {id}")))?;
        Ok(serde_json::from_slice(b)?)
    }

    /// (p, o) pairs for subject s.
    pub fn outgoing(&self, r: &RoTxn, s: Id) -> Result<Vec<(Id, Id)>> {
        let mut out = Vec::new();
        for item in self.spo.prefix_iter(r, &s.to_be_bytes())? {
            let (_, p, o) = split3(item?.0);
            out.push((p, o));
        }
        Ok(out)
    }

    /// (s, p) pairs pointing at object o.
    pub fn incoming(&self, r: &RoTxn, o: Id) -> Result<Vec<(Id, Id)>> {
        let mut out = Vec::new();
        for item in self.osp.prefix_iter(r, &o.to_be_bytes())? {
            let (_, s, p) = split3(item?.0);
            out.push((s, p));
        }
        Ok(out)
    }

    pub fn objects(&self, r: &RoTxn, s: Id, p: Id) -> Result<Vec<Id>> {
        let mut out = Vec::new();
        for item in self.spo.prefix_iter(r, &key2(s, p))? {
            out.push(split3(item?.0).2);
        }
        Ok(out)
    }

    /// Events that asserted (s, p, o) and are still in effect, oldest first.
    pub fn provenance(&self, r: &RoTxn, s: Id, p: Id, o: Id) -> Result<Vec<Ulid>> {
        let mut out = Vec::new();
        for item in self.prov.prefix_iter(r, &key3(s, p, o))? {
            out.push(prov_event(item?.0));
        }
        Ok(out)
    }

    pub fn is_inferred(&self, r: &RoTxn, s: Id, p: Id, o: Id) -> Result<bool> {
        Ok(self.inf.get(r, &key3(s, p, o))?.is_some())
    }

    /// Hash over the dictionary, triples, provenance and inferences: equal = identical index.
    pub fn fingerprint(&self) -> Result<String> {
        let r = self.read_txn()?;
        let mut h = Sha256::new();
        for item in self.id2term.iter(&r)? {
            let (id, t) = item?;
            h.update(id.to_be_bytes());
            h.update(t);
        }
        for db in [self.spo, self.prov, self.inf] {
            h.update(b"|");
            for item in db.iter(&r)? {
                h.update(item?.0);
            }
        }
        Ok(format!("sha256:{}", hex::encode(h.finalize())))
    }

    /// Start a write transaction. If the inference rules changed since the index was
    /// built, all inferences are recomputed first.
    pub fn batch<'a>(&'a self, inference: &'a Inference) -> Result<Batch<'a>> {
        let txn = self.env.write_txn()?;
        let next_id = match self.meta.get(&txn, "next_id")? {
            Some(b) => u64::from_be_bytes(b.try_into().unwrap()),
            None => 1,
        };
        let counts = self.counts(&txn)?;
        let watermark = self.watermark(&txn)?;
        let stale = self.inference_hash(&txn)?.as_deref() != Some(inference.hash.as_str());
        let mut batch = Batch {
            store: self,
            txn,
            next_id,
            counts,
            watermark,
            inference,
        };
        if stale {
            batch.reinfer_all()?;
        }
        Ok(batch)
    }
}

/// One write transaction applying events in ULID order.
pub struct Batch<'a> {
    store: &'a Store,
    txn: RwTxn<'a>,
    next_id: Id,
    counts: Counts,
    watermark: Option<Ulid>,
    inference: &'a Inference,
}

impl<'a> Batch<'a> {
    fn id_for(&mut self, t: &Term) -> Result<Id> {
        let h = term_hash(t);
        if let Some(id) = self.store.term2id.get(&self.txn, &h)? {
            return Ok(id);
        }
        let id = self.next_id;
        self.next_id += 1;
        self.store.term2id.put(&mut self.txn, &h, &id)?;
        self.store
            .id2term
            .put(&mut self.txn, &id, &serde_json::to_vec(t)?)?;
        self.counts.terms += 1;
        Ok(id)
    }

    fn lookup(&self, t: &Term) -> Result<Option<Id>> {
        Ok(self.store.term2id.get(&self.txn, &term_hash(t))?)
    }

    fn has_prov(&self, k: &[u8; 24]) -> Result<bool> {
        Ok(self
            .store
            .prov
            .prefix_iter(&self.txn, k)?
            .next()
            .transpose()?
            .is_some())
    }

    fn put_triple(&mut self, s: Id, p: Id, o: Id) -> Result<bool> {
        let k = key3(s, p, o);
        if self.store.spo.get(&self.txn, &k)?.is_some() {
            return Ok(false);
        }
        self.store.spo.put(&mut self.txn, &k, &())?;
        self.store.pos.put(&mut self.txn, &key3(p, o, s), &())?;
        self.store.osp.put(&mut self.txn, &key3(o, s, p), &())?;
        self.counts.triples += 1;
        Ok(true)
    }

    fn delete_triple(&mut self, s: Id, p: Id, o: Id) -> Result<()> {
        if self.store.spo.delete(&mut self.txn, &key3(s, p, o))? {
            self.store.pos.delete(&mut self.txn, &key3(p, o, s))?;
            self.store.osp.delete(&mut self.txn, &key3(o, s, p))?;
            self.counts.triples -= 1;
        }
        Ok(())
    }

    pub fn apply(&mut self, ev: &Event) -> Result<ApplyStats> {
        let eid = ev.ulid()?;
        if let Some(w) = self.watermark.filter(|w| eid <= *w) {
            return Err(Error::IndexStale(format!(
                "event {eid} is older than the watermark {w}; it was added out of order"
            )));
        }
        let mut stats = ApplyStats::default();
        let mut retyped: BTreeSet<Id> = BTreeSet::new();
        for op in &ev.ops {
            let s_term = Term::Iri(op.s.clone());
            let p_term = Term::Iri(op.p.clone());
            match op.op {
                OpKind::Add => {
                    let (s, p, o) = (
                        self.id_for(&s_term)?,
                        self.id_for(&p_term)?,
                        self.id_for(&op.o)?,
                    );
                    let k = key3(s, p, o);
                    if self.has_prov(&k)? {
                        stats.noop += 1;
                    } else {
                        stats.added += 1;
                    }
                    self.put_triple(s, p, o)?;
                    let mut pk = [0u8; 40];
                    pk[..24].copy_from_slice(&k);
                    pk[24..].copy_from_slice(&eid.0.to_be_bytes());
                    self.store.prov.put(&mut self.txn, &pk, &())?;
                    if op.p == rdf::RDF_TYPE {
                        retyped.insert(s);
                    }
                }
                OpKind::Retract => {
                    let ids = (
                        self.lookup(&s_term)?,
                        self.lookup(&p_term)?,
                        self.lookup(&op.o)?,
                    );
                    let (Some(s), Some(p), Some(o)) = ids else {
                        stats.noop += 1;
                        continue;
                    };
                    let k = key3(s, p, o);
                    let keys: Vec<Vec<u8>> = self
                        .store
                        .prov
                        .prefix_iter(&self.txn, &k)?
                        .map(|r| r.map(|(k, _)| k.to_vec()))
                        .collect::<std::result::Result<_, _>>()?;
                    if keys.is_empty() {
                        // not asserted (possibly only inferred): nothing to retract
                        stats.noop += 1;
                        continue;
                    }
                    for pk in keys {
                        self.store.prov.delete(&mut self.txn, &pk)?;
                    }
                    if self.store.inf.get(&self.txn, &k)?.is_none() {
                        self.delete_triple(s, p, o)?;
                    }
                    stats.retracted += 1;
                    if op.p == rdf::RDF_TYPE {
                        retyped.insert(s);
                    }
                }
            }
        }
        for s in retyped {
            self.reinfer(s)?;
        }
        self.counts.events += 1;
        self.watermark = Some(eid);
        Ok(stats)
    }

    /// Recompute the inferred types of one subject from its asserted types.
    fn reinfer(&mut self, s: Id) -> Result<()> {
        let Some(type_id) = self.lookup(&Term::Iri(rdf::RDF_TYPE.into()))? else {
            return Ok(());
        };
        let mut asserted = BTreeSet::new();
        for o in self.store.objects(&self.txn, s, type_id)? {
            if self.has_prov(&key3(s, type_id, o))? {
                asserted.insert(o);
            }
        }
        let mut wanted = BTreeSet::new();
        for o in &asserted {
            let Term::Iri(class) = self.store.term(&self.txn, *o)? else {
                continue;
            };
            for sup in self.inference.superclasses(&class).to_vec() {
                let id = self.id_for(&Term::Iri(sup))?;
                if !asserted.contains(&id) {
                    wanted.insert(id);
                }
            }
        }
        let mut current = BTreeSet::new();
        for item in self.store.inf.prefix_iter(&self.txn, &key2(s, type_id))? {
            current.insert(split3(item?.0).2);
        }
        for o in current.difference(&wanted) {
            let k = key3(s, type_id, *o);
            self.store.inf.delete(&mut self.txn, &k)?;
            self.counts.inferred -= 1;
            if !self.has_prov(&k)? {
                self.delete_triple(s, type_id, *o)?;
            }
        }
        for o in wanted.difference(&current) {
            self.store
                .inf
                .put(&mut self.txn, &key3(s, type_id, *o), &())?;
            self.counts.inferred += 1;
            self.put_triple(s, type_id, *o)?;
        }
        Ok(())
    }

    /// Recompute every inference (the ontology's class hierarchy changed).
    fn reinfer_all(&mut self) -> Result<()> {
        let mut subjects = BTreeSet::new();
        if let Some(type_id) = self.lookup(&Term::Iri(rdf::RDF_TYPE.into()))? {
            for item in self
                .store
                .pos
                .prefix_iter(&self.txn, &type_id.to_be_bytes())?
            {
                subjects.insert(split3(item?.0).2);
            }
        }
        for item in self.store.inf.iter(&self.txn)? {
            subjects.insert(split3(item?.0).0);
        }
        for s in subjects {
            self.reinfer(s)?;
        }
        Ok(())
    }

    /// Current values of (s, p), including changes applied earlier in this batch.
    pub fn values(&self, s: &str, p: &str) -> Result<Vec<Term>> {
        let (Some(s), Some(p)) = (
            self.lookup(&Term::Iri(s.into()))?,
            self.lookup(&Term::Iri(p.into()))?,
        ) else {
            return Ok(Vec::new());
        };
        self.store
            .objects(&self.txn, s, p)?
            .into_iter()
            .map(|o| self.store.term(&self.txn, o))
            .collect()
    }

    pub fn counts(&self) -> Counts {
        self.counts
    }

    pub fn commit(mut self) -> Result<()> {
        let m = self.store.meta;
        m.put(&mut self.txn, "next_id", &self.next_id.to_be_bytes())?;
        m.put(&mut self.txn, "counts", &serde_json::to_vec(&self.counts)?)?;
        m.put(
            &mut self.txn,
            "inference_hash",
            self.inference.hash.as_bytes(),
        )?;
        if let Some(w) = self.watermark {
            m.put(&mut self.txn, "watermark", &w.0.to_be_bytes())?;
        }
        self.txn.commit()?;
        Ok(())
    }
}
