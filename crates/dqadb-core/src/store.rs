//! MVP index store on LMDB (via heed): dictionary, SPO/POS/OSP permutations, provenance.
//! LMDB allows many reader processes alongside one writer, which a CLI-driven harness
//! (plus `dqadb serve`) needs. Everything here is derived and can be rebuilt from events.

use heed::byteorder::BigEndian;
use heed::types::{Bytes, Str, U64, Unit};
use heed::{Database, Env, EnvOpenOptions, RoTxn, RwTxn, WithTls};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;
use ulid::Ulid;

use crate::event::{Event, OpKind};
use crate::rdf::Term;
use crate::{Error, Result};

/// Bump when the on-disk layout changes; a mismatch forces `dqadb rebuild`.
pub const STORE_FORMAT: u32 = 1;
const MAP_SIZE: usize = 1 << 37; // 128 GiB of address space; disk use grows with data

type Id = u64;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Counts {
    pub events: u64,
    pub triples: u64,
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
    spo: Database<Bytes, Unit>,
    pos: Database<Bytes, Unit>,
    osp: Database<Bytes, Unit>,
    prov: Database<Bytes, Unit>,
}

fn key3(a: Id, b: Id, c: Id) -> [u8; 24] {
    let mut k = [0u8; 24];
    k[..8].copy_from_slice(&a.to_be_bytes());
    k[8..16].copy_from_slice(&b.to_be_bytes());
    k[16..].copy_from_slice(&c.to_be_bytes());
    k
}

fn split3(k: &[u8]) -> (Id, Id, Id) {
    let n = |i: usize| u64::from_be_bytes(k[i..i + 8].try_into().unwrap());
    (n(0), n(8), n(16))
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

    pub fn read_txn(&self) -> Result<RoTxn<'_, WithTls>> {
        Ok(self.env.read_txn()?)
    }

    pub fn watermark(&self, r: &RoTxn) -> Result<Option<Ulid>> {
        Ok(self
            .meta
            .get(r, "watermark")?
            .map(|b| Ulid(u128::from_be_bytes(b.try_into().unwrap()))))
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
            let (k, _) = item?;
            let (_, p, o) = split3(k);
            out.push((p, o));
        }
        Ok(out)
    }

    /// (s, p) pairs pointing at object o.
    pub fn incoming(&self, r: &RoTxn, o: Id) -> Result<Vec<(Id, Id)>> {
        let mut out = Vec::new();
        for item in self.osp.prefix_iter(r, &o.to_be_bytes())? {
            let (k, _) = item?;
            let (_, s, p) = split3(k);
            out.push((s, p));
        }
        Ok(out)
    }

    pub fn objects(&self, r: &RoTxn, s: Id, p: Id) -> Result<Vec<Id>> {
        let mut prefix = [0u8; 16];
        prefix[..8].copy_from_slice(&s.to_be_bytes());
        prefix[8..].copy_from_slice(&p.to_be_bytes());
        let mut out = Vec::new();
        for item in self.spo.prefix_iter(r, &prefix)? {
            out.push(split3(item?.0).2);
        }
        Ok(out)
    }

    /// Events that asserted (s, p, o) and are still in effect, oldest first.
    pub fn provenance(&self, r: &RoTxn, s: Id, p: Id, o: Id) -> Result<Vec<Ulid>> {
        let mut out = Vec::new();
        for item in self.prov.prefix_iter(r, &key3(s, p, o))? {
            let (k, _) = item?;
            out.push(Ulid(u128::from_be_bytes(k[24..40].try_into().unwrap())));
        }
        Ok(out)
    }

    /// Hash over the dictionary and all triples: equal fingerprints = identical index.
    pub fn fingerprint(&self) -> Result<String> {
        let r = self.read_txn()?;
        let mut h = Sha256::new();
        for item in self.id2term.iter(&r)? {
            let (id, t) = item?;
            h.update(id.to_be_bytes());
            h.update(t);
        }
        for item in self.spo.iter(&r)? {
            h.update(item?.0);
        }
        for item in self.prov.iter(&r)? {
            h.update(item?.0);
        }
        Ok(format!("sha256:{}", hex::encode(h.finalize())))
    }

    pub fn batch(&self) -> Result<Batch<'_>> {
        let txn = self.env.write_txn()?;
        let next_id = match self.meta.get(&txn, "next_id")? {
            Some(b) => u64::from_be_bytes(b.try_into().unwrap()),
            None => 1,
        };
        let counts = self.counts(&txn)?;
        let watermark = self.watermark(&txn)?;
        Ok(Batch {
            store: self,
            txn,
            next_id,
            counts,
            watermark,
        })
    }
}

/// One write transaction applying events in ULID order.
pub struct Batch<'a> {
    store: &'a Store,
    txn: RwTxn<'a>,
    next_id: Id,
    counts: Counts,
    watermark: Option<Ulid>,
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

    pub fn apply(&mut self, ev: &Event) -> Result<ApplyStats> {
        let eid = ev.ulid()?;
        if self.watermark.is_some_and(|w| eid <= w) {
            return Err(Error::IndexStale(format!(
                "event {eid} is older than the watermark {}; it was added out of order",
                self.watermark.unwrap()
            )));
        }
        let mut stats = ApplyStats::default();
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
                    if self.store.spo.get(&self.txn, &k)?.is_none() {
                        self.store.spo.put(&mut self.txn, &k, &())?;
                        self.store.pos.put(&mut self.txn, &key3(p, o, s), &())?;
                        self.store.osp.put(&mut self.txn, &key3(o, s, p), &())?;
                        self.counts.triples += 1;
                        stats.added += 1;
                    } else {
                        stats.noop += 1;
                    }
                    let mut pk = [0u8; 40];
                    pk[..24].copy_from_slice(&k);
                    pk[24..].copy_from_slice(&eid.0.to_be_bytes());
                    self.store.prov.put(&mut self.txn, &pk, &())?;
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
                    if !self.store.spo.delete(&mut self.txn, &k)? {
                        stats.noop += 1;
                        continue;
                    }
                    self.store.pos.delete(&mut self.txn, &key3(p, o, s))?;
                    self.store.osp.delete(&mut self.txn, &key3(o, s, p))?;
                    let keys: Vec<Vec<u8>> = self
                        .store
                        .prov
                        .prefix_iter(&self.txn, &k)?
                        .map(|r| r.map(|(k, _)| k.to_vec()))
                        .collect::<std::result::Result<_, _>>()?;
                    for pk in keys {
                        self.store.prov.delete(&mut self.txn, &pk)?;
                    }
                    self.counts.triples -= 1;
                    stats.retracted += 1;
                }
            }
        }
        self.counts.events += 1;
        self.watermark = Some(eid);
        Ok(stats)
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
        if let Some(w) = self.watermark {
            m.put(&mut self.txn, "watermark", &w.0.to_be_bytes())?;
        }
        self.txn.commit()?;
        Ok(())
    }
}
