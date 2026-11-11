//! Index store on SQLite (WAL mode): dictionary, SPO/POS/OSP permutations, provenance,
//! and materialised RDFS inferences. Everything here is derived and can be rebuilt
//! from events.
//!
//! Why SQLite (see the store evaluation in the spec): many reader processes alongside one
//! writer (every CLI call is a process, `dqadb serve` runs beside them), a file format
//! guaranteed stable to 2050, small files, cheap small commits, and agents can inspect
//! the index with `sqlite3`. SQLite is only the B-tree engine here: graph logic, RDFS
//! and SPARQL live in dqaDB.
//!
//! Only this module talks to SQLite. Callers use [`Reader`] (a read snapshot) and
//! [`Batch`] (one write transaction).
//!
//! A triple is in `spo`/`pos`/`osp` when it is asserted (has at least one `prov` row)
//! or inferred (has an `inf` row), or both.

use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;
use ulid::Ulid;

use crate::event::{Event, OpKind};
use crate::ontology::Inference;
use crate::rdf::{self, Term};
use crate::{Error, Result};

/// Bump when the layout changes; a mismatch forces `dqadb rebuild`.
/// 3: SQLite backend (1-2 were LMDB). 4: `props` table for typed literal filters.
/// 5: `vecs` / `vec_dead` tables for the vector projection. 6: 16 KB pages.
pub const STORE_FORMAT: u32 = 6;
const PAGE: i64 = 1024;
const BUSY_TIMEOUT: Duration = Duration::from_secs(10);

pub type Id = u64;
pub type Triple = (Id, Id, Id);

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value BLOB NOT NULL) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS terms(id INTEGER PRIMARY KEY, hash BLOB NOT NULL UNIQUE, term TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS spo(a INTEGER NOT NULL, b INTEGER NOT NULL, c INTEGER NOT NULL, PRIMARY KEY(a,b,c)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS pos(a INTEGER NOT NULL, b INTEGER NOT NULL, c INTEGER NOT NULL, PRIMARY KEY(a,b,c)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS osp(a INTEGER NOT NULL, b INTEGER NOT NULL, c INTEGER NOT NULL, PRIMARY KEY(a,b,c)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS prov(a INTEGER NOT NULL, b INTEGER NOT NULL, c INTEGER NOT NULL, ev BLOB NOT NULL, PRIMARY KEY(a,b,c,ev)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS inf(a INTEGER NOT NULL, b INTEGER NOT NULL, c INTEGER NOT NULL, PRIMARY KEY(a,b,c)) WITHOUT ROWID;
-- every literal-valued triple, with a comparable key: num for numbers, booleans,
-- dates (days since epoch) and date-times (seconds since epoch); txt for strings
CREATE TABLE IF NOT EXISTS props(p INTEGER NOT NULL, s INTEGER NOT NULL, o INTEGER NOT NULL, num REAL, txt TEXT, PRIMARY KEY(p,s,o)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS props_num ON props(p, num) WHERE num IS NOT NULL;
CREATE INDEX IF NOT EXISTS props_txt ON props(p, txt) WHERE txt IS NOT NULL;
-- vector projection: one f16 vector per entity; seq orders changes (base vs delta)
CREATE TABLE IF NOT EXISTS vecs(id INTEGER PRIMARY KEY, seq INTEGER NOT NULL, hash BLOB NOT NULL, vec BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS vecs_seq ON vecs(seq);
CREATE TABLE IF NOT EXISTS vec_dead(id INTEGER PRIMARY KEY, seq INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS vec_dead_seq ON vec_dead(seq);
";

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

/// What the vector projection reflects: the index watermark it caught up to and the
/// configuration (model id, embedded predicates) it was built with.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VecState {
    pub watermark: Option<String>,
    pub config: String,
}

/// One stored vector: content hash of its text and the f16 bytes.
#[derive(Debug, Clone)]
pub struct VecRow {
    pub hash: [u8; 32],
    pub vec: Vec<u8>,
}

/// Comparison in a `--where` filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpOp {
    fn sql(self) -> &'static str {
        match self {
            CmpOp::Eq => "=",
            CmpOp::Ne => "<>",
            CmpOp::Lt => "<",
            CmpOp::Le => "<=",
            CmpOp::Gt => ">",
            CmpOp::Ge => ">=",
        }
    }
}

/// A condition on one predicate's values.
#[derive(Debug, Clone, PartialEq)]
pub enum PropCond {
    /// the subject has any value for the predicate
    Has,
    /// value is this IRI (dictionary id)
    Iri(Id),
    Num(CmpOp, f64),
    Txt(CmpOp, String),
}

/// Comparable key of a literal for the `props` table: (num, txt).
pub fn literal_key(l: &rdf::Literal) -> (Option<f64>, Option<String>) {
    if l.language.is_some() {
        return (None, Some(l.value.clone()));
    }
    let local = l
        .datatype
        .as_deref()
        .and_then(|d| d.strip_prefix(rdf::XSD))
        .unwrap_or("string");
    match local {
        "integer" | "decimal" | "double" | "float" | "int" | "long" => (l.value.parse().ok(), None),
        "boolean" => (Some(if l.value == "true" { 1.0 } else { 0.0 }), None),
        "date" => (
            chrono::NaiveDate::parse_from_str(&l.value, "%Y-%m-%d")
                .ok()
                .map(|d| (d - chrono::NaiveDate::default()).num_days() as f64),
            None,
        ),
        "dateTime" => (
            chrono::DateTime::parse_from_rfc3339(&l.value)
                .ok()
                .map(|d| d.timestamp() as f64),
            None,
        ),
        _ => (None, Some(l.value.clone())),
    }
}

/// Index permutation; each table stores (a, b, c) in its own order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Perm {
    Spo,
    Pos,
    Osp,
}

impl Perm {
    fn table(self) -> &'static str {
        match self {
            Perm::Spo => "spo",
            Perm::Pos => "pos",
            Perm::Osp => "osp",
        }
    }

    /// (a, b, c) of this table -> (s, p, o)
    fn to_spo(self, (a, b, c): Triple) -> Triple {
        match self {
            Perm::Spo => (a, b, c),
            Perm::Pos => (c, a, b),
            Perm::Osp => (b, c, a),
        }
    }

    /// Best table for a pattern, and the bound prefix in that table's order.
    pub fn for_pattern(s: Option<Id>, p: Option<Id>, o: Option<Id>) -> (Perm, Vec<Id>) {
        match (s, p, o) {
            (Some(s), Some(p), Some(o)) => (Perm::Spo, vec![s, p, o]),
            (Some(s), Some(p), None) => (Perm::Spo, vec![s, p]),
            (Some(s), None, Some(o)) => (Perm::Osp, vec![o, s]),
            (Some(s), None, None) => (Perm::Spo, vec![s]),
            (None, Some(p), Some(o)) => (Perm::Pos, vec![p, o]),
            (None, Some(p), None) => (Perm::Pos, vec![p]),
            (None, None, Some(o)) => (Perm::Osp, vec![o]),
            (None, None, None) => (Perm::Spo, vec![]),
        }
    }
}

/// `SELECT a, b, c FROM <table> WHERE <prefix> [AND (a, b, c) > after] ORDER BY a, b, c
/// LIMIT PAGE` for every table, bound-prefix length (0..=3) and paging state.
fn page_sql(perm: Perm, bound: usize, after: bool) -> &'static str {
    static SQL: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    let all = SQL.get_or_init(|| {
        let mut out = Vec::new();
        for perm in [Perm::Spo, Perm::Pos, Perm::Osp] {
            for bound in 0..=3 {
                for after in [false, true] {
                    let cols = ["a", "b", "c"];
                    let mut conds: Vec<String> = (0..bound)
                        .map(|n| format!("{} = ?{}", cols[n], n + 1))
                        .collect();
                    if after {
                        conds.push(format!(
                            "(a, b, c) > (?{}, ?{}, ?{})",
                            bound + 1,
                            bound + 2,
                            bound + 3
                        ));
                    }
                    let filter = if conds.is_empty() {
                        String::new()
                    } else {
                        format!("WHERE {}", conds.join(" AND "))
                    };
                    out.push(format!(
                        "SELECT a, b, c FROM {} {filter} ORDER BY a, b, c LIMIT {PAGE}",
                        perm.table()
                    ));
                }
            }
        }
        out
    });
    let p = match perm {
        Perm::Spo => 0,
        Perm::Pos => 1,
        Perm::Osp => 2,
    };
    &all[p * 8 + bound * 2 + after as usize]
}

fn i(id: Id) -> i64 {
    id as i64
}

fn term_hash(t: &Term) -> [u8; 32] {
    Sha256::digest(t.ntriples().as_bytes()).into()
}

fn ulid_blob(u: Ulid) -> [u8; 16] {
    u.0.to_be_bytes()
}

fn blob_ulid(b: &[u8]) -> Result<Ulid> {
    let arr: [u8; 16] = b
        .try_into()
        .map_err(|_| Error::IndexStale("bad event id in prov".into()))?;
    Ok(Ulid(u128::from_be_bytes(arr)))
}

fn connect(path: &Path, bulk: bool) -> Result<Connection> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    conn.busy_timeout(BUSY_TIMEOUT)?;
    // Query planner stability guarantee. The bundled SQLite has STAT4, so statements
    // whose plan could depend on a bound value are marked for re-prepare, and
    // rusqlite's statement cache clears bindings when a statement goes back to the cache:
    // every reuse of a cached statement then recompiled it (~10 us of a 2 us lookup,
    // 950 lookups per BSBM T5 query). The index never runs ANALYZE, so plans do not
    // depend on bound values anyway; QPSG keeps the prepared plan.
    conn.set_db_config(
        rusqlite::config::DbConfig::SQLITE_DBCONFIG_ENABLE_QPSG,
        true,
    )?;
    // NORMAL in WAL mode: a power loss can drop the last commits of the *index*;
    // the events are already durable and the next command catches up from them.
    let sync = if bulk { "OFF" } else { "NORMAL" };
    conn.execute_batch(&format!(
        // 16 KB pages: a 1024-dim f16 vector (~2 KB) fills 4 KB pages only half
        // (one row per page); 16 KB pages hold 7. Applies when the file is created.
        "PRAGMA page_size=16384; PRAGMA journal_mode=WAL; PRAGMA synchronous={sync}; PRAGMA foreign_keys=OFF;
         PRAGMA cache_size=-65536; PRAGMA mmap_size=268435456;"
    ))?;
    Ok(conn)
}

pub struct Store {
    conn: Mutex<Connection>,
    path: PathBuf,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with(path, false)
    }

    /// For building a fresh index file that nobody else reads yet (rebuild).
    pub fn open_bulk(path: &Path) -> Result<Self> {
        Self::open_with(path, true)
    }

    fn open_with(path: &Path, bulk: bool) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let conn = connect(path, bulk)?;
        let format: Option<Vec<u8>> = conn
            .query_row("SELECT value FROM meta WHERE key = 'format'", [], |r| {
                r.get(0)
            })
            .optional()
            .or_else(|e| match e {
                // fresh file: no tables yet
                rusqlite::Error::SqliteFailure(_, Some(ref m)) if m.contains("no such table") => {
                    Ok(None)
                }
                e => Err(e),
            })?;
        match format {
            None => {
                conn.execute_batch(SCHEMA)?;
                conn.execute(
                    "INSERT OR IGNORE INTO meta VALUES ('format', ?1)",
                    [STORE_FORMAT.to_be_bytes().to_vec()],
                )?;
            }
            Some(b) => {
                let v = <[u8; 4]>::try_from(b.as_slice())
                    .map(u32::from_be_bytes)
                    .unwrap_or(0);
                if v != STORE_FORMAT {
                    return Err(Error::IndexStale(format!(
                        "store format {v}, this binary expects {STORE_FORMAT}"
                    )));
                }
                // added within format 6: makes "what changed since seq" an index lookup
                conn.execute_batch("CREATE INDEX IF NOT EXISTS vec_dead_seq ON vec_dead(seq)")?;
            }
        }
        Ok(Self {
            conn: Mutex::new(conn),
            path: path.to_path_buf(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn lock(&self) -> MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A consistent read snapshot until dropped.
    pub fn read(&self) -> Result<Reader<'_>> {
        let conn = self.lock();
        conn.execute_batch("BEGIN DEFERRED")?;
        Ok(Reader { conn })
    }

    /// Start a write transaction. If the inference rules changed since the index was
    /// built, all inferences are recomputed first.
    pub fn batch<'a>(&'a self, inference: &'a Inference) -> Result<Batch<'a>> {
        let conn = self.lock();
        conn.execute_batch("BEGIN IMMEDIATE")?;
        let r = Reader { conn };
        let next_id = r.meta_u64("next_id")?.unwrap_or(1);
        let counts = r.counts()?;
        let watermark = r.watermark()?;
        let stale = r.inference_hash()?.as_deref() != Some(inference.hash.as_str());
        let mut batch = Batch {
            r: Some(r),
            term_ids: Default::default(),
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

    /// Replace this index's content with another index file (used by rebuild).
    /// Uses SQLite's backup API, so other processes reading this file stay consistent.
    pub fn restore_from(&self, source: &Path) -> Result<()> {
        let mut conn = self.lock();
        conn.restore(
            rusqlite::MAIN_DB,
            source,
            None::<fn(rusqlite::backup::Progress)>,
        )?;
        Ok(())
    }

    /// Apply vector changes in one short write transaction and record the new state.
    /// `None` removes the entity's vector. `clear` drops all vectors first (config change).
    pub fn vec_write(
        &self,
        updates: &[(Id, Option<VecRow>)],
        state: &VecState,
        clear: bool,
    ) -> Result<()> {
        let conn = self.lock();
        conn.execute_batch("BEGIN IMMEDIATE")?;
        let r = Reader { conn };
        let res = (|| -> Result<()> {
            if clear {
                r.conn
                    .execute_batch("DELETE FROM vecs; DELETE FROM vec_dead;")?;
                // a new generation: any HNSW base built from the old rows is now invalid
                r.conn
                    .prepare_cached("INSERT OR REPLACE INTO meta VALUES ('vec_epoch', ?1)")?
                    .execute([ulid::Ulid::generate().to_string().into_bytes()])?;
            }
            let mut seq = r.meta_u64("vec_seq")?.unwrap_or(0);
            for (id, row) in updates {
                seq += 1;
                match row {
                    Some(row) => {
                        r.conn
                            .prepare_cached("INSERT OR REPLACE INTO vecs(id, seq, hash, vec) VALUES (?1, ?2, ?3, ?4)")?
                            .execute(params![i(*id), seq as i64, &row.hash[..], &row.vec])?;
                        r.conn
                            .prepare_cached("DELETE FROM vec_dead WHERE id = ?1")?
                            .execute([i(*id)])?;
                    }
                    None => {
                        if r.conn
                            .prepare_cached("DELETE FROM vecs WHERE id = ?1")?
                            .execute([i(*id)])?
                            == 1
                        {
                            r.conn
                                .prepare_cached(
                                    "INSERT OR REPLACE INTO vec_dead(id, seq) VALUES (?1, ?2)",
                                )?
                                .execute(params![i(*id), seq as i64])?;
                        }
                    }
                }
            }
            let put = |k: &str, v: Vec<u8>| -> Result<()> {
                r.conn
                    .prepare_cached("INSERT OR REPLACE INTO meta VALUES (?1, ?2)")?
                    .execute(params![k, v])?;
                Ok(())
            };
            put("vec_seq", seq.to_be_bytes().to_vec())?;
            put("vec_state", serde_json::to_vec(state)?)?;
            Ok(())
        })();
        match res {
            Ok(()) => {
                r.conn.execute_batch("COMMIT")?;
                Ok(())
            }
            Err(e) => Err(e), // Reader drop rolls back
        }
    }

    /// Hash over dictionary, triples, provenance and inferences: equal = identical index.
    pub fn fingerprint(&self) -> Result<String> {
        let r = self.read()?;
        let mut h = Sha256::new();
        {
            let mut st = r.conn.prepare("SELECT id, term FROM terms ORDER BY id")?;
            let mut rows = st.query([])?;
            while let Some(row) = rows.next()? {
                h.update(row.get::<_, i64>(0)?.to_be_bytes());
                h.update(row.get::<_, String>(1)?.as_bytes());
            }
        }
        for sql in [
            "SELECT a, b, c, NULL FROM spo ORDER BY a, b, c",
            "SELECT a, b, c, ev FROM prov ORDER BY a, b, c, ev",
            "SELECT a, b, c, NULL FROM inf ORDER BY a, b, c",
            "SELECT p, s, o, NULL FROM props ORDER BY p, s, o",
        ] {
            h.update(b"|");
            let mut st = r.conn.prepare(sql)?;
            let mut rows = st.query([])?;
            while let Some(row) = rows.next()? {
                for col in 0..3 {
                    h.update(row.get::<_, i64>(col)?.to_be_bytes());
                }
                if let Some(ev) = row.get::<_, Option<Vec<u8>>>(3)? {
                    h.update(ev);
                }
            }
        }
        Ok(format!("sha256:{}", hex::encode(h.finalize())))
    }
}

/// Read access. Holds the connection for its lifetime; drop it before writing.
pub struct Reader<'a> {
    conn: MutexGuard<'a, Connection>,
}

impl Drop for Reader<'_> {
    fn drop(&mut self) {
        if !self.conn.is_autocommit() {
            let _ = self.conn.execute_batch("ROLLBACK");
        }
    }
}

impl Reader<'_> {
    pub fn vec_state(&self) -> Result<Option<VecState>> {
        Ok(self
            .meta("vec_state")?
            .and_then(|b| serde_json::from_slice(&b).ok()))
    }

    pub fn vec_count(&self) -> Result<u64> {
        Ok(self
            .conn
            .query_row("SELECT count(*) FROM vecs", [], |r| r.get::<_, i64>(0))? as u64)
    }

    /// Content hash of the text an entity's stored vector was made from.
    pub fn vec_hash(&self, id: Id) -> Result<Option<Vec<u8>>> {
        Ok(self
            .conn
            .prepare_cached("SELECT hash FROM vecs WHERE id = ?1")?
            .query_row([i(id)], |r| r.get(0))
            .optional()?)
    }

    /// Visit every stored vector (f16 bytes), in id order.
    /// Generation of the vector table (changes whenever it is cleared).
    pub fn vec_epoch(&self) -> Result<Option<String>> {
        Ok(self
            .meta("vec_epoch")?
            .map(|b| String::from_utf8_lossy(&b).into_owned()))
    }

    /// Highest change sequence number applied to the vector table.
    pub fn vec_seq(&self) -> Result<u64> {
        Ok(self.meta_u64("vec_seq")?.unwrap_or(0))
    }

    /// Visit vectors changed after `seq` (the delta on top of an HNSW base).
    pub fn vec_since(&self, seq: u64, mut f: impl FnMut(Id, &[u8])) -> Result<()> {
        let mut st = self
            .conn
            .prepare_cached("SELECT id, vec FROM vecs INDEXED BY vecs_seq WHERE seq > ?1")?;
        let mut rows = st.query([seq as i64])?;
        while let Some(row) = rows.next()? {
            f(
                row.get::<_, i64>(0)? as Id,
                row.get_ref(1)?.as_blob().map_err(rusqlite::Error::from)?,
            );
        }
        Ok(())
    }

    /// Entities whose vector changed or was removed after `seq` (stale in an older base).
    pub fn vec_changed_since(&self, seq: u64) -> Result<Vec<Id>> {
        let mut out = Vec::new();
        for sql in [
            "SELECT id FROM vecs INDEXED BY vecs_seq WHERE seq > ?1",
            "SELECT id FROM vec_dead WHERE seq > ?1",
        ] {
            let mut st = self.conn.prepare_cached(sql)?;
            let rows = st.query_map([seq as i64], |r| r.get::<_, i64>(0))?;
            for id in rows {
                out.push(id? as Id);
            }
        }
        Ok(out)
    }

    /// How many entities changed or lost their vector after `seq`.
    pub fn vec_changed_count(&self, seq: u64) -> Result<u64> {
        let mut n = 0u64;
        for sql in [
            "SELECT count(*) FROM vecs INDEXED BY vecs_seq WHERE seq > ?1",
            "SELECT count(*) FROM vec_dead WHERE seq > ?1",
        ] {
            n += self
                .conn
                .prepare_cached(sql)?
                .query_row([seq as i64], |r| r.get::<_, i64>(0))? as u64;
        }
        Ok(n)
    }

    /// Visit vectors with change sequence <= `seq`, in id order, `page` rows at a time
    /// (used to build an HNSW base without loading everything into memory).
    pub fn vec_upto_pages(
        &self,
        seq: u64,
        page: usize,
        mut f: impl FnMut(Vec<(Id, Vec<u8>)>) -> Result<()>,
    ) -> Result<()> {
        let mut after: i64 = -1;
        loop {
            let mut st = self.conn.prepare_cached(
                "SELECT id, vec FROM vecs WHERE seq <= ?1 AND id > ?2 ORDER BY id LIMIT ?3",
            )?;
            let rows: Vec<(Id, Vec<u8>)> = st
                .query_map(params![seq as i64, after, page as i64], |r| {
                    Ok((r.get::<_, i64>(0)? as Id, r.get::<_, Vec<u8>>(1)?))
                })?
                .collect::<std::result::Result<_, _>>()?;
            let Some(last) = rows.last() else {
                return Ok(());
            };
            after = last.0 as i64;
            let done = rows.len() < page;
            f(rows)?;
            if done {
                return Ok(());
            }
        }
    }

    pub fn vec_each(&self, mut f: impl FnMut(Id, &[u8])) -> Result<()> {
        let mut st = self.conn.prepare("SELECT id, vec FROM vecs ORDER BY id")?;
        let mut rows = st.query([])?;
        while let Some(row) = rows.next()? {
            f(
                row.get::<_, i64>(0)? as Id,
                row.get_ref(1)?.as_blob().map_err(rusqlite::Error::from)?,
            );
        }
        Ok(())
    }

    /// Visit the stored vectors of the given entities.
    pub fn vec_of(&self, ids: &[Id], mut f: impl FnMut(Id, &[u8])) -> Result<()> {
        for chunk in ids.chunks(500) {
            let marks = vec!["?"; chunk.len()].join(",");
            let mut st = self
                .conn
                .prepare_cached(&format!("SELECT id, vec FROM vecs WHERE id IN ({marks})"))?;
            let mut rows = st.query(rusqlite::params_from_iter(chunk.iter().map(|x| i(*x))))?;
            while let Some(row) = rows.next()? {
                f(
                    row.get::<_, i64>(0)? as Id,
                    row.get_ref(1)?.as_blob().map_err(rusqlite::Error::from)?,
                );
            }
        }
        Ok(())
    }

    fn meta(&self, key: &str) -> Result<Option<Vec<u8>>> {
        Ok(self
            .conn
            .prepare_cached("SELECT value FROM meta WHERE key = ?1")?
            .query_row([key], |r| r.get(0))
            .optional()?)
    }

    fn meta_u64(&self, key: &str) -> Result<Option<u64>> {
        Ok(self
            .meta(key)?
            .and_then(|b| <[u8; 8]>::try_from(b.as_slice()).ok())
            .map(u64::from_be_bytes))
    }

    pub fn watermark(&self) -> Result<Option<Ulid>> {
        self.meta("watermark")?.map(|b| blob_ulid(&b)).transpose()
    }

    pub fn inference_hash(&self) -> Result<Option<String>> {
        Ok(self
            .meta("inference_hash")?
            .map(|b| String::from_utf8_lossy(&b).into_owned()))
    }

    pub fn counts(&self) -> Result<Counts> {
        Ok(match self.meta("counts")? {
            Some(b) => serde_json::from_slice(&b)?,
            None => Counts::default(),
        })
    }

    pub fn term_id(&self, t: &Term) -> Result<Option<Id>> {
        Ok(self
            .conn
            .prepare_cached("SELECT id FROM terms WHERE hash = ?1")?
            .query_row([&term_hash(t)[..]], |r| r.get::<_, i64>(0))
            .optional()?
            .map(|v| v as Id))
    }

    pub fn term(&self, id: Id) -> Result<Term> {
        let json: Option<String> = self
            .conn
            .prepare_cached("SELECT term FROM terms WHERE id = ?1")?
            .query_row([i(id)], |r| r.get(0))
            .optional()?;
        let json = json.ok_or_else(|| Error::IndexStale(format!("dangling term id {id}")))?;
        Ok(serde_json::from_str(&json)?)
    }

    /// Triples matching a pattern (any position may be unbound), loaded page by page.
    pub fn scan(&self, s: Option<Id>, p: Option<Id>, o: Option<Id>) -> Scan<'_, '_> {
        let (perm, prefix) = Perm::for_pattern(s, p, o);
        Scan {
            reader: self,
            perm,
            prefix,
            after: None,
            buf: VecDeque::new(),
            done: false,
        }
    }

    fn page(&self, perm: Perm, prefix: &[Id], after: Option<Triple>) -> Result<Vec<Triple>> {
        let mut args: Vec<i64> = prefix.iter().map(|v| i(*v)).collect();
        if let Some((a, b, c)) = after {
            args.extend([i(a), i(b), i(c)]);
        }
        // the SQL text is built once per (table, bound prefix, paging) shape: a lookup is
        // a cached-statement hit plus a B-tree seek, not string building (~3x faster)
        let sql = page_sql(perm, prefix.len(), after.is_some());
        let mut st = self.conn.prepare_cached(sql)?;
        let rows = st.query_map(rusqlite::params_from_iter(args), |r| {
            Ok((
                r.get::<_, i64>(0)? as Id,
                r.get::<_, i64>(1)? as Id,
                r.get::<_, i64>(2)? as Id,
            ))
        })?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    pub fn has_triple(&self, s: Id, p: Id, o: Id) -> Result<bool> {
        Ok(self
            .conn
            .prepare_cached("SELECT 1 FROM spo WHERE a = ?1 AND b = ?2 AND c = ?3")?
            .query_row(params![i(s), i(p), i(o)], |_| Ok(()))
            .optional()?
            .is_some())
    }

    /// (p, o) pairs for subject s.
    pub fn outgoing(&self, s: Id) -> Result<Vec<(Id, Id)>> {
        self.scan(Some(s), None, None)
            .map(|t| t.map(|(_, p, o)| (p, o)))
            .collect()
    }

    /// (s, p) pairs pointing at object o.
    pub fn incoming(&self, o: Id) -> Result<Vec<(Id, Id)>> {
        self.scan(None, None, Some(o))
            .map(|t| t.map(|(s, p, _)| (s, p)))
            .collect()
    }

    pub fn objects(&self, s: Id, p: Id) -> Result<Vec<Id>> {
        self.scan(Some(s), Some(p), None)
            .map(|t| t.map(|(_, _, o)| o))
            .collect()
    }

    /// Current values of (s, p) as terms.
    pub fn values(&self, s: &str, p: &str) -> Result<Vec<Term>> {
        let (Some(s), Some(p)) = (
            self.term_id(&Term::Iri(s.into()))?,
            self.term_id(&Term::Iri(p.into()))?,
        ) else {
            return Ok(Vec::new());
        };
        self.objects(s, p)?
            .into_iter()
            .map(|o| self.term(o))
            .collect()
    }

    /// Events that asserted (s, p, o) and are still in effect, oldest first.
    pub fn provenance(&self, s: Id, p: Id, o: Id) -> Result<Vec<Ulid>> {
        let mut st = self
            .conn
            .prepare_cached("SELECT ev FROM prov WHERE a = ?1 AND b = ?2 AND c = ?3 ORDER BY ev")?;
        let rows = st.query_map(params![i(s), i(p), i(o)], |r| r.get::<_, Vec<u8>>(0))?;
        rows.map(|b| blob_ulid(&b?)).collect()
    }

    pub fn asserted_by(&self, s: Id, p: Id, o: Id, event: Ulid) -> Result<bool> {
        Ok(self
            .conn
            .prepare_cached("SELECT 1 FROM prov WHERE a = ?1 AND b = ?2 AND c = ?3 AND ev = ?4")?
            .query_row(params![i(s), i(p), i(o), &ulid_blob(event)[..]], |_| Ok(()))
            .optional()?
            .is_some())
    }

    /// Subjects whose value for `p` satisfies `cond`, ascending and distinct.
    pub fn subjects_where(&self, p: Id, cond: &PropCond) -> Result<Vec<Id>> {
        let ids = |st: &mut rusqlite::CachedStatement<'_>,
                   args: &[&dyn rusqlite::ToSql]|
         -> Result<Vec<Id>> {
            let rows = st.query_map(args, |r| r.get::<_, i64>(0))?;
            Ok(rows
                .map(|v| v.map(|v| v as Id))
                .collect::<std::result::Result<_, _>>()?)
        };
        match cond {
            PropCond::Has => {
                let mut st = self
                    .conn
                    .prepare_cached("SELECT DISTINCT c FROM pos WHERE a = ?1 ORDER BY c")?;
                ids(&mut st, &[&i(p)])
            }
            PropCond::Iri(o) => {
                let mut st = self
                    .conn
                    .prepare_cached("SELECT c FROM pos WHERE a = ?1 AND b = ?2 ORDER BY c")?;
                ids(&mut st, &[&i(p), &i(*o)])
            }
            PropCond::Num(op, v) => {
                let sql = format!(
                    "SELECT DISTINCT s FROM props WHERE p = ?1 AND num {} ?2 ORDER BY s",
                    op.sql()
                );
                ids(&mut self.conn.prepare_cached(&sql)?, &[&i(p), v])
            }
            PropCond::Txt(op, v) => {
                let sql = format!(
                    "SELECT DISTINCT s FROM props WHERE p = ?1 AND txt {} ?2 ORDER BY s",
                    op.sql()
                );
                ids(&mut self.conn.prepare_cached(&sql)?, &[&i(p), v])
            }
        }
    }

    fn has_prov(&self, s: Id, p: Id, o: Id) -> Result<bool> {
        Ok(self
            .conn
            .prepare_cached("SELECT 1 FROM prov WHERE a = ?1 AND b = ?2 AND c = ?3 LIMIT 1")?
            .query_row(params![i(s), i(p), i(o)], |_| Ok(()))
            .optional()?
            .is_some())
    }

    fn is_inferred(&self, s: Id, p: Id, o: Id) -> Result<bool> {
        Ok(self
            .conn
            .prepare_cached("SELECT 1 FROM inf WHERE a = ?1 AND b = ?2 AND c = ?3")?
            .query_row(params![i(s), i(p), i(o)], |_| Ok(()))
            .optional()?
            .is_some())
    }

    fn inferred_objects(&self, s: Id, p: Id) -> Result<Vec<Id>> {
        let mut st = self
            .conn
            .prepare_cached("SELECT c FROM inf WHERE a = ?1 AND b = ?2 ORDER BY c")?;
        let rows = st.query_map(params![i(s), i(p)], |r| r.get::<_, i64>(0))?;
        Ok(rows
            .map(|v| v.map(|v| v as Id))
            .collect::<std::result::Result<_, _>>()?)
    }
}

/// Lazy, bounded-memory scan over one permutation (keyset pagination).
pub struct Scan<'r, 'c> {
    reader: &'r Reader<'c>,
    perm: Perm,
    prefix: Vec<Id>,
    after: Option<Triple>,
    buf: VecDeque<Triple>,
    done: bool,
}

impl Iterator for Scan<'_, '_> {
    type Item = Result<Triple>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.buf.is_empty() && !self.done {
            match self.reader.page(self.perm, &self.prefix, self.after) {
                Ok(rows) => {
                    self.done = (rows.len() as i64) < PAGE;
                    self.after = rows.last().copied();
                    self.buf.extend(rows);
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            }
        }
        self.buf.pop_front().map(|t| Ok(self.perm.to_spo(t)))
    }
}

/// One write transaction applying events in ULID order. Rolled back unless committed.
pub struct Batch<'a> {
    r: Option<Reader<'a>>,
    /// term ids seen in this batch (predicates and classes repeat on every entity)
    term_ids: std::cell::RefCell<HashMap<Term, Id>>,
    next_id: Id,
    counts: Counts,
    watermark: Option<Ulid>,
    inference: &'a Inference,
}

impl crate::ontology::Existing for Batch<'_> {
    fn values(&self, s: &str, p: &str) -> Vec<Term> {
        Batch::values(self, s, p).unwrap_or_default()
    }
    fn has(&self, s: &str, p: &str, o: &Term) -> bool {
        Batch::has(self, s, p, o).unwrap_or(false)
    }
}

impl<'a> Batch<'a> {
    fn r(&self) -> &Reader<'a> {
        self.r.as_ref().expect("batch used after commit")
    }

    fn exec(&self, sql: &str, args: impl rusqlite::Params) -> Result<usize> {
        Ok(self.r().conn.prepare_cached(sql)?.execute(args)?)
    }

    /// Id of a term already in the index (cached for the batch).
    fn known_id(&self, t: &Term) -> Result<Option<Id>> {
        if let Some(id) = self.term_ids.borrow().get(t) {
            return Ok(Some(*id));
        }
        let id = self.r().term_id(t)?;
        if let Some(id) = id {
            self.remember(t, id);
        }
        Ok(id)
    }

    fn remember(&self, t: &Term, id: Id) {
        let mut cache = self.term_ids.borrow_mut();
        if cache.len() > 1_000_000 {
            cache.clear();
        }
        cache.insert(t.clone(), id);
    }

    fn id_for(&mut self, t: &Term) -> Result<Id> {
        if let Some(id) = self.known_id(t)? {
            return Ok(id);
        }
        let id = self.next_id;
        self.next_id += 1;
        self.exec(
            "INSERT INTO terms(id, hash, term) VALUES (?1, ?2, ?3)",
            params![i(id), &term_hash(t)[..], serde_json::to_string(t)?],
        )?;
        self.counts.terms += 1;
        self.remember(t, id);
        Ok(id)
    }

    fn put_triple(&mut self, s: Id, p: Id, o: Id) -> Result<()> {
        self.put_triple_term(s, p, o, None).map(|_| ())
    }

    /// `o_term`: the object's term when the caller has it (saves a lookup).
    /// Returns whether the triple was new.
    fn put_triple_term(&mut self, s: Id, p: Id, o: Id, o_term: Option<&Term>) -> Result<bool> {
        if self.exec(
            "INSERT OR IGNORE INTO spo VALUES (?1, ?2, ?3)",
            params![i(s), i(p), i(o)],
        )? == 1
        {
            let owned;
            let term = match o_term {
                Some(t) => t,
                None => {
                    owned = self.r().term(o)?;
                    &owned
                }
            };
            if let Term::Literal(l) = term {
                let (num, txt) = literal_key(l);
                self.exec(
                    "INSERT INTO props VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![i(p), i(s), i(o), num, txt],
                )?;
            }
            self.exec(
                "INSERT INTO pos VALUES (?1, ?2, ?3)",
                params![i(p), i(o), i(s)],
            )?;
            self.exec(
                "INSERT INTO osp VALUES (?1, ?2, ?3)",
                params![i(o), i(s), i(p)],
            )?;
            self.counts.triples += 1;
            return Ok(true);
        }
        Ok(false)
    }

    fn delete_triple(&mut self, s: Id, p: Id, o: Id) -> Result<()> {
        if self.exec(
            "DELETE FROM spo WHERE a = ?1 AND b = ?2 AND c = ?3",
            params![i(s), i(p), i(o)],
        )? == 1
        {
            self.exec(
                "DELETE FROM pos WHERE a = ?1 AND b = ?2 AND c = ?3",
                params![i(p), i(o), i(s)],
            )?;
            self.exec(
                "DELETE FROM props WHERE p = ?1 AND s = ?2 AND o = ?3",
                params![i(p), i(s), i(o)],
            )?;
            self.exec(
                "DELETE FROM osp WHERE a = ?1 AND b = ?2 AND c = ?3",
                params![i(o), i(s), i(p)],
            )?;
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
                    // an asserted triple is always in spo, so a new spo row means new
                    if !self.put_triple_term(s, p, o, Some(&op.o))? && self.r().has_prov(s, p, o)? {
                        stats.noop += 1;
                    } else {
                        stats.added += 1;
                    }
                    self.exec(
                        "INSERT OR IGNORE INTO prov VALUES (?1, ?2, ?3, ?4)",
                        params![i(s), i(p), i(o), &ulid_blob(eid)[..]],
                    )?;
                    if op.p == rdf::RDF_TYPE {
                        retyped.insert(s);
                    }
                }
                OpKind::Retract => {
                    let ids = (
                        self.known_id(&s_term)?,
                        self.known_id(&p_term)?,
                        self.known_id(&op.o)?,
                    );
                    let (Some(s), Some(p), Some(o)) = ids else {
                        stats.noop += 1;
                        continue;
                    };
                    let removed = self.exec(
                        "DELETE FROM prov WHERE a = ?1 AND b = ?2 AND c = ?3",
                        params![i(s), i(p), i(o)],
                    )?;
                    if removed == 0 {
                        // not asserted (possibly only inferred): nothing to retract
                        stats.noop += 1;
                        continue;
                    }
                    if !self.r().is_inferred(s, p, o)? {
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
        let Some(type_id) = self.r().term_id(&Term::Iri(rdf::RDF_TYPE.into()))? else {
            return Ok(());
        };
        let mut asserted = BTreeSet::new();
        for o in self.r().objects(s, type_id)? {
            if self.r().has_prov(s, type_id, o)? {
                asserted.insert(o);
            }
        }
        let mut wanted = BTreeSet::new();
        for o in &asserted {
            let Term::Iri(class) = self.r().term(*o)? else {
                continue;
            };
            for sup in self.inference.superclasses(&class).to_vec() {
                let id = self.id_for(&Term::Iri(sup))?;
                if !asserted.contains(&id) {
                    wanted.insert(id);
                }
            }
        }
        let current: BTreeSet<Id> = self.r().inferred_objects(s, type_id)?.into_iter().collect();
        for o in current.difference(&wanted) {
            self.exec(
                "DELETE FROM inf WHERE a = ?1 AND b = ?2 AND c = ?3",
                params![i(s), i(type_id), i(*o)],
            )?;
            self.counts.inferred -= 1;
            if !self.r().has_prov(s, type_id, *o)? {
                self.delete_triple(s, type_id, *o)?;
            }
        }
        for o in wanted.difference(&current) {
            self.exec(
                "INSERT INTO inf VALUES (?1, ?2, ?3)",
                params![i(s), i(type_id), i(*o)],
            )?;
            self.counts.inferred += 1;
            self.put_triple(s, type_id, *o)?;
        }
        Ok(())
    }

    /// Recompute every inference (the ontology's class hierarchy changed).
    fn reinfer_all(&mut self) -> Result<()> {
        let mut subjects = BTreeSet::new();
        if let Some(type_id) = self.r().term_id(&Term::Iri(rdf::RDF_TYPE.into()))? {
            for t in self.r().scan(None, Some(type_id), None) {
                subjects.insert(t?.0);
            }
        }
        {
            let mut st = self.r().conn.prepare("SELECT DISTINCT a FROM inf")?;
            let rows = st.query_map([], |r| r.get::<_, i64>(0))?;
            for s in rows {
                subjects.insert(s? as Id);
            }
        }
        for s in subjects {
            self.reinfer(s)?;
        }
        Ok(())
    }

    /// Current values of (s, p), including changes applied earlier in this batch.
    /// Whether (s, p, o) is stored (cached term lookups, no term decoding).
    pub fn has(&self, s: &str, p: &str, o: &Term) -> Result<bool> {
        let (Some(s), Some(p), Some(o)) = (
            self.known_id(&Term::Iri(s.into()))?,
            self.known_id(&Term::Iri(p.into()))?,
            self.known_id(o)?,
        ) else {
            return Ok(false);
        };
        self.r().has_triple(s, p, o)
    }

    pub fn values(&self, s: &str, p: &str) -> Result<Vec<Term>> {
        let (Some(s), Some(p)) = (
            self.known_id(&Term::Iri(s.into()))?,
            self.known_id(&Term::Iri(p.into()))?,
        ) else {
            return Ok(Vec::new());
        };
        let r = self.r();
        r.objects(s, p)?.into_iter().map(|o| r.term(o)).collect()
    }

    pub fn counts(&self) -> Counts {
        self.counts
    }

    pub fn commit(mut self) -> Result<()> {
        let mut meta = vec![
            ("next_id", self.next_id.to_be_bytes().to_vec()),
            ("counts", serde_json::to_vec(&self.counts)?),
            ("inference_hash", self.inference.hash.as_bytes().to_vec()),
        ];
        if let Some(w) = self.watermark {
            meta.push(("watermark", ulid_blob(w).to_vec()));
        }
        for (k, v) in meta {
            self.exec("INSERT OR REPLACE INTO meta VALUES (?1, ?2)", params![k, v])?;
        }
        let r = self.r.take().expect("batch committed twice");
        r.conn.execute_batch("COMMIT")?;
        Ok(())
    }
}
