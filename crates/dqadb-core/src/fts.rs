//! Full-text projection on tantivy (spec §11, decision 2026-11-09).
//!
//! The text index is a derived projection like the SQLite index, with its own
//! watermark stored in the tantivy commit payload (written atomically with the data):
//! - `assert` never touches it, so writes stay fast;
//! - the next text search catches up: it re-indexes the entities touched by events
//!   after the payload watermark, in one commit carrying the new watermark;
//! - a crash at any point leaves an older watermark, so the next search repairs it;
//! - if the indexed predicates or the folding rules change, it rebuilds itself.
//!
//! One document per entity: the folded text of all its `dqadb_index: fts` literals.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tantivy::collector::{Count, FilterCollector, TopDocs};
use tantivy::query::QueryParser;
use tantivy::schema::{FAST, Field, INDEXED, STORED, Schema, TEXT, Value};
use tantivy::{Index, IndexReader, ReloadPolicy, TantivyDocument, Term as TTerm, doc};
use ulid::Ulid;
use unicode_normalization::UnicodeNormalization;

use crate::event::{self, OpKind};
use crate::rdf::Term;
use crate::store::{Id, Reader};
use crate::{Error, Result};

/// Bump when folding or the document layout changes (forces a re-index).
const FTS_FORMAT: u32 = 1;
const WRITER_HEAP: usize = 64 * 1024 * 1024;

/// Lowercase, strip combining marks, map đ to d: "Hải Phòng ĐƯỜNG" -> "hai phong duong".
pub fn fold(s: &str) -> String {
    s.nfd()
        .filter(|c| !unicode_normalization::char::is_combining_mark(*c))
        .map(|c| match c {
            'đ' | 'Đ' => 'd',
            c => c.to_lowercase().next().unwrap_or(c),
        })
        .collect()
}

/// `fold` for a query: the words are folded, but the operators AND / OR / NOT stay
/// upper case (Lucene / tantivy syntax), else `a OR b` would search for the word "or".
pub fn fold_query(q: &str) -> String {
    q.split_inclusive(char::is_whitespace)
        .map(|w| match w.trim_end() {
            "AND" | "OR" | "NOT" => w.to_string(),
            _ => fold(w),
        })
        .collect()
}

/// What the index reflects: the SQLite watermark it caught up to, and the configuration
/// (indexed predicates, folding version) it was built with.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FtsState {
    pub watermark: Option<String>,
    pub config: String,
}

pub fn config_hash(predicates: &[String]) -> String {
    let mut h = Sha256::new();
    h.update(format!("fts{FTS_FORMAT}\n"));
    let mut sorted = predicates.to_vec();
    sorted.sort();
    for p in sorted {
        h.update(format!("{p}\n"));
    }
    format!("sha256:{}", hex::encode(h.finalize()))
}

fn tantivy_err(e: tantivy::TantivyError) -> Error {
    Error::other(format!("full-text index: {e}"))
}

pub struct Fts {
    index: Index,
    reader: IndexReader,
    id: Field,
    text: Field,
}

fn schema() -> (Schema, Field, Field) {
    let mut b = Schema::builder();
    let id = b.add_u64_field("id", INDEXED | FAST | STORED);
    let text = b.add_text_field("text", TEXT);
    (b.build(), id, text)
}

impl Fts {
    /// Open the index, or create an empty one (also when the layout is not ours).
    pub fn open_or_create(dir: &Path) -> Result<Self> {
        let (schema, id, text) = schema();
        let index = match Index::open_in_dir(dir) {
            Ok(ix) if ix.schema() == schema => ix,
            _ => {
                if dir.exists() {
                    std::fs::remove_dir_all(dir)?;
                }
                std::fs::create_dir_all(dir)?;
                Index::create_in_dir(dir, schema).map_err(tantivy_err)?
            }
        };
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(tantivy_err)?;
        Ok(Self {
            index,
            reader,
            id,
            text,
        })
    }

    pub fn state(&self) -> Result<Option<FtsState>> {
        let metas = self.index.load_metas().map_err(tantivy_err)?;
        Ok(metas.payload.and_then(|p| serde_json::from_str(&p).ok()))
    }

    fn entity_text(r: &Reader, s: Id, preds: &[Id]) -> Result<String> {
        let mut parts = Vec::new();
        for p in preds {
            for o in r.objects(s, *p)? {
                if let Term::Literal(l) = r.term(o)? {
                    parts.push(l.value);
                }
            }
        }
        Ok(fold(&parts.join("\n")))
    }

    fn write(
        &self,
        r: &Reader,
        subjects: &BTreeSet<Id>,
        preds: &[Id],
        state: &FtsState,
        clear: bool,
    ) -> Result<usize> {
        let mut w = self
            .index
            .writer::<TantivyDocument>(WRITER_HEAP)
            .map_err(tantivy_err)?;
        if clear {
            w.delete_all_documents().map_err(tantivy_err)?;
        }
        let mut n = 0;
        for s in subjects {
            if !clear {
                w.delete_term(TTerm::from_field_u64(self.id, *s));
            }
            let text = Self::entity_text(r, *s, preds)?;
            if !text.trim().is_empty() {
                w.add_document(doc!(self.id => *s, self.text => text))
                    .map_err(tantivy_err)?;
                n += 1;
            }
        }
        let mut commit = w.prepare_commit().map_err(tantivy_err)?;
        commit.set_payload(&serde_json::to_string(state)?);
        commit.commit().map_err(tantivy_err)?;
        w.wait_merging_threads().map_err(tantivy_err)?;
        self.reader.reload().map_err(tantivy_err)?;
        Ok(n)
    }

    /// Index every entity that has text, replacing all previous content.
    pub fn rebuild(&self, r: &Reader, preds: &[Id], state: &FtsState) -> Result<usize> {
        let mut subjects = BTreeSet::new();
        for p in preds {
            for t in r.scan(None, Some(*p), None) {
                subjects.insert(t?.0);
            }
        }
        self.write(r, &subjects, preds, state, true)
    }

    /// Re-index only the given entities (their current text, or removal if none).
    pub fn update(
        &self,
        r: &Reader,
        subjects: &BTreeSet<Id>,
        preds: &[Id],
        state: &FtsState,
    ) -> Result<usize> {
        self.write(r, subjects, preds, state, false)
    }

    /// BM25 search on folded text, optionally restricted to `candidates`.
    /// Returns (entity id, score) best first, and the total number of matches.
    pub fn search(
        &self,
        query: &str,
        candidates: Option<Arc<HashSet<u64>>>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<(Id, f32)>, usize)> {
        let searcher = self.reader.searcher();
        // words are OR-ed, as in Lucene / Elasticsearch `match` / tantivy: BM25 ranks
        // documents with more (and rarer) query words first. AND-ing them drops every
        // document missing one word of a question (BEIR SciFact nDCG@10: 0.02 vs 0.66).
        // `a AND b`, `+a`, `"a b"` still require words.
        let parser = QueryParser::for_index(&self.index, vec![self.text]);
        let (q, _errors) = parser.parse_query_lenient(&fold_query(query));
        let top = TopDocs::with_limit(limit.max(1))
            .and_offset(offset)
            .order_by_score();
        let (docs, total) = match candidates {
            Some(c) => {
                let keep = c.clone();
                let top =
                    FilterCollector::new("id".to_string(), move |v: u64| keep.contains(&v), top);
                let count =
                    FilterCollector::new("id".to_string(), move |v: u64| c.contains(&v), Count);
                searcher.search(&q, &(top, count)).map_err(tantivy_err)?
            }
            None => searcher.search(&q, &(top, Count)).map_err(tantivy_err)?,
        };
        let mut out = Vec::with_capacity(docs.len());
        for (score, addr) in docs {
            let d: TantivyDocument = searcher.doc(addr).map_err(tantivy_err)?;
            if let Some(id) = d.get_first(self.id).and_then(|v| v.as_u64()) {
                out.push((id, score));
            }
        }
        Ok((out, total))
    }
}

/// Entities whose fts text may have changed in events after `after` up to `upto`.
pub fn touched_subjects(
    r: &Reader,
    events_dir: &Path,
    after: Option<Ulid>,
    upto: Option<Ulid>,
    predicates: &HashSet<String>,
) -> Result<BTreeSet<Id>> {
    let mut out = BTreeSet::new();
    for (id, path) in event::list_after(events_dir, after)? {
        if upto.is_some_and(|u| id > u) {
            break;
        }
        let ev = event::read_event(&path)?;
        for op in &ev.ops {
            let relevant =
                predicates.contains(&op.p) && matches!(op.op, OpKind::Add | OpKind::Retract);
            if relevant && let Some(s) = r.term_id(&Term::Iri(op.s.clone()))? {
                out.insert(s);
            }
        }
    }
    Ok(out)
}

pub fn dir(index_dir: &Path) -> PathBuf {
    index_dir.join("fts")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_folding_keeps_operators() {
        assert_eq!(fold_query("Đà Nẵng OR Huế"), "da nang OR hue");
        assert_eq!(fold_query("a AND NOT B"), "a AND NOT b");
        assert_eq!(fold_query("Oregon or ORe"), "oregon or ore");
    }
}
