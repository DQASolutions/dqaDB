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
//! One document per entity, fields (BM25F-like, as Elasticsearch `multi_match` over a
//! main field, boosted fields and a shingle field):
//! - `text`: the folded text of all its `dqadb_index: fts` literals;
//! - `w0`, `w1`, ...: again the literals of the slots with a weight (`fts^2`), one field
//!   per weight, searched with boost weight - 1 (so such a slot counts `weight` times);
//! - `bi`: word bigrams ("hai phong duong" -> "hai_phong phong_duong"; Lucene's
//!   ShingleFilter). Vietnamese writes words as syllables, so a bigram is often the
//!   word: "hop dong" (contract) is one meaning, not "hop" + "dong". Bigrams never decide
//!   what matches, they only add to the score of documents having the query's pairs.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tantivy::collector::{Count, FilterCollector, TopDocs};
use tantivy::query::{BooleanQuery, BoostQuery, Occur, Query, QueryParser, TermQuery};
use tantivy::schema::{
    FAST, Field, INDEXED, IndexRecordOption, STORED, Schema, TEXT, TextFieldIndexing,
    TextOptions, Value,
};
use tantivy::tokenizer::{LowerCaser, TextAnalyzer, WhitespaceTokenizer};
use tantivy::{Index, IndexReader, ReloadPolicy, TantivyDocument, Term as TTerm, doc};
use ulid::Ulid;
use unicode_normalization::UnicodeNormalization;

use crate::event::{self, OpKind};
use crate::rdf::Term;
use crate::store::{Id, Reader};
use crate::{Error, Result};

/// Bump when folding or the document layout changes (forces a re-index).
const FTS_FORMAT: u32 = 2;
/// Weight of the bigram field (tuned on BEIR SciFact and Zalo legal, bench/retrieval).
/// Vietnamese law (Zalo legal, 61k articles): nDCG@10 0.551 -> 0.713 at 0.4 (flat from 0.3
/// to 0.5); English (SciFact): 0.651 -> 0.650, falling above 0.5 (0.604 at 1).
const BIGRAM_BOOST: f32 = 0.4;
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

/// Bigrams of a folded query: adjacent words, the chain broken by operators (AND, OR,
/// NOT), field prefixes and excluded words (`-x`, `NOT x`).
pub fn query_bigrams(folded: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chain: Vec<&str> = Vec::new();
    let mut skip_next = false;
    let flush = |chain: &mut Vec<&str>, out: &mut Vec<String>| {
        for p in chain.windows(2) {
            out.push(format!("{}_{}", p[0], p[1]));
        }
        chain.clear();
    };
    for tok in folded.split_whitespace() {
        if matches!(tok, "AND" | "OR" | "NOT") {
            flush(&mut chain, &mut out);
            skip_next = tok == "NOT";
            continue;
        }
        if skip_next || tok.starts_with('-') || tok.contains(':') {
            skip_next = false;
            flush(&mut chain, &mut out);
            continue;
        }
        chain.extend(words(tok));
    }
    flush(&mut chain, &mut out);
    out.sort();
    out.dedup();
    out
}

/// What the index reflects: the SQLite watermark it caught up to, and the configuration
/// (indexed predicates, folding version) it was built with.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FtsState {
    pub watermark: Option<String>,
    pub config: String,
}

pub fn config_hash(predicates: &[(String, f32)]) -> String {
    let mut h = Sha256::new();
    h.update(format!("fts{FTS_FORMAT}\n"));
    let mut sorted = predicates.to_vec();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    for (p, w) in sorted {
        h.update(format!("{p} {w}\n"));
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
    bi: Field,
    /// (weight, field) for each distinct slot weight other than 1
    weighted: Vec<(f32, Field)>,
}

const BIGRAMS: &str = "dqadb_bigrams";

/// Distinct weights other than 1, ascending: one field each.
pub fn weights(predicates: &[(String, f32)]) -> Vec<f32> {
    let mut w: Vec<f32> = predicates
        .iter()
        .map(|p| p.1)
        .filter(|w| *w != 1.0)
        .collect();
    w.sort_by(f32::total_cmp);
    w.dedup();
    w
}

fn schema(weights: &[f32]) -> (Schema, Field, Field, Field, Vec<(f32, Field)>) {
    let mut b = Schema::builder();
    let id = b.add_u64_field("id", INDEXED | FAST | STORED);
    let text = b.add_text_field("text", TEXT);
    // bigrams are whole tokens ("hai_phong"): split on whitespace only; term frequencies
    // only (no positions needed)
    let bi_opts = TextOptions::default().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer(BIGRAMS)
            .set_index_option(IndexRecordOption::WithFreqs),
    );
    let bi = b.add_text_field("bi", bi_opts);
    let weighted = weights
        .iter()
        .enumerate()
        .map(|(i, w)| (*w, b.add_text_field(&format!("w{i}"), TEXT)))
        .collect();
    (b.build(), id, text, bi, weighted)
}

/// Words of folded text, as the default tokenizer splits them.
fn words(folded: &str) -> impl Iterator<Item = &str> {
    folded
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
}

/// Adjacent word pairs of folded text, space-separated: "a b c" -> "a_b b_c".
fn bigrams(folded: &str) -> String {
    let w: Vec<&str> = words(folded).collect();
    let mut out = String::new();
    for p in w.windows(2) {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(p[0]);
        out.push('_');
        out.push_str(p[1]);
    }
    out
}

impl Fts {
    /// Open the index, or create an empty one (also when the layout is not ours).
    /// `weights`: see [`weights`]; a different set rebuilds the index.
    pub fn open_or_create(dir: &Path, weights: &[f32]) -> Result<Self> {
        let (schema, id, text, bi, weighted) = schema(weights);
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
        index.tokenizers().register(
            BIGRAMS,
            TextAnalyzer::builder(WhitespaceTokenizer::default())
                .filter(LowerCaser)
                .build(),
        );
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
            bi,
            weighted,
        })
    }

    pub fn state(&self) -> Result<Option<FtsState>> {
        let metas = self.index.load_metas().map_err(tantivy_err)?;
        Ok(metas.payload.and_then(|p| serde_json::from_str(&p).ok()))
    }

    /// The entity's document, or None without text. Bigrams stay within one literal.
    fn entity_doc(&self, r: &Reader, s: Id, preds: &[(Id, f32)]) -> Result<Option<TantivyDocument>> {
        let mut d = TantivyDocument::default();
        let mut any = false;
        for (p, w) in preds {
            for o in r.objects(s, *p)? {
                if let Term::Literal(l) = r.term(o)? {
                    let folded = fold(&l.value);
                    if folded.trim().is_empty() {
                        continue;
                    }
                    any = true;
                    d.add_text(self.bi, bigrams(&folded));
                    if let Some((_, f)) = self.weighted.iter().find(|(x, _)| x == w) {
                        d.add_text(*f, &folded);
                    }
                    d.add_text(self.text, folded);
                }
            }
        }
        if !any {
            return Ok(None);
        }
        d.add_u64(self.id, s);
        Ok(Some(d))
    }

    fn write(
        &self,
        r: &Reader,
        subjects: &BTreeSet<Id>,
        preds: &[(Id, f32)],
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
            if let Some(d) = self.entity_doc(r, *s, preds)? {
                w.add_document(d).map_err(tantivy_err)?;
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
    pub fn rebuild(&self, r: &Reader, preds: &[(Id, f32)], state: &FtsState) -> Result<usize> {
        let mut subjects = BTreeSet::new();
        for (p, _) in preds {
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
        preds: &[(Id, f32)],
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
        // Weighted slots: the same words are also looked up in their own field, boosted
        // by weight - 1 (Elasticsearch `multi_match` most_fields / Solr `qf=title^2`).
        let folded = fold_query(query);
        let mut fields = vec![self.text];
        fields.extend(self.weighted.iter().map(|(_, f)| *f));
        let mut parser = QueryParser::for_index(&self.index, fields);
        for (w, f) in &self.weighted {
            parser.set_field_boost(*f, w - 1.0);
        }
        let (main, _errors) = parser.parse_query_lenient(&folded);
        // Bigrams of the query only add score (Elasticsearch: must = the query, should =
        // its shingles), so they never change which documents match.
        let pairs: Vec<Box<dyn Query>> = query_bigrams(&folded)
            .into_iter()
            .map(|b| {
                Box::new(TermQuery::new(
                    TTerm::from_field_text(self.bi, &b),
                    IndexRecordOption::WithFreqs,
                )) as Box<dyn Query>
            })
            .collect();
        let q: Box<dyn Query> = if pairs.is_empty() {
            main
        } else {
            // DQADB_BIGRAM_BOOST: for weight sweeps (bench/retrieval)
            let bigram_boost = std::env::var("DQADB_BIGRAM_BOOST")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(BIGRAM_BOOST);
            Box::new(BooleanQuery::new(vec![
                (Occur::Must, main),
                (
                    Occur::Should,
                    Box::new(BoostQuery::new(
                        Box::new(BooleanQuery::union(pairs)),
                        bigram_boost,
                    )),
                ),
            ]))
        };
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

    #[test]
    fn bigrams_of_text_and_queries() {
        assert_eq!(bigrams("hop dong lao-dong."), "hop_dong dong_lao lao_dong");
        assert_eq!(bigrams("mot"), "");
        assert_eq!(
            query_bigrams("hop dong OR lao dong -thu viec"),
            vec!["hop_dong", "lao_dong"]
        );
        assert_eq!(query_bigrams("\"hop dong\" NOT mau so"), vec!["hop_dong"]);
    }
}
