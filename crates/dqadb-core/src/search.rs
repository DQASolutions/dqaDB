//! Structured retrieval (spec §11): `search` by type and `--where` filters, `traverse`
//! along a property path, and `path` between two entities.

use serde::Serialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use crate::error::ValidationReport;
use crate::kb::{Kb, TypeRef};
use crate::ontology::{Ontology, Range, normalize_lexical};
use crate::rdf::{self, Literal, Term};
use crate::sparql::{self, Answer, Limits};
use crate::store::{CmpOp, Id, PropCond, Reader, literal_key};
use crate::vectors;
use crate::{Error, Result};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TextMode {
    /// text is a required condition; results ranked by semantic similarity
    #[default]
    Must,
    /// BM25 and semantic rankings merged with Reciprocal Rank Fusion
    Fuse,
}

#[derive(Debug, Clone, Default)]
pub struct SearchQuery {
    /// class CURIEs/IRIs; entity must have all of them (inferred types count)
    pub types: Vec<String>,
    /// filters such as `ex:country = "VN"`, `ex:value >= 10000`, `has ex:email`
    pub wheres: Vec<String>,
    /// full-text query (accents and đ/d ignored); results ranked by BM25
    pub text: Option<String>,
    /// semantic query, embedded with the configured model; ranked by cosine.
    /// With `text` too, see `text_mode`.
    pub semantic: Option<String>,
    /// How `text` combines with `semantic`: `Must` (default) keeps only text matches and
    /// ranks them by meaning; `Fuse` merges both rankings with RRF.
    pub text_mode: TextMode,
    pub limit: usize,
    /// opaque continuation from a previous page
    pub cursor: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Hit {
    pub iri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub types: Vec<TypeRef>,
    /// values of the filtered predicates, to show why it matched
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub matched: Vec<(String, String)>,
    /// BM25 score when the search had --text
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct SearchPage {
    pub hits: Vec<Hit>,
    pub total: usize,
    /// with --semantic: events not yet embedded (results may miss very recent entities)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pending: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Step {
    pub from: String,
    pub p: String,
    pub to: String,
    /// false when the edge was followed from object to subject
    pub forward: bool,
}

/// One parsed `--where` clause, with IRIs still unresolved.
#[derive(Debug, Clone, PartialEq)]
enum Clause {
    Has(String),
    Cmp(String, CmpOp, Vec<Value>),
}

#[derive(Debug, Clone, PartialEq)]
enum Value {
    Quoted(String),
    Bare(String),
}

impl Value {
    fn raw(&self) -> &str {
        match self {
            Value::Quoted(s) | Value::Bare(s) => s,
        }
    }
}

fn bad(clause: &str, msg: impl Into<String>) -> Error {
    let mut r = ValidationReport::default();
    r.push(
        format!("--where '{clause}'"),
        msg,
        Some("forms: 'ex:p = \"text\"', 'ex:p >= 10', 'ex:p in (\"A\", \"B\")', 'ex:p = dqadb:<id>', 'has ex:p'".into()),
    );
    Error::Validation(r)
}

fn parse_value(s: &str) -> Value {
    let s = s.trim();
    match s.strip_prefix('"').and_then(|x| x.strip_suffix('"')) {
        Some(inner) => Value::Quoted(inner.replace("\\\"", "\"")),
        None => Value::Bare(s.to_string()),
    }
}

/// Split a comma-separated list, respecting double quotes.
fn split_list(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    let mut prev = '\0';
    for c in s.chars() {
        match c {
            '"' if prev != '\\' => {
                quoted = !quoted;
                cur.push(c);
            }
            ',' if !quoted => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
        prev = c;
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

fn parse_clause(text: &str) -> Result<Clause> {
    let t = text.trim();
    if let Some(p) = t.strip_prefix("has ") {
        return Ok(Clause::Has(p.trim().to_string()));
    }
    let end = if t.starts_with('<') {
        t.find('>')
            .map(|i| i + 1)
            .ok_or_else(|| bad(text, "unterminated <IRI>"))?
    } else {
        t.find(|c: char| c.is_whitespace() || "=!<>".contains(c))
            .unwrap_or(t.len())
    };
    let (pred, rest) = t.split_at(end);
    let rest = rest.trim_start();
    if pred.is_empty() {
        return Err(bad(text, "missing predicate"));
    }
    if let Some(list) = rest
        .strip_prefix("in")
        .map(str::trim_start)
        .filter(|r| r.starts_with('('))
    {
        let inner = list
            .strip_prefix('(')
            .and_then(|x| x.strip_suffix(')'))
            .ok_or_else(|| bad(text, "expected (v1, v2, ...)"))?;
        let values: Vec<Value> = split_list(inner).iter().map(|v| parse_value(v)).collect();
        if values.is_empty() {
            return Err(bad(text, "empty list"));
        }
        return Ok(Clause::Cmp(pred.into(), CmpOp::Eq, values));
    }
    let ops = [
        ("!=", CmpOp::Ne),
        ("<=", CmpOp::Le),
        (">=", CmpOp::Ge),
        ("=", CmpOp::Eq),
        ("<", CmpOp::Lt),
        (">", CmpOp::Gt),
    ];
    let (op, value) = ops
        .iter()
        .find_map(|(sym, op)| rest.strip_prefix(sym).map(|v| (*op, v)))
        .ok_or_else(|| bad(text, "expected an operator: = != < <= > >= in"))?;
    if value.trim().is_empty() {
        return Err(bad(text, "missing value"));
    }
    Ok(Clause::Cmp(pred.into(), op, vec![parse_value(value)]))
}

/// Translate a clause into conditions on dictionary ids, using the slot's range to read
/// the value. Returns None when a referenced term is not in the index (nothing matches).
fn resolve(
    onto: &Ontology,
    r: &Reader,
    text: &str,
    clause: &Clause,
) -> Result<Option<(Id, Vec<PropCond>)>> {
    let pred_of = |p: &str| -> Result<String> {
        let iri = onto.prefixes.expand(p).map_err(|e| bad(text, e))?;
        if iri != rdf::RDF_TYPE && !onto.slots.contains_key(&iri) {
            return Err(bad(
                text,
                format!(
                    "unknown predicate {} (see: dqadb ontology show)",
                    onto.compact(&iri)
                ),
            ));
        }
        Ok(iri)
    };
    let (pred, conds) = match clause {
        Clause::Has(p) => (pred_of(p)?, vec![Some(PropCond::Has)]),
        Clause::Cmp(p, op, values) => {
            let iri = pred_of(p)?;
            let range = if iri == rdf::RDF_TYPE {
                Range::AnyIri
            } else {
                onto.slots[&iri].range.clone()
            };
            let mut conds = Vec::new();
            for v in values {
                conds.push(match &range {
                    Range::Class(_) | Range::AnyIri => {
                        if *op != CmpOp::Eq {
                            return Err(bad(
                                text,
                                "entity-valued predicates only support = and in",
                            ));
                        }
                        let o = onto.prefixes.expand(v.raw()).map_err(|e| bad(text, e))?;
                        r.term_id(&Term::Iri(o))?.map(PropCond::Iri)
                    }
                    Range::Enum(_) => Some(PropCond::Txt(*op, v.raw().to_string())),
                    Range::Datatype(dt) => {
                        let lex = normalize_lexical(dt, v.raw()).map_err(|e| bad(text, e))?;
                        let lit = Literal {
                            value: lex,
                            datatype: Some(dt.clone()),
                            language: None,
                        };
                        match literal_key(&lit) {
                            (Some(n), _) => Some(PropCond::Num(*op, n)),
                            (None, Some(t)) => Some(PropCond::Txt(*op, t)),
                            (None, None) => None,
                        }
                    }
                });
            }
            (iri, conds)
        }
    };
    let Some(pid) = r.term_id(&Term::Iri(pred))? else {
        return Ok(None);
    };
    let conds: Vec<PropCond> = conds.into_iter().flatten().collect();
    Ok(if conds.is_empty() {
        None
    } else {
        Some((pid, conds))
    })
}

/// Sorted-vector intersection.
fn intersect(a: Vec<Id>, b: &[Id]) -> Vec<Id> {
    let set: HashSet<Id> = b.iter().copied().collect();
    a.into_iter().filter(|x| set.contains(x)).collect()
}

fn types_of(r: &Reader, onto: &Ontology, id: Id) -> Result<Vec<TypeRef>> {
    let Some(t) = r.term_id(&Term::Iri(rdf::RDF_TYPE.into()))? else {
        return Ok(Vec::new());
    };
    r.objects(id, t)?
        .into_iter()
        .map(|o| {
            Ok(TypeRef {
                iri: onto.prefixes.display(&r.term(o)?),
                inferred: r.provenance(id, t, o)?.is_empty(),
            })
        })
        .collect()
}

pub(crate) fn label_of(r: &Reader, id: Id) -> Result<Option<String>> {
    let Some(lp) = r.term_id(&Term::Iri(rdf::RDFS_LABEL.into()))? else {
        return Ok(None);
    };
    Ok(match r.objects(id, lp)?.first() {
        Some(o) => match r.term(*o)? {
            Term::Literal(l) => Some(l.value),
            Term::Iri(i) => Some(i),
        },
        None => None,
    })
}

impl Kb {
    /// Entities matching all type and `--where` conditions, in stable (id) order.
    pub fn search(&self, q: &SearchQuery) -> Result<SearchPage> {
        if q.types.is_empty() && q.wheres.is_empty() && q.text.is_none() && q.semantic.is_none() {
            return Err(Error::other(
                "give at least one of --text, --semantic, --type, --where",
            ));
        }
        self.sync()?;
        let onto = self.ontology()?;
        let store = self.store_handle()?;
        // catch the text index up before taking the read snapshot (it needs the store)
        let fts = match &q.text {
            Some(_) => Some(self.fts_ready(&store, &onto)?),
            None => None,
        };
        let (qvec, pending) = match &q.semantic {
            Some(s) => (Some(self.embed_query(s)?), Some(self.vec_pending()?)),
            None => (None, None),
        };
        let r = store.read()?;

        let mut clauses: Vec<(String, Clause)> = q
            .types
            .iter()
            .map(|t| {
                (
                    format!("rdf:type = {t}"),
                    Clause::Cmp("rdf:type".into(), CmpOp::Eq, vec![Value::Bare(t.clone())]),
                )
            })
            .collect();
        for w in &q.wheres {
            clauses.push((w.clone(), parse_clause(w)?));
        }

        let mut result: Option<Vec<Id>> = None;
        let mut shown: Vec<Id> = Vec::new();
        for (text, clause) in &clauses {
            let Some((pid, conds)) = resolve(&onto, &r, text, clause)? else {
                result = Some(Vec::new());
                break;
            };
            if !text.starts_with("rdf:type") {
                shown.push(pid);
            }
            let mut ids: Vec<Id> = Vec::new();
            for c in &conds {
                ids.extend(r.subjects_where(pid, c)?);
            }
            ids.sort_unstable();
            ids.dedup();
            result = Some(match result {
                None => ids,
                Some(prev) => intersect(prev, &ids),
            });
            if result.as_ref().is_some_and(Vec::is_empty) {
                break;
            }
        }
        let offset: usize = match &q.cursor {
            Some(c) => c
                .parse()
                .map_err(|_| Error::other(format!("invalid cursor '{c}'")))?,
            None => 0,
        };
        let limit = q.limit.max(1);
        // (entity, score) in result order, and the total number of matches
        let window = offset + limit;
        let (page, total): (Vec<(Id, Option<f32>)>, usize) = match (&q.text, fts, &qvec) {
            (Some(text), Some(fts), Some(qv)) if q.text_mode == TextMode::Must => {
                // text is a condition: every text match (within the filters), ranked by meaning
                let candidates = result
                    .as_ref()
                    .map(|ids| Arc::new(ids.iter().copied().collect::<HashSet<u64>>()));
                let (_, n) = fts.search(text, candidates.clone(), 1, 0)?;
                let (matches, _) = fts.search(text, candidates, n.max(1), 0)?;
                let ids: Vec<Id> = matches.into_iter().map(|(id, _)| id).collect();
                let (ranked, total) = vectors::rank(&r, &self.index_dir(), qv, Some(&ids), window)?;
                (
                    ranked
                        .into_iter()
                        .skip(offset)
                        .map(|(id, sc)| (id, Some(sc)))
                        .collect(),
                    total,
                )
            }
            (Some(text), Some(fts), Some(qv)) => {
                // hybrid: fuse BM25 and cosine rankings (RRF), over the same candidates
                let pool = window * 4;
                let candidates = result
                    .as_ref()
                    .map(|ids| Arc::new(ids.iter().copied().collect::<HashSet<u64>>()));
                let (by_text, _) = fts.search(text, candidates, pool, 0)?;
                let (by_vec, _) =
                    vectors::rank(&r, &self.index_dir(), qv, result.as_deref(), pool)?;
                let lists = [
                    by_text.iter().map(|x| x.0).collect(),
                    by_vec.iter().map(|x| x.0).collect(),
                ];
                let all: HashSet<Id> = lists.iter().flatten().copied().collect();
                let fused = vectors::rrf(&lists, window);
                (
                    fused
                        .into_iter()
                        .skip(offset)
                        .map(|(id, sc)| (id, Some(sc)))
                        .collect(),
                    all.len(),
                )
            }
            (None, _, Some(qv)) => {
                let (ranked, total) =
                    vectors::rank(&r, &self.index_dir(), qv, result.as_deref(), window)?;
                (
                    ranked
                        .into_iter()
                        .skip(offset)
                        .map(|(id, sc)| (id, Some(sc)))
                        .collect(),
                    total,
                )
            }
            (Some(text), Some(fts), None) => {
                let candidates =
                    result.map(|ids| Arc::new(ids.into_iter().collect::<HashSet<u64>>()));
                let (ranked, total) = fts.search(text, candidates, limit, offset)?;
                (
                    ranked.into_iter().map(|(id, sc)| (id, Some(sc))).collect(),
                    total,
                )
            }
            _ => {
                let all = result.unwrap_or_default();
                (
                    all.iter()
                        .skip(offset)
                        .take(limit)
                        .map(|id| (*id, None))
                        .collect(),
                    all.len(),
                )
            }
        };
        let mut hits = Vec::new();
        for (id, score) in &page {
            let mut matched = Vec::new();
            for p in &shown {
                for o in r.objects(*id, *p)? {
                    matched.push((
                        onto.prefixes.display(&r.term(*p)?),
                        onto.prefixes.display(&r.term(o)?),
                    ));
                }
            }
            hits.push(Hit {
                iri: onto.prefixes.display(&r.term(*id)?),
                label: label_of(&r, *id)?,
                types: types_of(&r, &onto, *id)?,
                matched,
                score: *score,
            });
        }
        let next = offset + hits.len();
        Ok(SearchPage {
            total,
            pending,
            next_cursor: (next < total).then(|| next.to_string()),
            hits,
        })
    }

    /// Nodes reached from `iri` along a SPARQL property path (e.g. `ex:has_contract/rdfs:label`,
    /// `^ex:has_contract`, `ex:part_of+`).
    pub fn traverse(&self, iri: &str, path: &str, limit: usize) -> Result<Vec<Hit>> {
        let onto = self.ontology()?;
        let start = onto.prefixes.expand(iri).map_err(Error::Other)?;
        // The start entity is never a result, nor the node a final step is read from
        // (so `p/^p/rdfs:label` lists the partners' labels, not the start's own label).
        let query = match split_last_step(path) {
            Some((head, last)) => format!(
                "SELECT DISTINCT ?x WHERE {{ <{start}> {head} ?m . FILTER(?m != <{start}>) ?m {last} ?x }} ORDER BY ?x"
            ),
            None => format!("SELECT DISTINCT ?x WHERE {{ <{start}> {path} ?x }} ORDER BY ?x"),
        };
        let answer = self.query(
            &query,
            &Limits {
                rows: limit,
                ..Limits::default()
            },
        )?;
        let Answer::Solutions { rows, .. } = answer else {
            return Ok(Vec::new());
        };
        let store = self.store_handle()?;
        let r = store.read()?;
        let mut out = Vec::new();
        for row in rows {
            let Some(Some(term)) = row.into_iter().next() else {
                continue;
            };
            let t = sparql::from_ox(&term);
            // paths that come back (e.g. `p/^p`) would list the start entity itself
            if t.as_ref().and_then(Term::as_iri) == Some(start.as_str()) {
                continue;
            }
            let id = match &t {
                Some(t) => r.term_id(t)?,
                None => None,
            };
            out.push(Hit {
                iri: sparql::display(&onto.prefixes, &term),
                label: match (id, &t) {
                    (Some(id), Some(Term::Iri(_))) => label_of(&r, id)?,
                    _ => None,
                },
                types: match (id, &t) {
                    (Some(id), Some(Term::Iri(_))) => types_of(&r, &onto, id)?,
                    _ => Vec::new(),
                },
                matched: Vec::new(),
                score: None,
            });
        }
        Ok(out)
    }

    /// Shortest chain of entity-to-entity edges between two entities, followed in either
    /// direction. `rdf:type` edges are skipped (every instance would connect through its class).
    pub fn path(&self, from: &str, to: &str, max_depth: usize) -> Result<Option<Vec<Step>>> {
        const MAX_VISITED: usize = 200_000;
        self.sync()?;
        let onto = self.ontology()?;
        let store = self.store_handle()?;
        let r = store.read()?;
        let id_of = |x: &str| -> Result<Id> {
            let iri = onto.prefixes.expand(x).map_err(Error::Other)?;
            r.term_id(&Term::Iri(iri))?
                .ok_or_else(|| Error::NotFound(format!("entity {x}")))
        };
        let (a, b) = (id_of(from)?, id_of(to)?);
        let type_id = r.term_id(&Term::Iri(rdf::RDF_TYPE.into()))?;
        let mut is_iri: HashMap<Id, bool> = HashMap::new();
        let mut parent: HashMap<Id, (Id, Id, bool)> = HashMap::new();
        let mut queue = VecDeque::from([(a, 0usize)]);
        let mut seen = HashSet::from([a]);
        while let Some((node, depth)) = queue.pop_front() {
            if node == b {
                break;
            }
            if depth >= max_depth || seen.len() > MAX_VISITED {
                continue;
            }
            let mut edges: Vec<(Id, Id, bool)> = r
                .outgoing(node)?
                .into_iter()
                .map(|(p, o)| (p, o, true))
                .collect();
            edges.extend(r.incoming(node)?.into_iter().map(|(s, p)| (p, s, false)));
            for (p, next, forward) in edges {
                if Some(p) == type_id || seen.contains(&next) {
                    continue;
                }
                let iri = match is_iri.get(&next) {
                    Some(v) => *v,
                    None => {
                        let v = matches!(r.term(next)?, Term::Iri(_));
                        is_iri.insert(next, v);
                        v
                    }
                };
                if !iri {
                    continue;
                }
                seen.insert(next);
                parent.insert(next, (node, p, forward));
                queue.push_back((next, depth + 1));
            }
        }
        if a != b && !parent.contains_key(&b) {
            return Ok(None);
        }
        let name = |id: Id| -> Result<String> {
            let iri = onto.prefixes.display(&r.term(id)?);
            Ok(match label_of(&r, id)? {
                Some(l) => format!("{iri} \"{l}\""),
                None => iri,
            })
        };
        let mut steps = Vec::new();
        let mut cur = b;
        while let Some(&(prev, p, forward)) = parent.get(&cur) {
            let pname = onto.prefixes.display(&r.term(p)?);
            steps.push(Step {
                from: name(prev)?,
                p: pname,
                to: name(cur)?,
                forward,
            });
            cur = prev;
        }
        steps.reverse();
        Ok(Some(steps))
    }
}

/// Split a SPARQL property path at its last top-level `/` (outside parentheses, `<IRI>`s
/// and quotes): `a/(b|c)/d` -> (`a/(b|c)`, `d`).
fn split_last_step(path: &str) -> Option<(&str, &str)> {
    let (mut depth, mut in_iri, mut in_quote, mut at) = (0i32, false, false, None);
    for (i, c) in path.char_indices() {
        match c {
            '"' if !in_iri => in_quote = !in_quote,
            '<' if !in_quote => in_iri = true,
            '>' if !in_quote => in_iri = false,
            '(' if !in_iri && !in_quote => depth += 1,
            ')' if !in_iri && !in_quote => depth -= 1,
            '/' if depth == 0 && !in_iri && !in_quote => at = Some(i),
            _ => {}
        }
    }
    let i = at?;
    let (head, last) = (path[..i].trim(), path[i + 1..].trim());
    (!head.is_empty() && !last.is_empty()).then_some((head, last))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_property_paths_at_the_last_top_level_step() {
        assert_eq!(split_last_step("a/b/c"), Some(("a/b", "c")));
        assert_eq!(
            split_last_step("ex:p/(ex:q/ex:r)"),
            Some(("ex:p", "(ex:q/ex:r)"))
        );
        assert_eq!(
            split_last_step("<http://x/a/b>/ex:q"),
            Some(("<http://x/a/b>", "ex:q"))
        );
        assert_eq!(split_last_step("<http://x/a/b>"), None);
        assert_eq!(split_last_step("ex:p|ex:q"), None);
    }

    #[test]
    fn parses_where_clauses() {
        assert_eq!(
            parse_clause("has ex:email").unwrap(),
            Clause::Has("ex:email".into())
        );
        assert_eq!(
            parse_clause("ex:value>=10000").unwrap(),
            Clause::Cmp(
                "ex:value".into(),
                CmpOp::Ge,
                vec![Value::Bare("10000".into())]
            )
        );
        assert_eq!(
            parse_clause("ex:country in (\"VN\", \"S,G\")").unwrap(),
            Clause::Cmp(
                "ex:country".into(),
                CmpOp::Eq,
                vec![Value::Quoted("VN".into()), Value::Quoted("S,G".into())]
            )
        );
        assert_eq!(
            parse_clause("<http://x/p> != \"a\"").unwrap(),
            Clause::Cmp(
                "<http://x/p>".into(),
                CmpOp::Ne,
                vec![Value::Quoted("a".into())]
            )
        );
        assert!(parse_clause("ex:value 10").is_err());
    }
}
