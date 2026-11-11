//! SPARQL 1.1 over the index (spec §11), using Oxigraph's evaluator (`spareval`).
//!
//! Dataset model:
//! - default graph = the current state: asserted and inferred triples;
//! - named graph `<base>event:<ULID>` = the triples that event asserted and that are
//!   still in effect (provenance). `GRAPH ?g { ... }` therefore answers "who said this".

use oxrdf::{Literal as OxLiteral, NamedNode, Term as OxTerm, Triple};
use spareval::{CancellationToken, InternalQuad, QueryEvaluator, QueryResults, QueryableDataset};
use spargebra::SparqlParser;
use std::path::Path;
use std::time::Duration;
use ulid::Ulid;

use crate::error::ValidationReport;
use crate::event::{self, OpKind};
use crate::rdf::{Prefixes, Term, XSD_STRING};
use crate::store::{Id, Reader};
use crate::{Error, Result};

/// Term as seen by the evaluator: a dictionary id, an event graph, or a term that is not
/// in the index (a query constant or a computed value) and therefore matches nothing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DTerm {
    Id(Id),
    Event(Ulid),
    Ext(OxTerm),
}

#[derive(Clone, Copy)]
pub struct KbDataset<'a> {
    reader: &'a Reader<'a>,
    event_ns: &'a str,
    events_dir: &'a Path,
    cache: &'a TermCache,
}

/// Decoded terms for the duration of one query (FILTER, ORDER BY and joins decode the
/// same ids again and again; each decode is a dictionary lookup). Jena TDB keeps a
/// NodeId -> Node cache for the same reason. Ids are stable while the reader's snapshot
/// lives, so the cache needs no invalidation.
#[derive(Default)]
pub struct TermCache {
    terms: std::cell::RefCell<std::collections::HashMap<Id, OxTerm>>,
    values: std::cell::RefCell<std::collections::HashMap<Id, spareval::ExpressionTerm>>,
}

type Quads<'a> = Box<dyn Iterator<Item = Result<InternalQuad<DTerm>>> + 'a>;
type Triples<'a> = Box<dyn Iterator<Item = Result<(Id, Id, Id)>> + 'a>;

pub fn to_ox(t: &Term) -> OxTerm {
    match t {
        Term::Iri(i) => NamedNode::new_unchecked(i.clone()).into(),
        Term::Literal(l) => match (&l.language, &l.datatype) {
            (Some(lang), _) => {
                OxLiteral::new_language_tagged_literal_unchecked(l.value.clone(), lang.clone())
                    .into()
            }
            (None, Some(dt)) if dt != XSD_STRING => {
                OxLiteral::new_typed_literal(l.value.clone(), NamedNode::new_unchecked(dt.clone()))
                    .into()
            }
            _ => OxLiteral::new_simple_literal(l.value.clone()).into(),
        },
    }
}

pub fn from_ox(t: &OxTerm) -> Option<Term> {
    match t {
        OxTerm::NamedNode(n) => Some(Term::Iri(n.as_str().to_string())),
        OxTerm::Literal(l) => Some(match l.language() {
            Some(lang) => Term::lang(l.value(), lang),
            None => Term::typed(l.value(), l.datatype().as_str()),
        }),
        _ => None,
    }
}

impl<'a> KbDataset<'a> {
    pub fn new(
        reader: &'a Reader<'a>,
        event_ns: &'a str,
        events_dir: &'a Path,
        cache: &'a TermCache,
    ) -> Self {
        Self {
            reader,
            event_ns,
            events_dir,
            cache,
        }
    }

    fn decode(&self, id: Id) -> Result<OxTerm> {
        if let Some(t) = self.cache.terms.borrow().get(&id) {
            return Ok(t.clone());
        }
        let t = to_ox(&self.reader.term(id)?);
        self.cache.terms.borrow_mut().insert(id, t.clone());
        Ok(t)
    }

    /// Triples of the default graph matching a pattern of dictionary ids.
    fn triples(self, s: Option<Id>, p: Option<Id>, o: Option<Id>) -> Result<Triples<'a>> {
        if let (Some(s), Some(p), Some(o)) = (s, p, o) {
            let hit = self.reader.has_triple(s, p, o)?;
            return Ok(Box::new(hit.then_some(Ok((s, p, o))).into_iter()));
        }
        Ok(Box::new(self.reader.scan(s, p, o)))
    }

    /// Triples one event asserted that are still in effect.
    fn event_triples(
        self,
        id: Ulid,
        s: Option<Id>,
        p: Option<Id>,
        o: Option<Id>,
    ) -> Result<Vec<(Id, Id, Id)>> {
        let Ok(path) = event::find_event(self.events_dir, id) else {
            return Ok(Vec::new());
        };
        let ev = event::read_event(&path)?;
        let r = self.reader;
        let mut out = Vec::new();
        for op in ev.ops.iter().filter(|op| op.op == OpKind::Add) {
            let ids = (
                r.term_id(&Term::Iri(op.s.clone()))?,
                r.term_id(&Term::Iri(op.p.clone()))?,
                r.term_id(&op.o)?,
            );
            let (Some(ts), Some(tp), Some(to)) = ids else {
                continue;
            };
            if s.is_some_and(|x| x != ts)
                || p.is_some_and(|x| x != tp)
                || o.is_some_and(|x| x != to)
            {
                continue;
            }
            if r.asserted_by(ts, tp, to, id)? {
                out.push((ts, tp, to));
            }
        }
        out.sort();
        out.dedup();
        Ok(out)
    }

    fn quads(
        self,
        s: Option<&DTerm>,
        p: Option<&DTerm>,
        o: Option<&DTerm>,
        g: Option<Option<&DTerm>>,
    ) -> Result<Quads<'a>> {
        // a bound term that is not in the dictionary can never match
        let id = |t: Option<&DTerm>| match t {
            None => Some(None),
            Some(DTerm::Id(i)) => Some(Some(*i)),
            Some(_) => None,
        };
        let (Some(s), Some(p), Some(o)) = (id(s), id(p), id(o)) else {
            return Ok(Box::new(std::iter::empty()));
        };
        let quad = |(a, b, c): (Id, Id, Id), g: Option<DTerm>| InternalQuad {
            subject: DTerm::Id(a),
            predicate: DTerm::Id(b),
            object: DTerm::Id(c),
            graph_name: g,
        };
        match g {
            Some(None) => Ok(Box::new(
                self.triples(s, p, o)?
                    .map(move |r| r.map(|t| quad(t, None))),
            )),
            Some(Some(DTerm::Event(e))) => {
                let e = *e;
                let found = self.event_triples(e, s, p, o)?;
                Ok(Box::new(
                    found
                        .into_iter()
                        .map(move |t| Ok(quad(t, Some(DTerm::Event(e))))),
                ))
            }
            Some(Some(_)) => Ok(Box::new(std::iter::empty())),
            None => {
                let r = self.reader;
                Ok(Box::new(self.triples(s, p, o)?.flat_map(
                    move |t| -> Quads<'a> {
                        let t = match t {
                            Ok(t) => t,
                            Err(e) => return Box::new(std::iter::once(Err(e))),
                        };
                        match r.provenance(t.0, t.1, t.2) {
                            Ok(events) => Box::new(
                                events
                                    .into_iter()
                                    .map(move |e| Ok(quad(t, Some(DTerm::Event(e))))),
                            ),
                            Err(e) => Box::new(std::iter::once(Err(e))),
                        }
                    },
                )))
            }
        }
    }
}

impl<'a> QueryableDataset<'a> for KbDataset<'a> {
    type InternalTerm = DTerm;
    type Error = Error;

    fn internal_quads_for_pattern(
        &self,
        subject: Option<&DTerm>,
        predicate: Option<&DTerm>,
        object: Option<&DTerm>,
        graph_name: Option<Option<&DTerm>>,
    ) -> impl Iterator<Item = Result<InternalQuad<DTerm>>> + use<'a> {
        match self.quads(subject, predicate, object, graph_name) {
            Ok(it) => it,
            Err(e) => Box::new(std::iter::once(Err(e))),
        }
    }

    fn internalize_term(&self, term: OxTerm) -> Result<DTerm> {
        if let OxTerm::NamedNode(n) = &term
            && let Some(id) = n
                .as_str()
                .strip_prefix(self.event_ns)
                .and_then(|u| Ulid::from_string(u).ok())
        {
            return Ok(DTerm::Event(id));
        }
        Ok(match from_ox(&term) {
            Some(t) => match self.reader.term_id(&t)? {
                Some(id) => DTerm::Id(id),
                None => DTerm::Ext(term),
            },
            None => DTerm::Ext(term),
        })
    }

    fn externalize_expression_term(&self, term: DTerm) -> Result<spareval::ExpressionTerm> {
        let DTerm::Id(id) = term else {
            return Ok(self.externalize_term(term)?.into());
        };
        if let Some(v) = self.cache.values.borrow().get(&id) {
            return Ok(v.clone());
        }
        let v: spareval::ExpressionTerm = self.decode(id)?.into();
        self.cache.values.borrow_mut().insert(id, v.clone());
        Ok(v)
    }

    fn externalize_term(&self, term: DTerm) -> Result<OxTerm> {
        Ok(match term {
            DTerm::Id(id) => self.decode(id)?,
            DTerm::Event(e) => NamedNode::new_unchecked(format!("{}{e}", self.event_ns)).into(),
            DTerm::Ext(t) => t,
        })
    }
}

/// Query answer, already materialised and bounded by `limit`.
#[derive(Debug)]
pub enum Answer {
    Solutions {
        vars: Vec<String>,
        rows: Vec<Vec<Option<OxTerm>>>,
        truncated: bool,
    },
    Boolean(bool),
    Graph {
        triples: Vec<Triple>,
        truncated: bool,
    },
}

pub struct Limits {
    pub rows: usize,
    pub timeout: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            rows: 1000,
            timeout: Duration::from_secs(10),
        }
    }
}

/// Parse (with the KB's prefixes predeclared) and evaluate a query.
pub fn run(
    dataset: KbDataset<'_>,
    prefixes: &Prefixes,
    query: &str,
    limits: &Limits,
) -> Result<Answer> {
    let mut parser = SparqlParser::new();
    for (p, ns) in prefixes.pairs() {
        parser = parser
            .with_prefix(p, ns)
            .map_err(|e| Error::other(format!("prefix {p}: {e}")))?;
    }
    let parsed = parser.parse_query(query).map_err(|e| {
        let mut r = ValidationReport::default();
        let declared = prefixes.names().join(", ");
        r.push(
            "query",
            format!("SPARQL syntax error: {e}"),
            Some(format!("prefixes {declared} are predeclared")),
        );
        Error::Validation(r)
    })?;

    // watchdog: cancels the query at the timeout, and exits as soon as the query is done
    // (dropping `_done` disconnects the channel)
    let token = CancellationToken::new();
    let timer = token.clone();
    let timeout = limits.timeout;
    let (_done, done) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        if let Err(std::sync::mpsc::RecvTimeoutError::Timeout) = done.recv_timeout(timeout) {
            timer.cancel();
        }
    });
    let evaluator = QueryEvaluator::new().with_cancellation_token(token);
    let timed_out = |e: spareval::QueryEvaluationError| {
        if matches!(e, spareval::QueryEvaluationError::Cancelled) {
            Error::other(format!(
                "query cancelled after {}s; add LIMIT or narrow the pattern",
                timeout.as_secs()
            ))
        } else {
            Error::other(format!("query evaluation failed: {e}"))
        }
    };
    let results = evaluator
        .prepare(&parsed)
        .execute(dataset)
        .map_err(timed_out)?;
    Ok(match results {
        QueryResults::Boolean(b) => Answer::Boolean(b),
        QueryResults::Solutions(sols) => {
            let vars: Vec<String> = sols
                .variables()
                .iter()
                .map(|v| v.as_str().to_string())
                .collect();
            let mut rows = Vec::new();
            let mut truncated = false;
            for sol in sols {
                if rows.len() >= limits.rows {
                    truncated = true;
                    break;
                }
                let sol = sol.map_err(timed_out)?;
                rows.push(vars.iter().map(|v| sol.get(v.as_str()).cloned()).collect());
            }
            Answer::Solutions {
                vars,
                rows,
                truncated,
            }
        }
        QueryResults::Graph(triples) => {
            let mut out = Vec::new();
            let mut truncated = false;
            for t in triples {
                if out.len() >= limits.rows {
                    truncated = true;
                    break;
                }
                out.push(t.map_err(timed_out)?);
            }
            Answer::Graph {
                triples: out,
                truncated,
            }
        }
    })
}

/// SPARQL 1.1 Query Results JSON Format (W3C), plus a `truncated` flag.
pub fn to_json(answer: &Answer) -> serde_json::Value {
    fn term(t: &OxTerm) -> serde_json::Value {
        match t {
            OxTerm::NamedNode(n) => serde_json::json!({"type": "uri", "value": n.as_str()}),
            OxTerm::BlankNode(b) => serde_json::json!({"type": "bnode", "value": b.as_str()}),
            OxTerm::Literal(l) => {
                let mut v = serde_json::json!({"type": "literal", "value": l.value()});
                if let Some(lang) = l.language() {
                    v["xml:lang"] = lang.into();
                } else if l.datatype().as_str() != XSD_STRING {
                    v["datatype"] = l.datatype().as_str().into();
                }
                v
            }
            #[allow(unreachable_patterns)]
            other => serde_json::json!({"type": "unknown", "value": other.to_string()}),
        }
    }
    match answer {
        Answer::Boolean(b) => serde_json::json!({"head": {}, "boolean": b}),
        Answer::Solutions {
            vars,
            rows,
            truncated,
        } => {
            let bindings: Vec<serde_json::Value> = rows
                .iter()
                .map(|row| {
                    let mut m = serde_json::Map::new();
                    for (v, t) in vars.iter().zip(row) {
                        if let Some(t) = t {
                            m.insert(v.clone(), term(t));
                        }
                    }
                    serde_json::Value::Object(m)
                })
                .collect();
            serde_json::json!({"head": {"vars": vars}, "results": {"bindings": bindings}, "truncated": truncated})
        }
        Answer::Graph { triples, truncated } => {
            let ts: Vec<serde_json::Value> = triples
                .iter()
                .map(|t| serde_json::json!({"s": term(&t.subject.clone().into()), "p": term(&t.predicate.clone().into()), "o": term(&t.object)}))
                .collect();
            serde_json::json!({"triples": ts, "truncated": truncated})
        }
    }
}

/// Compact display of a term for text output.
pub fn display(prefixes: &Prefixes, t: &OxTerm) -> String {
    match from_ox(t) {
        Some(Term::Iri(i)) => prefixes.compact(&i),
        Some(Term::Literal(l)) => match l.language {
            Some(lang) => format!("\"{}\"@{lang}", l.value),
            None => match l.datatype.as_deref() {
                Some(XSD_STRING) | None => format!("\"{}\"", l.value),
                Some(dt)
                    if dt.ends_with("#integer")
                        || dt.ends_with("#decimal")
                        || dt.ends_with("#double")
                        || dt.ends_with("#boolean") =>
                {
                    l.value
                }
                Some(dt) => format!("\"{}\"^^{}", l.value, prefixes.compact(dt)),
            },
        },
        None => t.to_string(),
    }
}
