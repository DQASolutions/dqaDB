//! `dqadb import`: load an RDF file (N-Triples, N-Quads, Turtle, TriG, RDF/XML, JSON-LD)
//! as ordinary events. The file is copied once into `sources/` (by content hash) and
//! every event of the import refers to it; the triples themselves go into the events in
//! batches, so the events stay the self-contained source of truth.
//!
//! Two passes: the whole file is first parsed and validated against the ontology
//! (nothing written; `--dry-run` stops here), then written batch by batch. Blank nodes
//! get one minted `dqadb:` IRI per label for the whole file.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use oxrdf::{NamedOrBlankNode, Term as OxTerm};
use oxrdfio::{RdfFormat, RdfParser};
use serde::Serialize;
use ulid::Ulid;

use crate::error::ValidationReport;
use crate::event::OpKind;
use crate::ontology::InputOp;
use crate::rdf;
use crate::{Error, Kb, Result, WriteOpts};

/// Triples per event unless told otherwise.
pub const DEFAULT_BATCH: usize = 50_000;

#[derive(Debug, Clone)]
pub struct ImportOpts {
    /// Format name or file extension (nt, nq, ttl, trig, rdf/xml, jsonld); default: from
    /// the file extension.
    pub format: Option<String>,
    pub batch: usize,
    pub dry_run: bool,
}

impl Default for ImportOpts {
    fn default() -> Self {
        Self {
            format: None,
            batch: DEFAULT_BATCH,
            dry_run: false,
        }
    }
}

/// One kind of validation problem, counted over the whole file.
#[derive(Debug, Clone, Serialize)]
pub struct ImportIssue {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fix: Option<String>,
    pub count: usize,
    /// 1-based position of the first triple with this problem
    pub first_triple: usize,
}

#[derive(Debug, Default, Serialize)]
pub struct ImportReport {
    pub file: String,
    pub format: String,
    pub triples: usize,
    pub blank_nodes: usize,
    pub dry_run: bool,
    /// events written (empty for a dry run or when validation failed)
    pub events: Vec<String>,
    pub issues: Vec<ImportIssue>,
    pub seconds: f64,
}

pub fn format_of(path: &Path, given: Option<&str>) -> Result<RdfFormat> {
    let name = match given {
        Some(f) => f.to_ascii_lowercase(),
        None => {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            // foo.nt.zst etc. are not supported yet: the parser reads plain text
            ext.to_ascii_lowercase()
        }
    };
    let format = match name.as_str() {
        "nt" | "ntriples" | "n-triples" => RdfFormat::NTriples,
        "nq" | "nquads" | "n-quads" => RdfFormat::NQuads,
        "ttl" | "turtle" => RdfFormat::Turtle,
        "trig" => RdfFormat::TriG,
        "rdf" | "xml" | "owl" | "rdfxml" | "rdf/xml" => RdfFormat::RdfXml,
        "jsonld" | "json-ld" => RdfFormat::JsonLd {
            profile: Default::default(),
        },
        other => {
            return Err(Error::other(format!(
                "unknown RDF format '{other}' for {}; use --format nt|nq|ttl|trig|rdfxml|jsonld",
                path.display()
            )));
        }
    };
    Ok(format)
}

/// Streams the file as dqaDB input ops, `batch` at a time. Blank nodes are mapped to
/// minted IRIs (stable for the whole file); quads outside the default graph are an
/// error, since dqaDB's named graphs are its events.
struct Reader<'a> {
    quads: Box<dyn Iterator<Item = std::result::Result<oxrdf::Quad, oxrdfio::RdfParseError>> + 'a>,
    blanks: HashMap<String, String>,
    mint: Box<dyn FnMut() -> String + 'a>,
    read: usize,
}

impl Reader<'_> {
    fn next_batch(&mut self, n: usize) -> Result<Vec<InputOp>> {
        let mut out = Vec::with_capacity(n.min(1 << 16));
        while out.len() < n {
            let Some(q) = self.quads.next() else { break };
            self.read += 1;
            let q = q.map_err(|e| Error::other(format!("triple {}: {e}", self.read)))?;
            if !q.graph_name.is_default_graph() {
                return Err(Error::other(format!(
                    "triple {}: named graph {} is not supported (dqaDB's named graphs are its events); import the triples into the default graph",
                    self.read, q.graph_name
                )));
            }
            let s = match q.subject {
                NamedOrBlankNode::NamedNode(n) => format!("<{}>", n.as_str()),
                NamedOrBlankNode::BlankNode(b) => self.blank(b.as_str()),
            };
            let o = match q.object {
                OxTerm::NamedNode(n) => serde_json::Value::String(format!("<{}>", n.as_str())),
                OxTerm::BlankNode(b) => serde_json::Value::String(self.blank(b.as_str())),
                OxTerm::Literal(l) => {
                    let mut m = serde_json::Map::new();
                    m.insert("@value".into(), l.value().into());
                    match l.language() {
                        Some(lang) => m.insert("@language".into(), lang.into()),
                        None => m.insert(
                            "@type".into(),
                            format!("<{}>", l.datatype().as_str()).into(),
                        ),
                    };
                    serde_json::Value::Object(m)
                }
                #[allow(unreachable_patterns)]
                _ => {
                    return Err(Error::other(format!(
                        "triple {}: RDF 1.2 triple terms are not supported",
                        self.read
                    )));
                }
            };
            out.push(InputOp {
                op: OpKind::Add,
                s,
                p: format!("<{}>", q.predicate.as_str()),
                o,
            });
        }
        Ok(out)
    }

    fn blank(&mut self, label: &str) -> String {
        if let Some(iri) = self.blanks.get(label) {
            return iri.clone();
        }
        let iri = format!("<{}>", (self.mint)());
        self.blanks.insert(label.to_string(), iri.clone());
        iri
    }
}

fn open<'a>(path: &Path, format: RdfFormat, base: &'a str) -> Result<Reader<'a>> {
    let file = BufReader::with_capacity(1 << 20, File::open(path)?);
    let quads = RdfParser::from_format(format).for_reader(file);
    Ok(Reader {
        quads: Box::new(quads),
        blanks: HashMap::new(),
        mint: Box::new(move || format!("{base}{}", Ulid::generate())),
        read: 0,
    })
}

fn unbracket(s: &str) -> &str {
    s.strip_prefix('<')
        .and_then(|x| x.strip_suffix('>'))
        .unwrap_or(s)
}

/// Merge a batch's validation issues into per-message counts. Issue paths are
/// `ops[i]...`; `offset` turns `i` into the 1-based triple number in the file.
fn tally(report: ValidationReport, offset: usize, into: &mut BTreeMap<String, ImportIssue>) {
    for issue in report.issues {
        let i = issue
            .at
            .strip_prefix("ops[")
            .and_then(|r| r.split(']').next())
            .and_then(|n| n.parse::<usize>().ok())
            .unwrap_or(0);
        into.entry(issue.message.clone())
            .and_modify(|e| e.count += 1)
            .or_insert(ImportIssue {
                message: issue.message,
                fix: issue.fix,
                count: 1,
                first_triple: offset + i + 1,
            });
    }
}

impl Kb {
    /// Import an RDF file as events (see the module docs). `progress` is called with the
    /// number of triples processed so far, once per batch, in each pass.
    pub fn import(
        &self,
        path: &Path,
        opts: &ImportOpts,
        write: &WriteOpts,
        progress: &mut dyn FnMut(&str, usize),
    ) -> Result<ImportReport> {
        let started = Instant::now();
        if !path.is_file() {
            return Err(Error::other(format!("{}: no such file", path.display())));
        }
        let format = format_of(path, opts.format.as_deref())?;
        let batch = opts.batch.max(1);
        let base = self.config.base_iri.clone();
        let mut report = ImportReport {
            file: path.display().to_string(),
            format: format.name().to_string(),
            dry_run: opts.dry_run,
            ..Default::default()
        };

        // pass 1: parse and validate everything, write nothing. Required slots are
        // checked over the whole file (an entity's type and label may sit in different
        // batches): typed subjects -> required slots, and the (subject, required slot)
        // pairs the file provides.
        let onto = self.ontology()?;
        let mut issues: BTreeMap<String, ImportIssue> = BTreeMap::new();
        let mut needs: HashMap<String, Vec<(String, String, usize)>> = HashMap::new();
        let mut provided: HashSet<(String, String)> = HashSet::new();
        let mut reader = open(path, format, &base)?;
        loop {
            let offset = reader.read;
            let ops = reader.next_batch(batch)?;
            if ops.is_empty() {
                break;
            }
            for (i, op) in ops.iter().enumerate() {
                let (s, p) = (unbracket(&op.s), unbracket(&op.p));
                if p == rdf::RDF_TYPE {
                    if let Some(class) = op.o.as_str() {
                        for (cname, slot) in onto.required_slots(unbracket(class)) {
                            needs.entry(s.to_string()).or_default().push((
                                cname,
                                slot,
                                offset + i + 1,
                            ));
                        }
                    }
                } else if onto.is_required(p) {
                    provided.insert((s.to_string(), p.to_string()));
                }
            }
            match self.validate_with(&ops, false) {
                Ok(_) => {}
                Err(Error::Validation(r)) => tally(r, offset, &mut issues),
                Err(e) => return Err(e),
            }
            progress("validated", reader.read);
        }
        report.triples = reader.read;
        report.blank_nodes = reader.blanks.len();
        if !needs.is_empty() {
            let store = self.store()?;
            let r = store.read()?;
            for (s, list) in needs {
                for (cname, slot, at) in list {
                    if provided.contains(&(s.clone(), slot.clone()))
                        || !r.values(&s, &slot)?.is_empty()
                    {
                        continue;
                    }
                    let message = format!("{cname} requires {}", onto.short(&slot));
                    issues
                        .entry(message.clone())
                        .and_modify(|e| {
                            e.count += 1;
                            e.first_triple = e.first_triple.min(at);
                        })
                        .or_insert(ImportIssue {
                            message,
                            fix: Some(format!(
                                "add a {} triple for every such entity",
                                onto.short(&slot)
                            )),
                            count: 1,
                            first_triple: at,
                        });
                }
            }
        }
        report.issues = issues.into_values().collect();
        if report.triples == 0 {
            return Err(Error::other(format!("{}: no triples", path.display())));
        }
        if opts.dry_run || !report.issues.is_empty() {
            report.seconds = started.elapsed().as_secs_f64();
            return Ok(report);
        }

        // pass 2: write, one event per batch; the file is stored once
        let source = self.store_source_once(path)?;
        let mut reader = open(path, format, &base)?;
        let mut each = write.clone();
        each.source = None;
        let parts = report.triples.div_ceil(batch);
        loop {
            let ops = reader.next_batch(batch)?;
            if ops.is_empty() {
                break;
            }
            let part = format!(
                "import {}, part {}/{parts}",
                report.file,
                report.events.len() + 1
            );
            each.note = Some(match &write.note {
                Some(n) => format!("{n} ({part})"),
                None => part,
            });
            let r = self.assert_with_source(&ops, &each, Some(source.clone()), false)?;
            report.events.push(r.event_id);
            progress("written", reader.read);
        }
        report.seconds = started.elapsed().as_secs_f64();
        Ok(report)
    }
}
