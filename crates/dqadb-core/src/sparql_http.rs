//! SPARQL 1.1 Protocol (query operation, read-only), independent of the HTTP server:
//! `dqadb serve` passes the request parts in and writes the response out.
//!
//! - `GET ?query=...`, `POST` form (`query=...`), `POST` `application/sparql-query`
//! - results: SPARQL JSON (default), XML, CSV, TSV; graphs: N-Triples (default), Turtle,
//!   RDF/XML, JSON-LD; chosen from `Accept` (q-values honoured), 406 otherwise
//! - updates are refused: dqaDB changes only through events (`dqadb assert`/`import`)
//! - `default-graph-uri` / `named-graph-uri` are refused: the dataset is fixed (default
//!   graph = the KB, named graphs = its events)

use oxrdf::Variable;
use oxrdfio::{RdfFormat, RdfSerializer};
use sparesults::{QueryResultsFormat, QueryResultsSerializer};

use crate::sparql::{Answer, Limits};
use crate::{Error, Kb};

pub struct Request<'a> {
    /// `GET` or `POST`
    pub method: &'a str,
    /// raw URL query string (after `?`)
    pub url_query: Option<&'a str>,
    pub content_type: Option<&'a str>,
    pub accept: Option<&'a str>,
    pub body: &'a [u8],
}

#[derive(Debug)]
pub struct Response {
    pub status: u16,
    pub content_type: String,
    pub body: Vec<u8>,
}

fn text(status: u16, msg: impl Into<String>) -> Response {
    Response {
        status,
        content_type: "text/plain; charset=utf-8".into(),
        body: msg.into().into_bytes(),
    }
}

/// The query of a protocol request, or an error response.
fn query_of(req: &Request<'_>) -> Result<String, Response> {
    let mut params: Vec<(String, String)> = req
        .url_query
        .map(|q| form_urlencoded::parse(q.as_bytes()).into_owned().collect())
        .unwrap_or_default();
    let mime = req.content_type.map(|c| {
        c.split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase()
    });
    let mut direct = None;
    match (req.method, mime.as_deref()) {
        ("GET", _) => {}
        ("POST", Some("application/x-www-form-urlencoded")) => {
            params.extend(form_urlencoded::parse(req.body).into_owned());
        }
        ("POST", Some("application/sparql-query")) => {
            direct = Some(
                String::from_utf8(req.body.to_vec())
                    .map_err(|_| text(400, "the query is not valid UTF-8"))?,
            );
        }
        ("POST", Some("application/sparql-update")) => {
            return Err(text(400, UPDATES_REFUSED));
        }
        ("POST", _) => {
            return Err(text(
                415,
                "POST needs Content-Type application/sparql-query or application/x-www-form-urlencoded",
            ));
        }
        (m, _) => {
            return Err(text(
                405,
                format!("method {m} not allowed; use GET or POST"),
            ));
        }
    }
    for (k, _) in &params {
        match k.as_str() {
            "update" | "using-graph-uri" | "using-named-graph-uri" => {
                return Err(text(400, UPDATES_REFUSED));
            }
            "default-graph-uri" | "named-graph-uri" => {
                return Err(text(
                    400,
                    "default-graph-uri / named-graph-uri are not supported: the default graph is the KB and its named graphs are its events (use GRAPH in the query)",
                ));
            }
            _ => {}
        }
    }
    let mut queries: Vec<String> = params
        .into_iter()
        .filter(|(k, _)| k == "query")
        .map(|(_, v)| v)
        .collect();
    queries.extend(direct);
    match queries.len() {
        1 => Ok(queries.pop().unwrap()),
        0 => Err(text(400, "missing query (use ?query=... or a POST body)")),
        _ => Err(text(400, "more than one query given")),
    }
}

const UPDATES_REFUSED: &str = "SPARQL Update is not supported: dqaDB changes only through events (dqadb assert / dqadb import)";

/// Media ranges of an Accept header, best first (q-value, then order).
fn accepted(accept: Option<&str>) -> Vec<String> {
    let Some(accept) = accept.filter(|a| !a.trim().is_empty()) else {
        return vec!["*/*".into()];
    };
    let mut ranges: Vec<(f32, usize, String)> = accept
        .split(',')
        .enumerate()
        .filter_map(|(i, part)| {
            let mut it = part.split(';');
            let range = it.next()?.trim().to_ascii_lowercase();
            let q = it
                .filter_map(|p| p.trim().strip_prefix("q="))
                .find_map(|v| v.parse::<f32>().ok())
                .unwrap_or(1.0);
            (q > 0.0 && !range.is_empty()).then_some((q, i, range))
        })
        .collect();
    ranges.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    ranges.into_iter().map(|(_, _, r)| r).collect()
}

/// First supported format for the accepted ranges; `formats` lists (media type, format)
/// with the default first.
fn pick<F: Copy>(accept: Option<&str>, formats: &[(&'static str, F)]) -> Option<(&'static str, F)> {
    for range in accepted(accept) {
        if range == "*/*" {
            return formats.first().copied();
        }
        if let Some(main) = range.strip_suffix("/*") {
            if let Some(f) = formats
                .iter()
                .find(|(m, _)| m.starts_with(&format!("{main}/")))
            {
                return Some(*f);
            }
            continue;
        }
        if let Some(f) = formats.iter().find(|(m, _)| *m == range) {
            return Some(*f);
        }
    }
    None
}

const RESULT_FORMATS: [(&str, QueryResultsFormat); 5] = [
    ("application/sparql-results+json", QueryResultsFormat::Json),
    ("application/sparql-results+xml", QueryResultsFormat::Xml),
    ("text/csv", QueryResultsFormat::Csv),
    ("text/tab-separated-values", QueryResultsFormat::Tsv),
    ("application/json", QueryResultsFormat::Json),
];

fn graph_formats() -> [(&'static str, RdfFormat); 5] {
    [
        ("application/n-triples", RdfFormat::NTriples),
        ("text/turtle", RdfFormat::Turtle),
        ("application/rdf+xml", RdfFormat::RdfXml),
        (
            "application/ld+json",
            RdfFormat::JsonLd {
                profile: Default::default(),
            },
        ),
        ("text/plain", RdfFormat::NTriples),
    ]
}

/// Answer one SPARQL protocol request against the KB.
pub fn handle(kb: &Kb, req: &Request<'_>, limits: &Limits) -> Response {
    let query = match query_of(req) {
        Ok(q) => q,
        Err(r) => return r,
    };
    let answer = match kb.query(&query, limits) {
        Ok(a) => a,
        Err(Error::Validation(r)) => return text(400, r.to_string()),
        Err(e) => return text(500, e.to_string()),
    };
    let out = match answer {
        Answer::Solutions { vars, rows, .. } => {
            let Some((media, format)) = pick(req.accept, &RESULT_FORMATS) else {
                return not_acceptable(&RESULT_FORMATS.map(|f| f.0));
            };
            let vars: Vec<Variable> = vars.iter().map(Variable::new_unchecked).collect();
            write_solutions(format, &vars, &rows).map(|b| (media, b))
        }
        Answer::Boolean(b) => {
            let Some((media, format)) = pick(req.accept, &RESULT_FORMATS) else {
                return not_acceptable(&RESULT_FORMATS.map(|f| f.0));
            };
            QueryResultsSerializer::from_format(format)
                .serialize_boolean_to_writer(Vec::new(), b)
                .map(|b| (media, b))
        }
        Answer::Graph { triples, .. } => {
            let formats = graph_formats();
            let Some((media, format)) = pick(req.accept, &formats) else {
                return not_acceptable(&formats.map(|f| f.0));
            };
            let mut w = RdfSerializer::from_format(format).for_writer(Vec::new());
            triples
                .iter()
                .try_for_each(|t| w.serialize_triple(t))
                .and_then(|()| w.finish())
                .map(|b| (media, b))
        }
    };
    match out {
        Ok((media, body)) => Response {
            status: 200,
            content_type: if media.starts_with("text/") {
                format!("{media}; charset=utf-8")
            } else {
                media.to_string()
            },
            body,
        },
        Err(e) => text(500, format!("serializing the results failed: {e}")),
    }
}

fn write_solutions(
    format: QueryResultsFormat,
    vars: &[Variable],
    rows: &[Vec<Option<oxrdf::Term>>],
) -> std::io::Result<Vec<u8>> {
    let mut w = QueryResultsSerializer::from_format(format)
        .serialize_solutions_to_writer(Vec::new(), vars.to_vec())?;
    for row in rows {
        w.serialize(
            vars.iter()
                .zip(row)
                .filter_map(|(v, t)| t.as_ref().map(|t| (v, t))),
        )?;
    }
    w.finish()
}

fn not_acceptable(offered: &[&str]) -> Response {
    text(
        406,
        format!(
            "none of the accepted media types is available; offered: {}",
            offered.join(", ")
        ),
    )
}
