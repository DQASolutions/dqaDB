use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const RDF: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
pub const RDFS: &str = "http://www.w3.org/2000/01/rdf-schema#";
pub const XSD: &str = "http://www.w3.org/2001/XMLSchema#";
pub const OWL: &str = "http://www.w3.org/2002/07/owl#";

pub const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
pub const RDF_LANG_STRING: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString";
pub const RDFS_LABEL: &str = "http://www.w3.org/2000/01/rdf-schema#label";
pub const RDFS_COMMENT: &str = "http://www.w3.org/2000/01/rdf-schema#comment";
pub const XSD_STRING: &str = "http://www.w3.org/2001/XMLSchema#string";

pub fn xsd(local: &str) -> String {
    format!("{XSD}{local}")
}

/// An RDF term as stored in event files and in the index.
/// JSON form: a plain string is an IRI; an object is a JSON-LD style literal.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Term {
    Iri(String),
    Literal(Literal),
}

/// Normalised literal: exactly one of `datatype` / `language` is set.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Literal {
    #[serde(rename = "@value")]
    pub value: String,
    #[serde(rename = "@type", default, skip_serializing_if = "Option::is_none")]
    pub datatype: Option<String>,
    #[serde(rename = "@language", default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

impl Term {
    pub fn typed(value: impl Into<String>, datatype: impl Into<String>) -> Self {
        Term::Literal(Literal {
            value: value.into(),
            datatype: Some(datatype.into()),
            language: None,
        })
    }

    pub fn lang(value: impl Into<String>, language: impl Into<String>) -> Self {
        Term::Literal(Literal {
            value: value.into(),
            datatype: None,
            language: Some(language.into()),
        })
    }

    pub fn as_iri(&self) -> Option<&str> {
        match self {
            Term::Iri(i) => Some(i),
            Term::Literal(_) => None,
        }
    }

    /// Canonical N-Triples form; used as the dictionary key.
    pub fn ntriples(&self) -> String {
        match self {
            Term::Iri(i) => format!("<{i}>"),
            Term::Literal(l) => {
                let v = escape(&l.value);
                match (&l.language, &l.datatype) {
                    (Some(lang), _) => format!("\"{v}\"@{lang}"),
                    (None, Some(dt)) => format!("\"{v}\"^^<{dt}>"),
                    (None, None) => format!("\"{v}\"^^<{XSD_STRING}>"),
                }
            }
        }
    }
}

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ => out.push(c),
        }
    }
    out
}

/// Prefix map used to expand CURIEs (`ex:Company`) and to compact IRIs for display.
#[derive(Debug, Clone, Default)]
pub struct Prefixes {
    map: BTreeMap<String, String>,
}

impl Prefixes {
    pub fn with_defaults(base_iri: &str) -> Self {
        let mut p = Self::default();
        p.insert("rdf", RDF);
        p.insert("rdfs", RDFS);
        p.insert("xsd", XSD);
        p.insert("owl", OWL);
        p.insert("dqadb", base_iri);
        p
    }

    pub fn insert(&mut self, prefix: &str, namespace: &str) {
        self.map.insert(prefix.to_string(), namespace.to_string());
    }

    pub fn get(&self, prefix: &str) -> Option<&str> {
        self.map.get(prefix).map(String::as_str)
    }

    pub fn names(&self) -> Vec<&str> {
        self.map.keys().map(String::as_str).collect()
    }

    /// (prefix, namespace) pairs, sorted by prefix.
    pub fn pairs(&self) -> impl Iterator<Item = (&str, &str)> {
        self.map.iter().map(|(p, ns)| (p.as_str(), ns.as_str()))
    }

    /// Accepts `prefix:local`, `<absolute-iri>`, `http(s)://...` or `urn:...`.
    pub fn expand(&self, input: &str) -> Result<String, String> {
        let s = input.trim();
        if let Some(inner) = s.strip_prefix('<').and_then(|x| x.strip_suffix('>')) {
            return check_iri(inner);
        }
        if let Some((prefix, local)) = s.split_once(':') {
            if let Some(ns) = self.map.get(prefix) {
                return check_iri(&format!("{ns}{local}"));
            }
            if local.starts_with("//") || prefix == "urn" {
                return check_iri(s);
            }
            return Err(format!(
                "unknown prefix '{prefix}:' in '{s}'; known prefixes: {}",
                self.names().join(", ")
            ));
        }
        Err(format!(
            "'{s}' is not an IRI or CURIE (expected prefix:local, e.g. ex:Company)"
        ))
    }

    /// Shortest readable form: the longest matching namespace wins.
    pub fn compact(&self, iri: &str) -> String {
        let best = self
            .map
            .iter()
            .filter(|(_, ns)| iri.len() > ns.len() && iri.starts_with(ns.as_str()))
            .max_by_key(|(_, ns)| ns.len());
        match best {
            Some((p, ns)) => format!("{p}:{}", &iri[ns.len()..]),
            None => format!("<{iri}>"),
        }
    }

    pub fn display(&self, term: &Term) -> String {
        match term {
            Term::Iri(i) => self.compact(i),
            Term::Literal(l) => match &l.language {
                Some(lang) => format!("{}@{lang}", l.value),
                None => l.value.clone(),
            },
        }
    }
}

fn check_iri(s: &str) -> Result<String, String> {
    oxrdf::NamedNode::new(s)
        .map(|n| n.into_string())
        .map_err(|e| format!("invalid IRI '{s}': {e}"))
}
