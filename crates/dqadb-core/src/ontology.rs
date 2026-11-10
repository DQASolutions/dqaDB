//! Ontology loaded from LinkML YAML (spec §6), plus the interim validator.
//!
//! The validator enforces what the LinkML schema states directly (known predicates and
//! classes, ranges, enums, required, single-valued). Full SHACL validation against the
//! generated `shapes.ttl` replaces it later in M1.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::ValidationReport;
use crate::event::{Op, OpKind};
use crate::rdf::{self, Prefixes, Term};
use crate::{Error, Result};

pub const SCHEMA_SUFFIX: &str = ".linkml.yaml";

// ---------- LinkML YAML (only the subset dqaDB reads) ----------

#[derive(Debug, Default, Deserialize)]
struct LmSchema {
    #[serde(default)]
    prefixes: BTreeMap<String, LmPrefix>,
    default_prefix: Option<String>,
    default_range: Option<String>,
    #[serde(default)]
    classes: BTreeMap<String, Option<LmClass>>,
    #[serde(default)]
    slots: BTreeMap<String, Option<LmSlot>>,
    #[serde(default)]
    enums: BTreeMap<String, Option<LmEnum>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum LmPrefix {
    Simple(String),
    Full { prefix_reference: String },
}

#[derive(Debug, Default, Deserialize)]
struct LmClass {
    is_a: Option<String>,
    class_uri: Option<String>,
    description: Option<String>,
    #[serde(default)]
    slots: Vec<String>,
    #[serde(default)]
    attributes: BTreeMap<String, Option<LmSlot>>,
}

#[derive(Debug, Default, Clone, Deserialize)]
struct LmSlot {
    slot_uri: Option<String>,
    range: Option<String>,
    required: Option<bool>,
    multivalued: Option<bool>,
    description: Option<String>,
    #[serde(default)]
    annotations: BTreeMap<String, serde_yaml::Value>,
}

#[derive(Debug, Default, Deserialize)]
struct LmEnum {
    #[serde(default)]
    permissible_values: BTreeMap<String, serde_yaml::Value>,
}

// ---------- resolved model ----------

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Range {
    Datatype(String),
    Class(String),
    Enum(String),
    AnyIri,
}

#[derive(Debug, Clone, Serialize)]
pub struct SlotDef {
    pub name: String,
    pub iri: String,
    pub range: Range,
    pub required: bool,
    pub multivalued: bool,
    pub index: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClassDef {
    pub name: String,
    pub iri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_a: Option<String>,
    pub slots: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Ontology {
    pub prefixes: Prefixes,
    /// keyed by class IRI
    pub classes: BTreeMap<String, ClassDef>,
    /// keyed by predicate IRI
    pub slots: BTreeMap<String, SlotDef>,
    pub enums: BTreeMap<String, BTreeSet<String>>,
    /// `sha256:<hex>` over all schema files; recorded in every event.
    pub hash: String,
}

/// RDFS rules the index materialises: `rdf:type` along `rdfs:subClassOf` (LinkML `is_a`).
/// Domain/range are not used for entailment: in LinkML they are constraints, and inferring
/// types from them would hide data errors that validation should report.
#[derive(Debug, Clone, Default)]
pub struct Inference {
    /// class IRI -> proper ancestors, nearest first
    ancestors: BTreeMap<String, Vec<String>>,
    /// changes whenever the rules change; stored in the index to detect stale inferences
    pub hash: String,
}

impl Inference {
    pub fn new(ancestors: BTreeMap<String, Vec<String>>) -> Self {
        let mut h = Sha256::new();
        for (c, a) in &ancestors {
            h.update(format!("{c} < {}\n", a.join(" ")));
        }
        Self {
            ancestors,
            hash: format!("sha256:{}", hex::encode(h.finalize())),
        }
    }

    pub fn superclasses(&self, class_iri: &str) -> &[String] {
        self.ancestors
            .get(class_iri)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }
}

/// Input op as written by an agent: CURIEs, `_:tmp` placeholders and plain JSON values allowed.
#[derive(Debug, Clone, Deserialize)]
pub struct InputOp {
    pub op: OpKind,
    pub s: String,
    pub p: String,
    pub o: serde_json::Value,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Normalized {
    pub ops: Vec<Op>,
    /// `_:tmp` placeholder -> minted IRI
    pub id_map: BTreeMap<String, String>,
    pub warnings: Vec<String>,
}

fn builtin_range(name: &str) -> Option<Range> {
    let dt = match name {
        "string" | "str" => "string",
        "integer" | "int" => "integer",
        "float" => "float",
        "double" => "double",
        "decimal" => "decimal",
        "boolean" | "bool" => "boolean",
        "date" => "date",
        "datetime" => "dateTime",
        "uri" | "uriorcurie" => return Some(Range::AnyIri),
        _ => return None,
    };
    Some(Range::Datatype(rdf::xsd(dt)))
}

fn annotation_text(v: &serde_yaml::Value) -> Option<String> {
    match v {
        serde_yaml::Value::String(s) => Some(s.clone()),
        serde_yaml::Value::Mapping(m) => {
            m.get("value").and_then(|x| x.as_str()).map(str::to_string)
        }
        _ => None,
    }
}

/// Schema files, sorted, with their sha256; plus the combined hash.
pub fn schema_hash(ontology_dir: &Path) -> Result<(BTreeMap<String, String>, String)> {
    let mut files = BTreeMap::new();
    if ontology_dir.exists() {
        for entry in std::fs::read_dir(ontology_dir)? {
            let path = entry?.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if name.ends_with(SCHEMA_SUFFIX) {
                files.insert(
                    name.to_string(),
                    hex::encode(Sha256::digest(std::fs::read(&path)?)),
                );
            }
        }
    }
    let mut all = Sha256::new();
    for (name, h) in &files {
        all.update(format!("{name} {h}\n"));
    }
    Ok((files, format!("sha256:{}", hex::encode(all.finalize()))))
}

impl Ontology {
    pub fn load(ontology_dir: &Path, main_file: &Path, base_iri: &str) -> Result<Self> {
        let (files, hash) = schema_hash(ontology_dir)?;
        if files.is_empty() {
            return Err(Error::other(format!(
                "no *{SCHEMA_SUFFIX} files in {}",
                ontology_dir.display()
            )));
        }
        let mut schemas = Vec::new();
        for name in files.keys() {
            let path = ontology_dir.join(name);
            let schema: LmSchema = serde_yaml::from_str(&std::fs::read_to_string(&path)?)
                .map_err(|e| Error::other(format!("{}: {e}", path.display())))?;
            let is_main = main_file.file_name().and_then(|n| n.to_str()) == Some(name.as_str());
            schemas.push((is_main, schema));
        }
        // main schema first so its default_prefix / default_range win
        schemas.sort_by_key(|(is_main, _)| !*is_main);

        let mut prefixes = Prefixes::with_defaults(base_iri);
        for (_, s) in &schemas {
            for (p, v) in &s.prefixes {
                let ns = match v {
                    LmPrefix::Simple(ns) => ns,
                    LmPrefix::Full { prefix_reference } => prefix_reference,
                };
                if p != "dqadb" {
                    prefixes.insert(p, ns);
                }
            }
        }
        let default_prefix = schemas.iter().find_map(|(_, s)| s.default_prefix.clone());
        let default_range = schemas
            .iter()
            .find_map(|(_, s)| s.default_range.clone())
            .unwrap_or_else(|| "string".into());

        let iri_of = |explicit: &Option<String>, name: &str| -> Result<String> {
            let curie = match (explicit, &default_prefix) {
                (Some(u), _) => u.clone(),
                (None, Some(dp)) => format!("{dp}:{name}"),
                (None, None) => {
                    return Err(Error::other(format!(
                        "'{name}' has no uri and the schema has no default_prefix"
                    )));
                }
            };
            prefixes.expand(&curie).map_err(Error::Other)
        };

        let mut class_iris: HashMap<String, String> = HashMap::new();
        let mut enums = BTreeMap::new();
        for (_, s) in &schemas {
            for (name, c) in &s.classes {
                let explicit = c.as_ref().and_then(|c| c.class_uri.clone());
                class_iris.insert(name.clone(), iri_of(&explicit, name)?);
            }
            for (name, e) in &s.enums {
                let values = e
                    .as_ref()
                    .map(|e| e.permissible_values.keys().cloned().collect())
                    .unwrap_or_default();
                enums.insert(name.clone(), values);
            }
        }

        let resolve_range = |range: &str| -> Result<Range> {
            if let Some(r) = builtin_range(range) {
                return Ok(r);
            }
            if let Some(iri) = class_iris.get(range) {
                return Ok(Range::Class(iri.clone()));
            }
            if enums.contains_key(range) {
                return Ok(Range::Enum(range.to_string()));
            }
            Err(Error::other(format!(
                "unknown range '{range}' (not a type, class or enum)"
            )))
        };

        let make_slot = |name: &str, def: &LmSlot| -> Result<SlotDef> {
            let range = resolve_range(def.range.as_deref().unwrap_or(&default_range))?;
            let index = def
                .annotations
                .get("dqadb_index")
                .and_then(annotation_text)
                .map(|t| {
                    t.split(',')
                        .map(|x| x.trim().to_string())
                        .filter(|x| !x.is_empty())
                        .collect()
                })
                .unwrap_or_default();
            Ok(SlotDef {
                name: name.to_string(),
                iri: iri_of(&def.slot_uri, name)?,
                range,
                required: def.required.unwrap_or(false),
                multivalued: def.multivalued.unwrap_or(false),
                index,
                description: def.description.clone(),
            })
        };

        let mut slot_by_name: HashMap<String, SlotDef> = HashMap::new();
        for (_, s) in &schemas {
            for (name, def) in &s.slots {
                slot_by_name.insert(
                    name.clone(),
                    make_slot(name, &def.clone().unwrap_or_default())?,
                );
            }
        }

        let mut classes = BTreeMap::new();
        for (_, s) in &schemas {
            for (name, c) in &s.classes {
                let c = c.as_ref();
                let mut slot_iris = Vec::new();
                for sname in c.map(|c| c.slots.as_slice()).unwrap_or_default() {
                    let slot = slot_by_name.get(sname).ok_or_else(|| {
                        Error::other(format!("class {name} uses undefined slot '{sname}'"))
                    })?;
                    slot_iris.push(slot.iri.clone());
                }
                if let Some(c) = c {
                    for (aname, adef) in &c.attributes {
                        let slot = make_slot(aname, &adef.clone().unwrap_or_default())?;
                        slot_iris.push(slot.iri.clone());
                        slot_by_name.entry(aname.clone()).or_insert(slot);
                    }
                }
                let iri = class_iris[name].clone();
                let is_a = match c.and_then(|c| c.is_a.as_ref()) {
                    Some(parent) => Some(class_iris.get(parent).cloned().ok_or_else(|| {
                        Error::other(format!("class {name}: unknown is_a '{parent}'"))
                    })?),
                    None => None,
                };
                classes.insert(
                    iri.clone(),
                    ClassDef {
                        name: name.clone(),
                        iri,
                        is_a,
                        slots: slot_iris,
                        description: c.and_then(|c| c.description.clone()),
                    },
                );
            }
        }

        let mut slots: BTreeMap<String, SlotDef> = slot_by_name
            .into_values()
            .map(|s| (s.iri.clone(), s))
            .collect();
        for (iri, name, index) in [
            (rdf::RDFS_LABEL, "label", "fts,card"),
            (rdf::RDFS_COMMENT, "comment", "fts"),
        ] {
            slots.entry(iri.to_string()).or_insert_with(|| SlotDef {
                name: name.into(),
                iri: iri.into(),
                range: Range::Datatype(rdf::XSD_STRING.into()),
                required: false,
                multivalued: true,
                index: index.split(',').map(str::to_string).collect(),
                description: None,
            });
        }

        Ok(Self {
            prefixes,
            classes,
            slots,
            enums,
            hash,
        })
    }

    /// RDFS entailment rules derived from the schema (spec §7).
    pub fn inference(&self) -> Inference {
        let ancestors: BTreeMap<String, Vec<String>> = self
            .classes
            .keys()
            .map(|c| {
                (
                    c.clone(),
                    self.lineage(c)
                        .iter()
                        .skip(1)
                        .map(|a| a.iri.clone())
                        .collect(),
                )
            })
            .filter(|(_, a): &(String, Vec<String>)| !a.is_empty())
            .collect();
        Inference::new(ancestors)
    }

    /// The class and all its ancestors (is_a chain).
    pub fn lineage(&self, class_iri: &str) -> Vec<&ClassDef> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        let mut cur = self.classes.get(class_iri);
        while let Some(c) = cur {
            if !seen.insert(c.iri.clone()) {
                break;
            }
            out.push(c);
            cur = c.is_a.as_ref().and_then(|p| self.classes.get(p));
        }
        out
    }

    pub fn compact(&self, iri: &str) -> String {
        self.prefixes.compact(iri)
    }

    /// Validate and normalise input ops. `mint` creates a fresh entity IRI;
    /// `existing(s, p)` returns the values currently stored for (s, p).
    pub fn normalize(
        &self,
        input: &[InputOp],
        mint: &mut dyn FnMut() -> String,
        existing: &dyn Existing,
    ) -> std::result::Result<Normalized, ValidationReport> {
        let mut report = ValidationReport::default();
        let mut out = Normalized::default();
        if input.is_empty() {
            report.push(
                "ops",
                "no ops given",
                Some("pass at least one {op, s, p, o}".into()),
            );
        }

        for (i, op) in input.iter().enumerate() {
            let at = format!("ops[{i}]");
            let s = match self.node(&op.s, &mut out.id_map, mint) {
                Ok(s) => s,
                Err(e) => {
                    report.push(
                        format!("{at}.s"),
                        e,
                        Some("use a CURIE like dqadb:<ULID>, or _:name for a new entity".into()),
                    );
                    continue;
                }
            };
            let p = match self.prefixes.expand(&op.p) {
                Ok(p) => p,
                Err(e) => {
                    report.push(format!("{at}.p"), e, None);
                    continue;
                }
            };
            let o = if p == rdf::RDF_TYPE {
                self.type_object(&op.o, &at, &mut report)
            } else if let Some(slot) = self.slots.get(&p) {
                self.object(slot, &op.o, &at, &mut out.id_map, mint, &mut report)
            } else {
                report.push(
                    format!("{at}.p"),
                    format!("unknown predicate {}", self.compact(&p)),
                    Some(format!(
                        "use a slot defined in the ontology (see: dqadb ontology show); known: {}",
                        self.known_predicates()
                    )),
                );
                None
            };
            if let Some(o) = o {
                out.ops.push(Op { op: op.op, s, p, o });
            }
        }

        if report.is_empty() {
            self.check_cardinality(&out.ops, existing, &mut report, &mut out.warnings);
        }
        if report.is_empty() {
            Ok(out)
        } else {
            Err(report)
        }
    }

    fn known_predicates(&self) -> String {
        let mut names: Vec<String> = self.slots.keys().map(|k| self.compact(k)).collect();
        names.push("rdf:type".into());
        names.join(", ")
    }

    fn node(
        &self,
        raw: &str,
        id_map: &mut BTreeMap<String, String>,
        mint: &mut dyn FnMut() -> String,
    ) -> std::result::Result<String, String> {
        if raw.starts_with("_:") {
            return Ok(id_map.entry(raw.to_string()).or_insert_with(mint).clone());
        }
        self.prefixes.expand(raw)
    }

    fn type_object(
        &self,
        o: &serde_json::Value,
        at: &str,
        report: &mut ValidationReport,
    ) -> Option<Term> {
        let Some(raw) = o.as_str() else {
            report.push(
                format!("{at}.o"),
                "rdf:type needs a class IRI",
                Some("e.g. \"o\": \"ex:Company\"".into()),
            );
            return None;
        };
        match self.prefixes.expand(raw) {
            Ok(iri) if self.classes.contains_key(&iri) => Some(Term::Iri(iri)),
            Ok(iri) => {
                let known: Vec<String> = self.classes.keys().map(|k| self.compact(k)).collect();
                report.push(
                    format!("{at}.o"),
                    format!("unknown class {}", self.compact(&iri)),
                    Some(format!("known classes: {}", known.join(", "))),
                );
                None
            }
            Err(e) => {
                report.push(format!("{at}.o"), e, None);
                None
            }
        }
    }

    fn object(
        &self,
        slot: &SlotDef,
        o: &serde_json::Value,
        at: &str,
        id_map: &mut BTreeMap<String, String>,
        mint: &mut dyn FnMut() -> String,
        report: &mut ValidationReport,
    ) -> Option<Term> {
        let at = format!("{at}.o");
        let pname = self.compact(&slot.iri);
        match &slot.range {
            Range::Class(_) | Range::AnyIri => {
                let Some(raw) = o.as_str() else {
                    report.push(
                        at,
                        format!("{pname} expects an entity IRI, got {o}"),
                        Some("pass a string such as \"dqadb:<ULID>\" or \"_:name\"".into()),
                    );
                    return None;
                };
                match self.node(raw, id_map, mint) {
                    Ok(iri) => Some(Term::Iri(iri)),
                    Err(e) => {
                        report.push(at, e, None);
                        None
                    }
                }
            }
            Range::Enum(name) => {
                let Some(v) = o.as_str() else {
                    report.push(at, format!("{pname} expects one of enum {name}"), None);
                    return None;
                };
                if self.enums[name].contains(v) {
                    Some(Term::typed(v, rdf::XSD_STRING))
                } else {
                    let allowed: Vec<&str> = self.enums[name].iter().map(String::as_str).collect();
                    report.push(
                        at,
                        format!("'{v}' is not allowed for {pname}"),
                        Some(format!("allowed: {}", allowed.join(", "))),
                    );
                    None
                }
            }
            Range::Datatype(dt) => self.literal(dt, &pname, o, &at, report),
        }
    }

    fn literal(
        &self,
        dt: &str,
        pname: &str,
        o: &serde_json::Value,
        at: &str,
        report: &mut ValidationReport,
    ) -> Option<Term> {
        let short_dt = self.compact(dt);
        let (raw, given_type, lang) = match o {
            serde_json::Value::String(s) => (s.clone(), None, None),
            serde_json::Value::Number(n) => (n.to_string(), None, None),
            serde_json::Value::Bool(b) => (b.to_string(), None, None),
            serde_json::Value::Object(m) => {
                let value = match m.get("@value") {
                    Some(serde_json::Value::String(s)) => s.clone(),
                    Some(serde_json::Value::Number(n)) => n.to_string(),
                    Some(serde_json::Value::Bool(b)) => b.to_string(),
                    _ => {
                        report.push(
                            at,
                            "literal object needs \"@value\"",
                            Some(format!(
                                "e.g. {{\"@value\": \"...\", \"@type\": \"{short_dt}\"}}"
                            )),
                        );
                        return None;
                    }
                };
                let ty = m
                    .get("@type")
                    .and_then(|t| t.as_str())
                    .map(|t| self.prefixes.expand(t));
                let lang = m
                    .get("@language")
                    .and_then(|l| l.as_str())
                    .map(str::to_string);
                (value, ty, lang)
            }
            _ => {
                report.push(
                    at,
                    format!("{pname} expects a {short_dt} literal, got {o}"),
                    None,
                );
                return None;
            }
        };
        if let Some(ty) = given_type {
            match ty {
                Ok(t) if t == dt => {}
                Ok(t) => {
                    report.push(
                        at,
                        format!("{pname} expects {short_dt}, got @type {}", self.compact(&t)),
                        Some(format!("use \"@type\": \"{short_dt}\" or omit @type")),
                    );
                    return None;
                }
                Err(e) => {
                    report.push(at, e, None);
                    return None;
                }
            }
        }
        if let Some(lang) = lang {
            if dt != rdf::XSD_STRING {
                report.push(
                    at,
                    format!("@language is only allowed on string slots; {pname} is {short_dt}"),
                    None,
                );
                return None;
            }
            if oxrdf::Literal::new_language_tagged_literal(raw.clone(), lang.clone()).is_err() {
                report.push(
                    at,
                    format!("invalid language tag '{lang}'"),
                    Some("use a BCP 47 tag such as vi or en".into()),
                );
                return None;
            }
            return Some(Term::lang(raw, lang.to_ascii_lowercase()));
        }
        match normalize_lexical(dt, &raw) {
            Ok(v) => Some(Term::typed(v, dt)),
            Err(e) => {
                report.push(
                    at,
                    format!("{pname} expects {short_dt}: {e}"),
                    Some(format!(
                        "e.g. {{\"@value\": \"{}\", \"@type\": \"{short_dt}\"}}",
                        example_for(dt)
                    )),
                );
                None
            }
        }
    }

    fn check_cardinality(
        &self,
        ops: &[Op],
        existing: &dyn Existing,
        report: &mut ValidationReport,
        warnings: &mut Vec<String>,
    ) {
        // final values per (s,p) after this event
        let mut adds: BTreeMap<(&str, &str), Vec<&Term>> = BTreeMap::new();
        let mut retracts: HashSet<(&str, &str, &Term)> = HashSet::new();
        for op in ops {
            match op.op {
                OpKind::Add => adds.entry((&op.s, &op.p)).or_default().push(&op.o),
                OpKind::Retract => {
                    retracts.insert((&op.s, &op.p, &op.o));
                }
            }
        }
        for ((s, p), values) in &adds {
            let Some(slot) = self.slots.get(*p) else {
                continue;
            };
            if slot.multivalued {
                continue;
            }
            let mut finals: BTreeSet<&Term> = values.iter().copied().collect();
            let current = existing.values(s, p);
            for v in &current {
                if !retracts.contains(&(*s, *p, v)) {
                    finals.insert(v);
                }
            }
            if finals.len() > 1 {
                let shown: Vec<String> = finals.iter().map(|t| self.prefixes.display(t)).collect();
                report.push(
                    format!("{} {}", self.compact(s), self.compact(p)),
                    format!(
                        "{} is single-valued but would hold {} values: {}",
                        self.compact(p),
                        finals.len(),
                        shown.join(" | ")
                    ),
                    Some("retract the old value in the same event, then add the new one".into()),
                );
            }
        }

        // required slots for subjects typed in this event
        let mut typed: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for op in ops {
            if op.op == OpKind::Add
                && op.p == rdf::RDF_TYPE
                && let Term::Iri(c) = &op.o
            {
                typed.entry(&op.s).or_default().push(c);
            }
        }
        for (s, classes) in typed {
            for class in classes {
                for c in self.lineage(class) {
                    for slot_iri in &c.slots {
                        let slot = &self.slots[slot_iri];
                        if !slot.required {
                            continue;
                        }
                        let in_event = adds.contains_key(&(s, slot_iri.as_str()));
                        if !in_event && existing.values(s, slot_iri).is_empty() {
                            report.push(
                                self.compact(s),
                                format!("{} requires {}", c.name, self.compact(slot_iri)),
                                Some(format!(
                                    "add {{\"op\":\"add\",\"s\":\"{}\",\"p\":\"{}\",\"o\":...}}",
                                    self.compact(s),
                                    self.compact(slot_iri)
                                )),
                            );
                        }
                    }
                }
            }
        }

        for op in ops {
            if op.op == OpKind::Retract && !existing.has(&op.s, &op.p, &op.o) {
                warnings.push(format!(
                    "retract of a triple that is not stored: {} {} {}",
                    self.compact(&op.s),
                    self.compact(&op.p),
                    self.prefixes.display(&op.o)
                ));
            }
        }
    }
}

/// What is stored now, as seen by validation.
pub trait Existing {
    /// Current values of (s, p).
    fn values(&self, s: &str, p: &str) -> Vec<Term>;
    /// Whether (s, p, o) is stored.
    fn has(&self, s: &str, p: &str, o: &Term) -> bool {
        self.values(s, p).contains(o)
    }
}

impl<F: Fn(&str, &str) -> Vec<Term>> Existing for F {
    fn values(&self, s: &str, p: &str) -> Vec<Term> {
        self(s, p)
    }
}

fn example_for(dt: &str) -> &'static str {
    match dt.strip_prefix(rdf::XSD).unwrap_or("") {
        "integer" => "42",
        "decimal" => "50000.5",
        "double" | "float" => "1.5e3",
        "boolean" => "true",
        "date" => "2026-11-09",
        "dateTime" => "2026-11-09T10:00:00Z",
        _ => "text",
    }
}

/// Canonical lexical form per datatype, so equal values compare equal in the index.
pub fn normalize_lexical(dt: &str, raw: &str) -> std::result::Result<String, String> {
    let v = raw.trim();
    match dt.strip_prefix(rdf::XSD).unwrap_or("") {
        "string" => Ok(raw.to_string()),
        "integer" => v
            .parse::<i128>()
            .map(|n| n.to_string())
            .map_err(|_| format!("'{raw}' is not an integer")),
        "decimal" => normalize_decimal(v).ok_or_else(|| format!("'{raw}' is not a decimal")),
        "double" | "float" => v
            .parse::<f64>()
            .map(|_| v.to_string())
            .map_err(|_| format!("'{raw}' is not a number")),
        "boolean" => match v {
            "true" | "1" => Ok("true".into()),
            "false" | "0" => Ok("false".into()),
            _ => Err(format!("'{raw}' is not a boolean")),
        },
        "date" => chrono::NaiveDate::parse_from_str(v, "%Y-%m-%d")
            .map(|d| d.to_string())
            .map_err(|_| format!("'{raw}' is not a date (YYYY-MM-DD)")),
        "dateTime" => chrono::DateTime::parse_from_rfc3339(v)
            .map(|d| d.to_rfc3339())
            .map_err(|_| format!("'{raw}' is not an RFC 3339 date-time")),
        _ => Ok(raw.to_string()),
    }
}

fn normalize_decimal(v: &str) -> Option<String> {
    let (sign, digits) = match v.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", v.strip_prefix('+').unwrap_or(v)),
    };
    let (int, frac) = digits.split_once('.').unwrap_or((digits, ""));
    if int.is_empty() && frac.is_empty() {
        return None;
    }
    if !int.chars().all(|c| c.is_ascii_digit()) || !frac.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let int = int.trim_start_matches('0');
    let frac = frac.trim_end_matches('0');
    let int = if int.is_empty() { "0" } else { int };
    let body = if frac.is_empty() {
        int.to_string()
    } else {
        format!("{int}.{frac}")
    };
    Some(if body == "0" {
        body
    } else {
        format!("{sign}{body}")
    })
}

// ---------- build ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildLock {
    pub dqadb: u32,
    pub hash: String,
    pub sources: BTreeMap<String, String>,
    pub generator: String,
}

pub fn read_lock(generated_dir: &Path) -> Result<Option<BuildLock>> {
    let path = generated_dir.join("build.lock");
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some(serde_json::from_slice(&std::fs::read(path)?)?))
}

/// Error unless `_generated/build.lock` matches the current YAML.
pub fn check_fresh(ontology_dir: &Path, generated_dir: &Path) -> Result<String> {
    let (_, hash) = schema_hash(ontology_dir)?;
    match read_lock(generated_dir)? {
        None => Err(Error::OntologyMissing),
        Some(lock) if lock.hash != hash => Err(Error::OntologyStale),
        Some(_) => Ok(hash),
    }
}

/// Generate `ontology.ttl` + `shapes.ttl` with the LinkML CLI, then write `build.lock`.
/// `lock_only` skips generation (development without Python/LinkML installed).
pub fn build(
    ontology_dir: &Path,
    main_file: &Path,
    base_iri: &str,
    lock_only: bool,
) -> Result<BuildLock> {
    // fail early on schema errors dqaDB itself cannot read
    Ontology::load(ontology_dir, main_file, base_iri)?;
    let generated = ontology_dir.join("_generated");
    std::fs::create_dir_all(&generated)?;
    let generator = if lock_only {
        "none (--lock-only)".to_string()
    } else {
        run_generator("gen-owl", main_file, &generated.join("ontology.ttl"))?;
        run_generator("gen-shacl", main_file, &generated.join("shapes.ttl"))?;
        "linkml (gen-owl, gen-shacl)".to_string()
    };
    let (sources, hash) = schema_hash(ontology_dir)?;
    let lock = BuildLock {
        dqadb: 1,
        hash,
        sources,
        generator,
    };
    std::fs::write(
        generated.join("build.lock"),
        serde_json::to_vec_pretty(&lock)?,
    )?;
    Ok(lock)
}

fn run_generator(bin: &str, main_file: &Path, out: &PathBuf) -> Result<()> {
    let output = Command::new(bin).arg(main_file).output().map_err(|e| {
        Error::other(format!(
            "cannot run {bin}: {e}. Install LinkML (pip install linkml), or use --lock-only to skip TTL generation"
        ))
    })?;
    if !output.status.success() {
        return Err(Error::other(format!(
            "{bin} failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    std::fs::write(out, output.stdout)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimals_normalize() {
        assert_eq!(normalize_decimal("050000.500").as_deref(), Some("50000.5"));
        assert_eq!(normalize_decimal("-0.0").as_deref(), Some("0"));
        assert_eq!(normalize_decimal("+12").as_deref(), Some("12"));
        assert_eq!(normalize_decimal("1e3"), None);
    }
}
