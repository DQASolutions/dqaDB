//! Tier-3 benchmark: retrieval quality on BEIR datasets (nDCG@10, Recall@100), the way
//! BEIR scores it (trec_eval: linear gains, log2 discount; queries of the test qrels).
//!
//! Modes:
//!   load <dataset dir> <kb dir>     corpus.jsonl -> fresh dqaDB KB (title + text), embedded
//!   eval <dataset dir> <kb dir>     test queries in each search mode; prints the metrics
//!   (--text-only: skip embedding / the semantic and hybrid modes)
//!   subset <dataset dir> <out dir> <n>
//!                                   the first n corpus docs, and the test queries whose
//!                                   relevant docs are all among them (a quick run on a
//!                                   slow CPU; scores are not comparable to the full corpus)
//!
//! A dataset dir holds BEIR's corpus.jsonl, queries.jsonl and test.tsv (qrels).

use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use dqadb_core::search::{SearchQuery, TextMode};
use dqadb_core::{Kb, WriteOpts, import::ImportOpts};

const SCHEMA: &str = r#"id: urn:beir:schema
name: beir
prefixes:
  beir: "urn:beir:"
  linkml: https://w3id.org/linkml/
default_prefix: beir
default_range: string
imports:
  - linkml:types
classes:
  Doc:
    class_uri: beir:Doc
    slots: [label, text]
slots:
  label:
    slot_uri: rdfs:label
    annotations:
      dqadb_index: fts,card
  text:
    slot_uri: beir:text
    annotations:
      dqadb_index: fts,vector
"#;

fn jsonl(path: &Path) -> Vec<serde_json::Value> {
    BufReader::new(std::fs::File::open(path).unwrap())
        .lines()
        .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
        .collect()
}

/// Doc id -> IRI (ids are kept readable; anything outside [A-Za-z0-9-_.] is %-escaped).
fn doc_iri(id: &str) -> String {
    let mut s = String::from("urn:beir:doc:");
    for b in id.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.".contains(&b) {
            s.push(b as char);
        } else {
            s.push_str(&format!("%{b:02X}"));
        }
    }
    s
}

fn load(dataset: &Path, kb_dir: &Path, text_only: bool) {
    // a fresh KB, but the embedding cache (content-addressed) is kept: vectors are slow
    // to compute on a CPU and the same texts embed the same
    let cache = kb_dir.join("emb-cache");
    let kept = kb_dir.with_extension("emb-cache");
    let _ = std::fs::rename(&cache, &kept);
    let _ = std::fs::remove_dir_all(kb_dir);
    std::fs::create_dir_all(kb_dir).unwrap();
    Kb::init(kb_dir, "urn:dqadb:").unwrap();
    let _ = std::fs::rename(&kept, &cache);
    std::fs::write(kb_dir.join("ontology/schema.linkml.yaml"), SCHEMA).unwrap();
    let kb = Kb::open(kb_dir).unwrap();
    kb.build_ontology(true).unwrap();

    // N-Triples via oxrdf (correct escaping), then the regular import
    let nt = kb_dir.join("corpus.nt");
    let mut w = std::io::BufWriter::new(std::fs::File::create(&nt).unwrap());
    let label = oxrdf::NamedNode::new_unchecked("http://www.w3.org/2000/01/rdf-schema#label");
    let text = oxrdf::NamedNode::new_unchecked("urn:beir:text");
    let rdf_type =
        oxrdf::NamedNode::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    let doc = oxrdf::NamedNode::new_unchecked("urn:beir:Doc");
    let mut n = 0;
    for d in jsonl(&dataset.join("corpus.jsonl")) {
        let s = oxrdf::NamedNode::new(doc_iri(d["_id"].as_str().unwrap())).unwrap();
        writeln!(
            w,
            "{} .",
            oxrdf::Triple::new(s.clone(), rdf_type.clone(), doc.clone())
        )
        .unwrap();
        for (p, v) in [(&label, &d["title"]), (&text, &d["text"])] {
            let v = v.as_str().unwrap_or("").trim();
            if !v.is_empty() {
                writeln!(
                    w,
                    "{} .",
                    oxrdf::Triple::new(s.clone(), p.clone(), oxrdf::Literal::new_simple_literal(v))
                )
                .unwrap();
            }
        }
        n += 1;
    }
    drop(w);
    let t = Instant::now();
    let write = WriteOpts {
        actor: "system:bench".into(),
        ..Default::default()
    };
    let r = kb
        .import(&nt, &ImportOpts::default(), &write, &mut |_, _| {})
        .unwrap();
    assert!(r.issues.is_empty(), "{:?}", r.issues);
    eprintln!(
        "{n} docs, {} triples imported in {:.1}s; embedding ...",
        r.triples,
        t.elapsed().as_secs_f64()
    );
    if text_only {
        return;
    }
    let t = Instant::now();
    let e = kb.embed(false).unwrap();
    eprintln!(
        "embedded {} vectors ({} model calls) in {:.0}s",
        e.vectors_written,
        e.model_calls,
        t.elapsed().as_secs_f64()
    );
}

/// qrels: query id -> (doc id -> relevance)
fn qrels(path: &Path) -> BTreeMap<String, HashMap<String, f64>> {
    let mut out: BTreeMap<String, HashMap<String, f64>> = BTreeMap::new();
    for line in BufReader::new(std::fs::File::open(path).unwrap())
        .lines()
        .skip(1)
    {
        let line = line.unwrap();
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 3 {
            continue;
        }
        let rel: f64 = f[2].trim().parse().unwrap();
        out.entry(f[0].to_string())
            .or_default()
            .insert(f[1].to_string(), rel);
    }
    out
}

/// trec_eval ndcg_cut.k: gain = relevance, discount log2(rank + 1); ideal from the qrels.
fn ndcg_at(k: usize, ranked: &[String], rel: &HashMap<String, f64>) -> f64 {
    let dcg: f64 = ranked
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, d)| rel.get(d).copied().unwrap_or(0.0) / ((i + 2) as f64).log2())
        .sum();
    let mut ideal: Vec<f64> = rel.values().copied().filter(|r| *r > 0.0).collect();
    ideal.sort_by(|a, b| b.total_cmp(a));
    let idcg: f64 = ideal
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, r)| r / ((i + 2) as f64).log2())
        .sum();
    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

fn recall_at(k: usize, ranked: &[String], rel: &HashMap<String, f64>) -> f64 {
    let relevant: Vec<&String> = rel
        .iter()
        .filter(|(_, r)| **r > 0.0)
        .map(|(d, _)| d)
        .collect();
    if relevant.is_empty() {
        return 0.0;
    }
    let top: std::collections::HashSet<&String> = ranked.iter().take(k).collect();
    relevant.iter().filter(|d| top.contains(*d)).count() as f64 / relevant.len() as f64
}

/// The query's words joined with OR: BM25 over any of them, as in the BEIR baselines
/// (dqaDB's own default joins words with AND).
fn any_words(q: &str) -> String {
    q.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" OR ")
}

fn eval(dataset: &Path, kb_dir: &Path, text_only: bool) {
    let kb = Kb::open(kb_dir).unwrap();
    let qrels = qrels(&dataset.join("test.tsv"));
    let queries: HashMap<String, String> = jsonl(&dataset.join("queries.jsonl"))
        .into_iter()
        .map(|q| {
            (
                q["_id"].as_str().unwrap().to_string(),
                q["text"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    let to_id = |iri: &str| -> String {
        let iri = iri.trim_start_matches('<').trim_end_matches('>');
        // hits come compacted with the schema's prefixes (beir:doc:...)
        let raw = iri
            .strip_prefix("urn:beir:doc:")
            .or_else(|| iri.strip_prefix("beir:doc:"))
            .unwrap_or(iri);
        // undo doc_iri's escaping
        let mut out = Vec::new();
        let b = raw.as_bytes();
        let mut i = 0;
        while i < b.len() {
            if b[i] == b'%' && i + 2 < b.len() {
                out.push(u8::from_str_radix(&raw[i + 1..i + 3], 16).unwrap());
                i += 3;
            } else {
                out.push(b[i]);
                i += 1;
            }
        }
        String::from_utf8(out).unwrap()
    };
    let modes: [(&str, Box<dyn Fn(&str) -> SearchQuery>); 4] = [
        (
            "full-text BM25 (any word)",
            Box::new(|q: &str| SearchQuery {
                text: Some(any_words(q)),
                limit: 100,
                ..Default::default()
            }),
        ),
        (
            "full-text (raw query, dqaDB default)",
            Box::new(|q: &str| SearchQuery {
                text: Some(q.to_string()),
                limit: 100,
                ..Default::default()
            }),
        ),
        (
            "semantic (bge-m3)",
            Box::new(|q: &str| SearchQuery {
                semantic: Some(q.to_string()),
                limit: 100,
                ..Default::default()
            }),
        ),
        (
            "hybrid (RRF of both)",
            Box::new(|q: &str| SearchQuery {
                semantic: Some(q.to_string()),
                text: Some(any_words(q)),
                text_mode: TextMode::Fuse,
                limit: 100,
                ..Default::default()
            }),
        ),
    ];
    // optional second phrasing per query (ALQAC case facts): searched separately and
    // fused with the question by RRF (k = 60), as kb does with several phrasings
    let facts: HashMap<String, String> = jsonl(&dataset.join("queries.jsonl"))
        .into_iter()
        .filter_map(|q| {
            Some((
                q["_id"].as_str()?.to_string(),
                q["fact"].as_str()?.to_string(),
            ))
        })
        .collect();
    println!("{} test queries", qrels.len());
    for (name, make) in modes.iter().take(if text_only { 2 } else { 4 }) {
        let t = Instant::now();
        let (mut ndcg, mut rec, mut n) = (0.0, 0.0, 0usize);
        let (mut r10, mut r20, mut mrr) = (0.0, 0.0, 0.0);
        for (qid, rel) in &qrels {
            let Some(text) = queries.get(qid) else {
                continue;
            };
            let page = match kb.search(&make(text)) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("query {qid}: {e}");
                    n += 1;
                    continue;
                }
            };
            let ranked: Vec<String> = page
                .hits
                .iter()
                .map(|h| to_id(&h.iri))
                .filter(|d| d != qid)
                .collect();
            ndcg += ndcg_at(10, &ranked, rel);
            rec += recall_at(100, &ranked, rel);
            r10 += recall_at(10, &ranked, rel);
            r20 += recall_at(20, &ranked, rel);
            if let Some(i) = ranked.iter().position(|d| rel.get(d).is_some_and(|r| *r > 0.0)) {
                mrr += 1.0 / (i + 1) as f64;
            }
            n += 1;
        }
        println!(
            "  {name:38} nDCG@10 {:.3}  Recall@100 {:.3}  R@10 {:.3}  R@20 {:.3}  MRR {:.3}  ({:.0} ms/query)",
            ndcg / n as f64,
            rec / n as f64,
            r10 / n as f64,
            r20 / n as f64,
            mrr / n as f64,
            t.elapsed().as_secs_f64() * 1e3 / n as f64
        );
    }
    if facts.is_empty() {
        return;
    }
    let search = |text: &str| -> Vec<String> {
        kb.search(&SearchQuery {
            text: Some(text.to_string()),
            limit: 100,
            ..Default::default()
        })
        .map(|p| p.hits.iter().map(|h| to_id(&h.iri)).collect())
        .unwrap_or_default()
    };
    let (mut ndcg, mut rec, mut r10, mut r20, mut mrr, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0usize);
    for (qid, rel) in &qrels {
        let (Some(q), Some(f)) = (queries.get(qid), facts.get(qid)) else {
            continue;
        };
        let ranked = rrf(&[search(q), search(f)]);
        ndcg += ndcg_at(10, &ranked, rel);
        rec += recall_at(100, &ranked, rel);
        r10 += recall_at(10, &ranked, rel);
        r20 += recall_at(20, &ranked, rel);
        if let Some(i) = ranked.iter().position(|d| rel.get(d).is_some_and(|r| *r > 0.0)) {
            mrr += 1.0 / (i + 1) as f64;
        }
        n += 1;
    }
    let n = n.max(1) as f64;
    println!(
        "  {:38} nDCG@10 {:.3}  Recall@100 {:.3}  R@10 {:.3}  R@20 {:.3}  MRR {:.3}",
        "full-text, question + fact (RRF)",
        ndcg / n,
        rec / n,
        r10 / n,
        r20 / n,
        mrr / n
    );
}

fn rrf(lists: &[Vec<String>]) -> Vec<String> {
    let mut score: HashMap<&String, f64> = HashMap::new();
    for l in lists {
        for (i, d) in l.iter().enumerate() {
            *score.entry(d).or_default() += 1.0 / (60.0 + i as f64 + 1.0);
        }
    }
    let mut v: Vec<(&String, f64)> = score.into_iter().collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(b.0)));
    v.into_iter().map(|(d, _)| d.clone()).collect()
}

fn subset(dataset: &Path, out: &Path, n: usize) {
    std::fs::create_dir_all(out).unwrap();
    let corpus = std::fs::read_to_string(dataset.join("corpus.jsonl")).unwrap();
    let kept: Vec<&str> = corpus.lines().take(n).collect();
    let ids: std::collections::HashSet<String> = kept
        .iter()
        .map(|l| {
            serde_json::from_str::<serde_json::Value>(l).unwrap()["_id"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    std::fs::write(out.join("corpus.jsonl"), kept.join("\n") + "\n").unwrap();
    std::fs::copy(dataset.join("queries.jsonl"), out.join("queries.jsonl")).unwrap();
    let all = qrels(&dataset.join("test.tsv"));
    let mut tsv = String::from("query-id\tcorpus-id\tscore\n");
    let mut q = 0;
    for (qid, rel) in &all {
        if rel.keys().all(|d| ids.contains(d)) {
            q += 1;
            for (d, r) in rel {
                tsv.push_str(&format!("{qid}\t{d}\t{r}\n"));
            }
        }
    }
    std::fs::write(out.join("test.tsv"), tsv).unwrap();
    eprintln!("{} docs, {q} of {} test queries", ids.len(), all.len());
}

fn main() {
    // --text-only: no embeddings (load) / only the full-text modes (eval)
    let text_only = std::env::args().any(|a| a == "--text-only");
    let a: Vec<String> = std::env::args().filter(|a| a != "--text-only").collect();
    let p = |i: usize| PathBuf::from(&a[i]);
    match a.get(1).map(String::as_str) {
        Some("load") => load(&p(2), &p(3), text_only),
        Some("eval") => eval(&p(2), &p(3), text_only),
        Some("subset") => subset(&p(2), &p(3), a[4].parse().unwrap()),
        _ => {
            eprintln!(
                "usage: load|eval <dataset dir> <kb dir> | subset <dataset dir> <out dir> <n>"
            );
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndcg_matches_trec_eval_by_hand() {
        // relevant: a (2), b (1); ranking: x, a, b
        let rel: HashMap<String, f64> = [("a".into(), 2.0), ("b".into(), 1.0)].into();
        let ranked: Vec<String> = ["x", "a", "b"].iter().map(|s| s.to_string()).collect();
        let dcg = 2.0 / 3f64.log2() + 1.0 / 4f64.log2();
        let idcg = 2.0 / 2f64.log2() + 1.0 / 3f64.log2();
        assert!((ndcg_at(10, &ranked, &rel) - dcg / idcg).abs() < 1e-12);
        assert_eq!(recall_at(2, &ranked, &rel), 0.5);
        assert_eq!(doc_iri("MED-10"), "urn:beir:doc:MED-10");
        assert_eq!(doc_iri("a b"), "urn:beir:doc:a%20b");
    }
}
