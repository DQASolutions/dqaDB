use dqadb_core::ontology::InputOp;
use dqadb_core::sparql::{self, Answer, Limits};
use dqadb_core::{Error, Kb, WriteOpts};

fn ops(json: serde_json::Value) -> Vec<InputOp> {
    serde_json::from_value(json).unwrap()
}

fn opts() -> WriteOpts {
    WriteOpts {
        actor: "agent:test".into(),
        ..Default::default()
    }
}

fn new_kb() -> (tempfile::TempDir, Kb) {
    let dir = tempfile::tempdir().unwrap();
    let kb = Kb::init(dir.path(), "urn:dqadb:").unwrap();
    kb.build_ontology(true).unwrap();
    (dir, kb)
}

fn acme() -> Vec<InputOp> {
    ops(serde_json::json!([
        {"op": "add", "s": "_:acme", "p": "rdf:type", "o": "ex:Company"},
        {"op": "add", "s": "_:acme", "p": "rdfs:label", "o": {"@value": "ACME Logistics", "@language": "vi"}},
        {"op": "add", "s": "_:acme", "p": "ex:country", "o": "VN"},
        {"op": "add", "s": "_:acme", "p": "ex:has_contract", "o": "_:c1"},
        {"op": "add", "s": "_:c1", "p": "rdf:type", "o": "ex:Contract"},
        {"op": "add", "s": "_:c1", "p": "rdfs:label", "o": "HĐ 123"},
        {"op": "add", "s": "_:c1", "p": "ex:value", "o": "050000.00"},
        {"op": "add", "s": "_:c1", "p": "ex:status", "o": "active"}
    ]))
}

#[test]
fn assert_get_retract_and_rebuild_is_deterministic() {
    let (_dir, kb) = new_kb();
    let res = kb.assert(&acme(), &opts()).unwrap();
    assert_eq!(res.added, 8);
    let acme_iri = res.id_map["_:acme"].replace("urn:dqadb:", "dqadb:");
    let c1 = res.id_map["_:c1"].replace("urn:dqadb:", "dqadb:");

    let card = kb.get(&acme_iri).unwrap();
    assert_eq!(card.label.as_deref(), Some("ACME Logistics"));
    let types: Vec<(&str, bool)> = card
        .types
        .iter()
        .map(|t| (t.iri.as_str(), t.inferred))
        .collect();
    assert_eq!(
        types,
        vec![("ex:Company", false), ("ex:Organization", true)]
    );
    let contract = card
        .props
        .iter()
        .find(|p| p.p == "ex:has_contract")
        .unwrap();
    assert_eq!(contract.value, c1);
    assert_eq!(
        kb.get(&c1)
            .unwrap()
            .props
            .iter()
            .find(|p| p.p == "ex:value")
            .unwrap()
            .value,
        "50000"
    );

    // single-valued slot: adding a second country without retracting fails
    let err = kb
        .assert(
            &ops(serde_json::json!([{"op": "add", "s": acme_iri, "p": "ex:country", "o": "SG"}])),
            &opts(),
        )
        .unwrap_err();
    assert_eq!(err.exit_code(), 2, "{err}");

    // retract + add in one event is the update pattern
    kb.assert(
        &ops(serde_json::json!([
            {"op": "retract", "s": acme_iri, "p": "ex:country", "o": "VN"},
            {"op": "add", "s": acme_iri, "p": "ex:country", "o": "SG"}
        ])),
        &opts(),
    )
    .unwrap();
    let country = |kb: &Kb| {
        kb.get(&acme_iri)
            .unwrap()
            .props
            .into_iter()
            .find(|p| p.p == "ex:country")
            .unwrap()
            .value
    };
    assert_eq!(country(&kb), "SG");

    let before = kb.fingerprint().unwrap();
    let report = kb.rebuild().unwrap();
    assert_eq!(report.events, 2);
    assert_eq!(
        report.fingerprint, before,
        "rebuild must reproduce the same index"
    );
    assert_eq!(country(&kb), "SG");

    // deleting the index entirely and reading again catches up from the files
    drop(kb);
    std::fs::remove_dir_all(_dir.path().join(".dqadb")).unwrap();
    let kb = Kb::open(_dir.path()).unwrap();
    assert_eq!(country(&kb), "SG");
    assert_eq!(kb.fingerprint().unwrap(), before);
    assert!(kb.verify().unwrap().problems.is_empty());
}

#[test]
fn validation_reports_fixable_issues() {
    let (_dir, kb) = new_kb();
    let bad = ops(serde_json::json!([
        {"op": "add", "s": "_:x", "p": "rdf:type", "o": "ex:Contract"},
        {"op": "add", "s": "_:x", "p": "ex:value", "o": "50k"},
        {"op": "add", "s": "_:x", "p": "ex:status", "o": "paused"},
        {"op": "add", "s": "_:x", "p": "ex:colour", "o": "red"}
    ]));
    let Err(Error::Validation(report)) = kb.assert(&bad, &opts()) else {
        panic!("expected validation error")
    };
    let text = report.to_string();
    assert!(text.contains("expects xsd:decimal"), "{text}");
    assert!(text.contains("allowed: active, draft, ended"), "{text}");
    assert!(text.contains("unknown predicate ex:colour"), "{text}");
    // nothing was written
    assert!(kb.events_ls(10, None, None).unwrap().is_empty());
}

#[test]
fn required_slots_and_stale_ontology() {
    let (dir, kb) = new_kb();
    let missing_label =
        ops(serde_json::json!([{"op": "add", "s": "_:x", "p": "rdf:type", "o": "ex:Company"}]));
    let Err(Error::Validation(report)) = kb.assert(&missing_label, &opts()) else {
        panic!()
    };
    assert!(
        report.to_string().contains("requires rdfs:label"),
        "{report}"
    );

    let schema = dir.path().join("ontology/schema.linkml.yaml");
    let text = std::fs::read_to_string(&schema).unwrap();
    std::fs::write(&schema, format!("{text}\n# edited\n")).unwrap();
    let err = kb.assert(&acme(), &opts()).unwrap_err();
    assert_eq!(err.exit_code(), 4);
}

#[test]
fn retract_event_and_expected_version() {
    let (_dir, kb) = new_kb();
    let first = kb.assert(&acme(), &opts()).unwrap();
    let stale = WriteOpts {
        expected_version: Some("01AAAAAAAAAAAAAAAAAAAAAAAA".into()),
        ..opts()
    };
    assert_eq!(kb.assert(&acme(), &stale).unwrap_err().exit_code(), 6);

    let undo = kb
        .retract_event(&first.event_id, "wrong extraction", &opts())
        .unwrap();
    assert_eq!(undo.retracted, 8);
    let acme_iri = first.id_map["_:acme"].replace("urn:dqadb:", "dqadb:");
    assert_eq!(kb.get(&acme_iri).unwrap_err().exit_code(), 3);
    assert_eq!(kb.status().unwrap().counts.triples, 0);
}

fn select(kb: &Kb, q: &str) -> Vec<Vec<String>> {
    let prefixes = kb.ontology().unwrap().prefixes;
    match kb.query(q, &Limits::default()).unwrap() {
        Answer::Solutions { rows, .. } => rows
            .iter()
            .map(|r| {
                r.iter()
                    .map(|t| {
                        t.as_ref()
                            .map(|t| sparql::display(&prefixes, t))
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .collect(),
        other => panic!("expected solutions, got {other:?}"),
    }
}

#[test]
fn rdfs_subclass_inference_follows_types_and_ontology() {
    let (dir, kb) = new_kb();
    let res = kb.assert(&acme(), &opts()).unwrap();
    let acme_iri = res.id_map["_:acme"].replace("urn:dqadb:", "dqadb:");

    // ex:Company is_a ex:Organization -> inferred type, queryable via SPARQL
    let orgs = select(&kb, "SELECT ?o WHERE { ?o a ex:Organization }");
    assert_eq!(orgs, vec![vec![acme_iri.clone()]]);
    assert_eq!(kb.status().unwrap().counts.inferred, 1);

    // inferred facts never land in event files
    let ev = kb.event(&res.event_id).unwrap();
    assert!(
        ev.ops.iter().all(
            |o| !matches!(&o.o, dqadb_core::rdf::Term::Iri(i) if i.ends_with("#Organization"))
        )
    );

    // a contract becomes an organization once the hierarchy says so (no new events)
    let schema = dir.path().join("ontology/schema.linkml.yaml");
    let text = std::fs::read_to_string(&schema).unwrap();
    std::fs::write(
        &schema,
        text.replace("  Contract:\n", "  Contract:\n    is_a: Organization\n"),
    )
    .unwrap();
    kb.build_ontology(true).unwrap();
    assert_eq!(
        select(
            &kb,
            "SELECT (COUNT(*) AS ?n) WHERE { ?o a ex:Organization }"
        ),
        vec![vec!["2".to_string()]]
    );
    let incremental = kb.fingerprint().unwrap();
    assert_eq!(
        kb.rebuild().unwrap().fingerprint,
        incremental,
        "re-inference must equal a full rebuild"
    );

    // retracting the asserted type removes what was inferred from it
    kb.assert(&ops(serde_json::json!([{"op": "retract", "s": acme_iri, "p": "rdf:type", "o": "ex:Company"}])), &opts()).unwrap();
    let types = kb.get(&acme_iri).unwrap().types;
    assert!(types.is_empty(), "{types:?}");
    assert_eq!(kb.rebuild().unwrap().fingerprint, kb.fingerprint().unwrap());
}

#[test]
fn sparql_select_filter_ask_graph_and_errors() {
    let (_dir, kb) = new_kb();
    let res = kb.assert(&acme(), &opts()).unwrap();
    kb.assert(
        &ops(serde_json::json!([
            {"op": "add", "s": "_:c2", "p": "rdf:type", "o": "ex:Contract"},
            {"op": "add", "s": "_:c2", "p": "rdfs:label", "o": "HĐ 7"},
            {"op": "add", "s": "_:c2", "p": "ex:value", "o": 900}
        ])),
        &opts(),
    )
    .unwrap();

    let rows = select(
        &kb,
        "SELECT ?label ?v WHERE { ?c a ex:Contract ; rdfs:label ?label ; ex:value ?v FILTER(?v > 1000) }",
    );
    assert_eq!(
        rows,
        vec![vec!["\"HĐ 123\"".to_string(), "50000".to_string()]]
    );

    // property path + language-tagged label
    let rows = select(
        &kb,
        "SELECT ?name WHERE { ?org ex:has_contract/ex:status \"active\" ; rdfs:label ?name }",
    );
    assert_eq!(rows, vec![vec!["\"ACME Logistics\"@vi".to_string()]]);

    // provenance: which event asserted the country
    let rows = select(&kb, "SELECT ?g WHERE { GRAPH ?g { ?s ex:country \"VN\" } }");
    assert_eq!(rows, vec![vec![format!("dqadb:event:{}", res.event_id)]]);
    let g = format!("<urn:dqadb:event:{}>", res.event_id);
    let n = select(
        &kb,
        &format!("SELECT (COUNT(*) AS ?n) WHERE {{ GRAPH {g} {{ ?s ?p ?o }} }}"),
    );
    assert_eq!(n, vec![vec!["8".to_string()]]);

    match kb
        .query("ASK { ?x a ex:Company }", &Limits::default())
        .unwrap()
    {
        Answer::Boolean(b) => assert!(b),
        other => panic!("{other:?}"),
    }
    let limited = kb
        .query(
            "SELECT ?s ?p ?o WHERE { ?s ?p ?o }",
            &Limits {
                rows: 3,
                ..Limits::default()
            },
        )
        .unwrap();
    assert!(
        matches!(limited, Answer::Solutions { truncated: true, ref rows, .. } if rows.len() == 3)
    );

    let err = kb
        .query("SELECT ?x WHERE { ?x a }", &Limits::default())
        .unwrap_err();
    assert_eq!(err.exit_code(), 2);
}

#[test]
fn scans_cross_page_boundaries() {
    // 1,500 organisations x 2 triples: well past the 1,024-row scan page
    let (_dir, kb) = new_kb();
    let mut list = Vec::new();
    for n in 0..1500 {
        list.push(serde_json::json!({"op": "add", "s": format!("_:o{n}"), "p": "rdf:type", "o": "ex:Organization"}));
        list.push(serde_json::json!({"op": "add", "s": format!("_:o{n}"), "p": "rdfs:label", "o": format!("Org {n}")}));
    }
    kb.assert(&ops(serde_json::Value::Array(list)), &opts())
        .unwrap();
    let triples = kb.status().unwrap().counts.triples;
    assert_eq!(triples, 3000);
    let all = select(&kb, "SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }");
    assert_eq!(all, vec![vec!["3000".to_string()]]);
    let orgs = select(
        &kb,
        "SELECT (COUNT(DISTINCT ?s) AS ?n) WHERE { ?s a ex:Organization ; rdfs:label ?l }",
    );
    assert_eq!(orgs, vec![vec!["1500".to_string()]]);
    let first = kb.fingerprint().unwrap();
    assert_eq!(kb.rebuild().unwrap().fingerprint, first);
}

#[test]
fn search_traverse_and_path() {
    use dqadb_core::search::SearchQuery;
    let (_dir, kb) = new_kb();
    let res = kb.assert(&acme(), &opts()).unwrap();
    let acme_iri = res.id_map["_:acme"].replace("urn:dqadb:", "dqadb:");
    let c1 = res.id_map["_:c1"].replace("urn:dqadb:", "dqadb:");
    let beta = kb
        .assert(
            &ops(serde_json::json!([
                {"op": "add", "s": "_:b", "p": "rdf:type", "o": "ex:Company"},
                {"op": "add", "s": "_:b", "p": "rdfs:label", "o": "Beta Shipping"},
                {"op": "add", "s": "_:b", "p": "ex:country", "o": "SG"},
                {"op": "add", "s": "_:b", "p": "ex:has_contract", "o": c1},
                {"op": "add", "s": "_:b", "p": "ex:has_contract", "o": "_:c2"},
                {"op": "add", "s": "_:c2", "p": "rdf:type", "o": "ex:Contract"},
                {"op": "add", "s": "_:c2", "p": "rdfs:label", "o": "HĐ 7"},
                {"op": "add", "s": "_:c2", "p": "ex:value", "o": 900},
                {"op": "add", "s": "_:c2", "p": "ex:status", "o": "draft"}
            ])),
            &opts(),
        )
        .unwrap();
    let beta_iri = beta.id_map["_:b"].replace("urn:dqadb:", "dqadb:");

    let q = |types: &[&str], wheres: &[&str]| {
        let page = kb
            .search(&SearchQuery {
                types: types.iter().map(|s| s.to_string()).collect(),
                wheres: wheres.iter().map(|s| s.to_string()).collect(),
                limit: 10,
                cursor: None,
                text: None,
                semantic: None,
            })
            .unwrap();
        let mut labels: Vec<String> = page.hits.into_iter().filter_map(|h| h.label).collect();
        labels.sort();
        labels
    };
    // inferred type counts; equality, range, in, has, enum
    assert_eq!(
        q(&["ex:Organization"], &[]),
        vec!["ACME Logistics", "Beta Shipping"]
    );
    assert_eq!(
        q(&["ex:Company"], &["ex:country = \"VN\""]),
        vec!["ACME Logistics"]
    );
    assert_eq!(q(&[], &["ex:value >= 1000"]), vec!["HĐ 123"]);
    assert_eq!(q(&[], &["ex:value < 1000.5"]), vec!["HĐ 7"]);
    assert_eq!(
        q(&[], &["ex:country in (\"SG\", \"TH\")"]),
        vec!["Beta Shipping"]
    );
    assert_eq!(q(&["ex:Contract"], &["ex:status = draft"]), vec!["HĐ 7"]);
    assert_eq!(
        q(&[], &["has ex:has_contract"]),
        vec!["ACME Logistics", "Beta Shipping"]
    );
    assert_eq!(
        q(&[], &[&format!("ex:has_contract = {c1}")]),
        vec!["ACME Logistics", "Beta Shipping"]
    );
    assert!(q(&[], &["ex:country = \"JP\""]).is_empty());
    // bad input is a validation error (exit 2)
    let err = kb
        .search(&SearchQuery {
            wheres: vec!["ex:value >= lots".into()],
            limit: 10,
            ..Default::default()
        })
        .unwrap_err();
    assert_eq!(err.exit_code(), 2, "{err}");

    // pagination is stable and complete
    let page1 = kb
        .search(&SearchQuery {
            types: vec!["ex:Organization".into()],
            limit: 1,
            ..Default::default()
        })
        .unwrap();
    assert_eq!((page1.total, page1.next_cursor.as_deref()), (2, Some("1")));
    let page2 = kb
        .search(&SearchQuery {
            types: vec!["ex:Organization".into()],
            limit: 1,
            cursor: page1.next_cursor,
            ..Default::default()
        })
        .unwrap();
    assert!(page2.next_cursor.is_none());
    assert_ne!(page1.hits[0].iri, page2.hits[0].iri);

    // traverse: contracts of ACME, and who holds contract c1 (inverse path)
    let names: Vec<String> = kb
        .traverse(&acme_iri, "ex:has_contract/rdfs:label", 10)
        .unwrap()
        .into_iter()
        .map(|h| h.iri)
        .collect();
    assert_eq!(names, vec!["\"HĐ 123\""]);
    let holders = kb.traverse(&c1, "^ex:has_contract", 10).unwrap();
    assert_eq!(holders.len(), 2);

    // path: ACME -> c1 <- Beta
    let steps = kb.path(&acme_iri, &beta_iri, 4).unwrap().unwrap();
    assert_eq!(steps.len(), 2);
    assert!(steps[0].forward && !steps[1].forward);
    assert_eq!(steps[0].p, "ex:has_contract");
    assert!(kb.path(&acme_iri, &beta_iri, 1).unwrap().is_none());

    // filters survive a rebuild unchanged
    let fp = kb.fingerprint().unwrap();
    assert_eq!(kb.rebuild().unwrap().fingerprint, fp);
    assert_eq!(q(&[], &["ex:value >= 1000"]), vec!["HĐ 123"]);
}

#[test]
fn full_text_is_accent_insensitive_lazy_and_self_repairing() {
    use dqadb_core::search::SearchQuery;
    let (dir, kb) = new_kb();
    let res = kb.assert(&acme(), &opts()).unwrap();
    let acme_iri = res.id_map["_:acme"].replace("urn:dqadb:", "dqadb:");
    kb.assert(
        &ops(serde_json::json!([
            {"op": "add", "s": acme_iri, "p": "ex:description", "o": "Công ty vận tải biển tại Hải Phòng"},
            {"op": "add", "s": "_:b", "p": "rdf:type", "o": "ex:Company"},
            {"op": "add", "s": "_:b", "p": "rdfs:label", "o": "Beta Đường Sắt"},
            {"op": "add", "s": "_:b", "p": "ex:description", "o": "Vận chuyển hàng hóa bằng ĐƯỜNG SẮT Bắc Nam"},
            {"op": "add", "s": "_:b", "p": "ex:country", "o": "VN"}
        ])),
        &opts(),
    )
    .unwrap();
    let text = |kb: &Kb, t: &str, types: &[&str]| -> Vec<String> {
        kb.search(&SearchQuery {
            text: Some(t.into()),
            types: types.iter().map(|s| s.to_string()).collect(),
            limit: 10,
            ..Default::default()
        })
        .unwrap()
        .hits
        .into_iter()
        .filter_map(|h| h.label)
        .collect()
    };

    // accents and đ/d are ignored, case too; terms are ANDed
    assert_eq!(text(&kb, "hai phong", &[]), vec!["ACME Logistics"]);
    assert_eq!(text(&kb, "Hải Phòng", &[]), vec!["ACME Logistics"]);
    assert_eq!(text(&kb, "duong sat", &[]), vec!["Beta Đường Sắt"]);
    assert!(text(&kb, "hai phong duong", &[]).is_empty());
    // combined with a structured filter (filter first, then BM25)
    assert_eq!(text(&kb, "hd", &["ex:Contract"]), vec!["HĐ 123"]);
    assert!(text(&kb, "hd", &["ex:Company"]).is_empty());

    // lazy: a write does not touch the text index; the next text search catches up
    kb.assert(&ops(serde_json::json!([
        {"op": "add", "s": acme_iri, "p": "rdfs:comment", "o": "Đối tác chiến lược cảng Cát Lái"}
    ])), &opts()).unwrap();
    assert_eq!(text(&kb, "cat lai", &[]), vec!["ACME Logistics"]);
    // retracting the text removes the match
    kb.assert(&ops(serde_json::json!([
        {"op": "retract", "s": acme_iri, "p": "rdfs:comment", "o": "Đối tác chiến lược cảng Cát Lái"}
    ])), &opts()).unwrap();
    assert!(text(&kb, "cat lai", &[]).is_empty());

    // losing the text index entirely is repaired by the next search
    std::fs::remove_dir_all(dir.path().join(".dqadb/fts")).unwrap();
    assert_eq!(text(&kb, "hai phong", &[]), vec!["ACME Logistics"]);

    // indexing config follows the ontology: make ex:country full-text searchable
    let schema = dir.path().join("ontology/schema.linkml.yaml");
    let yaml = std::fs::read_to_string(&schema).unwrap();
    std::fs::write(
        &schema,
        yaml.replace(
            "      dqadb_index: filter\n  has_contract:",
            "      dqadb_index: filter,fts\n  has_contract:",
        ),
    )
    .unwrap();
    kb.build_ontology(true).unwrap();
    let mut vn = text(&kb, "vn", &[]);
    vn.sort();
    assert_eq!(vn, vec!["ACME Logistics", "Beta Đường Sắt"]);

    // rebuild re-creates both indexes
    let report = kb.rebuild().unwrap();
    assert_eq!(report.text_documents, 3);
    assert_eq!(text(&kb, "duong sat", &[]), vec!["Beta Đường Sắt"]);
}

fn hash_kb() -> (tempfile::TempDir, Kb) {
    let dir = tempfile::tempdir().unwrap();
    Kb::init(dir.path(), "urn:dqadb:").unwrap();
    let cfg = dir.path().join("dqadb.yaml");
    let yaml = std::fs::read_to_string(&cfg).unwrap();
    std::fs::write(
        &cfg,
        yaml.replace("model: bge-m3\n  dim: 1024", "model: hash-test\n  dim: 64"),
    )
    .unwrap();
    let kb = Kb::open(dir.path()).unwrap();
    assert_eq!(kb.config.embedding.model, "hash-test");
    kb.build_ontology(true).unwrap();
    (dir, kb)
}

#[test]
fn semantic_search_embeds_lazily_and_reuses_the_cache() {
    use dqadb_core::search::SearchQuery;
    let (_dir, kb) = hash_kb();
    let org = |label: &str, desc: &str| {
        serde_json::json!([
            {"op": "add", "s": "_:x", "p": "rdf:type", "o": "ex:Company"},
            {"op": "add", "s": "_:x", "p": "rdfs:label", "o": label},
            {"op": "add", "s": "_:x", "p": "ex:description", "o": desc}
        ])
    };
    let a = kb
        .assert(
            &ops(org(
                "ACME",
                "vận tải biển hàng hóa container cảng Hải Phòng",
            )),
            &opts(),
        )
        .unwrap();
    kb.assert(&ops(org("Beta", "rang xay cà phê và trà sữa")), &opts())
        .unwrap();
    kb.assert(
        &ops(org("Gamma", "vận tải đường bộ xe tải liên tỉnh")),
        &opts(),
    )
    .unwrap();
    let sem = |kb: &Kb, q: &str, types: &[&str], text: Option<&str>| {
        kb.search(&SearchQuery {
            semantic: Some(q.into()),
            text: text.map(Into::into),
            types: types.iter().map(|s| s.to_string()).collect(),
            limit: 3,
            ..Default::default()
        })
        .unwrap()
    };

    // nothing embedded yet: results are empty and pending says why
    let before = sem(&kb, "vận tải biển", &[], None);
    assert!(before.hits.is_empty());
    assert_eq!(before.pending, Some(3));

    let report = kb.embed_sync().unwrap();
    assert_eq!(
        (
            report.vectors_written,
            report.model_calls,
            report.full_rebuild
        ),
        (3, 3, true)
    );
    let page = sem(&kb, "vận tải biển container", &[], None);
    assert_eq!(page.pending, Some(0));
    let labels: Vec<_> = page.hits.iter().filter_map(|h| h.label.clone()).collect();
    assert_eq!(labels[0], "ACME");
    assert_eq!(labels.last().unwrap(), "Beta");

    // filters restrict the candidates; text + semantic are fused (RRF)
    assert!(sem(&kb, "vận tải", &["ex:Contract"], None).hits.is_empty());
    let hybrid = sem(&kb, "xe tải", &["ex:Company"], Some("van tai"));
    assert_eq!(hybrid.hits[0].label.as_deref(), Some("Gamma"));

    // incremental: only the changed entity is re-embedded
    let acme = a.id_map["_:x"].replace("urn:dqadb:", "dqadb:");
    kb.assert(&ops(serde_json::json!([
        {"op": "retract", "s": acme, "p": "ex:description", "o": "vận tải biển hàng hóa container cảng Hải Phòng"},
        {"op": "add", "s": acme, "p": "ex:description", "o": "phần mềm kế toán doanh nghiệp"}
    ])), &opts()).unwrap();
    assert_eq!(kb.vec_pending().unwrap(), 1);
    let r = kb.embed_sync().unwrap();
    assert_eq!(
        (
            r.entities_checked,
            r.vectors_written,
            r.model_calls,
            r.full_rebuild
        ),
        (1, 1, 1, false)
    );
    let top = sem(&kb, "vận tải biển container", &[], None);
    assert_ne!(top.hits[0].label.as_deref(), Some("ACME"));

    // a rebuild drops the index, but vectors come back from the cache without the model
    kb.rebuild().unwrap();
    let r = kb.embed_sync().unwrap();
    assert_eq!(
        (r.vectors_written, r.model_calls, r.full_rebuild),
        (3, 0, true)
    );
    assert_eq!(
        sem(&kb, "cà phê", &[], None).hits[0].label.as_deref(),
        Some("Beta")
    );
}

#[test]
fn hnsw_base_plus_delta_matches_exact_and_stays_correct() {
    use dqadb_core::search::SearchQuery;
    let (dir, kb) = hash_kb();
    let topics = [
        "vận tải biển",
        "cà phê rang xay",
        "phần mềm kế toán",
        "xây dựng cầu đường",
        "bảo hiểm y tế",
        "dệt may xuất khẩu",
    ];
    let mut list = Vec::new();
    for n in 0..300 {
        let t = topics[n % topics.len()];
        list.push(serde_json::json!({"op": "add", "s": format!("_:o{n}"), "p": "rdf:type", "o": "ex:Organization"}));
        list.push(serde_json::json!({"op": "add", "s": format!("_:o{n}"), "p": "rdfs:label", "o": format!("Org {n} {t}")}));
    }
    let res = kb
        .assert(&ops(serde_json::Value::Array(list)), &opts())
        .unwrap();
    let report = kb.embed(true).unwrap();
    let c = report.compaction.expect("forced compaction");
    assert!(c.built && c.vectors == 300, "{c:?}");
    assert!(dir.path().join(".dqadb/vec/base.usearch").exists());

    let sem = |q: &str, k: usize| -> Vec<String> {
        kb.search(&SearchQuery {
            semantic: Some(q.into()),
            limit: k,
            ..Default::default()
        })
        .unwrap()
        .hits
        .into_iter()
        .filter_map(|h| h.label)
        .collect()
    };
    let with_base = sem("bảo hiểm y tế", 10);
    assert!(
        with_base.iter().all(|l| l.contains("bảo hiểm y tế")),
        "{with_base:?}"
    );

    // same answer as an exact scan (no base)
    std::fs::rename(
        dir.path().join(".dqadb/vec/base.json"),
        dir.path().join("base.json.bak"),
    )
    .unwrap();
    let exact = sem("bảo hiểm y tế", 10);
    std::fs::rename(
        dir.path().join("base.json.bak"),
        dir.path().join(".dqadb/vec/base.json"),
    )
    .unwrap();
    // labels differ only by a number, so scores tie: compare the topic, not the tie order
    assert!(
        exact.iter().all(|l| l.contains("bảo hiểm y tế")),
        "{exact:?}"
    );
    assert_eq!(with_base.len(), exact.len());

    // change one entity after the base was built: the base copy is ignored, the delta wins
    let o0 = res.id_map["_:o0"].replace("urn:dqadb:", "dqadb:");
    kb.assert(
        &ops(serde_json::json!([
            {"op": "retract", "s": o0, "p": "rdfs:label", "o": "Org 0 vận tải biển"},
            {"op": "add", "s": o0, "p": "rdfs:label", "o": "Org 0 trí tuệ nhân tạo"}
        ])),
        &opts(),
    )
    .unwrap();
    // remove the text of another entity entirely
    let o6 = res.id_map["_:o6"].replace("urn:dqadb:", "dqadb:");
    kb.assert(
        &ops(serde_json::json!([
            {"op": "retract", "s": o6, "p": "rdfs:label", "o": "Org 6 vận tải biển"}
        ])),
        &opts(),
    )
    .unwrap();
    let r = kb.embed(false).unwrap();
    assert!(
        r.compaction.is_none(),
        "small delta must not trigger a rebuild"
    );
    assert_eq!(sem("trí tuệ nhân tạo", 1), vec!["Org 0 trí tuệ nhân tạo"]);
    let shipping = sem("vận tải biển", 60);
    assert!(
        !shipping
            .iter()
            .any(|l| l == "Org 0 vận tải biển" || l == "Org 6 vận tải biển"),
        "stale vectors returned"
    );

    // index rebuild = new vector-table generation: the old base is ignored until recompacted
    kb.rebuild().unwrap();
    kb.embed(false).unwrap();
    assert_eq!(sem("trí tuệ nhân tạo", 1), vec!["Org 0 trí tuệ nhân tạo"]);
    let c = kb.embed(true).unwrap().compaction.unwrap();
    assert_eq!(c.vectors, 299);
    assert_eq!(sem("trí tuệ nhân tạo", 1), vec!["Org 0 trí tuệ nhân tạo"]);
}
