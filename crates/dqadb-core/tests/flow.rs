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
    let prefixes = kb.ontology().unwrap().prefixes.clone();
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
                text_mode: Default::default(),
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

    // accents and đ/d are ignored, case too; any word matches, more words rank higher
    assert_eq!(text(&kb, "hai phong", &[]), vec!["ACME Logistics"]);
    assert_eq!(text(&kb, "Hải Phòng", &[]), vec!["ACME Logistics"]);
    assert_eq!(text(&kb, "duong sat", &[]), vec!["Beta Đường Sắt"]);
    assert_eq!(
        text(&kb, "hai phong duong", &[]),
        vec!["ACME Logistics", "Beta Đường Sắt"]
    );
    // required words: AND / + (operators are not folded into words)
    assert!(text(&kb, "hai AND phong AND duong", &[]).is_empty());
    assert!(text(&kb, "+hai +duong", &[]).is_empty());
    assert_eq!(text(&kb, "hai AND phong", &[]), vec!["ACME Logistics"]);
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
fn hnsw_segments_plus_delta_match_exact_and_stay_correct() {
    use dqadb_core::search::SearchQuery;
    // hash-test labels differ by one number, so vectors are near-duplicates and HNSW
    // (incremental or rebuilt) misses a few of them: the exact checks are about stale and
    // deleted vectors; recall checks allow a small loss (real-data recall: bench/vector)
    unsafe { std::env::set_var("DQADB_HNSW_EF", "1024") };
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
    assert!(c.built && c.vectors == 300 && c.segments == 1, "{c:?}");
    assert!(dir.path().join(".dqadb/vec/segments.json").exists());

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

    // same answer as an exact scan (no segments)
    std::fs::rename(
        dir.path().join(".dqadb/vec/segments.json"),
        dir.path().join("segments.json.bak"),
    )
    .unwrap();
    let exact = sem("bảo hiểm y tế", 10);
    std::fs::rename(
        dir.path().join("segments.json.bak"),
        dir.path().join(".dqadb/vec/segments.json"),
    )
    .unwrap();
    // labels differ only by a number, so scores tie: compare the topic, not the tie order
    assert!(
        exact.iter().all(|l| l.contains("bảo hiểm y tế")),
        "{exact:?}"
    );
    assert_eq!(with_base.len(), exact.len());

    // change one entity after the segment was built: its copy is ignored, the delta wins
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
        "small delta must not trigger a flush"
    );
    assert_eq!(sem("trí tuệ nhân tạo", 1), vec!["Org 0 trí tuệ nhân tạo"]);
    let shipping = sem("vận tải biển", 60);
    assert!(
        !shipping
            .iter()
            .any(|l| l == "Org 0 vận tải biển" || l == "Org 6 vận tải biển"),
        "stale vectors returned"
    );

    // flush: the old copies become deletions of segment 1, the update a new segment
    let c = dqadb_core::ann::build(
        &dir.path().join(".dqadb/index.sqlite"),
        &dir.path().join(".dqadb/vec"),
        64,
        false,
    )
    .unwrap();
    assert!(
        c.built && c.incremental && c.vectors == 299 && c.segments == 2 && c.written == 1,
        "{c:?}"
    );
    let m = dqadb_core::ann::load_manifest(&dir.path().join(".dqadb/vec")).unwrap();
    assert_eq!(
        m.segments
            .iter()
            .map(|s| (s.count, s.deleted))
            .collect::<Vec<_>>(),
        vec![(300, 2), (1, 0)]
    );
    assert_eq!(sem("trí tuệ nhân tạo", 1), vec!["Org 0 trí tuệ nhân tạo"]);
    let shipping = sem("vận tải biển", 60);
    // approximate: allow a few misses on near-duplicates
    let found = shipping
        .iter()
        .filter(|l| l.contains("vận tải biển"))
        .count();
    assert!(found >= 44, "{found}: {shipping:?}");
    assert!(
        !shipping
            .iter()
            .any(|l| l == "Org 0 vận tải biển" || l == "Org 6 vận tải biển"),
        "stale vectors returned after flush"
    );
    // remove a third of the entities, add new ones: segment 1 passes the deletion
    // threshold and is rebuilt from its live vectors (vacuum); the rest stays
    let mut list = Vec::new();
    for n in (1..300).filter(|n| n % 3 == 0 && *n != 6) {
        let t = topics[n % topics.len()];
        let s = res.id_map[&format!("_:o{n}")].replace("urn:dqadb:", "dqadb:");
        list.push(serde_json::json!({"op": "retract", "s": s, "p": "rdfs:label", "o": format!("Org {n} {t}")}));
    }
    for n in 300..340 {
        list.push(serde_json::json!({"op": "add", "s": format!("_:o{n}"), "p": "rdf:type", "o": "ex:Organization"}));
        list.push(serde_json::json!({"op": "add", "s": format!("_:o{n}"), "p": "rdfs:label", "o": format!("Org {n} năng lượng mặt trời")}));
    }
    kb.assert(&ops(serde_json::Value::Array(list)), &opts())
        .unwrap();
    kb.embed_sync().unwrap();
    let before = sem("năng lượng mặt trời", 40);
    let c = dqadb_core::ann::build(
        &dir.path().join(".dqadb/index.sqlite"),
        &dir.path().join(".dqadb/vec"),
        64,
        false,
    )
    .unwrap();
    assert!(c.incremental && c.vectors == 299 - 98 + 40, "{c:?}");
    let m = dqadb_core::ann::load_manifest(&dir.path().join(".dqadb/vec")).unwrap();
    assert!(m.segments.iter().all(|s| s.deleted == 0), "{m:?}");
    assert_eq!(c.written, 40 + 200, "{c:?}");
    // these 40 vectors are near-identical, the worst case for HNSW: check most are found
    let after = sem("năng lượng mặt trời", 40);
    let solar = after
        .iter()
        .filter(|l| l.contains("năng lượng mặt trời"))
        .count();
    assert!(solar >= 24, "{solar}: {after:?}");
    // nothing deleted comes back
    let all = sem("vận tải biển", 300);
    for n in (1..300).filter(|n| n % 3 == 0) {
        let t = topics[n % topics.len()];
        assert!(
            !all.contains(&format!("Org {n} {t}")),
            "deleted Org {n} returned"
        );
    }
    assert!(
        before.iter().all(|l| l.contains("năng lượng mặt trời")),
        "{before:?}"
    );
    // small flushes past the segment cap merge by appending into an existing graph;
    // each round also updates an entity of the previous round, so the graph appended
    // into carries deleted nodes (tombstoned in the graph, no deletion list)
    let live_ids: Vec<usize> = (1..300).filter(|n| n % 3 != 0).collect();
    let mut label: std::collections::HashMap<usize, String> = live_ids
        .iter()
        .map(|&n| (n, format!("Org {n} {}", topics[n % topics.len()])))
        .collect();
    let mut old_labels = Vec::new();
    let mut appended = false;
    let mut stale_read = false;
    for round in 0..dqadb_core::ann::max_segments() + 3 {
        let mut pick: Vec<usize> = live_ids[10 * round..10 * round + 10].to_vec();
        if round > 0 {
            pick.push(live_ids[10 * (round - 1)]);
        }
        let mut list = Vec::new();
        for &n in &pick {
            let s = res.id_map[&format!("_:o{n}")].replace("urn:dqadb:", "dqadb:");
            // the test embedder ignores numbers: spell them as letters
            let word = |mut x: usize| {
                let mut w = String::new();
                loop {
                    w.push((b'a' + (x % 26) as u8) as char);
                    x /= 26;
                    if x == 0 {
                        break w;
                    }
                }
            };
            let new = format!("Org {n} {}x{} hàng không", word(n), word(round));
            list.push(
                serde_json::json!({"op": "retract", "s": s, "p": "rdfs:label", "o": label[&n]}),
            );
            list.push(serde_json::json!({"op": "add", "s": s, "p": "rdfs:label", "o": new}));
            old_labels.push(std::mem::replace(label.get_mut(&n).unwrap(), new));
        }
        kb.assert(&ops(serde_json::Value::Array(list)), &opts())
            .unwrap();
        kb.embed_sync().unwrap();
        let before = dqadb_core::ann::load_manifest(&dir.path().join(".dqadb/vec")).unwrap();
        let c = dqadb_core::ann::build(
            &dir.path().join(".dqadb/index.sqlite"),
            &dir.path().join(".dqadb/vec"),
            64,
            false,
        )
        .unwrap();
        assert_eq!(c.vectors, 241, "{c:?}");
        // a reader still holding the previous manifest (its files may be gone) retries
        // with the current one
        let store =
            dqadb_core::store::Store::open(&dir.path().join(".dqadb/index.sqlite")).unwrap();
        let reader = store.read().unwrap();
        let q: Vec<f32> = (0..64).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let (hits, _) = dqadb_core::ann::search(
            &dir.path().join(".dqadb/vec"),
            &before,
            &reader,
            &q,
            5,
            None,
        )
        .unwrap();
        assert_eq!(hits.len(), 5);
        stale_read |= before.segments.iter().any(|s| {
            !dir.path()
                .join(format!(".dqadb/vec/{}.usearch", s.name))
                .exists()
        });
        let m = dqadb_core::ann::load_manifest(&dir.path().join(".dqadb/vec")).unwrap();
        assert!(
            m.segments.len() <= dqadb_core::ann::max_segments() + 1,
            "{m:?}"
        );
        appended |= m.segments.iter().any(|s| s.deleted > 0 && s.del.is_none());
        // files of replaced segments are removed right away
        let referenced =
            2 * m.segments.len() + m.segments.iter().filter(|s| s.del.is_some()).count();
        let on_disk = std::fs::read_dir(dir.path().join(".dqadb/vec"))
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with('s'))
            .filter(|e| e.file_name() != "segments.json")
            .count();
        assert_eq!(on_disk, referenced, "{m:?}");
        // an append inserts only the smaller segments' vectors, never all of them
        assert!(c.written < 241, "{c:?}");
    }
    assert!(
        appended,
        "no merge appended into a graph with deleted nodes"
    );
    assert!(
        stale_read,
        "no search ran on a manifest whose files were removed"
    );
    let all = sem("hàng không", 300);
    for l in &old_labels {
        assert!(!all.contains(l), "replaced label {l} returned");
    }
    for n in &live_ids[..10 * (dqadb_core::ann::max_segments() + 3)] {
        assert_eq!(sem(&label[n], 1), vec![label[n].clone()]);
    }

    // a forced full build puts everything back into one segment
    kb.assert(
        &ops(
            serde_json::json!([{"op": "add", "s": "_:z", "p": "rdf:type", "o": "ex:Organization"},
                                {"op": "add", "s": "_:z", "p": "rdfs:label", "o": "Z"}]),
        ),
        &opts(),
    )
    .unwrap();
    kb.embed_sync().unwrap();
    let c = dqadb_core::ann::build(
        &dir.path().join(".dqadb/index.sqlite"),
        &dir.path().join(".dqadb/vec"),
        64,
        true,
    )
    .unwrap();
    assert!(
        c.built && !c.incremental && c.vectors == 242 && c.segments == 1,
        "{c:?}"
    );
    assert_eq!(sem("trí tuệ nhân tạo", 1), vec!["Org 0 trí tuệ nhân tạo"]);

    // index rebuild = new vector-table generation: old segments are ignored until rebuilt
    kb.rebuild().unwrap();
    kb.embed(false).unwrap();
    assert_eq!(sem("trí tuệ nhân tạo", 1), vec!["Org 0 trí tuệ nhân tạo"]);
    let c = kb.embed(true).unwrap().compaction.unwrap();
    assert_eq!(c.vectors, 242);
    assert_eq!(sem("trí tuệ nhân tạo", 1), vec!["Org 0 trí tuệ nhân tạo"]);
}

#[test]
fn text_mode_must_requires_text_and_traverse_skips_start() {
    use dqadb_core::search::{SearchQuery, TextMode};
    let (_dir, kb) = hash_kb();
    let mut ids = std::collections::HashMap::new();
    for (k, label, desc) in [
        (
            "rail",
            "Đường Sắt Hàng Hóa",
            "vận tải hàng hóa bằng đường sắt",
        ),
        ("ship", "Vinaship", "hãng tàu container vận tải biển"),
        ("port", "Cảng Mới", "kho bãi container tại cảng"),
    ] {
        let r = kb
            .assert(
                &ops(serde_json::json!([
                    {"op": "add", "s": "_:x", "p": "rdf:type", "o": "ex:Company"},
                    {"op": "add", "s": "_:x", "p": "rdfs:label", "o": label},
                    {"op": "add", "s": "_:x", "p": "ex:description", "o": desc}
                ])),
                &opts(),
            )
            .unwrap();
        ids.insert(k, r.id_map["_:x"].replace("urn:dqadb:", "dqadb:"));
    }
    kb.embed_sync().unwrap();
    let run = |mode: TextMode| -> Vec<String> {
        kb.search(&SearchQuery {
            semantic: Some("vận tải hàng hóa".into()),
            text: Some("container".into()),
            text_mode: mode,
            limit: 3,
            ..Default::default()
        })
        .unwrap()
        .hits
        .into_iter()
        .filter_map(|h| h.label)
        .collect()
    };
    // must: only entities containing "container", ranked by meaning
    let must = run(TextMode::Must);
    assert_eq!(must.len(), 2, "{must:?}");
    assert!(!must.contains(&"Đường Sắt Hàng Hóa".to_string()));
    // fuse: semantic-only matches may appear
    assert!(run(TextMode::Fuse).contains(&"Đường Sắt Hàng Hóa".to_string()));

    // a contract shared by two companies; going there and back must not list the start
    kb.assert(
        &ops(serde_json::json!([
            {"op": "add", "s": "_:c", "p": "rdf:type", "o": "ex:Contract"},
            {"op": "add", "s": "_:c", "p": "rdfs:label", "o": "HĐ 1"},
            {"op": "add", "s": ids["ship"], "p": "ex:has_contract", "o": "_:c"},
            {"op": "add", "s": ids["port"], "p": "ex:has_contract", "o": "_:c"}
        ])),
        &opts(),
    )
    .unwrap();
    let partners: Vec<String> = kb
        .traverse(&ids["ship"], "ex:has_contract/^ex:has_contract", 10)
        .unwrap()
        .into_iter()
        .filter_map(|h| h.label)
        .collect();
    assert_eq!(partners, vec!["Cảng Mới"]);
    // the same, reading the partners' labels: the start's own label must not appear
    let labels: Vec<String> = kb
        .traverse(
            &ids["ship"],
            "ex:has_contract/^ex:has_contract/rdfs:label",
            10,
        )
        .unwrap()
        .into_iter()
        .map(|h| h.iri)
        .collect();
    assert_eq!(labels, vec!["\"Cảng Mới\""]);
}

#[test]
fn semantic_search_with_loose_filters_post_filters_correctly() {
    use dqadb_core::search::SearchQuery;
    let (_dir, kb) = hash_kb();
    let mut list = Vec::new();
    for n in 0..400 {
        let country = if n % 10 == 0 { "SG" } else { "VN" };
        list.push(serde_json::json!({"op": "add", "s": format!("_:o{n}"), "p": "rdf:type", "o": "ex:Company"}));
        list.push(serde_json::json!({"op": "add", "s": format!("_:o{n}"), "p": "rdfs:label", "o": format!("Org {n} vận tải biển container")}));
        list.push(serde_json::json!({"op": "add", "s": format!("_:o{n}"), "p": "ex:country", "o": country}));
    }
    kb.assert(&ops(serde_json::Value::Array(list)), &opts())
        .unwrap();
    kb.embed(true).unwrap();
    for (filter, want) in [
        ("ex:country = \"VN\"", "VN"),
        ("ex:country = \"SG\"", "SG"),
        ("ex:country != \"SG\"", "VN"),
    ] {
        let mut seen = std::collections::HashSet::new();
        let mut cursor = None;
        for _ in 0..3 {
            let page = kb
                .search(&SearchQuery {
                    semantic: Some("vận tải biển".into()),
                    wheres: vec![filter.into()],
                    limit: 5,
                    cursor: cursor.clone(),
                    ..Default::default()
                })
                .unwrap();
            assert_eq!(page.hits.len(), 5, "{filter}");
            for h in &page.hits {
                let card = kb.get(&h.iri).unwrap();
                let c = card.props.iter().find(|p| p.p == "ex:country").unwrap();
                assert_eq!(c.value, want, "{filter} returned {}", h.iri);
                assert!(
                    seen.insert(h.iri.clone()),
                    "duplicate across pages: {}",
                    h.iri
                );
            }
            cursor = page.next_cursor;
        }
    }
}

#[test]
fn import_turtle_validates_whole_file_then_writes_batched_events() {
    use dqadb_core::import::ImportOpts;
    let (dir, kb) = new_kb();
    // external IRIs, a blank node used across batches, typed / language / enum literals;
    // with batch 2, Acme's type and its required label land in different events
    let ttl = dir.path().join("orgs.ttl");
    std::fs::write(
        &ttl,
        r#"@prefix ex: <https://example.org/ont#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
<http://acme.example/org> a ex:Company ;
    ex:country "VN" ;
    rdfs:label "ACME"@en ;
    ex:has_contract _:c .
_:c a ex:Contract ;
    rdfs:label "HĐ 1" ;
    ex:value "1200.50"^^xsd:decimal ;
    ex:status "active" .
"#,
    )
    .unwrap();
    let mut seen = Vec::new();
    let dry = kb
        .import(
            &ttl,
            &ImportOpts {
                batch: 2,
                dry_run: true,
                ..Default::default()
            },
            &opts(),
            &mut |phase, n| seen.push((phase.to_string(), n)),
        )
        .unwrap();
    assert!(dry.issues.is_empty(), "{:?}", dry.issues);
    assert_eq!((dry.triples, dry.blank_nodes, dry.events.len()), (8, 1, 0));
    assert_eq!(seen.last().unwrap(), &("validated".to_string(), 8));
    assert!(!dir.path().join("sources").read_dir().unwrap().any(|_| true));
    assert_eq!(kb.status().unwrap().counts.triples, 0);

    let r = kb
        .import(
            &ttl,
            &ImportOpts {
                batch: 2,
                ..Default::default()
            },
            &opts(),
            &mut |_, _| {},
        )
        .unwrap();
    assert_eq!(r.events.len(), 4);
    // one stored copy of the file, referenced by every event
    let ev = kb.event(&r.events[3]).unwrap();
    let src = ev.source.unwrap();
    assert!(src.sha256.is_some() && src.path.ends_with("orgs.ttl"));
    assert!(ev.note.unwrap().contains("part 4/4"));
    let rows = select(
        &kb,
        "SELECT ?l ?v ?st WHERE { <http://acme.example/org> ex:has_contract ?c . ?c rdfs:label ?l ; ex:value ?v ; ex:status ?st }",
    );
    // decimals are stored canonical, as with `assert`
    assert_eq!(
        rows,
        vec![vec![
            "\"HĐ 1\"".to_string(),
            "1200.5".into(),
            "\"active\"".into()
        ]]
    );
    // the blank node became one dqadb: entity
    let c = select(&kb, "SELECT ?c WHERE { ?c a ex:Contract }");
    assert_eq!(c.len(), 1);
    assert!(c[0][0].starts_with("dqadb:"), "{c:?}");

    // invalid file: nothing written, issues counted per kind with the first position
    let bad = dir.path().join("bad.nt");
    std::fs::write(
        &bad,
        "<http://x.example/a> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <https://example.org/ont#Company> .\n\
         <http://x.example/a> <https://example.org/ont#ceo> \"Ann\" .\n\
         <http://x.example/b> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <https://example.org/ont#Company> .\n\
         <http://x.example/b> <https://example.org/ont#ceo> \"Bob\" .\n",
    )
    .unwrap();
    let before = kb.status().unwrap().counts.triples;
    let r = kb
        .import(&bad, &ImportOpts::default(), &opts(), &mut |_, _| {})
        .unwrap();
    assert!(r.events.is_empty());
    let issues: Vec<(String, usize, usize)> = r
        .issues
        .iter()
        .map(|i| (i.message.clone(), i.count, i.first_triple))
        .collect();
    assert_eq!(
        issues,
        vec![
            ("Organization requires rdfs:label".to_string(), 2, 1),
            ("unknown predicate ex:ceo".to_string(), 2, 2)
        ]
    );
    assert_eq!(kb.status().unwrap().counts.triples, before);
    // named graphs are refused
    let nq = dir.path().join("g.nq");
    std::fs::write(&nq, "<http://x.example/a> <http://www.w3.org/2000/01/rdf-schema#label> \"A\" <http://g.example/1> .\n").unwrap();
    let e = kb
        .import(&nq, &ImportOpts::default(), &opts(), &mut |_, _| {})
        .unwrap_err();
    assert!(e.to_string().contains("named graph"), "{e}");
}

#[test]
fn linkml_types_and_local_datetimes() {
    use dqadb_core::import::ImportOpts;
    let (dir, kb) = new_kb();
    // a schema type (decimal-valued currency) and an xsd:dateTime without timezone
    std::fs::write(
        dir.path().join("ontology/offer.linkml.yaml"),
        "id: https://example.org/ont/offer\nname: offer\nprefixes:\n  ex: https://example.org/ont#\n\
         default_prefix: ex\ntypes:\n  USD:\n    uri: ex:USD\n    typeof: decimal\n\
         classes:\n  Offer:\n    slots: [price, valid_from]\n\
         slots:\n  price:\n    range: USD\n  valid_from:\n    range: datetime\n",
    )
    .unwrap();
    kb.build_ontology(true).unwrap();
    let nt = dir.path().join("offer.nt");
    let line =
        |p: &str, v: &str| format!("<http://o.example/1> <https://example.org/ont#{p}> {v} .\n");
    std::fs::write(
        &nt,
        format!(
            "<http://o.example/1> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <https://example.org/ont#Offer> .\n{}{}",
            line("price", "\"12.50\"^^<https://example.org/ont#USD>"),
            line("valid_from", "\"2008-06-20T00:00:00\"^^<http://www.w3.org/2001/XMLSchema#dateTime>")
        ),
    )
    .unwrap();
    let r = kb
        .import(&nt, &ImportOpts::default(), &opts(), &mut |_, _| {})
        .unwrap();
    assert!(r.issues.is_empty(), "{:?}", r.issues);
    let rows = select(
        &kb,
        "SELECT ?p (DATATYPE(?p) AS ?t) ?v WHERE { ?o ex:price ?p ; ex:valid_from ?v }",
    );
    // checked as a decimal (canonical form), kept with its own datatype
    assert_eq!(
        rows,
        vec![vec![
            "\"12.5\"^^ex:USD".to_string(),
            "ex:USD".into(),
            "\"2008-06-20T00:00:00\"^^xsd:dateTime".into()
        ]]
    );
    // not a decimal: rejected by the type's base
    std::fs::write(
        &nt,
        line("price", "\"cheap\"^^<https://example.org/ont#USD>"),
    )
    .unwrap();
    let r = kb
        .import(&nt, &ImportOpts::default(), &opts(), &mut |_, _| {})
        .unwrap();
    assert_eq!(r.issues.len(), 1, "{:?}", r.issues);
    assert!(
        r.issues[0].message.contains("expects ex:USD"),
        "{:?}",
        r.issues
    );
}

#[test]
fn readers_find_new_events_through_head() {
    use dqadb_core::event;
    let (dir, kb) = new_kb();
    let events = dir.path().join("events");
    let first = kb.assert(&acme(), &opts()).unwrap();
    let head = |p: &std::path::Path| std::fs::read_to_string(p.join("HEAD")).unwrap();
    assert_eq!(head(&events).trim(), first.event_id);

    // another process writes: a fresh reader learns about it from HEAD alone
    let writer = Kb::open(dir.path()).unwrap();
    let second = writer
        .assert(
            &ops(
                serde_json::json!([{"op": "add", "s": "_:b", "p": "rdf:type", "o": "ex:Company"},
                                    {"op": "add", "s": "_:b", "p": "rdfs:label", "o": "Beta"}]),
            ),
            &opts(),
        )
        .unwrap();
    assert_eq!(head(&events).trim(), second.event_id);
    assert_eq!(
        select(
            &kb,
            "SELECT ?l WHERE { ?c a ex:Company ; rdfs:label ?l } ORDER BY ?l"
        )
        .len(),
        2
    );

    // an event file copied in by hand (HEAD not moved): neither readers nor writers list
    // the folders, so nobody sees it until `sync --full` scans them
    let mut ev = kb.event(&second.event_id).unwrap();
    let late = ulid::Ulid::generate();
    ev.event_id = late.to_string();
    for op in &mut ev.ops {
        op.s = op.s.replace(&second.id_map["_:b"], "urn:dqadb:late");
        if let dqadb_core::rdf::Term::Literal(l) = &mut op.o {
            l.value = "Late".into();
        }
    }
    event::write_atomic(&events, &ev).unwrap();
    assert_eq!(head(&events).trim(), second.event_id);
    let labels = |kb: &Kb| {
        select(
            kb,
            "SELECT ?l WHERE { ?c a ex:Company ; rdfs:label ?l } ORDER BY ?l",
        )
    };
    assert_eq!(labels(&kb).len(), 2);
    assert_eq!(kb.sync().unwrap(), 0);
    assert_eq!(kb.sync_full().unwrap(), 1);
    assert_eq!(head(&events).trim(), late.to_string());
    assert_eq!(labels(&kb).len(), 3);

    // a writer that died after moving HEAD but before writing its event file: the next
    // write finds HEAD on a missing file (one stat), repairs it and goes on
    std::fs::write(events.join("HEAD"), format!("{}\n", ulid::Ulid::generate())).unwrap();
    let zeta = kb
        .assert(
            &ops(
                serde_json::json!([{"op": "add", "s": "_:z", "p": "rdf:type", "o": "ex:Company"},
                                    {"op": "add", "s": "_:z", "p": "rdfs:label", "o": "Zeta"}]),
            ),
            &opts(),
        )
        .unwrap();
    assert_eq!(head(&events).trim(), zeta.event_id);
    assert_eq!(labels(&kb).len(), 4, "{:?}", labels(&kb));

    // no HEAD (older KB): readers fall back to listing the files
    std::fs::remove_file(events.join("HEAD")).unwrap();
    let fresh = Kb::open(dir.path()).unwrap();
    assert_eq!(labels(&fresh).len(), 4);
    // a HEAD ahead of the files is repaired by the next sync
    std::fs::write(events.join("HEAD"), format!("{}\n", ulid::Ulid::generate())).unwrap();
    assert_eq!(labels(&fresh).len(), 4);
    assert_eq!(
        event::head(&events).unwrap(),
        event::latest(&events).unwrap()
    );
}

#[test]
fn sparql_protocol_requests_and_content_negotiation() {
    use dqadb_core::sparql_http::{Request, handle};
    let (_dir, kb) = new_kb();
    kb.assert(&acme(), &opts()).unwrap();
    let limits = Limits::default();
    let enc = |q: &str| form_urlencoded::byte_serialize(q.as_bytes()).collect::<String>();
    let q = "SELECT ?l WHERE { ?c a ex:Company ; rdfs:label ?l }";
    let get = |url_query: &str, accept: Option<&str>| {
        handle(
            &kb,
            &Request {
                method: "GET",
                url_query: Some(url_query),
                content_type: None,
                accept,
                body: b"",
            },
            &limits,
        )
    };
    let body = |r: &dqadb_core::sparql_http::Response| String::from_utf8(r.body.clone()).unwrap();

    // GET, default format: SPARQL results JSON
    let r = get(&format!("query={}", enc(q)), None);
    assert_eq!(
        (r.status, r.content_type.as_str()),
        (200, "application/sparql-results+json")
    );
    let j: serde_json::Value = serde_json::from_slice(&r.body).unwrap();
    assert_eq!(j["head"]["vars"], serde_json::json!(["l"]));
    assert_eq!(j["results"]["bindings"][0]["l"]["value"], "ACME Logistics");
    assert_eq!(j["results"]["bindings"][0]["l"]["xml:lang"], "vi");
    // Accept with q-values: CSV preferred over XML
    let r = get(
        &format!("query={}", enc(q)),
        Some("application/sparql-results+xml;q=0.5, text/csv"),
    );
    assert_eq!(
        (r.status, r.content_type.as_str()),
        (200, "text/csv; charset=utf-8")
    );
    assert_eq!(body(&r), "l\r\nACME Logistics\r\n");
    let r = get(
        &format!("query={}", enc(q)),
        Some("application/sparql-results+xml"),
    );
    assert!(body(&r).contains("<sparql"), "{}", body(&r));
    // ASK
    let r = get(
        &format!("query={}", enc("ASK { ?c a ex:Contract }")),
        Some("application/json"),
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&r.body).unwrap()["boolean"],
        true
    );
    // CONSTRUCT: N-Triples by default, Turtle on request, 406 for a results-only type
    let c = "CONSTRUCT { ?c ex:country ?x } WHERE { ?c ex:country ?x }";
    let r = get(&format!("query={}", enc(c)), None);
    assert_eq!(r.content_type, "application/n-triples");
    assert!(
        body(&r).contains("<https://example.org/ont#country> \"VN\" ."),
        "{}",
        body(&r)
    );
    let r = get(&format!("query={}", enc(c)), Some("text/turtle"));
    assert_eq!(r.content_type, "text/turtle; charset=utf-8");
    let r = get(
        &format!("query={}", enc(c)),
        Some("application/sparql-results+json"),
    );
    assert_eq!(r.status, 406);

    // POST form and POST application/sparql-query
    let post = |ct: &str, b: &[u8]| {
        handle(
            &kb,
            &Request {
                method: "POST",
                url_query: None,
                content_type: Some(ct),
                accept: Some("text/tab-separated-values"),
                body: b,
            },
            &limits,
        )
    };
    let r = post(
        "application/x-www-form-urlencoded",
        format!("query={}", enc(q)).as_bytes(),
    );
    assert_eq!(
        (r.status, body(&r).as_str()),
        (200, "?l\n\"ACME Logistics\"@vi\n")
    );
    let r = post("application/sparql-query; charset=utf-8", q.as_bytes());
    assert_eq!(r.status, 200);

    // refused or malformed requests
    assert_eq!(
        post("application/sparql-update", b"INSERT DATA {}").status,
        400
    );
    assert_eq!(
        post(
            "application/x-www-form-urlencoded",
            b"update=INSERT+DATA+%7B%7D"
        )
        .status,
        400
    );
    assert_eq!(post("text/plain", q.as_bytes()).status, 415);
    assert_eq!(get("", None).status, 400);
    assert_eq!(
        get(&format!("query={}&query={}", enc(q), enc(q)), None).status,
        400
    );
    assert_eq!(
        get(&format!("query={}&default-graph-uri=urn%3Ax", enc(q)), None).status,
        400
    );
    let r = get(&format!("query={}", enc("SELECT WHERE")), None);
    assert_eq!(r.status, 400);
    assert!(body(&r).contains("SPARQL syntax error"), "{}", body(&r));
    let r = handle(
        &kb,
        &Request {
            method: "PUT",
            url_query: None,
            content_type: None,
            accept: None,
            body: b"",
        },
        &limits,
    );
    assert_eq!(r.status, 405);
}

#[test]
fn full_text_bigrams_and_slot_weights_rank() {
    use dqadb_core::search::SearchQuery;
    let (dir, kb) = new_kb();
    kb.assert(
        &ops(serde_json::json!([
            {"op": "add", "s": "_:a", "p": "rdf:type", "o": "ex:Company"},
            {"op": "add", "s": "_:a", "p": "rdfs:label", "o": "Alpha"},
            {"op": "add", "s": "_:a", "p": "ex:description", "o": "Đồng bộ dữ liệu và hợp nhất kho"},
            {"op": "add", "s": "_:b", "p": "rdf:type", "o": "ex:Company"},
            {"op": "add", "s": "_:b", "p": "rdfs:label", "o": "Beta"},
            {"op": "add", "s": "_:b", "p": "ex:description", "o": "Ký hợp đồng vận tải kho bãi lớn nhất miền Bắc"},
            {"op": "add", "s": "_:e", "p": "rdf:type", "o": "ex:Company"},
            {"op": "add", "s": "_:e", "p": "rdfs:label", "o": "Gamma"},
            {"op": "add", "s": "_:e", "p": "ex:description", "o": "logistics"},
            {"op": "add", "s": "_:f", "p": "rdf:type", "o": "ex:Company"},
            {"op": "add", "s": "_:f", "p": "rdfs:label", "o": "Logistics Xanh"},
            {"op": "add", "s": "_:f", "p": "ex:description", "o": "dịch vụ giao nhận hàng hóa toàn quốc"}
        ])),
        &opts(),
    )
    .unwrap();
    let text = |kb: &Kb, t: &str| -> Vec<String> {
        kb.search(&SearchQuery {
            text: Some(t.into()),
            limit: 10,
            ..Default::default()
        })
        .unwrap()
        .hits
        .into_iter()
        .filter_map(|h| h.label)
        .collect()
    };
    // both have "hop" and "dong"; Beta has them as the word "hợp đồng" (a bigram) and
    // ranks first although its text is longer; bigrams do not change what matches
    assert_eq!(text(&kb, "hợp đồng"), vec!["Beta", "Alpha"]);
    assert_eq!(text(&kb, "dong hop"), vec!["Alpha", "Beta"]);

    // slot weights: the short description wins until labels weigh 4x (fts^4)
    assert_eq!(text(&kb, "logistics"), vec!["Gamma", "Logistics Xanh"]);
    let schema = dir.path().join("ontology/schema.linkml.yaml");
    let yaml = std::fs::read_to_string(&schema).unwrap();
    std::fs::write(
        &schema,
        yaml.replacen("dqadb_index: fts,card", "dqadb_index: fts^4,card", 1),
    )
    .unwrap();
    kb.build_ontology(true).unwrap();
    assert_eq!(text(&kb, "logistics"), vec!["Logistics Xanh", "Gamma"]);
    // a weight below 1 is refused with the reason
    std::fs::write(
        &schema,
        yaml.replacen("dqadb_index: fts,card", "dqadb_index: fts^0.5,card", 1),
    )
    .unwrap();
    let err = kb.build_ontology(true).unwrap_err().to_string();
    assert!(err.contains(">= 1"), "{err}");
}
