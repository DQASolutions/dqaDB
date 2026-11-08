use dqadb_core::ontology::InputOp;
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
    assert_eq!(card.types, vec!["ex:Company"]);
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
