//! `dqadb` — the CLI harness for agents and people (spec §12).

use clap::{Args, Parser, Subcommand};
use dqadb_core::kb::{Card, EventSummary, WriteResult};
use dqadb_core::ontology::{InputOp, Range};
use dqadb_core::rdf::Prefixes;
use dqadb_core::search::{Hit, SearchQuery};
use dqadb_core::sparql::{self, Answer, Limits};
use dqadb_core::{Error, Kb, Result, WriteOpts};
use std::io::Read;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "dqadb",
    version,
    about = "dqaDB: ontology graph DB for AI agents. Files are the source of truth; the index is rebuildable."
)]
struct Cli {
    /// Knowledge base directory (default: search upwards from the current directory)
    #[arg(long, global = true, env = "DQADB_KB")]
    kb: Option<PathBuf>,
    /// Machine-readable JSON output
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Create a knowledge base: dqadb.yaml, folders and a sample ontology
    #[command(after_help = "Examples:\n  dqadb init my-kb\n  dqadb init . --base-iri urn:dqadb:")]
    Init {
        #[arg(default_value = ".")]
        path: PathBuf,
        #[arg(long, default_value = dqadb_core::config::DEFAULT_BASE_IRI)]
        base_iri: String,
    },
    /// Counts, watermark, and whether the index is in sync with the event files
    Status,
    /// Delete and rebuild the derived index (.dqadb/) from events
    Rebuild,
    /// Check every event file (parse, name, sharded path)
    Verify,
    /// Embed entities changed since the last run (rdfs:label + `dqadb_index: vector` fields).
    /// Uses the cache, so only new or changed text reaches the model.
    #[command(
        after_help = "Model: embedding.model_dir / DQADB_MODEL_DIR (model.onnx, model.onnx_data, tokenizer.json)\n\
        Runtime: embedding.onnxruntime / DQADB_ORT_LIB (official libonnxruntime.so)"
    )]
    Embed,
    /// Mint new entity IRIs
    #[command(after_help = "Examples:\n  dqadb new-id\n  dqadb new-id -n 5")]
    NewId {
        #[arg(short, default_value_t = 1)]
        n: usize,
    },
    /// Ontology (LinkML) commands
    #[command(subcommand)]
    Ontology(OntologyCmd),
    /// Validate ops and append them as one event
    #[command(
        after_help = "Input: a JSON array of ops, or {\"ops\": [...], \"note\": \"...\"}.\n\
        Op: {\"op\": \"add\"|\"retract\", \"s\": IRI, \"p\": IRI, \"o\": IRI or literal}. Use _:name for new entities.\n\n\
        Examples:\n  dqadb assert -f facts.json --source contract-123.pdf --actor agent:extractor\n  \
        echo '[{\"op\":\"add\",\"s\":\"_:a\",\"p\":\"rdf:type\",\"o\":\"ex:Company\"},{\"op\":\"add\",\"s\":\"_:a\",\"p\":\"rdfs:label\",\"o\":\"ACME\"}]' | dqadb assert -f -"
    )]
    Assert {
        #[command(flatten)]
        input: InputArgs,
        #[command(flatten)]
        write: WriteArgs,
    },
    /// Dry-run validation: report issues, write nothing
    #[command(after_help = "Example:\n  dqadb validate -f facts.json")]
    Validate {
        #[command(flatten)]
        input: InputArgs,
    },
    /// Retract every triple added by an earlier event
    #[command(
        after_help = "Example:\n  dqadb retract --event 01JC9X2K4M0A7Q8R3T5V6W7Y8Z --why \"wrong extraction\""
    )]
    Retract {
        #[arg(long)]
        event: String,
        #[arg(long)]
        why: String,
        #[command(flatten)]
        write: WriteArgs,
    },
    /// Entity card: types, properties, neighbours, provenance
    #[command(
        after_help = "Examples:\n  dqadb get dqadb:01JC9Y...\n  dqadb get dqadb:01JC9Y... --json"
    )]
    Get { iri: String },
    /// SPARQL 1.1 query over the current state (asserted + inferred facts)
    #[command(
        after_help = "KB prefixes (ex:, dqadb:, rdf:, rdfs:, xsd:, owl:) are predeclared.\n\
        Named graph dqadb:event:<ULID> holds what that event asserted (provenance).\n\n\
        Examples:\n  dqadb query 'SELECT ?c ?v WHERE { ?c a ex:Contract ; ex:value ?v FILTER(?v > 10000) }'\n  \
        dqadb query 'SELECT ?g WHERE { GRAPH ?g { ?s ex:country \"VN\" } }'\n  \
        dqadb query -f q.rq --limit 100 --json"
    )]
    Query {
        /// Query text (or use -f)
        sparql: Option<String>,
        /// Read the query from a file ('-' = stdin)
        #[arg(short, long, conflicts_with = "sparql")]
        file: Option<PathBuf>,
        /// Maximum rows returned
        #[arg(long, default_value_t = 1000)]
        limit: usize,
        /// Seconds before the query is cancelled
        #[arg(long, default_value_t = 10)]
        timeout: u64,
    },
    /// Find entities by type and field filters
    #[command(
        after_help = "Filters (repeat --where for AND):\n  ex:p = \"text\" | ex:p != \"text\" | ex:p >= 10 | ex:p < 2026-01-01\n  ex:p in (\"A\", \"B\") | ex:p = dqadb:<id> | has ex:p\n\n\
        Examples:\n  dqadb search --type ex:Company --where 'ex:country = \"VN\"'\n  \
        dqadb search --type ex:Contract --where 'ex:value >= 10000' --where 'ex:status in (active, draft)'"
    )]
    Search {
        /// Class the entity must have (inferred types count); repeatable
        #[arg(long = "type")]
        types: Vec<String>,
        /// Field filter; repeatable (AND)
        #[arg(long = "where")]
        wheres: Vec<String>,
        /// Full-text query, ranked by BM25 (accents and đ/d ignored)
        #[arg(long)]
        text: Option<String>,
        /// Semantic query (embedding cosine); combined with --text by RRF.
        /// Run `dqadb embed` first; the output reports events not yet embedded.
        #[arg(long)]
        semantic: Option<String>,
        #[arg(short = 'k', long, default_value_t = 10)]
        limit: usize,
        /// Continue from a previous page
        #[arg(long)]
        cursor: Option<String>,
    },
    /// Follow a SPARQL property path from an entity
    #[command(
        after_help = "Examples:\n  dqadb traverse dqadb:01JC... 'ex:has_contract'\n  \
        dqadb traverse dqadb:01JC... '^ex:has_contract'          (who points here)\n  \
        dqadb traverse dqadb:01JC... 'ex:has_contract/rdfs:label'"
    )]
    Traverse {
        iri: String,
        path: String,
        #[arg(short = 'k', long, default_value_t = 50)]
        limit: usize,
    },
    /// Shortest chain of entity links between two entities (either direction)
    #[command(after_help = "Example:\n  dqadb path dqadb:01JC... dqadb:01JD... --max-depth 4")]
    Path {
        from: String,
        to: String,
        #[arg(long, default_value_t = 4)]
        max_depth: usize,
    },
    /// Inspect event files
    #[command(subcommand)]
    Events(EventsCmd),
}

#[derive(Subcommand)]
enum OntologyCmd {
    /// Generate TTL + SHACL from LinkML and write build.lock
    Build {
        /// Only record the schema hash; skip gen-owl/gen-shacl (no Python needed)
        #[arg(long)]
        lock_only: bool,
    },
    /// List classes and predicates (call this before writing)
    #[command(
        after_help = "Examples:\n  dqadb ontology show\n  dqadb ontology show --class ex:Contract"
    )]
    Show {
        #[arg(long)]
        class: Option<String>,
    },
}

#[derive(Subcommand)]
enum EventsCmd {
    /// Print one event file
    Show { id: String },
    /// Newest events first
    Ls {
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long)]
        actor: Option<String>,
        /// YYYY-MM-DD (inclusive)
        #[arg(long)]
        since: Option<String>,
    },
}

#[derive(Args)]
struct InputArgs {
    /// Ops file (JSON); '-' reads stdin
    #[arg(short, long)]
    file: PathBuf,
}

#[derive(Args)]
struct WriteArgs {
    /// Who writes: agent:<name>, user:<name>, system:<name>
    #[arg(long, env = "DQADB_ACTOR")]
    actor: Option<String>,
    /// Source document; copied into sources/ by content hash
    #[arg(long)]
    source: Option<PathBuf>,
    /// Fail unless the newest event is this ULID (optimistic concurrency)
    #[arg(long)]
    expected_version: Option<String>,
}

impl WriteArgs {
    fn opts(self, note: Option<String>) -> WriteOpts {
        let actor = self.actor.unwrap_or_else(|| {
            format!(
                "user:{}",
                std::env::var("USER").unwrap_or_else(|_| "unknown".into())
            )
        });
        WriteOpts {
            actor,
            source: self.source,
            note,
            expected_version: self.expected_version,
        }
    }
}

fn read_input(args: &InputArgs) -> Result<(Vec<InputOp>, Option<String>)> {
    let text = if args.file.as_os_str() == "-" {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s)?;
        s
    } else {
        std::fs::read_to_string(&args.file)?
    };
    let value: serde_json::Value = serde_json::from_str(&text)?;
    let (ops, note) = match value {
        serde_json::Value::Array(_) => (value, None),
        serde_json::Value::Object(mut m) => {
            let note = m
                .remove("note")
                .and_then(|n| n.as_str().map(str::to_string));
            (
                m.remove("ops")
                    .ok_or_else(|| Error::other("input object needs an \"ops\" array"))?,
                note,
            )
        }
        _ => {
            return Err(Error::other(
                "input must be a JSON array of ops or {\"ops\": [...]}",
            ));
        }
    };
    let ops = serde_json::from_value(ops).map_err(|e| Error::other(format!("bad op: {e}")))?;
    Ok((ops, note))
}

fn print_json<T: serde::Serialize>(v: &T) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(v)?);
    Ok(())
}

fn print_write(json: bool, r: &WriteResult) -> Result<()> {
    if json {
        return print_json(r);
    }
    println!(
        "event {}  ops={} added={} retracted={}",
        r.event_id, r.ops, r.added, r.retracted
    );
    for (tmp, iri) in &r.id_map {
        println!("  {tmp} -> {}", iri.replacen("urn:dqadb:", "dqadb:", 1));
    }
    for w in &r.warnings {
        println!("  warning: {w}");
    }
    Ok(())
}

fn print_card(c: &Card) {
    let label = c
        .label
        .as_deref()
        .map(|l| format!("  \"{l}\""))
        .unwrap_or_default();
    let types: Vec<String> = c
        .types
        .iter()
        .map(|t| {
            if t.inferred {
                format!("~{}", t.iri)
            } else {
                t.iri.clone()
            }
        })
        .collect();
    println!("{}{label}  [{}]", c.iri, types.join(", "));
    let width = c.props.iter().map(|p| p.p.len()).max().unwrap_or(0);
    for p in &c.props {
        let label = p
            .label
            .as_deref()
            .map(|l| format!(" \"{l}\""))
            .unwrap_or_default();
        let arrow = if p.is_iri { "→ " } else { "" };
        println!("  {:width$}  {arrow}{}{label}", p.p, p.value);
    }
    if c.incoming_total > 0 {
        println!("  incoming ({}):", c.incoming_total);
        for p in &c.incoming {
            let label = p
                .label
                .as_deref()
                .map(|l| format!(" \"{l}\""))
                .unwrap_or_default();
            println!("    {}{label} {} → this", p.value, p.p);
        }
    }
    if let Some(e) = &c.last_event {
        let src = e
            .source
            .as_deref()
            .map(|s| format!(" · {s}"))
            .unwrap_or_default();
        println!(
            "  last change: {} · {} · {}{src}",
            e.event_id,
            e.actor,
            &e.ts[..10]
        );
    }
    println!(
        "Next: dqadb get <iri> | dqadb events show {}",
        c.last_event
            .as_ref()
            .map(|e| e.event_id.as_str())
            .unwrap_or("<id>")
    );
}

fn print_hit(h: &Hit) {
    let label = h
        .label
        .as_deref()
        .map(|l| format!("  \"{l}\""))
        .unwrap_or_default();
    let types: Vec<String> = h
        .types
        .iter()
        .map(|t| {
            if t.inferred {
                format!("~{}", t.iri)
            } else {
                t.iri.clone()
            }
        })
        .collect();
    let types = if types.is_empty() {
        String::new()
    } else {
        format!("  [{}]", types.join(", "))
    };
    let matched: Vec<String> = h.matched.iter().map(|(p, v)| format!("{p}={v}")).collect();
    let matched = if matched.is_empty() {
        String::new()
    } else {
        format!("  {}", matched.join(" "))
    };
    println!("{}{label}{types}{matched}", h.iri);
}

/// Tab-separated rows with a header; terms compacted with the KB prefixes.
fn print_answer(prefixes: &Prefixes, answer: &Answer) {
    match answer {
        Answer::Boolean(b) => println!("{b}"),
        Answer::Solutions {
            vars,
            rows,
            truncated,
        } => {
            println!(
                "{}",
                vars.iter()
                    .map(|v| format!("?{v}"))
                    .collect::<Vec<_>>()
                    .join("\t")
            );
            for row in rows {
                let cells: Vec<String> = row
                    .iter()
                    .map(|t| {
                        t.as_ref()
                            .map(|t| sparql::display(prefixes, t))
                            .unwrap_or_default()
                    })
                    .collect();
                println!("{}", cells.join("\t"));
            }
            let more = if *truncated {
                " (truncated: raise --limit or add LIMIT/filters)"
            } else {
                ""
            };
            println!("{} row(s){more}", rows.len());
        }
        Answer::Graph { triples, truncated } => {
            for t in triples {
                println!(
                    "{} {} {} .",
                    sparql::display(prefixes, &t.subject.clone().into()),
                    sparql::display(prefixes, &t.predicate.clone().into()),
                    sparql::display(prefixes, &t.object)
                );
            }
            let more = if *truncated { " (truncated)" } else { "" };
            println!("{} triple(s){more}", triples.len());
        }
    }
}

fn print_events(json: bool, list: &[EventSummary]) -> Result<()> {
    if json {
        return print_json(&list);
    }
    for e in list {
        let src = e
            .source
            .as_deref()
            .map(|s| format!("  src={s}"))
            .unwrap_or_default();
        let note = e
            .note
            .as_deref()
            .map(|s| format!("  \"{s}\""))
            .unwrap_or_default();
        println!(
            "{}  {}  {}  ops={}{src}{note}",
            e.event_id, e.ts, e.actor, e.ops
        );
    }
    if list.is_empty() {
        println!("no events");
    }
    Ok(())
}

fn run(cli: Cli) -> Result<()> {
    let json = cli.json;
    if let Cmd::Init { path, base_iri } = &cli.cmd {
        let kb = Kb::init(path, base_iri)?;
        if json {
            return print_json(&serde_json::json!({ "root": kb.root }));
        }
        println!("initialised {}", kb.root.display());
        println!("Next: edit ontology/schema.linkml.yaml, then: dqadb ontology build");
        return Ok(());
    }
    let start = cli.kb.clone().unwrap_or_else(|| PathBuf::from("."));
    let kb = Kb::open(&start)?;

    match cli.cmd {
        Cmd::Init { .. } => unreachable!(),
        Cmd::Status => {
            let s = kb.status()?;
            if json {
                return print_json(&s);
            }
            println!("kb         {}", s.root.display());
            println!("ontology   {}", s.ontology);
            println!(
                "events     {} indexed (latest on disk: {})",
                s.counts.events,
                s.latest_event.as_deref().unwrap_or("none")
            );
            println!(
                "triples    {} ({} inferred)",
                s.counts.triples, s.counts.inferred
            );
            println!("terms      {}", s.counts.terms);
            println!(
                "index      {}",
                if s.in_sync {
                    "in sync"
                } else {
                    "behind: next command catches up, or run: dqadb rebuild"
                }
            );
        }
        Cmd::Embed => {
            let r = kb.embed_sync()?;
            if json {
                return print_json(&r);
            }
            println!(
                "embedded: {} checked, {} written, {} removed, {} model call(s){} in {:.1}s",
                r.entities_checked,
                r.vectors_written,
                r.vectors_removed,
                r.model_calls,
                if r.full_rebuild { " (full)" } else { "" },
                r.seconds
            );
        }
        Cmd::Rebuild => {
            let r = kb.rebuild()?;
            if json {
                return print_json(&r);
            }
            println!(
                "rebuilt from {} events in {:.2}s: {} triples, {} terms",
                r.events, r.seconds, r.counts.triples, r.counts.terms
            );
            println!("fingerprint {}", r.fingerprint);
        }
        Cmd::Verify => {
            let r = kb.verify()?;
            if json {
                print_json(&r)?;
            } else {
                println!("{} event files, {} indexed", r.events, r.indexed_events);
                for p in &r.problems {
                    println!("  problem: {p}");
                }
                if r.problems.is_empty() {
                    println!("ok");
                }
            }
            if !r.problems.is_empty() {
                return Err(Error::other(format!(
                    "{} problem(s) found",
                    r.problems.len()
                )));
            }
        }
        Cmd::NewId { n } => {
            let ids = kb.new_ids(n);
            if json {
                return print_json(&ids);
            }
            ids.iter().for_each(|i| println!("{i}"));
        }
        Cmd::Ontology(OntologyCmd::Build { lock_only }) => {
            let lock = kb.build_ontology(lock_only)?;
            if json {
                return print_json(&lock);
            }
            println!(
                "ontology built: {} ({} file(s), generator: {})",
                lock.hash,
                lock.sources.len(),
                lock.generator
            );
        }
        Cmd::Ontology(OntologyCmd::Show { class }) => {
            let onto = kb.ontology()?;
            let selected: Vec<_> = match &class {
                Some(c) => {
                    let iri = onto.prefixes.expand(c).map_err(Error::Other)?;
                    if !onto.classes.contains_key(&iri) {
                        return Err(Error::NotFound(format!("class {c}")));
                    }
                    onto.lineage(&iri)
                }
                None => onto.classes.values().collect(),
            };
            if json {
                let slots: Vec<_> = onto.slots.values().collect();
                return print_json(
                    &serde_json::json!({ "classes": selected, "slots": slots, "enums": onto.enums }),
                );
            }
            for c in selected {
                let parent = c
                    .is_a
                    .as_deref()
                    .map(|p| format!(" is_a {}", onto.compact(p)))
                    .unwrap_or_default();
                println!("{}{parent}", onto.compact(&c.iri));
                for s in &c.slots {
                    let d = &onto.slots[s];
                    let range = match &d.range {
                        Range::Datatype(dt) | Range::Class(dt) => onto.compact(dt),
                        Range::Enum(e) => format!(
                            "enum {e} ({})",
                            onto.enums[e].iter().cloned().collect::<Vec<_>>().join("|")
                        ),
                        Range::AnyIri => "IRI".into(),
                    };
                    let flags = [
                        d.required.then_some("required"),
                        d.multivalued.then_some("multivalued"),
                    ]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .join(", ");
                    let index = if d.index.is_empty() {
                        String::new()
                    } else {
                        format!(" index={}", d.index.join(","))
                    };
                    println!("  {} : {range} {flags}{index}", onto.compact(s));
                }
            }
            println!("Also: rdf:type (class IRI), rdfs:label, rdfs:comment");
        }
        Cmd::Assert { input, write } => {
            let (ops, note) = read_input(&input)?;
            let r = kb.assert(&ops, &write.opts(note))?;
            print_write(json, &r)?;
        }
        Cmd::Validate { input } => {
            let (ops, _) = read_input(&input)?;
            let n = kb.validate(&ops)?;
            if json {
                return print_json(&n);
            }
            println!("ok: {} op(s) valid", n.ops.len());
            n.warnings.iter().for_each(|w| println!("  warning: {w}"));
        }
        Cmd::Retract { event, why, write } => {
            let r = kb.retract_event(&event, &why, &write.opts(None))?;
            print_write(json, &r)?;
        }
        Cmd::Get { iri } => {
            let card = kb.get(&iri)?;
            if json {
                return print_json(&card);
            }
            print_card(&card);
        }
        Cmd::Query {
            sparql,
            file,
            limit,
            timeout,
        } => {
            let text = match (sparql, file) {
                (Some(q), _) => q,
                (None, Some(f)) if f.as_os_str() == "-" => {
                    let mut s = String::new();
                    std::io::stdin().read_to_string(&mut s)?;
                    s
                }
                (None, Some(f)) => std::fs::read_to_string(f)?,
                (None, None) => return Err(Error::other("pass a query string or -f <file>")),
            };
            let limits = Limits {
                rows: limit,
                timeout: std::time::Duration::from_secs(timeout),
            };
            let answer = kb.query(&text, &limits)?;
            if json {
                return print_json(&sparql::to_json(&answer));
            }
            let prefixes = kb.ontology()?.prefixes;
            print_answer(&prefixes, &answer);
        }
        Cmd::Search {
            types,
            wheres,
            text,
            semantic,
            limit,
            cursor,
        } => {
            let page = kb.search(&SearchQuery {
                types,
                wheres,
                text,
                semantic,
                limit,
                cursor,
            })?;
            if json {
                return print_json(&page);
            }
            for h in &page.hits {
                print_hit(h);
            }
            if let Some(n) = page.pending.filter(|n| *n > 0) {
                println!("note: {n} event(s) not embedded yet; run: dqadb embed");
            }
            match &page.next_cursor {
                Some(c) => println!(
                    "{} of {} shown. Next: same command with --cursor {c}",
                    page.hits.len(),
                    page.total
                ),
                None => println!("{} result(s)", page.total),
            }
        }
        Cmd::Traverse { iri, path, limit } => {
            let hits = kb.traverse(&iri, &path, limit)?;
            if json {
                return print_json(&hits);
            }
            hits.iter().for_each(print_hit);
            println!("{} result(s)", hits.len());
        }
        Cmd::Path {
            from,
            to,
            max_depth,
        } => {
            let steps = kb.path(&from, &to, max_depth)?;
            if json {
                return print_json(&steps);
            }
            match steps {
                Some(steps) if steps.is_empty() => println!("same entity"),
                Some(steps) => {
                    for s in &steps {
                        if s.forward {
                            println!("{} --{}--> {}", s.from, s.p, s.to);
                        } else {
                            println!("{} <--{}-- {}", s.from, s.p, s.to);
                        }
                    }
                    println!("{} hop(s)", steps.len());
                }
                None => {
                    return Err(Error::NotFound(format!(
                        "no path within {max_depth} hops (rdf:type links are not followed)"
                    )));
                }
            }
        }
        Cmd::Events(EventsCmd::Show { id }) => print_json(&kb.event(&id)?)?,
        Cmd::Events(EventsCmd::Ls {
            limit,
            actor,
            since,
        }) => {
            let list = kb.events_ls(limit, actor.as_deref(), since.as_deref())?;
            print_events(json, &list)?;
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let json = cli.json;
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            if json {
                let detail = match &e {
                    Error::Validation(r) => serde_json::to_value(r).unwrap_or_default(),
                    _ => serde_json::Value::Null,
                };
                eprintln!(
                    "{}",
                    serde_json::json!({ "error": e.to_string(), "code": e.exit_code(), "detail": detail })
                );
            } else {
                eprint!("error: {e}");
                if !e.to_string().ends_with('\n') {
                    eprintln!();
                }
            }
            ExitCode::from(e.exit_code() as u8)
        }
    }
}
