//! `dqadb serve`: the resident process (spec §13/§14).
//! - keeps the embedding model loaded and answers `/embed` for CLI query embedding;
//! - runs the background vectorizer: `embed` on an interval (catch-up by watermark,
//!   HNSW base rebuild when the delta grows); writes never wait for it;
//! - serves a read-only SPARQL 1.1 Protocol endpoint at `/sparql` (GET, POST form,
//!   POST application/sparql-query; results by Accept), see `dqadb_core::sparql_http`;
//! - advertises itself in `.dqadb/serve.json` (removed on shutdown).
//!
//! Listens on 127.0.0.1 only.

use axum::body::Bytes;
use axum::extract::{RawQuery, State};
use axum::http::{HeaderMap, Method, StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use dqadb_core::serve::{ServeInfo, remove_info, write_info};
use dqadb_core::{Error, Kb, Result};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

type Reply = std::result::Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)>;

fn fail(e: impl std::fmt::Display) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({ "error": e.to_string() })),
    )
}

async fn health(State(kb): State<Arc<Kb>>) -> Reply {
    let k = kb.clone();
    let pending = tokio::task::spawn_blocking(move || k.vec_pending())
        .await
        .map_err(fail)?
        .map_err(fail)?;
    let (model, loaded) = kb.embedder_status();
    Ok(Json(serde_json::json!({
        "status": "ok",
        "kb": kb.root,
        "model": model,
        "model_loaded": loaded,
        "pending_events": pending,
    })))
}

#[derive(serde::Deserialize)]
struct EmbedRequest {
    texts: Vec<String>,
}

async fn embed(State(kb): State<Arc<Kb>>, Json(req): Json<EmbedRequest>) -> Reply {
    if req.texts.len() > 256 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "at most 256 texts per request" })),
        ));
    }
    let k = kb.clone();
    let vectors = tokio::task::spawn_blocking(move || k.embed_texts_local(&req.texts))
        .await
        .map_err(fail)?
        .map_err(fail)?;
    Ok(Json(
        serde_json::json!({ "model": kb.embedder_status().0, "vectors": vectors }),
    ))
}

struct App {
    kb: Arc<Kb>,
    query_timeout: Duration,
}

/// SPARQL 1.1 Protocol, query operation (read-only).
async fn sparql(
    State(app): State<Arc<App>>,
    method: Method,
    RawQuery(url_query): RawQuery,
    headers: HeaderMap,
    body: Bytes,
) -> axum::response::Response {
    let header = |h: header::HeaderName| {
        headers
            .get(h)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
    };
    let (content_type, accept) = (header(header::CONTENT_TYPE), header(header::ACCEPT));
    let limits = dqadb_core::sparql::Limits {
        rows: usize::MAX,
        timeout: app.query_timeout,
    };
    let kb = app.kb.clone();
    let r = tokio::task::spawn_blocking(move || {
        let req = dqadb_core::sparql_http::Request {
            method: method.as_str(),
            url_query: url_query.as_deref(),
            content_type: content_type.as_deref(),
            accept: accept.as_deref(),
            body: &body,
        };
        dqadb_core::sparql_http::handle(&kb, &req, &limits)
    })
    .await;
    match r {
        Ok(r) => (
            StatusCode::from_u16(r.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            [(header::CONTENT_TYPE, r.content_type)],
            r.body,
        )
            .into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

pub fn run(kb: Kb, port: u16, interval: Duration, query_timeout: Duration) -> Result<()> {
    let kb = Arc::new(kb);
    let (model, _) = kb.embedder_status();
    eprintln!("dqadb serve: loading embedding model {model} ...");
    let t = std::time::Instant::now();
    // without a usable model, SPARQL and the other reads are still served
    let embedding = match kb.embed_texts_local(&["warm-up".to_string()]) {
        Ok(_) => {
            eprintln!(
                "dqadb serve: model ready in {:.1}s",
                t.elapsed().as_secs_f64()
            );
            true
        }
        Err(e) => {
            eprintln!(
                "dqadb serve: embedding disabled ({e}); install it with: dqadb model install"
            );
            false
        }
    };

    let stop = Arc::new(AtomicBool::new(false));
    let worker = {
        let (kb, stop) = (kb.clone(), stop.clone());
        std::thread::spawn(move || {
            while embedding && !stop.load(Ordering::Relaxed) {
                match kb.embed(false) {
                    Ok(r) => {
                        if r.vectors_written + r.vectors_removed > 0 {
                            eprintln!(
                                "dqadb serve: embedded {} written, {} removed, {} model call(s) in {:.1}s",
                                r.vectors_written, r.vectors_removed, r.model_calls, r.seconds
                            );
                        }
                        if let Some(c) = r.compaction.filter(|c| c.built) {
                            eprintln!(
                                "dqadb serve: vector base rebuilt with {} vectors in {:.1}s",
                                c.vectors, c.seconds
                            );
                        }
                    }
                    Err(e) => eprintln!("dqadb serve: embed worker error: {e}"),
                }
                // sleep in short steps so shutdown is prompt
                let mut slept = Duration::ZERO;
                while slept < interval && !stop.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(200));
                    slept += Duration::from_millis(200);
                }
            }
        })
    };

    let index_dir = kb.index_dir();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let served: Result<()> = rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
            .await
            .map_err(|e| Error::other(format!("cannot listen on 127.0.0.1:{port}: {e}")))?;
        let port = listener.local_addr()?.port();
        write_info(
            &index_dir,
            &ServeInfo {
                pid: std::process::id(),
                port,
                model: model.clone(),
                started: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            },
        )?;
        eprintln!("dqadb serve: listening on http://127.0.0.1:{port} (Ctrl-C to stop)");
        eprintln!("dqadb serve: SPARQL endpoint http://127.0.0.1:{port}/sparql");
        let sparql_app = Arc::new(App {
            kb: kb.clone(),
            query_timeout,
        });
        let app = Router::new()
            .route("/health", get(health))
            .route("/embed", post(embed))
            .with_state(kb.clone())
            .merge(
                Router::new()
                    .route("/sparql", get(sparql).post(sparql))
                    .with_state(sparql_app),
            );
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
        Ok(())
    });
    remove_info(&index_dir);
    stop.store(true, Ordering::Relaxed);
    let _ = worker.join();
    eprintln!("dqadb serve: stopped");
    served
}
