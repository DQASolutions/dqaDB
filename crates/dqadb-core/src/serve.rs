//! Client side of `dqadb serve` (spec §13/§14): a resident process that keeps the
//! embedding model loaded. CLI commands find it through `.dqadb/serve.json` and ask it
//! to embed query texts, instead of loading the model (seconds, GBs) on every call.
//! Plain HTTP/1.1 on 127.0.0.1, std only.

use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::{Error, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServeInfo {
    pub pid: u32,
    pub port: u16,
    pub model: String,
    pub started: String,
}

pub fn info_path(index_dir: &Path) -> PathBuf {
    index_dir.join("serve.json")
}

pub fn read_info(index_dir: &Path) -> Option<ServeInfo> {
    serde_json::from_slice(&std::fs::read(info_path(index_dir)).ok()?).ok()
}

pub fn write_info(index_dir: &Path, info: &ServeInfo) -> Result<()> {
    std::fs::create_dir_all(index_dir)?;
    let tmp = index_dir.join("serve.json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(info)?)?;
    std::fs::rename(tmp, info_path(index_dir))?;
    Ok(())
}

pub fn remove_info(index_dir: &Path) {
    let _ = std::fs::remove_file(info_path(index_dir));
}

/// POST a JSON body to the local server; `None` when no server answers (caller falls back).
pub fn post(
    port: u16,
    path: &str,
    body: &serde_json::Value,
    timeout: Duration,
) -> Option<serde_json::Value> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let mut s = TcpStream::connect_timeout(&addr, Duration::from_millis(300)).ok()?;
    s.set_read_timeout(Some(timeout)).ok()?;
    s.set_write_timeout(Some(timeout)).ok()?;
    let payload = serde_json::to_vec(body).ok()?;
    let head = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    );
    s.write_all(head.as_bytes()).ok()?;
    s.write_all(&payload).ok()?;
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).ok()?;
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n")?;
    let status = std::str::from_utf8(&raw[..split])
        .ok()?
        .split_whitespace()
        .nth(1)?
        .to_string();
    if status != "200" {
        return None;
    }
    serde_json::from_slice(&raw[split + 4..]).ok()
}

/// Embed texts through a running server for this KB and model, if there is one.
pub fn embed_remote(
    index_dir: &Path,
    model_id: &str,
    texts: &[String],
) -> Option<Result<Vec<Vec<f32>>>> {
    let info = read_info(index_dir)?;
    if info.model != model_id {
        return None;
    }
    let reply = post(
        info.port,
        "/embed",
        &serde_json::json!({ "texts": texts }),
        Duration::from_secs(120),
    )?;
    let vectors: Option<Vec<Vec<f32>>> = serde_json::from_value(reply.get("vectors")?.clone()).ok();
    Some(vectors.ok_or_else(|| Error::other("dqadb serve returned a malformed embedding")))
}
