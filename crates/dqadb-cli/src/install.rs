//! `dqadb model install`: fetch the official embedding model and runtime, verified.
//! - model: BAAI/bge-m3 ONNX export from Hugging Face (sha256 from the Hub's LFS metadata)
//! - runtime: Microsoft's official ONNX Runtime shared library from its PyPI wheel
//!   (manylinux, glibc 2.27+; sha256 from PyPI)
//!
//! Installs into ~/.cache/dqadb/ (the default embedding paths), so no config is needed.

use dqadb_core::{Error, Result};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

const HF_REPO: &str = "BAAI/bge-m3";
const HF_FILES: [(&str, &str); 3] = [
    ("onnx/model.onnx", "model.onnx"),
    ("onnx/model.onnx_data", "model.onnx_data"),
    ("onnx/tokenizer.json", "tokenizer.json"),
];
pub const ORT_VERSION: &str = "1.23.2";

fn http_err(e: impl std::fmt::Display) -> Error {
    Error::other(format!("download failed: {e}"))
}

/// HTTP agent with timeouts, so a stalled network fails (and is retried) instead of hanging.
fn agent(body_timeout: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(20)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .timeout_recv_body(Some(body_timeout))
        .http_status_as_error(true)
        .build()
        .into()
}

fn get_json(url: &str) -> Result<serde_json::Value> {
    agent(Duration::from_secs(120))
        .get(url)
        .call()
        .map_err(http_err)?
        .into_body()
        .into_with_config()
        .limit(64 * 1024 * 1024)
        .read_to_string()
        .map_err(http_err)
        .and_then(|t| serde_json::from_str(&t).map_err(http_err))
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex::encode(h.finalize()))
}

/// Download `url` to `dest` unless it already exists with the expected hash. Writes to
/// `<dest>.part`, resumes it with HTTP Range after a failure (up to 5 attempts), and
/// renames only after the sha256 matches, so an interrupted download never leaves a bad file.
fn fetch(url: &str, dest: &Path, sha256: &str, size: u64) -> Result<()> {
    if dest.exists() && dest.metadata()?.len() == size && sha256_file(dest)? == sha256 {
        eprintln!(
            "  ok (already present, sha256 verified): {}",
            dest.display()
        );
        return Ok(());
    }
    eprintln!("  downloading {} ({:.1} MB)", url, size as f64 / 1e6);
    let tmp = dest.with_extension("part");
    // allow at least ~256 KB/s for the whole body, plus slack
    let body_timeout = Duration::from_secs(120 + size / (256 * 1024));
    let mut last_err = None;
    for attempt in 1..=5 {
        let have = tmp.metadata().map(|m| m.len()).unwrap_or(0);
        if have >= size && size > 0 {
            break;
        }
        let mut req = agent(body_timeout).get(url);
        if have > 0 {
            req = req.header("Range", &format!("bytes={have}-"));
        }
        let result = (|| -> Result<()> {
            let resp = req.call().map_err(http_err)?;
            let resumed = resp.status().as_u16() == 206;
            let mut out = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .append(resumed)
                .truncate(!resumed)
                .open(&tmp)?;
            let mut reader = resp.into_body().into_with_config().limit(u64::MAX).reader();
            let mut buf = vec![0u8; 1 << 20];
            let (mut done, mut shown) = (if resumed { have } else { 0 }, 0u64);
            loop {
                let n = reader.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                out.write_all(&buf[..n])?;
                done += n as u64;
                if size > 50_000_000 && done.saturating_sub(shown) > size / 20 {
                    shown = done;
                    eprintln!("    {:>3}%", done * 100 / size.max(1));
                }
            }
            out.sync_all()?;
            Ok(())
        })();
        match result {
            Ok(()) => {
                last_err = None;
                break;
            }
            Err(e) => {
                eprintln!("    attempt {attempt}/5 failed: {e}");
                last_err = Some(e);
                std::thread::sleep(Duration::from_secs(2 * attempt));
            }
        }
    }
    if let Some(e) = last_err {
        return Err(e);
    }
    let got = sha256_file(&tmp)?;
    if got != sha256 {
        let _ = std::fs::remove_file(&tmp);
        return Err(Error::other(format!(
            "checksum mismatch for {url}: expected {sha256}, got {got}"
        )));
    }
    std::fs::rename(&tmp, dest)?;
    eprintln!("  ok (sha256 verified): {}", dest.display());
    Ok(())
}

pub fn cache_root() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default()
        .join(".cache/dqadb")
}

pub fn install_model(model_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(model_dir)?;
    let meta = get_json(&format!(
        "https://huggingface.co/api/models/{HF_REPO}?blobs=true"
    ))?;
    let siblings = meta["siblings"]
        .as_array()
        .ok_or_else(|| Error::other("unexpected Hugging Face API reply"))?;
    eprintln!("model {HF_REPO} -> {}", model_dir.display());
    for (remote, local) in HF_FILES {
        let entry = siblings
            .iter()
            .find(|s| s["rfilename"] == remote)
            .ok_or_else(|| Error::other(format!("{remote} not found in {HF_REPO}")))?;
        let sha = entry["lfs"]["sha256"]
            .as_str()
            .ok_or_else(|| Error::other(format!("no sha256 published for {remote}")))?;
        let size = entry["lfs"]["size"].as_u64().unwrap_or(0);
        let url = format!("https://huggingface.co/{HF_REPO}/resolve/main/{remote}");
        fetch(&url, &model_dir.join(local), sha, size)?;
    }
    Ok(())
}

pub fn install_runtime(dest_dir: &Path) -> Result<PathBuf> {
    if !cfg!(target_os = "linux") {
        return Err(Error::other(
            "automatic ONNX Runtime install supports Linux; set embedding.onnxruntime to your libonnxruntime",
        ));
    }
    let arch = if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        "x86_64"
    };
    std::fs::create_dir_all(dest_dir)?;
    // already installed and untouched? (version + sha256 of the extracted library)
    let record = dest_dir.join("installed.json");
    let lib_path = dest_dir.join("libonnxruntime.so");
    if let Ok(prev) =
        std::fs::read(&record).map(|b| serde_json::from_slice::<serde_json::Value>(&b))
        && let Ok(prev) = prev
        && prev["version"] == ORT_VERSION
        && lib_path.exists()
        && prev["lib_sha256"].as_str() == Some(sha256_file(&lib_path)?.as_str())
    {
        eprintln!(
            "ONNX Runtime {ORT_VERSION}: ok (already installed, sha256 verified): {}",
            lib_path.display()
        );
        return Ok(lib_path);
    }
    let meta = get_json(&format!(
        "https://pypi.org/pypi/onnxruntime/{ORT_VERSION}/json"
    ))?;
    let file = meta["urls"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|f| {
            let n = f["filename"].as_str().unwrap_or("");
            n.contains("manylinux") && n.contains(arch) && n.ends_with(".whl")
        })
        .ok_or_else(|| {
            Error::other(format!(
                "no manylinux {arch} wheel for onnxruntime {ORT_VERSION}"
            ))
        })?;
    let url = file["url"].as_str().unwrap_or_default();
    let sha = file["digests"]["sha256"].as_str().unwrap_or_default();
    let size = file["size"].as_u64().unwrap_or(0);
    eprintln!(
        "ONNX Runtime {ORT_VERSION} (official Microsoft build) -> {}",
        dest_dir.display()
    );
    let wheel = dest_dir.join(format!("onnxruntime-{ORT_VERSION}.whl"));
    fetch(url, &wheel, sha, size)?;
    let mut zip = zip::ZipArchive::new(std::fs::File::open(&wheel)?)
        .map_err(|e| Error::other(format!("bad wheel: {e}")))?;
    let mut lib = None;
    for i in 0..zip.len() {
        let mut entry = zip
            .by_index(i)
            .map_err(|e| Error::other(format!("bad wheel: {e}")))?;
        let name = entry
            .name()
            .map_err(|e| Error::other(format!("bad wheel: {e}")))?
            .to_string();
        let base = name.rsplit('/').next().unwrap_or_default().to_string();
        if name.contains("/capi/") && base.starts_with("libonnxruntime") && base.contains(".so") {
            let target = if base.starts_with("libonnxruntime.so") {
                "libonnxruntime.so".to_string()
            } else {
                base.clone()
            };
            let path = dest_dir.join(&target);
            let mut out = std::fs::File::create(&path)?;
            std::io::copy(&mut entry, &mut out)?;
            if target == "libonnxruntime.so" {
                lib = Some(path);
            }
        }
    }
    let _ = std::fs::remove_file(&wheel);
    let lib = lib.ok_or_else(|| Error::other("libonnxruntime.so not found in the wheel"))?;
    let info = serde_json::json!({ "version": ORT_VERSION, "wheel_sha256": sha, "lib_sha256": sha256_file(&lib)? });
    std::fs::write(&record, serde_json::to_vec_pretty(&info)?)?;
    eprintln!("  ok: {}", lib.display());
    Ok(lib)
}
