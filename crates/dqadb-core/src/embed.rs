//! Embedding models (spec §14, decision 2026-11-09).
//!
//! Default: the official BAAI bge-m3 ONNX model (fp32) on the official Microsoft ONNX
//! Runtime, loaded dynamically (`libonnxruntime.so`, needs glibc 2.27). Dense embedding =
//! CLS token, L2-normalised. Embeddings are cached by content hash in
//! `emb-cache/<model>.sqlite`, so rebuilds never recompute them.

use rusqlite::{Connection, OptionalExtension, params};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use crate::config::EmbeddingConfig;
use crate::{Error, Result};

pub trait Embedder: Send + Sync {
    /// Identifies model + settings; part of every cache key and of the vector index config.
    fn id(&self) -> &str;
    fn dim(&self) -> usize;
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// Deterministic bag-of-words hashing embedder (`model: hash-test`): shared words give
/// similar vectors. For tests and demos only.
pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Embedder for HashEmbedder {
    fn id(&self) -> &str {
        "hash-test"
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0f32; self.dim];
                for w in crate::fts::fold(t)
                    .split(|c: char| !c.is_alphanumeric())
                    .filter(|w| !w.is_empty())
                {
                    let h = Sha256::digest(w.as_bytes());
                    v[u16::from_be_bytes([h[0], h[1]]) as usize % self.dim] += 1.0;
                }
                normalize(&mut v);
                v
            })
            .collect())
    }
}

/// bge-m3 (or any XLM-R style encoder exported to ONNX) on ONNX Runtime.
pub struct OrtEmbedder {
    id: String,
    dim: usize,
    session: Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
}

static ORT_ENV: OnceLock<std::result::Result<ort::environment::Environment, String>> =
    OnceLock::new();

fn ort_env(lib: &Path) -> Result<&'static ort::environment::Environment> {
    ORT_ENV
        .get_or_init(|| {
            ort::init_from(lib)
                .map_err(|e| e.to_string())
                .and_then(|b| b.with_name("dqadb").build().map_err(|e| e.to_string()))
        })
        .as_ref()
        .map_err(|e| {
            Error::other(format!(
                "cannot load ONNX Runtime from {}: {e}",
                lib.display()
            ))
        })
}

fn home_cache() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default()
        .join(".cache/dqadb")
}

/// Where the model files live, by precedence: env DQADB_MODEL_DIR, config, default cache.
pub fn model_dir(cfg: &EmbeddingConfig) -> PathBuf {
    std::env::var_os("DQADB_MODEL_DIR")
        .map(PathBuf::from)
        .or_else(|| cfg.model_dir.as_ref().map(PathBuf::from))
        .unwrap_or_else(|| home_cache().join("models").join(&cfg.model))
}

/// Where libonnxruntime.so lives, by precedence: env DQADB_ORT_LIB, config, default cache.
pub fn ort_lib(cfg: &EmbeddingConfig) -> PathBuf {
    std::env::var_os("DQADB_ORT_LIB")
        .map(PathBuf::from)
        .or_else(|| cfg.onnxruntime.as_ref().map(PathBuf::from))
        .unwrap_or_else(|| home_cache().join("onnxruntime/libonnxruntime.so"))
}

impl OrtEmbedder {
    pub fn load(cfg: &EmbeddingConfig) -> Result<Self> {
        let dir = model_dir(cfg);
        let model = dir.join("model.onnx");
        let tok = dir.join("tokenizer.json");
        for f in [&model, &tok] {
            if !f.exists() {
                return Err(Error::other(format!(
                    "embedding model file missing: {} (set embedding.model_dir or DQADB_MODEL_DIR; files: model.onnx, model.onnx_data, tokenizer.json)",
                    f.display()
                )));
            }
        }
        let lib = ort_lib(cfg);
        if !lib.exists() {
            return Err(Error::other(format!(
                "ONNX Runtime library missing: {} (set embedding.onnxruntime or DQADB_ORT_LIB to the official libonnxruntime.so)",
                lib.display()
            )));
        }
        let env = ort_env(&lib)?;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let e = |x: &dyn std::fmt::Display| Error::other(format!("ONNX Runtime: {x}"));
        let session = ort::session::Session::builder(env)
            .map_err(|x| e(&x))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|x| e(&x))?
            .with_intra_threads(threads)
            .map_err(|x| e(&x))?
            .commit_from_file(&model)
            .map_err(|x| e(&x))?;
        let mut tokenizer = tokenizers::Tokenizer::from_file(&tok).map_err(|x| e(&x))?;
        tokenizer.with_padding(Some(tokenizers::PaddingParams::default()));
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: 512,
                ..Default::default()
            }))
            .map_err(|x| e(&x))?;
        Ok(Self {
            id: format!("{}-onnx", cfg.model),
            dim: cfg.dim as usize,
            session: Mutex::new(session),
            tokenizer,
        })
    }
}

impl Embedder for OrtEmbedder {
    fn id(&self) -> &str {
        &self.id
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let e = |x: &dyn std::fmt::Display| Error::other(format!("embedding: {x}"));
        let enc = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|x| e(&x))?;
        let (b, l) = (enc.len(), enc[0].get_ids().len());
        let ids: Vec<i64> = enc
            .iter()
            .flat_map(|x| x.get_ids().iter().map(|v| *v as i64))
            .collect();
        let mask: Vec<i64> = enc
            .iter()
            .flat_map(|x| x.get_attention_mask().iter().map(|v| *v as i64))
            .collect();
        let ids = ort::value::Tensor::from_array(([b, l], ids)).map_err(|x| e(&x))?;
        let mask = ort::value::Tensor::from_array(([b, l], mask)).map_err(|x| e(&x))?;
        let mut session = self.session.lock().unwrap_or_else(|p| p.into_inner());
        let out = session.run(ort::inputs![ids, mask]).map_err(|x| e(&x))?;
        let (shape, data) = out[0].try_extract_tensor::<f32>().map_err(|x| e(&x))?;
        let dim = shape[shape.len() - 1] as usize;
        if dim != self.dim {
            return Err(Error::other(format!(
                "model returned {dim} dims, config says {}",
                self.dim
            )));
        }
        let per = data.len() / b;
        Ok((0..b)
            .map(|i| {
                let mut v = data[i * per..i * per + dim].to_vec(); // CLS token
                normalize(&mut v);
                v
            })
            .collect())
    }
}

/// Build the configured embedder.
pub fn load(cfg: &EmbeddingConfig) -> Result<Box<dyn Embedder>> {
    if cfg.model == "hash-test" {
        return Ok(Box::new(HashEmbedder::new(cfg.dim as usize)));
    }
    Ok(Box::new(OrtEmbedder::load(cfg)?))
}

// ---------- cache ----------

/// Content-addressed embedding cache: sha256(model id + text) -> f32 vector.
pub struct Cache {
    conn: Connection,
}

pub fn text_key(model_id: &str, text: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(model_id.as_bytes());
    h.update([0]);
    h.update(text.as_bytes());
    h.finalize().into()
}

impl Cache {
    pub fn open(dir: &Path, model_id: &str) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let conn = Connection::open(dir.join(format!("{model_id}.sqlite")))?;
        conn.busy_timeout(Duration::from_secs(30))?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
             CREATE TABLE IF NOT EXISTS emb(key BLOB PRIMARY KEY, vec BLOB NOT NULL) WITHOUT ROWID;",
        )?;
        Ok(Self { conn })
    }

    pub fn get(&self, key: &[u8; 32]) -> Result<Option<Vec<f32>>> {
        let blob: Option<Vec<u8>> = self
            .conn
            .prepare_cached("SELECT vec FROM emb WHERE key = ?1")?
            .query_row([&key[..]], |r| r.get(0))
            .optional()?;
        Ok(blob.map(|b| {
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        }))
    }

    pub fn put_many(&mut self, items: &[([u8; 32], Vec<f32>)]) -> Result<()> {
        let tx = self.conn.transaction()?;
        {
            let mut st = tx.prepare_cached("INSERT OR REPLACE INTO emb VALUES (?1, ?2)")?;
            for (k, v) in items {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                st.execute(params![&k[..], bytes])?;
            }
        }
        tx.commit()?;
        Ok(())
    }
}

/// Cache/index id of the configured model, known without loading it.
pub fn model_id(cfg: &EmbeddingConfig) -> String {
    if cfg.model == "hash-test" {
        "hash-test".into()
    } else {
        format!("{}-onnx", cfg.model)
    }
}

/// Loads the model on first use only: a fully cached run never pays the load cost.
pub struct LazyEmbedder {
    cfg: EmbeddingConfig,
    inner: OnceLock<Box<dyn Embedder>>,
    init: Mutex<()>,
}

impl LazyEmbedder {
    pub fn new(cfg: &EmbeddingConfig) -> Self {
        Self {
            cfg: cfg.clone(),
            inner: OnceLock::new(),
            init: Mutex::new(()),
        }
    }

    pub fn id(&self) -> String {
        model_id(&self.cfg)
    }

    pub fn loaded(&self) -> bool {
        self.inner.get().is_some()
    }

    /// The model, loaded once and shared; each `embed` call locks only for its batch.
    pub fn get(&self) -> Result<&dyn Embedder> {
        if let Some(e) = self.inner.get() {
            return Ok(e.as_ref());
        }
        let _guard = self.init.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(e) = self.inner.get() {
            return Ok(e.as_ref());
        }
        let _ = self.inner.set(load(&self.cfg)?);
        Ok(self.inner.get().expect("just set").as_ref())
    }
}

/// Embed `texts`, calling the model only for cache misses, in batches.
pub fn embed_cached(
    cache: &mut Cache,
    model: &LazyEmbedder,
    texts: &[String],
    batch: usize,
) -> Result<Vec<Vec<f32>>> {
    let id = model.id();
    let keys: Vec<[u8; 32]> = texts.iter().map(|t| text_key(&id, t)).collect();
    let mut out: Vec<Option<Vec<f32>>> =
        keys.iter().map(|k| cache.get(k)).collect::<Result<_>>()?;
    let missing: Vec<usize> = (0..texts.len()).filter(|i| out[*i].is_none()).collect();
    for chunk in missing.chunks(batch.max(1)) {
        let batch_texts: Vec<String> = chunk.iter().map(|i| texts[*i].clone()).collect();
        let vecs = model.get()?.embed(&batch_texts)?;
        cache.put_many(
            &chunk
                .iter()
                .zip(&vecs)
                .map(|(i, v)| (keys[*i], v.clone()))
                .collect::<Vec<_>>(),
        )?;
        for (i, v) in chunk.iter().zip(vecs) {
            out[*i] = Some(v);
        }
    }
    Ok(out.into_iter().map(|v| v.expect("filled")).collect())
}

/// f32 -> f16 bytes (index storage) and back.
pub fn to_f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| half::f16::from_f32(*x).to_le_bytes())
        .collect()
}

pub fn from_f16_bytes(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<2>()
        .0
        .iter()
        .map(|c| half::f16::from_le_bytes(*c).to_f32())
        .collect()
}

/// Cosine of `query` with an f16-encoded vector, decoding into a reused buffer
/// (bulk f16->f32 conversion uses the CPU's F16C instructions when available).
pub fn dot_f16(query: &[f32], blob: &[u8], bits: &mut Vec<u16>, buf: &mut Vec<f32>) -> f32 {
    use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};
    bits.clear();
    bits.extend(
        blob.as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c)),
    );
    buf.resize(bits.len(), 0.0);
    bits.reinterpret_cast::<half::f16>()
        .convert_to_f32_slice(buf);
    dot(query, buf)
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_embedder_is_deterministic_and_semantic_ish() {
        let e = HashEmbedder::new(64);
        let v = e
            .embed(&[
                "vận tải biển Hải Phòng".into(),
                "van tai bien".into(),
                "cà phê sữa".into(),
            ])
            .unwrap();
        assert!(dot(&v[0], &v[1]) > dot(&v[0], &v[2]));
        assert_eq!(
            v,
            e.embed(&[
                "vận tải biển Hải Phòng".into(),
                "van tai bien".into(),
                "cà phê sữa".into()
            ])
            .unwrap()
        );
        let rt = from_f16_bytes(&to_f16_bytes(&v[0]));
        assert!((dot(&v[0], &rt) - 1.0).abs() < 1e-3);
    }
}
