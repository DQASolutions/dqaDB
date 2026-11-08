use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::Result;

pub const CONFIG_FILE: &str = "dqadb.yaml";
pub const DEFAULT_BASE_IRI: &str = "urn:dqadb:";

/// Contents of `dqadb.yaml` at the root of a knowledge base.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub dqadb: u32,
    pub base_iri: String,
    #[serde(default = "default_ontology_main")]
    pub ontology_main: String,
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    #[serde(default = "default_language")]
    pub language_default: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub model: String,
    pub dim: u32,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: "bge-m3".into(),
            dim: 1024,
        }
    }
}

fn default_ontology_main() -> String {
    "ontology/schema.linkml.yaml".into()
}

fn default_language() -> String {
    "vi".into()
}

impl Config {
    pub fn new(base_iri: &str) -> Self {
        Self {
            dqadb: 1,
            base_iri: base_iri.into(),
            ontology_main: default_ontology_main(),
            embedding: EmbeddingConfig::default(),
            language_default: default_language(),
        }
    }

    pub fn load(root: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(root.join(CONFIG_FILE))?;
        Ok(serde_yaml::from_str(&text)?)
    }

    pub fn save(&self, root: &Path) -> Result<()> {
        std::fs::write(root.join(CONFIG_FILE), serde_yaml::to_string(self)?)?;
        Ok(())
    }
}
