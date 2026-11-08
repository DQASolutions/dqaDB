use std::fmt;
use std::path::PathBuf;

/// Errors carry a stable process exit code (see spec §12) so agents can branch on them.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Validation(ValidationReport),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("ontology has not been built yet. Run: dqadb ontology build")]
    OntologyMissing,
    #[error("ontology YAML changed since the last build. Run: dqadb ontology build")]
    OntologyStale,
    #[error("index is stale ({0}). Run: dqadb rebuild")]
    IndexStale(String),
    #[error("version conflict: watermark is {actual}, --expected-version was {expected}")]
    VersionConflict { expected: String, actual: String },
    #[error("timed out waiting for the write lock {0}")]
    LockTimeout(PathBuf),
    #[error("not a dqaDB knowledge base (no dqadb.yaml found from {0})")]
    NotAKb(PathBuf),
    #[error("{0}")]
    Other(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("yaml: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("store: {0}")]
    Store(#[from] rusqlite::Error),
}

impl Error {
    pub fn exit_code(&self) -> i32 {
        match self {
            Error::Validation(_) => 2,
            Error::NotFound(_) => 3,
            Error::OntologyMissing | Error::OntologyStale => 4,
            Error::IndexStale(_) => 5,
            Error::VersionConflict { .. } | Error::LockTimeout(_) => 6,
            _ => 1,
        }
    }

    pub fn other(msg: impl Into<String>) -> Self {
        Error::Other(msg.into())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// One problem found while validating input ops. `fix` tells an agent how to correct it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Issue {
    pub at: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fix: Option<String>,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ValidationReport {
    pub issues: Vec<Issue>,
}

impl ValidationReport {
    pub fn push(&mut self, at: impl Into<String>, message: impl Into<String>, fix: Option<String>) {
        self.issues.push(Issue {
            at: at.into(),
            message: message.into(),
            fix,
        });
    }

    pub fn is_empty(&self) -> bool {
        self.issues.is_empty()
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "validation failed: {} issue(s)", self.issues.len())?;
        for issue in &self.issues {
            writeln!(f, "  at {}: {}", issue.at, issue.message)?;
            if let Some(fix) = &issue.fix {
                writeln!(f, "    fix: {fix}")?;
            }
        }
        Ok(())
    }
}
