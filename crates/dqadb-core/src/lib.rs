//! dqaDB core: event files as the source of truth, a rebuildable RDF index on top.
//! See the spec v0.1 for the design; this crate implements milestone M1.

pub mod config;
pub mod error;
pub mod event;
pub mod kb;
pub mod lock;
pub mod ontology;
pub mod rdf;
pub mod search;
pub mod sparql;
pub mod store;

pub use error::{Error, Result};
pub use kb::{Kb, WriteOpts};
