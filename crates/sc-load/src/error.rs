use std::collections::TryReserveError;
use std::sync::Arc;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Error)]
pub enum Error {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid dataset: {0}")]
    InvalidDataset(String),
    #[error("resource limit exceeded: {0}")]
    ResourceLimit(String),
    #[error("stale plan: {0}")]
    StalePlan(String),
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    #[error("I/O error ({kind:?}): {message}")]
    Io {
        kind: std::io::ErrorKind,
        message: String,
    },
    #[error("decode error: {0}")]
    Decode(String),
    #[error("type promotion not allowed: {0}")]
    Promote(String),
    #[error("conversion failed: {0}")]
    Conversion(String),
    #[error("session cancelled")]
    Cancelled,
    #[error("session failed: {0}")]
    Session(Arc<Error>),
    #[error("worker panicked")]
    WorkerPanic,
    #[error("allocation failed: {0}")]
    Allocation(String),
    #[error("internal invariant violated: {0}")]
    Invariant(String),
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io {
            kind: error.kind(),
            message: error.to_string(),
        }
    }
}

impl From<sc_compress::Error> for Error {
    fn from(error: sc_compress::Error) -> Self {
        Self::InvalidDataset(error.to_string())
    }
}

impl From<dyn_blosc::Error> for Error {
    fn from(error: dyn_blosc::Error) -> Self {
        Self::Decode(error.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self::InvalidDataset(error.to_string())
    }
}

impl From<TryReserveError> for Error {
    fn from(error: TryReserveError) -> Self {
        Self::Allocation(error.to_string())
    }
}
