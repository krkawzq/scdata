use std::collections::TryReserveError;
use std::io;
use std::path::PathBuf;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    DynBlosc(#[from] dyn_blosc::Error),

    #[error(transparent)]
    Zip(#[from] zip::result::ZipError),

    #[error("allocation failed: {0}")]
    Allocation(#[from] TryReserveError),

    #[error("key not found: {key}")]
    NotFound { key: String },

    #[error("{0}")]
    InvalidArgument(String),

    #[error("{0}")]
    InvalidMeta(String),

    #[error("corrupt {context}: {message}")]
    CorruptData { context: String, message: String },

    #[error("path error for {path}: {message}")]
    Path { path: PathBuf, message: String },
}

impl Error {
    pub(crate) fn invalid_argument(message: impl Into<String>) -> Self {
        Self::InvalidArgument(message.into())
    }

    pub(crate) fn invalid_meta(message: impl Into<String>) -> Self {
        Self::InvalidMeta(message.into())
    }

    pub(crate) fn not_found(key: impl Into<String>) -> Self {
        Self::NotFound { key: key.into() }
    }

    pub(crate) fn corrupt(context: impl Into<String>, message: impl Into<String>) -> Self {
        Self::CorruptData {
            context: context.into(),
            message: message.into(),
        }
    }
}
