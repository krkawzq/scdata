use std::io;

use thiserror::Error;

use crate::codecs::CodecError;

pub type AccessResult<T> = Result<T, AccessError>;

/// Errors produced by the access scheduler.
#[derive(Debug, Error)]
pub enum AccessError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("codec error: {0}")]
    Codec(#[from] CodecError),

    #[error("access scheduler is shut down")]
    Shutdown,

    #[error("memory budget exhausted")]
    OutOfMemory,

    #[error("request queue is full at capacity {capacity}")]
    QueueFull { capacity: usize },

    #[error("invalid slice spec: {0}")]
    InvalidSlice(String),

    #[error("access CPU worker panicked")]
    CpuWorkerPanic,
}
