use std::io;

use thiserror::Error;

use crate::access::AccessError;
use crate::codecs::CodecError;

use super::array::DType;
use super::registry::DatasetId;

pub type DataBankResult<T> = Result<T, DataBankError>;

#[derive(Debug, Error)]
pub enum DataBankError {
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    #[error("invalid dataset id: {0:?}")]
    InvalidDatasetId(DatasetId),

    #[error("dataset is unloaded: {0:?}")]
    DatasetUnloaded(DatasetId),

    #[error("invalid array metadata: {0}")]
    InvalidArrayMeta(String),

    #[error("unsupported dtype {dtype:?} for {context}")]
    UnsupportedDType { dtype: DType, context: &'static str },

    #[error("cannot cast {src:?} to {dst:?}: {reason}")]
    CannotCast {
        src: DType,
        dst: DType,
        reason: &'static str,
    },

    #[error("cell index {cell} is out of range for {num_cells} cells")]
    CellIndexOutOfRange { cell: usize, num_cells: usize },

    #[error("gene index {gene} is out of range for {num_genes} genes")]
    GeneIndexOutOfRange { gene: usize, num_genes: usize },

    #[error("gene name is not present in dataset: {gene}")]
    GeneNameNotFound { gene: String },

    #[error("duplicate requested gene name: {gene}")]
    DuplicateGeneName { gene: String },

    #[error("buffer size mismatch: expected {expected}, actual {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },

    #[error("name buffer size mismatch: expected {expected}, actual {actual}")]
    NameBufferSizeMismatch { expected: usize, actual: usize },

    #[error("invalid CSR indptr: {0}")]
    IndptrInvalid(String),

    #[error("invalid CSR index: {0}")]
    CsrIndexInvalid(String),

    #[error("access error: {0}")]
    Access(#[from] AccessError),

    #[error("codec error: {0}")]
    Codec(#[from] CodecError),

    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("databank compute worker panicked")]
    ComputeWorkerPanic,

    #[error("databank compute pool is shut down")]
    ComputeShutdown,

    #[error("databank prefetch was cancelled")]
    PrefetchCancelled,

    #[error("databank prefetch producer panicked")]
    PrefetchProducerPanic,

    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
}
