//! Unified open path for dense / CSR stores (single meta read).

use std::sync::Arc;

use crate::csr::{load_indptr, CsrMatrix};
use crate::dense::DenseMatrix;
use crate::error::{Error, Result};
use crate::io_util::read_meta;
use crate::limits::ReadLimits;
use crate::meta::{Kind, MetaBody};
use crate::storage::{ByteStore, StoreLocation, META_FILE_NAME};

/// Opened sc-compress matrix of either kind, produced by a single metadata pass.
#[derive(Clone)]
pub enum Matrix {
    Dense(DenseMatrix),
    Csr(CsrMatrix),
}

/// I/O performed while opening a matrix (meta + optional CSR indptr).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpenStats {
    pub io_bytes: u64,
    pub io_ops: u64,
}

/// Result of [`Matrix::open_with_stats`]: handle plus open-time I/O accounting.
#[derive(Clone)]
pub struct OpenedMatrix {
    pub matrix: Matrix,
    pub stats: OpenStats,
}

impl Matrix {
    pub fn open(location: impl Into<StoreLocation>) -> Result<Self> {
        Ok(Self::open_with_stats(location, ReadLimits::default())?.matrix)
    }

    pub fn open_with_limits(
        location: impl Into<StoreLocation>,
        limits: ReadLimits,
    ) -> Result<Self> {
        Ok(Self::open_with_stats(location, limits)?.matrix)
    }

    /// Open and report how many bytes/ops were spent on meta (and CSR indptr).
    pub fn open_with_stats(
        location: impl Into<StoreLocation>,
        limits: ReadLimits,
    ) -> Result<OpenedMatrix> {
        let store = location.into().open()?;
        Self::from_store_with_stats(store, limits)
    }

    pub fn from_store(store: Arc<dyn ByteStore>) -> Result<Self> {
        Ok(Self::from_store_with_stats(store, ReadLimits::default())?.matrix)
    }

    pub fn from_store_with_limits(store: Arc<dyn ByteStore>, limits: ReadLimits) -> Result<Self> {
        Ok(Self::from_store_with_stats(store, limits)?.matrix)
    }

    /// Single-pass open: one `meta.json` read, then CSR indptr if needed.
    pub fn from_store_with_stats(
        store: Arc<dyn ByteStore>,
        limits: ReadLimits,
    ) -> Result<OpenedMatrix> {
        let limits = limits.validate()?;
        let meta_len = store.len(META_FILE_NAME)?;
        let file = read_meta(store.as_ref(), limits)?;
        let mut stats = OpenStats {
            io_bytes: meta_len,
            io_ops: 1,
        };
        let matrix = match file.into_body() {
            MetaBody::Dense(meta) => Matrix::Dense(DenseMatrix::from_parts(store, meta, limits)),
            MetaBody::Csr(meta) => {
                let (indptr, encoded_len) = load_indptr(store.as_ref(), &meta, limits)?;
                stats.io_bytes = stats
                    .io_bytes
                    .checked_add(encoded_len)
                    .ok_or_else(|| Error::invalid_meta("open I/O byte count overflow"))?;
                stats.io_ops += 1;
                Matrix::Csr(CsrMatrix::from_parts(store, meta, indptr, limits))
            }
        };
        Ok(OpenedMatrix { matrix, stats })
    }

    pub fn kind(&self) -> Kind {
        match self {
            Self::Dense(_) => Kind::Dense,
            Self::Csr(_) => Kind::Csr,
        }
    }

    pub fn store(&self) -> &Arc<dyn ByteStore> {
        match self {
            Self::Dense(m) => m.store(),
            Self::Csr(m) => m.store(),
        }
    }

    pub fn shape(&self) -> [u64; 2] {
        match self {
            Self::Dense(m) => m.shape(),
            Self::Csr(m) => m.shape(),
        }
    }

    pub fn n_rows(&self) -> u64 {
        self.shape()[0]
    }

    pub fn n_cols(&self) -> u64 {
        self.shape()[1]
    }

    pub fn value_dtype(&self) -> crate::dtype::DType {
        match self {
            Self::Dense(m) => m.dtype(),
            Self::Csr(m) => m.value_dtype(),
        }
    }

    pub fn limits(&self) -> ReadLimits {
        match self {
            Self::Dense(m) => m.limits(),
            Self::Csr(m) => m.limits(),
        }
    }
}
