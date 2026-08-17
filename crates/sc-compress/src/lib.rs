//! Chunked compression for row-major dense and CSR single-cell matrices.
//!
//! ```text
//! # dense — Blosc1 with fixed_cells chunks and blocks
//! store/
//!   meta.json
//!   data/0  data/1  ...
//!
//! # csr — DynBlosc with variable cell-aligned blocks
//! store/
//!   meta.json
//!   indptr
//!   indices/0 1 2 ...
//!   data/0 1 2 ...
//! ```
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod access;
mod array;
mod codec;
mod csr;
mod dense;
mod dtype;
mod error;
mod io_util;
mod kernel;
mod limits;
mod matrix;
mod meta;
mod numeric;
mod parallel;
mod partition;
mod range_decode;
mod select;
pub mod storage;

pub use array::{CsrArray, DenseArray, SelectedArray};
pub use codec::{BloscCodec, BloscOptions, Compressor, ShuffleMode};
pub use csr::{open_csr, open_csr_with_limits, CsrMatrix, CsrWriter};
pub use dense::{open_dense, open_dense_with_limits, DenseMatrix, DenseWriter};
pub use dtype::DType;
pub use error::{Error, Result};
pub use kernel::{build_col_map, CsrColMap};
pub use limits::ReadLimits;
pub use matrix::{Matrix, OpenStats, OpenedMatrix};
pub use meta::{
    ArrayMeta, ChunkGridMeta, CsrMeta, DenseMeta, Kind, MetaFile, PartitionMeta, FORMAT_NAME,
    FORMAT_VERSION,
};
pub use numeric::{IntegerIndex, MatrixValue};
pub use parallel::default_threads;
pub use partition::{Partition, DEFAULT_BLOCK_BUDGET, DEFAULT_CHUNK_BUDGET};
pub use select::{AxisIndex, CsrOutput, NormalizedAxis, NormalizedSelection, Selection};
pub use storage::{
    chunk_key, ByteStore, ByteStoreMut, DirectoryStore, PositionedValue, StoreLocation, ZipStore,
};
