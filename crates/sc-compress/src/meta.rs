use serde::{Deserialize, Serialize};

use crate::codec::Compressor;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::partition::{dense_blosc1_block_size, Partition};
use crate::storage::{validate_key, META_FILE_NAME};

pub const FORMAT_NAME: &str = "sc-compress";
pub const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Dense,
    Csr,
}

/// One on-disk array: path, element dtype, and compressor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArrayMeta {
    /// Relative path: directory for chunked arrays, or a single file name.
    pub path: String,
    pub dtype: DType,
    pub compressor: Compressor,
}

impl ArrayMeta {
    pub fn new(path: impl Into<String>, dtype: DType, compressor: Compressor) -> Self {
        Self {
            path: path.into(),
            dtype,
            compressor,
        }
    }

    pub(crate) fn validate(&self, kind: ArrayCompressorKind) -> Result<()> {
        validate_key(&self.path)
            .map_err(|error| Error::invalid_meta(format!("invalid array path: {error}")))?;
        if paths_overlap(&self.path, META_FILE_NAME) {
            return Err(Error::invalid_meta(
                "array path must not overwrite meta.json",
            ));
        }
        self.compressor
            .validate()
            .map_err(|error| Error::invalid_meta(error.to_string()))?;
        match kind {
            ArrayCompressorKind::Any => {}
            ArrayCompressorKind::Blosc1 if !self.compressor.is_blosc1() => {
                return Err(Error::invalid_meta(format!(
                    "array `{}` requires blosc1 compressor, got `{}`",
                    self.path,
                    self.compressor.id()
                )));
            }
            ArrayCompressorKind::DynBlosc if !self.compressor.is_dyn_blosc() => {
                return Err(Error::invalid_meta(format!(
                    "array `{}` requires dyn-blosc compressor, got `{}`",
                    self.path,
                    self.compressor.id()
                )));
            }
            ArrayCompressorKind::Blosc1 | ArrayCompressorKind::DynBlosc => {}
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArrayCompressorKind {
    Any,
    Blosc1,
    DynBlosc,
}

/// Zarr-style 1-D chunk grid: file `i` starts at logical cell offset `offsets[i]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkGridMeta {
    pub offsets: Vec<u64>,
}

impl ChunkGridMeta {
    pub fn from_cell_starts(starts: impl IntoIterator<Item = u64>) -> Self {
        Self {
            offsets: starts.into_iter().collect(),
        }
    }

    pub fn validate(&self, n_cells: u64) -> Result<()> {
        if self.offsets.is_empty() {
            if n_cells == 0 {
                return Ok(());
            }
            return Err(Error::invalid_meta(
                "chunk offsets empty but n_cells is non-zero",
            ));
        }
        if self.offsets[0] != 0 {
            return Err(Error::invalid_meta("chunk offsets must start at 0"));
        }
        for window in self.offsets.windows(2) {
            if window[1] <= window[0] {
                return Err(Error::invalid_meta(
                    "chunk offsets must be strictly increasing",
                ));
            }
            if window[1] >= n_cells {
                return Err(Error::invalid_meta(format!(
                    "chunk offset {} is outside 0..{n_cells}",
                    window[1]
                )));
            }
        }
        if n_cells > 0 && self.offsets.last().copied().unwrap_or(n_cells) >= n_cells {
            return Err(Error::invalid_meta("last chunk offset must be < n_cells"));
        }
        Ok(())
    }

    pub fn n_chunks(&self) -> usize {
        self.offsets.len()
    }

    pub(crate) fn overlapping_chunks(&self, start: u64, end: u64) -> std::ops::Range<usize> {
        if start == end || self.offsets.is_empty() {
            return 0..0;
        }
        let first = self
            .offsets
            .partition_point(|offset| *offset <= start)
            .saturating_sub(1);
        let past_last = self.offsets.partition_point(|offset| *offset < end);
        first..past_last
    }

    /// Chunk file id that contains logical cell `row` (`O(log n)` binary search).
    ///
    /// Requires a validated non-empty grid (`offsets[0] == 0`). The caller must
    /// ensure `row < n_cells`; out-of-range rows still map to the last chunk.
    pub fn chunk_of(&self, row: u64) -> Result<usize> {
        if self.offsets.is_empty() {
            return Err(Error::invalid_argument(
                "chunk grid is empty (no chunk files)",
            ));
        }
        Ok(self
            .offsets
            .partition_point(|offset| *offset <= row)
            .saturating_sub(1))
    }

    /// Half-open cell range covered by chunk file `id`.
    pub fn cell_range(&self, id: usize, n_cells: u64) -> Result<(u64, u64)> {
        let start = self.offsets.get(id).copied().ok_or_else(|| {
            Error::invalid_argument(format!(
                "chunk id {id} out of range ({} chunks)",
                self.offsets.len()
            ))
        })?;
        let end = self.offsets.get(id + 1).copied().unwrap_or(n_cells);
        if start > end || end > n_cells {
            return Err(Error::invalid_meta(format!(
                "invalid cell range for chunk {id}: [{start}, {end}) with n_cells {n_cells}"
            )));
        }
        Ok((start, end))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartitionMeta {
    pub chunk: Partition,
    pub block: Partition,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenseMeta {
    pub shape: [u64; 2],
    pub partition: PartitionMeta,
    /// Chunked array under `data/0`, `data/1`, ...
    pub data: ArrayMeta,
    pub chunks: ChunkGridMeta,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CsrMeta {
    pub shape: [u64; 2],
    pub nnz: u64,
    pub partition: PartitionMeta,
    /// Single compressed file (default path `indptr`).
    pub indptr: ArrayMeta,
    /// Chunked array under `indices/0`, ...
    pub indices: ArrayMeta,
    /// Chunked array under `data/0`, ...
    pub data: ArrayMeta,
    pub chunks: ChunkGridMeta,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum MetaBody {
    Dense(DenseMeta),
    Csr(CsrMeta),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetaFile {
    format: String,
    version: u32,
    #[serde(flatten)]
    body: MetaBody,
}

impl MetaFile {
    pub fn dense(meta: DenseMeta) -> Self {
        Self {
            format: FORMAT_NAME.into(),
            version: FORMAT_VERSION,
            body: MetaBody::Dense(meta),
        }
    }

    pub fn csr(meta: CsrMeta) -> Self {
        Self {
            format: FORMAT_NAME.into(),
            version: FORMAT_VERSION,
            body: MetaBody::Csr(meta),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.format != FORMAT_NAME {
            return Err(Error::invalid_meta(format!(
                "unexpected format `{}`",
                self.format
            )));
        }
        if self.version != FORMAT_VERSION {
            return Err(Error::invalid_meta(format!(
                "unsupported format version {}",
                self.version
            )));
        }
        match &self.body {
            MetaBody::Dense(meta) => {
                meta.data.validate(ArrayCompressorKind::Blosc1)?;
                validate_partition(&meta.partition)?;
                if !meta.partition.chunk.is_fixed_cells() {
                    return Err(Error::invalid_meta(
                        "dense chunk partition must be fixed_cells",
                    ));
                }
                if !meta.partition.block.is_fixed_cells() {
                    return Err(Error::invalid_meta(
                        "dense block partition must be fixed_cells",
                    ));
                }
                let block_size = meta
                    .data
                    .compressor
                    .blosc1_block_size()
                    .ok_or_else(|| Error::invalid_meta("dense compressor is not blosc1"))?;
                let Some(block_cells) = meta.partition.block.fixed_cells_n() else {
                    return Err(Error::invalid_meta(
                        "dense block partition must be fixed_cells",
                    ));
                };
                if meta.shape[1] > 0 {
                    let expected =
                        dense_blosc1_block_size(block_cells, meta.shape[1], meta.data.dtype.size())
                            .map_err(|error| Error::invalid_meta(error.to_string()))?;
                    if block_size != expected {
                        return Err(Error::invalid_meta(format!(
                            "dense blosc1 block_size {block_size} must equal fixed_cells block × n_genes × dtype = {expected}"
                        )));
                    }
                }
                meta.chunks.validate(meta.shape[0])?;
            }
            MetaBody::Csr(meta) => {
                meta.indptr.validate(ArrayCompressorKind::Any)?;
                meta.indices.validate(ArrayCompressorKind::DynBlosc)?;
                meta.data.validate(ArrayCompressorKind::DynBlosc)?;
                if meta.indptr.dtype != DType::U64 {
                    return Err(Error::invalid_meta("csr indptr dtype must be u64"));
                }
                if !meta.indices.dtype.is_csr_index() {
                    return Err(Error::invalid_meta("csr indices dtype must be u16 or u32"));
                }
                if meta.shape[1] == 0 && meta.nnz != 0 {
                    return Err(Error::invalid_meta(
                        "csr matrix with zero columns must have nnz = 0",
                    ));
                }
                if paths_overlap(&meta.indptr.path, &meta.indices.path)
                    || paths_overlap(&meta.indptr.path, &meta.data.path)
                    || paths_overlap(&meta.indices.path, &meta.data.path)
                {
                    return Err(Error::invalid_meta(
                        "csr array paths must be distinct and non-overlapping",
                    ));
                }
                validate_partition(&meta.partition)?;
                meta.chunks.validate(meta.shape[0])?;
            }
        }
        Ok(())
    }

    pub fn kind(&self) -> Kind {
        match &self.body {
            MetaBody::Dense(_) => Kind::Dense,
            MetaBody::Csr(_) => Kind::Csr,
        }
    }

    pub fn format(&self) -> &str {
        &self.format
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn as_dense(&self) -> Option<&DenseMeta> {
        match &self.body {
            MetaBody::Dense(meta) => Some(meta),
            MetaBody::Csr(_) => None,
        }
    }

    pub fn as_csr(&self) -> Option<&CsrMeta> {
        match &self.body {
            MetaBody::Csr(meta) => Some(meta),
            MetaBody::Dense(_) => None,
        }
    }

    pub(crate) fn into_body(self) -> MetaBody {
        self.body
    }
}

fn validate_partition(partition: &PartitionMeta) -> Result<()> {
    partition
        .chunk
        .validate()
        .map_err(|error| Error::invalid_meta(error.to_string()))?;
    partition
        .block
        .validate()
        .map_err(|error| Error::invalid_meta(error.to_string()))
}

fn paths_overlap(left: &str, right: &str) -> bool {
    left == right
        || left
            .strip_prefix(right)
            .is_some_and(|suffix| suffix.starts_with('/'))
        || right
            .strip_prefix(left)
            .is_some_and(|suffix| suffix.starts_with('/'))
}
