use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::access::{AccessItem, ChunkKey, FileRef};
use crate::codecs::{
    codec_from_json_str, codec_from_spec, codec_pipeline_from_specs,
    codec_pipeline_from_zarr_v2_json_str, codec_pipeline_from_zarr_v2_specs,
    codec_specs_from_json_str, codec_specs_from_json_value, CodecSpec, SharedCodec,
};
use crate::iopool::{FileId, IoPool};

use super::error::{DataBankError, DataBankResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16,
    BF16,
    F32,
    F64,
}

impl DType {
    pub fn item_size(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    pub fn is_csr_index(self) -> bool {
        matches!(self, Self::I32 | Self::U32 | Self::I64 | Self::U64)
    }
}

pub trait DataValue: sealed::Sealed + Copy + Send + Sync + 'static {
    const DTYPE: DType;
    fn zero() -> Self;
}

mod sealed {
    pub trait Sealed {}
}

macro_rules! impl_data_value {
    ($ty:ty, $dtype:expr, $zero:expr) => {
        impl sealed::Sealed for $ty {}

        impl DataValue for $ty {
            const DTYPE: DType = $dtype;

            fn zero() -> Self {
                $zero
            }
        }
    };
}

impl_data_value!(u8, DType::U8, 0);
impl_data_value!(i8, DType::I8, 0);
impl_data_value!(u16, DType::U16, 0);
impl_data_value!(i16, DType::I16, 0);
impl_data_value!(u32, DType::U32, 0);
impl_data_value!(i32, DType::I32, 0);
impl_data_value!(u64, DType::U64, 0);
impl_data_value!(i64, DType::I64, 0);
impl_data_value!(f32, DType::F32, 0.0);
impl_data_value!(f64, DType::F64, 0.0);

/// Opaque native-endian half-precision payload.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct F16Bits(pub u16);

impl sealed::Sealed for F16Bits {}

impl DataValue for F16Bits {
    const DTYPE: DType = DType::F16;

    fn zero() -> Self {
        Self(0)
    }
}

/// Opaque native-endian bfloat16 payload.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Bf16Bits(pub u16);

impl sealed::Sealed for Bf16Bits {}

impl DataValue for Bf16Bits {
    const DTYPE: DType = DType::BF16;

    fn zero() -> Self {
        Self(0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayOrder {
    C,
}

#[derive(Debug, Clone)]
pub struct ArrayMeta {
    /// Logical array shape.
    pub shape: Vec<usize>,
    /// Logical chunk shape used to compute the chunk grid.
    ///
    /// Physical edge chunks are stored cropped to their logical extent. For
    /// example, a `[5, 5]` array with `[2, 3]` chunks stores the last column
    /// chunk with width `2`, not with fill-value padding up to width `3`.
    /// The `ZarrV2*` codec metadata variants below describe codec pipelines
    /// only; this physical chunk layout is not standard Zarr's padded edge
    /// chunk layout.
    pub chunk_shape: Vec<usize>,
    /// Number of chunks on each axis. Must equal `ceil(shape / chunk_shape)`.
    pub chunk_grid_shape: Vec<usize>,
    pub dtype: DType,
    pub order: ArrayOrder,
    pub codec: ArrayCodecMeta,
    /// Encoded chunk storage. Entries are ordered by logical chunk index in
    /// C-order (row-major) traversal of `chunk_grid_shape`.
    pub chunks: ChunkStoreMeta,
}

#[derive(Debug, Clone, Default)]
pub enum ArrayCodecMeta {
    #[default]
    Uncompressed,
    CodecJson(String),
    CodecJsonValue(serde_json::Value),
    PipelineJson(String),
    PipelineJsonValue(serde_json::Value),
    /// Zarr v2 filter/compressor JSON converted into this crate's codec
    /// pipeline. This does not imply standard Zarr padded edge chunk storage.
    ZarrV2Json {
        filters: Option<String>,
        compressor: Option<String>,
    },
    /// Zarr v2 filter/compressor JSON values converted into this crate's codec
    /// pipeline. This does not imply standard Zarr padded edge chunk storage.
    ZarrV2JsonValue {
        filters: Option<serde_json::Value>,
        compressor: Option<serde_json::Value>,
    },
}

impl ArrayCodecMeta {
    fn build(&self) -> DataBankResult<SharedCodec> {
        match self {
            Self::Uncompressed => Ok(codec_from_spec(&CodecSpec::None)),
            Self::CodecJson(json) => Ok(codec_from_json_str(json)?),
            Self::CodecJsonValue(value) => Ok(CodecSpec::from_json_value(value)?.build()),
            Self::PipelineJson(json) => {
                let specs = codec_specs_from_json_str(json)?;
                Ok(codec_pipeline_from_specs(&specs))
            }
            Self::PipelineJsonValue(value) => {
                let specs = codec_specs_from_json_value(value)?;
                Ok(codec_pipeline_from_specs(&specs))
            }
            Self::ZarrV2Json {
                filters,
                compressor,
            } => Ok(codec_pipeline_from_zarr_v2_json_str(
                filters.as_deref(),
                compressor.as_deref(),
            )?),
            Self::ZarrV2JsonValue {
                filters,
                compressor,
            } => {
                let filters = filters
                    .as_ref()
                    .map(codec_specs_from_json_value)
                    .transpose()?
                    .unwrap_or_default();
                let compressor = compressor
                    .as_ref()
                    .map(CodecSpec::from_json_value)
                    .transpose()?
                    .filter(|spec| !matches!(spec, CodecSpec::None));
                Ok(codec_pipeline_from_zarr_v2_specs(
                    &filters,
                    compressor.as_ref(),
                ))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Array {
    pub shape: Vec<usize>,
    pub chunk_shape: Vec<usize>,
    pub chunk_grid_shape: Vec<usize>,
    pub dtype: DType,
    pub codec: SharedCodec,
    pub chunks: ChunkStore,
}

impl Array {
    pub fn from_meta(meta: ArrayMeta, io_pool: &IoPool) -> DataBankResult<Self> {
        validate_array_meta(&meta)?;
        let codec = meta.codec.build()?;
        let chunks = ChunkStore::from_meta(meta.chunks, io_pool)?;
        Ok(Self {
            shape: meta.shape,
            chunk_shape: meta.chunk_shape,
            chunk_grid_shape: meta.chunk_grid_shape,
            dtype: meta.dtype,
            codec,
            chunks,
        })
    }

    pub fn unregister_files(&self, io_pool: &IoPool) -> DataBankResult<()> {
        self.chunks.unregister_files(io_pool)
    }
}

#[derive(Debug, Clone)]
pub enum ChunkStoreMeta {
    /// All chunks are stored in one file and located by `(offset, len)`.
    ///
    /// `locations` must be ordered by C-order logical chunk index. Edge chunks
    /// are encoded after cropping to their logical extent.
    FileOffset {
        path: PathBuf,
        locations: Vec<FileChunkLocation>,
    },
    /// Each chunk is a separate file.
    ///
    /// `locations` must be ordered by C-order logical chunk index. Edge chunks
    /// are encoded after cropping to their logical extent.
    Directory {
        locations: Vec<DirectoryChunkLocationMeta>,
    },
    /// Encoded chunks already live in process memory.
    ///
    /// `chunks` must be ordered by C-order logical chunk index. Edge chunks
    /// are encoded after cropping to their logical extent.
    Memory { chunks: Vec<Arc<[u8]>> },
}

impl ChunkStoreMeta {
    pub fn len(&self) -> usize {
        match self {
            Self::FileOffset { locations, .. } => locations.len(),
            Self::Directory { locations } => locations.len(),
            Self::Memory { chunks } => chunks.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone)]
pub enum ChunkStore {
    FileOffset {
        file: RegisteredFile,
        locations: Vec<FileChunkLocation>,
    },
    Directory {
        locations: Vec<DirectoryChunkLocation>,
    },
    Memory {
        chunks: Vec<Arc<[u8]>>,
    },
}

impl ChunkStore {
    fn from_meta(meta: ChunkStoreMeta, io_pool: &IoPool) -> DataBankResult<Self> {
        match meta {
            ChunkStoreMeta::FileOffset { path, locations } => {
                let file = register_readonly_file(io_pool, &path)?;
                Ok(Self::FileOffset { file, locations })
            }
            ChunkStoreMeta::Directory { locations } => {
                let mut registered = Vec::with_capacity(locations.len());
                for location in locations {
                    let file = match register_readonly_file(io_pool, &location.path) {
                        Ok(file) => file,
                        Err(err) => {
                            unregister_registered_files(
                                io_pool,
                                registered
                                    .iter()
                                    .map(|entry: &DirectoryChunkLocation| entry.file),
                            );
                            return Err(err);
                        }
                    };
                    registered.push(DirectoryChunkLocation {
                        file,
                        len: location.len,
                    });
                }
                Ok(Self::Directory {
                    locations: registered,
                })
            }
            ChunkStoreMeta::Memory { chunks } => Ok(Self::Memory { chunks }),
        }
    }

    pub fn unregister_files(&self, io_pool: &IoPool) -> DataBankResult<()> {
        match self {
            Self::FileOffset { file, .. } => {
                io_pool.unregister_file(file.id).map_err(DataBankError::Io)
            }
            Self::Directory { locations } => {
                let mut first_error = None;
                for location in locations {
                    if let Err(err) = io_pool.unregister_file(location.file.id) {
                        first_error.get_or_insert(DataBankError::Io(err));
                    }
                }
                first_error.map_or(Ok(()), Err)
            }
            Self::Memory { .. } => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisteredFile {
    pub id: FileId,
    pub file_ref: FileRef,
}

impl RegisteredFile {
    pub fn new(id: FileId) -> DataBankResult<Self> {
        let file_ref = u64::try_from(id)
            .map(FileRef::new)
            .map_err(|_| DataBankError::InvalidArrayMeta("file id overflow".to_string()))?;
        Ok(Self { id, file_ref })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileChunkLocation {
    pub offset: u64,
    pub len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryChunkLocationMeta {
    pub path: PathBuf,
    pub len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectoryChunkLocation {
    pub file: RegisteredFile,
    pub len: usize,
}

#[derive(Debug, Clone)]
pub enum ChunkRef {
    AccessItem(AccessItem),
    Memory {
        bytes: Arc<[u8]>,
        codec: SharedCodec,
        expected_size: usize,
    },
}

pub fn logical_chunk_index(coords: &[usize], grid_shape: &[usize]) -> DataBankResult<usize> {
    if coords.len() != grid_shape.len() {
        return Err(DataBankError::InvalidArrayMeta(
            "chunk coord dimensionality mismatch".to_string(),
        ));
    }

    let mut index = 0usize;
    let mut stride = 1usize;
    for (&coord, &dim) in coords.iter().rev().zip(grid_shape.iter().rev()) {
        if coord >= dim {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "chunk coord {coord} is out of range for dim {dim}"
            )));
        }
        index = index
            .checked_add(coord.checked_mul(stride).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("chunk index overflow".to_string())
            })?)
            .ok_or_else(|| DataBankError::InvalidArrayMeta("chunk index overflow".to_string()))?;
        stride = stride
            .checked_mul(dim)
            .ok_or_else(|| DataBankError::InvalidArrayMeta("chunk stride overflow".to_string()))?;
    }
    Ok(index)
}

pub fn chunk_ref(array: &Array, chunk_index: usize) -> DataBankResult<ChunkRef> {
    let expected_size = linear_chunk_expected_size(array, chunk_index)?;
    match &array.chunks {
        ChunkStore::FileOffset { file, locations } => {
            let Some(location) = locations.get(chunk_index) else {
                return Err(invalid_chunk_index(chunk_index));
            };
            Ok(ChunkRef::AccessItem(AccessItem::new(
                ChunkKey::new(file.file_ref, location.offset, location.len),
                Arc::clone(&array.codec),
                Some(expected_size),
            )))
        }
        ChunkStore::Directory { locations } => {
            let Some(location) = locations.get(chunk_index) else {
                return Err(invalid_chunk_index(chunk_index));
            };
            Ok(ChunkRef::AccessItem(AccessItem::new(
                ChunkKey::new(location.file.file_ref, 0, location.len),
                Arc::clone(&array.codec),
                Some(expected_size),
            )))
        }
        ChunkStore::Memory { chunks } => {
            let Some(bytes) = chunks.get(chunk_index) else {
                return Err(invalid_chunk_index(chunk_index));
            };
            Ok(ChunkRef::Memory {
                bytes: Arc::clone(bytes),
                codec: Arc::clone(&array.codec),
                expected_size,
            })
        }
    }
}

fn validate_array_meta(meta: &ArrayMeta) -> DataBankResult<()> {
    match meta.order {
        ArrayOrder::C => {}
    }
    if meta.shape.is_empty() {
        return Err(DataBankError::InvalidArrayMeta(
            "shape must not be empty".to_string(),
        ));
    }
    if meta.shape.len() != meta.chunk_shape.len() || meta.shape.len() != meta.chunk_grid_shape.len()
    {
        return Err(DataBankError::InvalidArrayMeta(
            "shape/chunk_shape/chunk_grid_shape dimensionality mismatch".to_string(),
        ));
    }
    if meta.chunk_shape.contains(&0) {
        return Err(DataBankError::InvalidArrayMeta(
            "chunk_shape entries must be nonzero".to_string(),
        ));
    }
    for axis in 0..meta.shape.len() {
        let expected = div_ceil(meta.shape[axis], meta.chunk_shape[axis]);
        if meta.chunk_grid_shape[axis] != expected {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "chunk_grid_shape[{axis}] is {}, expected {expected}",
                meta.chunk_grid_shape[axis]
            )));
        }
    }
    let expected_chunks = product(&meta.chunk_grid_shape)
        .ok_or_else(|| DataBankError::InvalidArrayMeta("chunk count overflow".to_string()))?;
    if meta.chunks.len() != expected_chunks {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "chunk location count is {}, expected {expected_chunks}",
            meta.chunks.len()
        )));
    }
    Ok(())
}

fn linear_chunk_expected_size(array: &Array, chunk_index: usize) -> DataBankResult<usize> {
    let count = product(&array.chunk_grid_shape)
        .ok_or_else(|| DataBankError::InvalidArrayMeta("shape product overflow".to_string()))?;
    if chunk_index >= count {
        return Err(invalid_chunk_index(chunk_index));
    }

    let mut index = chunk_index;
    let mut elements = 1usize;
    for axis in (0..array.chunk_grid_shape.len()).rev() {
        let dim = array.chunk_grid_shape[axis];
        let coord = index % dim;
        index /= dim;
        let start = coord
            .checked_mul(array.chunk_shape[axis])
            .ok_or_else(|| DataBankError::InvalidArrayMeta("chunk start overflow".to_string()))?;
        let extent = array.chunk_shape[axis].min(array.shape[axis] - start);
        elements = elements.checked_mul(extent).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("chunk element count overflow".to_string())
        })?;
    }
    elements
        .checked_mul(array.dtype.item_size())
        .ok_or_else(|| DataBankError::InvalidArrayMeta("chunk byte size overflow".to_string()))
}

fn register_readonly_file(io_pool: &IoPool, path: &Path) -> DataBankResult<RegisteredFile> {
    let id = io_pool.register_readonly_file(path)?;
    match RegisteredFile::new(id) {
        Ok(file) => Ok(file),
        Err(err) => {
            let _ = io_pool.unregister_file(id);
            Err(err)
        }
    }
}

fn unregister_registered_files<I>(io_pool: &IoPool, files: I)
where
    I: IntoIterator<Item = RegisteredFile>,
{
    for file in files {
        let _ = io_pool.unregister_file(file.id);
    }
}

fn product(values: &[usize]) -> Option<usize> {
    values
        .iter()
        .try_fold(1usize, |acc, &value| acc.checked_mul(value))
}

fn div_ceil(n: usize, d: usize) -> usize {
    n / d + usize::from(n % d != 0)
}

fn invalid_chunk_index(index: usize) -> DataBankError {
    DataBankError::InvalidArrayMeta(format!("invalid chunk index {index}"))
}
