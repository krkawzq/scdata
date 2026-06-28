use std::collections::BTreeMap;
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

/// Compatibility input used by the current Python binding.
///
/// New code should build an [`ArraySpec`] directly. This type intentionally
/// keeps the old field names so the binding can migrate incrementally; it is
/// converted to [`ArraySpec`] before any dataset is built.
#[derive(Debug, Clone)]
pub struct ArrayMeta {
    pub shape: Vec<usize>,
    pub chunk_shape: Vec<usize>,
    pub chunk_grid_shape: Vec<usize>,
    pub dtype: DType,
    pub order: ArrayOrder,
    pub codec: ArrayCodecMeta,
    pub chunks: ChunkStoreMeta,
    /// Compatibility flag for older callers. The new representation is
    /// [`ArrayGridSpec::Rectilinear`], which carries explicit boundaries.
    pub variable_chunks: bool,
    /// Explicit rectilinear chunk boundaries, one boundary vector per axis.
    /// Regular grids use `None`.
    pub chunk_boundaries: Option<Vec<Vec<usize>>>,
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
    /// pipeline. This does not imply a storage layout.
    ZarrV2Json {
        filters: Option<String>,
        compressor: Option<String>,
    },
    /// Zarr v2 filter/compressor JSON values converted into this crate's codec
    /// pipeline. This does not imply a storage layout.
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

    fn is_uncompressed_compat(&self) -> bool {
        matches!(self, Self::Uncompressed)
    }
}

#[derive(Debug, Clone)]
pub struct ArraySpec {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub order: ArrayOrder,
    pub codec: ArrayCodecMeta,
    pub grid: ArrayGridSpec,
    pub chunks: Vec<ChunkSpec>,
}

#[derive(Debug, Clone)]
pub enum ArrayGridSpec {
    Regular {
        chunk_shape: Vec<usize>,
        edge: EdgeChunkLayout,
    },
    Rectilinear {
        axes: Vec<Vec<usize>>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeChunkLayout {
    /// Every decoded chunk has the full regular chunk shape.
    Padded,
    /// Edge chunks decode only to their logical extent.
    Cropped,
}

#[derive(Debug, Clone)]
pub struct ChunkSpec {
    pub source: ChunkSourceSpec,
    pub decoded_bytes: usize,
}

#[derive(Debug, Clone)]
pub enum ChunkSourceSpec {
    /// Encoded chunk bytes already live in process memory.
    Memory { bytes: Arc<[u8]> },
    /// Decoded chunk bytes already live in process memory.
    ///
    /// This is used for normalized all-zero fill chunks; regular in-memory
    /// callers should use [`Self::Memory`] so array codecs still apply.
    DecodedMemory { bytes: Arc<[u8]> },
    File {
        path: PathBuf,
        offset: u64,
        len: usize,
    },
    RegisteredFile {
        file: RegisteredFile,
        offset: u64,
        len: usize,
    },
}

#[derive(Debug, Clone)]
pub struct Array {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub codec: SharedCodec,
    pub grid: ArrayGrid,
    pub chunks: Vec<Chunk>,
    pub(crate) files: Vec<RegisteredFile>,
}

impl Array {
    #[allow(dead_code)]
    pub fn from_meta(meta: ArrayMeta, io_pool: &IoPool) -> DataBankResult<Self> {
        let spec = ArraySpec::from_compat_meta(meta)?;
        build_array_from_spec(spec, io_pool)
    }

    pub fn unregister_files(&self, io_pool: &IoPool) -> DataBankResult<()> {
        let mut first_error = None;
        for file in &self.files {
            if let Err(err) = io_pool.unregister_file(file.id) {
                first_error.get_or_insert(DataBankError::Io(err));
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    #[allow(dead_code)]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn chunk_grid_shape(&self) -> &[usize] {
        self.grid.grid_shape()
    }

    pub fn regular_chunk_shape(&self) -> Option<&[usize]> {
        self.grid.regular_chunk_shape()
    }

    pub fn regular_chunk_shape_required(&self, context: &'static str) -> DataBankResult<&[usize]> {
        self.regular_chunk_shape().ok_or_else(|| {
            DataBankError::InvalidArrayMeta(format!("{context} requires a regular chunk grid"))
        })
    }

    #[allow(dead_code)]
    pub fn chunk_ref(&self, chunk_index: usize) -> DataBankResult<ChunkRef> {
        chunk_ref(self, chunk_index)
    }

    pub fn memory_chunks(&self) -> Option<&[Chunk]> {
        self.chunks
            .iter()
            .all(|chunk| matches!(chunk.source, ChunkSource::Memory { .. }))
            .then_some(self.chunks.as_slice())
    }

    pub fn regular_1d_chunk_len(&self) -> Option<usize> {
        let [chunk_len] = self.grid.regular_chunk_shape()? else {
            return None;
        };
        Some(*chunk_len)
    }

    #[allow(dead_code)]
    pub fn one_dim_len(&self, context: &'static str) -> DataBankResult<usize> {
        let [len] = self.shape.as_slice() else {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "{context} requires 1D array, got shape {:?}",
                self.shape
            )));
        };
        Ok(*len)
    }

    pub fn range_piece_count_1d(&self, start: usize, end: usize) -> DataBankResult<usize> {
        self.grid.range_piece_count_1d(&self.shape, start, end)
    }
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub source: ChunkSource,
    pub decoded_bytes: usize,
}

#[derive(Debug, Clone)]
pub enum ChunkSource {
    Memory {
        bytes: Arc<[u8]>,
        decoded: bool,
    },
    File {
        file: RegisteredFile,
        offset: u64,
        len: usize,
    },
}

#[derive(Debug, Clone)]
pub enum ArrayGrid {
    Regular {
        chunk_shape: Vec<usize>,
        grid_shape: Vec<usize>,
        edge: EdgeChunkLayout,
    },
    Rectilinear {
        axes: Vec<RectilinearAxis>,
        grid_shape: Vec<usize>,
    },
}

#[derive(Debug, Clone)]
pub struct RectilinearAxis {
    /// Monotonic boundaries, length = chunks_on_axis + 1.
    pub boundaries: Vec<usize>,
}

impl ArrayGrid {
    pub fn from_spec(shape: &[usize], spec: ArrayGridSpec) -> DataBankResult<Self> {
        match spec {
            ArrayGridSpec::Regular { chunk_shape, edge } => {
                validate_regular_grid_shape(shape, &chunk_shape)?;
                let mut grid_shape = Vec::with_capacity(shape.len());
                for (&dim, &chunk) in shape.iter().zip(chunk_shape.iter()) {
                    grid_shape.push(div_ceil(dim, chunk));
                }
                Ok(Self::Regular {
                    chunk_shape,
                    grid_shape,
                    edge,
                })
            }
            ArrayGridSpec::Rectilinear { axes } => {
                if axes.len() != shape.len() {
                    return Err(DataBankError::InvalidArrayMeta(format!(
                        "rectilinear axis count {} does not match shape rank {}",
                        axes.len(),
                        shape.len()
                    )));
                }
                let mut parsed_axes = Vec::with_capacity(axes.len());
                let mut grid_shape = Vec::with_capacity(axes.len());
                for (axis_index, (boundaries, &dim)) in axes.into_iter().zip(shape).enumerate() {
                    validate_rectilinear_boundaries(axis_index, &boundaries, dim)?;
                    grid_shape.push(boundaries.len() - 1);
                    parsed_axes.push(RectilinearAxis { boundaries });
                }
                Ok(Self::Rectilinear {
                    axes: parsed_axes,
                    grid_shape,
                })
            }
        }
    }

    pub fn grid_shape(&self) -> &[usize] {
        match self {
            Self::Regular { grid_shape, .. } | Self::Rectilinear { grid_shape, .. } => grid_shape,
        }
    }

    pub fn regular_chunk_shape(&self) -> Option<&[usize]> {
        match self {
            Self::Regular { chunk_shape, .. } => Some(chunk_shape),
            Self::Rectilinear { .. } => None,
        }
    }

    pub fn num_chunks(&self) -> DataBankResult<usize> {
        product(self.grid_shape())
            .ok_or_else(|| DataBankError::InvalidArrayMeta("chunk count overflow".to_string()))
    }

    pub fn logical_chunk_index(&self, coords: &[usize]) -> DataBankResult<usize> {
        logical_chunk_index(coords, self.grid_shape())
    }

    pub fn chunk_coords(&self, chunk_index: usize) -> DataBankResult<Vec<usize>> {
        unravel_chunk_index(chunk_index, self.grid_shape())
    }

    pub fn decoded_extent_for_chunk(
        &self,
        shape: &[usize],
        chunk_index: usize,
    ) -> DataBankResult<Vec<usize>> {
        let coords = self.chunk_coords(chunk_index)?;
        match self {
            Self::Regular {
                chunk_shape, edge, ..
            } => match edge {
                EdgeChunkLayout::Padded => Ok(chunk_shape.clone()),
                EdgeChunkLayout::Cropped => regular_logical_extent(shape, chunk_shape, &coords),
            },
            Self::Rectilinear { axes, .. } => axes
                .iter()
                .zip(coords.iter())
                .map(|(axis, &coord)| {
                    axis.boundaries
                        .get(coord + 1)
                        .zip(axis.boundaries.get(coord))
                        .map(|(&end, &start)| end - start)
                        .ok_or_else(|| invalid_chunk_index(chunk_index))
                })
                .collect(),
        }
    }

    pub fn decoded_bytes_for_chunk(
        &self,
        shape: &[usize],
        dtype: DType,
        chunk_index: usize,
    ) -> DataBankResult<usize> {
        let elements =
            product(&self.decoded_extent_for_chunk(shape, chunk_index)?).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("chunk element count overflow".to_string())
            })?;
        elements
            .checked_mul(dtype.item_size())
            .ok_or_else(|| DataBankError::InvalidArrayMeta("chunk byte size overflow".to_string()))
    }

    pub fn plan_1d_range(
        &self,
        shape: &[usize],
        dtype: DType,
        chunks: &[Chunk],
        start: usize,
        end: usize,
    ) -> DataBankResult<Vec<RangePiece>> {
        validate_1d_range(shape, start, end, "1D range planning")?;
        if start == end {
            return Ok(Vec::new());
        }

        match self {
            Self::Regular { chunk_shape, .. } => {
                let [chunk_len] = chunk_shape.as_slice() else {
                    return Err(DataBankError::InvalidArrayMeta(format!(
                        "1D range planning requires 1D chunk shape, got {chunk_shape:?}"
                    )));
                };
                self.plan_regular_1d_range(shape[0], dtype, chunks, *chunk_len, start, end)
            }
            Self::Rectilinear { axes, .. } => {
                let [axis] = axes.as_slice() else {
                    return Err(DataBankError::InvalidArrayMeta(
                        "1D range planning requires a 1D rectilinear grid".to_string(),
                    ));
                };
                self.plan_rectilinear_1d_range(axis, dtype, chunks, start, end)
            }
        }
    }

    fn plan_regular_1d_range(
        &self,
        len: usize,
        dtype: DType,
        chunks: &[Chunk],
        chunk_len: usize,
        start: usize,
        end: usize,
    ) -> DataBankResult<Vec<RangePiece>> {
        let mut pieces = Vec::with_capacity(fixed_range_piece_count(start, end, chunk_len));
        let item_size = dtype.item_size();
        let mut pos = start;
        while pos < end {
            let chunk_index = pos / chunk_len;
            let chunk_start = chunk_index.checked_mul(chunk_len).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("1D chunk start overflow".to_string())
            })?;
            let logical_chunk_len = chunk_len.min(len - chunk_start);
            let in_chunk = pos - chunk_start;
            let elements = (end - pos).min(logical_chunk_len - in_chunk);
            let byte_start = in_chunk.checked_mul(item_size).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("range byte start overflow".to_string())
            })?;
            let byte_len = elements.checked_mul(item_size).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("range byte length overflow".to_string())
            })?;
            let byte_end = byte_start.checked_add(byte_len).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("range byte end overflow".to_string())
            })?;
            validate_range_inside_decoded_chunk(chunks, chunk_index, byte_end)?;
            pieces.push(RangePiece {
                chunk_index,
                byte_start,
                byte_end,
                elements,
            });
            pos += elements;
        }
        Ok(pieces)
    }

    fn plan_rectilinear_1d_range(
        &self,
        axis: &RectilinearAxis,
        dtype: DType,
        chunks: &[Chunk],
        start: usize,
        end: usize,
    ) -> DataBankResult<Vec<RangePiece>> {
        let item_size = dtype.item_size();
        let mut pieces = Vec::new();
        let mut chunk_index = rectilinear_chunk_for_pos(axis, start)?;
        let mut pos = start;
        while pos < end {
            let chunk_start = axis.boundaries[chunk_index];
            let chunk_end = axis.boundaries[chunk_index + 1];
            let in_chunk = pos - chunk_start;
            let elements = (end.min(chunk_end)) - pos;
            let byte_start = in_chunk.checked_mul(item_size).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("rectilinear byte start overflow".to_string())
            })?;
            let byte_len = elements.checked_mul(item_size).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("rectilinear byte length overflow".to_string())
            })?;
            let byte_end = byte_start.checked_add(byte_len).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("rectilinear byte end overflow".to_string())
            })?;
            validate_range_inside_decoded_chunk(chunks, chunk_index, byte_end)?;
            pieces.push(RangePiece {
                chunk_index,
                byte_start,
                byte_end,
                elements,
            });
            pos += elements;
            chunk_index += usize::from(pos < end);
        }
        Ok(pieces)
    }

    pub fn range_piece_count_1d(
        &self,
        shape: &[usize],
        start: usize,
        end: usize,
    ) -> DataBankResult<usize> {
        validate_1d_range(shape, start, end, "1D range piece count")?;
        if start == end {
            return Ok(0);
        }
        match self {
            Self::Regular { chunk_shape, .. } => {
                let [chunk_len] = chunk_shape.as_slice() else {
                    return Err(DataBankError::InvalidArrayMeta(format!(
                        "1D range piece count requires 1D chunk shape, got {chunk_shape:?}"
                    )));
                };
                Ok(fixed_range_piece_count(start, end, *chunk_len))
            }
            Self::Rectilinear { axes, .. } => {
                let [axis] = axes.as_slice() else {
                    return Err(DataBankError::InvalidArrayMeta(
                        "1D range piece count requires a 1D rectilinear grid".to_string(),
                    ));
                };
                let first = rectilinear_chunk_for_pos(axis, start)?;
                let last = rectilinear_chunk_for_pos(axis, end - 1)?;
                Ok(last - first + 1)
            }
        }
    }

    pub fn physical_row_width_2d(
        &self,
        shape: &[usize],
        chunk_row: usize,
        chunk_col: usize,
    ) -> DataBankResult<usize> {
        match self {
            Self::Regular {
                chunk_shape, edge, ..
            } => {
                let [_, chunk_cols] = chunk_shape.as_slice() else {
                    return Err(DataBankError::InvalidArrayMeta(format!(
                        "Dense2D requires 2D chunk shape, got {chunk_shape:?}"
                    )));
                };
                match edge {
                    EdgeChunkLayout::Padded => Ok(*chunk_cols),
                    EdgeChunkLayout::Cropped => {
                        let extent =
                            regular_logical_extent(shape, chunk_shape, &[chunk_row, chunk_col])?;
                        Ok(extent[1])
                    }
                }
            }
            Self::Rectilinear { .. } => Err(DataBankError::InvalidArrayMeta(
                "Dense2D does not support rectilinear chunk grids".to_string(),
            )),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RangePiece {
    pub chunk_index: usize,
    pub byte_start: usize,
    pub byte_end: usize,
    pub elements: usize,
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
    /// Byte offset within `path`. Directory stores use 0; zip stores can point
    /// into the zip file itself.
    pub offset: u64,
    pub len: usize,
}

/// Compatibility input from older Python/Rust callers.
#[derive(Debug, Clone)]
pub enum ChunkStoreMeta {
    /// All chunks are stored in one payload file and located by `(offset, len)`.
    ///
    /// This is treated as the legacy scdata payload layout: regular edge
    /// chunks are cropped to their logical extent.
    FileOffset {
        path: PathBuf,
        locations: Vec<FileChunkLocation>,
    },
    /// Each chunk is described by an explicit file path plus offset/length.
    ///
    /// Standard directory stores pass distinct paths and offset 0. Zip stores
    /// pass the same zip path with physical offsets.
    Directory {
        locations: Vec<DirectoryChunkLocationMeta>,
    },
    /// Encoded chunks already live in process memory.
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
pub enum ChunkRef {
    AccessItem(AccessItem),
    Memory {
        bytes: Arc<[u8]>,
        codec: SharedCodec,
        expected_size: usize,
        decoded: bool,
    },
}

impl ArraySpec {
    pub fn from_compat_meta(meta: ArrayMeta) -> DataBankResult<Self> {
        validate_compat_array_meta_basics(&meta)?;
        let dtype = meta.dtype;
        let chunk_count = meta.chunks.len();
        let grid = compat_grid_spec(&meta)?;
        let grid_for_size = ArrayGrid::from_spec(&meta.shape, grid.clone())?;
        let expected_chunks = grid_for_size.num_chunks()?;
        if chunk_count != expected_chunks {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "chunk location count is {chunk_count}, expected {expected_chunks}"
            )));
        }

        let decoded_sizes = (0..chunk_count)
            .map(|idx| grid_for_size.decoded_bytes_for_chunk(&meta.shape, dtype, idx))
            .collect::<DataBankResult<Vec<_>>>()?;

        let chunks = match meta.chunks {
            ChunkStoreMeta::FileOffset { path, locations } => locations
                .into_iter()
                .zip(decoded_sizes)
                .map(|(location, decoded_bytes)| ChunkSpec {
                    source: ChunkSourceSpec::File {
                        path: path.clone(),
                        offset: location.offset,
                        len: location.len,
                    },
                    decoded_bytes,
                })
                .collect(),
            ChunkStoreMeta::Directory { locations } => locations
                .into_iter()
                .zip(decoded_sizes)
                .map(|(location, decoded_bytes)| ChunkSpec {
                    source: ChunkSourceSpec::File {
                        path: location.path,
                        offset: location.offset,
                        len: location.len,
                    },
                    decoded_bytes,
                })
                .collect(),
            ChunkStoreMeta::Memory { chunks } => chunks
                .into_iter()
                .zip(decoded_sizes)
                .map(|(bytes, decoded_bytes)| ChunkSpec {
                    source: ChunkSourceSpec::Memory { bytes },
                    decoded_bytes,
                })
                .collect(),
        };

        Ok(Self {
            shape: meta.shape,
            dtype,
            order: meta.order,
            codec: meta.codec,
            grid,
            chunks,
        })
    }
}

pub fn build_array_from_spec(spec: ArraySpec, io_pool: &IoPool) -> DataBankResult<Array> {
    validate_array_spec_basics(&spec)?;
    let codec = spec.codec.build()?;
    let grid = ArrayGrid::from_spec(&spec.shape, spec.grid)?;
    let expected_chunks = grid.num_chunks()?;
    if spec.chunks.len() != expected_chunks {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "chunk count is {}, expected {expected_chunks}",
            spec.chunks.len()
        )));
    }

    let mut path_cache = BTreeMap::<PathBuf, RegisteredFile>::new();
    let mut files = Vec::<RegisteredFile>::new();
    let mut chunks = Vec::with_capacity(spec.chunks.len());

    for (chunk_index, chunk) in spec.chunks.iter().enumerate() {
        let expected_decoded =
            grid.decoded_bytes_for_chunk(&spec.shape, spec.dtype, chunk_index)?;
        if chunk.decoded_bytes != expected_decoded {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "chunk {chunk_index} decoded_bytes is {}, expected {expected_decoded}",
                chunk.decoded_bytes
            )));
        }
    }

    for (chunk_index, chunk) in spec.chunks.into_iter().enumerate() {
        let source = match chunk.source {
            ChunkSourceSpec::Memory { bytes } => ChunkSource::Memory {
                bytes,
                decoded: false,
            },
            ChunkSourceSpec::DecodedMemory { bytes } => {
                if bytes.len() != chunk.decoded_bytes {
                    unregister_registered_files(io_pool, &files);
                    return Err(DataBankError::InvalidArrayMeta(format!(
                        "decoded memory chunk {chunk_index} has {} bytes, expected {}",
                        bytes.len(),
                        chunk.decoded_bytes
                    )));
                }
                ChunkSource::Memory {
                    bytes,
                    decoded: true,
                }
            }
            ChunkSourceSpec::File { path, offset, len } => {
                if len == 0 {
                    ChunkSource::Memory {
                        bytes: zero_filled_chunk(chunk.decoded_bytes),
                        decoded: true,
                    }
                } else {
                    let file = if let Some(file) = path_cache.get(&path).copied() {
                        file
                    } else {
                        let file = match register_readonly_file(io_pool, &path) {
                            Ok(file) => file,
                            Err(err) => {
                                unregister_registered_files(io_pool, &files);
                                return Err(err);
                            }
                        };
                        path_cache.insert(path, file);
                        files.push(file);
                        file
                    };
                    ChunkSource::File { file, offset, len }
                }
            }
            ChunkSourceSpec::RegisteredFile { file, offset, len } => {
                if len == 0 {
                    ChunkSource::Memory {
                        bytes: zero_filled_chunk(chunk.decoded_bytes),
                        decoded: true,
                    }
                } else {
                    if !files.iter().any(|existing| existing.id == file.id) {
                        files.push(file);
                    }
                    ChunkSource::File { file, offset, len }
                }
            }
        };
        chunks.push(Chunk {
            source,
            decoded_bytes: chunk.decoded_bytes,
        });
    }

    Ok(Array {
        shape: spec.shape,
        dtype: spec.dtype,
        codec,
        grid,
        chunks,
        files,
    })
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
    let Some(chunk) = array.chunks.get(chunk_index) else {
        return Err(invalid_chunk_index(chunk_index));
    };
    match &chunk.source {
        ChunkSource::File { file, offset, len } => Ok(ChunkRef::AccessItem(AccessItem::new(
            ChunkKey::new(file.file_ref, *offset, *len),
            Arc::clone(&array.codec),
            Some(chunk.decoded_bytes),
        ))),
        ChunkSource::Memory { bytes, decoded } => Ok(ChunkRef::Memory {
            bytes: Arc::clone(bytes),
            codec: Arc::clone(&array.codec),
            expected_size: chunk.decoded_bytes,
            decoded: *decoded,
        }),
    }
}

fn compat_grid_spec(meta: &ArrayMeta) -> DataBankResult<ArrayGridSpec> {
    if meta.variable_chunks {
        let axes = infer_compat_rectilinear_axes(meta)?;
        return Ok(ArrayGridSpec::Rectilinear { axes });
    }

    let edge = match &meta.chunks {
        ChunkStoreMeta::FileOffset { .. } => EdgeChunkLayout::Cropped,
        ChunkStoreMeta::Directory { .. } | ChunkStoreMeta::Memory { .. } => EdgeChunkLayout::Padded,
    };
    Ok(ArrayGridSpec::Regular {
        chunk_shape: meta.chunk_shape.clone(),
        edge,
    })
}

fn infer_compat_rectilinear_axes(meta: &ArrayMeta) -> DataBankResult<Vec<Vec<usize>>> {
    if let Some(axes) = &meta.chunk_boundaries {
        return Ok(axes.clone());
    }
    if meta.shape.len() != 1 {
        return Err(DataBankError::InvalidArrayMeta(
            "compat variable_chunks only supports 1D rectilinear arrays".to_string(),
        ));
    }
    if !meta.codec.is_uncompressed_compat() {
        return Err(DataBankError::InvalidArrayMeta(
            "compat variable_chunks cannot infer rectilinear boundaries for compressed chunks"
                .to_string(),
        ));
    }

    let item_size = meta.dtype.item_size();
    let mut boundaries = Vec::with_capacity(meta.chunks.len() + 1);
    boundaries.push(0usize);
    let mut cursor = 0usize;
    for len in compat_chunk_encoded_lengths(&meta.chunks) {
        if len % item_size != 0 {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "rectilinear chunk byte length {len} is not divisible by item size {item_size}"
            )));
        }
        cursor = cursor.checked_add(len / item_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("rectilinear boundary overflow".to_string())
        })?;
        boundaries.push(cursor);
    }
    if cursor != meta.shape[0] {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "inferred rectilinear final boundary is {cursor}, expected {}",
            meta.shape[0]
        )));
    }
    Ok(vec![boundaries])
}

fn compat_chunk_encoded_lengths(chunks: &ChunkStoreMeta) -> Vec<usize> {
    match chunks {
        ChunkStoreMeta::FileOffset { locations, .. } => {
            locations.iter().map(|location| location.len).collect()
        }
        ChunkStoreMeta::Directory { locations } => {
            locations.iter().map(|location| location.len).collect()
        }
        ChunkStoreMeta::Memory { chunks } => chunks.iter().map(|chunk| chunk.len()).collect(),
    }
}

fn validate_compat_array_meta_basics(meta: &ArrayMeta) -> DataBankResult<()> {
    validate_array_shape(&meta.shape)?;
    match meta.order {
        ArrayOrder::C => {}
    }
    if meta.shape.len() != meta.chunk_shape.len() {
        return Err(DataBankError::InvalidArrayMeta(
            "shape/chunk_shape dimensionality mismatch".to_string(),
        ));
    }
    if meta.chunk_shape.contains(&0) {
        return Err(DataBankError::InvalidArrayMeta(
            "chunk_shape entries must be nonzero".to_string(),
        ));
    }
    if !meta.variable_chunks && meta.chunk_boundaries.is_some() {
        return Err(DataBankError::InvalidArrayMeta(
            "chunk_boundaries require variable_chunks=true".to_string(),
        ));
    }
    if !meta.variable_chunks {
        if meta.shape.len() != meta.chunk_grid_shape.len() {
            return Err(DataBankError::InvalidArrayMeta(
                "shape/chunk_grid_shape dimensionality mismatch".to_string(),
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
    }
    Ok(())
}

fn validate_array_spec_basics(spec: &ArraySpec) -> DataBankResult<()> {
    validate_array_shape(&spec.shape)?;
    match spec.order {
        ArrayOrder::C => {}
    }
    Ok(())
}

fn validate_array_shape(shape: &[usize]) -> DataBankResult<()> {
    if shape.is_empty() {
        return Err(DataBankError::InvalidArrayMeta(
            "shape must not be empty".to_string(),
        ));
    }
    Ok(())
}

fn validate_regular_grid_shape(shape: &[usize], chunk_shape: &[usize]) -> DataBankResult<()> {
    if shape.len() != chunk_shape.len() {
        return Err(DataBankError::InvalidArrayMeta(
            "shape/chunk_shape dimensionality mismatch".to_string(),
        ));
    }
    if chunk_shape.contains(&0) {
        return Err(DataBankError::InvalidArrayMeta(
            "chunk_shape entries must be nonzero".to_string(),
        ));
    }
    Ok(())
}

fn validate_rectilinear_boundaries(
    axis_index: usize,
    boundaries: &[usize],
    dim: usize,
) -> DataBankResult<()> {
    if boundaries.len() < 2 {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "rectilinear axis {axis_index} needs at least two boundaries"
        )));
    }
    if boundaries[0] != 0 {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "rectilinear axis {axis_index} must start at 0"
        )));
    }
    if *boundaries.last().unwrap() != dim {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "rectilinear axis {axis_index} final boundary is {}, expected {dim}",
            boundaries.last().unwrap()
        )));
    }
    for pair in boundaries.windows(2) {
        if pair[0] > pair[1] {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "rectilinear axis {axis_index} boundaries must be monotonic"
            )));
        }
    }
    Ok(())
}

fn validate_1d_range(
    shape: &[usize],
    start: usize,
    end: usize,
    context: &'static str,
) -> DataBankResult<()> {
    let [len] = shape else {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "{context} requires 1D array, got shape {shape:?}"
        )));
    };
    if start > end || end > *len {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "invalid 1D range [{start}, {end}) for length {len}"
        )));
    }
    Ok(())
}

fn validate_range_inside_decoded_chunk(
    chunks: &[Chunk],
    chunk_index: usize,
    byte_end: usize,
) -> DataBankResult<()> {
    let Some(chunk) = chunks.get(chunk_index) else {
        return Err(invalid_chunk_index(chunk_index));
    };
    if byte_end > chunk.decoded_bytes {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "planned byte end {byte_end} exceeds decoded chunk size {} for chunk {chunk_index}",
            chunk.decoded_bytes
        )));
    }
    Ok(())
}

fn regular_logical_extent(
    shape: &[usize],
    chunk_shape: &[usize],
    coords: &[usize],
) -> DataBankResult<Vec<usize>> {
    if shape.len() != chunk_shape.len() || shape.len() != coords.len() {
        return Err(DataBankError::InvalidArrayMeta(
            "regular extent dimensionality mismatch".to_string(),
        ));
    }
    let mut extent = Vec::with_capacity(shape.len());
    for ((&dim, &chunk), &coord) in shape.iter().zip(chunk_shape).zip(coords) {
        let start = coord.checked_mul(chunk).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("regular chunk start overflow".to_string())
        })?;
        if start >= dim && dim != 0 {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "regular chunk coord {coord} starts past dim {dim}"
            )));
        }
        extent.push(chunk.min(dim.saturating_sub(start)));
    }
    Ok(extent)
}

fn unravel_chunk_index(chunk_index: usize, grid_shape: &[usize]) -> DataBankResult<Vec<usize>> {
    let count = product(grid_shape)
        .ok_or_else(|| DataBankError::InvalidArrayMeta("chunk count overflow".to_string()))?;
    if chunk_index >= count {
        return Err(invalid_chunk_index(chunk_index));
    }
    let mut rem = chunk_index;
    let mut coords = vec![0usize; grid_shape.len()];
    for axis in (0..grid_shape.len()).rev() {
        let dim = grid_shape[axis];
        if dim == 0 {
            return Err(DataBankError::InvalidArrayMeta(
                "grid shape contains zero".to_string(),
            ));
        }
        coords[axis] = rem % dim;
        rem /= dim;
    }
    Ok(coords)
}

fn rectilinear_chunk_for_pos(axis: &RectilinearAxis, pos: usize) -> DataBankResult<usize> {
    let final_boundary = *axis.boundaries.last().unwrap_or(&0);
    if pos >= final_boundary {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "position {pos} is out of rectilinear axis range {final_boundary}"
        )));
    }
    let upper = axis.boundaries.partition_point(|&boundary| boundary <= pos);
    Ok(upper.saturating_sub(1))
}

pub(super) fn fixed_range_piece_count(start: usize, end: usize, chunk_len: usize) -> usize {
    debug_assert!(chunk_len > 0);
    if start == end {
        return 0;
    }
    debug_assert!(start < end);
    (end - 1) / chunk_len - start / chunk_len + 1
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

fn unregister_registered_files(io_pool: &IoPool, files: &[RegisteredFile]) {
    for file in files {
        let _ = io_pool.unregister_file(file.id);
    }
}

fn zero_filled_chunk(len: usize) -> Arc<[u8]> {
    if len == 0 {
        Arc::<[u8]>::from([])
    } else {
        Arc::from(vec![0u8; len].into_boxed_slice())
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::iopool::IoConfig;

    use super::*;

    static FILE_SEQ: AtomicU64 = AtomicU64::new(0);

    fn temp_file(bytes: &[u8]) -> PathBuf {
        let seq = FILE_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "scdata-databank-array-{}-{seq}",
            std::process::id()
        ));
        std::fs::write(&path, bytes).expect("write temp file");
        path
    }

    #[test]
    fn build_array_unregisters_files_when_late_chunk_validation_fails() {
        let path = temp_file(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let io_pool = IoPool::new(IoConfig::default()).expect("io pool");
        let spec = ArraySpec {
            shape: vec![4],
            dtype: DType::U32,
            order: ArrayOrder::C,
            codec: ArrayCodecMeta::Uncompressed,
            grid: ArrayGridSpec::Regular {
                chunk_shape: vec![2],
                edge: EdgeChunkLayout::Padded,
            },
            chunks: vec![
                ChunkSpec {
                    source: ChunkSourceSpec::File {
                        path,
                        offset: 0,
                        len: 8,
                    },
                    decoded_bytes: 8,
                },
                ChunkSpec {
                    source: ChunkSourceSpec::DecodedMemory {
                        bytes: Arc::from(vec![0u8; 7].into_boxed_slice()),
                    },
                    decoded_bytes: 8,
                },
            ],
        };

        let err = build_array_from_spec(spec, &io_pool).expect_err("array build should fail");
        assert!(
            matches!(err, DataBankError::InvalidArrayMeta(message) if message.contains("decoded memory chunk"))
        );
        assert!(
            io_pool.unregister_file(0).is_err(),
            "partially registered file should have been unregistered"
        );
    }
}
