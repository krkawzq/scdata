use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::access::{AccessItem, ChunkKey, FileRef};
use crate::codecs::SharedCodec;
use crate::iopool::{FileId, IoPool};

use crate::databank::error::{DataBankError, DataBankResult};

use super::dtype::DType;
use super::grid::{validate_array_shape, ArrayGrid};
use super::spec::{ArrayOrder, ArraySpec, ChunkSourceSpec};

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

fn validate_array_spec_basics(spec: &ArraySpec) -> DataBankResult<()> {
    validate_array_shape(&spec.shape)?;
    match spec.order {
        ArrayOrder::C => {}
    }
    Ok(())
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

fn invalid_chunk_index(index: usize) -> DataBankError {
    DataBankError::InvalidArrayMeta(format!("invalid chunk index {index}"))
}
