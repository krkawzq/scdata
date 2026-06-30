use std::path::PathBuf;
use std::sync::Arc;

use super::codec::ArrayCodecSpec;
use super::dtype::DType;
use super::storage::RegisteredFile;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayOrder {
    C,
}

#[derive(Debug, Clone)]
pub struct ArraySpec {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub order: ArrayOrder,
    pub codec: ArrayCodecSpec,
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
    /// A file id registered by the caller. The caller retains ownership of the
    /// registration; arrays built from this source will not unregister it.
    RegisteredFile {
        file: RegisteredFile,
        offset: u64,
        len: usize,
    },
}
