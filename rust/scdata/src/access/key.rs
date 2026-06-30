//! Chunk and decode cache keys.

use super::backend::FileRef;
use crate::codecs::{CodecCacheKey, SharedCodec};

/// Unique identity for one compressed chunk read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkKey {
    pub file: FileRef,
    pub offset: u64,
    pub len: usize,
}

impl ChunkKey {
    pub fn new(file: FileRef, offset: u64, len: usize) -> Self {
        Self { file, offset, len }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct DecodeKey {
    pub(crate) chunk: ChunkKey,
    pub(crate) codec: CodecCacheKey,
    pub(crate) expected_size: Option<usize>,
}

impl DecodeKey {
    pub(crate) fn new(chunk: ChunkKey, codec: &SharedCodec, expected_size: Option<usize>) -> Self {
        Self {
            chunk,
            codec: codec.cache_key(),
            expected_size,
        }
    }
}

#[inline]
pub(crate) fn shard_for_key(key: ChunkKey, shard_count: usize) -> usize {
    debug_assert!(shard_count > 0);
    if shard_count == 1 {
        return 0;
    }

    (mix_chunk_key(key) as usize) % shard_count
}

#[inline]
fn mix_chunk_key(key: ChunkKey) -> u64 {
    let mut x = key.file.0;
    x ^= key.offset.rotate_left(17);
    x ^= (key.len as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    splitmix64(x)
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}
