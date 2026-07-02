use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::access::{AccessError, ChunkKey};
use crate::codecs::{
    try_blosc_lz4_plan_from_encoded, try_blosc_lz4_plan_from_prefix, BloscHeader, BloscLz4Plan,
    CodecResult, ValidatedBloscBlockRange,
};

/// Native view of one Blosc-LZ4 chunk's validated block table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeBloscBlockIndex {
    pub(crate) header: BloscHeader,
    pub(crate) decoded_size: usize,
    pub(crate) compressed_size: usize,
    pub(crate) block_size: usize,
    pub(crate) type_size: usize,
    pub(crate) byte_shuffled: bool,
    pub(crate) memcpyed: bool,
    pub(crate) blocks: Vec<NativeBloscBlockRange>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NativeBloscBlockRange {
    pub(crate) block_idx: usize,
    pub(crate) payload_relative_offset: usize,
    pub(crate) compressed_len: usize,
    pub(crate) decoded_offset: usize,
    pub(crate) decoded_len: usize,
    pub(crate) leftover: bool,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct NativeBlockIndexCache {
    entries: Arc<Mutex<HashMap<ChunkKey, Arc<NativeBloscBlockIndex>>>>,
}

impl NativeBlockIndexCache {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn get(&self, key: ChunkKey) -> Option<Arc<NativeBloscBlockIndex>> {
        self.entries
            .lock()
            .expect("native block index cache lock poisoned")
            .get(&key)
            .cloned()
    }

    pub(crate) fn insert(
        &self,
        key: ChunkKey,
        index: Arc<NativeBloscBlockIndex>,
    ) -> Arc<NativeBloscBlockIndex> {
        let mut entries = self
            .entries
            .lock()
            .expect("native block index cache lock poisoned");
        Arc::clone(entries.entry(key).or_insert(index))
    }

    /// Return the cached index for `key`, building it via `init` on a miss.
    ///
    /// `init` returns `Ok(None)` when the chunk does not qualify for the
    /// native path (e.g. unsupported Blosc variant); such results are *not*
    /// cached, and the caller should fall back to the generic path. Only
    /// successfully built indices are stored.
    ///
    /// The init closure runs outside the cache lock so an IO-bound build
    /// (header + table read) does not block other lookups.
    pub(crate) async fn get_or_insert_with<F, Fut>(
        &self,
        key: ChunkKey,
        init: F,
    ) -> Result<Option<Arc<NativeBloscBlockIndex>>, AccessError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<Option<Arc<NativeBloscBlockIndex>>, AccessError>>,
    {
        if let Some(hit) = self.get(key) {
            return Ok(Some(hit));
        }
        let built = init().await?;
        if let Some(index) = &built {
            self.insert(key, Arc::clone(index));
        }
        Ok(built)
    }
}

/// Build a native block index from a full encoded Blosc payload.
///
/// The function is intentionally small: all Blosc header, offset-table, split,
/// memcpyed, bitshuffle, and format validation remains inside the shared codec
/// fast path. The native access path receives only validated ranges.
pub(crate) fn build_blosc_lz4_block_index(
    encoded: &[u8],
) -> CodecResult<Option<NativeBloscBlockIndex>> {
    let Some(plan) = try_blosc_lz4_plan_from_encoded("blosc", encoded)? else {
        return Ok(None);
    };
    Ok(Some(index_from_plan(plan)))
}

/// Build a native block index from only a Blosc header + block offset table.
///
/// This is the metadata primitive needed by partial IO: read the fixed header,
/// compute the block table length, read that prefix, and build validated
/// compressed block ranges without reading the full chunk payload.
pub(crate) fn build_blosc_lz4_block_index_from_header_table(
    header_table: &[u8],
) -> CodecResult<Option<NativeBloscBlockIndex>> {
    let Some(plan) = try_blosc_lz4_plan_from_prefix("blosc", header_table)? else {
        return Ok(None);
    };
    Ok(Some(index_from_plan(plan)))
}

pub(crate) fn index_from_plan(plan: BloscLz4Plan) -> NativeBloscBlockIndex {
    let header = plan.header;
    NativeBloscBlockIndex {
        header,
        decoded_size: header.decoded_size,
        compressed_size: header.compressed_size,
        block_size: header.blocksize,
        type_size: header.typesize,
        byte_shuffled: header.is_byte_shuffled(),
        memcpyed: plan.memcpyed,
        blocks: plan.blocks.into_iter().map(block_from_range).collect(),
    }
}

fn block_from_range(range: ValidatedBloscBlockRange) -> NativeBloscBlockRange {
    NativeBloscBlockRange {
        block_idx: range.block_idx,
        payload_relative_offset: range.payload_relative_offset,
        compressed_len: range.compressed_len,
        decoded_offset: range.decoded_offset,
        decoded_len: range.decoded_len,
        leftover: range.leftover,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexes_validated_blosc_lz4_block_ranges() {
        let encoded = manual_blosc_lz4_raw_blocks(&[b"abcd", b"efgh"]);
        let index = build_blosc_lz4_block_index(&encoded)
            .expect("valid index")
            .expect("Blosc LZ4 index");

        assert!(!index.memcpyed);
        assert_eq!(index.decoded_size, 8);
        assert_eq!(index.block_size, 4);
        assert_eq!(
            index.blocks,
            vec![
                NativeBloscBlockRange {
                    block_idx: 0,
                    payload_relative_offset: 24,
                    compressed_len: 8,
                    decoded_offset: 0,
                    decoded_len: 4,
                    leftover: false,
                },
                NativeBloscBlockRange {
                    block_idx: 1,
                    payload_relative_offset: 32,
                    compressed_len: 8,
                    decoded_offset: 4,
                    decoded_len: 4,
                    leftover: false,
                },
            ]
        );
    }

    #[test]
    fn indexes_from_header_table_without_payload() {
        let encoded = manual_blosc_lz4_raw_blocks(&[b"abcd", b"efgh"]);
        let full = build_blosc_lz4_block_index(&encoded)
            .expect("valid full index")
            .expect("Blosc LZ4 index");
        let header_table_len = blosc_src::BLOSC_MIN_HEADER_LENGTH as usize + full.blocks.len() * 4;
        let prefix = &encoded[..header_table_len];
        let partial = build_blosc_lz4_block_index_from_header_table(prefix)
            .expect("valid partial index")
            .expect("Blosc LZ4 partial index");

        assert_eq!(partial, full);
    }

    #[test]
    fn indexes_memcpyed_blosc_payload_as_direct_range() {
        let encoded = manual_blosc_lz4_memcpyed(b"abcdefgh");
        let index = build_blosc_lz4_block_index(&encoded)
            .expect("valid index")
            .expect("Blosc LZ4 memcpyed index");

        assert!(index.memcpyed);
        assert_eq!(index.decoded_size, 8);
        assert_eq!(
            index.blocks,
            vec![NativeBloscBlockRange {
                block_idx: 0,
                payload_relative_offset: blosc_src::BLOSC_MAX_OVERHEAD as usize,
                compressed_len: 8,
                decoded_offset: 0,
                decoded_len: 8,
                leftover: false,
            }]
        );
    }

    #[test]
    fn indexes_memcpyed_from_header_only() {
        let encoded = manual_blosc_lz4_memcpyed(b"abcdefgh");
        let prefix = &encoded[..blosc_src::BLOSC_MIN_HEADER_LENGTH as usize];
        let index = build_blosc_lz4_block_index_from_header_table(prefix)
            .expect("valid index")
            .expect("Blosc LZ4 memcpyed index");

        assert!(index.memcpyed);
        assert_eq!(
            index.blocks[0].payload_relative_offset,
            blosc_src::BLOSC_MAX_OVERHEAD as usize
        );
        assert_eq!(index.blocks[0].compressed_len, 8);
    }

    fn manual_blosc_lz4_raw_blocks(blocks: &[&[u8]]) -> Vec<u8> {
        assert!(!blocks.is_empty());
        let blocksize = blocks[0].len();
        assert!(blocks.iter().all(|block| block.len() == blocksize));
        let decoded_size = blocks.iter().map(|block| block.len()).sum::<usize>();
        let table_bytes = blocks.len() * 4;
        let compressed_size =
            blosc_src::BLOSC_MIN_HEADER_LENGTH as usize + table_bytes + decoded_size + table_bytes;
        let mut encoded = vec![0u8; blosc_src::BLOSC_MIN_HEADER_LENGTH as usize + table_bytes];
        encoded[0] = blosc_src::BLOSC_VERSION_FORMAT as u8;
        encoded[1] = blosc_src::BLOSC_LZ4_VERSION_FORMAT as u8;
        encoded[2] = (blosc_src::BLOSC_LZ4_FORMAT << 5) as u8;
        encoded[3] = 1;
        encoded[4..8].copy_from_slice(&(decoded_size as u32).to_le_bytes());
        encoded[8..12].copy_from_slice(&(blocksize as u32).to_le_bytes());
        encoded[12..16].copy_from_slice(&(compressed_size as u32).to_le_bytes());

        let mut offset = blosc_src::BLOSC_MIN_HEADER_LENGTH as usize + table_bytes;
        for (idx, block) in blocks.iter().enumerate() {
            let table_offset = blosc_src::BLOSC_MIN_HEADER_LENGTH as usize + idx * 4;
            encoded[table_offset..table_offset + 4].copy_from_slice(&(offset as i32).to_le_bytes());
            encoded.extend_from_slice(&(block.len() as i32).to_le_bytes());
            encoded.extend_from_slice(block);
            offset += 4 + block.len();
        }
        assert_eq!(encoded.len(), compressed_size);
        encoded
    }

    fn manual_blosc_lz4_memcpyed(raw: &[u8]) -> Vec<u8> {
        let compressed_size = blosc_src::BLOSC_MAX_OVERHEAD as usize + raw.len();
        let mut encoded = vec![0u8; blosc_src::BLOSC_MAX_OVERHEAD as usize];
        encoded[0] = blosc_src::BLOSC_VERSION_FORMAT as u8;
        encoded[1] = blosc_src::BLOSC_LZ4_VERSION_FORMAT as u8;
        encoded[2] = (blosc_src::BLOSC_LZ4_FORMAT << 5) as u8 | blosc_src::BLOSC_MEMCPYED as u8;
        encoded[3] = 1;
        encoded[4..8].copy_from_slice(&(raw.len() as u32).to_le_bytes());
        encoded[8..12].copy_from_slice(&(raw.len() as u32).to_le_bytes());
        encoded[12..16].copy_from_slice(&(compressed_size as u32).to_le_bytes());
        encoded.extend_from_slice(raw);
        encoded
    }
}
