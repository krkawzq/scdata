use crate::access::{AccessError, ChunkKey, SliceShape, SliceSpec};

use super::executor::NativeBlockConsumer;
use super::load::NativeLoadRequest;
use super::metadata::{NativeBloscBlockIndex, NativeBloscBlockRange};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeSliceBlockPlan {
    pub(crate) output_len: usize,
    pub(crate) output_fully_covered: bool,
    pub(crate) reads: Vec<NativeBlockReadPlan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeBlockReadPlan {
    pub(crate) block_idx: usize,
    pub(crate) request: NativeLoadRequest,
    pub(crate) consumers: Vec<NativeBlockConsumer>,
}

pub(crate) fn plan_blosc_slice_reads(
    index: &NativeBloscBlockIndex,
    key: ChunkKey,
    slice: &SliceSpec,
    priority: u8,
    request_id_base: u64,
) -> Result<Option<NativeSliceBlockPlan>, AccessError> {
    let plan = slice.plan(index.decoded_size)?;
    let Some(ranges) = plan.ranges() else {
        return Ok(None);
    };

    // Collect (block_idx, consumer) pairs into a single flat vector instead of
    // pre-allocating one Vec per block. A chunk can hold hundreds of blocks but
    // a random slice touches only a few; the flat vec trades one sort for
    // avoiding an N-vec allocation per access.
    let mut consumers: Vec<(usize, NativeBlockConsumer)> = Vec::new();
    for range in ranges {
        if range.src_start >= range.src_end {
            continue;
        }
        let mut block_idx = first_intersecting_block(&index.blocks, range.src_start);
        while let Some(block) = index.blocks.get(block_idx) {
            if block.decoded_offset >= range.src_end {
                break;
            }
            if intersects(block, range.src_start, range.src_end) {
                consumers.push((
                    block_idx,
                    NativeBlockConsumer {
                        decoded_start: range.src_start,
                        decoded_end: range.src_end,
                        output_offset: range.dst_offset,
                    },
                ));
            }
            block_idx += 1;
        }
    }

    // Group consumers by block_idx. `scatter_loaded_blosc_block` writes each
    // consumer to a disjoint output region, so intra-block consumer order has
    // no semantic effect — an unstable sort is fine.
    consumers.sort_unstable_by_key(|(block_idx, _)| *block_idx);

    let mut reads = Vec::with_capacity(consumers.len());
    let mut cursor = 0;
    while cursor < consumers.len() {
        let first_block_idx = consumers[cursor].0;
        let block = &index.blocks[first_block_idx];
        let group_start = cursor;
        cursor += 1;
        while cursor < consumers.len() && consumers[cursor].0 == first_block_idx {
            cursor += 1;
        }
        let block_consumers = consumers[group_start..cursor]
            .iter()
            .map(|(_, consumer)| *consumer)
            .collect::<Vec<_>>();
        let request_id = request_id_base
            .checked_add(reads.len() as u64)
            .ok_or_else(|| AccessError::InvalidSlice("native request id overflow".to_string()))?;
        reads.push(NativeBlockReadPlan {
            block_idx: block.block_idx,
            request: NativeLoadRequest {
                id: request_id,
                file: key.file,
                offset: key.offset + block.payload_relative_offset as u64,
                len: block.compressed_len,
                priority,
            },
            consumers: block_consumers,
        });
    }

    Ok(Some(NativeSliceBlockPlan {
        output_len: plan.output_len,
        output_fully_covered: matches!(plan.shape, SliceShape::Sequential),
        reads,
    }))
}

fn first_intersecting_block(blocks: &[NativeBloscBlockRange], start: usize) -> usize {
    let mut lo = 0usize;
    let mut hi = blocks.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let block = &blocks[mid];
        if block.decoded_offset + block.decoded_len <= start {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn intersects(block: &NativeBloscBlockRange, start: usize, end: usize) -> bool {
    if start >= end {
        return false;
    }
    let block_start = block.decoded_offset;
    let block_end = block_start + block.decoded_len;
    start < block_end && end > block_start
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::databank::native::metadata::build_blosc_lz4_block_index;

    #[test]
    fn plans_block_reads_for_scattered_slice() {
        let encoded = manual_blosc_lz4_raw_blocks(&[b"abcd", b"efgh"]);
        let index = build_blosc_lz4_block_index(&encoded)
            .expect("valid index")
            .expect("Blosc LZ4 index");
        let slice = SliceSpec::from_triples(vec![0, 1, 3, 4, 5, 7]).expect("slice");
        let key = ChunkKey::new(crate::access::FileRef::new(9), 1000, encoded.len());

        let plan = plan_blosc_slice_reads(&index, key, &slice, 3, 42)
            .expect("plan")
            .expect("partial plan");

        assert_eq!(plan.output_len, 6);
        assert!(!plan.output_fully_covered);
        assert_eq!(plan.reads.len(), 2);
        assert_eq!(plan.reads[0].block_idx, 0);
        assert_eq!(plan.reads[0].request.id, 42);
        assert_eq!(plan.reads[0].request.offset, 1024);
        assert_eq!(plan.reads[0].consumers.len(), 1);
        assert_eq!(plan.reads[1].block_idx, 1);
        assert_eq!(plan.reads[1].request.id, 43);
        assert_eq!(plan.reads[1].request.offset, 1032);
        assert_eq!(plan.reads[1].consumers.len(), 1);
    }

    #[test]
    fn marks_sequential_scatter_output_as_fully_covered() {
        let encoded = manual_blosc_lz4_raw_blocks(&[b"abcd", b"efgh"]);
        let index = build_blosc_lz4_block_index(&encoded)
            .expect("valid index")
            .expect("Blosc LZ4 index");
        let slice = SliceSpec::from_triples(vec![0, 1, 3, 2, 5, 7]).expect("slice");
        let key = ChunkKey::new(crate::access::FileRef::new(9), 1000, encoded.len());

        let plan = plan_blosc_slice_reads(&index, key, &slice, 3, 42)
            .expect("plan")
            .expect("partial plan");

        assert_eq!(plan.output_len, 4);
        assert!(plan.output_fully_covered);
    }

    #[test]
    fn full_slice_does_not_plan_partial_reads() {
        let encoded = manual_blosc_lz4_raw_blocks(&[b"abcd", b"efgh"]);
        let index = build_blosc_lz4_block_index(&encoded)
            .expect("valid index")
            .expect("Blosc LZ4 index");
        let key = ChunkKey::new(crate::access::FileRef::new(9), 1000, encoded.len());

        let plan = plan_blosc_slice_reads(&index, key, &SliceSpec::Full, 3, 42).expect("plan");

        assert!(plan.is_none());
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
}
