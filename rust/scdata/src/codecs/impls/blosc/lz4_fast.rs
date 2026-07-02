use std::cell::RefCell;

use super::super::super::buffer::set_vec_len_for_decode;
use super::super::super::spec::DecodeSlice;
use super::super::super::util::{
    decode_error, output_too_small, reserve_decode_buffer, verify_size,
};
use super::super::super::CodecResult;
use super::super::{lz4_decompress_raw_into, lz4_decompress_raw_partial_into};
use super::header::BloscHeader;
use super::shuffle::unshuffle_bytes;

thread_local! {
    static BLOSC_LZ4_SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static BLOSC_LZ4_BLOCK: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

#[derive(Debug, Clone)]
pub(crate) struct BloscLz4Plan {
    pub(crate) header: BloscHeader,
    pub(crate) blocks: Vec<ValidatedBloscBlockRange>,
    pub(crate) memcpyed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ValidatedBloscBlockRange {
    pub(crate) block_idx: usize,
    pub(crate) payload_relative_offset: usize,
    pub(crate) compressed_len: usize,
    pub(crate) decoded_offset: usize,
    pub(crate) decoded_len: usize,
    pub(crate) leftover: bool,
}

impl ValidatedBloscBlockRange {
    fn as_block(self) -> BloscLz4Block {
        BloscLz4Block {
            size: self.decoded_len,
            leftover: self.leftover,
            src_offset: self.payload_relative_offset,
            src_limit: self.payload_relative_offset + self.compressed_len,
        }
    }
}

pub(super) fn try_blosc_lz4_decode_into(
    codec: &str,
    encoded: &[u8],
    header: BloscHeader,
    output: &mut [u8],
    expected_size: Option<usize>,
) -> Option<CodecResult<usize>> {
    if header.compformat() != blosc_src::BLOSC_LZ4_FORMAT as u8 {
        return None;
    }
    if header.compversion != blosc_src::BLOSC_LZ4_VERSION_FORMAT as u8 {
        return Some(Err(decode_error(
            codec,
            "unsupported Blosc LZ4 format version",
        )));
    }
    if header.flags & 0x08 != 0 {
        return None;
    }
    if header.is_bit_shuffled() {
        return None;
    }

    Some(blosc_lz4_decode_into(
        codec,
        encoded,
        header,
        output,
        expected_size,
    ))
}

pub(super) fn try_blosc_lz4_decode_slice(
    codec: &str,
    encoded: &[u8],
    header: BloscHeader,
    slice: &DecodeSlice,
) -> CodecResult<Option<Vec<u8>>> {
    let Some(plan) = try_blosc_lz4_plan(codec, encoded, header)? else {
        return Ok(None);
    };
    validate_decode_slice(codec, header.decoded_size, slice)?;
    if slice.output_len == 0 {
        return Ok(Some(Vec::new()));
    }
    if header.decoded_size == 0 {
        return Ok(Some(vec![0u8; slice.output_len]));
    }

    if plan.memcpyed {
        let source = memcpyed_source(codec, encoded, header)?;
        return Ok(Some(materialize_decode_slice(codec, source, slice)?));
    }

    let touched = touched_blocks(codec, header, slice, plan.blocks.len())?;
    let touched_count = touched.iter().filter(|&&touched| touched).count();
    if touched_count == 0 {
        return Ok(Some(vec![0u8; slice.output_len]));
    }
    if touched_count == plan.blocks.len() {
        return Ok(None);
    }

    BLOSC_LZ4_SCRATCH.with(|scratch| {
        BLOSC_LZ4_BLOCK.with(|block_scratch| {
            let mut scratch = scratch.borrow_mut();
            let mut block_scratch = block_scratch.borrow_mut();
            if scratch.capacity() < header.blocksize {
                let additional = header.blocksize - scratch.capacity();
                reserve_decode_buffer(codec, &mut scratch, additional)?;
            }
            if !header.is_byte_shuffled() && block_scratch.capacity() < header.blocksize {
                let additional = header.blocksize - block_scratch.capacity();
                reserve_decode_buffer(codec, &mut block_scratch, additional)?;
            }
            if scratch.len() < header.blocksize {
                set_vec_len_for_decode(&mut scratch, header.blocksize);
            }
            if !header.is_byte_shuffled() && block_scratch.len() < header.blocksize {
                set_vec_len_for_decode(&mut block_scratch, header.blocksize);
            }

            let mut out = vec![0u8; slice.output_len];
            for (block, &is_touched) in plan.blocks.iter().zip(touched.iter()) {
                if !is_touched {
                    continue;
                }
                if header.is_byte_shuffled() {
                    let shuffled_block = &mut scratch[..block.decoded_len];
                    decode_blosc_lz4_block(
                        codec,
                        encoded,
                        header,
                        block.as_block(),
                        shuffled_block,
                    )?;
                    copy_shuffled_block_ranges(
                        header.typesize,
                        shuffled_block,
                        block.decoded_offset,
                        slice,
                        &mut out,
                    );
                } else {
                    let decoded_block = &mut block_scratch[..block.decoded_len];
                    decode_blosc_lz4_block(
                        codec,
                        encoded,
                        header,
                        block.as_block(),
                        decoded_block,
                    )?;
                    copy_block_ranges(decoded_block, block.decoded_offset, slice, &mut out);
                }
            }
            Ok(Some(out))
        })
    })
}

pub(crate) fn try_blosc_lz4_plan(
    codec: &str,
    encoded: &[u8],
    header: BloscHeader,
) -> CodecResult<Option<BloscLz4Plan>> {
    try_blosc_lz4_plan_inner(codec, encoded, header, true)
}

pub(crate) fn try_blosc_lz4_plan_from_header_table(
    codec: &str,
    header_table: &[u8],
    header: BloscHeader,
) -> CodecResult<Option<BloscLz4Plan>> {
    try_blosc_lz4_plan_inner(codec, header_table, header, false)
}

pub(crate) fn blosc_lz4_header_table_len(
    codec: &str,
    header: BloscHeader,
) -> CodecResult<Option<usize>> {
    if header.compformat() != blosc_src::BLOSC_LZ4_FORMAT as u8 {
        return Ok(None);
    }
    if header.compversion != blosc_src::BLOSC_LZ4_VERSION_FORMAT as u8 {
        return Err(decode_error(codec, "unsupported Blosc LZ4 format version"));
    }
    if header.flags & 0x08 != 0 || header.is_bit_shuffled() {
        return Ok(None);
    }
    if header.decoded_size == 0 || header.is_memcpyed() {
        return Ok(Some(blosc_src::BLOSC_MIN_HEADER_LENGTH as usize));
    }

    validate_lz4_header(codec, header)?;
    let nblocks = header.decoded_size.div_ceil(header.blocksize);
    let table_len = block_table_bytes(codec, nblocks)?;
    if table_len > header.compressed_size {
        return Err(decode_error(codec, "invalid Blosc block table"));
    }
    Ok(Some(table_len))
}

fn try_blosc_lz4_plan_inner(
    codec: &str,
    encoded: &[u8],
    header: BloscHeader,
    require_payload: bool,
) -> CodecResult<Option<BloscLz4Plan>> {
    if header.compformat() != blosc_src::BLOSC_LZ4_FORMAT as u8 {
        return Ok(None);
    }
    if header.compversion != blosc_src::BLOSC_LZ4_VERSION_FORMAT as u8 {
        return Err(decode_error(codec, "unsupported Blosc LZ4 format version"));
    }
    if header.flags & 0x08 != 0 || header.is_bit_shuffled() {
        return Ok(None);
    }
    if header.decoded_size == 0 {
        return Ok(Some(BloscLz4Plan {
            header,
            blocks: Vec::new(),
            memcpyed: false,
        }));
    }
    if header.is_memcpyed() {
        if require_payload {
            let _ = memcpyed_source(codec, encoded, header)?;
        } else if header.decoded_size + blosc_src::BLOSC_MAX_OVERHEAD as usize
            != header.compressed_size
        {
            return Err(decode_error(codec, "invalid Blosc memcpy buffer"));
        }
        return Ok(Some(BloscLz4Plan {
            header,
            blocks: vec![ValidatedBloscBlockRange {
                block_idx: 0,
                payload_relative_offset: blosc_src::BLOSC_MAX_OVERHEAD as usize,
                compressed_len: header.decoded_size,
                decoded_offset: 0,
                decoded_len: header.decoded_size,
                leftover: false,
            }],
            memcpyed: true,
        }));
    }

    validate_lz4_header(codec, header)?;
    let nblocks = header.decoded_size.div_ceil(header.blocksize);
    let bstarts_bytes = block_table_bytes(codec, nblocks)?;
    if bstarts_bytes > header.compressed_size {
        return Err(decode_error(codec, "invalid Blosc block table"));
    }
    if encoded.len() < bstarts_bytes {
        return Err(decode_error(
            codec,
            "buffer is shorter than a Blosc block table",
        ));
    }

    let leftover = header.decoded_size % header.blocksize;
    let mut blocks = Vec::with_capacity(nblocks);
    let mut decoded_offset = 0usize;
    let mut previous_src_offset = None;
    for block_idx in 0..nblocks {
        let is_last = block_idx + 1 == nblocks;
        let bsize = if is_last && leftover > 0 {
            leftover
        } else {
            header.blocksize
        };
        let src_offset = block_start(codec, encoded, block_idx)?;
        if src_offset < bstarts_bytes {
            return Err(decode_error(codec, "invalid Blosc block offset"));
        }
        if let Some(previous) = previous_src_offset {
            if src_offset < previous {
                return Err(decode_error(codec, "invalid Blosc block offset"));
            }
        }
        let src_limit = if is_last {
            header.compressed_size
        } else {
            block_start(codec, encoded, block_idx + 1)?
        };
        if src_offset >= src_limit || src_limit > header.compressed_size {
            return Err(decode_error(codec, "invalid Blosc block offset"));
        }
        blocks.push(ValidatedBloscBlockRange {
            block_idx,
            payload_relative_offset: src_offset,
            compressed_len: src_limit - src_offset,
            decoded_offset,
            decoded_len: bsize,
            leftover: is_last && leftover > 0,
        });
        previous_src_offset = Some(src_offset);
        decoded_offset += bsize;
    }
    Ok(Some(BloscLz4Plan {
        header,
        blocks,
        memcpyed: false,
    }))
}

fn blosc_lz4_decode_into(
    codec: &str,
    encoded: &[u8],
    header: BloscHeader,
    output: &mut [u8],
    expected_size: Option<usize>,
) -> CodecResult<usize> {
    verify_size(codec, header.decoded_size, expected_size)?;
    if output.len() < header.decoded_size {
        return Err(output_too_small(codec, header.decoded_size, output.len()));
    }
    if header.decoded_size == 0 {
        return Ok(0);
    }
    if header.blocksize == 0
        || header.blocksize > header.decoded_size
        || header.blocksize > blosc_src::BLOSC_MAX_BLOCKSIZE as usize
        || header.typesize == 0
        || header.typesize > blosc_src::BLOSC_MAX_TYPESIZE as usize
    {
        return Err(decode_error(codec, "invalid Blosc LZ4 header"));
    }

    if header.is_memcpyed() {
        if header.decoded_size + blosc_src::BLOSC_MAX_OVERHEAD as usize != header.compressed_size {
            return Err(decode_error(codec, "invalid Blosc memcpy buffer"));
        }
        let source = &encoded[blosc_src::BLOSC_MAX_OVERHEAD as usize
            ..blosc_src::BLOSC_MAX_OVERHEAD as usize + header.decoded_size];
        output[..header.decoded_size].copy_from_slice(source);
        return Ok(header.decoded_size);
    }

    let nblocks = header.decoded_size.div_ceil(header.blocksize);
    let bstarts_bytes = block_table_bytes(codec, nblocks)?;
    if bstarts_bytes > header.compressed_size {
        return Err(decode_error(codec, "invalid Blosc block table"));
    }

    BLOSC_LZ4_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        if header.is_byte_shuffled() {
            if scratch.capacity() < header.blocksize {
                let additional = header.blocksize - scratch.capacity();
                reserve_decode_buffer(codec, &mut scratch, additional)?;
            }
            if scratch.len() < header.blocksize {
                set_vec_len_for_decode(&mut scratch, header.blocksize);
            }
        }

        let leftover = header.decoded_size % header.blocksize;
        let mut decoded_offset = 0usize;
        for block_idx in 0..nblocks {
            let is_last = block_idx + 1 == nblocks;
            let bsize = if is_last && leftover > 0 {
                leftover
            } else {
                header.blocksize
            };
            let src_offset = block_start(codec, encoded, block_idx)?;
            if src_offset < bstarts_bytes {
                return Err(decode_error(codec, "invalid Blosc block offset"));
            }
            let src_limit = if is_last {
                header.compressed_size
            } else {
                block_start(codec, encoded, block_idx + 1)?
            };
            if src_offset >= src_limit || src_limit > header.compressed_size {
                return Err(decode_error(codec, "invalid Blosc block offset"));
            }
            let output_block = &mut output[decoded_offset..decoded_offset + bsize];
            let block = BloscLz4Block {
                size: bsize,
                leftover: is_last && leftover > 0,
                src_offset,
                src_limit,
            };

            if header.is_byte_shuffled() {
                let scratch_block = &mut scratch[..bsize];
                decode_blosc_lz4_block(codec, encoded, header, block, scratch_block)?;
                unshuffle_bytes(header.typesize, scratch_block, output_block);
            } else {
                decode_blosc_lz4_block(codec, encoded, header, block, output_block)?;
            }
            decoded_offset += bsize;
        }

        Ok(header.decoded_size)
    })
}

fn validate_lz4_header(codec: &str, header: BloscHeader) -> CodecResult<()> {
    if header.blocksize == 0
        || header.blocksize > header.decoded_size
        || header.blocksize > blosc_src::BLOSC_MAX_BLOCKSIZE as usize
        || header.typesize == 0
        || header.typesize > blosc_src::BLOSC_MAX_TYPESIZE as usize
    {
        return Err(decode_error(codec, "invalid Blosc LZ4 header"));
    }
    Ok(())
}

fn block_table_bytes(codec: &str, nblocks: usize) -> CodecResult<usize> {
    nblocks
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(blosc_src::BLOSC_MIN_HEADER_LENGTH as usize))
        .ok_or_else(|| decode_error(codec, "Blosc block table overflow"))
}

fn validate_decode_slice(codec: &str, decoded_size: usize, slice: &DecodeSlice) -> CodecResult<()> {
    for range in slice.ranges.iter().copied() {
        let Some(dst_end) = range.dst_offset.checked_add(range.len()) else {
            return Err(decode_error(codec, "invalid decode slice range"));
        };
        if range.src_start > range.src_end
            || range.src_end > decoded_size
            || dst_end > slice.output_len
        {
            return Err(decode_error(codec, "invalid decode slice range"));
        }
    }
    Ok(())
}

fn touched_blocks(
    codec: &str,
    header: BloscHeader,
    slice: &DecodeSlice,
    nblocks: usize,
) -> CodecResult<Vec<bool>> {
    let mut touched = vec![false; nblocks];
    for range in slice.ranges.iter().copied() {
        if range.src_start == range.src_end {
            continue;
        }
        let first = range.src_start / header.blocksize;
        let last = (range.src_end - 1) / header.blocksize;
        if first >= nblocks || last >= nblocks {
            return Err(decode_error(codec, "invalid decode slice block range"));
        }
        for slot in &mut touched[first..=last] {
            *slot = true;
        }
    }
    Ok(touched)
}

fn materialize_decode_slice(
    codec: &str,
    decoded: &[u8],
    slice: &DecodeSlice,
) -> CodecResult<Vec<u8>> {
    validate_decode_slice(codec, decoded.len(), slice)?;
    let mut out = vec![0u8; slice.output_len];
    for range in slice.ranges.iter().copied() {
        let dst_end = range.dst_offset + range.len();
        out[range.dst_offset..dst_end].copy_from_slice(&decoded[range.src_start..range.src_end]);
    }
    Ok(out)
}

fn memcpyed_source<'a>(
    codec: &str,
    encoded: &'a [u8],
    header: BloscHeader,
) -> CodecResult<&'a [u8]> {
    if header.decoded_size + blosc_src::BLOSC_MAX_OVERHEAD as usize != header.compressed_size {
        return Err(decode_error(codec, "invalid Blosc memcpy buffer"));
    }
    Ok(&encoded[blosc_src::BLOSC_MAX_OVERHEAD as usize
        ..blosc_src::BLOSC_MAX_OVERHEAD as usize + header.decoded_size])
}

fn copy_block_ranges(block: &[u8], block_offset: usize, slice: &DecodeSlice, out: &mut [u8]) {
    let block_end = block_offset + block.len();
    for range in slice.ranges.iter().copied() {
        let start = range.src_start.max(block_offset);
        let end = range.src_end.min(block_end);
        if start >= end {
            continue;
        }
        let dst_start = range.dst_offset + (start - range.src_start);
        let dst_end = dst_start + (end - start);
        out[dst_start..dst_end].copy_from_slice(&block[start - block_offset..end - block_offset]);
    }
}

fn copy_shuffled_block_ranges(
    typesize: usize,
    shuffled: &[u8],
    block_offset: usize,
    slice: &DecodeSlice,
    out: &mut [u8],
) {
    if typesize <= 1 {
        copy_block_ranges(shuffled, block_offset, slice, out);
        return;
    }
    let elements = shuffled.len() / typesize;
    let main_len = elements * typesize;
    let block_end = block_offset + shuffled.len();
    for range in slice.ranges.iter().copied() {
        let start = range.src_start.max(block_offset);
        let end = range.src_end.min(block_end);
        if start >= end {
            continue;
        }
        let dst_start = range.dst_offset + (start - range.src_start);
        for (pos, dst) in (start..end).zip(dst_start..) {
            let rel = pos - block_offset;
            let src = if rel < main_len {
                let element = rel / typesize;
                let byte = rel % typesize;
                byte * elements + element
            } else {
                rel
            };
            out[dst] = shuffled[src];
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BloscLz4Block {
    pub(crate) size: usize,
    pub(crate) leftover: bool,
    pub(crate) src_offset: usize,
    pub(crate) src_limit: usize,
}

pub(crate) fn decode_blosc_lz4_block(
    codec: &str,
    encoded: &[u8],
    header: BloscHeader,
    block: BloscLz4Block,
    output: &mut [u8],
) -> CodecResult<()> {
    decode_blosc_lz4_block_inner(codec, encoded, header, block, None, output)
}

pub(crate) fn decode_blosc_lz4_block_partial_prefixes(
    codec: &str,
    encoded: &[u8],
    header: BloscHeader,
    block: BloscLz4Block,
    split_prefixes: &[usize],
    output: &mut [u8],
) -> CodecResult<()> {
    decode_blosc_lz4_block_inner(codec, encoded, header, block, Some(split_prefixes), output)
}

pub(crate) fn blosc_lz4_block_split_count(header: BloscHeader, block: BloscLz4Block) -> usize {
    if !header.dont_split()
        && header.typesize <= 16
        && header.blocksize / header.typesize >= 128
        && !block.leftover
    {
        header.typesize
    } else {
        1
    }
}

fn decode_blosc_lz4_block_inner(
    codec: &str,
    encoded: &[u8],
    header: BloscHeader,
    block: BloscLz4Block,
    split_prefixes: Option<&[usize]>,
    output: &mut [u8],
) -> CodecResult<()> {
    let mut src_offset = block.src_offset;
    let nsplits = blosc_lz4_block_split_count(header, block);
    if let Some(prefixes) = split_prefixes {
        if prefixes.len() != nsplits {
            return Err(decode_error(
                codec,
                "invalid Blosc partial split prefix count",
            ));
        }
    }
    let split_size = split_size(codec, block.size, nsplits)?;
    let mut output_offset = 0usize;
    for split_idx in 0..nsplits {
        if src_offset + 4 > block.src_limit {
            return Err(decode_error(codec, "invalid Blosc split offset"));
        }
        let compressed_size = read_i32_le(&encoded[src_offset..src_offset + 4], codec)?;
        src_offset += 4;
        let compressed_size = usize::try_from(compressed_size)
            .map_err(|_| decode_error(codec, "negative Blosc split size"))?;
        if compressed_size > block.src_limit - src_offset {
            return Err(decode_error(codec, "invalid Blosc split size"));
        }

        let next_output_offset = output_offset
            .checked_add(split_size)
            .ok_or_else(|| decode_error(codec, "Blosc split output overflow"))?;
        if next_output_offset > output.len() {
            return Err(decode_error(codec, "invalid Blosc split output size"));
        }

        let requested_prefix = split_prefixes
            .map(|prefixes| prefixes[split_idx])
            .unwrap_or(split_size);
        if requested_prefix > split_size {
            return Err(decode_error(codec, "invalid Blosc partial split prefix"));
        }
        if requested_prefix > 0 {
            let split_output = &mut output[output_offset..output_offset + requested_prefix];
            if compressed_size == split_size {
                split_output.copy_from_slice(&encoded[src_offset..src_offset + requested_prefix]);
            } else if requested_prefix == split_size {
                lz4_decompress_raw_into(
                    codec,
                    &encoded[src_offset..src_offset + compressed_size],
                    split_output,
                )?;
            } else {
                lz4_decompress_raw_partial_into(
                    codec,
                    &encoded[src_offset..src_offset + compressed_size],
                    split_output,
                    split_size,
                )?;
            }
        }
        src_offset += compressed_size;
        output_offset = next_output_offset;
    }
    if src_offset != block.src_limit {
        return Err(decode_error(codec, "invalid Blosc block size"));
    }
    if output_offset != block.size {
        return Err(decode_error(codec, "invalid Blosc split block size"));
    }
    Ok(())
}

fn block_start(codec: &str, encoded: &[u8], block_idx: usize) -> CodecResult<usize> {
    let bstart_offset = blosc_src::BLOSC_MIN_HEADER_LENGTH as usize + block_idx * 4;
    let src_offset = read_i32_le(&encoded[bstart_offset..bstart_offset + 4], codec)?;
    usize::try_from(src_offset).map_err(|_| decode_error(codec, "negative Blosc block offset"))
}

fn split_size(codec: &str, blocksize: usize, nsplits: usize) -> CodecResult<usize> {
    if nsplits == 0 {
        return Err(decode_error(codec, "invalid Blosc split count"));
    }
    let split_size = blocksize / nsplits;
    if split_size * nsplits != blocksize {
        return Err(decode_error(codec, "invalid Blosc split block size"));
    }
    Ok(split_size)
}

fn read_i32_le(bytes: &[u8], codec: &str) -> CodecResult<i32> {
    let bytes: [u8; 4] = bytes
        .try_into()
        .map_err(|_| decode_error(codec, "short Blosc integer"))?;
    Ok(i32::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_size_rejects_non_divisible_full_block() {
        let err = split_size("blosc", 385, 3).expect_err("non-divisible split");
        assert!(err.to_string().contains("split block size"));
    }

    #[test]
    fn split_size_accepts_divisible_full_block() {
        assert_eq!(split_size("blosc", 384, 3).expect("split size"), 128);
    }

    #[test]
    fn fast_path_skips_all_bitshuffle_headers() {
        let header = BloscHeader {
            compversion: blosc_src::BLOSC_LZ4_VERSION_FORMAT as u8,
            flags: (blosc_src::BLOSC_LZ4_FORMAT << 5) as u8 | blosc_src::BLOSC_DOBITSHUFFLE as u8,
            typesize: 8,
            decoded_size: 4,
            blocksize: 4,
            compressed_size: blosc_src::BLOSC_MIN_HEADER_LENGTH as usize,
        };
        let mut output = [0u8; 4];

        assert!(try_blosc_lz4_decode_into(
            "blosc",
            &[0u8; blosc_src::BLOSC_MIN_HEADER_LENGTH as usize],
            header,
            &mut output,
            Some(4),
        )
        .is_none());
    }

    #[test]
    fn fast_path_falls_back_for_unknown_flags() {
        let header = BloscHeader {
            compversion: blosc_src::BLOSC_LZ4_VERSION_FORMAT as u8,
            flags: (blosc_src::BLOSC_LZ4_FORMAT << 5) as u8 | 0x08,
            typesize: 1,
            decoded_size: 4,
            blocksize: 4,
            compressed_size: blosc_src::BLOSC_MIN_HEADER_LENGTH as usize,
        };
        let mut output = [0u8; 4];

        assert!(try_blosc_lz4_decode_into(
            "blosc",
            &[0u8; blosc_src::BLOSC_MIN_HEADER_LENGTH as usize],
            header,
            &mut output,
            Some(4),
        )
        .is_none());
    }
}
