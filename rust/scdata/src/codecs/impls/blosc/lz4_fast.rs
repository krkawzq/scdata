use std::cell::RefCell;

use super::super::super::util::{decode_error, output_too_small, verify_size};
use super::super::super::CodecResult;
use super::super::lz4_decompress_raw_into;
use super::header::BloscHeader;
use super::shuffle::unshuffle_bytes;

thread_local! {
    static BLOSC_LZ4_SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
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
        return Some(Err(decode_error(codec, "unsupported Blosc future flags")));
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
    let bstarts_bytes = nblocks
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(blosc_src::BLOSC_MIN_HEADER_LENGTH as usize))
        .ok_or_else(|| decode_error(codec, "Blosc block table overflow"))?;
    if bstarts_bytes > header.compressed_size {
        return Err(decode_error(codec, "invalid Blosc block table"));
    }

    BLOSC_LZ4_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        if header.is_byte_shuffled() && scratch.len() < header.blocksize {
            scratch.resize(header.blocksize, 0);
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
            let bstart_offset = blosc_src::BLOSC_MIN_HEADER_LENGTH as usize + block_idx * 4;
            let src_offset = read_i32_le(&encoded[bstart_offset..bstart_offset + 4], codec)?;
            let src_offset = usize::try_from(src_offset)
                .map_err(|_| decode_error(codec, "negative Blosc block offset"))?;
            let output_block = &mut output[decoded_offset..decoded_offset + bsize];

            if header.is_byte_shuffled() {
                let scratch_block = &mut scratch[..bsize];
                decode_blosc_lz4_block(
                    codec,
                    encoded,
                    header,
                    bsize,
                    is_last && leftover > 0,
                    src_offset,
                    scratch_block,
                )?;
                unshuffle_bytes(header.typesize, scratch_block, output_block);
            } else {
                decode_blosc_lz4_block(
                    codec,
                    encoded,
                    header,
                    bsize,
                    is_last && leftover > 0,
                    src_offset,
                    output_block,
                )?;
            }
            decoded_offset += bsize;
        }

        Ok(header.decoded_size)
    })
}

fn decode_blosc_lz4_block(
    codec: &str,
    encoded: &[u8],
    header: BloscHeader,
    blocksize: usize,
    leftover_block: bool,
    mut src_offset: usize,
    output: &mut [u8],
) -> CodecResult<()> {
    let nsplits = if !header.dont_split()
        && header.typesize <= 16
        && header.blocksize / header.typesize >= 128
        && !leftover_block
    {
        header.typesize
    } else {
        1
    };
    let split_size = split_size(codec, blocksize, nsplits)?;
    let mut output_offset = 0usize;
    for _ in 0..nsplits {
        if src_offset + 4 > header.compressed_size {
            return Err(decode_error(codec, "invalid Blosc split offset"));
        }
        let compressed_size = read_i32_le(&encoded[src_offset..src_offset + 4], codec)?;
        src_offset += 4;
        let compressed_size = usize::try_from(compressed_size)
            .map_err(|_| decode_error(codec, "negative Blosc split size"))?;
        if compressed_size > header.compressed_size - src_offset {
            return Err(decode_error(codec, "invalid Blosc split size"));
        }

        let next_output_offset = output_offset
            .checked_add(split_size)
            .ok_or_else(|| decode_error(codec, "Blosc split output overflow"))?;
        if next_output_offset > output.len() {
            return Err(decode_error(codec, "invalid Blosc split output size"));
        }

        let split_output = &mut output[output_offset..next_output_offset];
        if compressed_size == split_size {
            split_output.copy_from_slice(&encoded[src_offset..src_offset + compressed_size]);
        } else {
            lz4_decompress_raw_into(
                codec,
                &encoded[src_offset..src_offset + compressed_size],
                split_output,
            )?;
        }
        src_offset += compressed_size;
        output_offset = next_output_offset;
    }
    if output_offset != blocksize {
        return Err(decode_error(codec, "invalid Blosc split block size"));
    }
    Ok(())
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
}
