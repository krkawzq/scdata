mod header;
mod lz4_fast;
mod shuffle;

use std::os::raw::c_void;

use super::super::buffer::{set_vec_len_for_decode, DecodeBuffer};
use super::super::spec::{sealed, ChunkCodec, CodecCacheKey, DecodeSlice};
use super::super::util::{
    decode_error, output_too_small, reserve_decode_buffer, vec_with_decode_capacity, verify_size,
};
use super::super::CodecResult;
use header::{blosc_decoded_size, blosc_header, blosc_header_prefix};
use lz4_fast::try_blosc_lz4_decode_into;

pub(crate) use header::BloscHeader;
pub(crate) use lz4_fast::{
    blosc_lz4_block_split_count, decode_blosc_lz4_block, decode_blosc_lz4_block_partial_prefixes,
    BloscLz4Block, BloscLz4Plan, ValidatedBloscBlockRange,
};
pub(crate) use shuffle::unshuffle_bytes;

pub(crate) fn try_blosc_lz4_plan_from_encoded(
    codec: &str,
    encoded: &[u8],
) -> CodecResult<Option<BloscLz4Plan>> {
    let header = blosc_header(codec, encoded)?;
    lz4_fast::try_blosc_lz4_plan(codec, encoded, header)
}

pub(crate) fn try_blosc_lz4_plan_from_prefix(
    codec: &str,
    header_table: &[u8],
) -> CodecResult<Option<BloscLz4Plan>> {
    let header = blosc_header_prefix(codec, header_table)?;
    lz4_fast::try_blosc_lz4_plan_from_header_table(codec, header_table, header)
}

pub(crate) fn blosc_lz4_header_table_len_from_prefix(
    codec: &str,
    header_bytes: &[u8],
) -> CodecResult<Option<usize>> {
    let header = blosc_header_prefix(codec, header_bytes)?;
    lz4_fast::blosc_lz4_header_table_len(codec, header)
}

#[derive(Debug)]
pub(crate) struct BloscCodec;

impl sealed::Sealed for BloscCodec {}

impl ChunkCodec for BloscCodec {
    fn name(&self) -> &str {
        "blosc"
    }

    fn cache_key(&self) -> CodecCacheKey {
        CodecCacheKey::Static("blosc")
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        let header = blosc_header(self.name(), encoded)?;
        let decoded_size = header.decoded_size;
        verify_size(self.name(), decoded_size, expected_size)?;

        let mut decoded = vec_with_decode_capacity(self.name(), decoded_size)?;
        set_vec_len_for_decode(&mut decoded, decoded_size);
        let written = blosc_decode_header_into_output(self.name(), encoded, header, &mut decoded)?;
        decoded.truncate(written);
        Ok(decoded)
    }

    fn decode_slice(
        &self,
        encoded: &[u8],
        slice: &DecodeSlice,
        expected_size: Option<usize>,
    ) -> CodecResult<Option<Vec<u8>>> {
        let header = blosc_header(self.name(), encoded)?;
        verify_size(self.name(), header.decoded_size, expected_size)?;
        lz4_fast::try_blosc_lz4_decode_slice(self.name(), encoded, header, slice)
    }

    fn decoded_size_hint(
        &self,
        encoded: &[u8],
        _expected_size: Option<usize>,
    ) -> CodecResult<Option<usize>> {
        blosc_decoded_size(self.name(), encoded).map(Some)
    }

    fn decode_into(
        &self,
        encoded: &[u8],
        mut output: DecodeBuffer<'_>,
        expected_size: Option<usize>,
    ) -> CodecResult<usize> {
        let header = blosc_header(self.name(), encoded)?;
        let decoded_size = header.decoded_size;
        verify_size(self.name(), decoded_size, expected_size)?;
        blosc_decode_header_into_output(self.name(), encoded, header, output.as_mut_slice())
    }

    fn decode_to_vec(
        &self,
        encoded: &[u8],
        mut output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        let header = blosc_header(self.name(), encoded)?;
        let decoded_size = header.decoded_size;
        verify_size(self.name(), decoded_size, expected_size)?;
        if output.capacity() < decoded_size {
            return Err(output_too_small(
                self.name(),
                decoded_size,
                output.capacity(),
            ));
        }
        set_vec_len_for_decode(&mut output, decoded_size);

        let written = blosc_decode_header_into_output(self.name(), encoded, header, &mut output)?;
        output.truncate(written);
        Ok(output)
    }

    fn decode_to_vec_grow(
        &self,
        encoded: &[u8],
        mut output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        let header = blosc_header(self.name(), encoded)?;
        let decoded_size = header.decoded_size;
        verify_size(self.name(), decoded_size, expected_size)?;
        output.clear();
        if output.capacity() < decoded_size {
            let additional = decoded_size - output.capacity();
            reserve_decode_buffer(self.name(), &mut output, additional)?;
        }
        set_vec_len_for_decode(&mut output, decoded_size);

        let written = blosc_decode_header_into_output(self.name(), encoded, header, &mut output)?;
        output.truncate(written);
        Ok(output)
    }

    fn decode_to_capacity_vec(
        &self,
        encoded: &[u8],
        output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        self.decode_to_vec(encoded, output, expected_size)
    }
}

fn blosc_decode_header_into_output(
    codec: &str,
    encoded: &[u8],
    header: header::BloscHeader,
    output: &mut [u8],
) -> CodecResult<usize> {
    let decoded_size = header.decoded_size;
    if output.len() < decoded_size {
        return Err(output_too_small(codec, decoded_size, output.len()));
    }
    let output = &mut output[..decoded_size];

    if let Some(result) = try_blosc_lz4_decode_into(codec, encoded, header, output, None) {
        return result;
    }

    let written = unsafe {
        blosc_src::blosc_decompress_ctx(
            encoded.as_ptr().cast::<c_void>(),
            output.as_mut_ptr().cast::<c_void>(),
            decoded_size,
            1,
        )
    };
    if written < 0 {
        return Err(decode_error(
            codec,
            format!("Blosc decompressor returned {written}"),
        ));
    }
    if written as usize != decoded_size {
        return Err(decode_error(
            codec,
            format!("expected Blosc to write {decoded_size} bytes, wrote {written}"),
        ));
    }

    Ok(decoded_size)
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::sync::Arc;

    use super::*;
    use crate::codecs::{DecodeRange, DecodeSlice};

    #[test]
    fn lz4_fast_path_decodes_valid_blosc_lz4_buffers() {
        let codec = BloscCodec;
        let raw = (0..4096)
            .flat_map(|value: u32| value.to_le_bytes())
            .collect::<Vec<_>>();

        for (shuffle, typesize) in [
            (blosc_src::BLOSC_NOSHUFFLE as i32, 1usize),
            (blosc_src::BLOSC_SHUFFLE as i32, 4usize),
        ] {
            let encoded = blosc_lz4_encode(&raw, shuffle, typesize, 1024);
            let decoded = codec
                .decode(&encoded, Some(raw.len()))
                .expect("valid Blosc LZ4 buffer should decode");
            assert_eq!(decoded, raw);
        }
    }

    #[test]
    fn lz4_fast_path_rejects_block_offsets_inside_header_table() {
        let codec = BloscCodec;
        let mut encoded = vec![0u8; 24];
        encoded[0] = blosc_src::BLOSC_VERSION_FORMAT as u8;
        encoded[1] = blosc_src::BLOSC_LZ4_VERSION_FORMAT as u8;
        encoded[2] = (blosc_src::BLOSC_LZ4_FORMAT << 5) as u8;
        encoded[3] = 1;
        encoded[4..8].copy_from_slice(&4u32.to_le_bytes());
        encoded[8..12].copy_from_slice(&4u32.to_le_bytes());
        let encoded_len = encoded.len() as u32;
        encoded[12..16].copy_from_slice(&encoded_len.to_le_bytes());
        encoded[16..20].copy_from_slice(&4i32.to_le_bytes());

        let err = codec
            .decode(&encoded, Some(4))
            .expect_err("block offset inside header/table should fail");

        assert!(
            matches!(err, super::super::super::CodecError::Decode { codec, message } if codec == "blosc" && message.contains("block offset"))
        );
    }

    #[test]
    fn lz4_fast_path_rejects_overlapping_block_offsets() {
        let codec = BloscCodec;
        let mut encoded = manual_blosc_lz4_raw_blocks(&[b"abcd", b"efgh"]);
        let first_block = i32::from_le_bytes(encoded[16..20].try_into().unwrap());
        encoded[20..24].copy_from_slice(&first_block.to_le_bytes());

        let err = codec
            .decode(&encoded, Some(8))
            .expect_err("overlapping block offsets should fail");

        assert!(
            matches!(err, super::super::super::CodecError::Decode { codec, message } if codec == "blosc" && message.contains("block offset"))
        );
    }

    #[test]
    fn lz4_fast_path_decodes_manual_raw_blocks() {
        let codec = BloscCodec;
        let encoded = manual_blosc_lz4_raw_blocks(&[b"abcd", b"efgh"]);

        let decoded = codec
            .decode(&encoded, Some(8))
            .expect("manual Blosc LZ4 raw blocks should decode");

        assert_eq!(&decoded, b"abcdefgh");
    }

    #[test]
    fn lz4_fast_path_decodes_byte_shuffled_slices() {
        let codec = BloscCodec;
        for typesize in [2usize, 4, 8] {
            let raw = (0..256 * 1024usize)
                .map(|idx| (idx.wrapping_mul(37).wrapping_add(11) & 0xff) as u8)
                .collect::<Vec<_>>();
            let encoded = blosc_lz4_encode(&raw, blosc_src::BLOSC_SHUFFLE as i32, typesize, 1024);
            let slice = DecodeSlice::new(
                Arc::from(
                    vec![
                        DecodeRange::new(0, 3, 31),
                        DecodeRange::new(28, 1000, 1100),
                        DecodeRange::new(128, raw.len() - 17, raw.len()),
                    ]
                    .into_boxed_slice(),
                ),
                145,
            );
            let partial = codec
                .decode_slice(&encoded, &slice, Some(raw.len()))
                .expect("slice decode")
                .expect("Blosc LZ4 should handle byte-shuffled slices");
            let mut expected = vec![0u8; slice.output_len];
            for range in slice.ranges.iter().copied() {
                expected[range.dst_offset..range.dst_offset + range.len()]
                    .copy_from_slice(&raw[range.src_start..range.src_end]);
            }
            assert_eq!(partial, expected, "typesize={typesize}");
        }
    }

    fn blosc_lz4_encode(raw: &[u8], shuffle: i32, typesize: usize, blocksize: usize) -> Vec<u8> {
        let compressor = CString::new("lz4").expect("static compressor name");
        let mut encoded = vec![0u8; raw.len() + blosc_src::BLOSC_MAX_OVERHEAD as usize];
        let written = unsafe {
            blosc_src::blosc_compress_ctx(
                5,
                shuffle,
                typesize,
                raw.len(),
                raw.as_ptr().cast::<c_void>(),
                encoded.as_mut_ptr().cast::<c_void>(),
                encoded.len(),
                compressor.as_ptr(),
                blocksize,
                1,
            )
        };
        assert!(written > 0, "Blosc compression failed with {written}");
        encoded.truncate(written as usize);
        encoded
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
