use super::super::buffer::{set_vec_len_for_decode, DecodeBuffer};
use super::super::spec::{sealed, ChunkCodec};
use super::super::util::{
    decode_error, output_too_small, reserve_decode_buffer, vec_with_decode_capacity, verify_size,
};
use super::super::CodecResult;

#[derive(Debug)]
pub(crate) struct Lz4Codec;

impl sealed::Sealed for Lz4Codec {}

fn lz4_decoded_size(codec: &str, encoded: &[u8]) -> CodecResult<usize> {
    if encoded.len() < 4 {
        return Err(decode_error(
            codec,
            "LZ4 buffer is shorter than size header",
        ));
    }
    let decoded_size =
        u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;
    if decoded_size > i32::MAX as usize {
        return Err(decode_error(
            codec,
            format!("LZ4 decoded payload is too large: {decoded_size}"),
        ));
    }
    Ok(decoded_size)
}

impl ChunkCodec for Lz4Codec {
    fn name(&self) -> &str {
        "lz4"
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        let decoded_size = lz4_decoded_size(self.name(), encoded)?;
        verify_size(self.name(), decoded_size, expected_size)?;
        let mut decoded = vec_with_decode_capacity(self.name(), decoded_size)?;
        set_vec_len_for_decode(&mut decoded, decoded_size);
        let written = lz4_decompress_known_size(self.name(), encoded, &mut decoded, decoded_size)?;
        decoded.truncate(written);
        Ok(decoded)
    }

    fn decoded_size_hint(
        &self,
        encoded: &[u8],
        _expected_size: Option<usize>,
    ) -> CodecResult<Option<usize>> {
        lz4_decoded_size(self.name(), encoded).map(Some)
    }

    fn decode_into(
        &self,
        encoded: &[u8],
        mut output: DecodeBuffer<'_>,
        expected_size: Option<usize>,
    ) -> CodecResult<usize> {
        let decoded_size = lz4_decoded_size(self.name(), encoded)?;
        verify_size(self.name(), decoded_size, expected_size)?;
        output.ensure_capacity(self.name(), decoded_size)?;
        lz4_decompress_known_size(
            self.name(),
            encoded,
            &mut output.as_mut_slice()[..decoded_size],
            decoded_size,
        )
    }

    fn decode_to_vec(
        &self,
        encoded: &[u8],
        mut output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        let decoded_size = lz4_decoded_size(self.name(), encoded)?;
        verify_size(self.name(), decoded_size, expected_size)?;
        if output.capacity() < decoded_size {
            return Err(output_too_small(
                self.name(),
                decoded_size,
                output.capacity(),
            ));
        }
        set_vec_len_for_decode(&mut output, decoded_size);
        let written = lz4_decompress_known_size(self.name(), encoded, &mut output, decoded_size)?;
        output.truncate(written);
        Ok(output)
    }

    fn decode_to_vec_grow(
        &self,
        encoded: &[u8],
        mut output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        let decoded_size = lz4_decoded_size(self.name(), encoded)?;
        verify_size(self.name(), decoded_size, expected_size)?;
        output.clear();
        if output.capacity() < decoded_size {
            let additional = decoded_size - output.capacity();
            reserve_decode_buffer(self.name(), &mut output, additional)?;
        }
        set_vec_len_for_decode(&mut output, decoded_size);
        let written = lz4_decompress_known_size(self.name(), encoded, &mut output, decoded_size)?;
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

fn lz4_decompress_known_size(
    codec: &str,
    encoded: &[u8],
    output: &mut [u8],
    decoded_size: usize,
) -> CodecResult<usize> {
    if output.len() < decoded_size {
        return Err(output_too_small(codec, decoded_size, output.len()));
    }

    lz4_decompress_raw_into(codec, &encoded[4..], &mut output[..decoded_size])?;
    Ok(decoded_size)
}

pub(crate) fn lz4_decompress_raw_into(
    codec: &str,
    compressed: &[u8],
    output: &mut [u8],
) -> CodecResult<()> {
    let compressed_size = i32::try_from(compressed.len()).map_err(|_| {
        decode_error(
            codec,
            format!("LZ4 compressed payload is too large: {}", compressed.len()),
        )
    })?;
    let max_decompressed_size = i32::try_from(output.len()).map_err(|_| {
        decode_error(
            codec,
            format!("LZ4 decoded payload is too large: {}", output.len()),
        )
    })?;

    let written = unsafe {
        lz4_sys::LZ4_decompress_safe(
            compressed.as_ptr().cast::<lz4_sys::c_char>(),
            output.as_mut_ptr().cast::<lz4_sys::c_char>(),
            compressed_size,
            max_decompressed_size,
        )
    };
    if written < 0 {
        return Err(decode_error(
            codec,
            format!("LZ4 decompressor returned {written}"),
        ));
    }
    verify_size(codec, written as usize, Some(output.len()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_decoded_size_that_exceeds_lz4_ffi_limit() {
        let codec = Lz4Codec;
        let encoded = [0x00, 0x00, 0x00, 0x80, 0x00];
        let err = codec
            .decode(&encoded, None)
            .expect_err("oversized LZ4 payload should fail before allocation");

        assert!(
            matches!(err, super::super::super::CodecError::Decode { codec, message } if codec == "lz4" && message.contains("too large"))
        );
    }
}
