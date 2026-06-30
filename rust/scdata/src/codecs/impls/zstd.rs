use std::io::Cursor;

use super::super::buffer::{set_vec_len_for_decode, DecodeBuffer};
use super::super::spec::{sealed, ChunkCodec, CodecCacheKey};
use super::super::util::{
    decode_error, output_too_small, reserve_decode_buffer, vec_with_decode_capacity, verify_size,
};
use super::super::CodecResult;

#[derive(Debug)]
pub(crate) struct ZstdCodec;

impl sealed::Sealed for ZstdCodec {}

fn zstd_decoded_size(codec: &str, encoded: &[u8]) -> CodecResult<Option<usize>> {
    match zstd::zstd_safe::get_frame_content_size(encoded) {
        Ok(Some(size)) => usize::try_from(size).map(Some).map_err(|_| {
            decode_error(
                codec,
                format!("Zstd frame content size {size} does not fit in usize"),
            )
        }),
        Ok(None) => Ok(None),
        Err(err) => Err(decode_error(codec, err.to_string())),
    }
}

fn zstd_output_size(
    codec: &str,
    encoded: &[u8],
    expected_size: Option<usize>,
) -> CodecResult<Option<usize>> {
    match (zstd_decoded_size(codec, encoded)?, expected_size) {
        (Some(actual), Some(expected)) => {
            verify_size(codec, actual, Some(expected))?;
            Ok(Some(actual))
        }
        (Some(actual), None) => Ok(Some(actual)),
        (None, Some(expected)) => Ok(Some(expected)),
        (None, None) => Ok(None),
    }
}

impl ChunkCodec for ZstdCodec {
    fn name(&self) -> &str {
        "zstd"
    }

    fn cache_key(&self) -> CodecCacheKey {
        CodecCacheKey::Static("zstd")
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        let decoded_size = zstd_output_size(self.name(), encoded, expected_size)?;
        if let Some(decoded_size) = decoded_size {
            let mut decoded = vec_with_decode_capacity(self.name(), decoded_size)?;
            set_vec_len_for_decode(&mut decoded, decoded_size);
            let written = zstd_decompress_into_slice(self.name(), encoded, &mut decoded)?;
            verify_size(self.name(), written, Some(decoded_size))?;
            decoded.truncate(written);
            verify_size(self.name(), decoded.len(), expected_size)?;
            return Ok(decoded);
        }

        let decoded = zstd::decode_all(Cursor::new(encoded))
            .map_err(|err| decode_error(self.name(), err.to_string()))?;
        verify_size(self.name(), decoded.len(), expected_size)?;
        Ok(decoded)
    }

    fn decoded_size_hint(
        &self,
        encoded: &[u8],
        expected_size: Option<usize>,
    ) -> CodecResult<Option<usize>> {
        zstd_output_size(self.name(), encoded, expected_size)
    }

    fn decode_into(
        &self,
        encoded: &[u8],
        mut output: DecodeBuffer<'_>,
        expected_size: Option<usize>,
    ) -> CodecResult<usize> {
        let decoded_size = zstd_output_size(self.name(), encoded, expected_size)?;
        if let Some(decoded_size) = decoded_size {
            output.ensure_capacity(self.name(), decoded_size)?;
            let written = zstd_decompress_into_slice(
                self.name(),
                encoded,
                &mut output.as_mut_slice()[..decoded_size],
            )?;
            verify_size(self.name(), written, Some(decoded_size))?;
            return Ok(written);
        }

        let written = zstd_decompress_into_slice(self.name(), encoded, output.as_mut_slice())?;
        verify_size(self.name(), written, expected_size)?;
        Ok(written)
    }

    fn decode_to_vec(
        &self,
        encoded: &[u8],
        mut output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        let decoded_size = zstd_output_size(self.name(), encoded, expected_size)?;
        let Some(decoded_size) = decoded_size else {
            return self.decode(encoded, expected_size);
        };

        if output.capacity() < decoded_size {
            return Err(output_too_small(
                self.name(),
                decoded_size,
                output.capacity(),
            ));
        }
        set_vec_len_for_decode(&mut output, decoded_size);
        let written = zstd_decompress_into_slice(self.name(), encoded, &mut output)?;
        verify_size(self.name(), written, Some(decoded_size))?;
        output.truncate(written);
        verify_size(self.name(), output.len(), expected_size)?;
        Ok(output)
    }

    fn decode_to_vec_grow(
        &self,
        encoded: &[u8],
        output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        let decoded_size = zstd_output_size(self.name(), encoded, expected_size)?;
        let Some(decoded_size) = decoded_size else {
            return self.decode(encoded, expected_size);
        };

        let mut output = output;
        output.clear();
        if output.capacity() < decoded_size {
            let additional = decoded_size - output.capacity();
            reserve_decode_buffer(self.name(), &mut output, additional)?;
        }
        set_vec_len_for_decode(&mut output, decoded_size);
        let written = zstd_decompress_into_slice(self.name(), encoded, &mut output)?;
        verify_size(self.name(), written, Some(decoded_size))?;
        output.truncate(written);
        verify_size(self.name(), output.len(), expected_size)?;
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

fn zstd_decompress_into_slice(
    codec: &str,
    encoded: &[u8],
    output: &mut [u8],
) -> CodecResult<usize> {
    zstd::zstd_safe::decompress(output, encoded)
        .map_err(|err| decode_error(codec, zstd::zstd_safe::get_error_name(err).to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_impossible_expected_size_before_allocating() {
        let codec = ZstdCodec;
        let encoded = zstd::bulk::compress(b"abc", 1).expect("zstd test encode");

        let err = codec
            .decode(&encoded, Some(usize::MAX))
            .expect_err("mismatched expected size should fail before allocation");

        assert!(
            matches!(err, super::super::super::CodecError::SizeMismatch { codec, expected: usize::MAX, actual: 3 } if codec == "zstd")
        );
    }

    #[test]
    fn rejects_invalid_frame_before_reserving_expected_size() {
        let codec = ZstdCodec;
        let err = codec
            .decode(b"not a zstd frame", Some(usize::MAX))
            .expect_err("invalid frame should fail before allocation");

        assert!(
            matches!(err, super::super::super::CodecError::Decode { codec, message } if codec == "zstd" && !message.contains("reserve decode buffer"))
        );
    }
}
