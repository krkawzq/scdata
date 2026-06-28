use std::io::{Cursor, Read};

use super::super::buffer::DecodeBuffer;
use super::super::spec::{sealed, ChunkCodec, LzmaCodecConfig};
use super::super::util::{decode_error, output_too_small, verify_size};
use super::super::{CodecError, CodecResult};

#[derive(Debug)]
pub(crate) struct GzipCodec;

impl sealed::Sealed for GzipCodec {}

impl ChunkCodec for GzipCodec {
    fn name(&self) -> &str {
        "gzip"
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        read_all(
            self.name(),
            flate2::read::GzDecoder::new(Cursor::new(encoded)),
            expected_size,
        )
    }

    fn decode_into(
        &self,
        encoded: &[u8],
        output: DecodeBuffer<'_>,
        expected_size: Option<usize>,
    ) -> CodecResult<usize> {
        read_to_buffer(
            self.name(),
            flate2::read::GzDecoder::new(Cursor::new(encoded)),
            output,
            expected_size,
        )
    }
}

#[derive(Debug)]
pub(crate) struct ZlibCodec;

impl sealed::Sealed for ZlibCodec {}

impl ChunkCodec for ZlibCodec {
    fn name(&self) -> &str {
        "zlib"
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        read_all(
            self.name(),
            flate2::read::ZlibDecoder::new(Cursor::new(encoded)),
            expected_size,
        )
    }

    fn decode_into(
        &self,
        encoded: &[u8],
        output: DecodeBuffer<'_>,
        expected_size: Option<usize>,
    ) -> CodecResult<usize> {
        read_to_buffer(
            self.name(),
            flate2::read::ZlibDecoder::new(Cursor::new(encoded)),
            output,
            expected_size,
        )
    }
}

#[derive(Debug)]
pub(crate) struct Bz2Codec;

impl sealed::Sealed for Bz2Codec {}

impl ChunkCodec for Bz2Codec {
    fn name(&self) -> &str {
        "bz2"
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        read_all(
            self.name(),
            bzip2::read::BzDecoder::new(Cursor::new(encoded)),
            expected_size,
        )
    }

    fn decode_into(
        &self,
        encoded: &[u8],
        output: DecodeBuffer<'_>,
        expected_size: Option<usize>,
    ) -> CodecResult<usize> {
        read_to_buffer(
            self.name(),
            bzip2::read::BzDecoder::new(Cursor::new(encoded)),
            output,
            expected_size,
        )
    }
}

#[derive(Debug)]
pub(crate) struct LzmaCodec {
    pub(crate) config: LzmaCodecConfig,
}

impl sealed::Sealed for LzmaCodec {}

impl ChunkCodec for LzmaCodec {
    fn name(&self) -> &str {
        "lzma"
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        self.validate_config()?;
        read_all(
            self.name(),
            xz2::read::XzDecoder::new(Cursor::new(encoded)),
            expected_size,
        )
    }

    fn decode_into(
        &self,
        encoded: &[u8],
        output: DecodeBuffer<'_>,
        expected_size: Option<usize>,
    ) -> CodecResult<usize> {
        self.validate_config()?;
        read_to_buffer(
            self.name(),
            xz2::read::XzDecoder::new(Cursor::new(encoded)),
            output,
            expected_size,
        )
    }
}

impl LzmaCodec {
    fn validate_config(&self) -> CodecResult<()> {
        if self.config.has_filters {
            return Err(CodecError::InvalidCodecConfig {
                codec: self.name().to_string(),
                message: "raw LZMA filter chains are not supported yet".to_string(),
            });
        }
        if self.config.format != 1 {
            return Err(CodecError::InvalidCodecConfig {
                codec: self.name().to_string(),
                message: format!(
                    "only numcodecs LZMA format=1 (XZ container) is supported, got {}",
                    self.config.format
                ),
            });
        }
        Ok(())
    }
}

fn read_all(
    codec: &str,
    mut reader: impl Read,
    expected_size: Option<usize>,
) -> CodecResult<Vec<u8>> {
    let mut decoded = Vec::with_capacity(expected_size.unwrap_or(0));
    reader
        .read_to_end(&mut decoded)
        .map_err(|err| decode_error(codec, err.to_string()))?;
    verify_size(codec, decoded.len(), expected_size)?;
    Ok(decoded)
}

fn read_to_buffer(
    codec: &str,
    mut reader: impl Read,
    mut output: DecodeBuffer<'_>,
    expected_size: Option<usize>,
) -> CodecResult<usize> {
    let capacity = match expected_size {
        Some(expected_size) => {
            output.ensure_capacity(codec, expected_size)?;
            expected_size
        }
        None => output.capacity(),
    };
    let mut written = 0usize;
    while written < capacity {
        let read = reader
            .read(&mut output.as_mut_slice()[written..capacity])
            .map_err(|err| decode_error(codec, err.to_string()))?;
        if read == 0 {
            verify_size(codec, written, expected_size)?;
            return Ok(written);
        }
        written += read;
    }

    let mut extra = [0u8; 1];
    let read = reader
        .read(&mut extra)
        .map_err(|err| decode_error(codec, err.to_string()))?;
    if read == 0 {
        verify_size(codec, written, expected_size)?;
        return Ok(written);
    }

    if let Some(expected) = expected_size {
        return Err(CodecError::SizeMismatch {
            codec: codec.to_string(),
            expected,
            actual: written + read,
        });
    }

    Err(output_too_small(codec, written + read, output.capacity()))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn read_to_buffer_does_not_write_past_expected_size() {
        let mut output = [0xaau8; 4];
        let err = read_to_buffer(
            "stream",
            Cursor::new(b"abcd"),
            DecodeBuffer::new(&mut output),
            Some(3),
        )
        .expect_err("payload longer than expected");

        assert!(
            matches!(err, CodecError::SizeMismatch { codec, expected: 3, actual: 4 } if codec == "stream")
        );
        assert_eq!(&output, b"abc\xaa");
    }
}
