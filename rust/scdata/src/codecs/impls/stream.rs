use std::io::{Cursor, ErrorKind, Read};

use super::super::buffer::{set_vec_len_for_decode, DecodeBuffer};
use super::super::spec::{sealed, ChunkCodec, LzmaCodecConfig};
use super::super::util::{decode_error, output_too_small, vec_with_decode_capacity, verify_size};
use super::super::{CodecError, CodecResult};
use xz2::stream::Stream;

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
        read_all(self.name(), self.decoder(encoded)?, expected_size)
    }

    fn decode_into(
        &self,
        encoded: &[u8],
        output: DecodeBuffer<'_>,
        expected_size: Option<usize>,
    ) -> CodecResult<usize> {
        read_to_buffer(self.name(), self.decoder(encoded)?, output, expected_size)
    }
}

impl LzmaCodec {
    fn decoder<'a>(
        &self,
        encoded: &'a [u8],
    ) -> CodecResult<xz2::read::XzDecoder<Cursor<&'a [u8]>>> {
        if self.config.has_filters {
            return Err(CodecError::InvalidCodecConfig {
                codec: self.name().to_string(),
                message: "raw LZMA filter chains are not supported yet".to_string(),
            });
        }

        let stream = match self.config.format {
            0 => Stream::new_auto_decoder(u64::MAX, 0),
            1 => Stream::new_stream_decoder(u64::MAX, 0),
            2 => Stream::new_lzma_decoder(u64::MAX),
            3 => {
                return Err(CodecError::InvalidCodecConfig {
                    codec: self.name().to_string(),
                    message: "raw LZMA streams are not supported yet".to_string(),
                });
            }
            format => {
                return Err(CodecError::InvalidCodecConfig {
                    codec: self.name().to_string(),
                    message: format!("unsupported numcodecs LZMA format {format}"),
                });
            }
        }
        .map_err(|err| decode_error(self.name(), err.to_string()))?;

        Ok(xz2::read::XzDecoder::new_stream(
            Cursor::new(encoded),
            stream,
        ))
    }
}

fn read_all(codec: &str, reader: impl Read, expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
    if let Some(expected_size) = expected_size {
        let mut decoded = vec_with_decode_capacity(codec, expected_size)?;
        set_vec_len_for_decode(&mut decoded, expected_size);
        let written = read_to_buffer(
            codec,
            reader,
            DecodeBuffer::new(decoded.as_mut_slice()),
            Some(expected_size),
        )?;
        decoded.truncate(written);
        return Ok(decoded);
    }

    let mut reader = reader;
    let mut decoded = Vec::new();
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
        let read = match reader.read(&mut output.as_mut_slice()[written..capacity]) {
            Ok(read) => read,
            Err(err) if err.kind() == ErrorKind::Interrupted => continue,
            Err(err) => return Err(decode_error(codec, err.to_string())),
        };
        if read == 0 {
            verify_size(codec, written, expected_size)?;
            return Ok(written);
        }
        written += read;
    }

    let mut extra = [0u8; 1];
    let read = loop {
        match reader.read(&mut extra) {
            Ok(read) => break read,
            Err(err) if err.kind() == ErrorKind::Interrupted => continue,
            Err(err) => return Err(decode_error(codec, err.to_string())),
        }
    };
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
    use std::io::{self, Cursor, ErrorKind, Read};

    use super::*;

    struct InterruptOnce<R> {
        inner: R,
        interrupted: bool,
    }

    impl<R> InterruptOnce<R> {
        fn new(inner: R) -> Self {
            Self {
                inner,
                interrupted: false,
            }
        }
    }

    impl<R: Read> Read for InterruptOnce<R> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if !self.interrupted {
                self.interrupted = true;
                return Err(io::Error::new(ErrorKind::Interrupted, "interrupted"));
            }
            self.inner.read(buf)
        }
    }

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

    #[test]
    fn read_to_buffer_retries_interrupted_reads() {
        let mut output = [0u8; 4];
        let written = read_to_buffer(
            "stream",
            InterruptOnce::new(Cursor::new(b"abcd")),
            DecodeBuffer::new(&mut output),
            Some(4),
        )
        .expect("interrupted read should be retried");

        assert_eq!(written, 4);
        assert_eq!(&output, b"abcd");
    }

    #[test]
    fn read_all_rejects_impossible_expected_size_before_allocating() {
        let codec = GzipCodec;
        let err = codec
            .decode(b"", Some(usize::MAX))
            .expect_err("impossible expected size should fail");

        assert!(
            matches!(err, CodecError::Decode { codec, message } if codec == "gzip" && message.contains("reserve decode buffer"))
        );
    }

    #[test]
    fn lzma_decodes_auto_xz_and_alone_formats() {
        let raw = b"scdata-lzma-format-test\x00\x01\x02".repeat(4);
        let xz_encoded = decode_hex("fd377a585a000004e6d6b4460200210116000000742fe5a3e0006700215d003998c8bdec457a6b22f0734b9796cc826b52aab1efb67ea467da173dc4276d000000000000323496ac8eed0cb800013d68ff1407091fb6f37d010000000004595a");
        let alone_encoded = decode_hex("5d00008000ffffffffffffffff003998c8bdec457a6b22f0734b9796cc826b52aab1efb67ea467da173dc4306de5fffffc08d000");

        let auto = LzmaCodec {
            config: LzmaCodecConfig {
                format: 0,
                ..Default::default()
            },
        };
        assert_eq!(
            auto.decode(&xz_encoded, Some(raw.len()))
                .expect("auto should decode XZ"),
            raw
        );
        assert_eq!(
            auto.decode(&alone_encoded, Some(raw.len()))
                .expect("auto should decode LZMA alone"),
            raw
        );

        let alone = LzmaCodec {
            config: LzmaCodecConfig {
                format: 2,
                ..Default::default()
            },
        };
        assert_eq!(
            alone
                .decode(&alone_encoded, Some(raw.len()))
                .expect("format=2 should decode LZMA alone"),
            raw
        );
    }

    fn decode_hex(hex: &str) -> Vec<u8> {
        assert_eq!(hex.len() % 2, 0);
        (0..hex.len())
            .step_by(2)
            .map(|idx| u8::from_str_radix(&hex[idx..idx + 2], 16).expect("valid hex"))
            .collect()
    }
}
