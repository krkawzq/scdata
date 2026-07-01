use super::buffer::DecodeBuffer;
use super::spec::{ChunkCodec, DecodeSlice};
use super::{CodecError, CodecResult};

pub(crate) struct DecodeRunner;

impl DecodeRunner {
    pub(crate) fn decode(
        codec: &dyn ChunkCodec,
        encoded: &[u8],
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        codec.decode(encoded, expected_size)
    }

    pub(crate) fn decode_vec_input(
        codec: &dyn ChunkCodec,
        input: Vec<u8>,
        output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<(Vec<u8>, Vec<u8>)> {
        if codec.prefers_decode_owned() {
            let decoded = codec.decode_owned(input, expected_size)?;
            return Ok((decoded, output));
        }

        let decoded = Self::decode_borrowed_to_vec(codec, &input, output, expected_size, None)?;
        Ok((decoded, input))
    }

    pub(crate) fn decode_borrowed_to_vec(
        codec: &dyn ChunkCodec,
        encoded: &[u8],
        output: Vec<u8>,
        expected_size: Option<usize>,
        slice: Option<&DecodeSlice>,
    ) -> CodecResult<Vec<u8>> {
        if let Some(slice) = slice {
            if let Some(decoded) = codec.decode_slice(encoded, slice, expected_size)? {
                return Ok(decoded);
            }
            let decoded = codec.decode_to_vec_grow(encoded, output, expected_size)?;
            return materialize_slice(codec.name(), &decoded, slice);
        }
        codec.decode_to_vec_grow(encoded, output, expected_size)
    }

    pub(crate) fn decode_into(
        codec: &dyn ChunkCodec,
        encoded: &[u8],
        output: DecodeBuffer<'_>,
        expected_size: Option<usize>,
    ) -> CodecResult<usize> {
        codec.decode_into(encoded, output, expected_size)
    }

    pub(crate) fn decode_to_initialized_vec(
        codec: &dyn ChunkCodec,
        encoded: &[u8],
        mut output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        let written = codec.decode_into(
            encoded,
            DecodeBuffer::new(output.as_mut_slice()),
            expected_size,
        )?;
        output.truncate(written);
        Ok(output)
    }

    pub(crate) fn decode_to_capacity_vec(
        codec: &dyn ChunkCodec,
        encoded: &[u8],
        output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        codec.decode_to_capacity_vec(encoded, output, expected_size)
    }
}

fn materialize_slice(codec: &str, decoded: &[u8], slice: &DecodeSlice) -> CodecResult<Vec<u8>> {
    let mut out = vec![0u8; slice.output_len];
    for range in slice.ranges.iter().copied() {
        let Some(dst_end) = range.dst_offset.checked_add(range.len()) else {
            return Err(CodecError::Decode {
                codec: codec.to_string(),
                message: "invalid decode slice range".to_string(),
            });
        };
        if range.src_start > range.src_end || range.src_end > decoded.len() || dst_end > out.len() {
            return Err(CodecError::Decode {
                codec: codec.to_string(),
                message: "invalid decode slice range".to_string(),
            });
        }
        out[range.dst_offset..dst_end].copy_from_slice(&decoded[range.src_start..range.src_end]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    use super::super::spec::sealed;
    use super::*;

    #[derive(Debug)]
    struct FastVecCodec {
        decode_to_vec_called: Arc<AtomicBool>,
    }

    impl sealed::Sealed for FastVecCodec {}

    impl ChunkCodec for FastVecCodec {
        fn name(&self) -> &str {
            "fast"
        }

        fn decode(&self, encoded: &[u8], _expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
            Ok(encoded.to_vec())
        }

        fn decoded_size_hint(
            &self,
            encoded: &[u8],
            _expected_size: Option<usize>,
        ) -> CodecResult<Option<usize>> {
            Ok(Some(encoded.len()))
        }

        fn decode_to_vec(
            &self,
            encoded: &[u8],
            mut output: Vec<u8>,
            _expected_size: Option<usize>,
        ) -> CodecResult<Vec<u8>> {
            self.decode_to_vec_called.store(true, Ordering::SeqCst);
            output.clear();
            output.extend_from_slice(encoded);
            Ok(output)
        }
    }

    #[test]
    fn borrowed_decode_uses_codec_vec_fastpath_when_size_is_known() {
        let called = Arc::new(AtomicBool::new(false));
        let codec = FastVecCodec {
            decode_to_vec_called: Arc::clone(&called),
        };

        let decoded =
            DecodeRunner::decode_borrowed_to_vec(&codec, b"abcdef", Vec::new(), Some(6), None)
                .expect("decode");

        assert_eq!(&decoded, b"abcdef");
        assert!(called.load(Ordering::SeqCst));
    }
}
