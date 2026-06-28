use super::buffer::DecodeBuffer;
use super::spec::ChunkCodec;
use super::CodecResult;

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

        let decoded = Self::decode_borrowed_to_vec(codec, &input, output, expected_size)?;
        Ok((decoded, input))
    }

    pub(crate) fn decode_borrowed_to_vec(
        codec: &dyn ChunkCodec,
        encoded: &[u8],
        output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
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

        let decoded = DecodeRunner::decode_borrowed_to_vec(&codec, b"abcdef", Vec::new(), Some(6))
            .expect("decode");

        assert_eq!(&decoded, b"abcdef");
        assert!(called.load(Ordering::SeqCst));
    }
}
