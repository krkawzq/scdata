use super::super::buffer::{set_vec_len_for_decode, DecodeBuffer};
use super::super::spec::{sealed, ChunkCodec, CodecCacheKey};
use super::super::util::{output_too_small, reserve_decode_buffer, verify_size};
use super::super::CodecResult;

#[derive(Debug, Default)]
pub struct UncompressedCodec;

impl sealed::Sealed for UncompressedCodec {}

impl ChunkCodec for UncompressedCodec {
    fn name(&self) -> &str {
        "none"
    }

    fn cache_key(&self) -> CodecCacheKey {
        CodecCacheKey::Static("none")
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        verify_size(self.name(), encoded.len(), expected_size)?;
        Ok(encoded.to_vec())
    }

    fn decode_owned(&self, encoded: Vec<u8>, expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        verify_size(self.name(), encoded.len(), expected_size)?;
        Ok(encoded)
    }

    fn decoded_size_hint(
        &self,
        encoded: &[u8],
        _expected_size: Option<usize>,
    ) -> CodecResult<Option<usize>> {
        Ok(Some(encoded.len()))
    }

    fn encoded_size_hint(&self, decoded_size: usize) -> Option<usize> {
        Some(decoded_size)
    }

    fn prefers_decode_owned(&self) -> bool {
        true
    }

    fn is_identity(&self) -> bool {
        true
    }

    fn decode_to_vec(
        &self,
        encoded: &[u8],
        mut output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        verify_size(self.name(), encoded.len(), expected_size)?;
        if output.capacity() < encoded.len() {
            return Err(output_too_small(
                self.name(),
                encoded.len(),
                output.capacity(),
            ));
        }
        set_vec_len_for_decode(&mut output, encoded.len());
        unsafe {
            std::ptr::copy_nonoverlapping(encoded.as_ptr(), output.as_mut_ptr(), encoded.len());
        }
        Ok(output)
    }

    fn decode_to_vec_grow(
        &self,
        encoded: &[u8],
        mut output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        verify_size(self.name(), encoded.len(), expected_size)?;
        output.clear();
        if output.capacity() < encoded.len() {
            let additional = encoded.len() - output.capacity();
            reserve_decode_buffer(self.name(), &mut output, additional)?;
        }
        set_vec_len_for_decode(&mut output, encoded.len());
        unsafe {
            std::ptr::copy_nonoverlapping(encoded.as_ptr(), output.as_mut_ptr(), encoded.len());
        }
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

    fn decode_into(
        &self,
        encoded: &[u8],
        mut output: DecodeBuffer<'_>,
        expected_size: Option<usize>,
    ) -> CodecResult<usize> {
        verify_size(self.name(), encoded.len(), expected_size)?;
        output.ensure_capacity(self.name(), encoded.len())?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                encoded.as_ptr(),
                output.as_mut_slice().as_mut_ptr(),
                encoded.len(),
            );
        }
        Ok(encoded.len())
    }
}
