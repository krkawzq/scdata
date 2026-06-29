use super::spec::ChunkCodec;
use super::util::output_too_small;
use super::CodecResult;

/// Borrowed output memory supplied by the caller.
#[derive(Debug)]
pub struct DecodeBuffer<'a> {
    bytes: &'a mut [u8],
}

impl<'a> DecodeBuffer<'a> {
    pub fn new(bytes: &'a mut [u8]) -> Self {
        Self { bytes }
    }

    pub fn capacity(&self) -> usize {
        self.bytes.len()
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        self.bytes
    }

    pub(crate) fn ensure_capacity(&self, codec: &str, required: usize) -> CodecResult<()> {
        if self.capacity() < required {
            return Err(output_too_small(codec, required, self.capacity()));
        }
        Ok(())
    }

    pub(crate) fn write(self, codec: &str, decoded: &[u8]) -> CodecResult<usize> {
        self.ensure_capacity(codec, decoded.len())?;
        self.bytes[..decoded.len()].copy_from_slice(decoded);
        Ok(decoded.len())
    }
}

pub(crate) fn decode_into_vec_no_grow<C: ChunkCodec + ?Sized>(
    codec: &C,
    encoded: &[u8],
    mut output: Vec<u8>,
    expected_size: Option<usize>,
) -> CodecResult<Vec<u8>> {
    if let Some(required) = codec.decoded_size_hint(encoded, expected_size)? {
        if output.capacity() < required {
            return Err(output_too_small(codec.name(), required, output.capacity()));
        }
        set_vec_len_for_decode(&mut output, required);
    }

    let written = codec.decode_into(
        encoded,
        DecodeBuffer::new(output.as_mut_slice()),
        expected_size,
    )?;
    output.truncate(written);
    Ok(output)
}

pub(crate) fn set_vec_len_for_decode(output: &mut Vec<u8>, len: usize) {
    debug_assert!(len <= output.capacity());
    // SAFETY: `Vec<u8>` has no drop glue. Callers either checked capacity or
    // reserved it, and immediately pass the exposed range to a decoder/copy path
    // that must write every returned byte without reading old contents.
    unsafe {
        output.set_len(len);
    }
}
