mod header;
mod lz4_fast;
mod shuffle;

use std::os::raw::c_void;

use super::super::buffer::{set_vec_len_for_decode, DecodeBuffer};
use super::super::spec::{sealed, ChunkCodec};
use super::super::util::{decode_error, output_too_small, verify_size};
use super::super::CodecResult;
use header::{blosc_decoded_size, blosc_header};
use lz4_fast::try_blosc_lz4_decode_into;

#[derive(Debug)]
pub(crate) struct BloscCodec;

impl sealed::Sealed for BloscCodec {}

impl ChunkCodec for BloscCodec {
    fn name(&self) -> &str {
        "blosc"
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        let header = blosc_header(self.name(), encoded)?;
        let decoded_size = header.decoded_size;
        verify_size(self.name(), decoded_size, expected_size)?;

        let mut decoded = Vec::with_capacity(decoded_size);
        set_vec_len_for_decode(&mut decoded, decoded_size);
        let written = blosc_decode_header_into_output(self.name(), encoded, header, &mut decoded)?;
        decoded.truncate(written);
        Ok(decoded)
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
            output.reserve_exact(decoded_size - output.capacity());
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
