use super::super::buffer::{set_vec_len_for_decode, DecodeBuffer};
use super::super::spec::{sealed, ChunkCodec};
use super::super::util::{decode_error, output_too_small, verify_size};
use super::super::CodecResult;

#[derive(Debug)]
pub(crate) struct Crc32Codec;

impl sealed::Sealed for Crc32Codec {}

impl ChunkCodec for Crc32Codec {
    fn name(&self) -> &str {
        "crc32"
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        if encoded.len() < 4 {
            return Err(decode_error(
                self.name(),
                "CRC32 buffer is shorter than 4 bytes",
            ));
        }

        let expected = u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
        let payload = &encoded[4..];
        let actual = crc32fast::hash(payload);
        if actual != expected {
            return Err(decode_error(
                self.name(),
                format!("checksum mismatch: expected {expected:#010x}, got {actual:#010x}"),
            ));
        }

        verify_size(self.name(), payload.len(), expected_size)?;
        Ok(payload.to_vec())
    }

    fn decode_into(
        &self,
        encoded: &[u8],
        output: DecodeBuffer<'_>,
        expected_size: Option<usize>,
    ) -> CodecResult<usize> {
        if encoded.len() < 4 {
            return Err(decode_error(
                self.name(),
                "CRC32 buffer is shorter than 4 bytes",
            ));
        }

        let expected = u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
        let payload = &encoded[4..];
        let actual = crc32fast::hash(payload);
        if actual != expected {
            return Err(decode_error(
                self.name(),
                format!("checksum mismatch: expected {expected:#010x}, got {actual:#010x}"),
            ));
        }

        verify_size(self.name(), payload.len(), expected_size)?;
        output.write(self.name(), payload)
    }

    fn decoded_size_hint(
        &self,
        encoded: &[u8],
        _expected_size: Option<usize>,
    ) -> CodecResult<Option<usize>> {
        if encoded.len() < 4 {
            return Err(decode_error(
                self.name(),
                "CRC32 buffer is shorter than 4 bytes",
            ));
        }
        Ok(Some(encoded.len() - 4))
    }

    fn encoded_size_hint(&self, decoded_size: usize) -> Option<usize> {
        decoded_size.checked_add(4)
    }

    fn decode_owned(
        &self,
        mut encoded: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        if encoded.len() < 4 {
            return Err(decode_error(
                self.name(),
                "CRC32 buffer is shorter than 4 bytes",
            ));
        }

        let expected = u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
        let payload_len = encoded.len() - 4;
        let actual = crc32fast::hash(&encoded[4..]);
        if actual != expected {
            return Err(decode_error(
                self.name(),
                format!("checksum mismatch: expected {expected:#010x}, got {actual:#010x}"),
            ));
        }

        verify_size(self.name(), payload_len, expected_size)?;
        encoded.copy_within(4.., 0);
        encoded.truncate(payload_len);
        Ok(encoded)
    }

    fn decode_to_vec(
        &self,
        encoded: &[u8],
        mut output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        if encoded.len() < 4 {
            return Err(decode_error(
                self.name(),
                "CRC32 buffer is shorter than 4 bytes",
            ));
        }

        let expected = u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
        let payload = &encoded[4..];
        let actual = crc32fast::hash(payload);
        if actual != expected {
            return Err(decode_error(
                self.name(),
                format!("checksum mismatch: expected {expected:#010x}, got {actual:#010x}"),
            ));
        }

        verify_size(self.name(), payload.len(), expected_size)?;
        if output.capacity() < payload.len() {
            return Err(output_too_small(
                self.name(),
                payload.len(),
                output.capacity(),
            ));
        }
        set_vec_len_for_decode(&mut output, payload.len());
        output.copy_from_slice(payload);
        Ok(output)
    }

    fn decode_to_vec_grow(
        &self,
        encoded: &[u8],
        mut output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        if encoded.len() < 4 {
            return Err(decode_error(
                self.name(),
                "CRC32 buffer is shorter than 4 bytes",
            ));
        }

        let expected = u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
        let payload = &encoded[4..];
        let actual = crc32fast::hash(payload);
        if actual != expected {
            return Err(decode_error(
                self.name(),
                format!("checksum mismatch: expected {expected:#010x}, got {actual:#010x}"),
            ));
        }

        verify_size(self.name(), payload.len(), expected_size)?;
        output.clear();
        if output.capacity() < payload.len() {
            output.reserve_exact(payload.len() - output.capacity());
        }
        set_vec_len_for_decode(&mut output, payload.len());
        output.copy_from_slice(payload);
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

    fn prefers_decode_owned(&self) -> bool {
        true
    }
}
