use crate::error::{Error, Result};
use crate::format::flags::{
    decode_shuffle, Codec, Shuffle, FLAG_BITSHUFFLE, FLAG_DONT_SPLIT, FLAG_MEMCPY, FLAG_SHUFFLE,
};

pub const FORMAT_VERSION: u8 = 2;
pub const HEADER_LEN: usize = 16;
pub const MAX_BUFFER_SIZE: usize = i32::MAX as usize - HEADER_LEN;
pub const MAX_BLOCK_SIZE: usize = (i32::MAX as usize - 255 * std::mem::size_of::<i32>()) / 3;
const KNOWN_FLAG_MASK: u8 =
    FLAG_SHUFFLE | FLAG_MEMCPY | FLAG_BITSHUFFLE | FLAG_DONT_SPLIT | 0b1110_0000;

/// Validated Blosc1 16-byte header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    codec_version: u8,
    flags: u8,
    element_size: u8,
    decoded_size: u32,
    block_size: u32,
    encoded_size: u32,
}

impl Header {
    pub fn new(
        codec_version: u8,
        flags: u8,
        element_size: u8,
        decoded_size: u32,
        block_size: u32,
        encoded_size: u32,
    ) -> Result<Self> {
        let header = Self {
            codec_version,
            flags,
            element_size,
            decoded_size,
            block_size,
            encoded_size,
        };
        header.validate()?;
        Ok(header)
    }

    pub fn parse(input: &[u8]) -> Result<Self> {
        if input.len() < HEADER_LEN {
            return Err(Error::InvalidFormat(
                "input shorter than Blosc1 header".into(),
            ));
        }
        if input[0] != FORMAT_VERSION {
            return Err(Error::InvalidFormat(format!(
                "unsupported Blosc1 version {:#04x}",
                input[0]
            )));
        }
        Self::new(
            input[1],
            input[2],
            input[3],
            u32::from_le_bytes(input[4..8].try_into().unwrap()),
            u32::from_le_bytes(input[8..12].try_into().unwrap()),
            u32::from_le_bytes(input[12..16].try_into().unwrap()),
        )
    }

    fn validate(self) -> Result<()> {
        if self.codec_version != 1 {
            return Err(Error::InvalidFormat(format!(
                "unsupported codec format version {}",
                self.codec_version
            )));
        }
        if self.element_size == 0 {
            return Err(Error::InvalidFormat("element size must be non-zero".into()));
        }
        if self.flags & FLAG_SHUFFLE != 0 && self.flags & FLAG_BITSHUFFLE != 0 {
            return Err(Error::InvalidFormat(
                "byte shuffle and bit shuffle flags are both set".into(),
            ));
        }
        if self.flags & !KNOWN_FLAG_MASK != 0 {
            return Err(Error::InvalidFormat(format!(
                "unsupported flag bits {:#04x}",
                self.flags & !KNOWN_FLAG_MASK
            )));
        }
        if self.encoded_size < HEADER_LEN as u32 {
            return Err(Error::InvalidFormat("encoded size is too small".into()));
        }
        if self.decoded_size as usize > MAX_BUFFER_SIZE {
            return Err(Error::InvalidFormat(format!(
                "Blosc1 decoded size {} exceeds {MAX_BUFFER_SIZE}",
                self.decoded_size
            )));
        }
        if self.encoded_size > i32::MAX as u32 {
            return Err(Error::InvalidFormat(format!(
                "Blosc1 encoded size {} exceeds {}",
                self.encoded_size,
                i32::MAX
            )));
        }
        if self.decoded_size == 0 {
            if self.encoded_size != HEADER_LEN as u32 {
                return Err(Error::InvalidFormat(
                    "empty Blosc1 chunk has an inconsistent encoded size".into(),
                ));
            }
            self.codec()?;
            return Ok(());
        }
        if self.block_size == 0 || self.block_size > self.decoded_size {
            return Err(Error::InvalidFormat(format!(
                "invalid Blosc1 block size {} for decoded size {}",
                self.block_size, self.decoded_size
            )));
        }
        if self.block_size as usize > MAX_BLOCK_SIZE {
            return Err(Error::InvalidFormat(format!(
                "Blosc1 block size {} exceeds {MAX_BLOCK_SIZE}",
                self.block_size
            )));
        }
        if self.is_raw() {
            let expected = (HEADER_LEN as u64) + u64::from(self.decoded_size);
            if u64::from(self.encoded_size) != expected {
                return Err(Error::InvalidFormat(
                    "raw Blosc1 chunk has inconsistent sizes".into(),
                ));
            }
        } else if self.index_prefix_len()? >= self.encoded_size as usize {
            return Err(Error::InvalidFormat(
                "Blosc1 block index leaves no encoded payload".into(),
            ));
        }
        self.codec()?;
        Ok(())
    }

    pub fn write(self, output: &mut [u8]) -> Result<()> {
        if output.len() < HEADER_LEN {
            return Err(Error::BufferTooSmall {
                need: HEADER_LEN,
                have: output.len(),
            });
        }
        output[0] = FORMAT_VERSION;
        output[1] = self.codec_version;
        output[2] = self.flags;
        output[3] = self.element_size;
        output[4..8].copy_from_slice(&self.decoded_size.to_le_bytes());
        output[8..12].copy_from_slice(&self.block_size.to_le_bytes());
        output[12..16].copy_from_slice(&self.encoded_size.to_le_bytes());
        Ok(())
    }

    pub fn codec_version(self) -> u8 {
        self.codec_version
    }

    pub fn flags(self) -> u8 {
        self.flags
    }

    pub fn element_size(self) -> usize {
        self.element_size as usize
    }

    pub fn decoded_size(self) -> usize {
        self.decoded_size as usize
    }

    pub fn block_size(self) -> usize {
        self.block_size as usize
    }

    pub fn encoded_size(self) -> usize {
        self.encoded_size as usize
    }

    pub fn block_count(self) -> usize {
        if self.decoded_size == 0 {
            0
        } else if self.is_raw() {
            1
        } else {
            self.decoded_size.div_ceil(self.block_size) as usize
        }
    }

    pub fn codec(self) -> Result<Codec> {
        Codec::from_wire_id(self.flags >> 5)
    }

    pub fn shuffle(self) -> Shuffle {
        decode_shuffle(self.flags)
    }

    pub fn split_blocks(self) -> bool {
        self.flags & FLAG_DONT_SPLIT == 0
    }

    pub fn is_raw(self) -> bool {
        self.flags & FLAG_MEMCPY != 0
    }

    pub fn index_prefix_len(self) -> Result<usize> {
        if self.decoded_size == 0 || self.is_raw() {
            return Ok(HEADER_LEN);
        }
        HEADER_LEN
            .checked_add(
                self.block_count()
                    .checked_mul(4)
                    .ok_or_else(|| Error::InvalidFormat("Blosc1 index size overflow".into()))?,
            )
            .ok_or_else(|| Error::InvalidFormat("Blosc1 index prefix overflow".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::flags::encode_flags;

    #[test]
    fn raw_chunk_has_one_payload_block() {
        let decoded_size = 8192;
        let header = Header::new(
            1,
            encode_flags(Codec::Lz4, Shuffle::Bits, false, true),
            4,
            decoded_size,
            256,
            decoded_size + HEADER_LEN as u32,
        )
        .unwrap();

        assert_eq!(header.block_count(), 1);
        assert_eq!(header.index_prefix_len().unwrap(), HEADER_LEN);
    }
}
