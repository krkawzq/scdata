use super::super::super::util::decode_error;
use super::super::super::CodecResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BloscHeader {
    pub(crate) compversion: u8,
    pub(crate) flags: u8,
    pub(crate) typesize: usize,
    pub(crate) decoded_size: usize,
    pub(crate) blocksize: usize,
    pub(crate) compressed_size: usize,
}

impl BloscHeader {
    pub(crate) fn compformat(self) -> u8 {
        (self.flags & 0xe0) >> 5
    }

    pub(crate) fn is_memcpyed(self) -> bool {
        self.flags & blosc_src::BLOSC_MEMCPYED as u8 != 0
    }

    pub(crate) fn is_byte_shuffled(self) -> bool {
        self.flags & blosc_src::BLOSC_DOSHUFFLE as u8 != 0 && self.typesize > 1
    }

    pub(crate) fn is_bit_shuffled(self) -> bool {
        self.flags & blosc_src::BLOSC_DOBITSHUFFLE as u8 != 0
    }

    pub(crate) fn dont_split(self) -> bool {
        self.flags & 0x10 != 0
    }
}

pub(crate) fn blosc_header(codec: &str, encoded: &[u8]) -> CodecResult<BloscHeader> {
    let header = blosc_header_prefix(codec, encoded)?;
    if header.compressed_size != encoded.len() {
        return Err(decode_error(codec, "invalid Blosc buffer"));
    }
    Ok(header)
}

pub(crate) fn blosc_header_prefix(codec: &str, encoded: &[u8]) -> CodecResult<BloscHeader> {
    if encoded.len() < blosc_src::BLOSC_MIN_HEADER_LENGTH as usize {
        return Err(decode_error(codec, "buffer is shorter than a Blosc header"));
    }

    if encoded[0] != blosc_src::BLOSC_VERSION_FORMAT as u8 {
        return Err(decode_error(codec, "unsupported Blosc format version"));
    }

    let decoded_size =
        u32::from_le_bytes([encoded[4], encoded[5], encoded[6], encoded[7]]) as usize;
    let blocksize = u32::from_le_bytes([encoded[8], encoded[9], encoded[10], encoded[11]]) as usize;
    let compressed_size =
        u32::from_le_bytes([encoded[12], encoded[13], encoded[14], encoded[15]]) as usize;
    if decoded_size > blosc_src::BLOSC_MAX_BUFFERSIZE as usize {
        return Err(decode_error(
            codec,
            format!("Blosc decoded payload is too large: {decoded_size}"),
        ));
    }
    Ok(BloscHeader {
        compversion: encoded[1],
        flags: encoded[2],
        typesize: encoded[3] as usize,
        decoded_size,
        blocksize,
        compressed_size,
    })
}

pub(crate) fn blosc_decoded_size(codec: &str, encoded: &[u8]) -> CodecResult<usize> {
    blosc_header(codec, encoded).map(|header| header.decoded_size)
}
