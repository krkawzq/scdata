use super::super::super::util::decode_error;
use super::super::super::CodecResult;

#[derive(Debug, Clone, Copy)]
pub(super) struct BloscHeader {
    pub(super) compversion: u8,
    pub(super) flags: u8,
    pub(super) typesize: usize,
    pub(super) decoded_size: usize,
    pub(super) blocksize: usize,
    pub(super) compressed_size: usize,
}

impl BloscHeader {
    pub(super) fn compformat(self) -> u8 {
        (self.flags & 0xe0) >> 5
    }

    pub(super) fn is_memcpyed(self) -> bool {
        self.flags & blosc_src::BLOSC_MEMCPYED as u8 != 0
    }

    pub(super) fn is_byte_shuffled(self) -> bool {
        self.flags & blosc_src::BLOSC_DOSHUFFLE as u8 != 0 && self.typesize > 1
    }

    pub(super) fn is_bit_shuffled(self) -> bool {
        self.flags & blosc_src::BLOSC_DOBITSHUFFLE as u8 != 0
    }

    pub(super) fn dont_split(self) -> bool {
        self.flags & 0x10 != 0
    }
}

pub(super) fn blosc_header(codec: &str, encoded: &[u8]) -> CodecResult<BloscHeader> {
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
    if compressed_size != encoded.len() {
        return Err(decode_error(codec, "invalid Blosc buffer"));
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

pub(super) fn blosc_decoded_size(codec: &str, encoded: &[u8]) -> CodecResult<usize> {
    blosc_header(codec, encoded).map(|header| header.decoded_size)
}
