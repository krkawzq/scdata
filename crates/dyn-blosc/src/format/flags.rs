use crate::error::{Error, Result};

/// Wire compressor ids (same enumeration as Blosc flags bits 5..7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Codec {
    BloscLz,
    Lz4,
    Zlib,
    Zstd,
}

impl Codec {
    pub(crate) fn wire_id(self) -> u8 {
        match self {
            Self::BloscLz => 0,
            Self::Lz4 => 1,
            Self::Zlib => 3,
            Self::Zstd => 4,
        }
    }

    pub(crate) fn from_wire_id(id: u8) -> Result<Self> {
        match id {
            0 => Ok(Self::BloscLz),
            1 => Ok(Self::Lz4),
            2 => Err(Error::UnsupportedCodec("snappy".into())),
            3 => Ok(Self::Zlib),
            4 => Ok(Self::Zstd),
            other => Err(Error::UnsupportedCodec(format!("wire id {other}"))),
        }
    }

    pub(crate) fn format_version(self) -> u8 {
        1
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::BloscLz => "blosclz",
            Self::Lz4 => "lz4",
            Self::Zlib => "zlib",
            Self::Zstd => "zstd",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Shuffle {
    None,
    Bytes,
    Bits,
}

pub(super) const FLAG_SHUFFLE: u8 = 0x01;
pub(super) const FLAG_MEMCPY: u8 = 0x02;
pub(super) const FLAG_BITSHUFFLE: u8 = 0x04;
pub(super) const FLAG_DONT_SPLIT: u8 = 0x10;

pub(crate) fn encode_flags(codec: Codec, shuffle: Shuffle, split_blocks: bool, raw: bool) -> u8 {
    let mut flags = codec.wire_id() << 5;
    match shuffle {
        Shuffle::None => {}
        Shuffle::Bytes => flags |= FLAG_SHUFFLE,
        Shuffle::Bits => flags |= FLAG_BITSHUFFLE,
    }
    if !split_blocks {
        flags |= FLAG_DONT_SPLIT;
    }
    if raw {
        flags |= FLAG_MEMCPY;
    }
    flags
}

pub(crate) fn decode_shuffle(flags: u8) -> Shuffle {
    if flags & FLAG_BITSHUFFLE != 0 {
        Shuffle::Bits
    } else if flags & FLAG_SHUFFLE != 0 {
        Shuffle::Bytes
    } else {
        Shuffle::None
    }
}
