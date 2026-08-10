use crate::error::{Error, Result};
use crate::format::{blosc1, dyn_blosc, Codec, Shuffle};

pub const BLOSC1_FORMAT_VERSION: u8 = blosc1::FORMAT_VERSION;
pub const BLOSC1_MAX_BLOCK_SIZE: usize = blosc1::MAX_BLOCK_SIZE;
pub const BLOSC1_MAX_BUFFER_SIZE: usize = blosc1::MAX_BUFFER_SIZE;
pub const DYN_BLOSC_FORMAT_VERSION: u8 = dyn_blosc::FORMAT_VERSION;

/// Fixed header length shared by Blosc1 and DynBlosc.
pub const HEADER_LEN: usize = 16;

const _: () = assert!(blosc1::HEADER_LEN == HEADER_LEN);
const _: () = assert!(dyn_blosc::HEADER_LEN == HEADER_LEN);

/// Supported Blosc wire-format versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BloscVersion {
    Blosc1,
    DynBlosc,
}

impl BloscVersion {
    pub const fn wire_version(self) -> u8 {
        match self {
            Self::Blosc1 => BLOSC1_FORMAT_VERSION,
            Self::DynBlosc => DYN_BLOSC_FORMAT_VERSION,
        }
    }

    pub(crate) fn detect(input: &[u8]) -> Result<Self> {
        match input.first().copied() {
            Some(BLOSC1_FORMAT_VERSION) => Ok(Self::Blosc1),
            Some(DYN_BLOSC_FORMAT_VERSION) => Ok(Self::DynBlosc),
            Some(version) => Err(Error::InvalidFormat(format!(
                "unsupported Blosc format version {version:#04x}"
            ))),
            None => Err(Error::InvalidFormat("missing Blosc version byte".into())),
        }
    }
}

/// A validated version-specific Blosc header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Header {
    Blosc1(blosc1::Header),
    DynBlosc(dyn_blosc::Header),
}

impl Header {
    /// Parse a Blosc1 or DynBlosc header from a fixed [`HEADER_LEN`]-byte buffer.
    ///
    /// Both formats use the same header length, so callers can always read
    /// exactly [`HEADER_LEN`] bytes first and hand them to this method—no
    /// separate version probe is required to decide how many bytes to fetch.
    pub fn from_bytes(bytes: &[u8; HEADER_LEN]) -> Result<Self> {
        match BloscVersion::detect(bytes)? {
            BloscVersion::Blosc1 => blosc1::Header::parse(bytes).map(Self::Blosc1),
            BloscVersion::DynBlosc => dyn_blosc::Header::parse(bytes).map(Self::DynBlosc),
        }
    }

    /// Parse a header from the leading [`HEADER_LEN`] bytes of `input`.
    pub fn parse(input: &[u8]) -> Result<Self> {
        if input.len() < HEADER_LEN {
            return Err(Error::InvalidFormat(format!(
                "input shorter than Blosc header ({HEADER_LEN} bytes)"
            )));
        }
        let bytes: &[u8; HEADER_LEN] = input[..HEADER_LEN].try_into().unwrap();
        Self::from_bytes(bytes)
    }

    pub fn version(self) -> BloscVersion {
        match self {
            Self::Blosc1(_) => BloscVersion::Blosc1,
            Self::DynBlosc(_) => BloscVersion::DynBlosc,
        }
    }

    pub fn codec_version(self) -> u8 {
        match self {
            Self::Blosc1(header) => header.codec_version(),
            Self::DynBlosc(header) => header.codec_version(),
        }
    }

    pub fn flags(self) -> u8 {
        match self {
            Self::Blosc1(header) => header.flags(),
            Self::DynBlosc(header) => header.flags(),
        }
    }

    pub fn element_size(self) -> usize {
        match self {
            Self::Blosc1(header) => header.element_size(),
            Self::DynBlosc(header) => header.element_size(),
        }
    }

    pub fn decoded_size(self) -> usize {
        match self {
            Self::Blosc1(header) => header.decoded_size(),
            Self::DynBlosc(header) => header.decoded_size(),
        }
    }

    pub fn encoded_size(self) -> usize {
        match self {
            Self::Blosc1(header) => header.encoded_size(),
            Self::DynBlosc(header) => header.encoded_size(),
        }
    }

    pub fn block_count(self) -> usize {
        match self {
            Self::Blosc1(header) => header.block_count(),
            Self::DynBlosc(header) => header.block_count(),
        }
    }

    pub fn codec(self) -> Result<Codec> {
        match self {
            Self::Blosc1(header) => header.codec(),
            Self::DynBlosc(header) => header.codec(),
        }
    }

    pub fn shuffle(self) -> Shuffle {
        match self {
            Self::Blosc1(header) => header.shuffle(),
            Self::DynBlosc(header) => header.shuffle(),
        }
    }

    pub fn split_blocks(self) -> bool {
        match self {
            Self::Blosc1(header) => header.split_blocks(),
            Self::DynBlosc(header) => header.split_blocks(),
        }
    }

    pub fn is_raw(self) -> bool {
        match self {
            Self::Blosc1(header) => header.is_raw(),
            Self::DynBlosc(header) => header.is_raw(),
        }
    }

    pub fn index_prefix_len(self) -> Result<usize> {
        match self {
            Self::Blosc1(header) => header.index_prefix_len(),
            Self::DynBlosc(header) => header.index_prefix_len(),
        }
    }

    pub fn write(self, output: &mut [u8]) -> Result<()> {
        match self {
            Self::Blosc1(header) => header.write(output),
            Self::DynBlosc(header) => header.write(output),
        }
    }
}
