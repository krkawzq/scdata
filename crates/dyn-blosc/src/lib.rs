//! Pure-Rust encoding and decoding for Blosc1 and DynBlosc chunks.
//!
//! [`Encoder`] and [`Decoder`] are schema + tool objects: they hold settings or
//! a validated header/index, never the payload itself. Callers supply source or
//! compressed bytes to each encode/decode method. [`Decoder`] dispatches from
//! the wire-format version byte; [`Encoder`] defaults to [`BloscVersion::DynBlosc`]
//! and can emit standard [`BloscVersion::Blosc1`] chunks.
//!
//! Both formats use a fixed [`HEADER_LEN`] (`16`) byte header. Read that many
//! bytes first, then call [`Header::from_bytes`].
#![deny(clippy::undocumented_unsafe_blocks)]
#![deny(unreachable_pub)]

mod bitshuffle;
mod blosclz;
mod codec;
mod compress;
mod decompress;
mod encoder;
mod error;
mod filter;
pub mod format;
mod layout;
mod partition;
mod ranges;
mod shuffle;

pub use decompress::{BlockDecoder, DecodeLimits, DecodeWorkspace, Decoder};
pub use encoder::{EncodeWorkspace, Encoder};
pub use error::{Error, LimitKind, Result};
pub use format::{
    BloscVersion, Codec, Header, Shuffle, BLOSC1_FORMAT_VERSION, BLOSC1_MAX_BLOCK_SIZE,
    BLOSC1_MAX_BUFFER_SIZE, DYN_BLOSC_FORMAT_VERSION, HEADER_LEN,
};
pub use layout::{BlockDescriptor, BlockLayout, BlockRange, Blocks, ChunkLayout, Metadata};
pub use ranges::{ByteMapping, ByteSelection};
