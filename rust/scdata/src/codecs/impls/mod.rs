mod blosc;
mod crc32;
mod identity;
mod lz4;
mod stream;
mod unsupported;
mod zstd;

pub(crate) use blosc::{
    blosc_lz4_block_split_count, blosc_lz4_header_table_len_from_prefix, decode_blosc_lz4_block,
    decode_blosc_lz4_block_partial_prefixes, try_blosc_lz4_plan_from_encoded,
    try_blosc_lz4_plan_from_prefix, unshuffle_bytes, BloscCodec, BloscHeader, BloscLz4Block,
    BloscLz4Plan, ValidatedBloscBlockRange,
};
pub(crate) use crc32::Crc32Codec;
pub use identity::UncompressedCodec;
pub(crate) use lz4::{lz4_decompress_raw_into, lz4_decompress_raw_partial_into, Lz4Codec};
pub(crate) use stream::{Bz2Codec, GzipCodec, LzmaCodec, ZlibCodec};
pub use unsupported::UnsupportedCodec;
pub(crate) use zstd::ZstdCodec;
