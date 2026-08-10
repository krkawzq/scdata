use std::io::{Read, Write};

use dyn_blosc::{BloscVersion, DecodeLimits, Decoder, Encoder, Shuffle, BLOSC1_MAX_BLOCK_SIZE};
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::limits::ReadLimits;
use crate::partition::{block_lengths_bytes, BlockTable};

/// Codec used inside a Blosc1 or DynBlosc chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BloscCodec {
    BloscLz,
    Lz4,
    Zlib,
    Zstd,
}

impl From<BloscCodec> for dyn_blosc::Codec {
    fn from(codec: BloscCodec) -> Self {
        match codec {
            BloscCodec::BloscLz => Self::BloscLz,
            BloscCodec::Lz4 => Self::Lz4,
            BloscCodec::Zlib => Self::Zlib,
            BloscCodec::Zstd => Self::Zstd,
        }
    }
}

/// Byte/bit reordering applied before Blosc compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShuffleMode {
    None,
    Bytes,
    Bits,
}

impl From<ShuffleMode> for Shuffle {
    fn from(shuffle: ShuffleMode) -> Self {
        match shuffle {
            ShuffleMode::None => Self::None,
            ShuffleMode::Bytes => Self::Bytes,
            ShuffleMode::Bits => Self::Bits,
        }
    }
}

/// Settings shared by Blosc1 and DynBlosc compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BloscOptions {
    pub codec: BloscCodec,
    #[serde(rename = "clevel")]
    pub compression_level: u8,
    pub shuffle: ShuffleMode,
    #[serde(default)]
    pub split_blocks: bool,
}

impl Default for BloscOptions {
    fn default() -> Self {
        Self {
            codec: BloscCodec::Lz4,
            compression_level: 5,
            shuffle: ShuffleMode::Bytes,
            split_blocks: false,
        }
    }
}

impl BloscOptions {
    #[must_use]
    pub const fn compression_level(mut self, level: u8) -> Self {
        self.compression_level = level;
        self
    }

    #[must_use]
    pub const fn shuffle(mut self, shuffle: ShuffleMode) -> Self {
        self.shuffle = shuffle;
        self
    }

    #[must_use]
    pub const fn split_blocks(mut self, enabled: bool) -> Self {
        self.split_blocks = enabled;
        self
    }

    fn validate(self) -> Result<()> {
        if self.compression_level > 9 {
            return Err(Error::invalid_argument(format!(
                "blosc compression level {} out of range 0..=9",
                self.compression_level
            )));
        }
        Ok(())
    }
}

/// Compression algorithm recorded in `meta.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "id", rename_all = "kebab-case")]
pub enum Compressor {
    Blosc1 {
        #[serde(flatten)]
        options: BloscOptions,
        /// Fixed decoded block size in bytes.
        block_size: u32,
    },
    DynBlosc {
        #[serde(flatten)]
        options: BloscOptions,
    },
    Zstd {
        #[serde(default = "default_zstd_level")]
        level: i32,
    },
    Zlib {
        #[serde(default = "default_zlib_level")]
        level: u32,
    },
    /// LZ4 block with a four-byte little-endian decoded-size prefix.
    Lz4,
    None,
}

const fn default_zstd_level() -> i32 {
    3
}

const fn default_zlib_level() -> u32 {
    6
}

impl Default for Compressor {
    fn default() -> Self {
        Self::dyn_blosc_lz4()
    }
}

impl Compressor {
    pub const DEFAULT_BLOSC1_BLOCK_SIZE: u32 = 64 * 1024;

    pub fn blosc1(options: BloscOptions, block_size: u32) -> Self {
        Self::Blosc1 {
            options,
            block_size,
        }
    }

    pub fn dyn_blosc(options: BloscOptions) -> Self {
        Self::DynBlosc { options }
    }

    pub fn blosc1_lz4(block_size: u32) -> Self {
        Self::blosc1(BloscOptions::default(), block_size)
    }

    pub fn dyn_blosc_lz4() -> Self {
        Self::dyn_blosc(BloscOptions::default())
    }

    pub const fn zstd(level: i32) -> Self {
        Self::Zstd { level }
    }

    pub const fn zlib(level: u32) -> Self {
        Self::Zlib { level }
    }

    pub const fn lz4() -> Self {
        Self::Lz4
    }

    pub const fn none() -> Self {
        Self::None
    }

    pub const fn id(&self) -> &'static str {
        match self {
            Self::Blosc1 { .. } => "blosc1",
            Self::DynBlosc { .. } => "dyn-blosc",
            Self::Zstd { .. } => "zstd",
            Self::Zlib { .. } => "zlib",
            Self::Lz4 => "lz4",
            Self::None => "none",
        }
    }

    pub const fn is_blosc1(&self) -> bool {
        matches!(self, Self::Blosc1 { .. })
    }

    pub const fn is_dyn_blosc(&self) -> bool {
        matches!(self, Self::DynBlosc { .. })
    }

    pub const fn blosc1_block_size(&self) -> Option<u32> {
        match self {
            Self::Blosc1 { block_size, .. } => Some(*block_size),
            _ => None,
        }
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Blosc1 {
                options,
                block_size,
            } => {
                options.validate()?;
                if *block_size == 0 {
                    return Err(Error::invalid_argument(
                        "blosc1 block_size must be non-zero",
                    ));
                }
                let block_size = usize::try_from(*block_size).map_err(|_| {
                    Error::invalid_argument("blosc1 block_size exceeds platform usize")
                })?;
                if block_size > BLOSC1_MAX_BLOCK_SIZE {
                    return Err(Error::invalid_argument(format!(
                        "blosc1 block_size {block_size} exceeds {BLOSC1_MAX_BLOCK_SIZE}"
                    )));
                }
                Ok(())
            }
            Self::DynBlosc { options } => options.validate(),
            Self::Zstd { level } => {
                if !zstd::compression_level_range().contains(level) {
                    return Err(Error::invalid_argument(format!(
                        "zstd level {level} is outside the supported range"
                    )));
                }
                Ok(())
            }
            Self::Zlib { level } => {
                if *level > 9 {
                    return Err(Error::invalid_argument(format!(
                        "zlib level {level} out of range 0..=9"
                    )));
                }
                Ok(())
            }
            Self::Lz4 | Self::None => Ok(()),
        }
    }

    pub(crate) fn encode_buffer(&self, source: &[u8], element_size: usize) -> Result<Vec<u8>> {
        self.validate()?;
        match self {
            Self::Blosc1 {
                options,
                block_size,
            } => encode_blosc(
                source,
                element_size,
                BloscEncoding {
                    version: BloscVersion::Blosc1,
                    options: *options,
                    partition: BloscPartition::Fixed(usize::try_from(*block_size).map_err(
                        |_| Error::invalid_argument("blosc1 block_size exceeds platform usize"),
                    )?),
                },
            ),
            Self::DynBlosc { options } => encode_blosc(
                source,
                element_size,
                BloscEncoding {
                    version: BloscVersion::DynBlosc,
                    options: *options,
                    partition: BloscPartition::Automatic,
                },
            ),
            Self::Zstd { level } => encode_zstd(source, *level),
            Self::Zlib { level } => {
                let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(*level));
                encoder.write_all(source)?;
                Ok(encoder.finish()?)
            }
            Self::Lz4 => {
                u32::try_from(source.len()).map_err(|_| {
                    Error::invalid_argument("lz4 source length exceeds the u32 size prefix")
                })?;
                Ok(lz4_flex::compress_prepend_size(source))
            }
            Self::None => fallible_copy(source),
        }
    }

    pub(crate) fn encode_partitioned(
        &self,
        source: &[u8],
        element_size: usize,
        blocks: &BlockTable,
    ) -> Result<Vec<u8>> {
        self.validate()?;
        let Self::DynBlosc { options } = self else {
            return Err(Error::invalid_argument(
                "variable block lengths require dyn-blosc",
            ));
        };
        encode_blosc(
            source,
            element_size,
            BloscEncoding {
                version: BloscVersion::DynBlosc,
                options: *options,
                partition: BloscPartition::Variable(block_lengths_bytes(blocks, element_size)?),
            },
        )
    }

    pub fn decode_exact(&self, encoded: &[u8], expected: usize) -> Result<Vec<u8>> {
        self.decode_exact_with_limits(encoded, expected, ReadLimits::default())
    }

    pub fn decode_exact_with_limits(
        &self,
        encoded: &[u8],
        expected: usize,
        limits: ReadLimits,
    ) -> Result<Vec<u8>> {
        self.validate()?;
        limits.check_encoded(encoded.len(), self.id())?;
        limits.check_decoded(expected, self.id())?;
        let decoded = match self {
            Self::Blosc1 { .. } | Self::DynBlosc { .. } => {
                let decoder_limits = DecodeLimits::unlimited()
                    .maximum_decoded_size(expected)
                    .maximum_block_size(expected)
                    .maximum_block_count(limits.block_count());
                let decoder = Decoder::from_encoded_with_limits(encoded, decoder_limits)?;
                let expected_version = if self.is_blosc1() {
                    BloscVersion::Blosc1
                } else {
                    BloscVersion::DynBlosc
                };
                if decoder.metadata().version != expected_version {
                    return Err(Error::corrupt(
                        "blosc chunk",
                        format!(
                            "metadata declares {}, encoded chunk is {:?}",
                            self.id(),
                            decoder.metadata().version
                        ),
                    ));
                }
                decoder.decode(encoded)?
            }
            Self::Zstd { .. } => decode_zstd_exact(encoded, expected)?,
            Self::Zlib { .. } => decode_zlib_limited(encoded, expected)?,
            Self::Lz4 => decode_lz4_exact(encoded, expected)?,
            Self::None => {
                if encoded.len() != expected {
                    return Err(Error::corrupt(
                        "none payload",
                        format!(
                            "encoded length {} does not match expected {expected}",
                            encoded.len()
                        ),
                    ));
                }
                fallible_copy(encoded)?
            }
        };
        if decoded.len() != expected {
            return Err(Error::corrupt(
                format!("{} payload", self.id()),
                format!(
                    "decoded length {} does not match expected {expected}",
                    decoded.len()
                ),
            ));
        }
        Ok(decoded)
    }
}

fn fallible_copy(source: &[u8]) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    output.try_reserve_exact(source.len())?;
    output.extend_from_slice(source);
    Ok(output)
}

fn encode_zstd(source: &[u8], level: i32) -> Result<Vec<u8>> {
    let maximum = zstd::zstd_safe::compress_bound(source.len());
    let mut output = Vec::new();
    output.try_reserve_exact(maximum)?;
    let mut encoder = zstd::bulk::Compressor::new(level)
        .map_err(|error| Error::corrupt("zstd encoder", error.to_string()))?;
    encoder
        .compress_to_buffer(source, &mut output)
        .map_err(|error| Error::corrupt("zstd encoder", error.to_string()))?;
    Ok(output)
}

fn decode_zstd_exact(encoded: &[u8], expected: usize) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    output.try_reserve_exact(expected)?;
    let mut decoder = zstd::bulk::Decompressor::new()
        .map_err(|error| Error::corrupt("zstd stream", error.to_string()))?;
    decoder
        .decompress_to_buffer(encoded, &mut output)
        .map_err(|error| Error::corrupt("zstd stream", error.to_string()))?;
    Ok(output)
}

struct BloscEncoding {
    version: BloscVersion,
    options: BloscOptions,
    partition: BloscPartition,
}

enum BloscPartition {
    Automatic,
    Fixed(usize),
    Variable(Vec<usize>),
}

fn encode_blosc(source: &[u8], element_size: usize, encoding: BloscEncoding) -> Result<Vec<u8>> {
    if !(1..=usize::from(u8::MAX)).contains(&element_size) {
        return Err(Error::invalid_argument(format!(
            "blosc element size {element_size} out of range 1..=255"
        )));
    }
    let mut encoder = Encoder::new()
        .version(encoding.version)
        .codec(encoding.options.codec.into())
        .compression_level(encoding.options.compression_level)
        .shuffle(if source.is_empty() {
            Shuffle::None
        } else {
            encoding.options.shuffle.into()
        })
        .element_size(element_size)
        .split_blocks(encoding.options.split_blocks);
    if !source.is_empty() {
        encoder = match encoding.partition {
            BloscPartition::Automatic => encoder.automatic_block_size(),
            BloscPartition::Fixed(bytes) => encoder.block_size(bytes),
            BloscPartition::Variable(lengths) if lengths.is_empty() => {
                return Err(Error::invalid_argument(
                    "non-empty dyn-blosc source requires at least one block",
                ));
            }
            BloscPartition::Variable(lengths) => encoder.block_lengths(lengths),
        };
    }
    Ok(encoder.encode(source)?)
}

fn decode_zlib_limited(encoded: &[u8], expected: usize) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    output.try_reserve_exact(expected)?;
    output.resize(expected, 0);
    let mut decoder = ZlibDecoder::new(encoded);
    let mut decoded = 0;
    while decoded < expected {
        let count = decoder.read(&mut output[decoded..])?;
        if count == 0 {
            break;
        }
        decoded += count;
    }
    output.truncate(decoded);
    let mut excess = [0u8; 1];
    if decoder.read(&mut excess)? != 0 {
        return Err(Error::corrupt(
            "zlib stream",
            format!("decoded length exceeds expected {expected}"),
        ));
    }
    if decoder.total_in()
        != u64::try_from(encoded.len())
            .map_err(|_| Error::invalid_argument("zlib encoded size exceeds u64"))?
    {
        return Err(Error::corrupt(
            "zlib stream",
            "trailing bytes after compressed stream",
        ));
    }
    Ok(output)
}

fn decode_lz4_exact(encoded: &[u8], expected: usize) -> Result<Vec<u8>> {
    let prefix: [u8; 4] = encoded
        .get(..4)
        .ok_or_else(|| Error::corrupt("lz4 block", "missing decoded-size prefix"))?
        .try_into()
        .map_err(|_| Error::corrupt("lz4 block", "invalid decoded-size prefix"))?;
    let declared = usize::try_from(u32::from_le_bytes(prefix))
        .map_err(|_| Error::corrupt("lz4 block", "decoded size exceeds platform usize"))?;
    if declared != expected {
        return Err(Error::corrupt(
            "lz4 block",
            format!("declared decoded length {declared} does not match expected {expected}"),
        ));
    }
    lz4_flex::decompress_size_prepended(encoded)
        .map_err(|error| Error::corrupt("lz4 block", error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_codec_roundtrips_and_checks_exact_size() {
        let source = b"single-cell compression payload".repeat(32);
        for compressor in [
            Compressor::blosc1_lz4(256),
            Compressor::dyn_blosc_lz4(),
            Compressor::zstd(1),
            Compressor::zlib(1),
            Compressor::lz4(),
            Compressor::none(),
        ] {
            let encoded = compressor.encode_buffer(&source, 1).unwrap();
            assert_eq!(
                compressor.decode_exact(&encoded, source.len()).unwrap(),
                source
            );
            assert!(compressor.decode_exact(&encoded, source.len() - 1).is_err());
        }
    }

    #[test]
    fn decode_limits_are_checked_before_expansion() {
        let source = vec![0u8; 4096];
        let encoded = Compressor::zlib(1).encode_buffer(&source, 1).unwrap();
        let limits = ReadLimits::default().maximum_decoded_size(64);
        assert!(Compressor::zlib(1)
            .decode_exact_with_limits(&encoded, source.len(), limits)
            .is_err());
    }

    #[test]
    fn zstd_capacity_overflow_returns_allocation_error() {
        let compressor = Compressor::zstd(1);
        let encoded = compressor.encode_buffer(b"payload", 1).unwrap();
        assert!(matches!(
            compressor.decode_exact_with_limits(&encoded, usize::MAX, ReadLimits::unlimited()),
            Err(Error::Allocation(_))
        ));
    }

    #[test]
    fn invalid_codec_settings_are_rejected() {
        assert!(
            Compressor::dyn_blosc(BloscOptions::default().compression_level(10))
                .validate()
                .is_err()
        );
        assert!(Compressor::zlib(10).validate().is_err());
        assert!(Compressor::blosc1_lz4(0).validate().is_err());
    }

    #[test]
    fn lz4_declared_size_must_match_expected() {
        let encoded = Compressor::lz4().encode_buffer(b"payload", 1).unwrap();
        assert!(Compressor::lz4().decode_exact(&encoded, 6).is_err());
    }

    #[test]
    fn codecs_reject_trailing_bytes() {
        let source = b"payload";
        for compressor in [
            Compressor::blosc1_lz4(64),
            Compressor::dyn_blosc_lz4(),
            Compressor::zstd(1),
            Compressor::zlib(1),
            Compressor::lz4(),
            Compressor::none(),
        ] {
            let mut encoded = compressor.encode_buffer(source, 1).unwrap();
            encoded.push(0);
            assert!(
                compressor.decode_exact(&encoded, source.len()).is_err(),
                "{} accepted trailing bytes",
                compressor.id()
            );
        }
    }
}
