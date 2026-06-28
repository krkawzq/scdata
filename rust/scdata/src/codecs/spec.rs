use std::fmt;
use std::sync::Arc;

use serde_json::Value;

use super::buffer::{decode_into_vec_no_grow, DecodeBuffer};
use super::impls::{
    BloscCodec, Bz2Codec, Crc32Codec, GzipCodec, Lz4Codec, LzmaCodec, UncompressedCodec,
    UnsupportedCodec, ZlibCodec, ZstdCodec,
};
use super::registry::cached_codec;
use super::util::{
    optional_blosc_shuffle, optional_bool, optional_i32, optional_string, optional_u32,
    optional_u8, optional_usize,
};
use super::{CodecError, CodecResult, SharedCodec};

pub(crate) mod sealed {
    pub trait Sealed {}
}

/// Decode one numcodecs-compatible zarr chunk into an owned output buffer.
pub trait ChunkCodec: sealed::Sealed + Send + Sync + fmt::Debug + 'static {
    fn name(&self) -> &str;

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>>;

    /// Decode into caller-owned output memory and return the number of bytes written.
    fn decode_into(
        &self,
        encoded: &[u8],
        output: DecodeBuffer<'_>,
        expected_size: Option<usize>,
    ) -> CodecResult<usize> {
        let decoded = self.decode(encoded, expected_size)?;
        output.write(self.name(), &decoded)
    }

    #[doc(hidden)]
    fn decode_owned(&self, encoded: Vec<u8>, expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        self.decode(&encoded, expected_size)
    }

    #[doc(hidden)]
    fn decode_to_vec(
        &self,
        encoded: &[u8],
        output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        decode_into_vec_no_grow(self, encoded, output, expected_size)
    }

    #[doc(hidden)]
    fn decode_to_vec_grow(
        &self,
        encoded: &[u8],
        mut output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        let Some(required) = self.decoded_size_hint(encoded, expected_size)? else {
            return self.decode(encoded, expected_size);
        };

        output.clear();
        if output.capacity() < required {
            output.reserve_exact(required - output.capacity());
        }
        self.decode_to_vec(encoded, output, expected_size)
    }

    #[doc(hidden)]
    fn decode_to_capacity_vec(
        &self,
        encoded: &[u8],
        output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        if self.decoded_size_hint(encoded, expected_size)?.is_none() {
            return self.decode(encoded, expected_size);
        }
        self.decode_to_vec(encoded, output, expected_size)
    }

    #[doc(hidden)]
    fn decoded_size_hint(
        &self,
        _encoded: &[u8],
        expected_size: Option<usize>,
    ) -> CodecResult<Option<usize>> {
        Ok(expected_size)
    }

    #[doc(hidden)]
    fn encoded_size_hint(&self, _decoded_size: usize) -> Option<usize> {
        None
    }

    #[doc(hidden)]
    fn prefers_decode_owned(&self) -> bool {
        false
    }

    #[doc(hidden)]
    fn is_identity(&self) -> bool {
        false
    }
}

/// Parsed numcodecs metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CodecSpec {
    None,
    Blosc(BloscCodecConfig),
    Zstd(ZstdCodecConfig),
    Gzip(LevelCodecConfig),
    Zlib(LevelCodecConfig),
    Lz4(Lz4CodecConfig),
    Bz2(LevelCodecConfig),
    Lzma(LzmaCodecConfig),
    Crc32,
    Unknown(String),
}

impl CodecSpec {
    pub fn name(&self) -> &str {
        match self {
            Self::None => "none",
            Self::Blosc(_) => "blosc",
            Self::Zstd(_) => "zstd",
            Self::Gzip(_) => "gzip",
            Self::Zlib(_) => "zlib",
            Self::Lz4(_) => "lz4",
            Self::Bz2(_) => "bz2",
            Self::Lzma(_) => "lzma",
            Self::Crc32 => "crc32",
            Self::Unknown(name) => name.as_str(),
        }
    }

    pub fn build(&self) -> SharedCodec {
        cached_codec(self)
    }

    pub(crate) fn build_uncached(&self) -> SharedCodec {
        match self {
            Self::None => Arc::new(UncompressedCodec),
            Self::Blosc(_) => Arc::new(BloscCodec),
            Self::Zstd(_) => Arc::new(ZstdCodec),
            Self::Gzip(_) => Arc::new(GzipCodec),
            Self::Zlib(_) => Arc::new(ZlibCodec),
            Self::Lz4(_) => Arc::new(Lz4Codec),
            Self::Bz2(_) => Arc::new(Bz2Codec),
            Self::Lzma(config) => Arc::new(LzmaCodec {
                config: config.clone(),
            }),
            Self::Crc32 => Arc::new(Crc32Codec),
            Self::Unknown(name) => Arc::new(UnsupportedCodec::new(name)),
        }
    }

    pub fn from_json_str(json: &str) -> CodecResult<Self> {
        let value = serde_json::from_str(json).map_err(|err| CodecError::InvalidCodecConfig {
            codec: "json".to_string(),
            message: err.to_string(),
        })?;
        Self::from_json_value(&value)
    }

    pub fn from_json_value(value: &Value) -> CodecResult<Self> {
        let Some(object) = value.as_object() else {
            if value.is_null() {
                return Ok(Self::None);
            }
            return Err(CodecError::InvalidCodecConfig {
                codec: "unknown".to_string(),
                message: "codec config must be a JSON object or null".to_string(),
            });
        };

        let id = object
            .get("id")
            .or_else(|| object.get("name"))
            .and_then(Value::as_str)
            .ok_or_else(|| CodecError::InvalidCodecConfig {
                codec: "unknown".to_string(),
                message: "codec config is missing string field `id`".to_string(),
            })?
            .to_ascii_lowercase();

        match id.as_str() {
            "none" | "null" => Ok(Self::None),
            "blosc" => Ok(Self::Blosc(BloscCodecConfig {
                cname: optional_string(object.get("cname"), "blosc", "cname")?
                    .unwrap_or_else(|| "lz4".to_string()),
                clevel: optional_u8(object.get("clevel"), "blosc", "clevel")?,
                shuffle: optional_blosc_shuffle(object.get("shuffle"))?,
                typesize: optional_usize(object.get("typesize"), "blosc", "typesize")?,
                blocksize: optional_usize(object.get("blocksize"), "blosc", "blocksize")?,
            })),
            "zstd" => Ok(Self::Zstd(ZstdCodecConfig {
                level: optional_i32(object.get("level"), "zstd", "level")?,
                checksum: optional_bool(object.get("checksum"), "zstd", "checksum")?,
            })),
            "gzip" => Ok(Self::Gzip(LevelCodecConfig {
                level: optional_i32(object.get("level"), "gzip", "level")?,
            })),
            "zlib" => Ok(Self::Zlib(LevelCodecConfig {
                level: optional_i32(object.get("level"), "zlib", "level")?,
            })),
            "lz4" => Ok(Self::Lz4(Lz4CodecConfig {
                acceleration: optional_i32(object.get("acceleration"), "lz4", "acceleration")?,
            })),
            "bz2" => Ok(Self::Bz2(LevelCodecConfig {
                level: optional_i32(object.get("level"), "bz2", "level")?,
            })),
            "lzma" => Ok(Self::Lzma(LzmaCodecConfig {
                format: optional_i32(object.get("format"), "lzma", "format")?.unwrap_or(1),
                check: optional_i32(object.get("check"), "lzma", "check")?,
                preset: optional_u32(object.get("preset"), "lzma", "preset")?,
                has_filters: object.get("filters").is_some_and(|value| !value.is_null()),
            })),
            "crc32" => Ok(Self::Crc32),
            _ => Ok(Self::Unknown(id)),
        }
    }
}

pub fn codec_specs_from_json_str(json: &str) -> CodecResult<Vec<CodecSpec>> {
    let value = serde_json::from_str(json).map_err(|err| CodecError::InvalidCodecConfig {
        codec: "json".to_string(),
        message: err.to_string(),
    })?;
    codec_specs_from_json_value(&value)
}

pub fn codec_specs_from_json_value(value: &Value) -> CodecResult<Vec<CodecSpec>> {
    match value {
        Value::Null => Ok(Vec::new()),
        Value::Array(items) => items.iter().map(CodecSpec::from_json_value).collect(),
        _ => Err(CodecError::InvalidCodecConfig {
            codec: "filters".to_string(),
            message: "filters must be a JSON array or null".to_string(),
        }),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BloscCodecConfig {
    pub cname: String,
    pub clevel: Option<u8>,
    pub shuffle: Option<BloscShuffle>,
    pub typesize: Option<usize>,
    pub blocksize: Option<usize>,
}

impl BloscCodecConfig {
    pub fn new(cname: impl Into<String>) -> Self {
        Self {
            cname: cname.into(),
            clevel: None,
            shuffle: None,
            typesize: None,
            blocksize: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BloscShuffle {
    NoShuffle,
    Shuffle,
    BitShuffle,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct ZstdCodecConfig {
    pub level: Option<i32>,
    pub checksum: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct LevelCodecConfig {
    pub level: Option<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Lz4CodecConfig {
    pub acceleration: Option<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LzmaCodecConfig {
    pub format: i32,
    pub check: Option<i32>,
    pub preset: Option<u32>,
    pub has_filters: bool,
}

impl Default for LzmaCodecConfig {
    fn default() -> Self {
        Self {
            format: 1,
            check: None,
            preset: None,
            has_filters: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::codecs::CodecPipeline;

    const RAW_HEX: &str = "7363646174612d6e756d636f646563732d636f6d7061743a000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f7363646174612d6e756d636f646563732d636f6d7061743a000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f7363646174612d6e756d636f646563732d636f6d7061743a000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f7363646174612d6e756d636f646563732d636f6d7061743a000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f";

    #[test]
    fn uncompressed_codec_copies_input() {
        let codec = UncompressedCodec;
        let decoded = codec
            .decode(b"abcdef", Some(6))
            .expect("uncompressed decode");
        assert_eq!(&decoded, b"abcdef");
    }

    #[test]
    fn uncompressed_codec_checks_expected_size() {
        let codec = UncompressedCodec;
        let err = codec.decode(b"abcdef", Some(5)).expect_err("size mismatch");
        assert!(matches!(err, CodecError::SizeMismatch { .. }));
    }

    #[test]
    fn decode_into_writes_caller_buffer() {
        let codec = UncompressedCodec;
        let mut output = [0u8; 8];

        let written = codec
            .decode_into(b"abcdef", DecodeBuffer::new(&mut output), Some(6))
            .expect("decode into output");

        assert_eq!(written, 6);
        assert_eq!(&output[..6], b"abcdef");
        assert_eq!(&output[6..], &[0, 0]);
    }

    #[test]
    fn decode_into_rejects_small_output_buffer() {
        let codec = UncompressedCodec;
        let mut output = [0u8; 3];

        let err = codec
            .decode_into(b"abcdef", DecodeBuffer::new(&mut output), Some(6))
            .expect_err("small output should fail");

        assert!(
            matches!(err, CodecError::OutputTooSmall { codec, required: 6, capacity: 3 } if codec == "none")
        );
    }

    #[test]
    fn unsupported_codec_reports_name() {
        let codec = CodecSpec::Unknown("unknown".to_string()).build();
        let err = codec.decode(b"payload", None).expect_err("unsupported");
        assert!(matches!(err, CodecError::Unsupported { codec } if codec == "unknown"));
    }

    #[test]
    fn build_reuses_cached_shared_codecs() {
        let spec = CodecSpec::from_json_str(
            r#"{"blocksize":0,"clevel":5,"cname":"zstd","id":"blosc","shuffle":1}"#,
        )
        .expect("parse config");

        let first = spec.build();
        let second = spec.build();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn codec_cache_ignores_encode_only_config() {
        let zstd_fast = CodecSpec::Zstd(ZstdCodecConfig {
            level: Some(1),
            checksum: Some(false),
        })
        .build();
        let zstd_slow = CodecSpec::Zstd(ZstdCodecConfig {
            level: Some(19),
            checksum: Some(true),
        })
        .build();
        assert!(Arc::ptr_eq(&zstd_fast, &zstd_slow));

        let blosc_lz4 = CodecSpec::Blosc(BloscCodecConfig::new("lz4")).build();
        let mut blosc_zstd_config = BloscCodecConfig::new("zstd");
        blosc_zstd_config.clevel = Some(9);
        blosc_zstd_config.shuffle = Some(BloscShuffle::BitShuffle);
        let blosc_zstd = CodecSpec::Blosc(blosc_zstd_config).build();
        assert!(Arc::ptr_eq(&blosc_lz4, &blosc_zstd));
    }

    #[test]
    fn lzma_cache_key_keeps_decode_config() {
        let default = CodecSpec::Lzma(LzmaCodecConfig::default()).build();
        let encode_only_config = LzmaCodecConfig {
            check: Some(-1),
            preset: Some(9),
            ..Default::default()
        };
        let encode_only = CodecSpec::Lzma(encode_only_config).build();
        assert!(Arc::ptr_eq(&default, &encode_only));

        let raw_filters_config = LzmaCodecConfig {
            has_filters: true,
            ..Default::default()
        };
        let raw_filters = CodecSpec::Lzma(raw_filters_config).build();
        assert!(!Arc::ptr_eq(&default, &raw_filters));
    }

    #[test]
    fn shared_pipeline_cache_reuses_pipeline() {
        let specs = vec![
            CodecSpec::Zstd(ZstdCodecConfig::default()),
            CodecSpec::Crc32,
        ];

        let first = crate::codecs::codec_pipeline_from_specs(&specs);
        let second = crate::codecs::codec_pipeline_from_specs(&specs);
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn single_stage_shared_pipeline_reuses_codec() {
        let spec = CodecSpec::Zstd(ZstdCodecConfig::default());
        let codec = spec.build();
        let pipeline = crate::codecs::codec_pipeline_from_specs(&[spec]);
        assert!(Arc::ptr_eq(&codec, &pipeline));
    }

    #[test]
    fn pipeline_cache_uses_normalized_codec_keys() {
        let first = crate::codecs::codec_pipeline_from_specs(&[
            CodecSpec::Zstd(ZstdCodecConfig {
                level: Some(1),
                checksum: Some(false),
            }),
            CodecSpec::Crc32,
        ]);
        let second = crate::codecs::codec_pipeline_from_specs(&[
            CodecSpec::Zstd(ZstdCodecConfig {
                level: Some(19),
                checksum: Some(true),
            }),
            CodecSpec::Crc32,
        ]);
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn empty_pipeline_is_identity() {
        let codec = CodecPipeline::new(Vec::new());
        assert!(codec.is_empty());
        assert!(codec.is_identity());

        let decoded = codec.decode(b"abcdef", Some(6)).expect("identity decode");
        assert_eq!(&decoded, b"abcdef");
    }

    #[test]
    fn pipeline_applies_final_size_check_only_once() {
        let codec = CodecPipeline::new(vec![
            Arc::new(UncompressedCodec),
            Arc::new(UncompressedCodec),
        ]);
        assert_eq!(codec.len(), 2);
        assert!(codec.is_identity());

        let decoded = codec.decode(b"abcdef", Some(6)).expect("pipeline decode");
        assert_eq!(&decoded, b"abcdef");

        let err = codec.decode(b"abcdef", Some(5)).expect_err("size mismatch");
        assert!(matches!(err, CodecError::SizeMismatch { codec, .. } if codec == "none"));
    }

    #[test]
    fn parses_numcodecs_config_json() {
        let spec = CodecSpec::from_json_str(
            r#"{"blocksize":0,"clevel":5,"cname":"zstd","id":"blosc","shuffle":1}"#,
        )
        .expect("parse blosc config");

        match spec {
            CodecSpec::Blosc(config) => {
                assert_eq!(config.cname, "zstd");
                assert_eq!(config.shuffle, Some(BloscShuffle::Shuffle));
            }
            other => panic!("expected blosc spec, got {other:?}"),
        }
    }

    #[test]
    fn rejects_non_string_blosc_cname() {
        let err = CodecSpec::from_json_str(r#"{"id":"blosc","cname":7}"#)
            .expect_err("invalid cname type");

        assert!(
            matches!(err, CodecError::InvalidCodecConfig { codec, message } if codec == "blosc" && message.contains("cname"))
        );
    }

    #[test]
    fn decodes_numcodecs_vectors() {
        let raw = decode_hex(RAW_HEX);
        let cases = [
            (
                r#"{"checksum":false,"id":"zstd","level":3}"#,
                "28b52ffd6060000d030084057363646174612d6e756d636f646563732d636f6d7061743a000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f010058c1a68628",
            ),
            (
                r#"{"id":"gzip","level":5}"#,
                "1f8b080043cf3f6a00ff2b4e4e492c49d4cd2bcd4dce4f494d2ed64dcecf2d482cb16260646266616563e7e0e4e2e6e1e5e3171014121611151397909492969195935750545256515553d7d0d4d2d6d1d5d33730343236313533b7b0b4b2b6b1b5b32f1e35176c2e00f39a977060010000",
            ),
            (
                r#"{"id":"zlib","level":1}"#,
                "78012b4e4e492c49d4cd2bcd4dce4f494d2ed64dcecf2d482cb16260646266616563e7e0e4e2e6e1e5e3171014121611151397909492969195935750545256515553d7d0d4d2d6d1d5d33730343236313533b7b0b4b2b6b1b5b32f1e35171c0e00325744a5",
            ),
            (
                r#"{"acceleration":1,"id":"lz4"}"#,
                "60010000ff497363646174612d6e756d636f646563732d636f6d7061743a000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f5800f0503b3c3d3e3f",
            ),
            (
                r#"{"id":"bz2","level":1}"#,
                "425a68313141592653591cb10cbd0000a9f9807fffffffffffffffae03ce003000d8073004609880601304d190d0c02608c4c73004609880601304d190d0c02608c4c155501a9a1a1847a8c08c04686468686988d30d4f4998924a259309a4070138e1388e3279c8729cc5028948a6739d042749d4542a958ae582c9d676168b65c2e978be603b4ee309de781e27f1a8f23ccc87a1ea7b194f73e0f9331f44668341a4fb3f0d440466d21222122222320379289a4c101248cae379fe2ee48a70a1203962197a",
            ),
            (
                r#"{"check":-1,"filters":null,"format":1,"id":"lzma","preset":null}"#,
                "fd377a585a000004e6d6b4460200210116000000742fe5a3e0015f005b5d003998c8bdec457a6b29835fbbac03ac4a88e3cb1c02c3847b5df8a43dc03d13409bb42c9ecce41a1de08dcefe36634529ec1ee480c952473bdebbad90e8844983fa2116a5285021d3899df1e26895a6de5c271ade9dc019cb96000000009441567d5c7400e0000177e002000000f0e6af57b1c467fb020000000004595a",
            ),
            (
                r#"{"id":"crc32"}"#,
                "f39a97707363646174612d6e756d636f646563732d636f6d7061743a000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f7363646174612d6e756d636f646563732d636f6d7061743a000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f7363646174612d6e756d636f646563732d636f6d7061743a000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f7363646174612d6e756d636f646563732d636f6d7061743a000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f",
            ),
            (
                r#"{"blocksize":0,"clevel":5,"cname":"zstd","id":"blosc","shuffle":1}"#,
                "02019101600100006001000083000000140000006b00000028b52ffd6060000d030084057363646174612d6e756d636f646563732d636f6d7061743a000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f010058c1a68628",
            ),
            (
                r#"{"blocksize":0,"clevel":5,"cname":"lz4","id":"blosc","shuffle":2}"#,
                "02012401600100006001000087000000140000006f000000bf6bef2faaaaaaaaaaaaaaaa0b000ebf83cc86cccccccccccccccc0b000ebfd43b4df0f0f0f0f0f0f0f00b000ebfc00a8d00ff00ff00ff00ff0b000eaf1181d00000ffff0000ff0b000f53ffffff000009000f0b000dbfbfff7e00000000000000000b000e0f020014500000000000",
            ),
        ];

        for (json, encoded_hex) in cases {
            let spec = CodecSpec::from_json_str(json).expect("parse config");
            let encoded = decode_hex(encoded_hex);
            let decoded = spec
                .build()
                .decode(&encoded, Some(raw.len()))
                .unwrap_or_else(|err| panic!("{json} failed: {err}"));
            assert_eq!(decoded, raw, "failed config {json}");
        }
    }

    #[test]
    fn zarr_v2_pipeline_decodes_compressor_then_reversed_filters() {
        let raw = decode_hex(RAW_HEX);
        let filters_json: Value =
            serde_json::from_str(r#"[{"id":"crc32"}]"#).expect("filters json");
        let compressor_json: Value =
            serde_json::from_str(r#"{"checksum":false,"id":"zstd","level":3}"#)
                .expect("compressor json");
        let encoded = decode_hex("28b52ffd6064002d0300c405f39a97707363646174612d6e756d636f646563732d636f6d7061743a000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f01005cc1a68628");

        let pipeline =
            CodecPipeline::from_zarr_v2_json_values(Some(&filters_json), Some(&compressor_json))
                .expect("build pipeline");
        assert_eq!(pipeline.name(), "zstd|crc32");

        let decoded = pipeline
            .decode(&encoded, Some(raw.len()))
            .expect("pipeline decode");
        assert_eq!(decoded, raw);
    }

    #[test]
    fn zarr_v2_pipeline_decodes_into_caller_buffer() {
        let raw = decode_hex(RAW_HEX);
        let filtered = crc32_encode(&raw);
        let encoded = zstd::encode_all(Cursor::new(&filtered), 3).expect("zstd encode");
        let pipeline = CodecPipeline::from_zarr_v2_specs(
            &[CodecSpec::Crc32],
            Some(&CodecSpec::Zstd(ZstdCodecConfig::default())),
        );
        let mut output = vec![0u8; raw.len()];

        let written = pipeline
            .decode_into(&encoded, DecodeBuffer::new(&mut output), Some(raw.len()))
            .expect("pipeline decode into output");

        assert_eq!(written, raw.len());
        assert_eq!(output, raw);
    }

    #[test]
    fn crc32_rejects_checksum_mismatch() {
        let codec = CodecSpec::Crc32.build();
        let mut encoded = decode_hex("f39a9770616263");
        encoded[4] = b'z';

        let err = codec.decode(&encoded, None).expect_err("crc mismatch");
        assert!(matches!(err, CodecError::Decode { codec, .. } if codec == "crc32"));
    }

    fn decode_hex(hex: &str) -> Vec<u8> {
        assert_eq!(hex.len() % 2, 0);
        (0..hex.len())
            .step_by(2)
            .map(|idx| u8::from_str_radix(&hex[idx..idx + 2], 16).expect("valid hex"))
            .collect()
    }

    fn crc32_encode(raw: &[u8]) -> Vec<u8> {
        let mut encoded = Vec::with_capacity(raw.len() + 4);
        encoded.extend_from_slice(&crc32fast::hash(raw).to_le_bytes());
        encoded.extend_from_slice(raw);
        encoded
    }
}
