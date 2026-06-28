//! Codec encode fixtures and decode helpers shared by the `modules`,
//! `stress`, and `codec_manifest` benches.
//!
//! These produce numcodecs-compatible encoded payloads so the benches can
//! exercise the decode paths without going through the Python exporter.
//! `encode_for_spec` is the single dispatch entry point: given a `CodecSpec`
//! and raw bytes, it produces the encoded payload the matching decoder expects.

use std::ffi::CString;
use std::io::{Cursor, Write};
use std::os::raw::c_void;
use std::sync::Arc;

use _scdata::codecs::{
    BloscCodecConfig, BloscShuffle, CodecSpec, DecodeBuffer, LevelCodecConfig, Lz4CodecConfig,
    LzmaCodecConfig, SharedCodec, ZstdCodecConfig,
};

/// Decode `encoded` into a freshly allocated buffer and return a checksum of
/// the result. The buffer is allocated per call — pair with a `_reused` bench
/// to isolate allocation cost.
pub fn decode_into_checksum(codec: &SharedCodec, encoded: &[u8], expected_size: usize) -> usize {
    let mut output = vec![0u8; expected_size];
    let written = codec
        .decode_into(encoded, DecodeBuffer::new(&mut output), Some(expected_size))
        .expect("decode");
    written ^ output[written / 2] as usize
}

/// Encode `raw` for the given spec. This dispatches to the same encoder each
/// codec's decoder expects (zstd frame, lz4 size-prefixed block, blosc buffer,
/// crc32-prefixed payload, etc.). `CodecSpec::Unknown` has no encoder and
/// panics — it only exists for the decode-side error path.
pub fn encode_for_spec(spec: &CodecSpec, raw: &[u8]) -> Vec<u8> {
    match spec {
        CodecSpec::None => raw.to_vec(),
        CodecSpec::Zstd(config) => zstd_encode(raw, config.level.unwrap_or(3)),
        CodecSpec::Lz4(_) => lz4_flex::block::compress_prepend_size(raw),
        CodecSpec::Zlib(config) => zlib_encode(raw, config.level.unwrap_or(1)),
        CodecSpec::Gzip(config) => gzip_encode(raw, config.level.unwrap_or(1)),
        CodecSpec::Bz2(config) => bz2_encode(raw, config.level.unwrap_or(5)),
        CodecSpec::Lzma(_) => lzma_encode(raw),
        CodecSpec::Crc32 => crc32_encode(raw),
        CodecSpec::Blosc(config) => blosc_encode_spec(raw, config),
        CodecSpec::Unknown(name) => panic!("cannot encode unknown codec `{name}`"),
    }
}

/// Default codec matrix used by the synth bench mode: covers every supported
/// codec with a representative level, plus a crc32+zstd pipeline.
pub fn default_codec_matrix() -> Vec<(&'static str, CodecSpec)> {
    vec![
        ("none", CodecSpec::None),
        (
            "zstd3",
            CodecSpec::Zstd(ZstdCodecConfig {
                level: Some(3),
                checksum: Some(false),
            }),
        ),
        (
            "zstd9",
            CodecSpec::Zstd(ZstdCodecConfig {
                level: Some(9),
                checksum: Some(false),
            }),
        ),
        ("lz4", CodecSpec::Lz4(Lz4CodecConfig::default())),
        (
            "zlib1",
            CodecSpec::Zlib(LevelCodecConfig { level: Some(1) }),
        ),
        (
            "gzip5",
            CodecSpec::Gzip(LevelCodecConfig { level: Some(5) }),
        ),
        ("bz2_5", CodecSpec::Bz2(LevelCodecConfig { level: Some(5) })),
        ("lzma", CodecSpec::Lzma(LzmaCodecConfig::default())),
        ("crc32", CodecSpec::Crc32),
        (
            "blosc_lz4_shuf",
            CodecSpec::Blosc(BloscCodecConfig {
                cname: "lz4".to_string(),
                clevel: Some(5),
                shuffle: Some(BloscShuffle::Shuffle),
                typesize: Some(4),
                blocksize: None,
            }),
        ),
        (
            "blosc_zstd_shuf",
            CodecSpec::Blosc(BloscCodecConfig {
                cname: "zstd".to_string(),
                clevel: Some(5),
                shuffle: Some(BloscShuffle::Shuffle),
                typesize: Some(4),
                blocksize: None,
            }),
        ),
    ]
}

pub fn zstd_encode(raw: &[u8], level: i32) -> Vec<u8> {
    zstd::encode_all(Cursor::new(raw), level).expect("zstd encode")
}

pub fn zlib_encode(raw: &[u8], level: i32) -> Vec<u8> {
    let mut encoder =
        flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(level as u32));
    encoder.write_all(raw).expect("zlib write");
    encoder.finish().expect("zlib finish")
}

pub fn gzip_encode(raw: &[u8], level: i32) -> Vec<u8> {
    let mut encoder =
        flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level as u32));
    encoder.write_all(raw).expect("gzip write");
    encoder.finish().expect("gzip finish")
}

pub fn bz2_encode(raw: &[u8], level: i32) -> Vec<u8> {
    let mut encoder =
        bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::new(level as u32));
    encoder.write_all(raw).expect("bz2 write");
    encoder.finish().expect("bz2 finish")
}

pub fn lzma_encode(raw: &[u8]) -> Vec<u8> {
    let mut encoder = xz2::write::XzEncoder::new(Vec::new(), 6);
    encoder.write_all(raw).expect("xz write");
    encoder.finish().expect("xz finish")
}

/// CRC32 numcodecs layout: 4-byte little-endian checksum prefix + payload.
pub fn crc32_encode(raw: &[u8]) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(raw.len() + 4);
    encoded.extend_from_slice(&crc32fast::hash(raw).to_le_bytes());
    encoded.extend_from_slice(raw);
    encoded
}

/// Compress with the C blosc library directly so we can vary cname / shuffle /
/// clevel independently of this crate's `CodecSpec` parser.
pub fn blosc_encode(
    raw: &[u8],
    typesize: usize,
    doshuffle: i32,
    clevel: i32,
    compressor: &str,
) -> Vec<u8> {
    let dest_size = raw.len() + blosc_src::BLOSC_MAX_OVERHEAD as usize;
    let mut dest = vec![0u8; dest_size];
    let compressor_c = CString::new(compressor).expect("compressor name");
    let written = unsafe {
        blosc_src::blosc_compress_ctx(
            clevel,
            doshuffle,
            typesize,
            raw.len(),
            raw.as_ptr() as *const c_void,
            dest.as_mut_ptr() as *mut c_void,
            dest_size,
            compressor_c.as_ptr(),
            0,
            1,
        )
    };
    assert!(written > 0, "blosc_compress_ctx failed: {written}");
    dest.truncate(written as usize);
    dest
}

/// blosc encode driven by a parsed `BloscCodecConfig` (cname / clevel / shuffle
/// / typesize). Shuffle maps to the C library's doshuffle flag.
fn blosc_encode_spec(raw: &[u8], config: &BloscCodecConfig) -> Vec<u8> {
    let typesize = config.typesize.unwrap_or(1).max(1);
    let doshuffle = match config.shuffle.unwrap_or(BloscShuffle::Shuffle) {
        BloscShuffle::NoShuffle => 0,
        BloscShuffle::Shuffle => 1,
        BloscShuffle::BitShuffle => 2,
    };
    let clevel = i32::from(config.clevel.unwrap_or(5));
    blosc_encode(raw, typesize, doshuffle, clevel, &config.cname)
}

/// Build a pipeline payload: `raw -> inner_codec -> crc32` prefix, matching a
/// `[Crc32, inner]` decode-order pipeline.
pub fn crc32_wrapped(inner_encoded: &[u8]) -> Vec<u8> {
    crc32_encode(inner_encoded)
}

/// Shared zstd-encoded payload used across codec and pool benches.
pub fn zstd_encoded_arc(raw: &[u8], level: i32) -> Arc<[u8]> {
    Arc::from(zstd::encode_all(Cursor::new(raw), level).expect("zstd encode"))
}
