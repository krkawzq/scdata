use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::io::{Cursor, Read};
use std::os::raw::c_void;
use std::sync::{Arc, OnceLock, RwLock};

use serde_json::Value;

use super::{CodecError, CodecResult, SharedCodec};

/// Borrowed output memory supplied by the caller.
#[derive(Debug)]
pub struct DecodeBuffer<'a> {
    bytes: &'a mut [u8],
}

impl<'a> DecodeBuffer<'a> {
    pub fn new(bytes: &'a mut [u8]) -> Self {
        Self { bytes }
    }

    pub fn capacity(&self) -> usize {
        self.bytes.len()
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        self.bytes
    }

    fn ensure_capacity(&self, codec: &str, required: usize) -> CodecResult<()> {
        if self.capacity() < required {
            return Err(output_too_small(codec, required, self.capacity()));
        }
        Ok(())
    }

    fn write(self, codec: &str, decoded: &[u8]) -> CodecResult<usize> {
        self.ensure_capacity(codec, decoded.len())?;
        self.bytes[..decoded.len()].copy_from_slice(decoded);
        Ok(decoded.len())
    }
}

/// Decode one numcodecs-compatible zarr chunk into an owned output buffer.
pub trait ChunkCodec: Send + Sync + fmt::Debug + 'static {
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

fn decode_into_vec_no_grow<C: ChunkCodec + ?Sized>(
    codec: &C,
    encoded: &[u8],
    mut output: Vec<u8>,
    expected_size: Option<usize>,
) -> CodecResult<Vec<u8>> {
    if let Some(required) = codec.decoded_size_hint(encoded, expected_size)? {
        if output.capacity() < required {
            return Err(output_too_small(codec.name(), required, output.capacity()));
        }
        set_vec_len_for_decode(&mut output, required);
    }

    let written = codec.decode_into(
        encoded,
        DecodeBuffer::new(output.as_mut_slice()),
        expected_size,
    )?;
    output.truncate(written);
    Ok(output)
}

fn decode_into_vec_grow<C: ChunkCodec + ?Sized>(
    codec: &C,
    encoded: &[u8],
    mut output: Vec<u8>,
    expected_size: Option<usize>,
) -> CodecResult<Vec<u8>> {
    let Some(required) = codec.decoded_size_hint(encoded, expected_size)? else {
        return codec.decode(encoded, expected_size);
    };

    output.clear();
    if output.capacity() < required {
        output.reserve_exact(required - output.capacity());
    }
    set_vec_len_for_decode(&mut output, required);

    let written = codec.decode_into(
        encoded,
        DecodeBuffer::new(output.as_mut_slice()),
        expected_size,
    )?;
    output.truncate(written);
    Ok(output)
}

fn set_vec_len_for_decode(output: &mut Vec<u8>, len: usize) {
    debug_assert!(len <= output.capacity());
    // SAFETY: `Vec<u8>` has no drop glue. Callers either checked capacity or
    // reserved it, and the slice is immediately passed to a decoder that
    // reports how many bytes it initialized. The vector is truncated to that
    // written length before it is returned.
    unsafe {
        output.set_len(len);
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

    fn build_uncached(&self) -> SharedCodec {
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
                cname: optional_string(object.get("cname")).unwrap_or_else(|| "lz4".to_string()),
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

static CODEC_CACHE: OnceLock<RwLock<HashMap<CodecSpec, SharedCodec>>> = OnceLock::new();
static PIPELINE_CACHE: OnceLock<RwLock<HashMap<Vec<CodecSpec>, SharedCodec>>> = OnceLock::new();

fn cached_codec(spec: &CodecSpec) -> SharedCodec {
    let cache = CODEC_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(codec) = cache
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(spec)
    {
        return Arc::clone(codec);
    }

    let codec = spec.build_uncached();
    let mut cache = cache
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(existing) = cache.get(spec) {
        return Arc::clone(existing);
    }
    cache.insert(spec.clone(), Arc::clone(&codec));
    codec
}

pub(crate) fn shared_pipeline_from_specs(specs: &[CodecSpec]) -> SharedCodec {
    cached_pipeline(specs.to_vec())
}

pub(crate) fn shared_pipeline_from_zarr_v2_specs(
    filters: &[CodecSpec],
    compressor: Option<&CodecSpec>,
) -> SharedCodec {
    let mut decode_order = Vec::with_capacity(filters.len() + usize::from(compressor.is_some()));
    if let Some(compressor) = compressor {
        decode_order.push(compressor.clone());
    }
    decode_order.extend(filters.iter().rev().cloned());
    cached_pipeline(decode_order)
}

fn cached_pipeline(specs: Vec<CodecSpec>) -> SharedCodec {
    let cache = PIPELINE_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(codec) = cache
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&specs)
    {
        return Arc::clone(codec);
    }

    let pipeline = CodecPipeline::from_specs(&specs).into_shared();
    let mut cache = cache
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(existing) = cache.get(&specs) {
        return Arc::clone(existing);
    }
    cache.insert(specs, Arc::clone(&pipeline));
    pipeline
}

/// Sequential decode pipeline.
///
/// `CodecPipeline::new` expects specs in decode order. For zarr v2 metadata,
/// use [`CodecPipeline::from_zarr_v2_specs`] so the compressor runs first and
/// filters run in reverse metadata order.
#[derive(Debug, Clone)]
pub struct CodecPipeline {
    codecs: Vec<SharedCodec>,
    name: String,
}

impl CodecPipeline {
    pub fn new(codecs: Vec<SharedCodec>) -> Self {
        let name = if codecs.is_empty() {
            "identity".to_string()
        } else {
            codecs
                .iter()
                .map(|codec| codec.name())
                .collect::<Vec<_>>()
                .join("|")
        };
        Self { codecs, name }
    }

    pub fn from_specs(specs: &[CodecSpec]) -> Self {
        Self::new(specs.iter().map(CodecSpec::build).collect())
    }

    pub fn from_zarr_v2_specs(filters: &[CodecSpec], compressor: Option<&CodecSpec>) -> Self {
        let mut decode_order =
            Vec::with_capacity(filters.len() + usize::from(compressor.is_some()));
        if let Some(compressor) = compressor {
            decode_order.push(compressor.build());
        }
        for filter in filters.iter().rev() {
            decode_order.push(filter.build());
        }
        Self::new(decode_order)
    }

    pub fn from_json_array_str(json: &str) -> CodecResult<Self> {
        let specs = codec_specs_from_json_str(json)?;
        Ok(Self::from_specs(&specs))
    }

    pub fn from_zarr_v2_json_values(
        filters: Option<&Value>,
        compressor: Option<&Value>,
    ) -> CodecResult<Self> {
        let filters = filters
            .map(codec_specs_from_json_value)
            .transpose()?
            .unwrap_or_default();
        let compressor = compressor
            .map(CodecSpec::from_json_value)
            .transpose()?
            .filter(|spec| !matches!(spec, CodecSpec::None));
        Ok(Self::from_zarr_v2_specs(&filters, compressor.as_ref()))
    }

    pub fn into_shared(self) -> SharedCodec {
        Arc::new(self)
    }

    pub fn is_empty(&self) -> bool {
        self.codecs.is_empty()
    }

    pub fn len(&self) -> usize {
        self.codecs.len()
    }

    fn stage_expected_sizes(&self, final_expected: Option<usize>) -> Vec<Option<usize>> {
        let mut sizes = vec![None; self.codecs.len()];
        let mut next_expected = final_expected;
        for idx in (0..self.codecs.len()).rev() {
            sizes[idx] = next_expected;
            next_expected = next_expected.and_then(|size| self.codecs[idx].encoded_size_hint(size));
        }
        sizes
    }
}

impl ChunkCodec for CodecPipeline {
    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        if self.codecs.is_empty() {
            verify_size(self.name(), encoded.len(), expected_size)?;
            return Ok(encoded.to_vec());
        }

        let stage_expected = self.stage_expected_sizes(expected_size);
        let mut current: Option<Vec<u8>> = None;
        let mut spare = Vec::new();
        for (idx, codec) in self.codecs.iter().enumerate() {
            let expected = stage_expected[idx];
            current = Some(match current.take() {
                Some(input) if codec.prefers_decode_owned() => {
                    codec.decode_owned(input, expected)?
                }
                Some(input) => {
                    let output = std::mem::take(&mut spare);
                    let decoded = decode_into_vec_grow(codec.as_ref(), &input, output, expected)?;
                    spare = input;
                    spare.clear();
                    decoded
                }
                None => {
                    let output = std::mem::take(&mut spare);
                    decode_into_vec_grow(codec.as_ref(), encoded, output, expected)?
                }
            });
        }

        current.ok_or_else(|| CodecError::InvalidConfig("empty codec pipeline".to_string()))
    }

    fn decode_into(
        &self,
        encoded: &[u8],
        output: DecodeBuffer<'_>,
        expected_size: Option<usize>,
    ) -> CodecResult<usize> {
        if self.codecs.is_empty() {
            verify_size(self.name(), encoded.len(), expected_size)?;
            return output.write(self.name(), encoded);
        }

        let stage_expected = self.stage_expected_sizes(expected_size);
        let mut current: Option<Vec<u8>> = None;
        let mut spare = Vec::new();
        for (idx, codec) in self.codecs.iter().enumerate() {
            let is_final = idx + 1 == self.codecs.len();
            let expected = stage_expected[idx];
            if is_final {
                return match current.take() {
                    Some(input) => codec.decode_into(&input, output, expected),
                    None => codec.decode_into(encoded, output, expected),
                };
            }

            current = Some(match current.take() {
                Some(input) if codec.prefers_decode_owned() => {
                    codec.decode_owned(input, expected)?
                }
                Some(input) => {
                    let output = std::mem::take(&mut spare);
                    let decoded = decode_into_vec_grow(codec.as_ref(), &input, output, expected)?;
                    spare = input;
                    spare.clear();
                    decoded
                }
                None => {
                    let output = std::mem::take(&mut spare);
                    decode_into_vec_grow(codec.as_ref(), encoded, output, expected)?
                }
            });
        }

        Err(CodecError::InvalidConfig(
            "empty codec pipeline".to_string(),
        ))
    }

    fn decoded_size_hint(
        &self,
        encoded: &[u8],
        expected_size: Option<usize>,
    ) -> CodecResult<Option<usize>> {
        if self.codecs.is_empty() {
            return Ok(Some(encoded.len()));
        }
        if expected_size.is_some() {
            return Ok(expected_size);
        }
        if self.codecs.len() == 1 {
            return self.codecs[0].decoded_size_hint(encoded, None);
        }
        Ok(None)
    }

    fn encoded_size_hint(&self, decoded_size: usize) -> Option<usize> {
        let mut size = decoded_size;
        for codec in self.codecs.iter().rev() {
            size = codec.encoded_size_hint(size)?;
        }
        Some(size)
    }

    fn is_identity(&self) -> bool {
        self.codecs.iter().all(|codec| codec.is_identity())
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

#[derive(Debug, Default)]
pub struct UncompressedCodec;

impl ChunkCodec for UncompressedCodec {
    fn name(&self) -> &str {
        "none"
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        verify_size(self.name(), encoded.len(), expected_size)?;
        Ok(encoded.to_vec())
    }

    fn decode_owned(&self, encoded: Vec<u8>, expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        verify_size(self.name(), encoded.len(), expected_size)?;
        Ok(encoded)
    }

    fn decoded_size_hint(
        &self,
        encoded: &[u8],
        _expected_size: Option<usize>,
    ) -> CodecResult<Option<usize>> {
        Ok(Some(encoded.len()))
    }

    fn encoded_size_hint(&self, decoded_size: usize) -> Option<usize> {
        Some(decoded_size)
    }

    fn prefers_decode_owned(&self) -> bool {
        true
    }

    fn is_identity(&self) -> bool {
        true
    }

    fn decode_to_vec(
        &self,
        encoded: &[u8],
        mut output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        verify_size(self.name(), encoded.len(), expected_size)?;
        if output.capacity() < encoded.len() {
            return Err(output_too_small(
                self.name(),
                encoded.len(),
                output.capacity(),
            ));
        }
        set_vec_len_for_decode(&mut output, encoded.len());
        unsafe {
            std::ptr::copy_nonoverlapping(encoded.as_ptr(), output.as_mut_ptr(), encoded.len());
        }
        Ok(output)
    }

    fn decode_into(
        &self,
        encoded: &[u8],
        mut output: DecodeBuffer<'_>,
        expected_size: Option<usize>,
    ) -> CodecResult<usize> {
        verify_size(self.name(), encoded.len(), expected_size)?;
        output.ensure_capacity(self.name(), encoded.len())?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                encoded.as_ptr(),
                output.as_mut_slice().as_mut_ptr(),
                encoded.len(),
            );
        }
        Ok(encoded.len())
    }
}

#[derive(Debug)]
pub struct UnsupportedCodec {
    name: String,
}

impl UnsupportedCodec {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl ChunkCodec for UnsupportedCodec {
    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn decode(&self, _encoded: &[u8], _expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        Err(CodecError::Unsupported {
            codec: self.name.clone(),
        })
    }
}

#[derive(Debug)]
struct BloscCodec;

#[derive(Debug, Clone, Copy)]
struct BloscHeader {
    compversion: u8,
    flags: u8,
    typesize: usize,
    decoded_size: usize,
    blocksize: usize,
    compressed_size: usize,
}

impl BloscHeader {
    fn compformat(self) -> u8 {
        (self.flags & 0xe0) >> 5
    }

    fn is_memcpyed(self) -> bool {
        self.flags & blosc_src::BLOSC_MEMCPYED as u8 != 0
    }

    fn is_byte_shuffled(self) -> bool {
        self.flags & blosc_src::BLOSC_DOSHUFFLE as u8 != 0 && self.typesize > 1
    }

    fn is_bit_shuffled(self) -> bool {
        self.flags & blosc_src::BLOSC_DOBITSHUFFLE as u8 != 0 && self.blocksize >= self.typesize
    }

    fn dont_split(self) -> bool {
        self.flags & 0x10 != 0
    }
}

fn blosc_header(codec: &str, encoded: &[u8]) -> CodecResult<BloscHeader> {
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

fn blosc_decoded_size(codec: &str, encoded: &[u8]) -> CodecResult<usize> {
    blosc_header(codec, encoded).map(|header| header.decoded_size)
}

thread_local! {
    static BLOSC_LZ4_SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn try_blosc_lz4_decode_into(
    codec: &str,
    encoded: &[u8],
    output: &mut [u8],
    expected_size: Option<usize>,
) -> Option<CodecResult<usize>> {
    let header = match blosc_header(codec, encoded) {
        Ok(header) => header,
        Err(err) => return Some(Err(err)),
    };

    if header.compformat() != blosc_src::BLOSC_LZ4_FORMAT as u8 {
        return None;
    }
    if header.compversion != blosc_src::BLOSC_LZ4_VERSION_FORMAT as u8 {
        return Some(Err(decode_error(
            codec,
            "unsupported Blosc LZ4 format version",
        )));
    }
    if header.flags & 0x08 != 0 {
        return Some(Err(decode_error(codec, "unsupported Blosc future flags")));
    }
    if header.is_bit_shuffled() {
        return None;
    }

    Some(blosc_lz4_decode_into(
        codec,
        encoded,
        header,
        output,
        expected_size,
    ))
}

fn blosc_lz4_decode_into(
    codec: &str,
    encoded: &[u8],
    header: BloscHeader,
    output: &mut [u8],
    expected_size: Option<usize>,
) -> CodecResult<usize> {
    verify_size(codec, header.decoded_size, expected_size)?;
    if output.len() < header.decoded_size {
        return Err(output_too_small(codec, header.decoded_size, output.len()));
    }
    if header.decoded_size == 0 {
        return Ok(0);
    }
    if header.blocksize == 0
        || header.blocksize > header.decoded_size
        || header.blocksize > blosc_src::BLOSC_MAX_BLOCKSIZE as usize
        || header.typesize == 0
        || header.typesize > blosc_src::BLOSC_MAX_TYPESIZE as usize
    {
        return Err(decode_error(codec, "invalid Blosc LZ4 header"));
    }

    if header.is_memcpyed() {
        if header.decoded_size + blosc_src::BLOSC_MAX_OVERHEAD as usize != header.compressed_size {
            return Err(decode_error(codec, "invalid Blosc memcpy buffer"));
        }
        let source = &encoded[blosc_src::BLOSC_MAX_OVERHEAD as usize
            ..blosc_src::BLOSC_MAX_OVERHEAD as usize + header.decoded_size];
        output[..header.decoded_size].copy_from_slice(source);
        return Ok(header.decoded_size);
    }

    let nblocks = header.decoded_size.div_ceil(header.blocksize);
    let bstarts_bytes = nblocks
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(blosc_src::BLOSC_MIN_HEADER_LENGTH as usize))
        .ok_or_else(|| decode_error(codec, "Blosc block table overflow"))?;
    if bstarts_bytes > header.compressed_size {
        return Err(decode_error(codec, "invalid Blosc block table"));
    }

    BLOSC_LZ4_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        if header.is_byte_shuffled() && scratch.len() < header.blocksize {
            scratch.resize(header.blocksize, 0);
        }

        let leftover = header.decoded_size % header.blocksize;
        let mut decoded_offset = 0usize;
        for block_idx in 0..nblocks {
            let is_last = block_idx + 1 == nblocks;
            let bsize = if is_last && leftover > 0 {
                leftover
            } else {
                header.blocksize
            };
            let bstart_offset = blosc_src::BLOSC_MIN_HEADER_LENGTH as usize + block_idx * 4;
            let src_offset = read_i32_le(&encoded[bstart_offset..bstart_offset + 4], codec)?;
            let src_offset = usize::try_from(src_offset)
                .map_err(|_| decode_error(codec, "negative Blosc block offset"))?;
            let output_block = &mut output[decoded_offset..decoded_offset + bsize];

            if header.is_byte_shuffled() {
                let scratch_block = &mut scratch[..bsize];
                decode_blosc_lz4_block(
                    codec,
                    encoded,
                    header,
                    bsize,
                    is_last && leftover > 0,
                    src_offset,
                    scratch_block,
                )?;
                unshuffle_bytes(header.typesize, scratch_block, output_block);
            } else {
                decode_blosc_lz4_block(
                    codec,
                    encoded,
                    header,
                    bsize,
                    is_last && leftover > 0,
                    src_offset,
                    output_block,
                )?;
            }
            decoded_offset += bsize;
        }

        Ok(header.decoded_size)
    })
}

fn decode_blosc_lz4_block(
    codec: &str,
    encoded: &[u8],
    header: BloscHeader,
    blocksize: usize,
    leftover_block: bool,
    mut src_offset: usize,
    output: &mut [u8],
) -> CodecResult<()> {
    let nsplits = if !header.dont_split()
        && header.typesize <= 16
        && header.blocksize / header.typesize >= 128
        && !leftover_block
    {
        header.typesize
    } else {
        1
    };
    let split_size = blocksize / nsplits;
    let mut output_offset = 0usize;
    for _ in 0..nsplits {
        if src_offset + 4 > header.compressed_size {
            return Err(decode_error(codec, "invalid Blosc split offset"));
        }
        let compressed_size = read_i32_le(&encoded[src_offset..src_offset + 4], codec)?;
        src_offset += 4;
        let compressed_size = usize::try_from(compressed_size)
            .map_err(|_| decode_error(codec, "negative Blosc split size"))?;
        if compressed_size > header.compressed_size - src_offset {
            return Err(decode_error(codec, "invalid Blosc split size"));
        }

        let split_output = &mut output[output_offset..output_offset + split_size];
        if compressed_size == split_size {
            split_output.copy_from_slice(&encoded[src_offset..src_offset + compressed_size]);
        } else {
            lz4_decompress_raw_into(
                codec,
                &encoded[src_offset..src_offset + compressed_size],
                split_output,
            )?;
        }
        src_offset += compressed_size;
        output_offset += split_size;
    }
    Ok(())
}

fn read_i32_le(bytes: &[u8], codec: &str) -> CodecResult<i32> {
    let bytes: [u8; 4] = bytes
        .try_into()
        .map_err(|_| decode_error(codec, "short Blosc integer"))?;
    Ok(i32::from_le_bytes(bytes))
}

fn unshuffle_bytes(typesize: usize, shuffled: &[u8], output: &mut [u8]) {
    debug_assert_eq!(shuffled.len(), output.len());
    match typesize {
        2 => unshuffle_2(shuffled, output),
        4 => unshuffle_4(shuffled, output),
        8 => unshuffle_8(shuffled, output),
        _ => unshuffle_generic(typesize, shuffled, output),
    }
}

fn unshuffle_2(shuffled: &[u8], output: &mut [u8]) {
    let elements = output.len() / 2;
    for idx in 0..elements {
        output[idx * 2] = shuffled[idx];
        output[idx * 2 + 1] = shuffled[elements + idx];
    }
    let rem = output.len() % 2;
    if rem != 0 {
        let output_offset = output.len() - rem;
        let shuffled_offset = shuffled.len() - rem;
        output[output_offset..].copy_from_slice(&shuffled[shuffled_offset..]);
    }
}

fn unshuffle_4(shuffled: &[u8], output: &mut [u8]) {
    let elements = output.len() / 4;
    #[cfg(target_endian = "little")]
    unsafe {
        let s0 = shuffled.as_ptr();
        let s1 = s0.add(elements);
        let s2 = s1.add(elements);
        let s3 = s2.add(elements);
        let dst = output.as_mut_ptr();
        for idx in 0..elements {
            let value = (*s0.add(idx) as u32)
                | ((*s1.add(idx) as u32) << 8)
                | ((*s2.add(idx) as u32) << 16)
                | ((*s3.add(idx) as u32) << 24);
            std::ptr::write_unaligned(dst.add(idx * 4).cast::<u32>(), value);
        }
    }
    #[cfg(not(target_endian = "little"))]
    {
        unshuffle_generic(4, shuffled, output);
    }
    let rem = output.len() % 4;
    if rem != 0 {
        let output_offset = output.len() - rem;
        let shuffled_offset = shuffled.len() - rem;
        output[output_offset..].copy_from_slice(&shuffled[shuffled_offset..]);
    }
}

fn unshuffle_8(shuffled: &[u8], output: &mut [u8]) {
    let elements = output.len() / 8;
    for idx in 0..elements {
        let offset = idx * 8;
        for byte in 0..8 {
            output[offset + byte] = shuffled[byte * elements + idx];
        }
    }
    let rem = output.len() % 8;
    if rem != 0 {
        let output_offset = output.len() - rem;
        let shuffled_offset = shuffled.len() - rem;
        output[output_offset..].copy_from_slice(&shuffled[shuffled_offset..]);
    }
}

fn unshuffle_generic(typesize: usize, shuffled: &[u8], output: &mut [u8]) {
    let elements = output.len() / typesize;
    for idx in 0..elements {
        let offset = idx * typesize;
        for byte in 0..typesize {
            output[offset + byte] = shuffled[byte * elements + idx];
        }
    }
    let rem = output.len() % typesize;
    if rem != 0 {
        let output_offset = output.len() - rem;
        let shuffled_offset = shuffled.len() - rem;
        output[output_offset..].copy_from_slice(&shuffled[shuffled_offset..]);
    }
}

impl ChunkCodec for BloscCodec {
    fn name(&self) -> &str {
        "blosc"
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        let decoded_size = blosc_decoded_size(self.name(), encoded)?;

        let mut decoded = Vec::with_capacity(decoded_size);
        set_vec_len_for_decode(&mut decoded, decoded_size);
        if let Some(result) =
            try_blosc_lz4_decode_into(self.name(), encoded, &mut decoded, expected_size)
        {
            let written = result?;
            decoded.truncate(written);
            return Ok(decoded);
        }

        let written = unsafe {
            blosc_src::blosc_decompress_ctx(
                encoded.as_ptr().cast::<c_void>(),
                decoded.as_mut_ptr().cast::<c_void>(),
                decoded.len(),
                1,
            )
        };
        if written < 0 {
            return Err(decode_error(
                self.name(),
                format!("Blosc decompressor returned {written}"),
            ));
        }
        if written as usize != decoded_size {
            return Err(decode_error(
                self.name(),
                format!("expected Blosc to write {decoded_size} bytes, wrote {written}"),
            ));
        }

        verify_size(self.name(), decoded.len(), expected_size)?;
        Ok(decoded)
    }

    fn decoded_size_hint(
        &self,
        encoded: &[u8],
        _expected_size: Option<usize>,
    ) -> CodecResult<Option<usize>> {
        blosc_decoded_size(self.name(), encoded).map(Some)
    }

    fn decode_into(
        &self,
        encoded: &[u8],
        mut output: DecodeBuffer<'_>,
        expected_size: Option<usize>,
    ) -> CodecResult<usize> {
        let decoded_size = blosc_decoded_size(self.name(), encoded)?;
        verify_size(self.name(), decoded_size, expected_size)?;
        output.ensure_capacity(self.name(), decoded_size)?;
        let output = &mut output.as_mut_slice()[..decoded_size];

        if let Some(result) = try_blosc_lz4_decode_into(self.name(), encoded, output, expected_size)
        {
            return result;
        }

        let written = unsafe {
            blosc_src::blosc_decompress_ctx(
                encoded.as_ptr().cast::<c_void>(),
                output.as_mut_ptr().cast::<c_void>(),
                decoded_size,
                1,
            )
        };
        if written < 0 {
            return Err(decode_error(
                self.name(),
                format!("Blosc decompressor returned {written}"),
            ));
        }
        if written as usize != decoded_size {
            return Err(decode_error(
                self.name(),
                format!("expected Blosc to write {decoded_size} bytes, wrote {written}"),
            ));
        }

        Ok(decoded_size)
    }

    fn decode_to_vec(
        &self,
        encoded: &[u8],
        mut output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        let decoded_size = blosc_decoded_size(self.name(), encoded)?;
        verify_size(self.name(), decoded_size, expected_size)?;
        if output.capacity() < decoded_size {
            return Err(output_too_small(
                self.name(),
                decoded_size,
                output.capacity(),
            ));
        }
        set_vec_len_for_decode(&mut output, decoded_size);

        if let Some(result) =
            try_blosc_lz4_decode_into(self.name(), encoded, &mut output, expected_size)
        {
            let written = result?;
            output.truncate(written);
            return Ok(output);
        }

        let written = unsafe {
            blosc_src::blosc_decompress_ctx(
                encoded.as_ptr().cast::<c_void>(),
                output.as_mut_ptr().cast::<c_void>(),
                decoded_size,
                1,
            )
        };
        if written < 0 {
            return Err(decode_error(
                self.name(),
                format!("Blosc decompressor returned {written}"),
            ));
        }
        if written as usize != decoded_size {
            return Err(decode_error(
                self.name(),
                format!("expected Blosc to write {decoded_size} bytes, wrote {written}"),
            ));
        }

        Ok(output)
    }
}

#[derive(Debug)]
struct ZstdCodec;

fn zstd_decoded_size(codec: &str, encoded: &[u8]) -> CodecResult<Option<usize>> {
    match zstd::zstd_safe::get_frame_content_size(encoded) {
        Ok(Some(size)) => usize::try_from(size).map(Some).map_err(|_| {
            decode_error(
                codec,
                format!("Zstd frame content size {size} does not fit in usize"),
            )
        }),
        Ok(None) => Ok(None),
        Err(err) => Err(decode_error(codec, err.to_string())),
    }
}

impl ChunkCodec for ZstdCodec {
    fn name(&self) -> &str {
        "zstd"
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        let decoded_size = match expected_size {
            Some(size) => Some(size),
            None => zstd_decoded_size(self.name(), encoded)?,
        };
        if let Some(decoded_size) = decoded_size {
            let mut decoded = Vec::with_capacity(decoded_size);
            set_vec_len_for_decode(&mut decoded, decoded_size);
            let written = zstd_decompress_into_slice(self.name(), encoded, &mut decoded)?;
            decoded.truncate(written);
            verify_size(self.name(), decoded.len(), expected_size)?;
            return Ok(decoded);
        }

        let decoded = zstd::decode_all(Cursor::new(encoded))
            .map_err(|err| decode_error(self.name(), err.to_string()))?;
        verify_size(self.name(), decoded.len(), expected_size)?;
        Ok(decoded)
    }

    fn decoded_size_hint(
        &self,
        encoded: &[u8],
        expected_size: Option<usize>,
    ) -> CodecResult<Option<usize>> {
        match expected_size {
            Some(size) => Ok(Some(size)),
            None => zstd_decoded_size(self.name(), encoded),
        }
    }

    fn decode_into(
        &self,
        encoded: &[u8],
        mut output: DecodeBuffer<'_>,
        expected_size: Option<usize>,
    ) -> CodecResult<usize> {
        let decoded_size = match expected_size {
            Some(size) => Some(size),
            None => zstd_decoded_size(self.name(), encoded)?,
        };
        if let Some(decoded_size) = decoded_size {
            verify_size(self.name(), decoded_size, expected_size)?;
            output.ensure_capacity(self.name(), decoded_size)?;
            let written = zstd_decompress_into_slice(
                self.name(),
                encoded,
                &mut output.as_mut_slice()[..decoded_size],
            )?;
            verify_size(self.name(), written, Some(decoded_size))?;
            return Ok(written);
        }

        let written = zstd_decompress_into_slice(self.name(), encoded, output.as_mut_slice())?;
        verify_size(self.name(), written, expected_size)?;
        Ok(written)
    }
}

fn zstd_decompress_into_slice(
    codec: &str,
    encoded: &[u8],
    output: &mut [u8],
) -> CodecResult<usize> {
    zstd::zstd_safe::decompress(output, encoded)
        .map_err(|err| decode_error(codec, zstd::zstd_safe::get_error_name(err).to_string()))
}

#[derive(Debug)]
struct GzipCodec;

impl ChunkCodec for GzipCodec {
    fn name(&self) -> &str {
        "gzip"
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        read_all(
            self.name(),
            flate2::read::GzDecoder::new(Cursor::new(encoded)),
            expected_size,
        )
    }

    fn decode_into(
        &self,
        encoded: &[u8],
        output: DecodeBuffer<'_>,
        expected_size: Option<usize>,
    ) -> CodecResult<usize> {
        read_to_buffer(
            self.name(),
            flate2::read::GzDecoder::new(Cursor::new(encoded)),
            output,
            expected_size,
        )
    }
}

#[derive(Debug)]
struct ZlibCodec;

impl ChunkCodec for ZlibCodec {
    fn name(&self) -> &str {
        "zlib"
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        read_all(
            self.name(),
            flate2::read::ZlibDecoder::new(Cursor::new(encoded)),
            expected_size,
        )
    }

    fn decode_into(
        &self,
        encoded: &[u8],
        output: DecodeBuffer<'_>,
        expected_size: Option<usize>,
    ) -> CodecResult<usize> {
        read_to_buffer(
            self.name(),
            flate2::read::ZlibDecoder::new(Cursor::new(encoded)),
            output,
            expected_size,
        )
    }
}

#[derive(Debug)]
struct Lz4Codec;

fn lz4_decoded_size(codec: &str, encoded: &[u8]) -> CodecResult<usize> {
    if encoded.len() < 4 {
        return Err(decode_error(
            codec,
            "LZ4 buffer is shorter than size header",
        ));
    }
    Ok(u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize)
}

impl ChunkCodec for Lz4Codec {
    fn name(&self) -> &str {
        "lz4"
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        let decoded_size = lz4_decoded_size(self.name(), encoded)?;
        let mut decoded = Vec::with_capacity(decoded_size);
        set_vec_len_for_decode(&mut decoded, decoded_size);
        let written = lz4_decompress_into_slice(self.name(), encoded, &mut decoded, expected_size)?;
        decoded.truncate(written);
        Ok(decoded)
    }

    fn decoded_size_hint(
        &self,
        encoded: &[u8],
        _expected_size: Option<usize>,
    ) -> CodecResult<Option<usize>> {
        lz4_decoded_size(self.name(), encoded).map(Some)
    }

    fn decode_into(
        &self,
        encoded: &[u8],
        output: DecodeBuffer<'_>,
        expected_size: Option<usize>,
    ) -> CodecResult<usize> {
        let decoded_size = lz4_decoded_size(self.name(), encoded)?;
        verify_size(self.name(), decoded_size, expected_size)?;
        output.ensure_capacity(self.name(), decoded_size)?;
        lz4_decompress_into_slice(
            self.name(),
            encoded,
            &mut output.bytes[..decoded_size],
            Some(decoded_size),
        )
    }

    fn decode_to_vec(
        &self,
        encoded: &[u8],
        mut output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        let decoded_size = lz4_decoded_size(self.name(), encoded)?;
        verify_size(self.name(), decoded_size, expected_size)?;
        if output.capacity() < decoded_size {
            return Err(output_too_small(
                self.name(),
                decoded_size,
                output.capacity(),
            ));
        }
        set_vec_len_for_decode(&mut output, decoded_size);
        let written =
            lz4_decompress_into_slice(self.name(), encoded, &mut output, Some(decoded_size))?;
        output.truncate(written);
        Ok(output)
    }
}

fn lz4_decompress_into_slice(
    codec: &str,
    encoded: &[u8],
    output: &mut [u8],
    expected_size: Option<usize>,
) -> CodecResult<usize> {
    let decoded_size = lz4_decoded_size(codec, encoded)?;
    verify_size(codec, decoded_size, expected_size)?;
    if output.len() < decoded_size {
        return Err(output_too_small(codec, decoded_size, output.len()));
    }

    lz4_decompress_raw_into(codec, &encoded[4..], &mut output[..decoded_size])?;
    Ok(decoded_size)
}

fn lz4_decompress_raw_into(codec: &str, compressed: &[u8], output: &mut [u8]) -> CodecResult<()> {
    let compressed_size = i32::try_from(compressed.len()).map_err(|_| {
        decode_error(
            codec,
            format!("LZ4 compressed payload is too large: {}", compressed.len()),
        )
    })?;
    let max_decompressed_size = i32::try_from(output.len()).map_err(|_| {
        decode_error(
            codec,
            format!("LZ4 decoded payload is too large: {}", output.len()),
        )
    })?;

    let written = unsafe {
        lz4_sys::LZ4_decompress_safe(
            compressed.as_ptr().cast::<lz4_sys::c_char>(),
            output.as_mut_ptr().cast::<lz4_sys::c_char>(),
            compressed_size,
            max_decompressed_size,
        )
    };
    if written < 0 {
        return Err(decode_error(
            codec,
            format!("LZ4 decompressor returned {written}"),
        ));
    }
    verify_size(codec, written as usize, Some(output.len()))?;
    Ok(())
}

#[derive(Debug)]
struct Bz2Codec;

impl ChunkCodec for Bz2Codec {
    fn name(&self) -> &str {
        "bz2"
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        read_all(
            self.name(),
            bzip2::read::BzDecoder::new(Cursor::new(encoded)),
            expected_size,
        )
    }

    fn decode_into(
        &self,
        encoded: &[u8],
        output: DecodeBuffer<'_>,
        expected_size: Option<usize>,
    ) -> CodecResult<usize> {
        read_to_buffer(
            self.name(),
            bzip2::read::BzDecoder::new(Cursor::new(encoded)),
            output,
            expected_size,
        )
    }
}

#[derive(Debug)]
struct LzmaCodec {
    config: LzmaCodecConfig,
}

impl ChunkCodec for LzmaCodec {
    fn name(&self) -> &str {
        "lzma"
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        if self.config.has_filters {
            return Err(CodecError::InvalidCodecConfig {
                codec: self.name().to_string(),
                message: "raw LZMA filter chains are not supported yet".to_string(),
            });
        }
        if self.config.format != 1 {
            return Err(CodecError::InvalidCodecConfig {
                codec: self.name().to_string(),
                message: format!(
                    "only numcodecs LZMA format=1 (XZ container) is supported, got {}",
                    self.config.format
                ),
            });
        }

        read_all(
            self.name(),
            xz2::read::XzDecoder::new(Cursor::new(encoded)),
            expected_size,
        )
    }

    fn decode_into(
        &self,
        encoded: &[u8],
        output: DecodeBuffer<'_>,
        expected_size: Option<usize>,
    ) -> CodecResult<usize> {
        if self.config.has_filters {
            return Err(CodecError::InvalidCodecConfig {
                codec: self.name().to_string(),
                message: "raw LZMA filter chains are not supported yet".to_string(),
            });
        }
        if self.config.format != 1 {
            return Err(CodecError::InvalidCodecConfig {
                codec: self.name().to_string(),
                message: format!(
                    "only numcodecs LZMA format=1 (XZ container) is supported, got {}",
                    self.config.format
                ),
            });
        }

        read_to_buffer(
            self.name(),
            xz2::read::XzDecoder::new(Cursor::new(encoded)),
            output,
            expected_size,
        )
    }
}

#[derive(Debug)]
struct Crc32Codec;

impl ChunkCodec for Crc32Codec {
    fn name(&self) -> &str {
        "crc32"
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        if encoded.len() < 4 {
            return Err(decode_error(
                self.name(),
                "CRC32 buffer is shorter than 4 bytes",
            ));
        }

        let expected = u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
        let payload = &encoded[4..];
        let actual = crc32fast::hash(payload);
        if actual != expected {
            return Err(decode_error(
                self.name(),
                format!("checksum mismatch: expected {expected:#010x}, got {actual:#010x}"),
            ));
        }

        verify_size(self.name(), payload.len(), expected_size)?;
        Ok(payload.to_vec())
    }

    fn decode_into(
        &self,
        encoded: &[u8],
        output: DecodeBuffer<'_>,
        expected_size: Option<usize>,
    ) -> CodecResult<usize> {
        if encoded.len() < 4 {
            return Err(decode_error(
                self.name(),
                "CRC32 buffer is shorter than 4 bytes",
            ));
        }

        let expected = u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
        let payload = &encoded[4..];
        let actual = crc32fast::hash(payload);
        if actual != expected {
            return Err(decode_error(
                self.name(),
                format!("checksum mismatch: expected {expected:#010x}, got {actual:#010x}"),
            ));
        }

        verify_size(self.name(), payload.len(), expected_size)?;
        output.write(self.name(), payload)
    }

    fn decoded_size_hint(
        &self,
        encoded: &[u8],
        _expected_size: Option<usize>,
    ) -> CodecResult<Option<usize>> {
        if encoded.len() < 4 {
            return Err(decode_error(
                self.name(),
                "CRC32 buffer is shorter than 4 bytes",
            ));
        }
        Ok(Some(encoded.len() - 4))
    }

    fn encoded_size_hint(&self, decoded_size: usize) -> Option<usize> {
        decoded_size.checked_add(4)
    }

    fn decode_owned(
        &self,
        mut encoded: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        if encoded.len() < 4 {
            return Err(decode_error(
                self.name(),
                "CRC32 buffer is shorter than 4 bytes",
            ));
        }

        let expected = u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
        let payload_len = encoded.len() - 4;
        let actual = crc32fast::hash(&encoded[4..]);
        if actual != expected {
            return Err(decode_error(
                self.name(),
                format!("checksum mismatch: expected {expected:#010x}, got {actual:#010x}"),
            ));
        }

        verify_size(self.name(), payload_len, expected_size)?;
        encoded.copy_within(4.., 0);
        encoded.truncate(payload_len);
        Ok(encoded)
    }

    fn prefers_decode_owned(&self) -> bool {
        true
    }
}

fn read_all(
    codec: &str,
    mut reader: impl Read,
    expected_size: Option<usize>,
) -> CodecResult<Vec<u8>> {
    let mut decoded = Vec::with_capacity(expected_size.unwrap_or(0));
    reader
        .read_to_end(&mut decoded)
        .map_err(|err| decode_error(codec, err.to_string()))?;
    verify_size(codec, decoded.len(), expected_size)?;
    Ok(decoded)
}

fn read_to_buffer(
    codec: &str,
    mut reader: impl Read,
    mut output: DecodeBuffer<'_>,
    expected_size: Option<usize>,
) -> CodecResult<usize> {
    if let Some(expected_size) = expected_size {
        output.ensure_capacity(codec, expected_size)?;
    }

    let capacity = output.capacity();
    let mut written = 0usize;
    while written < capacity {
        let read = reader
            .read(&mut output.as_mut_slice()[written..])
            .map_err(|err| decode_error(codec, err.to_string()))?;
        if read == 0 {
            verify_size(codec, written, expected_size)?;
            return Ok(written);
        }
        written += read;
    }

    let mut extra = [0u8; 1];
    let read = reader
        .read(&mut extra)
        .map_err(|err| decode_error(codec, err.to_string()))?;
    if read == 0 {
        verify_size(codec, written, expected_size)?;
        return Ok(written);
    }

    Err(output_too_small(
        codec,
        expected_size.unwrap_or(written + read),
        capacity,
    ))
}

fn verify_size(codec: &str, actual: usize, expected: Option<usize>) -> CodecResult<()> {
    if let Some(expected) = expected {
        if actual != expected {
            return Err(CodecError::SizeMismatch {
                codec: codec.to_string(),
                expected,
                actual,
            });
        }
    }
    Ok(())
}

fn output_too_small(codec: &str, required: usize, capacity: usize) -> CodecError {
    CodecError::OutputTooSmall {
        codec: codec.to_string(),
        required,
        capacity,
    }
}

fn decode_error(codec: &str, message: impl Into<String>) -> CodecError {
    CodecError::Decode {
        codec: codec.to_string(),
        message: message.into(),
    }
}

fn optional_string(value: Option<&Value>) -> Option<String> {
    value.and_then(Value::as_str).map(ToString::to_string)
}

fn optional_bool(value: Option<&Value>, codec: &str, field: &str) -> CodecResult<Option<bool>> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    value
        .as_bool()
        .map(Some)
        .ok_or_else(|| CodecError::InvalidCodecConfig {
            codec: codec.to_string(),
            message: format!("field `{field}` must be a boolean"),
        })
}

fn optional_i32(value: Option<&Value>, codec: &str, field: &str) -> CodecResult<Option<i32>> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let number = value
        .as_i64()
        .ok_or_else(|| CodecError::InvalidCodecConfig {
            codec: codec.to_string(),
            message: format!("field `{field}` must be an integer"),
        })?;
    i32::try_from(number)
        .map(Some)
        .map_err(|_| CodecError::InvalidCodecConfig {
            codec: codec.to_string(),
            message: format!("field `{field}` is out of i32 range"),
        })
}

fn optional_u32(value: Option<&Value>, codec: &str, field: &str) -> CodecResult<Option<u32>> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let number = value
        .as_u64()
        .ok_or_else(|| CodecError::InvalidCodecConfig {
            codec: codec.to_string(),
            message: format!("field `{field}` must be a non-negative integer"),
        })?;
    u32::try_from(number)
        .map(Some)
        .map_err(|_| CodecError::InvalidCodecConfig {
            codec: codec.to_string(),
            message: format!("field `{field}` is out of u32 range"),
        })
}

fn optional_u8(value: Option<&Value>, codec: &str, field: &str) -> CodecResult<Option<u8>> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let number = value
        .as_u64()
        .ok_or_else(|| CodecError::InvalidCodecConfig {
            codec: codec.to_string(),
            message: format!("field `{field}` must be a non-negative integer"),
        })?;
    u8::try_from(number)
        .map(Some)
        .map_err(|_| CodecError::InvalidCodecConfig {
            codec: codec.to_string(),
            message: format!("field `{field}` is out of u8 range"),
        })
}

fn optional_usize(value: Option<&Value>, codec: &str, field: &str) -> CodecResult<Option<usize>> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let number = value
        .as_u64()
        .ok_or_else(|| CodecError::InvalidCodecConfig {
            codec: codec.to_string(),
            message: format!("field `{field}` must be a non-negative integer"),
        })?;
    usize::try_from(number)
        .map(Some)
        .map_err(|_| CodecError::InvalidCodecConfig {
            codec: codec.to_string(),
            message: format!("field `{field}` is out of usize range"),
        })
}
fn optional_blosc_shuffle(value: Option<&Value>) -> CodecResult<Option<BloscShuffle>> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    match value {
        Value::Number(number) => match number.as_i64() {
            Some(0) => Ok(BloscShuffle::NoShuffle),
            Some(1) => Ok(BloscShuffle::Shuffle),
            Some(2) => Ok(BloscShuffle::BitShuffle),
            _ => Err(CodecError::InvalidCodecConfig {
                codec: "blosc".to_string(),
                message: "field `shuffle` must be 0, 1, or 2".to_string(),
            }),
        },
        Value::Bool(false) => Ok(BloscShuffle::NoShuffle),
        Value::Bool(true) => Ok(BloscShuffle::Shuffle),
        Value::String(text) => match text.to_ascii_lowercase().as_str() {
            "none" | "noshuffle" | "no_shuffle" => Ok(BloscShuffle::NoShuffle),
            "shuffle" | "byte" => Ok(BloscShuffle::Shuffle),
            "bitshuffle" | "bit_shuffle" => Ok(BloscShuffle::BitShuffle),
            _ => Err(CodecError::InvalidCodecConfig {
                codec: "blosc".to_string(),
                message: format!("unknown shuffle mode `{text}`"),
            }),
        },
        _ => Err(CodecError::InvalidCodecConfig {
            codec: "blosc".to_string(),
            message: "field `shuffle` must be an integer, boolean, or string".to_string(),
        }),
    }
    .map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn shared_pipeline_cache_reuses_pipeline() {
        let specs = vec![
            CodecSpec::Zstd(ZstdCodecConfig::default()),
            CodecSpec::Crc32,
        ];

        let first = shared_pipeline_from_specs(&specs);
        let second = shared_pipeline_from_specs(&specs);
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
