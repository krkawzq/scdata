use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use super::pipeline::CodecPipeline;
use super::spec::CodecSpec;
use super::SharedCodec;

static CODEC_CACHE: OnceLock<RwLock<HashMap<CodecKey, SharedCodec>>> = OnceLock::new();
static PIPELINE_CACHE: OnceLock<RwLock<HashMap<Vec<CodecKey>, SharedCodec>>> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CodecKey {
    None,
    Blosc,
    Zstd,
    Gzip,
    Zlib,
    Lz4,
    Bz2,
    Lzma { format: i32, has_filters: bool },
    Crc32,
    Unknown(String),
}

impl CodecKey {
    fn from_spec(spec: &CodecSpec) -> Self {
        match spec {
            CodecSpec::None => Self::None,
            CodecSpec::Blosc(_) => Self::Blosc,
            CodecSpec::Zstd(_) => Self::Zstd,
            CodecSpec::Gzip(_) => Self::Gzip,
            CodecSpec::Zlib(_) => Self::Zlib,
            CodecSpec::Lz4(_) => Self::Lz4,
            CodecSpec::Bz2(_) => Self::Bz2,
            CodecSpec::Lzma(config) => Self::Lzma {
                format: config.format,
                has_filters: config.has_filters,
            },
            CodecSpec::Crc32 => Self::Crc32,
            CodecSpec::Unknown(name) => Self::Unknown(name.clone()),
        }
    }
}

pub(crate) fn cached_codec(spec: &CodecSpec) -> SharedCodec {
    let key = CodecKey::from_spec(spec);
    let cache = CODEC_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(codec) = cache
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&key)
    {
        return Arc::clone(codec);
    }

    let codec = spec.build_uncached();
    let mut cache = cache
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(existing) = cache.get(&key) {
        return Arc::clone(existing);
    }
    cache.insert(key, Arc::clone(&codec));
    codec
}

pub(crate) fn shared_pipeline_from_specs(specs: &[CodecSpec]) -> SharedCodec {
    cached_pipeline(specs)
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
    cached_pipeline(&decode_order)
}

fn cached_pipeline(specs: &[CodecSpec]) -> SharedCodec {
    if let [spec] = specs {
        return spec.build();
    }

    let key: Vec<_> = specs.iter().map(CodecKey::from_spec).collect();
    let cache = PIPELINE_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(codec) = cache
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&key)
    {
        return Arc::clone(codec);
    }

    let pipeline = CodecPipeline::from_specs(specs).into_shared();
    let mut cache = cache
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(existing) = cache.get(&key) {
        return Arc::clone(existing);
    }
    cache.insert(key, Arc::clone(&pipeline));
    pipeline
}
