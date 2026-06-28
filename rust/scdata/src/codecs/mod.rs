//! Chunk codec registry and decode worker pool.
//!
//! The scheduler should submit one decode request per caller. IO may be
//! deduplicated upstream, but decoded outputs stay per-request so callers own
//! their returned buffers.

mod error;
mod pool;
mod spec;

use std::sync::Arc;

pub use error::{CodecError, CodecResult};
pub use pool::{DecodeFuture, DecodePool, DecodePoolConfig, DecodeRequest};
pub use spec::{
    codec_specs_from_json_str, codec_specs_from_json_value, BloscCodecConfig, BloscShuffle,
    ChunkCodec, CodecPipeline, CodecSpec, DecodeBuffer, LevelCodecConfig, Lz4CodecConfig,
    LzmaCodecConfig, UncompressedCodec, UnsupportedCodec, ZstdCodecConfig,
};

/// Shared codec implementation used by decode requests.
pub type SharedCodec = Arc<dyn ChunkCodec>;

/// Build a codec implementation from parsed zarr/numcodecs metadata.
pub fn codec_from_spec(spec: &CodecSpec) -> SharedCodec {
    spec.build()
}

pub fn codec_from_json_str(json: &str) -> CodecResult<SharedCodec> {
    Ok(CodecSpec::from_json_str(json)?.build())
}

/// Build a sequential codec pipeline from parsed zarr filters/compressor.
pub fn codec_pipeline_from_specs(specs: &[CodecSpec]) -> SharedCodec {
    spec::shared_pipeline_from_specs(specs)
}

/// Build a zarr v2 decode pipeline from metadata-order filters plus compressor.
pub fn codec_pipeline_from_zarr_v2_specs(
    filters: &[CodecSpec],
    compressor: Option<&CodecSpec>,
) -> SharedCodec {
    spec::shared_pipeline_from_zarr_v2_specs(filters, compressor)
}

pub fn codec_pipeline_from_zarr_v2_json_str(
    filters_json: Option<&str>,
    compressor_json: Option<&str>,
) -> CodecResult<SharedCodec> {
    let filters = filters_json.map(parse_json_value).transpose()?;
    let compressor = compressor_json.map(parse_json_value).transpose()?;
    let filters = filters
        .as_ref()
        .map(codec_specs_from_json_value)
        .transpose()?
        .unwrap_or_default();
    let compressor = compressor
        .as_ref()
        .map(CodecSpec::from_json_value)
        .transpose()?
        .filter(|spec| !matches!(spec, CodecSpec::None));
    Ok(spec::shared_pipeline_from_zarr_v2_specs(
        &filters,
        compressor.as_ref(),
    ))
}

fn parse_json_value(json: &str) -> CodecResult<serde_json::Value> {
    serde_json::from_str(json).map_err(|err| CodecError::InvalidCodecConfig {
        codec: "json".to_string(),
        message: err.to_string(),
    })
}
