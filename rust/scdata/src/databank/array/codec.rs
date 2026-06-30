use crate::codecs::{
    codec_from_json_str, codec_from_spec, codec_pipeline_from_specs,
    codec_pipeline_from_zarr_v2_json_str, codec_pipeline_from_zarr_v2_specs,
    codec_specs_from_json_str, codec_specs_from_json_value, CodecSpec, SharedCodec,
};

use crate::databank::error::DataBankResult;

#[derive(Debug, Clone, Default)]
pub enum ArrayCodecSpec {
    #[default]
    Uncompressed,
    CodecJson(String),
    CodecJsonValue(serde_json::Value),
    PipelineJson(String),
    PipelineJsonValue(serde_json::Value),
    /// Zarr v2 filter/compressor JSON converted into this crate's codec
    /// pipeline. This does not imply a storage layout.
    ZarrV2Json {
        filters: Option<String>,
        compressor: Option<String>,
    },
    /// Zarr v2 filter/compressor JSON values converted into this crate's codec
    /// pipeline. This does not imply a storage layout.
    ZarrV2JsonValue {
        filters: Option<serde_json::Value>,
        compressor: Option<serde_json::Value>,
    },
}

impl ArrayCodecSpec {
    pub(super) fn build(&self) -> DataBankResult<SharedCodec> {
        match self {
            Self::Uncompressed => Ok(codec_from_spec(&CodecSpec::None)),
            Self::CodecJson(json) => Ok(codec_from_json_str(json)?),
            Self::CodecJsonValue(value) => Ok(CodecSpec::from_json_value(value)?.build()),
            Self::PipelineJson(json) => {
                let specs = codec_specs_from_json_str(json)?;
                Ok(codec_pipeline_from_specs(&specs))
            }
            Self::PipelineJsonValue(value) => {
                let specs = codec_specs_from_json_value(value)?;
                Ok(codec_pipeline_from_specs(&specs))
            }
            Self::ZarrV2Json {
                filters,
                compressor,
            } => Ok(codec_pipeline_from_zarr_v2_json_str(
                filters.as_deref(),
                compressor.as_deref(),
            )?),
            Self::ZarrV2JsonValue {
                filters,
                compressor,
            } => {
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
                Ok(codec_pipeline_from_zarr_v2_specs(
                    &filters,
                    compressor.as_ref(),
                ))
            }
        }
    }
}
