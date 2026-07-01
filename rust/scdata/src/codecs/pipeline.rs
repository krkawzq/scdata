use std::sync::Arc;

use serde_json::Value;

use super::buffer::{set_vec_len_for_decode, DecodeBuffer};
use super::runner::DecodeRunner;
use super::spec::{
    codec_specs_from_json_str, codec_specs_from_json_value, sealed, ChunkCodec, CodecCacheKey,
    CodecSpec,
};
use super::util::{output_too_small, reserve_decode_buffer, verify_size};
use super::{CodecError, CodecResult, SharedCodec};

/// Sequential decode pipeline.
///
/// `CodecPipeline::new` expects specs in decode order. For zarr v2 metadata,
/// use [`CodecPipeline::from_zarr_v2_specs`] so the compressor runs first and
/// filters run in reverse metadata order.
#[derive(Debug, Clone)]
pub struct CodecPipeline {
    codecs: Vec<SharedCodec>,
    name: String,
    cache_key: CodecCacheKey,
    is_identity: bool,
}

impl CodecPipeline {
    pub fn new(codecs: Vec<SharedCodec>) -> Self {
        let is_identity = codecs.iter().all(|codec| codec.is_identity());
        let cache_key =
            CodecCacheKey::Pipeline(codecs.iter().map(|codec| codec.cache_key()).collect());
        let name = if codecs.is_empty() {
            "identity".to_string()
        } else {
            codecs
                .iter()
                .map(|codec| codec.name())
                .collect::<Vec<_>>()
                .join("|")
        };
        Self {
            codecs,
            name,
            cache_key,
            is_identity,
        }
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

impl sealed::Sealed for CodecPipeline {}

impl ChunkCodec for CodecPipeline {
    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn cache_key(&self) -> CodecCacheKey {
        self.cache_key.clone()
    }

    fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>> {
        if self.codecs.is_empty() {
            verify_size(self.name(), encoded.len(), expected_size)?;
            return Ok(encoded.to_vec());
        }
        if self.codecs.len() == 1 {
            return DecodeRunner::decode(self.codecs[0].as_ref(), encoded, expected_size);
        }

        let stage_expected = expected_size.map(|_| self.stage_expected_sizes(expected_size));
        let mut current: Option<Vec<u8>> = None;
        let mut spare = Vec::new();
        for (idx, codec) in self.codecs.iter().enumerate() {
            let expected = stage_expected
                .as_ref()
                .and_then(|stage_expected| stage_expected[idx]);
            if codec.is_identity() {
                let actual = current
                    .as_ref()
                    .map_or(encoded.len(), |decoded| decoded.len());
                verify_size(codec.name(), actual, expected)?;
                continue;
            }
            current = Some(match current.take() {
                Some(input) => {
                    let output = std::mem::take(&mut spare);
                    let (decoded, returned_spare) =
                        DecodeRunner::decode_vec_input(codec.as_ref(), input, output, expected)?;
                    spare = returned_spare;
                    spare.clear();
                    decoded
                }
                None => {
                    let output = std::mem::take(&mut spare);
                    DecodeRunner::decode_borrowed_to_vec(
                        codec.as_ref(),
                        encoded,
                        output,
                        expected,
                        None,
                    )?
                }
            });
        }

        match current {
            Some(decoded) => Ok(decoded),
            None => Ok(encoded.to_vec()),
        }
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
        if self.codecs.len() == 1 {
            return DecodeRunner::decode_into(
                self.codecs[0].as_ref(),
                encoded,
                output,
                expected_size,
            );
        }

        let stage_expected = expected_size.map(|_| self.stage_expected_sizes(expected_size));
        let mut current: Option<Vec<u8>> = None;
        let mut spare = Vec::new();
        for (idx, codec) in self.codecs.iter().enumerate() {
            let is_final = idx + 1 == self.codecs.len();
            let expected = stage_expected
                .as_ref()
                .and_then(|stage_expected| stage_expected[idx]);
            if is_final {
                return match current.take() {
                    Some(input) => {
                        if codec.is_identity() {
                            verify_size(codec.name(), input.len(), expected)?;
                            output.write(codec.name(), &input)
                        } else {
                            DecodeRunner::decode_into(codec.as_ref(), &input, output, expected)
                        }
                    }
                    None => {
                        if codec.is_identity() {
                            verify_size(codec.name(), encoded.len(), expected)?;
                            output.write(codec.name(), encoded)
                        } else {
                            DecodeRunner::decode_into(codec.as_ref(), encoded, output, expected)
                        }
                    }
                };
            }

            if codec.is_identity() {
                let actual = current
                    .as_ref()
                    .map_or(encoded.len(), |decoded| decoded.len());
                verify_size(codec.name(), actual, expected)?;
                continue;
            }

            current = Some(match current.take() {
                Some(input) => {
                    let output = std::mem::take(&mut spare);
                    let (decoded, returned_spare) =
                        DecodeRunner::decode_vec_input(codec.as_ref(), input, output, expected)?;
                    spare = returned_spare;
                    spare.clear();
                    decoded
                }
                None => {
                    let output = std::mem::take(&mut spare);
                    DecodeRunner::decode_borrowed_to_vec(
                        codec.as_ref(),
                        encoded,
                        output,
                        expected,
                        None,
                    )?
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
        if self.codecs.is_empty() || self.is_identity() {
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

    fn decode_to_vec(
        &self,
        encoded: &[u8],
        mut output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        let Some(required) = self.decoded_size_hint(encoded, expected_size)? else {
            return self.decode(encoded, expected_size);
        };
        if output.capacity() < required {
            return Err(output_too_small(self.name(), required, output.capacity()));
        }
        set_vec_len_for_decode(&mut output, required);
        let written = self.decode_into(
            encoded,
            DecodeBuffer::new(output.as_mut_slice()),
            expected_size,
        )?;
        output.truncate(written);
        Ok(output)
    }

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
            let additional = required - output.capacity();
            reserve_decode_buffer(self.name(), &mut output, additional)?;
        }
        set_vec_len_for_decode(&mut output, required);
        let written = self.decode_into(
            encoded,
            DecodeBuffer::new(output.as_mut_slice()),
            expected_size,
        )?;
        output.truncate(written);
        Ok(output)
    }

    fn decode_to_capacity_vec(
        &self,
        encoded: &[u8],
        output: Vec<u8>,
        expected_size: Option<usize>,
    ) -> CodecResult<Vec<u8>> {
        self.decode_to_vec(encoded, output, expected_size)
    }

    fn encoded_size_hint(&self, decoded_size: usize) -> Option<usize> {
        let mut size = decoded_size;
        for codec in self.codecs.iter().rev() {
            size = codec.encoded_size_hint(size)?;
        }
        Some(size)
    }

    fn is_identity(&self) -> bool {
        self.is_identity
    }
}
