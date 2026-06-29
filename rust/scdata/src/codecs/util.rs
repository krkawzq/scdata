use serde_json::Value;

use super::spec::BloscShuffle;
use super::{CodecError, CodecResult};

pub(crate) fn verify_size(codec: &str, actual: usize, expected: Option<usize>) -> CodecResult<()> {
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

pub(crate) fn output_too_small(codec: &str, required: usize, capacity: usize) -> CodecError {
    CodecError::OutputTooSmall {
        codec: codec.to_string(),
        required,
        capacity,
    }
}

pub(crate) fn decode_error(codec: &str, message: impl Into<String>) -> CodecError {
    CodecError::Decode {
        codec: codec.to_string(),
        message: message.into(),
    }
}

pub(crate) fn reserve_decode_buffer(
    codec: &str,
    output: &mut Vec<u8>,
    additional: usize,
) -> CodecResult<()> {
    output
        .try_reserve_exact(additional)
        .map_err(|err| decode_error(codec, format!("failed to reserve decode buffer: {err}")))
}

pub(crate) fn vec_with_decode_capacity(codec: &str, capacity: usize) -> CodecResult<Vec<u8>> {
    let mut output = Vec::new();
    reserve_decode_buffer(codec, &mut output, capacity)?;
    Ok(output)
}

pub(crate) fn optional_string(
    value: Option<&Value>,
    codec: &str,
    field: &str,
) -> CodecResult<Option<String>> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    value
        .as_str()
        .map(|value| Some(value.to_string()))
        .ok_or_else(|| CodecError::InvalidCodecConfig {
            codec: codec.to_string(),
            message: format!("field `{field}` must be a string"),
        })
}

pub(crate) fn optional_bool(
    value: Option<&Value>,
    codec: &str,
    field: &str,
) -> CodecResult<Option<bool>> {
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

pub(crate) fn optional_i32(
    value: Option<&Value>,
    codec: &str,
    field: &str,
) -> CodecResult<Option<i32>> {
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

pub(crate) fn optional_u32(
    value: Option<&Value>,
    codec: &str,
    field: &str,
) -> CodecResult<Option<u32>> {
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

pub(crate) fn optional_u8(
    value: Option<&Value>,
    codec: &str,
    field: &str,
) -> CodecResult<Option<u8>> {
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

pub(crate) fn optional_usize(
    value: Option<&Value>,
    codec: &str,
    field: &str,
) -> CodecResult<Option<usize>> {
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

pub(crate) fn optional_blosc_shuffle(value: Option<&Value>) -> CodecResult<Option<BloscShuffle>> {
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
