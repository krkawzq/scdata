use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::codecs::SharedCodec;
use crate::databank::DType;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_decode_index_payload, m)?)?;
    m.add_function(wrap_pyfunction!(_decode_index_chunks, m)?)?;
    Ok(())
}

#[pyfunction]
fn _decode_index_payload(
    py: Python<'_>,
    payload: Bound<'_, PyBytes>,
    offsets: Bound<'_, PyAny>,
    lengths: Bound<'_, PyAny>,
    dtype: Bound<'_, PyAny>,
    codec: Bound<'_, PyAny>,
    count: usize,
) -> PyResult<PyObject> {
    let offsets = super::arrays::extract_u64_vec(&offsets, "offsets")?;
    let lengths = super::arrays::extract_u64_vec(&lengths, "lengths")?;
    if offsets.len() != lengths.len() {
        return Err(PyValueError::new_err(format!(
            "offsets length {} != lengths length {}",
            offsets.len(),
            lengths.len()
        )));
    }
    let dtype = super::dtype::extract_dtype(&dtype)?;
    let codec = super::arrays::build_shared_codec(py, &codec)?;
    let payload = payload.as_bytes();
    let mut out = Vec::new();
    for (offset, len) in offsets.into_iter().zip(lengths) {
        let start = super::arrays::u64_to_usize(offset, "offsets")?;
        let len = super::arrays::u64_to_usize(len, "lengths")?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| PyValueError::new_err("index chunk byte range overflows usize"))?;
        let raw = payload.get(start..end).ok_or_else(|| {
            PyValueError::new_err(format!(
                "index chunk range [{start}, {end}) exceeds payload size {}",
                payload.len()
            ))
        })?;
        decode_index_chunk_into(raw, dtype, codec.as_ref(), &mut out)?;
    }
    finalize_index_output(py, out, count, false)
}

#[pyfunction]
fn _decode_index_chunks(
    py: Python<'_>,
    chunks: Bound<'_, PyAny>,
    dtype: Bound<'_, PyAny>,
    codec: Bound<'_, PyAny>,
    count: usize,
) -> PyResult<PyObject> {
    let dtype = super::dtype::extract_dtype(&dtype)?;
    let codec = super::arrays::build_shared_codec(py, &codec)?;
    let mut out = Vec::new();
    for item in chunks.try_iter()? {
        let item = item?;
        let raw = item.downcast::<PyBytes>()?;
        decode_index_chunk_into(raw.as_bytes(), dtype, codec.as_ref(), &mut out)?;
    }
    finalize_index_output(py, out, count, true)
}

fn decode_index_chunk_into(
    raw: &[u8],
    dtype: DType,
    codec: Option<&SharedCodec>,
    out: &mut Vec<u64>,
) -> PyResult<()> {
    let decoded;
    let bytes = if let Some(codec) = codec {
        decoded = codec
            .decode(raw, None)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        decoded.as_slice()
    } else {
        raw
    };
    decode_index_bytes_into(bytes, dtype, out)
}

fn decode_index_bytes_into(bytes: &[u8], dtype: DType, out: &mut Vec<u64>) -> PyResult<()> {
    let item_size = match dtype {
        DType::U8 | DType::I8 => 1,
        DType::U16 | DType::I16 => 2,
        DType::U32 | DType::I32 => 4,
        DType::U64 | DType::I64 => 8,
        other => {
            return Err(PyValueError::new_err(format!(
                "index array dtype must be integer, got {other:?}"
            )))
        }
    };
    if bytes.len() % item_size != 0 {
        return Err(PyValueError::new_err(format!(
            "decoded index chunk has {} bytes, not divisible by dtype item size {item_size}",
            bytes.len()
        )));
    }
    out.reserve(bytes.len() / item_size);
    match dtype {
        DType::U8 => out.extend(bytes.iter().map(|&value| u64::from(value))),
        DType::I8 => {
            for &byte in bytes {
                push_signed_index(i64::from(i8::from_le_bytes([byte])), out)?;
            }
        }
        DType::U16 => {
            for chunk in bytes.chunks_exact(2) {
                out.push(u64::from(u16::from_le_bytes([chunk[0], chunk[1]])));
            }
        }
        DType::I16 => {
            for chunk in bytes.chunks_exact(2) {
                push_signed_index(i64::from(i16::from_le_bytes([chunk[0], chunk[1]])), out)?;
            }
        }
        DType::U32 => {
            for chunk in bytes.chunks_exact(4) {
                out.push(u64::from(u32::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                ])));
            }
        }
        DType::I32 => {
            for chunk in bytes.chunks_exact(4) {
                push_signed_index(
                    i64::from(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
                    out,
                )?;
            }
        }
        DType::U64 => {
            for chunk in bytes.chunks_exact(8) {
                out.push(u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]));
            }
        }
        DType::I64 => {
            for chunk in bytes.chunks_exact(8) {
                push_signed_index(
                    i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]),
                    out,
                )?;
            }
        }
        _ => unreachable!("non-integer dtype rejected above"),
    }
    Ok(())
}

fn push_signed_index(value: i64, out: &mut Vec<u64>) -> PyResult<()> {
    let value = u64::try_from(value)
        .map_err(|_| PyValueError::new_err(format!("negative index value {value}")))?;
    out.push(value);
    Ok(())
}

fn finalize_index_output(
    py: Python<'_>,
    mut out: Vec<u64>,
    count: usize,
    allow_short: bool,
) -> PyResult<PyObject> {
    if allow_short {
        if out.len() > count {
            out.truncate(count);
        } else if out.len() < count {
            out.resize(count, 0);
        }
    }
    if out.len() != count {
        return Err(PyValueError::new_err(format!(
            "decoded index length {} != expected {count}",
            out.len()
        )));
    }
    Ok(out.into_pyarray(py).into_any().unbind())
}
