//! NumPy <-> typed buffer helpers for the low-level core API.

use std::collections::TryReserveError;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use sc_compress::DType;

use crate::error::{from_compress, from_rust, invalid_argument};

const CONVERT_WITHOUT_GIL_THRESHOLD: usize = 64 * 1024;

/// Plain numeric values that may be initialized from their native byte pattern.
///
/// # Safety
///
/// Implementors must accept every possible bit pattern, have no drop glue, and
/// use the same in-memory width as the encoded primitive.
unsafe trait DecodedPrimitive: numpy::Element + Copy + Send {}

unsafe impl DecodedPrimitive for u16 {}
unsafe impl DecodedPrimitive for u32 {}
unsafe impl DecodedPrimitive for u64 {}
unsafe impl DecodedPrimitive for i16 {}
unsafe impl DecodedPrimitive for i32 {}
unsafe impl DecodedPrimitive for i64 {}
unsafe impl DecodedPrimitive for f32 {}
unsafe impl DecodedPrimitive for f64 {}

pub(crate) fn dtype_name(dtype: DType) -> &'static str {
    dtype.as_str()
}

pub(crate) fn shape_usize(shape: [u64; 2]) -> PyResult<[usize; 2]> {
    Ok([
        usize::try_from(shape[0])
            .map_err(|_| invalid_argument("matrix row count exceeds usize"))?,
        usize::try_from(shape[1])
            .map_err(|_| invalid_argument("matrix column count exceeds usize"))?,
    ])
}

fn decode_primitive<T: DecodedPrimitive, const N: usize>(
    py: Python<'_>,
    bytes: Vec<u8>,
    from_le: fn([u8; N]) -> T,
) -> PyResult<Vec<T>> {
    if N == 0 || N != std::mem::size_of::<T>() {
        return Err(from_rust(sc_load::Error::Invariant(format!(
            "primitive decoder width {N} does not match element size {}",
            std::mem::size_of::<T>()
        ))));
    }
    if !bytes.len().is_multiple_of(N) {
        return Err(invalid_argument(format!(
            "decoded byte length {} is not a multiple of element size {N}",
            bytes.len()
        )));
    }
    let byte_len = bytes.len();
    let result = if byte_len >= CONVERT_WITHOUT_GIL_THRESHOLD {
        py.allow_threads(move || decode_primitive_inner(bytes, from_le))
    } else {
        decode_primitive_inner(bytes, from_le)
    };
    result.map_err(|error| from_compress(error.into()))
}

fn decode_primitive_inner<T: DecodedPrimitive, const N: usize>(
    bytes: Vec<u8>,
    from_le: fn([u8; N]) -> T,
) -> Result<Vec<T>, TryReserveError> {
    let count = bytes.len() / N;
    let mut values: Vec<T> = Vec::new();
    values.try_reserve_exact(count)?;
    #[cfg(target_endian = "little")]
    {
        let _ = from_le;
        // SAFETY: `DecodedPrimitive` is restricted to plain numeric types, the
        // width was checked by the caller, and the destination has `count` slots.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                values.as_mut_ptr().cast::<u8>(),
                bytes.len(),
            );
            values.set_len(count);
        }
    }
    #[cfg(target_endian = "big")]
    for chunk in bytes.chunks_exact(N) {
        let mut item = [0u8; N];
        item.copy_from_slice(chunk);
        values.push(from_le(item));
    }
    Ok(values)
}

fn vec_to_array2<'py, T: numpy::Element + Copy>(
    py: Python<'py>,
    values: Vec<T>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid_argument("dense reshape size overflow"))?;
    if values.len() != expected {
        return Err(invalid_argument(format!(
            "decoded value count {} does not match shape [{rows}, {cols}]",
            values.len()
        )));
    }
    PyArray1::from_vec(py, values)
        .reshape([rows, cols])
        .map(|array| array.into_any())
}

pub(crate) fn dense_bytes_to_array<'py>(
    py: Python<'py>,
    bytes: Vec<u8>,
    dtype: DType,
    shape: [u64; 2],
) -> PyResult<Bound<'py, PyAny>> {
    let [rows, cols] = shape_usize(shape)?;
    match dtype {
        DType::U16 => vec_to_array2(
            py,
            decode_primitive(py, bytes, u16::from_le_bytes)?,
            rows,
            cols,
        ),
        DType::U32 => vec_to_array2(
            py,
            decode_primitive(py, bytes, u32::from_le_bytes)?,
            rows,
            cols,
        ),
        DType::U64 => vec_to_array2(
            py,
            decode_primitive(py, bytes, u64::from_le_bytes)?,
            rows,
            cols,
        ),
        DType::I16 => vec_to_array2(
            py,
            decode_primitive(py, bytes, i16::from_le_bytes)?,
            rows,
            cols,
        ),
        DType::I32 => vec_to_array2(
            py,
            decode_primitive(py, bytes, i32::from_le_bytes)?,
            rows,
            cols,
        ),
        DType::I64 => vec_to_array2(
            py,
            decode_primitive(py, bytes, i64::from_le_bytes)?,
            rows,
            cols,
        ),
        DType::F32 => vec_to_array2(
            py,
            decode_primitive(py, bytes, f32::from_le_bytes)?,
            rows,
            cols,
        ),
        DType::F64 => vec_to_array2(
            py,
            decode_primitive(py, bytes, f64::from_le_bytes)?,
            rows,
            cols,
        ),
    }
}

pub(crate) fn csr_index_bytes_to_array<'py>(
    py: Python<'py>,
    bytes: Vec<u8>,
    dtype: DType,
) -> PyResult<Bound<'py, PyAny>> {
    match dtype {
        DType::U16 => {
            Ok(PyArray1::from_vec(py, decode_primitive(py, bytes, u16::from_le_bytes)?).into_any())
        }
        DType::U32 => {
            Ok(PyArray1::from_vec(py, decode_primitive(py, bytes, u32::from_le_bytes)?).into_any())
        }
        other => Err(invalid_argument(format!(
            "unsupported CSR index dtype `{other}`"
        ))),
    }
}

pub(crate) fn csr_data_bytes_to_array<'py>(
    py: Python<'py>,
    bytes: Vec<u8>,
    dtype: DType,
) -> PyResult<Bound<'py, PyAny>> {
    match dtype {
        DType::U16 => {
            Ok(PyArray1::from_vec(py, decode_primitive(py, bytes, u16::from_le_bytes)?).into_any())
        }
        DType::U32 => {
            Ok(PyArray1::from_vec(py, decode_primitive(py, bytes, u32::from_le_bytes)?).into_any())
        }
        DType::U64 => {
            Ok(PyArray1::from_vec(py, decode_primitive(py, bytes, u64::from_le_bytes)?).into_any())
        }
        DType::I16 => {
            Ok(PyArray1::from_vec(py, decode_primitive(py, bytes, i16::from_le_bytes)?).into_any())
        }
        DType::I32 => {
            Ok(PyArray1::from_vec(py, decode_primitive(py, bytes, i32::from_le_bytes)?).into_any())
        }
        DType::I64 => {
            Ok(PyArray1::from_vec(py, decode_primitive(py, bytes, i64::from_le_bytes)?).into_any())
        }
        DType::F32 => {
            Ok(PyArray1::from_vec(py, decode_primitive(py, bytes, f32::from_le_bytes)?).into_any())
        }
        DType::F64 => {
            Ok(PyArray1::from_vec(py, decode_primitive(py, bytes, f64::from_le_bytes)?).into_any())
        }
    }
}

pub(crate) fn u64_slice_to_array<'py>(
    py: Python<'py>,
    values: &[u64],
) -> PyResult<Bound<'py, PyAny>> {
    let bytes = std::mem::size_of_val(values);
    let copied = if bytes >= CONVERT_WITHOUT_GIL_THRESHOLD {
        py.allow_threads(|| copy_slice(values))
    } else {
        copy_slice(values)
    }
    .map_err(|error| from_compress(error.into()))?;
    Ok(u64_vec_to_array(py, copied))
}

pub(crate) fn u64_vec_to_array<'py>(py: Python<'py>, values: Vec<u64>) -> Bound<'py, PyAny> {
    PyArray1::from_vec(py, values).into_any()
}

fn copy_slice<T: Copy>(values: &[T]) -> Result<Vec<T>, TryReserveError> {
    let mut copied = Vec::new();
    copied.try_reserve_exact(values.len())?;
    copied.extend_from_slice(values);
    Ok(copied)
}

pub(crate) enum DenseValues<'a> {
    U16(&'a [u16]),
    U32(&'a [u32]),
    U64(&'a [u64]),
    I16(&'a [i16]),
    I32(&'a [i32]),
    I64(&'a [i64]),
    F32(&'a [f32]),
    F64(&'a [f64]),
}

fn shape_u64_from_array<T: numpy::Element>(array: &PyReadonlyArray2<'_, T>) -> PyResult<[u64; 2]> {
    Ok([
        u64::try_from(array.shape()[0])
            .map_err(|_| invalid_argument("matrix row count exceeds u64"))?,
        u64::try_from(array.shape()[1])
            .map_err(|_| invalid_argument("matrix column count exceeds u64"))?,
    ])
}

pub(crate) fn dispatch_dense<R>(
    values: &Bound<'_, PyAny>,
    write: impl FnOnce(DenseValues<'_>, [u64; 2]) -> PyResult<R>,
) -> PyResult<R> {
    if let Ok(array) = values.extract::<PyReadonlyArray2<'_, u16>>() {
        let slice = contiguous_2d(&array, "values")?;
        return write(DenseValues::U16(slice), shape_u64_from_array(&array)?);
    }
    if let Ok(array) = values.extract::<PyReadonlyArray2<'_, u32>>() {
        let slice = contiguous_2d(&array, "values")?;
        return write(DenseValues::U32(slice), shape_u64_from_array(&array)?);
    }
    if let Ok(array) = values.extract::<PyReadonlyArray2<'_, u64>>() {
        let slice = contiguous_2d(&array, "values")?;
        return write(DenseValues::U64(slice), shape_u64_from_array(&array)?);
    }
    if let Ok(array) = values.extract::<PyReadonlyArray2<'_, i16>>() {
        let slice = contiguous_2d(&array, "values")?;
        return write(DenseValues::I16(slice), shape_u64_from_array(&array)?);
    }
    if let Ok(array) = values.extract::<PyReadonlyArray2<'_, i32>>() {
        let slice = contiguous_2d(&array, "values")?;
        return write(DenseValues::I32(slice), shape_u64_from_array(&array)?);
    }
    if let Ok(array) = values.extract::<PyReadonlyArray2<'_, i64>>() {
        let slice = contiguous_2d(&array, "values")?;
        return write(DenseValues::I64(slice), shape_u64_from_array(&array)?);
    }
    if let Ok(array) = values.extract::<PyReadonlyArray2<'_, f32>>() {
        let slice = contiguous_2d(&array, "values")?;
        return write(DenseValues::F32(slice), shape_u64_from_array(&array)?);
    }
    if let Ok(array) = values.extract::<PyReadonlyArray2<'_, f64>>() {
        let slice = contiguous_2d(&array, "values")?;
        return write(DenseValues::F64(slice), shape_u64_from_array(&array)?);
    }
    Err(invalid_argument(
        "values must be a C-contiguous 2D NumPy array with dtype u16, u32, u64, i16, i32, i64, f32, or f64",
    ))
}

pub(crate) enum CsrData<'a> {
    U16(&'a [u16]),
    U32(&'a [u32]),
    U64(&'a [u64]),
    I16(&'a [i16]),
    I32(&'a [i32]),
    I64(&'a [i64]),
    F32(&'a [f32]),
    F64(&'a [f64]),
}

pub(crate) fn dispatch_csr_data(
    data: &Bound<'_, PyAny>,
    write: impl FnOnce(CsrData<'_>) -> PyResult<()>,
) -> PyResult<()> {
    if let Ok(array) = data.extract::<PyReadonlyArray1<'_, u16>>() {
        return write(CsrData::U16(contiguous_1d(&array, "data")?));
    }
    if let Ok(array) = data.extract::<PyReadonlyArray1<'_, u32>>() {
        return write(CsrData::U32(contiguous_1d(&array, "data")?));
    }
    if let Ok(array) = data.extract::<PyReadonlyArray1<'_, u64>>() {
        return write(CsrData::U64(contiguous_1d(&array, "data")?));
    }
    if let Ok(array) = data.extract::<PyReadonlyArray1<'_, i16>>() {
        return write(CsrData::I16(contiguous_1d(&array, "data")?));
    }
    if let Ok(array) = data.extract::<PyReadonlyArray1<'_, i32>>() {
        return write(CsrData::I32(contiguous_1d(&array, "data")?));
    }
    if let Ok(array) = data.extract::<PyReadonlyArray1<'_, i64>>() {
        return write(CsrData::I64(contiguous_1d(&array, "data")?));
    }
    if let Ok(array) = data.extract::<PyReadonlyArray1<'_, f32>>() {
        return write(CsrData::F32(contiguous_1d(&array, "data")?));
    }
    if let Ok(array) = data.extract::<PyReadonlyArray1<'_, f64>>() {
        return write(CsrData::F64(contiguous_1d(&array, "data")?));
    }
    Err(invalid_argument(
        "data must be a C-contiguous 1D NumPy array with dtype u16, u32, u64, i16, i32, i64, f32, or f64",
    ))
}

pub(crate) fn copy_u64_1d(array: &Bound<'_, PyAny>, context: &str) -> PyResult<Vec<u64>> {
    let values = array
        .extract::<PyReadonlyArray1<'_, u64>>()
        .map_err(|_| invalid_argument(format!("{context} must be a 1D NumPy uint64 array")))?;
    copy_slice(contiguous_1d(&values, context)?).map_err(|error| from_compress(error.into()))
}

fn contiguous_1d<'a, T: Copy + numpy::Element>(
    array: &'a PyReadonlyArray1<'a, T>,
    context: &str,
) -> PyResult<&'a [T]> {
    array
        .as_slice()
        .map_err(|_| invalid_argument(format!("{context} must be a C-contiguous 1D NumPy array")))
}

fn contiguous_2d<'a, T: Copy + numpy::Element>(
    array: &'a PyReadonlyArray2<'a, T>,
    context: &str,
) -> PyResult<&'a [T]> {
    array
        .as_slice()
        .map_err(|_| invalid_argument(format!("{context} must be a C-contiguous 2D NumPy array")))
}

#[cfg(test)]
mod tests {
    use super::{copy_slice, decode_primitive_inner};

    #[test]
    fn primitive_bytes_decode_in_little_endian_order() {
        let expected = [0u32, 1, u32::MAX, 0x1234_5678];
        let bytes = expected
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();

        let decoded = decode_primitive_inner::<u32, 4>(bytes, u32::from_le_bytes).unwrap();

        assert_eq!(decoded, expected);
    }

    #[test]
    fn fallible_slice_copy_preserves_values() {
        let expected = [0u64, 3, u64::MAX];
        assert_eq!(copy_slice(&expected).unwrap(), expected);
    }
}
