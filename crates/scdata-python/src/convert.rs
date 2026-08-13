//! NumPy <-> typed buffer helpers for the low-level core API.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use sc_compress::DType;

use crate::error::invalid_argument;

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

fn decode_primitive<T, const N: usize>(
    bytes: Vec<u8>,
    from_le: fn([u8; N]) -> T,
) -> PyResult<Vec<T>> {
    if !bytes.len().is_multiple_of(N) {
        return Err(invalid_argument(format!(
            "decoded byte length {} is not a multiple of element size {N}",
            bytes.len()
        )));
    }
    let mut values = Vec::with_capacity(bytes.len() / N);
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
    let array = PyArray1::from_vec(py, values);
    array.call_method1("reshape", (rows, cols))
}

pub(crate) fn dense_bytes_to_array<'py>(
    py: Python<'py>,
    bytes: Vec<u8>,
    dtype: DType,
    shape: [u64; 2],
) -> PyResult<Bound<'py, PyAny>> {
    let [rows, cols] = shape_usize(shape)?;
    match dtype {
        DType::U16 => vec_to_array2(py, decode_primitive(bytes, u16::from_le_bytes)?, rows, cols),
        DType::U32 => vec_to_array2(py, decode_primitive(bytes, u32::from_le_bytes)?, rows, cols),
        DType::I16 => vec_to_array2(py, decode_primitive(bytes, i16::from_le_bytes)?, rows, cols),
        DType::I32 => vec_to_array2(py, decode_primitive(bytes, i32::from_le_bytes)?, rows, cols),
        DType::F32 => vec_to_array2(py, decode_primitive(bytes, f32::from_le_bytes)?, rows, cols),
        DType::F64 => vec_to_array2(py, decode_primitive(bytes, f64::from_le_bytes)?, rows, cols),
        DType::U64 => Err(invalid_argument(
            "u64 is not a supported dense matrix value dtype",
        )),
    }
}

pub(crate) fn csr_index_bytes_to_array<'py>(
    py: Python<'py>,
    bytes: Vec<u8>,
    dtype: DType,
) -> PyResult<Bound<'py, PyAny>> {
    match dtype {
        DType::U16 => {
            Ok(PyArray1::from_vec(py, decode_primitive(bytes, u16::from_le_bytes)?).into_any())
        }
        DType::U32 => {
            Ok(PyArray1::from_vec(py, decode_primitive(bytes, u32::from_le_bytes)?).into_any())
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
            Ok(PyArray1::from_vec(py, decode_primitive(bytes, u16::from_le_bytes)?).into_any())
        }
        DType::U32 => {
            Ok(PyArray1::from_vec(py, decode_primitive(bytes, u32::from_le_bytes)?).into_any())
        }
        DType::I16 => {
            Ok(PyArray1::from_vec(py, decode_primitive(bytes, i16::from_le_bytes)?).into_any())
        }
        DType::I32 => {
            Ok(PyArray1::from_vec(py, decode_primitive(bytes, i32::from_le_bytes)?).into_any())
        }
        DType::F32 => {
            Ok(PyArray1::from_vec(py, decode_primitive(bytes, f32::from_le_bytes)?).into_any())
        }
        DType::F64 => {
            Ok(PyArray1::from_vec(py, decode_primitive(bytes, f64::from_le_bytes)?).into_any())
        }
        DType::U64 => Err(invalid_argument("u64 is not a supported CSR value dtype")),
    }
}

pub(crate) fn u64_slice_to_array<'py>(
    py: Python<'py>,
    values: &[u64],
) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyArray1::from_slice(py, values).into_any())
}

pub(crate) enum DenseValues<'a> {
    U16(&'a [u16]),
    U32(&'a [u32]),
    I16(&'a [i16]),
    I32(&'a [i32]),
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
    if let Ok(array) = values.extract::<PyReadonlyArray2<'_, i16>>() {
        let slice = contiguous_2d(&array, "values")?;
        return write(DenseValues::I16(slice), shape_u64_from_array(&array)?);
    }
    if let Ok(array) = values.extract::<PyReadonlyArray2<'_, i32>>() {
        let slice = contiguous_2d(&array, "values")?;
        return write(DenseValues::I32(slice), shape_u64_from_array(&array)?);
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
        "values must be a C-contiguous 2D NumPy array with dtype u16, u32, i16, i32, f32, or f64",
    ))
}

pub(crate) enum CsrData<'a> {
    U16(&'a [u16]),
    U32(&'a [u32]),
    I16(&'a [i16]),
    I32(&'a [i32]),
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
    if let Ok(array) = data.extract::<PyReadonlyArray1<'_, i16>>() {
        return write(CsrData::I16(contiguous_1d(&array, "data")?));
    }
    if let Ok(array) = data.extract::<PyReadonlyArray1<'_, i32>>() {
        return write(CsrData::I32(contiguous_1d(&array, "data")?));
    }
    if let Ok(array) = data.extract::<PyReadonlyArray1<'_, f32>>() {
        return write(CsrData::F32(contiguous_1d(&array, "data")?));
    }
    if let Ok(array) = data.extract::<PyReadonlyArray1<'_, f64>>() {
        return write(CsrData::F64(contiguous_1d(&array, "data")?));
    }
    Err(invalid_argument(
        "data must be a C-contiguous 1D NumPy array with dtype u16, u32, i16, i32, f32, or f64",
    ))
}

pub(crate) fn copy_u64_1d(array: &Bound<'_, PyAny>, context: &str) -> PyResult<Vec<u64>> {
    let values = array
        .extract::<PyReadonlyArray1<'_, u64>>()
        .map_err(|_| invalid_argument(format!("{context} must be a 1D NumPy uint64 array")))?;
    Ok(contiguous_1d(&values, context)?.to_vec())
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
