use numpy::{Element, PyReadonlyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyByteArray, PyBytes, PyString};

use crate::databank::{DType, GeneNameView};

pub(crate) fn extract_dtype(dtype: &Bound<'_, PyAny>) -> PyResult<DType> {
    let value = match dtype.extract::<String>() {
        Ok(value) => value,
        Err(_) => dtype.getattr("value")?.extract::<String>()?,
    };
    match value.as_str() {
        "u8" => Ok(DType::U8),
        "i8" => Ok(DType::I8),
        "u16" => Ok(DType::U16),
        "i16" => Ok(DType::I16),
        "u32" => Ok(DType::U32),
        "i32" => Ok(DType::I32),
        "u64" => Ok(DType::U64),
        "i64" => Ok(DType::I64),
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        "f32" => Ok(DType::F32),
        "f64" => Ok(DType::F64),
        other => Err(PyValueError::new_err(format!("unknown dtype {other:?}"))),
    }
}

pub(crate) fn dtype_to_py(py: Python<'_>, dtype: DType) -> PyResult<PyObject> {
    let code = dtype_code(dtype);
    let data_mod = py.import("scdata.data")?;
    let dtype_cls = data_mod.getattr("DType")?;
    Ok(dtype_cls.call1((code,))?.unbind())
}

pub(crate) fn dtype_code(dtype: DType) -> &'static str {
    match dtype {
        DType::U8 => "u8",
        DType::I8 => "i8",
        DType::U16 => "u16",
        DType::I16 => "i16",
        DType::U32 => "u32",
        DType::I32 => "i32",
        DType::U64 => "u64",
        DType::I64 => "i64",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::F32 => "f32",
        DType::F64 => "f64",
    }
}

pub(crate) fn intp_array_to_usize_vec(
    cells: &PyReadonlyArray1<'_, isize>,
    context: &str,
) -> PyResult<Vec<usize>> {
    let mut out = Vec::new();
    extend_signed_array_to_usize_vec_by(cells, context, |value| value as i128, &mut out)?;
    Ok(out)
}

pub(crate) fn index_any_to_usize_vec(
    obj: &Bound<'_, PyAny>,
    context: &str,
) -> PyResult<Vec<usize>> {
    let mut out = Vec::new();
    extend_index_any(obj, context, &mut out)?;
    Ok(out)
}

pub(crate) fn extend_index_any(
    obj: &Bound<'_, PyAny>,
    context: &str,
    out: &mut Vec<usize>,
) -> PyResult<usize> {
    if let Some(len) = extend_integer_array_any(obj, context, out)? {
        return Ok(len);
    }
    extend_iterable_to_usize_vec(obj, context, out)
}

pub(crate) fn extend_cells_any(obj: &Bound<'_, PyAny>, out: &mut Vec<usize>) -> PyResult<usize> {
    if let Some(len) = extend_integer_array_any(obj, "cells", out)? {
        return Ok(len);
    }
    if let Ok(cells) = obj.getattr("cells") {
        return extend_cells_any(&cells, out);
    }
    extend_iterable_to_usize_vec(obj, "cells", out)
}

fn extend_integer_array_any(
    obj: &Bound<'_, PyAny>,
    context: &str,
    out: &mut Vec<usize>,
) -> PyResult<Option<usize>> {
    macro_rules! try_signed_array {
        ($ty:ty) => {
            if let Ok(array) = obj.extract::<PyReadonlyArray1<'_, $ty>>() {
                return Ok(Some(extend_signed_array_to_usize_vec_by(
                    &array,
                    context,
                    |value| value as i128,
                    out,
                )?));
            }
        };
    }
    macro_rules! try_unsigned_array {
        ($ty:ty) => {
            if let Ok(array) = obj.extract::<PyReadonlyArray1<'_, $ty>>() {
                return Ok(Some(extend_unsigned_array_to_usize_vec_by(
                    &array,
                    context,
                    |value| value as u128,
                    out,
                )?));
            }
        };
    }

    try_signed_array!(isize);
    try_signed_array!(i64);
    try_signed_array!(i32);
    try_signed_array!(i16);
    try_signed_array!(i8);
    try_unsigned_array!(usize);
    try_unsigned_array!(u64);
    try_unsigned_array!(u32);
    try_unsigned_array!(u16);
    try_unsigned_array!(u8);

    Ok(None)
}

fn extend_signed_array_to_usize_vec_by<T, F>(
    cells: &PyReadonlyArray1<'_, T>,
    context: &str,
    cast: F,
    out: &mut Vec<usize>,
) -> PyResult<usize>
where
    T: Element + Copy,
    F: Fn(T) -> i128,
{
    let view = cells.as_array();
    let start = out.len();
    out.reserve(view.len());
    for (i, &cell) in view.iter().enumerate() {
        let value = cast(cell);
        if value < 0 {
            return Err(PyValueError::new_err(format!(
                "{context}[{i}] must be non-negative, got {value}"
            )));
        }
        if value > isize::MAX as i128 {
            return Err(PyValueError::new_err(format!(
                "{context}[{i}] must fit in numpy intp, got {value}"
            )));
        }
        out.push(value as usize);
    }
    Ok(out.len() - start)
}

fn extend_unsigned_array_to_usize_vec_by<T, F>(
    cells: &PyReadonlyArray1<'_, T>,
    context: &str,
    cast: F,
    out: &mut Vec<usize>,
) -> PyResult<usize>
where
    T: Element + Copy,
    F: Fn(T) -> u128,
{
    let view = cells.as_array();
    let start = out.len();
    out.reserve(view.len());
    for (i, &cell) in view.iter().enumerate() {
        let value = cast(cell);
        if value > isize::MAX as u128 {
            return Err(PyValueError::new_err(format!(
                "{context}[{i}] must fit in numpy intp, got {value}"
            )));
        }
        out.push(value as usize);
    }
    Ok(out.len() - start)
}

fn extend_iterable_to_usize_vec(
    obj: &Bound<'_, PyAny>,
    context: &str,
    out: &mut Vec<usize>,
) -> PyResult<usize> {
    if obj.is_instance_of::<PyString>()
        || obj.is_instance_of::<PyBytes>()
        || obj.is_instance_of::<PyByteArray>()
    {
        return Err(PyTypeError::new_err(format!(
            "{context} must be a 1D iterable of integers"
        )));
    }
    let py = obj.py();
    let mut operator = None;
    let start = out.len();
    for (i, item) in obj.try_iter()?.enumerate() {
        let item = item?;
        if item.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(format!(
                "{context}[{i}] must be an integer, got bool"
            )));
        }
        if let Ok(value) = item.extract::<usize>() {
            if value > isize::MAX as usize {
                return Err(PyValueError::new_err(format!(
                    "{context}[{i}] must fit in numpy intp, got {value}"
                )));
            }
            out.push(value);
            continue;
        }
        let op = match operator.as_ref() {
            Some(op) => op,
            None => {
                operator = Some(py.import("operator")?);
                operator.as_ref().expect("operator module just imported")
            }
        };
        let value: i128 = op.call_method1("index", (&item,))?.extract()?;
        if value < 0 {
            return Err(PyValueError::new_err(format!(
                "{context}[{i}] must be non-negative, got {value}"
            )));
        }
        if value > isize::MAX as i128 {
            return Err(PyValueError::new_err(format!(
                "{context}[{i}] must fit in numpy intp, got {value}"
            )));
        }
        out.push(value as usize);
    }
    Ok(out.len() - start)
}

pub(crate) fn gene_view_to_string(view: GeneNameView) -> String {
    if view.is_empty() {
        return String::new();
    }
    // SAFETY: `view` points into an Arc<str> owned by a dataset borrowed
    // through DataBank for the duration of the call.
    let bytes = unsafe { std::slice::from_raw_parts(view.ptr, view.len) };
    String::from_utf8_lossy(bytes).into_owned()
}
