use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
    let slice = cells.as_slice().map_err(|_| {
        PyValueError::new_err(format!("{context} must be a contiguous 1D np.intp array"))
    })?;
    let mut out = Vec::with_capacity(slice.len());
    for (i, &cell) in slice.iter().enumerate() {
        if cell < 0 {
            return Err(PyValueError::new_err(format!(
                "{context}[{i}] must be non-negative, got {cell}"
            )));
        }
        out.push(cell as usize);
    }
    Ok(out)
}

pub(crate) fn extract_cells_any(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(array) = obj.extract::<PyReadonlyArray1<'_, isize>>() {
        return intp_array_to_usize_vec(&array, "cells");
    }
    obj.extract::<Vec<usize>>()
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
