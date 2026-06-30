use numpy::IntoPyArray;
use pyo3::prelude::*;

use crate::access::ScheduledAccessConfig;
use crate::databank::{
    Bf16Bits, DType, DataBank as RustDataBank, DatasetId, F16Bits, MissingGenePolicy,
};

pub(crate) fn resolve_dtype(
    bank: &RustDataBank,
    id: DatasetId,
    dtype: Option<Bound<'_, PyAny>>,
) -> PyResult<DType> {
    match dtype {
        Some(obj) => super::dtype::extract_dtype(&obj),
        None => Ok(bank.dataset_dtype(id)?),
    }
}

pub(crate) fn access_cells_dispatch(
    py: Python<'_>,
    bank: &RustDataBank,
    id: DatasetId,
    cells: &[usize],
    dtype: DType,
    config: ScheduledAccessConfig,
) -> PyResult<PyObject> {
    macro_rules! arm {
        ($ty:ty) => {{
            let out: Vec<$ty> = bank.access_cells_owned_with_config(id, cells, config)?;
            out.into_pyarray(py).into_any().unbind()
        }};
    }
    let arr = match dtype {
        DType::U8 => arm!(u8),
        DType::I8 => arm!(i8),
        DType::U16 => arm!(u16),
        DType::I16 => arm!(i16),
        DType::U32 => arm!(u32),
        DType::I32 => arm!(i32),
        DType::U64 => arm!(u64),
        DType::I64 => arm!(i64),
        DType::F32 => arm!(f32),
        DType::F64 => arm!(f64),
        DType::F16 => {
            let out: Vec<F16Bits> = bank.access_cells_owned_with_config(id, cells, config)?;
            f16_bits_to_numpy(py, out)?
        }
        DType::BF16 => {
            let out: Vec<Bf16Bits> = bank.access_cells_owned_with_config(id, cells, config)?;
            bf16_bits_to_u16(out).into_pyarray(py).into_any().unbind()
        }
    };
    Ok(arr)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn access_cells_by_gene_names_dispatch(
    py: Python<'_>,
    bank: &RustDataBank,
    id: DatasetId,
    cells: &[usize],
    gene_names: &[String],
    missing: MissingGenePolicy,
    dtype: DType,
    config: ScheduledAccessConfig,
) -> PyResult<PyObject> {
    macro_rules! arm {
        ($ty:ty) => {{
            let out: Vec<$ty> = bank.access_cells_owned_by_gene_names_with_config(
                id, cells, gene_names, missing, config,
            )?;
            out.into_pyarray(py).into_any().unbind()
        }};
    }
    let arr = match dtype {
        DType::U8 => arm!(u8),
        DType::I8 => arm!(i8),
        DType::U16 => arm!(u16),
        DType::I16 => arm!(i16),
        DType::U32 => arm!(u32),
        DType::I32 => arm!(i32),
        DType::U64 => arm!(u64),
        DType::I64 => arm!(i64),
        DType::F32 => arm!(f32),
        DType::F64 => arm!(f64),
        DType::F16 => {
            let out: Vec<F16Bits> = bank.access_cells_owned_by_gene_names_with_config(
                id, cells, gene_names, missing, config,
            )?;
            f16_bits_to_numpy(py, out)?
        }
        DType::BF16 => {
            let out: Vec<Bf16Bits> = bank.access_cells_owned_by_gene_names_with_config(
                id, cells, gene_names, missing, config,
            )?;
            bf16_bits_to_u16(out).into_pyarray(py).into_any().unbind()
        }
    };
    Ok(arr)
}

pub(crate) fn buffer_to_numpy(
    py: Python<'_>,
    buffer: super::prefetch::PrefetchedBufferAny,
) -> PyResult<PyObject> {
    let arr = match buffer {
        super::prefetch::PrefetchedBufferAny::U8(v) => v.into_pyarray(py).into_any().unbind(),
        super::prefetch::PrefetchedBufferAny::I8(v) => v.into_pyarray(py).into_any().unbind(),
        super::prefetch::PrefetchedBufferAny::U16(v) => v.into_pyarray(py).into_any().unbind(),
        super::prefetch::PrefetchedBufferAny::I16(v) => v.into_pyarray(py).into_any().unbind(),
        super::prefetch::PrefetchedBufferAny::U32(v) => v.into_pyarray(py).into_any().unbind(),
        super::prefetch::PrefetchedBufferAny::I32(v) => v.into_pyarray(py).into_any().unbind(),
        super::prefetch::PrefetchedBufferAny::U64(v) => v.into_pyarray(py).into_any().unbind(),
        super::prefetch::PrefetchedBufferAny::I64(v) => v.into_pyarray(py).into_any().unbind(),
        super::prefetch::PrefetchedBufferAny::F32(v) => v.into_pyarray(py).into_any().unbind(),
        super::prefetch::PrefetchedBufferAny::F64(v) => v.into_pyarray(py).into_any().unbind(),
        super::prefetch::PrefetchedBufferAny::F16(v) => f16_bits_to_numpy(py, v)?,
        super::prefetch::PrefetchedBufferAny::BF16(v) => {
            bf16_bits_to_u16(v).into_pyarray(py).into_any().unbind()
        }
    };
    Ok(arr)
}

pub(crate) fn f16_bits_to_numpy(py: Python<'_>, bits: Vec<F16Bits>) -> PyResult<PyObject> {
    let arr = f16_bits_to_u16(bits).into_pyarray(py).into_any();
    Ok(arr.call_method1("view", ("float16",))?.unbind())
}

fn f16_bits_to_u16(bits: Vec<F16Bits>) -> Vec<u16> {
    debug_assert_eq!(std::mem::size_of::<F16Bits>(), std::mem::size_of::<u16>());
    let mut bits = bits;
    let ptr = bits.as_mut_ptr() as *mut u16;
    let len = bits.len();
    let cap = bits.capacity();
    std::mem::forget(bits);
    // SAFETY: F16Bits is #[repr(transparent)] over u16.
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

pub(crate) fn bf16_bits_to_u16(bits: Vec<Bf16Bits>) -> Vec<u16> {
    debug_assert_eq!(std::mem::size_of::<Bf16Bits>(), std::mem::size_of::<u16>());
    let mut bits = bits;
    let ptr = bits.as_mut_ptr() as *mut u16;
    let len = bits.len();
    let cap = bits.capacity();
    std::mem::forget(bits);
    // SAFETY: Bf16Bits is #[repr(transparent)] over u16.
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}
