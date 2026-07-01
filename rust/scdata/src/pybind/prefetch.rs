use numpy::IntoPyArray;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyTuple};
use pyo3::PyRefMut;

use crate::databank::{
    Bf16Bits, DType, DataBank as RustDataBank, DataBankResult, DatasetId, F16Bits,
    MissingGenePolicy, MultiBatchCells, PrefetchCells, ScheduledPrefetchConfig,
};

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPrefetchPlan>()?;
    m.add_class::<PyPrefetchCells>()?;
    Ok(())
}

pub(crate) struct PyMultiBatchSource {
    cells: std::vec::IntoIter<usize>,
    parts: Vec<FlatPlanPart>,
    batch_part_offsets: Vec<usize>,
    next_batch: usize,
    next_part: usize,
}

impl PyMultiBatchSource {
    pub(crate) fn from_plan(mut plan: PyRefMut<'_, PyPrefetchPlan>) -> PyResult<Self> {
        plan.take_source()
    }
}

#[pyclass(name = "_PrefetchPlan", module = "scdata._scdata")]
pub(crate) struct PyPrefetchPlan {
    plan: Option<FlatPrefetchPlan>,
}

struct FlatPrefetchPlan {
    cells: Vec<usize>,
    parts: Vec<FlatPlanPart>,
    batch_part_offsets: Vec<usize>,
}

#[derive(Clone, Copy)]
struct FlatPlanPart {
    dataset_idx: usize,
    len: usize,
}

impl PyPrefetchPlan {
    fn new(plan: FlatPrefetchPlan) -> Self {
        Self { plan: Some(plan) }
    }

    fn take_source(&mut self) -> PyResult<PyMultiBatchSource> {
        let Some(plan) = self.plan.take() else {
            return Err(PyValueError::new_err(
                "prefetch plan has already been consumed",
            ));
        };
        Ok(PyMultiBatchSource {
            cells: plan.cells.into_iter(),
            parts: plan.parts,
            batch_part_offsets: plan.batch_part_offsets,
            next_batch: 0,
            next_part: 0,
        })
    }
}

#[pymethods]
impl PyPrefetchPlan {
    #[staticmethod]
    fn single(batches: Bound<'_, PyAny>) -> PyResult<Self> {
        let mut cells = Vec::new();
        let mut parts = Vec::new();
        let mut batch_part_offsets = vec![0];
        for item in batches.try_iter()? {
            let item = item?;
            let len = super::dtype::extend_cells_any(&item, &mut cells)?;
            parts.push(FlatPlanPart {
                dataset_idx: 0,
                len,
            });
            batch_part_offsets.push(parts.len());
        }
        Ok(Self::new(FlatPrefetchPlan {
            cells,
            parts,
            batch_part_offsets,
        }))
    }

    #[staticmethod]
    fn multi(batches: Bound<'_, PyAny>) -> PyResult<Self> {
        let mut cells = Vec::new();
        let mut parts = Vec::new();
        let mut batch_part_offsets = vec![0];
        for batch in batches.try_iter()? {
            let batch = batch?;
            for part in batch.try_iter()? {
                let part = part?;
                let (dataset_idx, len) = append_multi_batch_part(&part, &mut cells)?;
                parts.push(FlatPlanPart { dataset_idx, len });
            }
            batch_part_offsets.push(parts.len());
        }
        Ok(Self::new(FlatPrefetchPlan {
            cells,
            parts,
            batch_part_offsets,
        }))
    }

    #[staticmethod]
    fn indexed(
        dataset_index: Bound<'_, PyAny>,
        cell_index: Bound<'_, PyAny>,
        batch_size: usize,
    ) -> PyResult<Self> {
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be positive"));
        }
        let dataset_index = super::dtype::index_any_to_usize_vec(&dataset_index, "dataset_index")?;
        let cells = super::dtype::index_any_to_usize_vec(&cell_index, "cell_index")?;
        if dataset_index.len() != cells.len() {
            return Err(PyValueError::new_err(format!(
                "dataset_index and cell_index must have the same length, got {} and {}",
                dataset_index.len(),
                cells.len()
            )));
        }
        Ok(Self::new(flat_plan_from_indexed(
            dataset_index,
            cells,
            batch_size,
        )))
    }
}

fn flat_plan_from_indexed(
    dataset_index: Vec<usize>,
    cells: Vec<usize>,
    batch_size: usize,
) -> FlatPrefetchPlan {
    let num_batches = cells.len().div_ceil(batch_size);
    let mut parts = Vec::new();
    let mut batch_part_offsets = Vec::with_capacity(num_batches + 1);
    batch_part_offsets.push(0);

    for batch_start in (0..cells.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(cells.len());
        let mut run_start = batch_start;
        while run_start < batch_end {
            let dataset_idx = dataset_index[run_start];
            let mut run_end = run_start + 1;
            while run_end < batch_end && dataset_index[run_end] == dataset_idx {
                run_end += 1;
            }
            parts.push(FlatPlanPart {
                dataset_idx,
                len: run_end - run_start,
            });
            run_start = run_end;
        }
        batch_part_offsets.push(parts.len());
    }

    FlatPrefetchPlan {
        cells,
        parts,
        batch_part_offsets,
    }
}

impl Iterator for PyMultiBatchSource {
    type Item = MultiBatchCells;

    fn next(&mut self) -> Option<Self::Item> {
        let end = *self.batch_part_offsets.get(self.next_batch + 1)?;
        let part_slice = &self.parts[self.next_part..end];
        let total_cells = part_slice.iter().map(|part| part.len).sum();
        let mut cells = Vec::with_capacity(total_cells);
        let mut parts = Vec::with_capacity(part_slice.len());
        for idx in self.next_part..end {
            let part = self.parts[idx];
            let start = cells.len();
            cells.extend(self.cells.by_ref().take(part.len));
            debug_assert_eq!(cells.len() - start, part.len);
            parts.push((part.dataset_idx, part.len));
        }
        self.next_batch += 1;
        self.next_part = end;
        Some(MultiBatchCells::from_flat_parts(cells, parts))
    }
}

fn append_multi_batch_part(
    part: &Bound<'_, PyAny>,
    cells: &mut Vec<usize>,
) -> PyResult<(usize, usize)> {
    let (dataset_idx_obj, cells_obj) = extract_multi_batch_pair(part)?;
    let dataset_idx = extract_dataset_idx(&dataset_idx_obj)?;
    let len = super::dtype::extend_cells_any(&cells_obj, cells)?;
    Ok((dataset_idx, len))
}

fn extract_multi_batch_pair<'py>(
    part: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    if let Ok(tuple) = part.downcast::<PyTuple>() {
        if tuple.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "multi prefetch batch parts must be (dataset_idx, cells), got tuple length {}",
                tuple.len()
            )));
        }
        return Ok((tuple.get_item(0)?, tuple.get_item(1)?));
    }

    let mut iter = part.try_iter()?;
    let Some(dataset_idx) = iter.next() else {
        return Err(PyValueError::new_err(
            "multi prefetch batch parts must be 2-item iterables: (dataset_idx, cells)",
        ));
    };
    let Some(cells) = iter.next() else {
        return Err(PyValueError::new_err(
            "multi prefetch batch parts must be 2-item iterables: (dataset_idx, cells)",
        ));
    };
    if iter.next().is_some() {
        return Err(PyValueError::new_err(
            "multi prefetch batch parts must be 2-item iterables: (dataset_idx, cells)",
        ));
    }
    Ok((dataset_idx?, cells?))
}

fn extract_dataset_idx(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "dataset_idx must be an integer, got bool",
        ));
    }
    let operator = obj.py().import("operator")?;
    let value: i128 = operator.call_method1("index", (obj,))?.extract()?;
    if value < 0 {
        return Err(PyValueError::new_err(format!(
            "dataset_idx must be non-negative, got {value}"
        )));
    }
    if value > usize::MAX as i128 {
        return Err(PyValueError::new_err(format!(
            "dataset_idx must fit in usize, got {value}"
        )));
    }
    Ok(value as usize)
}

enum PrefetchDispatch {
    U8(PrefetchCells<u8>),
    I8(PrefetchCells<i8>),
    U16(PrefetchCells<u16>),
    I16(PrefetchCells<i16>),
    U32(PrefetchCells<u32>),
    I32(PrefetchCells<i32>),
    U64(PrefetchCells<u64>),
    I64(PrefetchCells<i64>),
    F32(PrefetchCells<f32>),
    F64(PrefetchCells<f64>),
    F16(PrefetchCells<F16Bits>),
    BF16(PrefetchCells<Bf16Bits>),
}

impl Iterator for PrefetchDispatch {
    type Item = DataBankResult<PrefetchedBatchAny>;

    fn next(&mut self) -> Option<Self::Item> {
        macro_rules! next_batch {
            ($iter:expr, $variant:ident) => {
                match $iter.next() {
                    Some(Ok(batch)) => Some(Ok(PrefetchedBatchAny {
                        cells: batch.cells,
                        buffer: PrefetchedBufferAny::$variant(batch.buffer),
                        num_genes: batch.num_genes,
                    })),
                    Some(Err(err)) => Some(Err(err)),
                    None => None,
                }
            };
        }

        match self {
            Self::U8(iter) => next_batch!(iter, U8),
            Self::I8(iter) => next_batch!(iter, I8),
            Self::U16(iter) => next_batch!(iter, U16),
            Self::I16(iter) => next_batch!(iter, I16),
            Self::U32(iter) => next_batch!(iter, U32),
            Self::I32(iter) => next_batch!(iter, I32),
            Self::U64(iter) => next_batch!(iter, U64),
            Self::I64(iter) => next_batch!(iter, I64),
            Self::F32(iter) => next_batch!(iter, F32),
            Self::F64(iter) => next_batch!(iter, F64),
            Self::F16(iter) => next_batch!(iter, F16),
            Self::BF16(iter) => next_batch!(iter, BF16),
        }
    }
}

pub(crate) enum PrefetchedBufferAny {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    F16(Vec<F16Bits>),
    BF16(Vec<Bf16Bits>),
}

struct PrefetchedBatchAny {
    cells: Vec<usize>,
    buffer: PrefetchedBufferAny,
    num_genes: usize,
}

#[pyclass(name = "_PrefetchCells", module = "scdata._scdata")]
pub(crate) struct PyPrefetchCells {
    inner: Option<PrefetchDispatch>,
    gene_names: Vec<String>,
    prefetch_step: usize,
}

impl PyPrefetchCells {
    fn new(inner: PrefetchDispatch, requested_gene_names: Option<&[String]>) -> Self {
        let gene_names = requested_gene_names
            .map(<[String]>::to_vec)
            .unwrap_or_else(|| inner.gene_names());
        let prefetch_step = inner.prefetch_step();
        Self {
            inner: Some(inner),
            gene_names,
            prefetch_step,
        }
    }
}

#[pymethods]
impl PyPrefetchCells {
    #[getter]
    fn prefetch_step(&self) -> usize {
        self.prefetch_step
    }

    #[getter]
    fn gene_names(&self) -> Vec<String> {
        self.gene_names.clone()
    }

    fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let Some(iter) = self.inner.as_mut() else {
            return Ok(None);
        };
        match py.allow_threads(|| iter.next()) {
            None => {
                self.inner = None;
                Ok(None)
            }
            Some(Ok(batch)) => {
                let cells = usize_vec_to_intp_numpy(py, batch.cells)?;
                let buffer = super::dispatch::buffer_to_numpy(py, batch.buffer)?;
                let tuple = (cells, buffer, batch.num_genes).into_pyobject(py)?;
                Ok(Some(tuple.into_any().unbind()))
            }
            Some(Err(err)) => Err(err.into()),
        }
    }
}

fn usize_vec_to_intp_numpy(py: Python<'_>, cells: Vec<usize>) -> PyResult<PyObject> {
    if let Some((i, cell)) = cells
        .iter()
        .copied()
        .enumerate()
        .find(|(_, cell)| *cell > isize::MAX as usize)
    {
        return Err(PyValueError::new_err(format!(
            "cells[{i}] must fit in numpy intp, got {cell}"
        )));
    }
    debug_assert_eq!(std::mem::size_of::<usize>(), std::mem::size_of::<isize>());
    debug_assert_eq!(std::mem::align_of::<usize>(), std::mem::align_of::<isize>());
    let mut cells = cells;
    let ptr = cells.as_mut_ptr() as *mut isize;
    let len = cells.len();
    let cap = cells.capacity();
    std::mem::forget(cells);
    // SAFETY: `usize` and `isize` have identical layout on the target. Values
    // are checked above to fit in `isize`, so the reinterpreted vector contains
    // valid non-negative numpy intp indices.
    let cells = unsafe { Vec::from_raw_parts(ptr, len, cap) };
    Ok(cells.into_pyarray(py).into_any().unbind())
}

impl PrefetchDispatch {
    fn prefetch_step(&self) -> usize {
        match self {
            Self::U8(iter) => iter.prefetch_step(),
            Self::I8(iter) => iter.prefetch_step(),
            Self::U16(iter) => iter.prefetch_step(),
            Self::I16(iter) => iter.prefetch_step(),
            Self::U32(iter) => iter.prefetch_step(),
            Self::I32(iter) => iter.prefetch_step(),
            Self::U64(iter) => iter.prefetch_step(),
            Self::I64(iter) => iter.prefetch_step(),
            Self::F32(iter) => iter.prefetch_step(),
            Self::F64(iter) => iter.prefetch_step(),
            Self::F16(iter) => iter.prefetch_step(),
            Self::BF16(iter) => iter.prefetch_step(),
        }
    }

    fn gene_names(&self) -> Vec<String> {
        match self {
            Self::U8(iter) => prefetch_gene_names(iter),
            Self::I8(iter) => prefetch_gene_names(iter),
            Self::U16(iter) => prefetch_gene_names(iter),
            Self::I16(iter) => prefetch_gene_names(iter),
            Self::U32(iter) => prefetch_gene_names(iter),
            Self::I32(iter) => prefetch_gene_names(iter),
            Self::U64(iter) => prefetch_gene_names(iter),
            Self::I64(iter) => prefetch_gene_names(iter),
            Self::F32(iter) => prefetch_gene_names(iter),
            Self::F64(iter) => prefetch_gene_names(iter),
            Self::F16(iter) => prefetch_gene_names(iter),
            Self::BF16(iter) => prefetch_gene_names(iter),
        }
    }
}

fn prefetch_gene_names<T>(iter: &PrefetchCells<T>) -> Vec<String>
where
    T: crate::databank::DataValue,
{
    iter.gene_names()
        .iter()
        .map(|view| super::dtype::gene_view_to_string(*view))
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prefetch_cells_multi_dispatch(
    bank: &RustDataBank,
    ids: &[DatasetId],
    batch_source: PyMultiBatchSource,
    gene_names: Option<&[String]>,
    missing: MissingGenePolicy,
    config: ScheduledPrefetchConfig,
    dtype: DType,
) -> PyResult<PyPrefetchCells> {
    macro_rules! arm {
        ($variant:ident, $ty:ty) => {{
            let iter = if let Some(gene_names) = gene_names {
                bank.prefetch_cells_scheduled_multi_by_gene_names::<$ty, _, String>(
                    ids,
                    batch_source,
                    gene_names,
                    missing,
                    config,
                )?
            } else {
                bank.prefetch_cells_scheduled_multi::<$ty, _>(ids, batch_source, config)?
            };
            PrefetchDispatch::$variant(iter)
        }};
    }

    let dispatch = match dtype {
        DType::U8 => arm!(U8, u8),
        DType::I8 => arm!(I8, i8),
        DType::U16 => arm!(U16, u16),
        DType::I16 => arm!(I16, i16),
        DType::U32 => arm!(U32, u32),
        DType::I32 => arm!(I32, i32),
        DType::U64 => arm!(U64, u64),
        DType::I64 => arm!(I64, i64),
        DType::F32 => arm!(F32, f32),
        DType::F64 => arm!(F64, f64),
        DType::F16 => arm!(F16, F16Bits),
        DType::BF16 => arm!(BF16, Bf16Bits),
    };
    Ok(PyPrefetchCells::new(dispatch, gene_names))
}
