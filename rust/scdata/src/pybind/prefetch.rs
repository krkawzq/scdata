use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::databank::{
    Bf16Bits, DType, DataBank as RustDataBank, DataBankResult, DatasetId, F16Bits,
    MissingGenePolicy, MultiBatchCells, PrefetchCells, ScheduledPrefetchConfig,
};

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPrefetchCells>()?;
    Ok(())
}

pub(crate) struct PyMultiBatchSource {
    batches: std::vec::IntoIter<MultiBatchCells>,
}

impl PyMultiBatchSource {
    pub(crate) fn single(batches: Bound<'_, PyAny>) -> PyResult<Self> {
        let mut out = Vec::new();
        for item in batches.try_iter()? {
            let item = item?;
            out.push(MultiBatchCells::new(vec![(
                0,
                super::dtype::extract_cells_any(&item)?,
            )]));
        }
        Ok(Self {
            batches: out.into_iter(),
        })
    }

    pub(crate) fn multi(batches: Bound<'_, PyAny>) -> PyResult<Self> {
        let mut out = Vec::new();
        for batch in batches.try_iter()? {
            let batch = batch?;
            let mut parts = Vec::new();
            for part in batch.try_iter()? {
                parts.push(extract_multi_batch_part(&part?)?);
            }
            out.push(MultiBatchCells::new(parts));
        }
        Ok(Self {
            batches: out.into_iter(),
        })
    }
}

impl Iterator for PyMultiBatchSource {
    type Item = MultiBatchCells;

    fn next(&mut self) -> Option<Self::Item> {
        self.batches.next()
    }
}

fn extract_multi_batch_part(part: &Bound<'_, PyAny>) -> PyResult<(usize, Vec<usize>)> {
    let tuple = part.downcast::<PyTuple>()?;
    if tuple.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "multi prefetch batch parts must be (dataset_idx, cells), got tuple length {}",
            tuple.len()
        )));
    }
    let dataset_idx = tuple.get_item(0)?.extract::<usize>()?;
    let cells_obj = tuple.get_item(1)?;
    Ok((dataset_idx, super::dtype::extract_cells_any(&cells_obj)?))
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
    fn new(inner: PrefetchDispatch) -> Self {
        let gene_names = inner.gene_names();
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
                let cells = batch.cells.into_pyarray(py).into_any().unbind();
                let buffer = super::dispatch::buffer_to_numpy(py, batch.buffer)?;
                let tuple = (cells, buffer, batch.num_genes).into_pyobject(py)?;
                Ok(Some(tuple.into_any().unbind()))
            }
            Some(Err(err)) => Err(err.into()),
        }
    }
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
    Ok(PyPrefetchCells::new(dispatch))
}
