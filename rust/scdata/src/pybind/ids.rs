use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::databank::{DatasetId, MissingGenePolicy};

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDatasetId>()?;
    m.add_class::<PyMissingGenePolicy>()?;
    Ok(())
}

#[pyclass(name = "_DatasetId", frozen, module = "scdata._scdata")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PyDatasetId {
    slot: u32,
    generation: u32,
}

impl From<DatasetId> for PyDatasetId {
    fn from(id: DatasetId) -> Self {
        Self {
            slot: id.slot,
            generation: id.generation,
        }
    }
}

impl From<PyDatasetId> for DatasetId {
    fn from(id: PyDatasetId) -> Self {
        DatasetId {
            slot: id.slot,
            generation: id.generation,
        }
    }
}

#[pymethods]
impl PyDatasetId {
    #[new]
    fn new(slot: u32, generation: u32) -> Self {
        Self { slot, generation }
    }

    #[getter]
    fn slot(&self) -> u32 {
        self.slot
    }

    #[getter]
    fn generation(&self) -> u32 {
        self.generation
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetId(slot={}, generation={})",
            self.slot, self.generation
        )
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyDatasetId>()
            .is_ok_and(|other| *self == other)
    }

    fn __hash__(&self) -> u64 {
        (u64::from(self.slot)) | (u64::from(self.generation) << 32)
    }
}

#[pyclass(name = "_MissingGenePolicy", frozen, module = "scdata._scdata")]
#[derive(Clone, Copy)]
pub(crate) struct PyMissingGenePolicy {
    pub(crate) inner: MissingGenePolicy,
}

#[pymethods]
impl PyMissingGenePolicy {
    #[classattr]
    #[allow(non_snake_case)]
    fn ZERO() -> Self {
        Self {
            inner: MissingGenePolicy::Zero,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn ERROR() -> Self {
        Self {
            inner: MissingGenePolicy::Error,
        }
    }

    #[new]
    fn new(policy: &str) -> PyResult<Self> {
        match policy.to_ascii_lowercase().as_str() {
            "zero" => Ok(Self {
                inner: MissingGenePolicy::Zero,
            }),
            "error" => Ok(Self {
                inner: MissingGenePolicy::Error,
            }),
            other => Err(PyValueError::new_err(format!(
                "unknown MissingGenePolicy {other:?}; use 'zero' or 'error'"
            ))),
        }
    }

    fn __repr__(&self) -> &'static str {
        match self.inner {
            MissingGenePolicy::Zero => "MissingGenePolicy.ZERO",
            MissingGenePolicy::Error => "MissingGenePolicy.ERROR",
        }
    }
}
