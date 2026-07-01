use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::PyRefMut;

use crate::databank::{
    DataBank as RustDataBank, DatasetId, Dense1DSpec, Dense2DSpec, MissingGenePolicy, SparseCsrSpec,
};

use super::config::{PyDataBankConfig, PyScheduledAccessConfig, PyScheduledPrefetchConfig};
use super::ids::{PyDatasetId, PyMissingGenePolicy};
use super::prefetch::{
    prefetch_cells_multi_dispatch, PyMultiBatchSource, PyPrefetchCells, PyPrefetchPlan,
};

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataBank>()?;
    Ok(())
}

#[pyclass(name = "_DataBank", module = "scdata._scdata")]
struct PyDataBank {
    inner: RustDataBank,
}

#[pymethods]
impl PyDataBank {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyDataBankConfig>) -> PyResult<Self> {
        let config = config.map(|config| config.inner).unwrap_or_default();
        Ok(Self {
            inner: RustDataBank::new(config)?,
        })
    }

    fn config(&self) -> PyDataBankConfig {
        PyDataBankConfig {
            inner: self.inner.config().clone(),
        }
    }

    fn profile_snapshot(&self, py: Python<'_>) -> PyResult<PyObject> {
        super::profile::snapshot_to_py(py, self.inner.profile_snapshot())
    }

    fn profile_snapshot_and_reset(&self, py: Python<'_>) -> PyResult<PyObject> {
        super::profile::snapshot_to_py(py, self.inner.profile_snapshot_and_reset())
    }

    fn reset_profile(&self) {
        self.inner.reset_profile();
    }

    #[pyo3(signature = (dataset, store_path))]
    fn register_dense(
        &mut self,
        py: Python<'_>,
        dataset: Bound<'_, PyAny>,
        store_path: String,
    ) -> PyResult<PyDatasetId> {
        let gene_names: Vec<String> = dataset.getattr("gene_names")?.extract()?;
        let data = super::arrays::build_array_spec(py, &dataset.getattr("data")?, &store_path)?;
        let id = match data.shape.len() {
            1 => self
                .inner
                .register_dense_1d(Dense1DSpec { gene_names, data })?,
            2 => self
                .inner
                .register_dense_2d(Dense2DSpec { gene_names, data })?,
            rank => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "dense data must be 1D or 2D, got {rank}D"
                )))
            }
        };
        Ok(id.into())
    }

    #[pyo3(signature = (dataset, store_path))]
    fn register_sparse_csr(
        &mut self,
        py: Python<'_>,
        dataset: Bound<'_, PyAny>,
        store_path: String,
    ) -> PyResult<PyDatasetId> {
        let spec = SparseCsrSpec {
            gene_names: dataset.getattr("gene_names")?.extract()?,
            indptr: super::arrays::extract_u64_vec(&dataset.getattr("indptr")?, "indptr")?,
            indices: super::arrays::build_array_spec(
                py,
                &dataset.getattr("indices")?,
                &store_path,
            )?,
            data: super::arrays::build_array_spec(py, &dataset.getattr("data")?, &store_path)?,
            index_dtype: super::dtype::extract_dtype(&dataset.getattr("index_dtype")?)?,
            num_cells: dataset.getattr("num_cells")?.extract()?,
            num_genes: dataset.getattr("num_genes")?.extract()?,
        };
        Ok(self.inner.register_sparse_csr(spec)?.into())
    }

    fn unregister(&mut self, id: PyDatasetId) -> PyResult<()> {
        self.inner.unregister(id.into())?;
        Ok(())
    }

    fn dataset_genes(&self, id: PyDatasetId) -> PyResult<Vec<String>> {
        let views = self.inner.dataset_genes(id.into())?;
        Ok(views
            .iter()
            .map(|view| super::dtype::gene_view_to_string(*view))
            .collect())
    }

    fn dataset_num_cells(&self, id: PyDatasetId) -> PyResult<usize> {
        Ok(self.inner.dataset_num_cells(id.into())?)
    }

    fn dataset_num_genes(&self, id: PyDatasetId) -> PyResult<usize> {
        Ok(self.inner.dataset_num_genes(id.into())?)
    }

    fn dataset_dtype(&self, py: Python<'_>, id: PyDatasetId) -> PyResult<PyObject> {
        super::dtype::dtype_to_py(py, self.inner.dataset_dtype(id.into())?)
    }

    #[pyo3(signature = (id, cells, dtype=None, config=None))]
    fn access_cells(
        &self,
        py: Python<'_>,
        id: PyDatasetId,
        cells: Vec<usize>,
        dtype: Option<Bound<'_, PyAny>>,
        config: Option<PyScheduledAccessConfig>,
    ) -> PyResult<PyObject> {
        let id: DatasetId = id.into();
        let dtype = super::dispatch::resolve_dtype(&self.inner, id, dtype)?;
        let config = config.map(|config| config.inner).unwrap_or_default();
        super::dispatch::access_cells_dispatch(py, &self.inner, id, &cells, dtype, config)
    }

    #[pyo3(signature = (id, cells, dtype=None, config=None))]
    fn access_cells_array(
        &self,
        py: Python<'_>,
        id: PyDatasetId,
        cells: PyReadonlyArray1<'_, isize>,
        dtype: Option<Bound<'_, PyAny>>,
        config: Option<PyScheduledAccessConfig>,
    ) -> PyResult<PyObject> {
        let cells = super::dtype::intp_array_to_usize_vec(&cells, "cells")?;
        let id: DatasetId = id.into();
        let dtype = super::dispatch::resolve_dtype(&self.inner, id, dtype)?;
        let config = config.map(|config| config.inner).unwrap_or_default();
        super::dispatch::access_cells_dispatch(py, &self.inner, id, &cells, dtype, config)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (id, cells, gene_names, missing=None, dtype=None, config=None))]
    fn access_cells_by_gene_names(
        &self,
        py: Python<'_>,
        id: PyDatasetId,
        cells: Vec<usize>,
        gene_names: Vec<String>,
        missing: Option<PyMissingGenePolicy>,
        dtype: Option<Bound<'_, PyAny>>,
        config: Option<PyScheduledAccessConfig>,
    ) -> PyResult<PyObject> {
        let id: DatasetId = id.into();
        let dtype = super::dispatch::resolve_dtype(&self.inner, id, dtype)?;
        let missing = missing
            .map(|missing| missing.inner)
            .unwrap_or(MissingGenePolicy::Zero);
        let config = config.map(|config| config.inner).unwrap_or_default();
        super::dispatch::access_cells_by_gene_names_dispatch(
            py,
            &self.inner,
            id,
            &cells,
            &gene_names,
            missing,
            dtype,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (id, cells, gene_names, missing=None, dtype=None, config=None))]
    fn access_cells_by_gene_names_array(
        &self,
        py: Python<'_>,
        id: PyDatasetId,
        cells: PyReadonlyArray1<'_, isize>,
        gene_names: Vec<String>,
        missing: Option<PyMissingGenePolicy>,
        dtype: Option<Bound<'_, PyAny>>,
        config: Option<PyScheduledAccessConfig>,
    ) -> PyResult<PyObject> {
        let cells = super::dtype::intp_array_to_usize_vec(&cells, "cells")?;
        self.access_cells_by_gene_names(py, id, cells, gene_names, missing, dtype, config)
    }

    fn prefetch_cells_cache(&self, id: PyDatasetId, cells: Vec<usize>) -> PyResult<()> {
        self.inner.prefetch_cells(id.into(), &cells)?;
        Ok(())
    }

    fn prefetch_cells_cache_array(
        &self,
        id: PyDatasetId,
        cells: PyReadonlyArray1<'_, isize>,
    ) -> PyResult<()> {
        let cells = super::dtype::intp_array_to_usize_vec(&cells, "cells")?;
        self.prefetch_cells_cache(id, cells)
    }

    #[pyo3(signature = (dataset_ids, plan, output_dtype, gene_names=None, missing=None, config=None))]
    fn prefetch_cells(
        &self,
        dataset_ids: Vec<PyDatasetId>,
        plan: PyRefMut<'_, PyPrefetchPlan>,
        output_dtype: Bound<'_, PyAny>,
        gene_names: Option<Vec<String>>,
        missing: Option<PyMissingGenePolicy>,
        config: Option<PyScheduledPrefetchConfig>,
    ) -> PyResult<PyPrefetchCells> {
        let ids: Vec<DatasetId> = dataset_ids.into_iter().map(Into::into).collect();
        let dtype = super::dtype::extract_dtype(&output_dtype)?;
        let missing = missing
            .map(|missing| missing.inner)
            .unwrap_or(MissingGenePolicy::Zero);
        prefetch_cells_multi_dispatch(
            &self.inner,
            &ids,
            PyMultiBatchSource::from_plan(plan)?,
            gene_names.as_deref(),
            missing,
            config.map(|config| config.inner).unwrap_or_default(),
            dtype,
        )
    }

    #[pyo3(signature = (id, plan, config=None))]
    fn prefetch_cells_raw(
        &self,
        id: PyDatasetId,
        plan: PyRefMut<'_, PyPrefetchPlan>,
        config: Option<PyScheduledPrefetchConfig>,
    ) -> PyResult<PyPrefetchCells> {
        let id: DatasetId = id.into();
        let dtype = self.inner.dataset_dtype(id)?;
        prefetch_cells_multi_dispatch(
            &self.inner,
            &[id],
            PyMultiBatchSource::from_plan(plan)?,
            None,
            MissingGenePolicy::Zero,
            config.map(|config| config.inner).unwrap_or_default(),
            dtype,
        )
    }

    #[pyo3(signature = (id, plan, gene_names, missing=None, config=None))]
    fn prefetch_cells_by_gene_names_raw(
        &self,
        id: PyDatasetId,
        plan: PyRefMut<'_, PyPrefetchPlan>,
        gene_names: Vec<String>,
        missing: Option<PyMissingGenePolicy>,
        config: Option<PyScheduledPrefetchConfig>,
    ) -> PyResult<PyPrefetchCells> {
        let id: DatasetId = id.into();
        let dtype = self.inner.dataset_dtype(id)?;
        let missing = missing
            .map(|missing| missing.inner)
            .unwrap_or(MissingGenePolicy::Zero);
        prefetch_cells_multi_dispatch(
            &self.inner,
            &[id],
            PyMultiBatchSource::from_plan(plan)?,
            Some(&gene_names),
            missing,
            config.map(|config| config.inner).unwrap_or_default(),
            dtype,
        )
    }

    fn __repr__(&self) -> &'static str {
        "DataBank(scdata-rust)"
    }
}
