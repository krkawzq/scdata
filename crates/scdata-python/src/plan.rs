//! Immutable plan compilation from normalized primitive inputs.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sc_load::{compile, FeatureMap, Plan, PlanSpec, RowRef, Source, SourceId};

use crate::config::{plan_config_from_dict, session_config_from_dict};
use crate::dataset::PyDataset;
use crate::error::{invalid_input as invalid_argument, ResultExt};
use crate::output::output_spec_from_dict;
use crate::session::PySession;
#[cfg(all(target_os = "linux", target_has_atomic = "64"))]
use crate::shared::PySharedServer;
use crate::stats::plan_stats_to_dict;
#[cfg(all(target_os = "linux", target_has_atomic = "64"))]
use sc_load::SharedConfig;

#[pyclass(name = "_Plan", module = "scdata._core", frozen)]
pub(crate) struct PyPlan {
    inner: Plan,
}

#[pyfunction]
pub(crate) fn plan_meta<'py>(py: Python<'py>, plan: &PyPlan) -> PyResult<Bound<'py, PyDict>> {
    let inner = &plan.inner;
    let values = PyDict::new(py);
    values.set_item("batch_size", inner.batch_size())?;
    values.set_item("batch_count", inner.batch_count())?;
    values.set_item("prefetch_step", inner.prefetch_step())?;
    values.set_item("cache_capacity_bytes", inner.stats().cache_capacity_bytes)?;
    values.set_item("n_cols", inner.output_spec().n_cols())?;
    values.set_item("dtype", inner.output_spec().dtype().as_str())?;
    values.set_item("row_stride_bytes", inner.row_stride_bytes())?;
    values.set_item("is_empty", inner.is_empty())?;
    Ok(values)
}

#[pyfunction]
pub(crate) fn plan_stats<'py>(py: Python<'py>, plan: &PyPlan) -> PyResult<Bound<'py, PyDict>> {
    plan_stats_to_dict(py, plan.inner.stats())
}

#[pyfunction]
pub(crate) fn plan_open(
    py: Python<'_>,
    plan: &PyPlan,
    config: &Bound<'_, PyDict>,
) -> PyResult<PySession> {
    let plan = plan.inner.clone();
    let config = session_config_from_dict(config)?;
    let session = py.allow_threads(move || plan.open(config)).map_sc()?;
    Ok(PySession::new(session))
}

#[cfg(all(target_os = "linux", target_has_atomic = "64"))]
#[pyfunction]
#[pyo3(signature = (plan, config, world_size, max_control_bytes))]
pub(crate) fn plan_open_shared(
    py: Python<'_>,
    plan: &PyPlan,
    config: &Bound<'_, PyDict>,
    world_size: usize,
    max_control_bytes: usize,
) -> PyResult<PySharedServer> {
    let plan = plan.inner.clone();
    let config = session_config_from_dict(config)?;
    let shared = SharedConfig::new(world_size)
        .map_sc()?
        .with_max_control_bytes(max_control_bytes)
        .map_sc()?;
    let server = py
        .allow_threads(move || plan.open_shared(config, shared))
        .map_sc()?;
    PySharedServer::new(server).map_sc()
}

#[pyfunction]
#[pyo3(signature = (
    datasets,
    source_ids,
    feature_maps,
    row_source_ids,
    row_indices,
    output,
    batch_size,
    prefetch_step,
    config,
))]
#[expect(
    clippy::too_many_arguments,
    reason = "the private boundary accepts normalized primitive compile inputs"
)]
pub(crate) fn plan_compile(
    py: Python<'_>,
    datasets: Vec<Py<PyDataset>>,
    source_ids: Vec<u32>,
    feature_maps: &Bound<'_, PyList>,
    row_source_ids: Option<PyReadonlyArray1<'_, u32>>,
    row_indices: PyReadonlyArray1<'_, u64>,
    output: &Bound<'_, PyDict>,
    batch_size: usize,
    prefetch_step: usize,
    config: &Bound<'_, PyDict>,
) -> PyResult<PyPlan> {
    if datasets.len() != source_ids.len() || datasets.len() != feature_maps.len() {
        return Err(invalid_argument(format!(
            "datasets, source_ids, and feature_maps must have equal lengths (got {}, {}, and {})",
            datasets.len(),
            source_ids.len(),
            feature_maps.len()
        )));
    }

    let row_source_ids = row_source_ids
        .as_ref()
        .map(|values| {
            values.as_slice().map_err(|_| {
                invalid_argument("row_source_ids must be a C-contiguous 1D NumPy uint32 array")
            })
        })
        .transpose()?;
    let row_indices = row_indices.as_slice().map_err(|_| {
        invalid_argument("row_indices must be a C-contiguous 1D NumPy uint64 array")
    })?;
    if let Some(source_id_count) = row_source_ids
        .map(<[u32]>::len)
        .filter(|&count| count != row_indices.len())
    {
        return Err(invalid_argument(format!(
            "row_source_ids has length {source_id_count}, but row_indices has length {}",
            row_indices.len()
        )));
    }
    let implicit_source = match row_source_ids {
        Some(_) => None,
        None => match source_ids.as_slice() {
            [source_id] => Some(*source_id),
            _ => {
                return Err(invalid_argument(
                    "implicit row source ids require exactly one registered source",
                ))
            }
        },
    };

    let mut rust_sources = Vec::new();
    rust_sources
        .try_reserve_exact(datasets.len())
        .map_err(sc_load::Error::from)
        .map_sc()?;
    for (position, (dataset, source_id)) in datasets.into_iter().zip(source_ids).enumerate() {
        let mut source = Source::new(source_id, dataset.borrow(py).inner.clone());
        let feature_map = feature_maps.get_item(position)?;
        if !feature_map.is_none() {
            let feature_map = feature_map.downcast::<PyArray1<i64>>().map_err(|_| {
                invalid_argument("feature maps must be None or C-contiguous 1D NumPy int64 arrays")
            })?;
            let readonly = feature_map
                .try_readonly()
                .map_err(|_| invalid_argument("feature map is already mutably borrowed"))?;
            let values = readonly.as_slice().map_err(|_| {
                invalid_argument("feature maps must be None or C-contiguous 1D NumPy int64 arrays")
            })?;
            source = source.feature_map(FeatureMap::from_signed(values).map_sc()?);
        }
        rust_sources.push(source);
    }

    let mut rows = Vec::new();
    rows.try_reserve_exact(row_indices.len())
        .map_err(sc_load::Error::from)
        .map_sc()?;
    match row_source_ids {
        Some(source_ids) => rows.extend(
            source_ids
                .iter()
                .zip(row_indices)
                .map(|(&source, &row)| RowRef::new(SourceId::new(source), row)),
        ),
        None => {
            let source = SourceId::new(
                implicit_source
                    .ok_or_else(|| invalid_argument("implicit row source id was not resolved"))?,
            );
            rows.extend(row_indices.iter().map(|&row| RowRef::new(source, row)));
        }
    }

    let output = output_spec_from_dict(output)?;
    let config = plan_config_from_dict(config)?;
    let spec = PlanSpec::new(rust_sources, rows, output, batch_size, prefetch_step).config(config);
    let inner = py.allow_threads(move || compile(spec)).map_sc()?;
    Ok(PyPlan { inner })
}
