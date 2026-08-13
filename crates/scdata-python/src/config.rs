//! Conversion between normalized Python mappings and Rust runtime configs.

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use sc_load::{IoMode, PlanConfig, ResourceLimits, SessionConfig};

use crate::error::invalid_input as invalid_argument;

#[pyfunction]
pub(crate) fn plan_config_defaults(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    plan_config_to_dict(py, &PlanConfig::default())
}

#[pyfunction]
pub(crate) fn session_config_defaults(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    session_config_to_dict(py, &SessionConfig::default())
}

pub(crate) fn plan_config_from_dict(values: &Bound<'_, PyDict>) -> PyResult<PlanConfig> {
    Ok(PlanConfig {
        compile_io_concurrency: required(values, "compile_io_concurrency")?.extract()?,
        io_bandwidth_bytes_per_second: required(values, "io_bandwidth_bytes_per_second")?
            .extract()?,
        io_operations_per_second: required(values, "io_operations_per_second")?.extract()?,
        coalescing_distance: required(values, "coalescing_distance")?.extract()?,
        max_coalesced_io_bytes: required(values, "max_coalesced_io_bytes")?.extract()?,
        target_decoded_bytes_per_job: required(values, "target_decoded_bytes_per_job")?
            .extract()?,
        delta_bytes: required(values, "delta_bytes")?.extract()?,
        limits: ResourceLimits {
            max_output_buffer_bytes: required(values, "max_output_buffer_bytes")?.extract()?,
            max_compile_arena_bytes: required(values, "max_compile_arena_bytes")?.extract()?,
            max_compile_working_set_bytes: required(values, "max_compile_working_set_bytes")?
                .extract()?,
            max_retained_whole_key_bytes: required(values, "max_retained_whole_key_bytes")?
                .extract()?,
            max_blocks_per_job: required(values, "max_blocks_per_job")?.extract()?,
            max_cells_per_job: required(values, "max_cells_per_job")?.extract()?,
            max_encoded_bytes_per_side: required(values, "max_encoded_bytes_per_side")?
                .extract()?,
            max_decoded_bytes_per_job: required(values, "max_decoded_bytes_per_job")?.extract()?,
        },
    })
}

pub(crate) fn session_config_from_dict(values: &Bound<'_, PyDict>) -> PyResult<SessionConfig> {
    let mode = required(values, "io_mode")?.extract::<String>()?;
    let queue_depth = required(values, "queue_depth")?.extract()?;
    Ok(SessionConfig {
        worker_count: required(values, "worker_count")?.extract()?,
        io_mode: parse_io_mode(&mode, queue_depth)?,
        max_inflight_jobs_per_worker: required(values, "max_inflight_jobs_per_worker")?
            .extract()?,
        max_inflight_encoded_bytes_per_worker: required(
            values,
            "max_inflight_encoded_bytes_per_worker",
        )?
        .extract()?,
        max_decoded_bytes_per_worker: required(values, "max_decoded_bytes_per_worker")?
            .extract()?,
        max_total_inflight_io_ops: required(values, "max_total_inflight_io_ops")?.extract()?,
        max_total_inflight_encoded_bytes: required(values, "max_total_inflight_encoded_bytes")?
            .extract()?,
        max_total_decoded_bytes: required(values, "max_total_decoded_bytes")?.extract()?,
    })
}

fn plan_config_to_dict<'py>(py: Python<'py>, config: &PlanConfig) -> PyResult<Bound<'py, PyDict>> {
    let values = PyDict::new(py);
    values.set_item("compile_io_concurrency", config.compile_io_concurrency)?;
    values.set_item(
        "io_bandwidth_bytes_per_second",
        config.io_bandwidth_bytes_per_second,
    )?;
    values.set_item("io_operations_per_second", config.io_operations_per_second)?;
    values.set_item("coalescing_distance", config.coalescing_distance)?;
    values.set_item("max_coalesced_io_bytes", config.max_coalesced_io_bytes)?;
    values.set_item(
        "target_decoded_bytes_per_job",
        config.target_decoded_bytes_per_job,
    )?;
    values.set_item("delta_bytes", config.delta_bytes)?;
    values.set_item(
        "max_output_buffer_bytes",
        config.limits.max_output_buffer_bytes,
    )?;
    values.set_item(
        "max_compile_arena_bytes",
        config.limits.max_compile_arena_bytes,
    )?;
    values.set_item(
        "max_compile_working_set_bytes",
        config.limits.max_compile_working_set_bytes,
    )?;
    values.set_item(
        "max_retained_whole_key_bytes",
        config.limits.max_retained_whole_key_bytes,
    )?;
    values.set_item("max_blocks_per_job", config.limits.max_blocks_per_job)?;
    values.set_item("max_cells_per_job", config.limits.max_cells_per_job)?;
    values.set_item(
        "max_encoded_bytes_per_side",
        config.limits.max_encoded_bytes_per_side,
    )?;
    values.set_item(
        "max_decoded_bytes_per_job",
        config.limits.max_decoded_bytes_per_job,
    )?;
    Ok(values)
}

fn session_config_to_dict<'py>(
    py: Python<'py>,
    config: &SessionConfig,
) -> PyResult<Bound<'py, PyDict>> {
    let values = PyDict::new(py);
    values.set_item("worker_count", config.worker_count)?;
    values.set_item("io_mode", io_mode_name(config.io_mode))?;
    values.set_item("queue_depth", io_mode_queue_depth(config.io_mode))?;
    values.set_item(
        "max_inflight_jobs_per_worker",
        config.max_inflight_jobs_per_worker,
    )?;
    values.set_item(
        "max_inflight_encoded_bytes_per_worker",
        config.max_inflight_encoded_bytes_per_worker,
    )?;
    values.set_item(
        "max_decoded_bytes_per_worker",
        config.max_decoded_bytes_per_worker,
    )?;
    values.set_item(
        "max_total_inflight_io_ops",
        config.max_total_inflight_io_ops,
    )?;
    values.set_item(
        "max_total_inflight_encoded_bytes",
        config.max_total_inflight_encoded_bytes,
    )?;
    values.set_item("max_total_decoded_bytes", config.max_total_decoded_bytes)?;
    Ok(values)
}

fn required<'py>(values: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    values
        .get_item(key)?
        .ok_or_else(|| invalid_argument(format!("missing normalized config field `{key}`")))
}

pub(crate) fn parse_io_mode(name: &str, queue_depth: u32) -> PyResult<IoMode> {
    match name {
        "blocking" => Ok(IoMode::Blocking),
        "uring" => Ok(IoMode::Uring { queue_depth }),
        "auto" => Ok(IoMode::Auto { queue_depth }),
        other => Err(invalid_argument(format!(
            "unknown I/O mode `{other}`; expected 'auto', 'blocking', or 'uring'"
        ))),
    }
}

pub(crate) const fn io_mode_name(mode: IoMode) -> &'static str {
    match mode {
        IoMode::Blocking => "blocking",
        IoMode::Uring { .. } => "uring",
        IoMode::Auto { .. } => "auto",
    }
}

pub(crate) const fn io_mode_queue_depth(mode: IoMode) -> u32 {
    match mode {
        IoMode::Blocking => 0,
        IoMode::Uring { queue_depth } | IoMode::Auto { queue_depth } => queue_depth,
    }
}
