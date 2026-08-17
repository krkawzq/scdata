//! Conversion between normalized Python mappings and Rust runtime configs.

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use sc_load::{IoMergeOptions, IoMergePolicy, IoMode, PlanConfig, ResourceLimits, SessionConfig};

use crate::error::invalid_input as invalid_argument;

pub(crate) fn plan_config_from_dict(values: &Bound<'_, PyDict>) -> PyResult<PlanConfig> {
    let io_merge = required(values, "io_merge")?;
    let io_merge = io_merge
        .downcast::<PyDict>()
        .map_err(|_| invalid_argument("normalized io_merge config must be a mapping"))?;
    Ok(PlanConfig {
        compile_io_concurrency: required(values, "compile_io_concurrency")?.extract()?,
        io_merge: IoMergeOptions {
            policy: match required(io_merge, "policy")?.extract::<String>()?.as_str() {
                "off" => IoMergePolicy::Off,
                "adjacent" => IoMergePolicy::Adjacent,
                "cost" => IoMergePolicy::CostAware,
                value => {
                    return Err(invalid_argument(format!(
                        "unknown io_merge policy `{value}`"
                    )))
                }
            },
            max_coalesced_io_bytes: required(io_merge, "max_coalesced_io_bytes")?.extract()?,
            max_io_gap_bytes: required(io_merge, "max_io_gap_bytes")?.extract()?,
            max_io_amplification_ratio: required(io_merge, "max_io_amplification_ratio")?
                .extract()?,
            max_decode_ops_per_io_task: required(io_merge, "max_decode_ops_per_io_task")?
                .extract()?,
            max_decoded_bytes_per_io_task: required(io_merge, "max_decoded_bytes_per_io_task")?
                .extract()?,
            max_encoded_staging_bytes_per_task: required(
                io_merge,
                "max_encoded_staging_bytes_per_task",
            )?
            .extract()?,
            io_bandwidth_bytes_per_second: required(io_merge, "io_bandwidth_bytes_per_second")?
                .extract()?,
            io_operations_per_second: required(io_merge, "io_operations_per_second")?.extract()?,
            io_merge_delta_bytes: required(io_merge, "io_merge_delta_bytes")?.extract()?,
            initialize_parallelism_hint: required(io_merge, "initialize_parallelism_hint")?
                .extract()?,
            regular_io_parallelism_hint: required(io_merge, "regular_io_parallelism_hint")?
                .extract()?,
            min_tasks_per_worker: required(io_merge, "min_tasks_per_worker")?.extract()?,
        },
        cache_capacity_bytes: required(values, "cache_capacity_bytes")?.extract()?,
        cache_alignment: required(values, "cache_alignment")?.extract()?,
        cache_fragmentation_slack_bytes: required(values, "cache_fragmentation_slack_bytes")?
            .extract()?,
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
        worker_count: required(values, "num_workers")?.extract()?,
        initialize_workers: required(values, "initialize_workers")?.extract()?,
        initialize_inflight_io_ops: required(values, "initialize_inflight_io_ops")?.extract()?,
        initialize_inflight_encoded_bytes: required(values, "initialize_inflight_encoded_bytes")?
            .extract()?,
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
