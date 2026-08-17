//! Convert immutable Rust statistics snapshots into ordinary Python mappings.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use sc_load::{PlanStats, RuntimeStats, SessionState};

use crate::config::{io_mode_name, io_mode_queue_depth};

pub(crate) fn plan_stats_to_dict<'py>(
    py: Python<'py>,
    stats: &PlanStats,
) -> PyResult<Bound<'py, PyDict>> {
    let values = PyDict::new(py);
    values.set_item("input_rows", stats.input_rows)?;
    values.set_item("block_jobs", stats.block_jobs)?;
    values.set_item("jobs", stats.jobs)?;
    values.set_item("data_io_ops", stats.data_io_ops)?;
    values.set_item("indices_io_ops", stats.indices_io_ops)?;
    values.set_item("predicted_physical_bytes", stats.predicted_physical_bytes)?;
    values.set_item("gap_bytes", stats.gap_bytes)?;
    values.set_item(
        "max_encoded_bytes_per_side",
        stats.maximum_encoded_bytes_per_side,
    )?;
    values.set_item(
        "max_decoded_bytes_per_job",
        stats.maximum_decoded_bytes_per_job,
    )?;
    values.set_item("arena_bytes", stats.arena_bytes)?;
    values.set_item("compile_working_set_bytes", stats.compile_working_set_bytes)?;
    values.set_item("retained_whole_key_bytes", stats.retained_whole_key_bytes)?;
    values.set_item("output_ring_bytes", stats.output_ring_bytes)?;
    values.set_item("compile_time_io_bytes", stats.compile_time_io_bytes)?;
    values.set_item("compile_time_io_ops", stats.compile_time_io_ops)?;
    values.set_item("predicted_io_seconds", stats.predicted_io_seconds)?;
    values.set_item("cache_capacity_bytes", stats.cache_capacity_bytes)?;
    values.set_item("cache_arena_bytes", stats.cache_arena_bytes)?;
    values.set_item(
        "cache_alignment_loss_bytes",
        stats.cache_alignment_loss_bytes,
    )?;
    values.set_item("unique_cache_objects", stats.unique_cache_objects)?;
    values.set_item("residency_loads", stats.residency_loads)?;
    values.set_item("residency_reloads", stats.residency_reloads)?;
    values.set_item("cache_reference_hits", stats.cache_reference_hits)?;
    values.set_item("cache_reference_misses", stats.cache_reference_misses)?;
    values.set_item("cache_capacity_stalls", stats.cache_capacity_stalls)?;
    values.set_item(
        "cache_fragmentation_stalls",
        stats.cache_fragmentation_stalls,
    )?;
    values.set_item("cache_horizon_max_batches", stats.cache_horizon_max_batches)?;
    values.set_item("output_ring_slots", stats.output_ring_slots)?;
    values.set_item("initialize_io_tasks", stats.initialize_io_tasks)?;
    values.set_item("executable_tasks", stats.executable_tasks)?;
    values.set_item("dependency_edges", stats.dependency_edges)?;
    values.set_item("independent_block_loads", stats.independent_block_loads)?;
    values.set_item("fused_io_tasks", stats.fused_io_tasks)?;
    values.set_item("predicted_io_ops_saved", stats.predicted_io_ops_saved)?;
    values.set_item("io_payload_bytes", stats.io_payload_bytes)?;
    values.set_item("io_span_bytes", stats.io_span_bytes)?;
    values.set_item("io_read_amplification", stats.io_read_amplification)?;
    values.set_item(
        "max_decode_ops_per_io_task",
        stats.maximum_decode_ops_per_io_task,
    )?;
    values.set_item(
        "max_decoded_bytes_per_io_task",
        stats.maximum_decoded_bytes_per_io_task,
    )?;
    values.set_item("initialize_fused_io_tasks", stats.initialize_fused_io_tasks)?;
    values.set_item("regular_fused_io_tasks", stats.regular_fused_io_tasks)?;
    #[cfg(feature = "profile")]
    {
        values.set_item("compile_resolve_ns", stats.compile_resolve_ns)?;
        values.set_item("compile_finalize_ns", stats.compile_finalize_ns)?;
    }
    Ok(values)
}

pub(crate) fn runtime_stats_to_dict<'py>(
    py: Python<'py>,
    stats: &RuntimeStats,
) -> PyResult<Bound<'py, PyDict>> {
    let values = PyDict::new(py);
    values.set_item("requested_io_mode", io_mode_name(stats.requested_io_mode))?;
    values.set_item(
        "requested_queue_depth",
        io_mode_queue_depth(stats.requested_io_mode),
    )?;
    values.set_item("actual_io_mode", io_mode_name(stats.actual_io_mode))?;
    values.set_item(
        "actual_queue_depth",
        io_mode_queue_depth(stats.actual_io_mode),
    )?;
    values.set_item("num_workers", stats.worker_count)?;
    values.set_item(
        "max_inflight_jobs_per_worker",
        stats.max_inflight_jobs_per_worker,
    )?;
    values.set_item(
        "max_inflight_encoded_bytes_per_worker",
        stats.max_inflight_encoded_bytes_per_worker,
    )?;
    values.set_item(
        "max_decoded_bytes_per_worker",
        stats.max_decoded_bytes_per_worker,
    )?;
    values.set_item("state", session_state_name(stats.state))?;
    #[cfg(feature = "profile")]
    {
        values.set_item("physical_read_ops", stats.physical_read_ops)?;
        values.set_item("physical_read_bytes", stats.physical_read_bytes)?;
        values.set_item("short_read_retries", stats.short_read_retries)?;
        values.set_item(
            "whole_key_materializations",
            stats.whole_key_materializations,
        )?;
        values.set_item("uring_prepared_read_sqes", stats.uring_prepared_read_sqes)?;
        values.set_item("uring_submitted_read_sqes", stats.uring_submitted_read_sqes)?;
        values.set_item("uring_submit_calls", stats.uring_submit_calls)?;
        values.set_item("uring_cqes", stats.uring_cqes)?;
        values.set_item("uring_cancel_requests", stats.uring_cancel_requests)?;
        values.set_item("uring_cancel_cqes", stats.uring_cancel_cqes)?;
        values.set_item("io_wait_nanoseconds", stats.io_wait_nanoseconds)?;
        values.set_item("decode_nanoseconds", stats.decode_nanoseconds)?;
        values.set_item("validation_nanoseconds", stats.validation_nanoseconds)?;
        values.set_item("scatter_nanoseconds", stats.scatter_nanoseconds)?;
        values.set_item("completion_nanoseconds", stats.completion_nanoseconds)?;
        values.set_item("consumer_wait_nanoseconds", stats.consumer_wait_nanoseconds)?;
        values.set_item("completed_jobs", stats.completed_jobs)?;
        values.set_item("completed_cells", stats.completed_cells)?;
        values.set_item("decoded_blocks", stats.decoded_blocks)?;
        values.set_item("decoded_bytes", stats.decoded_bytes)?;
        values.set_item("peak_inflight_jobs", stats.peak_inflight_jobs)?;
        values.set_item("peak_inflight_read_ops", stats.peak_inflight_read_ops)?;
        values.set_item(
            "peak_inflight_encoded_bytes",
            stats.peak_inflight_encoded_bytes,
        )?;
    }
    Ok(values)
}

pub(crate) const fn session_state_name(state: SessionState) -> &'static str {
    match state {
        SessionState::Running => "running",
        SessionState::Failed => "failed",
        SessionState::Cancelled => "cancelled",
        SessionState::Finished => "finished",
    }
}
