//! Convert immutable Rust statistics snapshots into ordinary Python mappings.

use pyo3::prelude::*;
use pyo3::types::PyDict;
#[cfg(feature = "profile")]
use pyo3::types::PyList;
#[cfg(feature = "profile")]
use scdata::WorkerRuntimeStats;
use scdata::{PlanStats, RuntimeStats, SessionState};

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
        "maximum_encoded_bytes_per_side",
        stats.maximum_encoded_bytes_per_side,
    )?;
    values.set_item(
        "maximum_decoded_bytes_per_job",
        stats.maximum_decoded_bytes_per_job,
    )?;
    values.set_item("arena_bytes", stats.arena_bytes)?;
    values.set_item("compile_working_set_bytes", stats.compile_working_set_bytes)?;
    values.set_item("retained_whole_key_bytes", stats.retained_whole_key_bytes)?;
    values.set_item("output_ring_bytes", stats.output_ring_bytes)?;
    values.set_item("compile_time_io_bytes", stats.compile_time_io_bytes)?;
    values.set_item("compile_time_io_ops", stats.compile_time_io_ops)?;
    values.set_item("predicted_io_seconds", stats.predicted_io_seconds)?;
    #[cfg(feature = "profile")]
    {
        values.set_item("compile_resolve_ns", stats.compile_resolve_ns)?;
        values.set_item("compile_same_block_ns", stats.compile_same_block_ns)?;
        values.set_item("compile_merge_runs_ns", stats.compile_merge_runs_ns)?;
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
    values.set_item("worker_count", stats.worker_count)?;
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
        values.set_item("data_decode_nanoseconds", stats.data_decode_nanoseconds)?;
        values.set_item(
            "indices_decode_nanoseconds",
            stats.indices_decode_nanoseconds,
        )?;
        values.set_item("validation_nanoseconds", stats.validation_nanoseconds)?;
        values.set_item("scatter_nanoseconds", stats.scatter_nanoseconds)?;
        values.set_item(
            "scatter_kernel_nanoseconds",
            stats.scatter_kernel_nanoseconds,
        )?;
        values.set_item("completion_nanoseconds", stats.completion_nanoseconds)?;
        values.set_item("window_wait_nanoseconds", stats.window_wait_nanoseconds)?;
        values.set_item("consumer_wait_nanoseconds", stats.consumer_wait_nanoseconds)?;
        values.set_item("completed_jobs", stats.completed_jobs)?;
        values.set_item("completed_cells", stats.completed_cells)?;
        values.set_item("decoded_blocks", stats.decoded_blocks)?;
        values.set_item("decoded_bytes", stats.decoded_bytes)?;
        values.set_item("claim_cas_retries", stats.claim_cas_retries)?;
        values.set_item("window_block_events", stats.window_block_events)?;
        values.set_item("local_full_events", stats.local_full_events)?;
        values.set_item("peak_inflight_jobs", stats.peak_inflight_jobs)?;
        values.set_item("peak_inflight_read_ops", stats.peak_inflight_read_ops)?;
        values.set_item(
            "peak_inflight_encoded_bytes",
            stats.peak_inflight_encoded_bytes,
        )?;
        let workers = PyList::empty(py);
        for worker in &stats.workers {
            workers.append(worker_stats_to_dict(py, worker)?)?;
        }
        values.set_item("workers", workers)?;
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

#[cfg(feature = "profile")]
fn worker_stats_to_dict<'py>(
    py: Python<'py>,
    stats: &WorkerRuntimeStats,
) -> PyResult<Bound<'py, PyDict>> {
    let values = PyDict::new(py);
    values.set_item("worker_id", stats.worker_id)?;
    values.set_item("completed_jobs", stats.completed_jobs)?;
    values.set_item("completed_cells", stats.completed_cells)?;
    values.set_item("decoded_blocks", stats.decoded_blocks)?;
    values.set_item("decoded_bytes", stats.decoded_bytes)?;
    values.set_item("data_decode_nanoseconds", stats.data_decode_nanoseconds)?;
    values.set_item(
        "indices_decode_nanoseconds",
        stats.indices_decode_nanoseconds,
    )?;
    values.set_item("validation_nanoseconds", stats.validation_nanoseconds)?;
    values.set_item(
        "scatter_kernel_nanoseconds",
        stats.scatter_kernel_nanoseconds,
    )?;
    values.set_item("completion_nanoseconds", stats.completion_nanoseconds)?;
    values.set_item("window_wait_nanoseconds", stats.window_wait_nanoseconds)?;
    values.set_item("io_wait_nanoseconds", stats.io_wait_nanoseconds)?;
    values.set_item("physical_read_ops", stats.physical_read_ops)?;
    values.set_item("physical_read_bytes", stats.physical_read_bytes)?;
    values.set_item("short_read_retries", stats.short_read_retries)?;
    values.set_item(
        "whole_key_materializations",
        stats.whole_key_materializations,
    )?;
    values.set_item("claim_cas_retries", stats.claim_cas_retries)?;
    values.set_item("window_block_events", stats.window_block_events)?;
    values.set_item("local_full_events", stats.local_full_events)?;
    values.set_item("uring_prepared_read_sqes", stats.uring_prepared_read_sqes)?;
    values.set_item("uring_submitted_read_sqes", stats.uring_submitted_read_sqes)?;
    values.set_item("uring_submit_calls", stats.uring_submit_calls)?;
    values.set_item("uring_cqes", stats.uring_cqes)?;
    values.set_item("uring_cancel_requests", stats.uring_cancel_requests)?;
    values.set_item("uring_cancel_cqes", stats.uring_cancel_cqes)?;
    Ok(values)
}
