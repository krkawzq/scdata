use crate::profile::{
    ProfileComponent, ProfileMetricId, ProfileMetricKind, ProfileMetricSnapshot, ProfileOwnedRound,
    ProfileRecorder, ProfileRegistry, ProfileRuntime, ProfileScope, ProfileScopeId,
    ProfileScopeKind, ProfileSnapshot, ProfileTimer,
};

use super::super::config::ProjectedSparseDataGroupStrategy;
use super::super::sparse::SparseProjectedDataGroupProfiler;

crate::scdata_profile_component!(pub(crate) const DATABANK_COMPONENT = "databank");

crate::scdata_profile_scope!(
    pub(crate) const DATABANK_PREFETCH_SOURCE_SCOPE = DATABANK_COMPONENT,
    "prefetch-source"
);
crate::scdata_profile_scope!(
    pub(crate) const DATABANK_PREFETCH_SUBMIT_SCOPE = DATABANK_COMPONENT,
    "prefetch-submit"
);
crate::scdata_profile_scope!(
    pub(crate) const DATABANK_PREFETCH_COORDINATOR_SCOPE = DATABANK_COMPONENT,
    "prefetch-coordinator"
);
crate::scdata_profile_scope!(
    pub(crate) const DATABANK_PREFETCH_REQUEST_SCOPE = DATABANK_COMPONENT,
    "prefetch-request"
);
crate::scdata_profile_scope!(
    pub(crate) const DATABANK_PREFETCH_RESPONSE_SCOPE = DATABANK_COMPONENT,
    "prefetch-response"
);
crate::scdata_profile_scope!(
    pub(crate) const DATABANK_PREFETCH_ASSEMBLE_SCOPE = DATABANK_COMPONENT,
    "prefetch-assemble"
);
crate::scdata_profile_scope!(
    pub(crate) const DATABANK_PREFETCH_MEMORY_SCOPE = DATABANK_COMPONENT,
    "prefetch-memory"
);

const BATCH_SOURCE_NEXT_NS: ProfileMetricId =
    ProfileMetricId::duration(DATABANK_PREFETCH_SOURCE_SCOPE, "next");
const SOURCE_BATCHES: ProfileMetricId =
    ProfileMetricId::count(DATABANK_PREFETCH_SOURCE_SCOPE, "batches");
const SOURCE_CELLS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_PREFETCH_SOURCE_SCOPE, "cells");
const SOURCE_EXHAUSTED: ProfileMetricId =
    ProfileMetricId::count(DATABANK_PREFETCH_SOURCE_SCOPE, "exhausted");

const SUBMIT_REQUEST_NS: ProfileMetricId =
    ProfileMetricId::duration(DATABANK_PREFETCH_SUBMIT_SCOPE, "request");
const SUBMIT_RESPONSE_NS: ProfileMetricId =
    ProfileMetricId::duration(DATABANK_PREFETCH_SUBMIT_SCOPE, "response");
const SUBMIT_REQUEST_ERRORS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_PREFETCH_SUBMIT_SCOPE, "request-errors");
const SUBMIT_RESPONSE_ERRORS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_PREFETCH_SUBMIT_SCOPE, "response-errors");

const COORDINATOR_WAIT_NS: ProfileMetricId =
    ProfileMetricId::duration(DATABANK_PREFETCH_COORDINATOR_SCOPE, "wait");
const OUTPUT_SEND_NS: ProfileMetricId =
    ProfileMetricId::duration(DATABANK_PREFETCH_COORDINATOR_SCOPE, "output-send");
const EMITTED_BATCHES: ProfileMetricId =
    ProfileMetricId::count(DATABANK_PREFETCH_COORDINATOR_SCOPE, "emitted-batches");
const EMITTED_ERRORS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_PREFETCH_COORDINATOR_SCOPE, "emitted-errors");
const OUTPUT_DROPPED: ProfileMetricId =
    ProfileMetricId::count(DATABANK_PREFETCH_COORDINATOR_SCOPE, "output-dropped");
const CANCELLED: ProfileMetricId =
    ProfileMetricId::count(DATABANK_PREFETCH_COORDINATOR_SCOPE, "cancelled");

const REQUEST_JOBS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_PREFETCH_REQUEST_SCOPE, "jobs");
const REQUEST_QUEUE_WAIT_NS: ProfileMetricId =
    ProfileMetricId::duration(DATABANK_PREFETCH_REQUEST_SCOPE, "queue-wait");
const REQUEST_PLAN_NS: ProfileMetricId =
    ProfileMetricId::duration(DATABANK_PREFETCH_REQUEST_SCOPE, "plan");
const REQUEST_SCHEDULE_NS: ProfileMetricId =
    ProfileMetricId::duration(DATABANK_PREFETCH_REQUEST_SCOPE, "schedule");
const REQUEST_SEND_NS: ProfileMetricId =
    ProfileMetricId::duration(DATABANK_PREFETCH_REQUEST_SCOPE, "send");
const REQUEST_TOTAL_NS: ProfileMetricId =
    ProfileMetricId::duration(DATABANK_PREFETCH_REQUEST_SCOPE, "total");
const REQUEST_ERRORS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_PREFETCH_REQUEST_SCOPE, "errors");

const RESPONSE_JOBS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_PREFETCH_RESPONSE_SCOPE, "jobs");
const RESPONSE_QUEUE_WAIT_NS: ProfileMetricId =
    ProfileMetricId::duration(DATABANK_PREFETCH_RESPONSE_SCOPE, "queue-wait");
const RESPONSE_TOTAL_NS: ProfileMetricId =
    ProfileMetricId::duration(DATABANK_PREFETCH_RESPONSE_SCOPE, "total");
const RESPONSE_ERRORS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_PREFETCH_RESPONSE_SCOPE, "errors");

const ASSEMBLE_TOTAL_NS: ProfileMetricId =
    ProfileMetricId::duration(DATABANK_PREFETCH_ASSEMBLE_SCOPE, "total");
const ALLOC_NS: ProfileMetricId =
    ProfileMetricId::duration(DATABANK_PREFETCH_ASSEMBLE_SCOPE, "alloc");
const SCATTER_NS: ProfileMetricId =
    ProfileMetricId::duration(DATABANK_PREFETCH_ASSEMBLE_SCOPE, "scatter");
const SCATTER_CALLS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_PREFETCH_ASSEMBLE_SCOPE, "scatter-calls");
const SPARSE_PROJECTED_SELECTED_ONLY_CALLS: ProfileMetricId = ProfileMetricId::count(
    DATABANK_PREFETCH_ASSEMBLE_SCOPE,
    "sparse-projected-selected-only",
);
const SPARSE_PROJECTED_READ_ALL_CALLS: ProfileMetricId = ProfileMetricId::count(
    DATABANK_PREFETCH_ASSEMBLE_SCOPE,
    "sparse-projected-read-all",
);
const SPARSE_PROJECTED_TOTAL_GROUPS: ProfileMetricId = ProfileMetricId::count(
    DATABANK_PREFETCH_ASSEMBLE_SCOPE,
    "sparse-projected-total-groups",
);
const SPARSE_PROJECTED_SELECTED_GROUPS: ProfileMetricId = ProfileMetricId::count(
    DATABANK_PREFETCH_ASSEMBLE_SCOPE,
    "sparse-projected-selected-groups",
);
const SPARSE_PROJECTED_EMPTY_SELECTIONS: ProfileMetricId = ProfileMetricId::count(
    DATABANK_PREFETCH_ASSEMBLE_SCOPE,
    "sparse-projected-empty-selections",
);
const SPARSE_PROJECTED_INDEX_SCAN_NS: ProfileMetricId = ProfileMetricId::duration(
    DATABANK_PREFETCH_ASSEMBLE_SCOPE,
    "sparse-projected-index-scan",
);
const SPARSE_PROJECTED_DATA_LOAD_NS: ProfileMetricId =
    ProfileMetricId::duration(DATABANK_PREFETCH_MEMORY_SCOPE, "sparse-projected-data-load");
const SPARSE_PROJECTED_DATA_LOAD_CALLS: ProfileMetricId = ProfileMetricId::count(
    DATABANK_PREFETCH_MEMORY_SCOPE,
    "sparse-projected-data-load-calls",
);
const SPARSE_PROJECTED_SELECTED_BYTES: ProfileMetricId =
    ProfileMetricId::bytes(DATABANK_PREFETCH_MEMORY_SCOPE, "sparse-projected-selected");
const SPARSE_PROJECTED_DATA_BYTES: ProfileMetricId =
    ProfileMetricId::bytes(DATABANK_PREFETCH_MEMORY_SCOPE, "sparse-projected-data");
const ASSEMBLED_BATCHES: ProfileMetricId =
    ProfileMetricId::count(DATABANK_PREFETCH_ASSEMBLE_SCOPE, "batches");
const OUTPUT_CELLS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_PREFETCH_ASSEMBLE_SCOPE, "output-cells");
const OUTPUT_GENES: ProfileMetricId =
    ProfileMetricId::count(DATABANK_PREFETCH_ASSEMBLE_SCOPE, "output-genes");
const OUTPUT_ELEMENTS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_PREFETCH_ASSEMBLE_SCOPE, "output-elements");
const OUTPUT_BYTES: ProfileMetricId =
    ProfileMetricId::bytes(DATABANK_PREFETCH_ASSEMBLE_SCOPE, "output");

const SCHEDULED_DRAIN_NS: ProfileMetricId =
    ProfileMetricId::duration(DATABANK_PREFETCH_MEMORY_SCOPE, "scheduled-drain");
const SCHEDULED_DRAIN_CALLS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_PREFETCH_MEMORY_SCOPE, "scheduled-drain-calls");
const SCHEDULED_DRAIN_BYTES: ProfileMetricId =
    ProfileMetricId::bytes(DATABANK_PREFETCH_MEMORY_SCOPE, "scheduled-drain");
const MEMORY_LOAD_NS: ProfileMetricId =
    ProfileMetricId::duration(DATABANK_PREFETCH_MEMORY_SCOPE, "load");
const MEMORY_LOAD_CALLS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_PREFETCH_MEMORY_SCOPE, "load-calls");
const MEMORY_LOAD_BYTES: ProfileMetricId =
    ProfileMetricId::bytes(DATABANK_PREFETCH_MEMORY_SCOPE, "load");

#[derive(Clone)]
pub(crate) struct ScheduledPrefetchProfiler {
    runtime: ProfileRuntime,
}

pub(crate) struct ScheduledPrefetchProfileRound {
    round: ProfileOwnedRound,
}

impl ScheduledPrefetchProfileRound {
    pub(crate) fn end(self) -> ProfileSnapshot {
        self.round.end()
    }
}

impl ScheduledPrefetchProfiler {
    pub(crate) fn from_env() -> Self {
        Self {
            runtime: ProfileRuntime::from_env_lazy(scheduled_prefetch_profile_registry),
        }
    }

    pub(crate) fn begin_round(&self) -> Option<(ScheduledPrefetchProfileRound, Self)> {
        if !self.runtime.is_global_enabled() {
            return None;
        }
        let round = self.runtime.start_owned();
        Some((ScheduledPrefetchProfileRound { round }, self.clone()))
    }

    pub(crate) fn start_batch_source_next(&self) -> ProfileTimer {
        self.timer_for_metric(BATCH_SOURCE_NEXT_NS)
    }

    pub(crate) fn start_submit_request(&self) -> ProfileTimer {
        self.timer_for_metric(SUBMIT_REQUEST_NS)
    }

    pub(crate) fn start_submit_response(&self) -> ProfileTimer {
        self.timer_for_metric(SUBMIT_RESPONSE_NS)
    }

    pub(crate) fn start_coordinator_wait(&self) -> ProfileTimer {
        self.timer_for_metric(COORDINATOR_WAIT_NS)
    }

    pub(crate) fn start_output_send(&self) -> ProfileTimer {
        self.timer_for_metric(OUTPUT_SEND_NS)
    }

    pub(crate) fn start_request_queue_wait(&self) -> ProfileTimer {
        self.timer_for_metric(REQUEST_QUEUE_WAIT_NS)
    }

    pub(crate) fn start_request_plan(&self) -> ProfileTimer {
        self.timer_for_metric(REQUEST_PLAN_NS)
    }

    pub(crate) fn start_request_schedule(&self) -> ProfileTimer {
        self.timer_for_metric(REQUEST_SCHEDULE_NS)
    }

    pub(crate) fn start_request_send(&self) -> ProfileTimer {
        self.timer_for_metric(REQUEST_SEND_NS)
    }

    pub(crate) fn start_request_total(&self) -> ProfileTimer {
        self.timer_for_metric(REQUEST_TOTAL_NS)
    }

    pub(crate) fn start_response_queue_wait(&self) -> ProfileTimer {
        self.timer_for_metric(RESPONSE_QUEUE_WAIT_NS)
    }

    pub(crate) fn start_response_total(&self) -> ProfileTimer {
        self.timer_for_metric(RESPONSE_TOTAL_NS)
    }

    pub(crate) fn start_assemble_total(&self) -> ProfileTimer {
        self.timer_for_metric(ASSEMBLE_TOTAL_NS)
    }

    pub(crate) fn start_scheduled_drain(&self) -> ProfileTimer {
        self.timer_for_metric(SCHEDULED_DRAIN_NS)
    }

    pub(crate) fn start_alloc(&self) -> ProfileTimer {
        self.timer_for_metric(ALLOC_NS)
    }

    pub(crate) fn start_memory_load(&self) -> ProfileTimer {
        self.timer_for_metric(MEMORY_LOAD_NS)
    }

    pub(crate) fn start_scatter(&self) -> ProfileTimer {
        self.timer_for_metric(SCATTER_NS)
    }

    pub(crate) fn record_batch_source_next(&self, started: ProfileTimer) {
        self.record_timer(BATCH_SOURCE_NEXT_NS, started);
    }

    pub(crate) fn record_source_batch(&self, cells: usize) {
        self.record(|recorder| {
            recorder.inc(SOURCE_BATCHES);
            recorder.add_usize(SOURCE_CELLS, cells);
        });
    }

    pub(crate) fn inc_source_exhausted(&self) {
        self.inc(SOURCE_EXHAUSTED);
    }

    pub(crate) fn record_submit_request(&self, started: ProfileTimer) {
        self.record_timer(SUBMIT_REQUEST_NS, started);
    }

    pub(crate) fn record_submit_response(&self, started: ProfileTimer) {
        self.record_timer(SUBMIT_RESPONSE_NS, started);
    }

    pub(crate) fn record_coordinator_wait(&self, started: ProfileTimer) {
        self.record_timer(COORDINATOR_WAIT_NS, started);
    }

    pub(crate) fn record_output_send(&self, started: ProfileTimer) {
        self.record_timer(OUTPUT_SEND_NS, started);
    }

    pub(crate) fn record_request_queue_wait(&self, started: ProfileTimer) {
        self.record_timer(REQUEST_QUEUE_WAIT_NS, started);
    }

    pub(crate) fn record_request_plan(&self, started: ProfileTimer) {
        self.record_timer(REQUEST_PLAN_NS, started);
    }

    pub(crate) fn record_request_schedule(&self, started: ProfileTimer) {
        self.record_timer(REQUEST_SCHEDULE_NS, started);
    }

    pub(crate) fn record_request_send(&self, started: ProfileTimer) {
        self.record_timer(REQUEST_SEND_NS, started);
    }

    pub(crate) fn record_request_total(&self, started: ProfileTimer) {
        self.record_timer(REQUEST_TOTAL_NS, started);
    }

    pub(crate) fn record_response_queue_wait(&self, started: ProfileTimer) {
        self.record_timer(RESPONSE_QUEUE_WAIT_NS, started);
    }

    pub(crate) fn record_response_total(&self, started: ProfileTimer) {
        self.record_timer(RESPONSE_TOTAL_NS, started);
    }

    pub(crate) fn record_assemble_total(&self, started: ProfileTimer) {
        self.record_timer(ASSEMBLE_TOTAL_NS, started);
    }

    pub(crate) fn record_scheduled_drain(&self, started: ProfileTimer, bytes: usize) {
        self.record(|recorder| {
            recorder.record_timer(SCHEDULED_DRAIN_NS, started);
            recorder.inc(SCHEDULED_DRAIN_CALLS);
            recorder.add_usize(SCHEDULED_DRAIN_BYTES, bytes);
        });
    }

    pub(crate) fn record_alloc(&self, started: ProfileTimer) {
        self.record_timer(ALLOC_NS, started);
    }

    pub(crate) fn record_memory_load(&self, started: ProfileTimer, bytes: usize) {
        self.record(|recorder| {
            recorder.record_timer(MEMORY_LOAD_NS, started);
            recorder.inc(MEMORY_LOAD_CALLS);
            recorder.add_usize(MEMORY_LOAD_BYTES, bytes);
        });
    }

    pub(crate) fn record_scatter(&self, started: ProfileTimer) {
        self.record(|recorder| {
            recorder.record_timer(SCATTER_NS, started);
            recorder.inc(SCATTER_CALLS);
        });
    }

    pub(crate) fn record_assembled_batch(
        &self,
        cells: usize,
        genes: usize,
        elements: usize,
        bytes: usize,
    ) {
        self.record(|recorder| {
            recorder.inc(ASSEMBLED_BATCHES);
            recorder.add_usize(OUTPUT_CELLS, cells);
            recorder.add_usize(OUTPUT_GENES, genes);
            recorder.add_usize(OUTPUT_ELEMENTS, elements);
            recorder.add_usize(OUTPUT_BYTES, bytes);
        });
    }

    pub(crate) fn inc_request_job(&self) {
        self.inc(REQUEST_JOBS);
    }

    pub(crate) fn inc_response_job(&self) {
        self.inc(RESPONSE_JOBS);
    }

    pub(crate) fn inc_emitted_batch(&self) {
        self.inc(EMITTED_BATCHES);
    }

    pub(crate) fn inc_emitted_error(&self) {
        self.inc(EMITTED_ERRORS);
    }

    pub(crate) fn inc_output_dropped(&self) {
        self.inc(OUTPUT_DROPPED);
    }

    pub(crate) fn inc_cancelled(&self) {
        self.inc(CANCELLED);
    }

    pub(crate) fn inc_submit_request_error(&self) {
        self.inc(SUBMIT_REQUEST_ERRORS);
    }

    pub(crate) fn inc_submit_response_error(&self) {
        self.inc(SUBMIT_RESPONSE_ERRORS);
    }

    pub(crate) fn inc_request_error(&self) {
        self.inc(REQUEST_ERRORS);
    }

    pub(crate) fn inc_response_error(&self) {
        self.inc(RESPONSE_ERRORS);
    }

    fn timer_for_metric(&self, metric: ProfileMetricId) -> ProfileTimer {
        self.timer(metric.scope)
    }

    fn timer(&self, scope: ProfileScopeId) -> ProfileTimer {
        self.runtime
            .with_recorder(|recorder| recorder.timer(scope))
            .unwrap_or_else(ProfileTimer::disabled)
    }

    fn inc(&self, metric: ProfileMetricId) {
        self.record(|recorder| recorder.inc(metric));
    }

    fn record_timer(&self, metric: ProfileMetricId, timer: ProfileTimer) {
        self.record(|recorder| recorder.record_timer(metric, timer));
    }

    fn record(&self, f: impl FnOnce(ProfileRecorder<'_>)) {
        let _ = self.runtime.with_recorder(f);
    }

    pub(crate) fn print_summary(&self, snapshot: ProfileSnapshot) {
        let metrics = snapshot
            .metrics
            .iter()
            .filter_map(format_metric)
            .collect::<Vec<_>>()
            .join(" ");
        if metrics.is_empty() {
            println!("{}", snapshot.summary_line());
        } else {
            println!("{} {}", snapshot.summary_line(), metrics);
        }
    }
}

impl SparseProjectedDataGroupProfiler for ScheduledPrefetchProfiler {
    fn start_sparse_projected_index_scan(&self) -> ProfileTimer {
        self.timer_for_metric(SPARSE_PROJECTED_INDEX_SCAN_NS)
    }

    fn record_sparse_projected_index_scan(&self, started: ProfileTimer) {
        self.record_timer(SPARSE_PROJECTED_INDEX_SCAN_NS, started);
    }

    fn start_sparse_projected_data_load(&self) -> ProfileTimer {
        self.timer_for_metric(SPARSE_PROJECTED_DATA_LOAD_NS)
    }

    fn record_sparse_projected_data_load(&self, started: ProfileTimer, bytes: usize) {
        self.record(|recorder| {
            recorder.record_timer(SPARSE_PROJECTED_DATA_LOAD_NS, started);
            recorder.inc(SPARSE_PROJECTED_DATA_LOAD_CALLS);
            recorder.add_usize(SPARSE_PROJECTED_DATA_BYTES, bytes);
        });
    }

    fn record_sparse_projected_groups(
        &self,
        strategy: ProjectedSparseDataGroupStrategy,
        total_groups: usize,
        selected_groups: usize,
        selected_bytes: usize,
    ) {
        self.record(|recorder| {
            match strategy {
                ProjectedSparseDataGroupStrategy::SelectedOnly => {
                    recorder.inc(SPARSE_PROJECTED_SELECTED_ONLY_CALLS);
                }
                ProjectedSparseDataGroupStrategy::ReadAll => {
                    recorder.inc(SPARSE_PROJECTED_READ_ALL_CALLS);
                }
            }
            recorder.add_usize(SPARSE_PROJECTED_TOTAL_GROUPS, total_groups);
            recorder.add_usize(SPARSE_PROJECTED_SELECTED_GROUPS, selected_groups);
            recorder.add_usize(SPARSE_PROJECTED_SELECTED_BYTES, selected_bytes);
            if selected_groups == 0 {
                recorder.inc(SPARSE_PROJECTED_EMPTY_SELECTIONS);
            }
        });
    }
}

fn scheduled_prefetch_profile_registry() -> ProfileRegistry {
    ProfileRegistry::new()
        .with_component(
            ProfileComponent::new(DATABANK_COMPONENT).described("DataBank scheduled prefetch"),
        )
        .with_scope(
            ProfileScope::new(DATABANK_PREFETCH_SOURCE_SCOPE)
                .kind(ProfileScopeKind::Timer)
                .described("scheduled prefetch batch source polling"),
        )
        .with_scope(
            ProfileScope::new(DATABANK_PREFETCH_SUBMIT_SCOPE)
                .kind(ProfileScopeKind::Timer)
                .described("scheduled prefetch compute job submission"),
        )
        .with_scope(
            ProfileScope::new(DATABANK_PREFETCH_COORDINATOR_SCOPE)
                .kind(ProfileScopeKind::Event)
                .described("scheduled prefetch producer coordination"),
        )
        .with_scope(
            ProfileScope::new(DATABANK_PREFETCH_REQUEST_SCOPE)
                .kind(ProfileScopeKind::Event)
                .described("scheduled prefetch request planning jobs"),
        )
        .with_scope(
            ProfileScope::new(DATABANK_PREFETCH_RESPONSE_SCOPE)
                .kind(ProfileScopeKind::Event)
                .described("scheduled prefetch response assembly jobs"),
        )
        .with_scope(
            ProfileScope::new(DATABANK_PREFETCH_ASSEMBLE_SCOPE)
                .kind(ProfileScopeKind::Timer)
                .described("scheduled prefetch batch assembly"),
        )
        .with_scope(
            ProfileScope::new(DATABANK_PREFETCH_MEMORY_SCOPE)
                .kind(ProfileScopeKind::Bytes)
                .described("scheduled prefetch memory materialization"),
        )
}

fn format_metric(metric: &ProfileMetricSnapshot) -> Option<String> {
    let value = metric.value();
    if value == 0 {
        return None;
    }
    let name = format!("{}.{}", metric.id.scope, metric.id.name)
        .replace('.', "_")
        .replace('-', "_");
    Some(match metric.id.kind {
        ProfileMetricKind::DurationNs => format!("{name}_ms={:.3}", metric.as_ms()),
        ProfileMetricKind::Bytes => format!("{name}_mib={:.3}", metric.as_mib()),
        ProfileMetricKind::Count | ProfileMetricKind::Gauge | ProfileMetricKind::Custom(_) => {
            format!("{name}={value}")
        }
    })
}
