use std::sync::Arc;

use crate::profile::{
    ProfileComponent, ProfileMetricId, ProfileMetricKind, ProfileMetricSnapshot, ProfileOwnedRound,
    ProfileRecorder, ProfileRegistry, ProfileRuntime, ProfileScope, ProfileScopeKind,
    ProfileSnapshot, ProfileTimer,
};

crate::scdata_profile_component!(pub const DATABANK_COMPONENT = "databank");

crate::scdata_profile_scope!(pub const DATABANK_LIFECYCLE_SCOPE = DATABANK_COMPONENT, "lifecycle");
crate::scdata_profile_scope!(pub const DATABANK_ACCESS_SCOPE = DATABANK_COMPONENT, "access");
crate::scdata_profile_scope!(pub const DATABANK_PREFETCH_SCOPE = DATABANK_COMPONENT, "prefetch");
crate::scdata_profile_scope!(
    pub const DATABANK_SCHEDULED_API_SCOPE = DATABANK_COMPONENT,
    "scheduled-api"
);

const LIFECYCLE_WORK_NS: ProfileMetricId =
    ProfileMetricId::duration(DATABANK_LIFECYCLE_SCOPE, "work");
const REGISTER_CALLS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_LIFECYCLE_SCOPE, "register-calls");
const REGISTER_DENSE_1D: ProfileMetricId =
    ProfileMetricId::count(DATABANK_LIFECYCLE_SCOPE, "register-dense-1d");
const REGISTER_DENSE_2D: ProfileMetricId =
    ProfileMetricId::count(DATABANK_LIFECYCLE_SCOPE, "register-dense-2d");
const REGISTER_SPARSE_CSR: ProfileMetricId =
    ProfileMetricId::count(DATABANK_LIFECYCLE_SCOPE, "register-sparse-csr");
const REGISTER_GENES: ProfileMetricId =
    ProfileMetricId::count(DATABANK_LIFECYCLE_SCOPE, "register-genes");
const REGISTER_ERRORS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_LIFECYCLE_SCOPE, "register-errors");
const UNREGISTER_CALLS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_LIFECYCLE_SCOPE, "unregister-calls");
const UNREGISTER_RETIRED: ProfileMetricId =
    ProfileMetricId::count(DATABANK_LIFECYCLE_SCOPE, "unregister-retired");
const UNREGISTER_ERRORS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_LIFECYCLE_SCOPE, "unregister-errors");
const CLEANUP_CALLS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_LIFECYCLE_SCOPE, "cleanup-calls");
const CLEANUP_DATASETS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_LIFECYCLE_SCOPE, "cleanup-datasets");
const CLEANUP_RETAINED: ProfileMetricId =
    ProfileMetricId::count(DATABANK_LIFECYCLE_SCOPE, "cleanup-retained");
const CLEANUP_ERRORS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_LIFECYCLE_SCOPE, "cleanup-errors");
const DROP_CALLS: ProfileMetricId = ProfileMetricId::count(DATABANK_LIFECYCLE_SCOPE, "drop-calls");
const DROP_DATASETS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_LIFECYCLE_SCOPE, "drop-datasets");

const ACCESS_WORK_NS: ProfileMetricId = ProfileMetricId::duration(DATABANK_ACCESS_SCOPE, "work");
const ACCESS_CALLS: ProfileMetricId = ProfileMetricId::count(DATABANK_ACCESS_SCOPE, "calls");
const ACCESS_BORROWED_CALLS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_ACCESS_SCOPE, "borrowed");
const ACCESS_OWNED_CALLS: ProfileMetricId = ProfileMetricId::count(DATABANK_ACCESS_SCOPE, "owned");
const ACCESS_BY_GENE_NAMES_CALLS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_ACCESS_SCOPE, "by-gene-names");
const ACCESS_UNCHECKED_CALLS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_ACCESS_SCOPE, "unchecked");
const ACCESS_ERRORS: ProfileMetricId = ProfileMetricId::count(DATABANK_ACCESS_SCOPE, "errors");
const ACCESS_CELLS: ProfileMetricId = ProfileMetricId::count(DATABANK_ACCESS_SCOPE, "cells");
const ACCESS_GENES: ProfileMetricId = ProfileMetricId::count(DATABANK_ACCESS_SCOPE, "genes");
const ACCESS_OUTPUT_ELEMENTS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_ACCESS_SCOPE, "output-elements");
const ACCESS_OUTPUT_BYTES: ProfileMetricId =
    ProfileMetricId::bytes(DATABANK_ACCESS_SCOPE, "output");

const PREFETCH_WORK_NS: ProfileMetricId =
    ProfileMetricId::duration(DATABANK_PREFETCH_SCOPE, "work");
const PREFETCH_CALLS: ProfileMetricId = ProfileMetricId::count(DATABANK_PREFETCH_SCOPE, "calls");
const PREFETCH_ERRORS: ProfileMetricId = ProfileMetricId::count(DATABANK_PREFETCH_SCOPE, "errors");
const PREFETCH_CELLS: ProfileMetricId = ProfileMetricId::count(DATABANK_PREFETCH_SCOPE, "cells");

const SCHEDULED_WORK_NS: ProfileMetricId =
    ProfileMetricId::duration(DATABANK_SCHEDULED_API_SCOPE, "work");
const SCHEDULED_CALLS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_SCHEDULED_API_SCOPE, "calls");
const SCHEDULED_SINGLE_CALLS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_SCHEDULED_API_SCOPE, "single");
const SCHEDULED_SINGLE_BY_GENE_NAMES_CALLS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_SCHEDULED_API_SCOPE, "single-by-gene-names");
const SCHEDULED_MULTI_CALLS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_SCHEDULED_API_SCOPE, "multi");
const SCHEDULED_MULTI_BY_GENE_NAMES_CALLS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_SCHEDULED_API_SCOPE, "multi-by-gene-names");
const SCHEDULED_DATASETS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_SCHEDULED_API_SCOPE, "datasets");
const SCHEDULED_ERRORS: ProfileMetricId =
    ProfileMetricId::count(DATABANK_SCHEDULED_API_SCOPE, "errors");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DataBankRegisterKind {
    Dense1D,
    Dense2D,
    SparseCsr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DataBankAccessKind {
    Borrowed,
    ByGeneNames,
    Owned,
    OwnedByGeneNames,
    Unchecked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DataBankScheduledKind {
    Single,
    SingleByGeneNames,
    Multi,
    MultiByGeneNames,
}

#[derive(Debug, Clone)]
pub(crate) struct DataBankProfile {
    inner: Arc<DataBankProfileInner>,
}

#[derive(Debug)]
struct DataBankProfileInner {
    runtime: ProfileRuntime,
    _auto_round: Option<ProfileOwnedRound>,
}

impl DataBankProfile {
    #[cfg(all(test, feature = "profile"))]
    pub(crate) fn enabled(label: impl Into<String>) -> Self {
        Self::with_auto_round(
            ProfileRuntime::enabled_lazy(label, databank_profile_registry),
            None,
        )
    }

    pub(crate) fn from_env() -> Self {
        let runtime = ProfileRuntime::from_env_lazy(databank_profile_registry);
        let auto_round = runtime.is_global_enabled().then(|| runtime.start_owned());
        Self::with_auto_round(runtime, auto_round)
    }

    pub(crate) fn runtime(&self) -> &ProfileRuntime {
        &self.inner.runtime
    }

    pub(crate) fn snapshot(&self) -> ProfileSnapshot {
        self.inner.runtime.snapshot()
    }

    pub(crate) fn snapshot_and_reset(&self) -> ProfileSnapshot {
        self.inner.runtime.snapshot_and_reset()
    }

    pub(crate) fn reset_metrics(&self) {
        self.inner.runtime.reset_metrics();
    }

    pub(crate) fn lifecycle_timer(&self) -> ProfileTimer {
        self.timer(DATABANK_LIFECYCLE_SCOPE)
    }

    pub(crate) fn access_timer(&self) -> ProfileTimer {
        self.timer(DATABANK_ACCESS_SCOPE)
    }

    pub(crate) fn prefetch_timer(&self) -> ProfileTimer {
        self.timer(DATABANK_PREFETCH_SCOPE)
    }

    pub(crate) fn scheduled_timer(&self) -> ProfileTimer {
        self.timer(DATABANK_SCHEDULED_API_SCOPE)
    }

    pub(crate) fn record_register(
        &self,
        started: ProfileTimer,
        kind: DataBankRegisterKind,
        genes: usize,
        error: bool,
    ) {
        self.record(|recorder| {
            recorder.inc(REGISTER_CALLS);
            recorder.inc(register_kind_metric(kind));
            recorder.add_usize(REGISTER_GENES, genes);
            if error {
                recorder.inc(REGISTER_ERRORS);
            }
            recorder.record_timer(LIFECYCLE_WORK_NS, started);
        });
    }

    pub(crate) fn record_unregister(&self, started: ProfileTimer, retired: bool, error: bool) {
        self.record(|recorder| {
            recorder.inc(UNREGISTER_CALLS);
            if retired {
                recorder.inc(UNREGISTER_RETIRED);
            }
            if error {
                recorder.inc(UNREGISTER_ERRORS);
            }
            recorder.record_timer(LIFECYCLE_WORK_NS, started);
        });
    }

    pub(crate) fn record_cleanup(
        &self,
        started: ProfileTimer,
        datasets: usize,
        retained: usize,
        error: bool,
    ) {
        self.record(|recorder| {
            recorder.inc(CLEANUP_CALLS);
            recorder.add_usize(CLEANUP_DATASETS, datasets);
            recorder.add_usize(CLEANUP_RETAINED, retained);
            if error {
                recorder.inc(CLEANUP_ERRORS);
            }
            recorder.record_timer(LIFECYCLE_WORK_NS, started);
        });
    }

    pub(crate) fn record_drop(&self, started: ProfileTimer, datasets: usize) {
        self.record(|recorder| {
            recorder.inc(DROP_CALLS);
            recorder.add_usize(DROP_DATASETS, datasets);
            recorder.record_timer(LIFECYCLE_WORK_NS, started);
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_access(
        &self,
        started: ProfileTimer,
        kind: DataBankAccessKind,
        cells: usize,
        genes: usize,
        output_elements: usize,
        output_bytes: usize,
        error: bool,
    ) {
        self.record(|recorder| {
            recorder.inc(ACCESS_CALLS);
            record_access_kind(&recorder, kind);
            recorder.add_usize(ACCESS_CELLS, cells);
            recorder.add_usize(ACCESS_GENES, genes);
            recorder.add_usize(ACCESS_OUTPUT_ELEMENTS, output_elements);
            recorder.add_usize(ACCESS_OUTPUT_BYTES, output_bytes);
            if error {
                recorder.inc(ACCESS_ERRORS);
            }
            recorder.record_timer(ACCESS_WORK_NS, started);
        });
    }

    pub(crate) fn record_prefetch(&self, started: ProfileTimer, cells: usize, error: bool) {
        self.record(|recorder| {
            recorder.inc(PREFETCH_CALLS);
            recorder.add_usize(PREFETCH_CELLS, cells);
            if error {
                recorder.inc(PREFETCH_ERRORS);
            }
            recorder.record_timer(PREFETCH_WORK_NS, started);
        });
    }

    pub(crate) fn record_scheduled(
        &self,
        started: ProfileTimer,
        kind: DataBankScheduledKind,
        datasets: usize,
        error: bool,
    ) {
        self.record(|recorder| {
            recorder.inc(SCHEDULED_CALLS);
            recorder.inc(scheduled_kind_metric(kind));
            recorder.add_usize(SCHEDULED_DATASETS, datasets);
            if error {
                recorder.inc(SCHEDULED_ERRORS);
            }
            recorder.record_timer(SCHEDULED_WORK_NS, started);
        });
    }

    fn with_auto_round(runtime: ProfileRuntime, auto_round: Option<ProfileOwnedRound>) -> Self {
        Self {
            inner: Arc::new(DataBankProfileInner {
                runtime,
                _auto_round: auto_round,
            }),
        }
    }

    fn timer(&self, scope: crate::profile::ProfileScopeId) -> ProfileTimer {
        self.inner
            .runtime
            .with_recorder(|recorder| recorder.timer(scope))
            .unwrap_or_else(ProfileTimer::disabled)
    }

    fn record(&self, f: impl FnOnce(ProfileRecorder<'_>)) {
        let _ = self.inner.runtime.with_recorder(f);
    }
}

pub(crate) fn databank_profile_registry() -> ProfileRegistry {
    ProfileRegistry::new()
        .with_component(ProfileComponent::new(DATABANK_COMPONENT).described("DataBank facade"))
        .with_scope(
            ProfileScope::new(DATABANK_LIFECYCLE_SCOPE)
                .kind(ProfileScopeKind::Event)
                .described("dataset registration and lifecycle work"),
        )
        .with_scope(
            ProfileScope::new(DATABANK_ACCESS_SCOPE)
                .kind(ProfileScopeKind::Timer)
                .described("DataBank access facade calls"),
        )
        .with_scope(
            ProfileScope::new(DATABANK_PREFETCH_SCOPE)
                .kind(ProfileScopeKind::Timer)
                .described("DataBank direct prefetch facade calls"),
        )
        .with_scope(
            ProfileScope::new(DATABANK_SCHEDULED_API_SCOPE)
                .kind(ProfileScopeKind::Event)
                .described("scheduled prefetch API construction"),
        )
}

fn register_kind_metric(kind: DataBankRegisterKind) -> ProfileMetricId {
    match kind {
        DataBankRegisterKind::Dense1D => REGISTER_DENSE_1D,
        DataBankRegisterKind::Dense2D => REGISTER_DENSE_2D,
        DataBankRegisterKind::SparseCsr => REGISTER_SPARSE_CSR,
    }
}

fn record_access_kind(recorder: &ProfileRecorder<'_>, kind: DataBankAccessKind) {
    match kind {
        DataBankAccessKind::Borrowed => recorder.inc(ACCESS_BORROWED_CALLS),
        DataBankAccessKind::ByGeneNames => {
            recorder.inc(ACCESS_BORROWED_CALLS);
            recorder.inc(ACCESS_BY_GENE_NAMES_CALLS);
        }
        DataBankAccessKind::Owned => recorder.inc(ACCESS_OWNED_CALLS),
        DataBankAccessKind::OwnedByGeneNames => {
            recorder.inc(ACCESS_OWNED_CALLS);
            recorder.inc(ACCESS_BY_GENE_NAMES_CALLS);
        }
        DataBankAccessKind::Unchecked => recorder.inc(ACCESS_UNCHECKED_CALLS),
    }
}

fn scheduled_kind_metric(kind: DataBankScheduledKind) -> ProfileMetricId {
    match kind {
        DataBankScheduledKind::Single => SCHEDULED_SINGLE_CALLS,
        DataBankScheduledKind::SingleByGeneNames => SCHEDULED_SINGLE_BY_GENE_NAMES_CALLS,
        DataBankScheduledKind::Multi => SCHEDULED_MULTI_CALLS,
        DataBankScheduledKind::MultiByGeneNames => SCHEDULED_MULTI_BY_GENE_NAMES_CALLS,
    }
}

#[allow(dead_code)]
pub(crate) fn format_metric(metric: &ProfileMetricSnapshot) -> Option<String> {
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

#[cfg(all(test, feature = "profile"))]
pub(crate) mod test_metrics {
    #![allow(dead_code)]

    use super::*;

    pub(crate) const REGISTER_CALLS: ProfileMetricId = super::REGISTER_CALLS;
    pub(crate) const ACCESS_CALLS: ProfileMetricId = super::ACCESS_CALLS;
    pub(crate) const ACCESS_BY_GENE_NAMES_CALLS: ProfileMetricId =
        super::ACCESS_BY_GENE_NAMES_CALLS;
    pub(crate) const ACCESS_OUTPUT_ELEMENTS: ProfileMetricId = super::ACCESS_OUTPUT_ELEMENTS;
    pub(crate) const PREFETCH_CALLS: ProfileMetricId = super::PREFETCH_CALLS;
    pub(crate) const SCHEDULED_CALLS: ProfileMetricId = super::SCHEDULED_CALLS;
}
