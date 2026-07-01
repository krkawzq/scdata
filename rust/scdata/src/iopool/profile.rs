use std::sync::Arc;

#[cfg(any(feature = "profile", test))]
use crate::profile::ProfileMetricId;
use crate::profile::{
    ProfileComponent, ProfileOwnedRound, ProfileRegistry, ProfileRuntime, ProfileScope,
    ProfileScopeKind, ProfileSnapshot, ProfileTimer,
};

crate::scdata_profile_component!(pub(crate) const IOPOOL_COMPONENT = "iopool");

crate::scdata_profile_scope!(pub(crate) const IOPOOL_SUBMIT_SCOPE = IOPOOL_COMPONENT, "submit");
crate::scdata_profile_scope!(pub(crate) const IOPOOL_QUEUE_SCOPE = IOPOOL_COMPONENT, "queue");
crate::scdata_profile_scope!(
    pub(crate) const IOPOOL_OPERATION_SCOPE = IOPOOL_COMPONENT,
    "operation"
);

#[cfg(any(feature = "profile", test))]
const SUBMIT_CALLS: ProfileMetricId = ProfileMetricId::count(IOPOOL_SUBMIT_SCOPE, "calls");
#[cfg(any(feature = "profile", test))]
const READ_SUBMITS: ProfileMetricId = ProfileMetricId::count(IOPOOL_SUBMIT_SCOPE, "read");
#[cfg(any(feature = "profile", test))]
const WRITE_SUBMITS: ProfileMetricId = ProfileMetricId::count(IOPOOL_SUBMIT_SCOPE, "write");
#[cfg(any(feature = "profile", test))]
const FSYNC_SUBMITS: ProfileMetricId = ProfileMetricId::count(IOPOOL_SUBMIT_SCOPE, "fsync");
#[cfg(any(feature = "profile", test))]
const SYNC_DATA_SUBMITS: ProfileMetricId = ProfileMetricId::count(IOPOOL_SUBMIT_SCOPE, "sync-data");
#[cfg(any(feature = "profile", test))]
const TRUNCATE_SUBMITS: ProfileMetricId = ProfileMetricId::count(IOPOOL_SUBMIT_SCOPE, "truncate");
#[cfg(any(feature = "profile", test))]
const METADATA_SUBMITS: ProfileMetricId = ProfileMetricId::count(IOPOOL_SUBMIT_SCOPE, "metadata");

#[cfg(any(feature = "profile", test))]
const IMMEDIATE_COMPLETIONS: ProfileMetricId =
    ProfileMetricId::count(IOPOOL_QUEUE_SCOPE, "immediate");
#[cfg(any(feature = "profile", test))]
const DEDUP_HITS: ProfileMetricId = ProfileMetricId::count(IOPOOL_QUEUE_SCOPE, "dedup-hits");
#[cfg(any(feature = "profile", test))]
const QUEUE_FULL: ProfileMetricId = ProfileMetricId::count(IOPOOL_QUEUE_SCOPE, "queue-full");
#[cfg(any(feature = "profile", test))]
const CANCELLED_BEFORE_DISPATCH: ProfileMetricId =
    ProfileMetricId::count(IOPOOL_QUEUE_SCOPE, "cancelled-before-dispatch");
#[cfg(any(feature = "profile", test))]
const DISPATCHED: ProfileMetricId = ProfileMetricId::count(IOPOOL_QUEUE_SCOPE, "dispatched");
#[cfg(any(feature = "profile", test))]
const DISPATCH_WAIT_NS: ProfileMetricId =
    ProfileMetricId::duration(IOPOOL_QUEUE_SCOPE, "dispatch-wait");

#[cfg(any(feature = "profile", test))]
const OPERATION_CALLS: ProfileMetricId = ProfileMetricId::count(IOPOOL_OPERATION_SCOPE, "calls");
#[cfg(any(feature = "profile", test))]
const OPERATION_NS: ProfileMetricId = ProfileMetricId::duration(IOPOOL_OPERATION_SCOPE, "work");
#[cfg(any(feature = "profile", test))]
const READ_BYTES: ProfileMetricId = ProfileMetricId::bytes(IOPOOL_OPERATION_SCOPE, "read");
#[cfg(any(feature = "profile", test))]
const WRITE_BYTES: ProfileMetricId = ProfileMetricId::bytes(IOPOOL_OPERATION_SCOPE, "write");
#[cfg(any(feature = "profile", test))]
const OPERATION_ERRORS: ProfileMetricId = ProfileMetricId::count(IOPOOL_OPERATION_SCOPE, "errors");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IoCommandKind {
    Read,
    Write,
    Fsync,
    SyncData,
    Truncate,
    Metadata,
}

#[derive(Debug, Clone)]
pub(crate) struct IoPoolProfile {
    inner: Arc<IoPoolProfileInner>,
}

#[derive(Debug)]
struct IoPoolProfileInner {
    runtime: ProfileRuntime,
    _auto_round: Option<ProfileOwnedRound>,
}

pub(crate) fn iopool_profile_registry() -> ProfileRegistry {
    ProfileRegistry::new()
        .with_component(ProfileComponent::new(IOPOOL_COMPONENT).described("IO execution pool"))
        .with_scope(
            ProfileScope::new(IOPOOL_SUBMIT_SCOPE)
                .kind(ProfileScopeKind::Event)
                .described("IO command submissions"),
        )
        .with_scope(
            ProfileScope::new(IOPOOL_QUEUE_SCOPE)
                .kind(ProfileScopeKind::Event)
                .described("IO queue coordination"),
        )
        .with_scope(
            ProfileScope::new(IOPOOL_OPERATION_SCOPE)
                .kind(ProfileScopeKind::Timer)
                .described("IO operation execution"),
        )
}

pub(crate) fn register_iopool_profile(profile: &ProfileRuntime) {
    profile.ensure_registry_lazy(iopool_profile_registry);
}

impl IoPoolProfile {
    #[cfg(test)]
    pub(crate) fn disabled() -> Self {
        Self::with_auto_round(ProfileRuntime::disabled(), None)
    }

    pub(crate) fn from_env() -> Self {
        let runtime = ProfileRuntime::from_env_lazy(iopool_profile_registry);
        let auto_round = runtime.is_global_enabled().then(|| runtime.start_owned());
        Self::with_auto_round(runtime, auto_round)
    }

    pub(crate) fn new(runtime: ProfileRuntime) -> Self {
        register_iopool_profile(&runtime);
        Self::with_auto_round(runtime, None)
    }

    pub(crate) fn runtime(&self) -> &ProfileRuntime {
        &self.inner.runtime
    }

    #[inline]
    pub(crate) fn is_recording(&self) -> bool {
        self.inner.runtime.is_recording()
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

    #[inline]
    pub(crate) fn queue_timer(&self) -> ProfileTimer {
        if !self.is_recording() {
            return ProfileTimer::disabled();
        }
        self.inner
            .runtime
            .with_recorder(|ctx| ctx.timer(IOPOOL_QUEUE_SCOPE))
            .unwrap_or_else(ProfileTimer::disabled)
    }

    #[inline]
    pub(crate) fn operation_timer(&self) -> ProfileTimer {
        if !self.is_recording() {
            return ProfileTimer::disabled();
        }
        self.inner
            .runtime
            .with_recorder(|ctx| ctx.timer(IOPOOL_OPERATION_SCOPE))
            .unwrap_or_else(ProfileTimer::disabled)
    }

    fn with_auto_round(runtime: ProfileRuntime, auto_round: Option<ProfileOwnedRound>) -> Self {
        Self {
            inner: Arc::new(IoPoolProfileInner {
                runtime,
                _auto_round: auto_round,
            }),
        }
    }
}

#[inline]
pub(crate) fn record_iopool_submit(profile: &IoPoolProfile, kind: IoCommandKind) {
    #[cfg(not(feature = "profile"))]
    let _ = kind;

    if !profile.is_recording() {
        return;
    }
    crate::scdata_profile_record!(profile.runtime(), |ctx| {
        ctx.inc(SUBMIT_CALLS);
        ctx.inc(command_submit_metric(kind));
    });
}

#[inline]
pub(crate) fn record_iopool_immediate(profile: &IoPoolProfile) {
    if !profile.is_recording() {
        return;
    }
    crate::scdata_profile_record!(profile.runtime(), |ctx| {
        ctx.inc(IMMEDIATE_COMPLETIONS);
    });
}

#[inline]
pub(crate) fn record_iopool_dedup_hit(profile: &IoPoolProfile) {
    if !profile.is_recording() {
        return;
    }
    crate::scdata_profile_record!(profile.runtime(), |ctx| {
        ctx.inc(DEDUP_HITS);
    });
}

#[inline]
pub(crate) fn record_iopool_queue_full(profile: &IoPoolProfile) {
    if !profile.is_recording() {
        return;
    }
    crate::scdata_profile_record!(profile.runtime(), |ctx| {
        ctx.inc(QUEUE_FULL);
    });
}

#[inline]
pub(crate) fn record_iopool_cancelled_before_dispatch(profile: &IoPoolProfile) {
    if !profile.is_recording() {
        return;
    }
    crate::scdata_profile_record!(profile.runtime(), |ctx| {
        ctx.inc(CANCELLED_BEFORE_DISPATCH);
    });
}

#[inline]
pub(crate) fn record_iopool_dispatched(profile: &IoPoolProfile, queued_at: ProfileTimer) {
    #[cfg(not(feature = "profile"))]
    let _ = queued_at;

    if !profile.is_recording() {
        return;
    }
    crate::scdata_profile_record!(profile.runtime(), |ctx| {
        ctx.inc(DISPATCHED);
        ctx.record_timer(DISPATCH_WAIT_NS, queued_at);
    });
}

#[inline]
pub(crate) fn record_iopool_operation(
    profile: &IoPoolProfile,
    started: ProfileTimer,
    read_bytes: usize,
    write_bytes: usize,
    error: bool,
) {
    #[cfg(not(feature = "profile"))]
    let _ = (started, read_bytes, write_bytes, error);

    if !profile.is_recording() {
        return;
    }
    crate::scdata_profile_record!(profile.runtime(), |ctx| {
        ctx.inc(OPERATION_CALLS);
        ctx.record_timer(OPERATION_NS, started);
        ctx.add_usize(READ_BYTES, read_bytes);
        ctx.add_usize(WRITE_BYTES, write_bytes);
        if error {
            ctx.inc(OPERATION_ERRORS);
        }
    });
}

#[cfg(feature = "profile")]
#[inline]
fn command_submit_metric(kind: IoCommandKind) -> ProfileMetricId {
    match kind {
        IoCommandKind::Read => READ_SUBMITS,
        IoCommandKind::Write => WRITE_SUBMITS,
        IoCommandKind::Fsync => FSYNC_SUBMITS,
        IoCommandKind::SyncData => SYNC_DATA_SUBMITS,
        IoCommandKind::Truncate => TRUNCATE_SUBMITS,
        IoCommandKind::Metadata => METADATA_SUBMITS,
    }
}

#[cfg(test)]
pub(crate) mod test_metrics {
    use super::*;

    pub(crate) const SUBMIT_CALLS: ProfileMetricId = super::SUBMIT_CALLS;
    pub(crate) const READ_SUBMITS: ProfileMetricId = super::READ_SUBMITS;
    pub(crate) const WRITE_SUBMITS: ProfileMetricId = super::WRITE_SUBMITS;
    pub(crate) const FSYNC_SUBMITS: ProfileMetricId = super::FSYNC_SUBMITS;
    pub(crate) const SYNC_DATA_SUBMITS: ProfileMetricId = super::SYNC_DATA_SUBMITS;
    pub(crate) const TRUNCATE_SUBMITS: ProfileMetricId = super::TRUNCATE_SUBMITS;
    pub(crate) const METADATA_SUBMITS: ProfileMetricId = super::METADATA_SUBMITS;
    pub(crate) const IMMEDIATE_COMPLETIONS: ProfileMetricId = super::IMMEDIATE_COMPLETIONS;
    pub(crate) const DEDUP_HITS: ProfileMetricId = super::DEDUP_HITS;
    pub(crate) const QUEUE_FULL: ProfileMetricId = super::QUEUE_FULL;
    pub(crate) const CANCELLED_BEFORE_DISPATCH: ProfileMetricId = super::CANCELLED_BEFORE_DISPATCH;
    pub(crate) const DISPATCHED: ProfileMetricId = super::DISPATCHED;
    pub(crate) const OPERATION_CALLS: ProfileMetricId = super::OPERATION_CALLS;
    pub(crate) const READ_BYTES: ProfileMetricId = super::READ_BYTES;
    pub(crate) const WRITE_BYTES: ProfileMetricId = super::WRITE_BYTES;
    pub(crate) const OPERATION_ERRORS: ProfileMetricId = super::OPERATION_ERRORS;
}
