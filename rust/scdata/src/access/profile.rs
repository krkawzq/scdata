use std::sync::{Arc, Mutex};

use crate::profile::{
    ProfileComponent, ProfileConfig, ProfileMetricId, ProfileOwnedRound, ProfileRecorder,
    ProfileRegistry, ProfileRound, ProfileRuntime, ProfileScope, ProfileScopeId, ProfileScopeKind,
    ProfileSnapshot, ProfileTimer,
};

crate::scdata_profile_component!(pub const ACCESS_COMPONENT = "access");

crate::scdata_profile_scope!(pub const ACCESS_COMMAND_SCOPE = ACCESS_COMPONENT, "command");
crate::scdata_profile_scope!(pub const ACCESS_CACHE_SCOPE = ACCESS_COMPONENT, "cache");
crate::scdata_profile_scope!(pub const ACCESS_INFLIGHT_SCOPE = ACCESS_COMPONENT, "inflight");
crate::scdata_profile_scope!(pub const ACCESS_DECODE_SCOPE = ACCESS_COMPONENT, "decode");
crate::scdata_profile_scope!(pub const ACCESS_IO_SCOPE = ACCESS_COMPONENT, "io");
crate::scdata_profile_scope!(pub const ACCESS_MATERIALIZE_SCOPE = ACCESS_COMPONENT, "materialize");
crate::scdata_profile_scope!(pub const ACCESS_RESERVE_SCOPE = ACCESS_COMPONENT, "reserve");
crate::scdata_profile_scope!(pub const ACCESS_SCHEDULED_SCOPE = ACCESS_COMPONENT, "scheduled");

const COMMANDS: ProfileMetricId = ProfileMetricId::count(ACCESS_COMMAND_SCOPE, "commands");
const READ_COMMANDS: ProfileMetricId = ProfileMetricId::count(ACCESS_COMMAND_SCOPE, "read");
const PREFETCH_COMMANDS: ProfileMetricId = ProfileMetricId::count(ACCESS_COMMAND_SCOPE, "prefetch");
const SCHEDULED_DECODE_COMMANDS: ProfileMetricId =
    ProfileMetricId::count(ACCESS_COMMAND_SCOPE, "scheduled-decode");
const SCHEDULED_ENSURE_READY_COMMANDS: ProfileMetricId =
    ProfileMetricId::count(ACCESS_COMMAND_SCOPE, "scheduled-ensure-ready");
const SCHEDULED_TAKE_COMMANDS: ProfileMetricId =
    ProfileMetricId::count(ACCESS_COMMAND_SCOPE, "scheduled-take");
const SCHEDULED_CANCEL_COMMANDS: ProfileMetricId =
    ProfileMetricId::count(ACCESS_COMMAND_SCOPE, "scheduled-cancel");
const COMMAND_QUEUE_WAIT_NS: ProfileMetricId =
    ProfileMetricId::duration(ACCESS_COMMAND_SCOPE, "queue-wait");
const COMMAND_REJECTIONS: ProfileMetricId =
    ProfileMetricId::count(ACCESS_COMMAND_SCOPE, "rejections");
const COMMAND_QUEUE_FULL: ProfileMetricId =
    ProfileMetricId::count(ACCESS_COMMAND_SCOPE, "queue-full");
const COMMAND_SHUTDOWN: ProfileMetricId = ProfileMetricId::count(ACCESS_COMMAND_SCOPE, "shutdown");

const CACHE_HITS: ProfileMetricId = ProfileMetricId::count(ACCESS_CACHE_SCOPE, "hits");
const RAW_CACHE_HITS: ProfileMetricId = ProfileMetricId::count(ACCESS_CACHE_SCOPE, "raw-hits");
const DECODED_CACHE_HITS: ProfileMetricId =
    ProfileMetricId::count(ACCESS_CACHE_SCOPE, "decoded-hits");
const CACHE_MISSES: ProfileMetricId = ProfileMetricId::count(ACCESS_CACHE_SCOPE, "misses");
const CACHE_INSERTS: ProfileMetricId = ProfileMetricId::count(ACCESS_CACHE_SCOPE, "inserts");
const CACHE_REPLACEMENTS: ProfileMetricId =
    ProfileMetricId::count(ACCESS_CACHE_SCOPE, "replacements");
const CACHE_INSERT_FAILURES: ProfileMetricId =
    ProfileMetricId::count(ACCESS_CACHE_SCOPE, "insert-failures");
const CACHE_TOO_LARGE: ProfileMetricId = ProfileMetricId::count(ACCESS_CACHE_SCOPE, "too-large");
const CACHE_ALL_PINNED: ProfileMetricId = ProfileMetricId::count(ACCESS_CACHE_SCOPE, "all-pinned");
const CACHE_EVICTIONS: ProfileMetricId = ProfileMetricId::count(ACCESS_CACHE_SCOPE, "evictions");
const CACHE_EVICTED_BYTES: ProfileMetricId = ProfileMetricId::bytes(ACCESS_CACHE_SCOPE, "evicted");
const CACHE_REPLACED_BYTES: ProfileMetricId =
    ProfileMetricId::bytes(ACCESS_CACHE_SCOPE, "replaced");
const CACHE_UNCACHED_FALLBACKS: ProfileMetricId =
    ProfileMetricId::count(ACCESS_CACHE_SCOPE, "uncached-fallbacks");

const INFLIGHT_FIRST: ProfileMetricId = ProfileMetricId::count(ACCESS_INFLIGHT_SCOPE, "first");
const INFLIGHT_WAITS: ProfileMetricId = ProfileMetricId::count(ACCESS_INFLIGHT_SCOPE, "waits");
const INFLIGHT_WAIT_NS: ProfileMetricId = ProfileMetricId::duration(ACCESS_INFLIGHT_SCOPE, "wait");

const DECODE_FIRST: ProfileMetricId = ProfileMetricId::count(ACCESS_DECODE_SCOPE, "first");
const DECODE_WAITS: ProfileMetricId = ProfileMetricId::count(ACCESS_DECODE_SCOPE, "waits");
const DECODE_WAIT_NS: ProfileMetricId = ProfileMetricId::duration(ACCESS_DECODE_SCOPE, "wait");
const DECODE_CACHED: ProfileMetricId = ProfileMetricId::count(ACCESS_DECODE_SCOPE, "cached");
const IDENTITY_DECODE: ProfileMetricId = ProfileMetricId::count(ACCESS_DECODE_SCOPE, "identity");
const DECODE_CALLS: ProfileMetricId = ProfileMetricId::count(ACCESS_DECODE_SCOPE, "calls");
const DECODE_NS: ProfileMetricId = ProfileMetricId::duration(ACCESS_DECODE_SCOPE, "work");
const DECODE_ENCODED_BYTES: ProfileMetricId =
    ProfileMetricId::bytes(ACCESS_DECODE_SCOPE, "encoded");
const DECODE_DECODED_BYTES: ProfileMetricId =
    ProfileMetricId::bytes(ACCESS_DECODE_SCOPE, "decoded");
const DECODE_ERRORS: ProfileMetricId = ProfileMetricId::count(ACCESS_DECODE_SCOPE, "errors");

const IO_READS: ProfileMetricId = ProfileMetricId::count(ACCESS_IO_SCOPE, "reads");
const IO_READ_NS: ProfileMetricId = ProfileMetricId::duration(ACCESS_IO_SCOPE, "read");
const IO_REQUESTED_BYTES: ProfileMetricId = ProfileMetricId::bytes(ACCESS_IO_SCOPE, "requested");
const IO_READ_BYTES: ProfileMetricId = ProfileMetricId::bytes(ACCESS_IO_SCOPE, "read");
const IO_ERRORS: ProfileMetricId = ProfileMetricId::count(ACCESS_IO_SCOPE, "errors");

const MATERIALIZE_CALLS: ProfileMetricId =
    ProfileMetricId::count(ACCESS_MATERIALIZE_SCOPE, "calls");
const MATERIALIZE_NS: ProfileMetricId = ProfileMetricId::duration(ACCESS_MATERIALIZE_SCOPE, "work");
const MATERIALIZE_BYTES: ProfileMetricId =
    ProfileMetricId::bytes(ACCESS_MATERIALIZE_SCOPE, "output");
const MATERIALIZE_ERRORS: ProfileMetricId =
    ProfileMetricId::count(ACCESS_MATERIALIZE_SCOPE, "errors");

const RESERVE_CALLS: ProfileMetricId = ProfileMetricId::count(ACCESS_RESERVE_SCOPE, "calls");
const RESERVE_NS: ProfileMetricId = ProfileMetricId::duration(ACCESS_RESERVE_SCOPE, "wait");
const RESERVE_BYTES: ProfileMetricId = ProfileMetricId::bytes(ACCESS_RESERVE_SCOPE, "bytes");
const RESERVE_FAILURES: ProfileMetricId = ProfileMetricId::count(ACCESS_RESERVE_SCOPE, "failures");

const SCHEDULED_CANCELLED: ProfileMetricId =
    ProfileMetricId::count(ACCESS_SCHEDULED_SCOPE, "cancelled");
const SCHEDULED_STAGED_BYTES: ProfileMetricId =
    ProfileMetricId::bytes(ACCESS_SCHEDULED_SCOPE, "staged");
const SCHEDULED_EVICTIONS: ProfileMetricId =
    ProfileMetricId::count(ACCESS_SCHEDULED_SCOPE, "evictions");
const SCHEDULED_EVICTED_BYTES: ProfileMetricId =
    ProfileMetricId::bytes(ACCESS_SCHEDULED_SCOPE, "evicted");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AccessCommandKind {
    Read,
    Prefetch,
    ScheduledDecode,
    ScheduledEnsureReady,
    ScheduledTake,
    ScheduledCancel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AccessCacheHitKind {
    Raw,
    Decoded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AccessCommandRejectKind {
    QueueFull,
    Shutdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AccessCacheInsertFailureKind {
    TooLarge,
    AllPinned,
}

#[derive(Debug, Clone)]
pub struct AccessProfile {
    inner: Arc<AccessProfileInner>,
}

#[derive(Debug)]
struct AccessProfileInner {
    runtime: ProfileRuntime,
    auto_round: Mutex<Option<ProfileOwnedRound>>,
}

impl Default for AccessProfile {
    fn default() -> Self {
        Self::disabled()
    }
}

impl AccessProfile {
    fn new(runtime: ProfileRuntime, auto_round: Option<ProfileOwnedRound>) -> Self {
        Self {
            inner: Arc::new(AccessProfileInner {
                runtime,
                auto_round: Mutex::new(auto_round),
            }),
        }
    }

    pub fn disabled() -> Self {
        Self::new(
            ProfileRuntime::new_lazy(ProfileConfig::disabled(), access_profile_registry),
            None,
        )
    }

    pub fn enabled(label: impl Into<String>) -> Self {
        Self::new(
            ProfileRuntime::enabled_lazy(label, access_profile_registry),
            None,
        )
    }

    pub fn from_env() -> Self {
        let runtime = ProfileRuntime::from_env_lazy(access_profile_registry);
        let auto_round = runtime.is_global_enabled().then(|| runtime.start_owned());
        Self::new(runtime, auto_round)
    }

    pub fn from_runtime(runtime: ProfileRuntime) -> Self {
        runtime.ensure_registry_lazy(access_profile_registry);
        Self::new(runtime, None)
    }

    pub fn runtime(&self) -> &ProfileRuntime {
        &self.inner.runtime
    }

    pub fn into_runtime(self) -> ProfileRuntime {
        self.inner.runtime.clone()
    }

    pub fn start(&self) -> ProfileRound<'_> {
        self.clear_auto_round();
        self.inner.runtime.start()
    }

    pub fn snapshot(&self) -> ProfileSnapshot {
        self.inner.runtime.snapshot()
    }

    pub fn snapshot_and_reset(&self) -> ProfileSnapshot {
        self.inner.runtime.snapshot_and_reset()
    }

    pub fn reset_metrics(&self) {
        self.inner.runtime.reset_metrics();
    }

    pub(crate) fn command_timer(&self) -> ProfileTimer {
        self.timer(ACCESS_COMMAND_SCOPE)
    }

    pub(crate) fn timer(&self, scope: ProfileScopeId) -> ProfileTimer {
        self.with_recorder(|ctx| ctx.timer(scope))
            .unwrap_or_else(ProfileTimer::disabled)
    }

    pub(crate) fn record_command(&self, timer: ProfileTimer, kind: AccessCommandKind) {
        let _ = self.with_recorder(|ctx| {
            ctx.inc(COMMANDS);
            ctx.record_timer(COMMAND_QUEUE_WAIT_NS, timer);
            ctx.inc(match kind {
                AccessCommandKind::Read => READ_COMMANDS,
                AccessCommandKind::Prefetch => PREFETCH_COMMANDS,
                AccessCommandKind::ScheduledDecode => SCHEDULED_DECODE_COMMANDS,
                AccessCommandKind::ScheduledEnsureReady => SCHEDULED_ENSURE_READY_COMMANDS,
                AccessCommandKind::ScheduledTake => SCHEDULED_TAKE_COMMANDS,
                AccessCommandKind::ScheduledCancel => SCHEDULED_CANCEL_COMMANDS,
            });
        });
    }

    pub(crate) fn record_command_rejected(&self, kind: AccessCommandRejectKind) {
        let _ = self.with_recorder(|ctx| {
            ctx.inc(COMMAND_REJECTIONS);
            ctx.inc(match kind {
                AccessCommandRejectKind::QueueFull => COMMAND_QUEUE_FULL,
                AccessCommandRejectKind::Shutdown => COMMAND_SHUTDOWN,
            });
        });
    }

    pub(crate) fn record_cache_hit(&self, kind: AccessCacheHitKind) {
        let _ = self.with_recorder(|ctx| {
            ctx.inc(CACHE_HITS);
            ctx.inc(match kind {
                AccessCacheHitKind::Raw => RAW_CACHE_HITS,
                AccessCacheHitKind::Decoded => DECODED_CACHE_HITS,
            });
        });
    }

    pub(crate) fn record_cache_miss(&self) {
        self.inc(CACHE_MISSES);
    }

    pub(crate) fn record_load(
        &self,
        cache_hit: Option<AccessCacheHitKind>,
        cache_miss: bool,
        inflight_first: bool,
    ) {
        let _ = self.with_recorder(|ctx| {
            if let Some(kind) = cache_hit {
                ctx.inc(CACHE_HITS);
                ctx.inc(match kind {
                    AccessCacheHitKind::Raw => RAW_CACHE_HITS,
                    AccessCacheHitKind::Decoded => DECODED_CACHE_HITS,
                });
            }
            if cache_miss {
                ctx.inc(CACHE_MISSES);
            }
            if inflight_first {
                ctx.inc(INFLIGHT_FIRST);
            }
        });
    }

    pub(crate) fn record_cache_insert(
        &self,
        evicted_count: usize,
        evicted_bytes: usize,
        replaced_bytes: usize,
    ) {
        let _ = self.with_recorder(|ctx| {
            ctx.inc(CACHE_INSERTS);
            if replaced_bytes > 0 {
                ctx.inc(CACHE_REPLACEMENTS);
                ctx.add_usize(CACHE_REPLACED_BYTES, replaced_bytes);
            }
            record_eviction(&ctx, evicted_count, evicted_bytes);
        });
    }

    pub(crate) fn record_cache_insert_failure(
        &self,
        kind: AccessCacheInsertFailureKind,
        evicted_count: usize,
        evicted_bytes: usize,
    ) {
        let _ = self.with_recorder(|ctx| {
            ctx.inc(CACHE_INSERT_FAILURES);
            ctx.inc(match kind {
                AccessCacheInsertFailureKind::TooLarge => CACHE_TOO_LARGE,
                AccessCacheInsertFailureKind::AllPinned => CACHE_ALL_PINNED,
            });
            record_eviction(&ctx, evicted_count, evicted_bytes);
        });
    }

    pub(crate) fn record_cache_too_large(&self) {
        self.inc(CACHE_TOO_LARGE);
    }

    pub(crate) fn record_cache_eviction(&self, count: usize, bytes: usize) {
        let _ = self.with_recorder(|ctx| record_eviction(&ctx, count, bytes));
    }

    pub(crate) fn record_uncached_fallback(&self) {
        self.inc(CACHE_UNCACHED_FALLBACKS);
    }

    pub(crate) fn record_inflight_wait(&self, timer: ProfileTimer) {
        let _ = self.with_recorder(|ctx| {
            ctx.inc(INFLIGHT_WAITS);
            ctx.record_timer(INFLIGHT_WAIT_NS, timer);
        });
    }

    pub(crate) fn record_decode_first(&self) {
        self.inc(DECODE_FIRST);
    }

    pub(crate) fn record_decode_waiter(&self, timer: ProfileTimer) {
        let _ = self.with_recorder(|ctx| {
            ctx.inc(DECODE_WAITS);
            ctx.record_timer(DECODE_WAIT_NS, timer);
        });
    }

    pub(crate) fn record_decode_cached(&self) {
        self.inc(DECODE_CACHED);
    }

    pub(crate) fn record_identity_decode(&self) {
        self.inc(IDENTITY_DECODE);
    }

    pub(crate) fn record_decode(
        &self,
        timer: ProfileTimer,
        encoded_bytes: usize,
        decoded_bytes: Option<usize>,
        error: bool,
    ) {
        let _ = self.with_recorder(|ctx| {
            ctx.inc(DECODE_CALLS);
            ctx.record_timer(DECODE_NS, timer);
            ctx.add_usize(DECODE_ENCODED_BYTES, encoded_bytes);
            if let Some(decoded_bytes) = decoded_bytes {
                ctx.add_usize(DECODE_DECODED_BYTES, decoded_bytes);
            }
            if error {
                ctx.inc(DECODE_ERRORS);
            }
        });
    }

    pub(crate) fn record_io_read(
        &self,
        timer: ProfileTimer,
        requested_bytes: usize,
        read_bytes: Option<usize>,
        error: bool,
    ) {
        let _ = self.with_recorder(|ctx| {
            ctx.inc(IO_READS);
            ctx.record_timer(IO_READ_NS, timer);
            ctx.add_usize(IO_REQUESTED_BYTES, requested_bytes);
            if let Some(read_bytes) = read_bytes {
                ctx.add_usize(IO_READ_BYTES, read_bytes);
            }
            if error {
                ctx.inc(IO_ERRORS);
            }
        });
    }

    pub(crate) fn record_materialize(&self, timer: ProfileTimer, bytes: usize, error: bool) {
        let _ = self.with_recorder(|ctx| {
            ctx.inc(MATERIALIZE_CALLS);
            ctx.record_timer(MATERIALIZE_NS, timer);
            ctx.add_usize(MATERIALIZE_BYTES, bytes);
            if error {
                ctx.inc(MATERIALIZE_ERRORS);
            }
        });
    }

    pub(crate) fn record_reserve(&self, timer: ProfileTimer, bytes: usize, success: bool) {
        let _ = self.with_recorder(|ctx| {
            ctx.inc(RESERVE_CALLS);
            ctx.record_timer(RESERVE_NS, timer);
            ctx.add_usize(RESERVE_BYTES, bytes);
            if !success {
                ctx.inc(RESERVE_FAILURES);
            }
        });
    }

    pub(crate) fn record_scheduled_cancelled(&self) {
        self.inc(SCHEDULED_CANCELLED);
    }

    pub(crate) fn record_staged_bytes(&self, bytes: usize) {
        self.add_usize(SCHEDULED_STAGED_BYTES, bytes);
    }

    pub(crate) fn record_staged_eviction(&self, bytes: usize) {
        let _ = self.with_recorder(|ctx| {
            ctx.inc(SCHEDULED_EVICTIONS);
            ctx.add_usize(SCHEDULED_EVICTED_BYTES, bytes);
        });
    }

    fn with_recorder<R>(&self, f: impl FnOnce(ProfileRecorder<'_>) -> R) -> Option<R> {
        self.inner.runtime.with_recorder(f)
    }

    fn inc(&self, metric: ProfileMetricId) {
        let _ = self.with_recorder(|ctx| ctx.inc(metric));
    }

    fn add_usize(&self, metric: ProfileMetricId, value: usize) {
        let _ = self.with_recorder(|ctx| ctx.add_usize(metric, value));
    }

    fn clear_auto_round(&self) {
        let auto_round = self
            .inner
            .auto_round
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        #[cfg(feature = "profile")]
        drop(auto_round);
        #[cfg(not(feature = "profile"))]
        let _ = auto_round;
    }
}

fn record_eviction(ctx: &ProfileRecorder<'_>, count: usize, bytes: usize) {
    if count > 0 {
        ctx.add_usize(CACHE_EVICTIONS, count);
    }
    if bytes > 0 {
        ctx.add_usize(CACHE_EVICTED_BYTES, bytes);
    }
}

pub fn access_profile_registry() -> ProfileRegistry {
    ProfileRegistry::new()
        .with_component(
            ProfileComponent::new(ACCESS_COMPONENT).described("access scheduler runtime"),
        )
        .with_scope(
            ProfileScope::new(ACCESS_COMMAND_SCOPE)
                .kind(ProfileScopeKind::Event)
                .described("submitted access scheduler commands"),
        )
        .with_scope(
            ProfileScope::new(ACCESS_CACHE_SCOPE)
                .kind(ProfileScopeKind::Counter)
                .described("raw and decoded cache lookups"),
        )
        .with_scope(
            ProfileScope::new(ACCESS_INFLIGHT_SCOPE)
                .kind(ProfileScopeKind::Event)
                .described("shared raw read in-flight coordination"),
        )
        .with_scope(
            ProfileScope::new(ACCESS_DECODE_SCOPE)
                .kind(ProfileScopeKind::Event)
                .described("access-side decode decisions and waits"),
        )
        .with_scope(
            ProfileScope::new(ACCESS_IO_SCOPE)
                .kind(ProfileScopeKind::Bytes)
                .described("raw chunk IO requested by access"),
        )
        .with_scope(
            ProfileScope::new(ACCESS_MATERIALIZE_SCOPE)
                .kind(ProfileScopeKind::Timer)
                .described("decoded output materialization"),
        )
        .with_scope(
            ProfileScope::new(ACCESS_RESERVE_SCOPE)
                .kind(ProfileScopeKind::Timer)
                .described("memory budget reservations"),
        )
        .with_scope(
            ProfileScope::new(ACCESS_SCHEDULED_SCOPE)
                .kind(ProfileScopeKind::Event)
                .described("scheduled-access staging and cancellation"),
        )
}

#[cfg(all(test, feature = "profile"))]
pub(crate) mod test_metrics {
    #![allow(dead_code)]

    use super::*;

    pub(crate) const COMMANDS: ProfileMetricId = super::COMMANDS;
    pub(crate) const READ_COMMANDS: ProfileMetricId = super::READ_COMMANDS;
    pub(crate) const COMMAND_REJECTIONS: ProfileMetricId = super::COMMAND_REJECTIONS;
    pub(crate) const COMMAND_QUEUE_FULL: ProfileMetricId = super::COMMAND_QUEUE_FULL;
    pub(crate) const CACHE_HITS: ProfileMetricId = super::CACHE_HITS;
    pub(crate) const RAW_CACHE_HITS: ProfileMetricId = super::RAW_CACHE_HITS;
    pub(crate) const CACHE_MISSES: ProfileMetricId = super::CACHE_MISSES;
    pub(crate) const CACHE_INSERTS: ProfileMetricId = super::CACHE_INSERTS;
    pub(crate) const CACHE_REPLACEMENTS: ProfileMetricId = super::CACHE_REPLACEMENTS;
    pub(crate) const CACHE_INSERT_FAILURES: ProfileMetricId = super::CACHE_INSERT_FAILURES;
    pub(crate) const CACHE_TOO_LARGE: ProfileMetricId = super::CACHE_TOO_LARGE;
    pub(crate) const CACHE_EVICTIONS: ProfileMetricId = super::CACHE_EVICTIONS;
    pub(crate) const CACHE_EVICTED_BYTES: ProfileMetricId = super::CACHE_EVICTED_BYTES;
    pub(crate) const CACHE_REPLACED_BYTES: ProfileMetricId = super::CACHE_REPLACED_BYTES;
    pub(crate) const CACHE_UNCACHED_FALLBACKS: ProfileMetricId = super::CACHE_UNCACHED_FALLBACKS;
    pub(crate) const INFLIGHT_FIRST: ProfileMetricId = super::INFLIGHT_FIRST;
    pub(crate) const IO_READS: ProfileMetricId = super::IO_READS;
    pub(crate) const IO_REQUESTED_BYTES: ProfileMetricId = super::IO_REQUESTED_BYTES;
    pub(crate) const IO_READ_BYTES: ProfileMetricId = super::IO_READ_BYTES;
    pub(crate) const IDENTITY_DECODE: ProfileMetricId = super::IDENTITY_DECODE;
    pub(crate) const MATERIALIZE_CALLS: ProfileMetricId = super::MATERIALIZE_CALLS;
    pub(crate) const MATERIALIZE_BYTES: ProfileMetricId = super::MATERIALIZE_BYTES;
    pub(crate) const SCHEDULED_EVICTIONS: ProfileMetricId = super::SCHEDULED_EVICTIONS;
    pub(crate) const SCHEDULED_EVICTED_BYTES: ProfileMetricId = super::SCHEDULED_EVICTED_BYTES;
}
