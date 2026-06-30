use std::sync::Arc;

use crate::profile::{
    ProfileComponent, ProfileConfig, ProfileMetricId, ProfileOwnedRound, ProfileRecorder,
    ProfileRegistry, ProfileRound, ProfileRuntime, ProfileScope, ProfileScopeKind, ProfileSnapshot,
    ProfileTimer,
};

crate::scdata_profile_component!(pub const CODECS_COMPONENT = "codecs");

crate::scdata_profile_scope!(pub const CODECS_SUBMIT_SCOPE = CODECS_COMPONENT, "submit");
crate::scdata_profile_scope!(pub const CODECS_WORK_SCOPE = CODECS_COMPONENT, "work");

const SUBMIT_CALLS: ProfileMetricId = ProfileMetricId::count(CODECS_SUBMIT_SCOPE, "calls");
const SUBMIT_BLOCKING_CALLS: ProfileMetricId =
    ProfileMetricId::count(CODECS_SUBMIT_SCOPE, "blocking");
const SUBMIT_TRY_CALLS: ProfileMetricId = ProfileMetricId::count(CODECS_SUBMIT_SCOPE, "try");
const SUBMIT_ASYNC_CALLS: ProfileMetricId = ProfileMetricId::count(CODECS_SUBMIT_SCOPE, "async");
const SUBMIT_ERRORS: ProfileMetricId = ProfileMetricId::count(CODECS_SUBMIT_SCOPE, "errors");
const QUEUE_WAIT_NS: ProfileMetricId = ProfileMetricId::duration(CODECS_SUBMIT_SCOPE, "queue-wait");

const WORK_CALLS: ProfileMetricId = ProfileMetricId::count(CODECS_WORK_SCOPE, "calls");
const WORK_NS: ProfileMetricId = ProfileMetricId::duration(CODECS_WORK_SCOPE, "work");
const ENCODED_BYTES: ProfileMetricId = ProfileMetricId::bytes(CODECS_WORK_SCOPE, "encoded");
const DECODED_BYTES: ProfileMetricId = ProfileMetricId::bytes(CODECS_WORK_SCOPE, "decoded");
const WORK_ERRORS: ProfileMetricId = ProfileMetricId::count(CODECS_WORK_SCOPE, "errors");
const WORK_PANICS: ProfileMetricId = ProfileMetricId::count(CODECS_WORK_SCOPE, "panics");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CodecSubmitKind {
    Blocking,
    Try,
    Async,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CodecQueueTimer {
    timer: ProfileTimer,
    round: u64,
}

#[derive(Debug, Clone)]
pub struct CodecProfile {
    inner: Arc<CodecProfileInner>,
}

#[derive(Debug)]
struct CodecProfileInner {
    runtime: ProfileRuntime,
    _auto_round: Option<ProfileOwnedRound>,
}

pub(crate) struct CodecWorkProfile<'a> {
    recorder: Option<ProfileRecorder<'a>>,
    started: ProfileTimer,
    round: u64,
}

impl Default for CodecProfile {
    fn default() -> Self {
        Self::disabled()
    }
}

impl CodecProfile {
    pub fn disabled() -> Self {
        Self::new(
            ProfileRuntime::new_lazy(ProfileConfig::disabled(), codecs_profile_registry),
            None,
        )
    }

    pub fn enabled(label: impl Into<String>) -> Self {
        Self::new(
            ProfileRuntime::enabled_lazy(label, codecs_profile_registry),
            None,
        )
    }

    pub fn from_env() -> Self {
        let runtime = ProfileRuntime::from_env_lazy(codecs_profile_registry);
        let auto_round = runtime.is_global_enabled().then(|| runtime.start_owned());
        Self::new(runtime, auto_round)
    }

    pub fn from_runtime(runtime: ProfileRuntime) -> Self {
        runtime.ensure_registry_lazy(codecs_profile_registry);
        Self::new(runtime, None)
    }

    pub fn runtime(&self) -> &ProfileRuntime {
        &self.inner.runtime
    }

    pub fn into_runtime(self) -> ProfileRuntime {
        self.inner.runtime.clone()
    }

    pub fn start(&self) -> ProfileRound<'_> {
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

    pub(crate) fn record_submit(&self, kind: CodecSubmitKind) -> CodecQueueTimer {
        self.inner
            .runtime
            .with_recorder(|ctx| {
                ctx.inc(SUBMIT_CALLS);
                ctx.inc(submit_kind_calls(kind));
                CodecQueueTimer {
                    timer: ctx.timer(CODECS_SUBMIT_SCOPE),
                    round: ctx.round(),
                }
            })
            .unwrap_or_else(CodecQueueTimer::disabled)
    }

    pub(crate) fn record_submit_error(&self) {
        let _ = self.inner.runtime.with_recorder(|ctx| {
            ctx.inc(SUBMIT_ERRORS);
        });
    }

    pub(crate) fn start_work(&self) -> CodecWorkProfile<'_> {
        self.inner
            .runtime
            .with_recorder(CodecWorkProfile::recording)
            .unwrap_or_else(CodecWorkProfile::disabled)
    }

    fn new(runtime: ProfileRuntime, auto_round: Option<ProfileOwnedRound>) -> Self {
        Self {
            inner: Arc::new(CodecProfileInner {
                runtime,
                _auto_round: auto_round,
            }),
        }
    }
}

impl<'a> CodecWorkProfile<'a> {
    fn disabled() -> Self {
        Self {
            recorder: None,
            started: ProfileTimer::disabled(),
            round: 0,
        }
    }

    fn recording(recorder: ProfileRecorder<'a>) -> Self {
        let started = recorder.timer(CODECS_WORK_SCOPE);
        let round = recorder.round();
        Self {
            recorder: Some(recorder),
            started,
            round,
        }
    }

    pub(crate) fn record(
        &self,
        queued_at: CodecQueueTimer,
        encoded_bytes: usize,
        decoded_bytes: Option<usize>,
        error: bool,
        panicked: bool,
    ) {
        let Some(ctx) = &self.recorder else {
            return;
        };

        ctx.inc(WORK_CALLS);
        if queued_at.round == self.round {
            ctx.record_timer(QUEUE_WAIT_NS, queued_at.timer);
        }
        ctx.record_timer(WORK_NS, self.started);
        ctx.add_usize(ENCODED_BYTES, encoded_bytes);
        if let Some(decoded_bytes) = decoded_bytes {
            ctx.add_usize(DECODED_BYTES, decoded_bytes);
        }
        if error {
            ctx.inc(WORK_ERRORS);
        }
        if panicked {
            ctx.inc(WORK_PANICS);
        }
    }
}

impl CodecQueueTimer {
    fn disabled() -> Self {
        Self {
            timer: ProfileTimer::disabled(),
            round: 0,
        }
    }
}

fn submit_kind_calls(kind: CodecSubmitKind) -> ProfileMetricId {
    match kind {
        CodecSubmitKind::Blocking => SUBMIT_BLOCKING_CALLS,
        CodecSubmitKind::Try => SUBMIT_TRY_CALLS,
        CodecSubmitKind::Async => SUBMIT_ASYNC_CALLS,
    }
}

pub fn codecs_profile_registry() -> ProfileRegistry {
    crate::scdata_profile_registry!(
        components: [ProfileComponent::new(CODECS_COMPONENT).described("codec worker pool")],
        scopes: [
            ProfileScope::new(CODECS_SUBMIT_SCOPE)
                .kind(ProfileScopeKind::Event)
                .described("decode pool submissions"),
            ProfileScope::new(CODECS_WORK_SCOPE)
                .kind(ProfileScopeKind::Timer)
                .described("decode worker execution"),
        ],
    )
}

#[cfg(all(test, feature = "profile"))]
pub(crate) mod test_metrics {
    use super::*;

    pub(crate) const SUBMIT_CALLS: ProfileMetricId = super::SUBMIT_CALLS;
    pub(crate) const SUBMIT_BLOCKING_CALLS: ProfileMetricId = super::SUBMIT_BLOCKING_CALLS;
    pub(crate) const SUBMIT_TRY_CALLS: ProfileMetricId = super::SUBMIT_TRY_CALLS;
    pub(crate) const SUBMIT_ASYNC_CALLS: ProfileMetricId = super::SUBMIT_ASYNC_CALLS;
    pub(crate) const SUBMIT_ERRORS: ProfileMetricId = super::SUBMIT_ERRORS;
    pub(crate) const QUEUE_WAIT_NS: ProfileMetricId = super::QUEUE_WAIT_NS;
    pub(crate) const WORK_CALLS: ProfileMetricId = super::WORK_CALLS;
    pub(crate) const ENCODED_BYTES: ProfileMetricId = super::ENCODED_BYTES;
    pub(crate) const DECODED_BYTES: ProfileMetricId = super::DECODED_BYTES;
    pub(crate) const WORK_ERRORS: ProfileMetricId = super::WORK_ERRORS;
    pub(crate) const WORK_PANICS: ProfileMetricId = super::WORK_PANICS;
}
