//! Runtime-controlled profiling infrastructure for low-overhead scoped metrics.
//!
//! Profiling is configured in three layers:
//!
//! 1. a global switch
//! 2. component switches
//! 3. scope switches
//!
//! # Usage
//!
//! Profiling runs in explicit `start()` / `end()` rounds. Register components
//! and scopes while the runtime is idle, call [`ProfileRuntime::start`], and
//! record metrics through `scdata_profile_record!`, `scdata_profile_measure!`,
//! or [`ProfileRuntime::with_recorder`]. Call [`ProfileRound::end`] or
//! [`ProfileRuntime::end`] to collect a [`ProfileSnapshot`] and clear the
//! round's metric storage.
//!
//! Configuration changes apply to the next round. A [`ProfileRecorder`] is bound
//! to the round that created it; after `end()`, writes through that recorder are
//! ignored and cannot affect later rounds.
//!
//! # Safety
//!
//! This module does not use `unsafe`. Runtime state transitions are serialized,
//! and per-round recorders are backed by an atomic active token. `end()`
//! deactivates the token and detaches metric cells from the runtime, so late
//! writes through stale recorders cannot corrupt idle state or later rounds.
//!
//! # Compile-time disable
//!
//! The `profile` Cargo feature is enabled by default. Building without it keeps
//! this public API available but compiles the runtime to disabled no-op handles,
//! empty snapshots, and no metric storage. Optimized deployment builds can then
//! inline these constants and eliminate profiling branches from hot paths.
//! Use [`ProfileRuntime::new_lazy`] and the `scdata_profile_record!` /
//! `scdata_profile_measure!` macros for downstream profiling hooks that should
//! disappear entirely when the feature is disabled.

mod config;
mod counter;
#[cfg(all(test, not(feature = "profile")))]
mod disabled_tests;
#[cfg(feature = "profile")]
mod flag;
mod ids;
mod macros;
mod metric;
mod registry;
mod runtime;
mod snapshot;
#[cfg(all(test, feature = "profile"))]
mod tests;
mod timer;

pub use config::{ProfileConfig, ProfileDefault, ProfileRuleSet};
pub use ids::{ProfileComponentId, ProfilePattern, ProfileScopeId};
pub use metric::{ProfileMetricId, ProfileMetricKind};
pub use registry::{ProfileComponent, ProfileRegistry, ProfileScope, ProfileScopeKind};
pub use runtime::{ProfileOwnedRound, ProfilePhase, ProfileRecorder, ProfileRound, ProfileRuntime};
pub use snapshot::{
    ProfileComponentSnapshot, ProfileMetricSnapshot, ProfileScopeSnapshot, ProfileSnapshot,
};
pub use timer::ProfileTimer;

pub use counter::{
    avg_ns, bytes_to_mib, duration_ns, ns_to_ms, per_second, ratio, CounterSnapshot,
};
