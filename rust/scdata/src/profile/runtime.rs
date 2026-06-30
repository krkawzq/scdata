#[cfg(feature = "profile")]
use std::collections::BTreeMap;
#[cfg(feature = "profile")]
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
#[cfg(feature = "profile")]
use std::sync::{Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
#[cfg(feature = "profile")]
use std::time::Instant;

use super::config::ProfileConfig;
#[cfg(feature = "profile")]
use super::counter::duration_ns;
#[cfg(feature = "profile")]
use super::flag::ProfileFlagState;
#[cfg(feature = "profile")]
use super::flag::ProfileScopeFlag;
#[cfg(feature = "profile")]
use super::ids::ProfileComponentId;
use super::ids::ProfileScopeId;
#[cfg(feature = "profile")]
use super::metric::MetricCell;
#[cfg(feature = "profile")]
use super::metric::ProfileMetric;
use super::metric::ProfileMetricId;
use super::registry::ProfileRegistry;
#[cfg(feature = "profile")]
use super::registry::{ProfileComponent, ProfileScope};
#[cfg(not(feature = "profile"))]
use super::snapshot::ProfileSnapshot;
#[cfg(feature = "profile")]
use super::snapshot::{
    metric_snapshot, ProfileComponentSnapshot, ProfileScopeSnapshot, ProfileSnapshot,
};
use super::timer::ProfileTimer;

/// Runtime lifecycle phase.
///
/// - [`Idle`]: configuration and registry updates are allowed; no metrics are
///   recorded.
/// - [`Recording`]: configuration is frozen for the current round. Updating
///   configuration in this phase panics.
///
/// [`Idle`]: ProfilePhase::Idle
/// [`Recording`]: ProfilePhase::Recording
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProfilePhase {
    Idle,
    Recording,
}

/// Restricted recording context passed to profiling closures.
///
/// Code inside this context must be profiling-only. When the `profile` feature
/// is disabled, the macro APIs drop these closures entirely.
#[derive(Debug, Clone)]
pub struct ProfileRecorder<'a> {
    #[cfg(feature = "profile")]
    runtime: &'a ProfileRuntime,
    #[cfg(feature = "profile")]
    frozen: Arc<FrozenFlags>,
    #[cfg(not(feature = "profile"))]
    _marker: std::marker::PhantomData<&'a ProfileRuntime>,
}

impl<'a> ProfileRecorder<'a> {
    /// Starts a timer for the scope when the active round enables it.
    #[inline]
    pub fn timer(&self, scope: ProfileScopeId) -> ProfileTimer {
        #[cfg(feature = "profile")]
        {
            self.runtime.timer_for_round(&self.frozen, scope)
        }
        #[cfg(not(feature = "profile"))]
        {
            let _ = scope;
            ProfileTimer::disabled()
        }
    }

    /// Returns whether the active round enables a scope.
    #[inline]
    pub fn is_scope_enabled(&self, scope: ProfileScopeId) -> bool {
        #[cfg(feature = "profile")]
        {
            self.runtime.is_scope_enabled_for_round(&self.frozen, scope)
        }
        #[cfg(not(feature = "profile"))]
        {
            let _ = scope;
            false
        }
    }

    /// Returns the round this recorder is bound to.
    #[inline]
    pub fn round(&self) -> u64 {
        #[cfg(feature = "profile")]
        {
            self.frozen.round
        }
        #[cfg(not(feature = "profile"))]
        {
            0
        }
    }

    /// Increments a metric when it is available for the active round.
    #[inline]
    pub fn inc(&self, metric: ProfileMetricId) {
        self.add(metric, 1);
    }

    /// Adds a non-zero value to a metric when it is available for the active round.
    #[inline]
    pub fn add(&self, metric: ProfileMetricId, value: u64) {
        if value == 0 {
            return;
        }
        #[cfg(feature = "profile")]
        if let Some(metric) = self.runtime.metric_for_round(&self.frozen, metric) {
            metric.add(value);
        }
        #[cfg(not(feature = "profile"))]
        let _ = (metric, value);
    }

    /// Adds a usize value to a metric, saturating only on narrow targets.
    #[inline]
    pub fn add_usize(&self, metric: ProfileMetricId, value: usize) {
        self.add(metric, value.min(u64::MAX as usize) as u64);
    }

    /// Sets a metric when it is available for the active round.
    #[inline]
    pub fn set(&self, metric: ProfileMetricId, value: u64) {
        #[cfg(feature = "profile")]
        if let Some(metric) = self.runtime.metric_for_round(&self.frozen, metric) {
            metric.set(value);
        }
        #[cfg(not(feature = "profile"))]
        let _ = (metric, value);
    }

    /// Adds a timer's elapsed nanoseconds to a metric when the timer is active.
    #[inline]
    pub fn record_timer(&self, metric: ProfileMetricId, timer: ProfileTimer) {
        if !timer.is_enabled() {
            return;
        }
        #[cfg(feature = "profile")]
        if let Some(metric) = self.runtime.metric_for_round(&self.frozen, metric) {
            metric.record_timer(timer);
        }
        #[cfg(not(feature = "profile"))]
        let _ = (metric, timer);
    }
}

#[cfg(feature = "profile")]
const PHASE_IDLE: u8 = 0;
#[cfg(feature = "profile")]
const PHASE_RECORDING: u8 = 1;

#[cfg(feature = "profile")]
#[derive(Debug, Clone)]
pub struct ProfileRuntime {
    inner: Arc<ProfileInner>,
}

#[cfg(feature = "profile")]
#[derive(Debug)]
struct ProfileInner {
    /// Serializes lifecycle transitions with configuration and registry writes.
    transition: Mutex<()>,
    config: RwLock<ProfileConfig>,
    /// Fast phase reads are atomic; phase transitions are guarded by `transition`.
    phase: AtomicU8,
    round: AtomicU64,
    components: RwLock<BTreeMap<ProfileComponentId, ProfileComponent>>,
    scopes: RwLock<BTreeMap<ProfileScopeId, ProfileScope>>,
    metrics: RwLock<BTreeMap<ProfileMetricId, Arc<MetricCell>>>,
    /// Flags frozen for the active round. `None` while idle.
    frozen: RwLock<Option<Arc<FrozenFlags>>>,
}

#[cfg(feature = "profile")]
#[derive(Debug)]
struct FrozenFlags {
    active: Arc<AtomicBool>,
    global_enabled: bool,
    components: BTreeMap<ProfileComponentId, Arc<ProfileFlagState>>,
    scopes: BTreeMap<ProfileScopeId, Arc<ProfileFlagState>>,
    round: u64,
    started: Mutex<Instant>,
}

/// RAII guard for one profiling round.
///
/// Returned by [`ProfileRuntime::start`]. Calling [`ProfileRound::end`] consumes
/// the guard and returns the snapshot. Dropping the guard without calling
/// `end()` silently finishes the round and discards the snapshot, which prevents
/// the runtime from being left in [`ProfilePhase::Recording`] after early returns
/// or panics.
#[cfg(feature = "profile")]
pub struct ProfileRound<'a> {
    runtime: &'a ProfileRuntime,
    round: u64,
    finished: bool,
}

#[cfg(feature = "profile")]
impl<'a> ProfileRound<'a> {
    /// Ends the round, returns its snapshot, and clears metric storage.
    pub fn end(mut self) -> ProfileSnapshot {
        assert!(!self.finished, "ProfileRound::end: already finished");
        self.finished = true;
        self.runtime
            .end_round_inner(self.round, "ProfileRound::end")
    }

    /// Returns the round number owned by this guard.
    pub fn round(&self) -> u64 {
        self.round
    }
}

#[cfg(feature = "profile")]
impl Drop for ProfileRound<'_> {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.runtime.end_round_if_recording(self.round);
        }
    }
}

/// Owned RAII guard for one profiling round.
///
/// This is useful when a component needs to keep a round alive without tying the
/// guard lifetime to a borrowed runtime reference.
#[cfg(feature = "profile")]
#[derive(Debug)]
pub struct ProfileOwnedRound {
    runtime: ProfileRuntime,
    round: u64,
    finished: bool,
}

#[cfg(feature = "profile")]
impl ProfileOwnedRound {
    /// Ends the round, returns its snapshot, and clears metric storage.
    pub fn end(mut self) -> ProfileSnapshot {
        assert!(!self.finished, "ProfileOwnedRound::end: already finished");
        self.finished = true;
        self.runtime
            .end_round_inner(self.round, "ProfileOwnedRound::end")
    }

    /// Returns the round number owned by this guard.
    pub fn round(&self) -> u64 {
        self.round
    }
}

#[cfg(feature = "profile")]
impl Drop for ProfileOwnedRound {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.runtime.end_round_if_recording(self.round);
        }
    }
}

#[cfg(feature = "profile")]
impl ProfileRuntime {
    fn new(config: ProfileConfig, registry: ProfileRegistry) -> Self {
        let runtime = Self {
            inner: Arc::new(ProfileInner {
                transition: Mutex::new(()),
                config: RwLock::new(config),
                phase: AtomicU8::new(PHASE_IDLE),
                round: AtomicU64::new(0),
                components: RwLock::new(BTreeMap::new()),
                scopes: RwLock::new(BTreeMap::new()),
                metrics: RwLock::new(BTreeMap::new()),
                frozen: RwLock::new(None),
            }),
        };
        runtime.register_registry(registry);
        runtime
    }

    /// Creates a runtime and builds the registry only when profiling is compiled in.
    ///
    /// The no-profile implementation keeps the same signature but does not call
    /// the closure, so registry construction can be eliminated from deployment
    /// builds.
    #[inline]
    pub fn new_lazy(config: ProfileConfig, registry: impl FnOnce() -> ProfileRegistry) -> Self {
        Self::new(config, registry())
    }

    pub fn disabled() -> Self {
        Self::new(ProfileConfig::disabled(), ProfileRegistry::new())
    }

    /// Creates an enabled runtime with lazily constructed registry definitions.
    #[inline]
    pub fn enabled_lazy(
        label: impl Into<String>,
        registry: impl FnOnce() -> ProfileRegistry,
    ) -> Self {
        Self::new_lazy(ProfileConfig::enabled(label), registry)
    }

    /// Creates a runtime from environment variables with a lazy registry.
    #[inline]
    pub fn from_env_lazy(registry: impl FnOnce() -> ProfileRegistry) -> Self {
        Self::new_lazy(ProfileConfig::from_env(), registry)
    }

    /// Returns the current lifecycle phase.
    pub fn phase(&self) -> ProfilePhase {
        match self.inner.phase.load(Ordering::Acquire) {
            PHASE_RECORDING => ProfilePhase::Recording,
            _ => ProfilePhase::Idle,
        }
    }

    /// Returns true while a round is recording.
    pub fn is_recording(&self) -> bool {
        self.phase() == ProfilePhase::Recording
    }

    /// Number of rounds that have been started.
    pub fn round(&self) -> u64 {
        self.inner.round.load(Ordering::Relaxed)
    }

    pub fn config(&self) -> ProfileConfig {
        read_lock(&self.inner.config).clone()
    }

    /// Returns the current global profiling switch.
    pub fn is_global_enabled(&self) -> bool {
        read_lock(&self.inner.config).enabled
    }

    /// Replaces the runtime configuration.
    ///
    /// This is only allowed while idle. Calling it during a recording round
    /// panics so the current round remains deterministic.
    pub fn update_config(&self, config: ProfileConfig) {
        let _transition = self.transition_lock();
        self.require_idle("update_config");
        *write_lock(&self.inner.config) = config;
    }

    /// Starts one profiling round.
    ///
    /// This freezes the current configuration, clears stale metric cells, and
    /// returns a guard that owns the recording round.
    pub fn start(&self) -> ProfileRound<'_> {
        let round = self.enter_recording("start");
        ProfileRound {
            runtime: self,
            round,
            finished: false,
        }
    }

    /// Starts one profiling round and returns an owned guard.
    pub fn start_owned(&self) -> ProfileOwnedRound {
        let round = self.enter_recording("start_owned");
        ProfileOwnedRound {
            runtime: self.clone(),
            round,
            finished: false,
        }
    }

    /// Ends the active round, returns its snapshot, and clears recorded metrics.
    ///
    /// Prefer calling [`ProfileRound::end`] on the guard returned by [`start`].
    /// This method exists for call sites that cannot keep the guard. It panics
    /// when the runtime is idle.
    ///
    /// [`start`]: ProfileRuntime::start
    pub fn end(&self) -> ProfileSnapshot {
        self.end_inner()
    }

    /// Snapshots the current counters and resets them without ending the round.
    pub fn snapshot_and_reset(&self) -> ProfileSnapshot {
        let _transition = self.transition_lock();
        self.snapshot_inner(true)
    }

    /// Resets current counters without ending the round.
    pub fn reset_metrics(&self) {
        let _ = self.snapshot_and_reset();
    }

    fn register_registry(&self, registry: ProfileRegistry) {
        let _transition = self.transition_lock();
        self.require_idle("register_registry");
        self.register_registry_locked(registry);
    }

    /// Registers lazily constructed definitions while the runtime is idle.
    #[inline]
    pub fn register_registry_lazy(&self, registry: impl FnOnce() -> ProfileRegistry) {
        self.register_registry(registry());
    }

    /// Ensures that registry definitions exist for the runtime.
    ///
    /// Unlike [`register_registry`], this method is safe to call during an active
    /// round. Missing components and scopes are added to both the definition table
    /// and the current frozen flag set, so late-bound wrappers can start recording
    /// in the current round. Existing definitions are left unchanged.
    fn ensure_registry(&self, registry: ProfileRegistry) {
        let _transition = self.transition_lock();
        if self.phase() == ProfilePhase::Recording {
            self.ensure_registry_recording(registry);
        } else {
            self.register_registry_locked(registry);
        }
    }

    /// Ensures lazily constructed definitions exist for the current runtime.
    #[inline]
    pub fn ensure_registry_lazy(&self, registry: impl FnOnce() -> ProfileRegistry) {
        self.ensure_registry(registry());
    }

    fn scope_flag_for_round(
        &self,
        frozen: &Arc<FrozenFlags>,
        scope: ProfileScopeId,
    ) -> Option<ProfileScopeFlag> {
        if !frozen.active.load(Ordering::Acquire) {
            return None;
        }
        let state = frozen.scopes.get(&scope)?.clone();
        Some(ProfileScopeFlag::new(state))
    }

    fn is_scope_enabled_for_round(&self, frozen: &Arc<FrozenFlags>, scope: ProfileScopeId) -> bool {
        self.scope_flag_for_round(frozen, scope)
            .is_some_and(|flag| flag.is_enabled())
    }

    fn timer_for_round(&self, frozen: &Arc<FrozenFlags>, scope: ProfileScopeId) -> ProfileTimer {
        self.scope_flag_for_round(frozen, scope)
            .map_or_else(ProfileTimer::disabled, |flag| flag.timer())
    }

    fn metric_for_round(
        &self,
        frozen: &Arc<FrozenFlags>,
        id: ProfileMetricId,
    ) -> Option<ProfileMetric> {
        let flag = self.scope_flag_for_round(frozen, id.scope)?;
        if !flag.is_enabled() {
            return None;
        }

        {
            let metrics = read_lock(&self.inner.metrics);
            if let Some(cell) = metrics.get(&id) {
                return Some(ProfileMetric::new(flag, cell.clone()));
            }
        }

        let mut metrics = write_lock(&self.inner.metrics);
        if !flag.is_enabled() {
            return None;
        }
        let cell = metrics
            .entry(id)
            .or_insert_with(|| Arc::new(MetricCell::new(id)))
            .clone();
        Some(ProfileMetric::new(flag, cell))
    }

    /// Runs a profiling-only closure during an active recording round.
    ///
    /// Returns `None` while idle. Prefer the `scdata_profile_record!` and
    /// `scdata_profile_measure!` macros when downstream code should be compiled
    /// away with the `profile` feature disabled.
    #[inline]
    pub fn with_recorder<'a, R>(&'a self, f: impl FnOnce(ProfileRecorder<'a>) -> R) -> Option<R> {
        let frozen = read_lock(&self.inner.frozen).clone()?;
        if !frozen.global_enabled {
            return None;
        }
        if !frozen.active.load(Ordering::Acquire) {
            return None;
        }
        Some(f(ProfileRecorder {
            runtime: self,
            frozen,
        }))
    }

    pub fn snapshot(&self) -> ProfileSnapshot {
        let _transition = self.transition_lock();
        self.snapshot_inner(false)
    }

    fn enter_recording(&self, op: &'static str) -> u64 {
        let _transition = self.transition_lock();
        if self.phase() == ProfilePhase::Recording {
            panic!("ProfileRuntime::{op}: already recording; call end() first");
        }

        let config = self.config();
        let components = read_lock(&self.inner.components).clone();
        let scopes = read_lock(&self.inner.scopes).clone();
        self.clear_metric_cells();

        let active = Arc::new(AtomicBool::new(true));
        let mut frozen_components = BTreeMap::new();
        for def in components.values() {
            let enabled = config.component_enabled(def.id, def.default_enabled);
            frozen_components.insert(
                def.id,
                Arc::new(ProfileFlagState::new(enabled, Arc::clone(&active))),
            );
        }
        let mut frozen_scopes = BTreeMap::new();
        for def in scopes.values() {
            let component_default = components
                .get(&def.id.component())
                .map_or(true, |component| component.default_enabled);
            let enabled = config.scope_enabled(def.id, component_default, def.default_enabled);
            frozen_scopes.insert(
                def.id,
                Arc::new(ProfileFlagState::new(enabled, Arc::clone(&active))),
            );
        }

        let round = self.inner.round.fetch_add(1, Ordering::Relaxed) + 1;
        let frozen = Arc::new(FrozenFlags {
            active,
            global_enabled: config.enabled,
            components: frozen_components,
            scopes: frozen_scopes,
            round,
            started: Mutex::new(Instant::now()),
        });

        *write_lock(&self.inner.frozen) = Some(frozen);
        self.inner.phase.store(PHASE_RECORDING, Ordering::Release);
        round
    }

    fn end_inner(&self) -> ProfileSnapshot {
        self.finish_recording(None, true, "ProfileRuntime::end")
            .expect("ProfileRuntime::end: not recording; call start() first")
    }

    fn end_round_inner(&self, round: u64, op: &'static str) -> ProfileSnapshot {
        self.finish_recording(Some(round), true, op)
            .unwrap_or_else(|| panic!("{op}: round {round} is no longer active"))
    }

    fn end_round_if_recording(&self, round: u64) -> Option<ProfileSnapshot> {
        self.finish_recording(Some(round), false, "ProfileRound::drop")
    }

    fn finish_recording(
        &self,
        expected_round: Option<u64>,
        panic_if_inactive: bool,
        op: &'static str,
    ) -> Option<ProfileSnapshot> {
        let _transition = self.transition_lock();
        if self.phase() == ProfilePhase::Idle {
            if panic_if_inactive {
                panic!("{op}: not recording; call start() first");
            }
            return None;
        }
        let current_round = self.round();
        if expected_round.is_some_and(|round| round != current_round) {
            if panic_if_inactive {
                panic!("{op}: round is no longer active");
            }
            return None;
        }
        self.deactivate_current_round();
        let snapshot = self.snapshot_inner(true);
        self.clear_metric_cells();
        *write_lock(&self.inner.frozen) = None;
        self.inner.phase.store(PHASE_IDLE, Ordering::Release);
        Some(snapshot)
    }

    fn require_idle(&self, op: &'static str) {
        if self.phase() == ProfilePhase::Recording {
            panic!("ProfileRuntime::{op}: not allowed while recording; call end() first");
        }
    }

    fn register_component_locked(&self, component: ProfileComponent) {
        write_lock(&self.inner.components).insert(component.id, component);
    }

    fn register_scope_locked(&self, scope: ProfileScope) {
        self.ensure_component_locked(scope.id.component());
        write_lock(&self.inner.scopes).insert(scope.id, scope);
    }

    fn ensure_component_locked(&self, component: ProfileComponentId) {
        let exists = read_lock(&self.inner.components).contains_key(&component);
        if !exists {
            self.register_component_locked(ProfileComponent::new(component));
        }
    }

    fn register_registry_locked(&self, registry: ProfileRegistry) {
        for component in registry.components() {
            self.register_component_locked(component.clone());
        }
        for scope in registry.scopes() {
            self.register_scope_locked(scope.clone());
        }
    }

    fn ensure_registry_recording(&self, registry: ProfileRegistry) {
        let config = self.config();
        let mut components = write_lock(&self.inner.components);
        let mut scopes = write_lock(&self.inner.scopes);
        let mut frozen_guard = write_lock(&self.inner.frozen);
        let (active, global_enabled, mut frozen_components, mut frozen_scopes, round, started) = {
            let Some(frozen) = frozen_guard.as_ref() else {
                return;
            };
            (
                Arc::clone(&frozen.active),
                frozen.global_enabled,
                frozen.components.clone(),
                frozen.scopes.clone(),
                frozen.round,
                *lock_mutex(&frozen.started),
            )
        };

        for component in registry.components() {
            components
                .entry(component.id)
                .or_insert_with(|| component.clone());
            frozen_components.entry(component.id).or_insert_with(|| {
                let default_enabled = components
                    .get(&component.id)
                    .map_or(component.default_enabled, |definition| {
                        definition.default_enabled
                    });
                Arc::new(ProfileFlagState::new(
                    config.component_enabled(component.id, default_enabled),
                    Arc::clone(&active),
                ))
            });
        }

        for scope in registry.scopes() {
            let component = scope.id.component();
            components
                .entry(component)
                .or_insert_with(|| ProfileComponent::new(component));
            let component_default = components
                .get(&component)
                .map_or(true, |definition| definition.default_enabled);
            frozen_components.entry(component).or_insert_with(|| {
                Arc::new(ProfileFlagState::new(
                    config.component_enabled(component, component_default),
                    Arc::clone(&active),
                ))
            });
            scopes.entry(scope.id).or_insert_with(|| scope.clone());

            frozen_scopes.entry(scope.id).or_insert_with(|| {
                let scope_default = scopes
                    .get(&scope.id)
                    .map_or(scope.default_enabled, |definition| {
                        definition.default_enabled
                    });
                Arc::new(ProfileFlagState::new(
                    config.scope_enabled(scope.id, component_default, scope_default),
                    Arc::clone(&active),
                ))
            });
        }

        *frozen_guard = Some(Arc::new(FrozenFlags {
            active,
            global_enabled,
            components: frozen_components,
            scopes: frozen_scopes,
            round,
            started: Mutex::new(started),
        }));
    }

    fn deactivate_current_round(&self) {
        if let Some(frozen) = read_lock(&self.inner.frozen).clone() {
            let _metrics = write_lock(&self.inner.metrics);
            frozen.active.store(false, Ordering::Release);
        }
    }

    fn clear_metric_cells(&self) {
        write_lock(&self.inner.metrics).clear();
    }

    fn snapshot_inner(&self, reset: bool) -> ProfileSnapshot {
        let config = self.config();
        let components = read_lock(&self.inner.components).clone();
        let scopes = read_lock(&self.inner.scopes).clone();
        let metrics = read_lock(&self.inner.metrics).clone();
        let frozen = read_lock(&self.inner.frozen).clone();
        let frozen = frozen.as_ref();

        ProfileSnapshot {
            label: config.label,
            round: frozen.map(|f| f.round).unwrap_or(0),
            elapsed_ns: frozen
                .map(|f| duration_ns(lock_mutex(&f.started).elapsed()))
                .unwrap_or(0),
            global_enabled: config.enabled,
            components: components
                .values()
                .map(|def| {
                    let enabled = frozen
                        .and_then(|f| f.components.get(&def.id))
                        .map(|state| state.configured_enabled())
                        .unwrap_or(false);
                    ProfileComponentSnapshot {
                        id: def.id,
                        enabled,
                        default_enabled: def.default_enabled,
                        description: def.description.to_string(),
                    }
                })
                .collect(),
            scopes: scopes
                .values()
                .map(|def| {
                    let enabled = frozen
                        .and_then(|f| f.scopes.get(&def.id))
                        .map(|state| state.configured_enabled())
                        .unwrap_or(false);
                    ProfileScopeSnapshot {
                        id: def.id,
                        enabled,
                        default_enabled: def.default_enabled,
                        kind: def.kind,
                        description: def.description.to_string(),
                    }
                })
                .collect(),
            metrics: metrics
                .values()
                .map(|cell| metric_snapshot(cell, reset))
                .collect(),
        }
    }

    fn transition_lock(&self) -> MutexGuard<'_, ()> {
        lock_mutex(&self.inner.transition)
    }
}

#[cfg(not(feature = "profile"))]
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct ProfileRuntime;

#[cfg(not(feature = "profile"))]
pub struct ProfileRound<'a> {
    runtime: &'a ProfileRuntime,
}

#[cfg(not(feature = "profile"))]
impl<'a> ProfileRound<'a> {
    #[inline(always)]
    pub fn end(self) -> ProfileSnapshot {
        disabled_snapshot()
    }

    #[inline(always)]
    pub fn round(&self) -> u64 {
        let _ = self.runtime;
        0
    }
}

#[cfg(not(feature = "profile"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct ProfileOwnedRound {
    runtime: ProfileRuntime,
}

#[cfg(not(feature = "profile"))]
impl ProfileOwnedRound {
    #[inline(always)]
    pub fn end(self) -> ProfileSnapshot {
        let _ = self.runtime;
        disabled_snapshot()
    }

    #[inline(always)]
    pub fn round(&self) -> u64 {
        let _ = self.runtime;
        0
    }
}

#[cfg(not(feature = "profile"))]
impl ProfileRuntime {
    #[inline(always)]
    pub fn new_lazy(_config: ProfileConfig, _registry: impl FnOnce() -> ProfileRegistry) -> Self {
        Self
    }

    #[inline(always)]
    pub fn disabled() -> Self {
        Self
    }

    #[inline(always)]
    pub fn enabled_lazy(
        _label: impl Into<String>,
        _registry: impl FnOnce() -> ProfileRegistry,
    ) -> Self {
        Self
    }

    #[inline(always)]
    pub fn from_env_lazy(_registry: impl FnOnce() -> ProfileRegistry) -> Self {
        Self
    }

    #[inline(always)]
    pub fn phase(&self) -> ProfilePhase {
        ProfilePhase::Idle
    }

    #[inline(always)]
    pub fn is_recording(&self) -> bool {
        false
    }

    #[inline(always)]
    pub fn round(&self) -> u64 {
        0
    }

    #[inline(always)]
    pub fn config(&self) -> ProfileConfig {
        ProfileConfig::disabled()
    }

    #[inline(always)]
    pub fn is_global_enabled(&self) -> bool {
        false
    }

    #[inline(always)]
    pub fn update_config(&self, _config: ProfileConfig) {}

    #[inline(always)]
    pub fn start(&self) -> ProfileRound<'_> {
        ProfileRound { runtime: self }
    }

    #[inline(always)]
    pub fn start_owned(&self) -> ProfileOwnedRound {
        ProfileOwnedRound { runtime: *self }
    }

    #[inline(always)]
    pub fn end(&self) -> ProfileSnapshot {
        disabled_snapshot()
    }

    #[inline(always)]
    pub fn snapshot_and_reset(&self) -> ProfileSnapshot {
        disabled_snapshot()
    }

    #[inline(always)]
    pub fn reset_metrics(&self) {}

    #[inline(always)]
    pub fn register_registry_lazy(&self, _registry: impl FnOnce() -> ProfileRegistry) {}

    #[inline(always)]
    pub fn ensure_registry_lazy(&self, _registry: impl FnOnce() -> ProfileRegistry) {}

    #[inline(always)]
    pub fn with_recorder<'a, R>(&'a self, _f: impl FnOnce(ProfileRecorder<'a>) -> R) -> Option<R> {
        None
    }

    #[inline(always)]
    pub fn snapshot(&self) -> ProfileSnapshot {
        disabled_snapshot()
    }
}

#[cfg(not(feature = "profile"))]
fn disabled_snapshot() -> ProfileSnapshot {
    ProfileSnapshot {
        label: "scdata".to_string(),
        round: 0,
        elapsed_ns: 0,
        global_enabled: false,
        components: Vec::new(),
        scopes: Vec::new(),
        metrics: Vec::new(),
    }
}

// Poison recovery helpers.
//
// Safety: profiling is observational. If a lock is poisoned by a panic, keeping
// the existing state is safer than propagating another panic into the data path.

#[cfg(feature = "profile")]
fn read_lock<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(feature = "profile")]
fn write_lock<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(feature = "profile")]
fn lock_mutex<T>(lock: &Mutex<T>) -> MutexGuard<'_, T> {
    lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}
