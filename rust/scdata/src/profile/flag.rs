use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::timer::ProfileTimer;

#[derive(Debug)]
pub(super) struct ProfileFlagState {
    enabled: bool,
    active: Arc<AtomicBool>,
}

impl ProfileFlagState {
    pub(super) fn new(enabled: bool, active: Arc<AtomicBool>) -> Self {
        Self { enabled, active }
    }

    pub(super) fn configured_enabled(&self) -> bool {
        self.enabled
    }

    pub(super) fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    fn is_enabled(&self) -> bool {
        self.enabled && self.is_active()
    }
}

/// Scope-level switch handle used inside the profile runtime.
#[derive(Debug, Clone)]
pub(super) struct ProfileScopeFlag {
    state: Arc<ProfileFlagState>,
}

impl ProfileScopeFlag {
    pub(super) fn new(state: Arc<ProfileFlagState>) -> Self {
        Self { state }
    }

    pub(super) fn is_enabled(&self) -> bool {
        self.state.is_enabled()
    }

    /// Starts a timer only when this scope is enabled for the current round.
    pub(super) fn timer(&self) -> ProfileTimer {
        ProfileTimer::start(self.is_enabled())
    }
}
