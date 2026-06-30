use std::time::Duration;

#[cfg(feature = "profile")]
use std::time::Instant;

#[cfg(feature = "profile")]
use super::counter::duration_ns;

#[cfg(feature = "profile")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProfileTimer {
    started: Option<Instant>,
}

#[cfg(feature = "profile")]
impl ProfileTimer {
    pub fn start(enabled: bool) -> Self {
        Self {
            started: enabled.then(Instant::now),
        }
    }

    pub const fn disabled() -> Self {
        Self { started: None }
    }

    pub fn is_enabled(self) -> bool {
        self.started.is_some()
    }

    pub fn elapsed(self) -> Option<Duration> {
        self.started.map(|started| started.elapsed())
    }

    pub fn elapsed_ns(self) -> Option<u64> {
        self.elapsed().map(duration_ns)
    }
}

#[cfg(not(feature = "profile"))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProfileTimer;

#[cfg(not(feature = "profile"))]
impl ProfileTimer {
    #[inline(always)]
    pub fn start(_enabled: bool) -> Self {
        Self
    }

    #[inline(always)]
    pub fn disabled() -> Self {
        Self
    }

    #[inline(always)]
    pub fn is_enabled(self) -> bool {
        false
    }

    #[inline(always)]
    pub fn elapsed(self) -> Option<Duration> {
        None
    }

    #[inline(always)]
    pub fn elapsed_ns(self) -> Option<u64> {
        None
    }
}
