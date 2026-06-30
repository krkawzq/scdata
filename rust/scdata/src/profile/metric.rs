use std::fmt;
#[cfg(feature = "profile")]
use std::sync::Arc;

#[cfg(feature = "profile")]
use super::counter::AtomicCounter;
#[cfg(feature = "profile")]
use super::flag::ProfileScopeFlag;
use super::ids::ProfileScopeId;
#[cfg(feature = "profile")]
use super::timer::ProfileTimer;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum ProfileMetricKind {
    Count,
    Bytes,
    DurationNs,
    Gauge,
    Custom(&'static str),
}

impl ProfileMetricKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Count => "count",
            Self::Bytes => "bytes",
            Self::DurationNs => "duration_ns",
            Self::Gauge => "gauge",
            Self::Custom(name) => name,
        }
    }
}

impl fmt::Display for ProfileMetricKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub struct ProfileMetricId {
    pub scope: ProfileScopeId,
    pub name: &'static str,
    pub kind: ProfileMetricKind,
}

impl ProfileMetricId {
    pub const fn new(scope: ProfileScopeId, name: &'static str, kind: ProfileMetricKind) -> Self {
        Self { scope, name, kind }
    }

    pub const fn count(scope: ProfileScopeId, name: &'static str) -> Self {
        Self::new(scope, name, ProfileMetricKind::Count)
    }

    pub const fn bytes(scope: ProfileScopeId, name: &'static str) -> Self {
        Self::new(scope, name, ProfileMetricKind::Bytes)
    }

    pub const fn duration(scope: ProfileScopeId, name: &'static str) -> Self {
        Self::new(scope, name, ProfileMetricKind::DurationNs)
    }

    pub const fn gauge(scope: ProfileScopeId, name: &'static str) -> Self {
        Self::new(scope, name, ProfileMetricKind::Gauge)
    }
}

#[cfg(feature = "profile")]
#[derive(Debug)]
pub(super) struct MetricCell {
    pub(super) id: ProfileMetricId,
    pub(super) value: AtomicCounter,
}

#[cfg(feature = "profile")]
impl MetricCell {
    pub(super) fn new(id: ProfileMetricId) -> Self {
        Self {
            id,
            value: AtomicCounter::default(),
        }
    }
}

#[cfg(feature = "profile")]
#[derive(Debug, Clone)]
pub(super) struct ProfileMetric {
    flag: ProfileScopeFlag,
    cell: Arc<MetricCell>,
}

#[cfg(feature = "profile")]
impl ProfileMetric {
    pub(super) fn new(flag: ProfileScopeFlag, cell: Arc<MetricCell>) -> Self {
        Self { flag, cell }
    }

    fn is_enabled(&self) -> bool {
        self.flag.is_enabled()
    }

    pub(super) fn add(&self, value: u64) {
        if self.is_enabled() {
            self.cell.value.add(value);
        }
    }

    pub(super) fn set(&self, value: u64) {
        if self.is_enabled() {
            self.cell.value.set(value);
        }
    }

    pub(super) fn record_timer(&self, timer: ProfileTimer) {
        if self.is_enabled() {
            if let Some(elapsed_ns) = timer.elapsed_ns() {
                self.cell.value.add(elapsed_ns);
            }
        }
    }
}
