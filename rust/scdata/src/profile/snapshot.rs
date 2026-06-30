use std::fmt;

use super::counter::{bytes_to_mib, ns_to_ms, CounterSnapshot};
use super::ids::{ProfileComponentId, ProfileScopeId};
#[cfg(feature = "profile")]
use super::metric::MetricCell;
use super::metric::{ProfileMetricId, ProfileMetricKind};
use super::registry::ProfileScopeKind;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProfileComponentSnapshot {
    pub id: ProfileComponentId,
    pub enabled: bool,
    pub default_enabled: bool,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProfileScopeSnapshot {
    pub id: ProfileScopeId,
    pub enabled: bool,
    pub default_enabled: bool,
    pub kind: ProfileScopeKind,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProfileMetricSnapshot {
    pub id: ProfileMetricId,
    pub value: CounterSnapshot,
}

impl ProfileMetricSnapshot {
    pub fn value(&self) -> u64 {
        self.value.value
    }

    pub fn as_mib(&self) -> f64 {
        bytes_to_mib(self.value())
    }

    pub fn as_ms(&self) -> f64 {
        ns_to_ms(self.value())
    }

    pub fn kind(&self) -> ProfileMetricKind {
        self.id.kind
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProfileSnapshot {
    pub label: String,
    pub round: u64,
    pub elapsed_ns: u64,
    pub global_enabled: bool,
    pub components: Vec<ProfileComponentSnapshot>,
    pub scopes: Vec<ProfileScopeSnapshot>,
    pub metrics: Vec<ProfileMetricSnapshot>,
}

impl ProfileSnapshot {
    pub fn elapsed_ms(&self) -> f64 {
        ns_to_ms(self.elapsed_ns)
    }

    pub fn metric_value(&self, metric: ProfileMetricId) -> Option<u64> {
        self.metrics
            .iter()
            .find(|snapshot| snapshot.id == metric)
            .map(ProfileMetricSnapshot::value)
    }

    pub fn enabled_scope_count(&self) -> usize {
        self.scopes.iter().filter(|scope| scope.enabled).count()
    }

    pub fn summary_line(&self) -> String {
        format!(
            "scdata/profile label={} round={} global={} elapsed_ms={:.3} components={} scopes={} enabled_scopes={} metrics={}",
            self.label,
            self.round,
            self.global_enabled,
            self.elapsed_ms(),
            self.components.len(),
            self.scopes.len(),
            self.enabled_scope_count(),
            self.metrics.len(),
        )
    }
}

impl fmt::Display for ProfileSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.summary_line())
    }
}

#[cfg(feature = "profile")]
pub(super) fn metric_snapshot(cell: &MetricCell, reset: bool) -> ProfileMetricSnapshot {
    ProfileMetricSnapshot {
        id: cell.id,
        value: cell.value.snapshot(reset),
    }
}
