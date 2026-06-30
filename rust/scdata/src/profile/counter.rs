#[cfg(feature = "profile")]
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[cfg(feature = "profile")]
#[derive(Debug, Default)]
pub(super) struct AtomicCounter {
    value: AtomicU64,
}

#[cfg(feature = "profile")]
impl AtomicCounter {
    pub(super) fn add(&self, value: u64) {
        self.value.fetch_add(value, Ordering::Relaxed);
    }

    pub(super) fn set(&self, value: u64) {
        self.value.store(value, Ordering::Relaxed);
    }

    pub(super) fn load(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    pub(super) fn snapshot(&self, reset: bool) -> CounterSnapshot {
        let value = if reset {
            self.value.swap(0, Ordering::Relaxed)
        } else {
            self.load()
        };
        CounterSnapshot { value }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CounterSnapshot {
    pub value: u64,
}

impl CounterSnapshot {
    pub const fn new(value: u64) -> Self {
        Self { value }
    }

    pub const fn value(&self) -> u64 {
        self.value
    }
}

pub fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

pub fn ns_to_ms(ns: u64) -> f64 {
    ns as f64 / 1_000_000.0
}

pub fn bytes_to_mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

pub fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

pub fn avg_ns(total_ns: u64, count: u64) -> f64 {
    if count == 0 {
        0.0
    } else {
        total_ns as f64 / count as f64
    }
}

pub fn per_second(units: u64, elapsed_ns: u64) -> f64 {
    if elapsed_ns == 0 {
        0.0
    } else {
        units as f64 / (elapsed_ns as f64 / 1_000_000_000.0)
    }
}
