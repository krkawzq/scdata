//! Shared benchmark infrastructure.
//!
//! Bench targets are compiled as independent crates, so this module deliberately
//! keeps a broad toolbox and allows per-target dead code.

#![allow(dead_code)]

pub mod backends;
pub mod chunks;
pub mod codecs;
pub mod data;

use std::env;
use std::hint::black_box;
use std::path::PathBuf;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

use _scdata::profile::{
    bytes_to_mib, ns_to_ms, ProfileConfig, ProfileMetricKind, ProfileRegistry, ProfileRuleSet,
    ProfileRuntime, ProfileSnapshot,
};

#[derive(Debug, Clone, Copy)]
pub struct BenchConfig {
    pub warmups: usize,
    pub emit_profile: bool,
}

impl BenchConfig {
    pub fn from_env() -> Self {
        Self {
            warmups: env_usize("SCDATA_BENCH_WARMUPS").unwrap_or(3),
            emit_profile: env_flag_default("SCDATA_BENCH_EMIT_PROFILE", true),
        }
    }

    pub fn iterations(self, default: usize) -> usize {
        env_usize("SCDATA_BENCH_ITERS").unwrap_or(default).max(1)
    }
}

#[derive(Debug, Clone)]
pub struct BenchReport {
    pub name: String,
    pub iterations: usize,
    pub elapsed_ns: u64,
    pub bytes_per_iter: Option<usize>,
}

impl BenchReport {
    pub fn elapsed_s(&self) -> f64 {
        self.elapsed_ns as f64 / 1_000_000_000.0
    }

    pub fn ns_per_iter(&self) -> f64 {
        self.elapsed_ns as f64 / self.iterations as f64
    }

    pub fn throughput_mib_s(&self) -> Option<f64> {
        self.bytes_per_iter.map(|bytes| {
            bytes_to_mib((bytes as u64).saturating_mul(self.iterations as u64)) / self.elapsed_s()
        })
    }

    pub fn print(&self) {
        match self.throughput_mib_s() {
            Some(mib_s) => println!(
                "{:<54} iter={:<8} elapsed_s={:.6} ns_iter={:.1} throughput_mib_s={:.1}",
                self.name,
                self.iterations,
                self.elapsed_s(),
                self.ns_per_iter(),
                mib_s
            ),
            None => println!(
                "{:<54} iter={:<8} elapsed_s={:.6} ns_iter={:.1}",
                self.name,
                self.iterations,
                self.elapsed_s(),
                self.ns_per_iter()
            ),
        }
    }
}

pub fn bench(
    config: BenchConfig,
    name: &str,
    default_iters: usize,
    bytes_per_iter: Option<usize>,
    mut body: impl FnMut() -> usize,
) -> BenchReport {
    let iterations = config.iterations(default_iters);
    for _ in 0..config.warmups {
        black_box(body());
    }

    let started = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..iterations {
        checksum ^= black_box(body());
    }
    black_box(checksum);

    let report = BenchReport {
        name: name.to_string(),
        iterations,
        elapsed_ns: duration_ns(started.elapsed()),
        bytes_per_iter,
    };
    report.print();
    report
}

pub fn bench_profiled(
    config: BenchConfig,
    name: &str,
    default_iters: usize,
    bytes_per_iter: Option<usize>,
    runtime: &ProfileRuntime,
    mut body: impl FnMut() -> usize,
) -> ProfileSnapshot {
    for _ in 0..config.warmups {
        let round = runtime.start();
        black_box(body());
        black_box(round.end());
    }

    let iterations = config.iterations(default_iters);
    let round = runtime.start();
    let started = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..iterations {
        checksum ^= black_box(body());
    }
    black_box(checksum);
    let snapshot = round.end();

    let report = BenchReport {
        name: name.to_string(),
        iterations,
        elapsed_ns: duration_ns(started.elapsed()),
        bytes_per_iter,
    };
    report.print();
    if config.emit_profile {
        print_profile_snapshot(name, &snapshot);
    }
    snapshot
}

pub fn bench_existing_profile_round(
    config: BenchConfig,
    name: &str,
    default_iters: usize,
    bytes_per_iter: Option<usize>,
    runtime: &ProfileRuntime,
    mut body: impl FnMut() -> usize,
) -> ProfileSnapshot {
    for _ in 0..config.warmups {
        runtime.reset_metrics();
        black_box(body());
        black_box(runtime.snapshot_and_reset());
    }

    runtime.reset_metrics();
    let iterations = config.iterations(default_iters);
    let started = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..iterations {
        checksum ^= black_box(body());
    }
    black_box(checksum);
    let snapshot = runtime.snapshot_and_reset();

    let report = BenchReport {
        name: name.to_string(),
        iterations,
        elapsed_ns: duration_ns(started.elapsed()),
        bytes_per_iter,
    };
    report.print();
    if config.emit_profile {
        print_profile_snapshot(name, &snapshot);
    }
    snapshot
}

pub fn stress_mt<F>(
    config: BenchConfig,
    name: &str,
    threads: usize,
    default_iters: usize,
    bytes_per_op: Option<usize>,
    body: F,
) -> BenchReport
where
    F: Fn(usize) -> usize + Send + Sync + 'static,
{
    let threads = threads.max(1);
    let total_iters = config.iterations(default_iters);
    let iters_per_thread = (total_iters / threads).max(1);
    let body = Arc::new(body);
    for _ in 0..config.warmups {
        black_box(body(0));
    }

    let barrier = Arc::new(Barrier::new(threads));
    let started = Instant::now();
    let handles = (0..threads)
        .map(|thread_idx| {
            let body = Arc::clone(&body);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let mut checksum = 0usize;
                for _ in 0..iters_per_thread {
                    checksum ^= black_box(body(thread_idx));
                }
                checksum
            })
        })
        .collect::<Vec<_>>();

    let checksum = handles
        .into_iter()
        .map(|handle| handle.join().expect("bench worker"))
        .fold(0usize, |acc, item| acc ^ item);
    black_box(checksum);

    let report = BenchReport {
        name: format!("{name}/t{threads}"),
        iterations: iters_per_thread * threads,
        elapsed_ns: duration_ns(started.elapsed()),
        bytes_per_iter: bytes_per_op,
    };
    report.print();
    report
}

pub fn profile_runtime(
    label: impl Into<String>,
    registry: impl FnOnce() -> ProfileRegistry,
) -> ProfileRuntime {
    ProfileRuntime::new_lazy(
        ProfileConfig::enabled(label)
            .with_components(ProfileRuleSet::all())
            .with_scopes(ProfileRuleSet::all()),
        registry,
    )
}

pub fn selected_scope_profile_runtime(
    label: impl Into<String>,
    scopes: impl IntoIterator<Item = impl Into<String>>,
    registry: impl FnOnce() -> ProfileRegistry,
) -> ProfileRuntime {
    let mut rules = ProfileRuleSet::none();
    for scope in scopes {
        rules.enable(scope);
    }
    ProfileRuntime::new_lazy(
        ProfileConfig::enabled(label)
            .with_components(ProfileRuleSet::all())
            .with_scopes(rules),
        registry,
    )
}

pub fn disabled_profile_runtime(registry: impl FnOnce() -> ProfileRegistry) -> ProfileRuntime {
    ProfileRuntime::new_lazy(ProfileConfig::disabled(), registry)
}

pub fn print_profile_snapshot(case: &str, snapshot: &ProfileSnapshot) {
    println!("profile/{case} {}", snapshot.summary_line());
    for metric in snapshot.metrics.iter().filter(|metric| metric.value() > 0) {
        println!(
            "  {}",
            format_metric(
                metric.id.scope.full_name(),
                metric.id.name,
                metric.kind(),
                metric.value()
            )
        );
    }
}

pub fn maybe_print_profile_snapshot(config: BenchConfig, case: &str, snapshot: &ProfileSnapshot) {
    if config.emit_profile {
        print_profile_snapshot(case, snapshot);
    }
}

pub fn metric_value(snapshot: &ProfileSnapshot, scope: &str, name: &str) -> u64 {
    snapshot
        .metrics
        .iter()
        .find(|metric| metric.id.scope.full_name() == scope && metric.id.name == name)
        .map(|metric| metric.value())
        .unwrap_or(0)
}

fn format_metric(scope: String, name: &str, kind: ProfileMetricKind, value: u64) -> String {
    let key = format!("{scope}.{name}").replace(['.', '-'], "_");
    match kind {
        ProfileMetricKind::DurationNs => format!("{key}_ms={:.3}", ns_to_ms(value)),
        ProfileMetricKind::Bytes => format!("{key}_mib={:.3}", bytes_to_mib(value)),
        ProfileMetricKind::Count | ProfileMetricKind::Gauge | ProfileMetricKind::Custom(_) => {
            format!("{key}={value}")
        }
        _ => format!("{key}={value}"),
    }
}

pub struct ProfileEnvGuard {
    saved: Vec<(&'static str, Option<String>)>,
}

impl ProfileEnvGuard {
    pub fn enabled(label: &str) -> Self {
        let keys = [
            "SCDATA_PROFILE",
            "SCDATA_PROFILE_LABEL",
            "SCDATA_PROFILE_COMPONENTS",
            "SCDATA_PROFILE_SCOPES",
        ];
        let saved = keys
            .into_iter()
            .map(|key| (key, env::var(key).ok()))
            .collect::<Vec<_>>();
        env::set_var("SCDATA_PROFILE", "1");
        env::set_var("SCDATA_PROFILE_LABEL", label);
        env::set_var("SCDATA_PROFILE_COMPONENTS", "all");
        env::set_var("SCDATA_PROFILE_SCOPES", "all");
        Self { saved }
    }
}

impl Drop for ProfileEnvGuard {
    fn drop(&mut self) {
        for (key, value) in self.saved.drain(..) {
            match value {
                Some(value) => env::set_var(key, value),
                None => env::remove_var(key),
            }
        }
    }
}

pub fn payload(len: usize) -> Vec<u8> {
    (0..len)
        .map(|idx| {
            let mixed = (idx as u64)
                .wrapping_mul(0x9e37_79b9)
                .rotate_left((idx % 31) as u32)
                ^ ((idx / 4096) as u64);
            if idx % 17 == 0 {
                0
            } else {
                (mixed & 0xff) as u8
            }
        })
        .collect()
}

pub fn bench_data_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("bench-data");
    std::fs::create_dir_all(&dir).expect("create bench data dir");
    dir
}

pub fn env_usize(name: &str) -> Option<usize> {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

pub fn env_flag_default(name: &str, default: bool) -> bool {
    match env::var(name) {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        ),
        Err(_) => default,
    }
}

fn duration_ns(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}
