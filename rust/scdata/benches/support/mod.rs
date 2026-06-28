//! Shared benchmark infrastructure: timing harness, multi-threaded driver,
//! environment helpers, and payload generators.
//!
//! Each `[[bench]]` target declares `mod support;` and reuses the helpers
//! here instead of carrying its own copy of the encode / chunk / mock-backend
//! plumbing.
//!
//! `#![allow(dead_code)]` suppresses per-target unused warnings: every bench
//! compiles `mod support` as its own crate, so a helper used only by the
//! `stress` target looks dead to the `modules` target. The whole module is the
//! shared bench toolbox, so unused-within-one-target members are expected.

#![allow(dead_code)]

pub mod backends;
pub mod chunks;
pub mod codecs;
pub mod data;
pub mod manifest;

use std::hint::black_box;
use std::path::PathBuf;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

/// Tunables parsed from the environment.
///
/// - `SCDATA_BENCH_WARMUPS` — warm-up rounds before timing (default 3).
/// - `SCDATA_BENCH_ITERS`   — override per-benchmark iteration counts.
#[derive(Debug, Clone, Copy)]
pub struct BenchConfig {
    pub warmups: usize,
}

impl BenchConfig {
    pub fn from_env() -> Self {
        Self {
            warmups: env_usize("SCDATA_BENCH_WARMUPS").unwrap_or(3),
        }
    }

    pub fn iterations(self, default: usize) -> usize {
        env_usize("SCDATA_BENCH_ITERS").unwrap_or(default).max(1)
    }
}

/// Time a single-threaded closure, reporting ns/iter and optional throughput.
///
/// The closure returns a `usize` checksum that is folded into a black-box sink
/// so the optimizer cannot elide the work.
pub fn bench(
    config: BenchConfig,
    name: &str,
    default_iters: usize,
    bytes_per_iter: Option<usize>,
    mut body: impl FnMut() -> usize,
) {
    let iterations = config.iterations(default_iters);
    for _ in 0..config.warmups {
        black_box(body());
    }

    let started = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..iterations {
        checksum ^= black_box(body());
    }
    let elapsed = started.elapsed();
    black_box(checksum);

    let seconds = elapsed.as_secs_f64();
    let ns_per_iter = seconds * 1_000_000_000.0 / iterations as f64;
    match bytes_per_iter {
        Some(bytes) => {
            let mib = bytes as f64 * iterations as f64 / (1024.0 * 1024.0);
            println!(
                "{name:<42} iter={iterations:<7} elapsed_s={seconds:.6} ns_iter={ns_per_iter:.1} throughput_mib_s={:.1}",
                mib / seconds
            );
        }
        None => {
            println!(
                "{name:<42} iter={iterations:<7} elapsed_s={seconds:.6} ns_iter={ns_per_iter:.1}"
            );
        }
    }
}

/// Multi-threaded stress driver.
///
/// Spawns `threads` workers behind a barrier, runs `body(thread_idx)` per
/// iteration, and reports aggregate throughput plus thread scaling. `body`
/// receives its thread index so per-thread offsets can avoid false sharing.
pub fn stress_mt<F>(
    config: BenchConfig,
    name: &str,
    threads: usize,
    default_iters: usize,
    bytes_per_op: Option<usize>,
    body: F,
) where
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
    let start = Instant::now();
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let body = Arc::clone(&body);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let mut sum = 0usize;
                for _ in 0..iters_per_thread {
                    sum ^= black_box(body(t));
                }
                sum
            })
        })
        .collect();
    let checksum: usize = handles
        .into_iter()
        .map(|h| h.join().expect("stress thread"))
        .fold(0, |a, b| a ^ b);
    black_box(checksum);

    let elapsed = start.elapsed();
    let total_ops = iters_per_thread * threads;
    let seconds = elapsed.as_secs_f64();
    let ns_per_op = seconds * 1_000_000_000.0 / total_ops as f64;
    let kops = total_ops as f64 / seconds / 1000.0;
    match bytes_per_op {
        Some(b) => {
            let mib = b as f64 * total_ops as f64 / (1024.0 * 1024.0);
            println!(
                "{name:<50} threads={threads:<3} ops={total_ops:<9} elapsed_s={seconds:.4} ns_op={ns_per_op:.1} throughput_mib_s={:.1} kops={kops:.1}",
                mib / seconds
            );
        }
        None => println!(
            "{name:<50} threads={threads:<3} ops={total_ops:<9} elapsed_s={seconds:.4} ns_op={ns_per_op:.1} kops={kops:.1}"
        ),
    }
}

/// Deterministic pseudo-random payload with mixed entropy and periodic zero
/// runs (every 17th byte). Exercises both compressible and incompressible
/// regions of every codec under test.
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

/// Scratch directory for bench-generated data files, under the crate's
/// `target/bench-data` so it stays out of the source tree and survives across
/// bench targets in one run.
pub fn bench_data_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("bench-data");
    std::fs::create_dir_all(&dir).expect("create bench data dir");
    dir
}

pub fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

pub fn env_u32(name: &str) -> Option<u32> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
}

pub fn env_i32(name: &str) -> Option<i32> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
}

pub fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => !matches!(value.trim().to_ascii_lowercase().as_str(), "" | "0" | "false" | "no" | "off"),
        Err(_) => false,
    }
}
