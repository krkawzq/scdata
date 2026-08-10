#![cfg(all(target_os = "linux", target_has_atomic = "64"))]

use std::collections::HashMap;
use std::hint::black_box;
use std::os::fd::AsFd;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use sc_compress::{
    ByteStore, DenseMatrix, DenseWriter, Error as CompressError, Partition,
    Result as CompressResult,
};
use scdata::{
    compile, Dataset, Fill, IoMode, OutputDType, OutputSpec, Plan, PlanSpec, RowRef, SessionConfig,
    SharedClient, SharedConfig, Source, SourceId,
};

const DEFAULT_ROWS: usize = 8_192;
const DEFAULT_COLUMNS: usize = 1;
const DEFAULT_BATCH_SIZE: usize = 1;
const DEFAULT_PREFETCH_STEP: usize = 64;
const DEFAULT_WORLD_SIZE: usize = 4;
const DEFAULT_ITERATIONS: usize = 20;
const WARMUP_ITERATIONS: usize = 3;

#[derive(Clone, Copy)]
enum BenchMode {
    Both,
    Shared,
    Standard,
}

impl BenchMode {
    fn from_environment() -> Self {
        match std::env::var("SCDATA_BENCH_MODE").as_deref() {
            Ok("shared") => Self::Shared,
            Ok("standard") => Self::Standard,
            Ok("both") | Err(_) => Self::Both,
            Ok(value) => panic!("SCDATA_BENCH_MODE must be standard, shared, or both; got {value}"),
        }
    }
}

struct MemoryStore {
    values: HashMap<String, Arc<[u8]>>,
}

impl MemoryStore {
    fn from_directory(path: &std::path::Path) -> Self {
        let values = ["meta.json", "data/0"]
            .into_iter()
            .map(|key| {
                let bytes = std::fs::read(path.join(key)).unwrap();
                (key.to_owned(), Arc::<[u8]>::from(bytes))
            })
            .collect();
        Self { values }
    }

    fn value(&self, key: &str) -> CompressResult<&[u8]> {
        self.values
            .get(key)
            .map(AsRef::as_ref)
            .ok_or_else(|| CompressError::NotFound {
                key: key.to_owned(),
            })
    }
}

impl ByteStore for MemoryStore {
    fn len(&self, key: &str) -> CompressResult<u64> {
        u64::try_from(self.value(key)?.len())
            .map_err(|_| CompressError::InvalidArgument("benchmark value exceeds u64".into()))
    }

    fn read_range(&self, key: &str, offset: u64, len: usize) -> CompressResult<Vec<u8>> {
        let value = self.value(key)?;
        let start = usize::try_from(offset).map_err(|_| {
            CompressError::InvalidArgument("benchmark range offset exceeds usize".into())
        })?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| CompressError::InvalidArgument("benchmark range end overflow".into()))?;
        let bytes = value
            .get(start..end)
            .ok_or_else(|| CompressError::CorruptData {
                context: "benchmark store".into(),
                message: format!("range [{start}, {end}) exceeds {} bytes", value.len()),
            })?;
        Ok(bytes.to_vec())
    }

    fn exists(&self, key: &str) -> CompressResult<bool> {
        Ok(self.values.contains_key(key))
    }

    fn supports_efficient_range_reads(&self, _key: &str) -> CompressResult<bool> {
        Ok(true)
    }
}

#[derive(Clone, Copy)]
struct BenchConfig {
    rows: usize,
    columns: usize,
    batch_size: usize,
    prefetch_step: usize,
    world_size: usize,
    iterations: usize,
    mode: BenchMode,
}

impl BenchConfig {
    fn from_environment() -> Self {
        Self {
            rows: environment_usize("SCDATA_BENCH_ROWS", DEFAULT_ROWS),
            columns: environment_usize("SCDATA_BENCH_COLUMNS", DEFAULT_COLUMNS),
            batch_size: environment_usize("SCDATA_BENCH_BATCH_SIZE", DEFAULT_BATCH_SIZE),
            prefetch_step: environment_usize("SCDATA_BENCH_PREFETCH_STEP", DEFAULT_PREFETCH_STEP),
            world_size: environment_usize("SCDATA_BENCH_WORLD_SIZE", DEFAULT_WORLD_SIZE),
            iterations: environment_usize("SCDATA_BENCH_ITERATIONS", DEFAULT_ITERATIONS),
            mode: BenchMode::from_environment(),
        }
    }
}

fn environment_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("{name} must be a positive integer"))
        })
        .unwrap_or(default)
        .max(1)
}

fn session_config(worker_count: usize) -> SessionConfig {
    let mut config = SessionConfig::default();
    config.worker_count = worker_count;
    config.io_mode = IoMode::Blocking;
    config.max_total_inflight_encoded_bytes = config
        .max_inflight_encoded_bytes_per_worker
        .saturating_mul(worker_count);
    config.max_total_decoded_bytes = config
        .max_decoded_bytes_per_worker
        .saturating_mul(worker_count);
    config.max_total_inflight_io_ops = config.max_total_inflight_io_ops.max(worker_count);
    config
}

fn make_plan(config: BenchConfig) -> (tempfile::TempDir, Plan) {
    assert!(config.prefetch_step >= 2);
    let value_count = config.rows.checked_mul(config.columns).unwrap();
    let values = (0..value_count)
        .map(|value| u32::try_from(value).unwrap())
        .collect::<Vec<_>>();
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("shared-ring-benchmark");
    DenseWriter::new(&path)
        .chunk(Partition::fixed_cells(u64::try_from(config.rows).unwrap()))
        .block(Partition::fixed_cells(1))
        .threads(1)
        .write(
            &values,
            [
                u64::try_from(config.rows).unwrap(),
                u64::try_from(config.columns).unwrap(),
            ],
        )
        .unwrap();

    let store = Arc::new(MemoryStore::from_directory(&path));
    let dataset = Dataset::from_dense(DenseMatrix::from_store(store).unwrap());
    let source_id = SourceId::new(1);
    let rows = (0..config.rows)
        .map(|row| RowRef::new(source_id, u64::try_from(row).unwrap()))
        .collect();
    let output = OutputSpec::new(config.columns, OutputDType::U32, Fill::U32(0)).unwrap();
    let plan = compile(PlanSpec::new(
        vec![Source::new(source_id, dataset)],
        rows,
        output,
        config.batch_size,
        config.prefetch_step,
    ))
    .unwrap();
    (temporary, plan)
}

fn run_standard(plan: &Plan, worker_count: usize) -> usize {
    let mut session = plan.open(session_config(worker_count)).unwrap();
    let mut rows = 0usize;
    while let Some(batch) = session.next_batch().unwrap() {
        rows += black_box(batch.rows());
    }
    rows
}

fn run_shared(plan: &Plan, world_size: usize) -> usize {
    let server = plan
        .open_shared(
            session_config(world_size),
            SharedConfig::new(world_size).unwrap(),
        )
        .unwrap();
    let descriptors = (0..world_size)
        .map(|_| server.attach_fd().unwrap())
        .collect::<Vec<_>>();
    let producer = thread::spawn(move || server.run().unwrap());
    let consumers = descriptors
        .into_iter()
        .enumerate()
        .map(|(rank, descriptor)| {
            thread::spawn(move || {
                let mut client = SharedClient::attach(descriptor.as_fd(), rank).unwrap();
                let mut rows = 0usize;
                while let Some(batch) = client.next_batch().unwrap() {
                    rows += black_box(batch.rows());
                    batch.release().unwrap();
                }
                rows
            })
        })
        .collect::<Vec<_>>();
    let rows = consumers
        .into_iter()
        .map(|consumer| consumer.join().unwrap())
        .sum();
    producer.join().unwrap();
    rows
}

fn measure(
    iterations: usize,
    expected_rows: usize,
    mut run: impl FnMut() -> usize,
) -> Vec<Duration> {
    for _ in 0..WARMUP_ITERATIONS {
        assert_eq!(run(), expected_rows);
    }
    (0..iterations)
        .map(|_| {
            let started = Instant::now();
            assert_eq!(run(), expected_rows);
            started.elapsed()
        })
        .collect()
}

fn report(name: &str, batch_count: usize, mut samples: Vec<Duration>) {
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    let p95 = samples[(samples.len() * 95 / 100).min(samples.len() - 1)];
    let batches_per_second = batch_count as f64 / median.as_secs_f64();
    println!(
        "{name}: median={:.3} ms p95={:.3} ms throughput={batches_per_second:.0} batches/s",
        median.as_secs_f64() * 1_000.0,
        p95.as_secs_f64() * 1_000.0,
    );
}

fn main() {
    let config = BenchConfig::from_environment();
    let (_temporary, plan) = make_plan(config);
    let batch_count = plan.batch_count();
    println!(
        "rows={} columns={} batches={} batch_size={} prefetch_step={} world_size={} iterations={}",
        config.rows,
        config.columns,
        batch_count,
        config.batch_size,
        config.prefetch_step,
        config.world_size,
        config.iterations,
    );
    if matches!(config.mode, BenchMode::Both | BenchMode::Standard) {
        report(
            "standard",
            batch_count,
            measure(config.iterations, config.rows, || {
                run_standard(&plan, config.world_size)
            }),
        );
    }
    if matches!(config.mode, BenchMode::Both | BenchMode::Shared) {
        report(
            "shared",
            batch_count,
            measure(config.iterations, config.rows, || {
                run_shared(&plan, config.world_size)
            }),
        );
    }
}
