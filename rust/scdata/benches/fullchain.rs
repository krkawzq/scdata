//! End-to-end scheduled-prefetch stress harness.
//!
//! Builds a file-backed CSR dataset, consumes it through
//! [`DataBank::prefetch_cells_scheduled`] while optional background CPU and IO
//! contention runs, and reports aggregate throughput plus per-batch wait
//! percentiles. Env-driven so it can run at varying scales:
//!
//! ```sh
//! SCDATA_FULLCHAIN_PREFETCH=1 \
//! cargo bench --manifest-path rust/scdata/Cargo.toml --bench fullchain
//! ```
//!
//! This is a re-implementation of the original fullchain harness: the same
//! surface (file-backed CSR, scheduled prefetch, bg CPU/IO contention,
//! percentile waits) with a slimmer, fully `support`-backed implementation.

mod support;

use std::hint::black_box;
use std::io::Cursor;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

use _scdata::access::ScheduledAccessConfig;
use _scdata::databank::{
    ArrayCodecMeta, ArrayMeta, ArrayOrder, ChunkStoreMeta, DType, DataBank, DataBankConfig,
    FileChunkLocation, ScheduledPrefetchConfig, SparseCsrDatasetMeta,
};
#[cfg(feature = "uring")]
use _scdata::iopool::UringConfig;
use _scdata::iopool::{BaseIoConfig, IoConfig, ThreadedConfig};
use support::chunks::{make_csr_u32_f32_chunked_raw, make_csr_u32_f32_chunks_lz4, write_chunks_file};
use support::codecs::crc32_encode;
use support::{bench_data_dir, env_flag, env_usize, payload, BenchConfig};

fn main() {
    let config = BenchConfig::from_env();
    let Some(fc) = FullchainConfig::from_env(config) else {
        println!(
            "databank/fullchain_scheduled_prefetch skipped; set SCDATA_FULLCHAIN_PREFETCH=1 to enable"
        );
        return;
    };

    println!(
        "databank/fullchain_scheduled_prefetch_config codec={} batches={} batch_cells={} genes={} nnz_per_cell={} chunk_nnz={} prefetch_step={} io_backend={} io_workers={} io_shards={} io_inflight={} scheduler_shards={} decode_workers={} bg_cpu_threads={} bg_io_readers={} bg_io_file_mib={} bg_io_read_kib={}",
        fc.codec.name(),
        fc.batches,
        fc.batch_cells,
        fc.genes,
        fc.nnz_per_cell,
        fc.chunk_nnz,
        fc.prefetch_step,
        fc.io_backend.name(),
        fc.io_workers,
        fc.io_shards,
        fc.io_inflight,
        fc.scheduler_shards,
        fc.decode_workers,
        fc.bg_cpu_threads,
        fc.bg_io_readers,
        fc.bg_io_file_mib,
        fc.bg_io_read_kib,
    );

    let csr = prepare_csr_file(&fc);
    println!(
        "databank/fullchain_file cells={} nnz={} chunks={} raw_mib={:.1} encoded_mib={:.1} encoded_over_raw={:.3}",
        csr.cells,
        csr.nnz,
        csr.total_chunks,
        csr.raw_bytes as f64 / (1024.0 * 1024.0),
        csr.encoded_bytes as f64 / (1024.0 * 1024.0),
        csr.encoded_bytes as f64 / csr.raw_bytes.max(1) as f64,
    );

    run_fullchain(&fc, &csr);
    let _ = std::fs::remove_file(&csr.indices_path);
    let _ = std::fs::remove_file(&csr.data_path);
}

// ---------------------------------------------------------------------------
// Configuration.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum FullchainCodec {
    Raw,
    Lz4,
    Zstd,
    Crc32,
}

impl FullchainCodec {
    fn from_env() -> Self {
        match env_str("SCDATA_FULLCHAIN_CODEC")
            .map(|value| value.to_ascii_lowercase())
            .as_deref()
        {
            Some("zstd") => Self::Zstd,
            Some("crc32") => Self::Crc32,
            Some("raw") | Some("none") | Some("uncompressed") => Self::Raw,
            _ => Self::Lz4,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Lz4 => "lz4",
            Self::Zstd => "zstd",
            Self::Crc32 => "crc32",
        }
    }

    fn array_meta(self) -> ArrayCodecMeta {
        match self {
            Self::Raw => ArrayCodecMeta::Uncompressed,
            Self::Lz4 => ArrayCodecMeta::CodecJson(r#"{"id":"lz4"}"#.to_string()),
            Self::Zstd => ArrayCodecMeta::CodecJson(r#"{"id":"zstd","level":3}"#.to_string()),
            Self::Crc32 => ArrayCodecMeta::CodecJson(r#"{"id":"crc32"}"#.to_string()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum FullchainIoBackend {
    Threaded,
    #[cfg(feature = "uring")]
    Uring,
}

impl FullchainIoBackend {
    fn from_env() -> Self {
        match env_str("SCDATA_FULLCHAIN_IO_BACKEND")
            .map(|value| value.to_ascii_lowercase())
            .as_deref()
        {
            #[cfg(feature = "uring")]
            Some("uring") => Self::Uring,
            _ => Self::Threaded,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Threaded => "threaded",
            #[cfg(feature = "uring")]
            Self::Uring => "uring",
        }
    }
}

#[derive(Debug)]
struct FullchainConfig {
    codec: FullchainCodec,
    batches: usize,
    batch_cells: usize,
    genes: usize,
    nnz_per_cell: usize,
    chunk_nnz: usize,
    prefetch_step: usize,
    access_prefetch_step: usize,
    access_decode_ahead: usize,
    access_ready_ahead: usize,
    io_backend: FullchainIoBackend,
    io_workers: usize,
    io_shards: usize,
    io_inflight: usize,
    #[cfg(feature = "uring")]
    uring_entries: u32,
    scheduler_shards: usize,
    access_cpu_workers: usize,
    decode_workers: usize,
    fill_workers: usize,
    bg_cpu_threads: usize,
    bg_io_readers: usize,
    bg_io_file_mib: usize,
    bg_io_read_kib: usize,
    warmups: usize,
}

impl FullchainConfig {
    fn from_env(config: BenchConfig) -> Option<Self> {
        if !env_flag("SCDATA_FULLCHAIN_PREFETCH") {
            return None;
        }
        let chunk_nnz = env_usize("SCDATA_FULLCHAIN_CHUNK_NNZ").unwrap_or(1024);
        Some(Self {
            codec: FullchainCodec::from_env(),
            batches: env_usize("SCDATA_FULLCHAIN_BATCHES").unwrap_or(64),
            batch_cells: env_usize("SCDATA_FULLCHAIN_BATCH_CELLS").unwrap_or(128),
            genes: env_usize("SCDATA_FULLCHAIN_GENES").unwrap_or(4096),
            nnz_per_cell: env_usize("SCDATA_FULLCHAIN_NNZ_PER_CELL").unwrap_or(64),
            chunk_nnz,
            prefetch_step: env_usize("SCDATA_FULLCHAIN_PREFETCH_STEP").unwrap_or(2),
            access_prefetch_step: env_usize("SCDATA_FULLCHAIN_ACCESS_PREFETCH_STEP").unwrap_or(4),
            access_decode_ahead: env_usize("SCDATA_FULLCHAIN_ACCESS_DECODE_AHEAD").unwrap_or(2),
            access_ready_ahead: env_usize("SCDATA_FULLCHAIN_ACCESS_READY_AHEAD").unwrap_or(1),
            io_backend: FullchainIoBackend::from_env(),
            io_workers: env_usize("SCDATA_FULLCHAIN_IO_WORKERS").unwrap_or(8),
            io_shards: env_usize("SCDATA_FULLCHAIN_IO_SHARDS").unwrap_or(2),
            io_inflight: env_usize("SCDATA_FULLCHAIN_IO_INFLIGHT").unwrap_or(256),
            #[cfg(feature = "uring")]
            uring_entries: env_usize("SCDATA_FULLCHAIN_URING_ENTRIES").unwrap_or(512) as u32,
            scheduler_shards: env_usize("SCDATA_FULLCHAIN_SCHEDULER_SHARDS").unwrap_or(2),
            access_cpu_workers: env_usize("SCDATA_FULLCHAIN_ACCESS_CPU_WORKERS").unwrap_or(4),
            decode_workers: env_usize("SCDATA_FULLCHAIN_DECODE_WORKERS").unwrap_or(8),
            fill_workers: env_usize("SCDATA_FULLCHAIN_FILL_WORKERS").unwrap_or(4),
            bg_cpu_threads: env_usize("SCDATA_FULLCHAIN_BG_CPU_THREADS").unwrap_or(0),
            bg_io_readers: env_usize("SCDATA_FULLCHAIN_BG_IO_READERS").unwrap_or(0),
            bg_io_file_mib: env_usize("SCDATA_FULLCHAIN_BG_IO_FILE_MIB").unwrap_or(64),
            bg_io_read_kib: env_usize("SCDATA_FULLCHAIN_BG_IO_READ_KIB").unwrap_or(64),
            warmups: config.warmups,
        })
    }
}

// ---------------------------------------------------------------------------
// CSR file preparation.
// ---------------------------------------------------------------------------

struct CsrFile {
    indptr: Vec<u64>,
    indices_path: PathBuf,
    indices_locations: Vec<FileChunkLocation>,
    data_path: PathBuf,
    data_locations: Vec<FileChunkLocation>,
    nnz: usize,
    total_chunks: usize,
    cells: usize,
    raw_bytes: u64,
    encoded_bytes: u64,
}

fn prepare_csr_file(fc: &FullchainConfig) -> CsrFile {
    let cells = fc.batch_cells * fc.batches;
    let (indptr, indices_chunks, data_chunks) = match fc.codec {
        FullchainCodec::Raw => make_csr_u32_f32_chunked_raw(
            cells,
            fc.genes,
            fc.nnz_per_cell,
            fc.chunk_nnz,
        ),
        FullchainCodec::Lz4 => make_csr_u32_f32_chunks_lz4(
            cells,
            fc.genes,
            fc.nnz_per_cell,
            fc.chunk_nnz,
        ),
        FullchainCodec::Zstd => {
            let (indptr, idx, data) =
                make_csr_u32_f32_chunked_raw(cells, fc.genes, fc.nnz_per_cell, fc.chunk_nnz);
            (indptr, zstd_encode_chunks(idx, 3), zstd_encode_chunks(data, 3))
        }
        FullchainCodec::Crc32 => {
            let (indptr, idx, data) =
                make_csr_u32_f32_chunked_raw(cells, fc.genes, fc.nnz_per_cell, fc.chunk_nnz);
            (indptr, crc_wrap_chunks(idx), crc_wrap_chunks(data))
        }
    };

    let raw_bytes = (cells * fc.nnz_per_cell) as u64
        * (std::mem::size_of::<u32>() + std::mem::size_of::<f32>()) as u64;
    let encoded_bytes = indices_chunks
        .iter()
        .chain(data_chunks.iter())
        .map(|chunk| chunk.len() as u64)
        .sum();

    let (indices_path, indices_locations) =
        write_chunks_file("fullchain-indices", &indices_chunks);
    let (data_path, data_locations) = write_chunks_file("fullchain-data", &data_chunks);

    CsrFile {
        indptr,
        indices_path,
        indices_locations,
        data_path,
        data_locations,
        nnz: cells * fc.nnz_per_cell,
        total_chunks: indices_chunks.len(),
        cells,
        raw_bytes,
        encoded_bytes,
    }
}

fn zstd_encode_chunks(chunks: Vec<Arc<[u8]>>, level: i32) -> Vec<Arc<[u8]>> {
    chunks
        .into_iter()
        .map(|chunk| {
            Arc::from(
                zstd::encode_all(Cursor::new(&chunk[..]), level)
                    .expect("zstd encode")
                    .into_boxed_slice(),
            )
        })
        .collect()
}

fn crc_wrap_chunks(chunks: Vec<Arc<[u8]>>) -> Vec<Arc<[u8]>> {
    chunks
        .into_iter()
        .map(|chunk| Arc::from(crc32_encode(&chunk).into_boxed_slice()))
        .collect()
}

// ---------------------------------------------------------------------------
// Fullchain run: register, (optional) bg contention, prefetch, consume, report.
// ---------------------------------------------------------------------------

fn run_fullchain(fc: &FullchainConfig, csr: &CsrFile) {
    let mut bank = DataBank::new(fullchain_databank_config(fc)).expect("fullchain bank");
    let dataset_id = bank
        .register_sparse_csr(SparseCsrDatasetMeta {
            gene_names: (0..fc.genes).map(|idx| format!("gene_{idx}")).collect(),
            indptr: csr.indptr.clone(),
            indices: ArrayMeta {
                shape: vec![csr.nnz],
                chunk_shape: vec![fc.chunk_nnz],
                chunk_grid_shape: vec![csr.total_chunks],
                dtype: DType::U32,
                order: ArrayOrder::C,
                codec: fc.codec.array_meta(),
                chunks: ChunkStoreMeta::FileOffset {
                    path: csr.indices_path.clone(),
                    locations: csr.indices_locations.clone(),
                },
            },
            data: ArrayMeta {
                shape: vec![csr.nnz],
                chunk_shape: vec![fc.chunk_nnz],
                chunk_grid_shape: vec![csr.total_chunks],
                dtype: DType::F32,
                order: ArrayOrder::C,
                codec: fc.codec.array_meta(),
                chunks: ChunkStoreMeta::FileOffset {
                    path: csr.data_path.clone(),
                    locations: csr.data_locations.clone(),
                },
            },
            index_dtype: DType::U32,
            num_cells: csr.cells,
            num_genes: fc.genes,
        })
        .expect("register fullchain csr dataset");

    let batches = make_batches(fc);
    let sp_config = ScheduledPrefetchConfig {
        prefetch_step: fc.prefetch_step,
        access: ScheduledAccessConfig {
            prefetch_step: fc.access_prefetch_step,
            decode_ahead_steps: fc.access_decode_ahead,
            ready_ahead_steps: fc.access_ready_ahead,
        },
    };

    // Warm-up pass (no bg contention, no timing).
    for _ in 0..fc.warmups {
        let prefetch = bank
            .prefetch_cells_scheduled::<f32, _>(dataset_id, batches.clone(), sp_config)
            .expect("warmup prefetch");
        for batch in prefetch {
            let batch = batch.expect("warmup batch");
            black_box(batch.buffer.as_ptr());
        }
    }

    let bg = FullchainBackground::start(fc);
    let started = Instant::now();
    let mut waits: Vec<u64> = Vec::with_capacity(batches.len());
    let mut total_cells = 0usize;
    let mut total_values = 0usize;

    let prefetch = bank
        .prefetch_cells_scheduled::<f32, _>(dataset_id, batches, sp_config)
        .expect("fullchain prefetch");
    for batch_result in prefetch {
        let wait_start = Instant::now();
        let batch = batch_result.expect("prefetch batch");
        waits.push(wait_start.elapsed().as_nanos() as u64);
        total_cells += batch.cells.len();
        total_values += batch.buffer.len();
        black_box(batch.buffer.as_ptr());
    }
    let elapsed = started.elapsed();
    let bg_sum = bg.stop();

    waits.sort_unstable();
    let p50 = percentile(&waits, 500);
    let p99 = percentile(&waits, 990);
    let p999 = percentile(&waits, 999);
    let max_ns = *waits.last().unwrap_or(&0);
    let seconds = elapsed.as_secs_f64();
    let value_mib = total_values as f64 * std::mem::size_of::<f32>() as f64 / (1024.0 * 1024.0);

    println!(
        "databank/fullchain_scheduled_prefetch    batches={} cells={} values={} elapsed_s={seconds:.4} throughput_mib_s={:.1} kops={:.1} wait_ns_p50={p50} p99={p99} p999={p999} max={max_ns} bg_sum={bg_sum}",
        waits.len(),
        total_cells,
        total_values,
        value_mib / seconds,
        waits.len() as f64 / seconds / 1000.0,
    );
}

fn make_batches(fc: &FullchainConfig) -> Vec<Vec<usize>> {
    (0..fc.batches)
        .map(|batch| {
            let base = batch * fc.batch_cells;
            (0..fc.batch_cells).map(|i| base + i).collect()
        })
        .collect()
}

fn percentile(sorted: &[u64], permille: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (sorted.len() - 1) * permille / 1000;
    sorted[idx]
}

// ---------------------------------------------------------------------------
// Background CPU / IO contention.
// ---------------------------------------------------------------------------

struct FullchainBackground {
    stop: Arc<AtomicBool>,
    handles: Vec<JoinHandle<u64>>,
}

impl FullchainBackground {
    fn start(fc: &FullchainConfig) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles: Vec<JoinHandle<u64>> = Vec::new();

        for thread_idx in 0..fc.bg_cpu_threads {
            let stop = Arc::clone(&stop);
            handles.push(thread::spawn(move || run_bg_cpu(stop, thread_idx)));
        }

        if fc.bg_io_readers > 0 {
            let bg_path = bench_data_dir().join(format!("fullchain-bg-{}.bin", std::process::id()));
            std::fs::write(&bg_path, payload(fc.bg_io_file_mib * 1024 * 1024))
                .expect("write bg io file");
            for reader_idx in 0..fc.bg_io_readers {
                let stop = Arc::clone(&stop);
                let path = bg_path.clone();
                let kib = fc.bg_io_read_kib;
                handles.push(thread::spawn(move || run_bg_io(stop, &path, kib, reader_idx)));
            }
            // The bg file is removed after the run joins; hand it off via a
            // leak-guarded drop on stop. We remove it after join below.
            let stop_for_cleanup = Arc::clone(&stop);
            let cleanup_path = bg_path.clone();
            handles.push(thread::spawn(move || {
                // Park until told to stop, then clean up the bg file.
                while !stop_for_cleanup.load(Ordering::Relaxed) {
                    thread::sleep(std::time::Duration::from_millis(50));
                }
                let _ = std::fs::remove_file(&cleanup_path);
                0
            }));
        }

        Self { stop, handles }
    }

    fn stop(self) -> u64 {
        self.stop.store(true, Ordering::Release);
        self.handles
            .into_iter()
            .map(|handle| handle.join().unwrap_or(0))
            .sum()
    }
}

fn run_bg_cpu(stop: Arc<AtomicBool>, thread_idx: usize) -> u64 {
    let mut acc = thread_idx as u64;
    while !stop.load(Ordering::Relaxed) {
        acc = acc.wrapping_add(black_box(acc.rotate_left(3)).wrapping_mul(0x9e37_79b9));
    }
    acc
}

fn run_bg_io(stop: Arc<AtomicBool>, path: &Path, kib: usize, reader_idx: usize) -> u64 {
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(_) => return 0,
    };
    let mut buf = vec![0u8; kib.max(1) * 1024];
    let file_len = file
        .metadata()
        .map(|metadata| metadata.len() as usize)
        .unwrap_or(buf.len());
    if file_len == 0 {
        return 0;
    }
    let mut pos = (reader_idx * 4096) % file_len;
    let mut total = 0u64;
    while !stop.load(Ordering::Relaxed) {
        if pos + buf.len() > file_len {
            pos = 0;
        }
        match file.read_at(&mut buf, pos as u64) {
            Ok(read) => total += read as u64,
            Err(_) => break,
        }
        pos = (pos + buf.len()) % file_len;
    }
    total
}

// ---------------------------------------------------------------------------
// DataBank / IO configuration.
// ---------------------------------------------------------------------------

fn fullchain_databank_config(fc: &FullchainConfig) -> DataBankConfig {
    let mut config = DataBankConfig::default();
    config.io_config = fullchain_io_config(fc);
    config.decode_config.num_workers = fc.decode_workers;
    config.access_config.scheduler_shards = fc.scheduler_shards;
    config.access_config.cpu.num_workers = fc.access_cpu_workers;
    config.access_config.cache_capacity_bytes = 64 * 1024 * 1024;
    config.access_config.memory_budget_bytes = 256 * 1024 * 1024;
    config.fill_config.num_workers = fc.fill_workers;
    config
}

fn fullchain_io_config(fc: &FullchainConfig) -> IoConfig {
    let base = BaseIoConfig {
        max_in_flight: fc.io_inflight,
        priority_levels: 2,
        queue_shards: fc.io_shards,
        assume_non_overlapping_reads: true,
    };
    match fc.io_backend {
        FullchainIoBackend::Threaded => IoConfig::Threaded(ThreadedConfig {
            base,
            num_workers: fc.io_workers,
            cpus: None,
        }),
        #[cfg(feature = "uring")]
        FullchainIoBackend::Uring => IoConfig::Uring(UringConfig {
            base,
            entries: fc.uring_entries,
            drivers: 1,
            iowq_bounded_workers: 0,
            iowq_unbounded_workers: 0,
            registered_files: 64,
        }),
    }
}

fn env_str(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}
