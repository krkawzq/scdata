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
//! Matrix runs add `SCDATA_FULLCHAIN_MATRIX_PREFETCH_PROFILES` with
//! comma-separated `no-ahead,current,deep-ahead,deeper` values, so the same data
//! shape can compare shallow, default, and deeper scheduled lookahead.
//! `SCDATA_FULLCHAIN_PROFILE=0` disables the default stage-profile output.
//!
//! This is a re-implementation of the original fullchain harness: the same
//! surface (file-backed CSR, scheduled prefetch, bg CPU/IO contention,
//! percentile waits) with a slimmer, fully `support`-backed implementation.

mod support;

use std::collections::HashSet;
use std::hint::black_box;
use std::io::{Cursor, Write};
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
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
use support::chunks::{make_csr_u32_f32_chunked_raw, make_csr_u32_f32_chunks_lz4};
use support::codecs::crc32_encode;
use support::{bench_data_dir, env_flag, env_usize, payload, BenchConfig};

fn main() {
    let config = BenchConfig::from_env();
    let explicit_fullchain =
        env_flag("SCDATA_FULLCHAIN_PREFETCH") || env_flag("SCDATA_FULLCHAIN_MATRIX");
    let fc = FullchainConfig::from_env(config, explicit_fullchain);
    if !explicit_fullchain {
        println!(
            "databank/fullchain_scheduled_prefetch running quick default; set SCDATA_FULLCHAIN_PREFETCH=1 for larger env-driven runs or SCDATA_FULLCHAIN_MATRIX=1 for the matrix"
        );
    }

    if env_flag("SCDATA_FULLCHAIN_MATRIX") {
        run_fullchain_matrix(&fc);
    } else {
        run_fullchain_case(&fc);
    }
}

fn run_fullchain_case(fc: &FullchainConfig) {
    println!(
        "databank/fullchain_scheduled_prefetch_config label={} codec={} batches={} batch_cells={} genes={} nnz_per_cell={} chunk_nnz={} chunks_per_batch={} chunk_miss_permille={} prefetch_step={} access_prefetch_step={} access_decode_ahead={} access_ready_ahead={} io_backend={} io_workers={} io_shards={} io_inflight={} scheduler_shards={} access_cpu_workers={} decode_workers={} fill_workers={} bg_cpu_threads={} bg_io_readers={} bg_io_file_mib={} bg_io_read_kib={} bg_io_same_file={} bg_io_drop_cache={} repeat_encoded_chunk={} profile={} data_dir={}",
        fc.label,
        fc.codec.name(),
        fc.batches,
        fc.batch_cells,
        fc.genes,
        fc.nnz_per_cell,
        fc.chunk_nnz,
        fc.chunks_per_batch,
        fc.chunk_miss_permille,
        fc.prefetch_step,
        fc.access_prefetch_step,
        fc.access_decode_ahead,
        fc.access_ready_ahead,
        fc.io_backend.name(),
        fc.io_workers,
        fc.io_shards,
        fc.io_inflight,
        fc.scheduler_shards,
        fc.access_cpu_workers,
        fc.decode_workers,
        fc.fill_workers,
        fc.bg_cpu_threads,
        fc.bg_io_readers,
        fc.bg_io_file_mib,
        fc.bg_io_read_kib,
        fc.bg_io_same_file,
        fc.bg_io_drop_cache,
        fc.repeat_encoded_chunk,
        fc.profile,
        fc.data_dir.display(),
    );

    let csr = prepare_csr_file(fc);
    println!(
        "databank/fullchain_file cells={} nnz={} chunks={} raw_mib={:.1} encoded_mib={:.1} encoded_over_raw={:.3}",
        csr.cells,
        csr.nnz,
        csr.total_chunks,
        csr.raw_bytes as f64 / (1024.0 * 1024.0),
        csr.encoded_bytes as f64 / (1024.0 * 1024.0),
        csr.encoded_bytes as f64 / csr.raw_bytes.max(1) as f64,
    );

    run_fullchain(fc, &csr);
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
        env_str("SCDATA_FULLCHAIN_CODEC")
            .map(|value| value.to_ascii_lowercase())
            .as_deref()
            .and_then(Self::from_name)
            .unwrap_or(Self::Lz4)
    }

    fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "zstd" => Some(Self::Zstd),
            "crc32" => Some(Self::Crc32),
            "raw" | "none" | "uncompressed" => Some(Self::Raw),
            "lz4" => Some(Self::Lz4),
            _ => None,
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
        env_str("SCDATA_FULLCHAIN_IO_BACKEND")
            .map(|value| value.to_ascii_lowercase())
            .as_deref()
            .and_then(Self::from_name)
            .unwrap_or(Self::Threaded)
    }

    fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "threaded" | "thread" => Some(Self::Threaded),
            #[cfg(feature = "uring")]
            "uring" | "io_uring" => Some(Self::Uring),
            #[cfg(not(feature = "uring"))]
            "uring" | "io_uring" => None,
            _ => None,
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

#[derive(Debug, Clone)]
struct FullchainConfig {
    label: String,
    codec: FullchainCodec,
    batches: usize,
    batch_cells: usize,
    genes: usize,
    nnz_per_cell: usize,
    chunk_nnz: usize,
    chunks_per_batch: usize,
    chunk_miss_permille: usize,
    repeat_encoded_chunk: bool,
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
    #[cfg(feature = "uring")]
    uring_drivers: usize,
    #[cfg(feature = "uring")]
    uring_registered_files: u32,
    scheduler_shards: usize,
    access_cpu_workers: usize,
    decode_workers: usize,
    fill_workers: usize,
    bg_cpu_threads: usize,
    bg_io_readers: usize,
    bg_io_file_mib: usize,
    bg_io_read_kib: usize,
    bg_io_same_file: bool,
    bg_io_drop_cache: bool,
    data_dir: PathBuf,
    warmups: usize,
    profile: bool,
}

impl FullchainConfig {
    fn from_env(config: BenchConfig, explicit_fullchain: bool) -> Self {
        let default_batch_cells = if explicit_fullchain { 128 } else { 16 };
        let default_genes = if explicit_fullchain { 4096 } else { 512 };
        let default_nnz_per_cell = if explicit_fullchain { 64 } else { 16 };
        let default_batches = if explicit_fullchain { 64 } else { 8 };
        let default_chunk_nnz = if explicit_fullchain { 1024 } else { 256 };
        let default_prefetch_step = if explicit_fullchain { 2 } else { 1 };
        let default_access_prefetch = if explicit_fullchain { 4 } else { 2 };
        let default_decode_ahead = if explicit_fullchain { 2 } else { 1 };
        let default_io_workers = if explicit_fullchain { 8 } else { 2 };
        let default_io_shards = if explicit_fullchain { 2 } else { 1 };
        let default_io_inflight = if explicit_fullchain { 256 } else { 64 };
        let default_scheduler_shards = if explicit_fullchain { 2 } else { 1 };
        let default_access_cpu_workers = if explicit_fullchain { 4 } else { 2 };
        let default_decode_workers = if explicit_fullchain { 8 } else { 2 };
        let default_fill_workers = if explicit_fullchain { 4 } else { 2 };
        let batch_cells = env_usize("SCDATA_FULLCHAIN_BATCH_CELLS").unwrap_or(default_batch_cells);
        let genes = env_usize("SCDATA_FULLCHAIN_GENES").unwrap_or(default_genes);
        let nnz_per_cell =
            env_usize("SCDATA_FULLCHAIN_NNZ_PER_CELL").unwrap_or(default_nnz_per_cell);
        let batches = env_usize("SCDATA_FULLCHAIN_BATCHES")
            .or_else(|| batches_from_file_mib(batch_cells, nnz_per_cell))
            .unwrap_or(default_batches);
        let chunks_per_batch =
            env_usize("SCDATA_FULLCHAIN_CHUNKS_PER_BATCH").unwrap_or_else(|| {
                let chunk_nnz =
                    env_usize("SCDATA_FULLCHAIN_CHUNK_NNZ").unwrap_or(default_chunk_nnz);
                (batch_cells * nnz_per_cell)
                    .div_ceil(chunk_nnz.max(1))
                    .max(1)
            });
        let chunk_nnz = env_usize("SCDATA_FULLCHAIN_CHUNK_NNZ").unwrap_or_else(|| {
            (batch_cells * nnz_per_cell)
                .div_ceil(chunks_per_batch.max(1))
                .max(1)
        });
        Self {
            label: env_str("SCDATA_FULLCHAIN_LABEL").unwrap_or_else(|| {
                if explicit_fullchain {
                    "single".to_string()
                } else {
                    "quick".to_string()
                }
            }),
            codec: FullchainCodec::from_env(),
            batches,
            batch_cells,
            genes,
            nnz_per_cell,
            chunk_nnz,
            chunks_per_batch,
            chunk_miss_permille: env_usize("SCDATA_FULLCHAIN_CHUNK_MISS_PERMILLE")
                .unwrap_or(1000)
                .min(1000),
            repeat_encoded_chunk: env_flag("SCDATA_FULLCHAIN_REPEAT_ENCODED_CHUNK"),
            prefetch_step: env_usize_any(&[
                "SCDATA_FULLCHAIN_DATABANK_PREFETCH_STEP",
                "SCDATA_FULLCHAIN_PREFETCH_STEP",
            ])
            .unwrap_or(default_prefetch_step),
            access_prefetch_step: env_usize("SCDATA_FULLCHAIN_ACCESS_PREFETCH_STEP")
                .unwrap_or(default_access_prefetch),
            access_decode_ahead: env_usize("SCDATA_FULLCHAIN_ACCESS_DECODE_AHEAD")
                .unwrap_or(default_decode_ahead),
            access_ready_ahead: env_usize("SCDATA_FULLCHAIN_ACCESS_READY_AHEAD").unwrap_or(1),
            io_backend: FullchainIoBackend::from_env(),
            io_workers: env_usize("SCDATA_FULLCHAIN_IO_WORKERS").unwrap_or(default_io_workers),
            io_shards: env_usize("SCDATA_FULLCHAIN_IO_SHARDS").unwrap_or(default_io_shards),
            io_inflight: env_usize("SCDATA_FULLCHAIN_IO_INFLIGHT").unwrap_or(default_io_inflight),
            #[cfg(feature = "uring")]
            uring_entries: env_usize("SCDATA_FULLCHAIN_URING_ENTRIES")
                .unwrap_or(default_io_inflight.max(64)) as u32,
            #[cfg(feature = "uring")]
            uring_drivers: env_usize("SCDATA_FULLCHAIN_URING_DRIVERS").unwrap_or(1),
            #[cfg(feature = "uring")]
            uring_registered_files: env_usize("SCDATA_FULLCHAIN_URING_REGISTERED_FILES")
                .unwrap_or(64) as u32,
            scheduler_shards: env_usize("SCDATA_FULLCHAIN_SCHEDULER_SHARDS")
                .unwrap_or(default_scheduler_shards),
            access_cpu_workers: env_usize("SCDATA_FULLCHAIN_ACCESS_CPU_WORKERS")
                .unwrap_or(default_access_cpu_workers),
            decode_workers: env_usize("SCDATA_FULLCHAIN_DECODE_WORKERS")
                .unwrap_or(default_decode_workers),
            fill_workers: env_usize("SCDATA_FULLCHAIN_FILL_WORKERS")
                .unwrap_or(default_fill_workers),
            bg_cpu_threads: env_usize("SCDATA_FULLCHAIN_BG_CPU_THREADS").unwrap_or(0),
            bg_io_readers: env_usize("SCDATA_FULLCHAIN_BG_IO_READERS").unwrap_or(0),
            bg_io_file_mib: env_usize("SCDATA_FULLCHAIN_BG_IO_FILE_MIB").unwrap_or(64),
            bg_io_read_kib: env_usize("SCDATA_FULLCHAIN_BG_IO_READ_KIB").unwrap_or(64),
            bg_io_same_file: env_flag("SCDATA_FULLCHAIN_BG_IO_SAME_FILE"),
            bg_io_drop_cache: env_flag("SCDATA_FULLCHAIN_BG_IO_DROP_CACHE"),
            data_dir: env_path("SCDATA_FULLCHAIN_DIR").unwrap_or_else(bench_data_dir),
            warmups: env_usize("SCDATA_FULLCHAIN_WARMUP_BATCHES").unwrap_or_else(|| {
                if explicit_fullchain {
                    config.warmups
                } else {
                    config.warmups.min(1)
                }
            }),
            profile: env_flag_default("SCDATA_FULLCHAIN_PROFILE", true),
        }
    }
}

fn batches_from_file_mib(batch_cells: usize, nnz_per_cell: usize) -> Option<usize> {
    let file_mib = env_usize("SCDATA_FULLCHAIN_FILE_MIB")?;
    let target_bytes = file_mib.checked_mul(1024)?.checked_mul(1024)?;
    let bytes_per_batch = batch_cells
        .checked_mul(nnz_per_cell)?
        .checked_mul(std::mem::size_of::<u32>() + std::mem::size_of::<f32>())?;
    Some(target_bytes.div_ceil(bytes_per_batch.max(1)).max(1))
}

#[derive(Debug, Clone, Copy)]
enum MatrixBackground {
    Baseline,
    Cpu,
    Io,
    CpuIo,
}

impl MatrixBackground {
    fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "baseline" | "none" => Some(Self::Baseline),
            "cpu" | "bg_cpu" => Some(Self::Cpu),
            "io" | "bg_io" => Some(Self::Io),
            "cpu_io" | "bg_cpu_io" | "both" => Some(Self::CpuIo),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Cpu => "bg_cpu",
            Self::Io => "bg_io",
            Self::CpuIo => "bg_cpu_io",
        }
    }

    fn apply(self, config: &mut FullchainConfig, cpu_threads: usize, io_readers: usize) {
        match self {
            Self::Baseline => {
                config.bg_cpu_threads = 0;
                config.bg_io_readers = 0;
            }
            Self::Cpu => {
                config.bg_cpu_threads = cpu_threads;
                config.bg_io_readers = 0;
            }
            Self::Io => {
                config.bg_cpu_threads = 0;
                config.bg_io_readers = io_readers;
            }
            Self::CpuIo => {
                config.bg_cpu_threads = cpu_threads;
                config.bg_io_readers = io_readers;
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum MatrixPrefetchProfile {
    NoAhead,
    Current,
    DeepAhead,
    Deeper,
}

impl MatrixPrefetchProfile {
    fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "no-ahead" | "no_ahead" | "none" | "shallow" => Some(Self::NoAhead),
            "current" | "default" | "base" => Some(Self::Current),
            "deep-ahead" | "deep_ahead" | "deep" => Some(Self::DeepAhead),
            "deeper" | "very-deep" | "very_deep" => Some(Self::Deeper),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::NoAhead => "no_ahead",
            Self::Current => "current",
            Self::DeepAhead => "deep_ahead",
            Self::Deeper => "deeper",
        }
    }

    fn apply(self, config: &mut FullchainConfig) {
        match self {
            Self::NoAhead => {
                config.prefetch_step = 1;
                config.access_prefetch_step = 1;
                config.access_decode_ahead = 0;
                config.access_ready_ahead = 0;
            }
            Self::Current => {}
            Self::DeepAhead => {
                config.prefetch_step = 4;
                config.access_prefetch_step = 8;
                config.access_decode_ahead = 4;
                config.access_ready_ahead = 2;
            }
            Self::Deeper => {
                config.prefetch_step = 8;
                config.access_prefetch_step = 16;
                config.access_decode_ahead = 8;
                config.access_ready_ahead = 4;
            }
        }
    }
}

fn run_fullchain_matrix(base: &FullchainConfig) {
    let codecs = matrix_codecs();
    let backends = matrix_backends();
    let backgrounds = matrix_backgrounds();
    let prefetch_profiles = matrix_prefetch_profiles();
    let matrix_cpu_threads = env_usize("SCDATA_FULLCHAIN_MATRIX_BG_CPU_THREADS")
        .unwrap_or_else(|| base.bg_cpu_threads.max(4));
    let matrix_io_readers = env_usize("SCDATA_FULLCHAIN_MATRIX_BG_IO_READERS")
        .unwrap_or_else(|| base.bg_io_readers.max(4));

    println!(
        "databank/fullchain_matrix cases={} codecs={} backends={} backgrounds={} prefetch_profiles={}",
        codecs.len() * backends.len() * backgrounds.len() * prefetch_profiles.len(),
        codecs
            .iter()
            .map(|codec| codec.name())
            .collect::<Vec<_>>()
            .join(","),
        backends
            .iter()
            .map(|backend| backend.name())
            .collect::<Vec<_>>()
            .join(","),
        backgrounds
            .iter()
            .map(|background| background.name())
            .collect::<Vec<_>>()
            .join(","),
        prefetch_profiles
            .iter()
            .map(|profile| profile.name())
            .collect::<Vec<_>>()
            .join(","),
    );

    for codec in codecs {
        for backend in &backends {
            for background in &backgrounds {
                for prefetch_profile in &prefetch_profiles {
                    let mut config = base.clone();
                    config.codec = codec;
                    config.io_backend = *backend;
                    background.apply(&mut config, matrix_cpu_threads, matrix_io_readers);
                    prefetch_profile.apply(&mut config);
                    config.label = format!(
                        "{}-{}-{}-{}",
                        codec.name(),
                        backend.name(),
                        background.name(),
                        prefetch_profile.name()
                    );
                    run_fullchain_case(&config);
                }
            }
        }
    }
}

fn matrix_codecs() -> Vec<FullchainCodec> {
    parse_csv_env("SCDATA_FULLCHAIN_MATRIX_CODECS", FullchainCodec::from_name).unwrap_or_else(
        || {
            vec![
                FullchainCodec::Raw,
                FullchainCodec::Lz4,
                FullchainCodec::Zstd,
            ]
        },
    )
}

fn matrix_backends() -> Vec<FullchainIoBackend> {
    parse_csv_env(
        "SCDATA_FULLCHAIN_MATRIX_BACKENDS",
        FullchainIoBackend::from_name,
    )
    .unwrap_or_else(default_matrix_backends)
}

fn default_matrix_backends() -> Vec<FullchainIoBackend> {
    let backends = vec![FullchainIoBackend::Threaded];
    #[cfg(feature = "uring")]
    let backends = {
        let mut backends = backends;
        backends.push(FullchainIoBackend::Uring);
        backends
    };
    backends
}

fn matrix_backgrounds() -> Vec<MatrixBackground> {
    parse_csv_env(
        "SCDATA_FULLCHAIN_MATRIX_BACKGROUNDS",
        MatrixBackground::from_name,
    )
    .unwrap_or_else(|| {
        vec![
            MatrixBackground::Baseline,
            MatrixBackground::Cpu,
            MatrixBackground::Io,
            MatrixBackground::CpuIo,
        ]
    })
}

fn matrix_prefetch_profiles() -> Vec<MatrixPrefetchProfile> {
    parse_csv_env(
        "SCDATA_FULLCHAIN_MATRIX_PREFETCH_PROFILES",
        MatrixPrefetchProfile::from_name,
    )
    .unwrap_or_else(|| {
        vec![
            MatrixPrefetchProfile::NoAhead,
            MatrixPrefetchProfile::Current,
            MatrixPrefetchProfile::DeepAhead,
        ]
    })
}

fn parse_csv_env<T>(name: &str, parse: impl Fn(&str) -> Option<T>) -> Option<Vec<T>> {
    let value = env_str(name)?;
    let parsed = value.split(',').filter_map(parse).collect::<Vec<_>>();
    (!parsed.is_empty()).then_some(parsed)
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
    let (indptr, mut indices_chunks, mut data_chunks) = match fc.codec {
        FullchainCodec::Raw => {
            make_csr_u32_f32_chunked_raw(cells, fc.genes, fc.nnz_per_cell, fc.chunk_nnz)
        }
        FullchainCodec::Lz4 => {
            make_csr_u32_f32_chunks_lz4(cells, fc.genes, fc.nnz_per_cell, fc.chunk_nnz)
        }
        FullchainCodec::Zstd => {
            let (indptr, idx, data) =
                make_csr_u32_f32_chunked_raw(cells, fc.genes, fc.nnz_per_cell, fc.chunk_nnz);
            (
                indptr,
                zstd_encode_chunks(idx, 3),
                zstd_encode_chunks(data, 3),
            )
        }
        FullchainCodec::Crc32 => {
            let (indptr, idx, data) =
                make_csr_u32_f32_chunked_raw(cells, fc.genes, fc.nnz_per_cell, fc.chunk_nnz);
            (indptr, crc_wrap_chunks(idx), crc_wrap_chunks(data))
        }
    };

    if fc.repeat_encoded_chunk {
        repeat_matching_encoded_chunks(&mut indices_chunks);
        repeat_matching_encoded_chunks(&mut data_chunks);
    }

    let raw_bytes = (cells * fc.nnz_per_cell) as u64
        * (std::mem::size_of::<u32>() + std::mem::size_of::<f32>()) as u64;
    let encoded_bytes = indices_chunks
        .iter()
        .chain(data_chunks.iter())
        .map(|chunk| chunk.len() as u64)
        .sum();

    let (indices_path, indices_locations) =
        write_chunks_file_in_dir(&fc.data_dir, "fullchain-indices", &indices_chunks);
    let (data_path, data_locations) =
        write_chunks_file_in_dir(&fc.data_dir, "fullchain-data", &data_chunks);

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

fn repeat_matching_encoded_chunks(chunks: &mut [Arc<[u8]>]) {
    let Some(first) = chunks.first().cloned() else {
        return;
    };
    for chunk in chunks.iter_mut().skip(1) {
        if chunk.len() == first.len() {
            *chunk = Arc::clone(&first);
        }
    }
}

fn write_chunks_file_in_dir(
    dir: &Path,
    label: &str,
    chunks: &[Arc<[u8]>],
) -> (PathBuf, Vec<FileChunkLocation>) {
    std::fs::create_dir_all(dir).expect("create fullchain data dir");
    let path = dir.join(format!("{label}-{}.bin", std::process::id()));
    let mut file = std::fs::File::create(&path).expect("create fullchain chunk file");
    let mut offset = 0u64;
    let mut locations = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        file.write_all(chunk).expect("write fullchain chunk");
        locations.push(FileChunkLocation {
            offset,
            len: chunk.len(),
        });
        offset += chunk.len() as u64;
    }
    (path, locations)
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
                variable_chunks: false,
                chunk_boundaries: None,
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
                variable_chunks: false,
                chunk_boundaries: None,
            },
            index_dtype: DType::U32,
            num_cells: csr.cells,
            num_genes: fc.genes,
        })
        .expect("register fullchain csr dataset");

    let batches = make_batches(fc);
    let encoded_workload = encoded_workload(csr, &batches, fc.chunk_nnz);
    let sp_config = ScheduledPrefetchConfig {
        prefetch_step: fc.prefetch_step,
        access: ScheduledAccessConfig {
            prefetch_step: fc.access_prefetch_step,
            decode_ahead_steps: fc.access_decode_ahead,
            ready_ahead_steps: fc.access_ready_ahead,
        },
    };

    // Warm-up pass (no bg contention, no timing).
    clear_databank_prefetch_profile_env();
    for _ in 0..fc.warmups {
        let prefetch = bank
            .prefetch_cells_scheduled::<f32, _>(dataset_id, batches.clone(), sp_config)
            .expect("warmup prefetch");
        for batch in prefetch {
            let batch = batch.expect("warmup batch");
            black_box(batch.buffer.as_ptr());
        }
    }

    let bg = FullchainBackground::start(fc, csr);
    let started = Instant::now();
    let mut waits: Vec<u64> = Vec::with_capacity(batches.len());
    let mut total_cells = 0usize;
    let mut total_values = 0usize;

    configure_databank_prefetch_profile_env(fc);
    let mut prefetch = bank
        .prefetch_cells_scheduled::<f32, _>(dataset_id, batches, sp_config)
        .expect("fullchain prefetch");
    loop {
        let wait_start = Instant::now();
        let Some(batch_result) = prefetch.next() else {
            break;
        };
        let batch = batch_result.expect("prefetch batch");
        waits.push(wait_start.elapsed().as_nanos() as u64);
        total_cells += batch.cells.len();
        total_values += batch.buffer.len();
        black_box(batch.buffer.as_ptr());
    }
    let elapsed = started.elapsed();
    drop(prefetch);
    clear_databank_prefetch_profile_env();
    let bg_sum = bg.stop();

    waits.sort_unstable();
    let p50 = percentile(&waits, 500);
    let p99 = percentile(&waits, 990);
    let p999 = percentile(&waits, 999);
    let max_ns = *waits.last().unwrap_or(&0);
    let seconds = elapsed.as_secs_f64();
    let output_mib = total_values as f64 * std::mem::size_of::<f32>() as f64 / (1024.0 * 1024.0);
    let slice_mib = total_cells as f64
        * fc.nnz_per_cell as f64
        * (std::mem::size_of::<u32>() + std::mem::size_of::<f32>()) as f64
        / (1024.0 * 1024.0);
    let encoded_file_mib = csr.encoded_bytes as f64 / (1024.0 * 1024.0);
    let encoded_requested_mib = encoded_workload.requested_bytes as f64 / (1024.0 * 1024.0);
    let encoded_unique_mib = encoded_workload.unique_bytes as f64 / (1024.0 * 1024.0);

    println!(
        "databank/fullchain_scheduled_prefetch    label={} batches={} cells={} values={} elapsed_s={seconds:.4} throughput_mib_s={:.1} output_mib_s={:.1} slice_mib_s={:.1} encoded_requested_mib_s={:.1} encoded_unique_mib_s={:.1} encoded_file_mib={encoded_file_mib:.1} encoded_requested_mib={encoded_requested_mib:.1} encoded_unique_mib={encoded_unique_mib:.1} encoded_requested_chunks={} encoded_unique_chunks={} kops={:.1} wait_ns_p50={p50} p99={p99} p999={p999} max={max_ns} bg_sum={bg_sum}",
        fc.label,
        waits.len(),
        total_cells,
        total_values,
        output_mib / seconds,
        output_mib / seconds,
        slice_mib / seconds,
        encoded_requested_mib / seconds,
        encoded_unique_mib / seconds,
        encoded_workload.requested_chunks,
        encoded_workload.unique_chunks,
        waits.len() as f64 / seconds / 1000.0,
    );
}

fn make_batches(fc: &FullchainConfig) -> Vec<Vec<usize>> {
    let mut next_unique_batch = 0usize;
    let mut active_base = 0usize;
    (0..fc.batches)
        .map(|batch| {
            if batch == 0 || should_start_miss_batch(batch, fc.chunk_miss_permille) {
                active_base = next_unique_batch * fc.batch_cells;
                next_unique_batch += 1;
            }
            (0..fc.batch_cells).map(|i| active_base + i).collect()
        })
        .collect()
}

#[derive(Default)]
struct EncodedWorkload {
    requested_bytes: u64,
    unique_bytes: u64,
    requested_chunks: usize,
    unique_chunks: usize,
}

fn encoded_workload(csr: &CsrFile, batches: &[Vec<usize>], chunk_nnz: usize) -> EncodedWorkload {
    let chunk_nnz = chunk_nnz.max(1);
    let mut requested_bytes = 0u64;
    let mut requested_chunks = 0usize;
    let mut unique_chunks = HashSet::new();

    for batch in batches {
        let mut batch_chunks = HashSet::new();
        for &cell in batch {
            if cell + 1 >= csr.indptr.len() {
                continue;
            }
            let start = csr.indptr[cell] as usize;
            let end = csr.indptr[cell + 1] as usize;
            if start >= end {
                continue;
            }
            for chunk in (start / chunk_nnz)..=((end - 1) / chunk_nnz) {
                batch_chunks.insert(chunk);
            }
        }
        for chunk in batch_chunks {
            requested_bytes += encoded_chunk_bytes(csr, chunk);
            requested_chunks += 2;
            unique_chunks.insert(chunk);
        }
    }

    let unique_bytes = unique_chunks
        .iter()
        .map(|&chunk| encoded_chunk_bytes(csr, chunk))
        .sum();
    EncodedWorkload {
        requested_bytes,
        unique_bytes,
        requested_chunks,
        unique_chunks: unique_chunks.len() * 2,
    }
}

fn encoded_chunk_bytes(csr: &CsrFile, chunk: usize) -> u64 {
    let indices = csr
        .indices_locations
        .get(chunk)
        .map_or(0, |location| location.len as u64);
    let data = csr
        .data_locations
        .get(chunk)
        .map_or(0, |location| location.len as u64);
    indices + data
}

fn should_start_miss_batch(batch: usize, miss_permille: usize) -> bool {
    match miss_permille {
        0 => false,
        1000.. => true,
        value => (splitmix64(batch as u64) % 1000) < value as u64,
    }
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
    fn start(fc: &FullchainConfig, csr: &CsrFile) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles: Vec<JoinHandle<u64>> = Vec::new();

        for thread_idx in 0..fc.bg_cpu_threads {
            let stop = Arc::clone(&stop);
            handles.push(thread::spawn(move || run_bg_cpu(stop, thread_idx)));
        }

        if fc.bg_io_readers > 0 {
            let (bg_path, cleanup_bg_file) = if fc.bg_io_same_file {
                (csr.data_path.clone(), false)
            } else {
                let bg_path = fc
                    .data_dir
                    .join(format!("fullchain-bg-{}.bin", std::process::id()));
                std::fs::write(&bg_path, payload(fc.bg_io_file_mib * 1024 * 1024))
                    .expect("write bg io file");
                (bg_path, true)
            };
            for reader_idx in 0..fc.bg_io_readers {
                let stop = Arc::clone(&stop);
                let path = bg_path.clone();
                let kib = fc.bg_io_read_kib;
                let drop_cache = fc.bg_io_drop_cache;
                handles.push(thread::spawn(move || {
                    run_bg_io(stop, &path, kib, reader_idx, drop_cache)
                }));
            }
            if cleanup_bg_file {
                let stop_for_cleanup = Arc::clone(&stop);
                let cleanup_path = bg_path.clone();
                handles.push(thread::spawn(move || {
                    while !stop_for_cleanup.load(Ordering::Relaxed) {
                        thread::sleep(std::time::Duration::from_millis(50));
                    }
                    let _ = std::fs::remove_file(&cleanup_path);
                    0
                }));
            }
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

fn run_bg_io(
    stop: Arc<AtomicBool>,
    path: &Path,
    kib: usize,
    reader_idx: usize,
    drop_cache: bool,
) -> u64 {
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
    let mut state = splitmix64((reader_idx as u64).wrapping_add(0x9e37_79b9_7f4a_7c15));
    let mut total = 0u64;
    while !stop.load(Ordering::Relaxed) {
        state = splitmix64(state);
        let max_pos = file_len.saturating_sub(buf.len());
        let pos = if max_pos == 0 {
            0
        } else {
            (state as usize) % (max_pos + 1)
        };
        let read = match file.read_at(&mut buf, pos as u64) {
            Ok(read) => read,
            Err(_) => break,
        };
        total += read as u64;
        if drop_cache && read > 0 {
            drop_file_cache(file.as_raw_fd(), pos as u64, read);
        }
    }
    total
}

fn drop_file_cache(fd: std::os::unix::io::RawFd, offset: u64, len: usize) {
    #[cfg(target_os = "linux")]
    unsafe {
        let _ = libc::posix_fadvise(
            fd,
            offset as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        );
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (fd, offset, len);
    }
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

// ---------------------------------------------------------------------------
// DataBank / IO configuration.
// ---------------------------------------------------------------------------

fn fullchain_databank_config(fc: &FullchainConfig) -> DataBankConfig {
    let mut config = DataBankConfig {
        io_config: fullchain_io_config(fc),
        ..Default::default()
    };
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
            drivers: fc.uring_drivers,
            iowq_bounded_workers: 0,
            iowq_unbounded_workers: 0,
            registered_files: fc.uring_registered_files,
        }),
    }
}

fn env_str(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

fn env_flag_default(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        ),
        Err(_) => default,
    }
}

fn env_path(name: &str) -> Option<PathBuf> {
    env_str(name).map(PathBuf::from)
}

fn env_usize_any(names: &[&str]) -> Option<usize> {
    names.iter().find_map(|name| env_usize(name))
}

fn configure_databank_prefetch_profile_env(fc: &FullchainConfig) {
    if fc.profile {
        std::env::set_var("SCDATA_DATABANK_PREFETCH_PROFILE", "1");
        std::env::set_var("SCDATA_DATABANK_PREFETCH_PROFILE_LABEL", &fc.label);
    } else {
        clear_databank_prefetch_profile_env();
    }
}

fn clear_databank_prefetch_profile_env() {
    std::env::remove_var("SCDATA_DATABANK_PREFETCH_PROFILE");
    std::env::remove_var("SCDATA_DATABANK_PREFETCH_PROFILE_LABEL");
}
