//! Standalone DataBank benchmarks.
//!
//! This target keeps DataBank access paths separate from the broader module
//! sweep so larger single-cell shapes can be stressed directly.

mod support;

use std::process::ExitCode;

use _scdata::access::{AccessConfig, AccessCpuConfig, AccessProfile, ScheduledAccessConfig};
use _scdata::codecs::DecodePoolConfig;
use _scdata::databank::{
    ArrayCodecSpec, DataBank, DataBankConfig, EdgeChunkLayout, FillConfig, MissingGenePolicy,
    NativeAccessConfig, ScheduledPrefetchConfig,
};
use _scdata::iopool::{BaseIoConfig, IoConfig, ThreadedConfig};
use support::chunks::{
    dense1d_u32_spec, dense2d_u32_spec, make_csr_u32_f32_chunks, make_csr_u32_f32_chunks_lz4,
    make_dense1d_u32_chunks, make_dense_u32_chunks, make_dense_u32_chunks_lz4,
    sparse_csr_u32_f32_spec,
};
use support::{bench, env_flag_default, env_usize, BenchConfig};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Debug, Clone)]
struct Args {
    cells: usize,
    genes: usize,
    batch_cells: usize,
    batches: usize,
    nnz_per_cell: usize,
    chunk_rows: usize,
    chunk_cols: usize,
    dense1d_chunk_len: usize,
    csr_chunk_len: usize,
    gene_subset: usize,
    workers: usize,
    shards: usize,
    queue_capacity: usize,
    inflight: usize,
    cache_mib: usize,
    memory_mib: usize,
    keep_decoded: bool,
    prefetch_step: usize,
    access_prefetch_step: usize,
    access_decode_ahead: usize,
    access_ready_ahead: usize,
    cases: Vec<Case>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Case {
    Dense2dRaw,
    Dense2dLz4Genes,
    Dense1dOwned,
    SparseRaw,
    SparseLz4,
    Dense2dScheduled,
    SparseScheduled,
}

impl Case {
    fn parse(token: &str) -> Result<Self, String> {
        match token.trim().to_ascii_lowercase().as_str() {
            "dense2d" | "dense2d_raw" | "dense2d-raw" => Ok(Self::Dense2dRaw),
            "dense2d_lz4" | "dense2d-lz4" | "dense2d_lz4_genes" | "dense2d-lz4-genes" => {
                Ok(Self::Dense2dLz4Genes)
            }
            "dense1d" | "dense1d_owned" | "dense1d-owned" => Ok(Self::Dense1dOwned),
            "sparse" | "sparse_raw" | "sparse-raw" | "sparse_csr" | "sparse-csr" => {
                Ok(Self::SparseRaw)
            }
            "sparse_lz4" | "sparse-lz4" | "sparse_csr_lz4" | "sparse-csr-lz4" => {
                Ok(Self::SparseLz4)
            }
            "dense2d_scheduled" | "dense2d-scheduled" | "scheduled_dense" => {
                Ok(Self::Dense2dScheduled)
            }
            "sparse_scheduled" | "sparse-scheduled" | "scheduled_sparse" => {
                Ok(Self::SparseScheduled)
            }
            other => Err(format!("unknown databank case `{other}`")),
        }
    }

    fn all() -> Vec<Self> {
        vec![
            Self::Dense2dRaw,
            Self::Dense2dLz4Genes,
            Self::Dense1dOwned,
            Self::SparseRaw,
            Self::SparseLz4,
            Self::Dense2dScheduled,
            Self::SparseScheduled,
        ]
    }
}

fn run() -> Result<(), String> {
    let bench_config = BenchConfig::from_env();
    let args = parse_args()?;
    println!(
        "databank/standalone cells={} genes={} batch_cells={} batches={} nnz_per_cell={} chunk={}x{} dense1d_chunk_len={} csr_chunk_len={} workers={} shards={} queue={} inflight={} cache_mib={} memory_mib={} keep_decoded={} prefetch_step={} access=({}, {}, {}) cases={:?}",
        args.cells,
        args.genes,
        args.batch_cells,
        args.batches,
        args.nnz_per_cell,
        args.chunk_rows,
        args.chunk_cols,
        args.dense1d_chunk_len,
        args.csr_chunk_len,
        args.workers,
        args.shards,
        args.queue_capacity,
        args.inflight,
        args.cache_mib,
        args.memory_mib,
        args.keep_decoded,
        args.prefetch_step,
        args.access_prefetch_step,
        args.access_decode_ahead,
        args.access_ready_ahead,
        args.cases
    );

    let mut bank = DataBank::new(databank_config(&args)).map_err(|err| err.to_string())?;
    let dense2d_raw_id = if args.needs_dense2d_raw() {
        Some(
            bank.register_dense_2d(dense2d_u32_spec(
                args.cells,
                args.genes,
                args.chunk_rows,
                args.chunk_cols,
                make_dense_u32_chunks(args.cells, args.genes, args.chunk_rows, args.chunk_cols),
                ArrayCodecSpec::Uncompressed,
                EdgeChunkLayout::Padded,
            ))
            .map_err(|err| err.to_string())?,
        )
    } else {
        None
    };
    let dense2d_lz4_id = if args.has(Case::Dense2dLz4Genes) {
        Some(
            bank.register_dense_2d(dense2d_u32_spec(
                args.cells,
                args.genes,
                args.chunk_rows,
                args.chunk_cols,
                make_dense_u32_chunks_lz4(args.cells, args.genes, args.chunk_rows, args.chunk_cols),
                ArrayCodecSpec::CodecJson(r#"{"id":"lz4"}"#.to_string()),
                EdgeChunkLayout::Padded,
            ))
            .map_err(|err| err.to_string())?,
        )
    } else {
        None
    };
    let dense1d_id = if args.has(Case::Dense1dOwned) {
        Some(
            bank.register_dense_1d(dense1d_u32_spec(
                args.cells,
                args.genes,
                args.dense1d_chunk_len,
                make_dense1d_u32_chunks(args.cells, args.genes, args.dense1d_chunk_len),
                ArrayCodecSpec::Uncompressed,
            ))
            .map_err(|err| err.to_string())?,
        )
    } else {
        None
    };
    let sparse_raw_id = if args.needs_sparse_raw() {
        let (indptr, indices, data) = make_csr_u32_f32_chunks(
            args.cells,
            args.genes,
            args.nnz_per_cell,
            args.csr_chunk_len,
        );
        Some(
            bank.register_sparse_csr(sparse_csr_u32_f32_spec(
                args.cells,
                args.genes,
                indptr,
                args.csr_chunk_len,
                args.csr_chunk_len,
                indices,
                data,
                ArrayCodecSpec::Uncompressed,
            ))
            .map_err(|err| err.to_string())?,
        )
    } else {
        None
    };
    let sparse_lz4_id = if args.has(Case::SparseLz4) {
        let (indptr, indices, data) = make_csr_u32_f32_chunks_lz4(
            args.cells,
            args.genes,
            args.nnz_per_cell,
            args.csr_chunk_len,
        );
        Some(
            bank.register_sparse_csr(sparse_csr_u32_f32_spec(
                args.cells,
                args.genes,
                indptr,
                args.csr_chunk_len,
                args.csr_chunk_len,
                indices,
                data,
                ArrayCodecSpec::CodecJson(r#"{"id":"lz4"}"#.to_string()),
            ))
            .map_err(|err| err.to_string())?,
        )
    } else {
        None
    };

    let batch = make_batch(0, &args);
    let batches = make_batches(&args);
    let gene_subset = selected_gene_names(args.gene_subset.min(args.genes));
    let scheduled_config = ScheduledPrefetchConfig {
        prefetch_step: args.prefetch_step,
        access: ScheduledAccessConfig {
            prefetch_step: args.access_prefetch_step,
            decode_ahead_steps: args.access_decode_ahead,
            ready_ahead_steps: args.access_ready_ahead,
        },
        ..ScheduledPrefetchConfig::default()
    };

    if args.has(Case::Dense2dRaw) {
        let id = dense2d_raw_id.expect("dense2d raw id");
        bench(
            bench_config,
            "databank/dense2d_raw/access_cells",
            128,
            Some(args.batch_output_bytes::<u32>()),
            || {
                let mut out = vec![0u32; args.batch_output_elements()];
                bank.access_cells(id, &batch, &mut out, None)
                    .expect("dense2d raw access");
                out.len() ^ out[out.len() / 2] as usize
            },
        );
    }

    if args.has(Case::Dense2dLz4Genes) {
        let id = dense2d_lz4_id.expect("dense2d lz4 id");
        bench(
            bench_config,
            "databank/dense2d_lz4/access_gene_subset",
            128,
            Some(args.batch_cells * gene_subset.len() * std::mem::size_of::<u32>()),
            || {
                let mut out = vec![0u32; args.batch_cells * gene_subset.len()];
                bank.access_cells_by_gene_names(
                    id,
                    &batch,
                    &gene_subset,
                    &mut out,
                    None,
                    MissingGenePolicy::Error,
                )
                .expect("dense2d lz4 gene access");
                out.len() ^ out[out.len() / 2] as usize
            },
        );
    }

    if args.has(Case::Dense1dOwned) {
        let id = dense1d_id.expect("dense1d id");
        bench(
            bench_config,
            "databank/dense1d_raw/access_cells_owned",
            128,
            Some(args.batch_output_bytes::<u32>()),
            || {
                let out = bank
                    .access_cells_owned::<u32>(id, &batch)
                    .expect("dense1d owned access");
                out.len() ^ out[out.len() / 2] as usize
            },
        );
    }

    if args.has(Case::SparseRaw) {
        let id = sparse_raw_id.expect("sparse raw id");
        bench(
            bench_config,
            "databank/sparse_csr_raw/access_cells",
            128,
            Some(args.batch_output_bytes::<f32>()),
            || {
                let mut out = vec![0f32; args.batch_output_elements()];
                bank.access_cells(id, &batch, &mut out, None)
                    .expect("sparse raw access");
                out.len() ^ out[out.len() / 2].to_bits() as usize
            },
        );
    }

    if args.has(Case::SparseLz4) {
        let id = sparse_lz4_id.expect("sparse lz4 id");
        bench(
            bench_config,
            "databank/sparse_csr_lz4/access_cells",
            128,
            Some(args.batch_output_bytes::<f32>()),
            || {
                let mut out = vec![0f32; args.batch_output_elements()];
                bank.access_cells(id, &batch, &mut out, None)
                    .expect("sparse lz4 access");
                out.len() ^ out[out.len() / 2].to_bits() as usize
            },
        );
    }

    if args.has(Case::Dense2dScheduled) {
        let id = dense2d_raw_id.expect("dense2d raw id");
        bench(
            bench_config,
            "databank/dense2d_raw/prefetch_scheduled",
            16,
            Some(args.scheduled_output_bytes::<u32>()),
            || {
                let prefetch = bank
                    .prefetch_cells_scheduled::<u32, _>(id, batches.clone(), scheduled_config)
                    .expect("dense2d scheduled prefetch");
                prefetch.fold(0usize, |acc, result| {
                    let batch = result.expect("dense2d scheduled batch");
                    acc ^ batch.buffer.len() ^ batch.buffer[batch.buffer.len() / 2] as usize
                })
            },
        );
    }

    if args.has(Case::SparseScheduled) {
        let id = sparse_raw_id.expect("sparse raw id");
        bench(
            bench_config,
            "databank/sparse_csr_raw/prefetch_scheduled",
            16,
            Some(args.scheduled_output_bytes::<f32>()),
            || {
                let prefetch = bank
                    .prefetch_cells_scheduled::<f32, _>(id, batches.clone(), scheduled_config)
                    .expect("sparse scheduled prefetch");
                prefetch.fold(0usize, |acc, result| {
                    let batch = result.expect("sparse scheduled batch");
                    acc ^ batch.buffer.len()
                        ^ batch.buffer[batch.buffer.len() / 2].to_bits() as usize
                })
            },
        );
    }

    Ok(())
}

impl Args {
    fn from_env() -> Self {
        Self {
            cells: env_usize("SCDATA_DATABANK_CELLS").unwrap_or(2048),
            genes: env_usize("SCDATA_DATABANK_GENES").unwrap_or(4096),
            batch_cells: env_usize("SCDATA_DATABANK_BATCH_CELLS").unwrap_or(128),
            batches: env_usize("SCDATA_DATABANK_BATCHES").unwrap_or(16),
            nnz_per_cell: env_usize("SCDATA_DATABANK_NNZ_PER_CELL").unwrap_or(64),
            chunk_rows: env_usize("SCDATA_DATABANK_CHUNK_ROWS").unwrap_or(128),
            chunk_cols: env_usize("SCDATA_DATABANK_CHUNK_COLS").unwrap_or(256),
            dense1d_chunk_len: env_usize("SCDATA_DATABANK_DENSE1D_CHUNK_LEN").unwrap_or(4096),
            csr_chunk_len: env_usize("SCDATA_DATABANK_CSR_CHUNK_LEN").unwrap_or(4096),
            gene_subset: env_usize("SCDATA_DATABANK_GENE_SUBSET").unwrap_or(64),
            workers: env_usize("SCDATA_DATABANK_WORKERS").unwrap_or(4),
            shards: env_usize("SCDATA_DATABANK_SHARDS").unwrap_or(2),
            queue_capacity: env_usize("SCDATA_DATABANK_QUEUE").unwrap_or(512),
            inflight: env_usize("SCDATA_DATABANK_INFLIGHT").unwrap_or(512),
            cache_mib: env_usize("SCDATA_DATABANK_CACHE_MIB").unwrap_or(128),
            memory_mib: env_usize("SCDATA_DATABANK_MEMORY_MIB").unwrap_or(256),
            keep_decoded: env_flag_default("SCDATA_DATABANK_KEEP_DECODED", true),
            prefetch_step: env_usize("SCDATA_DATABANK_PREFETCH_STEP").unwrap_or(2),
            access_prefetch_step: env_usize("SCDATA_DATABANK_ACCESS_PREFETCH_STEP").unwrap_or(3),
            access_decode_ahead: env_usize("SCDATA_DATABANK_ACCESS_DECODE_AHEAD").unwrap_or(2),
            access_ready_ahead: env_usize("SCDATA_DATABANK_ACCESS_READY_AHEAD").unwrap_or(1),
            cases: parse_cases_env("SCDATA_DATABANK_CASES").unwrap_or_else(Case::all),
        }
    }

    fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("cells", self.cells),
            ("genes", self.genes),
            ("batch_cells", self.batch_cells),
            ("batches", self.batches),
            ("nnz_per_cell", self.nnz_per_cell),
            ("chunk_rows", self.chunk_rows),
            ("chunk_cols", self.chunk_cols),
            ("dense1d_chunk_len", self.dense1d_chunk_len),
            ("csr_chunk_len", self.csr_chunk_len),
            ("gene_subset", self.gene_subset),
            ("workers", self.workers),
            ("shards", self.shards),
            ("queue_capacity", self.queue_capacity),
            ("inflight", self.inflight),
            ("prefetch_step", self.prefetch_step),
            ("access_prefetch_step", self.access_prefetch_step),
            ("access_decode_ahead", self.access_decode_ahead),
            ("access_ready_ahead", self.access_ready_ahead),
        ] {
            if value == 0 {
                return Err(format!("{name} must be greater than 0"));
            }
        }
        if self.cases.is_empty() {
            return Err("at least one databank case is required".to_string());
        }
        Ok(())
    }

    fn has(&self, case: Case) -> bool {
        self.cases.contains(&case)
    }

    fn needs_dense2d_raw(&self) -> bool {
        self.has(Case::Dense2dRaw) || self.has(Case::Dense2dScheduled)
    }

    fn needs_sparse_raw(&self) -> bool {
        self.has(Case::SparseRaw) || self.has(Case::SparseScheduled)
    }

    fn batch_output_elements(&self) -> usize {
        self.batch_cells * self.genes
    }

    fn batch_output_bytes<T>(&self) -> usize {
        self.batch_output_elements() * std::mem::size_of::<T>()
    }

    fn scheduled_output_bytes<T>(&self) -> usize {
        self.batch_output_bytes::<T>() * self.batches
    }
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::from_env();
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--cells" => args.cells = parse_next(&mut iter, "--cells")?,
            "--genes" => args.genes = parse_next(&mut iter, "--genes")?,
            "--batch-cells" => args.batch_cells = parse_next(&mut iter, "--batch-cells")?,
            "--batches" => args.batches = parse_next(&mut iter, "--batches")?,
            "--nnz-per-cell" => args.nnz_per_cell = parse_next(&mut iter, "--nnz-per-cell")?,
            "--chunk-rows" => args.chunk_rows = parse_next(&mut iter, "--chunk-rows")?,
            "--chunk-cols" => args.chunk_cols = parse_next(&mut iter, "--chunk-cols")?,
            "--dense1d-chunk-len" => {
                args.dense1d_chunk_len = parse_next(&mut iter, "--dense1d-chunk-len")?
            }
            "--csr-chunk-len" => args.csr_chunk_len = parse_next(&mut iter, "--csr-chunk-len")?,
            "--gene-subset" => args.gene_subset = parse_next(&mut iter, "--gene-subset")?,
            "--workers" => args.workers = parse_next(&mut iter, "--workers")?,
            "--shards" => args.shards = parse_next(&mut iter, "--shards")?,
            "--queue" => args.queue_capacity = parse_next(&mut iter, "--queue")?,
            "--inflight" => args.inflight = parse_next(&mut iter, "--inflight")?,
            "--cache-mib" => args.cache_mib = parse_next(&mut iter, "--cache-mib")?,
            "--memory-mib" => args.memory_mib = parse_next(&mut iter, "--memory-mib")?,
            "--keep-decoded" => args.keep_decoded = true,
            "--no-keep-decoded" => args.keep_decoded = false,
            "--prefetch-step" => args.prefetch_step = parse_next(&mut iter, "--prefetch-step")?,
            "--access-prefetch-step" => {
                args.access_prefetch_step = parse_next(&mut iter, "--access-prefetch-step")?
            }
            "--access-decode-ahead" => {
                args.access_decode_ahead = parse_next(&mut iter, "--access-decode-ahead")?
            }
            "--access-ready-ahead" => {
                args.access_ready_ahead = parse_next(&mut iter, "--access-ready-ahead")?
            }
            "--cases" => args.cases = parse_cases(&next_arg(&mut iter, "--cases")?)?,
            "--bench" => {
                // Cargo passes this to custom bench binaries.
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    args.validate()?;
    Ok(args)
}

fn databank_config(args: &Args) -> DataBankConfig {
    DataBankConfig {
        io_config: IoConfig::Threaded(ThreadedConfig {
            base: BaseIoConfig {
                max_in_flight: args.inflight,
                queue_capacity: args.queue_capacity,
                priority_levels: 2,
                queue_shards: args.shards,
                assume_non_overlapping_reads: true,
            },
            num_workers: args.workers,
            cpus: None,
        }),
        decode_config: DecodePoolConfig {
            num_workers: args.workers,
            queue_capacity: args.queue_capacity,
            cpus: None,
        },
        access_config: AccessConfig {
            queue_capacity: args.queue_capacity,
            scheduler_shards: args.shards,
            cache_capacity_bytes: args.cache_mib * 1024 * 1024,
            memory_budget_bytes: args.memory_mib * 1024 * 1024,
            default_io_priority: 0,
            keep_decoded: args.keep_decoded,
            cpu: AccessCpuConfig {
                num_workers: args.workers,
                queue_capacity: args.queue_capacity,
                cpus: None,
            },
            profile: AccessProfile::from_env(),
        },
        fill_config: FillConfig {
            parallel: true,
            num_workers: args.workers,
            queue_capacity: args.queue_capacity,
            min_parallel_rows: 1,
            min_parallel_bytes: 1,
            cpus: None,
        },
        native_config: NativeAccessConfig::default(),
    }
}

fn make_batch(batch_idx: usize, args: &Args) -> Vec<usize> {
    (0..args.batch_cells)
        .map(|offset| (batch_idx * args.batch_cells + offset) % args.cells)
        .collect()
}

fn make_batches(args: &Args) -> Vec<Vec<usize>> {
    (0..args.batches)
        .map(|batch_idx| make_batch(batch_idx, args))
        .collect()
}

fn selected_gene_names(count: usize) -> Vec<String> {
    (0..count).map(|idx| format!("gene-{idx}")).collect()
}

fn parse_cases_env(name: &str) -> Option<Vec<Case>> {
    std::env::var(name)
        .ok()
        .and_then(|value| parse_cases(&value).ok())
}

fn parse_cases(value: &str) -> Result<Vec<Case>, String> {
    let trimmed = value.trim();
    if trimmed.eq_ignore_ascii_case("all") {
        return Ok(Case::all());
    }
    trimmed
        .split(',')
        .map(Case::parse)
        .collect::<Result<Vec<_>, _>>()
}

fn next_arg(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_next<T: std::str::FromStr>(
    iter: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, String> {
    next_arg(iter, flag)?
        .parse::<T>()
        .map_err(|_| format!("invalid value for {flag}"))
}

fn print_help() {
    println!(
        "cargo bench --bench databank -- [--cells 2048] [--genes 4096] [--batch-cells 128] [--batches 16] [--workers 4] [--cases dense2d_raw,dense2d_lz4_genes,dense1d_owned,sparse_raw,sparse_lz4,dense2d_scheduled,sparse_scheduled]"
    );
}
