//! End-to-end DataBank scheduled-prefetch benchmarks.

mod support;

use std::process::ExitCode;

use _scdata::access::{AccessConfig, AccessCpuConfig, AccessProfile, ScheduledAccessConfig};
use _scdata::codecs::DecodePoolConfig;
use _scdata::databank::{
    ArrayCodecSpec, DType, DataBank, DataBankConfig, EdgeChunkLayout, FillConfig,
    ScheduledPrefetchConfig, SparseCsrSpec,
};
use _scdata::iopool::{BaseIoConfig, IoConfig, ThreadedConfig};
use support::chunks::{
    dense2d_u32_spec, gene_names, make_csr_u32_f32_chunks, make_csr_u32_f32_chunks_crc32,
    make_csr_u32_f32_chunks_lz4, make_dense_u32_chunks, make_dense_u32_chunks_lz4,
    regular_file_array_spec, sparse_csr_u32_f32_spec, write_chunks_file,
};
use support::codecs::crc32_encode;
use support::{
    bench_existing_profile_round, env_flag_default, env_usize, BenchConfig, ProfileEnvGuard,
};

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
struct FullchainConfig {
    cells: usize,
    genes: usize,
    batch_cells: usize,
    batches: usize,
    nnz_per_cell: usize,
    chunk_rows: usize,
    chunk_cols: usize,
    csr_chunk_len: usize,
    prefetch_step: usize,
    access_prefetch_step: usize,
    access_decode_ahead: usize,
    access_ready_ahead: usize,
    file_backed: bool,
}

#[derive(Debug, Clone, Copy)]
enum CaseCodec {
    Raw,
    Lz4,
    Crc32,
}

impl CaseCodec {
    fn name(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Lz4 => "lz4",
            Self::Crc32 => "crc32",
        }
    }

    fn array_codec(self) -> ArrayCodecSpec {
        match self {
            Self::Raw => ArrayCodecSpec::Uncompressed,
            Self::Lz4 => ArrayCodecSpec::CodecJson(r#"{"id":"lz4"}"#.to_string()),
            Self::Crc32 => ArrayCodecSpec::CodecJson(r#"{"id":"crc32"}"#.to_string()),
        }
    }
}

fn run() -> Result<(), String> {
    let bench_config = BenchConfig::from_env();
    let config = FullchainConfig::from_env();
    let codecs = codecs_from_env();
    println!(
        "fullchain/profiled cells={} genes={} batch_cells={} batches={} nnz_per_cell={} chunk={}x{} csr_chunk_len={} prefetch_step={} access=({}, {}, {}) file_backed={} codecs={:?}",
        config.cells,
        config.genes,
        config.batch_cells,
        config.batches,
        config.nnz_per_cell,
        config.chunk_rows,
        config.chunk_cols,
        config.csr_chunk_len,
        config.prefetch_step,
        config.access_prefetch_step,
        config.access_decode_ahead,
        config.access_ready_ahead,
        config.file_backed,
        codecs.iter().map(|codec| codec.name()).collect::<Vec<_>>()
    );

    for codec in codecs {
        run_codec_case(bench_config, &config, codec)?;
    }
    Ok(())
}

fn run_codec_case(
    bench_config: BenchConfig,
    config: &FullchainConfig,
    codec: CaseCodec,
) -> Result<(), String> {
    let _env = ProfileEnvGuard::enabled(&format!("fullchain-{}", codec.name()));
    let mut bank = DataBank::new(databank_config()).map_err(|err| err.to_string())?;
    let dense_id = if config.file_backed {
        let chunks = dense_chunks(config, codec);
        let written = write_chunks_file(&format!("fullchain-dense-{}", codec.name()), &chunks);
        let data = regular_file_array_spec(
            vec![config.cells, config.genes],
            vec![config.chunk_rows, config.chunk_cols],
            DType::U32,
            &written,
            codec.array_codec(),
            EdgeChunkLayout::Padded,
        );
        bank.register_dense_2d(_scdata::databank::Dense2DSpec {
            gene_names: gene_names(config.genes),
            data,
        })
        .map_err(|err| err.to_string())?
    } else {
        bank.register_dense_2d(dense2d_u32_spec(
            config.cells,
            config.genes,
            config.chunk_rows,
            config.chunk_cols,
            dense_chunks(config, codec),
            codec.array_codec(),
            EdgeChunkLayout::Padded,
        ))
        .map_err(|err| err.to_string())?
    };

    let sparse_id = if config.file_backed {
        let (indptr, indices, data) = csr_chunks(config, codec);
        let nnz = indptr.last().copied().unwrap_or(0) as usize;
        let index_written =
            write_chunks_file(&format!("fullchain-csr-idx-{}", codec.name()), &indices);
        let data_written =
            write_chunks_file(&format!("fullchain-csr-data-{}", codec.name()), &data);
        bank.register_sparse_csr(SparseCsrSpec {
            gene_names: gene_names(config.genes),
            indptr,
            indices: regular_file_array_spec(
                vec![nnz],
                vec![config.csr_chunk_len],
                DType::U32,
                &index_written,
                codec.array_codec(),
                EdgeChunkLayout::Cropped,
            ),
            data: regular_file_array_spec(
                vec![nnz],
                vec![config.csr_chunk_len],
                DType::F32,
                &data_written,
                codec.array_codec(),
                EdgeChunkLayout::Cropped,
            ),
            index_dtype: DType::U32,
            num_cells: config.cells,
            num_genes: config.genes,
        })
        .map_err(|err| err.to_string())?
    } else {
        let (indptr, indices, data) = csr_chunks(config, codec);
        bank.register_sparse_csr(sparse_csr_u32_f32_spec(
            config.cells,
            config.genes,
            indptr,
            config.csr_chunk_len,
            config.csr_chunk_len,
            indices,
            data,
            codec.array_codec(),
        ))
        .map_err(|err| err.to_string())?
    };

    let batches = make_batches(config);
    let dense_bytes = config.batch_cells * config.genes * std::mem::size_of::<u32>();
    let sparse_bytes = config.batch_cells * config.genes * std::mem::size_of::<f32>();
    let prefetch_config = ScheduledPrefetchConfig {
        prefetch_step: config.prefetch_step,
        access: ScheduledAccessConfig {
            prefetch_step: config.access_prefetch_step,
            decode_ahead_steps: config.access_decode_ahead,
            ready_ahead_steps: config.access_ready_ahead,
        },
    };

    bench_existing_profile_round(
        bench_config,
        &format!("fullchain/{}/dense2d_direct", codec.name()),
        8,
        Some(dense_bytes * config.batches),
        bank.profile(),
        || {
            batches.iter().fold(0usize, |acc, batch| {
                let mut out = vec![0u32; batch.len() * config.genes];
                bank.access_cells(dense_id, batch, &mut out, None)
                    .expect("dense direct");
                acc ^ out.len() ^ out[out.len() / 2] as usize
            })
        },
    );

    bench_existing_profile_round(
        bench_config,
        &format!("fullchain/{}/dense2d_scheduled", codec.name()),
        4,
        Some(dense_bytes * config.batches),
        bank.profile(),
        || {
            let prefetch = bank
                .prefetch_cells_scheduled::<u32, _>(dense_id, batches.clone(), prefetch_config)
                .expect("dense scheduled");
            prefetch.fold(0usize, |acc, result| {
                let batch = result.expect("dense batch");
                acc ^ batch.buffer.len() ^ batch.buffer[batch.buffer.len() / 2] as usize
            })
        },
    );

    bench_existing_profile_round(
        bench_config,
        &format!("fullchain/{}/sparse_csr_scheduled", codec.name()),
        4,
        Some(sparse_bytes * config.batches),
        bank.profile(),
        || {
            let prefetch = bank
                .prefetch_cells_scheduled::<f32, _>(sparse_id, batches.clone(), prefetch_config)
                .expect("sparse scheduled");
            prefetch.fold(0usize, |acc, result| {
                let batch = result.expect("sparse batch");
                acc ^ batch.buffer.len() ^ batch.buffer[batch.buffer.len() / 2].to_bits() as usize
            })
        },
    );

    Ok(())
}

fn dense_chunks(config: &FullchainConfig, codec: CaseCodec) -> Vec<std::sync::Arc<[u8]>> {
    match codec {
        CaseCodec::Raw => make_dense_u32_chunks(
            config.cells,
            config.genes,
            config.chunk_rows,
            config.chunk_cols,
        ),
        CaseCodec::Lz4 => make_dense_u32_chunks_lz4(
            config.cells,
            config.genes,
            config.chunk_rows,
            config.chunk_cols,
        ),
        CaseCodec::Crc32 => make_dense_u32_chunks(
            config.cells,
            config.genes,
            config.chunk_rows,
            config.chunk_cols,
        )
        .into_iter()
        .map(|chunk| crc32_encode(&chunk).into())
        .collect(),
    }
}

fn csr_chunks(config: &FullchainConfig, codec: CaseCodec) -> support::chunks::CsrU32F32Chunks {
    match codec {
        CaseCodec::Raw => make_csr_u32_f32_chunks(
            config.cells,
            config.genes,
            config.nnz_per_cell,
            config.csr_chunk_len,
        ),
        CaseCodec::Lz4 => make_csr_u32_f32_chunks_lz4(
            config.cells,
            config.genes,
            config.nnz_per_cell,
            config.csr_chunk_len,
        ),
        CaseCodec::Crc32 => make_csr_u32_f32_chunks_crc32(
            config.cells,
            config.genes,
            config.nnz_per_cell,
            config.csr_chunk_len,
        ),
    }
}

fn make_batches(config: &FullchainConfig) -> Vec<Vec<usize>> {
    (0..config.batches)
        .map(|batch_idx| {
            (0..config.batch_cells)
                .map(|offset| (batch_idx * config.batch_cells + offset) % config.cells)
                .collect()
        })
        .collect()
}

fn databank_config() -> DataBankConfig {
    DataBankConfig {
        io_config: IoConfig::Threaded(ThreadedConfig {
            base: BaseIoConfig {
                max_in_flight: 256,
                priority_levels: 2,
                queue_shards: 2,
                assume_non_overlapping_reads: true,
            },
            num_workers: 4,
            cpus: None,
        }),
        decode_config: DecodePoolConfig {
            num_workers: 4,
            queue_capacity: 512,
            cpus: None,
        },
        access_config: AccessConfig {
            queue_capacity: 512,
            scheduler_shards: 2,
            cache_capacity_bytes: 128 * 1024 * 1024,
            memory_budget_bytes: 256 * 1024 * 1024,
            default_io_priority: 0,
            keep_decoded: true,
            cpu: AccessCpuConfig {
                num_workers: 4,
                queue_capacity: 512,
                cpus: None,
            },
            profile: AccessProfile::from_env(),
        },
        fill_config: FillConfig {
            parallel: true,
            num_workers: 4,
            queue_capacity: 512,
            min_parallel_rows: 1,
            min_parallel_bytes: 1,
            cpus: None,
        },
    }
}

impl FullchainConfig {
    fn from_env() -> Self {
        let explicit = env_flag_default("SCDATA_FULLCHAIN_PREFETCH", false)
            || env_flag_default("SCDATA_FULLCHAIN_MATRIX", false);
        Self {
            cells: env_usize("SCDATA_FULLCHAIN_CELLS").unwrap_or(if explicit { 2048 } else { 256 }),
            genes: env_usize("SCDATA_FULLCHAIN_GENES").unwrap_or(if explicit { 4096 } else { 512 }),
            batch_cells: env_usize("SCDATA_FULLCHAIN_BATCH_CELLS").unwrap_or(if explicit {
                128
            } else {
                32
            }),
            batches: env_usize("SCDATA_FULLCHAIN_BATCHES").unwrap_or(if explicit { 32 } else { 4 }),
            nnz_per_cell: env_usize("SCDATA_FULLCHAIN_NNZ_PER_CELL").unwrap_or(if explicit {
                64
            } else {
                16
            }),
            chunk_rows: env_usize("SCDATA_FULLCHAIN_CHUNK_ROWS").unwrap_or(if explicit {
                128
            } else {
                32
            }),
            chunk_cols: env_usize("SCDATA_FULLCHAIN_CHUNK_COLS").unwrap_or(if explicit {
                256
            } else {
                64
            }),
            csr_chunk_len: env_usize("SCDATA_FULLCHAIN_CSR_CHUNK_LEN").unwrap_or(if explicit {
                4096
            } else {
                512
            }),
            prefetch_step: env_usize("SCDATA_FULLCHAIN_PREFETCH_STEP").unwrap_or(2),
            access_prefetch_step: env_usize("SCDATA_FULLCHAIN_ACCESS_PREFETCH_STEP").unwrap_or(3),
            access_decode_ahead: env_usize("SCDATA_FULLCHAIN_ACCESS_DECODE_AHEAD").unwrap_or(2),
            access_ready_ahead: env_usize("SCDATA_FULLCHAIN_ACCESS_READY_AHEAD").unwrap_or(1),
            file_backed: env_flag_default("SCDATA_FULLCHAIN_FILE_BACKED", true),
        }
    }
}

fn codecs_from_env() -> Vec<CaseCodec> {
    std::env::var("SCDATA_FULLCHAIN_CODECS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|token| match token.trim().to_ascii_lowercase().as_str() {
                    "raw" | "none" => Some(CaseCodec::Raw),
                    "lz4" => Some(CaseCodec::Lz4),
                    "crc32" => Some(CaseCodec::Crc32),
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .filter(|items| !items.is_empty())
        .unwrap_or_else(|| vec![CaseCodec::Raw, CaseCodec::Lz4])
}
