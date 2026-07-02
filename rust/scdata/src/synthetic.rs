//! Profile-only native synthetic IO benchmark helpers.
//!
//! This module is intentionally hidden behind the `profile` feature. It drives
//! the Blosc-LZ4 native loader with an in-memory `IoBackend`, so benchmarks can
//! stress decode/scatter/completion without being capped by GPFS bandwidth.

use std::collections::{BTreeMap, HashMap};
use std::ffi::CString;
use std::io;
use std::os::raw::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use serde_json::{json, Value};

use crate::access::{
    AccessConfig, AccessCpuConfig, AccessItem, ChunkKey, FileRef, IoBackend, IoTask,
    ScheduledAccessConfig, SliceSpec,
};
use crate::codecs::codec_from_json_str;
use crate::codecs::DecodePoolConfig;
use crate::databank::native::{
    build_blosc_lz4_block_index, load_access_item_blosc_lz4_native, NativeBlockIndexCache,
};
use crate::databank::{
    ArrayCodecSpec, ArrayGridSpec, ArrayOrder, ArraySpec, ChunkSourceSpec, ChunkSpec, DType,
    DataBank, DataBankConfig, EdgeChunkLayout, FillConfig, MissingGenePolicy, NativeAccessConfig,
    NativeLoadCoalesceConfig, NativeMode, ProjectedSparseDataGroupStrategy, RegisteredFile,
    ScheduledPrefetchConfig, SparseCsrSpec,
};
use crate::iopool::{IoConfig, ThreadedConfig};
use crate::profile::{ProfileMetricKind, ProfileSnapshot};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeSyntheticOrder {
    Random,
    Sequential,
    Continuity,
}

impl NativeSyntheticOrder {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "random" => Ok(Self::Random),
            "sequential" | "seq" | "fully-sequential" | "fully_sequential" => Ok(Self::Sequential),
            "continuity" | "continuity-p" | "continuity_p" => Ok(Self::Continuity),
            other => Err(format!(
                "order must be random, sequential, or continuity, got {other:?}"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Sequential => "sequential",
            Self::Continuity => "continuity",
        }
    }
}

#[derive(Debug, Clone)]
pub struct NativeSyntheticConfig {
    pub scheduled: bool,
    pub batches: usize,
    pub warmup_batches: usize,
    pub batch_size: usize,
    pub workers: usize,
    pub fill_workers: usize,
    pub native_workers: usize,
    pub io_workers: usize,
    pub chunks: usize,
    pub genes: usize,
    pub source_genes: usize,
    pub cells_per_chunk: usize,
    pub cell_bytes: usize,
    pub block_size: usize,
    pub typesize: usize,
    pub shuffle: bool,
    pub entropy_fraction: f32,
    pub order: NativeSyntheticOrder,
    pub continuation_p: f64,
    pub seed: u64,
    pub projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    pub scheduled_prefetch_step: usize,
    pub scheduled_access_prefetch_step: usize,
    pub scheduled_decode_ahead_steps: usize,
    pub scheduled_ready_ahead_steps: usize,
    pub coalesce: NativeLoadCoalesceConfig,
}

impl Default for NativeSyntheticConfig {
    fn default() -> Self {
        Self {
            scheduled: false,
            batches: 2048,
            warmup_batches: 0,
            batch_size: 128,
            workers: 1,
            fill_workers: 0,
            native_workers: 0,
            io_workers: 0,
            chunks: 1,
            genes: 4096,
            source_genes: 0,
            cells_per_chunk: 2048,
            cell_bytes: 12 * 1024,
            block_size: 192 * 1024,
            typesize: 2,
            shuffle: true,
            entropy_fraction: 0.33,
            order: NativeSyntheticOrder::Random,
            continuation_p: 0.0,
            seed: 0x5eed_5eed_1234_5678,
            projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy::SelectedOnly,
            scheduled_prefetch_step: 128,
            scheduled_access_prefetch_step: 1024,
            scheduled_decode_ahead_steps: 512,
            scheduled_ready_ahead_steps: 256,
            coalesce: NativeLoadCoalesceConfig {
                max_window_us: 0,
                max_merged_len: 8 * 1024 * 1024,
                max_gap_bytes: 1024 * 1024,
                max_waste_ratio: 0.90,
                min_children: 2,
            },
        }
    }
}

pub fn run_native_synthetic(config: NativeSyntheticConfig) -> Result<Value, String> {
    validate_config(&config)?;
    if config.scheduled {
        return run_native_scheduled_synthetic(config);
    }

    let decoded_bytes = config
        .cells_per_chunk
        .checked_mul(config.cell_bytes)
        .ok_or_else(|| "cells_per_chunk * cell_bytes overflow".to_string())?;
    let raw = synthetic_decoded_payload(decoded_bytes, config.entropy_fraction, config.seed);
    let encoded = blosc_lz4_encode(
        &raw,
        if config.shuffle {
            blosc_src::BLOSC_SHUFFLE as i32
        } else {
            0
        },
        config.typesize,
        config.block_size,
    )?;
    drop(raw);

    let encoded: Arc<[u8]> = Arc::from(encoded.into_boxed_slice());
    let file = FileRef::new(91);
    let io = Arc::new(VirtualChunkIo::new(
        file,
        Arc::clone(&encoded),
        config_chunk_stride(&config),
    ));
    let cache = Arc::new(NativeBlockIndexCache::new());
    let codec = codec_from_json_str(r#"{"id":"blosc","cname":"lz4"}"#)
        .map_err(|err| format!("build Blosc codec: {err}"))?;
    let cells = build_cell_stream(&config);
    let total_cells = config.chunks * config.cells_per_chunk;
    let reuse = estimate_block_reuse(&config, &cells);

    let start = Instant::now();
    let mut handles = Vec::with_capacity(config.workers);
    for worker_idx in 0..config.workers {
        let cells = cells.clone();
        let io: Arc<dyn IoBackend> = io.clone();
        let cache = Arc::clone(&cache);
        let codec = codec.clone();
        let coalesce = config.coalesce.clone();
        let encoded_len = encoded.len();
        let config = config.clone();
        handles.push(thread::spawn(move || -> Result<WorkerStats, String> {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|err| format!("build tokio runtime: {err}"))?;
            let mut stats = WorkerStats::default();
            for batch_idx in (worker_idx..config.batches).step_by(config.workers) {
                let offset = batch_idx * config.batch_size;
                let batch_cells = &cells[offset..offset + config.batch_size];
                let batch_stats = runtime.block_on(process_synthetic_batch(
                    batch_cells,
                    &config,
                    io.clone(),
                    Arc::clone(&cache),
                    codec.clone(),
                    coalesce.clone(),
                    encoded_len,
                ))?;
                stats.merge(batch_stats);
            }
            Ok(stats)
        }));
    }

    let mut stats = WorkerStats::default();
    for handle in handles {
        let worker = handle
            .join()
            .map_err(|_| "synthetic worker thread panicked".to_string())??;
        stats.merge(worker);
    }
    let elapsed = start.elapsed();

    let elapsed_s = elapsed.as_secs_f64();
    let batches_per_s = config.batches as f64 / elapsed_s;
    let output_bytes_per_batch = stats.output_bytes as f64 / config.batches as f64;
    let read_bytes = io.read_bytes();
    let read_ops = io.read_ops();
    let read_bytes_per_batch = read_bytes as f64 / config.batches as f64;
    let read_ops_per_batch = read_ops as f64 / config.batches as f64;
    let encoded_ratio = decoded_bytes as f64 / encoded.len() as f64;

    Ok(json!({
        "config": {
            "batches": config.batches,
            "batch_size": config.batch_size,
            "workers": config.workers,
            "fill_workers": effective_workers(config.fill_workers, config.workers),
            "native_workers": effective_workers(config.native_workers, config.workers),
            "io_workers": effective_workers(config.io_workers, config.workers),
            "chunks": config.chunks,
            "cells_per_chunk": config.cells_per_chunk,
            "cell_bytes": config.cell_bytes,
            "decoded_chunk_bytes": decoded_bytes,
            "encoded_chunk_bytes": encoded.len(),
            "block_size": config.block_size,
            "typesize": config.typesize,
            "shuffle": config.shuffle,
            "entropy_fraction": config.entropy_fraction,
            "order": config.order.as_str(),
            "continuation_p": config.continuation_p,
            "seed": config.seed,
            "coalesce": {
                "max_window_us": config.coalesce.max_window_us,
                "max_merged_len": config.coalesce.max_merged_len,
                "max_gap_bytes": config.coalesce.max_gap_bytes,
                "max_waste_ratio": config.coalesce.max_waste_ratio,
                "min_children": config.coalesce.min_children,
            },
        },
        "shape": {
            "total_virtual_cells": total_cells,
            "csr_raw_bytes_per_cell_model": config.cell_bytes,
            "encoded_compression_ratio": encoded_ratio,
            "encoded_bytes_per_cell": encoded.len() as f64 / config.cells_per_chunk as f64,
        },
        "throughput": {
            "elapsed_s": elapsed_s,
            "batches_per_s": batches_per_s,
            "cells_per_s": batches_per_s * config.batch_size as f64,
            "output_bytes_per_batch": output_bytes_per_batch,
            "output_gib_per_s": stats.output_bytes as f64 / elapsed_s / 1024.0 / 1024.0 / 1024.0,
        },
        "virtual_io": {
            "read_bytes": read_bytes,
            "read_ops": read_ops,
            "read_bytes_per_batch": read_bytes_per_batch,
            "read_ops_per_batch": read_ops_per_batch,
            "read_gib_per_s": read_bytes as f64 / elapsed_s / 1024.0 / 1024.0 / 1024.0,
            "equivalent_batch_s_at_20gbps": 20_000_000_000.0 / read_bytes_per_batch.max(1.0),
            "equivalent_batch_s_at_50gbps": 50_000_000_000.0 / read_bytes_per_batch.max(1.0),
            "equivalent_batch_s_at_100gbps": 100_000_000_000.0 / read_bytes_per_batch.max(1.0),
        },
        "reuse": {
            "cell_block_touches": reuse.cell_block_touches,
            "unique_blocks_per_batch_mean": reuse.unique_blocks_per_batch_mean,
            "block_reuse_ratio": reuse.block_reuse_ratio,
        },
        "completion": {
            "processed_batches": stats.batches,
            "processed_items": stats.items,
            "output_bytes": stats.output_bytes,
            "checksum": stats.checksum,
        },
        "targets": target_report(config.order, batches_per_s),
    }))
}

fn run_native_scheduled_synthetic(config: NativeSyntheticConfig) -> Result<Value, String> {
    let nnz_per_cell = config
        .cell_bytes
        .checked_div(6)
        .ok_or_else(|| "cell_bytes / 6 overflow".to_string())?;
    if nnz_per_cell == 0 || nnz_per_cell * 6 != config.cell_bytes {
        return Err(
            "scheduled synthetic requires cell_bytes divisible by 6 (i32 index + u16 data)"
                .to_string(),
        );
    }
    let source_genes = config.source_genes.max(config.genes);
    if nnz_per_cell > source_genes {
        return Err(format!(
            "scheduled synthetic nnz_per_cell {nnz_per_cell} exceeds source_genes {source_genes}",
        ));
    }

    let cells_per_chunk = config.cells_per_chunk;
    let nnz_per_chunk = cells_per_chunk
        .checked_mul(nnz_per_cell)
        .ok_or_else(|| "cells_per_chunk * nnz_per_cell overflow".to_string())?;
    let total_cells = config
        .chunks
        .checked_mul(config.cells_per_chunk)
        .ok_or_else(|| "chunks * cells_per_chunk overflow".to_string())?;
    let total_nnz = total_cells
        .checked_mul(nnz_per_cell)
        .ok_or_else(|| "total_cells * nnz_per_cell overflow".to_string())?;

    let indices_raw = synthetic_indices_payload(cells_per_chunk, nnz_per_cell, source_genes);
    let data_raw = synthetic_decoded_payload_distributed(
        nnz_per_chunk * 2,
        config.entropy_fraction,
        config.seed ^ 0xdada_dada_1234_5678,
    );
    let shuffle = if config.shuffle {
        blosc_src::BLOSC_SHUFFLE as i32
    } else {
        0
    };
    let indices_encoded: Arc<[u8]> = Arc::from(
        blosc_lz4_encode(&indices_raw, shuffle, 4, config.block_size)?.into_boxed_slice(),
    );
    let data_encoded: Arc<[u8]> =
        Arc::from(blosc_lz4_encode(&data_raw, shuffle, 2, config.block_size)?.into_boxed_slice());
    drop(indices_raw);
    drop(data_raw);

    let indices_file = RegisteredFile::new(101).map_err(|err| err.to_string())?;
    let data_file = RegisteredFile::new(102).map_err(|err| err.to_string())?;
    let indices_stride = indices_encoded.len() as u64 + 4096;
    let data_stride = data_encoded.len() as u64 + 4096;
    let io = Arc::new(VirtualChunkIo::from_files([
        (
            indices_file.file_ref,
            Arc::clone(&indices_encoded),
            indices_stride,
        ),
        (data_file.file_ref, Arc::clone(&data_encoded), data_stride),
    ]));

    let mut bank =
        DataBank::new(scheduled_databank_config(&config)).map_err(|err| err.to_string())?;
    let io_backend: Arc<dyn IoBackend> = io.clone();
    bank.set_native_io_override_for_profile(io_backend);
    let dataset = scheduled_synthetic_sparse_spec(
        &config,
        nnz_per_cell,
        source_genes,
        total_cells,
        total_nnz,
        indices_file,
        indices_encoded.len(),
        indices_stride,
        data_file,
        data_encoded.len(),
        data_stride,
    )?;
    let dataset_id = bank
        .register_sparse_csr(dataset)
        .map_err(|err| err.to_string())?;

    let requested_gene_names = (0..config.genes)
        .map(|idx| format!("gene{idx}"))
        .collect::<Vec<_>>();
    let scheduled_config = ScheduledPrefetchConfig {
        prefetch_step: config.scheduled_prefetch_step,
        access: ScheduledAccessConfig {
            prefetch_step: config.scheduled_access_prefetch_step,
            decode_ahead_steps: config.scheduled_decode_ahead_steps,
            ready_ahead_steps: config.scheduled_ready_ahead_steps,
        },
        projected_sparse_data_strategy: config.projected_sparse_data_strategy,
        native_mode: NativeMode::Force,
    };

    if config.warmup_batches > 0 {
        let (_, warmup_batches) = scheduled_synthetic_batches(&config, config.warmup_batches);
        let _ = consume_scheduled_synthetic_stream(
            &bank,
            dataset_id,
            warmup_batches,
            &requested_gene_names,
            scheduled_config,
        )?;
    }

    let (measured_cells_stream, measured_batch_source) =
        scheduled_synthetic_batches(&config, config.batches);
    let reuse = estimate_block_reuse(&config, &measured_cells_stream);
    io.reset_counters();
    bank.reset_runtime_profiles();
    let started = Instant::now();
    let measured = consume_scheduled_synthetic_stream(
        &bank,
        dataset_id,
        measured_batch_source,
        &requested_gene_names,
        scheduled_config,
    )?;
    let elapsed_s = started.elapsed().as_secs_f64();
    let read_bytes = io.read_bytes();
    let read_ops = io.read_ops();
    let batches_per_s = measured.batches as f64 / elapsed_s.max(f64::MIN_POSITIVE);
    let read_bytes_per_batch = read_bytes as f64 / measured.batches.max(1) as f64;
    let databank_profile = profile_snapshot_json(bank.profile_snapshot_and_reset());
    let access_profile = profile_snapshot_json(bank.access_profile_snapshot_and_reset());
    let io_profile = profile_snapshot_json(bank.io_profile_snapshot_and_reset());
    let decode_profile = profile_snapshot_json(bank.decode_profile_snapshot_and_reset());

    Ok(json!({
        "mode": "scheduled",
        "config": {
            "batches": config.batches,
            "warmup_batches": config.warmup_batches,
            "batch_size": config.batch_size,
            "workers": config.workers,
            "fill_workers": effective_workers(config.fill_workers, config.workers),
            "native_workers": effective_workers(config.native_workers, config.workers),
            "io_workers": effective_workers(config.io_workers, config.workers),
            "chunks": config.chunks,
            "cells_per_chunk": config.cells_per_chunk,
            "genes": config.genes,
            "source_genes": source_genes,
            "cell_bytes": config.cell_bytes,
            "nnz_per_cell": nnz_per_cell,
            "block_size": config.block_size,
            "shuffle": config.shuffle,
            "order": config.order.as_str(),
            "continuation_p": config.continuation_p,
            "scheduled_prefetch_step": config.scheduled_prefetch_step,
            "scheduled_access_prefetch_step": config.scheduled_access_prefetch_step,
            "scheduled_decode_ahead_steps": config.scheduled_decode_ahead_steps,
            "scheduled_ready_ahead_steps": config.scheduled_ready_ahead_steps,
            "projected_sparse_data_strategy": config.projected_sparse_data_strategy.as_str(),
            "coalesce": {
                "max_window_us": config.coalesce.max_window_us,
                "max_merged_len": config.coalesce.max_merged_len,
                "max_gap_bytes": config.coalesce.max_gap_bytes,
                "max_waste_ratio": config.coalesce.max_waste_ratio,
                "min_children": config.coalesce.min_children,
            },
        },
        "shape": {
            "total_virtual_cells": total_cells,
            "total_nnz": total_nnz,
            "indices_encoded_chunk_bytes": indices_encoded.len(),
            "data_encoded_chunk_bytes": data_encoded.len(),
            "indices_compression_ratio": (nnz_per_chunk * 4) as f64 / indices_encoded.len() as f64,
            "data_compression_ratio": (nnz_per_chunk * 2) as f64 / data_encoded.len() as f64,
        },
        "throughput": {
            "elapsed_s": elapsed_s,
            "batches_per_s": batches_per_s,
            "cells_per_s": batches_per_s * config.batch_size as f64,
            "output_bytes": measured.output_bytes,
            "output_bytes_per_batch": measured.output_bytes as f64 / measured.batches.max(1) as f64,
            "output_gib_per_s": measured.output_bytes as f64 / elapsed_s.max(f64::MIN_POSITIVE) / 1024.0 / 1024.0 / 1024.0,
        },
        "virtual_io": {
            "read_bytes": read_bytes,
            "read_ops": read_ops,
            "read_bytes_per_batch": read_bytes_per_batch,
            "read_ops_per_batch": read_ops as f64 / measured.batches.max(1) as f64,
            "read_gb_per_s": read_bytes as f64 / elapsed_s.max(f64::MIN_POSITIVE) / 1e9,
            "equivalent_batch_s_at_20gbps": 20_000_000_000.0 / read_bytes_per_batch.max(1.0),
            "equivalent_batch_s_at_50gbps": 50_000_000_000.0 / read_bytes_per_batch.max(1.0),
            "equivalent_batch_s_at_100gbps": 100_000_000_000.0 / read_bytes_per_batch.max(1.0),
        },
        "reuse": {
            "cell_block_touches": reuse.cell_block_touches,
            "unique_blocks_per_batch_mean": reuse.unique_blocks_per_batch_mean,
            "block_reuse_ratio": reuse.block_reuse_ratio,
        },
        "completion": {
            "processed_batches": measured.batches,
            "checksum": measured.checksum,
        },
        "profiles": {
            "databank": databank_profile,
            "access": access_profile,
            "io": io_profile,
            "decode": decode_profile,
        },
    }))
}

fn profile_snapshot_json(snapshot: ProfileSnapshot) -> Value {
    let mut metrics = serde_json::Map::new();
    for metric in &snapshot.metrics {
        let name = format!("{}.{}", metric.id.scope, metric.id.name)
            .replace('.', "_")
            .replace('-', "_");
        let value = metric.value();
        let metric_value = match metric.id.kind {
            ProfileMetricKind::Bytes => json!({
                "kind": metric.id.kind.as_str(),
                "value": value,
                "mib": metric.as_mib(),
            }),
            ProfileMetricKind::DurationNs => json!({
                "kind": metric.id.kind.as_str(),
                "value": value,
                "ms": metric.as_ms(),
            }),
            _ => json!({
                "kind": metric.id.kind.as_str(),
                "value": value,
            }),
        };
        metrics.insert(name, metric_value);
    }
    json!({
        "label": snapshot.label,
        "round": snapshot.round,
        "elapsed_ms": snapshot.elapsed_ms(),
        "global_enabled": snapshot.global_enabled,
        "components": snapshot.components.len(),
        "scopes": snapshot.scopes.len(),
        "enabled_scopes": snapshot.enabled_scope_count(),
        "metrics": metrics,
    })
}

fn scheduled_synthetic_batches(
    config: &NativeSyntheticConfig,
    batches: usize,
) -> (Arc<[usize]>, Vec<Vec<usize>>) {
    let stream_config = NativeSyntheticConfig {
        batches,
        ..config.clone()
    };
    let cells = build_cell_stream(&stream_config);
    let batch_source = cells
        .chunks_exact(config.batch_size)
        .map(|batch| batch.to_vec())
        .collect::<Vec<_>>();
    (cells, batch_source)
}

fn consume_scheduled_synthetic_stream(
    bank: &DataBank,
    dataset_id: crate::databank::DatasetId,
    batches: Vec<Vec<usize>>,
    gene_names: &[String],
    config: ScheduledPrefetchConfig,
) -> Result<ScheduledSyntheticStats, String> {
    let expected_batches = batches.len();
    let mut stream = bank
        .prefetch_cells_scheduled_by_gene_names::<u16, _, _>(
            dataset_id,
            batches,
            gene_names,
            MissingGenePolicy::Zero,
            config,
        )
        .map_err(|err| err.to_string())?;
    let mut stats = ScheduledSyntheticStats::default();
    for _ in 0..expected_batches {
        let batch = stream
            .next()
            .ok_or_else(|| "scheduled synthetic stream ended early".to_string())?
            .map_err(|err| err.to_string())?;
        stats.batches += 1;
        stats.output_bytes += (batch.buffer.len() * std::mem::size_of::<u16>()) as u64;
        stats.checksum = stats.checksum.wrapping_add(checksum_u16(&batch.buffer));
    }
    Ok(stats)
}

async fn process_synthetic_batch(
    batch_cells: &[usize],
    config: &NativeSyntheticConfig,
    io: Arc<dyn IoBackend>,
    cache: Arc<NativeBlockIndexCache>,
    codec: crate::codecs::SharedCodec,
    coalesce: NativeLoadCoalesceConfig,
    encoded_len: usize,
) -> Result<WorkerStats, String> {
    let mut by_chunk: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    for (row, &global_cell) in batch_cells.iter().enumerate() {
        let chunk = global_cell / config.cells_per_chunk;
        let cell = global_cell % config.cells_per_chunk;
        by_chunk.entry(chunk).or_default().push((row, cell));
    }

    let mut stats = WorkerStats {
        batches: 1,
        ..WorkerStats::default()
    };
    for (chunk, entries) in by_chunk {
        let mut triples = Vec::with_capacity(entries.len() * 3);
        for (local_row, (_batch_row, cell)) in entries.iter().enumerate() {
            let dst = local_row * config.cell_bytes;
            let src = cell * config.cell_bytes;
            triples.push(dst);
            triples.push(src);
            triples.push(src + config.cell_bytes);
        }
        let slice = SliceSpec::from_triples(triples).map_err(|err| err.to_string())?;
        let key = ChunkKey::new(
            FileRef::new(91),
            chunk as u64 * config_chunk_stride(config),
            encoded_len,
        );
        let item = AccessItem::new(
            key,
            codec.clone(),
            Some(config.cells_per_chunk * config.cell_bytes),
        )
        .with_slice_spec(slice);
        let output = load_access_item_blosc_lz4_native(
            io.clone(),
            coalesce.clone(),
            &cache,
            None,
            None,
            &item,
            0,
        )
        .await
        .map_err(|err| err.to_string())?
        .ok_or_else(|| "native synthetic item fell back to generic path".to_string())?;
        stats.items += 1;
        stats.output_bytes += output.len() as u64;
        stats.checksum = stats.checksum.wrapping_add(checksum(&output));
    }
    Ok(stats)
}

fn validate_config(config: &NativeSyntheticConfig) -> Result<(), String> {
    if config.batches == 0 {
        return Err("batches must be greater than 0".to_string());
    }
    config
        .batches
        .checked_add(config.warmup_batches)
        .ok_or_else(|| "batches + warmup_batches overflow".to_string())?;
    if config.batch_size == 0 {
        return Err("batch_size must be greater than 0".to_string());
    }
    if config.workers == 0 {
        return Err("workers must be greater than 0".to_string());
    }
    if config.chunks == 0 {
        return Err("chunks must be greater than 0".to_string());
    }
    if config.genes == 0 {
        return Err("genes must be greater than 0".to_string());
    }
    if config.source_genes != 0 && config.source_genes < config.genes {
        return Err("source_genes must be zero or >= genes".to_string());
    }
    if config.cells_per_chunk == 0 {
        return Err("cells_per_chunk must be greater than 0".to_string());
    }
    if config.cell_bytes == 0 {
        return Err("cell_bytes must be greater than 0".to_string());
    }
    if config.block_size == 0 {
        return Err("block_size must be greater than 0".to_string());
    }
    if config.typesize == 0 {
        return Err("typesize must be greater than 0".to_string());
    }
    if !(0.0..=1.0).contains(&config.entropy_fraction) {
        return Err("entropy_fraction must be in [0, 1]".to_string());
    }
    if !(0.0..=1.0).contains(&config.continuation_p) {
        return Err("continuation_p must be in [0, 1]".to_string());
    }
    if config.scheduled {
        if config.cell_bytes % 6 != 0 {
            return Err(
                "scheduled synthetic requires cell_bytes divisible by 6 (i32 index + u16 data)"
                    .to_string(),
            );
        }
        let source_genes = config.source_genes.max(config.genes);
        if config.cell_bytes / 6 > source_genes {
            return Err("scheduled synthetic nnz_per_cell must be <= source_genes".to_string());
        }
        if config.scheduled_prefetch_step == 0
            || config.scheduled_access_prefetch_step == 0
            || config.scheduled_decode_ahead_steps == 0
            || config.scheduled_ready_ahead_steps == 0
        {
            return Err("scheduled prefetch/decode/ready values must be positive".to_string());
        }
    }
    config.coalesce.validate()
}

fn synthetic_decoded_payload(len: usize, entropy_fraction: f32, seed: u64) -> Vec<u8> {
    let entropy_len = ((len as f64) * entropy_fraction as f64).round() as usize;
    let mut out = vec![0u8; len];
    let mut rng = XorShift64::new(seed);
    for byte in out.iter_mut().take(entropy_len.min(len)) {
        *byte = rng.next_u64() as u8;
    }
    out
}

fn synthetic_decoded_payload_distributed(len: usize, entropy_fraction: f32, seed: u64) -> Vec<u8> {
    if entropy_fraction <= 0.0 {
        return vec![0u8; len];
    }
    if entropy_fraction >= 1.0 {
        return synthetic_decoded_payload(len, 1.0, seed);
    }
    let mut out = vec![0u8; len];
    let mut rng = XorShift64::new(seed);
    let threshold = entropy_fraction as f64;
    for byte in &mut out {
        if rng.next_f64() < threshold {
            *byte = rng.next_u64() as u8;
        }
    }
    out
}

fn build_cell_stream(config: &NativeSyntheticConfig) -> Arc<[usize]> {
    let total = config.batches * config.batch_size;
    let virtual_cells = config.chunks * config.cells_per_chunk;
    let mut rng = XorShift64::new(config.seed ^ 0xa5a5_5a5a_d00d_f00d);
    let mut cells = Vec::with_capacity(total);
    match config.order {
        NativeSyntheticOrder::Random => {
            for _ in 0..total {
                cells.push((rng.next_u64() as usize) % virtual_cells);
            }
        }
        NativeSyntheticOrder::Sequential => {
            let start = (rng.next_u64() as usize) % virtual_cells;
            for idx in 0..total {
                cells.push((start + idx) % virtual_cells);
            }
        }
        NativeSyntheticOrder::Continuity => {
            let mut current = (rng.next_u64() as usize) % virtual_cells;
            for _ in 0..total {
                cells.push(current);
                let continue_run = rng.next_f64() < config.continuation_p;
                if continue_run && current + 1 < virtual_cells {
                    current += 1;
                } else {
                    current = (rng.next_u64() as usize) % virtual_cells;
                }
            }
        }
    }
    Arc::from(cells.into_boxed_slice())
}

fn estimate_block_reuse(config: &NativeSyntheticConfig, cells: &[usize]) -> ReuseStats {
    let blocks_per_chunk = config
        .cells_per_chunk
        .checked_mul(config.cell_bytes)
        .unwrap_or(0)
        .div_ceil(config.block_size);
    let mut total_touches = 0u64;
    let mut total_unique = 0u64;
    for batch in cells.chunks_exact(config.batch_size) {
        let mut unique = std::collections::BTreeSet::new();
        for &global_cell in batch {
            let chunk = global_cell / config.cells_per_chunk;
            let cell = global_cell % config.cells_per_chunk;
            let start = cell * config.cell_bytes;
            let end = start + config.cell_bytes;
            let first = start / config.block_size;
            let last = (end - 1) / config.block_size;
            for block in first..=last {
                total_touches += 1;
                unique.insert(chunk * blocks_per_chunk + block);
            }
        }
        total_unique += unique.len() as u64;
    }
    let block_reuse_ratio = if total_touches == 0 {
        0.0
    } else {
        1.0 - (total_unique as f64 / total_touches as f64)
    };
    ReuseStats {
        cell_block_touches: total_touches,
        unique_blocks_per_batch_mean: total_unique as f64 / config.batches as f64,
        block_reuse_ratio,
    }
}

fn blosc_lz4_encode(
    raw: &[u8],
    shuffle: i32,
    typesize: usize,
    blocksize: usize,
) -> Result<Vec<u8>, String> {
    let compressor = CString::new("lz4").expect("static compressor name");
    let mut encoded = vec![0u8; raw.len() + blosc_src::BLOSC_MAX_OVERHEAD as usize];
    let written = unsafe {
        blosc_src::blosc_compress_ctx(
            5,
            shuffle,
            typesize,
            raw.len(),
            raw.as_ptr().cast::<c_void>(),
            encoded.as_mut_ptr().cast::<c_void>(),
            encoded.len(),
            compressor.as_ptr(),
            blocksize,
            1,
        )
    };
    if written <= 0 {
        return Err(format!("Blosc LZ4 compression failed with code {written}"));
    }
    encoded.truncate(written as usize);
    Ok(encoded)
}

fn target_report(order: NativeSyntheticOrder, batches_per_s: f64) -> Value {
    match order {
        NativeSyntheticOrder::Random => json!({
            "random_50gbps_2000bps": batches_per_s >= 2000.0,
            "random_100gbps_4000bps": batches_per_s >= 4000.0,
        }),
        NativeSyntheticOrder::Sequential | NativeSyntheticOrder::Continuity => json!({
            "sequential_20gbps_baseline_8000bps": batches_per_s >= 8000.0,
            "sequential_20gbps_target_16000bps": batches_per_s >= 16000.0,
            "sequential_50gbps_baseline_20000bps": batches_per_s >= 20000.0,
            "sequential_50gbps_target_40000bps": batches_per_s >= 40000.0,
            "sequential_100gbps_baseline_40000bps": batches_per_s >= 40000.0,
            "sequential_100gbps_target_80000bps": batches_per_s >= 80000.0,
        }),
    }
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .step_by(4096)
        .fold(bytes.len() as u64, |acc, byte| {
            acc.wrapping_mul(16_777_619).wrapping_add(*byte as u64)
        })
}

fn config_chunk_stride(config: &NativeSyntheticConfig) -> u64 {
    // The virtual backend maps every chunk offset to the same encoded payload.
    // The exact stride only needs to be stable and larger than the encoded len;
    // the backend takes the modulo against the stride.
    let decoded = config.cells_per_chunk * config.cell_bytes;
    (decoded + blosc_src::BLOSC_MAX_OVERHEAD as usize + 4096) as u64
}

#[allow(clippy::too_many_arguments)]
fn scheduled_synthetic_sparse_spec(
    config: &NativeSyntheticConfig,
    nnz_per_cell: usize,
    source_genes: usize,
    total_cells: usize,
    total_nnz: usize,
    indices_file: RegisteredFile,
    indices_encoded_len: usize,
    indices_stride: u64,
    data_file: RegisteredFile,
    data_encoded_len: usize,
    data_stride: u64,
) -> Result<SparseCsrSpec, String> {
    let nnz_per_chunk = config
        .cells_per_chunk
        .checked_mul(nnz_per_cell)
        .ok_or_else(|| "cells_per_chunk * nnz_per_cell overflow".to_string())?;
    let indptr = (0..=total_cells)
        .map(|cell| {
            cell.checked_mul(nnz_per_cell)
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| "indptr value overflow".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let gene_names = (0..source_genes)
        .map(|idx| format!("gene{idx}"))
        .collect::<Vec<_>>();
    Ok(SparseCsrSpec {
        gene_names,
        indptr,
        indices: scheduled_synthetic_array_spec(
            total_nnz,
            DType::I32,
            nnz_per_chunk,
            config.chunks,
            indices_file,
            indices_encoded_len,
            indices_stride,
            nnz_per_chunk * 4,
        ),
        data: scheduled_synthetic_array_spec(
            total_nnz,
            DType::U16,
            nnz_per_chunk,
            config.chunks,
            data_file,
            data_encoded_len,
            data_stride,
            nnz_per_chunk * 2,
        ),
        index_dtype: DType::I32,
        num_cells: total_cells,
        num_genes: source_genes,
    })
}

#[allow(clippy::too_many_arguments)]
fn scheduled_synthetic_array_spec(
    total_len: usize,
    dtype: DType,
    chunk_len: usize,
    chunks: usize,
    file: RegisteredFile,
    encoded_len: usize,
    stride: u64,
    decoded_bytes: usize,
) -> ArraySpec {
    let chunk_specs = (0..chunks)
        .map(|chunk| ChunkSpec {
            source: ChunkSourceSpec::RegisteredFile {
                file,
                offset: chunk as u64 * stride,
                len: encoded_len,
            },
            decoded_bytes,
        })
        .collect::<Vec<_>>();
    ArraySpec {
        shape: vec![total_len],
        dtype,
        order: ArrayOrder::C,
        codec: ArrayCodecSpec::CodecJson(r#"{"id":"blosc","cname":"lz4"}"#.to_string()),
        grid: ArrayGridSpec::Regular {
            chunk_shape: vec![chunk_len],
            edge: EdgeChunkLayout::Cropped,
        },
        chunks: chunk_specs,
    }
}

fn scheduled_databank_config(config: &NativeSyntheticConfig) -> DataBankConfig {
    let memory_budget_bytes = 512usize * 1024 * 1024 * 1024;
    let fill_workers = effective_workers(config.fill_workers, config.workers);
    let native_workers = effective_workers(config.native_workers, config.workers);
    let io_workers = effective_workers(config.io_workers, config.workers);
    DataBankConfig {
        io_config: IoConfig::Threaded(ThreadedConfig {
            num_workers: io_workers,
            ..ThreadedConfig::default()
        }),
        decode_config: DecodePoolConfig {
            num_workers: 1,
            ..DecodePoolConfig::default()
        },
        access_config: AccessConfig {
            scheduler_shards: 1,
            cache_capacity_bytes: memory_budget_bytes / 8,
            memory_budget_bytes,
            cpu: AccessCpuConfig {
                num_workers: 1,
                ..AccessCpuConfig::default()
            },
            ..AccessConfig::default()
        },
        fill_config: FillConfig {
            num_workers: fill_workers,
            ..FillConfig::default()
        },
        native_config: NativeAccessConfig {
            enabled: true,
            fused_workers: native_workers,
            request_prefetch_blocks: 16_384,
            memory_budget_bytes,
            response_queue_bytes_soft_limit: memory_budget_bytes / 2,
            response_queue_bytes_hard_limit: memory_budget_bytes * 3 / 4,
            load: crate::databank::NativeLoadConfig {
                scheduler_workers: native_workers,
                io_workers,
                coalesce: config.coalesce.clone(),
            },
            ..NativeAccessConfig::default()
        },
    }
}

fn effective_workers(explicit: usize, fallback: usize) -> usize {
    if explicit == 0 {
        fallback.max(1)
    } else {
        explicit
    }
}

fn synthetic_indices_payload(
    cells_per_chunk: usize,
    nnz_per_cell: usize,
    source_genes: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; cells_per_chunk * nnz_per_cell * 4];
    let step = (source_genes / nnz_per_cell).max(1);
    for cell in 0..cells_per_chunk {
        for gene in 0..nnz_per_cell {
            let offset = (cell * nnz_per_cell + gene) * 4;
            let source_gene = gene.saturating_mul(step).min(source_genes - 1);
            out[offset..offset + 4].copy_from_slice(&(source_gene as i32).to_le_bytes());
        }
    }
    out
}

fn checksum_u16(values: &[u16]) -> u64 {
    values
        .iter()
        .step_by(4096)
        .fold(values.len() as u64, |acc, value| {
            acc.wrapping_mul(16_777_619).wrapping_add(*value as u64)
        })
}

#[derive(Debug)]
struct VirtualChunkIo {
    files: HashMap<FileRef, VirtualFile>,
    read_bytes: AtomicU64,
    read_ops: AtomicU64,
}

impl VirtualChunkIo {
    fn new(file: FileRef, encoded: Arc<[u8]>, stride: u64) -> Self {
        Self::from_files([(file, encoded, stride)])
    }

    fn from_files<I>(files: I) -> Self
    where
        I: IntoIterator<Item = (FileRef, Arc<[u8]>, u64)>,
    {
        Self {
            files: files
                .into_iter()
                .map(|(file, encoded, stride)| {
                    (
                        file,
                        VirtualFile {
                            exact_ranges: precompute_virtual_exact_ranges(&encoded),
                            encoded,
                            stride,
                        },
                    )
                })
                .collect(),
            read_bytes: AtomicU64::new(0),
            read_ops: AtomicU64::new(0),
        }
    }

    fn read_bytes(&self) -> u64 {
        self.read_bytes.load(Ordering::Relaxed)
    }

    fn read_ops(&self) -> u64 {
        self.read_ops.load(Ordering::Relaxed)
    }

    fn reset_counters(&self) {
        self.read_bytes.store(0, Ordering::Relaxed);
        self.read_ops.store(0, Ordering::Relaxed);
    }
}

impl IoBackend for VirtualChunkIo {
    fn submit_read(&self, file: FileRef, offset: u64, len: usize, _priority: u8) -> IoTask {
        let result = if let Some(virtual_file) = self.files.get(&file) {
            let within = (offset % virtual_file.stride) as usize;
            let end = within.saturating_add(len);
            self.read_bytes.fetch_add(len as u64, Ordering::Relaxed);
            self.read_ops.fetch_add(1, Ordering::Relaxed);
            if end <= virtual_file.encoded.len() {
                if let Some(bytes) = virtual_file.exact_ranges.get(&(within, len)) {
                    Ok(Arc::clone(bytes))
                } else {
                    Ok(Arc::from(
                        virtual_file.encoded[within..end]
                            .to_vec()
                            .into_boxed_slice(),
                    ))
                }
            } else {
                virtual_file.read(offset, len)
            }
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                "synthetic virtual IO unknown file",
            ))
        };
        Box::pin(async move { result })
    }

    fn prefers_inline_reads(&self) -> bool {
        true
    }
}

#[derive(Debug)]
struct VirtualFile {
    encoded: Arc<[u8]>,
    stride: u64,
    exact_ranges: HashMap<(usize, usize), Arc<[u8]>>,
}

impl VirtualFile {
    fn read(&self, offset: u64, len: usize) -> io::Result<Arc<[u8]>> {
        if self.stride == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "synthetic virtual IO stride is zero",
            ));
        }
        let stride = usize::try_from(self.stride).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "synthetic virtual IO stride does not fit usize",
            )
        })?;
        if stride == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "synthetic virtual IO stride is zero",
            ));
        }

        let mut out = vec![0u8; len];
        let mut copied = 0usize;
        let mut cursor = offset;
        while copied < len {
            let within = (cursor % self.stride) as usize;
            let remaining = len - copied;
            if within < self.encoded.len() {
                let take = remaining.min(self.encoded.len() - within);
                out[copied..copied + take].copy_from_slice(&self.encoded[within..within + take]);
                copied += take;
                cursor = cursor.saturating_add(take as u64);
            } else {
                let take = remaining.min(stride - within);
                copied += take;
                cursor = cursor.saturating_add(take as u64);
            }
        }
        Ok(Arc::from(out.into_boxed_slice()))
    }
}

fn precompute_virtual_exact_ranges(encoded: &Arc<[u8]>) -> HashMap<(usize, usize), Arc<[u8]>> {
    let mut ranges = HashMap::new();
    let Ok(Some(index)) = build_blosc_lz4_block_index(encoded) else {
        return ranges;
    };
    for block in index.blocks {
        let start = block.payload_relative_offset;
        let len = block.compressed_len;
        let end = start + len;
        if end <= encoded.len() {
            ranges.insert(
                (start, len),
                Arc::from(encoded[start..end].to_vec().into_boxed_slice()),
            );
        }
    }
    ranges
}

#[derive(Debug, Default)]
struct WorkerStats {
    batches: u64,
    items: u64,
    output_bytes: u64,
    checksum: u64,
}

impl WorkerStats {
    fn merge(&mut self, other: Self) {
        self.batches += other.batches;
        self.items += other.items;
        self.output_bytes += other.output_bytes;
        self.checksum = self.checksum.wrapping_add(other.checksum);
    }
}

#[derive(Debug, Default)]
struct ScheduledSyntheticStats {
    batches: usize,
    output_bytes: u64,
    checksum: u64,
}

#[derive(Debug)]
struct ReuseStats {
    cell_block_touches: u64,
    unique_blocks_per_batch_mean: f64,
    block_reuse_ratio: f64,
}

#[derive(Debug, Clone, Copy)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        value as f64 / ((1u64 << 53) as f64)
    }
}

// ---------------------------------------------------------------------------
// Real-IO native bench: drives the Blosc-LZ4 native loader against a real file
// on disk through a configurable IoPool (threaded / io_uring), so throughput is
// capped by the filesystem (GPFS) rather than an in-memory virtual backend.
// ---------------------------------------------------------------------------

use std::path::PathBuf;

use crate::iopool::{BaseIoConfig, IoCommand, IoPool};

/// Which IO backend `run_native_real_io_bench` should build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealIoBackend {
    Threaded,
    Uring,
}

impl RealIoBackend {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "threaded" => Ok(Self::Threaded),
            "uring" => Ok(Self::Uring),
            other => Err(format!("backend must be threaded or uring, got {other:?}")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Threaded => "threaded",
            Self::Uring => "uring",
        }
    }
}

/// Configuration for the real-IO native bench.
#[derive(Debug, Clone)]
pub struct NativeRealIoConfig {
    /// Which IoPool backend to build.
    pub backend: RealIoBackend,
    /// Worker count for the IoPool (threaded `num_workers` / uring `drivers`).
    pub io_workers: usize,
    /// io_uring entries (SQ/CQ depth). Ignored for threaded.
    pub uring_entries: u32,
    /// `max_in_flight` for the IoPool (in-flight unique IO cap).
    pub max_in_flight: usize,
    /// `queue_capacity` for the IoPool.
    pub queue_capacity: usize,
    /// `queue_shards` for the IoPool.
    pub queue_shards: usize,
    /// Directory under which the bench writes its encoded chunk file.
    pub data_dir: PathBuf,
    /// Number of concurrent native worker threads driving batches.
    pub workers: usize,
    /// Total batches to process (across all workers).
    pub batches: usize,
    /// Batches to run before the timed region (warms caches / index).
    pub warmup_batches: usize,
    /// Cells per batch.
    pub batch_size: usize,
    // Synthetic chunk geometry — identical meaning to NativeSyntheticConfig.
    pub chunks: usize,
    pub genes: usize,
    pub cells_per_chunk: usize,
    pub cell_bytes: usize,
    pub block_size: usize,
    pub typesize: usize,
    pub shuffle: bool,
    pub entropy_fraction: f32,
    pub order: NativeSyntheticOrder,
    pub continuation_p: f64,
    pub seed: u64,
    pub coalesce: NativeLoadCoalesceConfig,
}

impl Default for NativeRealIoConfig {
    fn default() -> Self {
        Self {
            backend: RealIoBackend::Threaded,
            io_workers: 48,
            uring_entries: 1024,
            max_in_flight: 1024,
            queue_capacity: 4096,
            queue_shards: 8,
            data_dir: PathBuf::from("."),
            workers: 4,
            batches: 2048,
            warmup_batches: 64,
            batch_size: 128,
            chunks: 64,
            genes: 4096,
            cells_per_chunk: 2048,
            cell_bytes: 12 * 1024,
            block_size: 192 * 1024,
            typesize: 2,
            shuffle: true,
            entropy_fraction: 0.33,
            order: NativeSyntheticOrder::Random,
            continuation_p: 0.0,
            seed: 0x5eed_5eed_1234_5678,
            coalesce: NativeLoadCoalesceConfig {
                max_window_us: 0,
                max_merged_len: 8 * 1024 * 1024,
                max_gap_bytes: 1024 * 1024,
                max_waste_ratio: 0.90,
                min_children: 2,
            },
        }
    }
}

/// Adapter that turns an `IoPool` into an `access::IoBackend`.
/// Mirrors `databank::adapter::IoPoolBackend`, which is crate-private. It also
/// counts read bytes/ops itself, so the bench report does not depend on the
/// iopool profile registry.
struct PoolIoBackend {
    pool: Arc<IoPool>,
    file_id: crate::iopool::FileId,
    file_ref: FileRef,
    read_bytes: Arc<AtomicU64>,
    read_ops: Arc<AtomicU64>,
}

impl PoolIoBackend {
    fn new(pool: Arc<IoPool>, file_id: crate::iopool::FileId) -> Self {
        Self {
            pool,
            file_ref: FileRef::new(file_id as u64),
            file_id,
            read_bytes: Arc::new(AtomicU64::new(0)),
            read_ops: Arc::new(AtomicU64::new(0)),
        }
    }

    fn read_bytes(&self) -> u64 {
        self.read_bytes.load(Ordering::Relaxed)
    }

    fn read_ops(&self) -> u64 {
        self.read_ops.load(Ordering::Relaxed)
    }

    fn reset_counters(&self) {
        self.read_bytes.store(0, Ordering::Relaxed);
        self.read_ops.store(0, Ordering::Relaxed);
    }
}

impl IoBackend for PoolIoBackend {
    fn submit_read(&self, file: FileRef, offset: u64, len: usize, priority: u8) -> IoTask {
        // The bench only ever registers one chunk file; a mismatched FileRef
        // means the caller built a ChunkKey with the wrong id. Fail loud rather
        // than silently routing to the wrong fd.
        if file != self.file_ref {
            return Box::pin(async move {
                Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "real-IO bench read targets an unexpected FileRef",
                ))
            });
        }
        let pool = Arc::clone(&self.pool);
        let file_id = self.file_id;
        let read_bytes = Arc::clone(&self.read_bytes);
        let read_ops = Arc::clone(&self.read_ops);
        Box::pin(async move {
            let future = pool
                .submit_async(IoCommand::read(file_id, offset, len, priority as usize))
                .await?
                .await?;
            read_bytes.fetch_add(len as u64, Ordering::Relaxed);
            read_ops.fetch_add(1, Ordering::Relaxed);
            future.into_read_bytes()
        })
    }
}

/// Run the real-IO native bench and return a JSON report.
pub fn run_native_real_io_bench(config: NativeRealIoConfig) -> Result<Value, String> {
    validate_real_io_config(&config)?;

    let decoded_bytes = config
        .cells_per_chunk
        .checked_mul(config.cell_bytes)
        .ok_or_else(|| "cells_per_chunk * cell_bytes overflow".to_string())?;
    let raw = synthetic_decoded_payload(decoded_bytes, config.entropy_fraction, config.seed);
    let encoded = blosc_lz4_encode(
        &raw,
        if config.shuffle {
            blosc_src::BLOSC_SHUFFLE as i32
        } else {
            0
        },
        config.typesize,
        config.block_size,
    )?;
    drop(raw);
    let encoded: Arc<[u8]> = Arc::from(encoded.into_boxed_slice());
    let encoded_len = encoded.len();

    // Write one chunk payload to disk, then lay out `chunks` virtual chunks by
    // repeating it at a fixed stride — exactly what the virtual backend does,
    // except every read now goes through the kernel filesystem via IoPool.
    let stride = (encoded_len + 4096) as u64;
    let file_path = config.data_dir.join("scdata-real-io-bench-chunk.bin");
    write_chunk_file(&file_path, encoded.as_ref(), config.chunks, stride)?;

    let io_config = build_real_io_config(&config)?;
    // Profile is disabled here: the bench measures throughput itself and the
    // `PoolIoBackend` counts read bytes/ops, so we avoid the iopool profile
    // registry entirely.
    let pool = Arc::new(
        IoPool::new_with_profile(io_config, crate::profile::ProfileRuntime::disabled())
            .map_err(|err| format!("create IoPool: {err}"))?,
    );
    let file_id = pool
        .register_readonly_file(&file_path)
        .map_err(|err| format!("register bench file: {err}"))?;
    let backend = Arc::new(PoolIoBackend::new(Arc::clone(&pool), file_id));
    let file_ref = backend.file_ref;
    let io: Arc<dyn IoBackend> = backend.clone();

    let cache = Arc::new(NativeBlockIndexCache::new());
    let codec = codec_from_json_str(r#"{"id":"blosc","cname":"lz4"}"#)
        .map_err(|err| format!("build Blosc codec: {err}"))?;

    // Build a NativeSyntheticConfig that matches our geometry so the existing
    // cell-stream / batch / slice helpers reuse the same logic. The scheduled
    // path is not exercised here, so its fields keep defaults.
    let synth = NativeSyntheticConfig {
        scheduled: false,
        batches: config.batches + config.warmup_batches,
        warmup_batches: 0,
        batch_size: config.batch_size,
        workers: config.workers,
        fill_workers: 0,
        native_workers: 0,
        io_workers: 0,
        chunks: config.chunks,
        genes: config.genes,
        source_genes: config.genes,
        cells_per_chunk: config.cells_per_chunk,
        cell_bytes: config.cell_bytes,
        block_size: config.block_size,
        typesize: config.typesize,
        shuffle: config.shuffle,
        entropy_fraction: config.entropy_fraction,
        order: config.order,
        continuation_p: config.continuation_p,
        seed: config.seed,
        projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy::SelectedOnly,
        scheduled_prefetch_step: 128,
        scheduled_access_prefetch_step: 1024,
        scheduled_decode_ahead_steps: 512,
        scheduled_ready_ahead_steps: 256,
        coalesce: config.coalesce.clone(),
    };
    let total_cells = config.chunks * config.cells_per_chunk;

    // Warmup (untimed): prime the block-index cache and the page cache so the
    // timed region measures steady-state IO throughput, not cold-start misses.
    if config.warmup_batches > 0 {
        run_real_io_workers(
            &synth,
            &io,
            &cache,
            &codec,
            config.coalesce.clone(),
            encoded_len,
            file_ref,
            0,
            config.warmup_batches,
            stride,
        )?;
    }

    backend.reset_counters();
    let start = Instant::now();
    let stats = run_real_io_workers(
        &synth,
        &io,
        &cache,
        &codec,
        config.coalesce.clone(),
        encoded_len,
        file_ref,
        config.warmup_batches,
        config.batches,
        stride,
    )?;
    let elapsed = start.elapsed();

    let read_ops = backend.read_ops();
    let read_bytes_metric = backend.read_bytes();

    // Cleanup: unregister before dropping so the pool drains the file cleanly.
    let _ = pool.try_unregister_file(file_id);
    drop(backend);
    drop(pool);

    let elapsed_s = elapsed.as_secs_f64();
    let batches_per_s = config.batches as f64 / elapsed_s;
    let cells_per_s = batches_per_s * config.batch_size as f64;
    let output_gib_per_s = stats.output_bytes as f64 / elapsed_s / 1024.0 / 1024.0 / 1024.0;
    let read_gib_per_s = read_bytes_metric as f64 / elapsed_s / 1024.0 / 1024.0 / 1024.0;
    let encoded_ratio = decoded_bytes as f64 / encoded_len as f64;

    Ok(json!({
        "config": {
            "backend": config.backend.as_str(),
            "io_workers": config.io_workers,
            "uring_entries": config.uring_entries,
            "max_in_flight": config.max_in_flight,
            "queue_capacity": config.queue_capacity,
            "queue_shards": config.queue_shards,
            "workers": config.workers,
            "batches": config.batches,
            "warmup_batches": config.warmup_batches,
            "batch_size": config.batch_size,
            "chunks": config.chunks,
            "cells_per_chunk": config.cells_per_chunk,
            "cell_bytes": config.cell_bytes,
            "decoded_chunk_bytes": decoded_bytes,
            "encoded_chunk_bytes": encoded_len,
            "block_size": config.block_size,
            "typesize": config.typesize,
            "shuffle": config.shuffle,
            "entropy_fraction": config.entropy_fraction,
            "order": config.order.as_str(),
            "continuation_p": config.continuation_p,
            "seed": config.seed,
            "stride_bytes": stride,
            "file": file_path.display().to_string(),
            "coalesce": {
                "max_window_us": config.coalesce.max_window_us,
                "max_merged_len": config.coalesce.max_merged_len,
                "max_gap_bytes": config.coalesce.max_gap_bytes,
                "max_waste_ratio": config.coalesce.max_waste_ratio,
                "min_children": config.coalesce.min_children,
            },
        },
        "shape": {
            "total_virtual_cells": total_cells,
            "encoded_compression_ratio": encoded_ratio,
            "encoded_bytes_per_cell": encoded_len as f64 / config.cells_per_chunk as f64,
        },
        "throughput": {
            "elapsed_s": elapsed_s,
            "batches_per_s": batches_per_s,
            "cells_per_s": cells_per_s,
            "output_gib_per_s": output_gib_per_s,
        },
        "io": {
            "read_ops": read_ops,
            "read_bytes": read_bytes_metric,
            "read_gib_per_s": read_gib_per_s,
            "read_ops_per_s": read_ops as f64 / elapsed_s,
            "ops_per_worker_per_s": read_ops as f64 / elapsed_s / config.io_workers.max(1) as f64,
        },
        "completion": {
            "processed_batches": stats.batches,
            "processed_items": stats.items,
            "output_bytes": stats.output_bytes,
            "checksum": stats.checksum,
        },
    }))
}

#[allow(clippy::too_many_arguments)]
fn run_real_io_workers(
    synth: &NativeSyntheticConfig,
    io: &Arc<dyn IoBackend>,
    cache: &Arc<NativeBlockIndexCache>,
    codec: &crate::codecs::SharedCodec,
    coalesce: NativeLoadCoalesceConfig,
    encoded_len: usize,
    file_ref: FileRef,
    batch_offset: usize,
    batch_count: usize,
    stride: u64,
) -> Result<WorkerStats, String> {
    let cells = build_cell_stream(synth);
    let mut handles = Vec::with_capacity(synth.workers);
    for worker_idx in 0..synth.workers {
        let cells = Arc::clone(&cells);
        let io = Arc::clone(io);
        let cache = Arc::clone(cache);
        let codec = codec.clone();
        let coalesce = coalesce.clone();
        let synth = synth.clone();
        handles.push(thread::spawn(move || -> Result<WorkerStats, String> {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|err| format!("build tokio runtime: {err}"))?;
            let mut stats = WorkerStats::default();
            for idx in 0..batch_count {
                let global_batch = batch_offset + idx;
                if global_batch % synth.workers != worker_idx {
                    continue;
                }
                let offset = global_batch * synth.batch_size;
                let batch_cells = &cells[offset..offset + synth.batch_size];
                let batch_stats = runtime.block_on(process_real_io_batch(
                    batch_cells,
                    &synth,
                    io.clone(),
                    Arc::clone(&cache),
                    codec.clone(),
                    coalesce.clone(),
                    encoded_len,
                    file_ref,
                    stride,
                ))?;
                stats.merge(batch_stats);
            }
            Ok(stats)
        }));
    }
    let mut stats = WorkerStats::default();
    for handle in handles {
        let worker = handle
            .join()
            .map_err(|_| "real-IO worker thread panicked".to_string())??;
        stats.merge(worker);
    }
    Ok(stats)
}

/// Like `process_synthetic_batch`, but reads the chunk from a real file at a
/// per-chunk stride instead of the virtual backend's single-payload mapping.
#[allow(clippy::too_many_arguments)]
async fn process_real_io_batch(
    batch_cells: &[usize],
    config: &NativeSyntheticConfig,
    io: Arc<dyn IoBackend>,
    cache: Arc<NativeBlockIndexCache>,
    codec: crate::codecs::SharedCodec,
    coalesce: NativeLoadCoalesceConfig,
    encoded_len: usize,
    file_ref: FileRef,
    stride: u64,
) -> Result<WorkerStats, String> {
    let mut by_chunk: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    for (row, &global_cell) in batch_cells.iter().enumerate() {
        let chunk = global_cell / config.cells_per_chunk;
        let cell = global_cell % config.cells_per_chunk;
        by_chunk.entry(chunk).or_default().push((row, cell));
    }

    let mut stats = WorkerStats {
        batches: 1,
        ..WorkerStats::default()
    };
    for (chunk, entries) in by_chunk {
        let mut triples = Vec::with_capacity(entries.len() * 3);
        for (local_row, (_batch_row, cell)) in entries.iter().enumerate() {
            let dst = local_row * config.cell_bytes;
            let src = cell * config.cell_bytes;
            triples.push(dst);
            triples.push(src);
            triples.push(src + config.cell_bytes);
        }
        let slice = SliceSpec::from_triples(triples).map_err(|err| err.to_string())?;
        let key = ChunkKey::new(file_ref, chunk as u64 * stride, encoded_len);
        let item = AccessItem::new(
            key,
            codec.clone(),
            Some(config.cells_per_chunk * config.cell_bytes),
        )
        .with_slice_spec(slice);
        let output = load_access_item_blosc_lz4_native(
            io.clone(),
            coalesce.clone(),
            &cache,
            None,
            None,
            &item,
            0,
        )
        .await
        .map_err(|err| err.to_string())?
        .ok_or_else(|| "real-IO native item fell back to generic path".to_string())?;
        stats.items += 1;
        stats.output_bytes += output.len() as u64;
        stats.checksum = stats.checksum.wrapping_add(checksum(&output));
    }
    Ok(stats)
}

fn write_chunk_file(
    path: &std::path::Path,
    encoded: &[u8],
    chunks: usize,
    stride: u64,
) -> Result<(), String> {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(|err| format!("create {}: {err}", path.display()))?;
    use std::io::{Seek, SeekFrom, Write};
    let mut writer = std::io::BufWriter::new(&file);
    for chunk in 0..chunks {
        writer
            .seek(SeekFrom::Start(chunk as u64 * stride))
            .map_err(|err| format!("seek {}: {err}", path.display()))?;
        writer
            .write_all(encoded)
            .map_err(|err| format!("write {}: {err}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|err| format!("flush {}: {err}", path.display()))?;
    drop(writer);
    file.sync_data()
        .map_err(|err| format!("sync_data {}: {err}", path.display()))?;
    Ok(())
}

fn build_real_io_config(config: &NativeRealIoConfig) -> Result<crate::iopool::IoConfig, String> {
    let base = BaseIoConfig {
        max_in_flight: config.max_in_flight,
        queue_capacity: config.queue_capacity,
        priority_levels: 3,
        queue_shards: config.queue_shards,
        assume_non_overlapping_reads: true,
    };
    base.validate().map_err(|err| err.to_string())?;
    match config.backend {
        RealIoBackend::Threaded => Ok(crate::iopool::IoConfig::Threaded(
            crate::iopool::ThreadedConfig {
                base,
                num_workers: config.io_workers,
                cpus: None,
            },
        )),
        RealIoBackend::Uring => {
            // `UringConfig::validate` is `pub(super)`, so replicate its checks
            // here rather than calling it from this module.
            if config.uring_entries < 2 {
                return Err("uring_entries must be greater than 1".to_string());
            }
            if config.io_workers == 0 {
                return Err("uring drivers must be greater than 0".to_string());
            }
            let uring = crate::iopool::UringConfig {
                base,
                entries: config.uring_entries,
                drivers: config.io_workers,
                iowq_bounded_workers: 0,
                iowq_unbounded_workers: 0,
                registered_files: 0,
            };
            Ok(crate::iopool::IoConfig::Uring(uring))
        }
    }
}

fn validate_real_io_config(config: &NativeRealIoConfig) -> Result<(), String> {
    if config.io_workers == 0 {
        return Err("io_workers must be greater than 0".to_string());
    }
    if config.workers == 0 {
        return Err("workers must be greater than 0".to_string());
    }
    if config.batches == 0 {
        return Err("batches must be greater than 0".to_string());
    }
    if config.batch_size == 0 {
        return Err("batch_size must be greater than 0".to_string());
    }
    if config.chunks == 0 {
        return Err("chunks must be greater than 0".to_string());
    }
    if config.cells_per_chunk == 0 {
        return Err("cells_per_chunk must be greater than 0".to_string());
    }
    if config.cell_bytes == 0 {
        return Err("cell_bytes must be greater than 0".to_string());
    }
    if config.block_size == 0 {
        return Err("block_size must be greater than 0".to_string());
    }
    if config.typesize == 0 {
        return Err("typesize must be greater than 0".to_string());
    }
    if !(0.0..=1.0).contains(&config.entropy_fraction) {
        return Err("entropy_fraction must be in [0, 1]".to_string());
    }
    if !(0.0..=1.0).contains(&config.continuation_p) {
        return Err("continuation_p must be in [0, 1]".to_string());
    }
    if config.uring_entries < 2 && config.backend == RealIoBackend::Uring {
        return Err("uring_entries must be >= 2".to_string());
    }
    config.coalesce.validate()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_stream_has_more_block_reuse_than_random() {
        let base = NativeSyntheticConfig {
            batches: 32,
            batch_size: 128,
            chunks: 1,
            cells_per_chunk: 4096,
            cell_bytes: 12 * 1024,
            block_size: 192 * 1024,
            ..NativeSyntheticConfig::default()
        };
        let random_cells = build_cell_stream(&NativeSyntheticConfig {
            order: NativeSyntheticOrder::Random,
            ..base.clone()
        });
        let sequential_cells = build_cell_stream(&NativeSyntheticConfig {
            order: NativeSyntheticOrder::Sequential,
            ..base.clone()
        });

        let random = estimate_block_reuse(&base, &random_cells);
        let sequential = estimate_block_reuse(&base, &sequential_cells);

        assert!(sequential.block_reuse_ratio > random.block_reuse_ratio);
        assert!(sequential.unique_blocks_per_batch_mean < random.unique_blocks_per_batch_mean);
    }
}
