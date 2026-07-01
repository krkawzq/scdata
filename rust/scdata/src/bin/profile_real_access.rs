use _scdata::access::{AccessConfig, AccessCpuConfig, ScheduledAccessConfig};
use _scdata::codecs::DecodePoolConfig;
use _scdata::databank::{
    ArrayCodecSpec, ArrayGridSpec, ArrayOrder, ArraySpec, ChunkSourceSpec, ChunkSpec, DType,
    DataBank, DataBankConfig, DataValue, Dense1DSpec, Dense2DSpec, EdgeChunkLayout, FillConfig,
    MissingGenePolicy, MultiBatchCells, ProjectedSparseDataGroupStrategy, ScheduledPrefetchConfig,
    SparseCsrSpec,
};
use _scdata::iopool::{IoConfig, ThreadedConfig};
use _scdata::profile::{ProfileMetricKind, ProfileSnapshot};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::CString;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

type AppResult<T> = Result<T, Box<dyn std::error::Error>>;

const PY_HELPER: &str = r#"
import json
import random
import sys
from pathlib import Path

import numpy as np


def add_repo(path):
    if path:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _pick_rowcount(path):
    from scdata import DType, launch_all

    datasets = launch_all(Path(path))
    keys = ["X"]
    if "raw/X" in datasets:
        keys.append("raw/X")
    try:
        keys.extend(f"layers/{name}" for name in sorted(datasets.layers))
    except Exception:
        pass
    for key in keys:
        ds = datasets[key]
        if ds.dtype in {DType.U16, DType.U32}:
            return key, ds
    return None, None


def build_catalog(root, limit_datasets):
    root = Path(root)
    paths = sorted(root.rglob("*.zarr.zip"))
    if limit_datasets:
        paths = paths[: int(limit_datasets)]
    if not paths:
        raise RuntimeError(f"no .zarr.zip found under {root}")

    catalog = []
    skipped = []
    for path in paths:
        try:
            key, ds = _pick_rowcount(path)
            if ds is None:
                skipped.append({"path": str(path), "reason": "no u16/u32 matrix"})
                continue
            catalog.append(
                {
                    "path": str(path),
                    "matrix": key,
                    "dataset": ds,
                    "dtype": ds.dtype.value,
                    "kind": ds.kind,
                    "n_obs": int(ds.num_cells),
                    "n_vars": int(ds.num_genes),
                    "num_chunks": int(ds.num_chunks),
                }
            )
        except Exception as exc:
            skipped.append({"path": str(path), "reason": repr(exc)})

    if not catalog:
        raise RuntimeError(f"no rowcount datasets found under {root}")

    summary = {
        "root": str(root),
        "paths_seen": len(paths),
        "datasets": len(catalog),
        "skipped": len(skipped),
        "skipped_examples": skipped[:10],
        "counts": [entry["n_obs"] for entry in catalog],
        "total_cells": int(sum(entry["n_obs"] for entry in catalog)),
        "dtypes": sorted({entry["dtype"] for entry in catalog}),
        "kinds": sorted({entry["kind"] for entry in catalog}),
        "n_vars_min": int(min(entry["n_vars"] for entry in catalog)),
        "n_vars_max": int(max(entry["n_vars"] for entry in catalog)),
        "chunks_total": int(sum(entry["num_chunks"] for entry in catalog)),
        "entries": [
            {
                "path": entry["path"],
                "matrix": entry["matrix"],
                "dtype": entry["dtype"],
                "kind": entry["kind"],
                "n_obs": entry["n_obs"],
                "n_vars": entry["n_vars"],
                "num_chunks": entry["num_chunks"],
            }
            for entry in catalog[:20]
        ],
    }
    return catalog, json.dumps(summary, separators=(",", ":"))


def make_bank(memory_gib, threads):
    from scdata import DataBankConfig, ScDataBank

    threads = int(threads)
    if threads % 4 != 0:
        raise ValueError("--threads must be divisible by 4")
    workers = threads // 4
    cfg = DataBankConfig.make(
        backend="threaded",
        io__threaded__num_workers=workers,
        decode__num_workers=workers,
        access__cpu__num_workers=workers,
        fill__num_workers=workers,
        access__cache_capacity_bytes=int(memory_gib) * 1024**3 * 3 // 4,
        access__memory_budget_bytes=int(memory_gib) * 1024**3,
    )
    return ScDataBank(cfg)


def make_bank_split(memory_gib, io_workers, decode_workers, access_workers, fill_workers):
    from scdata import DataBankConfig, ScDataBank

    cfg = DataBankConfig.make(
        backend="threaded",
        io__threaded__num_workers=int(io_workers),
        decode__num_workers=int(decode_workers),
        access__cpu__num_workers=int(access_workers),
        fill__num_workers=int(fill_workers),
        access__cache_capacity_bytes=int(memory_gib) * 1024**3 * 3 // 4,
        access__memory_budget_bytes=int(memory_gib) * 1024**3,
    )
    return ScDataBank(cfg)


def register_catalog(bank, catalog):
    return [bank.register(entry["dataset"]) for entry in catalog]


def close_bank(bank):
    bank.close()


def resolve_genes(bank, ids, catalog, gene_mode, genes):
    gene_mode = str(gene_mode)
    if gene_mode == "native":
        n_vars = {entry["n_vars"] for entry in catalog}
        if len(n_vars) != 1:
            raise ValueError("native gene mode requires identical n_vars")
        return None
    names = bank.dataset_genes(ids[0])
    genes = int(genes)
    if genes:
        names = names[:genes]
    return names


def resolve_dtype(dtype):
    text = str(dtype).strip().lower()
    if text in ("stored", "native", "none", ""):
        return None
    return dtype


def make_prefetch_config(prefetch_step, access_prefetch_step, decode_ahead_steps, ready_ahead_steps):
    from scdata import ScheduledAccessConfig, ScheduledPrefetchConfig

    return ScheduledPrefetchConfig(
        prefetch_step=int(prefetch_step),
        access=ScheduledAccessConfig(
            prefetch_step=int(access_prefetch_step),
            decode_ahead_steps=int(decode_ahead_steps),
            ready_ahead_steps=int(ready_ahead_steps),
        ),
    )


def make_plan(dataset_index, cell_index, batch_size):
    from scdata import CellIndexPlan

    ds = np.asarray(dataset_index)
    ci = np.asarray(cell_index)
    if ds.size:
        ds_max = int(ds.max())
        if ds_max <= np.iinfo(np.uint16).max:
            ds = ds.astype(np.uint16, copy=False)
        elif ds_max <= np.iinfo(np.uint32).max:
            ds = ds.astype(np.uint32, copy=False)
        else:
            ds = ds.astype(np.uint64, copy=False)
    else:
        ds = ds.astype(np.uint16, copy=False)
    if ci.size:
        ci_max = int(ci.max())
        if ci_max <= np.iinfo(np.uint32).max:
            ci = ci.astype(np.uint32, copy=False)
        else:
            ci = ci.astype(np.uint64, copy=False)
    else:
        ci = ci.astype(np.uint32, copy=False)
    return CellIndexPlan(ds, ci, int(batch_size))


def prefetch_indexed(bank, ids, plan, genes, dtype, config):
    missing = "zero" if genes is not None else None
    return bank.prefetch_indexed(ids, plan, genes=genes, missing=missing, dtype=dtype, config=config)


def batch_metrics(batch):
    data = batch.data
    checksum = 0
    if data.size:
        checksum = int(data.flat[0])
        if data.size > 1:
            checksum ^= int(data.flat[-1]) << 1
    return int(len(batch.cells)), int(data.nbytes), int(checksum & ((1 << 64) - 1))


def reset_profile(bank):
    bank.reset_profile()


def profile_json(bank):
    return json.dumps(bank.profile_snapshot_and_reset(), separators=(",", ":"), default=str)


def _array_range_iter(ds, array_name, meta):
    lengths = np.asarray(meta.chunk_lengths, dtype=np.uint64)
    offsets = np.asarray(meta.chunk_offsets, dtype=np.uint64)
    if lengths.size == 0:
        return
    if meta.store_kind == "file":
        file_path = meta.payload_file_path or str(Path(ds.store_root) / meta.payload_path)
        for i in range(int(lengths.size)):
            yield {
                "path": file_path,
                "offset": int(offsets[i]),
                "length": int(lengths[i]),
                "array": array_name,
            }
    else:
        paths = tuple(meta.chunk_paths)
        files = tuple(meta.chunk_file_paths)
        root = Path(ds.store_root)
        for i in range(int(lengths.size)):
            if files:
                file_path = files[i]
            else:
                file_path = str(root / paths[i])
            offset = int(offsets[i]) if offsets.size == lengths.size else 0
            yield {
                "path": file_path,
                "offset": offset,
                "length": int(lengths[i]),
                "array": array_name,
            }


def _dataset_ranges(entry):
    ds = entry["dataset"]
    if ds.kind == "dense":
        yield from _array_range_iter(ds, "data", ds.data)
    else:
        yield from _array_range_iter(ds, "indices", ds.indices)
        yield from _array_range_iter(ds, "data", ds.data)


def chunk_ranges_json(catalog, samples, seed):
    samples = int(samples)
    rng = random.Random(int(seed))
    selected = []
    total = 0
    by_array = {}
    by_file = {}
    total_bytes = 0
    for dataset_idx, entry in enumerate(catalog):
        for item in _dataset_ranges(entry):
            length = int(item["length"])
            if length <= 0:
                continue
            total += 1
            total_bytes += length
            by_array[item["array"]] = by_array.get(item["array"], 0) + 1
            by_file[item["path"]] = by_file.get(item["path"], 0) + 1
            item = dict(item)
            item["dataset_idx"] = dataset_idx
            if samples <= 0:
                selected.append(item)
            elif len(selected) < samples:
                selected.append(item)
            else:
                j = rng.randrange(total)
                if j < samples:
                    selected[j] = item
    payload = {
        "total_ranges": total,
        "sampled_ranges": len(selected),
        "total_compressed_bytes": total_bytes,
        "by_array": by_array,
        "files": len(by_file),
        "top_files": sorted(by_file.items(), key=lambda kv: kv[1], reverse=True)[:10],
        "ranges": selected,
    }
    return json.dumps(payload, separators=(",", ":"))
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Both,
    Scheduled,
    Io,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Engine {
    RustCore,
    PyWrapper,
}

impl Engine {
    fn parse(value: &str) -> AppResult<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "rust" | "rust-core" | "core" => Ok(Self::RustCore),
            "py" | "python" | "py-wrapper" | "wrapper" => Ok(Self::PyWrapper),
            other => Err(format!("unknown engine `{other}`").into()),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::RustCore => "rust-core",
            Self::PyWrapper => "py-wrapper",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Order {
    Random,
    Grouped,
    Sequential,
}

impl Order {
    fn parse(value: &str) -> AppResult<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "random" | "shuffle" | "shuffled" => Ok(Self::Random),
            "grouped" | "sorted" | "dataset" | "by-dataset" => Ok(Self::Grouped),
            "sequential" | "seq" => Ok(Self::Sequential),
            other => Err(format!("unknown order `{other}`").into()),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Grouped => "grouped",
            Self::Sequential => "sequential",
        }
    }
}

#[derive(Debug, Clone)]
struct Args {
    root: PathBuf,
    repo_root: PathBuf,
    mode: Mode,
    engine: Engine,
    limit_datasets: usize,
    max_cells: Option<usize>,
    batch_sizes: Vec<usize>,
    orders: Vec<Order>,
    prefetch_steps: Vec<usize>,
    access_prefetch_steps: Vec<usize>,
    decode_ahead_steps: Vec<usize>,
    ready_ahead_steps: Vec<usize>,
    threads: usize,
    io_workers: usize,
    decode_workers: usize,
    access_workers: usize,
    fill_workers: usize,
    memory_gib: usize,
    gene_mode: String,
    projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    genes: usize,
    dtype: String,
    seed: u64,
    warmup_batches: usize,
    measure_batches: usize,
    out: Option<PathBuf>,
    io_samples: usize,
    io_threads: usize,
    io_max_bytes_per_read: usize,
    reuse_bank: bool,
}

#[derive(Debug, Clone)]
struct CaseConfig {
    batch_size: usize,
    order: Order,
    prefetch_step: usize,
    access_prefetch_step: usize,
    decode_ahead_steps: usize,
    ready_ahead_steps: usize,
}

#[derive(Debug, Clone)]
struct RangeSpec {
    path: String,
    offset: u64,
    length: usize,
    dataset_idx: usize,
    array: String,
}

enum RustDatasetSpec {
    Dense1D(Dense1DSpec),
    Dense2D(Dense2DSpec),
    Sparse(SparseCsrSpec),
}

impl RustDatasetSpec {
    fn gene_names(&self) -> &[String] {
        match self {
            Self::Dense1D(spec) => &spec.gene_names,
            Self::Dense2D(spec) => &spec.gene_names,
            Self::Sparse(spec) => &spec.gene_names,
        }
    }

    fn num_genes(&self) -> usize {
        match self {
            Self::Dense1D(spec) => spec.gene_names.len(),
            Self::Dense2D(spec) => spec.gene_names.len(),
            Self::Sparse(spec) => spec.num_genes,
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Self::Dense1D(spec) => spec.data.dtype,
            Self::Dense2D(spec) => spec.data.dtype,
            Self::Sparse(spec) => spec.data.dtype,
        }
    }
}

impl Clone for RustDatasetSpec {
    fn clone(&self) -> Self {
        match self {
            Self::Dense1D(spec) => Self::Dense1D(Dense1DSpec {
                gene_names: spec.gene_names.clone(),
                data: spec.data.clone(),
            }),
            Self::Dense2D(spec) => Self::Dense2D(Dense2DSpec {
                gene_names: spec.gene_names.clone(),
                data: spec.data.clone(),
            }),
            Self::Sparse(spec) => Self::Sparse(SparseCsrSpec {
                gene_names: spec.gene_names.clone(),
                indptr: spec.indptr.clone(),
                indices: spec.indices.clone(),
                data: spec.data.clone(),
                index_dtype: spec.index_dtype,
                num_cells: spec.num_cells,
                num_genes: spec.num_genes,
            }),
        }
    }
}

#[derive(Debug, Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut z = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        self.state = z;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn gen_range(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            return 0;
        }
        (self.next_u64() % upper as u64) as usize
    }
}

fn main() -> AppResult<()> {
    let args = parse_args()?;
    env::set_var("SCDATA_PROFILE", "1");
    env::set_var("SCDATA_PROFILE_LABEL", "profile-real-access");
    env::set_var("SCDATA_PROFILE_COMPONENTS", "all");
    env::set_var("SCDATA_PROFILE_SCOPES", "all");

    let mut writer = open_writer(&args)?;
    let mut ranges = Vec::new();

    Python::with_gil(|py| -> AppResult<()> {
        let helper = helper_module(py)?;
        helper.call_method1("add_repo", (args.repo_root.to_string_lossy().to_string(),))?;

        let scan_started = Instant::now();
        let built = helper.call_method1(
            "build_catalog",
            (args.root.to_string_lossy().to_string(), args.limit_datasets),
        )?;
        let (catalog_obj, catalog_json): (Py<PyAny>, String) = built.extract()?;
        let catalog = catalog_obj.bind(py);
        let scan_seconds = scan_started.elapsed().as_secs_f64();
        let catalog_value: Value = serde_json::from_str(&catalog_json)?;
        let counts = counts_from_summary(&catalog_value)?;

        emit(
            &mut writer,
            json!({
                "kind": "catalog",
                "scan_seconds": scan_seconds,
                "args": args_json(&args),
                "catalog": catalog_value,
            }),
        )?;

        if matches!(args.mode, Mode::Both | Mode::Scheduled) && args.engine == Engine::PyWrapper {
            let total = counts.iter().copied().sum::<usize>();
            let sample_n = args.max_cells.unwrap_or(total).min(total);
            let base_sample = sampled_global_indices(total, sample_n, args.seed);
            let offsets = offsets(&counts);
            let cases = case_configs(&args);

            if args.reuse_bank {
                let bank = helper.call_method1(
                    "make_bank_split",
                    (
                        args.memory_gib,
                        databank_io_workers(&args),
                        databank_decode_workers(&args),
                        databank_access_workers(&args),
                        databank_fill_workers(&args),
                    ),
                )?;
                let register_started = Instant::now();
                let ids = helper.call_method1("register_catalog", (&bank, &catalog))?;
                let register_seconds = register_started.elapsed().as_secs_f64();
                for (case_idx, case) in cases.iter().enumerate() {
                    run_case(
                        py,
                        &helper,
                        &mut writer,
                        &args,
                        &bank,
                        &ids,
                        &catalog,
                        &counts,
                        &offsets,
                        &base_sample,
                        case_idx,
                        case,
                        register_seconds,
                    )?;
                }
                let _ = helper.call_method1("close_bank", (&bank,));
            } else {
                for (case_idx, case) in cases.iter().enumerate() {
                    let bank = helper.call_method1(
                        "make_bank_split",
                        (
                            args.memory_gib,
                            databank_io_workers(&args),
                            databank_decode_workers(&args),
                            databank_access_workers(&args),
                            databank_fill_workers(&args),
                        ),
                    )?;
                    let register_started = Instant::now();
                    let ids = helper.call_method1("register_catalog", (&bank, &catalog))?;
                    let register_seconds = register_started.elapsed().as_secs_f64();
                    let result = run_case(
                        py,
                        &helper,
                        &mut writer,
                        &args,
                        &bank,
                        &ids,
                        &catalog,
                        &counts,
                        &offsets,
                        &base_sample,
                        case_idx,
                        case,
                        register_seconds,
                    );
                    let _ = helper.call_method1("close_bank", (&bank,));
                    result?;
                }
            }
        }

        if matches!(args.mode, Mode::Both | Mode::Scheduled) && args.engine == Engine::RustCore {
            let total = counts.iter().copied().sum::<usize>();
            let sample_n = args.max_cells.unwrap_or(total).min(total);
            let base_sample = sampled_global_indices(total, sample_n, args.seed);
            let offsets = offsets(&counts);
            let cases = case_configs(&args);
            let specs = build_rust_dataset_specs(&catalog)?;
            for (case_idx, case) in cases.iter().enumerate() {
                run_case_rust_core(
                    &mut writer,
                    &args,
                    &specs,
                    &counts,
                    &offsets,
                    &base_sample,
                    case_idx,
                    case,
                )?;
            }
        }

        if matches!(args.mode, Mode::Both | Mode::Io) && args.io_samples > 0 {
            let ranges_json: String = helper
                .call_method1("chunk_ranges_json", (&catalog, args.io_samples, args.seed))?
                .extract()?;
            let ranges_value: Value = serde_json::from_str(&ranges_json)?;
            emit(
                &mut writer,
                json!({
                    "kind": "io_sample_catalog",
                    "io_samples": args.io_samples,
                    "summary": ranges_value.clone(),
                }),
            )?;
            ranges = parse_ranges(&ranges_value)?;
        }

        Ok(())
    })?;

    if matches!(args.mode, Mode::Both | Mode::Io) && !ranges.is_empty() {
        let result = bench_io_ranges(
            &ranges,
            args.io_threads,
            args.io_max_bytes_per_read,
            args.seed,
        )?;
        emit(&mut writer, result)?;
    }

    Ok(())
}

fn helper_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let code = CString::new(PY_HELPER).expect("helper has no interior nul");
    PyModule::from_code(
        py,
        code.as_c_str(),
        pyo3::ffi::c_str!("profile_real_access.py"),
        pyo3::ffi::c_str!("profile_real_access"),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    _py: Python<'_>,
    helper: &Bound<'_, PyModule>,
    writer: &mut Option<File>,
    args: &Args,
    bank: &Bound<'_, PyAny>,
    ids: &Bound<'_, PyAny>,
    catalog: &Bound<'_, PyAny>,
    counts: &[usize],
    offsets: &[usize],
    base_sample: &[usize],
    case_idx: usize,
    case: &CaseConfig,
    register_seconds: f64,
) -> AppResult<()> {
    let order = ordered_indices(base_sample, counts, case.order, args.seed ^ case_idx as u64);
    let (dataset_index, cell_index) = plan_arrays(&order, offsets);
    let plan_stats = plan_stats(&dataset_index, case.batch_size);
    let plan_started = Instant::now();
    let plan = helper.call_method1(
        "make_plan",
        (dataset_index.clone(), cell_index.clone(), case.batch_size),
    )?;
    let config = helper.call_method1(
        "make_prefetch_config",
        (
            case.prefetch_step,
            case.access_prefetch_step,
            case.decode_ahead_steps,
            case.ready_ahead_steps,
        ),
    )?;
    let genes = helper.call_method1(
        "resolve_genes",
        (bank, ids, catalog, args.gene_mode.as_str(), args.genes),
    )?;
    let dtype = helper.call_method1("resolve_dtype", (args.dtype.as_str(),))?;
    let stream = helper.call_method1(
        "prefetch_indexed",
        (bank, ids, &plan, &genes, &dtype, &config),
    )?;
    let stream_setup_seconds = plan_started.elapsed().as_secs_f64();

    helper.call_method1("reset_profile", (bank,))?;
    let mut started: Option<Instant> = None;
    let mut measured_batches = 0usize;
    let mut measured_cells = 0usize;
    let mut measured_bytes = 0usize;
    let mut checksum = 0u64;
    let mut seen_batches = 0usize;

    for item in stream.try_iter()? {
        let batch = item?;
        let (cells, bytes, batch_checksum): (usize, usize, u64) =
            helper.call_method1("batch_metrics", (batch,))?.extract()?;
        seen_batches += 1;
        if seen_batches <= args.warmup_batches {
            if seen_batches == args.warmup_batches {
                helper.call_method1("reset_profile", (bank,))?;
            }
            continue;
        }
        if started.is_none() {
            started = Some(Instant::now());
        }
        measured_batches += 1;
        measured_cells += cells;
        measured_bytes += bytes;
        checksum = checksum.wrapping_add(batch_checksum);
        if args.measure_batches > 0 && measured_batches >= args.measure_batches {
            break;
        }
    }

    let seconds = started
        .map(|started| started.elapsed().as_secs_f64())
        .unwrap_or(0.0);
    let profile_text: String = helper.call_method1("profile_json", (bank,))?.extract()?;
    let profile: Value = serde_json::from_str(&profile_text).unwrap_or_else(|_| {
        json!({
            "parse_error": true,
            "raw": profile_text,
        })
    });

    emit(
        writer,
        json!({
            "kind": "scheduled_case",
            "case_idx": case_idx,
            "order": case.order.as_str(),
            "batch_size": case.batch_size,
            "prefetch_step": case.prefetch_step,
            "access_prefetch_step": case.access_prefetch_step,
            "decode_ahead_steps": case.decode_ahead_steps,
            "ready_ahead_steps": case.ready_ahead_steps,
            "threads": args.threads,
            "io_workers": databank_io_workers(args),
            "decode_workers": databank_decode_workers(args),
            "access_workers": databank_access_workers(args),
            "fill_workers": databank_fill_workers(args),
            "memory_gib": args.memory_gib,
            "gene_mode": args.gene_mode,
            "genes": args.genes,
            "dtype": args.dtype,
            "datasets": counts.len(),
            "sampled_cells": order.len(),
            "warmup_batches": args.warmup_batches,
            "seen_batches": seen_batches,
            "measured_batches": measured_batches,
            "measured_cells": measured_cells,
            "measured_output_bytes": measured_bytes,
            "seconds": seconds,
            "batches_per_s": rate(measured_batches, seconds),
            "cells_per_s": rate(measured_cells, seconds),
            "output_gb_per_s": measured_bytes as f64 / seconds.max(f64::MIN_POSITIVE) / 1e9,
            "checksum": checksum,
            "register_seconds": register_seconds,
            "stream_setup_seconds": stream_setup_seconds,
            "plan_stats": plan_stats,
            "profile": profile,
        }),
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_case_rust_core(
    writer: &mut Option<File>,
    args: &Args,
    specs: &[RustDatasetSpec],
    counts: &[usize],
    offsets: &[usize],
    base_sample: &[usize],
    case_idx: usize,
    case: &CaseConfig,
) -> AppResult<()> {
    let order = ordered_indices(base_sample, counts, case.order, args.seed ^ case_idx as u64);
    let (dataset_index, cell_index) = plan_arrays(&order, offsets);
    let plan_stats = plan_stats(&dataset_index, case.batch_size);
    let batches = build_multi_batches(&dataset_index, &cell_index, case.batch_size);
    let output_dtype = resolve_output_dtype(specs, &args.dtype)?;
    let gene_names = resolve_rust_genes(specs, &args.gene_mode, args.genes)?;
    let config = ScheduledPrefetchConfig {
        prefetch_step: case.prefetch_step,
        access: ScheduledAccessConfig {
            prefetch_step: case.access_prefetch_step,
            decode_ahead_steps: case.decode_ahead_steps,
            ready_ahead_steps: case.ready_ahead_steps,
        },
        projected_sparse_data_strategy: args.projected_sparse_data_strategy,
    };

    let mut bank = DataBank::new(rust_databank_config(args))?;
    let register_started = Instant::now();
    let mut ids = Vec::with_capacity(specs.len());
    for spec in specs.iter().cloned() {
        let id = match spec {
            RustDatasetSpec::Dense1D(spec) => bank.register_dense_1d(spec)?,
            RustDatasetSpec::Dense2D(spec) => bank.register_dense_2d(spec)?,
            RustDatasetSpec::Sparse(spec) => bank.register_sparse_csr(spec)?,
        };
        ids.push(id);
    }
    let register_seconds = register_started.elapsed().as_secs_f64();

    let stream_started = Instant::now();
    let measured = match output_dtype {
        DType::U16 => consume_rust_prefetch::<u16>(
            &bank,
            &ids,
            batches,
            gene_names.as_deref(),
            config,
            args.warmup_batches,
            args.measure_batches,
        )?,
        DType::U32 => consume_rust_prefetch::<u32>(
            &bank,
            &ids,
            batches,
            gene_names.as_deref(),
            config,
            args.warmup_batches,
            args.measure_batches,
        )?,
        other => {
            return Err(format!(
            "rust-core engine currently supports u16/u32 output for this benchmark, got {other:?}"
        )
            .into())
        }
    };
    let stream_setup_seconds = stream_started.elapsed().as_secs_f64() - measured.seconds;

    emit(
        writer,
        json!({
            "kind": "scheduled_case",
            "engine": "rust-core",
            "case_idx": case_idx,
            "order": case.order.as_str(),
            "batch_size": case.batch_size,
            "prefetch_step": case.prefetch_step,
            "access_prefetch_step": case.access_prefetch_step,
            "decode_ahead_steps": case.decode_ahead_steps,
            "ready_ahead_steps": case.ready_ahead_steps,
            "threads": args.threads,
            "io_workers": databank_io_workers(args),
            "decode_workers": databank_decode_workers(args),
            "access_workers": databank_access_workers(args),
            "fill_workers": databank_fill_workers(args),
            "memory_gib": args.memory_gib,
            "gene_mode": args.gene_mode,
            "projected_sparse_data_strategy": args.projected_sparse_data_strategy.as_str(),
            "genes": args.genes,
            "dtype": args.dtype,
            "output_dtype": dtype_name(output_dtype),
            "datasets": counts.len(),
            "sampled_cells": order.len(),
            "warmup_batches": args.warmup_batches,
            "seen_batches": measured.seen_batches,
            "measured_batches": measured.measured_batches,
            "measured_cells": measured.measured_cells,
            "measured_output_bytes": measured.measured_bytes,
            "seconds": measured.seconds,
            "batches_per_s": rate(measured.measured_batches, measured.seconds),
            "cells_per_s": rate(measured.measured_cells, measured.seconds),
            "output_gb_per_s": measured.measured_bytes as f64 / measured.seconds.max(f64::MIN_POSITIVE) / 1e9,
            "checksum": measured.checksum,
            "register_seconds": register_seconds,
            "stream_setup_seconds": stream_setup_seconds.max(0.0),
            "plan_stats": plan_stats,
            "profiles": {
                "databank": measured.databank_profile,
                "access": measured.access_profile,
                "io": measured.io_profile,
                "decode": measured.decode_profile,
            },
        }),
    )?;

    Ok(())
}

#[derive(Debug, Clone)]
struct RustMeasure {
    seen_batches: usize,
    measured_batches: usize,
    measured_cells: usize,
    measured_bytes: usize,
    checksum: u64,
    seconds: f64,
    databank_profile: Value,
    access_profile: Value,
    io_profile: Value,
    decode_profile: Value,
}

fn consume_rust_prefetch<T>(
    bank: &DataBank,
    ids: &[_scdata::databank::DatasetId],
    batches: Vec<MultiBatchCells>,
    gene_names: Option<&[String]>,
    config: ScheduledPrefetchConfig,
    warmup_batches: usize,
    measure_batches: usize,
) -> AppResult<RustMeasure>
where
    T: DataValue + ChecksumValue,
{
    let mut stream = if let Some(gene_names) = gene_names {
        bank.prefetch_cells_scheduled_multi_by_gene_names::<T, _, _>(
            ids,
            batches,
            gene_names,
            MissingGenePolicy::Zero,
            config,
        )?
    } else {
        bank.prefetch_cells_scheduled_multi::<T, _>(ids, batches, config)?
    };

    let mut started: Option<Instant> = None;
    let mut seen_batches = 0usize;
    let mut measured_batches = 0usize;
    let mut measured_cells = 0usize;
    let mut measured_bytes = 0usize;
    let mut checksum = 0u64;

    while let Some(batch) = stream.next() {
        let batch = batch?;
        seen_batches += 1;
        if seen_batches <= warmup_batches {
            if seen_batches == warmup_batches {
                bank.reset_runtime_profiles();
            }
            continue;
        }
        if started.is_none() {
            started = Some(Instant::now());
        }
        measured_batches += 1;
        measured_cells += batch.cells.len();
        measured_bytes += batch.buffer.len() * std::mem::size_of::<T>();
        if let Some(first) = batch.buffer.first() {
            checksum = checksum.wrapping_add(first.checksum_value());
        }
        if let Some(last) = batch.buffer.last() {
            checksum ^= last.checksum_value() << 1;
        }
        if measure_batches > 0 && measured_batches >= measure_batches {
            break;
        }
    }

    let seconds = started
        .map(|started| started.elapsed().as_secs_f64())
        .unwrap_or(0.0);
    drop(stream);
    let databank_profile = profile_snapshot_json(bank.profile_snapshot_and_reset());
    let access_profile = profile_snapshot_json(bank.access_profile_snapshot_and_reset());
    let io_profile = profile_snapshot_json(bank.io_profile_snapshot_and_reset());
    let decode_profile = profile_snapshot_json(bank.decode_profile_snapshot_and_reset());
    Ok(RustMeasure {
        seen_batches,
        measured_batches,
        measured_cells,
        measured_bytes,
        checksum,
        seconds,
        databank_profile,
        access_profile,
        io_profile,
        decode_profile,
    })
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

trait ChecksumValue {
    fn checksum_value(self) -> u64;
}

impl ChecksumValue for u16 {
    fn checksum_value(self) -> u64 {
        self as u64
    }
}

impl ChecksumValue for u32 {
    fn checksum_value(self) -> u64 {
        self as u64
    }
}

fn build_multi_batches(
    dataset_index: &[usize],
    cell_index: &[usize],
    batch_size: usize,
) -> Vec<MultiBatchCells> {
    let mut out = Vec::with_capacity(cell_index.len().div_ceil(batch_size));
    let mut start = 0usize;
    while start < cell_index.len() {
        let end = (start + batch_size).min(cell_index.len());
        let mut parts = Vec::<(usize, Vec<usize>)>::new();
        let mut run_start = start;
        while run_start < end {
            let dataset = dataset_index[run_start];
            let mut run_end = run_start + 1;
            while run_end < end && dataset_index[run_end] == dataset {
                run_end += 1;
            }
            parts.push((dataset, cell_index[run_start..run_end].to_vec()));
            run_start = run_end;
        }
        out.push(MultiBatchCells::new(parts));
        start = end;
    }
    out
}

fn default_workers(args: &Args) -> usize {
    args.threads / 4
}

fn databank_io_workers(args: &Args) -> usize {
    if args.io_workers == 0 {
        default_workers(args)
    } else {
        args.io_workers
    }
}

fn databank_decode_workers(args: &Args) -> usize {
    if args.decode_workers == 0 {
        default_workers(args)
    } else {
        args.decode_workers
    }
}

fn databank_access_workers(args: &Args) -> usize {
    if args.access_workers == 0 {
        default_workers(args)
    } else {
        args.access_workers
    }
}

fn databank_fill_workers(args: &Args) -> usize {
    if args.fill_workers == 0 {
        default_workers(args)
    } else {
        args.fill_workers
    }
}

fn rust_databank_config(args: &Args) -> DataBankConfig {
    let access_memory_bytes = args.memory_gib * 1024 * 1024 * 1024;
    DataBankConfig {
        io_config: IoConfig::Threaded(ThreadedConfig {
            num_workers: databank_io_workers(args),
            ..ThreadedConfig::default()
        }),
        decode_config: DecodePoolConfig {
            num_workers: databank_decode_workers(args),
            ..DecodePoolConfig::default()
        },
        access_config: AccessConfig {
            scheduler_shards: databank_access_workers(args),
            cache_capacity_bytes: access_memory_bytes * 3 / 4,
            memory_budget_bytes: access_memory_bytes,
            cpu: AccessCpuConfig {
                num_workers: databank_access_workers(args),
                ..AccessCpuConfig::default()
            },
            ..AccessConfig::default()
        },
        fill_config: FillConfig {
            num_workers: databank_fill_workers(args),
            ..FillConfig::default()
        },
    }
}

fn resolve_rust_genes(
    specs: &[RustDatasetSpec],
    gene_mode: &str,
    genes: usize,
) -> AppResult<Option<Vec<String>>> {
    match gene_mode {
        "native" => {
            let Some(first) = specs.first() else {
                return Err("no datasets".into());
            };
            let n = first.num_genes();
            if specs.iter().any(|spec| spec.num_genes() != n) {
                return Err("native gene mode requires identical n_vars".into());
            }
            Ok(None)
        }
        "first" => {
            let mut names = specs.first().ok_or("no datasets")?.gene_names().to_vec();
            if genes > 0 {
                names.truncate(genes);
            }
            Ok(Some(names))
        }
        other => Err(format!("unknown gene mode `{other}`").into()),
    }
}

fn resolve_output_dtype(specs: &[RustDatasetSpec], requested: &str) -> AppResult<DType> {
    let text = requested.trim().to_ascii_lowercase();
    if !matches!(text.as_str(), "stored" | "native" | "none" | "") {
        return parse_dtype_name(&text);
    }
    if specs.iter().any(|spec| spec.dtype() == DType::U32) {
        return Ok(DType::U32);
    }
    if specs.iter().all(|spec| spec.dtype() == DType::U16) {
        return Ok(DType::U16);
    }
    Err("rust-core auto dtype currently supports u16/u32 rowcount datasets".into())
}

fn build_rust_dataset_specs(catalog: &Bound<'_, PyAny>) -> AppResult<Vec<RustDatasetSpec>> {
    let mut out = Vec::new();
    for item in catalog.try_iter()? {
        let entry = item?;
        let ds = entry.get_item("dataset")?;
        out.push(build_rust_dataset_spec(&ds)?);
    }
    Ok(out)
}

fn build_rust_dataset_spec(ds: &Bound<'_, PyAny>) -> AppResult<RustDatasetSpec> {
    let kind: String = ds.getattr("kind")?.extract()?;
    let gene_names: Vec<String> = ds.getattr("gene_names")?.extract()?;
    let store_root: String = ds.getattr("store_root")?.extract()?;
    match kind.as_str() {
        "dense" => {
            let data = build_array_spec_from_py(&ds.getattr("data")?, &store_root)?;
            let ndim = data.shape.len();
            if ndim == 1 {
                Ok(RustDatasetSpec::Dense1D(Dense1DSpec { gene_names, data }))
            } else {
                Ok(RustDatasetSpec::Dense2D(Dense2DSpec { gene_names, data }))
            }
        }
        "sparse-csr" => {
            let indptr = extract_u64_vec(&ds.getattr("indptr")?, "indptr")?;
            let indices = build_array_spec_from_py(&ds.getattr("indices")?, &store_root)?;
            let data = build_array_spec_from_py(&ds.getattr("data")?, &store_root)?;
            let index_dtype = extract_dtype(&ds.getattr("index_dtype")?)?;
            Ok(RustDatasetSpec::Sparse(SparseCsrSpec {
                gene_names,
                indptr,
                indices,
                data,
                index_dtype,
                num_cells: ds.getattr("num_cells")?.extract()?,
                num_genes: ds.getattr("num_genes")?.extract()?,
            }))
        }
        other => Err(format!("unsupported dataset kind `{other}`").into()),
    }
}

fn build_array_spec_from_py(data: &Bound<'_, PyAny>, store_path: &str) -> AppResult<ArraySpec> {
    let shape: Vec<usize> = data.getattr("shape")?.extract()?;
    let chunk_shape: Vec<usize> = data.getattr("chunk_shape")?.extract()?;
    let dtype = extract_dtype(&data.getattr("dtype")?)?;
    let codec = build_codec(data.py(), &data.getattr("codec")?)?;
    let grid = build_grid_spec(data, &shape, chunk_shape)?;
    let decoded_bytes = chunk_decoded_bytes(&shape, dtype, &grid)?;
    let chunks = build_chunks(data, store_path, &decoded_bytes)?;
    Ok(ArraySpec {
        shape,
        dtype,
        order: ArrayOrder::C,
        codec,
        grid,
        chunks,
    })
}

fn build_grid_spec(
    data: &Bound<'_, PyAny>,
    shape: &[usize],
    chunk_shape: Vec<usize>,
) -> AppResult<ArrayGridSpec> {
    if let Some(axes) = optional_chunk_boundaries(data)? {
        validate_rectilinear_axes(shape, &axes)?;
        return Ok(ArrayGridSpec::Rectilinear { axes });
    }
    validate_regular_grid(shape, &chunk_shape)?;
    Ok(ArrayGridSpec::Regular {
        chunk_shape,
        edge: extract_edge_layout(data)?,
    })
}

fn optional_chunk_boundaries(data: &Bound<'_, PyAny>) -> AppResult<Option<Vec<Vec<usize>>>> {
    let value = data.getattr("chunk_boundaries")?;
    let axes: Vec<Vec<usize>> = value.extract()?;
    if axes.is_empty() {
        Ok(None)
    } else {
        Ok(Some(axes))
    }
}

fn extract_edge_layout(data: &Bound<'_, PyAny>) -> AppResult<EdgeChunkLayout> {
    let Ok(value) = data.getattr("edge") else {
        return Ok(EdgeChunkLayout::Padded);
    };
    let text: String = value.extract()?;
    match text.to_ascii_lowercase().as_str() {
        "padded" => Ok(EdgeChunkLayout::Padded),
        "cropped" => Ok(EdgeChunkLayout::Cropped),
        other => Err(format!("unknown edge layout `{other}`").into()),
    }
}

fn build_chunks(
    data: &Bound<'_, PyAny>,
    store_path: &str,
    decoded_bytes: &[usize],
) -> AppResult<Vec<ChunkSpec>> {
    let store_kind: String = data.getattr("store_kind")?.extract()?;
    let ranges = match store_kind.as_str() {
        "file" => file_ranges(data, store_path)?,
        "dir" => directory_ranges(data, store_path)?,
        other => return Err(format!("unknown store_kind `{other}`").into()),
    };
    if ranges.len() != decoded_bytes.len() {
        return Err(format!(
            "chunk source count {} != decoded chunk count {}",
            ranges.len(),
            decoded_bytes.len()
        )
        .into());
    }
    Ok(ranges
        .into_iter()
        .zip(decoded_bytes)
        .map(|((path, offset, len), &decoded_bytes)| ChunkSpec {
            source: ChunkSourceSpec::File { path, offset, len },
            decoded_bytes,
        })
        .collect())
}

fn file_ranges(data: &Bound<'_, PyAny>, store_path: &str) -> AppResult<Vec<(PathBuf, u64, usize)>> {
    let payload_path: String = data.getattr("payload_path")?.extract()?;
    let payload_file_path = optional_string_attr(data, "payload_file_path")?;
    let path = if payload_file_path.is_empty() {
        PathBuf::from(store_path).join(payload_path)
    } else {
        PathBuf::from(payload_file_path)
    };
    let offsets = extract_u64_vec(&data.getattr("chunk_offsets")?, "chunk_offsets")?;
    let lengths = extract_u64_vec(&data.getattr("chunk_lengths")?, "chunk_lengths")?;
    if offsets.len() != lengths.len() {
        return Err("chunk_offsets length != chunk_lengths length".into());
    }
    offsets
        .into_iter()
        .zip(lengths)
        .map(|(offset, len)| Ok((path.clone(), offset, usize::try_from(len)?)))
        .collect()
}

fn directory_ranges(
    data: &Bound<'_, PyAny>,
    store_path: &str,
) -> AppResult<Vec<(PathBuf, u64, usize)>> {
    let chunk_paths: Vec<String> = data.getattr("chunk_paths")?.extract()?;
    let n = chunk_paths.len();
    let chunk_file_paths: Vec<String> = data.getattr("chunk_file_paths")?.extract()?;
    let offsets = extract_u64_vec(&data.getattr("chunk_offsets")?, "chunk_offsets")?;
    let lengths = extract_u64_vec(&data.getattr("chunk_lengths")?, "chunk_lengths")?;
    if offsets.len() != n || lengths.len() != n {
        return Err("chunk_offsets/chunk_lengths count mismatch".into());
    }
    let root = PathBuf::from(store_path);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let path = if chunk_file_paths.is_empty() {
            root.join(&chunk_paths[i])
        } else {
            PathBuf::from(&chunk_file_paths[i])
        };
        out.push((path, offsets[i], usize::try_from(lengths[i])?));
    }
    Ok(out)
}

fn extract_u64_vec(obj: &Bound<'_, PyAny>, context: &str) -> AppResult<Vec<u64>> {
    if let Ok(array) = obj.extract::<PyReadonlyArray1<'_, u64>>() {
        let slice = array
            .as_slice()
            .map_err(|_| format!("{context} must be contiguous uint64"))?;
        return Ok(slice.to_vec());
    }
    Ok(obj.extract::<Vec<u64>>()?)
}

fn optional_string_attr(obj: &Bound<'_, PyAny>, name: &str) -> AppResult<String> {
    match obj.getattr(name) {
        Ok(value) => Ok(value.extract()?),
        Err(_) => Ok(String::new()),
    }
}

fn build_codec(py: Python<'_>, codec: &Bound<'_, PyAny>) -> AppResult<ArrayCodecSpec> {
    let is_uncompressed: bool = codec.getattr("is_uncompressed")?.extract()?;
    if is_uncompressed {
        return Ok(ArrayCodecSpec::Uncompressed);
    }
    let pair = codec.getattr("to_zarr")?.call0()?;
    let filters_obj = pair.get_item(0)?;
    let compressor_obj = pair.get_item(1)?;
    let json = py.import("json")?;
    let dumps = json.getattr("dumps")?;
    Ok(ArrayCodecSpec::ZarrV2Json {
        filters: json_opt(&dumps, &filters_obj)?,
        compressor: json_opt(&dumps, &compressor_obj)?,
    })
}

fn json_opt(dumps: &Bound<'_, PyAny>, obj: &Bound<'_, PyAny>) -> AppResult<Option<String>> {
    if obj.is_none() {
        return Ok(None);
    }
    Ok(Some(dumps.call1((obj.clone().unbind(),))?.extract()?))
}

fn extract_dtype(obj: &Bound<'_, PyAny>) -> AppResult<DType> {
    let value: String = obj.getattr("value")?.extract()?;
    parse_dtype_name(&value)
}

fn parse_dtype_name(value: &str) -> AppResult<DType> {
    match value.trim().to_ascii_lowercase().as_str() {
        "u8" => Ok(DType::U8),
        "i8" => Ok(DType::I8),
        "u16" => Ok(DType::U16),
        "i16" => Ok(DType::I16),
        "u32" => Ok(DType::U32),
        "i32" => Ok(DType::I32),
        "u64" => Ok(DType::U64),
        "i64" => Ok(DType::I64),
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        "f32" => Ok(DType::F32),
        "f64" => Ok(DType::F64),
        other => Err(format!("unknown dtype `{other}`").into()),
    }
}

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::U8 => "u8",
        DType::I8 => "i8",
        DType::U16 => "u16",
        DType::I16 => "i16",
        DType::U32 => "u32",
        DType::I32 => "i32",
        DType::U64 => "u64",
        DType::I64 => "i64",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::F32 => "f32",
        DType::F64 => "f64",
    }
}

fn validate_regular_grid(shape: &[usize], chunk_shape: &[usize]) -> AppResult<()> {
    if shape.is_empty() {
        return Err("array shape must not be empty".into());
    }
    if shape.len() != chunk_shape.len() {
        return Err("shape rank != chunk_shape rank".into());
    }
    if chunk_shape.iter().any(|&chunk| chunk == 0) {
        return Err("chunk_shape values must be positive".into());
    }
    Ok(())
}

fn validate_rectilinear_axes(shape: &[usize], axes: &[Vec<usize>]) -> AppResult<()> {
    if axes.len() != shape.len() {
        return Err("chunk_boundaries rank != shape rank".into());
    }
    for (bounds, &dim) in axes.iter().zip(shape) {
        if bounds.len() < 2
            || bounds.first().copied() != Some(0)
            || bounds.last().copied() != Some(dim)
        {
            return Err("invalid rectilinear chunk boundaries".into());
        }
        if bounds.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("chunk boundaries must be strictly increasing".into());
        }
    }
    Ok(())
}

fn chunk_decoded_bytes(
    shape: &[usize],
    dtype: DType,
    grid: &ArrayGridSpec,
) -> AppResult<Vec<usize>> {
    let grid_shape = grid_shape(shape, grid)?;
    let count = grid_shape
        .iter()
        .try_fold(1usize, |acc, &value| acc.checked_mul(value))
        .ok_or("chunk grid size overflow")?;
    let mut out = Vec::with_capacity(count);
    for chunk_index in 0..count {
        let coords = chunk_coords(chunk_index, &grid_shape);
        let elements = match grid {
            ArrayGridSpec::Regular { chunk_shape, edge } => {
                regular_chunk_elements(shape, chunk_shape, *edge, &coords)?
            }
            ArrayGridSpec::Rectilinear { axes } => rectilinear_chunk_elements(axes, &coords)?,
        };
        out.push(
            elements
                .checked_mul(dtype.item_size())
                .ok_or("decoded byte size overflow")?,
        );
    }
    Ok(out)
}

fn grid_shape(shape: &[usize], grid: &ArrayGridSpec) -> AppResult<Vec<usize>> {
    match grid {
        ArrayGridSpec::Regular { chunk_shape, .. } => Ok(shape
            .iter()
            .zip(chunk_shape)
            .map(|(&dim, &chunk)| dim.div_ceil(chunk))
            .collect()),
        ArrayGridSpec::Rectilinear { axes } => Ok(axes.iter().map(|axis| axis.len() - 1).collect()),
    }
}

fn chunk_coords(mut index: usize, grid_shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0usize; grid_shape.len()];
    for axis in (0..grid_shape.len()).rev() {
        let dim = grid_shape[axis];
        coords[axis] = index % dim;
        index /= dim;
    }
    coords
}

fn regular_chunk_elements(
    shape: &[usize],
    chunk_shape: &[usize],
    edge: EdgeChunkLayout,
    coords: &[usize],
) -> AppResult<usize> {
    shape
        .iter()
        .zip(chunk_shape)
        .zip(coords)
        .try_fold(1usize, |elements, ((&dim, &chunk), &coord)| {
            let extent = match edge {
                EdgeChunkLayout::Padded => chunk,
                EdgeChunkLayout::Cropped => dim.saturating_sub(coord * chunk).min(chunk),
            };
            elements.checked_mul(extent).ok_or("chunk element overflow")
        })
        .map_err(Into::into)
}

fn rectilinear_chunk_elements(axes: &[Vec<usize>], coords: &[usize]) -> AppResult<usize> {
    axes.iter()
        .zip(coords)
        .try_fold(1usize, |elements, (bounds, &coord)| {
            let extent = bounds[coord + 1] - bounds[coord];
            elements.checked_mul(extent).ok_or("chunk element overflow")
        })
        .map_err(Into::into)
}

fn bench_io_ranges(
    ranges: &[RangeSpec],
    threads: usize,
    max_bytes_per_read: usize,
    seed: u64,
) -> AppResult<Value> {
    let threads = threads.max(1).min(ranges.len().max(1));
    let mut order: Vec<usize> = (0..ranges.len()).collect();
    shuffle(&mut order, &mut SimpleRng::new(seed ^ 0xA11C_E5));

    let mut files = HashMap::<String, Arc<File>>::new();
    for range in ranges {
        if !files.contains_key(&range.path) {
            files.insert(range.path.clone(), Arc::new(File::open(&range.path)?));
        }
    }
    let files = Arc::new(files);
    let ranges = Arc::new(ranges.to_vec());
    let order = Arc::new(order);
    let started = Instant::now();

    let mut handles = Vec::with_capacity(threads);
    for worker in 0..threads {
        let files = Arc::clone(&files);
        let ranges = Arc::clone(&ranges);
        let order = Arc::clone(&order);
        handles.push(thread::spawn(
            move || -> io::Result<(usize, usize, u64, usize)> {
                let mut reads = 0usize;
                let mut bytes = 0usize;
                let mut checksum = 0u64;
                let mut truncated = 0usize;
                for pos in (worker..order.len()).step_by(threads) {
                    let range = &ranges[order[pos]];
                    let read_len = if max_bytes_per_read == 0 {
                        range.length
                    } else {
                        range.length.min(max_bytes_per_read)
                    };
                    if read_len < range.length {
                        truncated += 1;
                    }
                    let mut buf = vec![0u8; read_len];
                    let file = files.get(&range.path).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::NotFound,
                            format!("missing file {}", range.path),
                        )
                    })?;
                    read_exact_at(file, &mut buf, range.offset)?;
                    reads += 1;
                    bytes += read_len;
                    if let Some(first) = buf.first() {
                        checksum = checksum.wrapping_add(*first as u64);
                    }
                    if let Some(last) = buf.last() {
                        checksum ^= (*last as u64) << 8;
                    }
                }
                Ok((reads, bytes, checksum, truncated))
            },
        ));
    }

    let mut reads = 0usize;
    let mut bytes = 0usize;
    let mut checksum = 0u64;
    let mut truncated = 0usize;
    for handle in handles {
        let (r, b, c, t) = handle.join().map_err(|_| "io worker panicked")??;
        reads += r;
        bytes += b;
        checksum = checksum.wrapping_add(c);
        truncated += t;
    }
    let seconds = started.elapsed().as_secs_f64();
    let by_array = ranges
        .iter()
        .fold(HashMap::<String, usize>::new(), |mut acc, range| {
            *acc.entry(range.array.clone()).or_default() += 1;
            acc
        });
    let datasets_seen = ranges
        .iter()
        .map(|range| range.dataset_idx)
        .collect::<HashSet<_>>()
        .len();

    Ok(json!({
        "kind": "io_read_ranges",
        "threads": threads,
        "ranges": ranges.len(),
        "reads": reads,
        "bytes": bytes,
        "seconds": seconds,
        "read_gb_per_s": bytes as f64 / seconds.max(f64::MIN_POSITIVE) / 1e9,
        "reads_per_s": rate(reads, seconds),
        "checksum": checksum,
        "truncated_reads": truncated,
        "max_bytes_per_read": max_bytes_per_read,
        "files": files.len(),
        "datasets_seen": datasets_seen,
        "by_array": by_array,
    }))
}

fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    let mut filled = 0usize;
    while filled < buf.len() {
        let n = file.read_at(&mut buf[filled..], offset + filled as u64)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short read while sampling chunk ranges",
            ));
        }
        filled += n;
    }
    Ok(())
}

fn parse_ranges(value: &Value) -> AppResult<Vec<RangeSpec>> {
    let ranges = value
        .get("ranges")
        .and_then(Value::as_array)
        .ok_or("io range payload missing ranges")?;
    let mut out = Vec::with_capacity(ranges.len());
    for item in ranges {
        out.push(RangeSpec {
            path: item
                .get("path")
                .and_then(Value::as_str)
                .ok_or("range missing path")?
                .to_string(),
            offset: item
                .get("offset")
                .and_then(Value::as_u64)
                .ok_or("range missing offset")?,
            length: item
                .get("length")
                .and_then(Value::as_u64)
                .ok_or("range missing length")? as usize,
            dataset_idx: item
                .get("dataset_idx")
                .and_then(Value::as_u64)
                .ok_or("range missing dataset_idx")? as usize,
            array: item
                .get("array")
                .and_then(Value::as_str)
                .unwrap_or("data")
                .to_string(),
        });
    }
    Ok(out)
}

fn sampled_global_indices(total: usize, n: usize, seed: u64) -> Vec<usize> {
    if n >= total {
        let mut out: Vec<usize> = (0..total).collect();
        shuffle(&mut out, &mut SimpleRng::new(seed));
        return out;
    }

    let mut rng = SimpleRng::new(seed);
    let mut selected = HashSet::with_capacity(n);
    for j in (total - n)..total {
        let t = rng.gen_range(j + 1);
        if !selected.insert(t) {
            selected.insert(j);
        }
    }
    let mut out: Vec<usize> = selected.into_iter().collect();
    shuffle(&mut out, &mut rng);
    out
}

fn ordered_indices(base: &[usize], counts: &[usize], order: Order, seed: u64) -> Vec<usize> {
    let mut out = match order {
        Order::Sequential => {
            let total = counts.iter().copied().sum::<usize>();
            (0..base.len().min(total)).collect()
        }
        _ => base.to_vec(),
    };
    match order {
        Order::Random => shuffle(&mut out, &mut SimpleRng::new(seed ^ 0x5151_5151)),
        Order::Grouped => out.sort_unstable(),
        Order::Sequential => {}
    }
    out
}

fn shuffle(values: &mut [usize], rng: &mut SimpleRng) {
    if values.len() <= 1 {
        return;
    }
    for i in (1..values.len()).rev() {
        let j = rng.gen_range(i + 1);
        values.swap(i, j);
    }
}

fn offsets(counts: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(counts.len() + 1);
    out.push(0);
    let mut acc = 0usize;
    for count in counts {
        acc += *count;
        out.push(acc);
    }
    out
}

fn plan_arrays(order: &[usize], offsets: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut dataset_index = Vec::with_capacity(order.len());
    let mut cell_index = Vec::with_capacity(order.len());
    for global in order {
        let pos = offsets.partition_point(|offset| offset <= global);
        let dataset = pos.saturating_sub(1);
        dataset_index.push(dataset);
        cell_index.push(global - offsets[dataset]);
    }
    (dataset_index, cell_index)
}

fn plan_stats(dataset_index: &[usize], batch_size: usize) -> Value {
    let mut parts_per_batch = Vec::new();
    let mut total_parts = 0usize;
    let mut max_parts = 0usize;
    for batch in dataset_index.chunks(batch_size) {
        if batch.is_empty() {
            continue;
        }
        let mut parts = 1usize;
        for pair in batch.windows(2) {
            if pair[0] != pair[1] {
                parts += 1;
            }
        }
        total_parts += parts;
        max_parts = max_parts.max(parts);
        parts_per_batch.push(parts);
    }
    parts_per_batch.sort_unstable();
    let batches = parts_per_batch.len();
    let p50 = percentile(&parts_per_batch, 0.50);
    let p95 = percentile(&parts_per_batch, 0.95);
    json!({
        "batches": batches,
        "total_parts": total_parts,
        "avg_parts_per_batch": if batches == 0 { 0.0 } else { total_parts as f64 / batches as f64 },
        "max_parts_per_batch": max_parts,
        "p50_parts_per_batch": p50,
        "p95_parts_per_batch": p95,
    })
}

fn percentile(sorted: &[usize], q: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn counts_from_summary(value: &Value) -> AppResult<Vec<usize>> {
    let counts = value
        .get("counts")
        .and_then(Value::as_array)
        .ok_or("catalog summary missing counts")?;
    counts
        .iter()
        .map(|v| {
            v.as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| "catalog count is not a positive integer".into())
        })
        .collect()
}

fn case_configs(args: &Args) -> Vec<CaseConfig> {
    let mut out = Vec::new();
    for &batch_size in &args.batch_sizes {
        for &order in &args.orders {
            for &prefetch_step in &args.prefetch_steps {
                for &access_prefetch_step in &args.access_prefetch_steps {
                    for &decode_ahead_steps in &args.decode_ahead_steps {
                        for &ready_ahead_steps in &args.ready_ahead_steps {
                            out.push(CaseConfig {
                                batch_size,
                                order,
                                prefetch_step,
                                access_prefetch_step,
                                decode_ahead_steps,
                                ready_ahead_steps,
                            });
                        }
                    }
                }
            }
        }
    }
    out
}

fn emit(writer: &mut Option<File>, value: Value) -> AppResult<()> {
    let line = serde_json::to_string(&value)?;
    println!("{line}");
    if let Some(file) = writer.as_mut() {
        writeln!(file, "{line}")?;
        file.flush()?;
    }
    Ok(())
}

fn open_writer(args: &Args) -> AppResult<Option<File>> {
    let Some(path) = &args.out else {
        return Ok(None);
    };
    if path.as_os_str() == "-" {
        return Ok(None);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(Some(
        OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)?,
    ))
}

fn rate(count: usize, seconds: f64) -> f64 {
    count as f64 / seconds.max(f64::MIN_POSITIVE)
}

fn args_json(args: &Args) -> Value {
    json!({
        "root": args.root,
        "repo_root": args.repo_root,
        "mode": match args.mode {
            Mode::Both => "both",
            Mode::Scheduled => "scheduled",
            Mode::Io => "io",
        },
        "engine": args.engine.as_str(),
        "limit_datasets": args.limit_datasets,
        "max_cells": args.max_cells,
        "batch_sizes": args.batch_sizes,
        "orders": args.orders.iter().map(|order| order.as_str()).collect::<Vec<_>>(),
        "prefetch_steps": args.prefetch_steps,
        "access_prefetch_steps": args.access_prefetch_steps,
        "decode_ahead_steps": args.decode_ahead_steps,
        "ready_ahead_steps": args.ready_ahead_steps,
        "threads": args.threads,
        "io_workers": databank_io_workers(args),
        "decode_workers": databank_decode_workers(args),
        "access_workers": databank_access_workers(args),
        "fill_workers": databank_fill_workers(args),
        "memory_gib": args.memory_gib,
        "gene_mode": args.gene_mode,
        "projected_sparse_data_strategy": args.projected_sparse_data_strategy.as_str(),
        "genes": args.genes,
        "dtype": args.dtype,
        "seed": args.seed,
        "warmup_batches": args.warmup_batches,
        "measure_batches": args.measure_batches,
        "out": args.out,
        "io_samples": args.io_samples,
        "io_threads": args.io_threads,
        "io_max_bytes_per_read": args.io_max_bytes_per_read,
        "reuse_bank": args.reuse_bank,
    })
}

fn parse_args() -> AppResult<Args> {
    let mut args = Args {
        root: default_root(),
        repo_root: infer_repo_root()?,
        mode: Mode::Both,
        engine: Engine::RustCore,
        limit_datasets: 32,
        max_cells: Some(8192),
        batch_sizes: vec![128],
        orders: vec![Order::Random, Order::Grouped],
        prefetch_steps: vec![32],
        access_prefetch_steps: vec![64],
        decode_ahead_steps: vec![32],
        ready_ahead_steps: vec![16],
        threads: 96,
        io_workers: 0,
        decode_workers: 0,
        access_workers: 0,
        fill_workers: 0,
        memory_gib: 128,
        gene_mode: "first".to_string(),
        projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy::SelectedOnly,
        genes: 0,
        dtype: "stored".to_string(),
        seed: 0,
        warmup_batches: 2,
        measure_batches: 64,
        out: Some(default_output_path()?),
        io_samples: 2000,
        io_threads: 32,
        io_max_bytes_per_read: 0,
        reuse_bank: false,
    };

    let mut iter = env::args().skip(1).peekable();
    while let Some(arg) = iter.next() {
        if arg == "--help" || arg == "-h" {
            print_help();
            std::process::exit(0);
        }
        if arg == "--reuse-bank" {
            args.reuse_bank = true;
            continue;
        }
        if arg == "--no-io" {
            args.io_samples = 0;
            continue;
        }
        let (key, value) = if let Some((key, value)) = arg.split_once('=') {
            (key.to_string(), value.to_string())
        } else {
            let value = iter
                .next()
                .ok_or_else(|| format!("missing value for argument `{arg}`"))?;
            (arg, value)
        };
        match key.as_str() {
            "--root" => args.root = PathBuf::from(value),
            "--repo-root" => args.repo_root = PathBuf::from(value),
            "--mode" => {
                args.mode = match value.trim().to_ascii_lowercase().as_str() {
                    "both" => Mode::Both,
                    "scheduled" => Mode::Scheduled,
                    "io" => Mode::Io,
                    other => return Err(format!("unknown mode `{other}`").into()),
                }
            }
            "--engine" => args.engine = Engine::parse(&value)?,
            "--limit-datasets" => args.limit_datasets = parse_usize(&value, &key)?,
            "--max-cells" => {
                let n = parse_usize(&value, &key)?;
                args.max_cells = (n > 0).then_some(n);
            }
            "--batch-sizes" => args.batch_sizes = parse_usize_list(&value, &key)?,
            "--orders" => args.orders = parse_order_list(&value)?,
            "--prefetch-steps" => args.prefetch_steps = parse_usize_list(&value, &key)?,
            "--access-prefetch-steps" => {
                args.access_prefetch_steps = parse_usize_list(&value, &key)?
            }
            "--decode-ahead-steps" => args.decode_ahead_steps = parse_usize_list(&value, &key)?,
            "--ready-ahead-steps" => args.ready_ahead_steps = parse_usize_list(&value, &key)?,
            "--threads" => args.threads = parse_usize(&value, &key)?,
            "--io-workers" => args.io_workers = parse_usize(&value, &key)?,
            "--decode-workers" => args.decode_workers = parse_usize(&value, &key)?,
            "--access-workers" => args.access_workers = parse_usize(&value, &key)?,
            "--fill-workers" => args.fill_workers = parse_usize(&value, &key)?,
            "--memory-gib" => args.memory_gib = parse_usize(&value, &key)?,
            "--gene-mode" => args.gene_mode = value,
            "--projected-sparse-data-strategy" | "--sparse-data-strategy" => {
                args.projected_sparse_data_strategy =
                    ProjectedSparseDataGroupStrategy::parse(&value)?;
            }
            "--genes" => args.genes = parse_usize(&value, &key)?,
            "--dtype" => args.dtype = value,
            "--seed" => args.seed = value.parse()?,
            "--warmup-batches" => args.warmup_batches = parse_usize(&value, &key)?,
            "--measure-batches" => args.measure_batches = parse_usize(&value, &key)?,
            "--out" => {
                args.out = if value == "-" {
                    None
                } else {
                    Some(PathBuf::from(value))
                }
            }
            "--io-samples" => args.io_samples = parse_usize(&value, &key)?,
            "--io-threads" => args.io_threads = parse_usize(&value, &key)?,
            "--io-max-bytes-per-read" => args.io_max_bytes_per_read = parse_usize(&value, &key)?,
            other => return Err(format!("unknown argument `{other}`").into()),
        }
    }

    validate_args(&args)?;
    Ok(args)
}

fn validate_args(args: &Args) -> AppResult<()> {
    if args.threads == 0 || args.threads % 4 != 0 {
        return Err("--threads must be positive and divisible by 4".into());
    }
    if databank_io_workers(args) == 0
        || databank_decode_workers(args) == 0
        || databank_access_workers(args) == 0
        || databank_fill_workers(args) == 0
    {
        return Err("worker counts must be positive".into());
    }
    if args.batch_sizes.iter().any(|&x| x == 0)
        || args.prefetch_steps.iter().any(|&x| x == 0)
        || args.access_prefetch_steps.iter().any(|&x| x == 0)
        || args.decode_ahead_steps.iter().any(|&x| x == 0)
        || args.ready_ahead_steps.iter().any(|&x| x == 0)
    {
        return Err("batch/prefetch/decode/ready values must be positive".into());
    }
    match args.gene_mode.as_str() {
        "first" | "native" => {}
        other => return Err(format!("--gene-mode must be first or native, got `{other}`").into()),
    }
    Ok(())
}

fn parse_usize(value: &str, name: &str) -> AppResult<usize> {
    value
        .parse::<usize>()
        .map_err(|err| format!("{name} expects usize, got `{value}`: {err}").into())
}

fn parse_usize_list(value: &str, name: &str) -> AppResult<Vec<usize>> {
    let values: Result<Vec<_>, _> = value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| parse_usize(part.trim(), name))
        .collect();
    let values = values?;
    if values.is_empty() {
        return Err(format!("{name} must not be empty").into());
    }
    Ok(values)
}

fn parse_order_list(value: &str) -> AppResult<Vec<Order>> {
    let values: Result<Vec<_>, _> = value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(Order::parse)
        .collect();
    let values = values?;
    if values.is_empty() {
        return Err("--orders must not be empty".into());
    }
    Ok(values)
}

fn default_root() -> PathBuf {
    let base = env::var("mntwzq")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/mnt/shared-storage-user/dnacoding/wangzhongqi"));
    let root = base.join("Data/cellxgene/homo_spacian");
    if root.exists() {
        root
    } else {
        base.join("Data/cellxgene/Homo_sapiens")
    }
}

fn infer_repo_root() -> AppResult<PathBuf> {
    let mut dir = env::current_dir()?;
    loop {
        if dir.join("scdata/__init__.py").exists() {
            return Ok(dir);
        }
        if !dir.pop() {
            return env::current_dir().map_err(Into::into);
        }
    }
}

fn default_output_path() -> AppResult<PathBuf> {
    let ts = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    Ok(PathBuf::from(format!(
        "outputs/profile-real-access-{ts}.jsonl"
    )))
}

fn print_help() {
    println!(
        r#"profile_real_access: profile real .zarr.zip scheduled access and raw chunk IO

Typical:
  cargo run --no-default-features --features pybind-bench,profile --bin profile_real_access -- \
    --limit-datasets 32 --max-cells 8192 --batch-sizes 64,128,256 --orders random,grouped

Key options:
  --root PATH                         Dataset root, default follows test.py.
  --repo-root PATH                    Python repo root for importing local scdata.
  --mode both|scheduled|io            Default: both.
  --engine rust-core|py-wrapper       Default: rust-core.
  --limit-datasets N                  0 means all datasets. Default: 32.
  --max-cells N                       0 means all cells. Default: 8192.
  --batch-sizes A,B                   Default: 128.
  --orders random,grouped,sequential  Default: random,grouped.
  --prefetch-steps A,B                Default: 32.
  --access-prefetch-steps A,B         Default: 64.
  --decode-ahead-steps A,B            Default: 32.
  --ready-ahead-steps A,B             Default: 16.
  --threads N                         Must be divisible by 4. Default: 96.
  --io-workers N                      DataBank IO workers. 0 means --threads/4.
  --decode-workers N                  Decode workers. 0 means --threads/4.
  --access-workers N                  Access CPU workers. 0 means --threads/4.
  --fill-workers N                    DataBank request/response/fill workers. 0 means --threads/4.
  --memory-gib N                      Default: 128.
  --gene-mode first|native            Default: first.
  --projected-sparse-data-strategy selected_only|read_all
                                      Default: selected_only.
  --genes N                           0 means all genes in first dataset. Default: 0.
  --warmup-batches N                  Default: 2.
  --measure-batches N                 0 means all measured batches. Default: 64.
  --io-samples N                      Default: 2000. Use --no-io to disable.
  --io-threads N                      Default: 32.
  --io-max-bytes-per-read N           0 means full chunk range. Default: 0.
  --out PATH                          JSONL output path. Use '-' for stdout only.
  --reuse-bank                        Reuse one ScDataBank across scheduled cases.
"#
    );
}
