#!/usr/bin/env python3
"""scdata cell-access benchmark CLI.

A unified performance evaluation tool for the scdata DataBank. It scans a
directory of ``.zarr.zip`` stores, picks a row-count matrix (``u16``/``u32``)
from each, registers them with a :class:`~scdata.ScDataBank`, and measures
cell access throughput along two orthogonal axes.

Access *order* (``--order``):

* ``random``     — cells drawn in a shuffled order across the concatenated
  corpus (the random-access workload the DataBank is built for; default).
* ``sequential`` — cells read 0, 1, 2, ... front-to-back; the prefetch- and
  cache-friendly control that isolates seek cost from decode throughput.

Access *path* (``--mode``):

* ``unscheduled`` — one synchronous :meth:`ScDataBank.load` call per
  (dataset, batch) part; the baseline single-call path.
* ``scheduled``   — a streaming :meth:`ScDataBank.prefetch_indexed` over a
  :class:`~scdata.CellIndexPlan` built from the chosen order; the pipelined
  prefetch path, including the Blosc-LZ4 fast path when enabled.

The two axes compose: e.g. ``--order sequential --mode both`` compares the
unscheduled vs. scheduled paths on a sequential stream, while
``--order random`` does the same on a random stream.

Beyond raw throughput, the tool records:

* per-run ``profile_snapshot`` from the Rust core (``--profile``);
* the fast path's ``resolved_strategy`` / ``fallback_reason`` for scheduled
  runs (so you can see whether ``blosc_lz4_fast`` actually engaged);
* mean / stdev / min / max across ``--repeat`` runs (steady-state behavior);
* a machine-info block (hostname, CPU count, scdata version, ...) for
  reproducible comparison.

Subcommands:

* ``bench`` (default) — run the benchmark.
* ``scan``            — list available datasets without running anything.
* ``diff``            — compare two result JSON files.

Usage examples::

    # default: both paths, random order, fast path off, one repeat
    python scripts/bench_access.py

    # fast-path scheduled sweep with warmup + 3 repeats, save + summarize
    python scripts/bench_access.py bench \\
        --mode scheduled --fast-enabled --fast-mode force \\
        --warmup 8 --repeat 3 --profile --summary

    # sequential-order control vs random-order run
    python scripts/bench_access.py bench --order sequential --summary
    python scripts/bench_access.py bench --order random     --summary

    # quick smoke test on a handful of datasets
    python scripts/bench_access.py --limit-datasets 4 --max-cells 20000

    # inspect what is under a root
    python scripts/bench_access.py scan --root /path/to/cellxgene

    # compare two saved runs
    python scripts/bench_access.py diff run-a.json run-b.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from scdata import (
    CellIndexPlan,
    DType,
    DataBankConfig,
    ScDataBank,
    ScheduledAccessConfig,
    ScheduledPrefetchConfig,
    launch_all,
)
from scdata import __version__ as scdata_version

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # tqdm is optional; fall back to a no-op wrapper.

    def tqdm(iterable: Any, **_: Any) -> Any:
        return iterable


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_ROOT = (
    Path(os.environ.get("mntwzq", "/mnt/shared-storage-user/dnacoding/wangzhongqi"))
    / "Data/cellxgene/homo_spacian"
)
if not DEFAULT_ROOT.exists():
    DEFAULT_ROOT = DEFAULT_ROOT.with_name("Homo_sapiens")

DEFAULT_OUTPUT_DIR = Path("outputs/bench")

ROWCOUNT_DTYPES = {DType.U16, DType.U32}
DEFAULT_PREFETCH_STEP = 32
DEFAULT_ACCESS_PREFETCH_STEP = 64
DEFAULT_DECODE_AHEAD_STEPS = 32
DEFAULT_READY_AHEAD_STEPS = 16


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogEntry:
    path: Path
    matrix: str
    dataset: object
    dtype: str
    n_obs: int
    n_vars: int


def resolve_worker_counts(
    *,
    threads: int,
    io_workers: int,
    decode_workers: int,
    access_workers: int,
    fill_workers: int,
) -> tuple[int, int, int, int]:
    requested = (io_workers, decode_workers, access_workers, fill_workers)
    if any(value < 0 for value in requested):
        raise ValueError("worker counts must be non-negative")
    if any(value == 0 for value in requested):
        if threads % 4 != 0:
            raise ValueError("--threads must be divisible by 4 when a worker count is omitted")
        default_workers = threads // 4
    else:
        default_workers = 0
    counts = tuple(value or default_workers for value in requested)
    if any(value < 1 for value in counts):
        raise ValueError("resolved worker counts must be positive")
    return counts  # type: ignore[return-value]


def make_config(
    memory_gib: int,
    io_workers: int,
    decode_workers: int,
    access_workers: int,
    fill_workers: int,
    fast_enabled: bool,
    fast_fused_workers: int,
    fast_prefetch_blocks: int,
    fast_load_scheduler_workers: int,
    fast_load_io_workers: int,
    fast_coalesce_max_gap_bytes: int,
    fast_coalesce_max_waste_ratio: float,
    fast_coalesce_max_merged_len: int,
) -> DataBankConfig:
    return DataBankConfig.make(
        backend="threaded",
        io__threaded__num_workers=io_workers,
        decode__num_workers=decode_workers,
        access__cpu__num_workers=access_workers,
        fill__num_workers=fill_workers,
        access__cache_capacity_bytes=memory_gib * 1024**3 * 3 // 4,
        access__memory_budget_bytes=memory_gib * 1024**3,
        access__scheduler_shards=access_workers,
        fast__enabled=fast_enabled,
        fast__fused_workers=fast_fused_workers,
        fast__request_prefetch_blocks=fast_prefetch_blocks,
        fast__memory_budget_bytes=memory_gib * 1024**3,
        fast__response_queue_bytes_soft_limit=memory_gib * 1024**3 // 2,
        fast__response_queue_bytes_hard_limit=memory_gib * 1024**3 * 3 // 4,
        fast__load__scheduler_workers=fast_load_scheduler_workers,
        fast__load__io_workers=fast_load_io_workers,
        fast__load__coalesce__max_gap_bytes=fast_coalesce_max_gap_bytes,
        fast__load__coalesce__max_waste_ratio=fast_coalesce_max_waste_ratio,
        fast__load__coalesce__max_merged_len=fast_coalesce_max_merged_len,
    )


def pick_rowcount(path: Path) -> tuple[str | None, object | None]:
    datasets = launch_all(path)
    keys = ["X"]
    if "raw/X" in datasets:
        keys.append("raw/X")
    keys.extend(f"layers/{name}" for name in sorted(datasets.layers))
    for key in keys:
        ds = datasets[key]
        if ds.dtype in ROWCOUNT_DTYPES:
            return key, ds
    return None, None


def build_catalog(
    root: Path,
    limit_datasets: int,
    quiet: bool = False,
) -> tuple[list[CatalogEntry], list[tuple[Path, str]]]:
    paths = sorted(root.rglob("*.zarr.zip"))
    if limit_datasets:
        paths = paths[:limit_datasets]
    if not paths:
        raise SystemExit(f"no .zarr.zip found under {root}")

    catalog: list[CatalogEntry] = []
    skipped: list[tuple[Path, str]] = []
    for path in tqdm(paths, desc="scan", unit="dataset", disable=quiet):
        try:
            key, ds = pick_rowcount(path)
            if ds is None:
                skipped.append((path, "no u16/u32 matrix"))
                continue
            catalog.append(
                CatalogEntry(
                    path=path,
                    matrix=key,
                    dataset=ds,
                    dtype=ds.dtype.value,
                    n_obs=int(ds.num_cells),
                    n_vars=int(ds.num_genes),
                )
            )
        except Exception as err:  # noqa: BLE001 — catalog scan must keep going
            skipped.append((path, repr(err)))

    if not catalog:
        raise SystemExit(f"no rowcount datasets found under {root}")
    return catalog, skipped


def register_catalog(
    bank: ScDataBank,
    catalog: list[CatalogEntry],
    quiet: bool = False,
) -> list:
    ids = []
    for entry in tqdm(catalog, desc="register", unit="dataset", disable=quiet):
        ids.append(bank.register(entry.dataset))
    return ids


# ---------------------------------------------------------------------------
# Plan / order helpers
# ---------------------------------------------------------------------------


def flat_shuffle(counts: np.ndarray, seed: int, max_cells: int | None) -> np.ndarray:
    total = int(counts.sum())
    n = total if max_cells is None else min(int(max_cells), total)
    rng = np.random.default_rng(seed)
    if n == total:
        order = np.arange(total, dtype=np.int64)
        rng.shuffle(order)
        return order
    return rng.choice(total, size=n, replace=False).astype(np.int64, copy=False)


def flat_sequential(counts: np.ndarray, max_cells: int | None) -> np.ndarray:
    """Sequential flat cell order: 0, 1, 2, ... across the concatenated corpus.

    The natural control for ``flat_shuffle``: the same global index space, but
    read front-to-back so prefetch / cache effects dominate (versus random
    access amplifying seek cost).  ``max_cells`` takes the first ``n`` cells.
    """
    total = int(counts.sum())
    n = total if max_cells is None else min(int(max_cells), total)
    return np.arange(n, dtype=np.int64)


def build_order(counts: np.ndarray, order: str, seed: int, max_cells: int | None) -> np.ndarray:
    if order == "sequential":
        return flat_sequential(counts, max_cells)
    if order == "random":
        return flat_shuffle(counts, seed, max_cells)
    raise ValueError(f"unknown --order {order!r}")


def iter_batches(
    order: np.ndarray, offsets: np.ndarray, batch_size: int
) -> Iterator[list[tuple[int, np.ndarray]]]:
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        dataset_idx = np.searchsorted(offsets[1:], chunk, side="right")
        local_cells = chunk - offsets[dataset_idx]
        parts: list[tuple[int, np.ndarray]] = []
        i = 0
        while i < len(chunk):
            dataset = int(dataset_idx[i])
            j = i + 1
            while j < len(chunk) and int(dataset_idx[j]) == dataset:
                j += 1
            parts.append((dataset, local_cells[i:j].astype(np.intp, copy=False)))
            i = j
        yield parts


def indexed_plan(order: np.ndarray, offsets: np.ndarray, batch_size: int) -> CellIndexPlan:
    dataset_idx = np.searchsorted(offsets[1:], order, side="right")
    local_cells = order - offsets[dataset_idx]
    if len(offsets) - 2 <= np.iinfo(np.uint16).max:
        dataset_idx = dataset_idx.astype(np.uint16, copy=False)
    elif len(offsets) - 2 <= np.iinfo(np.uint32).max:
        dataset_idx = dataset_idx.astype(np.uint32, copy=False)
    if local_cells.size and int(local_cells.max()) <= np.iinfo(np.uint32).max:
        local_cells = local_cells.astype(np.uint32, copy=False)
    return CellIndexPlan(dataset_idx, local_cells, batch_size)


def resolve_genes(bank: ScDataBank, ids: list, args: argparse.Namespace):
    if args.gene_mode == "native":
        return None
    genes = bank.dataset_genes(ids[0])
    if args.genes:
        genes = genes[: args.genes]
    return genes


def resolve_dtype(args: argparse.Namespace) -> str | None:
    if args.dtype in ("stored", "native", "auto", "none"):
        return None
    return args.dtype


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def _sample(
    *,
    mode: str,
    cells: int,
    batches: int,
    parts: int,
    bytes_read: int,
    checksum: int,
    seconds: float,
    warmup_batches: int,
    resolved_strategy: str | None,
    fallback_reason: str | None,
) -> dict:
    return {
        "mode": mode,
        "cells": cells,
        "batches": batches,
        "parts": parts,
        "seconds": seconds,
        "cells_per_s": cells / seconds if seconds else 0.0,
        "gb_per_s": bytes_read / seconds / 1e9 if seconds else 0.0,
        "bytes": bytes_read,
        "checksum": checksum,
        "warmup_batches": warmup_batches,
        "resolved_strategy": resolved_strategy,
        "fallback_reason": fallback_reason,
    }


def bench_unscheduled_once(
    bank: ScDataBank,
    ids: list,
    order: np.ndarray,
    offsets: np.ndarray,
    args: argparse.Namespace,
    warmup: int,
) -> dict:
    genes = resolve_genes(bank, ids, args)
    dtype = resolve_dtype(args)
    missing = "zero" if genes is not None else None
    total_batches = (len(order) + args.batch_size - 1) // args.batch_size
    iterator = iter_batches(order, offsets, args.batch_size)

    cells = batches = parts_seen = bytes_read = checksum = 0
    for _ in range(warmup):
        try:
            next(iterator)
        except StopIteration:
            break
    started = time.perf_counter()
    remaining = max(0, total_batches - warmup)
    for parts in tqdm(
        iterator, total=remaining, desc="unscheduled", unit="batch", disable=args.quiet
    ):
        batches += 1
        parts_seen += len(parts)
        for dataset_idx, local_cells in parts:
            out = bank.load(
                ids[dataset_idx],
                local_cells,
                genes=genes,
                missing=missing,
                dtype=dtype,
            )
            cells += len(local_cells)
            bytes_read += out.data.nbytes
            if out.data.size:
                checksum = (checksum + int(out.data[0])) & 0xFFFFFFFF
    seconds = time.perf_counter() - started
    return _sample(
        mode="unscheduled",
        cells=cells,
        batches=batches,
        parts=parts_seen,
        bytes_read=bytes_read,
        checksum=checksum,
        seconds=seconds,
        warmup_batches=warmup,
        resolved_strategy=None,
        fallback_reason=None,
    )


def bench_scheduled_once(
    bank: ScDataBank,
    ids: list,
    catalog: list[CatalogEntry],
    order: np.ndarray,
    offsets: np.ndarray,
    args: argparse.Namespace,
    warmup: int,
) -> dict:
    if args.gene_mode == "native" and len({entry.n_vars for entry in catalog}) != 1:
        raise SystemExit("scheduled native mode requires identical n_vars; use --gene-mode first")
    genes = resolve_genes(bank, ids, args)
    dtype = resolve_dtype(args)
    missing = "zero" if genes is not None else None
    config = ScheduledPrefetchConfig(
        prefetch_step=args.prefetch_step,
        access=ScheduledAccessConfig(
            prefetch_step=args.access_prefetch_step,
            decode_ahead_steps=args.decode_ahead_steps,
            ready_ahead_steps=args.ready_ahead_steps,
        ),
        projected_sparse_data_strategy=args.projected_sparse_data_strategy,
        fast_mode=args.fast_mode,
    )
    plan = indexed_plan(order, offsets, args.batch_size)
    total_batches = plan.num_batches
    stream = bank.prefetch_indexed(
        ids,
        plan,
        genes=genes,
        missing=missing,
        dtype=dtype,
        config=config,
    )
    # These are fixed at stream construction time; capture before consuming.
    resolved_strategy = getattr(stream, "resolved_strategy", None)
    fallback_reason = getattr(stream, "fallback_reason", None)

    cells = batches = bytes_read = checksum = 0
    for _ in range(warmup):
        try:
            next(stream)
        except StopIteration:
            break
    started = time.perf_counter()
    remaining = max(0, total_batches - warmup)
    for batch in tqdm(stream, total=remaining, desc="scheduled", unit="batch", disable=args.quiet):
        batches += 1
        cells += len(batch.cells)
        bytes_read += batch.data.nbytes
        if batch.data.size:
            checksum = (checksum + int(batch.data[0])) & 0xFFFFFFFF
    seconds = time.perf_counter() - started
    return _sample(
        mode="scheduled",
        cells=cells,
        batches=batches,
        parts=-1,
        bytes_read=bytes_read,
        checksum=checksum,
        seconds=seconds,
        warmup_batches=warmup,
        resolved_strategy=resolved_strategy,
        fallback_reason=fallback_reason,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """JSON fallback: Path -> str, numpy scalars/arrays -> Python primitives."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default)


def machine_info() -> dict:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "scdata_version": scdata_version,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def _stats(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values) if values else 0.0,
        "stdev": statistics.stdev(values) if len(values) >= 2 else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def aggregate_summary(runs: list[dict]) -> dict:
    by_mode: dict[str, list[dict]] = {}
    for run in runs:
        for result in run["results"]:
            by_mode.setdefault(result["mode"], []).append(result)
    summary: dict[str, Any] = {}
    for mode, samples in by_mode.items():
        summary[mode] = {
            "runs": len(samples),
            "cells_per_s": _stats([s["cells_per_s"] for s in samples]),
            "gb_per_s": _stats([s["gb_per_s"] for s in samples]),
            "seconds": _stats([s["seconds"] for s in samples]),
            "resolved_strategy": samples[0]["resolved_strategy"],
            "fallback_reason": samples[0]["fallback_reason"],
        }
    return summary


def run_bench(args: argparse.Namespace) -> dict:
    scan_started = time.perf_counter()
    catalog, skipped = build_catalog(args.root, args.limit_datasets, quiet=args.quiet)
    scan_seconds = time.perf_counter() - scan_started

    counts = np.asarray([entry.n_obs for entry in catalog], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    order = build_order(counts, args.order, args.seed, args.max_cells)

    io_workers, decode_workers, access_workers, fill_workers = resolve_worker_counts(
        threads=args.threads,
        io_workers=args.io_workers,
        decode_workers=args.decode_workers,
        access_workers=args.access_workers,
        fill_workers=args.fill_workers,
    )
    cfg = make_config(
        args.memory_gib,
        io_workers,
        decode_workers,
        access_workers,
        fill_workers,
        args.fast_enabled,
        args.fast_fused_workers,
        args.fast_prefetch_blocks,
        args.fast_load_scheduler_workers or max(fill_workers, args.fast_fused_workers, 1),
        args.fast_load_io_workers or max(io_workers, 1),
        args.fast_coalesce_max_gap_bytes,
        args.fast_coalesce_max_waste_ratio,
        args.fast_coalesce_max_merged_len,
    )

    runs: list[dict] = []
    for repeat in range(args.repeat):
        bank = ScDataBank(cfg)
        try:
            register_started = time.perf_counter()
            ids = register_catalog(bank, catalog, quiet=args.quiet)
            register_seconds = time.perf_counter() - register_started

            results: list[dict] = []
            if args.mode in ("unscheduled", "both"):
                sample = bench_unscheduled_once(bank, ids, order, offsets, args, args.warmup)
                if args.profile:
                    sample["profile"] = bank.profile_snapshot_and_reset()
                results.append(sample)
            if args.mode in ("scheduled", "both"):
                sample = bench_scheduled_once(bank, ids, catalog, order, offsets, args, args.warmup)
                if args.profile:
                    sample["profile"] = bank.profile_snapshot_and_reset()
                results.append(sample)
        finally:
            bank.close()
        runs.append(
            {
                "repeat": repeat,
                "register_seconds": register_seconds,
                "results": results,
            }
        )

    return {
        "meta": {
            **machine_info(),
            "cli_args": vars(args),
        },
        "config": {
            "root": str(args.root),
            "mode": args.mode,
            "datasets": len(catalog),
            "skipped": len(skipped),
            "total_cells": int(counts.sum()),
            "sampled_cells": int(len(order)),
            "order": args.order,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "dtype": args.dtype,
            "gene_mode": args.gene_mode,
            "genes": args.genes,
            "projected_sparse_data_strategy": args.projected_sparse_data_strategy,
            "fast_mode": args.fast_mode,
            "fast_enabled": args.fast_enabled,
            "fast_fused_workers": args.fast_fused_workers,
            "fast_prefetch_blocks": args.fast_prefetch_blocks,
            "fast_load_workers": {
                "scheduler": cfg.fast_config.load.scheduler_workers,
                "io": cfg.fast_config.load.io_workers,
            },
            "fast_coalesce": {
                "max_gap_bytes": args.fast_coalesce_max_gap_bytes,
                "max_waste_ratio": args.fast_coalesce_max_waste_ratio,
                "max_merged_len": args.fast_coalesce_max_merged_len,
            },
            "threads": args.threads,
            "workers": {
                "io": io_workers,
                "decode": decode_workers,
                "access": access_workers,
                "fill": fill_workers,
            },
            "memory_gib": args.memory_gib,
            "prefetch_step": args.prefetch_step,
            "access_prefetch_step": args.access_prefetch_step,
            "decode_ahead_steps": args.decode_ahead_steps,
            "ready_ahead_steps": args.ready_ahead_steps,
            "warmup_batches": args.warmup,
            "repeat": args.repeat,
        },
        "scan_seconds": scan_seconds,
        "runs": runs,
        "summary": aggregate_summary(runs),
    }


# ---------------------------------------------------------------------------
# Output: persistence, summary, diff
# ---------------------------------------------------------------------------


def default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"bench-{stamp}.json"


def write_result(result: dict, output: Path | None) -> Path | None:
    if output is None:
        output = default_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dumps(result))
    return output


def print_summary(result: dict, *, file=None) -> None:
    file = file or sys.stderr
    cfg = result["config"]
    meta = result["meta"]
    print("", file=file)
    print("=" * 72, file=file)
    print(
        f"scdata bench | {meta['hostname']} | {meta['cpu_count']} CPUs | "
        f"scdata {meta['scdata_version']}",
        file=file,
    )
    print(
        f"datasets={cfg['datasets']}  cells={cfg['sampled_cells']:,}  "
        f"order={cfg['order']}  batch={cfg['batch_size']}  "
        f"repeat={cfg['repeat']}  warmup={cfg['warmup_batches']}",
        file=file,
    )
    print(
        f"workers io/decode/access/fill = "
        f"{cfg['workers']['io']}/{cfg['workers']['decode']}/"
        f"{cfg['workers']['access']}/{cfg['workers']['fill']}  "
        f"memory={cfg['memory_gib']}GiB  fast={cfg['fast_enabled']} ({cfg['fast_mode']})",
        file=file,
    )
    print("-" * 72, file=file)
    summary = result.get("summary", {})
    header = f"{'mode':<12} {'cell/s':>14} {'GB/s':>10} {'seconds':>10} {'strategy':<16}"
    print(header, file=file)
    for mode, stats in summary.items():
        cps = stats["cells_per_s"]
        gbs = stats["gb_per_s"]
        sec = stats["seconds"]
        strat = stats.get("resolved_strategy") or "-"
        print(
            f"{mode:<12} {cps['mean']:>14,.0f} {gbs['mean']:>10.2f} {sec['mean']:>10.3f} {strat:<16}",
            file=file,
        )
        if stats.get("runs", 0) > 1:
            print(
                f"{'':<12} {cps['min']:>14,.0f}..{cps['max']:>14,.0f} (stdev {cps['stdev']:.0f})",
                file=file,
            )
        if stats.get("fallback_reason"):
            print(f"{'':<12} fast-path fallback: {stats['fallback_reason']}", file=file)
    print("=" * 72, file=file)


def diff_results(a: dict, b: dict, *, file=None) -> None:
    """Print a throughput delta between two saved result objects (b vs a)."""
    file = file or sys.stderr
    sa = a.get("summary", {})
    sb = b.get("summary", {})
    print("", file=file)
    print("=" * 72, file=file)
    print("diff (B vs A)", file=file)
    print("-" * 72, file=file)
    print(
        f"{'mode':<12} {'A cell/s':>14} {'B cell/s':>14} {'Δ%':>10} {'B GB/s':>10}",
        file=file,
    )
    modes = sorted(set(sa) | set(sb))
    for mode in modes:
        if mode not in sa or mode not in sb:
            print(f"{mode:<12} (missing on one side)", file=file)
            continue
        ca = sa[mode]["cells_per_s"]["mean"]
        cb = sb[mode]["cells_per_s"]["mean"]
        delta = (cb - ca) / ca * 100.0 if ca else 0.0
        gbb = sb[mode]["gb_per_s"]["mean"]
        print(
            f"{mode:<12} {ca:>14,.0f} {cb:>14,.0f} {delta:>+9.1f}% {gbb:>10.2f}",
            file=file,
        )
    print("=" * 72, file=file)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_bench(args: argparse.Namespace) -> int:
    result = run_bench(args)

    output_path: Path | None = None
    if not args.no_output:
        output_path = write_result(result, args.output)
        result["_output"] = str(output_path)

    # Full JSON to stdout (pipe-friendly); human-readable bits to stderr.
    print(dumps(result))

    if args.summary:
        print_summary(result)
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        diff_results(baseline, result)
    if output_path:
        print(f"\nresult written to {output_path}", file=sys.stderr)
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    catalog, skipped = build_catalog(args.root, args.limit_datasets, quiet=args.quiet)
    total_cells = sum(entry.n_obs for entry in catalog)
    payload = {
        "root": str(args.root),
        "total_datasets": len(catalog),
        "skipped": len(skipped),
        "total_cells": total_cells,
        "datasets": [
            {
                "path": str(entry.path),
                "matrix": entry.matrix,
                "dtype": entry.dtype,
                "n_obs": entry.n_obs,
                "n_vars": entry.n_vars,
            }
            for entry in catalog
        ],
        "skipped_details": [{"path": str(p), "reason": r} for p, r in skipped],
    }
    print(dumps(payload))
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    diff_results(a, b)
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def add_bench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--mode", choices=("both", "unscheduled", "scheduled"), default="both")
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--limit-datasets", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--order",
        choices=("random", "sequential"),
        default="random",
        help="cell access order across the concatenated corpus; "
        "'sequential' reads 0,1,2,... (prefetch/cache-friendly control), "
        "'random' shuffles with --seed",
    )
    parser.add_argument(
        "--dtype",
        default="stored",
        help="output dtype; 'stored'/'auto'/'none' keep the stored dtype",
    )
    parser.add_argument("--gene-mode", choices=("first", "native"), default="first")
    parser.add_argument(
        "--projected-sparse-data-strategy",
        "--sparse-data-strategy",
        choices=("selected_only", "read_all"),
        default="selected_only",
    )
    parser.add_argument("--fast-mode", choices=("disabled", "auto", "force"), default="disabled")
    parser.add_argument("--fast-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fast-fused-workers", type=int, default=4)
    parser.add_argument("--fast-prefetch-blocks", type=int, default=4096)
    parser.add_argument(
        "--fast-load-scheduler-workers",
        type=int,
        default=0,
        help="0 means max(resolved --fill-workers, --fast-fused-workers)",
    )
    parser.add_argument(
        "--fast-load-io-workers",
        type=int,
        default=0,
        help="0 means resolved --io-workers",
    )
    parser.add_argument("--fast-coalesce-max-gap-bytes", type=int, default=16 * 1024)
    parser.add_argument("--fast-coalesce-max-waste-ratio", type=float, default=0.10)
    parser.add_argument("--fast-coalesce-max-merged-len", type=int, default=1024 * 1024)
    parser.add_argument(
        "--genes",
        type=int,
        default=0,
        help="0 means all genes from the first dataset",
    )
    parser.add_argument("--prefetch-step", type=int, default=DEFAULT_PREFETCH_STEP)
    parser.add_argument("--access-prefetch-step", type=int, default=DEFAULT_ACCESS_PREFETCH_STEP)
    parser.add_argument("--decode-ahead-steps", type=int, default=DEFAULT_DECODE_AHEAD_STEPS)
    parser.add_argument("--ready-ahead-steps", type=int, default=DEFAULT_READY_AHEAD_STEPS)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--io-workers", type=int, default=0, help="0 means --threads // 4")
    parser.add_argument("--decode-workers", type=int, default=0, help="0 means --threads // 4")
    parser.add_argument("--access-workers", type=int, default=0, help="0 means --threads // 4")
    parser.add_argument("--fill-workers", type=int, default=0, help="0 means --threads // 4")
    parser.add_argument("--memory-gib", type=int, default=128)

    # --- new benchmarking controls ---
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="batches to consume before timing (excludes cold-start / cache fill)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat the run this many times; report mean/stdev/min/max",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="capture bank.profile_snapshot_and_reset() per mode per run",
    )
    parser.add_argument("--output", type=Path, default=None, help="result JSON path")
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="do not write a result JSON file",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="previous result JSON to diff against (printed to stderr)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print a human-readable summary table to stderr",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="disable tqdm progress bars",
    )


def validate_bench_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.max_cells is not None and args.max_cells < 1:
        parser.error("--max-cells must be positive")
    if args.limit_datasets < 0:
        parser.error("--limit-datasets must be non-negative")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.genes < 0:
        parser.error("--genes must be non-negative")
    if args.prefetch_step < 1:
        parser.error("--prefetch-step must be positive")
    if args.access_prefetch_step < 1:
        parser.error("--access-prefetch-step must be positive")
    if args.decode_ahead_steps < 1:
        parser.error("--decode-ahead-steps must be positive")
    if args.ready_ahead_steps < 1:
        parser.error("--ready-ahead-steps must be positive")
    if args.fast_fused_workers < 1:
        parser.error("--fast-fused-workers must be positive")
    if args.fast_prefetch_blocks < 1:
        parser.error("--fast-prefetch-blocks must be positive")
    if args.fast_load_scheduler_workers < 0:
        parser.error("--fast-load-scheduler-workers must be non-negative")
    if args.fast_load_io_workers < 0:
        parser.error("--fast-load-io-workers must be non-negative")
    if args.fast_coalesce_max_gap_bytes < 0:
        parser.error("--fast-coalesce-max-gap-bytes must be non-negative")
    if not 0 <= args.fast_coalesce_max_waste_ratio <= 1:
        parser.error("--fast-coalesce-max-waste-ratio must be in [0, 1]")
    if args.fast_coalesce_max_merged_len < 1:
        parser.error("--fast-coalesce-max-merged-len must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.no_output and args.output is not None:
        parser.error("--no-output and --output are mutually exclusive")
    try:
        resolve_worker_counts(
            threads=args.threads,
            io_workers=args.io_workers,
            decode_workers=args.decode_workers,
            access_workers=args.access_workers,
            fill_workers=args.fill_workers,
        )
    except ValueError as err:
        parser.error(str(err))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench_access",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    p_bench = sub.add_parser(
        "bench",
        help="run the random-access benchmark (default)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_bench_args(p_bench)

    p_scan = sub.add_parser(
        "scan",
        help="list available .zarr.zip datasets under a root",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_scan.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p_scan.add_argument("--limit-datasets", type=int, default=0)
    p_scan.add_argument("--quiet", action="store_true", help="disable tqdm progress bars")

    p_diff = sub.add_parser(
        "diff",
        help="compare two result JSON files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_diff.add_argument("a", type=Path, help="baseline result JSON (A)")
    p_diff.add_argument("b", type=Path, help="new result JSON (B)")

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)

    # Back-compat: bare invocation (no subcommand) defaults to `bench`.
    known_subcommands = {"bench", "scan", "diff"}
    help_flags = {"-h", "--help"}
    if argv and argv[0] not in known_subcommands and argv[0] not in help_flags:
        argv = ["bench", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "bench":
        validate_bench_args(parser, args)
        return cmd_bench(args)
    if args.command == "scan":
        return cmd_scan(args)
    if args.command == "diff":
        return cmd_diff(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
