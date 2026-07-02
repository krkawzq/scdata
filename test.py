from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = lambda x, **_: x


DEFAULT_ROOT = Path(
    os.environ.get("mntwzq", "/mnt/shared-storage-user/dnacoding/wangzhongqi")
) / "Data/cellxgene/homo_spacian"
if not DEFAULT_ROOT.exists():
    DEFAULT_ROOT = DEFAULT_ROOT.with_name("Homo_sapiens")

ROWCOUNT_DTYPES = {DType.U16, DType.U32}
DEFAULT_PREFETCH_STEP = 32
DEFAULT_ACCESS_PREFETCH_STEP = 64
DEFAULT_DECODE_AHEAD_STEPS = 32
DEFAULT_READY_AHEAD_STEPS = 16


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
    return counts


def make_config(
    memory_gib: int,
    io_workers: int,
    decode_workers: int,
    access_workers: int,
    fill_workers: int,
    fast_enabled: bool,
    fast_fused_workers: int,
    fast_prefetch_blocks: int,
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
        fast__load__coalesce__max_gap_bytes=fast_coalesce_max_gap_bytes,
        fast__load__coalesce__max_waste_ratio=fast_coalesce_max_waste_ratio,
        fast__load__coalesce__max_merged_len=fast_coalesce_max_merged_len,
    )


def pick_rowcount(path: Path):
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
) -> tuple[list[CatalogEntry], list[tuple[Path, str]]]:
    paths = sorted(root.rglob("*.zarr.zip"))
    if limit_datasets:
        paths = paths[:limit_datasets]
    if not paths:
        raise SystemExit(f"no .zarr.zip found under {root}")

    catalog: list[CatalogEntry] = []
    skipped: list[tuple[Path, str]] = []
    for path in tqdm(paths, desc="scan", unit="dataset"):
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
        except Exception as err:
            skipped.append((path, repr(err)))

    if not catalog:
        raise SystemExit(f"no rowcount datasets found under {root}")
    return catalog, skipped


def register_catalog(bank: ScDataBank, catalog: list[CatalogEntry]) -> list:
    ids = []
    for entry in tqdm(catalog, desc="register", unit="dataset"):
        ids.append(bank.register(entry.dataset))
    return ids


def flat_shuffle(counts: np.ndarray, seed: int, max_cells: int | None) -> np.ndarray:
    total = int(counts.sum())
    n = total if max_cells is None else min(int(max_cells), total)
    rng = np.random.default_rng(seed)
    if n == total:
        order = np.arange(total, dtype=np.int64)
        rng.shuffle(order)
        return order
    return rng.choice(total, size=n, replace=False).astype(np.int64, copy=False)


def batch_parts(order: np.ndarray, offsets: np.ndarray) -> list[tuple[int, np.ndarray]]:
    dataset_idx = np.searchsorted(offsets[1:], order, side="right")
    local_cells = order - offsets[dataset_idx]
    parts: list[tuple[int, np.ndarray]] = []
    start = 0
    while start < len(order):
        dataset = int(dataset_idx[start])
        end = start + 1
        while end < len(order) and int(dataset_idx[end]) == dataset:
            end += 1
        parts.append((dataset, local_cells[start:end].astype(np.intp, copy=False)))
        start = end
    return parts


def iter_batches(order: np.ndarray, offsets: np.ndarray, batch_size: int):
    for start in range(0, len(order), batch_size):
        yield batch_parts(order[start : start + batch_size], offsets)


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
    if args.dtype in ("stored", "native", "none"):
        return None
    return args.dtype


def summarize(
    *,
    mode: str,
    cells: int,
    batches: int,
    parts: int,
    bytes_read: int,
    checksum: int,
    seconds: float,
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
    }


def bench_unscheduled(
    bank: ScDataBank,
    ids: list,
    order: np.ndarray,
    offsets: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    genes = resolve_genes(bank, ids, args)
    dtype = resolve_dtype(args)
    missing = "zero" if genes is not None else None
    total_batches = (len(order) + args.batch_size - 1) // args.batch_size
    cells = batches = parts_seen = bytes_read = checksum = 0
    started = time.perf_counter()
    iterator = iter_batches(order, offsets, args.batch_size)
    for parts in tqdm(iterator, total=total_batches, desc="unscheduled", unit="batch"):
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
    return summarize(
        mode="unscheduled",
        cells=cells,
        batches=batches,
        parts=parts_seen,
        bytes_read=bytes_read,
        checksum=checksum,
        seconds=seconds,
    )


def bench_scheduled(
    bank: ScDataBank,
    ids: list,
    catalog: list[CatalogEntry],
    order: np.ndarray,
    offsets: np.ndarray,
    args: argparse.Namespace,
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
    total_batches = (len(order) + args.batch_size - 1) // args.batch_size
    cells = batches = bytes_read = checksum = 0
    started = time.perf_counter()
    plan = indexed_plan(order, offsets, args.batch_size)
    stream = bank.prefetch_indexed(
        ids,
        plan,
        genes=genes,
        missing=missing,
        dtype=dtype,
        config=config,
    )
    for batch in tqdm(stream, total=total_batches, desc="scheduled", unit="batch"):
        batches += 1
        cells += len(batch.cells)
        bytes_read += batch.data.nbytes
        if batch.data.size:
            checksum = (checksum + int(batch.data[0])) & 0xFFFFFFFF
    seconds = time.perf_counter() - started
    return summarize(
        mode="scheduled",
        cells=cells,
        batches=batches,
        parts=-1,
        bytes_read=bytes_read,
        checksum=checksum,
        seconds=seconds,
    )


def run_once(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    catalog, skipped = build_catalog(args.root, args.limit_datasets)
    scan_seconds = time.perf_counter() - started
    counts = np.asarray([entry.n_obs for entry in catalog], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    order = flat_shuffle(counts, args.seed, args.max_cells)

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
        args.fast_coalesce_max_gap_bytes,
        args.fast_coalesce_max_waste_ratio,
        args.fast_coalesce_max_merged_len,
    )
    bank = ScDataBank(cfg)
    try:
        register_started = time.perf_counter()
        ids = register_catalog(bank, catalog)
        register_seconds = time.perf_counter() - register_started
        results = []
        if args.mode in ("unscheduled", "both"):
            results.append(bench_unscheduled(bank, ids, order, offsets, args))
        if args.mode in ("scheduled", "both"):
            results.append(bench_scheduled(bank, ids, catalog, order, offsets, args))
    finally:
        bank.close()

    return {
        "root": str(args.root),
        "mode": args.mode,
        "datasets": len(catalog),
        "skipped": len(skipped),
        "total_cells": int(counts.sum()),
        "sampled_cells": int(len(order)),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "dtype": args.dtype,
        "gene_mode": args.gene_mode,
        "projected_sparse_data_strategy": args.projected_sparse_data_strategy,
        "fast_mode": args.fast_mode,
        "fast_enabled": args.fast_enabled,
        "fast_fused_workers": args.fast_fused_workers,
        "fast_prefetch_blocks": args.fast_prefetch_blocks,
        "fast_coalesce": {
            "max_gap_bytes": args.fast_coalesce_max_gap_bytes,
            "max_waste_ratio": args.fast_coalesce_max_waste_ratio,
            "max_merged_len": args.fast_coalesce_max_merged_len,
        },
        "genes": args.genes,
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
        "scan_seconds": scan_seconds,
        "register_seconds": register_seconds,
        "results": results,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--mode", choices=("both", "unscheduled", "scheduled"), default="both")
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--limit-datasets", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="stored")
    parser.add_argument("--gene-mode", choices=("first", "native"), default="first")
    parser.add_argument(
        "--projected-sparse-data-strategy",
        "--sparse-data-strategy",
        choices=("selected_only", "read_all"),
        default="selected_only",
    )
    parser.add_argument(
        "--fast-mode",
        choices=("disabled", "auto", "force"),
        default="disabled",
    )
    parser.add_argument(
        "--fast-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--fast-fused-workers", type=int, default=4)
    parser.add_argument("--fast-prefetch-blocks", type=int, default=4096)
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
    parser.add_argument(
        "--access-prefetch-step",
        type=int,
        default=DEFAULT_ACCESS_PREFETCH_STEP,
    )
    parser.add_argument(
        "--decode-ahead-steps",
        type=int,
        default=DEFAULT_DECODE_AHEAD_STEPS,
    )
    parser.add_argument(
        "--ready-ahead-steps",
        type=int,
        default=DEFAULT_READY_AHEAD_STEPS,
    )
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument(
        "--io-workers",
        type=int,
        default=0,
        help="0 means --threads // 4",
    )
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=0,
        help="0 means --threads // 4",
    )
    parser.add_argument(
        "--access-workers",
        type=int,
        default=0,
        help="0 means --threads // 4",
    )
    parser.add_argument(
        "--fill-workers",
        type=int,
        default=0,
        help="0 means --threads // 4",
    )
    parser.add_argument("--memory-gib", type=int, default=128)
    args = parser.parse_args(argv)
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
    if args.fast_coalesce_max_gap_bytes < 0:
        parser.error("--fast-coalesce-max-gap-bytes must be non-negative")
    if not 0 <= args.fast_coalesce_max_waste_ratio <= 1:
        parser.error("--fast-coalesce-max-waste-ratio must be in [0, 1]")
    if args.fast_coalesce_max_merged_len < 1:
        parser.error("--fast-coalesce-max-merged-len must be positive")
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
    return args


def main(argv: Iterable[str] | None = None) -> None:
    result = run_once(parse_args(argv))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
