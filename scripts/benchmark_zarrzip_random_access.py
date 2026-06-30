#!/usr/bin/env python3
"""Benchmark random cell access for scdata zarrzip stores."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scdata import DataBankConfig, ScDataBank  # noqa: E402
from scdata.io import launch  # noqa: E402


DEFAULT_ROOT = Path(
    "/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/FFPE/"
    "20260625_dataset_zarrzip/20260625_M20_collected/high_quality"
)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    started = time.perf_counter()
    paths = discover_paths(args.root, args.limit_files, rng)
    if not paths:
        raise SystemExit(f"no .zarr.zip files found under {args.root}")

    print(f"files={len(paths)} root={args.root}", flush=True)
    for path in paths[: min(args.print_files, len(paths))]:
        print(f"FILE\t{path}", flush=True)
    if len(paths) > args.print_files:
        print(f"FILE\t... {len(paths) - args.print_files} more", flush=True)

    cfg = DataBankConfig.make(
        cache_capacity_bytes=args.cache_gib * 1024**3,
        memory_budget_bytes=args.memory_gib * 1024**3,
    )

    launch_started = time.perf_counter()
    datasets = []
    for path in paths:
        datasets.append((path, launch(path)))
    launch_seconds = time.perf_counter() - launch_started

    register_started = time.perf_counter()
    bank = ScDataBank(cfg)
    entries = []
    for path, ds in datasets:
        did = bank.register(ds)
        entries.append(
            {
                "path": path,
                "id": did,
                "n_obs": int(bank.dataset_num_cells(did)),
                "n_vars": int(bank.dataset_num_genes(did)),
                "dtype": str(bank.dataset_dtype(did).value),
            }
        )
    register_seconds = time.perf_counter() - register_started
    ready_seconds = time.perf_counter() - started

    print(
        f"registered={len(entries)} launch_s={launch_seconds:.3f} "
        f"register_s={register_seconds:.3f} ready_s={ready_seconds:.3f}",
        flush=True,
    )

    warmup = run_trials(
        bank=bank,
        entries=entries,
        iterations=args.warmup,
        batch_size=args.batch_size,
        np_rng=np_rng,
        dtype=args.dtype,
    )
    if args.warmup:
        print("warmup", format_summary(warmup), flush=True)

    measured = run_trials(
        bank=bank,
        entries=entries,
        iterations=args.iterations,
        batch_size=args.batch_size,
        np_rng=np_rng,
        dtype=args.dtype,
    )
    total_seconds = time.perf_counter() - started
    summary = summarize(measured)
    summary.update(
        {
            "root": str(args.root),
            "files": len(paths),
            "batch_size": args.batch_size,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "seed": args.seed,
            "cache_gib": args.cache_gib,
            "memory_gib": args.memory_gib,
            "launch_seconds": launch_seconds,
            "register_seconds": register_seconds,
            "ready_seconds": ready_seconds,
            "total_seconds": total_seconds,
            "end_to_end_seconds_per_request_including_init": total_seconds
            / max(args.iterations, 1),
            "selected_files": [
                {
                    "path": str(entry["path"]),
                    "n_obs": entry["n_obs"],
                    "n_vars": entry["n_vars"],
                    "dtype": entry["dtype"],
                }
                for entry in entries
            ],
        }
    )

    print("measured", format_summary(measured), flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    bank.close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--limit-files", type=int, default=32)
    parser.add_argument("--print-files", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--cache-gib", type=int, default=32)
    parser.add_argument("--memory-gib", type=int, default=128)
    parser.add_argument(
        "--dtype",
        help="Optional access dtype such as f32/u16. Omit to use stored dtype.",
    )
    args = parser.parse_args()
    if args.limit_files < 0:
        parser.error("--limit-files must be non-negative")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.cache_gib < 0:
        parser.error("--cache-gib must be non-negative")
    if args.memory_gib < 1:
        parser.error("--memory-gib must be >= 1")
    return args


def discover_paths(root: Path, limit: int, rng: random.Random) -> list[Path]:
    paths = sorted(path for path in root.rglob("*.zarr.zip") if path.is_file())
    if limit and len(paths) > limit:
        paths = rng.sample(paths, limit)
        paths.sort()
    return paths


def run_trials(
    *,
    bank: ScDataBank,
    entries: list[dict[str, Any]],
    iterations: int,
    batch_size: int,
    np_rng: np.random.Generator,
    dtype: str | None,
) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for i in range(iterations):
        entry = entries[i % len(entries)]
        n_obs = int(entry["n_obs"])
        n_vars = int(entry["n_vars"])
        cells = np_rng.integers(0, n_obs, size=batch_size, dtype=np.intp)

        started = time.perf_counter()
        result = bank.load(entry["id"], cells, dtype=dtype)
        elapsed = time.perf_counter() - started

        data = result.to_flat_numpy()
        # Touch the result so benchmark timing includes Python-visible materialization.
        checksum = int(data[0]) if data.size else 0
        trials.append(
            {
                "seconds": elapsed,
                "cells": int(batch_size),
                "values": int(batch_size * n_vars),
                "bytes": int(data.nbytes),
                "checksum": checksum,
                "file": str(entry["path"]),
            }
        )
    return trials


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    seconds = [float(t["seconds"]) for t in trials]
    total_s = sum(seconds)
    total_cells = sum(int(t["cells"]) for t in trials)
    total_values = sum(int(t["values"]) for t in trials)
    total_bytes = sum(int(t["bytes"]) for t in trials)
    return {
        "requests": len(trials),
        "latency_mean_ms": statistics.fmean(seconds) * 1000,
        "latency_median_ms": percentile(seconds, 50) * 1000,
        "latency_p95_ms": percentile(seconds, 95) * 1000,
        "latency_min_ms": min(seconds) * 1000,
        "latency_max_ms": max(seconds) * 1000,
        "measured_seconds": total_s,
        "cells_per_second": total_cells / total_s if total_s else 0.0,
        "values_per_second": total_values / total_s if total_s else 0.0,
        "materialized_gib_per_second": (total_bytes / 1024**3) / total_s if total_s else 0.0,
        "total_cells": total_cells,
        "total_values": total_values,
        "total_materialized_gib": total_bytes / 1024**3,
    }


def format_summary(trials: list[dict[str, Any]]) -> str:
    s = summarize(trials)
    return (
        f"requests={s['requests']} mean={s['latency_mean_ms']:.3f}ms "
        f"median={s['latency_median_ms']:.3f}ms p95={s['latency_p95_ms']:.3f}ms "
        f"cells/s={s['cells_per_second']:.1f} "
        f"values/s={s['values_per_second']:.1f} "
        f"GiB/s={s['materialized_gib_per_second']:.3f}"
    )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


if __name__ == "__main__":
    raise SystemExit(main())
