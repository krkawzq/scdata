from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import statistics
import tempfile
import time

import anndata as ad
import numpy as np
import pandas as pd
from scdata.anndata import write_scc

import scdata.load as sc_load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark distributed shared-ring iteration")
    parser.add_argument("--rows", type=int, default=8_192)
    parser.add_argument("--columns", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prefetch-step", type=int, default=64)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--zero-copy", action="store_true")
    parser.add_argument("--legacy-copy", action="store_true")
    parser.add_argument("--read", action="store_true")
    parser.add_argument("--legacy-read", action="store_true")
    parser.add_argument("--label", default="current")
    return parser.parse_args()


def make_plan(root: Path, args: argparse.Namespace) -> sc_load.Plan:
    values = np.arange(args.rows * args.columns, dtype=np.float32).reshape(
        args.rows,
        args.columns,
    )
    matrix = ad.AnnData(
        X=values,
        obs=pd.DataFrame(index=[f"c{row}" for row in range(args.rows)]),
        var=pd.DataFrame(index=[f"g{column}" for column in range(args.columns)]),
    )
    path = write_scc(matrix, root / "distributed-benchmark.scc", store="dir")
    dataset = sc_load.register(path)
    return sc_load.compile(
        dataset,
        range(args.rows),
        batch_size=args.batch_size,
        prefetch_step=args.prefetch_step,
    )


def consume(
    iterator: sc_load.DistributedIterator,
    *,
    legacy_copy: bool,
) -> tuple[int, float]:
    rows = 0
    checksum = 0.0
    while True:
        batch = iterator.next_batch()
        if batch is None:
            return rows, checksum
        if legacy_copy:
            owned = np.array(batch, dtype=iterator.dtype, order="C", copy=True, subok=False)
            del batch
            batch = owned
        rows += batch.shape[0]
        if batch.size:
            checksum += float(batch[0, 0])
        del batch


def consume_read(iterator: sc_load.DistributedIterator) -> tuple[int, float]:
    values = iterator.read()
    checksum = float(values[0, 0]) if values.size else 0.0
    return values.shape[0], checksum


def consume_legacy_read(iterator: sc_load.DistributedIterator) -> tuple[int, float]:
    values = np.empty(iterator.shape, dtype=iterator.dtype)
    offset = 0
    while True:
        batch = iterator.next_batch(copy=False)
        if batch is None:
            break
        stop = offset + batch.shape[0]
        values[offset:stop] = batch
        offset = stop
        del batch
    if offset != values.shape[0]:
        raise RuntimeError("legacy distributed read returned the wrong row count")
    checksum = float(values[0, 0]) if values.size else 0.0
    return values.shape[0], checksum


def run_once(
    plan: sc_load.Plan,
    args: argparse.Namespace,
    executor: ThreadPoolExecutor,
) -> None:
    config = sc_load.SessionConfig(worker_count=args.world_size, io_mode="blocking")
    with plan.open_distributed(args.world_size, config) as distributed:
        iterators = distributed.ranks(
            copy=not (args.zero_copy or args.legacy_copy or args.legacy_read)
        )
        if args.read:
            consumer = consume_read
        elif args.legacy_read:
            consumer = consume_legacy_read
        else:
            consumer = partial(consume, legacy_copy=args.legacy_copy)
        results = list(executor.map(consumer, iterators))
        distributed.wait()
    if sum(rows for rows, _ in results) != args.rows:
        raise RuntimeError("distributed benchmark returned the wrong row count")


def main() -> None:
    args = parse_args()
    for name in (
        "rows",
        "columns",
        "batch_size",
        "prefetch_step",
        "world_size",
        "iterations",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.zero_copy and args.legacy_copy:
        raise ValueError("--zero-copy and --legacy-copy are mutually exclusive")
    if args.read and (args.zero_copy or args.legacy_copy):
        raise ValueError("--read cannot be combined with copy-mode overrides")
    if args.legacy_read and (args.read or args.zero_copy or args.legacy_copy):
        raise ValueError("--legacy-read cannot be combined with another mode")

    with tempfile.TemporaryDirectory(prefix="sc-load-distributed-benchmark-") as temporary:
        plan = make_plan(Path(temporary), args)
        samples: list[float] = []
        with ThreadPoolExecutor(max_workers=args.world_size) as executor:
            for _ in range(args.warmup):
                run_once(plan, args, executor)
            for _ in range(args.iterations):
                started = time.perf_counter()
                run_once(plan, args, executor)
                samples.append(time.perf_counter() - started)

    samples.sort()
    median = statistics.median(samples)
    p95 = samples[min(len(samples) - 1, int(len(samples) * 0.95))]
    batch_count = (args.rows + args.batch_size - 1) // args.batch_size
    payload_bytes = args.rows * args.columns * np.dtype(np.float32).itemsize
    print(
        f"label={args.label} copy={not args.zero_copy} legacy_copy={args.legacy_copy} "
        f"read={args.read} legacy_read={args.legacy_read} "
        f"rows={args.rows} columns={args.columns} "
        f"batches={batch_count} world_size={args.world_size} iterations={args.iterations}"
    )
    print(
        f"median={median * 1_000:.3f} ms p95={p95 * 1_000:.3f} ms "
        f"throughput={payload_bytes / median / (1024**3):.3f} GiB/s"
    )


if __name__ == "__main__":
    main()
