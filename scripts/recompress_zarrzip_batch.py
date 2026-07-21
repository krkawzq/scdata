#!/usr/bin/env python3
"""Batch re-compress the Blosc ``blocksize`` of every ``.zarr.zip`` under a root.

This wraps :mod:`recompress_zarrzip_blocksize` (which handles a single store)
with:

  * recursive ``.zarr.zip`` discovery,
  * a complete metadata pre-check across every Blosc array before skipping an
    existing destination,
  * ``ProcessPoolExecutor`` parallelism over files (the recommended granularity
    per the single-file script's module docstring),
  * TSV logs (converted / skipped / failed) for idempotent reruns,
  * atomic per-file output — ``convert_store`` writes a sibling ``.tmp`` and
    ``os.replace``-s it onto the destination only after success.

Usage
-----
    uv run python scripts/recompress_zarrzip_batch.py \
        --input-root  .../20260625_dataset_zarrzip \
        --output-root .../20260625_dataset_bs64k \
        --blocksize 65536 --jobs 48 [--limit N] [--overwrite] [--dry-run]

Pre-check contract
------------------
``inspect_store`` reads every ZIP entry (including CRC validation) and every
array ``zarr.json``.  An existing destination is skipped only when its entry
set and Blosc-array set match the source and every destination Blosc array is
at the target.  A missing destination is always materialized, even when the
source already uses the target block size, so output-root workflows never
silently leave a requested store absent.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal

# Blosc encode/decode releases the GIL and is CPU-bound.  We parallelize over
# *files* with ProcessPoolExecutor; each worker handles one store sequentially
# (streaming chunks) so peak memory stays ~one chunk.  Pin BLAS / Blosc to one
# thread per worker so 48 processes don't fan out into 48*N threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("BLOSC_NTHREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_INPUT_ROOT = Path(
    "/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/FFPE/20260625_dataset_zarrzip"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/FFPE/20260625_dataset_bs64k"
)
DEFAULT_BLOCKSIZE = 64 * 1024


@dataclass(frozen=True)
class BatchTask:
    src: str
    dst: str


@dataclass(frozen=True)
class BatchResult:
    status: Literal["converted", "skipped", "failed", "dry_run"]
    src: str
    dst: str
    seconds: float = 0.0
    message: str = ""
    traceback: str = ""


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    sources = discover_zarr_zips(input_root)
    sources = apply_selection(
        sources,
        start=args.start,
        limit=args.limit,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    tasks = build_tasks(sources, input_root=input_root, output_root=output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[{timestamp()}] tasks={len(tasks)} input_root={input_root} "
        f"output_root={output_root} jobs={args.jobs} blocksize={args.blocksize} "
        f"overwrite={args.overwrite}",
        flush=True,
    )

    if args.dry_run:
        for task in tasks[: args.print_limit]:
            print(f"DRY_RUN\t{task.src}\t{task.dst}")
        append_results(log_dir / "dry_run.tsv", (dry_run_result(t) for t in tasks))
        print(f"[{timestamp()}] dry-run listed {min(len(tasks), args.print_limit)} tasks")
        return 0

    common = {
        "blocksize": args.blocksize,
        "overwrite": args.overwrite,
    }

    counts = {"converted": 0, "skipped": 0, "failed": 0}
    if args.jobs == 1:
        result_iter: Iterable[BatchResult] = (
            recompress_one(task, **common) for task in tasks
        )
    else:
        result_iter = iter_parallel_results(tasks, jobs=args.jobs, common=common)
    for result in result_iter:
        print_result(result)
        append_result(log_dir, result)
        if result.status in counts:
            counts[result.status] += 1

    print(
        f"[{timestamp()}] done converted={counts['converted']} skipped={counts['skipped']} "
        f"failed={counts['failed']} logs={log_dir}",
        flush=True,
    )
    return 1 if counts["failed"] and args.fail_on_error else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--blocksize",
        type=int,
        default=DEFAULT_BLOCKSIZE,
        help="Target Blosc block size in bytes (default: 65536 = 64 KiB).",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent processes.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    parser.add_argument("--start", type=int, default=0, help="Skip the first N files.")
    parser.add_argument("--limit", type=int, help="Process at most N files.")
    parser.add_argument(
        "--num-shards", type=int, default=1, help="Total shard count for manual/rjob sharding."
    )
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index to run.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-limit", type=int, default=20)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    if args.blocksize < 0:
        parser.error("--blocksize must be non-negative")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.start < 0:
        parser.error("--start must be non-negative")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must satisfy 0 <= shard-index < num-shards")
    return args


# ---------------------------------------------------------------------------
# discovery & task building
# ---------------------------------------------------------------------------


def discover_zarr_zips(input_root: Path) -> list[Path]:
    return sorted(input_root.rglob("*.zarr.zip"))


def apply_selection(
    sources: list[Path],
    *,
    start: int,
    limit: int | None,
    num_shards: int,
    shard_index: int,
) -> list[Path]:
    selected = sources[start:]
    if limit is not None:
        selected = selected[:limit]
    if num_shards > 1:
        selected = [p for i, p in enumerate(selected) if i % num_shards == shard_index]
    return selected


def build_tasks(
    sources: list[Path],
    *,
    input_root: Path,
    output_root: Path,
) -> list[BatchTask]:
    tasks: list[BatchTask] = []
    for src in sources:
        try:
            rel = src.relative_to(input_root)
        except ValueError:
            rel = Path(src.name)
        dst = output_root / rel
        tasks.append(BatchTask(src=str(src), dst=str(dst)))
    return tasks


# ---------------------------------------------------------------------------
# per-file worker
# ---------------------------------------------------------------------------


def _destination_is_complete_at_target(
    source_layout: Any, destination_layout: Any, target_blocksize: int
) -> bool:
    """Whether a validated destination has the same store and target Blosc set."""
    if source_layout.names != destination_layout.names:
        return False
    source_arrays = source_layout.blosc_blocksizes
    destination_arrays = destination_layout.blosc_blocksizes
    if source_arrays.keys() != destination_arrays.keys():
        return False
    return all(size == target_blocksize for size in destination_arrays.values())


def recompress_one(
    task: BatchTask,
    *,
    blocksize: int,
    overwrite: bool,
) -> BatchResult:
    """Re-compress a single ``.zarr.zip`` with the target blocksize."""
    started = time.perf_counter()
    src = Path(task.src)
    dst = Path(task.dst)
    # Task paths normally originate from resolved roots, but preserve correct
    # in-place behavior for equivalent relative paths or symlink spellings too.
    in_place = src.resolve() == dst.resolve()

    try:
        # ``inspect_store`` consumes every ZIP entry (including CRC checks) and
        # validates each Blosc pipeline.  A pre-existing target is skipped only
        # if *it*, rather than the source, is complete and at the target.
        from recompress_zarrzip_blocksize import convert_store, inspect_store

        source_layout = inspect_store(src)
        if dst.exists() and not overwrite:
            destination_layout = source_layout if in_place else inspect_store(dst)
            if _destination_is_complete_at_target(
                source_layout, destination_layout, blocksize
            ):
                arrays = destination_layout.blosc_blocksizes
                message = (
                    f"all {len(arrays)} destination Blosc arrays already at "
                    f"blocksize={blocksize}"
                    if arrays
                    else "complete destination has no Blosc arrays"
                )
                return BatchResult(
                    status="skipped",
                    src=task.src,
                    dst=task.dst,
                    seconds=time.perf_counter() - started,
                    message=message,
                )
            if not in_place:
                raise FileExistsError(
                    f"output exists but is incomplete, has a different entry/Blosc-array "
                    f"set, or is not at blocksize={blocksize}: {dst} "
                    "(pass --overwrite to replace)"
                )

        dst.parent.mkdir(parents=True, exist_ok=True)
        # If dst is absent, convert_store still materializes a complete copy even
        # when every source array already uses the target.  In-place work is safe
        # because convert_store writes and verifies a sibling temp ZIP first.
        stats = convert_store(
            src,
            dst,
            target_blocksize=blocksize,
            overwrite=overwrite or in_place,
            verify=True,
        )
        return BatchResult(
            status="converted",
            src=task.src,
            dst=task.dst,
            seconds=time.perf_counter() - started,
            message=(
                f"arrays={stats.blosc_arrays} chunks={stats.blosc_chunks} "
                f"copied={stats.copied_entries} "
                f"{stats.recompressed_bytes_in} -> {stats.recompressed_bytes_out} bytes"
            ),
        )
    except Exception as err:
        return BatchResult(
            status="failed",
            src=task.src,
            dst=task.dst,
            seconds=time.perf_counter() - started,
            message=f"{type(err).__name__}: {err}",
            traceback=traceback.format_exc(),
        )


# ---------------------------------------------------------------------------
# parallel runner
# ---------------------------------------------------------------------------


def iter_parallel_results(
    tasks: list[BatchTask],
    *,
    jobs: int,
    common: dict[str, Any],
) -> Iterable[BatchResult]:
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        future_to_task = {
            pool.submit(recompress_one, task, **common): task for task in tasks
        }
        for future in as_completed(future_to_task):
            yield future.result()


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


def append_result(log_dir: Path, result: BatchResult) -> None:
    append_results(log_dir / f"{result.status}.tsv", [result])


def append_results(path: Path, rows: Iterable[BatchResult]) -> None:
    rows = list(rows)
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=result_fieldnames(), delimiter="\t")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in result_fieldnames()})


def result_fieldnames() -> list[str]:
    return ["status", "src", "dst", "seconds", "message", "traceback"]


def dry_run_result(task: BatchTask) -> BatchResult:
    return BatchResult(status="dry_run", src=task.src, dst=task.dst)


def print_result(result: BatchResult) -> None:
    name = Path(result.src).name
    print(
        f"[{timestamp()}] {result.status}\t{name}\t{result.message}",
        flush=True,
    )


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
