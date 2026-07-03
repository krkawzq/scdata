#!/usr/bin/env python3
"""Batch-rewrite Blosc ``blocksize`` across many scdata ``.zarr.zip`` stores.

Wraps :mod:`recompress_zarrzip_blocksize` over a directory tree of stores:
each store is converted in place (temp file + ``os.replace``), verified with
``scdata.io.launch`` + per-chunk decode, and skipped if already at the target
blocksize — so an interrupted run can be re-invoked to pick up where it left off.

Designed to run on a PJLab worker (96 CPU, shared-storage IO).  Parallelism is
process-level (``ProcessPoolExecutor``); each worker holds one source zip open
and streams chunks, so peak memory is ~one chunk per worker.

Usage
-----
    python scripts/batch_recompress_blocksize.py \\
        /mnt/.../cellxgene/Homo_sapiens \\
        --blocksize 65536 --jobs 8 --log-dir ./recompress-logs

Progress is printed per store and a TSV log (converted/skipped/failed) is
appended under ``--log-dir`` for auditing and resume.
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
from pathlib import Path
from typing import Iterable, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import recompress_zarrzip_blocksize as rc  # noqa: E402

Status = Literal["converted", "skipped", "failed"]


@dataclass(frozen=True)
class Task:
    zip_path: str
    dataset_id: str


@dataclass(frozen=True)
class Result:
    status: Status
    zip_path: str
    dataset_id: str
    size_bytes: int
    seconds: float
    blosc_arrays: int
    blosc_chunks: int
    message: str
    traceback: str = ""


def discover(root: Path) -> list[Task]:
    tasks: list[Task] = []
    for p in sorted(root.rglob("full.zarr.zip")):
        tasks.append(Task(zip_path=str(p), dataset_id=p.parent.name))
    return tasks


def process_one(task: Task, *, blocksize: int, verify: bool) -> Result:
    """Convert one store in place.  Idempotent: skips if already at target."""
    started = time.perf_counter()
    path = Path(task.zip_path)
    try:
        size = path.stat().st_size
    except OSError as err:
        return Result("failed", task.zip_path, task.dataset_id, 0,
                      time.perf_counter() - started, 0, 0,
                      f"stat failed: {err}", traceback.format_exc())

    # Skip check: if every blosc array is already at the target blocksize,
    # there is nothing to do.  This makes the run resumable.
    try:
        already = _all_at_target(path, blocksize)
    except Exception as err:
        return Result("failed", task.zip_path, task.dataset_id, size,
                      time.perf_counter() - started, 0, 0,
                      f"inspect failed: {err}", traceback.format_exc())
    if already:
        return Result("skipped", task.zip_path, task.dataset_id, size,
                      time.perf_counter() - started, 0, 0,
                      "already at target blocksize")

    try:
        stats = rc.convert_store(
            path,
            path,  # in place: temp file + os.replace
            target_blocksize=blocksize,
            overwrite=True,
            verify=verify,
        )
        return Result(
            "converted", task.zip_path, task.dataset_id, size,
            time.perf_counter() - started,
            stats.blosc_arrays, stats.blosc_chunks, "ok",
        )
    except Exception as err:
        return Result("failed", task.zip_path, task.dataset_id, size,
                      time.perf_counter() - started, 0, 0,
                      f"{type(err).__name__}: {err}", traceback.format_exc())


def _all_at_target(path: Path, target: int) -> bool:
    """True if the store has no blosc array needing re-compression."""
    import json
    import zipfile

    if not path.is_file():
        return False
    with zipfile.ZipFile(path) as z:
        for key in z.namelist():
            if not key.endswith("zarr.json"):
                continue
            meta = json.loads(z.read(key))
            if meta.get("node_type") != "array":
                continue
            found = rc._find_blosc_codec(meta)
            if found is None:
                continue
            _, cfg = found
            if rc._needs_recompress(cfg, target):
                return False
    return True


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"error: root not a directory: {root}", file=sys.stderr)
        return 2

    tasks = discover(root)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    verify = not args.no_verify

    print(
        f"[{ts()}] root={root} tasks={len(tasks)} jobs={args.jobs} "
        f"blocksize={args.blocksize} verify={verify}",
        flush=True,
    )

    counts = {"converted": 0, "skipped": 0, "failed": 0}
    bytes_done = 0
    bytes_total = sum(
        Path(t.zip_path).stat().st_size for t in tasks if Path(t.zip_path).exists()
    )

    def iter_results() -> Iterable[Result]:
        if args.jobs <= 1:
            for t in tasks:
                yield process_one(t, blocksize=args.blocksize, verify=verify)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futures = {
                    pool.submit(process_one, t, blocksize=args.blocksize, verify=verify): t
                    for t in tasks
                }
                for fut in as_completed(futures):
                    yield fut.result()

    for r in iter_results():
        counts[r.status] = counts.get(r.status, 0) + 1
        bytes_done += r.size_bytes
        _append_result(log_dir, r)
        pct = 100.0 * bytes_done / bytes_total if bytes_total else 0.0
        msg = r.message if r.status != "failed" else r.message
        print(
            f"[{ts()}] {r.status:<9} {pct:5.1f}% {r.dataset_id} "
            f"({r.size_bytes/1024/1024:.1f} MiB, {r.seconds:.1f}s) {msg}",
            flush=True,
        )
        if r.status == "failed" and r.traceback:
            # keep the TSV tidy; full traceback to stderr
            print(r.traceback, file=sys.stderr)

    print(
        f"[{ts()}] done converted={counts['converted']} skipped={counts['skipped']} "
        f"failed={counts['failed']} logs={log_dir}",
        flush=True,
    )
    return 1 if counts["failed"] else 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path, help="Directory tree containing full.zarr.zip stores.")
    p.add_argument("--blocksize", type=int, default=64 * 1024)
    p.add_argument("--jobs", type=int, default=8, help="Concurrent processes.")
    p.add_argument("--log-dir", type=Path, default=Path("./recompress-logs"))
    p.add_argument("--no-verify", action="store_true", help="Skip per-store launch + chunk verify.")
    p.add_argument("--limit", type=int, help="Process only the first N tasks (for testing).")
    return p.parse_args()


def _append_result(log_dir: Path, r: Result) -> None:
    path = log_dir / f"{r.status}.tsv"
    fields = ["status", "dataset_id", "zip_path", "size_bytes", "seconds",
              "blosc_arrays", "blosc_chunks", "message"]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        if not exists:
            w.writeheader()
        w.writerow({f: getattr(r, f) for f in fields})


def ts() -> str:
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
