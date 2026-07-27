#!/usr/bin/env python3
"""Audit gene-list alignment across a tree of scdata ``.zarr.zip`` stores.

Reads each store's complete ``var`` index through :func:`scdata.io.read_var_names`
and reports distinct gene lists, merge feasibility, and union/intersection sizes.
The reader owns zarr details such as multi-chunk indices and categorical indices;
this script never opens expression-matrix chunks.

Usage
-----
    uv run python scripts/audit_gene_alignment.py \
        --root .../20260625_dataset [--jobs 48] [--limit N] [--dry-run]

Output goes to stdout (human-readable summary) and, when ``--output-tsv`` is
set, a TSV with one row per file for downstream auditing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("BLOSC_NTHREADS", "1")

DEFAULT_ROOT = Path(
    "/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/FFPE/20260625_dataset"
)


@dataclass(frozen=True)
class GeneInfo:
    """Gene-list fingerprint for a single store."""

    path: str
    n_genes: int
    gene_hash: str
    first3: tuple[str, ...]
    last3: tuple[str, ...]
    error: str = ""


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    files = apply_selection(
        sorted(root.rglob("*.zarr.zip")),
        start=args.start,
        limit=args.limit,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    if args.dry_run:
        print(f"[{timestamp()}] dry-run: selected {len(files)} files under {root}")
        for path in files[: args.print_limit]:
            print(f"DRY_RUN\t{path}")
        if len(files) > args.print_limit:
            print(f"... {len(files) - args.print_limit} more")
        print("dry-run: no stores were opened and no TSV was written")
        return 0

    print(
        f"[{timestamp()}] scanning {len(files)} files under {root} jobs={args.jobs}",
        flush=True,
    )
    if not files:
        print("no .zarr.zip files found", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    infos = [read_gene_info(path) for path in files] if args.jobs == 1 else list(
        iter_parallel(files, jobs=args.jobs)
    )
    elapsed = time.perf_counter() - t0

    ok = [info for info in infos if not info.error]
    failed = [info for info in infos if info.error]
    print(
        f"[{timestamp()}] scanned {len(infos)} files in {elapsed:.1f}s "
        f"({len(ok)} ok, {len(failed)} failed)",
        flush=True,
    )

    if failed:
        print(f"\n=== FAILED ({len(failed)}) ===")
        for info in failed[:20]:
            print(f"  {info.path}: {info.error}")
        if len(failed) > 20:
            print(f"  ... {len(failed) - 20} more")

    if not ok:
        print("no files succeeded — nothing to analyze", file=sys.stderr)
        return 1

    analyze(ok, root)
    if args.output_tsv:
        write_tsv(args.output_tsv, infos)
        print(f"\nwrote TSV: {args.output_tsv}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent processes.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Process at most N files.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output-tsv", type=Path, help="Write per-file TSV.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-limit", type=int, default=20)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.start < 0:
        parser.error("--start must be non-negative")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.print_limit < 0:
        parser.error("--print-limit must be non-negative")
    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must satisfy 0 <= shard-index < num-shards")
    return args


def read_gene_info(path: Path) -> GeneInfo:
    """Read and fingerprint the complete var index without loading ``X``."""
    try:
        from scdata.io import read_var_names

        # ``read_var_names`` handles zarr-v3 codec stacks, every chunk, and
        # categorical indices.  Converting elementwise preserves the index order.
        names = [str(name) for name in read_var_names(path, raw=False)]
        gene_hash = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()[:16]
        return GeneInfo(
            path=str(path),
            n_genes=len(names),
            gene_hash=gene_hash,
            first3=tuple(names[:3]),
            last3=tuple(names[-3:]),
        )
    except Exception as err:  # one malformed store must not abort the audit
        return GeneInfo(
            path=str(path),
            n_genes=0,
            gene_hash="",
            first3=(),
            last3=(),
            error=f"{type(err).__name__}: {err}",
        )


def iter_parallel(files: list[Path], *, jobs: int) -> Iterable[GeneInfo]:
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(read_gene_info, path) for path in files]
        for future in as_completed(futures):
            yield future.result()


def analyze(infos: list[GeneInfo], root: Path) -> None:
    by_hash: dict[str, list[GeneInfo]] = defaultdict(list)
    for info in infos:
        by_hash[info.gene_hash].append(info)
    groups = sorted(by_hash.items(), key=lambda item: -len(item[1]))

    print(f"\n{'=' * 60}\nGENE LIST ALIGNMENT REPORT\n{'=' * 60}")
    print(f"files scanned:       {len(infos)}")
    print(f"distinct gene lists: {len(groups)}\n")
    for number, (gene_hash, group) in enumerate(groups, start=1):
        representative = group[0]
        print(f"--- group {number}/{len(groups)}: hash={gene_hash} ---")
        print(f"  files:   {len(group)}")
        print(f"  n_genes: {representative.n_genes}")
        print(f"  first3:  {representative.first3}")
        print(f"  last3:   {representative.last3}")
        for sample in group[:5]:
            print(f"  sample:  {_relative_or_absolute(Path(sample.path), root)}")
        if len(group) > 5:
            print(f"  ... {len(group) - 5} more")
        print()

    print(f"{'=' * 60}\nMERGE FEASIBILITY\n{'=' * 60}")
    if len(groups) == 1:
        print(f"strict:       YES — all {len(infos)} files share the same gene list")
        print(f"union:        YES — trivially the same list ({infos[0].n_genes} genes)")
        print(f"intersection: YES — trivially the same list ({infos[0].n_genes} genes)")
        return

    print(f"strict:       NO — {len(groups)} distinct gene lists")
    representatives: list[tuple[str, list[str]]] = []
    for gene_hash, group in groups:
        print(f"  hash {gene_hash}: {len(group)} files, {group[0].n_genes} genes")
        representatives.append((gene_hash, read_var_names(Path(group[0].path))))

    seen: dict[str, None] = {}
    for _, names in representatives:
        for name in names:
            seen.setdefault(str(name), None)
    common = set(map(str, representatives[0][1]))
    for _, names in representatives[1:]:
        common.intersection_update(map(str, names))
    print(f"union:        YES — {len(seen)} genes (from {len(groups)} lists)")
    print(
        f"intersection: {'YES' if common else 'NO'} — {len(common)} genes shared "
        f"across all {len(groups)} lists"
    )


def read_var_names(path: Path) -> list[str]:
    """Return the complete name list for report representatives."""
    from scdata.io import read_var_names as _read_var_names

    return [str(name) for name in _read_var_names(path, raw=False)]


def _relative_or_absolute(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def apply_selection(
    files: list[Path], *, start: int, limit: int | None, num_shards: int, shard_index: int
) -> list[Path]:
    selected = files[start:]
    if limit is not None:
        selected = selected[:limit]
    if num_shards > 1:
        selected = [path for index, path in enumerate(selected) if index % num_shards == shard_index]
    return selected


def write_tsv(path: Path, infos: list[GeneInfo]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["path", "n_genes", "gene_hash", "first3", "last3", "error"])
        for info in infos:
            writer.writerow(
                [
                    info.path,
                    info.n_genes,
                    info.gene_hash,
                    "|".join(info.first3),
                    "|".join(info.last3),
                    info.error,
                ]
            )


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
