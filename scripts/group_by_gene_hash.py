#!/usr/bin/env python3
"""Group .zarr.zip stores by gene list and write per-group manifests.

Reads the TSV produced by ``audit_gene_alignment.py`` and, for each distinct
gene hash, writes:

  * ``<output_dir>/<group_NNN>_manifest.txt``  — one path per line, in the
    order scdata's ``Corpus`` will register them (file_id = line number).
  * ``<output_dir>/<group_NNN>_metadata.tsv``  — per-file metadata:
    ``file_id``, ``path``, ``sample_id``, ``source_group``, ``matrix_dir``,
    ``n_genes``, ``n_cells``, ``gene_hash``.

The manifest is the input to ``Corpus(paths, gene_alignment="strict")``; the
metadata TSV lets training code map integer ``file_ids`` (from
``ScDataBatch["file_ids"]``) back to human-readable sample labels.

Usage
-----
    uv run python scripts/group_by_gene_hash.py \
        --audit-tsv .../gene_audit.tsv \
        --root .../20260625_dataset \
        --output-dir .../gene_groups

    # then in training:
    #   paths = [line.strip() for line in open("group_001_manifest.txt")]
    #   corpus = Corpus(paths, gene_alignment="strict")
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("BLOSC_NTHREADS", "1")

DEFAULT_AUDIT_TSV = Path(
    "/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/FFPE/gene_audit.tsv"
)
DEFAULT_ROOT = Path(
    "/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/FFPE/20260625_dataset"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/FFPE/gene_groups"
)


@dataclass(frozen=True)
class FileMeta:
    path: str
    sample_id: str
    source_group: str
    matrix_dir: str
    n_genes: int
    n_cells: int
    gene_hash: str


def main() -> int:
    args = parse_args()
    audit_tsv = args.audit_tsv.resolve()
    args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load audit TSV.
    rows = []
    with audit_tsv.open("r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r.get("error"):
                continue
            rows.append(r)
    print(f"[{timestamp()}] loaded {len(rows)} files from {audit_tsv}", flush=True)

    # Group by gene_hash.
    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_hash[r["gene_hash"]].append(r)
    groups = sorted(by_hash.items(), key=lambda kv: -len(kv[1]))
    print(f"[{timestamp()}] {len(groups)} distinct gene lists", flush=True)

    # Enrich each file with sample_id / source_group / n_cells by reading
    # uns/scdata_source from the zarr.zip (fast: only metadata chunks).
    all_files = [r["path"] for group in groups for r in group[1]]
    print(f"[{timestamp()}] reading metadata from {len(all_files)} files...", flush=True)
    t0 = time.perf_counter()
    if args.jobs == 1:
        metas = [read_file_meta(p) for p in all_files]
    else:
        metas = list(iter_parallel(all_files, args.jobs, read_file_meta))
    elapsed = time.perf_counter() - t0
    print(f"[{timestamp()}] metadata read in {elapsed:.1f}s", flush=True)

    meta_by_path = {m.path: m for m in metas}

    # Write per-group manifest + metadata TSV.
    summary_lines = []
    for i, (ghash, group_rows) in enumerate(groups):
        group_num = i + 1
        group_name = f"group_{group_num:03d}"
        group_files = [r["path"] for r in group_rows]
        group_metas = [meta_by_path.get(p) for p in group_files]

        # Manifest: one path per line.
        manifest_path = output_dir / f"{group_name}_manifest.txt"
        with manifest_path.open("w", encoding="utf-8") as fh:
            for p in group_files:
                fh.write(p + "\n")

        # Metadata TSV.
        meta_path = output_dir / f"{group_name}_metadata.tsv"
        with meta_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow([
                "file_id", "path", "sample_id", "source_group", "matrix_dir",
                "n_genes", "n_cells", "gene_hash",
            ])
            for file_id, meta in enumerate(group_metas):
                if meta is None:
                    writer.writerow([file_id, group_files[file_id], "", "", "", 0, 0, ghash])
                else:
                    writer.writerow([
                        file_id, meta.path, meta.sample_id, meta.source_group,
                        meta.matrix_dir, meta.n_genes, meta.n_cells, ghash,
                    ])

        n_files = len(group_files)
        n_cells = sum(m.n_cells for m in group_metas if m)
        n_genes = group_metas[0].n_genes if group_metas and group_metas[0] else 0
        first3 = group_rows[0]["first3"]
        summary_lines.append(
            f"{group_name}  files={n_files:>5}  genes={n_genes:>6}  cells={n_cells:>10}  "
            f"hash={ghash}  first3={first3}"
        )

    # Write summary.
    summary_path = output_dir / "groups_summary.txt"
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(f"total groups: {len(groups)}\n")
        fh.write(f"total files:  {len(all_files)}\n")
        fh.write(f"total cells:  {sum(m.n_cells for m in metas if m)}\n\n")
        for line in summary_lines:
            fh.write(line + "\n")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for line in summary_lines:
        print(line)
    print(f"\nwrote {len(groups)} groups to {output_dir}")
    print(f"summary: {summary_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audit-tsv", type=Path, default=DEFAULT_AUDIT_TSV)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent processes.")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    return args


def read_file_meta(path_str: str) -> FileMeta:
    """Read sample_id / source_group / matrix_dir / n_cells from a zarr.zip.

    Uses ``scdata.io._anndata.read_zarr(metadata_only=True)`` which decodes
    ``uns/scdata_source`` and reads ``obs`` shape without touching the
    expression matrix — fast enough for 2841 files at ~20ms each.
    """
    from scdata.io._anndata import read_zarr

    path = Path(path_str)
    try:
        adata = read_zarr(path, metadata_only=True)
        src = adata.uns.get("scdata_source", {})
        return FileMeta(
            path=path_str,
            sample_id=str(src.get("sample_id", "")),
            source_group=str(src.get("source_group", "")),
            matrix_dir=str(src.get("matrix_dir", "")),
            n_genes=int(adata.n_vars),
            n_cells=int(adata.n_obs),
            gene_hash="",
        )
    except Exception:
        return FileMeta(
            path=path_str, sample_id="", source_group="", matrix_dir="",
            n_genes=0, n_cells=0, gene_hash="",
        )


def iter_parallel(items: list[str], jobs: int, fn: Any) -> Iterable[Any]:
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(fn, item): item for item in items}
        for future in as_completed(futures):
            yield future.result()


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
