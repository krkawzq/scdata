#!/usr/bin/env python3
"""Merge one strict gene-hash group into a single ``.zarr.zip`` store.

Every source must have an identical complete ``adata.var`` and only the merge
safe AnnData slots: ``X``, ``obs``, ``var``, and the whitelisted provenance
``uns`` keys.  Original ``obs`` columns are preserved and the biological labels
parsed from each filename are added.  The merged obs index is
``{sample_id}_{source_obs_index}``; collisions are rejected rather than silently
creating an invalid AnnData object.

Usage
-----
    uv run python scripts/merge_group.py \
        --groups-dir .../gene_groups --output-dir .../merged --group 1 \
        [--limit N] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("BLOSC_NTHREADS", "1")

DEFAULT_GROUPS_DIR = Path(
    "/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/FFPE/gene_groups"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/FFPE/merged"
)

_M20_RE = re.compile(r"^(\d{6,8})-([A-Z]{3})-(.+)$")
_PUB_RE = re.compile(r"^(?:f\.)?(.+?)\.d\d+_(.+)$")
LABEL_COLUMNS = ("sample_id", "tissue", "matrix_type", "batch", "donor_id")
ALLOWED_UNS_KEYS = frozenset({"scdata_source", "sample_metadata"})


@dataclass(frozen=True)
class FileLabels:
    sample_id: str
    tissue: str
    matrix_type: str
    batch: str
    donor_id: str


@dataclass(frozen=True)
class SourceMatrixInfo:
    path: Path
    num_cells: int
    num_genes: int
    nnz: int
    data_dtype: str


def parse_filename(stem: str) -> FileLabels:
    """Parse a zarr-zip filename stem into the labels added to ``obs``."""
    if stem.endswith(".raw"):
        base, matrix_type = stem[: -len(".raw")], "raw"
    elif stem.endswith(".filtered"):
        base, matrix_type = stem[: -len(".filtered")], "filtered"
    else:
        base, matrix_type = stem, ""

    m = _M20_RE.match(base)
    if m:
        return FileLabels(base, "", matrix_type, m.group(2), m.group(3))
    m = _PUB_RE.match(base)
    if m:
        return FileLabels(base, m.group(1).replace(".", " "), matrix_type, "", m.group(2))
    return FileLabels(base, "", matrix_type, "", base)


def identify_genome(
    gene_hash: str, n_genes: int, first_gene_symbol: str, gene_id_sample: str
) -> dict[str, str]:
    """Identify a readable output name from the complete first ``adata.var``."""
    del gene_hash, n_genes
    is_mouse = gene_id_sample.startswith("ENSMUSG") or (
        not gene_id_sample.startswith("ENSG") and _is_mouse_symbol(gene_id_sample)
    )
    if is_mouse:
        return {"species": "Mus_musculus", "build": "GRCm39", "annotation": "GENCODE"}
    if gene_id_sample.startswith("ENSG"):
        if first_gene_symbol == "TSPAN6":
            annotation = "ENSEMBL"
        elif first_gene_symbol == "TNFRSF4":
            annotation = "GENCODE_v32"
        elif first_gene_symbol in ("A1BG", "MIR1302-2HG", "SAMD11", "OR4F5"):
            annotation = "GENCODE"
        else:
            annotation = "ENSEMBL"
    else:
        annotation = "custom_rRNA"
    return {"species": "Homo_sapiens", "build": "GRCh38", "annotation": annotation}


def _is_mouse_symbol(symbol: str) -> bool:
    return len(symbol) > 1 and symbol[0].isupper() and symbol[1:].islower()


def validate_mergeable_adata(adata: Any, path: Path) -> None:
    """Reject AnnData state that this CSR-only merger cannot preserve safely."""
    populated = []
    for slot in ("layers", "obsm", "varm", "obsp", "varp"):
        value = getattr(adata, slot)
        if len(value):
            populated.append(slot)
    if adata.raw is not None:
        populated.append("raw")
    unexpected_uns = sorted(set(adata.uns) - ALLOWED_UNS_KEYS)
    if unexpected_uns:
        populated.append(f"uns ({', '.join(unexpected_uns)})")
    if populated:
        raise ValueError(
            f"{path}: merge rejects populated AnnData slots: {', '.join(populated)}; "
            "remove them explicitly or use a merger that preserves them"
        )


def add_labels_and_validate_obs(adata: Any, labels: FileLabels, path: Path) -> Any:
    """Copy original obs, reject label conflicts, and append categorical labels."""
    import pandas as pd

    conflicts = sorted(set(LABEL_COLUMNS).intersection(adata.obs.columns))
    if conflicts:
        raise ValueError(
            f"{path}: source obs already contains merge label columns: {', '.join(conflicts)}"
        )
    obs = adata.obs.copy()
    source_index = obs.index.astype(str)
    merged_index = pd.Index(
        [f"{labels.sample_id}_{barcode}" for barcode in source_index], dtype="object"
    )
    if merged_index.has_duplicates:
        raise ValueError(f"{path}: duplicate merged obs index within source file")
    obs.index = merged_index
    for column, value in zip(LABEL_COLUMNS, labels.__dict__.values(), strict=True):
        obs[column] = pd.Categorical([value] * len(obs))
    return obs


def strict_var_equal(reference: Any, candidate: Any) -> bool:
    """Compare complete var metadata, including index, columns, values, and dtypes."""
    return reference.equals(candidate)


def _first_var_value(var: Any, column: str) -> str:
    if len(var.index) == 0:
        raise ValueError("cannot identify genome from an empty var table")
    if column in var.columns:
        return str(var.iloc[0][column])
    return str(var.index[0])


def _scan_sparse_sources(paths: list[Path]) -> list[SourceMatrixInfo]:
    """Read only launch metadata so the merged CSR can be allocated once."""
    from scdata.data import SparseDataset
    from scdata.io import launch

    infos: list[SourceMatrixInfo] = []
    for path in paths:
        dataset = launch(path)
        if not isinstance(dataset, SparseDataset):
            raise ValueError(f"{path}: merge requires a sparse CSR X dataset")
        infos.append(
            SourceMatrixInfo(
                path=path,
                num_cells=dataset.num_cells,
                num_genes=dataset.num_genes,
                nnz=dataset.nnz,
                data_dtype=dataset.dtype.value,
            )
        )
    return infos


def _numpy_dtype(dtype: str) -> str:
    mapping = {
        "u8": "uint8",
        "i8": "int8",
        "u16": "uint16",
        "i16": "int16",
        "u32": "uint32",
        "i32": "int32",
        "u64": "uint64",
        "i64": "int64",
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
    }
    try:
        return mapping[dtype]
    except KeyError as err:
        raise ValueError(f"merge does not support data dtype {dtype!r}") from err


def _detected_memory_limit_bytes() -> int | None:
    limits: list[int] = []
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text and text != "max":
                value = int(text)
                if 0 < value < 1 << 60:
                    limits.append(value)
        except (FileNotFoundError, OSError, ValueError):
            pass
    try:
        physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        if physical > 0:
            limits.append(physical)
    except (OSError, ValueError):
        pass
    return min(limits) if limits else None


def _memory_budget_bytes(memory_budget_gib: float) -> int:
    if memory_budget_gib > 0:
        return int(memory_budget_gib * 1024**3)
    detected = _detected_memory_limit_bytes()
    if detected is None:
        raise ValueError("could not detect a memory limit; pass --memory-budget-gib explicitly")
    return detected * 4 // 5


def _estimate_merge_peak_bytes(infos: list[SourceMatrixInfo]) -> tuple[int, int, int]:
    """Return estimated peak, final CSR bytes, and largest source CSR bytes."""
    import numpy as np

    total_cells = sum(info.num_cells for info in infos)
    total_nnz = sum(info.nnz for info in infos)
    data_dtype = np.result_type(*[np.dtype(_numpy_dtype(info.data_dtype)) for info in infos])
    index_dtype = np.dtype(
        "int64" if max(total_nnz, max(info.num_genes for info in infos)) > 2**31 - 1 else "int32"
    )
    final_csr = (
        total_nnz * (data_dtype.itemsize + index_dtype.itemsize)
        + (total_cells + 1) * np.dtype("int64").itemsize
    )
    largest_source = max(
        info.nnz * (np.dtype(_numpy_dtype(info.data_dtype)).itemsize + index_dtype.itemsize)
        + (info.num_cells + 1) * np.dtype("int64").itemsize
        for info in infos
    )
    annotation_reserve = max(256 * 1024**2, total_cells * 1024)
    return final_csr + largest_source + annotation_reserve, final_csr, largest_source


def main() -> int:
    args = parse_args()
    if args.limit == 0:
        print("limit=0: no-op; no input was read and no output was created", flush=True)
        return 0

    groups_dir = args.groups_dir.resolve()
    group_name = f"group_{args.group:03d}"
    manifest_path = groups_dir / f"{group_name}_manifest.txt"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found", file=sys.stderr)
        return 2

    paths = [Path(line.strip()) for line in manifest_path.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        print(f"{group_name}: no input files selected; no-op", flush=True)
        return 0

    print(f"[{timestamp()}] {group_name}: {len(paths)} files", flush=True)
    if args.dry_run:
        for path in paths[:10]:
            labels = parse_filename(_zarr_stem(path))
            print(f"  {path}\n    {labels}")
        return 0

    # Deliberately defer directory creation until all input validation has passed.
    output_dir = args.output_dir.resolve()
    from scdata.io._anndata import read_zarr, write_zarr
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    infos = _scan_sparse_sources(paths)
    if len({info.num_genes for info in infos}) != 1:
        raise ValueError("source matrices have different gene counts")
    peak_bytes, final_csr_bytes, largest_source_bytes = _estimate_merge_peak_bytes(infos)
    budget_bytes = _memory_budget_bytes(args.memory_budget_gib)
    if peak_bytes > budget_bytes:
        raise MemoryError(
            "estimated merge peak exceeds the configured memory budget: "
            f"peak={peak_bytes / 1024**3:.2f} GiB, budget={budget_bytes / 1024**3:.2f} GiB, "
            f"final_csr={final_csr_bytes / 1024**3:.2f} GiB, "
            f"largest_source={largest_source_bytes / 1024**3:.2f} GiB; "
            "increase --memory-budget-gib or split the group"
        )

    data_dtype = np.result_type(*[np.dtype(_numpy_dtype(info.data_dtype)) for info in infos])
    index_dtype = np.dtype(
        "int64"
        if max(sum(info.nnz for info in infos), infos[0].num_genes) > 2**31 - 1
        else "int32"
    )
    total_cells = sum(info.num_cells for info in infos)
    total_nnz = sum(info.nnz for info in infos)
    all_data = np.empty(total_nnz, dtype=data_dtype)
    all_indices = np.empty(total_nnz, dtype=index_dtype)
    all_indptr = np.empty(total_cells + 1, dtype=np.int64)
    all_indptr[0] = 0
    obs_frames: list[Any] = []
    var_reference: Any | None = None
    file_cell_counts: list[int] = []
    cell_offset = nnz_offset = 0

    started = time.perf_counter()
    for index, (path, info) in enumerate(zip(paths, infos, strict=True), start=1):
        adata = read_zarr(path)
        try:
            validate_mergeable_adata(adata, path)
            if var_reference is None:
                var_reference = adata.var.copy(deep=True)
            elif not strict_var_equal(var_reference, adata.var):
                raise ValueError(
                    f"{path}: adata.var differs from the first source; strict merge requires "
                    "identical index, columns, values, and dtypes"
                )

            labels = parse_filename(_zarr_stem(path))
            obs_frames.append(add_labels_and_validate_obs(adata, labels, path))
            csr = adata.X
            if not sp.isspmatrix_csr(csr):
                csr = csr.tocsr()
            if csr.shape != (info.num_cells, info.num_genes) or csr.nnz != info.nnz:
                raise ValueError(f"{path}: matrix shape/nnz changed between metadata scan and read")
            next_nnz = nnz_offset + info.nnz
            next_cell = cell_offset + info.num_cells
            all_data[nnz_offset:next_nnz] = np.asarray(csr.data, dtype=data_dtype)
            all_indices[nnz_offset:next_nnz] = np.asarray(csr.indices, dtype=index_dtype)
            source_indptr = np.asarray(csr.indptr, dtype=np.int64)
            all_indptr[cell_offset + 1 : next_cell + 1] = source_indptr[1:] + nnz_offset
            nnz_offset = next_nnz
            cell_offset = next_cell
            file_cell_counts.append(info.num_cells)
        finally:
            del adata
        if index % 50 == 0 or index == len(paths):
            print(
                f"[{timestamp()}] loaded {index}/{len(paths)} files, cells={total_cells}, "
                f"nnz={total_nnz}, elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    assert var_reference is not None
    # ``pd.concat`` preserves every original obs column.  Missing source columns
    # become missing values; a label conflict was rejected above instead.
    obs = pd.concat(obs_frames, axis=0, copy=False)
    if not obs.index.is_unique:
        duplicates = obs.index[obs.index.duplicated()].unique()[:5].tolist()
        raise ValueError(f"duplicate merged obs index across source files: {duplicates!r}")

    if cell_offset != total_cells or nnz_offset != total_nnz:
        raise RuntimeError("internal merge fill count mismatch")
    merged_csr = sp.csr_matrix(
        (all_data, all_indices, all_indptr), shape=(total_cells, var_reference.shape[0])
    )

    # All metadata originated from the complete first ``adata.var``.  Only the
    # conversion bookkeeping column is deliberately removed from the output.
    output_var = var_reference.drop(columns=["scdata_original_var_names"], errors="ignore")
    merged = ad.AnnData(X=merged_csr, obs=obs, var=output_var)
    genome = identify_genome(
        gene_hash="",
        n_genes=merged.n_vars,
        first_gene_symbol=_first_var_value(var_reference, "gene_symbol"),
        gene_id_sample=_first_var_value(var_reference, "gene_id"),
    )
    out_name = (
        f"{genome['species']}_{genome['build']}_{genome['annotation']}_"
        f"{merged.n_vars}genes.zarr.zip"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = _available_output_path(output_dir / out_name, args.overwrite)

    print(f"[{timestamp()}] output: {out_path.name}", flush=True)
    write_started = time.perf_counter()
    write_zarr(
        merged,
        out_path,
        format="sparse",
        store="zip",
        compressor="blosc.lz4.level5",
        blocksize=65536,
    )
    print(
        f"[{timestamp()}] write done in {time.perf_counter() - write_started:.1f}s, "
        f"size={out_path.stat().st_size / 1024**3:.2f} GiB",
        flush=True,
    )
    _write_labels(out_path, paths, file_cell_counts)
    return 0


def _zarr_stem(path: Path) -> str:
    name = path.name
    return name[: -len(".zarr.zip")] if name.endswith(".zarr.zip") else path.stem


def _available_output_path(path: Path, overwrite: bool) -> Path:
    if overwrite or not path.exists():
        return path
    number = 2
    while True:
        candidate = path.with_name(f"{path.stem}_alt{number}{path.suffix}")
        if not candidate.exists():
            return candidate
        number += 1


def _write_labels(out_path: Path, paths: list[Path], cell_counts: list[int]) -> None:
    label_tsv = out_path.with_suffix(".labels.tsv")
    with label_tsv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            ["file_id", "path", "sample_id", "tissue", "matrix_type", "batch", "donor_id", "n_cells"]
        )
        for file_id, (path, n_cells) in enumerate(zip(paths, cell_counts, strict=True)):
            labels = parse_filename(_zarr_stem(path))
            writer.writerow([file_id, path, *labels.__dict__.values(), n_cells])
    print(f"[{timestamp()}] labels: {label_tsv}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--groups-dir", type=Path, default=DEFAULT_GROUPS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--group", type=int, required=True, help="Group number (1-based).")
    parser.add_argument("--limit", type=int, help="Process at most N files.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--memory-budget-gib",
        type=float,
        default=0.0,
        help=(
            "Maximum estimated merge peak in GiB; 0 auto-detects the cgroup/host "
            "limit and reserves 20%% headroom"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.group < 1:
        parser.error("--group must be >= 1")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.memory_budget_gib < 0:
        parser.error("--memory-budget-gib must be non-negative")
    return args


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
