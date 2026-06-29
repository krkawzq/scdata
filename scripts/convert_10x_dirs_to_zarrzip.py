#!/usr/bin/env python3
"""Batch-convert 10x Matrix Market directories to scdata ``.zarr.zip`` stores.

This script is intended for large FFPE-style collections where each sample is a
10x directory containing ``matrix.mtx(.gz)``, ``features.tsv(.gz)``, and
``barcodes.tsv(.gz)``.  It builds an AnnData object per directory, writes a
temporary scdata zarr v3 directory via ``scdata.io.write_zarr(store="dir")``,
then packs that directory into a ZIP_STORED ``.zarr.zip`` archive.

The high-level ``AnnDataZarrZipConverter`` can read a single ``.mtx.gz`` file,
but that loses 10x feature/barcode metadata.  This script treats the directory
as the conversion unit.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import shutil
import sys
import time
import traceback
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("BLOSC_NTHREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT_ROOT = Path(
    "/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/FFPE/20260625_dataset"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/FFPE/20260625_dataset_zarrzip"
)
MATRIX_DIR_NAMES = frozenset({"filtered_feature_bc_matrix", "raw_feature_bc_matrix"})
DEFAULT_COMPRESSOR = "blosc.lz4.level5"
_DataDtype = Literal[
    "auto",
    "preserve",
    "uint16",
    "uint32",
    "uint64",
    "int32",
    "int64",
    "float32",
    "float64",
]
DEFAULT_DATA_DTYPE: _DataDtype = "auto"


def configure_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"zarr v3 autosharding will be the default.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Consolidated metadata is currently not part in the Zarr format 3 specification.*",
        category=UserWarning,
    )


configure_warning_filters()


@dataclass(frozen=True)
class ConvertTask:
    source_dir: str
    target_zip: str


@dataclass(frozen=True)
class ConvertResult:
    status: Literal["converted", "skipped", "failed", "dry_run"]
    source_dir: str
    target_zip: str
    sample_id: str
    n_obs: int | None = None
    n_vars: int | None = None
    nnz: int | None = None
    input_bytes: int | None = None
    output_bytes: int | None = None
    seconds: float = 0.0
    message: str = ""
    traceback: str = ""


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    source_dirs = discover_source_dirs(
        input_root=input_root,
        manifest=args.manifest,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
    )
    source_dirs = apply_selection(
        source_dirs,
        start=args.start,
        limit=args.limit,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    tasks = build_tasks(
        source_dirs,
        input_root=input_root,
        output_root=output_root,
        drop_matrix_dir=not args.keep_matrix_dir_in_output_name,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir.resolve() if args.log_dir is not None else output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.write_manifest is not None:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(
            "\n".join(task.source_dir for task in tasks) + ("\n" if tasks else ""),
            encoding="utf-8",
        )

    print(
        f"[{timestamp()}] tasks={len(tasks)} input_root={input_root} "
        f"output_root={output_root} jobs={args.jobs} compressor={args.compressor} "
        f"data_dtype={args.data_dtype}",
        flush=True,
    )
    if args.dry_run:
        for task in tasks[: args.print_limit]:
            print(f"DRY_RUN\t{task.source_dir}\t{task.target_zip}")
        append_results(log_dir / "dry_run.tsv", (dry_run_result(task) for task in tasks))
        print(f"[{timestamp()}] dry-run listed {min(len(tasks), args.print_limit)} tasks")
        return 0

    common = {
        "chunk_size": args.chunk_size,
        "compressor": args.compressor,
        "data_dtype": args.data_dtype,
        "var_names": args.var_names,
        "make_var_names_unique": not args.no_make_var_names_unique,
        "sample_metadata": args.sample_metadata,
        "obs_sample_id": args.obs_sample_id,
        "overwrite": args.overwrite,
        "verify": not args.no_verify,
        "keep_zarr": args.keep_zarr,
        "keep_failed_zarr": args.keep_failed_zarr,
    }

    counts = {"converted": 0, "skipped": 0, "failed": 0}
    if args.jobs == 1:
        result_iter: Iterable[ConvertResult] = (
            convert_one_task(task, **common) for task in tasks
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
    parser.add_argument("--manifest", type=Path, help="Text file with one 10x matrix dir per line.")
    parser.add_argument("--write-manifest", type=Path, help="Write selected source dirs for audit.")
    parser.add_argument("--log-dir", type=Path, help="Directory for converted/skipped/failed TSV logs.")
    parser.add_argument("--include-regex", help="Only convert source dirs whose path matches this regex.")
    parser.add_argument("--exclude-regex", help="Skip source dirs whose path matches this regex.")
    parser.add_argument("--start", type=int, default=0, help="Skip the first N selected source dirs.")
    parser.add_argument("--limit", type=int, help="Convert at most N selected source dirs.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total shard count for manual/rjob sharding.")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index to run.")
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent processes. Keep low for large matrices.")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="scdata sparse chunk target.")
    parser.add_argument(
        "--compressor",
        default=DEFAULT_COMPRESSOR,
        help="Chunk compressor passed to scdata.io.write_zarr. Use 'none' to disable.",
    )
    parser.add_argument(
        "--data-dtype",
        choices=(
            "auto",
            "preserve",
            "uint16",
            "uint32",
            "uint64",
            "int32",
            "int64",
            "float32",
            "float64",
        ),
        default=DEFAULT_DATA_DTYPE,
        help=(
            "Dtype for Matrix Market values in X.data. 'auto' stores count matrices "
            "as uint16, promoting to uint32/uint64 when needed; use 'preserve' to "
            "keep scipy.mmread's dtype."
        ),
    )
    parser.add_argument("--var-names", choices=("symbol", "id"), default="symbol")
    parser.add_argument("--no-make-var-names-unique", action="store_true")
    parser.add_argument("--sample-metadata", choices=("none", "uns"), default="uns")
    parser.add_argument("--obs-sample-id", action="store_true", help="Add repeated sample_id/source_group columns to obs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target .zarr.zip files.")
    parser.add_argument("--no-verify", action="store_true", help="Skip scdata.io.launch verification after writing.")
    parser.add_argument("--keep-zarr", action="store_true", help="Keep intermediate .work.zarr directories after success.")
    parser.add_argument("--keep-failed-zarr", action="store_true", help="Keep intermediate zarr dirs when conversion fails.")
    parser.add_argument(
        "--keep-matrix-dir-in-output-name",
        action="store_true",
        help="Use raw_feature_bc_matrix.zarr.zip instead of parent-sample-name.zarr.zip.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-limit", type=int, default=20)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    if args.start < 0:
        parser.error("--start must be non-negative")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must satisfy 0 <= shard-index < num-shards")
    return args


def discover_source_dirs(
    *,
    input_root: Path,
    manifest: Path | None,
    include_regex: str | None,
    exclude_regex: str | None,
) -> list[Path]:
    if manifest is not None:
        source_dirs = [
            Path(line.strip()).resolve()
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        source_dirs = sorted({path.parent.resolve() for path in input_root.rglob("matrix.mtx*")})

    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    selected: list[Path] = []
    for source_dir in source_dirs:
        text = str(source_dir)
        if include is not None and not include.search(text):
            continue
        if exclude is not None and exclude.search(text):
            continue
        selected.append(source_dir)
    return selected


def apply_selection(
    source_dirs: list[Path],
    *,
    start: int,
    limit: int | None,
    num_shards: int,
    shard_index: int,
) -> list[Path]:
    selected = source_dirs[start:]
    if limit is not None:
        selected = selected[:limit]
    if num_shards > 1:
        selected = [path for i, path in enumerate(selected) if i % num_shards == shard_index]
    return selected


def build_tasks(
    source_dirs: list[Path],
    *,
    input_root: Path,
    output_root: Path,
    drop_matrix_dir: bool,
) -> list[ConvertTask]:
    first_pass = [
        ConvertTask(
            source_dir=str(source_dir),
            target_zip=str(default_target_zip(source_dir, input_root, output_root, drop_matrix_dir)),
        )
        for source_dir in source_dirs
    ]
    by_target: dict[str, list[Path]] = {}
    for task in first_pass:
        by_target.setdefault(task.target_zip, []).append(Path(task.source_dir))

    tasks: list[ConvertTask] = []
    for source_dir in source_dirs:
        target = default_target_zip(source_dir, input_root, output_root, drop_matrix_dir)
        if len(by_target.get(str(target), ())) > 1:
            target = default_target_zip(source_dir, input_root, output_root, drop_matrix_dir=False)
        tasks.append(ConvertTask(str(source_dir), str(target)))
    return tasks


def default_target_zip(
    source_dir: Path,
    input_root: Path,
    output_root: Path,
    drop_matrix_dir: bool,
) -> Path:
    try:
        rel = source_dir.resolve().relative_to(input_root)
    except ValueError:
        rel = Path(safe_name(str(source_dir).strip("/").replace("/", "__")))

    if drop_matrix_dir and rel.name in MATRIX_DIR_NAMES and rel.parent != Path("."):
        sample_rel = rel.parent
        return output_root / sample_rel.parent / f"{safe_name(sample_rel.name)}.zarr.zip"
    return output_root / rel.parent / f"{safe_name(rel.name)}.zarr.zip"


def convert_one_task(
    task: ConvertTask,
    *,
    chunk_size: int,
    compressor: str,
    data_dtype: _DataDtype,
    var_names: Literal["symbol", "id"],
    make_var_names_unique: bool,
    sample_metadata: Literal["none", "uns"],
    obs_sample_id: bool,
    overwrite: bool,
    verify: bool,
    keep_zarr: bool,
    keep_failed_zarr: bool,
) -> ConvertResult:
    started = time.perf_counter()
    source_dir = Path(task.source_dir)
    target_zip = Path(task.target_zip)
    sample_id = sample_id_from_source_dir(source_dir)
    target_zip.parent.mkdir(parents=True, exist_ok=True)

    if target_zip.exists() and not overwrite:
        return ConvertResult(
            status="skipped",
            source_dir=str(source_dir),
            target_zip=str(target_zip),
            sample_id=sample_id,
            output_bytes=target_zip.stat().st_size,
            seconds=time.perf_counter() - started,
            message="target exists",
        )

    work_zarr = target_zip.parent / f".{target_zip.name}.work.zarr"
    tmp_zip = target_zip.parent / f".{target_zip.name}.tmp"
    try:
        remove_path(work_zarr)
        remove_path(tmp_zip)

        adata = read_10x_directory(
            source_dir,
            data_dtype=data_dtype,
            var_names=var_names,
            make_var_names_unique=make_var_names_unique,
            sample_metadata=sample_metadata,
            obs_sample_id=obs_sample_id,
        )
        input_bytes = sum_input_bytes(source_dir)
        n_obs = int(adata.n_obs)
        n_vars = int(adata.n_vars)
        nnz = int(getattr(adata.X, "nnz", 0))

        from scdata.io import launch, write_zarr

        write_zarr(
            adata,
            work_zarr,
            format="sparse",
            layer_format="preserve",
            chunk_size=(chunk_size,),
            align_cells=True,
            store="dir",
            compressor=compressor,
        )
        zip_directory_stored(work_zarr, tmp_zip)
        os.replace(tmp_zip, target_zip)

        if verify:
            ds = launch(target_zip)
            if ds.num_cells != n_obs or ds.num_genes != n_vars:
                raise RuntimeError(
                    f"launch shape mismatch: got {(ds.num_cells, ds.num_genes)}, "
                    f"expected {(n_obs, n_vars)}"
                )

        output_bytes = target_zip.stat().st_size
        if not keep_zarr:
            remove_path(work_zarr)
        return ConvertResult(
            status="converted",
            source_dir=str(source_dir),
            target_zip=str(target_zip),
            sample_id=sample_id,
            n_obs=n_obs,
            n_vars=n_vars,
            nnz=nnz,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            seconds=time.perf_counter() - started,
            message="ok",
        )
    except Exception as err:
        remove_path(tmp_zip)
        if not keep_failed_zarr:
            remove_path(work_zarr)
        return ConvertResult(
            status="failed",
            source_dir=str(source_dir),
            target_zip=str(target_zip),
            sample_id=sample_id,
            seconds=time.perf_counter() - started,
            message=f"{type(err).__name__}: {err}",
            traceback=traceback.format_exc(),
        )


def read_10x_directory(
    source_dir: Path,
    *,
    data_dtype: _DataDtype,
    var_names: Literal["symbol", "id"],
    make_var_names_unique: bool,
    sample_metadata: Literal["none", "uns"],
    obs_sample_id: bool,
) -> Any:
    import anndata as ad
    import pandas as pd
    from scipy import sparse
    from scipy.io import mmread

    matrix_path = require_existing(source_dir, "matrix.mtx")
    features_path = require_existing(source_dir, "features.tsv")
    barcodes_path = require_existing(source_dir, "barcodes.tsv")

    with open_text_or_gzip(matrix_path, binary=True) as fh:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The default value for `spmatrix` is changing.*",
                category=DeprecationWarning,
            )
            matrix = mmread(fh)
    matrix = sparse.coo_matrix(matrix)
    matrix = coerce_sparse_data_dtype(matrix, data_dtype=data_dtype, source=matrix_path)

    features = read_features(features_path, pd=pd)
    barcodes = read_barcodes(barcodes_path, pd=pd)
    if matrix.shape == (features.shape[0], barcodes.shape[0]):
        x = matrix.transpose().tocsr()
    elif matrix.shape == (barcodes.shape[0], features.shape[0]):
        x = matrix.tocsr()
    else:
        raise ValueError(
            f"matrix shape {matrix.shape} does not match "
            f"features={features.shape[0]} and barcodes={barcodes.shape[0]}"
        )

    var = features.copy()
    selected_names = select_var_names(var, var_names)
    var["scdata_original_var_names"] = selected_names
    var.index = selected_names

    obs = pd.DataFrame(index=barcodes["barcode"].astype(str).to_numpy())
    sample_id = sample_id_from_source_dir(source_dir)
    if obs_sample_id:
        obs["sample_id"] = sample_id
        obs["source_group"] = source_group_from_source_dir(source_dir)

    adata = ad.AnnData(X=x, obs=obs, var=var)
    if make_var_names_unique:
        adata.var_names_make_unique()

    adata.uns["scdata_source"] = {
        "sample_id": sample_id,
        "source_dir": str(source_dir),
        "source_group": source_group_from_source_dir(source_dir),
        "matrix_dir": source_dir.name,
    }
    if sample_metadata == "uns":
        metadata = read_sample_metadata(source_dir, pd=pd)
        if metadata:
            adata.uns["sample_metadata"] = metadata
    return adata


def coerce_sparse_data_dtype(matrix: Any, *, data_dtype: _DataDtype, source: Path) -> Any:
    """Cast sparse matrix values while rejecting lossy integer conversions."""
    if data_dtype == "preserve":
        return matrix

    import numpy as np

    data = np.asarray(matrix.data)
    if data_dtype == "auto":
        target = infer_count_dtype(data, source=source, np=np)
    else:
        target = np.dtype(data_dtype)
        validate_data_dtype_cast(data, target=target, source=source, np=np)

    return matrix.astype(target, copy=False)


def infer_count_dtype(data: Any, *, source: Path, np: Any) -> Any:
    """Choose the smallest unsigned dtype that can store non-negative counts."""
    if data.size == 0:
        return np.dtype("uint16")

    min_value, max_value = integer_value_range(data, source=source, np=np)
    if min_value < 0:
        raise ValueError(
            f"{source}: auto count dtype requires non-negative values, got min={min_value}; "
            "use --data-dtype preserve for non-count matrices"
        )

    for name in ("uint16", "uint32", "uint64"):
        if max_value <= np.iinfo(np.dtype(name)).max:
            return np.dtype(name)
    raise OverflowError(f"{source}: count value {max_value} exceeds uint64")


def validate_data_dtype_cast(data: Any, *, target: Any, source: Path, np: Any) -> None:
    """Reject explicit dtype casts that would lose integer count information."""
    if data.size == 0 or not np.issubdtype(target, np.integer):
        return
    min_value, max_value = integer_value_range(data, source=source, np=np)
    limits = np.iinfo(target)
    if min_value < limits.min or max_value > limits.max:
        raise OverflowError(
            f"{source}: matrix values [{min_value}, {max_value}] exceed {target.name}"
        )


def integer_value_range(data: Any, *, source: Path, np: Any) -> tuple[int, int]:
    """Return integer min/max while rejecting complex, non-finite, or fractional values."""
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.complexfloating):
        raise ValueError(f"{source}: count matrix values must be real, got {data.dtype}")
    if np.issubdtype(data.dtype, np.floating):
        if not np.all(np.isfinite(data)):
            raise ValueError(f"{source}: count matrix values must be finite")
        if not np.all(data == np.trunc(data)):
            raise ValueError(
                f"{source}: count matrix values must be integers; "
                "use --data-dtype preserve for non-count matrices"
            )
    elif not np.issubdtype(data.dtype, np.integer) and data.dtype != np.dtype("bool"):
        raise ValueError(f"{source}: unsupported count matrix dtype {data.dtype}")

    return int(data.min()), int(data.max())


def require_existing(source_dir: Path, stem: str) -> Path:
    candidates = (source_dir / f"{stem}.gz", source_dir / stem)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing {stem}(.gz) in {source_dir}")


def open_text_or_gzip(path: Path, *, binary: bool = False) -> Any:
    if path.suffix == ".gz":
        return gzip.open(path, "rb" if binary else "rt")
    return path.open("rb" if binary else "rt")


def read_features(path: Path, *, pd: Any) -> Any:
    frame = pd.read_csv(path, sep="\t", header=None, compression="infer", dtype=str)
    if frame.shape[1] >= 3:
        names = ["gene_id", "gene_symbol", "feature_type"]
        names.extend(f"feature_extra_{i}" for i in range(frame.shape[1] - 3))
    elif frame.shape[1] == 2:
        names = ["gene_id", "gene_symbol"]
    elif frame.shape[1] == 1:
        names = ["gene_symbol"]
    else:
        raise ValueError(f"empty features table: {path}")
    frame.columns = names
    return frame.fillna("")


def read_barcodes(path: Path, *, pd: Any) -> Any:
    frame = pd.read_csv(path, sep="\t", header=None, compression="infer", dtype=str)
    if frame.shape[1] < 1:
        raise ValueError(f"empty barcodes table: {path}")
    return frame.iloc[:, [0]].rename(columns={0: "barcode"}).fillna("")


def select_var_names(var: Any, mode: Literal["symbol", "id"]) -> list[str]:
    if mode == "id" and "gene_id" in var:
        values = var["gene_id"].astype(str).tolist()
    elif mode == "symbol" and "gene_symbol" in var:
        values = var["gene_symbol"].astype(str).tolist()
    else:
        values = var.iloc[:, 0].astype(str).tolist()

    fallback = var["gene_id"].astype(str).tolist() if "gene_id" in var else []
    cleaned: list[str] = []
    for i, value in enumerate(values):
        text = str(value).strip()
        if not text and i < len(fallback):
            text = str(fallback[i]).strip()
        cleaned.append(text or f"feature_{i}")
    return cleaned


def read_sample_metadata(source_dir: Path, *, pd: Any) -> dict[str, Any]:
    for name in ("sample_metadata.csv.gz", "sample_metadata.csv"):
        path = source_dir / name
        if not path.is_file():
            continue
        frame = pd.read_csv(path, compression="infer")
        if frame.empty:
            return {}
        row = frame.iloc[0].to_dict()
        return {str(key): json_scalar(value) for key, value in row.items()}
    return {}


def json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if value != value:  # NaN
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def zip_directory_stored(source_dir: Path, target_zip: Path) -> None:
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for path in sorted(p for p in source_dir.rglob("*") if p.is_file()):
            zf.write(path, path.relative_to(source_dir).as_posix(), compress_type=zipfile.ZIP_STORED)


def iter_parallel_results(
    tasks: list[ConvertTask],
    *,
    jobs: int,
    common: dict[str, Any],
) -> Iterable[ConvertResult]:
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        future_to_task = {pool.submit(convert_one_task, task, **common): task for task in tasks}
        for future in as_completed(future_to_task):
            yield future.result()


def append_result(log_dir: Path, result: ConvertResult) -> None:
    append_results(log_dir / f"{result.status}.tsv", [result])


def append_results(path: Path, rows: Iterable[ConvertResult]) -> None:
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
    return [
        "status",
        "source_dir",
        "target_zip",
        "sample_id",
        "n_obs",
        "n_vars",
        "nnz",
        "input_bytes",
        "output_bytes",
        "seconds",
        "message",
        "traceback",
    ]


def dry_run_result(task: ConvertTask) -> ConvertResult:
    return ConvertResult(
        status="dry_run",
        source_dir=task.source_dir,
        target_zip=task.target_zip,
        sample_id=sample_id_from_source_dir(Path(task.source_dir)),
    )


def print_result(result: ConvertResult) -> None:
    print(
        f"[{timestamp()}] {result.status}\t{result.sample_id}\t"
        f"{result.n_obs or ''}x{result.n_vars or ''}\t{result.message}",
        flush=True,
    )


def sample_id_from_source_dir(source_dir: Path) -> str:
    if source_dir.name in MATRIX_DIR_NAMES and source_dir.parent.name:
        return source_dir.parent.name
    return source_dir.name


def source_group_from_source_dir(source_dir: Path) -> str:
    parts = source_dir.parts
    for marker in ("20260616_public_collected", "20260625_M20_collected"):
        if marker in parts:
            idx = parts.index(marker)
            if marker == "20260625_M20_collected" and idx + 1 < len(parts):
                return f"{marker}/{parts[idx + 1]}"
            return marker
    return ""


def safe_name(value: str) -> str:
    value = value.strip().replace(os.sep, "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "sample"


def sum_input_bytes(source_dir: Path) -> int:
    return sum(path.stat().st_size for path in source_dir.iterdir() if path.is_file())


def remove_path(path: Path) -> None:
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    except FileNotFoundError:
        pass


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
