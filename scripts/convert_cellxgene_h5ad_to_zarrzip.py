#!/usr/bin/env python3
"""Convert CELLxGENE H5AD datasets with ``scdata.io.write_zarr``.

This script is intentionally a thin batch wrapper around ``write_zarr``.  It
does not hand-write zarr metadata or chunks.  The conversion path is:

1. read ``full.h5ad`` as a full AnnData object,
2. normalize expression-matrix dtypes for ``X``, ``layers`` and ``raw.X``,
3. sanitize zarr-hostile keys such as ``/`` while storing a reversible mapping
   in ``uns['scdata_key_mapping']``,
4. call ``scdata.io.write_zarr(..., format='sparse', align_cells=True)`` into a
   temporary directory,
5. pack that directory as a single ``full.zarr.zip`` file.

Sources are kept by default.  Pass ``--remove-source`` only after auditing the
converted stores.
"""

from __future__ import annotations

import argparse
import csv
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
from typing import Any, Iterable, Literal, Mapping

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("BLOSC_NTHREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT_ROOT = Path(
    "/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens"
)
DEFAULT_SOURCE_NAME = "full.h5ad"
DEFAULT_TARGET_NAME = "full.zarr.zip"
DEFAULT_COMPRESSOR = "blosc.lz4.level5"
DEFAULT_CHUNK_SIZE = 1_000_000
DEFAULT_SAMPLE_SIZE = 1_000_000

Status = Literal["converted", "skipped", "failed", "dry_run"]
DataDtype = Literal["auto", "preserve", "float32"]
LayerFormat = Literal["preserve", "auto", "dense2d", "dense1d", "sparse"]


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
    dataset_dir: str
    source_h5ad: str
    target_zip: str
    dataset_id: str


@dataclass(frozen=True)
class ConvertResult:
    status: Status
    dataset_dir: str
    source_h5ad: str
    target_zip: str
    dataset_id: str
    n_obs: int | None = None
    n_vars: int | None = None
    nnz: int | None = None
    x_dtype: str = ""
    layers: str = ""
    raw: str = ""
    input_bytes: int | None = None
    output_bytes: int | None = None
    seconds: float = 0.0
    message: str = ""
    traceback: str = ""


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    tasks = discover_tasks(
        input_root=input_root,
        manifest=args.manifest,
        source_name=args.source_name,
        target_name=args.target_name,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
    )
    tasks = apply_selection(
        tasks,
        start=args.start,
        limit=args.limit,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    log_dir = args.log_dir.resolve() if args.log_dir is not None else input_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.write_manifest is not None:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(
            "\n".join(task.source_h5ad for task in tasks) + ("\n" if tasks else ""),
            encoding="utf-8",
        )

    print(
        f"[{timestamp()}] tasks={len(tasks)} input_root={input_root} jobs={args.jobs} "
        f"chunk_size={args.chunk_size} compressor={args.compressor} "
        f"data_dtype={args.data_dtype} layer_format={args.layer_format} "
        f"remove_source={args.remove_source}",
        flush=True,
    )

    if args.dry_run:
        for task in tasks[: args.print_limit]:
            print(f"DRY_RUN\t{task.source_h5ad}\t{task.target_zip}")
        append_results(log_dir / "dry_run.tsv", (dry_run_result(task) for task in tasks))
        print(f"[{timestamp()}] dry-run listed {min(len(tasks), args.print_limit)} tasks")
        return 0

    common = {
        "chunk_size": args.chunk_size,
        "compressor": args.compressor,
        "data_dtype": args.data_dtype,
        "sample_size": args.sample_size,
        "layer_format": args.layer_format,
        "overwrite": args.overwrite,
        "remove_source": args.remove_source,
        "verify": not args.no_verify,
        "keep_zarr": args.keep_zarr,
        "keep_failed_zarr": args.keep_failed_zarr,
        "sanitize_keys": not args.no_sanitize_keys,
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
    parser.add_argument("--source-name", default=DEFAULT_SOURCE_NAME)
    parser.add_argument("--target-name", default=DEFAULT_TARGET_NAME)
    parser.add_argument("--manifest", type=Path, help="Text file with one dataset dir or H5AD path per line.")
    parser.add_argument("--write-manifest", type=Path, help="Write selected H5AD paths for audit.")
    parser.add_argument("--log-dir", type=Path, help="Directory for converted/skipped/failed TSV logs.")
    parser.add_argument("--include-regex", help="Only convert paths matching this regex.")
    parser.add_argument("--exclude-regex", help="Skip paths matching this regex.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent processes. Keep low for large H5ADs.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--compressor", default=DEFAULT_COMPRESSOR)
    parser.add_argument(
        "--data-dtype",
        choices=("auto", "preserve", "float32"),
        default="auto",
        help=(
            "Expression matrix dtype policy for X, layers and raw.X. "
            "'auto' stores count-like matrices as uint16/uint32 and other floats as float32."
        ),
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument(
        "--layer-format",
        choices=("preserve", "auto", "dense2d", "dense1d", "sparse"),
        default="preserve",
        help="Passed directly to scdata.io.write_zarr.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--remove-source", action="store_true", help="Delete full.h5ad after successful conversion.")
    parser.add_argument("--no-verify", action="store_true", help="Skip scdata.io.launch verification.")
    parser.add_argument("--keep-zarr", action="store_true", help="Keep intermediate .work.zarr directories after success.")
    parser.add_argument("--keep-failed-zarr", action="store_true", help="Keep intermediate zarr dirs on failure.")
    parser.add_argument("--no-sanitize-keys", action="store_true", help="Do not rename zarr-hostile keys containing '/'.")
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
        parser.error("--shard-index must satisfy 0 <= shard_index < num_shards")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.sample_size <= 0:
        parser.error("--sample-size must be positive")
    return args


def discover_tasks(
    *,
    input_root: Path,
    manifest: Path | None,
    source_name: str,
    target_name: str,
    include_regex: str | None,
    exclude_regex: str | None,
) -> list[ConvertTask]:
    if manifest is not None:
        paths = read_manifest(manifest, source_name=source_name)
    else:
        paths = sorted(input_root.glob(f"*/{source_name}"))

    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    tasks: list[ConvertTask] = []
    for source in paths:
        dataset_dir = source.parent
        text = str(source)
        if include and not include.search(text):
            continue
        if exclude and exclude.search(text):
            continue
        tasks.append(
            ConvertTask(
                dataset_dir=str(dataset_dir),
                source_h5ad=str(source),
                target_zip=str(dataset_dir / target_name),
                dataset_id=safe_name(dataset_dir.name),
            )
        )
    return tasks


def read_manifest(path: Path, *, source_name: str) -> list[Path]:
    paths: list[Path] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        p = Path(text)
        if not p.is_absolute():
            p = path.parent / p
        paths.append(p if p.name == source_name or p.suffix == ".h5ad" else p / source_name)
    return sorted(paths)


def apply_selection(
    tasks: list[ConvertTask],
    *,
    start: int,
    limit: int | None,
    num_shards: int,
    shard_index: int,
) -> list[ConvertTask]:
    selected = tasks[start:]
    if limit is not None:
        selected = selected[:limit]
    if num_shards > 1:
        selected = [task for i, task in enumerate(selected) if i % num_shards == shard_index]
    return selected


def convert_one_task(
    task: ConvertTask,
    *,
    chunk_size: int,
    compressor: str,
    data_dtype: DataDtype,
    sample_size: int,
    layer_format: LayerFormat,
    overwrite: bool,
    remove_source: bool,
    verify: bool,
    keep_zarr: bool,
    keep_failed_zarr: bool,
    sanitize_keys: bool,
) -> ConvertResult:
    started = time.perf_counter()
    dataset_dir = Path(task.dataset_dir)
    source_h5ad = Path(task.source_h5ad)
    target_zip = Path(task.target_zip)
    work_zarr = target_zip.parent / f".{target_zip.name}.work.zarr"
    tmp_zip = target_zip.parent / f".{target_zip.name}.tmp"

    if not source_h5ad.is_file():
        if target_zip.is_file():
            return ConvertResult(
                status="skipped",
                dataset_dir=str(dataset_dir),
                source_h5ad=str(source_h5ad),
                target_zip=str(target_zip),
                dataset_id=task.dataset_id,
                output_bytes=target_zip.stat().st_size,
                seconds=time.perf_counter() - started,
                message="source missing and target exists",
            )
        return ConvertResult(
            status="failed",
            dataset_dir=str(dataset_dir),
            source_h5ad=str(source_h5ad),
            target_zip=str(target_zip),
            dataset_id=task.dataset_id,
            seconds=time.perf_counter() - started,
            message="source H5AD missing",
        )

    if target_zip.exists() and not overwrite:
        return ConvertResult(
            status="skipped",
            dataset_dir=str(dataset_dir),
            source_h5ad=str(source_h5ad),
            target_zip=str(target_zip),
            dataset_id=task.dataset_id,
            input_bytes=source_h5ad.stat().st_size,
            output_bytes=target_zip.stat().st_size,
            seconds=time.perf_counter() - started,
            message="target exists",
        )

    try:
        remove_path(work_zarr)
        remove_path(tmp_zip)
        target_zip.parent.mkdir(parents=True, exist_ok=True)

        adata = read_and_prepare_anndata(
            source_h5ad,
            data_dtype=data_dtype,
            sample_size=sample_size,
            sanitize_keys=sanitize_keys,
        )
        n_obs = int(adata.n_obs)
        n_vars = int(adata.n_vars)
        nnz = int(getattr(adata.X, "nnz", 0))
        x_dtype = matrix_dtype_name(adata.X)
        layers = ",".join(sorted(map(str, adata.layers.keys())))
        raw = "present" if adata.raw is not None else "missing"

        from scdata.io import launch, write_zarr

        write_zarr(
            adata,
            work_zarr,
            format="sparse",
            layer_format=layer_format,
            chunk_size=chunk_size,
            align_cells=True,
            store="dir",
            compressor=compressor,
        )
        zip_directory_stored(work_zarr, tmp_zip)
        if verify:
            # Verify the staged zip before publishing it so a failed conversion
            # never leaves an unvalidated target_zip behind (which would make
            # the next run skip it as "target exists").
            ds = launch(tmp_zip)
            if ds.num_cells != n_obs or ds.num_genes != n_vars:
                raise RuntimeError(
                    f"launch shape mismatch: got {(ds.num_cells, ds.num_genes)}, "
                    f"expected {(n_obs, n_vars)}"
                )
        os.replace(tmp_zip, target_zip)

        input_bytes = source_h5ad.stat().st_size
        output_bytes = target_zip.stat().st_size
        if remove_source:
            source_h5ad.unlink()
        if not keep_zarr:
            remove_path(work_zarr)
        return ConvertResult(
            status="converted",
            dataset_dir=str(dataset_dir),
            source_h5ad=str(source_h5ad),
            target_zip=str(target_zip),
            dataset_id=task.dataset_id,
            n_obs=n_obs,
            n_vars=n_vars,
            nnz=nnz,
            x_dtype=x_dtype,
            layers=layers,
            raw=raw,
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
            dataset_dir=str(dataset_dir),
            source_h5ad=str(source_h5ad),
            target_zip=str(target_zip),
            dataset_id=task.dataset_id,
            input_bytes=source_h5ad.stat().st_size if source_h5ad.exists() else None,
            seconds=time.perf_counter() - started,
            message=f"{type(err).__name__}: {err}",
            traceback=traceback.format_exc(),
        )


def read_and_prepare_anndata(
    source_h5ad: Path,
    *,
    data_dtype: DataDtype,
    sample_size: int,
    sanitize_keys: bool,
) -> Any:
    import anndata as ad

    adata = ad.read_h5ad(source_h5ad)
    coerce_expression_matrices(adata, data_dtype=data_dtype, sample_size=sample_size)
    if sanitize_keys:
        sanitize_anndata_keys(adata)
    metadata_path = source_h5ad.parent / "metadata.json"
    if metadata_path.is_file():
        adata.uns["cellxgene_metadata_json"] = metadata_path.read_text(encoding="utf-8")
    return adata


def coerce_expression_matrices(adata: Any, *, data_dtype: DataDtype, sample_size: int) -> None:
    if data_dtype == "preserve":
        return
    adata.X = coerce_matrix_dtype(adata.X, data_dtype=data_dtype, sample_size=sample_size)
    for key in list(adata.layers.keys()):
        adata.layers[key] = coerce_matrix_dtype(
            adata.layers[key],
            data_dtype=data_dtype,
            sample_size=sample_size,
        )
    if adata.raw is not None:
        import anndata as ad
        import pandas as pd

        raw_x = coerce_matrix_dtype(adata.raw.X, data_dtype=data_dtype, sample_size=sample_size)
        raw_obs = pd.DataFrame(index=adata.obs_names.copy())
        raw_adata = ad.AnnData(X=raw_x, obs=raw_obs, var=adata.raw.var.copy())
        for key, value in dict(adata.raw.varm).items():
            raw_adata.varm[key] = value
        adata.raw = raw_adata


def coerce_matrix_dtype(matrix: Any, *, data_dtype: DataDtype, sample_size: int) -> Any:
    import numpy as np
    from scipy import sparse

    if data_dtype == "preserve":
        return matrix
    if sparse.issparse(matrix):
        matrix = matrix.tocsr(copy=False)
        target = (
            np.dtype("float32")
            if data_dtype == "float32"
            else infer_target_dtype(matrix.data, sample_size=sample_size, np=np)
        )
        if target == matrix.data.dtype:
            return matrix
        out = matrix.copy()
        out.data = coerce_values(out.data, target=target, np=np)
        return out

    arr = np.asarray(matrix)
    target = np.dtype("float32") if data_dtype == "float32" else infer_target_dtype(arr, sample_size=sample_size, np=np)
    if target == arr.dtype:
        return arr
    return coerce_values(arr, target=target, np=np)


def infer_target_dtype(values: Any, *, sample_size: int, np: Any) -> Any:
    arr = np.asarray(values)
    if arr.size == 0:
        return np.dtype("uint16")
    if np.issubdtype(arr.dtype, np.complexfloating):
        raise ValueError(f"complex expression matrix dtype is unsupported: {arr.dtype}")
    if np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.dtype("bool"):
        min_value, max_value = integer_range(arr, np=np)
        if min_value < 0:
            return np.dtype("float32")
        return count_dtype(max_value, np=np)
    if np.issubdtype(arr.dtype, np.floating):
        sample = sample_values(arr, sample_size=sample_size, np=np)
        if not is_nonnegative_integer(sample, np=np):
            return np.dtype("float32")
        if not is_nonnegative_integer(arr, np=np):
            return np.dtype("float32")
        _min_value, max_value = integer_range(arr, np=np)
        return count_dtype(max_value, np=np)
    raise ValueError(f"unsupported expression matrix dtype: {arr.dtype}")


def sample_values(arr: Any, *, sample_size: int, np: Any) -> Any:
    flat = np.asarray(arr).ravel()
    if flat.size <= sample_size:
        return flat
    window = min(100_000, sample_size)
    segments = max(1, sample_size // window)
    starts = np.linspace(0, flat.size - window, num=segments, dtype=np.int64)
    return np.concatenate([flat[int(start) : int(start) + window] for start in starts])


def is_nonnegative_integer(values: Any, *, np: Any) -> bool:
    arr = np.asarray(values)
    if arr.size == 0:
        return True
    if np.issubdtype(arr.dtype, np.floating):
        return bool(np.all(np.isfinite(arr)) and np.all(arr >= 0) and np.all(arr == np.trunc(arr)))
    if np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.dtype("bool"):
        return bool(np.all(arr >= 0))
    return False


def integer_range(values: Any, *, np: Any) -> tuple[int, int]:
    arr = np.asarray(values)
    if arr.size == 0:
        return 0, 0
    return int(arr.min()), int(arr.max())


def count_dtype(max_value: int, *, np: Any) -> Any:
    if max_value <= np.iinfo(np.uint16).max:
        return np.dtype("uint16")
    if max_value <= np.iinfo(np.uint32).max:
        return np.dtype("uint32")
    raise OverflowError(f"count value {max_value} exceeds uint32")


def coerce_values(values: Any, *, target: Any, np: Any) -> Any:
    arr = np.asarray(values)
    target = np.dtype(target)
    if np.issubdtype(target, np.unsignedinteger):
        if not is_nonnegative_integer(arr, np=np):
            raise ValueError(f"cannot cast non-count values to {target.name}")
        if arr.size and int(arr.max()) > np.iinfo(target).max:
            raise OverflowError(f"value {int(arr.max())} exceeds {target.name}")
    return arr.astype(target, copy=False)


def sanitize_anndata_keys(adata: Any) -> None:
    mapping: dict[str, dict[str, str]] = {}
    sanitize_dataframe_axis(adata.obs, "obs", mapping)
    sanitize_dataframe_axis(adata.var, "var", mapping)
    sanitize_axis_mapping(adata.layers, "layers", mapping)
    sanitize_axis_mapping(adata.obsm, "obsm", mapping)
    sanitize_axis_mapping(adata.varm, "varm", mapping)
    sanitize_axis_mapping(adata.obsp, "obsp", mapping)
    sanitize_axis_mapping(adata.varp, "varp", mapping)

    raw_replacement = None
    if adata.raw is not None:
        import anndata as ad
        import pandas as pd

        raw_var = adata.raw.var.copy()
        raw_varm = dict(adata.raw.varm)
        sanitize_dataframe_axis(raw_var, "raw.var", mapping)
        raw_varm = sanitized_mapping_copy(raw_varm, "raw.varm", mapping)
        raw_obs = pd.DataFrame(index=adata.obs_names.copy())
        raw_replacement = ad.AnnData(X=adata.raw.X, obs=raw_obs, var=raw_var)
        for key, value in raw_varm.items():
            raw_replacement.varm[key] = value

    uns = sanitize_uns_mapping(dict(adata.uns), "uns", mapping)
    if mapping:
        uns["scdata_key_mapping"] = mapping_records(mapping)
    adata.uns.clear()
    adata.uns.update(uns)
    if raw_replacement is not None:
        adata.raw = raw_replacement


def sanitize_dataframe_axis(frame: Any, prefix: str, mapping: dict[str, dict[str, str]]) -> None:
    rename = unique_renames(list(map(str, frame.columns)))
    if rename:
        frame.rename(columns=rename, inplace=True)
        mapping[prefix + ".columns"] = rename
    index_name = frame.index.name
    if index_name is not None:
        safe = safe_zarr_key(str(index_name))
        if safe != str(index_name):
            frame.index.name = safe
            mapping[prefix + ".index_name"] = {str(index_name): safe}


def sanitize_axis_mapping(container: Any, prefix: str, mapping: dict[str, dict[str, str]]) -> None:
    rename = unique_renames(list(map(str, container.keys())))
    if not rename:
        return
    values = {rename.get(str(key), str(key)): container[key] for key in list(container.keys())}
    container.clear()
    for key, value in values.items():
        container[key] = value
    mapping[prefix] = rename


def sanitized_mapping_copy(
    source: Mapping[str, Any],
    prefix: str,
    mapping: dict[str, dict[str, str]],
) -> dict[str, Any]:
    rename = unique_renames(list(map(str, source.keys())))
    if rename:
        mapping[prefix] = rename
    return {rename.get(str(key), str(key)): value for key, value in source.items()}


def sanitize_uns_mapping(
    source: Mapping[str, Any],
    prefix: str,
    mapping: dict[str, dict[str, str]],
) -> dict[str, Any]:
    rename = unique_renames(list(map(str, source.keys())))
    if rename:
        mapping[prefix] = rename
    out: dict[str, Any] = {}
    for key, value in source.items():
        new_key = rename.get(str(key), str(key))
        if isinstance(value, Mapping):
            out[new_key] = sanitize_uns_mapping(value, f"{prefix}.{new_key}", mapping)
        else:
            out[new_key] = value
    return out


def mapping_records(mapping: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    locations: list[str] = []
    originals: list[str] = []
    stored: list[str] = []
    for location, renames in sorted(mapping.items()):
        for original, stored_name in sorted(renames.items()):
            locations.append(location)
            originals.append(original)
            stored.append(stored_name)
    return {"location": locations, "original": originals, "stored": stored}


def unique_renames(names: list[str]) -> dict[str, str]:
    used = set(names)
    rename: dict[str, str] = {}
    for name in names:
        safe = safe_zarr_key(name)
        if safe == name and names.count(name) == 1:
            continue
        candidate = safe
        i = 1
        while candidate in used and candidate != name:
            candidate = f"{safe}_{i}"
            i += 1
        if candidate != name:
            rename[name] = candidate
            used.add(candidate)
    return rename


def safe_zarr_key(name: str) -> str:
    return name.replace("%", "%25").replace("/", "%2F") or "empty"


def matrix_dtype_name(matrix: Any) -> str:
    import numpy as np
    from scipy import sparse

    if sparse.issparse(matrix):
        return np.dtype(matrix.data.dtype).name
    return np.dtype(np.asarray(matrix).dtype).name


def zip_directory_stored(source_dir: Path, target_zip: Path) -> None:
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
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
        "dataset_dir",
        "source_h5ad",
        "target_zip",
        "dataset_id",
        "n_obs",
        "n_vars",
        "nnz",
        "x_dtype",
        "layers",
        "raw",
        "input_bytes",
        "output_bytes",
        "seconds",
        "message",
        "traceback",
    ]


def dry_run_result(task: ConvertTask) -> ConvertResult:
    return ConvertResult(
        status="dry_run",
        dataset_dir=task.dataset_dir,
        source_h5ad=task.source_h5ad,
        target_zip=task.target_zip,
        dataset_id=task.dataset_id,
    )


def print_result(result: ConvertResult) -> None:
    shape = f"{result.n_obs or ''}x{result.n_vars or ''}"
    print(
        f"[{timestamp()}] {result.status:<9} {result.dataset_id}\t{shape}\t"
        f"{result.x_dtype}\t{result.message}",
        flush=True,
    )


def safe_name(value: str) -> str:
    value = value.strip().replace(os.sep, "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "dataset"


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
