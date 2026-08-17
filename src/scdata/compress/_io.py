"""High-level open/write entry points over NumPy and SciPy objects."""

from __future__ import annotations

import zipfile
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from scdata import _core
from scdata.compress._base import open_matrix
from scdata.compress._csr import ScCsr
from scdata.compress._dense import ScDense
from scdata.compress._limits import ReadLimits, resolve_read_limits
from scdata.compress._validate import (
    ensure_path,
    ensure_writable_path,
    is_sparse_matrix,
    normalize_csr_arrays,
    normalize_dense,
    require_scipy_csr,
)
from scdata.compress._codec import Codec, _codec_to_wire
from scdata.compress._write_options import WriteOptions, resolve_write_options


__all__ = [
    "open_store",
    "write",
    "write_csr",
    "write_csr_arrays",
    "write_dense",
]


def write(
    path: str | PathLike[str],
    matrix: Any,
    *,
    options: WriteOptions | None = None,
    codec: Codec | str | dict[str, Any] | None = None,
    indptr_codec: Codec | str | dict[str, Any] | None = None,
    num_workers: int | None = None,
    overwrite: bool = True,
) -> None:
    """Write a dense array-like or SciPy sparse matrix.

    Sparse inputs are normalized to canonical CSR. All other inputs are passed
    to :func:`write_dense` and coerced with :func:`numpy.asarray`.

    Partitioning follows :class:`~scdata.WriteOptions`: ``policy='cells'``
    → ``fixed_cells``; ``policy='budget'`` → ``bytes_budget`` for CSR. Dense
    ``budget`` policies are lowered in Python to ``fixed_cells``.
    """
    if is_sparse_matrix(matrix):
        write_csr(
            path,
            matrix,
            options=options,
            codec=codec,
            indptr_codec=indptr_codec,
            num_workers=num_workers,
            overwrite=overwrite,
        )
        return
    if indptr_codec is not None:
        raise ValueError("indptr_codec applies only to CSR inputs, not dense matrices")
    write_dense(
        path,
        matrix,
        options=options,
        codec=codec,
        num_workers=num_workers,
        overwrite=overwrite,
    )


def write_dense(
    path: str | PathLike[str],
    values: Any,
    *,
    options: WriteOptions | None = None,
    codec: Codec | str | dict[str, Any] | None = None,
    num_workers: int | None = None,
    overwrite: bool = True,
) -> None:
    """Write a dense matrix after NumPy dtype/layout normalization."""
    path_obj = ensure_path(path)
    ensure_writable_path(path_obj, overwrite=overwrite)
    opts = resolve_write_options(options, codec=codec, num_workers=num_workers)
    array = normalize_dense(values)
    row_bytes = int(array.shape[1]) * int(array.dtype.itemsize)
    chunk, block = opts.resolve(dense=True, row_bytes=row_bytes)
    _core.write_dense(
        str(path_obj),
        array,
        chunk_policy=chunk.policy,
        chunk_n=chunk.n,
        block_policy=block.policy,
        block_n=block.n,
        num_workers=opts.worker_count,
        compressor=_codec_to_wire(opts.resolved_codec(), role="dense"),
    )


def write_csr_arrays(
    path: str | PathLike[str],
    indptr: Any,
    indices: Any,
    data: Any,
    shape: tuple[int, int] | list[int],
    *,
    options: WriteOptions | None = None,
    codec: Codec | str | dict[str, Any] | None = None,
    indptr_codec: Codec | str | dict[str, Any] | None = None,
    num_workers: int | None = None,
    overwrite: bool = True,
) -> None:
    """Write explicit CSR buffers after structural and precision checks."""
    path_obj = ensure_path(path)
    ensure_writable_path(path_obj, overwrite=overwrite)
    opts = resolve_write_options(
        options,
        codec=codec,
        indptr_codec=indptr_codec,
        num_workers=num_workers,
    )
    chunk, block = opts.resolve(dense=False)
    indptr_u64, indices_u64, data_array, (n_rows, n_cols) = normalize_csr_arrays(
        indptr,
        indices,
        data,
        shape,
    )
    _core.write_csr(
        str(path_obj),
        indptr_u64,
        indices_u64,
        data_array,
        n_rows,
        n_cols,
        chunk_policy=chunk.policy,
        chunk_n=chunk.n,
        block_policy=block.policy,
        block_n=block.n,
        num_workers=opts.worker_count,
        compressor=_codec_to_wire(opts.resolved_codec(), role="csr"),
        indptr_compressor=_codec_to_wire(opts.resolved_indptr_codec(), role="indptr"),
    )


def write_csr(
    path: str | PathLike[str],
    matrix: Any,
    *,
    options: WriteOptions | None = None,
    codec: Codec | str | dict[str, Any] | None = None,
    indptr_codec: Codec | str | dict[str, Any] | None = None,
    num_workers: int | None = None,
    overwrite: bool = True,
) -> None:
    """Write a SciPy sparse matrix as canonical CSR.

    Duplicate coordinates are summed and column indices are sorted on a copy
    when the input is not already canonical.
    """
    path_obj = ensure_path(path)
    ensure_writable_path(path_obj, overwrite=overwrite)
    opts = resolve_write_options(
        options,
        codec=codec,
        indptr_codec=indptr_codec,
        num_workers=num_workers,
    )
    opts.resolve(dense=False)
    csr = require_scipy_csr(matrix)
    if not getattr(csr, "has_canonical_format", False):
        csr = csr.copy()
        csr.sum_duplicates()
        csr.sort_indices()
    write_csr_arrays(
        path_obj,
        np.asarray(csr.indptr),
        np.asarray(csr.indices),
        np.asarray(csr.data),
        tuple(csr.shape),
        options=opts,
        overwrite=overwrite,
    )


def open_store(
    path: str | PathLike[str] | zipfile.ZipFile,
    *,
    zip_prefix: str | None = None,
    limits: ReadLimits | None = None,
    max_metadata_size: int | None = None,
    max_encoded_size: int | None = None,
    max_decoded_size: int | None = None,
    max_block_count: int | None = None,
    num_workers: int | None = None,
) -> ScDense | ScCsr:
    """Open a directory store or a store inside a ZIP archive.

    A ZIP containing exactly one sc-compress prefix opens without
    ``zip_prefix``. For archives with multiple stores, the error lists the
    available prefixes. Resource keyword overrides apply on top of ``limits``
    or :data:`scdata.DEFAULT_READ_LIMITS`.
    """
    from scdata.compress import _zip as zip_api

    read_limits = resolve_read_limits(
        limits,
        max_metadata_size=max_metadata_size,
        max_encoded_size=max_encoded_size,
        max_decoded_size=max_decoded_size,
        max_block_count=max_block_count,
        num_workers=num_workers,
    )
    path_obj, resolved_prefix = _resolve_location(path, zip_prefix, zip_api)
    handle = _core.store_open(
        str(path_obj),
        zip_prefix=resolved_prefix,
        max_metadata_size=read_limits.max_metadata_size,
        max_encoded_size=read_limits.max_encoded_size,
        max_decoded_size=read_limits.max_decoded_size,
        max_block_count=read_limits.max_block_count,
        num_workers=read_limits.num_workers,
    )
    return open_matrix(handle, path_obj, resolved_prefix)


def _resolve_location(
    source: str | PathLike[str] | zipfile.ZipFile,
    zip_prefix: str | None,
    zip_api: Any,
) -> tuple[Path, str | None]:
    if zip_prefix is not None and not isinstance(zip_prefix, str):
        raise TypeError(f"zip_prefix must be str or None, got {type(zip_prefix).__name__}")

    if isinstance(source, zipfile.ZipFile):
        if source.mode != "r" and source.fp is not None:
            raise ValueError("ZipFile passed to open_store() must be opened in mode 'r' or closed")
        path = zip_api.archive_path(source)
        archive = True
    else:
        path = ensure_path(source)
        archive = zip_prefix is not None
        if not archive and not path.is_dir():
            archive = path.suffix.casefold() == ".zip"
            if path.is_file() and not archive:
                archive = zipfile.is_zipfile(path)

    if not archive:
        if path.exists() and not path.is_dir():
            raise ValueError(f"path is neither a directory nor a ZIP archive: {path}")
        return path, None

    if zip_prefix is not None:
        return path, zip_api.normalize_prefix(zip_prefix)

    if not path.exists():
        # Preserve the Rust error hierarchy for a missing archive.
        return path, ""
    prefixes = zip_api.list_stores(path)
    if len(prefixes) == 1:
        return path, prefixes[0]
    if not prefixes:
        raise ValueError(f"ZIP archive contains no sc-compress stores: {path}")
    choices = ", ".join(repr(prefix) for prefix in prefixes)
    raise ValueError(
        f"ZIP archive contains multiple sc-compress stores; "
        f"pass zip_prefix=... (available: {choices})"
    )
