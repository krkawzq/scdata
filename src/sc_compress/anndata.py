"""AnnData bridge for sc-compress (``.scc`` / ``.scc.zip``).

Layout is zarr-v3 / AnnData shaped.  Annotations (``obs`` / ``var`` / ``uns`` /
``varm`` / ``varp`` / …) use AnnData's own element writers.  Supported numeric,
**cell-aligned** matrices are stored as sc-compress:

* ``X``, ``layers/*``, ``raw/X``
* ``obsm/*``, ``obsp/*``

Cell axis must be the leftmost or rightmost dimension.  Rank-2 matrices with
cells on the left keep their dense/CSR form; higher-rank (or cell-on-right)
arrays are densified, cell axis moved to front, and the non-cell dimensions
flattened to a single column axis.  Python reshapes back on read.
AnnData-native containers such as a :class:`pandas.DataFrame` in ``obsm`` stay
on AnnData's zarr path instead of losing their tabular representation.

Containers:

* directory store (typically ``*.scc``)
* ``ZIP_STORED`` archive (typically ``*.scc.zip``)

:func:`read_scc` auto-detects the container.  :func:`write_scc` defaults to
``store="auto"`` (``.zip`` suffix → zip, otherwise directory).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import warnings
import zipfile
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from copy import copy as shallow_copy
from operator import index
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast

import numpy as np

from sc_compress._validate import ensure_path, ensure_writable_path, is_sparse_matrix
from sc_compress.exceptions import InvalidMetaError, PerformanceWarning, _invalid_argument
from sc_compress.format import VALUE_DTYPES, is_value_dtype
from sc_compress.io import write as write_matrix
from sc_compress.limits import ReadLimits, resolve_read_limits
from sc_compress.write_options import WriteOptions, resolve_write_options

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["read_scc", "write_scc"]

_Store = Literal["zip", "dir", "auto"]
ProgressCallback = Callable[[str, int, int], None]

# Marker on the zarr group that wraps an sc-compress matrix directory.
# ``encoding-type`` must be a type AnnData already registers a Group reader for
# (``get_read`` runs *before* ``read_dispatched`` callbacks).  ``dict`` is the
# lightest fit; ``sc-compress: true`` is what identifies the node as a matrix.
_SCC_FLAG_ATTR = "sc-compress"
_SCC_FORMAT_ATTR = "scc-format"
_SCC_FORMAT_VERSION = "0.1.0"
_SCC_SHAPE_ATTR = "shape"
_SCC_CELL_AXIS_ATTR = "cell-axis"
_UINT64_MAX = int(np.iinfo(np.uint64).max)

# Paths whose values are cell-aligned and therefore eligible for scc.
_CELL_MATRIX_PARENTS = frozenset({"layers", "obsm", "obsp"})

# Top-level AnnData slots that :func:`read_scc` can selectively load.
_LOADABLE_KEYS = frozenset(
    {"X", "layers", "raw", "obs", "var", "uns", "obsm", "obsp", "varm", "varp"}
)


# ---------------------------------------------------------------------------
# public write
# ---------------------------------------------------------------------------


def write_scc(
    adata: AnnData,
    path: str | os.PathLike[str],
    *,
    store: _Store = "auto",
    options: WriteOptions | None = None,
    n_workers: int | None = None,
    overwrite: bool = True,
    convert_strings_to_categoricals: bool = True,
    progress: ProgressCallback | None = None,
) -> Path:
    """Write an :class:`~anndata.AnnData` as a ``.scc`` directory or ``.scc.zip``.

    Parameters
    ----------
    adata
        Object to write.  Supported numeric cell-aligned matrices (``X``,
        ``layers``, ``raw.X``, ``obsm``, ``obsp``) go through sc-compress;
        everything else uses AnnData zarr writers (including ``uns`` / ``varm`` /
        ``varp`` and DataFrame-valued ``obsm`` entries).  Unsupported expression
        dtypes fail before the destination is touched and list accepted dtypes.
    path
        Destination path.  No suffix is appended — the caller picks the name
        (e.g. ``data.scc`` or ``data.scc.zip``).
    store
        ``"zip"`` writes a ``ZIP_STORED`` archive, ``"dir"`` a directory tree,
        ``"auto"`` (default) picks zip when ``path`` ends with ``.zip``.
    options
        Chunk/block partition knobs applied to every sc-compress matrix.
        Dense payloads require ``cells`` policies; CSR may use ``budget``.
    n_workers
        Per-matrix chunk workers. Overrides ``options.n_workers`` when provided.
    overwrite
        When false, refuse to replace an existing ``path``.
    convert_strings_to_categoricals
        Match :meth:`anndata.AnnData.write_zarr`: convert string columns in
        ``obs`` / ``var`` (and ``raw.var``) before writing.  When true, mutable
        annotations are copied while matrix payloads remain shared read-only, so
        the caller's object is not mutated or duplicated in full.
    progress
        Optional ``callback(name, index, total)`` invoked after each
        sc-compress matrix is written (``index`` is 1-based).
    """
    ad = _require_anndata()
    import zarr

    from anndata.experimental import write_dispatched

    if not isinstance(adata, ad.AnnData):
        _invalid_argument(f"adata must be an AnnData, got {type(adata).__name__}")
    if not isinstance(overwrite, (bool, np.bool_)):
        _invalid_argument(f"overwrite must be bool, got {type(overwrite).__name__}")
    if not isinstance(convert_strings_to_categoricals, (bool, np.bool_)):
        _invalid_argument(
            "convert_strings_to_categoricals must be bool, "
            f"got {type(convert_strings_to_categoricals).__name__}"
        )
    if progress is not None and not callable(progress):
        _invalid_argument(f"progress must be callable or None, got {type(progress).__name__}")
    overwrite = bool(overwrite)
    convert_strings_to_categoricals = bool(convert_strings_to_categoricals)

    root = ensure_path(path)
    store_kind = _resolve_store_kind(root, store)
    ensure_writable_path(root, overwrite=overwrite)
    write_opts = resolve_write_options(options, n_workers=n_workers)
    n_cells = int(adata.n_obs)
    has_dense_matrix = _preflight_write_adata(adata)
    write_opts.resolve(dense=has_dense_matrix)
    matrix_total = _count_scc_matrices(adata)
    if store_kind == "zip":
        _prepare_zip_target(root)

    if convert_strings_to_categoricals:
        adata = _copy_for_categorical_conversion(adata)

    staging_root = _make_temp_dir(root)
    cleanup_root: Path | None = staging_root
    written = 0

    try:
        with _anndata_zarr_context(), _write_config():
            g = zarr.open_group(str(staging_root), mode="w", zarr_format=3)
            g.attrs.setdefault("encoding-type", "anndata")
            g.attrs.setdefault("encoding-version", "0.1.0")

            def callback(
                write_func: Any,
                store: Any,
                elem_name: str,
                elem: Any,
                *,
                dataset_kwargs: Any,
                iospec: Any,
            ) -> None:
                nonlocal written
                del iospec
                rel = _matrix_relpath(store, elem_name)
                if _is_cell_matrix_path(rel) and _is_scc_matrix(elem):
                    _write_scc_matrix(
                        store,
                        elem_name,
                        elem,
                        root=staging_root,
                        n_cells=n_cells,
                        options=write_opts,
                    )
                    written += 1
                    if progress is not None:
                        progress(rel, written, matrix_total)
                    return
                write_func(
                    store,
                    elem_name,
                    elem,
                    dataset_kwargs=dataset_kwargs,
                )

            write_dispatched(g, "/", adata, callback=callback)

        if store_kind == "zip":
            _zip_directory(staging_root, root, overwrite=overwrite)
        else:
            _replace_directory(staging_root, root, overwrite=overwrite)
            cleanup_root = None
    finally:
        if cleanup_root is not None:
            _remove_path(cleanup_root)
    return root


def _copy_for_categorical_conversion(adata: AnnData) -> AnnData:
    """Copy mutable annotations while sharing matrix payloads read-only."""
    working = shallow_copy(adata)
    private_working = cast(Any, working)
    private_working._obs = adata.obs.copy()
    private_working._var = adata.var.copy()

    frames: list[Any] = [working.obs, working.var]
    if adata.raw is not None:
        raw = shallow_copy(adata.raw)
        private_raw = cast(Any, raw)
        private_raw._var = adata.raw.var.copy()
        private_working._raw = raw
        frames.append(raw.var)

    for frame in frames:
        working.strings_to_categoricals(frame)
    return working


# ---------------------------------------------------------------------------
# public read
# ---------------------------------------------------------------------------


def read_scc(
    path: str | os.PathLike[str],
    *,
    include: Collection[str] | None = None,
    exclude: Collection[str] | None = None,
    limits: ReadLimits | None = None,
    max_metadata_size: int | None = None,
    max_encoded_size: int | None = None,
    max_decoded_size: int | None = None,
    max_block_count: int | None = None,
    n_workers: int | None = None,
) -> AnnData:
    """Read a ``.scc`` directory or ``.scc.zip`` archive into AnnData.

    Parameters
    ----------
    path
        Store path.  A regular file is opened as a ZIP archive; a directory is
        opened as a zarr-v3 tree with embedded sc-compress matrices.
    include / exclude
        Optional filters over top-level AnnData slots
        (``X``, ``layers``, ``raw``, ``obs``, ``var``, ``uns``, ``obsm``,
        ``obsp``, ``varm``, ``varp``).  Pass at most one of them.  Excluding
        ``X`` / ``layers`` / ``raw`` yields empty placeholders (``X=None``,
        ``layers={}``, no ``raw``) because AnnData cannot keep ``raw`` without
        an in-memory ``X``.  Example equivalent of a metadata-focused read::

            read_scc(path, exclude=("X", "layers", "raw"))
    limits
        Reusable sc-compress resource limits for every embedded matrix.
    max_metadata_size / max_encoded_size / max_decoded_size / max_block_count
        Per-call overrides applied on top of ``limits``.  These match
        :func:`sc_compress.open_store` and are useful for large AnnData matrices.
    n_workers
        Per-matrix chunk workers. Overrides ``limits.n_workers`` when provided.
    """
    import zarr

    ad = _require_anndata()
    from anndata._io.specs import read_elem
    from anndata._io.utils import _read_legacy_raw
    from anndata._io.zarr import read_dataframe
    from anndata.compat import _clean_uns
    from anndata.experimental import read_dispatched

    root = ensure_path(path)
    load_keys = _resolve_load_keys(include, exclude)
    read_limits = resolve_read_limits(
        limits,
        max_metadata_size=max_metadata_size,
        max_encoded_size=max_encoded_size,
        max_decoded_size=max_decoded_size,
        max_block_count=max_block_count,
        n_workers=n_workers,
    )

    with _anndata_zarr_context():
        f, archive_path = _open_store_for_read(root)

        try:

            def callback(
                read_func: Any,
                elem_name: str,
                elem: Any,
                *,
                iospec: Any,
            ) -> Any:
                name = elem_name.lstrip("/")

                if _is_scc_node(elem):
                    if not _should_load_matrix(name, load_keys):
                        return None
                    return _read_scc_matrix(
                        root,
                        name,
                        archive_path,
                        attrs=_node_attrs(elem),
                        limits=read_limits,
                    )

                if iospec.encoding_type == "anndata" or elem_name.endswith("/"):
                    kwargs: dict[str, Any] = {}
                    for key, child in dict(elem).items():
                        if key.startswith("raw."):
                            continue
                        if key == "X" and "X" not in load_keys:
                            kwargs["X"] = None
                            continue
                        if key == "layers" and "layers" not in load_keys:
                            kwargs["layers"] = {}
                            continue
                        if key == "raw" and "raw" not in load_keys:
                            if any(child_key.startswith("raw.") for child_key in f):
                                _invalid_meta(
                                    "store contains both a modern 'raw' group and legacy 'raw.*' keys"
                                )
                            continue
                        if key in _LOADABLE_KEYS and key not in load_keys:
                            continue
                        kwargs[key] = read_dispatched(child, callback=callback)
                    return ad.AnnData(**kwargs)

                if elem_name.startswith("/raw."):
                    return None

                if elem_name in {"/obs", "/var"}:
                    return read_dataframe(elem)

                if elem_name == "/raw" or iospec.encoding_type == "raw":
                    if any(key.startswith("raw.") for key in f):
                        _invalid_meta(
                            "store contains both a modern 'raw' group and legacy 'raw.*' keys"
                        )
                    if "raw" not in load_keys:
                        return None
                    return read_func(elem)

                return read_func(elem)

            result = read_dispatched(f, callback=callback)
            if not isinstance(result, ad.AnnData):
                _invalid_meta(f"expected an AnnData at the store root, got {type(result).__name__}")
            adata = cast("AnnData", result)

            if "raw" in load_keys and "raw.X" in f:

                def read_legacy_elem(elem: Any) -> Any:
                    if not _is_scc_node(elem):
                        return read_elem(elem)
                    name = str(getattr(elem, "name", "raw.X")).lstrip("/")
                    return _read_scc_matrix(
                        root,
                        name,
                        archive_path,
                        attrs=_node_attrs(elem),
                        limits=read_limits,
                    )

                raw_kwargs = _read_legacy_raw(
                    f,
                    None,
                    read_dataframe,
                    read_legacy_elem,
                )
                raw = ad.AnnData(**raw_kwargs)
                raw.obs_names = adata.obs_names
                adata.raw = raw

            if "obs" in f and isinstance(f["obs"], zarr.Array):
                _clean_uns(adata)

            return adata
        finally:
            close = getattr(getattr(f, "store", None), "close", None)
            if close is not None:
                close()


def _resolve_load_keys(
    include: Collection[str] | None,
    exclude: Collection[str] | None,
) -> frozenset[str]:
    if include is not None and exclude is not None:
        _invalid_argument("pass only one of include= or exclude=")
    if include is None and exclude is None:
        return _LOADABLE_KEYS

    requested = include if include is not None else exclude
    if isinstance(requested, str):
        values: tuple[Any, ...] = (requested,)
    else:
        try:
            values = tuple(requested)  # type: ignore[arg-type]
        except TypeError:
            parameter = "include" if include is not None else "exclude"
            _invalid_argument(f"{parameter} must be a collection of AnnData slot names")
    invalid = [value for value in values if not isinstance(value, str)]
    if invalid:
        parameter = "include" if include is not None else "exclude"
        preview = ", ".join(repr(value) for value in invalid[:3])
        _invalid_argument(f"{parameter} entries must be strings, got {preview}")
    selected = frozenset(cast("tuple[str, ...]", values))
    unknown = selected - _LOADABLE_KEYS
    if unknown:
        preview = ", ".join(repr(name) for name in sorted(unknown))
        allowed = ", ".join(sorted(_LOADABLE_KEYS))
        _invalid_argument(f"unknown AnnData slot(s): {preview}; expected one of [{allowed}]")
    keys = selected if include is not None else _LOADABLE_KEYS - selected
    if "obs" not in keys or "var" not in keys:
        _invalid_argument("include/exclude must keep both 'obs' and 'var' (required by AnnData)")
    return keys


def _should_load_matrix(name: str, load_keys: frozenset[str]) -> bool:
    if name == "X":
        return "X" in load_keys
    if name in {"raw/X", "raw.X"} or name.startswith("raw/"):
        return "raw" in load_keys
    parts = name.split("/")
    if len(parts) == 2 and parts[0] in _LOADABLE_KEYS:
        return parts[0] in load_keys
    return True


def _count_scc_matrices(adata: AnnData) -> int:
    count = 0
    if adata.X is not None and _is_scc_matrix(adata.X):
        count += 1
    for name, matrix in dict(adata.layers).items():
        if name is None:
            continue
        if _is_scc_matrix(matrix):
            count += 1
    for matrix in dict(adata.obsm).values():
        if _is_scc_matrix(matrix):
            count += 1
    for matrix in dict(adata.obsp).values():
        if _is_scc_matrix(matrix):
            count += 1
    raw = adata.raw
    if raw is not None and raw.X is not None and _is_scc_matrix(raw.X):
        count += 1
    return count


# ---------------------------------------------------------------------------
# cell-matrix detection / flatten / restore
# ---------------------------------------------------------------------------


def _is_cell_matrix_path(name: str) -> bool:
    """Return whether ``name`` is a cell-aligned AnnData matrix path."""
    name = name.lstrip("/")
    if name in {"X", "raw/X", "raw.X"}:
        return True
    parts = name.split("/")
    return (
        len(parts) == 2 and parts[0] in _CELL_MATRIX_PARENTS and _is_safe_matrix_component(parts[1])
    )


def _is_safe_matrix_component(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value not in (".", "..")
        and "/" not in value
        and "\\" not in value
        and "\0" not in value
    )


def _validate_matrix_key(value: object, *, container: str) -> str:
    if not _is_safe_matrix_component(value):
        _invalid_argument(
            f"{container} keys must be non-empty path components; '/', '\\', '.', '..', "
            f"and NUL are not allowed; got {value!r}"
        )
    return cast(str, value)


def _validate_uns_keys(
    mapping: Mapping[Any, Any],
    *,
    context: str = "uns",
    active: set[int] | None = None,
) -> None:
    if active is None:
        active = set()
    identity = id(mapping)
    if identity in active:
        _invalid_argument(f"{context} contains a recursive mapping")
    active.add(identity)
    try:
        for raw_key, value in mapping.items():
            key = _validate_matrix_key(raw_key, container=context)
            if isinstance(value, Mapping):
                _validate_uns_keys(value, context=f"{context}[{key!r}]", active=active)
    finally:
        active.remove(identity)


def _is_numeric_matrix(elem: Any) -> bool:
    """Return whether ``elem`` is a numeric ndarray / sparse matrix (any rank ≥ 1)."""
    dtype = _matrix_dtype(elem)
    return dtype is not None and dtype.kind in ("f", "i", "u")


def _is_scc_matrix(elem: Any) -> bool:
    """Return whether ``elem`` can be represented losslessly by sc-compress."""
    dtype = _matrix_dtype(elem)
    return dtype is not None and dtype.kind in ("f", "i", "u") and is_value_dtype(dtype)


def _matrix_dtype(elem: Any) -> np.dtype[Any] | None:
    """Return a matrix-like object's dtype without materializing its values."""
    if elem is None:
        return None
    from scipy import sparse as sp

    try:
        import pandas as pd

        if isinstance(elem, pd.DataFrame):
            return None
    except ImportError:  # pragma: no cover
        pass

    shape = getattr(elem, "shape", None)
    dtype = getattr(elem, "dtype", None)
    if shape is None or dtype is None or len(shape) < 1:
        return None
    matrix_like = bool(
        sp.issparse(elem)
        or isinstance(elem, np.ndarray)
        or hasattr(elem, "__array__")
        or hasattr(elem, "__dlpack__")
        or hasattr(elem, "toarray")
        or hasattr(elem, "tocsr")
        or hasattr(elem, "to_memory")
    )
    if not matrix_like:
        return None
    try:
        return np.dtype(dtype)
    except (TypeError, ValueError):
        return None


def _resolve_cell_axis(shape: tuple[int, ...], n_cells: int, context: str) -> int:
    """Return ``0`` or ``-1`` for the cell axis; prefer left when both match."""
    if not shape:
        _invalid_argument(f"{context}: empty shape")
    left = int(shape[0]) == n_cells
    right = int(shape[-1]) == n_cells
    if left:
        return 0
    if right:
        return -1
    _invalid_argument(
        f"{context}: cell axis must be leftmost or rightmost (n_cells={n_cells}, shape={shape})"
    )


def _prepare_cell_matrix(
    matrix: Any,
    n_cells: int,
    context: str,
) -> tuple[Any, tuple[int, ...], int]:
    """Normalize a cell-aligned matrix for sc-compress storage.

    Returns ``(payload, original_shape, cell_axis)``.

    * Rank-2 with cells on the left: payload is the original dense/CSR object.
    * Otherwise: densify, move cell axis to front, flatten trailing dims to one
      axis → contiguous ``(n_cells, -1)`` dense array.
    """
    from scipy import sparse as sp

    raw_shape = getattr(matrix, "shape", None)
    if raw_shape is None:
        raw_shape = np.asarray(matrix).shape
    shape = tuple(int(x) for x in raw_shape)
    if len(shape) < 2:
        _invalid_argument(f"{context}: expected rank ≥ 2, got shape {shape}")
    cell_axis = _resolve_cell_axis(shape, n_cells, context)

    to_memory = getattr(matrix, "to_memory", None)
    if callable(to_memory) and getattr(matrix, "format", None) in ("csr", "csc"):
        matrix = to_memory()

    if len(shape) == 2 and cell_axis == 0:
        return matrix, shape, 0

    if sp.issparse(matrix):
        arr = np.asarray(matrix.toarray())
    else:
        arr = np.asarray(matrix)
    if cell_axis == -1:
        arr = np.moveaxis(arr, -1, 0)
    non_cell_shape = shape[1:] if cell_axis == 0 else shape[:-1]
    flat = np.ascontiguousarray(arr.reshape(n_cells, math.prod(non_cell_shape)))
    return flat, shape, cell_axis


def _restore_cell_matrix(payload: Any, attrs: dict[str, Any]) -> Any:
    """Inverse of :func:`_prepare_cell_matrix` using on-disk attributes."""
    from scipy import sparse as sp

    from sc_compress.array import ScCsr, ScDense

    # Materialized store payloads are ScDense / ScCsr; convert for AnnData.
    if isinstance(payload, ScCsr):
        payload = payload.to_scipy()
    elif isinstance(payload, ScDense):
        payload = payload.to_numpy()

    raw_shape = attrs.get(_SCC_SHAPE_ATTR)
    if raw_shape is None:
        return payload
    orig, cell_axis = _parse_shape_and_axis(attrs, context="sc-compress matrix")
    _validate_payload_shape(payload, orig, cell_axis, context="sc-compress matrix")

    if len(orig) == 2 and cell_axis == 0:
        # Sparse or dense 2D left-cell payload is already final.
        if sp.issparse(payload) or tuple(getattr(payload, "shape", ())) == orig:
            return payload
        return np.asarray(payload).reshape(orig)

    arr = payload.toarray() if sp.issparse(payload) else np.asarray(payload)
    if cell_axis == 0:
        return arr.reshape(orig)

    n_cells = orig[-1]
    rest = orig[:-1]
    mid = arr.reshape((n_cells,) + rest)
    return np.moveaxis(mid, 0, -1)


def _is_scc_node(elem: Any) -> bool:
    return _SCC_FLAG_ATTR in _node_attrs(elem)


def _node_attrs(elem: Any) -> dict[str, Any]:
    attrs = getattr(elem, "attrs", None)
    if attrs is None:
        return {}
    return dict(attrs)


def _parse_scc_layout(attrs: dict[str, Any], context: str) -> tuple[tuple[int, ...], int]:
    if attrs.get(_SCC_FLAG_ATTR) is not True:
        _invalid_meta(f"{context}: {_SCC_FLAG_ATTR!r} must be true")
    version = attrs.get(_SCC_FORMAT_ATTR)
    if version != _SCC_FORMAT_VERSION:
        _invalid_meta(
            f"{context}: unsupported {_SCC_FORMAT_ATTR!r} {version!r}; "
            f"expected {_SCC_FORMAT_VERSION!r}"
        )
    return _parse_shape_and_axis(attrs, context=context, require_axis=True)


def _parse_shape_and_axis(
    attrs: dict[str, Any],
    *,
    context: str,
    require_axis: bool = False,
) -> tuple[tuple[int, ...], int]:
    raw_shape = attrs.get(_SCC_SHAPE_ATTR)
    if isinstance(raw_shape, str) or raw_shape is None:
        _invalid_meta(f"{context}: missing or invalid {_SCC_SHAPE_ATTR!r}")
    try:
        shape_values = tuple(raw_shape)
    except TypeError:
        _invalid_meta(f"{context}: {_SCC_SHAPE_ATTR!r} must be a sequence of integers")
    if len(shape_values) < 2:
        _invalid_meta(f"{context}: {_SCC_SHAPE_ATTR!r} must contain at least two dimensions")

    shape: list[int] = []
    for position, value in enumerate(shape_values):
        if isinstance(value, (bool, np.bool_)):
            _invalid_meta(f"{context}: shape[{position}] must be a non-negative integer")
        try:
            dimension = index(value)
        except TypeError:
            _invalid_meta(f"{context}: shape[{position}] must be a non-negative integer")
        if dimension < 0:
            _invalid_meta(f"{context}: shape[{position}] must be non-negative, got {dimension}")
        if dimension > _UINT64_MAX:
            _invalid_meta(f"{context}: shape[{position}] exceeds uint64")
        shape.append(dimension)

    if require_axis and _SCC_CELL_AXIS_ATTR not in attrs:
        _invalid_meta(f"{context}: missing {_SCC_CELL_AXIS_ATTR!r}")
    raw_axis = attrs.get(_SCC_CELL_AXIS_ATTR, 0)
    if isinstance(raw_axis, (bool, np.bool_)):
        _invalid_meta(f"{context}: {_SCC_CELL_AXIS_ATTR!r} must be 0 or -1")
    try:
        cell_axis = index(raw_axis)
    except TypeError:
        _invalid_meta(f"{context}: {_SCC_CELL_AXIS_ATTR!r} must be 0 or -1")
    if cell_axis not in (0, -1):
        _invalid_meta(
            f"{context}: unsupported {_SCC_CELL_AXIS_ATTR!r} {cell_axis!r}; expected 0 or -1"
        )
    non_cell_shape = shape[1:] if cell_axis == 0 else shape[:-1]
    if math.prod(non_cell_shape) > _UINT64_MAX:
        _invalid_meta(f"{context}: flattened non-cell dimension exceeds uint64")
    return tuple(shape), cell_axis


def _validate_payload_shape(
    payload: Any,
    original_shape: tuple[int, ...],
    cell_axis: int,
    *,
    context: str,
) -> None:
    raw_payload_shape = getattr(payload, "shape", None)
    if raw_payload_shape is None:
        _invalid_meta(f"{context}: decoded payload has no shape")
    try:
        payload_shape = tuple(int(value) for value in raw_payload_shape)
    except (TypeError, ValueError):
        _invalid_meta(f"{context}: decoded payload has an invalid shape")
    if cell_axis == 0:
        expected = (original_shape[0], math.prod(original_shape[1:]))
    else:
        expected = (original_shape[-1], math.prod(original_shape[:-1]))
    if payload_shape != expected:
        _invalid_meta(
            f"{context}: payload shape {payload_shape} does not match "
            f"declared shape {original_shape} with cell axis {cell_axis} "
            f"(expected stored shape {expected})"
        )


def _matrix_relpath(parent_group: Any, key: str) -> str:
    key = key.lstrip("/")
    parent = str(getattr(parent_group, "path", "") or "").strip("/")
    return f"{parent}/{key}" if parent else key


def _matrix_path(root: Path, rel: str) -> Path:
    """Resolve a portable matrix path while rejecting traversal components."""
    parts = rel.split("/")
    if rel.startswith("/") or any(not _is_safe_matrix_component(part) for part in parts):
        _invalid_argument(f"invalid AnnData matrix path: {rel!r}")
    return root.joinpath(*parts)


def _write_scc_matrix(
    parent_group: Any,
    key: str,
    matrix: Any,
    *,
    root: Path,
    n_cells: int,
    options: WriteOptions,
) -> None:
    rel = _matrix_relpath(parent_group, key)
    path = _matrix_path(root, rel)
    if path.exists() or path.is_symlink():
        _remove_path(path)

    payload, orig_shape, cell_axis = _prepare_cell_matrix(matrix, n_cells, rel)
    write_matrix(path, payload, options=options, overwrite=True)

    meta = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            # Registered AnnData group encoding so read_dispatched can enter
            # our callback; identity is ``sc-compress``, not ``dict``.
            "encoding-type": "dict",
            "encoding-version": "0.1.0",
            _SCC_FLAG_ATTR: True,
            _SCC_FORMAT_ATTR: _SCC_FORMAT_VERSION,
            _SCC_SHAPE_ATTR: [int(x) for x in orig_shape],
            _SCC_CELL_AXIS_ATTR: int(cell_axis),
        },
    }
    (path / "zarr.json").write_text(
        json.dumps(meta, allow_nan=False, indent=2),
        encoding="utf-8",
    )


def _read_scc_matrix(
    root: Path,
    rel: str,
    archive_path: Path | None,
    *,
    attrs: dict[str, Any] | None = None,
    limits: ReadLimits,
) -> Any:
    from sc_compress.io import open_store

    layout_attrs = attrs or {}
    orig_shape, cell_axis = _parse_scc_layout(layout_attrs, rel)
    matrix_path = _matrix_path(root, rel)
    if archive_path is not None:
        with open_store(archive_path, zip_prefix=rel, limits=limits) as store:
            _validate_payload_shape(store, orig_shape, cell_axis, context=rel)
            payload = store.read()
    else:
        with open_store(matrix_path, limits=limits) as store:
            _validate_payload_shape(store, orig_shape, cell_axis, context=rel)
            payload = store.read()
    return _restore_cell_matrix(
        payload,
        {
            _SCC_SHAPE_ATTR: orig_shape,
            _SCC_CELL_AXIS_ATTR: cell_axis,
        },
    )


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _preflight_write_adata(adata: AnnData) -> bool:
    """Validate matrices and return whether any sc-compress payload is dense."""
    if adata.n_obs < 0 or adata.n_vars < 0:
        _invalid_argument(
            f"AnnData dimensions must be non-negative, got ({adata.n_obs}, {adata.n_vars})"
        )
    n_cells = int(adata.n_obs)
    has_dense_matrix = False

    if adata.X is not None:
        has_dense_matrix |= _validate_expression_matrix(
            adata.X,
            adata.n_obs,
            adata.n_vars,
            "X",
        )

    for raw_name, matrix in dict(adata.layers).items():
        if raw_name is None:
            continue
        name = _validate_matrix_key(raw_name, container="layers")
        has_dense_matrix |= _validate_expression_matrix(
            matrix, adata.n_obs, adata.n_vars, f"layer {name!r}"
        )

    for name, matrix in dict(adata.obsm).items():
        _validate_matrix_key(name, container="obsm")
        if _is_numeric_matrix(matrix):
            has_dense_matrix |= _validate_cell_aligned_matrix(
                matrix,
                n_cells,
                f"obsm[{name!r}]",
            )

    for name, matrix in dict(adata.obsp).items():
        _validate_matrix_key(name, container="obsp")
        if _is_numeric_matrix(matrix):
            has_dense_matrix |= _validate_cell_aligned_matrix(
                matrix,
                n_cells,
                f"obsp[{name!r}]",
            )

    for name in dict(adata.varm):
        _validate_matrix_key(name, container="varm")
    for name in dict(adata.varp):
        _validate_matrix_key(name, container="varp")
    _validate_uns_keys(adata.uns)

    raw = adata.raw
    if raw is None:
        return has_dense_matrix
    if raw.n_obs != adata.n_obs:
        _invalid_argument(f"AnnData.raw has {raw.n_obs} cells, expected {adata.n_obs}")
    if raw.X is not None:
        has_dense_matrix |= _validate_expression_matrix(
            raw.X,
            raw.n_obs,
            raw.n_vars,
            "raw.X",
        )
    for name in dict(getattr(raw, "varm", {})):
        _validate_matrix_key(name, container="raw.varm")
    return has_dense_matrix


def _validate_expression_matrix(matrix: Any, n_obs: int, n_var: int, context: str) -> bool:
    if matrix is None:
        _invalid_argument(f"matrix {context!r} is None")
    if not _is_numeric_matrix(matrix):
        _invalid_argument(f"matrix {context!r} is not a numeric array/sparse matrix")
    shape = tuple(int(x) for x in matrix.shape)
    expected = (int(n_obs), int(n_var))
    if shape != expected:
        _invalid_argument(f"matrix {context!r} has shape {shape}, expected {expected}")
    _validate_value_dtype(matrix, context=f"matrix {context!r}")
    return not is_sparse_matrix(matrix)


def _validate_cell_aligned_matrix(matrix: Any, n_cells: int, context: str) -> bool:
    if matrix is None:
        _invalid_argument(f"{context} is None")
    if not _is_numeric_matrix(matrix):
        _invalid_argument(f"{context} is not a numeric array/sparse matrix")
    shape = tuple(int(x) for x in matrix.shape)
    if len(shape) < 2:
        _invalid_argument(f"{context}: expected rank ≥ 2, got shape {shape}")
    cell_axis = _resolve_cell_axis(shape, n_cells, context)
    _validate_value_dtype(matrix, context=context)
    return not (is_sparse_matrix(matrix) and len(shape) == 2 and cell_axis == 0)


def _validate_value_dtype(matrix: Any, *, context: str) -> None:
    if np.ma.isMaskedArray(matrix) and np.ma.getmaskarray(matrix).any():
        _invalid_argument(f"{context} contains masked values; fill them explicitly before writing")
    dtype = _matrix_dtype(matrix)
    if dtype is not None and is_value_dtype(dtype):
        return
    dtype_name = "unknown" if dtype is None else str(dtype)
    allowed = ", ".join(sorted(value.name for value in VALUE_DTYPES))
    _invalid_argument(
        f"{context} dtype {dtype_name} is unsupported by sc-compress; "
        f"convert it to one of [{allowed}]"
    )


# ---------------------------------------------------------------------------
# store helpers (directory / zip), adapted from scdata.io._anndata
# ---------------------------------------------------------------------------


def _resolve_store_kind(path: Path, store: _Store) -> Literal["zip", "dir"]:
    if not isinstance(store, str):
        _invalid_argument(f"store must be 'auto', 'dir', or 'zip', got {type(store).__name__}")
    normalized = store.casefold()
    if normalized == "auto":
        return "zip" if path.name.lower().endswith(".zip") else "dir"
    if normalized == "dir":
        if path.name.lower().endswith(".zip"):
            warnings.warn(
                f"store='dir' will create a directory at {path} despite the .zip suffix",
                PerformanceWarning,
                stacklevel=3,
            )
        return "dir"
    if normalized == "zip":
        return "zip"
    _invalid_argument(f"store must be 'auto', 'dir', or 'zip', got {store!r}")


def _prepare_zip_target(root: Path) -> None:
    if root.is_dir():
        _invalid_argument(f"zip output path is a directory: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)


def _make_temp_dir(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent))


def _zip_directory(source: Path, target: Path, *, overwrite: bool) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for directory, directory_names, file_names in os.walk(source):
                directory_names.sort()
                file_names.sort()
                for file_name in file_names:
                    path = Path(directory, file_name)
                    zf.write(
                        path,
                        path.relative_to(source).as_posix(),
                        compress_type=zipfile.ZIP_STORED,
                    )
        ensure_writable_path(target, overwrite=overwrite)
        if overwrite:
            os.replace(tmp, target)
        else:
            try:
                os.link(tmp, target)
            except FileExistsError:
                _invalid_argument(f"path already exists: {target} (pass overwrite=True to replace)")
            tmp.unlink()
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _replace_directory(source: Path, target: Path, *, overwrite: bool) -> None:
    ensure_writable_path(target, overwrite=overwrite)
    backup: Path | None = None
    if target.exists() or target.is_symlink():
        if not overwrite:
            _invalid_argument(f"path already exists: {target} (pass overwrite=True to replace)")
        backup = _make_temp_backup_path(target)
        os.replace(target, backup)
    try:
        os.replace(source, target)
    except Exception:
        if backup is not None and backup.exists():
            os.replace(backup, target)
        raise
    if backup is not None:
        _remove_path(backup)


def _make_temp_backup_path(target: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".bak", dir=target.parent))
    tmp.rmdir()
    return tmp


def _remove_path(path: Path) -> None:
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    except FileNotFoundError:
        pass


def _open_store_for_read(path: Path) -> tuple[Any, Path | None]:
    """Open a directory or ZIP as a zarr group.

    Returns ``(group, archive_path)``.  ``archive_path`` is set for ZIP inputs
    so matrix reads can use :func:`sc_compress.open_store` with ``zip_prefix``.
    """
    import zarr
    from zarr.storage import ZipStore

    if path.is_file():
        if not zipfile.is_zipfile(path):
            _invalid_argument(f"path is a file but not a ZIP archive: {path}")
        store = ZipStore(str(path), mode="r")
        try:
            return zarr.open_group(store, mode="r"), path
        except BaseException:
            store.close()
            raise
    if path.is_dir():
        return zarr.open_group(str(path), mode="r"), None
    _invalid_argument(f"path does not exist: {path}")


@contextmanager
def _write_config() -> Iterator[None]:
    """Disable AnnData auto-sharding for annotation arrays during write."""
    ad = _require_anndata()

    override = (
        ad.settings.override(auto_shard_zarr_v3=False)
        if hasattr(ad.settings, "auto_shard_zarr_v3")
        else nullcontext()
    )
    with override:
        yield


@contextmanager
def _anndata_zarr_context() -> Iterator[None]:
    """Use AnnData's optional zarrs pipeline context when the version provides it."""
    try:
        from anndata._io import zarr as anndata_zarr
    except ImportError:
        context = nullcontext()
    else:
        factory = getattr(anndata_zarr, "zarrs_context", None)
        context = nullcontext() if factory is None else factory()
    with context:
        yield


def _require_anndata() -> Any:
    try:
        import anndata as ad
    except ModuleNotFoundError as error:
        if error.name == "anndata":
            raise ImportError(
                "AnnData support is optional; install it with `pip install 'sc-compress[anndata]'`"
            ) from None
        raise
    return ad


def _invalid_meta(message: str) -> NoReturn:
    error = InvalidMetaError(message)
    error.kind = "invalid_meta"
    raise error
