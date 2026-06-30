"""anndata <-> scdata zarr v3 bridge.

scdata stores are plain zarr v3 trees (one ``zarr.json`` per node, standard
per-chunk files), produced by :func:`write_zarr` and read back by
:func:`read_zarr`.  Staying inside the zarr v3 standard means a store written
for the Rust databank is also readable by stock ``anndata.read_zarr`` — with
two deliberate, spec-legal extensions:

* ``dense1d`` X — a flattened 1D ``[cells * genes]`` array chunked so a cell
  never spans two chunks.  The shape is not one anndata's ``AnnData``
  constructor accepts, so :func:`read_zarr` reshapes it to 2D on read.
* cell-aligned CSR — ``data`` / ``indices`` use a zarr v3 **rectilinear**
  chunk grid whose edges are CSR row boundaries, so each cell's nonzero run
  lives in one chunk.  Rectilinear is standard v3 (the metadata stores the
  full per-chunk edge list) but experimental in zarr, so read/write set
  ``zarr.config['array.rectilinear_chunks'] = True``.

Three X layouts are supported at write time: ``dense2d`` (standard 2D,
anndata-readable), ``dense1d`` (flattened, cell-aligned), ``sparse`` (CSR,
optionally cell-aligned via rectilinear).

Chunk sizing follows zarr conventions: ``chunk_size`` may be a list/tuple (used
verbatim as the zarr ``chunks`` tuple) or an int (the per-chunk element
count, broadcast across dimensions — a multi-dimensional array takes the
``ndim``-th root so each chunk holds roughly that many elements).
"""

from __future__ import annotations

import json
import os
import warnings
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, cast

import numpy as np

from scdata.io._launch import StoreError

if TYPE_CHECKING:
    from anndata import AnnData, Raw

__all__ = ["write_zarr", "read_zarr"]

_XFormat = Literal["dense2d", "dense1d", "sparse"]
_LayerFormat = Literal["preserve", "auto", "dense2d", "dense1d", "sparse"]
_Store = Literal["zip", "dir"]
_Compressor = str | Mapping[str, Any] | None

# scdata marker stored in matrix array/group ``attributes`` so :func:`read_zarr`
# can recover the layout without guessing from shape.  ``scdata-x`` is the
# legacy spelling; new stores write both for compatibility.
_SCDATA_MATRIX_ATTR = "scdata-matrix"
_SCDATA_X_ATTR = "scdata-x"
_SCDATA_SHAPE_2D = "scdata-shape-2d"

# Default chunk element count when ``chunk_size`` is an int.
_DEFAULT_CHUNK_ELEMENTS = 1_000_000
_DEFAULT_COMPRESSOR = "blosc.lz4.level5"


# ---------------------------------------------------------------------------
# public write
# ---------------------------------------------------------------------------


def write_zarr(
    adata: "AnnData",
    path: str | os.PathLike[str],
    *,
    format: _XFormat = "dense2d",
    layer_format: _LayerFormat = "preserve",
    chunk_size: int | list[int] | tuple[int, ...] = _DEFAULT_CHUNK_ELEMENTS,
    align_cells: bool = True,
    store: _Store = "zip",
    compressor: _Compressor = _DEFAULT_COMPRESSOR,
) -> Path:
    """Write an :class:`anndata.AnnData` as a scdata zarr v3 store.

    Parameters
    ----------
    adata:
        AnnData object to write.  ``X`` may be dense or CSR sparse; for
        ``format="sparse"`` a dense ``X`` is converted to CSR.
    path:
        Destination path.  For ``store="zip"`` this is the ``.zarr.zip`` file
        (created/overwritten); for ``store="dir"`` it is the zarr directory.
        No suffix is appended — the caller picks the name.
    format:
        X layout — ``"dense2d"`` (standard 2D), ``"dense1d"`` (flattened 1D,
        cell-aligned chunking), or ``"sparse"`` (CSR, optionally cell-aligned).
    layer_format:
        Layer layout.  ``"preserve"`` writes dense layers as standard 2D arrays
        and sparse layers as CSR.  ``"auto"`` writes sparse layers as CSR and
        dense layers as ``dense1d`` when ``align_cells`` is true.  Explicit
        ``"dense2d"``, ``"dense1d"``, or ``"sparse"`` applies that layout to
        every layer.
    chunk_size:
        Chunk shape.  A list/tuple is used verbatim as the zarr ``chunks``
        tuple; an int is the per-chunk element count, broadcast across
        dimensions (multi-dimensional arrays take the ``ndim``-th root).
    align_cells:
        When true, grow each chunk along the cell axis until it holds a whole
        number of cells, so a cell never spans two chunks.  Silently ignored
        for ``dense2d`` (whose cell axis is already chunk-aligned).  For
        ``sparse`` this produces a rectilinear (variable-length) chunk grid
        aligned to CSR row boundaries.  Applies to ``X``, sparse ``layers``,
        and ``raw.X`` alike.
    store:
        ``"zip"`` (default) writes a ``ZIP_STORED`` archive; ``"dir"`` writes
        a directory tree.
    compressor:
        Chunk compressor.  Defaults to ``"blosc.lz4.level5"``.  Pass ``None``
        or ``"none"`` for uncompressed chunks.
    """
    import zarr
    from anndata._io.specs import write_elem
    from zarr.storage import MemoryStore

    _enable_rectilinear()

    root = Path(os.fspath(path))
    _validate_write_options(
        format=format,
        layer_format=layer_format,
        chunk_size=chunk_size,
        store=store,
        compressor=compressor,
    )
    if store == "zip":
        _prepare_zip_target(root)
        zstore = MemoryStore()
    else:
        tmp_root = _make_temp_dir(root)
        zstore = _make_zarr_store(tmp_root, store)

    try:
        g = zarr.open_group(zstore, mode="w", zarr_format=3)
        g.attrs["encoding-type"] = "anndata"
        g.attrs["encoding-version"] = "0.1.0"

        with _suppress_known_zarr_write_warnings():
            # Everything except X is written by anndata's own element writers,
            # so obs/var/uns/obsm/... stay compatible with anndata.write_zarr.
            write_elem(g, "obs", adata.obs)
            write_elem(g, "var", adata.var)
            if adata.obsm:
                write_elem(g, "obsm", dict(adata.obsm))
            if adata.varm:
                write_elem(g, "varm", dict(adata.varm))
            if adata.obsp:
                write_elem(g, "obsp", dict(adata.obsp))
            if adata.varp:
                write_elem(g, "varp", dict(adata.varp))
            write_elem(g, "uns", dict(adata.uns))
            if adata.raw is not None:
                _write_raw(
                    g,
                    adata.raw,
                    format=format,
                    chunk_size=chunk_size,
                    align_cells=align_cells,
                    compressor=compressor,
                )

            _write_x(
                g,
                adata,
                format=format,
                chunk_size=chunk_size,
                align_cells=align_cells,
                compressor=compressor,
            )
            _write_layers(
                g,
                adata,
                layer_format=layer_format,
                chunk_size=chunk_size,
                align_cells=align_cells,
                compressor=compressor,
            )

        if store == "zip":
            _zip_store(zstore, root)
        else:
            _replace_directory(tmp_root, root)
            tmp_root = None
    finally:
        close = getattr(zstore, "close", None)
        if close is not None:
            close()
        if store == "dir" and tmp_root is not None:
            _remove_path(tmp_root)
    return root


# ---------------------------------------------------------------------------
# X writers
# ---------------------------------------------------------------------------


def _write_x(
    g: Any,
    adata: "AnnData",
    *,
    format: _XFormat,
    chunk_size: int | list[int] | tuple[int, ...],
    align_cells: bool,
    compressor: _Compressor,
) -> None:
    """Write the X array/group in the requested layout."""
    _write_matrix(
        g,
        "X",
        adata.X,
        n_obs=adata.n_obs,
        n_var=adata.n_vars,
        format=format,
        chunk_size=chunk_size,
        align_cells=align_cells,
        compressor=compressor,
    )


def _write_layers(
    g: Any,
    adata: "AnnData",
    *,
    layer_format: _LayerFormat,
    chunk_size: int | list[int] | tuple[int, ...],
    align_cells: bool,
    compressor: _Compressor,
) -> None:
    """Write all AnnData layers using the same scdata matrix layouts as X."""
    if not adata.layers:
        return
    layers = g.require_group("layers")
    layers.attrs.update({"encoding-type": "dict", "encoding-version": "0.1.0"})
    for raw_name, matrix in dict(adata.layers).items():
        name = _validate_layer_name(raw_name)
        fmt = _resolve_layer_format(matrix, layer_format, align_cells=align_cells)
        _write_matrix(
            layers,
            name,
            matrix,
            n_obs=adata.n_obs,
            n_var=adata.n_vars,
            format=fmt,
            chunk_size=chunk_size,
            align_cells=align_cells,
            compressor=compressor,
        )


def _write_raw(
    g: Any,
    raw: "Raw",
    *,
    format: _XFormat,
    chunk_size: int | list[int] | tuple[int, ...],
    align_cells: bool,
    compressor: _Compressor,
) -> None:
    """Write ``adata.raw`` so ``raw.X`` uses the same scdata layout as ``X``.

    ``var`` and ``varm`` go through anndata's element writers unchanged; only
    ``raw.X`` is rerouted through :func:`_write_matrix` so it picks up the same
    cell-aligned CSR / dense1d layout, compressor, and ``scdata-matrix`` marker
    as the main ``X``.  The group keeps anndata's ``raw`` encoding-type so stock
    ``anndata.read_zarr`` still recognizes the group; :func:`read_zarr` reads
    ``raw/X`` back through its scdata path via :func:`_is_matrix_root`.
    """
    from anndata._io.specs import write_elem

    sub = g.require_group("raw")
    sub.attrs.update({"encoding-type": "raw", "encoding-version": "0.1.0"})
    write_elem(sub, "var", raw.var)
    if raw.varm:
        write_elem(sub, "varm", dict(raw.varm))
    if raw.X is None:
        write_elem(sub, "X", None)
        return
    _write_matrix(
        sub,
        "X",
        raw.X,
        n_obs=raw.n_obs,
        n_var=raw.n_vars,
        format=format,
        chunk_size=chunk_size,
        align_cells=align_cells,
        compressor=compressor,
    )


def _write_matrix(
    g: Any,
    name: str,
    matrix: Any,
    *,
    n_obs: int,
    n_var: int,
    format: _XFormat,
    chunk_size: int | list[int] | tuple[int, ...],
    align_cells: bool,
    compressor: _Compressor,
) -> None:
    """Write one dense or CSR matrix under ``g[name]`` in a scdata layout."""
    from scipy import sparse as _sparse

    # ``AnnData.X`` / layers are wide unions (dense / sparse / backed / None);
    # treat it as Any here — scipy's issparse is not a TypeGuard, so the
    # sparse-only toarray()/tocsr() calls below would otherwise be rejected.
    if matrix is None:
        raise StoreError(f"matrix {name!r} is None")
    _validate_matrix_shape(matrix, n_obs, n_var, name)

    if format == "dense2d":
        dense = matrix.toarray() if _sparse.issparse(matrix) else np.asarray(matrix)
        chunk_shape = _resolve_chunk_size_2d(chunk_size, n_obs, n_var)
        _create_dense_array(
            g,
            name,
            dense,
            chunk_shape,
            attrs=_matrix_attrs("dense2d", n_obs, n_var),
            compressor=compressor,
        )
    elif format == "dense1d":
        dense = matrix.toarray() if _sparse.issparse(matrix) else np.asarray(matrix)
        flat = np.ascontiguousarray(dense).reshape(-1)
        chunk_shape = _resolve_chunk_size_1d(chunk_size, n_var, align_cells)
        _create_dense_array(
            g,
            name,
            flat,
            chunk_shape,
            attrs=_matrix_attrs("dense1d", n_obs, n_var),
            compressor=compressor,
        )
    elif format == "sparse":
        if _sparse.isspmatrix_csr(matrix):
            csr = matrix
        elif _sparse.issparse(matrix):
            csr = matrix.tocsr()
        else:
            # Dense input — ``format="sparse"`` promises a CSR conversion.
            csr = _sparse.csr_matrix(np.asarray(matrix))
        _write_csr_group(
            g,
            name,
            csr,
            chunk_size=chunk_size,
            align_cells=align_cells,
            compressor=compressor,
        )
    else:  # pragma: no cover - Literal exhausts the cases
        raise StoreError(f"unsupported X format: {format!r}")


def _matrix_attrs(kind: str, n_obs: int, n_var: int) -> dict[str, Any]:
    return {
        "encoding-type": "array",
        "encoding-version": "0.2.0",
        _SCDATA_MATRIX_ATTR: kind,
        _SCDATA_X_ATTR: kind,
        _SCDATA_SHAPE_2D: [int(n_obs), int(n_var)],
    }


def _validate_layer_name(name: object) -> str:
    if not isinstance(name, str) or not name:
        raise StoreError(f"layer names must be non-empty strings, got {name!r}")
    if "/" in name:
        raise StoreError(f"nested layer names are unsupported: {name!r}")
    return name


def _matrix_shape(matrix: Any) -> tuple[int, int]:
    shape = getattr(matrix, "shape", None)
    if shape is None:
        shape = np.asarray(matrix).shape
    if len(shape) != 2:
        raise StoreError(f"matrix must be 2D, got shape {tuple(shape)!r}")
    return int(shape[0]), int(shape[1])


def _validate_matrix_shape(matrix: Any, n_obs: int, n_var: int, context: str) -> None:
    shape = _matrix_shape(matrix)
    expected = (int(n_obs), int(n_var))
    if shape != expected:
        raise StoreError(f"matrix {context!r} has shape {shape}, expected {expected}")


def _resolve_layer_format(
    matrix: Any,
    layer_format: _LayerFormat,
    *,
    align_cells: bool,
) -> _XFormat:
    from scipy import sparse as _sparse

    if layer_format == "preserve":
        return "sparse" if _sparse.issparse(matrix) else "dense2d"
    if layer_format == "auto":
        return "sparse" if _sparse.issparse(matrix) else ("dense1d" if align_cells else "dense2d")
    if layer_format in ("dense2d", "dense1d", "sparse"):
        return layer_format
    raise StoreError(f"unsupported layer_format {layer_format!r}")


def _create_dense_array(
    g: Any,
    name: str,
    data: np.ndarray,
    chunk_shape: tuple[int, ...],
    *,
    attrs: dict[str, Any],
    compressor: _Compressor,
) -> None:
    """Create a v3 dense array with the scdata default codec pipeline."""
    from zarr.codecs import BytesCodec

    data = np.asarray(data, dtype=_little_endian_dtype(data.dtype))
    arr = g.create_array(
        name,
        shape=data.shape,
        dtype=data.dtype,
        chunks=chunk_shape,
        shards=None,
        filters=(),
        serializer=BytesCodec(endian="little"),
        compressors=_zarr_compressors(np.dtype(data.dtype), compressor),
        fill_value=_fill_value_for(data.dtype),
    )
    arr.attrs.update(attrs)
    arr[...] = data


def _write_csr_group(
    g: Any,
    name: str,
    csr: Any,
    *,
    chunk_size: int | list[int] | tuple[int, ...],
    align_cells: bool,
    compressor: _Compressor,
) -> None:
    """Write a CSR matrix as an anndata-compatible v3 group.

    ``indptr`` is always a single-chunk 1D array (small, read once).
    ``data`` / ``indices`` are 1D arrays of length ``nnz``.  With
    ``align_cells=True`` they use a rectilinear chunk grid whose edges are CSR
    row boundaries (``indptr`` values), so each cell's nonzero run lives in
    one chunk; with ``align_cells=False`` they use a regular chunk grid.
    """
    sub = g.require_group(name)
    sub.attrs.update(
        {
            "encoding-type": "csr_matrix",
            "encoding-version": "0.1.0",
            "shape": [int(csr.shape[0]), int(csr.shape[1])],
            _SCDATA_MATRIX_ATTR: "sparse",
            _SCDATA_X_ATTR: "sparse",
        }
    )

    indptr = np.asarray(csr.indptr)
    indices = np.asarray(csr.indices)
    data = np.asarray(csr.data)

    _create_dense_array(
        sub,
        "indptr",
        indptr,
        (indptr.shape[0],),
        attrs={
            "encoding-type": "array",
            "encoding-version": "0.2.0",
        },
        compressor=compressor,
    )

    if align_cells:
        boundaries = _aligned_cell_boundaries(indptr, _resolve_sparse_chunk_target(chunk_size))
        _write_rectilinear_array(
            sub,
            "indices",
            indices,
            boundaries,
            csr.indices.dtype,
            compressor=compressor,
        )
        _write_rectilinear_array(
            sub,
            "data",
            data,
            boundaries,
            csr.data.dtype,
            compressor=compressor,
        )
    else:
        nnz_chunks = _resolve_chunk_size_1d(chunk_size, 1, align_cells=False)
        _create_dense_array(
            sub,
            "indices",
            indices,
            nnz_chunks,
            attrs={
                "encoding-type": "array",
                "encoding-version": "0.2.0",
            },
            compressor=compressor,
        )
        _create_dense_array(
            sub,
            "data",
            data,
            nnz_chunks,
            attrs={
                "encoding-type": "array",
                "encoding-version": "0.2.0",
            },
            compressor=compressor,
        )


def _resolve_sparse_chunk_target(chunk_size: int | list[int] | tuple[int, ...]) -> int:
    """Resolve ``chunk_size`` to the target nnz count for cell-aligned CSR chunks."""
    if isinstance(chunk_size, (list, tuple)):
        if len(chunk_size) != 1:
            raise StoreError(f"sparse chunk_size list must have 1 entry, got {len(chunk_size)}")
        target = int(chunk_size[0])
    else:
        target = int(chunk_size)
    if target <= 0:
        raise StoreError(f"sparse chunk_size must be positive, got {target}")
    return target


def _aligned_cell_boundaries(indptr: np.ndarray, target: int) -> list[int]:
    """CSR nnz offsets that align chunks to whole cells.

    Walks cells left to right, accumulating a chunk until it holds at least
    ``target`` nnz (or the end), then starts a new chunk.  Every boundary is
    an ``indptr`` value, so each chunk contains a whole number of cells; the
    last boundary is ``indptr[-1]`` (== nnz).
    """
    n = indptr.shape[0] - 1
    if n <= 0:
        return [0, 0]
    boundaries = [int(indptr[0])]
    start_cell = 0
    while start_cell < n:
        cell = start_cell
        while cell < n and (int(indptr[cell]) - int(indptr[start_cell])) < target:
            cell += 1
        if cell == start_cell:  # one cell already exceeds target — take it alone
            cell = start_cell + 1
        boundaries.append(int(indptr[cell]))
        start_cell = cell
    if boundaries[-1] != int(indptr[-1]):
        boundaries.append(int(indptr[-1]))
    return boundaries


def _write_rectilinear_array(
    g: Any,
    name: str,
    values: np.ndarray,
    boundaries: list[int],
    dtype: Any,
    *,
    compressor: _Compressor,
) -> None:
    """Write a 1D array with a rectilinear (variable-length) chunk grid.

    Each ``[boundaries[i], boundaries[i+1])`` slice becomes one chunk file.
    The ``zarr.json`` declares a rectilinear chunk grid with the explicit edge
    list, so the on-disk layout is standard zarr v3 (readable by zarr/anndata
    with ``array.rectilinear_chunks`` enabled).  We write the metadata and
    chunk files by hand because zarr's ``create_array`` only accepts regular
    chunk grids.
    """
    total = int(values.shape[0])
    np_dtype = _little_endian_dtype(dtype)
    attrs = {
        "encoding-type": "array",
        "encoding-version": "0.2.0",
        _SCDATA_MATRIX_ATTR: "sparse-vlen",
        _SCDATA_X_ATTR: "sparse-vlen",
    }
    if total == 0:
        _create_dense_array(
            g,
            name,
            np.asarray(values, dtype=np_dtype),
            (1,),
            attrs=attrs,
            compressor=compressor,
        )
        return

    runs = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
    zarray = _v3_array_json(
        shape=[total],
        data_type=_v3_dtype_name(np_dtype),
        chunk_grid={
            "name": "rectilinear",
            "configuration": {
                "kind": "inline",
                "chunk_shapes": [runs],
            },
        },
        codecs=_v3_codecs(np_dtype, compressor),
        fill_value=_fill_value_for(np_dtype),
        attrs=attrs,
    )
    _write_v3_node(g, name, zarray, is_group=False)
    # Write one chunk file per run, keyed by the default chunk-key encoding
    # for a 1D grid: c/<i>.
    store = g.store
    base = _group_store_key(g, f"{name}/c")
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if end <= start:
            continue
        chunk_bytes = np.ascontiguousarray(values[start:end]).astype(np_dtype, copy=False).tobytes()
        chunk_bytes = _encode_chunk_bytes(chunk_bytes, np_dtype, compressor)
        _store_set_bytes(store, f"{base}/{i}", chunk_bytes)


# ---------------------------------------------------------------------------
# chunk shape resolution
# ---------------------------------------------------------------------------


def _resolve_chunk_size_2d(
    chunk_size: int | list[int] | tuple[int, ...],
    n_obs: int,
    n_var: int,
) -> tuple[int, ...]:
    """Resolve ``chunk_size`` for a 2D dense X.

    A list/tuple is used verbatim.  An int is the per-chunk element count;
    the array takes the square root per dimension (so a chunk holds roughly
    ``int`` elements), capped to the axis length.
    """
    if isinstance(chunk_size, (list, tuple)):
        if len(chunk_size) != 2:
            raise StoreError(f"dense2d chunk_size list must have 2 entries, got {len(chunk_size)}")
        size = (int(chunk_size[0]), int(chunk_size[1]))
        if any(value <= 0 for value in size):
            raise StoreError(f"dense2d chunk_size entries must be positive, got {size}")
        return size
    return _broadcast_int_chunk_size(int(chunk_size), 2, (n_obs, n_var))


def _resolve_chunk_size_1d(
    chunk_size: int | list[int] | tuple[int, ...],
    gene_count: int,
    align_cells: bool,
) -> tuple[int]:
    """Resolve ``chunk_size`` for a 1D (flattened) X.

    A list/tuple is used verbatim (length 1).  An int is the per-chunk element
    count.  When ``align_cells`` is true the chunk length is rounded up to a
    multiple of ``gene_count`` so every chunk holds whole cells.
    """
    if isinstance(chunk_size, (list, tuple)):
        if len(chunk_size) != 1:
            raise StoreError(f"dense1d chunk_size list must have 1 entry, got {len(chunk_size)}")
        size = int(chunk_size[0])
    else:
        size = int(chunk_size)
    if size <= 0:
        raise StoreError(f"dense1d chunk_size must be positive, got {size}")
    if align_cells and gene_count > 0:
        size = _ceil_div(size, gene_count) * gene_count
    return (size,)


def _broadcast_int_chunk_size(
    target: int,
    ndim: int,
    shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Broadcast an int chunk-target to a per-axis chunk shape.

    The int is the desired elements per chunk; each axis gets the ``ndim``-th
    root, rounded up, capped to the axis length.
    """
    if target <= 0:
        raise StoreError(f"chunk_size int must be positive, got {target}")
    per_axis = max(1, round(target ** (1.0 / ndim)))
    return tuple(min(per_axis, max(1, s)) for s in shape)


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _fill_value_for(dtype: Any) -> Any:
    """Return the zarr v3 fill value for a numpy dtype (0 for numeric)."""
    np_dt = np.dtype(dtype)
    if np_dt.kind == "f":
        return 0.0
    if np_dt.kind == "b":
        return False
    if np_dt.kind in ("i", "u"):
        return 0
    return 0


def _little_endian_dtype(dtype: Any) -> np.dtype:
    """Return the on-disk numeric dtype scdata writes for zarr v3 chunks."""
    return np.dtype(dtype).newbyteorder("<")


# ---------------------------------------------------------------------------
# v3 metadata + store helpers
# ---------------------------------------------------------------------------


def _enable_rectilinear() -> None:
    """Enable zarr's experimental rectilinear chunk grids (write + read)."""
    import zarr

    zarr.config.set({"array.rectilinear_chunks": True})


@contextmanager
def _suppress_known_zarr_write_warnings():
    """Suppress zarr migration notices that callers cannot act on per write."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"zarr v3 autosharding will be the default.*",
            category=UserWarning,
        )
        yield


def _v3_dtype_name(np_dtype: np.dtype) -> str:
    """Map a numpy dtype to a zarr v3 ``data_type`` string."""
    kind, size = np_dtype.kind, np_dtype.itemsize
    if kind == "f":
        return {2: "float16", 4: "float32", 8: "float64"}[size]
    if kind == "i":
        return {1: "int8", 2: "int16", 4: "int32", 8: "int64"}[size]
    if kind == "u":
        return {1: "uint8", 2: "uint16", 4: "uint32", 8: "uint64"}[size]
    if kind == "b":
        return "bool"
    raise StoreError(f"unsupported numpy dtype for v3 write: {np_dtype}")


def _default_v3_codecs(np_dtype: np.dtype) -> list[dict[str, Any]]:
    """The codec pipeline scdata writes by default for a numeric dtype."""
    return _v3_codecs(np_dtype, _DEFAULT_COMPRESSOR)


def _v3_bytes_codecs(np_dtype: np.dtype) -> list[dict[str, Any]]:
    """The v3 ArrayBytes serializer for raw little-endian numeric chunks."""
    serializer = {"name": "bytes"}
    if np_dtype.itemsize > 1:
        serializer = {"name": "bytes", "configuration": {"endian": "little"}}
    return [serializer]


def _v3_codecs(np_dtype: np.dtype, compressor: _Compressor) -> list[dict[str, Any]]:
    """Return the v3 serializer plus optional BytesBytes compressor codecs."""
    codecs = _v3_bytes_codecs(np_dtype)
    cfg = _compressor_config(compressor, np_dtype)
    if cfg is None:
        return codecs
    if cfg["id"] == "blosc":
        codecs.append(
            {
                "name": "blosc",
                "configuration": {
                    "typesize": int(cfg["typesize"]),
                    "cname": str(cfg["cname"]),
                    "clevel": int(cfg["clevel"]),
                    "shuffle": _blosc_shuffle_name(cfg["shuffle"]),
                    "blocksize": int(cfg["blocksize"]),
                },
            }
        )
        return codecs
    raise StoreError(f"unsupported compressor id: {cfg['id']!r}")


def _zarr_compressors(np_dtype: np.dtype, compressor: _Compressor) -> tuple[Any, ...]:
    """Return zarr v3 BytesBytes codec objects for dense arrays."""
    cfg = _compressor_config(compressor, np_dtype)
    if cfg is None:
        return ()
    if cfg["id"] == "blosc":
        from zarr.codecs import BloscCodec

        return (
            BloscCodec(
                typesize=int(cfg["typesize"]),
                cname=str(cfg["cname"]),
                clevel=int(cfg["clevel"]),
                shuffle=_blosc_shuffle_name(cfg["shuffle"]),
                blocksize=int(cfg["blocksize"]),
            ),
        )
    raise StoreError(f"unsupported compressor id: {cfg['id']!r}")


def _encode_chunk_bytes(raw: bytes, np_dtype: np.dtype, compressor: _Compressor) -> bytes:
    """Apply the write compressor to one manually-written chunk."""
    cfg = _compressor_config(compressor, np_dtype)
    if cfg is None:
        return raw
    if cfg["id"] == "blosc":
        from numcodecs import Blosc

        return bytes(
            Blosc(
                cname=str(cfg["cname"]),
                clevel=int(cfg["clevel"]),
                shuffle=int(cfg["shuffle"]),
                blocksize=int(cfg["blocksize"]),
                typesize=int(cfg["typesize"]),
            ).encode(raw)
        )
    raise StoreError(f"unsupported compressor id: {cfg['id']!r}")


def _compressor_config(compressor: _Compressor, np_dtype: np.dtype) -> dict[str, Any] | None:
    """Normalize public compressor input to a numcodecs-compatible config."""
    if compressor is None:
        return None
    if isinstance(compressor, str):
        return _compressor_config_from_string(compressor, np_dtype)
    if isinstance(compressor, Mapping):
        return _compressor_config_from_mapping(compressor, np_dtype)
    raise StoreError(
        "compressor must be None, a string such as 'blosc.lz4.level5', "
        f"or a mapping, got {type(compressor).__name__}"
    )


def _compressor_config_from_string(text: str, np_dtype: np.dtype) -> dict[str, Any] | None:
    value = text.strip().lower()
    if value in ("", "none", "null", "false", "0", "uncompressed"):
        return None
    if value == "blosc":
        return _blosc_config(np_dtype=np_dtype)
    if value.startswith("blosc."):
        parts = [part for part in value.split(".") if part]
        if len(parts) < 2:
            return _blosc_config(np_dtype=np_dtype)
        cname = parts[1]
        clevel = 5
        shuffle: int | str | None = 1
        for part in parts[2:]:
            if part.startswith("level"):
                clevel = int(part.removeprefix("level"))
            elif part.startswith("clevel"):
                clevel = int(part.removeprefix("clevel"))
            elif part.isdigit():
                clevel = int(part)
            elif part in ("noshuffle", "none", "shuffle", "bitshuffle"):
                shuffle = part
            else:
                raise StoreError(f"unsupported blosc compressor option: {part!r}")
        return _blosc_config(np_dtype=np_dtype, cname=cname, clevel=clevel, shuffle=shuffle)
    raise StoreError(
        f"unsupported compressor {text!r}; supported values include "
        "'blosc.lz4.level5' and None"
    )


def _compressor_config_from_mapping(
    value: Mapping[str, Any], np_dtype: np.dtype
) -> dict[str, Any] | None:
    name = value.get("id", value.get("name"))
    if name is None:
        raise StoreError("compressor mapping must include 'id' or 'name'")
    codec_id = str(name).strip().lower()
    if codec_id in ("none", "null", "uncompressed"):
        return None
    cfg = value.get("configuration")
    options = dict(cfg) if isinstance(cfg, Mapping) else dict(value)
    if codec_id == "blosc":
        return _blosc_config(
            np_dtype=np_dtype,
            cname=str(options.get("cname", "lz4")),
            clevel=int(options.get("clevel", options.get("level", 5))),
            shuffle=options.get("shuffle", 1),
            blocksize=int(options.get("blocksize", 0)),
            typesize=int(options.get("typesize", np_dtype.itemsize)),
        )
    raise StoreError(f"unsupported compressor id: {codec_id!r}")


def _blosc_config(
    *,
    np_dtype: np.dtype,
    cname: str = "lz4",
    clevel: int = 5,
    shuffle: int | str | None = 1,
    blocksize: int = 0,
    typesize: int | None = None,
) -> dict[str, Any]:
    if not 0 <= int(clevel) <= 9:
        raise StoreError(f"blosc clevel must be between 0 and 9, got {clevel}")
    if int(blocksize) < 0:
        raise StoreError(f"blosc blocksize must be non-negative, got {blocksize}")
    parsed_typesize = int(np_dtype.itemsize if typesize is None else typesize)
    if parsed_typesize <= 0:
        raise StoreError(f"blosc typesize must be positive, got {parsed_typesize}")
    return {
        "id": "blosc",
        "cname": str(cname),
        "clevel": int(clevel),
        "shuffle": _blosc_shuffle_int(shuffle),
        "blocksize": int(blocksize),
        "typesize": parsed_typesize,
    }


def _blosc_shuffle_int(value: int | str | None) -> int:
    if value is None:
        return 1
    if isinstance(value, str):
        text = value.strip().lower()
        names = {
            "0": 0,
            "none": 0,
            "noshuffle": 0,
            "no_shuffle": 0,
            "1": 1,
            "shuffle": 1,
            "byte": 1,
            "2": 2,
            "bitshuffle": 2,
            "bit_shuffle": 2,
        }
        if text in names:
            return names[text]
        raise StoreError(f"unsupported blosc shuffle value: {value!r}")
    parsed = int(value)
    if parsed not in (0, 1, 2):
        raise StoreError(f"unsupported blosc shuffle value: {value!r}")
    return parsed


def _blosc_shuffle_name(value: int | str | None) -> str:
    return {0: "noshuffle", 1: "shuffle", 2: "bitshuffle"}[_blosc_shuffle_int(value)]


def _v3_array_json(
    *,
    shape: list[int],
    data_type: str,
    chunk_grid: dict[str, Any],
    codecs: list[dict[str, Any]],
    fill_value: Any,
    attrs: dict[str, Any],
) -> dict[str, Any]:
    """Build a v3 array ``zarr.json`` dict."""
    return {
        "shape": shape,
        "data_type": data_type,
        "chunk_grid": chunk_grid,
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": fill_value,
        "codecs": codecs,
        "attributes": attrs,
        "zarr_format": 3,
        "node_type": "array",
        "storage_transformers": [],
    }


def _write_v3_node(g: Any, name: str, meta: dict[str, Any], *, is_group: bool) -> None:
    """Write a raw v3 ``zarr.json`` for a node under ``g``.

    Used for nodes zarr's high-level API cannot create (rectilinear arrays).
    The node must not already exist.
    """
    store = g.store
    key = _group_store_key(g, f"{name}/zarr.json")
    _store_set_bytes(store, key, (json.dumps(meta) + "\n").encode("utf-8"))


def _group_store_key(g: Any, key: str) -> str:
    """Return a store-root-relative key for ``key`` under group ``g``."""
    group_path = str(getattr(g, "path", "") or "").strip("/")
    return f"{group_path}/{key}" if group_path else key


def _validate_write_options(
    *,
    format: str,
    layer_format: str,
    chunk_size: int | list[int] | tuple[int, ...],
    store: str,
    compressor: _Compressor,
) -> None:
    """Validate cheap write options before touching the output path."""
    if store not in ("zip", "dir"):
        raise StoreError(f"unsupported store kind: {store!r}")
    if format not in ("dense2d", "dense1d", "sparse"):
        raise StoreError(f"unsupported X format: {format!r}")
    if layer_format not in ("preserve", "auto", "dense2d", "dense1d", "sparse"):
        raise StoreError(f"unsupported layer_format {layer_format!r}")
    _validate_chunk_size_values(chunk_size)
    _compressor_config(compressor, np.dtype("float32"))


def _validate_chunk_size_values(chunk_size: int | list[int] | tuple[int, ...]) -> None:
    """Validate the numeric part of ``chunk_size`` independent of layout rank."""
    values = chunk_size if isinstance(chunk_size, (list, tuple)) else (chunk_size,)
    if not values:
        raise StoreError("chunk_size must not be empty")
    for value in values:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as err:
            raise StoreError(f"chunk_size entries must be integers, got {value!r}") from err
        if parsed <= 0:
            raise StoreError(f"chunk_size entries must be positive, got {parsed}")


def _make_zarr_store(root: Path, store: _Store) -> Any:
    if store == "zip":
        raise StoreError("internal error: zip writes must use a memory store")
    if store == "dir":
        return str(root)
    raise StoreError(f"unsupported store kind: {store!r}")


def _prepare_zip_target(root: Path) -> None:
    """Validate and prepare a zip target without deleting existing files."""
    if root.is_dir():
        raise StoreError(f"zip output path is a directory: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)


def _make_temp_dir(target: Path) -> Path:
    """Create a sibling temporary directory for an output directory store."""
    import tempfile

    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent))


def _zip_store(store: Any, target: Path) -> None:
    """Pack an in-memory zarr store as a ZIP_STORED archive."""
    import tempfile

    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for key in sorted(_store_list(store)):
                zf.writestr(key, _store_get_bytes(store, key))
        os.replace(tmp, target)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _replace_directory(source: Path, target: Path) -> None:
    """Replace ``target`` with completed ``source``, rolling back on failure."""
    backup: Path | None = None
    if target.exists() or target.is_symlink():
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
    """Reserve a sibling path for temporarily moving an existing target."""
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".bak", dir=target.parent))
    tmp.rmdir()
    return tmp


def _remove_path(path: Path) -> None:
    """Remove a file, symlink, or directory tree if it still exists."""
    import shutil

    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# public read
# ---------------------------------------------------------------------------


def read_zarr(
    path: str | os.PathLike[str],
    *,
    metadata_only: bool = False,
) -> "AnnData":
    """Read a scdata zarr v3 store into an :class:`anndata.AnnData`.

    Stock ``anndata.read_zarr`` reads ``dense2d`` and regular-grid ``sparse``
    stores directly; this function additionally handles scdata's two
    extensions — the ``dense1d`` flattened X (reshaped to 2D on read) and the
    rectilinear cell-aligned CSR layout (read chunk-by-chunk) — and opens
    ``.zarr.zip`` archives (which stock ``anndata.read_zarr`` cannot, since
    ``zarr.open`` does not treat a ``.zarr.zip`` as a :class:`ZipStore`).

    The read path mirrors :func:`anndata.read_zarr` for everything outside the
    scdata matrix extensions, so legacy ``raw.*`` groups, pre-0.7 categorical
    cleanup, and Array-form ``obs`` are handled the same way stock anndata
    handles them — no data is dropped on third-party / older stores.

    Parameters
    ----------
    path:
        Store path — a zarr v3 directory or a ``.zarr.zip`` archive.
    metadata_only:
        When true, load every annotation (``obs``, ``var``, ``uns``, ``obsm``,
        ``varm``, ``obsp``, ``varp``) but skip the expression matrices
        (``X``, ``layers``), which are set to ``None`` / omitted, and drop
        ``raw`` entirely.  ``n_obs`` / ``n_vars`` come from ``obs`` / ``var``.
        Useful for inspecting a store without paying the matrix-load cost.
        ``raw`` is dropped (not just ``raw.X``) because anndata's in-memory
        :class:`~anndata.Raw` cannot exist without an X — it falls back to
        ``adata.X``, which is ``None`` in this mode; use a full read for raw.
        Stock ``anndata.read_zarr`` has no equivalent mode; the closest,
        :func:`anndata.experimental.read_lazy`, is fully lazy (it backs
        ``obs`` / ``var`` too) and cannot read scdata's ``dense1d`` or
        rectilinear layouts.
    """
    import zarr

    import anndata as ad
    from anndata._io.specs import read_elem
    from anndata._io.utils import _read_legacy_raw
    from anndata._io.zarr import read_dataframe
    from anndata.compat import _clean_uns
    from anndata.experimental import read_dispatched

    _enable_rectilinear()
    f = _open_store_for_read(path)

    try:
        def callback(read_func: Any, elem_name: str, elem: Any, *, iospec: Any) -> Any:
            name = elem_name.lstrip("/")
            attrs = _node_attrs(elem)
            kind = _matrix_kind(attrs)
            # scdata-extended matrix roots (``X``, ``layers/*``, ``raw/X``):
            # under ``metadata_only`` every matrix root short-circuits to
            # ``None`` so no matrix bytes are read; otherwise the scdata layouts
            # (``dense1d`` / ``sparse`` / ``sparse-vlen``) are rebuilt here, and
            # non-scdata matrices (e.g. a third-party ``dense2d`` X) fall through
            # to the default reader below.
            if _is_matrix_root(name):
                if metadata_only:
                    return None
                if kind in ("dense1d", "sparse", "sparse-vlen"):
                    return _read_matrix_scdata(f, name, kind, attrs)
            if iospec.encoding_type == "anndata" or elem_name.endswith("/"):
                # ``read_dispatched`` returns the anndata ``RWAble`` union; the
                # values are passed straight into the AnnData constructor, whose
                # own kwargs accept them at runtime.  Annotate as ``dict[str, Any]``
                # so pyright does not flag each RWAble member against the ctor.
                kwargs: dict[str, Any] = {}
                for k, v in dict(elem).items():
                    if k.startswith("raw."):
                        continue
                    if metadata_only and k == "X":
                        kwargs["X"] = None
                        continue
                    if metadata_only and k == "layers":
                        # ``AnnData.layers`` cannot hold ``None``; emit an empty
                        # dict instead of recursing (which would load matrices).
                        kwargs["layers"] = {}
                        continue
                    kwargs[k] = read_dispatched(v, callback=callback)
                return ad.AnnData(**kwargs)
            if elem_name.startswith("/raw."):
                return None
            if elem_name in {"/obs", "/var"}:
                return read_dataframe(elem)
            if elem_name == "/raw":
                if metadata_only:
                    # anndata's in-memory :class:`~anndata.Raw` cannot exist
                    # without an X — ``Raw.__init__`` falls back to
                    # ``adata.X.copy()``, which is ``None`` here, and a lazy zarr
                    # proxy only works for ``dense2d`` / regular-sparse layouts,
                    # not scdata's ``dense1d`` / rectilinear CSR.  Drop raw
                    # entirely in metadata-only mode; use a full read for raw.
                    return None
                # Modern raw group is read by the default reader (which recurses
                # through the callback, so ``raw/X`` picks up the scdata matrix
                # path).  Guard against a coexisting legacy ``raw.*`` layout the
                # way :func:`anndata.read_zarr` does.
                modern_raw = read_func(elem)
                if any(k.startswith("raw.") for k in f):
                    raise StoreError(
                        "store has both a modern 'raw' group and legacy 'raw.*' keys"
                    )
                return modern_raw
            return read_func(elem)

        adata = cast("AnnData", read_dispatched(f, callback=callback))

        # Legacy raw rebuild: pre-modern-raw-group stores keep ``raw.X`` /
        # ``raw.var`` / ``raw.varm`` flat at the root.  The root callback skips
        # ``raw.*`` keys, so rebuild raw here (matches :func:`anndata.read_zarr`).
        # Skipped under ``metadata_only`` — raw is dropped there (see ``/raw``
        # branch) and loading legacy ``raw.X`` would defeat the mode.
        if not metadata_only and "raw.X" in f:
            raw_kwargs = _read_legacy_raw(f, None, read_dataframe, read_elem)
            raw = ad.AnnData(**raw_kwargs)
            raw.obs_names = adata.obs_names
            adata.raw = raw

        # Pre-0.7 compat: ``obs`` stored as a zarr.Array leaks categoricals into
        # ``uns``; clean them the way stock anndata does.
        if isinstance(f["obs"], zarr.Array):
            _clean_uns(adata)

        return adata
    finally:
        close = getattr(getattr(f, "store", None), "close", None)
        if close is not None:
            close()


def _node_attrs(elem: Any) -> dict[str, Any]:
    attrs = getattr(elem, "attrs", None)
    if attrs is None:
        return {}
    return dict(attrs)


def _matrix_kind(attrs: dict[str, Any]) -> Any:
    return attrs.get(_SCDATA_MATRIX_ATTR) or attrs.get(_SCDATA_X_ATTR)


def _is_matrix_root(name: str) -> bool:
    if name == "X":
        return True
    parts = name.split("/")
    if len(parts) == 2 and parts[0] == "layers" and bool(parts[1]):
        return True
    # ``raw.X`` is written by :func:`_write_raw` in the same scdata layouts as
    # ``X`` (cell-aligned CSR / dense1d), so read it through the scdata path
    # too.  ``Raw`` has no layers, so only ``raw/X`` needs handling.
    return name == "raw/X"


def _read_matrix_scdata(f: Any, matrix_key: str, kind: str, attrs: dict[str, Any]) -> Any:
    """Read a scdata-extended matrix (dense1d reshape or CSR rebuild)."""
    if kind == "dense1d":
        from anndata._io.specs import read_elem

        shape2d = attrs.get(_SCDATA_SHAPE_2D)
        flat = read_elem(f[matrix_key])
        if shape2d is None:
            raise StoreError(f"dense1d {matrix_key} missing scdata-shape-2d attribute")
        return np.asarray(flat).reshape(int(shape2d[0]), int(shape2d[1]))
    if kind in ("sparse", "sparse-vlen"):
        return _read_csr(f, matrix_key)
    raise StoreError(f"unsupported scdata matrix kind: {kind!r}")


def _read_csr(f: Any, matrix_key: str) -> Any:
    """Rebuild a CSR matrix, reading rectilinear chunks where needed."""
    from scipy import sparse

    x = f[matrix_key]
    shape = list(x.attrs.get("shape", []))
    indptr = _read_sub_array(f, f"{matrix_key}/indptr")
    indices = _read_sub_array(f, f"{matrix_key}/indices")
    data = _read_sub_array(f, f"{matrix_key}/data")
    n_obs = int(shape[0]) if len(shape) >= 1 else int(indptr.shape[0]) - 1
    n_var = int(shape[1]) if len(shape) >= 2 else 0
    return sparse.csr_matrix(
        (np.asarray(data), np.asarray(indices), np.asarray(indptr)),
        shape=(n_obs, n_var),
    )


def _read_sub_array(f: Any, key: str) -> np.ndarray:
    """Read a 1D sub-array of X (indptr/indices/data).

    Regular-grid arrays go through ``read_elem``.  Rectilinear arrays are read
    through zarr's array slicing path: ``read_elem`` may inspect regular-only
    ``.chunks`` metadata, while ``node[:]`` supports rectilinear grids once the
    zarr feature flag is enabled.
    """
    from anndata._io.specs import read_elem

    node = f[key]
    attrs = dict(node.attrs)
    if _matrix_kind(attrs) != "sparse-vlen":
        return np.asarray(read_elem(node))
    try:
        return np.asarray(node[:])
    except Exception as err:
        raise StoreError(f"{key}: failed to read rectilinear array") from err


def _store_set_bytes(store: Any, key: str, value: bytes) -> None:
    """Write raw bytes through either zarr v3's async store API or old mapping stores."""
    set_async = getattr(store, "set", None)
    if set_async is None:
        store[key] = value
        return

    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync

    buffer = default_buffer_prototype().buffer.from_bytes(value)
    sync(set_async(key, buffer))


def _store_get_bytes(store: Any, key: str) -> bytes:
    """Read raw bytes through either zarr v3's async store API or old mapping stores."""
    get_async = getattr(store, "get", None)
    if get_async is None:
        raw = store[key]
        return bytes(raw) if isinstance(raw, memoryview) else raw

    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync

    raw = sync(get_async(key, default_buffer_prototype()))
    if raw is None:
        raise KeyError(key)
    if hasattr(raw, "to_bytes"):
        return raw.to_bytes()
    return bytes(raw) if isinstance(raw, memoryview) else raw


def _store_list(store: Any, prefix: str = "") -> list[str]:
    """List the keys in a zarr store (directory or ZipStore)."""
    list_prefix = getattr(store, "list_prefix", None)
    if list_prefix is not None:
        import inspect

        from zarr.core.sync import collect_aiterator, sync

        keys = collect_aiterator(list_prefix(prefix))
        # ``collect_aiterator`` is typed as ``tuple & Awaitable``; ``sync``
        # expects a Coroutine.  The awaitable branch is the coroutine form.
        return list(sync(cast(Any, keys)) if inspect.isawaitable(keys) else keys)

    list_all = getattr(store, "list", None)
    if list_all is not None:
        import inspect

        from zarr.core.sync import collect_aiterator, sync

        collected = collect_aiterator(list_all())
        keys = list(sync(cast(Any, collected)) if inspect.isawaitable(collected) else collected)
        return [k for k in keys if k.startswith(prefix)]

    try:
        keys = list(store.keys())
    except Exception:
        keys = [k for k in store]
    return [k for k in keys if k.startswith(prefix)]


def _open_store_for_read(path: str | os.PathLike[str]) -> Any:
    """Open a zarr v3 store for reading (directory or ``.zarr.zip``)."""
    import zarr
    from zarr.storage import ZipStore

    p = Path(os.fspath(path))
    if p.is_file():
        store = ZipStore(str(p), mode="r")
        return zarr.open_group(store, mode="r")
    return zarr.open(str(p), mode="r")
