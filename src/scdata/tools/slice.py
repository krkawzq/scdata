"""In-memory 2-D gather: ``slice(array)[rows, cols]``."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from scdata import _core
from scdata.compress._base import csr_from_raw, dense_from_raw, scalar_axes
from scdata.compress._index import normalize_key, rust_payload



class _SliceDense:
    __slots__ = ("_num_workers", "_values")

    def __init__(self, values: NDArray[Any], *, num_workers: int) -> None:
        self._values = values
        self._num_workers = num_workers

    def __getitem__(self, key: Any) -> Any:
        n_rows, n_cols = self._values.shape
        rows, cols, _ = normalize_key(key, n_rows, n_cols)
        array = _core.matrix_dense_select(
            self._values,
            rows.kind,
            rust_payload(rows),
            cols.kind,
            rust_payload(cols),
            num_workers=self._num_workers,
        )
        row_scalar, col_scalar = scalar_axes(key)
        if row_scalar and col_scalar:
            return array[0, 0]
        if row_scalar:
            return array[0]
        if col_scalar:
            return array[:, 0]
        return array


class _SliceCsr:
    __slots__ = ("_matrix", "_num_workers")

    def __init__(self, matrix: Any, *, num_workers: int) -> None:
        self._matrix = matrix
        self._num_workers = num_workers

    def __getitem__(self, key: Any) -> Any:
        n_rows, n_cols = (int(self._matrix.shape[0]), int(self._matrix.shape[1]))
        rows, cols, _ = normalize_key(key, n_rows, n_cols)
        row_scalar, col_scalar = scalar_axes(key)
        csr_output = "dense" if row_scalar and col_scalar else "sparse"
        result = _core.matrix_csr_select(
            np.ascontiguousarray(self._matrix.indptr, dtype=np.uint64),
            _csr_indices(self._matrix.indices),
            np.ascontiguousarray(self._matrix.data),
            n_rows,
            n_cols,
            rows.kind,
            rust_payload(rows),
            cols.kind,
            rust_payload(cols),
            csr_output=csr_output,
            num_workers=self._num_workers,
        )
        if row_scalar and col_scalar:
            return dense_from_raw(result)[0, 0]
        return csr_from_raw(result)


def slice(data: Any, *, num_workers: int = 1) -> _SliceDense | _SliceCsr:
    """Wrap a 2-D NumPy array or SciPy CSR matrix for ``wrapper[rows, cols]``."""
    if isinstance(num_workers, (bool, np.bool_)) or not isinstance(num_workers, (int, np.integer)):
        raise TypeError("num_workers must be an integer")
    workers = int(num_workers)
    if workers < 1:
        raise ValueError("num_workers must be greater than zero")

    sparse = _scipy_sparse_or_none()
    if sparse is not None and sparse.issparse(data):
        if getattr(data, "ndim", 2) != 2:
            raise ValueError(f"slice() only supports 2-D arrays, got ndim={data.ndim}")
        if getattr(data, "format", None) != "csr":
            raise TypeError(f"sparse slice() only supports CSR, got {type(data).__name__}")
        return _SliceCsr(data, num_workers=workers)

    try:
        array = np.asarray(data)
    except (TypeError, ValueError) as error:
        raise TypeError(f"slice() expected a 2-D array or CSR matrix: {error}") from error
    if array.ndim != 2:
        raise ValueError(f"slice() only supports 2-D arrays, got ndim={array.ndim}")
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return _SliceDense(array, num_workers=workers)


def _csr_indices(indices: Any) -> NDArray[Any]:
    array = np.ascontiguousarray(indices)
    if array.dtype == np.uint16 or array.dtype == np.uint32:
        return array
    if array.dtype.kind not in "iu":
        raise TypeError(f"CSR indices must be integer, got dtype {array.dtype}")
    if array.size and int(array.min()) < 0:
        raise ValueError("CSR indices must be non-negative")
    max_value = int(array.max()) if array.size else 0
    if max_value <= np.iinfo(np.uint16).max:
        return array.astype(np.uint16, copy=False)
    if max_value <= np.iinfo(np.uint32).max:
        return array.astype(np.uint32, copy=False)
    raise ValueError("CSR indices exceed uint32")


def _scipy_sparse_or_none() -> Any | None:
    try:
        from scipy import sparse
    except ImportError:
        return None
    return sparse
