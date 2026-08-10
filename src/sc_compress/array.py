"""In-memory dense / CSR matrix types with Rust-backed hot kernels.

``ScDense`` and ``ScCsr`` hold data as NumPy arrays but do **not** depend on
SciPy for indexing, filtering, or densification. All performance-critical paths
go through ``sc_compress._core``.
"""

from __future__ import annotations

from operator import index
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from sc_compress import _core
from sc_compress._index import AxisSpec, normalize_axis, normalize_key, rust_payload
from sc_compress.exceptions import _invalid_argument
from sc_compress.limits import DEFAULT_N_WORKERS

__all__ = ["ScCsr", "ScDense"]


def _workers(n_workers: int | None) -> int:
    if n_workers is None:
        return int(DEFAULT_N_WORKERS)
    if not isinstance(n_workers, int) or n_workers < 1:
        _invalid_argument(f"n_workers must be a positive int, got {n_workers!r}")
    return n_workers


def _pack_from_core(result: Any, *, n_workers: int) -> ScDense | ScCsr:
    kind = result[0]
    if kind == "dense":
        return ScDense(result[1], n_workers=n_workers)
    if kind == "csr":
        _, indices, data, indptr, shape = result
        return ScCsr(
            indptr=indptr,
            indices=indices,
            data=data,
            shape=(int(shape[0]), int(shape[1])),
            n_workers=n_workers,
        )
    _invalid_argument(f"unknown select result kind {kind!r}")


class ScDense:
    """Row-major dense matrix backed by a C-contiguous NumPy array."""

    __slots__ = ("_data", "_n_workers")

    def __init__(
        self,
        data: NDArray[Any],
        *,
        n_workers: int | None = None,
        copy: bool = False,
    ) -> None:
        arr = np.asarray(data)
        if arr.ndim != 2:
            _invalid_argument(f"ScDense requires a 2-D array, got shape {arr.shape}")
        native = arr.dtype.newbyteorder("=")
        if arr.dtype != native:
            arr = arr.astype(native, copy=False)
        if copy or not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        self._data = arr
        self._n_workers = _workers(n_workers)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self._data.shape[0]), int(self._data.shape[1])

    @property
    def ndim(self) -> int:
        return 2

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._data.dtype

    @property
    def n_workers(self) -> int:
        return self._n_workers

    @property
    def data(self) -> NDArray[Any]:
        """Underlying NumPy buffer (read-only view preferred by callers)."""
        return self._data

    def to_numpy(self, *, copy: bool = False) -> NDArray[Any]:
        return self._data.copy() if copy else self._data

    def __array__(self, dtype: np.dtype[Any] | None = None) -> NDArray[Any]:
        if dtype is None:
            return self._data
        return np.asarray(self._data, dtype=dtype)

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return f"ScDense(shape={self.shape!r}, dtype={self.dtype.name!r})"

    def __getitem__(self, key: Any) -> ScDense | NDArray[Any]:
        rows, cols, squeeze = normalize_key(key, self.shape[0], self.shape[1])
        out = self._select(rows, cols)
        if squeeze and out.shape[0] == 1 and out.shape[1] == 1:
            return out.to_numpy()[0, 0]
        if isinstance(key, tuple) and len(key) == 2:
            # NumPy-like squeeze for scalar axes.
            r_scalar = rows.kind == "range" and rows.out_len == 1 and _is_scalar_key(key[0])
            c_scalar = cols.kind == "range" and cols.out_len == 1 and _is_scalar_key(key[1])
            if r_scalar and c_scalar:
                return out.to_numpy()[0, 0]
            if r_scalar:
                return out.to_numpy()[0]
            if c_scalar:
                return out.to_numpy()[:, 0]
        elif _is_scalar_key(key):
            return out.to_numpy()[0]
        return out

    def select(
        self,
        rows: Any = slice(None),
        cols: Any = slice(None),
        *,
        n_workers: int | None = None,
    ) -> ScDense:
        r = normalize_axis(rows, self.shape[0], name="row")
        c = normalize_axis(cols, self.shape[1], name="col")
        return self._select(r, c, n_workers=n_workers)

    def _select(
        self,
        rows: AxisSpec,
        cols: AxisSpec,
        *,
        n_workers: int | None = None,
    ) -> ScDense:
        workers = _workers(n_workers if n_workers is not None else self._n_workers)
        result = _core._dense_select(
            self._data,
            rows.kind,
            rust_payload(rows),
            cols.kind,
            rust_payload(cols),
            n_workers=workers,
        )
        return ScDense(result, n_workers=workers)


class ScCsr:
    """CSR sparse matrix with NumPy-backed ``indptr`` / ``indices`` / ``data``.

    Not a SciPy matrix. Use :meth:`to_scipy` only when optional SciPy interop is
    required. Hot operations (slice, gather, densify) call Rust kernels.
    """

    __slots__ = ("_data", "_indices", "_indptr", "_n_workers", "_shape")

    def __init__(
        self,
        *,
        indptr: NDArray[Any],
        indices: NDArray[Any],
        data: NDArray[Any],
        shape: tuple[int, int],
        n_workers: int | None = None,
        copy: bool = False,
    ) -> None:
        n_rows, n_cols = int(shape[0]), int(shape[1])
        indptr_a = np.ascontiguousarray(indptr, dtype=np.uint64) if copy else np.asarray(indptr)
        if indptr_a.dtype != np.uint64 or not indptr_a.flags.c_contiguous:
            indptr_a = np.ascontiguousarray(indptr_a, dtype=np.uint64)
        indices_a = np.asarray(indices)
        if indices_a.dtype not in (np.uint16, np.uint32):
            # Promote safely for construction; kernels accept u16/u32 only.
            max_idx = int(indices_a.max()) if indices_a.size else 0
            dtype = np.uint16 if max_idx <= np.iinfo(np.uint16).max else np.uint32
            indices_a = np.ascontiguousarray(indices_a, dtype=dtype)
        elif copy or not indices_a.flags.c_contiguous:
            indices_a = np.ascontiguousarray(indices_a)
        data_a = np.asarray(data)
        native = data_a.dtype.newbyteorder("=")
        if data_a.dtype != native:
            data_a = data_a.astype(native, copy=False)
        if copy or not data_a.flags.c_contiguous:
            data_a = np.ascontiguousarray(data_a)
        if indptr_a.ndim != 1 or indptr_a.shape[0] != n_rows + 1:
            _invalid_argument(f"indptr length must be n_rows+1={n_rows + 1}, got {indptr_a.shape}")
        if indices_a.ndim != 1 or data_a.ndim != 1 or indices_a.shape[0] != data_a.shape[0]:
            _invalid_argument("indices and data must be 1-D with equal length")
        self._indptr = indptr_a
        self._indices = indices_a
        self._data = data_a
        self._shape = (n_rows, n_cols)
        self._n_workers = _workers(n_workers)

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def ndim(self) -> int:
        return 2

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._data.dtype

    @property
    def index_dtype(self) -> np.dtype[Any]:
        return self._indices.dtype

    @property
    def nnz(self) -> int:
        return int(self._indptr[-1]) if self._indptr.size else 0

    @property
    def n_workers(self) -> int:
        return self._n_workers

    @property
    def indptr(self) -> NDArray[np.uint64]:
        return self._indptr

    @property
    def indices(self) -> NDArray[Any]:
        return self._indices

    @property
    def data(self) -> NDArray[Any]:
        return self._data

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return (
            f"ScCsr(shape={self.shape!r}, dtype={self.dtype.name!r}, "
            f"nnz={self.nnz}, index_dtype={self.index_dtype.name!r})"
        )

    def __getitem__(self, key: Any) -> ScCsr | ScDense | Any:
        rows, cols, _ = normalize_key(key, self.shape[0], self.shape[1])
        # Scalar cell access densifies a 1×1 view.
        if (
            rows.kind == "range"
            and rows.out_len == 1
            and cols.kind == "range"
            and cols.out_len == 1
            and _is_scalar_key(key if not isinstance(key, tuple) else key[0])
        ):
            if isinstance(key, tuple) and len(key) == 2 and _is_scalar_key(key[1]):
                dense = self._select(rows, cols, csr_output="dense")
                assert isinstance(dense, ScDense)
                return dense.to_numpy()[0, 0]
        out = self._select(rows, cols, csr_output="sparse")
        if isinstance(key, tuple) and len(key) == 2:
            r_scalar = _is_scalar_key(key[0])
            c_scalar = _is_scalar_key(key[1])
            if r_scalar and c_scalar:
                dense = self._select(rows, cols, csr_output="dense")
                assert isinstance(dense, ScDense)
                return dense.to_numpy()[0, 0]
        elif _is_scalar_key(key) and isinstance(out, ScCsr):
            return out
        return out

    def select(
        self,
        rows: Any = slice(None),
        cols: Any = slice(None),
        *,
        csr_output: Literal["sparse", "dense"] = "sparse",
        n_workers: int | None = None,
    ) -> ScCsr | ScDense:
        r = normalize_axis(rows, self.shape[0], name="row")
        c = normalize_axis(cols, self.shape[1], name="col")
        return self._select(r, c, csr_output=csr_output, n_workers=n_workers)

    def to_dense(self, *, n_workers: int | None = None) -> ScDense:
        workers = _workers(n_workers if n_workers is not None else self._n_workers)
        arr = _core._csr_to_dense(
            self._indptr,
            self._indices,
            self._data,
            self._shape[0],
            self._shape[1],
            n_workers=workers,
        )
        return ScDense(arr, n_workers=workers)

    def to_numpy(self, *, n_workers: int | None = None) -> NDArray[Any]:
        return self.to_dense(n_workers=n_workers).to_numpy()

    def toarray(self, *, n_workers: int | None = None) -> NDArray[Any]:
        """Alias of :meth:`to_numpy` (SciPy-style name)."""
        return self.to_numpy(n_workers=n_workers)

    def todense(self, *, n_workers: int | None = None) -> ScDense:
        """Alias of :meth:`to_dense`."""
        return self.to_dense(n_workers=n_workers)

    def to_scipy(self) -> Any:
        """Optional SciPy bridge (requires the ``scipy`` extra)."""
        try:
            from scipy import sparse
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ScCsr.to_scipy() requires scipy; install sc-compress[scipy]"
            ) from exc
        return sparse.csr_matrix(
            (self._data, self._indices.astype(np.int64, copy=False), self._indptr.astype(np.int64)),
            shape=self._shape,
        )

    def _select(
        self,
        rows: AxisSpec,
        cols: AxisSpec,
        *,
        csr_output: Literal["sparse", "dense"] = "sparse",
        n_workers: int | None = None,
    ) -> ScCsr | ScDense:
        workers = _workers(n_workers if n_workers is not None else self._n_workers)
        result = _core._csr_select(
            self._indptr,
            self._indices,
            self._data,
            self._shape[0],
            self._shape[1],
            rows.kind,
            rust_payload(rows),
            cols.kind,
            rust_payload(cols),
            csr_output=csr_output,
            n_workers=workers,
        )
        return _pack_from_core(result, n_workers=workers)


def sc_dense_from_store_result(array: NDArray[Any], *, n_workers: int) -> ScDense:
    return ScDense(array, n_workers=n_workers)


def sc_csr_from_store_result(
    indices: NDArray[Any],
    data: NDArray[Any],
    indptr: NDArray[Any],
    shape: tuple[int, int],
    *,
    n_workers: int,
) -> ScCsr:
    return ScCsr(
        indptr=indptr,
        indices=indices,
        data=data,
        shape=shape,
        n_workers=n_workers,
    )


def _is_scalar_key(key: Any) -> bool:
    if isinstance(key, (bool, np.bool_)):
        return False
    if isinstance(key, slice) or key is Ellipsis:
        return False
    if isinstance(key, np.ndarray):
        return key.ndim == 0 and key.dtype.kind in "iu"
    try:
        index(key)
    except TypeError:
        return False
    return True
