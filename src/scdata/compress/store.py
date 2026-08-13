"""Opened matrix store with zarr-like on-demand row/column access."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from scdata import _core
from scdata.matrix._index import AxisSpec, normalize_key, rust_payload
from scdata.compress._validate import as_int, normalize_row_range
from scdata.matrix.array import ScCsr, ScDense
from scdata.exceptions import _call_core, _invalid_argument
from scdata.compress.format import dtype_from_storage
from scdata.compress.limits import ReadLimits

__all__ = ["Store", "StoreInfo"]


@dataclass(frozen=True, slots=True)
class StoreInfo:
    """Immutable metadata snapshot returned by :meth:`Store.info`."""

    path: Path
    zip_prefix: str | None
    kind: Literal["dense", "csr"]
    shape: tuple[int, int]
    dtype: np.dtype[np.generic]
    index_dtype: np.dtype[np.generic] | None
    storage_dtype: str
    storage_index_dtype: str | None
    nnz: int | None
    n_workers: int
    limits: ReadLimits

    def as_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for inspection or logging."""
        return {
            "path": self.path,
            "zip_prefix": self.zip_prefix,
            "kind": self.kind,
            "shape": self.shape,
            "dtype": self.dtype,
            "index_dtype": self.index_dtype,
            "storage_dtype": self.storage_dtype,
            "storage_index_dtype": self.storage_index_dtype,
            "nnz": self.nnz,
            "n_workers": self.n_workers,
            "limits": self.limits,
        }


class Store:
    """Opened matrix store with lazy, zarr-like on-demand slicing.

    Construct via :func:`scdata.compress.open`. Indexing materializes only the
    requested rows/columns through Rust kernels:

    * ``store[i:j]`` / ``store[rows]`` — row selection
    * ``store[rows, cols]`` — 2-D selection (cells × genes)
    * fancy indices, bool masks, and strided slices are all supported

    Dense stores return :class:`ScDense`; CSR stores return :class:`ScCsr`
    (or :class:`ScDense` when densified).
    """

    __slots__ = ("_handle", "_meta", "_path", "_zip_prefix")

    def __init__(
        self,
        handle: _core._Store,
        path: str | PathLike[str],
        zip_prefix: str | None,
        /,
    ) -> None:
        if not isinstance(handle, _core._Store):
            raise TypeError("handle must be scdata._core._Store")
        self._handle: _core._Store | None = handle
        self._meta = _call_core(_core.store_meta, handle)
        self._path = Path(path)
        self._zip_prefix = zip_prefix

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has released the underlying reader."""
        return self._handle is None

    @property
    def path(self) -> Path:
        """Filesystem path of the directory or ZIP archive."""
        return self._path

    @property
    def zip_prefix(self) -> str | None:
        """Archive prefix, or ``None`` for a directory store."""
        return self._zip_prefix

    @property
    def kind(self) -> Literal["dense", "csr"]:
        return self._require_meta()["kind"]

    @property
    def shape(self) -> tuple[int, int]:
        rows, columns = self._require_meta()["shape"]
        return int(rows), int(columns)

    @property
    def n_rows(self) -> int:
        return self.shape[0]

    @property
    def n_cols(self) -> int:
        return self.shape[1]

    @property
    def dtype(self) -> np.dtype[np.generic]:
        """Matrix value dtype as a NumPy dtype."""
        return dtype_from_storage(self.storage_dtype)

    @property
    def index_dtype(self) -> np.dtype[np.generic] | None:
        """CSR index dtype as a NumPy dtype, or ``None`` for dense stores."""
        name = self.storage_index_dtype
        return None if name is None else dtype_from_storage(name)

    @property
    def storage_dtype(self) -> str:
        """On-disk value dtype name (for example ``"f32"``)."""
        return self._require_meta()["value_dtype"]

    @property
    def storage_index_dtype(self) -> str | None:
        """On-disk CSR index dtype name, or ``None`` for dense stores."""
        return self._require_meta()["index_dtype"]

    @property
    def nnz(self) -> int | None:
        value = self._require_meta()["nnz"]
        return None if value is None else int(value)

    @property
    def n_workers(self) -> int:
        """Maximum chunk workers used by each decode call."""
        return int(self._require_meta()["n_workers"])

    @property
    def limits(self) -> ReadLimits:
        meta = self._require_meta()
        return ReadLimits(
            max_metadata_size=meta["maximum_metadata_size"],
            max_encoded_size=meta["maximum_encoded_size"],
            max_decoded_size=meta["maximum_decoded_size"],
            max_block_count=meta["maximum_block_count"],
            n_workers=meta["n_workers"],
        )

    def info(self) -> StoreInfo:
        """Return a typed snapshot of the store metadata and active limits."""
        return StoreInfo(
            path=self.path,
            zip_prefix=self.zip_prefix,
            kind=self.kind,
            shape=self.shape,
            dtype=self.dtype,
            index_dtype=self.index_dtype,
            storage_dtype=self.storage_dtype,
            storage_index_dtype=self.storage_index_dtype,
            nnz=self.nnz,
            n_workers=self.n_workers,
            limits=self.limits,
        )

    def indptr(self) -> NDArray[np.uint64] | None:
        """Return a copy of the resident CSR row offsets, if applicable."""
        values = _call_core(_core.store_indptr, self._require_open())
        return None if values is None else np.asarray(values)

    def read_rows(
        self,
        start: int = 0,
        stop: int | None = None,
        *,
        cols: Any = slice(None),
        csr_output: Literal["sparse", "dense"] = "sparse",
    ) -> ScDense | ScCsr:
        """Decode the half-open row interval ``[start, stop)`` (optional cols)."""
        if stop is None:
            stop = self.n_rows
        start_i, stop_i = normalize_row_range(start, stop, self.n_rows)
        rows = AxisSpec("range", (start_i, stop_i), stop_i - start_i)
        from scdata.matrix._index import normalize_axis

        col_spec = normalize_axis(cols, self.n_cols, name="col")
        return self._select(rows, col_spec, csr_output=csr_output)

    def read(
        self,
        rows: Any = None,
        cols: Any = None,
        *,
        csr_output: Literal["sparse", "dense"] = "sparse",
    ) -> ScDense | ScCsr:
        """Read rows/columns with the same semantics as :meth:`select`.

        ``read()`` loads the full matrix. Passing only ``rows`` keeps all columns.
        """
        if rows is None and cols is None:
            return self.read_rows(csr_output=csr_output)
        if cols is None:
            return self.select(rows, slice(None), csr_output=csr_output)
        return self.select(rows, cols, csr_output=csr_output)

    def select(
        self,
        rows: Any = slice(None),
        cols: Any = slice(None),
        *,
        csr_output: Literal["sparse", "dense"] = "sparse",
    ) -> ScDense | ScCsr:
        """Explicit 2-D select (cells × genes)."""
        from scdata.matrix._index import normalize_axis

        r = normalize_axis(rows, self.n_rows, name="row")
        c = normalize_axis(cols, self.n_cols, name="col")
        return self._select(r, c, csr_output=csr_output)

    def iter_batches(self, batch_size: int = 1024) -> Iterator[ScDense | ScCsr]:
        """Yield contiguous row batches without decoding the full matrix."""
        size = as_int(batch_size, name="batch_size", minimum=1)
        for start in range(0, self.n_rows, size):
            yield self.read_rows(start, min(start + size, self.n_rows))

    def iter_rows(self, batch_size: int = 1024) -> Iterator[ScDense | ScCsr | NDArray[Any]]:
        """Yield individual rows while decoding efficiently in batches.

        Dense stores yield 1-D NumPy row vectors (NumPy-like squeeze).
        CSR stores yield single-row :class:`ScCsr` matrices.
        """
        for batch in self.iter_batches(batch_size):
            for row in range(batch.shape[0]):
                yield batch[row]

    def close(self) -> None:
        """Release the Rust reader (via ``Drop``). Safe to call more than once."""
        handle = self._handle
        self._handle = None
        self._meta = None
        del handle

    def __enter__(self) -> Store:
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __len__(self) -> int:
        return self.n_rows

    def __iter__(self) -> Iterator[ScDense | ScCsr | NDArray[Any]]:
        return self.iter_rows()

    def __getitem__(self, key: Any) -> ScDense | ScCsr | Any:
        rows, cols, drop = normalize_key(key, self.n_rows, self.n_cols)
        out = self._select(rows, cols, csr_output="sparse")
        # NumPy-like squeeze for a single scalar row with all columns (dense).
        if drop and isinstance(out, ScDense) and cols.kind == "all" and out.shape[0] == 1:
            return out.to_numpy()[0]
        return out

    def __repr__(self) -> str:
        if self.closed:
            return f"Store(path={str(self.path)!r}, closed=True)"
        location = str(self.path)
        if self.zip_prefix is not None:
            location = f"{location}!/{self.zip_prefix}"
        details = [
            f"kind={self.kind!r}",
            f"shape={self.shape!r}",
            f"dtype={self.dtype.name!r}",
        ]
        if self.nnz is not None:
            details.append(f"nnz={self.nnz}")
        details.append(f"path={location!r}")
        return f"Store({', '.join(details)})"

    def _require_open(self) -> _core._Store:
        if self._handle is None:
            _invalid_argument("I/O operation on closed scdata.Store")
        return self._handle

    def _require_meta(self) -> dict[str, Any]:
        if self._handle is None or self._meta is None:
            _invalid_argument("I/O operation on closed scdata.Store")
        return self._meta

    def _select(
        self,
        rows: AxisSpec,
        cols: AxisSpec,
        *,
        csr_output: Literal["sparse", "dense"] = "sparse",
    ) -> ScDense | ScCsr:
        handle = self._require_open()
        result = _call_core(
            _core.store_select,
            handle,
            rows.kind,
            rust_payload(rows),
            cols.kind,
            rust_payload(cols),
            csr_output=csr_output,
        )
        kind = result[0]
        workers = self.n_workers
        if kind == "dense":
            return ScDense(result[1], n_workers=workers)
        if kind == "csr":
            _, indices, data, indptr, shape = result
            return ScCsr(
                indptr=indptr,
                indices=indices,
                data=data,
                shape=(int(shape[0]), int(shape[1])),
                n_workers=workers,
            )
        _invalid_argument(f"unknown select result kind {kind!r}")
