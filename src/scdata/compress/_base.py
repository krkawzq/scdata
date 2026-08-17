"""Shared on-disk handle for SCC dense and CSR matrices."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from operator import index
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from scdata import _core
from scdata.compress._index import AxisSpec, rust_payload
from scdata.exceptions import InternalError

if TYPE_CHECKING:
    from typing_extensions import Self

    from scdata.compress._codec import Codec
    from scdata.compress._csr import ScCsr
    from scdata.compress._dense import ScDense
    from scdata.compress._limits import ReadLimits


@dataclass(frozen=True)
class StoreInfo:
    """Immutable metadata for an opened SCC matrix."""

    path: Path
    zip_prefix: str | None
    kind: Literal["dense", "csr"]
    shape: tuple[int, int]
    dtype: np.dtype[np.generic]
    index_dtype: np.dtype[np.generic] | None
    storage_dtype: str
    storage_index_dtype: str | None
    nnz: int | None
    codec: Codec
    indptr_codec: Codec | None
    num_workers: int
    limits: ReadLimits

    def as_dict(self) -> dict[str, Any]:
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
            "codec": self.codec,
            "indptr_codec": self.indptr_codec,
            "num_workers": self.num_workers,
            "limits": self.limits,
        }


class MatrixStore:
    """Read-only SCC store. Slices materialize as NumPy or SciPy objects."""

    __slots__ = ("_handle", "_meta", "_path", "_zip_prefix")

    def __init__(
        self,
        handle: _core._Store,
        path: str | PathLike[str],
        zip_prefix: str | None,
        /,
        *,
        _meta: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(handle, _core._Store):
            raise TypeError("handle must be scdata._core._Store")
        self._handle: _core._Store | None = handle
        self._meta: dict[str, Any] | None = dict(
            _core.store_meta(handle) if _meta is None else _meta
        )
        self._path = Path(path)
        self._zip_prefix = zip_prefix

    @property
    def closed(self) -> bool:
        return self._handle is None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def zip_prefix(self) -> str | None:
        return self._zip_prefix

    @property
    def kind(self) -> Literal["dense", "csr"]:
        return cast(Literal["dense", "csr"], self._require_meta()["kind"])

    @property
    def shape(self) -> tuple[int, int]:
        rows, columns = self._require_meta()["shape"]
        return int(rows), int(columns)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def n_rows(self) -> int:
        return self.shape[0]

    @property
    def n_cols(self) -> int:
        return self.shape[1]

    @property
    def dtype(self) -> np.dtype[np.generic]:
        from scdata.compress._format import dtype_from_storage

        return dtype_from_storage(self.storage_dtype)

    @property
    def index_dtype(self) -> np.dtype[np.generic] | None:
        from scdata.compress._format import dtype_from_storage

        name = self.storage_index_dtype
        return None if name is None else dtype_from_storage(name)

    @property
    def storage_dtype(self) -> str:
        return str(self._require_meta()["value_dtype"])

    @property
    def storage_index_dtype(self) -> str | None:
        value = self._require_meta()["index_dtype"]
        return None if value is None else str(value)

    @property
    def codec(self) -> Codec:
        from scdata.compress._codec import _codec_from_wire

        return _codec_from_wire(self._require_meta()["compressor"])

    @property
    def indptr_codec(self) -> Codec | None:
        from scdata.compress._codec import _codec_from_wire

        payload = self._require_meta().get("indptr_compressor")
        return None if payload is None else _codec_from_wire(payload)

    @property
    def nnz(self) -> int | None:
        value = self._require_meta()["nnz"]
        return None if value is None else int(value)

    @property
    def num_workers(self) -> int:
        return int(self._require_meta()["num_workers"])

    @property
    def limits(self) -> ReadLimits:
        from scdata.compress._limits import ReadLimits

        meta = self._require_meta()
        return ReadLimits(
            max_metadata_size=meta["max_metadata_size"],
            max_encoded_size=meta["max_encoded_size"],
            max_decoded_size=meta["max_decoded_size"],
            max_block_count=meta["max_block_count"],
            num_workers=meta["num_workers"],
        )

    def info(self) -> StoreInfo:
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
            codec=self.codec,
            indptr_codec=self.indptr_codec,
            num_workers=self.num_workers,
            limits=self.limits,
        )

    def read_rows(
        self,
        start: int = 0,
        stop: int | None = None,
        *,
        cols: Any = slice(None),
        csr_output: Literal["sparse", "dense"] = "sparse",
    ) -> Any:
        """Materialize the half-open row interval ``[start, stop)``."""
        if stop is None:
            stop = self.n_rows
        from scdata.compress._validate import normalize_row_range

        start_i, stop_i = normalize_row_range(start, stop, self.n_rows)
        return self.select(slice(start_i, stop_i), cols, csr_output=csr_output)

    def read(
        self,
        rows: Any = None,
        cols: Any = None,
        *,
        csr_output: Literal["sparse", "dense"] = "sparse",
    ) -> Any:
        """Materialize all or part of the matrix into NumPy/SciPy memory."""
        row_key = slice(None) if rows is None else rows
        col_key = slice(None) if cols is None else cols
        return self.select(row_key, col_key, csr_output=csr_output)

    def select(
        self,
        rows: Any = slice(None),
        cols: Any = slice(None),
        *,
        csr_output: Literal["sparse", "dense"] = "sparse",
    ) -> Any:
        raise NotImplementedError

    def iter_batches(self, batch_size: int = 1024) -> Iterator[Any]:
        from scdata.compress._validate import as_int

        size = as_int(batch_size, name="batch_size", minimum=1)
        for start in range(0, self.n_rows, size):
            yield self.read_rows(start, min(start + size, self.n_rows))

    def iter_rows(self, batch_size: int = 1024) -> Iterator[Any]:
        for batch in self.iter_batches(batch_size):
            for row in range(batch.shape[0]):
                yield batch[row]

    def copy(self) -> Self:
        """Return another handle sharing the same native store."""
        handle = self._require_open()
        return type(self)(handle, self._path, self._zip_prefix, _meta=self._require_meta())

    def close(self) -> None:
        """Release the native reader. Safe to call repeatedly."""
        self._handle = None
        self._meta = None

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __len__(self) -> int:
        return self.n_rows

    def __iter__(self) -> Iterator[Any]:
        return self.iter_rows()

    def __repr__(self) -> str:
        class_name = type(self).__name__
        if self.closed:
            return f"{class_name}(path={str(self.path)!r}, closed=True)"
        location = str(self.path)
        if self.zip_prefix is not None:
            location = f"{location}!/{self.zip_prefix}"
        details = [f"shape={self.shape!r}", f"dtype={self.dtype.name!r}"]
        if self.nnz is not None:
            details.append(f"nnz={self.nnz}")
        details.append(f"path={location!r}")
        return f"{class_name}({', '.join(details)})"

    def _require_open(self) -> _core._Store:
        if self._handle is None:
            raise ValueError("I/O operation on closed SCC matrix")
        return self._handle

    def _require_meta(self) -> dict[str, Any]:
        if self._handle is None or self._meta is None:
            raise ValueError("I/O operation on closed SCC matrix")
        return self._meta

    def _select_raw(
        self,
        rows: AxisSpec,
        cols: AxisSpec,
        *,
        csr_output: Literal["sparse", "dense"],
    ) -> Any:
        return _core.store_select(
            self._require_open(),
            rows.kind,
            rust_payload(rows),
            cols.kind,
            rust_payload(cols),
            csr_output=csr_output,
        )


def open_matrix(
    handle: _core._Store,
    path: str | PathLike[str],
    zip_prefix: str | None,
) -> ScDense | ScCsr:
    """Wrap a private native handle in the matching public matrix class."""
    from scdata.compress._csr import ScCsr
    from scdata.compress._dense import ScDense

    meta = _core.store_meta(handle)
    kind = meta.get("kind")
    if kind == "dense":
        return ScDense(handle, path, zip_prefix, _meta=meta)
    if kind == "csr":
        return ScCsr(handle, path, zip_prefix, _meta=meta)
    raise InternalError(f"unknown SCC matrix kind {kind!r}")


def dense_from_raw(result: Any) -> NDArray[Any]:
    if not isinstance(result, tuple) or len(result) != 2 or result[0] != "dense":
        raise InternalError("native SCC select returned an invalid dense result")
    array = np.asarray(result[1])
    if array.ndim != 2:
        raise InternalError(f"native SCC dense result must be 2-D, got shape {array.shape}")
    return array


def csr_from_raw(result: Any) -> Any:
    if not isinstance(result, tuple) or len(result) != 5 or result[0] != "csr":
        raise InternalError("native SCC select returned an invalid CSR result")
    sparse = _scipy_sparse()
    _, indices, data, indptr, shape = result
    resolved_shape = (int(shape[0]), int(shape[1]))
    return sparse.csr_matrix(
        (
            np.asarray(data),
            np.asarray(indices, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=resolved_shape,
        copy=False,
    )


def scipy_sparse() -> Any:
    return _scipy_sparse()


def _scipy_sparse() -> Any:
    try:
        from scipy import sparse
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "reading an SCC CSR matrix requires scipy; install scdata-toolkit[scipy]"
        ) from error
    return sparse


def scalar_axes(key: Any) -> tuple[bool, bool]:
    if isinstance(key, tuple):
        if len(key) == 0:
            return False, False
        if len(key) == 1:
            return _is_scalar_key(key[0]), False
        if len(key) == 2:
            return _is_scalar_key(key[0]), _is_scalar_key(key[1])
        return False, False
    return _is_scalar_key(key), False


def _is_scalar_key(key: Any) -> bool:
    if isinstance(key, (bool, np.bool_)):
        return False
    if isinstance(key, slice) or key is Ellipsis or key is None:
        return False
    if isinstance(key, np.ndarray):
        return key.ndim == 0 and key.dtype.kind in "iu"
    try:
        index(key)
    except TypeError:
        return False
    return True
