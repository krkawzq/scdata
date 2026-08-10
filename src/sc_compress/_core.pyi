"""Type stubs for the Rust extension module ``sc_compress._core``."""

from __future__ import annotations

from os import PathLike
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

__version__: str
FORMAT_NAME: str
FORMAT_VERSION: int
VALUE_DTYPES: list[str]
INDEX_DTYPES: list[str]
DEFAULT_MAXIMUM_METADATA_SIZE: int
DEFAULT_MAXIMUM_ENCODED_SIZE: int
DEFAULT_MAXIMUM_DECODED_SIZE: int
DEFAULT_MAXIMUM_BLOCK_COUNT: int
DEFAULT_N_WORKERS: int

class ScCompressError(Exception):
    kind: str

class IoError(ScCompressError): ...
class JsonError(ScCompressError): ...
class CodecError(ScCompressError): ...
class ZipError(ScCompressError): ...
class AllocationError(ScCompressError): ...
class NotFoundError(ScCompressError): ...
class InvalidArgumentError(ScCompressError): ...
class InvalidMetaError(ScCompressError): ...
class CorruptDataError(ScCompressError): ...
class PathError(ScCompressError): ...

class _Store:
    @property
    def kind(self) -> Literal["dense", "csr"]: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def value_dtype(self) -> str: ...
    @property
    def index_dtype(self) -> str | None: ...
    @property
    def nnz(self) -> int | None: ...
    @property
    def maximum_metadata_size(self) -> int: ...
    @property
    def maximum_encoded_size(self) -> int: ...
    @property
    def maximum_decoded_size(self) -> int: ...
    @property
    def maximum_block_count(self) -> int: ...
    @property
    def n_workers(self) -> int: ...
    def indptr(self) -> NDArray[np.uint64] | None: ...
    def decode_dense_rows(self, start: int, end: int) -> NDArray[Any]: ...
    def decode_csr_rows(
        self, start: int, end: int
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[np.uint64]]: ...
    def select(
        self,
        row_kind: str,
        row_payload: Any,
        col_kind: str,
        col_payload: Any,
        *,
        csr_output: str = "sparse",
    ) -> Any: ...

def _dense_select(
    values: NDArray[Any],
    row_kind: str,
    row_payload: Any,
    col_kind: str,
    col_payload: Any,
    *,
    n_workers: int,
) -> NDArray[Any]: ...
def _csr_select(
    indptr: NDArray[np.uint64],
    indices: NDArray[Any],
    data: NDArray[Any],
    n_rows: int,
    n_cols: int,
    row_kind: str,
    row_payload: Any,
    col_kind: str,
    col_payload: Any,
    *,
    csr_output: str = "sparse",
    n_workers: int,
) -> Any: ...
def _csr_to_dense(
    indptr: NDArray[np.uint64],
    indices: NDArray[Any],
    data: NDArray[Any],
    n_rows: int,
    n_cols: int,
    *,
    n_workers: int,
) -> NDArray[Any]: ...
def _write_dense(
    path: str | PathLike[str],
    values: NDArray[Any],
    *,
    chunk_policy: str,
    chunk_n: int,
    block_policy: str,
    block_n: int,
    n_workers: int,
) -> None: ...
def _write_csr(
    path: str | PathLike[str],
    indptr: NDArray[np.uint64],
    indices: NDArray[np.uint64],
    data: NDArray[Any],
    n_rows: int,
    n_cols: int,
    *,
    chunk_policy: str,
    chunk_n: int,
    block_policy: str,
    block_n: int,
    n_workers: int,
) -> None: ...
def _open_store(
    path: str | PathLike[str],
    *,
    zip_prefix: str | None = None,
    maximum_metadata_size: int,
    maximum_encoded_size: int,
    maximum_decoded_size: int,
    maximum_block_count: int,
    n_workers: int,
) -> _Store: ...
