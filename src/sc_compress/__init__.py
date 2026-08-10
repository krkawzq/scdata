"""NumPy bindings for the sc-compress matrix format.

* Dense I/O uses :class:`ScDense` (NumPy-backed)
* CSR I/O uses :class:`ScCsr` (NumPy-backed, SciPy-free hot path)
* On-demand row/column select is implemented in Rust
* ``sc_compress._core`` is the private Rust extension
* ZIP helpers live in :mod:`sc_compress.zip`
"""

from __future__ import annotations

from . import zip
from ._core import __version__
from .array import ScCsr, ScDense
from .exceptions import (
    AllocationError,
    CodecError,
    CorruptDataError,
    InvalidArgumentError,
    InvalidMetaError,
    IoError,
    JsonError,
    NotFoundError,
    PathError,
    PerformanceWarning,
    ScCompressError,
    ScCompressWarning,
    ZipError,
    error_kind,
)
from .format import (
    FORMAT_NAME,
    FORMAT_VERSION,
    INDEX_DTYPES,
    STORAGE_INDEX_DTYPES,
    STORAGE_VALUE_DTYPES,
    VALUE_DTYPES,
    is_index_dtype,
    is_value_dtype,
)
from .io import (
    open_store,
    write,
    write_csr,
    write_csr_arrays,
    write_dense,
)
from .limits import DEFAULT_N_WORKERS, DEFAULT_READ_LIMITS, ReadLimits
from .store import Store, StoreInfo
from .write_options import DEFAULT_WRITE_OPTIONS, WriteOptions

# Optional AnnData bridge (imports anndata / zarr lazily inside the functions).
from .anndata import read_scc, write_scc

__all__ = [
    "DEFAULT_READ_LIMITS",
    "DEFAULT_N_WORKERS",
    "DEFAULT_WRITE_OPTIONS",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "INDEX_DTYPES",
    "STORAGE_INDEX_DTYPES",
    "STORAGE_VALUE_DTYPES",
    "VALUE_DTYPES",
    "AllocationError",
    "CodecError",
    "CorruptDataError",
    "InvalidArgumentError",
    "InvalidMetaError",
    "IoError",
    "JsonError",
    "NotFoundError",
    "PathError",
    "PerformanceWarning",
    "ReadLimits",
    "ScCompressError",
    "ScCompressWarning",
    "ScCsr",
    "ScDense",
    "Store",
    "StoreInfo",
    "WriteOptions",
    "ZipError",
    "__version__",
    "error_kind",
    "is_index_dtype",
    "is_value_dtype",
    "open_store",
    "read_scc",
    "write",
    "write_csr",
    "write_csr_arrays",
    "write_dense",
    "write_scc",
    "zip",
]
