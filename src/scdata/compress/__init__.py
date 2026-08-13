"""sc-compress store I/O: open, write, and on-demand select."""

from scdata.exceptions import Error, InvalidArgumentError, PerformanceWarning
from scdata.matrix import ScCsr, ScDense
from scdata.compress.format import (
    FORMAT_NAME,
    FORMAT_VERSION,
    INDEX_DTYPES,
    STORAGE_INDEX_DTYPES,
    STORAGE_VALUE_DTYPES,
    VALUE_DTYPES,
    is_index_dtype,
    is_value_dtype,
)
from scdata.compress.io import (
    open_store,
    write,
    write_csr,
    write_csr_arrays,
    write_dense,
)
from scdata.compress.limits import DEFAULT_N_WORKERS, DEFAULT_READ_LIMITS, ReadLimits
from scdata.compress.store import Store, StoreInfo
from scdata.compress.write_options import (
    DEFAULT_BLOCK_BUDGET,
    DEFAULT_CHUNK_BUDGET,
    DEFAULT_WRITE_OPTIONS,
    WriteOptions,
)
from scdata.compress import zip

__all__ = [
    "DEFAULT_BLOCK_BUDGET",
    "DEFAULT_CHUNK_BUDGET",
    "DEFAULT_N_WORKERS",
    "DEFAULT_READ_LIMITS",
    "DEFAULT_WRITE_OPTIONS",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "INDEX_DTYPES",
    "STORAGE_INDEX_DTYPES",
    "STORAGE_VALUE_DTYPES",
    "VALUE_DTYPES",
    "Error",
    "InvalidArgumentError",
    "PerformanceWarning",
    "ReadLimits",
    "ScCsr",
    "ScDense",
    "Store",
    "StoreInfo",
    "WriteOptions",
    "is_index_dtype",
    "is_value_dtype",
    "open",
    "open_store",
    "write",
    "write_csr",
    "write_csr_arrays",
    "write_dense",
    "zip",
]

open = open_store
