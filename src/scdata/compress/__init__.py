"""SCC store I/O: open, write, and on-demand select."""

from scdata.compress._csr import ScCsr
from scdata.compress._dense import ScDense
from scdata.compress._format import (
    FORMAT_NAME,
    FORMAT_VERSION,
    INDEX_DTYPES,
    STORAGE_INDEX_DTYPES,
    STORAGE_VALUE_DTYPES,
    VALUE_DTYPES,
    is_index_dtype,
    is_value_dtype,
)
from scdata.compress._io import (
    open_store,
    write,
    write_csr,
    write_csr_arrays,
    write_dense,
)
from scdata.compress._limits import DEFAULT_READ_LIMITS, ReadLimits
from scdata.compress._write_options import (
    DEFAULT_BLOCK_BUDGET,
    DEFAULT_CHUNK_BUDGET,
    DEFAULT_WRITE_OPTIONS,
    WriteOptions,
)
from scdata.compress import _zip as zip
from scdata.exceptions import Error, InvalidArgumentError, PerformanceWarning

open = open_store

__all__ = [
    "DEFAULT_BLOCK_BUDGET",
    "DEFAULT_CHUNK_BUDGET",
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
