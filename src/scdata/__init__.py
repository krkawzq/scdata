"""SCC-backed matrices and high-throughput single-cell data loading."""

from __future__ import annotations

from scdata import anndata, compress, load, tools
from scdata.anndata import read_scc, write_scc
from scdata.compress import (
    DEFAULT_READ_LIMITS,
    DEFAULT_WRITE_OPTIONS,
    Codec,
    ReadLimits,
    ScCsr,
    ScDense,
    WriteOptions,
    open_store,
    write,
    write_csr,
    write_dense,
)
from scdata.exceptions import Error
from scdata.load import (
    Dataset,
    IoMergeConfig,
    Plan,
    PlanConfig,
    Session,
    SessionConfig,
    compile,
    prefetch,
    register,
)

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_READ_LIMITS",
    "Codec",
    "DEFAULT_WRITE_OPTIONS",
    "Dataset",
    "Error",
    "IoMergeConfig",
    "Plan",
    "PlanConfig",
    "ReadLimits",
    "ScCsr",
    "ScDense",
    "Session",
    "SessionConfig",
    "WriteOptions",
    "__version__",
    "anndata",
    "compile",
    "compress",
    "load",
    "open_store",
    "prefetch",
    "read_scc",
    "register",
    "tools",
    "write",
    "write_csr",
    "write_dense",
    "write_scc",
]
