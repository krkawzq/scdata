"""scdata-toolkit: compress, AnnData, and prefetch loading for single-cell matrices."""

from scdata import _core
from scdata.anndata import read_scc, write_scc
from scdata.compress import (
    DEFAULT_N_WORKERS,
    DEFAULT_READ_LIMITS,
    DEFAULT_WRITE_OPTIONS,
    ReadLimits,
    Store,
    WriteOptions,
    write,
    write_csr,
    write_dense,
)
from scdata.compress import open as open_store
from scdata.exceptions import Error
from scdata.load import (
    Dataset,
    Plan,
    PlanConfig,
    Session,
    SessionConfig,
    compile,
    prefetch,
    register,
)
from scdata.matrix import ScCsr, ScDense

__version__ = _core.__version__

__all__ = [
    "DEFAULT_N_WORKERS",
    "DEFAULT_READ_LIMITS",
    "DEFAULT_WRITE_OPTIONS",
    "Dataset",
    "Error",
    "Plan",
    "PlanConfig",
    "ReadLimits",
    "ScCsr",
    "ScDense",
    "Session",
    "SessionConfig",
    "Store",
    "WriteOptions",
    "__version__",
    "compile",
    "open_store",
    "prefetch",
    "read_scc",
    "register",
    "write",
    "write_csr",
    "write_dense",
    "write_scc",
]
