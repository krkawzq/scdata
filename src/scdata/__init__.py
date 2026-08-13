"""SCC-backed matrices and high-throughput single-cell data loading."""

from __future__ import annotations

from typing import Any

from scdata import _core
from scdata import compress, load, tools
from scdata.compress import (
    DEFAULT_READ_LIMITS,
    DEFAULT_WRITE_OPTIONS,
    ReadLimits,
    ScCsr,
    ScDense,
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

__version__ = _core.__version__


def __getattr__(name: str) -> Any:
    if name in {"anndata", "read_scc", "write_scc"}:
        from importlib import import_module

        anndata_mod = import_module("scdata.anndata")
        if name == "anndata":
            return anndata_mod
        return getattr(anndata_mod, name)
    raise AttributeError(f"module 'scdata' has no attribute {name!r}")


__all__ = [
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
    "WriteOptions",
    "__version__",
    "anndata",
    "compile",
    "compress",
    "load",
    "open",
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

open = open_store
