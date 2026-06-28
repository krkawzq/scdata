"""Python interface for scdata."""

from __future__ import annotations

_MISSING_EXTENSION_ERROR: ModuleNotFoundError | None = None

try:
    from ._scdata import __version__, kernel_name, kernel_version
except ModuleNotFoundError as exc:
    if exc.name != "scdata._scdata":
        raise

    _MISSING_EXTENSION_ERROR = exc
    __version__ = "0.1.0"

    def kernel_name() -> str:
        raise RuntimeError(
            "scdata Rust extension is not built. Install the package with "
            "`maturin develop` or `uv pip install -e .` before using Rust-backed APIs."
        ) from _MISSING_EXTENSION_ERROR

    def kernel_version() -> str:
        return __version__


# Pure-Python modules: store metadata parsing, dataset types, and the cell
# access / batch carriers.  These do not depend on the Rust extension and are
# always importable, so the store format can be read (and the data types used)
# even before the Rust core is bound.
from scdata.data import (  # noqa: E402
    ArrayMeta,
    ArrayOrder,
    CellAccess,
    CellBatch,
    CellData,
    ChunkLocation,
    CodecPipeline,
    DataError,
    DenseDataset,
    DType,
    Dataset,
    DtypeParseError,
    CodecConfigError,
    ScDataLoader,
    SparseDataset,
)
from scdata.io import (  # noqa: E402
    AnnDataZarrZipConverter,
    Store,
    StoreError,
    launch,
    launch_store,
    read_zarr,
    write_zarr,
)

# Pythonic DataBank wrapper + config dataclasses (re-exported verbatim from the
# Rust extension).  When the extension is missing, the Rust-backed names raise
# on use; the pure-Python ``CellAccess`` / ``CellBatch`` / ``CellData`` above
# remain usable regardless.
try:
    from scdata.databank import (
        AccessConfig,
        AccessCpuConfig,
        BaseIoConfig,
        DataBankConfig,
        DataBankError,
        DatasetId,
        DecodePoolConfig,
        FillConfig,
        IoConfig,
        MissingGenePolicy,
        ScheduledAccessConfig,
        ScheduledPrefetchConfig,
        ScDataBank,
        ThreadedConfig,
        UringConfig,
    )
except ModuleNotFoundError:

    def _missing(name: str):
        def _raise(*args: object, **kwargs: object) -> None:
            raise RuntimeError(
                f"{name} requires the scdata Rust extension. Install the package "
                "with `maturin develop` or `uv pip install -e .`."
            ) from _MISSING_EXTENSION_ERROR

        return _raise

    ScDataBank = _missing("ScDataBank")  # type: ignore[assignment, misc]
    DataBankConfig = _missing("DataBankConfig")  # type: ignore[assignment, misc]
    DatasetId = _missing("DatasetId")  # type: ignore[assignment, misc]
    MissingGenePolicy = _missing("MissingGenePolicy")  # type: ignore[assignment, misc]
    DataBankError = RuntimeError  # type: ignore[assignment, misc]
    IoConfig = _missing("IoConfig")  # type: ignore[assignment, misc]
    UringConfig = _missing("UringConfig")  # type: ignore[assignment, misc]
    ThreadedConfig = _missing("ThreadedConfig")  # type: ignore[assignment, misc]
    BaseIoConfig = _missing("BaseIoConfig")  # type: ignore[assignment, misc]
    DecodePoolConfig = _missing("DecodePoolConfig")  # type: ignore[assignment, misc]
    AccessConfig = _missing("AccessConfig")  # type: ignore[assignment, misc]
    AccessCpuConfig = _missing("AccessCpuConfig")  # type: ignore[assignment, misc]
    FillConfig = _missing("FillConfig")  # type: ignore[assignment, misc]
    ScheduledAccessConfig = _missing("ScheduledAccessConfig")  # type: ignore[assignment, misc]
    ScheduledPrefetchConfig = _missing("ScheduledPrefetchConfig")  # type: ignore[assignment, misc]

__all__ = [
    "__version__",
    "kernel_name",
    "kernel_version",
    # databank (Rust-backed, Pythonic wrapper)
    "ScDataBank",
    "DataBankError",
    "DatasetId",
    "MissingGenePolicy",
    "DataBankConfig",
    "IoConfig",
    "UringConfig",
    "ThreadedConfig",
    "BaseIoConfig",
    "DecodePoolConfig",
    "AccessConfig",
    "AccessCpuConfig",
    "FillConfig",
    "ScheduledAccessConfig",
    "ScheduledPrefetchConfig",
    # data (pure Python — usable with or without the Rust extension)
    "CellAccess",
    "CellBatch",
    "CellData",
    "ScDataLoader",
    "ArrayMeta",
    "ArrayOrder",
    "ChunkLocation",
    "CodecPipeline",
    "DataError",
    "DtypeParseError",
    "CodecConfigError",
    "DenseDataset",
    "DType",
    "Dataset",
    "SparseDataset",
    # io
    "AnnDataZarrZipConverter",
    "Store",
    "StoreError",
    "launch",
    "launch_store",
    "read_zarr",
    "write_zarr",
]
