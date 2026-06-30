"""scdata: a Rust-backed store and torch data pipeline for single-cell data."""

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
    BankConfigSummary,
    CellAccess,
    CellBatch,
    CellData,
    CellIndexDataset,
    ChunkLocation,
    CodecPipeline,
    DataError,
    DenseDataset,
    DType,
    Dataset,
    DatasetCollection,
    DtypeParseError,
    CodecConfigError,
    LoaderStats,
    ScDataLoader,
    SparseDataset,
    stitch_dense_collate,
)
from scdata.io import (  # noqa: E402
    AnnDataZarrZipConverter,
    Store,
    StoreError,
    launch,
    launch_all,
    launch_store,
    launch_store_all,
    read_zarr,
    write_zarr,
)

# Pythonic DataBank wrapper + config dataclasses (re-exported verbatim from the
# Rust extension).  When the extension is missing, the Rust-backed names raise
# on use; the pure-Python ``CellAccess`` / ``CellBatch`` / ``CellData`` above
# remain usable regardless.
def _missing(name: str):
    def _raise(*args: object, **kwargs: object) -> None:
        raise RuntimeError(
            f"{name} requires the scdata Rust extension. Install the package "
            "with `maturin develop` or `uv pip install -e .`."
        ) from _MISSING_EXTENSION_ERROR

    return _raise


def _missing_tools(name: str):
    def _raise(*args: object, **kwargs: object) -> None:
        raise RuntimeError(f"{name} requires the scdata.tools module to be installed.")

    return _raise


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
except ModuleNotFoundError as exc:
    if exc.name != "scdata._scdata":
        raise
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

try:
    from scdata.corpus import Corpus
except ModuleNotFoundError as exc:
    if exc.name != "scdata._scdata":
        raise
    Corpus = _missing("Corpus")  # type: ignore[assignment, misc]

try:
    from scdata.tools import TuneConfigResult, TuneResult, tune
except ModuleNotFoundError as exc:
    if exc.name != "scdata.tools":
        raise
    tune = _missing_tools("tune")  # type: ignore[assignment, misc]
    TuneResult = _missing_tools("TuneResult")  # type: ignore[assignment, misc]
    TuneConfigResult = _missing_tools("TuneConfigResult")  # type: ignore[assignment, misc]

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
    "Corpus",
    # tools (Rust-backed workflows: tuning, ...)
    "tune",
    "TuneResult",
    "TuneConfigResult",
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
    "DatasetCollection",
    "SparseDataset",
    "CellIndexDataset",
    "stitch_dense_collate",
    "LoaderStats",
    "BankConfigSummary",
    # io
    "AnnDataZarrZipConverter",
    "Store",
    "StoreError",
    "launch",
    "launch_all",
    "launch_store",
    "launch_store_all",
    "read_zarr",
    "write_zarr",
]
