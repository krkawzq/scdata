"""Static prefetch planning for sc-compress matrices inside ``.scc`` stores.

The public API is implemented in Python. ``sc_load._core`` is a private PyO3
extension that owns storage handles, compiled plans, worker sessions, and the
copy from leased output-ring batches into NumPy-owned arrays.
"""

from sc_load._core import __version__
from sc_load.config import IoMode, PlanConfig, ReadLimits, ResourceLimits, SessionConfig
from sc_load.dataset import DatasetKind, RowRef, ScDataset, register
from sc_load.distributed import DistributedIterator, DistributedSession, distributed_prefetch
from sc_load.exceptions import (
    AllocationError,
    CancelledError,
    ConversionError,
    DecodeError,
    InternalError,
    InvalidConfigError,
    InvalidDatasetError,
    InvalidInputError,
    IoError,
    PromotionError,
    ResourceLimitError,
    ScLoadError,
    SessionError,
    StalePlanError,
    UnsupportedError,
    WorkerPanicError,
)
from sc_load.output import OutputSpec, OverflowPolicy
from sc_load.plan import Plan, Prefetch, Session, compile, prefetch
from sc_load.stats import PlanStats, RuntimeStats, SessionState

__all__ = [
    "AllocationError",
    "CancelledError",
    "ConversionError",
    "DatasetKind",
    "DecodeError",
    "DistributedIterator",
    "DistributedSession",
    "InternalError",
    "InvalidConfigError",
    "InvalidDatasetError",
    "InvalidInputError",
    "IoMode",
    "IoError",
    "OutputSpec",
    "OverflowPolicy",
    "Plan",
    "PlanConfig",
    "PlanStats",
    "Prefetch",
    "PromotionError",
    "ReadLimits",
    "ResourceLimitError",
    "ResourceLimits",
    "RowRef",
    "RuntimeStats",
    "ScDataset",
    "ScLoadError",
    "Session",
    "SessionConfig",
    "SessionError",
    "SessionState",
    "StalePlanError",
    "UnsupportedError",
    "WorkerPanicError",
    "__version__",
    "compile",
    "distributed_prefetch",
    "prefetch",
    "register",
]
