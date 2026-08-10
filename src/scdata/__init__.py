"""Static prefetch planning for sc-compress matrices inside ``.scc`` stores.

The public API is implemented in Python. ``scdata._core`` is a private PyO3
extension that owns storage handles, compiled plans, worker sessions, and the
copy from leased output-ring batches into NumPy-owned arrays.
"""

from scdata._core import __version__
from scdata.config import IoMode, PlanConfig, ReadLimits, ResourceLimits, SessionConfig
from scdata.dataset import DatasetKind, RowRef, ScDataset, register
from scdata.distributed import DistributedIterator, DistributedSession, distributed_prefetch
from scdata.exceptions import (
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
    ScdataError,
    SessionError,
    StalePlanError,
    UnsupportedError,
    WorkerPanicError,
)
from scdata.output import OutputSpec, OverflowPolicy
from scdata.plan import Plan, Prefetch, Session, compile, prefetch
from scdata.stats import PlanStats, RuntimeStats, SessionState

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
    "ScdataError",
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
