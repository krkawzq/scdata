"""Compile row requests into reusable prefetch plans."""

from scdata import _core
from scdata.compress.limits import ReadLimits
from scdata.exceptions import (
    AllocationError,
    CancelledError,
    ConversionError,
    DecodeError,
    Error,
    InternalError,
    InvalidConfigError,
    InvalidDatasetError,
    InvalidInputError,
    IoError,
    PromotionError,
    ResourceLimitError,
    SessionError,
    StalePlanError,
    UnsupportedError,
    WorkerPanicError,
)
from scdata.load.config import IoMode, PlanConfig, ResourceLimits, SessionConfig
from scdata.load.dataset import Dataset, DatasetKind, RowRef, register
from scdata.load.distributed import DistributedIterator, DistributedSession, distributed_prefetch
from scdata.load.output import OutputSpec, OverflowPolicy
from scdata.load.plan import Plan, Prefetch, Session, compile, prefetch
from scdata.load.stats import PlanStats, RuntimeStats, SessionState

__all__ = [
    "AllocationError",
    "CancelledError",
    "ConversionError",
    "Dataset",
    "DatasetKind",
    "DecodeError",
    "DistributedIterator",
    "DistributedSession",
    "Error",
    "InternalError",
    "InvalidConfigError",
    "InvalidDatasetError",
    "InvalidInputError",
    "IoError",
    "IoMode",
    "OutputSpec",
    "OverflowPolicy",
    "Plan",
    "PlanConfig",
    "PlanStats",
    "Prefetch",
    "PromotionError",
    "ReadLimits",
    "ResourceLimits",
    "ResourceLimitError",
    "RowRef",
    "RuntimeStats",
    "Session",
    "SessionError",
    "StalePlanError",
    "SessionConfig",
    "SessionState",
    "UnsupportedError",
    "WorkerPanicError",
    "_core",
    "compile",
    "distributed_prefetch",
    "prefetch",
    "register",
]
