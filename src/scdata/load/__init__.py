"""Compile row requests into reusable prefetch plans."""

from scdata.compress import ReadLimits
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
from scdata.load._config import IoMode, PlanConfig, ResourceLimits, SessionConfig
from scdata.load._dataset import Dataset, DatasetKind, RowRef, register
from scdata.load._distributed import DistributedIterator, DistributedSession, distributed_prefetch
from scdata.load._location import list_keys, read_feature_names, read_obs_names
from scdata.load._names import as_str_tuple, build_feature_map, locate_names
from scdata.load._output import OutputSpec, OverflowPolicy
from scdata.load._plan import Plan, Prefetch, Session, compile, prefetch
from scdata.load._stats import PlanStats, RuntimeStats, SessionState

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
    "as_str_tuple",
    "build_feature_map",
    "compile",
    "distributed_prefetch",
    "list_keys",
    "locate_names",
    "prefetch",
    "read_feature_names",
    "read_obs_names",
    "register",
]
