"""Immutable Python configuration for planning, I/O, and memory bounds."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, fields
from typing import Literal

from scdata import _core
from scdata._validate import as_float, as_int

IoMode = Literal["auto", "blocking", "uring"]
IoMergePolicy = Literal["off", "adjacent", "cost"]

__all__ = [
    "IoMergeConfig",
    "IoMergePolicy",
    "IoMode",
    "PlanConfig",
    "ResourceLimits",
    "SessionConfig",
]

_MIB = 1024 * 1024
_GIB = 1024 * 1024 * 1024
_U32_MAX = (1 << 32) - 1
DEFAULT_MAX_CONTROL_BYTES = 64 * _MIB


def _cpu_count() -> int:
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            count = len(affinity(0))
        except OSError:
            pass
        else:
            if count:
                return count
    return os.cpu_count() or 1


def _default_compile_io_concurrency() -> int:
    return min(_cpu_count(), 32)


@dataclass(frozen=True)
class ResourceLimits:
    """Hard limits for compiler arenas, output rings, and individual jobs."""

    max_output_buffer_bytes: int = 2 * _GIB
    max_compile_arena_bytes: int = 2 * _GIB
    max_compile_working_set_bytes: int = 40 * _GIB
    max_retained_whole_key_bytes: int = 512 * _MIB
    max_blocks_per_job: int = 4096
    max_cells_per_job: int = 1_000_000
    max_encoded_bytes_per_side: int = 1 * _GIB
    max_decoded_bytes_per_job: int = 2 * _GIB

    def __post_init__(self) -> None:
        for item in fields(self):
            name = item.name
            minimum = 0 if name == "max_retained_whole_key_bytes" else 1
            value = as_int(getattr(self, name), name, minimum=minimum)
            object.__setattr__(self, name, value)
        if self.max_compile_arena_bytes > self.max_compile_working_set_bytes:
            raise ValueError(
                "max_compile_arena_bytes must not exceed max_compile_working_set_bytes"
            )
        if self.max_retained_whole_key_bytes > self.max_compile_working_set_bytes:
            raise ValueError(
                "max_retained_whole_key_bytes must not exceed max_compile_working_set_bytes"
            )
        if self.max_encoded_bytes_per_side > _U32_MAX:
            raise ValueError("max_encoded_bytes_per_side must not exceed uint32 capacity")
        if self.max_decoded_bytes_per_job > _U32_MAX:
            raise ValueError("max_decoded_bytes_per_job must not exceed uint32 capacity")


@dataclass(frozen=True)
class IoMergeConfig:
    """Final cache-load fusion policy and hard per-I/O-task limits."""

    policy: IoMergePolicy = "adjacent"
    max_coalesced_io_bytes: int = 32 * _MIB
    max_io_gap_bytes: int = 0
    max_io_amplification_ratio: float = 1.0
    max_decode_ops_per_io_task: int = 64
    max_decoded_bytes_per_io_task: int = 64 * _MIB
    max_encoded_staging_bytes_per_task: int = 32 * _MIB
    io_bandwidth_bytes_per_second: float = float(8 * _GIB)
    io_operations_per_second: float = 100_000.0
    io_merge_delta_bytes: int = 4096
    initialize_parallelism_hint: int = 32
    regular_io_parallelism_hint: int = 32
    min_tasks_per_worker: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.policy, str):
            raise TypeError("policy must be a string")
        policy = self.policy.strip().lower()
        if policy not in {"off", "adjacent", "cost"}:
            raise ValueError("policy must be 'off', 'adjacent', or 'cost'")
        object.__setattr__(self, "policy", policy)
        for name in (
            "max_coalesced_io_bytes",
            "max_decode_ops_per_io_task",
            "max_decoded_bytes_per_io_task",
            "max_encoded_staging_bytes_per_task",
            "initialize_parallelism_hint",
            "regular_io_parallelism_hint",
            "min_tasks_per_worker",
        ):
            object.__setattr__(self, name, as_int(getattr(self, name), name, minimum=1))
        for name in ("max_io_gap_bytes", "io_merge_delta_bytes"):
            object.__setattr__(self, name, as_int(getattr(self, name), name))
        for name in (
            "max_io_amplification_ratio",
            "io_bandwidth_bytes_per_second",
            "io_operations_per_second",
        ):
            object.__setattr__(
                self,
                name,
                as_float(getattr(self, name), name, positive=True),
            )
        if self.max_coalesced_io_bytes > self.max_encoded_staging_bytes_per_task:
            raise ValueError(
                "max_coalesced_io_bytes must not exceed "
                "max_encoded_staging_bytes_per_task"
            )

    def _to_core(self) -> _core.IoMergeConfigDict:
        return {
            "policy": self.policy,
            "max_coalesced_io_bytes": self.max_coalesced_io_bytes,
            "max_io_gap_bytes": 0 if self.policy == "adjacent" else self.max_io_gap_bytes,
            "max_io_amplification_ratio": (
                1.0 if self.policy == "adjacent" else self.max_io_amplification_ratio
            ),
            "max_decode_ops_per_io_task": self.max_decode_ops_per_io_task,
            "max_decoded_bytes_per_io_task": self.max_decoded_bytes_per_io_task,
            "max_encoded_staging_bytes_per_task": self.max_encoded_staging_bytes_per_task,
            "io_bandwidth_bytes_per_second": self.io_bandwidth_bytes_per_second,
            "io_operations_per_second": self.io_operations_per_second,
            "io_merge_delta_bytes": self.io_merge_delta_bytes,
            "initialize_parallelism_hint": self.initialize_parallelism_hint,
            "regular_io_parallelism_hint": self.regular_io_parallelism_hint,
            "min_tasks_per_worker": self.min_tasks_per_worker,
        }


@dataclass(frozen=True)
class PlanConfig:
    """Static cache compiler resources and I/O fusion policy."""

    compile_io_concurrency: int = field(default_factory=_default_compile_io_concurrency)
    io_merge: IoMergeConfig = field(default_factory=IoMergeConfig)
    cache_capacity_bytes: int = 64 * _MIB
    cache_alignment: int = 64
    cache_fragmentation_slack_bytes: int = 64 * 1024
    limits: ResourceLimits = field(default_factory=ResourceLimits)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "compile_io_concurrency",
            as_int(self.compile_io_concurrency, "compile_io_concurrency", minimum=1),
        )
        if not isinstance(self.io_merge, IoMergeConfig):
            raise TypeError("io_merge must be an IoMergeConfig instance")
        object.__setattr__(
            self,
            "cache_capacity_bytes",
            as_int(self.cache_capacity_bytes, "cache_capacity_bytes", minimum=1),
        )
        alignment = as_int(self.cache_alignment, "cache_alignment", minimum=64)
        if alignment & (alignment - 1):
            raise ValueError("cache_alignment must be a power of two")
        object.__setattr__(self, "cache_alignment", alignment)
        object.__setattr__(
            self,
            "cache_fragmentation_slack_bytes",
            as_int(
                self.cache_fragmentation_slack_bytes,
                "cache_fragmentation_slack_bytes",
            ),
        )
        if not isinstance(self.limits, ResourceLimits):
            raise TypeError("limits must be a ResourceLimits instance")

    def _to_core(self) -> _core.PlanConfigDict:
        limits = self.limits
        return {
            "compile_io_concurrency": self.compile_io_concurrency,
            "io_merge": self.io_merge._to_core(),
            "cache_capacity_bytes": self.cache_capacity_bytes,
            "cache_alignment": self.cache_alignment,
            "cache_fragmentation_slack_bytes": self.cache_fragmentation_slack_bytes,
            "max_output_buffer_bytes": limits.max_output_buffer_bytes,
            "max_compile_arena_bytes": limits.max_compile_arena_bytes,
            "max_compile_working_set_bytes": limits.max_compile_working_set_bytes,
            "max_retained_whole_key_bytes": limits.max_retained_whole_key_bytes,
            "max_blocks_per_job": limits.max_blocks_per_job,
            "max_cells_per_job": limits.max_cells_per_job,
            "max_encoded_bytes_per_side": limits.max_encoded_bytes_per_side,
            "max_decoded_bytes_per_job": limits.max_decoded_bytes_per_job,
        }


@dataclass(frozen=True)
class SessionConfig:
    """Worker backend and aggregate resident-buffer limits for one session."""

    num_workers: int = field(default_factory=_cpu_count)
    initialize_workers: int | None = None
    initialize_inflight_io_ops: int | None = None
    initialize_inflight_encoded_bytes: int = 512 * _MIB
    io_mode: IoMode = "auto"
    queue_depth: int = 64
    max_inflight_jobs_per_worker: int = 32
    max_inflight_encoded_bytes_per_worker: int = 512 * _MIB
    max_decoded_bytes_per_worker: int = 2 * _GIB
    max_total_inflight_io_ops: int | None = None
    max_total_inflight_encoded_bytes: int | None = None
    max_total_decoded_bytes: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "num_workers",
            as_int(self.num_workers, "num_workers", minimum=1),
        )
        initialize_workers = self.num_workers if self.initialize_workers is None else as_int(
            self.initialize_workers, "initialize_workers", minimum=1
        )
        initialize_ops = initialize_workers if self.initialize_inflight_io_ops is None else as_int(
            self.initialize_inflight_io_ops,
            "initialize_inflight_io_ops",
            minimum=1,
        )
        object.__setattr__(self, "initialize_workers", initialize_workers)
        object.__setattr__(self, "initialize_inflight_io_ops", initialize_ops)
        object.__setattr__(
            self,
            "initialize_inflight_encoded_bytes",
            as_int(
                self.initialize_inflight_encoded_bytes,
                "initialize_inflight_encoded_bytes",
                minimum=1,
            ),
        )
        if not isinstance(self.io_mode, str):
            raise TypeError("io_mode must be a string")
        mode = self.io_mode.strip().lower()
        if mode not in {"auto", "blocking", "uring"}:
            raise ValueError("io_mode must be 'auto', 'blocking', or 'uring'")
        object.__setattr__(self, "io_mode", mode)
        queue_depth = as_int(self.queue_depth, "queue_depth", maximum=_U32_MAX)
        if mode != "blocking" and queue_depth < 2:
            raise ValueError("queue_depth must be at least 2 for auto or uring mode")
        object.__setattr__(self, "queue_depth", queue_depth)
        for name in (
            "max_inflight_jobs_per_worker",
            "max_inflight_encoded_bytes_per_worker",
            "max_decoded_bytes_per_worker",
        ):
            object.__setattr__(self, name, as_int(getattr(self, name), name, minimum=1))
        for name in (
            "max_total_inflight_io_ops",
            "max_total_inflight_encoded_bytes",
            "max_total_decoded_bytes",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, as_int(value, name, minimum=1))

    def _to_core(self) -> _core.SessionConfigDict:
        initialize_workers = self.initialize_workers
        initialize_inflight_io_ops = self.initialize_inflight_io_ops
        if initialize_workers is None or initialize_inflight_io_ops is None:
            raise RuntimeError("SessionConfig normalization did not initialize worker limits")
        io_ops_per_worker = self.queue_depth if self.io_mode != "blocking" else 1
        regular_io_ops = _checked_product(
            self.num_workers, io_ops_per_worker, "in-flight I/O ops"
        )
        required_io_ops = max(regular_io_ops, initialize_inflight_io_ops)
        regular_encoded = _checked_product(
            self.num_workers,
            self.max_inflight_encoded_bytes_per_worker,
            "in-flight encoded bytes",
        )
        required_encoded = max(regular_encoded, self.initialize_inflight_encoded_bytes)
        required_decoded = _checked_product(
            max(self.num_workers, initialize_workers),
            self.max_decoded_bytes_per_worker,
            "decoded bytes",
        )
        total_io_ops = _resolve_total(
            self.max_total_inflight_io_ops,
            required_io_ops,
            "max_total_inflight_io_ops",
        )
        total_encoded = _resolve_total(
            self.max_total_inflight_encoded_bytes,
            required_encoded,
            "max_total_inflight_encoded_bytes",
        )
        total_decoded = _resolve_total(
            self.max_total_decoded_bytes,
            required_decoded,
            "max_total_decoded_bytes",
        )
        return {
            "num_workers": self.num_workers,
            "initialize_workers": initialize_workers,
            "initialize_inflight_io_ops": initialize_inflight_io_ops,
            "initialize_inflight_encoded_bytes": self.initialize_inflight_encoded_bytes,
            "io_mode": self.io_mode,
            "queue_depth": self.queue_depth,
            "max_inflight_jobs_per_worker": self.max_inflight_jobs_per_worker,
            "max_inflight_encoded_bytes_per_worker": (self.max_inflight_encoded_bytes_per_worker),
            "max_decoded_bytes_per_worker": self.max_decoded_bytes_per_worker,
            "max_total_inflight_io_ops": total_io_ops,
            "max_total_inflight_encoded_bytes": total_encoded,
            "max_total_decoded_bytes": total_decoded,
        }


def _checked_product(left: int, right: int, name: str) -> int:
    value = left * right
    if value > sys.maxsize:
        raise ValueError(f"{name} exceeds the platform size limit")
    return value


def _resolve_total(value: int | None, required: int, name: str) -> int:
    if value is None:
        return required
    if value < required:
        raise ValueError(f"{name}={value} is smaller than the required {required}")
    return value
