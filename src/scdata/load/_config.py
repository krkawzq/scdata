"""Immutable Python configuration for planning, I/O, and memory bounds."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Literal

from scdata._validate import as_float, as_int

IoMode = Literal["auto", "blocking", "uring"]

__all__ = ["IoMode", "PlanConfig", "ResourceLimits", "SessionConfig"]

_MIB = 1024 * 1024
_GIB = 1024 * 1024 * 1024
_U32_MAX = (1 << 32) - 1
DEFAULT_MAX_CONTROL_BYTES = 64 * _MIB


def _cpu_count() -> int:
    return os.cpu_count() or 1


def _default_compile_io_concurrency() -> int:
    return min(_cpu_count(), 32)


@dataclass(frozen=True, slots=True)
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
        for name in self.__slots__:
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


@dataclass(frozen=True, slots=True)
class PlanConfig:
    """Static compiler cost model and resource limits."""

    compile_io_concurrency: int = field(default_factory=_default_compile_io_concurrency)
    io_bandwidth_bytes_per_second: float = float(8 * _GIB)
    io_operations_per_second: float = 100_000.0
    coalescing_distance: int = 1
    max_coalesced_io_bytes: int = 32 * _MIB
    target_decoded_bytes_per_job: int = 64 * _MIB
    delta_bytes: float = 4096.0
    limits: ResourceLimits = field(default_factory=ResourceLimits)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "compile_io_concurrency",
            as_int(self.compile_io_concurrency, "compile_io_concurrency", minimum=1),
        )
        object.__setattr__(
            self,
            "io_bandwidth_bytes_per_second",
            as_float(
                self.io_bandwidth_bytes_per_second,
                "io_bandwidth_bytes_per_second",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "io_operations_per_second",
            as_float(
                self.io_operations_per_second,
                "io_operations_per_second",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "coalescing_distance",
            as_int(self.coalescing_distance, "coalescing_distance", minimum=1),
        )
        object.__setattr__(
            self,
            "max_coalesced_io_bytes",
            as_int(self.max_coalesced_io_bytes, "max_coalesced_io_bytes", minimum=1),
        )
        object.__setattr__(
            self,
            "target_decoded_bytes_per_job",
            as_int(
                self.target_decoded_bytes_per_job,
                "target_decoded_bytes_per_job",
                minimum=1,
            ),
        )
        delta = as_float(self.delta_bytes, "delta_bytes")
        if delta < 0.0:
            raise ValueError("delta_bytes must be non-negative")
        object.__setattr__(self, "delta_bytes", delta)
        if not isinstance(self.limits, ResourceLimits):
            raise TypeError("limits must be a ResourceLimits instance")
        if self.target_decoded_bytes_per_job > self.limits.max_decoded_bytes_per_job:
            raise ValueError(
                "target_decoded_bytes_per_job must not exceed max_decoded_bytes_per_job"
            )

    def _validate_for(self, prefetch_step: int) -> None:
        if self.coalescing_distance >= prefetch_step:
            raise ValueError(
                "coalescing_distance must be smaller than prefetch_step "
                f"(got {self.coalescing_distance} and {prefetch_step})"
            )

    def _to_core(self) -> dict[str, int | float]:
        limits = self.limits
        return {
            "compile_io_concurrency": self.compile_io_concurrency,
            "io_bandwidth_bytes_per_second": self.io_bandwidth_bytes_per_second,
            "io_operations_per_second": self.io_operations_per_second,
            "coalescing_distance": self.coalescing_distance,
            "max_coalesced_io_bytes": self.max_coalesced_io_bytes,
            "target_decoded_bytes_per_job": self.target_decoded_bytes_per_job,
            "delta_bytes": self.delta_bytes,
            "max_output_buffer_bytes": limits.max_output_buffer_bytes,
            "max_compile_arena_bytes": limits.max_compile_arena_bytes,
            "max_compile_working_set_bytes": limits.max_compile_working_set_bytes,
            "max_retained_whole_key_bytes": limits.max_retained_whole_key_bytes,
            "max_blocks_per_job": limits.max_blocks_per_job,
            "max_cells_per_job": limits.max_cells_per_job,
            "max_encoded_bytes_per_side": limits.max_encoded_bytes_per_side,
            "max_decoded_bytes_per_job": limits.max_decoded_bytes_per_job,
        }


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Worker backend and aggregate resident-buffer limits for one session."""

    num_workers: int = field(default_factory=_cpu_count)
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

    def _to_core(self) -> dict[str, int | str]:
        io_ops_per_worker = self.queue_depth if self.io_mode != "blocking" else 1
        required_io_ops = _checked_product(self.num_workers, io_ops_per_worker, "in-flight I/O ops")
        required_encoded = _checked_product(
            self.num_workers,
            self.max_inflight_encoded_bytes_per_worker,
            "in-flight encoded bytes",
        )
        required_decoded = _checked_product(
            self.num_workers,
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
