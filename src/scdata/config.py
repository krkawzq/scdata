"""Immutable Python configuration for planning, I/O, and memory bounds."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Literal, cast

from scdata import _core
from scdata._validation import as_float, as_int

IoMode = Literal["auto", "blocking", "uring"]

__all__ = ["IoMode", "PlanConfig", "ReadLimits", "ResourceLimits", "SessionConfig"]

_PLAN_DEFAULTS = _core._plan_config_defaults()
_SESSION_DEFAULTS = _core._session_config_defaults()
_U32_MAX = (1 << 32) - 1


@dataclass(frozen=True, slots=True)
class ReadLimits:
    """Bounds applied while opening sc-compress metadata and indexes."""

    max_metadata_size: int = _core.DEFAULT_MAXIMUM_METADATA_SIZE
    max_encoded_size: int = _core.DEFAULT_MAXIMUM_ENCODED_SIZE
    max_decoded_size: int = _core.DEFAULT_MAXIMUM_DECODED_SIZE
    max_block_count: int = _core.DEFAULT_MAXIMUM_BLOCK_COUNT

    def __post_init__(self) -> None:
        for name in self.__slots__:
            object.__setattr__(self, name, as_int(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Hard limits for compiler arenas, output rings, and individual jobs."""

    max_output_buffer_bytes: int = int(_PLAN_DEFAULTS["max_output_buffer_bytes"])
    max_compile_arena_bytes: int = int(_PLAN_DEFAULTS["max_compile_arena_bytes"])
    max_compile_working_set_bytes: int = int(_PLAN_DEFAULTS["max_compile_working_set_bytes"])
    max_retained_whole_key_bytes: int = int(_PLAN_DEFAULTS["max_retained_whole_key_bytes"])
    max_blocks_per_job: int = int(_PLAN_DEFAULTS["max_blocks_per_job"])
    max_cells_per_job: int = int(_PLAN_DEFAULTS["max_cells_per_job"])
    max_encoded_bytes_per_side: int = int(_PLAN_DEFAULTS["max_encoded_bytes_per_side"])
    max_decoded_bytes_per_job: int = int(_PLAN_DEFAULTS["max_decoded_bytes_per_job"])

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

    compile_io_concurrency: int = int(_PLAN_DEFAULTS["compile_io_concurrency"])
    io_bandwidth_bytes_per_second: float = float(_PLAN_DEFAULTS["io_bandwidth_bytes_per_second"])
    io_operations_per_second: float = float(_PLAN_DEFAULTS["io_operations_per_second"])
    coalescing_distance: int = int(_PLAN_DEFAULTS["coalescing_distance"])
    max_coalesced_io_bytes: int = int(_PLAN_DEFAULTS["max_coalesced_io_bytes"])
    target_decoded_bytes_per_job: int = int(_PLAN_DEFAULTS["target_decoded_bytes_per_job"])
    delta_bytes: float = float(_PLAN_DEFAULTS["delta_bytes"])
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

    worker_count: int | None = None
    io_mode: IoMode = cast(IoMode, _SESSION_DEFAULTS["io_mode"])
    queue_depth: int = int(_SESSION_DEFAULTS["queue_depth"])
    max_inflight_jobs_per_worker: int = int(_SESSION_DEFAULTS["max_inflight_jobs_per_worker"])
    max_inflight_encoded_bytes_per_worker: int = int(
        _SESSION_DEFAULTS["max_inflight_encoded_bytes_per_worker"]
    )
    max_decoded_bytes_per_worker: int = int(_SESSION_DEFAULTS["max_decoded_bytes_per_worker"])
    max_total_inflight_io_ops: int | None = None
    max_total_inflight_encoded_bytes: int | None = None
    max_total_decoded_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.worker_count is not None:
            object.__setattr__(
                self,
                "worker_count",
                as_int(self.worker_count, "worker_count", minimum=1),
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
        workers = (
            int(_SESSION_DEFAULTS["worker_count"])
            if self.worker_count is None
            else self.worker_count
        )
        io_ops_per_worker = self.queue_depth if self.io_mode != "blocking" else 1
        required_io_ops = _checked_product(workers, io_ops_per_worker, "in-flight I/O ops")
        required_encoded = _checked_product(
            workers,
            self.max_inflight_encoded_bytes_per_worker,
            "in-flight encoded bytes",
        )
        required_decoded = _checked_product(
            workers,
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
            "worker_count": workers,
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
