"""Typed snapshots returned by compiled plans and live sessions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Literal

IoMode = Literal["auto", "blocking", "uring"]
SessionState = Literal["running", "failed", "cancelled", "finished"]

__all__ = ["PlanStats", "RuntimeStats", "SessionState"]


@dataclass(frozen=True)
class PlanStats:
    input_rows: int
    block_jobs: int
    jobs: int
    data_io_ops: int
    indices_io_ops: int
    predicted_physical_bytes: int
    gap_bytes: int
    max_encoded_bytes_per_side: int
    max_decoded_bytes_per_job: int
    arena_bytes: int
    compile_working_set_bytes: int
    retained_whole_key_bytes: int
    output_ring_bytes: int
    compile_time_io_bytes: int
    compile_time_io_ops: int
    predicted_io_seconds: float
    cache_capacity_bytes: int
    cache_arena_bytes: int
    cache_alignment_loss_bytes: int
    unique_cache_objects: int
    residency_loads: int
    residency_reloads: int
    cache_reference_hits: int
    cache_reference_misses: int
    cache_capacity_stalls: int
    cache_fragmentation_stalls: int
    cache_horizon_max_batches: int
    initialize_io_tasks: int
    executable_tasks: int
    dependency_edges: int
    independent_block_loads: int
    fused_io_tasks: int
    predicted_io_ops_saved: int
    io_payload_bytes: int
    io_span_bytes: int
    io_read_amplification: float
    max_decode_ops_per_io_task: int
    max_decoded_bytes_per_io_task: int
    initialize_fused_io_tasks: int
    regular_fused_io_tasks: int
    profile: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile", _freeze_mapping(self.profile))

    @classmethod
    def _from_mapping(cls, values: Mapping[str, Any]) -> PlanStats:
        known = {
            "input_rows",
            "block_jobs",
            "jobs",
            "data_io_ops",
            "indices_io_ops",
            "predicted_physical_bytes",
            "gap_bytes",
            "max_encoded_bytes_per_side",
            "max_decoded_bytes_per_job",
            "arena_bytes",
            "compile_working_set_bytes",
            "retained_whole_key_bytes",
            "output_ring_bytes",
            "compile_time_io_bytes",
            "compile_time_io_ops",
            "predicted_io_seconds",
            "cache_capacity_bytes",
            "cache_arena_bytes",
            "cache_alignment_loss_bytes",
            "unique_cache_objects",
            "residency_loads",
            "residency_reloads",
            "cache_reference_hits",
            "cache_reference_misses",
            "cache_capacity_stalls",
            "cache_fragmentation_stalls",
            "cache_horizon_max_batches",
            "initialize_io_tasks",
            "executable_tasks",
            "dependency_edges",
            "independent_block_loads",
            "fused_io_tasks",
            "predicted_io_ops_saved",
            "io_payload_bytes",
            "io_span_bytes",
            "io_read_amplification",
            "max_decode_ops_per_io_task",
            "max_decoded_bytes_per_io_task",
            "initialize_fused_io_tasks",
            "regular_fused_io_tasks",
        }
        profile = MappingProxyType(
            {key: value for key, value in values.items() if key not in known}
        )
        return cls(**{key: values[key] for key in known}, profile=profile)

    def as_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for logging or serialization."""
        return {
            item.name: _to_plain(self.profile)
            if item.name == "profile"
            else getattr(self, item.name)
            for item in fields(self)
        }


@dataclass(frozen=True)
class RuntimeStats:
    requested_io_mode: IoMode
    requested_queue_depth: int
    actual_io_mode: IoMode
    actual_queue_depth: int
    num_workers: int
    max_inflight_jobs_per_worker: int
    max_inflight_encoded_bytes_per_worker: int
    max_decoded_bytes_per_worker: int
    state: SessionState
    profile: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile", _freeze_mapping(self.profile))

    @classmethod
    def _from_mapping(cls, values: Mapping[str, Any]) -> RuntimeStats:
        known = {
            "requested_io_mode",
            "requested_queue_depth",
            "actual_io_mode",
            "actual_queue_depth",
            "num_workers",
            "max_inflight_jobs_per_worker",
            "max_inflight_encoded_bytes_per_worker",
            "max_decoded_bytes_per_worker",
            "state",
        }
        profile = MappingProxyType(
            {key: value for key, value in values.items() if key not in known}
        )
        return cls(**{key: values[key] for key in known}, profile=profile)

    def as_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for logging or serialization."""
        return {
            item.name: _to_plain(self.profile)
            if item.name == "profile"
            else getattr(self, item.name)
            for item in fields(self)
        }


def _freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze(value) for key, value in values.items()})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _to_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    return value
