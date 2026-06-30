"""Consumer-side health metrics for :class:`~scdata.data.ScDataLoader`.

These are **not** internal pipeline observability (cache hit rate, IO/decode
queue depth, sub-stage timings) — that lives behind the Rust core and is
exposed separately.  This module captures what the training loop actually
feels: how long ``next(loader)`` blocks on the bank (the *produce* time), how
long the collate step takes (the *consumer* time), and the resulting
end-to-end throughput.

The split that matters for tuning is
``consumer_fraction = 1 - produce / wall``:

* when it is high, the bank is **not** the bottleneck — deepening prefetch
  will not help, the consumer (model step / collate) is the lever;
* when it is low, the consumer is starved and prefetch / cache tuning is the
  right lever.

:func:`percentile` is shared here for callers that need the same wait-time
summary as :class:`LoaderStats`.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scdata.databank import DataBankConfig, ScheduledPrefetchConfig

__all__ = ["BankConfigSummary", "LoaderStats", "percentile"]


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile of ``values`` (``pct`` in 0-100).

    Returns ``0.0`` for an empty input.  Mirrors the helper the random-access
    benchmark script uses, lifted here so callers do not reinvent it.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


@dataclass(frozen=True, slots=True)
class BankConfigSummary:
    """Read-only snapshot of a bank's structural config (not live state).

    Built once from the :class:`~scdata.databank.DataBankConfig` the bank was
    constructed with, so a :class:`LoaderStats` snapshot carries enough context
    to interpret its throughput without re-reading the bank.
    """

    io_backend: str
    io_workers: int
    decode_workers: int
    cache_capacity_bytes: int
    memory_budget_bytes: int
    registered_datasets: int


@dataclass(frozen=True, slots=True)
class LoaderStats:
    """Consumer-side health snapshot of a loader iteration.

    Wait percentiles come from a bounded ring buffer of the most recent
    batches (so a long run reports recent health, not a whole-run average);
    ``batches_seen`` / ``stall_count`` are full-run counters.  Throughput uses
    wall seconds (produce + consumer), which is what the training loop
    experiences.
    """

    batches_seen: int
    cells_seen: int
    values_seen: int
    bytes_seen: int
    wall_seconds: float
    produce_seconds: float
    wait_p50_ms: float
    wait_p95_ms: float
    wait_p99_ms: float
    max_wait_ms: float
    stall_count: int
    stall_threshold_ms: float
    throughput_cells_per_s: float
    throughput_values_per_s: float
    throughput_gib_per_s: float
    consumer_fraction: float
    prefetch_config: "ScheduledPrefetchConfig | Mapping[str, Any] | None"
    bank_config_summary: "BankConfigSummary | None"


def _bank_config_summary(
    config: "DataBankConfig | None",
    registered: int,
) -> "BankConfigSummary | None":
    """Best-effort structural summary of a :class:`DataBankConfig`."""
    if config is None:
        return None
    io = config.io_config
    if io.backend == "uring":
        io_workers = io.uring_config.drivers
    else:
        io_workers = io.threaded_config.num_workers
    return BankConfigSummary(
        io_backend=io.backend,
        io_workers=io_workers,
        decode_workers=config.decode_config.num_workers,
        cache_capacity_bytes=config.access_config.cache_capacity_bytes,
        memory_budget_bytes=config.access_config.memory_budget_bytes,
        registered_datasets=registered,
    )


class _StatsCollector:
    """Accumulate per-batch wait/wall samples for :class:`LoaderStats`.

    Stores the last ``max_samples`` wait durations in a ring buffer for
    percentile computation; keeps online sums for aggregates and a full-run
    stall counter.  Thread-unsafe by design — ``ScDataLoader.__iter__`` runs
    single-threaded (``num_workers=0``), driven by the bank's own Rust pools.
    """

    __slots__ = (
        "_waits",
        "_max_samples",
        "_stall_threshold_s",
        "_batches",
        "_cells",
        "_values",
        "_bytes",
        "_produce",
        "_wall",
        "_stalls",
        "_prefetch_config",
        "_bank_config_summary",
    )

    def __init__(
        self,
        *,
        max_samples: int = 4096,
        stall_threshold_ms: float = 5.0,
        prefetch_config: "ScheduledPrefetchConfig | Mapping[str, Any] | None" = None,
        bank_config_summary: "BankConfigSummary | None" = None,
    ) -> None:
        if max_samples < 1:
            raise ValueError(f"max_samples must be >= 1, got {max_samples}")
        self._waits: deque[float] = deque(maxlen=max_samples)
        self._max_samples = max_samples
        self._stall_threshold_s = stall_threshold_ms / 1000.0
        self._batches = 0
        self._cells = 0
        self._values = 0
        self._bytes = 0
        self._produce = 0.0
        self._wall = 0.0
        self._stalls = 0
        self._prefetch_config = prefetch_config
        self._bank_config_summary = bank_config_summary

    def record_batch(
        self,
        *,
        wait_seconds: float,
        wall_seconds: float,
        num_cells: int,
        num_genes: int,
        bytes_: int,
    ) -> None:
        """Record one yielded batch.

        ``wait_seconds`` is the time blocked on the bank (the ``next()``
        calls); ``wall_seconds`` is the full per-batch time including collate.
        """
        self._waits.append(wait_seconds)
        self._batches += 1
        self._cells += num_cells
        self._values += num_cells * num_genes
        self._bytes += bytes_
        self._produce += wait_seconds
        self._wall += wall_seconds
        if wait_seconds > self._stall_threshold_s:
            self._stalls += 1

    def snapshot(self, *, reset: bool = True) -> LoaderStats:
        """Build a :class:`LoaderStats` from accumulated samples."""
        waits = list(self._waits)
        wall = self._wall
        produce = self._produce
        consumer_fraction = (1.0 - produce / wall) if wall > 0 else 0.0
        stats = LoaderStats(
            batches_seen=self._batches,
            cells_seen=self._cells,
            values_seen=self._values,
            bytes_seen=self._bytes,
            wall_seconds=wall,
            produce_seconds=produce,
            wait_p50_ms=percentile(waits, 50) * 1000.0,
            wait_p95_ms=percentile(waits, 95) * 1000.0,
            wait_p99_ms=percentile(waits, 99) * 1000.0,
            max_wait_ms=(max(waits) * 1000.0) if waits else 0.0,
            stall_count=self._stalls,
            stall_threshold_ms=self._stall_threshold_s * 1000.0,
            throughput_cells_per_s=(self._cells / wall) if wall > 0 else 0.0,
            throughput_values_per_s=(self._values / wall) if wall > 0 else 0.0,
            throughput_gib_per_s=(self._bytes / 1024**3 / wall) if wall > 0 else 0.0,
            consumer_fraction=consumer_fraction,
            prefetch_config=self._prefetch_config,
            bank_config_summary=self._bank_config_summary,
        )
        if reset:
            self.reset()
        return stats

    def reset(self) -> None:
        """Clear all accumulated samples and counters."""
        self._waits.clear()
        self._batches = 0
        self._cells = 0
        self._values = 0
        self._bytes = 0
        self._produce = 0.0
        self._wall = 0.0
        self._stalls = 0

    def set_prefetch_config(
        self,
        config: "ScheduledPrefetchConfig | Mapping[str, Any] | None",
    ) -> None:
        """Update the config stamp carried on subsequent snapshots."""
        self._prefetch_config = config

    @property
    def max_samples(self) -> int:
        return self._max_samples
