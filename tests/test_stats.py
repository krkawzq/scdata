"""Tests for the consumer-side health metrics collector."""

from __future__ import annotations

import statistics

import pytest

from scdata.data._stats import (
    BankConfigSummary,
    LoaderStats,
    _StatsCollector,
    percentile,
)


def test_percentile_empty_returns_zero() -> None:
    assert percentile([], 50) == 0.0


def test_percentile_single_element() -> None:
    assert percentile([7.0], 99) == 7.0


def test_percentile_linear_interpolation() -> None:
    values = [float(i) for i in range(1, 11)]  # 1..10
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 10.0
    assert percentile(values, 50) == 5.5  # midpoint of 5 and 6
    # p95: rank = 9 * 0.95 = 8.55 -> interpolate ordered[8]=9, ordered[9]=10
    assert percentile(values, 95) == pytest.approx(9.55)


def test_percentile_matches_sorted_index_for_p50() -> None:
    values = [3.0, 1.0, 2.0, 5.0, 4.0]
    assert percentile(values, 50) == pytest.approx(statistics.median(values))


def test_collector_rejects_bad_max_samples() -> None:
    with pytest.raises(ValueError):
        _StatsCollector(max_samples=0)


def test_collector_records_aggregates_and_percentiles() -> None:
    collector = _StatsCollector(max_samples=100, stall_threshold_ms=5.0)
    # waits in seconds: 0.001, 0.002, ..., 0.010
    for i in range(1, 11):
        wait = i / 1000.0
        collector.record_batch(
            wait_seconds=wait,
            wall_seconds=wait * 2,  # consumer == produce
            num_cells=10,
            num_genes=4,
            bytes_=10 * 4 * 4,
        )
    stats = collector.snapshot(reset=False)
    assert stats.batches_seen == 10
    assert stats.cells_seen == 100
    assert stats.values_seen == 400
    assert stats.bytes_seen == 100 * 4 * 4
    assert stats.produce_seconds == pytest.approx(sum(i / 1000.0 for i in range(1, 11)))
    assert stats.wall_seconds == pytest.approx(2 * stats.produce_seconds)
    # max wait is 0.010 s = 10 ms
    assert stats.max_wait_ms == pytest.approx(10.0, abs=1e-6)
    # p50 over [0.001..0.010] -> midpoint of 0.005,0.006 = 0.0055 s = 5.5 ms
    assert stats.wait_p50_ms == pytest.approx(5.5, abs=1e-6)
    # consumer == produce -> consumer_fraction = 1 - 0.5 = 0.5
    assert stats.consumer_fraction == pytest.approx(0.5)


def test_collector_stall_count() -> None:
    collector = _StatsCollector(stall_threshold_ms=5.0)
    collector.record_batch(wait_seconds=0.001, wall_seconds=0.001, num_cells=1, num_genes=1, bytes_=4)
    collector.record_batch(wait_seconds=0.020, wall_seconds=0.020, num_cells=1, num_genes=1, bytes_=4)  # 20ms > 5ms
    collector.record_batch(wait_seconds=0.004, wall_seconds=0.004, num_cells=1, num_genes=1, bytes_=4)
    stats = collector.snapshot(reset=False)
    assert stats.stall_count == 1
    assert stats.stall_threshold_ms == pytest.approx(5.0)


def test_collector_ring_buffer_keeps_recent_only() -> None:
    collector = _StatsCollector(max_samples=4)
    for i in range(1, 7):  # 6 samples, ring keeps last 4 -> [3,4,5,6] ms
        collector.record_batch(
            wait_seconds=i / 1000.0,
            wall_seconds=i / 1000.0,
            num_cells=1,
            num_genes=1,
            bytes_=4,
        )
    stats = collector.snapshot(reset=False)
    # batches_seen is a full-run counter (6), percentiles use the ring (last 4)
    assert stats.batches_seen == 6
    assert stats.wait_p50_ms == pytest.approx(4.5, abs=1e-6)  # midpoint of 4ms,5ms
    assert stats.max_wait_ms == pytest.approx(6.0, abs=1e-6)


def test_collector_reset_clears_everything() -> None:
    collector = _StatsCollector()
    collector.record_batch(wait_seconds=0.01, wall_seconds=0.01, num_cells=1, num_genes=1, bytes_=4)
    stats = collector.snapshot(reset=True)  # default resets
    assert stats.batches_seen == 1
    stats2 = collector.snapshot(reset=True)
    assert stats2.batches_seen == 0
    assert stats2.wait_p99_ms == 0.0


def test_collector_snapshot_no_reset_accumulates() -> None:
    collector = _StatsCollector()
    collector.record_batch(wait_seconds=0.001, wall_seconds=0.001, num_cells=1, num_genes=1, bytes_=4)
    collector.snapshot(reset=False)
    collector.record_batch(wait_seconds=0.002, wall_seconds=0.002, num_cells=1, num_genes=1, bytes_=4)
    stats = collector.snapshot(reset=True)
    assert stats.batches_seen == 2


def test_collector_set_prefetch_config_stamps_snapshot() -> None:
    collector = _StatsCollector()
    sentinel = object()  # the real config type comes from the Rust-backed layer
    collector.set_prefetch_config(sentinel)  # type: ignore[arg-type]
    collector.record_batch(wait_seconds=0.001, wall_seconds=0.001, num_cells=1, num_genes=1, bytes_=4)
    stats = collector.snapshot(reset=False)
    assert stats.prefetch_config is sentinel


def test_loader_stats_is_frozen() -> None:
    collector = _StatsCollector(bank_config_summary=None)
    collector.record_batch(wait_seconds=0.001, wall_seconds=0.002, num_cells=1, num_genes=1, bytes_=4)
    stats = collector.snapshot(reset=False)
    assert isinstance(stats, LoaderStats)
    with pytest.raises((AttributeError, Exception)):
        stats.batches_seen = 99  # type: ignore[misc]


def test_bank_config_summary_is_frozen() -> None:
    summary = BankConfigSummary(
        io_backend="threaded",
        io_workers=24,
        decode_workers=24,
        cache_capacity_bytes=128,
        memory_budget_bytes=256,
        registered_datasets=3,
    )
    with pytest.raises((AttributeError, Exception)):
        summary.io_backend = "uring"  # type: ignore[misc]
