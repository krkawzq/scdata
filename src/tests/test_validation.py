from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import scdata.load as sc_load


def test_default_worker_count_uses_process_affinity(monkeypatch: pytest.MonkeyPatch) -> None:
    import scdata.load._config as config

    monkeypatch.setattr(config.os, "sched_getaffinity", lambda _pid: {2, 4, 6})
    monkeypatch.setattr(config.os, "cpu_count", lambda: 99)
    assert config._cpu_count() == 3


def test_config_defaults_are_python_owned_and_immutable() -> None:
    plan = sc_load.PlanConfig()
    merge = sc_load.IoMergeConfig()
    session = sc_load.SessionConfig()
    assert plan.compile_io_concurrency >= 1
    assert plan.limits.max_output_buffer_bytes == 2 * 1024 * 1024 * 1024
    assert plan.io_merge == merge
    assert merge.policy == "adjacent"
    assert merge._to_core()["max_io_gap_bytes"] == 0
    assert session.num_workers >= 1
    assert session.io_mode == "auto"
    assert session.queue_depth == 64
    assert sc_load.ReadLimits().max_metadata_size == 1024 * 1024
    with pytest.raises(AttributeError):
        plan.cache_capacity_bytes = 4  # type: ignore[misc]


def test_config_rejects_invalid_values_before_ffi() -> None:
    with pytest.raises(ValueError, match="positive"):
        sc_load.IoMergeConfig(io_bandwidth_bytes_per_second=0)
    with pytest.raises(ValueError, match="must not exceed"):
        sc_load.IoMergeConfig(
            max_coalesced_io_bytes=1024,
            max_encoded_staging_bytes_per_task=512,
        )
    with pytest.raises(ValueError, match="queue_depth"):
        sc_load.SessionConfig(io_mode="uring", queue_depth=1)
    with pytest.raises(TypeError, match="bool"):
        sc_load.ResourceLimits(max_cells_per_job=True)
    with pytest.raises(ValueError, match="smaller than the required"):
        sc_load.SessionConfig(
            num_workers=2,
            io_mode="blocking",
            max_total_inflight_io_ops=1,
        )._to_core()


def test_output_and_rows_reject_lossy_or_ambiguous_values() -> None:
    assert sc_load.OutputSpec(2, "u16").dtype == np.dtype(np.uint16)
    assert sc_load.OutputSpec(2, np.int64, fill=np.iinfo(np.int64).min).dtype == np.dtype(np.int64)
    assert sc_load.OutputSpec(2, "uint64", fill=np.iinfo(np.uint64).max).dtype == np.dtype(
        np.uint64
    )
    with pytest.raises(TypeError, match="integer"):
        sc_load.OutputSpec(2, np.uint16, fill=1.5)
    with pytest.raises(ValueError, match="overflows"):
        sc_load.OutputSpec(2, np.float32, fill=1e300)
    with pytest.raises(ValueError, match="overflow_value"):
        sc_load.OutputSpec(2, np.uint16, overflow_value=4)
    with pytest.raises(TypeError, match="unsupported output dtype"):
        sc_load.OutputSpec(2, np.int8)
    with pytest.raises(TypeError, match="bool"):
        sc_load.RowRef(True, 0)


def test_register_rejects_missing_or_invalid_scc(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        sc_load.register(tmp_path / "missing.scc")
    bare = tmp_path / "not-scc"
    bare.mkdir()
    with pytest.raises(ValueError, match="no sc-compress store"):
        sc_load.register(bare, key="X")


def test_feature_map_length_is_validated(tmp_path: Path) -> None:
    import anndata as ad
    import pandas as pd

    from scdata.anndata import write_scc

    adata = ad.AnnData(
        X=np.arange(6, dtype=np.float32).reshape(2, 3),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    path = write_scc(adata, tmp_path / "sample.scc", store="dir")
    with pytest.raises(ValueError, match="feature_map has length"):
        sc_load.register(path, feature_map=[0, 1])
