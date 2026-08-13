from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scdata.anndata import write_scc
import scdata.load as sc_load


def test_config_defaults_come_from_rust_and_are_immutable() -> None:
    plan = sc_load.PlanConfig()
    session = sc_load.SessionConfig()
    assert plan.compile_io_concurrency >= 1
    assert plan.limits.max_output_buffer_bytes > 0
    assert session.worker_count is None
    assert session.queue_depth >= 2
    assert sc_load.ReadLimits().max_metadata_size > 0
    with pytest.raises(AttributeError):
        plan.coalescing_distance = 4


def test_config_rejects_invalid_values_before_ffi() -> None:
    with pytest.raises(ValueError, match="positive"):
        sc_load.PlanConfig(io_bandwidth_bytes_per_second=0)
    with pytest.raises(ValueError, match="queue_depth"):
        sc_load.SessionConfig(io_mode="uring", queue_depth=1)
    with pytest.raises(TypeError, match="bool"):
        sc_load.ResourceLimits(max_cells_per_job=True)
    with pytest.raises(ValueError, match="smaller than the required"):
        sc_load.SessionConfig(
            worker_count=2,
            io_mode="blocking",
            max_total_inflight_io_ops=1,
        )._to_core()


def test_output_and_rows_reject_lossy_or_ambiguous_values() -> None:
    assert sc_load.OutputSpec(2, "u16").dtype == np.dtype(np.uint16)
    with pytest.raises(TypeError, match="integer"):
        sc_load.OutputSpec(2, np.uint16, fill=1.5)
    with pytest.raises(ValueError, match="overflows"):
        sc_load.OutputSpec(2, np.float32, fill=1e300)
    with pytest.raises(ValueError, match="overflow_value"):
        sc_load.OutputSpec(2, np.uint16, overflow_value=4)
    with pytest.raises(TypeError, match="unsupported output dtype"):
        sc_load.OutputSpec(2, np.int64)
    with pytest.raises(TypeError, match="bool"):
        sc_load.RowRef(True, 0)


def test_register_rejects_missing_or_invalid_scc(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        sc_load.register(tmp_path / "missing.scc")
    bare = tmp_path / "not-scc"
    bare.mkdir()
    with pytest.raises(ValueError, match="no sc-compress store"):
        sc_load.register(bare, key="X")


def test_feature_map_length_is_validated(tmp_path: Path) -> None:
    adata = ad.AnnData(
        X=np.arange(6, dtype=np.float32).reshape(2, 3),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    path = write_scc(adata, tmp_path / "sample.scc", store="dir")
    with pytest.raises(ValueError, match="feature_map has length"):
        sc_load.register(path, feature_map=[0, 1])
