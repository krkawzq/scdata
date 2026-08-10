from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from sc_compress.anndata import write_scc
import scdata


def test_config_defaults_come_from_rust_and_are_immutable() -> None:
    plan = scdata.PlanConfig()
    session = scdata.SessionConfig()
    assert plan.compile_io_concurrency >= 1
    assert plan.limits.max_output_buffer_bytes > 0
    assert session.worker_count is None
    assert session.queue_depth >= 2
    assert scdata.ReadLimits().max_metadata_size > 0
    with pytest.raises(AttributeError):
        plan.coalescing_distance = 4


def test_config_rejects_invalid_values_before_ffi() -> None:
    with pytest.raises(ValueError, match="positive"):
        scdata.PlanConfig(io_bandwidth_bytes_per_second=0)
    with pytest.raises(ValueError, match="queue_depth"):
        scdata.SessionConfig(io_mode="uring", queue_depth=1)
    with pytest.raises(TypeError, match="bool"):
        scdata.ResourceLimits(max_cells_per_job=True)
    with pytest.raises(ValueError, match="smaller than the required"):
        scdata.SessionConfig(
            worker_count=2,
            io_mode="blocking",
            max_total_inflight_io_ops=1,
        )._to_core()


def test_output_and_rows_reject_lossy_or_ambiguous_values() -> None:
    assert scdata.OutputSpec(2, "u16").dtype == np.dtype(np.uint16)
    with pytest.raises(TypeError, match="integer"):
        scdata.OutputSpec(2, np.uint16, fill=1.5)
    with pytest.raises(ValueError, match="overflows"):
        scdata.OutputSpec(2, np.float32, fill=1e300)
    with pytest.raises(ValueError, match="overflow_value"):
        scdata.OutputSpec(2, np.uint16, overflow_value=4)
    with pytest.raises(TypeError, match="unsupported output dtype"):
        scdata.OutputSpec(2, np.int64)
    with pytest.raises(TypeError, match="bool"):
        scdata.RowRef(True, 0)


def test_register_rejects_missing_or_invalid_scc(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        scdata.register(tmp_path / "missing.scc")
    bare = tmp_path / "not-scc"
    bare.mkdir()
    with pytest.raises(ValueError, match="no sc-compress store"):
        scdata.register(bare, key="X")


def test_feature_map_length_is_validated(tmp_path: Path) -> None:
    adata = ad.AnnData(
        X=np.arange(6, dtype=np.float32).reshape(2, 3),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    path = write_scc(adata, tmp_path / "sample.scc", store="dir")
    with pytest.raises(ValueError, match="feature_map has length"):
        scdata.register(path, feature_map=[0, 1])
