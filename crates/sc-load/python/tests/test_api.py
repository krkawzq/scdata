from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from sc_compress.anndata import write_scc
import sc_load


def write_scc_x(tmp_path: Path, values: np.ndarray, name: str = "sample.scc") -> Path:
    n_obs, n_vars = values.shape
    adata = ad.AnnData(
        X=values,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )
    target = tmp_path / name
    write_scc(adata, target, store="dir" if not name.endswith(".zip") else "zip")
    return target


def test_register_feature_map_plan_and_owned_batches(tmp_path: Path) -> None:
    values = np.arange(12, dtype=np.uint16).reshape(4, 3).astype(np.float32)
    path = write_scc_x(tmp_path, values)
    dataset = sc_load.register(path, key="X", feature_map=[2, None, 0])
    assert dataset.kind in {"dense", "csr"}
    assert dataset.shape == (4, 3)
    assert dataset.feature_names == ("g0", "g1", "g2")
    assert "key='X'" in repr(dataset)

    output = sc_load.OutputSpec(4, np.float32, fill=-1)
    plan = sc_load.compile(
        dataset,
        [3, 1, 0],
        output=output,
        batch_size=2,
        prefetch_step=3,
    )
    assert plan.batch_count == 2
    assert plan.shape == (3, 4)
    assert plan.output is output
    assert plan.stats.input_rows == 3
    assert plan.stats.output_ring_bytes > 0
    assert plan.stats.as_dict()["input_rows"] == 3

    with plan.open(sc_load.SessionConfig(worker_count=1, io_mode="blocking")) as session:
        first = session.next_batch()
        second = session.next_batch()
        assert first is not None
        assert second is not None
        first_snapshot = first.copy()
        np.testing.assert_array_equal(
            first,
            np.array([[11, -1, 9, -1], [5, -1, 3, -1]], dtype=np.float32),
        )
        np.testing.assert_array_equal(second, np.array([[2, -1, 0, -1]], dtype=np.float32))
        np.testing.assert_array_equal(first, first_snapshot)
        assert session.rows_yielded == 3
        assert session.rows_remaining == 0
        assert session.next_batch() is None
        assert session.exhausted
        assert session.stats.state == "finished"
        assert session.stats.as_dict()["state"] == "finished"


def test_prefetch_is_reusable_via_plan_and_preserves_row_order(tmp_path: Path) -> None:
    values = np.arange(20, dtype=np.float32).reshape(5, 4)
    dataset = sc_load.register(write_scc_x(tmp_path, values))
    plan = sc_load.compile(
        dataset,
        np.array([4, 0, 2], dtype=np.uint64),
        batch_size=1,
        prefetch_step=2,
    )
    expected = values[[4, 0, 2]]
    config = sc_load.SessionConfig(worker_count=1, io_mode="blocking")
    np.testing.assert_array_equal(plan.read(config), expected)
    result = np.concatenate(
        list(sc_load.prefetch(dataset, [4, 0, 2], batch_size=1, prefetch_step=2, config=config))
    )
    np.testing.assert_array_equal(result, expected)
    assert result.flags.c_contiguous
    with plan.open(config) as session:
        np.testing.assert_array_equal(session.next_batch(), expected[:1])
        np.testing.assert_array_equal(session.read(), expected[1:])


def test_cancel_is_structured_and_close_is_idempotent(tmp_path: Path) -> None:
    values = np.arange(64, dtype=np.float32).reshape(16, 4)
    dataset = sc_load.register(write_scc_x(tmp_path, values))
    plan = sc_load.compile(
        dataset,
        range(16),
        output=sc_load.OutputSpec(4, np.float32),
        batch_size=2,
        prefetch_step=3,
    )
    session = plan.open(sc_load.SessionConfig(worker_count=1, io_mode="blocking"))
    session.cancel()
    with pytest.raises(sc_load.CancelledError) as raised:
        session.next_batch()
    assert raised.value.kind == "cancelled"
    session.close()
    session.close()
    assert session.closed
    assert session.state == "cancelled"


def test_empty_plan_finishes_without_batches() -> None:
    plan = sc_load.compile(
        [],
        [],
        output=sc_load.OutputSpec(3, np.float32),
        batch_size=4,
        prefetch_step=2,
    )
    assert plan.is_empty
    with plan.open(sc_load.SessionConfig(worker_count=1, io_mode="blocking")) as session:
        result = session.read()
        assert result.shape == (0, 3)
        assert result.dtype == np.dtype(np.float32)
        assert session.state == "finished"


def test_checked_conversion_policies(tmp_path: Path) -> None:
    values = np.array([[-3, 4]], dtype=np.int16)
    adata = ad.AnnData(
        X=values,
        obs=pd.DataFrame(index=["c0"]),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    dataset = sc_load.register(write_scc(adata, tmp_path / "signed.scc", store="dir"))

    failing = sc_load.compile(
        dataset,
        [0],
        output=sc_load.OutputSpec(2, np.uint16),
        batch_size=1,
        prefetch_step=2,
    )
    with failing.open(sc_load.SessionConfig(worker_count=1, io_mode="blocking")) as session:
        with pytest.raises(sc_load.SessionError):
            session.next_batch()

    filled = sc_load.compile(
        dataset,
        [0],
        output=sc_load.OutputSpec(2, np.uint16, fill=99, overflow="use_fill"),
        batch_size=1,
        prefetch_step=2,
    )
    with filled.open(sc_load.SessionConfig(worker_count=1, io_mode="blocking")) as session:
        np.testing.assert_array_equal(session.read(), np.array([[99, 4]], dtype=np.uint16))


def test_rounding_conversion_requires_opt_in(tmp_path: Path) -> None:
    adata = ad.AnnData(
        X=np.array([[16_777_217]], dtype=np.int32),
        obs=pd.DataFrame(index=["c0"]),
        var=pd.DataFrame(index=["g0"]),
    )
    dataset = sc_load.register(write_scc(adata, tmp_path / "int32.scc", store="dir"))
    common = dict(datasets=dataset, rows=[0], batch_size=1, prefetch_step=2)
    with pytest.raises(sc_load.PromotionError):
        sc_load.compile(**common, output=sc_load.OutputSpec(1, np.float32))

    plan = sc_load.compile(
        **common,
        output=sc_load.OutputSpec(1, np.float32, allow_float_rounding=True),
    )
    with plan.open(sc_load.SessionConfig(worker_count=1, io_mode="blocking")) as session:
        np.testing.assert_array_equal(session.read(), np.array([[16_777_216]], dtype=np.float32))


def test_scc_zip_and_layer_key(tmp_path: Path) -> None:
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    layer = values + 10
    adata = ad.AnnData(
        X=values,
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
        layers={"counts": layer},
    )
    archive = tmp_path / "matrix.scc.zip"
    write_scc(adata, archive, store="zip")

    x = sc_load.register(archive, key="X")
    assert x.zip_prefix == "X"
    assert x.feature_names == ("g0", "g1", "g2")
    counts = sc_load.register(archive, key="layers/counts")
    assert counts.zip_prefix == "layers/counts"

    config = sc_load.SessionConfig(worker_count=1, io_mode="auto")
    np.testing.assert_array_equal(
        sc_load.compile(x, [1], batch_size=1, prefetch_step=2).read(config),
        values[1:2],
    )
    np.testing.assert_array_equal(
        sc_load.compile(counts, [0], batch_size=1, prefetch_step=2).read(config),
        layer[0:1],
    )


def test_obsm_has_no_feature_names_and_identity_prefetch(tmp_path: Path) -> None:
    adata = ad.AnnData(
        X=np.arange(12, dtype=np.float32).reshape(3, 4),
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
    )
    adata.obsm["X_pca"] = np.arange(6, dtype=np.float32).reshape(3, 2)
    path = write_scc(adata, tmp_path / "emb.scc", store="dir")
    pca = sc_load.register(path, key="obsm/X_pca")
    assert pca.feature_names is None
    assert pca.shape == (3, 2)
    result = list(sc_load.prefetch(pca, [2, 0], batch_size=1, prefetch_step=2))
    np.testing.assert_array_equal(np.concatenate(result), adata.obsm["X_pca"][[2, 0]])


def test_dense_and_csr_sources_can_be_interleaved(tmp_path: Path) -> None:
    dense_values = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    dense = sc_load.register(write_scc_x(tmp_path, dense_values, "dense.scc"))

    csr_matrix = sp.csr_matrix(
        np.array(
            [
                [10, 0, 0, 40],
                [0, 20, 30, 0],
            ],
            dtype=np.float32,
        )
    )
    csr_adata = ad.AnnData(
        X=csr_matrix,
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
    )
    csr = sc_load.register(write_scc(csr_adata, tmp_path / "csr.scc", store="dir"))
    assert csr.kind == "csr"

    plan = sc_load.compile(
        [dense, csr],
        [(1, 1), (0, 0), (1, 0)],
        output=sc_load.OutputSpec(4, np.float32),
        batch_size=2,
        prefetch_step=3,
    )
    with plan.open(sc_load.SessionConfig(worker_count=2, io_mode="blocking")) as session:
        np.testing.assert_array_equal(
            session.read(),
            np.array([[0, 20, 30, 0], [1, 2, 3, 4], [10, 0, 0, 40]], dtype=np.float32),
        )


def test_register_accepts_sc_compress_read_limits(tmp_path: Path) -> None:
    import sc_compress as scc

    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    path = write_scc_x(tmp_path, values)
    dataset = sc_load.register(path, limits=scc.ReadLimits(max_block_count=10_000))
    assert dataset.limits.max_block_count == 10_000
    assert dataset.feature_names == ("g0", "g1", "g2")


def test_with_feature_map_returns_new_handle(tmp_path: Path) -> None:
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    base = sc_load.register(write_scc_x(tmp_path, values))
    mapped = base.with_feature_map([1, None, 0])
    assert base.feature_map is None
    assert mapped.feature_map == (1, None, 0)
    assert mapped.path == base.path
    assert mapped.key == base.key
