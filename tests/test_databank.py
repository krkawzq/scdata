"""End-to-end tests for the Pythonic ``ScDataBank`` wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scdata import DataBankConfig, IoConfig, MissingGenePolicy, ScDataBank
from scdata._scdata import DataBankError
from scdata.data import DatasetCollection, DenseDataset, DType, SparseDataset
from scdata.io import launch, launch_all, write_zarr

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sp = pytest.importorskip("scipy.sparse")
pytest.importorskip("zarr")


def _dense_adata(
    shape: tuple[int, int],
    np_dtype: Any,
    gene_names: list[str],
) -> tuple[Any, np.ndarray]:
    expected = np.arange(int(np.prod(shape)), dtype=np_dtype).reshape(shape)
    return (
        ad.AnnData(
            X=expected,
            obs=pd.DataFrame(index=[f"c{i}" for i in range(shape[0])]),
            var=pd.DataFrame(index=gene_names),
        ),
        expected,
    )


def _dense_store(
    tmp_path: Path,
    name: str,
    shape: tuple[int, int] = (3, 4),
    np_dtype: Any = np.float32,
    gene_names: list[str] | None = None,
    *,
    store: str = "dir",
    format: str = "dense2d",
) -> tuple[Path, DenseDataset, np.ndarray, list[str]]:
    if gene_names is None:
        gene_names = [f"g{i}" for i in range(shape[1])]
    adata, expected = _dense_adata(shape, np_dtype, gene_names)
    suffix = ".zarr.zip" if store == "zip" else ".zarr"
    root = write_zarr(
        adata,
        tmp_path / f"{name}{suffix}",
        format=format,  # type: ignore[arg-type]
        chunk_size=(2, min(3, shape[1])) if format == "dense2d" else (max(shape[1], 1) * 2,),
        store=store,  # type: ignore[arg-type]
    )
    ds = launch(root)
    assert isinstance(ds, DenseDataset)
    return root, ds, expected, gene_names


def test_register_dense_2d(tmp_path: Path) -> None:
    root, ds, _, genes = _dense_store(
        tmp_path, "dense", (3, 4), np.float32, ["g0", "g1", "g2", "g3"]
    )
    bank = ScDataBank()
    did = bank.register_dense(ds, str(root))
    assert bank.dataset_num_cells(did) == 3
    assert bank.dataset_num_genes(did) == 4
    assert bank.dataset_dtype(did) == DType.F32
    assert bank.dataset_genes(did) == genes
    bank.unregister(did)
    with pytest.raises(DataBankError):
        bank.dataset_genes(did)


def test_dataset_id_identity(tmp_path: Path) -> None:
    root, ds, _, _ = _dense_store(tmp_path, "id", (3, 4), np.float32, ["g0", "g1", "g2", "g3"])
    bank = ScDataBank()
    did = bank.register_dense(ds, str(root))
    assert did == did
    assert hash(did) == hash(did)
    assert "DatasetId" in repr(did)
    did2 = bank.register_dense(ds, str(root))
    assert did2 != did
    bank.unregister(did)
    bank.unregister(did2)


def test_config_dynamic_routing_and_validation() -> None:
    cfg = DataBankConfig.make(
        backend="uring",
        entries=512,
        cache_capacity_bytes=512 * 1024**2,
        io__uring__base__max_in_flight=2048,
    )

    assert cfg.io_config.backend == "uring"
    assert cfg.io_config.uring_config.entries == 512
    assert cfg.io_config.uring_config.base.max_in_flight == 2048
    assert cfg.access_config.cache_capacity_bytes == 512 * 1024**2

    cfg.update(fill__num_workers=8)
    assert cfg.fill_config.num_workers == 8

    with pytest.raises(TypeError, match="ambiguous"):
        DataBankConfig.make(num_workers=8)
    with pytest.raises(TypeError):
        DataBankConfig.make(entires=512)
    with pytest.raises(ValueError, match="backend"):
        IoConfig.make(backend="bad")
    with pytest.raises(ValueError, match="backend"):
        DataBankConfig.make(backend="bad")


def test_config_accepts_dict_and_nested_dicts() -> None:
    cfg = DataBankConfig.from_dict(
        {
            "io": {
                "backend": "uring",
                "uring": {"entries": 1024, "base": {"max_in_flight": 2048}},
            },
            "access_config": {
                "cache_capacity_bytes": 16 * 1024**2,
                "memory_budget_bytes": 32 * 1024**2,
                "cpu": {"num_workers": 2},
            },
            "decode": {"num_workers": 3},
            "fill__num_workers": 4,
        }
    )

    assert cfg.io_config.backend == "uring"
    assert cfg.io_config.uring_config.entries == 1024
    assert cfg.io_config.uring_config.base.max_in_flight == 2048
    assert cfg.access_config.cpu.num_workers == 2
    assert cfg.decode_config.num_workers == 3
    assert cfg.fill_config.num_workers == 4

    cfg.update({"io": {"uring": {"drivers": 2}}, "fill": {"queue_capacity": 16}})
    assert cfg.io_config.uring_config.drivers == 2
    assert cfg.fill_config.queue_capacity == 16

    direct = DataBankConfig(
        io_config={"backend": "threaded", "threaded": {"num_workers": 9}},
        access_config={"cpu": {"queue_capacity": 7}},
    )
    assert isinstance(direct.io_config, IoConfig)
    assert direct.io_config.threaded_config.num_workers == 9
    assert direct.access_config.cpu.queue_capacity == 7

    io = IoConfig.uring({"entries": 256, "base": {"priority_levels": 5}})
    assert io.backend == "uring"
    assert io.uring_config.entries == 256
    assert io.uring_config.base.priority_levels == 5

    bank = ScDataBank(
        {"access": {"cache_capacity_bytes": 16 * 1024**2, "memory_budget_bytes": 32 * 1024**2}}
    )
    bank.close()


def test_access_dense_values(tmp_path: Path) -> None:
    root, ds, expected, _ = _dense_store(
        tmp_path, "acc", (5, 6), np.float32, [f"g{i}" for i in range(6)]
    )
    bank = ScDataBank()
    did = bank.register_dense(ds, str(root))
    result = bank.load(did, [0, 1, 2, 3, 4], dtype="f32")
    out = np.asarray(result)
    assert out.shape == (30,)
    assert result.shape == (5, 6)
    assert result.var_names == tuple(f"g{i}" for i in range(6))
    assert np.shares_memory(result.to_numpy(), result.data)
    assert np.shares_memory(result.to_flat_numpy(), result.data)
    assert np.array_equal(out.reshape(5, 6), expected)
    out2 = np.asarray(bank.load(did, [3, 0, 4]))
    assert np.array_equal(out2.reshape(3, 6), expected[[3, 0, 4]])
    bank.unregister(did)


def test_load_and_prefetch_accept_numpy_inputs(tmp_path: Path) -> None:
    root, ds, expected, _ = _dense_store(
        tmp_path, "public_fast", (5, 6), np.float32, [f"g{i}" for i in range(6)]
    )
    bank = ScDataBank()
    did = bank.register_dense(ds, str(root))

    out = np.asarray(bank.load(did, np.array([4, 0, 2], dtype=np.intp)))
    assert np.array_equal(out.reshape(3, 6), expected[[4, 0, 2]])

    projected = np.asarray(bank.load(did, np.array([1, 3], dtype=np.intp), genes=["g5", "g1"]))
    assert np.array_equal(projected.reshape(2, 2), expected[[1, 3]][:, [5, 1]])

    batches = list(
        bank.prefetch(
            did,
            [np.array([0, 2], dtype=np.intp), np.array([4], dtype=np.intp)],
        )
    )
    assert [batch.cells.tolist() for batch in batches] == [[0, 2], [4]]
    assert np.array_equal(np.asarray(batches[0].data).reshape(2, 6), expected[[0, 2]])
    assert np.array_equal(np.asarray(batches[1].data).reshape(1, 6), expected[[4]])

    projected_batches = list(
        bank.prefetch(did, [np.array([0, 1], dtype=np.intp)], genes=["g0", "g5"])
    )
    assert len(projected_batches) == 1
    assert np.array_equal(
        np.asarray(projected_batches[0].data).reshape(2, 2),
        expected[[0, 1]][:, [0, 5]],
    )
    bank.unregister(did)


@pytest.mark.parametrize("format", ["dense2d", "dense1d"])
def test_access_dense_v3_layouts(tmp_path: Path, format: str) -> None:
    root, ds, expected, _ = _dense_store(
        tmp_path,
        f"acc_{format}",
        (5, 6),
        np.float64,
        [f"g{i}" for i in range(6)],
        format=format,
    )
    bank = ScDataBank()
    did = bank.register_dense(ds, str(root))
    out = np.asarray(bank.load(did, list(range(5))))
    assert np.array_equal(out.reshape(5, 6), expected)
    bank.unregister(did)


def test_register_launched_zip_dense(tmp_path: Path) -> None:
    _, ds, expected, _ = _dense_store(
        tmp_path,
        "zip_bank",
        (5, 6),
        np.float32,
        [f"g{i}" for i in range(6)],
        store="zip",
        format="dense1d",
    )

    bank = ScDataBank()
    did = bank.register(ds)
    out = np.asarray(bank.load(did, [4, 0, 2]))

    assert np.array_equal(out.reshape(3, 6), expected[[4, 0, 2]])
    bank.unregister(did)


def test_register_accepts_iterable_datasets(tmp_path: Path) -> None:
    roots = [
        _dense_store(
            tmp_path,
            "iter_bank_a",
            (3, 4),
            np.float32,
            ["g0", "g1", "g2", "g3"],
        )[0],
        _dense_store(tmp_path, "iter_bank_b", (5, 2), np.float64, ["h0", "h1"])[0],
    ]
    datasets = (launch(root) for root in roots)

    bank = ScDataBank()
    ids = bank.register(datasets)
    try:
        assert isinstance(ids, list)
        assert len(ids) == 2
        assert [bank.dataset_num_cells(did) for did in ids] == [3, 5]
        assert [bank.dataset_num_genes(did) for did in ids] == [4, 2]
        assert [bank.dataset_dtype(did) for did in ids] == [DType.F32, DType.F64]
    finally:
        for did in ids:
            bank.unregister(did)


def test_register_iterable_rolls_back_on_error(tmp_path: Path) -> None:
    root, _, _, _ = _dense_store(
        tmp_path, "iter_bank_rollback", (3, 4), np.float32, ["g0", "g1", "g2", "g3"]
    )
    datasets = (item for item in (launch(root), object()))

    bank = ScDataBank()
    with pytest.raises(TypeError, match="unsupported dataset type"):
        bank.register(datasets)

    assert "registered=0" in repr(bank)


def test_access_dtype_round_trip(tmp_path: Path) -> None:
    for np_dt in (np.int32, np.int64, np.uint8, np.float16, np.float32, np.float64):
        root, ds, expected, _ = _dense_store(
            tmp_path,
            f"dt_{np.dtype(np_dt).name}",
            (3, 4),
            np_dt,
            ["g0", "g1", "g2", "g3"],
        )
        bank = ScDataBank()
        did = bank.register_dense(ds, str(root))
        out = bank.load(did, [0, 1, 2])
        arr = np.asarray(out)
        assert arr.dtype == np.dtype(np_dt)
        assert np.array_equal(arr.reshape(3, 4), expected)
        bank.unregister(did)


def test_load_by_genes(tmp_path: Path) -> None:
    root, ds, expected, _ = _dense_store(
        tmp_path, "gn", (5, 6), np.float32, [f"g{i}" for i in range(6)]
    )
    bank = ScDataBank()
    did = bank.register_dense(ds, str(root))
    out = np.asarray(bank.load(did, [0, 2], genes=["g1", "g3"]))
    assert out.shape == (4,)
    assert np.array_equal(out.reshape(2, 2), expected[[0, 2]][:, [1, 3]])
    out_single = np.asarray(bank.load(did, [0, 2], genes="g1"))
    assert out_single.shape == (2,)
    assert np.array_equal(out_single.reshape(2, 1), expected[[0, 2]][:, [1]])
    bank.unregister(did)


def test_empty_gene_projection_returns_zero_columns(tmp_path: Path) -> None:
    root, ds, _, _ = _dense_store(
        tmp_path, "empty_genes", (5, 6), np.float32, [f"g{i}" for i in range(6)]
    )
    bank = ScDataBank()
    did = bank.register_dense(ds, str(root))
    try:
        loaded = bank.load(did, [0, 2], genes=[])
        assert loaded.shape == (2, 0)
        assert loaded.var_names == ()
        assert loaded.data.shape == (0,)

        prefetched = next(iter(bank.prefetch(did, [[0, 2]], genes=[])))
        assert prefetched.shape == (2, 0)
        assert prefetched.var_names == ()
        assert prefetched.data.shape == (0,)
    finally:
        bank.unregister(did)


def test_load_by_genes_missing_error(tmp_path: Path) -> None:
    root, ds, _, _ = _dense_store(tmp_path, "gn_err", (3, 4), np.float32, ["g0", "g1", "g2", "g3"])
    bank = ScDataBank()
    did = bank.register_dense(ds, str(root))
    with pytest.raises(DataBankError):
        bank.load(did, [0], genes=["nope"], missing=MissingGenePolicy.ERROR)
    with pytest.raises(DataBankError):
        bank.load(did, [0], genes=["nope"], missing="error")
    bank.unregister(did)


def test_access_unregistered_raises(tmp_path: Path) -> None:
    root, ds, _, _ = _dense_store(tmp_path, "unreg", (3, 4), np.float32, ["g0", "g1", "g2", "g3"])
    bank = ScDataBank()
    did = bank.register_dense(ds, str(root))
    bank.unregister(did)
    with pytest.raises(DataBankError):
        bank.load(did, [0])


def test_prefetch(tmp_path: Path) -> None:
    root, ds, expected, _ = _dense_store(
        tmp_path, "pf", (5, 6), np.float32, [f"g{i}" for i in range(6)]
    )
    bank = ScDataBank()
    did = bank.register_dense(ds, str(root))
    batches = [[0, 1], [2, 3, 4]]
    pf = bank.prefetch(did, batches, config={"prefetch_step": 2, "access": {"prefetch_step": 2}})
    seen = 0
    for batch in pf:
        assert hasattr(batch, "cells") and hasattr(batch, "data") and hasattr(batch, "num_genes")
        cells = np.asarray(batch.cells)
        data = np.asarray(batch.data)
        assert batch.num_genes == 6
        assert batch.var_names == tuple(f"g{i}" for i in range(6))
        assert batch.to_numpy().shape == (len(cells), 6)
        assert data.shape == (len(cells) * 6,)
        seen += len(cells)
        rows = expected[cells.tolist()]
        assert np.array_equal(data.reshape(len(cells), 6), rows)
    assert seen == 5
    bank.unregister(did)


def test_prefetch_multi_dataset_casts_to_single_output_dtype(tmp_path: Path) -> None:
    root16, ds16, expected16, _ = _dense_store(
        tmp_path,
        "pf_multi_f16",
        (4, 3),
        np.float16,
        ["g0", "g1", "g2"],
    )
    root32, ds32, expected32, _ = _dense_store(
        tmp_path,
        "pf_multi_f32",
        (4, 3),
        np.float32,
        ["g0", "g1", "g2"],
    )
    bank = ScDataBank()
    ids = [bank.register_dense(ds16, str(root16)), bank.register_dense(ds32, str(root32))]
    try:
        batches = [
            [(0, np.array([0, 2], dtype=np.intp)), (1, [1])],
            [(1, [3]), (0, [1])],
        ]
        out = list(bank.prefetch(ids, batches, dtype="f32"))

        assert [batch.cells.tolist() for batch in out] == [[0, 2, 1], [3, 1]]
        assert [batch.data.dtype for batch in out] == [np.dtype("float32"), np.dtype("float32")]
        assert np.array_equal(
            out[0].to_numpy(),
            np.vstack([expected16[[0, 2]].astype(np.float32), expected32[[1]]]),
        )
        assert np.array_equal(
            out[1].to_numpy(),
            np.vstack([expected32[[3]], expected16[[1]].astype(np.float32)]),
        )

        auto = next(iter(bank.prefetch(ids, [[(0, [0]), (1, [0])]], dtype="auto")))
        assert auto.data.dtype == np.dtype("float32")
        assert np.array_equal(
            auto.to_numpy(),
            np.vstack([expected16[[0]].astype(np.float32), expected32[[0]]]),
        )

        with pytest.raises(DataBankError):
            list(bank.prefetch(ids, [[(0, [0]), (1, [0])]], dtype="i32"))
    finally:
        for did in ids:
            bank.unregister(did)


def test_load_multi_dataset_uses_single_output_dtype(tmp_path: Path) -> None:
    root16, ds16, expected16, _ = _dense_store(
        tmp_path,
        "load_multi_f16",
        (3, 2),
        np.float16,
        ["g0", "g1"],
    )
    root32, ds32, expected32, _ = _dense_store(
        tmp_path,
        "load_multi_f32",
        (3, 2),
        np.float32,
        ["g0", "g1"],
    )
    bank = ScDataBank()
    ids = [bank.register_dense(ds16, str(root16)), bank.register_dense(ds32, str(root32))]
    try:
        out = bank.load(ids, [(0, [2, 0]), (1, np.array([1], dtype=np.intp))])

        assert out.cells.tolist() == [2, 0, 1]
        assert out.data.dtype == np.dtype("float32")
        assert out.var_names == ("g0", "g1")
        assert np.array_equal(
            out.to_numpy(),
            np.vstack([expected16[[2, 0]].astype(np.float32), expected32[[1]]]),
        )

        projected = bank.load(ids, [(1, [2]), (0, [1])], genes=["g1"], dtype="f16")
        assert projected.data.dtype == np.dtype("float16")
        assert projected.var_names == ("g1",)
        assert np.array_equal(
            projected.to_numpy(),
            np.vstack([expected32[[2]][:, [1]], expected16[[1]][:, [1]]]).astype(np.float16),
        )

        with pytest.raises(DataBankError):
            bank.load(ids, [(0, [0]), (1, [0])], dtype="i32")
    finally:
        for did in ids:
            bank.unregister(did)


def test_prefetch_by_genes(tmp_path: Path) -> None:
    root, ds, expected, _ = _dense_store(
        tmp_path, "pfgn", (5, 6), np.float32, [f"g{i}" for i in range(6)]
    )
    bank = ScDataBank()
    did = bank.register_dense(ds, str(root))
    pf = bank.prefetch(did, [[0, 1], [2]], genes="g0")
    seen = 0
    for batch in pf:
        cells = np.asarray(batch.cells)
        data = np.asarray(batch.data)
        assert batch.num_genes == 1
        assert batch.var_names == ("g0",)
        rows = expected[cells.tolist()][:, [0]]
        assert np.array_equal(data.reshape(len(cells), 1), rows)
        seen += len(cells)
    assert seen == 3
    bank.unregister(did)


def test_multiple_banks(tmp_path: Path) -> None:
    r1, d1_ds, _, _ = _dense_store(tmp_path, "b1", (3, 4), np.float32, ["g0", "g1", "g2", "g3"])
    r2, d2_ds, _, _ = _dense_store(tmp_path, "b2", (3, 4), np.float64, ["h0", "h1", "h2", "h3"])
    b1 = ScDataBank()
    b2 = ScDataBank()
    d1 = b1.register_dense(d1_ds, str(r1))
    d2 = b2.register_dense(d2_ds, str(r2))
    assert b1.dataset_genes(d1) == ["g0", "g1", "g2", "g3"]
    assert b2.dataset_genes(d2) == ["h0", "h1", "h2", "h3"]
    del b1
    assert b2.dataset_genes(d2) == ["h0", "h1", "h2", "h3"]


def test_closed_bank_raises_runtime_error() -> None:
    bank = ScDataBank()
    assert not bank.is_closed
    bank.close()
    assert bank.is_closed
    assert "closed" in repr(bank)
    with pytest.raises(RuntimeError, match="closed"):
        bank.dataset_num_cells(object())  # type: ignore[arg-type]


def _raw_sparse_store(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    """A sparse store whose raw.X has a different gene count than X."""
    x_matrix = sp.csr_matrix(
        np.array([[1, 0, 2, 0, 0], [0, 3, 0, 0, 4], [5, 0, 0, 6, 0]], dtype=np.float32)
    )
    raw_matrix = sp.csr_matrix(
        np.array([[7, 0, 8], [0, 9, 0], [1, 0, 2]], dtype=np.float32)
    )
    adata = ad.AnnData(
        X=x_matrix,
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3", "g4"]),
    )
    adata.raw = ad.AnnData(X=raw_matrix, var=pd.DataFrame(index=["r0", "r1", "r2"]))
    root = write_zarr(
        adata,
        tmp_path / "raw.zarr",
        format="sparse",
        chunk_size=(2,),
        align_cells=True,
        store="dir",
    )
    return root, x_matrix.toarray(), raw_matrix.toarray()


def test_launch_all_parses_raw_with_own_gene_space(tmp_path: Path) -> None:
    root, x_expected, raw_expected = _raw_sparse_store(tmp_path)

    dc = launch_all(root)
    assert isinstance(dc, DatasetCollection)
    assert dc.raw is not None
    assert isinstance(dc.raw, SparseDataset)
    # raw shares num_cells with X but has its own gene space.
    assert dc.raw.num_cells == dc.x.num_cells
    assert dc.raw.num_genes != dc.x.num_genes
    assert dc.raw.num_genes == 3
    assert tuple(dc.raw.gene_names) == ("r0", "r1", "r2")
    # cell-aligned CSR layout, just like X.
    assert dc.raw.indices.variable_chunks
    # Mapping interface exposes raw under the "raw/X" key.
    assert "raw/X" in dc
    assert "X" in dc
    assert dc["raw/X"] is dc.raw
    assert dc.keys() == ("X", "raw/X")
    assert len(dc) == 2

    bank = ScDataBank()
    registered = bank.register_all(dc)
    assert set(registered) == {"X", "raw/X"}

    raw_did = registered["raw/X"]
    assert bank.dataset_num_cells(raw_did) == 3
    assert bank.dataset_num_genes(raw_did) == 3
    assert bank.dataset_genes(raw_did) == ["r0", "r1", "r2"]

    out = np.asarray(bank.load(raw_did, [0, 1, 2], dtype="f32")).reshape(3, 3)
    assert np.array_equal(out, raw_expected)

    # X still loads correctly alongside raw.
    x_did = registered["X"]
    x_out = np.asarray(bank.load(x_did, [0, 1, 2], dtype="f32")).reshape(3, 5)
    assert np.array_equal(x_out, x_expected)


def test_launch_single_matrix_raw_key(tmp_path: Path) -> None:
    root, _, raw_expected = _raw_sparse_store(tmp_path)

    raw_ds = launch(root, matrix="raw/X")
    assert isinstance(raw_ds, SparseDataset)
    assert raw_ds.num_genes == 3
    assert tuple(raw_ds.gene_names) == ("r0", "r1", "r2")

    # "raw" shorthand resolves to the same matrix.
    short = launch(root, matrix="raw")
    assert short.num_genes == raw_ds.num_genes
    assert tuple(short.gene_names) == tuple(raw_ds.gene_names)

    bank = ScDataBank()
    did = bank.register(raw_ds)
    out = np.asarray(bank.load(did, [0, 1, 2], dtype="f32")).reshape(3, 3)
    assert np.array_equal(out, raw_expected)


def test_launch_all_without_raw(tmp_path: Path) -> None:
    root, _, _, _ = _dense_store(tmp_path, "noraw", (3, 4), np.float32, ["g0", "g1", "g2", "g3"])
    dc = launch_all(root)
    assert dc.raw is None
    assert "raw/X" not in dc
    assert dc.keys() == ("X",)
    assert len(dc) == 1
    with pytest.raises(KeyError):
        dc["raw/X"]
