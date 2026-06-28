"""End-to-end tests for the Pythonic ``ScDataBank`` wrapper.

These build directory stores with the ``conftest`` factories, wrap them into
:class:`scdata.data._dataset.DenseDataset` / ``SparseDataset`` by hand (so the
tests do not depend on the io-layer ``read`` entry point), register them into
an :class:`scdata.ScDataBank`, and cross-check decoded cell data against the
ground-truth arrays the factories wrote.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scdata import DataBankConfig, IoConfig, MissingGenePolicy, ScDataBank
from scdata._scdata import DataBankError
from scdata.data._dataset import (
    ArrayMeta,
    ArrayOrder,
    CodecPipeline,
    DenseDataset,
    DType,
)
from scdata.io import launch


# ---------------------------------------------------------------------------
# helpers: turn a conftest-built store into a Dataset object
# ---------------------------------------------------------------------------


def _array_meta(root: Path, key: str) -> ArrayMeta:
    zarray = json.loads((root / key / ".zarray").read_text())
    sc = json.loads((root / key / ".zattrs").read_text())["scdata"]
    offsets = np.array(sc["offsets"], dtype=np.uint64)
    lengths = np.array(sc["lengths"], dtype=np.uint64)
    dtype = DType.parse(zarray["dtype"])
    codec = CodecPipeline.from_zarr(zarray.get("filters"), zarray.get("compressor"))
    return ArrayMeta(
        shape=tuple(zarray["shape"]),
        chunk_shape=tuple(zarray["chunks"]),
        dtype=dtype,
        order=ArrayOrder.C,
        codec=codec,
        payload_path=f"{key}/payload.bin",
        chunk_offsets=offsets,
        chunk_lengths=lengths,
    )


def _dense_dataset(root: Path, gene_names: list[str]) -> DenseDataset:
    meta = _array_meta(root, "X")
    if meta.ndim == 2:
        num_cells, num_genes = meta.shape
    else:
        num_genes = len(gene_names)
        num_cells = meta.shape[0] // num_genes
    return DenseDataset(
        gene_names=tuple(gene_names), data=meta, num_cells=num_cells, num_genes=num_genes
    )


def _expected_dense(shape, np_dtype) -> np.ndarray:
    return np.arange(int(np.prod(shape)), dtype=np_dtype).reshape(shape)


# ---------------------------------------------------------------------------
# registration + queries
# ---------------------------------------------------------------------------


def test_register_dense_2d(dense_store_factory) -> None:
    root = dense_store_factory("dense", (3, 4), (2, 2), np.float32, None, ["g0", "g1", "g2", "g3"])
    ds = _dense_dataset(root, ["g0", "g1", "g2", "g3"])
    bank = ScDataBank()
    did = bank.register_dense(ds, str(root))
    assert bank.dataset_num_cells(did) == 3
    assert bank.dataset_num_genes(did) == 4
    assert bank.dataset_dtype(did) == DType.F32
    assert bank.dataset_genes(did) == ["g0", "g1", "g2", "g3"]
    bank.unregister(did)
    with pytest.raises(DataBankError):
        bank.dataset_genes(did)


def test_dataset_id_identity(dense_store_factory) -> None:
    root = dense_store_factory("id", (3, 4), (2, 2), np.float32, None, ["g0", "g1", "g2", "g3"])
    ds = _dense_dataset(root, ["g0", "g1", "g2", "g3"])
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


# ---------------------------------------------------------------------------
# cell access values
# ---------------------------------------------------------------------------


def test_access_dense_values(dense_store_factory) -> None:
    root = dense_store_factory("acc", (5, 6), (2, 3), np.float32, None, [f"g{i}" for i in range(6)])
    ds = _dense_dataset(root, [f"g{i}" for i in range(6)])
    bank = ScDataBank()
    did = bank.register_dense(ds, root)
    expected = _expected_dense((5, 6), np.float32)
    result = bank.load(did, [0, 1, 2, 3, 4], dtype="f32")
    out = np.asarray(result)
    assert out.shape == (30,)
    assert result.shape == (5, 6)
    assert result.var_names == tuple(f"g{i}" for i in range(6))
    assert np.shares_memory(result.to_numpy(), result.data)
    assert np.shares_memory(result.to_flat_numpy(), result.data)
    assert np.array_equal(out.reshape(5, 6), expected)
    # subset + reordered
    out2 = np.asarray(bank.load(did, [3, 0, 4]))
    assert np.array_equal(out2.reshape(3, 6), expected[[3, 0, 4]])
    bank.unregister(did)


def test_load_and_prefetch_accept_numpy_inputs(dense_store_factory) -> None:
    root = dense_store_factory(
        "public_fast",
        (5, 6),
        (2, 3),
        np.float32,
        None,
        [f"g{i}" for i in range(6)],
    )
    ds = _dense_dataset(root, [f"g{i}" for i in range(6)])
    bank = ScDataBank()
    did = bank.register_dense(ds, str(root))
    expected = _expected_dense((5, 6), np.float32)

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


@pytest.mark.parametrize("codec_name", ["zstd", "blosc", "gzip"])
def test_access_dense_codec(codec_configs, dense_store_factory, codec_name) -> None:
    root = dense_store_factory(
        f"acc_{codec_name}",
        (5, 6),
        (2, 3),
        np.float64,
        codec_configs[codec_name],
        [f"g{i}" for i in range(6)],
    )
    ds = _dense_dataset(root, [f"g{i}" for i in range(6)])
    bank = ScDataBank()
    did = bank.register_dense(ds, str(root))
    expected = _expected_dense((5, 6), np.float64)
    out = np.asarray(bank.load(did, list(range(5))))
    assert np.array_equal(out.reshape(5, 6), expected)
    bank.unregister(did)


def test_register_launched_zip_dense(dense_store_factory, zip_store_factory) -> None:
    root = dense_store_factory(
        "zip_bank_src",
        (5, 6),
        (2, 3),
        np.float32,
        None,
        [f"g{i}" for i in range(6)],
    )
    zip_path = zip_store_factory(root, "zip_bank.zarr.zip")
    ds = launch(zip_path)

    bank = ScDataBank()
    did = bank.register(ds)
    expected = _expected_dense((5, 6), np.float32)
    out = np.asarray(bank.load(did, [4, 0, 2]))

    assert np.array_equal(out.reshape(3, 6), expected[[4, 0, 2]])
    bank.unregister(did)


def test_access_dtype_round_trip(dense_store_factory) -> None:
    for np_dt in (np.int32, np.int64, np.uint8, np.float32, np.float64):
        root = dense_store_factory(
            f"dt_{np_dt.__name__}", (3, 4), (2, 2), np_dt, None, ["g0", "g1", "g2", "g3"]
        )
        ds = _dense_dataset(root, ["g0", "g1", "g2", "g3"])
        bank = ScDataBank()
        did = bank.register_dense(ds, str(root))
        out = bank.load(did, [0, 1, 2])
        arr = np.asarray(out)
        assert arr.dtype == np.dtype(np_dt)
        assert np.array_equal(arr.reshape(3, 4), _expected_dense((3, 4), np_dt))
        bank.unregister(did)


def test_load_by_genes(dense_store_factory) -> None:
    root = dense_store_factory("gn", (5, 6), (2, 3), np.float32, None, [f"g{i}" for i in range(6)])
    ds = _dense_dataset(root, [f"g{i}" for i in range(6)])
    bank = ScDataBank()
    did = bank.register_dense(ds, str(root))
    expected = _expected_dense((5, 6), np.float32)
    out = np.asarray(bank.load(did, [0, 2], genes=["g1", "g3"]))
    assert out.shape == (4,)
    assert np.array_equal(out.reshape(2, 2), expected[[0, 2]][:, [1, 3]])
    bank.unregister(did)


def test_load_by_genes_missing_error(dense_store_factory) -> None:
    root = dense_store_factory("gn_err", (3, 4), (2, 2), np.float32, None, ["g0", "g1", "g2", "g3"])
    ds = _dense_dataset(root, ["g0", "g1", "g2", "g3"])
    bank = ScDataBank()
    did = bank.register_dense(ds, str(root))
    with pytest.raises(DataBankError):
        bank.load(did, [0], genes=["nope"], missing=MissingGenePolicy.ERROR)
    with pytest.raises(DataBankError):
        bank.load(did, [0], genes=["nope"], missing="error")
    bank.unregister(did)


def test_access_unregistered_raises(dense_store_factory) -> None:
    root = dense_store_factory("unreg", (3, 4), (2, 2), np.float32, None, ["g0", "g1", "g2", "g3"])
    ds = _dense_dataset(root, ["g0", "g1", "g2", "g3"])
    bank = ScDataBank()
    did = bank.register_dense(ds, str(root))
    bank.unregister(did)
    with pytest.raises(DataBankError):
        bank.load(did, [0])


# ---------------------------------------------------------------------------
# prefetch iterator
# ---------------------------------------------------------------------------


def test_prefetch(dense_store_factory) -> None:
    root = dense_store_factory("pf", (5, 6), (2, 3), np.float32, None, [f"g{i}" for i in range(6)])
    ds = _dense_dataset(root, [f"g{i}" for i in range(6)])
    bank = ScDataBank()
    did = bank.register_dense(ds, str(root))
    expected = _expected_dense((5, 6), np.float32)
    batches = [[0, 1], [2, 3, 4]]
    pf = bank.prefetch(did, batches)
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


def test_prefetch_by_genes(dense_store_factory) -> None:
    root = dense_store_factory(
        "pfgn", (5, 6), (2, 3), np.float32, None, [f"g{i}" for i in range(6)]
    )
    ds = _dense_dataset(root, [f"g{i}" for i in range(6)])
    bank = ScDataBank()
    did = bank.register_dense(ds, str(root))
    expected = _expected_dense((5, 6), np.float32)
    pf = bank.prefetch(did, [[0, 1], [2]], genes=["g0", "g5"])
    seen = 0
    for batch in pf:
        cells = np.asarray(batch.cells)
        data = np.asarray(batch.data)
        assert batch.num_genes == 2
        assert batch.var_names == ("g0", "g5")
        rows = expected[cells.tolist()][:, [0, 5]]
        assert np.array_equal(data.reshape(len(cells), 2), rows)
        seen += len(cells)
    assert seen == 3
    bank.unregister(did)


# ---------------------------------------------------------------------------
# lifecycle: multiple banks
# ---------------------------------------------------------------------------


def test_multiple_banks(dense_store_factory) -> None:
    r1 = dense_store_factory("b1", (3, 4), (2, 2), np.float32, None, ["g0", "g1", "g2", "g3"])
    r2 = dense_store_factory("b2", (3, 4), (2, 2), np.float64, None, ["h0", "h1", "h2", "h3"])
    b1 = ScDataBank()
    b2 = ScDataBank()
    d1 = b1.register_dense(_dense_dataset(r1, ["g0", "g1", "g2", "g3"]), str(r1))
    d2 = b2.register_dense(_dense_dataset(r2, ["h0", "h1", "h2", "h3"]), str(r2))
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
