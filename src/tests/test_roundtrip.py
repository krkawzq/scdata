from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scdata.compress as scc
from scdata.compress._validate import csr_matrix_from_decoded


def test_public_storage_dtype_constants_are_immutable() -> None:
    assert isinstance(scc.STORAGE_VALUE_DTYPES, tuple)
    assert isinstance(scc.STORAGE_INDEX_DTYPES, tuple)
    assert {"i64", "u64"}.issubset(scc.STORAGE_VALUE_DTYPES)
    assert {np.dtype(np.int64), np.dtype(np.uint64)}.issubset(scc.VALUE_DTYPES)


def test_dense_roundtrip_and_row_range(tmp_path: Path) -> None:
    root = tmp_path / "dense"
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    scc.write_dense(
        root,
        values,
        options=scc.WriteOptions(
            chunk_policy="cells",
            chunk_cells=2,
            block_policy="cells",
            block_cells=1,
        ),
    )

    with scc.open_store(root) as store:
        assert store.kind == "dense"
        assert store.shape == (6, 4)
        assert len(store) == 6
        assert store.dtype == np.dtype(np.float32)
        assert store.storage_dtype == "f32"
        assert store.index_dtype is None
        assert store.storage_index_dtype is None
        assert "shape=(6, 4)" in repr(store)
        assert store.info().path == root
        assert store.info().limits == scc.DEFAULT_READ_LIMITS
        assert store.info().dtype == store.dtype
        assert store.info().index_dtype is None
        np.testing.assert_array_equal(store.read(), values)
        np.testing.assert_array_equal(store.read_rows(2, 5), values[2:5])
        np.testing.assert_array_equal(store[-1], values[-1])
        np.testing.assert_array_equal(store[1:6:2], values[1:6:2])
        np.testing.assert_array_equal(store[::-2], values[::-2])
        np.testing.assert_array_equal(np.stack(list(store.iter_rows(batch_size=2))), values)
        assert [batch.shape for batch in store.iter_batches(batch_size=4)] == [(4, 4), (2, 4)]

    assert store.closed
    assert "closed=True" in repr(store)
    with pytest.raises(ValueError, match="closed"):
        store.read()


def test_dense_rejects_bad_dtype_and_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        scc.write_dense(tmp_path / "bad", np.arange(3, dtype=np.float32))
    with pytest.raises(ValueError):
        scc.write_dense(tmp_path / "bad", np.arange(4, dtype=np.float16).reshape(2, 2))
    with pytest.raises(ValueError, match="cannot be converted"):
        scc.write_dense(tmp_path / "ragged", [[1], [2, 3]])

    masked = np.ma.array(
        np.arange(4, dtype=np.float32).reshape(2, 2),
        mask=[[False, True], [False, False]],
    )
    with pytest.raises(ValueError, match="contains masked values"):
        scc.write_dense(tmp_path / "masked", masked)


@pytest.mark.parametrize("shape", [(3, 0), (0, 3), (0, 0)])
def test_dense_zero_axes_roundtrip(tmp_path: Path, shape: tuple[int, int]) -> None:
    values = np.empty(shape, dtype=np.float64)
    root = tmp_path / f"zero-{shape[0]}-{shape[1]}"
    scc.write(root, values)
    result = scc.open_store(root).read()
    assert result.shape == shape
    assert result.dtype == values.dtype


@pytest.mark.parametrize("dtype", [">f4", ">i8", ">u8"])
def test_dense_normalizes_non_native_byte_order(tmp_path: Path, dtype: str) -> None:
    raw_values = {
        ">f4": [0, 1, 2, 3, 4, 5],
        ">i8": [
            np.iinfo(np.int64).min,
            -(1 << 53) - 1,
            -1,
            0,
            (1 << 53) + 1,
            np.iinfo(np.int64).max,
        ],
        ">u8": [0, 1, (1 << 53) + 1, 1 << 63, (1 << 63) + 1, np.iinfo(np.uint64).max],
    }
    values = np.asarray(raw_values[dtype], dtype=dtype).reshape(2, 3)
    root = tmp_path / f"big-endian-{dtype[-2:]}"
    scc.write(root, values)
    result = scc.open_store(root).read()
    assert result.dtype == values.dtype.newbyteorder("=")
    np.testing.assert_array_equal(result, values)


def test_dense_and_csr_preserve_int64_and_uint64_values(tmp_path: Path) -> None:
    sparse = pytest.importorskip("scipy.sparse")
    matrices = {
        "i64": np.array(
            [
                [np.iinfo(np.int64).min, -(1 << 53) - 1],
                [(1 << 53) + 1, np.iinfo(np.int64).max],
            ],
            dtype=np.int64,
        ),
        "u64": np.array(
            [
                [0, (1 << 53) + 1],
                [(1 << 63) + 1, np.iinfo(np.uint64).max],
            ],
            dtype=np.uint64,
        ),
    }

    for storage_dtype, values in matrices.items():
        dense_root = tmp_path / f"dense-{storage_dtype}"
        scc.write_dense(dense_root, values)
        dense = scc.open_store(dense_root)
        assert dense.storage_dtype == storage_dtype
        assert dense.dtype == values.dtype
        np.testing.assert_array_equal(dense.read(), values)
        np.testing.assert_array_equal(
            dense.select([1, 0], [1, 0]),
            values[[1, 0]][:, [1, 0]],
        )

        csr_root = tmp_path / f"csr-{storage_dtype}"
        scc.write_csr(csr_root, sparse.csr_matrix(values))
        csr = scc.open_store(csr_root)
        assert csr.storage_dtype == storage_dtype
        assert csr.dtype == values.dtype
        np.testing.assert_array_equal(csr.read().toarray(), values)
        np.testing.assert_array_equal(
            csr.select([1, 0], [1, 0], csr_output="dense"),
            values[[1, 0]][:, [1, 0]],
        )


def test_csr_roundtrip_via_scipy(tmp_path: Path) -> None:
    sparse = pytest.importorskip("scipy.sparse")
    root = tmp_path / "csr"
    csr = sparse.csr_matrix(
        (
            np.array([20, 0, 10], dtype=np.int32),
            np.array([2, 0, 1], dtype=np.int32),
            np.array([0, 2, 3], dtype=np.int64),
        ),
        shape=(2, 3),
    )
    # Writer canonicalizes row order.
    scc.write_csr(
        root,
        csr,
        options=scc.WriteOptions(
            chunk_policy="cells",
            chunk_cells=1,
            block_policy="cells",
            block_cells=1,
        ),
    )

    store = scc.open_store(root)
    assert store.kind == "csr"
    assert isinstance(store, scc.ScCsr)
    assert store.nnz == 3
    assert store.storage_index_dtype is not None
    assert store.index_dtype is not None
    full = store.read()
    assert sparse.issparse(full)
    np.testing.assert_array_equal(full.toarray(), np.array([[0, 0, 20], [0, 10, 0]]))

    batch = store.read_rows(1, 2)
    assert sparse.issparse(batch)
    assert batch.shape == (1, 3)
    np.testing.assert_array_equal(batch.toarray(), np.array([[0, 10, 0]]))

    dense_genes = store.select([1, 0], [2, 0], csr_output="dense")
    np.testing.assert_array_equal(dense_genes, np.array([[0, 0], [20, 0]]))


def test_generic_write_canonicalizes_duplicate_csr_entries(tmp_path: Path) -> None:
    sparse = pytest.importorskip("scipy.sparse")
    csr = sparse.csr_matrix(
        (
            np.array([1.5, 2.5], dtype=np.float32),
            np.array([1, 1], dtype=np.int32),
            np.array([0, 2], dtype=np.int32),
        ),
        shape=(1, 3),
    )
    assert not csr.has_canonical_format

    root = tmp_path / "duplicates"
    scc.write(root, csr)
    result = scc.open_store(root)[0]
    assert sparse.issparse(result)
    np.testing.assert_array_equal(result.toarray(), np.array([[0.0, 4.0, 0.0]]))


def test_write_csr_arrays_numpy_buffers(tmp_path: Path) -> None:
    pytest.importorskip("scipy.sparse")
    root = tmp_path / "csr-arrays"
    scc.write_csr_arrays(
        root,
        np.array([0, 1, 2], dtype=np.int64),
        np.array([0, 1], dtype=np.int32),
        np.array([1.0, 2.0], dtype="float32"),
        (2, 2),
    )
    store = scc.open_store(root)
    out = store.read()
    np.testing.assert_array_equal(
        out.toarray(), np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    )


def test_read_limits_reject_tiny_decoded(tmp_path: Path) -> None:
    root = tmp_path / "limited"
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    scc.write_dense(root, values)
    store = scc.open_store(root, max_decoded_size=8)
    assert store.limits.max_decoded_size == 8
    with pytest.raises(scc.Error):
        store.read()


def test_read_limits_object_and_python_integer_validation(tmp_path: Path) -> None:
    root = tmp_path / "limits"
    scc.write_dense(root, np.ones((2, 2), dtype=np.uint16))
    limits = scc.ReadLimits(max_decoded_size=np.int64(64))
    store = scc.open_store(root, limits=limits, max_block_count=np.int64(10))
    assert store.limits.max_decoded_size == 64
    assert store.limits.max_block_count == 10

    with pytest.raises(TypeError, match="max_decoded_size"):
        scc.ReadLimits(max_decoded_size=True)
    with pytest.raises(ValueError, match="platform limit"):
        scc.ReadLimits(max_decoded_size=1 << 128)
    with pytest.raises(TypeError, match="chunk_cells"):
        scc.write_dense(
            tmp_path / "bad-partition",
            np.ones((1, 1), dtype=np.uint16),
            options=scc.WriteOptions(
                chunk_policy="cells",
                chunk_cells=True,  # type: ignore[arg-type]
                block_policy="cells",
                block_cells=1,
            ),
        )
    sparse = pytest.importorskip("scipy.sparse")
    with pytest.raises(ValueError, match="chunk_budget is required"):
        scc.write_csr(
            tmp_path / "missing-budget",
            sparse.csr_matrix(np.ones((2, 2), dtype=np.float32)),
            options=scc.WriteOptions(
                chunk_policy="budget",
                chunk_budget=None,
                block_policy="cells",
                block_cells=1,
            ),
        )


def test_num_workers_is_configurable_for_writes_and_reads(tmp_path: Path) -> None:
    options = scc.WriteOptions(num_workers=np.int64(2))
    assert options.num_workers == 2

    root = tmp_path / "workers"
    values = np.arange(32, dtype=np.float32).reshape(8, 4)
    scc.write_dense(
        root,
        values,
        options=options,
        num_workers=np.int64(3),
    )

    limits = scc.ReadLimits(num_workers=np.int64(2))
    with scc.open_store(root, limits=limits, num_workers=np.int64(4)) as store:
        assert store.num_workers == 4
        assert store.limits.num_workers == 4
        assert store.info().num_workers == 4
        np.testing.assert_array_equal(store.read(), values)

    with pytest.raises(ValueError, match="num_workers"):
        scc.ReadLimits(num_workers=0)
    with pytest.raises(TypeError, match="num_workers"):
        scc.WriteOptions(num_workers=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_workers"):
        scc.open_store(root, num_workers=0)


def test_csr_bytes_budget_policy_roundtrip(tmp_path: Path) -> None:
    sparse = pytest.importorskip("scipy.sparse")
    root = tmp_path / "csr-budget"
    csr = sparse.random(6, 8, density=0.4, format="csr", dtype=np.float32)
    scc.write_csr(
        root,
        csr,
        options=scc.WriteOptions(
            chunk_policy="budget",
            chunk_budget=64,
            block_policy="budget",
            block_budget=32,
        ),
    )
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    assert meta["partition"]["chunk"] == {"strategy": "bytes_budget", "n": 64}
    assert meta["partition"]["block"] == {"strategy": "bytes_budget", "n": 32}
    out = scc.open_store(root).read()
    np.testing.assert_array_equal(out.toarray(), csr.toarray())
    with pytest.raises(TypeError, match=r"shape\[0\]"):
        scc.write_csr_arrays(
            tmp_path / "bad-shape",
            np.array([0], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
            (0.5, 1),
        )


def test_dense_bytes_budget_lowers_to_fixed_cells(tmp_path: Path) -> None:
    root = tmp_path / "dense-budget"
    # row_bytes = 4 cols * 4 bytes = 16; budgets 64 / 32 → ceil → 4 / 2 cells.
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    scc.write_dense(
        root,
        values,
        options=scc.WriteOptions(
            chunk_policy="budget",
            chunk_budget=64,
            block_policy="budget",
            block_budget=32,
        ),
    )
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    assert meta["partition"]["chunk"] == {"strategy": "fixed_cells", "n": 4}
    assert meta["partition"]["block"] == {"strategy": "fixed_cells", "n": 2}
    np.testing.assert_array_equal(scc.open_store(root).read(), values)


def test_default_write_options_use_byte_budgets() -> None:
    assert scc.DEFAULT_CHUNK_BUDGET == 100 << 20
    assert scc.DEFAULT_BLOCK_BUDGET == 64 << 10
    opts = scc.DEFAULT_WRITE_OPTIONS
    assert opts.chunk_policy == "budget"
    assert opts.block_policy == "budget"
    assert opts.chunk_budget == scc.DEFAULT_CHUNK_BUDGET
    assert opts.block_budget == scc.DEFAULT_BLOCK_BUDGET
    chunk, block = opts.resolve(dense=False)
    assert (chunk.policy, chunk.n) == ("budget", scc.DEFAULT_CHUNK_BUDGET)
    assert (block.policy, block.n) == ("budget", scc.DEFAULT_BLOCK_BUDGET)
    dense_chunk, dense_block = opts.resolve(dense=True, row_bytes=16)
    assert dense_chunk.policy == "cells"
    assert dense_block.policy == "cells"
    assert dense_chunk.n * 16 >= scc.DEFAULT_CHUNK_BUDGET
    assert dense_block.n * 16 >= scc.DEFAULT_BLOCK_BUDGET
    assert (dense_chunk.n - 1) * 16 < scc.DEFAULT_CHUNK_BUDGET
    assert (dense_block.n - 1) * 16 < scc.DEFAULT_BLOCK_BUDGET


def test_overwrite_false_refuses_existing_path(tmp_path: Path) -> None:
    root = tmp_path / "exists"
    values = np.ones((2, 2), dtype=np.float32)
    scc.write_dense(root, values)
    with pytest.raises(FileExistsError, match="already exists"):
        scc.write_dense(root, values, overwrite=False)


def test_write_rejects_output_path_without_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="must name a file or directory"):
        scc.write_dense(".", np.ones((1, 1), dtype=np.float32))


def test_decoded_csr_rejects_shape_outside_scipy_index_range() -> None:
    pytest.importorskip("scipy.sparse")
    with pytest.raises(ValueError, match="shape .* signed int64"):
        csr_matrix_from_decoded(
            np.array([], dtype=np.uint64),
            np.array([], dtype=np.float32),
            np.array([0, 0], dtype=np.uint64),
            n_rows=1,
            n_cols=1 << 63,
        )
