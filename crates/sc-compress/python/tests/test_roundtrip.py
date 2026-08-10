from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import sc_compress as scc
from sc_compress._validate import csr_matrix_from_decoded


def test_public_storage_dtype_constants_are_immutable() -> None:
    assert isinstance(scc.STORAGE_VALUE_DTYPES, tuple)
    assert isinstance(scc.STORAGE_INDEX_DTYPES, tuple)


def test_dense_roundtrip_and_row_range(tmp_path: Path) -> None:
    root = tmp_path / "dense"
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    scc.write_dense(root, values, options=scc.WriteOptions(chunk_cells=2))

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
    with pytest.raises(scc.InvalidArgumentError, match="closed"):
        store.read()


def test_dense_rejects_bad_dtype_and_shape(tmp_path: Path) -> None:
    with pytest.raises(scc.InvalidArgumentError):
        scc.write_dense(tmp_path / "bad", np.arange(3, dtype=np.float32))
    with pytest.raises(scc.InvalidArgumentError):
        scc.write_dense(tmp_path / "bad", np.arange(4, dtype=np.float16).reshape(2, 2))
    with pytest.raises(scc.InvalidArgumentError, match="cannot be converted"):
        scc.write_dense(tmp_path / "ragged", [[1], [2, 3]])

    masked = np.ma.array(
        np.arange(4, dtype=np.float32).reshape(2, 2),
        mask=[[False, True], [False, False]],
    )
    with pytest.raises(scc.InvalidArgumentError, match="contains masked values"):
        scc.write_dense(tmp_path / "masked", masked)


@pytest.mark.parametrize("shape", [(3, 0), (0, 3), (0, 0)])
def test_dense_zero_axes_roundtrip(tmp_path: Path, shape: tuple[int, int]) -> None:
    values = np.empty(shape, dtype=np.float64)
    root = tmp_path / f"zero-{shape[0]}-{shape[1]}"
    scc.write(root, values)
    result = scc.open_store(root).read()
    assert result.shape == shape
    assert result.dtype == values.dtype


def test_dense_normalizes_non_native_byte_order(tmp_path: Path) -> None:
    values = np.arange(6, dtype=">f4").reshape(2, 3)
    root = tmp_path / "big-endian"
    scc.write(root, values)
    result = scc.open_store(root).read()
    assert result.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(result, values)


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
    scc.write_csr(root, csr, options=scc.WriteOptions(chunk_cells=1, block_cells=1))

    store = scc.open_store(root)
    assert store.kind == "csr"
    assert store.nnz == 3
    assert store.storage_index_dtype is not None
    assert store.index_dtype is not None
    full = store.read()
    assert isinstance(full, scc.ScCsr)
    np.testing.assert_array_equal(full.toarray(), np.array([[0, 0, 20], [0, 10, 0]]))

    batch = store.read_rows(1, 2)
    assert isinstance(batch, scc.ScCsr)
    assert batch.shape == (1, 3)
    np.testing.assert_array_equal(batch.toarray(), np.array([[0, 10, 0]]))

    # 2-D gene subset densify path.
    dense_genes = store.select([1, 0], [2, 0], csr_output="dense")
    assert isinstance(dense_genes, scc.ScDense)
    np.testing.assert_array_equal(dense_genes.to_numpy(), np.array([[0, 0], [20, 0]]))


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
    assert isinstance(result, scc.ScCsr)
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
    with pytest.raises(scc.ScCompressError):
        store.read()


def test_read_limits_object_and_python_integer_validation(tmp_path: Path) -> None:
    root = tmp_path / "limits"
    scc.write_dense(root, np.ones((2, 2), dtype=np.uint16))
    limits = scc.ReadLimits(max_decoded_size=np.int64(64))
    store = scc.open_store(root, limits=limits, max_block_count=np.int64(10))
    assert store.limits.max_decoded_size == 64
    assert store.limits.max_block_count == 10

    with pytest.raises(scc.InvalidArgumentError, match="max_decoded_size"):
        scc.ReadLimits(max_decoded_size=True)
    with pytest.raises(scc.InvalidArgumentError, match="platform limit"):
        scc.ReadLimits(max_decoded_size=1 << 128)
    with pytest.raises(scc.InvalidArgumentError, match="chunk_cells"):
        scc.write_dense(
            tmp_path / "bad-partition",
            np.ones((1, 1), dtype=np.uint16),
            options=scc.WriteOptions(chunk_cells=True),  # type: ignore[arg-type]
        )
    sparse = pytest.importorskip("scipy.sparse")
    with pytest.raises(scc.InvalidArgumentError, match="chunk_budget is required"):
        scc.write_csr(
            tmp_path / "missing-budget",
            sparse.csr_matrix(np.ones((2, 2), dtype=np.float32)),
            options=scc.WriteOptions(chunk_policy="budget"),
        )
    with pytest.raises(scc.InvalidArgumentError, match="dense writes require"):
        scc.write_dense(
            tmp_path / "dense-budget",
            np.ones((2, 2), dtype=np.float32),
            options=scc.WriteOptions(chunk_policy="budget", chunk_budget=64),
        )


def test_n_workers_is_configurable_for_writes_and_reads(tmp_path: Path) -> None:
    assert scc.DEFAULT_N_WORKERS >= 1
    options = scc.WriteOptions(n_workers=np.int64(2))
    assert options.n_workers == 2

    root = tmp_path / "workers"
    values = np.arange(32, dtype=np.float32).reshape(8, 4)
    scc.write_dense(
        root,
        values,
        options=options,
        n_workers=np.int64(3),
    )

    limits = scc.ReadLimits(n_workers=np.int64(2))
    with scc.open_store(root, limits=limits, n_workers=np.int64(4)) as store:
        assert store.n_workers == 4
        assert store.limits.n_workers == 4
        assert store.info().n_workers == 4
        np.testing.assert_array_equal(store.read(), values)

    with pytest.raises(scc.InvalidArgumentError, match="n_workers"):
        scc.ReadLimits(n_workers=0)
    with pytest.raises(scc.InvalidArgumentError, match="n_workers"):
        scc.WriteOptions(n_workers=False)  # type: ignore[arg-type]
    with pytest.raises(scc.InvalidArgumentError, match="n_workers"):
        scc.open_store(root, n_workers=0)


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
    with pytest.raises(scc.InvalidArgumentError, match=r"shape\[0\]"):
        scc.write_csr_arrays(
            tmp_path / "bad-shape",
            np.array([0], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
            (0.5, 1),
        )


def test_overwrite_false_refuses_existing_path(tmp_path: Path) -> None:
    root = tmp_path / "exists"
    values = np.ones((2, 2), dtype=np.float32)
    scc.write_dense(root, values)
    with pytest.raises(scc.InvalidArgumentError, match="already exists"):
        scc.write_dense(root, values, overwrite=False)


def test_write_rejects_output_path_without_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(scc.InvalidArgumentError, match="must name a file or directory"):
        scc.write_dense(".", np.ones((1, 1), dtype=np.float32))


def test_decoded_csr_rejects_shape_outside_scipy_index_range() -> None:
    pytest.importorskip("scipy.sparse")
    with pytest.raises(scc.InvalidArgumentError, match="shape .* signed int64"):
        csr_matrix_from_decoded(
            np.array([], dtype=np.uint64),
            np.array([], dtype=np.float32),
            np.array([0, 0], dtype=np.uint64),
            n_rows=1,
            n_cols=1 << 63,
        )
