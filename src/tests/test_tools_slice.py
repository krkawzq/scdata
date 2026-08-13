from __future__ import annotations

import numpy as np
import pytest

from scdata.tools import slice as sc_slice


def test_slice_dense_fancy_and_scalar() -> None:
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    wrapped = sc_slice(values)
    np.testing.assert_array_equal(wrapped[[2, 0], 1:3], values[[2, 0], 1:3])
    assert wrapped[1, 2] == values[1, 2]


def test_slice_csr_returns_scipy() -> None:
    sparse = pytest.importorskip("scipy.sparse")
    matrix = sparse.csr_matrix(np.array([[1, 0, 2], [0, 3, 0]], dtype=np.float32))
    wrapped = sc_slice(matrix)
    out = wrapped[1, :]
    assert sparse.issparse(out)
    np.testing.assert_array_equal(out.toarray(), np.array([[0, 3, 0]], dtype=np.float32))
    assert wrapped[0, 2] == 2.0


def test_slice_rejects_non_csr_and_non_2d() -> None:
    sparse = pytest.importorskip("scipy.sparse")
    with pytest.raises(Exception, match="2-D"):
        sc_slice(np.arange(4, dtype=np.float32))
    with pytest.raises(Exception, match="CSR"):
        sc_slice(sparse.csc_matrix(np.eye(2, dtype=np.float32)))
    with pytest.raises(Exception, match="num_workers"):
        sc_slice(np.eye(2, dtype=np.float32), num_workers=0)
