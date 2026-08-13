"""On-disk CSR SCC matrix."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from scdata import _core
from scdata.compress._base import MatrixStore, csr_from_raw, dense_from_raw, scalar_axes
from scdata.compress._index import normalize_axis, normalize_key



class ScCsr(MatrixStore):
    """Read-only CSR matrix backed by an SCC store.

    Indexing and :meth:`to_memory` decode into a SciPy CSR matrix.
    """

    __slots__ = ()

    format = "csr"

    @property
    def indptr(self) -> NDArray[np.uint64]:
        values = _core.store_indptr(self._require_open())
        if values is None:
            raise TypeError("indptr is only available for CSR stores")
        return np.asarray(values)

    def __getitem__(self, key: Any) -> Any:
        rows, cols, _ = normalize_key(key, self.n_rows, self.n_cols)
        row_scalar, col_scalar = scalar_axes(key)
        if row_scalar and col_scalar:
            dense = dense_from_raw(self._select_raw(rows, cols, csr_output="dense"))
            return dense[0, 0]
        return csr_from_raw(self._select_raw(rows, cols, csr_output="sparse"))

    def select(
        self,
        rows: Any = slice(None),
        cols: Any = slice(None),
        *,
        csr_output: Literal["sparse", "dense"] = "sparse",
    ) -> Any:
        if csr_output not in ("sparse", "dense"):
            raise ValueError(f"csr_output must be 'sparse' or 'dense', got {csr_output!r}")
        row_spec = normalize_axis(rows, self.n_rows, name="row")
        col_spec = normalize_axis(cols, self.n_cols, name="col")
        raw = self._select_raw(row_spec, col_spec, csr_output=csr_output)
        return dense_from_raw(raw) if csr_output == "dense" else csr_from_raw(raw)

    def to_memory(self) -> Any:
        return self.select()

    def to_scipy(self) -> Any:
        return self.to_memory()

    def tocsr(self, *, copy: bool = False) -> Any:
        matrix = self.to_memory()
        return matrix.copy() if copy else matrix

    def to_numpy(self) -> NDArray[Any]:
        return self.select(csr_output="dense")

    def toarray(self) -> NDArray[Any]:
        return self.to_numpy()

    def todense(self) -> np.matrix[Any, Any]:
        return np.asmatrix(self.to_numpy())
