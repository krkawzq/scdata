"""On-disk dense SCC matrix."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from scdata.compress._base import MatrixStore, dense_from_raw, scalar_axes
from scdata.compress._index import normalize_axis, normalize_key



class ScDense(MatrixStore):
    """Read-only dense matrix backed by an SCC store.

    Indexing and :meth:`to_memory` decode into a NumPy ndarray.
    """

    __slots__ = ()

    def __getitem__(self, key: Any) -> Any:
        rows, cols, _ = normalize_key(key, self.n_rows, self.n_cols)
        array = dense_from_raw(self._select_raw(rows, cols, csr_output="dense"))
        row_scalar, col_scalar = scalar_axes(key)
        if row_scalar and col_scalar:
            return array[0, 0]
        if row_scalar:
            return array[0]
        if col_scalar:
            return array[:, 0]
        return array

    def select(
        self,
        rows: Any = slice(None),
        cols: Any = slice(None),
        *,
        csr_output: Literal["sparse", "dense"] = "dense",
    ) -> NDArray[Any]:
        if csr_output not in ("sparse", "dense"):
            raise ValueError(f"csr_output must be 'sparse' or 'dense', got {csr_output!r}")
        row_spec = normalize_axis(rows, self.n_rows, name="row")
        col_spec = normalize_axis(cols, self.n_cols, name="col")
        return dense_from_raw(self._select_raw(row_spec, col_spec, csr_output="dense"))

    def to_memory(self) -> NDArray[Any]:
        return self.select()

    def to_numpy(self, *, copy: bool = False) -> NDArray[Any]:
        array = self.to_memory()
        return array.copy() if copy else array

    def __array__(
        self,
        dtype: np.dtype[Any] | None = None,
        copy: bool | None = None,
    ) -> NDArray[Any]:
        array = self.to_memory()
        if dtype is not None:
            array = np.asarray(array, dtype=dtype)
        if copy:
            array = array.copy()
        return array
