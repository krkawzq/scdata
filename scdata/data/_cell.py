"""Cell access request, result, and batch types for the DataBank.

These are the pure-Python data carriers for the access and streaming-prefetch
paths.  They live in the data layer — no dependency on the Rust extension —
so the bank's execution layer (:mod:`scdata.databank`) is the only place that
touches the Rust core.

Three value types cover the whole surface:

* :class:`CellAccess` — input for one ``access_cells`` /
  ``access_cells_by_gene_names`` call, and the per-batch input unit for
  scheduled prefetch: which cells, optionally which genes.
* :class:`CellData` — output of one access call: the decoded 1D array plus
  enough shape (and the matching gene names) to interpret it.  Implements
  ``__array__`` so ``np.asarray(result)`` works directly, in addition to the
  :attr:`matrix` zero-copy view.
* :class:`CellBatch` — the **output** batch type yielded by the prefetch
  iterator (``cells`` + ``data`` + ``num_genes``).  It is always decoded; the
  prefetch *input* side is a :class:`CellAccess`, not a half-filled batch.

Layout contract
---------------
``data`` is a 1D row-major numpy array of shape ``[num_cells * num_genes]``;
cell ``i``'s genes occupy ``data[i*num_genes : (i+1)*num_genes]``.  The
``matrix`` property reshapes it to ``[num_cells, num_genes]`` — zero-copy when
``data`` is contiguous, which it always is coming out of Rust.

These types deliberately do **not** carry execution parameters
(``missing`` / ``dtype`` / ``config``): those belong to the bank's execution
layer, not to the data description.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["CellAccess", "CellData", "CellBatch"]


def _as_cell_index(value: Any, name: str) -> NDArray[np.intp]:
    """Coerce cell indices into a contiguous 1D ``intp`` numpy array.

    Accepts a list, tuple, range, or numpy array of any integer dtype.  The
    result is C-contiguous ``intp`` (numpy's native index dtype) so it can be
    passed straight to Rust via ``tolist()`` without a further copy.  Negative
    indices are rejected — they are never valid cell positions.
    """
    arr = np.asarray(value, dtype=np.intp)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.ndim}D")
    if arr.size and int(arr.min()) < 0:
        raise ValueError(f"{name} values must be non-negative")
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


@dataclass(frozen=True)
class CellAccess:
    """Input for a single cell-access call (also the prefetch batch input unit).

    Attributes:
        cells: 1D cell indices into the registered dataset — any order,
            subset, or repeats are allowed.
        gene_names: Optional gene-name subset to project each cell onto.  When
            ``None`` (the default) every gene is returned in dataset column
            order; otherwise only the named genes are returned, in the
            requested order.  Duplicate names are passed through verbatim —
            the bank returns a duplicate column, matching Rust semantics.
    """

    cells: NDArray[np.intp]
    gene_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        cells = _as_cell_index(self.cells, "cells")
        object.__setattr__(self, "cells", cells)
        if self.gene_names is not None:
            object.__setattr__(self, "gene_names", tuple(self.gene_names))

    @classmethod
    def from_cells(
        cls,
        cells: Any,
        gene_names: Any | None = None,
    ) -> "CellAccess":
        """Build a :class:`CellAccess` from a cell iterable (+ optional genes)."""
        return cls(cells=cells, gene_names=gene_names)

    @property
    def num_cells(self) -> int:
        """Number of cells requested (length of :attr:`cells`)."""
        return int(self.cells.shape[0])

    @property
    def is_gene_subset(self) -> bool:
        """True when only a subset of genes was requested."""
        return self.gene_names is not None


@dataclass(frozen=True)
class CellData:
    """Result of a single cell-access call.

    Attributes:
        cells: 1D ``intp`` numpy array of the cell indices this result covers,
            in the same order as the request.
        data: 1D row-major numpy array of shape
            ``[num_cells * num_genes]``.  For ``f16`` data the dtype is
            ``float16``; for ``bf16`` it is ``uint16`` holding the raw bfloat16
            bit pattern (numpy has no native bfloat16).
        num_genes: Number of gene columns per cell.
        gene_names: The gene names corresponding to each column, when known.
            ``None`` for a full-gene access that did not carry names back.
    """

    cells: NDArray[np.intp] = field(compare=False)
    data: NDArray[np.generic] = field(compare=False)
    num_genes: int
    gene_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.num_genes <= 0:
            raise ValueError(f"num_genes must be positive, got {self.num_genes}")
        cells = _as_cell_index(self.cells, "cells")
        data = np.asarray(self.data)
        if data.ndim != 1:
            raise ValueError(f"data must be 1D, got {data.ndim}D")
        expected = cells.shape[0] * self.num_genes
        if data.shape[0] != expected:
            raise ValueError(
                f"data length {data.shape[0]} != num_cells*num_genes "
                f"({cells.shape[0]}*{self.num_genes} = {expected})"
            )
        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "data", data)
        if self.gene_names is not None:
            object.__setattr__(self, "gene_names", tuple(self.gene_names))

    @classmethod
    def from_array(
        cls,
        cells: Any,
        data: Any,
        num_genes: int,
        gene_names: Any | None = None,
    ) -> "CellData":
        """Build a :class:`CellData` from a raw 1D result array."""
        return cls(cells=cells, data=data, num_genes=num_genes, gene_names=gene_names)

    @property
    def num_cells(self) -> int:
        """Number of cells in this result (length of :attr:`cells`)."""
        return int(self.cells.shape[0])

    @property
    def matrix(self) -> NDArray[np.generic]:
        """``data`` reshaped to ``[num_cells, num_genes]`` (zero-copy view)."""
        return self.data.reshape(self.num_cells, self.num_genes)

    def __array__(self, dtype: Any = None, copy: Any = None) -> NDArray[np.generic]:
        """Allow ``np.asarray(cell_data)`` to read the decoded payload.

        Returns the raw 1D ``data`` array, so callers that treat an access
        result as a plain ndarray keep working without touching ``.data``.
        """
        if dtype is None:
            return self.data
        return np.asarray(self.data, dtype=dtype)


@dataclass(frozen=True)
class CellBatch:
    """Decoded batch type for the streaming-prefetch path.

    A :class:`CellBatch` is always a *decoded output* — ``cells``, ``data``
    and ``num_genes`` are all populated.  It is yielded by the prefetch
    iterator after Rust decodes the batch.  The prefetch *input* side is a
    :class:`CellAccess` (cells only), not a half-filled batch; this keeps the
    input and output representations distinct and removes the ambiguity of a
    single type that is sometimes-decoded.

    Attributes:
        cells: 1D ``intp`` numpy array of cell indices in this batch (any
            order, subset, repeats).
        data: 1D row-major numpy array of shape
            ``[len(cells) * num_genes]``.  ``f16`` → ``float16``;
            ``bf16`` → ``uint16`` raw bit pattern.
        num_genes: Number of gene columns per cell.
    """

    cells: NDArray[np.intp] = field(compare=False)
    data: NDArray[np.generic] = field(compare=False)
    num_genes: int

    def __post_init__(self) -> None:
        if self.num_genes <= 0:
            raise ValueError(f"num_genes must be positive, got {self.num_genes}")
        cells = _as_cell_index(self.cells, "cells")
        data = np.asarray(self.data)
        if data.ndim != 1:
            raise ValueError(f"data must be 1D, got {data.ndim}D")
        expected = cells.shape[0] * self.num_genes
        if data.shape[0] != expected:
            raise ValueError(
                f"data length {data.shape[0]} != num_cells*num_genes "
                f"({cells.shape[0]}*{self.num_genes} = {expected})"
            )
        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "data", data)

    @classmethod
    def from_array(
        cls,
        cells: Any,
        data: Any,
        num_genes: int,
    ) -> "CellBatch":
        """Build a decoded batch from a raw 1D result array."""
        return cls(cells=cells, data=data, num_genes=num_genes)

    @property
    def num_cells(self) -> int:
        """Number of cells in this batch (length of :attr:`cells`)."""
        return int(self.cells.shape[0])

    @property
    def matrix(self) -> NDArray[np.generic]:
        """``data`` reshaped to ``[num_cells, num_genes]`` (zero-copy view)."""
        return self.data.reshape(self.num_cells, self.num_genes)
