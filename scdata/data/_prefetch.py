"""Scheduled prefetch iterator types for the DataBank."""

from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np
from numpy.typing import NDArray

from scdata.data._coerce import _as_gene_names
from scdata.data._cell import CellBatch

__all__ = ["PrefetchIterator"]


class PrefetchIterator:
    """Iterator yielding decoded :class:`~scdata.data._cell.CellBatch` objects.

    Wraps any iterable of ``(cells, data, num_genes)`` tuples — in particular
    the Rust ``_PrefetchCells`` pyclass, whose ``__next__`` returns exactly
    that tuple form.  Each tuple is lifted into a decoded :class:`CellBatch`,
    the output type of the prefetch pipeline.  Staying duck-typed keeps this
    class in the data layer (no import of the Rust extension); the execution
    layer constructs it with the Rust producer as the source.
    """

    __slots__ = ("_inner", "_gene_names")

    def __init__(
        self,
        inner: Iterable[tuple[NDArray[np.intp], NDArray[np.generic], int]],
        gene_names: Iterable[str] | None = None,
    ) -> None:
        # Bind the iterator protocol once; ``inner`` may be a Rust pyclass that
        # is itself an iterator (``__iter__`` returns self) or any Python
        # iterable.
        self._inner: Iterator[tuple[NDArray[np.intp], NDArray[np.generic], int]] = iter(inner)
        self._gene_names = _as_gene_names(gene_names) if gene_names is not None else None

    def __iter__(self) -> "PrefetchIterator":
        return self

    def __next__(self) -> CellBatch:
        cells, data, num_genes = next(self._inner)
        return CellBatch.from_array(
            cells=cells,
            data=data,
            num_genes=num_genes,
            gene_names=self._gene_names,
        )
