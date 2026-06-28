"""Scheduled prefetch input and iterator types for the DataBank.

These are the pure-Python data carriers for the streaming access path:
:class:`PrefetchBatches` describes what a ``ScDataBank.prefetch_cells`` /
``prefetch_cells_by_gene_names`` call consumes (a stream of cell-index
batches), and :class:`PrefetchIterator` describes what it returns (a lazily
decoded stream of :class:`~scdata.data._cell.CellBatch` objects).

Like :mod:`scdata.data._cell`, this module stays in the data layer — no
dependency on the Rust extension.  The bank's execution layer
(:mod:`scdata.databank`) builds a :class:`PrefetchBatches` from caller
arguments and wraps the Rust prefetch producer in a :class:`PrefetchIterator`,
which is just a thin adapter from ``(cells, data, num_genes)`` tuples (the
form the Rust iterator yields) to decoded :class:`CellBatch` instances.

The input side of the pipeline is a :class:`~scdata.data._cell.CellAccess`
(cells only, optionally a gene subset applied to every batch); the output side
is a decoded :class:`CellBatch`.  Keeping the input as ``CellAccess`` and the
output as ``CellBatch`` means the two representations are distinct types, not
one overloaded type distinguished by which fields happen to be filled.

Why batches are materialized up front
-------------------------------------
The Rust prefetch producer runs on its own thread and pulls batch index lists
from the source.  If that source were a live Python iterable, the producer
would need the GIL to advance it — while the consumer (which holds the GIL)
blocks waiting for the producer, a classic deadlock.  :class:`PrefetchBatches`
therefore materializes every batch into a contiguous ``intp`` numpy array at
construction time, under the GIL, so the producer stays off the GIL entirely.
For Python callers the batch list is already in memory, so this costs nothing;
truly streaming sources should feed Rust directly rather than through this
wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
from numpy.typing import NDArray

from scdata.data._cell import CellAccess, CellBatch, _as_cell_index

__all__ = ["PrefetchBatches", "PrefetchIterator"]


@dataclass(frozen=True)
class PrefetchBatches:
    """Input for a scheduled prefetch call.

    Attributes:
        batches: Tuple of :class:`~scdata.data._cell.CellAccess` instances, each
            carrying the cell indices for one batch (and, when a gene subset
            applies to every batch, the same ``gene_names``).  Stored as
            ``CellAccess`` so the prefetch input unit is the same type a single
            ``access_cells`` call takes.
        gene_names: Optional gene-name subset projected onto every batch;
            ``None`` means all genes in dataset column order.  When set, this
            is the authoritative projection — the per-batch ``CellAccess``
            gene lists are ignored, because the Rust entry point takes one
            shared gene list for the whole stream.
    """

    batches: tuple[CellAccess, ...]
    gene_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        materialized = tuple(self._coerce_batch(b, i) for i, b in enumerate(self.batches))
        object.__setattr__(self, "batches", materialized)
        if self.gene_names is not None:
            object.__setattr__(self, "gene_names", tuple(self.gene_names))

    @staticmethod
    def _coerce_batch(value: CellAccess | Iterable[int], index: int) -> CellAccess:
        """Normalize one batch input into a :class:`CellAccess`.

        Accepts a :class:`CellAccess` (used as-is; its ``gene_names`` are
        ignored in favor of the stream-wide :attr:`gene_names`), a 1D
        cell-index iterable, or a numpy array.
        """
        if isinstance(value, CellAccess):
            return value
        return CellAccess.from_cells(_as_cell_index(value, f"batches[{index}]"))

    @classmethod
    def from_iterable(
        cls,
        batches: Iterable[CellAccess | Iterable[int]],
        gene_names: Iterable[str] | None = None,
    ) -> "PrefetchBatches":
        """Build a :class:`PrefetchBatches` from an iterable of cell iterables.

        Each element may be a :class:`CellAccess` or any 1D cell-index
        iterable; it is normalized to a :class:`CellAccess` immediately (see
        the module docstring for why this is not lazy).
        """
        coerced = tuple(cls._coerce_batch(b, i) for i, b in enumerate(batches))
        names = tuple(gene_names) if gene_names is not None else None
        return cls(batches=coerced, gene_names=names)

    @property
    def num_batches(self) -> int:
        """Number of batches."""
        return len(self.batches)

    @property
    def total_cells(self) -> int:
        """Total cell requests across all batches (with repeats counted)."""
        return sum(b.num_cells for b in self.batches)

    def batch_cell_arrays(self) -> list[NDArray[np.intp]]:
        """Return contiguous numpy cell-index arrays for the Rust fast path."""
        return [b.cells for b in self.batches]

    def batch_cell_lists(self) -> list[list[int]]:
        """Return batches as a ``list[list[int]]`` for the Rust binding.

        The Rust ``prefetch_cells`` entry point takes ``Vec<Vec<usize>>``; this
        is the canonical Python-side shape to hand it.  Centralizing the
        conversion here keeps the execution layer free of layout details.
        """
        return [b.cells.tolist() for b in self.batches]


class PrefetchIterator:
    """Iterator yielding decoded :class:`~scdata.data._cell.CellBatch` objects.

    Wraps any iterable of ``(cells, data, num_genes)`` tuples — in particular
    the Rust ``_PrefetchCells`` pyclass, whose ``__next__`` returns exactly
    that tuple form.  Each tuple is lifted into a decoded :class:`CellBatch`,
    the output type of the prefetch pipeline.  Staying duck-typed keeps this
    class in the data layer (no import of the Rust extension); the execution
    layer constructs it with the Rust producer as the source.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: Iterable[tuple[NDArray[np.intp], NDArray[np.generic], int]]) -> None:
        # Bind the iterator protocol once; ``inner`` may be a Rust pyclass that
        # is itself an iterator (``__iter__`` returns self) or any Python
        # iterable.
        self._inner: Iterator[tuple[NDArray[np.intp], NDArray[np.generic], int]] = iter(inner)

    def __iter__(self) -> "PrefetchIterator":
        return self

    def __next__(self) -> CellBatch:
        cells, data, num_genes = next(self._inner)
        return CellBatch.from_array(cells=cells, data=data, num_genes=num_genes)
