"""Flat-index to ``(file_id, cell_id)`` adapter for multi-dataset corpora.

A :class:`CellIndexDataset` maps a single flat global index — the kind a
torch sampler yields — to a ``(file_id, cell_id)`` pair across multiple
registered datasets, via a prefix sum over per-file cell counts plus a
binary search.  It never materializes the full list of pairs, which matters
when the corpus holds tens of millions of cells.

The class bridges torch's map-style dataset protocol (``__getitem__`` takes
one integer index) and scdata's ``(file_id, cell_id)`` sample shape that
:class:`~scdata.data.ScDataLoader` expects.
"""

from __future__ import annotations

import operator
from bisect import bisect_right
from collections.abc import Sequence

__all__ = ["CellIndexDataset"]


class CellIndexDataset(Sequence[tuple[int, int]]):
    """Map a flat global index to ``(file_id, cell_id)`` via prefix-sum + bisect.

    A :class:`collections.abc.Sequence` so it satisfies the ``Sequence[...]``
    contract of :class:`~scdata.data.ScDataLoader`'s ``dataset`` argument; the
    torch map-style ``__getitem__`` / ``__len__`` protocol is unchanged.

    Args:
        cells_per_file: Cell count for each registered dataset, in ``file_id``
            order.  Must be non-negative integers.  An empty sequence yields a
            dataset of length 0.
    """

    __slots__ = ("_cells_per_file", "_offsets")

    def __init__(self, cells_per_file: Sequence[int]) -> None:
        counts = tuple(operator.index(c) for c in cells_per_file)
        for i, count in enumerate(counts):
            if count < 0:
                raise ValueError(f"cells_per_file[{i}] must be non-negative, got {count}")
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        self._cells_per_file: tuple[int, ...] = counts
        self._offsets: tuple[int, ...] = tuple(offsets)

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, index: int) -> tuple[int, int]:
        if isinstance(index, slice):
            raise TypeError("CellIndexDataset does not support slice indexing")
        idx = operator.index(index)
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(index)
        file_id = bisect_right(self._offsets, idx) - 1
        cell_id = idx - self._offsets[file_id]
        return file_id, cell_id

    @property
    def offsets(self) -> tuple[int, ...]:
        """Prefix sums of cell counts, length ``num_files + 1`` (leading 0)."""
        return self._offsets

    @property
    def num_files(self) -> int:
        """Number of datasets this index spans."""
        return len(self._cells_per_file)

    @property
    def cells_per_file(self) -> tuple[int, ...]:
        """Per-file cell counts, in ``file_id`` order."""
        return self._cells_per_file

    def __repr__(self) -> str:
        return f"CellIndexDataset(num_files={self.num_files}, num_cells={len(self)})"
