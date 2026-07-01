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
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, overload

import numpy as np
from numpy.typing import DTypeLike, NDArray

__all__ = ["CellIndexDataset", "CellIndexPlan"]


def _index_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got bool")
    return operator.index(value)


def _counts_tuple(
    cells_per_dataset: Sequence[int],
    name: str = "cells_per_dataset",
) -> tuple[int, ...]:
    counts = tuple(_index_int(c, f"{name}[{i}]") for i, c in enumerate(cells_per_dataset))
    for i, count in enumerate(counts):
        if count < 0:
            raise ValueError(f"{name}[{i}] must be non-negative, got {count}")
    return counts


def _unsigned_dtype(max_value: int, dtype: DTypeLike | None, *, kind: str) -> np.dtype[Any]:
    if dtype is None:
        if kind == "dataset":
            if max_value <= np.iinfo(np.uint16).max:
                return np.dtype(np.uint16)
            if max_value <= np.iinfo(np.uint32).max:
                return np.dtype(np.uint32)
            return np.dtype(np.uint64)
        return np.dtype(np.uint32 if max_value <= np.iinfo(np.uint32).max else np.uint64)
    resolved = np.dtype(dtype)
    if resolved.kind not in ("i", "u") or resolved.kind == "b":
        raise TypeError(f"{kind}_dtype must be an integer dtype")
    info = np.iinfo(resolved)
    if max_value > info.max:
        raise ValueError(f"{kind}_dtype {resolved} cannot represent max value {max_value}")
    return resolved


def _integer_vector(value: object, name: str, dtype: DTypeLike | None = None) -> NDArray[Any]:
    arr = np.asarray(value)
    if (
        arr.ndim == 0
        and not np.isscalar(value)
        and isinstance(value, Iterable)
        and not isinstance(value, (str, bytes, bytearray))
    ):
        arr = _strict_index_iterable(value, name)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D integer array")
    if arr.dtype.kind not in ("i", "u") or arr.dtype.kind == "b":
        raise TypeError(f"{name} must be a 1D integer array")
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _strict_index_iterable(value: Iterable[object], name: str) -> NDArray[np.intp]:
    out: list[int] = []
    for i, item in enumerate(value):
        try:
            out.append(_index_int(item, f"{name}[{i}]"))
        except TypeError as err:
            raise TypeError(f"{name} must be a 1D integer array") from err
    return np.asarray(out, dtype=np.intp)


def _validate_non_negative(arr: NDArray[Any], name: str) -> None:
    if arr.size == 0:
        return
    if arr.dtype.kind == "i" and int(arr.min()) < 0:
        raise ValueError(f"{name} values must be non-negative")


def _trim_arrays(
    dataset_index: NDArray[Any],
    cell_index: NDArray[Any],
    batch_size: int,
    drop_last: bool,
) -> tuple[NDArray[Any], NDArray[Any]]:
    if not drop_last:
        return dataset_index, cell_index
    stop = (dataset_index.shape[0] // batch_size) * batch_size
    return dataset_index[:stop], cell_index[:stop]


@dataclass(frozen=True)
class CellIndexPlan:
    """Numeric scheduled-prefetch plan for multi-dataset cell access.

    ``dataset_index[i]`` selects a dataset from the ``ids`` passed to
    :meth:`scdata.databank.ScDataBank.prefetch_indexed`; ``cell_index[i]`` is
    the local row id inside that dataset.  The arrays already encode the full
    access order.  Batching is defined by ``batch_size``.
    """

    dataset_index: NDArray[Any]
    cell_index: NDArray[Any]
    batch_size: int

    def __post_init__(self) -> None:
        batch_size = _index_int(self.batch_size, "batch_size")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        dataset_index = _integer_vector(self.dataset_index, "dataset_index")
        cell_index = _integer_vector(self.cell_index, "cell_index")
        if dataset_index.shape[0] != cell_index.shape[0]:
            raise ValueError(
                "dataset_index and cell_index must have the same length, "
                f"got {dataset_index.shape[0]} and {cell_index.shape[0]}"
            )
        _validate_non_negative(dataset_index, "dataset_index")
        _validate_non_negative(cell_index, "cell_index")
        object.__setattr__(self, "dataset_index", dataset_index)
        object.__setattr__(self, "cell_index", cell_index)
        object.__setattr__(self, "batch_size", batch_size)

    @classmethod
    def from_counts(
        cls,
        cells_per_dataset: Sequence[int],
        *,
        local_cells: Sequence[None | int | Iterable[int] | NDArray[Any]] | None = None,
        batch_size: int = 1024,
        drop_last: bool = False,
        dataset_dtype: DTypeLike | None = None,
        cell_dtype: DTypeLike | None = None,
    ) -> "CellIndexPlan":
        """Build a plan by concatenating per-dataset local cell ids.

        ``local_cells`` optionally supplies one entry per dataset:
        ``None`` means every local cell, an integer means the first ``n``
        cells, and an integer array/iterable supplies the exact local ids.
        """
        counts = _counts_tuple(cells_per_dataset)
        batch_size = _index_int(batch_size, "batch_size")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if local_cells is None:
            specs: tuple[None | int | Iterable[int] | NDArray[Any], ...] = (None,) * len(counts)
        else:
            specs = tuple(local_cells)
            if len(specs) != len(counts):
                raise ValueError(
                    "local_cells must have one entry per dataset, "
                    f"got {len(specs)} for {len(counts)} datasets"
                )

        prepared: list[None | int | NDArray[Any]] = []
        lengths: list[int] = []
        max_cell = 0
        for dataset_idx, (count, spec) in enumerate(zip(counts, specs, strict=True)):
            name = f"local_cells[{dataset_idx}]"
            if spec is None:
                prepared.append(None)
                lengths.append(count)
                if count:
                    max_cell = max(max_cell, count - 1)
                continue
            if isinstance(spec, (bool, np.bool_)):
                raise TypeError(f"{name} must be an integer, an integer array, or None")
            try:
                n = _index_int(spec, name)  # type: ignore[arg-type]
            except TypeError:
                arr = _integer_vector(spec, name)
                _validate_non_negative(arr, name)
                if arr.size:
                    max_seen = int(arr.max())
                    if max_seen >= count:
                        raise ValueError(
                            f"{name} values must be < cells_per_dataset[{dataset_idx}]={count}"
                        )
                    max_cell = max(max_cell, max_seen)
                prepared.append(arr)
                lengths.append(int(arr.shape[0]))
                continue
            if n < 0:
                raise ValueError(f"{name} must be non-negative, got {n}")
            if n > count:
                raise ValueError(f"{name}={n} exceeds cells_per_dataset[{dataset_idx}]={count}")
            prepared.append(n)
            lengths.append(n)
            if n:
                max_cell = max(max_cell, n - 1)

        dataset_dtype_resolved = _unsigned_dtype(
            max(len(counts) - 1, 0), dataset_dtype, kind="dataset"
        )
        cell_dtype_resolved = _unsigned_dtype(max_cell, cell_dtype, kind="cell")
        total = sum(lengths)
        dataset_index = np.empty(total, dtype=dataset_dtype_resolved)
        cell_index = np.empty(total, dtype=cell_dtype_resolved)

        offset = 0
        for dataset_idx, spec in enumerate(prepared):
            length = lengths[dataset_idx]
            stop = offset + length
            dataset_index[offset:stop] = dataset_idx
            if spec is None:
                cell_index[offset:stop] = np.arange(length, dtype=cell_dtype_resolved)
            elif isinstance(spec, int):
                cell_index[offset:stop] = np.arange(spec, dtype=cell_dtype_resolved)
            else:
                cell_index[offset:stop] = spec.astype(cell_dtype_resolved, copy=False)
            offset = stop

        dataset_index, cell_index = _trim_arrays(
            dataset_index, cell_index, batch_size, drop_last
        )
        return cls(dataset_index=dataset_index, cell_index=cell_index, batch_size=batch_size)

    @classmethod
    def from_global_indices(
        cls,
        cells_per_dataset: Sequence[int],
        indices: Iterable[int] | NDArray[Any],
        *,
        batch_size: int = 1024,
        drop_last: bool = False,
        dataset_dtype: DTypeLike | None = None,
        cell_dtype: DTypeLike | None = None,
    ) -> "CellIndexPlan":
        """Build a plan from flat corpus indices in the desired access order."""
        counts = _counts_tuple(cells_per_dataset)
        batch_size = _index_int(batch_size, "batch_size")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        total = sum(counts)
        global_index = _integer_vector(indices, "indices")
        if drop_last:
            stop = (global_index.shape[0] // batch_size) * batch_size
            global_index = global_index[:stop]
        _validate_non_negative(global_index, "indices")
        if global_index.size and int(global_index.max()) >= total:
            raise ValueError(f"indices values must be < total cells {total}")

        dataset_dtype_resolved = _unsigned_dtype(
            max(len(counts) - 1, 0), dataset_dtype, kind="dataset"
        )
        cell_dtype_resolved = _unsigned_dtype(
            max((count - 1 for count in counts if count), default=0),
            cell_dtype,
            kind="cell",
        )
        offsets = np.empty(len(counts) + 1, dtype=np.uint64)
        offsets[0] = 0
        if counts:
            offsets[1:] = np.cumsum(np.asarray(counts, dtype=np.uint64), dtype=np.uint64)
        global_u64 = global_index.astype(np.uint64, copy=False)
        dataset_index = np.searchsorted(offsets[1:], global_u64, side="right").astype(
            dataset_dtype_resolved, copy=False
        )
        cell_index = (global_u64 - offsets[dataset_index.astype(np.intp, copy=False)]).astype(
            cell_dtype_resolved,
            copy=False,
        )
        return cls(dataset_index=dataset_index, cell_index=cell_index, batch_size=batch_size)

    def take(
        self,
        order: Iterable[int] | NDArray[Any],
        *,
        batch_size: int | None = None,
        drop_last: bool = False,
    ) -> "CellIndexPlan":
        """Return a reordered view/copy of this plan using numpy indexing."""
        order_arr = _integer_vector(order, "order")
        _validate_non_negative(order_arr, "order")
        if order_arr.size and int(order_arr.max()) >= self.num_cells:
            raise ValueError(f"order values must be < num_cells {self.num_cells}")
        next_batch_size = (
            self.batch_size if batch_size is None else _index_int(batch_size, "batch_size")
        )
        dataset_index = self.dataset_index[order_arr]
        cell_index = self.cell_index[order_arr]
        dataset_index, cell_index = _trim_arrays(
            dataset_index, cell_index, next_batch_size, drop_last
        )
        return CellIndexPlan(
            dataset_index=dataset_index,
            cell_index=cell_index,
            batch_size=next_batch_size,
        )

    @property
    def num_cells(self) -> int:
        return int(self.cell_index.shape[0])

    @property
    def num_batches(self) -> int:
        return (self.num_cells + self.batch_size - 1) // self.batch_size

    def iter_batches(self) -> Iterator[tuple[NDArray[Any], NDArray[Any]]]:
        for start in range(0, self.num_cells, self.batch_size):
            stop = min(start + self.batch_size, self.num_cells)
            yield self.dataset_index[start:stop], self.cell_index[start:stop]

    def __len__(self) -> int:
        return self.num_cells


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
        counts = _counts_tuple(cells_per_file, "cells_per_file")
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        self._cells_per_file: tuple[int, ...] = counts
        self._offsets: tuple[int, ...] = tuple(offsets)

    def __len__(self) -> int:
        return self._offsets[-1]

    @overload
    def __getitem__(self, index: int) -> tuple[int, int]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[tuple[int, int]]: ...

    def __getitem__(self, index: int | slice) -> tuple[int, int] | Sequence[tuple[int, int]]:
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
