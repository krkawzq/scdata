"""Torch DataLoader adapter backed by :class:`scdata.databank.ScDataBank`."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Mapping as MappingABC
from os import PathLike
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Protocol,
    Sequence,
)
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from scdata.data._coerce import _as_gene_names, _coerce_index_int
from scdata.data._cell import CellBatch, _as_cell_index
from scdata.data._stats import BankConfigSummary, LoaderStats, _StatsCollector

try:
    import torch
    from torch.utils.data import DataLoader as _TorchDataLoader
except ModuleNotFoundError:  # pragma: no cover - exercised in torch-free envs.
    torch = None

    class _MissingTorchDataLoader:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ModuleNotFoundError("ScDataLoader requires torch. Install torch first.")

    _TorchDataLoader = _MissingTorchDataLoader

if TYPE_CHECKING:
    from scdata.data._dataset import DType
    from scdata.databank import DatasetId, MissingGenePolicy, ScDataBank, ScheduledPrefetchConfig
else:
    DType = object
    DatasetId = object
    MissingGenePolicy = object
    ScDataBank = object
    ScheduledPrefetchConfig = object

__all__ = ["ScDataLoader"]


class _SupportsPrefetch(Protocol):
    def prefetch(
        self,
        id: DatasetId | str | int | PathLike[str],
        batches: Iterable[Iterable[int] | NDArray[np.intp]],
        genes: Iterable[str] | None = None,
        *,
        missing: MissingGenePolicy | str | None = None,
        dtype: DType | str | None = None,
        config: ScheduledPrefetchConfig | Mapping[str, Any] | None = None,
    ) -> Iterator[CellBatch]: ...

    def prefetch_multi(
        self,
        ids: Sequence[DatasetId | str | int | PathLike[str]],
        batches: Iterable[Iterable[tuple[int, Iterable[int] | NDArray[np.intp]]]],
        genes: Iterable[str] | None = None,
        *,
        missing: MissingGenePolicy | str | None = None,
        dtype: DType | str | None = None,
        config: ScheduledPrefetchConfig | Mapping[str, Any] | None = None,
    ) -> Iterator[CellBatch]: ...


class _BatchPartPlan(TypedDict):
    file_id: int
    positions: NDArray[np.intp]
    cells: NDArray[np.intp]


class _BatchPlan(TypedDict):
    file_ids: NDArray[np.intp]
    cell_ids: NDArray[np.intp]
    parts: list[_BatchPartPlan]
    stream_parts: list[_BatchPartPlan]


class _ScDataBatchBase(TypedDict):
    file_ids: NDArray[np.intp]
    cell_ids: NDArray[np.intp]
    batch: CellBatch
    batches: Mapping[int, CellBatch]
    cells: Mapping[int, NDArray[np.intp]]
    positions: Mapping[int, NDArray[np.intp]]


class _ScDataSingleFileFields(TypedDict, total=False):
    file_id: int
    cells_in_file: NDArray[np.intp]
    positions_in_batch: NDArray[np.intp]


class ScDataBatch(_ScDataBatchBase, _ScDataSingleFileFields):
    """Dictionary passed to ``ScDataLoader`` user ``collate_fn``."""


def _sequential_positions(positions: NDArray[np.intp], start: int, length: int) -> bool:
    if positions.shape[0] != length:
        return False
    if length == 0:
        return True
    if int(positions[0]) != start:
        return False
    if int(positions[-1]) != start + length - 1:
        return False
    return bool(np.array_equal(positions, np.arange(start, start + length, dtype=np.intp)))


def _rows_for_positions(
    matrix: NDArray[np.generic],
    positions: NDArray[np.intp],
) -> NDArray[np.generic]:
    if positions.shape[0] == 0:
        return matrix[:0]
    start = int(positions[0])
    stop = start + positions.shape[0]
    if (
        0 <= start
        and stop <= matrix.shape[0]
        and _sequential_positions(
            positions,
            start,
            positions.shape[0],
        )
    ):
        return matrix[start:stop]
    return np.ascontiguousarray(matrix[positions])


class _LazyCellBatches(MappingABC[int, CellBatch]):
    def __init__(self, decoded: CellBatch, parts: Sequence[_BatchPartPlan]) -> None:
        self._decoded = decoded
        self._parts = {part["file_id"]: part for part in parts}
        self._order = [part["file_id"] for part in parts]
        self._cache: dict[int, CellBatch] = {}

    def __getitem__(self, file_id: int) -> CellBatch:
        if file_id in self._cache:
            return self._cache[file_id]
        part = self._parts[file_id]
        cells = part["cells"]
        positions = part["positions"]
        if _sequential_positions(positions, 0, self._decoded.num_cells) and np.array_equal(
            cells,
            self._decoded.cells,
        ):
            batch = self._decoded
        else:
            rows = _rows_for_positions(self._decoded.to_numpy(), positions)
            batch = CellBatch.from_array(
                cells=cells,
                data=rows.reshape(-1),
                num_genes=self._decoded.num_genes,
                gene_names=self._decoded.var_names,
            )
        self._cache[file_id] = batch
        return batch

    def __iter__(self) -> Iterator[int]:
        return iter(self._order)

    def __len__(self) -> int:
        return len(self._order)


def _identity_raw_collate(
    batch: Sequence[
        tuple[int, int] | list[int] | NDArray[np.intp] | Mapping[str, int | np.integer[Any]]
    ],
) -> Sequence[tuple[int, int] | list[int] | NDArray[np.intp] | Mapping[str, int | np.integer[Any]]]:
    return batch


def _identity_sc_collate(batch: ScDataBatch) -> ScDataBatch:
    return batch


def _to_int(value: int | np.integer[Any] | object, context: str) -> int:
    return _coerce_index_int(value, context)


def _parse_sample(
    sample: tuple[int, int] | list[int] | NDArray[np.intp] | Mapping[str, int | np.integer[Any]],
) -> tuple[int, int]:
    """Parse one dataset sample into ``(file_id, cell_id)``."""
    if isinstance(sample, Mapping):
        if "file_id" in sample:
            file_id = sample["file_id"]
        elif "fileid" in sample:
            file_id = sample["fileid"]
        else:
            raise KeyError("sample mapping must contain 'file_id' or 'fileid'")

        if "cell_id" in sample:
            cell_id = sample["cell_id"]
        elif "cellid" in sample:
            cell_id = sample["cellid"]
        else:
            raise KeyError("sample mapping must contain 'cell_id' or 'cellid'")
        return _to_int(file_id, "file_id"), _to_int(cell_id, "cell_id")

    if hasattr(sample, "tolist"):
        sample = sample.tolist()  # type: ignore
    try:
        file_id, cell_id = sample  # type: ignore[misc]
    except (TypeError, ValueError) as err:
        raise TypeError("dataset samples must be two values: (file_id, cell_id)") from err
    return _to_int(file_id, "file_id"), _to_int(cell_id, "cell_id")


def _batch_plan(
    samples: Sequence[
        tuple[int, int] | list[int] | NDArray[np.intp] | Mapping[str, int | np.integer[Any]]
    ],
) -> _BatchPlan:
    file_ids = np.empty(len(samples), dtype=np.intp)
    cell_ids = np.empty(len(samples), dtype=np.intp)
    grouped: OrderedDict[int, tuple[list[int], list[int]]] = OrderedDict()
    stream_rows: list[tuple[int, list[int]]] = []

    for pos, sample in enumerate(samples):
        file_id, cell_id = _parse_sample(sample)
        if file_id < 0:
            raise ValueError(f"file_id values must be non-negative, got {file_id}")
        if cell_id < 0:
            raise ValueError(f"cell_id values must be non-negative, got {cell_id}")
        file_ids[pos] = file_id
        cell_ids[pos] = cell_id
        grouped_entry = grouped.get(file_id)
        if grouped_entry is None:
            grouped_entry = ([], [])
            grouped[file_id] = grouped_entry
        grouped_entry[0].append(pos)
        grouped_entry[1].append(cell_id)
        if stream_rows and stream_rows[-1][0] == file_id:
            stream_rows[-1][1].append(cell_id)
        else:
            stream_rows.append((file_id, [cell_id]))

    parts: list[_BatchPartPlan] = []
    for file_id, (positions, cells) in grouped.items():
        parts.append(
            {
                "file_id": file_id,
                "positions": np.asarray(positions, dtype=np.intp),
                "cells": _as_cell_index(cells, "cell_id"),
            }
        )

    stream_parts: list[_BatchPartPlan] = [
        {
            "file_id": file_id,
            "positions": np.empty(0, dtype=np.intp),
            "cells": _as_cell_index(cells, "cell_id"),
        }
        for file_id, cells in stream_rows
    ]

    return {
        "file_ids": file_ids,
        "cell_ids": cell_ids,
        "parts": parts,
        "stream_parts": stream_parts,
    }


def _index_to_int(index: Any) -> int:
    return _to_int(index, "dataset index")


def _iter_index_batches(loader: "ScDataLoader") -> Iterable[list[int]]:
    batch_sampler = getattr(loader, "batch_sampler", None)
    if batch_sampler is not None:
        for batch in batch_sampler:
            if hasattr(batch, "tolist"):
                batch = batch.tolist()
            yield [_index_to_int(idx) for idx in batch]
        return

    sampler = getattr(loader, "sampler", None)
    if sampler is None:
        raise TypeError("ScDataLoader requires a map-style torch dataset or batch_sampler")
    for index in sampler:
        yield [_index_to_int(index)]


def _normalize_dataset_ids(
    dataset_ids: Mapping[int, DatasetId | str | int | PathLike[str]]
    | Sequence[DatasetId | str | int | PathLike[str]]
    | None,
) -> Callable[[int], DatasetId | str | int | PathLike[str]]:
    if dataset_ids is None:
        raise ValueError("dataset_ids is required when dataset samples use numeric file_id values")
    if isinstance(dataset_ids, Mapping):

        def resolve_mapping(file_id: int) -> DatasetId | str | int | PathLike[str]:
            try:
                return dataset_ids[file_id]
            except KeyError as err:
                raise KeyError(f"unknown file_id {file_id}") from err

        return resolve_mapping

    def resolve_sequence(file_id: int) -> DatasetId | str | int | PathLike[str]:
        try:
            return dataset_ids[file_id]
        except IndexError as err:
            raise IndexError(f"unknown file_id {file_id}") from err

    return resolve_sequence


# Sentinel for ``shuffle`` so the default (True) can be distinguished from an
# explicit ``shuffle=None``/``False`` when resolving sampler mutual exclusion.
_SHUFFLE_UNSET: Any = object()


def _reject_unsupported_torch_options(kwargs: Mapping[str, Any]) -> None:
    """Reject DataLoader options this adapter would otherwise ignore."""
    if _to_int(kwargs.get("num_workers", 0), "num_workers") != 0:
        raise ValueError("ScDataLoader does not support torch worker processes; use num_workers=0")
    if bool(kwargs.get("pin_memory", False)):
        raise ValueError("ScDataLoader does not support pin_memory=True")
    if kwargs.get("timeout", 0) not in (0, 0.0):
        raise ValueError("ScDataLoader does not support DataLoader timeout")
    if kwargs.get("worker_init_fn") is not None:
        raise ValueError("ScDataLoader does not support worker_init_fn")
    if kwargs.get("multiprocessing_context") is not None:
        raise ValueError("ScDataLoader does not support multiprocessing_context")
    if kwargs.get("prefetch_factor") is not None:
        raise ValueError("ScDataLoader does not support torch prefetch_factor")
    if bool(kwargs.get("persistent_workers", False)):
        raise ValueError("ScDataLoader does not support persistent_workers=True")


def _dataset_ids_as_tuple(
    dataset_ids: Mapping[int, DatasetId | str | int | PathLike[str]]
    | Sequence[DatasetId | str | int | PathLike[str]]
    | None,
) -> tuple[DatasetId | str | int | PathLike[str], ...]:
    """Flatten the ``dataset_ids`` argument into a stable tuple for introspection.

    A mapping is ordered by numeric file id; a sequence keeps its order.
    """
    if dataset_ids is None:
        return ()
    if isinstance(dataset_ids, Mapping):
        return tuple(dataset_ids[k] for k in sorted(dataset_ids))
    return tuple(dataset_ids)


class ScDataLoader(_TorchDataLoader):  # type: ignore[misc, valid-type]
    """Torch DataLoader whose samples are ``(file_id, cell_id)`` pairs.

    Torch still owns sampling, shuffling, ``batch_size``, ``sampler``,
    ``batch_sampler`` and ``drop_last``.  On ``iter(loader)``, this adapter
    materializes all torch batch indices, plans each batch as cross-dataset
    parts, starts one :meth:`ScDataBank.prefetch_multi` stream, and yields
    decoded dictionaries to ``collate_fn``.

    The dictionary always has ``file_ids`` and ``cell_ids`` in original batch
    order plus ``batch``, a single decoded :class:`CellBatch` with the same row
    order.  ``batches`` / ``cells`` / ``positions`` are retained as
    compatibility views grouped by file id.

    ``bank`` is **not** owned by the loader; the caller is responsible for its
    lifecycle (close it after iteration).  Access it via :attr:`bank` if needed.
    """

    def __init__(
        self,
        bank: "ScDataBank",
        dataset: Sequence[
            tuple[int, int] | list[int] | NDArray[np.intp] | Mapping[str, int | np.integer[Any]]
        ],
        *,
        batch_size: int = 1024,
        shuffle: Any = _SHUFFLE_UNSET,
        drop_last: bool = False,
        sampler: Any = None,
        batch_sampler: Any = None,
        dataset_ids: Mapping[int, DatasetId | str | int | PathLike[str]]
        | Sequence[DatasetId | str | int | PathLike[str]]
        | None = None,
        genes: Iterable[str] | None = None,
        missing: "MissingGenePolicy | str | None" = None,
        out_dtype: "DType | str | None" = None,
        prefetch_config: "ScheduledPrefetchConfig | Mapping[str, Any] | None" = None,
        collate_fn: Callable[[ScDataBatch], Any] | None = None,
        collect_stats: bool = True,
        bank_config_summary: BankConfigSummary | None = None,
        **torch_kwargs: Any,
    ) -> None:
        if torch is None:
            raise ModuleNotFoundError("ScDataLoader requires torch. Install torch first.")

        # Resolve shuffle against torch's mutual-exclusion rules: a provided
        # sampler or batch_sampler already defines the draw order, so an
        # unspecified shuffle auto-disables (-> None) instead of forcing the
        # caller to pass shuffle=False explicitly.  Otherwise shuffle defaults
        # to False (torch semantics); training entry points such as
        # :meth:`Corpus.loader` pass ``shuffle=True`` explicitly.
        if batch_sampler is not None:
            if shuffle is _SHUFFLE_UNSET:
                shuffle = None
            elif shuffle:
                raise ValueError("batch_sampler is mutually exclusive with shuffle")
        elif sampler is not None:
            if shuffle is _SHUFFLE_UNSET:
                shuffle = None
            elif shuffle:
                raise ValueError(
                    "sampler is mutually exclusive with shuffle; pass shuffle=False"
                )
        else:
            shuffle = False if shuffle is _SHUFFLE_UNSET else shuffle

        _reject_unsupported_torch_options(torch_kwargs)
        # torch's own collate path is bypassed by __iter__; pass a raw identity
        # so torch never tries to stack (file_id, cell_id) samples itself.
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            drop_last=drop_last,
            collate_fn=_identity_raw_collate,
            **torch_kwargs,
        )
        self.sc_bank: _SupportsPrefetch = cast(_SupportsPrefetch, bank)
        self.sc_dataset_id: Callable[[int], DatasetId | str | int | PathLike[str]] = (
            _normalize_dataset_ids(dataset_ids)
        )
        self._sc_dataset_ids: tuple[DatasetId | str | int | PathLike[str], ...] = (
            _dataset_ids_as_tuple(dataset_ids)
        )
        self.sc_genes: tuple[str, ...] | None = (
            _as_gene_names(genes, "genes") if genes is not None else None
        )
        self.sc_missing: MissingGenePolicy | str | None = missing
        self.sc_out_dtype: DType | str | None = out_dtype
        self.sc_prefetch_config: ScheduledPrefetchConfig | Mapping[str, Any] | None = (
            prefetch_config
        )
        self.sc_collate_fn: Callable[[ScDataBatch], Any] = (
            collate_fn if collate_fn is not None else _identity_sc_collate
        )
        self.sc_collect_stats: bool = collect_stats
        self._sc_stats: _StatsCollector | None = None
        if collect_stats:
            self._sc_stats = _StatsCollector(
                prefetch_config=prefetch_config,
                bank_config_summary=bank_config_summary,
            )

    # -- read-only introspection -------------------------------------------

    @property
    def bank(self) -> "ScDataBank":
        """The underlying :class:`~scdata.databank.ScDataBank` (not owned)."""
        return cast("ScDataBank", self.sc_bank)

    @property
    def gene_names(self) -> tuple[str, ...] | None:
        """Gene names batches are projected onto (``None`` = all genes)."""
        return self.sc_genes

    @property
    def dataset_ids(self) -> tuple[DatasetId | str | int | PathLike[str], ...]:
        """Dataset ids this loader draws from, in file_id order."""
        return self._sc_dataset_ids

    def __iter__(self) -> Iterator[Any]:
        plans = self._sc_build_plans()
        if not plans:
            return iter(())

        file_to_dataset_idx: OrderedDict[int, int] = OrderedDict()
        dataset_ids: list[DatasetId | str | int | PathLike[str]] = []
        for plan in plans:
            for part in plan["stream_parts"]:
                file_id = part["file_id"]
                if file_id not in file_to_dataset_idx:
                    file_to_dataset_idx[file_id] = len(dataset_ids)
                    dataset_ids.append(self.sc_dataset_id(file_id))

        def prefetch_batches() -> Iterator[list[tuple[int, NDArray[np.intp]]]]:
            for plan in plans:
                yield [
                    (file_to_dataset_idx[part["file_id"]], part["cells"])
                    for part in plan["stream_parts"]
                ]

        prefetch_kwargs: dict[str, Any] = {
            "genes": self.sc_genes,
            "missing": self.sc_missing,
            "config": self.sc_prefetch_config,
        }
        if self.sc_out_dtype is not None:
            prefetch_kwargs["dtype"] = self.sc_out_dtype
        prefetcher = iter(
            self.sc_bank.prefetch_multi(
                dataset_ids,
                prefetch_batches(),
                **prefetch_kwargs,
            )
        )

        def iterator() -> Iterator[Any]:
            perf_counter = time.perf_counter
            collect = self.sc_collect_stats
            collector = self._sc_stats
            for plan in plans:
                t0 = perf_counter() if collect else 0.0
                decoded = next(prefetcher)
                batches = _LazyCellBatches(decoded, plan["parts"])
                cells: dict[int, NDArray[np.intp]] = {
                    part["file_id"]: part["cells"] for part in plan["parts"]
                }
                positions: dict[int, NDArray[np.intp]] = {
                    part["file_id"]: part["positions"] for part in plan["parts"]
                }
                # ``wait`` covers the ``next()`` calls blocked on the bank;
                # ``wall`` additionally covers the collate step below.
                t1 = perf_counter() if collect else 0.0

                batch: ScDataBatch = {
                    "file_ids": plan["file_ids"],
                    "cell_ids": plan["cell_ids"],
                    "batch": decoded,
                    "batches": batches,
                    "cells": cells,
                    "positions": positions,
                }
                if len(batches) == 1:
                    file_id = next(iter(batches))
                    batch.update(
                        {
                            "file_id": file_id,
                            "cells_in_file": cells[file_id],
                            "positions_in_batch": positions[file_id],
                        }
                    )
                result = self.sc_collate_fn(batch)
                if collect and collector is not None:
                    t2 = perf_counter()
                    collector.record_batch(
                        wait_seconds=t1 - t0,
                        wall_seconds=t2 - t0,
                        num_cells=len(plan["cell_ids"]),
                        num_genes=decoded.num_genes,
                        bytes_=decoded.data.nbytes,
                    )
                yield result

        return iter(iterator())

    def stats(self, *, reset: bool = True) -> LoaderStats:
        """Return consumer-side health metrics collected during iteration.

        Requires ``collect_stats=True`` at construction.  By default the
        collector is reset after the snapshot, so each call reports a fresh
        window — pass ``reset=False`` to accumulate across calls.

        Only the most recent ``max_samples`` wait samples feed the
        percentiles (recent health), while ``batches_seen`` and
        ``stall_count`` are full-run counters.
        """
        if self._sc_stats is None:
            raise RuntimeError("stats() requires collect_stats=True at construction")
        return self._sc_stats.snapshot(reset=reset)

    @classmethod
    def from_paths(
        cls,
        bank: "ScDataBank",
        paths: Iterable[str | PathLike[str]],
        *,
        layer: str | None = None,
        matrix: str | None = None,
        gene_alignment: Literal["strict", "union", "intersection", "none"] = "strict",
        missing: "MissingGenePolicy | str | None" = None,
        batch_size: int = 1024,
        shuffle: bool = True,
        drop_last: bool = False,
        out_dtype: "DType | str | None" = None,
        prefetch_config: "ScheduledPrefetchConfig | Mapping[str, Any] | None" = None,
        collate_fn: Callable[[ScDataBatch], Any] | None = None,
        collect_stats: bool = True,
        **torch_kwargs: Any,
    ) -> "ScDataLoader":
        """Build a loader from ``paths`` and an existing ``bank``.

        Convenience entry point: parses and registers the stores via a
        :class:`~scdata.corpus.Corpus` built on ``bank`` (which the caller
        owns and must close), then returns the loader.  Use
        :class:`~scdata.corpus.Corpus` directly when you need bank/dataset
        access or lifecycle control.
        """
        from scdata.corpus import Corpus

        corpus = Corpus.from_bank(
            bank,
            paths,
            layer=layer,
            matrix=matrix,
            gene_alignment=gene_alignment,
            missing=missing,
        )
        return corpus.loader(
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            out_dtype=out_dtype,
            prefetch_config=prefetch_config,
            collate_fn=collate_fn,
            collect_stats=collect_stats,
            **torch_kwargs,
        )

    def _sc_build_plans(self) -> list[_BatchPlan]:
        plans: list[_BatchPlan] = []
        for indices in _iter_index_batches(self):
            samples = [self.dataset[index] for index in indices]
            plans.append(_batch_plan(samples))
        return plans
