"""Torch DataLoader adapter backed by :class:`scdata.databank.ScDataBank`."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from os import PathLike
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from scdata.data._cell import CellBatch, _as_cell_index

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
    from scdata.databank import DatasetId, MissingGenePolicy, ScDataBank, ScheduledPrefetchConfig
else:
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
        config: ScheduledPrefetchConfig | None = None,
    ) -> Iterator[CellBatch]: ...


class _BatchPartPlan(TypedDict):
    file_id: int
    positions: NDArray[np.intp]
    cells: NDArray[np.intp]


class _BatchPlan(TypedDict):
    file_ids: NDArray[np.intp]
    cell_ids: NDArray[np.intp]
    parts: list[_BatchPartPlan]


class _ScDataBatchBase(TypedDict):
    file_ids: NDArray[np.intp]
    cell_ids: NDArray[np.intp]
    batches: Mapping[int, CellBatch]
    cells: Mapping[int, NDArray[np.intp]]
    positions: Mapping[int, NDArray[np.intp]]


class _ScDataSingleFileFields(TypedDict, total=False):
    file_id: int
    batch: CellBatch
    cells_in_file: NDArray[np.intp]
    positions_in_batch: NDArray[np.intp]


class ScDataBatch(_ScDataBatchBase, _ScDataSingleFileFields):
    """Dictionary passed to ``ScDataLoader`` user ``collate_fn``."""


def _identity_raw_collate(
    batch: Sequence[
        tuple[int, int] | list[int] | NDArray[np.intp] | Mapping[str, int | np.integer[Any]]
    ],
) -> Sequence[tuple[int, int] | list[int] | NDArray[np.intp] | Mapping[str, int | np.integer[Any]]]:
    return batch


def _identity_sc_collate(batch: ScDataBatch) -> ScDataBatch:
    return batch


def _to_int(value: int | np.integer[Any] | object, context: str) -> int:
    if hasattr(value, "item"):
        value = value.item()  # type: ignore
    try:
        return int(value)  # type: ignore
    except (TypeError, ValueError) as err:
        raise TypeError(f"{context} must be an integer, got {value!r}") from err


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
    grouped: OrderedDict[int, list[tuple[int, int]]] = OrderedDict()

    for pos, sample in enumerate(samples):
        file_id, cell_id = _parse_sample(sample)
        if file_id < 0:
            raise ValueError(f"file_id values must be non-negative, got {file_id}")
        if cell_id < 0:
            raise ValueError(f"cell_id values must be non-negative, got {cell_id}")
        file_ids[pos] = file_id
        cell_ids[pos] = cell_id
        grouped.setdefault(file_id, []).append((pos, cell_id))

    parts: list[_BatchPartPlan] = []
    for file_id, rows in grouped.items():
        parts.append(
            {
                "file_id": file_id,
                "positions": np.asarray([pos for pos, _ in rows], dtype=np.intp),
                "cells": _as_cell_index([cell_id for _, cell_id in rows], "cell_id"),
            }
        )

    return {"file_ids": file_ids, "cell_ids": cell_ids, "parts": parts}


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


def _extract_positional_collate(
    args: tuple[Any, ...],
    collate_fn: Callable[[ScDataBatch], Any] | None,
) -> tuple[tuple[Any, ...], Callable[[ScDataBatch], Any] | None]:
    """Replace DataLoader's positional collate_fn with the raw identity fn."""
    if len(args) < 6:
        return args, collate_fn
    if collate_fn is not None:
        raise TypeError("collate_fn was passed both positionally and by keyword")
    mutable = list(args)
    collate_fn = mutable[5]
    mutable[5] = _identity_raw_collate
    return tuple(mutable), collate_fn


class ScDataLoader(_TorchDataLoader):  # type: ignore[misc, valid-type]
    """Torch DataLoader whose samples are ``(file_id, cell_id)`` pairs.

    Torch still owns sampling, shuffling, ``batch_size``, ``sampler``,
    ``batch_sampler`` and ``drop_last``.  On ``iter(loader)``, this adapter
    materializes all torch batch indices, groups each batch by ``file_id``,
    starts one :meth:`ScDataBank.prefetch` stream per file id, and yields
    decoded dictionaries to ``collate_fn``.

    The dictionary always has ``file_ids`` and ``cell_ids`` in original batch
    order.  For each file id, ``batches[file_id]`` is an existing
    :class:`CellBatch`, ``cells[file_id]`` are the requested cells for that
    file, and ``positions[file_id]`` maps rows back to original batch order.
    When the batch contains a single file id, convenience keys ``file_id``,
    ``batch``, ``cells_in_file`` and ``positions_in_batch`` are also present.
    """

    def __init__(
        self,
        bank: "ScDataBank",
        dataset: Sequence[
            tuple[int, int] | list[int] | NDArray[np.intp] | Mapping[str, int | np.integer[Any]]
        ],
        *args: Any,
        dataset_ids: Mapping[int, DatasetId | str | int | PathLike[str]]
        | Sequence[DatasetId | str | int | PathLike[str]]
        | None = None,
        genes: Iterable[str] | None = None,
        missing: "MissingGenePolicy | str | None" = None,
        prefetch_config: "ScheduledPrefetchConfig | None" = None,
        collate_fn: Callable[[ScDataBatch], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if torch is None:
            raise ModuleNotFoundError("ScDataLoader requires torch. Install torch first.")

        kwargs_collate = kwargs.pop("collate_fn", None)
        if kwargs_collate is not None:
            if collate_fn is not None:
                raise TypeError("collate_fn was passed twice")
            collate_fn = cast(Callable[[ScDataBatch], Any], kwargs_collate)

        args, collate_fn = _extract_positional_collate(args, collate_fn)
        if len(args) < 6:
            kwargs["collate_fn"] = _identity_raw_collate

        super().__init__(dataset, *args, **kwargs)
        self.sc_bank: _SupportsPrefetch = cast(_SupportsPrefetch, bank)
        self.sc_dataset_id: Callable[[int], DatasetId | str | int | PathLike[str]] = (
            _normalize_dataset_ids(dataset_ids)
        )
        self.sc_genes: tuple[str, ...] | None = tuple(genes) if genes is not None else None
        self.sc_missing: MissingGenePolicy | str | None = missing
        self.sc_prefetch_config: ScheduledPrefetchConfig | None = prefetch_config
        self.sc_collate_fn: Callable[[ScDataBatch], Any] = (
            collate_fn if collate_fn is not None else _identity_sc_collate
        )

    def __iter__(self) -> Iterator[Any]:
        plans = self._sc_build_plans()
        if not plans:
            return iter(())

        requests: dict[int, list[NDArray[np.intp]]] = defaultdict(list)
        for plan in plans:
            for part in plan["parts"]:
                requests[part["file_id"]].append(part["cells"])

        prefetchers = {
            file_id: iter(
                self.sc_bank.prefetch(
                    self.sc_dataset_id(file_id),
                    batches,
                    genes=self.sc_genes,
                    missing=self.sc_missing,
                    config=self.sc_prefetch_config,
                )
            )
            for file_id, batches in requests.items()
        }

        def iterator() -> Iterator[Any]:
            for plan in plans:
                batches: dict[int, CellBatch] = {}
                cells: dict[int, NDArray[np.intp]] = {}
                positions: dict[int, NDArray[np.intp]] = {}
                for part in plan["parts"]:
                    file_id = part["file_id"]
                    batches[file_id] = next(prefetchers[file_id])
                    cells[file_id] = part["cells"]
                    positions[file_id] = part["positions"]

                batch: ScDataBatch = {
                    "file_ids": plan["file_ids"],
                    "cell_ids": plan["cell_ids"],
                    "batches": batches,
                    "cells": cells,
                    "positions": positions,
                }
                if len(batches) == 1:
                    file_id = next(iter(batches))
                    batch.update(
                        {
                            "file_id": file_id,
                            "batch": batches[file_id],
                            "cells_in_file": cells[file_id],
                            "positions_in_batch": positions[file_id],
                        }
                    )
                yield self.sc_collate_fn(batch)

        return iter(iterator())

    def _sc_build_plans(self) -> list[_BatchPlan]:
        plans: list[_BatchPlan] = []
        for indices in _iter_index_batches(self):
            samples = [self.dataset[index] for index in indices]
            plans.append(_batch_plan(samples))
        return plans
