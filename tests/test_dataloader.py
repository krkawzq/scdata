"""Tests for the torch-style ScDataLoader adapter."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
import pytest

from scdata.data import CellBatch, ScDataLoader
from scdata.data._dataloader import ScDataBatch


class _PrefetchCall(TypedDict):
    id: str
    batches: list[NDArray[np.intp]]
    genes: tuple[str, ...] | None
    missing: object | None
    config: object | None


class _FakeBank:
    def __init__(self) -> None:
        self.calls: list[_PrefetchCall] = []

    def prefetch(
        self,
        id: str,
        batches: Iterable[NDArray[np.intp]],
        genes: Iterable[str] | None = None,
        missing: object | None = None,
        config: object | None = None,
    ) -> Iterator[CellBatch]:
        materialized = [np.asarray(batch, dtype=np.intp) for batch in batches]
        self.calls.append(
            {
                "id": id,
                "batches": [batch.copy() for batch in materialized],
                "genes": tuple(genes) if genes is not None else None,
                "missing": missing,
                "config": config,
            }
        )

        def iterator() -> Iterator[CellBatch]:
            for cells in materialized:
                data = np.stack([cells * 10, cells * 10 + 1], axis=1).reshape(-1)
                yield CellBatch.from_array(
                    cells=cells,
                    data=data,
                    num_genes=2,
                    gene_names=("g0", "g1"),
                )

        return iterator()


def _patch_fake_torch_base(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_init(
        self: ScDataLoader,
        dataset: list[tuple[int, int]],
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Iterable[int] | None = None,
        batch_sampler: Iterable[list[int]] | None = None,
        drop_last: bool = False,
        collate_fn: object | None = None,
        **kwargs: object,
    ) -> None:
        self.dataset = dataset
        self.collate_fn = collate_fn
        if batch_sampler is not None:
            self.batch_sampler = batch_sampler
            self.sampler = None
            return
        indices = list(sampler) if sampler is not None else list(range(len(dataset)))
        if shuffle:
            indices = list(reversed(indices))
        batches = []
        for i in range(0, len(indices), int(batch_size)):
            batch = indices[i : i + int(batch_size)]
            if len(batch) == int(batch_size) or not drop_last:
                batches.append(batch)
        self.batch_sampler = batches
        self.sampler = None

    monkeypatch.setattr(ScDataLoader.__mro__[1], "__init__", fake_init, raising=False)
    monkeypatch.setattr("scdata.data._dataloader.torch", object())


def test_sc_dataloader_passes_plain_dict_with_existing_cellbatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_torch_base(monkeypatch)

    dataset = [(0, 0), (1, 3), (0, 2), (1, 4), (0, 1)]
    bank = _FakeBank()
    seen: list[ScDataBatch] = []

    def collate(batch: ScDataBatch) -> ScDataBatch:
        seen.append(batch)
        return batch

    loader = ScDataLoader(
        bank,
        dataset,
        batch_size=2,
        dataset_ids=["ds0", "ds1"],
        genes=["g0", "g1"],
        collate_fn=collate,
    )

    iterator = iter(loader)
    assert len(bank.calls) == 2
    assert bank.calls[0]["id"] == "ds0"
    assert [batch.tolist() for batch in bank.calls[0]["batches"]] == [[0], [2], [1]]
    assert bank.calls[1]["id"] == "ds1"
    assert [batch.tolist() for batch in bank.calls[1]["batches"]] == [[3], [4]]

    batches = list(iterator)
    assert seen == batches
    assert all(isinstance(batch, dict) for batch in batches)
    assert all(
        isinstance(cell_batch, CellBatch)
        for batch in batches
        for cell_batch in batch["batches"].values()
    )
    assert batches[0]["file_ids"].tolist() == [0, 1]
    assert batches[0]["cell_ids"].tolist() == [0, 3]
    assert sorted(batches[0]["batches"]) == [0, 1]
    assert batches[0]["cells"][0].tolist() == [0]
    assert batches[0]["cells"][1].tolist() == [3]
    assert batches[2]["file_id"] == 0
    assert batches[2]["batch"].cells.tolist() == [1]
    assert batches[2]["batch"].matrix.tolist() == [[10, 11]]


def test_sc_dataloader_reuses_torch_batch_sampler_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_torch_base(monkeypatch)

    dataset = [(0, 0), (0, 1), (0, 2), (0, 3)]
    bank = _FakeBank()

    loader = ScDataLoader(
        bank,
        dataset,
        batch_sampler=[[2, 0], [3]],
        dataset_ids={0: "ds0"},
    )

    batches = list(iter(loader))
    assert [batch["cell_ids"].tolist() for batch in batches] == [[2, 0], [3]]
    assert [batch["batch"].cells.tolist() for batch in batches] == [[2, 0], [3]]


def test_sc_dataloader_reports_missing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scdata.data._dataloader.torch", None)

    with pytest.raises(ModuleNotFoundError, match="requires torch"):
        ScDataLoader(_FakeBank(), [], dataset_ids=[])
