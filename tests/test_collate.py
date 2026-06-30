"""Tests for the default ``stitch_dense_collate``."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scdata.data._cell import CellBatch  # noqa: E402
from scdata.data._collate import stitch_dense_collate  # noqa: E402
from scdata.data._dataloader import ScDataBatch  # noqa: E402


def _batch(cells: list[int], rows: list[list[float]], num_genes: int) -> CellBatch:
    """Build a CellBatch whose rows are ``rows`` (one per cell)."""
    flat = np.asarray(rows, dtype=np.float32).reshape(-1)
    return CellBatch.from_array(
        cells=np.asarray(cells, dtype=np.intp),
        data=flat,
        num_genes=num_genes,
        gene_names=tuple(f"g{i}" for i in range(num_genes)),
    )


def _make_batch_dict(
    file_ids: list[int],
    cell_ids: list[int],
    parts: dict[int, tuple[list[int], CellBatch]],  # file_id -> (positions, cell_batch)
) -> ScDataBatch:
    return {  # type: ignore[return-value]
        "file_ids": np.asarray(file_ids, dtype=np.intp),
        "cell_ids": np.asarray(cell_ids, dtype=np.intp),
        "batches": {fid: cb for fid, (_, cb) in parts.items()},
        "cells": {fid: np.asarray(cb.cells, dtype=np.intp) for fid, (_, cb) in parts.items()},
        "positions": {fid: np.asarray(pos, dtype=np.intp) for fid, (pos, _) in parts.items()},
    }


def test_stitch_single_file_full_batch() -> None:
    cb = _batch([0, 1], [[10.0, 20.0], [30.0, 40.0]], num_genes=2)
    batch = _make_batch_dict(
        file_ids=[0, 0],
        cell_ids=[0, 1],
        parts={0: ([0, 1], cb)},
    )
    out = stitch_dense_collate(batch)
    assert isinstance(out["x"], torch.Tensor)
    assert out["x"].shape == (2, 2)
    assert out["x"].dtype == torch.float32
    np.testing.assert_array_equal(out["x"].numpy(), [[10.0, 20.0], [30.0, 40.0]])
    assert out["file_ids"].tolist() == [0, 0]
    assert out["cell_ids"].tolist() == [0, 1]
    assert out["gene_names"] == ("g0", "g1")


def test_stitch_multi_file_preserves_original_order() -> None:
    # Original batch order: (file0,cell0), (file1,cell0), (file0,cell1)
    # file0 holds rows 0 and 2; file1 holds row 1.
    cb0 = _batch([0, 1], [[1.0, 2.0], [5.0, 6.0]], num_genes=2)
    cb1 = _batch([0], [[3.0, 4.0]], num_genes=2)
    batch = _make_batch_dict(
        file_ids=[0, 1, 0],
        cell_ids=[0, 0, 1],
        parts={0: ([0, 2], cb0), 1: ([1], cb1)},
    )
    out = stitch_dense_collate(batch)
    # Row order must follow the original batch order, not the per-file grouping.
    np.testing.assert_array_equal(
        out["x"].numpy(),
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    )
    assert out["file_ids"].tolist() == [0, 1, 0]
    assert out["cell_ids"].tolist() == [0, 0, 1]


def test_stitch_dtype_follows_cell_batch_data() -> None:
    cb = _batch([0], [[1.0, 2.0, 3.0]], num_genes=3)
    # Force a non-default dtype on the decoded payload.
    cb = CellBatch.from_array(
        cells=np.asarray([0], dtype=np.intp),
        data=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        num_genes=3,
        gene_names=("g0", "g1", "g2"),
    )
    batch = _make_batch_dict(file_ids=[0], cell_ids=[0], parts={0: ([0], cb)})
    out = stitch_dense_collate(batch)
    assert out["x"].dtype == torch.float64
    assert out["x"].shape == (1, 3)


def test_stitch_uses_positions_for_partial_overlap() -> None:
    # Two files each contributing one row, scattered to non-adjacent positions.
    cb0 = _batch([7], [[9.0, 9.0]], num_genes=2)
    cb1 = _batch([3], [[1.0, 1.0]], num_genes=2)
    batch = _make_batch_dict(
        file_ids=[0, 1],
        cell_ids=[7, 3],
        parts={0: ([1], cb0), 1: ([0], cb1)},
    )
    out = stitch_dense_collate(batch)
    np.testing.assert_array_equal(out["x"].numpy(), [[1.0, 1.0], [9.0, 9.0]])
