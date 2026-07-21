"""Default collate functions for :class:`~scdata.data.ScDataLoader`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from scdata.data._dataloader import ScDataBatch

__all__ = ["stitch_dense_collate"]


def stitch_dense_collate(batch: "ScDataBatch") -> dict[str, Any]:
    """Return the decoded dense batch as a torch tensor plus row metadata.

    Normal :class:`ScDataLoader` output includes ``"batch"`` already decoded in
    original row order.  That path only validates metadata and keeps the numpy
    matrix view zero-copy.  The legacy per-file fallback validates that every
    part has the same schema and scatters a complete, non-overlapping partition
    of output rows.
    """
    import torch

    file_ids = np.asarray(batch["file_ids"])
    cell_ids = np.asarray(batch["cell_ids"])
    if file_ids.ndim != 1:
        raise ValueError(f"file_ids must be 1D, got {file_ids.ndim}D")
    if cell_ids.ndim != 1:
        raise ValueError(f"cell_ids must be 1D, got {cell_ids.ndim}D")
    if file_ids.shape[0] != cell_ids.shape[0]:
        raise ValueError(
            "file_ids and cell_ids must have the same length, got "
            f"{file_ids.shape[0]} and {cell_ids.shape[0]}"
        )

    decoded = batch.get("batch")
    if decoded is not None:
        if decoded.num_cells != cell_ids.shape[0]:
            raise ValueError(
                "decoded batch row count must match cell_ids length, got "
                f"{decoded.num_cells} and {cell_ids.shape[0]}"
            )
        if not np.array_equal(decoded.cells, cell_ids):
            raise ValueError("decoded batch cells must match cell_ids in row order")
        # ``to_numpy`` is a reshape view for a decoded CellBatch; do not copy it.
        out = decoded.to_numpy()
        gene_names = decoded.var_names
    else:
        batches = batch["batches"]
        positions_by_file = batch["positions"]
        cells_by_file = batch["cells"]
        if not batches:
            raise ValueError("cannot stitch an empty batch without a decoded batch schema")
        if set(batches) != set(positions_by_file) or set(batches) != set(cells_by_file):
            raise ValueError("batches, positions, and cells must have identical file_id keys")

        first = next(iter(batches.values()))
        num_genes = first.num_genes
        dtype = first.data.dtype
        gene_names = first.var_names
        out = np.empty((cell_ids.shape[0], num_genes), dtype=dtype)
        occupied = np.zeros(cell_ids.shape[0], dtype=bool)

        for file_id, cell_batch in batches.items():
            positions = np.asarray(positions_by_file[file_id])
            expected_cells = np.asarray(cells_by_file[file_id])
            if positions.ndim != 1:
                raise ValueError(f"positions[{file_id}] must be 1D, got {positions.ndim}D")
            if not np.issubdtype(positions.dtype, np.integer):
                raise ValueError(f"positions[{file_id}] must have an integer dtype")
            if cell_batch.num_cells != positions.shape[0]:
                raise ValueError(
                    f"batches[{file_id}] row count {cell_batch.num_cells} does not match "
                    f"positions length {positions.shape[0]}"
                )
            if cell_batch.num_genes != num_genes:
                raise ValueError(
                    f"batches[{file_id}] num_genes {cell_batch.num_genes} does not match "
                    f"the first batch schema {num_genes}"
                )
            if cell_batch.data.dtype != dtype:
                raise ValueError(
                    f"batches[{file_id}] dtype {cell_batch.data.dtype} does not match "
                    f"the first batch dtype {dtype}"
                )
            if cell_batch.var_names != gene_names:
                raise ValueError(
                    f"batches[{file_id}] gene_names do not match the first batch schema"
                )
            if expected_cells.ndim != 1 or not np.array_equal(cell_batch.cells, expected_cells):
                raise ValueError(f"batches[{file_id}] cells must match cells[{file_id}]")
            if np.any(positions < 0) or np.any(positions >= cell_ids.shape[0]):
                raise ValueError(f"positions[{file_id}] contain an out-of-range row")
            if np.unique(positions).size != positions.size:
                raise ValueError(f"positions[{file_id}] must not repeat an output row")
            if np.any(occupied[positions]):
                raise ValueError(
                    "positions must assign each output row exactly once (found duplicate)"
                )
            if not np.all(file_ids[positions] == file_id):
                raise ValueError(f"positions[{file_id}] do not match file_ids")
            if not np.array_equal(cell_ids[positions], cell_batch.cells):
                raise ValueError(f"positions[{file_id}] do not match cell_ids")
            occupied[positions] = True
            out[positions] = cell_batch.to_numpy()

        if not np.all(occupied):
            raise ValueError(
                "positions must assign each output row exactly once (found missing row)"
            )

    return {
        "x": torch.from_numpy(out),
        "file_ids": torch.as_tensor(file_ids, dtype=torch.long),
        "cell_ids": torch.as_tensor(cell_ids, dtype=torch.long),
        "gene_names": gene_names,
    }
