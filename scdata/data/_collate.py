"""Default collate functions for :class:`~scdata.data.ScDataLoader`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from scdata.data._dataloader import ScDataBatch

__all__ = ["stitch_dense_collate"]


def stitch_dense_collate(batch: "ScDataBatch") -> dict[str, Any]:
    """Return the decoded dense batch as a torch tensor plus row metadata."""
    import torch

    cell_ids = batch["cell_ids"]
    decoded = batch.get("batch")
    if decoded is None:
        first = next(iter(batch["batches"].values()))
        out = np.empty((len(cell_ids), first.num_genes), dtype=first.data.dtype)
        for file_id, cell_batch in batch["batches"].items():
            out[batch["positions"][file_id]] = cell_batch.to_numpy()
        gene_names = first.var_names
    else:
        out = decoded.to_numpy()
        gene_names = decoded.var_names
    return {
        "x": torch.from_numpy(out),
        "file_ids": torch.as_tensor(batch["file_ids"], dtype=torch.long),
        "cell_ids": torch.as_tensor(cell_ids, dtype=torch.long),
        "gene_names": gene_names,
    }
