"""Dataset metadata and access carrier types for scdata stores.

These dataclasses are the Python view of what the Rust databank consumes and
produces.  They are produced by :mod:`scdata.io` when reading a store and
carry every field Rust needs (shape, chunk_shape, dtype, codec pipeline,
payload chunk index) without depending on the Rust extension at runtime.

The :mod:`scdata.data._cell` and :mod:`scdata.data._prefetch` modules add the
request/result carriers for the access and streaming-prefetch paths — also
pure Python, so the execution layer (:mod:`scdata.databank`) is the only place
that touches the Rust extension.  ``CellAccess`` is the input unit for both a
single access call and a prefetch batch; ``CellData`` / ``CellBatch`` are the
decoded outputs of the single-call and streaming paths respectively.
"""

from __future__ import annotations

from scdata.data._cell import CellAccess, CellBatch, CellData
from scdata.data._dataloader import ScDataLoader
from scdata.data._dataset import (
    ArrayMeta,
    ArrayOrder,
    ChunkLocation,
    CodecPipeline,
    DenseDataset,
    DType,
    Dataset,
    DataError,
    DtypeParseError,
    CodecConfigError,
    SparseDataset,
)
from scdata.data._prefetch import PrefetchBatches, PrefetchIterator

__all__ = [
    "ArrayMeta",
    "ArrayOrder",
    "ChunkLocation",
    "CodecPipeline",
    "DenseDataset",
    "DType",
    "Dataset",
    "DataError",
    "DtypeParseError",
    "CodecConfigError",
    "SparseDataset",
    "CellAccess",
    "CellBatch",
    "CellData",
    "ScDataLoader",
    "PrefetchBatches",
    "PrefetchIterator",
]
