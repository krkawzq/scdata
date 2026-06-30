"""Dataset metadata and access carrier types for scdata stores.

These dataclasses are the Python view of what the Rust databank consumes and
produces.  They are produced by :mod:`scdata.io` when reading a store and
carry every field Rust needs (shape, chunk_shape, dtype, codec pipeline,
chunk locations) without depending on the Rust extension at runtime.

:mod:`scdata.data._cell` and :mod:`scdata.data._prefetch` add the
request/result carriers for the access and streaming-prefetch paths.
``CellAccess`` is the input unit for both a single access call and a prefetch
batch; ``CellData`` / ``CellBatch`` are the decoded outputs of the single-call
and streaming paths respectively.
"""

from __future__ import annotations

from scdata.data._cell import CellAccess, CellBatch, CellData
from scdata.data._collate import stitch_dense_collate
from scdata.data._dataloader import ScDataLoader
from scdata.data._dataset import (
    ArrayMeta,
    ArrayOrder,
    ChunkLocation,
    CodecPipeline,
    DenseDataset,
    DType,
    Dataset,
    DatasetCollection,
    DataError,
    DtypeParseError,
    CodecConfigError,
    SparseDataset,
)
from scdata.data._index import CellIndexDataset
from scdata.data._prefetch import PrefetchBatches, PrefetchIterator
from scdata.data._stats import BankConfigSummary, LoaderStats

__all__ = [
    "ArrayMeta",
    "ArrayOrder",
    "ChunkLocation",
    "CodecPipeline",
    "DenseDataset",
    "DType",
    "Dataset",
    "DatasetCollection",
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
    "CellIndexDataset",
    "stitch_dense_collate",
    "LoaderStats",
    "BankConfigSummary",
]
