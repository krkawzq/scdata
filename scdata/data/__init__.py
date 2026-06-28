"""Dataset metadata types for scdata stores.

These dataclasses are the Python view of what the Rust databank consumes.
They are produced by :mod:`scdata.io` when reading a store and carry every
field Rust needs (shape, chunk_shape, dtype, codec pipeline, payload chunk
index) without depending on the Rust extension at runtime.
"""

from __future__ import annotations

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
]
