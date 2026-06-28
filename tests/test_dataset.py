"""Unit tests for scdata.data metadata types (no store IO)."""
from __future__ import annotations

import numpy as np
import pytest

from scdata.data._dataset import (
    ArrayMeta,
    ArrayOrder,
    ChunkLocation,
    CodecConfigError,
    CodecPipeline,
    DataError,
    DenseDataset,
    DtypeParseError,
    DType,
    SparseDataset,
)


# --------------------------------------------------------------------------
# DType
# --------------------------------------------------------------------------


@pytest.mark.parametrize("zarr,expected", [
    ("<f4", DType.F32),
    ("|u1", DType.U8),
    ("<u8", DType.U64),
    ("<i8", DType.I64),
    ("=f8", DType.F64),
    ("|i1", DType.I8),
    ("<f2", DType.F16),
    ("bf2", DType.BF16),
])
def test_dtype_parse(zarr, expected):
    assert DType.parse(zarr) == expected


@pytest.mark.parametrize("zarr", [">f4", ">i8", ">u1"])
def test_dtype_parse_rejects_big_endian(zarr):
    with pytest.raises(DtypeParseError, match="big-endian"):
        DType.parse(zarr)


@pytest.mark.parametrize("zarr", ["|S3", "<U3", "|O", "|b1", "<c8", "", "xyz"])
def test_dtype_parse_rejects_non_numeric(zarr):
    with pytest.raises(DtypeParseError):
        DType.parse(zarr)


def test_dtype_parse_none_rejected():
    with pytest.raises(DtypeParseError):
        DType.parse(None)


def test_dtype_parse_passthrough():
    assert DType.parse(DType.U16) == DType.U16


def test_dtype_parse_structured_record():
    # zarr wraps single-field arrays as [base, [(field, dtype)]].
    assert DType.parse(["<f4", [("f0", "<f4")]]) == DType.F32


@pytest.mark.parametrize("dtype,expected_size", [
    (DType.U8, 1), (DType.F16, 2), (DType.BF16, 2),
    (DType.F32, 4), (DType.I32, 4), (DType.U32, 4),
    (DType.F64, 8), (DType.I64, 8), (DType.U64, 8),
])
def test_dtype_item_size(dtype, expected_size):
    assert dtype.item_size == expected_size


@pytest.mark.parametrize("dtype,expected", [
    (DType.I32, True), (DType.U32, True), (DType.I64, True), (DType.U64, True),
    (DType.I8, False), (DType.U16, False), (DType.F32, False), (DType.F64, False),
])
def test_dtype_is_csr_index(dtype, expected):
    assert dtype.is_csr_index is expected


@pytest.mark.parametrize("np_dt,expected", [
    (np.float32, DType.F32), (np.float64, DType.F64),
    (np.uint8, DType.U8), (np.int64, DType.I64), (np.int16, DType.I16),
])
def test_dtype_from_numpy(np_dt, expected):
    assert DType.from_numpy(np_dt) == expected


def test_dtype_from_numpy_rejects_complex():
    with pytest.raises(DtypeParseError):
        DType.from_numpy(np.complex64)


# --------------------------------------------------------------------------
# ArrayOrder
# --------------------------------------------------------------------------


def test_array_order_only_c():
    assert [o.value for o in ArrayOrder] == ["C"]


def test_array_order_parse_default():
    assert ArrayOrder.parse(None) == ArrayOrder.C
    assert ArrayOrder.parse("c") == ArrayOrder.C


def test_array_order_rejects_f():
    with pytest.raises(DtypeParseError, match="F-order"):
        ArrayOrder.parse("F")


def test_array_order_rejects_unknown():
    with pytest.raises(DtypeParseError):
        ArrayOrder.parse("Z")


# --------------------------------------------------------------------------
# CodecPipeline
# --------------------------------------------------------------------------


def test_codec_pipeline_uncompressed():
    codec = CodecPipeline.from_zarr(None, None)
    assert codec.is_uncompressed
    assert codec.filters == ()
    assert codec.compressor is None


def test_codec_pipeline_preserves_raw_json():
    filters = [{"id": "shuffle", "elementsize": 4}]
    compressor = {"id": "zstd", "level": 5}
    codec = CodecPipeline.from_zarr(filters, compressor)
    assert codec.filters == ({"id": "shuffle", "elementsize": 4},)
    assert codec.compressor == {"id": "zstd", "level": 5}
    # to_zarr round-trips verbatim (deep copies, no mutation).
    f, c = codec.to_zarr()
    assert f == filters
    assert c == compressor
    assert f is not filters  # defensive copy


def test_codec_pipeline_rejects_non_dict_filter():
    with pytest.raises(CodecConfigError):
        CodecPipeline.from_zarr(["not-a-dict"], None)


def test_codec_pipeline_rejects_non_list_filters():
    with pytest.raises(CodecConfigError):
        CodecPipeline.from_zarr({"id": "shuffle"}, None)


def test_codec_pipeline_rejects_non_dict_compressor():
    with pytest.raises(CodecConfigError):
        CodecPipeline.from_zarr(None, "zstd")


def test_codec_pipeline_immutable():
    codec = CodecPipeline.from_zarr([{"id": "shuffle"}], {"id": "zstd"})
    with pytest.raises(Exception):
        codec.filters = ()  # type: ignore[misc]


# --------------------------------------------------------------------------
# ChunkLocation
# --------------------------------------------------------------------------


def test_chunk_location_negative_offset_rejected():
    with pytest.raises(ValueError):
        ChunkLocation(offset=-1, length=4)


def test_chunk_location_negative_length_rejected():
    with pytest.raises(ValueError):
        ChunkLocation(offset=0, length=-1)


def test_chunk_location_zero_length_ok():
    loc = ChunkLocation(offset=5, length=0)
    assert loc.offset == 5
    assert loc.length == 0


# --------------------------------------------------------------------------
# ArrayMeta
# --------------------------------------------------------------------------


def _chunk_locs(n: int) -> tuple[ChunkLocation, ...]:
    return tuple(ChunkLocation(offset=i * 8, length=8) for i in range(n))


@pytest.fixture
def csr_array_metas() -> tuple[ArrayMeta, ArrayMeta]:
    indices = ArrayMeta(
        shape=(6,), chunk_shape=(6,), dtype=DType.I32, chunks=_chunk_locs(1),
    )
    data = ArrayMeta(
        shape=(6,), chunk_shape=(6,), dtype=DType.F32, chunks=_chunk_locs(1),
    )
    return indices, data


def test_array_meta_chunk_count_validated():
    # shape (4,) chunk (2,) -> grid 2; supply 1 chunk -> reject.
    with pytest.raises(ValueError, match="chunks count"):
        ArrayMeta(
            shape=(4,), chunk_shape=(2,), dtype=DType.F32,
            chunks=_chunk_locs(1),
        )


def test_array_meta_rank_mismatch_rejected():
    with pytest.raises(ValueError, match="rank"):
        ArrayMeta(
            shape=(4, 4), chunk_shape=(2,), dtype=DType.F32,
            chunks=_chunk_locs(4),
        )


def test_array_meta_empty_shape_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        ArrayMeta(shape=(), chunk_shape=(), dtype=DType.F32, chunks=())


def test_array_meta_nonpositive_shape_rejected():
    with pytest.raises(ValueError, match="positive"):
        ArrayMeta(shape=(0,), chunk_shape=(2,), dtype=DType.F32, chunks=())


def test_array_meta_grid_computation():
    meta = ArrayMeta(
        shape=(5, 7), chunk_shape=(2, 3), dtype=DType.F32,
        chunks=_chunk_locs(9),  # ceil(5/2)=3, ceil(7/3)=3 -> 9
    )
    assert meta.ndim == 2
    assert meta.chunk_grid_shape == (3, 3)
    assert meta.num_chunks == 9
    assert meta.item_size == 4


# --------------------------------------------------------------------------
# DenseDataset / SparseDataset validation
# --------------------------------------------------------------------------


def test_dense_dataset_gene_count_mismatch():
    meta = ArrayMeta(
        shape=(4, 4), chunk_shape=(2, 2), dtype=DType.F32, chunks=_chunk_locs(4),
    )
    with pytest.raises(ValueError, match="gene_names"):
        DenseDataset(
            gene_names=("g0", "g1", "g2"), data=meta, num_cells=4, num_genes=4,
        )


def test_dense_dataset_element_count_mismatch():
    meta = ArrayMeta(
        shape=(5, 4), chunk_shape=(2, 2), dtype=DType.F32, chunks=_chunk_locs(6),
    )
    with pytest.raises(ValueError, match="elements"):
        DenseDataset(
            gene_names=tuple(f"g{i}" for i in range(4)), data=meta,
            num_cells=4, num_genes=4,
        )


def test_dense_dataset_1d_requires_genes():
    meta = ArrayMeta(
        shape=(12,), chunk_shape=(5,), dtype=DType.F32, chunks=_chunk_locs(3),
    )
    with pytest.raises(ValueError, match="positive num_genes"):
        DenseDataset(
            gene_names=(), data=meta, num_cells=3, num_genes=0,
        )


def test_dense_dataset_1d_ok():
    meta = ArrayMeta(
        shape=(12,), chunk_shape=(5,), dtype=DType.F32, chunks=_chunk_locs(3),
    )
    ds = DenseDataset(
        gene_names=tuple(f"g{i}" for i in range(4)), data=meta,
        num_cells=3, num_genes=4,
    )
    assert ds.num_cells == 3
    assert ds.num_genes == 4


def test_sparse_dataset_indptr_length_mismatch(csr_array_metas):
    indices, data = csr_array_metas
    with pytest.raises(ValueError, match="indptr length"):
        SparseDataset(
            gene_names=tuple(f"g{i}" for i in range(4)),
            indptr=(0, 2, 4),  # len 3, expected num_cells+1=4
            indices=indices, data=data, index_dtype=DType.I32,
            num_cells=3, num_genes=4,
        )


def test_sparse_dataset_indptr_non_monotonic(csr_array_metas):
    indices, data = csr_array_metas
    with pytest.raises(ValueError, match="monotonic"):
        SparseDataset(
            gene_names=tuple(f"g{i}" for i in range(4)),
            indptr=(0, 4, 2, 6),
            indices=indices, data=data, index_dtype=DType.I32,
            num_cells=3, num_genes=4,
        )


def test_sparse_dataset_indptr_must_start_at_zero(csr_array_metas):
    indices, data = csr_array_metas
    with pytest.raises(ValueError, match="start at 0"):
        SparseDataset(
            gene_names=tuple(f"g{i}" for i in range(4)),
            indptr=(1, 2, 4, 6),
            indices=indices, data=data, index_dtype=DType.I32,
            num_cells=3, num_genes=4,
        )


def test_sparse_dataset_indptr_non_negative(csr_array_metas):
    indices, data = csr_array_metas
    with pytest.raises(ValueError, match="non-negative"):
        SparseDataset(
            gene_names=tuple(f"g{i}" for i in range(4)),
            indptr=(0, -1, 4, 6),
            indices=indices, data=data, index_dtype=DType.I32,
            num_cells=3, num_genes=4,
        )


def test_sparse_dataset_index_dtype_invalid():
    indices = ArrayMeta(
        shape=(6,), chunk_shape=(6,), dtype=DType.F32, chunks=_chunk_locs(1),
    )
    data = ArrayMeta(
        shape=(6,), chunk_shape=(6,), dtype=DType.F32, chunks=_chunk_locs(1),
    )
    with pytest.raises(ValueError, match="CSR index dtype"):
        SparseDataset(
            gene_names=tuple(f"g{i}" for i in range(4)),
            indptr=(0, 2, 4, 6),
            indices=indices, data=data, index_dtype=DType.F32,
            num_cells=3, num_genes=4,
        )


def test_sparse_dataset_nnz_mismatch():
    indices = ArrayMeta(
        shape=(5,), chunk_shape=(5,), dtype=DType.I32, chunks=_chunk_locs(1),
    )
    data = ArrayMeta(
        shape=(6,), chunk_shape=(6,), dtype=DType.F32, chunks=_chunk_locs(1),
    )
    with pytest.raises(ValueError, match="nnz"):
        SparseDataset(
            gene_names=tuple(f"g{i}" for i in range(4)),
            indptr=(0, 2, 4, 6),  # nnz=6
            indices=indices, data=data, index_dtype=DType.I32,
            num_cells=3, num_genes=4,
        )


def test_sparse_dataset_dtype_mismatch():
    indices = ArrayMeta(
        shape=(6,), chunk_shape=(6,), dtype=DType.I64, chunks=_chunk_locs(1),
    )
    data = ArrayMeta(
        shape=(6,), chunk_shape=(6,), dtype=DType.F32, chunks=_chunk_locs(1),
    )
    with pytest.raises(ValueError, match="indices dtype"):
        SparseDataset(
            gene_names=tuple(f"g{i}" for i in range(4)),
            indptr=(0, 2, 4, 6),
            indices=indices, data=data, index_dtype=DType.I32,
            num_cells=3, num_genes=4,
        )


# --------------------------------------------------------------------------
# exception hierarchy
# --------------------------------------------------------------------------


def test_data_error_hierarchy():
    assert issubclass(DtypeParseError, DataError)
    assert issubclass(CodecConfigError, DataError)
    assert issubclass(DataError, ValueError)
