"""Read-path regression tests for scdata stores.

Covers dense/sparse, directory vs ZIP_STORED, codec variants, string-array
gene names (``|S`` / ``<U`` / ``|O``+VLenUTF8), multi-chunk compressed
metadata arrays, and every documented error path.
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

from scdata.data._dataset import ArrayOrder, DenseDataset, DType, SparseDataset
from scdata.io._launch import StoreError, launch


# --------------------------------------------------------------------------
# happy path: dense
# --------------------------------------------------------------------------


def test_dense_dir_uncompressed(dense_store_factory):
    root = dense_store_factory(
        "dense", (3, 4), (2, 2), np.float32, None,
        [f"gene{i}" for i in range(4)],
    )
    ds = launch(root)
    assert isinstance(ds, DenseDataset)
    assert ds.num_cells == 3
    assert ds.num_genes == 4
    assert ds.data.shape == (3, 4)
    assert ds.data.chunk_grid_shape == (2, 2)
    assert ds.data.num_chunks == 4
    assert ds.data.dtype == DType.F32
    assert ds.data.order == ArrayOrder.C
    assert tuple(ds.gene_names) == tuple(f"gene{i}" for i in range(4))
    assert ds.data.payload_path == "X/payload.bin"
    assert len(ds.data.chunks) == 4


def test_dense_dir_zstd(dense_store_factory, codec_configs):
    root = dense_store_factory(
        "dense_zstd", (5, 10), (2, 4), np.float64, codec_configs["zstd"],
        [f"g{i}" for i in range(10)],
    )
    ds = launch(root)
    assert isinstance(ds, DenseDataset)
    # ceil(5/2)=3, ceil(10/4)=3
    assert ds.data.chunk_grid_shape == (3, 3)
    assert ds.data.num_chunks == 9
    assert ds.data.dtype == DType.F64


def test_dense_1d(dense_store_factory):
    root = dense_store_factory(
        "dense_1d", (12,), (5,), np.float32, None,
        [f"g{i}" for i in range(4)],
    )
    ds = launch(root)
    assert isinstance(ds, DenseDataset)
    assert ds.num_cells == 3
    assert ds.num_genes == 4
    assert ds.data.shape == (12,)


def test_dense_multichunk_grid(dense_store_factory):
    root = dense_store_factory(
        "dense_grid", (5, 7), (2, 3), np.int32, None,
        [f"g{i}" for i in range(7)],
    )
    ds = launch(root)
    assert isinstance(ds, DenseDataset)
    # ceil(5/2)=3, ceil(7/3)=3 => 9 chunks
    assert ds.data.chunk_grid_shape == (3, 3)
    assert ds.data.num_chunks == 9
    assert ds.data.dtype == DType.I32


# --------------------------------------------------------------------------
# happy path: sparse CSR
# --------------------------------------------------------------------------


def test_sparse_csr_zstd(sparse_store_factory, codec_configs):
    root = sparse_store_factory(
        "sparse", 4, 6, 3, np.float32, np.int32, codec_configs["zstd"],
        [f"g{i}" for i in range(6)],
    )
    ds = launch(root)
    assert isinstance(ds, SparseDataset)
    assert ds.num_cells == 4
    assert ds.num_genes == 6
    assert len(ds.indptr) == 5
    assert ds.indptr == (0, 3, 6, 9, 12)
    assert ds.index_dtype == DType.I32
    assert ds.indices.dtype == DType.I32
    assert ds.indices.shape == (12,)
    assert ds.data.shape == (12,)
    assert tuple(ds.gene_names) == tuple(f"g{i}" for i in range(6))


def test_sparse_csr_i64_indices(sparse_store_factory, codec_configs):
    root = sparse_store_factory(
        "sparse_i64", 3, 5, 2, np.float64, np.int64, codec_configs["blosc"],
        [f"g{i}" for i in range(5)],
    )
    ds = launch(root)
    assert isinstance(ds, SparseDataset)
    assert ds.index_dtype == DType.I64
    assert ds.indices.dtype == DType.I64
    assert ds.data.dtype == DType.F64
    assert ds.indptr == (0, 2, 4, 6)


# --------------------------------------------------------------------------
# ZIP_STORED container
# --------------------------------------------------------------------------


def test_zip_store(dense_store_factory, zip_store_factory, codec_configs):
    src = dense_store_factory(
        "zip_src", (3, 4), (2, 2), np.float32, codec_configs["zstd"],
        [f"g{i}" for i in range(4)],
    )
    zip_path = zip_store_factory(src)
    ds = launch(zip_path)
    assert isinstance(ds, DenseDataset)
    assert ds.num_cells == 3
    assert ds.num_genes == 4
    assert tuple(ds.gene_names) == tuple(f"g{i}" for i in range(4))


def test_zip_uncompressed(dense_store_factory, zip_store_factory):
    src = dense_store_factory(
        "zip_unc_src", (3, 4), (2, 2), np.float32, None,
        [f"g{i}" for i in range(4)],
    )
    zip_path = zip_store_factory(src, "store_unc.zarr.zip")
    ds = launch(zip_path)
    assert isinstance(ds, DenseDataset)
    assert ds.data.num_chunks == 4


# --------------------------------------------------------------------------
# gene names: all string dtype variants
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "var_kind", "names", "var_compressor_name", "var_chunk_shape"),
    [
        ("fixed_S", "S", ["AAA", "BBB", "CCC", "DDD"], None, (16,)),
        ("unicode_U", "U", ["AAA", "BBB", "CCC", "DDD"], None, (16,)),
        (
            "object_vlen",
            "O",
            ["gene_alpha", "gene_beta", "gene_gamma", "gene_delta"],
            None,
            (16,),
        ),
        (
            "object_vlen_compressed_multichunk",
            "O",
            [f"gene_{i:03d}" for i in range(40)],
            "zstd",
            (16,),
        ),
    ],
)
def test_gene_name_string_variants(
    dense_store_factory,
    codec_configs,
    case: str,
    var_kind: str,
    names: list[str],
    var_compressor_name: str | None,
    var_chunk_shape,
):
    shape = (3, len(names))
    var_compressor = None if var_compressor_name is None else codec_configs[var_compressor_name]
    root = dense_store_factory(
        f"var_{case}", shape, (2, min(10, len(names))), np.float32, None, names,
        var_kind=var_kind,
        var_compressor=var_compressor,
        var_chunk_shape=var_chunk_shape,
    )
    ds = launch(root)
    assert list(ds.gene_names) == names


# --------------------------------------------------------------------------
# multi-chunk compressed metadata (B1 regression)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "compressor_name"),
    [
        ("zstd", "zstd"),
        ("blosc", "blosc"),
        ("gzip", "gzip"),
    ],
)
def test_multichunk_compressed_var_decodes_per_chunk(
    dense_store_factory,
    codec_configs,
    case: str,
    compressor_name: str,
):
    """Each metadata chunk is decoded independently, not as one concatenated block."""
    names = [f"gene_{i:02d}" for i in range(40)]
    root = dense_store_factory(
        f"var_{case}_multi", (3, 40), (2, 10), np.float32, None, names,
        var_kind="S", var_compressor=codec_configs[compressor_name], var_chunk_shape=(16,),
    )
    ds = launch(root)
    assert list(ds.gene_names) == names


# --------------------------------------------------------------------------
# error paths
# --------------------------------------------------------------------------


def test_missing_scdata_index(dense_store_factory):
    root = dense_store_factory(
        "no_idx", (3, 4), (2, 2), np.float32, None,
        [f"g{i}" for i in range(4)],
    )
    zattrs = json.loads((root / "X" / ".zattrs").read_text())
    zattrs.pop("scdata")
    (root / "X" / ".zattrs").write_text(json.dumps(zattrs))
    with pytest.raises(StoreError, match="scdata"):
        launch(root)


def test_chunk_count_mismatch(dense_store_factory):
    root = dense_store_factory(
        "mismatch", (3, 4), (2, 2), np.float32, None,
        [f"g{i}" for i in range(4)],
    )
    zattrs = json.loads((root / "X" / ".zattrs").read_text())
    zattrs["scdata"]["offsets"].pop()
    zattrs["scdata"]["lengths"].pop()
    (root / "X" / ".zattrs").write_text(json.dumps(zattrs))
    with pytest.raises(StoreError, match="chunk"):
        launch(root)


def test_negative_chunk_location_is_store_error(dense_store_factory):
    root = dense_store_factory(
        "negative_loc", (3, 4), (2, 2), np.float32, None,
        [f"g{i}" for i in range(4)],
    )
    zattrs = json.loads((root / "X" / ".zattrs").read_text())
    zattrs["scdata"]["offsets"][0] = -1
    (root / "X" / ".zattrs").write_text(json.dumps(zattrs))
    with pytest.raises(StoreError, match="invalid chunk location"):
        launch(root)


def test_payload_range_out_of_bounds_rejected(dense_store_factory):
    root = dense_store_factory(
        "bad_range", (3, 4), (2, 2), np.float32, None,
        [f"g{i}" for i in range(4)],
    )
    zattrs = json.loads((root / "X" / ".zattrs").read_text())
    zattrs["scdata"]["lengths"][0] = 10_000
    (root / "X" / ".zattrs").write_text(json.dumps(zattrs))
    with pytest.raises(StoreError, match="exceeds payload size"):
        launch(root)


def test_shape_entries_must_be_json_integers(dense_store_factory):
    root = dense_store_factory(
        "float_shape", (3, 4), (2, 2), np.float32, None,
        [f"g{i}" for i in range(4)],
    )
    zarray = json.loads((root / "X" / ".zarray").read_text())
    zarray["shape"] = [3.5, 4]
    (root / "X" / ".zarray").write_text(json.dumps(zarray))
    with pytest.raises(StoreError, match="JSON integers"):
        launch(root)


def test_csc_rejected(sparse_store_factory, codec_configs):
    root = sparse_store_factory(
        "csc", 4, 6, 3, np.float32, np.int32, codec_configs["zstd"],
        [f"g{i}" for i in range(6)],
    )
    zattrs = json.loads((root / "X" / ".zattrs").read_text())
    zattrs["encoding-type"] = "CSC"
    (root / "X" / ".zattrs").write_text(json.dumps(zattrs))
    with pytest.raises(StoreError, match="CSC"):
        launch(root)


def test_bigendian_rejected(dense_store_factory):
    root = dense_store_factory(
        "be", (3, 4), (2, 2), np.float32, None,
        [f"g{i}" for i in range(4)],
    )
    zarray = json.loads((root / "X" / ".zarray").read_text())
    zarray["dtype"] = ">f4"
    (root / "X" / ".zarray").write_text(json.dumps(zarray))
    # launch() must surface this as a StoreError, not leak a DtypeParseError.
    with pytest.raises(StoreError):
        launch(root)


def test_forder_rejected(dense_store_factory):
    root = dense_store_factory(
        "forder", (3, 4), (2, 2), np.float32, None,
        [f"g{i}" for i in range(4)],
    )
    zarray = json.loads((root / "X" / ".zarray").read_text())
    zarray["order"] = "F"
    (root / "X" / ".zarray").write_text(json.dumps(zarray))
    with pytest.raises(StoreError, match="F-order"):
        launch(root)


def test_missing_payload(dense_store_factory):
    root = dense_store_factory(
        "no_payload", (3, 4), (2, 2), np.float32, None,
        [f"g{i}" for i in range(4)],
    )
    (root / "X" / "payload.bin").unlink()
    with pytest.raises(StoreError, match="payload"):
        launch(root)


def test_missing_var_group(dense_store_factory):
    root = dense_store_factory(
        "no_var", (3, 4), (2, 2), np.float32, None,
        [f"g{i}" for i in range(4)],
    )
    shutil.rmtree(root / "var")
    with pytest.raises(StoreError, match="var"):
        launch(root)


def test_var_index_name_from_anndata_attrs(dense_store_factory):
    names = ["g0", "g1", "g2", "g3"]
    root = dense_store_factory(
        "custom_var_index", (3, 4), (2, 2), np.float32, None, names,
    )
    (root / "var" / "_index").rename(root / "var" / "gene_ids")
    (root / "var" / ".zattrs").write_text(json.dumps({
        "column-order": [],
        "_index": "gene_ids",
        "encoding-type": "dataframe",
        "encoding-version": "0.2.0",
    }))
    ds = launch(root)
    assert tuple(ds.gene_names) == tuple(names)


def test_v3_store_rejected(tmp_path: Path):
    root = tmp_path / "v3store"
    root.mkdir()
    (root / ".zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group"}))
    with pytest.raises(StoreError, match="v3"):
        launch(root)


def test_nonexistent_path(tmp_path: Path):
    with pytest.raises(StoreError, match="does not exist"):
        launch(tmp_path / "nope.zarr")


def test_csr_index_dtype_mismatch(sparse_store_factory):
    """indices dtype must be a valid CSR index type (i32/u32/i64/u64)."""
    root = sparse_store_factory(
        "idx_dtype", 4, 6, 3, np.float32, np.int32, None,
        [f"g{i}" for i in range(6)],
    )
    # Forge indices as f32 (invalid CSR index).
    zarray = json.loads((root / "X" / "indices" / ".zarray").read_text())
    zarray["dtype"] = "<f4"
    (root / "X" / "indices" / ".zarray").write_text(json.dumps(zarray))
    with pytest.raises(StoreError):
        launch(root)


def test_indptr_float_dtype_rejected(sparse_store_factory):
    root = sparse_store_factory(
        "indptr_f", 4, 6, 3, np.float32, np.int32, None,
        [f"g{i}" for i in range(6)],
    )
    zarray = json.loads((root / "X" / "indptr" / ".zarray").read_text())
    zarray["dtype"] = "<f8"
    (root / "X" / "indptr" / ".zarray").write_text(json.dumps(zarray))
    with pytest.raises(StoreError, match="integer"):
        launch(root)


def test_sparse_shape_attr_must_match_indptr_and_var(sparse_store_factory):
    root = sparse_store_factory(
        "sparse_shape", 4, 6, 3, np.float32, np.int32, None,
        [f"g{i}" for i in range(6)],
    )
    zattrs = json.loads((root / "X" / ".zattrs").read_text())
    zattrs["shape"] = [4, 99]
    (root / "X" / ".zattrs").write_text(json.dumps(zattrs))
    with pytest.raises(StoreError, match="99 genes"):
        launch(root)


# --------------------------------------------------------------------------
# store abstraction / container edge cases
# --------------------------------------------------------------------------


def test_zip_bad_archive(tmp_path: Path):
    bad = tmp_path / "bad.zarr.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(StoreError, match="zip"):
        launch(bad)


def test_zip_must_be_stored(tmp_path: Path, dense_store_factory):
    src = dense_store_factory(
        "zip_deflated_src", (3, 4), (2, 2), np.float32, None,
        [f"g{i}" for i in range(4)],
    )
    zip_path = tmp_path / "deflated.zarr.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(src).as_posix())
    with pytest.raises(StoreError, match="ZIP_STORED"):
        launch(zip_path)


def test_directory_store_path_traversal(dense_store_factory):
    """A payload path with .. must not escape the store root."""
    root = dense_store_factory(
        "traversal", (3, 4), (2, 2), np.float32, None,
        [f"g{i}" for i in range(4)],
    )
    zattrs = json.loads((root / "X" / ".zattrs").read_text())
    # Point the payload outside the store.
    zattrs["scdata"]["payload"] = "../../../etc/passwd"
    (root / "X" / ".zattrs").write_text(json.dumps(zattrs))
    with pytest.raises(StoreError, match="escapes store root"):
        launch(root)


def test_directory_store_is_dir_not_file(tmp_path: Path):
    with pytest.raises(StoreError, match="does not exist|not a"):
        launch(tmp_path)  # a directory with no .zgroup
