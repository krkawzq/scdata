"""Regression tests for the AnnData zarr v3 -> scdata databank bridge."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from scdata import ScDataBank
from scdata.data import DenseDataset, SparseDataset
from scdata.io import AnnDataZarrZipConverter, launch, launch_all, read_zarr, write_zarr
from scdata.io._launch import StoreError

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sp = pytest.importorskip("scipy.sparse")
pytest.importorskip("zarr")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


@pytest.fixture
def dense_adata():
    return ad.AnnData(
        X=np.arange(12, dtype=np.float32).reshape(3, 4),
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
    )


@pytest.fixture
def sparse_adata():
    matrix = sp.csr_matrix(
        np.array(
            [
                [1, 0, 2, 0],
                [0, 3, 0, 4],
                [5, 0, 0, 6],
            ],
            dtype=np.float32,
        )
    )
    adata = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
    )
    return adata, matrix


def _registered_matrix(ds):
    bank = ScDataBank()
    did = bank.register(ds)
    try:
        cells = list(range(bank.dataset_num_cells(did)))
        out = np.asarray(bank.load(did, cells))
        return out.reshape(len(cells), bank.dataset_num_genes(did))
    finally:
        bank.unregister(did)


def test_write_zarr_dense2d_dir_registers_with_databank(tmp_path: Path, dense_adata) -> None:
    root = write_zarr(
        dense_adata,
        tmp_path / "dense2d.zarr",
        format="dense2d",
        chunk_size=(2, 2),
        store="dir",
    )

    ds = launch(root)

    assert isinstance(ds, DenseDataset)
    assert ds.data.shape == (3, 4)
    assert ds.data.num_chunks == 4
    assert np.array_equal(_registered_matrix(ds), np.asarray(dense_adata.X))


def test_write_zarr_dense_big_endian_normalizes_for_rust(tmp_path: Path) -> None:
    data = np.arange(12, dtype=">f4").reshape(3, 4)
    adata = ad.AnnData(
        X=data,
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
    )
    root = write_zarr(
        adata,
        tmp_path / "dense_be.zarr",
        format="dense2d",
        chunk_size=(2, 2),
        store="dir",
    )

    ds = launch(root)

    assert np.array_equal(_registered_matrix(ds), np.asarray(data, dtype=np.float32))


def test_write_zarr_dense1d_zip_registers_with_databank(tmp_path: Path, dense_adata) -> None:
    root = write_zarr(
        dense_adata,
        tmp_path / "dense1d.zarr.zip",
        format="dense1d",
        chunk_size=(8,),
        store="zip",
    )

    ds = launch(root)

    assert isinstance(ds, DenseDataset)
    assert ds.data.shape == (12,)
    assert ds.data.chunk_file_paths
    assert set(ds.data.chunk_file_paths) == {str(root)}
    with zipfile.ZipFile(root) as zf:
        names = zf.namelist()
    assert len(names) == len(set(names))
    assert np.array_equal(_registered_matrix(ds), np.asarray(dense_adata.X))


def test_write_zarr_sparse_rectilinear_registers_with_databank(
    tmp_path: Path, sparse_adata
) -> None:
    adata, matrix = sparse_adata
    root = write_zarr(
        adata,
        tmp_path / "sparse.zarr",
        format="sparse",
        chunk_size=(4,),
        align_cells=True,
        store="dir",
    )

    ds = launch(root)
    data_meta = _read_json(root / "X" / "data" / "zarr.json")

    assert isinstance(ds, SparseDataset)
    assert ds.indices.variable_chunks
    assert ds.data.variable_chunks
    assert data_meta["codecs"][1] == {
        "name": "blosc",
        "configuration": {
            "typesize": 4,
            "cname": "lz4",
            "clevel": 5,
            "shuffle": "shuffle",
            "blocksize": 0,
        },
    }
    assert ds.data.codec.compressor == {
        "id": "blosc",
        "cname": "lz4",
        "clevel": 5,
        "shuffle": 1,
        "blocksize": 0,
        "typesize": 4,
    }
    assert tuple(np.asarray(ds.indptr).tolist()) == tuple(matrix.indptr.tolist())
    assert np.array_equal(_registered_matrix(ds), matrix.toarray())


def test_write_zarr_rejects_unknown_compressor(tmp_path: Path, sparse_adata) -> None:
    adata, matrix = sparse_adata
    with pytest.raises(StoreError, match="unsupported compressor"):
        write_zarr(
            adata,
            tmp_path / "sparse_alias.zarr",
            format="sparse",
            chunk_size=(4,),
            align_cells=True,
            store="dir",
            compressor="blocs.lz4.level5",
        )


def test_write_zarr_sparse_zero_nnz_round_trip(tmp_path: Path) -> None:
    matrix = sp.csr_matrix((3, 4), dtype=np.float32)
    adata = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
    )
    root = write_zarr(
        adata,
        tmp_path / "sparse_zero.zarr",
        format="sparse",
        chunk_size=(4,),
        align_cells=True,
        store="dir",
    )

    ds = launch(root)
    loaded = read_zarr(root)

    assert isinstance(ds, SparseDataset)
    assert ds.nnz == 0
    assert ds.indices.shape == (0,)
    assert ds.indices.num_chunks == 0
    assert loaded.X.shape == (3, 4)
    assert loaded.X.nnz == 0
    assert np.array_equal(_registered_matrix(ds), matrix.toarray())


def test_read_zarr_reshapes_dense1d(tmp_path: Path, dense_adata) -> None:
    root = write_zarr(
        dense_adata,
        tmp_path / "dense1d_read.zarr",
        format="dense1d",
        chunk_size=(8,),
        store="dir",
    )

    loaded = read_zarr(root)

    assert np.array_equal(np.asarray(loaded.X), np.asarray(dense_adata.X))


def test_launch_zlib_v3_codec_registers_with_databank(tmp_path: Path, dense_adata) -> None:
    from numcodecs import Zlib

    root = write_zarr(
        dense_adata,
        tmp_path / "zlib_dense.zarr",
        format="dense2d",
        chunk_size=(3, 4),
        store="dir",
    )
    raw = np.ascontiguousarray(np.asarray(dense_adata.X)).tobytes()
    (root / "X" / "c" / "0" / "0").write_bytes(Zlib(level=1).encode(raw))
    meta = _read_json(root / "X" / "zarr.json")
    meta["codecs"] = [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "zlib", "configuration": {"level": 1}},
    ]
    _write_json(root / "X" / "zarr.json", meta)

    ds = launch(root)

    assert ds.data.codec.compressor == {"id": "zlib", "level": 1}
    assert np.array_equal(_registered_matrix(ds), np.asarray(dense_adata.X))


def test_launch_lz4_v3_codec_registers_with_databank(tmp_path: Path, dense_adata) -> None:
    from numcodecs import LZ4

    root = write_zarr(
        dense_adata,
        tmp_path / "lz4_dense.zarr",
        format="dense2d",
        chunk_size=(3, 4),
        store="dir",
        compressor=None,
    )
    raw = np.ascontiguousarray(np.asarray(dense_adata.X)).tobytes()
    (root / "X" / "c" / "0" / "0").write_bytes(LZ4(acceleration=1).encode(raw))
    meta = _read_json(root / "X" / "zarr.json")
    meta["codecs"] = [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "lz4", "configuration": {"acceleration": 1}},
    ]
    _write_json(root / "X" / "zarr.json", meta)

    ds = launch(root)

    assert ds.data.codec.compressor == {"id": "lz4", "acceleration": 1}
    assert np.array_equal(_registered_matrix(ds), np.asarray(dense_adata.X))


def test_launch_multiple_v3_codecs_preserves_decode_order(
    tmp_path: Path, dense_adata
) -> None:
    from numcodecs import GZip, Zlib, Zstd

    root = write_zarr(
        dense_adata,
        tmp_path / "multi_codec_dense.zarr",
        format="dense2d",
        chunk_size=(3, 4),
        store="dir",
        compressor=None,
    )
    raw = np.ascontiguousarray(np.asarray(dense_adata.X)).tobytes()
    encoded = GZip(level=1).encode(Zstd(level=1).encode(Zlib(level=1).encode(raw)))
    (root / "X" / "c" / "0" / "0").write_bytes(encoded)
    meta = _read_json(root / "X" / "zarr.json")
    meta["codecs"] = [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "zlib", "configuration": {"level": 1}},
        {"name": "zstd", "configuration": {"level": 1, "checksum": False}},
        {"name": "gzip", "configuration": {"level": 1}},
    ]
    _write_json(root / "X" / "zarr.json", meta)

    ds = launch(root)

    assert ds.data.codec.compressor == {"id": "gzip", "level": 1}
    assert ds.data.codec.filters == (
        {"id": "zlib", "level": 1},
        {"id": "zstd", "level": 1, "checksum": False},
    )
    assert np.array_equal(_registered_matrix(ds), np.asarray(dense_adata.X))


def test_dense_layer_launch_read_and_register_all(tmp_path: Path, dense_adata) -> None:
    layer = np.asarray(dense_adata.X, dtype=np.float32) + 100
    dense_adata.layers["counts"] = layer
    root = write_zarr(
        dense_adata,
        tmp_path / "dense_layer.zarr",
        format="dense1d",
        layer_format="auto",
        chunk_size=(8,),
        store="dir",
    )

    layer_ds = launch(root, layer="counts")
    same_layer_ds = launch(root, matrix="layers/counts")
    datasets = launch_all(root)
    loaded = read_zarr(root)

    assert isinstance(layer_ds, DenseDataset)
    assert isinstance(same_layer_ds, DenseDataset)
    assert layer_ds.data.shape == (12,)
    assert datasets.keys() == ("X", "layers/counts")
    assert np.array_equal(np.asarray(loaded.layers["counts"]), layer)

    bank = ScDataBank()
    ids = bank.register_all(datasets)
    try:
        out = np.asarray(bank.load(ids["layers/counts"], [0, 1, 2]))
        assert np.array_equal(out.reshape(3, 4), layer)
    finally:
        bank.unregister_all(ids)


def test_sparse_layer_launch_and_read_rectilinear(tmp_path: Path, dense_adata) -> None:
    layer = sp.csr_matrix(
        np.array(
            [
                [1, 0, 2, 0],
                [0, 3, 0, 4],
                [5, 0, 0, 6],
            ],
            dtype=np.float32,
        )
    )
    dense_adata.layers["counts"] = layer
    root = write_zarr(
        dense_adata,
        tmp_path / "sparse_layer.zarr.zip",
        format="dense1d",
        layer_format="auto",
        chunk_size=(4,),
        align_cells=True,
        store="zip",
    )

    ds = launch(root, layer="counts")
    loaded = read_zarr(root)

    assert isinstance(ds, SparseDataset)
    assert ds.indices.variable_chunks
    assert ds.data.variable_chunks
    assert np.array_equal(_registered_matrix(ds), layer.toarray())
    assert np.array_equal(loaded.layers["counts"].toarray(), layer.toarray())


def test_converter_h5ad_dense_auto_writes_same_name_zip(tmp_path: Path, dense_adata) -> None:
    source = tmp_path / "dense_input.h5ad"
    dense_adata.write_h5ad(source)

    convert = AnnDataZarrZipConverter(chunk_size=(8,))
    root = convert(source)

    assert root == tmp_path / "dense_input.zarr.zip"
    ds = launch(root)
    assert isinstance(ds, DenseDataset)
    assert ds.data.shape == (12,)
    assert np.array_equal(_registered_matrix(ds), np.asarray(dense_adata.X))


def test_converter_can_disable_default_compressor(tmp_path: Path, dense_adata) -> None:
    source = tmp_path / "dense_uncompressed.h5ad"
    dense_adata.write_h5ad(source)

    convert = AnnDataZarrZipConverter(chunk_size=(8,), compressor=None)
    root = convert(source)
    ds = launch(root)

    assert ds.data.codec.compressor is None
    assert np.array_equal(_registered_matrix(ds), np.asarray(dense_adata.X))


def test_converter_call_can_disable_default_compressor(tmp_path: Path, dense_adata) -> None:
    source = tmp_path / "dense_call_uncompressed.h5ad"
    dense_adata.write_h5ad(source)

    convert = AnnDataZarrZipConverter(chunk_size=(8,))
    root = convert(source, compressor=None)
    ds = launch(root)

    assert ds.data.codec.compressor is None
    assert np.array_equal(_registered_matrix(ds), np.asarray(dense_adata.X))


def test_converter_h5ad_dense_layer_auto(tmp_path: Path, dense_adata) -> None:
    layer = np.asarray(dense_adata.X, dtype=np.float32) + 7
    dense_adata.layers["counts"] = layer
    source = tmp_path / "dense_layer_input.h5ad"
    dense_adata.write_h5ad(source)

    convert = AnnDataZarrZipConverter(chunk_size=(8,), layer_format="auto")
    root = convert(source)

    ds = launch(root, layer="counts")
    loaded = read_zarr(root)
    assert isinstance(ds, DenseDataset)
    assert ds.data.shape == (12,)
    assert np.array_equal(_registered_matrix(ds), layer)
    assert np.array_equal(np.asarray(loaded.layers["counts"]), layer)


def test_converter_h5ad_sparse_auto_writes_cell_aligned_zip(tmp_path: Path, sparse_adata) -> None:
    adata, matrix = sparse_adata
    source = tmp_path / "sparse_input.h5ad"
    adata.write_h5ad(source)

    convert = AnnDataZarrZipConverter(chunk_size=(4,), align_cells=True)
    root = convert(source)

    assert root == tmp_path / "sparse_input.zarr.zip"
    ds = launch(root)
    assert isinstance(ds, SparseDataset)
    assert ds.indices.variable_chunks
    assert ds.indices.chunk_boundaries == ((0, 4, 6),)
    assert np.array_equal(_registered_matrix(ds), matrix.toarray())


def test_converter_reads_zarr_directory_without_suffix(tmp_path: Path, dense_adata) -> None:
    source = tmp_path / "standard_zarr_store"
    dense_adata.write_zarr(source)

    convert = AnnDataZarrZipConverter(chunk_size=(8,))
    root = convert(source)

    assert root == tmp_path / "standard_zarr_store.zarr.zip"
    ds = launch(root)
    assert isinstance(ds, DenseDataset)
    assert np.array_equal(_registered_matrix(ds), np.asarray(dense_adata.X))


def test_converter_explicit_dense2d_and_output_dir(tmp_path: Path, dense_adata) -> None:
    source = tmp_path / "explicit.h5ad"
    out_dir = tmp_path / "converted"
    dense_adata.write_h5ad(source)

    convert = AnnDataZarrZipConverter(output_dir=out_dir, smart=False, format="dense2d")
    root = convert(source, read_format="h5ad")

    assert root == out_dir / "explicit.zarr.zip"
    ds = launch(root)
    assert isinstance(ds, DenseDataset)
    assert ds.data.shape == (3, 4)
