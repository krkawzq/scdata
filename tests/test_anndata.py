"""Regression tests for the AnnData zarr v3 -> scdata databank bridge."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from scdata import ScDataBank
from scdata.data import DenseDataset, SparseDataset
from scdata.io import AnnDataZarrZipConverter, launch, launch_all, read_zarr, write_zarr

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sp = pytest.importorskip("scipy.sparse")
pytest.importorskip("zarr")


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

    assert isinstance(ds, SparseDataset)
    assert ds.indices.variable_chunks
    assert ds.data.variable_chunks
    assert tuple(np.asarray(ds.indptr).tolist()) == tuple(matrix.indptr.tolist())
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
