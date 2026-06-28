"""Regression tests for the AnnData zarr -> scdata payload bridge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scdata.data import DenseDataset, SparseDataset
from scdata.io import convert_anndata_zarr, launch, write_anndata

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sp = pytest.importorskip("scipy.sparse")

pytestmark = pytest.mark.filterwarnings("ignore:Writing zarr v2 data:UserWarning")


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


def test_write_anndata_dense_converts_required_arrays(tmp_path: Path, dense_adata):
    root = write_anndata(dense_adata, tmp_path / "dense.zarr", chunks=(2, 2))
    ds = launch(root)

    assert isinstance(ds, DenseDataset)
    assert ds.data.shape == (3, 4)
    assert ds.data.num_chunks == 4
    assert tuple(ds.gene_names) == ("g0", "g1", "g2", "g3")
    assert (root / "X" / "payload.bin").is_file()
    assert not (root / "X" / "0.0").exists()
    assert (root / "var" / "_index" / "payload.bin").is_file()
    assert not (root / "var" / "_index" / "0").exists()
    assert not (root / ".zmetadata").exists()


def test_write_anndata_sparse_csr_preserves_anndata_shape(tmp_path: Path, sparse_adata):
    adata, matrix = sparse_adata
    root = write_anndata(adata, tmp_path / "sparse.zarr", chunks=(4,))
    ds = launch(root)

    assert isinstance(ds, SparseDataset)
    assert ds.num_cells == 3
    assert ds.num_genes == 4
    assert ds.indptr == tuple(matrix.indptr.tolist())
    assert tuple(ds.gene_names) == ("g0", "g1", "g2", "g3")
    x_attrs = json.loads((root / "X" / ".zattrs").read_text())
    assert x_attrs["encoding-type"] == "csr_matrix"
    assert x_attrs["shape"] == [3, 4]
    assert (root / "X" / "data" / "payload.bin").is_file()
    assert not (root / "X" / "data" / "0").exists()


def test_convert_anndata_zarr_can_keep_original_chunks(tmp_path: Path, dense_adata):
    root = tmp_path / "keep_chunks.zarr"
    dense_adata.write_zarr(root, chunks=(2, 2))

    convert_anndata_zarr(root, keep_zarr_chunks=True)
    ds = launch(root)

    assert isinstance(ds, DenseDataset)
    assert (root / "X" / "payload.bin").is_file()
    assert (root / "X" / "0.0").is_file()
