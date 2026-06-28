"""Read-path regression tests for zarr v3 scdata stores."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scdata.data import ArrayOrder, DenseDataset, DType, SparseDataset
from scdata.io import write_zarr
from scdata.io._launch import StoreError, launch, launch_all

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sp = pytest.importorskip("scipy.sparse")
pytest.importorskip("zarr")


def _dense_adata(
    shape: tuple[int, int] = (3, 4),
    dtype: Any = np.float32,
    genes: list[str] | None = None,
) -> tuple[Any, np.ndarray]:
    if genes is None:
        genes = [f"g{i}" for i in range(shape[1])]
    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    return (
        ad.AnnData(
            X=data,
            obs=pd.DataFrame(index=[f"c{i}" for i in range(shape[0])]),
            var=pd.DataFrame(index=genes),
        ),
        data,
    )


def _sparse_adata(dtype: Any = np.float32, index_dtype: Any = np.int32) -> tuple[Any, Any]:
    dense = np.array(
        [
            [1, 0, 2, 0],
            [0, 3, 0, 4],
            [5, 0, 0, 6],
        ],
        dtype=dtype,
    )
    matrix = sp.csr_matrix(dense)
    matrix.indices = matrix.indices.astype(index_dtype, copy=False)
    matrix.indptr = matrix.indptr.astype(index_dtype, copy=False)
    return (
        ad.AnnData(
            X=matrix,
            obs=pd.DataFrame(index=["c0", "c1", "c2"]),
            var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
        ),
        matrix,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_dense2d_dir_metadata(tmp_path: Path) -> None:
    adata, data = _dense_adata((3, 4), np.float32)
    root = write_zarr(
        adata, tmp_path / "dense.zarr", format="dense2d", chunk_size=(2, 2), store="dir"
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
    assert ds.data.payload_path == ""
    assert tuple(ds.gene_names) == ("g0", "g1", "g2", "g3")
    assert np.array_equal(np.asarray(adata.X), data)


def test_dense1d_layer_and_launch_all(tmp_path: Path) -> None:
    adata, data = _dense_adata((3, 4), np.float32)
    adata.layers["counts"] = data + 100
    root = write_zarr(
        adata,
        tmp_path / "dense_layer.zarr",
        format="dense1d",
        layer_format="auto",
        chunk_size=(8,),
        store="dir",
    )

    layer_ds = launch(root, layer="counts")
    collection = launch_all(root)

    assert isinstance(layer_ds, DenseDataset)
    assert layer_ds.data.shape == (12,)
    assert collection.keys() == ("X", "layers/counts")
    assert collection["counts"].data.shape == (12,)


def test_sparse_csr_v3_metadata(tmp_path: Path) -> None:
    adata, matrix = _sparse_adata(np.float32, np.int64)
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
    assert ds.num_cells == 3
    assert ds.num_genes == 4
    assert tuple(np.asarray(ds.indptr).tolist()) == tuple(matrix.indptr.tolist())
    assert ds.index_dtype == DType.I64
    assert ds.indices.dtype == DType.I64
    assert ds.indices.variable_chunks
    assert ds.data.variable_chunks


def test_zip_store_metadata(tmp_path: Path) -> None:
    adata, _ = _dense_adata((3, 4), np.float32)
    root = write_zarr(
        adata,
        tmp_path / "dense.zarr.zip",
        format="dense1d",
        chunk_size=(8,),
        store="zip",
    )

    ds = launch(root)

    assert isinstance(ds, DenseDataset)
    assert ds.data.shape == (12,)
    assert ds.data.chunk_file_paths
    assert set(ds.data.chunk_file_paths) == {str(root)}
    assert any(offset > 0 for offset in ds.data.chunk_offsets)


def test_dense_and_sparse_layers_launch_all(tmp_path: Path) -> None:
    adata, data = _dense_adata((3, 4), np.float32)
    sparse_layer = sp.csr_matrix(np.where(data % 2 == 0, data, 0).astype(np.float32))
    adata.layers["counts"] = data + 7
    adata.layers["spliced"] = sparse_layer
    root = write_zarr(
        adata,
        tmp_path / "mixed_layers.zarr",
        format="dense1d",
        layer_format="auto",
        chunk_size=(4,),
        align_cells=True,
        store="dir",
    )

    collection = launch_all(root)

    assert collection.keys() == ("X", "layers/counts", "layers/spliced")
    assert isinstance(collection["counts"], DenseDataset)
    assert isinstance(collection["spliced"], SparseDataset)


def test_var_index_name_from_anndata_attrs(tmp_path: Path) -> None:
    adata, _ = _dense_adata((3, 4), np.float32, ["a", "b", "c", "d"])
    root = write_zarr(adata, tmp_path / "custom_var_index.zarr", store="dir")
    (root / "var" / "_index").rename(root / "var" / "gene_ids")
    meta = _read_json(root / "var" / "zarr.json")
    meta.setdefault("attributes", {})["_index"] = "gene_ids"
    _write_json(root / "var" / "zarr.json", meta)

    ds = launch(root)

    assert tuple(ds.gene_names) == ("a", "b", "c", "d")


def test_v3_store_requires_x(tmp_path: Path) -> None:
    root = tmp_path / "v3store.zarr"
    root.mkdir()
    _write_json(root / "zarr.json", {"zarr_format": 3, "node_type": "group"})

    with pytest.raises(StoreError, match="X"):
        launch(root)


def test_non_v3_store_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "non_v3.zarr"
    root.mkdir()
    _write_json(root / ".zgroup", {"zarr_format": 2})

    with pytest.raises(StoreError, match="zarr v3"):
        launch(root)


def test_nonexistent_path(tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="does not exist"):
        launch(tmp_path / "nope.zarr")


def test_shape_entries_must_be_json_integers(tmp_path: Path) -> None:
    adata, _ = _dense_adata()
    root = write_zarr(adata, tmp_path / "float_shape.zarr", store="dir")
    meta = _read_json(root / "X" / "zarr.json")
    meta["shape"] = [3.5, 4]
    _write_json(root / "X" / "zarr.json", meta)

    with pytest.raises(StoreError, match="JSON integers"):
        launch(root)


def test_unsupported_dtype_is_store_error(tmp_path: Path) -> None:
    adata, _ = _dense_adata()
    root = write_zarr(adata, tmp_path / "bad_dtype.zarr", store="dir")
    meta = _read_json(root / "X" / "zarr.json")
    meta["data_type"] = "complex128"
    _write_json(root / "X" / "zarr.json", meta)

    with pytest.raises(StoreError, match="unsupported v3 data_type"):
        launch(root)


def test_f_order_rejected(tmp_path: Path) -> None:
    adata, _ = _dense_adata()
    root = write_zarr(adata, tmp_path / "forder.zarr", store="dir")
    meta = _read_json(root / "X" / "zarr.json")
    meta["order"] = "F"
    _write_json(root / "X" / "zarr.json", meta)

    with pytest.raises(StoreError, match="F-order"):
        launch(root)


def test_csc_rejected(tmp_path: Path) -> None:
    adata, _ = _sparse_adata()
    root = write_zarr(adata, tmp_path / "csc.zarr", format="sparse", store="dir")
    meta = _read_json(root / "X" / "zarr.json")
    meta.setdefault("attributes", {})["encoding-type"] = "CSC"
    _write_json(root / "X" / "zarr.json", meta)

    with pytest.raises(StoreError, match="CSC"):
        launch(root)


def test_csr_index_dtype_mismatch(tmp_path: Path) -> None:
    adata, _ = _sparse_adata()
    root = write_zarr(adata, tmp_path / "idx_dtype.zarr", format="sparse", store="dir")
    meta = _read_json(root / "X" / "indices" / "zarr.json")
    meta["data_type"] = "float32"
    _write_json(root / "X" / "indices" / "zarr.json", meta)

    with pytest.raises(StoreError, match="CSR index"):
        launch(root)


def test_indptr_float_dtype_rejected(tmp_path: Path) -> None:
    adata, _ = _sparse_adata()
    root = write_zarr(adata, tmp_path / "indptr_f.zarr", format="sparse", store="dir")
    meta = _read_json(root / "X" / "indptr" / "zarr.json")
    meta["data_type"] = "float64"
    _write_json(root / "X" / "indptr" / "zarr.json", meta)

    with pytest.raises(StoreError, match="integer"):
        launch(root)


def test_sparse_shape_attr_must_match_indptr_and_var(tmp_path: Path) -> None:
    adata, _ = _sparse_adata()
    root = write_zarr(adata, tmp_path / "sparse_shape.zarr", format="sparse", store="dir")
    meta = _read_json(root / "X" / "zarr.json")
    meta.setdefault("attributes", {})["shape"] = [3, 99]
    _write_json(root / "X" / "zarr.json", meta)

    with pytest.raises(StoreError, match="99 genes"):
        launch(root)


def test_missing_var_group(tmp_path: Path) -> None:
    adata, _ = _dense_adata()
    root = write_zarr(adata, tmp_path / "no_var.zarr", store="dir")
    shutil.rmtree(root / "var")

    with pytest.raises(StoreError, match="var"):
        launch(root)


def test_zip_bad_archive(tmp_path: Path) -> None:
    bad = tmp_path / "bad.zarr.zip"
    bad.write_bytes(b"not a zip")

    with pytest.raises(StoreError, match="zip"):
        launch(bad)


def test_zip_must_be_stored(tmp_path: Path) -> None:
    adata, _ = _dense_adata()
    src = write_zarr(adata, tmp_path / "zip_deflated_src.zarr", store="dir")
    zip_path = tmp_path / "deflated.zarr.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src).as_posix())

    with pytest.raises(StoreError, match="ZIP_STORED"):
        launch(zip_path)


def test_directory_store_is_not_zarr_v3(tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="zarr v3"):
        launch(tmp_path)
