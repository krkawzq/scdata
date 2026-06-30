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


def test_launch_multiple_v3_codecs_preserves_decode_order(tmp_path: Path, dense_adata) -> None:
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


def test_write_zarr_raw_sparse_aligned_round_trip(tmp_path: Path) -> None:
    """raw.X must use the same cell-aligned CSR layout as X and round-trip."""
    matrix = sp.csr_matrix(
        np.array(
            [
                [1, 0, 2, 0, 0],
                [0, 3, 0, 0, 4],
                [5, 0, 0, 6, 0],
            ],
            dtype=np.float32,
        )
    )
    raw_matrix = sp.csr_matrix(
        np.array(
            [
                [7, 0, 8],
                [0, 9, 0],
                [1, 0, 2],
            ],
            dtype=np.float32,
        )
    )
    adata = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3", "g4"]),
    )
    adata.raw = ad.AnnData(
        X=raw_matrix,
        var=pd.DataFrame(index=["r0", "r1", "r2"]),
    )

    root = write_zarr(
        adata,
        tmp_path / "raw.zarr",
        format="sparse",
        chunk_size=(2,),
        align_cells=True,
        store="dir",
    )

    # raw group keeps anndata's "raw" encoding-type; raw.X carries the scdata
    # marker and a rectilinear (cell-aligned) chunk grid just like X.
    raw_meta = _read_json(root / "raw" / "zarr.json")
    assert raw_meta["attributes"]["encoding-type"] == "raw"
    x_meta = _read_json(root / "raw" / "X" / "zarr.json")
    assert x_meta["attributes"]["scdata-matrix"] == "sparse"
    data_meta = _read_json(root / "raw" / "X" / "data" / "zarr.json")
    assert data_meta["chunk_grid"]["name"] == "rectilinear"
    indptr_meta = _read_json(root / "raw" / "X" / "indptr" / "zarr.json")
    assert indptr_meta["chunk_grid"]["name"] == "regular"

    loaded = read_zarr(root)
    assert loaded.raw is not None
    assert tuple(loaded.raw.X.shape) == raw_matrix.shape
    assert np.array_equal(loaded.raw.X.toarray(), raw_matrix.toarray())

    # zip store (what the convert_cellxgene script produces) round-trips raw.X.
    zip_root = write_zarr(
        adata,
        tmp_path / "raw.zarr.zip",
        format="sparse",
        chunk_size=(2,),
        align_cells=True,
        store="zip",
    )
    zipped = read_zarr(zip_root)
    assert zipped.raw is not None
    assert np.array_equal(zipped.raw.X.toarray(), raw_matrix.toarray())


def test_write_zarr_raw_dense1d_round_trip(tmp_path: Path) -> None:
    """raw.X follows the dense1d layout and is reshaped back on read."""
    raw_matrix = np.arange(6, dtype=np.float32).reshape(3, 2)
    adata = ad.AnnData(
        X=np.arange(12, dtype=np.float32).reshape(3, 4),
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
    )
    adata.raw = ad.AnnData(
        X=raw_matrix,
        var=pd.DataFrame(index=["r0", "r1"]),
    )

    root = write_zarr(
        adata,
        tmp_path / "raw1d.zarr",
        format="dense1d",
        align_cells=True,
        store="dir",
    )
    x_meta = _read_json(root / "raw" / "X" / "zarr.json")
    assert x_meta["attributes"]["scdata-matrix"] == "dense1d"

    loaded = read_zarr(root)
    assert loaded.raw is not None
    assert np.array_equal(np.asarray(loaded.raw.X), raw_matrix)


def test_write_zarr_dense_x_and_raw_to_sparse_round_trip(tmp_path: Path) -> None:
    """``format="sparse"`` converts dense X and dense raw.X to cell-aligned CSR."""
    x_matrix = np.arange(12, dtype=np.float32).reshape(3, 4)
    raw_matrix = np.arange(6, dtype=np.float32).reshape(3, 2)
    adata = ad.AnnData(
        X=x_matrix,
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
    )
    adata.raw = ad.AnnData(
        X=raw_matrix,
        var=pd.DataFrame(index=["r0", "r1"]),
    )

    root = write_zarr(
        adata,
        tmp_path / "dense_to_sparse.zarr",
        format="sparse",
        chunk_size=(2,),
        align_cells=True,
        store="dir",
    )

    assert _read_json(root / "X" / "zarr.json")["attributes"]["scdata-matrix"] == "sparse"
    assert _read_json(root / "raw" / "X" / "zarr.json")["attributes"]["scdata-matrix"] == "sparse"

    loaded = read_zarr(root)
    assert sp.issparse(loaded.X)
    assert np.array_equal(loaded.X.toarray(), x_matrix)
    assert loaded.raw is not None
    assert sp.issparse(loaded.raw.X)
    assert np.array_equal(loaded.raw.X.toarray(), raw_matrix)


@pytest.fixture
def rich_adata():
    """AnnData with every slot populated (obsm/varm/obsp/uns/layers/raw)."""
    rng = np.random.default_rng(0)
    n_obs, n_var = 6, 4
    adata = ad.AnnData(X=rng.normal(size=(n_obs, n_var)).astype(np.float32))
    adata.obs["cell_type"] = rng.choice(["A", "B"], n_obs).astype(object)
    adata.var_names = [f"g{i}" for i in range(n_var)]
    adata.obsm["X_pca"] = rng.normal(size=(n_obs, 2)).astype(np.float32)
    adata.varm["loadings"] = rng.normal(size=(n_var, 2)).astype(np.float32)
    adata.obsp["connectivities"] = sp.csr_matrix(rng.random((n_obs, n_obs)).astype(np.float32))
    adata.uns["meta"] = {"k": 7, "arr": np.arange(3, dtype=np.float32)}
    adata.layers["counts"] = sp.csr_matrix(rng.poisson(2, (n_obs, n_var)).astype(np.float32))
    adata.raw = ad.AnnData(
        X=sp.csr_matrix(rng.poisson(1, (n_obs, 3)).astype(np.float32)),
        var=pd.DataFrame(index=[f"r{i}" for i in range(3)]),
    )
    return adata


def test_read_zarr_metadata_only_skips_matrices(tmp_path: Path, rich_adata) -> None:
    """``metadata_only`` leaves X / layers / raw unloaded while shapes come from obs/var."""
    root = write_zarr(
        rich_adata,
        tmp_path / "rich.zarr",
        format="dense1d",
        layer_format="auto",
        store="dir",
    )

    loaded = read_zarr(root, metadata_only=True)

    assert loaded.X is None
    assert dict(loaded.layers) == {}
    assert loaded.raw is None
    # n_obs / n_vars are still known from obs / var.
    assert loaded.n_obs == rich_adata.n_obs
    assert loaded.n_vars == rich_adata.n_vars


def test_read_zarr_metadata_only_loads_annotations(tmp_path: Path, rich_adata) -> None:
    """``metadata_only`` still loads obs / var / uns / obsm / varm / obsp verbatim."""
    root = write_zarr(
        rich_adata,
        tmp_path / "rich_meta.zarr.zip",
        format="dense1d",
        layer_format="auto",
        store="zip",
    )

    loaded = read_zarr(root, metadata_only=True)

    assert list(loaded.obs["cell_type"]) == list(rich_adata.obs["cell_type"])
    assert list(loaded.var_names) == list(rich_adata.var_names)
    assert loaded.uns["meta"]["k"] == rich_adata.uns["meta"]["k"]
    assert np.array_equal(loaded.uns["meta"]["arr"], rich_adata.uns["meta"]["arr"])
    assert np.allclose(loaded.obsm["X_pca"], rich_adata.obsm["X_pca"])
    assert np.allclose(loaded.varm["loadings"], rich_adata.varm["loadings"])
    assert np.allclose(
        loaded.obsp["connectivities"].toarray(),
        rich_adata.obsp["connectivities"].toarray(),
    )


def test_read_zarr_legacy_flat_raw_rebuild(tmp_path: Path, rich_adata) -> None:
    """Pre-modern-raw stores keep ``raw.X`` / ``raw.var`` flat at the root.

    Mirrors the legacy layout :func:`anndata.read_zarr` rebuilds via
    ``_read_legacy_raw``; scdata's reader must do the same so older stores do
    not silently lose ``raw``.
    """
    import zarr

    from anndata._io.specs import write_elem

    root = tmp_path / "legacy.zarr"
    g = zarr.open_group(root, mode="w", zarr_format=3)
    g.attrs["encoding-type"] = "anndata"
    g.attrs["encoding-version"] = "0.1.0"
    write_elem(g, "obs", rich_adata.obs)
    write_elem(g, "var", rich_adata.var)
    write_elem(g, "X", rich_adata.X)
    write_elem(g, "uns", dict(rich_adata.uns))
    write_elem(g, "raw.X", rich_adata.raw.X)
    write_elem(g, "raw.var", rich_adata.raw.var)

    loaded = read_zarr(root)

    assert loaded.raw is not None
    assert loaded.raw.n_vars == rich_adata.raw.n_vars
    assert list(loaded.raw.var_names) == list(rich_adata.raw.var_names)
    assert np.array_equal(
        loaded.raw.X.toarray(),
        rich_adata.raw.X.toarray(),
    )
    # Main X is unaffected by the legacy raw path.
    assert np.array_equal(
        loaded.X.toarray() if sp.issparse(loaded.X) else loaded.X,
        rich_adata.X.toarray() if sp.issparse(rich_adata.X) else rich_adata.X,
    )

    # metadata_only drops raw on legacy stores too (raw cannot exist without X).
    meta_only = read_zarr(root, metadata_only=True)
    assert meta_only.raw is None
    assert meta_only.X is None
    assert list(meta_only.obs["cell_type"]) == list(rich_adata.obs["cell_type"])


def test_read_zarr_metadata_only_matches_full_read_annotations(tmp_path: Path, rich_adata) -> None:
    """Annotations read under ``metadata_only`` equal those from a full read."""
    root = write_zarr(
        rich_adata,
        tmp_path / "rich_cmp.zarr",
        format="sparse",
        layer_format="auto",
        align_cells=True,
        store="dir",
    )

    full = read_zarr(root)
    meta_only = read_zarr(root, metadata_only=True)

    assert list(meta_only.obs.index) == list(full.obs.index)
    assert list(meta_only.var_names) == list(full.var_names)
    assert set(meta_only.obsm.keys()) == set(full.obsm.keys())
    assert set(meta_only.varm.keys()) == set(full.varm.keys())
    assert set(meta_only.obsp.keys()) == set(full.obsp.keys())
    assert set(meta_only.uns.keys()) == set(full.uns.keys())
    # And the full read still recovers the matrices metadata_only skipped.
    assert full.X is not None and full.X.shape == rich_adata.X.shape
    assert "counts" in full.layers
    assert full.raw is not None
