"""AnnData ``.scc`` / ``.scc.zip`` round-trip tests."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sp = pytest.importorskip("scipy.sparse")
pytest.importorskip("zarr")

from scdata.anndata import read_scc, write_scc  # noqa: E402
from scdata.anndata._io import (  # noqa: E402
    _copy_for_categorical_conversion,
    _matrix_path,
    _prepare_cell_matrix,
    _resolve_load_keys,
    _restore_cell_matrix,
)
from scdata.compress import ScCsr, ScDense  # noqa: E402
from scdata.exceptions import (  # noqa: E402
    CorruptDataError,
    InvalidMetaError,
)


@pytest.fixture
def dense_adata() -> ad.AnnData:
    return ad.AnnData(
        X=np.arange(12, dtype=np.float32).reshape(3, 4),
        obs=pd.DataFrame({"batch": ["a", "b", "a"]}, index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
        layers={"counts": np.arange(12, dtype=np.float32).reshape(3, 4) + 1},
    )


@pytest.fixture
def sparse_adata() -> ad.AnnData:
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
    adata.obsm["X_pca"] = np.arange(6, dtype=np.float32).reshape(3, 2)
    adata.raw = ad.AnnData(
        X=matrix.copy(),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
    )
    return adata


@pytest.mark.parametrize("store", ["dir", "zip"])
def test_roundtrip_dense(tmp_path: Path, dense_adata: ad.AnnData, store: str) -> None:
    target = tmp_path / ("dense.scc.zip" if store == "zip" else "dense.scc")
    write_scc(dense_adata, target, store=store)  # type: ignore[arg-type]
    assert target.exists()
    if store == "zip":
        assert zipfile.is_zipfile(target)
    else:
        assert (target / "X" / "meta.json").is_file()
        assert (target / "X" / "zarr.json").is_file()

    out = read_scc(target)
    assert out.n_obs == 3 and out.n_vars == 4
    assert isinstance(out.X, ScDense)
    assert isinstance(out.layers["counts"], ScDense)
    np.testing.assert_array_equal(np.asarray(out.X), np.asarray(dense_adata.X))
    np.testing.assert_array_equal(
        np.asarray(out.layers["counts"]),
        np.asarray(dense_adata.layers["counts"]),
    )
    assert list(out.obs["batch"]) == ["a", "b", "a"]


def test_roundtrip_64_bit_integer_expression_matrices(tmp_path: Path) -> None:
    signed = np.array(
        [
            [np.iinfo(np.int64).min, -(1 << 53) - 1, -1],
            [0, (1 << 53) + 1, np.iinfo(np.int64).max],
        ],
        dtype=np.int64,
    )
    unsigned = np.array(
        [
            [0, 1, (1 << 53) + 1],
            [1 << 63, (1 << 63) + 1, np.iinfo(np.uint64).max],
        ],
        dtype=np.uint64,
    )
    adata = ad.AnnData(
        X=signed,
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
        layers={"unsigned": sp.csr_matrix(unsigned)},
    )

    target = write_scc(adata, tmp_path / "integer64.scc", store="dir")
    out = read_scc(target)

    assert isinstance(out.X, ScDense)
    assert isinstance(out.layers["unsigned"], ScCsr)
    assert np.asarray(out.X).dtype == np.dtype(np.int64)
    assert out.layers["unsigned"].dtype == np.dtype(np.uint64)
    np.testing.assert_array_equal(np.asarray(out.X), signed)
    np.testing.assert_array_equal(out.layers["unsigned"].toarray(), unsigned)


def test_roundtrip_sparse_zip_and_obsm_raw(tmp_path: Path, sparse_adata: ad.AnnData) -> None:
    target = tmp_path / "sparse.scc.zip"
    write_scc(sparse_adata, target)
    out = read_scc(target)

    assert isinstance(out.X, ScCsr)
    np.testing.assert_array_equal(out.X.toarray(), sparse_adata.X.toarray())
    np.testing.assert_array_equal(np.asarray(out.obsm["X_pca"]), sparse_adata.obsm["X_pca"])
    assert out.raw is not None
    assert isinstance(out.raw.X, ScCsr)
    np.testing.assert_array_equal(out.raw.X.toarray(), sparse_adata.raw.X.toarray())
    assert zipfile.is_zipfile(target)


def test_auto_detect_store_from_suffix(tmp_path: Path, dense_adata: ad.AnnData) -> None:
    directory = write_scc(
        dense_adata,
        tmp_path / "auto.scc",
        store="auto",
        num_workers=2,
    )
    assert directory.is_dir()
    archive = write_scc(dense_adata, tmp_path / "auto.scc.zip", store="auto")
    assert archive.is_file() and zipfile.is_zipfile(archive)
    out = read_scc(directory, num_workers=2)
    np.testing.assert_array_equal(np.asarray(out.X), dense_adata.X)


def test_exclude_expression_matrices(tmp_path: Path, sparse_adata: ad.AnnData) -> None:
    target = write_scc(sparse_adata, tmp_path / "meta.scc", store="dir")
    out = read_scc(target, exclude=("X", "layers", "raw"))

    assert out.X is None
    assert dict(out.layers) == {}
    assert out.raw is None
    assert out.n_obs == 3 and out.n_vars == 4
    np.testing.assert_array_equal(out.obsm["X_pca"], sparse_adata.obsm["X_pca"])


def test_include_obs_var_only(tmp_path: Path, sparse_adata: ad.AnnData) -> None:
    target = write_scc(sparse_adata, tmp_path / "slim.scc", store="dir")
    out = read_scc(target, include=("obs", "var"))

    assert out.X is None
    assert dict(out.layers) == {}
    assert out.raw is None
    assert dict(out.obsm) == {}
    assert out.n_obs == 3 and out.n_vars == 4


def test_write_does_not_mutate_categoricals(tmp_path: Path, dense_adata: ad.AnnData) -> None:
    before = dense_adata.obs["batch"].copy()
    write_scc(dense_adata, tmp_path / "cats.scc", store="dir")
    pd.testing.assert_series_equal(dense_adata.obs["batch"], before)


def test_write_progress_callback(tmp_path: Path, dense_adata: ad.AnnData) -> None:
    events: list[tuple[str, int, int]] = []
    write_scc(
        dense_adata,
        tmp_path / "progress.scc",
        store="dir",
        progress=lambda name, index, total: events.append((name, index, total)),
    )
    assert events
    assert events[-1][1] == events[-1][2]
    assert {name for name, _, _ in events} >= {"X", "layers/counts"}


def test_write_overwrite_false(tmp_path: Path, dense_adata: ad.AnnData) -> None:
    target = tmp_path / "once.scc"
    write_scc(dense_adata, target, store="dir")
    with pytest.raises(FileExistsError, match="already exists"):
        write_scc(dense_adata, target, store="dir", overwrite=False)


def test_dir_store_with_zip_suffix_warns(tmp_path: Path, dense_adata: ad.AnnData) -> None:
    from scdata.exceptions import PerformanceWarning

    target = tmp_path / "weird.scc.zip"
    with pytest.warns(PerformanceWarning, match=r"store='dir'"):
        write_scc(dense_adata, target, store="dir")
    assert target.is_dir()


def test_cell_matrices_use_scc_uns_and_varm_do_not(tmp_path: Path) -> None:
    adata = ad.AnnData(
        X=np.arange(12, dtype=np.float32).reshape(3, 4),
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
    )
    adata.obsm["X_pca"] = np.arange(6, dtype=np.float32).reshape(3, 2)
    adata.varm["PCs"] = np.arange(8, dtype=np.float32).reshape(4, 2)
    adata.uns["gene_corr"] = np.arange(16, dtype=np.float32).reshape(4, 4)

    root = write_scc(adata, tmp_path / "slots.scc", store="dir")

    assert (root / "X" / "meta.json").is_file()
    assert (root / "obsm" / "X_pca" / "meta.json").is_file()
    assert not (root / "uns" / "gene_corr" / "meta.json").exists()
    assert (root / "uns" / "gene_corr" / "zarr.json").is_file()
    assert not (root / "varm" / "PCs" / "meta.json").exists()
    assert (root / "varm" / "PCs" / "zarr.json").is_file()

    out = read_scc(root)
    np.testing.assert_array_equal(out.uns["gene_corr"], adata.uns["gene_corr"])
    np.testing.assert_array_equal(out.varm["PCs"], adata.varm["PCs"])
    np.testing.assert_array_equal(out.obsm["X_pca"], adata.obsm["X_pca"])


def test_multidim_obsm_flattened_as_dense_scc(tmp_path: Path) -> None:
    adata = ad.AnnData(
        X=np.arange(12, dtype=np.float32).reshape(3, 4),
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
    )
    tensor = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    adata.obsm["tensor"] = tensor

    root = write_scc(adata, tmp_path / "tensor.scc", store="dir")
    meta = json.loads((root / "obsm" / "tensor" / "zarr.json").read_text(encoding="utf-8"))
    assert meta["attributes"]["shape"] == [3, 2, 4]
    assert meta["attributes"]["cell-axis"] == 0
    assert (root / "obsm" / "tensor" / "meta.json").is_file()

    # On-disk payload is the flattened (n_cells, -1) dense view.
    import scdata.compress as scc

    flat = scc.open_store(root / "obsm" / "tensor").read()
    assert flat.shape == (3, 8)
    np.testing.assert_array_equal(flat, tensor.reshape(3, 8))

    out = read_scc(root)
    np.testing.assert_array_equal(out.obsm["tensor"], tensor)


def test_prepare_restore_cell_on_right() -> None:
    n_cells = 3
    original = np.arange(24, dtype=np.float32).reshape(2, 4, 3)  # cell on right
    payload, shape, cell_axis = _prepare_cell_matrix(original, n_cells, "demo")
    assert shape == (2, 4, 3)
    assert cell_axis == -1
    assert payload.shape == (3, 8)

    restored = _restore_cell_matrix(
        payload,
        {"shape": list(shape), "cell-axis": cell_axis},
    )
    np.testing.assert_array_equal(restored, original)


def test_obsm_dataframe_uses_anndata_writer(tmp_path: Path, dense_adata: ad.AnnData) -> None:
    scores = pd.DataFrame(
        {"score": [1.5, 2.5, 3.5]},
        index=dense_adata.obs_names.copy(),
    )
    dense_adata.obsm["scores"] = scores

    root = write_scc(dense_adata, tmp_path / "dataframe.scc", store="dir")

    assert not (root / "obsm" / "scores" / "meta.json").exists()
    loaded = read_scc(root)
    pd.testing.assert_frame_equal(loaded.obsm["scores"], scores)


def test_prepare_rank2_uses_shape_without_eager_array_conversion() -> None:
    class ShapeOnly:
        shape = (3, 2)

        def __array__(self) -> np.ndarray:
            raise AssertionError("rank-2 shape lookup must not materialize the matrix")

    matrix = ShapeOnly()
    payload, shape, cell_axis = _prepare_cell_matrix(matrix, 3, "shape-only")

    assert payload is matrix
    assert shape == (3, 2)
    assert cell_axis == 0


def test_prepare_restore_multidim_zero_cells() -> None:
    original = np.empty((0, 2, 3), dtype=np.float32)

    payload, shape, cell_axis = _prepare_cell_matrix(original, 0, "empty")

    assert payload.shape == (0, 6)
    restored = _restore_cell_matrix(
        payload,
        {"shape": list(shape), "cell-axis": cell_axis},
    )
    assert restored.shape == original.shape


@pytest.mark.parametrize("exclude_raw", [False, True])
def test_read_rejects_mixed_modern_and_legacy_raw(
    tmp_path: Path,
    sparse_adata: ad.AnnData,
    exclude_raw: bool,
) -> None:
    import zarr
    from anndata._io.specs import write_elem

    root = write_scc(sparse_adata, tmp_path / "mixed-raw.scc", store="dir")
    group = zarr.open_group(root, mode="a")
    write_elem(group, "raw.X", np.ones((3, 1), dtype=np.float32))
    write_elem(group, "raw.var", pd.DataFrame(index=["legacy"]))

    kwargs = {"exclude": ("X", "layers", "raw")} if exclude_raw else {}
    with pytest.raises(InvalidMetaError, match="both a modern 'raw' group"):
        read_scc(root, **kwargs)  # type: ignore[arg-type]


def test_read_rejects_unknown_scc_layout_version(
    tmp_path: Path,
    dense_adata: ad.AnnData,
) -> None:
    import zarr

    root = write_scc(dense_adata, tmp_path / "version.scc", store="dir")
    group = zarr.open_group(root, mode="a")
    group["X"].attrs["scc-format"] = "999.0.0"

    with pytest.raises(InvalidMetaError, match="unsupported 'scc-format'"):
        read_scc(root)


def test_read_rejects_scc_shape_mismatch(
    tmp_path: Path,
    dense_adata: ad.AnnData,
) -> None:
    import zarr

    root = write_scc(dense_adata, tmp_path / "shape.scc", store="dir")
    group = zarr.open_group(root, mode="a")
    group["X"].attrs["shape"] = [1, 12]

    with pytest.raises(InvalidMetaError, match="payload shape .* does not match"):
        read_scc(root)


def test_read_rejects_scc_shape_outside_format_range(
    tmp_path: Path,
    dense_adata: ad.AnnData,
) -> None:
    import zarr

    root = write_scc(dense_adata, tmp_path / "shape-overflow.scc", store="dir")
    group = zarr.open_group(root, mode="a")
    group["X"].attrs["shape"] = [3, 1 << 64]

    with pytest.raises(InvalidMetaError, match=r"shape\[1\] exceeds uint64"):
        read_scc(root)


def test_read_resource_limits_are_forwarded(
    tmp_path: Path,
    dense_adata: ad.AnnData,
) -> None:
    root = write_scc(dense_adata, tmp_path / "limits.scc", store="dir")

    with pytest.raises(CorruptDataError, match="decoded size .* exceeds configured limit"):
        np.asarray(read_scc(root, max_decoded_size=1).X)

    loaded = read_scc(root, max_decoded_size=1 << 20)
    np.testing.assert_array_equal(np.asarray(loaded.X), dense_adata.X)


def test_write_reports_unsupported_expression_dtype_before_touching_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "unsupported.scc"
    target.write_text("keep", encoding="utf-8")
    adata = ad.AnnData(X=np.arange(6, dtype=np.int8).reshape(3, 2))

    with pytest.raises(ValueError, match=r"matrix 'X' dtype int8"):
        write_scc(adata, target, store="dir")

    assert target.read_text(encoding="utf-8") == "keep"


def test_roundtrip_backed_sparse_anndata(tmp_path: Path) -> None:
    matrix = sp.csr_matrix(np.arange(12, dtype=np.float32).reshape(3, 4))
    source = tmp_path / "backed.h5ad"
    ad.AnnData(X=matrix).write_h5ad(source)
    backed = ad.read_h5ad(source, backed="r")
    try:
        root = write_scc(backed, tmp_path / "backed.scc", store="dir")
    finally:
        backed.file.close()

    loaded = read_scc(root)
    assert isinstance(loaded.X, ScCsr)
    np.testing.assert_array_equal(loaded.X.toarray(), matrix.toarray())


def test_read_native_anndata_zarr(tmp_path: Path, dense_adata: ad.AnnData) -> None:
    root = tmp_path / "native.zarr"
    dense_adata.write_zarr(root)

    loaded = read_scc(root)
    np.testing.assert_array_equal(np.asarray(loaded.X), dense_adata.X)
    np.testing.assert_array_equal(np.asarray(loaded.layers["counts"]), dense_adata.layers["counts"])


@pytest.mark.parametrize(
    ("slot", "key"),
    [("layers", "."), ("obsm", "a\\b"), ("obsp", "a/b")],
)
def test_write_rejects_unsafe_matrix_keys_before_touching_target(
    tmp_path: Path,
    dense_adata: ad.AnnData,
    slot: str,
    key: str,
) -> None:
    getattr(dense_adata, slot)[key] = np.ones((3, 4 if slot == "layers" else 3), dtype=np.float32)
    target = tmp_path / "unsafe.scc"
    target.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{slot} keys"):
        write_scc(dense_adata, target, store="dir")

    assert target.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("slot", ["varm", "varp", "uns"])
def test_write_preflights_unsafe_annotation_keys(
    tmp_path: Path,
    dense_adata: ad.AnnData,
    slot: str,
) -> None:
    if slot == "varm":
        dense_adata.varm["a\\b"] = np.ones((4, 1), dtype=np.float32)
    elif slot == "varp":
        dense_adata.varp["a\\b"] = np.ones((4, 4), dtype=np.float32)
    else:
        dense_adata.uns["safe"] = {"a\\b": 1}

    with pytest.raises(ValueError, match=rf"{slot}.*keys"):
        write_scc(dense_adata, tmp_path / "unsafe-annotation.scc", store="dir")


@pytest.mark.parametrize("rel", ["a//b", "a/./b", "a\\b"])
def test_matrix_path_rejects_noncanonical_aliases(tmp_path: Path, rel: str) -> None:
    with pytest.raises(ValueError, match="invalid AnnData matrix path"):
        _matrix_path(tmp_path, rel)


def test_load_key_filter_accepts_single_string_and_validates_entries() -> None:
    assert "raw" not in _resolve_load_keys(None, "raw")
    with pytest.raises(TypeError, match="include entries must be strings"):
        _resolve_load_keys(("obs", "var", 1), None)  # type: ignore[arg-type]


def test_categorical_conversion_shares_matrix_payloads(dense_adata: ad.AnnData) -> None:
    original_x = dense_adata.X
    converted = _copy_for_categorical_conversion(dense_adata)

    assert converted.X is original_x
    assert converted.layers["counts"] is dense_adata.layers["counts"]
    assert not isinstance(dense_adata.obs["batch"].dtype, pd.CategoricalDtype)
    assert isinstance(converted.obs["batch"].dtype, pd.CategoricalDtype)


@pytest.mark.parametrize("store", ["dir", "zip"])
def test_overwrite_false_rechecks_target_before_publish(
    tmp_path: Path,
    dense_adata: ad.AnnData,
    store: str,
) -> None:
    target = tmp_path / ("race.scc.zip" if store == "zip" else "race.scc")

    def occupy_target(_name: str, index: int, total: int) -> None:
        if index == total:
            target.write_text("concurrent writer", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        write_scc(
            dense_adata,
            target,
            store=store,  # type: ignore[arg-type]
            overwrite=False,
            progress=occupy_target,
        )

    assert target.read_text(encoding="utf-8") == "concurrent writer"
