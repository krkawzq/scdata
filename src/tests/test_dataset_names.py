from __future__ import annotations

import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scdata.anndata import write_scc
import scdata.compress as scc
import scdata.load as sc_load


def _write_scc(
    tmp_path: Path,
    values: np.ndarray,
    *,
    name: str = "sample.scc",
    layers: dict[str, np.ndarray] | None = None,
    obsm: dict[str, np.ndarray] | None = None,
) -> Path:
    n_obs, n_vars = values.shape
    adata = ad.AnnData(
        X=values,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )
    if layers:
        for key, matrix in layers.items():
            adata.layers[key] = matrix
    if obsm:
        for key, matrix in obsm.items():
            adata.obsm[key] = matrix
    target = tmp_path / name
    write_scc(adata, target, store="dir" if not name.endswith(".zip") else "zip")
    return target


def test_build_feature_map_matches_strings_and_accepts_pandas() -> None:
    source = pd.Index(["b", "a", "c", "a"])
    target = np.array(["a", "d", "c"], dtype=object)
    mapping = sc_load.build_feature_map(source, target)
    assert mapping == (None, 0, 2, None)

    with pytest.raises(TypeError, match="wrap a single name"):
        sc_load.build_feature_map("gene", ["gene"])
    located = sc_load.locate_names(["c0", "c1", "c2"], pd.Index(["c2", "c0"]))
    np.testing.assert_array_equal(located, np.array([2, 0], dtype=np.uint64))
    dropped = sc_load.locate_names(["c0", "c1"], ["c9", "c1"], missing="drop")
    np.testing.assert_array_equal(dropped, np.array([1], dtype=np.uint64))
    with pytest.raises(ValueError, match="not present"):
        sc_load.locate_names(["c0"], ["missing"])


def test_obs_names_and_list_keys(tmp_path: Path) -> None:
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    layer = values + 1
    pca = np.arange(6, dtype=np.float32).reshape(3, 2)
    path = _write_scc(
        tmp_path,
        values,
        layers={"counts": layer},
        obsm={"X_pca": pca},
    )
    assert sc_load.list_keys(path) == ["X", "layers/counts", "obsm/X_pca"]

    dataset = sc_load.register(path)
    assert dataset.obs_names == ("c0", "c1", "c2")
    assert dataset.n_obs == 3
    assert dataset.n_vars == 4
    assert dataset.var_names == ("g0", "g1", "g2", "g3")
    np.testing.assert_array_equal(dataset.rows_for(["c2", "c0"]), np.array([2, 0], dtype=np.uint64))

    pca_ds = sc_load.register(path, key="obsm/X_pca")
    assert pca_ds.feature_names is None
    assert pca_ds.obs_names == ("c0", "c1", "c2")

    archive = _write_scc(tmp_path, values, name="sample.scc.zip")
    assert "X" in sc_load.list_keys(archive)


def test_aligned_features_and_rows_for_compile(tmp_path: Path) -> None:
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    dataset = sc_load.register(_write_scc(tmp_path, values)).with_aligned_features(
        pd.Index(["g2", "g0", "extra"])
    )
    assert dataset.feature_map == (1, None, 0)
    output = sc_load.OutputSpec(3, np.float32, fill=-1)
    result = sc_load.compile(
        dataset,
        dataset.rows_for(["c3", "c1"]),
        output=output,
        batch_size=2,
        prefetch_step=2,
    ).read(sc_load.SessionConfig(num_workers=1, io_mode="blocking"))
    np.testing.assert_array_equal(
        result,
        np.array([[11, 9, -1], [5, 3, -1]], dtype=np.float32),
    )


def test_dataset_close_and_pickle(tmp_path: Path) -> None:
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    path = _write_scc(tmp_path, values)
    dataset = sc_load.register(path, feature_map=[2, None, 0])
    payload = pickle.dumps(dataset)
    dataset.close()
    dataset.close()
    assert dataset.closed
    assert dataset.obs_names == ("c0", "c1")
    with pytest.raises(ValueError, match="closed Dataset"):
        sc_load.compile(dataset, [0], batch_size=1, prefetch_step=2)
    with pytest.raises(ValueError, match="closed Dataset"):
        dataset.with_feature_map(None)

    restored = pickle.loads(payload)
    assert not restored.closed
    assert restored.feature_map == (2, None, 0)
    assert restored.obs_names == ("c0", "c1")
    assert restored.feature_names == ("g0", "g1", "g2")
    config = sc_load.SessionConfig(num_workers=1, io_mode="blocking")
    np.testing.assert_array_equal(
        sc_load.compile(
            restored,
            [1],
            output=sc_load.OutputSpec(4, np.float32, fill=-1),
            batch_size=1,
            prefetch_step=2,
        ).read(config),
        np.array([[5, -1, 3, -1]], dtype=np.float32),
    )

    closed_roundtrip = pickle.loads(pickle.dumps(dataset))
    assert not closed_roundtrip.closed
    with sc_load.register(path) as opened:
        assert not opened.closed
    assert opened.closed


def test_list_keys_bare_store(tmp_path: Path) -> None:
    values = np.arange(4, dtype=np.float32).reshape(2, 2)
    root = tmp_path / "bare"
    scc.write(root, values)
    assert sc_load.list_keys(root) == [""]
    dataset = sc_load.register(root)
    assert dataset.obs_names is None
    assert dataset.feature_names is None


def test_obs_names_length_is_validated(tmp_path: Path) -> None:
    path = _write_scc(tmp_path, np.arange(6, dtype=np.float32).reshape(2, 3))
    with pytest.raises(ValueError, match="obs_names has length"):
        sc_load.register(path, obs_names=["c0"])
