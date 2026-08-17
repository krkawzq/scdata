"""Shared pytest hooks for optional AnnData / zarr v3 coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

_ANNDATA_MODULES = {
    "test_anndata_scc.py",
    "test_api.py",
    "test_dataset_names.py",
    "test_distributed.py",
}
_ANNDATA_TESTS = {
    "test_feature_map_length_is_validated",
}


def _has_scc_anndata() -> bool:
    try:
        import anndata as ad
        import zarr
    except ImportError:
        return False
    major = int(str(zarr.__version__).split(".", 1)[0])
    return hasattr(ad, "settings") and major >= 3


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if _has_scc_anndata():
        return
    skip = pytest.mark.skip(reason="AnnData scc I/O requires anndata>=0.11 and zarr>=3")
    for item in items:
        path = Path(str(getattr(item, "path", item.fspath)))
        if path.name in _ANNDATA_MODULES or item.name in _ANNDATA_TESTS:
            item.add_marker(skip)
