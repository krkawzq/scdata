"""Store IO for scdata.

``launch`` parses a constrained zarr v3 metadata subset into
:mod:`scdata.data` dataset objects; the Rust databank then opens numeric chunk
files directly.  It does not invoke zarr's array-level decoder, so
``sharding_indexed`` and other unknown codecs must be rewritten with
:class:`AnnDataZarrZipConverter` before direct launch.

:func:`write_zarr` / :func:`read_zarr` bridge :class:`anndata.AnnData` to the
same zarr v3 layout.  Stock ``anndata.read_zarr`` can read compatible directory
stores, but cannot take a ``.zarr.zip`` filename directly; use
:func:`read_zarr` for that container (or open a ``zarr.storage.ZipStore``).
"""

from __future__ import annotations

from scdata.io._anndata import read_zarr, write_zarr
from scdata.io._convert import AnnDataZarrZipConverter
from scdata.io._launch import (
    Store,
    StoreError,
    launch,
    launch_all,
    launch_store,
    launch_store_all,
    read_var_names,
)

__all__ = [
    "AnnDataZarrZipConverter",
    "Store",
    "StoreError",
    "launch",
    "launch_all",
    "launch_store",
    "launch_store_all",
    "read_var_names",
    "read_zarr",
    "write_zarr",
]
