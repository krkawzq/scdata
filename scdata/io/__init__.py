"""Store IO for scdata.

Reading a scdata store parses its zarr v3 metadata into
:mod:`scdata.data` dataset objects.  The Rust databank then opens each chunk
file directly; this module does not decode chunk data itself.

:func:`write_zarr` / :func:`read_zarr` bridge :class:`anndata.AnnData` to the
same zarr v3 layout, so a store written for the Rust databank is also readable
by stock ``anndata.read_zarr`` (with two spec-legal scdata extensions).
"""

from __future__ import annotations

from scdata.io._anndata import read_zarr, write_zarr
from scdata.io._convert import AnnDataZarrZipConverter
from scdata.io._launch import Store, StoreError, launch, launch_store

__all__ = [
    "AnnDataZarrZipConverter",
    "Store",
    "StoreError",
    "launch",
    "launch_store",
    "read_zarr",
    "write_zarr",
]
