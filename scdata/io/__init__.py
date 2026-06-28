"""Store IO for scdata.

Reading a scdata store parses its zarr v2 metadata and scdata chunk index
into :mod:`scdata.data` dataset objects.  The Rust databank then opens the
payload file and reads chunks by offset; this module does not decode chunk
data itself.
"""

from __future__ import annotations

from scdata.io._anndata import convert_anndata_zarr, write_anndata
from scdata.io._launch import Store, StoreError, launch, launch_store

__all__ = ["Store", "StoreError", "convert_anndata_zarr", "launch", "launch_store", "write_anndata"]
