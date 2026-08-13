"""AnnData bridge for ``.scc`` / ``.scc.zip`` containers."""

from scdata.anndata._io import read_scc, write_scc

__all__ = ["read_scc", "write_scc"]
