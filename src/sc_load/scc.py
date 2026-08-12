"""Resolve AnnData ``.scc`` / ``.scc.zip`` containers to embedded sc-compress stores.

Path / ZIP discovery follows :mod:`sc_compress` (``open_store`` / ``zip.list_stores`` /
``zip.normalize_prefix``). Feature-name discovery reads only the corresponding
zarr dataframe index and does not materialize ``obs`` or matrix payloads.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Any, Literal

from sc_compress.exceptions import InvalidArgumentError, ScCompressError
from sc_compress.zip import META_NAME, list_stores, normalize_prefix

__all__ = [
    "MatrixLocation",
    "normalize_matrix_key",
    "read_feature_names",
    "resolve_matrix_location",
]

_StoreKind = Literal["dir", "zip"]


class MatrixLocation:
    """Resolved open target for one embedded sc-compress matrix."""

    __slots__ = ("container", "key", "kind", "open_path", "zip_prefix")

    def __init__(
        self,
        *,
        container: Path,
        key: str,
        kind: _StoreKind,
        open_path: Path,
        zip_prefix: str | None,
    ) -> None:
        self.container = container
        self.key = key
        self.kind = kind
        self.open_path = open_path
        self.zip_prefix = zip_prefix

    def __repr__(self) -> str:
        if self.zip_prefix is not None:
            location = f"{self.container}!/{self.zip_prefix}"
        else:
            location = str(self.open_path)
        return f"MatrixLocation(key={self.key!r}, location={location!r})"


def normalize_matrix_key(key: str) -> str:
    """Normalize an AnnData matrix key with sc-compress ZIP prefix rules."""
    try:
        normalized = normalize_prefix(key)
    except InvalidArgumentError as error:
        raise ValueError(str(error)) from None
    if not normalized:
        raise ValueError("key must be a non-empty matrix path such as 'X' or 'layers/counts'")
    return normalized


def resolve_matrix_location(
    path: str | os.PathLike[str],
    key: str = "X",
) -> MatrixLocation:
    """Map ``(scc_path, key)`` to the arguments expected by ``_open_dataset``.

    ZIP archives are validated with :func:`sc_compress.zip.list_stores`. Directory
    containers require ``{key}/meta.json``, matching an embedded sc-compress store.
    """
    container = _normalize_container_path(path)
    normalized_key = normalize_matrix_key(key)
    if container.is_file():
        if not zipfile.is_zipfile(container):
            raise ValueError(f"path is a file but not a ZIP archive: {container}")
        try:
            prefixes = set(list_stores(container))
        except (ScCompressError, OSError, zipfile.BadZipFile) as error:
            raise ValueError(f"failed to inspect ZIP archive {container}: {error}") from None
        if normalized_key not in prefixes:
            ordered = sorted(prefixes)
            available = ", ".join(repr(item) for item in ordered[:8]) or "<none>"
            if len(ordered) > 8:
                available += f", ... ({len(ordered)} total)"
            raise ValueError(
                f"ZIP archive {container} has no sc-compress store at key "
                f"{normalized_key!r} (available: {available})"
            )
        return MatrixLocation(
            container=container,
            key=normalized_key,
            kind="zip",
            open_path=container,
            zip_prefix=normalized_key,
        )
    if container.is_dir():
        open_path = container / normalized_key
        meta_path = open_path / META_NAME
        if not meta_path.is_file():
            raise ValueError(
                f"scc directory {container} has no sc-compress store at key {normalized_key!r}"
            )
        return MatrixLocation(
            container=container,
            key=normalized_key,
            kind="dir",
            open_path=open_path,
            zip_prefix=None,
        )
    if container.exists():
        raise ValueError(f"path must be a directory or ZIP file: {container}")
    raise ValueError(f"path does not exist: {container}")


def read_feature_names(
    path: str | os.PathLike[str],
    key: str = "X",
) -> tuple[str, ...] | None:
    """Return feature names for expression matrices, otherwise ``None``.

    ``X`` / ``layers/*`` read ``var``; ``raw/X`` reads ``raw/var``. Embedding
    keys such as ``obsm/*`` and ``obsp/*`` return ``None``.
    """
    container = _normalize_container_path(path)
    normalized_key = normalize_matrix_key(key)
    var_path = _feature_names_group(normalized_key)
    if var_path is None:
        return None
    return _read_index_names_zarr(container, var_path)


def _normalize_container_path(path: str | os.PathLike[str]) -> Path:
    try:
        return Path(os.fspath(path))
    except TypeError as error:
        raise TypeError("path must be str or os.PathLike[str]") from error


def _feature_names_group(key: str) -> str | None:
    if key == "X" or key.startswith("layers/"):
        return "var"
    if key == "raw/X":
        return "raw/var"
    return None


def _read_index_names_zarr(container: Path, var_path: str) -> tuple[str, ...]:
    import zarr
    from anndata._io.zarr import read_dataframe
    from zarr.storage import ZipStore

    if container.is_file():
        if not zipfile.is_zipfile(container):
            raise ValueError(f"path is a file but not a ZIP archive: {container}")
        store = ZipStore(str(container), mode="r")
        try:
            root = zarr.open_group(store, mode="r")
            return _read_index_names(root, var_path, read_dataframe)
        finally:
            store.close()
    if container.is_dir():
        root = zarr.open_group(str(container), mode="r")
        return _read_index_names(root, var_path, read_dataframe)
    if container.exists():
        raise ValueError(f"path must be a directory or ZIP file: {container}")
    raise ValueError(f"path does not exist: {container}")


def _read_index_names(
    root: Any,
    var_path: str,
    read_dataframe: Any,
) -> tuple[str, ...]:
    group = _get_group(root, var_path)
    if group is None:
        raise ValueError(f"scc store is missing feature-name group {var_path!r}")
    frame = read_dataframe(group)
    return tuple(str(name) for name in frame.index)


def _get_group(root: Any, path: str) -> Any | None:
    current = root
    for part in path.split("/"):
        try:
            current = current[part]
        except KeyError:
            return None
    return current
