"""Resolve AnnData ``.scc`` / ``.scc.zip`` containers to embedded sc-compress stores.

Path / ZIP discovery follows :mod:`scdata.compress` (``open_store`` /
``zip.list_stores`` / ``zip.normalize_prefix``). Name discovery reads only the
corresponding zarr dataframe index and does not materialize ``obs`` / ``var``
columns or matrix payloads.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Any, Literal

from scdata.compress import zip as zip_api
from scdata.exceptions import Error

__all__ = [
    "MatrixLocation",
    "list_keys",
    "normalize_matrix_key",
    "read_feature_names",
    "read_obs_names",
    "resolve_matrix_location",
]

META_NAME = zip_api.META_NAME
list_stores = zip_api.list_stores
normalize_prefix = zip_api.normalize_prefix

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
    normalized = normalize_prefix(key)
    if not normalized:
        raise ValueError("key must be a non-empty matrix path such as 'X' or 'layers/counts'")
    return normalized


def resolve_matrix_location(
    path: str | os.PathLike[str],
    key: str = "X",
) -> MatrixLocation:
    """Map ``(scc_path, key)`` to the path / ZIP prefix used to open a matrix.

    ZIP archives are validated with :func:`scdata.compress.zip.list_stores`. Directory
    containers require ``{key}/meta.json``. A directory whose root contains
    ``meta.json`` is treated as a bare store.
    """
    container = _normalize_container_path(path)
    normalized_key = normalize_matrix_key(key)
    if container.is_file():
        if not zipfile.is_zipfile(container):
            raise ValueError(f"path is a file but not a ZIP archive: {container}")
        try:
            prefixes = set(list_stores(container))
        except (Error, OSError, zipfile.BadZipFile) as error:
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
        nested = container / normalized_key
        if (nested / META_NAME).is_file():
            return MatrixLocation(
                container=container,
                key=normalized_key,
                kind="dir",
                open_path=nested,
                zip_prefix=None,
            )
        if (container / META_NAME).is_file():
            return MatrixLocation(
                container=container,
                key=normalized_key,
                kind="dir",
                open_path=container,
                zip_prefix=None,
            )
        raise ValueError(
            f"scc directory {container} has no sc-compress store at key {normalized_key!r}"
        )
    if container.exists():
        raise ValueError(f"path must be a directory or ZIP file: {container}")
    raise FileNotFoundError(f"path does not exist: {container}")


def list_keys(path: str | os.PathLike[str]) -> list[str]:
    """Return sorted sc-compress matrix keys inside an ``.scc`` / ``.scc.zip``.

    ZIP archives use :func:`scdata.compress.zip.list_stores`. Directory
    containers yield relative paths that contain ``meta.json`` (``"X"``,
    ``"layers/counts"``, …). A bare store directory whose root contains
    ``meta.json`` is reported as ``[""]``.
    """
    container = _normalize_container_path(path)
    if container.is_file():
        if not zipfile.is_zipfile(container):
            raise ValueError(f"path is a file but not a ZIP archive: {container}")
        try:
            return list_stores(container)
        except (Error, OSError, zipfile.BadZipFile) as error:
            raise ValueError(f"failed to inspect ZIP archive {container}: {error}") from None
    if container.is_dir():
        if (container / META_NAME).is_file():
            return [""]
        keys: list[str] = []
        for dirpath, dirnames, filenames in os.walk(container, followlinks=False):
            base = Path(dirpath)
            dirnames[:] = sorted(
                name
                for name in dirnames
                if not name.startswith(".") and not (base / name).is_symlink()
            )
            meta = base / META_NAME
            if META_NAME not in filenames or meta.is_symlink() or not meta.is_file():
                continue
            relative = base.relative_to(container).as_posix()
            if relative != ".":
                keys.append(relative)
        return sorted(keys)
    if container.exists():
        raise ValueError(f"path must be a directory or ZIP file: {container}")
    raise FileNotFoundError(f"path does not exist: {container}")


def read_feature_names(
    path: str | os.PathLike[str],
    key: str = "X",
) -> tuple[str, ...] | None:
    """Return feature names for expression matrices, otherwise ``None``.

    ``X`` / ``layers/*`` read ``var``; ``raw/X`` reads ``raw/var``. Embedding
    keys such as ``obsm/*`` and ``obsp/*`` return ``None``. A missing ``var``
    group (bare stores) yields ``None`` rather than an error.
    """
    container = _normalize_container_path(path)
    normalized_key = normalize_matrix_key(key)
    var_path = _feature_names_group(normalized_key)
    if var_path is None:
        return None
    if not _annotation_group_exists(container, var_path):
        return None
    return _read_index_names_zarr(container, var_path, missing_ok=True)


def read_obs_names(
    path: str | os.PathLike[str],
    key: str = "X",
) -> tuple[str, ...] | None:
    """Return observation names for cell-aligned matrices, otherwise ``None``.

    ``X``, ``layers/*``, ``raw/X``, ``obsm/*``, and ``obsp/*`` read the
    container-level ``obs`` index. A missing ``obs`` group (bare stores) yields
    ``None`` rather than an error.
    """
    container = _normalize_container_path(path)
    normalized_key = normalize_matrix_key(key)
    obs_path = _obs_names_group(normalized_key)
    if obs_path is None:
        return None
    if not _annotation_group_exists(container, obs_path):
        return None
    return _read_index_names_zarr(container, obs_path, missing_ok=True)


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


def _obs_names_group(key: str) -> str | None:
    if key == "X" or key == "raw/X" or key.startswith(("layers/", "obsm/", "obsp/")):
        return "obs"
    return None


def _annotation_group_exists(container: Path, group_path: str) -> bool:
    """Return whether a zarr group likely exists without opening the store."""
    markers = (
        f"{group_path}/zarr.json",
        f"{group_path}/.zgroup",
        f"{group_path}/.zarray",
    )
    if container.is_dir():
        target = container.joinpath(*group_path.split("/"))
        return any((target / name).is_file() for name in ("zarr.json", ".zgroup", ".zarray"))
    if not container.is_file():
        return False
    try:
        with zipfile.ZipFile(container, mode="r") as archive:
            names = archive.namelist()
    except zipfile.BadZipFile:
        return False
    return any(marker in names for marker in markers)


def _read_index_names_zarr(
    container: Path,
    var_path: str,
    *,
    missing_ok: bool = False,
) -> tuple[str, ...] | None:
    import zarr
    from anndata._io.zarr import read_dataframe
    from zarr.errors import GroupNotFoundError
    from zarr.storage import ZipStore

    try:
        if container.is_file():
            if not zipfile.is_zipfile(container):
                raise ValueError(f"path is a file but not a ZIP archive: {container}")
            store = ZipStore(str(container), mode="r")
            try:
                root = zarr.open_group(store, mode="r")
                return _read_index_names(root, var_path, read_dataframe, missing_ok=missing_ok)
            finally:
                store.close()
        if container.is_dir():
            root = zarr.open_group(str(container), mode="r")
            return _read_index_names(root, var_path, read_dataframe, missing_ok=missing_ok)
    except GroupNotFoundError:
        if missing_ok:
            return None
        raise
    if container.exists():
        raise ValueError(f"path must be a directory or ZIP file: {container}")
    raise FileNotFoundError(f"path does not exist: {container}")


def _read_index_names(
    root: Any,
    var_path: str,
    read_dataframe: Any,
    *,
    missing_ok: bool = False,
) -> tuple[str, ...] | None:
    group = _get_group(root, var_path)
    if group is None:
        if missing_ok:
            return None
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
