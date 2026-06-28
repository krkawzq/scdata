"""AnnData zarr bridge for scdata stores.

AnnData writes a normal zarr v2 tree: each array chunk is a separate file
(``X/0.0``, ``X/data/0``, ``var/_index/0``, ...).  scdata keeps AnnData's zarr
metadata but replaces the chunk-file fanout for Rust-facing arrays with one
``payload.bin`` plus ``.zattrs["scdata"]`` offset/length metadata.

This module owns that conversion.  It deliberately keeps imports light:
``anndata`` is only needed by :func:`write_anndata`; converting an already
written zarr directory is plain JSON and filesystem IO.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from scdata.io._launch import (
    StoreError,
    _expect_object,
    _expect_zarr_v2,
    _normalize_x_encoding,
    _parse_chunk_shape,
    _parse_shape,
)

__all__ = ["convert_anndata_zarr", "write_anndata"]


def write_anndata(
    adata: Any,
    path: str | os.PathLike[str],
    *,
    chunks: tuple[int, ...] | None = None,
    convert_strings_to_categoricals: bool = True,
    overwrite: bool = True,
    keep_zarr_chunks: bool = False,
) -> Path:
    """Write an :class:`anndata.AnnData` object as a scdata zarr directory.

    The first step delegates to ``adata.write_zarr`` so AnnData controls its own
    dataframe, CSR, string-array and encoding metadata.  The second step
    rewrites only the arrays the Rust databank needs into scdata's concatenated
    payload layout.
    """
    root = Path(os.fspath(path))
    if root.exists():
        if not overwrite:
            raise StoreError(f"output path already exists: {root}")
        if root.is_dir() and not root.is_symlink():
            shutil.rmtree(root)
        else:
            root.unlink()

    adata.write_zarr(
        root,
        chunks=chunks,
        convert_strings_to_categoricals=convert_strings_to_categoricals,
    )
    return convert_anndata_zarr(root, keep_zarr_chunks=keep_zarr_chunks)


def convert_anndata_zarr(
    path: str | os.PathLike[str],
    *,
    array_keys: tuple[str, ...] | None = None,
    keep_zarr_chunks: bool = False,
    payload_name: str = "payload.bin",
) -> Path:
    """Convert a filesystem AnnData zarr v2 store to scdata's payload layout.

    By default this converts ``X`` (dense) or ``X/data`` / ``X/indices`` /
    ``X/indptr`` (CSR), plus the AnnData-selected ``var`` index array.  Other
    AnnData groups are left untouched.
    """
    root = Path(os.fspath(path))
    if not root.is_dir():
        raise StoreError(f"AnnData zarr store must be a directory: {root}")
    _expect_zarr_v2(_expect_object(_read_json(root / ".zgroup"), ".zgroup"), ".zgroup")

    keys = array_keys if array_keys is not None else _default_array_keys(root)
    for key in keys:
        _convert_array(root, key, payload_name=payload_name, keep_zarr_chunks=keep_zarr_chunks)

    # AnnData writes consolidated metadata.  It is stale after we add scdata
    # attrs and optionally remove chunk files, so drop it instead of leaving a
    # misleading cache for zarr readers.
    zmetadata = root / ".zmetadata"
    if zmetadata.exists():
        zmetadata.unlink()
    return root


def _default_array_keys(root: Path) -> tuple[str, ...]:
    keys: list[str] = []

    x_key = root / "X"
    if (x_key / ".zarray").exists():
        keys.append("X")
    elif (x_key / ".zgroup").exists():
        zattrs = _expect_object(_read_json(x_key / ".zattrs"), "X/.zattrs")
        encoding = _normalize_x_encoding(zattrs.get("encoding-type"))
        if encoding == "CSR":
            keys.extend(["X/data", "X/indices", "X/indptr"])
        elif encoding == "CSC":
            raise StoreError("scdata does not write CSC matrices; convert AnnData.X to CSR")
        else:
            raise StoreError(f"unsupported AnnData X encoding-type: {zattrs.get('encoding-type')!r}")
    else:
        raise StoreError("AnnData zarr store has no X array or group")

    keys.append(_dataframe_index_array_key(root, "var"))
    return tuple(dict.fromkeys(keys))


def _dataframe_index_array_key(root: Path, group_key: str) -> str:
    group = root / group_key
    if not (group / ".zgroup").exists():
        raise StoreError(f"AnnData zarr store has no {group_key!r} group")

    candidates: list[str] = []
    zattrs_path = group / ".zattrs"
    if zattrs_path.exists():
        zattrs = _expect_object(_read_json(zattrs_path), f"{group_key}/.zattrs")
        index_name = zattrs.get("_index")
        if isinstance(index_name, str) and index_name:
            candidates.append(index_name)
    candidates.extend(["_index", "index"])

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (group / candidate / ".zarray").exists():
            return f"{group_key}/{candidate}"
    raise StoreError(f"cannot find {group_key} index array")


def _convert_array(
    root: Path,
    key: str,
    *,
    payload_name: str,
    keep_zarr_chunks: bool,
) -> None:
    array_dir = root.joinpath(*key.split("/"))
    zarray_path = array_dir / ".zarray"
    zattrs_path = array_dir / ".zattrs"
    if not zarray_path.exists():
        raise StoreError(f"{key}: missing .zarray")

    meta = _expect_object(_read_json(zarray_path), f"{key}/.zarray")
    _expect_zarr_v2(meta, f"{key}/.zarray")
    shape = _parse_shape(meta.get("shape"), f"{key}/.zarray")
    chunks = _parse_chunk_shape(meta.get("chunks"), f"{key}/.zarray")
    if len(shape) != len(chunks):
        raise StoreError(f"{key}: shape rank {len(shape)} != chunks rank {len(chunks)}")
    if len(shape) == 0:
        raise StoreError(f"{key}: scalar zarr arrays are not valid scdata arrays")

    chunk_keys = tuple(_chunk_key(coord, meta.get("dimension_separator", ".")) for coord in _chunk_coords(shape, chunks))
    chunk_paths = tuple(_chunk_path(array_dir, chunk_key) for chunk_key in chunk_keys)
    missing = [p for p in chunk_paths if not p.is_file()]

    zattrs = _expect_object(_read_json(zattrs_path), f"{key}/.zattrs") if zattrs_path.exists() else {}
    existing_scdata = zattrs.get("scdata")
    if missing:
        payload_path = array_dir / payload_name
        if len(missing) == len(chunk_paths) and isinstance(existing_scdata, dict) and payload_path.exists():
            return
        shown = ", ".join(str(p.relative_to(root)) for p in missing[:3])
        suffix = "" if len(missing) <= 3 else f" and {len(missing) - 3} more"
        raise StoreError(f"{key}: missing zarr chunk file(s): {shown}{suffix}")

    payload = bytearray()
    offsets: list[int] = []
    lengths: list[int] = []
    for chunk_file in chunk_paths:
        data = chunk_file.read_bytes()
        offsets.append(len(payload))
        lengths.append(len(data))
        payload.extend(data)

    (array_dir / payload_name).write_bytes(payload)
    zattrs["scdata"] = {"payload": payload_name, "offsets": offsets, "lengths": lengths}
    _write_json(zattrs_path, zattrs)

    if not keep_zarr_chunks:
        for chunk_file in chunk_paths:
            chunk_file.unlink()
        _prune_empty_dirs(array_dir)


def _chunk_coords(shape: tuple[int, ...], chunks: tuple[int, ...]):
    grid = tuple(_ceil_div(s, c) for s, c in zip(shape, chunks))
    total = 1
    for g in grid:
        total *= g
    for linear in range(total):
        coord = [0] * len(grid)
        x = linear
        for axis in range(len(grid) - 1, -1, -1):
            coord[axis] = x % grid[axis]
            x //= grid[axis]
        yield tuple(coord)


def _chunk_key(coord: tuple[int, ...], separator: Any) -> str:
    sep = "." if separator is None else str(separator)
    if sep not in (".", "/"):
        raise StoreError(f"unsupported zarr dimension_separator: {separator!r}")
    return sep.join(str(c) for c in coord)


def _chunk_path(array_dir: Path, chunk_key: str) -> Path:
    return array_dir.joinpath(*chunk_key.split("/"))


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StoreError(f"missing metadata entry: {path}")
    except json.JSONDecodeError as err:
        raise StoreError(f"invalid JSON in {path}: {err}") from err
    except OSError as err:
        raise StoreError(f"cannot read {path}: {err}") from err


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prune_empty_dirs(root: Path) -> None:
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass
