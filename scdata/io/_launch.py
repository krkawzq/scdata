"""Launch a scdata store by parsing its metadata for the Rust databank.

A scdata store is a zarr v2 tree (standard ``.zgroup`` / ``.zarray`` /
``.zattrs`` metadata) with a scdata-specific data layout: instead of one file
per chunk, every chunk of an array is concatenated into a single payload file,
and a ``(offset, length)`` index table is written into the array's ``.zattrs``
under the ``scdata`` key.  The Rust reader opens the payload once and reads
chunks by offset (``ChunkStoreMeta::FileOffset``), which is what the io_uring
fast path is designed for.  Chunk data is therefore not readable by stock
zarr/anndata (which expect per-chunk files); only the metadata is.

This module *launches* the store — it parses metadata only (the JSON
``.zgroup`` / ``.zarray`` / ``.zattrs`` files plus the scdata chunk index) and
produces :class:`~scdata.data._dataset.Dataset` objects ready for the Rust
databank.  It does not decode chunks; that is the Rust reader's job.  It also
does not depend on the ``zarr`` or ``anndata`` packages at runtime: zarr v2
metadata is plain JSON, and a ``.zarr.zip`` is a ZIP_STORED archive of those
JSON files and the payload.

Two containers are supported:

* ``.zarr`` — a directory tree on the filesystem.
* ``.zarr.zip`` — a single ZIP file.  scdata writes it with
  ``ZIP_STORED`` (no compression): the payload bytes are already compressed
  by the array codec, and double-compressing would make chunk offsets
  un-stable and prevent offset-based reads.
"""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scdata.data._dataset import (
    ArrayMeta,
    ArrayOrder,
    ChunkLocation,
    CodecPipeline,
    DataError,
    DenseDataset,
    DType,
    Dataset,
    SparseDataset,
)

__all__ = ["launch", "launch_store", "StoreError", "Store"]


class StoreError(Exception):
    """Raised when a scdata store cannot be parsed or is malformed."""


# ---------------------------------------------------------------------------
# Store abstraction: a minimal read-only view over a zarr v2 tree.
# ---------------------------------------------------------------------------


class Store:
    """Read-only view over a ``.zarr`` directory or ``.zarr.zip`` archive.

    Path keys are POSIX-style (``X/data/.zarray``).  Two backends share this
    interface so the parser is container-agnostic.
    """

    def read_text(self, key: str) -> str:
        """Return the UTF-8 text of a metadata file, or raise ``StoreError``."""
        raise NotImplementedError

    def read_bytes(self, key: str) -> bytes:
        """Return the raw bytes of an entry (payload or chunk), or raise."""
        raise NotImplementedError

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys that start with ``prefix`` (exact directory walk)."""
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def size(self, key: str) -> int:
        """Return the size in bytes of an entry, or raise ``StoreError``."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any held resources (zip handle, file descriptors)."""

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _DirectoryStore(Store):
    """A ``.zarr`` directory tree."""

    def __init__(self, root: Path) -> None:
        if not root.is_dir():
            raise StoreError(f"not a directory: {root}")
        self._root = root
        self._root_resolved = root.resolve()

    def _resolve(self, key: str) -> Path:
        # zarr keys are POSIX; on disk they map to os.path.join.
        rel = key.split("/")
        path = self._root.joinpath(*rel)
        # Defense against ``..`` / absolute segments escaping the store root.
        # Keys reach here only from trusted zarr metadata, but a malformed
        # store must not let a payload path read outside the tree.
        try:
            path.resolve().relative_to(self._root_resolved)
        except ValueError as err:
            raise StoreError(f"key escapes store root: {key!r}") from err
        return path

    def read_text(self, key: str) -> str:
        path = self._resolve(key)
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise StoreError(f"missing metadata entry: {key}")
        except OSError as err:
            raise StoreError(f"cannot read {key}: {err}") from err

    def read_bytes(self, key: str) -> bytes:
        path = self._resolve(key)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            raise StoreError(f"missing payload entry: {key}")
        except OSError as err:
            raise StoreError(f"cannot read {key}: {err}") from err

    def list_keys(self, prefix: str = "") -> list[str]:
        keys: list[str] = []
        for path in self._root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(self._root)
            keys.append("/".join(rel.parts))
        if not prefix:
            return keys
        # Match a directory prefix boundary so "X" does not match "X_extra".
        if prefix.endswith("/"):
            return [k for k in keys if k.startswith(prefix)]
        return [k for k in keys if k == prefix or k.startswith(prefix + "/")]

    def exists(self, key: str) -> bool:
        path = self._resolve(key)
        return path.exists()

    def size(self, key: str) -> int:
        path = self._resolve(key)
        try:
            return path.stat().st_size
        except FileNotFoundError:
            raise StoreError(f"missing payload entry: {key}")
        except OSError as err:
            raise StoreError(f"cannot stat {key}: {err}") from err


class _ZipStore(Store):
    """A ``.zarr.zip`` archive written with ``ZIP_STORED``.

    scdata writes zip archives uncompressed so chunk payload offsets are
    stable and readable directly.  We do not enforce STORED on read (a
    DEFLATE entry still decodes via zipfile), but a scdata-written store is
    always STORED.
    """

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise StoreError(f"not a file: {path}")
        try:
            self._zip = zipfile.ZipFile(path)
        except zipfile.BadZipFile as err:
            raise StoreError(f"bad zip archive: {path}") from err
        for info in self._zip.infolist():
            if info.is_dir():
                continue
            if info.compress_type != zipfile.ZIP_STORED:
                raise StoreError(
                    f"zip entry {info.filename!r} is compressed; "
                    "scdata .zarr.zip stores must use ZIP_STORED"
                )
        self._names = frozenset(self._zip.namelist())

    @staticmethod
    def _normalize(key: str) -> str:
        # Zip entries use forward slashes; strip a leading slash just in case.
        return key.lstrip("/")

    def read_text(self, key: str) -> str:
        name = self._normalize(key)
        if name not in self._names:
            raise StoreError(f"missing metadata entry: {key}")
        try:
            return self._zip.read(name).decode("utf-8")
        except KeyError:
            raise StoreError(f"missing metadata entry: {key}")
        except OSError as err:
            raise StoreError(f"cannot read {key}: {err}") from err

    def read_bytes(self, key: str) -> bytes:
        name = self._normalize(key)
        if name not in self._names:
            raise StoreError(f"missing payload entry: {key}")
        try:
            return self._zip.read(name)
        except KeyError:
            raise StoreError(f"missing payload entry: {key}")
        except OSError as err:
            raise StoreError(f"cannot read {key}: {err}") from err

    def list_keys(self, prefix: str = "") -> list[str]:
        keys = sorted(self._names)
        if not prefix:
            return keys
        if not prefix.endswith("/"):
            return [k for k in keys if k == prefix or k.startswith(prefix + "/")]
        return [k for k in keys if k.startswith(prefix)]

    def exists(self, key: str) -> bool:
        return self._normalize(key) in self._names

    def size(self, key: str) -> int:
        name = self._normalize(key)
        if name not in self._names:
            raise StoreError(f"missing payload entry: {key}")
        try:
            return self._zip.getinfo(name).file_size
        except KeyError:
            raise StoreError(f"missing payload entry: {key}")

    def close(self) -> None:
        self._zip.close()


def _open_store(path: str | os.PathLike[str]) -> Store:
    """Open a ``.zarr`` directory or ``.zarr.zip`` archive as a :class:`Store`."""
    p = Path(os.fspath(path))
    if p.is_dir():
        return _DirectoryStore(p)
    if p.is_file():
        return _ZipStore(p)
    raise StoreError(f"store path does not exist: {p}")


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _read_json(store: Store, key: str) -> Any:
    try:
        return json.loads(store.read_text(key))
    except json.JSONDecodeError as err:
        raise StoreError(f"invalid JSON in {key}: {err}") from err


def _expect_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StoreError(f"{context}: expected JSON object, got {type(value).__name__}")
    return value


def _is_json_int(value: Any) -> bool:
    """True for a JSON integer (rejects bools and floats)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _expect_zarr_v2(meta: dict[str, Any], context: str) -> None:
    fmt = meta.get("zarr_format")
    if fmt != 2:
        raise StoreError(f"{context}: expected zarr_format 2, got {fmt!r}")


# ---------------------------------------------------------------------------
# Parse wrappers: convert data-layer DataError into StoreError so launch()
# only ever raises StoreError.
# ---------------------------------------------------------------------------


def _parse_dtype(value: Any, context: str) -> DType:
    try:
        return DType.parse(value)
    except DataError as err:
        raise StoreError(f"{context}: {err}") from err


def _parse_order(value: Any, context: str) -> ArrayOrder:
    try:
        return ArrayOrder.parse(value)
    except DataError as err:
        raise StoreError(f"{context}: {err}") from err


def _parse_codec(filters: Any, compressor: Any, context: str) -> CodecPipeline:
    try:
        return CodecPipeline.from_zarr(filters, compressor)
    except DataError as err:
        raise StoreError(f"{context}: {err}") from err


# ---------------------------------------------------------------------------
# scdata chunk index (the "magic" on top of zarr v2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ChunkIndex:
    """scdata's per-array concatenated payload index.

    Stored in an array's ``.zattrs`` under ``scdata``:
    ``{"payload": "<relpath>", "offsets": [u64...], "lengths": [u64...]}``.
    The payload file holds every encoded chunk concatenated in C-order
    logical chunk index; edge chunks are cropped to logical extent.
    """

    payload: str
    locations: tuple[ChunkLocation, ...]

    @property
    def num_chunks(self) -> int:
        return len(self.locations)


_SCDATA_KEY = "scdata"


def _parse_chunk_index(store: Store, zattrs: dict[str, Any], array_key: str) -> _ChunkIndex:
    """Extract the scdata payload index from an array's ``.zattrs``."""
    raw = zattrs.get(_SCDATA_KEY)
    if raw is None:
        raise StoreError(
            f"array {array_key!r} has no '{_SCDATA_KEY}' chunk index; "
            f"not a scdata-written store"
        )
    obj = _expect_object(raw, f"{array_key} .zattrs[{_SCDATA_KEY!r}]")
    payload = obj.get("payload")
    if not isinstance(payload, str) or not payload:
        raise StoreError(
            f"{array_key} .zattrs[{_SCDATA_KEY!r}].payload must be a non-empty string"
        )
    offsets = obj.get("offsets")
    lengths = obj.get("lengths")
    if not isinstance(offsets, list) or not isinstance(lengths, list):
        raise StoreError(
            f"{array_key} .zattrs[{_SCDATA_KEY!r}].offsets/lengths must be lists"
        )
    if len(offsets) != len(lengths):
        raise StoreError(
            f"{array_key}: offsets/lengths length mismatch "
            f"({len(offsets)} vs {len(lengths)})"
        )
    locations: list[ChunkLocation] = []
    for i, (off, ln) in enumerate(zip(offsets, lengths)):
        if not _is_json_int(off) or not _is_json_int(ln):
            raise StoreError(
                f"{array_key} .zattrs[{_SCDATA_KEY!r}].offsets/lengths must be integers"
            )
        try:
            locations.append(ChunkLocation(offset=off, length=ln))
        except ValueError as err:
            raise StoreError(f"{array_key}: invalid chunk location {i}: {err}") from err
    return _ChunkIndex(payload=payload, locations=tuple(locations))


def _resolve_payload_path(store: Store, array_key: str, payload: str) -> str:
    """Resolve a payload reference relative to the array's own directory.

    ``payload`` in ``.zattrs`` is a path relative to the array's directory
    (a sibling of the array's ``.zarray``).  For an array at ``X/data``, a
    payload of ``payload.bin`` resolves to ``X/data/payload.bin``; a payload
    of ``../payload.bin`` resolves to ``X/payload.bin``.  For a directory
    store the result is a path within the tree; for a zip store it is the
    in-archive key.
    """
    if payload.startswith("/"):
        raise StoreError(f"payload path {payload!r} referenced by {array_key} is absolute")

    parts: list[str] = []
    for seg in array_key.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            raise StoreError(f"array key {array_key!r} escapes store root")
        parts.append(seg)
    for seg in payload.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if not parts:
                raise StoreError(
                    f"payload path {payload!r} referenced by {array_key} escapes store root"
                )
            parts.pop()
        else:
            parts.append(seg)
    resolved = "/".join(parts)
    if not store.exists(resolved):
        raise StoreError(f"payload file {resolved!r} referenced by {array_key} not found")
    return resolved


def _validate_payload_locations(
    store: Store,
    payload: str,
    locations: tuple[ChunkLocation, ...],
    context: str,
) -> None:
    """Validate that chunk byte ranges are ordered, non-overlapping and in-bounds."""
    size = store.size(payload)
    previous_end = 0
    for i, loc in enumerate(locations):
        end = loc.offset + loc.length
        if loc.offset < previous_end:
            raise StoreError(
                f"{context}: chunk {i} overlaps previous chunk "
                f"(offset {loc.offset} < previous end {previous_end})"
            )
        if end > size:
            raise StoreError(
                f"{context}: chunk {i} range [{loc.offset}, {end}) exceeds "
                f"payload size {size}"
            )
        previous_end = end


# ---------------------------------------------------------------------------
# zarr v2 array metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ZarrArray:
    """Parsed fields of a zarr v2 ``.zarray`` plus scdata chunk index."""

    shape: tuple[int, ...]
    chunk_shape: tuple[int, ...]
    dtype: DType
    order: ArrayOrder
    codec: CodecPipeline
    index: _ChunkIndex


def _parse_zarray(
    store: Store, array_key: str, *, strict: bool = True
) -> tuple[_ZarrArray, str]:
    """Parse ``<array_key>/.zarray`` + ``.zattrs`` (scdata index).

    When ``strict`` is True (the default, used by the Rust-facing reader) the
    chunk-grid size must equal the number of indexed payload chunks — the
    regular-grid invariant the Rust databank requires.  The anndata bridge
    passes ``strict=False`` so it can also read stores with variable-length
    cell-aligned chunks (sparse + ``align_cells``), where the index — not the
    grid — is authoritative.
    """
    zarray_key = f"{array_key}/.zarray"
    zattrs_key = f"{array_key}/.zattrs"

    meta = _expect_object(_read_json(store, zarray_key), zarray_key)
    _expect_zarr_v2(meta, zarray_key)

    shape = _parse_shape(meta.get("shape"), zarray_key)
    chunks = _parse_chunk_shape(meta.get("chunks"), zarray_key)
    if len(shape) != len(chunks):
        raise StoreError(
            f"{zarray_key}: shape rank {len(shape)} != chunks rank {len(chunks)}"
        )
    dtype = _parse_dtype(meta.get("dtype"), zarray_key)
    order = _parse_order(meta.get("order"), zarray_key)
    codec = _parse_codec(meta.get("filters"), meta.get("compressor"), zarray_key)

    zattrs: dict[str, Any] = {}
    if store.exists(zattrs_key):
        zattrs = _expect_object(_read_json(store, zattrs_key), zattrs_key)
    index = _parse_chunk_index(store, zattrs, array_key)

    # The chunk grid must match the number of indexed payload chunks.
    grid = 1
    for s, c in zip(shape, chunks):
        grid *= _ceil_div(s, c)
    if strict and index.num_chunks != grid:
        raise StoreError(
            f"{array_key}: scdata index has {index.num_chunks} chunks but "
            f"chunk grid expects {grid}"
        )

    payload = _resolve_payload_path(store, array_key, index.payload)
    _validate_payload_locations(store, payload, index.locations, array_key)
    return (
        _ZarrArray(
            shape=shape,
            chunk_shape=chunks,
            dtype=dtype,
            order=order,
            codec=codec,
            index=index,
        ),
        payload,
    )


def _parse_shape(raw: Any, context: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise StoreError(f"{context}: shape must be a non-empty list")
    if not all(_is_json_int(s) for s in raw):
        raise StoreError(f"{context}: shape entries must be JSON integers, got {raw!r}")
    shape = tuple(raw)
    if any(s < 0 for s in shape):
        raise StoreError(f"{context}: shape entries must be non-negative, got {shape}")
    return shape


def _parse_chunk_shape(raw: Any, context: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise StoreError(f"{context}: chunks must be a non-empty list")
    if not all(_is_json_int(c) for c in raw):
        raise StoreError(f"{context}: chunks entries must be JSON integers, got {raw!r}")
    chunks = tuple(raw)
    if any(c <= 0 for c in chunks):
        raise StoreError(f"{context}: chunks entries must be positive, got {chunks}")
    return chunks


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _build_array_meta(arr: _ZarrArray, payload: str, context: str) -> ArrayMeta:
    try:
        return ArrayMeta(
            shape=arr.shape,
            chunk_shape=arr.chunk_shape,
            dtype=arr.dtype,
            order=arr.order,
            codec=arr.codec,
            payload_path=payload,
            chunks=arr.index.locations,
        )
    except ValueError as err:
        raise StoreError(f"{context}: {err}") from err


# ---------------------------------------------------------------------------
# anndata X layer: dense vs sparse CSR
# ---------------------------------------------------------------------------


def _normalize_x_encoding(enc: Any) -> str | None:
    """Map an anndata ``encoding-type`` to a scdata canonical name.

    anndata writes ``"array"`` (dense), ``"csr_matrix"`` / ``"csc_matrix"``
    (sparse); scdata's own writer uses the short ``"CSR"`` / ``"CSC"`` forms.
    Both are accepted; anything else returns None.
    """
    if not isinstance(enc, str):
        return None
    n = enc.strip().lower()
    if n == "array":
        return "array"
    if n in ("csr", "csr_matrix"):
        return "CSR"
    if n in ("csc", "csc_matrix"):
        return "CSC"
    return None


def _detect_x_encoding(store: Store, x_key: str) -> str:
    """Read anndata's ``encoding-type`` from ``X/.zattrs``.

    Accepts both anndata's tags (``"array"`` / ``"csr_matrix"`` /
    ``"csc_matrix"``) and scdata's short forms (``"CSR"`` / ``"CSC"``).
    Without a tag, only a dense array (``X/.zarray``) is unambiguous; a
    sparse group cannot be told apart as CSR vs CSC, so we refuse rather
    than guess.
    """
    has_attrs = store.exists(f"{x_key}/.zattrs")
    if has_attrs:
        zattrs = _expect_object(_read_json(store, f"{x_key}/.zattrs"), f"{x_key}/.zattrs")
        encoding_type = zattrs.get("encoding-type")
        norm = _normalize_x_encoding(encoding_type)
        if norm is not None:
            return norm
        if encoding_type is not None:
            raise StoreError(
                f"{x_key}: unsupported 'encoding-type' in .zattrs: {encoding_type!r}"
            )
    # No (or incomplete) attrs: only dense arrays are safe to infer.
    if store.exists(f"{x_key}/.zarray"):
        return "array"
    if has_attrs:
        raise StoreError(
            f"{x_key}: unsupported or missing 'encoding-type' in .zattrs"
        )
    raise StoreError(f"{x_key}: missing .zattrs and not a dense array")


def _read_x_shape_attr(store: Store, x_key: str) -> tuple[int, int] | None:
    """Read anndata sparse matrix shape from ``X/.zattrs`` when present."""
    zattrs_key = f"{x_key}/.zattrs"
    if not store.exists(zattrs_key):
        return None
    zattrs = _expect_object(_read_json(store, zattrs_key), zattrs_key)
    raw = zattrs.get("shape")
    if raw is None:
        return None
    shape = _parse_shape(raw, zattrs_key)
    if len(shape) != 2:
        raise StoreError(f"{x_key}: sparse matrix shape must be 2D, got {shape}")
    return shape[0], shape[1]


def _read_gene_names(store: Store, var_key: str) -> tuple[str, ...]:
    """Read the ``_index`` column of the ``var`` group as gene names.

    anndata stores the var index under ``var/_index`` (a zarr array of
    strings) by default; some stores use ``var/index``.  We read whichever
    is present.  Returns a tuple of gene name strings.

    String arrays are metadata-only: they are not part of the Rust numeric
    ``DType`` model, so they are parsed here directly rather than routed
    through :func:`_parse_zarray` (which produces numeric ``ArrayMeta`` for
    Rust).
    """
    if not store.exists(f"{var_key}/.zgroup"):
        raise StoreError(f"missing var group: {var_key}")
    _expect_zarr_v2(
        _expect_object(_read_json(store, f"{var_key}/.zgroup"), f"{var_key}/.zgroup"),
        f"{var_key}/.zgroup",
    )

    index_key = None
    candidates: list[str] = []
    zattrs_key = f"{var_key}/.zattrs"
    if store.exists(zattrs_key):
        zattrs = _expect_object(_read_json(store, zattrs_key), zattrs_key)
        index_name = zattrs.get("_index")
        if isinstance(index_name, str) and index_name:
            candidates.append(index_name)
    candidates.extend(["_index", "index"])

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if store.exists(f"{var_key}/{candidate}/.zarray"):
            index_key = f"{var_key}/{candidate}"
            break
    if index_key is None:
        raise StoreError(
            f"cannot find var index array under {var_key}/_index or {var_key}/index"
        )

    meta = _expect_object(_read_json(store, f"{index_key}/.zarray"), index_key)
    _expect_zarr_v2(meta, f"{index_key}/.zarray")
    _parse_order(meta.get("order"), index_key)
    dtype_raw = meta.get("dtype")
    if not isinstance(dtype_raw, str):
        raise StoreError(f"{index_key}: dtype must be a string, got {dtype_raw!r}")
    kind = _string_dtype_kind(dtype_raw)
    if kind is None:
        raise StoreError(
            f"{index_key}: var index dtype {dtype_raw!r} is not a string type"
        )

    shape = _parse_shape(meta.get("shape"), index_key)
    if len(shape) != 1:
        raise StoreError(f"{var_key} index must be 1D, got shape {shape}")
    count = shape[0]
    codec = _parse_codec(meta.get("filters"), meta.get("compressor"), index_key)
    if kind == "O" and _find_vlen_utf8_filter(codec.filters) is None:
        raise StoreError(
            f"{index_key}: object dtype string array requires a VLenUTF8 filter"
        )

    zattrs: dict[str, Any] = {}
    zattrs_key = f"{index_key}/.zattrs"
    if store.exists(zattrs_key):
        zattrs = _expect_object(_read_json(store, zattrs_key), zattrs_key)
    index = _parse_chunk_index(store, zattrs, index_key)
    payload = _resolve_payload_path(store, index_key, index.payload)
    _validate_payload_locations(store, payload, index.locations, index_key)

    raw = store.read_bytes(payload)
    names = _decode_string_array(raw, index.locations, dtype_raw, codec, count)
    return tuple(names)


def _as_bytes(value: Any) -> bytes:
    """Coerce a numcodecs decode result (bytes / buffer / array) to bytes."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if hasattr(value, "tobytes"):
        return value.tobytes()
    return bytes(value)


def _string_dtype_kind(dtype: str) -> str | None:
    """Return the string kind of a zarr dtype: ``"S"`` / ``"U"`` / ``"O"`` or None.

    * ``|S<n>`` — fixed-width byte strings (NUL-padded, ``n`` bytes each).
    * ``<U<n>`` / ``>U<n>`` — fixed-width UTF-32 strings (``4*n`` bytes each).
    * ``|O`` — variable-length object strings (anndata's native form, encoded
      with a numcodecs VLenUTF8 filter rather than a fixed byte width).
    """
    text = dtype.strip()
    if text.startswith("|O"):
        return "O"
    if text.startswith("|S"):
        return "S"
    for prefix in ("<U", ">U", "|U"):
        if text.startswith(prefix):
            return "U"
    return None


def _is_vlen_utf8(config: dict[str, Any]) -> bool:
    cid = str(config.get("id", "")).lower().replace("_", "-")
    return cid == "vlen-utf8"


def _find_vlen_utf8_filter(filters: tuple[dict[str, Any], ...]) -> dict[str, Any] | None:
    for flt in filters:
        if _is_vlen_utf8(flt):
            return flt
    return None


def _decode_chunk_bytes(raw: bytes, codec: CodecPipeline) -> bytes:
    """Decode a single encoded chunk: compressor first, then reverse(filters).

    Each chunk is an independent codec frame — the concatenated payload is
    *never* decoded as one block, because most codecs (blosc, lz4, ...) only
    encode the bytes of their own chunk and cannot decode a concatenation of
    frames.  Edge chunks are cropped to logical extent at write time, so chunk
    byte lengths vary.
    """
    if codec.is_uncompressed:
        return raw
    from numcodecs import get_codec

    data = _as_bytes(raw)
    try:
        if codec.compressor is not None:
            data = _as_bytes(get_codec(dict(codec.compressor)).decode(data))
        for flt in reversed(codec.filters):
            data = _as_bytes(get_codec(dict(flt)).decode(data))
    except Exception as err:
        raise StoreError(f"failed to decode chunk with numcodecs: {err}") from err
    return data


def _decode_vlen_chunks(
    payload: bytes,
    chunks: tuple[ChunkLocation, ...],
    codec: CodecPipeline,
) -> list[str]:
    """Decode a per-chunk ``|O`` + VLenUTF8 string array.

    VLenUTF8 is the innermost filter (applied first on encode, last on
    decode); its decode yields a numpy object array of Python ``str``, so it
    cannot go through :func:`_decode_chunk_bytes` (which assumes bytes out).
    The compressor and any other byte-oriented filters are applied first.
    """
    from numcodecs import get_codec

    vlen_filter = _find_vlen_utf8_filter(codec.filters)
    if vlen_filter is None:
        raise StoreError("object dtype string array requires a VLenUTF8 filter")
    other_filters = [f for f in codec.filters if not _is_vlen_utf8(f)]
    vlen_codec = get_codec(dict(vlen_filter))
    out: list[str] = []
    for loc in chunks:
        data = payload[loc.offset:loc.offset + loc.length]
        try:
            if codec.compressor is not None:
                data = _as_bytes(get_codec(dict(codec.compressor)).decode(data))
            for flt in reversed(other_filters):
                data = _as_bytes(get_codec(dict(flt)).decode(data))
            arr = vlen_codec.decode(_as_bytes(data))
        except Exception as err:
            raise StoreError(f"failed to decode VLenUTF8 chunk: {err}") from err
        out.extend(arr.tolist())
    return out


def _decode_string_array(
    payload: bytes,
    chunks: tuple[ChunkLocation, ...],
    dtype: str,
    codec: CodecPipeline,
    count: int,
) -> list[str]:
    """Decode a 1D zarr v2 string array (possibly multi-chunk) to Python strings.

    Dispatches on dtype kind (``|S`` / ``<U`` / ``|O``); each chunk is sliced
    from the payload by its ``(offset, length)`` and decoded independently.
    """
    import numpy as np

    kind = _string_dtype_kind(dtype)
    if kind == "O":
        names = _decode_vlen_chunks(payload, chunks, codec)
    elif kind in ("S", "U"):
        np_dt = np.dtype(dtype)
        item = np_dt.itemsize
        names: list[str] = []
        for loc in chunks:
            raw = payload[loc.offset:loc.offset + loc.length]
            dec = _decode_chunk_bytes(raw, codec)
            if len(dec) % item != 0:
                raise StoreError(
                    f"string chunk decoded to {len(dec)} bytes, "
                    f"not a multiple of itemsize {item}"
                )
            arr = np.frombuffer(dec, dtype=np_dt)
            if kind == "S":
                names.extend(s.decode("utf-8").rstrip("\x00") for s in arr.tolist())
            else:
                names.extend(arr.tolist())
    else:
        raise StoreError(f"unsupported string dtype {dtype!r}")
    if len(names) != count:
        raise StoreError(
            f"string array decoded {len(names)} elements, expected {count}"
        )
    return names


# ---------------------------------------------------------------------------
# Dense / sparse assembly
# ---------------------------------------------------------------------------


def _build_dense_dataset(store: Store, x_key: str, gene_names: tuple[str, ...]) -> DenseDataset:
    """Build a :class:`DenseDataset` from ``X`` stored as a dense zarr array."""
    arr, payload = _parse_zarray(store, x_key)
    meta = _build_array_meta(arr, payload, x_key)

    if len(arr.shape) == 2:
        num_cells, num_genes = arr.shape
    elif len(arr.shape) == 1:
        # Flattened [cells * genes]; gene count comes from var.
        num_genes = len(gene_names)
        if num_genes == 0:
            raise StoreError("dense 1D array but var has no gene names")
        total = arr.shape[0]
        if total % num_genes != 0:
            raise StoreError(
                f"dense 1D length {total} not divisible by gene count {num_genes}"
            )
        num_cells = total // num_genes
    else:
        raise StoreError(f"dense X must be 1D or 2D, got shape {arr.shape}")

    if num_genes != len(gene_names):
        raise StoreError(
            f"X has {num_genes} genes but var has {len(gene_names)} gene names"
        )
    try:
        return DenseDataset(
            gene_names=gene_names,
            data=meta,
            num_cells=num_cells,
            num_genes=num_genes,
        )
    except ValueError as err:
        raise StoreError(f"{x_key}: {err}") from err


def _build_sparse_dataset(
    store: Store, x_key: str, gene_names: tuple[str, ...]
) -> SparseDataset:
    """Build a :class:`SparseDataset` from an anndata CSR ``X`` group."""
    if not store.exists(f"{x_key}/.zgroup"):
        raise StoreError(f"sparse X must be a zarr group: {x_key}")
    _expect_zarr_v2(
        _expect_object(_read_json(store, f"{x_key}/.zgroup"), f"{x_key}/.zgroup"),
        f"{x_key}/.zgroup",
    )
    x_shape = _read_x_shape_attr(store, x_key)

    # indptr: length num_cells+1, kept as a flat vector (not chunked on disk
    # by scdata — it is small and read once).
    indptr_arr, indptr_payload = _parse_zarray(store, f"{x_key}/indptr")
    if len(indptr_arr.shape) != 1:
        raise StoreError(f"{x_key}/indptr must be 1D, got shape {indptr_arr.shape}")
    num_cells = indptr_arr.shape[0] - 1
    indptr_raw = store.read_bytes(indptr_payload)
    indptr = _decode_index_array(
        indptr_raw,
        indptr_arr.index.locations,
        indptr_arr.dtype,
        indptr_arr.codec,
        num_cells + 1,
    )

    # indices / data: 1D arrays of length nnz, chunked along nnz axis.
    indices_arr, indices_payload = _parse_zarray(store, f"{x_key}/indices")
    data_arr, data_payload = _parse_zarray(store, f"{x_key}/data")
    if len(indices_arr.shape) != 1 or len(data_arr.shape) != 1:
        raise StoreError(f"{x_key}/indices and data must be 1D")

    nnz = int(indptr[-1]) if indptr else 0
    if indices_arr.shape[0] != nnz:
        raise StoreError(
            f"{x_key}/indices length {indices_arr.shape[0]} != nnz {nnz}"
        )
    if data_arr.shape[0] != nnz:
        raise StoreError(f"{x_key}/data length {data_arr.shape[0]} != nnz {nnz}")
    if not indices_arr.dtype.is_csr_index:
        raise StoreError(f"{x_key}/indices dtype {indices_arr.dtype!r} not a CSR index")

    if x_shape is not None:
        shape_cells, shape_genes = x_shape
        if shape_cells != num_cells:
            raise StoreError(
                f"{x_key}: X shape has {shape_cells} cells but indptr implies {num_cells}"
            )
        if shape_genes != len(gene_names):
            raise StoreError(
                f"{x_key}: X shape has {shape_genes} genes but var has "
                f"{len(gene_names)} gene names"
            )
        num_genes = shape_genes
    else:
        num_genes = len(gene_names)

    indices_meta = _build_array_meta(indices_arr, indices_payload, f"{x_key}/indices")
    data_meta = _build_array_meta(data_arr, data_payload, f"{x_key}/data")
    try:
        return SparseDataset(
            gene_names=gene_names,
            indptr=tuple(indptr),
            indices=indices_meta,
            data=data_meta,
            index_dtype=indices_arr.dtype,
            num_cells=num_cells,
            num_genes=num_genes,
        )
    except ValueError as err:
        raise StoreError(f"{x_key}: {err}") from err


def _decode_index_array(
    payload: bytes,
    chunks: tuple[ChunkLocation, ...],
    dtype: DType,
    codec: CodecPipeline,
    count: int,
) -> list[int]:
    """Decode a 1D integer array payload (indptr) to a list of Python ints.

    Each chunk is sliced by ``(offset, length)`` and decoded independently;
    the dtype must be an integer type (the CSR indptr is widened to ``u64``
    on the Rust side, so a float dtype is rejected here).
    """
    import numpy as np

    if dtype not in _INTEGER_DTYPES:
        raise StoreError(f"index array dtype {dtype!r} must be an integer type")
    np_dtype = np.dtype(_dtype_to_numpy(dtype))
    item = np_dtype.itemsize
    out: list[int] = []
    for loc in chunks:
        raw = payload[loc.offset:loc.offset + loc.length]
        dec = _decode_chunk_bytes(raw, codec)
        if len(dec) % item != 0:
            raise StoreError(
                f"index chunk decoded to {len(dec)} bytes, "
                f"not a multiple of itemsize {item}"
            )
        arr = np.frombuffer(dec, dtype=np_dtype)
        out.extend(int(x) for x in arr.tolist())
    if len(out) != count:
        raise StoreError(f"index array decoded {len(out)} elements, expected {count}")
    return out


_INTEGER_DTYPES = frozenset(
    {
        DType.U8, DType.I8, DType.U16, DType.I16,
        DType.U32, DType.I32, DType.U64, DType.I64,
    }
)


def _dtype_to_numpy(dtype: DType) -> str:
    return {
        DType.U8: "<u1",
        DType.I8: "<i1",
        DType.U16: "<u2",
        DType.I16: "<i2",
        DType.U32: "<u4",
        DType.I32: "<i4",
        DType.U64: "<u8",
        DType.I64: "<i8",
        DType.F16: "<f2",
        DType.BF16: "<f2",  # no native numpy bf16; interpreted as raw bytes by Rust
        DType.F32: "<f4",
        DType.F64: "<f8",
    }[dtype]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def launch(path: str | os.PathLike[str]) -> Dataset:
    """Launch a scdata store and return its dataset metadata.

    The store must be an anndata-readable zarr v2 tree (``.zarr`` directory
    or ``.zarr.zip`` archive) with scdata chunk indexes under each array's
    ``.zattrs["scdata"]``.  Returns a :class:`DenseDataset` or
    :class:`SparseDataset` carrying every field the Rust databank needs to
    open the store: shape, chunk_shape, dtype, codec pipeline, and the
    concatenated-chunk payload path plus ``(offset, length)`` table.

    This parses metadata only; chunk payloads are not decoded here.
    """
    with _open_store(path) as store:
        return launch_store(store)


def launch_store(store: Store) -> Dataset:
    """Launch an already-open :class:`Store` into a :class:`Dataset`.

    Use :func:`launch` for the common case; this entry point lets callers
    reuse a custom :class:`Store` implementation.
    """
    if not store.exists(".zgroup"):
        if store.exists(".zarr.json"):
            raise StoreError(
                "zarr v3 stores are not yet supported; "
                "scdata reads zarr v2 (.zgroup) stores"
            )
        raise StoreError("not a zarr group (missing .zgroup)")
    _expect_zarr_v2(_expect_object(_read_json(store, ".zgroup"), ".zgroup"), ".zgroup")
    # anndata root attrs are optional for scdata; we only need X and var.
    # X may be a dense array (X/.zarray) or a sparse group (X/.zgroup).
    if not (store.exists("X/.zarray") or store.exists("X/.zgroup")):
        raise StoreError("store has no X array or group")
    if not store.exists("var/.zgroup"):
        raise StoreError("store has no var group")

    # Detect X layout first so a CSC / unsupported-encoding error surfaces
    # before we spend effort reading var gene names.
    encoding = _detect_x_encoding(store, "X")
    if encoding == "CSC":
        raise StoreError("scdata does not read CSC matrices; store as CSR")
    if encoding not in ("array", "CSR"):
        raise StoreError(f"unsupported X encoding-type: {encoding!r}")

    gene_names = _read_gene_names(store, "var")
    if encoding == "array":
        return _build_dense_dataset(store, "X", gene_names)
    return _build_sparse_dataset(store, "X", gene_names)
