"""Launch a zarr store by parsing metadata for the Rust databank.

This module is the Python-side adapter between zarr/anndata metadata and the
Rust ``databank`` factories.  It reads metadata, resolves chunk addresses, and
returns :class:`~scdata.data._dataset.Dataset` objects; it does not decode
numeric chunks.  Dense/sparse dataset semantics are interpreted here, while the
Rust core receives normalized array metadata plus per-chunk file/off/len
locations.

Supported inputs:

* zarr v3 stores written by :func:`scdata.io.write_zarr` or compatible anndata
  writers.  Chunk files are standard zarr files; zip stores are mapped to the
  zip archive path plus each entry's physical byte offset.
"""

from __future__ import annotations

import json
import os
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from scdata.data._dataset import (
    ArrayMeta,
    ArrayOrder,
    CodecPipeline,
    DataError,
    DenseDataset,
    DType,
    Dataset,
    DatasetCollection,
    SparseDataset,
)

try:
    from scdata._scdata import (
        _decode_index_chunks,
        _zip_stored_offsets,
    )
except Exception:  # pragma: no cover - compatibility with older Rust wheels.
    _decode_index_chunks = None
    _zip_stored_offsets = None

__all__ = ["launch", "launch_all", "launch_store", "launch_store_all", "StoreError", "Store"]


class StoreError(Exception):
    """Raised when a scdata store cannot be parsed or is malformed."""


# ---------------------------------------------------------------------------
# Store abstraction: a minimal read-only view over a zarr tree.
# ---------------------------------------------------------------------------


class Store:
    """Read-only view over a ``.zarr`` directory or ``.zarr.zip`` archive.

    Path keys are POSIX-style (``X/data/c/0``).  Two backends share this
    interface so the parser is container-agnostic.
    """

    def read_text(self, key: str) -> str:
        """Return the UTF-8 text of a metadata file, or raise ``StoreError``."""
        raise NotImplementedError

    def read_bytes(self, key: str) -> bytes:
        """Return the raw bytes of an entry, or raise."""
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
        # store must not let a chunk path read outside the tree.
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
            raise StoreError(f"missing store entry: {key}")
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
            raise StoreError(f"missing store entry: {key}")
        except OSError as err:
            raise StoreError(f"cannot stat {key}: {err}") from err

    def source_path(self, key: str) -> str:
        """Filesystem file Rust should open for ``key``."""
        return os.fspath(self._resolve(key))


class _ZipStore(Store):
    """A ``.zarr.zip`` archive written with ``ZIP_STORED``.

    scdata writes zip archives uncompressed so chunk offsets are
    stable and readable directly.  We do not enforce STORED on read (a
    DEFLATE entry still decodes via zipfile), but a scdata-written store is
    always STORED.
    """

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise StoreError(f"not a file: {path}")
        self._path = path
        try:
            self._zip = zipfile.ZipFile(path)
        except zipfile.BadZipFile as err:
            raise StoreError(f"bad zip archive: {path}") from err
        infos: list[zipfile.ZipInfo] = []
        for info in self._zip.infolist():
            if info.is_dir():
                continue
            if info.compress_type != zipfile.ZIP_STORED:
                raise StoreError(
                    f"zip entry {info.filename!r} is compressed; "
                    "scdata .zarr.zip stores must use ZIP_STORED"
                )
            infos.append(info)
        self._names = frozenset(info.filename for info in infos)
        self._sizes = {info.filename: int(info.file_size) for info in infos}
        self._offsets: dict[str, int] = {}
        if _zip_stored_offsets is not None and infos:
            try:
                offsets = _zip_stored_offsets(
                    os.fspath(path), [int(info.header_offset) for info in infos]
                )
            except Exception as err:
                raise StoreError(f"cannot build zip manifest for {path}: {err}") from err
            self._offsets = {
                info.filename: int(offset) for info, offset in zip(infos, offsets, strict=True)
            }

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
            raise StoreError(f"missing store entry: {key}")
        try:
            return self._zip.read(name)
        except KeyError:
            raise StoreError(f"missing store entry: {key}")
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
            raise StoreError(f"missing store entry: {key}")
        return self._sizes[name]

    def source_path(self, key: str) -> str:
        """Filesystem file Rust should open for an in-archive key."""
        return os.fspath(self._path)

    def chunk_offset(self, key: str) -> int:
        """Physical byte offset of a ZIP_STORED entry's data."""
        name = self._normalize(key)
        if name not in self._names:
            raise StoreError(f"missing store entry: {key}")
        offset = self._offsets.get(name)
        if offset is not None:
            return offset
        info = self._zip.getinfo(name)
        try:
            with self._path.open("rb") as fh:
                fh.seek(info.header_offset)
                header = fh.read(30)
        except OSError as err:
            raise StoreError(f"cannot read zip local header for {key}: {err}") from err
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            raise StoreError(f"invalid zip local header for {key}")
        filename_len, extra_len = struct.unpack_from("<HH", header, 26)
        return int(info.header_offset + 30 + filename_len + extra_len)

    def chunk_offsets(self, keys: tuple[str, ...], lengths: tuple[int, ...]) -> tuple[int, ...]:
        """Physical data offsets for many ZIP_STORED entries."""
        return tuple(
            0 if length == 0 else self.chunk_offset(key)
            for key, length in zip(keys, lengths, strict=True)
        )

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


def _read_json(store: Store, key: str) -> object:
    try:
        return json.loads(store.read_text(key))
    except json.JSONDecodeError as err:
        raise StoreError(f"invalid JSON in {key}: {err}") from err


def _expect_object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise StoreError(f"{context}: expected JSON object, got {type(value).__name__}")
    return value


def _is_json_int(value: object) -> bool:
    """True for a JSON integer (rejects bools and floats)."""
    return isinstance(value, int) and not isinstance(value, bool)


# ---------------------------------------------------------------------------
# Parse wrappers: convert data-layer DataError into StoreError so launch()
# only ever raises StoreError.
# ---------------------------------------------------------------------------


def _parse_dtype(value: object, context: str) -> DType:
    try:
        return DType.parse(value)
    except DataError as err:
        raise StoreError(f"{context}: {err}") from err


def _parse_order(value: object, context: str) -> ArrayOrder:
    try:
        return ArrayOrder.parse(value)
    except DataError as err:
        raise StoreError(f"{context}: {err}") from err


# ---------------------------------------------------------------------------
# zarr v3 array metadata (``zarr.json`` per node, standard chunk files)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _V3Array:
    """Parsed fields of a zarr v3 array ``zarr.json`` plus chunk-file listing.

    Edge chunks are stored padded to ``chunk_shape`` with the fill value
    (standard zarr v3); the Rust databank decodes the full padded chunk and
    the access planner crops to the logical extent.
    """

    shape: tuple[int, ...]
    chunk_shape: tuple[int, ...]
    dtype: DType
    order: ArrayOrder
    codec: CodecPipeline
    #: One store-root-relative path per chunk, C-order logical index.
    chunk_paths: tuple[str, ...]
    #: Encoded byte size of each chunk file (parallel to ``chunk_paths``).
    chunk_lengths: tuple[int, ...]
    #: Raw ``attributes`` dict from ``zarr.json`` (encoding-type, shape, ...).
    attrs: dict[str, object]
    #: Explicit cumulative rectilinear boundaries. Empty for regular grids.
    chunk_boundaries: tuple[tuple[int, ...], ...] = ()
    #: Whether the chunk grid is rectilinear (variable-length chunks).
    rectilinear: bool = False


def _v3_node_type(meta: dict[str, object], context: str) -> str:
    nt = meta.get("node_type")
    if not isinstance(nt, str):
        raise StoreError(f"{context}: zarr.json missing 'node_type'")
    return nt


def _v3_dtype(meta: dict[str, object], context: str) -> DType:
    """Parse a v3 ``data_type`` field into a :class:`DType`.

    v3 uses bare type strings (``"float32"``, ``"int64"``) rather than the v2
    endianness-prefixed form (``"<f4"``).  anndata writes little-endian
    implicitly via the ``bytes`` codec; scdata stores are little-endian, so we
    map the bare name through the existing dtype decoder by synthesizing the
    little-endian form.  String arrays (``data_type: "string"``) are rejected
    here — they are metadata-only and handled by the gene-name reader.
    """
    raw = meta.get("data_type")
    if isinstance(raw, dict):
        # Parameterized type: {"name": "...", "configuration": {...}}.
        name = raw.get("name")
    else:
        name = raw
    if not isinstance(name, str):
        raise StoreError(f"{context}: zarr.json data_type must be a string, got {raw!r}")
    if name in ("string", "variable_length_utf8"):
        raise StoreError(
            f"{context}: string arrays are metadata-only; route through the gene-name reader"
        )
    # Map the v3 bare name to the little-endian zarr v2 dtype string the
    # existing DType.parse understands.
    v3_to_v2 = {
        "bool": "|b1",
        "int8": "|i1",
        "uint8": "|u1",
        "int16": "<i2",
        "uint16": "<u2",
        "int32": "<i4",
        "uint32": "<u4",
        "int64": "<i8",
        "uint64": "<u8",
        "float16": "<f2",
        "float32": "<f4",
        "float64": "<f8",
        "bfloat16": "bf2",
    }
    dtype_str = v3_to_v2.get(name)
    if dtype_str is None:
        raise StoreError(f"{context}: unsupported v3 data_type {name!r}")
    return _parse_dtype(dtype_str, context)


def _v3_codec_pipeline(codecs: object, context: str) -> CodecPipeline:
    """Convert a v3 ``codecs`` list into a numcodecs :class:`CodecPipeline`.

    v3 stores the whole pipeline as one ``codecs`` list: an ArrayBytes
    serializer (``bytes`` / ``vlen-utf8``) followed by zero or more
    BytesBytes compressors.  The serializer is a byte-layout codec that Rust
    does not re-implement (data is already little-endian on disk), so it is
    dropped; only the compressors are mapped to numcodecs configs that Rust
    rebuilds via ``codec_pipeline_from_zarr_v2_json_str``.

    Supports the compressors anndata/zarr write by default (``zstd``, ``blosc``,
    ``lz4``-via-blosc) plus an uncompressed pipeline (``bytes`` only).
    """
    if not isinstance(codecs, list):
        raise StoreError(f"{context}: zarr.json codecs must be a list, got {type(codecs).__name__}")
    compressors: list[dict[str, Any]] = []
    for entry in codecs:
        if not isinstance(entry, dict):
            raise StoreError(f"{context}: codec entry must be an object, got {entry!r}")
        name = entry.get("name")
        config = entry.get("configuration")
        if not isinstance(name, str):
            raise StoreError(f"{context}: codec entry missing 'name'")
        cfg: dict[str, Any] = dict(config) if isinstance(config, dict) else {}
        if name == "bytes":
            # Serializer (endian handling) — data is little-endian on disk;
            # Rust reads raw bytes.  No numcodecs filter produced.
            continue
        if name == "vlen-utf8":
            # String serializer (ArrayBytesCodec for ``data_type: "string"``).
            # String arrays are decoded by the gene-name reader, which calls
            # numcodecs VLenUTF8 directly; nothing is added to the Rust
            # numeric codec pipeline here.
            continue
        cid = _v3_codec_id(name)
        if cid is None:
            raise StoreError(f"{context}: unsupported v3 codec {name!r}")
        numcodecs_cfg = _v3_to_numcodecs(cid, cfg, context)
        if numcodecs_cfg is not None:
            compressors.append(numcodecs_cfg)
    if not compressors:
        return CodecPipeline()
    if len(compressors) == 1:
        return CodecPipeline(compressor=compressors[0])
    # v3 allows multiple BytesBytes codecs; scdata's Rust pipeline applies
    # filters-then-compressor, so fold extra compressors into the filter list
    # in reverse decode order (last listed = decoded first).
    return CodecPipeline(
        filters=tuple(reversed(compressors[:-1])),
        compressor=compressors[-1],
    )


def _v3_codec_id(name: str) -> str | None:
    """Map a v3 codec name to its numcodecs ``id`` (or None if not a compressor)."""
    return {
        "zstd": "zstd",
        "blosc": "blosc",
        "lz4": "lz4",
        "gzip": "gzip",
        "zlib": "zlib",
    }.get(name)


def _v3_to_numcodecs(codec_id: str, cfg: dict[str, Any], context: str) -> dict[str, Any] | None:
    """Translate a v3 compressor configuration to a numcodecs config dict."""
    if codec_id == "zstd":
        return {
            "id": "zstd",
            "level": int(cfg.get("level", 0)),
            "checksum": bool(cfg.get("checksum", False)),
        }
    if codec_id == "blosc":
        return {
            "id": "blosc",
            "cname": str(cfg.get("cname", "lz4")),
            "clevel": int(cfg.get("clevel", 5)),
            "shuffle": int(cfg.get("shuffle", 1)),
            "blocksize": int(cfg.get("blocksize", 0)),
            "typesize": int(cfg.get("typesize", 1)),
        }
    if codec_id == "lz4":
        # numcodecs has no standalone lz4 compressor entry that zarr v3 emits;
        # lz4 in v3 is normally via blosc.  Fall back to blosc-lz4.
        return {
            "id": "blosc",
            "cname": "lz4",
            "clevel": int(cfg.get("acceleration", 1)),
            "shuffle": 0,
            "blocksize": 0,
            "typesize": 1,
        }
    if codec_id == "gzip":
        return {"id": "gzip", "level": int(cfg.get("level", 5))}
    if codec_id == "zlib":
        return {"id": "zlib", "level": int(cfg.get("level", 5))}
    raise StoreError(f"{context}: cannot translate codec {codec_id!r} to numcodecs")


def _v3_chunk_key(separator: str, coord: tuple[int, ...], default_encoding: bool) -> str:
    """Build the on-disk chunk key for one chunk coordinate under v3.

    ``default_encoding=True`` uses the v3 default (``c`` prefix + separator),
    e.g. ``c/0/0``; ``False`` uses the v2-style encoding (no prefix), e.g.
    ``0.0``.  Both are legal v3; anndata writes the default form.
    """
    if default_encoding:
        return "c" + separator + separator.join(str(c) for c in coord)
    return separator.join(str(c) for c in coord)


def _chunk_coords(shape: tuple[int, ...], chunks: tuple[int, ...]):
    """Yield C-order chunk grid coordinates for ``shape`` / ``chunks``."""
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


def _v3_chunk_files(
    store: Store,
    array_key: str,
    shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    chunk_key_encoding: object,
    context: str,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """List the chunk files for a v3 array in C-order logical index.

    Returns ``(paths, lengths)`` where ``paths`` are store-root-relative keys
    and ``lengths`` are each chunk file's encoded byte size.  Missing chunk
    files are treated as zero-length (fill-value-only) chunks, matching zarr's
    "absent chunk == all fill value" semantics — but scdata-written stores
    always materialize every chunk, so a missing file on a scdata store is an
    error surfaced elsewhere.
    """
    sep = "/"
    default_encoding = True
    if isinstance(chunk_key_encoding, dict):
        name = chunk_key_encoding.get("name")
        cfg = chunk_key_encoding.get("configuration")
        if isinstance(name, str):
            if name == "v2":
                default_encoding = False
            elif name == "default":
                default_encoding = True
            else:
                raise StoreError(f"{context}: unsupported chunk_key_encoding {name!r}")
        if isinstance(cfg, dict):
            s = cfg.get("separator")
            if isinstance(s, str) and s in (".", "/"):
                sep = s

    paths: list[str] = []
    lengths: list[int] = []
    for coord in _chunk_coords(shape, chunk_shape):
        key = f"{array_key}/{_v3_chunk_key(sep, coord, default_encoding)}"
        if store.exists(key):
            paths.append(key)
            lengths.append(store.size(key))
        else:
            # Absent chunk: zarr treats it as all-fill-value.  We record a
            # zero-length placeholder so the chunk grid stays aligned; the
            # databank does not decode zero-length chunks (fill value only).
            paths.append(key)
            lengths.append(0)
    return tuple(paths), tuple(lengths)


def _v3_rectilinear_edges(
    grid_cfg: dict[str, object], shape: tuple[int, ...], context: str
) -> tuple[int, ...]:
    """Expand a v3 rectilinear chunk grid config into a flat edge list.

    scdata's cell-aligned CSR arrays are 1D, so we only need the first axis's
    edge list.  Each edge is a chunk's element count; their sum must equal
    ``shape[0]``.  Edges may be RLE-encoded (``[[count, value], ...]``) or a
    bare list of ints.
    """
    chunk_shapes = grid_cfg.get("chunk_shapes")
    if not isinstance(chunk_shapes, list) or not chunk_shapes:
        raise StoreError(f"{context}: rectilinear chunk_grid needs chunk_shapes")
    axis = chunk_shapes[0]
    edges: list[int] = []
    if isinstance(axis, int):
        # Bare int shorthand: a regular step that repeats to cover the axis.
        total = shape[0] if shape else 0
        n = _ceil_div(total, axis) if axis else 0
        edges = [axis] * n
    elif isinstance(axis, list):
        for e in axis:
            if isinstance(e, int):
                edges.append(e)
            elif isinstance(e, list) and len(e) == 2:
                # RLE: [count, value].
                count, value = e
                edges.extend([value] * count)
            else:
                raise StoreError(f"{context}: bad rectilinear edge {e!r}")
    else:
        raise StoreError(f"{context}: bad rectilinear axis spec {axis!r}")
    if not edges:
        raise StoreError(f"{context}: rectilinear edge lengths must not be empty")
    if any(edge <= 0 for edge in edges):
        raise StoreError(f"{context}: rectilinear edge lengths must be positive")
    total = sum(edges)
    expected = shape[0] if shape else 0
    if total != expected:
        raise StoreError(f"{context}: rectilinear edge lengths sum to {total}, expected {expected}")
    return tuple(edges)


def _rectilinear_boundaries(edges: tuple[int, ...]) -> tuple[int, ...]:
    out = [0]
    for edge in edges:
        out.append(out[-1] + edge)
    return tuple(out)


def _v3_rectilinear_chunk_files(
    store: Store,
    array_key: str,
    num_chunks: int,
    chunk_key_encoding: object,
    context: str,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """List the chunk files for a 1D rectilinear array (one file per edge)."""
    sep = "/"
    default_encoding = True
    if isinstance(chunk_key_encoding, dict):
        name = chunk_key_encoding.get("name")
        cfg = chunk_key_encoding.get("configuration")
        if isinstance(name, str) and name == "v2":
            default_encoding = False
        if isinstance(cfg, dict):
            s = cfg.get("separator")
            if isinstance(s, str) and s in (".", "/"):
                sep = s
    paths: list[str] = []
    lengths: list[int] = []
    for i in range(num_chunks):
        coord = (i,)
        key = f"{array_key}/{_v3_chunk_key(sep, coord, default_encoding)}"
        if store.exists(key):
            paths.append(key)
            lengths.append(store.size(key))
        else:
            paths.append(key)
            lengths.append(0)
    return tuple(paths), tuple(lengths)


def _parse_v3_array(store: Store, array_key: str) -> _V3Array:
    """Parse a v3 array ``zarr.json`` and list its chunk files."""
    key = f"{array_key}/zarr.json"
    meta = _expect_object(_read_json(store, key), key)
    if _v3_node_type(meta, key) != "array":
        raise StoreError(f"{key}: expected node_type 'array'")
    shape = _parse_shape(meta.get("shape"), key)
    chunk_grid = meta.get("chunk_grid")
    if not isinstance(chunk_grid, dict):
        raise StoreError(f"{key}: chunk_grid must be an object")
    grid_name = chunk_grid.get("name")
    grid_cfg = chunk_grid.get("configuration")
    if not isinstance(grid_cfg, dict):
        raise StoreError(f"{key}: chunk_grid missing configuration")

    rectilinear = False
    if grid_name == "regular":
        chunk_shape = _parse_chunk_shape(grid_cfg.get("chunk_shape"), key)
        chunk_paths, chunk_lengths = _v3_chunk_files(
            store,
            array_key,
            shape,
            chunk_shape,
            meta.get("chunk_key_encoding"),
            key,
        )
        chunk_boundaries: tuple[tuple[int, ...], ...] = ()
    elif grid_name == "rectilinear":
        # Variable-length chunk grid: the configuration lists per-axis edge
        # lengths.  scdata uses this for cell-aligned CSR 1D arrays; each edge
        # is one chunk file.  ``chunk_shape`` is a placeholder (first edge).
        rectilinear = True
        edges = _v3_rectilinear_edges(grid_cfg, shape, key)
        chunk_shape = (edges[0],) if edges else (1,)
        chunk_boundaries = (_rectilinear_boundaries(edges),)
        chunk_paths, chunk_lengths = _v3_rectilinear_chunk_files(
            store,
            array_key,
            len(edges),
            meta.get("chunk_key_encoding"),
            key,
        )
    else:
        raise StoreError(f"{key}: unsupported chunk_grid name {grid_name!r}")

    if len(shape) != len(chunk_shape):
        raise StoreError(f"{key}: shape rank {len(shape)} != chunks rank {len(chunk_shape)}")
    _validate_absent_chunk_fill_value(meta.get("fill_value"), chunk_lengths, key)
    dtype = _v3_dtype(meta, key)
    order = _parse_order(meta.get("order"), key)
    codec = _v3_codec_pipeline(meta.get("codecs"), key)
    attrs = (
        _expect_object(meta.get("attributes"), f"{key} attributes")
        if isinstance(meta.get("attributes"), dict)
        else {}
    )
    return _V3Array(
        shape=shape,
        chunk_shape=chunk_shape,
        dtype=dtype,
        order=order,
        codec=codec,
        chunk_paths=chunk_paths,
        chunk_lengths=chunk_lengths,
        attrs=attrs,
        chunk_boundaries=chunk_boundaries,
        rectilinear=rectilinear,
    )


def _validate_absent_chunk_fill_value(
    fill_value: object,
    chunk_lengths: tuple[int, ...],
    context: str,
) -> None:
    """Reject absent chunks when their zarr fill value is not zero."""
    if not any(length == 0 for length in chunk_lengths):
        return
    if _fill_value_is_zero(fill_value):
        return
    raise StoreError(
        f"{context}: absent chunks require zero fill_value for databank access; got {fill_value!r}"
    )


def _fill_value_is_zero(value: object) -> bool:
    """Whether a zarr JSON fill_value represents numeric zero."""
    if value is None:
        return True
    if isinstance(value, bool):
        return value is False
    if isinstance(value, (int, float)):
        return value == 0
    if isinstance(value, list):
        return all(_fill_value_is_zero(item) for item in value)
    return False


def _v3_array_to_meta(arr: _V3Array, store: Store) -> ArrayMeta:
    """Build a ``store_kind="dir"`` :class:`ArrayMeta` from a v3 array.

    ``store`` is consulted for per-chunk byte offsets: a directory store reads
    each chunk from offset 0, while a ``.zarr.zip`` store reads each chunk from
    its physical offset inside the zip archive.
    """
    chunk_offsets = _chunk_offsets(store, arr.chunk_paths, arr.chunk_lengths)
    chunk_file_paths = _chunk_source_paths(store, arr.chunk_paths)
    variable = _is_rectilinear(arr)
    return ArrayMeta.from_directory(
        shape=arr.shape,
        chunk_shape=arr.chunk_shape,
        dtype=arr.dtype,
        chunk_paths=arr.chunk_paths,
        chunk_file_paths=chunk_file_paths,
        chunk_lengths=arr.chunk_lengths,
        order=arr.order,
        codec=arr.codec,
        variable_chunks=variable,
        chunk_boundaries=arr.chunk_boundaries,
        chunk_offsets=chunk_offsets,
    )


def _is_rectilinear(arr: _V3Array) -> bool:
    """True if the array uses a rectilinear (variable-length) chunk grid."""
    # Stored on the parsed _V3Array via the raw metadata; we re-detect by
    # checking the chunk grid name carried alongside the array.  The launch
    # path sets this attribute when it parses the chunk grid.
    return bool(getattr(arr, "rectilinear", False))


def _chunk_offsets(
    store: Store, chunk_paths: tuple[str, ...], chunk_lengths: tuple[int, ...]
) -> tuple[int, ...]:
    """Per-chunk byte offset within its file.

    For a directory store (one file per chunk) this is always 0.  For a
    ``.zarr.zip`` store every chunk lives in the same zip file, so this is the
    chunk's physical byte offset inside the archive — the Rust reader preads
    the chunk directly out of the zip file.
    """
    if hasattr(store, "chunk_offsets"):
        return cast(Any, store).chunk_offsets(chunk_paths, chunk_lengths)
    if hasattr(store, "chunk_offset"):
        # Only ``_ZipStore`` exposes per-entry physical offsets; ``hasattr``
        # gates the duck-typed call, and the cast narrows past it for typing.
        offset_store = cast(Any, store)
        return tuple(
            0 if length == 0 else offset_store.chunk_offset(path)
            for path, length in zip(chunk_paths, chunk_lengths)
        )
    return tuple(0 for _ in chunk_paths)


def _chunk_source_paths(store: Store, chunk_paths: tuple[str, ...]) -> tuple[str, ...]:
    """Local files Rust should open for per-chunk zarr keys."""
    if hasattr(store, "source_path"):
        path_store = cast(Any, store)
        return tuple(path_store.source_path(path) for path in chunk_paths)
    return ()


# ---------------------------------------------------------------------------
# v3 dense / sparse dataset assembly
# ---------------------------------------------------------------------------


def _v3_build_dense_dataset(
    store: Store, x_key: str, gene_names: tuple[str, ...], store_root: str
) -> DenseDataset:
    """Build a :class:`DenseDataset` from a v3 dense ``X`` array.

    The array may be 2D ``[cells, genes]`` (standard, anndata-readable) or 1D
    ``[cells * genes]`` (scdata's flattened layout for cell-aligned chunking).
    For 1D the gene count comes from ``var`` and ``num_cells`` is derived.
    """
    arr = _parse_v3_array(store, x_key)
    meta = _v3_array_to_meta(arr, store)

    if len(arr.shape) == 2:
        num_cells, num_genes = arr.shape
    elif len(arr.shape) == 1:
        num_genes = len(gene_names)
        if num_genes == 0:
            raise StoreError("dense 1D array but var has no gene names")
        total = arr.shape[0]
        if total % num_genes != 0:
            raise StoreError(f"dense 1D length {total} not divisible by gene count {num_genes}")
        num_cells = total // num_genes
    else:
        raise StoreError(f"dense X must be 1D or 2D, got shape {arr.shape}")

    if num_genes != len(gene_names):
        raise StoreError(f"X has {num_genes} genes but var has {len(gene_names)} gene names")
    try:
        return DenseDataset(
            gene_names=gene_names,
            data=meta,
            num_cells=num_cells,
            num_genes=num_genes,
            store_root=store_root,
        )
    except ValueError as err:
        raise StoreError(f"{x_key}: {err}") from err


def _v3_read_gene_names(store: Store, var_key: str) -> tuple[str, ...]:
    """Read the var index as gene names from a v3 store.

    The var group is a v3 group; its ``_index`` child is a v3 string array
    (``data_type: "string"``).  String chunk files decode via the VLenUTF8
    codec embedded in the chunk itself, so we read+decode them here.
    """
    group_key = f"{var_key}/zarr.json"
    if not store.exists(group_key):
        raise StoreError(f"missing var group: {var_key}")
    group_meta = _expect_object(_read_json(store, group_key), group_key)
    if _v3_node_type(group_meta, group_key) != "group":
        raise StoreError(f"{var_key}: expected v3 group")

    attrs = group_meta.get("attributes")
    index_name = "_index"
    if isinstance(attrs, dict):
        declared = attrs.get("_index")
        if isinstance(declared, str) and declared:
            index_name = declared

    candidates = [index_name, "_index", "index"]
    seen: set[str] = set()
    index_key: str | None = None
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if store.exists(f"{var_key}/{candidate}/zarr.json"):
            index_key = f"{var_key}/{candidate}"
            break
    if index_key is None:
        raise StoreError(f"cannot find var index array under {var_key}")

    return _v3_read_string_array(store, index_key)


def _v3_read_string_array(store: Store, array_key: str) -> tuple[str, ...]:
    """Decode a v3 string array (``data_type: "string"``) to Python strings.

    v3 stores strings as ``data_type: "string"`` with a ``vlen-utf8`` serializer
    codec; each chunk file is an independently encoded VLenUTF8 frame (optionally
    compressed by a following BytesBytes codec).  We decode chunk-by-chunk.

    This bypasses :func:`_parse_v3_array` (which rejects string dtypes) and
    reads the chunk grid + codec pipeline directly.
    """
    key = f"{array_key}/zarr.json"
    meta = _expect_object(_read_json(store, key), key)
    if _v3_node_type(meta, key) != "array":
        raise StoreError(f"{key}: expected node_type 'array'")
    shape = _parse_shape(meta.get("shape"), key)
    if len(shape) != 1:
        raise StoreError(f"{array_key}: string array must be 1D, got shape {shape}")
    count = shape[0]
    chunk_grid = meta.get("chunk_grid")
    if not isinstance(chunk_grid, dict) or chunk_grid.get("name") != "regular":
        raise StoreError(f"{key}: string array must use a regular chunk grid")
    grid_cfg = chunk_grid.get("configuration")
    if not isinstance(grid_cfg, dict):
        raise StoreError(f"{key}: chunk_grid missing configuration")
    chunk_shape = _parse_chunk_shape(grid_cfg.get("chunk_shape"), key)
    codec = _v3_codec_pipeline(meta.get("codecs"), key)
    chunk_paths, chunk_lengths = _v3_chunk_files(
        store,
        array_key,
        shape,
        chunk_shape,
        meta.get("chunk_key_encoding"),
        key,
    )
    names: list[str] = []
    for path, length in zip(chunk_paths, chunk_lengths):
        if length == 0:
            continue
        raw = store.read_bytes(path)
        names.extend(_decode_v3_string_chunk(raw, codec))
    if len(names) < count:
        names.extend([""] * (count - len(names)))
    return tuple(names[:count])


def _decode_v3_string_chunk(raw: bytes, codec: CodecPipeline) -> list[str]:
    """Decode one v3 string-array chunk: compressor first, then VLenUTF8.

    The v3 ``vlen-utf8`` codec is the serializer (innermost); compressors in
    ``codec.filters`` / ``codec.compressor`` wrap it.  We reuse the existing
    numcodecs-backed decoders, then run VLenUTF8 on the decompressed bytes.
    """
    from numcodecs import VLenUTF8, get_codec

    data = raw
    try:
        if codec.compressor is not None:
            data = _as_bytes(get_codec(dict(codec.compressor)).decode(data))
        for flt in reversed(codec.filters):
            data = _as_bytes(get_codec(dict(flt)).decode(data))
    except Exception as err:
        raise StoreError(f"failed to decode v3 string chunk: {err}") from err
    arr = np.asarray(VLenUTF8().decode(_as_bytes(data)))
    return [str(s) for s in arr.tolist()]


def _v3_read_x_shape_attr(store: Store, x_key: str) -> tuple[int, int] | None:
    """Read anndata's sparse ``shape`` attr from a v3 ``X`` group's zarr.json."""
    key = f"{x_key}/zarr.json"
    if not store.exists(key):
        return None
    meta = _expect_object(_read_json(store, key), key)
    attrs = meta.get("attributes")
    if not isinstance(attrs, dict):
        return None
    raw = attrs.get("shape")
    if raw is None:
        return None
    shape = _parse_shape(raw, f"{key} attributes")
    if len(shape) != 2:
        raise StoreError(f"{x_key}: sparse matrix shape must be 2D, got {shape}")
    return shape[0], shape[1]


def _v3_build_sparse_dataset(
    store: Store, x_key: str, gene_names: tuple[str, ...], store_root: str
) -> SparseDataset:
    """Build a :class:`SparseDataset` from a v3 anndata CSR ``X`` group."""
    group_key = f"{x_key}/zarr.json"
    if not store.exists(group_key):
        raise StoreError(f"sparse X must be a v3 group: {x_key}")
    group_meta = _expect_object(_read_json(store, group_key), group_key)
    if _v3_node_type(group_meta, group_key) != "group":
        raise StoreError(f"{x_key}: expected v3 group for CSR matrix")
    x_shape = _v3_read_x_shape_attr(store, x_key)

    # indptr: length num_cells+1; decoded into a uint64 numpy array in memory.
    indptr_arr = _parse_v3_array(store, f"{x_key}/indptr")
    if len(indptr_arr.shape) != 1:
        raise StoreError(f"{x_key}/indptr must be 1D, got shape {indptr_arr.shape}")
    num_cells = indptr_arr.shape[0] - 1
    indptr = _v3_decode_index_array(store, indptr_arr, num_cells + 1)

    indices_arr = _parse_v3_array(store, f"{x_key}/indices")
    data_arr = _parse_v3_array(store, f"{x_key}/data")
    if len(indices_arr.shape) != 1 or len(data_arr.shape) != 1:
        raise StoreError(f"{x_key}/indices and data must be 1D")

    nnz = int(indptr[-1]) if len(indptr) else 0
    if indices_arr.shape[0] != nnz:
        raise StoreError(f"{x_key}/indices length {indices_arr.shape[0]} != nnz {nnz}")
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
                f"{x_key}: X shape has {shape_genes} genes but var has {len(gene_names)} gene names"
            )
        num_genes = shape_genes
    else:
        num_genes = len(gene_names)

    indices_meta = _v3_array_to_meta(indices_arr, store)
    data_meta = _v3_array_to_meta(data_arr, store)
    try:
        return SparseDataset(
            gene_names=gene_names,
            indptr=np.asarray(indptr, dtype=np.uint64),
            indices=indices_meta,
            data=data_meta,
            index_dtype=indices_arr.dtype,
            num_cells=num_cells,
            num_genes=num_genes,
            store_root=store_root,
        )
    except ValueError as err:
        raise StoreError(f"{x_key}: {err}") from err


def _v3_decode_index_array(store: Store, arr: _V3Array, count: int) -> list[int] | np.ndarray:
    """Decode a 1D integer v3 array (indptr) to uint64-compatible values."""
    if arr.dtype not in _INTEGER_DTYPES:
        raise StoreError(f"index array dtype {arr.dtype!r} must be an integer type")
    if _decode_index_chunks is not None:
        chunks = [
            store.read_bytes(path)
            for path, length in zip(arr.chunk_paths, arr.chunk_lengths, strict=True)
            if length != 0
        ]
        try:
            return _decode_index_chunks(chunks, arr.dtype, arr.codec, count)
        except Exception:
            pass
    np_dtype = np.dtype(_dtype_to_numpy(arr.dtype))
    item = np_dtype.itemsize
    out: list[int] = []
    for path, length in zip(arr.chunk_paths, arr.chunk_lengths):
        if length == 0:
            continue
        raw = store.read_bytes(path)
        dec = _decode_chunk_bytes(raw, arr.codec)
        if len(dec) % item != 0:
            raise StoreError(
                f"index chunk decoded to {len(dec)} bytes, not a multiple of itemsize {item}"
            )
        arr_vals = np.frombuffer(dec, dtype=np_dtype)
        out.extend(int(x) for x in arr_vals.tolist())
    if len(out) < count:
        out.extend([0] * (count - len(out)))
    return out[:count]


def _parse_shape(raw: object, context: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise StoreError(f"{context}: shape must be a non-empty list")
    if not all(_is_json_int(s) for s in raw):
        raise StoreError(f"{context}: shape entries must be JSON integers, got {raw!r}")
    shape = tuple(raw)
    if any(s < 0 for s in shape):
        raise StoreError(f"{context}: shape entries must be non-negative, got {shape}")
    return shape


def _parse_chunk_shape(raw: object, context: str) -> tuple[int, ...]:
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


def _as_bytes(value: Any) -> bytes:
    """Coerce a numcodecs decode result (bytes / buffer / array) to bytes.

    ``value`` is typed :data:`~typing.Any` because numcodecs ships no type
    stubs, so its ``decode`` return is ``Unknown`` to pyright; the runtime
    branches narrow it to the real bytes / buffer / ndarray shapes.
    """
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    # ndarray (or anything duck-typed with ``.tobytes`` from numcodecs).
    if hasattr(value, "tobytes"):
        return value.tobytes()
    return bytes(value)


def _decode_chunk_bytes(raw: bytes, codec: CodecPipeline) -> bytes:
    """Decode a single encoded chunk: compressor first, then reverse(filters).

    Each chunk is an independent codec frame — a concatenated byte stream is
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


_INTEGER_DTYPES = frozenset(
    {
        DType.U8,
        DType.I8,
        DType.U16,
        DType.I16,
        DType.U32,
        DType.I32,
        DType.U64,
        DType.I64,
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


def launch(
    path: str | os.PathLike[str],
    *,
    layer: str | None = None,
    matrix: str | None = None,
) -> Dataset:
    """Launch a scdata store and return its dataset metadata.

    The store must be a zarr v3 tree with standard per-chunk files.  Directory
    and ``ZIP_STORED`` containers are supported.  The returned
    :class:`DenseDataset` or :class:`SparseDataset` carries shape, dtype, codec
    metadata, chunk grid metadata, and normalized local file/off/len chunk
    locations, with ``store_root`` set to ``path`` so
    :class:`scdata.ScDataBank` can register it directly.

    This parses metadata only; numeric chunks are not decoded here.  By
    default the returned dataset describes ``X``.  Pass ``layer="counts"`` or
    ``matrix="layers/counts"`` to parse one AnnData layer instead.
    """
    store_root = os.fspath(path)
    matrix_key = _resolve_matrix_key(layer=layer, matrix=matrix)
    with _open_store(store_root) as store:
        return launch_store(store, store_root=store_root, matrix=matrix_key)


def launch_all(path: str | os.PathLike[str]) -> DatasetCollection:
    """Launch ``X`` and every direct child under ``layers`` from a store."""
    store_root = os.fspath(path)
    with _open_store(store_root) as store:
        return launch_store_all(store, store_root=store_root)


def launch_store(
    store: Store,
    *,
    store_root: str = "",
    layer: str | None = None,
    matrix: str | None = None,
) -> Dataset:
    """Launch an already-open :class:`Store` into a :class:`Dataset`.

    Use :func:`launch` for the common case; this entry point lets callers
    reuse a custom :class:`Store` implementation.  Pass ``store_root`` so the
    returned dataset records the store's filesystem path for the databank;
    when omitted it is left empty and the caller must pass the path to
    ``register_dense`` / ``register_sparse_csr`` explicitly.

    Only zarr v3 stores (``zarr.json`` per node, standard chunk files — the
    layout :func:`scdata.io.write_zarr` produces and anndata reads) are
    accepted.
    """
    matrix_key = _resolve_matrix_key(layer=layer, matrix=matrix)
    if store.exists("zarr.json"):
        return _launch_v3(store, store_root=store_root, matrix_key=matrix_key)
    raise StoreError("not a zarr v3 store (missing zarr.json)")


def launch_store_all(store: Store, *, store_root: str = "") -> DatasetCollection:
    """Launch ``X`` and all direct ``layers/*`` matrices from an open store."""
    if store.exists("zarr.json"):
        return _launch_v3_all(store, store_root=store_root)
    raise StoreError("not a zarr v3 store (missing zarr.json)")


def _resolve_matrix_key(*, layer: str | None, matrix: str | None) -> str:
    if layer is not None and matrix is not None:
        raise ValueError("pass either layer= or matrix=, not both")
    if layer is not None:
        return _layer_matrix_key(layer)
    if matrix is None:
        return "X"
    key = matrix.strip("/")
    if key == "X":
        return "X"
    if "/" not in key:
        return _layer_matrix_key(key)
    if key.startswith("layers/"):
        name = key[len("layers/") :]
        if name and "/" not in name:
            return key
    raise ValueError(f"unsupported matrix key {matrix!r}; expected 'X' or 'layers/<name>'")


def _layer_matrix_key(layer: str) -> str:
    name = str(layer)
    if not name or "/" in name:
        raise ValueError(f"layer names must be non-empty direct children, got {layer!r}")
    return f"layers/{name}"


def _launch_v3(store: Store, *, store_root: str, matrix_key: str = "X") -> Dataset:
    """Launch one matrix from a zarr v3 store."""
    gene_names = _v3_prepare_store(store)
    return _launch_v3_matrix(store, matrix_key, gene_names, store_root)


def _launch_v3_all(store: Store, *, store_root: str) -> DatasetCollection:
    """Launch a zarr v3 store (standard chunk files, anndata-compatible)."""
    gene_names = _v3_prepare_store(store)
    x = _launch_v3_matrix(store, "X", gene_names, store_root)
    layers = {
        name: _launch_v3_matrix(store, f"layers/{name}", gene_names, store_root)
        for name in _v3_layer_names(store)
    }
    return DatasetCollection(x=x, layers=layers, store_root=store_root)


def _v3_prepare_store(store: Store) -> tuple[str, ...]:
    root_meta = _expect_object(_read_json(store, "zarr.json"), "zarr.json")
    if _v3_node_type(root_meta, "zarr.json") != "group":
        raise StoreError("zarr.json root is not a group")
    if not store.exists("X/zarr.json"):
        raise StoreError("store has no X array or group")
    if not store.exists("var/zarr.json"):
        raise StoreError("store has no var group")
    return _v3_read_gene_names(store, "var")


def _launch_v3_matrix(
    store: Store,
    matrix_key: str,
    gene_names: tuple[str, ...],
    store_root: str,
) -> Dataset:
    meta_key = f"{matrix_key}/zarr.json"
    if not store.exists(meta_key):
        raise StoreError(f"store has no {matrix_key} array or group")

    matrix_meta = _expect_object(_read_json(store, meta_key), meta_key)
    node = _v3_node_type(matrix_meta, meta_key)
    attrs = matrix_meta.get("attributes") if isinstance(matrix_meta.get("attributes"), dict) else {}
    encoding_type = attrs.get("encoding-type") if isinstance(attrs, dict) else None
    if not isinstance(encoding_type, str):
        encoding_type = "array" if node == "array" else None
    if encoding_type in ("csc_matrix", "CSC"):
        raise StoreError("scdata does not read CSC matrices; store as CSR")

    if node == "array":
        return _v3_build_dense_dataset(store, matrix_key, gene_names, store_root)
    if node == "group" and encoding_type in ("csr_matrix", "CSR"):
        return _v3_build_sparse_dataset(store, matrix_key, gene_names, store_root)
    raise StoreError(
        f"unsupported {matrix_key} layout: node_type={node!r} encoding-type={encoding_type!r}"
    )


def _v3_layer_names(store: Store) -> tuple[str, ...]:
    if not store.exists("layers/zarr.json"):
        return ()
    names: set[str] = set()
    for key in store.list_keys("layers"):
        parts = key.split("/")
        if len(parts) == 3 and parts[0] == "layers" and parts[2] == "zarr.json" and parts[1]:
            names.add(parts[1])
    return tuple(sorted(names))
