"""Dataset metadata mirroring what the Rust databank needs to read a store.

These types are the Python-side view of a scdata store produced by
``scdata.io._launch``. They follow Python idioms (dataclasses, enums) rather
than mirroring the Rust structs field-for-field, but they carry every piece
of information the Rust :class:`DataBank` consumes from a store:

* logical ``shape`` / ``chunk_shape`` plus optional rectilinear boundaries,
* the element dtype (mapped to the Rust ``DType`` enum),
* the codec pipeline as raw numcodecs-compatible JSON,
* normalized chunk sources: a local file path plus ``(offset, length)`` for
  each encoded chunk, whether the original store was a directory, zip archive,
  or legacy concatenated payload.

The dataset layer does not open files or decode numeric chunks.  It only keeps
metadata and chunk addresses so the Rust databank can construct arrays and
datasets without knowing which upstream container produced them.

FFI note: the per-chunk index and the CSR ``indptr`` are stored as contiguous
``uint64`` numpy arrays so the Rust binding can borrow them zero-copy via
``PyReadonlyArray1::<u64>`` instead of walking a Python tuple element by
element.  Tuple views (``ArrayMeta.chunks`` / ``SparseDataset.indptr`` as a
tuple) are still available as derived properties for readability and for any
caller that has not switched to the numpy path.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ArrayOrder",
    "DType",
    "CodecPipeline",
    "ChunkLocation",
    "ArrayMeta",
    "DenseDataset",
    "SparseDataset",
    "Dataset",
    "DataError",
    "DtypeParseError",
    "CodecConfigError",
]


class DataError(ValueError):
    """Base for metadata parsing errors raised by :mod:`scdata.data`.

    The io layer catches :class:`DataError` and re-raises it as
    :class:`~scdata.io.StoreError` so :func:`~scdata.io.launch` exposes a single
    error type to callers.
    """


class DtypeParseError(DataError):
    """Raised when a zarr dtype/order string cannot be mapped to a scdata type."""


class CodecConfigError(DataError):
    """Raised when a numcodecs filter/compressor config is malformed."""


class ArrayOrder(str, Enum):
    """Memory layout of a zarr array, mirrors Rust ``ArrayOrder``.

    Rust only supports C-order arrays; scdata stores are always C-order, so
    F-order is rejected at parse time rather than silently accepted and later
    failing in the Rust reader.
    """

    C = "C"

    @classmethod
    def parse(cls, value: object) -> "ArrayOrder":
        if isinstance(value, ArrayOrder):
            return value
        if value is None:
            # zarr v2 defaults to C order when ``order`` is absent.
            return cls.C
        text = str(value).strip().upper()
        if text == "C":
            return cls.C
        if text == "F":
            raise DtypeParseError("F-order arrays are unsupported (scdata stores are C-order)")
        raise DtypeParseError(f"unsupported array order: {value!r}")


class DType(Enum):
    """Element dtype, mirrors the Rust ``DType`` enum (``databank/array.rs``).

    Rust supports a fixed set of numeric dtypes.  zarr v2 dtype strings are
    endianness-prefixed (``<f4``, ``|u8``, ``>i4``); scdata stores are always
    little-endian on disk, so ``<``, ``=`` and ``|`` are accepted.  ``>`` is
    rejected instead of silently byte-swapping metadata the Rust reader cannot
    decode as written.
    """

    U8 = "u8"
    I8 = "i8"
    U16 = "u16"
    I16 = "i16"
    U32 = "u32"
    I32 = "i32"
    U64 = "u64"
    I64 = "i64"
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"
    F64 = "f64"

    @property
    def item_size(self) -> int:
        match self:
            case DType.U8 | DType.I8:
                return 1
            case DType.U16 | DType.I16 | DType.F16 | DType.BF16:
                return 2
            case DType.U32 | DType.I32 | DType.F32:
                return 4
            case DType.U64 | DType.I64 | DType.F64:
                return 8

    @property
    def is_csr_index(self) -> bool:
        """Whether this dtype is valid for CSR ``indices`` (Rust ``is_csr_index``)."""
        return self in (DType.I32, DType.U32, DType.I64, DType.U64)

    @classmethod
    def parse(cls, dtype: object) -> "DType":
        """Parse a zarr v2 dtype field into a :class:`DType`.

        Accepts the standard zarr forms:

        * a base type string, e.g. ``"<f4"``, ``"|u8"``, ``">i4"``,
        * a structured record ``[base, [("f0", "<i4")]]`` — the base element
          type is used (scdata stores scalars, not structured records, but
          zarr wraps single-field arrays this way),
        * a :class:`DType` (passthrough).
        """
        if isinstance(dtype, DType):
            return dtype
        if dtype is None:
            raise DtypeParseError("dtype is None")

        base = _extract_base_dtype(dtype)
        return _decode_base_dtype(base)

    @classmethod
    def from_numpy(cls, dtype: Any) -> "DType":
        """Map a numpy dtype object to a :class:`DType`."""
        np_dtype = np.dtype(dtype)
        kind = np_dtype.kind
        size = np_dtype.itemsize
        match (kind, size):
            case ("u", 1):
                return cls.U8
            case ("i", 1):
                return cls.I8
            case ("u", 2):
                return cls.U16
            case ("i", 2):
                return cls.I16
            case ("u", 4):
                return cls.U32
            case ("i", 4):
                return cls.I32
            case ("u", 8):
                return cls.U64
            case ("i", 8):
                return cls.I64
            case ("f", 2):
                # numpy has no native bf16; a 2-byte float is f16 here.
                return cls.F16
            case ("f", 4):
                return cls.F32
            case ("f", 8):
                return cls.F64
            case _:
                raise DtypeParseError(f"unsupported numpy dtype: {np_dtype}")


def _extract_base_dtype(dtype: object) -> str:
    """Return the base type string from a zarr dtype field."""
    if isinstance(dtype, str):
        return dtype
    # zarr represents structured arrays as [base_str, [(field, dtype), ...]].
    if isinstance(dtype, list) and dtype:
        first = dtype[0]
        if isinstance(first, str):
            return first
    # Some libraries emit a dict with shape/descriptor; fall back to str().
    text = str(dtype).strip()
    if not text:
        raise DtypeParseError(f"empty dtype descriptor: {dtype!r}")
    return text


_BASE_DTYPE_MAP: dict[str, DType] = {
    "u1": DType.U8,
    "i1": DType.I8,
    "u2": DType.U16,
    "i2": DType.I16,
    "u4": DType.U32,
    "i4": DType.I32,
    "u8": DType.U64,
    "i8": DType.I64,
    "f2": DType.F16,
    "f4": DType.F32,
    "f8": DType.F64,
    # numcodecs/bfloat extensions used by some single-cell stores.
    "bf2": DType.BF16,
}


def _decode_base_dtype(base: str) -> DType:
    text = base.strip()
    if not text:
        raise DtypeParseError("empty base dtype string")

    # Strip endianness prefix.  scdata stores are little-endian on disk; we
    # accept '<' (little), '=' (native on our write targets), and '|' (not
    # applicable / byte).  Big-endian input is rejected because Rust decodes
    # bytes exactly as written.
    if text[0] in "<>=":
        prefix = text[0]
        body = text[1:]
        if prefix == ">":
            # Big-endian source is not produced by scdata; refuse rather than
            # silently byte-swap.
            raise DtypeParseError(
                f"big-endian dtype {base!r} is unsupported (scdata stores are little-endian)"
            )
    elif text[0] == "|":
        body = text[1:]
    else:
        body = text

    body = body.strip().lower()
    if not body:
        raise DtypeParseError(f"missing type code in {base!r}")

    if body not in _BASE_DTYPE_MAP:
        raise DtypeParseError(f"unsupported dtype {base!r} (body {body!r})")
    return _BASE_DTYPE_MAP[body]


@dataclass(frozen=True)
class CodecPipeline:
    """Raw numcodecs filter/compressor JSON for one array.

    Rust rebuilds the codec via ``codec_pipeline_from_zarr_v2_json_str`` from
    exactly these two fields, so we keep them verbatim — no re-interpretation
    on the Python side.  ``filters`` is the numcodecs filter list (applied in
    reverse on decode); ``compressor`` is the final compressor object.
    """

    filters: tuple[dict[str, Any], ...] = ()
    compressor: dict[str, Any] | None = None

    @classmethod
    def from_zarr(
        cls,
        filters: object | None,
        compressor: object | None,
    ) -> "CodecPipeline":
        """Build from the ``filters`` / ``compressor`` fields of a ``.zarray``."""
        if filters is None:
            filter_list: tuple[dict[str, Any], ...] = ()
        elif isinstance(filters, list):
            filter_list = tuple(_coerce_config_dict(f) for f in filters)
        else:
            raise CodecConfigError(f"filters must be a list, got {type(filters).__name__}")
        compressor_dict = _coerce_config_dict(compressor) if compressor is not None else None
        return cls(filters=filter_list, compressor=compressor_dict)

    @property
    def is_uncompressed(self) -> bool:
        return not self.filters and self.compressor is None

    def to_zarr(self) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
        """Return the ``(filters, compressor)`` pair as it appears in ``.zarray``."""
        filters = [dict(f) for f in self.filters] if self.filters else None
        compressor = dict(self.compressor) if self.compressor is not None else None
        return filters, compressor


def _coerce_config_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    raise CodecConfigError(f"codec config must be a JSON object, got {type(value).__name__}")


@dataclass(frozen=True)
class ChunkLocation:
    """Location of one encoded chunk inside the payload file.

    Mirrors Rust ``FileChunkLocation { offset: u64, len: usize }``.  Chunks
    are ordered by C-order (row-major) logical chunk index; the Rust array
    layer derives decoded size from the normalized grid/chunk metadata.
    """

    offset: int
    length: int

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError(f"chunk offset must be non-negative, got {self.offset}")
        if self.length < 0:
            raise ValueError(f"chunk length must be non-negative, got {self.length}")


def _as_u64_array(value: object, name: str) -> NDArray[np.uint64]:
    """Coerce a chunk index / indptr input into a contiguous 1D ``uint64`` array.

    Accepts a numpy array (any integer dtype) or any iterable of ints.  The
    result is always C-contiguous ``uint64`` so the Rust binding can borrow it
    zero-copy.  ``uint64`` matches Rust ``FileChunkLocation::offset`` /
    ``indptr: Vec<u64>``; values are range-checked because a negative Python
    int silently reinterprets as a huge unsigned value otherwise.
    """
    arr = np.asarray(value, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.ndim}D")
    if arr.size and arr.min() < 0:
        raise ValueError(f"{name} values must be non-negative")
    arr = arr.astype(np.uint64, copy=False)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


@dataclass(frozen=True)
class ArrayMeta:
    """Metadata for a single zarr array and how its chunks are stored.

    Two chunk stores are supported, selected by :attr:`store_kind`:

    * ``"file"`` — every encoded chunk is concatenated into one payload file
      (the legacy scdata v2 layout).  :attr:`payload_path` is the logical zarr
      key; :attr:`payload_file_path`, when set by :func:`scdata.io.launch`, is
      the actual local file Rust should open.  :attr:`chunk_offsets` /
      :attr:`chunk_lengths` give each encoded chunk's byte range.
    * ``"dir"`` — each chunk has its own logical zarr key.  Directory stores
      use one filesystem file per chunk; zip stores use the zip archive file
      plus a physical byte offset for each entry.  :attr:`chunk_paths` keeps the
      logical keys, while :attr:`chunk_file_paths` optionally carries the local
      file path Rust should open for each chunk.

    Regular v2 payload chunks are cropped at edge chunks; standard zarr v3
    chunks are padded.  Python only describes locations and grid metadata here;
    Rust derives the decoded size per chunk from the array grid.
    """

    shape: tuple[int, ...]
    chunk_shape: tuple[int, ...]
    dtype: DType
    order: ArrayOrder = ArrayOrder.C
    codec: CodecPipeline = field(default_factory=CodecPipeline)
    payload_path: str = ""
    #: Actual local file for ``payload_path``.  Empty means the Rust binding
    #: joins ``store_path / payload_path`` for backwards-compatible manual
    #: datasets.
    payload_file_path: str = ""
    store_kind: Literal["file", "dir"] = "file"
    #: Whether chunks have variable (per-chunk) decoded sizes (zarr v3
    #: rectilinear chunk grids).  False for regular grids; true for scdata's
    #: cell-aligned CSR layout.
    variable_chunks: bool = False
    #: Explicit rectilinear chunk boundaries, one tuple per axis.  Regular
    #: grids leave this empty.
    chunk_boundaries: tuple[tuple[int, ...], ...] = ()
    #: One store-root-relative path per chunk (``store_kind="dir"`` only).
    chunk_paths: tuple[str, ...] = ()
    #: Actual local file per chunk.  Empty means the Rust binding joins
    #: ``store_path / chunk_paths[i]``; zip stores set every entry to the archive
    #: path and use :attr:`chunk_offsets` for the in-archive byte offset.
    chunk_file_paths: tuple[str, ...] = ()
    # ``compare=False``: numpy arrays are not hashable and elementwise equality
    # does not yield a bool, so they are excluded from the dataclass-generated
    # ``__eq__`` / ``__hash__``.  Equality is still well-defined via the other
    # fields, and these objects are never used as dict keys in practice.
    chunk_offsets: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.uint64),
        compare=False,
        repr=False,
    )
    chunk_lengths: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.uint64),
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if len(self.shape) != len(self.chunk_shape):
            raise ValueError(
                f"shape rank {len(self.shape)} != chunk_shape rank {len(self.chunk_shape)}"
            )
        if len(self.shape) == 0:
            raise ValueError("array shape must be non-empty")
        if any(s <= 0 for s in self.shape):
            raise ValueError(f"shape must be positive, got {self.shape}")
        if any(c <= 0 for c in self.chunk_shape):
            raise ValueError(f"chunk_shape must be positive, got {self.chunk_shape}")
        if self.store_kind not in ("file", "dir"):
            raise ValueError(f"store_kind must be 'file' or 'dir', got {self.store_kind!r}")
        boundaries = tuple(tuple(int(x) for x in axis) for axis in self.chunk_boundaries)
        if boundaries:
            if not self.variable_chunks:
                raise ValueError("chunk_boundaries require variable_chunks=True")
            if len(boundaries) != len(self.shape):
                raise ValueError(
                    f"chunk_boundaries rank {len(boundaries)} != shape rank {len(self.shape)}"
                )
            for axis, (bounds, dim) in enumerate(zip(boundaries, self.shape)):
                if len(bounds) < 2:
                    raise ValueError(f"chunk_boundaries[{axis}] must contain at least two entries")
                if bounds[0] != 0:
                    raise ValueError(f"chunk_boundaries[{axis}] must start at 0")
                if bounds[-1] != dim:
                    raise ValueError(
                        f"chunk_boundaries[{axis}] final boundary {bounds[-1]} != shape {dim}"
                    )
                if any(a > b for a, b in zip(bounds, bounds[1:])):
                    raise ValueError(f"chunk_boundaries[{axis}] must be monotonic")
            object.__setattr__(self, "chunk_boundaries", boundaries)

        lengths = _as_u64_array(self.chunk_lengths, "chunk_lengths")
        object.__setattr__(self, "chunk_lengths", lengths)

        if self.store_kind == "file":
            offsets = _as_u64_array(self.chunk_offsets, "chunk_offsets")
            if offsets.shape[0] != lengths.shape[0]:
                raise ValueError(
                    f"chunk_offsets length {offsets.shape[0]} != "
                    f"chunk_lengths length {lengths.shape[0]}"
                )
            if offsets.shape[0] != self.num_chunks:
                raise ValueError(
                    f"chunks count {offsets.shape[0]} != chunk grid size {self.num_chunks}"
                )
            # scdata writes chunks concatenated in C-order, so offsets are
            # strictly non-decreasing.  This catches a mis-ordered index before
            # it reaches Rust, where an out-of-order read would silently fetch
            # the wrong bytes.  ``diff`` is done on a signed view: on ``uint64``
            # a decreasing pair would wrap to a huge positive value and silently
            # pass the check.
            if offsets.shape[0] >= 2 and np.any(np.diff(offsets.view(np.int64)) < 0):
                raise ValueError("chunk offsets must be monotonically non-decreasing")
            object.__setattr__(self, "chunk_offsets", offsets)
            object.__setattr__(self, "chunk_paths", ())
            object.__setattr__(self, "chunk_file_paths", ())
        else:  # "dir"
            paths = tuple(self.chunk_paths)
            if len(paths) != self.num_chunks:
                raise ValueError(
                    f"chunk_paths count {len(paths)} != chunk grid size {self.num_chunks}"
                )
            if lengths.shape[0] != self.num_chunks:
                raise ValueError(
                    f"chunk_lengths count {lengths.shape[0]} != chunk grid size {self.num_chunks}"
                )
            # Per-chunk byte offset within each chunk's file.  Always 0 for a
            # directory store (one file per chunk); for a ``.zarr.zip`` store
            # this is the chunk's physical offset inside the zip archive, so
            # the Rust reader preads the chunk directly out of the zip file.
            offsets = _as_u64_array(self.chunk_offsets, "chunk_offsets")
            if offsets.shape[0] != self.num_chunks:
                offsets = np.zeros(self.num_chunks, dtype=np.uint64)
            file_paths = tuple(self.chunk_file_paths)
            if file_paths and len(file_paths) != self.num_chunks:
                raise ValueError(
                    f"chunk_file_paths count {len(file_paths)} != chunk grid size {self.num_chunks}"
                )
            object.__setattr__(self, "chunk_paths", paths)
            object.__setattr__(self, "chunk_file_paths", file_paths)
            object.__setattr__(self, "chunk_offsets", offsets)

    @classmethod
    def from_chunks(
        cls,
        *,
        shape: tuple[int, ...],
        chunk_shape: tuple[int, ...],
        dtype: DType,
        chunks: Iterable[ChunkLocation],
        order: ArrayOrder = ArrayOrder.C,
        codec: CodecPipeline | None = None,
        payload_path: str = "",
        payload_file_path: str = "",
        chunk_offset_base: int = 0,
    ) -> "ArrayMeta":
        """Build a ``store_kind="file"`` :class:`ArrayMeta` from chunk locations.

        Convenience for callers that already hold ``(offset, length)`` pairs —
        the io layer and tests historically pass ``chunks=`` this way.  The
        pairs are copied into two contiguous ``uint64`` arrays once; afterwards
        the tuple form is available via the :attr:`chunks` property.
        """
        chunk_list = tuple(chunks)
        count = len(chunk_list)
        offsets = np.empty(count, dtype=np.uint64)
        lengths = np.empty(count, dtype=np.uint64)
        for i, loc in enumerate(chunk_list):
            offsets[i] = loc.offset + chunk_offset_base
            lengths[i] = loc.length
        return cls(
            shape=shape,
            chunk_shape=chunk_shape,
            dtype=dtype,
            order=order,
            codec=codec if codec is not None else CodecPipeline(),
            payload_path=payload_path,
            payload_file_path=payload_file_path,
            store_kind="file",
            chunk_offsets=offsets,
            chunk_lengths=lengths,
        )

    @classmethod
    def from_directory(
        cls,
        *,
        shape: tuple[int, ...],
        chunk_shape: tuple[int, ...],
        dtype: DType,
        chunk_paths: Iterable[str],
        chunk_lengths: Iterable[int],
        order: ArrayOrder = ArrayOrder.C,
        codec: CodecPipeline | None = None,
        variable_chunks: bool = False,
        chunk_boundaries: Iterable[Iterable[int]] | None = None,
        chunk_offsets: Iterable[int] | None = None,
        chunk_file_paths: Iterable[str] | None = None,
    ) -> "ArrayMeta":
        """Build a ``store_kind="dir"`` :class:`ArrayMeta` from per-chunk files.

        ``chunk_paths`` / ``chunk_lengths`` are one entry per chunk, ordered by
        C-order logical chunk index (the order zarr stores chunk files for a
        regular chunk grid).  Each path is relative to the store root; the
        databank joins it with the dataset's :attr:`store_root` at register
        time.

        ``chunk_offsets`` is the byte offset of each chunk within its file:
        always 0 for a directory store (one file per chunk), or the chunk's
        physical offset inside a ``.zarr.zip`` archive so the Rust reader
        preads the chunk directly out of the zip.  Defaults to all-zeros.

        ``chunk_file_paths`` is optional and normally filled by
        :func:`scdata.io.launch`.  When omitted, the Rust binding joins
        ``store_path`` with ``chunk_paths``.  Zip stores set it to the zip file
        path for every chunk.

        ``variable_chunks=True`` marks a zarr v3 rectilinear chunk grid
        (scdata's cell-aligned CSR layout).  Pass ``chunk_boundaries`` so Rust
        can map coordinates to chunks and compute each decoded chunk size.
        """
        return cls(
            shape=shape,
            chunk_shape=chunk_shape,
            dtype=dtype,
            order=order,
            codec=codec if codec is not None else CodecPipeline(),
            payload_path="",
            store_kind="dir",
            variable_chunks=variable_chunks,
            chunk_boundaries=tuple(tuple(axis) for axis in chunk_boundaries)
            if chunk_boundaries is not None
            else (),
            chunk_paths=tuple(chunk_paths),
            chunk_file_paths=tuple(chunk_file_paths) if chunk_file_paths is not None else (),
            chunk_lengths=np.asarray(tuple(chunk_lengths), dtype=np.uint64),
            chunk_offsets=np.asarray(tuple(chunk_offsets), dtype=np.uint64)
            if chunk_offsets is not None
            else np.empty(0, dtype=np.uint64),
        )

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def chunk_grid_shape(self) -> tuple[int, ...]:
        """Number of chunks along each axis: ``ceil(shape / chunk_shape)``."""
        if self.chunk_boundaries:
            return tuple(len(axis) - 1 for axis in self.chunk_boundaries)
        return tuple(_ceil_div(s, c) for s, c in zip(self.shape, self.chunk_shape))

    @property
    def num_chunks(self) -> int:
        n = 1
        for g in self.chunk_grid_shape:
            n *= g
        return n

    @property
    def item_size(self) -> int:
        return self.dtype.item_size

    @property
    def chunks(self) -> tuple[ChunkLocation, ...]:
        """Chunk locations as a tuple.

        For ``store_kind="file"`` offsets are relative to the file Rust opens
        unless :attr:`payload_file_path` points at a zip archive, in which case
        they are physical archive offsets.  For ``store_kind="dir"`` offsets
        are 0 for directory stores and physical archive offsets for zip stores.
        """
        if self.chunk_lengths.shape[0] == 0:
            return ()
        offsets = self.chunk_offsets.tolist()
        lengths = self.chunk_lengths.tolist()
        return tuple(ChunkLocation(offset=o, length=length) for o, length in zip(offsets, lengths))


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _validate_unique_gene_names(gene_names: tuple[str, ...]) -> None:
    """Reject duplicate *non-empty* gene names.

    ``access_cells_by_gene_names`` resolves requested names to column indices
    via the gene table; a non-empty duplicate would make the result ambiguous.
    The empty string is exempt: an empty name marks an anonymous (unmapped)
    column, of which there may legitimately be several after
    :meth:`DenseDataset.align_genes`.  Catching duplicates at construction is
    cheaper and clearer than a Rust-side error mid-read.
    """
    seen: set[str] = set()
    for name in gene_names:
        if not name:
            continue
        if name in seen:
            raise ValueError(f"gene_names must be unique, duplicate: {name!r}")
        seen.add(name)


def _align_gene_names(
    gene_names: tuple[str, ...],
    mapping: Mapping[str, str],
    default: str = "",
    *,
    keep: Literal["never", "first", "last"] = "never",
) -> tuple[str, ...]:
    """Standardize gene names through a symbol→canonical mapping.

    Each name is replaced by ``mapping[name]`` when present, otherwise by
    ``default`` (empty string by default).  The result has the same length and
    column order as the input — only the *names* change, never the data layout.

    ``keep`` controls what happens when two source columns map to the same
    *non-empty* canonical name (e.g. ``GAPDH`` and ``gapdh`` both mapping to
    ``GAPDH``), matching mainstream dedup conventions:

    * ``"never"`` (default) — raise :class:`ValueError`.  The downstream
      databank caches gene names and resolves ``access_cells_by_gene_names``
      requests against them; a non-empty duplicate would make a column
      ambiguous, so duplicates are rejected unless the caller explicitly
      chooses a resolution.
    * ``"first"`` — keep the canonical name on its first occurrence; later
      duplicates are treated as *unmapped* and fall back to ``default``.
    * ``"last"`` — keep the canonical name on its last occurrence; earlier
      duplicates fall back to ``default``.

    The ``default`` value (typically ``""``) is always exempt and may appear
    many times — unmapped columns are intentionally anonymous, not addressable.
    """
    aligned: list[str] = [mapping.get(name, default) for name in gene_names]

    # Indices of each non-empty canonical name, in encounter order.
    positions: dict[str, list[int]] = {}
    for i, canonical in enumerate(aligned):
        if canonical == default:
            continue
        positions.setdefault(canonical, []).append(i)

    for canonical, idxs in positions.items():
        if len(idxs) <= 1:
            continue
        if keep == "never":
            src = gene_names[idxs[0]]
            raise ValueError(
                f"align_genes maps {src!r} onto {canonical!r}, "
                f"already produced by another column (keep='never')"
            )
        if keep == "first":
            keep_idx = idxs[0]
        else:  # "last"
            keep_idx = idxs[-1]
        for i in idxs:
            if i != keep_idx:
                aligned[i] = default

    return tuple(aligned)


@dataclass(frozen=True)
class DenseDataset:
    """A dense single-cell matrix ``[cells, genes]``.

    The backing array may be stored as 2D ``[cells, genes]`` (Rust
    ``Dense2D``) or flattened 1D ``[cells * genes]`` (Rust ``Dense1D``);
    scdata normalizes to ``num_cells * num_genes == data.shape.product()``.
    """

    gene_names: tuple[str, ...]
    data: ArrayMeta
    num_cells: int
    num_genes: int
    #: Filesystem path of the store root holding ``data.payload_path``.
    #:
    #: Set by :func:`scdata.io.launch` so a dataset carries everything the
    #: Rust databank needs to open the store — the bank's ``register`` reads
    #: it from here rather than taking a second path argument.  Empty when the
    #: dataset was assembled by hand (tests); callers then pass ``store_path``
    #: explicitly to ``register_dense`` / ``register_sparse_csr``.
    store_root: str = ""

    def __post_init__(self) -> None:
        if self.num_cells <= 0:
            raise ValueError(f"dense dataset requires positive num_cells, got {self.num_cells}")
        if self.num_genes <= 0:
            raise ValueError(f"dense dataset requires positive num_genes, got {self.num_genes}")
        if len(self.gene_names) != self.num_genes:
            raise ValueError(
                f"gene_names count {len(self.gene_names)} != num_genes {self.num_genes}"
            )
        _validate_unique_gene_names(self.gene_names)
        elements = 1
        for s in self.data.shape:
            elements *= s
        if elements != self.num_cells * self.num_genes:
            raise ValueError(
                f"data has {elements} elements, expected "
                f"num_cells*num_genes = {self.num_cells * self.num_genes}"
            )

    @property
    def kind(self) -> str:
        """Canonical dataset kind: ``"dense"``."""
        return "dense"

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def num_chunks(self) -> int:
        return self.data.num_chunks

    @property
    def item_size(self) -> int:
        return self.data.item_size

    def align_genes(
        self,
        mapping: Mapping[str, str],
        default: str = "",
        *,
        keep: Literal["never", "first", "last"] = "never",
    ) -> "DenseDataset":
        """Return a copy with gene names standardized through ``mapping``.

        See :func:`_align_gene_names` for the rules: each name is replaced by
        ``mapping[name]`` (or ``default`` when absent), the column count and
        order are unchanged, and non-empty canonical-name collisions are
        resolved per ``keep`` (``"never"`` raises, ``"first"`` / ``"last"``
        demote the loser to ``default``).  Standardizing names here lets the
        databank cache one canonical gene table and resolve later
        ``access_cells_by_gene_names`` requests against a single naming scheme
        across datasets.
        """
        return replace(
            self,
            gene_names=_align_gene_names(self.gene_names, mapping, default, keep=keep),
        )


@dataclass(frozen=True)
class SparseDataset:
    """A CSR sparse single-cell matrix.

    Mirrors Rust ``SparseCsrDataset``: ``indptr`` is the length-``num_cells+1``
    CSR offset array (kept in memory, not chunked), stored as a contiguous
    ``uint64`` numpy array for zero-copy borrowing by the Rust binding;
    ``indices`` and ``data`` are 1D zarr arrays of length ``nnz``, chunked
    along the nnz axis.  Chunk alignment at write time keeps each cell's
    ``nnz`` run within a single chunk pair so a cell never spans two
    index/data chunk pairs.
    """

    gene_names: tuple[str, ...]
    # ``__post_init__`` coerces this to a contiguous ``uint64`` ndarray via
    # ``_as_u64_array`` (which accepts any iterable of ints), so a plain tuple
    # is accepted at construction — the declared union reflects that.
    indptr: np.ndarray | Sequence[int]
    indices: ArrayMeta
    data: ArrayMeta
    index_dtype: DType
    num_cells: int
    num_genes: int
    #: Filesystem path of the store root holding the ``indices`` / ``data``
    #: payload files.  See :attr:`DenseDataset.store_root`.
    store_root: str = ""

    def __post_init__(self) -> None:
        if self.num_cells <= 0:
            raise ValueError(f"sparse dataset requires positive num_cells, got {self.num_cells}")
        if self.num_genes <= 0:
            raise ValueError(f"sparse dataset requires positive num_genes, got {self.num_genes}")
        if len(self.gene_names) != self.num_genes:
            raise ValueError(
                f"gene_names count {len(self.gene_names)} != num_genes {self.num_genes}"
            )
        _validate_unique_gene_names(self.gene_names)

        indptr = _as_u64_array(self.indptr, "indptr")
        if indptr.shape[0] != self.num_cells + 1:
            raise ValueError(f"indptr length {indptr.shape[0]} != num_cells+1 {self.num_cells + 1}")
        if indptr.shape[0] > 0 and int(indptr[0]) != 0:
            raise ValueError(f"indptr must start at 0, got {int(indptr[0])}")
        if indptr.shape[0] >= 2 and np.any(np.diff(indptr.view(np.int64)) < 0):
            raise ValueError("indptr must be monotonically non-decreasing")
        object.__setattr__(self, "indptr", indptr)

        if not self.index_dtype.is_csr_index:
            raise ValueError(f"index_dtype {self.index_dtype!r} is not a valid CSR index dtype")
        if self.indices.ndim != 1 or self.data.ndim != 1:
            raise ValueError("sparse indices and data arrays must be 1D")

        nnz = int(indptr[-1]) if indptr.shape[0] > 0 else 0
        indices_len = 1
        for s in self.indices.shape:
            indices_len *= s
        data_len = 1
        for s in self.data.shape:
            data_len *= s
        if indices_len != nnz:
            raise ValueError(f"indices length {indices_len} != nnz {nnz}")
        if data_len != nnz:
            raise ValueError(f"data length {data_len} != nnz {nnz}")
        if self.indices.dtype != self.index_dtype:
            raise ValueError(
                f"indices dtype {self.indices.dtype!r} != index_dtype {self.index_dtype!r}"
            )

    @property
    def kind(self) -> str:
        """Canonical dataset kind: ``"sparse-csr"``."""
        return "sparse-csr"

    @property
    def nnz(self) -> int:
        """Number of stored nonzeros: ``indptr[-1]``."""
        return int(self.indptr[-1]) if len(self.indptr) > 0 else 0

    @property
    def ndim(self) -> int:
        """A CSR matrix is conceptually 2D ``[cells, genes]``."""
        return 2

    @property
    def num_chunks(self) -> int:
        """Chunk count of the ``indices`` array (``data`` is chunked identically)."""
        return self.indices.num_chunks

    @property
    def item_size(self) -> int:
        """Element size of the stored ``data`` values."""
        return self.data.item_size

    def align_genes(
        self,
        mapping: Mapping[str, str],
        default: str = "",
        *,
        keep: Literal["never", "first", "last"] = "never",
    ) -> "SparseDataset":
        """Return a copy with gene names standardized through ``mapping``.

        See :func:`_align_gene_names` for the rules: each name is replaced by
        ``mapping[name]`` (or ``default`` when absent), the column count and
        order are unchanged, and non-empty canonical-name collisions are
        resolved per ``keep`` (``"never"`` raises, ``"first"`` / ``"last"``
        demote the loser to ``default``).  Standardizing names here lets the
        databank cache one canonical gene table and resolve later
        ``access_cells_by_gene_names`` requests against a single naming scheme
        across datasets.
        """
        return replace(
            self,
            gene_names=_align_gene_names(self.gene_names, mapping, default, keep=keep),
        )


Dataset = DenseDataset | SparseDataset
"""Union type for a parsed dataset, returned by :mod:`scdata.io._launch`."""
