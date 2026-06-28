"""Dataset metadata mirroring what the Rust databank needs to read a store.

These types are the Python-side view of a scdata store produced by
``scdata.io._launch``. They follow Python idioms (dataclasses, enums) rather
than mirroring the Rust structs field-for-field, but they carry every piece
of information the Rust :class:`DataBank` consumes from a store:

* logical ``shape`` and ``chunk_shape`` (chunk alignment is enforced at write
  time so a single cell never spans two chunks),
* the element dtype (mapped to the Rust ``DType`` enum),
* the codec pipeline as raw numcodecs JSON (``filters`` + ``compressor``),
  which Rust rebuilds via ``codec_pipeline_from_zarr_v2_json_str``,
* the chunk store as a single concatenated payload file plus a per-chunk
  ``(offset, length)`` index table (Rust ``ChunkStoreMeta::FileOffset``).

The store carries standard zarr v2 metadata (``.zarray`` / ``.zgroup`` /
``.zattrs``), but chunk *data* lives in a single concatenated payload file
rather than one file per chunk — scdata reads it directly via the Rust
io_uring path.  Whether anndata can read chunk data back depends on the write
path also emitting standard per-chunk files; that is a write-path decision,
not yet implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

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
    def parse(cls, value: Any) -> "ArrayOrder":
        if isinstance(value, ArrayOrder):
            return value
        if value is None:
            # zarr v2 defaults to C order when ``order`` is absent.
            return cls.C
        text = str(value).strip().upper()
        if text == "C":
            return cls.C
        if text == "F":
            raise DtypeParseError(
                "F-order arrays are unsupported (scdata stores are C-order)"
            )
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
    def parse(cls, dtype: Any) -> "DType":
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
        import numpy as np

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


def _extract_base_dtype(dtype: Any) -> str:
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
        filters: Any | None,
        compressor: Any | None,
    ) -> "CodecPipeline":
        """Build from the ``filters`` / ``compressor`` fields of a ``.zarray``."""
        if filters is None:
            filter_list: tuple[dict[str, Any], ...] = ()
        elif isinstance(filters, list):
            filter_list = tuple(_coerce_config_dict(f) for f in filters)
        else:
            raise CodecConfigError(
                f"filters must be a list, got {type(filters).__name__}"
            )
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


def _coerce_config_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    raise CodecConfigError(
        f"codec config must be a JSON object, got {type(value).__name__}"
    )


@dataclass(frozen=True)
class ChunkLocation:
    """Location of one encoded chunk inside the payload file.

    Mirrors Rust ``FileChunkLocation { offset: u64, len: usize }``.  Chunks
    are ordered by C-order (row-major) logical chunk index; edge chunks are
    stored cropped to their logical extent (no padding), matching Rust's
    ``linear_chunk_expected_size`` semantics.
    """

    offset: int
    length: int

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError(f"chunk offset must be non-negative, got {self.offset}")
        if self.length < 0:
            raise ValueError(f"chunk length must be non-negative, got {self.length}")


@dataclass(frozen=True)
class ArrayMeta:
    """Metadata for a single zarr array backed by a concatenated payload file.

    This is the Python view of Rust ``ArrayMeta`` + ``ChunkStoreMeta::FileOffset``.
    The payload file holds every encoded chunk concatenated in C-order logical
    chunk index; ``chunks`` gives the ``(offset, length)`` of each chunk.
    """

    shape: tuple[int, ...]
    chunk_shape: tuple[int, ...]
    dtype: DType
    order: ArrayOrder = ArrayOrder.C
    codec: CodecPipeline = field(default_factory=CodecPipeline)
    payload_path: str = ""
    chunks: tuple[ChunkLocation, ...] = ()

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
        if len(self.chunks) != self.num_chunks:
            raise ValueError(
                f"chunks count {len(self.chunks)} != chunk grid size {self.num_chunks}"
            )

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def chunk_grid_shape(self) -> tuple[int, ...]:
        """Number of chunks along each axis: ``ceil(shape / chunk_shape)``."""
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


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


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

    def __post_init__(self) -> None:
        if self.num_cells <= 0:
            raise ValueError(f"dense dataset requires positive num_cells, got {self.num_cells}")
        if self.num_genes <= 0:
            raise ValueError(f"dense dataset requires positive num_genes, got {self.num_genes}")
        if len(self.gene_names) != self.num_genes:
            raise ValueError(
                f"gene_names count {len(self.gene_names)} != num_genes {self.num_genes}"
            )
        elements = 1
        for s in self.data.shape:
            elements *= s
        if elements != self.num_cells * self.num_genes:
            raise ValueError(
                f"data has {elements} elements, expected "
                f"num_cells*num_genes = {self.num_cells * self.num_genes}"
            )


@dataclass(frozen=True)
class SparseDataset:
    """A CSR sparse single-cell matrix.

    Mirrors Rust ``SparseCsrDataset``: ``indptr`` is the length-``num_cells+1``
    CSR offset array (kept in memory, not chunked); ``indices`` and ``data``
    are 1D zarr arrays of length ``nnz``, chunked along the nnz axis.  Chunk
    alignment at write time keeps each cell's ``nnz`` run within a single
    chunk pair so a cell never spans two index/data chunk pairs.
    """

    gene_names: tuple[str, ...]
    indptr: tuple[int, ...]
    indices: ArrayMeta
    data: ArrayMeta
    index_dtype: DType
    num_cells: int
    num_genes: int

    def __post_init__(self) -> None:
        if self.num_cells <= 0:
            raise ValueError(f"sparse dataset requires positive num_cells, got {self.num_cells}")
        if self.num_genes <= 0:
            raise ValueError(f"sparse dataset requires positive num_genes, got {self.num_genes}")
        if len(self.gene_names) != self.num_genes:
            raise ValueError(
                f"gene_names count {len(self.gene_names)} != num_genes {self.num_genes}"
            )
        if len(self.indptr) != self.num_cells + 1:
            raise ValueError(
                f"indptr length {len(self.indptr)} != num_cells+1 {self.num_cells + 1}"
            )
        if not self.index_dtype.is_csr_index:
            raise ValueError(
                f"index_dtype {self.index_dtype!r} is not a valid CSR index dtype"
            )
        if self.indices.ndim != 1 or self.data.ndim != 1:
            raise ValueError("sparse indices and data arrays must be 1D")
        if self.indptr[0] != 0:
            raise ValueError(f"indptr must start at 0, got {self.indptr[0]}")
        if any(v < 0 for v in self.indptr):
            raise ValueError("indptr values must be non-negative")
        if any(
            self.indptr[i] > self.indptr[i + 1] for i in range(len(self.indptr) - 1)
        ):
            raise ValueError("indptr must be monotonically non-decreasing")
        indices_len = 1
        for s in self.indices.shape:
            indices_len *= s
        data_len = 1
        for s in self.data.shape:
            data_len *= s
        nnz = int(self.indptr[-1]) if self.indptr else 0
        if indices_len != nnz:
            raise ValueError(f"indices length {indices_len} != nnz {nnz}")
        if data_len != nnz:
            raise ValueError(f"data length {data_len} != nnz {nnz}")
        if self.indices.dtype != self.index_dtype:
            raise ValueError(
                f"indices dtype {self.indices.dtype!r} != index_dtype {self.index_dtype!r}"
            )


Dataset = DenseDataset | SparseDataset
"""Union type for a parsed dataset, returned by :mod:`scdata.io._launch`."""
