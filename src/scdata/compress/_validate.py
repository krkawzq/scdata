"""Private call-site validation for paths, shapes, and NumPy buffers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from scdata._validate import as_int as _as_int
from scdata._validate import ensure_path as ensure_path
from scdata.compress._format import VALUE_DTYPES, is_value_dtype

_UINT64_MAX = int(np.iinfo(np.uint64).max)
_UINTP_MAX = int(np.iinfo(np.uintp).max)
_INT64_MAX = int(np.iinfo(np.int64).max)

PartitionPolicy = Literal["cells", "budget"]
DEFAULT_CHUNK_CELLS = 1024
DEFAULT_BLOCK_CELLS = 16
DEFAULT_CHUNK_BUDGET = 100 << 20  # 100 MiB
DEFAULT_BLOCK_BUDGET = 64 << 10  # 64 KiB


@dataclass(frozen=True)
class ResolvedPartition:
    """Normalized partition forwarded to ``scdata._core`` writers."""

    policy: PartitionPolicy
    n: int


def dense_cells_for_budget(budget: int, row_bytes: int) -> int:
    """Smallest fixed-cell count whose packed size is >= ``budget``.

    When a single row already exceeds the budget, returns ``1``.
    """
    budget_n = as_int(budget, name="budget", minimum=1)
    row_n = as_int(row_bytes, name="row_bytes", minimum=0)
    if row_n == 0:
        return 1
    return max(1, (budget_n + row_n - 1) // row_n)


def as_int(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int = _UINT64_MAX,
) -> int:
    """Coerce ``value`` to ``int`` with inclusive bounds."""
    return _as_int(value, name, minimum=minimum, maximum=maximum)


def normalize_policy(policy: object, *, name: str) -> PartitionPolicy:
    if not isinstance(policy, str):
        raise TypeError(f"{name} must be 'cells' or 'budget', got {type(policy).__name__}")
    normalized = policy.casefold().replace("-", "_")
    if normalized in {"cells", "fixed_cells"}:
        return "cells"
    if normalized in {"budget", "bytes_budget"}:
        return "budget"
    raise ValueError(f"{name} must be 'cells' or 'budget', got {policy!r}")


def resolve_partition(
    *,
    policy: object,
    cells: object | None,
    budget: object | None,
    default_cells: int,
    name: str,
) -> ResolvedPartition:
    """Resolve one chunk/block partition from policy + cells/budget knobs."""
    resolved_policy = normalize_policy(policy, name=f"{name}_policy")
    if resolved_policy == "cells":
        value = default_cells if cells is None else cells
        return ResolvedPartition(
            policy="cells",
            n=as_int(value, name=f"{name}_cells", minimum=1),
        )
    if budget is None:
        raise ValueError(f"{name}_budget is required when {name}_policy='budget'")
    return ResolvedPartition(
        policy="budget",
        n=as_int(budget, name=f"{name}_budget", minimum=1),
    )


def resolve_write_partitions(
    *,
    chunk_policy: object = "budget",
    block_policy: object = "budget",
    chunk_cells: object | None = None,
    block_cells: object | None = None,
    chunk_budget: object | None = DEFAULT_CHUNK_BUDGET,
    block_budget: object | None = DEFAULT_BLOCK_BUDGET,
    dense: bool = False,
    row_bytes: object | None = None,
) -> tuple[ResolvedPartition, ResolvedPartition]:
    """Resolve chunk/block partitions for a write call.

    ``policy='cells'`` maps to Rust ``Partition::FixedCells``; ``policy='budget'``
    maps to ``Partition::BytesBudget`` for CSR. Dense ``budget`` policies are
    lowered in Python to ``fixed_cells`` that meet or slightly exceed the budget
    given ``row_bytes = n_cols * dtype.itemsize``.
    """
    chunk = resolve_partition(
        policy=chunk_policy,
        cells=chunk_cells,
        budget=chunk_budget,
        default_cells=DEFAULT_CHUNK_CELLS,
        name="chunk",
    )
    block = resolve_partition(
        policy=block_policy,
        cells=block_cells,
        budget=block_budget,
        default_cells=DEFAULT_BLOCK_CELLS,
        name="block",
    )
    if not dense:
        return chunk, block

    if chunk.policy == "budget" or block.policy == "budget":
        if row_bytes is None:
            raise ValueError(
                "dense budget partitions require row_bytes "
                "(n_cols * dtype.itemsize) to lower to fixed_cells"
            )
        row_n = as_int(row_bytes, name="row_bytes", minimum=0)

        def _lower(part: ResolvedPartition) -> ResolvedPartition:
            if part.policy == "cells":
                return part
            return ResolvedPartition(
                policy="cells",
                n=dense_cells_for_budget(part.n, row_n),
            )

        chunk = _lower(chunk)
        block = _lower(block)
        if chunk.n < block.n:
            chunk = ResolvedPartition(policy="cells", n=block.n)
    return chunk, block


def ensure_writable_path(path: Path, *, overwrite: bool) -> None:
    """Refuse to clobber an existing path unless ``overwrite`` is true."""
    if path.name in ("", ".", ".."):
        raise ValueError(f"output path must name a file or directory, got {path}")
    if not isinstance(overwrite, (bool, np.bool_)):
        raise TypeError(f"overwrite must be bool, got {type(overwrite).__name__}")
    if path.exists() or path.is_symlink():
        if not overwrite:
            raise FileExistsError(f"path already exists: {path} (pass overwrite=True to replace)")


def is_sparse_matrix(matrix: Any) -> bool:
    """Return whether ``matrix`` is a SciPy sparse matrix (or backed CSR/CSC)."""
    try:
        from scipy import sparse
    except ImportError:  # pragma: no cover - optional dependency
        sparse = None
    if sparse is not None and sparse.issparse(matrix):
        return True
    return bool(
        getattr(matrix, "format", None) in ("csr", "csc")
        and callable(getattr(matrix, "to_memory", None))
    )


def normalize_row_range(start: int, stop: int, n_rows: int) -> tuple[int, int]:
    start_i = as_int(start, name="start")
    stop_i = as_int(stop, name="stop")
    if start_i < 0 or stop_i < start_i or stop_i > n_rows:
        raise ValueError(f"row range [{start_i}, {stop_i}) outside 0..{n_rows}")
    return start_i, stop_i


def _require_value_dtype(array: NDArray[Any], *, what: str) -> NDArray[Any]:
    native_dtype = array.dtype.newbyteorder("=")
    if not is_value_dtype(native_dtype):
        allowed = ", ".join(sorted(dtype.name for dtype in VALUE_DTYPES))
        raise ValueError(f"{what} dtype must be one of [{allowed}], got {array.dtype}")
    if array.dtype != native_dtype:
        array = array.astype(native_dtype, copy=False)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array


def _as_unmasked_array(value: Any, *, what: str) -> NDArray[Any]:
    try:
        if np.ma.isMaskedArray(value) and np.ma.getmaskarray(value).any():
            raise ValueError(f"{what} contains masked values; fill them explicitly before writing")
        return np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{what} cannot be converted to a NumPy array: {error}") from error


def normalize_dense(values: Any) -> NDArray[Any]:
    """Coerce to a C-contiguous 2D array with a supported value dtype."""
    array = _as_unmasked_array(values, what="dense values")
    if array.ndim != 2:
        raise ValueError(f"dense values must be 2-dimensional, got shape {array.shape}")
    return _require_value_dtype(array, what="dense values")


def promote_u64(array: Any, *, name: str, length: int | None = None) -> NDArray[np.uint64]:
    """Promote an integer index/offset array to contiguous ``uint64``."""
    values = _as_unmasked_array(array, what=name)
    if values.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {values.shape}")
    if length is not None and values.shape[0] != length:
        raise ValueError(f"{name} length {values.shape[0]} does not match expected {length}")
    if values.dtype.kind not in "iu":
        raise TypeError(f"{name} must be an integer array, got dtype {values.dtype}")
    if values.dtype.kind == "i" and values.size and int(values.min()) < 0:
        raise ValueError(f"{name} contains negative values")
    if values.dtype.itemsize > 8:
        raise ValueError(f"{name} dtype {values.dtype} exceeds uint64")
    return np.ascontiguousarray(values, dtype=np.uint64)


def normalize_csr_arrays(
    indptr: Any,
    indices: Any,
    data: Any,
    shape: Sequence[int] | tuple[int, int],
) -> tuple[NDArray[np.uint64], NDArray[np.uint64], NDArray[Any], tuple[int, int]]:
    """Validate CSR components; promote offsets/indices to contiguous ``uint64``."""
    try:
        shape_values = tuple(shape)
    except TypeError:
        raise TypeError(f"shape must be an iterable of two integers, got {shape!r}")
    if len(shape_values) != 2:
        raise ValueError(f"shape must have length 2, got {shape!r}")
    n_rows = as_int(shape_values[0], name="shape[0]")
    n_cols = as_int(shape_values[1], name="shape[1]")

    indptr_u64 = promote_u64(indptr, name="indptr", length=n_rows + 1)
    indices_u64 = promote_u64(indices, name="indices")
    data_arr = _as_unmasked_array(data, what="CSR data")
    if data_arr.ndim != 1:
        raise ValueError(f"data must be 1-dimensional, got shape {data_arr.shape}")
    data_arr = _require_value_dtype(data_arr, what="CSR data")
    if indices_u64.shape[0] != data_arr.shape[0]:
        raise ValueError(
            f"indices length {indices_u64.shape[0]} != data length {data_arr.shape[0]}"
        )
    if int(indptr_u64[0]) != 0:
        raise ValueError(f"indptr[0] must be 0, got {int(indptr_u64[0])}")
    nnz = int(indptr_u64[-1])
    if nnz != indices_u64.shape[0]:
        raise ValueError(f"indptr[-1] ({nnz}) does not match nnz ({indices_u64.shape[0]})")
    if nnz and int(indices_u64.max()) >= n_cols:
        raise ValueError(f"indices contain column {int(indices_u64.max())} outside 0..{n_cols}")
    if not np.all(indptr_u64[1:] >= indptr_u64[:-1]):
        raise ValueError("indptr must be monotonically non-decreasing")
    return indptr_u64, indices_u64, data_arr, (n_rows, n_cols)


def require_scipy_csr(matrix: Any) -> Any:
    """Return a SciPy CSR matrix, converting with ``tocsr()`` when needed."""
    sparse = _scipy_sparse()
    if getattr(matrix, "format", None) == "csr" and sparse.issparse(matrix):
        return matrix
    if sparse.issparse(matrix):
        return matrix.tocsr()
    if hasattr(matrix, "tocsr"):
        converted = matrix.tocsr()
        if getattr(converted, "format", None) == "csr" and sparse.issparse(converted):
            return converted
    raise TypeError("csr must be a scipy.sparse matrix (or provide .tocsr() -> csr_matrix)")


def csr_matrix_from_decoded(
    indices: Any,
    data: Any,
    indptr: Any,
    *,
    n_rows: int,
    n_cols: int,
) -> Any:
    """Build a SciPy CSR matrix from decoded buffers."""
    sparse = _scipy_sparse()
    if n_rows < 0 or n_cols < 0:
        raise ValueError(f"CSR shape must be non-negative, got ({n_rows}, {n_cols})")
    if n_rows > _INT64_MAX or n_cols > _INT64_MAX:
        raise ValueError(f"CSR shape ({n_rows}, {n_cols}) exceeds SciPy's signed int64 index range")

    indices_array = np.asarray(indices)
    max_index = int(indices_array.max()) if indices_array.size else 0
    if max_index > _INT64_MAX:
        raise ValueError("CSR column indices exceed SciPy's signed int64 index range")
    use_int64 = n_cols > np.iinfo(np.int32).max or max_index > np.iinfo(np.int32).max
    indices_signed = indices_array.astype(np.int64 if use_int64 else np.int32, copy=False)

    indptr_array = np.asarray(indptr)
    if indptr_array.size and int(indptr_array[-1]) > np.iinfo(np.int64).max:
        raise ValueError("CSR row offsets exceed SciPy's int64 index range")
    indptr_signed = indptr_array.astype(np.int64, copy=False)
    return sparse.csr_matrix(
        (np.asarray(data), indices_signed, indptr_signed),
        shape=(n_rows, n_cols),
    )


def _scipy_sparse() -> Any:
    try:
        from scipy import sparse
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("CSR support requires scipy; install with scdata-toolkit[scipy]") from exc
    return sparse
