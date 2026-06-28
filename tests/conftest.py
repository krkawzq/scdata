"""Fixture builders for scdata store regression tests.

These build anndata-readable zarr v2 trees *by hand* with the scdata twist:
each array's chunks are concatenated into one payload file and indexed by an
``(offset, length)`` table stored under ``.zattrs["scdata"]``.  This mirrors
what the (not-yet-written) scdata write path will produce, so read/write can
be cross-validated once write lands.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numcodecs import get_codec

ZSTD = {"id": "zstd", "level": 3, "checksum": False}
BLOSC = {
    "id": "blosc",
    "cname": "lz4",
    "clevel": 5,
    "shuffle": 1,
    "blocksize": 0,
    "typesize": 1,
}
GZIP = {"id": "gzip", "level": 5}
VLEN_UTF8 = {"id": "vlen-utf8"}


def product(xs):
    n = 1
    for x in xs:
        n *= x
    return n


def ceil_div(a, b):
    return -(-a // b)


def np_dtype_str(np_dtype) -> str:
    return np.dtype(np_dtype).str


def chunk_slices(shape, chunk_shape):
    """Yield C-order chunk slices, cropped to the array's logical extent."""
    grids = [ceil_div(s, c) for s, c in zip(shape, chunk_shape)]
    for idx in range(product(grids)):
        coord = [0] * len(grids)
        t = idx
        for axis in range(len(grids) - 1, -1, -1):
            coord[axis] = t % grids[axis]
            t //= grids[axis]
        yield tuple(
            slice(co * csh, min(co * csh + csh, sh))
            for co, csh, sh in zip(coord, chunk_shape, shape)
        )


def encode_chunk(raw: bytes, codec_config: dict[str, Any] | None) -> bytes:
    if not codec_config:
        return raw
    codec = get_codec(dict(codec_config))
    out = codec.encode(np.frombuffer(raw, dtype=np.uint8))
    return memoryview(out).tobytes() if not isinstance(out, bytes) else out


def encode_vlen_chunk(strings: list[str], codec_config: dict[str, Any] | None) -> bytes:
    """Encode an object string array chunk: VLenUTF8 then optional compressor."""
    vlen = get_codec(dict(VLEN_UTF8))
    data = vlen.encode(np.asarray(strings, dtype=object))
    data = data.tobytes() if hasattr(data, "tobytes") else bytes(data)
    if codec_config:
        comp = get_codec(dict(codec_config))
        out = comp.encode(np.frombuffer(data, dtype=np.uint8))
        data = memoryview(out).tobytes() if not isinstance(out, bytes) else out
    return data


def build_payload(encoded_chunks: list[bytes]) -> tuple[bytes, list[int], list[int]]:
    buf = bytearray()
    offsets, lengths = [], []
    for enc in encoded_chunks:
        offsets.append(len(buf))
        lengths.append(len(enc))
        buf.extend(enc)
    return bytes(buf), offsets, lengths


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def write_var_index(
    root: Path,
    gene_names: list[str],
    *,
    dtype_kind: str = "S",
    compressor: dict[str, Any] | None = None,
    chunk_shape: tuple[int, ...] = (16,),
) -> None:
    """Write ``var/_index`` as a string array with a scdata chunk index.

    dtype_kind: ``"S"`` fixed-width byte strings, ``"U"`` fixed-width UTF-32,
    or ``"O"`` variable-length (anndata-native, VLenUTF8-encoded).
    """
    vdir = root / "var"
    write_json(vdir / ".zgroup", {"zarr_format": 2})
    idx = vdir / "_index"
    idx.mkdir(parents=True, exist_ok=True)
    n = len(gene_names)

    if dtype_kind == "S":
        width = max((len(n.encode("utf-8")) for n in gene_names), default=1)
        arr = np.array(gene_names, dtype=f"|S{width}")
        zarr_dtype = f"|S{arr.dtype.itemsize}"
        raw = arr.tobytes()
        item = arr.dtype.itemsize
        cs = chunk_shape if chunk_shape else (n,)
        encoded = []
        start = 0
        for sl in chunk_slices([n], cs):
            cnt = sl[0].stop - sl[0].start
            encoded.append(encode_chunk(raw[start:start + cnt * item], compressor))
            start += cnt * item
        filters = None
    elif dtype_kind == "U":
        width = max((len(n) for n in gene_names), default=1)
        arr = np.array(gene_names, dtype=f"<U{width}")
        zarr_dtype = f"<U{width}"
        raw = arr.tobytes()
        item = arr.dtype.itemsize
        cs = chunk_shape if chunk_shape else (n,)
        encoded = []
        start = 0
        for sl in chunk_slices([n], cs):
            cnt = sl[0].stop - sl[0].start
            encoded.append(encode_chunk(raw[start:start + cnt * item], compressor))
            start += cnt * item
        filters = None
    elif dtype_kind == "O":
        zarr_dtype = "|O"
        cs = chunk_shape if chunk_shape else (n,)
        encoded = []
        start = 0
        for sl in chunk_slices([n], cs):
            cnt = sl[0].stop - sl[0].start
            encoded.append(encode_vlen_chunk(gene_names[start:start + cnt], compressor))
            start += cnt
        filters = [dict(VLEN_UTF8)]
    else:
        raise ValueError(f"unknown dtype_kind {dtype_kind!r}")

    payload, offsets, lengths = build_payload(encoded)
    (idx / "payload.bin").write_bytes(payload)
    write_json(idx / ".zarray", {
        "zarr_format": 2,
        "shape": [n],
        "chunks": list(cs),
        "dtype": zarr_dtype,
        "order": "C",
        "filters": filters,
        "compressor": compressor,
        "fill_value": "",
        "dimension_separator": ".",
    })
    write_json(idx / ".zattrs", {
        "encoding-type": "string-array",
        "encoding-version": "0.2.0",
        "scdata": {"payload": "payload.bin", "offsets": offsets, "lengths": lengths},
    })


def build_dense_store(
    root: Path,
    shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    np_dtype,
    compressor: dict[str, Any] | None,
    gene_names: list[str],
    *,
    var_kind: str = "S",
    var_compressor: dict[str, Any] | None = None,
    var_chunk_shape: tuple[int, ...] = (16,),
) -> Path:
    write_json(root / ".zgroup", {"zarr_format": 2})
    write_json(root / ".zattrs", {"encoding-type": "anndata", "encoding-version": "0.2.0"})
    xdir = root / "X"
    xdir.mkdir(parents=True, exist_ok=True)
    arr = np.arange(product(shape), dtype=np_dtype).reshape(shape)
    encoded = [
        encode_chunk(np.ascontiguousarray(arr[sl]).tobytes(order="C"), compressor)
        for sl in chunk_slices(shape, chunk_shape)
    ]
    payload, offsets, lengths = build_payload(encoded)
    (xdir / "payload.bin").write_bytes(payload)
    write_json(xdir / ".zarray", {
        "zarr_format": 2,
        "shape": list(shape),
        "chunks": list(chunk_shape),
        "dtype": np_dtype_str(np_dtype),
        "order": "C",
        "filters": None,
        "compressor": compressor,
        "fill_value": 0,
        "dimension_separator": ".",
    })
    write_json(xdir / ".zattrs", {
        "encoding-type": "array",
        "encoding-version": "0.2.0",
        "scdata": {"payload": "payload.bin", "offsets": offsets, "lengths": lengths},
    })
    write_var_index(
        root, gene_names, dtype_kind=var_kind,
        compressor=var_compressor, chunk_shape=var_chunk_shape,
    )
    return root


def build_sparse_store(
    root: Path,
    num_cells: int,
    num_genes: int,
    nnz_per_cell: int,
    np_dtype,
    index_np_dtype,
    compressor: dict[str, Any] | None,
    gene_names: list[str],
    *,
    var_kind: str = "S",
    var_compressor: dict[str, Any] | None = None,
) -> Path:
    write_json(root / ".zgroup", {"zarr_format": 2})
    write_json(root / ".zattrs", {"encoding-type": "anndata", "encoding-version": "0.2.0"})
    xdir = root / "X"
    write_json(xdir / ".zgroup", {"zarr_format": 2})
    write_json(xdir / ".zattrs", {"encoding-type": "CSR", "encoding-version": "0.2.0"})

    rng = np.random.default_rng(0)
    indptr = np.zeros(num_cells + 1, dtype=np.int64)
    for i in range(num_cells):
        indptr[i + 1] = indptr[i] + nnz_per_cell
    nnz = int(indptr[-1])
    indices = rng.integers(0, num_genes, size=nnz).astype(index_np_dtype)
    data = rng.standard_normal(nnz).astype(np_dtype)

    def write_arr(key: Path, arr1d, dtype_str, comp) -> None:
        key.mkdir(parents=True, exist_ok=True)
        n = arr1d.shape[0]
        encoded = [encode_chunk(np.ascontiguousarray(arr1d).tobytes(), comp)]
        payload, offsets, lengths = build_payload(encoded)
        (key / "payload.bin").write_bytes(payload)
        write_json(key / ".zarray", {
            "zarr_format": 2,
            "shape": [n],
            "chunks": [n],
            "dtype": dtype_str,
            "order": "C",
            "filters": None,
            "compressor": comp,
            "fill_value": 0,
            "dimension_separator": ".",
        })
        write_json(key / ".zattrs", {
            "scdata": {"payload": "payload.bin", "offsets": offsets, "lengths": lengths}
        })

    write_arr(xdir / "indptr", indptr, np_dtype_str(indptr.dtype), None)
    write_arr(xdir / "indices", indices, np_dtype_str(index_np_dtype), compressor)
    write_arr(xdir / "data", data, np_dtype_str(np_dtype), compressor)
    write_var_index(root, gene_names, dtype_kind=var_kind, compressor=var_compressor)
    return root


def zip_store(src: Path, zip_path: Path) -> Path:
    """Pack a directory store into a ZIP_STORED ``.zarr.zip`` archive."""
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(src).as_posix())
    return zip_path


@pytest.fixture(scope="session")
def codec_configs() -> dict[str, dict[str, Any]]:
    return {
        "zstd": ZSTD,
        "blosc": BLOSC,
        "gzip": GZIP,
        "vlen_utf8": VLEN_UTF8,
    }


@pytest.fixture
def dense_store_factory(tmp_path: Path):
    def factory(
        name: str,
        shape: tuple[int, ...] = (3, 4),
        chunk_shape: tuple[int, ...] = (2, 2),
        np_dtype=np.float32,
        compressor: dict[str, Any] | None = None,
        gene_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Path:
        if gene_names is None:
            n_genes = shape[-1] if len(shape) > 1 else 4
            gene_names = [f"g{i}" for i in range(n_genes)]
        return build_dense_store(
            tmp_path / name,
            shape,
            chunk_shape,
            np_dtype,
            compressor,
            gene_names,
            **kwargs,
        )

    return factory


@pytest.fixture
def sparse_store_factory(tmp_path: Path):
    def factory(
        name: str,
        num_cells: int = 4,
        num_genes: int = 6,
        nnz_per_cell: int = 3,
        np_dtype=np.float32,
        index_np_dtype=np.int32,
        compressor: dict[str, Any] | None = None,
        gene_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Path:
        if gene_names is None:
            gene_names = [f"g{i}" for i in range(num_genes)]
        return build_sparse_store(
            tmp_path / name,
            num_cells,
            num_genes,
            nnz_per_cell,
            np_dtype,
            index_np_dtype,
            compressor,
            gene_names,
            **kwargs,
        )

    return factory


@pytest.fixture
def zip_store_factory(tmp_path: Path):
    def factory(src: Path, name: str = "store.zarr.zip") -> Path:
        return zip_store(src, tmp_path / name)

    return factory
