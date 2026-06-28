#!/usr/bin/env python3
"""CLI benchmark for compression ratio and single-thread decode throughput.

The default input target is the Tahoe control H5AD used during development.
The benchmark samples numeric HDF5 datasets, compresses the same raw chunks
with each codec, then times repeated in-memory decode passes. Timed decode
does not include HDF5 reads or filesystem I/O.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import gc
import importlib
import importlib.metadata as metadata
import json
import math
import os
import platform
import random
import resource
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import product as iter_product
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("BLOSC_NTHREADS", "1")

import h5py
import numpy as np
from numcodecs import BZ2, GZip, LZ4, LZMA, Blosc, Zlib, Zstd

try:
    import zarr
except Exception:  # pragma: no cover
    zarr = None

try:
    import numcodecs
except Exception:  # pragma: no cover
    numcodecs = None

try:
    numcodecs_blosc = importlib.import_module("numcodecs.blosc")
except Exception:  # pragma: no cover
    numcodecs_blosc = None


DEFAULT_INPUT = Path(
    "/mnt/shared-storage-user/dnacoding/wangzhongqi/Code/Project/PRISM/data/"
    "tahoe_dmso_control.h5ad"
)
DEFAULT_OUTPUT_DIR = Path("outputs/bench_compression")

NumericKind = Literal["bool", "int", "uint", "float"]
SelectionMode = Literal["largest", "stratified", "all"]
VerifyMode = Literal["none", "first", "all"]
DecodeOrder = Literal["sequential", "random"]
Profile = Literal["quick", "default", "broad", "all"]
SortBy = Literal["decode", "ratio", "compress", "name", "input"]


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    dtype_kind: NumericKind
    itemsize: int
    n_items: int
    nbytes: int
    chunks: tuple[int, ...] | None
    compression: str | None


@dataclass(frozen=True)
class SampleChunk:
    dataset: str
    dtype: str
    itemsize: int
    offset: int
    shape: tuple[int, ...]
    raw: bytes

    @property
    def nbytes(self) -> int:
        return len(self.raw)

    @property
    def n_items(self) -> int:
        if self.itemsize == 0:
            return 0
        return self.nbytes // self.itemsize

    def ndarray(self) -> np.ndarray[Any, np.dtype[Any]]:
        return np.frombuffer(self.raw, dtype=np.dtype(self.dtype))


@dataclass
class DatasetSampleStats:
    sampled_bytes: int = 0
    sampled_chunks: int = 0
    read_seconds: float = 0.0


@dataclass(frozen=True)
class EncodedChunk:
    sample: SampleChunk
    payload: bytes


@dataclass
class CodecResult:
    status: str
    algorithm: str
    family: str
    raw_bytes: int
    compressed_bytes: int | None
    compressed_over_raw: float | None
    raw_over_compressed: float | None
    compress_seconds: float | None
    compress_mib_s: float | None
    decode_best_seconds: float | None
    decode_median_seconds: float | None
    decode_mean_seconds: float | None
    decode_std_seconds: float | None
    decode_best_mib_s: float | None
    decode_median_mib_s: float | None
    decode_best_gib_s: float | None
    decode_runs_seconds: list[float]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class CodecAdapter:
    name: str
    family: str
    notes: str

    def encode(self, chunk: SampleChunk) -> bytes:
        raise NotImplementedError

    def decode(self, payload: bytes, chunk: SampleChunk) -> Any:
        raise NotImplementedError


class UncompressedCopy(CodecAdapter):
    name = "none.copy"
    family = "baseline"
    notes = "uncompressed baseline; decode is one in-memory bytes copy"

    def encode(self, chunk: SampleChunk) -> bytes:
        return chunk.raw

    def decode(self, payload: bytes, chunk: SampleChunk) -> bytes:
        return memoryview(payload).tobytes()


class NumcodecBytes(CodecAdapter):
    def __init__(
        self,
        name: str,
        family: str,
        factory: Callable[[], Any],
        notes: str = "",
    ) -> None:
        self.name = name
        self.family = family
        self._factory = factory
        self._codec: Any | None = None
        self.notes = notes

    @property
    def codec(self) -> Any:
        if self._codec is None:
            self._codec = self._factory()
        return self._codec

    def encode(self, chunk: SampleChunk) -> bytes:
        return as_bytes(self.codec.encode(chunk.ndarray()))

    def decode(self, payload: bytes, chunk: SampleChunk) -> Any:
        return self.codec.decode(payload)


class BloscByItemsize(CodecAdapter):
    def __init__(
        self,
        cname: str,
        *,
        clevel: int,
        shuffle: str = "auto",
        blocksize: int = 0,
    ) -> None:
        self.cname = cname
        self.clevel = clevel
        self.shuffle = shuffle
        self.blocksize = blocksize
        suffix = f"clevel{clevel}"
        if shuffle != "auto":
            suffix += f".{shuffle}"
        self.name = f"blosc.{cname}.{suffix}"
        self.family = "blosc"
        self.notes = (
            "Blosc with dtype-aware typesize; auto shuffle matches Zarr "
            "BloscCodec evolution"
        )
        self._cache: dict[tuple[int, int], Blosc] = {}

    def _shuffle_code(self, chunk: SampleChunk) -> int:
        if self.shuffle == "auto":
            return Blosc.BITSHUFFLE if chunk.itemsize == 1 else Blosc.SHUFFLE
        mapping = {
            "noshuffle": Blosc.NOSHUFFLE,
            "shuffle": Blosc.SHUFFLE,
            "bitshuffle": Blosc.BITSHUFFLE,
        }
        return mapping[self.shuffle]

    def _codec(self, chunk: SampleChunk) -> Blosc:
        typesize = max(1, chunk.itemsize)
        shuffle = self._shuffle_code(chunk)
        key = (typesize, shuffle)
        if key not in self._cache:
            self._cache[key] = Blosc(
                cname=self.cname,
                clevel=self.clevel,
                shuffle=shuffle,
                blocksize=self.blocksize,
                typesize=typesize,
            )
        return self._cache[key]

    def encode(self, chunk: SampleChunk) -> bytes:
        return as_bytes(self._codec(chunk).encode(chunk.ndarray()))

    def decode(self, payload: bytes, chunk: SampleChunk) -> Any:
        return self._codec(chunk).decode(payload)


class UnavailableCodec(CodecAdapter):
    def __init__(self, name: str, family: str, reason: str) -> None:
        self.name = name
        self.family = family
        self.notes = reason

    def encode(self, chunk: SampleChunk) -> bytes:
        raise RuntimeError(self.notes)

    def decode(self, payload: bytes, chunk: SampleChunk) -> Any:
        raise RuntimeError(self.notes)


def parse_size(text: str) -> int:
    value = text.strip()
    suffixes = {
        "kib": 1 << 10,
        "mib": 1 << 20,
        "gib": 1 << 30,
        "tib": 1 << 40,
        "kb": 10**3,
        "mb": 10**6,
        "gb": 10**9,
        "tb": 10**12,
        "k": 1 << 10,
        "m": 1 << 20,
        "g": 1 << 30,
        "t": 1 << 40,
    }
    lower = value.lower()
    for suffix, scale in suffixes.items():
        if lower.endswith(suffix):
            return int(float(lower[: -len(suffix)]) * scale)
    return int(value)


def parse_csv_list(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        out.extend(item.strip() for item in value.split(",") if item.strip())
    return out


def human_bytes(n: int | float | None) -> str:
    if n is None:
        return "n/a"
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{value:.0f} B"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def as_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if hasattr(value, "tobytes"):
        return value.tobytes()
    return bytes(value)


def output_nbytes(value: Any) -> int:
    if isinstance(value, bytes | bytearray | memoryview):
        return len(value)
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    return len(value)


def version_of(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def numeric_kind(dtype: np.dtype[Any]) -> NumericKind | None:
    if dtype.kind == "b":
        return "bool"
    if dtype.kind == "i":
        return "int"
    if dtype.kind == "u":
        return "uint"
    if dtype.kind == "f":
        return "float"
    return None


def product(values: Iterable[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def configure_threads(threads: int) -> None:
    if numcodecs_blosc is not None:
        try:
            set_nthreads = getattr(numcodecs_blosc, "set_nthreads", None)
            if callable(set_nthreads):
                set_nthreads(threads)
        except Exception:
            pass
        try:
            setattr(numcodecs_blosc, "use_threads", threads > 1)
        except Exception:
            pass


def collect_dataset_infos(path: Path) -> list[DatasetInfo]:
    infos: list[DatasetInfo] = []
    with h5py.File(path, "r") as h5:
        def visit(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset) or obj.shape is None:
                return
            dtype = np.dtype(obj.dtype)
            kind = numeric_kind(dtype)
            size = obj.size
            if kind is None or size is None or size == 0 or dtype.itemsize <= 0:
                return
            infos.append(
                DatasetInfo(
                    name=name,
                    shape=tuple(int(v) for v in obj.shape),
                    dtype=str(dtype),
                    dtype_kind=kind,
                    itemsize=int(dtype.itemsize),
                    n_items=int(size),
                    nbytes=int(size * dtype.itemsize),
                    chunks=tuple(int(v) for v in obj.chunks) if obj.chunks else None,
                    compression=obj.compression,
                )
            )

        h5.visititems(visit)
    infos.sort(key=lambda x: x.nbytes, reverse=True)
    return infos


def matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def select_datasets(
    infos: list[DatasetInfo],
    *,
    include_patterns: list[str],
    exclude_patterns: list[str],
    min_dataset_bytes: int,
    max_datasets: int,
    selection: SelectionMode,
) -> list[DatasetInfo]:
    if include_patterns:
        candidates = [info for info in infos if matches_any(info.name, include_patterns)]
    else:
        candidates = [info for info in infos if info.nbytes >= min_dataset_bytes]

    if exclude_patterns:
        candidates = [info for info in candidates if not matches_any(info.name, exclude_patterns)]

    if not candidates:
        return []
    if selection == "all":
        return candidates[:max_datasets]
    if selection == "largest":
        return sorted(candidates, key=lambda x: x.nbytes, reverse=True)[:max_datasets]

    selected: list[DatasetInfo] = []
    largest_quota = max(1, max_datasets // 2)
    for info in sorted(candidates, key=lambda x: x.nbytes, reverse=True)[:largest_quota]:
        selected.append(info)

    seen_names = {info.name for info in selected}
    dtype_keys = sorted(
        {(info.dtype_kind, info.itemsize) for info in candidates},
        key=lambda x: (x[0], x[1]),
    )
    for key in dtype_keys:
        if len(selected) >= max_datasets:
            break
        reps = [
            info
            for info in candidates
            if (info.dtype_kind, info.itemsize) == key and info.name not in seen_names
        ]
        if reps:
            info = max(reps, key=lambda x: x.nbytes)
            selected.append(info)
            seen_names.add(info.name)

    for info in sorted(candidates, key=lambda x: x.nbytes, reverse=True):
        if len(selected) >= max_datasets:
            break
        if info.name not in seen_names:
            selected.append(info)
            seen_names.add(info.name)

    return selected


def sample_positions(n_units: int, block_units: int, n_blocks: int, seed: int) -> list[int]:
    max_start = max(0, n_units - block_units)
    if n_blocks <= 1:
        return [0]
    if n_blocks > max_start + 1:
        return list(range(max_start + 1))

    # Mix deterministic coverage of the full dataset with random interior points.
    edge_positions = np.linspace(0, max_start, num=min(n_blocks, 4), dtype=np.int64).tolist()
    rng = random.Random(seed)
    positions = {int(pos) for pos in edge_positions}
    while len(positions) < n_blocks:
        positions.add(rng.randint(0, max_start))
    return sorted(positions)


def read_dataset_block(ds: h5py.Dataset, offset: int, block_units: int) -> np.ndarray[Any, np.dtype[Any]]:
    if len(ds.shape) == 1:
        return np.asarray(ds[offset : offset + block_units])
    return np.asarray(ds[offset : offset + block_units, ...])


def read_samples(
    path: Path,
    selected: list[DatasetInfo],
    *,
    sample_bytes: int,
    block_bytes: int,
    min_sample_per_dataset: int,
    seed: int,
) -> tuple[list[SampleChunk], dict[str, DatasetSampleStats]]:
    selected_total = sum(info.nbytes for info in selected)
    chunks: list[SampleChunk] = []
    stats = {info.name: DatasetSampleStats() for info in selected}

    with h5py.File(path, "r") as h5:
        for index, info in enumerate(selected):
            ds = h5[info.name]
            assert isinstance(ds, h5py.Dataset)
            target = int(sample_bytes * info.nbytes / selected_total)
            target = max(min_sample_per_dataset, target)
            target = min(info.nbytes, target)

            if len(info.shape) == 1:
                unit_items = 1
                n_units = info.n_items
            else:
                unit_items = product(info.shape[1:])
                n_units = int(info.shape[0])

            unit_bytes = max(1, unit_items * info.itemsize)
            block_units = max(1, min(n_units, block_bytes // unit_bytes))
            block_raw_bytes = block_units * unit_bytes
            n_blocks = max(1, math.ceil(target / block_raw_bytes))

            positions = sample_positions(
                n_units=n_units,
                block_units=block_units,
                n_blocks=n_blocks,
                seed=seed + index,
            )
            for offset in positions:
                started = time.perf_counter()
                data = read_dataset_block(ds, offset, block_units)
                data = np.ascontiguousarray(data)
                raw = data.tobytes(order="C")
                elapsed = time.perf_counter() - started

                chunks.append(
                    SampleChunk(
                        dataset=info.name,
                        dtype=info.dtype,
                        itemsize=info.itemsize,
                        offset=offset,
                        shape=tuple(int(v) for v in data.shape),
                        raw=raw,
                    )
                )
                ds_stats = stats[info.name]
                ds_stats.sampled_bytes += len(raw)
                ds_stats.sampled_chunks += 1
                ds_stats.read_seconds += elapsed

    return chunks, stats


def add_codec(
    codecs: list[CodecAdapter],
    seen: set[str],
    codec: CodecAdapter,
) -> None:
    if codec.name not in seen:
        codecs.append(codec)
        seen.add(codec.name)


def optional_unavailable_codecs() -> list[CodecAdapter]:
    optional: list[CodecAdapter] = []
    try:
        from numcodecs.pcodec import PCodec  # type: ignore

        optional.append(
            NumcodecBytes(
                "pcodec.default",
                "pcodec",
                lambda: PCodec(),
                "optional numcodecs PCodec",
            )
        )
    except Exception as exc:
        optional.append(UnavailableCodec("pcodec.default", "pcodec", f"unavailable: {exc}"))

    try:
        from numcodecs.zfpy import ZFPY  # type: ignore

        optional.append(
            NumcodecBytes(
                "zfpy.default",
                "zfpy",
                lambda: ZFPY(),
                "optional numcodecs ZFPY; may be lossy depending on config",
            )
        )
    except Exception as exc:
        optional.append(UnavailableCodec("zfpy.default", "zfpy", f"unavailable: {exc}"))
    return optional


def build_codecs(
    *,
    profile: Profile,
    include_baseline: bool,
    include_optional: bool,
    skip_slow: bool,
    blosc_shuffles: list[str],
) -> list[CodecAdapter]:
    codecs: list[CodecAdapter] = []
    seen: set[str] = set()
    if include_baseline:
        add_codec(codecs, seen, UncompressedCopy())

    def add_num(
        name: str,
        family: str,
        factory: Callable[[], Any],
        notes: str = "",
        *,
        slow: bool = False,
    ) -> None:
        if slow and skip_slow:
            return
        add_codec(codecs, seen, NumcodecBytes(name, family, factory, notes))

    if profile == "quick":
        zstd_levels = [0]
        gzip_levels = []
        zlib_levels = []
        lz4_accels = [1]
        bz2_levels: list[int] = []
        lzma_presets: list[int | None] = []
        blosc_levels = [5]
        blosc_cnames = ["lz4", "zstd"]
    elif profile == "default":
        zstd_levels = [0, 3]
        gzip_levels = [5]
        zlib_levels = [1]
        lz4_accels = [1]
        bz2_levels = [1]
        lzma_presets = [None]
        blosc_levels = [5]
        blosc_cnames = ["lz4", "lz4hc", "blosclz", "snappy", "zlib", "zstd"]
    elif profile == "broad":
        zstd_levels = [0, 1, 3, 5]
        gzip_levels = [1, 5, 9]
        zlib_levels = [1, 3, 6]
        lz4_accels = [1, 4]
        bz2_levels = [1, 5]
        lzma_presets = [0, 3]
        blosc_levels = [1, 5, 9]
        blosc_cnames = ["lz4", "lz4hc", "blosclz", "snappy", "zlib", "zstd"]
    else:
        zstd_levels = [0, 1, 3, 5, 10, 15]
        gzip_levels = [1, 5, 9]
        zlib_levels = [1, 3, 6, 9]
        lz4_accels = [1, 4, 8]
        bz2_levels = [1, 5, 9]
        lzma_presets = [0, 3, 6]
        blosc_levels = [1, 3, 5, 7, 9]
        blosc_cnames = ["lz4", "lz4hc", "blosclz", "snappy", "zlib", "zstd"]

    for level in zstd_levels:
        add_num(
            f"zstd.level{level}",
            "zstd",
            lambda level=level: Zstd(level=level, checksum=False),
            "numcodecs/Zarr Zstd; level 0 is Zarr v3 default",
        )
    for level in gzip_levels:
        add_num(
            f"gzip.level{level}",
            "gzip",
            lambda level=level: GZip(level=level),
            "numcodecs GZip; Zarr GzipCodec default is level 5",
        )
    for level in zlib_levels:
        add_num(
            f"zlib.level{level}",
            "zlib",
            lambda level=level: Zlib(level=level),
            "numcodecs zlib",
        )
    for accel in lz4_accels:
        add_num(
            f"lz4.accel{accel}",
            "lz4",
            lambda accel=accel: LZ4(acceleration=accel),
            "numcodecs LZ4",
        )
    for level in bz2_levels:
        add_num(
            f"bz2.level{level}",
            "bz2",
            lambda level=level: BZ2(level=level),
            "numcodecs BZ2",
            slow=True,
        )
    for preset in lzma_presets:
        name = "lzma.default" if preset is None else f"lzma.preset{preset}"
        add_num(
            name,
            "lzma",
            (lambda: LZMA()) if preset is None else (lambda preset=preset: LZMA(preset=preset)),
            "numcodecs LZMA",
            slow=True,
        )

    for cname in blosc_cnames:
        for clevel in blosc_levels:
            for shuffle in blosc_shuffles:
                add_codec(
                    codecs,
                    seen,
                    BloscByItemsize(cname, clevel=clevel, shuffle=shuffle),
                )

    if include_optional:
        for codec in optional_unavailable_codecs():
            add_codec(codecs, seen, codec)

    return codecs


def filter_codecs(
    codecs: list[CodecAdapter],
    *,
    only_patterns: list[str],
    exclude_patterns: list[str],
) -> list[CodecAdapter]:
    filtered = codecs
    if only_patterns:
        filtered = [codec for codec in filtered if matches_any(codec.name, only_patterns)]
    if exclude_patterns:
        filtered = [codec for codec in filtered if not matches_any(codec.name, exclude_patterns)]
    return filtered


def verify_decoded(codec: CodecAdapter, records: list[EncodedChunk], mode: VerifyMode) -> None:
    if mode == "none":
        return
    if mode == "first":
        records_to_check = records[:1]
    else:
        records_to_check = records
    for record in records_to_check:
        decoded = as_bytes(codec.decode(record.payload, record.sample))
        if decoded != record.sample.raw:
            raise RuntimeError(
                f"round-trip mismatch for {codec.name} on dataset {record.sample.dataset}"
            )


def decode_pass(
    codec: CodecAdapter,
    records: list[EncodedChunk],
    *,
    order: DecodeOrder,
    seed: int,
) -> tuple[float, int]:
    if order == "random":
        rng = random.Random(seed)
        run_records = records.copy()
        rng.shuffle(run_records)
    else:
        run_records = records

    total_out = 0
    gc_was_enabled = gc.isenabled()
    gc.disable()
    started = time.perf_counter()
    try:
        for record in run_records:
            total_out += output_nbytes(codec.decode(record.payload, record.sample))
    finally:
        if gc_was_enabled:
            gc.enable()
    return time.perf_counter() - started, total_out


def aggregate_by_dataset(records: list[EncodedChunk]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for record in records:
        name = record.sample.dataset
        row = rows.setdefault(
            name,
            {
                "dataset": name,
                "dtype": record.sample.dtype,
                "chunks": 0,
                "raw_bytes": 0,
                "compressed_bytes": 0,
            },
        )
        row["chunks"] += 1
        row["raw_bytes"] += record.sample.nbytes
        row["compressed_bytes"] += len(record.payload)
    for row in rows.values():
        raw_bytes = row["raw_bytes"]
        compressed_bytes = row["compressed_bytes"]
        row["compressed_over_raw"] = compressed_bytes / raw_bytes if raw_bytes else None
        row["raw_over_compressed"] = raw_bytes / compressed_bytes if compressed_bytes else None
    return sorted(rows.values(), key=lambda x: x["raw_bytes"], reverse=True)


def benchmark_codec(
    codec: CodecAdapter,
    chunks: list[SampleChunk],
    *,
    repeats: int,
    warmups: int,
    verify: VerifyMode,
    decode_order: DecodeOrder,
    seed: int,
) -> tuple[CodecResult, list[dict[str, Any]]]:
    raw_bytes = sum(chunk.nbytes for chunk in chunks)
    records: list[EncodedChunk] = []

    gc.collect()
    started = time.perf_counter()
    for chunk in chunks:
        records.append(EncodedChunk(sample=chunk, payload=codec.encode(chunk)))
    compress_seconds = time.perf_counter() - started
    compressed_bytes = sum(len(record.payload) for record in records)

    verify_decoded(codec, records, verify)

    for i in range(warmups):
        elapsed, total_out = decode_pass(
            codec,
            records,
            order=decode_order,
            seed=seed + i,
        )
        if total_out != raw_bytes:
            raise RuntimeError(f"decoded {total_out} bytes, expected {raw_bytes}")
        # Keep the variable live so static analyzers do not consider this loop empty.
        _ = elapsed

    times: list[float] = []
    for i in range(repeats):
        elapsed, total_out = decode_pass(
            codec,
            records,
            order=decode_order,
            seed=seed + warmups + i,
        )
        if total_out != raw_bytes:
            raise RuntimeError(f"decoded {total_out} bytes, expected {raw_bytes}")
        times.append(elapsed)

    best = min(times) if times else None
    median = statistics.median(times) if times else None
    mean = statistics.mean(times) if times else None
    std = statistics.stdev(times) if len(times) > 1 else 0.0 if times else None
    per_dataset = aggregate_by_dataset(records)
    result = CodecResult(
        status="ok",
        algorithm=codec.name,
        family=codec.family,
        raw_bytes=raw_bytes,
        compressed_bytes=compressed_bytes,
        compressed_over_raw=compressed_bytes / raw_bytes if raw_bytes else None,
        raw_over_compressed=raw_bytes / compressed_bytes if compressed_bytes else None,
        compress_seconds=compress_seconds,
        compress_mib_s=raw_bytes / compress_seconds / (1 << 20) if compress_seconds else None,
        decode_best_seconds=best,
        decode_median_seconds=median,
        decode_mean_seconds=mean,
        decode_std_seconds=std,
        decode_best_mib_s=raw_bytes / best / (1 << 20) if best else None,
        decode_median_mib_s=raw_bytes / median / (1 << 20) if median else None,
        decode_best_gib_s=raw_bytes / best / (1 << 30) if best else None,
        decode_runs_seconds=times,
        notes=codec.notes,
    )
    del records
    gc.collect()
    return result, per_dataset


def skipped_result(codec: CodecAdapter, raw_bytes: int, exc: Exception) -> CodecResult:
    return CodecResult(
        status="skipped",
        algorithm=codec.name,
        family=codec.family,
        raw_bytes=raw_bytes,
        compressed_bytes=None,
        compressed_over_raw=None,
        raw_over_compressed=None,
        compress_seconds=None,
        compress_mib_s=None,
        decode_best_seconds=None,
        decode_median_seconds=None,
        decode_mean_seconds=None,
        decode_std_seconds=None,
        decode_best_mib_s=None,
        decode_median_mib_s=None,
        decode_best_gib_s=None,
        decode_runs_seconds=[],
        notes=f"{type(exc).__name__}: {exc}",
    )


def result_sort_key(sort_by: SortBy, index: dict[str, int]) -> Callable[[dict[str, Any]], Any]:
    if sort_by == "decode":
        return lambda row: (
            row["decode_best_mib_s"] is None,
            -(row["decode_best_mib_s"] or 0.0),
            index[row["algorithm"]],
        )
    if sort_by == "ratio":
        return lambda row: (
            row["compressed_over_raw"] is None,
            row["compressed_over_raw"] if row["compressed_over_raw"] is not None else float("inf"),
            index[row["algorithm"]],
        )
    if sort_by == "compress":
        return lambda row: (
            row["compress_mib_s"] is None,
            -(row["compress_mib_s"] or 0.0),
            index[row["algorithm"]],
        )
    if sort_by == "name":
        return lambda row: row["algorithm"]
    return lambda row: index[row["algorithm"]]


def summary_csv_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "status",
        "algorithm",
        "family",
        "raw_bytes",
        "compressed_bytes",
        "compressed_over_raw",
        "raw_over_compressed",
        "compress_seconds",
        "compress_mib_s",
        "decode_best_seconds",
        "decode_median_seconds",
        "decode_mean_seconds",
        "decode_std_seconds",
        "decode_best_mib_s",
        "decode_median_mib_s",
        "decode_best_gib_s",
        "notes",
    ]
    return [{field: row.get(field) for field in fields} for row in results]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_markdown(payload: dict[str, Any]) -> str:
    meta = payload["metadata"]
    lines = [
        "# Compression benchmark",
        "",
        f"- input: `{meta['input_path']}`",
        f"- sampled raw bytes: {human_bytes(meta['sampled_raw_bytes'])}",
        f"- codec profile: `{meta['profile']}`",
        f"- timed decode repeats: {meta['repeats']}; warmups: {meta['warmups']}",
        f"- decode order: `{meta['decode_order']}`; verify: `{meta['verify']}`",
        f"- host: `{meta['host']}`",
        f"- python: `{meta['python']}`",
        f"- zarr: `{meta['versions'].get('zarr')}`; "
        f"numcodecs: `{meta['versions'].get('numcodecs')}`",
        "",
        "## Sampled datasets",
        "",
        "| dataset | dtype | file raw bytes | sampled bytes | chunks | HDF5 compression | HDF5 read MiB/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in meta["datasets"]:
        read_mib_s = None
        if item["read_seconds"]:
            read_mib_s = item["sampled_bytes"] / item["read_seconds"] / (1 << 20)
        lines.append(
            f"| `{item['name']}` | `{item['dtype']}` | {human_bytes(item['nbytes'])} | "
            f"{human_bytes(item['sampled_bytes'])} | {item['sampled_chunks']} | "
            f"`{item['compression']}` | {fmt_float(read_mib_s, 1)} |"
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| algorithm | family | compressed/raw | raw/compressed | compress MiB/s | "
            "decode best MiB/s | decode median MiB/s | notes |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["results"]:
        if row["status"] != "ok":
            lines.append(
                f"| `{row['algorithm']}` | `{row['family']}` | skipped | skipped | "
                f"n/a | n/a | n/a | {row['notes']} |"
            )
            continue
        lines.append(
            f"| `{row['algorithm']}` | `{row['family']}` | "
            f"{fmt_float(row['compressed_over_raw'])} | "
            f"{fmt_float(row['raw_over_compressed'])} | "
            f"{fmt_float(row['compress_mib_s'], 1)} | "
            f"{fmt_float(row['decode_best_mib_s'], 1)} | "
            f"{fmt_float(row['decode_median_mib_s'], 1)} | {row['notes']} |"
        )
    lines.append("")
    return "\n".join(lines)


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    sample_bytes: int
    block_bytes: int
    profile: Profile


def size_slug(value: int) -> str:
    units = (
        ("TiB", 1 << 40),
        ("GiB", 1 << 30),
        ("MiB", 1 << 20),
        ("KiB", 1 << 10),
    )
    for suffix, scale in units:
        if value >= scale and value % scale == 0:
            return f"{value // scale}{suffix}"
    return f"{value}B"


def parse_size_list(values: Iterable[str] | None, fallback: int) -> list[int]:
    items = parse_csv_list(values)
    if not items:
        return [fallback]
    return [parse_size(item) for item in items]


def parse_profile_list(values: Iterable[str] | None, fallback: Profile) -> list[Profile]:
    valid = {"quick", "default", "broad", "all"}
    items = parse_csv_list(values)
    if not items:
        return [fallback]
    invalid = sorted(set(items) - valid)
    if invalid:
        raise ValueError(f"invalid profile values: {invalid}")
    return [item for item in items]  # type: ignore[return-value]


def build_run_matrix(
    *,
    sample_bytes_values: list[int],
    block_bytes_values: list[int],
    profile_values: list[Profile],
) -> list[RunConfig]:
    runs: list[RunConfig] = []
    for sample_bytes, block_bytes, profile in iter_product(
        sample_bytes_values,
        block_bytes_values,
        profile_values,
    ):
        run_id = (
            f"sample-{size_slug(sample_bytes)}_"
            f"block-{size_slug(block_bytes)}_"
            f"profile-{profile}"
        )
        runs.append(
            RunConfig(
                run_id=run_id,
                sample_bytes=sample_bytes,
                block_bytes=block_bytes,
                profile=profile,
            )
        )
    return runs


def output_paths(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "json": output_dir / "result.json",
        "summary_csv": output_dir / "summary.csv",
        "by_dataset_csv": output_dir / "by_dataset.csv",
        "md": output_dir / "report.md",
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> dict[str, Path]:
    paths = output_paths(output_dir)
    paths["json"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(paths["summary_csv"], summary_csv_rows(payload["results"]))
    write_csv(paths["by_dataset_csv"], payload["per_dataset_results"])
    paths["md"].write_text(format_markdown(payload), encoding="utf-8")
    return paths


def matrix_paths(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "matrix_json": output_dir / "matrix.json",
        "matrix_summary_csv": output_dir / "matrix_summary.csv",
        "matrix_runs_csv": output_dir / "matrix_runs.csv",
        "matrix_md": output_dir / "matrix_report.md",
    }


def write_matrix_outputs(
    output_dir: Path,
    *,
    matrix_payload: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    paths = matrix_paths(output_dir)
    paths["matrix_json"].write_text(
        json.dumps(matrix_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(paths["matrix_summary_csv"], summary_rows)
    write_csv(paths["matrix_runs_csv"], run_rows)
    paths["matrix_md"].write_text(format_matrix_markdown(matrix_payload), encoding="utf-8")
    return paths


def format_matrix_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Compression benchmark matrix",
        "",
        f"- input: `{payload['input_path']}`",
        f"- runs: {len(payload['runs'])}",
        f"- output dir: `{payload['output_dir']}`",
        "",
        "## Runs",
        "",
        "| run | sample bytes | block bytes | profile | output |",
        "|---|---:|---:|---:|---|",
    ]
    for run in payload["runs"]:
        lines.append(
            f"| `{run['run_id']}` | {human_bytes(run['sample_bytes'])} | "
            f"{human_bytes(run['block_bytes'])} | `{run['profile']}` | "
            f"`{run['output_dir']}` |"
        )
    lines.extend(
        [
            "",
            "## Best Decode Per Run",
            "",
            "| run | algorithm | compressed/raw | raw/compressed | decode MiB/s | report |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for run in payload["runs"]:
        best = run.get("best_decode")
        if not best:
            lines.append(f"| `{run['run_id']}` | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| `{run['run_id']}` | `{best['algorithm']}` | "
            f"{fmt_float(best['compressed_over_raw'])} | "
            f"{fmt_float(best['raw_over_compressed'])} | "
            f"{fmt_float(best['decode_best_mib_s'], 1)} | "
            f"`{run['output_dir']}/report.md` |"
        )
    lines.append("")
    return "\n".join(lines)


def print_dataset_table(infos: list[DatasetInfo]) -> None:
    print("numeric datasets:")
    for info in infos:
        print(
            f"{info.name}\tshape={info.shape}\tdtype={info.dtype}\t"
            f"raw={human_bytes(info.nbytes)}\tchunks={info.chunks}\t"
            f"compression={info.compression}"
        )


def print_codec_table(codecs: list[CodecAdapter]) -> None:
    print("codecs:")
    for codec in codecs:
        print(f"{codec.name}\tfamily={codec.family}\t{codec.notes}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark compression ratio and in-memory decode throughput for "
            "H5AD/HDF5 numeric chunks."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-bytes", type=parse_size, default=parse_size("512MiB"))
    parser.add_argument(
        "--sample-bytes-list",
        action="append",
        help="Comma-separated scan values, e.g. 128MiB,512MiB,1GiB.",
    )
    parser.add_argument("--block-bytes", type=parse_size, default=parse_size("8MiB"))
    parser.add_argument(
        "--block-bytes-list",
        action="append",
        help="Comma-separated scan values, e.g. 2MiB,8MiB,32MiB.",
    )
    parser.add_argument("--min-sample-per-dataset", type=parse_size, default=parse_size("4MiB"))
    parser.add_argument("--min-dataset-bytes", type=parse_size, default=parse_size("1MiB"))
    parser.add_argument("--max-datasets", type=int, default=12)
    parser.add_argument("--selection", choices=["largest", "stratified", "all"], default="stratified")
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset glob to include, e.g. 'X/*'. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        help="Dataset glob to exclude. Can be repeated or comma-separated.",
    )
    parser.add_argument("--profile", choices=["quick", "default", "broad", "all"], default="default")
    parser.add_argument(
        "--profile-list",
        action="append",
        help="Comma-separated scan values, e.g. quick,default,broad.",
    )
    parser.add_argument(
        "--blosc-shuffle",
        action="append",
        help=(
            "Blosc shuffle mode: auto,noshuffle,shuffle,bitshuffle. "
            "Can be repeated or comma-separated."
        ),
    )
    parser.add_argument(
        "--only-codec",
        action="append",
        help="Codec glob to include, e.g. 'blosc.*' or 'zstd.*'.",
    )
    parser.add_argument(
        "--exclude-codec",
        action="append",
        help="Codec glob to exclude.",
    )
    parser.add_argument("--include-optional", action="store_true", help="Try optional PCodec/ZFPY codecs.")
    parser.add_argument("--skip-slow", action="store_true", help="Skip BZ2/LZMA families.")
    parser.add_argument("--no-baseline", action="store_true", help="Drop uncompressed copy baseline.")
    parser.add_argument("--threads", type=int, default=1, help="Blosc threads. Default: 1.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--verify", choices=["none", "first", "all"], default="all")
    parser.add_argument("--decode-order", choices=["sequential", "random"], default="sequential")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sort", choices=["decode", "ratio", "compress", "name", "input"], default="decode")
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--list-codecs", action="store_true")
    parser.add_argument("--list-runs", action="store_true", help="Print the expanded scan matrix and exit.")
    return parser


def build_selected_codecs(
    args: argparse.Namespace,
    *,
    profile: Profile,
    blosc_shuffles: list[str],
    only_codecs: list[str],
    exclude_codecs: list[str],
) -> list[CodecAdapter]:
    codecs = build_codecs(
        profile=profile,
        include_baseline=not args.no_baseline,
        include_optional=args.include_optional,
        skip_slow=args.skip_slow,
        blosc_shuffles=blosc_shuffles,
    )
    return filter_codecs(
        codecs,
        only_patterns=only_codecs,
        exclude_patterns=exclude_codecs,
    )


def run_benchmark_once(
    *,
    args: argparse.Namespace,
    run: RunConfig,
    selected: list[DatasetInfo],
    output_dir: Path,
    blosc_shuffles: list[str],
    only_codecs: list[str],
    exclude_codecs: list[str],
) -> tuple[dict[str, Any], dict[str, Path]]:
    codecs = build_selected_codecs(
        args,
        profile=run.profile,
        blosc_shuffles=blosc_shuffles,
        only_codecs=only_codecs,
        exclude_codecs=exclude_codecs,
    )
    if not codecs:
        raise SystemExit("No codecs selected.")

    print(f"=== run {run.run_id} ===", flush=True)
    print("Selected datasets:", flush=True)
    for info in selected:
        print(
            f"  {info.name}: dtype={info.dtype}, raw={human_bytes(info.nbytes)}, "
            f"chunks={info.chunks}, hdf5_compression={info.compression}",
            flush=True,
        )
    print(f"Selected {len(codecs)} codecs. Use --list-codecs to inspect the set.", flush=True)
    print(
        f"Reading sample budget {human_bytes(run.sample_bytes)} "
        f"with block size {human_bytes(run.block_bytes)} from {args.input}",
        flush=True,
    )

    chunks, sample_stats = read_samples(
        args.input,
        selected,
        sample_bytes=run.sample_bytes,
        block_bytes=run.block_bytes,
        min_sample_per_dataset=args.min_sample_per_dataset,
        seed=args.seed,
    )
    sampled_raw_bytes = sum(chunk.nbytes for chunk in chunks)
    print(f"Sampled {human_bytes(sampled_raw_bytes)} in {len(chunks)} chunks", flush=True)

    results: list[dict[str, Any]] = []
    per_dataset_results: list[dict[str, Any]] = []
    run_order: dict[str, int] = {}
    for idx, codec in enumerate(codecs):
        run_order[codec.name] = idx
        print(f"Benchmarking {codec.name} ...", flush=True)
        started = time.perf_counter()
        try:
            result, per_dataset = benchmark_codec(
                codec,
                chunks,
                repeats=args.repeats,
                warmups=args.warmups,
                verify=args.verify,
                decode_order=args.decode_order,
                seed=args.seed + idx * 1000,
            )
        except Exception as exc:
            result = skipped_result(codec, sampled_raw_bytes, exc)
            per_dataset = []
        elapsed = time.perf_counter() - started
        results.append(result.to_dict())
        for row in per_dataset:
            row["run_id"] = run.run_id
            row["sample_bytes"] = run.sample_bytes
            row["block_bytes"] = run.block_bytes
            row["profile"] = run.profile
            row["algorithm"] = codec.name
            row["family"] = codec.family
            per_dataset_results.append(row)
        print(
            f"  status={result.status} compressed/raw={fmt_float(result.compressed_over_raw)} "
            f"decode_best_mib_s={fmt_float(result.decode_best_mib_s, 1)} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    sort_key = result_sort_key(args.sort, run_order)
    sorted_results = sorted(results, key=sort_key)

    dataset_meta: list[dict[str, Any]] = []
    for info in selected:
        stats = sample_stats[info.name]
        dataset_meta.append(
            {
                "name": info.name,
                "shape": info.shape,
                "dtype": info.dtype,
                "dtype_kind": info.dtype_kind,
                "itemsize": info.itemsize,
                "nbytes": info.nbytes,
                "chunks": info.chunks,
                "compression": info.compression,
                "sampled_bytes": stats.sampled_bytes,
                "sampled_chunks": stats.sampled_chunks,
                "read_seconds": stats.read_seconds,
            }
        )

    metadata_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run.run_id,
        "output_dir": str(output_dir),
        "input_path": str(args.input),
        "host": platform.node(),
        "python": sys.version.replace("\n", " "),
        "command": [Path(sys.argv[0]).name, *(sys.argv[1:])],
        "versions": {
            "zarr": version_of("zarr") or getattr(zarr, "__version__", None),
            "numcodecs": version_of("numcodecs") or getattr(numcodecs, "__version__", None),
            "h5py": version_of("h5py"),
            "numpy": version_of("numpy"),
        },
        "profile": run.profile,
        "threads": args.threads,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "verify": args.verify,
        "decode_order": args.decode_order,
        "sample_bytes_budget": run.sample_bytes,
        "sampled_raw_bytes": sampled_raw_bytes,
        "block_bytes": run.block_bytes,
        "min_sample_per_dataset": args.min_sample_per_dataset,
        "selection": args.selection,
        "sort": args.sort,
        "ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "single_thread_env": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "BLOSC_NTHREADS",
            )
        },
        "datasets": dataset_meta,
    }
    payload = {
        "metadata": metadata_payload,
        "results": sorted_results,
        "results_run_order": results,
        "per_dataset_results": per_dataset_results,
    }
    paths = write_outputs(output_dir, payload)
    print(format_markdown(payload), flush=True)
    print("Wrote:", flush=True)
    for name, path in paths.items():
        print(f"  {name}: {path}", flush=True)
    return payload, paths


def best_decode_row(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    ok_rows = [row for row in results if row.get("decode_best_mib_s") is not None]
    if not ok_rows:
        return None
    return max(ok_rows, key=lambda row: row["decode_best_mib_s"])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    include_patterns = parse_csv_list(args.dataset)
    exclude_patterns = parse_csv_list(args.exclude_dataset)
    only_codecs = parse_csv_list(args.only_codec)
    exclude_codecs = parse_csv_list(args.exclude_codec)
    blosc_shuffles = parse_csv_list(args.blosc_shuffle) or ["auto"]
    sample_bytes_values = parse_size_list(args.sample_bytes_list, args.sample_bytes)
    block_bytes_values = parse_size_list(args.block_bytes_list, args.block_bytes)
    try:
        profile_values = parse_profile_list(args.profile_list, args.profile)
    except ValueError as exc:
        parser.error(str(exc))

    valid_shuffles = {"auto", "noshuffle", "shuffle", "bitshuffle"}
    invalid_shuffles = set(blosc_shuffles) - valid_shuffles
    if invalid_shuffles:
        parser.error(f"invalid --blosc-shuffle values: {sorted(invalid_shuffles)}")

    runs = build_run_matrix(
        sample_bytes_values=sample_bytes_values,
        block_bytes_values=block_bytes_values,
        profile_values=profile_values,
    )
    if args.list_runs:
        print("runs:")
        for run in runs:
            print(
                f"{run.run_id}\tsample={human_bytes(run.sample_bytes)}\t"
                f"block={human_bytes(run.block_bytes)}\tprofile={run.profile}"
            )
        return 0

    configure_threads(args.threads)

    if args.list_codecs:
        for profile in profile_values:
            print(f"profile={profile}")
            codecs = build_selected_codecs(
                args,
                profile=profile,
                blosc_shuffles=blosc_shuffles,
                only_codecs=only_codecs,
                exclude_codecs=exclude_codecs,
            )
            print_codec_table(codecs)
        return 0

    infos = collect_dataset_infos(args.input)
    if args.list_datasets:
        print_dataset_table(infos)
        return 0

    selected = select_datasets(
        infos,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        min_dataset_bytes=args.min_dataset_bytes,
        max_datasets=args.max_datasets,
        selection=args.selection,
    )
    if not selected:
        raise SystemExit("No numeric datasets selected.")

    matrix_summary_rows: list[dict[str, Any]] = []
    matrix_run_rows: list[dict[str, Any]] = []
    matrix_runs: list[dict[str, Any]] = []
    for run in runs:
        run_output_dir = args.output_dir if len(runs) == 1 else args.output_dir / "runs" / run.run_id
        payload, paths = run_benchmark_once(
            args=args,
            run=run,
            selected=selected,
            output_dir=run_output_dir,
            blosc_shuffles=blosc_shuffles,
            only_codecs=only_codecs,
            exclude_codecs=exclude_codecs,
        )

        for row in payload["results"]:
            matrix_row = {
                "run_id": run.run_id,
                "sample_bytes": run.sample_bytes,
                "block_bytes": run.block_bytes,
                "profile": run.profile,
                "output_dir": str(run_output_dir),
            }
            matrix_row.update(row)
            matrix_summary_rows.append(matrix_row)

        best = best_decode_row(payload["results"])
        matrix_run_rows.append(
            {
                "run_id": run.run_id,
                "sample_bytes": run.sample_bytes,
                "block_bytes": run.block_bytes,
                "profile": run.profile,
                "output_dir": str(run_output_dir),
                "sampled_raw_bytes": payload["metadata"]["sampled_raw_bytes"],
                "best_decode_algorithm": best["algorithm"] if best else None,
                "best_decode_mib_s": best["decode_best_mib_s"] if best else None,
                "best_compressed_over_raw": best["compressed_over_raw"] if best else None,
                "best_raw_over_compressed": best["raw_over_compressed"] if best else None,
            }
        )
        matrix_runs.append(
            {
                "run_id": run.run_id,
                "sample_bytes": run.sample_bytes,
                "block_bytes": run.block_bytes,
                "profile": run.profile,
                "output_dir": str(run_output_dir),
                "paths": {name: str(path) for name, path in paths.items()},
                "best_decode": best,
            }
        )

    if len(runs) > 1:
        matrix_payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_path": str(args.input),
            "output_dir": str(args.output_dir),
            "command": [Path(sys.argv[0]).name, *(sys.argv[1:])],
            "runs": matrix_runs,
            "summary_rows": matrix_summary_rows,
            "run_rows": matrix_run_rows,
        }
        paths = write_matrix_outputs(
            args.output_dir,
            matrix_payload=matrix_payload,
            summary_rows=matrix_summary_rows,
            run_rows=matrix_run_rows,
        )
        print("Wrote matrix outputs:", flush=True)
        for name, path in paths.items():
            print(f"  {name}: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
