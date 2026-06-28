#!/usr/bin/env python3
"""Export numcodecs-encoded chunks for Rust decode benchmarks."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import bench_compression as bench


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def codec_config(name: str) -> dict[str, Any] | None:
    if name == "none.copy":
        return {"id": "none"}
    if match := re.fullmatch(r"zstd\.level(-?\d+)", name):
        return {"id": "zstd", "level": int(match.group(1)), "checksum": False}
    if match := re.fullmatch(r"gzip\.level(\d+)", name):
        return {"id": "gzip", "level": int(match.group(1))}
    if match := re.fullmatch(r"zlib\.level(\d+)", name):
        return {"id": "zlib", "level": int(match.group(1))}
    if match := re.fullmatch(r"lz4\.accel(\d+)", name):
        return {"id": "lz4", "acceleration": int(match.group(1))}
    if match := re.fullmatch(r"bz2\.level(\d+)", name):
        return {"id": "bz2", "level": int(match.group(1))}
    if name == "lzma.default":
        return {"id": "lzma", "format": 1, "check": -1, "preset": None, "filters": None}
    if match := re.fullmatch(r"lzma\.preset(\d+)", name):
        return {
            "id": "lzma",
            "format": 1,
            "check": -1,
            "preset": int(match.group(1)),
            "filters": None,
        }
    if match := re.fullmatch(r"blosc\.([^.]+)\.clevel(\d+)(?:\.([^.]+))?", name):
        shuffle_name = match.group(3) or "shuffle"
        shuffle = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}.get(shuffle_name, 1)
        return {
            "id": "blosc",
            "cname": match.group(1),
            "clevel": int(match.group(2)),
            "shuffle": shuffle,
            "typesize": 1,
            "blocksize": 0,
        }
    return None


def write_raw_chunks(path: Path, chunks: list[bench.SampleChunk]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    with path.open("wb") as fh:
        for index, chunk in enumerate(chunks):
            fh.write(chunk.raw)
            rows.append(
                {
                    "index": index,
                    "dataset": chunk.dataset,
                    "dtype": chunk.dtype,
                    "itemsize": chunk.itemsize,
                    "offset": chunk.offset,
                    "shape": chunk.shape,
                    "raw_offset": offset,
                    "raw_len": chunk.nbytes,
                }
            )
            offset += chunk.nbytes
    return rows


def encode_codec(
    codec: bench.CodecAdapter,
    chunks: list[bench.SampleChunk],
    encoded_path: Path,
    verify: bench.VerifyMode,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[bench.EncodedChunk] = []
    started = time.perf_counter()
    for chunk in chunks:
        records.append(bench.EncodedChunk(sample=chunk, payload=codec.encode(chunk)))
    compress_seconds = time.perf_counter() - started

    bench.verify_decoded(codec, records, verify)

    encoded_rows: list[dict[str, Any]] = []
    encoded_offset = 0
    with encoded_path.open("wb") as fh:
        for index, record in enumerate(records):
            fh.write(record.payload)
            encoded_rows.append(
                {
                    "chunk_index": index,
                    "payload_offset": encoded_offset,
                    "payload_len": len(record.payload),
                }
            )
            encoded_offset += len(record.payload)

    raw_bytes = sum(chunk.nbytes for chunk in chunks)
    compressed_bytes = encoded_offset
    meta = {
        "status": "ok",
        "algorithm": codec.name,
        "family": codec.family,
        "codec_config": codec_config(codec.name),
        "encoded_file": str(encoded_path.resolve()),
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
        "compressed_over_raw": compressed_bytes / raw_bytes if raw_bytes else None,
        "raw_over_compressed": raw_bytes / compressed_bytes if compressed_bytes else None,
        "compress_seconds": compress_seconds,
        "compress_mib_s": raw_bytes / compress_seconds / (1 << 20) if compress_seconds else None,
        "notes": codec.notes,
        "records": encoded_rows,
    }
    return meta, encoded_rows


def skipped_algorithm(codec: bench.CodecAdapter, raw_bytes: int, exc: Exception) -> dict[str, Any]:
    return {
        "status": "skipped",
        "algorithm": codec.name,
        "family": codec.family,
        "codec_config": codec_config(codec.name),
        "encoded_file": None,
        "raw_bytes": raw_bytes,
        "compressed_bytes": None,
        "compressed_over_raw": None,
        "raw_over_compressed": None,
        "compress_seconds": None,
        "compress_mib_s": None,
        "notes": f"{type(exc).__name__}: {exc}",
        "records": [],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=bench.DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-bytes", type=bench.parse_size, default=bench.parse_size("512MiB"))
    parser.add_argument("--sample-bytes-list", action="append")
    parser.add_argument("--block-bytes", type=bench.parse_size, default=bench.parse_size("8MiB"))
    parser.add_argument("--block-bytes-list", action="append")
    parser.add_argument(
        "--min-sample-per-dataset", type=bench.parse_size, default=bench.parse_size("4MiB")
    )
    parser.add_argument(
        "--min-dataset-bytes", type=bench.parse_size, default=bench.parse_size("1MiB")
    )
    parser.add_argument("--max-datasets", type=int, default=12)
    parser.add_argument(
        "--selection", choices=["largest", "stratified", "all"], default="stratified"
    )
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--exclude-dataset", action="append")
    parser.add_argument(
        "--profile", choices=["quick", "default", "broad", "all"], default="default"
    )
    parser.add_argument("--profile-list", action="append")
    parser.add_argument("--blosc-shuffle", action="append")
    parser.add_argument("--only-codec", action="append")
    parser.add_argument("--exclude-codec", action="append")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--skip-slow", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--verify", choices=["none", "first", "all"], default="all")
    parser.add_argument("--seed", type=int, default=17)
    return parser


def export_run(
    *,
    args: argparse.Namespace,
    run: bench.RunConfig,
    selected: list[bench.DatasetInfo],
    output_dir: Path,
    blosc_shuffles: list[str],
    only_codecs: list[str],
    exclude_codecs: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded_dir = output_dir / "encoded"
    encoded_dir.mkdir(parents=True, exist_ok=True)

    chunks, sample_stats = bench.read_samples(
        args.input,
        selected,
        sample_bytes=run.sample_bytes,
        block_bytes=run.block_bytes,
        min_sample_per_dataset=args.min_sample_per_dataset,
        seed=args.seed,
    )
    raw_file = output_dir / "raw.bin"
    chunk_rows = write_raw_chunks(raw_file, chunks)
    raw_bytes = sum(chunk.nbytes for chunk in chunks)

    codecs = bench.build_selected_codecs(
        args,
        profile=run.profile,
        blosc_shuffles=blosc_shuffles,
        only_codecs=only_codecs,
        exclude_codecs=exclude_codecs,
    )

    algorithms: list[dict[str, Any]] = []
    print(
        f"exporting {run.run_id}: {bench.human_bytes(raw_bytes)} in {len(chunks)} chunks",
        flush=True,
    )
    for codec in codecs:
        print(f"  encoding {codec.name} ...", flush=True)
        encoded_path = encoded_dir / f"{safe_name(codec.name)}.bin"
        try:
            if codec_config(codec.name) is None:
                raise RuntimeError("unsupported by Rust codec benchmark")
            algorithm, _records = encode_codec(codec, chunks, encoded_path, args.verify)
        except Exception as exc:
            if encoded_path.exists():
                encoded_path.unlink()
            algorithm = skipped_algorithm(codec, raw_bytes, exc)
        algorithms.append(algorithm)
        print(
            f"    status={algorithm['status']} compressed/raw="
            f"{bench.fmt_float(algorithm['compressed_over_raw'])}",
            flush=True,
        )

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

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run.run_id,
        "input_path": str(args.input),
        "output_dir": str(output_dir.resolve()),
        "sample_bytes": run.sample_bytes,
        "block_bytes": run.block_bytes,
        "profile": run.profile,
        "verify": args.verify,
        "raw_file": str(raw_file.resolve()),
        "raw_bytes": raw_bytes,
        "chunks": chunk_rows,
        "datasets": dataset_meta,
        "algorithms": algorithms,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "run_id": run.run_id,
        "sample_bytes": run.sample_bytes,
        "block_bytes": run.block_bytes,
        "profile": run.profile,
        "manifest": str(manifest_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "raw_bytes": raw_bytes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    include_patterns = bench.parse_csv_list(args.dataset)
    exclude_patterns = bench.parse_csv_list(args.exclude_dataset)
    only_codecs = bench.parse_csv_list(args.only_codec)
    exclude_codecs = bench.parse_csv_list(args.exclude_codec)
    blosc_shuffles = bench.parse_csv_list(args.blosc_shuffle) or ["auto"]
    sample_bytes_values = bench.parse_size_list(args.sample_bytes_list, args.sample_bytes)
    block_bytes_values = bench.parse_size_list(args.block_bytes_list, args.block_bytes)
    try:
        profile_values = bench.parse_profile_list(args.profile_list, args.profile)
    except ValueError as exc:
        parser.error(str(exc))

    bench.configure_threads(args.threads)

    infos = bench.collect_dataset_infos(args.input)
    selected = bench.select_datasets(
        infos,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        min_dataset_bytes=args.min_dataset_bytes,
        max_datasets=args.max_datasets,
        selection=args.selection,
    )
    if not selected:
        raise SystemExit("No numeric datasets selected.")

    runs = bench.build_run_matrix(
        sample_bytes_values=sample_bytes_values,
        block_bytes_values=block_bytes_values,
        profile_values=profile_values,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_rows = []
    for run in runs:
        run_dir = args.output_dir if len(runs) == 1 else args.output_dir / "runs" / run.run_id
        run_rows.append(
            export_run(
                args=args,
                run=run,
                selected=selected,
                output_dir=run_dir,
                blosc_shuffles=blosc_shuffles,
                only_codecs=only_codecs,
                exclude_codecs=exclude_codecs,
            )
        )

    matrix = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(args.input),
        "output_dir": str(args.output_dir.resolve()),
        "runs": run_rows,
    }
    matrix_path = args.output_dir / "matrix_manifest.json"
    matrix_path.write_text(json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {matrix_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
