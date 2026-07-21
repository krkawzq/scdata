#!/usr/bin/env python3
"""Streamly rewrite the Blosc ``blocksize`` of a scdata ``.zarr.zip`` store.

This is a one-off conversion tool.  Older scdata stores were written with
``blocksize=0`` (Blosc auto-selects, typically 128-256 KiB); the native
Blosc-LZ4 fast path prefers an explicit ``64 KiB`` block size to bound random
read amplification.  Re-running the full h5ad -> zarr pipeline would also work,
but it re-decodes the source h5ad (slow gzip) and rewrites all metadata.  This
script instead re-compresses only the Blosc-compressed chunks in place:

    old chunk (blosc, blocksize=0)
      -> numcodecs.Blosc(...).decode  (LZ4 decode, ~GB/s)
      -> numcodecs.Blosc(..., blocksize=N).encode  (LZ4 re-encode)
      -> new chunk (blosc, blocksize=N)

The decoded byte stream is untouched, so the array's dtype / shape / chunk grid
/ fill value / CSR layout are all preserved verbatim.  Only the Blosc
*configuration* in each ``zarr.json`` (the ``blocksize`` field) and the Blosc
chunk bytes change.  Every other store entry (``obs`` / ``var`` zstd arrays,
``uns``, group ``zarr.json`` nodes, string arrays) is copied byte-for-byte.

Correctness contract
--------------------
For a supported scdata v3 numeric pipeline (``bytes`` serializer followed by one
``blosc`` codec), output is byte-identical to what :func:`scdata.io.write_zarr`
would produce with the same Blosc parameters and target ``blocksize`` —
``numcodecs.Blosc.encode`` is deterministic given (decoded bytes, cname, clevel,
shuffle, typesize, blocksize), and we copy those five parameters straight from
the source ``zarr.json``.  This equivalence has been verified empirically
against ``write_zarr`` output for both ``typesize=2`` and ``typesize=4`` chunks.

The script never touches:
  * the ``bytes`` / ``vlen-utf8`` ArrayBytes serializer entry (endian/layout),
  * ``zstd`` / ``gzip`` / ``lz4`` (non-blosc) compressors,
  * group ``zarr.json`` nodes (no ``codecs`` field),
  * any ``zarr.json`` field other than the blosc ``blocksize``.

Usage
-----
    uv run python scripts/recompress_zarrzip_blocksize.py INPUT.zarr.zip \\
        --blocksize 65536 [--output OUTPUT.zarr.zip] [--overwrite] \\
        [--verify]

With ``--verify`` the converted store is opened with :func:`scdata.io.launch`
and every re-compressed chunk is decoded back and compared byte-for-byte against
the source chunk's decode, so a silent corruption would fail the run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Blosc encode/decode is CPU-bound and releases the GIL, but a single store is
# usually large enough that process-level parallelism over *stores* (run this
# script once per file) is the right granularity.  Within one store we stream
# chunks sequentially to keep memory bounded.

# ---------------------------------------------------------------------------
# Blosc helpers
# ---------------------------------------------------------------------------

_BLOSC_SHUFFLE_NAME_TO_INT = {
    "0": 0, "none": 0, "noshuffle": 0, "no_shuffle": 0,
    "1": 1, "shuffle": 1, "byte": 1,
    "2": 2, "bitshuffle": 2, "bit_shuffle": 2,
}


def _blosc_shuffle_int(value: Any) -> int:
    """Map a zarr v3 blosc ``shuffle`` value to the numcodecs int constant.

    Mirrors ``scdata.io._anndata._blosc_shuffle_int`` /
    ``scdata.io._launch._v3_blosc_shuffle`` so the script stays consistent with
    both the writer and the reader.
    """
    if value is None:
        return 1
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _BLOSC_SHUFFLE_NAME_TO_INT:
            return _BLOSC_SHUFFLE_NAME_TO_INT[key]
        raise ValueError(f"unsupported blosc shuffle value: {value!r}")
    parsed = int(value)
    if parsed not in (0, 1, 2):
        raise ValueError(f"unsupported blosc shuffle value: {value!r}")
    return parsed


@dataclass(frozen=True)
class BloscParams:
    """The five Blosc parameters that fully determine a chunk's encoding."""

    cname: str
    clevel: int
    shuffle: int
    typesize: int
    # blocksize is intentionally NOT part of identity: it is the *target* we
    # re-encode with, and Blosc may adjust it internally (e.g. 65536 requested
    # -> 131072 in the chunk header for typesize=2).  scdata's own write_zarr
    # has the same behavior: zarr.json says 65536, header says 131072.

    @classmethod
    def from_codec_config(cls, cfg: dict[str, Any]) -> "BloscParams":
        return cls(
            cname=str(cfg.get("cname", "lz4")),
            clevel=int(cfg.get("clevel", 5)),
            shuffle=_blosc_shuffle_int(cfg.get("shuffle", 1)),
            typesize=int(cfg.get("typesize", 1)),
        )

    def decoder(self) -> Any:
        # blocksize=0 on decode means "read blocksize from the chunk header" —
        # the decoder must not impose a blocksize, it must honor whatever the
        # source chunk encoded with.
        from numcodecs import Blosc
        return Blosc(
            cname=self.cname,
            clevel=self.clevel,
            shuffle=self.shuffle,
            blocksize=0,
            typesize=self.typesize,
        )

    def encoder(self, blocksize: int) -> Any:
        from numcodecs import Blosc
        return Blosc(
            cname=self.cname,
            clevel=self.clevel,
            shuffle=self.shuffle,
            blocksize=int(blocksize),
            typesize=self.typesize,
        )


# ---------------------------------------------------------------------------
# zarr.json inspection
# ---------------------------------------------------------------------------


def _unsupported_pipeline(array_path: str, detail: str) -> ValueError:
    return ValueError(
        f"{array_path}: unsupported Blosc codec pipeline ({detail}); expected "
        "[bytes serializer, single blosc compressor] from a scdata zarr v3 store"
    )


def _recompressible_blosc_codec(
    meta: dict[str, Any], array_path: str
) -> tuple[int, dict[str, Any]] | None:
    """Return the sole Blosc codec only for the scdata v3 pipeline we can prove.

    Non-Blosc arrays are copied without interpretation.  A Blosc array is safe
    to rewrite only when it is the numeric scdata form emitted by ``write_zarr``:
    ``bytes`` (little-endian or omitted config for one-byte dtypes) followed by
    exactly one ``blosc`` compressor.  In particular, do not mistake a Blosc
    nested in sharding or an additional codec for raw bytes that numcodecs can
    re-encode independently.
    """
    codecs = meta.get("codecs")
    if not isinstance(codecs, list):
        return None

    names = [entry.get("name") if isinstance(entry, dict) else None for entry in codecs]
    if any(name in {"sharding_indexed", "sharding"} for name in names):
        raise _unsupported_pipeline(array_path, "sharding codec")
    if "blosc" not in names:
        return None
    if meta.get("zarr_format") != 3:
        raise _unsupported_pipeline(array_path, "zarr_format is not 3")
    chunk_key_encoding = meta.get("chunk_key_encoding")
    if (
        not isinstance(chunk_key_encoding, dict)
        or chunk_key_encoding.get("name") != "default"
        or not isinstance(chunk_key_encoding.get("configuration"), dict)
        or chunk_key_encoding["configuration"].get("separator") != "/"
    ):
        raise _unsupported_pipeline(array_path, "non-default chunk key encoding")
    if len(codecs) != 2 or names != ["bytes", "blosc"]:
        raise _unsupported_pipeline(array_path, f"codec names are {names!r}")

    serializer = codecs[0]
    blosc = codecs[1]
    assert isinstance(serializer, dict) and isinstance(blosc, dict)  # implied by names
    serializer_cfg = serializer.get("configuration", {})
    if serializer_cfg is None:
        serializer_cfg = {}
    if (
        not isinstance(serializer_cfg, dict)
        or set(serializer_cfg) - {"endian"}
        or serializer_cfg.get("endian", "little") != "little"
    ):
        raise _unsupported_pipeline(array_path, "non-scdata bytes serializer configuration")

    cfg = blosc.get("configuration")
    if not isinstance(cfg, dict):
        raise _unsupported_pipeline(array_path, "blosc configuration is not an object")
    return 1, cfg


def _is_array_meta(meta: dict[str, Any]) -> bool:
    return isinstance(meta, dict) and meta.get("node_type") == "array"


@dataclass(frozen=True)
class StoreLayout:
    """Validated ZIP entry names and Blosc-array metadata for a zarr store."""

    names: frozenset[str]
    blosc_blocksizes: dict[str, int]


def inspect_store(path: Path) -> StoreLayout:
    """Read every ZIP entry and collect every supported Blosc array declaration.

    ``ZipFile.testzip`` consumes each complete entry, so this validates central
    directory consistency and every entry's CRC before callers use the result to
    skip a conversion.  Duplicate paths are rejected because a set comparison
    otherwise could call a partial archive complete.
    """
    with zipfile.ZipFile(path, mode="r") as zin:
        infos = zin.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError(f"{path}: ZIP contains duplicate entry names")
        for info in infos:
            if not info.is_dir() and info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(
                    f"{path}: ZIP entry {info.filename!r} is not ZIP_STORED"
                )
        bad_entry = zin.testzip()
        if bad_entry is not None:
            raise zipfile.BadZipFile(f"{path}: CRC failed for ZIP entry {bad_entry!r}")

        blocksizes: dict[str, int] = {}
        for key in names:
            if not key.endswith("zarr.json"):
                continue
            meta = json.loads(zin.read(key))
            if not _is_array_meta(meta):
                continue
            found = _recompressible_blosc_codec(meta, key)
            if found is not None:
                _, cfg = found
                blocksizes[key] = int(cfg.get("blocksize", 0))
    return StoreLayout(names=frozenset(names), blosc_blocksizes=blocksizes)


def blosc_array_blocksizes(path: Path) -> dict[str, int]:
    """Return every supported Blosc array's declared blocksize in ``path``.

    Inspection also validates every ZIP entry and rejects unsupported Blosc
    pipelines.  Batch conversion relies on this before declaring a destination
    complete enough to skip.
    """
    return inspect_store(path).blosc_blocksizes


# ---------------------------------------------------------------------------
# chunk discovery
# ---------------------------------------------------------------------------


def _array_prefix(zarr_json_key: str) -> str:
    """``"X/data/zarr.json"`` -> ``"X/data/"`` (the key prefix for its chunks)."""
    # zarr.json is always the last path segment; chunks live as siblings under
    # the same array directory, keyed "<prefix>c/<coords>".
    return zarr_json_key[: -len("zarr.json")]


def _chunk_array_prefix(key: str) -> str | None:
    """Return the array prefix for a scdata v3 default chunk key in O(1).

    scdata writes ``chunk_key_encoding={name: default, separator: /}``, so a
    chunk belongs to ``<array-prefix>c/<coords>``.  Splitting at the final
    marker handles an array directory literally named ``c`` and avoids scanning
    every array prefix for every chunk.
    """
    prefix, marker, _ = key.rpartition("/c/")
    if marker:
        return f"{prefix}/"
    if key.startswith("c/"):
        return ""
    return None


# ---------------------------------------------------------------------------
# the core re-compress
# ---------------------------------------------------------------------------


@dataclass
class ConvertStats:
    blosc_arrays: int = 0
    blosc_chunks: int = 0
    skipped_arrays: int = 0  # blosc arrays already at target blocksize
    copied_entries: int = 0
    recompressed_bytes_in: int = 0
    recompressed_bytes_out: int = 0
    decode_errors: int = 0


def _recompress_chunk(
    src_bytes: bytes, params: BloscParams, target_blocksize: int, *, verify: bool
) -> bytes:
    """Decode ``src_bytes`` with the source params, re-encode at target blocksize.

    When ``verify`` is set, the re-encoded chunk is decoded again and compared
    byte-for-byte against the source decode — catches any silent corruption.
    """
    decoder = params.decoder()
    encoder = params.encoder(target_blocksize)

    # numcodecs may return a bytearray/memoryview; normalize to bytes so the
    # encode input and the verify comparison are both exact.
    decoded = bytes(decoder.decode(src_bytes))
    reencoded = bytes(encoder.encode(decoded))

    if verify:
        roundtrip = bytes(decoder.decode(reencoded))
        if roundtrip != decoded:
            raise RuntimeError(
                "blosc roundtrip mismatch: re-encoded chunk decodes to different "
                f"bytes than the source (in={len(decoded)} out={len(roundtrip)})"
            )
    return reencoded


def _needs_recompress(
    cfg: dict[str, Any], target_blocksize: int
) -> bool:
    """True if the array's blosc blocksize differs from the target.

    An array already at the target blocksize is copied verbatim (chunk bytes and
    zarr.json both unchanged) — this makes the script idempotent and lets a
    partial rerun skip finished arrays.
    """
    current = int(cfg.get("blocksize", 0))
    return current != int(target_blocksize)


def convert_store(
    src: Path,
    dst: Path,
    *,
    target_blocksize: int,
    overwrite: bool,
    verify: bool,
) -> ConvertStats:
    """Stream-copy ``src`` zarr.zip to ``dst``, re-compressing blosc chunks.

    The output is written to a sibling temp file and ``os.replace``-d onto
    ``dst`` only after the whole store succeeds (and verification, if enabled),
    so a failure never leaves a half-written target.
    """
    if dst.exists() and not overwrite:
        raise FileExistsError(
            f"output exists: {dst} (pass --overwrite to replace)"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)

    stats = ConvertStats()
    # Track array prefixes that had their zarr.json blocksize rewritten, so the
    # chunk re-compress and the metadata rewrite stay consistent.
    recompress_prefixes: dict[str, BloscParams] = {}

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dst.name}.", suffix=".tmp", dir=dst.parent
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        # First pass: read every zarr.json, decide which arrays to re-compress.
        # We hold the source zip open read-only throughout; chunks are pulled on
        # demand so peak memory is ~one chunk.
        with zipfile.ZipFile(src, mode="r") as zin, zipfile.ZipFile(
            tmp, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as zout:
            names = zin.namelist()

            # 1. Write all zarr.json nodes first (small), rewriting blosc
            #    blocksize where needed.  Record the params for arrays we will
            #    re-compress so the chunk pass knows how.
            zarr_json_keys = [n for n in names if n.endswith("zarr.json")]
            for key in zarr_json_keys:
                raw_bytes = zin.read(key)
                meta = json.loads(raw_bytes)
                if not _is_array_meta(meta):
                    # group node: no codecs field — copy the original bytes
                    # verbatim so its (pretty-printed) formatting is preserved.
                    zout.writestr(key, raw_bytes)
                    stats.copied_entries += 1
                    continue
                found = _recompressible_blosc_codec(meta, key)
                if found is None:
                    # Non-Blosc arrays (zstd / uncompressed / string) are not
                    # interpreted and remain byte-identical.
                    zout.writestr(key, raw_bytes)
                    stats.copied_entries += 1
                    continue
                idx, cfg = found
                params = BloscParams.from_codec_config(cfg)
                if not _needs_recompress(cfg, target_blocksize):
                    # already at target blocksize: copy meta + (later) chunks verbatim.
                    zout.writestr(key, raw_bytes)
                    stats.skipped_arrays += 1
                    continue
                # rewrite the blocksize field in the blosc configuration.
                new_cfg = dict(cfg)
                new_cfg["blocksize"] = int(target_blocksize)
                new_codecs = list(meta["codecs"])
                new_codec_entry = dict(new_codecs[idx])
                new_codec_entry["configuration"] = new_cfg
                new_codecs[idx] = new_codec_entry
                new_meta = dict(meta)
                new_meta["codecs"] = new_codecs
                zout.writestr(key, _compact_json(new_meta))
                recompress_prefixes[_array_prefix(key)] = params
                stats.blosc_arrays += 1

            # 2. Stream chunk files.  For arrays marked for re-compress, decode
            #    + re-encode; for every other entry, byte-copy.  Deriving the
            #    array prefix from the scdata v3 ``.../c/<coords>`` chunk key is
            #    O(1), unlike testing every array prefix for every chunk.
            non_meta_keys = [n for n in names if not n.endswith("zarr.json")]
            for key in non_meta_keys:
                prefix = _chunk_array_prefix(key)
                params = recompress_prefixes.get(prefix) if prefix is not None else None
                if params is None:
                    # Not a changed Blosc chunk (or a skipped array): byte-copy.
                    zout.writestr(key, zin.read(key))
                    stats.copied_entries += 1
                    continue
                src_bytes = zin.read(key)
                if len(src_bytes) == 0:
                    # zero-length chunk (absent/fill-value): copy as-is, the
                    # databank does not decode it.
                    zout.writestr(key, src_bytes)
                    stats.copied_entries += 1
                    continue
                new_bytes = _recompress_chunk(
                    src_bytes, params, target_blocksize, verify=verify
                )
                zout.writestr(key, new_bytes)
                stats.blosc_chunks += 1
                stats.recompressed_bytes_in += len(src_bytes)
                stats.recompressed_bytes_out += len(new_bytes)

        # Verification happens against the on-disk temp file (the future dst),
        # using scdata's own launch path so we exercise the real reader.
        if verify:
            _verify_store(tmp, src, recompress_prefixes, target_blocksize)

        os.replace(tmp, dst)
        return stats
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except FileNotFoundError:
            pass


def _compact_json(meta: Any) -> bytes:
    """Serialize zarr.json with sorted keys and a trailing newline.

    scdata writes ``json.dumps(meta) + "\\n"`` (see _write_v3_node); we match
    that so a skipped array's zarr.json is byte-identical to the source.  For
    rewritten arrays only the blocksize field differs.
    """
    return (json.dumps(meta, sort_keys=False) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


def _verify_store(
    converted: Path,
    source: Path,
    recompress_prefixes: dict[str, BloscParams],
    target_blocksize: int,
) -> None:
    """Validate staged output before it can replace its destination.

    Every changed chunk is independently decoded from both archives and compared
    byte-for-byte.  ``ZipFile.testzip`` reads every output entry completely,
    forcing ZIP CRC validation even for unchanged metadata/string entries.
    """
    with zipfile.ZipFile(source, mode="r") as zin, zipfile.ZipFile(
        converted, mode="r"
    ) as zout:
        source_names = zin.namelist()
        converted_names = zout.namelist()
        if len(converted_names) != len(set(converted_names)):
            raise RuntimeError("verify: converted ZIP contains duplicate entry names")
        if set(source_names) != set(converted_names):
            missing = sorted(set(source_names) - set(converted_names))
            extra = sorted(set(converted_names) - set(source_names))
            raise RuntimeError(
                f"verify: converted ZIP entry set differs (missing={missing[:3]!r}, "
                f"extra={extra[:3]!r})"
            )
        bad_entry = zout.testzip()
        if bad_entry is not None:
            raise zipfile.BadZipFile(
                f"verify: converted ZIP CRC failed for entry {bad_entry!r}"
            )

        for prefix, params in recompress_prefixes.items():
            meta_key = f"{prefix}zarr.json"
            meta = json.loads(zout.read(meta_key))
            found = _recompressible_blosc_codec(meta, meta_key)
            if found is None or int(found[1].get("blocksize", 0)) != target_blocksize:
                raise RuntimeError(
                    f"verify: {meta_key} does not declare blocksize={target_blocksize}"
                )

        for key in source_names:
            prefix = _chunk_array_prefix(key)
            params = recompress_prefixes.get(prefix) if prefix is not None else None
            if params is None:
                continue
            src_chunk = zin.read(key)
            if not src_chunk:
                continue
            new_chunk = zout.read(key)
            src_decoded = bytes(params.decoder().decode(src_chunk))
            new_decoded = bytes(params.decoder().decode(new_chunk))
            if src_decoded != new_decoded:
                raise RuntimeError(f"verify: chunk {key} decode mismatch after re-compress")

    # Exercise scdata's reader only after the byte-level archive checks pass.
    from scdata.io import launch

    ds = launch(converted)
    _ = ds.num_cells, ds.num_genes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", type=Path, help="Source .zarr.zip (blocksize=0).")
    p.add_argument(
        "--output",
        type=Path,
        help=(
            "Destination .zarr.zip.  Defaults to overwriting the input "
            "(writes to a temp file, then atomically replaces)."
        ),
    )
    p.add_argument(
        "--blocksize",
        type=int,
        default=64 * 1024,
        help="Target Blosc block size in bytes (default: 65536 = 64 KiB).",
    )
    p.add_argument("--overwrite", action="store_true", help="Replace an existing output.")
    p.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Before publishing, CRC-check every ZIP entry, byte-compare every "
            "changed chunk, then open the staged store with scdata.io.launch."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be re-compressed without writing output.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    src: Path = args.input.resolve()
    if not src.is_file():
        print(f"error: input not found: {src}", file=sys.stderr)
        return 2
    if args.blocksize < 0:
        print(f"error: --blocksize must be non-negative, got {args.blocksize}", file=sys.stderr)
        return 2

    dst: Path = (args.output.resolve() if args.output is not None else src)

    if args.dry_run:
        return _dry_run(src, args.blocksize)

    overwrite = args.overwrite or (dst == src)
    started = time.perf_counter()
    stats = convert_store(
        src,
        dst,
        target_blocksize=args.blocksize,
        overwrite=overwrite,
        verify=args.verify,
    )
    elapsed = time.perf_counter() - started
    _print_stats(stats, elapsed, src, dst, args.blocksize)
    return 0


def _dry_run(src: Path, target_blocksize: int) -> int:
    with zipfile.ZipFile(src, mode="r") as zin:
        names = zin.namelist()
        recompress_prefixes: set[str] = set()
        already = 0
        for key in (n for n in names if n.endswith("zarr.json")):
            meta = json.loads(zin.read(key))
            if not _is_array_meta(meta):
                continue
            found = _recompressible_blosc_codec(meta, key)
            if found is None:
                continue
            _, cfg = found
            if not _needs_recompress(cfg, target_blocksize):
                already += 1
                continue
            recompress_prefixes.add(_array_prefix(key))
        chunks = sum(
            _chunk_array_prefix(key) in recompress_prefixes
            for key in names
            if not key.endswith("zarr.json")
        )
    print(
        f"dry-run: {src.name}\n"
        f"  blosc arrays to re-compress: {len(recompress_prefixes)}\n"
        f"  blosc chunks to re-compress: {chunks}\n"
        f"  blosc arrays already at blocksize={target_blocksize}: {already}"
    )
    return 0


def _print_stats(
    stats: ConvertStats, elapsed: float, src: Path, dst: Path, blocksize: int
) -> None:
    in_mib = stats.recompressed_bytes_in / (1024 * 1024)
    out_mib = stats.recompressed_bytes_out / (1024 * 1024)
    ratio = (stats.recompressed_bytes_out / stats.recompressed_bytes_in) if stats.recompressed_bytes_in else 0.0
    print(
        f"converted {src.name} -> {dst.name}\n"
        f"  target blocksize: {blocksize}\n"
        f"  blosc arrays re-compressed: {stats.blosc_arrays}\n"
        f"  blosc chunks re-compressed: {stats.blosc_chunks}\n"
        f"  blosc arrays skipped (already at target): {stats.skipped_arrays}\n"
        f"  entries copied verbatim: {stats.copied_entries}\n"
        f"  recompressed: {in_mib:.1f} MiB -> {out_mib:.1f} MiB (ratio {ratio:.3f})\n"
        f"  elapsed: {elapsed:.1f}s"
    )


if __name__ == "__main__":
    raise SystemExit(main())
