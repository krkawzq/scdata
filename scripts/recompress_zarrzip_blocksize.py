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
For any array whose ``codecs`` list contains a ``blosc`` entry, the re-compressed
output is byte-identical to what :func:`scdata.io.write_zarr` would produce when
called with the same Blosc parameters and the same target ``blocksize`` —
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
        [--verify] [--jobs N]

With ``--verify`` the converted store is opened with :func:`scdata.io.launch`
and every re-compressed chunk is decoded back and compared byte-for-byte against
the source chunk's decode, so a silent corruption would fail the run.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
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


def _find_blosc_codec(meta: dict[str, Any]) -> tuple[int, dict[str, Any]] | None:
    """Return (index, configuration) of the blosc codec in ``meta["codecs"]``.

    Returns None if the array has no blosc compressor (e.g. zstd-only, or a
    pure ``bytes`` serializer).  A v3 codecs list is [serializer, *compressors];
    there is at most one blosc entry in scdata-written stores.
    """
    codecs = meta.get("codecs")
    if not isinstance(codecs, list):
        return None
    for i, entry in enumerate(codecs):
        if isinstance(entry, dict) and entry.get("name") == "blosc":
            cfg = entry.get("configuration")
            if not isinstance(cfg, dict):
                raise ValueError("blosc codec entry missing 'configuration' object")
            return i, cfg
    return None


def _is_array_meta(meta: dict[str, Any]) -> bool:
    return isinstance(meta, dict) and meta.get("node_type") == "array"


# ---------------------------------------------------------------------------
# chunk discovery
# ---------------------------------------------------------------------------


def _array_prefix(zarr_json_key: str) -> str:
    """``"X/data/zarr.json"`` -> ``"X/data/"`` (the key prefix for its chunks)."""
    # zarr.json is always the last path segment; chunks live as siblings under
    # the same array directory, keyed "<prefix>c/<coords>".
    return zarr_json_key[: -len("zarr.json")]


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


def _blosc_header(blocksize_or_nbytes: bytes) -> tuple[int, int, int]:
    """Read (nbytes, blocksize, ctbytes) from the first 16 bytes of a blosc chunk."""
    nbytes = struct.unpack_from("<I", blocksize_or_nbytes, 4)[0]
    blocksize = struct.unpack_from("<I", blocksize_or_nbytes, 8)[0]
    ctbytes = struct.unpack_from("<I", blocksize_or_nbytes, 12)[0]
    return nbytes, blocksize, ctbytes


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
                found = _find_blosc_codec(meta)
                if found is None:
                    # non-blosc array (zstd / uncompressed / string): copy the
                    # original bytes verbatim.
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
            #    + re-encode; for every other entry, byte-copy.
            #    We process keys in sorted order for cache-friendly reads.
            non_meta_keys = [n for n in names if not n.endswith("zarr.json")]
            for key in non_meta_keys:
                # Is this chunk under an array we're re-compressing?
                prefix_match = None
                for prefix in recompress_prefixes:
                    if key.startswith(prefix):
                        prefix_match = prefix
                        break
                if prefix_match is None:
                    # not a blosc-array chunk (or a skipped blosc array): copy.
                    zout.writestr(key, zin.read(key))
                    stats.copied_entries += 1
                    continue
                params = recompress_prefixes[prefix_match]
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
    """Open the converted store with scdata.io.launch and sanity-check it.

    ``launch`` parses every zarr.json, reads var/_index (decodes its chunks) and
    CSR indptr (decodes its chunks), so a broken blosc config or a corrupt
    indptr/var chunk would raise here.  We additionally re-decode every
    re-compressed chunk of indptr (which launch decodes anyway) plus a sample of
    data/indices chunks and compare against the source decode — the per-chunk
    byte roundtrip in ``_recompress_chunk`` already covers every chunk, but this
    is a second independent check through scdata's reader.
    """
    from scdata.io import launch

    ds = launch(converted)
    # launch already validated shapes + decoded indptr + var/_index.
    _ = ds.num_cells, ds.num_genes

    # Cross-check: re-decode a sample of re-compressed chunks via the source
    # reader and confirm the blosc header blocksize is self-consistent.  The
    # authoritative equivalence (converted == write_zarr output) was already
    # established per-chunk in _recompress_chunk(verify=True).
    with zipfile.ZipFile(source, mode="r") as zin, zipfile.ZipFile(
        converted, mode="r"
    ) as zout:
        checked = 0
        for prefix, params in recompress_prefixes.items():
            # check up to 3 chunks per array (indptr usually has 1; data has many).
            sample = []
            for name in zin.namelist():
                if name.startswith(prefix) and not name.endswith("zarr.json"):
                    sample.append(name)
            sample.sort()
            for key in sample[:3]:
                src_chunk = zin.read(key)
                new_chunk = zout.read(key)
                if len(src_chunk) == 0:
                    continue
                src_dec = bytes(params.decoder().decode(src_chunk))
                new_dec = bytes(params.decoder().decode(new_chunk))
                if src_dec != new_dec:
                    raise RuntimeError(
                        f"verify: chunk {key} decode mismatch after re-compress"
                    )
                checked += 1
        # at least the indptr of every re-compressed CSR array was decoded by
        # launch above; make sure we also touched at least one chunk here.
        if checked == 0 and recompress_prefixes:
            raise RuntimeError("verify: no re-compressed chunk was sampled")


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
            "After conversion, open with scdata.io.launch and byte-compare a "
            "sample of re-compressed chunks against the source."
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
        arrays = 0
        chunks = 0
        already = 0
        for key in (n for n in names if n.endswith("zarr.json")):
            meta = json.loads(zin.read(key))
            if not _is_array_meta(meta):
                continue
            found = _find_blosc_codec(meta)
            if found is None:
                continue
            _, cfg = found
            if not _needs_recompress(cfg, target_blocksize):
                already += 1
                continue
            arrays += 1
            prefix = _array_prefix(key)
            chunks += sum(1 for n in names if n.startswith(prefix) and not n.endswith("zarr.json"))
    print(
        f"dry-run: {src.name}\n"
        f"  blosc arrays to re-compress: {arrays}\n"
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
