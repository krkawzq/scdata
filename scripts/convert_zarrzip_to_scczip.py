#!/usr/bin/env python3
"""Convert boss-format ``*.zarr.zip`` stores to ``*.scc.zip`` side-by-side.

Uses **scdata 0.1** ``read_zarr`` for the source layout (rectilinear CSR /
dense1d / ``.zarr.zip``) and **sc-compress 0.2** ``write_scc`` for the
destination.  Both packages live in the dedicated convert venv
(``.venv-convert``): ``scdata`` is the 0.1 wheel, ``sc_compress`` is 0.2.

Destination path is the source path with the ``.zarr.zip`` suffix replaced by
``.scc.zip``.  Existing destinations are skipped unless ``--overwrite``.

Examples
--------
# one small file (smoke test)
python scripts/convert_zarrzip_to_scczip.py path/to/filtered.zarr.zip

# first N files under a tree
python scripts/convert_zarrzip_to_scczip.py /data/dataset --limit 3

# full tree, skip existing
python scripts/convert_zarrzip_to_scczip.py /data/dataset --jobs 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _dst_for(src: Path) -> Path:
    name = src.name
    if name.endswith(".zarr.zip"):
        return src.with_name(name[: -len(".zarr.zip")] + ".scc.zip")
    if name.endswith(".zip"):
        return src.with_name(name[: -len(".zip")] + ".scc.zip")
    return src.with_name(name + ".scc.zip")


def _iter_sources(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if root.is_file():
            if root.name.endswith(".zarr.zip"):
                out.append(root.resolve())
            else:
                raise SystemExit(f"not a .zarr.zip file: {root}")
            continue
        if not root.is_dir():
            raise SystemExit(f"path not found: {root}")
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if name.endswith(".zarr.zip"):
                    out.append(Path(dirpath, name).resolve())
    out.sort()
    return out


def convert_one(
    src: str,
    *,
    overwrite: bool,
    n_workers: int | None,
    verify: bool,
) -> dict:
    """Worker entry: return a result dict (picklable)."""
    from scdata.io import read_zarr
    import sc_compress as scc
    from scipy import sparse
    import numpy as np

    src_path = Path(src)
    dst_path = _dst_for(src_path)
    t0 = time.perf_counter()
    result: dict = {
        "src": str(src_path),
        "dst": str(dst_path),
        "status": "ok",
        "n_obs": None,
        "n_vars": None,
        "src_bytes": src_path.stat().st_size if src_path.exists() else None,
        "dst_bytes": None,
        "seconds": None,
        "error": None,
    }

    try:
        if dst_path.exists() and not overwrite:
            result["status"] = "skipped"
            result["dst_bytes"] = dst_path.stat().st_size
            result["seconds"] = time.perf_counter() - t0
            return result

        adata = read_zarr(src_path)
        result["n_obs"] = int(adata.n_obs)
        result["n_vars"] = int(adata.n_vars)

        scc.write_scc(
            adata,
            dst_path,
            store="zip",
            overwrite=overwrite,
            n_workers=n_workers,
        )
        result["dst_bytes"] = dst_path.stat().st_size

        if verify:
            back = scc.read_scc(dst_path)
            if back.n_obs != adata.n_obs or back.n_vars != adata.n_vars:
                raise RuntimeError(
                    f"shape mismatch after roundtrip: "
                    f"{adata.n_obs}x{adata.n_vars} -> {back.n_obs}x{back.n_vars}"
                )
            n = min(16, adata.n_obs)
            x1 = adata.X
            x2 = back.X
            if sparse.issparse(x1):
                a = x1[:n].toarray()
            else:
                a = np.asarray(x1[:n])
            if sparse.issparse(x2):
                b = x2[:n].toarray()
            else:
                b = np.asarray(x2[:n])
            if not np.array_equal(a, b) and not np.allclose(a, b, equal_nan=True):
                maxdiff = float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64))))
                raise RuntimeError(f"value mismatch on first {n} rows, maxdiff={maxdiff}")

        result["seconds"] = time.perf_counter() - t0
        return result
    except Exception as exc:  # noqa: BLE001 - surface full failure per file
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["seconds"] = time.perf_counter() - t0
        result["traceback"] = traceback.format_exc()
        return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="`.zarr.zip` files and/or directories to walk",
    )
    p.add_argument("--limit", type=int, default=None, help="convert at most N sources (after sort)")
    p.add_argument("--offset", type=int, default=0, help="skip the first N sources (after sort)")
    p.add_argument("--overwrite", action="store_true", help="replace existing `.scc.zip`")
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="process-level parallelism (each job is one file; default 1)",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="sc-compress matrix write workers per file (default: package default)",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="round-trip check shape + first 16 rows after each write",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop scheduling more work after the first error",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sources = _iter_sources(list(args.paths))
    if args.offset:
        sources = sources[args.offset :]
    if args.limit is not None:
        sources = sources[: args.limit]
    if not sources:
        print("no .zarr.zip sources found", file=sys.stderr)
        return 2

    print(
        f"sources={len(sources)} jobs={args.jobs} overwrite={args.overwrite} "
        f"verify={args.verify} n_workers={args.n_workers}",
        flush=True,
    )

    ok = skipped = errors = 0
    kwargs = {
        "overwrite": bool(args.overwrite),
        "n_workers": args.n_workers,
        "verify": bool(args.verify),
    }

    def _handle(res: dict) -> None:
        nonlocal ok, skipped, errors
        status = res["status"]
        if status == "ok":
            ok += 1
        elif status == "skipped":
            skipped += 1
        else:
            errors += 1
        bits = [
            status,
            f"obs={res.get('n_obs')}",
            f"vars={res.get('n_vars')}",
            f"src={res.get('src_bytes')}",
            f"dst={res.get('dst_bytes')}",
            f"t={res.get('seconds'):.2f}s" if res.get("seconds") is not None else "t=?",
            res["src"],
            "->",
            res["dst"],
        ]
        print(" ".join(str(b) for b in bits), flush=True)
        if status == "error":
            print(res.get("error"), file=sys.stderr, flush=True)
            if res.get("traceback"):
                print(res["traceback"], file=sys.stderr, flush=True)

    if args.jobs <= 1:
        for src in sources:
            res = convert_one(str(src), **kwargs)
            _handle(res)
            if args.fail_fast and res["status"] == "error":
                break
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(convert_one, str(src), **kwargs): src for src in sources
            }
            for fut in as_completed(futures):
                res = fut.result()
                _handle(res)
                if args.fail_fast and res["status"] == "error":
                    pool.shutdown(wait=False, cancel_futures=True)
                    break

    print(f"done ok={ok} skipped={skipped} errors={errors}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
