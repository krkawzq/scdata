#!/usr/bin/env python3
"""Convert boss-format ``*.zarr.zip`` stores to ``*.scc.zip`` side-by-side.

Uses **scdata 0.1** ``read_zarr`` for the source layout (rectilinear CSR /
dense1d / ``.zarr.zip``) and **scdata-toolkit 0.2** ``write_scc`` for the
destination.  Both packages live in the dedicated convert venv
(``.venv-convert``): ``scdata`` is the 0.1 wheel (import ``scdata``),
``scdata-toolkit`` is installed as import ``scdata_toolkit`` so the two
do not collide.

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


_SCC_VALUE_DTYPES = {
    "float32",
    "float64",
    "int16",
    "int32",
    "int64",
    "uint16",
    "uint32",
    "uint64",
}


def _promote_matrix(matrix):
    """Losslessly widen dtypes that sc-compress cannot store (int8/uint8/bool/float16)."""
    import numpy as np
    from scipy import sparse

    if matrix is None:
        return matrix
    dtype = getattr(matrix, "dtype", None)
    if dtype is None:
        return matrix
    kind = np.dtype(dtype)
    if kind.name in _SCC_VALUE_DTYPES:
        return matrix
    if kind.kind == "b":
        target = np.uint16
    elif kind.kind == "u" and kind.itemsize <= 1:
        target = np.uint16
    elif kind.kind == "i" and kind.itemsize <= 1:
        target = np.int16
    elif kind == np.dtype("float16"):
        target = np.float32
    else:
        return matrix
    if sparse.issparse(matrix):
        return matrix.astype(target)
    return np.asarray(matrix).astype(target, copy=False)


def _promote_scc_dtypes(adata):
    """Widen unsupported numeric cell-aligned matrices so write_scc can accept them."""
    import pandas as pd

    adata.X = _promote_matrix(adata.X)
    for key in list(adata.layers.keys()):
        adata.layers[key] = _promote_matrix(adata.layers[key])
    for slot_name in ("obsm", "obsp"):
        slot = getattr(adata, slot_name)
        for key in list(slot.keys()):
            value = slot[key]
            if isinstance(value, pd.DataFrame):
                continue
            slot[key] = _promote_matrix(value)
    if adata.raw is None:
        return adata
    promoted = _promote_matrix(adata.raw.X)
    if promoted is adata.raw.X:
        return adata
    raw = adata.raw.to_adata()
    raw.X = promoted
    adata.raw = raw
    return adata


def _as_dense_rows(matrix, n: int):
    """First ``n`` rows as a dense ndarray (SciPy CSR or scdata-toolkit ScCsr)."""
    import numpy as np
    from scipy import sparse

    sl = matrix[:n]
    if sparse.issparse(sl):
        return np.asarray(sl.toarray())
    toarray = getattr(sl, "toarray", None)
    if callable(toarray):
        return np.asarray(toarray())
    to_numpy = getattr(sl, "to_numpy", None)
    if callable(to_numpy):
        return np.asarray(to_numpy())
    return np.asarray(sl)


def convert_one(
    src: str,
    *,
    overwrite: bool,
    num_workers: int | None,
    verify: bool,
) -> dict:
    """Worker entry: return a result dict (picklable)."""
    from scdata.io import read_zarr
    import scdata_toolkit as scc
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
        _promote_scc_dtypes(adata)

        scc.write_scc(
            adata,
            dst_path,
            store="zip",
            overwrite=overwrite,
            num_workers=num_workers,
            options=scc.WriteOptions(block_budget=64 << 10),
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
            a = _as_dense_rows(adata.X, n)
            b = _as_dense_rows(back.X, n)
            if not np.array_equal(a, b) and not np.allclose(
                a.astype(np.float64, copy=False),
                b.astype(np.float64, copy=False),
                equal_nan=True,
            ):
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
        "--num-workers",
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
        f"verify={args.verify} num_workers={args.num_workers} "
        f"block_budget={64 << 10}",
        flush=True,
    )

    ok = skipped = errors = 0
    kwargs = {
        "overwrite": bool(args.overwrite),
        "num_workers": args.num_workers,
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
