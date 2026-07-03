"""End-to-end test for recompress_zarrzip_blocksize.py.

Strategy: scdata's current write_zarr can no longer emit blocksize=0 (the
_resolve_blocksize helper maps 0 -> 65536).  Real legacy stores were written by
an older scdata that did emit 0.  To test the 0 -> 65536 path we:

  1. write a golden store with blocksize=65536 (what scdata produces today),
  2. synthesize a "legacy" store by re-encoding every blosc chunk of the golden
     store with blocksize=0 and rewriting each blosc zarr.json to blocksize=0,
  3. run the conversion script on the legacy store -> 65536,
  4. assert the converted store is byte-identical to the golden store:
     every zarr.json, every chunk, every non-blosc entry.

Byte-identity is the strongest correctness check: it means the script's output
matches what scdata.write_zarr would produce, so launch / Rust / native path
all see exactly the same bytes as a freshly-converted store.
"""
import json
import os
import struct
import sys
import tempfile
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from numcodecs import Blosc
from scipy import sparse

from scdata.io import write_zarr

SCRIPT = Path(__file__).resolve().parent / "recompress_zarrzip_blocksize.py"
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
import recompress_zarrzip_blocksize as rc  # noqa: E402

SHUF = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}


def build_adata():
    rng = np.random.default_rng(42)
    n_obs, n_var = 800, 1200
    nnz_per = 600
    rows = np.repeat(np.arange(n_obs), nnz_per)
    cols = rng.integers(0, n_var, size=n_obs * nnz_per)
    vals = rng.integers(0, 60000, size=n_obs * nnz_per).astype("uint16")
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(n_obs, n_var))
    X.sum_duplicates()
    obs = pd.DataFrame({
        "cell_type": pd.Categorical([f"t{i % 5}" for i in range(n_obs)]),
        "nCount": rng.integers(0, 100000, n_obs).astype("int32"),
        "score": rng.standard_normal(n_obs).astype("float32"),
    }, index=[f"c{i}" for i in range(n_obs)])
    var = pd.DataFrame({"gene_name": [f"g{i}" for i in range(n_var)]},
                       index=[f"ENSG{i:06d}" for i in range(n_var)])
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.raw = a.copy()
    a.layers["dense_norm"] = rng.standard_normal((n_obs, n_var)).astype("float32")
    a.layers["sp_counts"] = X.copy()
    return a


def synthesize_legacy(golden_zip, legacy_zip):
    """Re-encode every blosc chunk of golden with blocksize=0, set zarr.json blocksize=0."""
    with zipfile.ZipFile(golden_zip) as zin, zipfile.ZipFile(
        legacy_zip, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as zout:
        for key in zin.namelist():
            raw = zin.read(key)
            if key.endswith("zarr.json"):
                meta = json.loads(raw)
                if meta.get("node_type") == "array":
                    for c in meta.get("codecs", []):
                        if c.get("name") == "blosc":
                            cfg = c["configuration"]
                            cfg = dict(cfg)
                            cfg["blocksize"] = 0
                            c = dict(c)
                            c["configuration"] = cfg
                            # rebuild meta
                            new_codecs = list(meta["codecs"])
                            for i, cc in enumerate(new_codecs):
                                if cc.get("name") == "blosc":
                                    new_codecs[i] = c
                            meta = dict(meta)
                            meta["codecs"] = new_codecs
                            raw = (json.dumps(meta) + "\n").encode()
                zout.writestr(key, raw)
                continue
            # chunk file: re-encode with blocksize=0 if under a blosc array
            # (we detect by checking siblings — simpler: try decode)
            # find the array's zarr.json to know params
            arr_meta_key = key.rsplit("/c/", 1)[0] + "/zarr.json"
            try:
                ameta = json.loads(zin.read(arr_meta_key))
            except KeyError:
                zout.writestr(key, raw)
                continue
            blosc = next((c for c in ameta.get("codecs", []) if c.get("name") == "blosc"), None)
            if blosc is None or len(raw) == 0:
                zout.writestr(key, raw)
                continue
            cfg = blosc["configuration"]
            shuf = SHUF[cfg["shuffle"]]
            decoded = Blosc(cname=cfg["cname"], clevel=cfg["clevel"], shuffle=shuf,
                            blocksize=0, typesize=cfg["typesize"]).decode(raw)
            reenc = Blosc(cname=cfg["cname"], clevel=cfg["clevel"], shuffle=shuf,
                          blocksize=0, typesize=cfg["typesize"]).encode(decoded)
            zout.writestr(key, bytes(reenc))


def assert_stores_equal(a_zip, b_zip, label):
    """Assert two zarr.zip stores are byte-identical entry-by-entry."""
    za = zipfile.ZipFile(a_zip)
    zb = zipfile.ZipFile(b_zip)
    na, nb = set(za.namelist()), set(zb.namelist())
    assert na == nb, f"{label}: entry set differs\n  only in A: {sorted(na - nb)[:5]}\n  only in B: {sorted(nb - na)[:5]}"
    diffs = []
    for key in sorted(na):
        ra = za.read(key)
        rb = zb.read(key)
        if ra != rb:
            diffs.append((key, len(ra), len(rb)))
    if diffs:
        msg = f"{label}: {len(diffs)} entries differ (of {len(na)}):\n"
        for k, la, lb in diffs[:10]:
            msg += f"  {k}: A={la} B={lb}\n"
        raise AssertionError(msg)
    print(f"  OK {label}: {len(na)} entries byte-identical")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="recompress_test_"))
    print(f"workdir: {tmp}")
    adata = build_adata()

    golden = tmp / "golden.zarr.zip"
    legacy = tmp / "legacy.zarr.zip"
    converted = tmp / "converted.zarr.zip"

    print("writing golden store (blocksize=65536)...")
    write_zarr(adata, golden, format="sparse", align_cells=True, store="zip",
               compressor="blosc.lz4.level5", blocksize=65536)

    print("synthesizing legacy store (blocksize=0)...")
    synthesize_legacy(golden, legacy)

    # sanity: legacy really has blocksize=0 in zarr.json
    zl = zipfile.ZipFile(legacy)
    xd = json.loads(zl.read("X/data/zarr.json"))
    bs = next(c for c in xd["codecs"] if c["name"] == "blosc")["configuration"]["blocksize"]
    assert bs == 0, f"legacy synthesis failed: X/data blocksize={bs}"
    # and chunk header blocksize should differ from golden's
    ch_legacy = zl.read("X/data/c/0")
    zg = zipfile.ZipFile(golden)
    ch_golden = zg.read("X/data/c/0")
    print(f"  legacy X/data/c/0: {len(ch_legacy)} bytes, header bs={struct.unpack_from('<I', ch_legacy, 8)[0]}")
    print(f"  golden X/data/c/0: {len(ch_golden)} bytes, header bs={struct.unpack_from('<I', ch_golden, 8)[0]}")
    assert ch_legacy != ch_golden, "legacy and golden chunks should differ (different blocksize)"

    print("\n=== test 1: dry-run ===")
    rc.main(["--dry-run", str(legacy)])

    print("\n=== test 2: convert with --verify ===")
    rc.main([str(legacy), "--output", str(converted),
             "--blocksize", "65536", "--verify"])

    print("\n=== test 3: converted == golden (byte-identical) ===")
    assert_stores_equal(converted, golden, "converted vs golden")

    print("\n=== test 4: idempotency — convert converted again, no-op ===")
    rc.main([str(converted), "--output", str(converted),
             "--blocksize", "65536", "--overwrite", "--verify"])
    assert_stores_equal(converted, golden, "re-converted vs golden")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
