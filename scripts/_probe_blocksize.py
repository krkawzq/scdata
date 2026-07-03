import tempfile, os, zipfile, json, struct
import numpy as np
from scipy import sparse
import anndata as ad
import pandas as pd
from scdata.io import write_zarr

n_obs, n_var = 300, 500
rng = np.random.default_rng(1)
X = sparse.csr_matrix(
    (rng.integers(0, 50000, n_obs * 400).astype("uint16"),
     (np.repeat(np.arange(n_obs), 400), rng.integers(0, n_var, n_obs * 400))),
    shape=(n_obs, n_var),
)
X.sum_duplicates()
adata = ad.AnnData(
    X=X,
    obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
    var=pd.DataFrame(index=[f"g{i}" for i in range(n_var)]),
)
tmp = tempfile.mkdtemp()

cases = [
    ("default(no arg)", None),
    ("explicit 0", 0),
    ("explicit None", None),
    ("explicit 65536", 65536),
    ("explicit 131072", 131072),
]
for label, bs_arg in cases:
    out = os.path.join(tmp, f"{label}.zarr.zip".replace(" ", "_").replace("(", "").replace(")", ""))
    kw = dict(format="sparse", align_cells=True, store="zip", compressor="blosc.lz4.level5")
    if bs_arg is not None:
        kw["blocksize"] = bs_arg
    else:
        # for "explicit None" we want to pass None; for "default" we omit the arg
        if label.startswith("explicit"):
            kw["blocksize"] = None
    write_zarr(adata, out, **kw)
    z = zipfile.ZipFile(out)
    m = json.loads(z.read("X/data/zarr.json"))
    cfg = [c for c in m["codecs"] if c["name"] == "blosc"][0]["configuration"]
    ch = z.read("X/data/c/0")
    hdr_bs = struct.unpack_from("<I", ch, 8)[0]
    print(f"{label:>18}: zarr.json blocksize={cfg['blocksize']:>6}, chunk header={hdr_bs}")
