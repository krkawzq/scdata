# sc-compress Python binding

Dense / CSR materialization uses the in-memory types **`ScDense`** and
**`ScCsr`** (NumPy-backed). Hot paths — on-demand row/column select, fancy
indexing, densify, scatter — are implemented in Rust and do **not** depend on
SciPy. SciPy remains optional for write interop and ``ScCsr.to_scipy()``.

The private ``sc_compress._core`` extension opens stores, exposes format
metadata, and runs validated buffers through the Rust codec / kernel layer.
ZIP helpers live in ``sc_compress.zip``.

## Layout

| Path | Role |
|---|---|
| `crates/sc-compress/python/` | private maturin / PyO3 `_core` extension |
| `src/sc_compress/` | public Python API |

```text
src/sc_compress/
  __init__.py      # open_store / write / Store / exceptions / limits / read_scc
  anndata.py       # AnnData .scc / .scc.zip bridge (optional anndata extra)
  io.py            # directory open/write
  store.py         # Store, StoreInfo
  zip.py           # pack / list_stores / write_* into archives
  format.py        # FORMAT_*, storage/NumPy dtypes
  limits.py        # ReadLimits
  write_options.py # WriteOptions
  exceptions.py    # error hierarchy
  _validate.py     # private buffer/path checks
  _core.*          # Rust extension
```

## Develop

```bash
cd crates/sc-compress/python
maturin develop --uv
pytest
```

## Usage

```python
import numpy as np
import sc_compress as scc
from scipy import sparse

values = np.arange(24, dtype=np.float32).reshape(6, 4)
scc.write("matrix.sc", values, n_workers=8)  # dense/sparse dispatch stays in Python
# Defaults use byte budgets (chunk=100 MiB, block=400 KiB). CSR keeps bytes_budget;
# dense budgets are lowered in Python to fixed_cells that meet/exceed the target:
# scc.write_csr(path, csr, options=scc.WriteOptions(chunk_budget=1 << 20, block_budget=400 << 10))

with scc.open_store("matrix.sc", n_workers=8) as store:
    print(store)
    print(store.info())
    batch = store[2:5]                    # ScDense
    last = store[-1]                      # 1-D ndarray (scalar row)
    genes = store[:, [0, 3, 1]]            # column gather
    mini = store[[5, 1, 5], 1:3]          # fancy cells × gene strip
    for batch in store.iter_batches(batch_size=2):
        print(batch.shape)

csr = sparse.csr_matrix(values)
scc.write("matrix_csr.sc", csr)
with scc.open_store("matrix_csr.sc") as store:
    csr_batch = store[2:5]                              # ScCsr
    dense_genes = store.select([0, 2], [1, 0], csr_output="dense")  # ScDense
```

Indexing supports zarr-like on-demand loads: row/column slices, fancy integer
indices, boolean masks, and strided slices. ``iter_batches()`` avoids decoding
a full matrix for streaming workflows.

``Store.dtype`` / ``Store.index_dtype`` are NumPy dtypes.
``Store.storage_dtype`` / ``Store.storage_index_dtype`` are the on-disk names
(for example ``"f32"``).

## ZIP / :mod:`zipfile`

Writers still emit directory stores. Pack them with the stdlib ``zipfile`` API,
or write straight into an archive via :mod:`sc_compress.zip`:

```python
import zipfile
import sc_compress as scc

scc.write_dense("matrix.sc", values)
with zipfile.ZipFile("matrices.zip", "w", compression=zipfile.ZIP_STORED) as zf:
    scc.zip.pack(zf, "assay", "matrix.sc")

# or one-shot
scc.zip.write_dense("matrices.zip", "assay", values)

assert scc.zip.list_stores("matrices.zip") == ["assay"]
store = scc.open_store("matrices.zip")  # one prefix: selected automatically
# file-backed ZipFile also works:
with zipfile.ZipFile("matrices.zip") as zf:
    store = scc.open_store(zf)
```

If an archive contains multiple stores, pass ``zip_prefix``; the error from an
ambiguous open lists every available prefix. Prefer ``ZIP_STORED`` because the
matrix chunks are already compressed and stored entries support efficient range
reads.

Open-time resource limits can be reused as an immutable Python object. Direct
keyword overrides remain available and take precedence:

```python
limits = scc.ReadLimits(max_decoded_size=8 << 30, n_workers=8)
store = scc.open_store(path, limits=limits, max_block_count=2_000_000)
```

Write partitioning uses the same pattern via :class:`sc_compress.WriteOptions`:

```python
opts = scc.WriteOptions(chunk_budget=16 << 20, block_budget=256 << 10, n_workers=8)
# or fixed cell counts:
# opts = scc.WriteOptions(chunk_policy="cells", chunk_cells=512, block_policy="cells", block_cells=16)
scc.write(path, values, options=opts, overwrite=False)
```

``n_workers`` is a positive per-operation upper bound. It defaults to
``scc.DEFAULT_N_WORKERS`` (Rust's available parallelism), and a direct keyword
overrides the value stored in ``ReadLimits`` or ``WriteOptions``. The same keyword
is accepted by ZIP and AnnData read/write helpers.

Dtype checks use `numpy.dtype(...)` so aliases such as `"float32"` / `"f4"` are
accepted the same way NumPy accepts them. Non-native-endian arrays are converted
in Python before entering Rust.

## AnnData (``.scc`` / ``.scc.zip``)

Optional extra: `pip install 'sc-compress[anndata]'`. Annotations use AnnData's
zarr-v3 writers. Only **cell-aligned** matrices use sc-compress (`X`, `layers`,
`raw.X`, `obsm`, `obsp`). `uns` / `varm` / `varp` stay on the zarr path.
Multi-dimensional cell tensors (cell axis leftmost or rightmost) are densified
and flattened to `(n_cells, -1)` on write; Python reshapes on read.

```python
from sc_compress import write_scc, read_scc

write_scc(adata, "data.scc.zip")          # ZIP via suffix (store="auto")
write_scc(adata, "data.scc", store="dir") # directory tree

adata2 = read_scc("data.scc.zip")
meta = read_scc("data.scc", exclude=("X", "layers", "raw"))  # X=None, layers={}, no raw
```
