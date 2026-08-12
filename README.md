# sc-load

`sc-load` compiles an ordered list of `(source, row)` requests into a reusable,
block-granular prefetch plan over sc-compress matrices stored inside AnnData
[`.scc` / `.scc.zip`](crates/sc-compress/python/README.md) containers. Every execution
session owns its workers, output ring, cancellation state, and runtime
statistics.

The public API is Python-first. Container resolution, feature-name convenience
reads, validation, lifecycle helpers, typed statistics, and NumPy ergonomics
live in `src/sc_load`. The private `sc_load._core` extension only keeps opened
storage handles, compiled plans, execution sessions, and the batch copy into
owned NumPy memory.

## Development install

```sh
uv sync --extra dev
uv run maturin develop --release --strip --locked
```

Python 3.12 or newer and NumPy 2.2 or newer are required.

## Example

```python
import numpy as np
import sc_load

# Open one matrix inside an AnnData scc container (directory or ZIP).
dataset = sc_load.register("sample.scc.zip", key="X")

# Optional: attach a caller-built feature map (length == n_cols; None/-1 drops).
dataset = dataset.with_feature_map([2, None, 0, 1])
output = sc_load.OutputSpec(4, np.float32, fill=-1)

for batch in sc_load.prefetch(
    dataset,
    [7, 2, 11],
    output=output,
    batch_size=256,
    prefetch_step=8,
    config=sc_load.SessionConfig(io_mode="auto"),
):
    # `batch` is compact, NumPy-owned, and remains valid after the Rust ring
    # slot has been released and reused.
    consume(batch)

# For smaller results, compile once and materialize.
plan = sc_load.compile(dataset, [7, 2, 11], output=output)
matrix = plan.read()
```

`register()` accepts `.scc` directories and `.scc.zip` archives. Select the
matrix with `key=` (`"X"`, `"layers/<name>"`, `"raw/X"`, `"obsm/<name>"`, …).
Resolution uses `sc_compress.zip.list_stores` / the same `meta.json` layout as
`sc_compress.open_store`. Expression keys expose `feature_names` from `var` /
`raw/var` so callers can build maps externally; embedding keys leave
`feature_names` as `None`. `limits=` accepts `sc_load.ReadLimits` or
`sc_compress.ReadLimits`.

Multiple datasets use collection order as `source_id`:

```python
a = sc_load.register("a.scc", key="X", feature_map=map_a)
b = sc_load.register("b.scc", key="X", feature_map=map_b)
plan = sc_load.compile(
    [a, b],
    [(0, 10), (1, 3), (0, 11)],
    output=sc_load.OutputSpec(n_genes, np.float32, fill=0),
)
```

With one identity-mapped dataset, `compile()` / `prefetch()` can infer the
output width and dtype. Multi-dataset or feature-mapped plans require an
explicit `OutputSpec`.

## Distributed loading

`Plan.open_distributed()` starts one producer in the parent process and returns
one process-transferable iterator per rank. Storage reads and decoding happen
once; logical batches are assigned round-robin across ranks.

```python
import multiprocessing as mp

import sc_load


def consume_rank(loader: sc_load.DistributedIterator) -> None:
    with loader:
        for batch in loader:
            train_step(batch)


if __name__ == "__main__":
    context = mp.get_context("spawn")
    with plan.open_distributed(
        world_size=4,
        config=sc_load.SessionConfig(worker_count=8, io_mode="auto"),
    ) as distributed:
        processes = [
            context.Process(target=consume_rank, args=(loader,))
            for loader in distributed.ranks()
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"rank process exited with {process.exitcode}")
        distributed.wait()
```

The default `copy=True` yields compact, writable, NumPy-owned arrays and is the
safe choice for training loops. `distributed.rank(rank, copy=False)` is an
explicit read-only zero-copy mode: every retained array keeps one physical ring
generation leased, so release or drop views promptly. Holding enough views to
block the caller's own next batch raises `InvalidInputError` instead of
deadlocking. A rank iterator attaches lazily in its final process, may be
transferred only before consumption starts, and is the sole live owner of that
rank. Closing an incomplete attached iterator cancels the whole distributed
session so the producer and other ranks do not hang. `ranks()` creates its
remaining handles atomically: descriptor or allocation failure leaves every
rank retryable and closes handles created by that failed call.

Potentially rounding `i32/u32 -> f32` conversion requires
`allow_float_rounding=True`. Signedness/range failures are controlled by the
output overflow policy: `error`, `use_fill`, `use_value`, or `unchecked`.

## Ownership and cancellation

`Plan` is immutable and reusable. `Plan.open()` creates a fresh `Session`.
`prefetch()` returns a `Prefetch` iterator that opens a session on first use.
Blocking dataset open, plan compilation, batch waiting, and worker shutdown all
release the Python GIL. Standard-session batches are copied once into compact
NumPy storage and never alias a reusable ring slot; the distributed path's
explicit `copy=False` exception follows the lease rules described above.

Sessions and prefetch handles are context managers; `cancel()` wakes blocked
consumers, while `close()` cancels unfinished work, joins workers, and releases
the output ring. Both plan and session statistics are immutable typed snapshots
with `as_dict()` for logging and serialization.

The Rust execution model and lower-level invariants are documented in
[`crates/sc-load/README.md`](crates/sc-load/README.md).

## Publishing

CI builds and tests on every push/PR to `main`. Pushing a `v*` tag (for example
`v0.2.0`) builds Linux manylinux wheels (`x86_64`, `aarch64`) plus sdists for
both `sc-compress` and `sc-load`, then uploads them to PyPI.

1. Create a PyPI API token with upload access to both projects.
2. Add repository secret `PYPI_API_TOKEN`.
3. Keep package versions aligned (`pyproject.toml` and
   `crates/sc-compress/python/pyproject.toml`, currently driven by the Cargo
   workspace version).
4. Tag and push when ready:

```sh
git tag v0.2.0
git push origin v0.2.0
```
