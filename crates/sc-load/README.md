# sc-load

`sc-load` compiles an ordered list of `(source, row)` accesses into a static
decoded-cache execution graph over `sc-compress` stores. A logical `Job` is
exactly one output batch. Workers schedule only ready I/O/decode or scatter
tasks; they never claim an unresolved Job and wait on it.

## Execution model

Compilation performs the expensive dynamic work once:

1. resolve every row to independently decodable data/indices blocks;
2. deduplicate per-batch block requirements;
3. simulate a fixed-capacity decoded cache with aligned, dependency-aware
   Best-Fit extents;
4. assign each residency a cache offset, generation, availability epoch and
   `BlockReady` token;
5. fuse compatible physical reads without crossing source/read-view,
   availability epoch, owner Job, priority, staging or decode limits;
6. flatten batch Jobs, I/O/DecodeOps, dense/CSR scatter tasks, PrefixDone
   releases and output-ring releases into immutable arenas.

`Session::open` allocates one decoded cache and output ring, then lowers the
relocatable plan into pointer-rich session-local descriptors. `InitializeJob`
uses a temporary blocking pool to fill the initial cache. Ordinary workers then
execute the ready-task graph with earliest-consumer-batch priority.

```text
PrefixDone ──> IoDecodeLoadTask
                    ├─ DecodeOp ─> BlockReady ─> Dense/CsrScatter
                    └─ DecodeOp ─> BlockReady ─> Dense/CsrScatter
                                                   │
                                                   v
                                                JobDone
                                                   │
                                                   v
                                              BatchReady
```

Decoded-cache lifetime and output lifetime are independent:

- `cache_capacity_bytes` controls how far decoded residencies can be prefetched;
- `prefetch_step` is exactly the number of output-ring slots;
- `JobDone` advances cache `PrefixDone` without waiting for model consumption;
- dropping a `Batch` releases only its output generation.

The runtime has no cache allocator, LRU, residency hash lookup, eviction logic
or cache reference count. Cache addresses and overwrite dependencies are fixed
in the Plan.

## I/O backends and fusion

Blocking and io_uring execute the same graph through private worker strategies
in `session/blocking.rs` and `session/uring.rs`; scheduling, cache dependencies
and ring lifecycle stay in `session/mod.rs`. Blocking workers batch ready-node
claims and reuse one encoded staging vector. Each io_uring worker owns one ring,
keeps multiple positioned reads in flight up to queue/job/byte limits, reuses
best-fit slot buffers, resubmits short reads, validates slot generations, and
drains read/cancel CQEs before buffers are released. On Linux, explicit
`IoMode::Uring` requires positioned sources; `Auto` selects io_uring only when
every source is positioned and every worker ring can be created, otherwise it
selects blocking. Key-backed and Deflated ZIP sources use blocking.

`IoMergeOptions` supports:

- `Off`: one physical task per residency;
- `Adjacent` (default): only overlap/strict adjacency, with no read
  amplification;
- `CostAware`: bounded small-gap fusion using explicit bandwidth/IOPS cost.

Fusion never crosses owner Job, source/read view, cache availability epoch or
priority bucket. DecodeOps publish their own `BlockReady` immediately after
successful decode, so one slow block in a fused read does not hide already
decoded siblings. Hard limits cap span, gap, amplification, DecodeOps, decoded
bytes and encoded staging. Parallelism hints retain a minimum task floor.

## Example

```rust,no_run
use sc_load::{
    compile, Dataset, Fill, OutputDType, OutputSpec, PlanConfig, PlanSpec,
    RowRef, SessionConfig, Source, SourceId,
};

# fn run() -> sc_load::Result<()> {
let dataset = Dataset::open("matrix.sc-compress")?;
let rows = vec![
    RowRef::new(SourceId::new(0), 7),
    RowRef::new(SourceId::new(0), 2),
];
let output = OutputSpec::new(30_000, OutputDType::F32, Fill::F32(0.0))?;
let mut config = PlanConfig::default();
config.cache_capacity_bytes = 8 * 1024 * 1024 * 1024;
let plan = compile(
    PlanSpec::new(vec![Source::new(0, dataset)], rows, output, 256, 8)
        .config(config),
)?;

let mut session = plan.open(SessionConfig::default())?;
while let Some(batch) = session.next_batch()? {
    for row in 0..batch.rows() {
        consume_row(batch.row_as::<f32>(row)?);
    }
    // Drop/release only after any asynchronous consumer (for example H2D)
    // has finished using this output generation.
}
# Ok(())
# }
# fn consume_row(_: &[f32]) {}
```

## Precision and validation

Storage and output dtypes are
`i16 | i32 | i64 | u16 | u32 | u64 | f32 | f64`. Compilation allows only
declared promotions. Potentially rounding integer-to-float conversions require
explicit opt-in. Signedness overflow follows `OverflowPolicy` (`Error`,
`UseFill`, `UseValue`, or `Unchecked`).

CSR indices are validated for bounds and strict increasing order before the
unsafe scatter kernel. Each CSR output row is zero-initialized for structural
absence; `Fill` is then written only to output columns without a mapped source
feature. Fallible conversions are validated before batch publication.
Cache/output pointer lowering validates every extent; raw pointers never enter
reusable Plan state.

## Shared ring

On Linux with lock-free 64-bit atomics, `Plan::open_shared` runs the same
private decoded-cache graph against a sealed `memfd` output ring. Rank clients
receive round-robin logical batches and hold generation leases. Cache remains
producer-private. Rank exit, owner death, cancellation and ring reuse use the
existing futex control plane and do not alter static cache dependencies.

## Profiling

The `profile` feature adds compile/runtime counters. Compile statistics include
cache residency loads/reloads, hits/misses, capacity/fragmentation stalls,
cache horizon, independent/fused I/O tasks, saved I/O operations,
payload/span/amplification, DecodeOps per task and dependency edges. Runtime
statistics report physical reads, io_uring submission/completion, peak in-flight
resources, and measured I/O-wait, decode, validation, scatter, completion and
consumer-wait time. Stage times are summed across workers and are therefore
worker-time totals rather than session wall time. Profile-only timers and
counters are compiled out of ordinary builds.

The ignored `real_scatter_bench::benchmark_real_decoded_scatter` test isolates
decoded Dense/CSR → mapped output kernels. Dataset decode and CSR densification
happen before timing. It reads `real_dataset.txt` by default and covers source
mapping ratios `1`, `1/2`, `1/5`, `1/10`, both complete output genes and `1/3`
unmapped output genes. Run it on a worker in release mode, for example:

```bash
SC_LOAD_REAL_SCATTER_ROWS=128 \
taskset -c 64 cargo test -p sc-load --release --all-features --lib \
  real_scatter_bench::benchmark_real_decoded_scatter -- \
  --exact --ignored --nocapture --test-threads=1
```
