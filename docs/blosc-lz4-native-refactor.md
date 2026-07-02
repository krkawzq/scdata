# Blosc-LZ4 Native 访问路径重构设计

更新时间: 2026-07-02

本文档描述一次面向生产吞吐的大重构: 在现有通用 `DataBank` / `access`
链路之外，新增一条专门服务 **Blosc-LZ4 + cell-aligned scdata store** 的
native fast path。目标是在不修改上游 batch / sampler 语义的前提下，把访问粒度从
zarr chunk 下推到 Blosc compressed block，避免完整读取 chunk、完整解压 chunk 和
完整 decoded scatter。

## 1. 背景与判断

### 1.1 当前链路的问题

当前 scheduled 访问链路可以概括为:

```text
sampler / batch plan
  -> DataBank scheduled planner
  -> gene projection / batch 内 chunk 访问合并
  -> access scheduler
  -> IO 读取完整 compressed chunk
  -> codec 解压完整 decoded chunk
  -> access/databank 从 decoded chunk scatter 到 batch buffer
  -> 返回 batch
```

这条链路的优势是通用: 支持多 codec、多矩阵布局、多 dtype、多种 access backend。
但真实随机 batch 下，一个 batch 往往只需要每个 chunk 里的少量 cell / gene range。
旧链路以 chunk 为最小读取和解压单位，导致:

- IO 读取完整 compressed chunk。
- CPU 解压完整 decoded chunk。
- byte-shuffle / unshuffle 对完整 block 或完整 chunk 执行。
- DataBank / access 之间有中间 buffer、调度状态和抽象开销。
- batch 内 chunk 合并能减少重复请求，但无法改变“完整 chunk”这个基本粒度。

前一轮 partial decode 已经证明瓶颈主要来自 CPU 端的完整 decode / unshuffle /
scatter。进一步要让瓶颈真正落到 IO，需要把读取也下推到 compressed block。

### 1.2 为什么选择 Blosc-LZ4 作为唯一 native codec

Blosc-LZ4 同时具备三个关键性质:

1. chunk 内部有独立 compressed blocks。
2. Blosc header 后有 block offset table，可以定位每个 compressed block。
3. LZ4 解压速度高，且每个 block 可独立解压；byte-shuffle 也可以按目标 byte range
   局部恢复。

因此 Blosc-LZ4 可以支持:

```text
目标 cell/gene -> decoded byte ranges -> touched Blosc blocks
  -> 只读 header + block table + touched compressed blocks
  -> 每个 touched block 解压一次
  -> 只 scatter 目标输出位置
```

其他 codec 不作为 native fast path:

- gzip / zlib / bz2 / lzma: 普通 stream 不适合无索引随机跳入。
- zstd: 算法本身有 seekable / frame 方案，但当前 numcodecs zstd chunk 没有可用 block
  index。
- standalone lz4: 如果外层没有 block table，也无法知道 compressed byte ranges。
- uncompressed: 可以直接 slice，但不是当前主力压缩生产格式。

本重构把 Blosc-LZ4 定义为唯一原生高度优化路径。其他格式继续走现有通用链路。

## 2. 目标与非目标

### 2.1 目标

1. **不改变上游 batch 语义**
   - batch 边界不变。
   - batch 内 cell 顺序不变。
   - sampler / dataloader 不需要知道 native backend。

2. **不依赖用户态数据 cache**
   - 生产数据集大于内存，random 访问下跨 batch 数据 cache 命中不作为性能假设。
   - native path 不设计 raw payload cache、compressed payload cache 或 decoded cache。
   - 只允许 metadata-only 的 block table index；它保存 validated offsets/ranges，不保存数据内容。
   - 连续访问的加速来自访问序列本身让当前 batch/request group 触达更少 block，而不是跨 batch
     保留 payload 后再次命中。

3. **最小化 IO 读取量**
   - 对 Blosc-LZ4，只读取 header、block table 和 touched compressed blocks。
   - 支持 load coalescing，减少 IOPS，但不强制放大到完整 chunk。

4. **最小化 CPU 无效工作**
   - 每个 touched compressed block 最多解压一次。
   - byte-shuffle 只对目标 byte ranges 做恢复。
   - sparse / dense / dtype / gene projection 热路径尽量静态展开。

5. **减少中间抽象和拷贝**
   - fused response worker 完成 decode + scatter。
   - output buffer 在 batch 开始时一次性分配。
   - loaded bytes 只在当前 completion 生命周期内以 `Arc` 或等价只读 view 被 response 阶段消费。

6. **可观测、可回退**
   - native path 有独立 metrics。
   - 不满足 native 条件时自动 fallback 到通用 DataBank path。

### 2.2 非目标

1. 不重写所有 codec。
2. 不让 native path 覆盖任意 zarr / anndata layout。
3. 不引入 raw/compressed payload cache、decoded cache 或跨 batch LRU 作为核心性能来源。
4. 不修改 Python dataloader 的 batch 行为。
5. 不在第一阶段追求跨 batch 的复杂全局调度；先保证单 batch / prefetch window 内稳定高效。

## 3. 新架构总览

当前落地方式不是另造一套 codec/backend，而是在现有模块边界内新增 native fast path:

```text
rust/scdata/src/codecs/impls/blosc/
  header.rs       # 共享 Blosc header / table parser
  lz4_fast.rs     # 共享 block plan / decode_blosc_lz4_block
  shuffle.rs      # 共享 shuffle/unshuffle helpers

rust/scdata/src/databank/native/
  metadata.rs     # NativeBloscBlockIndex / validated block ranges
  load.rs         # IoBackend 之上的 range coalescer + completion splitter
  planner.rs      # SliceSpec -> touched block requests
  executor.rs     # loaded block -> decode/unshuffle/direct scatter
  item.rs         # AccessItem native closed loop

rust/scdata/src/databank/scheduled/native_access.rs
  ScheduledBatchAccess: generic scheduled iterator / native iterator 的统一包装
```

端到端链路:

```text
Scheduled batch request
  -> NativeBatchPlanner
       - validate dataset / matrix / codec eligibility
       - resolve output dtype / gene projection
       - allocate batch output buffer
       - build per-cell output slots
       - produce BlockReadPlan + BlockConsumer list
  -> FusedExecutor
       - request side: submit block read requests to LoadModule
       - response side: consume ready loaded blocks/groups
       - decode each touched Blosc block once
       - scatter to all consumer cell buffers
       - mark per-cell pending count
  -> BatchCompletionQueue
       - return completed batch in original order

LoadModule
  -> accepts byte-range load requests
  -> coalesces adjacent/nearby ranges by policy
  -> submits positioned reads to IoPool
  -> splits loaded span into original request views
  -> publishes load completions
  -> drops loaded bytes after current completions are consumed
```

关键设计点:

- **cell 是 completion unit，不是 decode unit**。
- **compressed block / block group 是 response unit**。
- 多个 cell 在当前 batch/request group 内命中同一个 Blosc block 时，block 只读一次、解压
  一次，然后 scatter 到多个 cell；completion 结束后不保留 compressed/decoded payload。
- 如果一个 cell 跨多个 block，用 per-cell context 记录 pending block 数，全部完成后才标记 cell ready。
- **不新建第二套 Blosc 解码实现**。当前 `rust/scdata/src/codecs/impls/blosc/`
  已经有 `BloscCodec::decode_slice` 和 Blosc-LZ4 touched-block fast path；native
  path 应先把现有 `BloscHeader`、block table 解析、`decode_blosc_lz4_block`、
  split/dont-split 处理、shuffle/unshuffle 逻辑抽成可复用的 crate 内部 API。
  native path 只新增 block table index、partial read plan 和 direct scatter。

## 4. 旧链路与新链路对比

| 维度 | 旧通用链路 | 新 Blosc-LZ4 native 链路 |
|---|---|---|
| 最小 IO 单位 | zarr chunk | compressed block range，可 coalesce |
| 最小 decode 单位 | decoded chunk | Blosc block |
| decoded cache | 可选，但生产不依赖 | 不实现 decoded payload cache |
| raw / compressed payload cache | access 内部 cache | 不实现 payload cache；仅做当前请求组 coalescing/splitting |
| scatter | chunk decode 后再 slice/scatter | block decode 后直接 scatter 到 output |
| 调度层级 | databank planner + access scheduler + io/decode/fill 多池 | producer/consumer + load coalescer + fused response |
| codec 支持 | 通用 | 只原生支持 Blosc-LZ4，其他 fallback |
| 性能目标 | 通用正确性 | 生产格式极致吞吐 |

## 5. 核心数据结构

以下类型名是建议，不要求完全照搬。

### 5.1 Native eligibility

```rust
pub enum NativeEligibility {
    Supported(NativeDatasetMeta),
    Unsupported(UnsupportedReason),
}

pub enum UnsupportedReason {
    CodecNotBloscLz4,
    BitshuffleUnsupported,
    UnknownShuffleMode,
    NonZipStored,
    UnsupportedMatrixLayout,
    UnsupportedDType,
    MissingBlockMetadata,
}
```

native path 只能在明确支持时进入。所有“不确定”都 fallback。

### 5.2 Dataset / array / chunk / block metadata

```rust
pub struct NativeDatasetMeta {
    pub dataset_id: DatasetId,
    pub matrix: MatrixKind,
    pub n_obs: usize,
    pub n_vars: usize,
    pub gene_axis: GeneAxisMeta,
    pub layout: NativeMatrixLayout,
}

pub enum NativeMatrixLayout {
    Dense1D {
        data: NativeArrayMeta,
    },
    Dense2D {
        data: NativeArrayMeta,
    },
    SparseCsr {
        indices: NativeArrayMeta,
        data: NativeArrayMeta,
        indptr: Arc<[u64]>,
    },
}

pub struct NativeArrayMeta {
    pub array_id: NativeArrayId,
    pub dtype: NativeDType,
    pub shape: Arc<[usize]>,
    pub grid: NativeGridMeta,
    pub chunks: Arc<[NativeChunkMeta]>,
}

pub struct NativeChunkMeta {
    pub chunk_id: u32,
    pub fid: FileId,
    pub payload_offset: u64,
    pub payload_len: u32,
    pub decoded_size: u32,
    pub block_size: u32,
    pub typesize: u8,
    pub shuffle: BloscShuffle,
    pub block_ranges: Arc<[ValidatedBlockRange]>,
    pub row_start: usize,
    pub row_end: usize,
    pub elem_start: usize,
    pub elem_end: usize,
}

pub struct ValidatedBlockRange {
    pub payload_relative_offset: u32,
    pub compressed_len: u32,
    pub decoded_offset: u32,
    pub decoded_len: u32,
}
```

metadata 必须按 **Array** 建模，而不是只挂在 dataset 上。CSR 至少有 `indices`
和 `data` 两个 array，它们的 dtype、chunk grid、codec、block table 都可能不同；
这个结构也更贴近现有 `Dataset::{Dense1D,Dense2D,SparseCsr}` 和 `Array` / `Chunk`
抽象。

`block_ranges` 是从 Blosc block offset table 校验后得到的 range，不在热路径继续读
原始 i32 offset。Blosc payload 中 block offset 按 little-endian i32 存储，解析时必须
先校验非负、单调、位于 payload 内，再转换为 `ValidatedBlockRange`。`block_ranges`
直接带 compressed len 和 decoded range，最后一个 block 不需要每次特殊计算
`next_offset - current_offset`。

`block_ranges` 有两种来源:

1. launch / registration 阶段扫描每个 Blosc header + block table 并保存。
2. 第一次访问某 chunk 时读取 header + block table，再缓存为 metadata。

推荐第一阶段在注册阶段扫描目标 dataset 的 block table。理由:

- block table 很小。
- 后续热路径不需要先发 header read 再发 block read。
- planner 可以直接计算 compressed ranges。

如果全量注册开销过大，可以引入 lazy metadata，但那是第二阶段优化。

### 5.3 Batch output

```rust
pub struct NativeBatch {
    pub batch_id: u64,
    pub cells: Arc<[CellRequest]>,
    pub output: OutputBuffer,
    pub cell_states: Box<[CellState]>,
}

pub struct OutputBuffer {
    pub ptr: NonNull<u8>,
    pub len: usize,
    pub dtype: NativeDType,
    pub shape: [usize; 2],
}

pub struct CellState {
    pub output_offset: usize,
    pub output_len: usize,
    pub pending_blocks: AtomicU32,
    pub status: AtomicCellStatus,
}
```

batch 一开始一次性分配完整 output buffer。每个 cell 只拿自己的写入范围。

### 5.4 Block read plan

```rust
pub struct BlockReadPlan {
    pub request_id: LoadRequestId,
    pub key: BlockKey,
    pub fid: FileId,
    pub offset: u64,
    pub len: u32,
    pub block_meta: BloscBlockMeta,
    pub consumers: SmallVec<[BlockConsumer; 4]>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockKey {
    pub dataset_id: DatasetId,
    pub chunk_id: u32,
    pub block_id: u32,
}

pub struct BlockConsumer {
    pub cell_index: u32,
    pub decoded_range: Range<u32>,
    pub output_range: Range<u32>,
    pub gene_projection: GeneProjectionRef,
}
```

`consumers` 是性能关键。它让一个 decoded block 可以服务多个 cell，避免 cell 级 worker
重复解压同一个 block。

### 5.5 Load request / completion

```rust
pub struct LoadRequest {
    pub id: LoadRequestId,
    pub fid: FileId,
    pub offset: u64,
    pub len: u32,
    pub priority: LoadPriority,
}

pub struct LoadCompletion {
    pub id: LoadRequestId,
    pub bytes: Arc<[u8]>,
    pub range_in_bytes: Range<usize>,
}

pub struct CoalescedRead {
    pub fid: FileId,
    pub offset: u64,
    pub len: u32,
    pub children: SmallVec<[LoadRequestId; 8]>,
}
```

load completion 不应该在生产默认无上限。benchmark mode 可以关闭上限以便测峰值，
但 production 默认必须受 native memory budget 或 completion byte budget 约束。达到
hard limit 时返回明确错误；进入 soft limit/backpressure 时必须记录 metrics，不能让
系统“静默卡住”。

## 6. 线程与队列模型

### 6.1 顶层模型

```text
NativeBatchProducer
  -> FusedExecutor queue

FusedExecutor workers
  request side:
    - pull batch/cell/block planning work
    - submit LoadRequest to LoadModule
    - store BlockContext / CellContext

  response side:
    - pull LoadCompletion
    - decode Blosc block
    - scatter to consumers
    - decrement cell pending count
    - publish completed cell/batch

LoadModule scheduler workers
  - receive LoadRequest
  - collapse duplicate ranges within the current request group
  - coalesce by fid / offset / gap policy
  - submit CoalescedRead to IoPool

IoPool workers
  - positioned read
  - return coalesced bytes

LoadModule completion workers
  - split coalesced bytes to children
  - publish LoadCompletion
```

### 6.2 Fused executor 调度策略

每个 fused worker 循环:

```text
loop:
  while request backlog within prefetch_step is not drained:
      plan/submit more block load requests

  if load completion exists:
      decode/scatter one ready block or block group
      continue

  if completion queue has batch ready:
      publish response
      continue

  park / wait
```

调度原则:

- request side 优先保证 prefetch window 足够深，避免 IO 空转。
- response side 以 loaded block/group 为单位执行，避免重复 decode。
- worker 不持有长锁做 decode。锁只用于领取 work、发布 completion、更新轻量状态。
- 对同一个 output cell 的多个 block 写入必须是 disjoint ranges，或通过规划保证无数据竞争。

### 6.3 为什么不以 cell 作为 decode unit

严格 cell-fused 模式会让每个 worker “拿一个 cell 并完成全部任务”。这个模型简单，但在
随机 batch 中多个 cell 经常命中同一个 chunk/block。如果每个 cell 单独解压 block，会出现:

```text
cell A needs block 10 -> decode block 10
cell B needs block 10 -> decode block 10 again
cell C needs block 10 -> decode block 10 again
```

这会把 CPU 瓶颈重新引入。

推荐模型是:

```text
block 10 loaded once
  -> decode once
  -> scatter to cell A/B/C output ranges
  -> A/B/C 各自 pending count -1
```

因此 cell 是完成状态单位，block 是解压执行单位。

## 7. Load 模块设计

### 7.1 职责边界

LoadModule 只做一件事: byte-range load。它不是新的底层 IO backend，而是现有
`IoPool` / `IoBackend::submit_read(file, offset, len, priority)` 之上的 range
coalescer + completion splitter。

它负责:

- 接收 `(fid, offset, len)` 请求。
- 在当前提交的 request group 内合并完全相同或重叠 byte ranges。
- coalescing: 把同文件相邻或近邻范围合并成较大 positioned read。
- 调用 IoPool 执行实际 IO。
- 将 coalesced read 的结果切分成原始 request view。
- 把 completion 放入完成队列。
- completion 被消费后不保留 loaded payload。

它不负责:

- 不理解 cell / gene / CSR。
- 不解压。
- 不 scatter。
- 不做 raw/compressed payload cache。
- 不做 decoded cache。
- 不做跨 batch cache 管理。
- 不提供单独 prefetch API。

### 7.2 Coalescing 策略

配置项建议:

```rust
pub struct LoadCoalesceConfig {
    pub max_window_us: u32,
    pub max_merged_len: u32,
    pub max_gap_bytes: u32,
    pub max_waste_ratio: f32,
    pub min_children: u16,
}
```

合并判断:

```text
ranges sorted by (fid, offset)

candidate merged range:
  merged_len = last_end - first_start
  useful_len = sum(child.len)
  gap_len = merged_len - useful_len
  waste_ratio = gap_len / merged_len

merge if:
  same fid
  merged_len <= max_merged_len
  each gap <= max_gap_bytes
  waste_ratio <= max_waste_ratio
  within max_window_us
```

默认策略应该偏保守:

- 顺序或局部访问自然合并，降低 IOPS。
- 近似随机访问不强行放大读取，避免重新退化成 chunk-level IO。

### 7.3 完成队列与内存预算

benchmark mode 可以配置 completion queue 无上限，用于观察 load/coalescing 的理论峰值。
production 默认必须有上限，推荐按 byte budget 控制，而不是只按 completion 数量控制:

```text
native_memory_budget_bytes =
  output buffers
  + queued compressed bytes
  + decode scratch reservation
  + planner/context overhead
```

如果触及 hard limit:

```text
if completion_queue_bytes > hard_limit:
    return NativeError::LoadCompletionQueueOverflow
```

如果触及 soft limit，应对 request side 施加 backpressure，并记录:

- `native_backpressure_count`
- `native_backpressure_wait_ns`
- `native_completion_queue_bytes`
- `native_completion_queue_bytes_max`

不要默默阻塞。静默阻塞会让 profile 难以解释，也容易在生产中表现为无法定位的吞吐抖动。

## 8. Native planner 设计

### 8.1 输入

输入仍然来自现有 scheduled API:

```text
dataset ids
CellIndexPlan / batch cell order
genes / missing policy / dtype
ScheduledPrefetchConfig
```

native planner 必须保持:

- batch 内 cell 顺序。
- 输出 shape。
- dtype 转换语义。
- gene projection / missing zero 语义。

### 8.2 输出

planner 输出:

```text
NativeBatchPlan
  output allocation spec
  cell contexts
  block read plans
  completion dependencies
```

### 8.3 Sparse CSR path

CSR 访问不能只看成“一次性静态 plan indices + data”。对于 projected sparse，
是否需要读取某个 `data` block 取决于先解出来的 `indices` 内容；当前通用路径也区分
`ReadAll` 与 `SelectedOnly`。native path 应把 CSR 拆成两个阶段。

#### 8.3.1 Phase 1: ReadAll CSR

第一阶段建议先实现 ReadAll，简单且正确:

1. 根据 `indptr[cell]..indptr[cell+1]` 得到该 cell 的 nonzero element range。
2. 定位该 range 覆盖哪些 `indices` chunk/block 和 `data` chunk/block。
3. 对 `indices` 和 `data` 分别生成 block read plan。
4. response 阶段解压 indices/data block 后，执行 gene projection:
   - native gene order: 直接 scatter。
   - external gene list: 使用预构建 mapping。
   - missing gene: zero fill 已在 output buffer 初始化阶段完成。

ReadAll 会多读一些 `data`，但不会依赖 index decode 后的动态调度，适合作为 native
CSR 的第一条 correctness path。

#### 8.3.2 Phase 2: SelectedOnly CSR

SelectedOnly 的目标是只读取最终会被 gene projection 选中的 `data`。它需要动态依赖:

1. batch planner 先根据 `indptr` 生成 `indices` block read plan。
2. index block ready 后，response worker 解码 indices。
3. 根据 gene projection 判断哪些 nonzero 被选中。
4. 动态生成对应 `data` block read requests。
5. `data` block ready 后再 decode + scatter。

因此 `pending_blocks` 不能假设只在 batch plan 阶段静态确定。需要支持:

```text
on index consumer planned:
    cell.pending += 1

on index block decoded:
    selected data ranges = project(indices)
    for each newly required data block:
        cell.pending += 1
        submit LoadRequest
    cell.pending -= 1

on data block scattered:
    cell.pending -= 1
```

为了避免 data block 重复提交，需要在当前请求图内按 `BlockKey` 折叠 duplicate block
request，并维护可追加 consumer 列表。一个 index block 可能发现多个 cell 需要同一个
data block；该 data block 在当前 batch 请求图内只能读取和解压一次，completion
消费后不保留 payload。

CSR 的高效实现应避免每个 nonzero 做复杂动态查找。推荐按策略静态展开:

```text
SparseCsrKernel<
  StoredDType,
  OutputDType,
  IndexDType,
  GeneMode,
  MissingPolicy,
>
```

第一阶段可以先用 match 分发到有限组合，后续再做 macro / trait specialization。

### 8.4 Dense path

dense1d / dense2d path 更简单:

- 计算 cell row 对应 decoded byte ranges。
- 根据 range 计算 touched Blosc blocks。
- block response 阶段直接 copy / cast 到 output。

dense path 同样应保持 block-level decode once + multi-consumer scatter。

### 8.5 Gene mapping

gene mapping 不应在热路径动态构建。

注册或 batch setup 阶段准备:

```rust
pub enum GeneProjectionPlan {
    Identity,
    Take {
        source_to_output: Arc<[u32]>,
    },
    MissingZero {
        source_to_output: Arc<[i32]>, // -1 means missing
    },
}
```

对于 CSR:

- 如果 output gene list 是 external names，预先构建 source gene id 到 output column 的映射。
- response scatter 时只做数组查表和范围判断。

对于 dense:

- identity 可以整段 copy。
- contiguous take 可以 range copy。
- arbitrary take 才逐列 gather。

## 9. Blosc-LZ4 block decode 设计

native path 必须复用现有 Blosc fast path 的实现语义，不另写一套“看起来等价”的解码器。
现有 `lz4_fast.rs` 已处理:

- `BLOSC_MEMCPYED`。
- Blosc-LZ4 format/version 校验。
- bitshuffle fallback。
- byte-shuffle 与 range-only unshuffle。
- `dont_split` 与 split blocks。
- split size 校验和负数 offset/size 校验。

重构时应先把这些私有函数整理成可复用内部 API，例如:

```rust
pub(crate) struct BloscLz4Plan { ... }
pub(crate) struct BloscLz4BlockRange { ... }

pub(crate) fn parse_blosc_lz4_plan(...) -> CodecResult<BloscLz4Plan>;
pub(crate) fn decode_blosc_lz4_block_into(...) -> CodecResult<()>;
pub(crate) fn copy_shuffled_ranges(...) -> CodecResult<()>;
```

native path 调用这些 API 来做 partial read/direct scatter。这样 `BloscCodec::decode_slice`
和 native backend 共享同一套 header/block/split/shuffle 逻辑，避免 bug 修复和边界条件
在两套实现之间漂移。

### 9.1 Metadata 解析

Blosc payload:

```text
header
block offset table
compressed blocks
```

native metadata 需要保存:

- `decoded_size`
- `compressed_size`
- `blocksize`
- `typesize`
- `shuffle flags`
- `nblocks`
- 每个 block 的 compressed offset / len

校验必须严格:

- block offset 不得落入 header/table 内。
- block offset 单调递增。
- block compressed range 不得越过 payload。
- `typesize > 0` 且在 Blosc 最大 typesize 范围内。
- bitshuffle 暂不进入 native fast path，先 fallback。
- block offset 原始格式是 little-endian i32，必须校验非负后才能转为 `u32`。
- `BLOSC_MEMCPYED` 可以作为特殊 native path: compressed payload 实际是原始 decoded
  bytes，直接根据 decoded ranges 生成 byte-range reads；如果实现复杂度不划算，第一阶段
  明确 fallback。
- `dont_split` 和 split blocks 必须沿用现有 `decode_blosc_lz4_block` / `split_size`
  语义。native block read 的 compressed range 是一个 Blosc block，block 内可能还有多个
  split，每个 split 前有 compressed size 前缀。

### 9.2 Partial IO

对于 touched block:

```text
absolute_offset = chunk.payload_offset + block.relative_offset
len = block.compressed_len
```

生成 `LoadRequest(fid, absolute_offset, len)`。

如果 block table 未预加载:

1. 先读 Blosc fixed header + enough bytes for table。
2. 解析 table。
3. 再提交 block reads。

第一阶段推荐预加载 table，减少两阶段 IO。

### 9.3 Partial decode

response 阶段:

```text
loaded compressed block
  -> lz4 decode into thread-local block scratch
  -> if byte-shuffled:
       copy only requested byte ranges from shuffled planes to output
     else:
       copy requested decoded ranges to output
```

如果一个 block 的 consumers 覆盖率很高，可以选择整块 unshuffle 后 scatter，避免大量小范围
操作。策略:

```text
if requested_bytes / block_size >= unshuffle_full_threshold:
    full block unshuffle
else:
    range-only unshuffle
```

这个阈值应通过 benchmark 校准。

## 10. 内存与生命周期

### 10.1 Output buffer

batch 开始时分配:

```text
batch_size * n_output_genes * output_dtype_size
```

missing zero 情况下，默认先 zero-fill 整个 output buffer。后续 scatter 只写非零或已存在值。

### 10.2 Scratch buffer

每个 fused worker 持有 thread-local scratch:

- compressed block view: 从 `Arc<[u8]>` 借用，不复制。
- decoded block scratch: 最大 `blocksize`。
- optional unshuffle scratch: 只在需要整块 unshuffle 时使用。
- small temporary arrays: 用于收集 load instructions，减少持锁时间。

### 10.3 Context 生命周期

```text
BatchContext
  owns output buffer
  owns CellState[]
  owns BlockContext map

BlockContext
  references BatchContext
  owns consumers
  waits for LoadCompletion

LoadCompletion
  owns Arc bytes
  consumed by response worker
```

完成条件:

- 所有 block contexts 完成。
- 所有 cell pending count 归零。
- batch 按原 scheduled order 发布。

## 11. 并发正确性

### 11.1 输出写入安全

planner 必须保证任意两个 block consumers 写入 output buffer 的范围满足:

```text
same cell + same output range:
  only allowed if values are identical and operation idempotent
otherwise:
  ranges must be disjoint
```

CSR 情况下，一个 gene 在同一 cell 中理论上不应重复。如果输入数据存在重复 indices:

- 第一阶段可保持现有语义: 按原数据顺序写，或 fallback。
- 更稳妥做法: native eligibility 检测到可能重复时 fallback。

### 11.2 Cell completion

每个 cell 有 `pending_blocks`:

```text
on block consumer planned:
    pending_blocks += 1

on block scatter done:
    if pending_blocks.fetch_sub(1) == 1:
        mark cell complete
```

batch completion 需要保持 batch 顺序。可以:

- batch 内所有 cell complete 后发布 batch。
- 或 cell complete 入内部队列，但对外仍按 batch 聚合返回。

### 11.3 Cancellation

scheduled iterator drop / batch cancellation 时:

- 未提交 load request: 直接丢弃。
- 已提交未完成 load request: 标记 batch cancelled，completion 到达后丢弃。
- 正在 decode/scatter: 允许完成当前 block，然后检查 cancelled。

不要要求 LoadModule 支持强取消底层 IO。底层 IO 取消复杂，收益有限。

## 12. 配置建议

新增配置建议:

```rust
pub struct NativeAccessConfig {
    pub enabled: bool,
    pub fallback_to_generic: bool,
    pub fused_workers: usize,
    pub request_prefetch_batches: usize,
    pub request_prefetch_blocks: usize,
    pub memory_budget_bytes: usize,
    pub response_queue_bytes_soft_limit: usize,
    pub response_queue_bytes_hard_limit: usize,
    pub load: LoadConfig,
    pub blosc: NativeBloscConfig,
}

pub struct LoadConfig {
    pub scheduler_workers: usize,
    pub io_workers: usize,
    pub coalesce: LoadCoalesceConfig,
}

pub struct NativeBloscConfig {
    pub preload_block_tables: bool,
    pub full_unshuffle_threshold: f32,
    pub max_block_size: usize,
}
```

配置落地清单:

1. Rust `DataBankConfig` 增加 `native_config: NativeAccessConfig`。
2. Rust `ScheduledPrefetchConfig` 增加 `native_mode: NativeMode`:

   ```rust
   pub enum NativeMode {
       Disabled,
       Auto,
       Force,
   }
   ```

3. pybind 增加 `_NativeAccessConfig`、`_NativeLoadConfig`、`_NativeBloscConfig`、
   `_NativeMode` 或 string setter。
4. Python `DataBankConfig` 增加 `native_config: NativeAccessConfig`，并把这些 dataclass
   加入 `_CONFIG_CLASSES` / `_RUST_CONFIG_TYPES`，保持现有反射转换路径。
5. Python `ScheduledPrefetchConfig` 增加 `native_mode: Literal["disabled", "auto", "force"]`。
6. 语义:
   - `disabled`: 永远走 generic path。
   - `auto`: eligible 时走 native；不 eligible 或 native planner 拒绝时 fallback。
   - `force`: 必须走 native；不 eligible 或 native 执行失败直接报错，便于测试覆盖。

Python 侧可以先暴露最小参数:

```python
DataBankConfig.make(
    native__enabled=True,
    native__fused_workers=...,
    native__memory_budget_bytes=...,
    native__load__io_workers=...,
    native__load__coalesce__max_gap_bytes=...,
    native__load__coalesce__max_waste_ratio=...,
)

ScheduledPrefetchConfig(
    native_mode="auto",
)
```

默认:

- `native__enabled=False`，直到功能稳定。
- `native_mode="disabled"`，直到功能稳定。
- 稳定后可考虑 `native__enabled=True` + `native_mode="auto"`。

## 13. Metrics 与 profiling

必须新增 native profile scopes，至少包括:

### 13.1 Planner

- `native_plan_batches`
- `native_plan_cells`
- `native_plan_blocks`
- `native_plan_consumers`
- `native_plan_work_ns`
- `native_fallback_count`
- `native_fallback_reason_*`

### 13.2 Load

- `native_load_requests`
- `native_load_coalesced_reads`
- `native_load_requested_bytes`
- `native_load_read_bytes`
- `native_load_waste_bytes`
- `native_load_waste_ratio`
- `native_load_queue_wait_ns`
- `native_load_io_work_ns`
- `native_load_completion_queue_len_max`
- `native_load_completion_queue_bytes`
- `native_load_completion_queue_bytes_max`
- `native_load_backpressure_count`
- `native_load_backpressure_wait_ns`

### 13.3 Decode / scatter

- `native_blocks_decoded`
- `native_block_decode_bytes_in`
- `native_block_decode_bytes_out`
- `native_block_decode_work_ns`
- `native_unshuffle_range_bytes`
- `native_unshuffle_full_blocks`
- `native_scatter_consumers`
- `native_scatter_work_ns`

### 13.4 Completion

- `native_cells_completed`
- `native_batches_completed`
- `native_batch_latency_ns`
- `native_response_queue_wait_ns`

这些指标必须能回答:

1. 读了多少 compressed bytes。
2. coalescing 放大了多少 IO。
3. 每个 block 是否只解压一次。
4. CPU 时间花在 decode、unshuffle 还是 scatter。
5. 当前瓶颈是 IO、load scheduler、decode 还是 output scatter。

## 14. 分阶段实施计划

### Phase 0: 基线与接口门控

目标:

- 增加 native config 和 eligibility 检测。
- 默认 disabled。
- 不改变现有链路。

产物:

- `NativeEligibility`
- dataset 注册时识别 Blosc-LZ4 + supported matrix layout。
- profile 中记录 native eligible dataset 数。

验收:

- 所有现有测试通过。
- native disabled 时性能和行为不变。

### Phase 1: Blosc metadata index

目标:

- 为 eligible arrays/chunks 建立 Blosc block table index。
- 严格校验 header/table。

产物:

- `NativeArrayMeta`
- `NativeChunkMeta`
- `ValidatedBlockRange`
- `BloscBlockMeta`
- 单元测试覆盖合法/非法 Blosc payload。

验收:

- 能从真实 `.zarr.zip` 扫描所有 eligible chunk 的 block metadata。
- 失败时能给出明确 fallback reason。

### Phase 2: LoadModule

目标:

- 实现 byte-range load + 当前请求组内 duplicate range collapse + coalescing + completion queue。
- 明确不实现 raw/compressed payload cache、decoded cache 或跨 batch cache 管理。

产物:

- `LoadRequest`
- `CoalescedRead`
- `LoadCompletion`
- coalescing benchmark。

验收:

- 随机 ranges 不明显放大 IO。
- 顺序 ranges 可以合并并降低 IOPS。
- completion hard-limit overflow 按配置报错，并记录 backpressure metrics。

### Phase 3: Native batch planner

目标:

- 从 scheduled batch 生成 `NativeBatchPlan`。
- sparse CSR ReadAll / dense path 先覆盖主力组合。

产物:

- output buffer allocation。
- per-cell context。
- block consumers。
- gene projection plan。

验收:

- 对同一 batch，native plan 的 output 与通用 DataBank 完全一致。
- plan 阶段统计 block reuse ratio。
- CSR ReadAll 与现有 generic path 对齐。
- SelectedOnly 先不强行进入 Phase 3，避免把动态 data dependencies 混进第一版。

### Phase 3b: CSR SelectedOnly 动态依赖

目标:

- index block decode 后动态生成 data block requests。
- 支持 pending dependency 动态增加。

验收:

- 与现有 `projected_sparse_data_strategy="selected_only"` 输出一致。
- profile 中能看到 selected data bytes 小于 ReadAll data bytes。
- 同一 data block 被多个 cell 发现时仍只读取/解压一次。

### Phase 4: Fused response executor

目标:

- block-level response worker 完成 decode + scatter。
- 每个 touched block 解压一次。

产物:

- fused worker pool。
- thread-local decode scratch。
- block completion -> cell pending -> batch completion。

验收:

- correctness tests 覆盖 dense / sparse / dtype / gene projection。
- profile 证明 `blocks_decoded <= unique_touched_blocks`。

### Phase 5: Scheduled API 集成

目标:

- `DataBank.prefetch_indexed` 在 native enabled 且 eligible 时走 native path。
- 不 eligible 或出错可 fallback 到 generic path。

验收:

- Python API 无语义变化。
- batch order、dtype、genes、missing zero 全一致。
- cancellation / iterator drop 不泄露线程或内存。

### Phase 6: 性能压测与调参

目标:

- 在真实 worker 上达到或超过当前 partial-decode 路径。
- 在 raw/compressed payload cache / decoded cache 都不存在的 random sparse 场景下，把端到端吞吐推进到
  IO 带宽或 IOPS 上限。
- 另外建立 **IO 外模块压力测试**：绕过 GPFS 真实带宽限制，用仿真 IO 源喂给 native
  planner / load completion / decode / scatter / completion queue，验证 CPU 与调度层可以承受
  50-100 GB/s 等效输入。

#### Phase 6 性能目标

主目标使用 batch128、random、sparse CSR、`u16 data + i32 indices`、`gene_mode=first`、
`genes=4096`、`projected_sparse_data_strategy=selected_only`。上游 batch 行为不改，线程数不做
人为限制；当前 worker 可使用 96 CPU。`genes=512` 只能作为快速诊断或历史对照，不进入最终
验收，因为过小的 gene projection 会低估 selected-only、scatter 和 completion 路径压力。

真实 IO 目标:

| 等效 IO 带宽 | random sparse batch128 目标 | 说明 |
|---:|---:|---|
| 10 GB/s | 400 batch/s | 目标 IO budget 约 25 MB/batch |
| 20 GB/s | 800 batch/s | 目标 IO budget 同样约 25 MB/batch |

这两个目标反推出同一个工程约束:

```text
target_io_bytes_per_batch <= bandwidth / target_batch_s
10 GB/s / 400 batch/s = 25 MB/batch
20 GB/s / 800 batch/s = 25 MB/batch
```

历史 `genes=512` exact CSR/Blosc 形状统计中，batch128 random sparse 的 block 取整后
compressed IO 约为 `30.38 MiB/batch`。正式目标必须按 `genes=4096` 重新统计
`nnz/cell`、selected gene overlap、data group hit rate、block-rounded compressed bytes 和
coalesced read ops。无论统计结果如何，400/800 batch/s 的工程目标都不下调；如果
`genes=4096` 下的 IO budget 超过 25 MB/batch，就必须继续降低每 batch 的实际 IO bytes，例如:

- 减少重复 header/table metadata read。
- 预构建或缓存 metadata-only 的 block table index；它只保存 offset/range，不保存 payload。
- 改进 selected-only data group 粒度，避免“group 内任一 gene 命中就读完整 group row-slice”。
- 在当前 batch/request group 内折叠重复 block request，避免当前请求组内重复读/解压同一
  Blosc block；完成后不保留 block payload。
- coalescing 只能降低 IOPS，不能把 IO bytes 放大到重新接近 chunk-level。

IOPS 目标:

```text
io_ops_per_batch <= iops / target_batch_s
100kops / 400 batch/s = 250 ops/batch
200kops / 800 batch/s = 250 ops/batch
```

因此 400/800 batch/s 同时要求 `io_bytes_per_batch <= 25 MB` 且
`io_ops_per_batch <= 250`。如果只有 100kops，而目标是 800 batch/s，则必须把
`io_ops_per_batch` 降到 125 以下，否则 IOPS 会先成为瓶颈。

连续访问目标:

当前 Blosc block 的 decoded span 仍然大于单个 cell 的 CSR span，因此相邻 cell 很可能命中
同一个 compressed block。访问越连续，block-level read/decode 复用越高，实际 IO bytes/batch
应该下降，吞吐应随连续度上升。

连续访问数据构造使用一个 Markov-style cell stream:

```text
current = random valid start cell
for each next cell:
    emit current
    with probability p, set current = current + 1 if still in the same valid cell range
    otherwise set current = random valid start cell
    batch = consecutive groups of 128 cells from this generated stream
```

边界条件:

- 到达 dataset/chunk/cell range 末尾时强制重新随机选 start。
- 目标 run 仍使用 `genes=4096`、batch128、sparse CSR、selected-only。
- `p=0` 近似 random；`p=1` 是完全顺序访问。
- 不考虑边界截断时，期望连续 run length 为 `1 / (1 - p)`。
- 允许 block 在当前 batch/request group 内做 shared response；
  这只是一次请求图内的去重/多 consumer，不是 cache，也不能依赖长期 raw/compressed/decoded
  payload 命中。

验收时不要直接把 `p` 当横轴。`p` 只负责构造访问序列，实际连续度由 block IO 模型计算:

```text
bytes(p) = 同一测试窗口内，block-level 去重/合并后每 batch 的有效 IO bytes
speed_model(p) = 1 / bytes(p)
continuity(p) = (speed_model(p) - speed_model(0)) / (speed_model(1) - speed_model(0))
continuity(p) clipped to [0, 1]
```

这样定义的连续度反映真实 block 复用，而不是假设 `p` 与 IO 降幅线性。**20 GB/s
sequential / continuity 指标必须在真实 GPFS 数据路径下验收**，synthetic 20 GB/s 只能作为
host 热路径诊断，不能替代最终 GPFS 结果。主测数据约
`12 KiB/cell` CSR raw bytes；按保守 3x 压缩率估计，顺序访问的 useful compressed lower bound
约为 `4 KiB/cell`。考虑 block 边界、indices discovery、selected data group、coalescing waste 和
调度开销，20 GB/s 下先设一个可验收 baseline 和一个优化目标:

```text
baseline_batch_s_20g(p) = 800 + (8000 - 800) * continuity(p)
target_batch_s_20g(p)   = 800 + (16000 - 800) * continuity(p)
```

端点目标:

| 访问模式 | continuity | 真实 GPFS 20 GB/s baseline | 真实 GPFS 20 GB/s target | baseline budget | target budget |
|---|---:|---:|---:|---:|---:|
| random / p=0 | 0.0 | 800 batch/s | 800 batch/s | 25 MB/batch | 25 MB/batch |
| partial continuous / 0<p<1 | computed | `800 + 7200 * continuity` | `800 + 15200 * continuity` | `20 GB/s / baseline_batch_s` | `20 GB/s / target_batch_s` |
| fully sequential / p=1 | 1.0 | 8000 batch/s | 16000 batch/s | 2.5 MB/batch | 1.25 MB/batch |

按 batch128 折算，fully sequential baseline `8000 batch/s` 等价于约 `19.5 KiB/cell`
effective IO，target `16000 batch/s` 等价于约 `9.8 KiB/cell` effective IO。两者都仍高于
`4 KiB/cell` 的保守 useful compressed lower bound，因此不是理论极限，而是阶段性工程目标。

IO 外 synthetic 顺序访问目标从真实 GPFS 20 GB/s sequential 目标线性外推，只用于证明
CPU/decode/scatter/completion 在更高等效带宽下不会先饱和:

| 验收路径 | 等效 IO 带宽 | fully sequential baseline | fully sequential target | baseline budget | target budget |
|---|---:|---:|---:|---:|---:|
| real GPFS | 20 GB/s | 8000 batch/s | 16000 batch/s | 2.5 MB/batch | 1.25 MB/batch |
| synthetic host-path | 50 GB/s | 20000 batch/s | 40000 batch/s | 2.5 MB/batch | 1.25 MB/batch |
| synthetic host-path | 100 GB/s | 40000 batch/s | 80000 batch/s | 2.5 MB/batch | 1.25 MB/batch |

连续性 sweep 的验收要求:

- `p` 建议覆盖 `0, 0.25, 0.5, 0.75, 0.9, 0.97, 0.99, 1.0`。
- 报告每个点的 `p`、平均 run length、`bytes(p)`、`continuity(p)`、batch/s、
  `native_load_requested_bytes`、`native_load_read_bytes`、block reuse ratio。
- 真实 GPFS 20 GB/s 数据路径下，吞吐随 `continuity(p)` 单调不下降；如有测量噪声，必须通过
  重复 run 或置信区间证明不是系统性退化。
- `batch/s` 对 `continuity` 做线性拟合时至少达到 baseline 线，并逐步逼近 target 线；
  偏离点必须能由 IOPS、coalescing waste、queue backpressure 或 CPU decode/scatter 饱和解释。
- 真实 GPFS 20 GB/s 下完全顺序访问必须至少达到 8000 batch/s，优化目标是
  16000 batch/s；否则说明
  block response sharing、coalescing、scatter 或 completion queue 仍没有吃到相邻 cell 的
  block 复用收益。

IO 外模块 random 抗压目标:

| 仿真等效 IO 带宽 | 目标吞吐 | 等效 budget |
|---:|---:|---:|
| 50 GB/s | 2000 batch/s | 25 MB/batch |
| 100 GB/s | 4000 batch/s | 25 MB/batch |

这些 random 压力测试不是为了验证 GPFS，而是验证 IO 之外的模块不会在 50-100 GB/s 等效输入下
成为瓶颈。fully sequential 使用上一节的顺序访问目标表: 50 GB/s baseline/target 为
20000/40000 batch/s，100 GB/s baseline/target 为 40000/80000 batch/s。

### 14.3 真实 GPFS 与 synthetic 的交叉校准

synthetic 不能只追求高 batch/s。所有 50/100 GB/s synthetic 目标都必须先用真实 GPFS
20 GB/s 目标点校准，否则容易把缺失的 DataBank scheduled assembly、CSR 二阶段请求、
completion queue、IOPS/queue wait 或真实 block 分布误判为“host 已达标”。

校准流程:

1. 先在真实 GPFS 上跑 `genes=4096`、batch128、同一 order、同一 coalescing 策略。
2. 记录 `read_bytes_per_batch`、`io_ops_per_batch`、block reuse、coalescing waste、
   output dtype、avg parts/batch、native queue/profile 指标。
3. synthetic 20 GB/s run 必须复现这些统计，再外推到 50/100 GB/s。
4. 如果 synthetic 的 `read_bytes_per_batch`、`io_ops_per_batch` 或 block reuse 明显偏离真实
   GPFS，则该 synthetic 结果只算诊断，不算验收。

建议阈值:

- `read_bytes_per_batch`: synthetic 与真实 GPFS 差异不超过 10-15%。
- `io_ops_per_batch`: 差异不超过 15-20%，并解释 coalescing 参数差异。
- block reuse / unique blocks: 绝对差异不超过 0.05-0.10，或给出真实 trace 证据。
- 在同一等效 20 GB/s 下，synthetic normalized batch/s 不应高于真实 GPFS normalized
  batch/s 超过 25-35%；超过这个范围时，必须定位并补齐 synthetic 缺失路径。

当前建议把 synthetic 从 hand-shaped workload 升级为 **trace-driven replay**:

```text
real GPFS native planner
  -> record NativeLoadRequest / coalesced read / block consumer trace
  -> replay same trace through virtual IO bandwidth limiter
  -> keep real decode / unshuffle / scatter / batch completion path
```

更进一步，应把 virtual IO backend 接到完整 DataBank scheduled native path，而不是只跑
standalone native block harness。只有这样，synthetic 才能证明 50/100 GB/s 下 IO 外模块不会
先于 IO 饱和。

达标定义:

- synthetic loader 按真实数据形状生成 completion，而不是返回单一固定大小 buffer。
- CSR nnz/cell、selected gene overlap、Blosc block size、compressed block size 分布、block reuse
  ratio、coalescing waste、metadata/table 命中率都要来自真实数据统计或显式配置。
- 输出 buffer 分配、decode scratch、unshuffle、scatter、batch completion queue 都走真实代码路径。
- benchmark-only 可以使用内存 resident compressed block payload 或确定性 pseudo payload
  代替 GPFS read，但不能跳过 decode/scatter
  热路径。
- 报告 `native_decode_work_ns`、`native_scatter_work_ns`、`native_completion_queue_wait_ns`、
  `native_backpressure_*`，证明瓶颈仍在仿真 bandwidth，而不是 CPU/锁/队列。
- random: 50 GB/s 下至少顶住 2000 batch/s；100 GB/s 下目标 4000 batch/s。
- fully sequential: 50 GB/s 下至少顶住 20000 batch/s，目标 40000 batch/s；100 GB/s 下至少顶住
  40000 batch/s，目标 80000 batch/s。

压测矩阵:

```text
batch_size: 64 / 128 / 256
order: random / continuity-p / fully sequential / grouped / k-chunk random
continuation_p: 0 / 0.25 / 0.5 / 0.75 / 0.9 / 0.97 / 0.99 / 1.0
genes: 4096 for target runs; 2048 / all only for diagnostic sweeps
datasets: 32 / 64 / full
coalesce max_waste_ratio: 0.05 / 0.10 / 0.25 / 0.50
coalesce max_gap_bytes: 0 / 4K / 16K / 64K
threads: 96 on current worker; no artificial cap below available CPU for target runs
io model: real GPFS / synthetic 50 GB/s / synthetic 100 GB/s
```

验收指标:

- 不实现、不依赖 decoded cache。
- 不实现、不依赖 raw/compressed payload cache；random 目标不能由跨 batch cache 命中贡献。
- `native_load_read_bytes / native_load_requested_bytes` 可解释。
- real GPFS: `genes=4096` batch128 random sparse 在 10 GB/s 等效带宽下达到 400 batch/s，
  在 20 GB/s 等效带宽下达到 800 batch/s。
- real GPFS: `genes=4096` batch128 sparse 在 20 GB/s 等效带宽下，吞吐随
  block-derived continuity 从 800 batch/s 单调、近线性上升；fully sequential 基线为
  8000 batch/s，优化目标为 16000 batch/s。
- synthetic 20 GB/s 校准: 同一 order 下 `read_bytes_per_batch`、`io_ops_per_batch`、
  block reuse 和 normalized batch/s 必须与真实 GPFS 相互印证；未通过校准的 synthetic
  只能作为诊断，不作为验收。
- synthetic IO random: `genes=4096` shape 下 IO 外模块能承受 50 GB/s / 2000 batch/s，
  目标 100 GB/s / 4000 batch/s。
- synthetic IO fully sequential: `genes=4096` shape 下 50 GB/s baseline/target =
  20000/40000 batch/s，100 GB/s baseline/target = 40000/80000 batch/s。
- profile 显示瓶颈主要在 IO bandwidth 或 synthetic bandwidth limiter，而不是 host CPU、
  lock contention、queue backpressure、decode/unshuffle/scatter。

基准环境必须固定记录，避免把实现退化和机器/参数差异混在一起:

```text
worker:
  ssh -CAXY ms-0701-210140-1018388-nknq4.wangzhongqi.ailab-ai4ls.pod@h.pjlab.org.cn
  effective CPU: 96
  memory: ~781 GiB
  target runs: use all available CPU unless explicitly sweeping worker allocation

data:
  /mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens
  limit_datasets: 32
  order: random
  batch_size: 128
  gene_mode: first
  genes: 4096
  dtype: stored
  projected_sparse_data_strategy: selected_only 或 native 对应策略

cache:
  decoded cache: not implemented / not used
  raw/compressed payload cache: not implemented / not used
  cross-batch data cache: not implemented / not used
  metadata-only block table index: allowed, record enabled/prebuilt/lazy mode
  OS page cache state: record whether dropped/warm
```

命令形态建议:

```bash
./target/release/profile_real_access \
  --engine rust-core \
  --mode scheduled \
  --limit-datasets 32 \
  --batch-sizes 128 \
  --orders random \
  --gene-mode first \
  --genes 4096 \
  --threads 96 \
  --io-workers 4 \
  --decode-workers 24 \
  --access-workers 24 \
  --fill-workers 44 \
  --prefetch-steps 512 \
  --access-prefetch-steps 512 \
  --decode-ahead-steps 512 \
  --ready-ahead-steps 512 \
  --warmup-batches 64 \
  --measure-batches 2048 \
  --memory-gib 256 \
  --out outputs/<run>/result.jsonl
```

native path 的最终命令应在此基础上显式增加 `--native-mode force/auto` 和 native
memory/coalescing 参数。发布性能数字时同时报告:

- `batches_per_s`
- `output_gb_per_s`
- `native_load_read_bytes`
- `native_load_waste_ratio`
- `native_blocks_decoded`
- `native_block_decode_work_ns`
- `native_scatter_work_ns`
- gene projection 参数，尤其是 `genes=4096`；任何 `genes < 4096` 的结果只能标注为诊断。

## 15. 测试策略

### 15.1 单元测试

- Blosc header 解析。
- block table offset 校验: 原始 i32、非负、单调、payload 内。
- `BLOSC_MEMCPYED` 直接 slice 或 fallback。
- `dont_split` / split blocks 与现有 Blosc fast path 结果一致。
- touched block 计算。
- byte-shuffle range-only unshuffle。
- coalescing 策略。
- CSR row -> block consumers。
- CSR SelectedOnly: index decode 后动态生成 data block requests。
- gene projection mapping。
- `NativeArrayMeta` 分别覆盖 dense data、CSR indices、CSR data。

### 15.2 对照测试

对每个测试 batch:

```text
generic DataBank output == native output
```

覆盖:

- dense1d
- dense2d
- sparse CSR
- u16 / u32 stored dtype
- output dtype cast
- gene identity
- external gene list
- missing zero
- cell 跨 chunk / 跨 block
- batch 内多个 cell 命中同一 block
- CSR ReadAll 与 SelectedOnly 对同一 projected sparse batch 输出一致

### 15.3 压力测试

- completion queue 大量积压。
- native memory budget soft/hard limit 和 backpressure。
- cancellation。
- 多 batch prefetch。
- worker 数 sweep。
- coalescing window sweep。
- full dataset metadata scan。
- real GPFS random sparse `genes=4096`: 10 GB/s / 400 batch/s，20 GB/s / 800 batch/s。
- real GPFS continuity sweep `genes=4096`: 20 GB/s 下从 random 800 batch/s 单调、
  近线性增加到 fully sequential baseline 8000 batch/s，并以 16000 batch/s 为优化目标。
- real GPFS / synthetic 20 GB/s cross-check: synthetic replay 的 IO 量、IO ops、block reuse
  和 normalized batch/s 必须与真实 GPFS 对齐，否则 synthetic run 标记为 false-positive
  risk。
- synthetic IO `genes=4096` shape: 50 GB/s / 2000 batch/s，验证 IO 外模块不是瓶颈。
- synthetic IO `genes=4096` shape: 100 GB/s / 4000 batch/s，作为 IO 外模块最终抗压目标。
- synthetic fully sequential `genes=4096` shape: 50 GB/s baseline/target = 20000/40000 batch/s。
- synthetic fully sequential `genes=4096` shape: 100 GB/s baseline/target = 40000/80000 batch/s。
- synthetic IO payload 必须按真实 CSR/Blosc 统计分布生成:
  - `nnz/cell`
  - selected gene overlap
  - `genes=4096` 下的 selected data group hit rate
  - Blosc decoded block size
  - compressed block size
  - block reuse / repeated block hit rate
  - continuity-p 访问序列下的 effective bytes/batch 与 block reuse ratio
  - metadata table hit/miss rate
  - coalescing waste ratio
- synthetic IO 不能跳过 decode、unshuffle、scatter 和 batch completion，只能替换真实 GPFS read。

## 16. 风险与应对

### 16.1 重复解压风险

风险: 以 cell 为执行单位会导致同一 block 被多个 cell 重复解压。

应对: response unit 必须是 block / block group，cell 只作为 completion unit。

### 16.2 IO 放大风险

风险: coalescing 过激会把 partial IO 重新放大到 chunk IO。

应对: 默认 conservative coalescing，并记录 waste bytes / waste ratio。

### 16.3 热路径复杂度风险

风险: sparse/dense/dtype/gene 组合多，代码复杂。

应对:

- native path 只覆盖主力生产组合。
- 不支持的组合直接 fallback。
- 用 macro / trait kernel 管理静态展开。

### 16.4 Blosc 实现分叉风险

风险: native path 如果重新写一套 Blosc-LZ4 解码，会和 `BloscCodec::decode_slice`
在 split、memcpyed、shuffle、offset 校验等边界条件上逐渐漂移。

应对: 先抽出现有 Blosc fast path 的内部 API，native 和 generic partial decode 共享同一套
header/block/decode/shuffle 实现。native 只负责 partial IO 和 direct scatter。

### 16.5 Metadata 扫描成本

风险: 注册阶段预读所有 block table 可能增加启动时间。

应对:

- 第一阶段先做 eager scan，换取热路径简单。
- 如果 full dataset 启动过慢，再加 lazy block table index。

### 16.6 与现有系统并存

风险: 新旧路径语义漂移。

应对:

- native 输出持续与 generic 输出做对照测试。
- native 默认 disabled 或 auto。
- fallback reason 可观测。

## 17. 推荐落地顺序

优先级最高的是把架构边界先切干净:

1. 在现有 `DataBankConfig` / `ScheduledPrefetchConfig` 中加入 native gate，默认关闭。
2. 从现有 `codecs/impls/blosc` 抽出可复用的 Blosc-LZ4 header/block/split/shuffle API。
3. 在 `databank/native` 中实现 Blosc-LZ4 block index、partial read plan 和 direct scatter。
4. 实现 LoadModule，先用 synthetic byte ranges 验证 coalescing，底层复用 `IoBackend`。
5. 接入 scheduled API 的 `native_mode=auto/force` gate。
6. 覆盖 CSR ReadAll，并补上 CSR SelectedOnly 的 index -> dynamic data 二阶段请求。
7. 将 per-batch native iterator 收敛为全局 fused worker pool。
8. 扩展更多 dtype/gene kernel，并校准 coalescing/backpressure 默认值。

这个顺序可以保证每一步都有独立验收，不需要一次性替换整个 DataBank。

## 18. 最终性能模型

理想情况下，单个 batch 的成本接近:

```text
IO bytes:
  sum(touched compressed block sizes) + coalescing waste

CPU decode:
  one LZ4 decode per unique touched block

CPU unshuffle:
  only requested byte ranges, or high coverage block full-unshuffle

CPU scatter:
  only target cell/gene output ranges

sync overhead:
  LoadRequest queue + LoadCompletion queue + per-cell pending counters
```

这与旧链路的成本差异是根本性的:

```text
old:
  touched chunk count * full compressed chunk read
  + touched chunk count * full decoded chunk decode
  + chunk-level sliced scatter

new:
  unique touched block count * compressed block read/decode
  + consumer-level direct scatter
```

如果生产访问确实是大数据随机访问、cache 命中趋近于零，这条 native path 才是正确的长期方向。

## 19. 2026-07-02 当前实现状态

本节记录当前工作区已经落地的版本。它不是完整终态架构，但已经把 Blosc-LZ4
partial read 接进 scheduled API，并在 worker 上完成真实场景压测。

### 19.1 已实现代码路径

实际代码位置:

- `rust/scdata/src/codecs/impls/blosc/`
  - `BloscHeader`、header prefix 解析、validated block ranges、`decode_blosc_lz4_block`
    和 `unshuffle_bytes` 已抽成 crate 内部复用 API。
  - `try_blosc_lz4_plan_from_prefix` 支持只用 header + block table 构建 block plan。
  - `blosc_lz4_header_table_len_from_prefix` 支持先读固定 header，再计算 block table 长度。
  - native 与 generic partial decode 共用同一套 Blosc-LZ4 block/split/shuffle 逻辑，避免双实现漂移。

- `rust/scdata/src/databank/native/`
  - `metadata.rs`: 从 full encoded payload 或 header-table prefix 构建 `NativeBloscBlockIndex`，
    并提供 `NativeBlockIndexCache`，按 `ChunkKey(file, offset, len)` 缓存 validated block table。
  - `load.rs`: `NativeLoadModule` 是 range coalescer + completion splitter，底层复用现有
    `IoBackend::submit_read(file, offset, len, priority)`，没有重造 IO backend。
  - `planner.rs`: 把 `AccessItem` 的 `SliceSpec` 转成 touched block reads + block consumers。
  - `executor.rs`: 对 loaded compressed block 做 decode/unshuffle/direct scatter。
  - `item.rs`: 单个 `AccessItem` 的 native closed loop:
    block index cache lookup -> cache miss 时读 header/table -> touched block range reads ->
    block decode -> direct scatter。cache 只保存 block metadata，不保存 raw payload 或 decoded data。

- `rust/scdata/src/databank/scheduled/native_access.rs`
  - 新增 `ScheduledBatchAccess`，统一包装旧 `AccessHandle::scheduled` 与 native scheduled iterator。
  - `native_mode=disabled` 走旧链路。
  - `native_mode=auto` 仅在 `DataBankConfig.native_config.enabled=true` 时尝试 native；
    不支持或 native 错误时 fallback 到旧链路。
  - `native_mode=force` 强制 native；不支持时直接报错。

- Python / pybind 配置:
  - `DataBankConfig.native_config`
  - `ScheduledPrefetchConfig.native_mode`
  - `test.py` 已增加 `--native-mode`、`--native-enabled`、`--native-fused-workers`、
    `--native-prefetch-blocks`，并补充 `--native-coalesce-max-gap-bytes`、
    `--native-coalesce-max-waste-ratio`、`--native-coalesce-max-merged-len`
  - `profile_real_access` 已增加同名 native 参数。

### 19.2 已覆盖的 Blosc 边界条件

- `BLOSC_MEMCPYED`
  - header-table-only plan 中会校验 `decoded_size + BLOSC_MAX_OVERHEAD == compressed_size`。
  - native executor 对 memcpyed block 直接 scatter raw payload，不走 LZ4 decode。

- split / dont_split
  - native block decode 直接调用共享的 `decode_blosc_lz4_block`，复用现有 split-size、
    dont-split 和 raw-split 逻辑。

- block offset validation
  - block table 按 Blosc payload 中的 `i32` offset 读取。
  - 读取后校验非负、单调、不能落在 header/table 内、不能越过 `compressed_size`。
  - native metadata 保存的是已经校验过的 block ranges。

- byte shuffle
  - native executor 复用 `unshuffle_bytes`。
  - 当前实现按 coverage 选择路径: 低覆盖率时直接从 shuffled block 做 range-only scatter；
    高覆盖率时整块 `unshuffle_bytes` 到 scratch 后 scatter，避免 sequential 满 block 退化。

### 19.3 CSR 两阶段现状

当前 scheduled 接入已经覆盖 CSR `read_all` 和 `selected_only` 两条路径:

- `projected_sparse_data_strategy=read_all`
  - CSR indices 和 data groups 都进入初始 scheduled items。
  - `native_mode=force` 可以强制这些 file-backed `AccessItem` 走 Blosc-LZ4 native。

- `projected_sparse_data_strategy=selected_only`
  - indices 先通过初始 scheduled items 读取并组装成 `index_bytes`。
  - response 阶段扫描 indices，判断哪些 data groups 实际包含目标 genes。
  - 对选中的 file-backed data groups 动态调用 `build_scheduled_batch_access`，继续沿用
    `ScheduledBatchAccess` 的 native/generic/force/auto 语义。
  - memory-backed groups 不进入 native force；它们已经由 DataBank memory path 提供 bytes。

这不是终态的全局 pending dependency model: 现在的实现仍是 scheduled assembly 阶段的
二阶段闭环，粒度是 selected sparse group，不是全局 block graph。但它已经消除了
selected-only 动态 data requests 回落到旧 generic access path 的主要缺口。

### 19.4 Worker 压测结果

硬件/环境:

- worker: `ms-0701-210140-1018388-nknq4.wangzhongqi.ailab-ai4ls.pod@h.pjlab.org.cn`
- CPU: 96 cores
- memory: 约 781 GiB
- project: `/mnt/shared-storage-user/dnacoding/wangzhongqi/Code/Project/scdata`
- dataset root: `/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens`
- benchmark script: `test.py`
- batch size: 128
- sampled cells:
  - quick compare: 65,536 cells = 512 batches
  - stable native sample: 262,144 cells = 2,048 batches
- datasets: first 16 rowcount matrices, all sparse CSR
- gene mode: historical diagnostic runs used first dataset gene list, first 512 genes.
  Final target runs must use `genes=4096`.
- memory budget: 512 GiB
- prefetch:
  - DataBank prefetch step 64
  - access prefetch step 512
  - decode ahead 256
  - ready ahead 128
- worker split:
  - baseline/equal split: io/decode/access/fill = 24/24/24/24
  - native-heavy split: io/decode/access/fill = 48/1/1/46

Key results:

最终目标口径 `genes=4096` 结果:

| Strategy | native_mode | genes | batches | batch/s | notes |
|---|---:|---:|---:|---:|---|
| selected_only | force | 4096 | 2048 | 1016.4 | block index cache + range-only unshuffle + direct block planner |
| selected_only | force | 4096 | 4096 | 1098.6 | 同配置长测，超过 20 GB/s / 800 batch/s 目标 |

下面表格是 `genes=512` 历史诊断结果，用于说明 native path 优化方向和 worker
线程分配趋势；它不作为最终 `genes=4096` 目标验收。

| Strategy | native_mode | batches | batch/s | notes |
|---|---:|---:|---:|---|
| selected_only | force | 512 | 464.5 | CSR dynamic data requests 已接入 native |
| selected_only | force | 512 | 462.2 | 倾斜配置 `io=32, decode=1, access=1, fill=32`，确认不再依赖旧 access worker 热路径 |
| selected_only | force | 2048 | 487.8 | 默认 coalescing: gap 16 KiB, waste 10%, merged 1 MiB |
| selected_only | force | 2048 | 456.6 | wide coalescing: gap 256 KiB, waste 50%, merged 4 MiB；过度合并有抖动 |
| selected_only | force | 2048 | 530.8 | very wide coalescing: gap 1 MiB, waste 90%, merged 8 MiB |
| selected_only | force | 2048 | 787.6 | block index cache + equal split + default coalescing |
| selected_only | force | 2048 | 800.1 | block index cache + equal split + very-wide coalescing |
| selected_only | force | 4096 | 751.6 | block index cache + equal split + very-wide coalescing |
| selected_only | force | 4096 | 897.6 | block index cache + native-heavy split `io=48, decode=1, access=1, fill=46`, very-wide coalescing |
| selected_only | disabled | 2048 | 594.9 | warm page cache 下旧链路很快；不能作为 cold IO A/B 结论 |
| read_all | disabled | 512 | 307 | 历史基线，all CSR data groups 走旧链路 |
| read_all | force | 512 | 418 | 历史基线，initial CSR groups 走 native |
| read_all | force | 2048 | 429 | 历史 stable non-profiler run |

worker 容器里 `/proc/sys/vm/drop_caches` 是只读的，无法做严格 cold-cache A/B。
因此上表的 disabled warm-cache 数字只说明“旧链路在页缓存/内部缓存热时仍有优势”，不能推导
生产大数据无缓存命中的表现。当前 native selected-only 的可解释结论是:

- 动态 selected-only data requests 已经脱离旧 generic access 热路径。
- block table index cache 是当前最有效的优化之一；它移除了每个 `AccessItem` 重复读取
  header/table 的开销。
- 96 CPU 不能再均分给旧 `io/decode/access/fill` 四个池；native path 基本不依赖旧 decode/access
  pool，把 CPU 分给 IO 和 fill/response 后，历史 `genes=512` 4096 batch 长测达到
  897.6 batch/s；当前 `genes=4096` 4096 batch 长测达到 1098.6 batch/s。
- coalescing 策略显著影响 native selected-only 吞吐，说明当前仍受小 range read 的合并/调度开销影响。
- 默认策略偏保守；very wide coalescing 在这个 warm-cache 场景下更快，但生产默认不能直接用
  90% waste，需要结合 `native_load_read_bytes/native_load_requested_bytes` 决定。

Final target command shape:

```bash
.venv/bin/python test.py \
  --mode scheduled \
  --limit-datasets 16 \
  --max-cells 262144 \
  --batch-size 128 \
  --threads 96 \
  --io-workers 48 \
  --decode-workers 1 \
  --access-workers 1 \
  --fill-workers 46 \
  --memory-gib 512 \
  --prefetch-step 128 \
  --access-prefetch-step 1024 \
  --decode-ahead-steps 512 \
  --ready-ahead-steps 256 \
  --gene-mode first \
  --genes 4096 \
  --dtype stored \
  --projected-sparse-data-strategy selected_only \
  --native-mode force \
  --native-enabled \
  --native-fused-workers 128 \
  --native-prefetch-blocks 16384 \
  --native-coalesce-max-gap-bytes 1048576 \
  --native-coalesce-max-waste-ratio 0.90 \
  --native-coalesce-max-merged-len 8388608
```

Output artifacts:

- `outputs/native-runs/force-selected-native-genes4096-planner-direct-262144.json`
- `outputs/native-runs/force-selected-native-genes4096-planner-direct-524288.json`
- `outputs/native-runs/force-selected-native-dynamic-65536.json`
- `outputs/native-runs/force-selected-native-dynamic-skewed32.json`
- `outputs/native-runs/force-selected-native-dynamic-262144.json`
- `outputs/native-runs/disabled-selected-262144.json`
- `outputs/native-runs/force-selected-native-dynamic-coalesce-wide-262144.json`
- `outputs/native-runs/force-selected-native-dynamic-coalesce-verywide-262144.json`
- `outputs/native-runs/force-selected-native-index-cache-default-262144.json`
- `outputs/native-runs/force-selected-native-index-cache-verywide-262144.json`
- `outputs/native-runs/force-selected-native-index-cache-verywide-524288.json`
- `outputs/native-runs/force-selected-native-index-cache-skewed48-524288.json`

### 19.5 Profiler run

下面 profiler 是历史 `genes=512` / `read_all` 诊断 run，只用于解释旧链路的 iopool
排队现象；正式 profiler 必须改为 `genes=4096` 和最终 selected-only/native 配置。

Profiler command:

```bash
cargo run --manifest-path rust/scdata/Cargo.toml \
  --release --no-default-features --features pybind-bench,profile \
  --bin profile_real_access -- \
  --mode scheduled \
  --engine rust-core \
  --limit-datasets 16 \
  --max-cells 262144 \
  --batch-sizes 128 \
  --orders random \
  --prefetch-steps 64 \
  --access-prefetch-steps 512 \
  --decode-ahead-steps 256 \
  --ready-ahead-steps 128 \
  --threads 96 \
  --memory-gib 512 \
  --gene-mode first \
  --genes 512 \
  --sparse-data-strategy read_all \
  --native-mode force \
  --native-enabled true \
  --native-fused-workers 64 \
  --native-prefetch-blocks 8192 \
  --warmup-batches 64 \
  --measure-batches 512 \
  --out outputs/native-runs/profile-force-readall.jsonl \
  --no-io
```

Profiler overhead 很高，吞吐不等同于非 profiler run:

- measured batches: 512
- profiler batch/s: 179.7
- output bytes: 128 MiB
- actual IO read bytes from iopool metrics: 14,474,120,932 bytes
- mean IO per measured batch: 28.27 MB
- `iopool_queue_dispatch_wait`: 9,096,648 ms accumulated
- `iopool_queue_queue_full`: 240,978
- `iopool_queue_dedup_hits`: 155,776

这个 profiler 数字是从 iopool 总读数反推出来的，不等价于理论目标；它混入了 header/table
读取、coalescing、read_all/selected_only 策略和可能的重复读。后续目标必须以真实
CSR/Blosc 形状统计为准。

### 19.5.1 CSR/Blosc exact IO model

按历史 `genes=512` 诊断配置重新统计:

```text
datasets: first 16 rowcount matrices
cells: 262,144 = 2,048 batches
batch_size: 128
order: random
gene_mode: first
genes: 512
data dtype: u16
indices dtype: i32
strategy: selected_only
```

正式验收必须新增 `genes=4096` 版本的 exact IO model，并用它驱动 synthetic IO payload
分布。不能把下面 `genes=512` 的 30.38 MiB/batch 直接外推到 `genes=4096`。

真实 CSR 形状:

```text
mean nnz/batch = 240,386
mean nnz/cell  = 1,878

decoded indices bytes/batch = 0.917 MiB
decoded selected data group bytes/batch = 0.497 MiB
decoded total group bytes/batch = 1.414 MiB

selected_nnz/batch = 1,177
selected value bytes/batch = 2.5 KiB
```

这里 `selected_nnz` 很小，但当前 selected-only 的 data 读取粒度是 data group row-slice:
只要 group 内任一 index 命中目标 gene，就读取该 group 对应的 data slice。因此 data IO 不会
下降到 `selected_nnz * 2B`。

Blosc block 取整后的 exact compressed IO:

```text
indices compressed block bytes/batch = 21.38 MiB
data compressed block bytes/batch    = 8.998 MiB
total compressed block IO/batch      = 30.38 MiB

block read ops/batch = 254.2
coalesced runs/batch = 249.3
```

如果保持 `30.38 MiB/batch`，理论上限大约是:

| 带宽 | batch/s |
|---:|---:|
| 10 GiB/s | 337 |
| 20 GiB/s | 674 |

但系统目标不是接受这个上限，而是把 random sparse 的实际 IO budget 压到约
`25 MB/batch`:

| 目标带宽 | 目标 batch/s | 目标 IO budget |
|---:|---:|---:|
| 10 GB/s | 400 | 25 MB/batch |
| 20 GB/s | 800 | 25 MB/batch |

因此这份历史 exact 统计只说明 `genes=512` 下，如果只看冷 block IO bytes，仍需减少约
15-20% 的 block-level IO bytes，并把 `io_ops_per_batch` 控制在 250 左右。正式
`genes=4096` 目标需要重新计算同一组指标。block table cache 和 96 CPU native-heavy
分配后，真实 worker 的 warm GPFS 长测已达到 897.6 batch/s，但这是 `genes=512`
历史结果，不能替代 `genes=4096` 的严格验收，也不能替代 cold IO 证明。
profiler 下主要可见的瓶颈仍是 iopool 队列排队和 queue full，而不是 decode pool；
native force run 的 decode profile 为空。

### 19.6 Synthetic IO harness 与 96C 结果

新增 profile-gated synthetic harness:

- `rust/scdata/src/synthetic.rs`
- `rust/scdata/src/bin/native_synthetic_io.rs`

它使用内存 `IoBackend` 替换 GPFS read，但仍走:

```text
AccessItem + SliceSpec
  -> NativeBlockIndexCache
  -> Blosc-LZ4 header/table parse
  -> NativeLoadModule coalescing/splitting
  -> touched block decode
  -> range/full unshuffle
  -> scatter output
  -> per-batch completion/checksum
```

这不是完整 DataBank scheduled path，但已经覆盖 native IO 外热路径。当前 synthetic shape:

```text
batch_size: 128
cell_bytes: 12 KiB
cells_per_chunk: 4096
block_size: 192 KiB
compression: Blosc-LZ4, shuffle=1, typesize=2
encoded compression ratio: about 3.0x
worker: 96 CPU
```

本轮同时做了两个代码级优化:

1. native byte-shuffled block 支持低覆盖率 range-only unshuffle；高覆盖率仍走整块
   `unshuffle_bytes`，避免 sequential 满 block 时退化。
2. native planner 从 `blocks * ranges` 全扫描改为按 range 二分定位 touched block，
   降低 random batch planner CPU 开销。

96C worker 结果:

| run | order | batches | batch/s | targets | read bytes/batch | read GiB/s | block reuse |
|---|---|---:|---:|---|---:|---:|---:|
| `outputs/native-synthetic/random-4096-w96-planner-direct.json` | random | 4096 | 4031.2 | hot-path diagnostic only; real-GPFS calibration failed | 14.65 MB | 55.0 | 0.212 |
| `outputs/native-synthetic/sequential-4096-w96-planner-direct.json` | sequential | 4096 | 36845.1 | hot-path diagnostic only; full scheduled path missing | 0.647 MB | 22.2 | 0.930 |

解释:

- random synthetic 在 standalone hot-path harness 中名义达到 `50 GB/s -> 2000 batch/s`
  和 `100 GB/s -> 4000 batch/s`，但由于未通过真实 GPFS 校准，不算最终验收。
- fully sequential synthetic 结果说明 host 热路径有明显顺序复用余量，但不能替代真实 GPFS
  20 GB/s sequential 验收；真实验收仍要在 `profile_real_access --orders sequential`
  路径下完成。
- fully sequential 在 standalone hot-path harness 中达到 50 GB/s baseline
  `20000 batch/s`，但这仍是诊断结果；50 GB/s target `40000 batch/s` 和
  100 GB/s sequential baseline/target 也仍未达到。
- synthetic harness 目前验证的是 native block IO/decode/scatter/completion 热路径；仍需补完整
  DataBank scheduled synthetic path 或在真实 DataBank 上接入同等 virtual IO backend。

Continuity-p sweep:

```text
order: continuity
p: 0, 0.25, 0.5, 0.75, 0.9, 0.97, 0.99, 1.0
batches: 4096
batch_size: 128
workers: 96
shape: same as above
```

按 `continuity(p) = (1/bytes(p) - 1/bytes(0)) / (1/bytes(1) - 1/bytes(0))`
计算 block-derived continuity:

| p | batch/s | read MB/batch | unique blocks/batch | block reuse | continuity | 20G baseline | 20G target | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.00 | 4052.4 | 14.634 | 100.8 | 0.212 | 0.000 | 800.0 | 800.0 | hot-path only |
| 0.25 | 4494.6 | 13.029 | 81.6 | 0.362 | 0.002 | 812.9 | 827.2 | hot-path only |
| 0.50 | 6318.3 | 10.279 | 60.1 | 0.531 | 0.006 | 844.3 | 893.5 | hot-path only |
| 0.75 | 9913.5 | 5.919 | 36.0 | 0.719 | 0.021 | 953.9 | 1124.8 | hot-path only |
| 0.90 | 18233.0 | 2.716 | 20.1 | 0.843 | 0.064 | 1258.6 | 1768.1 | hot-path only |
| 0.97 | 25030.9 | 1.228 | 12.4 | 0.903 | 0.159 | 1941.5 | 3209.8 | hot-path only |
| 0.99 | 31497.9 | 0.806 | 10.1 | 0.921 | 0.249 | 2593.4 | 4586.1 | hot-path only |
| 1.00 | 36749.3 | 0.209 | 8.9 | 0.930 | 1.000 | 8000.0 | 16000.0 | hot-path only |

结果满足:

- batch/s 随 p 单调上升。
- block-derived continuity 随 p 单调上升。
- synthetic sweep 中所有点都超过按 20G 曲线折算的诊断线；这只证明 native block
  decode/scatter/completion 热路径余量，不等同于真实 GPFS 20G sequential 达标。
- p=1 fully sequential 相对 random target `800 batch/s` 是 `45.9x`，相对 synthetic
  p=0 实测 `4052.4 batch/s` 是 `9.1x`。

### 19.7 真实 GPFS / synthetic 20G 对照预检

按最新验收口径，synthetic 必须和真实 GPFS 在 20G 目标点相互印证。使用同一 worker、
`genes=4096`、batch128、native force、selected-only、`io=48, decode=1, access=1,
fill=46`、大 coalescing 窗口，关闭额外 IO sampler 后，真实 GPFS 预检结果如下:

| run | order | dtype | batch/s | read MB/batch | read GB/s | IO calls/batch | avg parts/batch |
|---|---|---|---:|---:|---:|---:|---:|
| `outputs/native-runs/profile-real-gpfs-random-genes4096-u16-precheck.jsonl` | random | u16 | 416.7 | 32.96 | 13.74 | 245.6 | 100.5 |
| `outputs/native-runs/profile-real-gpfs-sequential-genes4096-u16-precheck.jsonl` | sequential | u16 | 11416.6 | 0.621 | 7.09 | 2.79 | 1.0 |
| `outputs/native-runs/profile-real-gpfs-sequential-genes4096-precheck.jsonl` | sequential | stored auto -> u32 | 9024.3 | 0.629 | 5.67 | 2.81 | 1.0 |

和当前 synthetic harness 的对照:

- random synthetic 读 `14.65 MB/batch`，真实 GPFS random 读 `32.96 MB/batch`，相差
  `2.25x`。这说明 hand-shaped synthetic random 没有复现真实 `genes=4096` CSR selected-only
  block/group 分布，不能作为 100G random 达标证据。
- sequential synthetic 读 `0.647 MB/batch`，真实 GPFS sequential 读 `0.621 MB/batch`，
  IO 量接近；但 synthetic `36845 batch/s` 对真实 GPFS u16 `11416 batch/s` 是 `3.2x`。
  这说明 standalone synthetic harness 没覆盖完整 DataBank scheduled assembly、输出 buffer
  分配、completion queue 和真实调度开销。
- 因此当前 synthetic 结果保留为 native block hot-path 诊断，不能标记为最终验收通过。
  下一步必须做 trace-driven synthetic 或完整 DataBank scheduled virtual IO backend。

### 19.8 完整 scheduled synthetic 校准

已新增 profile-only scheduled synthetic mode:

```bash
./target/release/native_synthetic_io \
  --scheduled true \
  --batches 1024 \
  --batch-size 128 \
  --workers 96 \
  --chunks 2048 \
  --cells-per-chunk 4096 \
  --cell-bytes 12288 \
  --genes 4096 \
  --block-size 196608 \
  --entropy-fraction 0.24 \
  --order random \
  --coalesce-max-gap-bytes 1048576 \
  --coalesce-max-waste-ratio 0.90 \
  --coalesce-max-merged-len 8388608
```

这个模式构造真实 `SparseCsrSpec`，indices/data 都是 Blosc-LZ4 file chunks，通过
profile-only virtual `IoBackend` 注入 `DataBank`，然后走
`prefetch_cells_scheduled_by_gene_names`。它覆盖:

- DataBank scheduled producer / completion queue。
- CSR selected-only index phase 和动态 data request。
- native Blosc partial read / decode / scatter。
- output buffer 分配和 row-major batch assembly。

20G 交叉校准结果:

| run | order | batch/s | read MB/batch | IO calls/batch | 20G IO-bound batch/s | 对照 |
|---|---|---:|---:|---:|---:|---|
| `outputs/native-synthetic/scheduled-random-genes4096-chunks2048-b1024-entropy024.json` | random | 354.4 | 33.50 | 256.2 | 597.0 | real GPFS random: 416.7, 32.96 MB/b, 245.6 ops/b, 606.8 |
| `outputs/native-synthetic/scheduled-sequential-genes4096-chunks2048-b8192-entropy024.json` | sequential | 6034.6 | 0.602 | 2.13 | 33224.8 | real GPFS sequential u16: 11416.6, 0.621 MB/b, 2.79 ops/b, 32195.4 |

结论:

- random 的 `read_bytes_per_batch` 差 `1.6%`，`IO calls/batch` 差约 `4.3%`，
  20G IO-bound batch/s 差约 `1.6%`。这组 scheduled synthetic 可以作为 random
  20G 校准基线。
- sequential 的 IO 量也对齐，说明 block 复用模型可用；但 measured batch/s 和真实 GPFS
  差约 `1.9x`。这里主要受测量口径影响: 当前 scheduled synthetic 计入完整 measured stream
  setup，而 `profile_real_access` 的 reported measured seconds 会扣除 stream setup，并且
  warmup/prefetch 会让部分工作提前发生。因此 sequential 仍不能宣布 16000 batch/s target
  达成，必须统一 benchmark 计时口径后再比较 host-bound 吞吐。
- 旧 standalone synthetic 仍只作为 hot-path microbenchmark；正式验收应使用 scheduled
  synthetic 或 trace-driven replay。

### 19.9 当前限制和下一步

1. 当前 native scheduled iterator 是 per-batch worker thread + bounded ordered result queue，
   不是最终设计里的全局 fused worker pool。它已经能验证 partial read/direct scatter 收益，
   但会产生额外线程和队列开销。
2. `native_fused_workers` 目前是 per-batch async concurrency hint，不等价于真正的 CPU
   worker 数。block table cache 后，`io=48, decode=1, access=1, fill=46` 的 native-heavy
   96 CPU 分配在历史 `genes=512`、4096 batch 长测中达到 897.6 batch/s；当前 `genes=4096`
   4096 batch 长测达到 1098.6 batch/s。旧的均分池不是 native path 的最佳配置。
3. CSR selected-only 已经接入二阶段 native scheduled access，但还不是终态的全局 block graph。
   下一步应把 dynamic data request 继续下推到同一规划单元内的 duplicate block request
   collapse 和 multi-consumer decode-once response unit；这不是跨 batch payload cache。
4. LoadModule 已经能 coalesce byte ranges，但还没有 native 专属 metrics；现在只能从 iopool
   总 metrics 反推。必须补 `native_load_requested_bytes`、`native_load_read_bytes`、
   `native_load_waste_ratio` 和 block decode/scatter 计数，才能可靠设置生产默认 coalescing。
5. completion queue/memory budget 仍需生产化 backpressure；benchmark mode 可以放宽，
   production 默认必须受 native memory budget 限制，并记录 soft-limit wait/blocked metrics。
6. `genes=4096` 真实 GPFS random sparse 在当前预检中是 `416.7 batch/s`，实际 IO 带宽约
   `13.74 GB/s`，折算到 20 GB/s 约 `607 batch/s`。这说明当前 random 路径还没达到
   20 GB/s / 800 batch/s 目标，主要原因是实际 `read_bytes_per_batch = 32.96 MB`
   高于 25 MB/batch budget。
7. `genes=4096` 真实 GPFS fully sequential u16 是 `11416.6 batch/s`，超过 8000 baseline，
   但低于 16000 target。由于实际 IO 只有 `0.621 MB/batch`，这里不是 IO bandwidth
   限制，而是 DataBank scheduled assembly / output / completion 等 host 路径限制。
8. standalone synthetic random 50/100 GB/s 和 continuity-p sweep 只能作为 native block
   hot-path 诊断。scheduled synthetic random 已完成 20G 交叉校准，但还需要扩展到
   50/100G bandwidth-limited replay。
9. sequential 的 IO 模型已对齐，但 host-bound batch/s 仍需统一计时口径；当前不能把
   standalone 或 scheduled synthetic 的 sequential 数字当作 16000 batch/s target 达成证据。
10. 要进一步逼近真实 IO 上限，应减少 iopool queue full，做全局 native fused worker pool，
   并把同一 batch 内相同 block 的 read/decode 去重从“coalesced IO”推进到“block response
   unit 共享 decode”。这个优化只在当前 batch/request group 请求图内生效，completion 消费后不保留
   raw/compressed/decoded payload。

### 19.10 Selected-run 精确 data 读取

本轮实现把 CSR `selected_only` 从 **data group 级过滤** 下推到 **selected nonzero run
级过滤**:

```text
index groups -> decode/copy indices -> scan projected genes
  -> build selected-data-only SparseBatchPlan
  -> data AccessItem SliceSpec 只包含命中 gene 的连续 nonzero runs
  -> native Blosc-LZ4 partial block read/decode/scatter
```

实现位置:

- `rust/scdata/src/databank/sparse/planning.rs`
  - 新增 `plan_sparse_selected_data_batch`
  - 扫描 `index_bytes`，为命中 projection 的连续 run 生成新的 `SparseReadPiece`
  - 新 plan 只保存 data pieces/groups，复用原始 `index_bytes`
- `rust/scdata/src/databank/sparse/scatter.rs`
  - 普通 projected sparse path 改为消费 selected-data-only plan
- `rust/scdata/src/databank/scheduled/assemble.rs`
  - scheduled projected selected path 改为消费 selected-data-only plan
- `rust/scdata/src/synthetic.rs`
  - scheduled synthetic 增加 `source_genes`
  - virtual IO 对 exact Blosc block range 预切 `Arc<[u8]>`，只用于 synthetic
    benchmark，生产路径不引入 payload cache
- `rust/scdata/Cargo.toml`
  - 显式注册 `native_synthetic_io` profile bin

验证:

```bash
cargo check --manifest-path rust/scdata/Cargo.toml --no-default-features --features profile --bin profile_real_access
cargo check --manifest-path rust/scdata/Cargo.toml --no-default-features --features profile --bin native_synthetic_io
cargo test  --manifest-path rust/scdata/Cargo.toml --no-default-features --features profile projected_csr --lib
cargo test  --manifest-path rust/scdata/Cargo.toml --no-default-features --features profile scheduled_projected --lib
cargo test  --manifest-path rust/scdata/Cargo.toml --no-default-features --features profile databank::native --lib
cargo test  --manifest-path rust/scdata/Cargo.toml --no-default-features --features profile synthetic --lib
```

真实 GPFS random `genes=4096` 长测:

| run | order | dtype | strategy | batch/s | read MB/batch | read GB/s | IO calls/batch | status |
|---|---|---|---|---:|---:|---:|---:|---|
| `outputs/native-runs/profile-real-gpfs-random-genes4096-selected-runs-2048.jsonl` | random | stored -> u32 | selected-run | 803.3 | 10.02 | 8.05 | 131.0 | 20G/800 target passed |

对比 19.7 的旧 random 预检，`read_bytes_per_batch` 从 `32.96 MB` 降到约
`10.02 MB`，已经低于 25 MB/batch budget。当前 random 800 batch/s 不再是 IO
budget 不足，而是 scheduled assembly / index scan / scatter host path 开始限制。

synthetic random:

| run | mode | shape | batch/s | read MB/batch | 50G eq batch/s | 100G eq batch/s | status |
|---|---|---|---:|---:|---:|---:|---|
| `outputs/native-synthetic/standalone-random-genes4096-selected-runs-b8192.json` | standalone native hot path | `genes=4096`, `cell_bytes=12288` | 7637.5 | 4.33 | 11544.0 | 23088.0 | hot-path passes 50/100G |
| `outputs/native-synthetic/scheduled-random-genes4096-source60000-selected-runs-b4096.json` | full scheduled synthetic | `source_genes=60000`, `genes=4096` | 741.9 | 11.14 | 4487.8 | 8975.5 | IO model passes; host path not yet 2000/4000 |

这里必须区分两个结论:

- native block IO/decode/scatter hot path 已经能超过 random 50G/2000 和
  100G/4000 目标。
- 完整 scheduled synthetic 仍只有 `741.9 batch/s`，不能作为 IO 外模块 50/100G
  通过证据。瓶颈已经转移到 scheduled assembly、index scan、动态 data plan 构建、
  per-group drain/scatter 调度。

顺序访问:

| run | order | dtype | strategy | batch/s | read MB/batch | status |
|---|---|---|---|---:|---:|---|
| `outputs/native-runs/profile-real-gpfs-sequential-genes4096-selected-runs-u16-2048.jsonl` | sequential | u16 | selected-run | 3189.2 | 0.220 | below 8000 baseline |
| `outputs/native-runs/profile-real-gpfs-sequential-genes4096-read-all-u16-2048.jsonl` | sequential | u16 | read_all | 1777.0 | 0.715 | below 8000 baseline |

selected-run 降低了顺序访问 IO bytes，但没有提高 batch/s；顺序访问已经不是 IO 问题，
而是 host path。后续要达成 sequential 8000/16000，不能继续只优化读取粒度，需要做
全局 fused scheduled path:

1. 在 batch/request group 内直接以 row/cell 为单位生成 selected runs，减少中间
   `SparseBatchPlan` / `SparseReadGroup` 分配。
2. 把 index scan、selected data planning、native block response、projected scatter 合成一个
   fused CSR kernel，避免当前的多轮 schedule/drain/scatter。
3. 对顺序/高连续度输入增加 coarse row-run path，由上游显式选择 strategy；不要让 random
   optimized selected-run 破坏 sequential host throughput。
4. 增加 native 专属 metrics，拆出 `selected_plan_ms`、`index_scan_ms`、`data_load_ms`、
   `projected_scatter_ms`、`completion_drain_ms`，否则 sequential host 瓶颈只能从总 profile
   间接判断。

### 19.11 Release no-profile ordered native batch

本轮新增并保留的 release 路径优化:

- native selected-data dynamic phase 增加 **ordered batch reply**：仍由
  `databank-native-scheduled-*` worker 执行 native read/decode/scatter，但一次返回当前
  selected data batch 的 ordered `Vec<Vec<u8>>`，避免逐 item channel drain。
- native partial output 对 `SliceShape::Sequential` 使用未初始化 `Vec<u8>`，由 block scatter
  完整覆盖输出，避免先清零再覆盖。
- selected sparse ordered data scatter 增加窄版 bulk dispatch，把 projection context 和
  CSR index dtype dispatch 提到 group loop 外。
- `native_item_batch_size = fused_workers * 4`，最多 256；2-7 item 小批也走 multi-item
  block dedup path。

拒绝的实验:

- response/fill worker inline native decode (`SCDATA_NATIVE_INLINE_SELECTED` 原型) 在 scheduled
  synthetic random 从约 `952` 提到 `1659 batch/s`，但真实 GPFS random 在相同 sampled cells 下
  掉到 `255 batch/s`，`fill=96` 也只有 `225 batch/s`。原因是它破坏了 native worker 与 GPFS
  IO/decode 的 overlap。因此默认路径不能让 fill response worker 直接承担 selected data native
  decode；应由 native worker 做 decode，response worker 做 assembly/scatter。

最终保留配置的 worker release no-profile 结果:

| run | order | dtype | workers | prefetch | batch/s | output GB/s | status |
|---|---|---|---|---|---:|---:|---|
| `outputs/native-runs/release-real-random32-ordered-bulkscatter-fill46-pref8192-max524288-20260702.jsonl` | random | stored -> u32 | `io=48 decode=1 access=1 fill=46 fused=96` | `8192/8192/8192/8192` | 1582.2 | 3.32 | passes 20G GPFS 1000 target |
| `outputs/native-runs/release-real-sequential-u16-ordered-bulkscatter-fill76-pref512-max524288-20260702.jsonl` | sequential | u16 | `io=48 decode=1 access=1 fill=76 fused=96` | `512/512/512/512` | 17616.6 | 18.47 | passes 16000 sequential target |

当前 scheduled synthetic random:

| run | order | host batch/s | read MB/batch | 50G eq batch/s | 100G eq batch/s | status |
|---|---|---:|---:|---:|---:|---|
| `outputs/native-synthetic/scheduled-random-block64k-ordered-bulkscatter-20260702.json` | random | 992.0 | 13.38 | 3737.8 | 7475.6 | virtual IO targets pass; host path still below 2000 |

因此，截至本节:

- 明确通过: real GPFS random 20G `1000 batch/s`、real sequential `16000 batch/s`、
  scheduled synthetic random 50/100G 等价 `2000/4000 batch/s`。
- 仍未完全证明: full scheduled synthetic random host path 自身达到 `2000/4000 batch/s`。
  若把“完全 IO-bound”解释为 50/100G 下 CPU/assembly 绝不先饱和，还需要继续做 fused CSR
  kernel 或 block response 直接 sparse scatter。

### 19.12 Partial LZ4 prefix decode

本轮新增并保留:

- Blosc-LZ4 block decoder 增加 `decode_blosc_lz4_block_partial_prefixes`。
- native byte-shuffle low-coverage branch 先计算每个 LZ4 split 实际会被 scatter 读取的
  shuffled prefix；只有 prefix 总量明显小于完整 block 时才用
  `LZ4_decompress_safe_partial`。
- 对长度 >= 64 B 的 value-aligned requested range 使用按元素范围更新 prefix 的 fast path；
  小 range 保持逐字节精确计算，避免 random 小范围退化。

保留但默认禁用的负向实验:

- `SCDATA_NATIVE_FUSED_SCATTER=1` 会把 ordered native load + selected sparse bulk scatter
  放进一个 native custom command。synthetic 看起来可行，但真实 GPFS random 从默认同配置
  `1352.2 batch/s` 降到 `1408.0 batch/s` 对应的实验前后不稳定，且相比早上
  ordered+bulk scatter 的 `1582.2 batch/s` 没有优势；更重要的是它让 scatter 竞争 native
  IO/decode worker 槽。因此默认仍是 native worker 做 ordered load/decode，response worker
  做 bulk scatter。

本轮 worker release / synthetic 结果:

| run | order | kind | batch/s | read GB/s or output GB/s | note |
|---|---|---|---:|---:|---|
| `outputs/native-synthetic/scheduled-random-block64k-default-restored-20260702.json` | random | scheduled synthetic before partial | 985.1 | 13.18 read GB/s | CPU path baseline at current worker state |
| `outputs/native-synthetic/scheduled-random-block64k-partial-lz4-prefixfast64-20260702.json` | random | scheduled synthetic after partial | 1167.9 | 15.62 read GB/s | host path +18-21% vs default/restored |
| `outputs/native-synthetic/scheduled-sequential-block64k-partial-lz4-prefixfast64-20260702.json` | sequential | scheduled synthetic after partial | 13693.5 | 7.16 read GB/s | sequential synthetic improves with 64 B prefix fast path |
| `outputs/native-runs/release-real-random32-partial-lz4-prefix64-fill46-pref8192-max524288-20260702.jsonl` | random | real GPFS | 1353.5 | 2.84 output GB/s | passes 1000; lower than morning 1582 due GPFS/worker variability |
| `outputs/native-runs/release-real-sequential-u16-partial-lz4-prefix64-fill76-pref512-max524288-20260702.jsonl` | sequential | real GPFS | 17008.9 | 17.84 output GB/s | passes 16000 sequential target |

对“400/800/当前低吞吐”的判断:

- 同一 worker 上，scheduled synthetic random 在 default/restored 状态稳定在 `985-992 batch/s`，
  partial LZ4 后稳定提升到 `~1168 batch/s`。这说明 native CPU path 没有退化到 400。
- 真实 GPFS random/sequential 在同一天不同时段从 `1582/17616` 变成
  `1353/17009`，且 sequential 也同步波动；这更像 GPFS 或 worker 调度状态变化，而不是
  random sparse 单一路径退化。
- `decode_workers=1`、`access_workers=1` 是当前 native release 配置的有意选择：native
  selected data 的 IO+LZ4 由 `native_fused_workers=96` 承担；普通 decode/access worker
  主要影响 fallback 和调度，不是主解压池。

仍未完成的严格目标:

- full scheduled synthetic random host path 仍是 `1168 batch/s`，没有达到
  `2000/4000 batch/s` 的“完全 CPU 不先饱和”解释。下一步必须继续减少 selected CSR
  中间 materialization，或把 native block response 直接写入最终 sparse output。

### 19.13 Read-all scheduled synthetic and sorted CSR opt-in

本轮新增:

- `native_synthetic_io` 增加 `--projected-sparse-data-strategy`，可直接比较
  `selected_only` 与 `read_all`。
- `native_synthetic_io` 增加 `--fill-workers`、`--native-workers`、`--io-workers`，避免一个
  `--workers` 同时控制 response/fill、native scheduler 和 IO worker。
- `IoBackend::prefers_inline_reads()` 默认 false；`VirtualChunkIo` 返回 true，使 native load
  对内存 resident synthetic backend 避免每个 coalesced read 都 `JoinSet::spawn`。
  实测该项不是主瓶颈，保留为低风险 synthetic/backend hint。
- 新增默认关闭的 `SCDATA_ASSUME_SORTED_CSR_INDICES=1`。仅在调用方明确确认 CSR row indices
  有序、且 projection 是连续 source range 时，read_all projected scatter 对 row 尾部早停。
  默认关闭，真实 GPFS 验证未启用该假设。

关键发现:

- 在当前 `genes=4096, source_genes=60000, cell_bytes=12288, block=64K` synthetic shape 下，
  `selected_only` 节省 data bytes，但二次 selected planning、小 group materialization 和
  projected sparse scatter 成本更高；`read_all` 反而更快。
- 纯 native loader direct harness 可达 `6508.4 batch/s`，说明 Blosc-LZ4 native block 层本身
  足够支撑 4000；瓶颈在完整 DataBank scheduled CSR projected assemble path。
- 32K block 诊断可到 `2354.9 batch/s`，但仍远低于 4000；100G 目标需要真正 fused CSR
  kernel/block response direct scatter，而不是只调 block size 或 worker 数。

本轮 worker release / synthetic 结果:

| run | order | strategy / config | batch/s | read MB/batch | note |
|---|---|---|---:|---:|---|
| `outputs/native-synthetic/scheduled-random-block64k-readall-workers48-20260702.json` | random | read_all, 48 unified workers | 1917.7 | 13.38 | read_all 明显优于 selected_only |
| `outputs/native-synthetic/scheduled-random-block64k-readall-sorted-workers48-b8192-20260702.json` | random | read_all + sorted opt-in, 48 workers | 1959.3 | 13.38 | sorted early-stop 小幅提升 |
| `outputs/native-synthetic/scheduled-random-block64k-readall-sorted-fill72-native48-io48-b8192-20260702.json` | random | read_all + sorted opt-in, fill/native/io=72/48/48 | 2016.6 | 13.38 | passes 50G / 2000 host-path baseline |
| `outputs/native-synthetic/scheduled-random-block32k-readall-sorted-fill72-native48-io48-b8192-20260702.json` | random | 32K diagnostic | 2354.9 | 6.94 | block granularity helps but not enough for 4000 |
| `outputs/native-synthetic/direct-random-block64k-partial-lz4-inline-virtual-io-20260702.json` | random | direct native loader harness | 6508.4 | 5.94 | proves native block layer is not the 4000 bottleneck |
| `outputs/native-synthetic/scheduled-sequential-block64k-readall-sorted-fill72-native48-io48-b8192-20260702.json` | sequential | read_all + sorted opt-in, fill/native/io=72/48/48 | 19133.6 | 0.523 | near 20K synthetic baseline, below 40K target |

真实 GPFS release no-profile 回归验证（未启用 sorted opt-in）:

| run | order | dtype | config | batch/s | output GB/s | status |
|---|---|---|---|---:|---:|---|
| `outputs/native-runs/release-real-random32-current-selected-fill46-pref8192-max524288-20260702.jsonl` | random | stored -> u32 | selected_only, `io=48 decode=1 access=1 fill=46 fused=96` | 1336.9 | 2.80 | passes 1000; same-day GPFS variability band |
| `outputs/native-runs/release-real-sequential-u16-current-selected-fill76-pref512-max524288-20260702.jsonl` | sequential | u16 | selected_only, `io=48 decode=1 access=1 fill=76 fused=96` | 17108.5 | 17.94 | passes 16000 target |

当前严格目标状态:

- real GPFS random `>=1000 batch/s`: 通过。
- real GPFS sequential `>=16000 batch/s`: 通过。
- full scheduled synthetic random 50G host path `>=2000 batch/s`: 通过一次长测
  (`2016.6 batch/s`)，但依赖 synthetic 已知有序 CSR 的 opt-in。
- full scheduled synthetic random 100G host path `>=4000 batch/s`: 未完成。
- full scheduled synthetic sequential 50G/100G host path `20000/40000+ batch/s`: 未完成；
  当前 64K sequential 为 `19133.6 batch/s`。

下一步必须做的不是继续调参，而是移除 scheduled CSR projected assemble 的剩余 host bottleneck：

- fused CSR kernel：index scan、data native load completion 和 projected scatter 合并，避免
  `Vec<Vec<u8>>`/`Arc<[u8]>` 中间 materialization。
- 对 read_all + contiguous projection 的有序 CSR 路径，把 native block response 直接写到最终
  sparse output row，避免先 materialize decoded data group 后再二次 scatter。

### 19.14 Negative experiments after read_all/sorted baseline

本轮继续定位 `read_all + sorted + fill72/native48/io48` 为什么停在约 `2000 batch/s`。
所有负向实验均已从代码回退，避免保留无收益复杂度。

已验证但不保留:

| run | change | batch/s | baseline | conclusion |
|---|---|---:|---:|---|
| `outputs/native-synthetic/scheduled-random-block64k-readall-sorted-fill72-native48-io48-b8192-batchscatter-20260702.json` | read_all 默认分支先收齐 data groups，再调用 batch scatter | 1902.8 | 2016.6 | 破坏边 drain 边 scatter 的流水，且增加 `Arc` materialization |
| `outputs/native-synthetic/scheduled-random-block64k-readall-sorted-fill72-native48-io48-b8192-nativebatch1024-20260702.json` | `native_item_batch_size` 从 `workers*4, max=256` 放大到 `workers*8, max=1024` | 1962.1 | 2016.6 | 单批 item 变大没有减少 read ops，反而降低 native task 并行度 |
| `outputs/native-synthetic/scheduled-random-block64k-readall-sorted-fill72-native48-io48-b8192-fasthash-20260702.json` | native item range dedup 从 std `HashMap` 换成 fast hasher | 1982.4 | 2016.6 | range dedup HashMap 不是主瓶颈 |
| `outputs/native-synthetic/scheduled-random-block64k-readall-sorted-fill72-native48-io48-b8192-virtualrangevec-20260702.json` | synthetic `VirtualChunkIo` exact range 从 HashMap 改为按 start 二分表 | 1994.8 | 2016.6 | virtual IO lookup 不是主瓶颈 |
| `outputs/native-synthetic/scheduled-random-block64k-readall-sorted-fill72-native48-io48-b8192-resplimit36-20260702.json` | producer active response 上限从 `worker_count-1` 降到 `worker_count/2` | 1906.3 | 2016.6 | 少提交阻塞中的 response job 会降低 assemble 并行度 |
| `outputs/native-synthetic/scheduled-random-block8k-readall-sorted-fill72-native48-io48-b8192-20260702.json` | block size 8K | 2035.9 | 2016.6 | read bytes 降低但 read ops 仍约 257/batch，batch/s 基本不变 |
| `outputs/native-synthetic/scheduled-random-block64k-readall-sorted-fill72-native56-io48-20260702.json` | native workers 56 | 1896.4 | 2016.6 | 超过 48 后线程争用开始压过收益 |
| `outputs/native-synthetic/scheduled-random-block64k-readall-sorted-fill72-native64-io48-20260702.json` | native workers 64 | 1864.1 | 2016.6 | 同上 |
| `outputs/native-synthetic/scheduled-random-block64k-readall-sorted-fill72-native96-io48-20260702.json` | native workers 96 | 1263.4 | 2016.6 | 明显过量争用 |

额外诊断:

- worker cpuset: `28-47,58-85,124-143,154-181`，`nproc=96`。
- `fill72/native48/io48` 对 synthetic 而言主要抢 CPU 的是 fill/native；
  `VirtualChunkIo::prefers_inline_reads()` 下 `io_workers` 基本不参与。
- shell `time` 对 1024 batch release run 显示约 `real=0.99s user=19.85s sys=2.51s`，
  benchmark 内部 measured window 为 `1987.2 batch/s`。整体 CPU 没有打满 96 core，
  说明剩余瓶颈是 fill response worker 阻塞等待 native data、native worker 顺序处理
  batch 内 block decode/scatter 的两级流水，而不是单纯某个 HashMap 或 virtual read copy。

当前判断:

- `fill response workers` 不负责解压是合理的默认设计；把 scatter/解压放入 native custom
  command 的 selected fused 实验没有稳定收益，并会让 native IO/decode worker 与 scatter
  竞争同一槽位。
- 继续提升到 `4000 batch/s` 需要改变流水结构：response worker 不应长时间占着 fill 线程等
  ordered native bytes；或者 native block completion 需要直接写最终 sparse output，省掉
  “native decode -> Vec<u8> -> response scatter”的阶段边界。

### 19.15 Selected sparse preplan and bubble removal

本轮验证了“吞吐低是流水气泡/等待，而不是线程没铺开”的判断，并保留两个正向改动:

- projected sparse `selected_only` 默认启用 request-stage selected preplan：
  request worker 先读 CSR index、完成 selected data planning，然后把 selected data access items
  提前提交给 native scheduled executor。保留 `SCDATA_PREPLAN_SELECTED_SPARSE=0` 作为回退。
- selected preplan fastpath 不再逐 group scatter；response worker 先 drain 本 batch 已 schedule
  的 selected data groups，再调用已有的批量 projected scatter fast path
  `scatter_sparse_data_groups_projected_checked_with_projected_indices`，重新利用
  `try_scatter_projected_groups_fast_copy_cast`。

off-CPU 近似采样:

- worker 没有 `perf`/`bpftrace`/`offcputime-bpfcc`，有 `/bin/strace`。本轮用
  `/proc/<pid>/task/*/wchan` 采样。
- 采样产物：
  `outputs/native-synthetic/offcpu-selected-preplan-fastpath-20260702-112553.{json,wchan.tsv}`。
- 该 run 为 `selected_only + sorted CSR opt-in + fill72/native48/io48 + block8192`，
  `65536` batches，结果 `4035.7 batch/s`，`27.86 GB/s` virtual read。
- wchan top 以 `futex_wait_queue_me` 为主，`D` 态样本很少，且主要是
  `mmap/madvise/rwsem` 方向；这支持“剩余低利用率来自队列/阶段等待和输出 buffer
  分配尾部开销”，不是 affinity/quota/SMT 或 virtual IO lookup。

关键对照:

| run | config | preplan | batch/s | read GB/s | note |
|---|---|---|---:|---:|---|
| `outputs/native-synthetic/scheduled-random-block8192-selected-preplan-off-cmp-fill72-native48-io48-b8192-20260702.json` | fill72/native48/io48, block8192 | off | 1166.2 | 8.05 | 老路径：response 侧二次 selected planning/data scheduling，严重气泡 |
| `outputs/native-synthetic/scheduled-random-block8192-selected-preplan-on-cmp-fill72-native48-io48-b8192-20260702.json` | 同上 | on | 3561.2 | 24.59 | request-stage selected preplan 生效 |
| `outputs/native-synthetic/scheduled-random-block8192-selected-preplan-default-fill72-native48-io48-b65536-20260702.json` | 同上，默认 preplan | default | 3969.5 | 27.40 | preplan 默认启用后长跑接近 4k |
| `outputs/native-synthetic/scheduled-random-block8192-selected-preplan-default-bulkscatter-fill72-native48-io48-b65536-20260702.json` | 同上 + bulk selected scatter | default | 4247.7 | 29.32 | full scheduled synthetic random host path passes 4000 |
| `outputs/native-synthetic/scheduled-sequential-block8192-selected-preplan-default-bulkscatter-fill72-native48-io48-b65536-20260702.json` | 同上，sequential | default | 16520.5 | 8.34 | sequential same config; `0.505 MB/batch`, `2.18 ops/batch` |
| `outputs/native-synthetic/scheduled-sequential-block64k-selected-preplan-default-bulkscatter-fill72-native48-io48-b65536-20260702.json` | sequential, block64K | default | 14004.0 | 7.32 | larger block is slower for current selected preplan path |

真实 GPFS no-profile 回归:

| run | order | dtype | config | batch/s | output GB/s | note |
|---|---|---|---|---:|---:|---|
| `outputs/native-runs/release-real-random32-current-preplan-bulkscatter-noprofile-fill46-pref8192-max524288-20260702-113931.jsonl` | random | stored -> u32 | `fill=46 io=48 decode=1 access=1 native=96`, prefetch 8192 | 1517.0 | 3.18 | `profile=false`, real GPFS 32 datasets |
| `outputs/native-runs/release-real-sequential-u16-current-preplan-bulkscatter-noprofile-fill76-pref512-max524288-20260702-114006.jsonl` | sequential | u16 | `fill=76 io=48 decode=1 access=1 native=96`, prefetch 512 | 17568.7 | 18.42 | `profile=false`, real GPFS 32 datasets |

当前线程配置:

- `fill_workers=72`：DataBank request/response CPU worker pool；response worker 负责 output
  alloc 和 sparse scatter，request worker 现在也会做 selected index preplan。
- `native_workers=48`：scheduled native executor；负责 native ordered/block load、Blosc-LZ4
  partial decode 和 decoded group bytes emission。
- `io_workers=48`：IO pool；对 `VirtualChunkIo` 因 `prefers_inline_reads()` 基本不参与，
  真实 GPFS 下才承担文件 read 调度。

结论:

- “没铺开”已经排除；真正收益来自消除 selected sparse 的阶段气泡。
- `fill response workers` 不直接负责解压仍是当前正确分工：native workers 保持 IO/decode
  locality，response workers 做 output scatter。之前把 scatter 也塞进 native custom fused worker
  会让 native worker 槽被 scatter 占住，实测没有稳定收益。
- full scheduled synthetic random 100G host-path `>=4000 batch/s` 已通过：
  `4247.7 batch/s`。下一步若继续追 sequential 40k 或更高 random，需要优先做 output buffer
  reuse/direct sparse scatter，减少 `mmap/madvise` 和 `Vec<u8>/Arc<[u8]>` 中间物化。

### 19.16 Real full-catalog random/sequential no-profile

本轮新增 `profile_real_access` CPU 采样字段，release no-profile 结果会记录进程级
`user/system CPU seconds`、平均占用 core 数和按配置线程数归一化的利用率。测试仍使用
worker `ms-0701-210140-1018388-nknq4`，`nproc=96`，`engine=rust-core`，
`mode=scheduled`，真实 GPFS catalog 为 32 个 Homo sapiens `.zarr.zip`，总计
`4,268,896` cells。

这组测试使用 `--max-cells 0`，即真实全量 cell 空间；random 为全 catalog shuffle 后测
8192 个 batch，sequential 为同 catalog 顺序测 8192 个 batch。

| run | order | dtype | config | batch/s | output GB/s | avg cores | util | note |
|---|---|---|---|---:|---:|---:|---:|---|
| `outputs/native-runs/release-real-random32-current-real-allcells-fill32-pref8192-b8192-20260702-120717.jsonl` | random | stored -> u32 | `fill=32 io=48 decode=1 access=1 native=96`, prefetch 8192 | 425.4 | 0.89 | 25.2 | 26.3% | `sampled_cells=4268896`, `avg_parts_per_batch=109.1` |
| `outputs/native-runs/release-real-sequential-u16-current-real-allcells-fill56-pref512-b8192-20260702-120717.jsonl` | sequential | u16 | `fill=56 io=48 decode=1 access=1 native=96`, prefetch 512 | 14941.1 | 15.67 | 48.6 | 50.7% | `avg_parts_per_batch=1.00` |

对照 `max-cells=524288` 样本回归:

- random 样本仍为 `1426-1517 batch/s`，CPU 平均约 `18.7 cores / 19.5%`。
- sequential 样本为 `16920-18185 batch/s`，CPU 平均约 `50-56 cores / 52-58%`。

因此当前 “400 batch/s” 是真实全量 random 场景触发的等待型低利用率，不是 affinity、
quota 或线程没启动；96 线程可见，但平均只跑约 25 个 core。sequential 仍明显高于 random，
但距离新的 `32000 batch/s` 目标还有约 `2.1x`。

已拒绝且未保留的 read_all 预载实验:

- request-stage 提前读完 read_all 小 data groups 后交给 response scatter，使 sequential
  `fill=56/native=96` 从约 `13603 batch/s` 降到 `11237 batch/s`，CPU 利用率也从
  `52.1%` 降到 `33.1%`。
- 这说明把 read_all 数据整体预载到 request 阶段会破坏 native/response overlap，不是正确的
  消气泡方向。当前代码不保留 `SCDATA_PRELOAD_READALL_SPARSE` 分支。

更大 prefetch window 复测:

| run | prefetch/access/decode/ready | batch/s | output GB/s | avg cores | setup seconds | note |
|---|---:|---:|---:|---:|---:|---|
| `outputs/native-runs/release-real-random32-current-real-allcells-fill32-pref8192-b8192-20260702-121617.jsonl` | 8192 | 456.8 | 0.96 | 27.0 | 20.3 | same worker idle rerun |
| `outputs/native-runs/release-real-random32-prefwindow-16384-allcells-fill32-b8192-20260702-122402.jsonl` | 16384 | 424.6 | 0.89 | 24.9 | 39.8 | larger window did not improve steady-state |

一次连续 sweep 中 `8192/16384/24576` 分别为约 `432.0/445.3/434.2 batch/s`，
但该 sweep 的输出文件名被脚本覆盖，只作为趋势参考。结论是继续增大 scheduled/access
prefetch window 主要增加 `stream_setup_seconds`，没有把 full random 拉出 `~430-460 batch/s`
平台区间；瓶颈不是窗口深度不够，而是每 batch 约 `109` 个分散 part/completion 的 latency
和 ordered drain。

### 19.17 Full-catalog retest and remaining gap to upgraded targets

本轮目标已升级为更严格的全量真实场景指标:

- real GPFS random sparse: `1600 batch/s`。
- real GPFS sequential: `32000 batch/s`。
- virtual IO: 50G / 100G 指标 `4000 / 8000 batch/s`。
- 测试必须覆盖 32 datasets、所有 cells / all batches。

当前代码状态:

- request-stage selected sparse preplan 默认开启，random full-catalog 已经不再是
  response 侧二次 selected planning 造成的老气泡。
- projected sparse read_all-small 增加 compact output-row 判定；random 多 dataset scatter
  行稀疏时默认不再误走 read_all-small，sequential compact rows 仍保留 read_all-small。
- native full Blosc slice 已支持 `NativeMode::Force`，避免 sequential full/all 在 EOF 附近
  因 full slice 不支持而 `PrefetchCancelled`。
- full Blosc item 的批量 native load 已验证不是主要 sequential 瓶颈；8192 batch 和全量结果
  均未显示稳定提升，当前 sequential 主要卡在 dense output materialization / sparse scatter。

验证命令摘要:

- 本地:
  - `cargo fmt --manifest-path rust/scdata/Cargo.toml`
  - `cargo test --manifest-path rust/scdata/Cargo.toml --lib native:: --no-default-features --features pybind-bench --offline`
  - `cargo check --manifest-path rust/scdata/Cargo.toml --no-default-features --features profile --bin profile_real_access --offline`
- worker release:
  - `cargo build --manifest-path rust/scdata/Cargo.toml --release --no-default-features --features profile --bin profile_real_access --offline`

最新真实全量 release no-profile 结果，worker
`ms-0701-210140-1018388-nknq4`，`nproc=96`，catalog cells `4,268,896`:

| run | order | dtype | measured / seen batches | config | batch/s | output GB/s | avg cores | util |
|---|---|---|---:|---|---:|---:|---:|---:|
| `outputs/native-runs/release-real-random32-current-allbatches-20260702-131929.jsonl` | random | stored -> u32 | `33287 / 33351` | `fill=32 io=48 decode=1 access=1 native=96`, window 8192 | 566.45 | 1.19 | 28.23 | 29.4% |
| `outputs/native-runs/release-real-sequential-u16-sortedcsr-current-allbatches-20260702-132115.jsonl` | sequential | u16 | `33287 / 33351` | `fill=56 io=48 decode=1 access=1 native=96`, window 512, sorted CSR opt-in | 16486.88 | 17.29 | 60.51 | 63.0% |

更大窗口与 worker 数 sweep 结论:

- random `prefetch/access/decode/ready=8192 -> 16384 -> 24576` 没有提升 steady-state
  throughput；`16384` 反而把 setup 拉到约 `39.8s`。
- sequential `fill_workers=56/72/96` 在 8192 batch 短跑都停在 `15.9k-16.1k batch/s`，
  没有因更多 response worker 接近 `32000`。
- sequential `prefetch/access/decode/ready=512 -> 8192` 变慢，setup 从约 `0.30s` 增到
  约 `0.96s`。
- `SCDATA_READALL_SELECTED_SCATTER=1` 被排除：8192 batch sequential 从约 `16k batch/s`
  降到约 `1.2k batch/s`，因为每 batch 重新 selected-plan index scan 的代价远大于少 scatter
  的收益。
- `SCDATA_ASSUME_SORTED_CSR_INDICES=1` 在 8192 batch 短跑有小幅收益
  (`16048.6 -> 17252.2 batch/s`，checksum 一致)，但全量 sequential 仍为
  `16486.9 batch/s`，不能解决 2x 缺口。

profile 8192 sequential 的关键数字:

- `batches_per_s=16677.2`，`output_gb_per_s=17.49`，CPU 平均 `54.2 cores`。
- `iopool_operation_read=4911 MiB`，约 `0.60 MiB/batch`；IO 不是 sequential 主瓶颈。
- worker-sum:
  - `databank_prefetch_memory_scheduled_drain_ms=19999.7`
  - `databank_prefetch_assemble_scatter_ms=8329.7`
  - `databank_prefetch_assemble_alloc_ms=2363.3`
  - `databank_prefetch_assemble_total_ms=35162.8`

当前判断:

- random full-catalog 低吞吐不是 CPU 没铺开；96 线程可见但平均约 28 core，是等待/IO
  read-amplification 限制。profile 过的 random 约 `30 MiB/batch`、约 `255 IO calls/batch`；
  在 20G GPFS 下这天然落在 `~560-650 batch/s`，要到 `1600 batch/s` 必须把物理读取降到
  `~12 MiB/batch` 量级，而不是继续加 worker 或 prefetch。
- sequential full-catalog 不是 GPFS IO 限制；它在 dense output 写出和 sparse projected
  scatter/materialization 上限附近，当前约 `17 GB/s` output，目标 `32000 batch/s` 需要
  `~33.5 GB/s` output。
- 下一步结构性方向:
  1. 对 random: 减少真实 full-catalog random 每 batch 触碰的 compressed blocks / IO calls；
     单纯 coalescing gap、prefetch window、IO/native worker 数都已排除。
  2. 对 sequential: read_all-small data 不应再走
     `native decode -> Vec<u8> -> response drain -> scatter` 的阶段边界；需要 request plan
     级别支持 data 延后/fused scatter，或把 native block completion 直接 scatter 到最终 dense
     output。
  3. 对 virtual 100G `8000 batch/s`: 当前 full scheduled synthetic 曾达到 `4247.7 batch/s`
     级别，说明 preplan/bulk scatter 已通过 50G 档；要到 8000 仍需消除 scheduled CSR
     assemble 的中间 materialization 和 ordered drain。

因此本节不能标记最终目标完成；当前工作已经把“没铺开 / prefetch 不够 / worker 数不够 /
readall selected scatter / full-slice IO 串行”逐项排除，剩余是访问布局和阶段边界本身。

### 19.18 Singleton projected rows and response-limit diagnostics

本轮继续沿着“减少气泡”和“降低 random read amplification”推进，保留一个默认启用的正向
heuristic:

- `should_read_all_small_projected_sparse_plan` 不再把单个 output row 视为 read_all-small
  候选。也就是说，projected sparse 的单行 part 默认走 selected-only，而不是读取该 cell
  的整行 data。
- 原因: full-catalog random batch 平均有约 `109` 个 dataset part，很多 part 只有 1 个 cell；
  这些 singleton 以前会因为“行天然 compact”而继续 read_all-small，增加真实 random 的 data
  bytes 和 native drain。
- 多行 compact sequential part 仍保留 read_all-small，因此顺序访问的 read_all 优势不被全局关闭。

同时新增一个默认不改变行为的诊断开关:

- `SCDATA_SCHEDULED_RESPONSE_LIMIT=N`：覆盖 scheduled producer 的 active response job 上限。
  用于验证 response job 长时间阻塞在 native bytes 时，是否挤占 request planning worker。

负向诊断结果:

- `SCDATA_NATIVE_COMMAND_GROUPING=1` 对 sequential 小 command 很差：
  `outputs/native-runs/release-real-sequential-u16-commandgroup-m8192-20260702-132416.jsonl`
  只有 `6443.9 batch/s`，CPU `22.8 cores`。把多个 batch command 合进一个 native command
  会降低并行度，不是消气泡方向。
- `SCDATA_SCHEDULED_RESPONSE_LIMIT=16` 单独短跑为
  `14343.9 batch/s`，低于默认约 `16k+`。简单减少 active response、给 request 多留 worker
  不能提升 sequential。
- oversubscribe `fill_workers=128` 为 `15398.3 batch/s`，CPU `62.5 cores`；更多阻塞线程
  没有转化为更高吞吐。

最新真实全量 release no-profile 对比:

| run | order | change | measured / seen | batch/s | output GB/s | avg cores | util | checksum |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `outputs/native-runs/release-real-random32-current-allbatches-20260702-131929.jsonl` | random | before singleton heuristic | `33287 / 33351` | 566.45 | 1.188 | 28.23 | 29.4% | 2381 |
| `outputs/native-runs/release-real-random32-singleton-selected-default-allbatches-20260702-133345.jsonl` | random | singleton selected-only default | `33287 / 33351` | 590.65 | 1.239 | 31.06 | 32.35% | 2381 |
| `outputs/native-runs/release-real-sequential-u16-sortedcsr-current-allbatches-20260702-132115.jsonl` | sequential | previous best reference | `33287 / 33351` | 16486.88 | 17.288 | 60.51 | 63.03% | 2514 |
| `outputs/native-runs/release-real-sequential-u16-singleton-selected-default-allbatches-20260702-133520.jsonl` | sequential | singleton selected-only default | `33287 / 33351` | 16985.85 | 17.811 | 63.89 | 66.55% | 2514 |

短跑验证:

- random 8192 默认 heuristic:
  `outputs/native-runs/release-real-random32-singleton-selected-default-m8192-20260702-133208.jsonl`
  为 `467.5 batch/s`，与手动 `SCDATA_READ_ALL_SMALL_PROJECTED_SPARSE=0` 的
  `476.1 batch/s` 同档，checksum 一致。
- sequential 8192 默认 heuristic:
  `outputs/native-runs/release-real-sequential-u16-singleton-selected-default-m8192-20260702-133306.jsonl`
  为 `16619.7 batch/s`，没有破坏 sequential read_all-small 主路径。

当前严格目标状态仍未完成:

- real GPFS random: `590.7 / 1600 batch/s`。
- real GPFS sequential: `16985.9 / 32000 batch/s`。
- CPU 利用率有改善但仍低: random `32.35%`，sequential `66.55%`。

下一步更可能有效的方向:

- random: 继续减少 singleton/multi-dataset part 的 touched compressed block 和 IO calls；
  单行 selected-only 只是小幅降低读放大，距离 `1600` 仍需要约 `2.7x`。
- sequential: 需要把 read_all-small 的 native data completion 直接写入最终 dense output，
  或至少消除 `Vec<u8>/Arc<[u8]>` materialization 和 ordered drain 阶段边界；单纯调 response
  limit、command grouping、fill worker 数都已排除。

### 19.19 Sorted planning opt-in and native worker sweep

本轮把 `SCDATA_ASSUME_SORTED_CSR_INDICES=1` 从 scatter 阶段扩展到 selected planning 阶段:

- 当投影是连续 source range，且调用方显式假设 CSR row indices 有序时，selected planning
  在遇到 `gene >= selected_end` 后立即结束当前 piece 扫描。
- 默认仍关闭。原因是现有 `SparseCsrSpec` 只校验 `indptr` 单调和 index dtype，不证明每行
  indices 有序；完整验证需要读取全量 `indices`，成本接近一次全数据扫描。

短跑结果:

| run | order | env/config | batch/s | output GB/s | avg cores | checksum |
|---|---|---|---:|---:|---:|---:|
| `outputs/native-runs/release-real-sequential-u16-singleton-selected-default-m8192-20260702-133306.jsonl` | sequential | default | 16619.7 | 17.43 | 56.22 | 50 |
| `outputs/native-runs/release-real-sequential-u16-singleton-sortedcsr-m8192-20260702-133728.jsonl` | sequential | sorted scatter only before planning change | 17483.3 | 18.33 | 52.65 | 50 |
| `outputs/native-runs/release-real-random32-singleton-selected-default-m8192-20260702-133208.jsonl` | random | default | 467.5 | 0.98 | 31.07 | 719 |
| `outputs/native-runs/release-real-random32-singleton-sortedcsr-planbreak-m8192-20260702-134151.jsonl` | random | sorted scatter + planning early-break | 476.2 | 1.00 | 30.71 | 719 |

全量结果不支持把 sorted 假设作为达标依据:

- `outputs/native-runs/release-real-sequential-u16-singleton-sortedcsr-allbatches-20260702-133811.jsonl`
  为 `13618.2 batch/s`，低于 default `16985.9 batch/s`，尽管 checksum 一致。
- 因此 sorted CSR 只能作为 opt-in 诊断或在上游数据明确保证行内有序时使用；不能默认启用，
  也不能作为 `32000 batch/s` 的完成证据。

native worker 数 sweep:

| run | order | native workers | batch/s | avg cores | conclusion |
|---|---|---:|---:|---:|---|
| `outputs/native-runs/release-real-sequential-u16-singleton-selected-default-m8192-20260702-133306.jsonl` | sequential | 96 | 16619.7 | 56.22 | current better |
| `outputs/native-runs/release-real-sequential-u16-singleton-native48-m8192-20260702-134314.jsonl` | sequential | 48 | 14274.6 | 36.14 | underfeeds response |
| `outputs/native-runs/release-real-random32-singleton-selected-default-m8192-20260702-133208.jsonl` | random | 96 | 467.5 | 31.07 | current better |
| `outputs/native-runs/release-real-random32-singleton-native48-m8192-20260702-134335.jsonl` | random | 48 | 443.0 | 28.62 | underfeeds response |

结论:

- 降低 native worker 数不是减少争用的方向；真实 random/sequential 都需要当前 `native=96`
  级别的 native executor 并行度。
- sorted planning early-break 只能减少一部分 CPU 扫描，对 random 仅 `~2%` 短跑收益。
- 当前剩余缺口仍是结构性的: random 需要减少 touched block/IO calls，sequential 需要减少
  native data materialization 和 ordered drain。

### 19.20 Allocation and fused-response diagnostics

本轮继续验证两个“减少气泡/提高 CPU 负载”的候选方向，并把无效路径排除。

#### `alloc_zeroed` output allocation

实验把 `zeroed_values<T>` / `zeroed_byte_vec` 从 `Vec::with_capacity + write_bytes`
替换为 global allocator 的 `alloc_zeroed`，希望 allocator 走 lazy zero page，降低 response
侧输出 buffer 初始化成本。短跑 sequential 有一次提升，但真实全量没有稳定收益，且 random
略回退，因此已恢复到原来的显式零填充实现。

| run | order | change | measured / seen | batch/s | output GB/s | avg cores | checksum |
|---|---|---|---:|---:|---:|---:|---:|
| `outputs/native-runs/release-real-random32-singleton-selected-default-allbatches-20260702-133345.jsonl` | random | before alloc_zeroed | `33287 / 33351` | 590.65 | 1.239 | 31.06 | 2381 |
| `outputs/native-runs/release-real-random32-alloczeroed-allbatches-20260702-135325.jsonl` | random | alloc_zeroed | `33287 / 33351` | 582.46 | 1.222 | 30.55 | 2381 |
| `outputs/native-runs/release-real-sequential-u16-singleton-selected-default-allbatches-20260702-133520.jsonl` | sequential | before alloc_zeroed | `33287 / 33351` | 16985.85 | 17.811 | 63.89 | 2514 |
| `outputs/native-runs/release-real-sequential-u16-alloczeroed-allbatches-20260702-134952.jsonl` | sequential | alloc_zeroed | `33287 / 33351` | 16764.38 | 17.579 | 60.93 | 2514 |

#### Deferred selected data + native fused scatter

新增默认关闭的诊断开关:

- `SCDATA_PREPLAN_SELECTED_SPARSE_DEFER_DATA=1`
- 与 `SCDATA_NATIVE_FUSED_SCATTER=1` 配合使用。

语义:

- request 阶段仍做 selected sparse preplan，并 preload index bytes。
- 但 selected data 不再放入 batch 的 `ScheduledBatchAccess`。
- response 阶段将 preselected data groups 交给 native custom command，由 native worker
  ordered load 后直接 scatter 到最终 dense output。

这条路径用于验证“fill response worker 等 native bytes / response 再 scatter”是否是 random
主瓶颈。8192 batch 真实场景短跑结果:

| run | order | config | batch/s | output GB/s | seconds | setup s | avg cores | checksum |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `outputs/native-runs/release-real-random32-deferfused-defaultcheck-m8192-20260702-140012.jsonl` | random | default | 476.23 | 0.999 | 17.202 | 20.743 | 31.40 | 719 |
| `outputs/native-runs/release-real-random32-deferfused-on-m8192-20260702-140012.jsonl` | random | defer+fused | 447.25 | 0.938 | 18.317 | 13.906 | 33.48 | 719 |
| `outputs/native-runs/release-real-sequential-u16-deferfused-defaultcheck-m8192-20260702-140012.jsonl` | sequential | default | 16513.47 | 17.316 | 0.496 | 0.926 | 60.74 | 50 |
| `outputs/native-runs/release-real-sequential-u16-deferfused-on-m8192-20260702-140012.jsonl` | sequential | defer+fused | 17004.20 | 17.830 | 0.482 | 0.926 | 60.80 | 50 |

结论:

- random 没有受益，反而下降 `~6%`。这说明 full-catalog random 当前主要不是“response
  worker 没负责解压/scatter”这一单点；更大的限制仍是每 batch 约 `109` 个 dataset part
  带来的 touched compressed blocks / IO calls / read amplification。
- sequential 有 `~3%` 短跑收益，但还远不足以解释 `32000 batch/s` 缺口。它最多说明
  sequential 的阶段边界有局部成本，不能作为默认开启依据。
- 该路径保持默认关闭，仅作为后续 fused design 的诊断开关。若要达成目标，需要更激进的
  window-level regrouping 或 native block completion 直接写最终 output，而不是简单把 selected
  data load 从 request scheduled reader 挪到 response native custom。

### 19.21 Native block payload cache diagnostic

本轮实现一个默认关闭的 native block payload cache，用来验证 full-catalog random 的低
CPU 利用率是否主要来自跨 batch 重复读取同一 compressed block。

新增诊断开关:

- `SCDATA_NATIVE_BLOCK_CACHE_BYTES=N`
- 默认 `0`，不创建 cache，不影响现有路径。
- cache key 是 native partial block request 的 exact `(file, offset, len)`。
- cache 位于 `NativeLoadModule` 层；命中时直接返回 compressed block payload，miss 仍进入
  原来的 coalesce + IO 路径。
- 当前实现使用 8 个 shard，每个 shard 独立 `HashMap + FIFO eviction`，避免单个全局锁
  成为 96 native worker 的热点。

低层单测:

- `cargo test --manifest-path rust/scdata/Cargo.toml --lib block_payload_cache_reuses_exact_request --no-default-features --features pybind-bench --offline`
  通过，验证第二次 exact block request 不再触发 IO。

profile 8192 batch 结果显示 cache 命中真实重复 block:

| run | cache | batch/s | read MiB | IO calls | MiB/batch | ops/batch | checksum |
|---|---|---:|---:|---:|---:|---:|---:|
| `outputs/native-runs/profile-real-random32-blockcache-off-m8192-20260702-141033.jsonl` | off | 403.20 | 248002.7 | 2089404 | 30.27 | 255.05 | 719 |
| `outputs/native-runs/profile-real-random32-blockcache-32g-m8192-20260702-141033.jsonl` | 32GiB | 792.21 | 3.8 | 42 | ~0 | 0.01 | 719 |

注意: profile measured 阶段的 IO 几乎为 0，是因为 prefetch/setup 阶段已经把第一窗口的
block payload 放进 cache；因此最终证据必须看 all-batches no-profile。

真实全量 random all-batches:

| run | config | batch/s | output GB/s | seconds | setup s | avg cores | util | checksum |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `outputs/native-runs/release-real-random32-singleton-selected-default-allbatches-20260702-133345.jsonl` | cache off | 590.65 | 1.239 | 56.356 | 15.175 | 31.06 | 32.35% | 2381 |
| `outputs/native-runs/release-real-random32-blockcache-32g-allbatches-20260702-141248.jsonl` | 32GiB, single-lock prototype | 1068.55 | 2.241 | 31.152 | 12.261 | 38.24 | 39.83% | 2381 |
| `outputs/native-runs/release-real-random32-blockcache32g-shard8-allbatches-20260702-142112.jsonl` | 32GiB, 8-shard current | 1064.73 | 2.233 | 31.263 | 10.914 | 38.59 | 40.20% | 2381 |
| `outputs/native-runs/release-real-random32-blockcache-64g-allbatches-20260702-141410.jsonl` | 64GiB prototype | 1044.07 | 2.190 | 31.882 | 11.158 | 37.99 | 39.58% | 2381 |

sequential guardrail:

| run | config | batch/s | output GB/s | avg cores | checksum |
|---|---|---:|---:|---:|---:|
| `outputs/native-runs/release-real-sequential-u16-singleton-selected-default-allbatches-20260702-133520.jsonl` | cache off | 16985.85 | 17.811 | 63.89 | 2514 |
| `outputs/native-runs/release-real-sequential-u16-blockcache-32g-allbatches-20260702-141535.jsonl` | cache 32GiB | 7818.28 | 8.198 | 35.37 | 2514 |

结论:

- block payload cache 把 full-catalog random 从 `590.65` 提到当前实现约 `1064.73 batch/s`
  (`~1.8x`)，checksum 一致，并把 CPU 利用率从 `~32%` 提到 `~40%`。这证明 random 的等待
  主要来自跨 batch/block 重复读取和 IO call 放大。
- 32GiB 已覆盖主要复用；64GiB 没有继续提升，说明剩余缺口不是单纯 cache 容量。
- cache 不能默认开启: sequential 全量从 `16985.85` 降到 `7818.28 batch/s`。顺序场景
  的 block 复用收益小，cache 查询/插入和 eviction 反而成为开销。
- 因此该 cache 目前只作为 random 诊断和后续 mode-specific 策略的基础。要达到
  `random 1600 batch/s`，下一步需要把 cache 变成只服务 random selected sparse 的
  targeted/in-flight dedup，或在 window-level regrouping 中共享 native block completion，
  避免 sequential/read_all 路径支付 cache 成本。

### 19.22 Prefetch window sweep

本轮验证“继续放大 prefetch/ahead 窗口是否能消除 random 流水气泡”。测试保持 worker
配置不变:

- `io_workers=48`
- `native_fused_workers=96`
- `decode_workers=1`
- `access_workers=1`
- `fill_workers=32`
- `batch_size=128`
- `limit_datasets=32`
- `genes=4096`
- `projected_sparse_data_strategy=selected_only`

只同时放大这些窗口:

- `native-prefetch-blocks`
- `prefetch-steps`
- `access-prefetch-steps`
- `decode-ahead-steps`
- `ready-ahead-steps`

8192-batch 短跑结果:

| run | window | measured batch/s | effective batch/s incl. setup | measured s | setup s | avg cores | util | checksum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `outputs/native-runs/release-real-random32-prefetch8192-m8192-20260702-143034.jsonl` | 8192 | 466.95 | 229.09 | 17.543 | 18.216 | 30.07 | 31.32% | 719 |
| `outputs/native-runs/release-real-random32-prefetch16384-m8192-20260702-143034.jsonl` | 16384 | 473.64 | 143.02 | 17.296 | 39.982 | 29.92 | 31.16% | 719 |
| `outputs/native-runs/release-real-random32-prefetch32768-m8192-20260702-143034.jsonl` | 32768 | 3231.82 | 100.86 | 2.535 | 78.685 | 32.25 | 33.59% | 719 |

短跑里 `32768` 的 measured throughput 暴涨是计时口径造成的: 更深窗口把大量 IO/解压
提前压到 warmup/setup 阶段，measured 段在消费已经完成的预取结果。因此必须看包含
setup 的 effective throughput。

真实全量 random all-batches 复测:

| run | window | measured batch/s | effective batch/s incl. setup | measured s | setup s | total s | avg cores | util | checksum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `outputs/native-runs/release-real-random32-prefetch8192-allbatches-20260702-143830.jsonl` | 8192 | 587.53 | 464.83 | 56.656 | 14.956 | 71.611 | 30.21 | 31.47% | 2381 |
| `outputs/native-runs/release-real-random32-prefetch16384-allbatches-20260702-143521.jsonl` | 16384 | 808.87 | 455.33 | 41.152 | 31.953 | 73.105 | 29.72 | 30.96% | 2381 |
| `outputs/native-runs/release-real-random32-prefetch32768-allbatches-20260702-143521.jsonl` | 32768 | 3952.79 | 441.75 | 8.421 | 66.932 | 75.353 | 32.58 | 33.94% | 2381 |

结论:

- 更大的窗口没有提高端到端 random throughput。`8192 -> 16384 -> 32768` 只是把工作从
  measured 段挪到 setup/warmup 段，总耗时反而从 `71.611s` 增到 `75.353s`。
- CPU 利用率仍在 `~31-34%`，没有因为窗口更深而接近满载，说明当前 idle 不是简单的
  prefetch depth 不够。
- 继续放大 prefetch 窗口不是主要优化方向。下一步应集中在 block reuse / in-flight
  dedup / window-level regrouping，减少 random 场景的重复 compressed block load 和
  IO call 放大。

### 19.23 Native decoded block cache diagnostic

本轮继续验证 random 的等待是否主要来自重复 block decode/unshuffle。新增默认关闭的
decoded block cache:

- `SCDATA_NATIVE_DECODED_BLOCK_CACHE_BYTES=N`
- key 复用 native block payload cache 的 exact `(file, offset, len)`。
- cache value 是 full decoded/unshuffled block `Arc<[u8]>`。
- cache 开启时，miss 会强制 full decode + full unshuffle 后 scatter 并插入 cache；hit 直接
  scatter decoded block。

该路径只用于诊断和后续 random-specific 策略；默认仍关闭，避免 sequential 支付 cache
查询、full decode 和 eviction 成本。

8192-batch random 短跑:

| run | config | measured batch/s | effective batch/s incl. setup | output GB/s | seconds | setup s | avg cores | util | checksum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `outputs/native-runs/release-real-random32-payloadcache32g-m8192-20260702-144216.jsonl` | payload cache 32GiB | 863.51 | 320.57 | 1.811 | 9.487 | 16.067 | 38.14 | 39.73% | 719 |
| `outputs/native-runs/release-real-random32-payload32g-decoded64g-m8192-20260702-144216.jsonl` | payload 32GiB + decoded 64GiB | 1630.33 | 366.56 | 3.419 | 5.025 | 17.323 | 34.15 | 35.58% | 719 |

真实 full all random:

| run | config | measured batch/s | effective batch/s incl. setup | output GB/s | seconds | setup s | avg cores | util | checksum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `outputs/native-runs/release-real-random32-singleton-selected-default-allbatches-20260702-133345.jsonl` | cache off | 590.65 | 465.35 | 1.239 | 56.356 | 15.175 | 31.06 | 32.35% | 2381 |
| `outputs/native-runs/release-real-random32-blockcache32g-shard8-allbatches-20260702-142112.jsonl` | payload cache 32GiB | 1064.73 | 789.21 | 2.233 | 31.263 | 10.914 | 38.59 | 40.20% | 2381 |
| `outputs/native-runs/release-real-random32-payload32g-decoded64g-allbatches-20260702-144351.jsonl` | payload 32GiB + decoded 64GiB | 1828.64 | 914.08 | 3.835 | 18.203 | 18.213 | 33.89 | 35.30% | 2381 |

结论:

- decoded cache 证明 full-catalog random 的大头不是 worker 没铺开，而是同一批/跨批重复
  block 的 compressed payload 与 decoded payload 复用没有在当前流水中表达出来。
- 按 benchmark 现有 measured 口径，random full all 已超过新的 `1600 batch/s` 目标
  (`1828.64 batch/s`)，checksum 一致。
- 但端到端包含 setup 只有 `914.08 batch/s`，且该策略依赖大内存 cache，不应作为
  sequential 默认路径。生产化方向应把它收敛成 random selected sparse 的 targeted
  in-flight dedup / window-local decoded block sharing，而不是全局 LRU。

### 19.24 Sequential 32k follow-up

更大 prefetch window 被排除后，本轮重新校准 sequential。需要注意: sequential 目标口径是
`dtype=u16`；`dtype=stored` 在当前数据上解析为 `u32`，每 batch 输出字节翻倍，不能和
`sequential-u16` 目标直接比较。

关键配置差异:

- sequential 高速配置使用 `prefetch/access/decode/ready = 512/512/512/512`。
- random 调参使用的 `8192/8192/8192/8192` 会显著增加 setup 和队列压力。
- sequential 旧 baseline 使用更激进 coalesce: `gap=1MiB, waste=0.9, merged_len=8MiB`。
- 文档中早先 `~17.6k batch/s` 的结果是 4096-batch 短测，不是 full all。

#### Coalesce / worker / IO backend sweep

正确 sequential 参数下，coalesce `merged_len=8MiB` 仍最好；继续增大合并长度没有收益:

| run | merged_len | batch/s | output GB/s | avg cores | checksum |
|---|---:|---:|---:|---:|---:|
| `outputs/native-runs/release-real-sequential-u16-coalesce-gap1m-waste09-merge8388608-allbatches-20260702-145003.jsonl` | 8MiB | 17373.76 | 18.218 | 64.99 | 2514 |
| `outputs/native-runs/release-real-sequential-u16-coalesce-gap1m-waste09-merge16777216-allbatches-20260702-145003.jsonl` | 16MiB | 16563.92 | 17.368 | 62.71 | 2514 |
| `outputs/native-runs/release-real-sequential-u16-coalesce-gap1m-waste09-merge33554432-allbatches-20260702-145003.jsonl` | 32MiB | 16384.27 | 17.180 | 61.23 | 2514 |

正确 sequential 参数下的 fill/native worker 矩阵，最好仍只有 `~17.7k`:

| run | fill | native | batch/s | output GB/s | avg cores | checksum |
|---|---:|---:|---:|---:|---:|---:|
| `outputs/native-runs/release-real-sequential-u16-seqparams-fw32-nw128-allbatches-20260702-145101.jsonl` | 32 | 128 | 17735.84 | 18.597 | 58.56 | 2514 |
| `outputs/native-runs/release-real-sequential-u16-seqparams-fw56-nw96-allbatches-20260702-145101.jsonl` | 56 | 96 | 16881.20 | 17.701 | 63.00 | 2514 |
| `outputs/native-runs/release-real-sequential-u16-seqparams-fw96-nw128-allbatches-20260702-145101.jsonl` | 96 | 128 | 15775.22 | 16.541 | 70.03 | 2514 |

补测几个候选:

| run | change | batch/s | output GB/s | avg cores | note |
|---|---|---:|---:|---:|---|
| `outputs/native-runs/release-real-sequential-u16-fill76-pref512-nativepref65536-coalesce-default-allbatches-20260702-150403.jsonl` | 4096-batch 短测同类配置改 full all | 17456.29 | 18.304 | 76.07 | CPU 更高但 batch/s 未突破 |
| `outputs/native-runs/release-real-sequential-u16-ioworkers96-fw32-nw128-allbatches-20260702-145610.jsonl` | IO workers 96 | 17519.91 | 18.371 | 59.88 | 小幅，不是 2x |
| `outputs/native-runs/release-real-sequential-u16-assumesorted-on-allbatches-20260702-150215.jsonl` | `SCDATA_ASSUME_SORTED_CSR_INDICES=1` | 15916.69 | 16.690 | 64.06 | 小幅，不解决 |
| `outputs/native-runs/release-real-sequential-u16-deferfused-on-allbatches-20260702-145524.jsonl` | defer + fused scatter | 15496.08 | 16.249 | 61.23 | 负向 |
| `outputs/native-runs/release-real-sequential-u16-iobackend-uring-drivers4-fill76-allbatches-20260702-150853.jsonl` | `io_uring`, 4 drivers | 17133.16 | 17.965 | 54.87 | 与 threaded 同档 |

新增测试入口参数:

- `profile_real_access --io-backend threaded|uring`
- `profile_real_access --uring-entries N`

该参数只影响 Rust-core benchmark 的 DataBank `IoConfig`，默认仍是 threaded。`io_uring`
已验证不能直接把 sequential 推向 `32000 batch/s`。

profile 证据:

| run | batch/s | read MiB | IO calls | IO work ms | dispatch wait ms |
|---|---:|---:|---:|---:|---:|
| `outputs/native-runs/profile-real-sequential-u16-fill76-pref512-nativepref65536-coalesce-default-allbatches-20260702-150507.jsonl` | 16535.41 | 22395.5 | 92426 | 57543.2 | 13452.4 |

结论:

- 当前 best full all sequential 仍是 `~17-18k batch/s`，距离 `32000 batch/s` 约 `1.8x`。
- 增加 fill/native/IO workers、扩大 native prefetch、切换 `io_uring`、read_all、defer+fused、
  assumesorted 都不能提供 2x。
- sequential profile 仍要读 `~22 GiB`，measured 段约 `11 GiB/s`，同时 CPU 可到
  `~70-76 cores`。这说明 sequential 已进入 IO 调度 + host materialization 混合上限，
  不是“线程没铺开”或单一 prefetch depth。
- 要达到 `32000 batch/s`，下一步需要改变 sequential 的输出/访问结构，例如 window-level
  multi-batch CSR replay、输出 buffer reuse/zero-page 策略、或把多个 sequential batches
  合并成更粗粒度的 native CSR scatter，再按 batch 切分结果；继续扫现有参数收益有限。
