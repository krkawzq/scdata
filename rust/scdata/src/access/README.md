# access 模块维护文档

本文档基于当前工作区的 `rust/scdata/src/access` 源码梳理。`access`
是 `scdata` Rust crate 的公开模块，负责把上层 DataBank 的 chunk 访问请求接到
底层定位 IO、压缩解码、缓存、内存预算和可预取的顺序访问调度上。

## 模块定位

`access` 不是文件格式解析层，也不是 DataBank 的 batch 组装层。它只关心“某个文件
的某段压缩 bytes 如何被读取、可选解码、可选切片后返回”。

核心职责：

1. 接收 chunk 级访问请求，并路由到固定 scheduler shard。
2. 对相同 `ChunkKey` 的并发 raw read 做 in-flight 去重。
3. 缓存 raw compressed bytes，并在 `keep_decoded = true` 时缓存 decoded bytes。
4. 用统一内存预算约束 cache、in-flight raw read、scheduled staging 和 ready buffer。
5. 支持普通一次性访问、raw prefetch 和 predictable iterator 的 scheduled access。
6. 提供 access 层 profiling 指标。

当前 `DataBank` 通过 `databank::adapter::{IoPoolBackend, DecodePoolBackend}` 把
`iopool::IoPool` 与 `codecs::DecodePool` 适配成这里的 `IoBackend` 和
`DecodeBackend`，再用 `AccessScheduler::spawn` 启动调度器。

## 文件结构

| 文件 | 职责 |
| --- | --- |
| `mod.rs` | 子模块声明和公开 re-export。 |
| `backend.rs` | scheduler 依赖的 IO/Decode backend trait。 |
| `key.rs` | `ChunkKey`、内部 `DecodeKey` 和 shard hash。 |
| `slice.rs` | decoded bytes 的 scatter-copy slice 规范和验证计划。 |
| `error.rs` | `AccessError` 与 `AccessResult`。 |
| `cpu.rs` | access 侧 CPU worker pool，用于大块 copy/scatter materialization。 |
| `cache.rs` | pin-aware raw/decoded LRU cache。 |
| `inflight.rs` | 相同 raw chunk miss 的 in-flight 去重。 |
| `membudget.rs` | cache、读入、staging 共用的 byte budget。 |
| `scheduled.rs` | scheduled access 服务端 staging store。 |
| `scheduler.rs` | 调度器主体、普通访问、prefetch、scheduled iterator 和状态机。 |
| `profile.rs` | access profiling scope、metric 和 `AccessProfile` 封装。 |

## 对外接口总览

`mod.rs` 当前公开导出这些符号：

```rust
pub use backend::{DecodeBackend, DecodeTask, FileRef, IoBackend, IoTask};
pub use cpu::AccessCpuConfig;
pub use error::{AccessError, AccessResult};
pub use key::ChunkKey;
pub use profile::{access_profile_registry, AccessProfile};
pub use scheduler::{
    AccessConfig, AccessHandle, AccessItem, AccessRequest, AccessScheduler,
    PrefetchCancel, PrefetchRequest, ScheduledAccess, ScheduledAccessConfig,
};
pub use slice::{RangeCopy, SliceSpec};
```

下面逐项说明。

### Backend 接口

#### `FileRef`

```rust
pub struct FileRef(pub u64);
impl FileRef {
    pub fn new(id: u64) -> Self;
}
```

`FileRef` 是 access 层传给 IO backend 的不透明文件句柄。access 本身不打开文件，
也不解释 `u64` 的含义。DataBank 当前把它映射到 `IoPool` 内部的 file id。

#### `IoTask`

```rust
pub type IoTask = Pin<Box<dyn Future<Output = io::Result<Arc<[u8]>>> + Send>>;
```

一次定位读的异步结果。成功时必须返回长度等于请求 `len` 的 `Arc<[u8]>`，否则
scheduler 会把它转成 `UnexpectedEof`。

#### `DecodeTask`

```rust
pub type DecodeTask = Pin<Box<dyn Future<Output = CodecResult<Vec<u8>>> + Send>>;
```

一次 chunk 解码的异步结果。成功返回 caller-owned decoded `Vec<u8>`。

#### `IoBackend`

```rust
pub trait IoBackend: Send + Sync + 'static {
    fn submit_read(&self, file: FileRef, offset: u64, len: usize, priority: u8) -> IoTask;
}
```

access scheduler 只要求 backend 能提交定位读。`priority` 来自
`AccessConfig::default_io_priority`，数值越小优先级越高这一语义由 `iopool`
实现侧使用。

#### `DecodeBackend`

```rust
pub trait DecodeBackend: Send + Sync + 'static {
    fn submit_decode(
        &self,
        codec: SharedCodec,
        encoded: Arc<[u8]>,
        expected_size: Option<usize>,
    ) -> DecodeTask;
}
```

提交一次 decode。`codec` 是 `crate::codecs::SharedCodec`，`expected_size` 用于
codec 层 size check。identity codec 可在 access 层直接绕过 decode backend。

### Key 接口

#### `ChunkKey`

```rust
pub struct ChunkKey {
    pub file: FileRef,
    pub offset: u64,
    pub len: usize,
}
impl ChunkKey {
    pub fn new(file: FileRef, offset: u64, len: usize) -> Self;
}
```

`ChunkKey` 唯一标识一个 raw compressed chunk read。它是 raw cache key、in-flight
去重 key 和 scheduler shard routing key。

内部还有 `DecodeKey`，由 `ChunkKey + codec.cache_key() + expected_size` 组成，
用于 decoded cache 和 decode in-flight 去重。这样同一个 raw chunk 用不同 codec、
不同 expected decoded size 访问时不会误复用 decoded bytes。

shard routing 使用 `file`、`offset`、`len` 混合后的 SplitMix64，再对 shard 数取模。
这样连续对齐 offset 的 chunks 会更均匀地分布到多个 scheduler shard。

### Slice 接口

#### `RangeCopy`

```rust
pub struct RangeCopy {
    pub dst_offset: usize,
    pub src_start: usize,
    pub src_end: usize,
}
impl RangeCopy {
    pub fn new(dst_offset: usize, src_start: usize, src_end: usize) -> Self;
}
```

表示一次从 decoded input 到 output 的半开区间 copy：

```text
output[dst_offset .. dst_offset + (src_end - src_start)]
    <- input[src_start .. src_end]
```

#### `SliceSpec`

```rust
pub enum SliceSpec {
    Full,
    Scatter(Arc<[RangeCopy]>),
    #[doc(hidden)]
    Invalid(String),
}
impl SliceSpec {
    pub fn full() -> Self;
    pub fn from_triples(triples: Vec<usize>) -> Result<Self, AccessError>;
    pub fn from_optional_triples(slice: Option<Vec<usize>>) -> Result<Self, AccessError>;
}
```

`SliceSpec::Full` 返回完整 decoded bytes。`Scatter` 支持多个 `RangeCopy`。
`from_triples` 的输入是扁平三元组数组：

```text
[dst0, src_start0, src_end0, dst1, src_start1, src_end1, ...]
```

验证规则在 materialization 前执行：

- 三元组长度必须是 3 的倍数。
- `src_start <= src_end`。
- `src_end <= decoded_len`。
- `dst_offset + (src_end - src_start)` 不能溢出。

输出长度是所有目标区间 end 的最大值。目标区间之间可以有空洞，空洞会保留为
`0`；目标区间也可以重叠，后面的 range 会覆盖前面的 bytes。

`#[doc(hidden)] Invalid(String)` 是为了 `AccessItem::with_slice` 和
`AccessRequest::with_slice` 能延迟错误：调用构造器本身不失败，真正执行时从 reply
返回 `invalid slice spec`。

内部 `SlicePlan` 会把 slice 分成三种 shape：

- `Full`：无需 scatter，可能直接复用 owned `Vec<u8>`。
- `Sequential`：目标从 0 开始顺序填满，使用未初始化 `Vec` 加
  `copy_nonoverlapping` 后 `set_len`。
- `Sparse`：先分配 `vec![0; output_len]`，再逐段 copy。

### Error 接口

```rust
pub type AccessResult<T> = Result<T, AccessError>;

pub enum AccessError {
    Io(io::Error),
    Codec(CodecError),
    Shutdown,
    OutOfMemory,
    QueueFull { capacity: usize },
    InvalidSlice(String),
    CpuWorkerPanic,
}
```

`AccessHandle::send`、`try_send`、`send_prefetch` 这类提交接口只同步返回提交失败
相关错误，例如 shutdown 或 queue full。实际 IO、decode、slice、OOM 错误会通过
请求的 oneshot reply 返回，当前统一转换为 `io::Error::other(err.to_string())`。

### CPU worker 配置

```rust
pub struct AccessCpuConfig {
    pub num_workers: usize,
    pub queue_capacity: usize,
    pub cpus: Option<Vec<usize>>,
}
```

默认值：

```text
num_workers = 4
queue_capacity = 1024
cpus = None
```

`validate()` 要求 worker 数和队列容量大于 0；如果指定 `cpus`，列表不能空且不能有
重复 CPU id。内部 `AccessCpuPool` 会创建 `access-cpu-{idx}` 线程；如果指定 CPU
affinity，会校验 CPU id 可用并把 worker 绑定到对应 core。

小输出会 inline materialize，避免线程切换：

```text
output_len <= 16 KiB
range_count <= 4
```

更大的 copy/scatter-copy 会发送到 CPU worker。worker panic 会被捕获并返回
`AccessError::CpuWorkerPanic`。

### Profiling 接口

#### `AccessProfile`

```rust
pub struct AccessProfile;
impl AccessProfile {
    pub fn disabled() -> Self;
    pub fn enabled(label: impl Into<String>) -> Self;
    pub fn from_env() -> Self;
    pub fn from_runtime(runtime: ProfileRuntime) -> Self;
    pub fn runtime(&self) -> &ProfileRuntime;
    pub fn into_runtime(self) -> ProfileRuntime;
    pub fn start(&self) -> ProfileRound<'_>;
    pub fn snapshot(&self) -> ProfileSnapshot;
    pub fn snapshot_and_reset(&self) -> ProfileSnapshot;
    pub fn reset_metrics(&self);
}
```

`AccessConfig::default()` 使用 `AccessProfile::from_env()`，因此在环境变量启用全局
profile 时 access 会自动开始一个 owned round。显式 `start()` 会清掉这个 auto
round。

#### `access_profile_registry()`

返回 access component 的 `ProfileRegistry`。注册的 scope：

| Scope | 说明 |
| --- | --- |
| `access.command` | 提交到 scheduler 的命令、排队时间、拒绝原因。 |
| `access.cache` | raw/decoded cache hit、miss、insert、replacement、eviction、fallback。 |
| `access.inflight` | raw read in-flight first/wait。 |
| `access.decode` | decode first/wait/cache/identity/call/error 和 bytes。 |
| `access.io` | raw IO read 次数、耗时、请求 bytes、实际 bytes、错误。 |
| `access.materialize` | decoded output materialization 次数、耗时、输出 bytes、错误。 |
| `access.reserve` | 内存预算 reserve 次数、等待时间、bytes、失败。 |
| `access.scheduled` | scheduled staging bytes、eviction、cancel。 |

### Scheduler 配置

#### `AccessConfig`

```rust
pub struct AccessConfig {
    pub queue_capacity: usize,
    pub scheduler_shards: usize,
    pub cache_capacity_bytes: usize,
    pub memory_budget_bytes: usize,
    pub default_io_priority: u8,
    pub keep_decoded: bool,
    pub cpu: AccessCpuConfig,
    pub profile: AccessProfile,
}
```

默认值：

```text
queue_capacity = 1024
scheduler_shards = 1
cache_capacity_bytes = 256 MiB
memory_budget_bytes = 512 MiB
default_io_priority = 0
keep_decoded = false
cpu = AccessCpuConfig::default()
profile = AccessProfile::from_env()
```

`validate()` 规则：

- `queue_capacity > 0`
- `scheduler_shards > 0`
- `cache_capacity_bytes > 0`
- `memory_budget_bytes > 0`
- `memory_budget_bytes >= cache_capacity_bytes`
- `scheduler_shards <= cache_capacity_bytes`
- `scheduler_shards <= memory_budget_bytes`
- `cpu.validate()` 通过

多 shard 时，cache 和 memory budget 会按 shard 拆分：

```text
base = total / shard_count
remainder 前几个 shard 各多 1 byte
```

每个 shard 独占一个 current-thread Tokio runtime 和一份非线程安全的状态
`Rc<RefCell<SchedulerState>>`，所以 shard 内状态转移不需要 mutex。

#### `ScheduledAccessConfig`

```rust
pub struct ScheduledAccessConfig {
    pub prefetch_step: usize,
    pub decode_ahead_steps: usize,
    pub ready_ahead_steps: usize,
}
```

默认值：

```text
prefetch_step = 2
decode_ahead_steps = 1
ready_ahead_steps = 0
```

这是 `AccessHandle::scheduled` 的 chunk 级 look-ahead 设置：

- `prefetch_step`：提前把 raw compressed bytes 读入 raw cache。
- `decode_ahead_steps`：提前完成 decode，结果可能是 decoded cache hit 或 scheduled
  staging。
- `ready_ahead_steps`：提前 materialize 成最终 `Vec<u8>`，包括 slice scatter-copy。

`max(prefetch_step, decode_ahead_steps, ready_ahead_steps)` 决定 client iterator 会从
source 中预先拉取多少 `AccessItem` 放进本地 buffer。

### 请求类型

#### `AccessItem`

```rust
pub struct AccessItem {
    pub key: ChunkKey,
    pub codec: SharedCodec,
    pub expected_size: Option<usize>,
    pub slice: SliceSpec,
}
impl AccessItem {
    pub fn new(key: ChunkKey, codec: SharedCodec, expected_size: Option<usize>) -> Self;
    pub fn with_slice(self, slice: Option<Vec<usize>>) -> Self;
    pub fn with_slice_spec(self, slice: SliceSpec) -> Self;
}
```

`AccessItem` 是不带 reply channel 的 chunk 访问描述，主要供 scheduled iterator 使用。
`slice` 默认是 `Full`。`with_slice(Some(vec![...]))` 使用延迟错误策略；要同步校验
则直接用 `SliceSpec::from_triples`。

#### `AccessRequest`

```rust
pub struct AccessRequest {
    pub key: ChunkKey,
    pub codec: SharedCodec,
    pub expected_size: Option<usize>,
    pub slice: SliceSpec,
    pub reply: oneshot::Sender<io::Result<Vec<u8>>>,
}
impl AccessRequest {
    pub fn new(
        key: ChunkKey,
        codec: SharedCodec,
        expected_size: Option<usize>,
        reply: oneshot::Sender<io::Result<Vec<u8>>>,
    ) -> Self;
    pub fn with_slice(self, slice: Option<Vec<usize>>) -> Self;
    pub fn with_slice_spec(self, slice: SliceSpec) -> Self;
}
```

普通一次性访问请求。提交后，结果通过 `reply` 返回。`Vec<u8>` 一定是 materialized
后的输出 bytes，不会把 cache 内部 `Arc<[u8]>` 暴露给 caller。

#### `PrefetchRequest`

```rust
pub struct PrefetchRequest {
    pub key: ChunkKey,
    pub reply: Option<oneshot::Sender<io::Result<()>>>,
}
impl PrefetchRequest {
    pub fn new(key: ChunkKey) -> Self;
    pub fn with_reply(self, reply: oneshot::Sender<io::Result<()>>) -> Self;
}
```

只预取 raw compressed bytes，不解码、不 materialize。目标是 warm raw cache。prefetch
不允许 uncached fallback；如果 chunk 比 raw cache capacity 还大，会返回 OOM。

### Handle 和启动接口

#### `AccessScheduler`

```rust
pub struct AccessScheduler;
impl AccessScheduler {
    pub fn spawn(
        config: AccessConfig,
        io: Box<dyn IoBackend>,
        decode: Box<dyn DecodeBackend>,
    ) -> io::Result<AccessHandle>;
}
```

启动调度器。每个 shard 创建一个独立 OS thread，线程名为 `scdata-access` 或
`scdata-access-{shard_idx}`，内部运行 current-thread Tokio runtime 和 `LocalSet`。
所有 shard 共享同一份 `IoBackend`、`DecodeBackend` 和 `AccessCpuPool` 的 `Arc`。

#### `AccessHandle`

```rust
pub struct AccessHandle;
impl AccessHandle {
    pub fn profiler(&self) -> AccessProfile;
    pub fn profile_snapshot(&self) -> ProfileSnapshot;
    pub fn profile_snapshot_and_reset(&self) -> ProfileSnapshot;
    pub fn reset_profile(&self);

    pub fn send(&self, request: AccessRequest) -> Result<(), AccessError>;
    pub async fn send_async(&self, request: AccessRequest) -> Result<(), AccessError>;
    pub fn try_send(&self, request: AccessRequest) -> Result<(), AccessError>;

    pub fn prefetch(
        &self,
        key: ChunkKey,
    ) -> Result<oneshot::Receiver<io::Result<()>>, AccessError>;
    pub fn send_prefetch(&self, request: PrefetchRequest) -> Result<(), AccessError>;

    pub fn scheduled<I>(
        &self,
        items: I,
        config: ScheduledAccessConfig,
    ) -> Result<ScheduledAccess<I::IntoIter>, AccessError>
    where
        I: IntoIterator<Item = AccessItem>;
}
```

`AccessHandle` 可 clone。提交命令时先按 `ChunkKey` 选 shard，然后把命令放进该 shard
的 bounded `flume` 队列。

- `send`：同步阻塞等待队列容量。
- `send_async`：异步等待队列容量。
- `try_send`：不等待；队列满返回 `AccessError::QueueFull { capacity }`。
- `prefetch`：便捷方法，创建 reply channel 并提交 raw prefetch。
- `send_prefetch`：提交 caller 自己构造的 `PrefetchRequest`。
- `scheduled`：创建阻塞 iterator，适合同步消费代码。

`send` 和 `send_async` 的“阻塞/等待”只发生在命令队列容量上，不等待 IO 或 decode
完成。

### Scheduled 访问和取消

#### `ScheduledAccess<I>`

```rust
pub struct ScheduledAccess<I>
where
    I: Iterator<Item = AccessItem>;

impl<I> Iterator for ScheduledAccess<I> {
    type Item = io::Result<Vec<u8>>;
}
impl<I> ScheduledAccess<I> {
    pub fn set_cancel_handle(&mut self, cancel: Arc<PrefetchCancel>);
}
```

`ScheduledAccess` 是 blocking iterator。`next()` 内部会 `blocking_recv()` 等待当前
scheduled item 的结果，因此不要在必须驱动其他 futures 的 Tokio runtime worker 上
直接调用。

创建时会先填充 look-ahead buffer 并发送初始 prefetch/decode/ready 命令。之后每次
`next()` 消费一个 item，再按当前位置补齐窗口。

#### `PrefetchCancel`

```rust
pub struct PrefetchCancel;
impl PrefetchCancel {
    pub fn new(handle: AccessHandle) -> Arc<Self>;
    pub fn is_cancelled(&self) -> bool;
    pub fn cancel_in_flight(&self) -> Option<ChunkKey>;
}
```

这是 scheduled consumer drop 场景的取消桥。`ScheduledAccess::next()` 在进入
`blocking_recv()` 前会记录当前 `(id, key)`。如果外部 consumer 被 drop，
`cancel_in_flight()` 会：

1. 设置 cancelled flag。
2. 读取当前 in-flight entry。
3. 向 scheduler 发送 `ScheduledCancel`。
4. 返回被取消的 `ChunkKey`。

这样能唤醒 scheduler 侧等待 `Pending` 的 `scheduled_take`，避免 producer join 卡住。

## 普通访问执行路径

一次 `AccessRequest` 的主要路径：

```text
AccessHandle::send
  -> shard queue
  -> run_scheduler / spawn_command
  -> handle_command(Read)
  -> access_item
  -> decoded_item_data
  -> load_chunk
  -> decode if needed
  -> materialize_decoded
  -> oneshot reply
```

### `load_chunk`

`load_chunk` 是 raw bytes 获取和缓存复用的核心：

1. 根据 `CacheUse` 先 pin cache。
   - `RawOnly`：只查 raw cache。
   - `RawOrDecoded(decode_key)`：先查 decoded cache，再查 raw cache。
2. cache hit 时返回 `LoadedChunk::cached`。`PinnedChunk` 里的 `PinGuard` 会阻止该
   cache entry 被驱逐。
3. cache miss 时在 `InflightTable` 注册。
   - `First`：当前任务拥有实际 IO read。
   - `Waiter`：等待已有 read 完成，然后回到循环重新查 cache。
4. first reader 先 `reserve_bytes(key.len)`，再调用 `IoBackend::submit_read`。
5. IO 成功后校验 `data.len() == key.len`。
6. 尝试 `cache.insert_if_absent(raw)`。
   - 成功则 pin raw entry 返回。
   - 如果已有 entry，释放当前 read 预留，pin 已有 raw entry 返回。
   - 如果 insert 失败且 `allow_uncached = true`，返回带 `MemoryReservation` 的
     uncached raw bytes。
   - 如果 insert 失败且 `allow_uncached = false`，返回 OOM。
7. 无论成功或失败，最终都会 `inflight.complete(key)` 唤醒 waiters。

普通访问和 scheduled decode 传 `allow_uncached = true`，因此 oversized 或 all-pinned
情况下仍可用临时 reservation 完成当前 caller；prefetch 传 `false`，因为 prefetch 的
目标就是放入 raw cache。

### Decode

`decoded_item_data` 处理 raw -> decoded：

1. 如果 `load_chunk` 返回 decoded cache hit，直接复用并记录 `decode.cached`。
2. 如果 raw hit 且 `codec.is_identity()`：
   - 不提交 decode backend。
   - 只校验 `expected_size` 是否等于 raw length。
   - raw bytes 作为 decoded bytes 使用。
3. 如果 `keep_decoded = false`：
   - 每次访问都调用 `DecodeBackend::submit_decode`。
   - 返回 owned `Vec<u8>`，不进入 decoded cache。
4. 如果 `keep_decoded = true`：
   - 使用 `decode_inflight: HashMap<DecodeKey, Arc<Notify>>` 去重并发 decode。
   - first decoder 完成后调用 `try_cache_decoded`。
   - 成功则返回 pinned decoded cache entry。
   - 失败则记录 uncached fallback，返回临时 `Arc<[u8]>`。
   - waiters 被唤醒后重新查 decoded cache。

decoded cache key 是 `DecodeKey`，所以等价 codec 实例只要 `codec.cache_key()` 相同
就能复用；不同 codec 或不同 expected size 不会复用。

### Materialization

`materialize_decoded` 把 decoded bytes 变成最终 `Vec<u8>`：

- `SliceSpec::Full + Owned(Vec<u8>)`：直接返回原 Vec。
- `SliceSpec::Full + Shared(Arc<[u8]>)`：copy 成 caller-owned Vec。
- `SliceSpec::Scatter`：按 `SlicePlan` copy/scatter。

小任务 inline，大任务交给 `AccessCpuPool`。交给 CPU worker 前不会把 cache pin 提前
drop；对 shared cache data 的 pin 会一直持有到 materialization 完成，避免中途被驱逐。

## Prefetch 执行路径

`PrefetchRequest` 只做 raw cache warming：

```text
AccessHandle::send_prefetch / prefetch
  -> try_prefetch_inline
  -> cache raw hit: reply Ok
  -> oversized relative to raw cache: reply OOM
  -> otherwise async prefetch_key
  -> load_chunk(..., allow_uncached=false, CacheUse::RawOnly)
```

如果 raw cache 已经有该 key，prefetch 不会提交 IO。prefetch 不检查 decoded cache，
也不会触发 decode。

## Scheduled access 执行路径

`ScheduledAccess` 针对可预测的 `AccessItem` 序列，把工作拆成三个 ahead 窗口：

```text
ready_ahead_steps    -> ScheduledEnsureReady
decode_ahead_steps   -> ScheduledDecode
prefetch_step        -> Prefetch
```

client 侧每个 item 有一个全局递增 `id` 和三个 sent flags：

```text
prefetch_sent
decode_sent
ready_sent
```

发送 ready 会同时标记 decode 和 prefetch 已发送；发送 decode 会同时标记 prefetch 已
发送，因为 ready/decode 路径本身会按需 load raw bytes。

### 服务端状态

`ScheduledStore` 用 `id -> ScheduledStage` 管理状态：

| Stage | 含义 |
| --- | --- |
| `Pending(Notify)` | 某个 async scheduled task 正在生产这个 id 的结果。 |
| `Complete` | decode 已经可由 cache/direct path 满足，没有额外 staged bytes。 |
| `Decoded { data, bytes }` | 已解码但未 materialize，bytes 已计入 memory budget。 |
| `Ready { data, bytes }` | 已 materialize 为最终 output Vec，bytes 已计入 memory budget。 |
| `Failed(String)` | 该 id 的预处理失败。 |
| `Cancelled` | 被取消，用于唤醒等待者并阻止结果落库。 |

`Decoded` 和 `Ready` 是可驱逐 staged buffer。`ScheduledStore` 维护一个
`evictable` FIFO 队列和 `evictable_ids` 去重集合；当队列积累过多 stale id 时会压缩。

### `ScheduledDecode`

`ScheduledDecode` 尝试提前完成 decode：

1. inline 快路径：
   - id 已存在：忽略重复命令。
   - decoded cache hit：写入 `Complete`。
   - raw cache hit + identity codec：校验 size 后写入 `Complete` 或 `Failed`。
2. 需要异步时写入 `Pending(notify)`，启动 `handle_scheduled_decode`。
3. `decode_for_schedule` 复用普通 `load_chunk` 和 decode 逻辑。
4. 结果落库：
   - decoded/raw identity 已在 cache 中可复用：`Complete`。
   - 需要暂存 decoded bytes：`Decoded { data, bytes }`。
   - 错误：`Failed(message)`。

当 `keep_decoded = false` 时，scheduled decode 会 reserve decoded output 大小并把
owned decoded `Vec<u8>` 放入 `Decoded` stage。当 `keep_decoded = true` 但 decoded
cache insert 失败时，也可能把 shared decoded bytes 放入 staging。

### `ScheduledEnsureReady`

`ScheduledEnsureReady` 尝试提前 materialize 最终输出：

1. 如果已有 `Ready` 或 `Failed`，直接完成。
2. 如果是 `Pending`，等待 pending notify。
3. 如果是 `Decoded`：
   - full slice + owned data 可以 inline move 到 `Ready`，不额外分配。
   - 其他情况调用 `make_ready_from_staged_data` materialize。
4. 如果是 `Complete` 或缺失，则走 `make_ready_direct`，按普通 access path 读取/解码
   并直接 materialize 成 ready buffer。

`make_ready_from_staged_data` 会处理 peak memory：很多场景需要同时持有 staged input
和 output buffer，因此要求 `staged_bytes + output_bytes <= memory_budget.capacity()`。
失败时释放 staged bytes 并返回 OOM。

### `ScheduledTake`

`ScheduledAccess::next()` 最终发送 `ScheduledTake` 并阻塞等 reply。服务端逻辑：

1. `Pending`：等待 notify，然后重试。
2. `Ready`：移除 stage，释放 ready bytes，返回 Vec。
3. `Decoded`：移除 stage，materialize，释放 output bytes，返回 Vec。
4. `Complete` 或缺失：走普通 `access_item` direct path。
5. `Failed` 或 `Cancelled`：返回 error。

`ScheduledAccess::drop` 会对 buffer 中尚未消费的 id 发送 `ScheduledCancel`。

### 取消

`cancel_scheduled` 会：

- 移除并 abort 该 id 的 `scheduled_tasks`。
- 如果 stage 是 `Pending`，写入 `Cancelled` 并 notify waiters。
- 如果 stage 是 `Decoded` 或 `Ready`，释放对应 bytes。
- 记录 scheduled cancelled metric。

对 `Pending` 写入 `Cancelled` 而不是直接删除，是为了让已经在等 notify 的
`scheduled_take` 能醒来并读到 cancelled 状态。

## 缓存实现细节

`ChunkCache` 是 shard-local 的 `Rc<RefCell<CacheInner>>`，只在 current-thread runtime
内使用，不做跨线程锁。

内部 key 分两类：

```rust
Raw(ChunkKey)
Decoded(DecodeKey)
```

同一个 chunk 的 raw 和 decoded payload 可以同时存在，分别占用 capacity。decoded
payload 存储 `data: Arc<[u8]>`、`decode_key` 和 `_codec: SharedCodec`。

LRU 实现：

- 每个 entry 有 `stamp` 和 `refcount`。
- hit 或 insert 会递增 clock，并把 `(key, stamp)` push 到 `lru` 队尾。
- `lru` 允许 stale records，evict 时跳过 stamp 不匹配的旧记录。
- `lru.len() > max(entries.len() * 4, 64)` 时压缩。
- `refcount > 0` 的 entry 被 pin，不能驱逐。
- `PinGuard::drop` 递减 refcount；refcount 变 0 时 notify waiters。

插入策略：

- `insert_if_absent`：已有同 key representation 时只 touch，不替换。raw read 完成时使用。
- `insert_or_replace`：替换同 decoded key 的未 pinned entry。decoded cache 写入时使用。
- item 大于 cache capacity 返回 `ItemTooLarge`。
- 需要驱逐但所有候选都 pinned 返回 `AllPinned`，同时报告已经驱逐的 bytes。

## 内存预算实现细节

`MemBudget` 维护：

```text
capacity
used
release_notify
```

预算覆盖：

- raw/decoded cache resident bytes。
- 正在读入但还未进入 cache 的 raw bytes。
- uncached fallback raw/decoded bytes 的临时 reservation。
- scheduled `Decoded` staging bytes。
- scheduled `Ready` buffer bytes。

`reserve_bytes` 的顺序：

1. 如果 bytes 大于 budget capacity，立刻 OOM。
2. 尝试直接 reserve。
3. 不够时驱逐 cache 中 LRU 且 unpinned 的 entry，并释放 evicted bytes。
4. 仍不够时驱逐 scheduled store 中未来可驱逐的 `Decoded`/`Ready` buffer。
5. 仍不够时等待 `release_notify`，被 release 或 unpin 唤醒后重试。

`try_reserve_bytes` 是非阻塞版本，主要用于 scheduled ready 的 inline/direct 路径。它也
会尝试驱逐 cache 和 staged buffer，但不会等待。

`MemoryReservation` 是 RAII guard。uncached raw 或 decoded shared fallback 返回给后续
步骤时，如果最终没有进入 staging，guard drop 会释放预算；如果要把 bytes 交给
scheduled staging，调用 `commit()` 后由 scheduled stage 负责释放。

## In-flight 去重

`InflightTable` 只去重 raw read：

- 第一个 miss 插入 `ChunkKey -> Notify`，返回 `RegisterResult::First`。
- 后续同 key miss 返回 `Waiter(OwnedNotified)`。
- first read 完成或失败后 remove entry 并 `notify_waiters()`。
- waiter 醒来后不直接使用 first reader 的结果，而是回到 `load_chunk` 循环重新查 cache。

decode 去重不在 `InflightTable`，而是在 `SchedulerState::decode_inflight`，key 是
`DecodeKey`，只在 `keep_decoded = true` 路径启用。

## 线程和并发模型

`AccessScheduler::spawn` 的结构：

```text
AccessHandle
  shards: Vec<flume::Sender<AccessCommandEnvelope>>

for each shard:
  OS thread
    current-thread Tokio runtime
    LocalSet
    SchedulerState { cache, inflight, decode_inflight, budget, scheduled, scheduled_tasks }
```

同一个 `ChunkKey` 永远路由到同一个 shard，因此 raw in-flight 和 cache 不需要跨 shard
协调。不同 key 可以分布到不同 shard 并行处理。

`SchedulerState` 使用 `Rc<RefCell<...>>`，所以调度器内部任务是 `spawn_local`，不会跨
线程移动。backend future 类型仍要求 `Send`，因为 `IoTask`/`DecodeTask` 的 public type
如此定义，也方便复用外部 pool。

CPU materialization worker 是单独的 thread pool，所有 scheduler shard 共享同一份
`AccessCpuPool`。

## Profiling 记录点

重要记录点：

- 命令进入 scheduler 时记录 command kind 和 queue wait。
- `try_send` 队列满或 channel 断开记录 rejection。
- cache hit/miss、insert、replacement、eviction、insert failure 都记录。
- raw in-flight first/wait 和 decode first/wait 分开记录。
- IO read 记录 requested bytes、actual bytes、错误。
- decode 记录 encoded bytes、decoded bytes、错误；identity bypass 单独记录。
- materialize 记录 output bytes 和错误。
- reserve 记录请求 bytes、等待耗时、失败。
- scheduled 记录 staged bytes、staged eviction 和 cancel。

这些指标可以通过 `AccessHandle::profile_snapshot()` 或 `AccessProfile` 直接读取。

## 使用示例

普通访问：

```rust
let handle = AccessScheduler::spawn(config, Box::new(io), Box::new(decode))?;
let key = ChunkKey::new(FileRef::new(0), offset, len);
let (tx, rx) = tokio::sync::oneshot::channel();
let request = AccessRequest::new(key, codec, Some(expected_size), tx);

handle.send(request)?;
let bytes = rx.await??;
```

带 slice 的访问：

```rust
let slice = SliceSpec::from_triples(vec![
    0, 0, 128,
    128, 512, 640,
])?;
let request = AccessRequest::new(key, codec, None, tx).with_slice_spec(slice);
handle.send(request)?;
```

raw prefetch：

```rust
let rx = handle.prefetch(key)?;
rx.await??;
```

scheduled iterator：

```rust
let items = keys.into_iter().map(|key| AccessItem::new(key, codec.clone(), None));
let mut iter = handle.scheduled(items, ScheduledAccessConfig::default())?;

for result in &mut iter {
    let bytes = result?;
    // consume bytes
}
```

## 注意事项

- `ScheduledAccess::next()` 是阻塞调用，不应直接放在需要继续 poll futures 的 Tokio
  runtime worker 上。
- `AccessRequest::with_slice` 和 `AccessItem::with_slice` 会延迟 slice 格式错误；如果
  调用方需要构造时失败，用 `SliceSpec::from_triples`。
- `prefetch` 只保证 raw cache warming，不会 warm decoded cache。
- `keep_decoded = true` 会提升重复访问性能，但 decoded cache 与 raw cache 共用
  `cache_capacity_bytes` 和 `memory_budget_bytes`，可能增加驱逐压力。
- `memory_budget_bytes` 必须大于等于 `cache_capacity_bytes`，否则没有 in-flight 和
  staging headroom。
- oversized chunk 在普通访问里可能通过 uncached fallback 完成，但 prefetch 会失败。
- queue full 只可能从 `try_send` 返回；`send` 和 `send_async` 会等待队列容量。
- scheduler shard 内部状态不是 `Send`，不要把 `SchedulerState` 相关内部类型拿到 shard
  线程之外使用。
