# iopool 模块说明

> 更新时间：2026-07-01
>
> 范围：`rust/scdata/src/iopool/`。本文面向维护者，说明这个模块负责什么、公开哪些 Rust 接口，以及队列、去重、屏障、backend、profile 等内部实现细节。

---

## 1. 模块定位

`iopool` 是 `scdata` Rust 核心里的底层定位 IO 执行池。它把上层提交的文件定位操作转成后台 IO work，按 `FileId + offset + len` 做随机读，也支持写入、fsync/sync_data、truncate、metadata。

在主数据路径里，它通常被 `databank/adapter.rs` 包装成 `IoBackend`，供 `access` scheduler 读取 zarr chunk 的 encoded bytes。Python 用户通常不会直接拿到 `IoPool`；Python 层主要暴露 IO 配置，真正的 pool 由 `DataBank` 内部创建。

这个模块解决的核心问题：

- 统一 threaded 和 `io_uring` 两种 backend，给上层相同的提交/完成语义。
- 管理文件句柄生命周期，用不透明 `FileId` 代替路径或 `File`。
- 限制全局在途 IO 数量和排队容量，给上层提供 backpressure。
- 按文件分片，减少队列锁竞争，同时保持同一文件内的 dedup 和 ordering 状态一致。
- 对完全相同的 read 请求做 in-flight dedup，避免同一 chunk 被并发重复读取。
- 支持优先级、取消、profile 统计和 backend 失败传播。

## 2. 文件结构

| 文件 | 作用 |
|---|---|
| `mod.rs` | 公开 API、配置类型、`IoCommand`/`IoOutput`/`IoFuture`/`IoPool`、`QueueCore` 调度核心、同步执行函数、单元测试 |
| `threaded.rs` | 可移植 threaded backend，用 worker 线程阻塞执行 `pread`/`pwrite` 等定位 IO |
| `uring.rs` | `feature = "uring"` 下启用的 `io_uring` backend，负责 SQE/CQE、eventfd 唤醒、fixed-file registration |
| `profile.rs` | crate 内部 profile registry 和 metrics 记录函数 |

`lib.rs` 里有 `pub mod iopool;`，因此 `mod.rs` 中的 `pub` 项会暴露为 crate 外部接口。`threaded`、`uring`、`profile` 子模块本身没有对 crate 外公开。

## 3. 对外接口总览

下面列的是 `rust/scdata/src/iopool/mod.rs` 里真正 `pub` 的 Rust 接口。

### 3.1 基础类型

```rust
pub type FileId = usize;
```

`FileId` 是 `IoPool::register_*` 返回的不透明文件句柄。调用方只应把它传回 `IoCommand`，不要依赖它的具体数值。内部会复用已经注销的 closed slot，所以旧 `FileId` 注销后不能继续使用。

```rust
pub enum BackendKind {
    Uring,
    Threaded,
}
```

表示当前 `IoConfig` 选择的 backend。

```rust
pub enum OpKind {
    Read,
}
```

目前只用于 read dedup key。只有 read 会被去重，write/sync/truncate/metadata 都不会共享。

```rust
pub struct RequestKey {
    pub file: FileId,
    pub offset: u64,
    pub len: usize,
    pub kind: OpKind,
}
```

read 去重键。匹配粒度是精确 `(file, offset, len, kind)`，不会合并相邻或重叠 byte range。

### 3.2 配置接口

#### `BaseIoConfig`

```rust
pub struct BaseIoConfig {
    pub max_in_flight: usize,
    pub queue_capacity: usize,
    pub priority_levels: usize,
    pub queue_shards: usize,
    pub assume_non_overlapping_reads: bool,
}
```

Rust 默认值：

| 字段 | 默认值 | 含义 |
|---|---:|---|
| `max_in_flight` | `1024` | backend 同时执行的最大唯一 IO 数。duplicate read 不额外占用 active slot |
| `queue_capacity` | `4096` | 已接纳但尚未完成/取消的最大唯一 IO 数。duplicate read 不额外占用 queue slot |
| `priority_levels` | `3` | 优先级层级数，`0` 最高 |
| `queue_shards` | `1` | 独立内部队列数量的请求值 |
| `assume_non_overlapping_reads` | `false` | 是否启用 read fast path，适合调用方能保证 read 与 write/truncate 不重叠的场景 |

公开方法：

```rust
impl BaseIoConfig {
    pub fn validate(&self) -> Result<(), String>;
}
```

`validate` 要求 `max_in_flight`、`queue_capacity`、`priority_levels`、`queue_shards` 都大于 0。

注意：Python 高层 dataclass 的默认值和 Rust `Default` 不完全一致。例如 `scdata.databank.BaseIoConfig` 当前默认 `max_in_flight=768`、`queue_shards=8`，最终会被转换成 Rust 配置。

#### `UringConfig`

```rust
pub struct UringConfig {
    pub base: BaseIoConfig,
    pub entries: u32,
    pub drivers: usize,
    pub iowq_bounded_workers: u32,
    pub iowq_unbounded_workers: u32,
    pub registered_files: u32,
}
```

Rust 默认值：

| 字段 | 默认值 | 含义 |
|---|---:|---|
| `base` | `BaseIoConfig::default()` | 通用队列配置 |
| `entries` | `256` | 每个 ring 的 submission/completion queue depth |
| `drivers` | `1` | driver 线程/ring 数 |
| `iowq_bounded_workers` | `0` | io-wq bounded worker 上限，`0` 表示内核默认 |
| `iowq_unbounded_workers` | `0` | io-wq unbounded worker 上限，`0` 表示内核默认 |
| `registered_files` | `4096` | sparse fixed-file slot 数，`0` 禁用 fixed files |

`validate` 是 `pub(super)`，外部通过 `IoConfig::validate()` 间接调用。它要求 `base.validate()` 通过、`entries >= 2`、`drivers > 0`。

如果编译时没有启用 `uring` feature，`IoPool::new(IoConfig::Uring(_))` 会返回 `Unsupported`。

#### `ThreadedConfig`

```rust
pub struct ThreadedConfig {
    pub base: BaseIoConfig,
    pub num_workers: usize,
    pub cpus: Option<Vec<usize>>,
}
```

Rust 默认值：

| 字段 | 默认值 | 含义 |
|---|---:|---|
| `base` | `BaseIoConfig::default()` | 通用队列配置 |
| `num_workers` | `8` | worker 线程数请求值 |
| `cpus` | `None` | 可选 CPU allow-list；存在时 worker round-robin pin 到这些 CPU |

公开方法：

```rust
impl ThreadedConfig {
    pub fn validate(&self) -> Result<(), String>;
}
```

`validate` 要求 `base.validate()` 通过、`num_workers > 0`、`cpus` 如果存在则非空且没有重复 CPU id。真实 worker 数是 `num_workers.max(1).min(max_in_flight).max(1)`。

#### `IoConfig`

```rust
pub enum IoConfig {
    Uring(UringConfig),
    Threaded(ThreadedConfig),
}
```

`Default` 是 `IoConfig::Threaded(ThreadedConfig::default())`。

公开方法：

```rust
impl IoConfig {
    pub fn kind(&self) -> BackendKind;
    pub fn base(&self) -> &BaseIoConfig;
    pub fn validate(&self) -> Result<(), String>;
}
```

内部会根据配置计算实际 shard 数：

```text
actual_queue_shards = requested.min(consumers).min(max_in_flight).max(1)
```

其中 `consumers` 对 threaded 是有效 worker 数，对 `io_uring` 是 `drivers`。这样可以避免创建没有消费者的空 shard。

### 3.3 IO 命令

```rust
pub enum IoCommand {
    Read { file: FileId, offset: u64, len: usize, priority: usize },
    Write { file: FileId, offset: u64, buf: Vec<u8>, priority: usize },
    Fsync { file: FileId, priority: usize },
    SyncData { file: FileId, priority: usize },
    Truncate { file: FileId, len: u64, priority: usize },
    Metadata { file: FileId, priority: usize },
}
```

公开构造和查询方法：

```rust
impl IoCommand {
    pub fn read(file: FileId, offset: u64, len: usize, priority: usize) -> Self;
    pub fn write(file: FileId, offset: u64, buf: Vec<u8>, priority: usize) -> Self;
    pub fn fsync(file: FileId, priority: usize) -> Self;
    pub fn sync_all(file: FileId, priority: usize) -> Self;
    pub fn sync_data(file: FileId, priority: usize) -> Self;
    pub fn truncate(file: FileId, len: u64, priority: usize) -> Self;
    pub fn metadata(file: FileId, priority: usize) -> Self;
    pub fn priority(&self) -> usize;
    pub fn dedup_key(&self) -> Option<RequestKey>;
    pub fn file(&self) -> FileId;
}
```

语义说明：

- `priority` 越小优先级越高；`priority >= priority_levels` 会在 submit 时返回 `InvalidInput`。
- `fsync` 和 `sync_all` 是同一个命令，输出是 `IoOutput::SyncAll`。
- `dedup_key()` 只有 `Read` 返回 `Some(RequestKey)`；其它命令返回 `None`。
- `Read { len: 0 }` 和空 `Write` 有零长度短路逻辑，但仍要尊重同文件上已经存在的 barrier/metadata 状态。

### 3.4 输出类型

```rust
pub struct IoMetadata {
    pub len: u64,
    pub is_file: bool,
    pub is_dir: bool,
    pub readonly: bool,
}
```

`Metadata` 命令的轻量输出。它由 `std::fs::Metadata` 或 `statx` 结果转换而来。

```rust
pub enum IoOutput {
    Read(Arc<[u8]>),
    Write { bytes: usize },
    SyncAll,
    SyncData,
    Truncate,
    Metadata(IoMetadata),
}
```

公开辅助方法：

```rust
impl IoOutput {
    pub fn read_bytes(&self) -> Option<&[u8]>;
    pub fn into_read_bytes(self) -> io::Result<Arc<[u8]>>;
    pub fn bytes_written(&self) -> Option<usize>;
    pub fn metadata(&self) -> Option<&IoMetadata>;
}
```

`into_read_bytes` 在输出不是 `Read` 时返回 `InvalidData`，常用于调用方明确知道自己提交的是 read 的路径。

### 3.5 `IoFuture`

```rust
pub struct IoFuture { /* private fields */ }
```

`IoPool::submit*` 返回的 completion future。内部是一个 `tokio::sync::oneshot::Receiver`，并保存 `Weak<QueueCore> + WorkId` 用于 drop 时取消未派发 work。

公开方法：

```rust
impl IoFuture {
    pub fn blocking_recv(self) -> io::Result<IoOutput>;
    pub fn blocking_recv_read(self) -> io::Result<Arc<[u8]>>;
    pub fn try_recv(&mut self) -> io::Result<Option<IoOutput>>;
    pub fn try_recv_read(&mut self) -> io::Result<Option<Arc<[u8]>>>;
}

impl Future for IoFuture {
    type Output = io::Result<IoOutput>;
}
```

生命周期和取消行为：

- `blocking_recv`/`blocking_recv_read` 会消费 future。
- `try_recv` 未完成时返回 `Ok(None)`，完成后 future 被标记为 consumed。
- `Future::poll` 完成后也会标记 consumed。
- 如果 future 在完成前被 drop，`QueueCore::release(id)` 会减少 refcount；若这是最后一个 waiter 且 work 仍处于 `Queued` 状态，则从队列移除，backend 不会看到它。
- 如果 work 已经 `Dispatched`，drop 只能丢弃 receiver；底层 IO 仍会完成，结果发送给已关闭 receiver 时会被忽略。

### 3.6 `IoPool`

```rust
pub struct IoPool { /* private fields */ }
```

公开构造、profile、文件注册、提交方法：

```rust
impl IoPool {
    pub fn new(config: IoConfig) -> io::Result<Self>;
    pub fn new_with_profile(config: IoConfig, profiler: ProfileRuntime) -> io::Result<Self>;

    pub fn profile(&self) -> &ProfileRuntime;
    pub fn profile_snapshot(&self) -> ProfileSnapshot;
    pub fn profile_snapshot_and_reset(&self) -> ProfileSnapshot;
    pub fn reset_profile(&self);

    pub fn register_file(&self, path: &Path) -> io::Result<FileId>;
    pub fn register_readwrite_file(&self, path: &Path) -> io::Result<FileId>;
    pub fn register_readonly_file(&self, path: &Path) -> io::Result<FileId>;
    pub fn register_file_with_options(
        &self,
        path: &Path,
        options: &OpenOptions,
    ) -> io::Result<FileId>;
    pub fn register_existing_file(&self, file: File) -> io::Result<FileId>;

    pub fn unregister_file(&self, file: FileId) -> io::Result<()>;
    pub fn try_unregister_file(&self, file: FileId) -> io::Result<()>;

    pub fn try_submit(&self, cmd: IoCommand) -> io::Result<IoFuture>;
    pub fn submit(&self, cmd: IoCommand) -> io::Result<IoFuture>;
    pub async fn submit_async(&self, cmd: IoCommand) -> io::Result<IoFuture>;
}
```

文件注册语义：

- `register_file` 保持历史行为：先尝试 read-write 打开，失败后 fallback 到 read-only。
- `register_readwrite_file` 要求读写打开成功。
- `register_readonly_file` 只读打开。
- `register_file_with_options` 直接使用调用方传入的 `OpenOptions`。
- `register_existing_file` 接管已有 `File` 的所有权并放入文件表。
- `unregister_file` 会阻塞等待该文件在所有 shard 上没有 active IO，然后释放 slot。
- `try_unregister_file` 不等待 active IO；如果文件仍活跃，返回 `WouldBlock` 并尽量恢复成 open 状态。

提交语义：

- `try_submit` 只尝试立即入队；队列满时返回 `WouldBlock`。
- `submit` 在队列满时阻塞当前线程等待 queue permit，然后再入队。
- `submit_async` 在队列满时异步等待 queue permit。
- 三者只是等待“提交进入队列”，不等待 IO 完成；IO 完成要继续等待返回的 `IoFuture`。

析构语义：

- `Drop for IoPool` 会对所有 queue 调 `shutdown()`，关闭 capacity limiter，唤醒 worker/driver，然后 join backend threads。
- 如果 backend thread 在 shutdown join 时 panic，会打印 `[iopool] backend thread panicked during shutdown`。

### 3.7 Python/PyO3 间接接口

`IoPool` 本体没有作为 Python class 暴露；Python 用户通过 `DataBank` 间接使用它。和本模块直接相关的 Python 可见接口是 IO 配置：

| Rust PyO3 class | Python typing/dataclass | 作用 |
|---|---|---|
| `scdata._scdata._BaseIoConfig` | `scdata.databank.BaseIoConfig` | 对应 `BaseIoConfig` |
| `scdata._scdata._UringConfig` | `scdata.databank.UringConfig` | 对应 `UringConfig` |
| `scdata._scdata._ThreadedConfig` | `scdata.databank.ThreadedConfig` | 对应 `ThreadedConfig` |
| `scdata._scdata._IoConfig` | `scdata.databank.IoConfig` | 对应 `IoConfig`，选择 `uring` 或 `threaded` |

PyO3 `_IoConfig` 公开：

```python
_IoConfig()
_IoConfig.uring(config: _UringConfig | None = None)
_IoConfig.threaded(config: _ThreadedConfig | None = None)
.kind -> Literal["uring", "threaded"]
.base -> _BaseIoConfig
.uring_config() -> _UringConfig | None
.threaded_config() -> _ThreadedConfig | None
.validate() -> None
```

高层 `scdata.databank.IoConfig` 还提供 dataclass 风格的 `IoConfig.uring(...)` / `IoConfig.threaded(...)` 便利构造，并支持 `base__max_in_flight` 这类嵌套 kwargs。

## 4. 内部数据流

一个普通 read 的路径如下：

```text
调用方
  -> IoPool::submit(IoCommand::read(file, offset, len, priority))
  -> select_queue: queue_for_file(file) = queues[file % queues.len()]
  -> 读 file_table，拿 Arc<File>
  -> QueueCore::try_submit_with_handle
       - 检查 failure/shutdown/priority
       - 尝试零长度短路
       - 尝试 read dedup
       - 获取 queue permit
       - 分配 WorkId / seq
       - 插入 table / dedup / active index / ready set
       - became_ready 则 notify_one + wake_backend
  -> backend 消费 work
       - threaded: QueueCore::pop -> execute_work
       - uring: pop_or_notification / try_pop_batch -> SQE -> CQE
  -> QueueCore::complete
       - remove inflight
       - refresh 被解锁的后继 ready work
       - release active_limiter
       - profile operation
       - send result to all waiters
  -> IoFuture 收到 IoOutput::Read(Arc<[u8]>)
```

同一 `FileId` 总是路由到同一个 shard。这样同一文件的 ordering、dedup、unregister 等状态不需要跨 shard 合并。全局 `max_in_flight` 和 `queue_capacity` 则由所有 shard 共享。

## 5. `QueueCore` 调度细节

`QueueCore` 是模块的核心状态机。它同时承担排队、优先级、去重、取消、文件活跃状态、backend 唤醒和失败传播。

### 5.1 关键内部结构

```rust
struct Inner {
    priority_levels: usize,
    queued: usize,
    fast_read_path: bool,
    fast_ready: Vec<VecDeque<WorkId>>,
    fast_active_reads: Vec<usize>,
    ready: BTreeSet<(usize, u64, WorkId)>,
    table: HashMap<WorkId, Inflight>,
    dedup: HashMap<RequestKey, WorkId>,
    active_by_file: Vec<Option<FileActive>>,
    next_id: u64,
    next_seq: u64,
    shutdown: bool,
    failure: Option<SharedIoError>,
}
```

```rust
struct Inflight {
    key: Option<RequestKey>,
    op: Option<IoOperation>,
    access: Access,
    waiters: Waiters,
    _queue_permit: QueuePermit,
    refcount: usize,
    state: InflightState, // Queued | Dispatched
    priority: usize,
    seq: u64,
    fast_read: bool,
    queued_at: ProfileTimer,
}
```

`table` 是所有在途 work 的事实源。`ready` 只保存可派发 work 的排序 key，`dedup` 只保存 read 请求到 work id 的索引。

### 5.2 两级容量控制

`QueueCore` 使用两个不同 limiter：

- `queue_limiter: tokio::sync::Semaphore`：限制已经被接纳但尚未完成/取消的唯一 work 数量，对应 `queue_capacity`。
- `active_limiter: ActiveLimiter`：限制 backend 已经派发但尚未完成的唯一 work 数量，对应 `max_in_flight`。

两者都是跨 shard 共享的 `Arc`，因此配置是全局限制，而不是每个 shard 各一份。

queue permit 存在 `Inflight` 里，直到 work 完成、失败或 queued 状态下被取消才释放；因此 dispatched 但未完成的 work 也会继续占用 `queue_capacity`。duplicate read 命中 dedup 时不会新建 work，因此不占新的 queue permit，也不占新的 active slot。

### 5.3 优先级

优先级数由 `priority_levels` 决定，`0` 最高。内部有两条 ready 路径：

- `ready: BTreeSet<(priority, seq, WorkId)>`：普通路径，天然按优先级和提交顺序排序。
- `fast_ready: Vec<VecDeque<WorkId>>`：read fast path，每个优先级一个队列。

`pop_ready_locked` 会同时看 fast 和 slow 的最小 key，取 `(priority, seq, id)` 更小的那个派发。

read dedup 命中时，如果新请求优先级更高，会调用 `promote_priority_locked` 提升原 work 的优先级。已经 dispatched 的 work 只能更新记录，不能重新排队。

### 5.4 文件内 ordering 和 barrier

内部把命令抽象成 `Access`：

| `IoCommand` | `Access` |
|---|---|
| `Read` | `Read { file }` |
| `Write` | `Write { file }` |
| `Fsync` / `SyncData` | `Sync { file }` |
| `Truncate` | `Truncate { file }` |
| `Metadata` | `Metadata { file }` |

普通路径会在 `active_by_file[file]` 里维护 `FileActive`：

```rust
struct FileActive {
    all: BTreeSet<(seq, WorkId)>,
    writes: BTreeSet<(seq, WorkId)>,
    barriers: BTreeSet<(seq, WorkId)>,
    truncates: BTreeSet<(seq, WorkId)>,
    metadata: BTreeSet<(seq, WorkId)>,
}
```

派发规则：

| Access | 阻挡条件 |
|---|---|
| `Read` | 有更早的 barrier |
| `Write` | 有更早的 barrier 或 metadata |
| `Sync` / `Truncate` | 同文件 fast read 计数为 0，且没有任何更早 active work |
| `Metadata` | 有更早的 barrier 或 write |

这里的 barrier 是 `Sync` 和 `Truncate`。普通 `Write` 不是全屏障，`Read` 也不会因为更早 `Write` 自动阻塞；这个池适合定位 chunk IO，调用方如果需要强顺序语义，应显式提交 sync/truncate 或在更高层保证不重叠。

work 完成或取消时，`refresh_ready_after_removed_locked` 会根据移除的 access 类型，只重新检查可能刚被解锁的后继 work，避免全队列扫描。

### 5.5 read fast path

`BaseIoConfig::assume_non_overlapping_reads = true` 时启用 read fast path。条件是：

```text
fast_read_path && Access::Read && 该文件当前没有 barrier
```

fast read 不进入 `active_by_file` 的 `all` set，只进入：

- `fast_ready[priority]`
- `fast_active_reads[file]`

这样大量已知不与写/截断重叠的随机 read 可以减少 active index 操作和 ordering 检查开销。`Sync`/`Truncate` 会检查 `fast_active_read_count(file) == 0`，因此仍能等待 fast read 排空。

风险点：开启后等于告诉队列“read 与 write/truncate 不重叠或由上层保证安全”。如果同一 byte range 上混合读写，不能依赖这个 fast path 维持读写顺序。

### 5.6 read dedup

`dedup: HashMap<RequestKey, WorkId>` 只记录 read work。命中条件：

- 请求是 `Read`，且 `(file, offset, len)` 精确相同。
- 目标 inflight 还存在，`refcount > 0`，access 相同。
- 没有同文件上 seq 更晚或相等的 `Truncate`。这是为了避免“truncate 之后提交的相同 read”错误复用 truncate 之前的 read。

命中后：

- `refcount += 1`
- 新 waiter 加入 `Waiters`
- 如果新请求 priority 更高则提升原 work
- 返回新的 `IoFuture`

完成时，`Waiters::send` 会把同一个 `IoOutput` 广播给所有 waiter。`IoOutput::Read` 内部是 `Arc<[u8]>`，clone 成本低。

### 5.7 取消

每个 `IoFuture` drop 时会调用 `QueueCore::release(id)`：

- `refcount` 饱和减 1。
- 如果 `refcount == 0` 且 work 仍是 `Queued`，从 `table`、`ready`、`active_by_file`、`dedup` 中移除。
- 如果 work 已经 `Dispatched`，不取消底层 IO，只让后续 send 失败并被忽略。

profile 会记录 `cancelled-before-dispatch`。

### 5.8 文件注销状态机

文件表 slot 有三种状态：

```rust
enum FileSlot {
    Open(Arc<File>),
    Closing { handle: Arc<File>, reopenable: bool },
    Closed,
}
```

提交时只接受 `Open` slot，并立即 clone 出 `Arc<File>` 放进 `IoOperation`。因此已经提交的 work 即使之后文件进入 `Closing`，也能继续使用原句柄完成。

`unregister_file`：

1. `Open -> Closing { reopenable: false }`。
2. 等所有 queue 上这个 file inactive。
3. `Closing -> Closed`。
4. `FileId` 放回 `free_file_ids`，后续注册可复用。

`try_unregister_file`：

1. `Open -> Closing { reopenable: true }`。
2. 如果发现 active IO，尝试恢复 `Closing -> Open`，返回 `WouldBlock`。
3. 如果没有 active IO，只有当 slot 仍是 reopenable closing 时才 finish。
4. 如果另一个阻塞 unregister 已经把 slot 标为 non-reopenable，则返回 `WouldBlock`，不抢占它。

`io_uring` backend 下注册/注销文件还会向对应 shard push fixed-file update。

## 6. backend 实现

### 6.1 threaded backend

`threaded.rs` 是可移植 fallback。启动流程：

```text
threaded::start(config, queues, file_table)
  -> resolve_cpu_affinity(config)
  -> worker_count = min(num_workers, max_in_flight).max(1)
  -> spawn iopool-wrk-{idx}
```

每个 worker 绑定到 `queues[worker_idx % queues.len()]`。worker loop：

```rust
while let Some(work) = queue.pop() {
    let id = work.id;
    let operation_started = work.operation_started;
    let result = execute_work(&file_table, work.op);
    queue.complete(id, operation_started, result);
}
```

实际 IO 由 `execute_work` 同步执行：

| 操作 | threaded 实现 |
|---|---|
| `Read` | `FileExt::read_exact_at` 读满指定 buffer，短读/EOF 返回错误 |
| `Write` | 自己的 `write_all_at` 循环处理短写 |
| `Fsync` | `File::sync_all` |
| `SyncData` | `File::sync_data` |
| `Truncate` | `File::set_len` |
| `Metadata` | `File::metadata` 后转成 `IoMetadata` |

read buffer 用 `MaybeUninit<u8>` 分配，避免先把整块 buffer 清零；读满后用 `assume_init_read_buffer` 转成 `Box<[u8]>`。

### 6.2 `io_uring` backend

`uring.rs` 只在 `feature = "uring"` 下编译。启动流程：

```text
uring::start(config, queues, file_table)
  -> 为每个 driver 创建 IoUring::new(entries)
  -> 可选 register_iowq_max_workers
  -> 创建 EventFd
  -> queue.register_wake_fd(eventfd)
  -> spawn iopool-uring-driver-{idx}
```

每个 driver 独占一个 ring 和 eventfd，并绑定到 `queues[driver_index % queues.len()]`。

driver loop 核心步骤：

1. `apply_fixed_file_updates`：消费注册/注销文件产生的 fixed-file update。
2. 如果没有 pending work，调用 `queue.pop_or_notification(cursor)` 阻塞等待 work、fixed-file update 或 shutdown。
3. 有 pending 容量时，`queue.try_pop_batch` 批量取 work。
4. `Pending::prepare` 把 `IoOperation` 转成 `Pending` 或立即完成结果。
5. `submit_pending` 分配 pending slot，构造 SQE 并 push 到 submission queue。
6. 注册 eventfd `PollAdd` wake SQE，`user_data = WAKE_TOKEN`。
7. `ring.submit_and_wait(1)` 提交批量 SQE 并等待至少一个 CQE。
8. `reap_completions` 扫 CQE，完成 work、重提短读/短写剩余部分，或处理 wake event。

read/write SQE 会设置 `squeue::Flags::ASYNC`，让可能阻塞的文件 IO 走 kernel io-wq，driver 线程主要负责 submission/completion 管理。

`Pending` 的特殊处理：

- read/write 单次 SQE 长度限制为 `u32::MAX`，超大 buffer 会分片继续提交。
- read 返回 0 且未读满时返回 `UnexpectedEof`。
- write 返回 0 且未写满时返回 `WriteZero`。
- `Metadata` 用 `statx(AT_EMPTY_PATH, STATX_BASIC_STATS)`；如果失败，fallback 到 `File::metadata()`。
- `Truncate` 如果 `Ftruncate` opcode 不支持，fallback 到 `File::set_len()`。
- fixed-file registration 失败或 update 失败时，backend 会禁用 fixed-file 路径并回退到普通 fd。

fixed-file registration：

- `registered_files == 0` 时禁用。
- 启用时调用 `register_files_sparse(capacity)`。
- slot 使用 `FileId` 作为 fixed-file index；超出容量或转换失败则回退普通 fd。
- 每个 driver 有自己的 `FixedFileRegistry` 和 update cursor。

eventfd 唤醒：

- `QueueCore::wake_backend` 对每个 registered wake fd 写入 `1`。
- `wake_pending` 防止重复写 eventfd。
- driver 收到 `WAKE_TOKEN` CQE 后 `drain_eventfd` 并 `acknowledge_wake()`。

driver 发生不可恢复错误时，`fail_driver` 会先把 pending work 逐个 `complete(Err(...))`，然后 `queue.fail(err)` 使队列进入 failure 状态，唤醒并拒绝后续 submit。

## 7. profile

`IoPool::new` 使用 `IoPoolProfile::from_env()`，按 crate 的 profile 环境配置决定是否记录。`IoPool::new_with_profile` 则接收调用方提供的 `ProfileRuntime`，并注册 iopool 的 profile registry。

公开 profile 入口在 `IoPool` 上：

- `profile() -> &ProfileRuntime`
- `profile_snapshot() -> ProfileSnapshot`
- `profile_snapshot_and_reset() -> ProfileSnapshot`
- `reset_profile()`

内部 registry：

| component/scope | 类型 | 含义 |
|---|---|---|
| `iopool/submit` | event | 记录提交次数和各命令类型提交数 |
| `iopool/queue` | event | 记录队列调度事件 |
| `iopool/operation` | timer | 记录实际 IO 操作耗时 |

主要 metrics：

| metric | 记录位置 |
|---|---|
| `submit.calls`、`submit.read/write/fsync/sync-data/truncate/metadata` | submit 入口 |
| `queue.immediate` | 零长度立即完成 |
| `queue.dedup-hits` | read dedup 命中 |
| `queue.queue-full` | submit 路径首次尝试获取 queue permit 时容量满 |
| `queue.cancelled-before-dispatch` | future drop 取消 queued work |
| `queue.dispatched`、`queue.dispatch-wait` | backend 派发 work |
| `operation.calls`、`operation.work` | IO 完成 |
| `operation.read`、`operation.write` | 完成结果里的读/写字节数 |
| `operation.errors` | IO 返回错误 |

## 8. 错误行为速查

| 场景 | 错误类型 |
|---|---|
| 配置校验失败 | `InvalidInput` |
| `Uring` backend 未编译 | `Unsupported` |
| `priority >= priority_levels` | `InvalidInput` |
| `FileId` 不存在、已 closed、closing 时新提交 | `InvalidInput` |
| `try_submit` 队列满 | `WouldBlock` |
| `submit`/`submit_async` 等待期间队列关闭 | `BrokenPipe` 或 queue failure 原始错误 |
| `try_unregister_file` 文件仍有 active IO | `WouldBlock` |
| read 未读满 | `UnexpectedEof` |
| write 返回 0 | `WriteZero` |
| offset 加法溢出 | `InvalidInput` |
| future 被重复消费 | `BrokenPipe` |
| oneshot sender 被 drop | `BrokenPipe` |

## 9. 维护注意点

- `queue_capacity` 和 `max_in_flight` 是全局共享，不是每个 shard 独立容量。
- 同一文件必须始终路由到同一 shard；否则 read dedup、barrier、unregister active 检查都会失效。
- read dedup 只对精确 range 生效，不处理范围合并。
- `Write` 不是全屏障，普通 `Read` 不会自动等更早 `Write`；需要强顺序的调用方要显式使用 sync/truncate 或在上层避免重叠。
- `assume_non_overlapping_reads` 只能在上层确认 read 与修改类操作不重叠时开启。
- `IoFuture::drop` 只能取消未派发 work，不能取消已经交给 backend 或内核的 IO。
- `register_existing_file` 会接管 `File` 并在内部用 `Arc<File>` 延长已提交 work 的生命周期。
- `io_uring` 的 fixed-file 是性能优化，不是语义依赖；失败会回退普通 fd。
- read buffer 使用 `MaybeUninit`，只有在 `read_exact_at` 或 CQE 累计读满后才能安全转成 initialized bytes。
- Python 层也有一套 dataclass 默认值和转换逻辑；改 Rust 默认值时要同步评估 `scdata/databank.py`、`_scdata.pyi` 和 PyO3 config 绑定。
