# scdata load 设计概览（组会版）

## 一句话概括

`scdata.load` 将任意顺序的单细胞行读取请求，预先编译成一个可复用的静态执行计划；运行时只做有界 I/O、解码和 Dense/CSR scatter，并通过共享内存环以零拷贝方式向单进程或多 rank 提供 NumPy batch。

## 我们要解决的问题

训练通常不是顺序读取一张完整矩阵，而是从一个或多个 SCC 数据集按指定顺序抽取细胞，并且还要处理：

- Dense 与 CSR 两种存储；
- 不同数据集之间的基因对齐、缺失列填充和 dtype 转换；
- 高并发 I/O、解压和预取；
- 有限内存下的缓存复用与背压；
- 单进程和多 rank 的一致数据通路。

我们的核心取舍是：**把复杂性放到编译期，把运行时压缩成机械执行。**

## 分层架构

```text
用户代码
  │  register / compile / prefetch / open_distributed
  ▼
Python: src/scdata/load
  策略、校验、名称/基因对齐、生命周期、序列化、NumPy 交互
  │  只传递规范化后的数组、字典和私有句柄
  ▼
PyO3: crates/scdata-python
  薄绑定：_Dataset / _Plan / _SharedServer / _SharedClient
  │
  ▼
Rust: crates/sc-load
  静态计划编译、缓存布局、I/O/解码 DAG、Dense/CSR scatter、shared ring
  │
  ▼
sc-compress store: .scc 目录或 .scc.zip
```

设计边界很明确：Python 决定“读什么、如何组织、如何呈现”，Rust 负责“高效且有界地执行”。`scdata._core` 是私有机械接口，不承担公开 API 设计。

## 三个核心设计

### 1. 静态编译：提前做完运行时原本要做的判断

训练开始前，行访问顺序、batch 划分、数据源、feature map 和输出格式都已经确定。我们利用这些先验信息，把它们编译成静态 Plan，而不是在每次读取时重新决策。

编译期会提前确定：

- 每个 row 对应哪些 data/indices block；
- 哪些 block 可以跨 row、跨 batch 复用，哪些需要重新加载；
- 物理 I/O 的 offset、长度、合并方式及执行优先级；
- 每个 decoded block 在 cache 中的位置和存活区间；
- Dense/CSR 使用哪种 scatter、feature map 和 dtype conversion 路径；
- 每个任务依赖谁，以及写入哪个 output-ring generation。

因此运行时不再反复解析元数据、查找 block、选择 kernel、分配 cache、判断淘汰对象或临时构造任务图。worker 的主要动作被压缩为：

```text
领取 ready task → I/O → decode → scatter → 发布完成事件
```

这样做有三个直接收益：减少热路径的分支和动态判断；将一次编译成本摊销到多个 epoch；在运行前就能得到内存上限、I/O 规模和依赖数量等可观测统计。

### 2. 静态 cache：提前确定 block 的空间和时间位置

这里的 cache 是固定容量的 **decoded block arena**，不是运行时维护的通用 LRU。编译器会模拟完整 batch 序列，并为每一次 block residency 静态确定：

- `offset + length`：写入 cache arena 的哪一段；
- `generation`：这段地址当前属于哪个 block 实例；
- 生命周期：何时加载、被哪些 scatter 消费、最后何时可覆盖；
- `available_after_batch`：复用该 extent 前必须完成到哪个 batch；
- cache hit、reload，以及容量或碎片导致的等待位置。

相同地址可以在不同时刻承载不同 generation，但新 generation 只有在旧 generation 的最后一个消费者完成后才能进入。这个先后关系在编译期就转化为依赖边，而不是运行时再抢锁判断。

```text
动态 cache：miss → 抢 cache 锁 → 查表 → 选 victim → 等引用释放 → 分配/覆盖

静态 cache：前驱完成 → 新 generation 变为 ready → 直接写入预定地址
```

因此运行时没有 cache allocator、LRU、residency hash lookup、动态 eviction 或 cache refcount。多个 worker 不会竞争“谁把哪个 block 放进 cache”，也不会领取一个 cache 尚未可用的任务后原地阻塞。仍然存在轻量的 ready queue、原子状态和 ring 同步，但昂贵且复杂的 **cache 管理竞争** 已被移到编译期消解。

### 3. 依赖 DAG：把缓存正确性和并行调度变成同一个问题

Plan 最终是一张有向无环图。它主要表达两类 cache 依赖：

1. **覆盖依赖**：新 block 要写入某个 cache extent，必须等旧 generation 的最后一次 scatter 完成；对应 `JobDone → PrefixDone → IoDecodeLoadTask`。
2. **数据依赖**：scatter 只有在需要的 data block，以及 CSR 的 indices block，都完成 I/O 和解码后才能执行；对应 `DecodeOp → BlockReady → Dense/CsrScatter`。

完整关系可以概括为：

```text
旧 generation 的最后一批 Scatter
                │
                ▼
             JobDone
                │
                ▼
            PrefixDone ───────┐   允许覆盖旧 cache extent
                              ▼
I/O(new generation) → DecodeOp → BlockReady(data) ──┐
                                                     ├→ Dense/CsrScatter → JobDone → BatchReady
I/O(indices)       → DecodeOp → BlockReady(indices) ─┘
```

运行时每个节点只有一个依赖计数器：计数归零才进入 ready queue；任务完成后只需递减后继节点。worker 永远只领取 ready 节点，所以不会占着线程、cache slot 或 staging buffer 等待未知前置条件。

这张 DAG 同时实现了：

- **正确性**：旧 cache 未用完就绝不会被覆盖，decoded 数据未就绪就绝不会 scatter；
- **并行性**：互不依赖的 I/O、decode 和 scatter 可以自然并行；
- **低阻塞**：等待表现为“节点尚未入队”，而不是 worker 持锁睡眠；
- **可预测性**：cache 复用、预取距离和背压关系在执行前已经固定。

从本质上说，我们不是在运行时“管理 cache”，而是在编译期生成一张描述 **数据何时可读、内存何时可写** 的 DAG，运行时只做拓扑执行。

## 从请求到 batch

### 1. 注册数据源

`register()` 打开 SCC 中的矩阵，读取 shape、dtype、Dense/CSR 类型及可选的 `obs`/`var` 名称。多个数据集可通过 `feature_map` 映射到统一输出列；未映射列使用显式 `fill`。

### 2. 编译静态 Plan

`compile()` 接收有序 `(source_id, row)` 请求、`batch_size`、`prefetch_step` 和 `OutputSpec`。Rust 编译器一次性完成：

```text
row resolution → block 去重 → cache residency 模拟 → I/O 合并 → DAG/arena 固化
```

内存中的 Plan 可跨 epoch 复用。落盘时保存的是不含文件指针和临时租约的可重定位模板；重新绑定会严格核对数据源 manifest，再编译 native plan，并拒绝已经变化的 stale plan。

### 3. 执行静态任务图

每次 `Plan.open()` 都创建独立 session、decoded cache、输出环和 worker，并把 Plan 中的相对 offset 降低为本 session 的实际指针。worker 只领取依赖已满足的任务，不会先占住一个未就绪 job 再等待。

```text
PrefixDone → I/O → Decode → BlockReady → Dense/CSR Scatter → JobDone → BatchReady
```

本地文件可使用 blocking 或 io_uring；`auto` 仅在所有数据源都支持 positioned I/O 且 worker ring 可创建时选择 io_uring，否则安全回退到 blocking。ZIP Deflate 或 key-backed store 使用 blocking。

### 4. 通过 shared ring 交付

当前 Python 普通 `Session` 也走 `world_size=1` 的 shared-ring 路径；分布式模式使用完全相同的生产者，只把 logical batch 按 `batch_id % world_size` 轮转给各 rank。

- Linux 上使用 sealed `memfd + mmap` 保存输出环；
- rank iterator 通过文件描述符传递，并在实际消费进程中延迟 attach；
- 默认 `copy=False` 返回只读 NumPy view，其 base 持有 batch generation lease；
- view 释放后 ACK，输出槽才能复用；`copy=True` 则返回紧凑、可写的 NumPy-owned 数组；
- rank 退出、owner 死亡、取消和 worker 错误都会唤醒等待方并终止整条链路。

这使单进程与多 rank 共用同一套正确性和背压语义。

## 两套独立的生命周期

系统刻意区分 decoded cache 与输出 batch：

| 资源 | 何时可以复用 | 控制参数 |
|---|---|---|
| decoded cache extent | 消费旧 generation 的所有 scatter 完成后 | `cache_capacity_bytes` |
| output ring slot | 消费者释放 NumPy view 后 | `prefetch_step` |

cache 复用由 `JobDone/PrefixDone` 推进，不必等待模型消费；output slot 复用则由消费者 lease/ACK 推进。模型消费慢时只会压住输出环，不会改变已编译好的 cache 依赖；反过来，cache 可以继续按静态计划推进到输出环允许的边界。

## 数值与内存安全

- 支持 `i16/i32/i64/u16/u32/u64/f32/f64`，输出 dtype、fill、舍入和溢出策略均显式声明；
- CSR indices 在进入 unsafe scatter kernel 前验证边界和严格递增；
- Dense/CSR 根据 feature map 选择 identity、runs、gather、packed 或 sparse lookup 等路径；
- 编译 arena、工作集、输出环、单 job block/cell 数、I/O staging 和 worker in-flight 资源都有硬上限；
- 运行时不维护 LRU、cache hash lookup、动态 allocator 或 cache refcount，地址和覆盖关系全部由 Plan 固化。

## 最小使用方式

```python
from scdata.load import OutputSpec, compile, register

dataset = register("atlas.scc", key="X")
plan = compile(
    dataset,
    rows,
    output=OutputSpec(dataset.n_cols, "float32"),
    batch_size=256,
    prefetch_step=8,
)

with plan.open(copy=False) as session:
    for batch in session:       # read-only zero-copy NumPy view
        train_step(batch)
```

多 rank 时改为 `plan.open_distributed(world_size)`，先创建每个 rank 的 iterator，再把 iterator 交给对应进程；底层编译计划和执行图不变。

## 设计收益与代价

**收益**

- 重复 epoch 可复用 Plan，运行时调度简单且可预测；
- block 级按需读取、缓存复用和 I/O 合并降低读放大与重复解码；
- Dense/CSR、多数据集基因对齐、dtype 转换统一输出为训练友好的 dense batch；
- 单进程和多 rank 共用实现，默认零拷贝并具备明确背压。

**代价**

- 首次编译需要扫描请求并读取必要元数据，适合“访问序列已知且会重复使用”的训练场景；
- Plan 与输入顺序、batch 参数和数据源 manifest 绑定，数据变化后必须重新编译；
- 零拷贝 view 的持有时间直接决定输出槽何时可复用，异步 H2D 场景必须在传输完成后释放。

## 汇报总结

scdata load 不是一个简单的并行 reader，而是一个面向训练访问序列的 **静态 I/O 编译器**。它的精髓可以归纳为三句话：

1. 用静态编译消除运行时重复的动态判断；
2. 用静态 cache 布局消除运行时的 cache 分配、淘汰与锁竞争；
3. 用依赖 DAG 同时表达 cache 覆盖安全、decoded 数据就绪和可并行任务。

Python 提供易用且严格的策略层，Rust 只需拓扑执行有界任务图，再通过 shared ring 将结果交付给本地或多 rank 消费者。
