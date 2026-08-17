# sc-load 静态 cache 执行图重构设计

## 1. 文档状态

本文定义 `sc-load` 下一代执行模型的目标架构和分阶段重构计划。它是实现约束，不是对当前代码的描述。

本轮重构的核心变化是：把当前“一个运行时 job 自己完成 I/O、解压、scatter，然后丢弃 decoded block”的模型，替换为“编译期模拟固定容量 cache，生成静态内存布局和依赖图；运行时只执行固定地址的读写与依赖同步”。

第一阶段不改变 Python 公共抽象。Python 继续负责数据集注册、feature mapping、输出 dtype/fill、batch 组织、Plan/Session 生命周期和 NumPy 暴露；Rust/PyO3 仍是私有机械执行边界。

## 2. 目标与非目标

### 2.1 目标

1. `Batch` 是 Plan 的最小输出和完成粒度，一个 `Batch` 严格对应一个 `Job`。
2. `Job` 是一组 task 的逻辑容器；worker 实际调度的是 ready task，而不是整份 Job。
3. decoded block 写入一次性申请的 session cache buffer，在多个 batch 之间复用。
4. cache offset、覆写顺序、block residency、依赖和优先级全部在编译期确定。
5. 运行时没有 cache allocator、LRU、hash lookup、动态驱逐或 cache 引用计数。
6. output ring 保留单消费队头和 generation lease，生产端持续填充直到 ring 或依赖图产生背压。
7. cache prefetch horizon 与 output ring horizon 解耦，使 cache 可以预取到 ring 之外。
8. blocking 与 io_uring 只是同一执行图的两种 I/O 执行策略，不拆成独立公共 I/O pool。

### 2.2 非目标

- 本轮不把 model batch size 固定到 Plan batch size。外部可以合并、切分或累积多个 Plan batch。
- 本轮不做运行时通用 cache，也不支持运行时改变访问顺序。
- 本轮不引入 buddy/slab、分页式 decoded block 或在线 compaction。
- 第一版不以 dense direct-to-output decode 为核心路径；它可以在新模型正确后作为一次性 block 的优化重新加入。
- 本轮不改变 SCC wire format、DynBlosc block format 或 feature mapping 语义。

## 3. 术语与不可变语义

### 3.1 Batch

`Batch` 是：

- 一段连续的输入访问序列；
- 一个固定 output ring generation；
- 一个 Job 的输出；
- 迭代器一次返回的内存视图；
- 完成通知、错误隔离和外部 release 的最小单位。

Plan batch size 是数据加载器的编译粒度。模型实际 batch size 可以在 Python/训练框架侧再次解耦，但不能改变 Plan 内部 batch 的顺序和生命周期。

### 3.2 Job

一个 Job 严格对应一个逻辑 batch：

```rust
struct Job {
    batch_id: u64,
    io_tasks: Range<usize>,
    csr_tasks: Range<usize>,
    dense_tasks: Range<usize>,
    output_slot: usize,
    output_generation: u64,
    completion_node: usize,
}
```

Job 是任务归属、优先级、统计和 batch completion 的容器，不是 worker 独占执行的原子。一个 Job 中的 task 可以由不同 worker 执行，也可以在不同时间变为 ready。

Job 完成的唯一含义是：该 batch 的所有必要输出写入已经完成并通过验证，可以发布对应 ring generation。不能因为 Job 中某个 task 已经被领取就认为 Job 已开始，也不能让 worker 领取一个依赖尚未满足的 task 后原地等待。

### 3.3 Task

Task 是 worker 的调度原子。第一版只有三种可执行 task：

1. `IoDecodeLoadTask`
2. `CsrScatterTask`
3. `DenseScatterTask`

依赖 gate、cache release fence 和 output-slot generation 是零工作量控制节点，不属于第四种可执行 task。它们在 predecessor 完成路径中更新依赖计数，不进入 worker ready queue。

### 3.4 Cache object 与 residency

`CacheObject` 是一个可独立解压的 SCC data 或 indices block。data 与 indices 即使覆盖相同 cells，也按独立 object 管理，因为 decoded size、物理位置和复用关系可以不同。

`Residency` 是某个 CacheObject 的一次驻留：

```text
Load(cache generation)
    -> zero or more batch readers
    -> JobDone / PrefixDone availability epoch
    -> the same address may be overwritten
```

同一个 CacheObject 在 cache 较小时可以产生多个 residency；每次重新驻留都会生成新的 load task 和 cache generation。

## 4. Plan 与 Session 的两层表示

### 4.1 可重用 Plan 只保存 relocatable 数据

可复用、可序列化的LogicalPlan不能保存session cache、output ring或worker staging的绝对裸指针；它保存ID和offset。worker实际执行的ExecutionPlan必须使用pointer。第一版不为节省LogicalPlan大小引入`u16`、24-bit index、位域或有损长度；内存表示优先使用`usize`，持久化wire表示统一使用经过边界检查的`u64`：

```rust
struct CacheSlice {
    offset: usize,
    len: usize,
    generation: u64,
}

struct OutputSlice {
    ring_offset: usize,
    len: usize,
    generation: u64,
}
```

文件以 source ID / read-source ID 表示，mapping 与 decoder metadata 以 arena index 表示。文件 offset、batch ID、generation 固定使用 `u64`；公开稳定的 SourceId 可以继续使用 `u32`。详细字段规则见第 19 节。

### 4.2 Session::open 强制构建 pointer-rich ExecutionPlan

Session 分配 cache buffer 和 output ring 后，必须执行一次copy-and-lower，而不是让worker直接执行index-based Plan IR。该阶段按实际Job/priority/cache访问顺序重排并复制所有hot descriptors，随后把引用修成session-local指针形式：

```text
cache_ptr  = cache_base + cache_offset
out_ptr    = output_base + ring_offset
mapping*   = plan mapping arena + mapping offset
decode_meta* = plan decoder arena + decoder offset
fd         = resolved session-local file descriptor
```

最终worker只接收`ExecutionPlan`中的pointer+length descriptor，不在hot loop中用task/source/mapping index随机访问多个全局Vec。整数索引只存在于LogicalPlan、PlanImage、bind和lowering阶段。

这些指针只在Session生命周期内有效，不进入可clone/reuse或序列化的LogicalPlan。ExecutionPlan构建完成后所有backing arenas冻结，不得reallocate或移动。

## 5. 三种执行 task

### 5.1 IoDecodeLoadTask

逻辑 ABI：

```text
(source, f_off, f_len,
 [(b_off, b_len, decode_meta, cache_target, ready_token), ...])
```

其中：

- `f_off..f_off+f_len` 是一次物理 range read；
- encoded 数据进入 worker-local 临时 buffer；
- `b_off` 相对于本次读取结果，而不是文件起点；
- `cache_target` 是 session cache 内的静态 offset/generation；
- 每个 decode op 写入不重叠的 cache range；
- 每个 block 解压成功后单独发布 `ready_token`，不必等待同一 read 中其他 block 解压完。

建议表示：

```rust
struct IoDecodeLoadTask {
    source: u32,
    file_offset: u64,
    file_len: usize,
    decode_ops: Range<usize>,
    earliest_consumer_batch: u64,
}

struct DecodeOp {
    encoded_offset: usize,
    encoded_len: usize,
    decoder: usize,
    cache_offset: usize,
    decoded_len: usize,
    cache_generation: u64,
    ready_node: usize,
}
```

物理 range coalescing 必须同时满足：

- 同一 read source；
- 编码区间重叠或合并有利；
- 不超过 encoded staging 上限；
- 不把相距过远优先级的 block 强行绑定为一个长 task；
- 每个 DecodeOp 仍能独立发布 ready。

Blocking worker 使用一个可增长复用的 `Vec<MaybeUninit<u8>>`，而不是每个 task 都向 OS 申请/释放映射。io_uring worker 为固定 in-flight slot 保留 encoded buffer，CQE 完成后执行相同的 decode op 列表。临时 encoded buffer 很小且不进入 decoded cache。

### 5.2 CsrScatterTask

最小逻辑 ABI：

```text
(data_ptr, indices_ptr, nnz, mapping, out_ptr)
```

Plan 形式使用 cache/output offset：

```rust
struct CsrScatterTask {
    data: CacheSlice,
    indices: CacheSlice,
    data_byte_range: Range<usize>,
    indices_byte_range: Range<usize>,
    mapping: usize,
    output: OutputSlice,
    source_plan: usize,
}
```

CSR indptr 已在编译时解析成每个 cell 的 data/indices byte range；运行时不需要重新遍历 block-local indptr。若未来需要把一个 task 扩展成多 cell，可额外引用连续 `CsrRowTask` arena，但基本语义仍是 cache read -> mapped scatter -> one output row。

结构验证和可能失败的 dtype conversion 必须在发布 batch 前完成。一个 cache block 的结构性验证若与 mapping 无关，可以提升到 load/residency 级只执行一次；与输出策略有关的检查保留在 scatter 路径。

### 5.3 DenseScatterTask

最小逻辑 ABI：

```text
(data_ptr, len, mapping, out_ptr)
```

建议表示：

```rust
struct DenseScatterTask {
    data: CacheSlice,
    source_byte_range: Range<usize>,
    mapping: usize,
    output: OutputSlice,
    source_plan: usize,
}
```

identity、runs、packed map、gather 和 dtype conversion 在 `SourcePlan`/mapping arena 中预绑定。输出 fill 必须由 Dense/CSR scatter 语义覆盖，不能依赖 ring slot 只在第一次使用时为零；ring generation 被复用时仍需得到完整初始化的逻辑 row。

## 6. 连续 task arena 与 Job task list

Plan 使用按类型连续存储：

```rust
struct PlanData {
    initialize: InitializeJob,
    jobs: Box<[Job]>,
    io_decode_tasks: Box<[IoDecodeLoadTask]>,
    decode_ops: Box<[DecodeOp]>,
    csr_scatter_tasks: Box<[CsrScatterTask]>,
    dense_scatter_tasks: Box<[DenseScatterTask]>,
    dependencies: DependencyGraph,
    cache_layout: CacheLayout,
    output_layout: OutputRingLayout,
    // sources, decoders, mappings, validation kernels, statistics ...
}
```

每个 Job 直接保存三种 task arena 的连续 range，而不是维护异构 `TaskRef` 跳转表。三种 task 自身分别位于连续大数组，并按 Job 顺序分段；依赖决定执行先后，不依赖异构 list 中的物理顺序。`InitializeJob` 引用 `io_decode_tasks` 的初始连续 prefix，普通 Job 的 I/O ranges 从该 prefix 之后开始。

任务逻辑归属遵循：

- scatter task 归属其输出 batch 的 Job；
- 初始、无需等待旧 cache generation 的 load task 归属独立 InitializeJob；
- 后续 cache residency 的 load task 归属最早消费该 residency 的 Job；
- 后续 load task 可以有早于 owner Job output slot 的 `available_after_batch`，因此能提前执行；
- 不把“为未来 batch 做的 load”计入当前 batch 的 completion path，避免远端预取拖慢队头 batch。

`available_after_batch` 表达“哪个有序完成前缀之后允许覆写”，`earliest_consumer_batch` 表达优先级和语义 owner；两者不能混为同一字段。

## 7. 编译期固定容量 cache 算法

### 7.1 输入

- 有序 `(source_id, row_id)` 序列；
- Plan batch size；
- cache capacity；
- output ring slots；
- SCC block metadata 和 decoded size；
- feature mapping、dtype/fill/overflow policy；
- I/O coalescing 和 staging 上限。

### 7.2 构建 batch block requirements

编译器先解析每个 cell 对应的 data/indices block，并为每个 batch 构建去重 requirements：

```text
BatchBlockRequirement {
    cache_object,
    cell/scatter consumers,
}
```

同一 batch 多次访问同一 block 只增加 reader 列表，不重复加载。data 与 indices 分开去重。

### 7.3 编译期 resident table

编译器维护：

```text
resident[CacheObject] -> {
    CacheSlice,
    compile_refcount,
    load_task,
    last_reader_batch,
}
```

该 refcount 只存在于编译器：它表示已经纳入预取窗口、尚未在编译模拟中完成的 batch references。它不会进入运行时 cache hot path。

### 7.4 贪婪填满固定 cache

从队头 batch 开始，编译器反复尝试把更远 batch 的 requirements 加入 cache：

1. 当前 batch 优先，batch ID 越远优先级越低。
2. 同一 batch 中，缺失 block 按 aligned decoded size 降序尝试，使用 Best-Fit extent。
3. 已 resident block：当前 batch requirement 绑定已有 residency，编译 refcount `+1`。
4. 缺失且能放入：创建 residency、cache offset、load task 和 load->reader 依赖，refcount 设为 1。
5. 当前未来 batch 中放不下的 block 暂留 pending；仍尝试该 batch 中能装入的更小 block，以减少孔洞。
6. pending batch 未完整绑定前，不越过它去处理更远 batch，保证最近 batch 优先。
7. 当没有任何 pending block 能放入时停止向后预取。

这使 cache 在 block 粒度上尽可能填满，但不承诺字节级 100% 利用率。停止时可能是：

- 总空闲不足；
- 总空闲足够，但不存在足够大的连续 extent；
- 当前 batch 自身 working set 大于 cache，这是编译错误。

### 7.5 batch 完成与编译期释放

模拟 batch 完成时，对它绑定的每个 residency 执行 refcount `-1`：

- refcount > 0：更远的已纳入 batch 仍需要该 residency；
- refcount == 0：该 cache extent 在逻辑上可覆写，归还 Best-Fit free extent tree，并把最后 reader batch 记录为 availability epoch。

释放后继续贪婪处理 pending/future batch，直到再次放不下。

### 7.6 extent allocator

编译期 extent allocator 使用双索引：

- address-ordered tree：相邻 free extent 合并；
- size-ordered tree：`O(log F)` Best-Fit；
- 64-byte alignment；
- 每个 free extent 保存一个 `available_after_batch`；
- 分割后的 remainder 保留原 availability epoch；
- 相邻 extent 合并时取两个 epoch 的 `max`。

这里 `F` 是 free extents 数。allocator 只在 compile 中运行，运行时没有对应数据结构。

### 7.7 复杂度

令：

- `N`：cell 数；
- `R`：batch->block references；
- `B`：单 batch 最大 block 数；
- `L`：residency load 次数；
- `F`：free extent 数；
- `T/E`：最终 task/edge 数。

则总体接近：

```text
O(N + R log B + L log F + T + E)
```

batch size 固定时 `B` 有界，实际主项是 `O(N + R + L log F)`。空间复杂度为 `O(N + R + U + F + T + E)`。

真实 826 数据集实验中，metadata/block 编译约 4.69 秒，trace 约 0.25 秒；24 GiB/48 GiB cache 模拟分别约 9.54 秒和 2.25 秒。

## 8. 从编译期 refcount 到运行时静态依赖

### 8.1 运行时不保留 cache refcount

编译器完成 residency 和 offset 分配后，把 refcount 生命周期展开成显式图：

```text
Load residency R
    -> Reader(batch a)
    -> Reader(batch b)
    -> Reader(batch c)

JobDone(a/b/c) -> PrefixDone(max(a,b,c))
PrefixDone(last_reader_batch) -> next writer of overlapping cache range
```

运行时没有 `resident` map、free list 或 cache refcount。运行时只有 task dependency counter 和完成状态。

### 8.2 PrefixDone 与 cache availability epoch

第一版采用有序完成前缀压缩任意 reader 依赖：

```text
PrefixDone(0) = JobDone(0)
PrefixDone(b) = PrefixDone(b - 1) AND JobDone(b)
```

一个 residency 在编译模拟中 refcount 归零时，记录最后一个已纳入的 reader batch `last_reader_batch`。释放出的 free extent 携带：

```text
available_after_batch = last_reader_batch
```

- 初始 cache extent 使用 `INITIAL`，归属 InitializeJob；
- 从 extent 分配的新 LoadTask 依赖 `PrefixDone(available_after_batch)`；
- 分割后的 remainder 继承相同 epoch；
- 相邻 extent 合并时 epoch 取 `max(left, right)`；
- 新 writer 因此只需要一个有序前缀依赖，不需要保存任意 reader fence 集合。

这比细粒度 reader fence 更保守：即使一个 residency 的 readers 已完成，只要更早 Job 尚未完成，PrefixDone 仍不会推进。但它与有序 batch 输出、最近 batch 优先和编译期按 batch 释放完全一致，能把 cache overwrite 依赖从潜在多 predecessor 压缩为一个 predecessor。若 profile 证明 prefix head-of-line 明显限制 cache load，再增加可选 fine-grained frontier，而不是第一版直接承担图规模和合并复杂度。

`JobDone`和`PrefixDone`是逻辑控制节点，不占三类worker task arena。第一版运行时不需要把PrefixDone链物化为普通DAG nodes；可以用第21.6节的单调prefix tracker压缩。

### 8.3 依赖图存储

```rust
struct DependencyGraph {
    initial_dependency_count: Box<[u32]>,
    successor_ranges: Box<[Range<usize>]>,
    successors: Box<[usize]>,
}
```

图结构在 Plan 中只读。每个 Session 只分配：

```rust
struct RuntimeNodeState {
    remaining_dependencies: AtomicU32,
    state: AtomicU8, // Waiting / Ready / Running / Done
}
```

producer 完成 cache/output 写入后以 Release 发布；successor dependency 归零并进入 ready queue 时建立 Acquire/AcqRel 可见性。

逻辑PrefixDone依赖使用专用结构：

```rust
struct PrefixReleasePlan {
    release_ranges: Box<[Range<usize>]>, // indexed by batch
    released_loads: Box<[usize]>,
}

struct RuntimePrefixState {
    job_done: Box<[AtomicBool]>,
    next_unfinished_batch: AtomicU64, // starts at 0
}
```

JobDone置位后尝试从`next_unfinished_batch`连续推进；每跨过一个batch b，批量释放`release_ranges[b]`中的LoadTasks。这样保留`PrefixDone(b)`语义，但不为每个batch存储`PrefixDone(b-1) -> PrefixDone(b)`和`JobDone(b) -> PrefixDone(b)`两条普通边。

编译结束必须对数据 task + control gate 的完整图做拓扑检查，并验证每个 task 只完成一次、每个 ring/cache 写地址的先后关系闭合。

## 9. 优先级与 ready-task 调度

### 9.1 worker 只领取 ready task

worker 不能领取依赖未满足的 task 后等待。否则所有 worker 都可能阻塞在未来 cache 上，而真正的 producer 尚未被领取。

task 只有在 `remaining_dependencies == 0` 后才进入 ready queue；worker 通过 CAS 将 `Ready -> Running`，完成后推进 successors。

### 9.2 batch 优先级

每个 task 编译：

```text
earliest_consumer_batch = 所有下游输出 batch 的最小值
```

优先级主键：

```text
(earliest_consumer_batch, critical_path_class, task_id)
```

- 当前最小未消费 batch 天然具有最高优先级；
- batch 越远优先级越低；
- 阻塞近端 batch 的 load/decode 会因 earliest consumer 较小自然提前；
- 相同 batch 内 task ID 提供确定性；
- 不需要在 consume head 每次移动时重写所有 task 的 priority。

实现上优先使用按 batch 分桶的 ready queues，而不是所有 task 竞争一个全局 heap。每个 worker 保留 local deque，dependency completion 将新 ready task 注入对应 batch bucket；必要时 work stealing。

### 9.3 防止远端预取影响队头

- 远端 load 不能占用队头 batch 所需的唯一 worker；至少保留 critical execution capacity。
- 每处理一批 prefetch task 后重新检查更高优先级 bucket。
- I/O coalescing不能把当前 batch block 与非常远的 block 绑定成一个无法抢占的大 decode loop。
- DecodeOp 按 earliest consumer 排序，并逐 block 发布 ready token。

## 10. Output ring、完成队列与外部背压

### 10.1 静态 outptr

逻辑 batch `b` 映射到：

```text
slot = b % ring_slots
generation = b
outptr = ring_base + slot * batch_stride + row_in_batch * row_stride
```

这些 offset 在编译时确定；Session lowering 后 scatter task 直接使用指针。

### 10.2 ring generation 依赖

- 前 `ring_slots` 个 generation 初始可写；
- batch `b + ring_slots` 的 scatter task 依赖消费者 release batch `b`；
- `BatchReady(b)` 依赖该 batch 所有 scatter task 完成；
- 消费者只移动单一有序队头；
- consumer release 发布下一代 output slot 可写，并唤醒更远 scatter task。

生产者可以乱序完成不同 batch，但外部迭代器仍按逻辑 batch 顺序返回。后续 batch 已完成而队头未完成时，它留在 ring 中，不绕过有序消费。

### 10.3 两种背压

系统同时存在：

1. cache dependency backpressure：新 LoadTask 等待重叠旧 residency 对应的 PrefixDone availability epoch；
2. output ring backpressure：远端 ScatterTask 等待消费者释放对应 slot generation。

worker 不在这两种依赖上阻塞；没有 ready task 时才休眠等待图状态变化。

## 11. 多级 prefetch

该架构天然形成三个层级：

### L1：encoded I/O staging

每 worker/io_uring slot 的小型临时 buffer，只覆盖 in-flight physical reads。它由执行配置限制，不进入长期 cache。

### L2：decoded cache prefetch

由固定 cache capacity 和静态 residency graph 决定。它可以远远超过 output ring slots；真实随机访问实验中：

- 24 GiB cache 中位可覆盖约 3,043 batches；
- 48 GiB cache 中位可覆盖约 14,662 batches。

### L3：output ring prefetch

由 `ring_slots` 和 consumer head 决定，只物化近期可交付 batch。

因此 decoded data 可以提前很远准备，但 output scatter 仍被 ring 限制。若 cache horizon 小于 ring horizon，ring 可能无法长期填满；这不是错误，但说明 cache 是实际瓶颈。Plan stats 必须报告两个 horizon，不能只暴露一个 `prefetch_step`。

建议逐步把旧的单一 `prefetch_step` 配置拆成：

```text
cache_capacity_bytes
output_ring_slots
io_inflight_limit
```

Python 可以保留一个便捷配置层，但 Rust Plan 必须使用明确资源量。

## 12. 正确性与安全不变量

编译器必须证明或验证：

1. 每个 requested cell 恰好属于一个 Job/output row。
2. 每个 scatter 的输入 CacheSlice 在读期间由一个成功发布的 residency generation 拥有。
3. 任意两个可能并发的 cache writers 不重叠。
4. 新 writer 覆盖旧地址前，对应 `PrefixDone(available_after_batch)` 已发布，因此旧 residency 的全部 readers 已完成。
5. 任意两个可能并发的 output writers 写入不重叠 row。
6. ring slot generation 未被消费者 release 前不会覆写。
7. batch 只有在所有逻辑 row 和 padding 初始化后才能发布。
8. 错误或 cancellation 不发布未完成 cache token 或 BatchReady。
9. 单 batch working set 不得超过 cache capacity；否则 compile 失败并报告所需字节。
10. DAG 无环，dependency counter 不溢出，task/successor 索引均有资源上限。

Plan 中不保存 session-owned raw pointers。Session-local unsafe pointer lowering和 slice 构造必须在每处陈述 disjointness、generation 和长度不变量。

## 13. 错误、取消和生命周期

- 第一个 I/O/decode/validation/scatter 错误进入 terminal session state；
- terminal transition 唤醒 consumer、ready workers、ring waits 和 shared-ring waits；
- 已部分写入但未发布的 cache generation 不可被 reader 使用；
- 已部分写入但未发布的 output generation 不可被 consumer 使用；
- cancellation 后不再领取新 task，但必须安全回收/等待已提交 io_uring CQE；
- cache buffer 不需要逐 block 清理；Session drop 统一释放整块 arena；
- output Batch lease 的现有零拷贝生命周期继续保持，异步 H2D 必须在 release 前完成。

## 14. 统计与可观测性

### Compile stats

- input cells / batches；
- unique data/indices blocks；
- cache capacity、alignment loss；
- residency loads / reloads / decoded bytes；
- reference hit ratio；
- capacity stalls / fragmentation stalls；
- stall utilization p50/p95；
- prefetch horizon p50/p95/max；
- executable tasks、control gates、edges；
- cache/ring arena bytes；
- compiler phase timings。

### Runtime stats（profile feature）

- task ready/run/completion counts by kind；
- ready queue wait/steal/contention；
- physical I/O ops/bytes/latency；
- decode blocks/bytes/time；
- scatter rows/bytes/time；
- dependency releases；
- ring-ready latency和 consumer wait；
- priority inversion或 critical bucket starvation事件。

24/48 GiB 真实随机访问基线：

| Cache | reference hit | decode amplification | median utilization | median cache horizon |
|---|---:|---:|---:|---:|
| 24 GiB | 50.15% | 6.171x | 98.82% | 3,043 batches |
| 48 GiB | 88.32% | 1.446x | 99.77% | 14,662 batches |

这些数字用于重构前后计划语义对照，不代表最终 runtime throughput。

## 15. 分阶段实现计划

### Phase 0：冻结语义和基线

- 保留当前公共 Python API 和数值测试；
- 固化真实 24/48 GiB cache compiler benchmark 数据；
- 为新 Plan IR 写尺寸和上限预算；
- 明确旧 `prefetch_step` 到新资源配置的兼容转换。

交付条件：本文档中的术语、配置和不变量不再有二义性。

### Phase 1：新 Plan IR 与连续 arenas

- 新增 InitializeJob、Job task ranges、三类 task、DecodeOp、CacheSlice、OutputSlice；
- 新增 immutable DependencyGraph、JobDone/PrefixDone/control gate；
- 实现PlanImage/PlanTemplate sectioned wire roundtrip和deserialize limits；
- 新旧 compiler 并存，运行时仍使用旧 Plan；
- 添加 Plan dump/统计，检查 arena size、task count和edge count。

交付条件：新 compiler 可以只编译并序列化/检查计划，不执行数据。

### Phase 2：block requirements 与 cache compiler

- 复用当前 cell->chunk->block metadata解析；
- 构建每 batch 去重 block requirements；
- 将实验 Rust Best-Fit/refcount 算法迁入 `sc-load` compiler；
- 生成 residency、cache offset、availability epoch、JobDone/PrefixDone控制链和load ownership；
- 在cache residency确定后执行Off/Adjacent/CostAware最终I/O fusion pass；
- 实现SourceLocator/Manifest严格bind、decoder/kernel重建；
- 增加Python lazy load/save/dumps/loads/pickle和source override；
- 编译期检测 deadlock、单 batch 超容量和 DAG cycle。

交付条件：真实 24/48 GiB 计划统计与独立实验器一致，且无需分配实际 cache bytes。

### Phase 3：blocking DAG runtime

- Session 一次性分配 cache buffer/output ring；
- 两遍分配ExecutionPlan arena，复制/重排hot descriptors并lowering为session-local pointers；
- 先用独立临时线程池执行InitializeJob并销毁初始化线程；
- 实现ready-task scheduler、batch priority和dependency completion；
- 实现blocking IoDecodeLoadTask及worker-local staging；
- 接入Dense/CSR scatter；
- 保留旧执行器作为对照开关。

交付条件：dense/CSR、mapping、dtype、fill、overflow、取消和错误传播全部通过；worker 不等待未满足依赖。

### Phase 4：output ring 与外部迭代器切换

- 用新 Job completion发布BatchReady；
- 接入单消费队头、generation lease和release backpressure；
- 验证慢consumer、保留Batch、异步copy和ring wrap；
- Python `Plan`/`Session` 接口保持兼容。

交付条件：普通 iterator 与共享 ring 的batch顺序、生命周期和取消语义一致。

### Phase 5：io_uring

- 将IoDecodeLoadTask lowering为per-worker ring SQE；
- 保留固定slot buffer和CQE lifetime；
- CQE后按DecodeOp优先级解压并逐block发布ready；
- 明确short read、cancel CQE、fallback和resource admission。

交付条件：blocking/io_uring数值一致，io_uring不可用时Auto安全回退。

### Phase 6：共享 ring 与分布式消费

- 复用现有单producer shared output mapping；
- cache仍为producer进程私有，不暴露给rank；
- 将新BatchReady/consumer release连接到现有futex控制面；
- 验证rank停滞、lease、owner死亡和取消。

### Phase 7：性能收敛与删除旧路径

- profile task/queue/decode/scatter/ring；
- 调整任务粒度、I/O coalescing和ready bucket实现；
- 恢复有收益的direct decode/fused scatter优化；
- 完成同build、固定CPU、重复配对benchmark后删除旧job执行器。

## 16. 验证矩阵

### 编译器属性测试

- 随机变长block、随机batch引用、不同cache容量；
- 每个cache read都有唯一覆盖它的load generation；
- address overlap与dependency关系一致；
- extent split/coalesce后availability epoch正确；
- 编译结果DAG无环；
- compile refcount最终归零；
- 24/48 GiB真实trace可重复。

### 数值测试

- Dense/CSR identity、mapping、drop、permutation；
- 所有storage/output dtype和允许的promotion；
- checked signedness与overflow policy；
- empty row、oversized single-cell block、跨chunk边界；
- batch尾部不足batch size；
- ring wrap后fill/padding仍正确。

### 并发与生命周期

- 1/2/N workers输出一致；
- producer慢、reader慢、scatter慢；
- 远端prefetch不能饿死当前batch；
- task panic/I/O error/decode error/cancel；
- Session/Batch提前drop；
- io_uring CQE与buffer slot不悬空；
- shared client/rank退出。

### 性能验证

- compile时间与内存；
- cache hit/decode amplification；
- decoded GiB/s、cells/s、batch latency；
- 24/48 GiB真实随机访问；
- blocking与io_uring分别报告；
- output ring不满时区分cache瓶颈、decode瓶颈和consumer背压。

## 17. 建议的第一版配置

```text
plan_batch_size        = 128（由Python/调用者明确指定）
cache_capacity_bytes   = 必填或由Python策略明确推导
output_ring_slots      = 独立配置，不再等同cache horizon
io_inflight_ops/bytes  = Session配置
cache_alignment        = 64 bytes
cache_fit              = compile-time Best-Fit
priority               = earliest consumer batch first
io_merge               = IoMergeConfig(policy="adjacent", ...)
```

cache arena 建议使用一次性匿名映射或等价未初始化大块分配，不使用 `MAP_POPULATE`，由decode worker first-touch；运行期间不逐block `mmap/munmap` 或 `MADV_DONTNEED`。NUMA策略应作为Session配置或部署策略单独验证。

## 18. 最终架构摘要

```text
Python request / rows / mappings / output policy
                    |
                    v
          Static Rust Compiler
  block requirements + bounded cache simulation
  Best-Fit offsets + residency generations
  load/scatter tasks + dependency/control graph
                    |
                    v
          Immutable relocatable Plan
                    |
             Session::open/lower
                    |
       cache arena       output ring
            |                 |
      LoadDecodeTask -> Dense/CsrScatterTask
            |                 |
            +---- DAG --------+
                    |
               BatchReady
                    |
          ordered external consumer
                    |
          ring generation release
```

编译器表现得像一个知道完整未来访问序列的动态堆与prefetch调度器；运行时不再管理cache，只执行编译好的地址、读写和先后依赖。

## 19. 字段表示：pointer、整数和精度

### 19.1 三层表示规则

同一个逻辑 task 有三种表示，不能混用字段责任：

| 层 | 生命周期 | 地址表示 | 目的 |
|---|---|---|---|
| LogicalPlan IR | 可 clone、可序列化前的逻辑计划 | `usize` index/offset、`u64` file offset/generation | 安全编译与检查 |
| Plan wire image | 跨进程、跨机器文件 | 小端 `u64` offset/len/index，显式 enum code | 稳定持久化 |
| ExecutionPlan | 单个 Session | `RawFd`、`NonNull<u8>`、pointer+length slices | 最短 worker hot path |

不可序列化或跨 Session 保存：

- `*const/*mut T`、`NonNull<T>`；
- `RawFd`/`File`；
- `Arc<dyn ByteStore>`；
- 函数指针、SIMD kernel pointer；
- `Atomic*`、ready queue slot和任何运行状态。

### 19.2 第一版整数类型决定

不为了压缩 Plan 大小引入低精度编码：

- arena index、task index、mapping index、内存 offset/len：内存中 `usize`；
- 文件 offset、batch ID、ring/cache generation：`u64`；
- public SourceId：保留 `u32`；
- dependency count：wire 中 `u64`，bind 后验证并转换为 `u32`/`AtomicU32`；
- dtype、task kind、source kind：显式 `repr(u8/u16)` code，但不位打包进 offset；
- row/cell index：`u64`，与数据集 shape 合同一致。

`usize` 让边界检查、切片和pointer lowering保持直接，避免大量 `try_from` 出现在compiler内部。wire统一使用`u64`避免32/64-bit平台改变结构布局；反序列化时所有`u64 -> usize/u32`都必须检查。

若后续 profile 证明 task descriptor bandwidth 是明显瓶颈，可以只压缩经过统计证明的冷字段；不能在第一版用隐式 sentinel、低bit flag或窄offset换取复杂度。

### 19.3 ExecutionPlan copy、重排与pointer lowering

Session创建并固定所有backing allocation后，先根据LogicalPlan计算ExecutionPlan的精确Layout，一次性分配execution descriptor arena，再按执行局部性复制、重排并fix up pointers：

1. initialize I/O tasks连续；
2. regular tasks按priority bucket、Job和task kind排列；
3. 同一I/O task的DecodeOps紧随task或位于连续side arena；
4. 同一cache generation的scatter groups相邻；
5. 小型、频繁访问的mapping header/decoder header可复制到hot arena；
6. 大mapping table只复制一次并由pointer共享；
7. dependency successor slices重排为ExecutionNode邻接pointer/length。

生成不可移动的runtime descriptors：

```rust
struct RuntimeDecodeOp {
    encoded_offset: usize,
    encoded_len: usize,
    decoder: *const BoundDecoder,
    target: NonNull<u8>,
    decoded_len: usize,
    ready_node: usize,
}

struct RuntimeIoDecodeLoadTask {
    source: *const RuntimeSource,
    file_offset: u64,
    file_len: usize,
    decode_ops: *const RuntimeDecodeOp,
    decode_op_count: usize,
}

struct RuntimeCsrScatterTask {
    data: *const u8,
    indices: *const u8,
    nnz: usize,
    mapping: *const BoundCsrMapping,
    output: NonNull<u8>,
    completion_node: usize,
}
```

ExecutionPlan创建后，descriptor/cache/output/mapping/decoder arenas都不得移动或reallocate。worker不得回退到LogicalPlan索引查找。raw pointer只在私有执行模块内可见；每个unsafe kernel调用仍需用本地长度和编译不变量构造精确slice，不能把整个cache变成长期`&mut [u8]`共享给workers。

copy-and-lower是每个Session一次性的强制成本。它允许可序列化Plan保持稳定、易验证，同时允许worker布局根据当前source backend、CPU ISA、cache/ring基址和worker配置重新优化。

### 19.4 稳定地址与drop顺序

ExecutionPlan是自引用raw-pointer结构，不能先构建Vec再继续push。必须两遍构建：

1. 统计所有runtime records和alignment，计算完整Layout；
2. 一次性分配稳定arena或若干最终`Box<[MaybeUninit<T>]>`；
3. 复制records但暂存relative fixup；
4. backing地址最终确定后统一写入raw pointers；
5. 验证每个pointer落在预期execution/cache/ring/source arena范围内；
6. 发布不可移动ExecutionPlan。

Rust实现可用私有`AlignedBuffer + NonNull`或多个最终Box，不对外创建自引用Rust references。ExecutionPlan不可Clone、不可序列化，移动其owner不能改变backing allocation地址。

drop顺序必须固定：停止领取task -> cancellation/CQE收敛 -> join普通workers -> drop io_uring rings/staging -> drop ExecutionPlan descriptors -> drop cache/output/source handles。任何worker存活时不得释放execution arena。

## 20. 原生 Plan 序列化与惰性资源绑定

### 20.1 分离 PlanImage、PlanTemplate、BoundPlan 与 ExecutionPlan

序列化不能直接dump Rust struct内存。定义三层对象：

```text
PlanImage     = 稳定二进制wire image，不持有I/O资源
PlanTemplate  = 已验证的native逻辑plan，仍未打开source
BoundPlan     = 已解析真实source/meta/decoder/kernel，可open Session
ExecutionPlan = Session::open后复制、重排、pointer lowering的worker专用plan
```

低层Rust接口建议：

```rust
impl Plan {
    fn encode(&self) -> Result<Vec<u8>>;
}

impl PlanTemplate {
    fn decode(bytes: &[u8], limits: PlanImageLimits) -> Result<Self>;
    fn sources(&self) -> &[SourceLocator];
    fn bind(self, resolver: &dyn SourceResolver, verify: VerifyMode) -> Result<Plan>;
}
```

`decode`只解析、校验和分配Plan核心arrays，不打开路径；`bind`才读取真实meta、CSR indptr和被引用chunk的decoder prefix。实际encoded payload直到Session执行IoDecodeLoadTask时才读取。

BoundPlan仍然可以被多个Session安全复用，内部使用索引/Arc管理source和semantic task。每个Session从BoundPlan构建自己的ExecutionPlan，因为cache/ring基址、fd、runtime atomics、worker配置和ISA-bound kernels都属于本次执行。

### 20.2 SourceLocator

每个source保存稳定定位信息，不保存fd或ZIP物理entry offset：

```rust
enum SourceLocator {
    Directory {
        path: Utf8Path,
        matrix_prefix: String,
    },
    Zip {
        archive_path: Utf8Path,
        matrix_prefix: String,
    },
}
```

路径默认保存Python注册时规范化后的绝对UTF-8路径。Python保存API可以选择相对于plan文件目录进行relativize，并在load时恢复。ZIP只保存archive path和logical prefix；bind时重新解析entry位置、compression method和positioned/WholeKey能力，绝不复用旧archive的物理base offset。

Python层支持source override：

```python
Plan.load(
    path,
    sources={source_id: new_path_or_location},
    verify="strict",
)
```

这用于共享盘迁移、容器挂载点变化和归档移动，不改变SourceId、shape或plan语义。

当前PyO3 compile边界只把`PyDataset.inner` clone进Rust，文件定位信息会在`Arc<dyn ByteStore>`后丢失。重构时必须让Python Dataset把`path/key/zip_prefix`正规化为SourceLocator并随Source一起传入compiler；不能从trait object反向猜路径。

第一版只有Directory和ZIP filesystem sources可直接原生序列化。内存store、自定义ByteStore或匿名fd source必须提供调用者定义的稳定`resolver_key`，否则`Plan.encode()`返回`Unsupported`并列出不可序列化source IDs，不能偷偷把payload嵌入plan image。

### 20.3 SourceManifest 与 stale plan 检测

序列化保存编译时source合同：

- kind、shape、value/index dtype；
- partition、chunk offsets；
- matrix meta内容digest；
- CSR indptr长度、末值和digest；
- 每个被引用chunk的logical key、declared encoded len、decoder-prefix digest；
- 每个被引用block的encoded/decoded len和block index；
- Directory/ZIP source能力分类，但不保存打开句柄。

默认`VerifyMode::Strict`在bind时重新读取meta、CSR indptr和被引用chunk prefix并比较manifest。任何不一致返回`StalePlan`，不能尝试“尽量执行”。绑定成功后，Directory/ZIP generation由新打开的handle固定，Session不再观察路径替换。

可以提供`MetadataOnly`作为显式诊断/受控环境模式，但Python公共API默认始终Strict；不提供静默关闭边界检查的pickle路径。

### 20.4 Decoder 与 kernel 不直接序列化

以下内容必须重新bind：

- `BlockDecoder`内部实现；
- dtype conversion函数指针；
- AVX-512/AVX2/SSE2/scalar dispatch；
- positioned file handle和ZIP cursor；
- mapping raw pointer。

wire保存semantic descriptor：source/chunk/block ID、dtype、mapping种类、overflow/fill策略和预期长度。bind从真实decoder prefix重建`BlockDecoder`并进行本机ISA dispatch。因此plan image可以在不同CPU feature集合上加载，并自动选择正确kernel。

### 20.5 Sectioned binary wire format

不使用`bincode(struct)`、Rust ABI dump或pickle嵌套Rust对象。采用显式小端section格式：

PlanImage保存的是LogicalPlan数组和索引关系，不保存ExecutionPlan的重排顺序、绝对pointer或runtime padding；这些内容在每次Session::open时重新生成。

```text
Header
  magic = "SCPLAN01"
  format_major / format_minor
  flags
  section_count
  total_len
  header_checksum

SectionTable[]
  section_kind
  section_version
  alignment
  offset
  length
  checksum

Sections
  Manifest / StringTable / Sources
  Config / OutputLayout / CacheLayout
  InitializeJob / Jobs
  IoDecodeTasks / DecodeOps
  CsrScatterTasks / DenseScatterTasks
  Mappings / SemanticDecoderDescriptors
  DependencyCounts / SuccessorRanges / Successors
  OptionalCompileStats
```

所有record有固定wire size或显式offset+count；所有section在读取前检查范围、重叠、alignment、数量上限和checksum。建议使用BLAKE3做image/manifest完整性校验，但checksum只用于损坏检测，不代替source stale验证。

第一版反序列化把wire records逐字段转换到native `Box<[T]>`，不直接mmap后把bytes cast成Rust structs。Plan相对cache/data很小，复制成本可忽略；显式转换避免alignment、endianness、bool/enum有效值和Rust布局造成未定义行为。以后若Plan image本身达到瓶颈，再为经过验证的plain-old-data sections增加只读mmap快路径。

兼容规则：

- major不匹配直接拒绝；
- minor只允许跳过标记为optional且未知的section；
- required section缺失、重复或version未知直接拒绝；
- package version仅用于诊断，不作为wire兼容判断；
- deserialize limits限制总bytes、tasks、edges、sources、strings和mapping entries。

### 20.6 Python API 与 pickle

Rust/PyO3只暴露低层机械接口：

```text
_core.plan_serialize(_Plan) -> bytes
_core.plan_deserialize(bytes) -> _PlanTemplate
_core.plan_bind(_PlanTemplate, source_overrides, verify) -> _Plan
_core.plan_template_meta(_PlanTemplate) -> dict
_core.plan_template_sources(_PlanTemplate) -> list[dict]
```

Python `Plan` 负责完整UX：

```python
plan.dumps() -> bytes
Plan.loads(blob, *, sources=None, verify="strict", lazy=True) -> Plan
plan.save(path, *, relative_sources=False)
Plan.load(path, *, sources=None, verify="strict", lazy=True) -> Plan
plan.bind(*, sources=None, verify="strict") -> Plan
```

`Plan.loads/load(lazy=True)`只创建`_PlanTemplate`；shape、dtype、batch_count、stats等属性来自image manifest，不触发I/O。第一次`open/open_distributed/read/iter_batches`时由Python调用bind。`lazy=False`显式立即验证source。

Python Plan内部允许二选一状态：`_template`或`_inner`，并用锁保证两个线程并发第一次open时只bind一次。bind成功后可以释放template的重复decoded arrays，但保留原生image bytes不应是强制要求；`dumps()`可由BoundPlan重新编码。bind失败不得留下半绑定Plan，修正路径后仍可通过显式`plan.bind(sources=...)`重试或重新load。

pickle使用版本化`__reduce_ex__`，返回模块级restore函数和native bytes，不pickle `_Plan` capsule、Dataset handle或Session：

```python
def __reduce_ex__(self, protocol):
    return (_restore_plan_v1, (self.dumps(),))
```

unpickle默认恢复lazy Plan，避免multiprocessing spawn在反序列化阶段同时打开数百个dataset。`Session`、正在消费的iterator、Batch lease和已bind的runtime pointer都不可pickle。

`Plan.save`由Python负责同目录临时文件、flush/fsync策略和`os.replace`原子提交；Rust只生成/解析bytes。对不可信`.scplan`先按PlanImageLimits解析，bind路径还应允许调用者提供resolver/allowlist。pickle本身仍按Python规则视为不可信代码格式。

## 21. Job/task/dependency 的内存布局

### 21.1 hot/cold分离

Plan内存分成：

- hot：Job、三类task、DecodeOp、dependency counts/ranges/successors；
- warm：mapping entries、semantic decoder descriptors、SourcePlan；
- cold：路径字符串、digests、serialization manifest、编译统计和诊断文本。

worker hot loop不能为了执行一个task触碰SourceLocator、字符串或digest页面。

### 21.2 LogicalPlan以Job为主序，ExecutionPlan再次重排

LogicalPlan中的三类task arena按`batch_id`递增排列，便于序列化、检查和定位。同一Job内部先采用稳定逻辑顺序：

- I/O task按`available_after_batch`、source、file offset排列并coalesce；
- DecodeOp按`earliest_consumer_batch`、encoded offset排列；
- CSR scatter优先按`(data_generation, indices_generation, output_offset)`排列；
- Dense scatter优先按`(data_generation, output_offset)`排列。

Session copy-and-lower时不要求保持LogicalPlan的物理index。它根据最终source pointer、cache/output pointer和runtime priority重新排列ExecutionTasks，并直接修复Job和ready buckets指向的新地址。这样同一Job ready时，task descriptor、mapping、cache输入和output rows都具有局部性。scatter按cache generation分组可连续复用同一decoded block；组内再按output pointer递增，限制ring写随机性。

LogicalJob直接保存每类task range：

```rust
struct Job {
    io_tasks: Range<usize>,
    csr_tasks: Range<usize>,
    dense_tasks: Range<usize>,
    // ...
}
```

不保存`Vec<Task>`、per-Job heap allocation或linked list。空range表示该Job没有对应类型。

RuntimeJob不再保存range/index，而保存已重排ExecutionTask slice：

```rust
struct RuntimeJob {
    io_tasks: *const RuntimeIoDecodeLoadTask,
    io_task_count: usize,
    csr_tasks: *const RuntimeCsrScatterGroup,
    csr_task_count: usize,
    dense_tasks: *const RuntimeDenseScatterGroup,
    dense_task_count: usize,
}
```

### 21.3 I/O 与 DecodeOp布局

Logical `IoDecodeLoadTask`保持小而定长，DecodeOp单独连续存储。copy-and-lower后，RuntimeIoDecodeLoadTask直接保存`RuntimeSource*`和连续`RuntimeDecodeOp* + count`；一个worker领取I/O task后顺序扫描descriptor，不再用source/decode-op整数索引跳转。

不把`decode_meta*`永久写进LogicalPlan；ExecutionPlan中的DecodeOp pointer指向连续`BoundDecoder` arena。cache target pointer按DecodeOp连续存储，CPU prefetcher可以顺序读取descriptor，即使target地址本身分散。

### 21.4 scatter row grouping

如果同一cache block在一个Job中服务多个cells，第一版可保留one-row scatter descriptor，但编译器应生成连续group：

```rust
struct CsrScatterGroup {
    data: CacheSlice,
    indices: CacheSlice,
    rows: Range<usize>,
}
```

`CsrRowTask`/`DenseRowTask`单独连续保存source byte range、mapping和output offset。runtime可以一个task处理整个group，也可以按阈值切成多个task；不能为每个cell重复保存相同cache pointers和decoder信息。

### 21.5 dependency CSR布局

dependency graph使用CSR形式，successors按`(target_batch, target_kind, target_index)`排序并去重。常见fanout通过控制节点压缩：

- one DecodeOp -> one BlockReady gate -> many scatter groups；
- many task completions -> one JobDone；
- JobDone -> compressed completed-prefix tracker；
- one consumer release -> one OutputSlotReady generation。

Runtime mutable状态与immutable graph分开分配，避免worker更新atomic时污染只读successor cache lines。atomic状态按worker shard或固定chunk做cache-line padding，不为每个节点浪费完整cache line。

copy-and-lower把Logical successor IDs转换为连续runtime successor descriptors。worker完成task时直接顺序扫描`successors_ptr..successors_ptr+count`；successor descriptor可直接保存目标`RuntimeNodeState*`和ready-queue metadata pointer，避免再次经过node ID索引表。mutable atomic state地址在ExecutionPlan fix-up前必须最终固定。

### 21.6 PrefixDone专用压缩

`PrefixDone`是严格单调的有序前缀，不值得作为通用CSR图节点逐个存储。Plan把所有`available_after_batch = b`的LoadTask ID连续放入`release_ranges[b]`；runtime维护JobDone bitmap和一个cache-line独占、初始为0的`next_unfinished_batch`。

任意worker完成Job b时：

1. Release写`job_done[b] = true`；
2. 尝试获得短临界区/prefix推进权；
3. 从`next_unfinished_batch`开始Acquire检查连续done bits；
4. 对每个新跨过的batch，批量把对应LoadTasks的prefix dependency标记完成；
5. 更新next_unfinished_batch并释放推进权。

推进操作总计只跨过每个batch一次，整体`O(batch_count + released_loads)`；不会由每个worker重复扫描完整前缀。它是依赖图的专用压缩表示，不是运行时cache引用计数。

## 22. cache布局与依赖质量的联合优化

### 22.1 为什么纯Best-Fit不够

两个free extents都能容纳block时，纯Best-Fit只看剩余孔洞；但它们可能有不同`available_after_batch`。选择更晚epoch会让LoadTask依赖更晚PrefixDone，延长关键路径；选择过大的早期extent又可能恶化碎片。

因此allocator目标不是单一最小gap，而是受碎片约束的依赖优化。

### 22.2 第一版候选选择

先计算所有可容纳candidate中的：

```text
best_fit_waste = min(extent.len - block.len)
fragmentation_slack = max(64 KiB, block.len / 16)
```

只保留：

```text
extent_waste <= best_fit_waste + fragmentation_slack
```

然后按以下lexicographic score选择：

```text
(
  available_after_batch,  // 越早越好，INITIAL最优
  dependency_cost,       // 第一版通常为0或1
  extent_waste,           // 越小越好
  address                 // 确定性tie-break
)
```

这保证依赖优化不会无限牺牲内存质量。`fragmentation_slack`必须进入PlanConfig和stats，后续用24/48 GiB真实trace比较dependency horizon、hit、fragmentation和edge count；不能凭直觉扩大。

### 22.3 batch内部优先级

编译器始终先完成最近的pending batch，不跨过不完整batch。batch内：

1. 已resident requirements先绑定并增加compile refcount；
2. 缺失block按decoded size降序；
3. size相同时，未来近距离复用次数更多者优先；
4. 再按source/block ID确定性排序。

这些规则提高填充率并优先留下更可能产生hit的residency，但不能改变“当前batch必须最终完整”的正确性要求。

### 22.4 I/O coalescing不能破坏依赖

第一版只coalesce同时满足以下条件的DecodeOps：

- 同source和可合并物理range；
- 相同`available_after_batch`；
- `earliest_consumer_batch`处于同一priority bucket；
- combined encoded bytes/decode ops不超过硬上限。

否则一个晚epoch block会把早期block绑定到更晚依赖，或者一个远端大decode loop阻塞当前batch。后续若放宽，必须把增加的critical-path latency纳入compiler cost。

### 22.5 降低Job依赖数

依赖优化按Job/block group聚合，而不是按cell建边：

- 同一Job内读取相同residency的所有rows共享一个BlockReady依赖；
- CSR data+indices可以经一个PairReady gate汇合后fanout到row group；
- JobDone由scatter group completion计数，不由每个row单独向全局图fanout；
- cache释放依赖PrefixDone(last_reader_batch)，而不是列出所有reader tasks。

PlanStats新增：每Job dependencies p50/p95/max、control gate数、edges per task、PrefixDone阻塞距离。compiler在两个布局碎片接近时优先选择更早epoch和更少边的方案。

## 23. ring消费与完整依赖图构建

### 23.1 图构建顺序

cache模拟结束后，按以下顺序生成图，避免边生成过程中反复修改大数组：

1. 为每个residency生成Load/BlockReady和cache slice；
2. 为每个Job生成Dense/CSR scatter groups；
3. 连接BlockReady/PairReady -> scatter；
4. 生成JobDone，连接该Job所有output-producing groups；
5. 生成JobDone bitmap索引和PrefixReleasePlan buckets；
6. 根据extent epoch把later Load加入对应prefix release bucket；
7. 生成OutputSlotReady generation，连接到对应scatter groups；
8. 生成BatchReady，连接JobDone与ring generation发布；
9. 排序、去重普通successors，计算initial dependency counts；
10. 拓扑检查和cache/output interval hazard复核。

### 23.2 JobDone、BatchReady与消费release不同

- `JobDone(b)`：batch b的输出计算已完成；
- `PrefixDone(b)`：0..b所有JobDone已完成，cache epoch可推进；
- `BatchReady(b)`：JobDone且对应ring generation完整可见，可被队头读取；
- `ConsumerReleased(b)`：外部不再持有batch b，ring slot下一generation可写。

不能用ConsumerReleased代替JobDone释放decoded cache，否则慢模型消费会不必要地延长cache residency；也不能用JobDone代替ConsumerReleased覆写output slot，否则零拷贝Batch会被破坏。

### 23.3 next_batch/release协议

`next_batch()`只观察当前`consume_head`：

1. 检查terminal state；
2. 检查slot generation等于consume_head；
3. Acquire读取BatchReady；
4. 返回持有generation lease的Batch；
5. Batch drop/release发布ConsumerReleased；
6. `consume_head += 1`，使更远output scatter可ready。

后续batch即使先完成也只留在ring，不进入无序完成队列。完成通知只在队头或至少一个等待中的相关generation变ready时唤醒，避免每task通知consumer。

### 23.4 外部背压与cache prefetch独立

ConsumerReleased只释放ring generation；PrefixDone只释放cache epoch。LoadTask可以在output ring已满时继续执行到cache依赖允许的远端；ScatterTask必须等待OutputSlotReady。这样cache长期保持高命中窗口，而ring只保存近期可交付dense output。

## 24. InitializeJob

### 24.1 定义

Plan额外包含一个不对应任何Batch的InitializeJob：

```rust
struct InitializeJob {
    io_tasks: Range<usize>,
    decoded_bytes: usize,
    io_bytes: usize,
}
```

它包含cache编译初始贪婪填充阶段生成、`available_after_batch = INITIAL`的全部纯IoDecodeLoadTasks。它不包含Dense/CSR scatter、output fill、JobDone或ring写入。

初始load即使最早消费者是很远的普通Job，也归InitializeJob；普通Job只保留对BlockReady的依赖。这样第一个Batch不再承担整个初始cache预热的串行负担。

### 24.2 独立临时线程池

InitializeJob不进入通用ready queue，也不由常驻workers判断特殊状态：

```text
Plan.open
  -> bind sources / allocate cache and ring / lower init pointers
  -> create temporary initialize pool
  -> parallel execute initialize io+decode ranges
  -> join and destroy initialize threads
  -> publish InitializeDone / seed regular dependency counters
  -> create ordinary worker pool
  -> return running Session
```

初始化线程与后续worker不共享thread objects、local deque、io_uring ring或scratch。普通worker hot loop因此没有`if initializing`分支。

### 24.3 初始化并发与资源上限

不能一task一thread。InitializeConfig至少包含：

```text
initialize_workers
initialize_inflight_io_ops
initialize_inflight_encoded_bytes
initialize_decoded_bytes（来自静态cache布局）
```

init tasks按`earliest_consumer_batch`优先，再按source/file offset分组。每个init worker持有独立encoded staging和DecodeWorkspace。第一版可以使用blocking positioned read；后续若初始化I/O明显受限，再为临时线程各建独立io_uring，而不是复用普通worker ring。

### 24.4 同步、错误与可见性

- `Plan.open`在Python侧释放GIL；
- init pool任一I/O/decode/validation错误设置first error并停止领取新init task；
- 所有已启动init线程必须join；
- 失败时不创建普通worker、不返回可用Session；
- 成功join建立cache写入可见性，再初始化regular dependency state；
- InitializeDone对应的BlockReady nodes直接标记完成，普通scatter/load successors按正常依赖入队。

初始化是同步的Session startup阶段，可能读取并解压接近整个24/48 GiB cache。这是明确的启动延迟换稳态命中率策略；stats必须单独报告initialize wall time、I/O、decode bytes和worker scaling。

### 24.5 序列化

InitializeJob及其IoDecodeLoadTask/DecodeOps属于Plan核心，完整写入PlanImage。反序列化decode不执行初始化；bind只恢复source和decoder；每次新的Session.open都为自己的cache arena重新执行InitializeJob，不能序列化已填充的cache内容或跨Session共享cache pointer。

## 25. 细化后的实现顺序

1. 先定义无pointer、`usize/u64`字段的Plan IR和InitializeJob/Job task ranges。
2. 实现PlanImage section writer/reader、limits和无I/O PlanTemplate roundtrip。
3. 实现SourceLocator/Manifest、严格bind和Python lazy Plan/load/save/pickle。
4. 将cache compiler改为`available_after_batch` epoch，并加入依赖感知Best-Fit score。
5. 生成IndependentBlockLoads并执行最终I/O fusion pass。
6. 生成JobDone、PrefixReleasePlan、BlockReady、OutputSlotReady控制结构并做拓扑/alias验证。
7. 实现InitializeJob独立blocking线程池。
8. 实现普通blocking ready-task runtime和ring消费。
9. 接入io_uring、shared ring和性能优化。

序列化应在runtime切换前完成：它会迫使Plan IR去除隐藏指针、函数地址和不可稳定字段，避免先写出只能在当前进程工作的执行结构后再返工。

## 26. cache 编译后的最终 I/O fusion pass

### 26.1 定位与顺序

I/O merge是最后一个会改变可执行task拓扑的语义优化pass。它必须位于cache编译之后、dependency graph和连续arena最终定型之前：

```text
cell -> block resolution
  -> batch requirements
  -> cache residency / refcount / Best-Fit / availability epoch
  -> independent BlockLoad + Scatter tasks
  -> I/O fusion pass
  -> BlockReady/JobDone/PrefixRelease/ring graph finalization
  -> task arena flatten + PlanImage
  -> Session copy/reorder/pointer lowering
```

cache编译前不能合并I/O，因为当时还不知道哪些block是hit、哪些residency需要reload、target cache offset、available epoch和owner Job。I/O fusion之后不能再改变cache residency、cache offset或batch requirement。

### 26.2 输入：IndependentBlockLoad

cache compiler先为每个实际residency生成独立load候选：

```rust
struct IndependentBlockLoad {
    owner: LoadOwner,              // InitializeJob or one regular Job
    source: usize,
    logical_key: usize,
    encoded_range: Range<u64>,
    decoded_len: usize,
    decoder: usize,
    cache_target: CacheSlice,
    available_after_batch: AvailabilityEpoch,
    earliest_consumer_batch: u64,
    block_ready_node: usize,
}
```

每个候选代表一次真实cache residency load，而不是unique on-disk block。一个block被驱逐后重载，会产生新的候选、target generation和BlockReady node。

### 26.3 严格 compatibility key

第一版只有compatibility key完全相同的候选才能进入同一fusion bucket：

```text
(
  owner,                     // InitializeJob或完全相同的regular Job
  source,
  logical_key/read view,
  available_after_batch,
  priority_bucket,
  backend_read_class,
)
```

具体约束：

- 不能跨fd、Directory chunk file、ZIP logical entry或ByteStore key合并；
- 不能跨availability epoch，否则早期load会等待更晚PrefixDone；
- regular load第一版不能跨owner Job，避免一个物理task同时进入多个Job task range；
- InitializeJob内部可以跨earliest consumer batch，但只能在同一priority bucket内合并；
- 不能跨positioned/range-key/whole-key backend class；
- source override在bind时必须保持backend read class，否则Strict bind拒绝或显式重新编译I/O pass。

这组限制比理论最大I/O合并更保守，但保证fusion是局部等价变换，不改变batch优先级、cache依赖和Job completion语义。

### 26.4 backend read class

#### Positioned

Directory文件、Stored ZIP entry或其他稳定positioned view。可以对同一view内的连续range做一次`pread/io_uring Read`。

#### RangeKey

支持有效`read_range_into`的ByteStore key。只能在同一key内部合并；offset相对于key起点。

#### WholeKey

Deflated ZIP等不支持高效range read的entry。一个task必须物化整个logical key，再从中执行多个DecodeOps：

- `file_offset = 0`；
- `file_len = declared whole-key len`；
- 不与其他key合并；
- 同一compatibility bucket内的required blocks应共享一次whole-key read；
- whole key是不可分割I/O单元，可以超过soft merge limit，但仍受hard encoded/staging limit；
- 若同一key在不同epoch重新加载，仍会再次物化，不引入额外encoded cache。

#### Empty

不生成I/O task，只生成必要的empty/fill scatter语义。

### 26.5 fusion输出

一个fusion group生成：

```rust
struct IoDecodeLoadTask {
    owner: LoadOwner,
    source: usize,
    file_offset: u64,
    file_len: usize,
    decode_ops: Range<usize>,
    available_after_batch: AvailabilityEpoch,
    earliest_consumer_batch: u64,
}

struct DecodeOp {
    encoded_offset: usize, // relative to this task's input buffer
    encoded_len: usize,
    decoder: usize,
    cache_target: CacheSlice,
    block_ready_node: usize,
}
```

`DecodeOp.encoded_offset`必须重新减去merged physical range start。DecodeOps按`(earliest_consumer_batch, original_encoded_offset)`排列，并在每个block成功后单独发布BlockReady；不能等待整个group完成后统一发布。

### 26.6 hard feasibility limits

任意fusion group必须同时满足：

```text
span_bytes             <= max_coalesced_io_bytes
gap_bytes              <= max_io_gap_bytes
span_bytes/payload     <= max_io_amplification_ratio
decode_ops             <= max_decode_ops_per_io_task
sum(decoded_bytes)     <= max_decoded_bytes_per_io_task
span_bytes             <= max_encoded_staging_bytes_per_task
```

并继续受全局ResourceLimits限制：

- 单个不可分割block/WholeKey超过soft limit时可以单独成task；
- 超过hard encoded/decoded/staging limit直接compile失败；
- 所有加法、range union和ratio计算检查overflow和finite value。

这些上限同时保护临时buffer、单task尾延迟和解压并行度。即使I/O成本模型认为一个超大group有利，也不能突破。

### 26.7 三种 policy

```rust
enum IoMergePolicy {
    Off,
    Adjacent,
    CostAware,
}
```

#### Off

每个residency block独立I/O，用于正确性对照、profile和特殊低延迟环境。

#### Adjacent

只合并overlap或严格相邻range，`gap_bytes == 0`。在hard limits内形成最大连续group，不增加读取字节。建议作为第一版默认值。

#### CostAware

允许小gap，使用显式带宽/IOPS模型和有界区间DP选择partition。必须显式配置或在真实profile后启用。

### 26.8 CostAware 成本模型

对一个包含`k`个block的候选group：

```text
payload_bytes = union of required encoded block bytes
span_bytes    = last.end - first.start
gap_bytes     = span_bytes - payload_bytes

separate_io_seconds = k / io_operations_per_second
                    + payload_bytes / io_bandwidth_bytes_per_second

merged_io_seconds   = 1 / io_operations_per_second
                    + (span_bytes + io_merge_delta_bytes)
                      / io_bandwidth_bytes_per_second
```

只有`merged_io_seconds < separate_io_seconds`且hard limits满足时，group才有资格合并。`io_merge_delta_bytes`是对merged read额外收取的保守不确定性成本，使边界收益不会触发合并；它不改变实际读取范围。

第一版不试图用不可靠的模型估计解压并行损失，而使用`max_decode_ops_per_io_task`和`max_decoded_bytes_per_io_task`硬限制单worker串行decode量。runtime profile稳定后，才考虑加入`decode_bandwidth_bytes_per_second`和critical-path penalty。

### 26.9 有界一维区间 DP

每个compatibility bucket按`encoded_range.start`排序。对有`n`个候选的bucket：

```text
dp[j] = min over feasible i < j:
        dp[i] + group_cost(i..j)
```

其中`group_cost`是一次merged read预测时间。搜索只回看最多`max_decode_ops_per_io_task`个block，并在span/gap/decoded bytes超过上限时立即停止，因此：

```text
time  = O(n * W)
space = O(n)
W <= max_decode_ops_per_io_task
```

确定性tie-break顺序：

1. 更低predicted seconds；
2. 更少gap bytes；
3. 更低read amplification；
4. 在成本相等时保留更多独立task，以保护并行度；
5. 更早split point。

Adjacent policy可以使用同一DP框架令gap上限为0，也可以线性greedy实现；两者必须产生确定性相同语义。

### 26.10 并行度下限与反向拆分

过度合并会减少可并行task。fusion完成后按owner/priority bucket检查：

```text
target_task_floor = io_parallelism_hint * min_tasks_per_worker
```

若某个大bucket合并后的task少于floor，按以下顺序拆分最大group：

1. 最大decoded bytes；
2. 最大DecodeOp count；
3. 最大encoded span；
4. 在block边界选择使两侧predicted cost最平衡的split。

直到达到floor或所有task都已不可再分。InitializeJob单独使用`initialize_parallelism_hint`；regular Jobs使用`regular_io_parallelism_hint`。这两个值是编译提示，不要求Session worker count完全一致，但Plan stats必须记录。

对只有少量physical keys的source，不能为了达到floor重复读取同一个WholeKey；WholeKey保持不可拆分I/O单元。

### 26.11 优先级和tail latency约束

fusion不能改变task的主要优先级：

- regular bucket要求同owner Job，因此天然同earliest batch；
- InitializeJob按priority bucket分组，默认bucket width为1 batch或一个显式小窗口；
- 当前/近期block不能与非常远block形成一个无法抢占的大decode loop；
- worker执行DecodeOps时每完成一个block就检查terminal state，并立即发布对应BlockReady；
- 不在DecodeOps中间抢占同一个I/O task，避免encoded staging生命周期复杂化；通过group size上限控制最长不可抢占时间。

### 26.12 与 dependency graph 的重写

fusion前：

```text
PrefixRelease -> IndependentLoad A -> BlockReady A
PrefixRelease -> IndependentLoad B -> BlockReady B
```

fusion后：

```text
PrefixRelease -> IoDecodeLoadTask(A, B)
                     |- DecodeOp A -> BlockReady A
                     `- DecodeOp B -> BlockReady B
```

因为compatibility key要求相同availability epoch，merged task只有一个prefix dependency。每个BlockReady和下游scatter边保持原样。独立load nodes被删除后，重新计算dependency counts、Job I/O ranges和successor CSR；不能在finalized graph上原地留下dead nodes。

IoDecodeLoadTask整体完成只用于task生命周期、错误统计和staging回收；JobDone仍由output-producing scatter groups决定，不能因为task包含远端DecodeOp而额外阻塞当前batch完成。

### 26.13 与序列化/bind/lowering 的关系

PlanImage保存fusion后的Logical IoDecodeLoadTasks和DecodeOps，因此反序列化不重新运行cache或I/O fusion compiler。SourceManifest保存backend read class和logical key合同。

Strict bind重新解析Directory/ZIP entry：

- logical key、declared length、decoder prefix必须匹配；
- backend read class必须兼容；
- ZIP物理base offset重新解析；
- RuntimeIoDecodeLoadTask重新计算fd/base pointer，但保持logical merged range。

ExecutionPlan copy-and-lower可以重新排列tasks和DecodeOps以改善局部性，但不能改变fusion group边界；改变group边界属于重新编译LogicalPlan。

### 26.14 Python 配置接口

Python负责用户可见策略，Rust只接收规范化机械参数：

```python
@dataclass(frozen=True)
class IoMergeConfig:
    policy: Literal["off", "adjacent", "cost"] = "adjacent"
    max_coalesced_io_bytes: int = 32 * MiB
    max_io_gap_bytes: int = 0
    max_io_amplification_ratio: float = 1.0
    max_decode_ops_per_io_task: int = 64
    max_decoded_bytes_per_io_task: int = 64 * MiB
    max_encoded_staging_bytes_per_task: int = 32 * MiB
    io_bandwidth_bytes_per_second: float = 8 * GiB
    io_operations_per_second: float = 100_000
    io_merge_delta_bytes: int = 4096
    initialize_parallelism_hint: int = 32
    regular_io_parallelism_hint: int = 32
    min_tasks_per_worker: int = 2
```

配置校验：

- 所有bytes/count/parallelism为正，`max_io_gap_bytes`可为0；
- bandwidth/IOPS/amplification为finite positive；
- Adjacent强制有效gap=0和amplification=1，不因用户填写更宽值而偷偷读取gap；
- CostAware才使用gap、bandwidth/IOPS和delta参数；
- Off忽略soft merge参数但仍校验hard resource limits；
- `max_coalesced_io_bytes <= max_encoded_staging_bytes_per_task`，除非明确允许不可分割WholeKey例外；
- 参数和最终resolved值进入PlanImage与PlanStats。

为了避免PlanConfig继续平铺膨胀，Python使用嵌套`IoMergeConfig`，PyO3转换为一个私有Rust `IoMergeOptions`。Rust不负责“auto guessing”用户机器策略。

迁移期间现有`PlanConfig.io_bandwidth_bytes_per_second`、`io_operations_per_second`、`max_coalesced_io_bytes`等flat字段可以由Python兼容层映射到IoMergeConfig，但native新compiler只接收resolved IoMergeOptions。重复在flat和nested位置显式提供且值冲突时，Python直接报错，不能静默选择一个。

### 26.15 建议默认与演进

第一版默认：

```text
policy                         = Adjacent
max_io_gap_bytes               = 0
max_io_amplification_ratio     = 1.0
max_coalesced_io_bytes         = 32 MiB
max_decode_ops_per_io_task     = 64
max_decoded_bytes_per_io_task  = 64 MiB
min_tasks_per_worker           = 2
```

演进顺序：

1. Off与Adjacent做数值/依赖图等价测试；
2. 真实24/48 GiB cache上比较I/O ops、bytes、decode并行度和batch latency；
3. 只有profile显示IOPS/syscall占比显著时启用CostAware小gap；
4. 不跨availability epoch和regular owner Job；
5. 若未来需要跨Job融合，先设计task多owner和completion语义，不能只放宽一个配置值。

### 26.16 统计与验证

Compile stats至少区分Initialize/Regular并报告：

- independent block loads；
- fused I/O tasks；
- predicted I/O ops saved；
- payload/span/gap bytes；
- read amplification；
- DecodeOps per task p50/p95/max；
- decoded bytes per task p50/p95/max；
- task floor触发和反向拆分次数；
- 因source/key/epoch/owner/priority/hard limit/unprofitable而拒绝的merge数；
- predicted I/O seconds before/after；
- fusion pass wall time和peak working set。

必须验证：

1. Off/Adjacent/CostAware数值输出完全一致；
2. 每个residency恰好由一个DecodeOp生产；
3. 每个DecodeOp input range位于merged staging内；
4. 每个cache target range和generation保持不变；
5. fusion前后BlockReady下游集合一致；
6. short read、EOF、decoder error和cancel不发布未完成block；
7. Directory、Stored ZIP、Deflated ZIP分别覆盖；
8. fusion不能增加跨epoch依赖或改变JobDone/BatchReady语义；
9. InitializeJob仍保留足够task并行度；
10. runtime benchmark同时报告I/O收益和decode/scatter是否退化。
