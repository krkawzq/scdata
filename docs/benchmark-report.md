# scdata Benchmark 报告与性能优化指南

> 本文档面向**负责性能优化的 agent / 工程师**。汇总当前所有 benchmark 结果,
> 定位已确认的瓶颈、给出源码引用、可复现命令与优化建议。所有数据来自
> `rust/scdata/benches/modules.rs`(轻量单元 bench)与 `rust/scdata/benches/stress.rs`
> (多线程压力 bench),在 CPU 机器上实测。

> **更新记录(2026-06-28)**:access 调度已从单线程重构为 **sharded 多线程**
> (`scheduler_shards`),原 P0-2(access 调度饱和)已关闭。zstd 已复核为
> **与 numcodecs 同档的算法上限**,不是 Rust codecs 的额外瓶颈。DataBank 已增加
> 自有 compute pool,并补齐 Dense1D 连续内存 fastpath、CSR single memory chunk fastpath
> 与 CSR memory multi-chunk direct scatter fastpath;**CSR/Dense1D 主路径以及 no-block
> multi-chunk CSR 压力测试均已通过 32k cells/s 与 20G/s 验收**。新增
> scheduled prefetch cells 全链路压力测试,可配置大文件、chunk miss 概率、每 batch 命中 chunk 数、
> databank/access prefetch step 和线程池规模。

---

## 状态摘要

| 目标 | 状态 | 数据 |
|---|---|---|
| **解压带宽 20G** | lz4/blosc ✅ / zstd 算法上限 | zstd t16=7.7G,lz4=23G,blosc=40G |
| **调度层带宽 20G** | ✅ 够(轻负载) | scheduled t16=24.2G,prefetch t8=24.2G,send evict t16=17.2G |
| **32k IOPS(kops)** | ✅ 富余 | send evict t16=275k(8.6×),scheduled t16=387k chunks/s |
| **databank no-block 压测** | ✅ 主格式已过线 | Dense1D=42.4G;CSR single=48.0/51.3G;CSR multi-chunk repeat=50.0/54.2G;scattered=50.2/51.5G |
| **scheduled prefetch 全链路** | ✅ bench 已补 / 继续隔离随机 IO | H200 256GiB raw lz4: threaded48/64 随机 encoded IO≈1.1GiB/s;重排 batch 可到 67.9k cells/s |
| **blocking direct reference** | 仅参考 | lz4 file repeat=8.3/7.7G,scattered=459/451M;未开启 prefetch,不能作为模块吞吐验收 |

**压力测试口径**:模块压力测试必须分层。codecs、decode_pool、iopool、access 分别测各自容量;
databank 压测使用 no-block 场景,假设下游没有等待 IO/decode/prefetch 阻塞,只测 databank
自己的 plan/group/project/scatter 开销。未开启 prefetch 的 file+compressed direct access
是 blocking reference,只能说明同步等待代价,不能作为 databank 吞吐验收。

**下一步**:CSR/Dense1D 与 no-block multi-chunk CSR 已达标。Dense2D 多 chunk 行访问天然会产生
更多 chunk miss 和小片段 scatter,保留兼容即可;主生产格式按 CSR 存储。scheduled prefetch
全链路 bench 已能测大文件/冷读/不同 chunk 命中分布,并拆分 full-chunk requested、miss chunk、
CSR slice 与 dense output 字节口径;同时增加消费端 `next()` 等待分位数与可选 background CPU/IO
干扰场景。后续需要在 IO 更强或更稳定的机器上继续复核 20GiB/s 真实 IO。

---

## 0. 文档导航

- [1. 环境与方法](#1-环境与方法)
- [2. 测试矩阵总览](#2-测试矩阵总览)
- [3. codecs 模块](#3-codecs-模块)
- [4. decode_pool 模块](#4-decode_pool-模块)
- [5. iopool 模块](#5-iopool-模块)
- [6. access 模块](#6-access-模块)
- [7. databank 模块](#7-databank-模块)
- [8. 已确认的瓶颈与优化建议](#8-已确认的瓶颈与优化建议)
- [9. 待补充的 benchmark](#9-待补充的-benchmark)
- [10. 可复现命令](#10-可复现命令)

---

## 1. 环境与方法

### 1.1 硬件 / 运行环境

| 项 | 值 |
|---|---|
| 机型 | rjob pod(CPU 机器) |
| CPU | INTEL XEON PLATINUM 8558,2 socket × 48 核 = 96 核 |
| 容器 cpuset | `80-95,176-191`(32 核,全在 NUMA node1) |
| NUMA | 2 node;容器内 CPU 单 node,内存 `Mems_allowed_list: 0-1` |
| 内存 | 195 Gi |
| `/tmp` | tmpfs 196G |
| 存储 | 共享存储(NFS 挂载),源码在 `/mnt/shared-storage-user/dnacoding/wangzhongqi/Code/Project/scdata` |
| 工具链 | cargo 1.95.0,`[profile.release]` 被 workspace 警告忽略(实际用 cargo 默认 release,opt-level=3,无 LTO) |

**重要环境限制**:
- 容器内 `numactl` 未安装,`taskset` 改亲和性被 cgroup 拒绝(`Invalid argument`)。
- IO bench 数据**落在 NFS 共享存储**,page cache 严重高估磁盘真实带宽
  (1M 顺序读到 ~52 GiB/s 是 cache 命中,非磁盘能力)。用户已声明"真实 IO 已测够",
  故本报告 IO 数据仅反映**调度层**能力,不反映磁盘上限。
- 无 `perf`,无法做 cache-miss / 指令级 profiling。

### 1.2 Bench 框架

自研轻量 harness(`benches/support/mod.rs:22` `bench` 函数):
- `SCDATA_BENCH_ITERS=<n>` 覆盖每项迭代数;`SCDATA_BENCH_WARMUPS=<n>` 预热次数(默认 3)。
- 单线程项输出 `iter / elapsed_s / ns_iter / throughput_mib_s`。
- 多线程项(`stress.rs` 的 `stress_mt`,`benches/stress.rs:59`)输出
  `threads / ops / elapsed_s / ns_op / throughput_mib_s / kops`。
- 多线程用 `Barrier` 同步起跑,每线程固定 `iters_per_thread = total_iters / threads`,
  结果 `xor` 校验防优化。

### 1.3 数据口径

- 带宽单位 **MiB/s**(1 MiB = 1024² 字节);20G 目标按 **20 GiB/s** 理解。
- IOPS 单位 **kops**(千 ops/s)。
- payload 为 `support::payload()` 生成的伪随机字节(确定性,可复现)。
- **模块压力测试分层计数**:
  - codecs/decode_pool/iopool/access 各自单独测试,用于确认下游组件容量。
  - databank no-block 压测使用 `Memory + Uncompressed/identity + multi-chunk`,绕开
    IO/decode 阻塞,但保留 databank 的 batch plan、chunk grouping、projection/scatter 开销。
  - `blocking_*_file_*` bench 是未开启 prefetch 的同步 direct access reference,会把每次调用等待
    IO/decode 的时间算进去;它不代表 scheduled/prefetch 场景下的模块吞吐上限。
  - `databank/fullchain_scheduled_prefetch` 是全链路压力测试,用 `FileOffset` CSR 文件 +
    `DataBank::prefetch_cells_scheduled`。输出给 `output_mib_s`、`slice_mib_s`、
    `encoded_file_mib` 和 `wait_ns_*`。20G IO 目标应结合 `encoded_file_mib`、冷读设置和
    外部物理 IO 计数解释,不是只看 output buffer 吞吐。

---

## 2. 测试矩阵总览

| 模块 | bench 文件 | 项数 | 覆盖维度 |
|---|---|---|---|
| codecs | modules.rs + stress.rs | ~30 | 9 种 codec × decode_into/decode_to_vec/decode/pipeline × 1M/64k/4k × ST + MT(2/4/8/16) |
| decode_pool | modules.rs + stress.rs | ~10 | submit/try_submit/submit_async × owned_buffer/zeroed × batch32x × MT |
| iopool | modules.rs + stress.rs | ~40 | read(4k/64k/1m) × 1q/4q sharding × write/fsync/sync_data/truncate/metadata/dedup/uring/register-unregister × ST+MT |
| access | modules.rs + stress.rs | ~20 | send/scheduled/prefetch × keep/evict × cache hit/miss × lookahead × MT(sharded=4) |
| databank | modules.rs + stress.rs | ~25 | Dense1D/2D/SparseCsr × Memory/FileOffset/Directory × no-block/blocking/full-chain scheduled × checked/unchecked × MT instances |

**总计 ~115 个测试点**,ST 基线 + MT 扩展性(2/4/8/16 线程)。

---

## 3. codecs 模块

源码:`src/codecs/spec.rs`(2110 行)、`src/codecs/pool.rs`(424 行)。

### 3.1 单核解码带宽(1 MiB chunk,`decode_into`)

| codec | ns/iter | MiB/s | 备注 |
|---|---|---|---|
| none (uncompressed) | 45354 | **22049** | memcpy 上界 |
| blosc lz4 + shuffle | 175897 | **5685** | 手写快路径 |
| blosc lz4 memcpyed | 56768 | 17616 | 无 shuffle,接近 memcpy |
| blosc lz4 noshuffle | 339889 | 2942 | |
| lz4 | 300270 | **3330** | |
| crc32 | 95285 | 10495 | 校验和 filter |
| zstd | 1057688 | **946** | 与 numcodecs 同档,zstd 算法上限 |
| pipeline crc32\|zstd | 1069017 | 935 | ≈ zstd |
| zlib / gzip | ~4.6M | ~217 | flate2 流式 |
| bz2 | 30.8M | 33 | |
| lzma (xz) | 23.0M | 44 | |

### 3.2 多核聚合带宽(1 MiB chunk,`stress.rs`,最新)

| codec | ST | t2 | t4 | t8 | t16 | 饱和点 |
|---|---|---|---|---|---|---|
| blosc lz4 | 6.6G | 10.2G | 20.8G | **40.3G** | 36.0G | t8(内存带宽饱和) |
| lz4 | 3.6G | 5.9G | 11.8G | 22.5G | **23.0G** | t16 仍在升 |
| zstd | 993M | 1.7G | 3.8G | 6.5G | **7.7G** | **t8 后近饱和** |
| none | ~20G | 33.9G | 64.3G | **108G** | 62.9G | t8 峰值(内存带宽) |
| zstd decode_to_vec | — | 1.8G | 3.7G | 6.5G | 6.5G | t8 饱和 |

> 注:zstd t16 在 7.7–8.4G 间波动(运行间噪声),结论不变:8 核后近饱和,远低于 20G。
> 这与 numcodecs 对齐,不是 Rust codecs 的额外性能问题。

### 3.3 关键结论

1. **zstd 多核到不了 20G,但不是 Rust 实现问题**:t16 仅 7.7G,t8→t16 提升约 19%(6.5→7.7),
   8 核后接近饱和。对齐 numcodecs 后确认这是 zstd 解码本身的算法/硬件上限。
   要 20G zstd 解压带宽,需要更高单核频率、更多有效核心,或改用更快 codec。
2. **lz4 / blosc 够覆盖 20G**:lz4 t16=23.0G,blosc t8=40.3G。
3. **blosc 手写快路径有效**:单核 6620 MiB/s,远高于 zstd(993)和 lz4(3596)。
4. **none 在 t8 达 108G 后回落** —— 内存带宽饱和后的噪音,属正常。

---

## 4. decode_pool 模块

源码:`src/codecs/pool.rs`。flume 有界队列 + N 个 OS 线程(`decode-wrk-{i}`),
`catch_unwind` 包裹防 panic 杀 worker。

### 4.1 单核 / 批量

| bench | ns/iter | MiB/s | 备注 |
|---|---|---|---|
| ST batch 32x zstd 64k | 566799 | 3529 | 32 个 request 串行 submit+recv |
| ST submit none 64k | 35206 | 1775 | 无 codec,纯调度开销 |
| ST submit zstd 64k | 79593 | 785 | |
| ST submit zstd 64k owned_buffer | 127357 | 491 | ← 见 4.3 |
| ST submit zstd 64k owned_zeroed | 145333 | 434 | 零填 buffer |
| ST submit none 64k owned_buffer | 34963 | 1788 | ← None 无惩罚 |

### 4.2 多线程(zstd 64k,4 workers)

| bench | MiB/s | kops |
|---|---|---|
| MT submit t2 | 1621 | 25.9 |
| MT submit t4 | 3267 | 52.3 |
| MT submit t8 | 3448 | 55.2 |
| MT submit t16 | 3506 | 56.1 |
| MT try_submit t4 | 3176 | 50.8 |
| MT try_submit t8 | 3453 | 55.2 |

**t8 后饱和**(~3.5G),瓶颈是 4 个 worker 线程的 CPU。

### 4.3 owned_buffer 反慢现象(已排查,非 bug)

**现象**:`decode_pool_submit_zstd_64k_owned_buffer`(492 MiB/s)比不带 buffer 的(785)慢 ~50μs。

**排查链**(逐项排除):
| 假设 | 实验 | 结果 |
|---|---|---|
| codec 实现差异 | direct 调 `decode`/`decode_to_vec`/`decode_into`(不经 pool) | 排除(四入口 1031–1088 MiB/s,差 ≤6%) |
| zstd_safe `WriteBuf` 差异 | 读 `zstd-safe-7.2.4/src/lib.rs:1672-1745` | 排除(`WriteBuf for [u8]` vs `for Vec` 的 C 调用 `ZSTD_decompress` 相同) |
| 缺页 | `vec![0u8]` 零填版 | 排除(零填不快) |
| 通用 pool + 跨线程 buffer | None codec + owned | 排除(无惩罚,1788 vs 1782) |
| 跨 NUMA | 查 cpuset | 排除(容器单 NUMA CPU) |
| worker 调度迁移 | pinned 绑核 | 排除(惩罚不变) |
| 数据量放大 | 1M owned vs 1M decode | **反转**(1M 时 owned 略快 949 vs 936,惩罚消失) |

**定位**:惩罚只在 `zstd + with_output_buffer + pool` 组合出现(64K 慢、1M 不慢的非单调)。
最可能根因是 zstd 解码非顺序写入对跨线程 caller buffer 的 cache 敏感性,
但容器内无 perf 无法验证到指令级。

**影响**:**不影响生产**。`DecodePoolBackend::submit_decode`(`src/databank/adapter.rs:57-61`)
构造请求时**不用** `with_output_buffer`,走 `codec.decode` 路径(worker 内部分配 Vec)。
此 50μs 惩罚仅是 bench 测量现象。bench 中保留了诊断用例作为回归基线。

---

## 5. iopool 模块

源码:`src/iopool/mod.rs`(2493 行)、`src/iopool/threaded.rs`、`src/iopool/uring.rs`(880 行)。
io_uring 与线程池双后端,优先级队列 + 读去重 + barrier 排序 + sharding + fast-read 旁路。

### 5.1 读带宽与读大小缩放(threaded,NFS page cache)

| read size | ST 1q | MT t16 1q | ST 4q | MT t16 4q |
|---|---|---|---|---|
| 4k | 155 MiB/s | 1055 MiB/s / **270k IOPS** | 196 MiB/s | 1237 MiB/s / **317k IOPS** |
| 64k | 2106 MiB/s | 17333 MiB/s | 2599 MiB/s | 22631 MiB/s |
| 1m | 8696 MiB/s | **51811 MiB/s** | 8788 MiB/s | 46539 MiB/s |

> 注:1M 读 52G 是 NFS page cache 命中,**非磁盘真实带宽**。用户已声明真实 IO 够。

### 5.2 多线程扩展性(64k read)

| threads | 1q MiB/s | 4q MiB/s | 4q IOPS |
|---|---|---|---|
| 2 | 4921 | 4870 | 78k |
| 4 | 11328 | 10712 | 171k |
| 8 | 16590 | **22845** | 366k |
| 16 | 17333 | 22631 | 362k |

**4q sharding 对小 IO 锁竞争有效**:4k read t8,1q=256k IOPS vs 4q=**450k IOPS**(下表)。

| 4k read t8 | 1q | 4q |
|---|---|---|
| MiB/s | 1001 | 1758 |
| IOPS | 256k | **450k** |

### 5.3 其他操作

| 操作 | ST / MT | 吞吐 |
|---|---|---|
| write 64k (MT t4) | 1q=1900 / 4q=1828 MiB/s | 30k ops |
| fsync (MT t4) | 1q=40.6k / 4q=90k ops | — |
| sync_data (MT t4) | 1q=41.2k / 4q=111k ops | — |
| metadata (ST) | 1q=22μs / 4q=16μs | — |
| **dedup x8 read** (MT t8) | **61 GiB/s** / 123k ops | 去重极有效 |
| register/unregister cycle (ST) | 6.5μs | 154k ops |

### 5.4 io_uring vs threaded(64k read)

| | ST | t2 | t4 | t8 | t16 |
|---|---|---|---|---|---|
| uring | 1512 MiB/s | 3713 | 4451 | 4014 | 4062 |
| threaded (1q) | 2106 MiB/s | 4921 | 11328 | 16590 | 17333 |

**uring 在 NFS 下不如 threaded**,且 t4 后不扩展。NFS 网络文件系统对 io_uring 无优势,
submission 开销反而拖累。换本地 NVMe 可能反转。

### 5.5 关键结论

- **iopool 调度层不是瓶颈**:1M=52G、4k=450k IOPS、dedup=61G,扩展性好。
- **sharding 对小 IO 高并发有效**(4k t8:1q→4q IOPS 翻倍)。
- **io_uring 在 NFS 下无优势**,生产若用 NFS 建议 threaded 后端。

---

## 6. access 模块

源码:`src/access/scheduler.rs`(**3386 行**,核心)、`src/access/cache.rs`、`src/access/cpu.rs` 等。

### 6.0 架构(已修复:sharded 多线程调度)

> **变更**:原单线程 `current-thread runtime + LocalSet` 已重构为 **sharded scheduler**。
> `AccessConfig.scheduler_shards`(默认 1)创建 N 个独立 runtime 线程,请求按 `ChunkKey`
> 哈希分片(`scheduler.rs:402` `shard_for_key`),每 shard 独享一份 cache/内存预算
> (`scheduler.rs:230` `for_shard` / `split_budget`)。同 chunk 的去重仍在 shard 内保证。
>
> **效果**:调度编排从单核扩展到 N 核,t2 即饱和的瓶颈消除,改为线性扩展(见 6.2)。
>
> **stress bench 配置**:`scheduler_shards: 4`(容器 32 核)。生产可按核数调。

### 6.1 单核基线(scheduler_shards: 4)

| bench | ns/iter | MiB/s | 备注 |
|---|---|---|---|
| send_read_decode_miss_64k (keep) | 53384 | 1171 | 每次不同 offset,cache miss |
| send_read_decode_miss_64k (evict) | 21358 | 2926 | evict 更快(无 decoded cache 维护) |
| scheduled_read_decode_16x64k (keep) | 265534 | 3766 | 批量流水线,16 chunk |
| scheduled_read_decode_16x64k (evict) | 300861 | 3324 | |
| prefetch_cold_64k | ~15μs | 4075 | fire-and-forget |

### 6.2 多线程(64k,scheduler_shards: 4)

| 路径 | ST | t2 | t4 | t8 | t16 | 趋势 |
|---|---|---|---|---|---|---|
| send evict | 2.9G | 6.2G | 11.1G | 17.0G | **17.2G** | 近线性,t16 微饱和 |
| send keep | 1.2G | 2.4G | 4.6G | 7.5G | **10.1G** | 线性,仍在升 |
| scheduled (keep) | 3.8G | 8.0G | 14.3G | 20.3G | **24.2G** | 线性,已超 20G |
| scheduled (evict) | 3.3G | 6.9G | 11.0G | 14.0G | — | 线性(t16 未测) |
| prefetch (keep) | — | 8.0G | 15.0G | **24.2G** | — | 线性,已超 20G |
| prefetch (evict) | — | 8.3G | 15.8G | **21.6G** | — | 线性,已超 20G |

### 6.3 修复前后对比(`scheduler_shards: 1` → `4`)

| 路径 | 修复前 t16 | 修复后 t16 | 提升 | 修复前饱和点 | 修复后 |
|---|---|---|---|---|---|
| send evict | 5.9G | **17.2G** | 2.9× | t2 饱和 | t16 微饱和 |
| send keep | 3.5G | **10.1G** | 2.9× | t8 饱和 | t16 仍在升 |
| scheduled keep | 7.2G(t8) | **24.2G** | 3.4× | t2 饱和 | 线性,超 20G |
| scheduled evict | 6.7G(t8) | 14.0G(t8) | 2.1× | t2 饱和 | t8 仍在升 |
| prefetch keep | 20G(t8) | 24.2G(t8) | 1.2× | — | 线性 |

**调度瓶颈已解决**:4 个 sharded runtime 打破单核编排上限,t2 饱和 → t16 线性扩展。

### 6.4 kops 维度(修复后)

| 路径 | ST kops | MT t8 kops | MT t16 kops |
|---|---|---|---|
| send evict | ~48 | 272 | **275** |
| send keep | ~19 | 120 | 161 |
| scheduled (16 chunks/op) | ~3.8 ops/s(=60k chunks/s) | 20.3 ops/s(=325k chunks/s) | **24.2 ops/s(=387k chunks/s)** |
| prefetch (keep) | ~65 | 388 | — |

**kops 远超 32k 目标**(send evict t16 = 275k,富余 8.6×;scheduled t16 = 387k chunks/s)。

### 6.5 关键结论

- **access 调度修复成功** ✅:从 t2 饱和改为线性扩展,kops 275k >> 32k。
- **调度层带宽够 20G** ✅:scheduled t16 = **24.2G**(已补测,线性扩展超 20G),
  prefetch t8 = 24.2G,send evict t16 = 17.2G。**调度层不再是瓶颈。**
- **scatter 计算的分工(重要)**:access 层的 slice 物化(`cpu.rs:123` `materialize`)
  走 **AccessCpuPool**(每 shard 4 worker,sharded=4 → 16 线程,多线程 ✅);
  databank 层的最终 scatter/project 由 **DataBankComputePool** 和格式专用 direct fastpath
  分担。CSR/Dense1D 主路径已在 no-block 压测中达标 —— 见 §7.4。
- **Caveat**:bench backend 是 `SliceIo`(内存切片)+ None codec,access 层负载太轻。
  若业务强制 zstd,端到端上限会由 zstd 解码本身决定(见 §3.2,zstd 多核仅 7.7G);
  这不再视为 Rust codecs 待修问题。
- **可调优**:`scheduler_shards` 当前 4,容器 32 核有空间,可试 8 进一步提升调度层富余度。

---

## 7. databank 模块

源码:`src/databank/mod.rs`、`src/databank/batch.rs`、`src/databank/compute.rs`、
`src/databank/array.rs`、`src/databank/dataset.rs` 等。
`DataBank` 是 facade,聚合 `IoPool` + `DecodePool` + `AccessScheduler` + `DatasetRegistry` +
`GeneInterner` + DataBank compute pool。compute pool 负责 databank 侧 plan/scatter/project
等 CPU 工作,不同调用方可共享同一个 `DataBank` 并阻塞等待自己的调用完成。

### 7.1 单核端到端(32 cells × 1024 genes)

| 场景 | MiB/s | 备注 |
|---|---|---|
| dense2d mem_unc checked | 3921 | 兼容路径,多列 chunk 小片段 scatter |
| dense2d mem_unc unchecked | 3823 | dense unchecked 回落到 checked |
| dense2d mem_zstd checked | 154 | **zstd 解码主导** |
| dense2d mem_zstd unchecked | 154 | |
| dense2d file_unc checked | 63.3 | NFS/direct blocking IO 主导 |
| dense2d dir_unc checked | 63.0 | 每 chunk 一文件,direct blocking |
| dense1d memory | **42354** | 连续内存 fastpath,约 10.8M cells/s |
| sparse_csr checked | **47954** | single memory chunk fastpath,约 24.6M cells/s |
| sparse_csr unchecked | **51295** | unchecked 热点路径,约 26.3M cells/s |
| sparse_csr no-block repeat checked | **50005** | 64 cells × 4096 genes,multi-chunk memory direct scatter |
| sparse_csr no-block repeat unchecked | **54176** | multi-hit same chunk 场景 |
| sparse_csr no-block scattered checked | **50216** | 每个 cell 基本命中不同 chunk |
| sparse_csr no-block scattered unchecked | **51544** | scattered chunk 场景 |
| sparse_csr blocking lz4 file repeat | 8308/7731 | 未开启 prefetch,blocking reference |
| sparse_csr blocking lz4 file scattered | 459/451 | 未开启 prefetch,大量同步等待 |
| register/unregister cycle | 122μs | 8.2k ops/s |

### 7.2 多实例并行(独立 DataBank,每实例独立 scheduler)

| threads | MiB/s | kops |
|---|---|---|
| 2 | 3277 | 26.2 |
| 4 | 3577 | 28.6 |
| 8 | 3473 | 27.8 |

多实例并行已有改善,但 dense2d 小批仍不作为主吞吐目标。真实部署需要按调用方并发数调
`AccessCpuConfig.num_workers` 与 `FillConfig.num_workers`,避免 access CPU pool、decode pool、
databank compute pool 同时过度订阅。

### 7.3 架构状态:`DataBank` 可跨线程共享

`GeneNameView` 仍是 FFI 友好的 `ptr/len` 视图,但生命周期由 dataset 内部的 `Arc<str>`
保活,已通过手动 `Send/Sync` 边界和测试确认 `DataBank: Send + Sync`。

```rust
// src/databank/interner.rs:6
#[repr(C)]
pub struct GeneNameView {
    pub ptr: *const u8,
    pub len: usize,
}
```

新增回归测试 `databank_is_send_sync` 与 `shared_databank_allows_concurrent_callers`。
对外单次调用仍是阻塞的;不同调用方可共享 `Arc<DataBank>` 并发调用,内部通过 access scheduler
和 databank compute pool 调度。

### 7.4 scatter 计算的分工与当前瓶颈(重要)

access 调度多线程化后,scatter 计算分布在两处:

| scatter 位置 | 线程 | 多线程? | 源码 |
|---|---|---|---|
| access 层 slice 物化(`materialize`) | AccessCpuPool(每 shard 4 worker) | ✅ 是 | `src/access/cpu.rs:123` |
| databank 层最终 scatter/project | DataBankComputePool | ✅ 是(大 batch 阈值触发) | `src/databank/compute.rs`,`src/databank/batch.rs` |

**access 层** 的 `materialize`(把解码字节按 slice 拷成连续 Vec)走 CPU pool,
阈值 `≤16KB` 且 `≤4 ranges` 才 inline 到调度线程(`cpu.rs:14-15`),否则丢 `access-cpu-*` worker。

**databank 层** 已增加自有 compute pool。Dense2D 使用 group-parallel scatter;Dense1D 对连续
uncompressed memory chunk 直接按 row copy;CSR 对 single memory chunk 直接从 `indptr/indices/data`
scatter。CSR 对 `Memory + identity codec + multi-chunk` 也有 direct scatter fastpath,按 indices/data
各自的 1D chunk 边界切分,直接写 dense output,跳过 batch plan/group staging 与 access scheduler。
projected CSR 会先筛 selected data groups,避免为 projection 读取未命中的 data chunk。

#### scatter 专项压测(4 MiB 输出,256 cells × 4096 genes)

| bench | MiB/s | 说明 |
|---|---|---|
| `ST_dense2d_scatter_4mib_output`(None) | **5386** | DataBank compute pool + group-parallel scatter |
| `ST_dense2d_scatter_zstd_4mib_output`(zstd) | 384 | zstd 解码 + scatter,被解码主导 |
| `ST_sparse_csr_noblock_repeat_chunk_checked_64x4096` | **50005** | 一个 batch 内频繁命中同一 chunk |
| `ST_sparse_csr_noblock_repeat_chunk_unchecked_64x4096` | **54176** | unchecked CSR 热路径 |
| `ST_sparse_csr_noblock_scattered_chunks_checked_64x4096` | **50216** | 一个 batch 内基本每 cell 命中不同 chunk |
| `ST_sparse_csr_noblock_scattered_chunks_unchecked_64x4096` | **51544** | scattered unchecked |

**分析**:
- Dense2D scatter 已从 3.4G 提升到约 5.4G,但仍低于目标。根因是 2D 布局按 cell 行访问时
  一个 cell 会命中多个列 chunk,产生大量小片段 copy 与 group 调度开销。Dense2D 保留兼容即可,
  不作为 20G 主目标。
- zstd scatter 仍被 zstd 解码主导。zstd 属算法/硬件上限,20G 路径应使用 lz4/blosc 或 none。
- Dense1D 与 CSR 是主目标,当前主路径已达标:Dense1D 42.4G,CSR single chunk checked/unchecked
  48.0/51.3G,multi-chunk no-block repeat/scattered 均 50G+。关键是对无阻塞 memory identity
  布局跳过通用 plan/group staging,只保留必要的 chunk 边界切分与 scatter。

#### 端到端瓶颈链(当前)

```
DataBank::access_cells (调用线程阻塞等待)
  ├─ plan + scheduled access (调度层,多线程,24G)     ← 已优化
  ├─ zstd 解码 (DecodePool,多线程,但 zstd 仅 7.7G)  ← codec 容量上限,非 Rust P0
  └─ databank scatter/project (compute pool)        ← CSR/Dense1D 主路径已达标
```

**databank no-block 带宽** 在 none/uncompressed 的 Dense1D、CSR single chunk 与 CSR multi-chunk
主路径上均超过 20G,cell/s 也远超 32k。未开启 prefetch 的 file+compressed direct access
仍可能被同步等待拉低;这类 blocking reference 不作为 databank scatter/project 吞吐验收。
下一步不再优先 Dense2D,而是复核 projection 与 scheduled prefetch 在真实调用方式下的稳定吞吐。

---

## 8. 已确认的瓶颈与优化建议

按优先级排序。**瓶颈 2(access 调度)已修复**,zstd 已排除为 Rust codecs 额外瓶颈。

### ✅ 已排除:zstd 不是 Rust codecs 额外瓶颈

- **现象**(最新):zstd t16 仅 **7.7 GiB/s**,t8=6.5G,t8→t16 提升 19% 后近饱和。
  对照:lz4 t16=23.0G,blosc t8=40.3G,均够 20G。
- **复核结果**:对齐 numcodecs 的 512MiB / 1MiB chunk manifest 后,Rust zstd 与 numcodecs
  在同一档:
  - numcodecs best:`zstd.level0` 959.6 MiB/s,`zstd.level3` 970.3 MiB/s。
  - Rust 当前 best:`zstd.level0` 979.1 MiB/s,`zstd.level3` 947.1 MiB/s。
- **API 对照**:同进程 probe 比较 `zstd::bulk::decompress_to_buffer`、复用
  `zstd::bulk::Decompressor`、`zstd_safe::decompress`;差异只有个位数百分点。
  DCtx 复用不是 7.7G→15G 的突破点。
- **当前实现**:`ZstdCodec` 优先信任上游 `expected_size`,unknown-size 时才读取
  frame content size;解压走 `zstd::zstd_safe::decompress`。兼容 numcodecs,无外部 API 变化。
- **结论**:zstd 到不了 20G 是 codec 选择/硬件容量问题,不是 Rust codecs 待修 bug。
  若目标是 20G,优先切 lz4/blosc;若必须 zstd,只能增加有效核心或更高单核性能。

### ✅ 瓶颈 2:access 调度单线程饱和 —— 已修复

- **状态**:**已修复**。重构为 sharded scheduler(`scheduler.rs:168` `scheduler_shards`,
  `scheduler.rs:402` `shard_for_key`),N 个独立 runtime 线程按 `ChunkKey` 哈希分片。
- **效果**(见 §6.3):send evict t16 从 5.9G → **17.2G**(2.9×),t2 饱和 → 线性扩展;
  **scheduled t16 = 24.2G**(已超 20G);kops 从 94k → **275k**;prefetch t8 达 24.2G。
- **遗留**:
  1. `scheduler_shards` 默认 1,stress bench 用 4。**生产需显式配置**(建议 ≈ 核数/4 ~ 核数/2)。
  2. 当前 bench 是 None codec + 内存 IO 轻负载,真实 zstd 负载下调度层是否仍够,需 §9.1 验证。
- **后续可选优化**(若调度层仍不够 20G):
  - `scheduler_shards` 提到 8(容器 32 核有空间)。
  - 继续降低单 op 调度线程工作量(已减少 scheduled 同 tick 重复命令、去掉 `next()`
    重复窗口扫描、延迟 clone scheduler 依赖,并 inline decoded-cache/ready/take 快路径)。
  - cache 查找移出调度线程(`ArcSwap` / lock-free,调用线程先查)。

### ✅ 瓶颈 5:CSR/Dense1D 热路径已达标(已修复)

- **现象**(2026-06-28 远端 CPU 机器,`SCDATA_BENCH_ITERS=512`):
  - Dense2D scatter(None)= **5386 MiB/s**;已并行化但受 2D 多列 chunk 小片段 copy 限制。
  - Dense1D memory= **42354 MiB/s**;约 **10.8M cells/s**,超过 20G/s 与 32k cells/s。
  - CSR single chunk checked/unchecked= **47954/51295 MiB/s**;约 **24.6M/26.3M cells/s**。
  - CSR no-block multi-chunk repeat checked/unchecked= **50005/54176 MiB/s**。
  - CSR no-block multi-chunk scattered checked/unchecked= **50216/51544 MiB/s**。
- **修复**:Dense1D 对 uncompressed memory 1D chunk 增加连续 row copy fastpath;CSR 对
  single memory chunk 增加直接 `indptr/indices/data` scatter fastpath;CSR 对 memory identity
  multi-chunk 增加专用 direct scatter,按 indices/data 各自 chunk 边界切片,避开 batch plan
  临时分配、index staging buffer、group→piece 映射和 access scheduled 迭代开销。
- **影响**:access 调度层和 iopool 够用,主格式 no-block databank 已不再受 plan/scatter/project 限制。
- **Dense2D 定性**:Dense2D 按 cell 行访问一个 cell 会命中多个列 chunk,天然会产生更多 chunk miss
  和小片段 copy。该路径只保留兼容,不作为 20G/s 主优化目标。
- **blocking reference**:`blocking_sparse_csr_lz4_file_*` 未开启 prefetch,repeat 约 8.3/7.7G、
  scattered 约 0.46/0.45G,这是同步等待 IO/decode 的 direct access 结果,不作为 databank 模块吞吐目标。
- **剩余验证**:projection 与 scheduled prefetch 仍需按真实调用方式单独压测;zstd 路径单独按
  7-12G 容量规划,不要拿 zstd 作为 20G 验收路径。
- **验收 bench**:已通过。最终复核 log:
  `target/databank-stress-20260628-132912-csr-direct-final.log`。

### 瓶颈 3:DataBank 多实例扩展仍需配置约束(P1)

- **现象**:多 DataBank 实例并行 t2=3277,t4=3577,t8=3473 MiB/s,已有改善但扩展仍有限。
- **根因**:default `AccessCpuConfig.num_workers=4` + `FillConfig.num_workers=4`,
  多实例并发时容易过度订阅;codec/decode/io pool 也会参与线程竞争。
- **优化方向**:
  1. 真实部署按并发实例数调 `num_workers`(如 8 实例 × 2 workers)。
  2. 评估 codec cache 的 `RwLock` 读路径是否可换 `ArcSwap` 或 hash 分片。
- **验收 bench**:`databank/MT_instances_dense2d_*` t8 应 ≥ t2 × 3。

### 瓶颈 4:`DataBank` Send/Sync 边界(已修复,需持续审计)

- **现象**:历史上 `GeneNameView` 的 `*const u8` 沿类型链传播 `!Send`,无法 `Arc<DataBank>` 跨线程共享。
- **源码**:`src/databank/interner.rs:6`。
- **状态**:已通过 `unsafe impl Send/Sync` 并用 `Arc<str>` 保活证明;测试覆盖
  `databank_is_send_sync` 与 `shared_databank_allows_concurrent_callers`。
- **后续审计**:新增任何 FFI view/裸指针字段时必须重新证明生命周期和跨线程只读语义。

### 非瓶颈(无需优化)

- **iopool 调度层**:52G / 450k IOPS / 61G dedup,扩展性好。
- **decode_pool owned_buffer 反慢**:不影响生产路径(见 §4.3)。
- **zstd Rust 实现**:已与 numcodecs 对齐;到不了 20G 是 zstd 算法/硬件上限,非 Rust bug。
- **lz4 / blosc 解压**:已够 20G。
- **sharding**:对小 IO 高并发有效。

---

## 9. 全链路 scheduled prefetch

access 调度与 databank no-block 主路径修复后,以下 gap 主要用于真实调用方式确认,不再代表
databank scatter/project 或 codecs 模块存在 P0 性能问题。

### 9.1 scheduled prefetch cells 全链路压测(已补)

bench 目标:`rust/scdata/benches/fullchain.rs`。
启用方式:`SCDATA_FULLCHAIN_PREFETCH=1 cargo bench ... --bench fullchain`。

覆盖能力:
- `FileOffset` CSR 数据集,路径是 `DataBank::prefetch_cells_scheduled::<f32, _>`。
- `SCDATA_FULLCHAIN_FILE_MIB` 在未显式设置 `SCDATA_FULLCHAIN_BATCHES` 时估算 batch 数,用于控制生成数据规模;
  `SCDATA_FULLCHAIN_DIR` 可指定大盘目录。
- `SCDATA_FULLCHAIN_CODEC` 控制 chunk codec:`raw`/`uncompressed`、`lz4`(默认)、`zstd`、`crc32`。
- `SCDATA_FULLCHAIN_REPEAT_ENCODED_CHUNK=1` 复用同尺寸 encoded chunk,每个 chunk 仍有独立 file offset。
- `SCDATA_FULLCHAIN_IO_BACKEND` 控制 IO backend:`threaded`(默认)或 `uring`。
- `SCDATA_FULLCHAIN_URING_DRIVERS` / `URING_ENTRIES` / `URING_REGISTERED_FILES`
  控制 io_uring driver 数、ring depth 和 fixed-file slots。
- `SCDATA_FULLCHAIN_CHUNKS_PER_BATCH` 控制目标 batch chunk 数;未设置 `SCDATA_FULLCHAIN_CHUNK_NNZ`
  时用它反推 chunk 长度。
- `SCDATA_FULLCHAIN_CHUNK_MISS_PERMILLE` 控制 batch cell window 的 miss 概率(0=全复用,1000=全 miss)。
- `SCDATA_FULLCHAIN_BATCH_CELLS`、`GENES`、`NNZ_PER_CELL`、`CHUNK_NNZ` 控制 CSR 形状。
- `SCDATA_FULLCHAIN_DATABANK_PREFETCH_STEP` 控制 databank batch ring 深度;旧名
  `SCDATA_FULLCHAIN_PREFETCH_STEP` 仍兼容。
- `SCDATA_FULLCHAIN_ACCESS_PREFETCH_STEP` / `ACCESS_DECODE_AHEAD` / `ACCESS_READY_AHEAD`
  控制 access scheduled lookahead。
- `SCDATA_FULLCHAIN_IO_WORKERS` / `IO_SHARDS` / `IO_INFLIGHT`、`SCHEDULER_SHARDS`、
  `ACCESS_CPU_WORKERS`、`DECODE_WORKERS`、`FILL_WORKERS` 控制下游线程池。
- `SCDATA_FULLCHAIN_LABEL` 写入日志,用于标记 baseline/bg_cpu/bg_io 等场景。
- `SCDATA_FULLCHAIN_WARMUP_BATCHES` 控制 fullchain warmup 轮数。
- `SCDATA_FULLCHAIN_BG_CPU_THREADS` 启动后台 CPU busy-loop 线程。
- `SCDATA_FULLCHAIN_BG_IO_READERS` 启动后台随机读线程;默认读取单独的 background 大文件。
- `SCDATA_FULLCHAIN_BG_IO_FILE_MIB` 控制 background IO 文件大小。
- `SCDATA_FULLCHAIN_BG_IO_READ_KIB` 控制每次 background read 大小。
- `SCDATA_FULLCHAIN_BG_IO_SAME_FILE=1` 改为对数据集 data 文件施加 background random read。
- `SCDATA_FULLCHAIN_BG_IO_DROP_CACHE=1` 让 background reader 每次 read 后调用 `DONTNEED`,用于更强冷读干扰。
- `SCDATA_FULLCHAIN_MATRIX=1` 启用轻量配置矩阵。默认矩阵是
  `raw,lz4,zstd` × 可用 IO backend × `baseline,bg_cpu,bg_io,bg_cpu_io`;
  可用 `SCDATA_FULLCHAIN_MATRIX_CODECS` / `MATRIX_BACKENDS` / `MATRIX_BACKGROUNDS` 缩小集合。

输出字段:
- `label`:当前运行标签;矩阵模式下自动写成 `codec-backend-background`。
- `throughput_mib_s` / `output_mib_s`:处理后的 dense cell buffer 吞吐。
- `slice_mib_s`:实际 CSR cell 非零值片段吞吐,等于 `cells * nnz_per_cell * (index+data 字长)`。
- `encoded_file_mib`:本次生成的 indices/data encoded 文件总大小。
- `wait_ns_p50/p99/p999/max`:消费者在 `PrefetchCells::next()` 上等待 batch 完成的时间分位数。
- `kops`:batch 消费速度。
- `bg_sum`:background 线程 checksum/读取字节折叠值,防止线程被优化掉。

实测机器:`ms-0628-121735-73258904-8rgr5.wangzhongqi.ailab-dnacoding.pod@h.pjlab.org.cn`。

| 场景 | 配置 | requested/miss IO | CSR slice | output | cells | log |
|---|---|---:|---:|---:|---:|---|
| 4GiB 文件,64 chunk/batch | chunk pair≈1MiB,batch=1024 cells,miss=100%,prefetch=8/64,IO workers=16 | **10814 MiB/s** | 84.5 MiB/s | 2704 MiB/s | 173k cells/s | `target/databank-fullchain-prefetch-20260628-141044-ms0628-largechunk.log` |
| 16GiB 文件,256 chunk/batch | chunk pair≈1MiB,batch=1024 cells,miss=100%,prefetch=4/512,IO workers=32 | **10941 MiB/s** | 21.4 MiB/s | 684 MiB/s | 43.8k cells/s | `target/databank-fullchain-prefetch-20260628-141848-ms0628-16g-slice.log` |
| 16GiB 文件,256 chunk/batch,更高并发 | access prefetch=1024,IO workers=64,shards=16 | 9764 MiB/s | 19.1 MiB/s | 610 MiB/s | 39k cells/s | `target/databank-fullchain-prefetch-20260628-141154-ms0628-16g-64io.log` |

实测机器:`ms-0628-142701-37384502-cfcd2.wangzhongqi.ailab-dnacoding.pod@h.pjlab.org.cn`
(8x H200,170 CPU,1.4TiB RAM)。

| 场景 | 配置 | encoded IO | decoded chunk | output | cells | next p50/p99 | log |
|---|---|---:|---:|---:|---:|---:|---|
| 256GiB raw / 232GiB lz4 文件 | `codec=lz4`,`repeat_encoded_chunk=1`,`io_backend=uring`,`uring_drivers=16`,`decode_workers=16`,1024 batches,100% miss | **950 MiB/s** | 1047 MiB/s | 65 MiB/s | 4.2k cells/s | 224ms / 577ms | `target/databank-fullchain-prefetch-20260628-144648-h200-256g-uring16-lz4.log` |

256GiB lz4 文件调参矩阵(128 batch,100% miss,1024 cells/batch,256 chunk/batch):

| backend/config | encoded IO | decoded chunk | cells | log |
|---|---:|---:|---:|---|
| threaded64,shards32,inflight32768 | **1144 MiB/s** | 1261 MiB/s | 5.0k/s | `target/databank-fullchain-prefetch-20260628-145708-h200-tune-uring16_base.log` |
| uring16,entries4096 | 893 MiB/s | 984 MiB/s | 3.9k/s | `target/databank-fullchain-prefetch-20260628-145737-h200-tune-uring16_p4096.log` |
| uring32,entries4096 | 823 MiB/s | 907 MiB/s | 3.6k/s | `target/databank-fullchain-prefetch-20260628-145814-h200-tune-uring32_p8192.log` |
| uring64,entries4096 | 855 MiB/s | 943 MiB/s | 3.8k/s | `target/databank-fullchain-prefetch-20260628-145853-h200-tune-uring64_p16384.log` |
| threaded96,shards64 | 918 MiB/s | 1012 MiB/s | 4.0k/s | `target/databank-fullchain-prefetch-20260628-145931-h200-tune-threaded96.log` |
| threaded160,shards80 | 957 MiB/s | 1055 MiB/s | 4.2k/s | `target/databank-fullchain-prefetch-20260628-150006-h200-tune-threaded160.log` |
| threaded48,shards32 | **1155 MiB/s** | 1273 MiB/s | 5.1k/s | `target/databank-fullchain-prefetch-20260628-150248-h200-tune2-threaded48_base.log` |

同一文件、不同 batch shape:

| batch shape | encoded IO | output | cells | next p50/p99 | log |
|---|---:|---:|---:|---:|---|
| 4096 cells / 128 chunks,batches=128 | 1078 MiB/s | 594 MiB/s | **38.0k/s** | 97ms / 397ms | `target/databank-fullchain-prefetch-20260628-150416-h200-shape-cells4096_chunks128.log` |
| 8192 cells / 256 chunks,batches=64 | 929 MiB/s | 512 MiB/s | **32.8k/s** | 188ms / 3615ms | `target/databank-fullchain-prefetch-20260628-150432-h200-shape-cells8192_chunks256.log` |
| 8192 cells / 128 chunks,batches=64 | 890 MiB/s | 982 MiB/s | **62.8k/s** | 99ms / 1532ms | `target/databank-fullchain-prefetch-20260628-150451-h200-shape-cells8192_chunks128.log` |
| 16384 cells / 128 chunks,batches=32 | 743 MiB/s | 1638 MiB/s | **104.8k/s** | 108ms / 1630ms | `target/databank-fullchain-prefetch-20260628-150503-h200-shape-cells16384_chunks128.log` |
| 8192 cells / 128 chunks,batches=512 | 963 MiB/s | 1062 MiB/s | **67.9k/s** | 104ms / 426ms | `target/databank-fullchain-prefetch-20260628-150541-h200-long-cells8192-chunks128.log` |

同一 232GiB encoded 文件 `dd iflag=direct bs=64M` 顺序读为 **1.7GB/s**:
`249380470784 bytes copied, 143.429 s, 1.7 GB/s`。

结论:
- bench 已经能按生产调用方式测 scheduled prefetch cells 的全链路吞吐,并区分 IO 字节与输出字节。
- `wait_ns_*` 指标现在用于判断 completed queue 是否被 drain 空;如果 consumer loop 几乎不做
  计算,它仍会等待 producer,这不再代表 `next()` 自己同步执行 databank plan/scatter。
- 当前新 CPU 机器上,冷/准冷文件 full-chain requested IO 在 9.8-10.8GiB/s;继续加 IO 线程没有提升。
- 同文件 `dd iflag=direct` 顺序读约 1.2GB/s,说明该 worker/GPFS 的真实冷读带宽波动很大;
  这组 full-chain 结果不能证明 databank 已经跑满 20GiB/s 物理 IO,但能作为后续机器调参的基准。
- H200 worker 上,这组 fullchain 随机 chunk 访问里 `threaded` 明显优于 `uring`;
  继续增加 uring drivers、threaded workers 或 prefetch depth 没有提升。
- batch shape 对 cell 吞吐影响很大:同样 256GiB lz4 文件,`4096 cells/128 chunks` 已过
  32k cells/s,`8192 cells/128 chunks` 长测稳定到 67.9k cells/s。这符合离线重排后一个 batch
  对 chunk 命中更集中的生产形态。
- 当前 fullchain 结果仍不能单独说明 GPFS 随机 IO 上限,因为 databank 组装、decode 和 output
  scatter 都在链路里。下一步需要同文件 iopool-level random read probe 隔离纯 IO。

**已有参考**:`databank/blocking_sparse_csr_lz4_file_*` 是 direct blocking reference:
repeat 约 8.3/7.7G,scattered 约 0.46/0.45G。它说明无 prefetch 时同步等待会主导吞吐,不作为
databank no-block 压测目标。

### 9.2 后续扩展

- 已增加 lz4/zstd full-chain 文件压缩配置;后续可补 blosc encoder。
- 增加 projected CSR full-chain 场景,指定 gene 顺序一次,多 batch 复用 projection。
- 在本地 NVMe 或确认有 >20GiB/s 顺序 direct read 的存储上重跑,确认 databank/access 是否能吃满。
- databank scheduled prefetch 已改为后台 producer + completed queue;后续 full-chain bench 需要增加
  consumer-side 模拟处理时间/队列 occupancy 指标,才能判断 Python dataloader 层是否完全隐藏延迟。
- 用 `SCDATA_FULLCHAIN_MATRIX=1` 固定跑 baseline / bg_cpu / bg_io / bg_cpu_io 矩阵,比较
  `wait_ns_p99`、`wait_ns_max` 和 `kops`。

### 9.3 iopool O_DIRECT / 本地 NVMe

**当前**:IO 在 NFS page cache 上,数据不可信(52G 是 cache)。
用户已声明 IO 够,但若要测 IO 子系统真实上限,需 O_DIRECT 或本地 NVMe。

---

## 10. 可复现命令

### 10.1 环境(CPU 机器)

```sh
ssh -CAXY ms-0622-195529-53044032-d284c.wangzhongqi.ailab-dnacoding.pod@h.pjlab.org.cn
```

容器 cpuset `80-95,176-191`(32 核,NUMA node1),cargo 在 `$HOME/.cargo/bin`,
源码在 `/mnt/shared-storage-user/dnacoding/wangzhongqi/Code/Project/scdata`。

### 10.2 跑 bench

```sh
# 轻量单元 bench(modules.rs)
export PATH=$HOME/.cargo/bin:$PATH
export CARGO_NET_OFFLINE=true
cargo bench --manifest-path \
  /mnt/shared-storage-user/dnacoding/wangzhongqi/Code/Project/scdata/rust/scdata/Cargo.toml \
  --bench modules

# 多线程压力 bench(stress.rs)
cargo bench --manifest-path \
  /mnt/shared-storage-user/dnacoding/wangzhongqi/Code/Project/scdata/rust/scdata/Cargo.toml \
  --bench stress

# 调迭代数
SCDATA_BENCH_ITERS=4096 SCDATA_BENCH_WARMUPS=5 cargo bench ... --bench stress
```

### 10.2.1 只跑 scheduled prefetch 全链路

```sh
export PATH=$HOME/.cargo/bin:$PATH
export CARGO_NET_OFFLINE=true
cd /mnt/shared-storage-user/dnacoding/wangzhongqi/Code/Project/scdata

CARGO_TARGET_DIR=target/fullchain-ms0628 \
SCDATA_FULLCHAIN_PREFETCH=1 \
SCDATA_FULLCHAIN_FILE_MIB=16384 \
SCDATA_FULLCHAIN_WARMUP_BATCHES=2 \
SCDATA_FULLCHAIN_BATCH_CELLS=1024 \
SCDATA_FULLCHAIN_CHUNKS_PER_BATCH=256 \
SCDATA_FULLCHAIN_CHUNK_MISS_PERMILLE=1000 \
SCDATA_FULLCHAIN_GENES=4096 \
SCDATA_FULLCHAIN_NNZ_PER_CELL=64 \
SCDATA_FULLCHAIN_CHUNK_NNZ=131072 \
SCDATA_FULLCHAIN_DATABANK_PREFETCH_STEP=4 \
SCDATA_FULLCHAIN_ACCESS_PREFETCH_STEP=512 \
SCDATA_FULLCHAIN_ACCESS_DECODE_AHEAD=256 \
SCDATA_FULLCHAIN_ACCESS_READY_AHEAD=64 \
SCDATA_FULLCHAIN_IO_WORKERS=32 \
SCDATA_FULLCHAIN_IO_SHARDS=8 \
SCDATA_FULLCHAIN_IO_INFLIGHT=8192 \
SCDATA_FULLCHAIN_SCHEDULER_SHARDS=8 \
SCDATA_FULLCHAIN_ACCESS_CPU_WORKERS=8 \
SCDATA_FULLCHAIN_DECODE_WORKERS=8 \
SCDATA_FULLCHAIN_FILL_WORKERS=4 \
cargo bench --manifest-path rust/scdata/Cargo.toml --bench fullchain
```

可选:`SCDATA_FULLCHAIN_DIR=<large-disk-dir>` 指定大文件目录;默认写入
`rust/scdata/target/bench-data`。

要一次跑覆盖矩阵,把第一行的启用变量替换为:

```sh
SCDATA_FULLCHAIN_MATRIX=1
```

默认矩阵是 `raw,lz4,zstd` × 可用 IO backend × `baseline,bg_cpu,bg_io,bg_cpu_io`。
可以用逗号分隔列表缩小:

```sh
SCDATA_FULLCHAIN_MATRIX_CODECS=lz4,zstd
SCDATA_FULLCHAIN_MATRIX_BACKENDS=threaded
SCDATA_FULLCHAIN_MATRIX_BACKGROUNDS=baseline,bg_io
```

常用场景只需要在上述命令前追加这些变量:

```sh
# baseline 标记
SCDATA_FULLCHAIN_LABEL=baseline

# 复用 chunk,模拟离线重排后 batch 内/跨 batch 命中热 chunk
SCDATA_FULLCHAIN_LABEL=low_miss \
SCDATA_FULLCHAIN_CHUNK_MISS_PERMILLE=100

# CPU background 干扰
SCDATA_FULLCHAIN_LABEL=bg_cpu \
SCDATA_FULLCHAIN_BG_CPU_THREADS=4

# IO background 干扰,默认读单独 background 大文件
SCDATA_FULLCHAIN_LABEL=bg_io \
SCDATA_FULLCHAIN_BG_IO_READERS=4 \
SCDATA_FULLCHAIN_BG_IO_FILE_MIB=4096 \
SCDATA_FULLCHAIN_BG_IO_READ_KIB=1024

# 更强干扰:background reader 周期性 DONTNEED,必要时也可以读同一数据文件
SCDATA_FULLCHAIN_LABEL=bg_io_cold \
SCDATA_FULLCHAIN_BG_IO_READERS=4 \
SCDATA_FULLCHAIN_BG_IO_DROP_CACHE=1
```

### 10.3 编译检查(开发盒,无需 ssh)

```sh
export PATH=/home/wangzhongqi/.cargo/bin:$PATH
cargo check --manifest-path rust/scdata/Cargo.toml --benches
```

### 10.4 bench 文件位置

| 文件 | 用途 |
|---|---|
| `rust/scdata/benches/modules.rs` | 单线程微基准:codecs / decode pool / iopool / access / databank + 数据管道 / missing 率 / 规模 / 场景矩阵 |
| `rust/scdata/benches/stress.rs` | 多线程压测:每个模块的 MT 扩展性 + missing 率 / 规模多线程 |
| `rust/scdata/benches/fullchain.rs` | 端到端 scheduled-prefetch 压测(env 驱动,后台 CPU/IO 压力 + percentile) |
| `rust/scdata/benches/codec_manifest.rs` | codec 解码 bench:`--manifest` 跑 Python 导出的真实数据;默认 synth 模式用 `support::data` 生成 payload |
| `rust/scdata/benches/support/mod.rs` | harness:`bench` / `stress_mt` / env helpers / `payload` / `bench_data_dir` |
| `rust/scdata/benches/support/data.rs` | 规范化数据生成管道(dtype × 分布 × missing 率 × 规模,确定性 PRNG) |
| `rust/scdata/benches/support/codecs.rs` | encode fixtures + `encode_for_spec` 统一入口 + `default_codec_matrix` |
| `rust/scdata/benches/support/chunks.rs` | dense / csr / directory chunk 构造与文件写入 |
| `rust/scdata/benches/support/backends.rs` | `SliceIo`(IoBackend)+ `CodecDecode`(DecodeBackend)mock |
| `rust/scdata/benches/support/manifest.rs` | manifest 驱动的 codec bench runner(verify / warmup / repeats / 统计 / CSV+JSON) |
| `rust/scdata/benches/iopool.rs` | 占位(真实 bench 在 `examples/iopool_bench.rs`) |

### 10.5 Cargo.toml 配置

`rust/scdata/Cargo.toml` 已注册 `modules` / `iopool` / `codec_manifest` / `stress` / `fullchain` 五个 bench,
均 `harness = false`。原 `src/bin/codec_manifest_bench.rs` 已删除,功能整合进 `codec_manifest` bench。

**已知 warning**:`[profile.release]` 写在子 crate 被忽略(workspace 根 `Cargo.toml` 才有效)。
当前用 cargo 默认 release(opt-level=3,无 LTO)。若要 LTO,把 `[profile.release]` 挪到
根 `Cargo.toml`:

```toml
# Cargo.toml(根)
[profile.release]
opt-level = 3
lto = true
```

---

## 附录:核心源码索引

| 模块 | 文件 | 行数 | 关键位置 |
|---|---|---|---|
| access | `src/access/scheduler.rs` | 3421 | sharded runtime: `:168`/`:402`/`:615`;`AccessHandle`: `:278`;`scheduled`: `:319`;inline scheduled fast paths: `:814`/`:879`/`:923`/`:1005` |
| access | `src/access/cache.rs` | 560 | LRU + pin: `:275`;`PinGuard`: `:186` |
| access | `src/access/cpu.rs` | 466 | scatter 池;inline ≤16KB: `:14` |
| codecs | `src/codecs/spec.rs` | 2125 | `ChunkCodec` trait: `:46`;`ZstdCodec`: `:1194`;codec cache: `:273` |
| codecs | `src/codecs/pool.rs` | 424 | `DecodePool`: `:164`;`submit`/`try_submit`/`submit_async`: `:210/218/233` |
| iopool | `src/iopool/mod.rs` | 2493 | `IoPool`: `:1596`;sharding: `:1757`;dedup: `:354` |
| iopool | `src/iopool/uring.rs` | 880 | io_uring 后端 |
| databank | `src/databank/mod.rs` | 455 | `DataBank`: `:40`;`access_cells`: `:119` |
| databank | `src/databank/batch.rs` | 7310 | access: `:287`;Dense1D fastpath: `:511`/`:836`;CSR fastpath: `:938`/`:1899`/`:2123`;unchecked: `:998`/`:2178` |
| databank | `src/databank/interner.rs` | 120 | `GeneNameView` `!Send`: `:6` |
| databank | `src/databank/adapter.rs` | 64 | `DecodePoolBackend` 生产路径: `:57-61` |

---

## 附录:优化优先级速查

> **更新(2026-06-28)**:access 调度已修复为 sharded 多线程(原 P0-2 已关闭)。
> zstd 已复核为与 numcodecs 同档,不再作为 Rust codecs P0。DataBank scatter 已有
> compute pool 并行路径,并新增 Dense1D 连续内存 fastpath、CSR single memory chunk fastpath
> 与 CSR memory multi-chunk direct scatter fastpath。CSR/Dense1D 主格式及 no-block multi-chunk
> CSR 已通过 20G/s 与 32k cells/s 验收。

| 优先级 | 瓶颈 | 状态 | 目标 | 验收 bench |
|---|---|---|---|---|
| ~~P0~~ | ~~CSR/Dense1D databank 热路径~~ | **✅ 已修复** | 20G/s + 32k cells/s | Dense1D=42.4G,CSR single=48.0/51.3G,multi-chunk=50G+ |
| ~~P0~~ | ~~zstd Rust 解压实现慢~~ | **✅ 已排除** | 与 numcodecs 对齐 | aligned manifest zstd-only |
| ~~P0~~ | ~~access 单线程调度饱和~~ | **✅ 已修复** | sharded 线性扩展 | `access/MT_scheduled_keep_16x64k/t16`=24.2G |
| P1 | DataBank 多实例调参 | 部分改善 | 避免线程过订阅 | `databank/MT_instances_dense2d_*/t8` |
| ~~P2~~ | ~~DataBank `!Send`~~ | **✅ 已修复** | 可 `Arc` 共享 | 编译通过 + MT 共享 bench |
| — | iopool / zstd / lz4 / blosc | 已定性 | zstd 容量规划,lz4/blosc 达 20G | — |

**当前结论**:
- **纯调度层带宽** ✅:scheduled t16=24.2G(超 20G),prefetch t8=24.2G,send evict t16=17.2G。
- **kops** ✅:275k >> 32k,富余 8.6×;scheduled t16=387k chunks/s。
- **databank no-block scatter/project** ✅:Dense2D 兼容路径 5.4G;Dense1D 42.4G;
  CSR single 48.0/51.3G;CSR multi-chunk repeat/scattered 均 50G+。
- **blocking direct access**:未开启 prefetch 的 file+compressed 数字只作同步等待参考,不作为模块吞吐目标。

**下一步**:进入真实调用方式验证阶段。若业务用 lz4/blosc/none(解压够 20G),重点确认
scheduled prefetch 下 file/directory、projection 与 CSR batch 不回退;若用 zstd,最终瓶颈仍会回到
zstd 解码容量。§9.1/§9.2 用于验证 scheduled/prefetch 场景,不是 direct blocking 压测。
