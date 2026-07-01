# DataBank 模块说明

本文档对应 `rust/scdata/src/databank` 当前工作区代码。这个模块是 Rust 侧单细胞数据访问的门面层，负责把外部传入的 dense / CSR sparse 数据集描述注册成可访问的数据集，并提供按 cell、按 gene name 投影、同步预取和 scheduled prefetch 的访问接口。

## 模块职责

`databank` 主要做五件事：

1. 管理数据集元数据和生命周期：注册 `Dense1D`、`Dense2D`、`SparseCsr` 数据集，返回 generation-checked 的 `DatasetId`，注销时释放由 DataBank 注册的文件句柄。
2. 抽象分块数组存储：用 `ArraySpec` 描述 shape、dtype、codec、chunk grid 和 chunk source，支持内存 chunk、已解码内存 chunk、文件 chunk 和外部已注册文件 chunk。
3. 访问 cell 矩阵：把请求的 cell 行映射为 chunk slice，调度 I/O 和 decode，把结果 scatter 到 row-major 输出 buffer。
4. 支持基因名投影：按请求 gene name 重排列、过滤列，缺失 gene 可补零或报错。
5. 支持 scheduled prefetch：后台 producer 把 batch source 转成多个独立 batch plan，预取 chunk，组装已解码 batch，并按原 batch 顺序返回。

外部模块只应依赖 `databank::mod.rs` 中 re-export 的 API。子模块里存在一些 `pub` 项，但由于子模块本身是私有 `mod`，它们不是 crate 外部的稳定接口。

## 文件结构

| 文件 / 目录 | 作用 |
| --- | --- |
| `mod.rs` | `DataBank` 门面、公共 re-export、生命周期和访问 API。 |
| `config.rs` | `DataBankConfig`、`FillConfig`、`ScheduledPrefetchConfig`。 |
| `dataset.rs` | 内部 `Dataset` enum、dense/sparse spec 校验和构建。 |
| `array/` | 分块数组元数据、dtype/cast、codec spec、grid/range 规划、chunk source 构建。 |
| `registry.rs` | `DatasetId` 和 slot/generation registry。 |
| `interner.rs` | gene name interning、`GeneNameView` 指针视图。 |
| `gene_axis.rs` | gene name 投影、缺失策略、多数据集 batch 行布局。 |
| `direct.rs` | 同步 `access_cells*` 分发层。 |
| `plan.rs` | dense/sparse 的 cell 到 chunk/range 的 planner。 |
| `dense/` | dense group、load、scatter 和 memory identity 快路径。 |
| `sparse/` | CSR sparse group、index/data load、scatter、unchecked 快路径。 |
| `scheduled/` | scheduled prefetch 的 planner、producer、assembler、类型和 profiling。 |
| `compute.rs` | DataBank 自己的 CPU worker pool，用于 scatter/组装。 |
| `adapter.rs` | `IoPool` / `DecodePool` 到 `AccessScheduler` backend trait 的适配器。 |
| `profile.rs` | DataBank 门面层 profiling 指标。 |
| `util.rs` | 提交 access request、slice copy、zero fill、prefetch key 收集等通用工具。 |

## 对外接口总览

`databank::mod.rs` 对外暴露以下类型：

```rust
pub use array::{
    ArrayCodecSpec, ArrayGridSpec, ArrayOrder, ArraySpec, Bf16Bits, ChunkSourceSpec,
    ChunkSpec, DType, DataValue, EdgeChunkLayout, F16Bits, RegisteredFile,
};
pub use batch::{MissingGenePolicy, MultiBatchCells, PrefetchCells, PrefetchedBatch};
pub use config::{DataBankConfig, FillConfig, ScheduledPrefetchConfig};
pub use dataset::{Dense1DSpec, Dense2DSpec, SparseCsrSpec};
pub use error::{DataBankError, DataBankResult};
pub use interner::GeneNameView;
pub use registry::DatasetId;

pub struct DataBank;
```

注意：`Array`、`Chunk`、`ChunkRef`、`Dataset`、`DatasetRegistry`、`GeneInterner`、`DenseSegment` 等虽然在私有子模块中是 `pub` 或 `pub(crate)`，但没有被 `databank` 顶层 re-export，不应作为模块外 API 使用。

## `DataBank`

`DataBank` 是主入口。它持有：

| 字段 | 内部含义 |
| --- | --- |
| `io_pool: Arc<IoPool>` | 文件注册和异步读。 |
| `_decode_pool: Arc<DecodePool>` | chunk 解码 worker pool，字段名前缀表示主要由 access scheduler 持有使用。 |
| `access: AccessHandle` | `AccessScheduler` 的句柄，统一调度 I/O、decode、cache 和 scheduled access。 |
| `compute: Arc<DataBankComputePool>` | DataBank 侧 CPU worker pool，用于 scatter/组装。 |
| `registry: DatasetRegistry` | `DatasetId` 到 `Arc<Dataset>` 的 slot/generation 表。 |
| `retired: Vec<Arc<Dataset>>` | 注销后仍被 scheduled iterator 持有的数据集，延迟释放文件句柄。 |
| `interner: GeneInterner` | gene name 字符串池。 |
| `config: DataBankConfig` | 构造时使用的配置副本。 |
| `profiler: DataBankProfile` | 门面层 profile runtime。 |

### 构造和配置

```rust
pub fn new(config: DataBankConfig) -> DataBankResult<Self>
pub fn config(&self) -> &DataBankConfig
```

`new` 会先 `config.validate()`，然后创建：

1. `IoPool`
2. `DecodePool`
3. `DataBankComputePool`
4. `AccessScheduler`

`AccessScheduler` 通过 `IoPoolBackend` 和 `DecodePoolBackend` 适配到现有 `IoPool` / `DecodePool`。

### Profiling

```rust
pub fn profile(&self) -> &ProfileRuntime
pub fn profile_snapshot(&self) -> ProfileSnapshot
pub fn profile_snapshot_and_reset(&self) -> ProfileSnapshot
pub fn reset_profile(&self)
```

`profile.rs` 定义了 `databank` component，包含四个 scope：

| Scope | 统计内容 |
| --- | --- |
| `databank.lifecycle` | register / unregister / cleanup / drop。 |
| `databank.access` | 同步访问调用、cells、genes、输出元素和字节数、错误数。 |
| `databank.prefetch` | 直接 `prefetch_cells` 调用。 |
| `databank.scheduled-api` | scheduled prefetch 构造调用、数据集数量、错误数。 |

### 数据集注册

```rust
pub fn register_dense_1d(&mut self, spec: Dense1DSpec) -> DataBankResult<DatasetId>
pub fn register_dense_2d(&mut self, spec: Dense2DSpec) -> DataBankResult<DatasetId>
pub fn register_sparse_csr(&mut self, spec: SparseCsrSpec) -> DataBankResult<DatasetId>
pub fn unregister(&mut self, id: DatasetId) -> DataBankResult<()>
```

注册流程：

1. `cleanup_retired()` 尝试释放之前延迟释放的数据集文件句柄。
2. `DatasetRegistry::ensure_can_register()` 检查 slot 数量。
3. `GeneInterner::intern_dataset()` intern gene names，得到 `DatasetGeneRefs` 和 `GeneNameView`。
4. 调用对应 `Dataset::*::from_spec` 做元数据校验、构建 `Array`、注册文件。
5. 成功后放入 `DatasetRegistry`，返回 `DatasetId { slot, generation }`。
6. 失败时释放本次 intern 的 gene 引用，避免 refcount 泄漏。

注销流程：

1. 按 `DatasetId` 从 registry 删除 dataset。
2. 释放 gene interner refcount。
3. 如果 `Arc::strong_count(&dataset) == 1`，立即 unregister DataBank 自己注册的文件。
4. 如果还有 scheduled iterator 等外部 `Arc` 持有数据集，把它放入 `retired`，后续 `cleanup_retired()` 或 `Drop` 再释放文件。

`Drop` 会清理 retired 数据集，并 drain registry 中仍注册的数据集。释放文件时忽略 drop 阶段的错误。

### 元数据查询

```rust
pub fn dataset_genes(&self, id: DatasetId) -> DataBankResult<&[GeneNameView]>
pub fn dataset_num_cells(&self, id: DatasetId) -> DataBankResult<usize>
pub fn dataset_num_genes(&self, id: DatasetId) -> DataBankResult<usize>
pub fn dataset_dtype(&self, id: DatasetId) -> DataBankResult<DType>
```

`dataset_genes` 返回的是指向 DataBank 内部 interned gene 字符串的 pointer/len 视图。这个 slice 和其中指针只在对应数据集保持注册期间有效。

### 同步访问：调用方提供输出 buffer

```rust
pub fn access_cells<T: DataValue>(
    &self,
    id: DatasetId,
    cells: &[usize],
    out: &mut [T],
    names: Option<&mut [GeneNameView]>,
) -> DataBankResult<()>

pub fn access_cells_values<T: DataValue>(
    &self,
    id: DatasetId,
    cells: &[usize],
    out: &mut [T],
) -> DataBankResult<()>

pub fn access_cells_with_config<T: DataValue>(
    &self,
    id: DatasetId,
    cells: &[usize],
    out: &mut [T],
    names: Option<&mut [GeneNameView]>,
    config: ScheduledAccessConfig,
) -> DataBankResult<()>
```

输出约定：

| 项 | 约定 |
| --- | --- |
| 输出 layout | row-major，长度为 `cells.len() * num_genes`。 |
| 行顺序 | 与 `cells` 请求顺序一致。 |
| 列顺序 | 默认数据集 gene 顺序。 |
| `names` | 如果提供，长度必须为输出 gene 数，返回每列的 `GeneNameView`。 |
| dtype | 源 dtype 必须能 cast 到 `T::DTYPE`；唯一禁止方向是 float 到 int。 |

`access_cells_values` 是不返回 gene names 的便捷 alias。

### 同步访问：按 gene name 投影

```rust
pub fn access_cells_by_gene_names<T, G>(
    &self,
    id: DatasetId,
    cells: &[usize],
    gene_names: &[G],
    out: &mut [T],
    names: Option<&mut [GeneNameView]>,
    missing: MissingGenePolicy,
) -> DataBankResult<()>
where
    T: DataValue,
    G: AsRef<str>

pub fn access_cells_by_gene_names_with_config<T, G>(
    &self,
    id: DatasetId,
    cells: &[usize],
    gene_names: &[G],
    out: &mut [T],
    names: Option<&mut [GeneNameView]>,
    missing: MissingGenePolicy,
    config: ScheduledAccessConfig,
) -> DataBankResult<()>
where
    T: DataValue,
    G: AsRef<str>
```

输出长度为 `cells.len() * gene_names.len()`。内部会编译 `GeneAxisPlan`：

1. 如果请求 gene 顺序与数据集完全一致，降级为 `DatasetOrder`，不做投影。
2. 否则构造 `CompiledGeneProjection`：
   - `output_by_source[source_gene] = output_col`
   - `output_names[output_col] = GeneNameView`
   - `selected_sources` 是排序后的源 gene 下标，用于 dense planner 合并连续 run。
3. 请求列表中重复 gene 会返回 `DuplicateGeneName`。
4. 数据集自身 gene name 重复会返回 `InvalidArrayMeta`。
5. 缺失 gene 由 `MissingGenePolicy` 控制。

### 同步访问：DataBank 分配输出 buffer

```rust
pub fn access_cells_owned<T: DataValue>(
    &self,
    id: DatasetId,
    cells: &[usize],
) -> DataBankResult<Vec<T>>

pub fn access_cells_owned_with_config<T: DataValue>(
    &self,
    id: DatasetId,
    cells: &[usize],
    config: ScheduledAccessConfig,
) -> DataBankResult<Vec<T>>

pub fn access_cells_owned_by_gene_names<T, G>(
    &self,
    id: DatasetId,
    cells: &[usize],
    gene_names: &[G],
    missing: MissingGenePolicy,
) -> DataBankResult<Vec<T>>
where
    T: DataValue,
    G: AsRef<str>

pub fn access_cells_owned_by_gene_names_with_config<T, G>(
    &self,
    id: DatasetId,
    cells: &[usize],
    gene_names: &[G],
    missing: MissingGenePolicy,
    config: ScheduledAccessConfig,
) -> DataBankResult<Vec<T>>
where
    T: DataValue,
    G: AsRef<str>

pub fn access_cells_alloc<T: DataValue>(
    &self,
    id: DatasetId,
    cells: &[usize],
) -> DataBankResult<Vec<T>>
```

这些方法分配并返回 row-major `Vec<T>`。分配出的 buffer 初始化为 `T::zero()`，所以 sparse 访问和缺失 gene 补零路径可跳过重复清零。`access_cells_alloc` 是 `access_cells_owned` 的 alias。

### Unchecked CSR 热路径

```rust
pub unsafe fn access_cells_unchecked<T: DataValue>(
    &self,
    id: DatasetId,
    cells: &[usize],
    out: &mut [T],
    names: Option<&mut [GeneNameView]>,
) -> DataBankResult<()>

pub unsafe fn access_cells_unchecked_with_config<T: DataValue>(
    &self,
    id: DatasetId,
    cells: &[usize],
    out: &mut [T],
    names: Option<&mut [GeneNameView]>,
    config: ScheduledAccessConfig,
) -> DataBankResult<()>
```

只有 CSR 数据集使用 unchecked 热路径；dense 数据集回退到 checked path。调用方必须保证：

1. `id` 是已注册数据集。
2. 对 CSR，`T` 精确匹配数据集 value dtype。
3. 所有 `cells` 均在范围内。
4. `out.len() == cells.len() * num_genes`。
5. `names` 为 `None` 或长度等于 `num_genes`。
6. CSR gene indices 全部非负且 `< num_genes`。

违反 CSR gene index 约束会导致越界写和内存破坏。实现中仍会做部分 I/O、decode、buffer 长度相关错误传播，但 scatter 主循环跳过核心边界检查。

### 直接 cache prefetch

```rust
pub fn prefetch_cells(&self, id: DatasetId, cells: &[usize]) -> DataBankResult<()>
```

这个方法只把相关文件 chunk key 送入 access layer 的 cache/prefetch，不返回数据。它会规划请求 cell 覆盖的 dense/sparse chunk range，去重 `ChunkKey`，然后等待所有 prefetch 完成。内存 chunk 不会产生 prefetch key。

### Scheduled prefetch

```rust
pub fn prefetch_cells_scheduled<T, I>(
    &self,
    id: DatasetId,
    batch_source: I,
    config: ScheduledPrefetchConfig,
) -> DataBankResult<PrefetchCells<T>>
where
    T: DataValue,
    I: IntoIterator,
    I::IntoIter: Send + 'static,
    I::Item: AsRef<[usize]> + Send

pub fn prefetch_cells_scheduled_by_gene_names<T, I, G>(
    &self,
    id: DatasetId,
    batch_source: I,
    gene_names: &[G],
    missing: MissingGenePolicy,
    config: ScheduledPrefetchConfig,
) -> DataBankResult<PrefetchCells<T>>
where
    T: DataValue,
    I: IntoIterator,
    I::IntoIter: Send + 'static,
    I::Item: AsRef<[usize]> + Send,
    G: AsRef<str>

pub fn prefetch_cells_scheduled_multi<T, I>(
    &self,
    ids: &[DatasetId],
    batch_source: I,
    config: ScheduledPrefetchConfig,
) -> DataBankResult<PrefetchCells<T>>
where
    T: DataValue,
    I: IntoIterator,
    I::IntoIter: Send + 'static,
    I::Item: Into<MultiBatchCells> + Send

pub fn prefetch_cells_scheduled_multi_by_gene_names<T, I, G>(
    &self,
    ids: &[DatasetId],
    batch_source: I,
    gene_names: &[G],
    missing: MissingGenePolicy,
    config: ScheduledPrefetchConfig,
) -> DataBankResult<PrefetchCells<T>>
where
    T: DataValue,
    I: IntoIterator,
    I::IntoIter: Send + 'static,
    I::Item: Into<MultiBatchCells> + Send,
    G: AsRef<str>
```

scheduled prefetch 返回一个 blocking iterator。每个 `batch_source` item 是一个 DataBank batch，不是 chunk batch。一个 DataBank batch 会展开为数量可变的 chunk group，再交给 access scheduler。结果缓存在 DataBank 级 completed queue 中，深度为 `config.prefetch_step`。

多数据集接口中：

1. 不传 `gene_names` 时，所有数据集必须 gene count 和 gene 顺序完全一致。
2. 传 `gene_names` 时，每个数据集独立编译 gene projection；缺失 gene 根据 `MissingGenePolicy` 处理。
3. 每个 batch 通过 `MultiBatchCells` 指明哪些 cell 来自哪个 dataset。

## 其他公共类型

### `DataBankConfig`

```rust
#[derive(Debug, Clone)]
pub struct DataBankConfig {
    pub io_config: IoConfig,
    pub decode_config: DecodePoolConfig,
    pub access_config: AccessConfig,
    pub fill_config: FillConfig,
}

impl Default for DataBankConfig
impl DataBankConfig {
    pub fn validate(&self) -> Result<(), String>
}
```

默认配置使用各子系统默认值，但把 `access_config.keep_decoded` 设为 `false`。`validate` 会递归校验 I/O、decode、access 和 fill config。

### `FillConfig`

```rust
#[derive(Debug, Clone)]
pub struct FillConfig {
    pub parallel: bool,
    pub num_workers: usize,
    pub queue_capacity: usize,
    pub min_parallel_rows: usize,
    pub min_parallel_bytes: usize,
    pub cpus: Option<Vec<usize>>,
}
```

默认：

| 字段 | 默认值 |
| --- | --- |
| `parallel` | `true` |
| `num_workers` | `4` |
| `queue_capacity` | `1024` |
| `min_parallel_rows` | `16` |
| `min_parallel_bytes` | `1 MiB` |
| `cpus` | `None` |

校验规则：

1. `parallel == true` 时 `num_workers > 0`。
2. `parallel == true` 时 `queue_capacity > 0`。
3. `cpus` 不能是空 vec。
4. `cpus` 不能包含重复 CPU id。

### `ScheduledPrefetchConfig`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledPrefetchConfig {
    pub prefetch_step: usize,
    pub access: ScheduledAccessConfig,
}
```

默认 `prefetch_step = 2`，`access = ScheduledAccessConfig::default()`。`prefetch_step` 必须大于 0。`prefetch_step` 表示 DataBank 层最多缓存多少个已经组装好的 decoded batch，不等同于 access scheduler 内部 chunk lookahead。

### 数据集 spec

```rust
#[derive(Debug)]
pub struct Dense1DSpec {
    pub gene_names: Vec<String>,
    pub data: ArraySpec,
}

#[derive(Debug)]
pub struct Dense2DSpec {
    pub gene_names: Vec<String>,
    pub data: ArraySpec,
}

#[derive(Debug)]
pub struct SparseCsrSpec {
    pub gene_names: Vec<String>,
    pub indptr: Vec<u64>,
    pub indices: ArraySpec,
    pub data: ArraySpec,
    pub index_dtype: DType,
    pub num_cells: usize,
    pub num_genes: usize,
}
```

注册时校验：

| 类型 | 关键校验 |
| --- | --- |
| `Dense1D` | `gene_names` 非空；`data.shape == [cells * genes]`；总长度可被 gene 数整除；gene name 不能空。 |
| `Dense2D` | `data.shape == [cells, genes]`；`gene_names.len() == genes`；只支持 regular chunk grid；gene name 不能空。 |
| `SparseCsr` | `indptr.len() == num_cells + 1`；`indptr` 单调非降；`index_dtype` 是 `I32/U32/I64/U64`；`gene_names.len() == num_genes`；`indices.shape == [nnz]`；`data.shape == [nnz]`；`indices.dtype == index_dtype`。 |

### `ArraySpec` 和 chunk 描述

```rust
#[derive(Debug, Clone)]
pub struct ArraySpec {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub order: ArrayOrder,
    pub codec: ArrayCodecSpec,
    pub grid: ArrayGridSpec,
    pub chunks: Vec<ChunkSpec>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayOrder { C }
```

`ArraySpec` 表示一个 C-order 分块数组。`build_array_from_spec` 是内部使用的构建器，会：

1. 校验 shape 非空。
2. 构建 codec pipeline。
3. 根据 grid 计算 chunk 数量和每个 chunk 应有的 decoded bytes。
4. 校验 `chunks.len()` 和每个 `ChunkSpec.decoded_bytes`。
5. 对文件路径去重注册 readonly file。
6. 对 `len == 0` 的 file/registered file chunk 生成全零 decoded memory chunk。

#### `ArrayGridSpec`

```rust
pub enum ArrayGridSpec {
    Regular {
        chunk_shape: Vec<usize>,
        edge: EdgeChunkLayout,
    },
    Rectilinear {
        axes: Vec<Vec<usize>>,
    },
}

pub enum EdgeChunkLayout {
    Padded,
    Cropped,
}
```

Regular grid 要求 `shape.len() == chunk_shape.len()` 且 chunk shape 全部非零。`Padded` 表示边缘 chunk decoded 成完整 `chunk_shape`，`Cropped` 表示边缘 chunk decoded 成实际逻辑范围。

Rectilinear grid 每个轴由 boundaries 描述，要求：

1. axis 数等于 rank。
2. 每个 axis 至少两个 boundary。
3. 第一个 boundary 是 0。
4. 最后一个 boundary 等于对应 shape 维度。
5. boundaries 单调非降。

当前 `Dense2D` 注册显式拒绝 rectilinear grid；1D range planning 支持 regular 和 rectilinear。

#### `ChunkSpec` / `ChunkSourceSpec`

```rust
pub struct ChunkSpec {
    pub source: ChunkSourceSpec,
    pub decoded_bytes: usize,
}

pub enum ChunkSourceSpec {
    Memory { bytes: Arc<[u8]> },
    DecodedMemory { bytes: Arc<[u8]> },
    File { path: PathBuf, offset: u64, len: usize },
    RegisteredFile { file: RegisteredFile, offset: u64, len: usize },
}
```

| Source | 含义 |
| --- | --- |
| `Memory` | 内存中已有 encoded chunk bytes，仍会经过 codec decode。 |
| `DecodedMemory` | 内存中已有 decoded chunk bytes，要求 `bytes.len() == decoded_bytes`。 |
| `File` | DataBank 负责按 path 注册 readonly file，注销/Drop 时释放。 |
| `RegisteredFile` | 调用方已经注册过 file，DataBank 不拥有该注册，因此不会 unregister。 |

### `ArrayCodecSpec`

```rust
pub enum ArrayCodecSpec {
    Uncompressed,
    CodecJson(String),
    CodecJsonValue(serde_json::Value),
    PipelineJson(String),
    PipelineJsonValue(serde_json::Value),
    ZarrV2Json { filters: Option<String>, compressor: Option<String> },
    ZarrV2JsonValue {
        filters: Option<serde_json::Value>,
        compressor: Option<serde_json::Value>,
    },
}
```

`Uncompressed` 构建 identity codec。其他变体会通过 `crate::codecs` 解析单 codec、pipeline 或 Zarr v2 filters/compressor，并不隐含任何 Zarr storage layout。

### `DType`、`DataValue`、半精度 wrapper

```rust
pub enum DType {
    U8, I8, U16, I16, U32, I32, U64, I64, F16, BF16, F32, F64,
}
```

`DType` 方法：

```rust
pub fn item_size(self) -> usize
pub fn is_csr_index(self) -> bool
pub fn is_int(self) -> bool
pub fn is_float(self) -> bool
pub fn is_half(self) -> bool
pub fn can_cast_to(self, dst: DType) -> bool
```

cast 策略很简单：只禁止 float 到 int；允许 float 到 float、int 到 float、int 到 int，包括有精度损失的转换。半精度通过 `half` crate 做 round-to-nearest-even。

```rust
pub trait DataValue: Copy + Send + Sync + 'static {
    const DTYPE: DType;
    fn zero() -> Self;
    fn cast_slice_from(src_bytes: &[u8], src_dtype: DType, dst: &mut [Self])
        -> DataBankResult<()>;
}
```

`DataValue` 是 sealed trait，外部不能为自定义类型实现。内置实现包括：

| Rust 类型 | DType |
| --- | --- |
| `u8` | `U8` |
| `i8` | `I8` |
| `u16` | `U16` |
| `i16` | `I16` |
| `u32` | `U32` |
| `i32` | `I32` |
| `u64` | `U64` |
| `i64` | `I64` |
| `f32` | `F32` |
| `f64` | `F64` |
| `F16Bits` | `F16` |
| `Bf16Bits` | `BF16` |

```rust
#[repr(transparent)]
pub struct F16Bits(pub u16);

#[repr(transparent)]
pub struct Bf16Bits(pub u16);
```

这两个类型保存 native-endian 的 16-bit float bit pattern，不直接暴露浮点值语义。

### `RegisteredFile`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisteredFile {
    pub id: FileId,
    pub file_ref: FileRef,
}

impl RegisteredFile {
    pub fn new(id: FileId) -> DataBankResult<Self>
}
```

用于 `ChunkSourceSpec::RegisteredFile`。`new` 把 `FileId` 转成 access layer 使用的 `FileRef`，如果 `FileId` 不能放进 `u64` 会返回 `InvalidArrayMeta("file id overflow")`。

### `DatasetId`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DatasetId {
    pub slot: u32,
    pub generation: u32,
}
```

`slot` 是 registry 槽位，`generation` 防止 use-after-unregister：槽位复用时 generation 会递增，旧 id 访问会返回 `InvalidDatasetId`。generation 使用 wrapping add，但永远不为 0。

### `GeneNameView`

```rust
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneNameView {
    pub ptr: *const u8,
    pub len: usize,
}

impl GeneNameView {
    pub const fn empty() -> Self
    pub fn is_empty(self) -> bool
}
```

这是 FFI 友好的 UTF-8 字节视图，指向 `Arc<str>` 内部 bytes。`GeneNameView` 被 unsafe 标记为 `Send` / `Sync`，生命周期约束由调用方遵守：对应 dataset unregister 或 DataBank drop 后，指针可能悬垂。

`GeneNameView::empty()` 用于缺失 gene 的补零列，`ptr = null` 且 `len = 0`。

### `MissingGenePolicy`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingGenePolicy {
    #[default]
    Zero,
    Error,
}
```

| Variant | 行为 |
| --- | --- |
| `Zero` | 缺失 gene 输出列保持零值，`GeneNameView::empty()`。 |
| `Error` | 遇到缺失 gene 返回 `GeneNameNotFound`。 |

### `MultiBatchCells`

```rust
#[derive(Debug, Clone, Default)]
pub struct MultiBatchCells;

impl MultiBatchCells {
    pub fn new(parts: Vec<(usize, Vec<usize>)>) -> Self
}
```

用于 scheduled multi-dataset prefetch。`parts` 中每项是 `(dataset_idx, cells)`。构造后内部会把所有 cells 拼成一个 flat vec，并记录每段对应哪个 dataset。输出 batch 的 cell 顺序就是 parts 顺序拼接后的顺序。

### `PrefetchedBatch<T>`

```rust
pub struct PrefetchedBatch<T: DataValue> {
    pub cells: Vec<usize>,
    pub buffer: Vec<T>,
    pub num_genes: usize,
}
```

`buffer` 是 row-major，长度等于 `cells.len() * num_genes`。scheduled prefetch 总是由 DataBank 分配输出 buffer，不接受外部 buffer。

### `PrefetchCells<T>`

```rust
pub struct PrefetchCells<T: DataValue>;

impl<T: DataValue> PrefetchCells<T> {
    pub fn prefetch_step(&self) -> usize
    pub fn gene_names(&self) -> &[GeneNameView]
}

impl<T: DataValue> Iterator for PrefetchCells<T> {
    type Item = DataBankResult<PrefetchedBatch<T>>;
}
```

`next()` 是 blocking receive。Drop 时会：

1. `cancel_all()` 所有 active scheduled access。
2. 关闭接收端。
3. join producer thread。

因此提前丢弃 iterator 会停止后台预取。

### `DataBankError` / `DataBankResult`

```rust
pub type DataBankResult<T> = Result<T, DataBankError>;
```

错误类型：

| Variant | 含义 |
| --- | --- |
| `InvalidConfig(String)` | 配置、输出长度等非数组元数据错误。 |
| `InvalidDatasetId(DatasetId)` | slot 不存在或 generation 不匹配。 |
| `DatasetUnloaded(DatasetId)` | slot/generation 有效但 dataset 已被 remove。 |
| `InvalidArrayMeta(String)` | shape、chunk、grid、dtype metadata 不一致。 |
| `UnsupportedDType { dtype, context }` | 某路径不支持该 dtype，例如 CSR index dtype。 |
| `CannotCast { src, dst, reason }` | dtype cast 被策略拒绝，主要是 float 到 int。 |
| `CellIndexOutOfRange { cell, num_cells }` | 请求 cell 越界。 |
| `GeneIndexOutOfRange { gene, num_genes }` | CSR gene index 或 projected output index 越界。 |
| `GeneNameNotFound { gene }` | `MissingGenePolicy::Error` 时缺失 gene。 |
| `DuplicateGeneName { gene }` | 请求 gene list 中重复。 |
| `BufferSizeMismatch { expected, actual }` | 输出 buffer 或中间 slice 长度不匹配。 |
| `NameBufferSizeMismatch { expected, actual }` | gene name 输出 buffer 长度不匹配。 |
| `IndptrInvalid(String)` | CSR `indptr` 非法。 |
| `CsrIndexInvalid(String)` | CSR index 负数或无法转换为 `usize`。 |
| `Access(AccessError)` | access scheduler 错误。 |
| `Codec(CodecError)` | codec 解析或 decode 错误。 |
| `Io(io::Error)` | I/O 或线程 spawn/join 通信相关错误。 |
| `ComputeWorkerPanic` | compute worker panic 被捕获。 |
| `ComputeShutdown` | compute pool channel 已关闭。 |
| `PrefetchCancelled` | scheduled prefetch 被取消。 |
| `PrefetchProducerPanic` | scheduled producer panic。 |
| `NotImplemented(&'static str)` | 预留未实现。 |

## 内部实现细节

### Registry 和生命周期

`DatasetRegistry` 内部是：

```rust
slots: Vec<DatasetSlot>
free_slots: Vec<u32>
```

每个 slot 存 `generation` 和 `Option<Arc<Dataset>>`。注册时优先复用 `free_slots`，复用会 `next_generation`，新 slot generation 从 1 开始。删除时 `dataset.take()`，slot 放回 free list。

DataBank 注销时不一定马上能释放文件，因为 scheduled prefetch 会通过 `Arc<Dataset>` 保持 dataset 活着。`retired` 列表保存这些已注销但仍被引用的数据集。后续每次 register/unregister 前和 drop 时都会检查 `Arc::strong_count == 1` 的 retired dataset 并释放文件。

### Gene interning 和投影

`GeneInterner` 用 `HashMap<String, InternedGene>` 保存全局字符串池，`InternedGene` 包含 `Arc<str>` 和 refcount。每个 dataset 持有 `DatasetGeneRefs`：

```rust
names: Vec<Arc<str>>
views: Vec<GeneNameView>
by_name: HashMap<Arc<str>, usize>
duplicate_name: Option<Arc<str>>
```

`CompiledGeneProjection` 是按 gene name 投影的核心结构：

```rust
output_by_source: Vec<usize>       // source gene -> output col，未选中为 usize::MAX
output_names: Vec<GeneNameView>    // output col -> gene view
selected_sources: Vec<usize>       // 已选 source gene，下标排序
```

dense projected planner 使用 `selected_sources` 合并连续 source gene run；scatter 时如果一个 segment 在投影后仍是连续 output run，就整段 copy/cast，否则逐元素投影。

### Array 构建和 chunk 引用

内部 `Array` 保存：

```rust
shape: Vec<usize>
dtype: DType
codec: SharedCodec
grid: ArrayGrid
chunks: Vec<Chunk>
files: Vec<RegisteredFile>
```

`ChunkRef` 有两类：

1. `AccessItem`：文件 chunk，通过 `AccessScheduler` 读和 decode。
2. `Memory`：内存 chunk，DataBank 自己直接 decode 或 slice。

`ChunkSourceSpec::File` 会按 path 去重注册，同一个 path 的多个 chunks 共用一个 `RegisteredFile`。`ChunkSourceSpec::RegisteredFile` 不进入 `files`，所以 DataBank 不会释放它。

### 同步访问分发

`direct.rs` 是 `DataBank` 门面后的第一层：

1. `validate_dtype_and_cells` 检查 dtype cast 和 cell 越界。
2. `validate_access` 检查输出 buffer 和 names buffer 长度。
3. 根据 `Dataset` enum 分发：
   - `Dense1D` -> `dense::access_dense_1d`
   - `Dense2D` -> `dense::access_dense_2d`
   - `SparseCsr` -> `sparse::access_sparse`
4. 按 gene name 时先构建 `GeneAxisPlan`，再走 projected dense/sparse path。

### Dense planner 和 scatter

Dense 使用 `DenseSegment` 表示一段要复制到输出矩阵的连续元素：

```rust
output_row: usize
output_col_start: usize
output_cols: usize
chunk: ChunkRef
source: ByteRange
```

`Dense1D` 把每个 cell row 视为 `data[cell * num_genes .. (cell + 1) * num_genes]`，再用 `ArrayGrid::for_each_1d_range` 切成跨 chunk 的 range。

`Dense2D` 要求 regular 2D chunk grid。每个 cell row 按 chunk column 拆成多个 segment，`row_slice_bytes` 计算在 decoded chunk 中的 byte range。`EdgeChunkLayout::Padded` 和 `Cropped` 通过 `physical_row_width_2d` 处理边缘 chunk 的物理行宽。

Dense 读取前会按 chunk 分组：

1. `dense_group_key` 用文件 key 或内存 ptr/len/codec/expected_size/decoded 去重。
2. `append_group_slice` 把多个 segment 的 source byte range 合并成 `SliceSpec`。
3. 文件 group 用 `access.scheduled(...)` 顺序拉取。
4. 内存 group 用 `load_memory_group` 本地 decode/slice。

Scatter 路径：

1. 如果输出够大且 compute pool 启用，先加载 groups，然后按输出行并行 scatter。
2. 全文件 group 走 scheduled 读取，边读边 scatter。
3. 混合内存/file group 走 sequential。
4. `Dense1D` 存在 memory identity 直接 memcpy 快路径：数据必须全是内存 chunk、codec identity 或已 decoded，并且源 dtype 精确等于 `T::DTYPE`。

### Sparse CSR planner 和 scatter

CSR 数据集包含：

```rust
indptr: Vec<u64>
indices: Array
data: Array
index_dtype: DType
num_cells: usize
num_genes: usize
```

访问 sparse 时输出 buffer 会先清零。普通 path：

1. `plan_sparse_rows` 用 `indptr[cell]..indptr[cell+1]` 得到每行 nnz span。
2. `plan_sparse_batch` 分别规划 indices 和 data 的 1D ranges。
3. `SparsePieceGroupBuilder` 按 chunk source 合并 `SparseReadPiece` 到 `SparseReadGroup`。
4. 先加载所有 index groups 到连续 `index_bytes`。
5. 再加载 data groups 并根据 indices scatter 到输出 row。

`SparseReadPiece` 记录：

```rust
chunk: ChunkRef
source: ByteRange
group_offset: usize
output_offset: usize
output_row: usize
index_offset: usize
elements: usize
bytes: usize
```

其中 index pieces 用 `output_offset` 指向 `index_bytes`，data pieces 用 `output_row` 和 `index_offset` 找到对应 row 和 index slice。

CSR index 支持 `u32/i32/u64/i64`，通过内部 `CsrIndex` trait 做 checked/unchecked gene 转换。checked path 会拒绝负数和超出 `usize` / `num_genes` 的 gene index。

Sparse 快路径：

1. `single_memory_identity_chunk_bytes`：indices 和 data 都是单个内存 identity/decoded chunk 时，直接用 `indptr` 对原始 bytes scatter。
2. `MemoryIdentity1DChunks`：indices 和 data 是多个 regular 1D 内存 identity/decoded chunk 时，按 chunk 边界逐段 scatter。
3. 文件 backed plan 中 indices 和 data group 可合并到同一个 scheduled reader，先读 indices 后读 data。
4. projected sparse 会先扫描 `index_bytes`，找出包含选中 gene 的 data groups，只加载必要 data groups。

### Scheduled prefetch pipeline

scheduled prefetch 由一个 producer thread 驱动，核心状态是：

```rust
next_read_seq: BatchSeq
next_emit_seq: BatchSeq
source_done: bool
stop_reading: bool
outstanding: usize
active_requests: usize
active_responses: usize
response_limit: usize
planned_ready: VecDeque<PlannedBatch>
completed: CompletedQueue<T>
```

流水线分两种 compute job：

1. request job：读取一个 batch，`plan_batch_multi` 生成 `BatchPlan` 和文件 `AccessItem` 列表，创建 `ScheduledAccess`，注册 cancel handle。
2. response job：消费 `ScheduledAccess` 的输出 bytes，调用 `assemble_planned_batch` 组装 `PrefetchedBatch<T>`。

producer 主循环反复执行：

1. `fill_request_window`：当 outstanding 小于 `prefetch_step` 时从 `batch_source.next()` 读新 batch，提交 request job。
2. `drain_messages`：收集 request 完成的 `PlannedBatch` 和 response 完成的 `PrefetchedBatch`。
3. `submit_ready_responses`：在 `response_limit` 内把 planned batch 提交组装。
4. `emit_ready`：按 `next_emit_seq` 顺序从 completed queue 取结果发送给 `PrefetchCells` iterator。

结果可能乱序完成，但一定按 batch source 原顺序 emit。`prefetch_step <= 32` 时 completed queue 使用小 Vec，超过后使用 `BTreeMap`。

`response_limit = prefetch_step.min(worker_count.saturating_sub(1).max(1))`，避免 response job 把 compute workers 全占满导致 request planning 饥饿。

Drop `PrefetchCells` 或任一不可恢复错误会调用 `PrefetchCancelRegistry::cancel_all()`，取消 active scheduled access，并停止继续读 batch source。

### 多数据集 batch 布局

`MultiBatchCells` 允许一个 output batch 混合多个 dataset 的 cell。planner 会：

1. 按 parts 顺序拼出 `output_cells`。
2. 按 dataset_idx 聚合 cells 和它们应该写入的 output row。
3. 如果实际只涉及一个 dataset 且 output row 是顺序的，降级为 single plan。
4. 否则每个 dataset 生成一个 `MultiBatchPlanPart`，组装时 scatter 到共享 buffer 的指定 output rows。

这保证 `PrefetchedBatch.cells` 的顺序与调用方传入 batch parts 的拼接顺序一致。

### Compute pool

`DataBankComputePool` 只服务 DataBank 内部 CPU 工作，不负责 I/O。它有两个队列：

| 队列 | 用途 |
| --- | --- |
| request | scheduled prefetch 的 batch planning。 |
| response | scheduled prefetch 的 batch assembling，以及同步访问的并行 scatter job。 |

worker loop 优先处理 response，再处理 request。这样已经读到数据、准备返回的 batch 不会被大量新 request planning 卡住。

`should_parallelize(rows, bytes)` 要求：

1. parallel pool 已启用。
2. 当前线程不是 DataBank worker。
3. `rows >= min_parallel_rows`。
4. `bytes >= min_parallel_bytes`。

如果在 DataBank worker 中调用 `run_jobs` 或 `submit_*`，会 inline 执行，避免 worker 等自己造成死锁。可选 `FillConfig.cpus` 会用 `core_affinity` pin worker。

### dtype cast 和 zero fill

所有 dense/sparse scatter 最终都走 `T::cast_slice_from(src_bytes, src_dtype, dst)`：

1. 源 dtype 和目标 dtype 相同时走 raw byte copy。
2. int/float/half 转换按元素转换。
3. float 到 int 返回 `CannotCast`。

`zero_values<T>` 用 `ptr::write_bytes` 清零，依赖 `DataValue` sealed 到数值 primitive 和 transparent half wrappers，这些类型的零值都是全零 bit pattern。

### Access scheduler 交互

文件 chunk 统一转成 `AccessItem`：

```rust
AccessItem::new(
    ChunkKey::new(file_ref, offset, len),
    codec,
    Some(decoded_bytes),
)
```

group 级访问会把 `SliceSpec` 改成多个 `RangeCopy`，让 access layer 只返回需要的 decoded bytes。内存 chunk 不能进入 file scheduled path，相关函数会返回 `InvalidArrayMeta` 或 `unreachable!`。

`load_memory_group` 的逻辑是：

1. 如果 chunk 已 decoded 或 codec identity，校验 `bytes.len() == expected_size` 后直接 slice。
2. 否则先 `codec.decode(bytes, Some(expected_size))`，再按 `SliceSpec` copy。

`copy_slices` 会识别目标 ranges 是否 packed；packed 时用 `Vec::with_capacity + extend_from_slice`，非 packed 时先分配全零输出再按 offset copy。

## 常见使用形态

```rust
use scdata::databank::{
    ArrayCodecSpec, ArrayGridSpec, ArrayOrder, ArraySpec, ChunkSourceSpec, ChunkSpec,
    DataBank, DataBankConfig, Dense2DSpec, DType, EdgeChunkLayout,
};

let mut bank = DataBank::new(DataBankConfig::default())?;
let id = bank.register_dense_2d(Dense2DSpec {
    gene_names,
    data: ArraySpec {
        shape: vec![num_cells, num_genes],
        dtype: DType::F32,
        order: ArrayOrder::C,
        codec: ArrayCodecSpec::Uncompressed,
        grid: ArrayGridSpec::Regular {
            chunk_shape: vec![1024, num_genes],
            edge: EdgeChunkLayout::Cropped,
        },
        chunks,
    },
})?;

let cells = vec![0, 10, 20];
let mut out = vec![0.0f32; cells.len() * bank.dataset_num_genes(id)?];
bank.access_cells_values(id, &cells, &mut out)?;
```

按 gene name 投影：

```rust
use scdata::databank::MissingGenePolicy;

let genes = ["Actb", "Gapdh", "MissingGene"];
let mut out = vec![0.0f32; cells.len() * genes.len()];
bank.access_cells_by_gene_names(
    id,
    &cells,
    &genes,
    &mut out,
    None,
    MissingGenePolicy::Zero,
)?;
```

scheduled prefetch：

```rust
use scdata::databank::ScheduledPrefetchConfig;

let batches = vec![vec![0, 1, 2], vec![100, 101, 102]];
let iter = bank.prefetch_cells_scheduled::<f32, _>(
    id,
    batches,
    ScheduledPrefetchConfig::default(),
)?;

for batch in iter {
    let batch = batch?;
    assert_eq!(batch.buffer.len(), batch.cells.len() * batch.num_genes);
}
```

## 需要特别注意的约束

1. `GeneNameView` 是裸 pointer/len 视图，不拥有字符串，dataset unregister 后可能悬垂。
2. `DataValue` 的 cast 策略允许精度损失，但禁止 float 到 int。
3. `Dense2D` 当前只支持 regular chunk grid。
4. `SparseCsr` 的 `indptr` 在注册时只校验长度和单调性；checked scatter 会在访问时校验 gene index，unchecked scatter 不校验。
5. `prefetch_cells_scheduled*` 会持有 `Arc<Dataset>`，所以注销 dataset 后，文件释放可能延迟到 iterator drop。
6. `ChunkSourceSpec::RegisteredFile` 的文件生命周期由调用方负责。
7. scheduled prefetch 的 batch plan 不跨 batch 合并 chunk，优化边界是单个 batch 内部。
8. 内部并行 scatter 只在同步访问路径触发；scheduled assemble 运行在 compute worker 中，依赖 inter-batch 并行而不是单 batch 内再嵌套并行。
