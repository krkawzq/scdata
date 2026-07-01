# `codecs` 模块说明

`crate::codecs` 是 scdata Rust 内核里的 chunk 解码层。它负责把 zarr/numcodecs 风格的 codec 元数据解析成可共享的解码器，将单个或多个 codec 组合成顺序解码 pipeline，并提供一个跨 chunk 的 worker pool 用于异步解码。

这个模块目前只面向解码路径，没有对外提供编码 API。`CodecSpec` 里会保留一些 numcodecs 的压缩参数，例如 `level`、`clevel`、`checksum`、`acceleration`、`preset`，但这些参数大多只影响写入/编码端；解码器构建和缓存时会主动忽略不影响解码行为的字段。

## 目录结构

```text
codecs/
  mod.rs                 public re-export 和便捷构造函数
  spec.rs                codec 元数据、trait 契约、JSON 解析
  registry.rs            全局 codec/pipeline 缓存
  pipeline.rs            多 stage 顺序解码 pipeline
  pool.rs                解码 worker pool、请求、future
  profile.rs             codecs 组件 profiling
  runner.rs              内部 decode fast-path 调度
  buffer.rs              caller-owned 输出 buffer 和 Vec 复用工具
  util.rs                错误构造、size 校验、JSON 字段解析
  impls/
    identity.rs          `none` / uncompressed codec
    unsupported.rs       unknown codec 占位实现
    crc32.rs             crc32 校验过滤器
    lz4.rs               numcodecs lz4 解码
    zstd.rs              zstd 解码
    stream.rs            gzip/zlib/bz2/lzma streaming 解码
    blosc/               blosc 解码和 Blosc-LZ4 快路径
```

## 模块职责

`codecs` 位于 IO 与数组 materialization 之间。上游从文件或内存读到 compressed chunk bytes，下游需要 decoded bytes。这个模块提供三类能力：

1. 元数据解析：把 zarr/numcodecs JSON 解析成 `CodecSpec` 或 `Vec<CodecSpec>`。
2. 解码执行：通过 `ChunkCodec` trait 把 encoded bytes 解码到新 `Vec<u8>` 或 caller-owned buffer。
3. 并行调度：通过 `DecodePool` 在线程池中执行跨 chunk 解码，并记录 profiling 指标。

crate 内典型路径：

```text
ArrayCodecSpec / zarr metadata
  -> CodecSpec 或 Vec<CodecSpec>
  -> SharedCodec = Arc<dyn ChunkCodec>
  -> AccessScheduler / DataBank
  -> DecodePool::submit_async
  -> worker thread
  -> Vec<u8> decoded chunk
```

## 公开接口总览

`src/lib.rs` 中有 `pub mod codecs;`，因此 `mod.rs` 里 `pub use` 的符号是 crate 的公开 API。

### 类型别名

```rust
pub type SharedCodec = Arc<dyn ChunkCodec>;
```

共享 codec 实现。所有构造函数最终返回这个类型，便于缓存、跨线程提交和多个 chunk 复用同一个 decoder。

### 公开类型的常见 trait impl

这些 trait impl 也是外部代码可依赖的接口：

| 类型 | 公开/派生 trait |
|---|---|
| `CodecError` | `Debug`、`thiserror::Error`、`Display`。 |
| `CodecCacheKey` | `Debug`、`Clone`、`PartialEq`、`Eq`、`Hash`。 |
| `CodecSpec` | `Debug`、`Clone`、`PartialEq`、`Eq`、`Hash`。 |
| `BloscCodecConfig` | `Debug`、`Clone`、`PartialEq`、`Eq`、`Hash`。 |
| `BloscShuffle` | `Debug`、`Clone`、`Copy`、`PartialEq`、`Eq`、`Hash`。 |
| `ZstdCodecConfig` | `Debug`、`Clone`、`PartialEq`、`Eq`、`Hash`、`Default`。 |
| `LevelCodecConfig` | `Debug`、`Clone`、`PartialEq`、`Eq`、`Hash`、`Default`。 |
| `Lz4CodecConfig` | `Debug`、`Clone`、`PartialEq`、`Eq`、`Hash`、`Default`。 |
| `LzmaCodecConfig` | `Debug`、`Clone`、`PartialEq`、`Eq`、`Hash`、`Default`。 |
| `DecodeBuffer` | `Debug`。 |
| `CodecPipeline` | `Debug`、`Clone`、`ChunkCodec`。 |
| `DecodePoolConfig` | `Debug`、`Clone`、`Default`。 |
| `DecodeRequest` | `Debug`。 |
| `DecodeOutput` | `Debug`。 |
| `DecodeFuture` | `Debug`、`Future<Output = CodecResult<Vec<u8>>>`。 |
| `CodecProfile` | `Debug`、`Clone`、`Default`。 |
| `UncompressedCodec` | `Debug`、`Default`、`ChunkCodec`。 |
| `UnsupportedCodec` | `Debug`、`ChunkCodec`。 |

### 顶层构造函数

```rust
pub fn codec_from_spec(spec: &CodecSpec) -> SharedCodec
```

从已解析的 `CodecSpec` 构建共享 codec。内部走全局缓存。

```rust
pub fn codec_from_json_str(json: &str) -> CodecResult<SharedCodec>
```

解析单个 codec JSON 对象并构建共享 codec。`null`、`{"id":"none"}`、`{"id":"null"}` 都表示 uncompressed。

```rust
pub fn codec_pipeline_from_specs(specs: &[CodecSpec]) -> SharedCodec
```

从已经处于“解码顺序”的 specs 构建 pipeline。空数组表示 identity pipeline；单 stage 会直接返回该 stage 的 cached codec。

```rust
pub fn codec_pipeline_from_zarr_v2_specs(
    filters: &[CodecSpec],
    compressor: Option<&CodecSpec>,
) -> SharedCodec
```

从 zarr v2 metadata 顺序构建 decode pipeline。zarr v2 写入顺序通常是 filters 后 compressor；读取解码时必须先跑 compressor，再按 metadata 的反序跑 filters。

```rust
pub fn codec_pipeline_from_zarr_v2_json_str(
    filters_json: Option<&str>,
    compressor_json: Option<&str>,
) -> CodecResult<SharedCodec>
```

字符串版本的 zarr v2 pipeline 构造器。`filters_json` 必须是 JSON array 或 `null`；`compressor_json` 必须是单个 JSON object 或 `null`。

### `CodecError` 和 `CodecResult`

```rust
pub type CodecResult<T> = Result<T, CodecError>;
```

`CodecError` 变体：

| 变体 | 含义 |
|---|---|
| `Unsupported { codec }` | 解析到了未知 codec，或显式创建了 unsupported codec，真正解码时报错。 |
| `InvalidCodecConfig { codec, message }` | JSON 配置类型错误、缺少字段、或当前实现不支持的配置。 |
| `Decode { codec, message }` | 底层解码器、校验、header 解析或内存 reserve 失败。 |
| `SizeMismatch { codec, expected, actual }` | 解码结果大小与 caller 传入的 `expected_size` 不一致。 |
| `OutputTooSmall { codec, required, capacity }` | caller-owned 输出 buffer 或复用 `Vec` 容量不足。 |
| `InvalidConfig(String)` | `DecodePoolConfig` 等运行时配置非法。 |
| `QueueFull { capacity }` | `DecodePool::try_submit` 遇到有界队列已满。 |
| `Shutdown` | pool 已关闭、worker channel 断开、future 被重复接收等。 |
| `WorkerPanic { codec }` | worker 线程捕获到 codec panic。 |
| `ThreadSpawn(io::Error)` | 创建 decode worker 线程失败。 |

### `ChunkCodec` trait

```rust
pub trait ChunkCodec: sealed::Sealed + Send + Sync + fmt::Debug + 'static
```

这是所有 decoder 的统一契约。它是 sealed trait：模块外不能新增实现，只能使用模块提供的 codec 或在 `codecs` 模块内部扩展新实现。

公开方法：

```rust
fn name(&self) -> &str
```

返回 codec 名称，用于 pipeline 命名、错误信息和 profiling。

```rust
fn cache_key(&self) -> CodecCacheKey
```

返回 decode-cache identity。默认是对象地址；内置 codec 会返回稳定 key。这个方法标了 `#[doc(hidden)]`，但仍是 trait 的公开方法。

```rust
fn decode(&self, encoded: &[u8], expected_size: Option<usize>) -> CodecResult<Vec<u8>>
```

解码到新的 owned `Vec<u8>`。`expected_size` 是最终 decoded bytes 数，如果提供则必须精确匹配。

```rust
fn decode_into(
    &self,
    encoded: &[u8],
    output: DecodeBuffer<'_>,
    expected_size: Option<usize>,
) -> CodecResult<usize>
```

解码到 caller 提供的 mutable slice，返回写入字节数。默认实现先 `decode` 再 copy；内置高性能实现会直接写入。

`#[doc(hidden)]` 的公开 fast-path 方法：

| 方法 | 用途 |
|---|---|
| `decode_owned(encoded: Vec<u8>, expected_size)` | 输入已经是 owned `Vec` 时，允许 codec 原地复用输入。例如 `none` 直接返回输入，`crc32` 会 `copy_within(4.., 0)` 去掉 checksum header。 |
| `decode_to_vec(encoded, output, expected_size)` | 使用 `output` 的现有 capacity，容量不足时报 `OutputTooSmall`，不增长。 |
| `decode_to_vec_grow(encoded, output, expected_size)` | 尽量复用 `output`，必要时 `try_reserve_exact` 增长。 |
| `decode_to_capacity_vec(encoded, output, expected_size)` | 把 `output.capacity()` 当可写空间，适合 `Vec::with_capacity(size)`。 |
| `decoded_size_hint(encoded, expected_size)` | 尽量返回 decoded size。`lz4`、`blosc`、`crc32` 通常可从 header 得到；`zstd` 可从 frame 或 `expected_size` 得到；streaming codec 常返回 `expected_size` 或 `None`。 |
| `encoded_size_hint(decoded_size)` | 从 decoded size 推导上一 stage 的 encoded size，用于 pipeline 中间 stage 的 size check。仅 identity、crc32、pipeline 等少数 codec 可精确返回。 |
| `prefers_decode_owned()` | 告诉 pipeline/runner 在输入 owned 时优先调用 `decode_owned`。 |
| `is_identity()` | 标识 identity codec，可让 pipeline 跳过实际 copy，同时保留 size check。 |

### `CodecSpec`

```rust
pub enum CodecSpec {
    None,
    Blosc(BloscCodecConfig),
    Zstd(ZstdCodecConfig),
    Gzip(LevelCodecConfig),
    Zlib(LevelCodecConfig),
    Lz4(Lz4CodecConfig),
    Bz2(LevelCodecConfig),
    Lzma(LzmaCodecConfig),
    Crc32,
    Unknown(String),
}
```

方法：

```rust
pub fn name(&self) -> &str
pub fn build(&self) -> SharedCodec
pub fn from_json_str(json: &str) -> CodecResult<Self>
pub fn from_json_value(value: &serde_json::Value) -> CodecResult<Self>
```

JSON 解析规则：

| JSON `id` / `name` | `CodecSpec` |
|---|---|
| `null` value | `None` |
| `"none"` 或 `"null"` | `None` |
| `"blosc"` | `Blosc(BloscCodecConfig)` |
| `"zstd"` | `Zstd(ZstdCodecConfig)` |
| `"gzip"` | `Gzip(LevelCodecConfig)` |
| `"zlib"` | `Zlib(LevelCodecConfig)` |
| `"lz4"` | `Lz4(Lz4CodecConfig)` |
| `"bz2"` | `Bz2(LevelCodecConfig)` |
| `"lzma"` | `Lzma(LzmaCodecConfig)` |
| `"crc32"` | `Crc32` |
| 其他 | `Unknown(lowercase_id)` |

配置对象必须是 JSON object 或 `null`。object 里必须有 string 类型的 `id` 或 `name` 字段，匹配时会转成 lowercase。

### 多 codec JSON parser

```rust
pub fn codec_specs_from_json_str(json: &str) -> CodecResult<Vec<CodecSpec>>
pub fn codec_specs_from_json_value(value: &serde_json::Value) -> CodecResult<Vec<CodecSpec>>
```

用于 zarr filters/pipeline。`null` 解析为空 `Vec`；array 中每个元素按单个 codec config 解析；其他类型报 `InvalidCodecConfig { codec: "filters", ... }`。

### codec 配置结构

```rust
pub struct BloscCodecConfig {
    pub cname: String,
    pub clevel: Option<u8>,
    pub shuffle: Option<BloscShuffle>,
    pub typesize: Option<usize>,
    pub blocksize: Option<usize>,
}

impl BloscCodecConfig {
    pub fn new(cname: impl Into<String>) -> Self
}
```

`cname` 默认 `"lz4"`。当前 decode 实现不根据 `cname` 选择路径；Blosc 真正的压缩格式来自 encoded buffer header。`clevel`、`shuffle`、`typesize`、`blocksize` 主要是 encode metadata，解析后保留。

```rust
pub enum BloscShuffle {
    NoShuffle,
    Shuffle,
    BitShuffle,
}
```

`shuffle` JSON 支持数字 `0/1/2`、bool，以及字符串：`none`、`noshuffle`、`no_shuffle`、`shuffle`、`byte`、`bitshuffle`、`bit_shuffle`。

```rust
pub struct ZstdCodecConfig {
    pub level: Option<i32>,
    pub checksum: Option<bool>,
}

pub struct LevelCodecConfig {
    pub level: Option<i32>,
}

pub struct Lz4CodecConfig {
    pub acceleration: Option<i32>,
}

pub struct LzmaCodecConfig {
    pub format: i32,
    pub check: Option<i32>,
    pub preset: Option<u32>,
    pub has_filters: bool,
}
```

`ZstdCodecConfig`、`LevelCodecConfig`、`Lz4CodecConfig` 都实现 `Default`。`LzmaCodecConfig::default()` 是 `format = 1`、无 `check`、无 `preset`、`has_filters = false`。

`LzmaCodecConfig::format` 的当前 decode 语义：

| format | 含义 |
|---|---|
| `0` | auto decoder，支持 XZ 和 LZMA-alone。 |
| `1` | XZ stream。 |
| `2` | LZMA-alone。 |
| `3` | raw LZMA stream，当前不支持。 |
| 其他 | 当前不支持。 |

如果 JSON 里 `filters` 存在且不是 `null`，`has_filters = true`，当前会在解码时报 `InvalidCodecConfig`，因为 raw LZMA filter chains 尚未实现。

### `CodecCacheKey`

```rust
pub enum CodecCacheKey {
    Static(&'static str),
    Lzma { format: i32, has_filters: bool },
    Unsupported(String),
    Pipeline(Vec<CodecCacheKey>),
    Object(usize),
}
```

用于 access 层 decoded-cache key。内置 codec 返回稳定 key；自定义对象默认会退化成对象地址 key。由于 `ChunkCodec` 被 sealed，正常业务路径不会从模块外构造新实现。

### `CodecPipeline`

`CodecPipeline` 是一个 `ChunkCodec` 实现，用于按顺序执行多个 stage。

构造和查询方法：

```rust
pub fn new(codecs: Vec<SharedCodec>) -> Self
pub fn from_specs(specs: &[CodecSpec]) -> Self
pub fn from_zarr_v2_specs(filters: &[CodecSpec], compressor: Option<&CodecSpec>) -> Self
pub fn from_json_array_str(json: &str) -> CodecResult<Self>
pub fn from_zarr_v2_json_values(
    filters: Option<&serde_json::Value>,
    compressor: Option<&serde_json::Value>,
) -> CodecResult<Self>
pub fn into_shared(self) -> SharedCodec
pub fn is_empty(&self) -> bool
pub fn len(&self) -> usize
```

命名规则：

| stage 数 | `name()` |
|---|---|
| 0 | `"identity"` |
| 1 | 单个 codec 名称 |
| N | 用 `|` 拼接，例如 `"zstd|crc32"` |

zarr v2 decode 顺序示例：

```text
metadata:
  filters = [crc32]
  compressor = zstd

decode pipeline:
  zstd -> crc32
```

### `DecodeBuffer`

```rust
pub struct DecodeBuffer<'a>
```

对 caller-owned `&'a mut [u8]` 的轻量 wrapper。

公开方法：

```rust
pub fn new(bytes: &'a mut [u8]) -> Self
pub fn capacity(&self) -> usize
```

内部方法会用 `ensure_capacity` 防止越界写，容量不足返回 `CodecError::OutputTooSmall`。

### `DecodePoolConfig`

```rust
pub struct DecodePoolConfig {
    pub num_workers: usize,
    pub queue_capacity: usize,
    pub cpus: Option<Vec<usize>>,
}
```

默认值：

```rust
num_workers = 4
queue_capacity = 1024
cpus = None
```

方法：

```rust
pub fn validate(&self) -> CodecResult<()>
pub fn worker_count(&self) -> usize
pub fn queue_capacity(&self) -> usize
```

校验规则：

| 字段 | 规则 |
|---|---|
| `num_workers` | 必须大于 0。 |
| `queue_capacity` | 必须大于 0。 |
| `cpus` | 如果提供，必须非空、不能重复，且每个 CPU id 必须能被 `core_affinity::get_core_ids()` 找到。 |

`worker_count()` 和 `queue_capacity()` 会返回至少 1 的值，但 `new` 前仍会调用严格的 `validate()`。

### `DecodeRequest` 和 `DecodeOutput`

```rust
pub struct DecodeRequest {
    pub codec: SharedCodec,
    pub encoded: Arc<[u8]>,
    pub expected_size: Option<usize>,
    pub output: DecodeOutput,
}
```

构造方法：

```rust
pub fn new(codec: SharedCodec, encoded: impl Into<Arc<[u8]>>) -> Self
pub fn from_spec(spec: &CodecSpec, encoded: impl Into<Arc<[u8]>>) -> Self
pub fn with_expected_size(self, expected_size: usize) -> Self
pub fn with_reuse_initialized_output(self, output: Vec<u8>) -> Self
pub fn with_reuse_capacity_output(self, output: Vec<u8>) -> Self
```

`DecodeOutput`：

```rust
pub enum DecodeOutput {
    Allocate,
    ReuseInitialized(Vec<u8>),
    ReuseCapacity(Vec<u8>),
}
```

| 策略 | 行为 |
|---|---|
| `Allocate` | worker 创建/增长新的输出 `Vec`。 |
| `ReuseInitialized(Vec<u8>)` | 只把 `Vec` 当前 `len()` 范围作为可写 buffer，绝不暗中增长。 |
| `ReuseCapacity(Vec<u8>)` | 当 decoded size 可知时，把 `Vec` 的 capacity 作为可写空间，适合传 `Vec::with_capacity(size)`。 |

### `DecodeFuture`

`DecodePool` 提交后返回的 future。

```rust
pub fn blocking_recv(self) -> CodecResult<Vec<u8>>
```

也实现了 `Future<Output = CodecResult<Vec<u8>>>`，因此异步代码可以直接 `.await`。

### `DecodePool`

跨 chunk 解码线程池。

构造方法：

```rust
pub fn new(config: DecodePoolConfig) -> CodecResult<Self>
pub fn with_profile(config: DecodePoolConfig, profile: ProfileRuntime) -> CodecResult<Self>
pub fn with_codec_profile(config: DecodePoolConfig, profile: CodecProfile) -> CodecResult<Self>
```

profiling 方法：

```rust
pub fn profiler(&self) -> &CodecProfile
pub fn profile(&self) -> &ProfileRuntime
pub fn profile_snapshot(&self) -> ProfileSnapshot
pub fn profile_snapshot_and_reset(&self) -> ProfileSnapshot
pub fn reset_profile(&self)
```

提交方法：

```rust
pub fn submit(&self, request: DecodeRequest) -> CodecResult<DecodeFuture>
pub fn try_submit(&self, request: DecodeRequest) -> CodecResult<DecodeFuture>
pub async fn submit_async(&self, request: DecodeRequest) -> CodecResult<DecodeFuture>
```

| 方法 | 队列满时行为 |
|---|---|
| `submit` | 阻塞当前线程，直到有界队列接受 work。 |
| `try_submit` | 不等待；满队列返回 `CodecError::QueueFull { capacity }`。 |
| `submit_async` | 先 `try_send`，满时 await `send_async`，await 完只表示任务入队成功，不表示解码完成。 |

`DecodePool` drop 时会关闭发送端，然后 join 所有 worker 线程。

### profiling API

公开符号：

```rust
pub const CODECS_COMPONENT: ProfileComponent
pub const CODECS_SUBMIT_SCOPE: ProfileScope
pub const CODECS_WORK_SCOPE: ProfileScope

pub struct CodecProfile
pub fn codecs_profile_registry() -> ProfileRegistry
```

`CodecProfile` 方法：

```rust
pub fn disabled() -> Self
pub fn enabled(label: impl Into<String>) -> Self
pub fn from_env() -> Self
pub fn from_runtime(runtime: ProfileRuntime) -> Self
pub fn runtime(&self) -> &ProfileRuntime
pub fn into_runtime(self) -> ProfileRuntime
pub fn start(&self) -> ProfileRound<'_>
pub fn snapshot(&self) -> ProfileSnapshot
pub fn snapshot_and_reset(&self) -> ProfileSnapshot
pub fn reset_metrics(&self)
```

内部注册的指标：

| scope | metric | 含义 |
|---|---|---|
| `codecs.submit` | `calls` | submit 总次数。 |
| `codecs.submit` | `blocking` / `try` / `async` | 不同 submit 方式次数。 |
| `codecs.submit` | `errors` | submit 阶段错误。 |
| `codecs.submit` | `queue-wait` | 从入队到 worker 开始处理的等待时间。 |
| `codecs.work` | `calls` | worker 处理次数。 |
| `codecs.work` | `work` | worker 解码耗时。 |
| `codecs.work` | `encoded` | 输入 bytes。 |
| `codecs.work` | `decoded` | 成功输出 bytes。 |
| `codecs.work` | `errors` | codec 返回错误次数，不含 panic。 |
| `codecs.work` | `panics` | codec panic 次数。 |

`queue-wait` 只在 submit 和 work 处于同一个 profiling round 时记录，避免跨 round 的等待时间污染新一轮统计。

### 公开 codec 实现

```rust
pub struct UncompressedCodec;
pub struct UnsupportedCodec;

impl UnsupportedCodec {
    pub fn new(name: impl Into<String>) -> Self
}
```

`UncompressedCodec` 对应 `CodecSpec::None`，名称为 `"none"`。它是 identity codec，支持 owned 输入原样返回、输出 buffer 直接 copy、size hint 精确等 fast path。

`UnsupportedCodec` 用于未知 codec。它保留 codec 名称，任何 `decode` 调用都会返回 `CodecError::Unsupported { codec }`。

## 内部实现细节

### JSON 配置解析

`CodecSpec::from_json_value` 接受 JSON object 或 `null`。object 中优先读 `id`，没有则读 `name`。字段解析都通过 `util.rs` 中的 typed helper 完成：

| helper | 规则 |
|---|---|
| `optional_string` | 缺失或 `null` => `None`，否则必须是 string。 |
| `optional_bool` | 缺失或 `null` => `None`，否则必须是 bool。 |
| `optional_i32` | 缺失或 `null` => `None`，否则必须是 i64 且落在 i32 范围。 |
| `optional_u32` | 缺失或 `null` => `None`，否则必须是非负整数且落在 u32 范围。 |
| `optional_u8` | 缺失或 `null` => `None`，否则必须是非负整数且落在 u8 范围。 |
| `optional_usize` | 缺失或 `null` => `None`，否则必须是非负整数且能转成 usize。 |
| `optional_blosc_shuffle` | 支持 numcodecs 常见的数字、bool、字符串形式。 |

未知 codec 不会在解析时失败，而是得到 `CodecSpec::Unknown(id)`。这使 metadata 可以被读取和缓存，真正需要解码时再返回 `Unsupported`。

### codec 和 pipeline 缓存

`registry.rs` 有两个全局缓存：

```text
CODEC_CACHE: OnceLock<RwLock<HashMap<CodecKey, SharedCodec>>>
PIPELINE_CACHE: OnceLock<RwLock<HashMap<Vec<CodecKey>, SharedCodec>>>
```

缓存 key 会规范化掉 encode-only 参数：

| CodecSpec | 缓存 key |
|---|---|
| `None` | `None` |
| `Blosc(_)` | `Blosc` |
| `Zstd(_)` | `Zstd` |
| `Gzip(_)` | `Gzip` |
| `Zlib(_)` | `Zlib` |
| `Lz4(_)` | `Lz4` |
| `Bz2(_)` | `Bz2` |
| `Lzma(config)` | `Lzma { format, has_filters }` |
| `Crc32` | `Crc32` |
| `Unknown(name)` | `Unknown(name)` |

因此两个不同的 `ZstdCodecConfig { level: Some(1) }` 和 `{ level: Some(19) }` 会复用同一个 decoder。`Lzma` 例外：`format` 和 `has_filters` 会改变解码器行为，所以保留在 key 里；`check` 和 `preset` 被认为是 encode-only。

缓存读取先走 read lock；miss 后构建，再用 write lock 插入。若 `RwLock` 被 poison，会用 `into_inner()` 继续工作。

pipeline 缓存同样使用规范化后的 `Vec<CodecKey>`。如果只有一个 stage，直接返回该 stage 的 codec，避免额外包一层 pipeline。

### pipeline 解码顺序和 size propagation

`CodecPipeline::new` 接受的 `codecs` 必须已经是解码顺序。`from_zarr_v2_specs` 会帮 zarr v2 metadata 重新排序：

```text
decode_order = [compressor] + reverse(filters)
```

pipeline 内部维护：

| 字段 | 用途 |
|---|---|
| `codecs` | stage 列表。 |
| `name` | 拼接后的名称。 |
| `cache_key` | `CodecCacheKey::Pipeline(...)`。 |
| `is_identity` | 所有 stage 都是 identity 时为 true。 |

如果 caller 提供 `expected_size`，pipeline 会通过 `encoded_size_hint` 从最后一个 stage 往前推导每个 stage 的 expected output size：

```text
final expected decoded size
  <- stage N encoded_size_hint
  <- stage N-1 encoded_size_hint
  ...
```

这个推导只能在 codec 能精确知道 encoded size 时成功；失败的中间 stage expected size 是 `None`。identity stage 会跳过实际 decode，但仍根据推导出来的 expected size 做 `verify_size`。

多 stage 解码时，pipeline 使用两个 `Vec<u8>` 轮换：

1. 第一阶段从 borrowed `encoded` 解码到 `spare`。
2. 后续阶段如果当前输入是 owned `Vec`，交给 `DecodeRunner::decode_vec_input`。
3. 如果 codec `prefers_decode_owned()`，可以直接消费输入 `Vec` 并返回 decoded `Vec`。
4. 否则把另一个 `Vec` 作为输出，解码后把输入 `Vec` 作为下一轮 spare。

`decode_into` 会尽量把最后一个 stage 直接写入 caller 的 `DecodeBuffer`，减少一次最终 copy。

### `DecodeRunner`

`runner.rs` 是内部调度器，把 `ChunkCodec` 的多个 fast-path 统一起来：

| 函数 | 行为 |
|---|---|
| `decode` | 直接调用 `codec.decode`。 |
| `decode_vec_input` | owned 输入时，如果 codec 偏好 owned path，调用 `decode_owned`；否则 borrowed decode 到输出 Vec，并把原输入 Vec 作为 spare 返回。 |
| `decode_borrowed_to_vec` | 调用 `decode_to_vec_grow`。 |
| `decode_into` | 调用 `decode_into`。 |
| `decode_to_initialized_vec` | 把 `Vec` 当前 len 暴露为 `DecodeBuffer`。 |
| `decode_to_capacity_vec` | 调用 codec 的 capacity fast path。 |

### 输出 buffer 和 `Vec` 复用

`DecodeBuffer` 只持有 `&mut [u8]`。所有直接写入路径都必须先检查容量，容量不足返回 `OutputTooSmall`。

`buffer.rs` 中的 `set_vec_len_for_decode` 使用 `unsafe { Vec::set_len(len) }`。安全前提：

1. 只用于 `Vec<u8>`，没有 drop glue。
2. 调用前已经检查或 reserve 了足够 capacity。
3. 立刻把暴露出来的 slice 交给 decoder/copy 路径。
4. decoder 必须完整写入返回范围内的字节，不能读取未初始化内容。

这种设计避免了为 decoded buffer 做多余初始化。

### `DecodePool` 工作模型

`DecodePool::with_codec_profile` 的初始化步骤：

1. 校验 `DecodePoolConfig`。
2. 如果指定 `cpus`，通过 `core_affinity::get_core_ids()` 校验 CPU id。
3. 创建 `flume::bounded(queue_capacity)` 作为 work queue。
4. 启动 `num_workers` 个 OS thread，线程名为 `decode-wrk-{idx}`。
5. worker 启动后如配置 CPU affinity，则 pin 当前线程。
6. worker 循环 `rx.recv()`，串行处理每个 `DecodeWork`。

每个 `DecodeWork` 包含：

```text
DecodeRequest
oneshot::Sender<CodecResult<Vec<u8>>>
CodecQueueTimer
```

worker 处理时会：

1. 记录 encoded bytes。
2. 根据 `DecodeOutput` 选择输出策略。
3. 用 `panic::catch_unwind` 包住实际解码。
4. codec error 原样返回；panic 转成 `CodecError::WorkerPanic`。
5. 记录 profiling，包括 queue wait、work time、decoded bytes、error/panic。
6. 通过 oneshot 返回结果。

`DecodeFuture::blocking_recv` 会消费 future。重复消费或 channel 断开都会返回 `Shutdown`。异步使用时，`DecodeFuture` 实现 `Future`，可以直接 `.await`。

### 各 codec 的解码格式和实现

#### `none` / `UncompressedCodec`

不解压，只校验 `expected_size` 并 copy 或复用输入。

fast path：

| path | 行为 |
|---|---|
| `decode_owned` | 校验后直接返回输入 `Vec`。 |
| `decode_to_vec` / `decode_to_vec_grow` | 复用 caller `Vec`，用 `ptr::copy_nonoverlapping` copy。 |
| `decode_into` | 直接 copy 到 caller buffer。 |
| `encoded_size_hint` | 返回 `decoded_size`。 |
| `is_identity` | true。 |

#### `crc32`

这是一个校验过滤器，不压缩。encoded 格式：

```text
u32 little-endian crc32(payload) + payload bytes
```

解码步骤：

1. 要求 encoded 至少 4 bytes。
2. 读取前 4 bytes 为 expected checksum。
3. 对 payload 计算 `crc32fast::hash`。
4. mismatch 返回 `CodecError::Decode`。
5. 校验 payload length 是否等于 `expected_size`。
6. 返回 payload。

`decode_owned` 会在校验后用 `copy_within(4.., 0)` 原地去掉 checksum header。`encoded_size_hint(decoded_size)` 返回 `decoded_size + 4`，带 overflow 检查。

#### `lz4`

实现 numcodecs lz4 风格格式：encoded 前 4 bytes 是 decoded size 的 `u32` little-endian，后面是 raw LZ4 payload。

解码步骤：

1. encoded 必须至少 4 bytes。
2. decoded size 不能超过 `i32::MAX`，因为 `lz4_sys::LZ4_decompress_safe` 参数是 `int`。
3. 根据 decoded size 精确分配或校验输出 buffer。
4. 调用 `LZ4_decompress_safe(compressed, output, compressed_size, decoded_size)`。
5. 要求返回写入长度精确等于 output length。

#### `zstd`

优先使用 zstd frame content size：

1. `zstd_safe::get_frame_content_size(encoded)` 返回 `Some(size)` 时，精确分配或校验输出 buffer，再用 `zstd_safe::decompress` 写入。
2. 返回 `None` 且 caller 提供 `expected_size` 时，把 `expected_size` 当输出目标大小。
3. 返回 `None` 且没有 `expected_size` 时，fallback 到 `zstd::decode_all(Cursor::new(encoded))`。
4. frame 解析错误会先返回 `Decode`，避免在 invalid frame + 巨大 `expected_size` 情况下先 reserve 巨大内存。

#### `gzip` / `zlib` / `bz2`

这三个 codec 走 `stream.rs` 里的 streaming reader：

| codec | reader |
|---|---|
| gzip | `flate2::read::GzDecoder` |
| zlib | `flate2::read::ZlibDecoder` |
| bz2 | `bzip2::read::BzDecoder` |

如果有 `expected_size`，先按 expected size reserve 输出，然后 `read_to_buffer`。`read_to_buffer` 在填满目标 buffer 后会额外读 1 byte：

| 情况 | 结果 |
|---|---|
| 刚好 EOF | 成功。 |
| 还能读出 1 byte 且有 expected size | `SizeMismatch`，actual 至少是 `expected + 1`。 |
| 还能读出 1 byte 且无 expected size | `OutputTooSmall`。 |

读取时会重试 `ErrorKind::Interrupted`。

#### `lzma`

基于 `xz2`。`LzmaCodecConfig` 影响 decoder 创建：

| `format` | decoder |
|---|---|
| `0` | `Stream::new_auto_decoder` |
| `1` | `Stream::new_stream_decoder`，XZ stream。 |
| `2` | `Stream::new_lzma_decoder`，LZMA-alone。 |
| `3` | raw LZMA stream，当前返回 `InvalidCodecConfig`。 |

`has_filters = true` 时也返回 `InvalidCodecConfig`，因为 raw LZMA filter chains 尚未实现。`check` 和 `preset` 不参与 decode。

#### `blosc`

Blosc 解码分两层：

1. 先解析并校验 Blosc header。
2. 尝试自实现 Blosc-LZ4 fast path。
3. 不满足 fast path 条件时 fallback 到 `blosc_src::blosc_decompress_ctx(..., nthreads = 1)`。

header 校验：

| 字段 | 规则 |
|---|---|
| 最小长度 | encoded 至少 `BLOSC_MIN_HEADER_LENGTH`。 |
| version | `encoded[0]` 必须等于 `BLOSC_VERSION_FORMAT`。 |
| decoded size | 从 bytes 4..8 读取 u32 little-endian，不能超过 `BLOSC_MAX_BUFFERSIZE`。 |
| blocksize | 从 bytes 8..12 读取。 |
| compressed size | 从 bytes 12..16 读取，必须等于 `encoded.len()`。 |
| flags/typesize | 保存到 `BloscHeader`，供 fast path 判断。 |

Blosc-LZ4 fast path 条件：

| 条件 | 不满足时 |
|---|---|
| `compformat() == BLOSC_LZ4_FORMAT` | 返回 `None`，fallback 到 Blosc C 解码器。 |
| `compversion == BLOSC_LZ4_VERSION_FORMAT` | 返回 `Decode` 错误。 |
| flags 不含 `0x08` 未知位 | fallback。 |
| 不是 bitshuffle | fallback。 |

fast path 支持：

| 情况 | 处理 |
|---|---|
| decoded size 为 0 | 直接返回 0。 |
| memcpyed buffer | 校验 `decoded_size + BLOSC_MAX_OVERHEAD == compressed_size`，直接 copy payload。 |
| 普通 LZ4 block | 读取 block offset table，逐 block 解码。 |
| byte shuffle | 先解到 thread-local scratch，再 `unshuffle_bytes` 写回输出。 |
| raw split | 如果 split 的 compressed size 等于 split size，直接 copy。 |
| compressed split | 调用模块内 raw LZ4 解码函数。 |

block table 校验会防止：

| 问题 | 错误 |
|---|---|
| offset 落在 header/table 区间 | `invalid Blosc block offset` |
| offset 递减或重叠 | `invalid Blosc block offset` |
| offset 超过 compressed size | `invalid Blosc block offset` |
| split header 不完整 | `invalid Blosc split offset` |
| split size 为负或越界 | `negative/invalid Blosc split size` |
| block 解码后剩余未消费 bytes | `invalid Blosc block size` |

byte unshuffle 在 `typesize` 为 2、4、8 时有专门实现，其他 `typesize` 走 generic path。`typesize = 4` 在 little-endian 平台使用 unaligned `u32` 写入优化。

### 内存和错误策略

这个模块尽量把错误提前到分配前：

| 场景 | 处理 |
|---|---|
| `expected_size` 与 header/frame size 不一致 | 在 reserve 输出前返回 `SizeMismatch`。 |
| lz4 decoded size 超过 FFI `i32` 上限 | 在分配前返回 `Decode`。 |
| blosc decoded size 超过 Blosc 上限 | 在分配前返回 `Decode`。 |
| zstd invalid frame | 在按 expected size reserve 前返回 `Decode`。 |
| `try_reserve_exact` 失败 | 转成 `CodecError::Decode { message: "failed to reserve decode buffer: ..." }`。 |

直接写入 caller buffer 时，任何容量不足都返回 `OutputTooSmall`，不会尝试扩容。只有 `decode_to_vec_grow` 和 `DecodeOutput::Allocate` 相关路径会主动增长 `Vec`。

## 常见用法

### 单个 codec

```rust
use _scdata::codecs::{codec_from_json_str, CodecResult};

fn decode_zstd(encoded: &[u8], expected_size: usize) -> CodecResult<Vec<u8>> {
    let codec = codec_from_json_str(r#"{"id":"zstd","level":3}"#)?;
    codec.decode(encoded, Some(expected_size))
}
```

### zarr v2 filters + compressor

```rust
use _scdata::codecs::{codec_pipeline_from_zarr_v2_json_str, CodecResult};

fn decode_zarr_v2(encoded: &[u8], expected_size: usize) -> CodecResult<Vec<u8>> {
    let codec = codec_pipeline_from_zarr_v2_json_str(
        Some(r#"[{"id":"crc32"}]"#),
        Some(r#"{"id":"zstd","level":3}"#),
    )?;
    codec.decode(encoded, Some(expected_size))
}
```

### 线程池提交

```rust
use _scdata::codecs::{CodecSpec, DecodePool, DecodePoolConfig, DecodeRequest};
use std::sync::Arc;

fn decode_in_pool(encoded: Arc<[u8]>, expected_size: usize) -> _scdata::codecs::CodecResult<Vec<u8>> {
    let pool = DecodePool::new(DecodePoolConfig::default())?;
    let request = DecodeRequest::from_spec(&CodecSpec::None, encoded)
        .with_expected_size(expected_size);
    pool.submit(request)?.blocking_recv()
}
```

## 扩展新 codec 时需要改的地方

由于 `ChunkCodec` 是 sealed trait，新增 codec 应该在 `codecs` 模块内部完成：

1. 在 `impls/` 下新增具体实现，并实现 `sealed::Sealed` 和 `ChunkCodec`。
2. 在 `impls/mod.rs` 中 re-export 给 `spec.rs` 使用。
3. 在 `CodecSpec` 中新增 enum variant 和配置结构。
4. 在 `CodecSpec::from_json_value` 中解析新 `id`。
5. 在 `CodecSpec::build_uncached` 中返回新实现。
6. 在 `registry.rs::CodecKey` 和 `CodecKey::from_spec` 中加入缓存 key。只把真正影响 decode 行为的字段放入 key。
7. 如果 codec 能知道输出大小，实现 `decoded_size_hint`；如果能从 decoded size 推回 encoded size，实现 `encoded_size_hint`，以提升 pipeline size check 和 buffer 复用效率。
8. 尽量实现 `decode_into`、`decode_to_vec`、`decode_to_vec_grow`，避免默认 `decode + copy`。
9. 增加 numcodecs 兼容向量测试、size mismatch 测试、caller buffer 太小测试，以及非法 header 不分配大内存的测试。
