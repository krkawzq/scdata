# scdata layers 支持重构方案

本文档描述如何在不改 Rust 数据访问内核的前提下,让 `scdata` 全面支持
AnnData `layers`。核心原则是:Rust 只负责访问一个已经注册的矩阵
dataset;Python IO 层负责把 `X` 和 `layers/<name>` 都解析成同一种
`DenseDataset` / `SparseDataset`。

## 1. 目标

1. `write_zarr` 能把 `adata.X` 和 `adata.layers` 中的矩阵都写成 scdata
   可直接注册的 zarr v3 布局。
2. `read_zarr` 能读回 `X` 和所有 layer,包括 scdata 扩展布局:
   `dense1d` 和 cell-aligned CSR rectilinear chunks。
3. `launch` 保持兼容,默认仍返回 `X` 的 `Dataset`。
4. 新增一个明确的 API,可以枚举并返回 store 中的所有矩阵:
   `X` 和 `layers/<name>`。
5. `ScDataBank` 继续按 `DatasetId` 工作;layer 注册后就是另一个
   dataset id。Rust core 不需要知道 layer 名称。
6. `launch` 只支持 zarr v3 store;早期自定义 payload store 已完全废弃。

## 2. 当前问题

### 2.1 `launch` 只解析 `X`

`scdata/io/_launch.py` 当前入口是:

- `launch(path) -> Dataset`
- `launch_store(store, store_root="") -> Dataset`

内部 `_launch_v3` 硬编码检查和解析 `X`:

- v3: `X/zarr.json`

这意味着即使 store 中有 `layers/counts`,Python 也没有 API 能把它解析成
`DenseDataset` / `SparseDataset`。

### 2.2 `write_zarr` 只优化 `X`

`scdata/io/_anndata.py` 当前写法是:

```python
if adata.layers:
    write_elem(g, "layers", dict(adata.layers))

_write_x(g, adata, format=format, chunk_size=chunk_size, align_cells=align_cells)
```

`X` 走 scdata 自己的 `_write_x`,layers 走 anndata 原生 writer。因此:

- dense layer 只会是标准 dense array,不会有 `dense1d` cell-aligned 布局;
- sparse layer 不会用 scdata 的 rectilinear CSR 写法;
- 后续 Rust register 无法依赖统一 metadata 发现和访问 layers。

### 2.3 `read_zarr` 只特殊处理 `/X`

`read_zarr` 的 callback 目前只在 `elem_name == "/X"` 时处理
`dense1d` / `sparse`:

```python
if name == "X" and x_kind in ("dense1d", "sparse", "sparse-vlen"):
    return _read_x_scdata(f, x_kind, x_attrs)
```

`layers/counts` 如果使用同样的 scdata 扩展布局,不会进入这条路径。

### 2.4 Rust 端已经足够通用

Rust core 的 `Dataset` enum 是矩阵级抽象:

- `Dense1D`
- `Dense2D`
- `SparseCsr`

`DataBank` 的所有访问 API 都以 `DatasetId` 为入口。只要 Python 端把
`layers/<name>` 解析成一个 `DenseDataset` 或 `SparseDataset`,现有
`register_dense` / `register_sparse_csr` 就可以注册它。无需新增 Rust-side
`Layer` 概念。

## 3. 设计原则

### 3.1 Matrix key 是 Python 层概念

定义统一 matrix key:

| 语义 | matrix key |
|---|---|
| 主矩阵 | `X` |
| layer `counts` | `layers/counts` |
| layer `lognorm` | `layers/lognorm` |

在用户 API 中可以接受 layer 简写,例如 `layer="counts"`;内部始终规范化为
`layers/counts`。

### 3.2 Dataset 仍表示单个矩阵

不要把 `layers` 字段塞进 `DenseDataset` / `SparseDataset`。这两个类型
继续表示一个矩阵。新增集合类型承载 AnnData store 的多矩阵结构。

推荐新增:

```python
@dataclass(frozen=True)
class DatasetCollection:
    x: Dataset
    layers: Mapping[str, Dataset]
    store_root: str = ""

    def __getitem__(self, key: str) -> Dataset: ...
    def keys(self) -> tuple[str, ...]: ...
    def items(self) -> Iterator[tuple[str, Dataset]]: ...
```

行为约定:

- `collection["X"]` 返回 `x`;
- `collection["layers/counts"]` 返回 counts layer;
- `collection["counts"]` 可以作为 layer 简写,但如果未来出现歧义,完整
  key 优先;
- `collection.layers` 的 key 是 layer name,不是完整 path。

该类型可以放在 `scdata/data/_dataset.py`,并从 `scdata.data` / `scdata`
导出。`Dataset = DenseDataset | SparseDataset` 不变。

### 3.3 On-disk marker 兼容

当前 scdata marker 名叫 `scdata-x`,虽然名字带 `x`,但可以继续读老字段。
推荐引入新字段并兼容旧字段:

```python
_SCDATA_MATRIX_ATTR = "scdata-matrix"
_SCDATA_X_ATTR = "scdata-x"  # legacy alias
```

新 writer 可以同时写:

```python
{
    "scdata-matrix": "dense1d",
    "scdata-x": "dense1d",
}
```

reader 按优先级读取:

```python
kind = attrs.get("scdata-matrix") or attrs.get("scdata-x")
```

这样老 store 不受影响,新 store 语义更准确。

## 4. Public API 方案

### 4.1 `launch` 保持默认兼容

当前代码大概率已有用户依赖:

```python
ds = scdata.io.launch(path)
did = bank.register(ds)
```

保持这个行为:默认返回 `X`。

建议扩展签名:

```python
def launch(
    path: str | os.PathLike[str],
    *,
    layer: str | None = None,
    matrix: str | None = None,
) -> Dataset: ...
```

规则:

- `launch(path)` 返回 `X`;
- `launch(path, layer="counts")` 返回 `layers/counts`;
- `launch(path, matrix="layers/counts")` 返回同一个 layer;
- `layer` 和 `matrix` 不能同时传;
- `matrix="X"` 等价于默认。

`launch_store` 同步扩展同样参数。

### 4.2 新增 `launch_all`

新增:

```python
def launch_all(path: str | os.PathLike[str]) -> DatasetCollection: ...

def launch_store_all(store: Store, *, store_root: str = "") -> DatasetCollection: ...
```

示例:

```python
datasets = scdata.io.launch_all("sample.zarr.zip")

ds_x = datasets["X"]
ds_counts = datasets["layers/counts"]

with ScDataBank() as bank:
    ids = bank.register_all(datasets)
    counts = bank.load(ids["layers/counts"], [0, 1, 2])
```

### 4.3 `ScDataBank.register_all`

Rust 不需要新增批量注册。Python wrapper 可提供便利方法:

```python
def register_all(self, datasets: DatasetCollection) -> dict[str, DatasetId]:
    out = {"X": self.register(datasets.x)}
    for name, ds in datasets.layers.items():
        out[f"layers/{name}"] = self.register(ds)
    return out
```

也可以新增对应的 unregister helper:

```python
def unregister_all(self, ids: Mapping[str, DatasetId]) -> None: ...
```

这只是循环调用现有 `register` / `unregister`,不会触及 Rust。

### 4.4 `ScDataLoader` 可选增强

现有 `ScDataLoader` 接受 `dataset_ids` 并调用 `bank.prefetch(id, ...)`。
layer 支持有两种选择:

1. 最小方案:调用方传入目标 layer 的 `DatasetId`。不改 loader。
2. 便利方案:允许 `dataset_ids` 的值是 `Mapping[str, DatasetId]`,并给
   loader 增加 `matrix="X"` / `layer="counts"` 参数,内部选择对应 id。

建议先实现最小方案,因为现有 loader 已经能处理任何 `DatasetId`。便利方案
可以放在后续 PR。

## 5. `scdata/io/_launch.py` 重构

### 5.1 拆出 matrix 级解析函数

把 `_launch_v3` 中的 `X` 硬编码拆成:

```python
def _launch_v3_matrix(
    store: Store,
    matrix_key: str,
    gene_names: tuple[str, ...],
    store_root: str,
) -> Dataset: ...
```

v3 逻辑:

- `matrix_key/zarr.json` 必须存在;
- `node_type == "array"`: dense;
- `node_type == "group"` 且 `encoding-type in {"csr_matrix", "CSR"}`: CSR;
- `encoding-type in {"csc_matrix", "CSC"}`: raise `StoreError`;
- 其他 encoding: raise `StoreError`;
- dense 复用 `_v3_build_dense_dataset`;
- sparse 复用 `_v3_build_sparse_dataset`。

### 5.2 枚举 layers

使用现有 `Store.list_keys(prefix)`。

v3:

```python
def _v3_layer_names(store: Store) -> tuple[str, ...]:
    if not store.exists("layers/zarr.json"):
        return ()
    keys = store.list_keys("layers")
    # direct child with layers/<name>/zarr.json
```

只枚举 direct child。writer 应拒绝 layer name 中包含 `/`,避免 nested layer
路径带来歧义。

### 5.3 新入口组织

推荐结构:

```python
def launch(path, *, layer=None, matrix=None):
    matrix_key = _resolve_matrix_key(layer=layer, matrix=matrix)
    with _open_store(path) as store:
        return launch_store(store, store_root=os.fspath(path), matrix=matrix_key)

def launch_all(path):
    with _open_store(path) as store:
        return launch_store_all(store, store_root=os.fspath(path))
```

`launch_store_all`:

1. 检查 root 是 v3;
2. 读取 `var` gene names 一次;
3. 解析 `X`;
4. 枚举 layers;
5. 对每个 layer 调用 matrix 级解析;
6. 校验每个 layer 的 `num_genes == X.num_genes`,且 gene names 一致;
7. 建议也校验 `num_cells == X.num_cells`,因为 AnnData layers 必须与
   `X.shape` 一致。

## 6. `scdata/io/_anndata.py` 重构

### 6.1 把 `_write_x` 改成 `_write_matrix`

当前 `_write_x(g, adata, ...)` 从 `adata.X` 取矩阵并固定写到 `"X"`。
改为:

```python
def _write_matrix(
    g: Any,
    key: str,
    matrix: Any,
    *,
    n_obs: int,
    n_var: int,
    format: _MatrixFormat,
    chunk_size: int | list[int] | tuple[int, ...],
    align_cells: bool,
) -> None: ...
```

`_write_x` 可以变成薄 wrapper:

```python
def _write_x(g, adata, *, format, chunk_size, align_cells):
    _write_matrix(g, "X", adata.X, n_obs=adata.n_obs, n_var=adata.n_vars, ...)
```

写 layer 时:

```python
layers_group = g.require_group("layers")
for name, matrix in adata.layers.items():
    _validate_layer_name(name)
    fmt = _resolve_layer_format(matrix, layer_format, name)
    _write_matrix(layers_group, name, matrix, n_obs=adata.n_obs, n_var=adata.n_vars, ...)
```

注意:传给 `_write_matrix` 的 `key` 是当前 group 下的相对名字。对于
`layers_group`,写 `name` 即落盘到 `layers/<name>`。

### 6.2 format 解析策略

`write_zarr` 现有 `format` 默认是 `"dense2d"`。为了兼容,不要改变 `X`
默认行为。

建议新增参数:

```python
layer_format: Literal["preserve", "auto", "dense2d", "dense1d", "sparse"]
| Mapping[str, Literal["auto", "dense2d", "dense1d", "sparse"]]
= "preserve"
```

语义:

- `"preserve"`:
  - sparse layer -> `"sparse"`;
  - dense layer -> `"dense2d"`;
  - 这是最兼容 stock anndata 的默认。
- `"auto"`:
  - sparse layer -> `"sparse"`;
  - dense layer -> `"dense1d"` when `align_cells=True`,否则 `"dense2d"`;
  - 这是性能优先。
- 显式 `"dense1d"` / `"dense2d"` / `"sparse"`:
  - 应用于所有 layer;
  - `"sparse"` 可把 dense layer 转 CSR,但要在文档里标明可能改变存储密度;
  - `"dense1d"` / `"dense2d"` 可把 sparse layer densify,必须谨慎。
- mapping:
  - per-layer override;
  - 未出现的 layer 走 `"preserve"` 或另一个 `default_layer_format`。

如果希望 API 更简单,第一版可以只支持:

```python
layer_format: Literal["preserve", "auto"] = "preserve"
```

后续再加 mapping。

### 6.3 shape 校验

每个 layer 必须满足:

```python
matrix.shape == (adata.n_obs, adata.n_vars)
```

稀疏矩阵用 `matrix.shape`;dense/backed array 用 `np.asarray(matrix).shape`
前先尽量读取 `.shape`,避免不必要 materialize。

错误示例:

```python
raise StoreError(
    f"layer {name!r} has shape {shape}, expected {(adata.n_obs, adata.n_vars)}"
)
```

### 6.4 read_zarr 改为 matrix root 级处理

新增 helper:

```python
def _is_matrix_root(name: str) -> bool:
    if name == "X":
        return True
    parts = name.split("/")
    return len(parts) == 2 and parts[0] == "layers" and bool(parts[1])
```

callback 中不要对任意带 scdata marker 的节点都拦截,否则
`layers/counts/data` 这种 CSR 子数组也可能被误拦截。只拦截 matrix root:

```python
attrs = dict(getattr(elem, "attrs", {}))
kind = _matrix_kind(attrs)
if _is_matrix_root(name) and kind in ("dense1d", "sparse"):
    return _read_matrix_scdata(f, name, kind, attrs)
```

`dense2d` 可以让 anndata/zarr 原生读,不需要特殊处理。

把这些函数改成接受 `matrix_key`:

```python
def _read_matrix_scdata(f, matrix_key: str, kind: str, attrs: dict[str, Any]) -> Any
def _read_csr(f, matrix_key: str) -> Any
def _read_sub_array(f, key: str) -> np.ndarray
```

`_read_csr` 内部从:

```python
x = f["X"]
indptr = _read_sub_array(f, "X/indptr")
```

改成:

```python
x = f[matrix_key]
indptr = _read_sub_array(f, f"{matrix_key}/indptr")
```

### 6.5 rectilinear marker

`_write_rectilinear_array` 当前给 `indices` / `data` 子数组写
`scdata-x: sparse-vlen`。保留它,因为 `_read_sub_array` 需要识别该数组是否
需要手工拼 chunk。

新增 `scdata-matrix` 后也应写在子数组上:

```python
attrs = {
    "encoding-type": "array",
    "encoding-version": "0.2.0",
    "scdata-matrix": "sparse-vlen",
    "scdata-x": "sparse-vlen",
}
```

## 7. `ScDataBank` 层

### 7.1 不改 Rust binding

现有 pybind:

- `_DataBank.register_dense(ds, store_path)`
- `_DataBank.register_sparse_csr(ds, store_path)`
- `_DataBank.access_cells(id, ...)`
- `_DataBank.prefetch_cells(id, ...)`

都只需要一个 `DatasetId`。layer 注册后就是一个普通 dataset id。

### 7.2 Python wrapper 便利方法

在 `scdata/databank.py` 增加:

```python
def register_all(self, datasets: DatasetCollection) -> dict[str, DatasetId]: ...
def unregister_all(self, ids: Mapping[str, DatasetId]) -> None: ...
```

实现细节:

- 先注册 `X`;
- 再按 layer name 排序注册 layers,保证测试可复现;
- 如果中途失败,回滚已注册 id,避免文件句柄泄漏;
- 返回 key 使用完整 matrix key: `"X"`、`"layers/counts"`。

## 8. 导出与类型 stub

需要更新:

- `scdata/data/__init__.py`
  - 导出 `DatasetCollection`
- `scdata/__init__.py`
  - 导出 `DatasetCollection`
  - 导出 `launch_all` / `launch_store_all`
- `scdata/io/__init__.py`
  - 导出 `launch_all` / `launch_store_all`
- `scdata/_scdata.pyi`
  - Rust stub 不需要增加 layer API
  - 如果 `ScDataBank.register_all` 是纯 Python wrapper,不在这里声明

如果项目有 pyright/mypy 配置,补充 public function annotations。

## 9. 测试计划

### 9.1 `write_zarr` / `read_zarr`

新增 `tests/test_anndata.py` 用例:

1. dense layer 默认 preserve:
   - `adata.layers["counts"] = dense`
   - `write_zarr(..., format="dense2d", layer_format="preserve")`
   - `read_zarr(root).layers["counts"]` 与原矩阵一致
2. dense layer `dense1d`:
   - `layer_format="auto"` 或 per-layer `"dense1d"`
   - 检查 `layers/counts` 有 `scdata-matrix == "dense1d"`
   - `read_zarr` reshape 正确
3. sparse layer regular CSR:
   - `align_cells=False`
   - `read_zarr` 返回 CSR,内容一致
4. sparse layer rectilinear CSR:
   - `align_cells=True`
   - `layers/counts/data` 和 `indices` 是 rectilinear
   - `read_zarr` 手工拼 chunk 正确
5. zip store:
   - 至少覆盖一个 dense1d layer 或 sparse rectilinear layer

### 9.2 `launch` / `launch_all`

新增用例:

1. `launch(root)` 仍返回 `X`;
2. `launch(root, layer="counts")` 返回 layer dataset;
3. `launch(root, matrix="layers/counts")` 等价;
4. `launch_all(root).keys() == ("X", "layers/counts", ...)`;
5. dense layer 注册后 `bank.load` 内容正确;
6. sparse layer 注册后 `bank.load` 内容正确;
7. missing layer 抛清晰错误:
   - `launch(root, layer="missing")`
8. layer gene 数不等于 `var` 报错;
9. layer cell 数不等于 `X` 时,`launch_all` 报错。

### 9.3 legacy store removal

1. 早期自定义 payload store 不再解析;
2. 只有 `.zgroup`、没有根 `zarr.json` 的 store 应报 `not a zarr v3 store`;
3. 测试夹具不得再手写旧 payload store,所有正向读写测试都通过
   `write_zarr` 生成 zarr v3 store。

### 9.4 error paths

1. layer name 包含 `/`:writer 拒绝;
2. CSC layer:launch/read 报 `scdata does not read CSC matrices; store as CSR`;
3. `layer` 和 `matrix` 同时传:TypeError 或 ValueError;
4. `matrix="layers/"` / `matrix="obs"`:ValueError。

## 10. 分阶段实施建议

### Phase 1: data model 和 launch

改动文件:

- `scdata/data/_dataset.py`
- `scdata/data/__init__.py`
- `scdata/io/_launch.py`
- `scdata/io/__init__.py`
- `scdata/__init__.py`

交付:

- `DatasetCollection`
- `launch(..., layer=..., matrix=...)`
- `launch_all`
- v3/v2 matrix 级解析
- tests 覆盖 launch 和 databank register layer

这是最小可用版本:即使 writer 还没改,只要 store 中已有 scdata-formatted
layer,就能注册访问。

### Phase 2: write_zarr layers

改动文件:

- `scdata/io/_anndata.py`
- `scdata/io/_convert.py` 如果 converter 需要暴露 layer format

交付:

- `_write_matrix`
- `_write_layers`
- `layer_format`
- layer shape/name 校验
- dense/sparse layer 写入测试

### Phase 3: read_zarr layers

改动文件:

- `scdata/io/_anndata.py`

交付:

- `_is_matrix_root`
- `_read_matrix_scdata`
- `_read_csr(f, matrix_key)`
- dense1d layer 和 sparse rectilinear layer 读回测试

### Phase 4: bank helpers 和文档

改动文件:

- `scdata/databank.py`
- `README.md` 或单独 usage doc

交付:

- `ScDataBank.register_all`
- 可选 `unregister_all`
- usage examples

## 11. 兼容性与迁移

### 11.1 Backward compatibility

- `launch(path)` 不变,仍返回 `X`;
- `write_zarr(..., format="dense2d")` 的 `X` 行为不变;
- 老 store 的 `scdata-x` marker 继续可读;
- Rust `_DataBank` API 不变。

### 11.2 行为变化

如果 `write_zarr` 默认开始用 scdata writer 写 layers,layer 的 zarr metadata
会与 anndata 原生 writer 不完全相同。为降低风险:

- 默认 `layer_format="preserve"`:dense layer 写标准 dense2d,sparse layer 写 CSR;
- 性能优先用户显式使用 `layer_format="auto"` 或 mapping;
- `read_zarr` 保证 scdata 自己写出的所有 layer 可恢复为 AnnData layers。

### 11.3 Store key 约束

第一版只支持 direct child layer:

- 支持: `layers/counts`
- 不支持: `layers/group/counts`

writer 应拒绝 layer name 中包含 `/`,reader 只枚举 direct child。

## 12. 验收标准

1. `pytest tests/test_anndata.py tests/test_read.py tests/test_databank.py`
   通过。
2. `launch(path)` 旧测试无需修改或只需因新增参数调整 import。
3. 对包含 dense + sparse layers 的 AnnData:

```python
root = write_zarr(adata, path, format="dense1d", layer_format="auto")
datasets = launch_all(root)
ids = bank.register_all(datasets)
np.asarray(bank.load(ids["layers/counts"], cells)).reshape(...)
```

结果与 `adata.layers["counts"]` 一致。

4. `read_zarr(root)` 返回的 `AnnData.X` 和 `AnnData.layers` 与写入前一致。
5. Rust crate 不需要新增 public API;如果 Rust 测试需要更新,只应是因为
   Python-facing metadata examples 变化,不是因为 layer 语义进入 Rust core。

## 13. 非目标

以下内容不放入第一版:

- Rust 直接解析 AnnData zarr store;
- Rust `DatasetId` 绑定 layer 名称;
- 一次 Rust call 同时访问多个 layer;
- 跨 layer 共享 chunk scheduling 的联合 prefetch 优化;
- 支持 nested layer path。

这些都可以在 Python layer 支持稳定后再评估。
