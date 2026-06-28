//! PyO3 bindings for the scdata DataBank.
//!
//! Binds the full DataBank surface: opening a DataBank (with configurable IO /
//! decode / access / fill pools), registering Python-parsed datasets
//! (`scdata.data.DenseDataset` / `SparseDataset`) against a directory store,
//! querying gene names / counts / dtype, reading cell data back as numpy arrays
//! (full or gene-name-projected), and streaming cell batches through a
//! scheduled prefetch iterator.
//!
//! These bindings only touch `lib.rs` (to register the module members) and this
//! file.  The databank / access / codecs / iopool cores keep their generic
//! trait constraints (`DataValue`, `AsRef<str>`) untouched — this layer does all
//! the PyObject reflection so the Rust core stays optimizable.
//!
//! The bridge reads Python dataset objects by attribute (``getattr``) rather
//! than requiring a fixed ``__dict__`` shape, so the pure-Python
//! ``scdata.data._dataset`` dataclasses feed Rust without any Python-side
//! ``to_rust`` helper.  Each Python ``ArrayMeta`` is rebuilt as a Rust
//! ``ArrayMeta`` whose codec is carried as ``ArrayCodecMeta::ZarrV2Json`` (the
//! verbatim numcodecs filter/compressor JSON) — exactly the form Rust
//! ``Array::from_meta`` rebuilds the pipeline from.
//!
//! ``payload_path`` on a Python ``ArrayMeta`` is a key *relative to the store
//! root* (e.g. ``X/payload.bin``), not a filesystem path.  The register entry
//! points therefore take an explicit ``store_path`` (the ``.zarr`` directory)
//! and join it with each payload key.  ZIP stores are not supported by this
//! binding yet: their payload lives inside an archive, not on the filesystem.
//!
//! Cell access dispatches on dtype: `access_cells_owned::<T>` is generic in
//! Rust, but pyo3 cannot lift that generic across the GIL, so each numeric
//! dtype has its own call arm (`access_cells_dispatch`).  The Rust core
//! allocates and fills a `Vec<T>`, which `PyArray1::from_slice` moves into a
//! contiguous numpy array — no extra copy.  `f16` reinterprets the opaque
//! `F16Bits` bits as numpy `float16`; `bf16` has no numpy dtype, so its raw
//! bit pattern is returned as `uint16` for the caller to view as bfloat16.

use std::path::PathBuf;

use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;

use numpy::PyArray1;

use crate::databank::DataBankError as RustDataBankError;
use crate::databank::{
    ArrayCodecMeta, ArrayMeta, ArrayOrder, Bf16Bits, ChunkStoreMeta, DType, DataBank as RustDataBank,
    DataBankConfig, DataBankResult, DatasetId, Dense1DMeta, Dense2DMeta, F16Bits, FileChunkLocation,
    GeneNameView, MissingGenePolicy, PrefetchCells, PrefetchedBatch, ScheduledPrefetchConfig,
    SparseCsrDatasetMeta,
};
use crate::codecs::DecodePoolConfig;
use crate::access::{AccessConfig, AccessCpuConfig, ScheduledAccessConfig};
use crate::iopool::{BaseIoConfig, IoConfig, ThreadedConfig, UringConfig};

create_exception!(_scdata, DataBankError, pyo3::exceptions::PyRuntimeError);

impl From<RustDataBankError> for PyErr {
    fn from(err: RustDataBankError) -> Self {
        DataBankError::new_err(err.to_string())
    }
}

/// Register the Python-facing names exposed by this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataBank>()?;
    m.add_class::<PyDatasetId>()?;
    m.add_class::<PyMissingGenePolicy>()?;
    m.add_class::<PyDataBankConfig>()?;
    m.add_class::<PyIoConfig>()?;
    m.add_class::<PyUringConfig>()?;
    m.add_class::<PyThreadedConfig>()?;
    m.add_class::<PyBaseIoConfig>()?;
    m.add_class::<PyDecodePoolConfig>()?;
    m.add_class::<PyAccessConfig>()?;
    m.add_class::<PyAccessCpuConfig>()?;
    m.add_class::<PyFillConfig>()?;
    m.add_class::<PyScheduledAccessConfig>()?;
    m.add_class::<PyScheduledPrefetchConfig>()?;
    m.add_class::<PyPrefetchCells>()?;
    m.add("DataBankError", m.py().get_type::<DataBankError>())?;
    Ok(())
}

// ===========================================================================
// DatasetId
// ===========================================================================

/// Opaque handle returned by `DataBank.register_*`.
///
/// Wraps the Rust `(slot, generation)` pair so a stale id from an unregistered
/// dataset is rejected rather than silently naming a reused slot.
#[pyclass(name = "_DatasetId", frozen, module = "scdata._scdata")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PyDatasetId {
    slot: u32,
    generation: u32,
}

impl From<DatasetId> for PyDatasetId {
    fn from(id: DatasetId) -> Self {
        Self {
            slot: id.slot,
            generation: id.generation,
        }
    }
}

impl From<PyDatasetId> for DatasetId {
    fn from(id: PyDatasetId) -> Self {
        DatasetId {
            slot: id.slot,
            generation: id.generation,
        }
    }
}

#[pymethods]
impl PyDatasetId {
    #[new]
    fn new(slot: u32, generation: u32) -> Self {
        Self { slot, generation }
    }

    #[getter]
    fn slot(&self) -> u32 {
        self.slot
    }

    #[getter]
    fn generation(&self) -> u32 {
        self.generation
    }

    fn __repr__(&self) -> String {
        format!("DatasetId(slot={}, generation={})", self.slot, self.generation)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.extract::<PyDatasetId>().map_or(false, |o| *self == o)
    }

    fn __hash__(&self) -> u64 {
        (u64::from(self.slot)) | (u64::from(self.generation) << 32)
    }
}

// ===========================================================================
// MissingGenePolicy
// ===========================================================================

/// Policy for gene names requested via `access_cells_by_gene_names` that are
/// absent from the dataset.
#[pyclass(name = "_MissingGenePolicy", frozen, module = "scdata._scdata")]
#[derive(Clone, Copy)]
struct PyMissingGenePolicy {
    inner: MissingGenePolicy,
}

#[pymethods]
impl PyMissingGenePolicy {
    #[classattr]
    fn ZERO() -> Self {
        Self { inner: MissingGenePolicy::Zero }
    }

    #[classattr]
    fn ERROR() -> Self {
        Self { inner: MissingGenePolicy::Error }
    }

    #[new]
    fn new(policy: &str) -> PyResult<Self> {
        match policy.to_ascii_lowercase().as_str() {
            "zero" => Ok(Self { inner: MissingGenePolicy::Zero }),
            "error" => Ok(Self { inner: MissingGenePolicy::Error }),
            other => Err(PyValueError::new_err(format!(
                "unknown MissingGenePolicy {other:?}; use 'zero' or 'error'"
            ))),
        }
    }

    fn __repr__(&self) -> &'static str {
        match self.inner {
            MissingGenePolicy::Zero => "MissingGenePolicy.ZERO",
            MissingGenePolicy::Error => "MissingGenePolicy.ERROR",
        }
    }
}

// ===========================================================================
// Config types
// ===========================================================================
//
// Each config mirrors its Rust counterpart as a pyclass with a `#[new]` that
// defaults and `#[getter]`/`#[setter]` per field, so Python users configure by
// attribute assignment.  `build_*` helpers translate a fully-populated pyclass
// tree into the Rust struct tree the DataBank constructor expects.

#[pyclass(name = "_BaseIoConfig", module = "scdata._scdata")]
#[derive(Clone)]
struct PyBaseIoConfig {
    inner: BaseIoConfig,
}

#[pymethods]
impl PyBaseIoConfig {
    #[new]
    fn new() -> Self {
        Self { inner: BaseIoConfig::default() }
    }
    #[getter] fn get_max_in_flight(&self) -> usize { self.inner.max_in_flight }
    #[setter] fn set_max_in_flight(&mut self, value: usize) { self.inner.max_in_flight = value; }
    #[getter] fn get_priority_levels(&self) -> usize { self.inner.priority_levels }
    #[setter] fn set_priority_levels(&mut self, value: usize) { self.inner.priority_levels = value; }
    #[getter] fn get_queue_shards(&self) -> usize { self.inner.queue_shards }
    #[setter] fn set_queue_shards(&mut self, value: usize) { self.inner.queue_shards = value; }
    #[getter] fn get_assume_non_overlapping_reads(&self) -> bool { self.inner.assume_non_overlapping_reads }
    #[setter] fn set_assume_non_overlapping_reads(&mut self, value: bool) { self.inner.assume_non_overlapping_reads = value; }

    fn __repr__(&self) -> String {
        format!("BaseIoConfig(max_in_flight={}, queue_shards={})", self.inner.max_in_flight, self.inner.queue_shards)
    }
}

#[pyclass(name = "_UringConfig", module = "scdata._scdata")]
#[derive(Clone)]
struct PyUringConfig {
    inner: UringConfig,
}

#[pymethods]
impl PyUringConfig {
    #[new]
    fn new() -> Self {
        Self { inner: UringConfig::default() }
    }
    #[getter] fn get_base(&self) -> PyBaseIoConfig { PyBaseIoConfig { inner: self.inner.base.clone() } }
    #[setter] fn set_base(&mut self, value: PyBaseIoConfig) { self.inner.base = value.inner; }
    #[getter] fn get_entries(&self) -> u32 { self.inner.entries }
    #[setter] fn set_entries(&mut self, value: u32) { self.inner.entries = value; }
    #[getter] fn get_drivers(&self) -> usize { self.inner.drivers }
    #[setter] fn set_drivers(&mut self, value: usize) { self.inner.drivers = value; }
    #[getter] fn get_iowq_bounded_workers(&self) -> u32 { self.inner.iowq_bounded_workers }
    #[setter] fn set_iowq_bounded_workers(&mut self, value: u32) { self.inner.iowq_bounded_workers = value; }
    #[getter] fn get_iowq_unbounded_workers(&self) -> u32 { self.inner.iowq_unbounded_workers }
    #[setter] fn set_iowq_unbounded_workers(&mut self, value: u32) { self.inner.iowq_unbounded_workers = value; }
    #[getter] fn get_registered_files(&self) -> u32 { self.inner.registered_files }
    #[setter] fn set_registered_files(&mut self, value: u32) { self.inner.registered_files = value; }
}

#[pyclass(name = "_ThreadedConfig", module = "scdata._scdata")]
#[derive(Clone)]
struct PyThreadedConfig {
    inner: ThreadedConfig,
}

#[pymethods]
impl PyThreadedConfig {
    #[new]
    fn new() -> Self {
        Self { inner: ThreadedConfig::default() }
    }
    #[getter] fn get_base(&self) -> PyBaseIoConfig { PyBaseIoConfig { inner: self.inner.base.clone() } }
    #[setter] fn set_base(&mut self, value: PyBaseIoConfig) { self.inner.base = value.inner; }
    #[getter] fn get_num_workers(&self) -> usize { self.inner.num_workers }
    #[setter] fn set_num_workers(&mut self, value: usize) { self.inner.num_workers = value; }
    #[getter] fn get_cpus(&self) -> Option<Vec<usize>> { self.inner.cpus.clone() }
    #[setter] fn set_cpus(&mut self, value: Option<Vec<usize>>) { self.inner.cpus = value; }
}

/// IO backend selection: `IoConfig.uring(UringConfig())` or `.threaded(ThreadedConfig())`.
#[pyclass(name = "_IoConfig", module = "scdata._scdata")]
#[derive(Clone)]
struct PyIoConfig {
    inner: IoConfig,
}

#[pymethods]
impl PyIoConfig {
    #[new]
    fn new() -> Self {
        Self { inner: IoConfig::default() }
    }

    #[staticmethod]
    fn uring(config: Option<PyUringConfig>) -> Self {
        Self {
            inner: IoConfig::Uring(config.map(|c| c.inner).unwrap_or_default()),
        }
    }

    #[staticmethod]
    fn threaded(config: Option<PyThreadedConfig>) -> Self {
        Self {
            inner: IoConfig::Threaded(config.map(|c| c.inner).unwrap_or_default()),
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            IoConfig::Uring(_) => "uring",
            IoConfig::Threaded(_) => "threaded",
        }
    }

    fn __repr__(&self) -> String {
        format!("IoConfig({})", self.kind())
    }
}

#[pyclass(name = "_DecodePoolConfig", module = "scdata._scdata")]
#[derive(Clone)]
struct PyDecodePoolConfig {
    inner: DecodePoolConfig,
}

#[pymethods]
impl PyDecodePoolConfig {
    #[new]
    fn new() -> Self {
        Self { inner: DecodePoolConfig::default() }
    }
    #[getter] fn get_num_workers(&self) -> usize { self.inner.num_workers }
    #[setter] fn set_num_workers(&mut self, value: usize) { self.inner.num_workers = value; }
    #[getter] fn get_queue_capacity(&self) -> usize { self.inner.queue_capacity }
    #[setter] fn set_queue_capacity(&mut self, value: usize) { self.inner.queue_capacity = value; }
    #[getter] fn get_cpus(&self) -> Option<Vec<usize>> { self.inner.cpus.clone() }
    #[setter] fn set_cpus(&mut self, value: Option<Vec<usize>>) { self.inner.cpus = value; }
}

#[pyclass(name = "_AccessCpuConfig", module = "scdata._scdata")]
#[derive(Clone)]
struct PyAccessCpuConfig {
    inner: AccessCpuConfig,
}

#[pymethods]
impl PyAccessCpuConfig {
    #[new]
    fn new() -> Self {
        Self { inner: AccessCpuConfig::default() }
    }
    #[getter] fn get_num_workers(&self) -> usize { self.inner.num_workers }
    #[setter] fn set_num_workers(&mut self, value: usize) { self.inner.num_workers = value; }
    #[getter] fn get_queue_capacity(&self) -> usize { self.inner.queue_capacity }
    #[setter] fn set_queue_capacity(&mut self, value: usize) { self.inner.queue_capacity = value; }
    #[getter] fn get_cpus(&self) -> Option<Vec<usize>> { self.inner.cpus.clone() }
    #[setter] fn set_cpus(&mut self, value: Option<Vec<usize>>) { self.inner.cpus = value; }
}

#[pyclass(name = "_AccessConfig", module = "scdata._scdata")]
#[derive(Clone)]
struct PyAccessConfig {
    inner: AccessConfig,
}

#[pymethods]
impl PyAccessConfig {
    #[new]
    fn new() -> Self {
        Self { inner: AccessConfig::default() }
    }
    #[getter] fn get_queue_capacity(&self) -> usize { self.inner.queue_capacity }
    #[setter] fn set_queue_capacity(&mut self, value: usize) { self.inner.queue_capacity = value; }
    #[getter] fn get_scheduler_shards(&self) -> usize { self.inner.scheduler_shards }
    #[setter] fn set_scheduler_shards(&mut self, value: usize) { self.inner.scheduler_shards = value; }
    #[getter] fn get_cache_capacity_bytes(&self) -> usize { self.inner.cache_capacity_bytes }
    #[setter] fn set_cache_capacity_bytes(&mut self, value: usize) { self.inner.cache_capacity_bytes = value; }
    #[getter] fn get_memory_budget_bytes(&self) -> usize { self.inner.memory_budget_bytes }
    #[setter] fn set_memory_budget_bytes(&mut self, value: usize) { self.inner.memory_budget_bytes = value; }
    #[getter] fn get_default_io_priority(&self) -> u8 { self.inner.default_io_priority }
    #[setter] fn set_default_io_priority(&mut self, value: u8) { self.inner.default_io_priority = value; }
    #[getter] fn get_keep_decoded(&self) -> bool { self.inner.keep_decoded }
    #[setter] fn set_keep_decoded(&mut self, value: bool) { self.inner.keep_decoded = value; }
    #[getter] fn get_cpu(&self) -> PyAccessCpuConfig { PyAccessCpuConfig { inner: self.inner.cpu.clone() } }
    #[setter] fn set_cpu(&mut self, value: PyAccessCpuConfig) { self.inner.cpu = value.inner; }
}

#[pyclass(name = "_FillConfig", module = "scdata._scdata")]
#[derive(Clone)]
struct PyFillConfig {
    inner: crate::databank::FillConfig,
}

#[pymethods]
impl PyFillConfig {
    #[new]
    fn new() -> Self {
        Self { inner: crate::databank::FillConfig::default() }
    }
    #[getter] fn get_parallel(&self) -> bool { self.inner.parallel }
    #[setter] fn set_parallel(&mut self, value: bool) { self.inner.parallel = value; }
    #[getter] fn get_num_workers(&self) -> usize { self.inner.num_workers }
    #[setter] fn set_num_workers(&mut self, value: usize) { self.inner.num_workers = value; }
    #[getter] fn get_queue_capacity(&self) -> usize { self.inner.queue_capacity }
    #[setter] fn set_queue_capacity(&mut self, value: usize) { self.inner.queue_capacity = value; }
    #[getter] fn get_min_parallel_rows(&self) -> usize { self.inner.min_parallel_rows }
    #[setter] fn set_min_parallel_rows(&mut self, value: usize) { self.inner.min_parallel_rows = value; }
    #[getter] fn get_min_parallel_bytes(&self) -> usize { self.inner.min_parallel_bytes }
    #[setter] fn set_min_parallel_bytes(&mut self, value: usize) { self.inner.min_parallel_bytes = value; }
    #[getter] fn get_cpus(&self) -> Option<Vec<usize>> { self.inner.cpus.clone() }
    #[setter] fn set_cpus(&mut self, value: Option<Vec<usize>>) { self.inner.cpus = value; }
}

#[pyclass(name = "_DataBankConfig", module = "scdata._scdata")]
#[derive(Clone)]
struct PyDataBankConfig {
    inner: DataBankConfig,
}

#[pymethods]
impl PyDataBankConfig {
    #[new]
    fn new() -> Self {
        Self { inner: DataBankConfig::default() }
    }

    #[getter] fn get_io_config(&self) -> PyIoConfig { PyIoConfig { inner: self.inner.io_config.clone() } }
    #[setter] fn set_io_config(&mut self, value: PyIoConfig) { self.inner.io_config = value.inner; }
    #[getter] fn get_decode_config(&self) -> PyDecodePoolConfig { PyDecodePoolConfig { inner: self.inner.decode_config.clone() } }
    #[setter] fn set_decode_config(&mut self, value: PyDecodePoolConfig) { self.inner.decode_config = value.inner; }
    #[getter] fn get_access_config(&self) -> PyAccessConfig { PyAccessConfig { inner: self.inner.access_config.clone() } }
    #[setter] fn set_access_config(&mut self, value: PyAccessConfig) { self.inner.access_config = value.inner; }
    #[getter] fn get_fill_config(&self) -> PyFillConfig { PyFillConfig { inner: self.inner.fill_config.clone() } }
    #[setter] fn set_fill_config(&mut self, value: PyFillConfig) { self.inner.fill_config = value.inner; }
}

#[pyclass(name = "_ScheduledAccessConfig", module = "scdata._scdata")]
#[derive(Clone, Copy)]
struct PyScheduledAccessConfig {
    inner: ScheduledAccessConfig,
}

#[pymethods]
impl PyScheduledAccessConfig {
    #[new]
    fn new() -> Self {
        Self { inner: ScheduledAccessConfig::default() }
    }
    #[getter] fn get_prefetch_step(&self) -> usize { self.inner.prefetch_step }
    #[setter] fn set_prefetch_step(&mut self, value: usize) { self.inner.prefetch_step = value; }
    #[getter] fn get_decode_ahead_steps(&self) -> usize { self.inner.decode_ahead_steps }
    #[setter] fn set_decode_ahead_steps(&mut self, value: usize) { self.inner.decode_ahead_steps = value; }
    #[getter] fn get_ready_ahead_steps(&self) -> usize { self.inner.ready_ahead_steps }
    #[setter] fn set_ready_ahead_steps(&mut self, value: usize) { self.inner.ready_ahead_steps = value; }
}

#[pyclass(name = "_ScheduledPrefetchConfig", module = "scdata._scdata")]
#[derive(Clone, Copy)]
struct PyScheduledPrefetchConfig {
    inner: ScheduledPrefetchConfig,
}

#[pymethods]
impl PyScheduledPrefetchConfig {
    #[new]
    fn new() -> Self {
        Self { inner: ScheduledPrefetchConfig::default() }
    }
    #[getter] fn get_prefetch_step(&self) -> usize { self.inner.prefetch_step }
    #[setter] fn set_prefetch_step(&mut self, value: usize) { self.inner.prefetch_step = value; }
    #[getter] fn get_access(&self) -> PyScheduledAccessConfig { PyScheduledAccessConfig { inner: self.inner.access } }
    #[setter] fn set_access(&mut self, value: PyScheduledAccessConfig) { self.inner.access = value.inner; }
}

// ===========================================================================
// DataBank
// ===========================================================================

/// Single-cell DataBank: registers parsed datasets and serves cell access.
///
/// Constructed with a `DataBankConfig`; the Rust core owns its io_uring / thread
/// pool, decode pool, access scheduler, and compute pool.  Dropping the Python
/// object shuts those down via the Rust `Drop` impl.
#[pyclass(name = "_DataBank", module = "scdata._scdata")]
struct PyDataBank {
    inner: RustDataBank,
}

#[pymethods]
impl PyDataBank {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyDataBankConfig>) -> PyResult<Self> {
        let config = config.map(|c| c.inner).unwrap_or_default();
        let inner = RustDataBank::new(config)?;
        Ok(Self { inner })
    }

    /// Register a dense dataset parsed by ``scdata.read``.
    ///
    /// ``store_path`` is the filesystem path to the ``.zarr`` directory holding
    /// the payload files; the dataset's ``payload_path`` is a key relative to
    /// it.  ZIP stores are not supported yet.
    fn register_dense(
        &mut self,
        py: Python<'_>,
        ds: Bound<'_, PyAny>,
        store_path: String,
    ) -> PyResult<PyDatasetId> {
        let gene_names: Vec<String> = ds.getattr("gene_names")?.extract()?;
        let data_obj = ds.getattr("data")?;
        let array_meta = build_array_meta(py, &data_obj, &store_path)?;
        let id = match array_meta.shape.len() {
            1 => self.inner.register_dense_1d(Dense1DMeta {
                gene_names,
                data: array_meta,
            })?,
            2 => self.inner.register_dense_2d(Dense2DMeta {
                gene_names,
                data: array_meta,
            })?,
            n => {
                return Err(PyValueError::new_err(format!(
                    "dense data must be 1D or 2D, got {n}D"
                )))
            }
        };
        Ok(id.into())
    }

    /// Register a CSR sparse dataset parsed by ``scdata.read``.
    ///
    /// ``store_path`` is the filesystem path to the ``.zarr`` directory holding
    /// the payload files; each array's ``payload_path`` is a key relative to it.
    fn register_sparse_csr(
        &mut self,
        py: Python<'_>,
        ds: Bound<'_, PyAny>,
        store_path: String,
    ) -> PyResult<PyDatasetId> {
        let gene_names: Vec<String> = ds.getattr("gene_names")?.extract()?;
        let indptr: Vec<u64> = ds.getattr("indptr")?.extract()?;
        let indices_obj = ds.getattr("indices")?;
        let data_obj = ds.getattr("data")?;
        let indices = build_array_meta(py, &indices_obj, &store_path)?;
        let data = build_array_meta(py, &data_obj, &store_path)?;
        let index_dtype = extract_dtype(&ds.getattr("index_dtype")?)?;
        let num_cells: usize = ds.getattr("num_cells")?.extract()?;
        let num_genes: usize = ds.getattr("num_genes")?.extract()?;
        let meta = SparseCsrDatasetMeta {
            gene_names,
            indptr,
            indices,
            data,
            index_dtype,
            num_cells,
            num_genes,
        };
        let id = self.inner.register_sparse_csr(meta)?;
        Ok(id.into())
    }

    /// Unregister a dataset, releasing its file handles and gene intern refs.
    fn unregister(&mut self, id: PyDatasetId) -> PyResult<()> {
        self.inner.unregister(id.into())?;
        Ok(())
    }

    /// Gene names for ``id``, in column order matching any access result.
    fn dataset_genes(&self, id: PyDatasetId) -> PyResult<Vec<String>> {
        let views = self.inner.dataset_genes(id.into())?;
        Ok(views.iter().map(|v| gene_view_to_string(*v)).collect())
    }

    /// Number of cells (rows) in the registered dataset.
    fn dataset_num_cells(&self, id: PyDatasetId) -> PyResult<usize> {
        Ok(self.inner.dataset_num_cells(id.into())?)
    }

    /// Number of genes (columns) in the registered dataset.
    fn dataset_num_genes(&self, id: PyDatasetId) -> PyResult<usize> {
        Ok(self.inner.dataset_num_genes(id.into())?)
    }

    /// Stored value dtype of the registered dataset, as a scdata ``DType``.
    fn dataset_dtype(&self, py: Python<'_>, id: PyDatasetId) -> PyResult<PyObject> {
        let dtype = self.inner.dataset_dtype(id.into())?;
        dtype_to_py(py, dtype)
    }

    /// Access cells and return a 1D numpy array of shape ``[len(cells) * num_genes]``.
    ///
    /// ``cells`` are cell indices into the registered dataset.  ``dtype`` is the
    /// scdata ``DType`` (or its ``.value`` string, e.g. ``"f32"``); when
    /// omitted it is inferred from the dataset's stored value dtype.  The
    /// result is row-major: cell ``i``'s genes occupy
    /// ``out[i*num_genes : (i+1)*num_genes]``.
    ///
    /// All numeric dtypes are supported.  ``f16`` returns numpy ``float16``;
    /// ``bf16`` returns ``uint16`` raw bit patterns (numpy has no native
    /// bfloat16) for the caller to view as bfloat16.
    #[pyo3(signature = (id, cells, dtype=None))]
    fn access_cells(
        &self,
        py: Python<'_>,
        id: PyDatasetId,
        cells: Vec<usize>,
        dtype: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let dtype = resolve_dtype(&self.inner, id.into(), dtype)?;
        access_cells_dispatch(py, &self.inner, id.into(), &cells, dtype)
    }

    /// Access cells projected onto a subset of gene names.
    ///
    /// Returns a 1D numpy array of shape ``[len(cells) * len(gene_names)]``.
    /// ``missing`` controls genes absent from the dataset: ``MissingGenePolicy.ZERO``
    /// fills a zero column, ``.ERROR`` raises.  Genes are returned in the
    /// requested order.
    #[pyo3(signature = (id, cells, gene_names, missing=None, dtype=None))]
    fn access_cells_by_gene_names(
        &self,
        py: Python<'_>,
        id: PyDatasetId,
        cells: Vec<usize>,
        gene_names: Vec<String>,
        missing: Option<PyMissingGenePolicy>,
        dtype: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let missing = missing.map(|m| m.inner).unwrap_or(MissingGenePolicy::Zero);
        let dtype = resolve_dtype(&self.inner, id.into(), dtype)?;
        access_cells_by_gene_names_dispatch(
            py,
            &self.inner,
            id.into(),
            &cells,
            &gene_names,
            missing,
            dtype,
        )
    }

    /// Build a scheduled prefetch iterator over a stream of cell batches.
    ///
    /// ``batches`` is an iterable of cell-index lists.  The returned iterator
    /// yields ``(cells, numpy_array, num_genes)`` tuples, where the array is
    /// row-major ``[len(cells) * num_genes]``.  ``config`` tunes the
    /// databank-level ring-buffer depth and the access-layer look-ahead.
    #[pyo3(signature = (id, batches, config=None))]
    fn prefetch_cells(
        &self,
        py: Python<'_>,
        id: PyDatasetId,
        batches: Bound<'_, PyAny>,
        config: Option<PyScheduledPrefetchConfig>,
    ) -> PyResult<PyPrefetchCells> {
        let config = config.map(|c| c.inner).unwrap_or_default();
        let dtype = self.inner.dataset_dtype(id.into())?;
        let batch_source = PyBatchSource::new(batches)?;
        prefetch_cells_dispatch(py, &self.inner, id.into(), batch_source, config, dtype)
    }

    /// Like ``prefetch_cells`` but each batch is projected onto ``gene_names``.
    #[pyo3(signature = (id, batches, gene_names, missing=None, config=None))]
    fn prefetch_cells_by_gene_names(
        &self,
        py: Python<'_>,
        id: PyDatasetId,
        batches: Bound<'_, PyAny>,
        gene_names: Vec<String>,
        missing: Option<PyMissingGenePolicy>,
        config: Option<PyScheduledPrefetchConfig>,
    ) -> PyResult<PyPrefetchCells> {
        let config = config.map(|c| c.inner).unwrap_or_default();
        let missing = missing.map(|m| m.inner).unwrap_or(MissingGenePolicy::Zero);
        let dtype = self.inner.dataset_dtype(id.into())?;
        let batch_source = PyBatchSource::new(batches)?;
        prefetch_cells_by_gene_names_dispatch(
            py,
            &self.inner,
            id.into(),
            batch_source,
            &gene_names,
            missing,
            config,
            dtype,
        )
    }

    fn __repr__(&self) -> &'static str {
        "DataBank(scdata-rust)"
    }
}

// ===========================================================================
// Cell access: dispatch on dtype to the concrete `access_cells_owned::<T>`.
// ===========================================================================

/// Resolve a dtype argument: explicit Python DType/string wins, else infer
/// from the dataset's stored value dtype.
fn resolve_dtype(
    bank: &RustDataBank,
    id: DatasetId,
    dtype: Option<Bound<'_, PyAny>>,
) -> PyResult<DType> {
    match dtype {
        Some(obj) => extract_dtype(&obj),
        None => Ok(bank.dataset_dtype(id.into())?),
    }
}

fn access_cells_dispatch(
    py: Python<'_>,
    bank: &RustDataBank,
    id: DatasetId,
    cells: &[usize],
    dtype: DType,
) -> PyResult<PyObject> {
    // Each arm builds a concrete `PyArray1<T>` and immediately erases it to a
    // `PyObject` so the match arms share one return type despite differing `T`.
    macro_rules! arm {
        ($ty:ty) => {{
            let out: Vec<$ty> = bank.access_cells_owned(id, cells)?;
            PyArray1::from_slice(py, &out).into_any().unbind()
        }};
    }
    let arr = match dtype {
        DType::U8 => arm!(u8),
        DType::I8 => arm!(i8),
        DType::U16 => arm!(u16),
        DType::I16 => arm!(i16),
        DType::U32 => arm!(u32),
        DType::I32 => arm!(i32),
        DType::U64 => arm!(u64),
        DType::I64 => arm!(i64),
        DType::F32 => arm!(f32),
        DType::F64 => arm!(f64),
        DType::F16 => {
            let out: Vec<F16Bits> = bank.access_cells_owned(id, cells)?;
            let bits = f16_bits_to_u16(out);
            PyArray1::from_slice(py, &bits).into_any().unbind()
        }
        DType::BF16 => {
            let out: Vec<Bf16Bits> = bank.access_cells_owned(id, cells)?;
            let bits = bf16_bits_to_u16(out);
            PyArray1::from_slice(py, &bits).into_any().unbind()
        }
    };
    Ok(arr)
}

fn access_cells_by_gene_names_dispatch(
    py: Python<'_>,
    bank: &RustDataBank,
    id: DatasetId,
    cells: &[usize],
    gene_names: &[String],
    missing: MissingGenePolicy,
    dtype: DType,
) -> PyResult<PyObject> {
    macro_rules! arm {
        ($ty:ty) => {{
            let out: Vec<$ty> =
                bank.access_cells_owned_by_gene_names(id, cells, gene_names, missing)?;
            PyArray1::from_slice(py, &out).into_any().unbind()
        }};
    }
    let arr = match dtype {
        DType::U8 => arm!(u8),
        DType::I8 => arm!(i8),
        DType::U16 => arm!(u16),
        DType::I16 => arm!(i16),
        DType::U32 => arm!(u32),
        DType::I32 => arm!(i32),
        DType::U64 => arm!(u64),
        DType::I64 => arm!(i64),
        DType::F32 => arm!(f32),
        DType::F64 => arm!(f64),
        DType::F16 => {
            let out: Vec<F16Bits> =
                bank.access_cells_owned_by_gene_names(id, cells, gene_names, missing)?;
            let bits = f16_bits_to_u16(out);
            PyArray1::from_slice(py, &bits).into_any().unbind()
        }
        DType::BF16 => {
            let out: Vec<Bf16Bits> =
                bank.access_cells_owned_by_gene_names(id, cells, gene_names, missing)?;
            let bits = bf16_bits_to_u16(out);
            PyArray1::from_slice(py, &bits).into_any().unbind()
        }
    };
    Ok(arr)
}

/// Reinterpret a `Vec<F16Bits>` as `Vec<u16>` bit patterns (numpy float16).
///
/// `F16Bits` is `#[repr(transparent)]` over `u16`, so the reinterpretation is
/// sound and preserves the native-endian half-precision bit layout numpy
/// expects for `float16`.
fn f16_bits_to_u16(bits: Vec<F16Bits>) -> Vec<u16> {
    // SAFETY: F16Bits is #[repr(transparent)] over u16; same size, alignment,
    // and layout.  The Vec ownership is transferred via raw parts.
    debug_assert_eq!(std::mem::size_of::<F16Bits>(), std::mem::size_of::<u16>());
    let mut bits = bits;
    let ptr = bits.as_mut_ptr() as *mut u16;
    let len = bits.len();
    let cap = bits.capacity();
    std::mem::forget(bits);
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

/// Reinterpret a `Vec<Bf16Bits>` as `Vec<u16>` bit patterns (numpy uint16).
fn bf16_bits_to_u16(bits: Vec<Bf16Bits>) -> Vec<u16> {
    // SAFETY: Bf16Bits is #[repr(transparent)] over u16; same layout.
    debug_assert_eq!(std::mem::size_of::<Bf16Bits>(), std::mem::size_of::<u16>());
    let mut bits = bits;
    let ptr = bits.as_mut_ptr() as *mut u16;
    let len = bits.len();
    let cap = bits.capacity();
    std::mem::forget(bits);
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

// ===========================================================================
// Scheduled prefetch iterator
// ===========================================================================

/// Python-facing batch source: cell-index lists pre-collected from any iterable.
///
/// The Python iterable is fully drained under the GIL at construction time into
/// an owned `Vec<Vec<usize>>`.  This keeps the Rust producer thread off the
/// GIL entirely — without this, the producer would block waiting for the GIL
/// while the consumer (which holds it) blocks waiting for the producer, a
/// classic deadlock.  For Python callers the batch list is already in memory,
/// so pre-collecting costs nothing; truly streaming sources should feed Rust
/// directly rather than through this Python wrapper.
struct PyBatchSource {
    batches: std::vec::IntoIter<Vec<usize>>,
}

impl PyBatchSource {
    fn new(batches: Bound<'_, PyAny>) -> PyResult<Self> {
        let mut out = Vec::new();
        for item in batches.try_iter()? {
            out.push(item?.extract::<Vec<usize>>()?);
        }
        Ok(Self { batches: out.into_iter() })
    }
}

impl Iterator for PyBatchSource {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Vec<usize>> {
        self.batches.next()
    }
}

/// Owned prefetch iterator that erases the value type, so a single pyclass can
/// hold any-dtype iterator.
enum PrefetchDispatch {
    U8(PrefetchCells<u8>),
    I8(PrefetchCells<i8>),
    U16(PrefetchCells<u16>),
    I16(PrefetchCells<i16>),
    U32(PrefetchCells<u32>),
    I32(PrefetchCells<i32>),
    U64(PrefetchCells<u64>),
    I64(PrefetchCells<i64>),
    F32(PrefetchCells<f32>),
    F64(PrefetchCells<f64>),
    F16(PrefetchCells<F16Bits>),
    BF16(PrefetchCells<Bf16Bits>),
}

impl Iterator for PrefetchDispatch {
    type Item = DataBankResult<PrefetchedBatchAny>;

    fn next(&mut self) -> Option<Self::Item> {
        // Pull the next batch from the typed producer and erase its buffer into
        // `PrefetchedBufferAny`.  One match arm per dtype variant keeps the
        // dispatch monomorphic on the producer side.
        match self {
            PrefetchDispatch::U8(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny { cells: b.cells, buffer: PrefetchedBufferAny::U8(b.buffer), num_genes: b.num_genes })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
            PrefetchDispatch::I8(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny { cells: b.cells, buffer: PrefetchedBufferAny::I8(b.buffer), num_genes: b.num_genes })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
            PrefetchDispatch::U16(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny { cells: b.cells, buffer: PrefetchedBufferAny::U16(b.buffer), num_genes: b.num_genes })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
            PrefetchDispatch::I16(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny { cells: b.cells, buffer: PrefetchedBufferAny::I16(b.buffer), num_genes: b.num_genes })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
            PrefetchDispatch::U32(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny { cells: b.cells, buffer: PrefetchedBufferAny::U32(b.buffer), num_genes: b.num_genes })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
            PrefetchDispatch::I32(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny { cells: b.cells, buffer: PrefetchedBufferAny::I32(b.buffer), num_genes: b.num_genes })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
            PrefetchDispatch::U64(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny { cells: b.cells, buffer: PrefetchedBufferAny::U64(b.buffer), num_genes: b.num_genes })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
            PrefetchDispatch::I64(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny { cells: b.cells, buffer: PrefetchedBufferAny::I64(b.buffer), num_genes: b.num_genes })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
            PrefetchDispatch::F32(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny { cells: b.cells, buffer: PrefetchedBufferAny::F32(b.buffer), num_genes: b.num_genes })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
            PrefetchDispatch::F64(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny { cells: b.cells, buffer: PrefetchedBufferAny::F64(b.buffer), num_genes: b.num_genes })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
            PrefetchDispatch::F16(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny { cells: b.cells, buffer: PrefetchedBufferAny::F16(b.buffer), num_genes: b.num_genes })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
            PrefetchDispatch::BF16(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny { cells: b.cells, buffer: PrefetchedBufferAny::BF16(b.buffer), num_genes: b.num_genes })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
        }
    }
}

enum PrefetchedBufferAny {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    F16(Vec<F16Bits>),
    BF16(Vec<Bf16Bits>),
}

struct PrefetchedBatchAny {
    cells: Vec<usize>,
    buffer: PrefetchedBufferAny,
    num_genes: usize,
}

#[pyclass(name = "_PrefetchCells", module = "scdata._scdata")]
struct PyPrefetchCells {
    inner: Option<PrefetchDispatch>,
}

impl PyPrefetchCells {
    fn new(inner: PrefetchDispatch) -> Self {
        Self { inner: Some(inner) }
    }
}

#[pymethods]
impl PyPrefetchCells {
    fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let Some(iter) = self.inner.as_mut() else {
            return Ok(None);
        };
        match iter.next() {
            None => {
                self.inner = None;
                Ok(None)
            }
            Some(Ok(batch)) => {
                let arr = buffer_to_numpy(py, batch.buffer)?;
                let cells = PyArray1::from_slice(py, &batch.cells).into_any().unbind();
                let tuple = (cells, arr, batch.num_genes).into_pyobject(py)?;
                Ok(Some(tuple.into_any().unbind()))
            }
            Some(Err(err)) => Err(err.into()),
        }
    }
}

fn buffer_to_numpy(py: Python<'_>, buffer: PrefetchedBufferAny) -> PyResult<PyObject> {
    let arr = match buffer {
        PrefetchedBufferAny::U8(v) => PyArray1::from_slice(py, &v).into_any().unbind(),
        PrefetchedBufferAny::I8(v) => PyArray1::from_slice(py, &v).into_any().unbind(),
        PrefetchedBufferAny::U16(v) => PyArray1::from_slice(py, &v).into_any().unbind(),
        PrefetchedBufferAny::I16(v) => PyArray1::from_slice(py, &v).into_any().unbind(),
        PrefetchedBufferAny::U32(v) => PyArray1::from_slice(py, &v).into_any().unbind(),
        PrefetchedBufferAny::I32(v) => PyArray1::from_slice(py, &v).into_any().unbind(),
        PrefetchedBufferAny::U64(v) => PyArray1::from_slice(py, &v).into_any().unbind(),
        PrefetchedBufferAny::I64(v) => PyArray1::from_slice(py, &v).into_any().unbind(),
        PrefetchedBufferAny::F32(v) => PyArray1::from_slice(py, &v).into_any().unbind(),
        PrefetchedBufferAny::F64(v) => PyArray1::from_slice(py, &v).into_any().unbind(),
        PrefetchedBufferAny::F16(v) => {
            let bits = f16_bits_to_u16(v);
            PyArray1::from_slice(py, &bits).into_any().unbind()
        }
        PrefetchedBufferAny::BF16(v) => {
            let bits = bf16_bits_to_u16(v);
            PyArray1::from_slice(py, &bits).into_any().unbind()
        }
    };
    Ok(arr)
}

fn prefetch_cells_dispatch(
    _py: Python<'_>,
    bank: &RustDataBank,
    id: DatasetId,
    batch_source: PyBatchSource,
    config: ScheduledPrefetchConfig,
    dtype: DType,
) -> PyResult<PyPrefetchCells> {
    let dispatch = match dtype {
        DType::U8 => PrefetchDispatch::U8(bank.prefetch_cells_scheduled(id, batch_source, config)?),
        DType::I8 => PrefetchDispatch::I8(bank.prefetch_cells_scheduled(id, batch_source, config)?),
        DType::U16 => PrefetchDispatch::U16(bank.prefetch_cells_scheduled(id, batch_source, config)?),
        DType::I16 => PrefetchDispatch::I16(bank.prefetch_cells_scheduled(id, batch_source, config)?),
        DType::U32 => PrefetchDispatch::U32(bank.prefetch_cells_scheduled(id, batch_source, config)?),
        DType::I32 => PrefetchDispatch::I32(bank.prefetch_cells_scheduled(id, batch_source, config)?),
        DType::U64 => PrefetchDispatch::U64(bank.prefetch_cells_scheduled(id, batch_source, config)?),
        DType::I64 => PrefetchDispatch::I64(bank.prefetch_cells_scheduled(id, batch_source, config)?),
        DType::F32 => PrefetchDispatch::F32(bank.prefetch_cells_scheduled(id, batch_source, config)?),
        DType::F64 => PrefetchDispatch::F64(bank.prefetch_cells_scheduled(id, batch_source, config)?),
        DType::F16 => PrefetchDispatch::F16(bank.prefetch_cells_scheduled(id, batch_source, config)?),
        DType::BF16 => PrefetchDispatch::BF16(bank.prefetch_cells_scheduled(id, batch_source, config)?),
    };
    Ok(PyPrefetchCells::new(dispatch))
}

fn prefetch_cells_by_gene_names_dispatch(
    _py: Python<'_>,
    bank: &RustDataBank,
    id: DatasetId,
    batch_source: PyBatchSource,
    gene_names: &[String],
    missing: MissingGenePolicy,
    config: ScheduledPrefetchConfig,
    dtype: DType,
) -> PyResult<PyPrefetchCells> {
    let dispatch = match dtype {
        DType::U8 => PrefetchDispatch::U8(bank.prefetch_cells_scheduled_by_gene_names(id, batch_source, gene_names, missing, config)?),
        DType::I8 => PrefetchDispatch::I8(bank.prefetch_cells_scheduled_by_gene_names(id, batch_source, gene_names, missing, config)?),
        DType::U16 => PrefetchDispatch::U16(bank.prefetch_cells_scheduled_by_gene_names(id, batch_source, gene_names, missing, config)?),
        DType::I16 => PrefetchDispatch::I16(bank.prefetch_cells_scheduled_by_gene_names(id, batch_source, gene_names, missing, config)?),
        DType::U32 => PrefetchDispatch::U32(bank.prefetch_cells_scheduled_by_gene_names(id, batch_source, gene_names, missing, config)?),
        DType::I32 => PrefetchDispatch::I32(bank.prefetch_cells_scheduled_by_gene_names(id, batch_source, gene_names, missing, config)?),
        DType::U64 => PrefetchDispatch::U64(bank.prefetch_cells_scheduled_by_gene_names(id, batch_source, gene_names, missing, config)?),
        DType::I64 => PrefetchDispatch::I64(bank.prefetch_cells_scheduled_by_gene_names(id, batch_source, gene_names, missing, config)?),
        DType::F32 => PrefetchDispatch::F32(bank.prefetch_cells_scheduled_by_gene_names(id, batch_source, gene_names, missing, config)?),
        DType::F64 => PrefetchDispatch::F64(bank.prefetch_cells_scheduled_by_gene_names(id, batch_source, gene_names, missing, config)?),
        DType::F16 => PrefetchDispatch::F16(bank.prefetch_cells_scheduled_by_gene_names(id, batch_source, gene_names, missing, config)?),
        DType::BF16 => PrefetchDispatch::BF16(bank.prefetch_cells_scheduled_by_gene_names(id, batch_source, gene_names, missing, config)?),
    };
    Ok(PyPrefetchCells::new(dispatch))
}

// ===========================================================================
// Python dataset -> Rust meta helpers
// ===========================================================================

fn build_array_meta(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    store_path: &str,
) -> PyResult<ArrayMeta> {
    let shape: Vec<usize> = data.getattr("shape")?.extract()?;
    let chunk_shape: Vec<usize> = data.getattr("chunk_shape")?.extract()?;
    if shape.len() != chunk_shape.len() {
        return Err(PyValueError::new_err(format!(
            "shape rank {} != chunk_shape rank {}",
            shape.len(),
            chunk_shape.len()
        )));
    }
    let chunk_grid_shape: Vec<usize> = shape
        .iter()
        .zip(chunk_shape.iter())
        .map(|(&s, &c)| div_ceil(s, c))
        .collect();

    let dtype = extract_dtype(&data.getattr("dtype")?)?;
    let codec = build_codec(py, &data.getattr("codec")?)?;

    let payload_path: String = data.getattr("payload_path")?.extract()?;
    let locations = extract_locations(&data.getattr("chunks")?)?;
    let chunks = ChunkStoreMeta::FileOffset {
        path: PathBuf::from(store_path).join(payload_path),
        locations,
    };

    Ok(ArrayMeta {
        shape,
        chunk_shape,
        chunk_grid_shape,
        dtype,
        order: ArrayOrder::C,
        codec,
        chunks,
    })
}

fn extract_dtype(dtype: &Bound<'_, PyAny>) -> PyResult<DType> {
    // Python `DType` is a str enum; `.value` is the lowercase code ("f32", ...).
    let value: String = dtype.getattr("value")?.extract()?;
    match value.as_str() {
        "u8" => Ok(DType::U8),
        "i8" => Ok(DType::I8),
        "u16" => Ok(DType::U16),
        "i16" => Ok(DType::I16),
        "u32" => Ok(DType::U32),
        "i32" => Ok(DType::I32),
        "u64" => Ok(DType::U64),
        "i64" => Ok(DType::I64),
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        "f32" => Ok(DType::F32),
        "f64" => Ok(DType::F64),
        other => Err(PyValueError::new_err(format!("unknown dtype {other:?}"))),
    }
}

/// Build the Python-side `DType` enum object matching a Rust `DType`.
fn dtype_to_py(py: Python<'_>, dtype: DType) -> PyResult<PyObject> {
    let code = match dtype {
        DType::U8 => "u8",
        DType::I8 => "i8",
        DType::U16 => "u16",
        DType::I16 => "i16",
        DType::U32 => "u32",
        DType::I32 => "i32",
        DType::U64 => "u64",
        DType::I64 => "i64",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::F32 => "f32",
        DType::F64 => "f64",
    };
    // `DType` is a Python str enum whose value is the lowercase code, so
    // `DType("f32")` resolves to the `DType.F32` member.
    let data_mod = py.import("scdata.data")?;
    let dtype_cls = data_mod.getattr("DType")?;
    let member = dtype_cls.call1((code,))?;
    Ok(member.unbind())
}

fn build_codec(py: Python<'_>, codec: &Bound<'_, PyAny>) -> PyResult<ArrayCodecMeta> {
    let is_uncompressed: bool = codec.getattr("is_uncompressed")?.extract()?;
    if is_uncompressed {
        return Ok(ArrayCodecMeta::Uncompressed);
    }

    // `CodecPipeline.to_zarr()` -> (filters: list[dict] | None, compressor: dict | None)
    let pair = codec.getattr("to_zarr")?.call0()?;
    let filters_obj = pair.get_item(0)?;
    let compressor_obj = pair.get_item(1)?;

    let json = py.import("json")?;
    let dumps = json.getattr("dumps")?;
    let filters = json_opt(&dumps, &filters_obj)?;
    let compressor = json_opt(&dumps, &compressor_obj)?;
    Ok(ArrayCodecMeta::ZarrV2Json {
        filters,
        compressor,
    })
}

fn json_opt(dumps: &Bound<'_, PyAny>, obj: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if obj.is_none() {
        return Ok(None);
    }
    let s = dumps.call1((obj.clone().unbind(),))?.extract::<String>()?;
    Ok(Some(s))
}

fn extract_locations(chunks: &Bound<'_, PyAny>) -> PyResult<Vec<FileChunkLocation>> {
    let n = chunks.len()?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let item = chunks.get_item(i)?;
        let offset: u64 = item.getattr("offset")?.extract()?;
        let len: usize = item.getattr("length")?.extract()?;
        out.push(FileChunkLocation { offset, len });
    }
    Ok(out)
}

fn gene_view_to_string(view: GeneNameView) -> String {
    if view.is_empty() {
        return String::new();
    }
    // SAFETY: `view` points into an `Arc<str>` owned by a registered dataset's
    // gene table.  The caller (`PyDataBank::dataset_genes`) holds `&self` —
    // i.e. a borrow of the DataBank — for the duration of this call, so the
    // owning dataset cannot be unregistered concurrently and the pointer
    // remains valid.  Bytes are UTF-8 by construction (interned from `String`).
    let bytes = unsafe { std::slice::from_raw_parts(view.ptr, view.len) };
    String::from_utf8_lossy(bytes).into_owned()
}

fn div_ceil(n: usize, d: usize) -> usize {
    n / d + usize::from(n % d != 0)
}

// Keep `PyString` import live for future payload-key extraction paths.
#[allow(unused_imports)]
use pyo3::types::PyString as _PyString;
