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
//! Python ``ArrayMeta`` keeps zarr logical keys (``payload_path`` /
//! ``chunk_paths``) and may also carry resolved local source files
//! (``payload_file_path`` / ``chunk_file_paths``).  When those source paths are
//! present, the binding passes them directly to Rust with the provided offsets;
//! otherwise it falls back to the legacy ``store_path / key`` join.  This lets
//! directory and ZIP_STORED stores share the same Rust file/off/len abstraction.
//!
//! Cell access dispatches on dtype: `access_cells_owned::<T>` is generic in
//! Rust, but pyo3 cannot lift that generic across the GIL, so each numeric
//! dtype has its own call arm (`access_cells_dispatch`).  The Rust core
//! allocates and fills a `Vec<T>`, which is handed directly to numpy through
//! `IntoPyArray` so the return path does not copy the decoded buffer.  `f16`
//! reinterprets the opaque
//! `F16Bits` bits as numpy `float16`; `bf16` has no numpy dtype, so its raw
//! bit pattern is returned as `uint16` for the caller to view as bfloat16.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::create_exception;
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::access::{AccessConfig, AccessCpuConfig, ScheduledAccessConfig};
use crate::codecs::{codec_pipeline_from_zarr_v2_json_str, DecodePoolConfig, SharedCodec};
use crate::databank::DataBankError as RustDataBankError;
use crate::databank::{
    ArrayCodecMeta, ArrayMeta, ArrayOrder, Bf16Bits, ChunkStoreMeta, DType,
    DataBank as RustDataBank, DataBankConfig, DataBankResult, DatasetId, Dense1DMeta, Dense2DMeta,
    DirectoryChunkLocationMeta, F16Bits, FileChunkLocation, GeneNameView, MissingGenePolicy,
    PrefetchCells, ScheduledPrefetchConfig, SparseCsrDatasetMeta,
};
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
    m.add_function(wrap_pyfunction!(_zip_stored_offsets, m)?)?;
    m.add_function(wrap_pyfunction!(_decode_index_payload, m)?)?;
    m.add_function(wrap_pyfunction!(_decode_index_chunks, m)?)?;
    m.add("DataBankError", m.py().get_type::<DataBankError>())?;
    Ok(())
}

#[pyfunction]
fn _zip_stored_offsets(path: String, header_offsets: Vec<u64>) -> PyResult<Vec<u64>> {
    let mut file = File::open(&path)
        .map_err(|err| PyOSError::new_err(format!("cannot open zip archive {path}: {err}")))?;
    let mut out = Vec::with_capacity(header_offsets.len());
    let mut header = [0u8; 30];
    for offset in header_offsets {
        file.seek(SeekFrom::Start(offset)).map_err(|err| {
            PyOSError::new_err(format!("cannot seek zip local header at {offset}: {err}"))
        })?;
        file.read_exact(&mut header).map_err(|err| {
            PyOSError::new_err(format!("cannot read zip local header at {offset}: {err}"))
        })?;
        if &header[..4] != b"PK\x03\x04" {
            return Err(PyValueError::new_err(format!(
                "invalid zip local header at {offset}"
            )));
        }
        let filename_len = u16::from_le_bytes([header[26], header[27]]) as u64;
        let extra_len = u16::from_le_bytes([header[28], header[29]]) as u64;
        out.push(offset + 30 + filename_len + extra_len);
    }
    Ok(out)
}

#[pyfunction]
fn _decode_index_payload(
    py: Python<'_>,
    payload: Bound<'_, PyBytes>,
    offsets: Bound<'_, PyAny>,
    lengths: Bound<'_, PyAny>,
    dtype: Bound<'_, PyAny>,
    codec: Bound<'_, PyAny>,
    count: usize,
) -> PyResult<PyObject> {
    let offsets = extract_u64_vec(&offsets, "offsets")?;
    let lengths = extract_u64_vec(&lengths, "lengths")?;
    if offsets.len() != lengths.len() {
        return Err(PyValueError::new_err(format!(
            "offsets length {} != lengths length {}",
            offsets.len(),
            lengths.len()
        )));
    }
    let dtype = extract_dtype(&dtype)?;
    let codec = build_shared_codec(py, &codec)?;
    let payload = payload.as_bytes();
    let mut out = Vec::new();
    for (offset, len) in offsets.into_iter().zip(lengths) {
        let start = u64_to_usize(offset, "offsets")?;
        let len = u64_to_usize(len, "lengths")?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| PyValueError::new_err("index chunk byte range overflows usize"))?;
        let raw = payload.get(start..end).ok_or_else(|| {
            PyValueError::new_err(format!(
                "index chunk range [{start}, {end}) exceeds payload size {}",
                payload.len()
            ))
        })?;
        decode_index_chunk_into(raw, dtype, codec.as_ref(), &mut out)?;
    }
    finalize_index_output(py, out, count, false)
}

#[pyfunction]
fn _decode_index_chunks(
    py: Python<'_>,
    chunks: Bound<'_, PyAny>,
    dtype: Bound<'_, PyAny>,
    codec: Bound<'_, PyAny>,
    count: usize,
) -> PyResult<PyObject> {
    let dtype = extract_dtype(&dtype)?;
    let codec = build_shared_codec(py, &codec)?;
    let mut out = Vec::new();
    for item in chunks.try_iter()? {
        let item = item?;
        let raw = item.downcast::<PyBytes>()?;
        decode_index_chunk_into(raw.as_bytes(), dtype, codec.as_ref(), &mut out)?;
    }
    finalize_index_output(py, out, count, true)
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
        format!(
            "DatasetId(slot={}, generation={})",
            self.slot, self.generation
        )
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.extract::<PyDatasetId>().is_ok_and(|o| *self == o)
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
    #[allow(non_snake_case)]
    fn ZERO() -> Self {
        Self {
            inner: MissingGenePolicy::Zero,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn ERROR() -> Self {
        Self {
            inner: MissingGenePolicy::Error,
        }
    }

    #[new]
    fn new(policy: &str) -> PyResult<Self> {
        match policy.to_ascii_lowercase().as_str() {
            "zero" => Ok(Self {
                inner: MissingGenePolicy::Zero,
            }),
            "error" => Ok(Self {
                inner: MissingGenePolicy::Error,
            }),
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
        Self {
            inner: BaseIoConfig::default(),
        }
    }
    #[getter]
    fn get_max_in_flight(&self) -> usize {
        self.inner.max_in_flight
    }
    #[setter]
    fn set_max_in_flight(&mut self, value: usize) {
        self.inner.max_in_flight = value;
    }
    #[getter]
    fn get_priority_levels(&self) -> usize {
        self.inner.priority_levels
    }
    #[setter]
    fn set_priority_levels(&mut self, value: usize) {
        self.inner.priority_levels = value;
    }
    #[getter]
    fn get_queue_shards(&self) -> usize {
        self.inner.queue_shards
    }
    #[setter]
    fn set_queue_shards(&mut self, value: usize) {
        self.inner.queue_shards = value;
    }
    #[getter]
    fn get_assume_non_overlapping_reads(&self) -> bool {
        self.inner.assume_non_overlapping_reads
    }
    #[setter]
    fn set_assume_non_overlapping_reads(&mut self, value: bool) {
        self.inner.assume_non_overlapping_reads = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "BaseIoConfig(max_in_flight={}, queue_shards={})",
            self.inner.max_in_flight, self.inner.queue_shards
        )
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
        Self {
            inner: UringConfig::default(),
        }
    }
    #[getter]
    fn get_base(&self) -> PyBaseIoConfig {
        PyBaseIoConfig {
            inner: self.inner.base.clone(),
        }
    }
    #[setter]
    fn set_base(&mut self, value: PyBaseIoConfig) {
        self.inner.base = value.inner;
    }
    #[getter]
    fn get_entries(&self) -> u32 {
        self.inner.entries
    }
    #[setter]
    fn set_entries(&mut self, value: u32) {
        self.inner.entries = value;
    }
    #[getter]
    fn get_drivers(&self) -> usize {
        self.inner.drivers
    }
    #[setter]
    fn set_drivers(&mut self, value: usize) {
        self.inner.drivers = value;
    }
    #[getter]
    fn get_iowq_bounded_workers(&self) -> u32 {
        self.inner.iowq_bounded_workers
    }
    #[setter]
    fn set_iowq_bounded_workers(&mut self, value: u32) {
        self.inner.iowq_bounded_workers = value;
    }
    #[getter]
    fn get_iowq_unbounded_workers(&self) -> u32 {
        self.inner.iowq_unbounded_workers
    }
    #[setter]
    fn set_iowq_unbounded_workers(&mut self, value: u32) {
        self.inner.iowq_unbounded_workers = value;
    }
    #[getter]
    fn get_registered_files(&self) -> u32 {
        self.inner.registered_files
    }
    #[setter]
    fn set_registered_files(&mut self, value: u32) {
        self.inner.registered_files = value;
    }
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
        Self {
            inner: ThreadedConfig::default(),
        }
    }
    #[getter]
    fn get_base(&self) -> PyBaseIoConfig {
        PyBaseIoConfig {
            inner: self.inner.base.clone(),
        }
    }
    #[setter]
    fn set_base(&mut self, value: PyBaseIoConfig) {
        self.inner.base = value.inner;
    }
    #[getter]
    fn get_num_workers(&self) -> usize {
        self.inner.num_workers
    }
    #[setter]
    fn set_num_workers(&mut self, value: usize) {
        self.inner.num_workers = value;
    }
    #[getter]
    fn get_cpus(&self) -> Option<Vec<usize>> {
        self.inner.cpus.clone()
    }
    #[setter]
    fn set_cpus(&mut self, value: Option<Vec<usize>>) {
        self.inner.cpus = value;
    }
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
        Self {
            inner: IoConfig::default(),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (config=None))]
    fn uring(config: Option<PyUringConfig>) -> Self {
        Self {
            inner: IoConfig::Uring(config.map(|c| c.inner).unwrap_or_default()),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (config=None))]
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
        Self {
            inner: DecodePoolConfig::default(),
        }
    }
    #[getter]
    fn get_num_workers(&self) -> usize {
        self.inner.num_workers
    }
    #[setter]
    fn set_num_workers(&mut self, value: usize) {
        self.inner.num_workers = value;
    }
    #[getter]
    fn get_queue_capacity(&self) -> usize {
        self.inner.queue_capacity
    }
    #[setter]
    fn set_queue_capacity(&mut self, value: usize) {
        self.inner.queue_capacity = value;
    }
    #[getter]
    fn get_cpus(&self) -> Option<Vec<usize>> {
        self.inner.cpus.clone()
    }
    #[setter]
    fn set_cpus(&mut self, value: Option<Vec<usize>>) {
        self.inner.cpus = value;
    }
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
        Self {
            inner: AccessCpuConfig::default(),
        }
    }
    #[getter]
    fn get_num_workers(&self) -> usize {
        self.inner.num_workers
    }
    #[setter]
    fn set_num_workers(&mut self, value: usize) {
        self.inner.num_workers = value;
    }
    #[getter]
    fn get_queue_capacity(&self) -> usize {
        self.inner.queue_capacity
    }
    #[setter]
    fn set_queue_capacity(&mut self, value: usize) {
        self.inner.queue_capacity = value;
    }
    #[getter]
    fn get_cpus(&self) -> Option<Vec<usize>> {
        self.inner.cpus.clone()
    }
    #[setter]
    fn set_cpus(&mut self, value: Option<Vec<usize>>) {
        self.inner.cpus = value;
    }
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
        Self {
            inner: AccessConfig::default(),
        }
    }
    #[getter]
    fn get_queue_capacity(&self) -> usize {
        self.inner.queue_capacity
    }
    #[setter]
    fn set_queue_capacity(&mut self, value: usize) {
        self.inner.queue_capacity = value;
    }
    #[getter]
    fn get_scheduler_shards(&self) -> usize {
        self.inner.scheduler_shards
    }
    #[setter]
    fn set_scheduler_shards(&mut self, value: usize) {
        self.inner.scheduler_shards = value;
    }
    #[getter]
    fn get_cache_capacity_bytes(&self) -> usize {
        self.inner.cache_capacity_bytes
    }
    #[setter]
    fn set_cache_capacity_bytes(&mut self, value: usize) {
        self.inner.cache_capacity_bytes = value;
    }
    #[getter]
    fn get_memory_budget_bytes(&self) -> usize {
        self.inner.memory_budget_bytes
    }
    #[setter]
    fn set_memory_budget_bytes(&mut self, value: usize) {
        self.inner.memory_budget_bytes = value;
    }
    #[getter]
    fn get_default_io_priority(&self) -> u8 {
        self.inner.default_io_priority
    }
    #[setter]
    fn set_default_io_priority(&mut self, value: u8) {
        self.inner.default_io_priority = value;
    }
    #[getter]
    fn get_keep_decoded(&self) -> bool {
        self.inner.keep_decoded
    }
    #[setter]
    fn set_keep_decoded(&mut self, value: bool) {
        self.inner.keep_decoded = value;
    }
    #[getter]
    fn get_cpu(&self) -> PyAccessCpuConfig {
        PyAccessCpuConfig {
            inner: self.inner.cpu.clone(),
        }
    }
    #[setter]
    fn set_cpu(&mut self, value: PyAccessCpuConfig) {
        self.inner.cpu = value.inner;
    }
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
        Self {
            inner: crate::databank::FillConfig::default(),
        }
    }
    #[getter]
    fn get_parallel(&self) -> bool {
        self.inner.parallel
    }
    #[setter]
    fn set_parallel(&mut self, value: bool) {
        self.inner.parallel = value;
    }
    #[getter]
    fn get_num_workers(&self) -> usize {
        self.inner.num_workers
    }
    #[setter]
    fn set_num_workers(&mut self, value: usize) {
        self.inner.num_workers = value;
    }
    #[getter]
    fn get_queue_capacity(&self) -> usize {
        self.inner.queue_capacity
    }
    #[setter]
    fn set_queue_capacity(&mut self, value: usize) {
        self.inner.queue_capacity = value;
    }
    #[getter]
    fn get_min_parallel_rows(&self) -> usize {
        self.inner.min_parallel_rows
    }
    #[setter]
    fn set_min_parallel_rows(&mut self, value: usize) {
        self.inner.min_parallel_rows = value;
    }
    #[getter]
    fn get_min_parallel_bytes(&self) -> usize {
        self.inner.min_parallel_bytes
    }
    #[setter]
    fn set_min_parallel_bytes(&mut self, value: usize) {
        self.inner.min_parallel_bytes = value;
    }
    #[getter]
    fn get_cpus(&self) -> Option<Vec<usize>> {
        self.inner.cpus.clone()
    }
    #[setter]
    fn set_cpus(&mut self, value: Option<Vec<usize>>) {
        self.inner.cpus = value;
    }
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
        Self {
            inner: DataBankConfig::default(),
        }
    }

    #[getter]
    fn get_io_config(&self) -> PyIoConfig {
        PyIoConfig {
            inner: self.inner.io_config.clone(),
        }
    }
    #[setter]
    fn set_io_config(&mut self, value: PyIoConfig) {
        self.inner.io_config = value.inner;
    }
    #[getter]
    fn get_decode_config(&self) -> PyDecodePoolConfig {
        PyDecodePoolConfig {
            inner: self.inner.decode_config.clone(),
        }
    }
    #[setter]
    fn set_decode_config(&mut self, value: PyDecodePoolConfig) {
        self.inner.decode_config = value.inner;
    }
    #[getter]
    fn get_access_config(&self) -> PyAccessConfig {
        PyAccessConfig {
            inner: self.inner.access_config.clone(),
        }
    }
    #[setter]
    fn set_access_config(&mut self, value: PyAccessConfig) {
        self.inner.access_config = value.inner;
    }
    #[getter]
    fn get_fill_config(&self) -> PyFillConfig {
        PyFillConfig {
            inner: self.inner.fill_config.clone(),
        }
    }
    #[setter]
    fn set_fill_config(&mut self, value: PyFillConfig) {
        self.inner.fill_config = value.inner;
    }
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
        Self {
            inner: ScheduledAccessConfig::default(),
        }
    }
    #[getter]
    fn get_prefetch_step(&self) -> usize {
        self.inner.prefetch_step
    }
    #[setter]
    fn set_prefetch_step(&mut self, value: usize) {
        self.inner.prefetch_step = value;
    }
    #[getter]
    fn get_decode_ahead_steps(&self) -> usize {
        self.inner.decode_ahead_steps
    }
    #[setter]
    fn set_decode_ahead_steps(&mut self, value: usize) {
        self.inner.decode_ahead_steps = value;
    }
    #[getter]
    fn get_ready_ahead_steps(&self) -> usize {
        self.inner.ready_ahead_steps
    }
    #[setter]
    fn set_ready_ahead_steps(&mut self, value: usize) {
        self.inner.ready_ahead_steps = value;
    }
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
        Self {
            inner: ScheduledPrefetchConfig::default(),
        }
    }
    #[getter]
    fn get_prefetch_step(&self) -> usize {
        self.inner.prefetch_step
    }
    #[setter]
    fn set_prefetch_step(&mut self, value: usize) {
        self.inner.prefetch_step = value;
    }
    #[getter]
    fn get_access(&self) -> PyScheduledAccessConfig {
        PyScheduledAccessConfig {
            inner: self.inner.access,
        }
    }
    #[setter]
    fn set_access(&mut self, value: PyScheduledAccessConfig) {
        self.inner.access = value.inner;
    }
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
    /// ``store_path`` is the filesystem path to the zarr directory or
    /// ZIP_STORED archive.  Datasets produced by ``scdata.io.launch`` already
    /// carry resolved source files; manually-built datasets use ``store_path``
    /// as the root for their logical keys.
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
    /// ``store_path`` is the filesystem path to the zarr directory or
    /// ZIP_STORED archive.
    fn register_sparse_csr(
        &mut self,
        py: Python<'_>,
        ds: Bound<'_, PyAny>,
        store_path: String,
    ) -> PyResult<PyDatasetId> {
        let gene_names: Vec<String> = ds.getattr("gene_names")?.extract()?;
        let indptr = extract_u64_vec(&ds.getattr("indptr")?, "indptr")?;
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

    /// Fast path for numpy ``intp`` cell-index arrays.
    ///
    /// This avoids Python ``list`` / Python ``int`` materialization on the hot
    /// access boundary.  The public Python wrapper normalizes user input to a
    /// contiguous ``np.intp`` array and calls this method when available.
    #[pyo3(signature = (id, cells, dtype=None))]
    fn access_cells_array(
        &self,
        py: Python<'_>,
        id: PyDatasetId,
        cells: PyReadonlyArray1<'_, isize>,
        dtype: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let cells = crate::pybind::intp_array_to_usize_vec(&cells, "cells")?;
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

    /// Fast path for numpy ``intp`` cell-index arrays plus a gene projection.
    #[pyo3(signature = (id, cells, gene_names, missing=None, dtype=None))]
    fn access_cells_by_gene_names_array(
        &self,
        py: Python<'_>,
        id: PyDatasetId,
        cells: PyReadonlyArray1<'_, isize>,
        gene_names: Vec<String>,
        missing: Option<PyMissingGenePolicy>,
        dtype: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let cells = crate::pybind::intp_array_to_usize_vec(&cells, "cells")?;
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

    /// Fast path variant accepting an iterable of contiguous ``np.intp`` arrays.
    #[pyo3(signature = (id, batches, config=None))]
    fn prefetch_cells_arrays(
        &self,
        py: Python<'_>,
        id: PyDatasetId,
        batches: Bound<'_, PyAny>,
        config: Option<PyScheduledPrefetchConfig>,
    ) -> PyResult<PyPrefetchCells> {
        self.prefetch_cells(py, id, batches, config)
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

    /// Fast path variant accepting an iterable of contiguous ``np.intp`` arrays.
    #[pyo3(signature = (id, batches, gene_names, missing=None, config=None))]
    fn prefetch_cells_by_gene_names_arrays(
        &self,
        py: Python<'_>,
        id: PyDatasetId,
        batches: Bound<'_, PyAny>,
        gene_names: Vec<String>,
        missing: Option<PyMissingGenePolicy>,
        config: Option<PyScheduledPrefetchConfig>,
    ) -> PyResult<PyPrefetchCells> {
        self.prefetch_cells_by_gene_names(py, id, batches, gene_names, missing, config)
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
        None => Ok(bank.dataset_dtype(id)?),
    }
}

fn intp_array_to_usize_vec(
    cells: &PyReadonlyArray1<'_, isize>,
    context: &str,
) -> PyResult<Vec<usize>> {
    let slice = cells.as_slice().map_err(|_| {
        PyValueError::new_err(format!("{context} must be a contiguous 1D np.intp array"))
    })?;
    let mut out = Vec::with_capacity(slice.len());
    for (i, &cell) in slice.iter().enumerate() {
        if cell < 0 {
            return Err(PyValueError::new_err(format!(
                "{context}[{i}] must be non-negative, got {cell}"
            )));
        }
        out.push(cell as usize);
    }
    Ok(out)
}

fn extract_cells_any(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(array) = obj.extract::<PyReadonlyArray1<'_, isize>>() {
        return intp_array_to_usize_vec(&array, "cells");
    }
    obj.extract::<Vec<usize>>()
}

#[allow(clippy::too_many_arguments)]
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
            out.into_pyarray(py).into_any().unbind()
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
            bits.into_pyarray(py).into_any().unbind()
        }
        DType::BF16 => {
            let out: Vec<Bf16Bits> = bank.access_cells_owned(id, cells)?;
            let bits = bf16_bits_to_u16(out);
            bits.into_pyarray(py).into_any().unbind()
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
            out.into_pyarray(py).into_any().unbind()
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
            bits.into_pyarray(py).into_any().unbind()
        }
        DType::BF16 => {
            let out: Vec<Bf16Bits> =
                bank.access_cells_owned_by_gene_names(id, cells, gene_names, missing)?;
            let bits = bf16_bits_to_u16(out);
            bits.into_pyarray(py).into_any().unbind()
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
            let item = item?;
            out.push(extract_cells_any(&item)?);
        }
        Ok(Self {
            batches: out.into_iter(),
        })
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
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny {
                    cells: b.cells,
                    buffer: PrefetchedBufferAny::U8(b.buffer),
                    num_genes: b.num_genes,
                })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
            PrefetchDispatch::I8(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny {
                    cells: b.cells,
                    buffer: PrefetchedBufferAny::I8(b.buffer),
                    num_genes: b.num_genes,
                })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
            PrefetchDispatch::U16(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny {
                    cells: b.cells,
                    buffer: PrefetchedBufferAny::U16(b.buffer),
                    num_genes: b.num_genes,
                })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
            PrefetchDispatch::I16(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny {
                    cells: b.cells,
                    buffer: PrefetchedBufferAny::I16(b.buffer),
                    num_genes: b.num_genes,
                })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
            PrefetchDispatch::U32(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny {
                    cells: b.cells,
                    buffer: PrefetchedBufferAny::U32(b.buffer),
                    num_genes: b.num_genes,
                })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
            PrefetchDispatch::I32(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny {
                    cells: b.cells,
                    buffer: PrefetchedBufferAny::I32(b.buffer),
                    num_genes: b.num_genes,
                })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
            PrefetchDispatch::U64(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny {
                    cells: b.cells,
                    buffer: PrefetchedBufferAny::U64(b.buffer),
                    num_genes: b.num_genes,
                })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
            PrefetchDispatch::I64(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny {
                    cells: b.cells,
                    buffer: PrefetchedBufferAny::I64(b.buffer),
                    num_genes: b.num_genes,
                })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
            PrefetchDispatch::F32(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny {
                    cells: b.cells,
                    buffer: PrefetchedBufferAny::F32(b.buffer),
                    num_genes: b.num_genes,
                })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
            PrefetchDispatch::F64(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny {
                    cells: b.cells,
                    buffer: PrefetchedBufferAny::F64(b.buffer),
                    num_genes: b.num_genes,
                })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
            PrefetchDispatch::F16(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny {
                    cells: b.cells,
                    buffer: PrefetchedBufferAny::F16(b.buffer),
                    num_genes: b.num_genes,
                })),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            },
            PrefetchDispatch::BF16(it) => match it.next() {
                Some(Ok(b)) => Some(Ok(PrefetchedBatchAny {
                    cells: b.cells,
                    buffer: PrefetchedBufferAny::BF16(b.buffer),
                    num_genes: b.num_genes,
                })),
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
                let cells = batch.cells.into_pyarray(py).into_any().unbind();
                let tuple = (cells, arr, batch.num_genes).into_pyobject(py)?;
                Ok(Some(tuple.into_any().unbind()))
            }
            Some(Err(err)) => Err(err.into()),
        }
    }
}

fn buffer_to_numpy(py: Python<'_>, buffer: PrefetchedBufferAny) -> PyResult<PyObject> {
    let arr = match buffer {
        PrefetchedBufferAny::U8(v) => v.into_pyarray(py).into_any().unbind(),
        PrefetchedBufferAny::I8(v) => v.into_pyarray(py).into_any().unbind(),
        PrefetchedBufferAny::U16(v) => v.into_pyarray(py).into_any().unbind(),
        PrefetchedBufferAny::I16(v) => v.into_pyarray(py).into_any().unbind(),
        PrefetchedBufferAny::U32(v) => v.into_pyarray(py).into_any().unbind(),
        PrefetchedBufferAny::I32(v) => v.into_pyarray(py).into_any().unbind(),
        PrefetchedBufferAny::U64(v) => v.into_pyarray(py).into_any().unbind(),
        PrefetchedBufferAny::I64(v) => v.into_pyarray(py).into_any().unbind(),
        PrefetchedBufferAny::F32(v) => v.into_pyarray(py).into_any().unbind(),
        PrefetchedBufferAny::F64(v) => v.into_pyarray(py).into_any().unbind(),
        PrefetchedBufferAny::F16(v) => {
            let bits = f16_bits_to_u16(v);
            bits.into_pyarray(py).into_any().unbind()
        }
        PrefetchedBufferAny::BF16(v) => {
            let bits = bf16_bits_to_u16(v);
            bits.into_pyarray(py).into_any().unbind()
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
        DType::U8 => {
            PrefetchDispatch::U8(bank.prefetch_cells_scheduled(id, batch_source, config)?)
        }
        DType::I8 => {
            PrefetchDispatch::I8(bank.prefetch_cells_scheduled(id, batch_source, config)?)
        }
        DType::U16 => {
            PrefetchDispatch::U16(bank.prefetch_cells_scheduled(id, batch_source, config)?)
        }
        DType::I16 => {
            PrefetchDispatch::I16(bank.prefetch_cells_scheduled(id, batch_source, config)?)
        }
        DType::U32 => {
            PrefetchDispatch::U32(bank.prefetch_cells_scheduled(id, batch_source, config)?)
        }
        DType::I32 => {
            PrefetchDispatch::I32(bank.prefetch_cells_scheduled(id, batch_source, config)?)
        }
        DType::U64 => {
            PrefetchDispatch::U64(bank.prefetch_cells_scheduled(id, batch_source, config)?)
        }
        DType::I64 => {
            PrefetchDispatch::I64(bank.prefetch_cells_scheduled(id, batch_source, config)?)
        }
        DType::F32 => {
            PrefetchDispatch::F32(bank.prefetch_cells_scheduled(id, batch_source, config)?)
        }
        DType::F64 => {
            PrefetchDispatch::F64(bank.prefetch_cells_scheduled(id, batch_source, config)?)
        }
        DType::F16 => {
            PrefetchDispatch::F16(bank.prefetch_cells_scheduled(id, batch_source, config)?)
        }
        DType::BF16 => {
            PrefetchDispatch::BF16(bank.prefetch_cells_scheduled(id, batch_source, config)?)
        }
    };
    Ok(PyPrefetchCells::new(dispatch))
}

#[allow(clippy::too_many_arguments)]
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
        DType::U8 => PrefetchDispatch::U8(bank.prefetch_cells_scheduled_by_gene_names(
            id,
            batch_source,
            gene_names,
            missing,
            config,
        )?),
        DType::I8 => PrefetchDispatch::I8(bank.prefetch_cells_scheduled_by_gene_names(
            id,
            batch_source,
            gene_names,
            missing,
            config,
        )?),
        DType::U16 => PrefetchDispatch::U16(bank.prefetch_cells_scheduled_by_gene_names(
            id,
            batch_source,
            gene_names,
            missing,
            config,
        )?),
        DType::I16 => PrefetchDispatch::I16(bank.prefetch_cells_scheduled_by_gene_names(
            id,
            batch_source,
            gene_names,
            missing,
            config,
        )?),
        DType::U32 => PrefetchDispatch::U32(bank.prefetch_cells_scheduled_by_gene_names(
            id,
            batch_source,
            gene_names,
            missing,
            config,
        )?),
        DType::I32 => PrefetchDispatch::I32(bank.prefetch_cells_scheduled_by_gene_names(
            id,
            batch_source,
            gene_names,
            missing,
            config,
        )?),
        DType::U64 => PrefetchDispatch::U64(bank.prefetch_cells_scheduled_by_gene_names(
            id,
            batch_source,
            gene_names,
            missing,
            config,
        )?),
        DType::I64 => PrefetchDispatch::I64(bank.prefetch_cells_scheduled_by_gene_names(
            id,
            batch_source,
            gene_names,
            missing,
            config,
        )?),
        DType::F32 => PrefetchDispatch::F32(bank.prefetch_cells_scheduled_by_gene_names(
            id,
            batch_source,
            gene_names,
            missing,
            config,
        )?),
        DType::F64 => PrefetchDispatch::F64(bank.prefetch_cells_scheduled_by_gene_names(
            id,
            batch_source,
            gene_names,
            missing,
            config,
        )?),
        DType::F16 => PrefetchDispatch::F16(bank.prefetch_cells_scheduled_by_gene_names(
            id,
            batch_source,
            gene_names,
            missing,
            config,
        )?),
        DType::BF16 => PrefetchDispatch::BF16(bank.prefetch_cells_scheduled_by_gene_names(
            id,
            batch_source,
            gene_names,
            missing,
            config,
        )?),
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
    let dtype = extract_dtype(&data.getattr("dtype")?)?;
    let codec = build_codec(py, &data.getattr("codec")?)?;

    let store_kind: String = data.getattr("store_kind")?.extract()?;
    // `variable_chunks` is optional on the Python side (defaults to False for
    // regular grids); rectilinear cell-aligned CSR arrays set it True.
    let variable_chunks: bool = match data.getattr("variable_chunks") {
        Ok(v) => v.extract().unwrap_or(false),
        Err(_) => false,
    };
    let chunk_boundaries: Option<Vec<Vec<usize>>> = match data.getattr("chunk_boundaries") {
        Ok(value) => {
            let axes: Vec<Vec<usize>> = value.extract()?;
            if axes.is_empty() {
                None
            } else {
                Some(axes)
            }
        }
        Err(_) => None,
    };
    let chunk_grid_shape: Vec<usize> = if let Some(axes) = &chunk_boundaries {
        let mut grid = Vec::with_capacity(axes.len());
        for (axis, boundaries) in axes.iter().enumerate() {
            if boundaries.len() < 2 {
                return Err(PyValueError::new_err(format!(
                    "chunk_boundaries[{axis}] must contain at least one interval"
                )));
            }
            let chunks = boundaries.len() - 1;
            grid.push(chunks);
        }
        grid
    } else {
        shape
            .iter()
            .zip(chunk_shape.iter())
            .map(|(&s, &c)| div_ceil(s, c))
            .collect()
    };
    let chunks = match store_kind.as_str() {
        "file" => {
            let payload_path: String = data.getattr("payload_path")?.extract()?;
            let payload_file_path = optional_string_attr(data, "payload_file_path")?;
            let path = if payload_file_path.is_empty() {
                PathBuf::from(store_path).join(payload_path)
            } else {
                PathBuf::from(payload_file_path)
            };
            let locations = match extract_locations_from_offset_arrays(data)? {
                Some(locations) => locations,
                None => extract_locations(&data.getattr("chunks")?)?,
            };
            ChunkStoreMeta::FileOffset { path, locations }
        }
        "dir" => {
            // Standard zarr tree: one file per chunk at offset 0. ZIP_STORED
            // stores: every chunk opens the archive path and uses its physical
            // in-archive offset. Python exposes both cases through optional
            // `chunk_file_paths`.
            let chunk_paths_obj = data.getattr("chunk_paths")?;
            let chunk_paths: Vec<String> = chunk_paths_obj.extract()?;
            let n = chunk_paths.len();
            let chunk_file_paths = match data.getattr("chunk_file_paths") {
                Ok(paths) => {
                    let values: Vec<String> = paths.extract()?;
                    if values.is_empty() {
                        None
                    } else {
                        Some(values)
                    }
                }
                Err(_) => None,
            };
            let chunk_file_count = chunk_file_paths.as_ref().map_or(0, Vec::len);
            if chunk_file_count != 0 && chunk_file_count != n {
                return Err(PyValueError::new_err(format!(
                    "chunk_file_paths count {chunk_file_count} != chunk_paths count {n}"
                )));
            }
            let chunk_offsets =
                extract_optional_u64_vec_attr(data, "chunk_offsets")?.unwrap_or_else(|| vec![0; n]);
            let chunk_lengths = extract_u64_vec(&data.getattr("chunk_lengths")?, "chunk_lengths")?;
            if chunk_offsets.len() != n {
                return Err(PyValueError::new_err(format!(
                    "chunk_offsets count {} != chunk_paths count {n}",
                    chunk_offsets.len()
                )));
            }
            if chunk_lengths.len() != n {
                return Err(PyValueError::new_err(format!(
                    "chunk_lengths count {} != chunk_paths count {n}",
                    chunk_lengths.len()
                )));
            }
            let mut locations = Vec::with_capacity(n);
            let store_root = PathBuf::from(store_path);
            for (i, rel) in chunk_paths.into_iter().enumerate() {
                let path = if let Some(paths) = &chunk_file_paths {
                    PathBuf::from(&paths[i])
                } else {
                    store_root.join(rel)
                };
                let offset = chunk_offsets[i];
                let len = u64_to_usize(chunk_lengths[i], "chunk_lengths")?;
                locations.push(DirectoryChunkLocationMeta { path, offset, len });
            }
            ChunkStoreMeta::Directory { locations }
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown store_kind {other:?} (expected 'file' or 'dir')"
            )))
        }
    };

    Ok(ArrayMeta {
        shape,
        chunk_shape,
        chunk_grid_shape,
        dtype,
        order: ArrayOrder::C,
        codec,
        chunks,
        variable_chunks,
        chunk_boundaries,
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

fn extract_u64_vec(obj: &Bound<'_, PyAny>, context: &str) -> PyResult<Vec<u64>> {
    if let Ok(array) = obj.extract::<PyReadonlyArray1<'_, u64>>() {
        let slice = array.as_slice().map_err(|_| {
            PyValueError::new_err(format!("{context} must be a contiguous 1D uint64 array"))
        })?;
        return Ok(slice.to_vec());
    }
    obj.extract::<Vec<u64>>()
        .map_err(|err| PyValueError::new_err(format!("{context}: {err}")))
}

fn extract_optional_u64_vec_attr(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<Vec<u64>>> {
    match obj.getattr(name) {
        Ok(value) => Ok(Some(extract_u64_vec(&value, name)?)),
        Err(_) => Ok(None),
    }
}

fn u64_to_usize(value: u64, context: &str) -> PyResult<usize> {
    usize::try_from(value).map_err(|_| {
        PyValueError::new_err(format!("{context} value {value} does not fit in usize"))
    })
}

fn extract_locations_from_offset_arrays(
    data: &Bound<'_, PyAny>,
) -> PyResult<Option<Vec<FileChunkLocation>>> {
    let Some(offsets) = extract_optional_u64_vec_attr(data, "chunk_offsets")? else {
        return Ok(None);
    };
    let lengths = extract_u64_vec(&data.getattr("chunk_lengths")?, "chunk_lengths")?;
    if offsets.len() != lengths.len() {
        return Err(PyValueError::new_err(format!(
            "chunk_offsets length {} != chunk_lengths length {}",
            offsets.len(),
            lengths.len()
        )));
    }
    let mut out = Vec::with_capacity(offsets.len());
    for (offset, len) in offsets.into_iter().zip(lengths) {
        out.push(FileChunkLocation {
            offset,
            len: u64_to_usize(len, "chunk_lengths")?,
        });
    }
    Ok(Some(out))
}

fn build_shared_codec(py: Python<'_>, codec: &Bound<'_, PyAny>) -> PyResult<Option<SharedCodec>> {
    match build_codec(py, codec)? {
        ArrayCodecMeta::Uncompressed => Ok(None),
        ArrayCodecMeta::ZarrV2Json {
            filters,
            compressor,
        } => codec_pipeline_from_zarr_v2_json_str(filters.as_deref(), compressor.as_deref())
            .map(Some)
            .map_err(|err| PyValueError::new_err(err.to_string())),
        other => Err(PyValueError::new_err(format!(
            "unsupported index codec metadata: {other:?}"
        ))),
    }
}

fn decode_index_chunk_into(
    raw: &[u8],
    dtype: DType,
    codec: Option<&SharedCodec>,
    out: &mut Vec<u64>,
) -> PyResult<()> {
    let decoded;
    let bytes = if let Some(codec) = codec {
        decoded = codec
            .decode(raw, None)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        decoded.as_slice()
    } else {
        raw
    };
    decode_index_bytes_into(bytes, dtype, out)
}

fn decode_index_bytes_into(bytes: &[u8], dtype: DType, out: &mut Vec<u64>) -> PyResult<()> {
    let item_size = match dtype {
        DType::U8 | DType::I8 => 1,
        DType::U16 | DType::I16 => 2,
        DType::U32 | DType::I32 => 4,
        DType::U64 | DType::I64 => 8,
        other => {
            return Err(PyValueError::new_err(format!(
                "index array dtype must be integer, got {other:?}"
            )))
        }
    };
    if bytes.len() % item_size != 0 {
        return Err(PyValueError::new_err(format!(
            "decoded index chunk has {} bytes, not divisible by dtype item size {item_size}",
            bytes.len()
        )));
    }
    out.reserve(bytes.len() / item_size);
    match dtype {
        DType::U8 => out.extend(bytes.iter().map(|&value| u64::from(value))),
        DType::I8 => {
            for &byte in bytes {
                let value = i8::from_le_bytes([byte]);
                push_signed_index(i64::from(value), out)?;
            }
        }
        DType::U16 => {
            for chunk in bytes.chunks_exact(2) {
                out.push(u64::from(u16::from_le_bytes([chunk[0], chunk[1]])));
            }
        }
        DType::I16 => {
            for chunk in bytes.chunks_exact(2) {
                let value = i16::from_le_bytes([chunk[0], chunk[1]]);
                push_signed_index(i64::from(value), out)?;
            }
        }
        DType::U32 => {
            for chunk in bytes.chunks_exact(4) {
                out.push(u64::from(u32::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                ])));
            }
        }
        DType::I32 => {
            for chunk in bytes.chunks_exact(4) {
                let value = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                push_signed_index(i64::from(value), out)?;
            }
        }
        DType::U64 => {
            for chunk in bytes.chunks_exact(8) {
                out.push(u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]));
            }
        }
        DType::I64 => {
            for chunk in bytes.chunks_exact(8) {
                let value = i64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                push_signed_index(value, out)?;
            }
        }
        _ => unreachable!("non-integer dtype rejected above"),
    }
    Ok(())
}

fn push_signed_index(value: i64, out: &mut Vec<u64>) -> PyResult<()> {
    let value = u64::try_from(value)
        .map_err(|_| PyValueError::new_err(format!("negative index value {value}")))?;
    out.push(value);
    Ok(())
}

fn finalize_index_output(
    py: Python<'_>,
    mut out: Vec<u64>,
    count: usize,
    allow_short: bool,
) -> PyResult<PyObject> {
    if allow_short {
        if out.len() > count {
            out.truncate(count);
        } else if out.len() < count {
            out.resize(count, 0);
        }
    }
    validate_index_output_len(out.len(), count)?;
    Ok(out.into_pyarray(py).into_any().unbind())
}

fn validate_index_output_len(actual: usize, expected: usize) -> PyResult<()> {
    if actual != expected {
        return Err(PyValueError::new_err(format!(
            "decoded index length {actual} != expected {expected}"
        )));
    }
    Ok(())
}

fn optional_string_attr(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<String> {
    match obj.getattr(name) {
        Ok(value) => value.extract(),
        Err(_) => Ok(String::new()),
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
