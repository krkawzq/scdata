use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::access::{AccessConfig, AccessCpuConfig, ScheduledAccessConfig};
use crate::codecs::DecodePoolConfig;
use crate::databank::{
    DataBankConfig, FillConfig, NativeAccessConfig, NativeBloscConfig, NativeLoadCoalesceConfig,
    NativeLoadConfig, NativeMode, ProjectedSparseDataGroupStrategy, ScheduledPrefetchConfig,
};
use crate::iopool::{BaseIoConfig, IoConfig, ThreadedConfig, UringConfig};

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataBankConfig>()?;
    m.add_class::<PyIoConfig>()?;
    m.add_class::<PyUringConfig>()?;
    m.add_class::<PyThreadedConfig>()?;
    m.add_class::<PyBaseIoConfig>()?;
    m.add_class::<PyDecodePoolConfig>()?;
    m.add_class::<PyAccessConfig>()?;
    m.add_class::<PyAccessCpuConfig>()?;
    m.add_class::<PyFillConfig>()?;
    m.add_class::<PyFastAccessConfig>()?;
    m.add_class::<PyFastLoadConfig>()?;
    m.add_class::<PyFastLoadCoalesceConfig>()?;
    m.add_class::<PyFastBloscConfig>()?;
    m.add_class::<PyScheduledAccessConfig>()?;
    m.add_class::<PyScheduledPrefetchConfig>()?;
    Ok(())
}

fn validate_result<E: std::fmt::Display>(result: Result<(), E>) -> PyResult<()> {
    result.map_err(|err| PyValueError::new_err(err.to_string()))
}

#[pyclass(name = "_BaseIoConfig", module = "scdata._scdata")]
#[derive(Clone)]
pub(crate) struct PyBaseIoConfig {
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
    fn max_in_flight(&self) -> usize {
        self.inner.max_in_flight
    }

    #[setter]
    fn set_max_in_flight(&mut self, value: usize) {
        self.inner.max_in_flight = value;
    }

    #[getter]
    fn queue_capacity(&self) -> usize {
        self.inner.queue_capacity
    }

    #[setter]
    fn set_queue_capacity(&mut self, value: usize) {
        self.inner.queue_capacity = value;
    }

    #[getter]
    fn priority_levels(&self) -> usize {
        self.inner.priority_levels
    }

    #[setter]
    fn set_priority_levels(&mut self, value: usize) {
        self.inner.priority_levels = value;
    }

    #[getter]
    fn queue_shards(&self) -> usize {
        self.inner.queue_shards
    }

    #[setter]
    fn set_queue_shards(&mut self, value: usize) {
        self.inner.queue_shards = value;
    }

    #[getter]
    fn assume_non_overlapping_reads(&self) -> bool {
        self.inner.assume_non_overlapping_reads
    }

    #[setter]
    fn set_assume_non_overlapping_reads(&mut self, value: bool) {
        self.inner.assume_non_overlapping_reads = value;
    }

    fn validate(&self) -> PyResult<()> {
        validate_result(self.inner.validate())
    }
}

#[pyclass(name = "_UringConfig", module = "scdata._scdata")]
#[derive(Clone)]
pub(crate) struct PyUringConfig {
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
    fn base(&self) -> PyBaseIoConfig {
        PyBaseIoConfig {
            inner: self.inner.base.clone(),
        }
    }

    #[setter]
    fn set_base(&mut self, value: PyBaseIoConfig) {
        self.inner.base = value.inner;
    }

    #[getter]
    fn entries(&self) -> u32 {
        self.inner.entries
    }

    #[setter]
    fn set_entries(&mut self, value: u32) {
        self.inner.entries = value;
    }

    #[getter]
    fn drivers(&self) -> usize {
        self.inner.drivers
    }

    #[setter]
    fn set_drivers(&mut self, value: usize) {
        self.inner.drivers = value;
    }

    #[getter]
    fn iowq_bounded_workers(&self) -> u32 {
        self.inner.iowq_bounded_workers
    }

    #[setter]
    fn set_iowq_bounded_workers(&mut self, value: u32) {
        self.inner.iowq_bounded_workers = value;
    }

    #[getter]
    fn iowq_unbounded_workers(&self) -> u32 {
        self.inner.iowq_unbounded_workers
    }

    #[setter]
    fn set_iowq_unbounded_workers(&mut self, value: u32) {
        self.inner.iowq_unbounded_workers = value;
    }

    #[getter]
    fn registered_files(&self) -> u32 {
        self.inner.registered_files
    }

    #[setter]
    fn set_registered_files(&mut self, value: u32) {
        self.inner.registered_files = value;
    }

    fn validate(&self) -> PyResult<()> {
        validate_result(self.inner.validate())
    }
}

#[pyclass(name = "_ThreadedConfig", module = "scdata._scdata")]
#[derive(Clone)]
pub(crate) struct PyThreadedConfig {
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
    fn base(&self) -> PyBaseIoConfig {
        PyBaseIoConfig {
            inner: self.inner.base.clone(),
        }
    }

    #[setter]
    fn set_base(&mut self, value: PyBaseIoConfig) {
        self.inner.base = value.inner;
    }

    #[getter]
    fn num_workers(&self) -> usize {
        self.inner.num_workers
    }

    #[setter]
    fn set_num_workers(&mut self, value: usize) {
        self.inner.num_workers = value;
    }

    #[getter]
    fn cpus(&self) -> Option<Vec<usize>> {
        self.inner.cpus.clone()
    }

    #[setter]
    fn set_cpus(&mut self, value: Option<Vec<usize>>) {
        self.inner.cpus = value;
    }

    fn validate(&self) -> PyResult<()> {
        validate_result(self.inner.validate())
    }
}

#[pyclass(name = "_IoConfig", module = "scdata._scdata")]
#[derive(Clone)]
pub(crate) struct PyIoConfig {
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
            inner: IoConfig::Uring(config.map(|config| config.inner).unwrap_or_default()),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (config=None))]
    fn threaded(config: Option<PyThreadedConfig>) -> Self {
        Self {
            inner: IoConfig::Threaded(config.map(|config| config.inner).unwrap_or_default()),
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            IoConfig::Uring(_) => "uring",
            IoConfig::Threaded(_) => "threaded",
        }
    }

    #[getter]
    fn base(&self) -> PyBaseIoConfig {
        PyBaseIoConfig {
            inner: self.inner.base().clone(),
        }
    }

    fn uring_config(&self) -> Option<PyUringConfig> {
        match &self.inner {
            IoConfig::Uring(config) => Some(PyUringConfig {
                inner: config.clone(),
            }),
            IoConfig::Threaded(_) => None,
        }
    }

    fn threaded_config(&self) -> Option<PyThreadedConfig> {
        match &self.inner {
            IoConfig::Threaded(config) => Some(PyThreadedConfig {
                inner: config.clone(),
            }),
            IoConfig::Uring(_) => None,
        }
    }

    fn validate(&self) -> PyResult<()> {
        validate_result(self.inner.validate())
    }
}

#[pyclass(name = "_DecodePoolConfig", module = "scdata._scdata")]
#[derive(Clone)]
pub(crate) struct PyDecodePoolConfig {
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
    fn num_workers(&self) -> usize {
        self.inner.num_workers
    }

    #[setter]
    fn set_num_workers(&mut self, value: usize) {
        self.inner.num_workers = value;
    }

    #[getter]
    fn queue_capacity(&self) -> usize {
        self.inner.queue_capacity
    }

    #[setter]
    fn set_queue_capacity(&mut self, value: usize) {
        self.inner.queue_capacity = value;
    }

    #[getter]
    fn cpus(&self) -> Option<Vec<usize>> {
        self.inner.cpus.clone()
    }

    #[setter]
    fn set_cpus(&mut self, value: Option<Vec<usize>>) {
        self.inner.cpus = value;
    }

    fn validate(&self) -> PyResult<()> {
        validate_result(self.inner.validate())
    }
}

#[pyclass(name = "_AccessCpuConfig", module = "scdata._scdata")]
#[derive(Clone)]
pub(crate) struct PyAccessCpuConfig {
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
    fn num_workers(&self) -> usize {
        self.inner.num_workers
    }

    #[setter]
    fn set_num_workers(&mut self, value: usize) {
        self.inner.num_workers = value;
    }

    #[getter]
    fn queue_capacity(&self) -> usize {
        self.inner.queue_capacity
    }

    #[setter]
    fn set_queue_capacity(&mut self, value: usize) {
        self.inner.queue_capacity = value;
    }

    #[getter]
    fn cpus(&self) -> Option<Vec<usize>> {
        self.inner.cpus.clone()
    }

    #[setter]
    fn set_cpus(&mut self, value: Option<Vec<usize>>) {
        self.inner.cpus = value;
    }

    fn validate(&self) -> PyResult<()> {
        validate_result(self.inner.validate())
    }
}

#[pyclass(name = "_AccessConfig", module = "scdata._scdata")]
#[derive(Clone)]
pub(crate) struct PyAccessConfig {
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
    fn queue_capacity(&self) -> usize {
        self.inner.queue_capacity
    }

    #[setter]
    fn set_queue_capacity(&mut self, value: usize) {
        self.inner.queue_capacity = value;
    }

    #[getter]
    fn scheduler_shards(&self) -> usize {
        self.inner.scheduler_shards
    }

    #[setter]
    fn set_scheduler_shards(&mut self, value: usize) {
        self.inner.scheduler_shards = value;
    }

    #[getter]
    fn cache_capacity_bytes(&self) -> usize {
        self.inner.cache_capacity_bytes
    }

    #[setter]
    fn set_cache_capacity_bytes(&mut self, value: usize) {
        self.inner.cache_capacity_bytes = value;
    }

    #[getter]
    fn memory_budget_bytes(&self) -> usize {
        self.inner.memory_budget_bytes
    }

    #[setter]
    fn set_memory_budget_bytes(&mut self, value: usize) {
        self.inner.memory_budget_bytes = value;
    }

    #[getter]
    fn default_io_priority(&self) -> u8 {
        self.inner.default_io_priority
    }

    #[setter]
    fn set_default_io_priority(&mut self, value: u8) {
        self.inner.default_io_priority = value;
    }

    #[getter]
    fn keep_decoded(&self) -> bool {
        self.inner.keep_decoded
    }

    #[setter]
    fn set_keep_decoded(&mut self, value: bool) {
        self.inner.keep_decoded = value;
    }

    #[getter]
    fn cpu(&self) -> PyAccessCpuConfig {
        PyAccessCpuConfig {
            inner: self.inner.cpu.clone(),
        }
    }

    #[setter]
    fn set_cpu(&mut self, value: PyAccessCpuConfig) {
        self.inner.cpu = value.inner;
    }

    fn validate(&self) -> PyResult<()> {
        validate_result(self.inner.validate())
    }
}

#[pyclass(name = "_FillConfig", module = "scdata._scdata")]
#[derive(Clone)]
pub(crate) struct PyFillConfig {
    inner: FillConfig,
}

#[pymethods]
impl PyFillConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: FillConfig::default(),
        }
    }

    #[getter]
    fn parallel(&self) -> bool {
        self.inner.parallel
    }

    #[setter]
    fn set_parallel(&mut self, value: bool) {
        self.inner.parallel = value;
    }

    #[getter]
    fn num_workers(&self) -> usize {
        self.inner.num_workers
    }

    #[setter]
    fn set_num_workers(&mut self, value: usize) {
        self.inner.num_workers = value;
    }

    #[getter]
    fn queue_capacity(&self) -> usize {
        self.inner.queue_capacity
    }

    #[setter]
    fn set_queue_capacity(&mut self, value: usize) {
        self.inner.queue_capacity = value;
    }

    #[getter]
    fn min_parallel_rows(&self) -> usize {
        self.inner.min_parallel_rows
    }

    #[setter]
    fn set_min_parallel_rows(&mut self, value: usize) {
        self.inner.min_parallel_rows = value;
    }

    #[getter]
    fn min_parallel_bytes(&self) -> usize {
        self.inner.min_parallel_bytes
    }

    #[setter]
    fn set_min_parallel_bytes(&mut self, value: usize) {
        self.inner.min_parallel_bytes = value;
    }

    #[getter]
    fn cpus(&self) -> Option<Vec<usize>> {
        self.inner.cpus.clone()
    }

    #[setter]
    fn set_cpus(&mut self, value: Option<Vec<usize>>) {
        self.inner.cpus = value;
    }

    fn validate(&self) -> PyResult<()> {
        validate_result(self.inner.validate())
    }
}

#[pyclass(name = "_FastAccessConfig", module = "scdata._scdata")]
#[derive(Clone)]
pub(crate) struct PyFastAccessConfig {
    inner: NativeAccessConfig,
}

#[pymethods]
impl PyFastAccessConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: NativeAccessConfig::default(),
        }
    }

    #[getter]
    fn enabled(&self) -> bool {
        self.inner.enabled
    }

    #[setter]
    fn set_enabled(&mut self, value: bool) {
        self.inner.enabled = value;
    }

    #[getter]
    fn fused_workers(&self) -> usize {
        self.inner.fused_workers
    }

    #[setter]
    fn set_fused_workers(&mut self, value: usize) {
        self.inner.fused_workers = value;
    }

    #[getter]
    fn request_prefetch_batches(&self) -> usize {
        self.inner.request_prefetch_batches
    }

    #[setter]
    fn set_request_prefetch_batches(&mut self, value: usize) {
        self.inner.request_prefetch_batches = value;
    }

    #[getter]
    fn request_prefetch_blocks(&self) -> usize {
        self.inner.request_prefetch_blocks
    }

    #[setter]
    fn set_request_prefetch_blocks(&mut self, value: usize) {
        self.inner.request_prefetch_blocks = value;
    }

    #[getter]
    fn memory_budget_bytes(&self) -> usize {
        self.inner.memory_budget_bytes
    }

    #[setter]
    fn set_memory_budget_bytes(&mut self, value: usize) {
        self.inner.memory_budget_bytes = value;
    }

    #[getter]
    fn response_queue_bytes_soft_limit(&self) -> usize {
        self.inner.response_queue_bytes_soft_limit
    }

    #[setter]
    fn set_response_queue_bytes_soft_limit(&mut self, value: usize) {
        self.inner.response_queue_bytes_soft_limit = value;
    }

    #[getter]
    fn response_queue_bytes_hard_limit(&self) -> usize {
        self.inner.response_queue_bytes_hard_limit
    }

    #[setter]
    fn set_response_queue_bytes_hard_limit(&mut self, value: usize) {
        self.inner.response_queue_bytes_hard_limit = value;
    }

    #[getter]
    fn load(&self) -> PyFastLoadConfig {
        PyFastLoadConfig {
            inner: self.inner.load.clone(),
        }
    }

    #[setter]
    fn set_load(&mut self, value: PyFastLoadConfig) {
        self.inner.load = value.inner;
    }

    #[getter]
    fn blosc(&self) -> PyFastBloscConfig {
        PyFastBloscConfig {
            inner: self.inner.blosc.clone(),
        }
    }

    #[setter]
    fn set_blosc(&mut self, value: PyFastBloscConfig) {
        self.inner.blosc = value.inner;
    }

    fn validate(&self) -> PyResult<()> {
        validate_result(self.inner.validate())
    }
}

#[pyclass(name = "_FastLoadConfig", module = "scdata._scdata")]
#[derive(Clone)]
pub(crate) struct PyFastLoadConfig {
    inner: NativeLoadConfig,
}

#[pymethods]
impl PyFastLoadConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: NativeLoadConfig::default(),
        }
    }

    #[getter]
    fn scheduler_workers(&self) -> usize {
        self.inner.scheduler_workers
    }

    #[setter]
    fn set_scheduler_workers(&mut self, value: usize) {
        self.inner.scheduler_workers = value;
    }

    #[getter]
    fn io_workers(&self) -> usize {
        self.inner.io_workers
    }

    #[setter]
    fn set_io_workers(&mut self, value: usize) {
        self.inner.io_workers = value;
    }

    #[getter]
    fn coalesce(&self) -> PyFastLoadCoalesceConfig {
        PyFastLoadCoalesceConfig {
            inner: self.inner.coalesce.clone(),
        }
    }

    #[setter]
    fn set_coalesce(&mut self, value: PyFastLoadCoalesceConfig) {
        self.inner.coalesce = value.inner;
    }

    fn validate(&self) -> PyResult<()> {
        validate_result(self.inner.validate())
    }
}

#[pyclass(name = "_FastLoadCoalesceConfig", module = "scdata._scdata")]
#[derive(Clone)]
pub(crate) struct PyFastLoadCoalesceConfig {
    inner: NativeLoadCoalesceConfig,
}

#[pymethods]
impl PyFastLoadCoalesceConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: NativeLoadCoalesceConfig::default(),
        }
    }

    #[getter]
    fn max_window_us(&self) -> u32 {
        self.inner.max_window_us
    }

    #[setter]
    fn set_max_window_us(&mut self, value: u32) {
        self.inner.max_window_us = value;
    }

    #[getter]
    fn max_merged_len(&self) -> usize {
        self.inner.max_merged_len
    }

    #[setter]
    fn set_max_merged_len(&mut self, value: usize) {
        self.inner.max_merged_len = value;
    }

    #[getter]
    fn max_gap_bytes(&self) -> usize {
        self.inner.max_gap_bytes
    }

    #[setter]
    fn set_max_gap_bytes(&mut self, value: usize) {
        self.inner.max_gap_bytes = value;
    }

    #[getter]
    fn max_waste_ratio(&self) -> f32 {
        self.inner.max_waste_ratio
    }

    #[setter]
    fn set_max_waste_ratio(&mut self, value: f32) {
        self.inner.max_waste_ratio = value;
    }

    #[getter]
    fn min_children(&self) -> usize {
        self.inner.min_children
    }

    #[setter]
    fn set_min_children(&mut self, value: usize) {
        self.inner.min_children = value;
    }

    fn validate(&self) -> PyResult<()> {
        validate_result(self.inner.validate())
    }
}

#[pyclass(name = "_FastBloscConfig", module = "scdata._scdata")]
#[derive(Clone)]
pub(crate) struct PyFastBloscConfig {
    inner: NativeBloscConfig,
}

#[pymethods]
impl PyFastBloscConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: NativeBloscConfig::default(),
        }
    }

    #[getter]
    fn preload_block_tables(&self) -> bool {
        self.inner.preload_block_tables
    }

    #[setter]
    fn set_preload_block_tables(&mut self, value: bool) {
        self.inner.preload_block_tables = value;
    }

    #[getter]
    fn full_unshuffle_threshold(&self) -> f32 {
        self.inner.full_unshuffle_threshold
    }

    #[setter]
    fn set_full_unshuffle_threshold(&mut self, value: f32) {
        self.inner.full_unshuffle_threshold = value;
    }

    #[getter]
    fn max_block_size(&self) -> usize {
        self.inner.max_block_size
    }

    #[setter]
    fn set_max_block_size(&mut self, value: usize) {
        self.inner.max_block_size = value;
    }

    fn validate(&self) -> PyResult<()> {
        validate_result(self.inner.validate())
    }
}

#[pyclass(name = "_DataBankConfig", module = "scdata._scdata")]
#[derive(Clone)]
pub(crate) struct PyDataBankConfig {
    pub(crate) inner: DataBankConfig,
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
    fn io_config(&self) -> PyIoConfig {
        PyIoConfig {
            inner: self.inner.io_config.clone(),
        }
    }

    #[setter]
    fn set_io_config(&mut self, value: PyIoConfig) {
        self.inner.io_config = value.inner;
    }

    #[getter]
    fn decode_config(&self) -> PyDecodePoolConfig {
        PyDecodePoolConfig {
            inner: self.inner.decode_config.clone(),
        }
    }

    #[setter]
    fn set_decode_config(&mut self, value: PyDecodePoolConfig) {
        self.inner.decode_config = value.inner;
    }

    #[getter]
    fn access_config(&self) -> PyAccessConfig {
        PyAccessConfig {
            inner: self.inner.access_config.clone(),
        }
    }

    #[setter]
    fn set_access_config(&mut self, value: PyAccessConfig) {
        self.inner.access_config = value.inner;
    }

    #[getter]
    fn fill_config(&self) -> PyFillConfig {
        PyFillConfig {
            inner: self.inner.fill_config.clone(),
        }
    }

    #[setter]
    fn set_fill_config(&mut self, value: PyFillConfig) {
        self.inner.fill_config = value.inner;
    }

    #[getter]
    fn fast_config(&self) -> PyFastAccessConfig {
        PyFastAccessConfig {
            inner: self.inner.native_config.clone(),
        }
    }

    #[setter]
    fn set_fast_config(&mut self, value: PyFastAccessConfig) {
        self.inner.native_config = value.inner;
    }

    fn validate(&self) -> PyResult<()> {
        validate_result(self.inner.validate())
    }
}

#[pyclass(name = "_ScheduledAccessConfig", module = "scdata._scdata")]
#[derive(Clone, Copy)]
pub(crate) struct PyScheduledAccessConfig {
    pub(crate) inner: ScheduledAccessConfig,
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
    fn prefetch_step(&self) -> usize {
        self.inner.prefetch_step
    }

    #[setter]
    fn set_prefetch_step(&mut self, value: usize) {
        self.inner.prefetch_step = value;
    }

    #[getter]
    fn decode_ahead_steps(&self) -> usize {
        self.inner.decode_ahead_steps
    }

    #[setter]
    fn set_decode_ahead_steps(&mut self, value: usize) {
        self.inner.decode_ahead_steps = value;
    }

    #[getter]
    fn ready_ahead_steps(&self) -> usize {
        self.inner.ready_ahead_steps
    }

    #[setter]
    fn set_ready_ahead_steps(&mut self, value: usize) {
        self.inner.ready_ahead_steps = value;
    }

    fn validate(&self) -> PyResult<()> {
        Ok(())
    }
}

#[pyclass(name = "_ScheduledPrefetchConfig", module = "scdata._scdata")]
#[derive(Clone, Copy)]
pub(crate) struct PyScheduledPrefetchConfig {
    pub(crate) inner: ScheduledPrefetchConfig,
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
    fn prefetch_step(&self) -> usize {
        self.inner.prefetch_step
    }

    #[setter]
    fn set_prefetch_step(&mut self, value: usize) {
        self.inner.prefetch_step = value;
    }

    #[getter]
    fn access(&self) -> PyScheduledAccessConfig {
        PyScheduledAccessConfig {
            inner: self.inner.access,
        }
    }

    #[setter]
    fn set_access(&mut self, value: PyScheduledAccessConfig) {
        self.inner.access = value.inner;
    }

    #[getter]
    fn projected_sparse_data_strategy(&self) -> &'static str {
        self.inner.projected_sparse_data_strategy.as_str()
    }

    #[setter]
    fn set_projected_sparse_data_strategy(&mut self, value: &str) -> PyResult<()> {
        self.inner.projected_sparse_data_strategy =
            ProjectedSparseDataGroupStrategy::parse(value).map_err(PyValueError::new_err)?;
        Ok(())
    }

    #[getter]
    fn fast_mode(&self) -> &'static str {
        self.inner.native_mode.as_str()
    }

    #[setter]
    fn set_fast_mode(&mut self, value: &str) -> PyResult<()> {
        self.inner.native_mode = NativeMode::parse(value).map_err(PyValueError::new_err)?;
        Ok(())
    }

    fn validate(&self) -> PyResult<()> {
        validate_result(self.inner.validate())
    }
}
