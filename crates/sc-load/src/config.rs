use crate::{Error, Result};

const GIB: usize = 1024 * 1024 * 1024;
const MIB: usize = 1024 * 1024;
const DEFAULT_MAX_COMPILE_WORKING_SET_BYTES: usize = 40usize.saturating_mul(GIB);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoMode {
    Blocking,
    Uring { queue_depth: u32 },
    Auto { queue_depth: u32 },
}

impl Default for IoMode {
    fn default() -> Self {
        Self::Auto { queue_depth: 64 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoMergePolicy {
    Off,
    Adjacent,
    CostAware,
}

#[derive(Debug, Clone)]
pub struct IoMergeOptions {
    pub policy: IoMergePolicy,
    pub max_coalesced_io_bytes: usize,
    pub max_io_gap_bytes: usize,
    pub max_io_amplification_ratio: f64,
    pub max_decode_ops_per_io_task: usize,
    pub max_decoded_bytes_per_io_task: usize,
    pub max_encoded_staging_bytes_per_task: usize,
    pub io_bandwidth_bytes_per_second: f64,
    pub io_operations_per_second: f64,
    pub io_merge_delta_bytes: usize,
    pub initialize_parallelism_hint: usize,
    pub regular_io_parallelism_hint: usize,
    pub min_tasks_per_worker: usize,
}

impl Default for IoMergeOptions {
    fn default() -> Self {
        Self {
            policy: IoMergePolicy::Adjacent,
            max_coalesced_io_bytes: 32 * MIB,
            max_io_gap_bytes: 0,
            max_io_amplification_ratio: 1.0,
            max_decode_ops_per_io_task: 64,
            max_decoded_bytes_per_io_task: 64 * MIB,
            max_encoded_staging_bytes_per_task: 32 * MIB,
            io_bandwidth_bytes_per_second: 8.0 * GIB as f64,
            io_operations_per_second: 100_000.0,
            io_merge_delta_bytes: 4096,
            initialize_parallelism_hint: 32,
            regular_io_parallelism_hint: 32,
            min_tasks_per_worker: 2,
        }
    }
}

impl IoMergeOptions {
    fn validate(&self) -> Result<()> {
        let positive = [
            ("max_coalesced_io_bytes", self.max_coalesced_io_bytes),
            (
                "max_decode_ops_per_io_task",
                self.max_decode_ops_per_io_task,
            ),
            (
                "max_decoded_bytes_per_io_task",
                self.max_decoded_bytes_per_io_task,
            ),
            (
                "max_encoded_staging_bytes_per_task",
                self.max_encoded_staging_bytes_per_task,
            ),
            (
                "initialize_parallelism_hint",
                self.initialize_parallelism_hint,
            ),
            (
                "regular_io_parallelism_hint",
                self.regular_io_parallelism_hint,
            ),
            ("min_tasks_per_worker", self.min_tasks_per_worker),
        ];
        for (name, value) in positive {
            if value == 0 {
                return Err(Error::InvalidConfig(format!("{name} must be positive")));
            }
        }
        if self.max_coalesced_io_bytes > self.max_encoded_staging_bytes_per_task {
            return Err(Error::InvalidConfig(
                "max_coalesced_io_bytes exceeds max_encoded_staging_bytes_per_task".into(),
            ));
        }
        for (name, value) in [
            (
                "max_io_amplification_ratio",
                self.max_io_amplification_ratio,
            ),
            (
                "io_bandwidth_bytes_per_second",
                self.io_bandwidth_bytes_per_second,
            ),
            ("io_operations_per_second", self.io_operations_per_second),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::InvalidConfig(format!(
                    "{name} must be finite and positive"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ResourceLimits {
    pub max_output_buffer_bytes: usize,
    pub max_compile_arena_bytes: usize,
    /// Maximum charged payload simultaneously retained while compiling a plan.
    /// Defaults to 40 GiB on targets whose address space can represent it.
    pub max_compile_working_set_bytes: usize,
    /// Aggregate encoded WholeKey bytes retained in a compiled plan so the
    /// runtime can reuse compilation reads. Set to zero to disable retention.
    pub max_retained_whole_key_bytes: usize,
    pub max_blocks_per_job: usize,
    pub max_cells_per_job: usize,
    pub max_encoded_bytes_per_side: usize,
    /// Maximum combined decoded payload in one job. Compact task offsets use
    /// u32, so values above `u32::MAX` are rejected during configuration.
    pub max_decoded_bytes_per_job: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_output_buffer_bytes: 2 * GIB,
            max_compile_arena_bytes: 2 * GIB,
            max_compile_working_set_bytes: DEFAULT_MAX_COMPILE_WORKING_SET_BYTES,
            max_retained_whole_key_bytes: 512 * MIB,
            max_blocks_per_job: 4096,
            max_cells_per_job: 1_000_000,
            max_encoded_bytes_per_side: 1024 * 1024 * 1024,
            max_decoded_bytes_per_job: 2 * 1024 * 1024 * 1024,
        }
    }
}

impl ResourceLimits {
    pub(crate) fn validate(&self) -> Result<()> {
        let non_zero = [
            ("max_output_buffer_bytes", self.max_output_buffer_bytes),
            ("max_compile_arena_bytes", self.max_compile_arena_bytes),
            (
                "max_compile_working_set_bytes",
                self.max_compile_working_set_bytes,
            ),
            ("max_blocks_per_job", self.max_blocks_per_job),
            ("max_cells_per_job", self.max_cells_per_job),
            (
                "max_encoded_bytes_per_side",
                self.max_encoded_bytes_per_side,
            ),
            ("max_decoded_bytes_per_job", self.max_decoded_bytes_per_job),
        ];
        for (name, value) in non_zero {
            if value == 0 {
                return Err(Error::InvalidConfig(format!("{name} must be positive")));
            }
        }
        if self.max_compile_arena_bytes > self.max_compile_working_set_bytes {
            return Err(Error::InvalidConfig(
                "max_compile_arena_bytes exceeds max_compile_working_set_bytes".into(),
            ));
        }
        if self.max_retained_whole_key_bytes > self.max_compile_working_set_bytes {
            return Err(Error::InvalidConfig(
                "max_retained_whole_key_bytes exceeds max_compile_working_set_bytes".into(),
            ));
        }
        if self.max_decoded_bytes_per_job > u32::MAX as usize {
            return Err(Error::InvalidConfig(
                "max_decoded_bytes_per_job exceeds compact task offset capacity".into(),
            ));
        }
        if self.max_encoded_bytes_per_side > u32::MAX as usize {
            return Err(Error::InvalidConfig(
                "max_encoded_bytes_per_side exceeds compact block offset capacity".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct PlanConfig {
    /// Maximum number of chunk metadata reads issued concurrently during compilation.
    pub compile_io_concurrency: usize,
    pub io_merge: IoMergeOptions,
    /// Fixed decoded-cache capacity compiled into the residency graph.
    pub cache_capacity_bytes: usize,
    /// Cache extent alignment used by the compile-time allocator.
    pub cache_alignment: usize,
    /// Maximum extra waste admitted while preferring an earlier availability
    /// epoch over the exact best-fit extent.
    pub cache_fragmentation_slack_bytes: usize,
    pub limits: ResourceLimits,
}

impl Default for PlanConfig {
    fn default() -> Self {
        let compile_io_concurrency = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(32);
        Self {
            compile_io_concurrency,
            io_merge: IoMergeOptions::default(),
            cache_capacity_bytes: 64 * MIB,
            cache_alignment: 64,
            cache_fragmentation_slack_bytes: 64 * 1024,
            limits: ResourceLimits::default(),
        }
    }
}

impl PlanConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.compile_io_concurrency == 0 {
            return Err(Error::InvalidConfig(
                "compile_io_concurrency must be positive".into(),
            ));
        }
        if self.cache_capacity_bytes == 0 {
            return Err(Error::InvalidConfig(
                "cache_capacity_bytes must be positive".into(),
            ));
        }
        if self.cache_alignment == 0 || !self.cache_alignment.is_power_of_two() {
            return Err(Error::InvalidConfig(
                "cache_alignment must be a positive power of two".into(),
            ));
        }
        if self.cache_alignment < 64 {
            return Err(Error::InvalidConfig(
                "cache_alignment must be at least 64 bytes".into(),
            ));
        }
        self.io_merge.validate()?;
        self.limits.validate()
    }
}

#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub worker_count: usize,
    pub initialize_workers: usize,
    pub initialize_inflight_io_ops: usize,
    pub initialize_inflight_encoded_bytes: usize,
    pub io_mode: IoMode,
    pub max_inflight_jobs_per_worker: usize,
    pub max_inflight_encoded_bytes_per_worker: usize,
    pub max_decoded_bytes_per_worker: usize,
    pub max_total_inflight_io_ops: usize,
    pub max_total_inflight_encoded_bytes: usize,
    pub max_total_decoded_bytes: usize,
}

impl Default for SessionConfig {
    fn default() -> Self {
        let worker_count = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let queue_depth = 64usize;
        let encoded_bytes_per_worker = 512 * 1024 * 1024;
        let decoded_bytes_per_worker = 2 * 1024 * 1024 * 1024;
        Self {
            worker_count,
            initialize_workers: worker_count,
            initialize_inflight_io_ops: worker_count,
            initialize_inflight_encoded_bytes: 512 * MIB,
            io_mode: IoMode::Auto {
                queue_depth: queue_depth as u32,
            },
            max_inflight_jobs_per_worker: 32,
            max_inflight_encoded_bytes_per_worker: encoded_bytes_per_worker,
            max_decoded_bytes_per_worker: decoded_bytes_per_worker,
            max_total_inflight_io_ops: worker_count.saturating_mul(queue_depth),
            max_total_inflight_encoded_bytes: worker_count.saturating_mul(encoded_bytes_per_worker),
            max_total_decoded_bytes: worker_count.saturating_mul(decoded_bytes_per_worker),
        }
    }
}

impl SessionConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.worker_count == 0 {
            return Err(Error::InvalidConfig("worker_count must be positive".into()));
        }
        if self.initialize_workers == 0
            || self.initialize_inflight_io_ops == 0
            || self.initialize_inflight_encoded_bytes == 0
        {
            return Err(Error::InvalidConfig(
                "initialize worker and in-flight limits must be positive".into(),
            ));
        }
        if self.max_inflight_jobs_per_worker == 0 {
            return Err(Error::InvalidConfig(
                "max_inflight_jobs_per_worker must be positive".into(),
            ));
        }
        self.worker_count
            .checked_mul(self.max_inflight_jobs_per_worker)
            .ok_or_else(|| Error::InvalidConfig("total in-flight job count overflow".into()))?;
        if self.max_inflight_encoded_bytes_per_worker == 0
            || self.max_decoded_bytes_per_worker == 0
            || self.max_total_inflight_encoded_bytes == 0
            || self.max_total_decoded_bytes == 0
            || self.max_total_inflight_io_ops == 0
        {
            return Err(Error::InvalidConfig(
                "in-flight resource limits must be positive".into(),
            ));
        }
        let depth = match self.io_mode {
            IoMode::Blocking => 0,
            IoMode::Uring { queue_depth } | IoMode::Auto { queue_depth } => {
                if queue_depth < 2 {
                    return Err(Error::InvalidConfig(
                        "io_uring queue depth must be at least 2".into(),
                    ));
                }
                queue_depth
            }
        };
        let regular_ops = self
            .worker_count
            .checked_mul((depth as usize).max(1))
            .ok_or_else(|| Error::InvalidConfig("total I/O depth overflow".into()))?;
        let required_ops = regular_ops.max(self.initialize_inflight_io_ops);
        if required_ops > self.max_total_inflight_io_ops {
            return Err(Error::ResourceLimit(format!(
                "session requires up to {required_ops} in-flight I/O operations, limit is {}",
                self.max_total_inflight_io_ops
            )));
        }
        let regular_bytes = self
            .worker_count
            .checked_mul(self.max_inflight_encoded_bytes_per_worker)
            .ok_or_else(|| Error::InvalidConfig("total in-flight byte limit overflow".into()))?;
        let required_bytes = regular_bytes.max(self.initialize_inflight_encoded_bytes);
        if required_bytes > self.max_total_inflight_encoded_bytes {
            return Err(Error::ResourceLimit(format!(
                "session requires up to {required_bytes} in-flight encoded bytes, limit is {}",
                self.max_total_inflight_encoded_bytes
            )));
        }
        let decoded_workers = self.worker_count.max(self.initialize_workers);
        let total_decoded = decoded_workers
            .checked_mul(self.max_decoded_bytes_per_worker)
            .ok_or_else(|| Error::InvalidConfig("total decoded byte limit overflow".into()))?;
        if total_decoded > self.max_total_decoded_bytes {
            return Err(Error::ResourceLimit(format!(
                "session requires up to {total_decoded} decoded workspace bytes, limit is {}",
                self.max_total_decoded_bytes
            )));
        }
        Ok(())
    }
}
