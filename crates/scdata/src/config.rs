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
    pub io_bandwidth_bytes_per_second: f64,
    pub io_operations_per_second: f64,
    pub coalescing_distance: usize,
    /// Soft upper bound for the combined encoded span produced by cross-block
    /// coalescing. An indivisible source block may be larger and remains
    /// subject to `ResourceLimits::max_encoded_bytes_per_side`.
    pub max_coalesced_io_bytes: usize,
    /// Soft decoded-payload target for a merged job. A single indivisible
    /// source block may exceed it but remains subject to the hard resource cap.
    pub target_decoded_bytes_per_job: usize,
    pub delta_bytes: f64,
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
            io_bandwidth_bytes_per_second: 8.0 * 1024.0 * 1024.0 * 1024.0,
            io_operations_per_second: 100_000.0,
            coalescing_distance: 1,
            max_coalesced_io_bytes: 32 * MIB,
            target_decoded_bytes_per_job: 64 * MIB,
            delta_bytes: 4096.0,
            limits: ResourceLimits::default(),
        }
    }
}

impl PlanConfig {
    pub(crate) fn validate(&self, prefetch_step: usize) -> Result<()> {
        if self.compile_io_concurrency == 0 {
            return Err(Error::InvalidConfig(
                "compile_io_concurrency must be positive".into(),
            ));
        }
        if self.max_coalesced_io_bytes == 0 {
            return Err(Error::InvalidConfig(
                "max_coalesced_io_bytes must be positive".into(),
            ));
        }
        if self.target_decoded_bytes_per_job == 0 {
            return Err(Error::InvalidConfig(
                "target_decoded_bytes_per_job must be positive".into(),
            ));
        }
        if self.target_decoded_bytes_per_job > self.limits.max_decoded_bytes_per_job {
            return Err(Error::InvalidConfig(
                "target_decoded_bytes_per_job exceeds max_decoded_bytes_per_job".into(),
            ));
        }
        if !self.io_bandwidth_bytes_per_second.is_finite()
            || self.io_bandwidth_bytes_per_second <= 0.0
        {
            return Err(Error::InvalidConfig(
                "io_bandwidth_bytes_per_second must be finite and positive".into(),
            ));
        }
        if !self.io_operations_per_second.is_finite() || self.io_operations_per_second <= 0.0 {
            return Err(Error::InvalidConfig(
                "io_operations_per_second must be finite and positive".into(),
            ));
        }
        if !self.delta_bytes.is_finite() || self.delta_bytes < 0.0 {
            return Err(Error::InvalidConfig(
                "delta_bytes must be finite and non-negative".into(),
            ));
        }
        let merge_threshold =
            self.io_bandwidth_bytes_per_second / self.io_operations_per_second + self.delta_bytes;
        if !merge_threshold.is_finite() {
            return Err(Error::InvalidConfig(
                "I/O balance plus delta_bytes must be finite".into(),
            ));
        }
        if self.coalescing_distance == 0 || self.coalescing_distance >= prefetch_step {
            return Err(Error::InvalidConfig(format!(
                "coalescing_distance must be in 1..prefetch_step (got {} and {prefetch_step})",
                self.coalescing_distance
            )));
        }
        self.limits.validate()
    }
}

#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub worker_count: usize,
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
            IoMode::Uring { queue_depth } | IoMode::Auto { queue_depth } => queue_depth,
        };
        if depth == 1 {
            return Err(Error::InvalidConfig(
                "io_uring queue depth must be at least 2".into(),
            ));
        }
        if depth > 0 {
            let total_ops = self
                .worker_count
                .checked_mul(depth as usize)
                .ok_or_else(|| Error::InvalidConfig("total io_uring depth overflow".into()))?;
            if total_ops > self.max_total_inflight_io_ops {
                return Err(Error::ResourceLimit(format!(
                    "worker_count * queue_depth is {total_ops}, limit is {}",
                    self.max_total_inflight_io_ops
                )));
            }
        }
        let total_bytes = self
            .worker_count
            .checked_mul(self.max_inflight_encoded_bytes_per_worker)
            .ok_or_else(|| Error::InvalidConfig("total in-flight byte limit overflow".into()))?;
        if total_bytes > self.max_total_inflight_encoded_bytes {
            return Err(Error::ResourceLimit(format!(
                "worker_count * per-worker encoded cap is {total_bytes}, limit is {}",
                self.max_total_inflight_encoded_bytes
            )));
        }
        let total_decoded = self
            .worker_count
            .checked_mul(self.max_decoded_bytes_per_worker)
            .ok_or_else(|| Error::InvalidConfig("total decoded byte limit overflow".into()))?;
        if total_decoded > self.max_total_decoded_bytes {
            return Err(Error::ResourceLimit(format!(
                "worker_count * per-worker decoded cap is {total_decoded}, limit is {}",
                self.max_total_decoded_bytes
            )));
        }
        Ok(())
    }
}
