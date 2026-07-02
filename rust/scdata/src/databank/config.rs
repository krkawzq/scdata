use std::collections::BTreeSet;

use crate::access::{AccessConfig, ScheduledAccessConfig};
use crate::codecs::DecodePoolConfig;
use crate::iopool::IoConfig;

/// Data loading strategy for projected sparse CSR data groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectedSparseDataGroupStrategy {
    /// Scan CSR indices first and load only data groups containing requested genes.
    SelectedOnly,
    /// Load every data group covered by the planned CSR rows.
    ReadAll,
}

impl Default for ProjectedSparseDataGroupStrategy {
    fn default() -> Self {
        Self::SelectedOnly
    }
}

impl ProjectedSparseDataGroupStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SelectedOnly => "selected_only",
            Self::ReadAll => "read_all",
        }
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "selected_only" | "selected-only" | "selected" => Ok(Self::SelectedOnly),
            "read_all" | "read-all" | "all" => Ok(Self::ReadAll),
            other => Err(format!(
                "projected_sparse_data_strategy must be 'selected_only' or 'read_all', got {other:?}"
            )),
        }
    }
}

/// Configuration for the single-cell DataBank facade.
#[derive(Debug, Clone)]
pub struct DataBankConfig {
    pub io_config: IoConfig,
    pub decode_config: DecodePoolConfig,
    pub access_config: AccessConfig,
    pub fill_config: FillConfig,
    pub native_config: NativeAccessConfig,
}

impl Default for DataBankConfig {
    fn default() -> Self {
        Self {
            io_config: IoConfig::default(),
            decode_config: DecodePoolConfig::default(),
            access_config: AccessConfig {
                keep_decoded: false,
                ..AccessConfig::default()
            },
            fill_config: FillConfig::default(),
            native_config: NativeAccessConfig::default(),
        }
    }
}

impl DataBankConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.io_config.validate()?;
        self.decode_config
            .validate()
            .map_err(|err| err.to_string())?;
        self.access_config.validate()?;
        self.fill_config.validate()?;
        self.native_config.validate()?;
        Ok(())
    }
}

/// Per-call routing for the Blosc-LZ4 native access path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeMode {
    Disabled,
    Auto,
    Force,
}

impl Default for NativeMode {
    fn default() -> Self {
        Self::Disabled
    }
}

impl NativeMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Auto => "auto",
            Self::Force => "force",
        }
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "disabled" | "off" | "false" => Ok(Self::Disabled),
            "auto" => Ok(Self::Auto),
            "force" | "forced" | "on" | "true" => Ok(Self::Force),
            other => Err(format!(
                "native_mode must be 'disabled', 'auto', or 'force', got {other:?}"
            )),
        }
    }
}

/// Top-level configuration for the Blosc-LZ4 native access path.
#[derive(Debug, Clone)]
pub struct NativeAccessConfig {
    pub enabled: bool,
    pub fused_workers: usize,
    pub request_prefetch_batches: usize,
    pub request_prefetch_blocks: usize,
    pub memory_budget_bytes: usize,
    pub response_queue_bytes_soft_limit: usize,
    pub response_queue_bytes_hard_limit: usize,
    pub load: NativeLoadConfig,
    pub blosc: NativeBloscConfig,
}

impl Default for NativeAccessConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            fused_workers: 4,
            request_prefetch_batches: 8,
            request_prefetch_blocks: 4096,
            memory_budget_bytes: 8 * 1024 * 1024 * 1024,
            response_queue_bytes_soft_limit: 4 * 1024 * 1024 * 1024,
            response_queue_bytes_hard_limit: 6 * 1024 * 1024 * 1024,
            load: NativeLoadConfig::default(),
            blosc: NativeBloscConfig::default(),
        }
    }
}

impl NativeAccessConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.enabled {
            if self.fused_workers == 0 {
                return Err("native.fused_workers must be greater than 0".to_string());
            }
            if self.request_prefetch_batches == 0 {
                return Err("native.request_prefetch_batches must be greater than 0".to_string());
            }
            if self.request_prefetch_blocks == 0 {
                return Err("native.request_prefetch_blocks must be greater than 0".to_string());
            }
        }
        if self.response_queue_bytes_soft_limit > self.response_queue_bytes_hard_limit {
            return Err("native.response_queue_bytes_soft_limit must be <= hard_limit".to_string());
        }
        if self.response_queue_bytes_hard_limit > self.memory_budget_bytes {
            return Err(
                "native.response_queue_bytes_hard_limit must be <= memory_budget_bytes".to_string(),
            );
        }
        self.load.validate()?;
        self.blosc.validate()?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct NativeLoadConfig {
    pub scheduler_workers: usize,
    pub io_workers: usize,
    pub coalesce: NativeLoadCoalesceConfig,
}

impl Default for NativeLoadConfig {
    fn default() -> Self {
        Self {
            scheduler_workers: 1,
            io_workers: 4,
            coalesce: NativeLoadCoalesceConfig::default(),
        }
    }
}

impl NativeLoadConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.scheduler_workers == 0 {
            return Err("native.load.scheduler_workers must be greater than 0".to_string());
        }
        if self.io_workers == 0 {
            return Err("native.load.io_workers must be greater than 0".to_string());
        }
        self.coalesce.validate()
    }
}

#[derive(Debug, Clone)]
pub struct NativeLoadCoalesceConfig {
    pub max_window_us: u32,
    pub max_merged_len: usize,
    pub max_gap_bytes: usize,
    pub max_waste_ratio: f32,
    pub min_children: usize,
}

impl Default for NativeLoadCoalesceConfig {
    fn default() -> Self {
        Self {
            max_window_us: 50,
            max_merged_len: 1024 * 1024,
            max_gap_bytes: 16 * 1024,
            max_waste_ratio: 0.10,
            min_children: 2,
        }
    }
}

impl NativeLoadCoalesceConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.max_waste_ratio) {
            return Err("native.load.coalesce.max_waste_ratio must be in [0, 1]".to_string());
        }
        if self.min_children == 0 {
            return Err("native.load.coalesce.min_children must be greater than 0".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct NativeBloscConfig {
    pub preload_block_tables: bool,
    pub full_unshuffle_threshold: f32,
    pub max_block_size: usize,
}

impl Default for NativeBloscConfig {
    fn default() -> Self {
        Self {
            preload_block_tables: true,
            full_unshuffle_threshold: 0.75,
            max_block_size: blosc_src::BLOSC_MAX_BLOCKSIZE as usize,
        }
    }
}

impl NativeBloscConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.full_unshuffle_threshold) {
            return Err("native.blosc.full_unshuffle_threshold must be in [0, 1]".to_string());
        }
        if self.max_block_size == 0 {
            return Err("native.blosc.max_block_size must be greater than 0".to_string());
        }
        Ok(())
    }
}

/// Per-call settings for scheduled DataBank cell prefetch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledPrefetchConfig {
    /// Number of databank batches kept decoded in the ring buffer.
    pub prefetch_step: usize,
    /// Access-layer scheduled look-ahead for file-backed chunk reads.
    pub access: ScheduledAccessConfig,
    /// Strategy for projected sparse CSR data groups.
    pub projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy,
    /// Native Blosc-LZ4 access routing for this scheduled request.
    pub native_mode: NativeMode,
}

impl Default for ScheduledPrefetchConfig {
    fn default() -> Self {
        Self {
            prefetch_step: 2,
            access: ScheduledAccessConfig::default(),
            projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy::default(),
            native_mode: NativeMode::default(),
        }
    }
}

impl ScheduledPrefetchConfig {
    pub fn validate(self) -> Result<(), String> {
        if self.prefetch_step == 0 {
            return Err("prefetch_step must be greater than 0".to_string());
        }
        Ok(())
    }
}

/// Configuration for future access-side output materialization.
#[derive(Debug, Clone)]
pub struct FillConfig {
    pub parallel: bool,
    pub num_workers: usize,
    pub queue_capacity: usize,
    pub min_parallel_rows: usize,
    pub min_parallel_bytes: usize,
    pub cpus: Option<Vec<usize>>,
}

impl Default for FillConfig {
    fn default() -> Self {
        Self {
            parallel: true,
            num_workers: 4,
            queue_capacity: 1024,
            min_parallel_rows: 16,
            min_parallel_bytes: 1024 * 1024,
            cpus: None,
        }
    }
}

impl FillConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.parallel && self.num_workers == 0 {
            return Err("fill.num_workers must be greater than 0".to_string());
        }
        if self.parallel && self.queue_capacity == 0 {
            return Err("fill.queue_capacity must be greater than 0".to_string());
        }
        if self.cpus.as_ref().is_some_and(Vec::is_empty) {
            return Err("fill.cpus list must not be empty".to_string());
        }
        if let Some(cpus) = &self.cpus {
            let unique = cpus.iter().copied().collect::<BTreeSet<_>>();
            if unique.len() != cpus.len() {
                return Err("fill.cpus list contains duplicate entries".to_string());
            }
        }
        Ok(())
    }
}
