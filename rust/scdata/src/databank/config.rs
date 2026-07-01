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
}

impl Default for ScheduledPrefetchConfig {
    fn default() -> Self {
        Self {
            prefetch_step: 2,
            access: ScheduledAccessConfig::default(),
            projected_sparse_data_strategy: ProjectedSparseDataGroupStrategy::default(),
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
