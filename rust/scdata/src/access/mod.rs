//! Chunk access scheduling.
//!
//! This module owns the async boundary between Python-facing requests,
//! positioned IO, compressed-byte caching, and per-request decoding. It shards
//! mutable scheduler state across one or more current-thread Tokio runtimes, so
//! each shard keeps cache, in-flight, and memory-budget transitions lock-free
//! while independent chunk keys can scale across scheduler threads.

mod cache;
mod callback;
mod cpu;
mod error;
mod inflight;
mod membudget;
mod scheduler;

pub use callback::{DecodeBackend, DecodeTask, FileRef, IoBackend, IoTask};
pub use cpu::AccessCpuConfig;
pub use error::{AccessError, AccessResult};
pub use scheduler::{
    AccessConfig, AccessHandle, AccessItem, AccessRequest, AccessScheduler, ChunkKey,
    PrefetchCancel, PrefetchRequest, ScheduledAccess, ScheduledAccessConfig,
};
