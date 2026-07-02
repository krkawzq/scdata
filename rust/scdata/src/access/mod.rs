//! Chunk access scheduling.
//!
//! This module owns the async boundary between Python-facing requests,
//! positioned IO, compressed-byte caching, and per-request decoding. It shards
//! mutable scheduler state across one or more current-thread Tokio runtimes, so
//! each shard keeps cache, in-flight, and memory-budget transitions lock-free
//! while independent chunk keys can scale across scheduler threads.

mod backend;
mod cache;
mod cpu;
mod error;
mod inflight;
mod key;
mod membudget;
mod profile;
mod scheduled;
mod scheduler;
mod slice;

pub use backend::{DecodeBackend, DecodeTask, FileRef, IoBackend, IoTask};
pub use cpu::AccessCpuConfig;
pub use error::{AccessError, AccessResult};
pub use key::ChunkKey;
pub use profile::{access_profile_registry, AccessProfile};
pub use scheduler::{
    AccessConfig, AccessHandle, AccessItem, AccessRequest, AccessScheduler, PrefetchCancel,
    PrefetchRequest, ScheduledAccess, ScheduledAccessConfig,
};
pub(crate) use slice::SliceShape;
pub use slice::{RangeCopy, SliceSpec};
