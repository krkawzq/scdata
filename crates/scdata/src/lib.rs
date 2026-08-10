//! Static compilation and execution for block-granular single-cell prefetch.
//!
//! A [`compile`] call turns immutable sc-compress datasets and an ordered row
//! list into a reusable [`Plan`]. Each call to [`Plan::open`] creates an
//! independent output ring and worker lifecycle; plans never share mutable
//! session state.

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod compiler;
mod config;
mod convert;
mod dtype;
mod error;
mod output;
mod plan;
mod scatter;
mod session;
mod source;

#[cfg(all(target_os = "linux", target_has_atomic = "64"))]
mod share;

#[cfg(test)]
mod tests;

pub use compiler::{compile, PlanSpec};
pub use config::{IoMode, PlanConfig, ResourceLimits, SessionConfig};
pub use dtype::{promote_kind, OutputDType, OutputValue, PromoteKind, StorageDType};
pub use error::{Error, Result};
pub use output::{Fill, FloatCastPolicy, OutputSpec, OverflowPolicy};
pub use plan::{Plan, PlanStats};
#[cfg(feature = "profile")]
pub use session::WorkerRuntimeStats;
pub use session::{Batch, CancellationHandle, RuntimeStats, Session, SessionState};
#[cfg(all(target_os = "linux", target_has_atomic = "64"))]
pub use share::{
    SharedBatch, SharedCancellationHandle, SharedClient, SharedClientCancellationHandle,
    SharedConfig, SharedServer, DEFAULT_MAX_SHARED_CONTROL_BYTES,
};
pub use source::{Dataset, FeatureMap, RowRef, Source, SourceId};

pub use sc_compress::{ReadLimits, StoreLocation};
