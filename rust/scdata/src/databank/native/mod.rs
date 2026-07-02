//! Blosc-LZ4 native access path scaffolding.
//!
//! This module owns the DataBank-side native metadata and execution path. The
//! codec-specific parsing/validation stays in `codecs::impls::blosc`; native
//! code consumes validated block ranges instead of duplicating Blosc logic.
#![allow(dead_code, unused_imports)]

mod executor;
mod item;
mod load;
mod metadata;
mod planner;

pub(crate) use executor::{
    scatter_loaded_blosc_block, scatter_loaded_blosc_block_multi_output, NativeBlockConsumer,
    NativeBlockDecodedCache, NativeBlockScratch,
};
pub(crate) use item::{load_access_item_blosc_lz4_native, load_access_items_blosc_lz4_native};
pub(crate) use load::{
    coalesce_load_requests, CoalescedChild, CoalescedRead, NativeBlockCacheKey,
    NativeBlockPayloadCache, NativeLoadCompletion, NativeLoadModule, NativeLoadRequest,
};
pub(crate) use metadata::{
    build_blosc_lz4_block_index, build_blosc_lz4_block_index_from_header_table, index_from_plan,
    NativeBlockIndexCache, NativeBloscBlockIndex, NativeBloscBlockRange,
};
pub(crate) use planner::{plan_blosc_slice_reads, NativeBlockReadPlan, NativeSliceBlockPlan};
