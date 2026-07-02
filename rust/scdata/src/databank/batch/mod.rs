pub(super) use super::direct::access_cells_validated;
pub use super::direct::{
    access_cells, access_cells_by_gene_names, access_cells_by_gene_names_owned,
    access_cells_unchecked, prefetch_cells,
};
pub(super) use super::gene_axis::validate_dtype_and_cells;
pub use super::gene_axis::{MissingGenePolicy, MultiBatchCells};
pub use super::scheduled::{
    prefetch_cells_scheduled, prefetch_cells_scheduled_by_gene_names,
    prefetch_cells_scheduled_multi, prefetch_cells_scheduled_multi_by_gene_names, PrefetchCells,
    PrefetchedBatch,
};
pub(super) use super::scheduled::{
    prefetch_cells_scheduled_by_gene_names_with_native,
    prefetch_cells_scheduled_multi_by_gene_names_with_native,
    prefetch_cells_scheduled_multi_with_native, prefetch_cells_scheduled_with_native,
};

#[cfg(test)]
use super::gene_axis::*;
#[cfg(test)]
use super::plan;
#[cfg(test)]
use super::scheduled::*;
#[cfg(test)]
use super::sparse::*;
#[cfg(test)]
use super::util::*;

#[cfg(test)]
mod tests;
