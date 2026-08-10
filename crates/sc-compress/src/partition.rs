use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// How rows are packed into chunks or DynBlosc blocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "strategy", rename_all = "snake_case")]
pub enum Partition {
    /// Exactly `n` cells per part (last part may be shorter).
    FixedCells { n: u64 },
    /// Greedy whole-cell packing under a decoded-byte budget of `n`.
    ///
    /// Cells are appended while the cumulative cost stays `<= n`. A single cell
    /// whose own cost already exceeds `n` still forms its own part.
    BytesBudget { n: u64 },
}

impl Partition {
    pub const fn fixed_cells(n: u64) -> Self {
        Self::FixedCells { n }
    }

    pub const fn bytes_budget(n: u64) -> Self {
        Self::BytesBudget { n }
    }

    pub const fn is_fixed_cells(&self) -> bool {
        matches!(self, Self::FixedCells { .. })
    }

    pub const fn fixed_cells_n(&self) -> Option<u64> {
        match self {
            Self::FixedCells { n } => Some(*n),
            Self::BytesBudget { .. } => None,
        }
    }

    pub fn validate(&self) -> Result<()> {
        let n = match self {
            Self::FixedCells { n } | Self::BytesBudget { n } => *n,
        };
        if n == 0 {
            return Err(Error::invalid_argument(
                "partition size must be greater than zero",
            ));
        }
        Ok(())
    }
}

/// Blosc1 `block_size` implied by a dense `fixed_cells` block partition.
pub(crate) fn dense_blosc1_block_size(
    block_cells: u64,
    n_genes: u64,
    element_size: usize,
) -> Result<u32> {
    if n_genes == 0 {
        return Ok(1);
    }
    let element_size = u64::try_from(element_size)
        .map_err(|_| Error::invalid_argument("element size exceeds u64"))?;
    let bytes = block_cells
        .checked_mul(n_genes)
        .and_then(|cells_genes| cells_genes.checked_mul(element_size))
        .ok_or_else(|| Error::invalid_argument("dense block byte size overflow"))?;
    u32::try_from(bytes)
        .map_err(|_| Error::invalid_argument("dense block byte size exceeds blosc1 u32 block_size"))
}

/// Visit dense chunk file boundaries over `n_cells` rows.
pub(crate) fn visit_dense_chunks(
    n_cells: u64,
    n_genes: u64,
    partition: &Partition,
    mut visit: impl FnMut(usize, ChunkSpan) -> Result<()>,
) -> Result<()> {
    let Partition::FixedCells { n } = partition else {
        return Err(Error::invalid_argument(
            "dense chunks require fixed_cells partition",
        ));
    };
    if n_cells == 0 {
        return Ok(());
    }
    partition.validate()?;
    let n_cells = to_usize(n_cells, "dense cell count")?;
    let per = to_usize(*n, "fixed_cells")?;
    let mut start = 0;
    let mut id = 0;
    while start < n_cells {
        let end = start + per.min(n_cells - start);
        visit(id, uniform_span(start, end, n_genes)?)?;
        start = end;
        id += 1;
    }
    Ok(())
}

/// Visit CSR chunk file boundaries from global `indptr`.
pub(crate) fn visit_csr_chunks(
    indptr: &[u64],
    element_bytes: usize,
    partition: &Partition,
    mut visit: impl FnMut(usize, ChunkSpan) -> Result<()>,
) -> Result<()> {
    partition.validate()?;
    validate_indptr(indptr)?;
    let n_cells = indptr.len().saturating_sub(1);
    if n_cells == 0 {
        return Ok(());
    }
    let mut id = 0;
    visit_csr_groups(indptr, element_bytes, partition, |start, end| {
        visit(id, csr_span(indptr, start, end)?)?;
        id += 1;
        Ok(())
    })?;
    Ok(())
}

/// Plan cell-aligned blocks for one CSR chunk from an `indptr` slice.
pub(crate) fn plan_csr_blocks(
    local_indptr: &[u64],
    element_bytes: usize,
    partition: &Partition,
) -> Result<BlockTable> {
    partition.validate()?;
    validate_monotonic_indptr(local_indptr)?;
    let n_cells = local_indptr.len().saturating_sub(1);
    if n_cells == 0 {
        return Ok(BlockTable { blocks: Vec::new() });
    }
    let mut blocks = Vec::new();
    reserve_fixed_group_capacity(&mut blocks, n_cells, partition)?;
    visit_csr_groups(local_indptr, element_bytes, partition, |start, end| {
        append_block(
            &mut blocks,
            start,
            end,
            local_indptr[end] - local_indptr[start],
        )
    })?;
    merge_trailing_empty_block(&mut blocks)?;
    Ok(BlockTable { blocks })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockEntry {
    pub first_cell: u64,
    pub n_cells: u64,
    pub nnz: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BlockTable {
    pub blocks: Vec<BlockEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChunkSpan {
    pub cell_start: u64,
    pub cell_end: u64,
    pub nnz_start: u64,
    pub nnz_end: u64,
}

fn csr_span(indptr: &[u64], start: usize, end: usize) -> Result<ChunkSpan> {
    Ok(ChunkSpan {
        cell_start: u64::try_from(start)
            .map_err(|_| Error::invalid_argument("cell start exceeds u64"))?,
        cell_end: u64::try_from(end)
            .map_err(|_| Error::invalid_argument("cell end exceeds u64"))?,
        nnz_start: indptr[start],
        nnz_end: indptr[end],
    })
}

fn uniform_span(start: usize, end: usize, nnz_per_cell: u64) -> Result<ChunkSpan> {
    let cell_start =
        u64::try_from(start).map_err(|_| Error::invalid_argument("cell start exceeds u64"))?;
    let cell_end =
        u64::try_from(end).map_err(|_| Error::invalid_argument("cell end exceeds u64"))?;
    Ok(ChunkSpan {
        cell_start,
        cell_end,
        nnz_start: cell_start
            .checked_mul(nnz_per_cell)
            .ok_or_else(|| Error::invalid_argument("dense span start overflow"))?,
        nnz_end: cell_end
            .checked_mul(nnz_per_cell)
            .ok_or_else(|| Error::invalid_argument("dense span end overflow"))?,
    })
}

fn append_block(blocks: &mut Vec<BlockEntry>, start: usize, end: usize, nnz: u64) -> Result<()> {
    let n_cells = u64::try_from(end - start)
        .map_err(|_| Error::invalid_argument("block cell count exceeds u64"))?;
    // dyn-blosc forbids zero-length blocks. Fold empty cell groups into neighbors.
    if nnz == 0 {
        if let Some(last) = blocks.last_mut() {
            last.n_cells = last
                .n_cells
                .checked_add(n_cells)
                .ok_or_else(|| Error::invalid_argument("block cell count overflow"))?;
            return Ok(());
        }
        return push_fallible(
            blocks,
            BlockEntry {
                first_cell: u64::try_from(start)
                    .map_err(|_| Error::invalid_argument("block start exceeds u64"))?,
                n_cells,
                nnz: 0,
            },
        );
    }

    if blocks.last().is_some_and(|last| last.nnz == 0) {
        let last = blocks
            .last_mut()
            .ok_or_else(|| Error::invalid_argument("missing empty block"))?;
        last.n_cells = last
            .n_cells
            .checked_add(n_cells)
            .ok_or_else(|| Error::invalid_argument("block cell count overflow"))?;
        last.nnz = nnz;
        return Ok(());
    }

    push_fallible(
        blocks,
        BlockEntry {
            first_cell: u64::try_from(start)
                .map_err(|_| Error::invalid_argument("block start exceeds u64"))?,
            n_cells,
            nnz,
        },
    )
}

fn merge_trailing_empty_block(blocks: &mut Vec<BlockEntry>) -> Result<()> {
    while let Some(empty) = blocks.last().copied().filter(|block| block.nnz == 0) {
        let _ = blocks.pop();
        if let Some(last) = blocks.last_mut() {
            last.n_cells = last
                .n_cells
                .checked_add(empty.n_cells)
                .ok_or_else(|| Error::invalid_argument("block cell count overflow"))?;
        }
    }
    Ok(())
}

fn visit_csr_groups(
    indptr: &[u64],
    element_bytes: usize,
    partition: &Partition,
    mut visit: impl FnMut(usize, usize) -> Result<()>,
) -> Result<()> {
    let n = indptr.len() - 1;
    match partition {
        Partition::FixedCells { n: per } => {
            let per = to_usize(*per, "fixed_cells")?;
            let mut start = 0;
            while start < n {
                let end = start + per.min(n - start);
                visit(start, end)?;
                start = end;
            }
        }
        Partition::BytesBudget { n: max_bytes } => {
            let element_bytes = u64::try_from(element_bytes)
                .map_err(|_| Error::invalid_argument("csr element size exceeds u64"))?;
            let mut start = 0;
            while start < n {
                let mut end = start;
                let mut cost = 0u64;
                while end < n {
                    let next_nnz = indptr[end + 1]
                        .checked_sub(indptr[end])
                        .ok_or_else(|| Error::invalid_argument("indptr must be monotonic"))?;
                    let next_cost = next_nnz
                        .checked_mul(element_bytes)
                        .ok_or_else(|| Error::invalid_argument("csr byte cost overflow"))?;
                    let would_cost = cost
                        .checked_add(next_cost)
                        .ok_or_else(|| Error::invalid_argument("csr byte cost overflow"))?;
                    if end > start && would_cost > *max_bytes {
                        break;
                    }
                    cost = would_cost;
                    end += 1;
                    if next_cost > *max_bytes {
                        break;
                    }
                }
                if end == start {
                    return Err(Error::invalid_argument(
                        "bytes_budget planner failed to advance",
                    ));
                }
                visit(start, end)?;
                start = end;
            }
        }
    }
    Ok(())
}

fn reserve_fixed_group_capacity<T>(
    output: &mut Vec<T>,
    n_cells: usize,
    partition: &Partition,
) -> Result<()> {
    if let Partition::FixedCells { n } = partition {
        let per = to_usize(*n, "fixed_cells")?;
        output.try_reserve_exact(n_cells.div_ceil(per))?;
    }
    Ok(())
}

fn push_fallible<T>(output: &mut Vec<T>, value: T) -> Result<()> {
    if output.len() == output.capacity() {
        output.try_reserve(1)?;
    }
    output.push(value);
    Ok(())
}

pub(crate) fn validate_indptr(indptr: &[u64]) -> Result<()> {
    if indptr.is_empty() {
        return Err(Error::invalid_argument("indptr must not be empty"));
    }
    if indptr[0] != 0 {
        return Err(Error::invalid_argument("indptr must start at 0"));
    }
    validate_monotonic_indptr(indptr)
}

fn validate_monotonic_indptr(indptr: &[u64]) -> Result<()> {
    if indptr.is_empty() {
        return Err(Error::invalid_argument("indptr must not be empty"));
    }
    for window in indptr.windows(2) {
        if window[1] < window[0] {
            return Err(Error::invalid_argument(
                "indptr must be monotonically non-decreasing",
            ));
        }
    }
    Ok(())
}

/// Convert a block table into dyn-blosc `block_lengths` (decoded bytes per block).
pub(crate) fn block_lengths_bytes(blocks: &BlockTable, element_size: usize) -> Result<Vec<usize>> {
    let mut lengths = Vec::new();
    lengths.try_reserve_exact(blocks.blocks.len())?;
    for block in &blocks.blocks {
        let nnz = usize::try_from(block.nnz)
            .map_err(|_| Error::invalid_argument("block nnz exceeds usize"))?;
        lengths.push(
            nnz.checked_mul(element_size)
                .ok_or_else(|| Error::invalid_argument("block byte length overflow"))?,
        );
    }
    Ok(lengths)
}

fn to_usize(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::invalid_argument(format!("{context} exceeds usize")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_sized_partitions_are_rejected() {
        assert!(Partition::fixed_cells(0).validate().is_err());
        assert!(Partition::bytes_budget(0).validate().is_err());
        assert!(visit_dense_chunks(1, 1, &Partition::fixed_cells(0), |_, _| Ok(())).is_err());
    }

    #[test]
    fn descending_csr_cost_is_rejected() {
        assert!(
            visit_csr_chunks(&[0, 2, 1], 4, &Partition::bytes_budget(16), |_, _| Ok(())).is_err()
        );
    }

    #[test]
    fn overflowing_csr_byte_cost_is_rejected() {
        assert!(visit_csr_chunks(
            &[0, u64::MAX],
            2,
            &Partition::bytes_budget(u64::MAX),
            |_, _| Ok(())
        )
        .is_err());
    }

    #[test]
    fn bytes_budget_groups_are_greedy_and_keep_oversized_cells() {
        assert_eq!(
            collect_csr_groups(&[0, 1, 3, 3], 4, &Partition::bytes_budget(8)).unwrap(),
            vec![(0, 1), (1, 3)]
        );
        assert_eq!(
            collect_csr_groups(&[0, 3], 4, &Partition::bytes_budget(8)).unwrap(),
            vec![(0, 1)]
        );
    }

    fn collect_csr_groups(
        indptr: &[u64],
        element_bytes: usize,
        partition: &Partition,
    ) -> Result<Vec<(usize, usize)>> {
        let mut groups = Vec::new();
        visit_csr_groups(indptr, element_bytes, partition, |start, end| {
            push_fallible(&mut groups, (start, end))
        })?;
        Ok(groups)
    }
}
