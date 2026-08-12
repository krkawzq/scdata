//! Dataset open handles and compile-time source registration.

use std::collections::HashSet;
use std::sync::Arc;

use sc_compress::{
    ByteStore, CsrMatrix, CsrMeta, DenseMatrix, DenseMeta, Kind, Matrix, ReadLimits, StoreLocation,
};

use crate::dtype::StorageDType;
use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceId(u32);

impl SourceId {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

impl From<u32> for SourceId {
    fn from(value: u32) -> Self {
        Self::new(value)
    }
}

/// One row sample reference: `(source, row)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RowRef {
    pub source: SourceId,
    pub row: u64,
}

impl RowRef {
    pub const fn new(source: SourceId, row: u64) -> Self {
        Self { source, row }
    }
}

/// Feature → output column map. `None` drops the source feature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureMap {
    targets: Vec<Option<usize>>,
}

impl FeatureMap {
    pub fn new(targets: impl IntoIterator<Item = Option<usize>>) -> Result<Self> {
        let iterator = targets.into_iter();
        let (lower, upper) = iterator.size_hint();
        let mut targets = Vec::new();
        targets.try_reserve_exact(upper.unwrap_or(lower))?;
        for target in iterator {
            if targets.len() == targets.capacity() {
                targets.try_reserve(1)?;
            }
            targets.push(target);
        }
        let mut seen = HashSet::new();
        seen.try_reserve(targets.len())?;
        for target in targets.iter().copied().flatten() {
            if !seen.insert(target) {
                return Err(Error::InvalidInput(format!(
                    "feature map contains duplicate output target {target}"
                )));
            }
        }
        Ok(Self { targets })
    }

    pub fn from_signed(targets: &[i64]) -> Result<Self> {
        let mut parsed = Vec::new();
        parsed.try_reserve_exact(targets.len())?;
        for (source, target) in targets.iter().copied().enumerate() {
            if target == -1 {
                parsed.push(None);
            } else if target < -1 {
                return Err(Error::InvalidInput(format!(
                    "feature map source {source} has invalid target {target}"
                )));
            } else {
                parsed.push(Some(usize::try_from(target).map_err(|_| {
                    Error::InvalidInput(format!("feature map target {target} exceeds usize"))
                })?));
            }
        }
        Self::new(parsed)
    }

    pub fn len(&self) -> usize {
        self.targets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    pub fn targets(&self) -> &[Option<usize>] {
        &self.targets
    }

    pub(crate) fn into_targets(self) -> Vec<Option<usize>> {
        self.targets
    }
}

#[derive(Clone)]
pub struct Dataset {
    pub(crate) store: Arc<dyn ByteStore>,
    pub(crate) kind: DatasetKind,
    pub(crate) limits: ReadLimits,
    pub(crate) initial_io_bytes: u64,
    pub(crate) initial_io_ops: u64,
}

#[derive(Clone)]
pub(crate) enum DatasetKind {
    Dense(DenseMeta),
    Csr { meta: CsrMeta, indptr: Arc<[u64]> },
}

impl Dataset {
    pub fn open(location: impl Into<StoreLocation>) -> Result<Self> {
        Self::open_with_limits(location, ReadLimits::default())
    }

    /// Open via sc-compress first-class [`Matrix`] path (single meta read).
    pub fn open_with_limits(
        location: impl Into<StoreLocation>,
        limits: ReadLimits,
    ) -> Result<Self> {
        let opened = Matrix::open_with_stats(location, limits)?;
        Ok(Self::from_matrix(
            opened.matrix,
            opened.stats.io_bytes,
            opened.stats.io_ops,
        ))
    }

    pub fn from_dense(matrix: DenseMatrix) -> Self {
        let limits = matrix.limits();
        let (store, meta, _) = matrix.into_parts();
        Self {
            store,
            kind: DatasetKind::Dense(meta),
            limits,
            initial_io_bytes: 0,
            initial_io_ops: 0,
        }
    }

    pub fn from_csr(matrix: CsrMatrix) -> Self {
        let limits = matrix.limits();
        let (store, meta, indptr, _) = matrix.into_parts();
        Self {
            store,
            kind: DatasetKind::Csr { meta, indptr },
            limits,
            initial_io_bytes: 0,
            initial_io_ops: 0,
        }
    }

    fn from_matrix(matrix: Matrix, io_bytes: u64, io_ops: u64) -> Self {
        match matrix {
            Matrix::Dense(matrix) => {
                let limits = matrix.limits();
                let (store, meta, _) = matrix.into_parts();
                Self {
                    store,
                    kind: DatasetKind::Dense(meta),
                    limits,
                    initial_io_bytes: io_bytes,
                    initial_io_ops: io_ops,
                }
            }
            Matrix::Csr(matrix) => {
                let limits = matrix.limits();
                let (store, meta, indptr, _) = matrix.into_parts();
                Self {
                    store,
                    kind: DatasetKind::Csr { meta, indptr },
                    limits,
                    initial_io_bytes: io_bytes,
                    initial_io_ops: io_ops,
                }
            }
        }
    }

    pub fn kind(&self) -> Kind {
        match &self.kind {
            DatasetKind::Dense(_) => Kind::Dense,
            DatasetKind::Csr { .. } => Kind::Csr,
        }
    }

    pub fn shape(&self) -> [u64; 2] {
        match &self.kind {
            DatasetKind::Dense(meta) => meta.shape,
            DatasetKind::Csr { meta, .. } => meta.shape,
        }
    }

    pub fn n_rows(&self) -> u64 {
        self.shape()[0]
    }

    pub fn n_cols(&self) -> u64 {
        self.shape()[1]
    }

    pub fn dtype(&self) -> StorageDType {
        match &self.kind {
            DatasetKind::Dense(meta) => meta.data.dtype,
            DatasetKind::Csr { meta, .. } => meta.data.dtype,
        }
    }
}

/// Registered compile-time source: id + dataset + optional feature map.
#[derive(Clone)]
pub struct Source {
    pub id: SourceId,
    pub dataset: Dataset,
    pub feature_map: Option<FeatureMap>,
}

impl Source {
    pub fn new(id: impl Into<SourceId>, dataset: Dataset) -> Self {
        Self {
            id: id.into(),
            dataset,
            feature_map: None,
        }
    }

    #[must_use]
    pub fn feature_map(mut self, feature_map: FeatureMap) -> Self {
        self.feature_map = Some(feature_map);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct OutputSlot {
    row_offset_and_fresh: usize,
}

impl OutputSlot {
    const FRESH: usize = 1;

    pub(crate) fn new(row_offset: usize, fresh: bool) -> Option<Self> {
        if row_offset & Self::FRESH != 0 {
            return None;
        }
        Some(Self {
            row_offset_and_fresh: row_offset | (usize::from(fresh) * Self::FRESH),
        })
    }

    pub(crate) fn row_offset(self) -> usize {
        self.row_offset_and_fresh & !Self::FRESH
    }

    pub(crate) fn is_fresh(self) -> bool {
        self.row_offset_and_fresh & Self::FRESH != 0
    }
}
