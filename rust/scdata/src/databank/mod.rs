//! Single-cell DataBank facade.
//!
//! This module owns metadata registration, file-handle lifecycle, cell access,
//! generation-checked dataset ids, and gene-name views.

mod adapter;
mod array;
mod batch;
mod compute;
mod config;
mod dataset;
mod dense;
mod direct;
mod error;
mod gene_axis;
mod interner;
mod plan;
mod profile;
mod registry;
mod scheduled;
mod sparse;
mod util;

use std::sync::Arc;

pub use array::{
    ArrayCodecSpec, ArrayGridSpec, ArrayOrder, ArraySpec, Bf16Bits, ChunkSourceSpec, ChunkSpec,
    DType, DataValue, EdgeChunkLayout, F16Bits, RegisteredFile,
};
pub use batch::{MissingGenePolicy, MultiBatchCells, PrefetchCells, PrefetchedBatch};
pub use config::{
    DataBankConfig, FillConfig, ProjectedSparseDataGroupStrategy, ScheduledPrefetchConfig,
};
pub use dataset::{Dense1DSpec, Dense2DSpec, SparseCsrSpec};
pub use error::{DataBankError, DataBankResult};
pub use interner::GeneNameView;
pub use registry::DatasetId;

use crate::access::{AccessHandle, AccessScheduler, ScheduledAccessConfig};
use crate::codecs::DecodePool;
use crate::iopool::IoPool;
use crate::profile::{ProfileRuntime, ProfileSnapshot};

use adapter::{DecodePoolBackend, IoPoolBackend};
use compute::DataBankComputePool;
use dataset::{Dataset, Dense1DDataset, Dense2DDataset, SparseCsrDataset};
use interner::GeneInterner;
use profile::{DataBankAccessKind, DataBankProfile, DataBankRegisterKind, DataBankScheduledKind};
use registry::DatasetRegistry;

pub struct DataBank {
    io_pool: Arc<IoPool>,
    _decode_pool: Arc<DecodePool>,
    access: AccessHandle,
    compute: Arc<DataBankComputePool>,
    registry: DatasetRegistry,
    retired: Vec<Arc<Dataset>>,
    interner: GeneInterner,
    config: DataBankConfig,
    profiler: DataBankProfile,
}

impl DataBank {
    pub fn new(config: DataBankConfig) -> DataBankResult<Self> {
        Self::new_with_profile(config, DataBankProfile::from_env())
    }

    #[cfg(all(test, feature = "profile"))]
    pub(crate) fn new_with_profiler(
        config: DataBankConfig,
        profiler: DataBankProfile,
    ) -> DataBankResult<Self> {
        Self::new_with_profile(config, profiler)
    }

    fn new_with_profile(config: DataBankConfig, profiler: DataBankProfile) -> DataBankResult<Self> {
        config.validate().map_err(DataBankError::InvalidConfig)?;
        let io_pool = Arc::new(IoPool::new(config.io_config.clone())?);
        let decode_pool = Arc::new(DecodePool::new(config.decode_config.clone())?);
        let compute = Arc::new(DataBankComputePool::new(config.fill_config.clone())?);
        let access = AccessScheduler::spawn(
            config.access_config.clone(),
            Box::new(IoPoolBackend::new(Arc::clone(&io_pool))),
            Box::new(DecodePoolBackend::new(Arc::clone(&decode_pool))),
        )?;
        Ok(Self {
            io_pool,
            _decode_pool: decode_pool,
            access,
            compute,
            registry: DatasetRegistry::new(),
            retired: Vec::new(),
            interner: GeneInterner::new(),
            config,
            profiler,
        })
    }

    pub fn profile(&self) -> &ProfileRuntime {
        self.profiler.runtime()
    }

    pub fn profile_snapshot(&self) -> ProfileSnapshot {
        self.profiler.snapshot()
    }

    pub fn profile_snapshot_and_reset(&self) -> ProfileSnapshot {
        self.profiler.snapshot_and_reset()
    }

    pub fn reset_profile(&self) {
        self.profiler.reset_metrics();
    }

    pub fn access_profile_snapshot_and_reset(&self) -> ProfileSnapshot {
        self.access.profile_snapshot_and_reset()
    }

    pub fn io_profile_snapshot_and_reset(&self) -> ProfileSnapshot {
        self.io_pool.profile_snapshot_and_reset()
    }

    pub fn decode_profile_snapshot_and_reset(&self) -> ProfileSnapshot {
        self._decode_pool.profile_snapshot_and_reset()
    }

    pub fn reset_runtime_profiles(&self) {
        self.reset_profile();
        self.access.reset_profile();
        self.io_pool.reset_profile();
        self._decode_pool.reset_profile();
    }

    pub fn register_dense_1d(&mut self, spec: Dense1DSpec) -> DataBankResult<DatasetId> {
        let started = self.profiler.lifecycle_timer();
        let gene_count = spec.gene_names.len();
        let result = (|| {
            self.cleanup_retired()?;
            self.registry.ensure_can_register()?;
            let genes = self.interner.intern_dataset(&spec.gene_names);
            match Dense1DDataset::from_spec(genes.clone(), spec, self.io_pool.as_ref()) {
                Ok(dataset) => self.registry.register(Dataset::Dense1D(dataset)),
                Err(err) => {
                    self.interner.release_dataset(&genes);
                    Err(err)
                }
            }
        })();
        self.profiler.record_register(
            started,
            DataBankRegisterKind::Dense1D,
            gene_count,
            result.is_err(),
        );
        result
    }

    pub fn register_dense_2d(&mut self, spec: Dense2DSpec) -> DataBankResult<DatasetId> {
        let started = self.profiler.lifecycle_timer();
        let gene_count = spec.gene_names.len();
        let result = (|| {
            self.cleanup_retired()?;
            self.registry.ensure_can_register()?;
            let genes = self.interner.intern_dataset(&spec.gene_names);
            match Dense2DDataset::from_spec(genes.clone(), spec, self.io_pool.as_ref()) {
                Ok(dataset) => self.registry.register(Dataset::Dense2D(dataset)),
                Err(err) => {
                    self.interner.release_dataset(&genes);
                    Err(err)
                }
            }
        })();
        self.profiler.record_register(
            started,
            DataBankRegisterKind::Dense2D,
            gene_count,
            result.is_err(),
        );
        result
    }

    pub fn register_sparse_csr(&mut self, spec: SparseCsrSpec) -> DataBankResult<DatasetId> {
        let started = self.profiler.lifecycle_timer();
        let gene_count = spec.gene_names.len();
        let result = (|| {
            self.cleanup_retired()?;
            self.registry.ensure_can_register()?;
            let genes = self.interner.intern_dataset(&spec.gene_names);
            match SparseCsrDataset::from_spec(genes.clone(), spec, self.io_pool.as_ref()) {
                Ok(dataset) => self.registry.register(Dataset::SparseCsr(dataset)),
                Err(err) => {
                    self.interner.release_dataset(&genes);
                    Err(err)
                }
            }
        })();
        self.profiler.record_register(
            started,
            DataBankRegisterKind::SparseCsr,
            gene_count,
            result.is_err(),
        );
        result
    }

    pub fn unregister(&mut self, id: DatasetId) -> DataBankResult<()> {
        let started = self.profiler.lifecycle_timer();
        let result = (|| {
            self.cleanup_retired()?;
            let dataset = self.registry.remove(id)?;
            self.interner.release_dataset(dataset.genes());
            let retired = Arc::strong_count(&dataset) != 1;
            self.unregister_files_or_retire(dataset)?;
            Ok(retired)
        })();
        self.profiler.record_unregister(
            started,
            result.as_ref().copied().unwrap_or(false),
            result.is_err(),
        );
        result.map(|_| ())
    }

    pub fn access_cells<T: DataValue>(
        &self,
        id: DatasetId,
        cells: &[usize],
        out: &mut [T],
        names: Option<&mut [GeneNameView]>,
    ) -> DataBankResult<()> {
        self.access_cells_with_config(id, cells, out, names, ScheduledAccessConfig::default())
    }

    pub fn access_cells_values<T: DataValue>(
        &self,
        id: DatasetId,
        cells: &[usize],
        out: &mut [T],
    ) -> DataBankResult<()> {
        self.access_cells(id, cells, out, None)
    }

    pub fn access_cells_with_config<T: DataValue>(
        &self,
        id: DatasetId,
        cells: &[usize],
        out: &mut [T],
        names: Option<&mut [GeneNameView]>,
        config: ScheduledAccessConfig,
    ) -> DataBankResult<()> {
        let started = self.profiler.access_timer();
        let output_elements = out.len();
        let output_bytes = output_bytes::<T>(output_elements);
        let dataset = match self.registry.get(id) {
            Ok(dataset) => dataset,
            Err(err) => {
                self.profiler.record_access(
                    started,
                    DataBankAccessKind::Borrowed,
                    cells.len(),
                    0,
                    output_elements,
                    output_bytes,
                    true,
                );
                return Err(err);
            }
        };
        let output_genes = dataset.num_genes();
        let result = batch::access_cells(
            &self.access,
            &self.compute,
            &config,
            dataset,
            cells,
            out,
            names,
        );
        self.profiler.record_access(
            started,
            DataBankAccessKind::Borrowed,
            cells.len(),
            output_genes,
            output_elements,
            output_bytes,
            result.is_err(),
        );
        result
    }

    pub fn access_cells_by_gene_names<T, G>(
        &self,
        id: DatasetId,
        cells: &[usize],
        gene_names: &[G],
        out: &mut [T],
        names: Option<&mut [GeneNameView]>,
        missing: MissingGenePolicy,
    ) -> DataBankResult<()>
    where
        T: DataValue,
        G: AsRef<str>,
    {
        self.access_cells_by_gene_names_with_config(
            id,
            cells,
            gene_names,
            out,
            names,
            missing,
            ScheduledAccessConfig::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn access_cells_by_gene_names_with_config<T, G>(
        &self,
        id: DatasetId,
        cells: &[usize],
        gene_names: &[G],
        out: &mut [T],
        names: Option<&mut [GeneNameView]>,
        missing: MissingGenePolicy,
        config: ScheduledAccessConfig,
    ) -> DataBankResult<()>
    where
        T: DataValue,
        G: AsRef<str>,
    {
        let started = self.profiler.access_timer();
        let output_elements = out.len();
        let output_bytes = output_bytes::<T>(output_elements);
        let output_genes = gene_names.len();
        let dataset = match self.registry.get(id) {
            Ok(dataset) => dataset,
            Err(err) => {
                self.profiler.record_access(
                    started,
                    DataBankAccessKind::ByGeneNames,
                    cells.len(),
                    output_genes,
                    output_elements,
                    output_bytes,
                    true,
                );
                return Err(err);
            }
        };
        let result = batch::access_cells_by_gene_names(
            &self.access,
            &self.compute,
            &config,
            dataset,
            cells,
            gene_names,
            out,
            names,
            missing,
        );
        self.profiler.record_access(
            started,
            DataBankAccessKind::ByGeneNames,
            cells.len(),
            output_genes,
            output_elements,
            output_bytes,
            result.is_err(),
        );
        result
    }

    /// Access trusted CSR cells through the unchecked hot path.
    ///
    /// Dense datasets fall back to the checked path. For CSR datasets this skips
    /// access-time dtype, cell, output-length, and gene-index validation in the
    /// scatter loop.
    ///
    /// # Safety
    ///
    /// For CSR datasets, callers must guarantee that `id` refers to a loaded CSR
    /// dataset, `T` exactly matches the dataset value dtype, all `cells` are in
    /// range, `out.len() == cells.len() * num_genes`, every CSR gene index is
    /// non-negative and less than `num_genes`, and `names` is either `None` or
    /// has length `num_genes`. Violating the CSR index invariant can cause
    /// immediate out-of-bounds writes and memory corruption.
    pub unsafe fn access_cells_unchecked<T: DataValue>(
        &self,
        id: DatasetId,
        cells: &[usize],
        out: &mut [T],
        names: Option<&mut [GeneNameView]>,
    ) -> DataBankResult<()> {
        unsafe {
            self.access_cells_unchecked_with_config(
                id,
                cells,
                out,
                names,
                ScheduledAccessConfig::default(),
            )
        }
    }

    /// Access trusted CSR cells through the unchecked hot path with an explicit
    /// scheduled-access configuration.
    ///
    /// # Safety
    ///
    /// The caller must uphold the same invariants as
    /// [`Self::access_cells_unchecked`]: `id` must name a registered dataset,
    /// `T` must match the dataset value dtype, all requested `cells` must be
    /// in range, `out` must have exactly `cells.len() * num_genes` elements,
    /// every CSR gene index must be non-negative and less than `num_genes`,
    /// and `names`, when present, must have length `num_genes`. Violating the
    /// CSR index invariant can cause immediate out-of-bounds writes and memory
    /// corruption.
    pub unsafe fn access_cells_unchecked_with_config<T: DataValue>(
        &self,
        id: DatasetId,
        cells: &[usize],
        out: &mut [T],
        names: Option<&mut [GeneNameView]>,
        config: ScheduledAccessConfig,
    ) -> DataBankResult<()> {
        let started = self.profiler.access_timer();
        let output_elements = out.len();
        let output_bytes = output_bytes::<T>(output_elements);
        let dataset = match self.registry.get(id) {
            Ok(dataset) => dataset,
            Err(err) => {
                self.profiler.record_access(
                    started,
                    DataBankAccessKind::Unchecked,
                    cells.len(),
                    0,
                    output_elements,
                    output_bytes,
                    true,
                );
                return Err(err);
            }
        };
        let output_genes = dataset.num_genes();
        let result = unsafe {
            batch::access_cells_unchecked(
                &self.access,
                &self.compute,
                &config,
                dataset,
                cells,
                out,
                names,
            )
        };
        self.profiler.record_access(
            started,
            DataBankAccessKind::Unchecked,
            cells.len(),
            output_genes,
            output_elements,
            output_bytes,
            result.is_err(),
        );
        result
    }

    pub fn prefetch_cells(&self, id: DatasetId, cells: &[usize]) -> DataBankResult<()> {
        let started = self.profiler.prefetch_timer();
        let dataset = match self.registry.get(id) {
            Ok(dataset) => dataset,
            Err(err) => {
                self.profiler.record_prefetch(started, cells.len(), true);
                return Err(err);
            }
        };
        let result = batch::prefetch_cells(&self.access, dataset, cells);
        self.profiler
            .record_prefetch(started, cells.len(), result.is_err());
        result
    }

    /// Access cells with a databank-allocated output buffer.
    ///
    /// Equivalent to [`Self::access_cells`] but the row-major buffer
    /// (`cells.len() * num_genes` values) is allocated and returned by the
    /// databank instead of being supplied by the caller.
    pub fn access_cells_owned<T: DataValue>(
        &self,
        id: DatasetId,
        cells: &[usize],
    ) -> DataBankResult<Vec<T>> {
        self.access_cells_owned_with_config(id, cells, ScheduledAccessConfig::default())
    }

    pub fn access_cells_owned_with_config<T: DataValue>(
        &self,
        id: DatasetId,
        cells: &[usize],
        config: ScheduledAccessConfig,
    ) -> DataBankResult<Vec<T>> {
        let started = self.profiler.access_timer();
        let dataset = match self.registry.get(id) {
            Ok(dataset) => dataset,
            Err(err) => {
                self.profiler.record_access(
                    started,
                    DataBankAccessKind::Owned,
                    cells.len(),
                    0,
                    0,
                    0,
                    true,
                );
                return Err(err);
            }
        };
        let output_genes = dataset.num_genes();
        let attempted_elements = cells.len().checked_mul(output_genes).unwrap_or(0);
        let result = (|| {
            batch::validate_dtype_and_cells::<T>(dataset, cells)?;
            let total = cells.len().checked_mul(output_genes).ok_or_else(|| {
                DataBankError::InvalidConfig("output length overflow".to_string())
            })?;
            let mut out = vec![T::zero(); total];
            batch::access_cells_validated(
                &self.access,
                &self.compute,
                &config,
                dataset,
                cells,
                &mut out,
                true,
            )?;
            Ok(out)
        })();
        let output_elements = result
            .as_ref()
            .map_or(attempted_elements, std::vec::Vec::len);
        self.profiler.record_access(
            started,
            DataBankAccessKind::Owned,
            cells.len(),
            output_genes,
            output_elements,
            output_bytes::<T>(output_elements),
            result.is_err(),
        );
        result
    }

    pub fn access_cells_owned_by_gene_names<T, G>(
        &self,
        id: DatasetId,
        cells: &[usize],
        gene_names: &[G],
        missing: MissingGenePolicy,
    ) -> DataBankResult<Vec<T>>
    where
        T: DataValue,
        G: AsRef<str>,
    {
        self.access_cells_owned_by_gene_names_with_config(
            id,
            cells,
            gene_names,
            missing,
            ScheduledAccessConfig::default(),
        )
    }

    pub fn access_cells_owned_by_gene_names_with_config<T, G>(
        &self,
        id: DatasetId,
        cells: &[usize],
        gene_names: &[G],
        missing: MissingGenePolicy,
        config: ScheduledAccessConfig,
    ) -> DataBankResult<Vec<T>>
    where
        T: DataValue,
        G: AsRef<str>,
    {
        let started = self.profiler.access_timer();
        let output_genes = gene_names.len();
        let dataset = match self.registry.get(id) {
            Ok(dataset) => dataset,
            Err(err) => {
                self.profiler.record_access(
                    started,
                    DataBankAccessKind::OwnedByGeneNames,
                    cells.len(),
                    output_genes,
                    0,
                    0,
                    true,
                );
                return Err(err);
            }
        };
        let attempted_elements = cells.len().checked_mul(output_genes).unwrap_or(0);
        let result = batch::access_cells_by_gene_names_owned(
            &self.access,
            &self.compute,
            &config,
            dataset,
            cells,
            gene_names,
            missing,
        );
        let output_elements = result
            .as_ref()
            .map_or(attempted_elements, std::vec::Vec::len);
        self.profiler.record_access(
            started,
            DataBankAccessKind::OwnedByGeneNames,
            cells.len(),
            output_genes,
            output_elements,
            output_bytes::<T>(output_elements),
            result.is_err(),
        );
        result
    }

    /// Alias for [`Self::access_cells_owned`].
    pub fn access_cells_alloc<T: DataValue>(
        &self,
        id: DatasetId,
        cells: &[usize],
    ) -> DataBankResult<Vec<T>> {
        self.access_cells_owned(id, cells)
    }

    /// Build a scheduled prefetcher over a stream of cell batches.
    ///
    /// `batch_source` yields one batch of cell indices at a time. The
    /// prefetcher plans each batch independently, streams the resulting chunk
    /// reads into the access scheduler using the access config embedded in
    /// `config`, and caches decoded results in a databank-owned ring buffer of
    /// depth `config.prefetch_step`.
    ///
    /// The databank iterator (batches) is not aligned with the access iterator
    /// (chunk groups): one batch expands to a variable number of chunk groups,
    /// tracked per-batch by the prefetcher.
    ///
    /// Because results are cached in the ring buffer, no external output buffer
    /// is accepted. Consume the returned iterator to pull decoded batches.
    pub fn prefetch_cells_scheduled<T, I>(
        &self,
        id: DatasetId,
        batch_source: I,
        config: ScheduledPrefetchConfig,
    ) -> DataBankResult<PrefetchCells<T>>
    where
        T: DataValue,
        I: IntoIterator,
        I::IntoIter: Send + 'static,
        I::Item: AsRef<[usize]> + Send,
    {
        let started = self.profiler.scheduled_timer();
        let dataset = match self.registry.get_arc(id) {
            Ok(dataset) => dataset,
            Err(err) => {
                self.profiler
                    .record_scheduled(started, DataBankScheduledKind::Single, 1, true);
                return Err(err);
            }
        };
        let result = batch::prefetch_cells_scheduled(
            &self.access,
            Arc::clone(&self.compute),
            dataset,
            batch_source.into_iter(),
            config,
        );
        self.profiler
            .record_scheduled(started, DataBankScheduledKind::Single, 1, result.is_err());
        result
    }

    pub fn prefetch_cells_scheduled_by_gene_names<T, I, G>(
        &self,
        id: DatasetId,
        batch_source: I,
        gene_names: &[G],
        missing: MissingGenePolicy,
        config: ScheduledPrefetchConfig,
    ) -> DataBankResult<PrefetchCells<T>>
    where
        T: DataValue,
        I: IntoIterator,
        I::IntoIter: Send + 'static,
        I::Item: AsRef<[usize]> + Send,
        G: AsRef<str>,
    {
        let started = self.profiler.scheduled_timer();
        let dataset = match self.registry.get_arc(id) {
            Ok(dataset) => dataset,
            Err(err) => {
                self.profiler.record_scheduled(
                    started,
                    DataBankScheduledKind::SingleByGeneNames,
                    1,
                    true,
                );
                return Err(err);
            }
        };
        let result = batch::prefetch_cells_scheduled_by_gene_names(
            &self.access,
            Arc::clone(&self.compute),
            dataset,
            batch_source.into_iter(),
            gene_names,
            missing,
            config,
        );
        self.profiler.record_scheduled(
            started,
            DataBankScheduledKind::SingleByGeneNames,
            1,
            result.is_err(),
        );
        result
    }

    pub fn prefetch_cells_scheduled_multi<T, I>(
        &self,
        ids: &[DatasetId],
        batch_source: I,
        config: ScheduledPrefetchConfig,
    ) -> DataBankResult<PrefetchCells<T>>
    where
        T: DataValue,
        I: IntoIterator,
        I::IntoIter: Send + 'static,
        I::Item: Into<MultiBatchCells> + Send,
    {
        let started = self.profiler.scheduled_timer();
        let datasets = match self.dataset_arcs(ids) {
            Ok(datasets) => datasets,
            Err(err) => {
                self.profiler.record_scheduled(
                    started,
                    DataBankScheduledKind::Multi,
                    ids.len(),
                    true,
                );
                return Err(err);
            }
        };
        let result = batch::prefetch_cells_scheduled_multi(
            &self.access,
            Arc::clone(&self.compute),
            datasets,
            batch_source.into_iter(),
            config,
        );
        self.profiler.record_scheduled(
            started,
            DataBankScheduledKind::Multi,
            ids.len(),
            result.is_err(),
        );
        result
    }

    pub fn prefetch_cells_scheduled_multi_by_gene_names<T, I, G>(
        &self,
        ids: &[DatasetId],
        batch_source: I,
        gene_names: &[G],
        missing: MissingGenePolicy,
        config: ScheduledPrefetchConfig,
    ) -> DataBankResult<PrefetchCells<T>>
    where
        T: DataValue,
        I: IntoIterator,
        I::IntoIter: Send + 'static,
        I::Item: Into<MultiBatchCells> + Send,
        G: AsRef<str>,
    {
        let started = self.profiler.scheduled_timer();
        let datasets = match self.dataset_arcs(ids) {
            Ok(datasets) => datasets,
            Err(err) => {
                self.profiler.record_scheduled(
                    started,
                    DataBankScheduledKind::MultiByGeneNames,
                    ids.len(),
                    true,
                );
                return Err(err);
            }
        };
        let result = batch::prefetch_cells_scheduled_multi_by_gene_names(
            &self.access,
            Arc::clone(&self.compute),
            datasets,
            batch_source.into_iter(),
            gene_names,
            missing,
            config,
        );
        self.profiler.record_scheduled(
            started,
            DataBankScheduledKind::MultiByGeneNames,
            ids.len(),
            result.is_err(),
        );
        result
    }

    fn dataset_arcs(&self, ids: &[DatasetId]) -> DataBankResult<Arc<[Arc<Dataset>]>> {
        if ids.is_empty() {
            return Err(DataBankError::InvalidConfig(
                "prefetch requires at least one dataset".to_string(),
            ));
        }
        let mut datasets = Vec::with_capacity(ids.len());
        for &id in ids {
            datasets.push(self.registry.get_arc(id)?);
        }
        Ok(Arc::from(datasets.into_boxed_slice()))
    }

    /// Borrow the gene-name views for a registered dataset.
    ///
    /// The returned slice has length `num_genes` and matches the column order
    /// of every `access_cells*` / `prefetch_cells_scheduled` result for `id`.
    pub fn dataset_genes(&self, id: DatasetId) -> DataBankResult<&[GeneNameView]> {
        let dataset = self.registry.get(id)?;
        Ok(dataset.genes().views())
    }

    /// Number of cells (rows) in the registered dataset.
    pub fn dataset_num_cells(&self, id: DatasetId) -> DataBankResult<usize> {
        let dataset = self.registry.get(id)?;
        Ok(dataset.num_cells())
    }

    /// Number of genes (columns) in the registered dataset.
    pub fn dataset_num_genes(&self, id: DatasetId) -> DataBankResult<usize> {
        let dataset = self.registry.get(id)?;
        Ok(dataset.num_genes())
    }

    /// Stored value dtype of the registered dataset.
    pub fn dataset_dtype(&self, id: DatasetId) -> DataBankResult<DType> {
        let dataset = self.registry.get(id)?;
        Ok(dataset.data_dtype())
    }

    pub fn config(&self) -> &DataBankConfig {
        &self.config
    }

    fn unregister_files_or_retire(&mut self, dataset: Arc<Dataset>) -> DataBankResult<()> {
        if Arc::strong_count(&dataset) == 1 {
            dataset.unregister_files(self.io_pool.as_ref())
        } else {
            self.retired.push(dataset);
            Ok(())
        }
    }

    fn cleanup_retired(&mut self) -> DataBankResult<()> {
        let started = self.profiler.lifecycle_timer();
        let inspected = self.retired.len();
        let mut first_error = None;
        let mut retained = Vec::with_capacity(self.retired.len());

        for dataset in self.retired.drain(..) {
            if Arc::strong_count(&dataset) == 1 {
                if let Err(err) = dataset.unregister_files(self.io_pool.as_ref()) {
                    first_error.get_or_insert(err);
                }
            } else {
                retained.push(dataset);
            }
        }

        self.retired = retained;
        let retained = self.retired.len();
        let result = first_error.map_or(Ok(()), Err);
        self.profiler
            .record_cleanup(started, inspected, retained, result.is_err());
        result
    }
}

impl Drop for DataBank {
    fn drop(&mut self) {
        let started = self.profiler.lifecycle_timer();
        let _ = self.cleanup_retired();
        let mut drained = 0usize;
        for dataset in self.registry.drain() {
            drained += 1;
            self.interner.release_dataset(dataset.genes());
            if Arc::strong_count(&dataset) == 1 {
                let _ = dataset.unregister_files(self.io_pool.as_ref());
            }
        }
        self.profiler.record_drop(started, drained);
    }
}

fn output_bytes<T>(elements: usize) -> usize {
    elements.saturating_mul(std::mem::size_of::<T>())
}
