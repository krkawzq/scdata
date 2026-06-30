use super::array::{build_array_from_spec, Array, ArraySpec, DType};
use super::error::{DataBankError, DataBankResult};
use super::interner::DatasetGeneRefs;
use crate::iopool::IoPool;

#[derive(Debug)]
pub enum Dataset {
    Dense1D(Dense1DDataset),
    Dense2D(Dense2DDataset),
    SparseCsr(SparseCsrDataset),
}

impl Dataset {
    pub fn genes(&self) -> &DatasetGeneRefs {
        match self {
            Self::Dense1D(dataset) => &dataset.genes,
            Self::Dense2D(dataset) => &dataset.genes,
            Self::SparseCsr(dataset) => &dataset.genes,
        }
    }

    pub fn num_cells(&self) -> usize {
        match self {
            Self::Dense1D(dataset) => dataset.num_cells,
            Self::Dense2D(dataset) => dataset.num_cells,
            Self::SparseCsr(dataset) => dataset.num_cells,
        }
    }

    pub fn num_genes(&self) -> usize {
        match self {
            Self::Dense1D(dataset) => dataset.num_genes,
            Self::Dense2D(dataset) => dataset.num_genes,
            Self::SparseCsr(dataset) => dataset.num_genes,
        }
    }

    pub fn data_dtype(&self) -> DType {
        match self {
            Self::Dense1D(dataset) => dataset.data.dtype,
            Self::Dense2D(dataset) => dataset.data.dtype,
            Self::SparseCsr(dataset) => dataset.data.dtype,
        }
    }

    pub fn unregister_files(&self, io_pool: &IoPool) -> DataBankResult<()> {
        match self {
            Self::Dense1D(dataset) => dataset.data.unregister_files(io_pool),
            Self::Dense2D(dataset) => dataset.data.unregister_files(io_pool),
            Self::SparseCsr(dataset) => {
                let indices_result = dataset.indices.unregister_files(io_pool);
                let data_result = dataset.data.unregister_files(io_pool);
                indices_result.and(data_result)
            }
        }
    }
}

#[derive(Debug)]
pub struct Dense1DDataset {
    pub genes: DatasetGeneRefs,
    pub data: Array,
    pub num_cells: usize,
    pub num_genes: usize,
}

#[derive(Debug)]
pub struct Dense2DDataset {
    pub genes: DatasetGeneRefs,
    pub data: Array,
    pub num_cells: usize,
    pub num_genes: usize,
}

#[derive(Debug)]
pub struct Dense1DSpec {
    pub gene_names: Vec<String>,
    pub data: ArraySpec,
}

#[derive(Debug)]
pub struct Dense2DSpec {
    pub gene_names: Vec<String>,
    pub data: ArraySpec,
}

#[derive(Debug)]
pub(crate) struct ResolvedDense1DSpec {
    pub genes: DatasetGeneRefs,
    pub data: ArraySpec,
}

#[derive(Debug)]
pub(crate) struct ResolvedDense2DSpec {
    pub genes: DatasetGeneRefs,
    pub data: ArraySpec,
}

impl Dense1DDataset {
    pub fn from_spec(
        genes: DatasetGeneRefs,
        spec: Dense1DSpec,
        io_pool: &IoPool,
    ) -> DataBankResult<Self> {
        build_dense_1d_dataset(
            ResolvedDense1DSpec {
                genes,
                data: spec.data,
            },
            io_pool,
        )
    }
}

impl Dense2DDataset {
    pub fn from_spec(
        genes: DatasetGeneRefs,
        spec: Dense2DSpec,
        io_pool: &IoPool,
    ) -> DataBankResult<Self> {
        build_dense_2d_dataset(
            ResolvedDense2DSpec {
                genes,
                data: spec.data,
            },
            io_pool,
        )
    }
}

#[derive(Debug)]
pub struct SparseCsrDataset {
    pub genes: DatasetGeneRefs,
    pub indptr: Vec<u64>,
    pub indices: Array,
    pub data: Array,
    pub index_dtype: DType,
    pub num_cells: usize,
    pub num_genes: usize,
}

#[derive(Debug)]
pub struct SparseCsrSpec {
    pub gene_names: Vec<String>,
    pub indptr: Vec<u64>,
    pub indices: ArraySpec,
    pub data: ArraySpec,
    pub index_dtype: DType,
    pub num_cells: usize,
    pub num_genes: usize,
}

#[derive(Debug)]
pub(crate) struct ResolvedSparseCsrSpec {
    pub genes: DatasetGeneRefs,
    pub indptr: Vec<u64>,
    pub indices: ArraySpec,
    pub data: ArraySpec,
    pub index_dtype: DType,
    pub num_cells: usize,
    pub num_genes: usize,
}

impl SparseCsrDataset {
    pub fn from_spec(
        genes: DatasetGeneRefs,
        spec: SparseCsrSpec,
        io_pool: &IoPool,
    ) -> DataBankResult<Self> {
        build_sparse_csr_dataset(
            ResolvedSparseCsrSpec {
                genes,
                indptr: spec.indptr,
                indices: spec.indices,
                data: spec.data,
                index_dtype: spec.index_dtype,
                num_cells: spec.num_cells,
                num_genes: spec.num_genes,
            },
            io_pool,
        )
    }
}

pub(crate) fn build_dense_1d_dataset(
    spec: ResolvedDense1DSpec,
    io_pool: &IoPool,
) -> DataBankResult<Dense1DDataset> {
    let num_genes = spec.genes.len();
    if num_genes == 0 {
        return Err(DataBankError::InvalidArrayMeta(
            "Dense1D requires at least one gene name".to_string(),
        ));
    }
    validate_gene_names(&spec.genes)?;
    let &[total_elements] = spec.data.shape.as_slice() else {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "Dense1D data must be 1D [cells * genes], got {:?}",
            spec.data.shape
        )));
    };
    if total_elements % num_genes != 0 {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "Dense1D length {total_elements} is not divisible by gene count {num_genes}"
        )));
    }

    let num_cells = total_elements / num_genes;
    let data = build_array_from_spec(spec.data, io_pool)?;
    Ok(Dense1DDataset {
        genes: spec.genes,
        data,
        num_cells,
        num_genes,
    })
}

pub(crate) fn build_dense_2d_dataset(
    spec: ResolvedDense2DSpec,
    io_pool: &IoPool,
) -> DataBankResult<Dense2DDataset> {
    let &[num_cells, num_genes] = spec.data.shape.as_slice() else {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "Dense2D data must be 2D [cells, genes], got {:?}",
            spec.data.shape
        )));
    };
    if spec.genes.len() != num_genes {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "gene name count is {}, expected {num_genes}",
            spec.genes.len()
        )));
    }
    validate_gene_names(&spec.genes)?;
    match &spec.data.grid {
        super::array::ArrayGridSpec::Regular { .. } => {}
        super::array::ArrayGridSpec::Rectilinear { .. } => {
            return Err(DataBankError::InvalidArrayMeta(
                "Dense2D requires a regular chunk grid".to_string(),
            ))
        }
    }

    let data = build_array_from_spec(spec.data, io_pool)?;
    Ok(Dense2DDataset {
        genes: spec.genes,
        data,
        num_cells,
        num_genes,
    })
}

pub(crate) fn build_sparse_csr_dataset(
    spec: ResolvedSparseCsrSpec,
    io_pool: &IoPool,
) -> DataBankResult<SparseCsrDataset> {
    let expected_indptr_len = spec
        .num_cells
        .checked_add(1)
        .ok_or_else(|| DataBankError::IndptrInvalid("num_cells + 1 overflows usize".to_string()))?;
    if spec.indptr.len() != expected_indptr_len {
        return Err(DataBankError::IndptrInvalid(format!(
            "indptr length is {}, expected {}",
            spec.indptr.len(),
            expected_indptr_len
        )));
    }
    if spec.indptr.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(DataBankError::IndptrInvalid(
            "indptr must be monotonically nondecreasing".to_string(),
        ));
    }
    if !spec.index_dtype.is_csr_index() {
        return Err(DataBankError::UnsupportedDType {
            dtype: spec.index_dtype,
            context: "CSR indices",
        });
    }
    if spec.genes.len() != spec.num_genes {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "gene name count is {}, expected {}",
            spec.genes.len(),
            spec.num_genes
        )));
    }
    validate_gene_names(&spec.genes)?;

    let nnz = usize::try_from(*spec.indptr.last().unwrap_or(&0))
        .map_err(|_| DataBankError::IndptrInvalid("nnz does not fit in usize".to_string()))?;
    validate_1d_spec_len(&spec.indices, nnz, "CSR indices")?;
    validate_1d_spec_len(&spec.data, nnz, "CSR data")?;
    if spec.indices.dtype != spec.index_dtype {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "index dtype mismatch: array {:?}, meta {:?}",
            spec.indices.dtype, spec.index_dtype
        )));
    }

    let indices = build_array_from_spec(spec.indices, io_pool)?;
    let data = match build_array_from_spec(spec.data, io_pool) {
        Ok(data) => data,
        Err(err) => {
            let _ = indices.unregister_files(io_pool);
            return Err(err);
        }
    };

    Ok(SparseCsrDataset {
        genes: spec.genes,
        indptr: spec.indptr,
        indices,
        data,
        index_dtype: spec.index_dtype,
        num_cells: spec.num_cells,
        num_genes: spec.num_genes,
    })
}

fn validate_1d_spec_len(
    array: &ArraySpec,
    expected: usize,
    context: &'static str,
) -> DataBankResult<()> {
    if array.shape.as_slice() != [expected] {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "{context} shape is {:?}, expected [{expected}]",
            array.shape
        )));
    }
    Ok(())
}

fn validate_gene_names(genes: &DatasetGeneRefs) -> DataBankResult<()> {
    if let Some(index) = genes.names().iter().position(|name| name.is_empty()) {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "gene name at index {index} must not be empty"
        )));
    }
    Ok(())
}
