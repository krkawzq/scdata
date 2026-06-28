use super::array::{Array, ArrayMeta, DType};
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
pub struct Dense1DMeta {
    pub gene_names: Vec<String>,
    pub data: ArrayMeta,
}

#[derive(Debug)]
pub struct Dense2DMeta {
    pub gene_names: Vec<String>,
    pub data: ArrayMeta,
}

impl Dense1DDataset {
    pub fn from_meta(
        genes: DatasetGeneRefs,
        meta: Dense1DMeta,
        io_pool: &IoPool,
    ) -> DataBankResult<Self> {
        let num_genes = genes.len();
        if num_genes == 0 {
            return Err(DataBankError::InvalidArrayMeta(
                "Dense1D requires at least one gene name".to_string(),
            ));
        }
        validate_gene_names(&genes)?;
        let &[total_elements] = meta.data.shape.as_slice() else {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "Dense1D data must be 1D [cells * genes], got {:?}",
                meta.data.shape
            )));
        };
        if total_elements % num_genes != 0 {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "Dense1D length {total_elements} is not divisible by gene count {num_genes}"
            )));
        }

        let num_cells = total_elements / num_genes;
        let data = Array::from_meta(meta.data, io_pool)?;
        Ok(Self {
            genes,
            data,
            num_cells,
            num_genes,
        })
    }
}

impl Dense2DDataset {
    pub fn from_meta(
        genes: DatasetGeneRefs,
        meta: Dense2DMeta,
        io_pool: &IoPool,
    ) -> DataBankResult<Self> {
        let &[num_cells, num_genes] = meta.data.shape.as_slice() else {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "Dense2D data must be 2D [cells, genes], got {:?}",
                meta.data.shape
            )));
        };
        if genes.len() != num_genes {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "gene name count is {}, expected {num_genes}",
                genes.len()
            )));
        }
        validate_gene_names(&genes)?;

        let data = Array::from_meta(meta.data, io_pool)?;
        Ok(Self {
            genes,
            data,
            num_cells,
            num_genes,
        })
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
pub struct SparseCsrDatasetMeta {
    pub gene_names: Vec<String>,
    pub indptr: Vec<u64>,
    pub indices: ArrayMeta,
    pub data: ArrayMeta,
    pub index_dtype: DType,
    pub num_cells: usize,
    pub num_genes: usize,
}

impl SparseCsrDataset {
    pub fn from_meta(
        genes: DatasetGeneRefs,
        meta: SparseCsrDatasetMeta,
        io_pool: &IoPool,
    ) -> DataBankResult<Self> {
        let expected_indptr_len = meta.num_cells.checked_add(1).ok_or_else(|| {
            DataBankError::IndptrInvalid("num_cells + 1 overflows usize".to_string())
        })?;
        if meta.indptr.len() != expected_indptr_len {
            return Err(DataBankError::IndptrInvalid(format!(
                "indptr length is {}, expected {}",
                meta.indptr.len(),
                expected_indptr_len
            )));
        }
        if meta.indptr.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(DataBankError::IndptrInvalid(
                "indptr must be monotonically nondecreasing".to_string(),
            ));
        }
        if !meta.index_dtype.is_csr_index() {
            return Err(DataBankError::UnsupportedDType {
                dtype: meta.index_dtype,
                context: "CSR indices",
            });
        }
        if genes.len() != meta.num_genes {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "gene name count is {}, expected {}",
                genes.len(),
                meta.num_genes
            )));
        }
        validate_gene_names(&genes)?;

        let nnz = usize::try_from(*meta.indptr.last().unwrap_or(&0))
            .map_err(|_| DataBankError::IndptrInvalid("nnz does not fit in usize".to_string()))?;
        validate_1d_meta_len(&meta.indices, nnz, "CSR indices")?;
        validate_1d_meta_len(&meta.data, nnz, "CSR data")?;
        if meta.indices.dtype != meta.index_dtype {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "index dtype mismatch: array {:?}, meta {:?}",
                meta.indices.dtype, meta.index_dtype
            )));
        }

        let indices = Array::from_meta(meta.indices, io_pool)?;
        let data = match Array::from_meta(meta.data, io_pool) {
            Ok(data) => data,
            Err(err) => {
                let _ = indices.unregister_files(io_pool);
                return Err(err);
            }
        };

        Ok(Self {
            genes,
            indptr: meta.indptr,
            indices,
            data,
            index_dtype: meta.index_dtype,
            num_cells: meta.num_cells,
            num_genes: meta.num_genes,
        })
    }
}

fn validate_1d_meta_len(
    array: &ArrayMeta,
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
