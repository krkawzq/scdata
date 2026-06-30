#![allow(clippy::too_many_arguments)]

use super::*;

pub(crate) fn access_dense_1d<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &Dense1DDataset,
    cells: &[usize],
    out: &mut [T],
) -> DataBankResult<()> {
    if try_access_dense_1d_memory_identity_direct(dataset, cells, out)? {
        return Ok(());
    }
    let segments = plan::plan_dense_1d(dataset, cells)?;
    access_dense_segments(
        access,
        compute,
        access_config,
        dataset.num_genes,
        &segments,
        dataset.data.dtype,
        out,
    )
}

pub(crate) fn access_dense_1d_projected<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &Dense1DDataset,
    cells: &[usize],
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
    out_is_zeroed: bool,
) -> DataBankResult<()> {
    if gene_axis.requires_dense_zero_fill() && !out_is_zeroed {
        zero_values(out);
    }
    let segments = match gene_axis.projection() {
        Some(projection) => {
            plan::plan_dense_1d_selected_sources(dataset, cells, &projection.selected_sources)?
        }
        None => plan::plan_dense_1d(dataset, cells)?,
    };
    access_dense_segments_projected(
        access,
        compute,
        access_config,
        dataset.num_genes,
        gene_axis.output_genes(dataset.num_genes),
        &segments,
        dataset.data.dtype,
        gene_axis,
        out,
    )
}

pub(crate) fn access_dense_2d<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &Dense2DDataset,
    cells: &[usize],
    out: &mut [T],
) -> DataBankResult<()> {
    let segments = plan::plan_dense_2d(dataset, cells)?;
    access_dense_segments(
        access,
        compute,
        access_config,
        dataset.num_genes,
        &segments,
        dataset.data.dtype,
        out,
    )
}

pub(crate) fn access_dense_2d_projected<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &Dense2DDataset,
    cells: &[usize],
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
    out_is_zeroed: bool,
) -> DataBankResult<()> {
    if gene_axis.requires_dense_zero_fill() && !out_is_zeroed {
        zero_values(out);
    }
    let segments = match gene_axis.projection() {
        Some(projection) => {
            plan::plan_dense_2d_selected_sources(dataset, cells, &projection.selected_sources)?
        }
        None => plan::plan_dense_2d(dataset, cells)?,
    };
    access_dense_segments_projected(
        access,
        compute,
        access_config,
        dataset.num_genes,
        gene_axis.output_genes(dataset.num_genes),
        &segments,
        dataset.data.dtype,
        gene_axis,
        out,
    )
}

pub(crate) fn access_dense_segments<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    num_genes: usize,
    segments: &[DenseSegment],
    src_dtype: DType,
    out: &mut [T],
) -> DataBankResult<()> {
    let groups = group_dense_segments(segments)?;
    let output_bytes = out.len().checked_mul(mem::size_of::<T>()).ok_or_else(|| {
        DataBankError::InvalidConfig("dense output byte length overflow".to_string())
    })?;
    let output_rows = row_count_for_width(out.len(), num_genes);
    if compute.should_parallelize(output_rows, output_bytes) {
        let loaded_groups = load_dense_groups_for_parallel(access, access_config, &groups)?;
        if try_scatter_dense_rows_parallel(
            compute,
            num_genes,
            num_genes,
            segments,
            &groups,
            &loaded_groups,
            None,
            src_dtype,
            out,
        )? {
            return Ok(());
        }
    }
    if groups
        .iter()
        .all(|group| matches!(group.source, DenseGroupSource::AccessItem(_)))
    {
        access_dense_groups_scheduled(
            access,
            access_config,
            num_genes,
            segments,
            &groups,
            src_dtype,
            out,
        )
    } else {
        access_dense_groups_sequential(access, num_genes, segments, &groups, src_dtype, out)
    }
}

pub(crate) fn access_dense_segments_projected<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset_num_genes: usize,
    output_genes: usize,
    segments: &[DenseSegment],
    src_dtype: DType,
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    if segments.is_empty() {
        return Ok(());
    }
    let groups = group_dense_segments(segments)?;
    let output_bytes = out.len().checked_mul(mem::size_of::<T>()).ok_or_else(|| {
        DataBankError::InvalidConfig("dense output byte length overflow".to_string())
    })?;
    let output_rows = row_count_for_width(out.len(), output_genes);
    if compute.should_parallelize(output_rows, output_bytes) {
        let loaded_groups = load_dense_groups_for_parallel(access, access_config, &groups)?;
        if try_scatter_dense_rows_parallel(
            compute,
            dataset_num_genes,
            output_genes,
            segments,
            &groups,
            &loaded_groups,
            gene_axis.projection().cloned(),
            src_dtype,
            out,
        )? {
            return Ok(());
        }
    }
    if groups
        .iter()
        .all(|group| matches!(group.source, DenseGroupSource::AccessItem(_)))
    {
        access_dense_groups_scheduled_projected(
            access,
            access_config,
            dataset_num_genes,
            output_genes,
            segments,
            &groups,
            src_dtype,
            gene_axis,
            out,
        )
    } else {
        access_dense_groups_sequential_projected(
            access,
            dataset_num_genes,
            output_genes,
            segments,
            &groups,
            src_dtype,
            gene_axis,
            out,
        )
    }
}

pub(crate) fn access_dense_groups_scheduled<T: DataValue>(
    access: &AccessHandle,
    access_config: &ScheduledAccessConfig,
    num_genes: usize,
    segments: &[DenseSegment],
    groups: &[DenseReadGroup],
    src_dtype: DType,
    out: &mut [T],
) -> DataBankResult<()> {
    if groups.is_empty() {
        return Ok(());
    }

    let mut scheduled = access.scheduled(
        groups.iter().map(file_dense_group_access_item),
        *access_config,
    )?;
    for group in groups {
        let bytes = scheduled.next().ok_or_else(|| {
            DataBankError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "scheduled dense access ended early",
            ))
        })??;
        scatter_dense_group(num_genes, segments, group, &bytes, src_dtype, out)?;
    }
    if scheduled.next().is_some() {
        return Err(DataBankError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "scheduled dense access returned extra output",
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn access_dense_groups_scheduled_projected<T: DataValue>(
    access: &AccessHandle,
    access_config: &ScheduledAccessConfig,
    dataset_num_genes: usize,
    output_genes: usize,
    segments: &[DenseSegment],
    groups: &[DenseReadGroup],
    src_dtype: DType,
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    let mut scheduled = access.scheduled(
        groups.iter().map(file_dense_group_access_item),
        *access_config,
    )?;
    for group in groups {
        let bytes = scheduled.next().ok_or_else(|| {
            DataBankError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "scheduled dense access ended early",
            ))
        })??;
        scatter_dense_group_projected(
            dataset_num_genes,
            output_genes,
            segments,
            group,
            &bytes,
            src_dtype,
            gene_axis,
            out,
        )?;
    }
    if scheduled.next().is_some() {
        return Err(DataBankError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "scheduled dense access returned extra output",
        )));
    }
    Ok(())
}

pub(crate) fn access_dense_groups_sequential<T: DataValue>(
    access: &AccessHandle,
    num_genes: usize,
    segments: &[DenseSegment],
    groups: &[DenseReadGroup],
    src_dtype: DType,
    out: &mut [T],
) -> DataBankResult<()> {
    for group in groups {
        if try_scatter_dense_memory_identity_group(num_genes, segments, group, src_dtype, out)? {
            continue;
        }
        let bytes = load_dense_group(access, group)?;
        scatter_dense_group(num_genes, segments, group, &bytes, src_dtype, out)?;
    }
    Ok(())
}

pub(crate) fn access_dense_groups_sequential_projected<T: DataValue>(
    access: &AccessHandle,
    dataset_num_genes: usize,
    output_genes: usize,
    segments: &[DenseSegment],
    groups: &[DenseReadGroup],
    src_dtype: DType,
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    for group in groups {
        if try_scatter_dense_memory_identity_group_projected(
            dataset_num_genes,
            output_genes,
            segments,
            group,
            src_dtype,
            gene_axis,
            out,
        )? {
            continue;
        }
        let bytes = load_dense_group(access, group)?;
        scatter_dense_group_projected(
            dataset_num_genes,
            output_genes,
            segments,
            group,
            &bytes,
            src_dtype,
            gene_axis,
            out,
        )?;
    }
    Ok(())
}

pub(crate) fn try_access_dense_1d_memory_identity_direct<T: DataValue>(
    dataset: &Dense1DDataset,
    cells: &[usize],
    out: &mut [T],
) -> DataBankResult<bool> {
    // This path is a raw byte memcpy and only correct when the on-disk dtype
    // matches `T` exactly; any cast (e.g. bf16->f32) must fall through to the
    // segment scatter path which honours `src_dtype`.
    if dataset.data.dtype != T::DTYPE {
        return Ok(false);
    }
    let Some(chunks) = MemoryIdentity1DChunks::from_array(&dataset.data)? else {
        return Ok(false);
    };
    if mem::size_of::<T>() != T::DTYPE.item_size() {
        return Err(DataBankError::BufferSizeMismatch {
            expected: T::DTYPE.item_size(),
            actual: mem::size_of::<T>(),
        });
    }

    let chunk_len = chunks.chunk_len;
    let value_size = mem::size_of::<T>();
    let out_addr = out.as_mut_ptr() as usize;
    let out_len = out.len();

    for (output_row, &cell) in cells.iter().enumerate() {
        let row_start = cell.checked_mul(dataset.num_genes).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("Dense1D row start overflow".to_string())
        })?;
        let row_end = row_start.checked_add(dataset.num_genes).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("Dense1D row end overflow".to_string())
        })?;
        let mut pos = row_start;
        let mut output_col = 0usize;
        while pos < row_end {
            let chunk_index = pos / chunk_len;
            let in_chunk = pos % chunk_len;
            let chunk = chunks.chunk_bytes(chunk_index)?;
            let chunk_start = chunk_index.checked_mul(chunk_len).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("Dense1D chunk start overflow".to_string())
            })?;
            let physical_chunk_len = chunks.physical_chunk_len_at_start(chunk_start)?;
            let elements = (row_end - pos).min(physical_chunk_len - in_chunk);
            let src_start = in_chunk.checked_mul(value_size).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("Dense1D source byte start overflow".to_string())
            })?;
            let bytes = elements.checked_mul(value_size).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("Dense1D source byte length overflow".to_string())
            })?;
            let src_end = src_start.checked_add(bytes).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("Dense1D source byte end overflow".to_string())
            })?;
            let dst_start = output_row
                .checked_mul(dataset.num_genes)
                .and_then(|base| base.checked_add(output_col))
                .ok_or_else(|| {
                    DataBankError::InvalidConfig("Dense1D output offset overflow".to_string())
                })?;
            let dst_end = dst_start.checked_add(elements).ok_or_else(|| {
                DataBankError::InvalidConfig("Dense1D output end overflow".to_string())
            })?;
            if src_end > chunk.len() || dst_end > out_len {
                return Err(DataBankError::InvalidArrayMeta(
                    "Dense1D direct memory copy is out of range".to_string(),
                ));
            }
            let out_ptr = out_addr as *mut T;
            unsafe {
                ptr::copy_nonoverlapping(
                    chunk.as_ptr().add(src_start),
                    out_ptr.add(dst_start).cast::<u8>(),
                    bytes,
                );
            }
            pos += elements;
            output_col += elements;
        }
    }
    Ok(true)
}
