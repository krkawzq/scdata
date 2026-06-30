#![allow(clippy::too_many_arguments)]

use super::*;

pub(crate) fn try_scatter_dense_memory_identity_group<T: DataValue>(
    num_genes: usize,
    segments: &[DenseSegment],
    group: &DenseReadGroup,
    src_dtype: DType,
    out: &mut [T],
) -> DataBankResult<bool> {
    let DenseGroupSource::Memory {
        bytes,
        codec,
        expected_size,
        decoded,
    } = &group.source
    else {
        return Ok(false);
    };
    if !(*decoded || codec.is_identity()) {
        return Ok(false);
    }
    if bytes.len() != *expected_size {
        return Err(CodecError::SizeMismatch {
            codec: codec.name().to_string(),
            expected: *expected_size,
            actual: bytes.len(),
        }
        .into());
    }

    scatter_dense_group_from_decoded_source(
        num_genes,
        segments,
        group,
        bytes.as_ref(),
        src_dtype,
        out,
    )?;
    Ok(true)
}

pub(crate) fn try_scatter_dense_memory_identity_group_projected<T: DataValue>(
    dataset_num_genes: usize,
    output_genes: usize,
    segments: &[DenseSegment],
    group: &DenseReadGroup,
    src_dtype: DType,
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<bool> {
    let DenseGroupSource::Memory {
        bytes,
        codec,
        expected_size,
        decoded,
    } = &group.source
    else {
        return Ok(false);
    };
    if !(*decoded || codec.is_identity()) {
        return Ok(false);
    }
    if bytes.len() != *expected_size {
        return Err(CodecError::SizeMismatch {
            codec: codec.name().to_string(),
            expected: *expected_size,
            actual: bytes.len(),
        }
        .into());
    }

    scatter_dense_group_from_decoded_source_projected(
        dataset_num_genes,
        output_genes,
        segments,
        group,
        bytes.as_ref(),
        src_dtype,
        gene_axis,
        out,
    )?;
    Ok(true)
}

pub(crate) fn scatter_dense_group_from_decoded_source<T: DataValue>(
    num_genes: usize,
    segments: &[DenseSegment],
    group: &DenseReadGroup,
    decoded: &[u8],
    src_dtype: DType,
    out: &mut [T],
) -> DataBankResult<()> {
    for part in &group.parts {
        let segment = segments.get(part.segment_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("dense group segment index is invalid".to_string())
        })?;
        let len = dense_segment_bytes(segment, src_dtype)?;
        if part.bytes != len || segment.source.len() != len {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "dense group part length is {}, source length is {}, expected {len}",
                part.bytes,
                segment.source.len()
            )));
        }
        if segment.source.end > decoded.len() {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "dense chunk decoded length is {}, expected at least {}",
                decoded.len(),
                segment.source.end
            )));
        }
        scatter_dense_segment(
            num_genes,
            segment,
            &decoded[segment.source.start..segment.source.end],
            src_dtype,
            out,
        )?;
    }
    Ok(())
}

pub(crate) fn scatter_dense_group_from_decoded_source_projected<T: DataValue>(
    dataset_num_genes: usize,
    output_genes: usize,
    segments: &[DenseSegment],
    group: &DenseReadGroup,
    decoded: &[u8],
    src_dtype: DType,
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    for part in &group.parts {
        let segment = segments.get(part.segment_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("dense group segment index is invalid".to_string())
        })?;
        let len = dense_segment_bytes(segment, src_dtype)?;
        if part.bytes != len || segment.source.len() != len {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "dense group part length is {}, source length is {}, expected {len}",
                part.bytes,
                segment.source.len()
            )));
        }
        if segment.source.end > decoded.len() {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "dense chunk decoded length is {}, expected at least {}",
                decoded.len(),
                segment.source.end
            )));
        }
        scatter_dense_segment_projected(
            dataset_num_genes,
            output_genes,
            segment,
            &decoded[segment.source.start..segment.source.end],
            src_dtype,
            gene_axis,
            out,
        )?;
    }
    Ok(())
}

pub(crate) fn scatter_dense_group<T: DataValue>(
    num_genes: usize,
    segments: &[DenseSegment],
    group: &DenseReadGroup,
    bytes: &[u8],
    src_dtype: DType,
    out: &mut [T],
) -> DataBankResult<()> {
    if bytes.len() != group.bytes {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "dense group decoded length is {}, expected {}",
            bytes.len(),
            group.bytes
        )));
    }

    for part in &group.parts {
        let segment = segments.get(part.segment_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("dense group segment index is invalid".to_string())
        })?;
        let len = dense_segment_bytes(segment, src_dtype)?;
        if part.bytes != len {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "dense group part length is {}, expected {len}",
                part.bytes
            )));
        }
        let end = part.group_offset.checked_add(len).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("dense group byte offset overflow".to_string())
        })?;
        if end > bytes.len() {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "dense group decoded length is {}, expected at least {end}",
                bytes.len()
            )));
        }
        scatter_dense_segment(
            num_genes,
            segment,
            &bytes[part.group_offset..end],
            src_dtype,
            out,
        )?;
    }
    Ok(())
}

pub(crate) fn scatter_dense_group_projected<T: DataValue>(
    dataset_num_genes: usize,
    output_genes: usize,
    segments: &[DenseSegment],
    group: &DenseReadGroup,
    bytes: &[u8],
    src_dtype: DType,
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    if bytes.len() != group.bytes {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "dense group decoded length is {}, expected {}",
            bytes.len(),
            group.bytes
        )));
    }

    for part in &group.parts {
        let segment = segments.get(part.segment_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("dense group segment index is invalid".to_string())
        })?;
        let len = dense_segment_bytes(segment, src_dtype)?;
        if part.bytes != len {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "dense group part length is {}, expected {len}",
                part.bytes
            )));
        }
        let end = part.group_offset.checked_add(len).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("dense group byte offset overflow".to_string())
        })?;
        if end > bytes.len() {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "dense group decoded length is {}, expected at least {end}",
                bytes.len()
            )));
        }
        scatter_dense_segment_projected(
            dataset_num_genes,
            output_genes,
            segment,
            &bytes[part.group_offset..end],
            src_dtype,
            gene_axis,
            out,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_scatter_dense_rows_parallel<T: DataValue>(
    compute: &DataBankComputePool,
    dataset_num_genes: usize,
    output_genes: usize,
    segments: &[DenseSegment],
    groups: &[DenseReadGroup],
    loaded_groups: &[DenseLoadedGroup],
    projection: Option<CompiledGeneProjection>,
    src_dtype: DType,
    out: &mut [T],
) -> DataBankResult<bool> {
    if output_genes == 0 || segments.is_empty() {
        return Ok(true);
    }
    if out.len() % output_genes != 0 {
        return Err(DataBankError::BufferSizeMismatch {
            expected: out.len().div_ceil(output_genes) * output_genes,
            actual: out.len(),
        });
    }
    let output_rows = out.len() / output_genes;
    let output_bytes = out.len().checked_mul(mem::size_of::<T>()).ok_or_else(|| {
        DataBankError::InvalidConfig("dense output byte length overflow".to_string())
    })?;
    if !compute.should_parallelize(output_rows, output_bytes) {
        return Ok(false);
    }
    if loaded_groups.len() != groups.len() {
        return Err(DataBankError::InvalidArrayMeta(
            "dense loaded group count mismatch".to_string(),
        ));
    }
    for (group, loaded) in groups.iter().zip(loaded_groups.iter()) {
        if let DenseLoadedGroup::Packed(bytes) = loaded {
            if bytes.len() != group.bytes {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "dense group decoded length is {}, expected {}",
                    bytes.len(),
                    group.bytes
                )));
            }
        }
    }

    let segments: Arc<[DenseSegment]> = Arc::from(segments.to_vec().into_boxed_slice());
    let groups: Arc<[DenseReadGroup]> = Arc::from(groups.to_vec().into_boxed_slice());
    let loaded_groups: Arc<[DenseLoadedGroup]> =
        Arc::from(loaded_groups.to_vec().into_boxed_slice());
    let projection = Arc::new(projection);
    let out_addr = out.as_mut_ptr() as usize;
    let out_len = out.len();
    let job_count = compute.worker_count().min(groups.len()).max(1);
    let groups_per_job = groups.len().div_ceil(job_count);
    let mut jobs = Vec::with_capacity(job_count);

    for group_start in (0..groups.len()).step_by(groups_per_job) {
        let group_end = (group_start + groups_per_job).min(groups.len());
        let segments = Arc::clone(&segments);
        let groups = Arc::clone(&groups);
        let loaded_groups = Arc::clone(&loaded_groups);
        let projection = Arc::clone(&projection);
        let job: ComputeJob = Box::new(move || {
            scatter_dense_group_range_checked::<T>(
                dataset_num_genes,
                output_genes,
                segments.as_ref(),
                groups.as_ref(),
                loaded_groups.as_ref(),
                projection.as_ref().as_ref(),
                group_start,
                group_end,
                src_dtype,
                out_addr,
                out_len,
            )
        });
        jobs.push(job);
    }

    compute.run_jobs(jobs)?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_dense_group_range_checked<T: DataValue>(
    dataset_num_genes: usize,
    output_genes: usize,
    segments: &[DenseSegment],
    groups: &[DenseReadGroup],
    loaded_groups: &[DenseLoadedGroup],
    projection: Option<&CompiledGeneProjection>,
    group_start: usize,
    group_end: usize,
    src_dtype: DType,
    out_addr: usize,
    out_len: usize,
) -> DataBankResult<()> {
    for group_index in group_start..group_end {
        let group = groups.get(group_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("dense group index is invalid".to_string())
        })?;
        let loaded = loaded_groups.get(group_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("dense loaded group index is invalid".to_string())
        })?;
        for part in &group.parts {
            let segment = segments.get(part.segment_index).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("dense group segment index is invalid".to_string())
            })?;
            let bytes = dense_loaded_part_bytes(segment, part, loaded)?;
            if let Some(projection) = projection {
                scatter_dense_segment_to_output_projected::<T>(
                    dataset_num_genes,
                    output_genes,
                    segment,
                    bytes,
                    src_dtype,
                    projection,
                    out_addr,
                    out_len,
                )?;
            } else {
                scatter_dense_segment_to_output::<T>(
                    output_genes,
                    segment,
                    bytes,
                    src_dtype,
                    out_addr,
                    out_len,
                )?;
            }
        }
    }
    Ok(())
}

pub(crate) fn dense_loaded_part_bytes<'a>(
    segment: &DenseSegment,
    part: &DenseGroupPart,
    loaded: &'a DenseLoadedGroup,
) -> DataBankResult<&'a [u8]> {
    match loaded {
        DenseLoadedGroup::Packed(bytes) => {
            let end = part.group_offset.checked_add(part.bytes).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("dense group byte offset overflow".to_string())
            })?;
            if end > bytes.len() {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "dense group decoded length is {}, expected at least {end}",
                    bytes.len()
                )));
            }
            Ok(&bytes[part.group_offset..end])
        }
        DenseLoadedGroup::DecodedSource(bytes) => {
            if segment.source.len() != part.bytes {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "dense group part length is {}, source length is {}",
                    part.bytes,
                    segment.source.len()
                )));
            }
            if segment.source.end > bytes.len() {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "dense chunk decoded length is {}, expected at least {}",
                    bytes.len(),
                    segment.source.end
                )));
            }
            Ok(&bytes[segment.source.start..segment.source.end])
        }
    }
}

pub(crate) fn scatter_dense_segment_to_output<T: DataValue>(
    output_genes: usize,
    segment: &DenseSegment,
    bytes: &[u8],
    src_dtype: DType,
    out_addr: usize,
    out_len: usize,
) -> DataBankResult<()> {
    let expected_bytes = dense_segment_bytes(segment, src_dtype)?;
    if bytes.len() != expected_bytes {
        return Err(DataBankError::BufferSizeMismatch {
            expected: expected_bytes,
            actual: bytes.len(),
        });
    }
    let dst_start = segment
        .output_row
        .checked_mul(output_genes)
        .and_then(|base| base.checked_add(segment.output_col_start))
        .ok_or_else(|| DataBankError::InvalidConfig("output offset overflow".to_string()))?;
    let dst_end = dst_start
        .checked_add(segment.output_cols)
        .ok_or_else(|| DataBankError::InvalidConfig("output offset overflow".to_string()))?;
    if dst_end > out_len {
        return Err(DataBankError::BufferSizeMismatch {
            expected: dst_end,
            actual: out_len,
        });
    }
    // SAFETY: dense batch planning partitions every output row by disjoint
    // gene ranges. Group-parallel jobs therefore write non-overlapping value
    // slots even when they target the same output row.  `dst_start..dst_end`
    // is within `out_len`, derived from the caller-owned `out: &mut [T]` whose
    // address is `out_addr`.
    unsafe {
        let out_ptr = out_addr as *mut T;
        let dst = std::slice::from_raw_parts_mut(out_ptr.add(dst_start), segment.output_cols);
        T::cast_slice_from(bytes, src_dtype, dst)?;
    }
    Ok(())
}

pub(crate) fn scatter_dense_segment_to_output_projected<T: DataValue>(
    dataset_num_genes: usize,
    output_genes: usize,
    segment: &DenseSegment,
    bytes: &[u8],
    src_dtype: DType,
    projection: &CompiledGeneProjection,
    out_addr: usize,
    out_len: usize,
) -> DataBankResult<()> {
    let expected_bytes = dense_segment_bytes(segment, src_dtype)?;
    if bytes.len() != expected_bytes {
        return Err(DataBankError::BufferSizeMismatch {
            expected: expected_bytes,
            actual: bytes.len(),
        });
    }
    let source_end = segment
        .output_col_start
        .checked_add(segment.output_cols)
        .ok_or_else(|| DataBankError::InvalidConfig("dense source column overflow".to_string()))?;
    if source_end > dataset_num_genes {
        return Err(DataBankError::InvalidArrayMeta(
            "dense segment exceeds gene count".to_string(),
        ));
    }
    let row_base = segment
        .output_row
        .checked_mul(output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("dense output row overflow".to_string()))?;
    let row_end = row_base
        .checked_add(output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("dense output row end overflow".to_string()))?;
    if row_end > out_len {
        return Err(DataBankError::BufferSizeMismatch {
            expected: row_end,
            actual: out_len,
        });
    }

    let src_size = src_dtype.item_size();
    let out_ptr = out_addr as *mut T;
    if let Some(output_col_start) =
        projection.contiguous_output_for_source_run(segment.output_col_start, segment.output_cols)
    {
        let dst_start = row_base.checked_add(output_col_start).ok_or_else(|| {
            DataBankError::InvalidConfig("dense output offset overflow".to_string())
        })?;
        let dst_end = dst_start
            .checked_add(segment.output_cols)
            .ok_or_else(|| DataBankError::InvalidConfig("dense output end overflow".to_string()))?;
        if dst_end > row_end {
            return Err(DataBankError::BufferSizeMismatch {
                expected: dst_end,
                actual: out_len,
            });
        }
        // SAFETY: the projected source run maps to the same contiguous output
        // order, and group-parallel jobs write disjoint output columns.
        unsafe {
            let dst = std::slice::from_raw_parts_mut(out_ptr.add(dst_start), segment.output_cols);
            T::cast_slice_from(bytes, src_dtype, dst)?;
        }
        return Ok(());
    }
    for local_col in 0..segment.output_cols {
        let source_col = segment.output_col_start + local_col;
        let Some(output_col) = projection.output_for_source(source_col) else {
            continue;
        };
        if output_col >= output_genes {
            return Err(DataBankError::GeneIndexOutOfRange {
                gene: output_col,
                num_genes: output_genes,
            });
        }
        let src_elem = &bytes[local_col * src_size..(local_col + 1) * src_size];
        // SAFETY: requested gene projection rejects duplicates, so source
        // ranges from different groups map to disjoint output columns.
        unsafe {
            let dst = std::slice::from_raw_parts_mut(out_ptr.add(row_base + output_col), 1);
            T::cast_slice_from(src_elem, src_dtype, dst)?;
        }
    }
    Ok(())
}
