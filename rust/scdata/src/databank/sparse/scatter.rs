#![allow(clippy::too_many_arguments)]

use super::*;

pub(crate) fn load_sparse_data_group_bytes(
    access: &AccessHandle,
    access_config: &ScheduledAccessConfig,
    plan: &SparseBatchPlan,
) -> DataBankResult<Vec<Arc<[u8]>>> {
    if plan
        .data_groups
        .iter()
        .all(|group| matches!(group.source, SparseGroupSource::AccessItem(_)))
    {
        let mut scheduled = access.scheduled(
            plan.data_groups.iter().map(file_sparse_group_access_item),
            *access_config,
        )?;
        let mut out = Vec::with_capacity(plan.data_groups.len());
        for group in &plan.data_groups {
            let bytes = scheduled.next().ok_or_else(|| {
                DataBankError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "scheduled CSR data access ended early",
                ))
            })??;
            if bytes.len() != group.bytes {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "CSR data group decoded length is {}, expected {}",
                    bytes.len(),
                    group.bytes
                )));
            }
            out.push(Arc::from(bytes.into_boxed_slice()));
        }
        if scheduled.next().is_some() {
            return Err(DataBankError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "scheduled CSR data access returned extra output",
            )));
        }
        Ok(out)
    } else {
        let mut out = Vec::with_capacity(plan.data_groups.len());
        for group in &plan.data_groups {
            let bytes = load_sparse_group(access, group)?;
            if bytes.len() != group.bytes {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "CSR data group decoded length is {}, expected {}",
                    bytes.len(),
                    group.bytes
                )));
            }
            out.push(Arc::from(bytes.into_boxed_slice()));
        }
        Ok(out)
    }
}

pub(crate) fn load_sparse_selected_data_group_bytes(
    access: &AccessHandle,
    access_config: &ScheduledAccessConfig,
    plan: &SparseBatchPlan,
    selected_groups: &[usize],
    cancel: Option<&Arc<PrefetchCancel>>,
) -> DataBankResult<Vec<Arc<[u8]>>> {
    if selected_groups.iter().all(|&group_index| {
        matches!(
            plan.data_groups[group_index].source,
            SparseGroupSource::AccessItem(_)
        )
    }) {
        let mut scheduled = schedule_sparse_selected_file_groups(
            access,
            access_config,
            plan,
            selected_groups,
            cancel,
        )?
        .expect("file-backed selected groups should create a scheduled reader");
        let mut out = Vec::with_capacity(selected_groups.len());
        for &group_index in selected_groups {
            if let Some(cancel) = cancel {
                if cancel.is_cancelled() {
                    return Err(DataBankError::PrefetchCancelled);
                }
            }
            let group = &plan.data_groups[group_index];
            let bytes = next_scheduled_sparse_group_bytes(&mut scheduled)?;
            if bytes.len() != group.bytes {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "CSR data group decoded length is {}, expected {}",
                    bytes.len(),
                    group.bytes
                )));
            }
            out.push(Arc::from(bytes.into_boxed_slice()));
        }
        finish_scheduled_sparse_group_bytes(Some(scheduled))?;
        Ok(out)
    } else {
        let mut scheduled = schedule_sparse_selected_file_groups(
            access,
            access_config,
            plan,
            selected_groups,
            cancel,
        )?;
        let mut out = Vec::with_capacity(selected_groups.len());
        for &group_index in selected_groups {
            if let Some(cancel) = cancel {
                if cancel.is_cancelled() {
                    return Err(DataBankError::PrefetchCancelled);
                }
            }
            let group = &plan.data_groups[group_index];
            let bytes = match &group.source {
                SparseGroupSource::AccessItem(_) => {
                    let scheduled = scheduled.as_mut().ok_or_else(|| {
                        DataBankError::InvalidArrayMeta(
                            "CSR file group missing scheduled reader".to_string(),
                        )
                    })?;
                    next_scheduled_sparse_group_bytes(scheduled)?
                }
                SparseGroupSource::Memory { .. } => load_sparse_group(access, group)?,
            };
            if bytes.len() != group.bytes {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "CSR data group decoded length is {}, expected {}",
                    bytes.len(),
                    group.bytes
                )));
            }
            out.push(Arc::from(bytes.into_boxed_slice()));
        }
        finish_scheduled_sparse_group_bytes(scheduled)?;
        Ok(out)
    }
}

pub(crate) fn load_sparse_data_groups_and_scatter_projected_checked<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    index_bytes: Vec<u8>,
    gene_axis: &GeneAxisPlan,
    cancel: Option<&Arc<PrefetchCancel>>,
    active_rows: Option<usize>,
    out: &mut [T],
) -> DataBankResult<()> {
    if plan.data_groups.is_empty() {
        return Ok(());
    }

    // Box the index buffer once; both the parallel and serial scatter paths
    // borrow from this shared `Arc<[u8]>` instead of cloning the whole buffer.
    let index_bytes: Arc<[u8]> = Arc::from(index_bytes.into_boxed_slice());

    let mut selected_groups = Vec::with_capacity(plan.data_groups.len());
    for (group_index, group) in plan.data_groups.iter().enumerate() {
        if sparse_data_group_has_selected_values(
            dataset,
            &plan.data_pieces,
            group,
            index_bytes.as_ref(),
            gene_axis,
        )? {
            selected_groups.push(group_index);
        }
    }
    if selected_groups.is_empty() {
        return Ok(());
    }

    let output_genes = gene_axis.output_genes(dataset.num_genes);
    let output_rows = row_count_for_width(out.len(), output_genes);
    let parallel_rows = active_rows.unwrap_or(output_rows);
    let parallel_bytes = parallel_rows
        .checked_mul(output_genes)
        .and_then(|values| values.checked_mul(mem::size_of::<T>()))
        .ok_or_else(|| {
            DataBankError::InvalidConfig("sparse active output byte length overflow".to_string())
        })?;
    if compute.should_parallelize(parallel_rows, parallel_bytes) {
        let data_group_bytes = load_sparse_selected_data_group_bytes(
            access,
            access_config,
            plan,
            &selected_groups,
            cancel,
        )?;
        if try_scatter_sparse_rows_parallel_checked_with_group_indices(
            compute,
            dataset,
            plan,
            Arc::clone(&index_bytes),
            data_group_bytes,
            output_genes,
            gene_axis.projection().cloned(),
            Some(selected_groups.clone()),
            out,
        )? {
            return Ok(());
        }
    }

    if selected_groups.iter().all(|&group_index| {
        matches!(
            plan.data_groups[group_index].source,
            SparseGroupSource::AccessItem(_)
        )
    }) {
        let mut scheduled = schedule_sparse_selected_file_groups(
            access,
            access_config,
            plan,
            &selected_groups,
            cancel,
        )?
        .expect("file-backed selected groups should create a scheduled reader");
        for &group_index in &selected_groups {
            if let Some(cancel) = cancel {
                if cancel.is_cancelled() {
                    return Err(DataBankError::PrefetchCancelled);
                }
            }
            let group = &plan.data_groups[group_index];
            let bytes = next_scheduled_sparse_group_bytes(&mut scheduled)?;
            scatter_sparse_data_group_projected_checked(
                dataset,
                &plan.data_pieces,
                group,
                index_bytes.as_ref(),
                &bytes,
                gene_axis,
                out,
            )?;
        }
        finish_scheduled_sparse_group_bytes(Some(scheduled))?;
    } else {
        let mut scheduled = schedule_sparse_selected_file_groups(
            access,
            access_config,
            plan,
            &selected_groups,
            cancel,
        )?;
        for &group_index in &selected_groups {
            let group = &plan.data_groups[group_index];
            if let Some(cancel) = cancel {
                if cancel.is_cancelled() {
                    return Err(DataBankError::PrefetchCancelled);
                }
            }
            if try_scatter_sparse_memory_identity_data_group_projected_checked(
                dataset,
                &plan.data_pieces,
                group,
                index_bytes.as_ref(),
                gene_axis,
                out,
            )? {
                continue;
            }
            let bytes = match &group.source {
                SparseGroupSource::AccessItem(_) => {
                    let scheduled = scheduled.as_mut().ok_or_else(|| {
                        DataBankError::InvalidArrayMeta(
                            "CSR file group missing scheduled reader".to_string(),
                        )
                    })?;
                    next_scheduled_sparse_group_bytes(scheduled)?
                }
                SparseGroupSource::Memory { .. } => load_sparse_group(access, group)?,
            };
            scatter_sparse_data_group_projected_checked(
                dataset,
                &plan.data_pieces,
                group,
                index_bytes.as_ref(),
                &bytes,
                gene_axis,
                out,
            )?;
        }
        finish_scheduled_sparse_group_bytes(scheduled)?;
    }
    Ok(())
}

pub(crate) unsafe fn load_sparse_data_groups_and_scatter_unchecked<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    index_bytes: Vec<u8>,
    out: &mut [T],
) -> DataBankResult<()> {
    if plan.data_groups.is_empty() {
        return Ok(());
    }
    // Box once; both scatter paths borrow from this shared arc.
    let index_bytes: Arc<[u8]> = Arc::from(index_bytes.into_boxed_slice());
    let output_rows = row_count_for_width(out.len(), dataset.num_genes);
    let output_bytes = out.len().checked_mul(mem::size_of::<T>()).ok_or_else(|| {
        DataBankError::InvalidConfig("sparse output byte length overflow".to_string())
    })?;
    if compute.should_parallelize(output_rows, output_bytes) {
        let data_group_bytes = load_sparse_data_group_bytes(access, access_config, plan)?;
        if unsafe {
            try_scatter_sparse_rows_parallel_unchecked(
                compute,
                dataset,
                plan,
                Arc::clone(&index_bytes),
                data_group_bytes,
                out,
            )
        }? {
            return Ok(());
        }
    }
    if plan
        .data_groups
        .iter()
        .all(|group| matches!(group.source, SparseGroupSource::AccessItem(_)))
    {
        let mut scheduled = access.scheduled(
            plan.data_groups.iter().map(file_sparse_group_access_item),
            *access_config,
        )?;
        for group in &plan.data_groups {
            let bytes = scheduled.next().ok_or_else(|| {
                DataBankError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "scheduled CSR data access ended early",
                ))
            })??;
            unsafe {
                scatter_sparse_data_group_unchecked(
                    dataset,
                    &plan.data_pieces,
                    group,
                    index_bytes.as_ref(),
                    &bytes,
                    out,
                );
            }
        }
        if scheduled.next().is_some() {
            return Err(DataBankError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "scheduled CSR data access returned extra output",
            )));
        }
    } else {
        for group in &plan.data_groups {
            if unsafe {
                try_scatter_sparse_memory_identity_data_group_unchecked(
                    dataset,
                    &plan.data_pieces,
                    group,
                    index_bytes.as_ref(),
                    out,
                )
            }? {
                continue;
            }
            let bytes = load_sparse_group(access, group)?;
            unsafe {
                scatter_sparse_data_group_unchecked(
                    dataset,
                    &plan.data_pieces,
                    group,
                    index_bytes.as_ref(),
                    &bytes,
                    out,
                );
            }
        }
    }
    Ok(())
}

pub(crate) fn scatter_sparse_data_group_checked<T: DataValue>(
    dataset: &SparseCsrDataset,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    data_bytes: &[u8],
    out: &mut [T],
) -> DataBankResult<()> {
    if data_bytes.len() != group.bytes {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "CSR data group decoded length is {}, expected {}",
            data_bytes.len(),
            group.bytes
        )));
    }

    match dataset.index_dtype {
        DType::U32 => scatter_sparse_data_group_checked_typed::<T, u32>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            data_bytes,
            dataset.data.dtype,
            out,
        ),
        DType::I32 => scatter_sparse_data_group_checked_typed::<T, i32>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            data_bytes,
            dataset.data.dtype,
            out,
        ),
        DType::U64 => scatter_sparse_data_group_checked_typed::<T, u64>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            data_bytes,
            dataset.data.dtype,
            out,
        ),
        DType::I64 => scatter_sparse_data_group_checked_typed::<T, i64>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            data_bytes,
            dataset.data.dtype,
            out,
        ),
        dtype => Err(DataBankError::UnsupportedDType {
            dtype,
            context: "CSR indices",
        }),
    }
}

pub(crate) fn try_scatter_sparse_rows_parallel_checked<T: DataValue>(
    compute: &DataBankComputePool,
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    index_bytes: Arc<[u8]>,
    data_group_bytes: Vec<Arc<[u8]>>,
    output_genes: usize,
    projection: Option<CompiledGeneProjection>,
    out: &mut [T],
) -> DataBankResult<bool> {
    try_scatter_sparse_rows_parallel_checked_with_group_indices(
        compute,
        dataset,
        plan,
        index_bytes,
        data_group_bytes,
        output_genes,
        projection,
        None,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_scatter_sparse_rows_parallel_checked_with_group_indices<T: DataValue>(
    compute: &DataBankComputePool,
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    index_bytes: Arc<[u8]>,
    data_group_bytes: Vec<Arc<[u8]>>,
    output_genes: usize,
    projection: Option<CompiledGeneProjection>,
    loaded_group_indices: Option<Vec<usize>>,
    out: &mut [T],
) -> DataBankResult<bool> {
    if output_genes == 0 || plan.data_pieces.is_empty() {
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
        DataBankError::InvalidConfig("sparse output byte length overflow".to_string())
    })?;
    if !compute.should_parallelize(output_rows, output_bytes) {
        return Ok(false);
    }
    let skip_unloaded_groups = loaded_group_indices.is_some() && projection.is_some();
    let group_to_loaded = sparse_loaded_group_indices(
        plan,
        &data_group_bytes,
        loaded_group_indices.as_deref(),
        skip_unloaded_groups,
    )?;

    let row_pieces = sparse_row_piece_ranges(plan, output_rows)?;
    let piece_groups = sparse_piece_group_indices(plan)?;
    let pieces: Arc<[SparseReadPiece]> = Arc::from(plan.data_pieces.clone().into_boxed_slice());
    let piece_groups: Arc<[usize]> = Arc::from(piece_groups.into_boxed_slice());
    let group_to_loaded: Arc<[usize]> = Arc::from(group_to_loaded.into_boxed_slice());
    let row_pieces = Arc::new(row_pieces);
    let data_group_bytes: Arc<[Arc<[u8]>]> = Arc::from(data_group_bytes.into_boxed_slice());
    let projection = Arc::new(projection);
    let out_addr = out.as_mut_ptr() as usize;
    let out_len = out.len();
    let num_genes = dataset.num_genes;
    let index_dtype = dataset.index_dtype;
    let src_dtype = dataset.data.dtype;
    let job_count = compute.worker_count().min(output_rows).max(1);
    let rows_per_job = output_rows.div_ceil(job_count);
    let mut jobs = Vec::with_capacity(job_count);

    for row_start in (0..output_rows).step_by(rows_per_job) {
        let row_end = (row_start + rows_per_job).min(output_rows);
        let pieces = Arc::clone(&pieces);
        let piece_groups = Arc::clone(&piece_groups);
        let group_to_loaded = Arc::clone(&group_to_loaded);
        let row_pieces = Arc::clone(&row_pieces);
        let data_group_bytes = Arc::clone(&data_group_bytes);
        let index_bytes = Arc::clone(&index_bytes);
        let projection = Arc::clone(&projection);
        let job: ComputeJob = Box::new(move || match index_dtype {
            DType::U32 => scatter_sparse_row_range_checked_typed::<T, u32>(
                num_genes,
                output_genes,
                pieces.as_ref(),
                piece_groups.as_ref(),
                group_to_loaded.as_ref(),
                row_pieces.as_ref(),
                data_group_bytes.as_ref(),
                index_bytes.as_ref(),
                src_dtype,
                projection.as_ref().as_ref(),
                skip_unloaded_groups,
                row_start,
                row_end,
                out_addr,
                out_len,
            ),
            DType::I32 => scatter_sparse_row_range_checked_typed::<T, i32>(
                num_genes,
                output_genes,
                pieces.as_ref(),
                piece_groups.as_ref(),
                group_to_loaded.as_ref(),
                row_pieces.as_ref(),
                data_group_bytes.as_ref(),
                index_bytes.as_ref(),
                src_dtype,
                projection.as_ref().as_ref(),
                skip_unloaded_groups,
                row_start,
                row_end,
                out_addr,
                out_len,
            ),
            DType::U64 => scatter_sparse_row_range_checked_typed::<T, u64>(
                num_genes,
                output_genes,
                pieces.as_ref(),
                piece_groups.as_ref(),
                group_to_loaded.as_ref(),
                row_pieces.as_ref(),
                data_group_bytes.as_ref(),
                index_bytes.as_ref(),
                src_dtype,
                projection.as_ref().as_ref(),
                skip_unloaded_groups,
                row_start,
                row_end,
                out_addr,
                out_len,
            ),
            DType::I64 => scatter_sparse_row_range_checked_typed::<T, i64>(
                num_genes,
                output_genes,
                pieces.as_ref(),
                piece_groups.as_ref(),
                group_to_loaded.as_ref(),
                row_pieces.as_ref(),
                data_group_bytes.as_ref(),
                index_bytes.as_ref(),
                src_dtype,
                projection.as_ref().as_ref(),
                skip_unloaded_groups,
                row_start,
                row_end,
                out_addr,
                out_len,
            ),
            dtype => Err(DataBankError::UnsupportedDType {
                dtype,
                context: "CSR indices",
            }),
        });
        jobs.push(job);
    }

    compute.run_jobs(jobs)?;
    Ok(true)
}

pub(crate) unsafe fn try_scatter_sparse_rows_parallel_unchecked<T: DataValue>(
    compute: &DataBankComputePool,
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    index_bytes: Arc<[u8]>,
    data_group_bytes: Vec<Arc<[u8]>>,
    out: &mut [T],
) -> DataBankResult<bool> {
    let output_genes = dataset.num_genes;
    if output_genes == 0 || plan.data_pieces.is_empty() {
        return Ok(true);
    }
    let output_rows = out.len() / output_genes;
    let output_bytes = out.len().checked_mul(mem::size_of::<T>()).ok_or_else(|| {
        DataBankError::InvalidConfig("sparse output byte length overflow".to_string())
    })?;
    if !compute.should_parallelize(output_rows, output_bytes) {
        return Ok(false);
    }
    debug_assert_eq!(data_group_bytes.len(), plan.data_groups.len());

    let row_pieces = sparse_row_piece_ranges(plan, output_rows)?;
    let piece_groups = sparse_piece_group_indices(plan)?;
    let pieces: Arc<[SparseReadPiece]> = Arc::from(plan.data_pieces.clone().into_boxed_slice());
    let piece_groups: Arc<[usize]> = Arc::from(piece_groups.into_boxed_slice());
    let row_pieces = Arc::new(row_pieces);
    let data_group_bytes: Arc<[Arc<[u8]>]> = Arc::from(data_group_bytes.into_boxed_slice());
    let out_addr = out.as_mut_ptr() as usize;
    let index_dtype = dataset.index_dtype;
    let job_count = compute.worker_count().min(output_rows).max(1);
    let rows_per_job = output_rows.div_ceil(job_count);
    let mut jobs = Vec::with_capacity(job_count);

    for row_start in (0..output_rows).step_by(rows_per_job) {
        let row_end = (row_start + rows_per_job).min(output_rows);
        let pieces = Arc::clone(&pieces);
        let piece_groups = Arc::clone(&piece_groups);
        let row_pieces = Arc::clone(&row_pieces);
        let data_group_bytes = Arc::clone(&data_group_bytes);
        let index_bytes = Arc::clone(&index_bytes);
        let job: ComputeJob = Box::new(move || match index_dtype {
            DType::U32 => unsafe {
                scatter_sparse_row_range_unchecked_typed::<T, u32>(
                    output_genes,
                    pieces.as_ref(),
                    piece_groups.as_ref(),
                    row_pieces.as_ref(),
                    data_group_bytes.as_ref(),
                    index_bytes.as_ref(),
                    row_start,
                    row_end,
                    out_addr,
                );
                Ok(())
            },
            DType::I32 => unsafe {
                scatter_sparse_row_range_unchecked_typed::<T, i32>(
                    output_genes,
                    pieces.as_ref(),
                    piece_groups.as_ref(),
                    row_pieces.as_ref(),
                    data_group_bytes.as_ref(),
                    index_bytes.as_ref(),
                    row_start,
                    row_end,
                    out_addr,
                );
                Ok(())
            },
            DType::U64 => unsafe {
                scatter_sparse_row_range_unchecked_typed::<T, u64>(
                    output_genes,
                    pieces.as_ref(),
                    piece_groups.as_ref(),
                    row_pieces.as_ref(),
                    data_group_bytes.as_ref(),
                    index_bytes.as_ref(),
                    row_start,
                    row_end,
                    out_addr,
                );
                Ok(())
            },
            DType::I64 => unsafe {
                scatter_sparse_row_range_unchecked_typed::<T, i64>(
                    output_genes,
                    pieces.as_ref(),
                    piece_groups.as_ref(),
                    row_pieces.as_ref(),
                    data_group_bytes.as_ref(),
                    index_bytes.as_ref(),
                    row_start,
                    row_end,
                    out_addr,
                );
                Ok(())
            },
            _ => unreachable!("CSR index dtype was validated during registration"),
        });
        jobs.push(job);
    }

    compute.run_jobs(jobs)?;
    Ok(true)
}

pub(crate) fn sparse_piece_group_indices(plan: &SparseBatchPlan) -> DataBankResult<Vec<usize>> {
    let mut piece_groups = vec![usize::MAX; plan.data_pieces.len()];
    for (group_index, group) in plan.data_groups.iter().enumerate() {
        for &piece_index in &group.parts {
            let Some(slot) = piece_groups.get_mut(piece_index) else {
                return Err(DataBankError::InvalidArrayMeta(
                    "CSR data group piece index is invalid".to_string(),
                ));
            };
            *slot = group_index;
        }
    }
    if piece_groups.contains(&usize::MAX) {
        return Err(DataBankError::InvalidArrayMeta(
            "CSR data piece is missing its group".to_string(),
        ));
    }
    Ok(piece_groups)
}

pub(crate) fn sparse_loaded_group_indices(
    plan: &SparseBatchPlan,
    data_group_bytes: &[Arc<[u8]>],
    loaded_group_indices: Option<&[usize]>,
    allow_unloaded_groups: bool,
) -> DataBankResult<Vec<usize>> {
    let mut group_to_loaded = vec![usize::MAX; plan.data_groups.len()];
    match loaded_group_indices {
        Some(indices) => {
            if indices.len() != data_group_bytes.len() {
                return Err(DataBankError::InvalidArrayMeta(
                    "CSR loaded group index count mismatch".to_string(),
                ));
            }
            for (loaded_index, &group_index) in indices.iter().enumerate() {
                let Some(group) = plan.data_groups.get(group_index) else {
                    return Err(DataBankError::InvalidArrayMeta(
                        "CSR loaded group index is out of range".to_string(),
                    ));
                };
                if group_to_loaded[group_index] != usize::MAX {
                    return Err(DataBankError::InvalidArrayMeta(
                        "CSR loaded group index is duplicated".to_string(),
                    ));
                }
                let bytes = &data_group_bytes[loaded_index];
                if bytes.len() != group.bytes {
                    return Err(DataBankError::InvalidArrayMeta(format!(
                        "CSR data group decoded length is {}, expected {}",
                        bytes.len(),
                        group.bytes
                    )));
                }
                group_to_loaded[group_index] = loaded_index;
            }
            if !allow_unloaded_groups && group_to_loaded.contains(&usize::MAX) {
                return Err(DataBankError::InvalidArrayMeta(
                    "CSR data group bytes are incomplete".to_string(),
                ));
            }
        }
        None => {
            if data_group_bytes.len() != plan.data_groups.len() {
                return Err(DataBankError::InvalidArrayMeta(
                    "CSR data group byte count mismatch".to_string(),
                ));
            }
            for (group_index, (group, bytes)) in plan
                .data_groups
                .iter()
                .zip(data_group_bytes.iter())
                .enumerate()
            {
                if bytes.len() != group.bytes {
                    return Err(DataBankError::InvalidArrayMeta(format!(
                        "CSR data group decoded length is {}, expected {}",
                        bytes.len(),
                        group.bytes
                    )));
                }
                group_to_loaded[group_index] = group_index;
            }
        }
    }
    Ok(group_to_loaded)
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SparseRowPieceRange {
    start: usize,
    end: usize,
}

pub(crate) fn sparse_row_piece_ranges(
    plan: &SparseBatchPlan,
    output_rows: usize,
) -> DataBankResult<Vec<SparseRowPieceRange>> {
    let mut row_pieces = vec![SparseRowPieceRange::default(); output_rows];
    let mut piece_index = 0usize;
    for (output_row, row_range) in row_pieces.iter_mut().enumerate() {
        let start = piece_index;
        while let Some(piece) = plan.data_pieces.get(piece_index) {
            if piece.output_row < output_row {
                return Err(DataBankError::InvalidArrayMeta(
                    "CSR data pieces are not ordered by output row".to_string(),
                ));
            }
            if piece.output_row != output_row {
                break;
            }
            piece_index += 1;
        }
        *row_range = SparseRowPieceRange {
            start,
            end: piece_index,
        };
    }
    if let Some(piece) = plan.data_pieces.get(piece_index) {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "CSR data piece output row {} is out of range for {output_rows} rows",
            piece.output_row
        )));
    }
    Ok(row_pieces)
}

pub(crate) fn scatter_sparse_row_range_checked_typed<T, I>(
    num_genes: usize,
    output_genes: usize,
    pieces: &[SparseReadPiece],
    piece_groups: &[usize],
    group_to_loaded: &[usize],
    row_pieces: &[SparseRowPieceRange],
    data_group_bytes: &[Arc<[u8]>],
    index_bytes: &[u8],
    src_dtype: DType,
    projection: Option<&CompiledGeneProjection>,
    skip_unloaded_groups: bool,
    row_start: usize,
    row_end: usize,
    out_addr: usize,
    out_len: usize,
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    let out_ptr = out_addr as *mut T;

    for output_row in row_start..row_end {
        let row_piece_range = row_pieces.get(output_row).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR row piece index is out of range".to_string())
        })?;
        let row_base = output_row.checked_mul(output_genes).ok_or_else(|| {
            DataBankError::InvalidConfig("sparse output row overflow".to_string())
        })?;
        let row_end_offset = row_base.checked_add(output_genes).ok_or_else(|| {
            DataBankError::InvalidConfig("sparse output row end overflow".to_string())
        })?;
        if row_end_offset > out_len {
            return Err(DataBankError::BufferSizeMismatch {
                expected: row_end_offset,
                actual: out_len,
            });
        }
        // SAFETY: the caller partitions jobs by non-overlapping output row
        // ranges. This task only creates row slices inside its assigned range.
        let row_out =
            unsafe { std::slice::from_raw_parts_mut(out_ptr.add(row_base), output_genes) };
        for piece_index in row_piece_range.start..row_piece_range.end {
            let piece = pieces.get(piece_index).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("CSR row piece index is invalid".to_string())
            })?;
            if piece.output_row != output_row {
                return Err(DataBankError::InvalidArrayMeta(
                    "CSR row piece belongs to a different output row".to_string(),
                ));
            }
            let group_index = *piece_groups.get(piece_index).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("CSR piece group index is invalid".to_string())
            })?;
            let loaded_index = *group_to_loaded.get(group_index).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("CSR loaded group map index is invalid".to_string())
            })?;
            if loaded_index == usize::MAX {
                if skip_unloaded_groups {
                    continue;
                }
                return Err(DataBankError::InvalidArrayMeta(
                    "CSR data group bytes are missing".to_string(),
                ));
            }
            let data_bytes = data_group_bytes.get(loaded_index).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("CSR data group index is invalid".to_string())
            })?;
            let data_end = piece.group_offset.checked_add(piece.bytes).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("CSR data group offset overflow".to_string())
            })?;
            let index_len = piece.elements.checked_mul(index_size).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("CSR index slice length overflow".to_string())
            })?;
            let index_end = piece.index_offset.checked_add(index_len).ok_or_else(|| {
                DataBankError::InvalidArrayMeta("CSR index slice offset overflow".to_string())
            })?;
            if data_end > data_bytes.len() || index_end > index_bytes.len() {
                return Err(DataBankError::InvalidArrayMeta(
                    "CSR row scatter is out of range".to_string(),
                ));
            }
            scatter_sparse_values_to_row_checked_typed::<T, I>(
                num_genes,
                projection,
                piece.elements,
                &index_bytes[piece.index_offset..index_end],
                &data_bytes[piece.group_offset..data_end],
                src_dtype,
                row_out,
            )?;
        }
    }
    Ok(())
}

pub(crate) fn scatter_sparse_values_to_row_checked_typed<T, I>(
    num_genes: usize,
    projection: Option<&CompiledGeneProjection>,
    elements: usize,
    index_bytes: &[u8],
    data_bytes: &[u8],
    src_dtype: DType,
    row_out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let value_size = src_dtype.item_size();
    let expected_data_bytes = elements.checked_mul(value_size).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR data byte length overflow".to_string())
    })?;
    if data_bytes.len() != expected_data_bytes {
        return Err(DataBankError::BufferSizeMismatch {
            expected: expected_data_bytes,
            actual: data_bytes.len(),
        });
    }
    let index_size = mem::size_of::<I>();
    let expected_index_bytes = elements.checked_mul(index_size).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR index byte length overflow".to_string())
    })?;
    if index_bytes.len() != expected_index_bytes {
        return Err(DataBankError::BufferSizeMismatch {
            expected: expected_index_bytes,
            actual: index_bytes.len(),
        });
    }

    let index_ptr = index_bytes.as_ptr().cast::<I>();
    for nz in 0..elements {
        let gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) }.checked_gene()?;
        if gene >= num_genes {
            return Err(DataBankError::GeneIndexOutOfRange { gene, num_genes });
        }
        let Some(output_col) = projection
            .and_then(|projection| projection.output_for_source(gene))
            .or_else(|| projection.is_none().then_some(gene))
        else {
            continue;
        };
        if output_col >= row_out.len() {
            return Err(DataBankError::GeneIndexOutOfRange {
                gene: output_col,
                num_genes: row_out.len(),
            });
        }
        let src_start = nz * value_size;
        T::cast_slice_from(
            &data_bytes[src_start..src_start + value_size],
            src_dtype,
            &mut row_out[output_col..output_col + 1],
        )?;
    }
    Ok(())
}

unsafe fn scatter_sparse_row_range_unchecked_typed<T, I>(
    output_genes: usize,
    pieces: &[SparseReadPiece],
    piece_groups: &[usize],
    row_pieces: &[SparseRowPieceRange],
    data_group_bytes: &[Arc<[u8]>],
    index_bytes: &[u8],
    row_start: usize,
    row_end: usize,
    out_addr: usize,
) where
    T: DataValue,
    I: CsrIndex,
{
    let value_size = mem::size_of::<T>();
    let index_size = mem::size_of::<I>();
    let out_ptr = out_addr as *mut T;

    for output_row in row_start..row_end {
        let row_base = output_row * output_genes;
        // SAFETY: jobs are partitioned by disjoint output row ranges.
        let row_ptr = unsafe { out_ptr.add(row_base) };
        let row_piece_range = unsafe { row_pieces.get_unchecked(output_row) };
        for piece_index in row_piece_range.start..row_piece_range.end {
            let piece = unsafe { pieces.get_unchecked(piece_index) };
            let group_index = unsafe { *piece_groups.get_unchecked(piece_index) };
            let data_bytes = unsafe { data_group_bytes.get_unchecked(group_index) };
            let data_end = piece.group_offset + piece.bytes;
            let index_len = piece.elements * index_size;
            let index_end = piece.index_offset + index_len;
            let piece_indices = unsafe { index_bytes.get_unchecked(piece.index_offset..index_end) };
            let piece_data = unsafe { data_bytes.get_unchecked(piece.group_offset..data_end) };
            unsafe {
                scatter_sparse_values_to_row_unchecked_typed::<T, I>(
                    output_genes,
                    piece.elements,
                    piece_indices,
                    piece_data,
                    row_ptr,
                    value_size,
                );
            }
        }
    }
}

unsafe fn scatter_sparse_values_to_row_unchecked_typed<T, I>(
    output_genes: usize,
    elements: usize,
    index_bytes: &[u8],
    data_bytes: &[u8],
    row_ptr: *mut T,
    value_size: usize,
) where
    T: DataValue,
    I: CsrIndex,
{
    let index_ptr = index_bytes.as_ptr().cast::<I>();
    let data_ptr = data_bytes.as_ptr();
    for nz in 0..elements {
        let gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) }.unchecked_gene();
        debug_assert!(gene < output_genes);
        let value = unsafe { ptr::read_unaligned(data_ptr.add(nz * value_size).cast::<T>()) };
        unsafe { ptr::write(row_ptr.add(gene), value) };
    }
}

pub(crate) fn try_scatter_sparse_memory_identity_data_group_checked<T: DataValue>(
    dataset: &SparseCsrDataset,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    out: &mut [T],
) -> DataBankResult<bool> {
    let SparseGroupSource::Memory {
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

    scatter_sparse_data_group_checked_from_decoded_source(
        dataset,
        pieces,
        group,
        index_bytes,
        bytes.as_ref(),
        out,
    )?;
    Ok(true)
}

pub(crate) fn scatter_sparse_data_group_checked_from_decoded_source<T: DataValue>(
    dataset: &SparseCsrDataset,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    decoded: &[u8],
    out: &mut [T],
) -> DataBankResult<()> {
    match dataset.index_dtype {
        DType::U32 => scatter_sparse_data_group_checked_typed_from_decoded_source::<T, u32>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            decoded,
            dataset.data.dtype,
            out,
        ),
        DType::I32 => scatter_sparse_data_group_checked_typed_from_decoded_source::<T, i32>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            decoded,
            dataset.data.dtype,
            out,
        ),
        DType::U64 => scatter_sparse_data_group_checked_typed_from_decoded_source::<T, u64>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            decoded,
            dataset.data.dtype,
            out,
        ),
        DType::I64 => scatter_sparse_data_group_checked_typed_from_decoded_source::<T, i64>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            decoded,
            dataset.data.dtype,
            out,
        ),
        dtype => Err(DataBankError::UnsupportedDType {
            dtype,
            context: "CSR indices",
        }),
    }
}

pub(crate) fn sparse_data_group_has_selected_values(
    dataset: &SparseCsrDataset,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    gene_axis: &GeneAxisPlan,
) -> DataBankResult<bool> {
    let Some(projection) = gene_axis.projection() else {
        return Ok(true);
    };
    if projection.selected_sources.is_empty() {
        return validate_sparse_data_group_indices(dataset, pieces, group, index_bytes)
            .map(|_| false);
    }

    match dataset.index_dtype {
        DType::U32 => sparse_data_group_has_selected_values_typed::<u32>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            projection,
        ),
        DType::I32 => sparse_data_group_has_selected_values_typed::<i32>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            projection,
        ),
        DType::U64 => sparse_data_group_has_selected_values_typed::<u64>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            projection,
        ),
        DType::I64 => sparse_data_group_has_selected_values_typed::<i64>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
            projection,
        ),
        dtype => Err(DataBankError::UnsupportedDType {
            dtype,
            context: "CSR indices",
        }),
    }
}

pub(crate) fn validate_sparse_data_group_indices(
    dataset: &SparseCsrDataset,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
) -> DataBankResult<()> {
    match dataset.index_dtype {
        DType::U32 => validate_sparse_data_group_indices_typed::<u32>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
        ),
        DType::I32 => validate_sparse_data_group_indices_typed::<i32>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
        ),
        DType::U64 => validate_sparse_data_group_indices_typed::<u64>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
        ),
        DType::I64 => validate_sparse_data_group_indices_typed::<i64>(
            dataset.num_genes,
            pieces,
            group,
            index_bytes,
        ),
        dtype => Err(DataBankError::UnsupportedDType {
            dtype,
            context: "CSR indices",
        }),
    }
}

pub(crate) fn sparse_data_group_has_selected_values_typed<I>(
    num_genes: usize,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    projection: &CompiledGeneProjection,
) -> DataBankResult<bool>
where
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    for &piece_index in &group.parts {
        let piece = pieces.get(piece_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR data group piece index is invalid".to_string())
        })?;
        let index_len = piece.elements.checked_mul(index_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice length overflow".to_string())
        })?;
        let index_end = piece.index_offset.checked_add(index_len).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice offset overflow".to_string())
        })?;
        if index_end > index_bytes.len() {
            return Err(DataBankError::InvalidArrayMeta(
                "CSR index grouped scan is out of range".to_string(),
            ));
        }
        let index_ptr = index_bytes[piece.index_offset..index_end]
            .as_ptr()
            .cast::<I>();
        for nz in 0..piece.elements {
            let gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) }.checked_gene()?;
            if gene >= num_genes {
                return Err(DataBankError::GeneIndexOutOfRange { gene, num_genes });
            }
            if projection.output_for_source(gene).is_some() {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

pub(crate) fn validate_sparse_data_group_indices_typed<I>(
    num_genes: usize,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
) -> DataBankResult<()>
where
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    for &piece_index in &group.parts {
        let piece = pieces.get(piece_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR data group piece index is invalid".to_string())
        })?;
        let index_len = piece.elements.checked_mul(index_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice length overflow".to_string())
        })?;
        let index_end = piece.index_offset.checked_add(index_len).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice offset overflow".to_string())
        })?;
        if index_end > index_bytes.len() {
            return Err(DataBankError::InvalidArrayMeta(
                "CSR index grouped scan is out of range".to_string(),
            ));
        }
        let index_ptr = index_bytes[piece.index_offset..index_end]
            .as_ptr()
            .cast::<I>();
        for nz in 0..piece.elements {
            let gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) }.checked_gene()?;
            if gene >= num_genes {
                return Err(DataBankError::GeneIndexOutOfRange { gene, num_genes });
            }
        }
    }
    Ok(())
}

pub(crate) fn scatter_sparse_data_group_projected_checked<T: DataValue>(
    dataset: &SparseCsrDataset,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    data_bytes: &[u8],
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    if data_bytes.len() != group.bytes {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "CSR data group decoded length is {}, expected {}",
            data_bytes.len(),
            group.bytes
        )));
    }
    let Some(projection) = gene_axis.projection() else {
        return scatter_sparse_data_group_checked(
            dataset,
            pieces,
            group,
            index_bytes,
            data_bytes,
            out,
        );
    };
    let projection = SparseProjectionCtx {
        num_genes: dataset.num_genes,
        output_genes: projection.output_genes(),
        projection,
    };

    match dataset.index_dtype {
        DType::U32 => scatter_sparse_data_group_projected_checked_typed::<T, u32>(
            projection,
            pieces,
            group,
            index_bytes,
            data_bytes,
            dataset.data.dtype,
            out,
        ),
        DType::I32 => scatter_sparse_data_group_projected_checked_typed::<T, i32>(
            projection,
            pieces,
            group,
            index_bytes,
            data_bytes,
            dataset.data.dtype,
            out,
        ),
        DType::U64 => scatter_sparse_data_group_projected_checked_typed::<T, u64>(
            projection,
            pieces,
            group,
            index_bytes,
            data_bytes,
            dataset.data.dtype,
            out,
        ),
        DType::I64 => scatter_sparse_data_group_projected_checked_typed::<T, i64>(
            projection,
            pieces,
            group,
            index_bytes,
            data_bytes,
            dataset.data.dtype,
            out,
        ),
        dtype => Err(DataBankError::UnsupportedDType {
            dtype,
            context: "CSR indices",
        }),
    }
}

pub(crate) fn try_scatter_sparse_memory_identity_data_group_projected_checked<T: DataValue>(
    dataset: &SparseCsrDataset,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<bool> {
    let SparseGroupSource::Memory {
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

    scatter_sparse_data_group_projected_checked_from_decoded_source(
        dataset,
        pieces,
        group,
        index_bytes,
        bytes.as_ref(),
        gene_axis,
        out,
    )?;
    Ok(true)
}

pub(crate) fn scatter_sparse_data_group_projected_checked_from_decoded_source<T: DataValue>(
    dataset: &SparseCsrDataset,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    decoded: &[u8],
    gene_axis: &GeneAxisPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    let Some(projection) = gene_axis.projection() else {
        return scatter_sparse_data_group_checked_from_decoded_source(
            dataset,
            pieces,
            group,
            index_bytes,
            decoded,
            out,
        );
    };
    let projection = SparseProjectionCtx {
        num_genes: dataset.num_genes,
        output_genes: projection.output_genes(),
        projection,
    };

    match dataset.index_dtype {
        DType::U32 => {
            scatter_sparse_data_group_projected_checked_typed_from_decoded_source::<T, u32>(
                projection,
                pieces,
                group,
                index_bytes,
                decoded,
                dataset.data.dtype,
                out,
            )
        }
        DType::I32 => {
            scatter_sparse_data_group_projected_checked_typed_from_decoded_source::<T, i32>(
                projection,
                pieces,
                group,
                index_bytes,
                decoded,
                dataset.data.dtype,
                out,
            )
        }
        DType::U64 => {
            scatter_sparse_data_group_projected_checked_typed_from_decoded_source::<T, u64>(
                projection,
                pieces,
                group,
                index_bytes,
                decoded,
                dataset.data.dtype,
                out,
            )
        }
        DType::I64 => {
            scatter_sparse_data_group_projected_checked_typed_from_decoded_source::<T, i64>(
                projection,
                pieces,
                group,
                index_bytes,
                decoded,
                dataset.data.dtype,
                out,
            )
        }
        dtype => Err(DataBankError::UnsupportedDType {
            dtype,
            context: "CSR indices",
        }),
    }
}

pub(crate) fn scatter_sparse_data_group_checked_typed<T, I>(
    num_genes: usize,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    data_bytes: &[u8],
    src_dtype: DType,
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    for &piece_index in &group.parts {
        let piece = pieces.get(piece_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR data group piece index is invalid".to_string())
        })?;
        let data_end = piece.group_offset.checked_add(piece.bytes).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR data group offset overflow".to_string())
        })?;
        let index_len = piece.elements.checked_mul(index_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice length overflow".to_string())
        })?;
        let index_end = piece.index_offset.checked_add(index_len).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice offset overflow".to_string())
        })?;
        if data_end > data_bytes.len() || index_end > index_bytes.len() {
            return Err(DataBankError::InvalidArrayMeta(
                "CSR data grouped scatter is out of range".to_string(),
            ));
        }
        scatter_sparse_values_checked_typed::<T, I>(
            num_genes,
            piece.output_row,
            piece.elements,
            &index_bytes[piece.index_offset..index_end],
            &data_bytes[piece.group_offset..data_end],
            src_dtype,
            out,
        )?;
    }
    Ok(())
}

pub(crate) fn scatter_sparse_data_group_checked_typed_from_decoded_source<T, I>(
    num_genes: usize,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    decoded: &[u8],
    src_dtype: DType,
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    for &piece_index in &group.parts {
        let piece = pieces.get(piece_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR data group piece index is invalid".to_string())
        })?;
        if piece.source.len() != piece.bytes {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "CSR data source range length is {}, expected {}",
                piece.source.len(),
                piece.bytes
            )));
        }
        let index_len = piece.elements.checked_mul(index_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice length overflow".to_string())
        })?;
        let index_end = piece.index_offset.checked_add(index_len).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice offset overflow".to_string())
        })?;
        if piece.source.end > decoded.len() || index_end > index_bytes.len() {
            return Err(DataBankError::InvalidArrayMeta(
                "CSR data direct scatter is out of range".to_string(),
            ));
        }
        scatter_sparse_values_checked_typed::<T, I>(
            num_genes,
            piece.output_row,
            piece.elements,
            &index_bytes[piece.index_offset..index_end],
            &decoded[piece.source.start..piece.source.end],
            src_dtype,
            out,
        )?;
    }
    Ok(())
}

pub(crate) fn scatter_sparse_data_group_projected_checked_typed<T, I>(
    projection: SparseProjectionCtx<'_>,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    data_bytes: &[u8],
    src_dtype: DType,
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    for &piece_index in &group.parts {
        let piece = pieces.get(piece_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR data group piece index is invalid".to_string())
        })?;
        let data_end = piece.group_offset.checked_add(piece.bytes).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR data group offset overflow".to_string())
        })?;
        let index_len = piece.elements.checked_mul(index_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice length overflow".to_string())
        })?;
        let index_end = piece.index_offset.checked_add(index_len).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice offset overflow".to_string())
        })?;
        if data_end > data_bytes.len() || index_end > index_bytes.len() {
            return Err(DataBankError::InvalidArrayMeta(
                "CSR data grouped scatter is out of range".to_string(),
            ));
        }
        scatter_sparse_values_projected_checked_typed::<T, I>(
            projection,
            piece.output_row,
            piece.elements,
            &index_bytes[piece.index_offset..index_end],
            &data_bytes[piece.group_offset..data_end],
            src_dtype,
            out,
        )?;
    }
    Ok(())
}

pub(crate) fn scatter_sparse_data_group_projected_checked_typed_from_decoded_source<T, I>(
    projection: SparseProjectionCtx<'_>,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    decoded: &[u8],
    src_dtype: DType,
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    for &piece_index in &group.parts {
        let piece = pieces.get(piece_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR data group piece index is invalid".to_string())
        })?;
        if piece.source.len() != piece.bytes {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "CSR data source range length is {}, expected {}",
                piece.source.len(),
                piece.bytes
            )));
        }
        let index_len = piece.elements.checked_mul(index_size).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice length overflow".to_string())
        })?;
        let index_end = piece.index_offset.checked_add(index_len).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR index slice offset overflow".to_string())
        })?;
        if piece.source.end > decoded.len() || index_end > index_bytes.len() {
            return Err(DataBankError::InvalidArrayMeta(
                "CSR data projected direct scatter is out of range".to_string(),
            ));
        }
        scatter_sparse_values_projected_checked_typed::<T, I>(
            projection,
            piece.output_row,
            piece.elements,
            &index_bytes[piece.index_offset..index_end],
            &decoded[piece.source.start..piece.source.end],
            src_dtype,
            out,
        )?;
    }
    Ok(())
}

pub(crate) unsafe fn scatter_sparse_data_group_unchecked<T: DataValue>(
    dataset: &SparseCsrDataset,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    data_bytes: &[u8],
    out: &mut [T],
) {
    debug_assert_eq!(data_bytes.len(), group.bytes);
    match dataset.index_dtype {
        DType::U32 => unsafe {
            scatter_sparse_data_group_unchecked_typed::<T, u32>(
                dataset.num_genes,
                pieces,
                group,
                index_bytes,
                data_bytes,
                dataset.data.dtype,
                out,
            );
        },
        DType::I32 => unsafe {
            scatter_sparse_data_group_unchecked_typed::<T, i32>(
                dataset.num_genes,
                pieces,
                group,
                index_bytes,
                data_bytes,
                dataset.data.dtype,
                out,
            );
        },
        DType::U64 => unsafe {
            scatter_sparse_data_group_unchecked_typed::<T, u64>(
                dataset.num_genes,
                pieces,
                group,
                index_bytes,
                data_bytes,
                dataset.data.dtype,
                out,
            );
        },
        DType::I64 => unsafe {
            scatter_sparse_data_group_unchecked_typed::<T, i64>(
                dataset.num_genes,
                pieces,
                group,
                index_bytes,
                data_bytes,
                dataset.data.dtype,
                out,
            );
        },
        _ => unreachable!("CSR index dtype was validated during registration"),
    }
}

unsafe fn try_scatter_sparse_memory_identity_data_group_unchecked<T: DataValue>(
    dataset: &SparseCsrDataset,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    out: &mut [T],
) -> DataBankResult<bool> {
    let SparseGroupSource::Memory {
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

    unsafe {
        scatter_sparse_data_group_unchecked_from_decoded_source(
            dataset,
            pieces,
            group,
            index_bytes,
            bytes.as_ref(),
            out,
        );
    }
    Ok(true)
}

pub(crate) unsafe fn scatter_sparse_data_group_unchecked_from_decoded_source<T: DataValue>(
    dataset: &SparseCsrDataset,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    decoded: &[u8],
    out: &mut [T],
) {
    match dataset.index_dtype {
        DType::U32 => unsafe {
            scatter_sparse_data_group_unchecked_typed_from_decoded_source::<T, u32>(
                dataset.num_genes,
                pieces,
                group,
                index_bytes,
                decoded,
                dataset.data.dtype,
                out,
            );
        },
        DType::I32 => unsafe {
            scatter_sparse_data_group_unchecked_typed_from_decoded_source::<T, i32>(
                dataset.num_genes,
                pieces,
                group,
                index_bytes,
                decoded,
                dataset.data.dtype,
                out,
            );
        },
        DType::U64 => unsafe {
            scatter_sparse_data_group_unchecked_typed_from_decoded_source::<T, u64>(
                dataset.num_genes,
                pieces,
                group,
                index_bytes,
                decoded,
                dataset.data.dtype,
                out,
            );
        },
        DType::I64 => unsafe {
            scatter_sparse_data_group_unchecked_typed_from_decoded_source::<T, i64>(
                dataset.num_genes,
                pieces,
                group,
                index_bytes,
                decoded,
                dataset.data.dtype,
                out,
            );
        },
        _ => unreachable!("CSR index dtype was validated during registration"),
    }
}

pub(crate) unsafe fn scatter_sparse_data_group_unchecked_typed<T, I>(
    num_genes: usize,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    data_bytes: &[u8],
    src_dtype: DType,
    out: &mut [T],
) where
    T: DataValue,
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    for &piece_index in &group.parts {
        // SAFETY: group parts are produced by plan_sparse_batch and refer to the
        // same `pieces` slice passed here.
        let piece = unsafe { pieces.get_unchecked(piece_index) };
        let data_end = piece.group_offset + piece.bytes;
        let index_len = piece.elements * index_size;
        let index_end = piece.index_offset + index_len;
        // SAFETY: unchecked CSR access requires valid planned byte ranges.
        let piece_indices = unsafe { index_bytes.get_unchecked(piece.index_offset..index_end) };
        let piece_data = unsafe { data_bytes.get_unchecked(piece.group_offset..data_end) };
        unsafe {
            scatter_sparse_values_unchecked_typed::<T, I>(
                num_genes,
                piece.output_row,
                piece.elements,
                piece_indices,
                piece_data,
                src_dtype,
                out,
            );
        }
    }
}

pub(crate) unsafe fn scatter_sparse_data_group_unchecked_typed_from_decoded_source<T, I>(
    num_genes: usize,
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    index_bytes: &[u8],
    decoded: &[u8],
    src_dtype: DType,
    out: &mut [T],
) where
    T: DataValue,
    I: CsrIndex,
{
    let index_size = mem::size_of::<I>();
    for &piece_index in &group.parts {
        // SAFETY: group parts are produced by plan_sparse_batch and refer to the
        // same `pieces` slice passed here.
        let piece = unsafe { pieces.get_unchecked(piece_index) };
        let index_len = piece.elements * index_size;
        let index_end = piece.index_offset + index_len;
        // SAFETY: unchecked CSR access requires valid planned byte ranges.
        let piece_indices = unsafe { index_bytes.get_unchecked(piece.index_offset..index_end) };
        let piece_data = unsafe { decoded.get_unchecked(piece.source.start..piece.source.end) };
        unsafe {
            scatter_sparse_values_unchecked_typed::<T, I>(
                num_genes,
                piece.output_row,
                piece.elements,
                piece_indices,
                piece_data,
                src_dtype,
                out,
            );
        }
    }
}

pub(crate) fn scatter_sparse_values_checked_typed<T, I>(
    num_genes: usize,
    output_row: usize,
    elements: usize,
    index_bytes: &[u8],
    data_bytes: &[u8],
    src_dtype: DType,
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let value_size = src_dtype.item_size();
    let row_base = output_row
        .checked_mul(num_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("sparse output row overflow".to_string()))?;
    let row_end = row_base.checked_add(num_genes).ok_or_else(|| {
        DataBankError::InvalidConfig("sparse output row end overflow".to_string())
    })?;
    if row_end > out.len() {
        return Err(DataBankError::BufferSizeMismatch {
            expected: row_end,
            actual: out.len(),
        });
    }
    let expected_data_bytes = elements.checked_mul(value_size).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR data byte length overflow".to_string())
    })?;
    if data_bytes.len() != expected_data_bytes {
        return Err(DataBankError::BufferSizeMismatch {
            expected: expected_data_bytes,
            actual: data_bytes.len(),
        });
    }
    let index_size = mem::size_of::<I>();
    let expected_index_bytes = elements.checked_mul(index_size).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR index byte length overflow".to_string())
    })?;
    if index_bytes.len() != expected_index_bytes {
        return Err(DataBankError::BufferSizeMismatch {
            expected: expected_index_bytes,
            actual: index_bytes.len(),
        });
    }

    let index_ptr = index_bytes.as_ptr().cast::<I>();
    for nz in 0..elements {
        let gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) }.checked_gene()?;
        if gene >= num_genes {
            return Err(DataBankError::GeneIndexOutOfRange { gene, num_genes });
        }
        let src_start = nz * value_size;
        T::cast_slice_from(
            &data_bytes[src_start..src_start + value_size],
            src_dtype,
            &mut out[row_base + gene..row_base + gene + 1],
        )?;
    }
    Ok(())
}

pub(crate) fn scatter_sparse_values_projected_checked_typed<T, I>(
    projection: SparseProjectionCtx<'_>,
    output_row: usize,
    elements: usize,
    index_bytes: &[u8],
    data_bytes: &[u8],
    src_dtype: DType,
    out: &mut [T],
) -> DataBankResult<()>
where
    T: DataValue,
    I: CsrIndex,
{
    let value_size = src_dtype.item_size();
    let row_base = output_row
        .checked_mul(projection.output_genes)
        .ok_or_else(|| DataBankError::InvalidConfig("sparse output row overflow".to_string()))?;
    let row_end = row_base
        .checked_add(projection.output_genes)
        .ok_or_else(|| {
            DataBankError::InvalidConfig("sparse output row end overflow".to_string())
        })?;
    if row_end > out.len() {
        return Err(DataBankError::BufferSizeMismatch {
            expected: row_end,
            actual: out.len(),
        });
    }
    let expected_data_bytes = elements.checked_mul(value_size).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR data byte length overflow".to_string())
    })?;
    if data_bytes.len() != expected_data_bytes {
        return Err(DataBankError::BufferSizeMismatch {
            expected: expected_data_bytes,
            actual: data_bytes.len(),
        });
    }
    let index_size = mem::size_of::<I>();
    let expected_index_bytes = elements.checked_mul(index_size).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("CSR index byte length overflow".to_string())
    })?;
    if index_bytes.len() != expected_index_bytes {
        return Err(DataBankError::BufferSizeMismatch {
            expected: expected_index_bytes,
            actual: index_bytes.len(),
        });
    }

    let index_ptr = index_bytes.as_ptr().cast::<I>();
    for nz in 0..elements {
        let gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) }.checked_gene()?;
        if gene >= projection.num_genes {
            return Err(DataBankError::GeneIndexOutOfRange {
                gene,
                num_genes: projection.num_genes,
            });
        }
        let Some(output_col) = projection.projection.output_for_source(gene) else {
            continue;
        };
        let src_start = nz * value_size;
        T::cast_slice_from(
            &data_bytes[src_start..src_start + value_size],
            src_dtype,
            &mut out[row_base + output_col..row_base + output_col + 1],
        )?;
    }
    Ok(())
}

pub(crate) unsafe fn scatter_sparse_values_unchecked_typed<T, I>(
    num_genes: usize,
    output_row: usize,
    elements: usize,
    index_bytes: &[u8],
    data_bytes: &[u8],
    src_dtype: DType,
    out: &mut [T],
) where
    T: DataValue,
    I: CsrIndex,
{
    let row_base = output_row * num_genes;
    let value_size = src_dtype.item_size();
    let index_ptr = index_bytes.as_ptr().cast::<I>();

    for nz in 0..elements {
        let raw_gene = unsafe { ptr::read_unaligned(index_ptr.add(nz)) };
        debug_assert!(
            raw_gene.checked_gene().is_ok_and(|gene| gene < num_genes),
            "unchecked CSR gene index is negative or out of range"
        );
        let gene = unsafe { raw_gene.unchecked_gene() };
        let src_start = nz * value_size;
        // SAFETY: `nz < elements`; `data_bytes.len()` was validated as
        // `elements * value_size`; `out` has `row_base + num_genes` slots and
        // `gene < num_genes`; the dtype gate validated `src_dtype.can_cast_to(T::DTYPE)`.
        let src_elem = unsafe { data_bytes.get_unchecked(src_start..src_start + value_size) };
        let dst = unsafe { out.get_unchecked_mut(row_base + gene..row_base + gene + 1) };
        let _ = T::cast_slice_from(src_elem, src_dtype, dst);
    }
}
