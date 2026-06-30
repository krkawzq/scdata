use super::*;

pub(crate) fn load_sparse_group(
    access: &AccessHandle,
    group: &SparseReadGroup,
) -> DataBankResult<Vec<u8>> {
    match &group.source {
        SparseGroupSource::AccessItem(item) => {
            let mut item = item.clone();
            item.slice = group.slice.clone();
            submit_access_item(access, item)
        }
        SparseGroupSource::Memory {
            bytes,
            codec,
            expected_size,
            decoded,
        } => load_memory_group(
            bytes.as_ref(),
            codec,
            *expected_size,
            *decoded,
            &group.slice,
        ),
    }
}

pub(crate) type SparseGroupScheduledAccess = ScheduledAccess<std::vec::IntoIter<AccessItem>>;

pub(crate) fn schedule_sparse_selected_file_groups(
    access: &AccessHandle,
    access_config: &ScheduledAccessConfig,
    plan: &SparseBatchPlan,
    selected_groups: &[usize],
    cancel: Option<&Arc<PrefetchCancel>>,
) -> DataBankResult<Option<SparseGroupScheduledAccess>> {
    if cancel.is_some_and(|cancel| cancel.is_cancelled()) {
        return Err(DataBankError::PrefetchCancelled);
    }

    let mut items = Vec::with_capacity(selected_groups.len());
    for &group_index in selected_groups {
        let group = &plan.data_groups[group_index];
        if matches!(group.source, SparseGroupSource::AccessItem(_)) {
            items.push(file_sparse_group_access_item(group));
        }
    }
    if items.is_empty() {
        return Ok(None);
    }
    let mut scheduled = access.scheduled(items, *access_config)?;
    if let Some(cancel) = cancel {
        scheduled.set_cancel_handle(Arc::clone(cancel));
    }
    Ok(Some(scheduled))
}

pub(crate) fn next_scheduled_sparse_group_bytes(
    scheduled: &mut SparseGroupScheduledAccess,
) -> DataBankResult<Vec<u8>> {
    scheduled
        .next()
        .ok_or_else(|| {
            DataBankError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "scheduled CSR data access ended early",
            ))
        })?
        .map_err(DataBankError::Io)
}

pub(crate) fn finish_scheduled_sparse_group_bytes(
    scheduled: Option<SparseGroupScheduledAccess>,
) -> DataBankResult<()> {
    let Some(mut scheduled) = scheduled else {
        return Ok(());
    };
    match scheduled.next() {
        None => Ok(()),
        Some(Ok(_)) => Err(DataBankError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "scheduled CSR data access returned extra output",
        ))),
        Some(Err(err)) => Err(DataBankError::Io(err)),
    }
}

pub(crate) fn load_sparse_index_groups(
    access: &AccessHandle,
    access_config: &ScheduledAccessConfig,
    plan: &SparseBatchPlan,
) -> DataBankResult<Vec<u8>> {
    let mut out = zeroed_byte_vec(plan.index_bytes);
    if plan.index_groups.is_empty() {
        return Ok(out);
    }
    if plan
        .index_groups
        .iter()
        .all(|group| matches!(group.source, SparseGroupSource::AccessItem(_)))
    {
        load_sparse_index_groups_scheduled(access, access_config, plan, &mut out)?;
    } else {
        load_sparse_index_groups_sequential(access, plan, &mut out)?;
    }
    Ok(out)
}

pub(crate) fn sparse_plan_is_file_backed(plan: &SparseBatchPlan) -> bool {
    plan.index_groups
        .iter()
        .chain(plan.data_groups.iter())
        .all(|group| matches!(group.source, SparseGroupSource::AccessItem(_)))
}

pub(crate) fn access_sparse_file_groups_checked<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    if plan.index_groups.is_empty() && plan.data_groups.is_empty() {
        return Ok(());
    }
    let mut index_bytes = zeroed_byte_vec(plan.index_bytes);
    let mut scheduled = access.scheduled(
        plan.index_groups
            .iter()
            .chain(plan.data_groups.iter())
            .map(file_sparse_group_access_item),
        *access_config,
    )?;

    for group in &plan.index_groups {
        let bytes = scheduled.next().ok_or_else(|| {
            DataBankError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "scheduled CSR combined access ended during indices",
            ))
        })??;
        copy_sparse_group_to_index_buffer(&plan.index_pieces, group, &bytes, &mut index_bytes)?;
    }

    let output_rows = row_count_for_width(out.len(), dataset.num_genes);
    let output_bytes = out.len().checked_mul(mem::size_of::<T>()).ok_or_else(|| {
        DataBankError::InvalidConfig("sparse output byte length overflow".to_string())
    })?;
    if compute.should_parallelize(output_rows, output_bytes) {
        let mut data_group_bytes = Vec::with_capacity(plan.data_groups.len());
        for group in &plan.data_groups {
            let bytes = scheduled.next().ok_or_else(|| {
                DataBankError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "scheduled CSR combined access ended during data",
                ))
            })??;
            if bytes.len() != group.bytes {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "CSR data group decoded length is {}, expected {}",
                    bytes.len(),
                    group.bytes
                )));
            }
            data_group_bytes.push(Arc::from(bytes.into_boxed_slice()));
        }
        try_scatter_sparse_rows_parallel_checked(
            compute,
            dataset,
            plan,
            Arc::from(index_bytes.into_boxed_slice()),
            data_group_bytes,
            dataset.num_genes,
            None,
            out,
        )?;
    } else {
        for group in &plan.data_groups {
            let bytes = scheduled.next().ok_or_else(|| {
                DataBankError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "scheduled CSR combined access ended during data",
                ))
            })??;
            scatter_sparse_data_group_checked(
                dataset,
                &plan.data_pieces,
                group,
                &index_bytes,
                &bytes,
                out,
            )?;
        }
    }

    if scheduled.next().is_some() {
        return Err(DataBankError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "scheduled CSR combined access returned extra output",
        )));
    }
    Ok(())
}

pub(crate) unsafe fn access_sparse_file_groups_unchecked<T: DataValue>(
    access: &AccessHandle,
    compute: &DataBankComputePool,
    access_config: &ScheduledAccessConfig,
    dataset: &SparseCsrDataset,
    plan: &SparseBatchPlan,
    out: &mut [T],
) -> DataBankResult<()> {
    if plan.index_groups.is_empty() && plan.data_groups.is_empty() {
        return Ok(());
    }
    let mut index_bytes = zeroed_byte_vec(plan.index_bytes);
    let mut scheduled = access.scheduled(
        plan.index_groups
            .iter()
            .chain(plan.data_groups.iter())
            .map(file_sparse_group_access_item),
        *access_config,
    )?;

    for group in &plan.index_groups {
        let bytes = scheduled.next().ok_or_else(|| {
            DataBankError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "scheduled CSR combined access ended during indices",
            ))
        })??;
        copy_sparse_group_to_index_buffer(&plan.index_pieces, group, &bytes, &mut index_bytes)?;
    }

    let output_rows = row_count_for_width(out.len(), dataset.num_genes);
    let output_bytes = out.len().checked_mul(mem::size_of::<T>()).ok_or_else(|| {
        DataBankError::InvalidConfig("sparse output byte length overflow".to_string())
    })?;
    if compute.should_parallelize(output_rows, output_bytes) {
        let mut data_group_bytes = Vec::with_capacity(plan.data_groups.len());
        for group in &plan.data_groups {
            let bytes = scheduled.next().ok_or_else(|| {
                DataBankError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "scheduled CSR combined access ended during data",
                ))
            })??;
            debug_assert_eq!(bytes.len(), group.bytes);
            data_group_bytes.push(Arc::from(bytes.into_boxed_slice()));
        }
        unsafe {
            try_scatter_sparse_rows_parallel_unchecked(
                compute,
                dataset,
                plan,
                Arc::from(index_bytes.into_boxed_slice()),
                data_group_bytes,
                out,
            )?;
        }
    } else {
        for group in &plan.data_groups {
            let bytes = scheduled.next().ok_or_else(|| {
                DataBankError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "scheduled CSR combined access ended during data",
                ))
            })??;
            unsafe {
                scatter_sparse_data_group_unchecked(
                    dataset,
                    &plan.data_pieces,
                    group,
                    &index_bytes,
                    &bytes,
                    out,
                );
            }
        }
    }

    if scheduled.next().is_some() {
        return Err(DataBankError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "scheduled CSR combined access returned extra output",
        )));
    }
    Ok(())
}

pub(crate) fn load_sparse_index_groups_scheduled(
    access: &AccessHandle,
    access_config: &ScheduledAccessConfig,
    plan: &SparseBatchPlan,
    out: &mut [u8],
) -> DataBankResult<()> {
    let mut scheduled = access.scheduled(
        plan.index_groups.iter().map(file_sparse_group_access_item),
        *access_config,
    )?;
    for group in &plan.index_groups {
        let bytes = scheduled.next().ok_or_else(|| {
            DataBankError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "scheduled CSR index access ended early",
            ))
        })??;
        copy_sparse_group_to_index_buffer(&plan.index_pieces, group, &bytes, out)?;
    }
    if scheduled.next().is_some() {
        return Err(DataBankError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "scheduled CSR index access returned extra output",
        )));
    }
    Ok(())
}

pub(crate) fn load_sparse_index_groups_sequential(
    access: &AccessHandle,
    plan: &SparseBatchPlan,
    out: &mut [u8],
) -> DataBankResult<()> {
    for group in &plan.index_groups {
        if try_copy_sparse_memory_identity_group_to_index_buffer(&plan.index_pieces, group, out)? {
            continue;
        }
        let bytes = load_sparse_group(access, group)?;
        copy_sparse_group_to_index_buffer(&plan.index_pieces, group, &bytes, out)?;
    }
    Ok(())
}

pub(crate) fn copy_sparse_group_to_index_buffer(
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    bytes: &[u8],
    out: &mut [u8],
) -> DataBankResult<()> {
    if bytes.len() != group.bytes {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "CSR group decoded length is {}, expected {}",
            bytes.len(),
            group.bytes
        )));
    }

    for &piece_index in &group.parts {
        let piece = pieces.get(piece_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR group piece index is invalid".to_string())
        })?;
        let src_end = piece.group_offset.checked_add(piece.bytes).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR group byte offset overflow".to_string())
        })?;
        let dst_end = piece
            .output_offset
            .checked_add(piece.bytes)
            .ok_or_else(|| {
                DataBankError::InvalidArrayMeta("CSR index output offset overflow".to_string())
            })?;
        if src_end > bytes.len() || dst_end > out.len() {
            return Err(DataBankError::InvalidArrayMeta(
                "CSR index grouped copy is out of range".to_string(),
            ));
        }
        out[piece.output_offset..dst_end].copy_from_slice(&bytes[piece.group_offset..src_end]);
    }
    Ok(())
}

pub(crate) fn try_copy_sparse_memory_identity_group_to_index_buffer(
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    out: &mut [u8],
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

    copy_sparse_group_from_decoded_source_to_index_buffer(pieces, group, bytes.as_ref(), out)?;
    Ok(true)
}

pub(crate) fn copy_sparse_group_from_decoded_source_to_index_buffer(
    pieces: &[SparseReadPiece],
    group: &SparseReadGroup,
    decoded: &[u8],
    out: &mut [u8],
) -> DataBankResult<()> {
    for &piece_index in &group.parts {
        let piece = pieces.get(piece_index).ok_or_else(|| {
            DataBankError::InvalidArrayMeta("CSR group piece index is invalid".to_string())
        })?;
        if piece.source.len() != piece.bytes {
            return Err(DataBankError::InvalidArrayMeta(format!(
                "CSR source range length is {}, expected {}",
                piece.source.len(),
                piece.bytes
            )));
        }
        let dst_end = piece
            .output_offset
            .checked_add(piece.bytes)
            .ok_or_else(|| {
                DataBankError::InvalidArrayMeta("CSR index output offset overflow".to_string())
            })?;
        if piece.source.end > decoded.len() || dst_end > out.len() {
            return Err(DataBankError::InvalidArrayMeta(
                "CSR index direct copy is out of range".to_string(),
            ));
        }
        out[piece.output_offset..dst_end]
            .copy_from_slice(&decoded[piece.source.start..piece.source.end]);
    }
    Ok(())
}

pub(crate) fn load_sparse_data_groups_and_scatter_checked<T: DataValue>(
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
        if try_scatter_sparse_rows_parallel_checked(
            compute,
            dataset,
            plan,
            Arc::clone(&index_bytes),
            data_group_bytes,
            dataset.num_genes,
            None,
            out,
        )? {
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
            scatter_sparse_data_group_checked(
                dataset,
                &plan.data_pieces,
                group,
                index_bytes.as_ref(),
                &bytes,
                out,
            )?;
        }
        if scheduled.next().is_some() {
            return Err(DataBankError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "scheduled CSR data access returned extra output",
            )));
        }
    } else {
        for group in &plan.data_groups {
            if try_scatter_sparse_memory_identity_data_group_checked(
                dataset,
                &plan.data_pieces,
                group,
                index_bytes.as_ref(),
                out,
            )? {
                continue;
            }
            let bytes = load_sparse_group(access, group)?;
            scatter_sparse_data_group_checked(
                dataset,
                &plan.data_pieces,
                group,
                index_bytes.as_ref(),
                &bytes,
                out,
            )?;
        }
    }
    Ok(())
}
