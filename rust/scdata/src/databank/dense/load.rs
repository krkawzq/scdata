use super::*;

pub(crate) fn load_dense_group(
    access: &AccessHandle,
    group: &DenseReadGroup,
) -> DataBankResult<Vec<u8>> {
    match &group.source {
        DenseGroupSource::AccessItem(item) => {
            let mut item = item.clone();
            item.slice = group.slice.clone();
            submit_access_item(access, item)
        }
        DenseGroupSource::Memory {
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

pub(crate) fn load_dense_groups_for_parallel(
    access: &AccessHandle,
    access_config: &ScheduledAccessConfig,
    groups: &[DenseReadGroup],
) -> DataBankResult<Vec<DenseLoadedGroup>> {
    if groups
        .iter()
        .all(|group| matches!(group.source, DenseGroupSource::AccessItem(_)))
    {
        let mut scheduled = access.scheduled(
            groups.iter().map(file_dense_group_access_item),
            *access_config,
        )?;
        let mut out = Vec::with_capacity(groups.len());
        for group in groups {
            let bytes = scheduled.next().ok_or_else(|| {
                DataBankError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "scheduled dense access ended early",
                ))
            })??;
            if bytes.len() != group.bytes {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "dense group decoded length is {}, expected {}",
                    bytes.len(),
                    group.bytes
                )));
            }
            out.push(DenseLoadedGroup::Packed(Arc::from(
                bytes.into_boxed_slice(),
            )));
        }
        if scheduled.next().is_some() {
            return Err(DataBankError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "scheduled dense access returned extra output",
            )));
        }
        Ok(out)
    } else {
        groups
            .iter()
            .map(|group| load_dense_group_for_parallel(access, group))
            .collect()
    }
}

pub(crate) fn load_dense_group_for_parallel(
    access: &AccessHandle,
    group: &DenseReadGroup,
) -> DataBankResult<DenseLoadedGroup> {
    match &group.source {
        DenseGroupSource::AccessItem(_) => {
            let bytes = load_dense_group(access, group)?;
            if bytes.len() != group.bytes {
                return Err(DataBankError::InvalidArrayMeta(format!(
                    "dense group decoded length is {}, expected {}",
                    bytes.len(),
                    group.bytes
                )));
            }
            Ok(DenseLoadedGroup::Packed(Arc::from(
                bytes.into_boxed_slice(),
            )))
        }
        DenseGroupSource::Memory {
            bytes,
            codec,
            expected_size,
            decoded,
        } => {
            if *decoded || codec.is_identity() {
                if bytes.len() != *expected_size {
                    return Err(CodecError::SizeMismatch {
                        codec: codec.name().to_string(),
                        expected: *expected_size,
                        actual: bytes.len(),
                    }
                    .into());
                }
                Ok(DenseLoadedGroup::DecodedSource(Arc::clone(bytes)))
            } else {
                let decoded = codec.decode(bytes.as_ref(), Some(*expected_size))?;
                if decoded.len() != *expected_size {
                    return Err(CodecError::SizeMismatch {
                        codec: codec.name().to_string(),
                        expected: *expected_size,
                        actual: decoded.len(),
                    }
                    .into());
                }
                Ok(DenseLoadedGroup::DecodedSource(Arc::from(
                    decoded.into_boxed_slice(),
                )))
            }
        }
    }
}
