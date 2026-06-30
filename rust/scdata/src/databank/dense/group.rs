use super::*;

pub(crate) fn group_dense_segments(
    segments: &[DenseSegment],
) -> DataBankResult<Vec<DenseReadGroup>> {
    let mut groups = Vec::with_capacity(segments.len());
    let mut by_key = fast_hash_map_with_capacity(segments.len());

    for (segment_index, segment) in segments.iter().enumerate() {
        let key = dense_group_key(&segment.chunk);
        let group_index = if let Some(&group_index) = by_key.get(&key) {
            group_index
        } else {
            let group_index = groups.len();
            by_key.insert(key, group_index);
            groups.push(DenseReadGroup {
                source: dense_group_source(&segment.chunk),
                slice: SliceSpec::Full,
                slice_ranges: Vec::new(),
                parts: Vec::new(),
                bytes: 0,
            });
            group_index
        };

        let group = &mut groups[group_index];
        let bytes = segment.source.len();
        let group_offset = append_group_slice(group, segment.source, bytes)?;
        group.parts.push(DenseGroupPart {
            segment_index,
            group_offset,
            bytes,
        });
    }

    for group in &mut groups {
        group.finalize_slice();
    }
    Ok(groups)
}

pub(crate) fn dense_group_key(chunk: &ChunkRef) -> DenseGroupKey {
    match chunk {
        ChunkRef::AccessItem(item) => DenseGroupKey::File {
            key: item.key,
            codec: codec_id(&item.codec),
            expected_size: item.expected_size,
        },
        ChunkRef::Memory {
            bytes,
            codec,
            expected_size,
            decoded,
        } => DenseGroupKey::Memory {
            ptr: bytes.as_ptr() as usize,
            len: bytes.len(),
            codec: codec_id(codec),
            expected_size: *expected_size,
            decoded: *decoded,
        },
    }
}

pub(crate) fn dense_group_source(chunk: &ChunkRef) -> DenseGroupSource {
    match chunk {
        ChunkRef::AccessItem(item) => {
            let mut item = item.clone();
            item.slice = SliceSpec::Full;
            DenseGroupSource::AccessItem(item)
        }
        ChunkRef::Memory {
            bytes,
            codec,
            expected_size,
            decoded,
        } => DenseGroupSource::Memory {
            bytes: Arc::clone(bytes),
            codec: Arc::clone(codec),
            expected_size: *expected_size,
            decoded: *decoded,
        },
    }
}

pub(crate) fn append_group_slice(
    group: &mut DenseReadGroup,
    source: ByteRange,
    bytes: usize,
) -> DataBankResult<usize> {
    if source.len() != bytes {
        return Err(DataBankError::InvalidArrayMeta(format!(
            "dense source range length is {}, expected {bytes}",
            source.len()
        )));
    }
    let output_offset = group.bytes;
    group
        .slice_ranges
        .push(RangeCopy::new(output_offset, source.start, source.end));
    group.bytes = group.bytes.checked_add(bytes).ok_or_else(|| {
        DataBankError::InvalidArrayMeta("dense grouped read byte length overflow".to_string())
    })?;
    Ok(output_offset)
}

pub(crate) fn group_access_item(group: &DenseReadGroup) -> DataBankResult<AccessItem> {
    match &group.source {
        DenseGroupSource::AccessItem(item) => {
            let mut item = item.clone();
            item.slice = group.slice.clone();
            Ok(item)
        }
        DenseGroupSource::Memory { .. } => Err(DataBankError::InvalidArrayMeta(
            "memory chunk reached file scheduled path".to_string(),
        )),
    }
}

pub(crate) fn file_dense_group_access_item(group: &DenseReadGroup) -> AccessItem {
    match &group.source {
        DenseGroupSource::AccessItem(item) => {
            let mut item = item.clone();
            item.slice = group.slice.clone();
            item
        }
        DenseGroupSource::Memory { .. } => {
            unreachable!("memory chunk reached dense file-backed scheduled path")
        }
    }
}
