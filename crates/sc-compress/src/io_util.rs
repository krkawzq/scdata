use crate::error::{Error, Result};
use crate::limits::ReadLimits;
use crate::meta::MetaFile;
use crate::storage::{ByteStore, ByteStoreMut, META_FILE_NAME};

pub(crate) fn write_meta(store: &mut dyn ByteStoreMut, meta: &MetaFile) -> Result<()> {
    let text = serde_json::to_string_pretty(meta)?;
    store.write(META_FILE_NAME, text.as_bytes())
}

pub(crate) fn read_meta(store: &dyn ByteStore, limits: ReadLimits) -> Result<MetaFile> {
    let bytes = store.read_limited(META_FILE_NAME, limits.metadata_size())?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|err| Error::invalid_meta(format!("meta.json is not utf-8: {err}")))?;
    let file: MetaFile = serde_json::from_str(text)?;
    file.validate()?;
    Ok(file)
}

pub(crate) fn u64_slice_to_le_bytes(values: &[u64]) -> Result<Vec<u8>> {
    let capacity = values
        .len()
        .checked_mul(8)
        .ok_or_else(|| Error::invalid_argument("u64 byte length overflow"))?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity)?;
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

pub(crate) fn u64_slice_from_le_bytes(bytes: &[u8]) -> Result<Vec<u64>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(Error::invalid_meta(format!(
            "u64 buffer length {} is not a multiple of 8",
            bytes.len()
        )));
    }
    let mut values = Vec::new();
    values.try_reserve_exact(bytes.len() / 8)?;
    for chunk in bytes.chunks_exact(8) {
        let array: [u8; 8] = chunk
            .try_into()
            .map_err(|_| Error::invalid_meta("invalid u64 byte chunk"))?;
        values.push(u64::from_le_bytes(array));
    }
    Ok(values)
}
