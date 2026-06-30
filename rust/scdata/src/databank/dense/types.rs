use super::*;

#[derive(Debug, Clone)]
pub(crate) enum DenseGroupSource {
    AccessItem(AccessItem),
    Memory {
        bytes: Arc<[u8]>,
        codec: SharedCodec,
        expected_size: usize,
        decoded: bool,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct DenseReadGroup {
    pub(crate) source: DenseGroupSource,
    pub(crate) slice: SliceSpec,
    pub(crate) slice_ranges: Vec<RangeCopy>,
    pub(crate) parts: Vec<DenseGroupPart>,
    pub(crate) bytes: usize,
}

impl DenseReadGroup {
    pub(crate) fn finalize_slice(&mut self) {
        self.slice = SliceSpec::from_ranges(std::mem::take(&mut self.slice_ranges));
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DenseGroupPart {
    pub(crate) segment_index: usize,
    pub(crate) group_offset: usize,
    pub(crate) bytes: usize,
}

#[derive(Debug, Clone)]
pub(crate) enum DenseLoadedGroup {
    Packed(Arc<[u8]>),
    DecodedSource(Arc<[u8]>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum DenseGroupKey {
    File {
        key: ChunkKey,
        codec: usize,
        expected_size: Option<usize>,
    },
    Memory {
        ptr: usize,
        len: usize,
        codec: usize,
        expected_size: usize,
        decoded: bool,
    },
}
