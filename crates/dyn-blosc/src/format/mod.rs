pub mod blosc1;
pub mod dyn_blosc;
mod flags;
mod header;
mod index;

pub use flags::{Codec, Shuffle};
pub use header::{
    BloscVersion, Header, BLOSC1_FORMAT_VERSION, BLOSC1_MAX_BLOCK_SIZE, BLOSC1_MAX_BUFFER_SIZE,
    DYN_BLOSC_FORMAT_VERSION, HEADER_LEN,
};

pub(crate) use flags::encode_flags;
pub(crate) use index::{BlockEntry, Index};
