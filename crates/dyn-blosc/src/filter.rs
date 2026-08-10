use crate::bitshuffle;
use crate::error::{Error, Result};
use crate::format::Shuffle;
use crate::shuffle;

pub(crate) fn apply_filter(
    mode: Shuffle,
    typesize: usize,
    src: &[u8],
    dest: &mut [u8],
    tmp: &mut [u8],
) -> Result<()> {
    if typesize == 0 {
        return Err(Error::InvalidArgument(
            "filter element size must be non-zero".into(),
        ));
    }
    if dest.len() < src.len() {
        return Err(Error::BufferTooSmall {
            need: src.len(),
            have: dest.len(),
        });
    }
    match mode {
        Shuffle::None => dest[..src.len()].copy_from_slice(src),
        Shuffle::Bytes => {
            if typesize <= 1 {
                dest[..src.len()].copy_from_slice(src);
            } else {
                // SAFETY: `typesize > 1`; the destination length check above
                // proves both buffers cover `src.len()`, and separate borrows
                // guarantee that the buffers do not overlap.
                unsafe {
                    shuffle::shuffle_unchecked(typesize, src.len(), src, &mut dest[..src.len()]);
                }
            }
        }
        Shuffle::Bits => {
            if src.len() < typesize {
                dest[..src.len()].copy_from_slice(src);
            } else {
                if tmp.len() < src.len() {
                    return Err(Error::BufferTooSmall {
                        need: src.len(),
                        have: tmp.len(),
                    });
                }
                // SAFETY: `typesize` is non-zero; dyn-blosc block sizes fit in
                // positive i32; both destination buffers were checked above,
                // and their distinct borrows do not overlap `src`.
                let rc = unsafe {
                    bitshuffle::bitshuffle_unchecked(
                        typesize,
                        src.len(),
                        src,
                        &mut dest[..src.len()],
                        &mut tmp[..src.len()],
                    )
                };
                if rc < 0 {
                    return Err(Error::Filter(format!("bitshuffle failed: {rc}")));
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn reverse_filter(
    mode: Shuffle,
    typesize: usize,
    src: &[u8],
    dest: &mut [u8],
    tmp: &mut [u8],
) -> Result<()> {
    if typesize == 0 {
        return Err(Error::InvalidArgument(
            "filter element size must be non-zero".into(),
        ));
    }
    if dest.len() < src.len() {
        return Err(Error::BufferTooSmall {
            need: src.len(),
            have: dest.len(),
        });
    }
    match mode {
        Shuffle::None => dest[..src.len()].copy_from_slice(src),
        Shuffle::Bytes => {
            if typesize <= 1 {
                dest[..src.len()].copy_from_slice(src);
            } else {
                // SAFETY: `typesize > 1`; the destination length check above
                // proves both buffers cover `src.len()`, and separate borrows
                // guarantee that the buffers do not overlap.
                unsafe {
                    shuffle::unshuffle_unchecked(typesize, src.len(), src, &mut dest[..src.len()]);
                }
            }
        }
        Shuffle::Bits => {
            if src.len() < typesize {
                dest[..src.len()].copy_from_slice(src);
            } else {
                if tmp.len() < src.len() {
                    return Err(Error::BufferTooSmall {
                        need: src.len(),
                        have: tmp.len(),
                    });
                }
                // SAFETY: `typesize` is non-zero; dyn-blosc block sizes fit in
                // positive i32; both destination buffers were checked above,
                // and their distinct borrows do not overlap `src`.
                let rc = unsafe {
                    bitshuffle::bitunshuffle_unchecked(
                        typesize,
                        src.len(),
                        src,
                        &mut dest[..src.len()],
                        &mut tmp[..src.len()],
                    )
                };
                if rc < 0 {
                    return Err(Error::Filter(format!("bitunshuffle failed: {rc}")));
                }
            }
        }
    }
    Ok(())
}
