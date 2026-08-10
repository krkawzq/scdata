use crate::blosclz;
use crate::error::{Error, Result};
use crate::format::Codec;

#[derive(Default)]
pub(crate) struct EncodeContext {
    blosclz: blosclz::Workspace,
    zlib: Option<(u8, flate2::Compress)>,
    zstd: Option<(i32, zstd::bulk::Compressor<'static>)>,
}

impl std::fmt::Debug for EncodeContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EncodeContext")
            .field("blosclz", &self.blosclz)
            .field("zlib_initialized", &self.zlib.is_some())
            .field("zstd_initialized", &self.zstd.is_some())
            .finish()
    }
}

#[derive(Default)]
pub(crate) struct DecodeContext {
    zlib: Option<flate2::Decompress>,
    zstd: Option<zstd::bulk::Decompressor<'static>>,
}

impl std::fmt::Debug for DecodeContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DecodeContext")
            .field("zlib_initialized", &self.zlib.is_some())
            .field("zstd_initialized", &self.zstd.is_some())
            .finish()
    }
}

/// Compress one split of a block with the given compressor.
///
/// Returns the compressed size. `Ok(0)` means "not compressible" and the
/// caller stores the piece raw.
pub(crate) fn compress_piece(
    codec: Codec,
    level: u8,
    input: &[u8],
    output: &mut [u8],
    split_block: bool,
    context: &mut EncodeContext,
) -> Result<usize> {
    if input.is_empty() {
        return Ok(0);
    }
    match codec {
        Codec::BloscLz => {
            let level = usize::from(level.max(1));
            context.blosclz.prepare(level)?;
            Ok(blosclz::blosclz_compress_with_workspace(
                input,
                output,
                level,
                split_block,
                &mut context.blosclz,
            ))
        }
        Codec::Lz4 => lz4_flex::block::compress_into(input, output)
            .map_err(|e| Error::Codec(format!("lz4 compress failed: {e}"))),
        Codec::Zlib => {
            use flate2::{Compress, Compression, FlushCompress, Status};
            let encoder = match &mut context.zlib {
                Some((current_level, encoder)) if *current_level == level => {
                    encoder.reset();
                    encoder
                }
                slot => {
                    *slot = Some((
                        level,
                        Compress::new(Compression::new(u32::from(level)), true),
                    ));
                    &mut slot.as_mut().expect("zlib context was initialized").1
                }
            };
            match encoder.compress(input, output, FlushCompress::Finish) {
                Ok(Status::StreamEnd) if encoder.total_in() == input.len() as u64 => {
                    Ok(encoder.total_out() as usize)
                }
                Ok(_) | Err(_) => Ok(0),
            }
        }
        Codec::Zstd => {
            let level = zstd_level(level);
            let compressor = match &mut context.zstd {
                Some((current_level, compressor)) if *current_level == level => compressor,
                slot => match zstd::bulk::Compressor::new(level) {
                    Ok(compressor) => {
                        *slot = Some((level, compressor));
                        &mut slot.as_mut().expect("zstd context was initialized").1
                    }
                    Err(_) => return Ok(0),
                },
            };
            match compressor.compress_to_buffer(input, output) {
                Ok(n) => Ok(n),
                Err(_) => Ok(0),
            }
        }
    }
}

/// Decompress one split of a block.
pub(crate) fn decompress_piece(
    codec: Codec,
    input: &[u8],
    output: &mut [u8],
    context: &mut DecodeContext,
) -> Result<usize> {
    if output.is_empty() {
        return Ok(0);
    }
    match codec {
        Codec::BloscLz => {
            let n = blosclz::blosclz_decompress(input, output);
            if n == 0 {
                return Err(Error::Codec("blosclz decompress failed".into()));
            }
            Ok(n)
        }
        Codec::Lz4 => lz4_flex::block::decompress_into(input, output)
            .map_err(|e| Error::Codec(format!("lz4 decompress failed: {e}"))),
        Codec::Zlib => {
            use flate2::{Decompress, FlushDecompress, Status};
            let decoder = context.zlib.get_or_insert_with(|| Decompress::new(true));
            decoder.reset(true);
            match decoder.decompress(input, output, FlushDecompress::Finish) {
                Ok(Status::StreamEnd) if decoder.total_in() == input.len() as u64 => {
                    Ok(decoder.total_out() as usize)
                }
                Ok(status) => Err(Error::Codec(format!(
                    "zlib stream did not end cleanly ({status:?})"
                ))),
                Err(error) => Err(Error::Codec(format!("zlib decompress failed: {error}"))),
            }
        }
        Codec::Zstd => {
            let decoder = match &mut context.zstd {
                Some(decoder) => decoder,
                slot => {
                    *slot = Some(zstd::bulk::Decompressor::new().map_err(|error| {
                        Error::Codec(format!("zstd decoder initialization failed: {error}"))
                    })?);
                    slot.as_mut().expect("zstd context was initialized")
                }
            };
            decoder
                .decompress_to_buffer(input, output)
                .map_err(|e| Error::Codec(format!("zstd decompress failed: {e}")))
        }
    }
}

/// Map dyn-blosc clevel (0..9) onto zstd compression levels, matching the
/// mapping c-blosc uses: 1..9 -> odd levels up to 22, with 8 and 9 pushed
/// to the top of the zstd range.
fn zstd_level(level: u8) -> i32 {
    let max = zstd::zstd_safe::max_c_level();
    match level {
        0 => 1,
        1..=7 => i32::from(level) * 2 - 1,
        8 => max - 2,
        _ => max,
    }
}
