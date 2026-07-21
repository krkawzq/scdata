use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::databank::error::DataBankError;
use crate::iopool::{IoConfig, IoPool};

use super::*;

static FILE_SEQ: AtomicU64 = AtomicU64::new(0);

// ---- cast engine -------------------------------------------------------

#[test]
fn cast_f16_to_f32_matches_numpy() {
    // f16 1.0 == bits 0x3C00 → f32 1.0
    let src = u16::to_ne_bytes(0x3C00u16);
    let mut dst = vec![0.0f32; 1];
    f32::cast_slice_from(&src, DType::F16, &mut dst).unwrap();
    assert_eq!(dst[0], 1.0f32);

    // f16 1.5 == bits 0x3E00
    let src = u16::to_ne_bytes(0x3E00u16);
    f32::cast_slice_from(&src, DType::F16, &mut dst).unwrap();
    assert_eq!(dst[0], 1.5f32);
}

#[test]
fn cast_bf16_to_f32_matches_ml_bfloat() {
    // bf16 1.0 == bits 0x3F80 (same exponent layout as f32's 1.0 in the top
    // 16 bits) → f32 1.0
    let src = u16::to_ne_bytes(0x3F80u16);
    let mut dst = vec![0.0f32; 1];
    f32::cast_slice_from(&src, DType::BF16, &mut dst).unwrap();
    assert_eq!(dst[0], 1.0f32);

    // bf16 1.5 == bits 0x3FC0
    let src = u16::to_ne_bytes(0x3FC0u16);
    f32::cast_slice_from(&src, DType::BF16, &mut dst).unwrap();
    assert_eq!(dst[0], 1.5f32);
}

#[test]
fn cast_f32_to_f16_downcasts_with_rounding() {
    // f32 1.0 → f16 bits 0x3C00
    let src = f32::to_ne_bytes(1.0f32);
    let mut dst = vec![F16Bits(0); 1];
    F16Bits::cast_slice_from(&src, DType::F32, &mut dst).unwrap();
    assert_eq!(dst[0].0, 0x3C00);

    // f32 1.5 → f16 bits 0x3E00
    let src = f32::to_ne_bytes(1.5f32);
    F16Bits::cast_slice_from(&src, DType::F32, &mut dst).unwrap();
    assert_eq!(dst[0].0, 0x3E00);
}

#[test]
fn cast_f32_to_bf16_downcasts() {
    // f32 1.0 → bf16 bits 0x3F80
    let src = f32::to_ne_bytes(1.0f32);
    let mut dst = vec![Bf16Bits(0); 1];
    Bf16Bits::cast_slice_from(&src, DType::F32, &mut dst).unwrap();
    assert_eq!(dst[0].0, 0x3F80);
}

#[test]
fn cast_int_to_float_loses_no_precision_small() {
    let src = u8::to_ne_bytes(7u8);
    let mut dst = vec![0.0f32; 1];
    f32::cast_slice_from(&src, DType::U8, &mut dst).unwrap();
    assert_eq!(dst[0], 7.0f32);
}

#[test]
fn cast_u32_to_u8_truncates_low_byte() {
    let src = u32::to_ne_bytes(0x0000_01FFu32); // 511 → low byte 0xFF
    let mut dst = vec![0u8; 1];
    u8::cast_slice_from(&src, DType::U32, &mut dst).unwrap();
    assert_eq!(dst[0], 0xFF);
}

#[test]
fn cast_float_to_int_is_rejected() {
    let src = f32::to_ne_bytes(1.0f32);
    let mut dst = vec![0i32; 1];
    let err = i32::cast_slice_from(&src, DType::F32, &mut dst).unwrap_err();
    assert!(matches!(
        err,
        DataBankError::CannotCast {
            src: DType::F32,
            dst: DType::I32,
            ..
        }
    ));
}

#[test]
fn cast_identity_is_byte_copy() {
    let src: Vec<u8> = (0..8u32).flat_map(u32::to_ne_bytes).collect();
    let mut dst = vec![0u32; 8];
    u32::cast_slice_from(&src, DType::U32, &mut dst).unwrap();
    assert_eq!(dst, (0..8u32).collect::<Vec<_>>());
}

#[test]
fn dtype_can_cast_to_policy_is_complete_and_matches_the_cast_engine() {
    let all = [
        DType::U8,
        DType::I8,
        DType::U16,
        DType::I16,
        DType::U32,
        DType::I32,
        DType::U64,
        DType::I64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ];
    for src in all {
        for dst in all {
            assert_eq!(
                src.can_cast_to(dst),
                !(src.is_float() && dst.is_int()),
                "unexpected cast policy for {src:?} -> {dst:?}",
            );
        }
    }

    // Explicit integer narrowing is intentionally Rust-`as` conversion.
    let mut signed = [0i64; 1];
    i64::cast_slice_from(&u64::MAX.to_ne_bytes(), DType::U64, &mut signed).unwrap();
    assert_eq!(signed, [-1]);
    let mut unsigned = [0u64; 1];
    u64::cast_slice_from(&(-1i64).to_ne_bytes(), DType::I64, &mut unsigned).unwrap();
    assert_eq!(unsigned, [u64::MAX]);
}

fn temp_file(bytes: &[u8]) -> PathBuf {
    let seq = FILE_SEQ.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "scdata-databank-array-{}-{seq}",
        std::process::id()
    ));
    std::fs::write(&path, bytes).expect("write temp file");
    path
}

#[test]
fn build_array_unregisters_files_when_late_chunk_validation_fails() {
    let path = temp_file(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let io_pool = IoPool::new(IoConfig::default()).expect("io pool");
    let spec = ArraySpec {
        shape: vec![4],
        dtype: DType::U32,
        order: ArrayOrder::C,
        codec: ArrayCodecSpec::Uncompressed,
        grid: ArrayGridSpec::Regular {
            chunk_shape: vec![2],
            edge: EdgeChunkLayout::Padded,
        },
        chunks: vec![
            ChunkSpec {
                source: ChunkSourceSpec::File {
                    path,
                    offset: 0,
                    len: 8,
                },
                decoded_bytes: 8,
            },
            ChunkSpec {
                source: ChunkSourceSpec::DecodedMemory {
                    bytes: Arc::from(vec![0u8; 7].into_boxed_slice()),
                },
                decoded_bytes: 8,
            },
        ],
    };

    let err = build_array_from_spec(spec, &io_pool).expect_err("array build should fail");
    assert!(
        matches!(err, DataBankError::InvalidArrayMeta(message) if message.contains("decoded memory chunk"))
    );
    assert!(
        io_pool.unregister_file(0).is_err(),
        "partially registered file should have been unregistered"
    );
}

#[test]
fn rectilinear_codec_tail_decodes_padded_extent_but_ranges_stay_logical() {
    let default_grid = ArrayGrid::from_spec(
        &[6],
        ArrayGridSpec::Rectilinear {
            logical_axes: vec![vec![0, 4, 6]],
            codec_axes: None,
        },
    )
    .expect("default rectilinear grid");
    assert_eq!(
        default_grid
            .decoded_extent_for_chunk(&[6], 1)
            .expect("default decoded extent"),
        vec![2]
    );

    let io_pool = IoPool::new(IoConfig::default()).expect("io pool");
    let first_chunk: Vec<u8> = (0_u32..4).flat_map(u32::to_ne_bytes).collect();
    let padded_last_chunk: Vec<u8> = (4_u32..8).flat_map(u32::to_ne_bytes).collect();
    let spec = ArraySpec {
        shape: vec![6],
        dtype: DType::U32,
        order: ArrayOrder::C,
        codec: ArrayCodecSpec::Uncompressed,
        grid: ArrayGridSpec::Rectilinear {
            logical_axes: vec![vec![0, 4, 6]],
            codec_axes: Some(vec![vec![0, 4, 8]]),
        },
        chunks: vec![
            ChunkSpec {
                source: ChunkSourceSpec::Memory {
                    bytes: Arc::from(first_chunk.into_boxed_slice()),
                },
                decoded_bytes: 16,
            },
            ChunkSpec {
                source: ChunkSourceSpec::Memory {
                    bytes: Arc::from(padded_last_chunk.clone().into_boxed_slice()),
                },
                decoded_bytes: 16,
            },
        ],
    };

    let array = build_array_from_spec(spec, &io_pool).expect("array accepts codec extent");
    assert_eq!(
        array
            .grid
            .decoded_extent_for_chunk(&array.shape, 1)
            .expect("codec extent"),
        vec![4]
    );
    assert_eq!(
        array.chunks[1].decoded_bytes,
        4 * std::mem::size_of::<u32>()
    );
    let ChunkRef::Memory {
        bytes,
        codec,
        expected_size,
        decoded,
    } = chunk_ref(&array, 1).expect("last chunk ref")
    else {
        panic!("expected memory chunk");
    };
    assert!(!decoded);
    assert_eq!(expected_size, array.chunks[1].decoded_bytes);
    assert_eq!(
        codec
            .decode(&bytes, Some(expected_size))
            .expect("decode padded tail"),
        padded_last_chunk
    );

    let mut pieces = Vec::new();
    array
        .grid
        .for_each_1d_range(&array.shape, array.dtype, &array.chunks, 4, 6, |piece| {
            pieces.push(piece);
            Ok(())
        })
        .expect("logical tail range");
    assert_eq!(pieces.len(), 1);
    assert_eq!(pieces[0].chunk_index, 1);
    assert_eq!(pieces[0].byte_start, 0);
    assert_eq!(pieces[0].byte_end, 2 * std::mem::size_of::<u32>());
    assert_eq!(pieces[0].elements, 2);
}

#[test]
fn registered_file_source_is_not_unregistered_by_array() {
    let path = temp_file(&[1, 2, 3, 4]);
    let io_pool = IoPool::new(IoConfig::default()).expect("io pool");
    let file_id = io_pool
        .register_readonly_file(&path)
        .expect("register external file");
    let file = RegisteredFile::new(file_id).expect("registered file");
    let spec = ArraySpec {
        shape: vec![1],
        dtype: DType::U32,
        order: ArrayOrder::C,
        codec: ArrayCodecSpec::Uncompressed,
        grid: ArrayGridSpec::Regular {
            chunk_shape: vec![1],
            edge: EdgeChunkLayout::Padded,
        },
        chunks: vec![ChunkSpec {
            source: ChunkSourceSpec::RegisteredFile {
                file,
                offset: 0,
                len: 4,
            },
            decoded_bytes: 4,
        }],
    };

    let array = build_array_from_spec(spec, &io_pool).expect("array");
    array
        .unregister_files(&io_pool)
        .expect("array unregister should not close external file");
    io_pool
        .unregister_file(file_id)
        .expect("caller still owns registered file");
}
