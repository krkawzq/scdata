//! Single-threaded micro-benchmarks across every public module surface:
//! codecs (parse / build / decode / pipeline / zarr v2 / errors), the decode
//! worker pool, the IO pool (both backends, every op), the access scheduler
//! (cache states / prefetch / scheduled / slice), and the DataBank facade
//! (dense1d / dense2d / sparse csr / file / directory / multi-dtype /
//! by-gene-names / owned / prefetch / unregister).
//!
//! ```sh
//! cargo bench --manifest-path rust/scdata/Cargo.toml --bench modules
//! ```

mod support;

use std::fs::{File, OpenOptions};
use std::io::Cursor;
#[cfg(feature = "uring")]
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "pybind-bench")]
use numpy::IntoPyArray;
#[cfg(feature = "pybind-bench")]
use pyo3::prelude::*;
#[cfg(feature = "pybind-bench")]
use pyo3::types::{PyBytes, PyList, PyModule};

use _scdata::access::{
    AccessConfig, AccessCpuConfig, AccessHandle, AccessItem, AccessRequest, AccessScheduler,
    ChunkKey, FileRef, PrefetchRequest, ScheduledAccessConfig,
};
use _scdata::codecs::{
    codec_from_json_str, codec_from_spec, codec_pipeline_from_specs,
    codec_pipeline_from_zarr_v2_json_str, codec_pipeline_from_zarr_v2_specs,
    codec_specs_from_json_str, BloscCodecConfig, ChunkCodec, CodecError, CodecPipeline, CodecSpec,
    DecodeBuffer, DecodePool, DecodePoolConfig, DecodeRequest, LevelCodecConfig, Lz4CodecConfig,
    LzmaCodecConfig, SharedCodec, UncompressedCodec, UnsupportedCodec, ZstdCodecConfig,
};
use _scdata::databank::{
    ArrayCodecMeta, ArrayMeta, ArrayOrder, Bf16Bits, ChunkStoreMeta, DType, DataBank,
    DataBankConfig, DataValue, Dense1DMeta, Dense2DMeta, F16Bits, FileChunkLocation, FillConfig,
    GeneNameView, MissingGenePolicy, ScheduledPrefetchConfig, SparseCsrDatasetMeta,
};
#[cfg(feature = "uring")]
use _scdata::iopool::UringConfig;
use _scdata::iopool::{
    BaseIoConfig, FileId, IoCommand, IoConfig, IoOutput, IoPool, OpKind, RequestKey, ThreadedConfig,
};
use support::backends::{CodecDecode, SliceIo};
use support::chunks::{
    make_csr_i32_f32_chunks, make_csr_i64_f32_chunks, make_csr_u32_f32_chunks,
    make_csr_u64_f32_chunks, make_dense1d_u32_chunks, make_dense1d_u32_variable_chunks,
    make_dense_u32_chunks, make_dense_u32_chunks_padded, make_dense_u32_chunks_zstd,
    write_chunks_directory_offsets, write_dense_u32_directory, write_dense_u32_file,
    write_dense_u32_file_cropped,
};
use support::codecs::{
    blosc_encode, bz2_encode, crc32_encode, decode_into_checksum, encode_for_spec, gzip_encode,
    lzma_encode, zlib_encode,
};
use support::data::{DataDist, DataProfile, Rng};
use support::{bench, payload, BenchConfig};

fn main() {
    let config = BenchConfig::from_env();

    println!("scdata module benchmarks");
    println!(
        "set SCDATA_BENCH_ITERS=<n> to override per-benchmark iteration counts; SCDATA_BENCH_WARMUPS defaults to 3"
    );

    bench_codecs(config);
    bench_decode_pool(config);
    bench_iopool(config);
    bench_access(config);
    bench_databank(config);
    #[cfg(feature = "pybind-bench")]
    bench_pybind(config);
    bench_data_pipeline(config);
    bench_missing_rate(config);
    bench_scale_sweep(config);
    bench_scenario_matrix(config);
}

// ---------------------------------------------------------------------------
// codecs: parse / build / decode across every supported codec, pipeline, and
// zarr v2 metadata variant, plus error-path coverage.
// ---------------------------------------------------------------------------

fn bench_codecs(config: BenchConfig) {
    let raw = payload(1024 * 1024);
    let raw_64k = payload(64 * 1024);
    let raw_4k = payload(4 * 1024);
    let none = CodecSpec::None.build();
    let zstd = CodecSpec::Zstd(ZstdCodecConfig::default()).build();
    let lz4 = CodecSpec::Lz4(Lz4CodecConfig::default()).build();
    let zlib = CodecSpec::Zlib(LevelCodecConfig::default()).build();
    let gzip = CodecSpec::Gzip(LevelCodecConfig::default()).build();
    let bz2 = CodecSpec::Bz2(LevelCodecConfig::default()).build();
    let lzma = CodecSpec::Lzma(LzmaCodecConfig::default()).build();
    let crc32 = CodecSpec::Crc32.build();
    let blosc = CodecSpec::Blosc(BloscCodecConfig::new("lz4")).build();

    let zstd_encoded = zstd::encode_all(Cursor::new(&raw), 3).expect("zstd encode");
    let zstd_encoded_64k = zstd::encode_all(Cursor::new(&raw_64k), 3).expect("zstd encode 64k");
    let zstd_encoded_4k = zstd::encode_all(Cursor::new(&raw_4k), 3).expect("zstd encode 4k");
    let lz4_encoded = lz4_flex::block::compress_prepend_size(&raw);
    let zlib_encoded = zlib_encode(&raw, 1);
    let gzip_encoded = gzip_encode(&raw, 5);
    let bz2_encoded = bz2_encode(&raw, 5);
    let lzma_encoded = lzma_encode(&raw);
    let crc32_encoded = crc32_encode(&raw);
    let blosc_shuffle_encoded = blosc_encode(&raw, 4, 1, 5, "lz4");
    let blosc_noshuffle_encoded = blosc_encode(&raw, 4, 0, 5, "lz4");
    let blosc_memcpyed_encoded = blosc_encode(&raw, 4, 0, 0, "lz4");

    // pipeline payload: raw -> zstd -> crc32-prepend. Decode order [crc32, zstd].
    let zstd_for_pipeline = zstd::encode_all(Cursor::new(&raw), 3).expect("zstd encode pipeline");
    let pipeline_encoded = crc32_encode(&zstd_for_pipeline);
    let pipeline: SharedCodec = codec_pipeline_from_specs(&[
        CodecSpec::Crc32,
        CodecSpec::Zstd(ZstdCodecConfig::default()),
    ]);

    let codec_json = r#"{"checksum":false,"id":"zstd","level":3}"#;
    let pipeline_json = r#"[{"id":"crc32"},{"checksum":false,"id":"zstd","level":3}]"#;

    bench(config, "codecs/parse_codec_json", 20_000, None, || {
        CodecSpec::from_json_str(codec_json)
            .expect("parse codec")
            .name()
            .len()
    });
    bench(config, "codecs/parse_pipeline_json", 10_000, None, || {
        codec_specs_from_json_str(pipeline_json)
            .expect("parse pipeline")
            .len()
    });
    bench(config, "codecs/build_cached_codec", 200_000, None, || {
        Arc::as_ptr(&CodecSpec::Zstd(ZstdCodecConfig::default()).build()) as *const () as usize
    });
    bench(
        config,
        "codecs/build_cached_pipeline",
        100_000,
        None,
        || {
            Arc::as_ptr(&codec_pipeline_from_specs(&[
                CodecSpec::Crc32,
                CodecSpec::Zstd(ZstdCodecConfig::default()),
            ])) as *const () as usize
        },
    );
    bench(config, "codecs/codec_from_json_str", 20_000, None, || {
        Arc::as_ptr(&codec_from_json_str(codec_json).expect("codec from json")) as *const ()
            as usize
    });
    bench(config, "codecs/codec_from_spec", 50_000, None, || {
        Arc::as_ptr(&codec_from_spec(&CodecSpec::Zstd(
            ZstdCodecConfig::default(),
        ))) as *const () as usize
    });
    bench(
        config,
        "codecs/decode_buffer_capacity",
        200_000,
        None,
        || {
            let mut output = [0u8; 4096];
            DecodeBuffer::new(&mut output).capacity()
        },
    );
    bench(
        config,
        "codecs/decode_into_none_1m",
        512,
        Some(raw.len()),
        || {
            let mut output = vec![0u8; raw.len()];
            let written = none
                .decode_into(&raw, DecodeBuffer::new(&mut output), Some(raw.len()))
                .expect("none decode");
            written ^ output[written / 2] as usize
        },
    );
    bench(
        config,
        "codecs/decode_into_zstd_1m",
        256,
        Some(raw.len()),
        || decode_into_checksum(&zstd, &zstd_encoded, raw.len()),
    );
    bench(
        config,
        "codecs/decode_into_zstd_64k",
        2_048,
        Some(raw_64k.len()),
        || decode_into_checksum(&zstd, &zstd_encoded_64k, raw_64k.len()),
    );
    bench(
        config,
        "codecs/decode_into_zstd_4k",
        20_000,
        Some(raw_4k.len()),
        || decode_into_checksum(&zstd, &zstd_encoded_4k, raw_4k.len()),
    );
    bench(
        config,
        "codecs/decode_into_lz4_1m",
        512,
        Some(raw.len()),
        || decode_into_checksum(&lz4, &lz4_encoded, raw.len()),
    );
    bench(
        config,
        "codecs/decode_into_zlib_1m",
        128,
        Some(raw.len()),
        || decode_into_checksum(&zlib, &zlib_encoded, raw.len()),
    );
    bench(
        config,
        "codecs/decode_into_gzip_1m",
        128,
        Some(raw.len()),
        || decode_into_checksum(&gzip, &gzip_encoded, raw.len()),
    );
    bench(
        config,
        "codecs/decode_into_bz2_1m",
        64,
        Some(raw.len()),
        || decode_into_checksum(&bz2, &bz2_encoded, raw.len()),
    );
    bench(
        config,
        "codecs/decode_into_lzma_1m",
        48,
        Some(raw.len()),
        || decode_into_checksum(&lzma, &lzma_encoded, raw.len()),
    );
    bench(
        config,
        "codecs/decode_into_blosc_lz4_shuffle_1m",
        512,
        Some(raw.len()),
        || decode_into_checksum(&blosc, &blosc_shuffle_encoded, raw.len()),
    );
    bench(
        config,
        "codecs/decode_into_blosc_lz4_noshuffle_1m",
        512,
        Some(raw.len()),
        || decode_into_checksum(&blosc, &blosc_noshuffle_encoded, raw.len()),
    );
    bench(
        config,
        "codecs/decode_into_blosc_memcpyed_1m",
        512,
        Some(raw.len()),
        || decode_into_checksum(&blosc, &blosc_memcpyed_encoded, raw.len()),
    );
    bench(
        config,
        "codecs/decode_into_crc32_1m",
        512,
        Some(raw.len()),
        || decode_into_checksum(&crc32, &crc32_encoded, raw.len()),
    );
    bench(
        config,
        "codecs/decode_pipeline_crc32_zstd_1m",
        192,
        Some(raw.len()),
        || decode_into_checksum(&pipeline, &pipeline_encoded, raw.len()),
    );

    // Blosc compressor cname sweep (decode path covers both the lz4 fast path
    // and the C `blosc_decompress_ctx` fallback for snappy/zlib/zstd).
    for cname in ["lz4", "snappy", "zlib", "zstd"] {
        let codec = CodecSpec::Blosc(BloscCodecConfig::new(cname)).build();
        let encoded = blosc_encode(&raw, 4, 1, 5, cname);
        bench(
            config,
            &format!("codecs/decode_blosc_{cname}_shuffle_1m"),
            384,
            Some(raw.len()),
            || decode_into_checksum(&codec, &encoded, raw.len()),
        );
    }
    // Blosc compression level sweep (lz4 cname, shuffle on).
    for clevel in [1i32, 5, 9] {
        let codec = CodecSpec::Blosc(BloscCodecConfig::new("lz4")).build();
        let encoded = blosc_encode(&raw, 4, 1, clevel, "lz4");
        bench(
            config,
            &format!("codecs/decode_blosc_lz4_clevel{clevel}_1m"),
            384,
            Some(raw.len()),
            || decode_into_checksum(&codec, &encoded, raw.len()),
        );
    }

    // CodecPipeline surface: empty/identity, json-array construction, length.
    bench(
        config,
        "codecs/pipeline_empty_identity",
        5_000,
        None,
        || {
            let pipeline = CodecPipeline::new(Vec::new());
            let is_empty = pipeline.is_empty();
            let len = pipeline.len();
            let decoded = pipeline.decode(b"abcdef", Some(6)).expect("identity");
            usize::from(is_empty) ^ len ^ decoded.len()
        },
    );
    bench(
        config,
        "codecs/pipeline_from_json_array_str",
        10_000,
        None,
        || {
            CodecPipeline::from_json_array_str(pipeline_json)
                .expect("pipeline from json")
                .len()
        },
    );

    // Zarr v2 pipelines: build from JSON values, JSON strings, and specs, then
    // decode a raw -> crc32 -> zstd payload.
    let zarr_v2_filters_json: serde_json::Value =
        serde_json::from_str(r#"[{"id":"crc32"}]"#).expect("filters json");
    let zarr_v2_compressor_json: serde_json::Value =
        serde_json::from_str(r#"{"checksum":false,"id":"zstd","level":3}"#)
            .expect("compressor json");
    let zarr_v2_inner = crc32_encode(&raw);
    let zarr_v2_encoded = zstd::encode_all(Cursor::new(&zarr_v2_inner), 3).expect("zstd encode");
    let zarr_v2_pipeline_values = CodecPipeline::from_zarr_v2_json_values(
        Some(&zarr_v2_filters_json),
        Some(&zarr_v2_compressor_json),
    )
    .expect("zarr v2 pipeline")
    .into_shared();
    let zarr_v2_pipeline_json =
        codec_pipeline_from_zarr_v2_json_str(Some(r#"[{"id":"crc32"}]"#), Some(codec_json))
            .expect("zarr v2 json pipeline");
    let zarr_v2_pipeline_specs = codec_pipeline_from_zarr_v2_specs(
        &[CodecSpec::Crc32],
        Some(&CodecSpec::Zstd(ZstdCodecConfig::default())),
    );
    bench(
        config,
        "codecs/decode_zarr_v2_crc32_zstd_values_1m",
        192,
        Some(raw.len()),
        || decode_into_checksum(&zarr_v2_pipeline_values, &zarr_v2_encoded, raw.len()),
    );
    bench(
        config,
        "codecs/decode_zarr_v2_crc32_zstd_json_1m",
        192,
        Some(raw.len()),
        || decode_into_checksum(&zarr_v2_pipeline_json, &zarr_v2_encoded, raw.len()),
    );
    bench(
        config,
        "codecs/decode_zarr_v2_crc32_zstd_specs_1m",
        192,
        Some(raw.len()),
        || decode_into_checksum(&zarr_v2_pipeline_specs, &zarr_v2_encoded, raw.len()),
    );

    // Error paths: unsupported codec, size mismatch, crc32 checksum mismatch.
    let unknown = CodecSpec::Unknown("mystery".to_string()).build();
    bench(config, "codecs/error_unknown_decode", 50_000, None, || {
        let err = unknown.decode(b"x", None).expect_err("unsupported");
        matches!(err, CodecError::Unsupported { .. }) as usize
    });
    bench(config, "codecs/error_size_mismatch", 50_000, None, || {
        let err = UncompressedCodec
            .decode(b"abcdef", Some(5))
            .expect_err("size mismatch");
        matches!(err, CodecError::SizeMismatch { .. }) as usize
    });
    let mut bad_crc = crc32_encode(&raw_64k);
    bad_crc[4] ^= 0xff;
    bench(config, "codecs/error_crc32_mismatch", 5_000, None, || {
        let err = crc32.decode(&bad_crc, None).expect_err("crc mismatch");
        matches!(err, CodecError::Decode { .. }) as usize
    });
    bench(
        config,
        "codecs/build_unsupported_codec",
        100_000,
        None,
        || UnsupportedCodec::new("mystery").name().len(),
    );

    // Isolation micro-benches for the owned-buffer regression: same zstd 64k
    // payload, three codec entry points, no pool. Locates whether the cost is
    // in zstd's `decompress` vs `decompress_to_buffer`, in the `decode_to_vec`
    // wrapper, or in per-iteration allocation.
    let zstd_codec = zstd.clone();
    let mut reused_slice = vec![0u8; raw_64k.len()];
    bench(
        config,
        "codecs/decode_zstd_64k_direct",
        2_048,
        Some(raw_64k.len()),
        || {
            let decoded = zstd_codec
                .decode(&zstd_encoded_64k, Some(raw_64k.len()))
                .expect("decode");
            decoded.len() ^ decoded[decoded.len() / 2] as usize
        },
    );
    bench(
        config,
        "codecs/decode_to_vec_zstd_64k_direct",
        2_048,
        Some(raw_64k.len()),
        || {
            let out = Vec::with_capacity(raw_64k.len());
            let decoded = zstd_codec
                .decode_to_vec(&zstd_encoded_64k, out, Some(raw_64k.len()))
                .expect("decode_to_vec");
            decoded.len() ^ decoded[decoded.len() / 2] as usize
        },
    );
    bench(
        config,
        "codecs/decode_into_zstd_64k_reused",
        2_048,
        Some(raw_64k.len()),
        || {
            let written = zstd_codec
                .decode_into(
                    &zstd_encoded_64k,
                    DecodeBuffer::new(&mut reused_slice),
                    Some(raw_64k.len()),
                )
                .expect("decode_into");
            written ^ reused_slice[written / 2] as usize
        },
    );
    bench(
        config,
        "codecs/decode_into_zstd_64k_alloc",
        2_048,
        Some(raw_64k.len()),
        || {
            let mut output = vec![0u8; raw_64k.len()];
            let written = zstd_codec
                .decode_into(
                    &zstd_encoded_64k,
                    DecodeBuffer::new(&mut output),
                    Some(raw_64k.len()),
                )
                .expect("decode_into");
            written ^ output[written / 2] as usize
        },
    );
}

// ---------------------------------------------------------------------------
// decode pool: submission paths, real codec work, caller-owned output buffer,
// try_submit, from_spec, and config validation.
// ---------------------------------------------------------------------------

fn bench_decode_pool(config: BenchConfig) {
    let raw: Arc<[u8]> = Arc::from(payload(64 * 1024));
    let zstd_encoded: Arc<[u8]> =
        Arc::from(zstd::encode_all(Cursor::new(&raw[..]), 3).expect("zstd encode"));
    let pool = DecodePool::new(DecodePoolConfig {
        num_workers: 2,
        queue_capacity: 64,
        cpus: None,
    })
    .expect("decode pool");
    let none_codec = CodecSpec::None.build();
    let zstd_codec = CodecSpec::Zstd(ZstdCodecConfig::default()).build();

    bench(
        config,
        "codecs/decode_pool_submit_none_64k",
        2048,
        Some(raw.len()),
        || {
            let request = DecodeRequest::new(Arc::clone(&none_codec), Arc::clone(&raw))
                .with_expected_size(raw.len())
                .with_reuse_capacity_output(Vec::with_capacity(raw.len()));
            let decoded = pool
                .submit(request)
                .expect("submit decode")
                .blocking_recv()
                .expect("decode");
            decoded.len() ^ decoded[decoded.len() / 2] as usize
        },
    );
    bench(
        config,
        "codecs/decode_pool_submit_zstd_64k",
        2048,
        Some(raw.len()),
        || {
            let request = DecodeRequest::new(Arc::clone(&zstd_codec), Arc::clone(&zstd_encoded))
                .with_expected_size(raw.len());
            let decoded = pool
                .submit(request)
                .expect("submit decode")
                .blocking_recv()
                .expect("decode");
            decoded.len() ^ decoded[decoded.len() / 2] as usize
        },
    );
    bench(
        config,
        "codecs/decode_pool_from_spec_zstd_64k",
        2048,
        Some(raw.len()),
        || {
            let request = DecodeRequest::from_spec(
                &CodecSpec::Zstd(ZstdCodecConfig::default()),
                Arc::clone(&zstd_encoded),
            )
            .with_expected_size(raw.len());
            let decoded = pool
                .submit(request)
                .expect("submit decode")
                .blocking_recv()
                .expect("decode");
            decoded.len() ^ decoded[decoded.len() / 2] as usize
        },
    );
    bench(
        config,
        "codecs/decode_pool_try_submit_zstd_64k",
        2048,
        Some(raw.len()),
        || {
            let request = DecodeRequest::new(Arc::clone(&zstd_codec), Arc::clone(&zstd_encoded))
                .with_expected_size(raw.len());
            let decoded = pool
                .try_submit(request)
                .expect("try_submit decode")
                .blocking_recv()
                .expect("decode");
            decoded.len() ^ decoded[decoded.len() / 2] as usize
        },
    );
    bench(
        config,
        "codecs/decode_pool_submit_zstd_64k_owned_buffer",
        2048,
        Some(raw.len()),
        || {
            let request = DecodeRequest::new(Arc::clone(&zstd_codec), Arc::clone(&zstd_encoded))
                .with_expected_size(raw.len())
                .with_reuse_capacity_output(Vec::with_capacity(raw.len()));
            let decoded = pool
                .submit(request)
                .expect("submit decode")
                .blocking_recv()
                .expect("decode");
            decoded.len() ^ decoded[decoded.len() / 2] as usize
        },
    );
    // Owned buffer but zero-filled by the caller (touches every page before
    // crossing to the worker). If this is fast, the penalty was page faults on
    // the worker's first write to uninitialized capacity.
    bench(
        config,
        "codecs/decode_pool_submit_zstd_64k_owned_zeroed",
        2048,
        Some(raw.len()),
        || {
            let request = DecodeRequest::new(Arc::clone(&zstd_codec), Arc::clone(&zstd_encoded))
                .with_expected_size(raw.len())
                .with_reuse_initialized_output(vec![0u8; raw.len()]);
            let decoded = pool
                .submit(request)
                .expect("submit decode")
                .blocking_recv()
                .expect("decode");
            decoded.len() ^ decoded[decoded.len() / 2] as usize
        },
    );
    // None codec + owned buffer: isolates whether the penalty is codec-independent
    // (pure pool + cross-thread buffer transfer + copy).
    bench(
        config,
        "codecs/decode_pool_submit_none_64k_owned_buffer",
        2048,
        Some(raw.len()),
        || {
            let request = DecodeRequest::new(Arc::clone(&none_codec), Arc::clone(&raw))
                .with_expected_size(raw.len())
                .with_reuse_capacity_output(Vec::with_capacity(raw.len()));
            let decoded = pool
                .submit(request)
                .expect("submit decode")
                .blocking_recv()
                .expect("decode");
            decoded.len() ^ decoded[decoded.len() / 2] as usize
        },
    );
    bench(
        config,
        "codecs/decode_pool_batch_32x_zstd_64k",
        128,
        Some(raw.len() * 32),
        || {
            let futures = (0..32)
                .map(|_| {
                    let request =
                        DecodeRequest::new(Arc::clone(&zstd_codec), Arc::clone(&zstd_encoded))
                            .with_expected_size(raw.len());
                    pool.submit(request).expect("submit decode")
                })
                .collect::<Vec<_>>();
            futures.into_iter().fold(0usize, |acc, future| {
                let decoded = future.blocking_recv().expect("decode");
                acc ^ decoded.len() ^ decoded[decoded.len() / 2] as usize
            })
        },
    );
    bench(
        config,
        "codecs/decode_pool_drop_future_zstd_64k",
        512,
        Some(raw.len()),
        || {
            let request = DecodeRequest::new(Arc::clone(&zstd_codec), Arc::clone(&zstd_encoded))
                .with_expected_size(raw.len());
            let future = pool.submit(request).expect("submit dropped decode");
            drop(future);
            let request = DecodeRequest::new(Arc::clone(&zstd_codec), Arc::clone(&zstd_encoded))
                .with_expected_size(raw.len());
            let decoded = pool
                .submit(request)
                .expect("submit follow-up decode")
                .blocking_recv()
                .expect("follow-up decode");
            decoded.len() ^ decoded[decoded.len() / 2] as usize
        },
    );

    // 1 MiB comparison: does the owned-buffer penalty scale with payload size
    // (data migration / page faults) or stay flat (fixed scheduling overhead)?
    let raw_1m: Arc<[u8]> = Arc::from(payload(1024 * 1024));
    let zstd_encoded_1m: Arc<[u8]> =
        Arc::from(zstd::encode_all(Cursor::new(&raw_1m[..]), 3).expect("zstd encode 1m"));
    bench(
        config,
        "codecs/decode_pool_try_submit_queue_full",
        32,
        None,
        || {
            let small_pool = DecodePool::new(DecodePoolConfig {
                num_workers: 1,
                queue_capacity: 1,
                cpus: None,
            })
            .expect("small decode pool");
            let mut futures = Vec::new();
            let mut queue_full = 0usize;
            for _ in 0..32 {
                let request =
                    DecodeRequest::new(Arc::clone(&zstd_codec), Arc::clone(&zstd_encoded_1m))
                        .with_expected_size(raw_1m.len());
                match small_pool.try_submit(request) {
                    Ok(future) => futures.push(future),
                    Err(CodecError::QueueFull { .. }) => queue_full += 1,
                    Err(err) => panic!("unexpected try_submit error: {err}"),
                }
            }
            for future in futures {
                let _ = future.blocking_recv();
            }
            queue_full
        },
    );
    bench(
        config,
        "codecs/decode_pool_submit_zstd_1m",
        192,
        Some(raw_1m.len()),
        || {
            let request = DecodeRequest::new(Arc::clone(&zstd_codec), Arc::clone(&zstd_encoded_1m))
                .with_expected_size(raw_1m.len());
            let decoded = pool
                .submit(request)
                .expect("submit decode")
                .blocking_recv()
                .expect("decode");
            decoded.len() ^ decoded[decoded.len() / 2] as usize
        },
    );
    bench(
        config,
        "codecs/decode_pool_submit_zstd_1m_owned_buffer",
        192,
        Some(raw_1m.len()),
        || {
            let request = DecodeRequest::new(Arc::clone(&zstd_codec), Arc::clone(&zstd_encoded_1m))
                .with_expected_size(raw_1m.len())
                .with_reuse_capacity_output(Vec::with_capacity(raw_1m.len()));
            let decoded = pool
                .submit(request)
                .expect("submit decode")
                .blocking_recv()
                .expect("decode");
            decoded.len() ^ decoded[decoded.len() / 2] as usize
        },
    );

    // Config validation: valid path and the three reject paths.
    bench(
        config,
        "codecs/decode_pool_config_validate_ok",
        20_000,
        None,
        || DecodePoolConfig::default().validate().is_ok() as usize,
    );
    bench(
        config,
        "codecs/decode_pool_config_validate_err",
        20_000,
        None,
        || {
            let mut bad = DecodePoolConfig {
                num_workers: 0,
                ..Default::default()
            };
            let r1 = bad.validate().is_err();
            bad.num_workers = 1;
            bad.queue_capacity = 0;
            let r2 = bad.validate().is_err();
            bad.queue_capacity = 1;
            bad.cpus = Some(Vec::new());
            let r3 = bad.validate().is_err();
            usize::from(r1) + usize::from(r2) + usize::from(r3)
        },
    );
    bench(
        config,
        "codecs/decode_pool_config_accessors",
        50_000,
        None,
        || {
            let config = DecodePoolConfig::default();
            config.worker_count() ^ config.queue_capacity()
        },
    );
}

// ---------------------------------------------------------------------------
// iopool: read size scaling, sharding, writes, sync, truncate, register
// lifecycle (existing/options/try_unregister), try_recv polling, priority &
// dedup_key, config validation, and the io_uring backend.
// ---------------------------------------------------------------------------

fn bench_iopool(config: BenchConfig) {
    let data = payload(8 * 1024 * 1024);
    let path = support::bench_data_dir().join(format!("iopool-{}.bin", std::process::id()));
    std::fs::write(&path, &data).expect("write iopool bench data");

    let pool = IoPool::new(IoConfig::Threaded(ThreadedConfig {
        base: BaseIoConfig {
            max_in_flight: 128,
            priority_levels: 2,
            queue_shards: 2,
            assume_non_overlapping_reads: true,
        },
        num_workers: 4,
        cpus: None,
    }))
    .expect("iopool");
    let file = pool.register_readonly_file(&path).expect("register file");

    bench_read_size(
        config,
        &pool,
        file,
        &data,
        4 * 1024,
        "iopool/threaded_read_4k",
    );
    bench_read_size(
        config,
        &pool,
        file,
        &data,
        64 * 1024,
        "iopool/threaded_read_64k",
    );
    bench_read_size(
        config,
        &pool,
        file,
        &data,
        1024 * 1024,
        "iopool/threaded_read_1m",
    );

    bench(
        config,
        "iopool/threaded_duplicate_read_64k_x4",
        1024,
        Some(64 * 1024 * 4),
        || {
            let futures = (0..4)
                .map(|_| {
                    pool.submit(IoCommand::read(file, 0, 64 * 1024, 0))
                        .expect("submit duplicate read")
                })
                .collect::<Vec<_>>();
            futures.into_iter().fold(0usize, |acc, future| {
                let bytes = future.blocking_recv_read().expect("read duplicate");
                acc ^ bytes.len() ^ bytes[bytes.len() / 2] as usize
            })
        },
    );

    // try_recv polling path: submit, then spin try_recv_read until ready.
    bench(
        config,
        "iopool/threaded_read_try_recv_64k",
        2048,
        Some(64 * 1024),
        || {
            let mut future = pool
                .submit(IoCommand::read(file, 0, 64 * 1024, 0))
                .expect("submit read");
            let bytes = loop {
                match future.try_recv_read().expect("try_recv") {
                    Some(bytes) => break bytes,
                    None => std::hint::spin_loop(),
                }
            };
            bytes.len() ^ bytes[bytes.len() / 2] as usize
        },
    );
    bench(
        config,
        "iopool/threaded_error_read_past_eof",
        1024,
        None,
        || {
            pool.submit(IoCommand::read(file, data.len() as u64 + 1, 4096, 0))
                .expect("submit eof read")
                .blocking_recv_read()
                .expect_err("read past EOF")
                .to_string()
                .len()
        },
    );

    let rw_file = pool
        .register_readwrite_file(&path)
        .expect("register readwrite file");
    bench(
        config,
        "iopool/threaded_write_64k",
        1024,
        Some(64 * 1024),
        || {
            let buf = vec![0xA5u8; 64 * 1024];
            pool.submit(IoCommand::write(rw_file, 0, buf, 0))
                .expect("submit write")
                .blocking_recv()
                .expect("write")
                .bytes_written()
                .expect("write output")
        },
    );
    bench(config, "iopool/threaded_sync_all", 1024, None, || {
        let out = pool
            .submit(IoCommand::sync_all(rw_file, 0))
            .expect("submit sync_all")
            .blocking_recv()
            .expect("sync_all");
        usize::from(matches!(out, IoOutput::SyncAll))
    });
    bench(config, "iopool/threaded_sync_data", 1024, None, || {
        let out = pool
            .submit(IoCommand::sync_data(rw_file, 0))
            .expect("submit sync_data")
            .blocking_recv()
            .expect("sync_data");
        usize::from(matches!(out, IoOutput::SyncData))
    });
    pool.unregister_file(rw_file).expect("unregister rw file");

    // truncate on a dedicated small file so it never disturbs the read path.
    let trunc_path =
        support::bench_data_dir().join(format!("iopool-trunc-{}.bin", std::process::id()));
    std::fs::write(&trunc_path, &data[..1024 * 64]).expect("write trunc file");
    let trunc_file = pool
        .register_readwrite_file(&trunc_path)
        .expect("register trunc file");
    bench(config, "iopool/threaded_truncate", 1024, None, || {
        let out = pool
            .submit(IoCommand::truncate(trunc_file, 1024, 0))
            .expect("submit truncate")
            .blocking_recv()
            .expect("truncate");
        usize::from(matches!(out, IoOutput::Truncate))
    });
    bench(config, "iopool/threaded_metadata", 4096, None, || {
        pool.submit(IoCommand::metadata(file, 0))
            .expect("submit metadata")
            .blocking_recv()
            .expect("metadata")
            .metadata()
            .expect("metadata output")
            .len as usize
    });
    bench(
        config,
        "iopool/threaded_drop_future_read_64k",
        1024,
        Some(64 * 1024),
        || {
            let future = pool
                .submit(IoCommand::read(file, 0, 64 * 1024, 0))
                .expect("submit dropped read");
            drop(future);
            let bytes = pool
                .submit(IoCommand::read(file, 0, 64 * 1024, 0))
                .expect("submit follow-up read")
                .blocking_recv_read()
                .expect("follow-up read");
            bytes.len() ^ bytes[bytes.len() / 2] as usize
        },
    );
    pool.unregister_file(trunc_file)
        .expect("unregister trunc file");
    let _ = std::fs::remove_file(trunc_path);

    pool.unregister_file(file).expect("unregister file");

    // Sharded pool: 4 independent queues, 8 workers.
    let sharded = IoPool::new(IoConfig::Threaded(ThreadedConfig {
        base: BaseIoConfig {
            max_in_flight: 256,
            priority_levels: 2,
            queue_shards: 4,
            assume_non_overlapping_reads: true,
        },
        num_workers: 8,
        cpus: None,
    }))
    .expect("sharded iopool");
    let sfile = sharded
        .register_readonly_file(&path)
        .expect("register sharded file");
    bench(
        config,
        "iopool/threaded_read_sharded_4q_64k",
        2048,
        Some(64 * 1024),
        || {
            let offset = (0usize).wrapping_mul(64 * 1024) % (data.len() - 64 * 1024);
            let bytes = sharded
                .submit(IoCommand::read(sfile, offset as u64, 64 * 1024, 0))
                .expect("submit read")
                .blocking_recv_read()
                .expect("read");
            bytes.len() ^ bytes[bytes.len() / 2] as usize
        },
    );
    sharded
        .unregister_file(sfile)
        .expect("unregister sharded file");

    // Register lifecycle: register_file (rw-or-ro fallback), register_file_with_options,
    // register_existing_file, and try_unregister_file.
    bench(config, "iopool/register_file_cycle", 1024, None, || {
        let id = pool.register_file(&path).expect("register_file");
        pool.unregister_file(id).expect("unregister");
        id
    });
    bench(
        config,
        "iopool/register_file_with_options_cycle",
        1024,
        None,
        || {
            let mut options = OpenOptions::new();
            options.read(true);
            let id = pool
                .register_file_with_options(&path, &options)
                .expect("register_file_with_options");
            pool.unregister_file(id).expect("unregister");
            id
        },
    );
    bench(
        config,
        "iopool/register_existing_file_cycle",
        1024,
        None,
        || {
            let file = File::open(&path).expect("open existing");
            let id = pool
                .register_existing_file(file)
                .expect("register_existing_file");
            pool.unregister_file(id).expect("unregister");
            id
        },
    );
    bench(config, "iopool/try_unregister_file", 1024, None, || {
        let id = pool.register_readonly_file(&path).expect("register");
        pool.try_unregister_file(id).is_ok() as usize
    });

    // Lightweight command introspection: priority() and dedup_key().
    bench(
        config,
        "iopool/cmd_priority_dedup_key",
        100_000,
        None,
        || {
            let read = IoCommand::read(file, 128, 4096, 2);
            let write = IoCommand::write(file, 0, Vec::new(), 1);
            let trunc = IoCommand::truncate(file, 0, 0);
            let read_key = read.dedup_key().map(|k| key_bits(&k)).unwrap_or(0);
            let write_key = write.dedup_key().is_some() as usize;
            let trunc_key = trunc.dedup_key().is_some() as usize;
            read.priority() ^ read_key ^ write_key ^ trunc_key
        },
    );

    // Config validation surface.
    bench(config, "iopool/base_config_validate", 50_000, None, || {
        BaseIoConfig::default().validate().is_ok() as usize
    });
    bench(
        config,
        "iopool/threaded_config_validate",
        50_000,
        None,
        || ThreadedConfig::default().validate().is_ok() as usize,
    );
    bench(
        config,
        "iopool/io_config_kind_base_validate",
        20_000,
        None,
        || {
            let config = IoConfig::Threaded(ThreadedConfig::default());
            let kind = matches!(config.kind(), _scdata::iopool::BackendKind::Threaded);
            let base_ok = config.base().validate().is_ok();
            let validate_ok = config.validate().is_ok();
            usize::from(kind) + usize::from(base_ok) + usize::from(validate_ok)
        },
    );

    #[cfg(feature = "uring")]
    {
        bench_iopool_uring(config, &path, &data);
    }

    let _ = std::fs::remove_file(path);
}

#[allow(deprecated)]
fn key_bits(key: &RequestKey) -> usize {
    key.file
        ^ key.offset as usize
        ^ key.len
        ^ match key.kind {
            OpKind::Read => 1,
            OpKind::Write => 2,
            OpKind::Fsync => 3,
        }
}

fn bench_read_size(
    config: BenchConfig,
    pool: &IoPool,
    file: FileId,
    data: &[u8],
    read_size: usize,
    name: &str,
) {
    let max_offset = data.len() - read_size;
    let mut next = 0usize;
    bench(config, name, 2048, Some(read_size), move || {
        let offset = (next * read_size) % (max_offset + 1);
        next = next.wrapping_add(1);
        let bytes = pool
            .submit(IoCommand::read(file, offset as u64, read_size, 0))
            .expect("submit read")
            .blocking_recv_read()
            .expect("read");
        bytes.len() ^ bytes[bytes.len() / 2] as usize
    });
}

#[cfg(feature = "uring")]
fn bench_iopool_uring(config: BenchConfig, path: &Path, data: &[u8]) {
    let pool = match IoPool::new(IoConfig::Uring(UringConfig {
        base: BaseIoConfig {
            max_in_flight: 128,
            priority_levels: 2,
            queue_shards: 1,
            assume_non_overlapping_reads: true,
        },
        entries: 256,
        drivers: 1,
        iowq_bounded_workers: 0,
        iowq_unbounded_workers: 0,
        registered_files: 64,
    })) {
        Ok(pool) => pool,
        Err(err)
            if matches!(
                err.raw_os_error(),
                Some(libc::ENOSYS | libc::EPERM | libc::EACCES)
            ) =>
        {
            eprintln!("[iopool] skipping io_uring benchmark (unavailable): {err}");
            return;
        }
        Err(err) => panic!("create uring pool: {err}"),
    };
    let file = pool.register_readonly_file(path).expect("register file");
    let read_size = 64 * 1024usize;
    let max_offset = data.len() - read_size;
    let mut next = 0usize;
    bench(
        config,
        "iopool/uring_read_64k",
        2048,
        Some(read_size),
        || {
            let offset = (next * read_size) % (max_offset + 1);
            next = next.wrapping_add(1);
            let bytes = pool
                .submit(IoCommand::read(file, offset as u64, read_size, 0))
                .expect("submit read")
                .blocking_recv_read()
                .expect("read");
            bytes.len() ^ bytes[bytes.len() / 2] as usize
        },
    );
    pool.unregister_file(file).expect("unregister uring file");
}

// ---------------------------------------------------------------------------
// access: cache hit/miss, keep_decoded toggle, lookahead depth, prefetch,
// try_send, send_prefetch (no reply), scatter-copy slice, config validation.
// ---------------------------------------------------------------------------

fn bench_access(config: BenchConfig) {
    let chunk = 64 * 1024usize;
    let backing = Arc::<[u8]>::from(payload(4 * 1024 * 1024));

    let handle_keep = AccessScheduler::spawn(
        AccessConfig {
            queue_capacity: 128,
            scheduler_shards: 1,
            cache_capacity_bytes: 2 * 1024 * 1024,
            memory_budget_bytes: 4 * 1024 * 1024,
            default_io_priority: 0,
            keep_decoded: true,
            cpu: AccessCpuConfig {
                num_workers: 2,
                queue_capacity: 128,
                cpus: None,
            },
        },
        Box::new(SliceIo::new(Arc::clone(&backing))),
        Box::new(CodecDecode),
    )
    .expect("access scheduler (keep_decoded)");
    let handle_evict = AccessScheduler::spawn(
        AccessConfig {
            queue_capacity: 128,
            scheduler_shards: 1,
            cache_capacity_bytes: 2 * 1024 * 1024,
            memory_budget_bytes: 4 * 1024 * 1024,
            default_io_priority: 0,
            keep_decoded: false,
            cpu: AccessCpuConfig {
                num_workers: 2,
                queue_capacity: 128,
                cpus: None,
            },
        },
        Box::new(SliceIo::new(Arc::clone(&backing))),
        Box::new(CodecDecode),
    )
    .expect("access scheduler (no keep_decoded)");
    let codec = CodecSpec::None.build();

    // Cold path: every iteration reads a different offset (cache miss).
    let mut offset = 0usize;
    bench(
        config,
        "access/send_read_decode_miss_64k",
        2048,
        Some(chunk),
        || {
            let start = offset % (backing.len() - chunk);
            offset = offset.wrapping_add(chunk);
            send_read(&handle_keep, start, chunk, &codec)
        },
    );

    // Hot path: fixed offset, warmed into the decoded cache before timing.
    bench(
        config,
        "access/send_read_decode_cache_hit_64k",
        8_192,
        Some(chunk),
        || send_read(&handle_keep, 0, chunk, &codec),
    );

    // try_send path (non-blocking submit).
    let mut offset_try = 0usize;
    bench(
        config,
        "access/try_send_read_decode_miss_64k",
        2048,
        Some(chunk),
        || {
            let start = offset_try % (backing.len() - chunk);
            offset_try = offset_try.wrapping_add(chunk);
            try_send_read(&handle_keep, start, chunk, &codec)
        },
    );

    // Scatter-copy slice: only the [0, chunk) prefix of the decoded chunk is
    // copied back to the caller.
    bench(
        config,
        "access/send_read_with_slice_64k",
        2048,
        Some(chunk),
        || send_read_with_slice(&handle_keep, 0, chunk, &codec),
    );
    bench(config, "access/error_invalid_slice", 2048, None, || {
        send_read_error(&handle_keep, 0, chunk, &codec, Some(vec![0, 0, chunk + 1]))
    });

    let handle_oom = AccessScheduler::spawn(
        AccessConfig {
            queue_capacity: 16,
            scheduler_shards: 1,
            cache_capacity_bytes: 1024,
            memory_budget_bytes: 1024,
            default_io_priority: 0,
            keep_decoded: true,
            cpu: AccessCpuConfig {
                num_workers: 1,
                queue_capacity: 16,
                cpus: None,
            },
        },
        Box::new(SliceIo::new(Arc::clone(&backing))),
        Box::new(CodecDecode),
    )
    .expect("access scheduler (oom)");
    bench(
        config,
        "access/error_memory_budget_exhausted",
        512,
        None,
        || send_read_error(&handle_oom, 0, chunk, &codec, None),
    );

    // Prefetch cold: rotating keys evict from the 2 MiB cache, so each
    // prefetch performs real IO + cache insert.
    let prefetch_keys: Vec<ChunkKey> = (0..64)
        .map(|idx| ChunkKey::new(FileRef(1), (idx * chunk) as u64, chunk))
        .collect();
    let mut prefetch_idx = 0usize;
    bench(
        config,
        "access/prefetch_cold_64k",
        2048,
        Some(chunk),
        || {
            let key = prefetch_keys[prefetch_idx % prefetch_keys.len()];
            prefetch_idx = prefetch_idx.wrapping_add(1);
            handle_keep
                .prefetch(key)
                .expect("prefetch")
                .blocking_recv()
                .expect("prefetch reply")
                .expect("prefetch result");
            key.offset as usize ^ key.len
        },
    );

    // send_prefetch with no reply channel: fire-and-forget submit cost.
    let mut pf_no_reply_idx = 0usize;
    bench(
        config,
        "access/send_prefetch_no_reply_64k",
        2048,
        Some(chunk),
        || {
            let key = prefetch_keys[pf_no_reply_idx % prefetch_keys.len()];
            pf_no_reply_idx = pf_no_reply_idx.wrapping_add(1);
            handle_keep
                .send_prefetch(PrefetchRequest::new(key))
                .expect("send_prefetch");
            key.offset as usize ^ key.len
        },
    );

    let scheduled_items = (0..16)
        .map(|idx| {
            AccessItem::new(
                ChunkKey::new(FileRef(1), (idx * chunk) as u64, chunk),
                Arc::clone(&codec),
                Some(chunk),
            )
        })
        .collect::<Vec<_>>();
    bench(
        config,
        "access/scheduled_read_decode_16x64k_keep",
        512,
        Some(chunk * scheduled_items.len()),
        || {
            let scheduled = handle_keep
                .scheduled(
                    scheduled_items.clone(),
                    ScheduledAccessConfig {
                        prefetch_step: 4,
                        decode_ahead_steps: 2,
                        ready_ahead_steps: 1,
                    },
                )
                .expect("scheduled access");
            scheduled.fold(0usize, |acc, result| {
                let bytes = result.expect("scheduled result");
                acc ^ bytes.len() ^ bytes[bytes.len() / 2] as usize
            })
        },
    );
    bench(
        config,
        "access/scheduled_read_decode_16x64k_evict",
        512,
        Some(chunk * scheduled_items.len()),
        || {
            let scheduled = handle_evict
                .scheduled(
                    scheduled_items.clone(),
                    ScheduledAccessConfig {
                        prefetch_step: 4,
                        decode_ahead_steps: 2,
                        ready_ahead_steps: 1,
                    },
                )
                .expect("scheduled access");
            scheduled.fold(0usize, |acc, result| {
                let bytes = result.expect("scheduled result");
                acc ^ bytes.len() ^ bytes[bytes.len() / 2] as usize
            })
        },
    );
    bench(
        config,
        "access/scheduled_read_decode_16x64k_deep_lookahead",
        512,
        Some(chunk * scheduled_items.len()),
        || {
            let scheduled = handle_keep
                .scheduled(
                    scheduled_items.clone(),
                    ScheduledAccessConfig {
                        prefetch_step: 8,
                        decode_ahead_steps: 4,
                        ready_ahead_steps: 2,
                    },
                )
                .expect("scheduled access");
            scheduled.fold(0usize, |acc, result| {
                let bytes = result.expect("scheduled result");
                acc ^ bytes.len() ^ bytes[bytes.len() / 2] as usize
            })
        },
    );

    // Config validation: valid path plus the rejection rules.
    bench(config, "access/config_validate_ok", 20_000, None, || {
        AccessConfig::default().validate().is_ok() as usize
    });
    bench(config, "access/config_validate_err", 10_000, None, || {
        let mut bad = AccessConfig {
            queue_capacity: 0,
            ..Default::default()
        };
        let r1 = bad.validate().is_err();
        bad.queue_capacity = 1;
        bad.scheduler_shards = 0;
        let r2 = bad.validate().is_err();
        bad.scheduler_shards = 1;
        bad.cache_capacity_bytes = 0;
        let r3 = bad.validate().is_err();
        bad.cache_capacity_bytes = 1;
        bad.memory_budget_bytes = 0;
        let r4 = bad.validate().is_err();
        usize::from(r1) + usize::from(r2) + usize::from(r3) + usize::from(r4)
    });
    bench(config, "access/cpu_config_validate", 50_000, None, || {
        AccessCpuConfig::default().validate().is_ok() as usize
    });
}

fn send_read(handle: &AccessHandle, start: usize, chunk: usize, codec: &SharedCodec) -> usize {
    let (tx, rx) = tokio::sync::oneshot::channel();
    handle
        .send(AccessRequest::new(
            ChunkKey::new(FileRef(1), start as u64, chunk),
            Arc::clone(codec),
            Some(chunk),
            tx,
        ))
        .expect("send access request");
    let bytes = rx
        .blocking_recv()
        .expect("access reply")
        .expect("access result");
    bytes.len() ^ bytes[bytes.len() / 2] as usize
}

fn try_send_read(handle: &AccessHandle, start: usize, chunk: usize, codec: &SharedCodec) -> usize {
    let (tx, rx) = tokio::sync::oneshot::channel();
    handle
        .try_send(AccessRequest::new(
            ChunkKey::new(FileRef(1), start as u64, chunk),
            Arc::clone(codec),
            Some(chunk),
            tx,
        ))
        .expect("try_send access request");
    let bytes = rx
        .blocking_recv()
        .expect("access reply")
        .expect("access result");
    bytes.len() ^ bytes[bytes.len() / 2] as usize
}

fn send_read_with_slice(
    handle: &AccessHandle,
    start: usize,
    chunk: usize,
    codec: &SharedCodec,
) -> usize {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let request = AccessRequest::new(
        ChunkKey::new(FileRef(1), start as u64, chunk),
        Arc::clone(codec),
        Some(chunk),
        tx,
    )
    .with_slice(Some(vec![0, 0, chunk]));
    handle.send(request).expect("send sliced request");
    let bytes = rx
        .blocking_recv()
        .expect("access reply")
        .expect("access result");
    bytes.len() ^ bytes[0] as usize
}

fn send_read_error(
    handle: &AccessHandle,
    start: usize,
    chunk: usize,
    codec: &SharedCodec,
    slice: Option<Vec<usize>>,
) -> usize {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let request = AccessRequest::new(
        ChunkKey::new(FileRef(1), start as u64, chunk),
        Arc::clone(codec),
        Some(chunk),
        tx,
    )
    .with_slice(slice);
    handle.send(request).expect("send error request");
    rx.blocking_recv()
        .expect("access error reply")
        .expect_err("access request should fail")
        .to_string()
        .len()
}

// ---------------------------------------------------------------------------
// databank: dense1d, compressed dense2d, sparse csr, file-backed store,
// directory store, multi-dtype, by-gene-names, owned/alloc, prefetch,
// unregister, gene views, and config validation.
// ---------------------------------------------------------------------------

fn bench_databank(config: BenchConfig) {
    let cells = 1024usize;
    let genes = 1024usize;
    let chunk_rows = 16usize;
    let chunk_cols = 128usize;
    let chunk_grid = vec![cells / chunk_rows, genes / chunk_cols];
    let chunks = make_dense_u32_chunks(cells, genes, chunk_rows, chunk_cols);
    let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
    let id = bank
        .register_dense_2d(Dense2DMeta {
            gene_names: (0..genes).map(|idx| format!("gene_{idx}")).collect(),
            data: ArrayMeta {
                shape: vec![cells, genes],
                chunk_shape: vec![chunk_rows, chunk_cols],
                chunk_grid_shape: chunk_grid.clone(),
                dtype: DType::U32,
                order: ArrayOrder::C,
                codec: ArrayCodecMeta::Uncompressed,
                chunks: ChunkStoreMeta::Memory { chunks },
                variable_chunks: false,
                chunk_boundaries: None,
            },
        })
        .expect("register dense dataset");
    let selected = (0..32).map(|idx| idx * 7 % cells).collect::<Vec<_>>();
    let mut out = vec![0u32; selected.len() * genes];

    bench(
        config,
        "databank/dense2d_memory_access_32x1024",
        256,
        Some(out.len() * std::mem::size_of::<u32>()),
        || {
            bank.access_cells(id, &selected, &mut out, None)
                .expect("access cells");
            out.iter().step_by(257).fold(0usize, |acc, value| {
                acc ^ usize::try_from(*value).unwrap_or(0)
            })
        },
    );
    bench(
        config,
        "databank/dense2d_access_values_wrapper_32x1024",
        256,
        Some(out.len() * std::mem::size_of::<u32>()),
        || {
            bank.access_cells_values(id, &selected, &mut out)
                .expect("access cells values");
            out.iter().step_by(257).fold(0usize, |acc, value| {
                acc ^ usize::try_from(*value).unwrap_or(0)
            })
        },
    );
    let tuned_access = ScheduledAccessConfig {
        prefetch_step: 8,
        decode_ahead_steps: 4,
        ready_ahead_steps: 2,
    };
    bench(
        config,
        "databank/dense2d_access_with_config_32x1024",
        192,
        Some(out.len() * std::mem::size_of::<u32>()),
        || {
            bank.access_cells_with_config(id, &selected, &mut out, None, tuned_access)
                .expect("access cells with config");
            out.iter().step_by(257).fold(0usize, |acc, value| {
                acc ^ usize::try_from(*value).unwrap_or(0)
            })
        },
    );

    // Dense2D with zstd-compressed chunks: decode + scatter end to end.
    let zstd_chunks = make_dense_u32_chunks_zstd(cells, genes, chunk_rows, chunk_cols, 3);
    let zstd_id = bank
        .register_dense_2d(Dense2DMeta {
            gene_names: (0..genes).map(|idx| format!("gene_{idx}")).collect(),
            data: ArrayMeta {
                shape: vec![cells, genes],
                chunk_shape: vec![chunk_rows, chunk_cols],
                chunk_grid_shape: chunk_grid,
                dtype: DType::U32,
                order: ArrayOrder::C,
                codec: ArrayCodecMeta::CodecJson(r#"{"id":"zstd","level":3}"#.to_string()),
                chunks: ChunkStoreMeta::Memory {
                    chunks: zstd_chunks,
                },
                variable_chunks: false,
                chunk_boundaries: None,
            },
        })
        .expect("register zstd dense dataset");
    let mut zstd_out = vec![0u32; selected.len() * genes];
    bench(
        config,
        "databank/dense2d_zstd_memory_access_32x1024",
        192,
        Some(zstd_out.len() * std::mem::size_of::<u32>()),
        || {
            bank.access_cells(zstd_id, &selected, &mut zstd_out, None)
                .expect("access cells");
            zstd_out.iter().step_by(257).fold(0usize, |acc, value| {
                acc ^ usize::try_from(*value).unwrap_or(0)
            })
        },
    );

    // Zarr v2 codec metadata (filters=[crc32], compressor=zstd) on dense2d.
    let zarr_v2_chunks: Vec<Arc<[u8]>> =
        make_dense_u32_chunks(cells, genes, chunk_rows, chunk_cols)
            .into_iter()
            .map(|raw| {
                let filtered = crc32_encode(&raw);
                Arc::from(
                    zstd::encode_all(Cursor::new(&filtered), 3)
                        .expect("zstd encode")
                        .into_boxed_slice(),
                )
            })
            .collect();
    let zarr_v2_id = bank
        .register_dense_2d(Dense2DMeta {
            gene_names: (0..genes).map(|idx| format!("gene_{idx}")).collect(),
            data: ArrayMeta {
                shape: vec![cells, genes],
                chunk_shape: vec![chunk_rows, chunk_cols],
                chunk_grid_shape: vec![cells / chunk_rows, genes / chunk_cols],
                dtype: DType::U32,
                order: ArrayOrder::C,
                codec: ArrayCodecMeta::ZarrV2Json {
                    filters: Some(r#"[{"id":"crc32"}]"#.to_string()),
                    compressor: Some(r#"{"id":"zstd","level":3}"#.to_string()),
                },
                chunks: ChunkStoreMeta::Memory {
                    chunks: zarr_v2_chunks,
                },
                variable_chunks: false,
                chunk_boundaries: None,
            },
        })
        .expect("register zarr v2 dense dataset");
    let mut zarr_v2_out = vec![0u32; selected.len() * genes];
    bench(
        config,
        "databank/dense2d_zarr_v2_memory_access_32x1024",
        160,
        Some(zarr_v2_out.len() * std::mem::size_of::<u32>()),
        || {
            bank.access_cells(zarr_v2_id, &selected, &mut zarr_v2_out, None)
                .expect("access cells");
            zarr_v2_out.iter().step_by(257).fold(0usize, |acc, value| {
                acc ^ usize::try_from(*value).unwrap_or(0)
            })
        },
    );

    // Dense1D: cell-major 1D layout.
    let d1_cells = 1024usize;
    let d1_genes = 1024usize;
    let d1_chunk_len = 4096usize;
    let d1_total = d1_cells * d1_genes;
    let d1_chunks = make_dense1d_u32_chunks(d1_cells, d1_genes, d1_chunk_len);
    let d1_id = bank
        .register_dense_1d(Dense1DMeta {
            gene_names: (0..d1_genes).map(|idx| format!("g{idx}")).collect(),
            data: ArrayMeta {
                shape: vec![d1_total],
                chunk_shape: vec![d1_chunk_len],
                chunk_grid_shape: vec![d1_total / d1_chunk_len],
                dtype: DType::U32,
                order: ArrayOrder::C,
                codec: ArrayCodecMeta::Uncompressed,
                chunks: ChunkStoreMeta::Memory { chunks: d1_chunks },
                variable_chunks: false,
                chunk_boundaries: None,
            },
        })
        .expect("register dense1d dataset");
    let d1_selected = (0..32).map(|idx| idx * 7 % d1_cells).collect::<Vec<_>>();
    let mut d1_out = vec![0u32; d1_selected.len() * d1_genes];
    bench(
        config,
        "databank/dense1d_memory_access_32x1024",
        256,
        Some(d1_out.len() * std::mem::size_of::<u32>()),
        || {
            bank.access_cells(d1_id, &d1_selected, &mut d1_out, None)
                .expect("access cells");
            d1_out.iter().step_by(257).fold(0usize, |acc, value| {
                acc ^ usize::try_from(*value).unwrap_or(0)
            })
        },
    );

    // Dense1D with explicit rectilinear variable chunks.
    let v_cells = 8usize;
    let v_genes = 5usize;
    let v_boundaries = vec![0usize, 7, 13, 25, v_cells * v_genes];
    let v_chunks = make_dense1d_u32_variable_chunks(v_cells, v_genes, &v_boundaries);
    let v_id = bank
        .register_dense_1d(Dense1DMeta {
            gene_names: (0..v_genes).map(|idx| format!("vg{idx}")).collect(),
            data: ArrayMeta {
                shape: vec![v_cells * v_genes],
                chunk_shape: vec![v_boundaries[1] - v_boundaries[0]],
                chunk_grid_shape: vec![v_boundaries.len() - 1],
                dtype: DType::U32,
                order: ArrayOrder::C,
                codec: ArrayCodecMeta::Uncompressed,
                chunks: ChunkStoreMeta::Memory { chunks: v_chunks },
                variable_chunks: true,
                chunk_boundaries: Some(vec![v_boundaries]),
            },
        })
        .expect("register variable dense1d dataset");
    let v_selected = vec![0usize, 3, 7, 2];
    let mut v_out = vec![0u32; v_selected.len() * v_genes];
    bench(
        config,
        "databank/dense1d_variable_chunks_4x5",
        1024,
        Some(v_out.len() * std::mem::size_of::<u32>()),
        || {
            bank.access_cells(v_id, &v_selected, &mut v_out, None)
                .expect("access variable dense1d");
            v_out.iter().fold(0usize, |acc, value| {
                acc ^ usize::try_from(*value).unwrap_or(0)
            })
        },
    );

    // Sparse CSR: checked vs unchecked scatter.
    let csr_cells = 128usize;
    let csr_genes = 256usize;
    let nnz_per_cell = 8usize;
    let nnz = csr_cells * nnz_per_cell;
    let (indptr, indices_chunk, data_chunk) =
        make_csr_u32_f32_chunks(csr_cells, csr_genes, nnz_per_cell);
    let csr_id = bank
        .register_sparse_csr(SparseCsrDatasetMeta {
            gene_names: (0..csr_genes).map(|idx| format!("gene_{idx}")).collect(),
            indptr,
            indices: ArrayMeta {
                shape: vec![nnz],
                chunk_shape: vec![nnz],
                chunk_grid_shape: vec![1],
                dtype: DType::U32,
                order: ArrayOrder::C,
                codec: ArrayCodecMeta::Uncompressed,
                chunks: ChunkStoreMeta::Memory {
                    chunks: vec![indices_chunk],
                },
                variable_chunks: false,
                chunk_boundaries: None,
            },
            data: ArrayMeta {
                shape: vec![nnz],
                chunk_shape: vec![nnz],
                chunk_grid_shape: vec![1],
                dtype: DType::F32,
                order: ArrayOrder::C,
                codec: ArrayCodecMeta::Uncompressed,
                chunks: ChunkStoreMeta::Memory {
                    chunks: vec![data_chunk],
                },
                variable_chunks: false,
                chunk_boundaries: None,
            },
            index_dtype: DType::U32,
            num_cells: csr_cells,
            num_genes: csr_genes,
        })
        .expect("register sparse csr dataset");
    let csr_selected = (0..32).map(|idx| idx * 3 % csr_cells).collect::<Vec<_>>();
    let mut csr_out = vec![0.0f32; csr_selected.len() * csr_genes];
    bench(
        config,
        "databank/sparse_csr_memory_access_checked_32x256",
        256,
        Some(csr_out.len() * std::mem::size_of::<f32>()),
        || {
            bank.access_cells(csr_id, &csr_selected, &mut csr_out, None)
                .expect("access cells");
            csr_out
                .iter()
                .step_by(251)
                .fold(0usize, |acc, value| acc ^ value.to_bits() as usize)
        },
    );
    bench(
        config,
        "databank/sparse_csr_memory_access_unchecked_32x256",
        512,
        Some(csr_out.len() * std::mem::size_of::<f32>()),
        || {
            unsafe {
                bank.access_cells_unchecked(csr_id, &csr_selected, &mut csr_out, None)
                    .expect("access cells unchecked");
            }
            csr_out
                .iter()
                .step_by(251)
                .fold(0usize, |acc, value| acc ^ value.to_bits() as usize)
        },
    );
    bench(
        config,
        "databank/sparse_csr_unchecked_with_config_32x256",
        384,
        Some(csr_out.len() * std::mem::size_of::<f32>()),
        || {
            unsafe {
                bank.access_cells_unchecked_with_config(
                    csr_id,
                    &csr_selected,
                    &mut csr_out,
                    None,
                    tuned_access,
                )
                .expect("access cells unchecked with config");
            }
            csr_out
                .iter()
                .step_by(251)
                .fold(0usize, |acc, value| acc ^ value.to_bits() as usize)
        },
    );
    bench_sparse_csr_index_dtype(
        config,
        "i32",
        DType::I32,
        make_csr_i32_f32_chunks(csr_cells, csr_genes, nnz_per_cell),
        csr_cells,
        csr_genes,
        nnz_per_cell,
    );
    bench_sparse_csr_index_dtype(
        config,
        "u64",
        DType::U64,
        make_csr_u64_f32_chunks(csr_cells, csr_genes, nnz_per_cell),
        csr_cells,
        csr_genes,
        nnz_per_cell,
    );
    bench_sparse_csr_index_dtype(
        config,
        "i64",
        DType::I64,
        make_csr_i64_f32_chunks(csr_cells, csr_genes, nnz_per_cell),
        csr_cells,
        csr_genes,
        nnz_per_cell,
    );

    // File-backed dense2d (uncompressed): real IO through the IoPool.
    let fo_path = support::bench_data_dir().join(format!("databank-fo-{}.bin", std::process::id()));
    let fo_locations = write_dense_u32_file(&fo_path, cells, genes, chunk_rows, chunk_cols);
    let fo_id = bank
        .register_dense_2d(Dense2DMeta {
            gene_names: (0..genes).map(|idx| format!("gene_{idx}")).collect(),
            data: ArrayMeta {
                shape: vec![cells, genes],
                chunk_shape: vec![chunk_rows, chunk_cols],
                chunk_grid_shape: vec![cells / chunk_rows, genes / chunk_cols],
                dtype: DType::U32,
                order: ArrayOrder::C,
                codec: ArrayCodecMeta::Uncompressed,
                chunks: ChunkStoreMeta::FileOffset {
                    path: fo_path.clone(),
                    locations: fo_locations,
                },
                variable_chunks: false,
                chunk_boundaries: None,
            },
        })
        .expect("register fileoffset dense dataset");
    let mut fo_out = vec![0u32; selected.len() * genes];
    bench(
        config,
        "databank/dense2d_fileoffset_access_32x1024",
        128,
        Some(fo_out.len() * std::mem::size_of::<u32>()),
        || {
            bank.access_cells(fo_id, &selected, &mut fo_out, None)
                .expect("access cells");
            fo_out.iter().step_by(257).fold(0usize, |acc, value| {
                acc ^ usize::try_from(*value).unwrap_or(0)
            })
        },
    );

    // Directory-backed dense2d: one file per chunk.
    let dir_dir = support::bench_data_dir().join(format!("databank-dir-{}", std::process::id()));
    std::fs::create_dir_all(&dir_dir).expect("create dir store");
    let dir_locations = write_dense_u32_directory(&dir_dir, cells, genes, chunk_rows, chunk_cols);
    let dir_id = bank
        .register_dense_2d(Dense2DMeta {
            gene_names: (0..genes).map(|idx| format!("gene_{idx}")).collect(),
            data: ArrayMeta {
                shape: vec![cells, genes],
                chunk_shape: vec![chunk_rows, chunk_cols],
                chunk_grid_shape: vec![cells / chunk_rows, genes / chunk_cols],
                dtype: DType::U32,
                order: ArrayOrder::C,
                codec: ArrayCodecMeta::Uncompressed,
                chunks: ChunkStoreMeta::Directory {
                    locations: dir_locations,
                },
                variable_chunks: false,
                chunk_boundaries: None,
            },
        })
        .expect("register directory dense dataset");
    let mut dir_out = vec![0u32; selected.len() * genes];
    bench(
        config,
        "databank/dense2d_directory_access_32x1024",
        128,
        Some(dir_out.len() * std::mem::size_of::<u32>()),
        || {
            bank.access_cells(dir_id, &selected, &mut dir_out, None)
                .expect("access cells");
            dir_out.iter().step_by(257).fold(0usize, |acc, value| {
                acc ^ usize::try_from(*value).unwrap_or(0)
            })
        },
    );
    let _ = std::fs::remove_dir_all(&dir_dir);

    // Non-divisible dense2d edges: memory/directory stores are physically
    // padded; legacy file-offset stores are physically cropped.
    let edge_cells = 35usize;
    let edge_genes = 70usize;
    let edge_chunk_rows = 16usize;
    let edge_chunk_cols = 32usize;
    let edge_grid = vec![
        edge_cells.div_ceil(edge_chunk_rows),
        edge_genes.div_ceil(edge_chunk_cols),
    ];
    let edge_selected = vec![0usize, 15, 16, 34];
    let padded_chunks =
        make_dense_u32_chunks_padded(edge_cells, edge_genes, edge_chunk_rows, edge_chunk_cols);
    let edge_mem_id = bank
        .register_dense_2d(Dense2DMeta {
            gene_names: (0..edge_genes)
                .map(|idx| format!("edge_gene_{idx}"))
                .collect(),
            data: ArrayMeta {
                shape: vec![edge_cells, edge_genes],
                chunk_shape: vec![edge_chunk_rows, edge_chunk_cols],
                chunk_grid_shape: edge_grid.clone(),
                dtype: DType::U32,
                order: ArrayOrder::C,
                codec: ArrayCodecMeta::Uncompressed,
                chunks: ChunkStoreMeta::Memory {
                    chunks: padded_chunks.clone(),
                },
                variable_chunks: false,
                chunk_boundaries: None,
            },
        })
        .expect("register padded edge memory dataset");
    let mut edge_mem_out = vec![0u32; edge_selected.len() * edge_genes];
    bench(
        config,
        "databank/dense2d_memory_padded_edges_4x70",
        512,
        Some(edge_mem_out.len() * std::mem::size_of::<u32>()),
        || {
            bank.access_cells(edge_mem_id, &edge_selected, &mut edge_mem_out, None)
                .expect("access padded edges");
            edge_mem_out.iter().fold(0usize, |acc, value| {
                acc ^ usize::try_from(*value).unwrap_or(0)
            })
        },
    );

    let edge_fo_path =
        support::bench_data_dir().join(format!("databank-edge-fo-{}.bin", std::process::id()));
    let edge_fo_locations = write_dense_u32_file_cropped(
        &edge_fo_path,
        edge_cells,
        edge_genes,
        edge_chunk_rows,
        edge_chunk_cols,
    );
    let edge_fo_id = bank
        .register_dense_2d(Dense2DMeta {
            gene_names: (0..edge_genes)
                .map(|idx| format!("edge_gene_{idx}"))
                .collect(),
            data: ArrayMeta {
                shape: vec![edge_cells, edge_genes],
                chunk_shape: vec![edge_chunk_rows, edge_chunk_cols],
                chunk_grid_shape: edge_grid.clone(),
                dtype: DType::U32,
                order: ArrayOrder::C,
                codec: ArrayCodecMeta::Uncompressed,
                chunks: ChunkStoreMeta::FileOffset {
                    path: edge_fo_path.clone(),
                    locations: edge_fo_locations,
                },
                variable_chunks: false,
                chunk_boundaries: None,
            },
        })
        .expect("register cropped edge fileoffset dataset");
    let mut edge_fo_out = vec![0u32; edge_selected.len() * edge_genes];
    bench(
        config,
        "databank/dense2d_fileoffset_cropped_edges_4x70",
        256,
        Some(edge_fo_out.len() * std::mem::size_of::<u32>()),
        || {
            bank.access_cells(edge_fo_id, &edge_selected, &mut edge_fo_out, None)
                .expect("access cropped edges");
            edge_fo_out.iter().fold(0usize, |acc, value| {
                acc ^ usize::try_from(*value).unwrap_or(0)
            })
        },
    );

    let zipstyle_path =
        support::bench_data_dir().join(format!("databank-zipstyle-{}.bin", std::process::id()));
    let zipstyle_locations = write_chunks_directory_offsets(&zipstyle_path, &padded_chunks);
    let zipstyle_id = bank
        .register_dense_2d(Dense2DMeta {
            gene_names: (0..edge_genes)
                .map(|idx| format!("edge_gene_{idx}"))
                .collect(),
            data: ArrayMeta {
                shape: vec![edge_cells, edge_genes],
                chunk_shape: vec![edge_chunk_rows, edge_chunk_cols],
                chunk_grid_shape: edge_grid.clone(),
                dtype: DType::U32,
                order: ArrayOrder::C,
                codec: ArrayCodecMeta::Uncompressed,
                chunks: ChunkStoreMeta::Directory {
                    locations: zipstyle_locations,
                },
                variable_chunks: false,
                chunk_boundaries: None,
            },
        })
        .expect("register zip-style directory dataset");
    let mut zipstyle_out = vec![0u32; edge_selected.len() * edge_genes];
    bench(
        config,
        "databank/dense2d_directory_offset_zipstyle_edges_4x70",
        256,
        Some(zipstyle_out.len() * std::mem::size_of::<u32>()),
        || {
            bank.access_cells(zipstyle_id, &edge_selected, &mut zipstyle_out, None)
                .expect("access zip-style directory");
            zipstyle_out.iter().fold(0usize, |acc, value| {
                acc ^ usize::try_from(*value).unwrap_or(0)
            })
        },
    );

    let rect_rows = vec![0usize, 16, edge_cells];
    let rect_cols = vec![0usize, 32, edge_genes];
    let rect_chunks =
        make_dense_u32_rectilinear_chunks(edge_cells, edge_genes, &rect_rows, &rect_cols);
    let rect_gene_names: Vec<String> = (0..edge_genes)
        .map(|idx| format!("edge_gene_{idx}"))
        .collect();
    let rect_grid = vec![rect_rows.len() - 1, rect_cols.len() - 1];
    let mut rect_bank = DataBank::new(DataBankConfig::default()).expect("rectilinear error bank");
    bench(
        config,
        "databank/error_dense2d_rectilinear_grid",
        2048,
        None,
        || {
            rect_bank
                .register_dense_2d(Dense2DMeta {
                    gene_names: rect_gene_names.clone(),
                    data: ArrayMeta {
                        shape: vec![edge_cells, edge_genes],
                        chunk_shape: vec![edge_chunk_rows, edge_chunk_cols],
                        chunk_grid_shape: rect_grid.clone(),
                        dtype: DType::U32,
                        order: ArrayOrder::C,
                        codec: ArrayCodecMeta::Uncompressed,
                        chunks: ChunkStoreMeta::Memory {
                            chunks: rect_chunks.clone(),
                        },
                        variable_chunks: true,
                        chunk_boundaries: Some(vec![rect_rows.clone(), rect_cols.clone()]),
                    },
                })
                .expect_err("dense2d rectilinear is unsupported")
                .to_string()
                .len()
        },
    );
    let _ = std::fs::remove_file(&edge_fo_path);
    let _ = std::fs::remove_file(&zipstyle_path);

    // by-gene-names: select a column subset by name, Zero / Error policies.
    let by_names: Vec<String> = (0..32)
        .map(|idx| format!("gene_{}", idx * 7 % genes))
        .collect();
    let mut by_out = vec![0u32; selected.len() * by_names.len()];
    bench(
        config,
        "databank/dense2d_by_gene_names_zero_32x32",
        192,
        Some(by_out.len() * std::mem::size_of::<u32>()),
        || {
            bank.access_cells_by_gene_names(
                id,
                &selected,
                &by_names,
                &mut by_out,
                None,
                MissingGenePolicy::Zero,
            )
            .expect("access by gene names");
            by_out.iter().step_by(131).fold(0usize, |acc, value| {
                acc ^ usize::try_from(*value).unwrap_or(0)
            })
        },
    );
    // MissingGenePolicy::Error on a present selection must succeed.
    bench(
        config,
        "databank/dense2d_by_gene_names_error_present_32x32",
        192,
        Some(by_out.len() * std::mem::size_of::<u32>()),
        || {
            bank.access_cells_by_gene_names(
                id,
                &selected,
                &by_names,
                &mut by_out,
                None,
                MissingGenePolicy::Error,
            )
            .expect("access by gene names error");
            by_out.iter().step_by(131).fold(0usize, |acc, value| {
                acc ^ usize::try_from(*value).unwrap_or(0)
            })
        },
    );
    let by_names_missing: Vec<String> = vec![
        "gene_1".to_string(),
        "missing_gene".to_string(),
        "gene_3".to_string(),
        "gene_5".to_string(),
    ];
    let mut by_missing_out = vec![0u32; selected.len() * by_names_missing.len()];
    bench(
        config,
        "databank/dense2d_by_gene_names_zero_missing_32x4",
        192,
        Some(by_missing_out.len() * std::mem::size_of::<u32>()),
        || {
            bank.access_cells_by_gene_names(
                id,
                &selected,
                &by_names_missing,
                &mut by_missing_out,
                None,
                MissingGenePolicy::Zero,
            )
            .expect("access by gene names zero missing");
            by_missing_out.iter().fold(0usize, |acc, value| {
                acc ^ usize::try_from(*value).unwrap_or(0)
            })
        },
    );
    bench(
        config,
        "databank/error_by_gene_names_missing",
        5000,
        None,
        || {
            bank.access_cells_by_gene_names(
                id,
                &selected[..1],
                &by_names_missing,
                &mut by_missing_out[..by_names_missing.len()],
                None,
                MissingGenePolicy::Error,
            )
            .expect_err("missing gene should error")
            .to_string()
            .len()
        },
    );

    // Owned / alloc output buffers (databank-allocated).
    bench(
        config,
        "databank/dense2d_owned_32x1024",
        128,
        Some(selected.len() * genes * std::mem::size_of::<u32>()),
        || {
            let out: Vec<u32> = bank.access_cells_owned(id, &selected).expect("owned");
            out.len() ^ out[0] as usize
        },
    );
    bench(
        config,
        "databank/dense2d_owned_with_config_32x1024",
        96,
        Some(selected.len() * genes * std::mem::size_of::<u32>()),
        || {
            let out: Vec<u32> = bank
                .access_cells_owned_with_config(id, &selected, tuned_access)
                .expect("owned with config");
            out.len() ^ out[0] as usize
        },
    );
    bench(
        config,
        "databank/dense2d_alloc_32x1024",
        128,
        Some(selected.len() * genes * std::mem::size_of::<u32>()),
        || {
            let out: Vec<u32> = bank.access_cells_alloc(id, &selected).expect("alloc");
            out.len() ^ out[0] as usize
        },
    );
    bench(
        config,
        "databank/dense2d_owned_by_gene_names_32x32",
        128,
        Some(selected.len() * by_names.len() * std::mem::size_of::<u32>()),
        || {
            let out: Vec<u32> = bank
                .access_cells_owned_by_gene_names(id, &selected, &by_names, MissingGenePolicy::Zero)
                .expect("owned by gene names");
            out.len() ^ out[0] as usize
        },
    );

    // prefetch_cells: warm raw bytes into the access cache.
    bench(
        config,
        "databank/prefetch_cells_dense2d_32",
        64,
        None,
        || {
            bank.prefetch_cells(id, &selected).expect("prefetch");
            selected.len()
        },
    );
    let scheduled_batches = vec![vec![0usize, 7, 14, 21], vec![3, 5], Vec::new()];
    bench(
        config,
        "databank/prefetch_scheduled_by_gene_names_zero_missing",
        64,
        Some(
            scheduled_batches
                .iter()
                .map(|batch| batch.len() * by_names_missing.len() * std::mem::size_of::<u32>())
                .sum(),
        ),
        || {
            let prefetch = bank
                .prefetch_cells_scheduled_by_gene_names::<u32, _, _>(
                    id,
                    scheduled_batches.clone(),
                    &by_names_missing,
                    MissingGenePolicy::Zero,
                    ScheduledPrefetchConfig::default(),
                )
                .expect("scheduled by gene names");
            prefetch.fold(0usize, |acc, batch| {
                let batch = batch.expect("scheduled batch");
                acc ^ batch.buffer.len() ^ batch.num_genes
            })
        },
    );
    bench(
        config,
        "databank/prefetch_scheduled_drop_after_first_batch",
        128,
        Some(scheduled_batches[0].len() * genes * std::mem::size_of::<u32>()),
        || {
            let mut prefetch = bank
                .prefetch_cells_scheduled::<u32, _>(
                    id,
                    scheduled_batches.clone(),
                    ScheduledPrefetchConfig::default(),
                )
                .expect("scheduled prefetch");
            let first = prefetch
                .next()
                .expect("first scheduled batch")
                .expect("first scheduled batch result");
            let checksum = first.buffer.len() ^ first.cells.len() ^ first.num_genes;
            drop(prefetch);
            checksum
        },
    );

    // dataset_genes: borrow gene-name views.
    bench(config, "databank/dataset_genes_len", 10_000, None, || {
        bank.dataset_genes(id).expect("dataset genes").len()
    });
    bench(
        config,
        "databank/dataset_meta_getters",
        20_000,
        None,
        || {
            bank.dataset_num_cells(id).expect("dataset num cells")
                ^ bank.dataset_num_genes(id).expect("dataset num genes")
                ^ bank.dataset_dtype(id).expect("dataset dtype").item_size()
        },
    );
    let _genes_view: &[GeneNameView] = bank.dataset_genes(id).expect("dataset genes");
    let _ = _genes_view;

    // register_dense alias + unregister lifecycle. `Dense2DMeta` is not
    // `Clone`, so rebuild the meta each iteration; the chunk bytes are shared
    // via `Arc` clones so only the registration churn is measured.
    let cycle_chunks = Arc::new(make_dense_u32_chunks(cells, genes, chunk_rows, chunk_cols));
    bench(
        config,
        "databank/register_dense_unregister_cycle",
        64,
        None,
        || {
            let cycle_id = bank
                .register_dense(Dense2DMeta {
                    gene_names: (0..genes).map(|idx| format!("gene_{idx}")).collect(),
                    data: ArrayMeta {
                        shape: vec![cells, genes],
                        chunk_shape: vec![chunk_rows, chunk_cols],
                        chunk_grid_shape: vec![cells / chunk_rows, genes / chunk_cols],
                        dtype: DType::U32,
                        order: ArrayOrder::C,
                        codec: ArrayCodecMeta::Uncompressed,
                        chunks: ChunkStoreMeta::Memory {
                            chunks: (*cycle_chunks).clone(),
                        },
                        variable_chunks: false,
                        chunk_boundaries: None,
                    },
                })
                .expect("register_dense alias");
            bank.unregister(cycle_id).expect("unregister");
            cycle_id.slot as usize
        },
    );
    let lifecycle_cells = 64usize;
    let lifecycle_genes = 64usize;
    let lifecycle_chunk_rows = 16usize;
    let lifecycle_chunk_cols = 16usize;
    let lifecycle_chunks = Arc::new(make_dense_u32_chunks(
        lifecycle_cells,
        lifecycle_genes,
        lifecycle_chunk_rows,
        lifecycle_chunk_cols,
    ));
    bench(
        config,
        "databank/lifecycle_create_register_drop_cycle",
        32,
        None,
        || {
            let mut local_bank = DataBank::new(DataBankConfig::default()).expect("local bank");
            let local_id = local_bank
                .register_dense_2d(Dense2DMeta {
                    gene_names: (0..lifecycle_genes)
                        .map(|idx| format!("life_gene_{idx}"))
                        .collect(),
                    data: ArrayMeta {
                        shape: vec![lifecycle_cells, lifecycle_genes],
                        chunk_shape: vec![lifecycle_chunk_rows, lifecycle_chunk_cols],
                        chunk_grid_shape: vec![
                            lifecycle_cells / lifecycle_chunk_rows,
                            lifecycle_genes / lifecycle_chunk_cols,
                        ],
                        dtype: DType::U32,
                        order: ArrayOrder::C,
                        codec: ArrayCodecMeta::Uncompressed,
                        chunks: ChunkStoreMeta::Memory {
                            chunks: (*lifecycle_chunks).clone(),
                        },
                        variable_chunks: false,
                        chunk_boundaries: None,
                    },
                })
                .expect("register lifecycle dataset");
            let mut local_out = vec![0u32; 4 * lifecycle_genes];
            local_bank
                .access_cells(local_id, &[0, 7, 31, 63], &mut local_out, None)
                .expect("lifecycle access");
            local_out[0] as usize ^ local_out.len()
        },
    );

    // Multi-dtype dense2d access: 1/2/4/8-byte integer and float dtypes.
    bench_typed_dense2d::<u8>(config, DType::U8, "u8");
    bench_typed_dense2d::<i8>(config, DType::I8, "i8");
    bench_typed_dense2d::<u16>(config, DType::U16, "u16");
    bench_typed_dense2d::<i16>(config, DType::I16, "i16");
    bench_typed_dense2d::<u32>(config, DType::U32, "u32");
    bench_typed_dense2d::<u64>(config, DType::U64, "u64");
    bench_typed_dense2d::<i32>(config, DType::I32, "i32");
    bench_typed_dense2d::<i64>(config, DType::I64, "i64");
    bench_typed_dense2d::<f32>(config, DType::F32, "f32");
    bench_typed_dense2d::<f64>(config, DType::F64, "f64");
    bench_typed_dense2d::<F16Bits>(config, DType::F16, "f16");
    bench_typed_dense2d::<Bf16Bits>(config, DType::BF16, "bf16");

    // Config validation surface for the databank facade.
    bench(config, "databank/config_validate", 10_000, None, || {
        DataBankConfig::default().validate().is_ok() as usize
    });
    bench(
        config,
        "databank/fill_config_validate",
        20_000,
        None,
        || FillConfig::default().validate().is_ok() as usize,
    );
    bench(
        config,
        "databank/scheduled_prefetch_config_validate",
        20_000,
        None,
        || ScheduledPrefetchConfig::default().validate().is_ok() as usize,
    );
    bench(
        config,
        "databank/error_cell_out_of_range",
        10_000,
        None,
        || {
            bank.access_cells(id, &[cells], &mut out[..genes], None)
                .expect_err("cell out of range")
                .to_string()
                .len()
        },
    );
    bench(
        config,
        "databank/error_dtype_mismatch",
        10_000,
        None,
        || {
            let mut wrong = vec![0f32; genes];
            bank.access_cells(id, &[0], &mut wrong, None)
                .expect_err("dtype mismatch")
                .to_string()
                .len()
        },
    );
    bench(
        config,
        "databank/error_output_len_mismatch",
        10_000,
        None,
        || {
            let mut short = vec![0u32; genes - 1];
            bank.access_cells(id, &[0], &mut short, None)
                .expect_err("output length mismatch")
                .to_string()
                .len()
        },
    );

    let _ = std::fs::remove_file(fo_path);
}

// ---------------------------------------------------------------------------
// pybind: Rust-side PyO3 binding coverage. This does not call the Python
// wrapper layer; it registers the PyO3 classes/functions into an in-process
// module and exercises the Python-facing fast paths from the Rust bench target.
// ---------------------------------------------------------------------------

#[cfg(feature = "pybind-bench")]
fn bench_pybind(config: BenchConfig) {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let module = PyModule::new(py, "_scdata_bench").expect("create pybind bench module");
        _scdata::pybind::register(&module).expect("register pybind module");
        let fixture = make_pybind_fixture(py, &module);

        bench(config, "pybind/register_module_classes", 1000, None, || {
            let module = PyModule::new(py, "_scdata_bench_iter").expect("create module");
            _scdata::pybind::register(&module).expect("register module");
            module.dict().len()
        });

        bench(
            config,
            "pybind/config_get_set_roundtrip",
            5000,
            None,
            || {
                let cfg = module
                    .getattr("_DataBankConfig")
                    .expect("DataBankConfig class")
                    .call0()
                    .expect("DataBankConfig()");
                let access = cfg.getattr("access_config").expect("access_config");
                let queue_capacity: usize = access
                    .getattr("queue_capacity")
                    .expect("queue_capacity")
                    .extract()
                    .expect("extract queue_capacity");
                access
                    .setattr("queue_capacity", queue_capacity)
                    .expect("set queue_capacity");
                queue_capacity
            },
        );

        bench(config, "pybind/register_dense_fileoffset", 64, None, || {
            let bank = module
                .getattr("_DataBank")
                .expect("DataBank class")
                .call0()
                .expect("DataBank()");
            let id = bank
                .call_method1("register_dense", (fixture.dataset.clone_ref(py), ""))
                .expect("register_dense");
            bank.call_method1("dataset_num_cells", (id,))
                .expect("dataset_num_cells")
                .extract::<usize>()
                .expect("extract cells")
        });

        bench(
            config,
            "pybind/access_cells_vec_16x64",
            256,
            Some(fixture.cells.len() * fixture.genes * std::mem::size_of::<u32>()),
            || {
                let arr = fixture
                    .bank
                    .call_method1(
                        py,
                        "access_cells",
                        (fixture.id.clone_ref(py), fixture.cells.clone()),
                    )
                    .expect("access_cells");
                arr.bind(py).len().expect("numpy len")
            },
        );

        bench(
            config,
            "pybind/access_cells_array_16x64",
            256,
            Some(fixture.cells.len() * fixture.genes * std::mem::size_of::<u32>()),
            || {
                let arr = fixture
                    .bank
                    .call_method1(
                        py,
                        "access_cells_array",
                        (fixture.id.clone_ref(py), fixture.cells_array.clone_ref(py)),
                    )
                    .expect("access_cells_array");
                arr.bind(py).len().expect("numpy len")
            },
        );

        bench(
            config,
            "pybind/access_cells_by_gene_names_array_zero_missing",
            192,
            Some(fixture.cells.len() * fixture.by_names.len() * std::mem::size_of::<u32>()),
            || {
                let arr = fixture
                    .bank
                    .call_method1(
                        py,
                        "access_cells_by_gene_names_array",
                        (
                            fixture.id.clone_ref(py),
                            fixture.cells_array.clone_ref(py),
                            fixture.by_names.clone(),
                        ),
                    )
                    .expect("access_cells_by_gene_names_array");
                arr.bind(py).len().expect("numpy len")
            },
        );

        bench(config, "pybind/prefetch_cells_lists", 128, None, || {
            let prefetch = fixture
                .bank
                .call_method1(
                    py,
                    "prefetch_cells",
                    (fixture.id.clone_ref(py), fixture.batches.clone_ref(py)),
                )
                .expect("prefetch_cells");
            drain_py_prefetch(py, &prefetch)
        });

        bench(config, "pybind/prefetch_cells_arrays", 128, None, || {
            let prefetch = fixture
                .bank
                .call_method1(
                    py,
                    "prefetch_cells_arrays",
                    (
                        fixture.id.clone_ref(py),
                        fixture.batches_arrays.clone_ref(py),
                    ),
                )
                .expect("prefetch_cells_arrays");
            drain_py_prefetch(py, &prefetch)
        });

        bench(
            config,
            "pybind/prefetch_cells_by_gene_names_arrays_zero_missing",
            96,
            None,
            || {
                let prefetch = fixture
                    .bank
                    .call_method1(
                        py,
                        "prefetch_cells_by_gene_names_arrays",
                        (
                            fixture.id.clone_ref(py),
                            fixture.batches_arrays.clone_ref(py),
                            fixture.by_names.clone(),
                        ),
                    )
                    .expect("prefetch_cells_by_gene_names_arrays");
                drain_py_prefetch(py, &prefetch)
            },
        );

        bench(
            config,
            "pybind/decode_index_payload_u32",
            1024,
            None,
            || {
                let payload = PyBytes::new(py, &fixture.index_payload);
                let arr = fixture
                    .decode_index_payload
                    .call1(
                        py,
                        (
                            payload,
                            fixture.index_offsets.clone(),
                            fixture.index_lengths.clone(),
                            fixture.dtype.clone_ref(py),
                            fixture.codec.clone_ref(py),
                            fixture.index_count,
                        ),
                    )
                    .expect("decode_index_payload");
                arr.bind(py).len().expect("numpy len")
            },
        );

        bench(config, "pybind/decode_index_chunks_u32", 1024, None, || {
            let first = PyBytes::new(
                py,
                &fixture.index_payload[..fixture.index_payload.len() / 2],
            );
            let second = PyBytes::new(
                py,
                &fixture.index_payload[fixture.index_payload.len() / 2..],
            );
            let chunks = PyList::empty(py);
            chunks.append(first).expect("append first index chunk");
            chunks.append(second).expect("append second index chunk");
            let arr = fixture
                .decode_index_chunks
                .call1(
                    py,
                    (
                        chunks,
                        fixture.dtype.clone_ref(py),
                        fixture.codec.clone_ref(py),
                        fixture.index_count,
                    ),
                )
                .expect("decode_index_chunks");
            arr.bind(py).len().expect("numpy len")
        });

        let _ = std::fs::remove_file(&fixture.path);
    });
}

#[cfg(feature = "pybind-bench")]
struct PyBindFixture {
    path: std::path::PathBuf,
    bank: Py<PyAny>,
    id: Py<PyAny>,
    dataset: Py<PyAny>,
    dtype: Py<PyAny>,
    codec: Py<PyAny>,
    cells: Vec<usize>,
    cells_array: Py<PyAny>,
    batches: Py<PyAny>,
    batches_arrays: Py<PyAny>,
    by_names: Vec<String>,
    genes: usize,
    decode_index_payload: Py<PyAny>,
    decode_index_chunks: Py<PyAny>,
    index_payload: Vec<u8>,
    index_offsets: Vec<u64>,
    index_lengths: Vec<u64>,
    index_count: usize,
}

#[cfg(feature = "pybind-bench")]
fn make_pybind_fixture<'py>(py: Python<'py>, module: &Bound<'py, PyModule>) -> PyBindFixture {
    let cells = 128usize;
    let genes = 64usize;
    let chunk_rows = 16usize;
    let chunk_cols = 16usize;
    let path = support::bench_data_dir().join(format!("pybind-dense-{}.bin", std::process::id()));
    let locations = write_dense_u32_file(&path, cells, genes, chunk_rows, chunk_cols);
    let offsets = locations.iter().map(|loc| loc.offset).collect::<Vec<_>>();
    let lengths = locations
        .iter()
        .map(|loc| loc.len as u64)
        .collect::<Vec<_>>();
    let dtype = py_dtype(py, "u32");
    let codec = py_uncompressed_codec(py);

    let data = py_namespace(py);
    data.setattr("shape", vec![cells, genes])
        .expect("set shape");
    data.setattr("chunk_shape", vec![chunk_rows, chunk_cols])
        .expect("set chunk_shape");
    data.setattr("dtype", dtype.clone_ref(py))
        .expect("set dtype");
    data.setattr("codec", codec.clone_ref(py))
        .expect("set codec");
    data.setattr("store_kind", "file").expect("set store_kind");
    data.setattr("payload_path", "").expect("set payload_path");
    data.setattr("payload_file_path", path.to_string_lossy().to_string())
        .expect("set payload_file_path");
    data.setattr("chunk_offsets", offsets)
        .expect("set chunk_offsets");
    data.setattr("chunk_lengths", lengths)
        .expect("set chunk_lengths");
    data.setattr("variable_chunks", false)
        .expect("set variable_chunks");

    let dataset = py_namespace(py);
    dataset
        .setattr(
            "gene_names",
            (0..genes)
                .map(|idx| format!("gene_{idx}"))
                .collect::<Vec<_>>(),
        )
        .expect("set gene_names");
    dataset.setattr("data", data).expect("set data");
    let dataset = dataset.unbind();

    let bank = module
        .getattr("_DataBank")
        .expect("DataBank class")
        .call0()
        .expect("DataBank()")
        .unbind();
    let id = bank
        .call_method1(py, "register_dense", (dataset.clone_ref(py), ""))
        .expect("register_dense");
    let cells_vec = (0..16).map(|idx| idx * 7 % cells).collect::<Vec<_>>();
    let cells_array = cells_vec
        .iter()
        .map(|&cell| cell as isize)
        .collect::<Vec<_>>()
        .into_pyarray(py)
        .into_any()
        .unbind();
    let batches = PyList::new(
        py,
        [
            vec![0usize, 7, 14, 21],
            vec![3usize, 5],
            Vec::<usize>::new(),
        ],
    )
    .expect("batches list")
    .into_any()
    .unbind();
    let batches_arrays = PyList::empty(py);
    for batch in [
        vec![0isize, 7, 14, 21],
        vec![3isize, 5],
        Vec::<isize>::new(),
    ] {
        batches_arrays
            .append(batch.into_pyarray(py))
            .expect("append batch array");
    }
    let batches_arrays = batches_arrays.into_any().unbind();
    let by_names = vec![
        "gene_1".to_string(),
        "missing_gene".to_string(),
        "gene_3".to_string(),
        "gene_5".to_string(),
    ];
    let decode_index_payload = module
        .getattr("_decode_index_payload")
        .expect("decode index payload fn")
        .unbind();
    let decode_index_chunks = module
        .getattr("_decode_index_chunks")
        .expect("decode index chunks fn")
        .unbind();
    let index_count = 32usize;
    let mut index_payload = Vec::with_capacity(index_count * std::mem::size_of::<u32>());
    for value in 0..index_count as u32 {
        index_payload.extend_from_slice(&value.to_ne_bytes());
    }
    let half = (index_payload.len() / 2) as u64;

    PyBindFixture {
        path,
        bank,
        id,
        dataset,
        dtype,
        codec,
        cells: cells_vec,
        cells_array,
        batches,
        batches_arrays,
        by_names,
        genes,
        decode_index_payload,
        decode_index_chunks,
        index_payload,
        index_offsets: vec![0, half],
        index_lengths: vec![half, half],
        index_count,
    }
}

#[cfg(feature = "pybind-bench")]
fn py_namespace<'py>(py: Python<'py>) -> Bound<'py, PyAny> {
    py.import("types")
        .expect("import types")
        .getattr("SimpleNamespace")
        .expect("SimpleNamespace")
        .call0()
        .expect("SimpleNamespace()")
}

#[cfg(feature = "pybind-bench")]
fn py_dtype(py: Python<'_>, code: &str) -> Py<PyAny> {
    let dtype = py_namespace(py);
    dtype.setattr("value", code).expect("set dtype value");
    dtype.unbind()
}

#[cfg(feature = "pybind-bench")]
fn py_uncompressed_codec(py: Python<'_>) -> Py<PyAny> {
    let codec = py_namespace(py);
    codec
        .setattr("is_uncompressed", true)
        .expect("set codec flag");
    codec.unbind()
}

#[cfg(feature = "pybind-bench")]
fn drain_py_prefetch(py: Python<'_>, prefetch: &Py<PyAny>) -> usize {
    let mut sum = 0usize;
    loop {
        let item = prefetch
            .call_method0(py, "__next__")
            .expect("prefetch __next__");
        let item = item.bind(py);
        if item.is_none() {
            break;
        }
        sum ^= item.len().expect("prefetch tuple len");
    }
    sum
}

/// Deterministic dense2D chunk bytes for an arbitrary item size, so the
/// multi-dtype bench can stay generic over `T: DataValue`.
fn make_dense_bytes(
    cells: usize,
    genes: usize,
    chunk_rows: usize,
    chunk_cols: usize,
    item_size: usize,
) -> Vec<Arc<[u8]>> {
    let row_chunks = cells / chunk_rows;
    let col_chunks = genes / chunk_cols;
    let chunk_bytes = chunk_rows * chunk_cols * item_size;
    let mut out = Vec::with_capacity(row_chunks * col_chunks);
    for row_chunk in 0..row_chunks {
        for col_chunk in 0..col_chunks {
            let tag = (row_chunk * col_chunks + col_chunk) as u8;
            let bytes = (0..chunk_bytes)
                .map(|b| tag.wrapping_add(b as u8))
                .collect::<Vec<u8>>();
            out.push(Arc::from(bytes.into_boxed_slice()));
        }
    }
    out
}

fn make_dense_u32_rectilinear_chunks(
    cells: usize,
    genes: usize,
    row_boundaries: &[usize],
    col_boundaries: &[usize],
) -> Vec<Arc<[u8]>> {
    assert_eq!(row_boundaries.first().copied(), Some(0));
    assert_eq!(row_boundaries.last().copied(), Some(cells));
    assert_eq!(col_boundaries.first().copied(), Some(0));
    assert_eq!(col_boundaries.last().copied(), Some(genes));

    let mut chunks = Vec::with_capacity((row_boundaries.len() - 1) * (col_boundaries.len() - 1));
    for row_window in row_boundaries.windows(2) {
        for col_window in col_boundaries.windows(2) {
            let row_start = row_window[0];
            let row_end = row_window[1];
            let col_start = col_window[0];
            let col_end = col_window[1];
            let mut bytes = Vec::with_capacity(
                (row_end - row_start) * (col_end - col_start) * std::mem::size_of::<u32>(),
            );
            for cell in row_start..row_end {
                for gene in col_start..col_end {
                    let value = ((cell as u32) << 16) ^ gene as u32;
                    bytes.extend_from_slice(&value.to_ne_bytes());
                }
            }
            chunks.push(Arc::from(bytes.into_boxed_slice()));
        }
    }
    chunks
}

fn bench_sparse_csr_index_dtype(
    config: BenchConfig,
    label: &str,
    index_dtype: DType,
    chunks: (Vec<u64>, Arc<[u8]>, Arc<[u8]>),
    cells: usize,
    genes: usize,
    nnz_per_cell: usize,
) {
    let (indptr, indices_chunk, data_chunk) = chunks;
    let nnz = cells * nnz_per_cell;
    let mut bank = DataBank::new(DataBankConfig::default()).expect("csr index bank");
    let id = bank
        .register_sparse_csr(SparseCsrDatasetMeta {
            gene_names: (0..genes).map(|idx| format!("gene_{idx}")).collect(),
            indptr,
            indices: ArrayMeta {
                shape: vec![nnz],
                chunk_shape: vec![nnz],
                chunk_grid_shape: vec![1],
                dtype: index_dtype,
                order: ArrayOrder::C,
                codec: ArrayCodecMeta::Uncompressed,
                chunks: ChunkStoreMeta::Memory {
                    chunks: vec![indices_chunk],
                },
                variable_chunks: false,
                chunk_boundaries: None,
            },
            data: ArrayMeta {
                shape: vec![nnz],
                chunk_shape: vec![nnz],
                chunk_grid_shape: vec![1],
                dtype: DType::F32,
                order: ArrayOrder::C,
                codec: ArrayCodecMeta::Uncompressed,
                chunks: ChunkStoreMeta::Memory {
                    chunks: vec![data_chunk],
                },
                variable_chunks: false,
                chunk_boundaries: None,
            },
            index_dtype,
            num_cells: cells,
            num_genes: genes,
        })
        .expect("register csr index dataset");
    let selected = (0..32).map(|idx| idx * 3 % cells).collect::<Vec<_>>();
    let mut out = vec![0.0f32; selected.len() * genes];
    bench(
        config,
        &format!("databank/sparse_csr_index_{label}_checked_32x256"),
        256,
        Some(out.len() * std::mem::size_of::<f32>()),
        || {
            bank.access_cells(id, &selected, &mut out, None)
                .expect("access csr index dtype");
            out.iter()
                .step_by(251)
                .fold(0usize, |acc, value| acc ^ value.to_bits() as usize)
        },
    );
}

fn bench_typed_dense2d<T: DataValue>(config: BenchConfig, dtype: DType, label: &str) {
    let cells = 256usize;
    let genes = 128usize;
    let chunk_rows = 16usize;
    let chunk_cols = 32usize;
    let item_size = std::mem::size_of::<T>();
    let chunks = make_dense_bytes(cells, genes, chunk_rows, chunk_cols, item_size);
    let mut bank = DataBank::new(DataBankConfig::default()).expect("typed bank");
    let id = bank
        .register_dense_2d(Dense2DMeta {
            gene_names: (0..genes).map(|idx| format!("g{idx}")).collect(),
            data: ArrayMeta {
                shape: vec![cells, genes],
                chunk_shape: vec![chunk_rows, chunk_cols],
                chunk_grid_shape: vec![cells / chunk_rows, genes / chunk_cols],
                dtype,
                order: ArrayOrder::C,
                codec: ArrayCodecMeta::Uncompressed,
                chunks: ChunkStoreMeta::Memory { chunks },
                variable_chunks: false,
                chunk_boundaries: None,
            },
        })
        .expect("register typed dense dataset");
    let selected: Vec<usize> = (0..16).map(|i| i * 7 % cells).collect();
    let mut out: Vec<T> = vec![T::zero(); selected.len() * genes];
    bench(
        config,
        &format!("databank/dense2d_{label}_access_16x128"),
        256,
        Some(out.len() * item_size),
        || {
            bank.access_cells(id, &selected, &mut out, None)
                .expect("access cells");
            let bytes = unsafe {
                std::slice::from_raw_parts(out.as_ptr() as *const u8, out.len() * item_size)
            };
            bytes
                .iter()
                .step_by(64)
                .fold(0usize, |acc, byte| acc ^ *byte as usize)
        },
    );
}

// ---------------------------------------------------------------------------
// Normalized data pipeline: generate raw → encode → decode round-trip across
// dtype × distribution × missing-rate, exercising the support::data generator
// and support::codecs encode_for_spec dispatch end to end.
// ---------------------------------------------------------------------------

fn bench_data_pipeline(config: BenchConfig) {
    // Generator cost: raw payload synthesis for every (dtype, dist, missing).
    for dist in DataDist::ALL {
        for &missing in &[0u32, 250, 500, 950] {
            let profile = DataProfile {
                dtype: DType::U32,
                dist,
                missing_permille: missing,
                chunk_bytes: 256 * 1024,
                num_chunks: 4,
                seed: 17,
            };
            let label = format!("data/gen_{}", profile.label());
            bench(config, &label, 32, Some(profile.total_bytes()), || {
                let raw = profile.generate();
                raw.len() ^ raw[raw.len() / 2] as usize
            });
        }
    }

    // Encode + decode round-trip for one payload across the codec matrix.
    let profile = DataProfile {
        dtype: DType::U32,
        dist: DataDist::Uniform,
        missing_permille: 250,
        chunk_bytes: 256 * 1024,
        num_chunks: 4,
        seed: 17,
    };
    let raw = profile.generate();
    let codec_specs = support::codecs::default_codec_matrix();
    for (name, spec) in &codec_specs {
        let encoded = encode_for_spec(spec, &raw);
        let codec = spec.build();
        let label = format!("data/roundtrip_{}_{}", name, profile.label());
        bench(config, &label, 64, Some(raw.len()), || {
            decode_into_checksum(&codec, &encoded, raw.len())
        });
    }

    // PRNG throughput baseline (splitmix64).
    bench(config, "data/prng_splitmix64_1m", 64, Some(1 << 20), || {
        let mut rng = Rng::new(17);
        let mut acc = 0u64;
        for _ in 0..(1 << 14) {
            acc ^= rng.next_u64();
        }
        acc as usize
    });
}

// ---------------------------------------------------------------------------
// Missing-rate sweep: dense2d and sparse-csr access as the zero/NaN fraction
// varies. Exercises the scatter path with progressively sparser payloads.
// ---------------------------------------------------------------------------

fn bench_missing_rate(config: BenchConfig) {
    let cells = 512usize;
    let genes = 256usize;
    let chunk_rows = 16usize;
    let chunk_cols = 32usize;
    let item_size = std::mem::size_of::<u32>();

    for &missing in &[0u32, 100, 250, 500, 750, 950] {
        // Dense2D with missing-rate-adjusted raw chunks: the generator fills
        // the missing fraction with zeros, so scatter copies increasingly
        // sparse data.
        let chunks: Vec<Arc<[u8]>> = (0..(cells / chunk_rows * genes / chunk_cols))
            .map(|chunk_idx| {
                let profile = DataProfile {
                    dtype: DType::U32,
                    dist: DataDist::Uniform,
                    missing_permille: missing,
                    chunk_bytes: chunk_rows * chunk_cols * item_size,
                    num_chunks: 1,
                    seed: 17 + chunk_idx as u64,
                };
                Arc::from(profile.generate().into_boxed_slice())
            })
            .collect();
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = bank
            .register_dense_2d(Dense2DMeta {
                gene_names: (0..genes).map(|idx| format!("g{idx}")).collect(),
                data: ArrayMeta {
                    shape: vec![cells, genes],
                    chunk_shape: vec![chunk_rows, chunk_cols],
                    chunk_grid_shape: vec![cells / chunk_rows, genes / chunk_cols],
                    dtype: DType::U32,
                    order: ArrayOrder::C,
                    codec: ArrayCodecMeta::Uncompressed,
                    chunks: ChunkStoreMeta::Memory { chunks },
                    variable_chunks: false,
                    chunk_boundaries: None,
                },
            })
            .expect("register dense2d missing-rate dataset");
        let selected: Vec<usize> = (0..32).map(|i| i * 7 % cells).collect();
        let mut out = vec![0u32; selected.len() * genes];
        bench(
            config,
            &format!("missing/dense2d_u32_miss{missing}_32x256"),
            128,
            Some(out.len() * item_size),
            || {
                bank.access_cells(id, &selected, &mut out, None)
                    .expect("access");
                out.iter()
                    .step_by(131)
                    .fold(0usize, |acc, v| acc ^ usize::try_from(*v).unwrap_or(0))
            },
        );

        // Sparse CSR with the same missing rate: nnz_per_cell scales down as
        // missing goes up, so this isolates scatter cost from payload density.
        let nnz_per_cell = ((genes as f32) * (1.0 - missing as f32 / 1000.0)).ceil() as usize;
        let nnz_per_cell = nnz_per_cell.max(1);
        let (indptr, indices_chunk, data_chunk) =
            make_csr_u32_f32_chunks(cells, genes, nnz_per_cell);
        let csr_id = bank
            .register_sparse_csr(SparseCsrDatasetMeta {
                gene_names: (0..genes).map(|idx| format!("g{idx}")).collect(),
                indptr,
                indices: ArrayMeta {
                    shape: vec![cells * nnz_per_cell],
                    chunk_shape: vec![cells * nnz_per_cell],
                    chunk_grid_shape: vec![1],
                    dtype: DType::U32,
                    order: ArrayOrder::C,
                    codec: ArrayCodecMeta::Uncompressed,
                    chunks: ChunkStoreMeta::Memory {
                        chunks: vec![indices_chunk],
                    },
                    variable_chunks: false,
                    chunk_boundaries: None,
                },
                data: ArrayMeta {
                    shape: vec![cells * nnz_per_cell],
                    chunk_shape: vec![cells * nnz_per_cell],
                    chunk_grid_shape: vec![1],
                    dtype: DType::F32,
                    order: ArrayOrder::C,
                    codec: ArrayCodecMeta::Uncompressed,
                    chunks: ChunkStoreMeta::Memory {
                        chunks: vec![data_chunk],
                    },
                    variable_chunks: false,
                    chunk_boundaries: None,
                },
                index_dtype: DType::U32,
                num_cells: cells,
                num_genes: genes,
            })
            .expect("register sparse csr missing-rate dataset");
        let csr_selected: Vec<usize> = (0..32).map(|i| i * 3 % cells).collect();
        let mut csr_out = vec![0.0f32; csr_selected.len() * genes];
        bench(
            config,
            &format!("missing/sparse_csr_miss{missing}_nnz{nnz_per_cell}_32x256"),
            128,
            Some(csr_out.len() * std::mem::size_of::<f32>()),
            || {
                bank.access_cells(csr_id, &csr_selected, &mut csr_out, None)
                    .expect("access");
                csr_out
                    .iter()
                    .step_by(251)
                    .fold(0usize, |acc, v| acc ^ v.to_bits() as usize)
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Scale sweep: chunk size, cell count, and gene count scaling for dense2d
// memory access. Locates where per-chunk overhead vs scatter copy dominates.
// ---------------------------------------------------------------------------

fn bench_scale_sweep(config: BenchConfig) {
    let genes = 1024usize;
    let selected: Vec<usize> = (0..32).map(|i| i * 7 % 4096).collect();

    // Chunk-size sweep (cells fixed at 1024, vary chunk_cols).
    for &chunk_cols in &[16usize, 64, 128, 256, 1024] {
        let cells = 1024usize;
        let chunk_rows = 16usize;
        let chunks = make_dense_u32_chunks(cells, genes, chunk_rows, chunk_cols);
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = bank
            .register_dense_2d(Dense2DMeta {
                gene_names: (0..genes).map(|idx| format!("g{idx}")).collect(),
                data: ArrayMeta {
                    shape: vec![cells, genes],
                    chunk_shape: vec![chunk_rows, chunk_cols],
                    chunk_grid_shape: vec![cells / chunk_rows, genes / chunk_cols],
                    dtype: DType::U32,
                    order: ArrayOrder::C,
                    codec: ArrayCodecMeta::Uncompressed,
                    chunks: ChunkStoreMeta::Memory { chunks },
                    variable_chunks: false,
                    chunk_boundaries: None,
                },
            })
            .expect("register scale dataset");
        let mut out = vec![0u32; selected.len() * genes];
        bench(
            config,
            &format!("scale/dense2d_chunkcols{chunk_cols}_32x1024"),
            128,
            Some(out.len() * 4),
            || {
                bank.access_cells(id, &selected, &mut out, None)
                    .expect("access");
                out.iter()
                    .step_by(257)
                    .fold(0usize, |acc, v| acc ^ usize::try_from(*v).unwrap_or(0))
            },
        );
    }

    // Cell-count sweep (genes fixed, vary cells; output scales with selected).
    for &cells in &[256usize, 1024, 4096, 16384] {
        let chunk_rows = 16usize;
        let chunk_cols = 128usize;
        let chunks = make_dense_u32_chunks(cells, genes, chunk_rows, chunk_cols);
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = bank
            .register_dense_2d(Dense2DMeta {
                gene_names: (0..genes).map(|idx| format!("g{idx}")).collect(),
                data: ArrayMeta {
                    shape: vec![cells, genes],
                    chunk_shape: vec![chunk_rows, chunk_cols],
                    chunk_grid_shape: vec![cells / chunk_rows, genes / chunk_cols],
                    dtype: DType::U32,
                    order: ArrayOrder::C,
                    codec: ArrayCodecMeta::Uncompressed,
                    chunks: ChunkStoreMeta::Memory { chunks },
                    variable_chunks: false,
                    chunk_boundaries: None,
                },
            })
            .expect("register cell-scale dataset");
        let sel: Vec<usize> = (0..32).map(|i| i * 7 % cells).collect();
        let mut out = vec![0u32; sel.len() * genes];
        bench(
            config,
            &format!("scale/dense2d_cells{cells}_32x1024"),
            128,
            Some(out.len() * 4),
            || {
                bank.access_cells(id, &sel, &mut out, None).expect("access");
                out.iter()
                    .step_by(257)
                    .fold(0usize, |acc, v| acc ^ usize::try_from(*v).unwrap_or(0))
            },
        );
    }

    // Gene-count sweep (cells fixed, vary genes).
    for &g in &[64usize, 256, 1024, 4096] {
        let cells = 1024usize;
        let chunk_rows = 16usize;
        let chunk_cols = g.min(128);
        let chunks = make_dense_u32_chunks(cells, g, chunk_rows, chunk_cols);
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let id = bank
            .register_dense_2d(Dense2DMeta {
                gene_names: (0..g).map(|idx| format!("g{idx}")).collect(),
                data: ArrayMeta {
                    shape: vec![cells, g],
                    chunk_shape: vec![chunk_rows, chunk_cols],
                    chunk_grid_shape: vec![cells / chunk_rows, g / chunk_cols],
                    dtype: DType::U32,
                    order: ArrayOrder::C,
                    codec: ArrayCodecMeta::Uncompressed,
                    chunks: ChunkStoreMeta::Memory { chunks },
                    variable_chunks: false,
                    chunk_boundaries: None,
                },
            })
            .expect("register gene-scale dataset");
        let sel: Vec<usize> = (0..32).map(|i| i * 7 % cells).collect();
        let mut out = vec![0u32; sel.len() * g];
        bench(
            config,
            &format!("scale/dense2d_genes{g}_32x{}", g),
            128,
            Some(out.len() * 4),
            || {
                bank.access_cells(id, &sel, &mut out, None).expect("access");
                out.iter()
                    .step_by(257)
                    .fold(0usize, |acc, v| acc ^ usize::try_from(*v).unwrap_or(0))
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Scenario matrix: dense2d × store (memory/file/directory) × codec
// (none/zstd/lz4) × missing-rate. Cross-product of the axes that matter for
// end-to-end databank access.
// ---------------------------------------------------------------------------

fn bench_scenario_matrix(config: BenchConfig) {
    let cells = 512usize;
    let genes = 256usize;
    let chunk_rows = 16usize;
    let chunk_cols = 32usize;
    let item_size = std::mem::size_of::<u32>();
    let selected: Vec<usize> = (0..32).map(|i| i * 7 % cells).collect();
    let chunk_grid = vec![cells / chunk_rows, genes / chunk_cols];

    let codec_specs: &[(&str, ArrayCodecMeta)] = &[
        ("none", ArrayCodecMeta::Uncompressed),
        (
            "zstd",
            ArrayCodecMeta::CodecJson(r#"{"id":"zstd","level":3}"#.to_string()),
        ),
        (
            "lz4",
            ArrayCodecMeta::CodecJson(r#"{"id":"lz4"}"#.to_string()),
        ),
    ];

    for &missing in &[0u32, 500] {
        // Pre-generate raw chunks once per missing rate; stores share them.
        let raw_chunks: Vec<Vec<u8>> = (0..(chunk_grid[0] * chunk_grid[1]))
            .map(|chunk_idx| {
                let profile = DataProfile {
                    dtype: DType::U32,
                    dist: DataDist::Uniform,
                    missing_permille: missing,
                    chunk_bytes: chunk_rows * chunk_cols * item_size,
                    num_chunks: 1,
                    seed: 17 + chunk_idx as u64,
                };
                profile.generate()
            })
            .collect();

        // Memory store × codec.
        for (codec_name, codec_meta) in codec_specs {
            let chunks: Vec<Arc<[u8]>> = raw_chunks
                .iter()
                .map(|raw| {
                    let spec = match codec_meta {
                        ArrayCodecMeta::Uncompressed => CodecSpec::None,
                        ArrayCodecMeta::CodecJson(json) => {
                            CodecSpec::from_json_str(json).expect("parse codec")
                        }
                        _ => unreachable!(),
                    };
                    Arc::from(encode_for_spec(&spec, raw).into_boxed_slice())
                })
                .collect();
            let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
            let id = bank
                .register_dense_2d(Dense2DMeta {
                    gene_names: (0..genes).map(|idx| format!("g{idx}")).collect(),
                    data: ArrayMeta {
                        shape: vec![cells, genes],
                        chunk_shape: vec![chunk_rows, chunk_cols],
                        chunk_grid_shape: chunk_grid.clone(),
                        dtype: DType::U32,
                        order: ArrayOrder::C,
                        codec: codec_meta.clone(),
                        chunks: ChunkStoreMeta::Memory { chunks },
                        variable_chunks: false,
                        chunk_boundaries: None,
                    },
                })
                .expect("register scenario dataset");
            let mut out = vec![0u32; selected.len() * genes];
            bench(
                config,
                &format!("scenario/mem_{codec_name}_miss{missing}_32x256"),
                96,
                Some(out.len() * item_size),
                || {
                    bank.access_cells(id, &selected, &mut out, None)
                        .expect("access");
                    out.iter()
                        .step_by(131)
                        .fold(0usize, |acc, v| acc ^ usize::try_from(*v).unwrap_or(0))
                },
            );
        }

        // File-backed store (uncompressed) × missing.
        let fo_path = support::bench_data_dir().join(format!(
            "scenario-fo-miss{missing}-{}.bin",
            std::process::id()
        ));
        let mut file = std::fs::File::create(&fo_path).expect("create scenario file");
        let mut locations = Vec::with_capacity(raw_chunks.len());
        let mut offset = 0u64;
        for chunk in &raw_chunks {
            use std::io::Write;
            file.write_all(chunk).expect("write scenario chunk");
            locations.push(FileChunkLocation {
                offset,
                len: chunk.len(),
            });
            offset += chunk.len() as u64;
        }
        file.sync_all().expect("sync scenario file");
        let mut bank = DataBank::new(DataBankConfig::default()).expect("databank");
        let fo_id = bank
            .register_dense_2d(Dense2DMeta {
                gene_names: (0..genes).map(|idx| format!("g{idx}")).collect(),
                data: ArrayMeta {
                    shape: vec![cells, genes],
                    chunk_shape: vec![chunk_rows, chunk_cols],
                    chunk_grid_shape: chunk_grid.clone(),
                    dtype: DType::U32,
                    order: ArrayOrder::C,
                    codec: ArrayCodecMeta::Uncompressed,
                    chunks: ChunkStoreMeta::FileOffset {
                        path: fo_path.clone(),
                        locations,
                    },
                    variable_chunks: false,
                    chunk_boundaries: None,
                },
            })
            .expect("register scenario file dataset");
        let mut fo_out = vec![0u32; selected.len() * genes];
        bench(
            config,
            &format!("scenario/file_none_miss{missing}_32x256"),
            64,
            Some(fo_out.len() * item_size),
            || {
                bank.access_cells(fo_id, &selected, &mut fo_out, None)
                    .expect("access");
                fo_out
                    .iter()
                    .step_by(131)
                    .fold(0usize, |acc, v| acc ^ usize::try_from(*v).unwrap_or(0))
            },
        );
        let _ = std::fs::remove_file(&fo_path);
    }
}
