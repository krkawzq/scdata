//! Stress benchmarks: multi-threaded concurrent load across every module and
//! every operation, measuring aggregate throughput and thread scaling rather
//! than single-operation latency. Run on a many-core machine:
//!
//! ```sh
//! cargo bench --manifest-path rust/scdata/Cargo.toml --bench stress
//! ```
//!
//! For the end-to-end scheduled-prefetch harness (env-driven, file-backed CSR
//! with background CPU/IO contention) see the `fullchain` bench target.

mod support;

use std::hint::black_box;
use std::io::Cursor;
#[cfg(feature = "uring")]
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

use _scdata::access::{
    AccessConfig, AccessCpuConfig, AccessHandle, AccessItem, AccessRequest, AccessScheduler,
    ChunkKey, FileRef, PrefetchRequest, ScheduledAccessConfig,
};
use _scdata::codecs::{
    codec_pipeline_from_specs, BloscCodecConfig, CodecSpec, DecodeBuffer, DecodePool,
    DecodePoolConfig, DecodeRequest, Lz4CodecConfig, SharedCodec, ZstdCodecConfig,
};
use _scdata::databank::{
    ArrayCodecMeta, ArrayMeta, ArrayOrder, ChunkStoreMeta, DType, DataBank, DataBankConfig,
    Dense1DMeta, Dense2DMeta, MissingGenePolicy, ScheduledPrefetchConfig, SparseCsrDatasetMeta,
};
#[cfg(feature = "uring")]
use _scdata::iopool::UringConfig;
use _scdata::iopool::{BaseIoConfig, IoCommand, IoConfig, IoOutput, IoPool, ThreadedConfig};
use support::backends::{CodecDecode, SliceIo};
use support::chunks::{
    make_csr_u32_f32_chunked_raw, make_csr_u32_f32_chunks, make_csr_u32_f32_chunks_lz4,
    make_dense1d_u32_chunks, make_dense_u32_chunks, make_dense_u32_chunks_padded,
    make_dense_u32_chunks_zstd, write_chunks_file, write_dense_u32_directory, write_dense_u32_file,
    write_dense_u32_file_cropped,
};
use support::codecs::{blosc_encode, crc32_encode, decode_into_checksum, encode_for_spec};
use support::data::{DataDist, DataProfile};
use support::{payload, stress_mt, BenchConfig};

const THREAD_COUNTS: &[usize] = &[2, 4, 8, 16];

fn main() {
    let config = BenchConfig::from_env();
    println!("scdata stress benchmarks (multi-threaded concurrent load)");
    println!(
        "SCDATA_BENCH_ITERS=<n> overrides total op count (split across threads); SCDATA_BENCH_WARMUPS defaults to 3"
    );

    stress_codecs(config);
    stress_decode_pool(config);
    stress_iopool(config);
    stress_access(config);
    stress_databank(config);
    stress_missing_rate(config);
    stress_scale(config);
}

// ---------------------------------------------------------------------------
// codecs: every codec, every decode entry point, multi-threaded scaling.
// ---------------------------------------------------------------------------

fn stress_codecs(config: BenchConfig) {
    let raw_1m: Arc<[u8]> = Arc::from(payload(1024 * 1024));
    let zstd_codec = CodecSpec::Zstd(ZstdCodecConfig::default()).build();
    let lz4_codec = CodecSpec::Lz4(Lz4CodecConfig::default()).build();
    let blosc_codec = CodecSpec::Blosc(BloscCodecConfig::new("lz4")).build();
    let none_codec = CodecSpec::None.build();
    let zstd_encoded: Arc<[u8]> =
        Arc::from(zstd::encode_all(Cursor::new(&raw_1m[..]), 3).expect("zstd encode"));
    let lz4_encoded: Arc<[u8]> = Arc::from(lz4_flex::block::compress_prepend_size(&raw_1m));
    let blosc_encoded: Arc<[u8]> = Arc::from(blosc_encode(&raw_1m, 4, 1, 5, "lz4"));
    let pipeline: SharedCodec = codec_pipeline_from_specs(&[
        CodecSpec::Crc32,
        CodecSpec::Zstd(ZstdCodecConfig::default()),
    ]);
    let pipeline_encoded: Arc<[u8]> = {
        let z = zstd::encode_all(Cursor::new(&raw_1m[..]), 3).expect("zstd");
        Arc::from(crc32_encode(&z))
    };

    support::bench(
        config,
        "codecs/ST_decode_zstd_1m",
        192,
        Some(raw_1m.len()),
        || decode_into_checksum(&zstd_codec, &zstd_encoded, raw_1m.len()),
    );
    support::bench(
        config,
        "codecs/ST_decode_lz4_1m",
        384,
        Some(raw_1m.len()),
        || decode_into_checksum(&lz4_codec, &lz4_encoded, raw_1m.len()),
    );
    support::bench(
        config,
        "codecs/ST_decode_blosc_lz4_1m",
        384,
        Some(raw_1m.len()),
        || decode_into_checksum(&blosc_codec, &blosc_encoded, raw_1m.len()),
    );
    support::bench(
        config,
        "codecs/ST_decode_pipeline_crc32_zstd_1m",
        160,
        Some(raw_1m.len()),
        || decode_into_checksum(&pipeline, &pipeline_encoded, raw_1m.len()),
    );

    // Multi-threaded: codecs are stateless; verify linear scaling and absence
    // of global locks (blosc uses thread-local scratch, zstd bulk per-call DCtx).
    for codec_kind in ["zstd", "lz4", "blosc", "none"] {
        let (codec, encoded): (SharedCodec, Arc<[u8]>) = match codec_kind {
            "zstd" => (zstd_codec.clone(), Arc::clone(&zstd_encoded)),
            "lz4" => (lz4_codec.clone(), Arc::clone(&lz4_encoded)),
            "blosc" => (blosc_codec.clone(), Arc::clone(&blosc_encoded)),
            _ => (none_codec.clone(), Arc::clone(&raw_1m)),
        };
        for &threads in THREAD_COUNTS {
            let codec = Arc::clone(&codec);
            let encoded = Arc::clone(&encoded);
            let raw_1m = Arc::clone(&raw_1m);
            stress_mt(
                config,
                &format!("codecs/MT_decode_{codec_kind}_1m/t{threads}"),
                threads,
                192,
                Some(raw_1m.len()),
                move |_| {
                    let mut out = vec![0u8; raw_1m.len()];
                    let written = codec
                        .decode_into(&encoded, DecodeBuffer::new(&mut out), Some(raw_1m.len()))
                        .expect("decode");
                    written ^ out[written / 2] as usize
                },
            );
        }
    }

    // Pipeline decode under concurrency (two-stage crc32 + zstd).
    for &threads in THREAD_COUNTS {
        let pipeline = Arc::clone(&pipeline);
        let encoded = Arc::clone(&pipeline_encoded);
        let raw_1m = Arc::clone(&raw_1m);
        stress_mt(
            config,
            &format!("codecs/MT_decode_pipeline_crc32_zstd_1m/t{threads}"),
            threads,
            160,
            Some(raw_1m.len()),
            move |_| {
                let mut out = vec![0u8; raw_1m.len()];
                let written = pipeline
                    .decode_into(&encoded, DecodeBuffer::new(&mut out), Some(raw_1m.len()))
                    .expect("decode");
                written ^ out[written / 2] as usize
            },
        );
    }

    // decode_to_vec (caller-owned buffer) under concurrency.
    for &threads in THREAD_COUNTS {
        let codec = Arc::clone(&zstd_codec);
        let encoded = Arc::clone(&zstd_encoded);
        let raw_1m = Arc::clone(&raw_1m);
        stress_mt(
            config,
            &format!("codecs/MT_decode_to_vec_zstd_1m/t{threads}"),
            threads,
            192,
            Some(raw_1m.len()),
            move |_| {
                let out = Vec::with_capacity(raw_1m.len());
                let decoded = codec
                    .decode_to_vec(&encoded, out, Some(raw_1m.len()))
                    .expect("decode_to_vec");
                decoded.len() ^ decoded[decoded.len() / 2] as usize
            },
        );
    }
}

// ---------------------------------------------------------------------------
// decode pool: all submission paths under concurrent load.
// ---------------------------------------------------------------------------

fn stress_decode_pool(config: BenchConfig) {
    let raw: Arc<[u8]> = Arc::from(payload(64 * 1024));
    let zstd_encoded: Arc<[u8]> =
        Arc::from(zstd::encode_all(Cursor::new(&raw[..]), 3).expect("zstd encode"));
    let pool = Arc::new(
        DecodePool::new(DecodePoolConfig {
            num_workers: 4,
            queue_capacity: 256,
            cpus: None,
        })
        .expect("decode pool"),
    );
    let zstd_codec = CodecSpec::Zstd(ZstdCodecConfig::default()).build();

    support::bench(
        config,
        "codecs/ST_decode_pool_batch_32x_zstd_64k",
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

    for &threads in THREAD_COUNTS {
        let pool = Arc::clone(&pool);
        let codec = Arc::clone(&zstd_codec);
        let encoded = Arc::clone(&zstd_encoded);
        let raw = Arc::clone(&raw);
        stress_mt(
            config,
            &format!("codecs/MT_decode_pool_submit_zstd_64k/t{threads}"),
            threads,
            512,
            Some(raw.len()),
            move |_| {
                let request = DecodeRequest::new(Arc::clone(&codec), Arc::clone(&encoded))
                    .with_expected_size(raw.len());
                let decoded = pool
                    .submit(request)
                    .expect("submit decode")
                    .blocking_recv()
                    .expect("decode");
                decoded.len() ^ decoded[decoded.len() / 2] as usize
            },
        );
    }

    for &threads in [4usize, 8, 16].iter() {
        let pool = Arc::clone(&pool);
        let codec = Arc::clone(&zstd_codec);
        let encoded = Arc::clone(&zstd_encoded);
        let raw = Arc::clone(&raw);
        stress_mt(
            config,
            &format!("codecs/MT_decode_pool_try_submit_zstd_64k/t{threads}"),
            threads,
            512,
            Some(raw.len()),
            move |_| {
                let request = DecodeRequest::new(Arc::clone(&codec), Arc::clone(&encoded))
                    .with_expected_size(raw.len());
                let future = match pool.try_submit(request) {
                    Ok(future) => future,
                    Err(_) => return 0,
                };
                let decoded = future.blocking_recv().expect("decode");
                decoded.len() ^ decoded[decoded.len() / 2] as usize
            },
        );
    }
}

// ---------------------------------------------------------------------------
// iopool: every operation, both backends, sharding, concurrency.
// ---------------------------------------------------------------------------

fn stress_iopool(config: BenchConfig) {
    let data = payload(8 * 1024 * 1024);
    let path = support::bench_data_dir().join(format!("stress-iopool-{}.bin", std::process::id()));
    std::fs::write(&path, &data).expect("write stress iopool data");

    for (qshards, label) in [(1usize, "1q"), (4, "4q")] {
        let pool = Arc::new(
            IoPool::new(IoConfig::Threaded(ThreadedConfig {
                base: BaseIoConfig {
                    max_in_flight: 256,
                    priority_levels: 2,
                    queue_shards: qshards,
                    assume_non_overlapping_reads: true,
                },
                num_workers: 8,
                cpus: None,
            }))
            .expect("threaded iopool"),
        );
        let file = pool.register_readonly_file(&path).expect("register file");

        for read_size in [4 * 1024usize, 64 * 1024, 1024 * 1024] {
            let max_offset = data.len() - read_size;
            let counter = AtomicUsize::new(0);
            support::bench(
                config,
                &format!("iopool/ST_threaded_read_{label}_{}k", read_size / 1024),
                2048,
                Some(read_size),
                || {
                    let offset =
                        (counter.fetch_add(1, Ordering::Relaxed) * read_size) % (max_offset + 1);
                    let bytes = pool
                        .submit(IoCommand::read(file, offset as u64, read_size, 0))
                        .expect("submit read")
                        .blocking_recv_read()
                        .expect("read");
                    bytes.len() ^ bytes[bytes.len() / 2] as usize
                },
            );
            for &threads in THREAD_COUNTS {
                let pool = Arc::clone(&pool);
                let counter = Arc::new(AtomicUsize::new(0));
                stress_mt(
                    config,
                    &format!(
                        "iopool/MT_threaded_read_{label}_{}k/t{threads}",
                        read_size / 1024
                    ),
                    threads,
                    1024,
                    Some(read_size),
                    move |_| {
                        let idx = counter.fetch_add(1, Ordering::Relaxed);
                        let offset = (idx * read_size) % (max_offset + 1);
                        let bytes = pool
                            .submit(IoCommand::read(file, offset as u64, read_size, 0))
                            .expect("submit read")
                            .blocking_recv_read()
                            .expect("read");
                        bytes.len() ^ bytes[bytes.len() / 2] as usize
                    },
                );
            }
        }

        let rw_file = pool
            .register_readwrite_file(&path)
            .expect("register rw file");
        for &threads in [2usize, 4].iter() {
            let pool = Arc::clone(&pool);
            stress_mt(
                config,
                &format!("iopool/MT_threaded_write_{label}_64k/t{threads}"),
                threads,
                512,
                Some(64 * 1024),
                move |t| {
                    let buf = vec![0xA5u8; 64 * 1024];
                    let offset = (t % 64) as u64 * (64 * 1024) as u64;
                    pool.submit(IoCommand::write(rw_file, offset, buf, 0))
                        .expect("submit write")
                        .blocking_recv()
                        .expect("write")
                        .bytes_written()
                        .expect("write output")
                },
            );
        }
        for &threads in [2usize, 4].iter() {
            let pool = Arc::clone(&pool);
            stress_mt(
                config,
                &format!("iopool/MT_threaded_fsync_{label}/t{threads}"),
                threads,
                512,
                None,
                move |_| {
                    let out = pool
                        .submit(IoCommand::fsync(rw_file, 0))
                        .expect("submit fsync")
                        .blocking_recv()
                        .expect("fsync");
                    if matches!(out, IoOutput::SyncAll) {
                        1
                    } else {
                        0
                    }
                },
            );
        }
        for &threads in [2usize, 4].iter() {
            let pool = Arc::clone(&pool);
            stress_mt(
                config,
                &format!("iopool/MT_threaded_sync_data_{label}/t{threads}"),
                threads,
                512,
                None,
                move |_| {
                    let out = pool
                        .submit(IoCommand::sync_data(rw_file, 0))
                        .expect("submit sync_data")
                        .blocking_recv()
                        .expect("sync_data");
                    if matches!(out, IoOutput::SyncData) {
                        1
                    } else {
                        0
                    }
                },
            );
        }
        // truncate on a dedicated file so it never disturbs the read path.
        let trunc_path = support::bench_data_dir().join(format!(
            "stress-iopool-trunc-{label}-{}.bin",
            std::process::id()
        ));
        std::fs::write(&trunc_path, &data[..64 * 1024]).expect("write trunc file");
        let trunc_file = pool
            .register_readwrite_file(&trunc_path)
            .expect("register trunc file");
        for &threads in [2usize, 4].iter() {
            let pool = Arc::clone(&pool);
            stress_mt(
                config,
                &format!("iopool/MT_threaded_truncate_{label}/t{threads}"),
                threads,
                512,
                None,
                move |_| {
                    let out = pool
                        .submit(IoCommand::truncate(trunc_file, 1024, 0))
                        .expect("submit truncate")
                        .blocking_recv()
                        .expect("truncate");
                    if matches!(out, IoOutput::Truncate) {
                        1
                    } else {
                        0
                    }
                },
            );
        }
        pool.unregister_file(trunc_file)
            .expect("unregister trunc file");
        let _ = std::fs::remove_file(trunc_path);

        support::bench(
            config,
            &format!("iopool/ST_threaded_metadata_{label}"),
            4096,
            None,
            || {
                pool.submit(IoCommand::metadata(file, 0))
                    .expect("submit metadata")
                    .blocking_recv()
                    .expect("metadata")
                    .metadata()
                    .expect("metadata output")
                    .len as usize
            },
        );

        pool.unregister_file(rw_file).expect("unregister rw file");
        pool.unregister_file(file).expect("unregister file");
    }

    // Duplicate-read dedup under concurrency (same offset, many waiters).
    {
        let pool = Arc::new(
            IoPool::new(IoConfig::Threaded(ThreadedConfig {
                base: BaseIoConfig {
                    max_in_flight: 256,
                    priority_levels: 2,
                    queue_shards: 1,
                    assume_non_overlapping_reads: false,
                },
                num_workers: 4,
                cpus: None,
            }))
            .expect("dedup iopool"),
        );
        let file = pool.register_readonly_file(&path).expect("register file");
        for &threads in [4usize, 8, 16].iter() {
            let pool = Arc::clone(&pool);
            stress_mt(
                config,
                &format!("iopool/MT_duplicate_read_64k_x8_dedup/t{threads}"),
                threads,
                512,
                Some(64 * 1024 * 8),
                move |_| {
                    let futures = (0..8)
                        .map(|_| {
                            pool.submit(IoCommand::read(file, 0, 64 * 1024, 0))
                                .expect("submit dup read")
                        })
                        .collect::<Vec<_>>();
                    futures.into_iter().fold(0usize, |acc, future| {
                        let bytes = future.blocking_recv_read().expect("read dup");
                        acc ^ bytes.len() ^ bytes[bytes.len() / 2] as usize
                    })
                },
            );
        }
        pool.unregister_file(file).expect("unregister dedup file");
    }

    // register/unregister lifecycle (single-thread, generation churn).
    {
        let pool = IoPool::new(IoConfig::Threaded(ThreadedConfig {
            base: BaseIoConfig::default(),
            num_workers: 2,
            cpus: None,
        }))
        .expect("lifecycle iopool");
        support::bench(
            config,
            "iopool/ST_register_unregister_cycle",
            1024,
            None,
            || {
                let id = pool.register_readonly_file(&path).expect("register");
                pool.unregister_file(id).expect("unregister");
                id
            },
        );
    }

    #[cfg(feature = "uring")]
    {
        stress_iopool_uring(config, &path, &data);
    }

    let _ = std::fs::remove_file(path);
}

#[cfg(feature = "uring")]
fn stress_iopool_uring(config: BenchConfig, path: &Path, data: &[u8]) {
    let pool = match IoPool::new(IoConfig::Uring(UringConfig {
        base: BaseIoConfig {
            max_in_flight: 256,
            priority_levels: 2,
            queue_shards: 1,
            assume_non_overlapping_reads: true,
        },
        entries: 512,
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
            eprintln!("[iopool] skipping io_uring stress (unavailable): {err}");
            return;
        }
        Err(err) => panic!("create uring pool: {err}"),
    };
    let pool = Arc::new(pool);
    let file = pool.register_readonly_file(path).expect("register file");
    let read_size = 64 * 1024usize;
    let max_offset = data.len() - read_size;

    let counter = AtomicUsize::new(0);
    support::bench(
        config,
        "iopool/ST_uring_read_64k",
        2048,
        Some(read_size),
        || {
            let offset = (counter.fetch_add(1, Ordering::Relaxed) * read_size) % (max_offset + 1);
            let bytes = pool
                .submit(IoCommand::read(file, offset as u64, read_size, 0))
                .expect("submit read")
                .blocking_recv_read()
                .expect("read");
            bytes.len() ^ bytes[bytes.len() / 2] as usize
        },
    );
    for &threads in THREAD_COUNTS {
        let pool = Arc::clone(&pool);
        let counter = Arc::new(AtomicUsize::new(0));
        stress_mt(
            config,
            &format!("iopool/MT_uring_read_64k/t{threads}"),
            threads,
            1024,
            Some(read_size),
            move |_| {
                let idx = counter.fetch_add(1, Ordering::Relaxed);
                let offset = (idx * read_size) % (max_offset + 1);
                let bytes = pool
                    .submit(IoCommand::read(file, offset as u64, read_size, 0))
                    .expect("submit read")
                    .blocking_recv_read()
                    .expect("read");
                bytes.len() ^ bytes[bytes.len() / 2] as usize
            },
        );
    }
    pool.unregister_file(file).expect("unregister uring file");
}

// ---------------------------------------------------------------------------
// access: all paths, cache states, keep_decoded, concurrency. The MT runs use
// scheduler sharding to expose whether access orchestration can scale.
// ---------------------------------------------------------------------------

fn stress_access(config: BenchConfig) {
    let chunk = 64 * 1024usize;
    let backing = Arc::<[u8]>::from(payload(8 * 1024 * 1024));

    for keep in [true, false] {
        let label = if keep { "keep" } else { "evict" };
        let handle = Arc::new(
            AccessScheduler::spawn(
                AccessConfig {
                    queue_capacity: 256,
                    scheduler_shards: 4,
                    cache_capacity_bytes: 4 * 1024 * 1024,
                    memory_budget_bytes: 8 * 1024 * 1024,
                    default_io_priority: 0,
                    keep_decoded: keep,
                    cpu: AccessCpuConfig {
                        num_workers: 4,
                        queue_capacity: 256,
                        cpus: None,
                    },
                },
                Box::new(SliceIo::new(Arc::clone(&backing))),
                Box::new(CodecDecode),
            )
            .expect("access scheduler"),
        );
        let codec = CodecSpec::None.build();

        let counter = AtomicUsize::new(0);
        support::bench(
            config,
            &format!("access/ST_send_{label}_miss_64k"),
            2048,
            Some(chunk),
            || {
                let start =
                    (counter.fetch_add(1, Ordering::Relaxed) * chunk) % (backing.len() - chunk);
                send_read(&handle, start, chunk, &codec)
            },
        );

        // All threads share one AccessHandle; with scheduler sharding this
        // exposes whether orchestration keeps scaling under high kops.
        for &threads in THREAD_COUNTS {
            let handle = Arc::clone(&handle);
            let codec = Arc::clone(&codec);
            let backing = Arc::clone(&backing);
            let counter = Arc::new(AtomicUsize::new(0));
            stress_mt(
                config,
                &format!("access/MT_send_{label}_64k/t{threads}"),
                threads,
                1024,
                Some(chunk),
                move |_| {
                    let start =
                        (counter.fetch_add(1, Ordering::Relaxed) * chunk) % (backing.len() - chunk);
                    send_read(&handle, start, chunk, &codec)
                },
            );
        }

        // try_send (non-blocking submit) under concurrency.
        for &threads in THREAD_COUNTS {
            let handle = Arc::clone(&handle);
            let codec = Arc::clone(&codec);
            let backing = Arc::clone(&backing);
            let counter = Arc::new(AtomicUsize::new(0));
            stress_mt(
                config,
                &format!("access/MT_try_send_{label}_64k/t{threads}"),
                threads,
                1024,
                Some(chunk),
                move |_| {
                    let start =
                        (counter.fetch_add(1, Ordering::Relaxed) * chunk) % (backing.len() - chunk);
                    try_send_read(&handle, start, chunk, &codec)
                },
            );
        }

        // Scatter-copy slice under concurrency: only a prefix of each decoded
        // chunk is copied back, isolating scatter from full-payload copy cost.
        for &threads in THREAD_COUNTS {
            let handle = Arc::clone(&handle);
            let codec = Arc::clone(&codec);
            let backing = Arc::clone(&backing);
            let counter = Arc::new(AtomicUsize::new(0));
            stress_mt(
                config,
                &format!("access/MT_send_slice_{label}_64k/t{threads}"),
                threads,
                1024,
                Some(chunk),
                move |_| {
                    let start =
                        (counter.fetch_add(1, Ordering::Relaxed) * chunk) % (backing.len() - chunk);
                    send_read_with_slice(&handle, start, chunk, &codec)
                },
            );
        }

        let scheduled_items: Arc<Vec<_>> = Arc::new(
            (0..16)
                .map(|idx| {
                    AccessItem::new(
                        ChunkKey::new(FileRef(1), (idx * chunk) as u64, chunk),
                        Arc::clone(&codec),
                        Some(chunk),
                    )
                })
                .collect(),
        );
        support::bench(
            config,
            &format!("access/ST_scheduled_{label}_16x64k"),
            384,
            Some(chunk * 16),
            || scheduled_run(&handle, &scheduled_items),
        );
        for &threads in THREAD_COUNTS {
            let handle = Arc::clone(&handle);
            let items = Arc::clone(&scheduled_items);
            stress_mt(
                config,
                &format!("access/MT_scheduled_{label}_16x64k/t{threads}"),
                threads,
                256,
                Some(chunk * 16),
                move |_| scheduled_run(&handle, &items),
            );
        }

        let prefetch_keys: Arc<Vec<ChunkKey>> = Arc::new(
            (0..64)
                .map(|idx| ChunkKey::new(FileRef(1), (idx * chunk) as u64, chunk))
                .collect(),
        );
        for &threads in [2usize, 4, 8].iter() {
            let handle = Arc::clone(&handle);
            let keys = Arc::clone(&prefetch_keys);
            let counter = Arc::new(AtomicUsize::new(0));
            stress_mt(
                config,
                &format!("access/MT_prefetch_{label}_64k/t{threads}"),
                threads,
                512,
                Some(chunk),
                move |_| {
                    let key = keys[counter.fetch_add(1, Ordering::Relaxed) % keys.len()];
                    handle
                        .prefetch(key)
                        .expect("prefetch")
                        .blocking_recv()
                        .expect("prefetch reply")
                        .expect("prefetch result");
                    key.offset as usize ^ key.len
                },
            );
        }
        // send_prefetch with no reply channel under concurrency.
        for &threads in [2usize, 4, 8].iter() {
            let handle = Arc::clone(&handle);
            let keys = Arc::clone(&prefetch_keys);
            let counter = Arc::new(AtomicUsize::new(0));
            stress_mt(
                config,
                &format!("access/MT_send_prefetch_no_reply_{label}_64k/t{threads}"),
                threads,
                512,
                Some(chunk),
                move |_| {
                    let key = keys[counter.fetch_add(1, Ordering::Relaxed) % keys.len()];
                    handle
                        .send_prefetch(PrefetchRequest::new(key))
                        .expect("send_prefetch");
                    key.offset as usize ^ key.len
                },
            );
        }
    }

    let handle_oom = Arc::new(
        AccessScheduler::spawn(
            AccessConfig {
                queue_capacity: 64,
                scheduler_shards: 1,
                cache_capacity_bytes: 1024,
                memory_budget_bytes: 1024,
                default_io_priority: 0,
                keep_decoded: true,
                cpu: AccessCpuConfig {
                    num_workers: 1,
                    queue_capacity: 64,
                    cpus: None,
                },
            },
            Box::new(SliceIo::new(Arc::clone(&backing))),
            Box::new(CodecDecode),
        )
        .expect("oom access scheduler"),
    );
    let codec = CodecSpec::None.build();
    for &threads in [2usize, 4, 8].iter() {
        let handle = Arc::clone(&handle_oom);
        let codec = Arc::clone(&codec);
        stress_mt(
            config,
            &format!("access/MT_error_memory_budget_exhausted_64k/t{threads}"),
            threads,
            512,
            None,
            move |_| send_read_error(&handle, 0, chunk, &codec, None),
        );
    }

    let handle_invalid = Arc::new(
        AccessScheduler::spawn(
            AccessConfig {
                queue_capacity: 64,
                scheduler_shards: 2,
                cache_capacity_bytes: 4 * 1024 * 1024,
                memory_budget_bytes: 8 * 1024 * 1024,
                default_io_priority: 0,
                keep_decoded: true,
                cpu: AccessCpuConfig {
                    num_workers: 2,
                    queue_capacity: 64,
                    cpus: None,
                },
            },
            Box::new(SliceIo::new(Arc::clone(&backing))),
            Box::new(CodecDecode),
        )
        .expect("invalid-slice access scheduler"),
    );
    for &threads in [2usize, 4, 8].iter() {
        let handle = Arc::clone(&handle_invalid);
        let codec = Arc::clone(&codec);
        stress_mt(
            config,
            &format!("access/MT_error_invalid_slice_64k/t{threads}"),
            threads,
            512,
            None,
            move |_| send_read_error(&handle, 0, chunk, &codec, Some(vec![0, 0, chunk + 1])),
        );
    }
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

fn scheduled_run(handle: &AccessHandle, items: &[AccessItem]) -> usize {
    let scheduled = handle
        .scheduled(
            items.to_vec(),
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
}

// ---------------------------------------------------------------------------
// databank: dense1d, compressed dense2d, sparse csr, file-backed store,
// directory store, multi-instance scaling, by-gene-names, owned output,
// prefetch, and register/unregister churn.
// ---------------------------------------------------------------------------

fn stress_databank(config: BenchConfig) {
    let cells = 1024usize;
    let genes = 1024usize;
    let chunk_rows = 16usize;
    let chunk_cols = 128usize;
    let chunk_grid = vec![cells / chunk_rows, genes / chunk_cols];

    let mem_chunks = make_dense_u32_chunks(cells, genes, chunk_rows, chunk_cols);
    let zstd_mem_chunks = make_dense_u32_chunks_zstd(cells, genes, chunk_rows, chunk_cols, 3);
    let fo_path =
        support::bench_data_dir().join(format!("stress-databank-fo-{}.bin", std::process::id()));
    let fo_locations = write_dense_u32_file(&fo_path, cells, genes, chunk_rows, chunk_cols);
    let dir_dir =
        support::bench_data_dir().join(format!("stress-databank-dir-{}", std::process::id()));
    std::fs::create_dir_all(&dir_dir).expect("create dir store");
    let dir_locations = write_dense_u32_directory(&dir_dir, cells, genes, chunk_rows, chunk_cols);

    let cases: Vec<(&str, ArrayCodecMeta, ChunkStoreMeta)> = vec![
        (
            "mem_unc",
            ArrayCodecMeta::Uncompressed,
            ChunkStoreMeta::Memory {
                chunks: mem_chunks.clone(),
            },
        ),
        (
            "mem_zstd",
            ArrayCodecMeta::CodecJson(r#"{"id":"zstd","level":3}"#.to_string()),
            ChunkStoreMeta::Memory {
                chunks: zstd_mem_chunks.clone(),
            },
        ),
        (
            "file_unc",
            ArrayCodecMeta::Uncompressed,
            ChunkStoreMeta::FileOffset {
                path: fo_path.clone(),
                locations: fo_locations.clone(),
            },
        ),
        (
            "dir_unc",
            ArrayCodecMeta::Uncompressed,
            ChunkStoreMeta::Directory {
                locations: dir_locations.clone(),
            },
        ),
    ];

    let mut bank = DataBank::new(stress_databank_config()).expect("databank");
    let mut ids: Vec<(&str, _scdata::databank::DatasetId)> = Vec::new();
    for (label, codec, store) in &cases {
        let id = bank
            .register_dense_2d(Dense2DMeta {
                gene_names: (0..genes).map(|idx| format!("gene_{idx}")).collect(),
                data: ArrayMeta {
                    shape: vec![cells, genes],
                    chunk_shape: vec![chunk_rows, chunk_cols],
                    chunk_grid_shape: chunk_grid.clone(),
                    dtype: DType::U32,
                    order: ArrayOrder::C,
                    codec: codec.clone(),
                    chunks: store.clone(),
                    variable_chunks: false,
                    chunk_boundaries: None,
                },
            })
            .expect("register dense dataset");
        ids.push((label, id));
    }

    let selected: Vec<usize> = (0..32).map(|idx| idx * 7 % cells).collect();

    // ST dense2d access (checked + unchecked) per store/codec.
    for (label, id) in &ids {
        let mut out = vec![0u32; selected.len() * genes];
        support::bench(
            config,
            &format!("databank/ST_dense2d_{label}_checked_32x1024"),
            192,
            Some(out.len() * 4),
            || {
                bank.access_cells(*id, &selected, &mut out, None)
                    .expect("access");
                out.iter()
                    .step_by(257)
                    .fold(0usize, |acc, v| acc ^ usize::try_from(*v).unwrap_or(0))
            },
        );
        support::bench(
            config,
            &format!("databank/ST_dense2d_{label}_unchecked_32x1024"),
            256,
            Some(out.len() * 4),
            || {
                unsafe {
                    bank.access_cells_unchecked(*id, &selected, &mut out, None)
                        .expect("access");
                }
                out.iter()
                    .step_by(257)
                    .fold(0usize, |acc, v| acc ^ usize::try_from(*v).unwrap_or(0))
            },
        );
    }

    // Non-divisible edge chunks: compare padded memory chunks against cropped
    // legacy file-offset chunks under the same access pattern.
    let edge_cells = 35usize;
    let edge_genes = 70usize;
    let edge_chunk_rows = 16usize;
    let edge_chunk_cols = 32usize;
    let edge_grid = vec![
        edge_cells.div_ceil(edge_chunk_rows),
        edge_genes.div_ceil(edge_chunk_cols),
    ];
    let edge_selected = vec![0usize, 15, 16, 34];
    let edge_mem_chunks =
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
                    chunks: edge_mem_chunks,
                },
                variable_chunks: false,
                chunk_boundaries: None,
            },
        })
        .expect("register edge memory dataset");
    let edge_fo_path =
        support::bench_data_dir().join(format!("stress-edge-fo-{}.bin", std::process::id()));
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
                chunk_grid_shape: edge_grid,
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
        .expect("register edge fileoffset dataset");
    for (label, id) in [
        ("memory_padded_edges", edge_mem_id),
        ("fileoffset_cropped_edges", edge_fo_id),
    ] {
        let mut edge_out = vec![0u32; edge_selected.len() * edge_genes];
        support::bench(
            config,
            &format!("databank/ST_dense2d_{label}_4x70"),
            512,
            Some(edge_out.len() * 4),
            || {
                bank.access_cells(id, &edge_selected, &mut edge_out, None)
                    .expect("access edge dataset");
                edge_out
                    .iter()
                    .fold(0usize, |acc, v| acc ^ usize::try_from(*v).unwrap_or(0))
            },
        );
    }

    // Dense1D register + ST access.
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
        .expect("register dense1d");
    let d1_selected: Vec<usize> = (0..32).map(|idx| idx * 7 % d1_cells).collect();
    let mut d1_out = vec![0u32; d1_selected.len() * d1_genes];
    support::bench(
        config,
        "databank/ST_dense1d_memory_checked_32x1024",
        256,
        Some(d1_out.len() * 4),
        || {
            bank.access_cells(d1_id, &d1_selected, &mut d1_out, None)
                .expect("access");
            d1_out
                .iter()
                .step_by(257)
                .fold(0usize, |acc, v| acc ^ usize::try_from(*v).unwrap_or(0))
        },
    );

    // Sparse CSR register + ST access (checked + unchecked).
    let csr_cells = 256usize;
    let csr_genes = 512usize;
    let csr_nnz_per_cell = 8usize;
    let csr_nnz = csr_cells * csr_nnz_per_cell;
    let (indptr, indices_chunk, data_chunk) =
        make_csr_u32_f32_chunks(csr_cells, csr_genes, csr_nnz_per_cell);
    let csr_id = bank
        .register_sparse_csr(SparseCsrDatasetMeta {
            gene_names: (0..csr_genes).map(|idx| format!("gene_{idx}")).collect(),
            indptr,
            indices: ArrayMeta {
                shape: vec![csr_nnz],
                chunk_shape: vec![csr_nnz],
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
                shape: vec![csr_nnz],
                chunk_shape: vec![csr_nnz],
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
        .expect("register sparse csr");
    let csr_selected: Vec<usize> = (0..64).map(|idx| idx * 3 % csr_cells).collect();
    let mut csr_out = vec![0.0f32; csr_selected.len() * csr_genes];
    support::bench(
        config,
        "databank/ST_sparse_csr_checked_64x512",
        256,
        Some(csr_out.len() * 4),
        || {
            bank.access_cells(csr_id, &csr_selected, &mut csr_out, None)
                .expect("access");
            csr_out
                .iter()
                .step_by(251)
                .fold(0usize, |acc, v| acc ^ v.to_bits() as usize)
        },
    );
    support::bench(
        config,
        "databank/ST_sparse_csr_unchecked_64x512",
        384,
        Some(csr_out.len() * 4),
        || {
            unsafe {
                bank.access_cells_unchecked(csr_id, &csr_selected, &mut csr_out, None)
                    .expect("access");
            }
            csr_out
                .iter()
                .step_by(251)
                .fold(0usize, |acc, v| acc ^ v.to_bits() as usize)
        },
    );

    // by-gene-names (Zero + Error-present) + owned/alloc output, single-thread.
    let mem_id = ids
        .iter()
        .find(|(label, _)| *label == "mem_unc")
        .map(|(_, id)| *id)
        .expect("mem_unc id");
    let by_names: Vec<String> = (0..32)
        .map(|idx| format!("gene_{}", idx * 7 % genes))
        .collect();
    let mut by_out = vec![0u32; selected.len() * by_names.len()];
    support::bench(
        config,
        "databank/ST_dense2d_by_gene_names_zero_32x32",
        192,
        Some(by_out.len() * 4),
        || {
            bank.access_cells_by_gene_names(
                mem_id,
                &selected,
                &by_names,
                &mut by_out,
                None,
                MissingGenePolicy::Zero,
            )
            .expect("access by gene names");
            by_out
                .iter()
                .step_by(131)
                .fold(0usize, |acc, v| acc ^ usize::try_from(*v).unwrap_or(0))
        },
    );
    support::bench(
        config,
        "databank/ST_dense2d_by_gene_names_error_present_32x32",
        192,
        Some(by_out.len() * 4),
        || {
            bank.access_cells_by_gene_names(
                mem_id,
                &selected,
                &by_names,
                &mut by_out,
                None,
                MissingGenePolicy::Error,
            )
            .expect("access by gene names error");
            by_out
                .iter()
                .step_by(131)
                .fold(0usize, |acc, v| acc ^ usize::try_from(*v).unwrap_or(0))
        },
    );
    support::bench(
        config,
        "databank/ST_dense2d_owned_32x1024",
        128,
        Some(selected.len() * genes * 4),
        || {
            let out: Vec<u32> = bank.access_cells_owned(mem_id, &selected).expect("owned");
            out.len() ^ out[0] as usize
        },
    );
    support::bench(
        config,
        "databank/ST_dense2d_alloc_32x1024",
        128,
        Some(selected.len() * genes * 4),
        || {
            let out: Vec<u32> = bank.access_cells_alloc(mem_id, &selected).expect("alloc");
            out.len() ^ out[0] as usize
        },
    );
    support::bench(
        config,
        "databank/ST_prefetch_cells_dense2d_32",
        64,
        None,
        || {
            bank.prefetch_cells(mem_id, &selected).expect("prefetch");
            selected.len()
        },
    );
    support::bench(
        config,
        "databank/ST_dataset_genes_len",
        10_000,
        None,
        || bank.dataset_genes(mem_id).expect("dataset genes").len(),
    );

    // Sparse CSR no-blocking pressure tests. These use multi-chunk memory
    // storage so databank still exercises per-batch plan/group/scatter, but the
    // access/decode/io layers are not part of this throughput number.
    let csr_nb_cells = 4096usize;
    let csr_nb_genes = 4096usize;
    let csr_nb_nnz_per_cell = 64usize;
    let csr_nb_chunk_cells = 64usize;
    let csr_nb_chunk_len = csr_nb_chunk_cells * csr_nb_nnz_per_cell;
    let csr_nb_nnz = csr_nb_cells * csr_nb_nnz_per_cell;
    let (csr_nb_indptr, csr_nb_indices_chunks, csr_nb_data_chunks) = make_csr_u32_f32_chunked_raw(
        csr_nb_cells,
        csr_nb_genes,
        csr_nb_nnz_per_cell,
        csr_nb_chunk_len,
    );
    let csr_nb_id = bank
        .register_sparse_csr(SparseCsrDatasetMeta {
            gene_names: (0..csr_nb_genes).map(|idx| format!("gene_{idx}")).collect(),
            indptr: csr_nb_indptr,
            indices: ArrayMeta {
                shape: vec![csr_nb_nnz],
                chunk_shape: vec![csr_nb_chunk_len],
                chunk_grid_shape: vec![csr_nb_indices_chunks.len()],
                dtype: DType::U32,
                order: ArrayOrder::C,
                codec: ArrayCodecMeta::Uncompressed,
                chunks: ChunkStoreMeta::Memory {
                    chunks: csr_nb_indices_chunks,
                },
                variable_chunks: false,
                chunk_boundaries: None,
            },
            data: ArrayMeta {
                shape: vec![csr_nb_nnz],
                chunk_shape: vec![csr_nb_chunk_len],
                chunk_grid_shape: vec![csr_nb_data_chunks.len()],
                dtype: DType::F32,
                order: ArrayOrder::C,
                codec: ArrayCodecMeta::Uncompressed,
                chunks: ChunkStoreMeta::Memory {
                    chunks: csr_nb_data_chunks,
                },
                variable_chunks: false,
                chunk_boundaries: None,
            },
            index_dtype: DType::U32,
            num_cells: csr_nb_cells,
            num_genes: csr_nb_genes,
        })
        .expect("register sparse csr no-block");
    let csr_nb_repeat: Vec<usize> = (0..64).collect();
    let csr_nb_scattered: Vec<usize> = (0..64).map(|idx| idx * csr_nb_chunk_cells).collect();
    let mut csr_nb_out = vec![0.0f32; csr_nb_repeat.len() * csr_nb_genes];
    support::bench(
        config,
        "databank/ST_sparse_csr_noblock_repeat_chunk_checked_64x4096",
        256,
        Some(csr_nb_out.len() * 4),
        || {
            bank.access_cells(csr_nb_id, &csr_nb_repeat, &mut csr_nb_out, None)
                .expect("access");
            csr_nb_out
                .iter()
                .step_by(4099)
                .fold(0usize, |acc, v| acc ^ v.to_bits() as usize)
        },
    );
    support::bench(
        config,
        "databank/ST_sparse_csr_noblock_repeat_chunk_unchecked_64x4096",
        384,
        Some(csr_nb_out.len() * 4),
        || {
            unsafe {
                bank.access_cells_unchecked(csr_nb_id, &csr_nb_repeat, &mut csr_nb_out, None)
                    .expect("access");
            }
            csr_nb_out
                .iter()
                .step_by(4099)
                .fold(0usize, |acc, v| acc ^ v.to_bits() as usize)
        },
    );
    support::bench(
        config,
        "databank/ST_sparse_csr_noblock_scattered_chunks_checked_64x4096",
        256,
        Some(csr_nb_out.len() * 4),
        || {
            bank.access_cells(csr_nb_id, &csr_nb_scattered, &mut csr_nb_out, None)
                .expect("access");
            csr_nb_out
                .iter()
                .step_by(4099)
                .fold(0usize, |acc, v| acc ^ v.to_bits() as usize)
        },
    );
    support::bench(
        config,
        "databank/ST_sparse_csr_noblock_scattered_chunks_unchecked_64x4096",
        384,
        Some(csr_nb_out.len() * 4),
        || {
            unsafe {
                bank.access_cells_unchecked(csr_nb_id, &csr_nb_scattered, &mut csr_nb_out, None)
                    .expect("access");
            }
            csr_nb_out
                .iter()
                .step_by(4099)
                .fold(0usize, |acc, v| acc ^ v.to_bits() as usize)
        },
    );

    // Sparse CSR blocking file-backed reference benches. These include waiting
    // for file IO and decode and are not used as the databank no-blocking
    // throughput target.
    let csr_lz4_cells = csr_nb_cells;
    let csr_lz4_genes = csr_nb_genes;
    let csr_lz4_nnz_per_cell = csr_nb_nnz_per_cell;
    let csr_lz4_chunk_cells = csr_nb_chunk_cells;
    let csr_lz4_chunk_len = csr_nb_chunk_len;
    let csr_lz4_nnz = csr_nb_nnz;
    let (csr_lz4_indptr, csr_lz4_indices_chunks, csr_lz4_data_chunks) = make_csr_u32_f32_chunks_lz4(
        csr_lz4_cells,
        csr_lz4_genes,
        csr_lz4_nnz_per_cell,
        csr_lz4_chunk_len,
    );
    let (csr_lz4_indices_path, csr_lz4_indices_locations) =
        write_chunks_file("stress-csr-lz4-indices", &csr_lz4_indices_chunks);
    let (csr_lz4_data_path, csr_lz4_data_locations) =
        write_chunks_file("stress-csr-lz4-data", &csr_lz4_data_chunks);
    let csr_lz4_id = bank
        .register_sparse_csr(SparseCsrDatasetMeta {
            gene_names: (0..csr_lz4_genes)
                .map(|idx| format!("gene_{idx}"))
                .collect(),
            indptr: csr_lz4_indptr,
            indices: ArrayMeta {
                shape: vec![csr_lz4_nnz],
                chunk_shape: vec![csr_lz4_chunk_len],
                chunk_grid_shape: vec![csr_lz4_indices_chunks.len()],
                dtype: DType::U32,
                order: ArrayOrder::C,
                codec: ArrayCodecMeta::CodecJson(r#"{"id":"lz4"}"#.to_string()),
                chunks: ChunkStoreMeta::FileOffset {
                    path: csr_lz4_indices_path,
                    locations: csr_lz4_indices_locations,
                },
                variable_chunks: false,
                chunk_boundaries: None,
            },
            data: ArrayMeta {
                shape: vec![csr_lz4_nnz],
                chunk_shape: vec![csr_lz4_chunk_len],
                chunk_grid_shape: vec![csr_lz4_data_chunks.len()],
                dtype: DType::F32,
                order: ArrayOrder::C,
                codec: ArrayCodecMeta::CodecJson(r#"{"id":"lz4"}"#.to_string()),
                chunks: ChunkStoreMeta::FileOffset {
                    path: csr_lz4_data_path,
                    locations: csr_lz4_data_locations,
                },
                variable_chunks: false,
                chunk_boundaries: None,
            },
            index_dtype: DType::U32,
            num_cells: csr_lz4_cells,
            num_genes: csr_lz4_genes,
        })
        .expect("register sparse csr lz4");
    let csr_lz4_repeat: Vec<usize> = (0..64).collect();
    let csr_lz4_scattered: Vec<usize> = (0..64).map(|idx| idx * csr_lz4_chunk_cells).collect();
    let mut csr_lz4_out = vec![0.0f32; csr_lz4_repeat.len() * csr_lz4_genes];
    support::bench(
        config,
        "databank/blocking_sparse_csr_lz4_file_repeat_chunk_checked_64x4096",
        192,
        Some(csr_lz4_out.len() * 4),
        || {
            bank.access_cells(csr_lz4_id, &csr_lz4_repeat, &mut csr_lz4_out, None)
                .expect("access");
            csr_lz4_out
                .iter()
                .step_by(4099)
                .fold(0usize, |acc, v| acc ^ v.to_bits() as usize)
        },
    );
    support::bench(
        config,
        "databank/blocking_sparse_csr_lz4_file_repeat_chunk_unchecked_64x4096",
        256,
        Some(csr_lz4_out.len() * 4),
        || {
            unsafe {
                bank.access_cells_unchecked(csr_lz4_id, &csr_lz4_repeat, &mut csr_lz4_out, None)
                    .expect("access");
            }
            csr_lz4_out
                .iter()
                .step_by(4099)
                .fold(0usize, |acc, v| acc ^ v.to_bits() as usize)
        },
    );
    support::bench(
        config,
        "databank/blocking_sparse_csr_lz4_file_scattered_chunks_checked_64x4096",
        128,
        Some(csr_lz4_out.len() * 4),
        || {
            bank.access_cells(csr_lz4_id, &csr_lz4_scattered, &mut csr_lz4_out, None)
                .expect("access");
            csr_lz4_out
                .iter()
                .step_by(4099)
                .fold(0usize, |acc, v| acc ^ v.to_bits() as usize)
        },
    );
    support::bench(
        config,
        "databank/blocking_sparse_csr_lz4_file_scattered_chunks_unchecked_64x4096",
        192,
        Some(csr_lz4_out.len() * 4),
        || {
            unsafe {
                bank.access_cells_unchecked(csr_lz4_id, &csr_lz4_scattered, &mut csr_lz4_out, None)
                    .expect("access");
            }
            csr_lz4_out
                .iter()
                .step_by(4099)
                .fold(0usize, |acc, v| acc ^ v.to_bits() as usize)
        },
    );

    // register/unregister lifecycle (generation churn) — &mut, before Arc move.
    support::bench(
        config,
        "databank/ST_register_unregister_cycle",
        256,
        None,
        || {
            let id = bank
                .register_dense_2d(Dense2DMeta {
                    gene_names: (0..genes).map(|idx| format!("g{idx}")).collect(),
                    data: ArrayMeta {
                        shape: vec![cells, genes],
                        chunk_shape: vec![chunk_rows, chunk_cols],
                        chunk_grid_shape: chunk_grid.clone(),
                        dtype: DType::U32,
                        order: ArrayOrder::C,
                        codec: ArrayCodecMeta::Uncompressed,
                        chunks: ChunkStoreMeta::Memory {
                            chunks: mem_chunks.clone(),
                        },
                        variable_chunks: false,
                        chunk_boundaries: None,
                    },
                })
                .expect("register");
            bank.unregister(id).expect("unregister");
            id.slot as usize
        },
    );

    // DataBank is !Send (GeneNameView holds *const u8), so shared-bank
    // concurrency is not its usage model. Instead, verify that independent
    // DataBank instances run in parallel without a hidden global lock.
    stress_databank_mt_instances(
        config,
        mem_chunks.clone(),
        cells,
        genes,
        chunk_rows,
        chunk_cols,
    );
    stress_databank_mt_by_gene_names(
        config,
        mem_chunks.clone(),
        cells,
        genes,
        chunk_rows,
        chunk_cols,
    );
    stress_databank_mt_owned(
        config,
        mem_chunks.clone(),
        cells,
        genes,
        chunk_rows,
        chunk_cols,
    );
    stress_databank_mt_scheduled_by_gene_names(
        config,
        mem_chunks.clone(),
        cells,
        genes,
        chunk_rows,
        chunk_cols,
    );
    stress_databank_mt_lifecycle(config);

    // --- scatter-focused: large output to isolate databank-side scatter
    // (copy decoded bytes into caller `out`). This runs on the caller thread,
    // NOT the access CPU pool — verify whether single-thread scatter becomes
    // a ceiling at large output sizes.
    {
        let s_cells = 1024usize;
        let s_genes = 4096usize;
        let s_chunk_rows = 32usize;
        let s_chunk_cols = 256usize;
        let s_grid = vec![s_cells / s_chunk_rows, s_genes / s_chunk_cols];
        let s_chunks = make_dense_u32_chunks(s_cells, s_genes, s_chunk_rows, s_chunk_cols);
        let mut s_bank = DataBank::new(DataBankConfig::default()).expect("scatter bank");
        let s_id = s_bank
            .register_dense_2d(Dense2DMeta {
                gene_names: (0..s_genes).map(|i| format!("g{i}")).collect(),
                data: ArrayMeta {
                    shape: vec![s_cells, s_genes],
                    chunk_shape: vec![s_chunk_rows, s_chunk_cols],
                    chunk_grid_shape: s_grid,
                    dtype: DType::U32,
                    order: ArrayOrder::C,
                    codec: ArrayCodecMeta::Uncompressed,
                    chunks: ChunkStoreMeta::Memory { chunks: s_chunks },
                    variable_chunks: false,
                    chunk_boundaries: None,
                },
            })
            .expect("register scatter dataset");
        // 256 cells × 4096 genes × 4B = 4 MiB output per op — scatter-bound.
        let s_sel: Vec<usize> = (0..256).map(|i| i * 3 % s_cells).collect();
        let mut s_out = vec![0u32; s_sel.len() * s_genes];
        support::bench(
            config,
            "databank/ST_dense2d_scatter_4mib_output",
            96,
            Some(s_out.len() * 4),
            || {
                s_bank
                    .access_cells(s_id, &s_sel, &mut s_out, None)
                    .expect("access");
                s_out
                    .iter()
                    .step_by(1023)
                    .fold(0usize, |acc, v| acc ^ usize::try_from(*v).unwrap_or(0))
            },
        );
        // Same workload, zstd — decode pool vs scatter interaction.
        let s_zstd_chunks =
            make_dense_u32_chunks_zstd(s_cells, s_genes, s_chunk_rows, s_chunk_cols, 3);
        let mut s_bank_z = DataBank::new(DataBankConfig::default()).expect("scatter zstd bank");
        let s_zid = s_bank_z
            .register_dense_2d(Dense2DMeta {
                gene_names: (0..s_genes).map(|i| format!("g{i}")).collect(),
                data: ArrayMeta {
                    shape: vec![s_cells, s_genes],
                    chunk_shape: vec![s_chunk_rows, s_chunk_cols],
                    chunk_grid_shape: vec![s_cells / s_chunk_rows, s_genes / s_chunk_cols],
                    dtype: DType::U32,
                    order: ArrayOrder::C,
                    codec: ArrayCodecMeta::CodecJson(r#"{"id":"zstd","level":3}"#.to_string()),
                    chunks: ChunkStoreMeta::Memory {
                        chunks: s_zstd_chunks,
                    },
                    variable_chunks: false,
                    chunk_boundaries: None,
                },
            })
            .expect("register scatter zstd dataset");
        let mut s_zout = vec![0u32; s_sel.len() * s_genes];
        support::bench(
            config,
            "databank/ST_dense2d_scatter_zstd_4mib_output",
            64,
            Some(s_zout.len() * 4),
            || {
                s_bank_z
                    .access_cells(s_zid, &s_sel, &mut s_zout, None)
                    .expect("access");
                s_zout
                    .iter()
                    .step_by(1023)
                    .fold(0usize, |acc, v| acc ^ usize::try_from(*v).unwrap_or(0))
            },
        );
    }

    let _ = std::fs::remove_file(fo_path);
    let _ = std::fs::remove_file(edge_fo_path);
    let _ = std::fs::remove_dir_all(dir_dir);
}

fn stress_databank_mt_instances(
    config: BenchConfig,
    chunks: Vec<Arc<[u8]>>,
    cells: usize,
    genes: usize,
    chunk_rows: usize,
    chunk_cols: usize,
) {
    let selected: Vec<usize> = (0..32).map(|i| i * 7 % cells).collect();
    let chunk_grid = vec![cells / chunk_rows, genes / chunk_cols];
    let bytes_per_op = selected.len() * genes * 4;

    for &threads in [2usize, 4, 8].iter() {
        let per_thread = (config.iterations(96) / threads).max(1);
        let barrier = Arc::new(Barrier::new(threads));
        let chunks = Arc::new(chunks.clone());
        let selected = Arc::new(selected.clone());
        let chunk_grid = Arc::new(chunk_grid.clone());

        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                let chunks = Arc::clone(&chunks);
                let selected = Arc::clone(&selected);
                let chunk_grid = Arc::clone(&chunk_grid);
                thread::spawn(move || {
                    let mut bank = DataBank::new(DataBankConfig::default()).expect("bank");
                    let id = bank
                        .register_dense_2d(Dense2DMeta {
                            gene_names: (0..genes).map(|i| format!("g{i}")).collect(),
                            data: ArrayMeta {
                                shape: vec![cells, genes],
                                chunk_shape: vec![chunk_rows, chunk_cols],
                                chunk_grid_shape: (*chunk_grid).clone(),
                                dtype: DType::U32,
                                order: ArrayOrder::C,
                                codec: ArrayCodecMeta::Uncompressed,
                                chunks: ChunkStoreMeta::Memory {
                                    chunks: (*chunks).clone(),
                                },
                                variable_chunks: false,
                                chunk_boundaries: None,
                            },
                        })
                        .expect("register");
                    let mut out = vec![0u32; selected.len() * genes];
                    barrier.wait();
                    let t0 = Instant::now();
                    let mut sum = 0usize;
                    for _ in 0..per_thread {
                        bank.access_cells(id, &selected, &mut out, None)
                            .expect("access");
                        sum ^= out
                            .iter()
                            .step_by(257)
                            .fold(0usize, |a, v| a ^ usize::try_from(*v).unwrap_or(0));
                    }
                    (sum, t0.elapsed())
                })
            })
            .collect();
        let results: Vec<(usize, std::time::Duration)> = handles
            .into_iter()
            .map(|h| h.join().expect("instance thread"))
            .collect();
        let checksum: usize = results.iter().map(|(s, _)| *s).fold(0, |a, b| a ^ b);
        black_box(checksum);
        let elapsed = results.iter().map(|(_, e)| *e).max().unwrap_or_default();
        let total_ops = per_thread * threads;
        let seconds = elapsed.as_secs_f64();
        let mib = bytes_per_op as f64 * total_ops as f64 / (1024.0 * 1024.0);
        println!(
            "databank/MT_instances_dense2d_checked_32x1024/t{threads}    threads={threads} ops={total_ops:<8} elapsed_s={seconds:.4} throughput_mib_s={:.1} kops={:.1}",
            mib / seconds,
            total_ops as f64 / seconds / 1000.0
        );
    }
}

/// Per-instance by-gene-names scaling: each thread owns a DataBank and selects
/// a column subset by name with the Zero missing-gene policy.
fn stress_databank_mt_by_gene_names(
    config: BenchConfig,
    chunks: Vec<Arc<[u8]>>,
    cells: usize,
    genes: usize,
    chunk_rows: usize,
    chunk_cols: usize,
) {
    let selected: Vec<usize> = (0..32).map(|i| i * 7 % cells).collect();
    let by_names: Vec<String> = (0..32).map(|i| format!("g{}", i * 7 % genes)).collect();
    let chunk_grid = vec![cells / chunk_rows, genes / chunk_cols];
    let bytes_per_op = selected.len() * by_names.len() * 4;

    for &threads in [2usize, 4, 8].iter() {
        let per_thread = (config.iterations(96) / threads).max(1);
        let barrier = Arc::new(Barrier::new(threads));
        let chunks = Arc::new(chunks.clone());
        let selected = Arc::new(selected.clone());
        let by_names = Arc::new(by_names.clone());
        let chunk_grid = Arc::new(chunk_grid.clone());

        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                let chunks = Arc::clone(&chunks);
                let selected = Arc::clone(&selected);
                let by_names = Arc::clone(&by_names);
                let chunk_grid = Arc::clone(&chunk_grid);
                thread::spawn(move || {
                    let mut bank = DataBank::new(DataBankConfig::default()).expect("bank");
                    let id = bank
                        .register_dense_2d(Dense2DMeta {
                            gene_names: (0..genes).map(|i| format!("g{i}")).collect(),
                            data: ArrayMeta {
                                shape: vec![cells, genes],
                                chunk_shape: vec![chunk_rows, chunk_cols],
                                chunk_grid_shape: (*chunk_grid).clone(),
                                dtype: DType::U32,
                                order: ArrayOrder::C,
                                codec: ArrayCodecMeta::Uncompressed,
                                chunks: ChunkStoreMeta::Memory {
                                    chunks: (*chunks).clone(),
                                },
                                variable_chunks: false,
                                chunk_boundaries: None,
                            },
                        })
                        .expect("register");
                    let mut out = vec![0u32; selected.len() * by_names.len()];
                    barrier.wait();
                    let t0 = Instant::now();
                    let mut sum = 0usize;
                    for _ in 0..per_thread {
                        bank.access_cells_by_gene_names(
                            id,
                            &selected,
                            &by_names,
                            &mut out,
                            None,
                            MissingGenePolicy::Zero,
                        )
                        .expect("access by gene names");
                        sum ^= out
                            .iter()
                            .step_by(131)
                            .fold(0usize, |a, v| a ^ usize::try_from(*v).unwrap_or(0));
                    }
                    (sum, t0.elapsed())
                })
            })
            .collect();
        let results: Vec<(usize, std::time::Duration)> = handles
            .into_iter()
            .map(|h| h.join().expect("by-gene-names thread"))
            .collect();
        let checksum: usize = results.iter().map(|(s, _)| *s).fold(0, |a, b| a ^ b);
        black_box(checksum);
        let elapsed = results.iter().map(|(_, e)| *e).max().unwrap_or_default();
        let total_ops = per_thread * threads;
        let seconds = elapsed.as_secs_f64();
        let mib = bytes_per_op as f64 * total_ops as f64 / (1024.0 * 1024.0);
        println!(
            "databank/MT_instances_by_gene_names_zero_32x32/t{threads}    threads={threads} ops={total_ops:<8} elapsed_s={seconds:.4} throughput_mib_s={:.1} kops={:.1}",
            mib / seconds,
            total_ops as f64 / seconds / 1000.0
        );
    }
}

/// Per-instance owned-output scaling: each thread owns a DataBank and receives
/// a databank-allocated `Vec<u32>` per access.
fn stress_databank_mt_owned(
    config: BenchConfig,
    chunks: Vec<Arc<[u8]>>,
    cells: usize,
    genes: usize,
    chunk_rows: usize,
    chunk_cols: usize,
) {
    let selected: Vec<usize> = (0..32).map(|i| i * 7 % cells).collect();
    let chunk_grid = vec![cells / chunk_rows, genes / chunk_cols];
    let bytes_per_op = selected.len() * genes * 4;

    for &threads in [2usize, 4, 8].iter() {
        let per_thread = (config.iterations(64) / threads).max(1);
        let barrier = Arc::new(Barrier::new(threads));
        let chunks = Arc::new(chunks.clone());
        let selected = Arc::new(selected.clone());
        let chunk_grid = Arc::new(chunk_grid.clone());

        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                let chunks = Arc::clone(&chunks);
                let selected = Arc::clone(&selected);
                let chunk_grid = Arc::clone(&chunk_grid);
                thread::spawn(move || {
                    let mut bank = DataBank::new(DataBankConfig::default()).expect("bank");
                    let id = bank
                        .register_dense_2d(Dense2DMeta {
                            gene_names: (0..genes).map(|i| format!("g{i}")).collect(),
                            data: ArrayMeta {
                                shape: vec![cells, genes],
                                chunk_shape: vec![chunk_rows, chunk_cols],
                                chunk_grid_shape: (*chunk_grid).clone(),
                                dtype: DType::U32,
                                order: ArrayOrder::C,
                                codec: ArrayCodecMeta::Uncompressed,
                                chunks: ChunkStoreMeta::Memory {
                                    chunks: (*chunks).clone(),
                                },
                                variable_chunks: false,
                                chunk_boundaries: None,
                            },
                        })
                        .expect("register");
                    barrier.wait();
                    let t0 = Instant::now();
                    let mut sum = 0usize;
                    for _ in 0..per_thread {
                        let out: Vec<u32> = bank.access_cells_owned(id, &selected).expect("owned");
                        sum ^= out[0] as usize ^ out.len();
                    }
                    (sum, t0.elapsed())
                })
            })
            .collect();
        let results: Vec<(usize, std::time::Duration)> = handles
            .into_iter()
            .map(|h| h.join().expect("owned thread"))
            .collect();
        let checksum: usize = results.iter().map(|(s, _)| *s).fold(0, |a, b| a ^ b);
        black_box(checksum);
        let elapsed = results.iter().map(|(_, e)| *e).max().unwrap_or_default();
        let total_ops = per_thread * threads;
        let seconds = elapsed.as_secs_f64();
        let mib = bytes_per_op as f64 * total_ops as f64 / (1024.0 * 1024.0);
        println!(
            "databank/MT_instances_owned_32x1024/t{threads}    threads={threads} ops={total_ops:<8} elapsed_s={seconds:.4} throughput_mib_s={:.1} kops={:.1}",
            mib / seconds,
            total_ops as f64 / seconds / 1000.0
        );
    }
}

/// Per-instance scheduled projected prefetch scaling: each thread owns a
/// DataBank and repeatedly builds a small scheduled prefetcher with a missing
/// gene that must be zero-filled.
fn stress_databank_mt_scheduled_by_gene_names(
    config: BenchConfig,
    chunks: Vec<Arc<[u8]>>,
    cells: usize,
    genes: usize,
    chunk_rows: usize,
    chunk_cols: usize,
) {
    let batches: Vec<Vec<usize>> = vec![
        (0..8).map(|i| i * 7 % cells).collect(),
        (0..4).map(|i| i * 11 % cells).collect(),
        Vec::new(),
    ];
    let by_names: Vec<String> = vec![
        "g1".to_string(),
        "missing_gene".to_string(),
        format!("g{}", genes / 2),
        format!("g{}", genes - 1),
    ];
    let chunk_grid = vec![cells / chunk_rows, genes / chunk_cols];
    let bytes_per_op = batches.iter().map(Vec::len).sum::<usize>() * by_names.len() * 4;

    for &threads in [2usize, 4, 8].iter() {
        let per_thread = (config.iterations(32) / threads).max(1);
        let barrier = Arc::new(Barrier::new(threads));
        let chunks = Arc::new(chunks.clone());
        let batches = Arc::new(batches.clone());
        let by_names = Arc::new(by_names.clone());
        let chunk_grid = Arc::new(chunk_grid.clone());

        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                let chunks = Arc::clone(&chunks);
                let batches = Arc::clone(&batches);
                let by_names = Arc::clone(&by_names);
                let chunk_grid = Arc::clone(&chunk_grid);
                thread::spawn(move || {
                    let mut bank = DataBank::new(DataBankConfig::default()).expect("bank");
                    let id = bank
                        .register_dense_2d(Dense2DMeta {
                            gene_names: (0..genes).map(|i| format!("g{i}")).collect(),
                            data: ArrayMeta {
                                shape: vec![cells, genes],
                                chunk_shape: vec![chunk_rows, chunk_cols],
                                chunk_grid_shape: (*chunk_grid).clone(),
                                dtype: DType::U32,
                                order: ArrayOrder::C,
                                codec: ArrayCodecMeta::Uncompressed,
                                chunks: ChunkStoreMeta::Memory {
                                    chunks: (*chunks).clone(),
                                },
                                variable_chunks: false,
                                chunk_boundaries: None,
                            },
                        })
                        .expect("register");
                    barrier.wait();
                    let t0 = Instant::now();
                    let mut sum = 0usize;
                    for _ in 0..per_thread {
                        let prefetch = bank
                            .prefetch_cells_scheduled_by_gene_names::<u32, _, _>(
                                id,
                                (*batches).clone(),
                                by_names.as_slice(),
                                MissingGenePolicy::Zero,
                                ScheduledPrefetchConfig::default(),
                            )
                            .expect("scheduled projected prefetch");
                        for batch in prefetch {
                            let batch = batch.expect("prefetch batch");
                            sum ^= batch.buffer.len() ^ batch.num_genes ^ batch.cells.len();
                        }
                    }
                    (sum, t0.elapsed())
                })
            })
            .collect();
        let results: Vec<(usize, std::time::Duration)> = handles
            .into_iter()
            .map(|h| h.join().expect("scheduled by-gene thread"))
            .collect();
        let checksum: usize = results.iter().map(|(s, _)| *s).fold(0, |a, b| a ^ b);
        black_box(checksum);
        let elapsed = results.iter().map(|(_, e)| *e).max().unwrap_or_default();
        let total_ops = per_thread * threads;
        let seconds = elapsed.as_secs_f64();
        let mib = bytes_per_op as f64 * total_ops as f64 / (1024.0 * 1024.0);
        println!(
            "databank/MT_instances_scheduled_by_gene_names_zero_missing/t{threads}    threads={threads} ops={total_ops:<8} elapsed_s={seconds:.4} throughput_mib_s={:.1} kops={:.1}",
            mib / seconds,
            total_ops as f64 / seconds / 1000.0
        );
    }
}

fn stress_databank_mt_lifecycle(config: BenchConfig) {
    let cells = 64usize;
    let genes = 64usize;
    let chunk_rows = 16usize;
    let chunk_cols = 16usize;
    let chunks = Arc::new(make_dense_u32_chunks(cells, genes, chunk_rows, chunk_cols));
    let selected = Arc::new(vec![0usize, 7, 31, 63]);
    let bytes_per_op = selected.len() * genes * std::mem::size_of::<u32>();

    for &threads in [2usize, 4].iter() {
        let per_thread = (config.iterations(16) / threads).max(1);
        let barrier = Arc::new(Barrier::new(threads));
        let chunks = Arc::clone(&chunks);
        let selected = Arc::clone(&selected);
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                let chunks = Arc::clone(&chunks);
                let selected = Arc::clone(&selected);
                thread::spawn(move || {
                    barrier.wait();
                    let t0 = Instant::now();
                    let mut sum = 0usize;
                    for _ in 0..per_thread {
                        let mut bank = DataBank::new(DataBankConfig::default()).expect("bank");
                        let id = bank
                            .register_dense_2d(Dense2DMeta {
                                gene_names: (0..genes).map(|i| format!("g{i}")).collect(),
                                data: ArrayMeta {
                                    shape: vec![cells, genes],
                                    chunk_shape: vec![chunk_rows, chunk_cols],
                                    chunk_grid_shape: vec![cells / chunk_rows, genes / chunk_cols],
                                    dtype: DType::U32,
                                    order: ArrayOrder::C,
                                    codec: ArrayCodecMeta::Uncompressed,
                                    chunks: ChunkStoreMeta::Memory {
                                        chunks: (*chunks).clone(),
                                    },
                                    variable_chunks: false,
                                    chunk_boundaries: None,
                                },
                            })
                            .expect("register lifecycle dataset");
                        let mut out = vec![0u32; selected.len() * genes];
                        bank.access_cells(id, &selected, &mut out, None)
                            .expect("lifecycle access");
                        sum ^= out[0] as usize ^ out.len();
                    }
                    (sum, t0.elapsed())
                })
            })
            .collect();
        let results: Vec<(usize, std::time::Duration)> = handles
            .into_iter()
            .map(|h| h.join().expect("lifecycle thread"))
            .collect();
        let checksum: usize = results.iter().map(|(s, _)| *s).fold(0, |a, b| a ^ b);
        black_box(checksum);
        let elapsed = results.iter().map(|(_, e)| *e).max().unwrap_or_default();
        let total_ops = per_thread * threads;
        let seconds = elapsed.as_secs_f64();
        let mib = bytes_per_op as f64 * total_ops as f64 / (1024.0 * 1024.0);
        println!(
            "databank/MT_lifecycle_create_register_drop_4x64/t{threads}    threads={threads} ops={total_ops:<8} elapsed_s={seconds:.4} throughput_mib_s={:.1} kops={:.1}",
            mib / seconds,
            total_ops as f64 / seconds / 1000.0
        );
    }
}

fn stress_databank_config() -> DataBankConfig {
    let mut config = DataBankConfig::default();
    config.access_config.scheduler_shards = 4;
    config.access_config.cpu.num_workers = 8;
    config.decode_config.num_workers = 8;
    config.fill_config.num_workers = 4;
    config
}

// ---------------------------------------------------------------------------
// Missing-rate × multi-threaded decode: each thread decodes an independent
// payload whose zero/NaN fraction varies. Exposes whether scatter / decode
// scaling holds as payloads get sparser.
// ---------------------------------------------------------------------------

fn stress_missing_rate(config: BenchConfig) {
    let raw_len = 256 * 1024;
    for &missing in &[0u32, 250, 500, 950] {
        let profile = DataProfile {
            dtype: DType::U32,
            dist: DataDist::Uniform,
            missing_permille: missing,
            chunk_bytes: raw_len,
            num_chunks: 1,
            seed: 17,
        };
        // zstd decode across threads at each missing rate.
        let raw: Arc<[u8]> = Arc::from(profile.generate());
        let spec = CodecSpec::Zstd(ZstdCodecConfig::default());
        let encoded = Arc::from(encode_for_spec(&spec, &raw[..]));
        let codec = spec.build();
        for &threads in THREAD_COUNTS {
            let codec = Arc::clone(&codec);
            let encoded = Arc::clone(&encoded);
            stress_mt(
                config,
                &format!("missing/MT_decode_zstd_miss{missing}_256k/t{threads}"),
                threads,
                192,
                Some(raw_len),
                move |_| {
                    let mut out = vec![0u8; raw_len];
                    let written = codec
                        .decode_into(&encoded, DecodeBuffer::new(&mut out), Some(raw_len))
                        .expect("decode");
                    written ^ out[written / 2] as usize
                },
            );
        }

        // Per-instance dense2d access at each missing rate: each thread owns a
        // DataBank with sparse-payload chunks.
        let cells = 256usize;
        let genes = 128usize;
        let chunk_rows = 16usize;
        let chunk_cols = 32usize;
        let item_size = std::mem::size_of::<u32>();
        let chunk_bytes = chunk_rows * chunk_cols * item_size;
        let chunks: Arc<Vec<Arc<[u8]>>> = Arc::new(
            (0..(cells / chunk_rows * genes / chunk_cols))
                .map(|chunk_idx| {
                    let p = DataProfile {
                        dtype: DType::U32,
                        dist: DataDist::Uniform,
                        missing_permille: missing,
                        chunk_bytes,
                        num_chunks: 1,
                        seed: 17 + chunk_idx as u64,
                    };
                    Arc::from(p.generate().into_boxed_slice())
                })
                .collect(),
        );
        let selected: Arc<Vec<usize>> = Arc::new((0..16).map(|i| i * 7 % cells).collect());
        let bytes_per_op = selected.len() * genes * item_size;
        for &threads in [2usize, 4, 8].iter() {
            let per_thread = (config.iterations(64) / threads).max(1);
            let barrier = Arc::new(Barrier::new(threads));
            let chunks = Arc::clone(&chunks);
            let selected = Arc::clone(&selected);
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    let barrier = Arc::clone(&barrier);
                    let chunks = Arc::clone(&chunks);
                    let selected = Arc::clone(&selected);
                    thread::spawn(move || {
                        let mut bank = DataBank::new(DataBankConfig::default()).expect("bank");
                        let id = bank
                            .register_dense_2d(Dense2DMeta {
                                gene_names: (0..genes).map(|i| format!("g{i}")).collect(),
                                data: ArrayMeta {
                                    shape: vec![cells, genes],
                                    chunk_shape: vec![chunk_rows, chunk_cols],
                                    chunk_grid_shape: vec![cells / chunk_rows, genes / chunk_cols],
                                    dtype: DType::U32,
                                    order: ArrayOrder::C,
                                    codec: ArrayCodecMeta::Uncompressed,
                                    chunks: ChunkStoreMeta::Memory {
                                        chunks: (*chunks).clone(),
                                    },
                                    variable_chunks: false,
                                    chunk_boundaries: None,
                                },
                            })
                            .expect("register");
                        let mut out = vec![0u32; selected.len() * genes];
                        barrier.wait();
                        let t0 = Instant::now();
                        let mut sum = 0usize;
                        for _ in 0..per_thread {
                            bank.access_cells(id, &selected, &mut out, None)
                                .expect("access");
                            sum ^= out
                                .iter()
                                .step_by(131)
                                .fold(0usize, |a, v| a ^ usize::try_from(*v).unwrap_or(0));
                        }
                        (sum, t0.elapsed())
                    })
                })
                .collect();
            let results: Vec<(usize, std::time::Duration)> = handles
                .into_iter()
                .map(|h| h.join().expect("missing-rate thread"))
                .collect();
            let checksum: usize = results.iter().map(|(s, _)| *s).fold(0, |a, b| a ^ b);
            black_box(checksum);
            let elapsed = results.iter().map(|(_, e)| *e).max().unwrap_or_default();
            let total_ops = per_thread * threads;
            let seconds = elapsed.as_secs_f64();
            let mib = bytes_per_op as f64 * total_ops as f64 / (1024.0 * 1024.0);
            println!(
                "missing/MT_instances_dense2d_miss{missing}_16x128/t{threads}    threads={threads} ops={total_ops:<8} elapsed_s={seconds:.4} throughput_mib_s={:.1} kops={:.1}",
                mib / seconds,
                total_ops as f64 / seconds / 1000.0
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Scale × multi-threaded decode: chunk-size scaling under concurrency. Each
// thread decodes an independent zstd payload so the bench measures per-thread
// decode cost without cross-thread buffer sharing.
// ---------------------------------------------------------------------------

fn stress_scale(config: BenchConfig) {
    let spec = CodecSpec::Zstd(ZstdCodecConfig::default());
    let codec = spec.build();
    for &raw_len in &[16 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024] {
        let profile = DataProfile {
            dtype: DType::U32,
            dist: DataDist::Uniform,
            missing_permille: 0,
            chunk_bytes: raw_len,
            num_chunks: 1,
            seed: 17,
        };
        let raw: Arc<[u8]> = Arc::from(profile.generate());
        let encoded = Arc::from(encode_for_spec(&spec, &raw[..]));
        for &threads in THREAD_COUNTS {
            let codec = Arc::clone(&codec);
            let encoded = Arc::clone(&encoded);
            stress_mt(
                config,
                &format!(
                    "scale/MT_decode_zstd_{}/t{threads}",
                    support::data::fmt_bytes(raw_len)
                ),
                threads,
                192,
                Some(raw_len),
                move |_| {
                    let mut out = vec![0u8; raw_len];
                    let written = codec
                        .decode_into(&encoded, DecodeBuffer::new(&mut out), Some(raw_len))
                        .expect("decode");
                    written ^ out[written / 2] as usize
                },
            );
        }
    }
}
