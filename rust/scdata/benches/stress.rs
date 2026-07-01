//! Multi-threaded stress benchmarks with profile snapshots.

mod support;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::oneshot;

use _scdata::access::{
    AccessConfig, AccessCpuConfig, AccessProfile, AccessRequest, AccessScheduler, ChunkKey, FileRef,
};
use _scdata::codecs::{
    codec_from_spec, codecs_profile_registry, DecodePool, DecodePoolConfig, DecodeRequest,
    Lz4CodecConfig, ZstdCodecConfig,
};
use _scdata::iopool::{BaseIoConfig, IoCommand, IoConfig, IoPool, ThreadedConfig};
use _scdata::profile::ProfileRegistry;

use support::backends::{CodecDecode, SliceIo};
use support::codecs::encode_for_spec;
use support::{payload, profile_runtime, stress_mt, BenchConfig};

fn main() {
    let config = BenchConfig::from_env();
    let threads = stress_threads();
    println!("scdata stress benchmarks: profile-first rewrite threads={threads:?}");
    stress_decode_pool(config, &threads);
    stress_iopool(config, &threads);
    stress_access(config, &threads);
}

fn stress_decode_pool(config: BenchConfig, threads: &[usize]) {
    let raw = payload(64 * 1024);
    let expected_size = raw.len();
    let spec = _scdata::codecs::CodecSpec::Zstd(ZstdCodecConfig {
        level: Some(3),
        checksum: Some(false),
    });
    let encoded: Arc<[u8]> = encode_for_spec(&spec, &raw).into();
    let runtime = profile_runtime("stress-codecs", codecs_profile_registry);
    let pool = Arc::new(
        DecodePool::with_profile(
            DecodePoolConfig {
                num_workers: 4,
                queue_capacity: 512,
                cpus: None,
            },
            runtime.clone(),
        )
        .expect("decode pool"),
    );

    for &thread_count in threads {
        let pool = Arc::clone(&pool);
        let encoded = Arc::clone(&encoded);
        let spec = spec.clone();
        let round = runtime.start();
        stress_mt(
            config,
            "stress/codecs/decode_pool_zstd_64k",
            thread_count,
            4096,
            Some(expected_size),
            move |_| {
                let bytes = pool
                    .submit(
                        DecodeRequest::from_spec(&spec, Arc::clone(&encoded))
                            .with_expected_size(expected_size),
                    )
                    .expect("submit decode")
                    .blocking_recv()
                    .expect("decode");
                bytes.len() ^ bytes[bytes.len() / 2] as usize
            },
        );
        support::maybe_print_profile_snapshot(
            config,
            &format!("stress/codecs/decode_pool_zstd_64k/t{thread_count}"),
            &round.end(),
        );
    }
}

fn stress_iopool(config: BenchConfig, threads: &[usize]) {
    let path = support::bench_data_dir().join(format!("stress-iopool-{}.bin", std::process::id()));
    std::fs::write(&path, payload(8 * 1024 * 1024)).expect("write iopool fixture");
    let runtime = profile_runtime("stress-iopool", ProfileRegistry::new);
    let pool = Arc::new(
        IoPool::new_with_profile(
            IoConfig::Threaded(ThreadedConfig {
                base: BaseIoConfig {
                    max_in_flight: 512,
                    queue_capacity: 2048,
                    priority_levels: 2,
                    queue_shards: 4,
                    assume_non_overlapping_reads: true,
                },
                num_workers: 4,
                cpus: None,
            }),
            runtime.clone(),
        )
        .expect("iopool"),
    );
    let file = pool.register_readonly_file(&path).expect("register file");
    let offset_counter = Arc::new(AtomicUsize::new(0));

    for &thread_count in threads {
        let pool = Arc::clone(&pool);
        let offset_counter = Arc::clone(&offset_counter);
        let round = runtime.start();
        stress_mt(
            config,
            "stress/iopool/threaded_read_64k",
            thread_count,
            4096,
            Some(64 * 1024),
            move |_| {
                let idx = offset_counter.fetch_add(1, Ordering::Relaxed);
                let offset = ((idx * 64 * 1024) % (8 * 1024 * 1024 - 64 * 1024)) as u64;
                let bytes = pool
                    .submit(IoCommand::read(file, offset, 64 * 1024, 0))
                    .expect("submit read")
                    .blocking_recv_read()
                    .expect("read");
                bytes.len() ^ bytes[bytes.len() / 2] as usize
            },
        );
        support::maybe_print_profile_snapshot(
            config,
            &format!("stress/iopool/threaded_read_64k/t{thread_count}"),
            &round.end(),
        );
    }

    pool.unregister_file(file).expect("unregister file");
    let _ = std::fs::remove_file(path);
}

fn stress_access(config: BenchConfig, threads: &[usize]) {
    let raw = payload(64 * 1024);
    let expected_size = raw.len();
    let spec = _scdata::codecs::CodecSpec::Lz4(Lz4CodecConfig::default());
    let codec = codec_from_spec(&spec);
    let encoded = encode_for_spec(&spec, &raw);
    let key = ChunkKey::new(FileRef::new(19), 0, encoded.len());
    let runtime = profile_runtime("stress-access", _scdata::access::access_profile_registry);
    let access = Arc::new(
        AccessScheduler::spawn(
            AccessConfig {
                queue_capacity: 512,
                scheduler_shards: 4,
                cache_capacity_bytes: 32 * 1024 * 1024,
                memory_budget_bytes: 64 * 1024 * 1024,
                default_io_priority: 0,
                keep_decoded: true,
                cpu: AccessCpuConfig {
                    num_workers: 4,
                    queue_capacity: 512,
                    cpus: None,
                },
                profile: AccessProfile::from_runtime(runtime.clone()),
            },
            Box::new(SliceIo::new(Arc::from(encoded.into_boxed_slice()))),
            Box::new(CodecDecode),
        )
        .expect("access scheduler"),
    );

    for &thread_count in threads {
        let access = Arc::clone(&access);
        let codec = Arc::clone(&codec);
        let round = runtime.start();
        stress_mt(
            config,
            "stress/access/read_cache_lz4_64k",
            thread_count,
            4096,
            Some(expected_size),
            move |_| {
                let (tx, rx) = oneshot::channel();
                access
                    .send(AccessRequest::new(
                        key,
                        Arc::clone(&codec),
                        Some(expected_size),
                        tx,
                    ))
                    .expect("send access");
                let bytes = rx
                    .blocking_recv()
                    .expect("access reply")
                    .expect("access read");
                bytes.len() ^ bytes[bytes.len() / 2] as usize
            },
        );
        support::maybe_print_profile_snapshot(
            config,
            &format!("stress/access/read_cache_lz4_64k/t{thread_count}"),
            &round.end(),
        );
    }
}

fn stress_threads() -> Vec<usize> {
    std::env::var("SCDATA_STRESS_THREADS")
        .ok()
        .and_then(|value| {
            value
                .split(',')
                .map(|token| token.trim().parse::<usize>())
                .collect::<Result<Vec<_>, _>>()
                .ok()
        })
        .filter(|threads| !threads.is_empty())
        .unwrap_or_else(|| vec![2, 4, 8])
}
