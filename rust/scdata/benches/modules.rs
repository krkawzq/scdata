//! Module-level profile-first micro benchmarks.

mod support;

use std::hint::black_box;
use std::sync::Arc;

use tokio::sync::oneshot;

use _scdata::access::{
    AccessConfig, AccessCpuConfig, AccessItem, AccessProfile, AccessRequest, AccessScheduler,
    ChunkKey, FileRef, ScheduledAccessConfig,
};
use _scdata::codecs::{
    codec_from_json_str, codec_from_spec, codecs_profile_registry, DecodePool, DecodePoolConfig,
    DecodeRequest, Lz4CodecConfig, SharedCodec, ZstdCodecConfig,
};
use _scdata::databank::{
    ArrayCodecSpec, DType, DataBank, DataBankConfig, EdgeChunkLayout, FillConfig,
    MissingGenePolicy, NativeAccessConfig, ScheduledPrefetchConfig,
};
use _scdata::iopool::{BaseIoConfig, IoCommand, IoConfig, IoPool, ThreadedConfig};
use _scdata::profile::ProfileRegistry;

use support::backends::{CodecDecode, SliceIo};
use support::chunks::{
    dense1d_u32_rectilinear_spec, dense1d_u32_spec, dense2d_u32_spec, make_csr_u32_f32_chunks,
    make_dense1d_u32_chunks, make_dense1d_u32_variable_chunks, make_dense_u32_chunks,
    make_dense_u32_chunks_lz4, sparse_csr_u32_f32_spec,
};
use support::codecs::{decode_into_checksum, encode_for_spec};
use support::data::{DataDist, DataProfile};
use support::{
    bench, bench_existing_profile_round, bench_profiled, metric_value, payload, profile_runtime,
    BenchConfig, ProfileEnvGuard,
};

fn main() {
    let config = BenchConfig::from_env();
    println!("scdata module benchmarks: profile-first rewrite");
    bench_codec_surface(config);
    bench_decode_pool_profile(config);
    bench_iopool_profile(config);
    bench_access_profile(config);
    bench_databank_profile(config);
    bench_data_profiles(config);
}

fn bench_codec_surface(config: BenchConfig) {
    let raw = payload(256 * 1024);
    let zstd_spec = _scdata::codecs::CodecSpec::Zstd(ZstdCodecConfig {
        level: Some(3),
        checksum: Some(false),
    });
    let lz4_spec = _scdata::codecs::CodecSpec::Lz4(Lz4CodecConfig::default());
    let zstd = codec_from_spec(&zstd_spec);
    let lz4 = codec_from_spec(&lz4_spec);
    let zstd_encoded = encode_for_spec(&zstd_spec, &raw);
    let lz4_encoded = encode_for_spec(&lz4_spec, &raw);
    let json = r#"{"id":"zstd","level":3,"checksum":false}"#;

    bench(config, "codecs/parse_zstd_json", 20_000, None, || {
        codec_from_json_str(json).expect("codec json").name().len()
    });
    bench(config, "codecs/build_zstd_spec", 50_000, None, || {
        Arc::as_ptr(&codec_from_spec(&zstd_spec)) as *const () as usize
    });
    bench(
        config,
        "codecs/direct_decode_zstd_256k",
        256,
        Some(raw.len()),
        || decode_into_checksum(&zstd, &zstd_encoded, raw.len()),
    );
    bench(
        config,
        "codecs/direct_decode_lz4_256k",
        512,
        Some(raw.len()),
        || decode_into_checksum(&lz4, &lz4_encoded, raw.len()),
    );
}

fn bench_decode_pool_profile(config: BenchConfig) {
    let raw = payload(128 * 1024);
    let spec = _scdata::codecs::CodecSpec::Zstd(ZstdCodecConfig {
        level: Some(3),
        checksum: Some(false),
    });
    let encoded: Arc<[u8]> = encode_for_spec(&spec, &raw).into();
    let runtime = profile_runtime("bench-codecs-pool", codecs_profile_registry);
    let pool = DecodePool::with_profile(
        DecodePoolConfig {
            num_workers: 2,
            queue_capacity: 128,
            cpus: None,
        },
        runtime.clone(),
    )
    .expect("decode pool");

    let snapshot = bench_profiled(
        config,
        "codecs/profiled_decode_pool_zstd_128k",
        256,
        Some(raw.len()),
        &runtime,
        || {
            let decoded = pool
                .submit(
                    DecodeRequest::from_spec(&spec, Arc::clone(&encoded))
                        .with_expected_size(raw.len()),
                )
                .expect("submit decode")
                .blocking_recv()
                .expect("decode");
            decoded.len() ^ decoded[decoded.len() / 2] as usize
        },
    );
    assert!(metric_value(&snapshot, "codecs.submit", "calls") > 0);
    assert!(metric_value(&snapshot, "codecs.work", "calls") > 0);
}

fn bench_iopool_profile(config: BenchConfig) {
    let path = support::bench_data_dir().join(format!("modules-iopool-{}.bin", std::process::id()));
    std::fs::write(&path, payload(2 * 1024 * 1024)).expect("write iopool fixture");

    let runtime = profile_runtime("bench-iopool", ProfileRegistry::new);
    let pool =
        IoPool::new_with_profile(threaded_io_config(2, 128, 2), runtime.clone()).expect("iopool");
    let file = pool.register_readwrite_file(&path).expect("register file");

    let mut read_idx = 0usize;
    let read_snapshot = bench_profiled(
        config,
        "iopool/profiled_threaded_read_64k",
        512,
        Some(64 * 1024),
        &runtime,
        || {
            let offset = ((read_idx * 64 * 1024) % (2 * 1024 * 1024 - 64 * 1024)) as u64;
            read_idx += 1;
            let bytes = pool
                .submit(IoCommand::read(file, offset, 64 * 1024, 0))
                .expect("submit read")
                .blocking_recv_read()
                .expect("read");
            bytes.len() ^ bytes[bytes.len() / 2] as usize
        },
    );
    assert!(metric_value(&read_snapshot, "iopool.submit", "calls") > 0);
    assert!(metric_value(&read_snapshot, "iopool.operation", "read") > 0);

    let write_buf = payload(4096);
    let op_snapshot = bench_profiled(
        config,
        "iopool/profiled_write_metadata_sync",
        128,
        Some(write_buf.len()),
        &runtime,
        || {
            pool.submit(IoCommand::write(file, 128 * 1024, write_buf.clone(), 0))
                .expect("submit write")
                .blocking_recv()
                .expect("write");
            pool.submit(IoCommand::metadata(file, 0))
                .expect("submit metadata")
                .blocking_recv()
                .expect("metadata");
            pool.submit(IoCommand::sync_data(file, 0))
                .expect("submit sync")
                .blocking_recv()
                .expect("sync");
            write_buf.len()
        },
    );
    assert!(metric_value(&op_snapshot, "iopool.submit", "write") > 0);
    assert!(metric_value(&op_snapshot, "iopool.submit", "metadata") > 0);

    pool.unregister_file(file).expect("unregister file");
    let _ = std::fs::remove_file(path);
}

fn bench_access_profile(config: BenchConfig) {
    let raw = payload(64 * 1024);
    let spec = _scdata::codecs::CodecSpec::Lz4(Lz4CodecConfig::default());
    let codec = codec_from_spec(&spec);
    let encoded = encode_for_spec(&spec, &raw);
    let key = ChunkKey::new(FileRef::new(7), 0, encoded.len());
    let runtime = profile_runtime("bench-access", _scdata::access::access_profile_registry);
    let access = AccessScheduler::spawn(
        AccessConfig {
            queue_capacity: 128,
            scheduler_shards: 2,
            cache_capacity_bytes: 8 * 1024 * 1024,
            memory_budget_bytes: 16 * 1024 * 1024,
            default_io_priority: 0,
            keep_decoded: true,
            cpu: AccessCpuConfig {
                num_workers: 2,
                queue_capacity: 128,
                cpus: None,
            },
            profile: AccessProfile::from_runtime(runtime.clone()),
        },
        Box::new(SliceIo::new(Arc::from(encoded.into_boxed_slice()))),
        Box::new(CodecDecode),
    )
    .expect("access scheduler");

    let cold = bench_profiled(
        BenchConfig {
            warmups: 0,
            ..config
        },
        "access/profiled_cold_read_lz4_64k",
        256,
        Some(raw.len()),
        &runtime,
        || access_read_checksum(&access, key, Arc::clone(&codec), raw.len()),
    );
    assert!(metric_value(&cold, "access.command", "read") > 0);
    assert!(metric_value(&cold, "access.decode", "calls") > 0);

    let cached = bench_profiled(
        config,
        "access/profiled_cache_hit_read_lz4_64k",
        512,
        Some(raw.len()),
        &runtime,
        || access_read_checksum(&access, key, Arc::clone(&codec), raw.len()),
    );
    assert!(metric_value(&cached, "access.cache", "hits") > 0);

    let scheduled = bench_profiled(
        config,
        "access/profiled_scheduled_lz4_64k_x8",
        128,
        Some(raw.len() * 8),
        &runtime,
        || {
            let items = (0..8)
                .map(|_| AccessItem::new(key, Arc::clone(&codec), Some(raw.len())))
                .collect::<Vec<_>>();
            let iter = access
                .scheduled(
                    items,
                    ScheduledAccessConfig {
                        prefetch_step: 3,
                        decode_ahead_steps: 2,
                        ready_ahead_steps: 1,
                    },
                )
                .expect("scheduled access");
            iter.fold(0usize, |acc, result| {
                let bytes = result.expect("scheduled read");
                acc ^ bytes.len() ^ bytes[bytes.len() / 2] as usize
            })
        },
    );
    assert!(metric_value(&scheduled, "access.command", "scheduled-take") > 0);
}

fn bench_databank_profile(config: BenchConfig) {
    let _env = ProfileEnvGuard::enabled("bench-databank");
    let mut bank = DataBank::new(databank_config()).expect("databank");

    let cells = 96;
    let genes = 64;
    let dense_id = bank
        .register_dense_2d(dense2d_u32_spec(
            cells,
            genes,
            16,
            16,
            make_dense_u32_chunks(cells, genes, 16, 16),
            ArrayCodecSpec::Uncompressed,
            EdgeChunkLayout::Padded,
        ))
        .expect("register dense2d");
    let dense_lz4_id = bank
        .register_dense_2d(dense2d_u32_spec(
            cells,
            genes,
            16,
            16,
            make_dense_u32_chunks_lz4(cells, genes, 16, 16),
            ArrayCodecSpec::CodecJson(r#"{"id":"lz4"}"#.to_string()),
            EdgeChunkLayout::Padded,
        ))
        .expect("register dense2d lz4");
    let dense1d_id = bank
        .register_dense_1d(dense1d_u32_spec(
            cells,
            genes,
            512,
            make_dense1d_u32_chunks(cells, genes, 512),
            ArrayCodecSpec::Uncompressed,
        ))
        .expect("register dense1d");
    let boundaries = vec![0, 333, 1024, cells * genes];
    let rect_id = bank
        .register_dense_1d(dense1d_u32_rectilinear_spec(
            cells,
            genes,
            boundaries.clone(),
            make_dense1d_u32_variable_chunks(cells, genes, &boundaries),
            ArrayCodecSpec::Uncompressed,
        ))
        .expect("register rect dense1d");
    let (indptr, indices, data) = make_csr_u32_f32_chunks(cells, genes, 8, 256);
    let sparse_id = bank
        .register_sparse_csr(sparse_csr_u32_f32_spec(
            cells,
            genes,
            indptr,
            256,
            256,
            indices,
            data,
            ArrayCodecSpec::Uncompressed,
        ))
        .expect("register sparse");

    let batch = (0..32).collect::<Vec<_>>();
    bench_existing_profile_round(
        config,
        "databank/profiled_dense2d_access",
        256,
        Some(batch.len() * genes * std::mem::size_of::<u32>()),
        bank.profile(),
        || {
            let mut out = vec![0u32; batch.len() * genes];
            bank.access_cells(dense_id, &batch, &mut out, None)
                .expect("dense2d access");
            out.len() ^ out[out.len() / 2] as usize
        },
    );

    bench_existing_profile_round(
        config,
        "databank/profiled_dense2d_lz4_access_by_gene_names",
        128,
        Some(batch.len() * 8 * std::mem::size_of::<u32>()),
        bank.profile(),
        || {
            let genes = [
                "gene-1", "gene-3", "gene-5", "gene-7", "gene-9", "gene-11", "gene-13", "gene-15",
            ];
            let mut out = vec![0u32; batch.len() * genes.len()];
            bank.access_cells_by_gene_names(
                dense_lz4_id,
                &batch,
                &genes,
                &mut out,
                None,
                MissingGenePolicy::Error,
            )
            .expect("dense gene access");
            out.len() ^ out[out.len() / 2] as usize
        },
    );

    bench_existing_profile_round(
        config,
        "databank/profiled_dense1d_rectilinear_owned",
        256,
        Some(batch.len() * genes * std::mem::size_of::<u32>()),
        bank.profile(),
        || {
            let out = bank
                .access_cells_owned::<u32>(rect_id, &batch)
                .expect("dense1d owned");
            out.len() ^ out[out.len() / 2] as usize
        },
    );

    bench_existing_profile_round(
        config,
        "databank/profiled_sparse_csr_access",
        128,
        Some(batch.len() * genes * std::mem::size_of::<f32>()),
        bank.profile(),
        || {
            let mut out = vec![0f32; batch.len() * genes];
            bank.access_cells(sparse_id, &batch, &mut out, None)
                .expect("sparse access");
            out.len() ^ out[out.len() / 2].to_bits() as usize
        },
    );

    bench_existing_profile_round(
        config,
        "databank/profiled_prefetch_and_scheduled",
        64,
        Some(batch.len() * genes * std::mem::size_of::<u32>()),
        bank.profile(),
        || {
            bank.prefetch_cells(dense1d_id, &batch)
                .expect("direct prefetch");
            let batches = vec![batch.clone(), (32..64).collect::<Vec<_>>()];
            let prefetch = bank
                .prefetch_cells_scheduled::<u32, _>(
                    dense1d_id,
                    batches,
                    ScheduledPrefetchConfig::default(),
                )
                .expect("scheduled prefetch");
            prefetch.fold(0usize, |acc, result| {
                let batch = result.expect("prefetch batch");
                acc ^ batch.buffer.len() ^ batch.buffer[batch.buffer.len() / 2] as usize
            })
        },
    );
}

fn bench_data_profiles(config: BenchConfig) {
    for dist in [
        DataDist::Counting,
        DataDist::LowEntropy,
        DataDist::HighEntropy,
    ] {
        let profile = DataProfile {
            dtype: DType::F32,
            dist,
            missing_permille: 250,
            chunk_bytes: 64 * 1024,
            num_chunks: 8,
            seed: 17,
        };
        bench(
            config,
            &format!("data/generate_{}", profile.label()),
            128,
            Some(profile.total_bytes()),
            || {
                let raw = profile.generate();
                raw.len() ^ raw[raw.len() / 2] as usize
            },
        );
    }
}

fn access_read_checksum(
    access: &_scdata::access::AccessHandle,
    key: ChunkKey,
    codec: SharedCodec,
    expected_size: usize,
) -> usize {
    let (tx, rx) = oneshot::channel();
    access
        .send(AccessRequest::new(key, codec, Some(expected_size), tx))
        .expect("send access");
    let bytes = rx
        .blocking_recv()
        .expect("access reply")
        .expect("access read");
    black_box(bytes.len() ^ bytes[bytes.len() / 2] as usize)
}

fn threaded_io_config(workers: usize, max_in_flight: usize, shards: usize) -> IoConfig {
    IoConfig::Threaded(ThreadedConfig {
        base: BaseIoConfig {
            max_in_flight,
            queue_capacity: max_in_flight.saturating_mul(4).max(max_in_flight),
            priority_levels: 2,
            queue_shards: shards,
            assume_non_overlapping_reads: true,
        },
        num_workers: workers,
        cpus: None,
    })
}

fn databank_config() -> DataBankConfig {
    DataBankConfig {
        io_config: threaded_io_config(2, 128, 1),
        decode_config: DecodePoolConfig {
            num_workers: 2,
            queue_capacity: 128,
            cpus: None,
        },
        access_config: AccessConfig {
            queue_capacity: 128,
            scheduler_shards: 2,
            cache_capacity_bytes: 32 * 1024 * 1024,
            memory_budget_bytes: 64 * 1024 * 1024,
            default_io_priority: 0,
            keep_decoded: true,
            cpu: AccessCpuConfig {
                num_workers: 2,
                queue_capacity: 128,
                cpus: None,
            },
            profile: AccessProfile::from_env(),
        },
        fill_config: FillConfig {
            parallel: true,
            num_workers: 2,
            queue_capacity: 128,
            min_parallel_rows: 1,
            min_parallel_bytes: 1,
            cpus: None,
        },
        native_config: NativeAccessConfig::default(),
    }
}
