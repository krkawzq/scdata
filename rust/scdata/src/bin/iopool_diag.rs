//! iopool 诊断基准: 直接用 IoPool API, 绕过 native load, 隔离测 IO 后端特性。
//!
//! 回答的问题:
//! 1. GPFS 上 io_uring 多在途是否真的有效 (单线程 depth 扫描)
//! 2. threaded vs uring 单线程同 depth 对比
//! 3. 多线程 N × depth K 总吞吐
//! 4. 多独立 pool (模拟 per-worker ring) 是否比单 pool 多 driver 更好

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use _scdata::iopool::{
    BaseIoConfig, IoCommand, IoConfig, IoFuture, IoPool, ThreadedConfig, UringConfig,
};
use _scdata::profile::ProfileRuntime;

fn main() {
    match parse_args().and_then(run) {
        Ok(()) => {}
        Err(err) => {
            eprintln!("iopool_diag: {err}");
            std::process::exit(2);
        }
    }
}

#[derive(Debug, Clone)]
struct Args {
    file: PathBuf,
    read_size: usize,
    total_ops: usize,
    warmup_ops: usize,
    // depth 扫描 (单线程)
    depths: Vec<usize>,
    // 线程扫描 (固定 depth)
    thread_counts: Vec<usize>,
    // 多 pool 测试的 pool 数
    pool_counts: Vec<usize>,
    // uring entries
    uring_entries: u32,
    // 要测的 backend 列表
    backends: Vec<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let mut a = Args {
        file: PathBuf::from(
            "/mnt/shared-storage-user/dnacoding/wangzhongqi/tmp-iobench/scdata-real-io-bench-chunk.bin",
        ),
        read_size: 196_608, // 192 KiB (一个 blosc block)
        total_ops: 8192,
        warmup_ops: 512,
        depths: vec![1, 2, 4, 8, 16, 32, 64],
        thread_counts: vec![1, 4, 8, 16, 32, 48],
        pool_counts: vec![1, 4, 8, 16, 32],
        uring_entries: 1024,
        backends: vec!["threaded".to_string(), "uring".to_string()],
    };
    while let Some(arg) = args.next() {
        let (key, value) = if let Some((k, v)) = arg.split_once('=') {
            (k.to_string(), v.to_string())
        } else {
            let v = args
                .next()
                .ok_or_else(|| format!("missing value for {arg}"))?;
            (arg, v)
        };
        match key.as_str() {
            "--file" => a.file = value.into(),
            "--read-size" => a.read_size = parse_usize(&value, &key)?,
            "--total-ops" => a.total_ops = parse_usize(&value, &key)?,
            "--warmup-ops" => a.warmup_ops = parse_usize(&value, &key)?,
            "--depths" => a.depths = parse_list(&value)?,
            "--threads" => a.thread_counts = parse_list(&value)?,
            "--pools" => a.pool_counts = parse_list(&value)?,
            "--uring-entries" => a.uring_entries = parse_u32(&value, &key)?,
            "--backends" => a.backends = value.split(',').map(str::to_string).collect(),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(a)
}

fn parse_usize(v: &str, k: &str) -> Result<usize, String> {
    v.parse().map_err(|e| format!("{k} must be usize: {e}"))
}
fn parse_u32(v: &str, k: &str) -> Result<u32, String> {
    v.parse().map_err(|e| format!("{k} must be u32: {e}"))
}
fn parse_list(v: &str) -> Result<Vec<usize>, String> {
    v.split(',')
        .map(|s| {
            s.trim()
                .parse()
                .map_err(|e| format!("bad list element: {e}"))
        })
        .collect()
}

fn run(args: Args) -> Result<(), String> {
    let data_len = std::fs::metadata(&args.file)
        .map_err(|e| format!("stat {}: {e}", args.file.display()))?
        .len() as usize;
    if data_len < args.read_size {
        return Err(format!(
            "file {} ({data_len} bytes) smaller than read_size {}",
            args.file.display(),
            args.read_size
        ));
    }
    eprintln!(
        "iopool_diag file={} ({} bytes) read_size={} total_ops={}",
        args.file.display(),
        data_len,
        args.read_size,
        args.total_ops
    );
    eprintln!(
        "  backends={:?} depths={:?} threads={:?} pools={:?}",
        args.backends, args.depths, args.thread_counts, args.pool_counts
    );

    println!(
        "test\tbackend\tworkers\tdepth\tpools\telapsed_s\tops_per_s\tread_gib_per_s\tlatency_us_mean"
    );

    for backend in &args.backends {
        // 1. 单线程 depth 扫描: 验证多在途是否有效
        for &depth in &args.depths {
            run_single_thread_depth(&args, backend, depth, data_len)?;
        }

        // 2. 多线程扫描 (每线程固定 depth=8)
        for &threads in &args.thread_counts {
            run_multi_thread(&args, backend, threads, 8, data_len)?;
        }
    }

    // 3. 多独立 pool: 模拟 per-worker ring
    //    N 个独立 IoPool(每 pool 单线程), 每个 pool depth=8
    //    对 threaded 和 uring 都测, 分离 "锁分散" vs "io_uring 本身" 两个因素
    for backend in &args.backends {
        for &pools in &args.pool_counts {
            run_multi_pool(&args, backend, pools, 8, data_len)?;
        }
    }

    Ok(())
}

fn make_config(args: &Args, backend: &str, workers: usize) -> Result<IoConfig, String> {
    let base = BaseIoConfig {
        max_in_flight: 4096,
        queue_capacity: 16384,
        priority_levels: 3,
        queue_shards: workers.max(1).min(16),
        assume_non_overlapping_reads: true,
    };
    base.validate().map_err(|e| e.to_string())?;
    match backend {
        "threaded" => Ok(IoConfig::Threaded(ThreadedConfig {
            base,
            num_workers: workers,
            cpus: None,
        })),
        "uring" => {
            if args.uring_entries < 2 {
                return Err("uring_entries must be >= 2".into());
            }
            if workers == 0 {
                return Err("uring drivers must be > 0".into());
            }
            Ok(IoConfig::Uring(UringConfig {
                base,
                entries: args.uring_entries,
                drivers: workers,
                iowq_bounded_workers: 0,
                iowq_unbounded_workers: 0,
                registered_files: 0,
            }))
        }
        other => Err(format!("unknown backend {other}")),
    }
}

/// 单线程: submit `depth` 个 read, 批量 try_recv 轮询直到全部完成, 重复 total_ops/depth 轮。
/// 这模拟 "单线程驱动 depth 个在途 IO" —— 即 per-worker ring 的核心行为。
fn run_single_thread_depth(
    args: &Args,
    backend: &str,
    depth: usize,
    data_len: usize,
) -> Result<(), String> {
    let config = make_config(args, backend, 1)?;
    let pool = Arc::new(
        IoPool::new_with_profile(config, ProfileRuntime::disabled())
            .map_err(|e| format!("create pool: {e}"))?,
    );
    let file_id = pool
        .register_readonly_file(&args.file)
        .map_err(|e| format!("register: {e}"))?;
    let max_offset = (data_len - args.read_size) as u64;

    let ops_per_round = depth;
    let rounds = (args.total_ops / ops_per_round.max(1)).max(1);
    let warmup_rounds = (args.warmup_ops / ops_per_round.max(1)).max(1);

    let counter = AtomicU64::new(0);
    let next_offset = || {
        let idx = counter.fetch_add(1, Ordering::Relaxed);
        (idx as u64 * args.read_size as u64) % (max_offset + 1)
    };

    // warmup
    for _ in 0..warmup_rounds {
        run_one_round(&pool, file_id, depth, args.read_size, &next_offset);
    }
    let start = Instant::now();
    let mut total_done = 0u64;
    for _ in 0..rounds {
        total_done += run_one_round(&pool, file_id, depth, args.read_size, &next_offset) as u64;
    }
    let elapsed = start.elapsed();

    let _ = pool.try_unregister_file(file_id);
    drop(pool);

    print_result(
        "single_depth",
        backend,
        1,
        depth,
        1,
        elapsed,
        total_done,
        args.read_size,
    );
    Ok(())
}

fn run_one_round(
    pool: &IoPool,
    file_id: _scdata::iopool::FileId,
    depth: usize,
    read_size: usize,
    next_offset: &impl Fn() -> u64,
) -> usize {
    let mut pending: Vec<IoFuture> = Vec::with_capacity(depth);
    for _ in 0..depth {
        let offset = next_offset();
        match pool.try_submit(IoCommand::read(file_id, offset, read_size, 0)) {
            Ok(fut) => pending.push(fut),
            Err(_) => break, // queue full, will retry next round
        }
    }
    if pending.is_empty() {
        return 0;
    }
    // 批量轮询: spin try_recv 直到全部完成
    let mut done = 0usize;
    let mut next_idx = 0usize;
    while done < pending.len() {
        if next_idx >= pending.len() {
            next_idx = 0;
        }
        match pending[next_idx].try_recv_read() {
            Ok(Some(_)) => {
                done += 1;
                // 移除已完成的 (swap_remove 保持 O(1))
                pending.swap_remove(next_idx);
                // 不递增 next_idx, 因为 swap_remove 把末尾换到了当前位置
            }
            Ok(None) => {
                next_idx += 1;
            }
            Err(e) => {
                eprintln!("read error: {e}");
                done += 1;
                pending.swap_remove(next_idx);
            }
        }
    }
    done
}

/// 多线程: N 个 worker, 每个独立单线程 depth 循环. 共用同一个 pool.
fn run_multi_thread(
    args: &Args,
    backend: &str,
    threads: usize,
    depth: usize,
    data_len: usize,
) -> Result<(), String> {
    let config = make_config(args, backend, threads)?;
    let pool = Arc::new(
        IoPool::new_with_profile(config, ProfileRuntime::disabled())
            .map_err(|e| format!("create pool: {e}"))?,
    );
    let file_id = pool
        .register_readonly_file(&args.file)
        .map_err(|e| format!("register: {e}"))?;
    let max_offset = (data_len - args.read_size) as u64;

    let ops_per_thread = (args.total_ops / threads.max(1)).max(1);
    let warmup_per_thread = (args.warmup_ops / threads.max(1)).max(1);

    let counter = Arc::new(AtomicU64::new(0));

    // warmup
    let warmup_pool = Arc::clone(&pool);
    let warmup_counter = Arc::clone(&counter);
    std::thread::scope(|s| {
        for _ in 0..threads {
            let pool = Arc::clone(&warmup_pool);
            let counter = Arc::clone(&warmup_counter);
            s.spawn(move || {
                let next = || {
                    let idx = counter.fetch_add(1, Ordering::Relaxed);
                    (idx as u64 * args.read_size as u64) % (max_offset + 1)
                };
                let rounds = (warmup_per_thread / depth.max(1)).max(1);
                for _ in 0..rounds {
                    run_one_round(&pool, file_id, depth, args.read_size, &next);
                }
            });
        }
    });

    counter.store(0, Ordering::Relaxed);
    let start = Instant::now();
    let total_done = Arc::new(AtomicU64::new(0));
    std::thread::scope(|s| {
        for _ in 0..threads {
            let pool = Arc::clone(&pool);
            let counter = Arc::clone(&counter);
            let total_done = Arc::clone(&total_done);
            s.spawn(move || {
                let next = || {
                    let idx = counter.fetch_add(1, Ordering::Relaxed);
                    (idx as u64 * args.read_size as u64) % (max_offset + 1)
                };
                let rounds = (ops_per_thread / depth.max(1)).max(1);
                let mut local = 0u64;
                for _ in 0..rounds {
                    local += run_one_round(&pool, file_id, depth, args.read_size, &next) as u64;
                }
                total_done.fetch_add(local, Ordering::Relaxed);
            });
        }
    });
    let elapsed = start.elapsed();
    let total_done = total_done.load(Ordering::Relaxed);

    let _ = pool.try_unregister_file(file_id);
    drop(pool);

    print_result(
        "multi_thread",
        backend,
        threads,
        depth,
        1,
        elapsed,
        total_done,
        args.read_size,
    );
    Ok(())
}

/// 多独立 pool (仅 uring, 每个 pool drivers=1): 模拟 per-worker ring.
/// N 个 pool, 每个 pool 一个线程 depth=8. 总线程数 = N.
fn run_multi_pool(
    args: &Args,
    backend: &str,
    pools: usize,
    depth: usize,
    data_len: usize,
) -> Result<(), String> {
    let max_offset = (data_len - args.read_size) as u64;
    let ops_per_pool = (args.total_ops / pools.max(1)).max(1);
    let warmup_per_pool = (args.warmup_ops / pools.max(1)).max(1);

    // 预建 N 个独立 pool (每 pool 单线程: uring drivers=1 / threaded num_workers=1)
    let mut pool_list: Vec<Arc<IoPool>> = Vec::with_capacity(pools);
    let mut file_ids: Vec<_scdata::iopool::FileId> = Vec::with_capacity(pools);
    for _ in 0..pools {
        let base = BaseIoConfig {
            max_in_flight: 4096,
            queue_capacity: 16384,
            priority_levels: 3,
            queue_shards: 1,
            assume_non_overlapping_reads: true,
        };
        let cfg = match backend {
            "threaded" => IoConfig::Threaded(ThreadedConfig {
                base,
                num_workers: 1,
                cpus: None,
            }),
            "uring" => IoConfig::Uring(UringConfig {
                base,
                entries: args.uring_entries,
                drivers: 1,
                iowq_bounded_workers: 0,
                iowq_unbounded_workers: 0,
                registered_files: 0,
            }),
            other => return Err(format!("unknown backend {other}")),
        };
        let pool = Arc::new(
            IoPool::new_with_profile(cfg, ProfileRuntime::disabled())
                .map_err(|e| format!("create pool #{pools}: {e}"))?,
        );
        let fid = pool
            .register_readonly_file(&args.file)
            .map_err(|e| format!("register: {e}"))?;
        pool_list.push(pool);
        file_ids.push(fid);
    }

    let counter = Arc::new(AtomicU64::new(0));

    // warmup
    {
        let warmup_counter = Arc::clone(&counter);
        std::thread::scope(|s| {
            for (i, pool) in pool_list.iter().enumerate() {
                let pool = Arc::clone(pool);
                let counter = Arc::clone(&warmup_counter);
                let file_id = file_ids[i];
                s.spawn(move || {
                    let next = || {
                        let idx = counter.fetch_add(1, Ordering::Relaxed);
                        (idx as u64 * args.read_size as u64) % (max_offset + 1)
                    };
                    let rounds = (warmup_per_pool / depth.max(1)).max(1);
                    for _ in 0..rounds {
                        run_one_round(&pool, file_id, depth, args.read_size, &next);
                    }
                });
            }
        });
    }

    counter.store(0, Ordering::Relaxed);
    let start = Instant::now();
    let total_done = Arc::new(AtomicU64::new(0));
    std::thread::scope(|s| {
        for (i, pool) in pool_list.iter().enumerate() {
            let pool = Arc::clone(pool);
            let counter = Arc::clone(&counter);
            let total_done = Arc::clone(&total_done);
            let file_id = file_ids[i];
            s.spawn(move || {
                let next = || {
                    let idx = counter.fetch_add(1, Ordering::Relaxed);
                    (idx as u64 * args.read_size as u64) % (max_offset + 1)
                };
                let rounds = (ops_per_pool / depth.max(1)).max(1);
                let mut local = 0u64;
                for _ in 0..rounds {
                    local += run_one_round(&pool, file_id, depth, args.read_size, &next) as u64;
                }
                total_done.fetch_add(local, Ordering::Relaxed);
            });
        }
    });
    let elapsed = start.elapsed();
    let total_done = total_done.load(Ordering::Relaxed);

    for (i, pool) in pool_list.iter().enumerate() {
        let _ = pool.try_unregister_file(file_ids[i]);
    }
    drop(pool_list);

    print_result(
        "multi_pool",
        backend,
        pools,
        depth,
        pools,
        elapsed,
        total_done,
        args.read_size,
    );
    Ok(())
}

fn print_result(
    test: &str,
    backend: &str,
    workers: usize,
    depth: usize,
    pools: usize,
    elapsed: std::time::Duration,
    total_ops: u64,
    read_size: usize,
) {
    let elapsed_s = elapsed.as_secs_f64();
    let ops_per_s = total_ops as f64 / elapsed_s;
    let read_gib_per_s = total_ops as f64 * read_size as f64 / elapsed_s / 1024.0 / 1024.0 / 1024.0;
    let latency_us_mean = if total_ops > 0 {
        elapsed_s / total_ops as f64 * 1e6
    } else {
        0.0
    };
    println!(
        "{test}\t{backend}\t{workers}\t{depth}\t{pools}\t{elapsed_s:.3}\t{ops_per_s:.0}\t{read_gib_per_s:.3}\t{latency_us_mean:.1}"
    );
}

fn print_help() {
    println!(
        r#"iopool_diag

Direct IoPool IO backend diagnostic. Bypasses native load; isolates backend
throughput characteristics on a single file.

Options:
  --file PATH         file to read (default: bench chunk file)
  --read-size N       bytes per read (default: 196608 = 192 KiB)
  --total-ops N       total read ops per test (default: 8192)
  --warmup-ops N      warmup ops (default: 512)
  --depths LIST       comma list of depths for single-thread scan (default: 1,2,4,8,16,32,64)
  --threads LIST      comma list of worker counts for multi-thread scan (default: 1,4,8,16,32,48)
  --pools LIST        comma list of pool counts for multi-pool uring test (default: 1,4,8,16,32)
  --uring-entries N   SQ/CQ depth (default: 1024)
  --backends LIST     comma list: threaded,uring (default: threaded,uring)

Tests:
  single_depth    1 thread, submit `depth` reads then poll recv. Measures
                  whether io_uring multi-inflight helps on GPFS.
  multi_thread    N threads, each depth=8, shared pool. Current backend model.
  multi_pool_uring N independent uring pools (drivers=1), each 1 thread depth=8.
                  Simulates per-worker ring model.
"#
    );
}
