#![cfg(all(target_os = "linux", target_has_atomic = "64"))]

mod common;

use std::os::fd::AsFd;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use sc_compress::{DenseWriter, Partition};
use scdata::{
    compile, Dataset, Error, Fill, IoMode, OutputDType, OutputSpec, PlanSpec, RowRef, SharedClient,
    SharedConfig, Source, SourceId,
};

use common::blocking;

fn dense_plan() -> (tempfile::TempDir, scdata::Plan, Vec<Vec<u32>>) {
    dense_plan_with_rows(6)
}

fn dense_plan_with_rows(n_rows: usize) -> (tempfile::TempDir, scdata::Plan, Vec<Vec<u32>>) {
    dense_plan_with_prefetch(n_rows, 4)
}

fn dense_plan_with_prefetch(
    n_rows: usize,
    prefetch_step: usize,
) -> (tempfile::TempDir, scdata::Plan, Vec<Vec<u32>>) {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("dense-shared");
    let value_count = u16::try_from(n_rows * 3).unwrap();
    let values = (0..value_count).collect::<Vec<_>>();
    DenseWriter::new(&path)
        .chunk(Partition::fixed_cells(3))
        .block(Partition::fixed_cells(1))
        .write(&values, [u64::try_from(n_rows).unwrap(), 3])
        .unwrap();

    let dataset = Dataset::open(&path).unwrap();
    let source_id = SourceId::new(1);
    let source = Source::new(source_id, dataset);
    let requested = (0..u64::try_from(n_rows).unwrap()).collect::<Vec<_>>();
    let rows = requested
        .iter()
        .copied()
        .map(|row| RowRef::new(source_id, row))
        .collect();
    let output = OutputSpec::new(3, OutputDType::U32, Fill::U32(0)).unwrap();
    let plan = compile(PlanSpec::new(vec![source], rows, output, 2, prefetch_step)).unwrap();
    let expected = requested
        .into_iter()
        .map(|row| {
            let base = u32::try_from(row * 3).unwrap();
            vec![base, base + 1, base + 2]
        })
        .collect::<Vec<_>>();
    (temporary, plan, expected)
}

fn rows_from_batch(batch: &scdata::SharedBatch) -> Vec<Vec<u32>> {
    let bytes = batch.bytes().unwrap();
    let cols = batch.n_cols();
    let stride = batch.row_stride_bytes() / std::mem::size_of::<u32>();
    let values =
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u32>(), bytes.len() / 4) };
    (0..batch.rows())
        .map(|row| values[row * stride..row * stride + cols].to_vec())
        .collect()
}

#[test]
fn shared_ring_single_rank_matches_standard_path() {
    let (_temporary, plan, expected) = dense_plan();
    let (standard, _) = common::drain_rows::<u32>(&plan, 1);

    let server = plan
        .open_shared(blocking(1), SharedConfig::new(1).unwrap())
        .unwrap();
    let fd = server.attach_fd().unwrap();
    let (done_tx, done_rx) = mpsc::channel();
    let client = thread::spawn(move || {
        let mut client = SharedClient::attach(fd.as_fd(), 0).unwrap();
        let mut rows = Vec::new();
        while let Some(batch) = client.next_batch().unwrap() {
            rows.extend(rows_from_batch(&batch));
            batch.release().unwrap();
        }
        done_tx.send(rows).unwrap();
    });
    server.run().unwrap();
    let shared_rows = done_rx.recv().unwrap();
    client.join().unwrap();

    assert_eq!(standard, expected);
    assert_eq!(shared_rows, expected);
}

#[test]
fn shared_ring_round_robin_splits_batches_by_rank() {
    let (_temporary, plan, expected) = dense_plan();
    assert_eq!(plan.batch_count(), 3);

    let server = plan
        .open_shared(blocking(2), SharedConfig::new(2).unwrap())
        .unwrap();
    let fd0 = server.attach_fd().unwrap();
    let fd1 = server.attach_fd().unwrap();

    let (tx0, rx0) = mpsc::channel();
    let (tx1, rx1) = mpsc::channel();
    let c0 = thread::spawn(move || {
        let mut client = SharedClient::attach(fd0.as_fd(), 0).unwrap();
        let mut rows = Vec::new();
        let mut logicals = Vec::new();
        while let Some(batch) = client.next_batch().unwrap() {
            assert_eq!(batch.logical_batch() % 2, 0);
            logicals.push(batch.logical_batch());
            rows.extend(rows_from_batch(&batch));
            batch.release().unwrap();
        }
        tx0.send((rows, logicals)).unwrap();
    });
    let c1 = thread::spawn(move || {
        let mut client = SharedClient::attach(fd1.as_fd(), 1).unwrap();
        let mut rows = Vec::new();
        let mut logicals = Vec::new();
        while let Some(batch) = client.next_batch().unwrap() {
            assert_eq!(batch.logical_batch() % 2, 1);
            logicals.push(batch.logical_batch());
            rows.extend(rows_from_batch(&batch));
            batch.release().unwrap();
        }
        tx1.send((rows, logicals)).unwrap();
    });

    server.run().unwrap();
    let (rows0, logicals0) = rx0.recv().unwrap();
    let (rows1, logicals1) = rx1.recv().unwrap();
    c0.join().unwrap();
    c1.join().unwrap();

    assert_eq!(logicals0, vec![0, 2]);
    assert_eq!(logicals1, vec![1]);
    assert_eq!(
        rows0,
        vec![
            expected[0].clone(),
            expected[1].clone(),
            expected[4].clone(),
            expected[5].clone()
        ]
    );
    assert_eq!(rows1, vec![expected[2].clone(), expected[3].clone()]);
}

#[test]
fn shared_ring_cancel_wakes_waiting_client() {
    let (_temporary, plan, _) = dense_plan();
    let server = plan
        .open_shared(blocking(1), SharedConfig::new(1).unwrap())
        .unwrap();
    let fd = server.attach_fd().unwrap();
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let client = thread::spawn(move || {
        let mut client = SharedClient::attach(fd.as_fd(), 0).unwrap();
        started_tx.send(()).unwrap();
        // No producer will publish; cancel should unblock this wait.
        match client.next_batch() {
            Ok(Some(batch)) => {
                drop(batch);
                Ok(())
            }
            Ok(None) => Ok(()),
            Err(error) => Err(error),
        }
    });
    started_rx.recv().unwrap();
    // Give the client a moment to enter the futex wait.
    thread::sleep(std::time::Duration::from_millis(50));
    server.cancel();
    let result = client.join().unwrap();
    assert!(matches!(result, Err(Error::Cancelled)));
}

#[test]
fn shared_ring_pipelines_ranks_without_head_of_line_blocking() {
    let (_temporary, plan, expected) = dense_plan_with_rows(8);
    assert_eq!(plan.batch_count(), 4);
    let server = plan
        .open_shared(blocking(2), SharedConfig::new(2).unwrap())
        .unwrap();
    let fd0 = server.attach_fd().unwrap();
    let fd1 = server.attach_fd().unwrap();
    let server_thread = thread::spawn(move || server.run());

    let mut rank0 = SharedClient::attach(fd0.as_fd(), 0).unwrap();
    let mut rank1 = SharedClient::attach(fd1.as_fd(), 1).unwrap();
    let batch0 = rank0.next_batch().unwrap().unwrap();
    let batch2 = rank0.next_batch().unwrap().unwrap();
    let batch1 = rank1.next_batch().unwrap().unwrap();
    assert_eq!(batch0.logical_batch(), 0);
    assert_eq!(batch2.logical_batch(), 2);
    assert_eq!(batch1.logical_batch(), 1);
    batch1.release().unwrap();

    let (done_tx, done_rx) = mpsc::channel();
    let rank1_thread = thread::spawn(move || {
        let batch3 = rank1.next_batch().unwrap().unwrap();
        let rows = rows_from_batch(&batch3);
        let logical = batch3.logical_batch();
        batch3.release().unwrap();
        assert!(rank1.next_batch().unwrap().is_none());
        done_tx.send((logical, rows)).unwrap();
    });
    let (logical, rows) = match done_rx.recv_timeout(Duration::from_secs(2)) {
        Ok(value) => value,
        Err(error) => {
            drop(batch0);
            drop(batch2);
            rank1_thread.join().unwrap();
            panic!("rank 1 was blocked behind rank 0: {error}");
        }
    };
    assert_eq!(logical, 3);
    assert_eq!(rows, vec![expected[6].clone(), expected[7].clone()]);

    // Out-of-order release is safe; global ring reuse remains in-order.
    batch2.release().unwrap();
    batch0.release().unwrap();
    assert!(rank0.next_batch().unwrap().is_none());
    drop(rank0);
    rank1_thread.join().unwrap();
    server_thread.join().unwrap().unwrap();

    // A fresh client resumes after the released generations instead of replaying them.
    let mut resumed = SharedClient::attach(fd0.as_fd(), 0).unwrap();
    assert!(resumed.next_batch().unwrap().is_none());
}

fn assert_ranked_generations(n_rows: usize, prefetch_step: usize, world_size: usize) {
    assert_ranked_generations_with_mode(n_rows, prefetch_step, world_size, IoMode::Blocking);
}

fn assert_ranked_generations_with_mode(
    n_rows: usize,
    prefetch_step: usize,
    world_size: usize,
    io_mode: IoMode,
) {
    let (_temporary, plan, _) = dense_plan_with_prefetch(n_rows, prefetch_step);
    let batch_count = plan.batch_count();
    let server = plan
        .open_shared(
            common::session_config(world_size, io_mode),
            SharedConfig::new(world_size).unwrap(),
        )
        .unwrap();
    let descriptors = (0..world_size)
        .map(|_| server.attach_fd().unwrap())
        .collect::<Vec<_>>();
    let server_thread = thread::spawn(move || server.run());

    let consumers = descriptors
        .into_iter()
        .enumerate()
        .map(|(rank, descriptor)| {
            thread::spawn(move || {
                let mut client = SharedClient::attach(descriptor.as_fd(), rank).unwrap();
                let mut logicals = Vec::new();
                let mut rows = 0usize;
                while let Some(batch) = client.next_batch().unwrap() {
                    assert_eq!(batch.logical_batch() % world_size, rank);
                    logicals.push(batch.logical_batch());
                    rows += batch.rows();
                    batch.release().unwrap();
                }
                (logicals, rows)
            })
        })
        .collect::<Vec<_>>();

    let mut logicals = Vec::new();
    let mut rows = 0usize;
    for consumer in consumers {
        let (rank_logicals, rank_rows) = consumer.join().unwrap();
        logicals.extend(rank_logicals);
        rows += rank_rows;
    }
    server_thread.join().unwrap().unwrap();
    logicals.sort_unstable();
    assert_eq!(logicals, (0..batch_count).collect::<Vec<_>>());
    assert_eq!(rows, n_rows);
}

#[test]
fn shared_ring_reuses_slots_across_many_ranked_generations() {
    assert_ranked_generations(512, 8, 4);
}

#[test]
fn standard_single_consumer_reuses_slots_across_many_generations() {
    let (_temporary, plan, expected) = dense_plan_with_prefetch(512, 8);
    let (rows, _) = common::drain_rows::<u32>(&plan, 1);
    assert_eq!(rows, expected);
}

#[test]
fn shared_ring_supports_non_power_of_two_rank_and_ring_counts() {
    assert_ranked_generations(30, 3, 3);
}

#[cfg(feature = "uring")]
#[test]
fn shared_ring_runs_positioned_io_through_io_uring() {
    assert_ranked_generations_with_mode(128, 8, 4, IoMode::Uring { queue_depth: 2 });
}

#[test]
fn shared_ring_rejects_duplicate_rank_owners() {
    let (_temporary, plan, _) = dense_plan();
    let server = plan
        .open_shared(blocking(1), SharedConfig::new(1).unwrap())
        .unwrap();
    let fd = server.attach_fd().unwrap();
    let first = SharedClient::attach(fd.as_fd(), 0).unwrap();
    assert!(matches!(
        SharedClient::attach(fd.as_fd(), 0),
        Err(Error::InvalidInput(_))
    ));
    drop(first);
    server.cancel();
    assert!(matches!(
        SharedClient::attach(fd.as_fd(), 0),
        Err(Error::Cancelled)
    ));
}

#[test]
fn shared_ring_reports_a_self_lease_stall_instead_of_deadlocking() {
    let (_temporary, plan, _) = dense_plan_with_prefetch(6, 2);
    let server = plan
        .open_shared(blocking(1), SharedConfig::new(1).unwrap())
        .unwrap();
    let fd = server.attach_fd().unwrap();
    let server_thread = thread::spawn(move || server.run());
    let mut client = SharedClient::attach(fd.as_fd(), 0).unwrap();

    let first = client.next_batch().unwrap().unwrap();
    let second = client.next_batch().unwrap().unwrap();
    assert!(matches!(client.next_batch(), Err(Error::InvalidInput(_))));
    first.release().unwrap();
    second.release().unwrap();
    let third = client.next_batch().unwrap().unwrap();
    third.release().unwrap();
    assert!(client.next_batch().unwrap().is_none());
    server_thread.join().unwrap().unwrap();
}

#[test]
fn dropping_an_incomplete_client_cancels_the_producer() {
    let (_temporary, plan, _) = dense_plan();
    let server = plan
        .open_shared(blocking(1), SharedConfig::new(1).unwrap())
        .unwrap();
    let fd = server.attach_fd().unwrap();
    let client = SharedClient::attach(fd.as_fd(), 0).unwrap();
    let (done_tx, done_rx) = mpsc::channel();
    let server_thread = thread::spawn(move || {
        done_tx.send(server.run()).unwrap();
    });

    drop(client);
    let result = done_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("incomplete client drop did not cancel the producer");
    assert!(matches!(result, Err(Error::Cancelled)));
    server_thread.join().unwrap();
}

#[test]
fn shared_cancellation_handle_wakes_producer_ack_wait() {
    let (_temporary, plan, _) = dense_plan();
    let server = plan
        .open_shared(blocking(1), SharedConfig::new(1).unwrap())
        .unwrap();
    let cancellation = server.cancellation_handle();
    let (done_tx, done_rx) = mpsc::channel();
    let server_thread = thread::spawn(move || {
        done_tx.send(server.run()).unwrap();
    });
    thread::sleep(Duration::from_millis(50));
    cancellation.cancel();
    let result = done_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("cancelled shared producer did not wake");
    assert!(matches!(result, Err(Error::Cancelled)));
    server_thread.join().unwrap();
}

#[test]
fn shared_control_region_obeys_its_resource_limit() {
    let (_temporary, plan, _) = dense_plan();
    let config = SharedConfig::new(1)
        .unwrap()
        .with_max_control_bytes(1)
        .unwrap();
    assert!(matches!(
        plan.open_shared(blocking(1), config),
        Err(Error::ResourceLimit(_))
    ));
}
