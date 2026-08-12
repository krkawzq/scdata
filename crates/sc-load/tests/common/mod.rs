use sc_load::{IoMode, OutputValue, Plan, RuntimeStats, SessionConfig, SessionState};

pub fn session_config(worker_count: usize, io_mode: IoMode) -> SessionConfig {
    let mut config = SessionConfig::default();
    config.worker_count = worker_count;
    config.io_mode = io_mode;
    config.max_total_inflight_encoded_bytes = config
        .max_inflight_encoded_bytes_per_worker
        .saturating_mul(worker_count);
    config.max_total_decoded_bytes = config
        .max_decoded_bytes_per_worker
        .saturating_mul(worker_count);
    config.max_total_inflight_io_ops = match io_mode {
        IoMode::Blocking => config.max_total_inflight_io_ops.max(1),
        IoMode::Uring { queue_depth } | IoMode::Auto { queue_depth } => {
            worker_count.saturating_mul(queue_depth as usize).max(1)
        }
    };
    config
}

pub fn blocking(worker_count: usize) -> SessionConfig {
    session_config(worker_count, IoMode::Blocking)
}

pub fn drain_rows<T: OutputValue>(plan: &Plan, worker_count: usize) -> (Vec<Vec<T>>, RuntimeStats) {
    let mut session = plan.open(blocking(worker_count)).unwrap();
    let mut rows = Vec::new();
    let mut logical_batch = 0;

    while let Some(batch) = session.next_batch().unwrap() {
        assert_eq!(batch.logical_batch(), logical_batch);
        for row in 0..batch.rows() {
            rows.push(batch.row_as::<T>(row).unwrap().to_vec());
        }
        logical_batch += 1;
    }

    assert_eq!(session.state(), SessionState::Finished);
    let stats = session.stats();
    assert_eq!(stats.state, SessionState::Finished);
    (rows, stats)
}
