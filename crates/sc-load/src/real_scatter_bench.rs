use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use sc_compress::{open_csr, CsrOutput, DType, StoreLocation};

use crate::compiler::{build_default_ranges, build_dense_map, choose_dense_whole_fill};
use crate::convert::ConvertOp;
use crate::plan::{CellTask, CsrMap, SourcePlan, UNMAPPED_TARGET_U32};
use crate::scatter::{scatter_row_prevalidated, validate_row, FillOp, IndexOp};
use crate::source::OutputSlot;
use crate::{Fill, OutputDType, OutputSpec};

const TARGET_BYTES_PER_SAMPLE: usize = 128 * 1024 * 1024;
const ROUNDS: usize = 7;

#[test]
#[ignore = "manual decoded real-data scatter benchmark"]
fn benchmark_real_decoded_scatter() {
    let list = std::env::var_os("SC_LOAD_REAL_SCATTER_LIST")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("real_dataset.txt"));
    let rows = parse_env_usize("SC_LOAD_REAL_SCATTER_ROWS", 256);
    let skip_datasets = parse_env_usize("SC_LOAD_REAL_SCATTER_SKIP_DATASETS", 0);
    let maximum_datasets = parse_env_usize("SC_LOAD_REAL_SCATTER_MAX_DATASETS", usize::MAX);
    let datasets = read_dataset_list(&list);
    assert!(
        !datasets.is_empty(),
        "{} has no dataset paths",
        list.display()
    );

    for path in datasets
        .into_iter()
        .skip(skip_datasets)
        .take(maximum_datasets)
    {
        benchmark_dataset(&path, rows);
    }
}

fn benchmark_dataset(path: &Path, requested_rows: usize) {
    assert!(path.exists(), "dataset does not exist: {}", path.display());
    let csr = open_csr(StoreLocation::zip(path, "X"))
        .unwrap_or_else(|error| panic!("failed to open {} as CSR: {error}", path.display()));
    let n_rows = usize::try_from(csr.n_rows()).unwrap();
    let n_cols = usize::try_from(csr.n_cols()).unwrap();
    let rows = requested_rows.min(n_rows);
    assert!(rows > 0 && n_cols > 0);
    let value_dtype = csr.value_dtype();
    let index_dtype = csr.index_dtype();
    let value_size = value_dtype.size();
    let index_size = index_dtype.size();
    let (indices, values) = csr.decode_rows(0..rows as u64).unwrap();
    let dense = csr
        .load_rows(0, rows as u64, CsrOutput::Dense)
        .unwrap()
        .into_dense()
        .unwrap();
    assert_eq!(dense.shape(), [rows, n_cols]);
    assert_eq!(dense.dtype(), value_dtype);
    let source_indptr = csr.indptr();
    assert_eq!(source_indptr[0], 0);

    let mut csr_tasks = Vec::with_capacity(rows);
    let mut dense_tasks = Vec::with_capacity(rows);
    for row in 0..rows {
        let start = usize::try_from(source_indptr[row]).unwrap();
        let end = usize::try_from(source_indptr[row + 1]).unwrap();
        csr_tasks.push(
            CellTask::new(
                OutputSlot::new(0).unwrap(),
                start * value_size..end * value_size,
                Some(start * index_size..end * index_size),
            )
            .unwrap(),
        );
        dense_tasks.push(
            CellTask::new(
                OutputSlot::new(0).unwrap(),
                row * n_cols * value_size..(row + 1) * n_cols * value_size,
                None,
            )
            .unwrap(),
        );
    }

    eprintln!(
        "REAL_SCATTER_DATASET path={} rows={} cols={} nnz={} dtype={} index_dtype={}",
        path.display(),
        rows,
        n_cols,
        source_indptr[rows],
        value_dtype,
        index_dtype,
    );

    for denominator in [1usize, 2, 5, 10] {
        for missing_third in [false, true] {
            let mapping = build_mapping(n_cols, denominator, missing_third, value_size);
            let identity = denominator == 1 && !missing_third;
            let output = zero_output(mapping.output_cols, value_dtype);
            let fill = FillOp::new(&output.fill().encode()[..value_size]);
            let convert = ConvertOp::resolve(value_dtype, &output).unwrap();
            let dense_map = (!identity)
                .then(|| {
                    build_dense_map(
                        mapping.targets.clone(),
                        value_size,
                        value_size,
                        mapping.output_cols,
                        convert.dense_gather_min_entries(),
                    )
                })
                .transpose()
                .unwrap();
            let ranges =
                build_default_ranges(Some(&mapping.targets), mapping.output_cols, value_size)
                    .unwrap();
            let planner_whole =
                choose_dense_whole_fill(Some(&mapping.targets), value_size, ranges.len()).unwrap();
            let dense_direct = SourcePlan {
                n_cols,
                value_dtype,
                index: None,
                feature_map: None,
                dense_map,
                dense_fill_whole: false,
                default_ranges: Arc::clone(&ranges),
                convert,
            };
            let dense_whole = SourcePlan {
                dense_fill_whole: true,
                default_ranges: Default::default(),
                ..dense_direct.clone()
            };
            let csr_targets: Arc<[u32]> = Arc::from(mapping.csr_targets.clone());
            let csr_source = SourcePlan {
                n_cols,
                value_dtype,
                index: IndexOp::new(index_dtype),
                feature_map: (!identity).then(|| CsrMap::Packed32(Arc::clone(&csr_targets))),
                dense_map: None,
                dense_fill_whole: false,
                default_ranges: ranges,
                convert,
            };

            for task in &dense_tasks {
                validate_row(&dense_direct, task, dense.values(), &[]).unwrap();
            }
            for task in &csr_tasks {
                validate_row(&csr_source, task, &values, &indices).unwrap();
            }
            assert_same_outputs(
                &dense_direct,
                &dense_whole,
                &csr_source,
                &dense_tasks,
                &csr_tasks,
                dense.values(),
                &values,
                &indices,
                mapping.output_cols * value_size,
                fill,
            );
            assert_branchless_outputs(
                &csr_source,
                &csr_tasks,
                &values,
                &indices,
                &csr_targets,
                index_size,
                value_size,
                mapping.output_cols * value_size,
                fill,
            );

            let row_bytes = mapping.output_cols * value_size;
            let sweep_bytes = rows.saturating_mul(row_bytes).max(1);
            let iterations = TARGET_BYTES_PER_SAMPLE.div_ceil(sweep_bytes).clamp(1, 128);
            let mut output_row = vec![0xA5; row_bytes];
            let direct_samples = measure_samples(
                &dense_direct,
                &dense_tasks,
                dense.values(),
                &[],
                &mut output_row,
                row_bytes,
                fill,
                iterations,
            );
            let whole_samples = measure_samples(
                &dense_whole,
                &dense_tasks,
                dense.values(),
                &[],
                &mut output_row,
                row_bytes,
                fill,
                iterations,
            );
            let csr_samples = measure_samples(
                &csr_source,
                &csr_tasks,
                &values,
                &indices,
                &mut output_row,
                row_bytes,
                fill,
                iterations,
            );
            let branchless_samples = measure_branchless_samples(
                &csr_tasks,
                &values,
                &indices,
                &csr_targets,
                index_size,
                value_size,
                &mut output_row,
                iterations,
            );
            let adaptive_csr_samples = measure_adaptive_csr_samples(
                &csr_source,
                &csr_tasks,
                &values,
                &indices,
                &csr_targets,
                index_size,
                value_size,
                &mut output_row,
                row_bytes,
                fill,
                mapping.mapped * 3 >= n_cols && mapping.mapped * 3 <= n_cols * 2,
                iterations,
            );
            let direct_ns = median_ns_per_cell(direct_samples, rows, iterations);
            let whole_ns = median_ns_per_cell(whole_samples, rows, iterations);
            let csr_ns = median_ns_per_cell(csr_samples, rows, iterations);
            let branchless_ns = median_ns_per_cell(branchless_samples, rows, iterations);
            let adaptive_csr_ns = median_ns_per_cell(adaptive_csr_samples, rows, iterations);
            let mapped_hits = count_mapped_hits(&indices, index_size, &csr_targets);
            let hit_fraction = mapped_hits as f64 / (indices.len() / index_size).max(1) as f64;
            let auto_ns = if planner_whole { whole_ns } else { direct_ns };
            eprintln!(
                "REAL_SCATTER ratio=1/{denominator} missing={} mapped={} hit_fraction={hit_fraction:.4} out_cols={} gap_runs={} planner={} iterations={} dense_direct_ns_cell={direct_ns:.2} dense_whole_ns_cell={whole_ns:.2} whole_over_direct={:.4} dense_auto_ns_cell={auto_ns:.2} csr_ns_cell={csr_ns:.2} csr_branchless_ns_cell={branchless_ns:.2} csr_adaptive_ns_cell={adaptive_csr_ns:.2} csr_over_branchless={:.4} csr_over_adaptive={:.4}",
                if missing_third { "1/3" } else { "0" },
                mapping.mapped,
                mapping.output_cols,
                dense_direct.default_ranges.len(),
                if planner_whole { "whole" } else { "ranges" },
                iterations,
                whole_ns / direct_ns,
                csr_ns / branchless_ns,
                csr_ns / adaptive_csr_ns,
            );
        }
    }
}

struct Mapping {
    targets: Vec<Option<usize>>,
    csr_targets: Vec<u32>,
    mapped: usize,
    output_cols: usize,
}

fn build_mapping(
    n_cols: usize,
    denominator: usize,
    missing_third: bool,
    value_size: usize,
) -> Mapping {
    let mapped = n_cols.div_ceil(denominator).max(1);
    let output_cols = if missing_third {
        mapped.checked_mul(3).unwrap().div_ceil(2)
    } else {
        mapped
    };
    let mut targets = vec![None; n_cols];
    let mut csr_targets = vec![UNMAPPED_TARGET_U32; n_cols];
    for selected in 0..mapped {
        let source = selected * n_cols / mapped;
        let target = if missing_third {
            let pair = selected / 2;
            pair * 3 + selected % 2
        } else {
            selected
        };
        targets[source] = Some(target);
        csr_targets[source] = u32::try_from(target * value_size).unwrap();
    }
    Mapping {
        targets,
        csr_targets,
        mapped,
        output_cols,
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "benchmark correctness contract is explicit"
)]
fn assert_same_outputs(
    dense_direct: &SourcePlan,
    dense_whole: &SourcePlan,
    csr: &SourcePlan,
    dense_tasks: &[CellTask],
    csr_tasks: &[CellTask],
    dense_values: &[u8],
    csr_values: &[u8],
    indices: &[u8],
    row_bytes: usize,
    fill: FillOp,
) {
    let mut direct = vec![0xA5; row_bytes];
    let mut whole = vec![0x5A; row_bytes];
    let mut sparse = vec![0x3C; row_bytes];
    for (dense_task, csr_task) in dense_tasks.iter().zip(csr_tasks) {
        unsafe {
            // SAFETY: setup validated these exact buffers and task ranges.
            scatter_row_prevalidated(
                dense_direct,
                dense_task,
                dense_values,
                &[],
                &mut direct,
                row_bytes,
                fill,
            )
            .unwrap();
            // SAFETY: setup validated the same dense mapping and buffers.
            scatter_row_prevalidated(
                dense_whole,
                dense_task,
                dense_values,
                &[],
                &mut whole,
                row_bytes,
                fill,
            )
            .unwrap();
            // SAFETY: setup validated the canonical CSR buffers and mapping.
            scatter_row_prevalidated(
                csr,
                csr_task,
                csr_values,
                indices,
                &mut sparse,
                row_bytes,
                fill,
            )
            .unwrap();
        }
        assert_eq!(direct, whole);
        assert_eq!(direct, sparse);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "benchmark correctness contract is explicit"
)]
fn assert_branchless_outputs(
    source: &SourcePlan,
    tasks: &[CellTask],
    values: &[u8],
    indices: &[u8],
    targets: &[u32],
    index_size: usize,
    value_size: usize,
    row_bytes: usize,
    fill: FillOp,
) {
    let mut expected = vec![0xA5; row_bytes];
    let mut actual = vec![0x5A; row_bytes];
    for task in tasks {
        unsafe {
            // SAFETY: setup validated this canonical CSR row and output extent.
            scatter_row_prevalidated(
                source,
                task,
                values,
                indices,
                &mut expected,
                row_bytes,
                fill,
            )
            .unwrap();
            // SAFETY: the same validation covers the experimental copy kernel.
            scatter_csr_copy_branchless(
                task,
                values,
                indices,
                targets,
                index_size,
                value_size,
                &mut actual,
            );
        }
        assert_eq!(actual, expected);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "benchmark kernel contract is explicit"
)]
fn measure_samples(
    source: &SourcePlan,
    tasks: &[CellTask],
    values: &[u8],
    indices: &[u8],
    output: &mut [u8],
    row_bytes: usize,
    fill: FillOp,
    iterations: usize,
) -> Vec<Duration> {
    let run = |output: &mut [u8]| {
        for _ in 0..iterations {
            for task in tasks {
                unsafe {
                    // SAFETY: benchmark setup validates every task once before timing;
                    // decoded inputs are immutable and `output` is uniquely borrowed.
                    scatter_row_prevalidated(
                        source,
                        task,
                        values,
                        indices,
                        black_box(&mut *output),
                        row_bytes,
                        fill,
                    )
                    .unwrap();
                }
            }
        }
    };
    run(output);
    (0..ROUNDS)
        .map(|_| {
            let started = Instant::now();
            run(output);
            started.elapsed()
        })
        .collect()
}

#[expect(
    clippy::too_many_arguments,
    reason = "benchmark kernel contract is explicit"
)]
fn measure_branchless_samples(
    tasks: &[CellTask],
    values: &[u8],
    indices: &[u8],
    targets: &[u32],
    index_size: usize,
    value_size: usize,
    output: &mut [u8],
    iterations: usize,
) -> Vec<Duration> {
    let run = |output: &mut [u8]| {
        for _ in 0..iterations {
            for task in tasks {
                unsafe {
                    // SAFETY: setup validated every canonical row, map target,
                    // input extent, and output extent before timing.
                    scatter_csr_copy_branchless(
                        task,
                        values,
                        indices,
                        targets,
                        index_size,
                        value_size,
                        black_box(&mut *output),
                    );
                }
            }
        }
    };
    run(output);
    (0..ROUNDS)
        .map(|_| {
            let started = Instant::now();
            run(output);
            started.elapsed()
        })
        .collect()
}

#[expect(
    clippy::too_many_arguments,
    reason = "benchmark kernel contract is explicit"
)]
fn measure_adaptive_csr_samples(
    source: &SourcePlan,
    tasks: &[CellTask],
    values: &[u8],
    indices: &[u8],
    targets: &[u32],
    index_size: usize,
    value_size: usize,
    output: &mut [u8],
    row_bytes: usize,
    fill: FillOp,
    branchless_candidate: bool,
    iterations: usize,
) -> Vec<Duration> {
    let run = |output: &mut [u8]| {
        for _ in 0..iterations {
            for task in tasks {
                unsafe {
                    // SAFETY: setup validated every row and both candidate
                    // kernels implement the same exact-copy mapping contract.
                    if branchless_candidate
                        && should_use_branchless(task, indices, targets, index_size)
                    {
                        scatter_csr_copy_branchless(
                            task,
                            values,
                            indices,
                            targets,
                            index_size,
                            value_size,
                            black_box(&mut *output),
                        );
                    } else {
                        scatter_row_prevalidated(
                            source,
                            task,
                            values,
                            indices,
                            black_box(&mut *output),
                            row_bytes,
                            fill,
                        )
                        .unwrap();
                    }
                }
            }
        }
    };
    run(output);
    (0..ROUNDS)
        .map(|_| {
            let started = Instant::now();
            run(output);
            started.elapsed()
        })
        .collect()
}

fn should_use_branchless(
    task: &CellTask,
    indices: &[u8],
    targets: &[u32],
    index_size: usize,
) -> bool {
    const MIN_NNZ: usize = 128;
    const WINDOWS: usize = 4;
    const WINDOW_LEN: usize = 8;
    const SAMPLES: usize = WINDOWS * WINDOW_LEN;
    let row = &indices[task.indices_range()];
    let count = row.len() / index_size;
    if count < MIN_NNZ {
        return false;
    }
    let mut hits = 0usize;
    let mut transitions = 0usize;
    let mut periodic_matches = [0usize; 4];
    for window in 0..WINDOWS {
        let start = window * (count - WINDOW_LEN) / (WINDOWS - 1);
        let mut previous = None;
        let mut outcomes = [false; WINDOW_LEN];
        for (local, outcome) in outcomes.iter_mut().enumerate() {
            let column = unsafe {
                // SAFETY: every window lies inside `0..count`, and setup
                // validated complete canonical indices.
                read_index(row, index_size, start + local)
            };
            let mapped = targets[column] != UNMAPPED_TARGET_U32;
            *outcome = mapped;
            hits += usize::from(mapped);
            transitions += usize::from(previous.is_some_and(|value| value != mapped));
            previous = Some(mapped);
        }
        for (lag_index, matches) in periodic_matches.iter_mut().enumerate() {
            let lag = lag_index + 1;
            *matches += (lag..WINDOW_LEN)
                .filter(|&local| outcomes[local] == outcomes[local - lag])
                .count();
        }
    }
    let adjacent_pairs = WINDOWS * (WINDOW_LEN - 1);
    let locally_irregular = periodic_matches
        .iter()
        .enumerate()
        .all(|(lag_index, &matches)| {
            let pairs = WINDOWS * (WINDOW_LEN - lag_index - 1);
            matches * 4 < pairs * 3
        });
    hits * 5 >= SAMPLES
        && hits * 5 <= SAMPLES * 4
        && transitions * 4 >= adjacent_pairs
        && transitions * 4 <= adjacent_pairs * 3
        && locally_irregular
}

fn count_mapped_hits(indices: &[u8], index_size: usize, targets: &[u32]) -> usize {
    let count = indices.len() / index_size;
    (0..count)
        .filter(|&element| {
            let column = unsafe {
                // SAFETY: `element < count` and decoded indices are complete.
                read_index(indices, index_size, element)
            };
            targets[column] != UNMAPPED_TARGET_U32
        })
        .count()
}

unsafe fn scatter_csr_copy_branchless(
    task: &CellTask,
    values: &[u8],
    indices: &[u8],
    targets: &[u32],
    index_size: usize,
    value_size: usize,
    output: &mut [u8],
) {
    output.fill(0);
    // SAFETY: benchmark setup validated this task's data range.
    let row_values = unsafe { values.get_unchecked(task.data_range()) };
    // SAFETY: benchmark setup validated this task's index range.
    let row_indices = unsafe { indices.get_unchecked(task.indices_range()) };
    let count = row_values.len() / value_size;
    debug_assert_eq!(row_indices.len() / index_size, count);
    match value_size {
        2 => unsafe {
            // SAFETY: the wrapper validated `count` two-byte values.
            scatter_csr_copy_branchless_width::<2>(
                row_values,
                row_indices,
                targets,
                index_size,
                output,
                count,
            )
        },
        4 => unsafe {
            // SAFETY: the wrapper validated `count` four-byte values.
            scatter_csr_copy_branchless_width::<4>(
                row_values,
                row_indices,
                targets,
                index_size,
                output,
                count,
            )
        },
        8 => unsafe {
            // SAFETY: the wrapper validated `count` eight-byte values.
            scatter_csr_copy_branchless_width::<8>(
                row_values,
                row_indices,
                targets,
                index_size,
                output,
                count,
            )
        },
        _ => unreachable!("matrix value width is 2, 4, or 8"),
    }
}

unsafe fn scatter_csr_copy_branchless_width<const BYTES: usize>(
    row_values: &[u8],
    row_indices: &[u8],
    targets: &[u32],
    index_size: usize,
    output: &mut [u8],
    count: usize,
) {
    let mut sink = [0u8; 8];
    for element in 0..count {
        let column = unsafe {
            // SAFETY: `element < count` and setup validated complete indices.
            read_index(row_indices, index_size, element)
        };
        // SAFETY: decoded column bounds were validated against this map.
        let target = unsafe { *targets.get_unchecked(column) };
        let mapped = target != UNMAPPED_TARGET_U32;
        let mask = 0usize.wrapping_sub(mapped as usize);
        let safe_target = target as usize & mask;
        // SAFETY: mapped targets lie in output; the sentinel masks to zero.
        let mapped_destination = unsafe { output.as_mut_ptr().add(safe_target) };
        let destination =
            std::hint::select_unpredictable(mapped, mapped_destination, sink.as_mut_ptr());
        // SAFETY: setup validated this source element; the selected destination
        // is either one complete output element or the local sink.
        unsafe {
            std::ptr::copy_nonoverlapping(
                row_values.as_ptr().add(element * BYTES),
                destination,
                BYTES,
            );
        }
    }
}

unsafe fn read_index(indices: &[u8], index_size: usize, element: usize) -> usize {
    // SAFETY: caller proves `element` addresses one complete packed index.
    let index = unsafe { indices.as_ptr().add(element * index_size) };
    if index_size == 2 {
        // SAFETY: caller proves a complete possibly unaligned u16 index.
        usize::from(u16::from_le(unsafe {
            index.cast::<u16>().read_unaligned()
        }))
    } else {
        // SAFETY: caller proves a complete possibly unaligned u32 index.
        u32::from_le(unsafe { index.cast::<u32>().read_unaligned() }) as usize
    }
}

fn median_ns_per_cell(mut samples: Vec<Duration>, rows: usize, iterations: usize) -> f64 {
    samples.sort_unstable();
    samples[samples.len() / 2].as_secs_f64() * 1e9 / (rows * iterations) as f64
}

fn zero_output(n_cols: usize, dtype: DType) -> OutputSpec {
    match dtype {
        DType::I16 => OutputSpec::new(n_cols, OutputDType::I16, Fill::I16(0)),
        DType::I32 => OutputSpec::new(n_cols, OutputDType::I32, Fill::I32(0)),
        DType::I64 => OutputSpec::new(n_cols, OutputDType::I64, Fill::I64(0)),
        DType::U16 => OutputSpec::new(n_cols, OutputDType::U16, Fill::U16(0)),
        DType::U32 => OutputSpec::new(n_cols, OutputDType::U32, Fill::U32(0)),
        DType::U64 => OutputSpec::new(n_cols, OutputDType::U64, Fill::U64(0)),
        DType::F32 => OutputSpec::new(n_cols, OutputDType::F32, Fill::F32(0.0)),
        DType::F64 => OutputSpec::new(n_cols, OutputDType::F64, Fill::F64(0.0)),
    }
    .unwrap()
}

fn read_dataset_list(path: &Path) -> Vec<PathBuf> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(PathBuf::from)
        .collect()
}

fn parse_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|value| value.parse::<usize>().unwrap())
        .unwrap_or(default)
}
