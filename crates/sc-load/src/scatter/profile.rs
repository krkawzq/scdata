//! Optional decoded-scatter profiling used by the `scatter_profile` example.

pub(crate) fn run_from_env() {
    let suite = std::env::var("SC_LOAD_SCATTER_PROFILE").unwrap_or_else(|_| "real".into());
    if suite == "real" || suite == "all" {
        real::run();
    }
    if suite != "real" {
        synthetic::run(&suite);
    }
}

mod synthetic {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use sc_compress::DType as StorageDType;

    use crate::compiler::{build_default_ranges, build_dense_map, choose_dense_whole_fill};
    use crate::convert::ConvertOp;
    use crate::dtype::{promote_kind, OutputDType, PromoteKind};
    use crate::output::{Fill, FloatCastPolicy, OutputSpec, OverflowPolicy};
    use crate::plan::{
        csr_sparse_binary_is_cheaper, CellTask, CsrMap, CsrSparseMap, DenseMap, SourcePlan,
        UNMAPPED_TARGET_U32,
    };
    use crate::scatter::{scatter_row_prevalidated, validate_row, FillOp, IndexOp};
    use crate::source::OutputSlot;

    const STORAGE_DTYPES: [StorageDType; 8] = [
        StorageDType::I16,
        StorageDType::I32,
        StorageDType::I64,
        StorageDType::U16,
        StorageDType::U32,
        StorageDType::U64,
        StorageDType::F32,
        StorageDType::F64,
    ];
    const OUTPUT_DTYPES: [OutputDType; 8] = [
        OutputDType::I16,
        OutputDType::I32,
        OutputDType::I64,
        OutputDType::U16,
        OutputDType::U32,
        OutputDType::U64,
        OutputDType::F32,
        OutputDType::F64,
    ];
    fn benchmark_all_conversion_fastpaths() {
        const N_COLS: usize = 16 * 1024;
        for src in STORAGE_DTYPES {
            for dst in OUTPUT_DTYPES {
                let Some(kind) = promote_kind(src, dst) else {
                    continue;
                };
                scan_conversion_mode(
                    src,
                    dst,
                    kind,
                    "error",
                    OverflowPolicy::Error,
                    false,
                    N_COLS,
                );
                if kind == PromoteKind::CheckedSign {
                    scan_conversion_mode(
                        src,
                        dst,
                        kind,
                        "fallback",
                        OverflowPolicy::UseValue(sentinel(dst)),
                        true,
                        N_COLS,
                    );
                    scan_conversion_mode(
                        src,
                        dst,
                        kind,
                        "unchecked",
                        OverflowPolicy::Unchecked,
                        true,
                        N_COLS,
                    );
                }
            }
        }
    }

    fn benchmark_gather32_thresholds() {
        const COUNTS: [usize; 11] = [1, 2, 3, 4, 7, 8, 15, 16, 31, 64, 256];
        for src in STORAGE_DTYPES {
            for dst in OUTPUT_DTYPES {
                let Some(kind) = promote_kind(src, dst) else {
                    continue;
                };
                let policy = if kind == PromoteKind::CheckedSign {
                    OverflowPolicy::UseValue(sentinel(dst))
                } else {
                    OverflowPolicy::Error
                };
                for count in COUNTS {
                    let n_cols = count * 2;
                    let output = output_spec(count, dst, policy.clone(), kind);
                    let fill = FillOp::new(&output.fill().encode()[..dst.size()]);
                    let input = values(src, n_cols, kind == PromoteKind::CheckedSign);
                    let convert = ConvertOp::resolve(src, &output).unwrap();
                    if convert.dense_gather_min_entries().is_none() {
                        continue;
                    }
                    let task = task(0..input.len(), None);
                    let gather = source(
                        n_cols,
                        src,
                        None,
                        Some(DenseMap::Gather32 {
                            source_offsets: Arc::from(
                                (0..n_cols)
                                    .step_by(2)
                                    .map(|column| i32::try_from(column * src.size()).unwrap())
                                    .collect::<Vec<_>>(),
                            ),
                            target_byte: 0,
                            covers_output: true,
                        }),
                        false,
                        Default::default(),
                        convert,
                    );
                    let packed = source(
                        n_cols,
                        src,
                        None,
                        Some(DenseMap::Packed32 {
                            entries: Arc::from(
                                (0..n_cols)
                                    .step_by(2)
                                    .map(|column| {
                                        u64::from((column * src.size()) as u32)
                                            | (u64::from((column / 2 * dst.size()) as u32) << 32)
                                    })
                                    .collect::<Vec<_>>(),
                            ),
                            covers_output: true,
                        }),
                        false,
                        Default::default(),
                        convert,
                    );
                    let (_, _, speedup) = paired_scatter(
                        &gather,
                        &packed,
                        &task,
                        &input,
                        &[],
                        count * dst.size(),
                        fill,
                        (2 * 1024 * 1024usize)
                            .div_ceil(input.len().saturating_add(count * dst.size()).max(1))
                            .clamp(64, 16_384),
                    );
                    eprintln!(
                        "GATHER_THRESHOLD src={} dst={} count={} planner_min={} speedup={:.4}",
                        src,
                        dst,
                        count,
                        convert.dense_gather_min_entries().unwrap(),
                        speedup,
                    );
                }
            }
        }
    }

    fn benchmark_csr_identity_initialization_thresholds() {
        const N_COLS: usize = 32 * 1024;
        const DENSITIES: [(usize, usize); 9] = [
            (1, 100),
            (1, 20),
            (1, 10),
            (1, 4),
            (1, 2),
            (3, 4),
            (9, 10),
            (99, 100),
            (1, 1),
        ];
        for src in STORAGE_DTYPES {
            for dst in OUTPUT_DTYPES {
                let Some(kind) = promote_kind(src, dst) else {
                    continue;
                };
                let output = output_spec(N_COLS, dst, OverflowPolicy::Error, kind);
                let fill = FillOp::new(&output.fill().encode()[..dst.size()]);
                let convert = ConvertOp::resolve(src, &output).unwrap();
                for (numerator, denominator) in DENSITIES {
                    let nnz = N_COLS * numerator / denominator;
                    for pattern in [
                        CsrPattern::Uniform,
                        CsrPattern::Clustered,
                        CsrPattern::Blocks,
                    ] {
                        let columns = csr_columns(N_COLS, nnz, pattern);
                        let indices = encode_indices(&columns, StorageDType::U16);
                        let data = values(src, nnz, false);
                        let task = task(0..data.len(), Some(0..indices.len()));
                        let source = source(
                            N_COLS,
                            src,
                            IndexOp::new(StorageDType::U16),
                            None,
                            false,
                            Default::default(),
                            convert,
                        );
                        validate_row(&source, &task, &data, &indices).unwrap();
                        let iterations = (4 * 1024 * 1024usize)
                            .div_ceil(data.len().saturating_add(N_COLS * dst.size()).max(1))
                            .clamp(4, 4096);
                        let (current_ns, gaps_ns, speedup) = paired_csr_identity_init(
                            &source,
                            &task,
                            &data,
                            &indices,
                            N_COLS * dst.size(),
                            fill,
                            iterations,
                        );
                        eprintln!(
                            "CSR_INIT_SCAN src={} dst={} density={}/{} pattern={} nnz={} current_ns={:.2} gaps_ns={:.2} speedup={:.4}",
                            src,
                            dst,
                            numerator,
                            denominator,
                            pattern.name(),
                            nnz,
                            current_ns,
                            gaps_ns,
                            speedup,
                        );
                    }
                }
            }
        }
    }

    fn benchmark_dense_initialization_thresholds() {
        const N_SOURCE: usize = 32 * 1024;
        const MAPPING: [usize; 4] = [1, 2, 5, 10];
        const DEFAULTS: [(&str, usize, usize); 5] = [
            ("0", 0, 1),
            ("1_10", 1, 10),
            ("1_3", 1, 3),
            ("1_2", 1, 2),
            ("3_4", 3, 4),
        ];
        for src in STORAGE_DTYPES {
            for dst in OUTPUT_DTYPES {
                let Some(kind) = promote_kind(src, dst) else {
                    continue;
                };
                for fill_class in [FillClass::Zero, FillClass::NonZero] {
                    for mapping_denominator in MAPPING {
                        let mapped = N_SOURCE.div_ceil(mapping_denominator);
                        for (default_name, default_numerator, default_denominator) in DEFAULTS {
                            let output_cols = if default_numerator == 0 {
                                mapped
                            } else {
                                mapped
                                    .checked_mul(default_denominator)
                                    .unwrap()
                                    .div_ceil(default_denominator - default_numerator)
                            };
                            let default_cols = output_cols - mapped;
                            for layout in [
                                DefaultLayout::Tail,
                                DefaultLayout::Runs64,
                                DefaultLayout::Interleaved,
                            ] {
                                let positions = mapped_positions(mapped, default_cols, layout);
                                let mut targets = vec![None; N_SOURCE];
                                for (selected, target) in positions.into_iter().enumerate() {
                                    let source = selected * N_SOURCE / mapped;
                                    targets[source] = Some(target);
                                }
                                let output =
                                    OutputSpec::new(output_cols, dst, fill_value(dst, fill_class))
                                        .unwrap()
                                        .float_cast(if kind == PromoteKind::RoundingToFloat {
                                            FloatCastPolicy::AllowRounding
                                        } else {
                                            FloatCastPolicy::ExactOnly
                                        });
                                let convert = ConvertOp::resolve(src, &output).unwrap();
                                let dense_map = build_dense_map(
                                    targets.clone(),
                                    src.size(),
                                    dst.size(),
                                    output_cols,
                                    convert.dense_gather_min_entries(),
                                )
                                .unwrap();
                                let map_kind = dense_map_name(&dense_map);
                                let ranges =
                                    build_default_ranges(Some(&targets), output_cols, dst.size())
                                        .unwrap();
                                let fill_bytes = output.fill().encode();
                                let planner_whole = choose_dense_whole_fill(
                                    Some(mapped),
                                    dst.size(),
                                    ranges.len(),
                                    Some(&dense_map),
                                    &fill_bytes[..dst.size()],
                                )
                                .unwrap();
                                let direct = source(
                                    N_SOURCE,
                                    src,
                                    None,
                                    Some(dense_map),
                                    false,
                                    Arc::clone(&ranges),
                                    convert,
                                );
                                let whole = SourcePlan {
                                    dense_fill_whole: true,
                                    default_ranges: Default::default(),
                                    ..direct.clone()
                                };
                                let data = values(src, N_SOURCE, false);
                                let task = task(0..data.len(), None);
                                let fill = FillOp::new(&output.fill().encode()[..dst.size()]);
                                let iterations = (4 * 1024 * 1024usize)
                                    .div_ceil(
                                        data.len().saturating_add(output_cols * dst.size()).max(1),
                                    )
                                    .clamp(4, 4096);
                                let (direct_ns, whole_ns, speedup) = paired_scatter(
                                    &direct,
                                    &whole,
                                    &task,
                                    &data,
                                    &[],
                                    output_cols * dst.size(),
                                    fill,
                                    iterations,
                                );
                                let bytes_per_gap = mapped
                                    .saturating_mul(dst.size())
                                    .checked_div(ranges.len().max(1))
                                    .unwrap_or(0);
                                eprintln!(
                                    "DENSE_INIT_SCAN src={} dst={} fill={} map_kind={} mapping=1/{} defaults={} layout={} mapped={} output={} gaps={} bytes_per_gap={} planner={} direct_ns={:.2} whole_ns={:.2} speedup={:.4}",
                                    src,
                                    dst,
                                    fill_class.name(),
                                    map_kind,
                                    mapping_denominator,
                                    default_name,
                                    layout.name(),
                                    mapped,
                                    output_cols,
                                    ranges.len(),
                                    bytes_per_gap,
                                    if planner_whole { "whole" } else { "ranges" },
                                    direct_ns,
                                    whole_ns,
                                    speedup,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    fn benchmark_csr_sparse_mapping_thresholds() {
        const N_COLS: usize = 32 * 1024;
        const MAP_DENOMINATORS: [usize; 8] = [2, 5, 10, 20, 50, 100, 500, 1000];
        const NNZ_DENOMINATORS: [usize; 5] = [2, 5, 10, 20, 100];
        for src in STORAGE_DTYPES {
            for dst in OUTPUT_DTYPES {
                let Some(kind) = promote_kind(src, dst) else {
                    continue;
                };
                let output = output_spec(N_COLS, dst, OverflowPolicy::Error, kind);
                let fill = FillOp::new(&output.fill().encode()[..dst.size()]);
                let convert = ConvertOp::resolve(src, &output).unwrap();
                for map_denominator in MAP_DENOMINATORS {
                    let mapped = N_COLS.div_ceil(map_denominator);
                    for nnz_denominator in NNZ_DENOMINATORS {
                        let nnz = N_COLS.div_ceil(nnz_denominator);
                        for permuted in [false, true] {
                            let mapped_columns = if permuted {
                                permuted_columns(N_COLS, mapped, 40_503, 17)
                            } else {
                                uniform_columns(N_COLS, mapped)
                            };
                            let row_columns = if permuted {
                                permuted_columns(N_COLS, nnz, 23_771, 101)
                            } else {
                                uniform_columns(N_COLS, nnz)
                            };
                            let indices = encode_indices(&row_columns, StorageDType::U16);
                            let data = values(src, nnz, false);
                            let task = task(0..data.len(), Some(0..indices.len()));
                            let mut dense_targets = vec![UNMAPPED_TARGET_U32; N_COLS];
                            let mut sparse_targets = Vec::with_capacity(mapped);
                            for (target, &column) in mapped_columns.iter().enumerate() {
                                let target_byte = u32::try_from(target * dst.size()).unwrap();
                                dense_targets[column] = target_byte;
                                sparse_targets.push((column, target * dst.size()));
                            }
                            let current = SourcePlan {
                                feature_map: Some(CsrMap::Packed32(Arc::from(dense_targets))),
                                ..source(
                                    N_COLS,
                                    src,
                                    IndexOp::new(StorageDType::U16),
                                    None,
                                    false,
                                    Default::default(),
                                    convert,
                                )
                            };
                            validate_row(&current, &task, &data, &indices).unwrap();
                            let iterations = (4 * 1024 * 1024usize)
                                .div_ceil(data.len().saturating_add(mapped * dst.size()).max(1))
                                .clamp(4, 4096);
                            let (current_ns, binary_ns, speedup) = paired_csr_sparse_map(
                                &current,
                                &task,
                                &data,
                                &indices,
                                &sparse_targets,
                                mapped * dst.size(),
                                fill,
                                iterations,
                            );
                            eprintln!(
                                "CSR_SPARSE_MAP_SCAN src={} dst={} mapping=1/{} nnz=1/{} distribution={} mapped={} nnz_count={} current_ns={:.2} binary_ns={:.2} speedup={:.4}",
                                src,
                                dst,
                                map_denominator,
                                nnz_denominator,
                                if permuted { "permuted" } else { "uniform" },
                                mapped,
                                nnz,
                                current_ns,
                                binary_ns,
                                speedup,
                            );
                        }
                    }
                }
            }
        }
    }

    fn benchmark_csr_hybrid_boundaries() {
        const N_COLS: usize = 32 * 1024;
        const BOUNDARIES: [(usize, usize); 6] = [
            (100, 2),
            (500, 2),
            (500, 5),
            (1000, 10),
            (1000, 20),
            (1000, 100),
        ];
        for src in STORAGE_DTYPES {
            for dst in OUTPUT_DTYPES {
                let Some(kind) = promote_kind(src, dst) else {
                    continue;
                };
                let output = output_spec(N_COLS, dst, OverflowPolicy::Error, kind);
                let fill = FillOp::new(&output.fill().encode()[..dst.size()]);
                let convert = ConvertOp::resolve(src, &output).unwrap();
                for (map_denominator, nnz_denominator) in BOUNDARIES {
                    let mapped = N_COLS.div_ceil(map_denominator);
                    let nnz = N_COLS.div_ceil(nnz_denominator);
                    for permuted in [false, true] {
                        let mapped_columns = if permuted {
                            permuted_columns(N_COLS, mapped, 40_503, 17)
                        } else {
                            uniform_columns(N_COLS, mapped)
                        };
                        let row_columns = if permuted {
                            permuted_columns(N_COLS, nnz, 23_771, 101)
                        } else {
                            uniform_columns(N_COLS, nnz)
                        };
                        let indices = encode_indices(&row_columns, StorageDType::U16);
                        let data = values(src, nnz, false);
                        let task = task(0..data.len(), Some(0..indices.len()));
                        let mut dense_targets = vec![UNMAPPED_TARGET_U32; N_COLS];
                        let mut sparse_targets = Vec::with_capacity(mapped);
                        for (target, &column) in mapped_columns.iter().enumerate() {
                            let target_byte = u32::try_from(target * dst.size()).unwrap();
                            dense_targets[column] = target_byte;
                            sparse_targets
                                .push(u64::from(column as u32) | (u64::from(target_byte) << 32));
                        }
                        let dense: Arc<[u32]> = Arc::from(dense_targets);
                        let packed = SourcePlan {
                            feature_map: Some(CsrMap::Packed32(Arc::clone(&dense))),
                            ..source(
                                N_COLS,
                                src,
                                IndexOp::new(StorageDType::U16),
                                None,
                                false,
                                Default::default(),
                                convert,
                            )
                        };
                        let hybrid = SourcePlan {
                            feature_map: Some(CsrMap::Packed32(dense)),
                            csr_sparse_map: Some(CsrSparseMap::Packed32(Arc::from(sparse_targets))),
                            ..packed.clone()
                        };
                        validate_row(&hybrid, &task, &data, &indices).unwrap();
                        validate_row(&packed, &task, &data, &indices).unwrap();
                        let iterations = (4 * 1024 * 1024usize)
                            .div_ceil(data.len().saturating_add(mapped * dst.size()).max(1))
                            .clamp(4, 4096);
                        let (hybrid_ns, packed_ns, speedup) = paired_scatter(
                            &hybrid,
                            &packed,
                            &task,
                            &data,
                            &indices,
                            mapped * dst.size(),
                            fill,
                            iterations,
                        );
                        eprintln!(
                            "CSR_HYBRID_VERIFY src={} dst={} mapping=1/{} nnz=1/{} distribution={} selected={} hybrid_ns={:.2} packed_ns={:.2} speedup={:.4}",
                            src,
                            dst,
                            map_denominator,
                            nnz_denominator,
                            if permuted { "permuted" } else { "uniform" },
                            if csr_sparse_binary_is_cheaper(mapped, nnz) {
                                "sparse"
                            } else {
                                "dense"
                            },
                            hybrid_ns,
                            packed_ns,
                            speedup,
                        );
                    }
                }
            }
        }
    }

    fn permuted_columns(
        n_cols: usize,
        count: usize,
        multiplier: usize,
        offset: usize,
    ) -> Vec<usize> {
        debug_assert!(n_cols.is_power_of_two());
        let mut columns = (0..count)
            .map(|index| index.wrapping_mul(multiplier).wrapping_add(offset) & (n_cols - 1))
            .collect::<Vec<_>>();
        columns.sort_unstable();
        columns
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "paired benchmark contract is explicit"
    )]
    fn paired_csr_sparse_map(
        current: &SourcePlan,
        task: &CellTask,
        data: &[u8],
        indices: &[u8],
        sparse_targets: &[(usize, usize)],
        row_bytes: usize,
        fill: FillOp,
        iterations: usize,
    ) -> (f64, f64, f64) {
        const ROUNDS: usize = 7;
        let mut current_row = vec![0xA5; row_bytes];
        let mut binary_row = vec![0x5A; row_bytes];
        let mut current_samples = Vec::with_capacity(ROUNDS);
        let mut binary_samples = Vec::with_capacity(ROUNDS);
        for round in 0..ROUNDS {
            let current_run = |row: &mut [u8]| {
                let started = Instant::now();
                for _ in 0..iterations {
                    unsafe {
                        // SAFETY: setup validated the exact current CSR mapping.
                        scatter_row_prevalidated(
                            current, task, data, indices, row, row_bytes, fill,
                        )
                        .unwrap();
                    }
                }
                started.elapsed()
            };
            let binary_run = |row: &mut [u8]| {
                let started = Instant::now();
                for _ in 0..iterations {
                    unsafe {
                        // SAFETY: setup validated canonical indices, mapped
                        // conversions, and every sparse target extent.
                        scatter_csr_sparse_binary(
                            current,
                            task,
                            data,
                            indices,
                            sparse_targets,
                            row,
                        )
                        .unwrap();
                    }
                }
                started.elapsed()
            };
            if round & 1 == 0 {
                current_samples.push(current_run(&mut current_row));
                binary_samples.push(binary_run(&mut binary_row));
            } else {
                binary_samples.push(binary_run(&mut binary_row));
                current_samples.push(current_run(&mut current_row));
            }
        }
        assert_eq!(current_row, binary_row);
        let current = median(current_samples).as_secs_f64() * 1e9 / iterations as f64;
        let binary = median(binary_samples).as_secs_f64() * 1e9 / iterations as f64;
        (current, binary, current / binary)
    }

    unsafe fn scatter_csr_sparse_binary(
        source: &SourcePlan,
        task: &CellTask,
        data: &[u8],
        indices: &[u8],
        sparse_targets: &[(usize, usize)],
        row: &mut [u8],
    ) -> crate::Result<()> {
        row.fill(0);
        // SAFETY: setup validated this task's complete data range.
        let values = unsafe { data.get_unchecked(task.data_range()) };
        // SAFETY: setup validated this task's complete index range.
        let index_bytes = unsafe { indices.get_unchecked(task.indices_range()) };
        let count = index_bytes.len() / source.index.unwrap().size as usize;
        for &(column, target) in sparse_targets {
            if let Some(position) = binary_search_benchmark_index(index_bytes, 2, count, column) {
                // SAFETY: validation covered this mapped value and output target.
                unsafe {
                    source.convert.convert_one_prevalidated(
                        values.as_ptr().add(position << source.convert.src_shift),
                        row.as_mut_ptr().add(target),
                    )?;
                }
            }
        }
        Ok(())
    }

    fn binary_search_benchmark_index(
        indices: &[u8],
        index_size: usize,
        mut start: usize,
        target: usize,
    ) -> Option<usize> {
        let mut end = start;
        start = 0;
        while start < end {
            let middle = start + (end - start) / 2;
            let value = read_benchmark_index(indices, index_size, middle);
            if value < target {
                start = middle + 1;
            } else {
                end = middle;
            }
        }
        (start < indices.len() / index_size
            && read_benchmark_index(indices, index_size, start) == target)
            .then_some(start)
    }

    fn dense_map_name(map: &DenseMap) -> &'static str {
        match map {
            DenseMap::Packed32 { .. } => "packed",
            DenseMap::Gather32 { .. } => "gather",
            DenseMap::Wide { .. } => "wide",
            DenseMap::Runs { .. } => "runs",
        }
    }

    #[derive(Clone, Copy)]
    enum FillClass {
        Zero,
        NonZero,
    }

    impl FillClass {
        fn name(self) -> &'static str {
            match self {
                Self::Zero => "zero",
                Self::NonZero => "nonzero",
            }
        }
    }

    #[derive(Clone, Copy)]
    enum DefaultLayout {
        Tail,
        Runs64,
        Interleaved,
    }

    impl DefaultLayout {
        fn name(self) -> &'static str {
            match self {
                Self::Tail => "tail",
                Self::Runs64 => "runs64",
                Self::Interleaved => "interleaved",
            }
        }
    }

    fn mapped_positions(mapped: usize, defaults: usize, layout: DefaultLayout) -> Vec<usize> {
        if defaults == 0 {
            return (0..mapped).collect();
        }
        let requested_gaps = match layout {
            DefaultLayout::Tail => 1,
            DefaultLayout::Runs64 => 64,
            DefaultLayout::Interleaved => mapped.min(defaults),
        };
        let gaps = requested_gaps.min(mapped).min(defaults).max(1);
        let mut positions = Vec::with_capacity(mapped);
        let mut cursor = 0usize;
        for gap in 0..gaps {
            let mapped_start = gap * mapped / gaps;
            let mapped_end = (gap + 1) * mapped / gaps;
            let default_start = gap * defaults / gaps;
            let default_end = (gap + 1) * defaults / gaps;
            for _ in mapped_start..mapped_end {
                positions.push(cursor);
                cursor += 1;
            }
            cursor += default_end - default_start;
        }
        assert_eq!(positions.len(), mapped);
        assert_eq!(cursor, mapped + defaults);
        positions
    }

    fn fill_value(dtype: OutputDType, class: FillClass) -> Fill {
        match (dtype, class) {
            (OutputDType::I16, FillClass::Zero) => Fill::I16(0),
            (OutputDType::I16, FillClass::NonZero) => Fill::I16(7),
            (OutputDType::I32, FillClass::Zero) => Fill::I32(0),
            (OutputDType::I32, FillClass::NonZero) => Fill::I32(7),
            (OutputDType::I64, FillClass::Zero) => Fill::I64(0),
            (OutputDType::I64, FillClass::NonZero) => Fill::I64(7),
            (OutputDType::U16, FillClass::Zero) => Fill::U16(0),
            (OutputDType::U16, FillClass::NonZero) => Fill::U16(7),
            (OutputDType::U32, FillClass::Zero) => Fill::U32(0),
            (OutputDType::U32, FillClass::NonZero) => Fill::U32(7),
            (OutputDType::U64, FillClass::Zero) => Fill::U64(0),
            (OutputDType::U64, FillClass::NonZero) => Fill::U64(7),
            (OutputDType::F32, FillClass::Zero) => Fill::F32(0.0),
            (OutputDType::F32, FillClass::NonZero) => Fill::F32(7.0),
            (OutputDType::F64, FillClass::Zero) => Fill::F64(0.0),
            (OutputDType::F64, FillClass::NonZero) => Fill::F64(7.0),
        }
    }

    #[derive(Clone, Copy)]
    enum CsrPattern {
        Uniform,
        Clustered,
        Blocks,
    }

    impl CsrPattern {
        fn name(self) -> &'static str {
            match self {
                Self::Uniform => "uniform",
                Self::Clustered => "clustered",
                Self::Blocks => "blocks",
            }
        }
    }

    fn csr_columns(n_cols: usize, nnz: usize, pattern: CsrPattern) -> Vec<usize> {
        match pattern {
            CsrPattern::Uniform => uniform_columns(n_cols, nnz),
            CsrPattern::Clustered => (0..nnz).collect(),
            CsrPattern::Blocks => {
                let mut columns = Vec::with_capacity(nnz);
                let mut start = 0usize;
                while columns.len() < nnz {
                    for column in start..(start + 8).min(n_cols) {
                        if columns.len() == nnz {
                            break;
                        }
                        columns.push(column);
                    }
                    start = start.saturating_add(16);
                    if start >= n_cols {
                        start = 8;
                    }
                }
                columns.sort_unstable();
                columns.dedup();
                if columns.len() < nnz {
                    for column in 0..n_cols {
                        if columns.binary_search(&column).is_err() {
                            columns.push(column);
                            if columns.len() == nnz {
                                break;
                            }
                        }
                    }
                    columns.sort_unstable();
                }
                columns
            }
        }
    }

    fn paired_csr_identity_init(
        source: &SourcePlan,
        task: &CellTask,
        data: &[u8],
        indices: &[u8],
        row_bytes: usize,
        fill: FillOp,
        iterations: usize,
    ) -> (f64, f64, f64) {
        const ROUNDS: usize = 7;
        let mut current_row = vec![0xA5; row_bytes];
        let mut gaps_row = vec![0x5A; row_bytes];
        let mut current_samples = Vec::with_capacity(ROUNDS);
        let mut gaps_samples = Vec::with_capacity(ROUNDS);
        for round in 0..ROUNDS {
            let current = |row: &mut [u8]| {
                let started = Instant::now();
                for _ in 0..iterations {
                    unsafe {
                        // SAFETY: benchmark setup validated this exact CSR row.
                        scatter_row_prevalidated(source, task, data, indices, row, row_bytes, fill)
                            .unwrap();
                    }
                }
                started.elapsed()
            };
            let gaps = |row: &mut [u8]| {
                let started = Instant::now();
                for _ in 0..iterations {
                    unsafe {
                        // SAFETY: benchmark setup validated the canonical identity
                        // row consumed by the experimental gap/run kernel.
                        scatter_csr_identity_gaps(source, task, data, indices, row, row_bytes)
                            .unwrap();
                    }
                }
                started.elapsed()
            };
            if round & 1 == 0 {
                current_samples.push(current(&mut current_row));
                gaps_samples.push(gaps(&mut gaps_row));
            } else {
                gaps_samples.push(gaps(&mut gaps_row));
                current_samples.push(current(&mut current_row));
            }
        }
        assert_eq!(current_row, gaps_row);
        let current = median(current_samples).as_secs_f64() * 1e9 / iterations as f64;
        let gaps = median(gaps_samples).as_secs_f64() * 1e9 / iterations as f64;
        (current, gaps, current / gaps)
    }

    unsafe fn scatter_csr_identity_gaps(
        source: &SourcePlan,
        task: &CellTask,
        data: &[u8],
        indices: &[u8],
        row: &mut [u8],
        row_bytes: usize,
    ) -> crate::Result<()> {
        let index = source.index.unwrap();
        // SAFETY: benchmark setup validated this task's data range.
        let values = unsafe { data.get_unchecked(task.data_range()) };
        // SAFETY: benchmark setup validated this task's index range.
        let index_bytes = unsafe { indices.get_unchecked(task.indices_range()) };
        let count = values.len() >> source.convert.src_shift;
        let mut element = 0usize;
        let mut output_column = 0usize;
        while element < count {
            let column = read_benchmark_index(index_bytes, index.size as usize, element);
            let gap_bytes = (column - output_column) << source.convert.dst_shift;
            if gap_bytes != 0 {
                // SAFETY: canonical indices and the output extent bound this gap.
                unsafe {
                    row.as_mut_ptr()
                        .add(output_column << source.convert.dst_shift)
                        .write_bytes(0, gap_bytes);
                }
            }
            let mut run = 1usize;
            while element + run < count
                && read_benchmark_index(index_bytes, index.size as usize, element + run)
                    == column + run
            {
                run += 1;
            }
            // SAFETY: validation covers this complete contiguous value run and its
            // identity-mapped, non-overlapping output region.
            unsafe {
                source.convert.convert_slice_unchecked(
                    values.as_ptr().add(element << source.convert.src_shift),
                    row.as_mut_ptr().add(column << source.convert.dst_shift),
                    run,
                )?;
            }
            element += run;
            output_column = column + run;
        }
        let written = output_column << source.convert.dst_shift;
        if written < row_bytes {
            // SAFETY: `written <= row_bytes` and this is the untouched suffix.
            unsafe {
                row.as_mut_ptr()
                    .add(written)
                    .write_bytes(0, row_bytes - written)
            };
        }
        Ok(())
    }

    fn read_benchmark_index(indices: &[u8], index_size: usize, element: usize) -> usize {
        let offset = element * index_size;
        if index_size == 2 {
            u16::from_le_bytes(indices[offset..offset + 2].try_into().unwrap()) as usize
        } else {
            u32::from_le_bytes(indices[offset..offset + 4].try_into().unwrap()) as usize
        }
    }

    fn scan_conversion_mode(
        src: StorageDType,
        dst: OutputDType,
        kind: PromoteKind,
        policy_name: &str,
        policy: OverflowPolicy,
        invalid: bool,
        n_cols: usize,
    ) {
        let output = output_spec(n_cols, dst, policy, kind);
        let fill = FillOp::new(&output.fill().encode()[..dst.size()]);
        let input = values(src, n_cols, invalid);
        let specialized = ConvertOp::resolve(src, &output).unwrap();
        let mut generic = specialized;
        generic.force_generic_for_test();

        let identity_task = task(0..input.len(), None);
        let identity = source(
            n_cols,
            src,
            None,
            None,
            false,
            Default::default(),
            specialized,
        );
        let generic_identity = SourcePlan {
            convert: generic,
            ..identity.clone()
        };
        report_pair(
            src,
            dst,
            policy_name,
            "dense_identity",
            &identity,
            &generic_identity,
            &identity_task,
            &input,
            &[],
            n_cols * dst.size(),
            fill,
        );

        let mapped = n_cols / 2;
        let packed_entries = (0..n_cols)
            .step_by(2)
            .map(|column| {
                u64::from((column * src.size()) as u32)
                    | (u64::from((column / 2 * dst.size()) as u32) << 32)
            })
            .collect::<Vec<_>>();
        let packed = source(
            n_cols,
            src,
            None,
            Some(DenseMap::Packed32 {
                entries: Arc::from(packed_entries),
                covers_output: true,
            }),
            false,
            Default::default(),
            specialized,
        );
        let generic_packed = SourcePlan {
            convert: generic,
            ..packed.clone()
        };
        report_pair(
            src,
            dst,
            policy_name,
            "dense_packed_1_2",
            &packed,
            &generic_packed,
            &identity_task,
            &input,
            &[],
            mapped * dst.size(),
            fill,
        );

        if specialized.dense_gather_min_entries().is_some() {
            let gather = source(
                n_cols,
                src,
                None,
                Some(DenseMap::Gather32 {
                    source_offsets: Arc::from(
                        (0..n_cols)
                            .step_by(2)
                            .map(|column| i32::try_from(column * src.size()).unwrap())
                            .collect::<Vec<_>>(),
                    ),
                    target_byte: 0,
                    covers_output: true,
                }),
                false,
                Default::default(),
                specialized,
            );
            report_pair(
                src,
                dst,
                policy_name,
                "dense_gather_vs_packed_1_2",
                &gather,
                &packed,
                &identity_task,
                &input,
                &[],
                mapped * dst.size(),
                fill,
            );
        }

        for (density_name, nnz) in [("1_10", n_cols / 10), ("9_10", n_cols * 9 / 10)] {
            let columns = uniform_columns(n_cols, nnz);
            let indices = encode_indices(&columns, StorageDType::U16);
            let data = values(src, nnz, invalid);
            let csr_task = task(0..data.len(), Some(0..indices.len()));
            let csr = source(
                n_cols,
                src,
                IndexOp::new(StorageDType::U16),
                None,
                false,
                Default::default(),
                specialized,
            );
            let generic_csr = SourcePlan {
                convert: generic,
                ..csr.clone()
            };
            report_pair(
                src,
                dst,
                policy_name,
                &format!("csr_identity_nnz_{density_name}"),
                &csr,
                &generic_csr,
                &csr_task,
                &data,
                &indices,
                n_cols * dst.size(),
                fill,
            );

            let mut targets = vec![UNMAPPED_TARGET_U32; n_cols];
            for (target, column) in (0..n_cols).step_by(2).enumerate() {
                targets[column] = u32::try_from(target * dst.size()).unwrap();
            }
            let mapped_csr = SourcePlan {
                feature_map: Some(CsrMap::Packed32(Arc::from(targets))),
                ..csr.clone()
            };
            let generic_mapped_csr = SourcePlan {
                convert: generic,
                ..mapped_csr.clone()
            };
            report_pair(
                src,
                dst,
                policy_name,
                &format!("csr_mapped_1_2_nnz_{density_name}"),
                &mapped_csr,
                &generic_mapped_csr,
                &csr_task,
                &data,
                &indices,
                mapped * dst.size(),
                fill,
            );
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "benchmark case dimensions are explicit"
    )]
    fn report_pair(
        src: StorageDType,
        dst: OutputDType,
        policy: &str,
        path: &str,
        candidate: &SourcePlan,
        baseline: &SourcePlan,
        task: &CellTask,
        data: &[u8],
        indices: &[u8],
        row_bytes: usize,
        fill: FillOp,
    ) {
        if candidate.requires_runtime_validation() {
            validate_row(candidate, task, data, indices).unwrap();
            validate_row(baseline, task, data, indices).unwrap();
        }
        let bytes_per_iteration = data.len().saturating_add(row_bytes).max(1);
        let iterations = (8 * 1024 * 1024usize)
            .div_ceil(bytes_per_iteration)
            .clamp(4, 4096);
        let (candidate_ns, baseline_ns, speedup) = paired_scatter(
            candidate, baseline, task, data, indices, row_bytes, fill, iterations,
        );
        eprintln!(
            "FASTPATH_SCAN src={} dst={} policy={} path={} elements={} candidate_ns={:.2} baseline_ns={:.2} speedup={:.4}",
            src,
            dst,
            policy,
            path,
            data.len() / src.size(),
            candidate_ns,
            baseline_ns,
            speedup,
        );
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "paired benchmark contract is explicit"
    )]
    fn paired_scatter(
        candidate: &SourcePlan,
        baseline: &SourcePlan,
        task: &CellTask,
        data: &[u8],
        indices: &[u8],
        row_bytes: usize,
        fill: FillOp,
        iterations: usize,
    ) -> (f64, f64, f64) {
        const ROUNDS: usize = 7;
        let mut candidate_row = vec![0xA5; row_bytes];
        let mut baseline_row = vec![0x5A; row_bytes];
        let mut candidate_samples = Vec::with_capacity(ROUNDS);
        let mut baseline_samples = Vec::with_capacity(ROUNDS);
        for round in 0..ROUNDS {
            let measure = |source: &SourcePlan, row: &mut [u8]| {
                let started = Instant::now();
                for _ in 0..iterations {
                    unsafe {
                        // SAFETY: benchmark setup validated the immutable inputs;
                        // each measurement uniquely borrows its output row.
                        scatter_row_prevalidated(source, task, data, indices, row, row_bytes, fill)
                            .unwrap();
                    }
                }
                started.elapsed()
            };
            if round & 1 == 0 {
                candidate_samples.push(measure(candidate, &mut candidate_row));
                baseline_samples.push(measure(baseline, &mut baseline_row));
            } else {
                baseline_samples.push(measure(baseline, &mut baseline_row));
                candidate_samples.push(measure(candidate, &mut candidate_row));
            }
        }
        assert_eq!(candidate_row, baseline_row);
        let candidate = median(candidate_samples).as_secs_f64() * 1e9 / iterations as f64;
        let baseline = median(baseline_samples).as_secs_f64() * 1e9 / iterations as f64;
        (candidate, baseline, baseline / candidate)
    }

    fn median(mut samples: Vec<Duration>) -> Duration {
        samples.sort_unstable();
        samples[samples.len() / 2]
    }

    fn uniform_columns(n_cols: usize, count: usize) -> Vec<usize> {
        (0..count)
            .map(|index| index * n_cols / count.max(1))
            .collect()
    }

    fn benchmark_identity_short_rows() {
        const COUNTS: [usize; 10] = [1, 2, 3, 4, 7, 8, 15, 16, 31, 64];
        for src in STORAGE_DTYPES {
            for dst in OUTPUT_DTYPES {
                let Some(kind) = promote_kind(src, dst) else {
                    continue;
                };
                let policy = if kind == PromoteKind::CheckedSign {
                    OverflowPolicy::UseValue(sentinel(dst))
                } else {
                    OverflowPolicy::Error
                };
                for count in COUNTS {
                    let output = output_spec(count, dst, policy.clone(), kind);
                    let fill = FillOp::new(&output.fill().encode()[..dst.size()]);
                    let data = values(src, count, kind == PromoteKind::CheckedSign);
                    let specialized = ConvertOp::resolve(src, &output).unwrap();
                    let mut generic = specialized;
                    generic.force_generic_for_test();
                    let dense_task = task(0..data.len(), None);
                    let dense = source(
                        count,
                        src,
                        None,
                        None,
                        false,
                        Default::default(),
                        specialized,
                    );
                    let generic_dense = SourcePlan {
                        convert: generic,
                        ..dense.clone()
                    };
                    let iterations = (512 * 1024usize)
                        .div_ceil(data.len().saturating_add(count * dst.size()).max(1))
                        .clamp(256, 32_768);
                    let (_, _, dense_speedup) = paired_scatter(
                        &dense,
                        &generic_dense,
                        &dense_task,
                        &data,
                        &[],
                        count * dst.size(),
                        fill,
                        iterations,
                    );

                    let columns = (0..count).collect::<Vec<_>>();
                    let indices = encode_indices(&columns, StorageDType::U16);
                    let csr_task = task(0..data.len(), Some(0..indices.len()));
                    let csr = source(
                        count,
                        src,
                        IndexOp::new(StorageDType::U16),
                        None,
                        false,
                        Default::default(),
                        specialized,
                    );
                    let generic_csr = SourcePlan {
                        convert: generic,
                        ..csr.clone()
                    };
                    let (_, _, csr_speedup) = paired_scatter(
                        &csr,
                        &generic_csr,
                        &csr_task,
                        &data,
                        &indices,
                        count * dst.size(),
                        fill,
                        iterations,
                    );
                    eprintln!(
                        "IDENTITY_SHORT_SCAN src={} dst={} count={} dense_speedup={:.4} csr_speedup={:.4}",
                        src, dst, count, dense_speedup, csr_speedup,
                    );
                }
            }
        }
    }

    fn source(
        n_cols: usize,
        dtype: StorageDType,
        index: Option<IndexOp>,
        dense_map: Option<DenseMap>,
        dense_fill_whole: bool,
        default_ranges: Arc<[crate::plan::OutputRange]>,
        convert: ConvertOp,
    ) -> SourcePlan {
        SourcePlan {
            n_cols,
            value_dtype: dtype,
            index,
            feature_map: None,
            csr_sparse_map: None,
            dense_map,
            dense_fill_whole,
            default_ranges,
            convert,
        }
    }

    fn task(data: std::ops::Range<usize>, indices: Option<std::ops::Range<usize>>) -> CellTask {
        CellTask::new(OutputSlot::new(0).unwrap(), data, indices).unwrap()
    }

    fn output_spec(
        n_cols: usize,
        dtype: OutputDType,
        overflow: OverflowPolicy,
        kind: PromoteKind,
    ) -> OutputSpec {
        let mut output = OutputSpec::new(n_cols, dtype, zero(dtype))
            .unwrap()
            .overflow(overflow)
            .unwrap();
        if kind == PromoteKind::RoundingToFloat {
            output = output.float_cast(FloatCastPolicy::AllowRounding);
        }
        output
    }

    fn zero(dtype: OutputDType) -> Fill {
        match dtype {
            OutputDType::I16 => Fill::I16(0),
            OutputDType::I32 => Fill::I32(0),
            OutputDType::I64 => Fill::I64(0),
            OutputDType::U16 => Fill::U16(0),
            OutputDType::U32 => Fill::U32(0),
            OutputDType::U64 => Fill::U64(0),
            OutputDType::F32 => Fill::F32(0.0),
            OutputDType::F64 => Fill::F64(0.0),
        }
    }

    fn sentinel(dtype: OutputDType) -> Fill {
        match dtype {
            OutputDType::I16 => Fill::I16(7),
            OutputDType::I32 => Fill::I32(7),
            OutputDType::I64 => Fill::I64(7),
            OutputDType::U16 => Fill::U16(7),
            OutputDType::U32 => Fill::U32(7),
            OutputDType::U64 => Fill::U64(7),
            OutputDType::F32 => Fill::F32(7.0),
            OutputDType::F64 => Fill::F64(7.0),
        }
    }

    fn values(dtype: StorageDType, count: usize, invalid: bool) -> Vec<u8> {
        let mut output = Vec::with_capacity(count * dtype.size());
        for index in 0..count {
            match dtype {
                StorageDType::I16 => output.extend_from_slice(
                    &(if invalid && index % 5 == 0 {
                        -3i16
                    } else {
                        index as i16 % 251
                    })
                    .to_le_bytes(),
                ),
                StorageDType::I32 => output.extend_from_slice(
                    &(if invalid && index % 5 == 0 {
                        -3i32
                    } else {
                        index as i32 % 65_521
                    })
                    .to_le_bytes(),
                ),
                StorageDType::I64 => output.extend_from_slice(
                    &(if invalid && index % 5 == 0 {
                        -3i64
                    } else {
                        index as i64 * 17
                    })
                    .to_le_bytes(),
                ),
                StorageDType::U16 => output.extend_from_slice(
                    &(if invalid && index % 5 == 0 {
                        u16::MAX
                    } else {
                        index as u16 % 251
                    })
                    .to_le_bytes(),
                ),
                StorageDType::U32 => output.extend_from_slice(
                    &(if invalid && index % 5 == 0 {
                        u32::MAX
                    } else {
                        index as u32 * 17
                    })
                    .to_le_bytes(),
                ),
                StorageDType::U64 => output.extend_from_slice(
                    &(if invalid && index % 5 == 0 {
                        u64::MAX
                    } else {
                        index as u64 * 17
                    })
                    .to_le_bytes(),
                ),
                StorageDType::F32 => {
                    output.extend_from_slice(&((index as f32 - 17.0) * 0.25).to_le_bytes())
                }
                StorageDType::F64 => {
                    output.extend_from_slice(&((index as f64 - 17.0) * 0.25).to_le_bytes())
                }
            }
        }
        output
    }

    fn encode_indices(columns: &[usize], dtype: StorageDType) -> Vec<u8> {
        let mut output = Vec::with_capacity(columns.len() * dtype.size());
        for &column in columns {
            match dtype {
                StorageDType::U16 => output.extend_from_slice(&(column as u16).to_le_bytes()),
                StorageDType::U32 => output.extend_from_slice(&(column as u32).to_le_bytes()),
                _ => unreachable!(),
            }
        }
        output
    }

    pub(super) fn run(suite: &str) {
        match suite {
            "all" => {
                benchmark_all_conversion_fastpaths();
                benchmark_gather32_thresholds();
                benchmark_csr_identity_initialization_thresholds();
                benchmark_dense_initialization_thresholds();
                benchmark_csr_sparse_mapping_thresholds();
                benchmark_csr_hybrid_boundaries();
                benchmark_identity_short_rows();
            }
            "fastpaths" => benchmark_all_conversion_fastpaths(),
            "gather" => benchmark_gather32_thresholds(),
            "csr-init" => benchmark_csr_identity_initialization_thresholds(),
            "dense-init" => benchmark_dense_initialization_thresholds(),
            "csr-sparse" => benchmark_csr_sparse_mapping_thresholds(),
            "csr-hybrid" => benchmark_csr_hybrid_boundaries(),
            "identity" => benchmark_identity_short_rows(),
            other => panic!("unknown SC_LOAD_SCATTER_PROFILE suite: {other}"),
        }
    }
}

mod real {
    use std::hint::black_box;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use sc_compress::{open_csr, CsrOutput, DType, StoreLocation};

    use crate::compiler::{build_default_ranges, build_dense_map, choose_dense_whole_fill};
    use crate::convert::ConvertOp;
    use crate::plan::{
        csr_sparse_binary_is_cheaper, CellTask, CsrMap, CsrSparseMap, SourcePlan,
        UNMAPPED_TARGET_U32,
    };
    use crate::scatter::{scatter_row_prevalidated, validate_row, FillOp, IndexOp};
    use crate::source::OutputSlot;
    use crate::{Fill, OutputDType, OutputSpec};

    const TARGET_BYTES_PER_SAMPLE: usize = 128 * 1024 * 1024;
    const ROUNDS: usize = 7;

    pub(super) fn run() {
        let list = std::env::var_os("SC_LOAD_REAL_SCATTER_LIST")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("real_dataset.txt"));
        let rows = parse_env_usize("SC_LOAD_REAL_SCATTER_ROWS", 256);
        let minimum_denominator = parse_env_usize("SC_LOAD_REAL_SCATTER_MIN_DENOMINATOR", 1);
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
            benchmark_dataset(&path, rows, minimum_denominator);
        }
    }

    fn benchmark_dataset(path: &Path, requested_rows: usize, minimum_denominator: usize) {
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

        for denominator in [1usize, 2, 5, 10, 20, 50, 100, 500, 1000] {
            if denominator < minimum_denominator {
                continue;
            }
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
                let fill_bytes = output.fill().encode();
                let planner_whole = choose_dense_whole_fill(
                    Some(mapping.mapped),
                    value_size,
                    ranges.len(),
                    dense_map.as_ref(),
                    &fill_bytes[..value_size],
                )
                .unwrap();
                let dense_direct = SourcePlan {
                    n_cols,
                    value_dtype,
                    index: None,
                    feature_map: None,
                    csr_sparse_map: None,
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
                let csr_dense_source = SourcePlan {
                    n_cols,
                    value_dtype,
                    index: IndexOp::new(index_dtype),
                    feature_map: (!identity).then(|| CsrMap::Packed32(Arc::clone(&csr_targets))),
                    csr_sparse_map: None,
                    dense_map: None,
                    dense_fill_whole: false,
                    default_ranges: ranges,
                    convert,
                };
                let csr_sparse_map = (!identity
                    && csr_sparse_binary_is_cheaper(mapping.mapped, n_cols))
                .then(|| {
                    CsrSparseMap::Packed32(Arc::from(
                        mapping
                            .csr_targets
                            .iter()
                            .enumerate()
                            .filter(|(_, target)| **target != UNMAPPED_TARGET_U32)
                            .map(|(source, &target)| {
                                u64::from(source as u32) | (u64::from(target) << 32)
                            })
                            .collect::<Vec<_>>(),
                    ))
                });
                let csr_source = SourcePlan {
                    csr_sparse_map,
                    ..csr_dense_source.clone()
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
                let csr_dense_samples = measure_samples(
                    &csr_dense_source,
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
                let csr_dense_ns = median_ns_per_cell(csr_dense_samples, rows, iterations);
                let branchless_ns = median_ns_per_cell(branchless_samples, rows, iterations);
                let adaptive_csr_ns = median_ns_per_cell(adaptive_csr_samples, rows, iterations);
                let mapped_hits = count_mapped_hits(&indices, index_size, &csr_targets);
                let hit_fraction = mapped_hits as f64 / (indices.len() / index_size).max(1) as f64;
                let auto_ns = if planner_whole { whole_ns } else { direct_ns };
                eprintln!(
                    "REAL_SCATTER ratio=1/{denominator} missing={} mapped={} hit_fraction={hit_fraction:.4} out_cols={} gap_runs={} planner={} iterations={} dense_direct_ns_cell={direct_ns:.2} dense_whole_ns_cell={whole_ns:.2} whole_over_direct={:.4} dense_auto_ns_cell={auto_ns:.2} csr_auto_ns_cell={csr_ns:.2} csr_dense_ns_cell={csr_dense_ns:.2} csr_auto_speedup={:.4} csr_branchless_ns_cell={branchless_ns:.2} csr_adaptive_ns_cell={adaptive_csr_ns:.2} csr_over_branchless={:.4} csr_over_adaptive={:.4}",
                    if missing_third { "1/3" } else { "0" },
                    mapping.mapped,
                    mapping.output_cols,
                    dense_direct.default_ranges.len(),
                    if planner_whole { "whole" } else { "ranges" },
                    iterations,
                    whole_ns / direct_ns,
                    csr_dense_ns / csr_ns,
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
}
