use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use sc_compress::{AxisIndex, CsrArray, CsrOutput, DType, DenseArray};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarding the unchanged layout preserves `System`'s allocator contract.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarding the unchanged layout preserves `System`'s allocator contract.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: `pointer` and `layout` came from the corresponding `System` allocation.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwarding the original allocation and requested size preserves
        // `System`'s reallocation contract.
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        new_pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn reset_allocations() {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
}

fn median<T: Ord + Copy>(values: &mut [T]) -> T {
    values.sort_unstable();
    values[values.len() / 2]
}

fn measure<I, R>(
    name: &str,
    threads: usize,
    mut prepare: impl FnMut() -> I,
    mut operation: impl FnMut(I) -> R,
) {
    for _ in 0..2 {
        black_box(operation(prepare()));
    }

    let mut elapsed = Vec::with_capacity(11);
    let mut allocations = Vec::with_capacity(11);
    let mut allocated_bytes = Vec::with_capacity(11);
    for _ in 0..11 {
        let input = prepare();
        reset_allocations();
        let start = Instant::now();
        let result = operation(input);
        elapsed.push(start.elapsed());
        allocations.push(ALLOCATIONS.load(Ordering::Relaxed));
        allocated_bytes.push(ALLOCATED_BYTES.load(Ordering::Relaxed));
        black_box(result);
    }

    let elapsed = median(&mut elapsed);
    let allocations = median(&mut allocations);
    let allocated_bytes = median(&mut allocated_bytes);
    println!(
        "{name} threads={threads} median={elapsed:?} allocations={allocations} allocated_bytes={allocated_bytes}"
    );
}

fn dense_fixture() -> DenseArray {
    let rows = 10_000usize;
    let cols = 1_024usize;
    DenseArray::from_bytes(
        [rows, cols],
        DType::F32,
        vec![0; rows * cols * DType::F32.size()],
    )
    .unwrap()
}

fn csr_fixture() -> CsrArray {
    let rows = 20_000usize;
    let cols = 4_096usize;
    let nnz_per_row = 128usize;
    let indptr = (0..=rows)
        .map(|row| (row * nnz_per_row) as u64)
        .collect::<Vec<_>>();
    let packed_row = (0..nnz_per_row as u16)
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let mut indices = Vec::with_capacity(rows * packed_row.len());
    for _ in 0..rows {
        indices.extend_from_slice(&packed_row);
    }
    let data = vec![0; rows * nnz_per_row * DType::F32.size()];
    CsrArray::from_parts([rows, cols], DType::U16, DType::F32, indptr, indices, data).unwrap()
}

fn main() {
    let dense = dense_fixture();
    let dense_rows = AxisIndex::strided(0, dense.n_rows() as i64, 2)
        .normalize(dense.n_rows() as u64)
        .unwrap();
    let dense_cols = AxisIndex::positions((0..256u64).map(|column| column * 3 + 1))
        .normalize(dense.n_cols() as u64)
        .unwrap();

    let csr = csr_fixture();
    let range_cols = AxisIndex::range(32, 96)
        .normalize(csr.n_cols() as u64)
        .unwrap();
    let gather_cols = AxisIndex::positions((0..128u64).step_by(2))
        .normalize(csr.n_cols() as u64)
        .unwrap();

    for threads in [1, 4] {
        measure(
            "dense_gather",
            threads,
            || (),
            |()| {
                dense
                    .select_normalized(&dense_rows, &dense_cols, threads)
                    .unwrap()
            },
        );
        measure(
            "csr_range_sparse",
            threads,
            || csr.clone(),
            |input| {
                input
                    .select_columns(&range_cols, CsrOutput::Sparse, threads)
                    .unwrap()
            },
        );
        measure(
            "csr_gather_sparse",
            threads,
            || csr.clone(),
            |input| {
                input
                    .select_columns(&gather_cols, CsrOutput::Sparse, threads)
                    .unwrap()
            },
        );
        measure(
            "csr_gather_dense",
            threads,
            || csr.clone(),
            |input| {
                input
                    .select_columns(&gather_cols, CsrOutput::Dense, threads)
                    .unwrap()
            },
        );
    }
}
