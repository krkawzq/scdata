use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use sc_compress::{ByteStore, DirectoryStore};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every operation forwards the original allocation contract unchanged
// to the system allocator and only records successful allocations.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is forwarded unchanged.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is forwarded unchanged.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: `pointer` and `layout` came from the system allocator.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the original allocation and requested size are forwarded unchanged.
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

fn median<T: Ord + Copy>(samples: &mut [T]) -> T {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn offsets(file_len: usize, read_len: usize, iterations: usize) -> Vec<u64> {
    let positions = file_len / read_len;
    (0..iterations)
        .map(|index| ((index * 131 % positions) * read_len) as u64)
        .collect()
}

fn measure_allocating(
    store: &DirectoryStore,
    read_len: usize,
    offsets: &[u64],
) -> (Duration, usize, usize) {
    black_box(store.read_range("value", offsets[0], read_len).unwrap());
    let mut elapsed = Vec::with_capacity(11);
    let mut allocations = Vec::with_capacity(11);
    let mut allocated_bytes = Vec::with_capacity(11);
    for _ in 0..11 {
        ALLOCATIONS.store(0, Ordering::Relaxed);
        ALLOCATED_BYTES.store(0, Ordering::Relaxed);
        let start = Instant::now();
        for &offset in offsets {
            black_box(store.read_range("value", offset, read_len).unwrap());
        }
        elapsed.push(start.elapsed());
        allocations.push(ALLOCATIONS.load(Ordering::Relaxed));
        allocated_bytes.push(ALLOCATED_BYTES.load(Ordering::Relaxed));
    }
    (
        median(&mut elapsed),
        median(&mut allocations),
        median(&mut allocated_bytes),
    )
}

fn measure_reused(
    store: &DirectoryStore,
    read_len: usize,
    offsets: &[u64],
) -> (Duration, usize, usize) {
    let mut output = Vec::new();
    store
        .read_range_into("value", offsets[0], read_len, &mut output)
        .unwrap();
    let mut elapsed = Vec::with_capacity(11);
    let mut allocations = Vec::with_capacity(11);
    let mut allocated_bytes = Vec::with_capacity(11);
    for _ in 0..11 {
        ALLOCATIONS.store(0, Ordering::Relaxed);
        ALLOCATED_BYTES.store(0, Ordering::Relaxed);
        let start = Instant::now();
        for &offset in offsets {
            store
                .read_range_into("value", offset, read_len, &mut output)
                .unwrap();
            black_box(&output);
        }
        elapsed.push(start.elapsed());
        allocations.push(ALLOCATIONS.load(Ordering::Relaxed));
        allocated_bytes.push(ALLOCATED_BYTES.load(Ordering::Relaxed));
    }
    (
        median(&mut elapsed),
        median(&mut allocations),
        median(&mut allocated_bytes),
    )
}

fn main() {
    const FILE_LEN: usize = 8 * 1024 * 1024;
    const ITERATIONS: usize = 2_048;
    let temp = tempfile::tempdir().unwrap();
    let values = (0..FILE_LEN)
        .map(|index| (index.wrapping_mul(17) & 0xff) as u8)
        .collect::<Vec<_>>();
    std::fs::write(temp.path().join("value"), values).unwrap();
    let store = DirectoryStore::open(temp.path()).unwrap();

    for read_len in [4 * 1024, 64 * 1024] {
        let offsets = offsets(FILE_LEN, read_len, ITERATIONS);
        let allocating = measure_allocating(&store, read_len, &offsets);
        let reused = measure_reused(&store, read_len, &offsets);
        println!(
            "range_read bytes={read_len} iterations={ITERATIONS} \
             allocating_median={:?} allocating_allocations={} allocating_bytes={} \
             reused_median={:?} reused_allocations={} reused_bytes={} speedup={:.3}x",
            allocating.0,
            allocating.1,
            allocating.2,
            reused.0,
            reused.1,
            reused.2,
            allocating.0.as_secs_f64() / reused.0.as_secs_f64(),
        );
    }
}
