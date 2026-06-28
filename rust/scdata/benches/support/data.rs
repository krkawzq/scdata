//! Normalized synthetic data generation pipeline.
//!
//! Produces deterministic raw byte payloads across the axes that matter for
//! codec and databank benches: dtype, value distribution, missing-value ratio,
//! and chunk geometry. The benches use these profiles to sweep coverage
//! (scenario × scale × missing-rate × distribution) without depending on the
//! Python numcodecs exporter.
//!
//! Determinism: every generator is seeded by the profile so runs are
//! reproducible. `splitmix64` is the only PRNG — no `std::time` or `rand`.

use _scdata::databank::DType;

/// Value distribution for the non-missing elements of a payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataDist {
    /// Uniformly random bytes (high entropy, near-incompressible).
    Uniform,
    /// Monotonically increasing counter (moderate entropy).
    Counting,
    /// Single repeating value (maximally compressible).
    Constant,
    /// Few distinct values from a small pool (high compression).
    LowEntropy,
    /// Cryptographic-style random bytes (incompressible baseline).
    HighEntropy,
}

impl DataDist {
    pub fn label(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Counting => "counting",
            Self::Constant => "constant",
            Self::LowEntropy => "low_entropy",
            Self::HighEntropy => "high_entropy",
        }
    }

    pub const ALL: [Self; 5] = [
        Self::Uniform,
        Self::Counting,
        Self::Constant,
        Self::LowEntropy,
        Self::HighEntropy,
    ];
}

/// A complete description of a synthetic payload.
///
/// `missing_permille` is the fraction (in per-mille) of elements set to the
/// dtype's missing value: zero for integer dtypes, NaN for float dtypes. It
/// simulates sparse / dropout single-cell data.
#[derive(Debug, Clone, Copy)]
pub struct DataProfile {
    pub dtype: DType,
    pub dist: DataDist,
    pub missing_permille: u32,
    pub chunk_bytes: usize,
    pub num_chunks: usize,
    pub seed: u64,
}

impl DataProfile {
    /// Human-readable label stable across runs, e.g. `u32_uniform_miss250_c64kx16`.
    pub fn label(&self) -> String {
        format!(
            "{}_{}_miss{}_c{}_x{}",
            dtype_label(self.dtype),
            self.dist.label(),
            self.missing_permille,
            fmt_bytes(self.chunk_bytes),
            self.num_chunks,
        )
    }

    pub fn total_bytes(&self) -> usize {
        self.chunk_bytes * self.num_chunks
    }

    pub fn item_size(&self) -> usize {
        self.dtype.item_size()
    }

    /// Generate the full raw payload (`chunk_bytes * num_chunks` bytes).
    pub fn generate(&self) -> Vec<u8> {
        let item_size = self.item_size();
        let total = self.total_bytes();
        debug_assert!(total % item_size == 0, "total bytes must be item-aligned");
        let num_items = total / item_size;

        let mut out = vec![0u8; total];
        let mut rng = Rng::new(self.seed);
        let missing = self.dtype_missing_value();
        let missing_threshold = u32::MAX / 1000 * self.missing_permille.min(1000);

        for idx in 0..num_items {
            let is_missing = self.missing_permille > 0 && rng.next_u32() < missing_threshold;
            let bytes = if is_missing {
                missing
            } else {
                self.value_bytes(idx, &mut rng)
            };
            let start = idx * item_size;
            out[start..start + item_size].copy_from_slice(&bytes[..item_size]);
        }
        out
    }

    /// Split a raw payload into `num_chunks` owned slices of `chunk_bytes`.
    pub fn generate_chunks(&self) -> Vec<Vec<u8>> {
        let raw = self.generate();
        raw.chunks_exact(self.chunk_bytes)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    fn value_bytes(&self, idx: usize, rng: &mut Rng) -> [u8; 8] {
        let item = self.item_size();
        let raw: u64 = match self.dist {
            DataDist::Uniform | DataDist::HighEntropy => rng.next_u64(),
            DataDist::Counting => idx as u64,
            DataDist::Constant => 0x4241_4241_4241_4241,
            DataDist::LowEntropy => {
                let pool = [0u64, 1, 2, 4, 8, 16, 32, 64];
                pool[idx % pool.len()]
            }
        };
        let mut buf = [0u8; 8];
        buf[..item].copy_from_slice(&raw.to_ne_bytes()[..item]);
        buf
    }

    /// Missing value as raw little-endian bytes: zero for integers, NaN for
    /// floats (f32/f64/f16/bf16).
    fn dtype_missing_value(&self) -> [u8; 8] {
        let mut buf = [0u8; 8];
        let nan_bytes = match self.dtype {
            DType::F32 => (f32::NAN.to_bits() as u64).to_ne_bytes(),
            DType::F64 => f64::NAN.to_bits().to_ne_bytes(),
            DType::F16 | DType::BF16 => (0x7e00u64).to_ne_bytes(),
            _ => return buf,
        };
        let item = self.item_size();
        buf[..item].copy_from_slice(&nan_bytes[..item]);
        buf
    }
}

/// Format a byte count as a compact human-readable string (e.g. `64k`, `1m`).
pub fn fmt_bytes(bytes: usize) -> String {
    if bytes >= 1 << 20 {
        format!("{}m", bytes >> 20)
    } else if bytes >= 1 << 10 {
        format!("{}k", bytes >> 10)
    } else {
        format!("{}b", bytes)
    }
}

pub fn dtype_label(dtype: DType) -> &'static str {
    match dtype {
        DType::U8 => "u8",
        DType::I8 => "i8",
        DType::U16 => "u16",
        DType::I16 => "i16",
        DType::U32 => "u32",
        DType::I32 => "i32",
        DType::U64 => "u64",
        DType::I64 => "i64",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::F32 => "f32",
        DType::F64 => "f64",
    }
}

/// SplitMix64 PRNG — deterministic, no std::time / rand dependency.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9e37_79b9_7f4a_7c15),
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
}
