use dyn_blosc::{Codec, Decoder, Encoder, Shuffle};

fn sine_f32(count: usize) -> Vec<u8> {
    (0..count)
        .flat_map(|index| (((index as f32) * 0.017).sin()).to_le_bytes())
        .collect()
}

#[test]
fn every_codec_filter_and_level_roundtrips() {
    let source = sine_f32(25_000);
    for codec in [Codec::BloscLz, Codec::Lz4, Codec::Zlib, Codec::Zstd] {
        for shuffle in [Shuffle::None, Shuffle::Bytes, Shuffle::Bits] {
            for level in [1, 5, 9] {
                let encoded = Encoder::new()
                    .codec(codec)
                    .shuffle(shuffle)
                    .element_size(4)
                    .compression_level(level)
                    .encode(&source)
                    .unwrap();
                let decoded = Decoder::from_encoded(&encoded)
                    .unwrap()
                    .decode(&encoded)
                    .unwrap_or_else(|error| panic!("{codec:?} {shuffle:?} level={level}: {error}"));
                assert_eq!(decoded, source, "{codec:?} {shuffle:?} level={level}");
            }
        }
    }
}

#[test]
fn split_and_unsplit_blocks_roundtrip() {
    let source = sine_f32(50_000);
    for codec in [Codec::BloscLz, Codec::Lz4, Codec::Zlib, Codec::Zstd] {
        for split_blocks in [false, true] {
            let encoded = Encoder::new()
                .codec(codec)
                .shuffle(Shuffle::Bytes)
                .element_size(4)
                .split_blocks(split_blocks)
                .encode(&source)
                .unwrap();
            let decoded = Decoder::from_encoded(&encoded)
                .unwrap()
                .decode(&encoded)
                .unwrap_or_else(|error| panic!("{codec:?} split_blocks={split_blocks}: {error}"));
            assert_eq!(decoded, source, "{codec:?} split_blocks={split_blocks}");
        }
    }
}

#[test]
fn incompressible_input_is_stored_losslessly() {
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut source = Vec::with_capacity(256 * 1024);
    for _ in 0..source.capacity() / 8 {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        source.extend_from_slice(&state.wrapping_mul(0x2545F4914F6CDD1D).to_le_bytes());
    }
    for codec in [Codec::BloscLz, Codec::Lz4, Codec::Zlib, Codec::Zstd] {
        let encoded = Encoder::new()
            .codec(codec)
            .shuffle(Shuffle::None)
            .element_size(1)
            .encode(&source)
            .unwrap();
        assert_eq!(
            Decoder::from_encoded(&encoded)
                .unwrap()
                .decode(&encoded)
                .unwrap(),
            source,
            "{codec:?}"
        );
    }
}
