use dyn_blosc::{
    ByteMapping, ByteSelection, Codec, DecodeLimits, DecodeWorkspace, Decoder, EncodeWorkspace,
    Encoder, Error, LimitKind, Shuffle,
};

fn sample(length: usize) -> Vec<u8> {
    (0..length)
        .map(|index| ((index * 7 + 1) & 0xff) as u8)
        .collect()
}

#[test]
fn invalid_encoder_options_are_rejected() {
    let source = sample(100);
    for encoder in [
        Encoder::new().element_size(0),
        Encoder::new().element_size(256),
        Encoder::new().compression_level(10),
        Encoder::new().threads(0),
        Encoder::new().block_size(0),
        Encoder::new().block_lengths([99]),
        Encoder::new().block_lengths([0, 100]),
        // Non-final block length not a multiple of typesize (4).
        Encoder::new().block_lengths([50, 50]),
        Encoder::new().block_lengths([33, 67]),
    ] {
        assert!(
            matches!(encoder.encode(&source), Err(Error::InvalidOptions(_))),
            "{encoder:?}"
        );
    }

    // Fixed size that is not a multiple of typesize must fail once it yields
    // more than one block.
    assert!(matches!(
        Encoder::new()
            .element_size(4)
            .block_size(101)
            .encode(&sample(303)),
        Err(Error::InvalidOptions(_))
    ));

    // Single-block payloads may end mid-element; only block *starts* must align.
    assert!(Encoder::new()
        .element_size(4)
        .block_size(101)
        .encode(&sample(101))
        .is_ok());
    assert!(Encoder::new()
        .element_size(4)
        .block_lengths([100])
        .encode(&sample(100))
        .is_ok());
}

#[test]
fn empty_inputs_still_validate_explicit_block_partitions() {
    for encoder in [
        Encoder::new().block_size(0),
        Encoder::new().block_lengths([]),
        Encoder::new().block_lengths([1]),
    ] {
        assert!(
            matches!(encoder.encode(&[]), Err(Error::InvalidOptions(_))),
            "{encoder:?}"
        );
    }
    assert!(Encoder::new().automatic_block_size().encode(&[]).is_ok());
}

#[test]
fn decode_into_checks_output_size() {
    let source = sample(4096);
    let encoded = Encoder::new().element_size(4).encode(&source).unwrap();
    let decoder = Decoder::from_encoded(&encoded).unwrap();

    let mut short = vec![0; source.len() - 1];
    assert!(matches!(
        decoder.decode_into(&encoded, &mut short),
        Err(Error::BufferTooSmall { .. })
    ));

    let mut exact = vec![0; source.len()];
    assert_eq!(
        decoder.decode_into(&encoded, &mut exact).unwrap(),
        source.len()
    );
    assert_eq!(exact, source);
}

#[test]
fn decoder_limit_is_enforced_before_output_allocation() {
    let encoded = Encoder::new().encode(&sample(4096)).unwrap();
    assert!(matches!(
        Decoder::from_encoded(&encoded)
            .unwrap()
            .with_limits(DecodeLimits::unlimited().maximum_decoded_size(1024))
            .unwrap_err(),
        Error::LimitExceeded {
            kind: LimitKind::DecodedSize,
            actual: 4096,
            limit: 1024
        }
    ));
    assert!(matches!(
        Decoder::from_encoded_with_limits(
            &encoded,
            DecodeLimits::unlimited().maximum_block_size(1024)
        ),
        Err(Error::LimitExceeded {
            kind: LimitKind::BlockSize,
            ..
        })
    ));
}

#[test]
fn complete_chunk_must_match_decoder_schema() {
    let encoded = Encoder::new()
        .block_size(256)
        .encode(&sample(2048))
        .unwrap();
    let decoder = Decoder::from_encoded(&encoded).unwrap();
    let mut different_schema = encoded.clone();
    different_schema[2] ^= 0x10;

    assert!(matches!(
        decoder.decode(&different_schema),
        Err(Error::SchemaMismatch(_))
    ));
}

#[test]
fn decode_limits_reject_block_count_before_index_parsing() {
    let encoded = Encoder::new().block_size(64).encode(&sample(4096)).unwrap();
    let header = &encoded[..16];
    assert!(matches!(
        Decoder::from_prefix_with_limits(header, DecodeLimits::unlimited().maximum_block_count(1)),
        Err(Error::LimitExceeded {
            kind: LimitKind::BlockCount,
            ..
        })
    ));
}

#[test]
fn workspaces_can_be_reused_and_block_encoding_is_transactional() {
    let source = sample(4096);
    let encoded = Encoder::new().block_size(256).encode(&source).unwrap();
    let decoder = Decoder::from_encoded(&encoded).unwrap();
    let selection = ByteSelection::contiguous(100..900).unwrap();
    let mut output = vec![0; selection.output_len()];
    let mut decode_workspace = DecodeWorkspace::new();
    for _ in 0..2 {
        decoder
            .decode_selection_into(&encoded, &selection, &mut output, &mut decode_workspace)
            .unwrap();
        assert_eq!(output, source[100..900]);
    }

    let encoder = Encoder::new();
    let mut appended = vec![1, 2, 3];
    let mut encode_workspace = EncodeWorkspace::new();
    assert!(encoder
        .encode_block_into(&[], &mut appended, &mut encode_workspace)
        .is_err());
    assert_eq!(appended, [1, 2, 3]);

    for (codec, level) in [
        (Codec::Zlib, 1),
        (Codec::Zlib, 9),
        (Codec::Zstd, 1),
        (Codec::Zstd, 9),
    ] {
        let encoder = Encoder::new()
            .codec(codec)
            .compression_level(level)
            .shuffle(Shuffle::Bytes)
            .element_size(4);
        let mut payload = Vec::new();
        encoder
            .encode_block_into(&source, &mut payload, &mut encode_workspace)
            .unwrap();
        let descriptor = dyn_blosc::BlockDescriptor::new(source.len(), payload.len()).unwrap();
        let decoder = Decoder::from_layout(encoder.chunk_layout(&[descriptor]).unwrap());
        let mut decoded = vec![0; source.len()];
        decoder
            .decode_block_into(0, &payload, &mut decoded, &mut decode_workspace)
            .unwrap();
        assert_eq!(decoded, source, "{codec:?} level={level}");
    }

    let impossible =
        dyn_blosc::BlockDescriptor::new(128, encoder.maximum_encoded_block_len(128).unwrap() + 1)
            .unwrap();
    assert!(encoder.chunk_layout(&[impossible]).is_err());
}

#[test]
fn block_decoders_are_self_contained_and_reusable() {
    let source = sample(4096);
    let encoded = Encoder::new()
        .element_size(4)
        .shuffle(Shuffle::Bytes)
        .block_size(256)
        .encode(&source)
        .unwrap();
    let decoder = Decoder::from_encoded(&encoded).unwrap();
    assert!(!decoder.header().is_raw());
    let block = decoder.block(3).unwrap();
    let block_decoder = decoder.block_decoder(3).unwrap();
    let payload = encoded[block.encoded_range()].to_vec();
    let expected = source[block.decoded_range()].to_vec();
    drop(decoder);

    assert_eq!(block_decoder.encoded_len(), payload.len());
    assert_eq!(block_decoder.decoded_len(), expected.len());
    let mut workspace = DecodeWorkspace::new();
    let mut output = vec![0; block_decoder.decoded_len()];
    for _ in 0..2 {
        assert_eq!(
            block_decoder
                .decode_into(&payload, &mut output, &mut workspace)
                .unwrap(),
            expected.len()
        );
        assert_eq!(output, expected);
    }

    assert!(matches!(
        block_decoder.decode_into(&payload[..payload.len() - 1], &mut output, &mut workspace),
        Err(Error::InvalidArgument(_))
    ));
    let short_output_len = output.len() - 1;
    assert!(matches!(
        block_decoder.decode_into(&payload, &mut output[..short_output_len], &mut workspace),
        Err(Error::BufferTooSmall { .. })
    ));
}

#[test]
fn byte_selections_validate_ranges_and_destinations() {
    let (reversed_start, reversed_end) = (10, 5);
    assert!(ByteMapping::new(reversed_start..reversed_end, 0).is_err());
    assert!(ByteMapping::new(0..2, usize::MAX).is_err());
    assert!(ByteSelection::new(
        vec![
            ByteMapping::new(0..10, 0).unwrap(),
            ByteMapping::new(20..30, 5).unwrap(),
        ],
        20,
    )
    .is_err());

    let source = sample(1024);
    let encoded = Encoder::new().encode(&source).unwrap();
    let decoder = Decoder::from_encoded(&encoded).unwrap();
    assert!(decoder.decode_bytes(&encoded, 500..2000).is_err());
    assert!(decoder.decode_items(&encoded, 300..400).is_err());
}

#[test]
fn selection_can_reorder_disjoint_ranges() {
    let source = sample(1024);
    let encoded = Encoder::new().block_size(64).encode(&source).unwrap();
    let selection = ByteSelection::new(
        vec![
            ByteMapping::new(100..120, 20).unwrap(),
            ByteMapping::new(500..520, 0).unwrap(),
        ],
        40,
    )
    .unwrap();
    let decoded = Decoder::from_encoded(&encoded)
        .unwrap()
        .decode_selection(&encoded, &selection)
        .unwrap();
    assert_eq!(&decoded[..20], &source[500..520]);
    assert_eq!(&decoded[20..], &source[100..120]);
}

#[test]
fn large_disjoint_selection_roundtrips() {
    const BLOCKS: usize = 512;
    const BLOCK_SIZE: usize = 64;

    let source = sample(BLOCKS * BLOCK_SIZE);
    let encoded = Encoder::new()
        .block_size(BLOCK_SIZE)
        .shuffle(Shuffle::None)
        .encode(&source)
        .unwrap();
    let mappings = (0..BLOCKS)
        .map(|block| {
            let source_start = block * BLOCK_SIZE + block % BLOCK_SIZE;
            ByteMapping::new(source_start..source_start + 1, BLOCKS - block - 1).unwrap()
        })
        .collect();
    let selection = ByteSelection::new(mappings, BLOCKS).unwrap();
    let decoded = Decoder::from_encoded(&encoded)
        .unwrap()
        .decode_selection(&encoded, &selection)
        .unwrap();
    let expected = (0..BLOCKS)
        .rev()
        .map(|block| source[block * BLOCK_SIZE + block % BLOCK_SIZE])
        .collect::<Vec<_>>();
    assert_eq!(decoded, expected);
}

#[test]
fn sparse_and_empty_selections_decode_the_requested_layout() {
    let source = sample(64 * 1024);
    let encoded = Encoder::new()
        .block_size(64)
        .shuffle(Shuffle::None)
        .encode(&source)
        .unwrap();
    let decoder = Decoder::from_encoded(&encoded).unwrap();
    let selection = ByteSelection::new(
        vec![
            ByteMapping::new(63..130, 227).unwrap(),
            ByteMapping::new(32_000..32_127, 0).unwrap(),
            ByteMapping::new(65_000..65_100, 127).unwrap(),
        ],
        294,
    )
    .unwrap();
    let decoded = decoder.decode_selection(&encoded, &selection).unwrap();
    assert_eq!(&decoded[0..127], &source[32_000..32_127]);
    assert_eq!(&decoded[127..227], &source[65_000..65_100]);
    assert_eq!(&decoded[227..294], &source[63..130]);

    let empty = ByteSelection::new(Vec::new(), 128).unwrap();
    assert_eq!(
        decoder.decode_selection(&encoded, &empty).unwrap(),
        vec![0; 128]
    );

    let mut reused = vec![0xA5; 128];
    let mut workspace = DecodeWorkspace::new();
    decoder
        .decode_selection_into(&encoded, &empty, &mut reused, &mut workspace)
        .unwrap();
    assert_eq!(reused, vec![0; 128]);
}

#[test]
fn overlapping_source_selections_roundtrip() {
    let source = sample(4096);
    let encoded = Encoder::new()
        .block_size(64)
        .shuffle(Shuffle::None)
        .encode(&source)
        .unwrap();
    let selection = ByteSelection::new(
        vec![
            ByteMapping::new(60..180, 0).unwrap(),
            ByteMapping::new(100..164, 120).unwrap(),
        ],
        184,
    )
    .unwrap();

    let decoded = Decoder::from_encoded(&encoded)
        .unwrap()
        .decode_selection(&encoded, &selection)
        .unwrap();
    assert_eq!(&decoded[..120], &source[60..180]);
    assert_eq!(&decoded[120..], &source[100..164]);
}

#[test]
fn selection_sweep_matches_direct_copy_for_random_mappings() {
    let source = sample(10_000);
    let encoded = Encoder::new()
        .element_size(1)
        .shuffle(Shuffle::None)
        .block_lengths([332, 2048, 8, 4096, 776, 2740])
        .encode(&source)
        .unwrap();
    let decoder = Decoder::from_encoded(&encoded).unwrap();
    let mut workspace = DecodeWorkspace::new();
    let mut state = 0xD1B5_4A32_D192_ED03u64;

    for case in 0..128 {
        let mut next = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state
        };
        let mapping_count = (next() as usize % 20) + 1;
        let mut mappings = Vec::with_capacity(mapping_count);
        let mut copies = Vec::with_capacity(mapping_count);
        let mut output_len = next() as usize % 7;
        for _ in 0..mapping_count {
            output_len += next() as usize % 7;
            let length = next() as usize % 201;
            let source_start = next() as usize % (source.len() - length + 1);
            mappings
                .push(ByteMapping::new(source_start..source_start + length, output_len).unwrap());
            copies.push((source_start..source_start + length, output_len));
            output_len += length;
        }
        output_len += next() as usize % 7;
        if case % 2 == 1 {
            mappings.reverse();
        }
        let selection = ByteSelection::new(mappings, output_len).unwrap();
        let mut expected = vec![0; output_len];
        for (source_range, destination_start) in copies {
            expected[destination_start..destination_start + source_range.len()]
                .copy_from_slice(&source[source_range]);
        }

        assert_eq!(
            decoder.decode_selection(&encoded, &selection).unwrap(),
            expected
        );
        let mut reused = vec![0xA5; output_len];
        decoder
            .decode_selection_into(&encoded, &selection, &mut reused, &mut workspace)
            .unwrap();
        assert_eq!(reused, expected);
    }
}

#[test]
fn every_public_codec_supports_variable_blocks() {
    let source = sample(10_000);
    for codec in [Codec::BloscLz, Codec::Lz4, Codec::Zlib, Codec::Zstd] {
        let encoded = Encoder::new()
            .codec(codec)
            .shuffle(Shuffle::Bytes)
            .element_size(4)
            .block_lengths([332, 2048, 8, 4096, 776, 2740])
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
