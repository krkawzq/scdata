# dyn-blosc

`dyn-blosc` is a pure-Rust encoder and decoder with one unified API for two
wire formats:

- **Blosc1**: the standard 16-byte Blosc1 header, fixed decoded block size, and
  4-byte block-offset index. Chunks interoperate with `c-blosc`.
- **DynBlosc**: an independent format with the same 16-byte header length and
  8-byte index entries, allowing every block to have a different decoded length.

Decoders inspect the first header byte and dispatch to the corresponding
header, metadata, and index parser. Because both formats use `HEADER_LEN`
(`16`) bytes, callers can always read a fixed prefix first and parse with
`Header::from_bytes` without a separate version probe for the fetch size.
Encoders default to DynBlosc; select Blosc1 explicitly when standard
interoperability is required.

[`Encoder`] and [`Decoder`] are schema + tool objects: they hold settings or a
validated header/index, never the payload itself. Callers supply bytes to each
encode/decode method.

```rust
use std::mem::size_of;

use dyn_blosc::{BloscVersion, Codec, DecodeLimits, Decoder, Encoder, Header, HEADER_LEN, Shuffle};

let source = (0..4096_u32)
    .flat_map(u32::to_le_bytes)
    .collect::<Vec<_>>();

let encoded = Encoder::new()
    .version(BloscVersion::Blosc1)
    .codec(Codec::Zstd)
    .compression_level(5)
    .shuffle(Shuffle::Bytes)
    .element_size(size_of::<u32>())
    .block_size(16 * 1024)
    .encode(&source)?;

// Fixed-size header read: always fetch HEADER_LEN bytes, then parse.
let mut header_buf = [0u8; HEADER_LEN];
header_buf.copy_from_slice(&encoded[..HEADER_LEN]);
let header = Header::from_bytes(&header_buf)?;
assert_eq!(header.decoded_size(), source.len());

let limits = DecodeLimits::unlimited()
    .maximum_decoded_size(64 * 1024 * 1024)
    .maximum_block_size(1024 * 1024)
    .maximum_block_count(4096);
let decoder = Decoder::from_encoded_with_limits(&encoded, limits)?;
assert_eq!(decoder.decode(&encoded)?, source);
assert_eq!(decoder.decode_items(&encoded, 10..20)?, source[40..80]);

// Fine-grained path: inspect block ranges from the prefix, load one block,
// then decode into a caller-owned buffer.
let prefix_len = Decoder::index_prefix_len(&encoded)?;
let schema = Decoder::from_prefix_with_limits(&encoded[..prefix_len], limits)?;
let range = schema.block(0).expect("block 0 exists");
let mut out = vec![0; range.decoded_len()];
let mut workspace = dyn_blosc::DecodeWorkspace::new();
schema.decode_block_into(
    0,
    &encoded[range.encoded_range()],
    &mut out,
    &mut workspace,
)?;
# Ok::<(), dyn_blosc::Error>(())
```

`Decoder::from_prefix` validates the header and block index without requiring
the compressed payload to be in memory. The same schema can then decode a full
chunk, one loaded block, item ranges, or a `ByteSelection`.

Complete-chunk methods verify that the supplied header and index match the
decoder schema. Detached block methods deliberately cannot verify where a block
came from; callers are responsible for fetching the bytes described by the
corresponding `BlockRange`.

Use `from_prefix_with_limits` or `from_encoded_with_limits` for untrusted
metadata. Limits are checked before allocating the block index or decode
workspace.

## Independent block encoding

`EncodeWorkspace` and `DecodeWorkspace` are reusable scratch buffers. Low-level
block methods require them explicitly, so a caller can control allocation and
reuse memory across many blocks.

```rust
use dyn_blosc::{BlockDescriptor, EncodeWorkspace, Encoder};

let encoder = Encoder::new().element_size(4);
let sources: [&[u8]; 2] = [&source[..8192], &source[8192..]];
let mut workspace = EncodeWorkspace::new();
let mut payloads = Vec::new();
let mut descriptors = Vec::new();

for block in sources {
    let mut payload = Vec::new();
    encoder.encode_block_into(block, &mut payload, &mut workspace)?;
    descriptors.push(BlockDescriptor::new(block.len(), payload.len())?);
    payloads.push(payload);
}

// ChunkLayout contains only validated header/index metadata.
let layout = encoder.chunk_layout(&descriptors)?;
let prefix = layout.prefix()?;
let complete_chunk = layout.assemble(payloads.iter().map(Vec::as_slice))?;
assert_eq!(&complete_chunk[..prefix.len()], prefix);
# Ok::<(), dyn_blosc::Error>(())
```

Methods ending in `_into` may partially modify their destination on decode
errors. `Encoder::encode_block_into` is transactional: on error it restores the
destination `Vec` to its original length.

`format::blosc1` and `format::dyn_blosc` expose the version-specific validated
headers. The top-level `Header`, `Metadata`, `Encoder`, and `Decoder` types are
the unified interface. Blosc2 is intentionally unsupported.
