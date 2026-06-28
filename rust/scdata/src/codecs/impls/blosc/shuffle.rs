pub(super) fn unshuffle_bytes(typesize: usize, shuffled: &[u8], output: &mut [u8]) {
    debug_assert_eq!(shuffled.len(), output.len());
    match typesize {
        2 => unshuffle_2(shuffled, output),
        4 => unshuffle_4(shuffled, output),
        8 => unshuffle_8(shuffled, output),
        _ => unshuffle_generic(typesize, shuffled, output),
    }
}

fn unshuffle_2(shuffled: &[u8], output: &mut [u8]) {
    let elements = output.len() / 2;
    for idx in 0..elements {
        output[idx * 2] = shuffled[idx];
        output[idx * 2 + 1] = shuffled[elements + idx];
    }
    let rem = output.len() % 2;
    if rem != 0 {
        let output_offset = output.len() - rem;
        let shuffled_offset = shuffled.len() - rem;
        output[output_offset..].copy_from_slice(&shuffled[shuffled_offset..]);
    }
}

fn unshuffle_4(shuffled: &[u8], output: &mut [u8]) {
    let elements = output.len() / 4;
    #[cfg(target_endian = "little")]
    unsafe {
        let s0 = shuffled.as_ptr();
        let s1 = s0.add(elements);
        let s2 = s1.add(elements);
        let s3 = s2.add(elements);
        let dst = output.as_mut_ptr();
        for idx in 0..elements {
            let value = (*s0.add(idx) as u32)
                | ((*s1.add(idx) as u32) << 8)
                | ((*s2.add(idx) as u32) << 16)
                | ((*s3.add(idx) as u32) << 24);
            std::ptr::write_unaligned(dst.add(idx * 4).cast::<u32>(), value);
        }
    }
    #[cfg(not(target_endian = "little"))]
    {
        unshuffle_generic(4, shuffled, output);
    }
    let rem = output.len() % 4;
    if rem != 0 {
        let output_offset = output.len() - rem;
        let shuffled_offset = shuffled.len() - rem;
        output[output_offset..].copy_from_slice(&shuffled[shuffled_offset..]);
    }
}

fn unshuffle_8(shuffled: &[u8], output: &mut [u8]) {
    let elements = output.len() / 8;
    for idx in 0..elements {
        let offset = idx * 8;
        for byte in 0..8 {
            output[offset + byte] = shuffled[byte * elements + idx];
        }
    }
    let rem = output.len() % 8;
    if rem != 0 {
        let output_offset = output.len() - rem;
        let shuffled_offset = shuffled.len() - rem;
        output[output_offset..].copy_from_slice(&shuffled[shuffled_offset..]);
    }
}

fn unshuffle_generic(typesize: usize, shuffled: &[u8], output: &mut [u8]) {
    let elements = output.len() / typesize;
    for idx in 0..elements {
        let offset = idx * typesize;
        for byte in 0..typesize {
            output[offset + byte] = shuffled[byte * elements + idx];
        }
    }
    let rem = output.len() % typesize;
    if rem != 0 {
        let output_offset = output.len() - rem;
        let shuffled_offset = shuffled.len() - rem;
        output[output_offset..].copy_from_slice(&shuffled[shuffled_offset..]);
    }
}
