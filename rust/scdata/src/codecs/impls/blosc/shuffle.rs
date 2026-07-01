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
    #[cfg(target_endian = "little")]
    unsafe {
        let s0 = shuffled.as_ptr();
        let s1 = s0.add(elements);
        let dst = output.as_mut_ptr();
        for idx in 0..elements {
            let value = (*s0.add(idx) as u16) | ((*s1.add(idx) as u16) << 8);
            std::ptr::write_unaligned(dst.add(idx * 2).cast::<u16>(), value);
        }
    }
    #[cfg(not(target_endian = "little"))]
    {
        unshuffle_generic(2, shuffled, output);
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
    #[cfg(target_endian = "little")]
    unsafe {
        let s0 = shuffled.as_ptr();
        let s1 = s0.add(elements);
        let s2 = s1.add(elements);
        let s3 = s2.add(elements);
        let s4 = s3.add(elements);
        let s5 = s4.add(elements);
        let s6 = s5.add(elements);
        let s7 = s6.add(elements);
        let dst = output.as_mut_ptr();
        for idx in 0..elements {
            let value = (*s0.add(idx) as u64)
                | ((*s1.add(idx) as u64) << 8)
                | ((*s2.add(idx) as u64) << 16)
                | ((*s3.add(idx) as u64) << 24)
                | ((*s4.add(idx) as u64) << 32)
                | ((*s5.add(idx) as u64) << 40)
                | ((*s6.add(idx) as u64) << 48)
                | ((*s7.add(idx) as u64) << 56);
            std::ptr::write_unaligned(dst.add(idx * 8).cast::<u64>(), value);
        }
    }
    #[cfg(not(target_endian = "little"))]
    {
        unshuffle_generic(8, shuffled, output);
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
