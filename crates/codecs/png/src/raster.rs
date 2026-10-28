//! Scanline packing and filter method 0.
//!
//! Filters run on bytes. A pass of Adam7 is its own image: the previous
//! scanline is the previous row of that pass, and an empty pass contributes
//! no filter byte.

use cutlass_decode::BitDepth;

pub(crate) const PASS_X0: [u32; 7] = [0, 4, 0, 2, 0, 1, 0];
pub(crate) const PASS_Y0: [u32; 7] = [0, 0, 4, 0, 2, 0, 1];
pub(crate) const PASS_DX: [u32; 7] = [8, 8, 4, 4, 2, 2, 1];
pub(crate) const PASS_DY: [u32; 7] = [8, 8, 8, 4, 4, 2, 2];

pub(crate) struct Pass {
    pub x0: u32,
    pub dx: u32,
    pub y0: u32,
    pub dy: u32,
    pub cols: u32,
    pub rows: u32,
}

pub(crate) fn full_pass(width: u32, height: u32) -> Pass {
    Pass {
        x0: 0,
        dx: 1,
        y0: 0,
        dy: 1,
        cols: width,
        rows: height,
    }
}

pub(crate) fn adam_pass(pass: usize, width: u32, height: u32) -> Pass {
    Pass {
        x0: PASS_X0[pass],
        dx: PASS_DX[pass],
        y0: PASS_Y0[pass],
        dy: PASS_DY[pass],
        cols: pass_len(PASS_X0[pass], PASS_DX[pass], width),
        rows: pass_len(PASS_Y0[pass], PASS_DY[pass], height),
    }
}

pub(crate) fn pass_len(origin: u32, step: u32, full: u32) -> u32 {
    if full <= origin {
        0
    } else {
        (full - origin).div_ceil(step)
    }
}

pub(crate) fn pixel_bits(channels: usize, depth: BitDepth) -> u32 {
    (channels as u32).saturating_mul(u32::from(depth.bits()))
}

pub(crate) fn bytes_per_pixel(pixel_bits: u32) -> usize {
    usize::try_from(pixel_bits.div_ceil(8)).unwrap_or(1).max(1)
}

pub(crate) fn row_bytes(width: u32, pixel_bits: u32) -> Option<usize> {
    let bits = u64::from(width).checked_mul(u64::from(pixel_bits))?;
    usize::try_from(bits.div_ceil(8)).ok()
}

pub(crate) fn filtered_len(
    width: u32,
    height: u32,
    pixel_bits: u32,
    interlace: bool,
) -> Option<usize> {
    if !interlace {
        let row = row_bytes(width, pixel_bits)?;
        return row
            .checked_add(1)?
            .checked_mul(usize::try_from(height).ok()?);
    }
    let mut total = 0usize;
    for pass in 0..7 {
        let rows = pass_len(PASS_Y0[pass], PASS_DY[pass], height);
        let cols = pass_len(PASS_X0[pass], PASS_DX[pass], width);
        if rows == 0 || cols == 0 {
            continue;
        }
        let row = row_bytes(cols, pixel_bits)?;
        let lines = usize::try_from(rows).ok()?;
        total = total.checked_add(row.checked_add(1)?.checked_mul(lines)?)?;
    }
    Some(total)
}

pub(crate) fn sample_count(width: u32, height: u32, channels: usize) -> Option<usize> {
    let pixels = u64::from(width).checked_mul(u64::from(height))?;
    usize::try_from(pixels.checked_mul(channels as u64)?).ok()
}

pub(crate) fn pixel_offset(width: u32, x: u32, y: u32, channels: usize) -> Option<usize> {
    let stride = usize::try_from(width).ok()?.checked_mul(channels)?;
    let row = usize::try_from(y).ok()?.checked_mul(stride)?;
    let col = usize::try_from(x).ok()?.checked_mul(channels)?;
    row.checked_add(col)
}

/// Reverse one filter type. `prev` is the previous reconstructed scanline.
pub(crate) fn unfilter(kind: u8, row: &mut [u8], prev: &[u8], bpp: usize) -> Result<(), ()> {
    if prev.len() != row.len() || bpp == 0 || bpp > row.len() {
        return Err(());
    }
    match kind {
        0 => Ok(()),
        1 => {
            for index in bpp..row.len() {
                row[index] = row[index].wrapping_add(row[index - bpp]);
            }
            Ok(())
        }
        2 => {
            for (byte, above) in row.iter_mut().zip(prev) {
                *byte = byte.wrapping_add(*above);
            }
            Ok(())
        }
        3 => {
            for index in 0..row.len() {
                let left = if index >= bpp { row[index - bpp] } else { 0 };
                let sum = u16::from(left) + u16::from(prev[index]);
                row[index] = row[index].wrapping_add((sum / 2) as u8);
            }
            Ok(())
        }
        4 => {
            for index in 0..row.len() {
                let left = if index >= bpp { row[index - bpp] } else { 0 };
                let above = prev[index];
                let upper_left = if index >= bpp { prev[index - bpp] } else { 0 };
                row[index] = row[index].wrapping_add(paeth(left, above, upper_left));
            }
            Ok(())
        }
        _ => Err(()),
    }
}

/// Apply one filter type. `src` is the original scanline.
pub(crate) fn filter_into(kind: u8, src: &[u8], prev: &[u8], bpp: usize, dest: &mut [u8]) {
    match kind {
        1 => {
            for index in 0..src.len() {
                let left = if index >= bpp { src[index - bpp] } else { 0 };
                dest[index] = src[index].wrapping_sub(left);
            }
        }
        2 => {
            for index in 0..src.len() {
                dest[index] = src[index].wrapping_sub(prev[index]);
            }
        }
        3 => {
            for index in 0..src.len() {
                let left = if index >= bpp { src[index - bpp] } else { 0 };
                let sum = u16::from(left) + u16::from(prev[index]);
                dest[index] = src[index].wrapping_sub((sum / 2) as u8);
            }
        }
        4 => {
            for index in 0..src.len() {
                let left = if index >= bpp { src[index - bpp] } else { 0 };
                let above = prev[index];
                let upper_left = if index >= bpp { prev[index - bpp] } else { 0 };
                dest[index] = src[index].wrapping_sub(paeth(left, above, upper_left));
            }
        }
        _ => dest.copy_from_slice(src),
    }
}

/// Paeth predictor. Ties prefer the left byte, then the byte above.
fn paeth(left: u8, above: u8, upper_left: u8) -> u8 {
    let a = i32::from(left);
    let b = i32::from(above);
    let c = i32::from(upper_left);
    let estimate = a + b - c;
    let pa = (estimate - a).abs();
    let pb = (estimate - b).abs();
    let pc = (estimate - c).abs();
    if pa <= pb && pa <= pc {
        left
    } else if pb <= pc {
        above
    } else {
        upper_left
    }
}

pub(crate) fn pack(samples: &[u16], depth: BitDepth, dest: &mut [u8]) {
    dest.fill(0);
    match depth {
        BitDepth::Sixteen => {
            for (index, sample) in samples.iter().enumerate() {
                let bytes = sample.to_be_bytes();
                dest[index * 2] = bytes[0];
                dest[index * 2 + 1] = bytes[1];
            }
        }
        BitDepth::Eight => {
            for (index, sample) in samples.iter().enumerate() {
                dest[index] = *sample as u8;
            }
        }
        BitDepth::One | BitDepth::Two | BitDepth::Four => {
            let step = u32::from(depth.bits());
            let mask = (1u16 << depth.bits()) - 1;
            let mut bit = 0u32;
            for sample in samples {
                let value = (*sample & mask) as u8;
                let byte_index = (bit / 8) as usize;
                let offset = 8 - depth.bits() - (bit % 8) as u8;
                dest[byte_index] |= value << offset;
                bit += step;
            }
        }
    }
}

pub(crate) fn unpack(packed: &[u8], depth: BitDepth, dest: &mut [u16]) {
    match depth {
        BitDepth::Sixteen => {
            for (index, sample) in dest.iter_mut().enumerate() {
                let start = index * 2;
                *sample = u16::from_be_bytes([packed[start], packed[start + 1]]);
            }
        }
        BitDepth::Eight => {
            for (sample, byte) in dest.iter_mut().zip(packed) {
                *sample = u16::from(*byte);
            }
        }
        BitDepth::One | BitDepth::Two | BitDepth::Four => {
            let step = u32::from(depth.bits());
            let mask = (1u8 << depth.bits()) - 1;
            let mut bit = 0u32;
            for sample in dest.iter_mut() {
                let byte = packed[(bit / 8) as usize];
                let offset = 8 - depth.bits() - (bit % 8) as u8;
                *sample = u16::from((byte >> offset) & mask);
                bit += step;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{filter_into, pack, paeth, unfilter, unpack};
    use cutlass_decode::BitDepth;

    #[test]
    fn paeth_ties_prefer_the_left_byte() {
        assert_eq!(paeth(5, 5, 5), 5);
        assert_eq!(paeth(0, 0, 5), 0);
        assert_eq!(paeth(1, 0, 0), 1);
    }

    #[test]
    fn sub_up_average_and_paeth_round_trip_a_row() {
        let src = [10u8, 20, 30, 40];
        let prev = [1u8, 2, 3, 4];
        for kind in 0..5 {
            let mut filtered = [0u8; 4];
            filter_into(kind, &src, &prev, 1, &mut filtered);
            let mut row = filtered;
            unfilter(kind, &mut row, &prev, 1).expect("filter");
            assert_eq!(row, src);
        }
    }

    #[test]
    fn one_bit_samples_pack_from_the_high_bit() {
        let mut packed = [0u8; 1];
        pack(&[1, 0, 1, 1], BitDepth::One, &mut packed);
        assert_eq!(packed, [0b1011_0000]);
        let mut samples = [0u16; 4];
        unpack(&packed, BitDepth::One, &mut samples);
        assert_eq!(samples, [1, 0, 1, 1]);
    }

    #[test]
    fn sixteen_bit_samples_are_big_endian() {
        let mut packed = [0u8; 4];
        pack(&[1, 0x0102], BitDepth::Sixteen, &mut packed);
        assert_eq!(packed, [0, 1, 1, 2]);
        let mut samples = [0u16; 2];
        unpack(&packed, BitDepth::Sixteen, &mut samples);
        assert_eq!(samples, [1, 0x0102]);
    }
}
