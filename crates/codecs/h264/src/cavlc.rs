//! CAVLC residual blocks, clause 9.2.

use cutlass_decode::CodecError;

use crate::bits::Bits;
use crate::tables::{COEFF_CHROMA_DC, COEFF_NC01, COEFF_NC8, COEFF_NC23, COEFF_NC47, RUN, ZEROS};
use crate::{bits, tables::Vlc};

pub fn coeff_token(bits: &mut Bits<'_>, nc: i32) -> Result<(u8, u8), CodecError> {
    let table: &[Vlc] = if nc < 2 {
        COEFF_NC01
    } else if nc < 4 {
        COEFF_NC23
    } else if nc < 8 {
        COEFF_NC47
    } else {
        COEFF_NC8
    };
    let code = bits::vlc(bits, table)?;
    Ok((code.a, code.b))
}

pub fn chroma_dc_token(bits: &mut Bits<'_>) -> Result<(u8, u8), CodecError> {
    let code = bits::vlc(bits, COEFF_CHROMA_DC)?;
    Ok((code.a, code.b))
}

/// One residual block in scan order. `max_coeff` is 15 when the DC is elsewhere, else 16.
pub fn residual(bits: &mut Bits<'_>, nc: i32, max_coeff: usize) -> Result<Vec<i32>, CodecError> {
    let (total, trailing) = if nc < 0 {
        chroma_dc_token(bits)?
    } else {
        coeff_token(bits, nc)?
    };
    let total = usize::from(total);
    let trailing = usize::from(trailing);
    if total > max_coeff || trailing > total {
        return Err(bits.malformed("coeff token"));
    }
    let mut levels = vec![0i32; total];
    let mut suffix = if total > 10 && trailing < 3 { 1 } else { 0 };
    for (i, slot) in levels.iter_mut().enumerate() {
        if i < trailing {
            *slot = if bits.bit()? { -1 } else { 1 };
            continue;
        }
        *slot = level(bits, &mut suffix, i == trailing && trailing < 3)?;
        if suffix == 0 {
            suffix = 1;
        }
        let threshold = 3u32 << (suffix - 1);
        if slot.unsigned_abs() > threshold && suffix < 6 {
            suffix += 1;
        }
    }
    let mut runs = vec![0usize; total];
    if total > 0 && total < max_coeff {
        let zeros = total_zeros(bits, total, nc < 0)?;
        let mut left = zeros;
        for slot in runs.iter_mut().take(total.saturating_sub(1)) {
            if left == 0 {
                break;
            }
            let run = run_before(bits, left)?;
            if run > left {
                return Err(bits.malformed("run before"));
            }
            *slot = run;
            left -= run;
        }
        runs[total - 1] = left;
    }
    let mut out = vec![0i32; max_coeff];
    let mut coeff_num = -1isize;
    for i in (0..total).rev() {
        coeff_num += runs[i] as isize + 1;
        if coeff_num < 0 || coeff_num as usize >= max_coeff {
            return Err(bits.malformed("coeff level"));
        }
        out[coeff_num as usize] = levels[i];
    }
    Ok(out)
}

fn level(bits: &mut Bits<'_>, suffix: &mut u32, bump: bool) -> Result<i32, CodecError> {
    let mut prefix = 0u32;
    while !bits.bit()? {
        prefix += 1;
        if prefix > 20 {
            return Err(bits.malformed("level prefix"));
        }
    }
    let suffix_size = if prefix == 14 && *suffix == 0 {
        4
    } else if prefix >= 15 {
        prefix - 3
    } else {
        *suffix
    };
    let mut code = prefix.min(15) << *suffix;
    if (suffix_size > 0 || prefix >= 14) && suffix_size > 0 {
        code += bits.u(suffix_size)?;
    }
    if prefix >= 15 && *suffix == 0 {
        code += 15;
    }
    if prefix >= 16 {
        code += (1 << (prefix - 3)) - 4096;
    }
    if bump {
        code += 2;
    }
    if code & 1 == 0 {
        Ok(((code + 2) >> 1) as i32)
    } else {
        Ok(-(((code + 1) >> 1) as i32))
    }
}

fn total_zeros(bits: &mut Bits<'_>, total: usize, chroma_dc: bool) -> Result<usize, CodecError> {
    if chroma_dc {
        let table = crate::tables::CHROMA_ZEROS[total - 1];
        return Ok(usize::from(bits::vlc(bits, table)?.a));
    }
    let table = ZEROS[total - 1];
    Ok(usize::from(bits::vlc(bits, table)?.a))
}

fn run_before(bits: &mut Bits<'_>, zeros_left: usize) -> Result<usize, CodecError> {
    let index = zeros_left.min(7) - 1;
    Ok(usize::from(bits::vlc(bits, RUN[index])?.a))
}

#[cfg(test)]
mod tests {
    use crate::bits::Bits;

    use super::residual;

    #[test]
    fn residual_after_145_bits_matches_a_fresh_reader() {
        let data = [
            0x88, 0x84, 0x27, 0xc4, 0x22, 0x2b, 0x06, 0x04, 0x21, 0xa7, 0x01, 0x0a, 0x36, 0x60,
            0x01, 0xeb, 0x50, 0xd6, 0x0d, 0x70, 0x02, 0x62, 0x78, 0x67, 0x42, 0x09, 0x49, 0xc2,
        ];
        let mut bits = Bits::new(&data);
        for _ in 0..145 {
            bits.bit().unwrap();
        }
        assert_eq!(bits.pos(), 145);
        let coeffs = residual(&mut bits, 8, 16).unwrap();
        let nonzero = coeffs.iter().copied().filter(|value| *value != 0).count();
        assert_eq!(nonzero, 2);
    }
}
