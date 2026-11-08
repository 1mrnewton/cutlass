//! Adaptive hybrid transform mantissas (A/52:2018 Annex E clause 3.4): the
//! vector and gain adaptive quantizers, and the six-point inverse DCT that
//! spreads one frame of AHT mantissas over its six blocks.

use std::f32::consts::{FRAC_1_SQRT_2, PI, SQRT_2};

use cutlass_decode::CodecError;

use crate::bits::BitReader;
use crate::tables::vq::{
    VQ_HEBAP1, VQ_HEBAP2, VQ_HEBAP3, VQ_HEBAP4, VQ_HEBAP5, VQ_HEBAP6, VQ_HEBAP7,
};
use crate::tables::{GAQ_A1, GAQ_A2, GAQ_A4, GAQ_B2, GAQ_B4, HEBAP_BITS};

/// One bin's six mantissas, one per block, before the exponent.
pub(crate) type Premantissas = [[f32; 6]; 256];

/// The six mantissas of VQ index `index` at `hebap` 1 to 7.
fn vq(hebap: u8, index: usize) -> Option<&'static [i16; 6]> {
    match hebap {
        1 => VQ_HEBAP1.get(index),
        2 => VQ_HEBAP2.get(index),
        3 => VQ_HEBAP3.get(index),
        4 => VQ_HEBAP4.get(index),
        5 => VQ_HEBAP5.get(index),
        6 => VQ_HEBAP6.get(index),
        7 => VQ_HEBAP7.get(index),
        _ => None,
    }
}

fn fraction(value: u16) -> f32 {
    f32::from(value as i16) / 32768.0
}

/// Read one channel's AHT mantissas (`chgaqmod`, the gains, and the
/// mantissas of `hebap` above zero) for bins `start..end`, then invert the
/// DCT. Bins with `hebap` 0 come back as zeros and `zero[bin]` set, so the
/// caller can dither them block by block.
pub(crate) fn read(
    r: &mut BitReader<'_>,
    hebap: &[u8; 256],
    start: usize,
    end: usize,
    out: &mut Premantissas,
    zero: &mut [bool; 256],
) -> Result<(), CodecError> {
    let gaqmod = r.read(2, "gaqmod")?;
    let end_bap = if gaqmod < 2 { 12 } else { 17 };
    let gained = |bap: u8| gaqmod != 0 && (8..end_bap).contains(&bap);

    // log2 of each gain, in ascending bin order of the gained bins.
    let mut gains = [0u8; 256];
    let mut count = 0;
    match gaqmod {
        1 | 2 => {
            for &bap in &hebap[start..end] {
                if gained(bap) {
                    gains[count] = (r.read(1, "gaqgain")? as u8) << (gaqmod - 1);
                    count += 1;
                }
            }
        }
        3 => {
            let wanted = hebap[start..end].iter().filter(|&&b| gained(b)).count();
            while count < wanted {
                let at = r.pos() as u64;
                let code = r.read(5, "gaqgain")?;
                if code > 26 {
                    return Err(CodecError::Malformed {
                        at_bit: at,
                        what: "gaqgain group",
                    });
                }
                for m in [code / 9, (code % 9) / 3, code % 3] {
                    if count < wanted {
                        gains[count] = m as u8;
                        count += 1;
                    }
                }
            }
        }
        _ => {}
    }

    let mut next_gain = 0;
    for bin in start..end {
        let bap = hebap[bin];
        let slot = &mut out[bin];
        zero[bin] = bap == 0;
        if bap == 0 {
            *slot = [0.0; 6];
            continue;
        }
        let bits = HEBAP_BITS[usize::from(bap.min(19))];
        if bap < 8 {
            let at = r.pos() as u64;
            let index = r.read(bits, "vq index")? as usize;
            let Some(vector) = vq(bap, index) else {
                return Err(CodecError::Malformed {
                    at_bit: at,
                    what: "vq index",
                });
            };
            for (value, &q) in slot.iter_mut().zip(vector) {
                *value = f32::from(q) / 32768.0;
            }
        } else {
            let log_gain = if gained(bap) {
                let g = gains[next_gain];
                next_gain += 1;
                g
            } else {
                0
            };
            read_gaq(r, bap, bits, log_gain, slot)?;
        }
        idct6(slot);
    }
    Ok(())
}

/// Six GAQ mantissas of one bin (Table E3.5 and E3.6).
fn read_gaq(
    r: &mut BitReader<'_>,
    hebap: u8,
    bits: u32,
    log_gain: u8,
    slot: &mut [f32; 6],
) -> Result<(), CodecError> {
    let row = usize::from(hebap - 8);
    for value in slot.iter_mut() {
        if log_gain == 0 {
            let q = r.signed(bits, "gaq mantissa")?;
            let x = q as f32 / (1u32 << (bits - 1)) as f32;
            *value = x + fraction(GAQ_A1[row] as u16) * x;
            continue;
        }
        let small = bits - u32::from(log_gain);
        let q = r.signed(small, "gaq mantissa")?;
        if q != -(1 << (small - 1)) {
            *value = q as f32 / (1u32 << (bits - 1)) as f32;
            continue;
        }
        let large = bits - (2 - u32::from(log_gain));
        let q = r.signed(large, "gaq large mantissa")?;
        let x = q as f32 / (1u32 << (large - 1)) as f32;
        let (a, b) = if log_gain == 1 {
            (GAQ_A2[row], if x >= 0.0 { 0x4000 } else { GAQ_B2[row] })
        } else {
            (GAQ_A4[row], if x >= 0.0 { 0x2000 } else { GAQ_B4[row] })
        };
        *value = x + fraction(a) * x + fraction(b);
    }
    Ok(())
}

/// `C(m) = sqrt(2) * sum_j R(j) X(j) cos(pi (2m + 1) j / 12)`, with
/// `R(0) = 1/sqrt(2)` and `R(j) = 1` above it (clause 3.4.5).
pub(crate) fn idct6(x: &mut [f32; 6]) {
    let mut out = [0.0f32; 6];
    for (m, value) in out.iter_mut().enumerate() {
        let mut sum = x[0] * FRAC_1_SQRT_2;
        for (j, &xj) in x.iter().enumerate().skip(1) {
            sum += xj * (PI * ((2 * m + 1) * j) as f32 / 12.0).cos();
        }
        *value = SQRT_2 * sum;
    }
    *x = out;
}

/// The forward DCT an AHT encoder would use: the transpose of `idct6`
/// scaled so that `idct6(dct6(x)) == x`.
#[cfg(test)]
fn dct6(c: &mut [f32; 6]) {
    let mut out = [0.0f32; 6];
    for (j, value) in out.iter_mut().enumerate() {
        let r = if j == 0 { FRAC_1_SQRT_2 } else { 1.0 };
        let mut sum = 0.0;
        for (m, &cm) in c.iter().enumerate() {
            sum += cm * (PI * ((2 * m + 1) * j) as f32 / 12.0).cos();
        }
        *value = SQRT_2 * r * sum / 6.0;
    }
    *c = out;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;

    #[test]
    fn dct_pair_reconstructs() {
        let original = [0.3, -0.2, 0.9, -0.7, 0.05, 0.4];
        let mut x = original;
        dct6(&mut x);
        idct6(&mut x);
        for (a, b) in x.iter().zip(original) {
            assert!((a - b).abs() < 1e-5, "{a} {b}");
        }
    }

    #[test]
    fn idct_of_dc_is_flat() {
        let mut x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        idct6(&mut x);
        for value in x {
            assert!((value - 1.0).abs() < 1e-6);
        }
    }

    fn hebap_row(values: &[(usize, u8)]) -> [u8; 256] {
        let mut hebap = [0u8; 256];
        for &(bin, bap) in values {
            hebap[bin] = bap;
        }
        hebap
    }

    #[test]
    fn vq_reads_one_index_for_six_blocks() {
        let mut w = BitWriter::with_capacity(8);
        w.put(2, 0); // gaqmod 0
        w.put(2, 3); // hebap 1, index 3
        let bytes = w.finish();
        let hebap = hebap_row(&[(0, 1)]);
        let mut out = [[0.0; 6]; 256];
        let mut zero = [false; 256];
        read(
            &mut BitReader::new(&bytes),
            &hebap,
            0,
            2,
            &mut out,
            &mut zero,
        )
        .unwrap();
        let mut expected = VQ_HEBAP1[3].map(|q| f32::from(q) / 32768.0);
        idct6(&mut expected);
        assert_eq!(out[0], expected);
        assert!(zero[1] && !zero[0]);
    }

    #[test]
    fn gaq_large_mantissa_follows_the_tag() {
        // hebap 8 is 3 bits; gain 2 makes small mantissas 2 bits and the
        // tag "10"; the large mantissa is then 2 bits.
        let mut w = BitWriter::with_capacity(8);
        w.put(2, 1); // gaqmod 1
        w.put(1, 1); // Gk = 2
        w.signed(2, 1); // small: 1/4
        w.signed(2, -2); // tag
        w.signed(2, 1); // large x = 0.5
        for _ in 0..4 {
            w.signed(2, 0);
        }
        let bytes = w.finish();
        let hebap = hebap_row(&[(0, 8)]);
        let mut out = [[0.0; 6]; 256];
        let mut zero = [false; 256];
        let mut r = BitReader::new(&bytes);
        read(&mut r, &hebap, 0, 1, &mut out, &mut zero).unwrap();
        assert_eq!(r.pos(), 2 + 1 + 2 + 2 + 2 + 8);
        let mut pre = out[0];
        dct6(&mut pre);
        assert!((pre[0] - 0.25).abs() < 1e-5);
        let large = 0.5 + fraction(0xd555) * 0.5 + 0.5;
        assert!((pre[1] - large).abs() < 1e-5, "{}", pre[1]);
    }

    #[test]
    fn gaq_group_code_above_26_is_malformed() {
        let mut w = BitWriter::with_capacity(8);
        w.put(2, 3);
        w.put(5, 27);
        w.put(16, 0);
        let bytes = w.finish();
        let hebap = hebap_row(&[(0, 9)]);
        let mut out = [[0.0; 6]; 256];
        let mut zero = [false; 256];
        let error = read(
            &mut BitReader::new(&bytes),
            &hebap,
            0,
            1,
            &mut out,
            &mut zero,
        );
        assert!(matches!(error, Err(CodecError::Malformed { .. })));
    }
}
