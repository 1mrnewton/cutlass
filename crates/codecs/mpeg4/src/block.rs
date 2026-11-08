//! Texture coefficients: run-level decoding (Section 7.4.1) and inverse
//! quantisation (Section 7.4.4).

use cutlass_decode::CodecError;

use crate::bits::BitReader;
use crate::tables::{CoefTable, tables};

/// How coefficients of one block are coded.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Coding {
    /// Section 6.3.8 VLCs with the three escape modes.
    Vlc,
    /// The reversible VLCs of data partitioned VOPs.
    Reversible,
    /// The H.263 escape of short-header pictures.
    Short,
}

/// Reads one block's coefficients as quantised levels into `levels`
/// (raster order), placing the first at scan position `start`.
pub(crate) fn read_coefs(
    r: &mut BitReader<'_>,
    table: &CoefTable,
    coding: Coding,
    scan: &[u8; 64],
    start: usize,
    levels: &mut [i32; 64],
) -> Result<(), CodecError> {
    let escape = table.codes.len() as u16;
    let mut i = start;
    loop {
        let Some(symbol) = table.vlc.read(r) else {
            return Err(r.malformed("texture coefficient"));
        };
        let (last, run, level) = if symbol != escape {
            let Some(c) = table.codes.get(usize::from(symbol)) else {
                return Err(r.malformed("texture coefficient"));
            };
            (c.last, usize::from(c.run), signed(r, i32::from(c.level)))
        } else {
            match coding {
                Coding::Short => {
                    let last = r.bit();
                    let run = r.read(6) as usize;
                    let level = r.read_signed(8);
                    if level == 0 || level == -128 {
                        return Err(r.malformed("short header escape level"));
                    }
                    (last, run, level)
                }
                Coding::Reversible => {
                    r.marker("reversible escape")?;
                    let last = r.bit();
                    let run = r.read(6) as usize;
                    r.marker("reversible escape")?;
                    let level = r.read(11) as i32;
                    if r.read(5) != 0b10000 {
                        return Err(r.malformed("reversible escape tail"));
                    }
                    let level = if r.bit() { -level } else { level };
                    if level == 0 {
                        return Err(r.malformed("reversible escape level"));
                    }
                    (last, run, level)
                }
                Coding::Vlc => {
                    if !r.bit() {
                        // Level offset by the largest the code table holds.
                        let (last, run, level) = plain(r, table, escape)?;
                        let extra = i32::from(table.max_level[usize::from(last)][run.min(63)]);
                        (last, run, signed(r, level + extra))
                    } else if !r.bit() {
                        // Run offset by the longest the code table holds.
                        let (last, run, level) = plain(r, table, escape)?;
                        let index = (level as usize).min(63);
                        let extra = usize::from(table.max_run[usize::from(last)][index]) + 1;
                        (last, run + extra, signed(r, level))
                    } else {
                        let last = r.bit();
                        let run = r.read(6) as usize;
                        r.marker("escape level")?;
                        let level = r.read_signed(12);
                        r.marker("escape level")?;
                        (last, run, level)
                    }
                }
            }
        };
        i += run;
        let Some(&pos) = scan.get(i) else {
            return Err(r.malformed("coefficient past the block"));
        };
        if let Some(slot) = levels.get_mut(usize::from(pos)) {
            *slot = level;
        }
        if last {
            break;
        }
        i += 1;
        if r.overrun() {
            return Err(r.malformed("texture past the end"));
        }
    }
    r.check("texture")
}

/// A non-escape code and its unsigned level.
fn plain(
    r: &mut BitReader<'_>,
    table: &CoefTable,
    escape: u16,
) -> Result<(bool, usize, i32), CodecError> {
    match table.vlc.read(r) {
        Some(symbol) if symbol != escape => match table.codes.get(usize::from(symbol)) {
            Some(c) => Ok((c.last, usize::from(c.run), i32::from(c.level))),
            None => Err(r.malformed("escape code")),
        },
        _ => Err(r.malformed("escape code")),
    }
}

fn signed(r: &mut BitReader<'_>, level: i32) -> i32 {
    if r.bit() { -level } else { level }
}

/// Reads `dct_dc_size` and `dct_dc_differential` of an intra block.
pub(crate) fn read_dc(r: &mut BitReader<'_>, chroma: bool) -> Result<i32, CodecError> {
    let t = tables();
    let vlc = if chroma { &t.dc_chroma } else { &t.dc_luma };
    let Some(size) = vlc.read(r) else {
        return Err(r.malformed("dct_dc_size"));
    };
    let size = u32::from(size);
    if size == 0 {
        return Ok(0);
    }
    let v = r.read_xbits(size);
    if size > 8 {
        r.marker("dct_dc_differential")?;
    }
    Ok(v)
}

/// The inverse quantiser of a VOL.
#[derive(Clone, Copy)]
pub(crate) enum Quant<'a> {
    /// Second method (H.263).
    H263,
    /// First method, with raster-order weighting matrices.
    Mpeg {
        intra: &'a [u8; 64],
        inter: &'a [u8; 64],
    },
}

#[inline]
fn saturate(v: i32) -> i16 {
    v.clamp(-2048, 2047) as i16
}

/// Inverse quantises an intra block: `dc` is the reconstructed DC
/// coefficient, `levels` the quantised AC levels in raster order.
pub(crate) fn dequant_intra(levels: &[i32; 64], dc: i32, q: i32, quant: Quant<'_>) -> [i16; 64] {
    let mut out = [0i16; 64];
    out[0] = saturate(dc);
    match quant {
        Quant::H263 => {
            let (mul, add) = (2 * q, (q - 1) | 1);
            for (o, &l) in out.iter_mut().zip(levels).skip(1) {
                if l != 0 {
                    *o = saturate(l.signum() * (l.abs() * mul + add));
                }
            }
        }
        Quant::Mpeg { intra, .. } => {
            let mut sum = i32::from(out[0]);
            for ((o, &l), &w) in out.iter_mut().zip(levels).zip(intra).skip(1) {
                if l != 0 {
                    let v = saturate(l.signum() * ((l.abs() * 2 * q * i32::from(w)) >> 4));
                    sum += i32::from(v);
                    *o = v;
                }
            }
            if sum & 1 == 0 {
                out[63] ^= 1;
            }
        }
    }
    out
}

/// Inverse quantises an inter block from levels in raster order.
pub(crate) fn dequant_inter(levels: &[i32; 64], q: i32, quant: Quant<'_>) -> [i16; 64] {
    let mut out = [0i16; 64];
    match quant {
        Quant::H263 => {
            let (mul, add) = (2 * q, (q - 1) | 1);
            for (o, &l) in out.iter_mut().zip(levels) {
                if l != 0 {
                    *o = saturate(l.signum() * (l.abs() * mul + add));
                }
            }
        }
        Quant::Mpeg { inter, .. } => {
            let mut sum = 0i32;
            for ((o, &l), &w) in out.iter_mut().zip(levels).zip(inter) {
                if l != 0 {
                    let v = saturate(l.signum() * (((2 * l.abs() + 1) * q * i32::from(w)) >> 4));
                    sum += i32::from(v);
                    *o = v;
                }
            }
            if sum & 1 == 0 {
                out[63] ^= 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{Coding, Quant, dequant_inter, dequant_intra, read_coefs, read_dc};
    use crate::bits::{BitReader, BitWriter};
    use crate::tables::{DEFAULT_INTER_MATRIX, DEFAULT_INTRA_MATRIX, TCOEF_ESCAPE, ZIGZAG, tables};

    #[test]
    fn reads_codes_and_all_three_escapes() {
        let t = &tables().inter;
        let mut w = BitWriter::new();
        // (0, 0, 1) then sign +.
        let (len, code) = t.code(false, 0, 1).expect("code");
        w.put(u32::from(len), u32::from(code));
        w.put(1, 0);
        // Escape 1: level 1 + max_level[0][1] at run 1, negative.
        w.put(u32::from(TCOEF_ESCAPE.0), u32::from(TCOEF_ESCAPE.1));
        w.put(1, 0);
        let (len, code) = t.code(false, 1, 1).expect("code");
        w.put(u32::from(len), u32::from(code));
        w.put(1, 1);
        // Escape 2: run 0 + max_run[0][1] + 1 at level 1.
        w.put(u32::from(TCOEF_ESCAPE.0), u32::from(TCOEF_ESCAPE.1));
        w.put(2, 0b10);
        let (len, code) = t.code(false, 0, 1).expect("code");
        w.put(u32::from(len), u32::from(code));
        w.put(1, 0);
        // Escape 3: last, run 0, level -300.
        w.put(u32::from(TCOEF_ESCAPE.0), u32::from(TCOEF_ESCAPE.1));
        w.put(2, 0b11);
        w.put(1, 1);
        w.put(6, 0);
        w.put(1, 1);
        w.put_signed(12, -300);
        w.put(1, 1);
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        let mut levels = [0i32; 64];
        read_coefs(&mut r, t, Coding::Vlc, &ZIGZAG, 0, &mut levels).expect("block");
        let run2 = usize::from(t.max_run[0][1]) + 1;
        let level1 = 1 + i32::from(t.max_level[0][1]);
        let mut want = [0i32; 64];
        want[usize::from(ZIGZAG[0])] = 1;
        want[usize::from(ZIGZAG[2])] = -level1;
        want[usize::from(ZIGZAG[3 + run2])] = 1;
        want[usize::from(ZIGZAG[4 + run2])] = -300;
        assert_eq!(levels, want);
    }

    #[test]
    fn refuses_a_run_past_the_block_and_a_truncated_block() {
        let t = &tables().inter;
        let mut w = BitWriter::new();
        // Runs of 26 land on positions 26 and 53; the third passes 63.
        let (len, code) = t.code(false, 26, 1).expect("code");
        for _ in 0..3 {
            w.put(u32::from(len), u32::from(code));
            w.put(1, 0);
        }
        let bytes = w.finish();
        let mut levels = [0i32; 64];
        let mut r = BitReader::new(&bytes);
        assert!(read_coefs(&mut r, t, Coding::Vlc, &ZIGZAG, 0, &mut levels).is_err());
        let mut r = BitReader::new(&bytes[..1]);
        assert!(read_coefs(&mut r, t, Coding::Vlc, &ZIGZAG, 0, &mut levels).is_err());
    }

    #[test]
    fn reads_dc_differentials() {
        let mut w = BitWriter::new();
        // Luma size 3 is '010' (Table B-13), then 101 (5), and -5 as 010.
        for bits in [0b101, 0b010] {
            w.put(3, 0b010);
            w.put(3, bits);
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert_eq!(read_dc(&mut r, false).expect("dc"), 5);
        assert_eq!(read_dc(&mut r, false).expect("dc"), -5);
    }

    #[test]
    fn inverse_quantises_both_methods() {
        let mut levels = [0i32; 64];
        levels[1] = 3;
        levels[8] = -2;
        // H.263: 2q|l| + q (odd q), or q - 1 (even q).
        let out = dequant_inter(&levels, 5, Quant::H263);
        assert_eq!((out[1], out[8]), (35, -25));
        let out = dequant_inter(&levels, 4, Quant::H263);
        assert_eq!((out[1], out[8]), (27, -19));
        let out = dequant_intra(&levels, 1024, 4, Quant::H263);
        assert_eq!((out[0], out[1], out[8]), (1024, 27, -19));
        // MPEG: (2|l| + 1) q w / 16 for inter, 2 |l| q w / 16 for intra,
        // then the sum's parity toggles coefficient 63.
        let quant = Quant::Mpeg {
            intra: &DEFAULT_INTRA_MATRIX,
            inter: &DEFAULT_INTER_MATRIX,
        };
        let out = dequant_inter(&levels, 4, quant);
        let w1 = i32::from(DEFAULT_INTER_MATRIX[1]);
        let w8 = i32::from(DEFAULT_INTER_MATRIX[8]);
        assert_eq!(i32::from(out[1]), (7 * 4 * w1) >> 4);
        assert_eq!(i32::from(out[8]), -((5 * 4 * w8) >> 4));
        let sum: i32 = out.iter().map(|&v| i32::from(v)).sum();
        assert_eq!(sum & 1, 1, "mismatch control leaves an odd sum");
        let out = dequant_intra(&levels, 800, 4, quant);
        let sum: i32 = out.iter().map(|&v| i32::from(v)).sum();
        assert_eq!(sum & 1, 1);
        // Saturation.
        levels[1] = 2000;
        let out = dequant_inter(&levels, 31, Quant::H263);
        assert_eq!(out[1], 2047);
    }
}
