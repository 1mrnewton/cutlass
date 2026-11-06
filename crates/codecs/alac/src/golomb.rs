//! Adaptive Golomb coding of prediction residuals.
//!
//! The rules are the ones in Apple's reference `ag_dec.c` and `ag_enc.c`:
//! a running mean picks the Golomb parameter for each residual, a prefix of
//! nine ones escapes to a raw value, and a small mean switches to coding a
//! run of zeros. Arithmetic is unsigned 32-bit with wrap-around, as in the
//! reference, so a hostile stream decodes the same way the reference would.

use cutlass_decode::CodecError;

use crate::bits::{BitReader, BitSink};

/// Mean scale: the mean is kept in units of `1 / 2^QBSHIFT`.
const QBSHIFT: u32 = 9;
const QB: u32 = 1 << QBSHIFT;
const MMULSHIFT: u32 = 2;
const MDENSHIFT: u32 = QBSHIFT - MMULSHIFT - 1;
const MOFF: u32 = 1 << (MDENSHIFT - 2);
const BITOFF: i64 = 24;
/// A prefix this long is the escape to a raw value.
const MAX_PREFIX: u32 = 9;
/// Bits of a raw zero-run length.
const RUN_BITS: u32 = 16;
/// A mean above this after one value resets to it.
const MEAN_CLAMP: u32 = 0xffff;
/// Longest zero run one code carries.
const MAX_RUN: u32 = 65_535;
/// The encoder escapes a code longer than this.
const MAX_CODE_BITS: u32 = 25;

/// Adaptation parameters for one channel of one element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AgParams {
    mb0: u32,
    pb: u32,
    kb: u32,
    wb: u32,
}

impl AgParams {
    /// `mb` is the initial mean, `pb` the adaptation rate, `kb` the largest
    /// Golomb parameter. `kb` is 1 through 31; the caller checks it.
    pub(crate) fn new(mb: u32, pb: u32, kb: u32) -> Self {
        let kb = kb.min(31);
        Self {
            mb0: mb,
            pb,
            kb,
            wb: (1u32 << kb) - 1,
        }
    }
}

#[inline]
fn lg3a(value: u32) -> u32 {
    31 - value.wrapping_add(3).leading_zeros()
}

/// One Golomb code: a unary prefix, then the remainder, or the escape.
#[inline]
fn get(reader: &mut BitReader<'_>, m: u32, k: u32, escape_bits: u32) -> Result<u32, CodecError> {
    let prefix = (!reader.peek32()).leading_zeros();
    if prefix >= MAX_PREFIX {
        reader.skip(MAX_PREFIX as usize, "golomb escape")?;
        return reader.read(escape_bits, "golomb escape value");
    }
    reader.skip(prefix as usize + 1, "golomb prefix")?;
    let base = prefix.wrapping_mul(m);
    let v = reader.peek32() >> (32 - k);
    if v < 2 {
        reader.skip(k as usize - 1, "golomb remainder")?;
        Ok(base)
    } else {
        reader.skip(k as usize, "golomb remainder")?;
        Ok(base.wrapping_add(v - 1))
    }
}

/// Run-length parameter for a zero run after mean `mb`.
fn run_k(mb: u32, at_bit: u64) -> Result<u32, CodecError> {
    let k = i64::from(mb.leading_zeros()) - BITOFF + i64::from(mb.wrapping_add(MOFF) >> MDENSHIFT);
    if !(1..=16).contains(&k) {
        return Err(CodecError::Malformed {
            at_bit,
            what: "zero-run parameter",
        });
    }
    Ok(k as u32)
}

/// Decode `out.len()` residuals. `max_size` is the width of an escaped
/// value: the channel's bits after shift and mix.
pub(crate) fn decompress(
    params: AgParams,
    reader: &mut BitReader<'_>,
    out: &mut [i32],
    max_size: u32,
) -> Result<(), CodecError> {
    if params.kb == 0 {
        return Err(CodecError::Unsupported {
            what: "compressed frame with a zero Golomb limit",
        });
    }
    let num = out.len();
    let mut mb = params.mb0;
    let mut zmode = 0u32;
    let mut c = 0usize;
    while c < num {
        let k = lg3a(mb >> QBSHIFT).min(params.kb);
        let m = (1u32 << k) - 1;
        let n = get(reader, m, k, max_size)?;
        let ndecode = n.wrapping_add(zmode);
        let multiplier = -((ndecode & 1) as i32) | 1;
        out[c] = ((ndecode.wrapping_add(1) >> 1) as i32).wrapping_mul(multiplier);
        c += 1;
        mb = params
            .pb
            .wrapping_mul(ndecode)
            .wrapping_add(mb)
            .wrapping_sub(params.pb.wrapping_mul(mb) >> QBSHIFT);
        if n > MEAN_CLAMP {
            mb = MEAN_CLAMP;
        }
        zmode = 0;
        if (mb << MMULSHIFT) < QB && c < num {
            zmode = 1;
            let k = run_k(mb, reader.at_bit())?;
            let mz = ((1u32 << k) - 1) & params.wb;
            let run = get(reader, mz, k, RUN_BITS)? as usize;
            if run > num - c {
                return Err(CodecError::Malformed {
                    at_bit: reader.at_bit(),
                    what: "zero run past the frame",
                });
            }
            out[c..c + run].fill(0);
            c += run;
            if run >= MAX_RUN as usize {
                zmode = 0;
            }
            mb = 0;
        }
    }
    Ok(())
}

/// Write one code for `n`, escaping to `escape_bits` raw bits when the
/// prefix or the whole code would be too long.
#[inline]
fn put<S: BitSink>(sink: &mut S, n: u32, m: u32, k: u32, escape_bits: u32) {
    let div = n / m;
    let rem = n - div * m;
    let exact = u32::from(rem == 0);
    let bits = div.saturating_add(k + 1 - exact);
    if div >= MAX_PREFIX || bits > MAX_CODE_BITS {
        sink.put((1u32 << MAX_PREFIX) - 1, MAX_PREFIX);
        sink.put(n, escape_bits);
        return;
    }
    let value = (((1u32 << div) - 1) << (bits - div)) + rem + 1 - exact;
    sink.put(value, bits);
}

/// Encode `input` residuals into `sink`. `bit_size` is the escape width,
/// 1 through 32. Every residual must already fit `bit_size` signed bits.
pub(crate) fn compress<S: BitSink>(params: AgParams, input: &[i32], sink: &mut S, bit_size: u32) {
    let num = input.len();
    let mut mb = params.mb0;
    let mut zmode = 0u32;
    let mut c = 0usize;
    while c < num {
        let k = lg3a(mb >> QBSHIFT).min(params.kb);
        let m = (1u32 << k) - 1;
        let del = input[c];
        c += 1;
        let zig = (del.unsigned_abs() << 1).wrapping_sub(((del >> 31) & 1) as u32);
        let n = zig.wrapping_sub(zmode);
        put(sink, n, m, k, bit_size);
        mb = params
            .pb
            .wrapping_mul(zig)
            .wrapping_add(mb)
            .wrapping_sub(params.pb.wrapping_mul(mb) >> QBSHIFT);
        if n > MEAN_CLAMP {
            mb = MEAN_CLAMP;
        }
        zmode = 0;
        if (mb << MMULSHIFT) < QB && c < num {
            zmode = 1;
            let mut run = 0u32;
            while c < num && input[c] == 0 {
                c += 1;
                run += 1;
                if run >= MAX_RUN {
                    zmode = 0;
                    break;
                }
            }
            // The mean is below `QB / 4` here, so this parameter is 1 through 8.
            let k = run_k(mb, 0).unwrap_or(1);
            let mz = ((1u32 << k) - 1) & params.wb;
            put(sink, run, mz.max(1), k, RUN_BITS);
            mb = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AgParams, compress, decompress};
    use crate::bits::{BitCounter, BitReader, BitWriter};

    fn round_trip(values: &[i32], bits: u32) -> u64 {
        let params = AgParams::new(10, 40, 14);
        let mut writer = BitWriter::default();
        compress(params, values, &mut writer, bits);
        let mut counter = BitCounter::default();
        compress(params, values, &mut counter, bits);
        let written = writer.position() as u64;
        assert_eq!(counter.bits, written);
        let bytes = writer.into_bytes();
        let mut reader = BitReader::new(&bytes);
        let mut out = vec![7; values.len()];
        decompress(params, &mut reader, &mut out, bits).unwrap();
        assert_eq!(out, values);
        assert_eq!(reader.position() as u64, written);
        written
    }

    #[test]
    fn residuals_round_trip_through_every_code_path() {
        let mut values = Vec::new();
        // Small values, then silence long enough to need the run code twice,
        // then extremes that take the raw escape.
        for i in 0..400 {
            values.push((i % 7) - 3);
        }
        values.extend(std::iter::repeat_n(0, 70_000));
        values.extend([1, -1, 0, 0, 5]);
        values.extend([65_535, -65_536, 32_767, -32_768, 0, 0]);
        values.extend((0..2000).map(|i| (i * 7919) % 40_000 - 20_000));
        round_trip(&values, 17);
        let wide: Vec<i32> = (0..500)
            .map(|i: i32| i.wrapping_mul(0x0123_4567) >> 1)
            .collect();
        round_trip(&wide, 32);
        // `i32::MIN` has no code: its zigzag value wraps, as in the reference.
        // The encoder never meets it, because no channel is wider than 17 bits.
        round_trip(&[i32::MIN + 1, i32::MAX, 0, -1], 32);
    }

    #[test]
    fn a_run_past_the_frame_is_malformed() {
        let params = AgParams::new(10, 40, 14);
        let mut writer = BitWriter::default();
        compress(params, &[0; 100], &mut writer, 16);
        let bytes = writer.into_bytes();
        let mut out = vec![0; 50];
        let mut reader = BitReader::new(&bytes);
        assert!(decompress(params, &mut reader, &mut out, 16).is_err());
    }

    #[test]
    fn truncated_codes_fail_and_do_not_loop() {
        let params = AgParams::new(10, 40, 14);
        let mut out = vec![0; 4096];
        for bytes in [
            &[][..],
            &[0xff][..],
            &[0xff, 0xff, 0xff][..],
            &[0x00, 0x00][..],
        ] {
            let mut reader = BitReader::new(bytes);
            assert!(decompress(params, &mut reader, &mut out, 16).is_err());
        }
    }
}
