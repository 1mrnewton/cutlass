//! Band shapes: splitting, stereo, folding, and TF changes (RFC 6716
//! sections 4.3.4 to 4.3.6, `bands.c`), shared by the encoder and decoder,
//! with the RFC 8251 section 9 hybrid folding fix and the section 10
//! option to drop the intensity phase inversion for a mono downmix.

use super::coder::Coder;
use super::rate::{BITRES, bits2pulses, cache_max_bits, get_pulses, pulses2bits};
use super::tables::{
    EBANDS, LOGN, NB_EBANDS, QTHETA_OFFSET, QTHETA_OFFSET_TWOPHASE, SPREAD_AGGRESSIVE,
};
use super::vq;

/// The CELT linear congruential generator (`celt_lcg_rand`).
#[inline]
pub(crate) fn lcg_rand(seed: u32) -> u32 {
    seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}

/// `FRAC_MUL16`: a Q15 product of two values truncated to 16 bits.
#[inline]
fn frac_mul16(a: i32, b: i32) -> i32 {
    (16384 + i32::from(a as i16) * i32::from(b as i16)) >> 15
}

/// The bit-exact cosine of the stereo and split angle (`bitexact_cos`).
fn bitexact_cos(x: i32) -> i32 {
    let tmp = (4096 + x * x) >> 13;
    let x2 = tmp;
    let x2 = i32::from(
        ((32767 - x2) + frac_mul16(x2, -7651 + frac_mul16(x2, 8277 + frac_mul16(-626, x2)))) as i16,
    );
    1 + x2
}

/// log2(sin/cos) in Q11 (`bitexact_log2tan`).
fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    let lc = 32 - (icos as u32).leading_zeros() as i32;
    let ls = 32 - (isin as u32).leading_zeros() as i32;
    let icos = icos << (15 - lc);
    let isin = isin << (15 - ls);
    (ls - lc) * (1 << 11) + frac_mul16(isin, frac_mul16(isin, -2597) + 7932)
        - frac_mul16(icos, frac_mul16(icos, -2597) + 7932)
}

/// floor(sqrt(v)) (`isqrt32`).
fn isqrt32(v: u32) -> u32 {
    let mut g = f64::from(v).sqrt() as u32;
    while u64::from(g) * u64::from(g) > u64::from(v) {
        g -= 1;
    }
    while u64::from(g + 1) * u64::from(g + 1) <= u64::from(v) {
        g += 1;
    }
    g
}

/// One level of the Haar transform across `stride` interleaved vectors.
pub(crate) fn haar1(x: &mut [f32], n0: usize, stride: usize) {
    let half = n0 >> 1;
    let s = std::f32::consts::FRAC_1_SQRT_2;
    for i in 0..stride {
        for j in 0..half {
            let a = stride * 2 * j + i;
            let b = stride * (2 * j + 1) + i;
            if b >= x.len() {
                continue;
            }
            let t1 = s * x[a];
            let t2 = s * x[b];
            x[a] = t1 + t2;
            x[b] = t1 - t2;
        }
    }
}

/// `ordery_table`: Hadamard order for strides 2, 4, 8, and 16.
const ORDERY: [usize; 30] = [
    1, 0, 3, 0, 2, 1, 7, 0, 4, 3, 6, 1, 5, 2, 15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5,
];

fn ordery(stride: usize) -> Option<&'static [usize]> {
    match stride {
        2 | 4 | 8 | 16 => ORDERY.get(stride - 2..2 * stride - 2),
        _ => None,
    }
}

fn deinterleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    if n > x.len() || stride == 0 {
        return;
    }
    let mut tmp = vec![0f32; n];
    let order = if hadamard { ordery(stride) } else { None };
    for i in 0..stride {
        let row = order.map_or(i, |o| o[i]);
        for j in 0..n0 {
            tmp[row * n0 + j] = x[j * stride + i];
        }
    }
    x[..n].copy_from_slice(&tmp);
}

fn interleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    if n > x.len() || stride == 0 {
        return;
    }
    let mut tmp = vec![0f32; n];
    let order = if hadamard { ordery(stride) } else { None };
    for i in 0..stride {
        let row = order.map_or(i, |o| o[i]);
        for j in 0..n0 {
            tmp[j * stride + i] = x[row * n0 + j];
        }
    }
    x[..n].copy_from_slice(&tmp);
}

/// Resolution of the split angle (`compute_qn`).
fn compute_qn(n: i32, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    const EXP2_TABLE8: [i32; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];
    let mut n2 = 2 * n - 1;
    if stereo && n == 2 {
        n2 -= 1;
    }
    let mut qb = (b - pulse_cap - (4 << BITRES)).min((b + n2 * offset) / n2);
    qb = qb.min(8 << BITRES);
    if qb < (1 << BITRES >> 1) {
        1
    } else {
        let qn = EXP2_TABLE8[(qb & 7) as usize] >> (14 - (qb >> BITRES));
        ((qn + 1) >> 1) << 1
    }
}

/// Mixes the decoded mid and side back to left and right (`stereo_merge`).
fn stereo_merge(x: &mut [f32], y: &mut [f32], mid: f32) {
    let mut xp = 0f32;
    let mut side = 0f32;
    for (&a, &b) in x.iter().zip(y.iter()) {
        xp += a * b;
        side += b * b;
    }
    xp *= mid;
    let mid2 = mid;
    let el = mid2 * mid2 + side - 2.0 * xp;
    let er = mid2 * mid2 + side + 2.0 * xp;
    if er < 6e-4 || el < 6e-4 {
        y.copy_from_slice(x);
        return;
    }
    let lgain = 1.0 / el.sqrt();
    let rgain = 1.0 / er.sqrt();
    for (a, b) in x.iter_mut().zip(y.iter_mut()) {
        let l = mid * *a;
        let r = *b;
        *a = lgain * (l - r);
        *b = rgain * (l + r);
    }
}

/// Encoder: replaces `x` with the energy-weighted sum of both channels
/// (`intensity_stereo`).
fn intensity_stereo(x: &mut [f32], y: &[f32], band_e: &[f32], band: usize) {
    let left = band_e.get(band).copied().unwrap_or(0.0);
    let right = band_e.get(band + NB_EBANDS).copied().unwrap_or(0.0);
    let norm = 1e-15 + (1e-15 + left * left + right * right).sqrt();
    let a1 = left / norm;
    let a2 = right / norm;
    for (l, &r) in x.iter_mut().zip(y) {
        *l = a1 * *l + a2 * r;
    }
}

/// Encoder: left and right to mid and side (`stereo_split`).
fn stereo_split(x: &mut [f32], y: &mut [f32]) {
    let s = std::f32::consts::FRAC_1_SQRT_2;
    for (a, b) in x.iter_mut().zip(y.iter_mut()) {
        let l = s * *a;
        let r = s * *b;
        *a = l + r;
        *b = r - l;
    }
}

/// `bit_interleave_table` of `quant_band`.
const BIT_INTERLEAVE: [u32; 16] = [0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3];
/// `bit_deinterleave_table` of `quant_band`.
const BIT_DEINTERLEAVE: [u32; 16] = [
    0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F, 0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3, 0xFC, 0xFF,
];

/// State shared by every band of one frame.
struct Ctx<'r, 'c, 'a> {
    coder: &'r mut Coder<'c, 'a>,
    encode: bool,
    band: usize,
    intensity: usize,
    spread: usize,
    remaining_bits: i32,
    seed: u32,
    band_e: &'r [f32],
    disable_inv: bool,
}

/// One band (or a split half of one) as `quant_band` sees it.
struct Part<'p> {
    x: &'p mut [f32],
    y: Option<&'p mut [f32]>,
    b: i32,
    blocks: usize,
    tf_change: i32,
    lowband: Option<&'p mut [f32]>,
    lm: i32,
    lowband_out: Option<&'p mut [f32]>,
    level: u32,
    gain: f32,
    fill: u32,
}

fn quant_band(ctx: &mut Ctx<'_, '_, '_>, part: Part<'_>) -> u32 {
    let Part {
        x,
        mut y,
        mut b,
        blocks,
        mut tf_change,
        mut lowband,
        mut lm,
        mut lowband_out,
        level,
        gain,
        mut fill,
    } = part;
    let encode = ctx.encode;
    let resynth = !encode;
    let n0 = x.len();
    let mut n = n0;
    let mut bb = blocks.max(1);
    let mut b0 = bb;
    let long_blocks = b0 == 1;
    let mut n_b = n / bb;
    let mut n_b0 = n_b;
    let stereo = y.is_some();
    let mut split = stereo;
    let mut time_divide = 0;
    let mut recombine = 0usize;
    let mut inv = false;
    let mut mid = 0f32;
    let mut cm = 0u32;

    if n == 1 {
        let mut chans: [Option<&mut [f32]>; 2] = [Some(x), y];
        for ch in chans.iter_mut().flatten() {
            let mut sign = 0u32;
            if ctx.remaining_bits >= 1 << BITRES {
                let want = u32::from(ch[0] < 0.0);
                sign = ctx.coder.bits(want, 1);
                ctx.remaining_bits -= 1 << BITRES;
            }
            if resynth {
                ch[0] = if sign != 0 { -1.0 } else { 1.0 };
            }
        }
        if let (Some(out), Some(x)) = (lowband_out, chans[0].as_deref())
            && let Some(slot) = out.first_mut()
        {
            *slot = x[0];
        }
        return 1;
    }

    if !stereo && level == 0 {
        if tf_change > 0 {
            recombine = tf_change as usize;
        }
        for k in 0..recombine {
            if encode {
                haar1(x, n >> k, 1 << k);
            }
            if let Some(lb) = lowband.as_deref_mut() {
                haar1(lb, n >> k, 1 << k);
            }
            fill = BIT_INTERLEAVE[(fill & 0xF) as usize]
                | (BIT_INTERLEAVE[((fill >> 4) & 0xF) as usize] << 2);
        }
        bb = (bb >> recombine).max(1);
        n_b <<= recombine;
        while n_b & 1 == 0 && tf_change < 0 {
            if encode {
                haar1(x, n_b, bb);
            }
            if let Some(lb) = lowband.as_deref_mut() {
                haar1(lb, n_b, bb);
            }
            fill |= fill << bb;
            bb <<= 1;
            n_b >>= 1;
            time_divide += 1;
            tf_change += 1;
        }
        b0 = bb;
        n_b0 = n_b;
        if b0 > 1 {
            if encode {
                deinterleave_hadamard(x, n_b >> recombine, b0 << recombine, long_blocks);
            }
            if let Some(lb) = lowband.as_deref_mut() {
                deinterleave_hadamard(lb, n_b >> recombine, b0 << recombine, long_blocks);
            }
        }
    }

    if !stereo
        && lm != -1
        && n > 2
        && let Some(max) = cache_max_bits(ctx.band, lm)
        && b > max + 12
    {
        n >>= 1;
        split = true;
        lm -= 1;
        if bb == 1 {
            fill = (fill & 1) | (fill << 1);
        }
        bb = (bb + 1) >> 1;
    }

    if split {
        // x and y are the two halves (mono) or the two channels (stereo).
        let (xa, ya): (&mut [f32], &mut [f32]) = match y.as_deref_mut() {
            Some(y) => (&mut *x, y),
            None => x.split_at_mut(n),
        };
        let mut itheta = 0i32;
        let pulse_cap = LOGN[ctx.band] + lm * (1 << BITRES);
        let offset = (pulse_cap >> 1)
            - if stereo && n == 2 {
                QTHETA_OFFSET_TWOPHASE
            } else {
                QTHETA_OFFSET
            };
        let mut qn = compute_qn(n as i32, b, offset, pulse_cap, stereo);
        if stereo && ctx.band >= ctx.intensity {
            qn = 1;
        }
        if encode {
            itheta = vq::stereo_itheta(xa, ya, stereo);
        }
        let tell = ctx.coder.tell_frac();
        if qn != 1 {
            if encode {
                itheta = (itheta * qn + 8192) >> 14;
            }
            if stereo && n > 2 {
                let p0 = 3i32;
                let x0 = qn / 2;
                let ft = (p0 * (x0 + 1) + x0) as u32;
                let range = |x: i32| -> (u32, u32) {
                    if x <= x0 {
                        ((p0 * x) as u32, (p0 * (x + 1)) as u32)
                    } else {
                        (
                            (x - 1 - x0 + (x0 + 1) * p0) as u32,
                            (x - x0 + (x0 + 1) * p0) as u32,
                        )
                    }
                };
                match &mut *ctx.coder {
                    Coder::Enc(e) => {
                        let (fl, fh) = range(itheta);
                        e.encode(fl, fh, ft);
                    }
                    Coder::Dec(d) => {
                        let fs = d.decode(ft) as i32;
                        let x = if fs < (x0 + 1) * p0 {
                            fs / p0
                        } else {
                            x0 + 1 + (fs - (x0 + 1) * p0)
                        };
                        let (fl, fh) = range(x);
                        d.update(fl, fh.min(ft), ft);
                        itheta = x;
                    }
                }
            } else if b0 > 1 || stereo {
                itheta = ctx.coder.uint(itheta as u32, (qn + 1) as u32) as i32;
            } else {
                let half = qn >> 1;
                let ft = ((half + 1) * (half + 1)) as u32;
                match &mut *ctx.coder {
                    Coder::Enc(e) => {
                        let fs = if itheta <= half {
                            itheta + 1
                        } else {
                            qn + 1 - itheta
                        };
                        let fl = if itheta <= half {
                            (itheta * (itheta + 1)) >> 1
                        } else {
                            ft as i32 - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1)
                        };
                        e.encode(fl as u32, (fl + fs) as u32, ft);
                    }
                    Coder::Dec(d) => {
                        let fm = d.decode(ft);
                        let (fl, fs);
                        if fm < ((half * (half + 1)) >> 1) as u32 {
                            itheta = (isqrt32(8 * fm + 1) as i32 - 1) >> 1;
                            fs = itheta + 1;
                            fl = (itheta * (itheta + 1)) >> 1;
                        } else {
                            itheta = (2 * (qn + 1) - isqrt32(8 * (ft - fm - 1) + 1) as i32) >> 1;
                            fs = qn + 1 - itheta;
                            fl = ft as i32 - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1);
                        }
                        let fl = fl.clamp(0, ft as i32 - 1);
                        let fh = (fl + fs).clamp(fl + 1, ft as i32);
                        d.update(fl as u32, fh as u32, ft);
                    }
                }
            }
            itheta = itheta * 16384 / qn;
            if encode && stereo {
                if itheta == 0 {
                    intensity_stereo(xa, ya, ctx.band_e, ctx.band);
                } else {
                    stereo_split(xa, ya);
                }
            }
        } else if stereo {
            if encode {
                inv = itheta > 8192 && !ctx.disable_inv;
                if inv {
                    for v in ya.iter_mut() {
                        *v = -*v;
                    }
                }
                intensity_stereo(xa, ya, ctx.band_e, ctx.band);
            }
            inv = if b > 2 << BITRES && ctx.remaining_bits > 2 << BITRES {
                ctx.coder.bit_logp(inv, 2)
            } else {
                false
            };
            if ctx.disable_inv {
                inv = false;
            }
            itheta = 0;
        }
        let qalloc = ctx.coder.tell_frac() - tell;
        b -= qalloc;

        let orig_fill = fill;
        let (imid, iside, mut delta);
        if itheta == 0 {
            imid = 32767;
            iside = 0;
            fill &= (1 << bb) - 1;
            delta = -16384;
        } else if itheta == 16384 {
            imid = 0;
            iside = 32767;
            fill &= ((1 << bb) - 1) << bb;
            delta = 16384;
        } else {
            imid = bitexact_cos(itheta);
            iside = bitexact_cos(16384 - itheta);
            delta = frac_mul16(((n as i32) - 1) << 7, bitexact_log2tan(iside, imid));
        }
        mid = imid as f32 / 32768.0;
        let side = iside as f32 / 32768.0;

        if n == 2 && stereo {
            let mut mbits = b;
            let mut sbits = 0;
            if itheta != 0 && itheta != 16384 {
                sbits = 1 << BITRES;
            }
            mbits -= sbits;
            let swap = itheta > 8192;
            ctx.remaining_bits -= qalloc + sbits;
            let (x2, y2): (&mut [f32], &mut [f32]) = if swap { (ya, xa) } else { (xa, ya) };
            let mut sign = 0u32;
            if sbits != 0 {
                let want = u32::from(x2[0] * y2[1] - x2[1] * y2[0] < 0.0);
                sign = ctx.coder.bits(want, 1);
            }
            let sign = 1.0 - 2.0 * sign as f32;
            cm = quant_band(
                ctx,
                Part {
                    x: &mut *x2,
                    y: None,
                    b: mbits,
                    blocks: bb,
                    tf_change,
                    lowband: lowband.take(),
                    lm,
                    lowband_out: lowband_out.take(),
                    level,
                    gain,
                    fill: orig_fill,
                },
            );
            y2[0] = -sign * x2[1];
            y2[1] = sign * x2[0];
            if resynth {
                let (xs, ys): (&mut [f32], &mut [f32]) = if swap { (y2, x2) } else { (x2, y2) };
                xs[0] *= mid;
                xs[1] *= mid;
                ys[0] *= side;
                ys[1] *= side;
                let t = xs[0];
                xs[0] = t - ys[0];
                ys[0] += t;
                let t = xs[1];
                xs[1] = t - ys[1];
                ys[1] += t;
            }
        } else {
            if b0 > 1 && !stereo && (itheta & 0x3fff) != 0 {
                if itheta > 8192 {
                    delta -= delta >> (4 - lm).clamp(0, 31);
                } else {
                    delta = 0.min(delta + (((n as i32) << BITRES) >> (5 - lm).clamp(0, 31)));
                }
            }
            let mut mbits = 0.max(b.min((b - delta) / 2));
            let mut sbits = b - mbits;
            ctx.remaining_bits -= qalloc;

            let (lb1, lb2) = match lowband.take() {
                Some(lb) if !stereo => {
                    let at = n.min(lb.len());
                    let (a, c) = lb.split_at_mut(at);
                    (Some(a), Some(c))
                }
                other => (other, None),
            };
            let out1 = if stereo { lowband_out.take() } else { None };
            let next_level = if stereo { 0 } else { level + 1 };
            let shift = if stereo { 0 } else { b0 >> 1 };
            let mid_gain = if stereo { 1.0 } else { gain * mid };
            let side_gain = gain * side;
            let mut rebalance = ctx.remaining_bits;
            if mbits >= sbits {
                cm = quant_band(
                    ctx,
                    Part {
                        x: xa,
                        y: None,
                        b: mbits,
                        blocks: bb,
                        tf_change,
                        lowband: lb1,
                        lm,
                        lowband_out: out1,
                        level: next_level,
                        gain: mid_gain,
                        fill,
                    },
                );
                rebalance = mbits - (rebalance - ctx.remaining_bits);
                if rebalance > 3 << BITRES && itheta != 0 {
                    sbits += rebalance - (3 << BITRES);
                }
                cm |= quant_band(
                    ctx,
                    Part {
                        x: ya,
                        y: None,
                        b: sbits,
                        blocks: bb,
                        tf_change,
                        lowband: lb2,
                        lm,
                        lowband_out: None,
                        level: next_level,
                        gain: side_gain,
                        fill: fill >> bb,
                    },
                ) << shift;
            } else {
                cm = quant_band(
                    ctx,
                    Part {
                        x: ya,
                        y: None,
                        b: sbits,
                        blocks: bb,
                        tf_change,
                        lowband: lb2,
                        lm,
                        lowband_out: None,
                        level: next_level,
                        gain: side_gain,
                        fill: fill >> bb,
                    },
                ) << shift;
                rebalance = sbits - (rebalance - ctx.remaining_bits);
                if rebalance > 3 << BITRES && itheta != 16384 {
                    mbits += rebalance - (3 << BITRES);
                }
                cm |= quant_band(
                    ctx,
                    Part {
                        x: xa,
                        y: None,
                        b: mbits,
                        blocks: bb,
                        tf_change,
                        lowband: lb1,
                        lm,
                        lowband_out: out1,
                        level: next_level,
                        gain: mid_gain,
                        fill,
                    },
                );
            }
        }
    } else {
        let mut q = bits2pulses(ctx.band, lm, b);
        let mut curr_bits = pulses2bits(ctx.band, lm, q);
        ctx.remaining_bits -= curr_bits;
        while ctx.remaining_bits < 0 && q > 0 {
            ctx.remaining_bits += curr_bits;
            q -= 1;
            curr_bits = pulses2bits(ctx.band, lm, q);
            ctx.remaining_bits -= curr_bits;
        }
        if q != 0 {
            let k = get_pulses(q) as usize;
            cm = if encode {
                vq::alg_quant(x, k, ctx.spread, bb, ctx.coder)
            } else {
                vq::alg_unquant(x, k, ctx.spread, bb, ctx.coder, gain)
            };
        } else if resynth {
            let cm_mask = ((1u64 << bb) - 1) as u32;
            fill &= cm_mask;
            if fill == 0 {
                x.fill(0.0);
            } else {
                match lowband.as_deref() {
                    None => {
                        for v in x.iter_mut() {
                            ctx.seed = lcg_rand(ctx.seed);
                            *v = ((ctx.seed as i32) >> 20) as f32;
                        }
                        cm = cm_mask;
                    }
                    Some(lb) => {
                        for (j, v) in x.iter_mut().enumerate() {
                            ctx.seed = lcg_rand(ctx.seed);
                            let tmp = if ctx.seed & 0x8000 != 0 {
                                1.0 / 256.0
                            } else {
                                -1.0 / 256.0
                            };
                            *v = lb.get(j).copied().unwrap_or(0.0) + tmp;
                        }
                        cm = fill;
                    }
                }
                vq::renormalise(x, gain);
            }
        }
    }

    if resynth {
        if stereo {
            if let Some(y) = y {
                if n != 2 {
                    stereo_merge(x, y, mid);
                }
                if inv {
                    for v in y.iter_mut() {
                        *v = -*v;
                    }
                }
            }
        } else if level == 0 {
            if b0 > 1 {
                interleave_hadamard(x, n_b >> recombine, b0 << recombine, long_blocks);
            }
            n_b = n_b0;
            bb = b0;
            for _ in 0..time_divide {
                bb >>= 1;
                n_b <<= 1;
                cm |= cm >> bb;
                haar1(x, n_b, bb);
            }
            for k in 0..recombine {
                cm = BIT_DEINTERLEAVE[(cm & 0xF) as usize];
                haar1(x, n0 >> k, 1 << k);
            }
            bb <<= recombine;
            if let Some(out) = lowband_out {
                let scale = (n0 as f32).sqrt();
                for (o, v) in out.iter_mut().zip(x.iter()) {
                    *o = scale * v;
                }
            }
            cm &= ((1u64 << bb) - 1) as u32;
        }
    }
    cm
}

/// The frame-level inputs of [`quant_all_bands`].
pub(crate) struct AllBands<'r> {
    pub start: usize,
    pub end: usize,
    /// Encoder only: band amplitudes for intensity stereo.
    pub band_e: &'r [f32],
    pub pulses: &'r [i32; NB_EBANDS],
    pub short_blocks: bool,
    pub spread: usize,
    pub dual_stereo: bool,
    pub intensity: usize,
    pub tf_res: &'r [i32; NB_EBANDS],
    pub total_bits: i32,
    pub balance: i32,
    pub lm: usize,
    pub coded_bands: usize,
    pub disable_inv: bool,
}

/// Codes the shapes of all bands (`quant_all_bands`). `x` and `y` hold the
/// normalized spectrum of each channel (`y` is `None` in mono); the decoder
/// overwrites them. Returns the collapse masks, two per band.
pub(crate) fn quant_all_bands(
    p: &AllBands<'_>,
    x: &mut [f32],
    mut y: Option<&mut [f32]>,
    coder: &mut Coder<'_, '_>,
    seed: &mut u32,
) -> [u8; 2 * NB_EBANDS] {
    let encode = coder.encoding();
    let resynth = !encode;
    let channels = if y.is_some() { 2 } else { 1 };
    let m = 1usize << p.lm;
    let blocks = if p.short_blocks { m } else { 1 };
    let span = m * EBANDS[NB_EBANDS];
    let mut norm = vec![0f32; channels * span];
    let mut collapse = [0u8; 2 * NB_EBANDS];
    let mut lowband_offset = 0usize;
    let mut update_lowband = true;
    let mut balance = p.balance;
    let mut dual_stereo = p.dual_stereo;
    let mut ctx = Ctx {
        coder,
        encode,
        band: 0,
        intensity: p.intensity,
        spread: p.spread,
        remaining_bits: 0,
        seed: *seed,
        band_e: p.band_e,
        disable_inv: p.disable_inv,
    };
    let start = p.start;
    let end = p.end.min(NB_EBANDS);
    let mut lowband_buf = vec![0f32; m * 22];
    let mut lowband_buf2 = vec![0f32; m * 22];
    for i in start..end {
        let lo = m * EBANDS[i];
        let hi = m * EBANDS[i + 1];
        let n = hi - lo;
        let tell = ctx.coder.tell_frac();
        if i != start {
            balance -= tell;
        }
        ctx.remaining_bits = p.total_bits - tell - 1;
        let b = if i < p.coded_bands {
            let curr_balance = balance / 3.min(p.coded_bands as i32 - i as i32);
            0.max(16383.min((ctx.remaining_bits + 1).min(p.pulses[i] + curr_balance)))
        } else {
            0
        };
        if resynth
            && (lo as i64 - n as i64 >= (m * EBANDS[start]) as i64 || i == start + 1)
            && (update_lowband || lowband_offset == 0)
        {
            lowband_offset = i;
        }
        if i == start + 1 {
            // RFC 8251 section 9: repeat part of the first band so the
            // second band can always fold.
            let n1 = m * (EBANDS[start + 1] - EBANDS[start]);
            let n2 = m * (EBANDS[start + 2] - EBANDS[start + 1]);
            let offset = m * EBANDS[start];
            if n2 > n1 && 2 * n1 >= n2 {
                for c in 0..channels {
                    let base = c * span + offset;
                    norm.copy_within(base + 2 * n1 - n2..base + n1, base + n1);
                }
            }
        }
        let tf_change = p.tf_res[i];
        let mut effective_lowband: Option<usize> = None;
        let (mut x_cm, mut y_cm);
        if lowband_offset != 0 && (p.spread != SPREAD_AGGRESSIVE || blocks > 1 || tf_change < 0) {
            let eff = (m * EBANDS[start]).max((m * EBANDS[lowband_offset]).saturating_sub(n));
            effective_lowband = Some(eff);
            let mut fold_start = lowband_offset;
            loop {
                if fold_start == 0 {
                    break;
                }
                fold_start -= 1;
                if m * EBANDS[fold_start] <= eff {
                    break;
                }
            }
            let mut fold_end = lowband_offset - 1;
            loop {
                fold_end += 1;
                if !(fold_end < i && m * EBANDS[fold_end] < eff + n) {
                    break;
                }
            }
            x_cm = 0u32;
            y_cm = 0u32;
            let mut fold_i = fold_start;
            loop {
                x_cm |= u32::from(collapse[fold_i * channels]);
                y_cm |= u32::from(collapse[fold_i * channels + channels - 1]);
                fold_i += 1;
                if fold_i >= fold_end {
                    break;
                }
            }
        } else {
            x_cm = ((1u64 << blocks) - 1) as u32;
            y_cm = x_cm;
        }
        if dual_stereo && i == p.intensity {
            dual_stereo = false;
            if channels == 2 {
                for j in m * EBANDS[start]..lo {
                    norm[j] = 0.5 * (norm[j] + norm[span + j]);
                }
            }
        }
        // Copy the folding source so the band can write its own output into
        // `norm` (the RFC 8251 copy above may overlap the two).
        let mut lowband1: Option<&mut [f32]> = None;
        let mut lowband2: Option<&mut [f32]> = None;
        if let Some(eff) = effective_lowband {
            let take = n.min(span.saturating_sub(eff));
            lowband_buf[..n].fill(0.0);
            lowband_buf[..take].copy_from_slice(&norm[eff..eff + take]);
            lowband1 = Some(&mut lowband_buf[..n]);
            if channels == 2 {
                lowband_buf2[..n].fill(0.0);
                lowband_buf2[..take].copy_from_slice(&norm[span + eff..span + eff + take]);
                lowband2 = Some(&mut lowband_buf2[..n]);
            }
        }
        let (norm1, norm2) = norm.split_at_mut(span);
        ctx.band = i;
        let xb = &mut x[lo..hi];
        if dual_stereo {
            x_cm = quant_band(
                &mut ctx,
                Part {
                    x: xb,
                    y: None,
                    b: b / 2,
                    blocks,
                    tf_change,
                    lowband: lowband1,
                    lm: p.lm as i32,
                    lowband_out: Some(&mut norm1[lo..hi]),
                    level: 0,
                    gain: 1.0,
                    fill: x_cm,
                },
            );
            if let Some(yv) = y.as_deref_mut() {
                y_cm = quant_band(
                    &mut ctx,
                    Part {
                        x: &mut yv[lo..hi],
                        y: None,
                        b: b / 2,
                        blocks,
                        tf_change,
                        lowband: lowband2,
                        lm: p.lm as i32,
                        lowband_out: norm2.get_mut(lo..hi),
                        level: 0,
                        gain: 1.0,
                        fill: y_cm,
                    },
                );
            }
        } else {
            x_cm = quant_band(
                &mut ctx,
                Part {
                    x: xb,
                    y: y.as_deref_mut().map(|yv| &mut yv[lo..hi]),
                    b,
                    blocks,
                    tf_change,
                    lowband: lowband1,
                    lm: p.lm as i32,
                    lowband_out: Some(&mut norm1[lo..hi]),
                    level: 0,
                    gain: 1.0,
                    fill: x_cm | y_cm,
                },
            );
            y_cm = x_cm;
        }
        collapse[i * channels] = x_cm as u8;
        collapse[i * channels + channels - 1] = y_cm as u8;
        balance += p.pulses[i] + tell;
        update_lowband = b > ((n as i32) << BITRES);
    }
    *seed = ctx.seed;
    collapse
}

/// The inputs of [`anti_collapse`].
pub(crate) struct AntiCollapse<'r> {
    pub collapse: &'r [u8; 2 * NB_EBANDS],
    pub lm: usize,
    pub channels: usize,
    /// Bins per channel in `x`.
    pub size: usize,
    pub bands: std::ops::Range<usize>,
    /// This frame's energies and the two before it.
    pub log_e: &'r [f32],
    pub prev1_log_e: &'r [f32],
    pub prev2_log_e: &'r [f32],
    pub pulses: &'r [i32; NB_EBANDS],
    pub seed: u32,
}

/// Fills blocks that got no pulses with noise at the level the energy
/// history allows (`anti_collapse`). `x` holds `channels` spectra of `size`.
pub(crate) fn anti_collapse(x: &mut [f32], p: &AntiCollapse<'_>) {
    let lm = p.lm;
    let channels = p.channels;
    let mut seed = p.seed;
    for i in p.bands.clone() {
        let n0 = EBANDS[i + 1] - EBANDS[i];
        let depth = (1 + p.pulses[i]) / ((n0 << lm) as i32);
        let thresh = 0.5 * super::energy::exp2(-0.125 * depth as f32);
        let sqrt_1 = 1.0 / ((n0 << lm) as f32).sqrt();
        for c in 0..channels {
            let mut prev1 = p.prev1_log_e[c * NB_EBANDS + i];
            let mut prev2 = p.prev2_log_e[c * NB_EBANDS + i];
            if channels == 1 {
                prev1 = prev1.max(p.prev1_log_e[NB_EBANDS + i]);
                prev2 = prev2.max(p.prev2_log_e[NB_EBANDS + i]);
            }
            let ediff = (p.log_e[c * NB_EBANDS + i] - prev1.min(prev2)).max(0.0);
            let mut r = 2.0 * super::energy::exp2(-ediff);
            if lm == 3 {
                r *= std::f32::consts::SQRT_2;
            }
            r = thresh.min(r) * sqrt_1;
            let base = c * p.size + (EBANDS[i] << lm);
            let Some(band) = x.get_mut(base..base + (n0 << lm)) else {
                continue;
            };
            let mut renormalize = false;
            for k in 0..1usize << lm {
                if u32::from(p.collapse[i * channels + c]) & (1 << k) == 0 {
                    for j in 0..n0 {
                        seed = lcg_rand(seed);
                        band[(j << lm) + k] = if seed & 0x8000 != 0 { r } else { -r };
                    }
                    renormalize = true;
                }
            }
            if renormalize {
                vq::renormalise(band, 1.0);
            }
        }
    }
}

/// Spectrum from unit-norm bands and amplitudes (`denormalise_bands`).
pub(crate) fn denormalise(x: &[f32], freq: &mut [f32], band_e: &[f32], end: usize, m: usize) {
    let n = freq.len();
    for i in 0..end.min(NB_EBANDS) {
        let g = band_e[i];
        for j in m * EBANDS[i]..(m * EBANDS[i + 1]).min(n) {
            freq[j] = x[j] * g;
        }
    }
    let from = (m * EBANDS[end.min(NB_EBANDS)]).min(n);
    freq[from..].fill(0.0);
}

/// Band amplitudes of one channel's spectrum (`compute_band_energies`).
pub(crate) fn band_energies(freq: &[f32], band_e: &mut [f32], end: usize, m: usize) {
    for i in 0..end.min(NB_EBANDS) {
        let sum: f32 = freq[m * EBANDS[i]..m * EBANDS[i + 1]]
            .iter()
            .map(|v| v * v)
            .sum();
        band_e[i] = (1e-27 + sum).sqrt();
    }
}

/// Unit-norm bands of one channel's spectrum (`normalise_bands`).
pub(crate) fn normalise(freq: &[f32], x: &mut [f32], band_e: &[f32], end: usize, m: usize) {
    for i in 0..end.min(NB_EBANDS) {
        let g = 1.0 / (1e-27 + band_e[i]);
        for j in m * EBANDS[i]..m * EBANDS[i + 1] {
            x[j] = freq[j] * g;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        bitexact_cos, bitexact_log2tan, compute_qn, deinterleave_hadamard, haar1,
        interleave_hadamard, isqrt32,
    };

    #[test]
    fn exact_helpers_match_reference_values() {
        // Values worked through the reference's integer arithmetic.
        // The reference never passes 0 (itheta 0 is special-cased); its
        // 16-bit result would wrap there.
        assert_eq!(bitexact_cos(64), 32767);
        assert_eq!(bitexact_cos(16384), 16554);
        assert_eq!(bitexact_cos(8192), 23171);
        assert_eq!(bitexact_log2tan(23171, 23171), 0);
        for v in [0u32, 1, 2, 3, 4, 15, 16, 17, 1 << 20, u32::MAX] {
            let g = isqrt32(v);
            assert!(u64::from(g) * u64::from(g) <= u64::from(v));
            assert!(u64::from(g + 1) * u64::from(g + 1) > u64::from(v));
        }
        assert_eq!(compute_qn(16, 0, 0, 0, false), 1);
    }

    #[test]
    fn hadamard_interleaving_inverts() {
        for stride in [2usize, 4, 8, 16] {
            for hadamard in [false, true] {
                let n0 = 3;
                let x0: Vec<f32> = (0..n0 * stride).map(|v| v as f32).collect();
                let mut x = x0.clone();
                deinterleave_hadamard(&mut x, n0, stride, hadamard);
                interleave_hadamard(&mut x, n0, stride, hadamard);
                assert_eq!(x, x0);
            }
        }
        let mut x = vec![1.0f32, 3.0, -2.0, 5.0];
        haar1(&mut x, 4, 1);
        haar1(&mut x, 4, 1);
        for (a, b) in x.iter().zip([1.0f32, 3.0, -2.0, 5.0]) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
