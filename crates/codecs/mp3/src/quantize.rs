//! Quantization of one granule of one channel: per-band quantizer steps
//! from the allowed noise, scalefactors that reach them, the global gain
//! that meets a bit budget, and the Huffman layout of the result.
//!
//! The quantizer is ISO/IEC 11172-3 Annex C.1.5.4: `ix = nint(|xr|^(3/4) /
//! 2^(3 step / 16) - 0.0946)`. Instead of the informative outer loop
//! (amplify a band, requantize, repeat), each band's coarsest step that
//! keeps its noise under the allowed noise is searched directly; the
//! scalefactors then lift every band to its step from the global gain,
//! and one search over the global gain meets the budget with that shape
//! held, so a short budget raises the noise of every band by the same
//! number of decibels.

use crate::analysis::SHORT;
use crate::huffenc::{
    Division, MAX_VALUE, best_division, choose_table, count1, extents, quick_division,
};
use crate::layer3::pow43;
use crate::sideinfo::GranuleInfo;
use crate::tables::{PRETAB, SHORT_WIDTHS, SLEN, long_starts, short_starts};

/// `step` is `global_gain - 210` less the band's amplification, in
/// quarters of 1.5 dB: reconstruction is `ix^(4/3) * 2^(step / 4)`.
const STEP_MIN: i32 = -210;
const STEP_MAX: i32 = 45;
/// The largest bit count `part2_3_length` holds.
pub(crate) const MAX_GRANULE_BITS: u32 = 4095;

/// Which scalefactor scales a run of lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Slot {
    /// Long band 0..=20.
    Long(usize),
    /// Short band 0..=11, window.
    Short(usize, usize),
    /// Long band 21 or short band 12 (with its window): no scalefactor.
    Fixed(Option<usize>),
}

/// A run of lines in bitstream order with one quantizer step.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Band {
    pub from: usize,
    pub to: usize,
    pub slot: Slot,
}

/// The coding bands of a block type at a rate table, covering 0..576.
pub(crate) fn bands(block_type: u8, rate_table: usize) -> Vec<Band> {
    let mut out = Vec::with_capacity(39);
    if block_type == SHORT {
        let starts = short_starts(rate_table);
        let widths = SHORT_WIDTHS[rate_table];
        for band in 0..13 {
            let width = usize::from(widths[band]);
            let base = 3 * usize::from(starts[band]);
            for w in 0..3 {
                let from = base + w * width;
                out.push(Band {
                    from,
                    to: from + width,
                    slot: if band < 12 {
                        Slot::Short(band, w)
                    } else {
                        Slot::Fixed(Some(w))
                    },
                });
            }
        }
    } else {
        let starts = long_starts(rate_table);
        for band in 0..22 {
            out.push(Band {
                from: usize::from(starts[band]),
                to: if band == 21 {
                    576
                } else {
                    usize::from(starts[band + 1])
                },
                slot: if band < 21 {
                    Slot::Long(band)
                } else {
                    Slot::Fixed(None)
                },
            });
        }
    }
    out
}

/// What the quantizer needs about the stream.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Context {
    pub lsf: bool,
    pub rate_table: usize,
    /// The region-0 boundary of a window-switching granule, in lines.
    pub switch_region0: usize,
}

impl Context {
    pub(crate) fn new(lsf: bool, rate_table: usize, block_type: u8) -> Self {
        // ISO/IEC 11172-3 2.4.2.7 and ISO/IEC 13818-3 2.4.3.4: 36 lines,
        // or 54 for an LSF start or stop block; 8 kHz doubles both.
        let base = if block_type != SHORT && lsf { 54 } else { 36 };
        let switch_region0 = if rate_table == 8 { base * 2 } else { base };
        Self {
            lsf,
            rate_table,
            switch_region0,
        }
    }
}

/// One coded granule of one channel.
#[derive(Clone, Debug)]
pub(crate) struct Coded {
    pub info: GranuleInfo,
    pub sf_long: [u8; 22],
    pub sf_short: [[u8; 3]; 13],
    /// LSF: `slen` per partition.
    pub slen: [u8; 4],
    pub ix: [i32; 576],
    pub part2_bits: u32,
    pub part3_bits: u32,
    /// Lines in the big-value region and in the count1 region.
    pub big_end: usize,
    pub count1_end: usize,
}

impl Coded {
    pub(crate) fn silent(block_type: u8) -> Self {
        let info = GranuleInfo {
            window_switching: block_type != 0,
            block_type,
            ..GranuleInfo::default()
        };
        Self {
            info,
            sf_long: [0; 22],
            sf_short: [[0; 3]; 13],
            slen: [0; 4],
            ix: [0; 576],
            part2_bits: 0,
            part3_bits: 0,
            big_end: 0,
            count1_end: 0,
        }
    }

    pub(crate) fn bits(&self) -> u32 {
        self.part2_bits + self.part3_bits
    }
}

/// `2^(-3 step / 16)`: the factor on `|xr|^(3/4)` before rounding.
fn quantizer(step: i32) -> f32 {
    (-3.0 * f64::from(step) / 16.0).exp2() as f32
}

/// `2^(step / 4)`: the reconstruction factor on `ix^(4/3)`.
fn reconstruction(step: i32) -> f32 {
    (f64::from(step) / 4.0).exp2() as f32
}

fn quantize_line(xr34: f32, factor: f32) -> u32 {
    let value = xr34 * factor + 0.4054;
    if value >= (MAX_VALUE + 1) as f32 {
        MAX_VALUE + 1
    } else {
        value as u32
    }
}

/// Noise energy of `lines` quantized at `step`, or `None` past the
/// largest codable value.
fn band_noise(xr: &[f32], xr34: &[f32], step: i32) -> Option<f32> {
    let factor = quantizer(step);
    let back = reconstruction(step);
    let pow = pow43();
    let mut noise = 0f32;
    for (x, x34) in xr.iter().zip(xr34) {
        let q = quantize_line(*x34, factor);
        if q > MAX_VALUE {
            return None;
        }
        let d = x.abs() - pow[q as usize] * back;
        noise += d * d;
    }
    Some(noise)
}

/// The coarsest step at which the band's noise stays at or under `xmin`,
/// or `None` when the band may be coded as all zero.
fn band_step(xr: &[f32], xr34: &[f32], xmin: f32) -> Option<i32> {
    let energy: f32 = xr.iter().map(|x| x * x).sum();
    if energy <= xmin {
        return None;
    }
    let fits = |step: i32| band_noise(xr, xr34, step).is_some_and(|noise| noise <= xmin);
    // Finer steps raise the values: find the finest step the tables code.
    let codable = |step: i32| band_noise(xr, xr34, step).is_some();
    let mut lo = STEP_MIN;
    if !codable(lo) {
        let mut hi = STEP_MAX;
        if !codable(hi) {
            return Some(STEP_MAX);
        }
        // Invariant: `lo` overflows, `hi` codes.
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if codable(mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        lo = hi;
    }
    if !fits(lo) {
        return Some(lo);
    }
    let mut hi = STEP_MAX + 1;
    // Invariant: `lo` fits; `hi` does not, or is past the range.
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(lo)
}

/// The shape the scalefactors give a granule: per band, how many quarter
/// steps finer than the global gain it is quantized.
#[derive(Clone, Debug)]
struct Shape {
    scale: bool,
    preflag: bool,
    subblock_gain: [u8; 3],
    sf_long: [u8; 22],
    sf_short: [[u8; 3]; 13],
    /// The step the global gain sets, before band amplification.
    global_step: i32,
}

impl Shape {
    fn amplification(&self, slot: Slot) -> i32 {
        let unit = if self.scale { 4 } else { 2 };
        match slot {
            Slot::Long(band) => {
                let pre = if self.preflag { PRETAB[band] } else { 0 };
                unit * (i32::from(self.sf_long[band]) + i32::from(pre))
            }
            Slot::Short(band, w) => {
                unit * i32::from(self.sf_short[band][w]) + 8 * i32::from(self.subblock_gain[w])
            }
            Slot::Fixed(Some(w)) => 8 * i32::from(self.subblock_gain[w]),
            Slot::Fixed(None) => 0,
        }
    }
}

/// The largest scalefactor each slot may carry.
fn sf_limit(ctx: &Context, slot: Slot) -> i32 {
    if ctx.lsf {
        // Partition table 0: slen up to 4, 4, 3, 3.
        match slot {
            Slot::Long(band) if band < 11 => 15,
            Slot::Short(band, _) if band < 6 => 15,
            Slot::Long(_) | Slot::Short(..) => 7,
            Slot::Fixed(_) => 0,
        }
    } else {
        match slot {
            Slot::Long(band) if band < 11 => 15,
            Slot::Short(band, _) if band < 6 => 15,
            Slot::Long(_) | Slot::Short(..) => 7,
            Slot::Fixed(_) => 0,
        }
    }
}

/// Scalefactors that bring each band to `steps` from the global step.
fn shape_for(
    ctx: &Context,
    bands: &[Band],
    steps: &[Option<i32>],
    block_type: u8,
    scale: bool,
) -> Shape {
    let unit = if scale { 4 } else { 2 };
    // The coarsest global step every band can still be lifted to.
    let mut global = STEP_MIN;
    let mut reach = i32::MAX;
    for (band, step) in bands.iter().zip(steps) {
        let Some(step) = *step else { continue };
        global = global.max(step);
        // The band without a scalefactor takes the global step: holding the
        // whole granule to its precision costs more than its noise.
        let lift = match band.slot {
            Slot::Fixed(_) => continue,
            slot => unit * sf_limit(ctx, slot) + if block_type == SHORT { 8 * 7 } else { 0 },
        };
        reach = reach.min(step + lift);
    }
    let global = global.min(reach).clamp(STEP_MIN, STEP_MAX);
    let mut shape = Shape {
        scale,
        preflag: false,
        subblock_gain: [0; 3],
        sf_long: [0; 22],
        sf_short: [[0; 3]; 13],
        global_step: global,
    };
    if block_type == SHORT {
        // Subblock gain takes the part of each window's lift all its bands
        // share, and at least what the scalefactors cannot reach.
        let mut least = [i32::MAX; 3];
        let mut beyond = [0i32; 3];
        for (band, step) in bands.iter().zip(steps) {
            if let (Slot::Short(_, w) | Slot::Fixed(Some(w)), Some(step)) = (band.slot, step) {
                let need = (global - step).max(0);
                least[w] = least[w].min(need);
                if matches!(band.slot, Slot::Short(..)) {
                    beyond[w] = beyond[w].max(need - unit * sf_limit(ctx, band.slot));
                }
            }
        }
        for w in 0..3 {
            if least[w] != i32::MAX {
                let shared = least[w] / 8;
                let required = (beyond[w] + 7) / 8;
                shape.subblock_gain[w] = shared.max(required).clamp(0, 7) as u8;
            }
        }
    }
    for (band, step) in bands.iter().zip(steps) {
        let Some(step) = *step else { continue };
        let need = (global - step).max(0);
        match band.slot {
            Slot::Long(index) => {
                let sf = (need + unit - 1) / unit;
                shape.sf_long[index] = sf.clamp(0, sf_limit(ctx, band.slot)) as u8;
            }
            Slot::Short(index, w) => {
                let need = (need - 8 * i32::from(shape.subblock_gain[w])).max(0);
                let sf = (need + unit - 1) / unit;
                shape.sf_short[index][w] = sf.clamp(0, sf_limit(ctx, band.slot)) as u8;
            }
            Slot::Fixed(_) => {}
        }
    }
    if block_type != SHORT && !ctx.lsf && (11..21).all(|b| shape.sf_long[b] >= PRETAB[b]) {
        shape.preflag = true;
        for (sf, pre) in shape.sf_long[11..21].iter_mut().zip(&PRETAB[11..21]) {
            *sf -= pre;
        }
    }
    shape
}

/// Noise is not monotonic in the step over a few lines: a band lifted
/// past the step it asked for can land over its allowed noise. Lift such
/// a band further while its scalefactor has room.
fn refine(
    ctx: &Context,
    bands: &[Band],
    xr: &[f32; 576],
    xr34: &[f32; 576],
    xmin: &[f32],
    shape: &mut Shape,
) {
    for (band, allowed) in bands.iter().zip(xmin) {
        let range = band.from..band.to;
        let energy: f32 = xr[range.clone()].iter().map(|x| x * x).sum();
        if energy <= *allowed {
            continue;
        }
        let limit = sf_limit(ctx, band.slot);
        for _ in 0..4 {
            let step = shape.global_step - shape.amplification(band.slot);
            let over = band_noise(&xr[range.clone()], &xr34[range.clone()], step)
                .is_none_or(|noise| noise > *allowed);
            if !over {
                break;
            }
            let sf = match band.slot {
                Slot::Long(index) => &mut shape.sf_long[index],
                Slot::Short(index, w) => &mut shape.sf_short[index][w],
                Slot::Fixed(_) => break,
            };
            if i32::from(*sf) >= limit {
                break;
            }
            *sf += 1;
        }
    }
}

/// Whether some band needed more lift than its scalefactor can give.
fn clipped(ctx: &Context, bands: &[Band], steps: &[Option<i32>], shape: &Shape) -> bool {
    bands.iter().zip(steps).any(|(band, step)| {
        step.is_some_and(|step| {
            !matches!(band.slot, Slot::Fixed(_))
                && shape.global_step - shape.amplification(band.slot) > step
                && sf_limit(ctx, band.slot) > 0
        })
    })
}

/// Quantize every line at the shape's steps with `global_gain`.
/// `None` when a value is past what the tables code.
fn quantize_all(
    bands: &[Band],
    xr34: &[f32; 576],
    shape: &Shape,
    global_gain: i32,
    ix: &mut [i32; 576],
    signs: &[bool; 576],
) -> Option<()> {
    for band in bands {
        let step = global_gain - 210 - shape.amplification(band.slot);
        let factor = quantizer(step);
        for line in band.from..band.to {
            let q = quantize_line(xr34[line], factor);
            if q > MAX_VALUE {
                return None;
            }
            let q = q as i32;
            ix[line] = if signs[line] { -q } else { q };
        }
    }
    Some(())
}

/// Scalefactor bits and the `scalefac_compress` (MPEG-1) or `slen` (LSF).
fn part2(ctx: &Context, shape: &Shape, block_type: u8) -> (u32, u16, [u8; 4]) {
    let bits_for = |max: u8| -> u8 {
        if max == 0 {
            0
        } else {
            (8 - max.leading_zeros()) as u8
        }
    };
    if ctx.lsf {
        // Partition table 0 of ISO/IEC 13818-3 Table B.?: slen1, slen2 in
        // 0..=4 and slen3, slen4 in 0..=3.
        let values: Vec<u8> = if block_type == SHORT {
            (0..12)
                .flat_map(|b| shape.sf_short[b].iter().copied())
                .collect()
        } else {
            shape.sf_long[..21].to_vec()
        };
        let counts = if block_type == SHORT {
            [9usize, 9, 9, 9]
        } else {
            [6, 5, 5, 5]
        };
        let mut slen = [0u8; 4];
        let mut at = 0;
        let mut bits = 0u32;
        for (part, count) in counts.iter().enumerate() {
            let max = values[at..at + count].iter().copied().max().unwrap_or(0);
            slen[part] = bits_for(max).min(if part < 2 { 4 } else { 3 });
            bits += u32::from(slen[part]) * *count as u32;
            at += count;
        }
        let compress = ((u16::from(slen[0]) * 5 + u16::from(slen[1])) << 4)
            | (u16::from(slen[2]) << 2)
            | u16::from(slen[3]);
        return (bits, compress, slen);
    }
    let (first, second, n1, n2) = if block_type == SHORT {
        let first = (0..6).flat_map(|b| shape.sf_short[b]).max().unwrap_or(0);
        let second = (6..12).flat_map(|b| shape.sf_short[b]).max().unwrap_or(0);
        (first, second, 18u32, 18u32)
    } else {
        let first = shape.sf_long[..11].iter().copied().max().unwrap_or(0);
        let second = shape.sf_long[11..21].iter().copied().max().unwrap_or(0);
        (first, second, 11, 10)
    };
    let need1 = bits_for(first);
    let need2 = bits_for(second);
    let mut best = (u32::MAX, 15u16);
    for (compress, (&s1, &s2)) in SLEN[0].iter().zip(&SLEN[1]).enumerate() {
        if s1 >= need1 && s2 >= need2 {
            let bits = n1 * u32::from(s1) + n2 * u32::from(s2);
            if bits < best.0 {
                best = (bits, compress as u16);
            }
        }
    }
    let compress = usize::from(best.1);
    (best.0, best.1, [SLEN[0][compress], SLEN[1][compress], 0, 0])
}

/// Bits of the Huffman part and the layout that gives them.
fn part3(
    ctx: &Context,
    ix: &[i32; 576],
    block_type: u8,
    exact: bool,
) -> (u32, Division, bool, usize, usize) {
    let (big_end, count1_end) = extents(ix);
    let division = if block_type == 0 {
        if exact {
            best_division(ix, ctx.rate_table, big_end)
        } else {
            quick_division(ix, ctx.rate_table, big_end)
        }
    } else {
        let r0 = ctx.switch_region0.min(big_end);
        let (t0, b0) = choose_table(ix, 0, r0);
        let (t1, b1) = choose_table(ix, r0, big_end);
        Division {
            region0_count: 0,
            region1_count: 0,
            table_select: [t0, t1, 0],
            bits: b0 + b1,
        }
    };
    let (table_b, quad_bits) = count1(ix, big_end, count1_end);
    (
        division.bits + quad_bits,
        division,
        table_b,
        big_end,
        count1_end,
    )
}

/// The spectrum of one granule of one channel, ready to quantize.
pub(crate) struct Spectrum<'a> {
    pub xr: &'a [f32; 576],
    /// Allowed noise energy per band of [`bands`].
    pub xmin: &'a [f32],
    pub block_type: u8,
}

/// How the global gain is chosen.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Target {
    /// The finest quantization that fits `bits`.
    Budget(u32),
    /// The quantization the allowed noise asks for, but no more than `bits`.
    Noise(u32),
}

/// Quantize and lay out one granule of one channel.
pub(crate) fn code(ctx: &Context, spectrum: &Spectrum<'_>, target: Target) -> Coded {
    let block_type = spectrum.block_type;
    let bands = bands(block_type, ctx.rate_table);
    let xr = spectrum.xr;
    let mut xr34 = [0f32; 576];
    let mut signs = [false; 576];
    for (line, value) in xr.iter().enumerate() {
        let a = value.abs();
        xr34[line] = (a * a.sqrt()).sqrt();
        signs[line] = *value < 0.0;
    }
    let steps: Vec<Option<i32>> = bands
        .iter()
        .zip(spectrum.xmin)
        .map(|(band, xmin)| band_step(&xr[band.from..band.to], &xr34[band.from..band.to], *xmin))
        .collect();
    if steps.iter().all(Option::is_none) {
        return Coded::silent(block_type);
    }
    let mut shape = shape_for(ctx, &bands, &steps, block_type, false);
    if clipped(ctx, &bands, &steps, &shape) {
        shape = shape_for(ctx, &bands, &steps, block_type, true);
    }
    refine(ctx, &bands, xr, &xr34, spectrum.xmin, &mut shape);
    let (part2_bits, compress, slen) = part2(ctx, &shape, block_type);
    let limit = match target {
        Target::Budget(bits) | Target::Noise(bits) => bits.min(MAX_GRANULE_BITS),
    };
    let mut ix = [0i32; 576];
    let fits = |gain: i32, ix: &mut [i32; 576]| -> Option<u32> {
        quantize_all(&bands, &xr34, &shape, gain, ix, &signs)?;
        let (bits, ..) = part3(ctx, ix, block_type, false);
        Some(bits + part2_bits)
    };
    // The coarsest gain that keeps every value codable is where noise
    // targets start; the budget search runs from 0 to 255.
    let ideal = (shape.global_step + 210).clamp(0, 255);
    let lowest = match target {
        Target::Budget(_) => 0,
        Target::Noise(_) => ideal,
    };
    let mut gain = None;
    let fits_limit = |g: i32, ix: &mut [i32; 576]| fits(g, ix).is_some_and(|bits| bits <= limit);
    if fits_limit(lowest, &mut ix) {
        gain = Some(lowest);
    } else {
        let mut lo = lowest;
        let mut hi = 255;
        if fits_limit(hi, &mut ix) {
            // Invariant: `lo` does not fit, `hi` does.
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if fits_limit(mid, &mut ix) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            gain = Some(hi);
        }
    }
    let Some(gain) = gain else {
        // Not even the coarsest gain fits: drop the spectrum.
        return Coded::silent(block_type);
    };
    // The exact division can only be cheaper than the quick one.
    if quantize_all(&bands, &xr34, &shape, gain, &mut ix, &signs).is_none() {
        return Coded::silent(block_type);
    }
    let (part3_bits, division, table_b, big_end, count1_end) = part3(ctx, &ix, block_type, true);
    let info = GranuleInfo {
        part2_3_length: 0,
        big_values: (big_end / 2) as u16,
        global_gain: gain as u8,
        scalefac_compress: compress,
        window_switching: block_type != 0,
        block_type,
        mixed: false,
        table_select: division.table_select,
        subblock_gain: shape.subblock_gain,
        region0_count: division.region0_count,
        region1_count: division.region1_count,
        preflag: shape.preflag,
        scalefac_scale: shape.scale,
        count1_table_b: table_b,
    };
    let mut coded = Coded {
        info,
        sf_long: shape.sf_long,
        sf_short: shape.sf_short,
        slen,
        ix,
        part2_bits,
        part3_bits,
        big_end,
        count1_end,
    };
    coded.info.part2_3_length = coded.bits() as u16;
    coded
}

/// Noise-to-mask ratio per band of a coded granule, for tests and tuning.
#[cfg(test)]
pub(crate) fn noise_ratios(ctx: &Context, spectrum: &Spectrum<'_>, coded: &Coded) -> Vec<f32> {
    let shape = Shape {
        scale: coded.info.scalefac_scale,
        preflag: coded.info.preflag,
        subblock_gain: coded.info.subblock_gain,
        sf_long: coded.sf_long,
        sf_short: coded.sf_short,
        global_step: 0,
    };
    let pow = pow43();
    bands(spectrum.block_type, ctx.rate_table)
        .iter()
        .zip(spectrum.xmin)
        .map(|(band, xmin)| {
            let step = i32::from(coded.info.global_gain) - 210 - shape.amplification(band.slot);
            let back = reconstruction(step);
            let noise: f32 = (band.from..band.to)
                .map(|line| {
                    let q = coded.ix[line].unsigned_abs() as usize;
                    let d = spectrum.xr[line].abs() - pow[q] * back;
                    d * d
                })
                .sum();
            noise / xmin.max(1e-30)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{Context, Spectrum, Target, bands, code, noise_ratios};
    use crate::analysis::{NORMAL, SHORT};

    fn spectrum(seed: u32) -> [f32; 576] {
        let mut xr = [0f32; 576];
        let mut state = seed;
        for (line, slot) in xr.iter_mut().enumerate() {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let r = ((state >> 8) & 0xFFFF) as f32 / 65_536.0 - 0.5;
            *slot = r * 0.05 / (1.0 + line as f32 / 20.0);
        }
        xr[10] = 0.4;
        xr
    }

    #[test]
    fn a_noise_target_keeps_every_band_under_its_allowed_noise() {
        for block in [NORMAL, SHORT] {
            let ctx = Context::new(false, 1, block);
            let xr = spectrum(7);
            let bands = bands(block, 1);
            let xmin: Vec<f32> = bands
                .iter()
                .map(|band| {
                    let energy: f32 = xr[band.from..band.to].iter().map(|x| x * x).sum();
                    energy * 0.01
                })
                .collect();
            let spec = Spectrum {
                xr: &xr,
                xmin: &xmin,
                block_type: block,
            };
            let coded = code(&ctx, &spec, Target::Noise(4095));
            let ratios = noise_ratios(&ctx, &spec, &coded);
            for (index, ratio) in ratios.iter().enumerate() {
                // Bands without a scalefactor ride on the global gain.
                let fixed = matches!(bands[index].slot, super::Slot::Fixed(_));
                if !fixed {
                    assert!(*ratio <= 1.0001, "block {block} band {index}: {ratio}");
                }
            }
            assert!(coded.bits() <= 4095);
        }
    }

    #[test]
    fn a_budget_is_met_and_more_bits_mean_less_noise() {
        let ctx = Context::new(false, 1, NORMAL);
        let xr = spectrum(3);
        let xmin = vec![1e-9f32; 22];
        let spec = Spectrum {
            xr: &xr,
            xmin: &xmin,
            block_type: NORMAL,
        };
        let mut last = f32::MAX;
        for budget in [300, 800, 1600, 3000] {
            let coded = code(&ctx, &spec, Target::Budget(budget));
            assert!(coded.bits() <= budget, "{} > {budget}", coded.bits());
            let noise: f32 = noise_ratios(&ctx, &spec, &coded).iter().sum();
            assert!(noise < last, "{budget}: {noise} vs {last}");
            last = noise;
        }
    }
}
