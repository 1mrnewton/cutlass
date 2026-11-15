//! The encoder's dead-zone quantizer (`cutlass-vp9`'s fast path): each
//! transform coefficient to the nearest level below it plus a rounding
//! offset, with the sums its rate-distortion estimate needs.
//!
//! `floor(a + round)` is taken by rounding `a + round - 0.5` at 2^23,
//! where a float's last place is 1 and the integer sits in its low bits;
//! an exact half rounds to even. The vector kernel and its scalar twin do
//! the same float operations on each coefficient, so their levels are
//! the same; their sums differ only in the order they are added.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(test)]
mod tests;

/// The sums over the coefficients quantized.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sums {
    /// Squared error of the levels, in the coefficients' scale.
    pub err: f32,
    /// Squared size of the coefficients (the error of coding none).
    pub energy: f32,
    /// A rough rate: about 6 bits a level other than 0, one more for each
    /// step up to 16.
    pub bits: i32,
    /// Whether a level is other than 0.
    pub nonzero: bool,
}

/// A coefficient's magnitude is held below this, far above any level.
const CAP: f32 = 4_000_000.0;
const MAGIC: f32 = 8_388_608.0;

/// The quantizer of one coefficient: its level and its sums.
#[inline(always)]
fn one(v: f32, inv: f32, step: f32, rr: f32) -> (i32, f32, i32) {
    let a = (v.abs() * inv).min(CAP);
    let t = (a + rr) + MAGIC;
    let q = t.to_bits() as i32 - 0x4b00_0000;
    let e = (a - (t - MAGIC)) * step;
    (
        if v < 0.0 { -q } else { q },
        e * e,
        q.min(1) * 5 + q.min(16),
    )
}

/// Quantizes `coef` into `levels` (the shorter's length) at `1 / inv` a
/// level, `step` a level in the coefficients' scale, with offset `round`
/// (0.5 rounds to nearest).
pub fn dz(coef: &[f32], levels: &mut [i32], inv: f32, step: f32, round: f32) -> Sums {
    #[cfg(target_arch = "x86_64")]
    {
        if super::avx2() {
            // SAFETY: `avx2` checked at run time that the CPU has AVX2,
            // the only feature `avx2::dz` enables.
            #[allow(unsafe_code)]
            return unsafe { avx2::dz(coef, levels, inv, step, round) };
        }
    }
    dz_scalar(coef, levels, inv, step, round)
}

/// The scalar twin of [`dz`].
pub fn dz_scalar(coef: &[f32], levels: &mut [i32], inv: f32, step: f32, round: f32) -> Sums {
    let mut s = Sums::default();
    let mut nz = 0;
    for (l, &v) in levels.iter_mut().zip(coef) {
        let (q, e, b) = one(v, inv, step, round - 0.5);
        *l = q;
        s.err += e;
        s.energy += v * v;
        s.bits += b;
        nz |= q;
    }
    s.nonzero = nz != 0;
    s
}

/// What [`scan`] finds over a transform block's coefficients.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Scan {
    /// The coefficients' squared sizes summed: eight lanes (every eighth
    /// coefficient) each summed in order, then the lanes in order.
    pub energy: f32,
    /// One past the largest scan index of a coefficient at `th` steps or
    /// more (0 if none).
    pub last: u32,
    /// With a [`Trial`]: what the dead zone's nonzero levels could gain at
    /// most, each one's error under its value's less its band's least
    /// token cost (gains under zero taken as zero).
    pub gain: f32,
}

/// The dead zone's levels of a search trial, made by [`scan`].
#[derive(Debug)]
pub struct Trial<'a> {
    /// Each coefficient's level, `floor(a + round)` of its size `a` in
    /// steps.
    pub levels: &'a mut [u32],
    /// Each level's context energy (VP9's `energy_class`, 0 to 5).
    pub energy: &'a mut [u8],
    /// The dead zone's offset.
    pub round: f32,
    /// A step, in the coefficients' scale.
    pub step: f32,
    /// The first scan index of the last band (21, or 13 at 4×4).
    pub last_band: u16,
    /// Per band, the least a nonzero level's token costs, weighted as the
    /// error is.
    pub band_cost: [f32; 6],
}

/// Each coefficient's size in steps (`|c| · inv`, into `sizes`) and the
/// sums of [`Scan`]; with a `trial`, its dead-zone level, context energy
/// and gain too. `iscan` gives each coefficient's scan index. Takes whole
/// groups of eight (the shortest slice's length, rounded down). Sizes
/// must stay under 2^31 steps.
pub fn scan(
    coef: &[f32],
    iscan: &[u16],
    sizes: &mut [f32],
    inv: f32,
    th: f32,
    trial: Option<&mut Trial<'_>>,
) -> Scan {
    #[cfg(target_arch = "x86_64")]
    {
        if super::avx2() {
            // SAFETY: `avx2` checked at run time that the CPU has AVX2,
            // the only feature `avx2::scan` enables.
            #[allow(unsafe_code)]
            return unsafe { avx2::scan(coef, iscan, sizes, inv, th, trial) };
        }
    }
    scan_scalar(coef, iscan, sizes, inv, th, trial)
}

/// A level's context energy: VP9's `energy_class` of `min(l, 11)`.
#[inline(always)]
fn energy_of(l: u32) -> u8 {
    u8::from(l >= 1) + u8::from(l >= 2) + u8::from(l >= 3) + u8::from(l >= 5) + u8::from(l >= 11)
}

/// The band of scan index `k` (`last_band` the first of band 5).
#[inline(always)]
fn band_of(k: u16, last_band: u16) -> usize {
    usize::from(k >= 1)
        + usize::from(k >= 3)
        + usize::from(k >= 6)
        + usize::from(k >= 10)
        + usize::from(k >= last_band)
}

/// The length [`scan`] takes.
fn scan_len(coef: &[f32], iscan: &[u16], sizes: &[f32], trial: Option<&Trial<'_>>) -> usize {
    let mut n = coef.len().min(iscan.len()).min(sizes.len());
    if let Some(t) = trial {
        n = n.min(t.levels.len()).min(t.energy.len());
    }
    n / 8 * 8
}

/// The scalar twin of [`scan`]: the same values, and the same sums but
/// for the order the gains are added in.
pub fn scan_scalar(
    coef: &[f32],
    iscan: &[u16],
    sizes: &mut [f32],
    inv: f32,
    th: f32,
    mut trial: Option<&mut Trial<'_>>,
) -> Scan {
    let n = scan_len(coef, iscan, sizes, trial.as_deref());
    let mut sums = [0.0f32; 8];
    let mut out = Scan::default();
    for i in 0..n {
        let v = coef[i];
        let a = v.abs() * inv;
        sizes[i] = a;
        sums[i % 8] += v * v;
        let k = iscan[i];
        if a >= th {
            out.last = out.last.max(u32::from(k) + 1);
        }
        if let Some(t) = trial.as_deref_mut() {
            let l = (a + t.round) as u32;
            t.levels[i] = l;
            t.energy[i] = energy_of(l);
            if l != 0 {
                let (z, d) = (a * t.step, (a - l as f32) * t.step);
                let g = z * z - d * d - t.band_cost[band_of(k, t.last_band)];
                out.gain += g.max(0.0);
            }
        }
    }
    out.energy = sums.iter().sum();
    out
}
