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
