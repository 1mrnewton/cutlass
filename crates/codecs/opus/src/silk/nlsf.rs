//! Normalized line spectral frequencies to LPC coefficients
//! (RFC 6716 sections 4.2.7.5.2-4.2.7.5.8; reference `NLSF_unpack.c`,
//! `NLSF_decode.c`, `NLSF_VQ_weights_laroia.c`, `NLSF_stabilize.c`,
//! `NLSF2A.c`, `LPC_inv_pred_gain.c`, `bwexpander.c`, `bwexpander_32.c`),
//! with the RFC 8251 section 6 and 7 overflow fixes.

use super::fixed::{
    clz32, div32, inverse32_varq, lshift, rshift_round, rshift_round64, smlawb, smmul, smulbb,
    smulww, sqrt_approx,
};
use super::tables::{LSF_COS_TAB_Q12, NlsfCodebook};

/// Highest LPC order.
pub(crate) const MAX_LPC_ORDER: usize = 16;

/// `NLSF_QUANT_MAX_AMPLITUDE`.
pub(crate) const NLSF_QUANT_MAX_AMPLITUDE: i32 = 4;

/// `SILK_FIX_CONST(NLSF_QUANT_LEVEL_ADJ, 10)`.
const NLSF_QUANT_LEVEL_ADJ_Q10: i32 = 102;

/// `SILK_FIX_CONST(1.0 / MAX_PREDICTION_POWER_GAIN, 30)`: a filter whose
/// inverse prediction gain is below this is too close to unstable.
const MIN_INV_PRED_GAIN_Q30: i32 = 107_374;

/// `MAX_LPC_STABILIZE_ITERATIONS`.
const MAX_LPC_STABILIZE_ITERATIONS: u32 = 16;

/// `silk_NLSF_unpack`: the second-stage table offset into
/// `cb.ec_icdf` and the backward predictor for each coefficient.
pub(crate) fn unpack(
    cb: &NlsfCodebook,
    cb1_index: usize,
) -> ([usize; MAX_LPC_ORDER], [u8; MAX_LPC_ORDER]) {
    let mut ec_ix = [0usize; MAX_LPC_ORDER];
    let mut pred_q8 = [0u8; MAX_LPC_ORDER];
    let order = cb.order;
    let stride = 2 * NLSF_QUANT_MAX_AMPLITUDE as usize + 1;
    let base = cb1_index * order / 2;
    for i in (0..order).step_by(2) {
        let entry = cb.ec_sel.get(base + i / 2).copied().unwrap_or(0);
        ec_ix[i] = usize::from((entry >> 1) & 7) * stride;
        let p0 = i + usize::from(entry & 1) * (order - 1);
        pred_q8[i] = cb.pred_q8.get(p0).copied().unwrap_or(0);
        ec_ix[i + 1] = usize::from((entry >> 5) & 7) * stride;
        let p1 = i + usize::from((entry >> 4) & 1) * (order - 1) + 1;
        pred_q8[i + 1] = cb.pred_q8.get(p1).copied().unwrap_or(0);
    }
    (ec_ix, pred_q8)
}

/// `silk_NLSF_residual_dequant`: the second-stage residual in Q10.
fn residual_dequant(
    indices: &[i32],
    pred_q8: &[u8; MAX_LPC_ORDER],
    step_q16: i32,
    order: usize,
) -> [i16; MAX_LPC_ORDER] {
    let mut x_q10 = [0i16; MAX_LPC_ORDER];
    let mut out_q10: i32 = 0;
    for i in (0..order).rev() {
        let pred_q10 = smulbb(out_q10, i32::from(pred_q8[i])) >> 8;
        out_q10 = lshift(indices.get(i).copied().unwrap_or(0), 10);
        if out_q10 > 0 {
            out_q10 -= NLSF_QUANT_LEVEL_ADJ_Q10;
        } else if out_q10 < 0 {
            out_q10 += NLSF_QUANT_LEVEL_ADJ_Q10;
        }
        out_q10 = smlawb(pred_q10, out_q10, step_q16);
        x_q10[i] = out_q10 as i16;
    }
    x_q10
}

/// `silk_NLSF_VQ_weights_laroia` with `NLSF_W_Q = 2`.
fn laroia_weights(nlsf_q15: &[i16]) -> [i16; MAX_LPC_ORDER] {
    let d = nlsf_q15.len();
    let mut w = [0i16; MAX_LPC_ORDER];
    let inv = |gap: i32| div32(1 << 17, gap.max(1));
    let at = |i: usize| i32::from(nlsf_q15[i]);
    let mut tmp1 = inv(at(0));
    let mut tmp2 = inv(at(1) - at(0));
    w[0] = (tmp1 + tmp2).min(i32::from(i16::MAX)) as i16;
    let mut k = 1;
    while k + 1 < d {
        tmp1 = inv(at(k + 1) - at(k));
        w[k] = (tmp1 + tmp2).min(i32::from(i16::MAX)) as i16;
        tmp2 = inv(at(k + 2) - at(k + 1));
        w[k + 1] = (tmp1 + tmp2).min(i32::from(i16::MAX)) as i16;
        k += 2;
    }
    tmp1 = inv((1 << 15) - at(d - 1));
    w[d - 1] = (tmp1 + tmp2).min(i32::from(i16::MAX)) as i16;
    w
}

/// `silk_NLSF_decode`: the quantized NLSF vector for one frame.
/// `indices[0]` is the first-stage index and `indices[1..=order]` the
/// residual indices.
pub(crate) fn decode(cb: &NlsfCodebook, indices: &[i32]) -> [i16; MAX_LPC_ORDER] {
    let order = cb.order;
    let mut nlsf = [0i16; MAX_LPC_ORDER];
    let cb1_index = usize::try_from(indices.first().copied().unwrap_or(0)).unwrap_or(0);
    for (i, out) in nlsf.iter_mut().enumerate().take(order) {
        let v = cb
            .cb1_nlsf_q8
            .get(cb1_index * order + i)
            .copied()
            .unwrap_or(0);
        *out = (i16::from(v)) << 7;
    }
    let (_, pred_q8) = unpack(cb, cb1_index);
    let res_q10 = residual_dequant(
        indices.get(1..).unwrap_or(&[]),
        &pred_q8,
        cb.quant_step_size_q16,
        order,
    );
    let w = laroia_weights(&nlsf[..order]);
    for i in 0..order {
        let w_q9 = sqrt_approx(lshift(i32::from(w[i]), 16));
        let v = i32::from(nlsf[i]) + div32(lshift(i32::from(res_q10[i]), 14), w_q9);
        nlsf[i] = v.clamp(0, 32767) as i16;
    }
    stabilize(&mut nlsf[..order], cb.delta_min_q15);
    nlsf
}

/// `silk_NLSF_stabilize`: moves the NLSFs apart until every gap is at
/// least `delta_min_q15` (which has one more entry than `nlsf_q15`).
pub(crate) fn stabilize(nlsf_q15: &mut [i16], delta_min_q15: &[i16]) {
    const MAX_LOOPS: usize = 20;
    let l = nlsf_q15.len();
    if l == 0 || delta_min_q15.len() < l + 1 {
        return;
    }
    let dmin = |i: usize| i32::from(delta_min_q15[i]);
    for _ in 0..MAX_LOOPS {
        // The smallest distance and where it is.
        let mut min_diff = i32::from(nlsf_q15[0]) - dmin(0);
        let mut at = 0;
        for i in 1..l {
            let diff = i32::from(nlsf_q15[i]) - (i32::from(nlsf_q15[i - 1]) + dmin(i));
            if diff < min_diff {
                min_diff = diff;
                at = i;
            }
        }
        let diff = (1 << 15) - (i32::from(nlsf_q15[l - 1]) + dmin(l));
        if diff < min_diff {
            min_diff = diff;
            at = l;
        }
        if min_diff >= 0 {
            return;
        }
        if at == 0 {
            nlsf_q15[0] = delta_min_q15[0];
        } else if at == l {
            nlsf_q15[l - 1] = ((1 << 15) - dmin(l)) as i16;
        } else {
            let mut min_center: i32 = (0..at).map(dmin).sum();
            min_center += dmin(at) >> 1;
            let mut max_center: i32 = 1 << 15;
            for k in (at + 1..=l).rev() {
                max_center -= dmin(k);
            }
            max_center -= dmin(at) >> 1;
            let sum = i32::from(nlsf_q15[at - 1]) + i32::from(nlsf_q15[at]);
            let center = limit(rshift_round(sum, 1), min_center, max_center) as i16;
            nlsf_q15[at - 1] = (i32::from(center) - (dmin(at) >> 1)) as i16;
            nlsf_q15[at] = (i32::from(nlsf_q15[at - 1]) + dmin(at)) as i16;
        }
    }
    // Fallback after MAX_LOOPS: sort, then push apart from both ends.
    nlsf_q15.sort_unstable();
    nlsf_q15[0] = nlsf_q15[0].max(delta_min_q15[0]);
    for i in 1..l {
        // RFC 8251 section 7: saturate so a huge NLSF cannot wrap.
        let floor = nlsf_q15[i - 1].saturating_add(delta_min_q15[i]);
        nlsf_q15[i] = nlsf_q15[i].max(floor);
    }
    nlsf_q15[l - 1] = i32::from(nlsf_q15[l - 1]).min((1 << 15) - dmin(l)) as i16;
    for i in (0..l - 1).rev() {
        let ceil = i32::from(nlsf_q15[i + 1]) - dmin(i + 1);
        nlsf_q15[i] = i32::from(nlsf_q15[i]).min(ceil) as i16;
    }
}

/// `silk_LIMIT`: clamps `a` between two bounds given in either order.
fn limit(a: i32, l1: i32, l2: i32) -> i32 {
    if l1 > l2 {
        a.clamp(l2, l1)
    } else {
        a.clamp(l1, l2)
    }
}

/// `silk_bwexpander`: chirps a Q12 filter by `chirp_q16`.
pub(crate) fn bwexpander(ar: &mut [i16], mut chirp_q16: i32) {
    let Some((last, head)) = ar.split_last_mut() else {
        return;
    };
    let chirp_minus_one_q16 = chirp_q16 - 65536;
    for a in head {
        *a = rshift_round(chirp_q16.wrapping_mul(i32::from(*a)), 16) as i16;
        chirp_q16 += rshift_round(chirp_q16.wrapping_mul(chirp_minus_one_q16), 16);
    }
    *last = rshift_round(chirp_q16.wrapping_mul(i32::from(*last)), 16) as i16;
}

/// `silk_bwexpander_32`: chirps a 32-bit filter by `chirp_q16`.
fn bwexpander_32(ar: &mut [i32], mut chirp_q16: i32) {
    let Some((last, head)) = ar.split_last_mut() else {
        return;
    };
    let chirp_minus_one_q16 = chirp_q16 - 65536;
    for a in head {
        *a = smulww(chirp_q16, *a);
        chirp_q16 += rshift_round(chirp_q16.wrapping_mul(chirp_minus_one_q16), 16);
    }
    *last = smulww(chirp_q16, *last);
}

/// Q of the 32-bit polynomial arithmetic in NLSF2A.
const QA_NLSF: u32 = 16;

/// `silk_NLSF2A_find_poly`: the polynomial whose roots are the cosines
/// `c_lsf[0], c_lsf[2], ...` (every other entry).
fn find_poly(out: &mut [i32; MAX_LPC_ORDER / 2 + 1], c_lsf: &[i32], dd: usize) {
    let mul = |a: i32, b: i32| rshift_round64(i64::from(a) * i64::from(b), QA_NLSF) as i32;
    out[0] = 1 << QA_NLSF;
    out[1] = c_lsf[0].wrapping_neg();
    for k in 1..dd {
        let ftmp = c_lsf[2 * k];
        out[k + 1] = lshift(out[k - 1], 1).wrapping_sub(mul(ftmp, out[k]));
        for n in (2..=k).rev() {
            out[n] = out[n].wrapping_add(out[n - 2].wrapping_sub(mul(ftmp, out[n - 1])));
        }
        out[1] = out[1].wrapping_sub(ftmp);
    }
}

/// `silk_NLSF2A`: monic whitening filter coefficients in Q12 for an NLSF
/// vector of order 10 or 16.
pub(crate) fn nlsf2a(a_q12: &mut [i16], nlsf_q15: &[i16]) {
    // Orderings that the reference found most accurate for find_poly.
    const ORDERING16: [usize; 16] = [0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1];
    const ORDERING10: [usize; 10] = [0, 9, 6, 3, 4, 5, 8, 1, 2, 7];
    let d = nlsf_q15.len().min(a_q12.len()).min(MAX_LPC_ORDER);
    if d < 2 {
        return;
    }
    let ordering: &[usize] = if d == 16 { &ORDERING16 } else { &ORDERING10 };
    let mut cos_lsf_qa = [0i32; MAX_LPC_ORDER];
    for (k, &v) in nlsf_q15.iter().enumerate().take(d) {
        let nlsf = i32::from(v).max(0);
        let f_int = (nlsf >> 8).min(127) as usize;
        let f_frac = nlsf - ((f_int as i32) << 8);
        let cos_val = i32::from(LSF_COS_TAB_Q12[f_int]);
        let delta = i32::from(LSF_COS_TAB_Q12[f_int + 1]) - cos_val;
        let slot = ordering.get(k).copied().unwrap_or(k);
        cos_lsf_qa[slot] = rshift_round((cos_val << 8) + delta * f_frac, 20 - QA_NLSF);
    }
    let dd = d / 2;
    let mut p = [0i32; MAX_LPC_ORDER / 2 + 1];
    let mut q = [0i32; MAX_LPC_ORDER / 2 + 1];
    find_poly(&mut p, &cos_lsf_qa[..d], dd);
    find_poly(&mut q, &cos_lsf_qa[1..d], dd);
    let mut a32_qa1 = [0i32; MAX_LPC_ORDER];
    for k in 0..dd {
        let ptmp = p[k + 1].wrapping_add(p[k]);
        let qtmp = q[k + 1].wrapping_sub(q[k]);
        a32_qa1[k] = qtmp.wrapping_neg().wrapping_sub(ptmp);
        a32_qa1[d - k - 1] = qtmp.wrapping_sub(ptmp);
    }
    let a32 = &mut a32_qa1[..d];
    let shift = QA_NLSF + 1 - 12;
    // Limit the coefficients so they fit in 16 bits.
    let mut i = 0;
    while i < 10 {
        let mut maxabs = 0i32;
        let mut idx = 0usize;
        for (k, &v) in a32.iter().enumerate() {
            let absval = v.wrapping_abs();
            if absval > maxabs {
                maxabs = absval;
                idx = k;
            }
        }
        maxabs = rshift_round(maxabs, shift);
        if maxabs > i32::from(i16::MAX) {
            maxabs = maxabs.min(163_838);
            let num = lshift(maxabs - i32::from(i16::MAX), 14);
            let den = (maxabs * (idx as i32 + 1)) >> 2;
            let sc_q16 = 65470 - div32(num, den);
            bwexpander_32(a32, sc_q16);
        } else {
            break;
        }
        i += 1;
    }
    if i == 10 {
        for k in 0..d {
            a_q12[k] = super::fixed::sat16(rshift_round(a32[k], shift));
            a32[k] = i32::from(a_q12[k]) << shift;
        }
    } else {
        for k in 0..d {
            a_q12[k] = rshift_round(a32[k], shift) as i16;
        }
    }
    for i in 0..MAX_LPC_STABILIZE_ITERATIONS {
        if lpc_inverse_pred_gain(&a_q12[..d]) < MIN_INV_PRED_GAIN_Q30 {
            // Too close to unstable: expand the bandwidth of the unscaled
            // coefficients and measure again.
            bwexpander_32(a32, 65536 - (2 << i));
            for k in 0..d {
                a_q12[k] = rshift_round(a32[k], shift) as i16;
            }
        } else {
            break;
        }
    }
}

/// Q of `LPC_inverse_pred_gain_QA`.
const QA_GAIN: u32 = 24;
/// `SILK_FIX_CONST(0.99975, QA)`.
const A_LIMIT: i32 = 16_773_022;

/// `LPC_inverse_pred_gain_QA`: the inverse prediction gain in Q30, or 0
/// for an unstable filter. Runs the reflection-coefficient recursion.
fn inverse_pred_gain_qa(a_qa: &[i32; MAX_LPC_ORDER], order: usize) -> i32 {
    let mut cur = *a_qa;
    let mut inv_gain_q30: i32 = 1 << 30;
    for k in (1..order).rev() {
        if cur[k] > A_LIMIT || cur[k] < -A_LIMIT {
            return 0;
        }
        let rc_q31 = lshift(cur[k], 31 - QA_GAIN).wrapping_neg();
        let rc_mult1_q30 = (1i32 << 30).wrapping_sub(smmul(rc_q31, rc_q31));
        let mult2q = 32 - clz32(rc_mult1_q30.wrapping_abs());
        let rc_mult2 = inverse32_varq(rc_mult1_q30, mult2q + 30);
        inv_gain_q30 = lshift(smmul(inv_gain_q30, rc_mult1_q30), 2);
        let mut next = [0i32; MAX_LPC_ORDER];
        for n in 0..k {
            let frac = rshift_round64(i64::from(cur[k - n - 1]) * i64::from(rc_q31), 31) as i32;
            let tmp_qa = cur[n].saturating_sub(frac);
            // RFC 8251 section 6: a value past 32 bits means unstable.
            let tmp64 = rshift_round64(
                i64::from(tmp_qa) * i64::from(rc_mult2),
                mult2q.clamp(1, 63) as u32,
            );
            let Ok(v) = i32::try_from(tmp64) else {
                return 0;
            };
            next[n] = v;
        }
        cur = next;
    }
    if cur[0] > A_LIMIT || cur[0] < -A_LIMIT {
        return 0;
    }
    let rc_q31 = lshift(cur[0], 31 - QA_GAIN).wrapping_neg();
    let rc_mult1_q30 = (1i32 << 30).wrapping_sub(smmul(rc_q31, rc_q31));
    lshift(smmul(inv_gain_q30, rc_mult1_q30), 2)
}

/// `silk_LPC_inverse_pred_gain`: the inverse prediction gain of a Q12
/// filter in Q30, or 0 when the filter is unstable.
pub(crate) fn lpc_inverse_pred_gain(a_q12: &[i16]) -> i32 {
    let order = a_q12.len().min(MAX_LPC_ORDER);
    let mut a_qa = [0i32; MAX_LPC_ORDER];
    let mut dc_resp: i32 = 0;
    for (k, &a) in a_q12.iter().take(order).enumerate() {
        dc_resp += i32::from(a);
        a_qa[k] = i32::from(a) << (QA_GAIN - 12);
    }
    if dc_resp >= 4096 || order == 0 {
        return 0;
    }
    inverse_pred_gain_qa(&a_qa, order)
}

#[cfg(test)]
mod tests {
    use super::super::tables::{NLSF_CB_NB_MB, NLSF_CB_WB};
    use super::{
        MAX_LPC_ORDER, MIN_INV_PRED_GAIN_Q30, bwexpander, decode, lpc_inverse_pred_gain, nlsf2a,
        stabilize, unpack,
    };
    use std::f64::consts::PI;

    /// The direct-form polynomial for an NLSF vector, in double precision:
    /// P(z) and Q(z) from the cosines, then A = (P + Q) / 2.
    fn nlsf2a_double(nlsf_q15: &[i16]) -> Vec<f64> {
        let d = nlsf_q15.len();
        let mul = |poly: &[f64], c: f64| {
            // poly * (1 - 2 c z^-1 + z^-2)
            let mut out = vec![0.0; poly.len() + 2];
            for (i, &p) in poly.iter().enumerate() {
                out[i] += p;
                out[i + 1] -= 2.0 * c * p;
                out[i + 2] += p;
            }
            out
        };
        let mut p = vec![1.0];
        let mut q = vec![1.0];
        for (k, &f) in nlsf_q15.iter().enumerate() {
            let c = (PI * f64::from(f) / 32768.0).cos();
            if k % 2 == 0 {
                p = mul(&p, c);
            } else {
                q = mul(&q, c);
            }
        }
        // P gets (1 + z^-1), Q gets (1 - z^-1).
        let mut a = vec![0.0; d + 2];
        for i in 0..p.len() {
            a[i] += p[i];
            a[i + 1] += p[i];
        }
        for i in 0..q.len() {
            a[i] += q[i];
            a[i + 1] -= q[i];
        }
        // A(z) = 1 - sum a_k z^-k with a_k = -(coefficient of z^-k) / 2.
        (1..=d).map(|k| -a[k] / 2.0).collect()
    }

    #[test]
    fn nlsf2a_matches_a_double_precision_reference() {
        let wb: [i16; 16] = [
            1200, 2600, 4300, 6100, 7900, 9800, 11800, 13700, 15600, 17600, 19700, 21700, 23800,
            25900, 28000, 30100,
        ];
        let nb: [i16; 10] = [
            2000, 4800, 7600, 10500, 13300, 16200, 19100, 22100, 25200, 28400,
        ];
        for nlsf in [&wb[..], &nb[..]] {
            let mut a_q12 = [0i16; MAX_LPC_ORDER];
            nlsf2a(&mut a_q12[..nlsf.len()], nlsf);
            let exact = nlsf2a_double(nlsf);
            for (k, &e) in exact.iter().enumerate() {
                let got = f64::from(a_q12[k]) / 4096.0;
                // The table is a piecewise-linear cosine, so allow its error.
                assert!(
                    (got - e).abs() < 0.02,
                    "order {} a[{k}]: {got} vs {e}",
                    nlsf.len()
                );
            }
        }
    }

    #[test]
    fn stabilize_orders_and_spaces_the_vector() {
        let dmin = &NLSF_CB_NB_MB.delta_min_q15;
        // Out of order, touching, and past both ends.
        let mut nlsf: [i16; 10] = [
            -5, 9000, 8000, 8000, 12000, 30000, 20000, 20001, 32767, 32767,
        ];
        stabilize(&mut nlsf, dmin);
        assert!(nlsf[0] >= dmin[0]);
        for i in 1..10 {
            assert!(
                i32::from(nlsf[i]) - i32::from(nlsf[i - 1]) >= i32::from(dmin[i]),
                "{nlsf:?}"
            );
        }
        assert!(i32::from(nlsf[9]) <= 32768 - i32::from(dmin[10]));
        // An already valid vector is left alone.
        let valid: [i16; 10] = [
            2000, 4800, 7600, 10500, 13300, 16200, 19100, 22100, 25200, 28400,
        ];
        let mut copy = valid;
        stabilize(&mut copy, dmin);
        assert_eq!(copy, valid);
        // RFC 8251 section 7: extreme values do not wrap.
        let mut extreme = [i16::MAX; 16];
        stabilize(&mut extreme, NLSF_CB_WB.delta_min_q15);
        for i in 1..16 {
            assert!(extreme[i] > extreme[i - 1]);
        }
    }

    #[test]
    fn inverse_prediction_gain_rejects_unstable_filters() {
        // A single pole at 0.5: gain 1 / (1 - 0.25).
        let mut a = [0i16; 10];
        a[0] = 2048;
        let g = f64::from(lpc_inverse_pred_gain(&a)) / f64::from(1 << 30);
        assert!((g - 0.75).abs() < 1e-3, "{g}");
        // A pole outside the unit circle.
        a[0] = 4096 + 400;
        assert_eq!(lpc_inverse_pred_gain(&a), 0);
        // A second-order resonator with |pole| = 1.01: 2r cos(w), -r^2.
        let r: f64 = 1.01;
        let mut b = [0i16; 10];
        b[0] = (2.0 * r * 0.3f64.cos() * 4096.0) as i16;
        b[1] = (-r * r * 4096.0) as i16;
        assert_eq!(lpc_inverse_pred_gain(&b), 0);
        // RFC 8251 section 6: coefficients that overflow 32 bits are unstable,
        // not wrapped.
        let huge = [
            i16::MAX,
            i16::MIN,
            i16::MAX,
            i16::MIN,
            i16::MAX,
            i16::MIN,
            0,
            0,
            0,
            0,
        ];
        assert_eq!(lpc_inverse_pred_gain(&huge), 0);
        let all = [i16::MIN; 16];
        let _ = lpc_inverse_pred_gain(&all);
    }

    #[test]
    fn nlsf2a_output_is_stable_for_crowded_frequencies() {
        // Nearly coincident NLSFs give a resonant filter that NLSF2A must
        // bandwidth-expand into stability.
        let nlsf: [i16; 16] = [
            100, 140, 180, 220, 4000, 4040, 4080, 4120, 9000, 9040, 9080, 9120, 20000, 20040,
            20080, 20120,
        ];
        let mut a_q12 = [0i16; 16];
        nlsf2a(&mut a_q12, &nlsf);
        assert!(lpc_inverse_pred_gain(&a_q12) >= MIN_INV_PRED_GAIN_Q30);
    }

    #[test]
    fn decode_gives_a_stable_sorted_vector_for_every_first_stage_index() {
        for cb in [&NLSF_CB_NB_MB, &NLSF_CB_WB] {
            for cb1 in 0..cb.n_vectors {
                for residual in [-10, 0, 10] {
                    let mut indices = [residual; 17];
                    indices[0] = cb1 as i32;
                    let nlsf = decode(cb, &indices[..=cb.order]);
                    for i in 1..cb.order {
                        assert!(nlsf[i] > nlsf[i - 1]);
                    }
                    let (ec_ix, _) = unpack(cb, cb1);
                    assert!(ec_ix.iter().all(|&ix| ix + 9 <= cb.ec_icdf.len()));
                }
            }
        }
    }

    #[test]
    fn bandwidth_expansion_scales_each_tap_by_a_power_of_the_chirp() {
        let mut a = [4096i16; 4];
        bwexpander(&mut a, 32768);
        assert_eq!(a, [2048, 1024, 512, 256]);
    }
}
