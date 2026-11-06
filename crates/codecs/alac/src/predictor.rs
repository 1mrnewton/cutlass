//! The adaptive FIR predictor, after Apple's reference `dp_enc.c` and
//! `dp_dec.c`.
//!
//! Order 0 stores samples as they are. Order 31 is a first difference. Any
//! other order predicts each sample from the `order` samples before it,
//! measured against the sample `order + 1` back, and nudges each coefficient
//! by one toward the sign of the residual. Coefficient 0 weighs the newest
//! sample. Arithmetic wraps at 32 bits and every stored value is
//! sign-extended to the channel width, as in the reference.

use crate::bits::sign_extend;

/// Coefficient scale the reference encoder uses: `1 / 2^9`.
pub(crate) const DENSHIFT_DEFAULT: u32 = 9;
/// Most coefficients one channel signals: the field is five bits.
pub(crate) const MAX_COEFS: usize = 32;
/// Coefficient sets the encoder keeps per channel, one per order.
pub(crate) const SEARCH_SETS: usize = 16;

const AINIT: i32 = 38;
const BINIT: i32 = -29;
const CINIT: i32 = -2;

/// The reference's starting coefficients: a smooth three-tap shape.
pub(crate) fn init_coefs(coefs: &mut [i16], denshift: u32) {
    let den = 1i32 << denshift;
    coefs.fill(0);
    if let [a, b, c, ..] = coefs {
        *a = ((AINIT * den) >> 4) as i16;
        *b = ((BINIT * den) >> 4) as i16;
        *c = ((CINIT * den) >> 4) as i16;
    }
}

#[inline]
fn sign(value: i32) -> i32 {
    value.signum()
}

/// Turn samples into residuals. `pc[0..num]` is written; `coefs` adapts.
///
/// `input` and `pc` may be longer than `num`. As in the reference, the
/// first value and the warm-up values are written even past `num`, as far
/// as both buffers reach, so a cost estimate sees the same buffer the
/// reference would.
pub(crate) fn pc_block(
    input: &[i32],
    pc: &mut [i32],
    num: usize,
    coefs: &mut [i16],
    order: usize,
    chanbits: u32,
    denshift: u32,
) {
    let reach = input.len().min(pc.len());
    if reach == 0 {
        return;
    }
    pc[0] = input[0];
    let num = num.min(reach);
    if order == 0 {
        if num > 1 {
            pc[1..num].copy_from_slice(&input[1..num]);
        }
        return;
    }
    if order == 31 {
        for j in 1..num {
            pc[j] = sign_extend(input[j].wrapping_sub(input[j - 1]), chanbits);
        }
        return;
    }
    let warm = order.min(reach - 1);
    for j in 1..=warm {
        pc[j] = sign_extend(input[j].wrapping_sub(input[j - 1]), chanbits);
    }
    let lim = order + 1;
    if num <= lim || coefs.len() < order {
        return;
    }
    let denhalf = 1i32 << (denshift.max(1) - 1);
    let coefs = &mut coefs[..order];
    for j in lim..num {
        let top = input[j - lim];
        let history = &input[j - lim + 1..j];
        let mut sum = 0i32;
        for (k, coef) in coefs.iter().enumerate() {
            let past = history[order - 1 - k];
            sum = sum.wrapping_add(i32::from(*coef).wrapping_mul(past.wrapping_sub(top)));
        }
        let del = input[j]
            .wrapping_sub(top)
            .wrapping_sub(sum.wrapping_add(denhalf) >> denshift);
        let del = sign_extend(del, chanbits);
        pc[j] = del;
        adapt(coefs, history, top, del, order, denshift);
    }
}

/// Turn residuals back into samples. `out[0..num]` is written; `coefs` adapts.
/// `pc` and `out` are separate buffers of at least `num` values.
pub(crate) fn unpc_block(
    pc: &[i32],
    out: &mut [i32],
    coefs: &mut [i16],
    order: usize,
    chanbits: u32,
    denshift: u32,
) {
    let num = pc.len().min(out.len());
    if num == 0 {
        return;
    }
    out[0] = pc[0];
    if order == 0 {
        out[1..num].copy_from_slice(&pc[1..num]);
        return;
    }
    if order == 31 {
        for j in 1..num {
            out[j] = sign_extend(pc[j].wrapping_add(out[j - 1]), chanbits);
        }
        return;
    }
    let warm = order.min(num - 1);
    for j in 1..=warm {
        out[j] = sign_extend(pc[j].wrapping_add(out[j - 1]), chanbits);
    }
    let lim = order + 1;
    if num <= lim {
        return;
    }
    let denhalf = 1i32 << (denshift.max(1) - 1);
    let coefs = &mut coefs[..order];
    for j in lim..num {
        let top = out[j - lim];
        let (done, rest) = out.split_at_mut(j);
        let history = &done[j - lim + 1..];
        let mut sum = 0i32;
        for (k, coef) in coefs.iter().enumerate() {
            let past = history[order - 1 - k];
            sum = sum.wrapping_add(i32::from(*coef).wrapping_mul(past.wrapping_sub(top)));
        }
        let del = pc[j];
        let value = del
            .wrapping_add(top)
            .wrapping_add(sum.wrapping_add(denhalf) >> denshift);
        rest[0] = sign_extend(value, chanbits);
        adapt(coefs, history, top, del, order, denshift);
    }
}

/// In-place first difference integration, the first pass of a mode that
/// is not zero.
pub(crate) fn integrate_in_place(values: &mut [i32], chanbits: u32) {
    for j in 1..values.len() {
        values[j] = sign_extend(values[j].wrapping_add(values[j - 1]), chanbits);
    }
}

/// Nudge each coefficient toward the residual's sign, oldest tap first,
/// until the residual is accounted for.
#[inline]
fn adapt(coefs: &mut [i16], history: &[i32], top: i32, del: i32, order: usize, denshift: u32) {
    let sg = sign(del);
    if sg == 0 {
        return;
    }
    let mut del0 = del;
    for k in (0..order).rev() {
        let dd = top.wrapping_sub(history[order - 1 - k]);
        let sgn = sign(dd);
        let weight = (order - k) as i32;
        if sg > 0 {
            coefs[k] = coefs[k].wrapping_sub(sgn as i16);
            del0 = del0.wrapping_sub(weight.wrapping_mul(sgn.wrapping_mul(dd) >> denshift));
            if del0 <= 0 {
                break;
            }
        } else {
            coefs[k] = coefs[k].wrapping_add(sgn as i16);
            del0 = del0
                .wrapping_sub(weight.wrapping_mul(sgn.wrapping_neg().wrapping_mul(dd) >> denshift));
            if del0 >= 0 {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DENSHIFT_DEFAULT, init_coefs, integrate_in_place, pc_block, unpc_block};

    fn signal(len: usize) -> Vec<i32> {
        (0..len)
            .map(|i| {
                let t = i as f64;
                ((t * 0.031).sin() * 9000.0 + (t * 0.17).sin() * 1200.0) as i32 + (i as i32 % 5)
            })
            .collect()
    }

    #[test]
    fn every_order_inverts_with_the_same_starting_coefficients() {
        let input = signal(1500);
        for order in [0usize, 1, 2, 4, 8, 12, 16, 30, 31] {
            let mut enc = [0i16; 32];
            init_coefs(&mut enc, DENSHIFT_DEFAULT);
            let mut dec = enc;
            let mut pc = vec![0; input.len()];
            pc_block(
                &input,
                &mut pc,
                input.len(),
                &mut enc,
                order,
                17,
                DENSHIFT_DEFAULT,
            );
            let mut out = vec![0; input.len()];
            unpc_block(&pc, &mut out, &mut dec, order, 17, DENSHIFT_DEFAULT);
            assert_eq!(out, input, "order {order}");
            assert_eq!(enc, dec, "coefficients adapt the same way, order {order}");
        }
    }

    #[test]
    fn prediction_shrinks_a_smooth_signal() {
        let input = signal(4096);
        let mut coefs = [0i16; 32];
        init_coefs(&mut coefs, DENSHIFT_DEFAULT);
        let mut pc = vec![0; input.len()];
        pc_block(
            &input,
            &mut pc,
            input.len(),
            &mut coefs,
            8,
            17,
            DENSHIFT_DEFAULT,
        );
        let raw: i64 = input.iter().map(|v| i64::from(v.unsigned_abs())).sum();
        let residual: i64 = pc.iter().map(|v| i64::from(v.unsigned_abs())).sum();
        assert!(residual * 10 < raw, "residual {residual} raw {raw}");
    }

    #[test]
    fn residuals_wrap_to_the_channel_width() {
        // Full-scale steps in 16 bits wrap the same way in both directions.
        let input: Vec<i32> = (0..64)
            .map(|i| if i % 2 == 0 { 32_767 } else { -32_768 })
            .collect();
        let mut enc = [0i16; 32];
        init_coefs(&mut enc, DENSHIFT_DEFAULT);
        let mut dec = enc;
        let mut pc = vec![0; 64];
        pc_block(&input, &mut pc, 64, &mut enc, 4, 16, DENSHIFT_DEFAULT);
        assert!(pc.iter().all(|v| (-32_768..=32_767).contains(v)));
        let mut out = vec![0; 64];
        unpc_block(&pc, &mut out, &mut dec, 4, 16, DENSHIFT_DEFAULT);
        assert_eq!(out, input);
    }

    #[test]
    fn integration_undoes_a_first_difference() {
        let input = signal(200);
        let mut coefs = [0i16; 32];
        let mut pc = vec![0; 200];
        pc_block(&input, &mut pc, 200, &mut coefs, 31, 20, 0);
        integrate_in_place(&mut pc, 20);
        assert_eq!(pc, input);
    }
}
