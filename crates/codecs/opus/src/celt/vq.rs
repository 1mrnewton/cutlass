//! Pyramid vector quantization of one band shape (RFC 6716 sections
//! 4.3.4.2 and 4.3.4.3, `vq.c`).

use super::coder::Coder;
use super::cwrs;
use super::tables::SPREAD_NONE;

use std::f32::consts::PI;

/// `celt_cos_norm`: cos(pi/2 * x) as the float reference computes it.
#[inline]
fn cos_norm(x: f32) -> f32 {
    f64::from((0.5 * PI) * x).cos() as f32
}

fn exp_rotation1(x: &mut [f32], stride: usize, c: f32, s: f32) {
    let len = x.len();
    if len <= stride {
        return;
    }
    for i in 0..len - stride {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 - s * x2;
    }
    if len < 2 * stride + 1 {
        return;
    }
    for i in (0..len - 2 * stride).rev() {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 - s * x2;
    }
}

/// The spreading rotation of section 4.3.4.3; `forward` before
/// quantization, inverse after (`exp_rotation`).
pub(crate) fn exp_rotation(x: &mut [f32], forward: bool, stride: usize, k: usize, spread: usize) {
    const SPREAD_FACTOR: [usize; 3] = [15, 10, 5];
    let len = x.len();
    if 2 * k >= len || spread == SPREAD_NONE || stride == 0 {
        return;
    }
    let factor = SPREAD_FACTOR[(spread - 1).min(2)];
    let gain = len as f32 / (len + factor * k) as f32;
    let theta = 0.5 * (gain * gain);
    let c = cos_norm(theta);
    let s = cos_norm(1.0 - theta);
    let mut stride2 = 0usize;
    if len >= 8 * stride {
        stride2 = 1;
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < len {
            stride2 += 1;
        }
    }
    let sub = len / stride;
    for block in x.chunks_exact_mut(sub).take(stride) {
        if forward {
            exp_rotation1(block, 1, c, -s);
            if stride2 > 0 {
                exp_rotation1(block, stride2, s, -c);
            }
        } else {
            if stride2 > 0 {
                exp_rotation1(block, stride2, s, c);
            }
            exp_rotation1(block, 1, c, s);
        }
    }
}

/// Scales the pulses to `gain` times unit norm (`normalise_residual`).
fn normalise_residual(iy: &[i32], x: &mut [f32], ryy: f32, gain: f32) {
    let g = gain / ryy.sqrt();
    for (out, &p) in x.iter_mut().zip(iy) {
        *out = g * p as f32;
    }
}

/// Which of the `b` interleaved blocks got a pulse (`extract_collapse_mask`).
fn extract_collapse_mask(iy: &[i32], b: usize) -> u32 {
    if b <= 1 {
        return 1;
    }
    let n0 = iy.len() / b;
    let mut mask = 0u32;
    for i in 0..b {
        if iy[i * n0..(i + 1) * n0].iter().any(|&v| v != 0) {
            mask |= 1 << i;
        }
    }
    mask
}

/// Decodes `k` pulses into a unit-norm band of `gain` (`alg_unquant`).
pub(crate) fn alg_unquant(
    x: &mut [f32],
    k: usize,
    spread: usize,
    b: usize,
    coder: &mut Coder<'_, '_>,
    gain: f32,
) -> u32 {
    let n = x.len();
    let mut iy = vec![0i32; n];
    cwrs::code_pulses(coder, &mut iy, k);
    let ryy: f32 = iy.iter().map(|&v| (v * v) as f32).sum();
    normalise_residual(&iy, x, ryy.max(1.0), gain);
    exp_rotation(x, false, b, k, spread);
    extract_collapse_mask(&iy, b)
}

/// Searches the `k`-pulse codeword nearest the direction of `x` and codes it
/// (`alg_quant`). `x` is left unspecified.
pub(crate) fn alg_quant(
    x: &mut [f32],
    k: usize,
    spread: usize,
    b: usize,
    coder: &mut Coder<'_, '_>,
) -> u32 {
    let n = x.len();
    let mut iy = vec![0i32; n];
    exp_rotation(x, true, b, k, spread);
    search_pulses(x, k, &mut iy);
    cwrs::code_pulses(coder, &mut iy, k);
    extract_collapse_mask(&iy, b)
}

/// The PVQ search of `alg_quant`: a projection onto the pyramid, then a
/// greedy pulse-by-pulse completion maximizing the normalized correlation.
pub(crate) fn search_pulses(x: &mut [f32], k: usize, iy: &mut [i32]) {
    let n = x.len();
    let mut signx = vec![false; n];
    let mut y = vec![0f32; n];
    for j in 0..n {
        signx[j] = x[j] < 0.0 || (x[j] == 0.0 && x[j].is_sign_negative());
        x[j] = x[j].abs();
        if !x[j].is_finite() {
            x[j] = 0.0;
        }
        iy[j] = 0;
    }
    let mut xy = 0f32;
    let mut yy = 0f32;
    let mut pulses_left = k as i32;
    if k > n >> 1 {
        let mut sum: f32 = x.iter().sum();
        if !(sum > 1e-15 && sum < 64.0) {
            x.fill(0.0);
            x[0] = 1.0;
            sum = 1.0;
        }
        let rcp = (k as f32 - 1.0) * (1.0 / sum);
        for j in 0..n {
            iy[j] = (rcp * x[j]).floor() as i32;
            y[j] = iy[j] as f32;
            yy += y[j] * y[j];
            xy += x[j] * y[j];
            y[j] *= 2.0;
            pulses_left -= iy[j];
        }
    }
    if pulses_left > n as i32 + 3 {
        let tmp = pulses_left as f32;
        yy += tmp * tmp;
        yy += tmp * y[0];
        iy[0] += pulses_left;
        pulses_left = 0;
    }
    for _ in 0..pulses_left.max(0) {
        let mut best_id = 0usize;
        let mut best_num = -1e15f32;
        let mut best_den = 0f32;
        yy += 1.0;
        for j in 0..n {
            let rxy = xy + x[j];
            let ryy = yy + y[j];
            let rxy = rxy * rxy;
            if best_den * rxy > ryy * best_num {
                best_den = ryy;
                best_num = rxy;
                best_id = j;
            }
        }
        xy += x[best_id];
        yy += y[best_id];
        y[best_id] += 2.0;
        iy[best_id] += 1;
    }
    for j in 0..n {
        if signx[j] {
            iy[j] = -iy[j];
        }
    }
}

/// Scales `x` to norm `gain` (`renormalise_vector`).
pub(crate) fn renormalise(x: &mut [f32], gain: f32) {
    let e: f32 = 1e-15 + x.iter().map(|v| v * v).sum::<f32>();
    let g = gain / e.sqrt();
    for v in x.iter_mut() {
        *v *= g;
    }
}

/// The angle between mid and side, 0..=16384 (`stereo_itheta`).
pub(crate) fn stereo_itheta(x: &[f32], y: &[f32], stereo: bool) -> i32 {
    let mut emid = 1e-15f32;
    let mut eside = 1e-15f32;
    for (&a, &b) in x.iter().zip(y) {
        let (m, s) = if stereo {
            (0.5 * a + 0.5 * b, 0.5 * a - 0.5 * b)
        } else {
            (a, b)
        };
        emid += m * m;
        eside += s * s;
    }
    let mid = emid.sqrt();
    let side = eside.sqrt();
    let t = (0.5
        + 16384.0 * std::f32::consts::FRAC_2_PI * f64::from(side).atan2(f64::from(mid)) as f32)
        .floor();
    if t.is_finite() {
        (t as i32).clamp(0, 16384)
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::{exp_rotation, search_pulses};

    #[test]
    fn rotation_inverts_and_keeps_the_norm() {
        for &(len, stride, k, spread) in &[
            (16usize, 1usize, 2usize, 2usize),
            (64, 4, 5, 3),
            (40, 1, 3, 1),
            (176, 8, 4, 2),
        ] {
            let x0: Vec<f32> = (0..len)
                .map(|i| ((i * 37 % 11) as f32 - 5.0) / 7.0)
                .collect();
            let mut x = x0.clone();
            exp_rotation(&mut x, true, stride, k, spread);
            let n0: f32 = x0.iter().map(|v| v * v).sum();
            let n1: f32 = x.iter().map(|v| v * v).sum();
            assert!((n0 - n1).abs() < 1e-3 * n0);
            exp_rotation(&mut x, false, stride, k, spread);
            for (a, b) in x.iter().zip(&x0) {
                assert!((a - b).abs() < 1e-4, "{a} {b}");
            }
        }
    }

    #[test]
    fn search_uses_exactly_k_pulses() {
        let mut seed = 3u32;
        for n in [2usize, 3, 8, 16, 36] {
            for k in [1usize, 2, 5, 17, 40] {
                let mut x: Vec<f32> = (0..n)
                    .map(|_| {
                        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        (seed >> 9) as f32 / 4_194_304.0 - 1.0
                    })
                    .collect();
                let sign: Vec<bool> = x.iter().map(|v| *v < 0.0).collect();
                let mut iy = vec![0; n];
                search_pulses(&mut x, k, &mut iy);
                assert_eq!(
                    iy.iter().map(|v| v.unsigned_abs() as usize).sum::<usize>(),
                    k
                );
                for (p, s) in iy.iter().zip(sign) {
                    assert!(*p == 0 || (*p < 0) == s);
                }
            }
        }
        let mut silent = vec![0.0f32; 8];
        let mut iy = vec![0; 8];
        search_pulses(&mut silent, 6, &mut iy);
        assert_eq!(iy.iter().map(|v| v.unsigned_abs()).sum::<u32>(), 6);
    }
}
