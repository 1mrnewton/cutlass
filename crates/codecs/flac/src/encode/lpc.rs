//! Linear prediction: windows, autocorrelation, Levinson-Durbin, coefficient
//! quantization, and residuals (RFC 9639 section 9.2.6).
//!
//! The analysis runs in `f64`. Only the quantized integer coefficients
//! reach the bitstream, and the residual is computed from them in `i64`
//! exactly as a decoder predicts, so analysis rounding costs bits, never
//! correctness.

use super::settings::Apodization;

/// Highest LPC order the format allows.
pub(crate) const MAX_ORDER: usize = 32;

/// Appends the windows `apodization` stands for, each `len` long.
pub(crate) fn windows(apodization: Apodization, len: usize, out: &mut Vec<Vec<f64>>) {
    match apodization {
        Apodization::Tukey(p) => out.push(tukey(len, p)),
        Apodization::PartialTukey(n) => {
            let n = usize::from(n.max(1));
            for part in 0..n {
                let start = len * part / n;
                let end = len * (part + 1) / n;
                let mut window = vec![0.0; len];
                window[start..end].copy_from_slice(&tukey(end - start, 0.5));
                out.push(window);
            }
        }
        Apodization::PunchoutTukey(n) => {
            let n = usize::from(n.max(1));
            if n == 1 {
                out.push(tukey(len, 0.5));
                return;
            }
            for part in 0..n {
                let start = len * part / n;
                let end = len * (part + 1) / n;
                let mut window = tukey(len, 0.5);
                window[start..end].fill(0.0);
                out.push(window);
            }
        }
    }
}

/// A Tukey window: flat, with cosine tapers over `p` of the length.
fn tukey(len: usize, p: f64) -> Vec<f64> {
    let mut window = vec![1.0; len];
    let taper = ((p / 2.0) * len as f64) as usize;
    if taper < 2 || len < 4 {
        return window;
    }
    let taper = taper.min(len / 2);
    let span = (taper - 1) as f64;
    for n in 0..taper {
        let value = 0.5 - 0.5 * (std::f64::consts::PI * n as f64 / span).cos();
        window[n] = value;
        window[len - 1 - n] = value;
    }
    window
}

/// Autocorrelation of `samples` under `window` for lags `0..lags`.
pub(crate) fn autocorrelation(
    samples: &[i64],
    window: &[f64],
    lags: usize,
    scratch: &mut Vec<f64>,
    out: &mut [f64; MAX_ORDER + 1],
) {
    scratch.clear();
    scratch.extend(
        samples
            .iter()
            .zip(window)
            .map(|(sample, weight)| *sample as f64 * weight),
    );
    let data = scratch.as_slice();
    for (lag, slot) in out.iter_mut().enumerate().take(lags) {
        let mut sum = 0.0;
        for (a, b) in data[lag..].iter().zip(data) {
            sum += a * b;
        }
        *slot = sum;
    }
}

/// Levinson-Durbin. `coefficients[order - 1][..order]` predicts a sample
/// from the `order` before it, most recent first. `errors[order - 1]` is
/// that predictor's error energy. Returns the highest order computed, which
/// is lower than `max_order` when the error reaches zero.
pub(crate) fn levinson(
    autoc: &[f64; MAX_ORDER + 1],
    max_order: usize,
    coefficients: &mut [[f64; MAX_ORDER]; MAX_ORDER],
    errors: &mut [f64; MAX_ORDER],
) -> usize {
    let mut error = autoc[0];
    if error.is_nan() || error <= 0.0 {
        return 0;
    }
    let mut a = [0.0f64; MAX_ORDER];
    for i in 0..max_order {
        let mut acc = autoc[i + 1];
        for j in 0..i {
            acc -= a[j] * autoc[i - j];
        }
        let k = acc / error;
        if !k.is_finite() {
            return i;
        }
        let previous = a;
        a[i] = k;
        for j in 0..i {
            a[j] = previous[j] - k * previous[i - 1 - j];
        }
        error *= 1.0 - k * k;
        coefficients[i][..=i].copy_from_slice(&a[..=i]);
        errors[i] = error;
        if error <= 0.0 {
            return i + 1;
        }
    }
    max_order
}

/// The reference encoder's coefficient precision for a block length and
/// sample depth.
pub(crate) fn precision(bits: u32, block: usize) -> u32 {
    if bits < 16 {
        return (2 + bits / 2).max(5);
    }
    if bits == 16 {
        return match block {
            0..=192 => 7,
            193..=384 => 8,
            385..=576 => 9,
            577..=1152 => 10,
            1153..=2304 => 11,
            2305..=4608 => 12,
            _ => 13,
        };
    }
    match block {
        0..=384 => 13,
        385..=1152 => 14,
        _ => 15,
    }
}

/// Expected bits for each order from its error energy, and the order with
/// the fewest: residual bits plus the warm-up and coefficients.
pub(crate) fn best_order(
    errors: &[f64; MAX_ORDER],
    orders: usize,
    block: usize,
    bits: u32,
    precision: u32,
) -> usize {
    let scale = 0.5 / block as f64;
    let mut best = (f64::INFINITY, 1);
    for (index, &error) in errors.iter().enumerate().take(orders) {
        let order = index + 1;
        let per_sample = if error > 0.0 {
            (0.5 * (scale * error).log2()).max(0.0)
        } else if error < 0.0 {
            1e32
        } else {
            0.0
        };
        let total = per_sample * block.saturating_sub(order) as f64
            + order as f64 * f64::from(bits + precision);
        if total < best.0 {
            best = (total, order);
        }
    }
    best.1
}

/// Quantizes `coefficients` to `precision`-bit integers in `out`, with
/// error feedback, and returns the shift. `None` when the coefficients are
/// all zero or too large for any shift.
pub(crate) fn quantize(coefficients: &[f64], precision: u32, out: &mut [i32]) -> Option<u32> {
    let magnitude = precision - 1;
    let high = (1i64 << magnitude) - 1;
    let low = -(1i64 << magnitude);
    let peak = coefficients
        .iter()
        .fold(0.0f64, |peak, c| peak.max(c.abs()));
    if !peak.is_normal() {
        return None;
    }
    // peak = m 2^e with 1 <= m < 2.
    let exponent = i32::try_from((peak.to_bits() >> 52) & 0x7FF).unwrap_or(0) - 1023;
    let shift = (i32::try_from(magnitude).unwrap_or(0) - exponent - 1).min(15);
    if shift < -16 {
        return None;
    }
    let scale = 2f64.powi(shift);
    let mut error = 0.0;
    for (coefficient, slot) in coefficients.iter().zip(out.iter_mut()) {
        error += coefficient * scale;
        let q = (error.round() as i64).clamp(low, high);
        error -= q as f64;
        *slot = i32::try_from(q).unwrap_or(0);
    }
    Some(u32::try_from(shift.max(0)).unwrap_or(0))
}

/// The residual of `samples` under quantized `coefficients` and `shift`,
/// from sample `order` on. `false` when some residual is outside the 32
/// bits a FLAC residual may take.
pub(crate) fn residual(
    samples: &[i64],
    coefficients: &[i32],
    shift: u32,
    out: &mut Vec<i32>,
) -> bool {
    let order = coefficients.len();
    out.clear();
    if samples.len() <= order {
        return true;
    }
    out.reserve(samples.len() - order);
    let mut reversed = [0i64; MAX_ORDER];
    for (slot, c) in reversed.iter_mut().zip(coefficients.iter().rev()) {
        *slot = i64::from(*c);
    }
    let reversed = &reversed[..order];
    for window in samples.windows(order + 1) {
        let (history, current) = window.split_at(order);
        let mut sum = 0i64;
        for (c, x) in reversed.iter().zip(history) {
            sum += c * x;
        }
        let value = current[0] - (sum >> shift);
        match i32::try_from(value) {
            Ok(value) if value != i32::MIN => out.push(value),
            _ => return false,
        }
    }
    true
}

/// The residual of a fixed predictor of `order` (0 to 4).
pub(crate) fn fixed_residual(samples: &[i64], order: usize, out: &mut Vec<i32>) -> bool {
    out.clear();
    if samples.len() <= order {
        return true;
    }
    out.reserve(samples.len() - order);
    for i in order..samples.len() {
        let x = |back: usize| samples[i - back];
        let value = match order {
            0 => x(0),
            1 => x(0) - x(1),
            2 => x(0) - 2 * x(1) + x(2),
            3 => x(0) - 3 * x(1) + 3 * x(2) - x(3),
            _ => x(0) - 4 * x(1) + 6 * x(2) - 4 * x(3) + x(4),
        };
        match i32::try_from(value) {
            Ok(value) if value != i32::MIN => out.push(value),
            _ => return false,
        }
    }
    true
}

/// The fixed order whose residual magnitudes sum lowest, and that sum,
/// from one pass over the samples after the first four.
pub(crate) fn best_fixed_order(samples: &[i64]) -> (usize, u64) {
    let n = samples.len();
    if n <= 4 {
        return (0, samples.iter().map(|x| x.unsigned_abs()).sum());
    }
    let mut sums = [0u64; 5];
    let (mut a, mut b, mut c, mut d) = (
        samples[3],
        samples[3] - samples[2],
        samples[3] - 2 * samples[2] + samples[1],
        samples[3] - 3 * samples[2] + 3 * samples[1] - samples[0],
    );
    for &x in &samples[4..] {
        let e0 = x;
        let e1 = e0 - a;
        let e2 = e1 - b;
        let e3 = e2 - c;
        let e4 = e3 - d;
        sums[0] += e0.unsigned_abs();
        sums[1] += e1.unsigned_abs();
        sums[2] += e2.unsigned_abs();
        sums[3] += e3.unsigned_abs();
        sums[4] += e4.unsigned_abs();
        a = e0;
        b = e1;
        c = e2;
        d = e3;
    }
    let mut best = 0;
    for order in 1..5 {
        if sums[order] < sums[best] {
            best = order;
        }
    }
    (best, sums[best])
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_ORDER, autocorrelation, best_fixed_order, fixed_residual, levinson, quantize, residual,
        tukey,
    };

    #[test]
    fn a_sinusoid_is_predicted_by_order_two() {
        let samples: Vec<i64> = (0..4096)
            .map(|i| (10_000.0 * (i as f64 * 0.05).sin()) as i64)
            .collect();
        let window = tukey(4096, 0.5);
        let mut autoc = [0.0; MAX_ORDER + 1];
        autocorrelation(&samples, &window, 3, &mut Vec::new(), &mut autoc);
        let mut coefficients = [[0.0; MAX_ORDER]; MAX_ORDER];
        let mut errors = [0.0; MAX_ORDER];
        assert_eq!(levinson(&autoc, 2, &mut coefficients, &mut errors), 2);
        // x[n] = 2 cos(w) x[n-1] - x[n-2].
        assert!((coefficients[1][0] - 2.0 * 0.05f64.cos()).abs() < 1e-3);
        assert!((coefficients[1][1] + 1.0).abs() < 1e-3);
        let mut q = [0i32; 2];
        let shift = quantize(&coefficients[1][..2], 12, &mut q).expect("q");
        let mut out = Vec::new();
        assert!(residual(&samples, &q, shift, &mut out));
        assert!(out.iter().all(|r| r.abs() < 8), "{:?}", &out[..8]);
    }

    #[test]
    fn quantized_coefficients_stay_inside_the_precision() {
        let mut q = [0i32; 3];
        let shift = quantize(&[1.9, -0.95, 0.01], 5, &mut q).expect("q");
        assert!(q.iter().all(|c| (-16..=15).contains(c)), "{q:?}");
        assert!(shift <= 15);
        assert!(quantize(&[0.0, 0.0], 12, &mut q).is_none());
        // Huge coefficients take shift zero rather than a negative one.
        assert_eq!(quantize(&[40_000.0], 15, &mut q), Some(0));
    }

    #[test]
    fn fixed_orders_match_the_rfc_and_refuse_wide_residuals() {
        let ramp: Vec<i64> = (0..64).map(|i| i * i).collect();
        assert_eq!(best_fixed_order(&ramp), (3, 0));
        let mut out = Vec::new();
        assert!(fixed_residual(&ramp, 3, &mut out));
        assert!(out.iter().all(|r| *r == 0));
        let wide = vec![1i64 << 32, -(1i64 << 32), 1i64 << 32];
        assert!(!fixed_residual(&wide, 1, &mut out));
    }
}
