//! The pitch comb filter (RFC 6716 section 4.3.7.1, `comb_filter`).
//!
//! The decoder runs it as an IIR post-filter in place; the encoder runs the
//! matching FIR pre-filter with negated gains. Over the first `overlap`
//! samples the old period, gain, and taps cross-fade into the new ones with
//! the squared overlap window.

use super::tables::WINDOW;

/// Tap gains per tapset: the reference's Q15 values.
const GAINS: [[f32; 3]; 3] = [
    [10048.0 / 32768.0, 7112.0 / 32768.0, 4248.0 / 32768.0],
    [15200.0 / 32768.0, 8784.0 / 32768.0, 0.0],
    [26208.0 / 32768.0, 3280.0 / 32768.0, 0.0],
];

/// Shortest and longest comb filter periods.
pub(crate) const MIN_PERIOD: usize = 15;
pub(crate) const MAX_PERIOD: usize = 1024;

/// One side of the cross-fade: period, gain, and tapset.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Comb {
    pub period: usize,
    pub gain: f32,
    pub tapset: usize,
}

/// Filters `len` samples of `buf` starting at `at`, in place (the decoder's
/// IIR form). Samples before `at` are the history; `at` must be at least
/// the largest period plus 2.
pub(crate) fn comb_filter_inplace(
    buf: &mut [f32],
    at: usize,
    len: usize,
    from: Comb,
    to: Comb,
    overlap: usize,
) {
    if at < MAX_PERIOD + 2 || at + len > buf.len() {
        return;
    }
    let t0 = from.period.clamp(MIN_PERIOD, MAX_PERIOD);
    let t1 = to.period.clamp(MIN_PERIOD, MAX_PERIOD);
    let tap0 = GAINS[from.tapset.min(2)];
    let tap1 = GAINS[to.tapset.min(2)];
    let (g00, g01, g02) = (
        from.gain * tap0[0],
        from.gain * tap0[1],
        from.gain * tap0[2],
    );
    let (g10, g11, g12) = (to.gain * tap1[0], to.gain * tap1[1], to.gain * tap1[2]);
    // An unchanged filter needs no cross-fade.
    let overlap = if from == to {
        0
    } else {
        overlap.min(len).min(WINDOW.len())
    };
    for (i, &w) in WINDOW.iter().enumerate().take(overlap) {
        let n = at + i;
        let f = w * w;
        let nf = 1.0 - f;
        buf[n] = buf[n]
            + nf * g00 * buf[n - t0]
            + nf * g01 * (buf[n - t0 - 1] + buf[n - t0 + 1])
            + nf * g02 * (buf[n - t0 - 2] + buf[n - t0 + 2])
            + f * g10 * buf[n - t1]
            + f * g11 * (buf[n - t1 - 1] + buf[n - t1 + 1])
            + f * g12 * (buf[n - t1 - 2] + buf[n - t1 + 2]);
    }
    if g10 == 0.0 && g11 == 0.0 && g12 == 0.0 {
        return;
    }
    for i in overlap..len {
        let n = at + i;
        buf[n] = buf[n]
            + g10 * buf[n - t1]
            + g11 * (buf[n - t1 - 1] + buf[n - t1 + 1])
            + g12 * (buf[n - t1 - 2] + buf[n - t1 + 2]);
    }
}

/// The FIR form: `y[i] = x[i] + taps of x` for `len` samples, where `x`
/// holds at least `MAX_PERIOD + 2` samples of history before index `at`.
pub(crate) fn comb_filter_fir(
    y: &mut [f32],
    x: &[f32],
    at: usize,
    len: usize,
    from: Comb,
    to: Comb,
    overlap: usize,
) {
    if at < MAX_PERIOD + 2 || at + len > x.len() || y.len() < len {
        return;
    }
    let t0 = from.period.clamp(MIN_PERIOD, MAX_PERIOD);
    let t1 = to.period.clamp(MIN_PERIOD, MAX_PERIOD);
    let tap0 = GAINS[from.tapset.min(2)];
    let tap1 = GAINS[to.tapset.min(2)];
    let (g00, g01, g02) = (
        from.gain * tap0[0],
        from.gain * tap0[1],
        from.gain * tap0[2],
    );
    let (g10, g11, g12) = (to.gain * tap1[0], to.gain * tap1[1], to.gain * tap1[2]);
    // An unchanged filter needs no cross-fade.
    let overlap = if from == to {
        0
    } else {
        overlap.min(len).min(WINDOW.len())
    };
    for i in 0..overlap {
        let n = at + i;
        let f = WINDOW[i] * WINDOW[i];
        let nf = 1.0 - f;
        y[i] = x[n]
            + nf * g00 * x[n - t0]
            + nf * g01 * (x[n - t0 - 1] + x[n - t0 + 1])
            + nf * g02 * (x[n - t0 - 2] + x[n - t0 + 2])
            + f * g10 * x[n - t1]
            + f * g11 * (x[n - t1 - 1] + x[n - t1 + 1])
            + f * g12 * (x[n - t1 - 2] + x[n - t1 + 2]);
    }
    for (i, out) in y.iter_mut().enumerate().take(len).skip(overlap) {
        let n = at + i;
        *out = x[n]
            + g10 * x[n - t1]
            + g11 * (x[n - t1 - 1] + x[n - t1 + 1])
            + g12 * (x[n - t1 - 2] + x[n - t1 + 2]);
    }
}

#[cfg(test)]
mod tests {
    use super::{Comb, MAX_PERIOD, comb_filter_fir, comb_filter_inplace};

    #[test]
    fn the_post_filter_undoes_the_pre_filter() {
        let hist = MAX_PERIOD + 2;
        let len = 960;
        let x: Vec<f32> = (0..hist + len + 2)
            .map(|i| ((i * 7919) % 1000) as f32 - 500.0)
            .collect();
        let from = Comb {
            period: 100,
            gain: 0.3,
            tapset: 1,
        };
        let to = Comb {
            period: 130,
            gain: 0.5,
            tapset: 2,
        };
        let neg = |c: Comb| Comb { gain: -c.gain, ..c };
        // Pre-filter the whole signal in the FIR form, keeping the history.
        let mut pre = x.clone();
        let mut out = vec![0f32; len];
        comb_filter_fir(&mut out, &x, hist, len, neg(from), neg(to), 120);
        pre[hist..hist + len].copy_from_slice(&out);
        // The history before `hist` was not filtered; the IIR needs the
        // original there, which the pre-filter left alone.
        comb_filter_inplace(&mut pre, hist, len, from, to, 120);
        for i in hist..hist + len {
            assert!(
                (pre[i] - x[i]).abs() < 1e-2,
                "at {i}: {} vs {}",
                pre[i],
                x[i]
            );
        }
    }
}
