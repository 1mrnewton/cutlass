//! Pitch analysis and LPC helpers (`pitch.c`, `celt_lpc.c`), used by the
//! decoder's loss concealment and the encoder's pre-filter.

/// The two best normalized correlations (`find_best_pitch`).
fn find_best_pitch(xcorr: &[f32], y: &[f32], len: usize, max_pitch: usize) -> [usize; 2] {
    let mut syy = 1f32;
    let mut best_num = [-1f32; 2];
    let mut best_den = [0f32; 2];
    let mut best = [0usize, 1];
    for v in y.iter().take(len) {
        syy += v * v;
    }
    for i in 0..max_pitch {
        let xc = xcorr.get(i).copied().unwrap_or(0.0);
        if xc > 0.0 {
            let num = xc * xc;
            if num * best_den[1] > best_num[1] * syy {
                if num * best_den[0] > best_num[0] * syy {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best[1] = best[0];
                    best_num[0] = num;
                    best_den[0] = syy;
                    best[0] = i;
                } else {
                    best_num[1] = num;
                    best_den[1] = syy;
                    best[1] = i;
                }
            }
        }
        let a = y.get(i + len).copied().unwrap_or(0.0);
        let b = y.get(i).copied().unwrap_or(0.0);
        syy += a * a - b * b;
        syy = syy.max(1.0);
    }
    best
}

/// Autocorrelation of `x` for lags `0..=lag`, with the ends windowed by
/// `window` over `overlap` samples (`_celt_autocorr`).
pub(crate) fn autocorr(x: &[f32], ac: &mut [f32], window: Option<&[f32]>, lag: usize) {
    let n = x.len();
    let mut xx = x.to_vec();
    if let Some(w) = window {
        let overlap = w.len().min(n / 2);
        for i in 0..overlap {
            xx[i] = x[i] * w[i];
            xx[n - i - 1] = x[n - i - 1] * w[i];
        }
    }
    for l in 0..=lag.min(ac.len().saturating_sub(1)) {
        let mut d = 0f32;
        for i in l..n {
            d += xx[i] * xx[i - l];
        }
        ac[l] = d;
    }
    if let Some(a0) = ac.first_mut() {
        *a0 += 10.0;
    }
}

/// Levinson-Durbin recursion (`_celt_lpc`): `lpc` gets `ac.len() - 1`
/// coefficients.
pub(crate) fn lpc(lpc: &mut [f32], ac: &[f32]) {
    let p = lpc.len();
    lpc.fill(0.0);
    let Some(&ac0) = ac.first() else {
        return;
    };
    let mut error = ac0;
    if ac0 == 0.0 {
        return;
    }
    for i in 0..p {
        let mut rr = 0f32;
        for j in 0..i {
            rr += lpc[j] * ac[i - j];
        }
        rr += ac[i + 1];
        let r = -rr / error;
        lpc[i] = r;
        for j in 0..(i + 1) >> 1 {
            let t1 = lpc[j];
            let t2 = lpc[i - 1 - j];
            lpc[j] = t1 + r * t2;
            lpc[i - 1 - j] = t2 + r * t1;
        }
        error -= r * r * error;
        if error < 0.001 * ac0 {
            break;
        }
    }
}

/// FIR filter with memory, in place (`celt_fir`).
pub(crate) fn fir(x: &mut [f32], num: &[f32], mem: &mut [f32]) {
    let ord = num.len().min(mem.len());
    for v in x.iter_mut() {
        let mut sum = *v;
        for j in 0..ord {
            sum += num[j] * mem[j];
        }
        for j in (1..ord).rev() {
            mem[j] = mem[j - 1];
        }
        if ord > 0 {
            mem[0] = *v;
        }
        *v = sum;
    }
}

/// IIR filter with memory, in place (`celt_iir`).
pub(crate) fn iir(x: &mut [f32], den: &[f32], mem: &mut [f32]) {
    let ord = den.len().min(mem.len());
    for v in x.iter_mut() {
        let mut sum = *v;
        for j in 0..ord {
            sum -= den[j] * mem[j];
        }
        for j in (1..ord).rev() {
            mem[j] = mem[j - 1];
        }
        if ord > 0 {
            mem[0] = sum;
        }
        *v = sum;
    }
}

/// Half-rate, whitened sum of the channels for the pitch search
/// (`pitch_downsample`). `x_lp` gets `len / 2` samples.
pub(crate) fn downsample(x: &[&[f32]], x_lp: &mut [f32], len: usize) {
    let half = len >> 1;
    x_lp[..half].fill(0.0);
    for ch in x {
        if ch.len() < len {
            continue;
        }
        for i in 1..half {
            x_lp[i] += 0.5 * (0.5 * (ch[2 * i - 1] + ch[2 * i + 1]) + ch[2 * i]);
        }
        x_lp[0] += 0.5 * (0.5 * ch[1] + ch[0]);
    }
    let mut ac = [0f32; 5];
    autocorr(&x_lp[..half], &mut ac, None, 4);
    ac[0] *= 1.0001;
    for (i, a) in ac.iter_mut().enumerate().skip(1) {
        let f = 0.008 * i as f32;
        *a -= *a * f * f;
    }
    let mut coefs = [0f32; 4];
    lpc(&mut coefs, &ac);
    let mut tmp = 1.0f32;
    for c in coefs.iter_mut() {
        tmp *= 0.9;
        *c *= tmp;
    }
    let mut mem = [0f32; 4];
    fir(&mut x_lp[..half], &coefs, &mut mem);
    let mut mem = [0f32; 1];
    fir(&mut x_lp[..half], &[0.8], &mut mem);
}

/// The lag in `0..max_pitch` that best aligns `y` with `x_lp`
/// (`pitch_search`): a 4x-decimated search, then a 2x refinement around
/// the two best candidates, then a half-sample tweak.
pub(crate) fn search(x_lp: &[f32], y: &[f32], len: usize, max_pitch: usize) -> usize {
    let lag = len + max_pitch;
    let x4: Vec<f32> = (0..len >> 2)
        .map(|j| x_lp.get(2 * j).copied().unwrap_or(0.0))
        .collect();
    let y4: Vec<f32> = (0..lag >> 2)
        .map(|j| y.get(2 * j).copied().unwrap_or(0.0))
        .collect();
    let mut xcorr = vec![0f32; max_pitch >> 1];
    for (i, xc) in xcorr.iter_mut().enumerate().take(max_pitch >> 2) {
        let sum: f32 = x4
            .iter()
            .enumerate()
            .map(|(j, &a)| a * y4.get(i + j).copied().unwrap_or(0.0))
            .sum();
        *xc = sum.max(-1.0);
    }
    let best = find_best_pitch(&xcorr, &y4, len >> 2, max_pitch >> 2);
    for (i, xc) in xcorr.iter_mut().enumerate() {
        *xc = 0.0;
        let near = |b: usize| (i as i64 - 2 * b as i64).abs() <= 2;
        if !near(best[0]) && !near(best[1]) {
            continue;
        }
        let mut sum = 0f32;
        for j in 0..len >> 1 {
            sum += x_lp.get(j).copied().unwrap_or(0.0) * y.get(i + j).copied().unwrap_or(0.0);
        }
        *xc = sum.max(-1.0);
    }
    let best = find_best_pitch(&xcorr, y, len >> 1, max_pitch >> 1);
    let mut offset = 0i64;
    if best[0] > 0 && best[0] + 1 < max_pitch >> 1 {
        let a = xcorr[best[0] - 1];
        let b = xcorr[best[0]];
        let c = xcorr[best[0] + 1];
        if c - a > 0.7 * (b - a) {
            offset = 1;
        } else if a - c > 0.7 * (b - c) {
            offset = -1;
        }
    }
    (2 * best[0] as i64 - offset).max(0) as usize
}

const SECOND_CHECK: [i32; 16] = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];

/// Moves a pitch estimate off multiples of the true period
/// (`remove_doubling`). `x` holds `maxperiod` samples of history then `n`
/// new ones, at half rate. Returns the period (full rate) and its gain.
pub(crate) fn remove_doubling(
    x: &[f32],
    maxperiod: i32,
    minperiod: i32,
    n: i32,
    t0_in: i32,
    prev_period: i32,
    prev_gain: f32,
) -> (i32, f32) {
    let minperiod0 = minperiod;
    let maxperiod = maxperiod / 2;
    let minperiod = minperiod / 2;
    let mut t0 = t0_in / 2;
    let prev_period = prev_period / 2;
    let n = n / 2;
    if t0 >= maxperiod {
        t0 = maxperiod - 1;
    }
    let base = maxperiod as usize;
    let at = |i: i32| -> f32 {
        x.get((base as i64 + i64::from(i)) as usize)
            .copied()
            .unwrap_or(0.0)
    };
    let corr = |t: i32| -> (f32, f32) {
        let mut xy = 0f32;
        let mut yy = 0f32;
        for i in 0..n {
            xy += at(i) * at(i - t);
            yy += at(i - t) * at(i - t);
        }
        (xy, yy)
    };
    let mut xx = 0f32;
    for i in 0..n {
        xx += at(i) * at(i);
    }
    let (xy, yy) = corr(t0);
    let mut best_xy = xy;
    let mut best_yy = yy;
    let g0 = xy / (1.0 + xx * yy).sqrt();
    let mut g = g0;
    let mut t = t0;
    for k in 2..=15i32 {
        let t1 = (2 * t0 + k) / (2 * k);
        if t1 < minperiod {
            break;
        }
        let t1b = if k == 2 {
            if t1 + t0 > maxperiod { t0 } else { t0 + t1 }
        } else {
            (2 * SECOND_CHECK[k as usize] * t0 + k) / (2 * k)
        };
        let (xy1, yy1) = corr(t1);
        let (xy2, yy2) = corr(t1b);
        let xy = xy1 + xy2;
        let yy = yy1 + yy2;
        let g1 = xy / (1.0 + 2.0 * xx * yy).sqrt();
        let cont = if (t1 - prev_period).abs() <= 1 {
            prev_gain
        } else if (t1 - prev_period).abs() <= 2 && 5 * k * k < t0 {
            0.5 * prev_gain
        } else {
            0.0
        };
        if g1 > 0.3 + 0.4 * g0 - cont {
            best_xy = xy;
            best_yy = yy;
            t = t1;
            g = g1;
        }
    }
    let mut pg = if best_yy <= best_xy {
        1.0
    } else {
        best_xy / (best_yy + 1.0)
    };
    let mut xc = [0f32; 3];
    for (k, slot) in xc.iter_mut().enumerate() {
        *slot = corr(t + k as i32 - 1).0;
    }
    let offset = if xc[2] - xc[0] > 0.7 * (xc[1] - xc[0]) {
        1
    } else if xc[0] - xc[2] > 0.7 * (xc[1] - xc[2]) {
        -1
    } else {
        0
    };
    if pg > g {
        pg = g;
    }
    let period = (2 * t + offset).max(minperiod0);
    (period, pg.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::{downsample, lpc, search};

    #[test]
    fn a_periodic_signal_has_its_period_found() {
        let period = 200usize;
        let len = 2048usize;
        let sig: Vec<f32> = (0..len)
            .map(|i| {
                let ph = (i % period) as f32 / period as f32;
                1000.0 * (ph * std::f32::consts::TAU).sin()
                    + 500.0 * (ph * 2.0 * std::f32::consts::TAU).sin()
            })
            .collect();
        let mut lp = vec![0f32; len / 2];
        downsample(&[&sig], &mut lp, len);
        let off = 720usize;
        let t = search(&lp[off / 2..], &lp, len - off, off - 100);
        let pitch = off - t;
        assert!(
            pitch % period <= 2 || period - pitch % period <= 2,
            "pitch {pitch}"
        );
    }

    #[test]
    fn lpc_predicts_an_ar_process() {
        // x[n] = 0.9 x[n-1]: the first coefficient is about -0.9.
        let mut ac = [0f32; 3];
        for (i, a) in ac.iter_mut().enumerate() {
            *a = 0.9f32.powi(i as i32);
        }
        let mut c = [0f32; 2];
        lpc(&mut c, &ac);
        assert!((c[0] + 0.9).abs() < 1e-3 && c[1].abs() < 1e-3, "{c:?}");
    }
}
