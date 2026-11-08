//! The A/52 filterbank (section 7.9.4): one 512-point or two 256-point
//! IMDCTs per block through a complex IFFT, the KBD window of Table 7.33,
//! and overlap-add. The encoder's forward transform (section 8.2.3.2) is
//! the transpose of the same steps.

use std::f64::consts::PI;

/// Coefficients per block.
pub(crate) const COEFS: usize = 256;

/// Samples one transform covers.
pub(crate) const SPAN: usize = 512;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Complex {
    re: f32,
    im: f32,
}

impl Complex {
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

/// Radix-2 complex FFT of one power-of-two size.
struct Fft {
    size: usize,
    twiddles: Vec<Complex>,
    order: Vec<usize>,
}

impl Fft {
    fn new(size: usize) -> Self {
        let bits = size.trailing_zeros();
        let order = (0..size)
            .map(|index| index.reverse_bits() >> (usize::BITS - bits))
            .collect();
        let twiddles = (0..size / 2)
            .map(|k| {
                let angle = 2.0 * PI * k as f64 / size as f64;
                Complex {
                    re: angle.cos() as f32,
                    im: angle.sin() as f32,
                }
            })
            .collect();
        Self {
            size,
            twiddles,
            order,
        }
    }

    /// In place. `inverse` uses `e^{+j}` (the spec's IFFT, unscaled),
    /// otherwise `e^{-j}`.
    fn run(&self, data: &mut [Complex], inverse: bool) {
        for (index, &target) in self.order.iter().enumerate() {
            if index < target {
                data.swap(index, target);
            }
        }
        let mut half = 1;
        while half < self.size {
            let stride = self.size / (half * 2);
            for start in (0..self.size).step_by(half * 2) {
                for k in 0..half {
                    let mut twiddle = self.twiddles[k * stride];
                    if !inverse {
                        twiddle = twiddle.conj();
                    }
                    let odd = data[start + k + half].mul(twiddle);
                    let even = data[start + k];
                    data[start + k] = Complex {
                        re: even.re + odd.re,
                        im: even.im + odd.im,
                    };
                    data[start + k + half] = Complex {
                        re: even.re - odd.re,
                        im: even.im - odd.im,
                    };
                }
            }
            half *= 2;
        }
    }
}

/// Zeroth-order modified Bessel function of the first kind.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let quarter = x * x / 4.0;
    for k in 1..200 {
        term *= quarter / (k * k) as f64;
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

/// The Kaiser-Bessel derived window with alpha 5 that Table 7.33 lists:
/// the first half of a 512-sample window.
fn kbd_window() -> [f32; COEFS] {
    let alpha = 5.0;
    let n = COEFS as f64;
    let kernel: Vec<f64> = (0..=COEFS)
        .map(|p| {
            let ratio = 2.0 * p as f64 / n - 1.0;
            bessel_i0(PI * alpha * (1.0 - ratio * ratio).max(0.0).sqrt())
        })
        .collect();
    let total: f64 = kernel.iter().sum();
    let mut window = [0f32; COEFS];
    let mut running = 0.0;
    for (slot, value) in window.iter_mut().zip(&kernel) {
        running += value;
        *slot = (running / total).sqrt() as f32;
    }
    window
}

/// Twiddles, window and FFTs for both block lengths.
pub(crate) struct Transform {
    window: [f32; COEFS],
    long_twiddle: Vec<Complex>,
    short_twiddle: Vec<Complex>,
    long_fft: Fft,
    short_fft: Fft,
}

impl Transform {
    pub(crate) fn new() -> Self {
        let n = SPAN as f64;
        let long_twiddle = (0..SPAN / 4)
            .map(|k| {
                let angle = 2.0 * PI * (8 * k + 1) as f64 / (8.0 * n);
                Complex {
                    re: -angle.cos() as f32,
                    im: -angle.sin() as f32,
                }
            })
            .collect();
        let short_twiddle = (0..SPAN / 8)
            .map(|k| {
                let angle = 2.0 * PI * (8 * k + 1) as f64 / (4.0 * n);
                Complex {
                    re: -angle.cos() as f32,
                    im: -angle.sin() as f32,
                }
            })
            .collect();
        Self {
            window: kbd_window(),
            long_twiddle,
            short_twiddle,
            long_fft: Fft::new(SPAN / 4),
            short_fft: Fft::new(SPAN / 8),
        }
    }

    pub(crate) fn window(&self) -> &[f32; COEFS] {
        &self.window
    }

    /// Section 7.9.4.1 steps 2 to 5: `out` is the windowed 512-sample block.
    pub(crate) fn inverse_long(&self, coefs: &[f32; COEFS], out: &mut [f32; SPAN]) {
        const N: usize = SPAN;
        let mut z = [Complex::default(); N / 4];
        for (k, slot) in z.iter_mut().enumerate() {
            let input = Complex {
                re: coefs[N / 2 - 2 * k - 1],
                im: coefs[2 * k],
            };
            *slot = input.mul(self.long_twiddle[k]);
        }
        self.long_fft.run(&mut z, true);
        for (n, slot) in z.iter_mut().enumerate() {
            *slot = slot.mul(self.long_twiddle[n]);
        }
        let w = &self.window;
        for n in 0..N / 8 {
            out[2 * n] = -z[N / 8 + n].im * w[2 * n];
            out[2 * n + 1] = z[N / 8 - n - 1].re * w[2 * n + 1];
            out[N / 4 + 2 * n] = -z[n].re * w[N / 4 + 2 * n];
            out[N / 4 + 2 * n + 1] = z[N / 4 - n - 1].im * w[N / 4 + 2 * n + 1];
            out[N / 2 + 2 * n] = -z[N / 8 + n].re * w[N / 2 - 2 * n - 1];
            out[N / 2 + 2 * n + 1] = z[N / 8 - n - 1].im * w[N / 2 - 2 * n - 2];
            out[3 * N / 4 + 2 * n] = z[n].im * w[N / 4 - 2 * n - 1];
            out[3 * N / 4 + 2 * n + 1] = -z[N / 4 - n - 1].re * w[N / 4 - 2 * n - 2];
        }
    }

    /// Section 7.9.4.2: two interleaved 256-sample transforms.
    pub(crate) fn inverse_short(&self, coefs: &[f32; COEFS], out: &mut [f32; SPAN]) {
        const N: usize = SPAN;
        let mut z1 = [Complex::default(); N / 8];
        let mut z2 = [Complex::default(); N / 8];
        for k in 0..N / 8 {
            let twiddle = self.short_twiddle[k];
            let x1 = |i: usize| coefs[2 * i];
            let x2 = |i: usize| coefs[2 * i + 1];
            z1[k] = Complex {
                re: x1(N / 4 - 2 * k - 1),
                im: x1(2 * k),
            }
            .mul(twiddle);
            z2[k] = Complex {
                re: x2(N / 4 - 2 * k - 1),
                im: x2(2 * k),
            }
            .mul(twiddle);
        }
        self.short_fft.run(&mut z1, true);
        self.short_fft.run(&mut z2, true);
        for n in 0..N / 8 {
            z1[n] = z1[n].mul(self.short_twiddle[n]);
            z2[n] = z2[n].mul(self.short_twiddle[n]);
        }
        let w = &self.window;
        for n in 0..N / 8 {
            out[2 * n] = -z1[n].im * w[2 * n];
            out[2 * n + 1] = z1[N / 8 - n - 1].re * w[2 * n + 1];
            out[N / 4 + 2 * n] = -z1[n].re * w[N / 4 + 2 * n];
            out[N / 4 + 2 * n + 1] = z1[N / 8 - n - 1].im * w[N / 4 + 2 * n + 1];
            out[N / 2 + 2 * n] = -z2[n].re * w[N / 2 - 2 * n - 1];
            out[N / 2 + 2 * n + 1] = z2[N / 8 - n - 1].im * w[N / 2 - 2 * n - 2];
            out[3 * N / 4 + 2 * n] = z2[n].im * w[N / 4 - 2 * n - 1];
            out[3 * N / 4 + 2 * n + 1] = -z2[N / 8 - n - 1].re * w[N / 4 - 2 * n - 2];
        }
    }

    /// The forward transform the decoder's long IMDCT inverts: the
    /// transpose of [`Self::inverse_long`], window included, scaled so
    /// that overlap-add of the inverse returns the input.
    pub(crate) fn forward_long(&self, input: &[f32; SPAN], coefs: &mut [f32; COEFS]) {
        const N: usize = SPAN;
        let w = &self.window;
        let mut z = [Complex::default(); N / 4];
        for n in 0..N / 8 {
            z[N / 8 + n].im -= input[2 * n] * w[2 * n];
            z[N / 8 - n - 1].re += input[2 * n + 1] * w[2 * n + 1];
            z[n].re -= input[N / 4 + 2 * n] * w[N / 4 + 2 * n];
            z[N / 4 - n - 1].im += input[N / 4 + 2 * n + 1] * w[N / 4 + 2 * n + 1];
            z[N / 8 + n].re -= input[N / 2 + 2 * n] * w[N / 2 - 2 * n - 1];
            z[N / 8 - n - 1].im += input[N / 2 + 2 * n + 1] * w[N / 2 - 2 * n - 2];
            z[n].im += input[3 * N / 4 + 2 * n] * w[N / 4 - 2 * n - 1];
            z[N / 4 - n - 1].re -= input[3 * N / 4 + 2 * n + 1] * w[N / 4 - 2 * n - 2];
        }
        for (n, slot) in z.iter_mut().enumerate() {
            *slot = slot.mul(self.long_twiddle[n].conj());
        }
        self.long_fft.run(&mut z, false);
        let scale = FORWARD_SCALE / (N / 4) as f32;
        for (k, slot) in z.iter().enumerate() {
            let value = slot.mul(self.long_twiddle[k].conj());
            coefs[N / 2 - 2 * k - 1] = value.re * scale;
            coefs[2 * k] = value.im * scale;
        }
    }

    /// The transpose of [`Self::inverse_short`], scaled the same way.
    pub(crate) fn forward_short(&self, input: &[f32; SPAN], coefs: &mut [f32; COEFS]) {
        const N: usize = SPAN;
        let w = &self.window;
        let mut z1 = [Complex::default(); N / 8];
        let mut z2 = [Complex::default(); N / 8];
        for n in 0..N / 8 {
            z1[n].im -= input[2 * n] * w[2 * n];
            z1[N / 8 - n - 1].re += input[2 * n + 1] * w[2 * n + 1];
            z1[n].re -= input[N / 4 + 2 * n] * w[N / 4 + 2 * n];
            z1[N / 8 - n - 1].im += input[N / 4 + 2 * n + 1] * w[N / 4 + 2 * n + 1];
            z2[n].re -= input[N / 2 + 2 * n] * w[N / 2 - 2 * n - 1];
            z2[N / 8 - n - 1].im += input[N / 2 + 2 * n + 1] * w[N / 2 - 2 * n - 2];
            z2[n].im += input[3 * N / 4 + 2 * n] * w[N / 4 - 2 * n - 1];
            z2[N / 8 - n - 1].re -= input[3 * N / 4 + 2 * n + 1] * w[N / 4 - 2 * n - 2];
        }
        for n in 0..N / 8 {
            z1[n] = z1[n].mul(self.short_twiddle[n].conj());
            z2[n] = z2[n].mul(self.short_twiddle[n].conj());
        }
        self.short_fft.run(&mut z1, false);
        self.short_fft.run(&mut z2, false);
        let scale = FORWARD_SCALE / (N / 8) as f32;
        for k in 0..N / 8 {
            let twiddle = self.short_twiddle[k].conj();
            let a = z1[k].mul(twiddle);
            let b = z2[k].mul(twiddle);
            coefs[2 * (N / 4 - 2 * k - 1)] = a.re * scale;
            coefs[2 * (2 * k)] = a.im * scale;
            coefs[2 * (N / 4 - 2 * k - 1) + 1] = b.re * scale;
            coefs[2 * (2 * k) + 1] = b.im * scale;
        }
    }
}

/// The transpose of the unscaled IFFT path, times this, is the inverse of
/// the decoder's IMDCT with its overlap-add factor of 2.
const FORWARD_SCALE: f32 = 0.5;

/// Section 7.9.4.1 step 6: `pcm[n] = 2 (x[n] + delay[n])`, then keep the
/// second half.
pub(crate) fn overlap_add(block: &[f32; SPAN], delay: &mut [f32; COEFS], pcm: &mut [f32]) {
    for ((out, carried), value) in pcm.iter_mut().zip(delay.iter()).zip(block.iter()) {
        *out = 2.0 * (value + carried);
    }
    delay.copy_from_slice(&block[COEFS..]);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Table 7.33 as printed, to five decimals.
    const TABLE_7_33: [f32; COEFS] = [
        0.00014, 0.00024, 0.00037, 0.00051, 0.00067, 0.00086, 0.00107, 0.00130, 0.00157, 0.00187,
        0.00220, 0.00256, 0.00297, 0.00341, 0.00390, 0.00443, 0.00501, 0.00564, 0.00632, 0.00706,
        0.00785, 0.00871, 0.00962, 0.01061, 0.01166, 0.01279, 0.01399, 0.01526, 0.01662, 0.01806,
        0.01959, 0.02121, 0.02292, 0.02472, 0.02662, 0.02863, 0.03073, 0.03294, 0.03527, 0.03770,
        0.04025, 0.04292, 0.04571, 0.04862, 0.05165, 0.05481, 0.05810, 0.06153, 0.06508, 0.06878,
        0.07261, 0.07658, 0.08069, 0.08495, 0.08935, 0.09389, 0.09859, 0.10343, 0.10842, 0.11356,
        0.11885, 0.12429, 0.12988, 0.13563, 0.14152, 0.14757, 0.15376, 0.16011, 0.16661, 0.17325,
        0.18005, 0.18699, 0.19407, 0.20130, 0.20867, 0.21618, 0.22382, 0.23161, 0.23952, 0.24757,
        0.25574, 0.26404, 0.27246, 0.28100, 0.28965, 0.29841, 0.30729, 0.31626, 0.32533, 0.33450,
        0.34376, 0.35311, 0.36253, 0.37204, 0.38161, 0.39126, 0.40096, 0.41072, 0.42054, 0.43040,
        0.44030, 0.45023, 0.46020, 0.47019, 0.48020, 0.49022, 0.50025, 0.51028, 0.52031, 0.53033,
        0.54033, 0.55031, 0.56026, 0.57019, 0.58007, 0.58991, 0.59970, 0.60944, 0.61912, 0.62873,
        0.63827, 0.64774, 0.65713, 0.66643, 0.67564, 0.68476, 0.69377, 0.70269, 0.71150, 0.72019,
        0.72877, 0.73723, 0.74557, 0.75378, 0.76186, 0.76981, 0.77762, 0.78530, 0.79283, 0.80022,
        0.80747, 0.81457, 0.82151, 0.82831, 0.83496, 0.84145, 0.84779, 0.85398, 0.86001, 0.86588,
        0.87160, 0.87716, 0.88257, 0.88782, 0.89291, 0.89785, 0.90264, 0.90728, 0.91176, 0.91610,
        0.92028, 0.92432, 0.92822, 0.93197, 0.93558, 0.93906, 0.94240, 0.94560, 0.94867, 0.95162,
        0.95444, 0.95713, 0.95971, 0.96217, 0.96451, 0.96674, 0.96887, 0.97089, 0.97281, 0.97463,
        0.97635, 0.97799, 0.97953, 0.98099, 0.98236, 0.98366, 0.98488, 0.98602, 0.98710, 0.98811,
        0.98905, 0.98994, 0.99076, 0.99153, 0.99225, 0.99291, 0.99353, 0.99411, 0.99464, 0.99513,
        0.99558, 0.99600, 0.99639, 0.99674, 0.99706, 0.99736, 0.99763, 0.99788, 0.99811, 0.99831,
        0.99850, 0.99867, 0.99882, 0.99895, 0.99908, 0.99919, 0.99929, 0.99938, 0.99946, 0.99953,
        0.99959, 0.99965, 0.99969, 0.99974, 0.99978, 0.99981, 0.99984, 0.99986, 0.99988, 0.99990,
        0.99992, 0.99993, 0.99994, 0.99995, 0.99996, 0.99997, 0.99998, 0.99998, 0.99998, 0.99999,
        0.99999, 0.99999, 0.99999, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000,
        1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000,
    ];

    #[test]
    fn the_window_matches_table_7_33() {
        let window = kbd_window();
        for (index, (computed, printed)) in window.iter().zip(TABLE_7_33).enumerate() {
            assert!(
                (computed - printed).abs() < 6e-6,
                "w[{index}] = {computed} against {printed}"
            );
        }
    }

    #[test]
    fn the_window_is_power_complementary() {
        let window = kbd_window();
        for n in 0..COEFS {
            let sum = window[n] * window[n] + window[COEFS - 1 - n] * window[COEFS - 1 - n];
            assert!((sum - 1.0).abs() < 1e-6, "n {n}: {sum}");
        }
    }

    fn noise(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (*seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    }

    /// The direct IMDCT of section 7.9.4 for one long block, before the window.
    fn direct_long(coefs: &[f32; COEFS]) -> Vec<f64> {
        let n = SPAN as f64;
        (0..SPAN)
            .map(|i| {
                let mut sum = 0.0;
                for (k, value) in coefs.iter().enumerate() {
                    let angle = 2.0 * PI / (4.0 * n) * (2 * i + 1) as f64 * (2 * k + 1) as f64
                        + PI / 4.0 * (2 * k + 1) as f64;
                    sum += f64::from(*value) * angle.cos();
                }
                sum
            })
            .collect()
    }

    #[test]
    fn the_fast_long_inverse_is_a_scaled_direct_imdct() {
        let transform = Transform::new();
        let mut seed = 7;
        let mut coefs = [0f32; COEFS];
        for value in &mut coefs {
            *value = noise(&mut seed);
        }
        let mut fast = [0f32; SPAN];
        transform.inverse_long(&coefs, &mut fast);
        let direct = direct_long(&coefs);
        let w = transform.window();
        let mut ratio = None;
        for i in 0..SPAN {
            let window = if i < COEFS { w[i] } else { w[SPAN - 1 - i] };
            let expected = direct[i] * f64::from(window);
            if expected.abs() > 0.5 {
                let r = f64::from(fast[i]) / expected;
                let first = *ratio.get_or_insert(r);
                assert!((r - first).abs() < 1e-3, "sample {i}: {r} against {first}");
            }
        }
        assert!(ratio.is_some());
    }

    fn reconstruct(short_pattern: &[bool]) -> f32 {
        let transform = Transform::new();
        let blocks = short_pattern.len();
        let mut seed = 99;
        let signal: Vec<f32> = (0..(blocks + 1) * COEFS)
            .map(|_| noise(&mut seed))
            .collect();
        let mut delay = [0f32; COEFS];
        let mut output = vec![0f32; (blocks + 1) * COEFS];
        for (b, &short) in short_pattern.iter().enumerate() {
            let mut input = [0f32; SPAN];
            input.copy_from_slice(&signal[b * COEFS..b * COEFS + SPAN]);
            let mut coefs = [0f32; COEFS];
            let mut block = [0f32; SPAN];
            if short {
                transform.forward_short(&input, &mut coefs);
                transform.inverse_short(&coefs, &mut block);
            } else {
                transform.forward_long(&input, &mut coefs);
                transform.inverse_long(&coefs, &mut block);
            }
            overlap_add(&block, &mut delay, &mut output[b * COEFS..(b + 1) * COEFS]);
        }
        let mut worst = 0f32;
        for i in COEFS..blocks * COEFS {
            worst = worst.max((output[i] - signal[i]).abs());
        }
        worst
    }

    #[test]
    fn long_blocks_reconstruct_their_input() {
        assert!(reconstruct(&[false; 8]) < 1e-5);
    }

    #[test]
    fn short_and_mixed_blocks_reconstruct_their_input() {
        assert!(reconstruct(&[true; 8]) < 1e-5);
        assert!(reconstruct(&[false, true, false, false, true, true, false, false]) < 1e-5);
    }
}
