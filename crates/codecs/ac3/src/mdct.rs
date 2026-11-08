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

    /// Table 7.33 as printed, in units of 1e-5.
    const TABLE_7_33: [u32; COEFS] = [
        14, 24, 37, 51, 67, 86, 107, 130, 157, 187, 220, 256, 297, 341, 390, 443, 501, 564, 632,
        706, 785, 871, 962, 1061, 1166, 1279, 1399, 1526, 1662, 1806, 1959, 2121, 2292, 2472, 2662,
        2863, 3073, 3294, 3527, 3770, 4025, 4292, 4571, 4862, 5165, 5481, 5810, 6153, 6508, 6878,
        7261, 7658, 8069, 8495, 8935, 9389, 9859, 10343, 10842, 11356, 11885, 12429, 12988, 13563,
        14152, 14757, 15376, 16011, 16661, 17325, 18005, 18699, 19407, 20130, 20867, 21618, 22382,
        23161, 23952, 24757, 25574, 26404, 27246, 28100, 28965, 29841, 30729, 31626, 32533, 33450,
        34376, 35311, 36253, 37204, 38161, 39126, 40096, 41072, 42054, 43040, 44030, 45023, 46020,
        47019, 48020, 49022, 50025, 51028, 52031, 53033, 54033, 55031, 56026, 57019, 58007, 58991,
        59970, 60944, 61912, 62873, 63827, 64774, 65713, 66643, 67564, 68476, 69377, 70269, 71150,
        72019, 72877, 73723, 74557, 75378, 76186, 76981, 77762, 78530, 79283, 80022, 80747, 81457,
        82151, 82831, 83496, 84145, 84779, 85398, 86001, 86588, 87160, 87716, 88257, 88782, 89291,
        89785, 90264, 90728, 91176, 91610, 92028, 92432, 92822, 93197, 93558, 93906, 94240, 94560,
        94867, 95162, 95444, 95713, 95971, 96217, 96451, 96674, 96887, 97089, 97281, 97463, 97635,
        97799, 97953, 98099, 98236, 98366, 98488, 98602, 98710, 98811, 98905, 98994, 99076, 99153,
        99225, 99291, 99353, 99411, 99464, 99513, 99558, 99600, 99639, 99674, 99706, 99736, 99763,
        99788, 99811, 99831, 99850, 99867, 99882, 99895, 99908, 99919, 99929, 99938, 99946, 99953,
        99959, 99965, 99969, 99974, 99978, 99981, 99984, 99986, 99988, 99990, 99992, 99993, 99994,
        99995, 99996, 99997, 99998, 99998, 99998, 99999, 99999, 99999, 99999, 100000, 100000,
        100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000,
    ];

    #[test]
    fn the_window_matches_table_7_33() {
        let window = kbd_window();
        for (index, (computed, printed)) in window.iter().zip(TABLE_7_33).enumerate() {
            let printed = printed as f32 / 100_000.0;
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
