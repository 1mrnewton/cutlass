//! The MDCT of one block (Vorbis I section 4.3.7), through a DCT-IV of
//! half the coefficients, which is one complex FFT of a quarter block.
//!
//! For a block of `n = 2m` samples:
//!
//! - inverse: `y[i] = sum over k < m of X[k] cos(pi / m (i + 1/2 + m/2)(k + 1/2))`
//! - forward: `X[k] = sum over i < n of x[i] cos(pi / m (i + 1/2 + m/2)(k + 1/2))`
//!
//! Neither is scaled; the caller scales.

use std::f64::consts::PI;

/// A radix-2 complex FFT of one power-of-two size.
#[derive(Clone, Debug)]
struct Fft {
    size: usize,
    /// Bit-reversed index of each position.
    reversed: Vec<u32>,
    /// `exp(-2 pi i k / size)` for `k < size / 2`.
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Fft {
    fn new(size: usize) -> Self {
        let bits = size.trailing_zeros();
        let reversed = (0..size as u32)
            .map(|i| {
                if bits == 0 {
                    0
                } else {
                    i.reverse_bits() >> (32 - bits)
                }
            })
            .collect();
        let half = size / 2;
        let angle = |k: usize| -2.0 * PI * k as f64 / size as f64;
        Self {
            size,
            reversed,
            cos: (0..half).map(|k| angle(k).cos() as f32).collect(),
            sin: (0..half).map(|k| angle(k).sin() as f32).collect(),
        }
    }

    /// The forward transform, in place.
    fn forward(&self, re: &mut [f32], im: &mut [f32]) {
        let n = self.size;
        for (i, &reversed) in self.reversed[..n].iter().enumerate() {
            let j = reversed as usize;
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut span = 1;
        while span < n {
            let step = n / (span * 2);
            for start in (0..n).step_by(span * 2) {
                for k in 0..span {
                    let (wr, wi) = (self.cos[k * step], self.sin[k * step]);
                    let a = start + k;
                    let b = a + span;
                    let tr = re[b] * wr - im[b] * wi;
                    let ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            span *= 2;
        }
    }
}

/// The transforms for one block size.
#[derive(Clone, Debug)]
pub(crate) struct Mdct {
    /// Coefficients: half the block.
    m: usize,
    fft: Fft,
    /// `exp(-i pi (j + 1/8) / m)` for `j < m / 2`, before and after the FFT.
    cos: Vec<f32>,
    sin: Vec<f32>,
}

/// Scratch for one transform; reused across blocks.
#[derive(Clone, Debug, Default)]
pub(crate) struct Scratch {
    re: Vec<f32>,
    im: Vec<f32>,
    dct: Vec<f32>,
}

impl Mdct {
    /// Transforms of a block of `n` samples (a power of two, at least 8).
    pub(crate) fn new(n: usize) -> Self {
        let m = n / 2;
        let angle = |j: usize| -PI * (j as f64 + 0.125) / m as f64;
        Self {
            m,
            fft: Fft::new(m / 2),
            cos: (0..m / 2).map(|j| angle(j).cos() as f32).collect(),
            sin: (0..m / 2).map(|j| angle(j).sin() as f32).collect(),
        }
    }

    /// The DCT-IV of `input` (`m` values) into `output`:
    /// `u[k] = sum over j of x[j] cos(pi / m (j + 1/2)(k + 1/2))`.
    fn dct4(&self, input: &[f32], output: &mut [f32], scratch: &mut Scratch) {
        let m = self.m;
        let half = m / 2;
        scratch.re.resize(half, 0.0);
        scratch.im.resize(half, 0.0);
        for j in 0..half {
            let (a, b) = (input[2 * j], input[m - 1 - 2 * j]);
            let (c, s) = (self.cos[j], self.sin[j]);
            scratch.re[j] = a * c - b * s;
            scratch.im[j] = a * s + b * c;
        }
        self.fft.forward(&mut scratch.re, &mut scratch.im);
        for k in 0..half {
            let (zr, zi) = (scratch.re[k], scratch.im[k]);
            let (c, s) = (self.cos[k], self.sin[k]);
            output[2 * k] = zr * c - zi * s;
            output[m - 1 - 2 * k] = -(zr * s + zi * c);
        }
    }

    /// The inverse transform of `m` coefficients into `2m` samples.
    pub(crate) fn inverse(&self, coefficients: &[f32], out: &mut [f32], scratch: &mut Scratch) {
        let m = self.m;
        let mut dct = std::mem::take(&mut scratch.dct);
        dct.resize(m, 0.0);
        self.dct4(coefficients, &mut dct, scratch);
        let (q, h) = (m / 2, 3 * m / 2);
        out[..q].copy_from_slice(&dct[q..2 * q]);
        for i in q..h {
            out[i] = -dct[h - 1 - i];
        }
        for i in h..2 * m {
            out[i] = -dct[i - h];
        }
        scratch.dct = dct;
    }

    /// The forward transform of `2m` samples into `m` coefficients.
    pub(crate) fn forward(&self, samples: &[f32], out: &mut [f32], scratch: &mut Scratch) {
        let m = self.m;
        let mut folded = std::mem::take(&mut scratch.dct);
        folded.resize(m, 0.0);
        let (q, h) = (m / 2, 3 * m / 2);
        for i in 0..q {
            folded[i] = -samples[h - 1 - i] - samples[h + i];
        }
        for i in q..m {
            folded[i] = samples[i - q] - samples[h - 1 - i];
        }
        self.dct4(&folded, out, scratch);
        scratch.dct = folded;
    }
}

#[cfg(test)]
mod tests {
    use super::{Mdct, Scratch};
    use std::f64::consts::PI;

    fn basis(m: usize, i: usize, k: usize) -> f64 {
        (PI / m as f64 * (i as f64 + 0.5 + m as f64 / 2.0) * (k as f64 + 0.5)).cos()
    }

    fn noise(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn both_directions_match_the_definition() {
        for n in [8, 64, 256, 2048] {
            let m = n / 2;
            let mdct = Mdct::new(n);
            let mut scratch = Scratch::default();
            let coefficients = noise(m, 7);
            let mut out = vec![0.0; n];
            mdct.inverse(&coefficients, &mut out, &mut scratch);
            let mut worst = 0.0_f64;
            for (i, &got) in out.iter().enumerate() {
                let want: f64 = (0..m)
                    .map(|k| f64::from(coefficients[k]) * basis(m, i, k))
                    .sum();
                worst = worst.max((f64::from(got) - want).abs());
            }
            assert!(worst < 1e-3 * (m as f64).sqrt(), "inverse n={n}: {worst}");
            let samples = noise(n, 11);
            let mut spectrum = vec![0.0; m];
            mdct.forward(&samples, &mut spectrum, &mut scratch);
            let mut worst = 0.0_f64;
            for (k, &got) in spectrum.iter().enumerate() {
                let want: f64 = (0..n).map(|i| f64::from(samples[i]) * basis(m, i, k)).sum();
                worst = worst.max((f64::from(got) - want).abs());
            }
            assert!(worst < 1e-3 * (n as f64).sqrt(), "forward n={n}: {worst}");
        }
    }

    #[test]
    fn windowed_overlap_add_reconstructs_at_four_over_n() {
        // Princen-Bradley sine window; forward scaled by 4/n.
        let n = 256;
        let m = n / 2;
        let mdct = Mdct::new(n);
        let mut scratch = Scratch::default();
        let window: Vec<f32> = (0..n)
            .map(|i| ((PI * (i as f64 + 0.5) / n as f64).sin()) as f32)
            .collect();
        let signal = noise(n * 4, 3);
        let mut out = vec![0.0_f32; n * 4 + n];
        let mut block = vec![0.0; n];
        let mut spectrum = vec![0.0; m];
        let mut start = 0;
        while start + n <= signal.len() {
            for i in 0..n {
                block[i] = signal[start + i] * window[i];
            }
            mdct.forward(&block, &mut spectrum, &mut scratch);
            for value in &mut spectrum {
                *value *= 4.0 / n as f32;
            }
            mdct.inverse(&spectrum, &mut block, &mut scratch);
            for i in 0..n {
                out[start + i] += block[i] * window[i];
            }
            start += m;
        }
        for i in m..signal.len() - n {
            assert!((out[i] - signal[i]).abs() < 1e-4, "sample {i}");
        }
    }
}
