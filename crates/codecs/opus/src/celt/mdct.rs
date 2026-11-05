//! The MDCT and its inverse, through a DCT-IV of half size computed with a
//! mixed-radix complex FFT of a quarter size.
//!
//! With N coefficients and 2N samples,
//! `X[k] = sum_n x[n] cos(pi/N (n + 1/2 + N/2) (k + 1/2))` and the inverse is
//! the transpose of that sum, unscaled. `forward` multiplies by `2/N`, the
//! scale the CELT reference analysis uses (its FFT divides by N/2), so a
//! windowed overlap-add of `inverse(forward(x))` returns `x`.

use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Cpx {
    re: f32,
    im: f32,
}

impl Cpx {
    #[inline]
    fn mul(self, o: Self) -> Self {
        Self {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
    #[inline]
    fn add(self, o: Self) -> Self {
        Self {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }
}

/// A forward complex FFT of one size, without scaling.
struct Fft {
    n: usize,
    /// (radix, remaining length) per stage.
    factors: Vec<(usize, usize)>,
    /// exp(-2 pi i k / n).
    twiddles: Vec<Cpx>,
}

impl Fft {
    fn new(n: usize) -> Self {
        let mut factors = Vec::new();
        let mut rest = n;
        while rest > 1 {
            let p = if rest.is_multiple_of(4) {
                4
            } else if rest.is_multiple_of(2) {
                2
            } else if rest.is_multiple_of(3) {
                3
            } else if rest.is_multiple_of(5) {
                5
            } else {
                rest
            };
            rest /= p;
            factors.push((p, rest));
        }
        let twiddles = (0..n)
            .map(|k| {
                let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
                Cpx {
                    re: a.cos() as f32,
                    im: a.sin() as f32,
                }
            })
            .collect();
        Self {
            n,
            factors,
            twiddles,
        }
    }

    fn run(&self, input: &[Cpx], out: &mut [Cpx]) {
        self.work(out, input, 0, 1, 0);
    }

    fn work(&self, out: &mut [Cpx], input: &[Cpx], offset: usize, fstride: usize, stage: usize) {
        let Some(&(p, m)) = self.factors.get(stage) else {
            if let (Some(o), Some(i)) = (out.first_mut(), input.get(offset)) {
                *o = *i;
            }
            return;
        };
        if m == 1 {
            for (k, o) in out.iter_mut().take(p).enumerate() {
                *o = input.get(offset + k * fstride).copied().unwrap_or_default();
            }
        } else {
            for (k, chunk) in out.chunks_exact_mut(m).take(p).enumerate() {
                self.work(chunk, input, offset + k * fstride, fstride * p, stage + 1);
            }
        }
        self.butterfly(out, fstride, p, m);
    }

    fn butterfly(&self, out: &mut [Cpx], fstride: usize, p: usize, m: usize) {
        let tw = &self.twiddles;
        let n = self.n;
        match p {
            2 => {
                for u in 0..m {
                    let t = out[u + m].mul(tw[u * fstride]);
                    let a = out[u];
                    out[u] = a.add(t);
                    out[u + m] = a.sub(t);
                }
            }
            4 => {
                for u in 0..m {
                    let a0 = out[u];
                    let a1 = out[u + m].mul(tw[u * fstride]);
                    let a2 = out[u + 2 * m].mul(tw[2 * u * fstride]);
                    let a3 = out[u + 3 * m].mul(tw[3 * u * fstride]);
                    let s0 = a0.add(a2);
                    let s1 = a0.sub(a2);
                    let s2 = a1.add(a3);
                    let s3 = a1.sub(a3);
                    // Multiply s3 by -i.
                    let s3 = Cpx {
                        re: s3.im,
                        im: -s3.re,
                    };
                    out[u] = s0.add(s2);
                    out[u + m] = s1.add(s3);
                    out[u + 2 * m] = s0.sub(s2);
                    out[u + 3 * m] = s1.sub(s3);
                }
            }
            _ => {
                let mut scratch = [Cpx::default(); 8];
                let mut big = Vec::new();
                let s: &mut [Cpx] = if p <= 8 {
                    &mut scratch[..p]
                } else {
                    big.resize(p, Cpx::default());
                    &mut big
                };
                for u in 0..m {
                    for q in 0..p {
                        s[q] = out[u + q * m].mul(tw[(q * u * fstride) % n]);
                    }
                    for q2 in 0..p {
                        let mut acc = s[0];
                        let step = (q2 * m * fstride) % n;
                        let mut at = 0usize;
                        for v in s.iter().skip(1) {
                            at = (at + step) % n;
                            acc = acc.add(v.mul(tw[at]));
                        }
                        out[u + q2 * m] = acc;
                    }
                }
            }
        }
    }
}

/// A DCT-IV of size `n` (even) through an FFT of `n / 2`.
struct Dct4 {
    n: usize,
    fft: Fft,
    /// exp(-i pi k / n) for the input.
    pre: Vec<Cpx>,
    /// exp(-i pi (k + 1/4) / n) for the output.
    post: Vec<Cpx>,
}

impl Dct4 {
    fn new(n: usize) -> Self {
        let half = n / 2;
        let tw = |x: f64| {
            let a = -std::f64::consts::PI * x / n as f64;
            Cpx {
                re: a.cos() as f32,
                im: a.sin() as f32,
            }
        };
        Self {
            n,
            fft: Fft::new(half),
            pre: (0..half).map(|k| tw(k as f64)).collect(),
            post: (0..half).map(|k| tw(k as f64 + 0.25)).collect(),
        }
    }

    /// `out[k] = scale * sum_n v[n] cos(pi/N (n + 1/2)(k + 1/2))`.
    fn run(&self, v: &[f32], out: &mut [f32], scale: f32, buf: &mut Vec<Cpx>) {
        let n = self.n;
        let half = n / 2;
        buf.clear();
        buf.resize(n, Cpx::default());
        let (a, b) = buf.split_at_mut(half);
        for i in 0..half {
            let c = Cpx {
                re: v[2 * i],
                im: v[n - 1 - 2 * i],
            };
            a[i] = c.mul(self.pre[i]);
        }
        self.fft.run(a, b);
        for k in 0..half {
            let s = b[k].mul(self.post[k]);
            out[2 * k] = scale * s.re;
            out[n - 1 - 2 * k] = -scale * s.im;
        }
    }
}

/// MDCTs of 120, 240, 480, and 960 coefficients.
pub(crate) struct Mdct {
    dct: [Dct4; 4],
}

pub(crate) fn mdct() -> &'static Mdct {
    static MDCT: OnceLock<Mdct> = OnceLock::new();
    MDCT.get_or_init(|| Mdct {
        dct: [
            Dct4::new(120),
            Dct4::new(240),
            Dct4::new(480),
            Dct4::new(960),
        ],
    })
}

/// Scratch buffers so a frame allocates nothing.
#[derive(Default)]
pub(crate) struct MdctScratch {
    v: Vec<f32>,
    w: Vec<f32>,
    c: Vec<Cpx>,
}

impl Mdct {
    fn dct(&self, n: usize) -> Option<&Dct4> {
        self.dct.iter().find(|d| d.n == n)
    }

    /// `out` (N values) from `x` (2N samples), scaled by 2/N.
    pub(crate) fn forward(&self, x: &[f32], out: &mut [f32], s: &mut MdctScratch) {
        let n = out.len();
        let Some(dct) = self.dct(n) else {
            return;
        };
        if x.len() < 2 * n {
            return;
        }
        let h = n / 2;
        s.v.clear();
        s.v.resize(n, 0.0);
        for i in 0..h {
            s.v[i] = -x[3 * h - 1 - i] - x[3 * h + i];
            s.v[h + i] = x[i] - x[n - 1 - i];
        }
        dct.run(&s.v, out, 2.0 / n as f32, &mut s.c);
    }

    /// `y` (2N samples) from `coef` (N values), unscaled.
    pub(crate) fn inverse(&self, coef: &[f32], y: &mut [f32], s: &mut MdctScratch) {
        let n = coef.len();
        let Some(dct) = self.dct(n) else {
            return;
        };
        if y.len() < 2 * n {
            return;
        }
        let h = n / 2;
        s.w.clear();
        s.w.resize(n, 0.0);
        dct.run(coef, &mut s.w, 1.0, &mut s.c);
        for i in 0..h {
            y[i] = s.w[h + i];
            y[n - 1 - i] = -s.w[h + i];
            y[3 * h - 1 - i] = -s.w[i];
            y[3 * h + i] = -s.w[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MdctScratch, mdct};

    fn direct_inverse(coef: &[f32]) -> Vec<f64> {
        let n = coef.len();
        (0..2 * n)
            .map(|t| {
                (0..n)
                    .map(|k| {
                        f64::from(coef[k])
                            * (std::f64::consts::PI / n as f64
                                * (t as f64 + 0.5 + n as f64 / 2.0)
                                * (k as f64 + 0.5))
                                .cos()
                    })
                    .sum()
            })
            .collect()
    }

    fn direct_forward(x: &[f32], n: usize) -> Vec<f64> {
        (0..n)
            .map(|k| {
                (0..2 * n)
                    .map(|t| {
                        f64::from(x[t])
                            * (std::f64::consts::PI / n as f64
                                * (t as f64 + 0.5 + n as f64 / 2.0)
                                * (k as f64 + 0.5))
                                .cos()
                    })
                    .sum::<f64>()
                    * 2.0
                    / n as f64
            })
            .collect()
    }

    #[test]
    fn fast_transforms_match_the_direct_sums() {
        let mut s = MdctScratch::default();
        let mut seed = 5u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / 8_388_608.0 - 1.0
        };
        for n in [120usize, 240, 480, 960] {
            let coef: Vec<f32> = (0..n).map(|_| rnd()).collect();
            let mut y = vec![0f32; 2 * n];
            mdct().inverse(&coef, &mut y, &mut s);
            let want = direct_inverse(&coef);
            let peak = want.iter().fold(0f64, |a, v| a.max(v.abs()));
            for (a, b) in y.iter().zip(&want) {
                assert!((f64::from(*a) - b).abs() < 1e-4 * peak, "inverse n {n}");
            }
            let x: Vec<f32> = (0..2 * n).map(|_| rnd()).collect();
            let mut out = vec![0f32; n];
            mdct().forward(&x, &mut out, &mut s);
            let want = direct_forward(&x, n);
            let peak = want.iter().fold(0f64, |a, v| a.max(v.abs()));
            for (a, b) in out.iter().zip(&want) {
                assert!((f64::from(*a) - b).abs() < 1e-4 * peak, "forward n {n}");
            }
        }
    }

    #[test]
    fn impulses_at_every_boundary_reconstruct_through_overlap_add() {
        // Princen-Bradley sine window over the full 2N, so TDAC applies.
        let n = 120usize;
        let w: Vec<f32> = (0..2 * n)
            .map(|i| ((std::f64::consts::PI * (i as f64 + 0.5) / (2 * n) as f64).sin()) as f32)
            .collect();
        let mut s = MdctScratch::default();
        for pos in [0usize, 1, n - 1, n, 2 * n - 1, 3 * n - 1] {
            let mut sig = vec![0f32; 4 * n];
            sig[n + pos.min(2 * n - 1)] = 1.0;
            let mut acc = vec![0f32; 4 * n];
            for block in 0..3 {
                let x: Vec<f32> = (0..2 * n).map(|i| sig[block * n + i] * w[i]).collect();
                let mut c = vec![0f32; n];
                mdct().forward(&x, &mut c, &mut s);
                let mut y = vec![0f32; 2 * n];
                mdct().inverse(&c, &mut y, &mut s);
                for i in 0..2 * n {
                    acc[block * n + i] += y[i] * w[i];
                }
            }
            for i in n..3 * n {
                assert!(
                    (acc[i] - sig[i]).abs() < 1e-4,
                    "pos {pos} at {i}: {}",
                    acc[i]
                );
            }
        }
    }
}
