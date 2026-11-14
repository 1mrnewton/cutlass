//! Hybrid synthesis: alias reduction, IMDCT with overlap, frequency
//! inversion, and the 32-band polyphase synthesis filterbank.
//!
//! ISO/IEC 11172-3 clause 2.4.3.4 (Layer III `IMDCT`, windows, overlap-add,
//! alias reduction) and clause 2.4.3.2 with Figure A.2 (synthesis subband
//! filter). The algorithms follow the normative formulas directly; there
//! is no fast factorisation.

use std::f64::consts::PI;
use std::sync::OnceLock;

use crate::tables::{ALIAS_C, SYNTH_WINDOW_HALF};

/// Precomputed transform tables, shared and read-only.
pub(crate) struct Kernels {
    /// `cos(pi/72 * (2i + 1 + 18) * (2k + 1))`, `[i][k]`, 36 by 18.
    long_cos: [[f32; 18]; 36],
    /// `cos(pi/24 * (2i + 1 + 6) * (2k + 1))`, `[i][k]`, 12 by 6.
    short_cos: [[f32; 6]; 12],
    /// Long windows by block type 0, 1, 3 (index 2 unused).
    long_windows: [[f32; 36]; 4],
    /// `sin(pi/12 * (i + 0.5))`.
    short_window: [f32; 12],
    /// `cs[i]` and `ca[i]` of alias reduction.
    pub(crate) alias: [(f32, f32); 8],
    /// `cos((2k + 1) * m * pi / 64)`, `[m][k]`, m in 0..=32.
    matrix: [[f32; 32]; 33],
    /// Synthesis window `D[i]`.
    window: [f32; 512],
}

pub(crate) fn kernels() -> &'static Kernels {
    static KERNELS: OnceLock<Box<Kernels>> = OnceLock::new();
    KERNELS.get_or_init(|| Box::new(build()))
}

fn build() -> Kernels {
    let mut long_cos = [[0f32; 18]; 36];
    for (i, row) in long_cos.iter_mut().enumerate() {
        for (k, slot) in row.iter_mut().enumerate() {
            *slot = (PI / 72.0 * (2 * i + 1 + 18) as f64 * (2 * k + 1) as f64).cos() as f32;
        }
    }
    let mut short_cos = [[0f32; 6]; 12];
    for (i, row) in short_cos.iter_mut().enumerate() {
        for (k, slot) in row.iter_mut().enumerate() {
            *slot = (PI / 24.0 * (2 * i + 1 + 6) as f64 * (2 * k + 1) as f64).cos() as f32;
        }
    }
    let long_sin = |i: usize| (PI / 36.0 * (i as f64 + 0.5)).sin() as f32;
    let short_sin = |i: usize| (PI / 12.0 * (i as f64 + 0.5)).sin() as f32;
    let normal: [f32; 36] = std::array::from_fn(long_sin);
    let start: [f32; 36] = std::array::from_fn(|i| match i {
        0..=17 => long_sin(i),
        18..=23 => 1.0,
        24..=29 => short_sin(i - 18),
        _ => 0.0,
    });
    let stop: [f32; 36] = std::array::from_fn(|i| match i {
        0..=5 => 0.0,
        6..=11 => short_sin(i - 6),
        12..=17 => 1.0,
        _ => long_sin(i),
    });
    let long_windows = [normal, start, [0.0; 36], stop];
    let mut short_window = [0f32; 12];
    for (i, slot) in short_window.iter_mut().enumerate() {
        *slot = short_sin(i);
    }
    let mut alias = [(0f32, 0f32); 8];
    for (slot, c) in alias.iter_mut().zip(ALIAS_C) {
        let norm = (1.0 + c * c).sqrt();
        *slot = ((1.0 / norm) as f32, (c / norm) as f32);
    }
    let mut matrix = [[0f32; 32]; 33];
    for (m, row) in matrix.iter_mut().enumerate() {
        for (k, slot) in row.iter_mut().enumerate() {
            *slot = ((2 * k + 1) as f64 * m as f64 * PI / 64.0).cos() as f32;
        }
    }
    Kernels {
        long_cos,
        short_cos,
        long_windows,
        short_window,
        alias,
        matrix,
        window: synthesis_window(),
    }
}

/// The 512-tap synthesis window `D[i]` of ISO/IEC 11172-3 Table 3-B.3.
pub(crate) fn synthesis_window() -> [f32; 512] {
    let mut window = [0f32; 512];
    for (i, value) in SYNTH_WINDOW_HALF.iter().copied().enumerate() {
        let d = f64::from(value) / 65_536.0;
        window[i] = d as f32;
        if i != 0 {
            window[512 - i] = if i % 64 == 0 { d as f32 } else { -d as f32 };
        }
    }
    window
}

/// Alias reduction between subbands, in place.
///
/// Long blocks reduce every boundary; mixed blocks only the boundary
/// between subbands 0 and 1; pure short blocks none.
pub(crate) fn alias_reduce(xr: &mut [f32; 576], boundaries: usize) {
    let k = kernels();
    for sb in 1..=boundaries.min(31) {
        let base = sb * 18;
        for (i, &(cs, ca)) in k.alias.iter().enumerate() {
            let below = xr[base - 1 - i];
            let above = xr[base + i];
            xr[base - 1 - i] = below * cs - above * ca;
            xr[base + i] = above * cs + below * ca;
        }
    }
}

/// IMDCT, windowing, overlap-add and frequency inversion of one granule.
///
/// `xr` is in subband order: subband `sb` owns lines `18 * sb ..`, and a
/// short block stores window `w` coefficient `k` at `18 * sb + 3 * k + w`.
/// `long_subbands` is how many leading subbands use the long transform
/// with `long_type` (2 for a mixed block: the normal window). `out[t][sb]`
/// receives 18 time samples per subband.
pub(crate) fn hybrid(
    xr: &[f32; 576],
    long_type: u8,
    long_subbands: usize,
    nonzero_subbands: usize,
    overlap: &mut [[f32; 18]; 32],
    out: &mut [[f32; 32]; 18],
) {
    let k = kernels();
    let mut z = [0f32; 36];
    for sb in 0..32 {
        let input = &xr[sb * 18..sb * 18 + 18];
        if sb >= nonzero_subbands {
            for t in 0..18 {
                out[t][sb] = overlap[sb][t];
            }
            overlap[sb] = [0.0; 18];
        } else if sb < long_subbands {
            let window = &k.long_windows[usize::from(long_type)];
            for (i, slot) in z.iter_mut().enumerate() {
                let row = &k.long_cos[i];
                let mut sum = 0f32;
                for (x, c) in input.iter().zip(row.iter()) {
                    sum += x * c;
                }
                *slot = sum * window[i];
            }
            for t in 0..18 {
                out[t][sb] = z[t] + overlap[sb][t];
                overlap[sb][t] = z[t + 18];
            }
        } else {
            z = [0.0; 36];
            for w in 0..3 {
                for i in 0..12 {
                    let row = &k.short_cos[i];
                    let mut sum = 0f32;
                    for (kk, c) in row.iter().enumerate() {
                        sum += input[3 * kk + w] * c;
                    }
                    z[6 + 6 * w + i] += sum * k.short_window[i];
                }
            }
            for t in 0..18 {
                out[t][sb] = z[t] + overlap[sb][t];
                overlap[sb][t] = z[t + 18];
            }
        }
        if sb % 2 == 1 {
            for t in (1..18).step_by(2) {
                out[t][sb] = -out[t][sb];
            }
        }
    }
}

/// One channel of the polyphase synthesis filterbank.
#[derive(Clone)]
pub(crate) struct Synthesis {
    v: [f32; 1024],
}

impl Default for Synthesis {
    fn default() -> Self {
        Self { v: [0.0; 1024] }
    }
}

impl Synthesis {
    /// 32 subband samples in, 32 PCM samples out.
    pub(crate) fn run(&mut self, subbands: &[f32; 32], pcm: &mut [f32]) {
        let k = kernels();
        self.v.copy_within(0..960, 64);
        // X[m] = sum_k S[k] cos((2k+1) m pi / 64), m in 0..=32.
        let mut x = [0f32; 33];
        for (m, slot) in x.iter_mut().enumerate().take(32) {
            let row = &k.matrix[m];
            let mut sum = 0f32;
            for (s, c) in subbands.iter().zip(row.iter()) {
                sum += s * c;
            }
            *slot = sum;
        }
        // V[i] = sum_k S[k] cos((16 + i)(2k + 1) pi / 64).
        for (i, value) in self.v[..64].iter_mut().enumerate() {
            let n = 16 + i;
            *value = match n {
                16..=32 => x[n],
                33..=48 => -x[64 - n],
                _ => -x[n.abs_diff(64)],
            };
        }
        for (j, sample) in pcm.iter_mut().enumerate().take(32) {
            let mut sum = 0f32;
            for i in 0..8 {
                sum += k.window[64 * i + j] * self.v[128 * i + j];
                sum += k.window[64 * i + 32 + j] * self.v[128 * i + 96 + j];
            }
            *sample = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Synthesis, kernels, synthesis_window};
    use std::f64::consts::PI;

    #[test]
    fn the_window_keeps_the_values_the_standard_prints() {
        let d = synthesis_window();
        assert_eq!(d[0], 0.0);
        assert!((f64::from(d[1]) + 0.000_015_259).abs() < 1e-9);
        assert!((f64::from(d[256]) - 1.144_989_014).abs() < 1e-9);
        assert!((f64::from(d[257]) - 1.144_287_109).abs() < 1e-9);
        assert!((f64::from(d[511]) - 0.000_015_259).abs() < 1e-9);
    }

    #[test]
    fn the_matrix_form_equals_the_direct_sum() {
        let mut subbands = [0f32; 32];
        for (k, s) in subbands.iter_mut().enumerate() {
            *s = ((k * 7919) % 13) as f32 / 13.0 - 0.5;
        }
        let mut synth = Synthesis::default();
        let mut pcm = [0f32; 32];
        synth.run(&subbands, &mut pcm);
        for (i, value) in synth.v[..64].iter().enumerate() {
            let mut direct = 0f64;
            for (k, s) in subbands.iter().enumerate() {
                direct += f64::from(*s) * ((16 + i) as f64 * (2 * k + 1) as f64 * PI / 64.0).cos();
            }
            assert!((direct - f64::from(*value)).abs() < 1e-5, "V[{i}]");
        }
    }

    #[test]
    fn alias_butterflies_are_rotations() {
        for &(cs, ca) in &kernels().alias {
            assert!((cs * cs + ca * ca - 1.0).abs() < 1e-6);
        }
    }
}
