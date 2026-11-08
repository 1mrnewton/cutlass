//! Encoder analysis: the 32-band polyphase analysis filterbank, the MDCT
//! of each subband with block switching, and the alias-reduction
//! butterflies.
//!
//! ISO/IEC 11172-3 Annex C.1.3 (analysis subband filter) and C.1.5.3
//! (Layer III MDCT, windows, aliasing butterfly). The analysis window is
//! the synthesis window `D[i]` of Table 3-B.3 divided by 32, which is
//! what makes the pair reconstruct (checked by the tests below).

use std::f64::consts::PI;
use std::sync::OnceLock;

use crate::synth::{kernels, synthesis_window};
use crate::tables::{SHORT_WIDTHS, short_starts};

/// Block types, as the side information names them.
pub(crate) const NORMAL: u8 = 0;
pub(crate) const START: u8 = 1;
pub(crate) const SHORT: u8 = 2;
pub(crate) const STOP: u8 = 3;

struct Tables {
    /// Analysis window `C[i]`.
    window: [f32; 512],
    /// `cos((2k + 1)(i - 16) pi / 64)`, `[k][i]`.
    matrix: [[f32; 64]; 32],
    /// `cos(pi/72 (2n + 1 + 18)(2k + 1)) / 9`, `[k][n]`.
    long_cos: [[f32; 36]; 18],
    /// `cos(pi/24 (2n + 1 + 6)(2k + 1)) / 3`, `[k][n]`.
    short_cos: [[f32; 12]; 6],
    /// Long windows by block type (index 2 unused).
    long_windows: [[f32; 36]; 4],
    short_window: [f32; 12],
}

fn tables() -> &'static Tables {
    static TABLES: OnceLock<Box<Tables>> = OnceLock::new();
    TABLES.get_or_init(|| Box::new(build()))
}

fn build() -> Tables {
    let d = synthesis_window();
    let window: [f32; 512] = std::array::from_fn(|i| d[i] / 32.0);
    let matrix: [[f32; 64]; 32] = std::array::from_fn(|k| {
        std::array::from_fn(|i| ((2 * k + 1) as f64 * (i as f64 - 16.0) * PI / 64.0).cos() as f32)
    });
    let long_cos: [[f32; 36]; 18] = std::array::from_fn(|k| {
        std::array::from_fn(|n| {
            ((PI / 72.0 * (2 * n + 1 + 18) as f64 * (2 * k + 1) as f64).cos() / 9.0) as f32
        })
    });
    let short_cos: [[f32; 12]; 6] = std::array::from_fn(|k| {
        std::array::from_fn(|n| {
            ((PI / 24.0 * (2 * n + 1 + 6) as f64 * (2 * k + 1) as f64).cos() / 3.0) as f32
        })
    });
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
    Tables {
        window,
        matrix,
        long_cos,
        short_cos,
        long_windows: [normal, start, [0.0; 36], stop],
        short_window: std::array::from_fn(short_sin),
    }
}

/// One channel of the analysis filterbank.
#[derive(Clone)]
pub(crate) struct Analysis {
    /// The last 512 input samples, newest first.
    x: [f32; 512],
}

impl Default for Analysis {
    fn default() -> Self {
        Self { x: [0.0; 512] }
    }
}

impl Analysis {
    /// 32 new input samples in, 32 subband samples out.
    pub(crate) fn run(&mut self, input: &[f32], out: &mut [f32; 32]) {
        let t = tables();
        self.x.copy_within(0..480, 32);
        for (i, sample) in input.iter().take(32).enumerate() {
            self.x[31 - i] = *sample;
        }
        let mut y = [0f32; 64];
        for (i, slot) in y.iter_mut().enumerate() {
            let mut sum = 0f32;
            for j in 0..8 {
                sum += t.window[i + 64 * j] * self.x[i + 64 * j];
            }
            *slot = sum;
        }
        for (k, value) in out.iter_mut().enumerate() {
            let row = &t.matrix[k];
            let mut sum = 0f32;
            for (c, v) in row.iter().zip(y.iter()) {
                sum += c * v;
            }
            *value = sum;
        }
    }
}

/// The MDCT of one granule of one channel.
///
/// `previous` and `current` are 18 subband samples per subband, `[t][sb]`,
/// frequency inversion not yet applied. The result is in bitstream order:
/// subband order with alias reduction for long blocks; for short blocks,
/// scalefactor band by band, window by window, as the spectrum is coded.
pub(crate) fn mdct(
    previous: &[[f32; 32]; 18],
    current: &[[f32; 32]; 18],
    block_type: u8,
    rate_table: usize,
    xr: &mut [f32; 576],
) {
    let t = tables();
    let mut z = [0f32; 36];
    let mut short = [[[0f32; 6]; 3]; 32];
    for sb in 0..32 {
        let invert = sb % 2 == 1;
        for n in 0..18 {
            let sign = if invert && n % 2 == 1 { -1.0 } else { 1.0 };
            z[n] = previous[n][sb] * sign;
            z[n + 18] = current[n][sb] * sign;
        }
        if block_type == SHORT {
            for (w, window) in short[sb].iter_mut().enumerate() {
                let input = &z[6 + 6 * w..18 + 6 * w];
                for (k, slot) in window.iter_mut().enumerate() {
                    *slot = input
                        .iter()
                        .zip(&t.short_window)
                        .zip(&t.short_cos[k])
                        .map(|((x, w), c)| x * w * c)
                        .sum();
                }
            }
        } else {
            let window = &t.long_windows[usize::from(block_type)];
            for k in 0..18 {
                let row = &t.long_cos[k];
                let mut sum = 0f32;
                for n in 0..36 {
                    sum += z[n] * window[n] * row[n];
                }
                xr[18 * sb + k] = sum;
            }
        }
    }
    if block_type == SHORT {
        let starts = short_starts(rate_table);
        let widths = SHORT_WIDTHS[rate_table];
        for band in 0..13 {
            let width = usize::from(widths[band]);
            let from = usize::from(starts[band]);
            for w in 0..3 {
                for j in 0..width {
                    let f = from + j;
                    xr[3 * from + w * width + j] = short[f / 6][w][f % 6];
                }
            }
        }
    } else {
        let k = kernels();
        for sb in 1..32 {
            let base = sb * 18;
            for (i, &(cs, ca)) in k.alias.iter().enumerate() {
                let below = xr[base - 1 - i];
                let above = xr[base + i];
                xr[base - 1 - i] = below * cs + above * ca;
                xr[base + i] = above * cs - below * ca;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Analysis, NORMAL, SHORT, START, STOP, mdct};
    use crate::synth::{Synthesis, alias_reduce, hybrid};

    /// Analysis, MDCT, then the decoder's own inverse path, with the block
    /// types given per granule.
    fn round_trip(input: &[f32], blocks: &[u8]) -> Vec<f32> {
        let mut analysis = Analysis::default();
        let mut synthesis = Synthesis::default();
        let mut overlap = [[0f32; 18]; 32];
        let mut previous = [[0f32; 32]; 18];
        let mut out = Vec::new();
        for (granule, &block) in blocks.iter().enumerate() {
            let mut current = [[0f32; 32]; 18];
            for (t, slot) in current.iter_mut().enumerate() {
                let at = granule * 576 + t * 32;
                let chunk: Vec<f32> = (at..at + 32)
                    .map(|i| input.get(i).copied().unwrap_or(0.0))
                    .collect();
                analysis.run(&chunk, slot);
            }
            let mut xr = [0f32; 576];
            mdct(&previous, &current, block, 0, &mut xr);
            previous = current;
            // The decoder side: reorder short blocks, undo aliasing.
            if block == SHORT {
                let mut scratch = [0f32; 576];
                let starts = crate::tables::short_starts(0);
                let widths = crate::tables::SHORT_WIDTHS[0];
                for band in 0..13 {
                    let width = usize::from(widths[band]);
                    let base = 3 * usize::from(starts[band]);
                    for j in 0..width {
                        for w in 0..3 {
                            scratch[base + 3 * j + w] = xr[base + w * width + j];
                        }
                    }
                }
                xr = scratch;
                let mut subbands = [[0f32; 32]; 18];
                hybrid(&xr, 0, 0, 32, &mut overlap, &mut subbands);
                for slot in &subbands {
                    let mut pcm = [0f32; 32];
                    synthesis.run(slot, &mut pcm);
                    out.extend_from_slice(&pcm);
                }
            } else {
                alias_reduce(&mut xr, 31);
                let mut subbands = [[0f32; 32]; 18];
                hybrid(&xr, block, 32, 32, &mut overlap, &mut subbands);
                for slot in &subbands {
                    let mut pcm = [0f32; 32];
                    synthesis.run(slot, &mut pcm);
                    out.extend_from_slice(&pcm);
                }
            }
        }
        out
    }

    fn signal(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32;
                0.3 * (t * 0.031).sin() + 0.2 * (t * 0.7).sin() + 0.1 * (t * 2.9).cos()
            })
            .collect()
    }

    /// The delay of analysis, MDCT and synthesis together, in samples.
    const DELAY: usize = 576 + 481;

    fn check(blocks: &[u8]) {
        let input = signal(576 * blocks.len());
        let out = round_trip(&input, blocks);
        let mut worst = 0f32;
        for i in 0..out.len() - DELAY {
            worst = worst.max((out[i + DELAY] - input[i]).abs());
        }
        // The polyphase pair is near, not exactly, perfect reconstruction:
        // about -84 dB against this 0.6-peak signal.
        assert!(worst < 1e-4, "{blocks:?}: {worst}");
    }

    #[test]
    fn long_blocks_reconstruct_the_input() {
        check(&[NORMAL; 8]);
    }

    #[test]
    fn a_switch_to_short_and_back_reconstructs_the_input() {
        check(&[
            NORMAL, NORMAL, START, SHORT, SHORT, STOP, NORMAL, START, SHORT, STOP, NORMAL, NORMAL,
        ]);
    }
}
