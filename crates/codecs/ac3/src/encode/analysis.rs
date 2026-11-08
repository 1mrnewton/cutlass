//! Transient detection (A/52 section 8.2.2) and the forward transform of
//! one channel, block by block.

use std::f32::consts::PI;

use crate::mdct::{COEFS, SPAN, Transform};

/// Peaks below this, on the high-passed block, force a long block.
const SILENCE: f32 = 100.0 / 32_768.0;

/// Per-level thresholds `T[j]`: a segment whose peak, times this, exceeds
/// the previous segment's peak marks a transient.
const THRESHOLDS: [f32; 3] = [0.1, 0.075, 0.05];

/// One direct form I biquad section.
#[derive(Clone, Copy, Debug, Default)]
struct Biquad {
    b: [f32; 3],
    a: [f32; 2],
    x: [f32; 2],
    y: [f32; 2],
}

impl Biquad {
    /// A Butterworth-family high-pass section at `cutoff` with quality `q`.
    fn high_pass(sample_rate: u32, cutoff: f32, q: f32) -> Self {
        let w0 = 2.0 * PI * cutoff / sample_rate as f32;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self {
            b: [
                (1.0 + cos) / 2.0 / a0,
                -(1.0 + cos) / a0,
                (1.0 + cos) / 2.0 / a0,
            ],
            a: [-2.0 * cos / a0, (1.0 - alpha) / a0],
            x: [0.0; 2],
            y: [0.0; 2],
        }
    }

    fn run(&mut self, input: f32) -> f32 {
        let out = self.b[0] * input + self.b[1] * self.x[0] + self.b[2] * self.x[1]
            - self.a[0] * self.y[0]
            - self.a[1] * self.y[1];
        self.x = [input, self.x[0]];
        self.y = [out, self.y[0]];
        out
    }
}

/// The section 8.2.2 detector: an 8 kHz high-pass, then peak ratios of
/// adjacent segments at three time scales.
#[derive(Clone, Debug)]
pub(crate) struct Transients {
    filter: [Biquad; 2],
    /// The last segment's peak on each level of the previous tree.
    last: [f32; 3],
}

impl Transients {
    pub(crate) fn new(sample_rate: u32) -> Self {
        // A fourth-order Butterworth high-pass as two cascaded biquads.
        Self {
            filter: [
                Biquad::high_pass(sample_rate, 8_000.0, 0.541_196_1),
                Biquad::high_pass(sample_rate, 8_000.0, 1.306_563),
            ],
            last: [0.0; 3],
        }
    }

    /// Sample index of the first transient in `block`, at 64-sample
    /// resolution. `None` when the block stays below the silence floor or
    /// no segment jumps.
    pub(crate) fn locate(&mut self, block: &[f32; COEFS]) -> Option<usize> {
        let mut level1 = 0.0f32;
        let mut level2 = [0.0f32; 2];
        let mut level3 = [0.0f32; 4];
        for (n, &sample) in block.iter().enumerate() {
            let first = self.filter[0].run(sample);
            let high = self.filter[1].run(first).abs();
            level1 = level1.max(high);
            level2[n / 128] = level2[n / 128].max(high);
            level3[n / 64] = level3[n / 64].max(high);
        }
        let previous = self.last;
        self.last = [level1, level2[1], level3[3]];
        if level1 < SILENCE {
            return None;
        }
        let first = |peaks: &[f32], before: f32, threshold: f32| {
            let mut prior = before;
            for (index, &peak) in peaks.iter().enumerate() {
                if peak * threshold > prior {
                    return Some(index);
                }
                prior = peak;
            }
            None
        };
        if let Some(quarter) = first(&level3, previous[2], THRESHOLDS[2]) {
            return Some(quarter * 64);
        }
        if let Some(half) = first(&level2, previous[1], THRESHOLDS[1]) {
            return Some(half * 128);
        }
        first(&[level1], previous[0], THRESHOLDS[0]).map(|_whole| 0)
    }
}

/// One channel's history and detector.
#[derive(Clone, Debug)]
pub(crate) struct Analysis {
    history: [f32; COEFS],
    transients: Transients,
}

impl Analysis {
    pub(crate) fn new(sample_rate: u32) -> Self {
        Self {
            history: [0.0; COEFS],
            transients: Transients::new(sample_rate),
        }
    }

    /// Transform the block whose new samples are `block`. With
    /// `may_switch`, a transient in those samples selects the pair of
    /// short transforms. Returns `blksw` and the sample index of the
    /// transient inside the block.
    pub(crate) fn block(
        &mut self,
        transform: &Transform,
        block: &[f32; COEFS],
        may_switch: bool,
        coefs: &mut [f32; COEFS],
    ) -> (bool, Option<usize>) {
        let at = self.transients.locate(block);
        let short = may_switch && at.is_some();
        let mut input = [0.0f32; SPAN];
        input[..COEFS].copy_from_slice(&self.history);
        input[COEFS..].copy_from_slice(block);
        if short {
            transform.forward_short(&input, coefs);
        } else {
            transform.forward_long(&input, coefs);
        }
        self.history = *block;
        (short, at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(len: usize, freq: f32, amplitude: f32) -> Vec<f32> {
        (0..len)
            .map(|n| amplitude * (2.0 * PI * freq * n as f32 / 48_000.0).sin())
            .collect()
    }

    fn blocks(samples: &[f32]) -> impl Iterator<Item = [f32; COEFS]> + '_ {
        samples.as_chunks::<COEFS>().0.iter().copied()
    }

    #[test]
    fn a_steady_tone_is_never_a_transient() {
        let mut detector = Transients::new(48_000);
        let tone = sine(256 * 40, 1_000.0, 0.5);
        let mut blocks = blocks(&tone);
        // The filter starts at rest, so the onset of the tone is a transient.
        for block in blocks.by_ref().take(2) {
            detector.locate(&block);
        }
        let flagged = blocks
            .filter(|block| detector.locate(block).is_some())
            .count();
        assert_eq!(flagged, 0);
    }

    #[test]
    fn a_click_after_quiet_is_a_transient_in_its_block_only() {
        let mut samples = sine(256 * 8, 500.0, 0.01);
        samples[256 * 5 + 100] = 0.9;
        let mut detector = Transients::new(48_000);
        let flags: Vec<bool> = blocks(&samples)
            .map(|b| detector.locate(&b).is_some())
            .collect();
        assert!(flags[5], "{flags:?}");
        assert!(!flags[..5].contains(&true), "{flags:?}");
    }

    #[test]
    fn silence_forces_a_long_block() {
        let mut detector = Transients::new(48_000);
        let mut block = [0.0f32; COEFS];
        block[200] = SILENCE * 0.5;
        assert!(detector.locate(&[0.0; COEFS]).is_none());
        assert!(detector.locate(&block).is_none());
    }

    #[test]
    fn the_forward_transform_follows_the_flag() {
        let transform = Transform::new();
        let mut analysis = Analysis::new(48_000);
        let mut coefs = [0.0f32; COEFS];
        let mut block = [0.0f32; COEFS];
        block[255] = 0.8;
        assert!(
            !analysis
                .block(&transform, &[0.0; COEFS], true, &mut coefs)
                .0
        );
        assert!(analysis.block(&transform, &block, true, &mut coefs).0);
        let mut again = Analysis::new(48_000);
        again.block(&transform, &[0.0; COEFS], false, &mut coefs);
        assert!(!again.block(&transform, &block, false, &mut coefs).0);
    }
}
