//! The masking threshold of one block: how much noise each MDCT bin can
//! carry unheard.
//!
//! The spectrum is grouped into bands half a Bark wide. Each band's energy
//! spreads to its neighbors through the Schroeder spreading function,
//! normalized so a flat spectrum masks itself at the band offset. The
//! offset blends tone-masking-noise (14.5 dB + Bark) and noise-masking-
//! tone (5.5 dB) by the masker's tonality, measured as spectral flatness.
//! A tonal band therefore contributes less masking than a noisy one.
//! The absolute threshold of hearing (Terhardt) sets the least the
//! threshold can be, with full scale taken as 114 dB SPL.

/// One block size's analysis tables.
#[derive(Clone, Debug)]
pub(crate) struct Bands {
    /// First bin of each band, then the bin count.
    edges: Vec<usize>,
    bark: Vec<f32>,
    /// Spreading from band `i` to band `j` at `[j * count + i]`, linear,
    /// divided by what a flat spectrum spreads into `j`.
    spread: Vec<f32>,
    /// Absolute threshold per bin, as power.
    ath: Vec<f32>,
}

/// Frequency in Hz to Bark (Zwicker and Terhardt).
pub(crate) fn bark(frequency: f64) -> f64 {
    13.0 * (0.000_76 * frequency).atan() + 3.5 * (frequency / 7_500.0).powi(2).atan()
}

/// dB SPL of a full-scale sine. A 96 dB reference puts this threshold
/// above the noise of an ordinary mix at the edges of hearing, and those
/// bands are then left silent.
const FULL_SCALE_SPL: f64 = 114.0;

/// Terhardt's threshold in quiet, dB SPL.
fn threshold_in_quiet(frequency: f64) -> f64 {
    let khz = (frequency / 1_000.0).max(0.02);
    3.64 * khz.powf(-0.8) - 6.5 * (-0.6 * (khz - 3.3).powi(2)).exp() + 1e-3 * khz.powi(4)
}

/// Schroeder's spreading function, dB, for a maskee `dz` Bark above the
/// masker.
fn spreading(dz: f64) -> f64 {
    let d = dz + 0.474;
    15.81 + 7.5 * d - 17.5 * (1.0 + d * d).sqrt()
}

/// Sum of squared MDCT coefficients of a full-scale sine, on this
/// transform's `4/n` scale. That power is treated as 96 dB SPL.
pub(crate) const FULL_SCALE: f64 = 1.23;

impl Bands {
    /// Tables for `bins` MDCT bins at `rate`.
    pub(crate) fn new(bins: usize, rate: u32) -> Self {
        let hz = |k: f64| (k + 0.5) * f64::from(rate) / (2.0 * bins as f64);
        let mut edges = vec![0];
        let mut start_bark = bark(hz(0.0));
        for k in 1..bins {
            let z = bark(hz(k as f64));
            if z - start_bark >= 0.5 {
                edges.push(k);
                start_bark = z;
            }
        }
        edges.push(bins);
        let count = edges.len() - 1;
        let bark_of: Vec<f32> = (0..count)
            .map(|b| bark(hz((edges[b] + edges[b + 1]) as f64 / 2.0 - 0.5)) as f32)
            .collect();
        let mut spread = vec![0.0_f32; count * count];
        for j in 0..count {
            let mut total = 0.0;
            for i in 0..count {
                let dz = f64::from(bark_of[j] - bark_of[i]);
                let value = if dz.abs() > 8.0 {
                    0.0
                } else {
                    10_f64.powf(spreading(dz) / 10.0)
                };
                spread[j * count + i] = value as f32;
                total += value;
            }
            for i in 0..count {
                spread[j * count + i] /= total as f32;
            }
        }
        let ath = (0..bins)
            .map(|k| {
                let db = threshold_in_quiet(hz(k as f64)).min(60.0) - FULL_SCALE_SPL;
                (FULL_SCALE * 10_f64.powf(db / 10.0)) as f32
            })
            .collect();
        Self {
            edges,
            bark: bark_of,
            spread,
            ath,
        }
    }

    pub(crate) fn count(&self) -> usize {
        self.edges.len() - 1
    }

    /// Bin edges of the half-Bark bands, including 0 and the bin count.
    pub(crate) fn edges(&self) -> &[usize] {
        &self.edges
    }

    /// The masking threshold of `spectrum` per bin, as power, into
    /// `mask`. `shift_db` lowers it (more quality); `ath_shift_db` moves
    /// the threshold in quiet.
    pub(crate) fn mask(
        &self,
        spectrum: &[f32],
        shift_db: f32,
        ath_shift_db: f32,
        scratch: &mut Scratch,
        mask: &mut [f32],
    ) {
        let count = self.count();
        let bins = spectrum.len();
        scratch.energy.clear();
        scratch.offset.clear();
        for b in 0..count {
            let (start, end) = (self.edges[b], self.edges[b + 1]);
            let energy: f32 = spectrum[start..end].iter().map(|x| x * x).sum();
            scratch.energy.push(energy);
            // Flatness over the band and one bin either side.
            let low = start.saturating_sub(1);
            let high = (end + 1).min(bins);
            let mut log_sum = 0.0_f32;
            let mut sum = 0.0_f32;
            for &x in &spectrum[low..high] {
                let power = x * x + 1e-20;
                log_sum += power.ln();
                sum += power;
            }
            let len = (high - low) as f32;
            let flatness_db = 10.0 / std::f32::consts::LN_10 * (log_sum / len - (sum / len).ln());
            let tonality = ((-flatness_db - 6.0) / 14.0).clamp(0.0, 1.0);
            let offset = tonality * (14.5 + self.bark[b]) + (1.0 - tonality) * 5.5;
            scratch.offset.push(offset);
        }
        let ath_gain = 10_f32.powf(ath_shift_db / 10.0);
        let shift = 10_f32.powf(-shift_db / 10.0);
        for j in 0..count {
            let row = &self.spread[j * count..(j + 1) * count];
            // The offset belongs to the masker. A tonal band's energy is
            // lowered before it spreads, so a noisy neighbor does not
            // amplify it again.
            let spread: f32 = row
                .iter()
                .zip(&scratch.energy)
                .zip(&scratch.offset)
                .map(|((weight, energy), offset)| weight * energy * 10_f32.powf(-offset / 10.0))
                .sum();
            let (start, end) = (self.edges[j], self.edges[j + 1]);
            let per_bin = spread * shift / (end - start) as f32;
            for (slot, &quiet) in mask[start..end].iter_mut().zip(&self.ath[start..end]) {
                *slot = per_bin.max(quiet * ath_gain);
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Scratch {
    energy: Vec<f32>,
    offset: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::{Bands, FULL_SCALE, Scratch, bark};
    use crate::mdct::{Mdct, Scratch as MdctScratch};
    use crate::window::Windows;

    fn spectrum_of(signal: impl FnMut(usize) -> f32, n: usize) -> Vec<f32> {
        let windows = Windows::new([256, n]);
        let shape = windows.shape(true, true, true);
        let mut block: Vec<f32> = (0..n).map(signal).collect();
        windows.apply(&shape, &mut block);
        let mut out = vec![0.0; n / 2];
        Mdct::new(n).forward(&block, &mut out, &mut MdctScratch::default());
        for value in &mut out {
            *value *= 4.0 / n as f32;
        }
        out
    }

    #[test]
    fn bands_are_half_a_bark() {
        let bands = Bands::new(1024, 44_100);
        assert!((44..=54).contains(&bands.count()), "{}", bands.count());
        assert!((bark(22_050.0) - 25.0).abs() < 1.0);
        let short = Bands::new(128, 44_100);
        assert!(short.count() < bands.count());
    }

    #[test]
    fn a_full_scale_sine_has_the_reference_power() {
        let n = 2048;
        let spectrum = spectrum_of(
            |i| (2.0 * std::f32::consts::PI * 1_000.0 * i as f32 / 44_100.0).sin(),
            n,
        );
        let power: f32 = spectrum.iter().map(|x| x * x).sum();
        assert!(
            (f64::from(power) / FULL_SCALE).log10().abs() < 0.15,
            "{power}"
        );
    }

    #[test]
    fn a_tone_masks_less_than_noise_of_the_same_energy() {
        let n = 2048;
        let bands = Bands::new(n / 2, 44_100);
        let tone = spectrum_of(
            |i| 0.1 * (2.0 * std::f32::consts::PI * 2_000.0 * i as f32 / 44_100.0).sin(),
            n,
        );
        let mut state = 1_u32;
        let noise = spectrum_of(
            |_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                0.1 * ((state as f32 / u32::MAX as f32) * 2.0 - 1.0)
            },
            n,
        );
        let energy = |s: &[f32]| s.iter().map(|x| x * x).sum::<f32>();
        let scale = (energy(&tone) / energy(&noise)).sqrt();
        let noise: Vec<f32> = noise.iter().map(|x| x * scale).collect();
        let mut scratch = Scratch::default();
        let mut tone_mask = vec![0.0; n / 2];
        let mut noise_mask = vec![0.0; n / 2];
        bands.mask(&tone, 0.0, 0.0, &mut scratch, &mut tone_mask);
        bands.mask(&noise, 0.0, 0.0, &mut scratch, &mut noise_mask);
        // The mask of a tone peaks at the tone and falls away from it.
        let peak = (2_000.0 / 44_100.0 * n as f32) as usize;
        assert!(tone_mask[peak] > tone_mask[peak / 2] * 10.0);
        assert!(tone_mask[peak] > tone_mask[peak * 3] * 10.0);
        // Low-frequency quiet threshold dominates a sum over every bin.
        // Compare the mask to the energy in the tone's own band.
        let band = peak.saturating_sub(4)..(peak + 5).min(tone.len());
        let efficiency = |spectrum: &[f32], mask: &[f32]| {
            let energy: f32 = spectrum[band.clone()].iter().map(|x| x * x).sum();
            let covered: f32 = mask[band.clone()].iter().sum();
            covered / energy
        };
        let tone_eff = efficiency(&tone, &tone_mask);
        let noise_eff = efficiency(&noise, &noise_mask);
        assert!(
            tone_eff * 4.0 < noise_eff,
            "tone {tone_eff} noise {noise_eff}"
        );
        // Silence is masked by the threshold in quiet alone.
        let mut quiet = vec![0.0; n / 2];
        bands.mask(&vec![0.0; n / 2], 0.0, 0.0, &mut scratch, &mut quiet);
        let at_4k = (4_000.0 / 44_100.0 * (n / 2) as f32) as usize;
        assert!(
            quiet[at_4k] > 0.0 && quiet[at_4k] < 1e-8,
            "{}",
            quiet[at_4k]
        );
        assert!(quiet.iter().all(|&m| m > 0.0 && m.is_finite()));
    }
}
