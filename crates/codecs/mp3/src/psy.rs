//! Psychoacoustic model: masking thresholds per scalefactor band, the
//! perceptual entropy, and attack detection for block switching.
//!
//! The structure follows ISO/IEC 11172-3 Annex D, psychoacoustic model 2:
//! a Hann-windowed FFT (1024 lines for long blocks, three of 256 for short
//! blocks), energy and unpredictability per partition of about a third of
//! a Bark, a spreading function across partitions, a tonality index from
//! the spread unpredictability, a signal-to-mask offset between the
//! tone-masking-noise and noise-masking-tone values, pre-echo control
//! against the two previous thresholds, and the absolute threshold of
//! hearing. The spreading function is Schroeder's; the threshold in quiet
//! is Terhardt's approximation with a full-scale sine at 96 dB SPL.
//!
//! The model returns, per band, the ratio of allowed noise to signal energy
//! in the FFT domain. The encoder multiplies that ratio by the band's MDCT
//! energy, so the two transforms never need a common scale.

use std::f64::consts::PI;

use crate::tables::{LONG_WIDTHS, SHORT_WIDTHS, long_starts, short_starts};

const LONG_FFT: usize = 1024;
/// Bins either side of a bin that its local flatness is measured over.
const FLATNESS_REACH: usize = 6;
const SHORT_FFT: usize = 256;

/// Settings that trade bits for noise.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Params {
    /// Tone-masking-noise offset, dB.
    pub tmn: f32,
    /// Noise-masking-tone offset, dB.
    pub nmt: f32,
    /// Offset added to the threshold in quiet, dB.
    pub ath_offset: f32,
    /// Lines past this frequency are not coded, Hz.
    pub lowpass: f32,
}

/// What the model says about one granule of one channel.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Masking {
    /// Allowed noise over signal energy, per long band.
    pub long: [f32; 22],
    /// The same per short band and window.
    pub short: [[f32; 3]; 13],
    /// Signal energy per long band, in the spectrum's units: with
    /// `long`, the threshold itself.
    pub long_energy: [f32; 22],
}

/// Bark of a frequency in Hz (Zwicker and Terhardt).
fn bark(f: f64) -> f64 {
    13.0 * (0.000_76 * f).atan() + 3.5 * ((f / 7500.0) * (f / 7500.0)).atan()
}

/// Threshold in quiet in dB SPL (Terhardt), held at the 20 kHz value above.
fn ath_db(f: f64) -> f64 {
    let khz = (f.max(20.0) / 1000.0).min(20.0);
    3.64 * khz.powf(-0.8) - 6.5 * (-0.6 * (khz - 3.3) * (khz - 3.3)).exp() + 0.001 * khz.powi(4)
}

/// A band's threshold from its bins: the masking share of each bin,
/// floored by the band's lowest threshold in quiet, so the steep curve
/// at the bottom bin (DC) does not stand for the whole band.
fn band_threshold(masking: &[f32], quiet: &[f32]) -> f32 {
    let floor = quiet.iter().copied().fold(f32::INFINITY, f32::min);
    masking.iter().map(|m| m.max(floor)).sum()
}

/// Schroeder's spreading function, as a power ratio, for a maskee `dz`
/// Bark above the masker.
fn spreading(dz: f64) -> f64 {
    let x = dz + 0.474;
    let db = 15.81 + 7.5 * x - 17.5 * (1.0 + x * x).sqrt();
    if db < -60.0 {
        0.0
    } else {
        10f64.powf(db / 10.0)
    }
}

/// Radix-2 FFT tables for one size.
struct Fft {
    n: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
    window: Vec<f32>,
    reverse: Vec<usize>,
}

impl Fft {
    fn new(n: usize) -> Self {
        let bits = n.trailing_zeros();
        let reverse = (0..n)
            .map(|i| i.reverse_bits() >> (usize::BITS - bits))
            .collect();
        Self {
            n,
            cos: (0..n / 2)
                .map(|k| (2.0 * PI * k as f64 / n as f64).cos() as f32)
                .collect(),
            sin: (0..n / 2)
                .map(|k| (2.0 * PI * k as f64 / n as f64).sin() as f32)
                .collect(),
            window: (0..n)
                .map(|i| (0.5 - 0.5 * (2.0 * PI * (i as f64 + 0.5) / n as f64).cos()) as f32)
                .collect(),
            reverse,
        }
    }

    /// Windowed power spectrum, bins `0..=n/2`, scaled so that a
    /// full-scale sine sums to about 1, and the phase of each bin.
    fn power(&self, input: &[f32], energy: &mut [f32], phase: &mut [f32]) {
        let n = self.n;
        let mut re = vec![0f32; n];
        let mut im = vec![0f32; n];
        for i in 0..n {
            re[self.reverse[i]] = input.get(i).copied().unwrap_or(0.0) * self.window[i];
        }
        let mut size = 2;
        while size <= n {
            let half = size / 2;
            let stride = n / size;
            for start in (0..n).step_by(size) {
                for k in 0..half {
                    let (c, s) = (self.cos[k * stride], -self.sin[k * stride]);
                    let (a, b) = (start + k, start + k + half);
                    let tr = re[b] * c - im[b] * s;
                    let ti = re[b] * s + im[b] * c;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            size *= 2;
        }
        let scale = 32.0 / (3.0 * (n * n) as f32);
        for k in 0..=n / 2 {
            energy[k] = (re[k] * re[k] + im[k] * im[k]) * scale;
            phase[k] = im[k].atan2(re[k]);
        }
    }
}

/// A run of FFT bins analysed together.
#[derive(Clone, Debug)]
struct Partition {
    from: usize,
    to: usize,
    /// The threshold in quiet as an energy: the lowest over the bins.
    ath: f32,
}

/// Partitions of `bins` FFT bins at `rate`, at most a third of a Bark wide.
fn partitions(rate: f64, fft: usize, ath_offset: f32) -> (Vec<Partition>, Vec<Vec<f32>>) {
    let bins = fft / 2 + 1;
    let hz = |k: f64| k * rate / fft as f64;
    let mut parts = Vec::new();
    let mut from = 0;
    while from < bins {
        let start = bark(hz(from as f64));
        let mut to = from + 1;
        while to < bins && bark(hz(to as f64 + 0.5)) - start < 1.0 / 3.0 {
            to += 1;
        }
        let ath = (from..to)
            .map(|k| ath_db(hz(k as f64)))
            .fold(f64::INFINITY, f64::min);
        let energy = 10f64.powf((ath + f64::from(ath_offset) - 96.0) / 10.0);
        parts.push(Partition {
            from,
            to,
            ath: energy as f32,
        });
        from = to;
    }
    let centers: Vec<f64> = parts
        .iter()
        .map(|p| bark(hz((p.from + p.to) as f64 / 2.0)))
        .collect();
    // spread[maskee][masker], each row scaled so its masker weights sum to 1.
    let spread = centers
        .iter()
        .map(|z_maskee| {
            let row: Vec<f64> = centers
                .iter()
                .map(|z_masker| spreading(z_maskee - z_masker))
                .collect();
            let sum: f64 = row.iter().sum();
            row.iter().map(|v| (v / sum.max(1e-30)) as f32).collect()
        })
        .collect();
    (parts, spread)
}

/// For each band of `starts`, which FFT bins it covers: band lines are
/// `lines` per half spectrum and bins `bins` per half spectrum.
fn band_bins(starts: &[u16], widths: &[u16], lines: usize, bins: usize) -> Vec<(usize, usize)> {
    starts
        .iter()
        .zip(widths)
        .map(|(start, width)| {
            let from = usize::from(*start) * bins / lines;
            let to = ((usize::from(*start) + usize::from(*width)) * bins).div_ceil(lines);
            (from, to.max(from + 1).min(bins))
        })
        .collect()
}

struct Channel {
    /// Magnitude and phase of the last two long spectra.
    history: [(Vec<f32>, Vec<f32>); 2],
    /// Thresholds per partition of the last two granules.
    previous: [Vec<f32>; 2],
    /// How far the threshold in quiet is lowered for quiet passages, as
    /// an energy factor in `ATH_FLOOR..=1`.
    quiet: f32,
}

/// Spectrum energy (a full-scale sine is about 1) at and above which the
/// threshold in quiet stands as calibrated.
const ATH_LOUD: f32 = 0.1;
/// The most a quiet passage lowers the threshold in quiet: 30 dB.
const ATH_FLOOR: f32 = 1e-3;
/// How fast the lowered threshold rises back per granule.
const ATH_RELEASE: f32 = 0.85;

/// The model for every channel of one stream.
pub(crate) struct Psy {
    rate: u32,
    params: Params,
    long: Fft,
    short: Fft,
    long_parts: Vec<Partition>,
    long_spread: Vec<Vec<f32>>,
    short_parts: Vec<Partition>,
    short_spread: Vec<Vec<f32>>,
    long_bands: Vec<(usize, usize)>,
    short_bands: Vec<(usize, usize)>,
    channels: Vec<Channel>,
}

impl Psy {
    pub(crate) fn new(rate: u32, rate_table: usize, channels: usize, params: Params) -> Self {
        let (long_parts, long_spread) = partitions(f64::from(rate), LONG_FFT, params.ath_offset);
        let (short_parts, short_spread) = partitions(f64::from(rate), SHORT_FFT, params.ath_offset);
        let long_starts = long_starts(rate_table);
        let mut long_widths = [0u16; 22];
        long_widths.copy_from_slice(&LONG_WIDTHS[rate_table]);
        // The last long band runs to line 576.
        long_widths[21] = 576 - long_starts[21];
        let long_bands = band_bins(&long_starts[..22], &long_widths, 576, LONG_FFT / 2);
        let short_starts = short_starts(rate_table);
        let short_bands = band_bins(
            &short_starts[..13],
            &SHORT_WIDTHS[rate_table],
            192,
            SHORT_FFT / 2,
        );
        let bins = LONG_FFT / 2 + 1;
        let parts = long_parts.len();
        Self {
            rate,
            params,
            long: Fft::new(LONG_FFT),
            short: Fft::new(SHORT_FFT),
            long_parts,
            long_spread,
            short_parts,
            short_spread,
            long_bands,
            short_bands,
            channels: (0..channels)
                .map(|_| Channel {
                    history: [
                        (vec![0.0; bins], vec![0.0; bins]),
                        (vec![0.0; bins], vec![0.0; bins]),
                    ],
                    previous: [vec![f32::MAX; parts], vec![f32::MAX; parts]],
                    quiet: 1.0,
                })
                .collect(),
        }
    }

    /// Analyse one granule of channel `ch`. `long` is the 1024 samples the
    /// long block is centred on; `short` the three 256-sample windows.
    pub(crate) fn analyze(&mut self, ch: usize, long: &[f32], short: [&[f32]; 3]) -> Masking {
        let bins = LONG_FFT / 2 + 1;
        let mut energy = vec![0f32; bins];
        let mut phase = vec![0f32; bins];
        self.long.power(long, &mut energy, &mut phase);
        let magnitude: Vec<f32> = energy.iter().map(|e| e.sqrt()).collect();

        // Unpredictability of each bin from the two previous spectra.
        let channel = &mut self.channels[ch];
        // Quiet passages are heard turned up: lower the threshold in quiet
        // with the loudness, and let it rise back slowly.
        let loudness: f32 = energy.iter().sum();
        let level = (loudness / ATH_LOUD).clamp(ATH_FLOOR, 1.0);
        channel.quiet = level.min(channel.quiet / ATH_RELEASE).min(1.0);
        let quiet_scale = channel.quiet;
        let mut unpredictable = vec![0f32; bins];
        {
            let (m1, p1) = &channel.history[0];
            let (m2, p2) = &channel.history[1];
            for k in 0..bins {
                let r_hat = 2.0 * m1[k] - m2[k];
                let phi_hat = 2.0 * p1[k] - p2[k];
                let (r, phi) = (magnitude[k], phase[k]);
                let dx = r * phi.cos() - r_hat * phi_hat.cos();
                let dy = r * phi.sin() - r_hat * phi_hat.sin();
                let denominator = r + r_hat.abs();
                unpredictable[k] = if denominator > 0.0 {
                    ((dx * dx + dy * dy).sqrt() / denominator).min(1.0)
                } else {
                    1.0
                };
            }
        }
        channel.history.swap(0, 1);
        channel.history[0] = (magnitude, phase);

        // A tone that moves defeats the prediction; a peak that stands out
        // of its neighbourhood is tonal all the same. The local flatness
        // (geometric over arithmetic mean, square-rooted) bounds the
        // unpredictability: about 0.75 for noise, near 0 for a peak.
        {
            let logs: Vec<f32> = energy.iter().map(|e| (e + 1e-20).ln()).collect();
            for (k, slot) in unpredictable.iter_mut().enumerate() {
                let from = k.saturating_sub(FLATNESS_REACH);
                let to = (k + FLATNESS_REACH + 1).min(bins);
                let count = (to - from) as f32;
                let mean = energy[from..to].iter().sum::<f32>() / count;
                let geometric = (logs[from..to].iter().sum::<f32>() / count).exp();
                if mean > 0.0 {
                    let flatness = (geometric / mean).sqrt();
                    *slot = slot.min(flatness);
                }
            }
        }
        let parts = &self.long_parts;
        let mut e_part = vec![0f32; parts.len()];
        let mut c_part = vec![0f32; parts.len()];
        for (index, part) in parts.iter().enumerate() {
            for k in part.from..part.to {
                e_part[index] += energy[k];
                c_part[index] += energy[k] * unpredictable[k];
            }
        }
        let mut threshold = vec![0f32; parts.len()];
        for (index, row) in self.long_spread.iter().enumerate() {
            let mut spread_e = 0f32;
            let mut spread_c = 0f32;
            for (masker, weight) in row.iter().enumerate() {
                spread_e += weight * e_part[masker];
                spread_c += weight * c_part[masker];
            }
            let cb = if spread_e > 0.0 {
                (spread_c / spread_e).clamp(1e-6, 1.0)
            } else {
                1.0
            };
            let tonality = (-0.299 - 0.43 * cb.ln()).clamp(0.0, 1.0);
            let snr = tonality * self.params.tmn + (1.0 - tonality) * self.params.nmt;
            let mut nb = spread_e * 10f32.powf(-snr / 10.0);
            // Pre-echo control against the two previous granules.
            let previous = &channel.previous;
            nb = nb
                .min(2.0 * previous[0][index])
                .min(16.0 * previous[1][index]);
            threshold[index] = nb;
        }
        channel.previous.swap(0, 1);
        channel.previous[0].clone_from(&threshold);

        // Per bin: the partition threshold spread evenly, and the
        // threshold in quiet.
        let mut bin_threshold = vec![0f32; bins];
        let mut bin_quiet = vec![0f32; bins];
        for (index, part) in parts.iter().enumerate() {
            let count = (part.to - part.from) as f32;
            let share = threshold[index] / count;
            let quiet = part.ath * quiet_scale;
            for k in part.from..part.to {
                bin_threshold[k] = share;
                bin_quiet[k] = quiet;
            }
        }
        let cutoff = (self.params.lowpass * LONG_FFT as f32 / self.rate as f32) as usize;
        let mut long_ratio = [0f32; 22];
        let mut long_energy = [0f32; 22];
        for (band, (from, to)) in self.long_bands.iter().enumerate() {
            let e: f32 = energy[*from..*to].iter().sum();
            long_energy[band] = if *from >= cutoff { 0.0 } else { e };
            let t = band_threshold(&bin_threshold[*from..*to], &bin_quiet[*from..*to]);
            long_ratio[band] = if *from >= cutoff {
                f32::MAX
            } else {
                t / e.max(1e-20)
            };
        }

        let short_ratio = self.short_masking(short, quiet_scale);
        Masking {
            long: long_ratio,
            long_energy,
            short: short_ratio,
        }
    }

    fn short_masking(&self, windows: [&[f32]; 3], quiet_scale: f32) -> [[f32; 3]; 13] {
        let bins = SHORT_FFT / 2 + 1;
        let mut out = [[0f32; 3]; 13];
        let cutoff = (self.params.lowpass * SHORT_FFT as f32 / self.rate as f32) as usize;
        for (w, input) in windows.iter().enumerate() {
            let mut energy = vec![0f32; bins];
            let mut phase = vec![0f32; bins];
            self.short.power(input, &mut energy, &mut phase);
            let parts = &self.short_parts;
            let e_part: Vec<f32> = parts
                .iter()
                .map(|p| energy[p.from..p.to].iter().sum())
                .collect();
            let mut bin_threshold = vec![0f32; bins];
            let mut bin_quiet = vec![0f32; bins];
            for (index, row) in self.short_spread.iter().enumerate() {
                let spread: f32 = row.iter().zip(&e_part).map(|(w, e)| w * e).sum();
                let nb = spread * 10f32.powf(-self.params.nmt / 10.0);
                let part = &parts[index];
                let share = nb / (part.to - part.from) as f32;
                for k in part.from..part.to {
                    bin_threshold[k] = share;
                    bin_quiet[k] = part.ath * quiet_scale;
                }
            }
            for (band, (from, to)) in self.short_bands.iter().enumerate() {
                let e: f32 = energy[*from..*to].iter().sum();
                let t = band_threshold(&bin_threshold[*from..*to], &bin_quiet[*from..*to]);
                out[band][w] = if *from >= cutoff {
                    f32::MAX
                } else {
                    t / e.max(1e-20)
                };
            }
        }
        out
    }
}

/// Attack detection over the high-passed input: the energy of each
/// 64-sample block against the blocks before it.
pub(crate) struct Attack {
    history: Vec<f32>,
    last: f32,
}

/// An attack is a block this much louder than the loudest of the eight before it.
const ATTACK_RATIO: f32 = 10.0;
/// Below this block energy nothing is an attack (about -60 dBFS over 64 samples).
const ATTACK_FLOOR: f32 = 64.0 * 1e-6;

impl Attack {
    pub(crate) fn new() -> Self {
        Self {
            history: vec![0.0; 8],
            last: 0.0,
        }
    }

    /// Feed the next 576 samples; true when one of their blocks is an attack.
    pub(crate) fn feed(&mut self, samples: &[f32]) -> bool {
        let mut attack = false;
        for block in samples.chunks(64) {
            let mut energy = 0f32;
            for sample in block {
                let high = sample - self.last;
                self.last = *sample;
                energy += high * high;
            }
            let loudest = self.history.iter().copied().fold(0f32, f32::max);
            if energy > ATTACK_FLOOR && energy > ATTACK_RATIO * loudest.max(ATTACK_FLOOR / 4.0) {
                attack = true;
            }
            self.history.rotate_left(1);
            if let Some(slot) = self.history.last_mut() {
                *slot = energy;
            }
        }
        attack
    }
}

#[cfg(test)]
mod tests {
    use super::{Attack, Fft, Params, Psy, ath_db, bark};

    fn params() -> Params {
        Params {
            tmn: 18.0,
            nmt: 6.0,
            ath_offset: 0.0,
            lowpass: 17_000.0,
        }
    }

    #[test]
    fn a_full_scale_sine_sums_to_about_one() {
        let fft = Fft::new(1024);
        let input: Vec<f32> = (0..1024)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44_100.0).sin())
            .collect();
        let mut energy = vec![0f32; 513];
        let mut phase = vec![0f32; 513];
        fft.power(&input, &mut energy, &mut phase);
        let total: f32 = energy.iter().sum();
        assert!((total - 1.0).abs() < 0.05, "{total}");
        let peak =
            energy.iter().enumerate().fold(
                (0, 0f32),
                |best, (k, e)| if *e > best.1 { (k, *e) } else { best },
            );
        assert_eq!(peak.0, (1000.0f32 * 1024.0 / 44_100.0).round() as usize);
    }

    #[test]
    fn the_scales_are_where_the_literature_puts_them() {
        assert!((bark(1000.0) - 8.5).abs() < 0.2);
        assert!(ath_db(3500.0) < 0.0);
        assert!(ath_db(100.0) > 20.0);
    }

    #[test]
    fn a_tone_masks_less_of_itself_than_noise_does() {
        let mut tone_model = Psy::new(44_100, 1, 1, params());
        let mut noise_model = Psy::new(44_100, 1, 1, params());
        let tone: Vec<f32> = (0..4096)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44_100.0).sin())
            .collect();
        let mut state = 1u32;
        let noise: Vec<f32> = (0..4096)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((state >> 16) as f32 / 32_768.0 - 1.0) * 0.5
            })
            .collect();
        let mut tone_ratio = 0.0;
        let mut noise_ratio = 0.0;
        for g in 0..4 {
            let at = 576 * g;
            let t = tone_model.analyze(
                0,
                &tone[at..at + 1024],
                [
                    &tone[at..at + 256],
                    &tone[at + 192..at + 448],
                    &tone[at + 384..at + 640],
                ],
            );
            let n = noise_model.analyze(
                0,
                &noise[at..at + 1024],
                [
                    &noise[at..at + 256],
                    &noise[at + 192..at + 448],
                    &noise[at + 384..at + 640],
                ],
            );
            // Band 6 at 44.1 kHz holds 1 kHz (lines 24..30).
            tone_ratio = t.long[6];
            noise_ratio = n.long[6];
        }
        assert!(
            tone_ratio < noise_ratio / 4.0,
            "{tone_ratio} vs {noise_ratio}"
        );
    }

    #[test]
    fn a_click_after_quiet_is_an_attack_and_steady_noise_is_not() {
        let mut attack = Attack::new();
        let quiet = vec![0f32; 576];
        assert!(!attack.feed(&quiet));
        let mut click = vec![0f32; 576];
        click[300] = 0.8;
        click[301] = -0.6;
        assert!(attack.feed(&click));
        let mut attack = Attack::new();
        let mut state = 9u32;
        let noise: Vec<f32> = (0..576 * 4)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((state >> 16) as f32 / 32_768.0 - 1.0) * 0.3
            })
            .collect();
        attack.feed(&noise[..576]);
        for g in 1..4 {
            assert!(!attack.feed(&noise[576 * g..576 * (g + 1)]), "granule {g}");
        }
    }
}
