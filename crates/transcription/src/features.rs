use std::sync::OnceLock;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

/// The rate the local recognizer accepts.
pub const SAMPLE_RATE: u32 = 16_000;
/// Analysis window, 25 ms at [`SAMPLE_RATE`].
pub const WINDOW: usize = 400;
/// Hop between frames, 10 ms at [`SAMPLE_RATE`].
pub const HOP: usize = 160;
/// FFT size. The window is zero-padded to this length.
pub const N_FFT: usize = 512;
/// Mel bins the acoustic model reads.
pub const N_MELS: usize = 80;
/// One frame, in milliseconds. This matches [`HOP`].
pub const HOP_MS: i64 = 10;

const FLOOR: f32 = 1.0e-10;

struct MelFilter {
    #[cfg(test)]
    center_hz: f32,
    weights: Vec<(usize, f32)>,
}

/// How many frames [`log_mel`] emits for `samples` mono samples.
#[must_use]
pub fn frame_count(samples: usize) -> usize {
    if samples == 0 {
        return 0;
    }
    let padded = samples.max(WINDOW);
    1 + (padded - WINDOW) / HOP
}

/// Log-mel energies, row-major `[frames, N_MELS]`.
pub fn log_mel(samples: &[f32]) -> Result<Vec<f32>, FeatureError> {
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(FeatureError::Samples);
    }
    let frames = frame_count(samples.len());
    if frames == 0 {
        return Ok(Vec::new());
    }
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let filters = filterbank();
    let mut window = vec![0.0; WINDOW];
    let mut spectrum = vec![Complex::new(0.0, 0.0); N_FFT];
    let mut power = vec![0.0; N_FFT / 2 + 1];
    let mut mel = vec![0.0; frames * N_MELS];
    for frame in 0..frames {
        fill_window(samples, frame, &mut window);
        power_spectrum(&window, fft.as_ref(), &mut spectrum, &mut power);
        let row = frame * N_MELS;
        for (bin, filter) in filters.iter().enumerate() {
            let mut energy = 0.0;
            for (index, weight) in &filter.weights {
                energy += power[*index] * weight;
            }
            mel[row + bin] = energy.max(FLOOR).ln();
        }
    }
    Ok(mel)
}

/// Center frequency of mel bin `index`, in hertz.
#[cfg(test)]
#[must_use]
pub fn mel_center_hz(index: usize) -> Option<f32> {
    filterbank().get(index).map(|filter| filter.center_hz)
}

/// Why a waveform did not become mel frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureError {
    /// A sample was NaN or infinite.
    Samples,
}

fn fill_window(samples: &[f32], frame: usize, window: &mut [f32]) {
    let start = frame * HOP;
    for (index, slot) in window.iter_mut().enumerate() {
        let sample = samples.get(start + index).copied().unwrap_or(0.0);
        let hann = 0.5 * (1.0 - (std::f32::consts::TAU * index as f32 / (WINDOW - 1) as f32).cos());
        *slot = sample * hann;
    }
}

fn power_spectrum(
    window: &[f32],
    fft: &dyn Fft<f32>,
    spectrum: &mut [Complex<f32>],
    power: &mut [f32],
) {
    for bin in spectrum.iter_mut() {
        *bin = Complex::new(0.0, 0.0);
    }
    for (bin, sample) in spectrum.iter_mut().zip(window) {
        *bin = Complex::new(*sample, 0.0);
    }
    fft.process(spectrum);
    for (bin, value) in power.iter_mut().zip(spectrum.iter()) {
        *bin = value.re.mul_add(value.re, value.im * value.im);
    }
}

fn filterbank() -> &'static [MelFilter] {
    static BANK: OnceLock<Vec<MelFilter>> = OnceLock::new();
    BANK.get_or_init(build_filterbank)
}

fn build_filterbank() -> Vec<MelFilter> {
    let nyquist = SAMPLE_RATE as f32 / 2.0;
    let points = N_MELS + 2;
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(nyquist);
    let mut edges = Vec::with_capacity(points);
    for index in 0..points {
        let mel = mel_min + (mel_max - mel_min) * index as f32 / (points - 1) as f32;
        edges.push(mel_to_hz(mel) * N_FFT as f32 / SAMPLE_RATE as f32);
    }
    let bins = N_FFT / 2 + 1;
    let mut filters = Vec::with_capacity(N_MELS);
    for index in 0..N_MELS {
        let left = edges[index];
        let center = edges[index + 1];
        let right = edges[index + 2];
        let mut weights = Vec::new();
        for bin in 0..bins {
            let bin_f = bin as f32;
            let weight = if bin_f >= left && bin_f <= center && center > left {
                (bin_f - left) / (center - left)
            } else if bin_f >= center && bin_f <= right && right > center {
                (right - bin_f) / (right - center)
            } else {
                0.0
            };
            if weight > 0.0 {
                weights.push((bin, weight));
            }
        }
        filters.push(MelFilter {
            #[cfg(test)]
            center_hz: center * SAMPLE_RATE as f32 / N_FFT as f32,
            weights,
        });
    }
    filters
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_counts_follow_the_hop() {
        assert_eq!(frame_count(0), 0);
        assert_eq!(frame_count(1), 1);
        assert_eq!(frame_count(WINDOW), 1);
        assert_eq!(frame_count(16_000), 98);
        assert!(log_mel(&[]).unwrap().is_empty());
        assert_eq!(log_mel(&vec![0.0; WINDOW]).unwrap().len(), N_MELS);
    }

    #[test]
    fn a_tone_peaks_near_its_frequency() {
        let mut samples = Vec::with_capacity(16_000);
        for index in 0..16_000 {
            let time = index as f32 / SAMPLE_RATE as f32;
            samples.push((std::f32::consts::TAU * 1_000.0 * time).sin());
        }
        let mel = log_mel(&samples).unwrap();
        assert_eq!(mel.len(), 98 * N_MELS);
        assert!(mel.iter().all(|value| value.is_finite()));
        let mut best = 0;
        let mut best_value = f32::MIN;
        for (bin, value) in mel.iter().take(N_MELS).enumerate() {
            if *value > best_value {
                best_value = *value;
                best = bin;
            }
        }
        let center = mel_center_hz(best).unwrap();
        assert!((center - 1_000.0).abs() < 200.0, "center {center}");
    }

    #[test]
    fn a_non_finite_sample_is_refused() {
        assert_eq!(log_mel(&[f32::NAN]).unwrap_err(), FeatureError::Samples);
    }
}
