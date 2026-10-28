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

/// Preemphasis used by the Parakeet frontend.
const PREEMPHASIS: f32 = 0.97;
/// Floor added before the log, `2^-24`.
const LOG_GUARD: f32 = 1.0 / 16_777_216.0;
/// Added to the per-bin standard deviation.
const NORM_EPS: f32 = 1.0e-5;
/// Encoder frames are one mel hop times the FastConformer stride.
pub const ENCODER_HOP_MS: i64 = HOP_MS * 8;

/// How many Parakeet mel frames `samples` produces.
///
/// The transform is centered, and the last centered frame is dropped so the
/// length matches the mask the model was trained with. Fewer than two frames
/// cannot be normalized, so that input yields none.
#[must_use]
pub fn conformer_frames(samples: usize) -> usize {
    let frames = samples / HOP;
    if frames < 2 { 0 } else { frames }
}

/// Parakeet log-mel, row-major `[frames, N_MELS]`, mean and variance removed.
pub fn conformer_mel(samples: &[f32]) -> Result<Vec<f32>, FeatureError> {
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(FeatureError::Samples);
    }
    let frames = conformer_frames(samples.len());
    if frames == 0 {
        return Ok(Vec::new());
    }
    let emphasized = preemphasis(samples);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let window = analysis_window();
    let filters = slaney_bank();
    let mut spectrum = vec![Complex::new(0.0, 0.0); N_FFT];
    let mut power = vec![0.0; N_FFT / 2 + 1];
    let mut mel = vec![0.0; frames * N_MELS];
    for frame in 0..frames {
        centered_spectrum(
            &emphasized,
            frame,
            window,
            fft.as_ref(),
            &mut spectrum,
            &mut power,
        );
        let row = frame * N_MELS;
        for (bin, filter) in filters.iter().enumerate() {
            let mut energy = 0.0;
            for (index, weight) in &filter.weights {
                energy += power[*index] * weight;
            }
            mel[row + bin] = (energy + LOG_GUARD).ln();
        }
    }
    normalize_bins(&mut mel, frames);
    Ok(mel)
}

fn preemphasis(samples: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(samples.len());
    let Some((first, rest)) = samples.split_first() else {
        return out;
    };
    out.push(*first);
    let mut previous = *first;
    for sample in rest {
        out.push(sample - PREEMPHASIS * previous);
        previous = *sample;
    }
    out
}

fn analysis_window() -> &'static [f32] {
    static WINDOW_512: OnceLock<Vec<f32>> = OnceLock::new();
    WINDOW_512.get_or_init(|| {
        let mut window = vec![0.0; N_FFT];
        let left = (N_FFT - WINDOW) / 2;
        for index in 0..WINDOW {
            let hann =
                0.5 * (1.0 - (std::f32::consts::TAU * index as f32 / (WINDOW - 1) as f32).cos());
            window[left + index] = hann;
        }
        window
    })
}

fn centered_spectrum(
    samples: &[f32],
    frame: usize,
    window: &[f32],
    fft: &dyn Fft<f32>,
    spectrum: &mut [Complex<f32>],
    power: &mut [f32],
) {
    let origin = frame * HOP;
    let pad = N_FFT / 2;
    for (index, bin) in spectrum.iter_mut().enumerate() {
        let sample_at = origin + index;
        let sample = sample_at
            .checked_sub(pad)
            .and_then(|at| samples.get(at))
            .copied()
            .unwrap_or(0.0);
        *bin = Complex::new(sample * window[index], 0.0);
    }
    fft.process(spectrum);
    for (bin, value) in power.iter_mut().zip(spectrum.iter()) {
        *bin = value.re.mul_add(value.re, value.im * value.im);
    }
}

fn normalize_bins(mel: &mut [f32], frames: usize) {
    let divisor = (frames - 1) as f32;
    for bin in 0..N_MELS {
        let mut sum = 0.0;
        for frame in 0..frames {
            sum += mel[frame * N_MELS + bin];
        }
        let mean = sum / frames as f32;
        let mut square = 0.0;
        for frame in 0..frames {
            let delta = mel[frame * N_MELS + bin] - mean;
            square += delta * delta;
        }
        let scale = (square / divisor).sqrt() + NORM_EPS;
        for frame in 0..frames {
            let value = &mut mel[frame * N_MELS + bin];
            *value = (*value - mean) / scale;
        }
    }
}

fn slaney_bank() -> &'static [MelFilter] {
    static BANK: OnceLock<Vec<MelFilter>> = OnceLock::new();
    BANK.get_or_init(build_slaney_bank)
}

fn build_slaney_bank() -> Vec<MelFilter> {
    let points = N_MELS + 2;
    let mel_max = hz_to_mel_slaney(SAMPLE_RATE as f32 / 2.0);
    let mut edges = Vec::with_capacity(points);
    for index in 0..points {
        let mel = mel_max * index as f32 / (points - 1) as f32;
        edges.push(mel_to_hz_slaney(mel));
    }
    let bins = N_FFT / 2 + 1;
    let mut filters = Vec::with_capacity(N_MELS);
    for index in 0..N_MELS {
        let left = edges[index];
        let center = edges[index + 1];
        let right = edges[index + 2];
        let width = edges[index + 2] - edges[index];
        let norm = if width > 0.0 { 2.0 / width } else { 0.0 };
        let mut weights = Vec::new();
        for bin in 0..bins {
            let hz = bin as f32 * SAMPLE_RATE as f32 / N_FFT as f32;
            let lower = if center > left {
                (hz - left) / (center - left)
            } else {
                0.0
            };
            let upper = if right > center {
                (right - hz) / (right - center)
            } else {
                0.0
            };
            let weight = lower.min(upper).max(0.0) * norm;
            if weight > 0.0 {
                weights.push((bin, weight));
            }
        }
        filters.push(MelFilter {
            #[cfg(test)]
            center_hz: center,
            weights,
        });
    }
    filters
}

fn hz_to_mel_slaney(hz: f32) -> f32 {
    let step = 200.0 / 3.0;
    let min_log_hz = 1_000.0;
    let min_log_mel = min_log_hz / step;
    let logstep = 6.4f32.ln() / 27.0;
    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        hz / step
    }
}

fn mel_to_hz_slaney(mel: f32) -> f32 {
    let step = 200.0 / 3.0;
    let min_log_hz = 1_000.0;
    let min_log_mel = min_log_hz / step;
    let logstep = 6.4f32.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        step * mel
    }
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

    #[test]
    fn conformer_frames_drop_the_last_centered_frame() {
        assert_eq!(conformer_frames(0), 0);
        assert_eq!(conformer_frames(HOP), 0);
        assert_eq!(conformer_frames(16_000), 100);
    }

    #[test]
    fn silence_stays_near_zero() {
        let mel = conformer_mel(&vec![0.0; 16_000]).unwrap();
        assert_eq!(mel.len(), 100 * N_MELS);
        assert!(mel.iter().all(|value| value.is_finite()));
        let peak = mel.iter().fold(0.0f32, |peak, value| peak.max(value.abs()));
        assert!(peak < 1.0, "peak {peak}");
    }

    #[test]
    fn a_conformer_tone_is_finite() {
        let mut samples = Vec::with_capacity(16_000);
        for index in 0..16_000 {
            let time = index as f32 / SAMPLE_RATE as f32;
            samples.push((std::f32::consts::TAU * 1_000.0 * time).sin());
        }
        let mel = conformer_mel(&samples).unwrap();
        assert_eq!(mel.len(), 100 * N_MELS);
        assert!(mel.iter().all(|value| value.is_finite()));
        assert!(mel.iter().any(|value| value.abs() > 0.1));
    }
}
