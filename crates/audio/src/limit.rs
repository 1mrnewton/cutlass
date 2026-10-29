//! Lookahead peak limiter for the mixed buffer.
//!
//! Kind 0 skips the limiter. Any other kind limits. Samples are full-scale
//! `f32`, so an `i16` buffer is judged after dividing by 32767 and the
//! ceiling is 1.0 either way. Lookahead is 5 ms and release is 50 ms.
//! Those times are the usual limiter defaults; the trace did not record an
//! attack or a release. Kind 2 was traced with makeup gain 1.0, and no
//! other kind-specific law was traced, so every enabled kind uses unity
//! makeup.
//!
//! Channels share one gain, taken from the loudest channel. A buffer that
//! already sits inside the ceiling is copied.

use std::collections::VecDeque;

use cutlass_codec::access::Pcm;

use crate::Error;
use crate::buffer::{Built, Input};

/// Full-scale ceiling for `f32` samples.
const CEILING: f32 = 1.0;

/// Limit `input` when `kind` is not 0.
///
/// Kind 0 copies the buffer, including peaks above full scale. An empty
/// buffer stays empty. A zero sample rate is refused.
pub fn limit(input: &Pcm, kind: i32) -> Result<Pcm, Error> {
    let source = Input::open(input)?;
    if kind == 0 || source.frames == 0 {
        return Ok(input.clone());
    }
    if peak_of(&source) <= CEILING {
        return Ok(input.clone());
    }
    let attack = lookahead_frames(source.rate);
    let release = release_frames(source.rate).max(1.0);
    let peaks = frame_peaks(&source);
    let envelope = lookahead_max(&peaks, attack);
    let mut out = Built::new(source.format, input.samples, source.channels)?;
    let mut gain = needed(envelope.first().copied().unwrap_or(0.0));
    let mut origin = gain;
    let mut origin_at = 0_usize;
    for (frame, env) in envelope.iter().copied().enumerate() {
        let target = needed(env);
        if target < gain {
            gain = target;
            origin = target;
            origin_at = frame;
        } else if gain < 1.0 {
            let elapsed = frame - origin_at;
            let risen = f64::from(origin) + (1.0 - f64::from(origin)) * (elapsed as f64 / release);
            gain = (risen as f32).min(target).min(1.0);
        }
        let cap = CEILING;
        for channel in 0..source.channels {
            let value = (source.sample(frame, channel) * gain).clamp(-cap, cap);
            out.push_unit(value);
        }
    }
    out.pack(source.rate, source.layout.clone())
}

fn lookahead_frames(rate: u32) -> usize {
    usize::try_from(rate / 200).unwrap_or(1).max(1)
}

fn release_frames(rate: u32) -> f64 {
    f64::from(rate / 20).max(1.0)
}

fn peak_of(input: &Input<'_>) -> f32 {
    let mut peak = 0.0_f32;
    for frame in 0..input.frames {
        for channel in 0..input.channels {
            let sample = input.sample(frame, channel).abs();
            if sample > peak {
                peak = sample;
            }
        }
    }
    peak
}

fn frame_peaks(input: &Input<'_>) -> Vec<f32> {
    let mut peaks = Vec::with_capacity(input.frames);
    for frame in 0..input.frames {
        let mut peak = 0.0_f32;
        for channel in 0..input.channels {
            let sample = input.sample(frame, channel).abs();
            if sample > peak {
                peak = sample;
            }
        }
        peaks.push(peak);
    }
    peaks
}

fn needed(peak: f32) -> f32 {
    if peak <= CEILING || peak == 0.0 {
        1.0
    } else {
        CEILING / peak
    }
}

fn lookahead_max(peaks: &[f32], attack: usize) -> Vec<f32> {
    let mut envelope = vec![0.0_f32; peaks.len()];
    let mut window: VecDeque<usize> = VecDeque::new();
    let mut right = 0_usize;
    for (frame, slot) in envelope.iter_mut().enumerate() {
        let end = (frame + attack).min(peaks.len());
        while right < end {
            while window
                .back()
                .is_some_and(|index| peaks[*index] <= peaks[right])
            {
                window.pop_back();
            }
            window.push_back(right);
            right += 1;
        }
        while window.front().is_some_and(|index| *index < frame) {
            window.pop_front();
        }
        if let Some(index) = window.front().copied() {
            *slot = peaks[index];
        }
    }
    envelope
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use cutlass_codec::access::{ChannelLayout, Pcm, SampleFormat};

    use super::limit;
    use crate::Error;

    fn mono(samples: Vec<f32>) -> Pcm {
        match Pcm::from_f32(48_000, ChannelLayout::mono(), samples) {
            Ok(pcm) => pcm,
            Err(err) => panic!("pcm: {err:?}"),
        }
    }

    fn data(pcm: &Pcm) -> &[f32] {
        match pcm.as_f32() {
            Some(samples) => samples,
            None => panic!("expected f32"),
        }
    }

    #[test]
    fn a_buffer_inside_full_scale_is_copied() {
        let input = mono(vec![0.25, -0.5, 0.0, 1.0]);
        let output = limit(&input, 1).expect("limit");
        assert_eq!(data(&output), data(&input));
    }

    #[test]
    fn kind_zero_leaves_a_peak_above_full_scale() {
        let input = mono(vec![2.0, -1.5]);
        let output = limit(&input, 0).expect("limit");
        assert_eq!(data(&output), data(&input));
    }

    #[test]
    fn a_constant_over_full_scale_lands_on_the_ceiling() {
        let input = mono(vec![2.0; 64]);
        for kind in [1_i32, 2, -3] {
            let output = limit(&input, kind).expect("limit");
            assert_eq!(data(&output), &[1.0; 64], "kind {kind}");
        }
    }

    #[test]
    fn a_spike_never_exceeds_the_ceiling_and_stays_local() {
        let mut samples = vec![0.1_f32; 2_000];
        samples[1_000] = 4.0;
        let output = limit(&mono(samples), 1).expect("limit");
        let data = data(&output);
        for sample in data {
            assert!(sample.abs() <= 1.0 + 1.0e-5, "{sample}");
        }
        assert!((data[1_000] - 1.0).abs() < 1.0e-5, "{}", data[1_000]);
        assert_eq!(data[700], 0.1);
    }

    #[test]
    fn channels_share_the_gain_of_the_louder_one() {
        let input = match Pcm::from_f32(48_000, ChannelLayout::stereo(), vec![2.0, 0.5]) {
            Ok(pcm) => pcm,
            Err(err) => panic!("{err:?}"),
        };
        let output = limit(&input, 1).expect("limit");
        let samples = data(&output);
        assert!((samples[0] - 1.0).abs() < 1.0e-5, "{}", samples[0]);
        assert!((samples[1] - 0.25).abs() < 1.0e-5, "{}", samples[1]);
    }

    #[test]
    fn i16_full_scale_is_unchanged() {
        let input = match Pcm::from_i16(48_000, ChannelLayout::mono(), vec![0, 32_767, -32_767]) {
            Ok(pcm) => pcm,
            Err(err) => panic!("{err:?}"),
        };
        let output = limit(&input, 1).expect("limit");
        assert_eq!(output.format, SampleFormat::I16);
        assert_eq!(output.as_i16(), input.as_i16());
    }

    #[test]
    fn empty_input_stays_empty() {
        let input = mono(Vec::new());
        let output = limit(&input, 1).expect("limit");
        assert_eq!(output.samples, 0);
    }

    #[test]
    fn a_zero_rate_or_a_non_finite_sample_is_refused() {
        let broken = match Pcm::from_f32(0, ChannelLayout::mono(), vec![0.2]) {
            Ok(pcm) => pcm,
            Err(err) => panic!("{err:?}"),
        };
        assert_eq!(limit(&broken, 1), Err(Error::Rate));
        let input = mono(vec![f32::NAN]);
        assert_eq!(limit(&input, 1), Err(Error::NonFinite));
    }
}
