//! Kaiser-windowed sinc resample.
//!
//! The kernel is 65 taps: 32 samples on each side of the center. Downsampling
//! sets the cutoff at the output Nyquist. Taps that fall outside the buffer
//! are omitted and the remaining kernel is scaled to unit gain at DC, so a
//! constant stays a constant at a cut. A ratio that lands on an input frame
//! copies that frame. The caller picks the mix rate.

use cutlass_codec::access::Pcm;

use crate::Error;
use crate::buffer::{Built, Input, as_i64, empty_like, scaled_rate};

/// Taps on each side of the center, not counting the center tap.
pub(crate) const RADIUS: i64 = 32;

/// Kaiser window beta. About 60 dB of stopband attenuation.
const BETA: f64 = 8.5;

/// `I0(BETA)`, computed once per render.
fn i0_beta() -> f64 {
    bessel_i0(BETA)
}

/// Resample `input` to `target_rate`.
///
/// An empty buffer stays empty at `target_rate`. The same rate is a copy.
/// A zero rate is refused. The output frame count is
/// `round(frames * target_rate / source_rate)`.
pub fn resample(input: &Pcm, target_rate: u32) -> Result<Pcm, Error> {
    if target_rate == 0 {
        return Err(Error::Rate);
    }
    let source = Input::open(input)?;
    if source.frames == 0 {
        return empty_like(&source, target_rate);
    }
    if source.rate == target_rate {
        return Ok(input.clone());
    }
    let frames = scaled_rate(input.samples, source.rate, target_rate)?;
    render(
        &source,
        frames,
        target_rate,
        Step::Ratio {
            source: source.rate,
            target: target_rate,
        },
    )
}

/// How far the read head moves for each output frame.
pub(crate) enum Step {
    /// `source / target` input frames per output frame.
    Ratio { source: u32, target: u32 },
    /// Playback speed. Pitch follows this step, so a speed above 1 lowpasses.
    Speed(f64),
}

impl Step {
    fn cutoff(&self) -> f64 {
        match *self {
            Self::Ratio { source, target } => {
                if source > target {
                    f64::from(target) / f64::from(source)
                } else {
                    1.0
                }
            }
            Self::Speed(speed) => {
                if speed > 1.0 {
                    1.0 / speed
                } else {
                    1.0
                }
            }
        }
    }

    fn center(&self, index: u32) -> f64 {
        match *self {
            Self::Ratio { source, target } => {
                let product = u128::from(index) * u128::from(source);
                let den = u128::from(target);
                let whole = product / den;
                let rem = product % den;
                whole as f64 + (rem as f64) / f64::from(target)
            }
            Self::Speed(speed) => f64::from(index) * speed,
        }
    }

    /// Input frame index when this output frame is an impulse of the kernel.
    fn exact_frame(&self, index: u32) -> Option<usize> {
        match *self {
            Self::Ratio { source, target } => {
                if source > target {
                    return None;
                }
                let product = u128::from(index) * u128::from(source);
                let den = u128::from(target);
                if product % den != 0 {
                    return None;
                }
                usize::try_from(product / den).ok()
            }
            Self::Speed(speed) => {
                if speed > 1.0 {
                    return None;
                }
                let pos = f64::from(index) * speed;
                let nearest = pos.round();
                if (pos - nearest).abs() > 1e-8 || nearest < 0.0 {
                    return None;
                }
                let whole = as_i64(nearest)?;
                usize::try_from(whole).ok()
            }
        }
    }
}

pub(crate) fn render(
    input: &Input<'_>,
    output_frames: u32,
    output_rate: u32,
    step: Step,
) -> Result<Pcm, Error> {
    if output_frames == 0 {
        return empty_like(input, output_rate);
    }
    let mut out = Built::new(input.format, output_frames, input.channels)?;
    let fc = step.cutoff();
    let norm = i0_beta();
    for index in 0..output_frames {
        if let Some(frame) = step
            .exact_frame(index)
            .filter(|frame| *frame < input.frames)
        {
            for channel in 0..input.channels {
                out.push_copied(input, frame, channel);
            }
            continue;
        }
        let center = step.center(index);
        let mut acc = [0.0_f64; 8];
        let weight = accumulate(&mut acc, input, center, fc, norm);
        for slot in acc.iter().take(input.channels) {
            let value = if weight.abs() < 1.0e-20 {
                0.0
            } else {
                slot / weight
            };
            out.push_unit(value as f32);
        }
    }
    out.pack(output_rate, input.layout.clone())
}

fn accumulate(acc: &mut [f64; 8], input: &Input<'_>, center: f64, fc: f64, norm: f64) -> f64 {
    let half = RADIUS as f64;
    let Some(first) = as_i64((center - half).ceil()) else {
        return 0.0;
    };
    let Some(last) = as_i64((center + half).floor()) else {
        return 0.0;
    };
    let mut weight = 0.0;
    let mut idx = first;
    while idx <= last {
        let offset = idx as f64 - center;
        let coeff = fc * sinc(fc * offset) * kaiser(offset, norm);
        if let Ok(frame) = usize::try_from(idx)
            && frame < input.frames
        {
            weight += coeff;
            for (channel, slot) in acc.iter_mut().enumerate().take(input.channels) {
                *slot += f64::from(input.sample(frame, channel)) * coeff;
            }
        }
        idx += 1;
    }
    weight
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1.0e-12 {
        1.0
    } else {
        let pix = x * std::f64::consts::PI;
        pix.sin() / pix
    }
}

fn kaiser(offset: f64, norm: f64) -> f64 {
    let x = offset / RADIUS as f64;
    if x.abs() > 1.0 || norm == 0.0 {
        return 0.0;
    }
    let inside = (1.0 - x * x).max(0.0).sqrt();
    bessel_i0(BETA * inside) / norm
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half_sq = (x * 0.5) * (x * 0.5);
    for k in 1..=24 {
        let kf = f64::from(k);
        term *= half_sq / (kf * kf);
        sum += term;
    }
    sum
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use cutlass_codec::access::{ChannelLayout, Pcm, SampleFormat};

    use super::{RADIUS, resample};
    use crate::Error;

    fn mono(samples: Vec<f32>, rate: u32) -> Pcm {
        match Pcm::from_f32(rate, ChannelLayout::mono(), samples) {
            Ok(pcm) => pcm,
            Err(err) => panic!("pcm: {err:?}"),
        }
    }

    fn samples(pcm: &Pcm) -> &[f32] {
        match pcm.as_f32() {
            Some(data) => data,
            None => panic!("expected f32"),
        }
    }

    #[test]
    fn same_rate_copies_every_sample() {
        let input = mono(vec![0.0, -0.25, 0.5, 1.0], 48_000);
        let output = resample(&input, 48_000).expect("resample");
        assert_eq!(samples(&output), samples(&input));
        assert_eq!(output.rate, 48_000);
    }

    #[test]
    fn upsample_by_two_keeps_source_frames_on_even_outputs() {
        let input = mono(vec![0.0, 1.0], 8_000);
        let output = resample(&input, 16_000).expect("resample");
        let data = samples(&output);
        assert_eq!(output.samples, 4);
        assert_eq!(data[0], 0.0);
        assert_eq!(data[2], 1.0);
        assert!(data[1] > 0.3 && data[1] < 0.7, "midpoint {}", data[1]);
    }

    #[test]
    fn a_constant_stays_constant_at_a_non_integer_ratio() {
        let input = mono(vec![0.25; 64], 48_000);
        let output = resample(&input, 44_100).expect("resample");
        assert_eq!(output.rate, 44_100);
        assert!(output.samples > 0);
        for sample in samples(&output) {
            assert!((sample - 0.25).abs() < 1.0e-5, "{sample}");
        }
    }

    #[test]
    fn downsampling_removes_energy_above_the_new_nyquist() {
        let rate = 8_000_u32;
        let frames = 4_096_usize;
        let mut input = Vec::with_capacity(frames);
        for n in 0..frames {
            let t = n as f64 / f64::from(rate);
            input.push((2.0 * std::f64::consts::PI * 3_000.0 * t).sin() as f32);
        }
        let output = resample(&mono(input, rate), 2_000).expect("resample");
        let data = samples(&output);
        let margin = RADIUS as usize + 8;
        let body = &data[margin..data.len() - margin];
        let energy: f64 = body
            .iter()
            .map(|sample| f64::from(*sample) * f64::from(*sample))
            .sum();
        let rms = (energy / body.len() as f64).sqrt();
        assert!(rms < 0.05, "stopband rms {rms} (kernel radius {RADIUS})");
    }

    #[test]
    fn stereo_channels_stay_on_their_own_even_frames() {
        let input = match Pcm::from_f32(8_000, ChannelLayout::stereo(), vec![0.0, 0.5, 1.0, -0.5]) {
            Ok(pcm) => pcm,
            Err(err) => panic!("{err:?}"),
        };
        let output = resample(&input, 16_000).expect("resample");
        let data = samples(&output);
        assert_eq!(output.layout.channels().len(), 2);
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 0.5);
        assert_eq!(data[4], 1.0);
        assert_eq!(data[5], -0.5);
    }

    #[test]
    fn empty_input_stays_empty_at_the_target_rate() {
        let input = mono(Vec::new(), 48_000);
        let output = resample(&input, 44_100).expect("resample");
        assert_eq!(output.samples, 0);
        assert_eq!(output.rate, 44_100);
        assert_eq!(output.format, SampleFormat::F32);
    }

    #[test]
    fn a_zero_rate_is_refused() {
        let input = mono(vec![0.0], 48_000);
        assert_eq!(resample(&input, 0), Err(Error::Rate));
        let broken = match Pcm::from_f32(0, ChannelLayout::mono(), vec![0.0]) {
            Ok(pcm) => pcm,
            Err(err) => panic!("{err:?}"),
        };
        assert_eq!(resample(&broken, 48_000), Err(Error::Rate));
    }

    #[test]
    fn a_non_finite_sample_is_refused() {
        let input = mono(vec![f32::NAN], 48_000);
        assert_eq!(resample(&input, 44_100), Err(Error::NonFinite));
        let input = mono(vec![f32::INFINITY], 48_000);
        assert_eq!(resample(&input, 44_100), Err(Error::NonFinite));
    }

    #[test]
    fn an_output_longer_than_the_frame_counter_allocates_nothing() {
        let input = mono(vec![0.0; 100_000], 1);
        assert_eq!(resample(&input, 100_000), Err(Error::TooLong));
    }

    #[test]
    fn i16_upsample_keeps_source_codes_on_even_frames() {
        let input = match Pcm::from_i16(8_000, ChannelLayout::mono(), vec![0, 1_000, -2_000]) {
            Ok(pcm) => pcm,
            Err(err) => panic!("{err:?}"),
        };
        let output = resample(&input, 16_000).expect("resample");
        let data = match output.as_i16() {
            Some(data) => data,
            None => panic!("expected i16"),
        };
        assert_eq!(output.samples, 6);
        assert_eq!(data[0], 0);
        assert_eq!(data[2], 1_000);
        assert_eq!(data[4], -2_000);
    }

    #[test]
    fn i16_same_rate_keeps_the_codes() {
        let input = match Pcm::from_i16(48_000, ChannelLayout::mono(), vec![0, 1_000, -32_768]) {
            Ok(pcm) => pcm,
            Err(err) => panic!("{err:?}"),
        };
        let output = resample(&input, 48_000).expect("resample");
        assert_eq!(output.as_i16(), input.as_i16());
    }
}
