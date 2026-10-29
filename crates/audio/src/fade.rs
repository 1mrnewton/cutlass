//! Per-clip fade.
//!
//! Durations are microseconds on the same clock as a segment. Curve 0 skips
//! the fade. Curve 1 is a quarter sine, 2 is linear, 3 is logarithmic, and
//! 4 is exponential. Those are the five curves that were traced. A fade
//! longer than the buffer covers the buffer. Overlapping in and out
//! multiply. Samples outside a fade are copied.

use cutlass_codec::access::Pcm;

use crate::Error;
use crate::buffer::{Built, Input};

/// Curve stored on a fade material.
///
/// The numbers are the material's `fade_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FadeCurve {
    /// Curve 0. Samples stay as they are.
    Skip = 0,
    /// Curve 1. Quarter sine, `sin(t * π / 2)`.
    Sine = 1,
    /// Curve 2. Linear slope.
    Linear = 2,
    /// Curve 3. Logarithmic.
    Logarithmic = 3,
    /// Curve 4. Exponential.
    Exponential = 4,
}

impl FadeCurve {
    /// Map a material `fade_type`. Any other number is refused.
    pub const fn from_type(value: i32) -> Result<Self, Error> {
        match value {
            0 => Ok(Self::Skip),
            1 => Ok(Self::Sine),
            2 => Ok(Self::Linear),
            3 => Ok(Self::Logarithmic),
            4 => Ok(Self::Exponential),
            _ => Err(Error::Fade),
        }
    }
}

/// Apply a fade-in and a fade-out.
///
/// Durations are microseconds. Zero leaves that edge alone. A negative
/// duration is refused. Curve [`FadeCurve::Skip`] copies the buffer.
pub fn fade(
    input: &Pcm,
    fade_in_us: i64,
    fade_out_us: i64,
    curve: FadeCurve,
) -> Result<Pcm, Error> {
    let source = Input::open(input)?;
    if source.frames == 0 {
        return Ok(input.clone());
    }
    if matches!(curve, FadeCurve::Skip) {
        return Ok(input.clone());
    }
    let frames = input.samples;
    let fade_in = microseconds_to_frames(fade_in_us, source.rate, frames)?;
    let fade_out = microseconds_to_frames(fade_out_us, source.rate, frames)?;
    if fade_in == 0 && fade_out == 0 {
        return Ok(input.clone());
    }
    let mut out = Built::new(source.format, frames, source.channels)?;
    for frame in 0..frames {
        let gain = region_gain(curve, frame, frames, fade_in, fade_out);
        for channel in 0..source.channels {
            let index = usize::try_from(frame).map_err(|_| Error::TooLong)?;
            if gain == 1.0 {
                out.push_copied(&source, index, channel);
            } else {
                out.push_unit(source.sample(index, channel) * gain);
            }
        }
    }
    out.pack(source.rate, source.layout.clone())
}

fn microseconds_to_frames(duration: i64, rate: u32, frames: u32) -> Result<u32, Error> {
    if duration < 0 {
        return Err(Error::Fade);
    }
    if duration == 0 || frames == 0 {
        return Ok(0);
    }
    let count = i128::from(duration) * i128::from(rate) / 1_000_000;
    if count <= 0 {
        return Ok(0);
    }
    let count = u32::try_from(count).unwrap_or(u32::MAX);
    Ok(count.min(frames))
}

fn region_gain(curve: FadeCurve, frame: u32, frames: u32, fade_in: u32, fade_out: u32) -> f32 {
    let mut gain = 1.0_f64;
    if fade_in > 0 && frame < fade_in {
        gain *= curve_gain(curve, frame, fade_in);
    }
    if fade_out > 0 {
        let from_end = frames - 1 - frame;
        if from_end < fade_out {
            gain *= curve_gain(curve, from_end, fade_out);
        }
    }
    gain as f32
}

/// Gain at `index / range`, before silence is scaled to unity.
///
/// `index` runs from 0 at silence toward `range` at unity. The formulas
/// match the traced quarter-sine, linear, log, and exp fades.
fn curve_gain(curve: FadeCurve, index: u32, range: u32) -> f64 {
    if range == 0 {
        return 1.0;
    }
    let gain = (f64::from(index) / f64::from(range)).clamp(0.0, 1.0);
    match curve {
        FadeCurve::Skip => 1.0,
        FadeCurve::Linear => gain,
        FadeCurve::Sine => (gain * std::f64::consts::FRAC_PI_2).sin(),
        FadeCurve::Logarithmic => {
            if gain == 0.0 {
                0.0
            } else {
                (1.0 + 0.2 * gain.log10()).clamp(0.0, 1.0)
            }
        }
        FadeCurve::Exponential => (-11.512_925_464_970_227 * (1.0 - gain)).exp(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use cutlass_codec::access::{ChannelLayout, Pcm};

    use super::{FadeCurve, fade};

    fn mono(samples: Vec<f32>, rate: u32) -> Pcm {
        match Pcm::from_f32(rate, ChannelLayout::mono(), samples) {
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
    fn linear_fade_in_matches_the_slope() {
        let input = mono(vec![1.0; 8], 4_000);
        let output = fade(&input, 1_000, 0, FadeCurve::Linear).expect("fade");
        assert_eq!(data(&output), &[0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn quarter_sine_matches_the_known_midpoint() {
        let input = mono(vec![1.0, 1.0], 4_000);
        let output = fade(&input, 500, 0, FadeCurve::Sine).expect("fade");
        let samples = data(&output);
        assert_eq!(samples[0], 0.0);
        assert!((samples[1] - 0.707_106_77).abs() < 1.0e-6, "{}", samples[1]);
    }

    #[test]
    fn logarithmic_and_exponential_match_known_gains() {
        let input = mono(vec![1.0, 1.0], 4_000);
        let log = fade(&input, 500, 0, FadeCurve::Logarithmic).expect("fade");
        let exp = fade(&input, 500, 0, FadeCurve::Exponential).expect("fade");
        assert_eq!(data(&log)[0], 0.0);
        assert!(
            (data(&log)[1] - 0.939_794).abs() < 1.0e-5,
            "{}",
            data(&log)[1]
        );
        assert!((data(&exp)[0] - 1.0e-5).abs() < 1.0e-7, "{}", data(&exp)[0]);
        assert!(
            (data(&exp)[1] - 0.003_162_277_7).abs() < 1.0e-6,
            "{}",
            data(&exp)[1]
        );
    }

    #[test]
    fn overlapping_linear_fades_multiply() {
        let input = mono(vec![1.0; 4], 1_000);
        let output = fade(&input, 4_000, 4_000, FadeCurve::Linear).expect("fade");
        assert_eq!(data(&output), &[0.0, 0.125, 0.125, 0.0]);
    }

    #[test]
    fn a_fade_longer_than_the_buffer_covers_every_frame() {
        let input = mono(vec![1.0; 4], 48_000);
        let output = fade(&input, 10_000_000, 0, FadeCurve::Linear).expect("fade");
        assert_eq!(data(&output), &[0.0, 0.25, 0.5, 0.75]);
    }

    #[test]
    fn skip_and_zero_durations_copy() {
        let input = mono(vec![0.2, -0.4], 48_000);
        let skipped = fade(&input, 1_000_000, 1_000_000, FadeCurve::Skip).expect("fade");
        let zero = fade(&input, 0, 0, FadeCurve::Linear).expect("fade");
        assert_eq!(data(&skipped), data(&input));
        assert_eq!(data(&zero), data(&input));
    }

    #[test]
    fn a_negative_duration_or_unknown_curve_is_refused() {
        let input = mono(vec![0.2], 48_000);
        assert_eq!(
            fade(&input, -1, 0, FadeCurve::Linear),
            Err(crate::Error::Fade)
        );
        assert_eq!(FadeCurve::from_type(5), Err(crate::Error::Fade));
        assert_eq!(FadeCurve::from_type(-1), Err(crate::Error::Fade));
        assert_eq!(FadeCurve::from_type(0), Ok(FadeCurve::Skip));
    }

    #[test]
    fn empty_input_stays_empty() {
        let input = mono(Vec::new(), 48_000);
        let output = fade(&input, 1_000, 1_000, FadeCurve::Linear).expect("fade");
        assert_eq!(output.samples, 0);
    }

    #[test]
    fn stereo_uses_one_gain_and_copies_the_tail() {
        let input = match Pcm::from_f32(4_000, ChannelLayout::stereo(), vec![1.0, -1.0, 0.5, 0.25])
        {
            Ok(pcm) => pcm,
            Err(err) => panic!("{err:?}"),
        };
        let output = fade(&input, 250, 0, FadeCurve::Linear).expect("fade");
        let samples = data(&output);
        assert_eq!(samples[0], 0.0);
        assert_eq!(samples[1], 0.0);
        assert_eq!(samples[2], 0.5);
        assert_eq!(samples[3], 0.25);
    }
}
