//! Per-clip fade.
//!
//! Durations are microseconds on the same clock as a segment. Curve 0 skips
//! the fade. Curve 1 is a quarter sine, 2 is linear, 3 is logarithmic, and
//! 4 is exponential. Those are the five curves that were traced. A fade
//! longer than the clip covers the clip. Overlapping in and out multiply.
//! Samples outside a fade are copied.
//!
//! A fade belongs to the clip's two edges. [`fade`] takes a buffer that is
//! the whole clip. A mix that works one window at a time passes each window
//! to [`fade_window`] with where that window sits in the clip, so a window
//! in the middle of the clip is left as it is.

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

/// Where a buffer sits in the clip it was cut from.
///
/// Both are microseconds on the clip's own clock, which is the clock the
/// fade durations use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FadeSpan {
    /// From the clip's first sample to the buffer's first sample.
    pub start_us: i64,
    /// The clip from its first sample to its last.
    pub clip_us: i64,
}

/// Apply a fade-in and a fade-out to a buffer that is the whole clip.
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
    if skips(&source, curve) {
        return Ok(input.clone());
    }
    let clip = u64::from(input.samples);
    let fade_in = microseconds_to_frames(fade_in_us, source.rate)?.min(clip);
    let fade_out = microseconds_to_frames(fade_out_us, source.rate)?.min(clip);
    apply(
        input,
        &source,
        Ramps {
            start: 0,
            clip,
            fade_in,
            fade_out,
        },
        curve,
    )
}

/// Apply a clip's fades to one window of it.
///
/// The fade-in covers the clip's first `fade_in_us` and the fade-out its
/// last `fade_out_us`, wherever `input` sits: a window between the two is
/// returned as it came. A window that runs past the clip's end stays at the
/// fade-out's last gain. A negative duration, start, or clip length is
/// refused. Curve [`FadeCurve::Skip`] copies the buffer.
pub fn fade_window(
    input: &Pcm,
    span: FadeSpan,
    fade_in_us: i64,
    fade_out_us: i64,
    curve: FadeCurve,
) -> Result<Pcm, Error> {
    let source = Input::open(input)?;
    if skips(&source, curve) {
        return Ok(input.clone());
    }
    let clip = microseconds_to_frames(span.clip_us, source.rate)?;
    let start = microseconds_to_frames(span.start_us, source.rate)?;
    let fade_in = microseconds_to_frames(fade_in_us, source.rate)?.min(clip);
    let fade_out = microseconds_to_frames(fade_out_us, source.rate)?.min(clip);
    apply(
        input,
        &source,
        Ramps {
            start,
            clip,
            fade_in,
            fade_out,
        },
        curve,
    )
}

/// A buffer's first frame, its clip's length, and the two ramps, all in
/// frames of the clip.
#[derive(Clone, Copy)]
struct Ramps {
    start: u64,
    clip: u64,
    fade_in: u64,
    fade_out: u64,
}

impl Ramps {
    /// Whether no frame of a buffer `frames` long is inside either ramp.
    fn miss(self, frames: u64) -> bool {
        let end = self.start.saturating_add(frames);
        let clear_of_in = self.fade_in == 0 || self.start >= self.fade_in;
        let clear_of_out = self.fade_out == 0 || end <= self.clip - self.fade_out;
        clear_of_in && clear_of_out
    }

    fn gain(self, curve: FadeCurve, frame: u64) -> f32 {
        let at = self.start.saturating_add(frame);
        let mut gain = 1.0_f64;
        if self.fade_in > 0 && at < self.fade_in {
            gain *= curve_gain(curve, at, self.fade_in);
        }
        if self.fade_out > 0 {
            // Past the clip's last frame this stays 0, the end of the ramp.
            let from_end = self.clip.saturating_sub(1).saturating_sub(at);
            if from_end < self.fade_out {
                gain *= curve_gain(curve, from_end, self.fade_out);
            }
        }
        gain as f32
    }
}

/// Whether the buffer is copied before any duration is read.
fn skips(source: &Input<'_>, curve: FadeCurve) -> bool {
    source.frames == 0 || matches!(curve, FadeCurve::Skip)
}

fn apply(input: &Pcm, source: &Input<'_>, ramps: Ramps, curve: FadeCurve) -> Result<Pcm, Error> {
    let frames = input.samples;
    if ramps.miss(u64::from(frames)) {
        return Ok(input.clone());
    }
    let mut out = Built::new(source.format, frames, source.channels)?;
    for frame in 0..frames {
        let gain = ramps.gain(curve, u64::from(frame));
        for channel in 0..source.channels {
            let index = usize::try_from(frame).map_err(|_| Error::TooLong)?;
            if gain == 1.0 {
                out.push_copied(source, index, channel);
            } else {
                out.push_unit(source.sample(index, channel) * gain);
            }
        }
    }
    out.pack(source.rate, source.layout.clone())
}

/// How many frames at `rate` fit in `duration`. A negative duration is
/// refused.
fn microseconds_to_frames(duration: i64, rate: u32) -> Result<u64, Error> {
    if duration < 0 {
        return Err(Error::Fade);
    }
    let count = i128::from(duration) * i128::from(rate) / 1_000_000;
    Ok(u64::try_from(count).unwrap_or(u64::MAX))
}

/// Gain at `index / range`, before silence is scaled to unity.
///
/// `index` runs from 0 at silence toward `range` at unity. The formulas
/// match the traced quarter-sine, linear, log, and exp fades.
fn curve_gain(curve: FadeCurve, index: u64, range: u64) -> f64 {
    if range == 0 {
        return 1.0;
    }
    let gain = (index as f64 / range as f64).clamp(0.0, 1.0);
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

    use super::{FadeCurve, FadeSpan, fade, fade_window};

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

    /// A 1 s clip at 1 kHz with a 20 ms fade in and a 60 ms fade out, cut
    /// into windows of `window` frames and faded one window at a time.
    fn windowed(clip: &[f32], window: usize) -> Vec<f32> {
        let mut out = Vec::new();
        for (index, chunk) in clip.chunks(window).enumerate() {
            let span = FadeSpan {
                start_us: i64::try_from(index * window).expect("start") * 1_000,
                clip_us: i64::try_from(clip.len()).expect("clip") * 1_000,
            };
            let faded = fade_window(
                &mono(chunk.to_vec(), 1_000),
                span,
                20_000,
                60_000,
                FadeCurve::Linear,
            )
            .expect("window");
            out.extend_from_slice(data(&faded));
        }
        out
    }

    #[test]
    fn a_window_in_the_middle_of_the_clip_is_left_alone() {
        let clip: Vec<f32> = (0..1_000_u16)
            .map(|n| 0.5 + f32::from(n) / 4_000.0)
            .collect();
        let whole = fade(
            &mono(clip.clone(), 1_000),
            20_000,
            60_000,
            FadeCurve::Linear,
        )
        .expect("whole");
        // A window shorter than either fade, the case that used to fade
        // every window from end to end.
        let heard = windowed(&clip, 16);
        assert_eq!(heard, data(&whole));
        assert_eq!(heard[..20], data(&whole)[..20]);
        assert_eq!(heard[0], 0.0);
        assert!(heard[10] < clip[10]);
        assert_eq!(heard[20..940], clip[20..940]);
        assert!(heard[941] < clip[941]);
        assert_eq!(heard[999], 0.0);
        // Windows of other lengths cut the same clip.
        assert_eq!(windowed(&clip, 33), data(&whole));
        assert_eq!(windowed(&clip, 1_000), data(&whole));
    }

    #[test]
    fn a_window_clear_of_both_fades_is_the_same_buffer() {
        let samples = match Pcm::from_i16(1_000, ChannelLayout::mono(), vec![7, -9, 11, 6_000]) {
            Ok(pcm) => pcm,
            Err(err) => panic!("{err:?}"),
        };
        let span = FadeSpan {
            start_us: 500_000,
            clip_us: 1_000_000,
        };
        let output = fade_window(&samples, span, 20_000, 60_000, FadeCurve::Sine).expect("fade");
        assert_eq!(output, samples);
        // One frame into the fade out and the buffer is rebuilt.
        let edge = FadeSpan {
            start_us: 937_000,
            clip_us: 1_000_000,
        };
        let output = fade_window(&samples, edge, 20_000, 60_000, FadeCurve::Linear).expect("fade");
        assert_eq!(output.as_i16(), Some(&[7, -9, 11, 5_900][..]));
    }

    #[test]
    fn a_window_past_the_clip_end_stays_at_the_end_of_the_fade_out() {
        let input = mono(vec![1.0; 4], 1_000);
        let span = FadeSpan {
            start_us: 998_000,
            clip_us: 1_000_000,
        };
        let output = fade_window(&input, span, 0, 4_000, FadeCurve::Linear).expect("fade");
        assert_eq!(data(&output), &[0.25, 0.0, 0.0, 0.0]);
        // With no fade out the mix decides what is heard past the end.
        let output = fade_window(&input, span, 4_000, 0, FadeCurve::Linear).expect("fade");
        assert_eq!(data(&output), data(&input));
    }

    #[test]
    fn a_fade_longer_than_the_clip_covers_the_clip_in_every_window() {
        let span = FadeSpan {
            start_us: 2_000,
            clip_us: 4_000,
        };
        let input = mono(vec![1.0; 2], 1_000);
        let output = fade_window(&input, span, 10_000_000, 0, FadeCurve::Linear).expect("fade");
        assert_eq!(data(&output), &[0.5, 0.75]);
    }

    #[test]
    fn a_window_before_the_clip_or_a_negative_clip_is_refused() {
        let input = mono(vec![0.2], 48_000);
        let before = FadeSpan {
            start_us: -1,
            clip_us: 1_000,
        };
        assert_eq!(
            fade_window(&input, before, 10, 10, FadeCurve::Linear),
            Err(crate::Error::Fade)
        );
        let negative = FadeSpan {
            start_us: 0,
            clip_us: -1,
        };
        assert_eq!(
            fade_window(&input, negative, 10, 10, FadeCurve::Linear),
            Err(crate::Error::Fade)
        );
        assert_eq!(
            fade_window(&input, before, -1, 0, FadeCurve::Skip).as_ref(),
            Ok(&input)
        );
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
