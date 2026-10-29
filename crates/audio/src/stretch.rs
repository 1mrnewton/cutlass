//! Time stretch for a playback speed.
//!
//! Speed is the playback rate: 2 plays in half the time. Keeping pitch
//! overlap-adds Hann grains of about 20 ms at a 50 percent hop, and searches
//! a few milliseconds for the alignment. A buffer shorter than four frames
//! has no grain, so each output frame copies the nearest input frame.
//! Letting the pitch follow the speed skips that search and resamples with
//! the speed as the step, which lowpasses when the speed is above 1.

use cutlass_codec::access::Pcm;

use crate::Error;
use crate::buffer::{Built, Input, as_u32, empty_like};
use crate::resample::{Step, render};

/// Stretch `input` by `speed`.
///
/// `keep_pitch` holds the pitch. When it is false, the pitch moves with the
/// speed. Speed 1 is a copy. An empty buffer stays empty. A speed that is
/// not finite and positive is refused.
pub fn stretch(input: &Pcm, speed: f64, keep_pitch: bool) -> Result<Pcm, Error> {
    let source = Input::open(input)?;
    if source.frames == 0 {
        return empty_like(&source, source.rate);
    }
    if !speed.is_finite() || speed <= 0.0 {
        return Err(Error::Speed);
    }
    if speed == 1.0 {
        return Ok(input.clone());
    }
    let frames = stretched_frames(input.samples, speed)?;
    if keep_pitch {
        wsola(&source, frames, speed)
    } else {
        render(&source, frames, source.rate, Step::Speed(speed))
    }
}

fn stretched_frames(frames: u32, speed: f64) -> Result<u32, Error> {
    as_u32((f64::from(frames) / speed).round())
}

fn wsola(input: &Input<'_>, output_frames: u32, speed: f64) -> Result<Pcm, Error> {
    if output_frames == 0 {
        return empty_like(input, input.rate);
    }
    if input.frames < 4 {
        return hold(input, output_frames, speed);
    }
    let window = window_len(input.rate, input.frames);
    let hop = window / 2;
    if hop == 0 {
        return hold(input, output_frames, speed);
    }
    let channels = input.channels;
    let out_len = usize::try_from(output_frames).map_err(|_| Error::TooLong)?;
    let mixed_len = out_len.checked_mul(channels).ok_or(Error::TooLong)?;
    let mut mixed = vec![0.0_f64; mixed_len];
    let mut weight = vec![0.0_f64; out_len];
    let hann: Vec<f64> = (0..window).map(|n| hann(n, window)).collect();
    let mut prev = 0_usize;
    let mut index = 0_u32;
    while let Ok(index_us) = usize::try_from(index) {
        let Some(synth) = index_us.checked_mul(hop) else {
            break;
        };
        if synth >= out_len {
            break;
        }
        let nominal = (f64::from(index) * hop as f64 * speed).round();
        let analysis = if index == 0 {
            0_usize
        } else {
            let natural = prev + hop;
            let delta = search(input, natural, nominal, hop);
            placed(nominal + f64::from(delta), window, input.frames)
        };
        for (tap, coeff) in hann.iter().copied().enumerate() {
            let dst = synth + tap;
            if dst >= out_len {
                break;
            }
            let src = analysis + tap;
            weight[dst] += coeff;
            if src < input.frames {
                for channel in 0..channels {
                    mixed[dst * channels + channel] +=
                        f64::from(input.sample(src, channel)) * coeff;
                }
            }
        }
        prev = analysis;
        let Some(next) = index.checked_add(1) else {
            break;
        };
        index = next;
        if index > output_frames {
            break;
        }
    }
    let mut out = Built::new(input.format, output_frames, channels)?;
    for frame in 0..out_len {
        let scale = weight[frame];
        for channel in 0..channels {
            let value = if scale > 1.0e-18 {
                mixed[frame * channels + channel] / scale
            } else {
                let src = (f64::from(u32::try_from(frame).unwrap_or(0)) * speed).round();
                let src = src.clamp(0.0, (input.frames - 1) as f64);
                f64::from(input.sample(src as usize, channel))
            };
            out.push_unit(value as f32);
        }
    }
    out.pack(input.rate, input.layout.clone())
}

fn hold(input: &Input<'_>, output_frames: u32, speed: f64) -> Result<Pcm, Error> {
    if input.frames == 0 || output_frames == 0 {
        return empty_like(input, input.rate);
    }
    let mut out = Built::new(input.format, output_frames, input.channels)?;
    let last = input.frames - 1;
    for index in 0..output_frames {
        let src = (f64::from(index) * speed).round();
        let src = src.clamp(0.0, last as f64);
        let frame = src as usize;
        for channel in 0..input.channels {
            out.push_copied(input, frame, channel);
        }
    }
    out.pack(input.rate, input.layout.clone())
}

fn window_len(rate: u32, frames: usize) -> usize {
    let twenty_ms = usize::try_from(rate / 50).unwrap_or(0);
    let mut window = twenty_ms.clamp(32, 2_048);
    if window % 2 == 1 {
        window -= 1;
    }
    if window > frames {
        window = frames - (frames % 2);
    }
    window.max(4)
}

fn hann(index: usize, len: usize) -> f64 {
    let n = f64::from(u32::try_from(index).unwrap_or(0));
    let width = f64::from(u32::try_from(len).unwrap_or(1)).max(1.0);
    0.5 * (1.0 - (std::f64::consts::TAU * n / width).cos())
}

fn placed(nominal: f64, window: usize, frames: usize) -> usize {
    let max = frames.saturating_sub(window) as f64;
    let clamped = nominal.clamp(0.0, max);
    clamped as usize
}

/// Offset, in frames, that best matches the previous grain's continuation.
fn search(input: &Input<'_>, natural: usize, nominal: f64, hop: usize) -> i32 {
    let radius = i32::try_from(hop / 2).unwrap_or(1).max(1);
    let mut best = 0_i32;
    let mut best_score = score(input, natural, nominal, hop);
    for delta in -radius..=radius {
        if delta == 0 {
            continue;
        }
        let next = score(input, natural, nominal + f64::from(delta), hop);
        if next > best_score {
            best_score = next;
            best = delta;
        }
    }
    best
}

fn score(input: &Input<'_>, natural: usize, candidate: f64, hop: usize) -> f64 {
    if candidate < 0.0 {
        return f64::NEG_INFINITY;
    }
    let start = candidate as usize;
    if start >= input.frames || natural >= input.frames {
        return f64::NEG_INFINITY;
    }
    let available = hop.min(input.frames - natural).min(input.frames - start);
    let mut total = 0.0;
    for offset in 0..available {
        for channel in 0..input.channels {
            let left = f64::from(input.sample(natural + offset, channel));
            let right = f64::from(input.sample(start + offset, channel));
            total += left * right;
        }
    }
    total
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use cutlass_codec::access::{ChannelLayout, Pcm};

    use super::stretch;
    use crate::Error;

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

    fn crossings(samples: &[f32]) -> f64 {
        let mut count = 0.0_f64;
        for pair in samples.windows(2) {
            if (pair[0] > 0.0) != (pair[1] > 0.0) {
                count += 1.0;
            }
        }
        count / samples.len() as f64
    }

    fn middle(all: &[f32]) -> &[f32] {
        let start = all.len() / 4;
        let end = all.len() * 3 / 4;
        &all[start..end]
    }

    #[test]
    fn unit_speed_copies() {
        let input = mono(vec![0.1, -0.2, 0.3], 48_000);
        let output = stretch(&input, 1.0, true).expect("stretch");
        assert_eq!(data(&output), data(&input));
    }

    #[test]
    fn pitch_follows_speed_and_keeps_even_frames_when_slowing_down() {
        let input = mono(vec![0.0, 0.5, -0.25, 1.0], 8_000);
        let output = stretch(&input, 0.5, false).expect("stretch");
        let samples = data(&output);
        assert_eq!(output.samples, 8);
        assert_eq!(output.rate, 8_000);
        assert_eq!(samples[0], 0.0);
        assert_eq!(samples[2], 0.5);
        assert_eq!(samples[4], -0.25);
        assert_eq!(samples[6], 1.0);
    }

    #[test]
    fn keep_pitch_holds_a_constant_and_the_new_length() {
        let input = mono(vec![0.25; 800], 8_000);
        let output = stretch(&input, 0.5, true).expect("stretch");
        assert_eq!(output.samples, 1_600);
        assert_eq!(output.rate, 8_000);
        for sample in data(&output) {
            assert!((sample - 0.25).abs() < 1.0e-5, "{sample}");
        }
    }

    #[test]
    fn keep_pitch_holds_the_period_of_a_sine() {
        let rate = 8_000_u32;
        let freq = 1_000.0_f64;
        let mut samples = Vec::with_capacity(1_600);
        for n in 0..1_600 {
            let t = n as f64 / f64::from(rate);
            samples.push((2.0 * std::f64::consts::PI * freq * t).sin() as f32);
        }
        let input = mono(samples, rate);
        let original = crossings(middle(data(&input)));
        let held = stretch(&input, 0.5, true).expect("stretch");
        let followed = stretch(&input, 2.0, false).expect("stretch");
        let held_rate = crossings(middle(data(&held)));
        let followed_rate = crossings(middle(data(&followed)));
        assert!(
            (held_rate - original).abs() / original < 0.1,
            "held {held_rate} original {original}"
        );
        assert!(
            (followed_rate - 2.0 * original).abs() / original < 0.15,
            "followed {followed_rate} original {original}"
        );
        assert!(
            (held_rate - original).abs() < (held_rate - followed_rate).abs(),
            "keep-pitch moved toward the sped-up rate"
        );
    }

    #[test]
    fn a_single_sample_repeats_for_the_new_length() {
        let input = mono(vec![0.5], 48_000);
        let output = stretch(&input, 0.5, true).expect("stretch");
        assert_eq!(data(&output), &[0.5, 0.5]);
    }

    #[test]
    fn empty_input_stays_empty() {
        let input = mono(Vec::new(), 48_000);
        let output = stretch(&input, 0.5, true).expect("stretch");
        assert_eq!(output.samples, 0);
        assert_eq!(output.rate, 48_000);
    }

    #[test]
    fn a_bad_speed_is_refused() {
        let input = mono(vec![0.0, 0.1], 48_000);
        assert_eq!(stretch(&input, 0.0, true), Err(Error::Speed));
        assert_eq!(stretch(&input, -1.0, false), Err(Error::Speed));
        assert_eq!(stretch(&input, f64::NAN, true), Err(Error::Speed));
        assert_eq!(stretch(&input, f64::INFINITY, true), Err(Error::Speed));
    }
}
