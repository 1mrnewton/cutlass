//! Fold a surround clip into the stereo mix.
//!
//! The matrix is the ITU-R BS.775 / ATSC A/52 downmix that FFmpeg's
//! `swresample` builds by default (`build_matrix` in `rematrix.c`, with
//! `clev` and `slev` at -3 dB and `lfe_mix_level` at zero):
//!
//! ```text
//! L = FL + 0.7071 FC + FLC + 0.7071 (BL + SL) + 0.5 BC
//! R = FR + 0.7071 FC + FRC + 0.7071 (BR + SR) + 0.5 BC
//! ```
//!
//! The LFE channel is dropped. Back and side surrounds both fold into the
//! front pair at -3 dB, and a back center is split between them at -3 dB
//! each (-6 dB per side).
//!
//! The matrix is not normalised. `swresample` divides it by its largest row
//! sum only when the output samples are integers; for float output, as
//! here, it leaves the gains alone. A 5.1 frame at full scale on every
//! channel sums to `1 + 0.7071 + 0.7071 = 2.414` per side. The clip keeps
//! its level beside a stereo clip, and peaks are left to the mix: the
//! limiter runs on the sum, and export clamps when it quantizes.

use cutlass_codec::access::{Channel, ChannelLayout, Pcm, SampleFormat};

use crate::Error;
use crate::buffer::I16_SCALE;

/// Maximum channels on a codec [`ChannelLayout`].
const MAX_CHANNELS: usize = 8;

/// -3 dB: the center and surround gains.
const MINUS_3_DB: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Left and right gains for each input channel.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Matrix {
    channels: usize,
    left: [f32; MAX_CHANNELS],
    right: [f32; MAX_CHANNELS],
}

impl Matrix {
    fn new(from: &ChannelLayout) -> Result<Self, Error> {
        let channels = from.len();
        if channels <= 1 || channels > MAX_CHANNELS {
            return Err(Error::Layout);
        }
        let mut left = [0.0; MAX_CHANNELS];
        let mut right = [0.0; MAX_CHANNELS];
        for (index, channel) in from.channels().iter().enumerate() {
            let (to_left, to_right) = gains(*channel);
            left[index] = to_left;
            right[index] = to_right;
        }
        Ok(Self {
            channels,
            left,
            right,
        })
    }
}

/// What one speaker adds to the left and the right output.
const fn gains(channel: Channel) -> (f32, f32) {
    match channel {
        Channel::FrontLeft | Channel::FrontLeftOfCenter => (1.0, 0.0),
        Channel::FrontRight | Channel::FrontRightOfCenter => (0.0, 1.0),
        Channel::FrontCenter => (MINUS_3_DB, MINUS_3_DB),
        Channel::BackLeft | Channel::SideLeft => (MINUS_3_DB, 0.0),
        Channel::BackRight | Channel::SideRight => (0.0, MINUS_3_DB),
        Channel::BackCenter => (0.5, 0.5),
        Channel::Lfe => (0.0, 0.0),
    }
}

/// `input` folded into `target`, as `f32`.
///
/// The same layout is returned unchanged, bit for bit and in its own sample
/// format. The target must be stereo. A mono input is refused: the mix
/// already plays a mono clip at full level on both sides, which is not a
/// downmix. Any other layout, including a two-channel file whose speakers
/// are not left and right, goes through the matrix in one pass.
pub fn downmix(input: &Pcm, target: &ChannelLayout) -> Result<Pcm, Error> {
    if input.layout == *target {
        return Ok(input.clone());
    }
    let mut out = Vec::new();
    downmix_into(input, target, &mut out)?;
    Pcm::from_f32(input.rate, target.clone(), out).map_err(|_| Error::Layout)
}

/// [`downmix`] into a buffer the caller keeps.
///
/// `out` is cleared and holds interleaved stereo `f32` afterwards. A buffer
/// with room for the frames is not reallocated, so a caller that keeps one
/// across clips allocates nothing per call. The same layout is copied
/// across as `f32`. A non-finite sample, or a sum that overflows, is
/// [`Error::NonFinite`] and leaves `out` empty.
pub fn downmix_into(input: &Pcm, target: &ChannelLayout, out: &mut Vec<f32>) -> Result<(), Error> {
    out.clear();
    if input.rate == 0 {
        return Err(Error::Rate);
    }
    if *target != ChannelLayout::stereo() {
        return Err(Error::Layout);
    }
    let frames = usize::try_from(input.samples).map_err(|_| Error::TooLong)?;
    let wanted = frames.checked_mul(2).ok_or(Error::TooLong)?;
    let outcome = if input.layout == *target {
        copy(input, wanted, out)
    } else {
        let matrix = Matrix::new(&input.layout)?;
        out.reserve(wanted);
        match input.format {
            SampleFormat::F32 => {
                let samples = input.as_f32().ok_or(Error::Layout)?;
                fold(&matrix, samples, frames, out, |sample| sample)
            }
            SampleFormat::I16 => {
                let samples = input.as_i16().ok_or(Error::Layout)?;
                fold(&matrix, samples, frames, out, |sample| {
                    f32::from(sample) / I16_SCALE
                })
            }
        }
    };
    if outcome.is_err() {
        out.clear();
    }
    outcome
}

/// One pass: each input frame is read once and writes one output pair.
fn fold<T: Copy>(
    matrix: &Matrix,
    samples: &[T],
    frames: usize,
    out: &mut Vec<f32>,
    unit: impl Fn(T) -> f32,
) -> Result<(), Error> {
    let channels = matrix.channels;
    if samples.len() != frames.checked_mul(channels).ok_or(Error::TooLong)? {
        return Err(Error::Layout);
    }
    let left = &matrix.left[..channels];
    let right = &matrix.right[..channels];
    for frame in samples.chunks_exact(channels) {
        let mut to_left = 0.0_f32;
        let mut to_right = 0.0_f32;
        for ((sample, gain_left), gain_right) in frame.iter().zip(left).zip(right) {
            let value = unit(*sample);
            to_left += value * gain_left;
            to_right += value * gain_right;
        }
        // A NaN or infinity on any input, even one the matrix drops, makes
        // the sum non-finite.
        if !(to_left.is_finite() && to_right.is_finite()) {
            return Err(Error::NonFinite);
        }
        out.push(to_left);
        out.push(to_right);
    }
    Ok(())
}

fn copy(input: &Pcm, wanted: usize, out: &mut Vec<f32>) -> Result<(), Error> {
    out.reserve(wanted);
    match input.format {
        SampleFormat::F32 => {
            let samples = input.as_f32().ok_or(Error::Layout)?;
            if samples.iter().any(|sample| !sample.is_finite()) {
                return Err(Error::NonFinite);
            }
            out.extend_from_slice(samples);
        }
        SampleFormat::I16 => {
            let samples = input.as_i16().ok_or(Error::Layout)?;
            out.extend(samples.iter().map(|sample| f32::from(*sample) / I16_SCALE));
        }
    }
    if out.len() != wanted {
        return Err(Error::Layout);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
