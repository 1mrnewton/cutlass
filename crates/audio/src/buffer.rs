//! Views over interleaved PCM. Processing uses full-scale `f32`.
//! `i16` is converted at the boundary and copied unchanged when a tap
//! lands on one source frame.

use cutlass_codec::access::{ChannelLayout, Pcm, SampleFormat};

use crate::Error;

/// Full scale for [`SampleFormat::I16`], matching `cutlass_codec`.
pub(crate) const I16_SCALE: f32 = 32_767.0;

/// Maximum channels on a codec [`ChannelLayout`].
const MAX_CHANNELS: usize = 8;

pub(crate) struct Input<'a> {
    pub format: SampleFormat,
    pub rate: u32,
    pub layout: &'a ChannelLayout,
    pub frames: usize,
    pub channels: usize,
    f32_data: &'a [f32],
    i16_data: &'a [i16],
}

impl<'a> Input<'a> {
    pub(crate) fn open(pcm: &'a Pcm) -> Result<Self, Error> {
        if pcm.rate == 0 {
            return Err(Error::Rate);
        }
        let channels = pcm.layout.len();
        if channels == 0 || channels > MAX_CHANNELS {
            return Err(Error::Layout);
        }
        let frames = usize::try_from(pcm.samples).map_err(|_| Error::TooLong)?;
        match pcm.format {
            SampleFormat::F32 => {
                let data = pcm.as_f32().ok_or(Error::Layout)?;
                if data.len() != frames * channels {
                    return Err(Error::Layout);
                }
                if data.iter().any(|sample| !sample.is_finite()) {
                    return Err(Error::NonFinite);
                }
                Ok(Self {
                    format: pcm.format,
                    rate: pcm.rate,
                    layout: &pcm.layout,
                    frames,
                    channels,
                    f32_data: data,
                    i16_data: &[],
                })
            }
            SampleFormat::I16 => {
                let data = pcm.as_i16().ok_or(Error::Layout)?;
                if data.len() != frames * channels {
                    return Err(Error::Layout);
                }
                Ok(Self {
                    format: pcm.format,
                    rate: pcm.rate,
                    layout: &pcm.layout,
                    frames,
                    channels,
                    f32_data: &[],
                    i16_data: data,
                })
            }
        }
    }

    pub(crate) fn sample(&self, frame: usize, channel: usize) -> f32 {
        let Some(index) = frame
            .checked_mul(self.channels)
            .and_then(|base| base.checked_add(channel))
        else {
            return 0.0;
        };
        match self.format {
            SampleFormat::F32 => self.f32_data.get(index).copied().unwrap_or(0.0),
            SampleFormat::I16 => self
                .i16_data
                .get(index)
                .map(|sample| f32::from(*sample) / I16_SCALE)
                .unwrap_or(0.0),
        }
    }

    fn i16_at(&self, frame: usize, channel: usize) -> i16 {
        let Some(index) = frame
            .checked_mul(self.channels)
            .and_then(|base| base.checked_add(channel))
        else {
            return 0;
        };
        self.i16_data.get(index).copied().unwrap_or(0)
    }
}

pub(crate) fn empty_like(input: &Input<'_>, rate: u32) -> Result<Pcm, Error> {
    pack(
        input.format,
        rate,
        input.layout.clone(),
        Vec::new(),
        Vec::new(),
    )
}

pub(crate) fn quantize_i16(value: f32) -> i16 {
    let scaled = f64::from(value) * f64::from(I16_SCALE);
    if !scaled.is_finite() {
        return if value.is_sign_negative() {
            i16::MIN
        } else {
            i16::MAX
        };
    }
    let rounded = scaled.round();
    if rounded >= f64::from(i16::MAX) {
        i16::MAX
    } else if rounded <= f64::from(i16::MIN) {
        i16::MIN
    } else {
        // `rounded` is finite and inside the i16 range.
        #[allow(clippy::cast_possible_truncation)]
        {
            rounded as i16
        }
    }
}

pub(crate) enum Built {
    F32(Vec<f32>),
    I16(Vec<i16>),
}

impl Built {
    pub(crate) fn new(format: SampleFormat, frames: u32, channels: usize) -> Result<Self, Error> {
        let len = usize::try_from(frames).map_err(|_| Error::TooLong)?;
        let len = len.checked_mul(channels).ok_or(Error::TooLong)?;
        Ok(match format {
            SampleFormat::F32 => Self::F32(Vec::with_capacity(len)),
            SampleFormat::I16 => Self::I16(Vec::with_capacity(len)),
        })
    }

    pub(crate) fn push_copied(&mut self, input: &Input<'_>, frame: usize, channel: usize) {
        match self {
            Self::F32(samples) => samples.push(input.sample(frame, channel)),
            Self::I16(samples) => samples.push(input.i16_at(frame, channel)),
        }
    }

    pub(crate) fn push_unit(&mut self, value: f32) {
        match self {
            Self::F32(samples) => samples.push(value),
            Self::I16(samples) => samples.push(quantize_i16(value)),
        }
    }

    pub(crate) fn pack(self, rate: u32, layout: ChannelLayout) -> Result<Pcm, Error> {
        match self {
            Self::F32(samples) => pack(SampleFormat::F32, rate, layout, samples, Vec::new()),
            Self::I16(samples) => pack(SampleFormat::I16, rate, layout, Vec::new(), samples),
        }
    }
}

fn pack(
    format: SampleFormat,
    rate: u32,
    layout: ChannelLayout,
    f32_samples: Vec<f32>,
    i16_samples: Vec<i16>,
) -> Result<Pcm, Error> {
    match format {
        SampleFormat::F32 => Pcm::from_f32(rate, layout, f32_samples).map_err(|_| Error::Layout),
        SampleFormat::I16 => Pcm::from_i16(rate, layout, i16_samples).map_err(|_| Error::Layout),
    }
}

/// Whole frames after rounding `frames * to_rate / from_rate` to nearest.
pub(crate) fn scaled_rate(frames: u32, from_rate: u32, to_rate: u32) -> Result<u32, Error> {
    let den = u128::from(from_rate);
    if den == 0 {
        return Err(Error::Rate);
    }
    let num = u128::from(frames) * u128::from(to_rate);
    let out = (num + den / 2) / den;
    u32::try_from(out).map_err(|_| Error::TooLong)
}

pub(crate) fn as_i64(value: f64) -> Option<i64> {
    // `i64::MAX` is not an exact f64, so the upper bound is the next power of two.
    if !value.is_finite() || value < i64::MIN as f64 || value >= i64::MAX as f64 {
        return None;
    }
    #[allow(clippy::cast_possible_truncation)]
    Some(value as i64)
}

pub(crate) fn as_u32(value: f64) -> Result<u32, Error> {
    if !value.is_finite() || value < 0.0 || value > f64::from(u32::MAX) {
        return Err(Error::TooLong);
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok(value as u32)
}
