//! Between FLAC integer samples and the shared [`Pcm`] buffer.
//!
//! The mix stores [`SampleFormat::I16`] or [`SampleFormat::F32`]. Full scale
//! is the positive maximum, as for `i16` (32767) and the PCM crate's s24
//! (8388607): a `bits`-deep sample `s` is `s / (2^(bits-1) - 1)` in `F32`.
//! For depths through 24 that division is undone exactly by
//! [`f32_to_sample`]; `f32_round_trips_every_24_bit_sample` checks all
//! 2^24 values. Deeper samples do not fit an `f32` mantissa and are rounded.
//! `I16` carries depths through 16 exactly, shifted left to 16 bits.

use cutlass_decode::{Channel, ChannelLayout, CodecError, Pcm, SampleFormat};

/// The channel order of RFC 9639 section 9.1.3 for `channels` coded channels.
pub fn layout(channels: u8) -> Result<ChannelLayout, CodecError> {
    use Channel::{
        BackCenter, BackLeft, BackRight, FrontCenter, FrontLeft, FrontRight, Lfe, SideLeft,
        SideRight,
    };
    let order = match channels {
        1 => vec![FrontCenter],
        2 => vec![FrontLeft, FrontRight],
        3 => vec![FrontLeft, FrontRight, FrontCenter],
        4 => vec![FrontLeft, FrontRight, BackLeft, BackRight],
        5 => vec![FrontLeft, FrontRight, FrontCenter, BackLeft, BackRight],
        6 => vec![FrontLeft, FrontRight, FrontCenter, Lfe, BackLeft, BackRight],
        7 => vec![
            FrontLeft,
            FrontRight,
            FrontCenter,
            Lfe,
            BackCenter,
            SideLeft,
            SideRight,
        ],
        8 => vec![
            FrontLeft,
            FrontRight,
            FrontCenter,
            Lfe,
            BackLeft,
            BackRight,
            SideLeft,
            SideRight,
        ],
        _ => {
            return Err(CodecError::Unsupported {
                what: "flac channel count",
            });
        }
    };
    ChannelLayout::new(order).map_err(|_| CodecError::Unsupported {
        what: "flac channel count",
    })
}

/// Positive full scale of a `bits`-deep sample.
pub(crate) fn full_scale(bits: u8) -> f64 {
    f64::from((1u32 << (bits - 1)) - 1)
}

/// `sample` at `bits` depth as an `F32` mix sample.
pub(crate) fn sample_to_f32(sample: i32, bits: u8) -> f32 {
    (f64::from(sample) / full_scale(bits)) as f32
}

/// The `bits`-deep integer an `F32` mix sample stands for.
///
/// A non-finite value is [`CodecError::NonFinite`]. A value outside the
/// depth's range after rounding is [`CodecError::OutOfRange`]: it is not
/// clipped.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the encoder reads mix samples with it")
)]
pub(crate) fn f32_to_sample(value: f32, bits: u8) -> Result<i32, CodecError> {
    if !value.is_finite() {
        return Err(CodecError::NonFinite);
    }
    let scaled = (f64::from(value) * full_scale(bits)).round();
    let high = full_scale(bits);
    let low = -high - 1.0;
    if scaled < low || scaled > high {
        return Err(CodecError::OutOfRange);
    }
    Ok(scaled as i32)
}

/// Builds the output buffer in `format`.
pub(crate) fn to_pcm(
    samples: &[i32],
    bits: u8,
    rate: u32,
    layout: ChannelLayout,
    format: SampleFormat,
) -> Result<Pcm, CodecError> {
    let pcm = match format {
        SampleFormat::I16 => {
            if bits > 16 {
                return Err(CodecError::Unsupported {
                    what: "16-bit output of a deeper flac stream",
                });
            }
            let shift = 16 - u32::from(bits);
            let out = samples
                .iter()
                .map(|sample| i16::try_from(sample << shift).unwrap_or(0))
                .collect();
            Pcm::from_i16(rate, layout, out)
        }
        SampleFormat::F32 => {
            let out = samples
                .iter()
                .map(|sample| sample_to_f32(*sample, bits))
                .collect();
            Pcm::from_f32(rate, layout, out)
        }
    };
    pcm.map_err(|_| CodecError::Limit { what: "pcm buffer" })
}

/// Reads mix samples back into `bits`-deep integers, appending to `out`.
#[expect(dead_code, reason = "the encoder reads mix samples with it")]
pub(crate) fn from_pcm(pcm: &Pcm, bits: u8, out: &mut Vec<i32>) -> Result<(), CodecError> {
    match pcm.format {
        SampleFormat::I16 => {
            let data = pcm.as_i16().ok_or(CodecError::InvalidApi { what: "pcm" })?;
            if bits != 16 {
                return Err(CodecError::Unsupported {
                    what: "16-bit input to a flac stream of another depth",
                });
            }
            out.extend(data.iter().map(|sample| i32::from(*sample)));
        }
        SampleFormat::F32 => {
            let data = pcm.as_f32().ok_or(CodecError::InvalidApi { what: "pcm" })?;
            let start = out.len();
            out.reserve(data.len());
            for value in data {
                match f32_to_sample(*value, bits) {
                    Ok(sample) => out.push(sample),
                    Err(error) => {
                        out.truncate(start);
                        return Err(error);
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{f32_to_sample, layout, sample_to_f32};
    use cutlass_decode::{Channel, CodecError};

    #[test]
    fn f32_round_trips_every_24_bit_sample() {
        for sample in -(1i32 << 23)..(1i32 << 23) {
            let back = f32_to_sample(sample_to_f32(sample, 24), 24).expect("in range");
            assert_eq!(back, sample);
        }
        for bits in [4u8, 8, 12, 16, 20] {
            let half = 1i32 << (bits - 1);
            for sample in -half..half {
                assert_eq!(f32_to_sample(sample_to_f32(sample, bits), bits), Ok(sample));
            }
        }
    }

    #[test]
    fn full_scale_is_the_positive_maximum_and_out_of_range_is_refused() {
        assert_eq!(sample_to_f32(32_767, 16), 1.0);
        assert_eq!(f32_to_sample(1.0, 16), Ok(32_767));
        assert_eq!(f32_to_sample(-1.0, 16), Ok(-32_767));
        assert_eq!(f32_to_sample(1.01, 16), Err(CodecError::OutOfRange));
        assert_eq!(f32_to_sample(f32::NAN, 16), Err(CodecError::NonFinite));
    }

    #[test]
    fn channel_orders_follow_the_rfc_table() {
        assert_eq!(layout(1).expect("mono").channels(), &[Channel::FrontCenter]);
        let six = layout(6).expect("5.1");
        assert_eq!(six.channels()[3], Channel::Lfe);
        assert_eq!(layout(8).expect("7.1").len(), 8);
        assert!(layout(0).is_err() && layout(9).is_err());
    }
}
