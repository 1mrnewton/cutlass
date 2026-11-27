//! Encoder rate settings shared by NVENC and VA-API, the same as on
//! Windows and Apple.

use cutlass_decode::{CodecParams, ColorRange};
use cutlass_encode::FrameTime;

/// Pictures from one IDR to the next.
pub const KEYFRAME_INTERVAL: u32 = 30;

/// The rate the encoder starts at, before the first picture names one.
pub const DEFAULT_RATE: (u32, u32) = (30, 1);

/// The bitrate asked for, or 8 bits a pixel, at least 64 kbit/s.
pub fn target_bitrate(params: &CodecParams) -> u32 {
    if params.bitrate != 0 {
        return params.bitrate;
    }
    let pixels = u64::from(params.width).saturating_mul(u64::from(params.height));
    u32::try_from(pixels.saturating_mul(8).max(64_000)).unwrap_or(u32::MAX)
}

/// The peak a variable-rate stream may reach: twice the target.
pub fn peak_bitrate(bitrate: u32) -> u32 {
    bitrate.saturating_mul(2)
}

/// Pictures per second as a reduced fraction, from one picture's duration.
pub fn frame_rate(time: FrameTime) -> (u32, u32) {
    let numerator = time.timescale();
    let Ok(denominator) = u32::try_from(time.duration()) else {
        return DEFAULT_RATE;
    };
    if denominator == 0 || numerator == 0 {
        return DEFAULT_RATE;
    }
    let divisor = gcd(numerator, denominator);
    (numerator / divisor, denominator / divisor)
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

/// The colour an encoder writes in the VUI, as CICP values. `None` when
/// the parameters name none, so the encoder's own default stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vui {
    pub primaries: u8,
    pub transfer: u8,
    pub matrix: u8,
    pub full_range: bool,
    /// Whether primaries, transfer or matrix is named, so the colour
    /// description is written.
    pub described: bool,
}

impl Vui {
    pub fn of(params: &CodecParams) -> Option<Self> {
        let primaries = params.primaries.cicp();
        let transfer = params.transfer.cicp();
        let matrix = params.matrix.cicp();
        let described = primaries != 2 || transfer != 2 || matrix != 2;
        let full_range = params.range == ColorRange::Full;
        if !described && params.range == ColorRange::Unspecified {
            return None;
        }
        Some(Self {
            primaries,
            transfer,
            matrix,
            full_range,
            described,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_decode::{CodecId, ColorPrimaries, MatrixCoefficients, Transfer};

    #[test]
    fn the_bitrate_defaults_to_eight_bits_a_pixel() {
        let mut params = CodecParams::video(CodecId::H264, 1920, 1080);
        assert_eq!(target_bitrate(&params), 1920 * 1080 * 8);
        params.width = 16;
        params.height = 16;
        assert_eq!(target_bitrate(&params), 64_000);
        params.bitrate = 5_000_000;
        assert_eq!(target_bitrate(&params), 5_000_000);
        assert_eq!(peak_bitrate(u32::MAX), u32::MAX);
    }

    #[test]
    fn the_rate_is_reduced_from_one_duration() {
        let time = FrameTime::new(0, 1001, 30_000).expect("time");
        assert_eq!(frame_rate(time), (30_000, 1001));
        let time = FrameTime::new(0, 512, 12_800).expect("time");
        assert_eq!(frame_rate(time), (25, 1));
    }

    #[test]
    fn colour_is_written_only_when_named() {
        let mut params = CodecParams::video(CodecId::Hevc, 64, 64);
        assert_eq!(Vui::of(&params), None);
        params.range = ColorRange::Full;
        assert_eq!(
            Vui::of(&params),
            Some(Vui {
                primaries: 2,
                transfer: 2,
                matrix: 2,
                full_range: true,
                described: false,
            })
        );
        params.primaries = ColorPrimaries::Bt709;
        params.transfer = Transfer::Bt709;
        params.matrix = MatrixCoefficients::Bt709;
        params.range = ColorRange::Limited;
        let vui = Vui::of(&params).expect("vui");
        assert_eq!((vui.primaries, vui.transfer, vui.matrix), (1, 1, 1));
        assert!(vui.described && !vui.full_range);
    }
}
