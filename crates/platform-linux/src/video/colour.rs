//! Picture tags from a stream's colour description.
//!
//! The hardware hands back the CICP code points of the stream (VUI in H.264
//! and HEVC, the colour config in AV1). They are tagged the way the Rust
//! decoders tag the same stream, so a clip looks the same whichever decoder
//! opened it. A code point the stream does not give stays unspecified.

use cutlass_decode::{ColorMatrix, ColorPrimaries, ColorRange, PictureInfo, PixelFormat, Transfer};

/// CICP code points and the range flag of one stream.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Signal {
    pub primaries: u8,
    pub transfer: u8,
    pub matrix: u8,
    pub full_range: Option<bool>,
}

impl Signal {
    /// The codes a stream with no colour description implies.
    #[cfg(test)]
    pub const UNSPECIFIED: Self = Self {
        primaries: 2,
        transfer: 2,
        matrix: 2,
        full_range: None,
    };
}

/// An 8-bit 4:2:0 picture of `width` by `height` tagged with `signal`.
pub fn picture_info(width: u32, height: u32, signal: Signal) -> PictureInfo {
    let mut info = PictureInfo::new(
        width,
        height,
        PixelFormat::Yuv420p,
        primaries_of(signal.primaries),
        transfer_of(signal.transfer),
    );
    info.matrix = matrix_of(signal.matrix);
    info.cicp_primaries = signal.primaries;
    info.cicp_transfer = signal.transfer;
    info.cicp_matrix = signal.matrix;
    info.range = match signal.full_range {
        Some(true) => ColorRange::Full,
        Some(false) => ColorRange::Limited,
        None => ColorRange::Unspecified,
    };
    info
}

fn primaries_of(code: u8) -> ColorPrimaries {
    match code {
        1 => ColorPrimaries::Bt709,
        4 => ColorPrimaries::Bt470m,
        5 | 6 => ColorPrimaries::Bt601,
        7 => ColorPrimaries::Smpte240m,
        8 => ColorPrimaries::Film,
        9 => ColorPrimaries::Bt2020,
        10 => ColorPrimaries::Smpte428,
        11 => ColorPrimaries::Smpte431,
        12 => ColorPrimaries::Smpte432,
        22 => ColorPrimaries::Ebu3213,
        _ => ColorPrimaries::Unspecified,
    }
}

fn transfer_of(code: u8) -> Transfer {
    match code {
        1 | 6 => Transfer::Bt709,
        4 | 5 => Transfer::Bt601,
        7 => Transfer::Smpte240m,
        8 => Transfer::Linear,
        9 => Transfer::Log100,
        10 => Transfer::LogSqrt,
        11 => Transfer::Iec61966_2_4,
        12 => Transfer::Bt1361,
        13 => Transfer::Srgb,
        14 | 15 => Transfer::Bt2020,
        16 => Transfer::Pq,
        17 => Transfer::Smpte428,
        18 => Transfer::Hlg,
        _ => Transfer::Unspecified,
    }
}

fn matrix_of(code: u8) -> ColorMatrix {
    match code {
        0 => ColorMatrix::Identity,
        1 => ColorMatrix::Bt709,
        4 => ColorMatrix::Fcc,
        5 | 6 => ColorMatrix::Bt601,
        7 => ColorMatrix::Smpte240m,
        8 => ColorMatrix::Ycgco,
        9 | 10 => ColorMatrix::Bt2020,
        11 => ColorMatrix::Smpte2085,
        12 => ColorMatrix::ChromaDerived,
        13 => ColorMatrix::ChromaDerivedConstant,
        14 => ColorMatrix::Ictcp,
        15 => ColorMatrix::IptC2,
        16 => ColorMatrix::YcgcoRe,
        17 => ColorMatrix::YcgcoRo,
        _ => ColorMatrix::Unspecified,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bt2020_pq_full_range_is_tagged_and_unspecified_stays_so() {
        let info = picture_info(
            64,
            48,
            Signal {
                primaries: 9,
                transfer: 16,
                matrix: 9,
                full_range: Some(true),
            },
        );
        assert_eq!(info.primaries, ColorPrimaries::Bt2020);
        assert_eq!(info.transfer, Transfer::Pq);
        assert_eq!(info.matrix, ColorMatrix::Bt2020);
        assert_eq!(info.range, ColorRange::Full);
        assert_eq!((info.cicp_primaries, info.cicp_transfer), (9, 16));

        let plain = picture_info(64, 48, Signal::UNSPECIFIED);
        assert_eq!(plain.primaries, ColorPrimaries::Unspecified);
        assert_eq!(plain.range, ColorRange::Unspecified);
        assert_eq!(plain.pixel, PixelFormat::Yuv420p);
    }
}
