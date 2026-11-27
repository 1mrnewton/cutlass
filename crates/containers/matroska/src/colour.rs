//! A video track's Colour element to the colour a stream carries, and back.
//!
//! Source: RFC 9559, 5.1.4.1.28.16 to 5.1.4.1.28.40. Matroska's
//! MatrixCoefficients, TransferCharacteristics and Primaries tables are the
//! ISO/IEC 23091-2 tables, so a value is the code [`StreamColor`] holds: a
//! code that table defines is kept, and 2 (unspecified), a reserved code or
//! a value past the table is unknown. Range 1 is limited and 2 full; 0
//! (unspecified) and 3 (defined by the matrix and transfer) are unknown, as
//! FFmpeg's `matroskadec.c` reads them.
//!
//! The light level is CTA-861.3's MaxCLL and MaxFALL, a 0 being unknown.
//! The mastering display is kept only when MasteringMetadata states all ten
//! values, chromaticities from 0 to 1 and luminances from 0 up, the minimum
//! below the maximum; it is converted to the fixed-point units of
//! [`MasteringDisplay`] (chromaticities in 1/65536, the maximum luminance in
//! 1/256 and the minimum in 1/16384 cd/m²), so a value read back after a
//! write is within half a unit of those.

use cutlass_codec::StreamColor;
use cutlass_codec::access::{ColorRange, ContentLight, MasteringDisplay};

use crate::ebml::{put_float, put_master, put_uint};
use crate::ids;
use crate::meta::{Colour, Mastering};

/// Chromaticity units per 1 in [`MasteringDisplay`].
const CHROMATICITY: f64 = 65_536.0;
/// Maximum luminance units per cd/m².
const LUMINANCE_MAX: f64 = 256.0;
/// Minimum luminance units per cd/m².
const LUMINANCE_MIN: f64 = 16_384.0;

/// The colour a stream carries for a track's Colour element.
#[must_use]
pub fn stream_color(colour: &Colour) -> StreamColor {
    StreamColor {
        primaries: code(colour.primaries, |code| matches!(code, 1 | 4..=12 | 22)),
        transfer: code(colour.transfer_characteristics, |code| {
            matches!(code, 1 | 4..=18)
        }),
        matrix: code(colour.matrix_coefficients, |code| {
            matches!(code, 0 | 1 | 4..=17)
        }),
        range: match colour.range {
            Some(1) => ColorRange::Limited,
            Some(2) => ColorRange::Full,
            _ => ColorRange::Unspecified,
        },
        content_light: content_light(colour.max_cll, colour.max_fall),
        mastering_display: colour.mastering.as_ref().and_then(mastering_display),
    }
}

fn code(value: Option<u64>, defined: impl Fn(u8) -> bool) -> Option<u8> {
    value
        .and_then(|value| u8::try_from(value).ok())
        .filter(|code| defined(*code))
}

fn content_light(max_cll: Option<u64>, max_fall: Option<u64>) -> Option<ContentLight> {
    let level = |value: Option<u64>| u16::try_from(value.unwrap_or(0)).unwrap_or(u16::MAX);
    let light = ContentLight {
        max_cll: level(max_cll),
        max_fall: level(max_fall),
    };
    (light.max_cll > 0 || light.max_fall > 0).then_some(light)
}

fn mastering_display(mastering: &Mastering) -> Option<MasteringDisplay> {
    let chromaticity = |value: Option<f64>| {
        let value = value.filter(|value| (0.0..=1.0).contains(value))?;
        // At most 65,536 after rounding, which one unit below caps.
        Some((value * CHROMATICITY).round().min(f64::from(u16::MAX)) as u16)
    };
    let pair = |pair: [Option<f64>; 2]| Some([chromaticity(pair[0])?, chromaticity(pair[1])?]);
    let max = mastering.luminance_max.filter(|max| *max > 0.0)?;
    let min = mastering
        .luminance_min
        .filter(|min| (0.0..max).contains(min))?;
    let luminance = |value: f64, scale: f64| {
        let units = (value * scale).round();
        // Non-negative and capped, so the cast is exact.
        units.min(f64::from(u32::MAX)) as u32
    };
    Some(MasteringDisplay {
        primaries: [
            pair(mastering.primaries[0])?,
            pair(mastering.primaries[1])?,
            pair(mastering.primaries[2])?,
        ],
        white_point: pair(mastering.white_point)?,
        max_luminance: luminance(max, LUMINANCE_MAX),
        min_luminance: luminance(min, LUMINANCE_MIN),
    })
}

/// Appends a Colour element for `color` to a Video element's body, or
/// nothing when `color` states nothing. Children go in the order RFC 9559
/// lists them; one the stream does not state is left out, so the file's
/// default (unspecified) stands for it.
pub fn put_colour(video: &mut Vec<u8>, color: &StreamColor) {
    let mut body = Vec::new();
    if let Some(matrix) = color.matrix {
        put_uint(&mut body, ids::MATRIX_COEFFICIENTS, u64::from(matrix));
    }
    match color.range {
        ColorRange::Limited | ColorRange::Studio => put_uint(&mut body, ids::RANGE, 1),
        ColorRange::Full => put_uint(&mut body, ids::RANGE, 2),
        ColorRange::Unspecified => {}
    }
    if let Some(transfer) = color.transfer {
        put_uint(
            &mut body,
            ids::TRANSFER_CHARACTERISTICS,
            u64::from(transfer),
        );
    }
    if let Some(primaries) = color.primaries {
        put_uint(&mut body, ids::PRIMARIES, u64::from(primaries));
    }
    if let Some(light) = color.content_light {
        if light.max_cll > 0 {
            put_uint(&mut body, ids::MAX_CLL, u64::from(light.max_cll));
        }
        if light.max_fall > 0 {
            put_uint(&mut body, ids::MAX_FALL, u64::from(light.max_fall));
        }
    }
    if let Some(display) = color.mastering_display {
        let mut mastering = Vec::new();
        let ids = [
            [ids::PRIMARY_R_CHROMATICITY_X, ids::PRIMARY_R_CHROMATICITY_Y],
            [ids::PRIMARY_G_CHROMATICITY_X, ids::PRIMARY_G_CHROMATICITY_Y],
            [ids::PRIMARY_B_CHROMATICITY_X, ids::PRIMARY_B_CHROMATICITY_Y],
        ];
        for (pair, ids) in display.primaries.iter().zip(ids) {
            for (value, id) in pair.iter().zip(ids) {
                put_float(&mut mastering, id, f64::from(*value) / CHROMATICITY);
            }
        }
        let [x, y] = display.white_point;
        put_float(
            &mut mastering,
            ids::WHITE_POINT_CHROMATICITY_X,
            f64::from(x) / CHROMATICITY,
        );
        put_float(
            &mut mastering,
            ids::WHITE_POINT_CHROMATICITY_Y,
            f64::from(y) / CHROMATICITY,
        );
        put_float(
            &mut mastering,
            ids::LUMINANCE_MAX,
            f64::from(display.max_luminance) / LUMINANCE_MAX,
        );
        put_float(
            &mut mastering,
            ids::LUMINANCE_MIN,
            f64::from(display.min_luminance) / LUMINANCE_MIN,
        );
        put_master(&mut body, ids::MASTERING_METADATA, &mastering);
    }
    if !body.is_empty() {
        put_master(video, ids::COLOUR, &body);
    }
}

#[cfg(test)]
mod tests {
    use super::{put_colour, stream_color};
    use crate::codec::map_track;
    use crate::ebml::{Children, put_bytes, put_float, put_master, put_uint};
    use crate::ids;
    use crate::meta::{Colour, Mastering, parse_tracks};
    use cutlass_codec::StreamColor;
    use cutlass_codec::access::{ColorRange, ContentLight, MasteringDisplay};

    /// A TrackEntry for a `codec` video track whose Video element holds
    /// `colour` as the Colour element's body.
    fn video_track(codec: &[u8], colour: &[u8]) -> Vec<u8> {
        let mut video = Vec::new();
        put_uint(&mut video, ids::PIXEL_WIDTH, 64);
        put_uint(&mut video, ids::PIXEL_HEIGHT, 32);
        put_master(&mut video, ids::COLOUR, colour);
        let mut entry = Vec::new();
        put_uint(&mut entry, ids::TRACK_NUMBER, 1);
        put_uint(&mut entry, ids::TRACK_TYPE, 1);
        put_bytes(&mut entry, ids::CODEC_ID, codec);
        put_master(&mut entry, ids::VIDEO, &video);
        let mut out = Vec::new();
        put_master(&mut out, ids::TRACK_ENTRY, &entry);
        out
    }

    /// The colour the stream of a hand-built `codec` track carries.
    fn colour_of(codec: &[u8], colour: &[u8]) -> (Option<Colour>, StreamColor) {
        let tracks = parse_tracks(&video_track(codec, colour), 64, 1024).expect("tracks");
        let track = &tracks[0];
        let mapped = map_track(track).expect("mapped");
        (track.colour, mapped.params.color())
    }

    /// An HDR10 Colour: BT.2020 non-constant luminance, PQ, BT.2020
    /// primaries, broadcast range, a light level and a mastering display
    /// (P3 primaries, D65, 1000 and 0.005 cd/m², as HDR10 masters state).
    fn hdr10() -> Vec<u8> {
        let mut mastering = Vec::new();
        for (id, value) in [
            (ids::PRIMARY_R_CHROMATICITY_X, 0.68),
            (ids::PRIMARY_R_CHROMATICITY_Y, 0.32),
            (ids::PRIMARY_G_CHROMATICITY_X, 0.265),
            (ids::PRIMARY_G_CHROMATICITY_Y, 0.69),
            (ids::PRIMARY_B_CHROMATICITY_X, 0.15),
            (ids::PRIMARY_B_CHROMATICITY_Y, 0.06),
            (ids::WHITE_POINT_CHROMATICITY_X, 0.3127),
            (ids::WHITE_POINT_CHROMATICITY_Y, 0.329),
            (ids::LUMINANCE_MAX, 1000.0),
            (ids::LUMINANCE_MIN, 0.005),
        ] {
            put_float(&mut mastering, id, value);
        }
        let mut colour = Vec::new();
        put_uint(&mut colour, ids::MATRIX_COEFFICIENTS, 9);
        put_uint(&mut colour, ids::RANGE, 1);
        put_uint(&mut colour, ids::TRANSFER_CHARACTERISTICS, 16);
        put_uint(&mut colour, ids::PRIMARIES, 9);
        put_uint(&mut colour, ids::MAX_CLL, 1000);
        put_uint(&mut colour, ids::MAX_FALL, 400);
        put_master(&mut colour, ids::MASTERING_METADATA, &mastering);
        colour
    }

    const HDR10_DISPLAY: MasteringDisplay = MasteringDisplay {
        // 0.68 × 65536 = 44564.48, 0.32 × 65536 = 20971.52, and so on.
        primaries: [[44564, 20972], [17367, 45220], [9830, 3932]],
        white_point: [20493, 21561],
        max_luminance: 256_000,
        min_luminance: 82,
    };

    #[test]
    fn every_colour_field_is_read_from_the_element() {
        let (stored, color) = colour_of(b"V_VP9", &hdr10());
        let stored = stored.expect("colour");
        assert_eq!(stored.matrix_coefficients, Some(9));
        assert_eq!(stored.range, Some(1));
        assert_eq!(stored.transfer_characteristics, Some(16));
        assert_eq!(stored.primaries, Some(9));
        assert_eq!(stored.max_cll, Some(1000));
        assert_eq!(stored.max_fall, Some(400));
        let mastering = stored.mastering.expect("mastering");
        assert_eq!(mastering.primaries[1], [Some(0.265), Some(0.69)]);
        assert_eq!(mastering.white_point, [Some(0.3127), Some(0.329)]);
        assert_eq!(mastering.luminance_max, Some(1000.0));
        assert_eq!(mastering.luminance_min, Some(0.005));
        assert_eq!(
            color,
            StreamColor {
                primaries: Some(9),
                transfer: Some(16),
                matrix: Some(9),
                range: ColorRange::Limited,
                content_light: Some(ContentLight {
                    max_cll: 1000,
                    max_fall: 400,
                }),
                mastering_display: Some(HDR10_DISPLAY),
            }
        );
    }

    /// The colour is the container's, whatever the codec: every video
    /// CodecID the mapping knows carries it, and so does an unknown one.
    #[test]
    fn every_video_codec_carries_the_colour() {
        for codec in [
            &b"V_VP9"[..],
            b"V_VP8",
            b"V_AV1",
            b"V_MPEG4/ISO/AVC",
            b"V_MPEGH/ISO/HEVC",
            b"V_PRORES",
            b"V_MPEG2",
            b"V_QUICKTIME",
        ] {
            let (_, color) = colour_of(codec, &hdr10());
            assert_eq!(color.transfer, Some(16), "{codec:?}");
            assert_eq!(color.mastering_display, Some(HDR10_DISPLAY), "{codec:?}");
        }
    }

    #[test]
    fn unspecified_and_reserved_values_are_unknown() {
        let mut colour = Vec::new();
        put_uint(&mut colour, ids::MATRIX_COEFFICIENTS, 2);
        put_uint(&mut colour, ids::RANGE, 0);
        put_uint(&mut colour, ids::TRANSFER_CHARACTERISTICS, 2);
        put_uint(&mut colour, ids::PRIMARIES, 2);
        let (stored, color) = colour_of(b"V_VP9", &colour);
        assert_eq!(stored.expect("colour").primaries, Some(2));
        assert!(color.is_empty());

        for (matrix, range, transfer, primaries) in [
            (3, 3, 0, 0),
            (18, 4, 3, 3),
            (300, 9, 19, 13),
            (u64::MAX, 2, 255, 21),
        ] {
            let mut colour = Vec::new();
            put_uint(&mut colour, ids::MATRIX_COEFFICIENTS, matrix);
            put_uint(&mut colour, ids::RANGE, range);
            put_uint(&mut colour, ids::TRANSFER_CHARACTERISTICS, transfer);
            put_uint(&mut colour, ids::PRIMARIES, primaries);
            let (_, color) = colour_of(b"V_VP9", &colour);
            assert_eq!(color.matrix, None, "matrix {matrix}");
            assert_eq!(color.transfer, None, "transfer {transfer}");
            assert_eq!(color.primaries, None, "primaries {primaries}");
            let full = if range == 2 {
                ColorRange::Full
            } else {
                ColorRange::Unspecified
            };
            assert_eq!(color.range, full, "range {range}");
        }

        // An empty Colour states nothing, and neither does no Colour.
        let (stored, color) = colour_of(b"V_VP9", &[]);
        assert_eq!(stored, Some(Colour::default()));
        assert!(color.is_empty());
    }

    #[test]
    fn every_defined_code_is_kept() {
        for (matrix, transfer, primaries) in [
            (0, 1, 1),
            (1, 4, 4),
            (14, 18, 12),
            (17, 14, 22),
            (10, 15, 5),
        ] {
            let mut colour = Vec::new();
            put_uint(&mut colour, ids::MATRIX_COEFFICIENTS, matrix);
            put_uint(&mut colour, ids::RANGE, 2);
            put_uint(&mut colour, ids::TRANSFER_CHARACTERISTICS, transfer);
            put_uint(&mut colour, ids::PRIMARIES, primaries);
            let (_, color) = colour_of(b"V_MPEG4/ISO/AVC", &colour);
            assert_eq!(color.matrix, Some(matrix as u8));
            assert_eq!(color.transfer, Some(transfer as u8));
            assert_eq!(color.primaries, Some(primaries as u8));
            assert_eq!(color.range, ColorRange::Full);
        }
    }

    #[test]
    fn a_light_level_or_mastering_display_needs_its_values() {
        // MaxCLL alone is a light level whose MaxFALL is unknown (0).
        let mut colour = Vec::new();
        put_uint(&mut colour, ids::MAX_CLL, 70_000);
        let (_, color) = colour_of(b"V_AV1", &colour);
        assert_eq!(
            color.content_light,
            Some(ContentLight {
                max_cll: u16::MAX,
                max_fall: 0,
            })
        );
        // Zeros are unknown.
        let mut colour = Vec::new();
        put_uint(&mut colour, ids::MAX_CLL, 0);
        put_uint(&mut colour, ids::MAX_FALL, 0);
        assert_eq!(colour_of(b"V_AV1", &colour).1.content_light, None);

        // A mastering display missing a value, with a chromaticity past 1,
        // a minimum at its maximum, or a value that is not a number, is
        // not a mastering display.
        let full = Mastering {
            primaries: [
                [Some(0.68), Some(0.32)],
                [Some(0.265), Some(0.69)],
                [Some(0.15), Some(0.06)],
            ],
            white_point: [Some(0.3127), Some(0.329)],
            luminance_max: Some(1000.0),
            luminance_min: Some(0.005),
        };
        let colour = |mastering: Mastering| Colour {
            mastering: Some(mastering),
            ..Colour::default()
        };
        assert_eq!(
            stream_color(&colour(full)).mastering_display,
            Some(HDR10_DISPLAY)
        );
        let mut missing = full;
        missing.luminance_min = None;
        let mut past_one = full;
        past_one.primaries[2][0] = Some(1.5);
        let mut level = full;
        level.luminance_min = Some(1000.0);
        let mut white = full;
        white.white_point[1] = None;
        for broken in [missing, past_one, level, white] {
            assert_eq!(stream_color(&colour(broken)).mastering_display, None);
        }
        // A NaN in the file is read as a missing value.
        let mut mastering = Vec::new();
        put_float(&mut mastering, ids::LUMINANCE_MAX, f64::NAN);
        let mut element = Vec::new();
        put_master(&mut element, ids::MASTERING_METADATA, &mastering);
        let (stored, color) = colour_of(b"V_AV1", &element);
        assert_eq!(
            stored.expect("colour").mastering.expect("m").luminance_max,
            None
        );
        assert_eq!(color.mastering_display, None);
    }

    /// A Colour that is not well formed is left out and the track opens.
    #[test]
    fn a_colour_that_does_not_parse_is_left_out() {
        // MatrixCoefficients nine octets long.
        let mut colour = Vec::new();
        put_bytes(&mut colour, ids::MATRIX_COEFFICIENTS, &[0; 9]);
        let (stored, color) = colour_of(b"V_VP9", &colour);
        assert_eq!(stored, None);
        assert!(color.is_empty());
        // A float of three octets inside MasteringMetadata.
        let mut mastering = Vec::new();
        put_bytes(&mut mastering, ids::LUMINANCE_MAX, &[0; 3]);
        let mut colour = Vec::new();
        put_uint(&mut colour, ids::PRIMARIES, 9);
        put_master(&mut colour, ids::MASTERING_METADATA, &mastering);
        let (stored, _) = colour_of(b"V_VP9", &colour);
        assert_eq!(stored, None);
    }

    #[test]
    fn written_colour_reads_back_the_same() {
        let color = StreamColor {
            primaries: Some(9),
            transfer: Some(18),
            matrix: Some(0),
            range: ColorRange::Full,
            content_light: Some(ContentLight {
                max_cll: 1000,
                max_fall: 0,
            }),
            mastering_display: Some(HDR10_DISPLAY),
        };
        let mut video = Vec::new();
        put_colour(&mut video, &color);
        let body = Children::new(&video)
            .next()
            .expect("colour")
            .expect("element");
        assert_eq!(body.id, ids::COLOUR);
        let (_, back) = colour_of(b"V_AV1", body.data);
        assert_eq!(back, color);
        // Studio swing is broadcast range.
        let studio = StreamColor {
            range: ColorRange::Studio,
            ..StreamColor::default()
        };
        let mut video = Vec::new();
        put_colour(&mut video, &studio);
        let body = Children::new(&video)
            .next()
            .expect("colour")
            .expect("element");
        assert_eq!(colour_of(b"V_AV1", body.data).1.range, ColorRange::Limited);
        // Nothing stated writes nothing.
        let mut video = Vec::new();
        put_colour(&mut video, &StreamColor::default());
        assert!(video.is_empty());
    }
}
