//! Levels (AV1 Annex A.3): the smallest one a stream fits.
//!
//! A level bounds the picture size, the sample rates, the bitrate and the
//! tile counts a decoder must handle. The sequence header names one per
//! operating point, and a player or a hardware decoder may refuse a level it
//! does not reach, so the encoder names the lowest level that holds instead
//! of 31, "maximum parameters".

/// One row of the level table. Levels the specification names without
/// values (2.2, 2.3, 3.2, 3.3, 4.2, 4.3, 7.x) are left out.
struct Level {
    idx: u8,
    max_pic_size: u64,
    max_h_size: u64,
    max_v_size: u64,
    max_display_rate: u64,
    max_decode_rate: u64,
    /// Megabits per second on the main tier.
    main_mbps: f64,
    /// Megabits per second on the high tier; 0 where the level has none.
    high_mbps: f64,
    max_tiles: u64,
    max_tile_cols: u64,
}

const fn level(
    idx: u8,
    max_pic_size: u64,
    (max_h_size, max_v_size): (u64, u64),
    (max_display_rate, max_decode_rate): (u64, u64),
    (main_mbps, high_mbps): (f64, f64),
    (max_tiles, max_tile_cols): (u64, u64),
) -> Level {
    Level {
        idx,
        max_pic_size,
        max_h_size,
        max_v_size,
        max_display_rate,
        max_decode_rate,
        main_mbps,
        high_mbps,
        max_tiles,
        max_tile_cols,
    }
}

#[rustfmt::skip]
const LEVELS: [Level; 14] = [
    level(0,     147_456, (2048, 1152),  (    4_423_680,     5_529_600), (  1.5,   0.0), (  8,  4)),
    level(1,     278_784, (2816, 1584),  (    8_363_520,    10_454_400), (  3.0,   0.0), (  8,  4)),
    level(4,     665_856, (4352, 2448),  (   19_975_680,    24_969_600), (  6.0,   0.0), ( 16,  6)),
    level(5,   1_065_024, (5504, 3096),  (   31_950_720,    39_938_400), ( 10.0,   0.0), ( 16,  6)),
    level(8,   2_359_296, (6144, 3456),  (   70_778_880,    77_856_768), ( 12.0,  30.0), ( 32,  8)),
    level(9,   2_359_296, (6144, 3456),  (  141_557_760,   155_713_536), ( 20.0,  50.0), ( 32,  8)),
    level(12,  8_912_896, (8192, 4352),  (  267_386_880,   273_715_200), ( 30.0, 100.0), ( 64,  8)),
    level(13,  8_912_896, (8192, 4352),  (  534_773_760,   547_430_400), ( 40.0, 160.0), ( 64,  8)),
    level(14,  8_912_896, (8192, 4352),  (1_069_547_520, 1_094_860_800), ( 60.0, 240.0), ( 64,  8)),
    level(15,  8_912_896, (8192, 4352),  (1_069_547_520, 1_176_502_272), ( 60.0, 240.0), ( 64,  8)),
    level(16, 35_651_584, (16384, 8704), (1_069_547_520, 1_176_502_272), ( 60.0, 240.0), (128, 16)),
    level(17, 35_651_584, (16384, 8704), (2_139_095_040, 2_189_721_600), (100.0, 480.0), (128, 16)),
    level(18, 35_651_584, (16384, 8704), (4_278_190_080, 4_379_443_200), (160.0, 800.0), (128, 16)),
    level(19, 35_651_584, (16384, 8704), (4_278_190_080, 4_706_009_088), (160.0, 800.0), (128, 16)),
];

/// `seq_level_idx` for a stream no level in the table holds.
pub const MAX_PARAMETERS: u8 = 31;

/// What a level is chosen from.
#[derive(Clone, Copy, Debug)]
pub struct Stream {
    pub width: usize,
    pub height: usize,
    /// Pictures shown per second.
    pub shown_per_second: f64,
    /// Frames decoded per second: hidden frames that are not shown through
    /// `show_existing_frame` add to the pictures shown.
    pub decoded_per_second: f64,
    /// Target bits per second; `None` when the stream has no target.
    pub bitrate: Option<u64>,
    pub profile: u8,
    pub tiles: usize,
    pub tile_cols: usize,
}

/// The lowest `seq_level_idx` whose limits `stream` keeps, and the tier
/// (0 main, 1 high) its bitrate needs. The high tier starts at level 4.0.
/// A stream past level 6.3 gets [`MAX_PARAMETERS`].
///
/// The bitrate limit is the tier's rate times the profile factor (A.3: 1, 2
/// and 3 for profiles 0, 1 and 2). The header rate and the compression
/// ratio depend on the coded frames and are not predicted here.
pub fn pick(stream: &Stream) -> (u8, u8) {
    let pic_size = (stream.width * stream.height) as u64;
    let display_rate = pic_size as f64 * stream.shown_per_second;
    let decode_rate = pic_size as f64 * stream.decoded_per_second;
    let factor = match stream.profile {
        0 => 1.0,
        1 => 2.0,
        _ => 3.0,
    };
    for l in &LEVELS {
        if pic_size > l.max_pic_size
            || stream.width as u64 > l.max_h_size
            || stream.height as u64 > l.max_v_size
            || display_rate > l.max_display_rate as f64
            || decode_rate > l.max_decode_rate as f64
            || stream.tiles as u64 > l.max_tiles
            || stream.tile_cols as u64 > l.max_tile_cols
        {
            continue;
        }
        let Some(bitrate) = stream.bitrate else {
            return (l.idx, 0);
        };
        let bits = bitrate as f64;
        if bits <= l.main_mbps * factor * 1e6 {
            return (l.idx, 0);
        }
        if bits <= l.high_mbps * factor * 1e6 {
            return (l.idx, 1);
        }
    }
    (MAX_PARAMETERS, 0)
}

#[cfg(test)]
mod tests {
    use super::{MAX_PARAMETERS, Stream, pick};

    fn stream(width: usize, height: usize, fps: f64) -> Stream {
        Stream {
            width,
            height,
            shown_per_second: fps,
            decoded_per_second: fps,
            bitrate: None,
            profile: 0,
            tiles: 1,
            tile_cols: 1,
        }
    }

    #[test]
    fn common_sizes_get_the_level_of_the_specification() {
        // 240p and 360p are 2.0 and 2.1, 480p and 720p are 3.0 and 3.1,
        // 1080p is 4.0 at 30 and 4.1 at 60, 2160p is 5.0, 5.1 and 5.2 at
        // 30, 60 and 120, and 4320p is 6.0, 6.1 and 6.2.
        for (w, h, fps, want) in [
            (426, 240, 30.0, 0),
            (640, 360, 30.0, 1),
            (854, 480, 30.0, 4),
            (1280, 720, 30.0, 5),
            (1920, 1080, 30.0, 8),
            (1920, 1080, 60.0, 9),
            (3840, 2160, 30.0, 12),
            (3840, 2160, 60.0, 13),
            (3840, 2160, 120.0, 14),
            (7680, 4320, 30.0, 16),
            (7680, 4320, 60.0, 17),
            (7680, 4320, 120.0, 18),
        ] {
            assert_eq!(pick(&stream(w, h, fps)), (want, 0), "{w}x{h} at {fps}");
        }
    }

    #[test]
    fn each_limit_raises_the_level() {
        // A tall picture inside the picture size of 2.0 is past its height.
        assert_eq!(pick(&stream(100, 1200, 30.0)).0, 1);
        // Hidden frames count toward the decode rate, not the display rate.
        let mut s = stream(1920, 1080, 34.0);
        assert_eq!(pick(&s).0, 8);
        s.decoded_per_second = 34.0 * 9.0 / 8.0;
        assert_eq!(pick(&s).0, 9);
        // Tiles and tile columns.
        let mut s = stream(1280, 720, 30.0);
        s.tiles = 20;
        assert_eq!(pick(&s).0, 8);
        s.tiles = 8;
        s.tile_cols = 8;
        assert_eq!(pick(&s).0, 8);
        s.tile_cols = 9;
        assert_eq!(pick(&s).0, 16);
    }

    #[test]
    fn a_bitrate_picks_the_tier_and_the_profile_scales_it() {
        let mut s = stream(1920, 1080, 30.0);
        s.bitrate = Some(12_000_000);
        assert_eq!(pick(&s), (8, 0));
        s.bitrate = Some(12_000_001);
        assert_eq!(pick(&s), (8, 1));
        s.bitrate = Some(30_000_001);
        assert_eq!(pick(&s), (9, 1));
        // Below level 4.0 there is no high tier: the level rises.
        let mut s = stream(640, 360, 30.0);
        s.bitrate = Some(3_000_001);
        assert_eq!(pick(&s), (4, 0));
        // Profile 1 doubles the rate a tier allows.
        s.profile = 1;
        assert_eq!(pick(&s), (1, 0));
    }

    #[test]
    fn a_stream_past_the_last_level_is_maximum_parameters() {
        assert_eq!(pick(&stream(16384, 16384, 30.0)), (MAX_PARAMETERS, 0));
        assert_eq!(pick(&stream(7680, 4320, 300.0)), (MAX_PARAMETERS, 0));
        let mut s = stream(1920, 1080, 30.0);
        s.bitrate = Some(900_000_000);
        assert_eq!(pick(&s), (MAX_PARAMETERS, 0));
    }
}
