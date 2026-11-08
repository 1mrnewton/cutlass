//! Quantization matrices and Apple's target data rates, per profile.

use crate::profile::Profile;

/// The matrices Apple's own encoder writes for each profile, read from
/// VideoToolbox streams, raster order `[v * 8 + u]`.
const PROXY: [u8; 64] = [
    4, 7, 9, 11, 13, 14, 63, 63, 7, 7, 11, 12, 14, 63, 63, 63, 9, 11, 13, 14, 63, 63, 63, 63, 11,
    11, 13, 14, 63, 63, 63, 63, 11, 13, 14, 63, 63, 63, 63, 63, 13, 14, 63, 63, 63, 63, 63, 63, 13,
    63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
];
const LT: [u8; 64] = [
    4, 5, 6, 7, 9, 11, 13, 15, 5, 5, 7, 8, 11, 13, 15, 17, 6, 7, 9, 11, 13, 15, 15, 17, 7, 7, 9,
    11, 13, 15, 17, 19, 7, 9, 11, 13, 14, 16, 19, 23, 9, 11, 13, 14, 16, 19, 23, 29, 9, 11, 13, 15,
    17, 21, 28, 35, 11, 13, 16, 17, 21, 28, 35, 41,
];
const STANDARD: [u8; 64] = [
    4, 4, 5, 5, 6, 7, 7, 9, 4, 4, 5, 6, 7, 7, 9, 9, 5, 5, 6, 7, 7, 9, 9, 10, 5, 5, 6, 7, 7, 9, 9,
    10, 5, 6, 7, 7, 8, 9, 10, 12, 6, 7, 7, 8, 9, 10, 12, 15, 6, 7, 7, 9, 10, 11, 14, 17, 7, 7, 9,
    10, 11, 14, 17, 21,
];
const HQ: [u8; 64] = [
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5,
    4, 4, 4, 4, 4, 4, 5, 5, 4, 4, 4, 4, 4, 5, 5, 6, 4, 4, 4, 4, 5, 5, 6, 7, 4, 4, 4, 4, 5, 6, 7, 7,
];
const XQ_LUMA: [u8; 64] = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3,
    2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 4, 2, 2, 2, 2, 3, 3, 4, 4,
];

/// Luma and chroma matrices for `profile`.
pub(crate) fn matrices(profile: Profile) -> (&'static [u8; 64], &'static [u8; 64]) {
    match profile {
        Profile::Proxy => (&PROXY, &PROXY),
        Profile::Lt => (&LT, &LT),
        Profile::Standard => (&STANDARD, &STANDARD),
        Profile::Hq | Profile::P4444 => (&HQ, &HQ),
        Profile::Xq => (&XQ_LUMA, &HQ),
    }
}

/// Apple's ProRes 422 HQ targets at 29.97 frames a second, from the ProRes
/// white paper (April 2022): pixels per frame, then megabits a second.
const HQ_TARGETS: [(f64, f64); 12] = [
    (720.0 * 486.0, 63.0),
    (960.0 * 720.0, 94.0),
    (1280.0 * 720.0, 110.0),
    (1440.0 * 1080.0, 189.0),
    (1920.0 * 1080.0, 220.0),
    (2048.0 * 1080.0, 251.0),
    (2048.0 * 1556.0, 340.0),
    (3840.0 * 2160.0, 884.0),
    (4096.0 * 2160.0, 943.0),
    (5120.0 * 2700.0, 1473.0),
    (6144.0 * 3240.0, 2121.0),
    (8192.0 * 4320.0, 3771.0),
];

/// Each profile's target over HQ's, from the 1920x1080 row of the same table.
fn factor(profile: Profile) -> f64 {
    match profile {
        Profile::Proxy => 45.0 / 220.0,
        Profile::Lt => 102.0 / 220.0,
        Profile::Standard => 147.0 / 220.0,
        Profile::Hq => 1.0,
        Profile::P4444 => 330.0 / 220.0,
        Profile::Xq => 495.0 / 220.0,
    }
}

/// Apple's target in megabits a second at 29.97 frames a second, without alpha.
/// Sizes between rows of the table interpolate; sizes outside keep the
/// nearest row's bits per pixel.
#[must_use]
pub fn target_mbps(profile: Profile, width: u32, height: u32) -> f64 {
    let pixels = f64::from(width) * f64::from(height);
    let (first, last) = (HQ_TARGETS[0], HQ_TARGETS[HQ_TARGETS.len() - 1]);
    let hq = if pixels <= first.0 {
        first.1 * pixels / first.0
    } else if pixels >= last.0 {
        last.1 * pixels / last.0
    } else {
        let mut value = last.1;
        for pair in HQ_TARGETS.windows(2) {
            let ((p0, r0), (p1, r1)) = (pair[0], pair[1]);
            if pixels <= p1 {
                value = r0 + (r1 - r0) * (pixels - p0) / (p1 - p0);
                break;
            }
        }
        value
    };
    hq * factor(profile)
}

/// Bits one frame may spend: the target over 29.97 frames, so a frame's
/// budget is the same at any frame rate, as Apple's encoder does.
#[must_use]
pub fn target_frame_bits(profile: Profile, width: u32, height: u32) -> u64 {
    (target_mbps(profile, width, height) * 1e6 * 1001.0 / 30_000.0) as u64
}

#[cfg(test)]
mod tests {
    use super::{matrices, target_mbps};
    use crate::profile::Profile;

    #[test]
    fn white_paper_rows_come_back() {
        let near = |a: f64, b: f64| (a - b).abs() < 0.5;
        assert!(near(target_mbps(Profile::Hq, 1920, 1080), 220.0));
        assert!(near(target_mbps(Profile::Proxy, 1920, 1080), 45.0));
        assert!(near(target_mbps(Profile::Lt, 1920, 1080), 102.0));
        assert!(near(target_mbps(Profile::Standard, 1920, 1080), 147.0));
        assert!(near(target_mbps(Profile::P4444, 1920, 1080), 330.0));
        assert!(near(target_mbps(Profile::Xq, 1920, 1080), 495.0));
        assert!(near(target_mbps(Profile::Hq, 3840, 2160), 884.0));
        // 4K rows for the other profiles land within 1% of the paper.
        for (profile, paper) in [
            (Profile::Proxy, 182.0),
            (Profile::Lt, 410.0),
            (Profile::Standard, 589.0),
            (Profile::P4444, 1326.0),
            (Profile::Xq, 1989.0),
        ] {
            let ours = target_mbps(profile, 3840, 2160);
            assert!((ours / paper - 1.0).abs() < 0.01, "{profile:?} {ours}");
        }
        let small = target_mbps(Profile::Hq, 360, 243);
        assert!(near(small * 4.0, 63.0));
    }

    #[test]
    fn every_weight_is_legal() {
        for profile in Profile::ALL {
            let (luma, chroma) = matrices(profile);
            assert!(luma.iter().chain(chroma).all(|&w| (2..=63).contains(&w)));
        }
    }
}
