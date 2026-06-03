//! Color conversion for the CPU compositor.
//!
//! This is the **reference** color path: correct and simple, the ground truth a
//! future GPU shader must match. It assumes **BT.709, limited (video) range**,
//! which is the common case for HD/UHD H.264/HEVC. Real pipelines should read
//! the matrix/range from stream metadata (BT.601 for SD, full-range JPEG, etc.);
//! wiring that through is a known follow-up, tracked here rather than silently
//! hard-coded elsewhere.

/// Convert one BT.709 limited-range YUV sample to opaque RGBA8.
pub fn yuv_to_rgba(y: u8, u: u8, v: u8) -> [u8; 4] {
    // Limited range: Y in [16,235], chroma centered at 128.
    let c = y as f32 - 16.0;
    let d = u as f32 - 128.0;
    let e = v as f32 - 128.0;

    let r = 1.164 * c + 1.793 * e;
    let g = 1.164 * c - 0.213 * d - 0.533 * e;
    let b = 1.164 * c + 2.112 * d;

    [clamp_u8(r), clamp_u8(g), clamp_u8(b), 255]
}

fn clamp_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limited_range_black_and_white() {
        assert_eq!(yuv_to_rgba(16, 128, 128), [0, 0, 0, 255]);
        assert_eq!(yuv_to_rgba(235, 128, 128), [255, 255, 255, 255]);
    }

    #[test]
    fn primaries_are_in_expected_corner() {
        // High V pushes red; high U pushes blue.
        let red = yuv_to_rgba(81, 90, 240);
        assert!(red[0] > 200 && red[2] < 80, "got {red:?}");
        let blue = yuv_to_rgba(41, 240, 110);
        assert!(blue[2] > 200 && blue[0] < 80, "got {blue:?}");
    }
}
