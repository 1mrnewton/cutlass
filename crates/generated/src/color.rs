use std::fmt;

use cutlass_picture::{CpuBytes, rgba_len};

/// Why a solid frame could not be filled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillError {
    /// An edge is past the device limit.
    Extent,
}

impl fmt::Display for FillError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("the frame size is past the limit")
    }
}

impl std::error::Error for FillError {}

/// RGBA bytes from a packed `0xAARRGGBB` value.
#[must_use]
pub fn rgba_from_packed(packed: u32) -> [u8; 4] {
    [
        byte(packed, 16),
        byte(packed, 8),
        byte(packed, 0),
        byte(packed, 24),
    ]
}

fn byte(packed: u32, shift: u32) -> u8 {
    let value = (packed >> shift) & 0xff;
    u8::try_from(value).unwrap_or(0)
}

/// `0xAARRGGBB`, `#AARRGGBB`, or eight hex digits. Anything else is refused.
#[must_use]
pub fn parse_packed(text: &str) -> Option<u32> {
    let text = text.trim();
    let hex = text
        .strip_prefix("0x")
        .or_else(|| text.strip_prefix("0X"))
        .or_else(|| text.strip_prefix('#'))
        .unwrap_or(text);
    if hex.len() != 8 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    u32::from_str_radix(hex, 16).ok()
}

/// Fill every pixel with `packed`. One buffer, four bytes per pixel.
pub fn fill(width: u32, height: u32, packed: u32) -> Result<CpuBytes, FillError> {
    let len = rgba_len(width, height).ok_or(FillError::Extent)?;
    let mut rgba = Vec::new();
    rgba.try_reserve_exact(len).map_err(|_| FillError::Extent)?;
    let pixel = rgba_from_packed(packed);
    let pixels = len / 4;
    for _ in 0..pixels {
        rgba.extend_from_slice(&pixel);
    }
    CpuBytes::new(width, height, rgba).map_err(|_| FillError::Extent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_packed_color_fills_every_pixel() {
        let opaque = fill(2, 2, 0xFF33_6699).unwrap();
        assert_eq!(opaque.rgba().len(), 16);
        assert_eq!(opaque.rgba(), &[0x33, 0x66, 0x99, 0xFF].repeat(4));
        let translucent = fill(1, 1, 0x80FF_0000).unwrap();
        assert_eq!(translucent.rgba(), &[0xFF, 0x00, 0x00, 0x80]);
    }

    #[test]
    fn a_packed_color_is_eight_hex_digits() {
        assert_eq!(parse_packed("0xFF336699"), Some(0xFF33_6699));
        assert_eq!(parse_packed("#80ff0000"), Some(0x80FF_0000));
        assert_eq!(parse_packed("00ff00ff"), Some(0x00FF_00FF));
        assert_eq!(parse_packed("  #11223344  "), Some(0x1122_3344));
        assert_eq!(parse_packed("white"), None);
        assert_eq!(parse_packed("0xFF3366"), None);
        assert_eq!(parse_packed(""), None);
    }

    #[test]
    fn an_edge_past_the_limit_is_refused() {
        assert_eq!(fill(100_000, 1, 0), Err(FillError::Extent));
    }
}
