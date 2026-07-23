//! Color math for the desktop color picker.
//!
//! Pure conversions used by the Slint `ColorUtil` global and by unit tests.
//! Slint can construct colors with `hsv()` but cannot parse/format hex at
//! runtime, and RGBA↔HSV round-trips need hue preservation so blacks/grays
//! don't collapse the picker's hue/sat state.

/// RGB → HSV. Hue in `[0, 360)`, saturation and value in `[0, 1]`.
///
/// Achromatic inputs (`s == 0`) yield `h == 0`; callers that care about
/// preserving a previous hue should use [`rgb_to_hsv_preserving`].
pub fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let r = r.clamp(0.0, 1.0);
    let g = g.clamp(0.0, 1.0);
    let b = b.clamp(0.0, 1.0);

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let h = if delta <= f32::EPSILON {
        0.0
    } else if (max - r).abs() <= f32::EPSILON {
        60.0 * (((g - b) / delta) % 6.0)
    } else if (max - g).abs() <= f32::EPSILON {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };
    let h = if h < 0.0 { h + 360.0 } else { h };

    let s = if max <= f32::EPSILON {
        0.0
    } else {
        delta / max
    };

    (h, s, max)
}

/// HSV → RGB. Hue in degrees (wrapped); saturation and value in `[0, 1]`.
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let s = s.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    let h = h.rem_euclid(360.0);

    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r1, g1, b1) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    (r1 + m, g1 + m, b1 + m)
}

/// Convert RGB → HSV while preserving hue (and sat when black) so the
/// picker's internal HSV state doesn't collapse on grays/blacks.
///
/// * Black (`v ≈ 0`): keep `prev_h` and `prev_s`.
/// * Gray / white (`s ≈ 0`, `v > 0`): keep `prev_h`, set `s = 0`.
pub fn rgb_to_hsv_preserving(r: f32, g: f32, b: f32, prev_h: f32, prev_s: f32) -> (f32, f32, f32) {
    let (h, s, v) = rgb_to_hsv(r, g, b);
    const EPS: f32 = 1.0 / 255.0;
    if v <= EPS {
        (prev_h.rem_euclid(360.0), prev_s.clamp(0.0, 1.0), 0.0)
    } else if s <= EPS {
        (prev_h.rem_euclid(360.0), 0.0, v)
    } else {
        (h, s, v)
    }
}

/// Map normalized SV-square coordinates (`nx`,`ny` in `[0, 1]`, origin top-left)
/// to saturation / value.
///
/// Kept for the color picker (and unit tests). The bin target does not call
/// these yet, so silence unused warnings until the Slint panel lands.
#[allow(dead_code)]
pub fn sv_from_norm(nx: f32, ny: f32) -> (f32, f32) {
    (nx.clamp(0.0, 1.0), (1.0 - ny).clamp(0.0, 1.0))
}

/// Map saturation / value to normalized SV-square coordinates.
#[allow(dead_code)]
pub fn norm_from_sv(s: f32, v: f32) -> (f32, f32) {
    (s.clamp(0.0, 1.0), (1.0 - v.clamp(0.0, 1.0)).clamp(0.0, 1.0))
}

/// Map normalized hue-strip `nx` in `[0, 1]` to hue degrees `[0, 360)`.
#[allow(dead_code)]
pub fn hue_from_norm(nx: f32) -> f32 {
    (nx.clamp(0.0, 1.0) * 360.0).rem_euclid(360.0)
}

/// Map hue degrees to normalized hue-strip position.
#[allow(dead_code)]
pub fn norm_from_hue(h: f32) -> f32 {
    (h.rem_euclid(360.0) / 360.0).clamp(0.0, 1.0)
}

/// Map normalized alpha-strip `nx` in `[0, 1]` to alpha in `[0, 1]`.
#[allow(dead_code)]
pub fn alpha_from_norm(nx: f32) -> f32 {
    nx.clamp(0.0, 1.0)
}

/// Map alpha in `[0, 1]` to normalized alpha-strip position.
#[allow(dead_code)]
pub fn norm_from_alpha(a: f32) -> f32 {
    a.clamp(0.0, 1.0)
}

/// Parse a hex color string into RGBA bytes.
///
/// Accepts `#RGB`, `#RRGGBB`, `#RRGGBBAA` (leading `#` optional, case-insensitive).
/// Short `#RGB` expands by digit doubling; alpha defaults to `0xFF` when omitted.
pub fn parse_hex(input: &str) -> Option<[u8; 4]> {
    let s = input.trim();
    let s = s.strip_prefix('#').unwrap_or(s);
    if !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    match s.len() {
        3 => {
            let r = nibble(s.as_bytes()[0])? * 0x11;
            let g = nibble(s.as_bytes()[1])? * 0x11;
            let b = nibble(s.as_bytes()[2])? * 0x11;
            Some([r, g, b, 0xFF])
        }
        6 => {
            let r = byte_pair(&s[0..2])?;
            let g = byte_pair(&s[2..4])?;
            let b = byte_pair(&s[4..6])?;
            Some([r, g, b, 0xFF])
        }
        8 => {
            let r = byte_pair(&s[0..2])?;
            let g = byte_pair(&s[2..4])?;
            let b = byte_pair(&s[4..6])?;
            let a = byte_pair(&s[6..8])?;
            Some([r, g, b, a])
        }
        _ => None,
    }
}

/// Format RGBA as `RRGGBB`, or `RRGGBBAA` when alpha is not fully opaque.
/// No leading `#`.
pub fn format_hex(r: u8, g: u8, b: u8, a: u8) -> String {
    if a == 0xFF {
        format!("{r:02X}{g:02X}{b:02X}")
    } else {
        format!("{r:02X}{g:02X}{b:02X}{a:02X}")
    }
}

fn nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn byte_pair(s: &str) -> Option<u8> {
    u8::from_str_radix(s, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn almost_eq(a: f32, b: f32, eps: f32) {
        assert!((a - b).abs() <= eps, "{a} ≉ {b} (eps={eps})");
    }

    fn rgb_u8_roundtrip(r: u8, g: u8, b: u8) {
        let (rf, gf, bf) = (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
        let (h, s, v) = rgb_to_hsv(rf, gf, bf);
        let (r2, g2, b2) = hsv_to_rgb(h, s, v);
        let to_u8 = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round() as i32;
        assert!((to_u8(r2) - i32::from(r)).abs() <= 1, "R {r} → {r2}");
        assert!((to_u8(g2) - i32::from(g)).abs() <= 1, "G {g} → {g2}");
        assert!((to_u8(b2) - i32::from(b)).abs() <= 1, "B {b} → {b2}");
    }

    #[test]
    fn rgb_hsv_anchors() {
        let (h, s, v) = rgb_to_hsv(0.0, 0.0, 0.0);
        almost_eq(s, 0.0, 1e-5);
        almost_eq(v, 0.0, 1e-5);
        let _ = h;

        let (_, s, v) = rgb_to_hsv(1.0, 1.0, 1.0);
        almost_eq(s, 0.0, 1e-5);
        almost_eq(v, 1.0, 1e-5);

        let (h, s, v) = rgb_to_hsv(1.0, 0.0, 0.0);
        almost_eq(h, 0.0, 1e-3);
        almost_eq(s, 1.0, 1e-5);
        almost_eq(v, 1.0, 1e-5);

        let (_, s, v) = rgb_to_hsv(0.5, 0.5, 0.5);
        almost_eq(s, 0.0, 1e-5);
        almost_eq(v, 0.5, 1e-5);

        let (r, g, b) = hsv_to_rgb(0.0, 0.0, 0.0);
        almost_eq(r, 0.0, 1e-5);
        almost_eq(g, 0.0, 1e-5);
        almost_eq(b, 0.0, 1e-5);

        let (r, g, b) = hsv_to_rgb(0.0, 0.0, 1.0);
        almost_eq(r, 1.0, 1e-5);
        almost_eq(g, 1.0, 1e-5);
        almost_eq(b, 1.0, 1e-5);

        let (r, g, b) = hsv_to_rgb(0.0, 1.0, 1.0);
        almost_eq(r, 1.0, 1e-5);
        almost_eq(g, 0.0, 1e-5);
        almost_eq(b, 0.0, 1e-5);
    }

    #[test]
    fn rgb_hsv_roundtrip_within_one_over_255() {
        for &(r, g, b) in &[
            (0u8, 0, 0),
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (128, 128, 128),
            (64, 64, 64),
            (192, 192, 192),
            (255, 128, 0),
            (30, 144, 255),
            (1, 2, 3),
            (254, 1, 127),
        ] {
            rgb_u8_roundtrip(r, g, b);
        }
    }

    #[test]
    fn hue_preservation_black_and_gray() {
        let (h, s, v) = rgb_to_hsv_preserving(0.0, 0.0, 0.0, 210.0, 0.75);
        almost_eq(h, 210.0, 1e-3);
        almost_eq(s, 0.75, 1e-5);
        almost_eq(v, 0.0, 1e-5);

        let (h, s, v) = rgb_to_hsv_preserving(0.5, 0.5, 0.5, 210.0, 0.75);
        almost_eq(h, 210.0, 1e-3);
        almost_eq(s, 0.0, 1e-5);
        almost_eq(v, 0.5, 1e-5);

        let (h, s, v) = rgb_to_hsv_preserving(1.0, 0.0, 0.0, 210.0, 0.75);
        almost_eq(h, 0.0, 1e-3);
        almost_eq(s, 1.0, 1e-5);
        almost_eq(v, 1.0, 1e-5);
    }

    #[test]
    fn position_mapping_sv_hue_alpha() {
        assert_eq!(sv_from_norm(0.0, 0.0), (0.0, 1.0));
        assert_eq!(sv_from_norm(1.0, 1.0), (1.0, 0.0));
        assert_eq!(sv_from_norm(0.25, 0.75), (0.25, 0.25));
        assert_eq!(norm_from_sv(0.25, 0.25), (0.25, 0.75));

        almost_eq(hue_from_norm(0.0), 0.0, 1e-5);
        almost_eq(hue_from_norm(0.5), 180.0, 1e-3);
        almost_eq(norm_from_hue(180.0), 0.5, 1e-5);
        almost_eq(norm_from_hue(360.0), 0.0, 1e-5);

        almost_eq(alpha_from_norm(0.4), 0.4, 1e-5);
        almost_eq(norm_from_alpha(0.4), 0.4, 1e-5);
    }

    #[test]
    fn parse_hex_accepted_forms() {
        assert_eq!(parse_hex("#f00"), Some([0xFF, 0x00, 0x00, 0xFF]));
        assert_eq!(parse_hex("0F0"), Some([0x00, 0xFF, 0x00, 0xFF]));
        assert_eq!(parse_hex("#00ff00"), Some([0x00, 0xFF, 0x00, 0xFF]));
        assert_eq!(parse_hex("AABBCC"), Some([0xAA, 0xBB, 0xCC, 0xFF]));
        assert_eq!(parse_hex("#11223380"), Some([0x11, 0x22, 0x33, 0x80]));
        assert_eq!(parse_hex("11223380"), Some([0x11, 0x22, 0x33, 0x80]));
        assert_eq!(parse_hex("  #AbC  "), Some([0xAA, 0xBB, 0xCC, 0xFF]));
    }

    #[test]
    fn parse_hex_rejects_garbage() {
        assert!(parse_hex("").is_none());
        assert!(parse_hex("#").is_none());
        assert!(parse_hex("#12").is_none());
        assert!(parse_hex("#1234").is_none());
        assert!(parse_hex("#12345").is_none());
        assert!(parse_hex("#1234567").is_none());
        assert!(parse_hex("#123456789").is_none());
        assert!(parse_hex("#gg0000").is_none());
        assert!(parse_hex("red").is_none());
        assert!(parse_hex("#12 34 56").is_none());
    }

    #[test]
    fn format_hex_roundtrips() {
        assert_eq!(format_hex(0xFF, 0x00, 0x00, 0xFF), "FF0000");
        assert_eq!(format_hex(0x11, 0x22, 0x33, 0x80), "11223380");
        assert_eq!(format_hex(0x00, 0x00, 0x00, 0xFF), "000000");
        assert_eq!(format_hex(0xAB, 0xCD, 0xEF, 0x00), "ABCDEF00");

        for &(hex, rgba) in &[
            ("#f00", [0xFFu8, 0x00, 0x00, 0xFF]),
            ("00FF00", [0x00, 0xFF, 0x00, 0xFF]),
            ("#11223380", [0x11, 0x22, 0x33, 0x80]),
        ] {
            let parsed = parse_hex(hex).unwrap();
            assert_eq!(parsed, rgba);
            let formatted = format_hex(parsed[0], parsed[1], parsed[2], parsed[3]);
            assert_eq!(parse_hex(&formatted), Some(rgba));
        }
    }
}
