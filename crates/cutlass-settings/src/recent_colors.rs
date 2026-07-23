//! Most-recent-first color history for the desktop picker.
//!
//! Stored under `[appearance].recent_colors` as hex strings (`RRGGBB` or
//! `RRGGBBAA`). Pure list logic lives here so unit tests don't need TOML IO.

/// Cap on the persisted / in-memory recent list.
pub const MAX_RECENT_COLORS: usize = 12;

/// RGBA bytes (0–255), matching the rest of the Cutlass color model.
pub type Rgba = [u8; 4];

/// Insert `color` at the front of `list`, removing any prior equal entry and
/// truncating to [`MAX_RECENT_COLORS`]. Equality is full RGBA (alpha matters).
pub fn push_recent(list: &mut Vec<Rgba>, color: Rgba) {
    list.retain(|c| *c != color);
    list.insert(0, color);
    list.truncate(MAX_RECENT_COLORS);
}

/// Format RGBA as `RRGGBB`, or `RRGGBBAA` when alpha is not fully opaque.
/// No leading `#` — matches the desktop `ColorUtil` convention.
pub fn format_hex(rgba: Rgba) -> String {
    let [r, g, b, a] = rgba;
    if a == 0xFF {
        format!("{r:02X}{g:02X}{b:02X}")
    } else {
        format!("{r:02X}{g:02X}{b:02X}{a:02X}")
    }
}

/// Parse a hex color string into RGBA bytes.
///
/// Accepts `#RGB`, `#RRGGBB`, `#RRGGBBAA` (leading `#` optional, case-insensitive).
/// Short `#RGB` expands by digit doubling; alpha defaults to `0xFF` when omitted.
pub fn parse_hex(input: &str) -> Option<Rgba> {
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

    #[test]
    fn push_recent_orders_dedupes_and_caps() {
        let mut list = Vec::new();
        push_recent(&mut list, [1, 0, 0, 255]);
        push_recent(&mut list, [0, 1, 0, 255]);
        push_recent(&mut list, [1, 0, 0, 255]); // moves to front
        assert_eq!(list, vec![[1, 0, 0, 255], [0, 1, 0, 255]]);

        for i in 0..20u8 {
            push_recent(&mut list, [i, i, i, 255]);
        }
        assert_eq!(list.len(), MAX_RECENT_COLORS);
        assert_eq!(list[0], [19, 19, 19, 255]);
        assert_eq!(list[MAX_RECENT_COLORS - 1], [8, 8, 8, 255]);
    }

    #[test]
    fn push_recent_treats_alpha_as_distinct() {
        let mut list = Vec::new();
        push_recent(&mut list, [10, 20, 30, 255]);
        push_recent(&mut list, [10, 20, 30, 128]);
        assert_eq!(list.len(), 2);
        assert_eq!(list[0], [10, 20, 30, 128]);
    }

    #[test]
    fn hex_round_trip() {
        assert_eq!(parse_hex("ff00aa"), Some([0xFF, 0x00, 0xAA, 0xFF]));
        assert_eq!(parse_hex("#ABC"), Some([0xAA, 0xBB, 0xCC, 0xFF]));
        assert_eq!(parse_hex("01020380"), Some([1, 2, 3, 0x80]));
        assert_eq!(format_hex([1, 2, 3, 255]), "010203");
        assert_eq!(format_hex([1, 2, 3, 0x80]), "01020380");
        assert!(parse_hex("zz").is_none());
        assert!(parse_hex("12345").is_none());
    }
}
