//! Apple's `iTunSMPB` gapless note in `moov/udta/meta/ilst`.
//!
//! An encoder that writes no edit list states an AAC track's priming there:
//! a `----` item whose `name` is `iTunSMPB` and whose `data` is text of hex
//! fields, ` 00000000 00000840 000002C0 00000000000A4100 ...`. The second is
//! the priming, the third the padding after the last valid sample, and the
//! fourth the valid samples between them, all in the track's media ticks.
//! `afconvert` and iTunes write it; FFmpeg's `mov_read_custom` reads it.

use crate::legacy::Gapless;
use crate::limits::MAX_ATOMS;
use crate::parse;

/// Boxes walked under one `udta` before the note is given up.
const MAX_BOXES: u32 = 4_096;

/// The most priming taken, as FFmpeg's `mov_read_custom` bounds it.
const MAX_PRIMING: u64 = 16_384;

/// The note in a `udta` payload. Anything malformed is no note: user data
/// never stops a file from opening.
pub fn in_udta(udta: &[u8]) -> Option<Gapless> {
    let mut budget = MAX_ATOMS.saturating_sub(MAX_BOXES);
    let meta = child(udta, b"meta", &mut budget)?;
    // ISO `meta` is a full box. QuickTime's has no version word, so its
    // first child's type sits at byte 4.
    let body = if meta.get(4..8) == Some(b"hdlr") {
        meta
    } else {
        meta.get(4..)?
    };
    let ilst = child(body, b"ilst", &mut budget)?;
    for item in parse::atoms(ilst, 0, &mut budget).ok()? {
        if item.kind != *b"----" {
            continue;
        }
        let mut name = None;
        let mut data = None;
        for part in parse::atoms(item.payload, 0, &mut budget).ok()? {
            match &part.kind {
                // Version and flags, then the text.
                b"name" => name = part.payload.get(4..),
                // Version and type, a locale, then the text.
                b"data" => data = part.payload.get(8..),
                _ => {}
            }
        }
        if name == Some(b"iTunSMPB".as_slice())
            && let Some(note) = data.and_then(parse_text)
        {
            return Some(note);
        }
    }
    None
}

fn child<'a>(data: &'a [u8], kind: &[u8; 4], budget: &mut u32) -> Option<&'a [u8]> {
    parse::atoms(data, 0, budget)
        .ok()?
        .into_iter()
        .find(|atom| atom.kind == *kind)
        .map(|atom| atom.payload)
}

/// The priming, remainder and valid counts: the second, third and fourth
/// hex fields. A field is at most 16 digits, as a 64-bit count is.
fn parse_text(text: &[u8]) -> Option<Gapless> {
    let text = std::str::from_utf8(text).ok()?;
    let mut fields = text
        .split(|byte: char| byte.is_ascii_whitespace() || byte == '\0')
        .filter(|field| !field.is_empty());
    let mut next = || {
        let field = fields.next()?;
        if field.len() > 16 {
            return None;
        }
        u64::from_str_radix(field, 16).ok()
    };
    let _reserved = next()?;
    let priming = next()?;
    let remainder = next()?;
    let valid = next()?;
    if priming >= MAX_PRIMING {
        return None;
    }
    Some(Gapless {
        priming,
        remainder,
        valid,
    })
}

#[cfg(test)]
mod tests {
    use super::{Gapless, in_udta, parse_text};

    const TEXT: &[u8] = b" 00000000 00000840 000002C0 00000000000A4100 00000000 00000000";

    fn box_of(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = u32::try_from(payload.len() + 8)
            .unwrap()
            .to_be_bytes()
            .to_vec();
        out.extend(kind);
        out.extend(payload);
        out
    }

    fn udta(name: &[u8], text: &[u8], full_meta: bool) -> Vec<u8> {
        let mut item = box_of(b"mean", b"\0\0\0\0com.apple.iTunes");
        item.extend(box_of(b"name", &[b"\0\0\0\0".as_slice(), name].concat()));
        item.extend(box_of(
            b"data",
            &[b"\0\0\0\x01\0\0\0\0".as_slice(), text].concat(),
        ));
        let mut meta = if full_meta { vec![0; 4] } else { Vec::new() };
        meta.extend(box_of(b"hdlr", &[0; 25]));
        // An unrelated item first: the note is found among others.
        let mut ilst = box_of(b"\xa9too", &box_of(b"data", b"\0\0\0\x01\0\0\0\0tool"));
        ilst.extend(box_of(b"----", &item));
        meta.extend(box_of(b"ilst", &ilst));
        box_of(b"meta", &meta)
    }

    #[test]
    fn the_note_afconvert_writes_is_read() {
        let want = Gapless {
            priming: 2_112,
            remainder: 704,
            valid: 672_000,
        };
        assert_eq!(in_udta(&udta(b"iTunSMPB", TEXT, true)), Some(want));
        // QuickTime's `meta` has no version word.
        assert_eq!(in_udta(&udta(b"iTunSMPB", TEXT, false)), Some(want));
    }

    #[test]
    fn another_item_or_broken_text_is_no_note() {
        assert_eq!(in_udta(&udta(b"iTunNORM", TEXT, true)), None);
        assert_eq!(in_udta(&udta(b"iTunSMPB", b" 00000000 zz", true)), None);
        assert_eq!(
            in_udta(&udta(b"iTunSMPB", b" 00000000 00000840", true)),
            None
        );
        assert_eq!(in_udta(&udta(b"iTunSMPB", b"\xff\xfe", true)), None);
        // A field wider than a 64-bit count would shift the ones after it.
        assert_eq!(parse_text(b"0 00000000000000840 2C0 A4100"), None);
        // Priming past what an encoder needs, as FFmpeg bounds it.
        assert_eq!(parse_text(b"0 4000 2C0 A4100"), None);
        assert!(parse_text(b"0 3FFF 2C0 A4100").is_some());
    }

    #[test]
    fn truncated_user_data_is_no_note() {
        let whole = udta(b"iTunSMPB", TEXT, true);
        for len in 0..whole.len() {
            assert_eq!(in_udta(&whole[..len]), None, "{len}");
        }
        assert_eq!(in_udta(&[0xff; 64]), None);
    }

    #[test]
    fn a_note_must_cover_the_track_and_trim_something() {
        let note = Gapless {
            priming: 2_112,
            remainder: 704,
            valid: 672_000,
        };
        assert_eq!(note.fitting(674_816), Some(note));
        assert_eq!(note.fitting(674_815), None);
        let nothing = Gapless {
            priming: 0,
            remainder: 0,
            valid: 4_096,
        };
        assert_eq!(nothing.fitting(4_096), None);
        let huge = Gapless {
            priming: 1,
            remainder: u64::MAX,
            valid: u64::MAX,
        };
        assert_eq!(huge.fitting(0), None);
    }
}
