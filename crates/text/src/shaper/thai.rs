//! What Thai and Lao need before a face's own tables see them.
//!
//! The vowel sara am is one character that is drawn as two: a small
//! circle over the consonant and a vowel after it. A face's tables expect
//! the two, with the circle under any tone mark and not after it, so the
//! character is split and its circle moved back over the marks above the
//! consonant, as every shaper of these scripts does.

use super::buffer::Glyph;

/// Sara am, in Thai or in Lao, which is laid out the same 128 further on.
fn sara_am(code: u32) -> bool {
    code & !0x80 == 0x0E33
}

/// A mark set above its consonant, in either script.
fn above(code: u32) -> bool {
    matches!(code & !0x80, 0x0E31 | 0x0E34..=0x0E37 | 0x0E3B | 0x0E47..=0x0E4E)
}

/// Split each sara am of `glyphs` into its circle and its vowel.
/// `describe` makes the glyph of a character at a place in the text.
pub(super) fn split(glyphs: Vec<Glyph>, describe: impl Fn(char, usize) -> Glyph) -> Vec<Glyph> {
    if !glyphs
        .iter()
        .any(|glyph| sara_am(u32::from(glyph.character)))
    {
        return glyphs;
    }
    let mut out: Vec<Glyph> = Vec::with_capacity(glyphs.len() + 4);
    for glyph in glyphs {
        let code = u32::from(glyph.character);
        let parts = if sara_am(code) {
            // The circle is 26 after sara am, and the vowel one before it.
            char::from_u32(code + 0x1A).zip(char::from_u32(code - 1))
        } else {
            None
        };
        let Some((circle, vowel)) = parts else {
            out.push(glyph);
            continue;
        };
        // The circle goes before the marks above the consonant, and the
        // consonant, its marks, the circle and the vowel are one place in
        // the text.
        let mut start = out.len();
        while start > 0 && above(u32::from(out[start - 1].character)) {
            start -= 1;
        }
        let from = start.saturating_sub(1);
        let cluster = out[from..]
            .iter()
            .map(|earlier| earlier.cluster)
            .min()
            .map_or(glyph.cluster, |earliest| earliest.min(glyph.cluster));
        for earlier in &mut out[from..] {
            earlier.cluster = cluster;
        }
        let mut circle = describe(circle, cluster);
        circle.continues = true;
        // It keeps its place among the marks: it is not sorted with them.
        circle.combining = 0;
        out.insert(start, circle);
        let mut vowel = describe(vowel, cluster);
        vowel.continues = true;
        out.push(vowel);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::split;
    use crate::shaper::buffer::Glyph;

    fn parts(text: &str) -> Vec<(u32, usize)> {
        let glyphs: Vec<Glyph> = text
            .char_indices()
            .map(|(at, character)| Glyph::new(character, at))
            .collect();
        split(glyphs, Glyph::new)
            .iter()
            .map(|glyph| (u32::from(glyph.character), glyph.cluster))
            .collect()
    }

    #[test]
    fn sara_am_is_its_circle_and_its_vowel() {
        // Consonant, sara am: consonant, circle, vowel, in one place.
        assert_eq!(
            parts("\u{E19}\u{E33}"),
            [(0xE19, 0), (0xE4D, 0), (0xE32, 0)]
        );
        // With a tone mark: the circle goes under the mark, before it.
        assert_eq!(
            parts("\u{E19}\u{E49}\u{E33}"),
            [(0xE19, 0), (0xE4D, 0), (0xE49, 0), (0xE32, 0)]
        );
        // Lao the same way.
        assert_eq!(
            parts("\u{E99}\u{EC9}\u{EB3}"),
            [(0xE99, 0), (0xECD, 0), (0xEC9, 0), (0xEB2, 0)]
        );
        // The letters before keep their own places.
        assert_eq!(
            parts("\u{E01}\u{E19}\u{E33}"),
            [(0xE01, 0), (0xE19, 3), (0xE4D, 3), (0xE32, 3)]
        );
    }

    #[test]
    fn other_text_is_left_as_it_is() {
        assert_eq!(
            parts("\u{E01}\u{E32}a"),
            [(0xE01, 0), (0xE32, 3), (0x61, 6)]
        );
        // A sara am with nothing before it is still two.
        assert_eq!(parts("\u{E33}"), [(0xE4D, 0), (0xE32, 0)]);
        assert!(parts("").is_empty());
        let started = std::time::Instant::now();
        let long = "\u{E19}\u{E49}\u{E33}".repeat(50_000);
        assert_eq!(parts(&long).len(), 200_000);
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
    }
}
