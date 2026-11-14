//! The forms letters take in Arabic and the scripts written like it.
//!
//! A letter joins the one after it, the one before it, both, or neither,
//! and the font draws it alone, first, in the middle, or last to suit.
//! Each letter's form is found here from the joining type of its
//! neighbours; marks between two letters do not come between them. The
//! extra forms Syriac gives the letter alaph are not told apart.

use super::buffer::Glyph;
use crate::unicode::{joining, props};

/// A letter's form, as the bit of its feature in a glyph's mask.
pub(crate) const ISOLATED: u32 = 1 << 2;
pub(crate) const FINAL: u32 = 1 << 3;
pub(crate) const MEDIAL: u32 = 1 << 4;
pub(crate) const INITIAL: u32 = 1 << 5;

/// The forms and the font feature that draws each.
pub(crate) const FORMS: [(&[u8; 4], u32); 4] = [
    (b"isol", ISOLATED),
    (b"fina", FINAL),
    (b"medi", MEDIAL),
    (b"init", INITIAL),
];

/// How a letter joins: not at all, only to the letter after it, only to
/// the one before it, or to both.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Joins {
    Neither,
    After,
    Before,
    Both,
}

/// Give each letter of `glyphs`, in reading order, the mask bit of its
/// form.
pub(crate) fn join(glyphs: &mut [Glyph]) {
    // The last letter seen, and whether it joins the one after it.
    let mut before: Option<(usize, bool)> = None;
    for index in 0..glyphs.len() {
        let joins = match props(glyphs[index].character).joining() {
            joining::TRANSPARENT => continue,
            joining::DUAL | joining::CAUSING => Joins::Both,
            joining::LEFT => Joins::After,
            joining::RIGHT => Joins::Before,
            _ => Joins::Neither,
        };
        if joins == Joins::Neither {
            before = None;
            continue;
        }
        let joined = before.is_some_and(|(_, onward)| onward)
            && matches!(joins, Joins::Both | Joins::Before);
        if joined && let Some((earlier, _)) = before {
            // The letter before now has one after it.
            let glyph = &mut glyphs[earlier];
            glyph.mask = if glyph.mask & FINAL != 0 {
                (glyph.mask & !FINAL) | MEDIAL
            } else {
                (glyph.mask & !ISOLATED) | INITIAL
            };
        }
        glyphs[index].mask |= if joined { FINAL } else { ISOLATED };
        before = Some((index, matches!(joins, Joins::Both | Joins::After)));
    }
}

#[cfg(test)]
mod tests {
    use super::{FINAL, INITIAL, ISOLATED, MEDIAL, join};
    use crate::shaper::buffer::Glyph;

    fn forms(text: &str) -> Vec<&'static str> {
        let mut glyphs: Vec<Glyph> = text
            .char_indices()
            .map(|(at, character)| Glyph::new(character, at))
            .collect();
        join(&mut glyphs);
        glyphs
            .iter()
            .map(|glyph| match glyph.mask {
                ISOLATED => "isol",
                FINAL => "fina",
                MEDIAL => "medi",
                INITIAL => "init",
                0 => "none",
                _ => "mixed",
            })
            .collect()
    }

    #[test]
    fn letters_take_the_form_their_neighbours_give() {
        // beh beh beh: first, middle, last.
        assert_eq!(forms("\u{628}\u{628}\u{628}"), ["init", "medi", "fina"]);
        assert_eq!(forms("\u{628}"), ["isol"]);
        // Alef joins only the letter before it: beh alef beh.
        assert_eq!(forms("\u{628}\u{627}\u{628}"), ["init", "fina", "isol"]);
        assert_eq!(forms("\u{627}\u{627}"), ["isol", "isol"]);
        // A space or a Latin letter ends the word.
        assert_eq!(
            forms("\u{628}\u{628} \u{628}"),
            ["init", "fina", "none", "isol"]
        );
        assert_eq!(forms("a\u{628}a"), ["none", "isol", "none"]);
    }

    #[test]
    fn marks_do_not_come_between_letters_and_joiners_join() {
        // A fatha between two letters leaves them joined.
        assert_eq!(forms("\u{628}\u{64E}\u{628}"), ["init", "none", "fina"]);
        // A tatweel joins on both sides.
        assert_eq!(forms("\u{628}\u{640}\u{628}"), ["init", "medi", "fina"]);
        // A zero width joiner makes a letter take its joined form alone,
        // and a non-joiner keeps two letters apart.
        assert_eq!(forms("\u{628}\u{200D}"), ["init", "fina"]);
        assert_eq!(forms("\u{628}\u{200C}\u{628}"), ["isol", "none", "isol"]);
        assert!(forms("").is_empty());
    }
}
