//! The forms letters take in Arabic and the scripts written like it.
//!
//! A letter joins the one after it, the one before it, both, or neither,
//! and the font draws it alone, first, in the middle, or last to suit.
//! Each letter's form is found here from the joining type of its
//! neighbours; marks between two letters do not come between them.
//! Syriac's alaph has three forms more: it ends a word one way after a
//! letter that does not join it, another after dalath or rish, and a
//! letter after a joined alaph takes a middle form of its own.

use super::buffer::Glyph;
use crate::unicode::{joining, props};

/// A letter's form, as the bit of its feature in a glyph's mask.
pub(crate) const ISOLATED: u32 = 1 << 2;
pub(crate) const FINAL: u32 = 1 << 3;
pub(crate) const MEDIAL: u32 = 1 << 4;
pub(crate) const INITIAL: u32 = 1 << 5;
/// Syriac's own: alaph at the end after a letter that does not join it,
/// alaph after dalath or rish, and the letter after a joined alaph.
pub(crate) const FINAL_2: u32 = 1 << 6;
pub(crate) const FINAL_3: u32 = 1 << 7;
pub(crate) const MEDIAL_2: u32 = 1 << 8;

/// Every form's bit.
const ANY_FORM: u32 = ISOLATED | FINAL | MEDIAL | INITIAL | FINAL_2 | FINAL_3 | MEDIAL_2;

/// The forms and the font feature that draws each, in the order the
/// features are applied.
pub(crate) const FORMS: [(&[u8; 4], u32); 7] = [
    (b"isol", ISOLATED),
    (b"fina", FINAL),
    (b"fin2", FINAL_2),
    (b"fin3", FINAL_3),
    (b"medi", MEDIAL),
    (b"med2", MEDIAL_2),
    (b"init", INITIAL),
];

/// How a letter joins, as a column of [`STEPS`]: not at all, only to the
/// letter after it, only to the one before it, to both; and the two
/// groups of Syriac letters that join only the one before them and have
/// forms of their own.
const NEITHER: usize = 0;
const AFTER: usize = 1;
const BEFORE: usize = 2;
const BOTH: usize = 3;
const ALAPH: usize = 4;
const DALATH_RISH: usize = 5;

/// One step: the form the letter before now takes (0 leaves it), the form
/// this letter takes for now, and the state after it.
type Step = (u32, u32, usize);

/// For each state, what a letter of each kind does. The states: the
/// letter before does not join (0); it joins only what is before it, or
/// is a lone alaph (1); it is alone so far and joins what follows (2); it
/// is joined to the letter before it and joins what follows (3); it is an
/// alaph joined to the letter before it (4); it is an alaph in one of its
/// own end forms (5); it is dalath or rish (6). This is the table every
/// shaper of these scripts uses.
const STEPS: [[Step; 6]; 7] = [
    [
        (0, 0, 0),
        (0, ISOLATED, 2),
        (0, ISOLATED, 1),
        (0, ISOLATED, 2),
        (0, ISOLATED, 1),
        (0, ISOLATED, 6),
    ],
    [
        (0, 0, 0),
        (0, ISOLATED, 2),
        (0, ISOLATED, 1),
        (0, ISOLATED, 2),
        (0, FINAL_2, 5),
        (0, ISOLATED, 6),
    ],
    [
        (0, 0, 0),
        (0, ISOLATED, 2),
        (INITIAL, FINAL, 1),
        (INITIAL, FINAL, 3),
        (INITIAL, FINAL, 4),
        (INITIAL, FINAL, 6),
    ],
    [
        (0, 0, 0),
        (0, ISOLATED, 2),
        (MEDIAL, FINAL, 1),
        (MEDIAL, FINAL, 3),
        (MEDIAL, FINAL, 4),
        (MEDIAL, FINAL, 6),
    ],
    [
        (0, 0, 0),
        (0, ISOLATED, 2),
        (MEDIAL_2, ISOLATED, 1),
        (MEDIAL_2, ISOLATED, 2),
        (MEDIAL_2, FINAL_2, 5),
        (MEDIAL_2, ISOLATED, 6),
    ],
    [
        (0, 0, 0),
        (0, ISOLATED, 2),
        (ISOLATED, ISOLATED, 1),
        (ISOLATED, ISOLATED, 2),
        (ISOLATED, FINAL_2, 5),
        (ISOLATED, ISOLATED, 6),
    ],
    [
        (0, 0, 0),
        (0, ISOLATED, 2),
        (0, ISOLATED, 1),
        (0, ISOLATED, 2),
        (0, FINAL_3, 5),
        (0, ISOLATED, 6),
    ],
];

/// Give each letter of `glyphs`, in reading order, the mask bit of its
/// form.
pub(crate) fn join(glyphs: &mut [Glyph]) {
    let mut before: Option<usize> = None;
    let mut state = 0;
    for index in 0..glyphs.len() {
        let character = glyphs[index].character;
        let kind = match (u32::from(character), props(character).joining()) {
            (_, joining::TRANSPARENT) => continue,
            (0x0710, _) => ALAPH,
            (0x0715 | 0x0716 | 0x072A | 0x072F, _) => DALATH_RISH,
            (_, joining::DUAL | joining::CAUSING) => BOTH,
            (_, joining::LEFT) => AFTER,
            (_, joining::RIGHT) => BEFORE,
            _ => NEITHER,
        };
        let (earlier, form, next) = STEPS[state][kind];
        if earlier != 0
            && let Some(glyph) = before.and_then(|before| glyphs.get_mut(before))
        {
            glyph.mask = (glyph.mask & !ANY_FORM) | earlier;
        }
        glyphs[index].mask = (glyphs[index].mask & !ANY_FORM) | form;
        before = Some(index);
        state = next;
    }
}

/// Give each Mongolian variation selector the form of the glyph before
/// it: the selector picks a variant of that letter in that form, and a
/// font's lookup for the form reads the two together.
pub(crate) fn lend_to_selectors(glyphs: &mut [Glyph]) {
    for index in 1..glyphs.len() {
        if matches!(u32::from(glyphs[index].character), 0x180B..=0x180D | 0x180F) {
            let form = glyphs[index - 1].mask & ANY_FORM;
            glyphs[index].mask = (glyphs[index].mask & !ANY_FORM) | form;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FINAL, FINAL_2, FINAL_3, INITIAL, ISOLATED, MEDIAL, MEDIAL_2, join, lend_to_selectors,
    };
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
                FINAL_2 => "fin2",
                FINAL_3 => "fin3",
                MEDIAL_2 => "med2",
                0 => "none",
                _ => "mixed",
            })
            .collect()
    }

    #[test]
    fn a_mongolian_selector_takes_the_form_of_the_glyph_before_it() {
        let mut glyphs: Vec<Glyph> = "\u{1820}\u{180B}\u{1820}\u{1885}\u{180C}\u{180F}"
            .char_indices()
            .map(|(at, character)| Glyph::new(character, at))
            .collect();
        join(&mut glyphs);
        lend_to_selectors(&mut glyphs);
        let forms: Vec<u32> = glyphs.iter().map(|glyph| glyph.mask).collect();
        // The first selector after the first letter, the second after a
        // mark that has no form, and the third after that selector.
        assert_eq!(forms, [INITIAL, INITIAL, FINAL, 0, 0, 0]);
        // A selector that starts the text has nothing to take.
        let mut alone = vec![Glyph::new('\u{180B}', 0)];
        join(&mut alone);
        lend_to_selectors(&mut alone);
        assert_eq!(alone[0].mask, 0);
        lend_to_selectors(&mut []);
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
    fn syriac_alaph_takes_its_own_forms() {
        // Alaph alone, and after a letter that joins it: beth alaph.
        assert_eq!(forms("\u{710}"), ["isol"]);
        assert_eq!(forms("\u{712}\u{710}"), ["init", "fina"]);
        // After a letter that does not join what follows it, alaph ends
        // the word in its second form: waw alaph.
        assert_eq!(forms("\u{718}\u{710}"), ["isol", "fin2"]);
        // After dalath or rish, in its third: dalath alaph, rish alaph.
        assert_eq!(forms("\u{715}\u{710}"), ["isol", "fin3"]);
        assert_eq!(forms("\u{72A}\u{710}"), ["isol", "fin3"]);
        // Sogdian zhain joins as they do and is not of their group.
        assert_eq!(forms("\u{74D}\u{710}"), ["isol", "fin2"]);
        // The letter after a joined alaph takes the second middle form:
        // beth alaph beth, and alaph itself keeps its end form.
        assert_eq!(forms("\u{712}\u{710}\u{712}"), ["init", "med2", "isol"]);
        // Two alaphs after a letter: the first is joined, the second ends
        // in its second form and makes the first the second middle form.
        assert_eq!(forms("\u{712}\u{710}\u{710}"), ["init", "med2", "fin2"]);
        // An alaph in its own end form goes back to standing alone when
        // a letter that joins nothing before it follows.
        assert_eq!(forms("\u{718}\u{710}\u{718}"), ["isol", "isol", "isol"]);
        // A mark between dalath and alaph changes nothing.
        assert_eq!(forms("\u{715}\u{732}\u{710}"), ["isol", "none", "fin3"]);
        // A long word is walked once.
        let started = std::time::Instant::now();
        let long = "\u{712}\u{710}\u{715}".repeat(50_000);
        assert_eq!(forms(&long).len(), 150_000);
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
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
