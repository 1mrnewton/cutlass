//! Khmer.
//!
//! A Khmer syllable is a consonant, the consonants written under it, each
//! after the sign that puts it there (the coeng), and its vowels and
//! marks. A vowel may be written before the consonant it is spoken after,
//! or in two parts around it, and the subscript `ro` is written before
//! the consonant it stands under. So the text is cut into syllables, those
//! parts are moved to the front of their syllable, and the font's forms
//! are applied: first the ones a syllable's own parts take, then the ones
//! that span it. The steps follow the OpenType specification for Khmer
//! and give what HarfBuzz gives.

use cutlass_font::{Font, Tag};

use super::apply::Engine;
use super::buffer::{Buffer, Glyph};
use super::pattern::{Pattern, cut, each_syllable, either, maybe, merge, number, one, repeat, seq};
use super::syllables::insert_dotted_circles;
use super::{Feature, lookups};
use crate::TextError;

/// Kinds of character. The numbers are the ones the specification's
/// categories go by.
pub(super) mod kind {
    pub(crate) const OTHER: u8 = 0;
    pub(crate) const CONSONANT: u8 = 1;
    pub(crate) const VOWEL: u8 = 2;
    pub(crate) const COENG: u8 = 4;
    pub(crate) const NON_JOINER: u8 = 5;
    pub(crate) const JOINER: u8 = 6;
    pub(crate) const PLACEHOLDER: u8 = 10;
    pub(crate) const DOTTED_CIRCLE: u8 = 11;
    pub(crate) const RO: u8 = 15;
    pub(crate) const VOWEL_ABOVE: u8 = 20;
    pub(crate) const VOWEL_BELOW: u8 = 21;
    pub(crate) const VOWEL_BEFORE: u8 = 22;
    pub(crate) const VOWEL_AFTER: u8 = 23;
    /// A shifter: it moves a consonant from one series to the other.
    pub(crate) const SHIFTER: u8 = 25;
    /// Signs above the consonant.
    pub(crate) const SIGN_ABOVE: u8 = 26;
    /// Signs after the syllable.
    pub(crate) const SIGN_AFTER: u8 = 27;
}

/// Kinds of syllable, kept in the low bits of a glyph's syllable number.
mod syllable {
    pub(super) const CONSONANT: u8 = 0;
    pub(super) const BROKEN: u8 = 1;
    pub(super) const OTHER: u8 = 2;
}

// The feature each mask bit stands for.
const PREF: u32 = 1 << 8;
const BLWF: u32 = 1 << 9;
const ABVF: u32 = 1 << 10;
const PSTF: u32 = 1 << 11;
const CFAR: u32 = 1 << 12;

/// The kind of `character`.
pub(super) fn describe(character: char) -> u8 {
    match u32::from(character) {
        0x1780..=0x1799 | 0x179B..=0x17A2 => kind::CONSONANT,
        0x179A => kind::RO,
        0x17A3..=0x17B3 => kind::VOWEL,
        0x17B6 | 0x17BF | 0x17C0 | 0x17C4 | 0x17C5 => kind::VOWEL_AFTER,
        0x17B7..=0x17BA | 0x17BE => kind::VOWEL_ABOVE,
        0x17BB..=0x17BD => kind::VOWEL_BELOW,
        0x17C1..=0x17C3 => kind::VOWEL_BEFORE,
        0x17C6 | 0x17CB | 0x17CD..=0x17D1 => kind::SIGN_ABOVE,
        0x17C7 | 0x17C8 | 0x17D3 | 0x17DD => kind::SIGN_AFTER,
        0x17C9 | 0x17CA | 0x17CC => kind::SHIFTER,
        0x17D2 => kind::COENG,
        0x200C => kind::NON_JOINER,
        0x200D => kind::JOINER,
        0x25CC => kind::DOTTED_CIRCLE,
        0x17D9
        | 0x17E0..=0x17E9
        | 0x002D
        | 0x0030..=0x0039
        | 0x00A0
        | 0x00D7
        | 0x2010..=0x2015
        | 0x2022
        | 0x25FB..=0x25FE => kind::PLACEHOLDER,
        _ => kind::OTHER,
    }
}

/// Set each glyph's kind.
pub(crate) fn describe_all(glyphs: &mut [Glyph]) {
    for glyph in glyphs {
        glyph.kind = describe(glyph.character);
    }
}

/// A vowel written in two parts, as the part before the consonant and
/// itself: Unicode does not take these apart, and a font draws the two
/// parts as two glyphs.
pub(crate) fn take_apart(character: char) -> Option<(char, Option<char>)> {
    matches!(
        character,
        '\u{17BE}' | '\u{17BF}' | '\u{17C0}' | '\u{17C4}' | '\u{17C5}'
    )
    .then_some(('\u{17C1}', Some(character)))
}

/// The kinds of syllable, each with its pattern, in the order of choice
/// between two that match the same text.
fn grammar() -> Vec<(u8, Pattern)> {
    use kind::{
        COENG, CONSONANT, DOTTED_CIRCLE, JOINER, NON_JOINER, PLACEHOLDER, RO, SHIFTER, SIGN_ABOVE,
        SIGN_AFTER, VOWEL, VOWEL_ABOVE, VOWEL_AFTER, VOWEL_BEFORE, VOWEL_BELOW,
    };
    let letter = || one(&[CONSONANT, RO, VOWEL]);
    let joiner = || one(&[JOINER, NON_JOINER]);
    // A letter and the shifter on it.
    let shifted = || {
        seq(vec![
            letter(),
            maybe(seq(vec![maybe(joiner()), one(&[SHIFTER])])),
        ])
    };
    let signs = || repeat(seq(vec![repeat(joiner()), one(&[SIGN_ABOVE])]));
    let vowels = || {
        seq(vec![
            maybe(one(&[VOWEL_BEFORE])),
            signs(),
            maybe(one(&[VOWEL_BELOW])),
            signs(),
            maybe(seq(vec![maybe(joiner()), one(&[VOWEL_ABOVE])])),
            signs(),
            maybe(one(&[VOWEL_AFTER])),
        ])
    };
    let tail = || {
        seq(vec![
            signs(),
            vowels(),
            signs(),
            maybe(seq(vec![one(&[COENG]), letter()])),
            repeat(one(&[SIGN_AFTER])),
        ])
    };
    // What may follow a letter, and what a syllable with no letter is.
    let broken = || {
        seq(vec![
            maybe(one(&[SHIFTER])),
            repeat(seq(vec![one(&[COENG]), shifted()])),
            either(vec![one(&[COENG]), tail()]),
        ])
    };
    vec![
        (
            syllable::CONSONANT,
            seq(vec![
                either(vec![shifted(), one(&[PLACEHOLDER, DOTTED_CIRCLE])]),
                broken(),
            ]),
        ),
        (syllable::BROKEN, broken()),
    ]
}

/// Number each glyph's syllable. `true` when a syllable has no letter to
/// stand on.
fn find_syllables(glyphs: &mut [Glyph]) -> bool {
    let kinds: Vec<u8> = glyphs.iter().map(|glyph| glyph.kind).collect();
    let cuts = cut(&kinds, &grammar(), syllable::OTHER);
    number(glyphs, &cuts);
    cuts.iter().any(|(_, kind)| *kind == syllable::BROKEN)
}

/// Run the substitutions for a run of Khmer.
pub(crate) fn substitute<'a>(
    font: &Font<'a>,
    mut engine: Option<Engine<'_, 'a>>,
    buffer: &mut Buffer,
    scripts: &[Tag],
) -> Result<(), TextError> {
    let table = font.gsub();
    {
        let glyphs = buffer.glyphs_mut();
        let broken = find_syllables(glyphs);
        if broken {
            insert_dotted_circles(font, glyphs, syllable::BROKEN, kind::DOTTED_CIRCLE, None);
        }
        each_syllable(glyphs, reorder);
    }
    let (Some(engine), Some(table)) = (engine.as_mut(), table) else {
        return Ok(());
    };
    // The forms of a syllable's own parts, all at once and each held to
    // its syllable.
    let masked = |tag: &[u8; 4], mask: u32| Feature::masked(tag, mask).manual_both();
    let first = [
        Feature::every(b"locl"),
        Feature::every(b"ccmp"),
        masked(b"pref", PREF),
        masked(b"blwf", BLWF),
        masked(b"abvf", ABVF),
        masked(b"pstf", PSTF),
        masked(b"cfar", CFAR),
    ];
    for mut lookup in lookups(table, scripts, &first)? {
        lookup.per_syllable = true;
        engine.apply(buffer, &lookup)?;
    }
    for glyph in buffer.glyphs_mut().iter_mut() {
        glyph.syllable = 0;
    }
    // The forms that span a syllable, with what every script shares. The
    // script asks for its contextual ligatures whatever the reader set.
    let every = |tag: &[u8; 4]| Feature::every(tag).manual_both();
    let last = [
        every(b"pres"),
        every(b"abvs"),
        every(b"blws"),
        every(b"psts"),
        Feature::every(b"rlig"),
        Feature::every(b"calt"),
        Feature::every(b"rclt"),
        Feature::every(b"clig"),
    ];
    for lookup in lookups(table, scripts, &last)? {
        engine.apply(buffer, &lookup)?;
    }
    Ok(())
}

/// Put a syllable's parts in the order they are written: a vowel that
/// goes before the consonant, and the subscript `ro` with its coeng, at
/// the front.
fn reorder(glyphs: &mut [Glyph], start: usize, end: usize) {
    if glyphs[start].syllable & 0x0F == syllable::OTHER {
        return;
    }
    // Everything after the first letter may take a form below, above or
    // behind it.
    for glyph in &mut glyphs[start + 1..end] {
        glyph.mask |= BLWF | ABVF | PSTF;
    }
    let mut coengs = 0;
    for index in start + 1..end {
        if glyphs[index].kind == kind::COENG && coengs <= 2 && index + 1 < end {
            coengs += 1;
            if glyphs[index + 1].kind == kind::RO {
                glyphs[index].mask |= PREF;
                glyphs[index + 1].mask |= PREF;
                // The coeng and the ro go to the start.
                merge(glyphs, start, index + 2);
                glyphs[start..index + 2].rotate_right(2);
                // What follows may be drawn another way after a ro that
                // was moved.
                for glyph in &mut glyphs[index + 2..end] {
                    glyph.mask |= CFAR;
                }
                coengs = 2;
            }
        } else if glyphs[index].kind == kind::VOWEL_BEFORE {
            merge(glyphs, start, index + 1);
            glyphs[start..=index].rotate_right(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{describe, find_syllables, kind, reorder, syllable, take_apart};
    use crate::shaper::buffer::Glyph;
    use crate::shaper::pattern::each_syllable;

    fn glyphs_of(text: &str) -> Vec<Glyph> {
        text.char_indices()
            .map(|(at, character)| {
                let mut glyph = Glyph::new(character, at);
                glyph.kind = describe(character);
                glyph
            })
            .collect()
    }

    /// The syllables of `text`: each one's characters and its kind.
    fn syllables(text: &str) -> Vec<(String, u8)> {
        let mut glyphs = glyphs_of(text);
        let _broken = find_syllables(&mut glyphs);
        let mut found: Vec<(String, u8, u8)> = Vec::new();
        for glyph in &glyphs {
            match found.last_mut().filter(|last| last.2 == glyph.syllable) {
                Some(last) => last.0.push(glyph.character),
                None => found.push((
                    glyph.character.to_string(),
                    glyph.syllable & 0x0F,
                    glyph.syllable,
                )),
            }
        }
        found
            .into_iter()
            .map(|(text, kind, _)| (text, kind))
            .collect()
    }

    /// `text` after its syllables are put in written order.
    fn reordered(text: &str) -> String {
        let mut glyphs = glyphs_of(text);
        let _broken = find_syllables(&mut glyphs);
        each_syllable(&mut glyphs, reorder);
        glyphs.iter().map(|glyph| glyph.character).collect()
    }

    #[test]
    fn characters_have_their_kind() {
        assert_eq!(describe('\u{1780}'), kind::CONSONANT);
        assert_eq!(describe('\u{179A}'), kind::RO);
        assert_eq!(describe('\u{17A5}'), kind::VOWEL);
        assert_eq!(describe('\u{17D2}'), kind::COENG);
        assert_eq!(describe('\u{17C1}'), kind::VOWEL_BEFORE);
        assert_eq!(describe('\u{17B6}'), kind::VOWEL_AFTER);
        assert_eq!(describe('\u{17B8}'), kind::VOWEL_ABOVE);
        assert_eq!(describe('\u{17BB}'), kind::VOWEL_BELOW);
        assert_eq!(describe('\u{17C9}'), kind::SHIFTER);
        assert_eq!(describe('\u{17C6}'), kind::SIGN_ABOVE);
        assert_eq!(describe('\u{17C7}'), kind::SIGN_AFTER);
        assert_eq!(describe('\u{17E1}'), kind::PLACEHOLDER);
        assert_eq!(describe('\u{25CC}'), kind::DOTTED_CIRCLE);
        assert_eq!(describe('\u{200C}'), kind::NON_JOINER);
        assert_eq!(describe('a'), kind::OTHER);
        assert_eq!(describe('\u{17D4}'), kind::OTHER);
        // A vowel in two parts is its left part and itself.
        assert_eq!(take_apart('\u{17BE}'), Some(('\u{17C1}', Some('\u{17BE}'))));
        assert_eq!(take_apart('\u{17C1}'), None);
        assert_eq!(take_apart('a'), None);
    }

    #[test]
    fn text_is_cut_into_syllables() {
        // khmae: kha with a subscript mo and a vowel, then ro.
        assert_eq!(
            syllables("\u{1781}\u{17D2}\u{1798}\u{17C2}\u{179A}"),
            [
                (
                    "\u{1781}\u{17D2}\u{1798}\u{17C2}".to_owned(),
                    syllable::CONSONANT
                ),
                ("\u{179A}".to_owned(), syllable::CONSONANT),
            ]
        );
        // A shifter and two vowels; a final coeng stays with its letter.
        assert_eq!(syllables("\u{1789}\u{17C9}\u{17B6}\u{17C6}").len(), 1);
        assert_eq!(syllables("\u{1780}\u{17D2}").len(), 1);
        // A vowel or a coeng with no letter before it is broken.
        assert_eq!(syllables("\u{17C1}\u{1780}")[0].1, syllable::BROKEN);
        assert_eq!(syllables("\u{17D2}\u{1780}")[0].1, syllable::BROKEN);
        // Latin and punctuation are each on their own.
        assert_eq!(
            syllables("a\u{17D4}"),
            [
                ("a".to_owned(), syllable::OTHER),
                ("\u{17D4}".to_owned(), syllable::OTHER)
            ]
        );
        assert!(syllables("").is_empty());
    }

    #[test]
    fn a_vowel_before_and_a_subscript_ro_go_to_the_front() {
        // ke: the vowel is written first.
        assert_eq!(reordered("\u{1780}\u{17C1}"), "\u{17C1}\u{1780}");
        // kro: the coeng and the ro are written first.
        assert_eq!(
            reordered("\u{1780}\u{17D2}\u{179A}"),
            "\u{17D2}\u{179A}\u{1780}"
        );
        // A subscript, a subscript ro and a vowel before: the vowel
        // first, then the ro, then the letter and its subscript.
        assert_eq!(
            reordered("\u{179F}\u{17D2}\u{178F}\u{17D2}\u{179A}\u{17C1}"),
            "\u{17C1}\u{17D2}\u{179A}\u{179F}\u{17D2}\u{178F}"
        );
        // Another syllable is left alone, and so is Latin.
        assert_eq!(reordered("a\u{1780}\u{17C1}b"), "a\u{17C1}\u{1780}b");
        // Moved glyphs share their place in the text with what they
        // passed.
        let mut glyphs = glyphs_of("\u{1780}\u{17C1}");
        let _broken = find_syllables(&mut glyphs);
        each_syllable(&mut glyphs, reorder);
        assert!(glyphs.iter().all(|glyph| glyph.cluster == 0));
    }

    #[test]
    fn a_long_run_is_cut_and_ordered_in_time() {
        let started = std::time::Instant::now();
        for unit in [
            "\u{1780}\u{17D2}\u{179A}\u{17C1}",
            "\u{17D2}",
            "\u{17C1}",
            "\u{1780}\u{17C6}",
        ] {
            let text = unit.repeat(10_000);
            assert!(!reordered(&text).is_empty());
        }
        // One letter under thousands of signs is one syllable.
        let mut heavy = String::from("\u{1780}");
        heavy.push_str(&"\u{17C6}".repeat(20_000));
        assert_eq!(syllables(&heavy).len(), 1);
        assert!(started.elapsed() < std::time::Duration::from_secs(30));
    }
}
