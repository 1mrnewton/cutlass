//! Myanmar, and the other languages written in its script.
//!
//! A syllable is a consonant, the consonants stacked under it, its medial
//! consonants, its vowels and its tones. The medial ra and the vowel e are
//! written before the consonant they are spoken after, and a ra with an
//! asat and a virama at the start of a syllable, the kinzi, is drawn over
//! the consonant that follows. So the text is cut into syllables, each
//! one's parts are sorted into the order they are written, and the font's
//! forms are applied a feature at a time. The steps follow the OpenType
//! specification for Myanmar and give what HarfBuzz gives.

use cutlass_font::{Font, Tag};

use super::apply::Engine;
use super::buffer::{Buffer, Glyph};
use super::pattern::{
    Pattern, cut, each_syllable, either, maybe, number, one, repeat, seq, sort_by,
};
use super::syllables::insert_dotted_circles;
use super::{Feature, lookups};
use crate::TextError;

/// Kinds of character. The numbers are the ones the specification's
/// categories go by.
pub(super) mod kind {
    pub(crate) const OTHER: u8 = 0;
    pub(crate) const CONSONANT: u8 = 1;
    pub(crate) const VOWEL: u8 = 2;
    pub(crate) const DOT_BELOW: u8 = 3;
    pub(crate) const VIRAMA: u8 = 4;
    pub(crate) const NON_JOINER: u8 = 5;
    pub(crate) const JOINER: u8 = 6;
    /// The visarga and the Shan tones.
    pub(crate) const TONE: u8 = 8;
    pub(crate) const ANUSVARA: u8 = 9;
    pub(crate) const PLACEHOLDER: u8 = 10;
    pub(crate) const DOTTED_CIRCLE: u8 = 11;
    pub(crate) const RA: u8 = 15;
    pub(crate) const MEDIAL: u8 = 16;
    /// A consonant with the sign that stacks it, as one character.
    pub(crate) const STACKED: u8 = 18;
    pub(crate) const VOWEL_ABOVE: u8 = 20;
    pub(crate) const VOWEL_BELOW: u8 = 21;
    pub(crate) const VOWEL_BEFORE: u8 = 22;
    pub(crate) const VOWEL_AFTER: u8 = 23;
    pub(crate) const ASAT: u8 = 32;
    pub(crate) const MEDIAL_HA: u8 = 35;
    pub(crate) const MEDIAL_RA: u8 = 36;
    pub(crate) const MEDIAL_WA: u8 = 37;
    pub(crate) const MEDIAL_YA: u8 = 38;
    /// The tones of Pwo Karen and others.
    pub(crate) const OTHER_TONE: u8 = 39;
    pub(crate) const SELECTOR: u8 = 40;
    pub(crate) const MEDIAL_LA: u8 = 41;
    /// A tone written as a raised digit after the syllable.
    pub(crate) const TONE_AFTER: u8 = 57;
}

/// Where in its syllable a character goes; sorting by this is the
/// reordering.
mod place {
    pub(super) const BEFORE_VOWEL: u8 = 2;
    pub(super) const BEFORE_CONSONANT: u8 = 3;
    pub(super) const BASE: u8 = 4;
    pub(super) const AFTER_MAIN: u8 = 5;
    pub(super) const BEFORE_BELOW: u8 = 7;
    pub(super) const BELOW: u8 = 8;
    pub(super) const AFTER_BELOW: u8 = 9;
}

/// Kinds of syllable, kept in the low bits of a glyph's syllable number.
mod syllable {
    pub(super) const CONSONANT: u8 = 0;
    pub(super) const BROKEN: u8 = 2;
    pub(super) const OTHER: u8 = 3;
}

/// The kind of `character`.
pub(super) fn describe(character: char) -> u8 {
    match u32::from(character) {
        0x1004 | 0x101B | 0x105A => kind::RA,
        0x1000..=0x1020
        | 0x103F
        | 0x104E
        | 0x1050
        | 0x1051
        | 0x105B..=0x105D
        | 0x1061
        | 0x1065
        | 0x1066
        | 0x106E..=0x1070
        | 0x1075..=0x1081
        | 0x108E
        | 0xA9E0..=0xA9E4
        | 0xA9E7..=0xA9EF
        | 0xA9FA..=0xA9FE
        | 0xAA60..=0xAA6F
        | 0xAA71..=0xAA73
        | 0xAA7A
        | 0xAA7E
        | 0xAA7F => kind::CONSONANT,
        0x1021..=0x102A | 0x1052..=0x1055 => kind::VOWEL,
        0x102B | 0x102C | 0x1056 | 0x1057 | 0x1062 | 0x1067 | 0x1068 | 0x1083 => kind::VOWEL_AFTER,
        0x102D | 0x102E | 0x1033..=0x1035 | 0x1071..=0x1074 | 0x1085 | 0x1086 | 0x109D | 0xA9E5 => {
            kind::VOWEL_ABOVE
        }
        0x102F | 0x1030 | 0x1058 | 0x1059 => kind::VOWEL_BELOW,
        0x1031 | 0x1084 => kind::VOWEL_BEFORE,
        0x1032 | 0x1036 => kind::ANUSVARA,
        0x1037 | 0xAA7C | 0xAA7D => kind::DOT_BELOW,
        0x1038 | 0x1087..=0x108D | 0x108F | 0x109A..=0x109C => kind::TONE,
        0x1039 => kind::VIRAMA,
        0x103A => kind::ASAT,
        0x103B | 0x105E | 0x105F => kind::MEDIAL_YA,
        0x103C => kind::MEDIAL_RA,
        0x103D | 0x1082 => kind::MEDIAL_WA,
        0x103E => kind::MEDIAL_HA,
        0x1060 => kind::MEDIAL_LA,
        0x1063 | 0x1064 | 0x1069..=0x106D | 0xAA7B => kind::OTHER_TONE,
        0x1040..=0x104B
        | 0x1090..=0x1099
        | 0xA9F0..=0xA9F9
        | 0xAA74..=0xAA76
        | 0x002D
        | 0x0030..=0x0039
        | 0x00A0
        | 0x00D7
        | 0x2010..=0x2015
        | 0x2022
        | 0x25FB..=0x25FE => kind::PLACEHOLDER,
        0x00B2 | 0x00B3 | 0x2074 | 0x2082..=0x2084 => kind::TONE_AFTER,
        0x200C => kind::NON_JOINER,
        0x200D => kind::JOINER,
        0x25CC => kind::DOTTED_CIRCLE,
        0xFE00..=0xFE0F => kind::SELECTOR,
        _ => kind::OTHER,
    }
}

/// Set each glyph's kind.
pub(crate) fn describe_all(glyphs: &mut [Glyph]) {
    for glyph in glyphs {
        glyph.kind = describe(glyph.character);
    }
}

/// The kinds of syllable, each with its pattern, in the order of choice
/// between two that match the same text.
fn grammar() -> Vec<(u8, Pattern)> {
    use kind::{
        ANUSVARA, ASAT, CONSONANT, DOT_BELOW, DOTTED_CIRCLE, JOINER, MEDIAL_HA, MEDIAL_LA,
        MEDIAL_RA, MEDIAL_WA, MEDIAL_YA, NON_JOINER, OTHER_TONE, PLACEHOLDER, RA, SELECTOR,
        STACKED, TONE, TONE_AFTER, VIRAMA, VOWEL, VOWEL_ABOVE, VOWEL_AFTER, VOWEL_BEFORE,
        VOWEL_BELOW,
    };
    let joiner = || one(&[JOINER, NON_JOINER]);
    let kinzi = || seq(vec![one(&[RA]), one(&[ASAT]), one(&[VIRAMA])]);
    let selector = || maybe(one(&[SELECTOR]));
    let dot = || maybe(seq(vec![one(&[DOT_BELOW]), maybe(one(&[ASAT]))]));
    let medials = || {
        seq(vec![
            maybe(one(&[MEDIAL_YA])),
            maybe(one(&[ASAT])),
            maybe(one(&[MEDIAL_RA])),
            maybe(seq(vec![
                either(vec![
                    seq(vec![
                        one(&[MEDIAL_WA]),
                        maybe(one(&[MEDIAL_HA])),
                        maybe(one(&[MEDIAL_LA])),
                    ]),
                    seq(vec![one(&[MEDIAL_HA]), maybe(one(&[MEDIAL_LA]))]),
                    one(&[MEDIAL_LA]),
                ]),
                maybe(one(&[ASAT])),
            ])),
        ])
    };
    let main_vowels = || {
        seq(vec![
            repeat(seq(vec![one(&[VOWEL_BEFORE]), selector()])),
            repeat(one(&[VOWEL_ABOVE])),
            repeat(one(&[VOWEL_BELOW])),
            repeat(one(&[ANUSVARA])),
            dot(),
        ])
    };
    let post_vowels = || {
        seq(vec![
            one(&[VOWEL_AFTER]),
            maybe(one(&[MEDIAL_HA])),
            maybe(one(&[MEDIAL_LA])),
            repeat(one(&[ASAT])),
            repeat(one(&[VOWEL_ABOVE])),
            repeat(one(&[ANUSVARA])),
            dot(),
        ])
    };
    let tones = || {
        either(vec![
            one(&[TONE, TONE_AFTER]),
            seq(vec![
                one(&[OTHER_TONE]),
                repeat(one(&[ANUSVARA])),
                maybe(one(&[DOT_BELOW])),
                maybe(one(&[ASAT])),
            ]),
        ])
    };
    let complex_tail = || {
        seq(vec![
            repeat(one(&[ASAT])),
            medials(),
            main_vowels(),
            repeat(post_vowels()),
            repeat(tones()),
            maybe(joiner()),
        ])
    };
    let tail = || {
        seq(vec![
            repeat(seq(vec![
                one(&[VIRAMA]),
                one(&[CONSONANT, RA, VOWEL]),
                selector(),
            ])),
            either(vec![one(&[VIRAMA]), complex_tail()]),
        ])
    };
    vec![
        (
            syllable::CONSONANT,
            seq(vec![
                maybe(either(vec![kinzi(), one(&[STACKED])])),
                one(&[CONSONANT, RA, VOWEL, PLACEHOLDER, DOTTED_CIRCLE]),
                selector(),
                tail(),
            ]),
        ),
        // A joiner or a raised digit alone is not a syllable.
        (syllable::OTHER, one(&[JOINER, NON_JOINER, TONE_AFTER])),
        (
            syllable::BROKEN,
            seq(vec![maybe(kinzi()), selector(), tail()]),
        ),
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

/// Run the substitutions for a run in the Myanmar script. `ligatures`
/// makes the ones a reader can do without.
pub(crate) fn substitute<'a>(
    font: &Font<'a>,
    mut engine: Option<Engine<'_, 'a>>,
    buffer: &mut Buffer,
    scripts: &[Tag],
    ligatures: bool,
) -> Result<(), TextError> {
    let table = font.gsub();
    let broken = find_syllables(buffer.glyphs_mut());
    let mut stage = |buffer: &mut Buffer, features: &[Feature]| -> Result<(), TextError> {
        let (Some(engine), Some(table)) = (engine.as_mut(), table) else {
            return Ok(());
        };
        for mut lookup in lookups(table, scripts, features)? {
            lookup.per_syllable = true;
            engine.apply(buffer, &lookup)?;
        }
        Ok(())
    };
    stage(buffer, &[Feature::every(b"locl"), Feature::every(b"ccmp")])?;
    {
        let glyphs = buffer.glyphs_mut();
        if broken {
            insert_dotted_circles(font, glyphs, syllable::BROKEN, kind::DOTTED_CIRCLE, None);
        }
        each_syllable(glyphs, reorder);
    }
    // Each form on its own, so one sees what the one before it made.
    for tag in [b"rphf", b"pref", b"blwf", b"pstf"] {
        stage(buffer, &[Feature::every(tag).manual()])?;
    }
    for glyph in buffer.glyphs_mut().iter_mut() {
        glyph.syllable = 0;
    }
    let (Some(engine), Some(table)) = (engine.as_mut(), table) else {
        return Ok(());
    };
    // The forms that span a syllable, with what every script shares.
    let mut last = vec![
        Feature::every(b"pres").manual(),
        Feature::every(b"abvs").manual(),
        Feature::every(b"blws").manual(),
        Feature::every(b"psts").manual(),
        Feature::every(b"rlig"),
        Feature::every(b"calt"),
        Feature::every(b"rclt"),
    ];
    if ligatures {
        last.extend([Feature::every(b"clig"), Feature::every(b"liga")]);
    }
    for lookup in lookups(table, scripts, &last)? {
        engine.apply(buffer, &lookup)?;
    }
    Ok(())
}

/// Whether a glyph is a letter a syllable can stand on. A glyph made of
/// several no longer counts as what one of them was.
fn is_consonant(glyph: &Glyph) -> bool {
    !glyph.ligated
        && matches!(
            glyph.kind,
            kind::CONSONANT
                | kind::STACKED
                | kind::RA
                | kind::MEDIAL
                | kind::VOWEL
                | kind::PLACEHOLDER
                | kind::DOTTED_CIRCLE
        )
}

/// Put a syllable's parts in the order they are written: the medial ra
/// and the vowel e before the consonant, the kinzi after it, and the
/// marks below before the ones that follow them.
fn reorder(glyphs: &mut [Glyph], start: usize, end: usize) {
    let kind_of = glyphs[start].syllable & 0x0F;
    if kind_of != syllable::CONSONANT && kind_of != syllable::BROKEN {
        return;
    }
    let kinzi = start + 3 <= end
        && glyphs[start].kind == kind::RA
        && glyphs[start + 1].kind == kind::ASAT
        && glyphs[start + 2].kind == kind::VIRAMA;
    let limit = if kinzi { start + 3 } else { start };
    // The main consonant is the first after the kinzi.
    let base = (limit..end)
        .find(|index| is_consonant(&glyphs[*index]))
        .unwrap_or(start);
    let mut index = start;
    if kinzi {
        for glyph in &mut glyphs[start..start + 3] {
            glyph.place = place::AFTER_MAIN;
        }
        index = start + 3;
    }
    while index < base {
        glyphs[index].place = place::BEFORE_CONSONANT;
        index += 1;
    }
    if index < end {
        glyphs[index].place = place::BASE;
        index += 1;
    }
    let mut after = place::AFTER_MAIN;
    for index in index..end {
        let kind = glyphs[index].kind;
        glyphs[index].place = if kind == kind::MEDIAL_RA {
            place::BEFORE_CONSONANT
        } else if kind == kind::VOWEL_BEFORE {
            place::BEFORE_VOWEL
        } else if kind == kind::SELECTOR {
            // A selector stays with the character it follows.
            glyphs[index - 1].place
        } else if after == place::AFTER_MAIN && kind == kind::VOWEL_BELOW {
            after = place::BELOW;
            after
        } else if after == place::BELOW && kind == kind::ANUSVARA {
            // An anusvara goes before the vowel below it follows.
            place::BEFORE_BELOW
        } else if after == place::BELOW && kind == kind::VOWEL_BELOW {
            after
        } else if after == place::BELOW {
            after = place::AFTER_BELOW;
            after
        } else {
            after
        };
    }
    sort_by(glyphs, start, end, |glyph| glyph.place);
    // Several vowels written before the consonant come out in the order
    // they were typed, each with its selector after it.
    let before: Vec<usize> = (start..end)
        .filter(|index| glyphs[*index].place == place::BEFORE_VOWEL)
        .collect();
    if let (Some(first), Some(last)) = (before.first().copied(), before.last().copied())
        && first < last
    {
        glyphs[first..=last].reverse();
        let mut from = first;
        for index in first..=last {
            if glyphs[index].kind == kind::VOWEL_BEFORE {
                glyphs[from..=index].reverse();
                from = index + 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{describe, find_syllables, kind, reorder, syllable};
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
        assert_eq!(describe('\u{1000}'), kind::CONSONANT);
        assert_eq!(describe('\u{1004}'), kind::RA);
        assert_eq!(describe('\u{101B}'), kind::RA);
        assert_eq!(describe('\u{1021}'), kind::VOWEL);
        assert_eq!(describe('\u{1031}'), kind::VOWEL_BEFORE);
        assert_eq!(describe('\u{102C}'), kind::VOWEL_AFTER);
        assert_eq!(describe('\u{102D}'), kind::VOWEL_ABOVE);
        assert_eq!(describe('\u{102F}'), kind::VOWEL_BELOW);
        assert_eq!(describe('\u{1036}'), kind::ANUSVARA);
        assert_eq!(describe('\u{1037}'), kind::DOT_BELOW);
        assert_eq!(describe('\u{1038}'), kind::TONE);
        assert_eq!(describe('\u{1039}'), kind::VIRAMA);
        assert_eq!(describe('\u{103A}'), kind::ASAT);
        assert_eq!(describe('\u{103B}'), kind::MEDIAL_YA);
        assert_eq!(describe('\u{103C}'), kind::MEDIAL_RA);
        assert_eq!(describe('\u{103D}'), kind::MEDIAL_WA);
        assert_eq!(describe('\u{103E}'), kind::MEDIAL_HA);
        assert_eq!(describe('\u{1060}'), kind::MEDIAL_LA);
        assert_eq!(describe('\u{1063}'), kind::OTHER_TONE);
        assert_eq!(describe('\u{1041}'), kind::PLACEHOLDER);
        assert_eq!(describe('\u{FE00}'), kind::SELECTOR);
        assert_eq!(describe('\u{B2}'), kind::TONE_AFTER);
        assert_eq!(describe('\u{104C}'), kind::OTHER);
        assert_eq!(describe('a'), kind::OTHER);
    }

    #[test]
    fn text_is_cut_into_syllables() {
        // myanma: ma with medial ra, na, ma with aa.
        assert_eq!(
            syllables("\u{1019}\u{103C}\u{1014}\u{103A}\u{1019}\u{102C}"),
            [
                ("\u{1019}\u{103C}".to_owned(), syllable::CONSONANT),
                ("\u{1014}\u{103A}".to_owned(), syllable::CONSONANT),
                ("\u{1019}\u{102C}".to_owned(), syllable::CONSONANT),
            ]
        );
        // A kinzi belongs to the consonant after it, and a stacked
        // consonant to the one above it.
        assert_eq!(
            syllables("\u{1004}\u{103A}\u{1039}\u{1002}\u{102B}").len(),
            1
        );
        assert_eq!(syllables("\u{1000}\u{1039}\u{1000}\u{102C}").len(), 1);
        // A vowel with no consonant is broken, and so is a lone virama.
        assert_eq!(syllables("\u{1031}")[0].1, syllable::BROKEN);
        assert_eq!(syllables("\u{1039}")[0].1, syllable::BROKEN);
        // A joiner alone is no syllable; two tones with no letter are a
        // broken one.
        assert_eq!(syllables("\u{200D}")[0].1, syllable::OTHER);
        assert_eq!(syllables("\u{B2}")[0].1, syllable::OTHER);
        assert_eq!(
            syllables("\u{1038}\u{1038}"),
            [("\u{1038}\u{1038}".to_owned(), syllable::BROKEN)]
        );
        assert_eq!(syllables("a")[0].1, syllable::OTHER);
        assert!(syllables("").is_empty());
    }

    #[test]
    fn parts_written_before_the_consonant_are_moved_there() {
        // The vowel e, and the medial ra, before their consonant.
        assert_eq!(reordered("\u{1000}\u{1031}"), "\u{1031}\u{1000}");
        assert_eq!(reordered("\u{1000}\u{103C}"), "\u{103C}\u{1000}");
        assert_eq!(
            reordered("\u{1000}\u{103C}\u{1031}"),
            "\u{1031}\u{103C}\u{1000}"
        );
        // The kinzi goes after the consonant it is drawn over.
        assert_eq!(
            reordered("\u{1004}\u{103A}\u{1039}\u{1002}\u{102B}"),
            "\u{1002}\u{1004}\u{103A}\u{1039}\u{102B}"
        );
        // An anusvara typed after a vowel below goes before it.
        assert_eq!(
            reordered("\u{1000}\u{102F}\u{1036}"),
            "\u{1000}\u{1036}\u{102F}"
        );
        // A stacked consonant and its vowel stay in place.
        assert_eq!(
            reordered("\u{1000}\u{1039}\u{1001}\u{102C}"),
            "\u{1000}\u{1039}\u{1001}\u{102C}"
        );
        // Two vowels before: the later one first, each with its
        // selector after it.
        assert_eq!(
            reordered("\u{1000}\u{1031}\u{FE00}\u{1084}"),
            "\u{1084}\u{1031}\u{FE00}\u{1000}"
        );
        assert_eq!(reordered("a\u{1000}\u{1031}b"), "a\u{1031}\u{1000}b");
    }

    #[test]
    fn a_long_run_is_cut_and_ordered_in_time() {
        let started = std::time::Instant::now();
        for unit in [
            "\u{1000}\u{103C}\u{1031}",
            "\u{1039}",
            "\u{1031}",
            "\u{1038}",
        ] {
            let text = unit.repeat(10_000);
            assert!(!reordered(&text).is_empty());
        }
        // One consonant under thousands of tones is one syllable.
        let mut heavy = String::from("\u{1000}");
        heavy.push_str(&"\u{1038}".repeat(20_000));
        assert_eq!(syllables(&heavy).len(), 1);
        assert_eq!(reordered(&heavy).chars().count(), 20_001);
        assert!(started.elapsed() < std::time::Duration::from_secs(30));
    }
}
