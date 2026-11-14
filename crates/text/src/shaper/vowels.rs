//! Dotted circles between a vowel and a sign that would pass for another
//! vowel.
//!
//! In some scripts a vowel letter with a vowel sign after it looks just
//! like another vowel letter, so one word can be typed two ways that read
//! the same. Each script's specification lists those pairs, and a shaper
//! shows them apart with a dotted circle between the two, as HarfBuzz
//! does.

use super::buffer::Glyph;
use super::described;
use crate::script::Script;

/// A script, a letter of it, and the signs kept apart from that letter.
type Row = ([u8; 4], u32, &'static [u32]);

/// The pairs, in the order of their letters.
const PAIRS: &[Row] = &[
    (
        *b"Deva",
        0x0905,
        &[
            0x093A, 0x093B, 0x093E, 0x0945, 0x0946, 0x0949, 0x094A, 0x094B, 0x094C, 0x094F, 0x0956,
            0x0957,
        ],
    ),
    (*b"Deva", 0x0906, &[0x093A, 0x0945, 0x0946, 0x0947, 0x0948]),
    (*b"Deva", 0x0909, &[0x0941]),
    (*b"Deva", 0x090F, &[0x0945, 0x0946, 0x0947]),
    (*b"Beng", 0x0985, &[0x09BE]),
    (*b"Beng", 0x098B, &[0x09C3]),
    (*b"Beng", 0x098C, &[0x09E2]),
    (*b"Guru", 0x0A05, &[0x0A3E, 0x0A48, 0x0A4C]),
    (*b"Guru", 0x0A72, &[0x0A3F, 0x0A40, 0x0A47]),
    (*b"Guru", 0x0A73, &[0x0A41, 0x0A42, 0x0A4B]),
    (
        *b"Gujr",
        0x0A85,
        &[0x0ABE, 0x0AC5, 0x0AC7, 0x0AC8, 0x0AC9, 0x0ACB, 0x0ACC],
    ),
    (*b"Gujr", 0x0AC5, &[0x0ABE]),
    (*b"Orya", 0x0B05, &[0x0B3E]),
    (*b"Orya", 0x0B0F, &[0x0B57]),
    (*b"Orya", 0x0B13, &[0x0B57]),
    (*b"Taml", 0x0B85, &[0x0BC2]),
    (*b"Telu", 0x0C12, &[0x0C4C, 0x0C55]),
    (*b"Telu", 0x0C3F, &[0x0C55]),
    (*b"Telu", 0x0C46, &[0x0C55]),
    (*b"Telu", 0x0C4A, &[0x0C55]),
    (*b"Knda", 0x0C89, &[0x0CBE]),
    (*b"Knda", 0x0C8B, &[0x0CBE]),
    (*b"Knda", 0x0C92, &[0x0CCC]),
    (*b"Mlym", 0x0D07, &[0x0D57]),
    (*b"Mlym", 0x0D09, &[0x0D57]),
    (*b"Mlym", 0x0D0E, &[0x0D46]),
    (*b"Mlym", 0x0D12, &[0x0D3E, 0x0D57]),
    (*b"Sinh", 0x0D85, &[0x0DCF, 0x0DD0, 0x0DD1]),
    (*b"Sinh", 0x0D8B, &[0x0DDF]),
    (*b"Sinh", 0x0D8D, &[0x0DD8]),
    (*b"Sinh", 0x0D8F, &[0x0DDF]),
    (
        *b"Sinh",
        0x0D91,
        &[0x0DCA, 0x0DD9, 0x0DDA, 0x0DDC, 0x0DDD, 0x0DDE],
    ),
    (*b"Sinh", 0x0D94, &[0x0DDF]),
    (*b"Brah", 0x11005, &[0x11038]),
    (*b"Brah", 0x1100B, &[0x1103E]),
    (*b"Brah", 0x1100F, &[0x11042]),
    (*b"Khoj", 0x11200, &[0x1122C, 0x11231, 0x11233]),
    (*b"Khoj", 0x11206, &[0x1122C]),
    (*b"Khoj", 0x1122C, &[0x11230, 0x11231]),
    (*b"Khoj", 0x11240, &[0x1122E]),
    (
        *b"Sind",
        0x112B0,
        &[0x112E0, 0x112E5, 0x112E6, 0x112E7, 0x112E8],
    ),
    (*b"Tirh", 0x11481, &[0x114B0]),
    (*b"Tirh", 0x1148B, &[0x114BA]),
    (*b"Tirh", 0x1148D, &[0x114BA]),
    (*b"Tirh", 0x114AA, &[0x114B5, 0x114B6]),
    (*b"Modi", 0x11600, &[0x11639, 0x1163A]),
    (*b"Modi", 0x11601, &[0x11639, 0x1163A]),
    (*b"Takr", 0x11680, &[0x116AD, 0x116B4, 0x116B5]),
    (*b"Takr", 0x11686, &[0x116B2]),
];

/// Whether `second` is kept apart from `first` in `script`.
fn apart(script: Script, first: char, second: char) -> bool {
    let first = u32::from(first);
    PAIRS
        .binary_search_by_key(&first, |(_, letter, _)| *letter)
        .ok()
        .and_then(|at| PAIRS.get(at))
        .is_some_and(|(code, _, signs)| *code == script.0 && signs.contains(&u32::from(second)))
}

/// Where the glyph at `at` is in the text once it is counted with the
/// letter it goes with.
fn place(glyphs: &[Glyph], at: usize) -> usize {
    let start = (0..=at)
        .rev()
        .find(|index| glyphs.get(*index).is_some_and(|glyph| !glyph.continues))
        .unwrap_or(0);
    let end = (at + 1..glyphs.len())
        .find(|index| glyphs.get(*index).is_some_and(|glyph| !glyph.continues))
        .unwrap_or(glyphs.len());
    glyphs
        .get(start..end)
        .and_then(|group| group.iter().map(|glyph| glyph.cluster).min())
        .unwrap_or(0)
}

/// Put a dotted circle between each pair `script` keeps apart. The
/// circle is counted with the sign after it.
pub(super) fn separate(script: Script, glyphs: Vec<Glyph>) -> Vec<Glyph> {
    let character = |at: usize| glyphs.get(at).map(|glyph| glyph.character);
    // The glyphs a circle goes before.
    let mut before: Vec<(usize, usize)> = Vec::new();
    let mut index = 0;
    while index + 1 < glyphs.len() {
        let (Some(first), Some(second)) = (character(index), character(index + 1)) else {
            break;
        };
        if apart(script, first, second) {
            before.push((index + 1, place(&glyphs, index + 1)));
            index += 2;
        } else if script.0 == *b"Deva"
            && first == '\u{930}'
            && second == '\u{94D}'
            && character(index + 2) == Some('\u{907}')
        {
            // The vowel i after a hooked ra would pass for the vowel ii.
            before.push((index + 2, place(&glyphs, index + 2)));
            index += 3;
        } else {
            index += 1;
        }
    }
    if before.is_empty() {
        return glyphs;
    }
    let mut out = Vec::with_capacity(glyphs.len() + before.len());
    let mut places = before.into_iter().peekable();
    for (index, glyph) in glyphs.into_iter().enumerate() {
        if let Some((_, cluster)) = places.next_if(|(at, _)| *at == index) {
            out.push(described('\u{25CC}', cluster));
        }
        out.push(glyph);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{PAIRS, separate};
    use crate::script::Script;
    use crate::shaper::buffer::Glyph;
    use crate::shaper::described;

    fn glyphs_of(text: &str) -> Vec<Glyph> {
        let mut glyphs: Vec<Glyph> = text
            .char_indices()
            .map(|(at, character)| described(character, at))
            .collect();
        crate::shaper::mark_continuations(&mut glyphs);
        glyphs
    }

    fn separated(script: &[u8; 4], text: &str) -> String {
        separate(Script(*script), glyphs_of(text))
            .iter()
            .map(|glyph| glyph.character)
            .collect()
    }

    #[test]
    fn a_vowel_and_a_sign_that_pass_for_another_are_shown_apart() {
        // Devanagari a with the sign aa would pass for the letter aa.
        assert_eq!(
            separated(b"Deva", "\u{905}\u{93E}"),
            "\u{905}\u{25CC}\u{93E}"
        );
        // Sinhala, Khojki, Modi and Takri pairs.
        assert_eq!(
            separated(b"Sinh", "\u{D85}\u{DCF}"),
            "\u{D85}\u{25CC}\u{DCF}"
        );
        assert_eq!(
            separated(b"Khoj", "\u{11200}\u{1122C}"),
            "\u{11200}\u{25CC}\u{1122C}"
        );
        assert_eq!(
            separated(b"Modi", "\u{11601}\u{1163A}"),
            "\u{11601}\u{25CC}\u{1163A}"
        );
        assert_eq!(
            separated(b"Takr", "\u{11686}\u{116B2}"),
            "\u{11686}\u{25CC}\u{116B2}"
        );
        // Every pair in a word, and the letters between left alone.
        assert_eq!(
            separated(b"Deva", "\u{915}\u{905}\u{93E}\u{915}\u{93E}\u{90F}\u{947}"),
            "\u{915}\u{905}\u{25CC}\u{93E}\u{915}\u{93E}\u{90F}\u{25CC}\u{947}"
        );
        // The vowel i after ra and a virama.
        assert_eq!(
            separated(b"Deva", "\u{930}\u{94D}\u{907}"),
            "\u{930}\u{94D}\u{25CC}\u{907}"
        );
    }

    #[test]
    fn other_pairs_and_other_scripts_are_left_alone() {
        assert_eq!(separated(b"Deva", "\u{905}\u{93F}"), "\u{905}\u{93F}");
        assert_eq!(
            separated(b"Deva", "\u{930}\u{94D}\u{915}"),
            "\u{930}\u{94D}\u{915}"
        );
        // A pair of one script in a run of another is not that script's.
        assert_eq!(separated(b"Beng", "\u{905}\u{93E}"), "\u{905}\u{93E}");
        assert_eq!(separated(b"Java", "\u{A984}\u{A9B4}"), "\u{A984}\u{A9B4}");
        assert_eq!(separated(b"Deva", ""), "");
        assert_eq!(separated(b"Deva", "\u{905}"), "\u{905}");
        assert_eq!(separated(b"Deva", "\u{93E}"), "\u{93E}");
    }

    #[test]
    fn the_circle_is_counted_with_the_sign_and_its_letter() {
        // In the text the circle is where its sign is, which is where
        // the letter before the sign is.
        let glyphs = separate(Script(*b"Deva"), glyphs_of("\u{915}\u{905}\u{93E}"));
        let clusters: Vec<usize> = glyphs.iter().map(|glyph| glyph.cluster).collect();
        assert_eq!(clusters, [0, 3, 3, 6]);
        assert!(!glyphs[2].continues);
        // After ra and a virama it is where the vowel is.
        let glyphs = separate(Script(*b"Deva"), glyphs_of("\u{930}\u{94D}\u{907}"));
        let clusters: Vec<usize> = glyphs.iter().map(|glyph| glyph.cluster).collect();
        assert_eq!(clusters, [0, 3, 6, 6]);
    }

    #[test]
    fn the_pairs_are_in_order_and_a_long_run_is_done_in_time() {
        assert!(PAIRS.windows(2).all(|pair| pair[0].1 < pair[1].1));
        let started = std::time::Instant::now();
        let text = "\u{905}\u{93E}".repeat(100_000);
        assert_eq!(separated(b"Deva", &text).chars().count(), 300_000);
        let text = "\u{930}\u{94D}".repeat(100_000);
        assert_eq!(separated(b"Deva", &text).chars().count(), 200_000);
        assert!(started.elapsed() < std::time::Duration::from_secs(10));
    }
}
