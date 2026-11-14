//! What the scripts written by syllable share once a run is cut.

use cutlass_font::{Font, GlyphClass};

use super::buffer::Glyph;
use crate::unicode::gc;

/// A syllable with no letter to carry its signs gets a dotted circle to
/// carry them, when the font has one. `broken` is the kind of syllable
/// that lacks its letter and `circle` the kind of character the circle
/// counts as. The circle goes at the start of the syllable, after any
/// characters of kind `after`.
pub(super) fn insert_dotted_circles(
    font: &Font<'_>,
    glyphs: &mut Vec<Glyph>,
    broken: u8,
    circle: u8,
    after: Option<u8>,
) {
    let Some(id) = font.glyph_id('\u{25CC}').ok().flatten() else {
        return;
    };
    let class = font
        .gdef()
        .filter(|gdef| gdef.has_classes())
        .map_or(GlyphClass::Base, |gdef| gdef.class(id));
    let mut index = 0;
    let mut last = 0;
    while index < glyphs.len() {
        let syllable = glyphs[index].syllable;
        if syllable == last || syllable & 0x0F != broken {
            last = syllable;
            index += 1;
            continue;
        }
        last = syllable;
        let mut at = index;
        while at < glyphs.len()
            && glyphs[at].syllable == syllable
            && after.is_some_and(|kind| glyphs[at].kind == kind)
        {
            at += 1;
        }
        let mut glyph = Glyph::new('\u{25CC}', glyphs[index].cluster);
        glyph.id = id;
        glyph.kind = circle;
        glyph.syllable = syllable;
        glyph.mask = glyphs[index].mask;
        glyph.class = class;
        glyph.category = gc::OTHER_LETTER;
        glyphs.insert(at, glyph);
        index = at + 1;
    }
}

#[cfg(test)]
mod tests {
    use super::insert_dotted_circles;
    use crate::shaper::buffer::Glyph;

    const SHAPING: &[u8] = include_bytes!("../../../font/fixtures/shaping.ttf");
    const SAMPLE: &[u8] = include_bytes!("../../../font/fixtures/sample.ttf");

    /// Glyphs of the given kinds, each a syllable number.
    fn run(parts: &[(u8, u8)]) -> Vec<Glyph> {
        parts
            .iter()
            .enumerate()
            .map(|(at, (kind, syllable))| {
                let mut glyph = Glyph::new('x', at * 3);
                glyph.kind = *kind;
                glyph.syllable = *syllable;
                glyph.mask = 5;
                glyph
            })
            .collect()
    }

    #[test]
    fn a_broken_syllable_gets_a_circle_at_its_start_or_after_its_hook() {
        // The shaping fixture has a dotted circle, glyph 33.
        let font = cutlass_font::read(SHAPING).expect("fixture");
        // Three syllables: whole, broken, broken with a hook first.
        let (whole, broken, hooked) = (0x10, 0x21, 0x31);
        let mut glyphs = run(&[
            (1, whole),
            (2, broken),
            (2, broken),
            (9, hooked),
            (2, hooked),
        ]);
        insert_dotted_circles(&font, &mut glyphs, 1, 7, Some(9));
        let seen: Vec<(u16, u8, u8)> = glyphs
            .iter()
            .map(|glyph| (glyph.id, glyph.kind, glyph.syllable))
            .collect();
        assert_eq!(
            seen,
            [
                (0, 1, whole),
                (33, 7, broken),
                (0, 2, broken),
                (0, 2, broken),
                (0, 9, hooked),
                (33, 7, hooked),
                (0, 2, hooked),
            ]
        );
        // The circle stands where its syllable starts in the text, with
        // the masks of the glyph it is put before.
        assert_eq!((glyphs[1].cluster, glyphs[1].mask), (3, 5));
        assert_eq!(glyphs[5].cluster, 9);
        // A font without the circle changes nothing.
        let bare = cutlass_font::read(SAMPLE).expect("fixture");
        let mut glyphs = run(&[(2, broken)]);
        insert_dotted_circles(&bare, &mut glyphs, 1, 7, None);
        assert_eq!(glyphs.len(), 1);
        // Many broken syllables, each numbered apart, are each given one.
        let font = cutlass_font::read(SHAPING).expect("fixture");
        let parts: Vec<(u8, u8)> = (0..3_000_u32)
            .map(|at| (2, (u8::try_from(at % 15).expect("count") + 1) << 4 | 1))
            .collect();
        let mut glyphs = run(&parts);
        insert_dotted_circles(&font, &mut glyphs, 1, 7, None);
        assert_eq!(glyphs.len(), 6_000);
    }
}
