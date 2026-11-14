//! A string and its fonts become glyph ids, advances, and offsets.
//!
//! This step does not draw. The string is cut into paragraphs at each
//! character that ends a line; such a character is a break, not a glyph.
//! Each paragraph's characters get their direction from the Unicode
//! bidirectional algorithm, and its glyphs stay in the order the text is
//! stored: layout turns each line into the order it is seen. Advances and
//! kerning come from the font. A character with no glyph is recorded as
//! missing and uses glyph 0, whose outline is the font's missing glyph;
//! one that is never drawn, such as a joiner or a direction mark, makes no
//! glyph when the face has none for it. A run may take each character from
//! one of several faces; kerning only joins letters of the same face.

use cutlass_font::Font;

use crate::TextError;
use crate::bidi::{Levels, resolve};
use crate::linebreak::ends_line;
use crate::unicode::{mirror, props};

/// Characters in one `shape` call. A longer string is refused.
pub const MAX_CHARS: usize = 65_536;

/// One glyph, in font design units. Y grows upward.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapedGlyph {
    /// Byte index, in the source string, of the first character it draws.
    pub source: usize,
    /// That character.
    pub character: char,
    pub glyph_id: u16,
    /// The character map had no glyph. `glyph_id` is the font's missing glyph.
    pub missing: bool,
    /// How far the pen moves after it. Kerning with its neighbours is in
    /// here.
    pub advance: i32,
    /// Where it is drawn from the pen.
    pub x_offset: i32,
    pub y_offset: i32,
    /// Which of the faces it was shaped in; 0 is the title's own.
    pub face: usize,
    /// Its direction: even runs left to right, odd right to left.
    pub level: u8,
    /// It is the first glyph of its character, or of the characters drawn
    /// as one. A line is never cut before any other.
    pub first: bool,
    /// What kerning with an earlier glyph changed, so a line that is cut
    /// between the two can take it back.
    pub(crate) pair: Pair,
}

/// What a pair's kerning changed on its second glyph and on its first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Pair {
    /// How many glyphs back the first glyph of the pair is; 0 is no pair.
    pub(crate) back: usize,
    /// Added to the first glyph's advance.
    pub(crate) first_advance: i32,
    /// Added to this glyph's offset and to its advance.
    pub(crate) x_offset: i32,
    pub(crate) advance: i32,
}

impl ShapedGlyph {
    /// Take back the kerning between this glyph and one before it in
    /// `earlier`, the glyphs up to this one: the two are on different
    /// lines now.
    pub(crate) fn unpair(&mut self, earlier: &mut [ShapedGlyph]) {
        let pair = std::mem::take(&mut self.pair);
        if pair.back == 0 {
            return;
        }
        self.x_offset -= pair.x_offset;
        self.advance -= pair.advance;
        if let Some(first) = earlier
            .len()
            .checked_sub(pair.back)
            .and_then(|index| earlier.get_mut(index))
        {
            first.advance -= pair.first_advance;
        }
    }
}

/// The glyphs between two breaks, in the order the text is stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Paragraph {
    /// 0 when the paragraph runs left to right, 1 for right to left.
    pub base: u8,
    pub glyphs: Vec<ShapedGlyph>,
    /// Byte index of each of its characters, and of its end.
    starts: Vec<usize>,
    levels: Levels,
}

impl Paragraph {
    /// The level of each glyph of `row`, a line of this paragraph that
    /// ends at byte `end`: its own, but for the spaces at the line's end
    /// and before a tab, which fall back to the paragraph's.
    pub(crate) fn line_levels(&self, row: &[ShapedGlyph], end: usize) -> Vec<u8> {
        let place = |byte: usize| self.starts.partition_point(|start| *start < byte);
        let Some(first) = row.first() else {
            return Vec::new();
        };
        let from = place(first.source);
        let line = self.levels.line_levels(from, place(end));
        row.iter()
            .map(|glyph| {
                place(glyph.source)
                    .checked_sub(from)
                    .and_then(|index| line.get(index))
                    .copied()
                    .unwrap_or(glyph.level)
            })
            .collect()
    }

    /// Byte index where the paragraph's text ends.
    pub(crate) fn end(&self) -> usize {
        self.starts.last().copied().unwrap_or(0)
    }
}

/// The shaped string: one paragraph for each run between breaks. There
/// are no pixels here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shaped {
    pub paragraphs: Vec<Paragraph>,
}

impl Shaped {
    /// Every glyph, in the order the text is stored.
    pub fn glyphs(&self) -> impl Iterator<Item = &ShapedGlyph> {
        self.paragraphs
            .iter()
            .flat_map(|paragraph| paragraph.glyphs.iter())
    }
}

/// Read `bytes` as one face and shape `text`.
pub fn shape_text(bytes: &[u8], text: &str) -> Result<Shaped, TextError> {
    let font = cutlass_font::read(bytes).map_err(TextError::Font)?;
    shape(&font, text)
}

/// Shape `text` with a face that has already been read.
pub fn shape(font: &Font<'_>, text: &str) -> Result<Shaped, TextError> {
    shape_faces(std::slice::from_ref(font), text, &|_| 0)
}

/// Shape `text`, each character in the face `pick` names for its byte
/// index; a face that is not in `fonts` is the first.
pub(crate) fn shape_faces(
    fonts: &[Font<'_>],
    text: &str,
    pick: &dyn Fn(usize) -> usize,
) -> Result<Shaped, TextError> {
    if text.chars().count() > MAX_CHARS {
        return Err(TextError::Limit);
    }
    let mut paragraphs = Vec::new();
    if text.is_empty() {
        return Ok(Shaped { paragraphs });
    }
    let mut start = 0;
    let mut characters = text.char_indices().peekable();
    while let Some((at, character)) = characters.next() {
        if !ends_line(character) {
            continue;
        }
        paragraphs.push(paragraph(fonts, text, start, at, pick)?);
        if character == '\r' {
            let _line_feed = characters.next_if(|(_, next)| *next == '\n');
        }
        start = characters.peek().map_or(text.len(), |(next, _)| *next);
    }
    paragraphs.push(paragraph(fonts, text, start, text.len(), pick)?);
    Ok(Shaped { paragraphs })
}

/// Shape bytes `start..end` of `text`, which hold no break.
fn paragraph(
    fonts: &[Font<'_>],
    text: &str,
    start: usize,
    end: usize,
    pick: &dyn Fn(usize) -> usize,
) -> Result<Paragraph, TextError> {
    let part = text.get(start..end).ok_or(TextError::Limit)?;
    let levels = resolve(part);
    let mut starts: Vec<usize> = part.char_indices().map(|(at, _)| start + at).collect();
    starts.push(end);
    let mut glyphs: Vec<ShapedGlyph> = Vec::with_capacity(starts.len());
    for (index, (at, character)) in part.char_indices().enumerate() {
        let source = start + at;
        let level = levels.levels.get(index).copied().unwrap_or(levels.base);
        let face = Some(pick(source))
            .filter(|face| *face < fonts.len())
            .unwrap_or(0);
        let Some(font) = fonts.get(face) else {
            return Err(TextError::Limit);
        };
        // A character that runs right to left is drawn as its mirror
        // image, when it has one the face can draw.
        let mirrored = (level % 2 == 1)
            .then(|| mirror(character))
            .flatten()
            .and_then(|other| font.glyph_id(other).ok().flatten());
        let found = match mirrored {
            Some(glyph_id) => Some(glyph_id),
            None => font.glyph_id(character).map_err(TextError::Font)?,
        };
        let (glyph_id, missing) = match found {
            Some(glyph_id) => (glyph_id, false),
            None if props(character).default_ignorable() => continue,
            None => (font.missing_glyph(), true),
        };
        let mut advance = i32::from(font.advance(glyph_id).map_err(TextError::Font)?);
        let mut x_offset = 0;
        let mut pair = Pair::default();
        // Kerning is shared between the pair: half off the first glyph's
        // advance, and the rest moves the second.
        if let Some(before) = glyphs
            .last_mut()
            .filter(|before| before.face == face && before.level == level && level % 2 == 0)
        {
            let kerning = i32::from(font.kerning(before.glyph_id, glyph_id));
            let first = kerning >> 1;
            before.advance += first;
            advance += kerning - first;
            x_offset = kerning - first;
            if kerning != 0 {
                pair = Pair {
                    back: 1,
                    first_advance: first,
                    x_offset,
                    advance: kerning - first,
                };
            }
        }
        glyphs.push(ShapedGlyph {
            source,
            character,
            glyph_id,
            missing,
            advance,
            x_offset,
            y_offset: 0,
            face,
            level,
            first: true,
            pair,
        });
    }
    Ok(Paragraph {
        base: levels.base,
        glyphs,
        starts,
        levels,
    })
}

#[cfg(test)]
mod tests {
    use super::{Shaped, ShapedGlyph, shape, shape_text};
    use crate::TextError;
    use cutlass_font::{FontError, read, read_collection};

    const TTF: &[u8] = include_bytes!("../../font/fixtures/sample.ttf");
    const OTF: &[u8] = include_bytes!("../../font/fixtures/sample.otf");
    const TTC: &[u8] = include_bytes!("../../font/fixtures/sample.ttc");

    /// Where each glyph is drawn, with the pen moving left to right.
    fn pens(glyphs: &[ShapedGlyph]) -> Vec<i32> {
        let mut pen = 0;
        glyphs
            .iter()
            .map(|glyph| {
                let x = pen + glyph.x_offset;
                pen += glyph.advance;
                x
            })
            .collect()
    }

    fn only(shaped: &Shaped) -> &[ShapedGlyph] {
        assert_eq!(shaped.paragraphs.len(), 1);
        &shaped.paragraphs[0].glyphs
    }

    #[test]
    fn advances_and_kerning_come_from_the_font() {
        let font = read(TTF).expect("ttf");
        let shaped = shape(&font, "AB").expect("shape");
        let glyphs = only(&shaped);
        assert_eq!(glyphs.len(), 2);
        let (a, b) = (&glyphs[0], &glyphs[1]);
        assert_eq!(a.character, 'A');
        assert_eq!(a.glyph_id, 2);
        assert!(!a.missing && a.first);
        assert_eq!(font.advance(2).expect("advance"), 600);
        assert_eq!(font.kerning(2, 3), -80);
        // The pair's kerning is shared: the run is 80 shorter, and B is
        // drawn 80 nearer to A.
        assert_eq!(a.advance + b.advance, 600 + 700 - 80);
        assert_eq!((a.x_offset, a.y_offset), (0, 0));
        assert_eq!(b.glyph_id, 3);
        assert_eq!(pens(glyphs), [0, 520]);

        let swapped = shape(&font, "BA").expect("shape");
        assert_eq!(pens(only(&swapped)), [0, 740]);

        let spaced = shape(&font, "A B").expect("space");
        let glyphs = only(&spaced);
        assert_eq!(glyphs[1].glyph_id, 1);
        assert_eq!(glyphs[1].advance, 250);
        assert_eq!(pens(glyphs)[2], 850);

        let otf = read(OTF).expect("otf");
        let cubic = shape(&otf, "AB").expect("otf");
        assert_eq!(otf.advance(2).expect("advance"), 640);
        assert_eq!(pens(only(&cubic)), [0, 560]);

        let face = read_collection(TTC, 1).expect("face");
        let collected = shape(&face, "AB").expect("collection");
        let glyphs = only(&collected);
        assert_eq!((glyphs[0].advance, glyphs[1].advance), (800, 900));
        assert_eq!(pens(glyphs), [0, 800]);
    }

    #[test]
    fn a_missing_glyph_uses_the_font_missing_outline() {
        let font = read(TTF).expect("ttf");
        let shaped = shape(&font, "Z").expect("shape");
        let missing = &only(&shaped)[0];
        assert!(missing.missing);
        assert_eq!(missing.glyph_id, font.missing_glyph());
        assert_eq!(
            missing.advance,
            i32::from(font.advance(font.missing_glyph()).expect("advance"))
        );
        assert_eq!(missing.advance, 500);
        assert_eq!(
            font.outline(missing.glyph_id).expect("outline"),
            font.missing_outline().expect("missing")
        );
        let emoji = shape(&font, "😀").expect("emoji");
        let emoji = &only(&emoji)[0];
        assert!(emoji.missing);
        assert_eq!(emoji.glyph_id, 0);
    }

    #[test]
    fn an_empty_string_and_a_newline_make_no_false_glyphs() {
        let font = read(TTF).expect("ttf");
        assert!(shape(&font, "").expect("empty").paragraphs.is_empty());
        let broken = shape(&font, "A\nB").expect("break");
        assert_eq!(broken.paragraphs.len(), 2);
        assert_eq!(broken.glyphs().count(), 2);
        // Kerning does not reach across the break.
        assert_eq!(broken.paragraphs[1].glyphs[0].advance, 700);
        assert_eq!(broken.paragraphs[1].glyphs[0].x_offset, 0);
        assert_eq!(broken.paragraphs[1].glyphs[0].source, 2);
        let crlf = shape(&font, "A\r\nB").expect("crlf");
        assert_eq!(crlf.paragraphs.len(), 2);
        assert_eq!(crlf.paragraphs[1].glyphs[0].source, 3);
        // A break at either end leaves an empty paragraph there, and every
        // character that ends a line is one.
        let ends = shape(&font, "\nA\n").expect("ends");
        assert_eq!(ends.paragraphs.len(), 3);
        assert!(ends.paragraphs[0].glyphs.is_empty() && ends.paragraphs[2].glyphs.is_empty());
        assert_eq!(
            shape(&font, "A\u{2028}B\u{85}A")
                .expect("others")
                .paragraphs
                .len(),
            3
        );
    }

    #[test]
    fn right_to_left_text_is_levelled_and_mirrored() {
        let font = read(TTF).expect("ttf");
        // Hebrew letters the fixture lacks still take their direction.
        let shaped = shape(&font, "A \u{5D0}\u{5D1} B").expect("mixed");
        let paragraph = &shaped.paragraphs[0];
        assert_eq!(paragraph.base, 0);
        let levels: Vec<u8> = paragraph.glyphs.iter().map(|glyph| glyph.level).collect();
        assert_eq!(levels, [0, 0, 1, 1, 0, 0]);
        let hebrew = shape(&font, "\u{5D0}A").expect("hebrew");
        assert_eq!(hebrew.paragraphs[0].base, 1);
        let levels: Vec<u8> = hebrew.paragraphs[0]
            .glyphs
            .iter()
            .map(|g| g.level)
            .collect();
        assert_eq!(levels, [1, 2]);
        // A direction mark makes no glyph, and turns what follows around.
        let turned = shape(&font, "\u{202E}AB").expect("override");
        let glyphs = &turned.paragraphs[0].glyphs;
        assert_eq!(glyphs.len(), 2);
        assert_eq!((glyphs[0].source, glyphs[0].level), (3, 1));
        // Kerning is not applied to a pair read right to left.
        assert_eq!((glyphs[0].advance, glyphs[1].advance), (600, 700));
        assert!(
            shape(&font, "\u{200D}\u{FE0F}")
                .expect("joiners")
                .glyphs()
                .next()
                .is_none()
        );
        // The spaces that end a line fall back to the paragraph's level.
        let trailing = shape(&font, "A \u{5D0} ").expect("trailing");
        let paragraph = &trailing.paragraphs[0];
        assert_eq!(
            paragraph.line_levels(&paragraph.glyphs, paragraph.end()),
            [0, 0, 1, 0]
        );
        assert_eq!(paragraph.line_levels(&paragraph.glyphs[..3], 4), [0, 0, 1]);
        assert!(paragraph.line_levels(&[], 0).is_empty());
    }

    #[test]
    fn kerning_can_be_taken_back_where_a_line_is_cut() {
        let font = read(TTF).expect("ttf");
        let shaped = shape(&font, "AB").expect("shape");
        let mut glyphs = only(&shaped).to_vec();
        let (first, second) = glyphs.split_at_mut(1);
        second[0].unpair(first);
        assert_eq!((glyphs[0].advance, glyphs[1].advance), (600, 700));
        assert_eq!(glyphs[1].x_offset, 0);
        // Taking it back twice changes nothing more.
        let (first, second) = glyphs.split_at_mut(1);
        second[0].unpair(first);
        assert_eq!((glyphs[0].advance, glyphs[1].advance), (600, 700));
    }

    #[test]
    fn a_truncated_font_is_an_error() {
        assert!(matches!(
            shape_text(&[], "A"),
            Err(TextError::Font(FontError::Truncated))
        ));
        assert!(matches!(
            shape_text(&[0, 1, 0, 0], "AB"),
            Err(TextError::Font(FontError::Truncated))
        ));
        assert!(matches!(
            shape_text(TTC, "A"),
            Err(TextError::Font(FontError::Collection))
        ));
        let font = read(TTF).expect("ttf");
        let huge = "A".repeat(super::MAX_CHARS + 1);
        assert!(matches!(shape(&font, &huge), Err(TextError::Limit)));
    }

    #[test]
    fn a_long_run_keeps_one_position_per_character() {
        let font = read(TTF).expect("ttf");
        let shaped = shape(&font, &"A".repeat(4_096)).expect("shape");
        let glyphs = only(&shaped);
        assert_eq!(glyphs.len(), 4_096);
        assert!(
            glyphs
                .iter()
                .all(|glyph| glyph.advance == 600 && glyph.glyph_id == 2 && !glyph.missing)
        );
        assert_eq!(pens(glyphs).last().copied(), Some(4_095 * 600));
        assert_eq!(glyphs.last().expect("last").source, 4_095);
    }
}
