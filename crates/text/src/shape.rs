//! A string and a font become glyph ids, positions, and advances.
//!
//! This step does not draw. Advances and kerning come from the font. A
//! character with no glyph is recorded as missing and uses glyph 0, whose
//! outline is the font's missing glyph. A newline is a break, not a glyph.

use cutlass_font::Font;

use crate::TextError;

/// Characters in one `shape` call. A longer string is refused.
pub const MAX_CHARS: usize = 65_536;

/// One placed glyph, still in font design units.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapedGlyph {
    /// Byte index of `character` in the source string.
    pub source: usize,
    pub character: char,
    pub glyph_id: u16,
    /// The character map had no glyph. `glyph_id` is the font's missing glyph.
    pub missing: bool,
    pub advance: u16,
    /// Adjustment applied before this glyph, from the previous glyph in the run.
    pub kerning: i16,
    pub x: i32,
    pub y: i32,
}

/// A glyph or a line break from the source string.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShapeItem {
    Glyph(ShapedGlyph),
    Break,
}

/// The shaped run. There are no pixels here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shaped {
    pub items: Vec<ShapeItem>,
}

/// Read `bytes` as one face and shape `text`.
pub fn shape_text(bytes: &[u8], text: &str) -> Result<Shaped, TextError> {
    let font = cutlass_font::read(bytes).map_err(TextError::Font)?;
    shape(&font, text)
}

/// Shape `text` with a face that has already been read.
pub fn shape(font: &Font<'_>, text: &str) -> Result<Shaped, TextError> {
    if text.chars().count() > MAX_CHARS {
        return Err(TextError::Limit);
    }
    let mut items = Vec::new();
    let mut pen = 0_i32;
    let mut previous: Option<u16> = None;
    let mut chars = text.char_indices().peekable();
    while let Some((source, character)) = chars.next() {
        if character == '\n' || character == '\r' {
            if character == '\r' {
                let _line_feed = chars.next_if(|(_, next)| *next == '\n');
            }
            items.push(ShapeItem::Break);
            previous = None;
            pen = 0;
            continue;
        }
        let (glyph_id, missing) = match font.glyph_id(character).map_err(TextError::Font)? {
            Some(glyph_id) => (glyph_id, false),
            None => (font.missing_glyph(), true),
        };
        let advance = font.advance(glyph_id).map_err(TextError::Font)?;
        let kerning = previous.map_or(0, |left| font.kerning(left, glyph_id));
        let x = pen
            .checked_add(i32::from(kerning))
            .ok_or(TextError::Limit)?;
        pen = x.checked_add(i32::from(advance)).ok_or(TextError::Limit)?;
        items.push(ShapeItem::Glyph(ShapedGlyph {
            source,
            character,
            glyph_id,
            missing,
            advance,
            kerning,
            x,
            y: 0,
        }));
        previous = Some(glyph_id);
    }
    Ok(Shaped { items })
}

#[cfg(test)]
mod tests {
    use super::{ShapeItem, shape, shape_text};
    use crate::TextError;
    use cutlass_font::{FontError, read, read_collection};

    const TTF: &[u8] = include_bytes!("../../font/fixtures/sample.ttf");
    const OTF: &[u8] = include_bytes!("../../font/fixtures/sample.otf");
    const TTC: &[u8] = include_bytes!("../../font/fixtures/sample.ttc");

    fn glyph(item: &ShapeItem) -> &super::ShapedGlyph {
        match item {
            ShapeItem::Glyph(glyph) => glyph,
            ShapeItem::Break => panic!("break"),
        }
    }

    #[test]
    fn advances_and_kerning_come_from_the_font() {
        let font = read(TTF).expect("ttf");
        let shaped = shape(&font, "AB").expect("shape");
        assert_eq!(shaped.items.len(), 2);
        let a = glyph(&shaped.items[0]);
        let b = glyph(&shaped.items[1]);
        assert_eq!(a.character, 'A');
        assert_eq!(a.glyph_id, 2);
        assert!(!a.missing);
        assert_eq!(a.advance, font.advance(2).expect("advance"));
        assert_eq!(a.advance, 600);
        assert_eq!(a.kerning, 0);
        assert_eq!(a.x, 0);
        assert_eq!(a.y, 0);
        assert_eq!(b.glyph_id, 3);
        assert_eq!(b.advance, 700);
        assert_eq!(b.kerning, font.kerning(2, 3));
        assert_eq!(b.kerning, -80);
        assert_eq!(b.x, 520);

        let swapped = shape(&font, "BA").expect("shape");
        assert_eq!(glyph(&swapped.items[1]).x, 740);
        assert_eq!(glyph(&swapped.items[1]).kerning, 40);

        let spaced = shape(&font, "A B").expect("space");
        assert_eq!(glyph(&spaced.items[1]).glyph_id, 1);
        assert_eq!(glyph(&spaced.items[1]).advance, 250);
        assert_eq!(glyph(&spaced.items[2]).x, 850);

        let otf = read(OTF).expect("otf");
        let cubic = shape(&otf, "AB").expect("otf");
        assert_eq!(glyph(&cubic.items[0]).advance, 640);
        assert_eq!(glyph(&cubic.items[1]).kerning, -80);
        assert_eq!(glyph(&cubic.items[1]).x, 560);

        let face = read_collection(TTC, 1).expect("face");
        let collected = shape(&face, "AB").expect("collection");
        assert_eq!(glyph(&collected.items[0]).advance, 800);
        assert_eq!(glyph(&collected.items[1]).advance, 900);
        assert_eq!(glyph(&collected.items[1]).kerning, 0);
        assert_eq!(glyph(&collected.items[1]).x, 800);
    }

    #[test]
    fn a_missing_glyph_uses_the_font_missing_outline() {
        let font = read(TTF).expect("ttf");
        let shaped = shape(&font, "Z").expect("shape");
        let missing = glyph(&shaped.items[0]);
        assert!(missing.missing);
        assert_eq!(missing.glyph_id, font.missing_glyph());
        assert_eq!(
            missing.advance,
            font.advance(font.missing_glyph()).expect("advance")
        );
        assert_eq!(missing.advance, 500);
        assert_eq!(
            font.outline(missing.glyph_id).expect("outline"),
            font.missing_outline().expect("missing")
        );
        let emoji = shape(&font, "😀").expect("emoji");
        let emoji = glyph(&emoji.items[0]);
        assert!(emoji.missing);
        assert_eq!(emoji.glyph_id, 0);
    }

    #[test]
    fn an_empty_string_and_a_newline_make_no_false_glyphs() {
        let font = read(TTF).expect("ttf");
        assert!(shape(&font, "").expect("empty").items.is_empty());
        let broken = shape(&font, "A\nB").expect("break");
        assert!(matches!(broken.items[1], ShapeItem::Break));
        assert_eq!(glyph(&broken.items[2]).kerning, 0);
        assert_eq!(glyph(&broken.items[2]).x, 0);
        let crlf = shape(&font, "A\r\nB").expect("crlf");
        assert_eq!(crlf.items.len(), 3);
        assert!(matches!(crlf.items[1], ShapeItem::Break));
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
        assert_eq!(shaped.items.len(), 4_096);
        assert!(shaped.items.iter().all(|item| {
            let glyph = glyph(item);
            glyph.advance == 600 && glyph.glyph_id == 2 && !glyph.missing
        }));
        assert_eq!(glyph(shaped.items.last().expect("last")).x, 4_095 * 600);
    }
}
