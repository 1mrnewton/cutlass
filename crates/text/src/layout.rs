//! Lines, alignment, and spacing from a text material.
//!
//! `line_max_width` of zero does not wrap: the run stays on one line unless
//! the string contains a break. A negative width is refused. Alignment is 0
//! left, 1 center, and 2 right. `letter_spacing` and `line_spacing` are
//! fractions of `font_size`.

use cutlass_font::Font;
use cutlass_model::MaterialText;

use crate::TextError;
use crate::shape::{ShapeItem, ShapedGlyph, shape};

/// Horizontal alignment from [`MaterialText::alignment`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alignment {
    Left,
    Center,
    Right,
}

/// One glyph on a line, in pixels. Y is the line baseline, top to bottom.
#[derive(Clone, Debug, PartialEq)]
pub struct PlacedGlyph {
    pub source: usize,
    pub character: char,
    pub glyph_id: u16,
    pub missing: bool,
    pub x: f32,
    pub advance: f32,
}

/// One line after wrapping and alignment.
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    pub glyphs: Vec<PlacedGlyph>,
    pub width: f32,
    pub baseline: f32,
}

/// The laid-out run. An empty string has no lines.
#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub lines: Vec<Line>,
    pub font_size: f32,
    /// Pixels per font design unit.
    pub scale: f32,
    pub alignment: Alignment,
}

impl Layout {
    pub fn glyphs(&self) -> impl Iterator<Item = &PlacedGlyph> {
        self.lines.iter().flat_map(|line| line.glyphs.iter())
    }
}

/// Read `bytes` as one face and lay out `material`.
pub fn layout_text(bytes: &[u8], material: &MaterialText) -> Result<Layout, TextError> {
    let font = cutlass_font::read(bytes).map_err(TextError::Font)?;
    layout(&font, material)
}

/// Lay out `material` with a face that has already been read.
pub fn layout(font: &Font<'_>, material: &MaterialText) -> Result<Layout, TextError> {
    if material.typesetting != 0 {
        return Err(TextError::Typesetting);
    }
    let text = crate::richtext::plain(material)?;
    let alignment = alignment(material.alignment)?;
    let font_size = positive_f32(material.font_size)?;
    let letter_spacing = finite_f32(material.letter_spacing)?;
    let line_spacing = finite_f32(material.line_spacing)?;
    let width = finite_f32(material.line_max_width)?;
    if width < 0.0 {
        return Err(TextError::Measure);
    }
    let step = font_size * (1.0 + line_spacing);
    let letter_px = letter_spacing * font_size;
    let scale = font_size / f32::from(font.units_per_em());
    if !(step.is_finite()
        && step > 0.0
        && letter_px.is_finite()
        && scale.is_finite()
        && scale > 0.0)
    {
        return Err(TextError::Measure);
    }
    if text.is_empty() {
        return Ok(Layout {
            lines: Vec::new(),
            font_size,
            scale,
            alignment,
        });
    }
    let shaped = shape(font, text)?;
    let rows = wrap(&shaped.items, width, scale, letter_px)?;
    place(rows, font_size, scale, letter_px, width, step, alignment)
}

fn alignment(value: i32) -> Result<Alignment, TextError> {
    match value {
        0 => Ok(Alignment::Left),
        1 => Ok(Alignment::Center),
        2 => Ok(Alignment::Right),
        _ => Err(TextError::Alignment),
    }
}

fn wrap(
    items: &[ShapeItem],
    width: f32,
    scale: f32,
    letter_px: f32,
) -> Result<Vec<Vec<ShapedGlyph>>, TextError> {
    let mut lines: Vec<Vec<ShapedGlyph>> = Vec::new();
    let mut used = 0.0_f32;
    for item in items {
        match item {
            ShapeItem::Break => {
                if lines.is_empty() {
                    lines.push(Vec::new());
                }
                lines.push(Vec::new());
                used = 0.0;
            }
            ShapeItem::Glyph(glyph) => {
                let advance = f32::from(glyph.advance) * scale;
                let starting = lines.last().is_none_or(|line| line.is_empty());
                let kern = if starting {
                    0.0
                } else {
                    f32::from(glyph.kerning) * scale
                };
                let letter = if starting { 0.0 } else { letter_px };
                let next = used + kern + letter + advance;
                if !next.is_finite() || !advance.is_finite() {
                    return Err(TextError::Limit);
                }
                let wrap_here = width > 0.0 && !starting && next > width;
                if wrap_here {
                    lines.push(vec![glyph.clone()]);
                    used = advance;
                } else {
                    if lines.is_empty() {
                        lines.push(Vec::new());
                    }
                    let Some(line) = lines.last_mut() else {
                        return Err(TextError::Limit);
                    };
                    line.push(glyph.clone());
                    used = if starting { advance } else { next };
                }
            }
        }
    }
    Ok(lines)
}

fn place(
    rows: Vec<Vec<ShapedGlyph>>,
    font_size: f32,
    scale: f32,
    letter_px: f32,
    width: f32,
    step: f32,
    alignment: Alignment,
) -> Result<Layout, TextError> {
    let mut lines = Vec::with_capacity(rows.len());
    let mut baseline = font_size;
    for row in rows {
        let (glyphs, line_width) = place_line(&row, scale, letter_px)?;
        if !baseline.is_finite() {
            return Err(TextError::Limit);
        }
        lines.push(Line {
            glyphs,
            width: line_width,
            baseline,
        });
        baseline += step;
    }
    let content = lines.iter().map(|line| line.width).fold(0.0_f32, f32::max);
    let box_width = if width > 0.0 { width } else { content };
    if !box_width.is_finite() {
        return Err(TextError::Limit);
    }
    for line in &mut lines {
        let shift = match alignment {
            Alignment::Left => 0.0,
            Alignment::Center => (box_width - line.width) / 2.0,
            Alignment::Right => box_width - line.width,
        };
        if !shift.is_finite() {
            return Err(TextError::Limit);
        }
        for glyph in &mut line.glyphs {
            glyph.x += shift;
            if !glyph.x.is_finite() {
                return Err(TextError::Limit);
            }
        }
    }
    Ok(Layout {
        lines,
        font_size,
        scale,
        alignment,
    })
}

fn place_line(
    row: &[ShapedGlyph],
    scale: f32,
    letter_px: f32,
) -> Result<(Vec<PlacedGlyph>, f32), TextError> {
    let mut glyphs = Vec::with_capacity(row.len());
    let mut pen = 0.0_f32;
    for (index, glyph) in row.iter().enumerate() {
        let advance = f32::from(glyph.advance) * scale;
        if index > 0 {
            pen += f32::from(glyph.kerning) * scale + letter_px;
        }
        if !pen.is_finite() || !advance.is_finite() {
            return Err(TextError::Limit);
        }
        glyphs.push(PlacedGlyph {
            source: glyph.source,
            character: glyph.character,
            glyph_id: glyph.glyph_id,
            missing: glyph.missing,
            x: pen,
            advance,
        });
        pen += advance;
    }
    if !pen.is_finite() {
        return Err(TextError::Limit);
    }
    Ok((glyphs, pen))
}

fn finite_f32(value: f64) -> Result<f32, TextError> {
    if !value.is_finite() {
        return Err(TextError::Measure);
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    let value = value as f32;
    if !value.is_finite() {
        return Err(TextError::Measure);
    }
    Ok(value)
}

fn positive_f32(value: f64) -> Result<f32, TextError> {
    let value = finite_f32(value)?;
    if value <= 0.0 {
        Err(TextError::Measure)
    } else {
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::{layout, layout_text};
    use crate::TextError;
    use cutlass_font::FontError;
    use cutlass_model::MaterialText;

    const TTF: &[u8] = include_bytes!("../../font/fixtures/sample.ttf");

    fn material(content: &str) -> MaterialText {
        let mut text = MaterialText::new("t");
        text.content = content.to_owned();
        text.font_size = 1_000.0;
        text.line_max_width = 0.0;
        text
    }

    fn font() -> cutlass_font::Font<'static> {
        cutlass_font::read(TTF).expect("ttf")
    }

    #[test]
    fn an_empty_string_produces_no_glyphs() {
        let laid = layout(&font(), &material("")).expect("layout");
        assert!(laid.lines.is_empty());
        assert_eq!(laid.glyphs().count(), 0);
    }

    #[test]
    fn a_width_of_zero_stays_on_one_line() {
        let face = font();
        let laid = layout(&face, &material("AB AB")).expect("layout");
        assert_eq!(laid.lines.len(), 1);
        assert_eq!(laid.lines[0].glyphs.len(), 5);
        assert_eq!(laid.lines[0].glyphs[1].x.to_bits(), 520.0_f32.to_bits());
        assert_eq!(laid.lines[0].glyphs[1].glyph_id, 3);
        let long = layout(&face, &material(&"AB".repeat(32))).expect("long");
        assert_eq!(long.lines.len(), 1);
        assert_eq!(long.glyphs().count(), 64);
    }

    #[test]
    fn a_positive_width_wraps_and_alignment_shifts_the_line() {
        let face = font();
        let mut wrapped = material("AB");
        wrapped.line_max_width = 1_000.0;
        let laid = layout(&face, &wrapped).expect("wrap");
        assert_eq!(laid.lines.len(), 2);
        assert_eq!(laid.lines[0].glyphs[0].character, 'A');
        assert_eq!(laid.lines[1].glyphs[0].character, 'B');
        assert_eq!(laid.lines[1].glyphs[0].x.to_bits(), 0.0_f32.to_bits());

        let mut text = material("A");
        text.line_max_width = 2_000.0;
        text.alignment = 0;
        let left = layout(&face, &text).expect("left");
        assert_eq!(left.lines[0].glyphs[0].x.to_bits(), 0.0_f32.to_bits());
        text.alignment = 1;
        let center = layout(&face, &text).expect("center");
        assert_eq!(center.lines[0].glyphs[0].x.to_bits(), 700.0_f32.to_bits());
        text.alignment = 2;
        let right = layout(&face, &text).expect("right");
        assert_eq!(right.lines[0].glyphs[0].x.to_bits(), 1_400.0_f32.to_bits());
    }

    #[test]
    fn spacing_moves_glyphs_and_baselines() {
        let face = font();
        let mut letters = material("AA");
        letters.letter_spacing = 1.0;
        let laid = layout(&face, &letters).expect("letters");
        assert_eq!(laid.lines[0].glyphs[1].x.to_bits(), 1_600.0_f32.to_bits());

        let mut lines = material("A\nA");
        lines.line_spacing = 0.5;
        let laid = layout(&face, &lines).expect("lines");
        assert_eq!(laid.lines.len(), 2);
        assert_eq!(laid.lines[0].baseline.to_bits(), 1_000.0_f32.to_bits());
        assert_eq!(laid.lines[1].baseline.to_bits(), 2_500.0_f32.to_bits());
    }

    #[test]
    fn a_bad_measure_or_a_truncated_font_is_an_error() {
        let face = font();
        let mut text = material("A");
        text.line_max_width = -1.0;
        assert!(matches!(layout(&face, &text), Err(TextError::Measure)));
        text.line_max_width = 0.0;
        text.font_size = f64::NAN;
        assert!(matches!(layout(&face, &text), Err(TextError::Measure)));
        text.font_size = 1_000.0;
        text.alignment = 9;
        assert!(matches!(layout(&face, &text), Err(TextError::Alignment)));
        text.alignment = 0;
        text.typesetting = 1;
        assert!(matches!(layout(&face, &text), Err(TextError::Typesetting)));
        text.typesetting = 0;
        text.is_rich_text = true;
        assert!(matches!(
            layout(&face, &text),
            Err(TextError::RichTextCommands)
        ));
        assert!(matches!(
            layout_text(&[], &material("A")),
            Err(TextError::Font(FontError::Truncated))
        ));
        assert!(matches!(
            layout_text(&[0, 1, 0, 0], &material("A")),
            Err(TextError::Font(FontError::Truncated))
        ));
    }
}
