//! Lines, alignment, and spacing from a text material.
//!
//! `line_max_width` of zero does not wrap: the run stays on one line unless
//! the string contains a break. A negative width is refused. Alignment is 0
//! left, 1 center, and 2 right. `letter_spacing` and `line_spacing` are
//! fractions of `font_size`. A rich title's range may draw its letters at
//! another size; a line is as tall as its largest letters.
//!
//! Vertical text (`typesetting` 1) stands each letter upright in a column,
//! one em a letter, columns running right to left. `line_max_width` is then
//! the column height, and alignment 3, 4 and 5 put a short column at the
//! top, middle or bottom of the tallest (0, 1 and 2 read the same way).

use cutlass_font::Font;
use cutlass_model::MaterialText;

use crate::TextError;
use crate::shape::{ShapeItem, ShapedGlyph, shape_faces};

/// Alignment from [`MaterialText::alignment`]: left, center and right
/// across a line, or top, middle and bottom down a vertical column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alignment {
    Left,
    Center,
    Right,
    Top,
    Middle,
    Bottom,
}

/// Where a vertical letter's baseline sits in its em, from the top. Fonts
/// here carry no ascent, so this is the usual share for Latin faces.
pub(crate) const VERTICAL_BASELINE: f32 = 0.8;

/// One glyph on a line, in pixels. Y is the line baseline, top to bottom.
#[derive(Clone, Debug, PartialEq)]
pub struct PlacedGlyph {
    pub source: usize,
    pub character: char,
    pub glyph_id: u16,
    pub missing: bool,
    pub x: f32,
    /// This glyph's own baseline: its line's, or its place down a column.
    pub y: f32,
    pub advance: f32,
    /// Times the layout's `font_size`.
    pub size: f32,
    /// Which of the layout's faces draws it; 0 is the title's own.
    pub face: usize,
}

/// One line after wrapping and alignment; a column for vertical text, whose
/// `width` is then its height and `baseline` its first letter's.
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
    /// Pixels per font design unit of the title's own face.
    pub scale: f32,
    /// Pixels per design unit of each face, the title's own first.
    pub scales: Vec<f32>,
    pub alignment: Alignment,
}

impl Layout {
    /// Whether the lines are vertical columns.
    #[must_use]
    pub fn vertical(&self) -> bool {
        matches!(
            self.alignment,
            Alignment::Top | Alignment::Middle | Alignment::Bottom
        )
    }

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
    layout_faces(std::slice::from_ref(font), material, &|_| 0)
}

/// Lay out `material` with `fonts`, the title's own face first, each
/// character in the face `pick` names for its byte index.
pub(crate) fn layout_faces(
    fonts: &[Font<'_>],
    material: &MaterialText,
    pick: &dyn Fn(usize) -> usize,
) -> Result<Layout, TextError> {
    let Some(font) = fonts.first() else {
        return Err(TextError::Limit);
    };
    let words = crate::richtext::read(material)?;
    let text = words.text.as_ref();
    let alignment = alignment(material.alignment, material.typesetting)?;
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
    let mut scales = Vec::with_capacity(fonts.len());
    for face in fonts {
        let face_scale = font_size / f32::from(face.units_per_em());
        if !(face_scale.is_finite() && face_scale > 0.0) {
            return Err(TextError::Measure);
        }
        scales.push(face_scale);
    }
    if text.is_empty() {
        return Ok(Layout {
            lines: Vec::new(),
            font_size,
            scale,
            scales,
            alignment,
        });
    }
    let shaped = shape_faces(fonts, text, pick)?;
    // Pixels per design unit of each glyph: its face's, times its size.
    let units = |glyph: &ShapedGlyph| {
        let face = scales.get(glyph.face).copied().unwrap_or(scale);
        face * crate::richtext::size_at(&words.runs, glyph.source)
    };
    let size = |glyph: &ShapedGlyph| crate::richtext::size_at(&words.runs, glyph.source);
    let metrics = Metrics {
        font_size,
        letter_px,
        width,
        step,
    };
    let mut laid = if matches!(
        alignment,
        Alignment::Top | Alignment::Middle | Alignment::Bottom
    ) {
        let em = |glyph: &ShapedGlyph| font_size * size(glyph);
        let rows = wrap_with(&shaped.items, width, letter_px, &em)?;
        place_columns(rows, metrics, alignment, &size, &units)?
    } else {
        let advance = |glyph: &ShapedGlyph| f32::from(glyph.advance) * units(glyph);
        let kern = |glyph: &ShapedGlyph| f32::from(glyph.kerning) * units(glyph);
        let rows = wrap_kerned(&shaped.items, width, letter_px, &advance, &kern)?;
        place(rows, metrics, alignment, &size, &units)?
    };
    laid.scale = scale;
    laid.scales = scales;
    Ok(laid)
}

fn alignment(value: i32, typesetting: i32) -> Result<Alignment, TextError> {
    match (typesetting, value) {
        (0, 0) => Ok(Alignment::Left),
        (0, 1) => Ok(Alignment::Center),
        (0, 2) => Ok(Alignment::Right),
        (1, 0 | 3) => Ok(Alignment::Top),
        (1, 1 | 4) => Ok(Alignment::Middle),
        (1, 2 | 5) => Ok(Alignment::Bottom),
        (0 | 1, _) => Err(TextError::Alignment),
        _ => Err(TextError::Typesetting),
    }
}

/// Sizes from the material, in pixels; `scale` is pixels per design unit.
#[derive(Clone, Copy)]
struct Metrics {
    font_size: f32,
    letter_px: f32,
    width: f32,
    step: f32,
}

/// Wrap upright letters down columns: each is `length` long, with no
/// kerning between them.
fn wrap_with(
    items: &[ShapeItem],
    width: f32,
    letter_px: f32,
    length: &dyn Fn(&ShapedGlyph) -> f32,
) -> Result<Vec<Vec<ShapedGlyph>>, TextError> {
    wrap_kerned(items, width, letter_px, length, &|_| 0.0)
}

fn wrap_kerned(
    items: &[ShapeItem],
    width: f32,
    letter_px: f32,
    length: &dyn Fn(&ShapedGlyph) -> f32,
    kerning: &dyn Fn(&ShapedGlyph) -> f32,
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
                let advance = length(glyph);
                let starting = lines.last().is_none_or(|line| line.is_empty());
                let kern = if starting { 0.0 } else { kerning(glyph) };
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
    metrics: Metrics,
    alignment: Alignment,
    size: &dyn Fn(&ShapedGlyph) -> f32,
    units: &dyn Fn(&ShapedGlyph) -> f32,
) -> Result<Layout, TextError> {
    let Metrics {
        font_size,
        letter_px,
        width,
        step,
    } = metrics;
    let mut lines = Vec::with_capacity(rows.len());
    let mut baseline = 0.0_f32;
    for (index, row) in rows.iter().enumerate() {
        let tallest = row.iter().map(size).fold(1.0_f32, f32::max);
        baseline += if index == 0 { font_size } else { step } * tallest;
        let (glyphs, line_width) = place_line(row, letter_px, size, units)?;
        if !baseline.is_finite() {
            return Err(TextError::Limit);
        }
        lines.push(Line {
            glyphs,
            width: line_width,
            baseline,
        });
    }
    let content = lines.iter().map(|line| line.width).fold(0.0_f32, f32::max);
    let box_width = if width > 0.0 { width } else { content };
    if !box_width.is_finite() {
        return Err(TextError::Limit);
    }
    for line in &mut lines {
        let shift = match alignment {
            Alignment::Center => (box_width - line.width) / 2.0,
            Alignment::Right => box_width - line.width,
            _ => 0.0,
        };
        if !shift.is_finite() {
            return Err(TextError::Limit);
        }
        for glyph in &mut line.glyphs {
            glyph.x += shift;
            glyph.y = line.baseline;
            if !glyph.x.is_finite() {
                return Err(TextError::Limit);
            }
        }
    }
    Ok(Layout {
        lines,
        font_size,
        scale: 0.0,
        scales: Vec::new(),
        alignment,
    })
}

/// Stand each column's letters one under the next, centered across the
/// column, the first column on the right.
fn place_columns(
    rows: Vec<Vec<ShapedGlyph>>,
    metrics: Metrics,
    alignment: Alignment,
    size: &dyn Fn(&ShapedGlyph) -> f32,
    units: &dyn Fn(&ShapedGlyph) -> f32,
) -> Result<Layout, TextError> {
    let Metrics {
        font_size,
        letter_px,
        width: height,
        step,
    } = metrics;
    // Each column's thickness and its distance from the previous one.
    let pitches: Vec<(f32, f32)> = rows
        .iter()
        .enumerate()
        .map(|(index, row)| {
            let widest = row.iter().map(size).fold(1.0_f32, f32::max);
            let pitch = if index == 0 { font_size } else { step };
            (font_size * widest, pitch * widest)
        })
        .collect();
    let across: f32 = pitches.iter().map(|(_, pitch)| pitch).sum();
    let mut lines = Vec::with_capacity(rows.len());
    let mut right = across;
    for (row, (thick, pitch)) in rows.iter().zip(&pitches) {
        let middle = right - thick / 2.0;
        right -= pitch;
        let mut glyphs = Vec::with_capacity(row.len());
        let mut pen = 0.0_f32;
        for (index, glyph) in row.iter().enumerate() {
            let times = size(glyph);
            let em = font_size * times;
            if index > 0 {
                pen += letter_px;
            }
            let advance = f32::from(glyph.advance) * units(glyph);
            let (x, y) = (middle - advance / 2.0, pen + em * VERTICAL_BASELINE);
            if ![x, y, advance].iter().all(|value| value.is_finite()) {
                return Err(TextError::Limit);
            }
            glyphs.push(PlacedGlyph {
                source: glyph.source,
                character: glyph.character,
                glyph_id: glyph.glyph_id,
                missing: glyph.missing,
                x,
                y,
                advance,
                size: times,
                face: glyph.face,
            });
            pen += em;
        }
        let baseline = glyphs
            .first()
            .map_or(font_size * VERTICAL_BASELINE, |glyph| glyph.y);
        lines.push(Line {
            glyphs,
            width: pen,
            baseline,
        });
    }
    let content = lines.iter().map(|line| line.width).fold(0.0_f32, f32::max);
    let box_height = if height > 0.0 { height } else { content };
    for line in &mut lines {
        let shift = match alignment {
            Alignment::Middle => (box_height - line.width) / 2.0,
            Alignment::Bottom => box_height - line.width,
            _ => 0.0,
        };
        if !shift.is_finite() {
            return Err(TextError::Limit);
        }
        line.baseline += shift;
        for glyph in &mut line.glyphs {
            glyph.y += shift;
        }
    }
    Ok(Layout {
        lines,
        font_size,
        scale: 0.0,
        scales: Vec::new(),
        alignment,
    })
}

fn place_line(
    row: &[ShapedGlyph],
    letter_px: f32,
    size: &dyn Fn(&ShapedGlyph) -> f32,
    units: &dyn Fn(&ShapedGlyph) -> f32,
) -> Result<(Vec<PlacedGlyph>, f32), TextError> {
    let mut glyphs = Vec::with_capacity(row.len());
    let mut pen = 0.0_f32;
    for (index, glyph) in row.iter().enumerate() {
        let times = size(glyph);
        let scale = units(glyph);
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
            y: 0.0,
            advance,
            size: times,
            face: glyph.face,
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
    fn vertical_text_stacks_letters_in_columns_from_the_right() {
        let face = font();
        let mut text = material("AB\nA");
        text.typesetting = 1;
        text.alignment = 3;
        let laid = layout(&face, &text).expect("vertical");
        assert!(laid.vertical());
        assert_eq!(laid.lines.len(), 2);
        let (first, second) = (&laid.lines[0], &laid.lines[1]);
        // One em a letter, the baseline 0.8 down each.
        assert_eq!(first.glyphs[0].y.to_bits(), 800.0_f32.to_bits());
        assert_eq!(first.glyphs[1].y.to_bits(), 1_800.0_f32.to_bits());
        assert_eq!(first.width.to_bits(), 2_000.0_f32.to_bits());
        // The first column is on the right, each letter centered in it.
        let center = |glyph: &super::PlacedGlyph| glyph.x + glyph.advance / 2.0;
        assert_eq!(center(&first.glyphs[0]).to_bits(), 1_500.0_f32.to_bits());
        assert_eq!(center(&second.glyphs[0]).to_bits(), 500.0_f32.to_bits());

        text.alignment = 5;
        let bottom = layout(&face, &text).expect("bottom");
        assert_eq!(bottom.lines[1].glyphs[0].y.to_bits(), 1_800.0_f32.to_bits());
        text.alignment = 4;
        let middle = layout(&face, &text).expect("middle");
        assert_eq!(middle.lines[1].glyphs[0].y.to_bits(), 1_300.0_f32.to_bits());

        // A column height wraps the letters onto the next column.
        text.content = "AB".to_owned();
        text.alignment = 3;
        text.line_max_width = 1_500.0;
        let wrapped = layout(&face, &text).expect("wrapped");
        assert_eq!(wrapped.lines.len(), 2);
        assert_eq!(wrapped.lines[1].glyphs[0].character, 'B');
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
        text.typesetting = 2;
        assert!(matches!(layout(&face, &text), Err(TextError::Typesetting)));
        text.typesetting = 1;
        text.alignment = 6;
        assert!(matches!(layout(&face, &text), Err(TextError::Alignment)));
        text.alignment = 0;
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
