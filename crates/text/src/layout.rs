//! Lines, alignment, and spacing from a text material.
//!
//! `line_max_width` of zero does not wrap: the run stays on one line unless
//! the string contains a break. A negative width is refused. A wider run
//! wraps where the Unicode line breaking rules let a line end, which is
//! after a word's spaces for most scripts and between ideographs; a word
//! is cut only when it alone is wider than the line, and never inside a
//! letter with its marks or an emoji sequence. The spaces a line wraps
//! after hang past its end and do not count in its width. Alignment is 0
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
use crate::bidi::visual_order;
use crate::cache::{TextCache, layout_key};
use crate::faces::{FaceSource, Wanted, gather};
use crate::linebreak::{Boundary, boundaries};
use crate::shape::{Shaped, ShapedGlyph, shape_faces};

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
/// A line's glyphs are in the order they are seen, left to right, which
/// for text that runs right to left is not the order it is stored in.
#[derive(Clone, Debug, PartialEq)]
pub struct PlacedGlyph {
    pub source: usize,
    pub character: char,
    pub glyph_id: u16,
    pub missing: bool,
    pub x: f32,
    /// This glyph's own baseline: its line's, or its place down a column.
    pub y: f32,
    /// How far above that baseline it is drawn: a mark set on its letter.
    pub lift: f32,
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
    /// The lines are laid across, to be turned a quarter clockwise when
    /// drawn: this is how a script written only down the page, with its
    /// lines from the left, stands in columns. The first line is then
    /// the lowest, so that it is the leftmost once turned.
    pub turned: bool,
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

/// Read `bytes` as the title's own face and lay out `material`, with the
/// other faces `faces` supplies: those a rich title's ranges name, and a
/// fallback for a character its own face cannot draw.
pub fn layout_with<'f>(
    bytes: &'f [u8],
    material: &MaterialText,
    faces: &'f dyn FaceSource,
) -> Result<Layout, TextError> {
    let font = cutlass_font::read(bytes).map_err(TextError::Font)?;
    let words = crate::richtext::read(material)?;
    let gathered = gather(font, &words.text, &words.runs, Wanted::of(material), faces);
    layout_faces(&gathered.fonts, material, &|source| gathered.pick(source))
}

/// [`layout_with`], keeping the layout in `cache`: while the words, the
/// faces and the sizes are the same, it is not made again.
pub fn layout_cached<'f>(
    bytes: &'f [u8],
    material: &MaterialText,
    faces: &'f dyn FaceSource,
    cache: &TextCache,
) -> Result<std::sync::Arc<Layout>, TextError> {
    let font = cutlass_font::read(bytes).map_err(TextError::Font)?;
    let words = crate::richtext::read(material)?;
    let gathered = gather(font, &words.text, &words.runs, Wanted::of(material), faces);
    let pick = |source: usize| gathered.pick(source);
    let key = layout_key(&gathered.fonts, material, &words.text, &words.runs, &pick);
    cache.layout(key, || layout_faces(&gathered.fonts, material, &pick))
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
            turned: false,
        });
    }
    // Letters spaced apart are not drawn as ligatures.
    let shaped = shape_faces(fonts, text, pick, letter_px == 0.0)?;
    let breaks = boundaries(text);
    // A line is cut only before the first glyph of a character.
    let boundary = |glyph: &ShapedGlyph| {
        if glyph.first {
            breaks
                .get(glyph.source)
                .copied()
                .unwrap_or(Boundary::Joined)
        } else {
            Boundary::Joined
        }
    };
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
    let vertical = matches!(
        alignment,
        Alignment::Top | Alignment::Middle | Alignment::Bottom
    );
    // Mongolian and Phags-pa letters join along their line and cannot be
    // stood one under another: their columns are lines turned.
    let turned = vertical && crate::script::written_down_the_page(text);
    let (alignment, vertical) = if turned {
        let across = match alignment {
            Alignment::Middle => Alignment::Center,
            Alignment::Bottom => Alignment::Right,
            _ => Alignment::Left,
        };
        (across, false)
    } else {
        (alignment, vertical)
    };
    let mut laid = if vertical {
        // One em for each character, however many glyphs draw it.
        let em = |glyph: &ShapedGlyph| {
            if glyph.first {
                font_size * size(glyph)
            } else {
                0.0
            }
        };
        let measure = Measure {
            width,
            letter_px,
            length: &em,
            boundary: &boundary,
        };
        let rows = wrap(&shaped, &measure)?;
        place_columns(&rows, metrics, alignment, &size, &units)?
    } else {
        #[allow(clippy::cast_precision_loss)]
        let advance = |glyph: &ShapedGlyph| glyph.advance as f32 * units(glyph);
        let measure = Measure {
            width,
            letter_px,
            length: &advance,
            boundary: &boundary,
        };
        let rows = wrap(&shaped, &measure)?;
        place(&shaped, &rows, metrics, alignment, turned, &size, &units)?
    };
    laid.turned = turned;
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

/// How glyphs are measured along a line, or down a column.
struct Measure<'a> {
    /// The most a line holds; zero does not wrap.
    width: f32,
    /// The space put between two characters.
    letter_px: f32,
    /// A glyph's own length: its advance, or one em down a column.
    length: &'a dyn Fn(&ShapedGlyph) -> f32,
    /// What may happen to the line before a glyph.
    boundary: &'a dyn Fn(&ShapedGlyph) -> Boundary,
}

/// One line's glyphs, in the order the text is stored.
#[derive(Default)]
struct Row {
    /// Which paragraph of the shaped text it is a line of.
    paragraph: usize,
    glyphs: Vec<ShapedGlyph>,
    /// Byte index where its text ends.
    end: usize,
    /// The line went on to the next for want of room, so the spaces at its
    /// end hang.
    wrapped: bool,
    /// Where the next glyph starts.
    pen: f32,
    /// Where the last glyph that is not a space ends.
    ink: f32,
    /// Glyphs from the start known to hold no place to cut, so a row that
    /// cannot be cut is not searched again for each glyph it takes.
    searched: usize,
}

impl Row {
    fn push(&mut self, glyph: ShapedGlyph, measure: &Measure<'_>) -> Result<(), TextError> {
        let advance = (measure.length)(&glyph);
        let lead = if self.glyphs.is_empty() || !glyph.first {
            0.0
        } else {
            measure.letter_px
        };
        let pen = self.pen + lead + advance;
        if !pen.is_finite() || !advance.is_finite() {
            return Err(TextError::Limit);
        }
        self.pen = pen;
        if !glyph.character.is_whitespace() {
            self.ink = pen;
        }
        self.glyphs.push(glyph);
        Ok(())
    }

    /// Where to cut a row that is too long: before the last glyph a line
    /// may start at, or, when it is one long word, the last glyph that is
    /// not joined to the one before it. The first glyph is never the cut.
    fn cut(&mut self, measure: &Measure<'_>) -> Option<usize> {
        let mut inside = None;
        let from = self.searched.max(1);
        for (index, glyph) in self.glyphs.iter().enumerate().skip(from).rev() {
            match (measure.boundary)(glyph) {
                Boundary::Allowed | Boundary::Mandatory => return Some(index),
                Boundary::Inside if inside.is_none() => inside = Some(index),
                Boundary::Inside | Boundary::Joined => {}
            }
        }
        if inside.is_none() {
            self.searched = self.glyphs.len();
        }
        inside
    }

    /// How many glyphs at its end hang past the line: the spaces a row
    /// wrapped after.
    fn hanging(&self) -> usize {
        if !self.wrapped {
            return 0;
        }
        self.glyphs
            .iter()
            .rev()
            .take_while(|glyph| glyph.character.is_whitespace())
            .count()
    }
}

/// Fill rows from the shaped paragraphs, starting a new one for each
/// paragraph and wherever the row would pass the width.
fn wrap(shaped: &Shaped, measure: &Measure<'_>) -> Result<Vec<Row>, TextError> {
    let mut rows: Vec<Row> = Vec::new();
    for (index, paragraph) in shaped.paragraphs.iter().enumerate() {
        rows.push(Row {
            paragraph: index,
            end: paragraph.end(),
            ..Row::default()
        });
        for glyph in &paragraph.glyphs {
            let Some(row) = rows.last_mut() else {
                return Err(TextError::Limit);
            };
            row.push(glyph.clone(), measure)?;
            if measure.width > 0.0 {
                rewrap(&mut rows, measure)?;
            }
        }
    }
    Ok(rows)
}

/// Move what the last row cannot hold onto new rows, until the last fits
/// or cannot be cut.
fn rewrap(rows: &mut Vec<Row>, measure: &Measure<'_>) -> Result<(), TextError> {
    loop {
        let Some(row) = rows.last_mut() else {
            return Err(TextError::Limit);
        };
        if row.ink <= measure.width {
            return Ok(());
        }
        let Some(at) = row.cut(measure) else {
            return Ok(());
        };
        let mut tail = row.glyphs.split_off(at);
        // Kerning does not reach from one line to the next.
        if let Some(first) = tail.first_mut() {
            first.unpair(&mut row.glyphs);
        }
        let mut next = Row {
            paragraph: row.paragraph,
            end: row.end,
            ..Row::default()
        };
        row.wrapped = true;
        if let Some(first) = tail.first() {
            row.end = first.source;
        }
        for glyph in tail {
            next.push(glyph, measure)?;
        }
        rows.push(next);
    }
}

/// Lay `rows` one under another; with `upward`, the first lowest and
/// each later one above it, each still as tall as it is.
fn place(
    shaped: &Shaped,
    rows: &[Row],
    metrics: Metrics,
    alignment: Alignment,
    upward: bool,
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
    // Each row's baseline, counted down from the top in the order the
    // rows stand.
    let mut baselines = vec![0.0_f32; rows.len()];
    let mut baseline = 0.0_f32;
    let mut stand = |index: usize, first: bool| {
        let tallest = rows[index].glyphs.iter().map(size).fold(1.0_f32, f32::max);
        baseline += if first { font_size } else { step } * tallest;
        baselines[index] = baseline;
    };
    if upward {
        for (place, index) in (0..rows.len()).rev().enumerate() {
            stand(index, place == 0);
        }
    } else {
        for index in 0..rows.len() {
            stand(index, index == 0);
        }
    }
    for (row, baseline) in rows.iter().zip(baselines) {
        let levels = shaped
            .paragraphs
            .get(row.paragraph)
            .map(|paragraph| paragraph.line_levels(&row.glyphs, row.end))
            .unwrap_or_default();
        let (glyphs, line_width) = place_line(row, &levels, letter_px, size, units)?;
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
        turned: false,
    })
}

/// Stand each column's characters one under the next, centered across
/// the column, the first column on the right. A character's glyphs share
/// its place: a mark stays on its letter.
fn place_columns(
    rows: &[Row],
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
            let widest = row.glyphs.iter().map(size).fold(1.0_f32, f32::max);
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
        let hanging = row.glyphs.len() - row.hanging();
        let mut glyphs = Vec::with_capacity(row.glyphs.len());
        let mut pen = 0.0_f32;
        let mut ink = 0.0_f32;
        // The character being stood: its left edge and its baseline.
        let (mut left, mut along, mut down) = (middle, 0.0_f32, 0.0_f32);
        for (index, glyph) in row.glyphs.iter().enumerate() {
            let times = size(glyph);
            let scale = units(glyph);
            #[allow(clippy::cast_precision_loss)]
            let (advance, x_offset, y_offset) = (
                glyph.advance as f32 * scale,
                glyph.x_offset as f32 * scale,
                glyph.y_offset as f32 * scale,
            );
            if glyph.first || index == 0 {
                let em = font_size * times;
                if index > 0 {
                    pen += letter_px;
                }
                left = middle - advance / 2.0;
                along = 0.0;
                down = pen + em * VERTICAL_BASELINE;
                pen += em;
                if index < hanging {
                    ink = pen;
                }
            }
            let (x, y) = (left + along + x_offset, down);
            along += advance;
            if ![x, y, advance, y_offset]
                .iter()
                .all(|value| value.is_finite())
            {
                return Err(TextError::Limit);
            }
            glyphs.push(PlacedGlyph {
                source: glyph.source,
                character: glyph.character,
                glyph_id: glyph.glyph_id,
                missing: glyph.missing,
                x,
                y,
                lift: y_offset,
                advance,
                size: times,
                face: glyph.face,
            });
        }
        let baseline = glyphs
            .first()
            .map_or(font_size * VERTICAL_BASELINE, |glyph| glyph.y);
        lines.push(Line {
            glyphs,
            width: if row.wrapped { ink } else { pen },
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
        turned: false,
    })
}

/// The row's glyphs along its line, in the order they are seen, and the
/// line's width: up to its last glyph, or, for a row that wrapped, without
/// the spaces that hang at its end.
fn place_line(
    row: &Row,
    levels: &[u8],
    letter_px: f32,
    size: &dyn Fn(&ShapedGlyph) -> f32,
    units: &dyn Fn(&ShapedGlyph) -> f32,
) -> Result<(Vec<PlacedGlyph>, f32), TextError> {
    let hanging = row.glyphs.len() - row.hanging();
    let mut glyphs = Vec::with_capacity(row.glyphs.len());
    let mut pen = 0.0_f32;
    // Where the glyphs that do not hang start and end.
    let mut ink: Option<(f32, f32)> = None;
    let mut before: Option<usize> = None;
    for index in visual_order(levels) {
        let Some(glyph) = row.glyphs.get(index) else {
            return Err(TextError::Limit);
        };
        let times = size(glyph);
        let scale = units(glyph);
        #[allow(clippy::cast_precision_loss)]
        let (advance, x_offset, y_offset) = (
            glyph.advance as f32 * scale,
            glyph.x_offset as f32 * scale,
            glyph.y_offset as f32 * scale,
        );
        // Space goes between characters, not between one's own glyphs.
        if before.is_some_and(|source| source != glyph.source) {
            pen += letter_px;
        }
        before = Some(glyph.source);
        let x = pen + x_offset;
        if ![x, advance, y_offset].iter().all(|value| value.is_finite()) {
            return Err(TextError::Limit);
        }
        glyphs.push(PlacedGlyph {
            source: glyph.source,
            character: glyph.character,
            glyph_id: glyph.glyph_id,
            missing: glyph.missing,
            x,
            y: 0.0,
            lift: y_offset,
            advance,
            size: times,
            face: glyph.face,
        });
        let start = pen;
        pen += advance;
        if index < hanging {
            ink = Some(ink.map_or((start, pen), |(from, to)| (from.min(start), to.max(pen))));
        }
    }
    if !pen.is_finite() {
        return Err(TextError::Limit);
    }
    if !row.wrapped {
        return Ok((glyphs, pen));
    }
    // The spaces that hang sit outside the line, on whichever side the
    // paragraph ends.
    let (from, to) = ink.unwrap_or((0.0, 0.0));
    for glyph in &mut glyphs {
        glyph.x -= from;
    }
    Ok((glyphs, to - from))
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

    fn words(laid: &super::Layout) -> Vec<String> {
        laid.lines
            .iter()
            .map(|line| line.glyphs.iter().map(|glyph| glyph.character).collect())
            .collect()
    }

    #[test]
    fn a_line_wraps_after_a_word_and_its_spaces_hang() {
        let face = font();
        // "AA " is 1200 of letters and a space of 250.
        let mut text = material("AA AA AA");
        text.line_max_width = 1_500.0;
        let laid = layout(&face, &text).expect("wrap");
        assert_eq!(words(&laid), ["AA ", "AA ", "AA"]);
        assert_eq!(laid.lines[0].width.to_bits(), 1_200.0_f32.to_bits());
        assert_eq!(laid.lines[1].glyphs[0].x.to_bits(), 0.0_f32.to_bits());
        assert_eq!(laid.lines[2].width.to_bits(), 1_200.0_f32.to_bits());

        // The hanging space does not push a centered line off its middle.
        text.content = "AA AA".to_owned();
        text.line_max_width = 1_300.0;
        text.alignment = 1;
        let centered = layout(&face, &text).expect("center");
        assert_eq!(words(&centered), ["AA ", "AA"]);
        assert_eq!(centered.lines[0].glyphs[0].x.to_bits(), 50.0_f32.to_bits());
        assert_eq!(centered.lines[1].glyphs[0].x.to_bits(), 50.0_f32.to_bits());

        // Spaces alone never start a new line, and a width that holds
        // everything changes nothing.
        text.content = "AA    ".to_owned();
        text.alignment = 0;
        assert_eq!(words(&layout(&face, &text).expect("spaces")), ["AA    "]);
        text.content = "AA AA".to_owned();
        text.line_max_width = 2_650.0;
        let whole = layout(&face, &text).expect("whole");
        assert_eq!(words(&whole), ["AA AA"]);
        assert_eq!(whole.lines[0].width.to_bits(), 2_650.0_f32.to_bits());
    }

    #[test]
    fn a_word_is_cut_only_when_it_alone_is_too_wide() {
        let face = font();
        let mut text = material("A AAAA");
        text.line_max_width = 1_500.0;
        let laid = layout(&face, &text).expect("long word");
        assert_eq!(words(&laid), ["A ", "AA", "AA"]);

        // A letter keeps its accent: the cut falls before the letter.
        text.content = "AA\u{301}".to_owned();
        text.line_max_width = 1_000.0;
        let accented = layout(&face, &text).expect("accent");
        let sources: Vec<Vec<usize>> = accented
            .lines
            .iter()
            .map(|line| line.glyphs.iter().map(|glyph| glyph.source).collect())
            .collect();
        assert_eq!(sources, [vec![0], vec![1, 1]]);
        // A flag is not cut between its halves, however narrow the line.
        text.content = "\u{1F1EC}\u{1F1ED}".to_owned();
        text.line_max_width = 600.0;
        assert_eq!(layout(&face, &text).expect("flag").lines.len(), 1);
        // A hyphen and a break the string holds both end a line.
        text.content = "AA-AA\nA".to_owned();
        text.line_max_width = 1_800.0;
        assert_eq!(
            words(&layout(&face, &text).expect("hyphen")),
            ["AA-", "AA", "A"]
        );
    }

    #[test]
    fn right_to_left_runs_are_placed_in_the_order_they_are_seen() {
        let face = font();
        // Hebrew letters are missing from the fixture: 500 wide each.
        let laid = layout(&face, &material("AB \u{5D0}\u{5D1}")).expect("mixed");
        assert_eq!(words(&laid), ["AB \u{5D1}\u{5D0}"]);
        let xs: Vec<f32> = laid.lines[0].glyphs.iter().map(|glyph| glyph.x).collect();
        assert_eq!(xs, [0.0, 520.0, 1_220.0, 1_470.0, 1_970.0]);
        // Each glyph still names the byte its character is stored at.
        let sources: Vec<usize> = laid.lines[0].glyphs.iter().map(|g| g.source).collect();
        assert_eq!(sources, [0, 1, 2, 5, 3]);

        // A paragraph that starts right to left runs that way: its Latin
        // word is seen first, and kept in its own order.
        let hebrew = layout(&face, &material("\u{5D0} AB")).expect("hebrew");
        assert_eq!(words(&hebrew), ["AB \u{5D0}"]);
        assert_eq!(hebrew.lines[0].width.to_bits(), 1_970.0_f32.to_bits());

        // Each paragraph takes its own direction.
        let two = layout(&face, &material("\u{5D0}\u{5D1}\nAB")).expect("two");
        assert_eq!(words(&two), ["\u{5D1}\u{5D0}", "AB"]);

        // A direction override turns Latin around, and draws nothing itself.
        let turned = layout(&face, &material("\u{202E}AB")).expect("override");
        // The pair is kerned as it is seen: B then A.
        assert_eq!(words(&turned), ["BA"]);
        assert_eq!(turned.lines[0].glyphs[1].x.to_bits(), 740.0_f32.to_bits());
    }

    #[test]
    fn a_right_to_left_line_wraps_in_reading_order_and_hangs_its_space_left() {
        let face = font();
        let mut text = material("\u{5D0}\u{5D1} \u{5D2}");
        text.line_max_width = 1_200.0;
        let laid = layout(&face, &text).expect("wrap");
        // The first two letters read are on the first line.
        assert_eq!(words(&laid), [" \u{5D1}\u{5D0}", "\u{5D2}"]);
        let first = &laid.lines[0];
        assert_eq!(first.width.to_bits(), 1_000.0_f32.to_bits());
        let xs: Vec<f32> = first.glyphs.iter().map(|glyph| glyph.x).collect();
        assert_eq!(xs, [-250.0, 0.0, 500.0]);
        // Aligned right, the letters end at the box's edge.
        text.alignment = 2;
        let right = layout(&face, &text).expect("right");
        assert_eq!(right.lines[0].glyphs[2].x.to_bits(), 700.0_f32.to_bits());
        assert_eq!(right.lines[1].glyphs[0].x.to_bits(), 700.0_f32.to_bits());
    }

    const SHAPING: &[u8] = include_bytes!("../../font/fixtures/shaping.ttf");

    fn shaping() -> cutlass_font::Font<'static> {
        cutlass_font::read(SHAPING).expect("fixture")
    }

    fn ids(laid: &super::Layout) -> Vec<Vec<u16>> {
        laid.lines
            .iter()
            .map(|line| line.glyphs.iter().map(|glyph| glyph.glyph_id).collect())
            .collect()
    }

    #[test]
    fn shaped_glyphs_are_placed_with_their_offsets() {
        let face = shaping();
        // Three Arabic letters, joined, seen from the last to the first.
        let arabic = layout(&face, &material("\u{628}\u{628}\u{628}")).expect("arabic");
        assert_eq!(ids(&arabic), [[13, 12, 11]]);
        let line = &arabic.lines[0];
        let xs: Vec<f32> = line.glyphs.iter().map(|glyph| glyph.x).collect();
        assert_eq!(xs, [0.0, 800.0, 1_200.0]);
        let sources: Vec<usize> = line.glyphs.iter().map(|glyph| glyph.source).collect();
        assert_eq!(sources, [4, 2, 0]);
        assert_eq!(line.width.to_bits(), 1_600.0_f32.to_bits());

        // An accent sits on its letter: back over it, and 50 lower than
        // the face draws it alone.
        let accent = layout(&face, &material("A\u{301}V")).expect("accent");
        let glyphs = &accent.lines[0].glyphs;
        assert_eq!(ids(&accent), [[2, 7, 3]]);
        assert_eq!((glyphs[1].x, glyphs[1].lift), (450.0, -50.0));
        assert_eq!((glyphs[1].source, glyphs[1].advance), (0, 0.0));
        assert_eq!((glyphs[0].lift, glyphs[2].x), (0.0, 600.0));
        // The accent's letter and the next are still a kerned pair.
        assert_eq!(
            layout(&face, &material("AV")).expect("pair").lines[0].width,
            1_100.0
        );

        // A vowel sign written before its consonant is drawn before it,
        // and both are the one character.
        let sign = layout(&face, &material("\u{915}\u{93F}")).expect("sign");
        assert_eq!(ids(&sign), [[27, 22]]);
        assert!(sign.lines[0].glyphs.iter().all(|glyph| glyph.source == 0));
    }

    #[test]
    fn ligatures_stay_whole_and_spacing_undoes_them() {
        let face = shaping();
        // Each ligature is one glyph, and a line is not cut inside one.
        let mut text = material("fifi");
        text.line_max_width = 600.0;
        let laid = layout(&face, &text).expect("ligatures");
        assert_eq!(ids(&laid), [[6], [6]]);
        // Letters spaced apart are drawn one by one, the space between.
        let mut spaced = material("fi");
        spaced.letter_spacing = 0.1;
        let laid = layout(&face, &spaced).expect("spaced");
        assert_eq!(ids(&laid), [[4, 5]]);
        assert_eq!(laid.lines[0].glyphs[1].x.to_bits(), 400.0_f32.to_bits());
        // A sign and its consonant stay on one line however narrow it is,
        // and stand in one place down a column.
        let mut narrow = material("\u{915}\u{93F}\u{915}\u{93F}");
        narrow.line_max_width = 100.0;
        assert_eq!(
            ids(&layout(&face, &narrow).expect("narrow")),
            [[27, 22], [27, 22]]
        );
        narrow.typesetting = 1;
        narrow.line_max_width = 0.0;
        let column = layout(&face, &narrow).expect("column");
        let ys: Vec<f32> = column.lines[0].glyphs.iter().map(|glyph| glyph.y).collect();
        assert_eq!(ys, [800.0, 800.0, 1_800.0, 1_800.0]);
    }

    #[test]
    fn a_row_that_cannot_be_cut_is_searched_once() {
        let face = font();
        // One letter and marks that all belong to it: no place to cut.
        let mut text = material(&format!("A{}", "\u{301}".repeat(30_000)));
        text.line_max_width = 700.0;
        let started = std::time::Instant::now();
        let laid = layout(&face, &text).expect("marks");
        assert_eq!(laid.lines.len(), 1);
        assert_eq!(laid.glyphs().count(), 30_001);
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
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
    fn a_script_written_down_the_page_is_laid_across_to_be_turned() {
        const JOINED: &[u8] = include_bytes!("../../font/fixtures/joined.ttf");
        let face = cutlass_font::read(JOINED).expect("face");
        // Three Mongolian letters, a line end, and one more.
        let mut text = material("\u{1820}\u{1820}\u{1820}\n\u{1820}");
        text.typesetting = 1;
        text.alignment = 3;
        let laid = layout(&face, &text).expect("columns");
        // Not letters stood one under another: lines, to be turned.
        assert!(laid.turned && !laid.vertical());
        assert_eq!(laid.alignment, super::Alignment::Left);
        assert_eq!(laid.lines.len(), 2);
        // The same lines as the text laid across, joined as it joins.
        let mut across = text.clone();
        across.typesetting = 0;
        across.alignment = 0;
        let flat = layout(&face, &across).expect("lines");
        assert!(!flat.turned);
        for (turned, plain) in laid.lines.iter().zip(&flat.lines) {
            let ids = |line: &super::Line| -> Vec<(u16, u32)> {
                line.glyphs
                    .iter()
                    .map(|glyph| (glyph.glyph_id, glyph.x.to_bits()))
                    .collect()
            };
            assert_eq!(ids(turned), ids(plain));
        }
        // But the first line is the lowest, to be the leftmost column
        // once the picture is turned a quarter clockwise.
        assert_eq!(flat.lines[0].baseline.to_bits(), 1_000.0_f32.to_bits());
        assert_eq!(flat.lines[1].baseline.to_bits(), 2_000.0_f32.to_bits());
        assert_eq!(laid.lines[1].baseline.to_bits(), 1_000.0_f32.to_bits());
        assert_eq!(laid.lines[0].baseline.to_bits(), 2_000.0_f32.to_bits());
        assert!(
            laid.lines[0]
                .glyphs
                .iter()
                .all(|glyph| glyph.y.to_bits() == 2_000.0_f32.to_bits())
        );
        // Top, middle and bottom of the column are the line's left,
        // center and right.
        text.alignment = 4;
        assert_eq!(
            layout(&face, &text).expect("middle").alignment,
            super::Alignment::Center
        );
        text.alignment = 5;
        assert_eq!(
            layout(&face, &text).expect("bottom").alignment,
            super::Alignment::Right
        );
        // A sign or a digit before the first letter does not decide, and
        // Latin letters first do: those stand one under another.
        text.content = "1. \u{1820}".to_owned();
        assert!(layout(&face, &text).expect("columns").turned);
        let latin = font();
        let mut upright = material("AB");
        upright.typesetting = 1;
        upright.alignment = 3;
        let stood = layout(&latin, &upright).expect("stood");
        assert!(stood.vertical() && !stood.turned);
        // Lines across are never turned.
        assert!(!layout(&face, &across).expect("lines").turned);
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

        // A column ends after a word too, and its space hangs below.
        text.content = "AA AA".to_owned();
        text.line_max_width = 2_500.0;
        let columns = layout(&face, &text).expect("columns");
        assert_eq!(words(&columns), ["AA ", "AA"]);
        assert_eq!(columns.lines[0].width.to_bits(), 2_000.0_f32.to_bits());
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
