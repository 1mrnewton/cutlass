//! A short text's letters as outlines, for a mask cut in the shape of the
//! words. Each paragraph is one line, centred under the one before; a
//! right-to-left paragraph is set from its right. Curves are cut into
//! short straight pieces, fine enough for a mask the size of the screen.

use crate::{TextError, shape::shape};

/// Straight pieces each curve is cut into.
const PIECES: usize = 4;
/// A line's height, in ems.
const LINE: f32 = 1.2;

/// Closed polygons in ems, y up, and the box round them as `[x0, y0, x1,
/// y1]`.
pub type TextOutlines = (Vec<Vec<[f32; 2]>>, [f32; 4]);

/// `text`'s letters in the face in `bytes`: closed polygons in ems, y up,
/// filled by the non-zero rule, and the box round them as `[x0, y0, x1,
/// y1]`. A text with nothing to draw has no polygons and an empty box.
pub fn text_outlines(bytes: &[u8], text: &str) -> Result<TextOutlines, TextError> {
    let font = cutlass_font::read(bytes).map_err(TextError::Font)?;
    let shaped = shape(&font, text)?;
    #[allow(clippy::cast_precision_loss)]
    let em = f32::from(font.units_per_em().max(1));
    let mut lines = Vec::new();
    for paragraph in &shaped.paragraphs {
        let mut polygons = Vec::new();
        let mut pen = 0_f32;
        let glyphs: Vec<_> = if paragraph.base % 2 == 1 {
            paragraph.glyphs.iter().rev().collect()
        } else {
            paragraph.glyphs.iter().collect()
        };
        for glyph in glyphs {
            #[allow(clippy::cast_precision_loss)]
            let (dx, dy) = (pen + glyph.x_offset as f32, glyph.y_offset as f32);
            let outline = font.outline(glyph.glyph_id).map_err(TextError::Font)?;
            for contour in &outline.contours {
                let polygon = flatten(contour, dx, dy, em);
                if polygon.len() >= 3 {
                    polygons.push(polygon);
                }
            }
            #[allow(clippy::cast_precision_loss)]
            {
                pen += glyph.advance as f32;
            }
        }
        lines.push((polygons, pen / em));
    }
    let mut all = Vec::new();
    #[allow(clippy::cast_precision_loss)]
    for (row, (polygons, width)) in lines.into_iter().enumerate() {
        let (shift, drop) = (-width / 2.0, row as f32 * LINE);
        all.extend(polygons.into_iter().map(|polygon| {
            polygon
                .into_iter()
                .map(|[x, y]| [x + shift, y - drop])
                .collect::<Vec<_>>()
        }));
    }
    let mut bounds = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
    for [x, y] in all.iter().flatten() {
        bounds = [
            bounds[0].min(*x),
            bounds[1].min(*y),
            bounds[2].max(*x),
            bounds[3].max(*y),
        ];
    }
    if all.is_empty() {
        bounds = [0.0; 4];
    }
    Ok((all, bounds))
}

/// One contour as a polygon in ems, moved by `(dx, dy)` font units.
fn flatten(contour: &cutlass_font::Contour, dx: f32, dy: f32, em: f32) -> Vec<[f32; 2]> {
    let at = |x: f32, y: f32| [(x + dx) / em, (y + dy) / em];
    let mut points = vec![at(contour.x, contour.y)];
    let mut from = (contour.x, contour.y);
    for segment in &contour.segments {
        match *segment {
            cutlass_font::Segment::Line { x, y } => {
                points.push(at(x, y));
                from = (x, y);
            }
            cutlass_font::Segment::Quad { cx, cy, x, y } => {
                for step in 1..=PIECES {
                    #[allow(clippy::cast_precision_loss)]
                    let t = step as f32 / PIECES as f32;
                    let u = 1.0 - t;
                    points.push(at(
                        u * u * from.0 + 2.0 * u * t * cx + t * t * x,
                        u * u * from.1 + 2.0 * u * t * cy + t * t * y,
                    ));
                }
                from = (x, y);
            }
            cutlass_font::Segment::Cubic {
                c1x,
                c1y,
                c2x,
                c2y,
                x,
                y,
            } => {
                for step in 1..=PIECES {
                    #[allow(clippy::cast_precision_loss)]
                    let t = step as f32 / PIECES as f32;
                    let u = 1.0 - t;
                    let (a, b, c, d) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
                    points.push(at(
                        a * from.0 + b * c1x + c * c2x + d * x,
                        a * from.1 + b * c1y + c * c2y + d * y,
                    ));
                }
                from = (x, y);
            }
        }
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;

    const TTF: &[u8] = include_bytes!("../../font/fixtures/sample.ttf");

    #[test]
    fn a_words_letters_are_closed_outlines_round_the_middle() {
        let (polygons, [x0, y0, x1, y1]) = text_outlines(TTF, "AB").unwrap();
        assert!(!polygons.is_empty());
        assert!(polygons.iter().all(|polygon| polygon.len() >= 3));
        // Centred across, standing on the baseline.
        assert!(x0 < 0.0 && x1 > 0.0, "{x0} {x1}");
        assert!((x0 + x1).abs() < (x1 - x0) * 0.5);
        assert!(y1 > 0.3 && y0 > -0.5, "{y0} {y1}");
        // A second line sits under the first.
        let (_, two) = text_outlines(TTF, "AB\nAB").unwrap();
        assert!(two[1] < y0 - 0.5);
        let (none, empty) = text_outlines(TTF, "").unwrap();
        assert!(none.is_empty());
        assert_eq!(empty, [0.0; 4]);
    }
}
