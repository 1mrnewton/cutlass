//! A picture of glyphs that are already placed.
//!
//! The caption sampler chooses the glyphs, the fill, and a pixel shift.
//! This module paints them with the same atlas as a text run. It does not
//! read a draft. The picture bounds are the unshifted glyphs, so a later
//! center placement keeps that shift. Ink that leaves the box is clipped.

use cutlass_picture::{CpuBytes, Picture, rgba_len};

use crate::TextError;
use crate::atlas::Atlas;
use crate::layout::PlacedGlyph;
use crate::mesh::{GlyphRecord, glyph_quad};
use crate::source::{Blit, blit};

/// One glyph to paint. `shift_x` and `shift_y` are pixels, right and down.
pub struct DrawnGlyph<'a> {
    pub placed: &'a PlacedGlyph,
    pub baseline: f32,
    pub fill: [u8; 4],
    pub shift_x: f32,
    pub shift_y: f32,
}

/// Paint `glyphs` with the face in `bytes` at `font_size`.
///
/// An empty list is a clear 1×1 picture. A face that cannot be read, or a
/// glyph that cannot be placed, is an error.
pub fn paint_glyphs(
    bytes: &[u8],
    font_size: f32,
    glyphs: &[DrawnGlyph<'_>],
) -> Result<Picture, TextError> {
    if !font_size.is_finite() || font_size <= 0.0 {
        return Err(TextError::Measure);
    }
    if glyphs.is_empty() {
        let bytes = CpuBytes::new(1, 1, vec![0, 0, 0, 0]).map_err(|_| TextError::Limit)?;
        return Ok(Picture::Cpu(bytes));
    }
    let font = cutlass_font::read(bytes).map_err(TextError::Font)?;
    let scale = font_size / f32::from(font.units_per_em());
    if !scale.is_finite() {
        return Err(TextError::Measure);
    }
    let mut atlas = Atlas::new();
    let mut ink = Vec::with_capacity(glyphs.len());
    for glyph in glyphs {
        let _image = atlas.glyph(&font, glyph.placed.glyph_id)?;
        let image = atlas.image(glyph.placed.glyph_id).ok_or(TextError::Limit)?;
        let (x, y, width, height) =
            glyph_quad(glyph.placed, glyph.baseline, font_size, scale, image)?;
        if ![x, y, glyph.shift_x, glyph.shift_y]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(TextError::Limit);
        }
        ink.push(InkGlyph {
            record: GlyphRecord {
                glyph_id: glyph.placed.glyph_id,
                missing: glyph.placed.missing,
                source: glyph.placed.source,
                line: 0,
                x,
                y,
                width,
                height,
            },
            fill: glyph.fill,
            shift_x: glyph.shift_x,
            shift_y: glyph.shift_y,
        });
    }
    let (origin_x, origin_y, width, height) = bounds(&ink)?;
    let len = rgba_len(width, height).ok_or(TextError::Limit)?;
    let mut rgba = vec![0_u8; len];
    for glyph in &ink {
        let Some(image) = atlas.image(glyph.record.glyph_id) else {
            continue;
        };
        let mut record = glyph.record.clone();
        record.x += glyph.shift_x;
        record.y += glyph.shift_y;
        blit(Blit {
            rgba: &mut rgba,
            picture_w: width,
            picture_h: height,
            record: &record,
            image,
            fill: glyph.fill,
            origin_x,
            origin_y,
        });
    }
    let bytes = CpuBytes::new(width, height, rgba).map_err(|_| TextError::Limit)?;
    Ok(Picture::Cpu(bytes))
}

struct InkGlyph {
    record: GlyphRecord,
    fill: [u8; 4],
    shift_x: f32,
    shift_y: f32,
}

fn bounds(ink: &[InkGlyph]) -> Result<(f32, f32, u32, u32), TextError> {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    let mut any = false;
    for glyph in ink {
        let record = &glyph.record;
        if record.width <= 0.0 || record.height <= 0.0 {
            continue;
        }
        any = true;
        min_x = min_x.min(record.x);
        min_y = min_y.min(record.y);
        max_x = max_x.max(record.x + record.width);
        max_y = max_y.max(record.y + record.height);
    }
    if !any {
        return Ok((0.0, 0.0, 1, 1));
    }
    let width = edge(max_x - min_x)?;
    let height = edge(max_y - min_y)?;
    if rgba_len(width, height).is_none() {
        return Err(TextError::Limit);
    }
    Ok((min_x, min_y, width, height))
}

fn edge(value: f32) -> Result<u32, TextError> {
    if !value.is_finite() || value <= 0.0 || value > 8_192.0 {
        return Err(TextError::Limit);
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let edge = value.ceil() as u32;
    if edge == 0 { Ok(1) } else { Ok(edge) }
}

#[cfg(test)]
mod tests {
    use super::{DrawnGlyph, paint_glyphs};
    use crate::layout::layout_text;
    use cutlass_model::MaterialText;
    use cutlass_picture::Picture;

    const TTF: &[u8] = include_bytes!("../../font/fixtures/sample.ttf");

    fn material() -> MaterialText {
        let mut text = MaterialText::new("glyph");
        text.content = "A".to_owned();
        text.font_size = 64.0;
        text
    }

    fn opaque_xs(rgba: &[u8], width: u32) -> (u32, u32) {
        let mut min = width;
        let mut max = 0_u32;
        for (index, pixel) in rgba.chunks(4).enumerate() {
            if pixel == [255, 0, 0, 255] {
                let x = u32::try_from(index).unwrap_or(0) % width;
                min = min.min(x);
                max = max.max(x);
            }
        }
        (min, max)
    }

    #[test]
    fn a_shift_moves_the_ink_and_an_empty_list_is_clear() {
        let text = material();
        let layout = layout_text(TTF, &text).expect("layout");
        let line = layout.lines.first().expect("line");
        let placed = line.glyphs.first().expect("glyph");
        let drawn = DrawnGlyph {
            placed,
            baseline: line.baseline,
            fill: [255, 0, 0, 255],
            shift_x: 0.0,
            shift_y: 0.0,
        };
        let Picture::Cpu(still) = paint_glyphs(TTF, layout.font_size, &[drawn]).expect("paint")
        else {
            panic!("platform");
        };
        let (left, _) = opaque_xs(still.rgba(), still.width());
        assert!(left < still.width());

        let moved = DrawnGlyph {
            placed,
            baseline: line.baseline,
            fill: [255, 0, 0, 255],
            shift_x: 24.0,
            shift_y: 0.0,
        };
        let Picture::Cpu(shifted) = paint_glyphs(TTF, layout.font_size, &[moved]).expect("shift")
        else {
            panic!("platform");
        };
        let (shifted_left, _) = opaque_xs(shifted.rgba(), shifted.width());
        assert_eq!(shifted.width(), still.width());
        assert!(shifted_left > left);

        let Picture::Cpu(empty) = paint_glyphs(TTF, 64.0, &[]).expect("empty") else {
            panic!("platform");
        };
        assert_eq!(empty.rgba(), &[0, 0, 0, 0]);
    }
}
