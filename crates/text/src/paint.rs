//! A picture of glyphs that are already placed.
//!
//! The caption sampler chooses the glyphs, the fill, and a pixel shift.
//! This module paints them with the same atlas as a text run. It does not
//! read a draft. The picture is centred on the unshifted glyphs, so a later
//! center placement keeps a shift; it grows evenly about that centre to
//! hold glyphs that are shifted, sized, or boxed.

use cutlass_picture::{CpuBytes, Picture, rgba_len};

use crate::TextError;
use crate::atlas::Atlas;
use crate::layout::PlacedGlyph;
use crate::mesh::{GlyphRecord, glyph_quad};
use crate::source::{Blit, blit};

/// One glyph to paint. `shift_x` and `shift_y` are pixels, right and down;
/// `scale` sizes it about its own middle. A `back` that is not clear is a
/// box behind it, joined with its neighbours' boxes of the same color on
/// its line into one round box.
pub struct DrawnGlyph<'a> {
    pub placed: &'a PlacedGlyph,
    pub baseline: f32,
    pub fill: [u8; 4],
    pub shift_x: f32,
    pub shift_y: f32,
    pub scale: f32,
    /// Straight RGBA.
    pub back: [u8; 4],
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
        let scale = if glyph.scale.is_finite() && glyph.scale > 0.0 {
            glyph.scale
        } else {
            1.0
        };
        let (sized_w, sized_h) = (width * scale, height * scale);
        ink.push(InkGlyph {
            record: GlyphRecord {
                glyph_id: glyph.placed.glyph_id,
                missing: glyph.placed.missing,
                source: glyph.placed.source,
                line: 0,
                face: 0,
                x: x + (width - sized_w) / 2.0 + glyph.shift_x,
                y: y + (height - sized_h) / 2.0 + glyph.shift_y,
                width: sized_w,
                height: sized_h,
            },
            fill: glyph.fill,
            line: glyph.baseline,
            back: glyph.back,
            rest: (x, y, width, height),
        });
    }
    let boxes = word_boxes(&ink, font_size);
    let (origin_x, origin_y, width, height) = bounds(&ink, &boxes)?;
    let len = rgba_len(width, height).ok_or(TextError::Limit)?;
    let mut rgba = vec![0_u8; len];
    for word in &boxes {
        paint_round_box(&mut rgba, width, height, word, (origin_x, origin_y));
    }
    for glyph in &ink {
        let Some(image) = atlas.image(glyph.record.glyph_id) else {
            continue;
        };
        let record = glyph.record.clone();
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
    /// Where it is drawn, shifted and sized.
    record: GlyphRecord,
    fill: [u8; 4],
    /// Its line's baseline, so boxes on two lines stay apart.
    line: f32,
    back: [u8; 4],
    /// Where it is at rest: left, top, width, height.
    rest: (f32, f32, f32, f32),
}

/// A box behind a run of glyphs, in picture pixels before the origin.
pub(crate) struct WordBox {
    pub(crate) left: f32,
    pub(crate) top: f32,
    pub(crate) right: f32,
    pub(crate) bottom: f32,
    /// Straight RGBA.
    pub(crate) color: [u8; 4],
    pub(crate) radius: f32,
}

impl WordBox {
    /// A box round `record` with room about it, for text at `font_size`.
    pub(crate) fn round(record: &GlyphRecord, color: [u8; 4], font_size: f32) -> Self {
        let (pad_x, pad_y) = (font_size * 0.12, font_size * 0.08);
        Self {
            left: record.x - pad_x,
            top: record.y - pad_y,
            right: record.x + record.width + pad_x,
            bottom: record.y + record.height + pad_y,
            color,
            radius: font_size * 0.18,
        }
    }

    /// Grow to hold `other` too.
    pub(crate) fn join(&mut self, other: &Self) {
        self.left = self.left.min(other.left);
        self.top = self.top.min(other.top);
        self.right = self.right.max(other.right);
        self.bottom = self.bottom.max(other.bottom);
    }
}

/// One round box behind each run of drawn glyphs that share a box color
/// and a line, with room around them.
fn word_boxes(ink: &[InkGlyph], font_size: f32) -> Vec<WordBox> {
    let mut boxes: Vec<WordBox> = Vec::new();
    let mut last: Option<([u8; 4], f32)> = None;
    for glyph in ink {
        let record = &glyph.record;
        if glyph.back[3] == 0 {
            if record.width > 0.0 {
                last = None;
            }
            continue;
        }
        if record.width <= 0.0 || record.height <= 0.0 {
            continue;
        }
        let around = WordBox::round(record, glyph.back, font_size);
        let joins = last == Some((glyph.back, glyph.line));
        match boxes.last_mut() {
            Some(open) if joins => open.join(&around),
            _ => boxes.push(around),
        }
        last = Some((glyph.back, glyph.line));
    }
    boxes
}

/// A box with round corners and one pixel of soft edge, laid over what is
/// there.
pub(crate) fn paint_round_box(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    word: &WordBox,
    origin: (f32, f32),
) {
    let half_w = (word.right - word.left) / 2.0;
    let half_h = (word.bottom - word.top) / 2.0;
    let radius = word.radius.min(half_w).min(half_h).max(0.0);
    let center_x = word.left - origin.0 + half_w;
    let center_y = word.top - origin.1 + half_h;
    for y in 0..height {
        for x in 0..width {
            #[allow(clippy::cast_precision_loss)]
            let (px, py) = (x as f32 + 0.5 - center_x, y as f32 + 0.5 - center_y);
            let qx = px.abs() - (half_w - radius);
            let qy = py.abs() - (half_h - radius);
            let outside = qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0) - radius;
            let cover = (0.5 - outside).clamp(0.0, 1.0);
            if cover <= 0.0 {
                continue;
            }
            let at = ((y * width + x) * 4) as usize;
            let alpha = f32::from(word.color[3]) / 255.0 * cover;
            for channel in 0..3 {
                let over = f32::from(word.color[channel]) * alpha;
                let under = f32::from(rgba[at + channel]) * (1.0 - alpha);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                {
                    rgba[at + channel] = (over + under).round().min(255.0) as u8;
                }
            }
            let under = f32::from(rgba[at + 3]) / 255.0;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                rgba[at + 3] = ((alpha + under * (1.0 - alpha)) * 255.0).round() as u8;
            }
        }
    }
}

fn bounds(ink: &[InkGlyph], boxes: &[WordBox]) -> Result<(f32, f32, u32, u32), TextError> {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    let mut any = false;
    for glyph in ink {
        let (x, y, width, height) = glyph.rest;
        if width <= 0.0 || height <= 0.0 {
            continue;
        }
        any = true;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x + width);
        max_y = max_y.max(y + height);
    }
    // Grow evenly about the resting centre to hold what is drawn.
    let (mut reach_x, mut reach_y) = (0.0_f32, 0.0_f32);
    let mut hold = |left: f32, top: f32, right: f32, bottom: f32| {
        reach_x = reach_x.max(min_x - left).max(right - max_x);
        reach_y = reach_y.max(min_y - top).max(bottom - max_y);
    };
    for glyph in ink {
        let record = &glyph.record;
        if record.width > 0.0 && record.height > 0.0 {
            hold(
                record.x,
                record.y,
                record.x + record.width,
                record.y + record.height,
            );
        }
    }
    for word in boxes {
        hold(word.left, word.top, word.right, word.bottom);
    }
    if any {
        min_x -= reach_x;
        max_x += reach_x;
        min_y -= reach_y;
        max_y += reach_y;
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
            scale: 1.0,
            back: [0, 0, 0, 0],
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
            scale: 1.0,
            back: [0, 0, 0, 0],
        };
        let Picture::Cpu(shifted) = paint_glyphs(TTF, layout.font_size, &[moved]).expect("shift")
        else {
            panic!("platform");
        };
        let (shifted_left, _) = opaque_xs(shifted.rgba(), shifted.width());
        // The picture grows evenly, so its centre stays on the glyph at rest.
        assert!(shifted.width().abs_diff(still.width() + 48) <= 1);
        assert!(shifted_left > left + 40);

        // A box goes behind, and a larger glyph grows the picture.
        let boxed = DrawnGlyph {
            placed,
            baseline: line.baseline,
            fill: [255, 0, 0, 255],
            shift_x: 0.0,
            shift_y: 0.0,
            scale: 1.5,
            back: [0, 0, 255, 255],
        };
        let Picture::Cpu(boxed) = paint_glyphs(TTF, layout.font_size, &[boxed]).expect("box")
        else {
            panic!("platform");
        };
        assert!(boxed.width() > still.width() && boxed.height() > still.height());
        assert!(
            boxed
                .rgba()
                .chunks(4)
                .any(|pixel| pixel == [0, 0, 255, 255])
        );
        assert!(
            boxed
                .rgba()
                .chunks(4)
                .any(|pixel| pixel == [255, 0, 0, 255])
        );

        let Picture::Cpu(empty) = paint_glyphs(TTF, 64.0, &[]).expect("empty") else {
            panic!("platform");
        };
        assert_eq!(empty.rgba(), &[0, 0, 0, 0]);
    }
}
