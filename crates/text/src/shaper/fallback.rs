//! Places marks by the shapes of the glyphs, for a face whose own
//! positioning does not place them.
//!
//! Each mark is set against the box of the letter it follows: centered
//! above or below it, or to one side, as its combining class says, and
//! stacked clear of the marks already there. The result is plain but
//! legible, which is what an old face with no anchors can give.

use std::collections::HashMap;

use cutlass_font::morx::DELETED_GLYPH;
use cutlass_font::{Font, Segment};

use super::buffer::Glyph;
use crate::unicode::{gc, props};

/// Combining classes, as Unicode numbers them.
mod class {
    pub(super) const ATTACHED_BELOW_LEFT: u8 = 200;
    pub(super) const ATTACHED_BELOW: u8 = 202;
    pub(super) const ATTACHED_ABOVE: u8 = 214;
    pub(super) const ATTACHED_ABOVE_RIGHT: u8 = 216;
    pub(super) const BELOW_LEFT: u8 = 218;
    pub(super) const BELOW: u8 = 220;
    pub(super) const BELOW_RIGHT: u8 = 222;
    pub(super) const ABOVE_LEFT: u8 = 228;
    pub(super) const ABOVE: u8 = 230;
    pub(super) const ABOVE_RIGHT: u8 = 232;
    pub(super) const DOUBLE_BELOW: u8 = 233;
    pub(super) const DOUBLE_ABOVE: u8 = 234;
}

/// The side of its letter a mark with combining class `combining` goes
/// on, for the marks whose class does not say: Hebrew points, Arabic
/// vowels, and the signs of Thai, Lao and Tibetan.
pub(crate) fn side(character: char, combining: u8) -> u8 {
    if combining >= 200 {
        return combining;
    }
    let code = u32::from(character);
    if combining == 0 {
        return match code {
            0x0E31 | 0x0E34..=0x0E37 | 0x0E47 | 0x0E4C..=0x0E4E => class::ABOVE_RIGHT,
            0x0EB1 | 0x0EB4..=0x0EB7 | 0x0EBB | 0x0ECC | 0x0ECD => class::ABOVE,
            0x0EBC => class::BELOW,
            _ => 0,
        };
    }
    if code == 0x0E3A {
        return class::BELOW_RIGHT;
    }
    match combining {
        10..=18 | 20 | 22 | 29 | 32 | 118 | 129 | 132 => class::BELOW,
        23 => class::ATTACHED_ABOVE,
        24 | 107 => class::ABOVE_RIGHT,
        19 | 25 => class::ABOVE_LEFT,
        26..=28 | 30 | 31 | 33..=36 | 122 | 130 => class::ABOVE,
        103 => class::BELOW_RIGHT,
        other => other,
    }
}

/// A box around a glyph's ink: its left edge, its top, its width, and its
/// height, which is negative as the box runs down from the top.
#[derive(Clone, Copy)]
struct Extents {
    left: i32,
    top: i32,
    width: i32,
    height: i32,
}

struct Boxes<'f, 'a> {
    font: &'f Font<'a>,
    seen: HashMap<u16, Option<Extents>>,
}

impl Boxes<'_, '_> {
    fn of(&mut self, glyph: u16) -> Option<Extents> {
        let font = self.font;
        *self.seen.entry(glyph).or_insert_with(|| {
            let outline = font.outline(glyph).ok()?;
            let mut bounds: Option<(f32, f32, f32, f32)> = None;
            let mut hold = |x: f32, y: f32| {
                bounds = Some(match bounds {
                    None => (x, y, x, y),
                    Some((left, bottom, right, top)) => {
                        (left.min(x), bottom.min(y), right.max(x), top.max(y))
                    }
                });
            };
            for contour in &outline.contours {
                hold(contour.x, contour.y);
                for segment in &contour.segments {
                    match *segment {
                        Segment::Line { x, y } => hold(x, y),
                        Segment::Quad { cx, cy, x, y } => {
                            hold(cx, cy);
                            hold(x, y);
                        }
                        Segment::Cubic {
                            c1x,
                            c1y,
                            c2x,
                            c2y,
                            x,
                            y,
                        } => {
                            hold(c1x, c1y);
                            hold(c2x, c2y);
                            hold(x, y);
                        }
                    }
                }
            }
            let (left, bottom, right, top) = bounds?;
            #[allow(clippy::cast_possible_truncation)]
            let whole = |value: f32| value.round() as i32;
            Some(Extents {
                left: whole(left),
                top: whole(top),
                width: whole(right) - whole(left),
                height: whole(bottom) - whole(top),
            })
        })
    }
}

fn is_mark(glyph: &Glyph) -> bool {
    props(glyph.character).is_mark()
        || matches!(
            glyph.category,
            gc::NONSPACING_MARK | gc::ENCLOSING_MARK | gc::SPACING_MARK
        )
}

/// Place every mark of `glyphs` against the letter before it. `backward`
/// is the direction the run is read in; `adjust` moves a mark that could
/// not be placed back over its letter.
pub(crate) fn place_marks(font: &Font<'_>, glyphs: &mut [Glyph], backward: bool, adjust: bool) {
    let mut boxes = Boxes {
        font,
        seen: HashMap::new(),
    };
    let mut base = 0;
    while base < glyphs.len() {
        // A glyph a state machine deleted is no letter for a mark to
        // sit on: the marks after it belong to the letter before it.
        let gone = |glyph: &Glyph| glyph.id == DELETED_GLYPH;
        if is_mark(&glyphs[base]) || gone(&glyphs[base]) {
            base += 1;
            continue;
        }
        let mut end = base + 1;
        while end < glyphs.len()
            && (is_mark(&glyphs[end]) || glyphs[end].steps_over() || gone(&glyphs[end]))
        {
            end += 1;
        }
        if end - base >= 2 {
            around_base(&mut boxes, glyphs, base, end, backward, adjust);
        }
        base = end;
    }
}

fn around_base(
    boxes: &mut Boxes<'_, '_>,
    glyphs: &mut [Glyph],
    base: usize,
    end: usize,
    backward: bool,
    adjust: bool,
) {
    let Some(mut base_box) = boxes.of(glyphs[base].id) else {
        // A letter with no ink: its marks take no room and stay put.
        for glyph in &mut glyphs[base + 1..end] {
            if glyph.category == gc::NONSPACING_MARK {
                if adjust {
                    glyph.x_offset -= glyph.x_advance;
                }
                glyph.x_advance = 0;
            }
        }
        return;
    };
    base_box.top += glyphs[base].y_offset;
    // Across the line the letter's advance is a better guide than its ink.
    base_box.left = 0;
    base_box.width = boxes
        .font
        .advance(glyphs[base].id)
        .map_or(base_box.width, i32::from);
    let lig_id = glyphs[base].lig_id;
    let parts = i32::from(glyphs[base].parts());
    let mut x_offset = 0;
    if !backward {
        x_offset -= glyphs[base].x_advance;
    }
    let gap = i32::from(boxes.font.units_per_em()) / 16;
    let mut part_box = base_box;
    let mut last_part = -1;
    let mut last_class = 255_u8;
    let mut cluster_box = base_box;
    for glyph in &mut glyphs[base + 1..end] {
        let class = glyph.side;
        if class == 0 {
            if backward {
                x_offset += glyph.x_advance;
            } else {
                x_offset -= glyph.x_advance;
            }
            continue;
        }
        if parts > 1 {
            // A mark on a ligature goes over its own part of it.
            let mut part = i32::from(glyph.lig_part) - 1;
            if lig_id == 0 || lig_id != glyph.lig_id || part >= parts || part < 0 {
                part = parts - 1;
            }
            if last_part != part {
                last_part = part;
                last_class = 255;
                part_box = base_box;
                let along = if backward { parts - 1 - part } else { part };
                part_box.left += along * part_box.width / parts;
                part_box.width /= parts;
            }
        }
        if last_class != class {
            last_class = class;
            cluster_box = part_box;
        }
        if let Some(mark_box) = boxes.of(glyph.id) {
            glyph.x_offset = 0;
            glyph.y_offset = 0;
            place_one(glyph, &mut cluster_box, mark_box, class, backward, gap);
        }
        glyph.x_advance = 0;
        glyph.x_offset += x_offset;
    }
}

/// Set one mark against `base`, and grow `base` to hold it so the next
/// mark on that side goes beyond it.
fn place_one(
    glyph: &mut Glyph,
    base: &mut Extents,
    mark: Extents,
    class: u8,
    backward: bool,
    gap: i32,
) {
    // Across the line.
    glyph.x_offset += match class {
        class::DOUBLE_BELOW | class::DOUBLE_ABOVE if backward => {
            base.left - mark.width / 2 - mark.left
        }
        class::DOUBLE_BELOW | class::DOUBLE_ABOVE => {
            base.left + base.width - mark.width / 2 - mark.left
        }
        class::ATTACHED_BELOW_LEFT | class::BELOW_LEFT | class::ABOVE_LEFT => base.left - mark.left,
        class::ATTACHED_ABOVE_RIGHT | class::BELOW_RIGHT | class::ABOVE_RIGHT => {
            base.left + base.width - mark.width - mark.left
        }
        _ => base.left + (base.width - mark.width) / 2 - mark.left,
    };
    // Down or up the line.
    let below = matches!(
        class,
        class::DOUBLE_BELOW | class::BELOW_LEFT | class::BELOW | class::BELOW_RIGHT
    );
    let above = matches!(
        class,
        class::DOUBLE_ABOVE | class::ABOVE_LEFT | class::ABOVE | class::ABOVE_RIGHT
    );
    if below || matches!(class, class::ATTACHED_BELOW_LEFT | class::ATTACHED_BELOW) {
        if below {
            base.height -= gap;
        }
        glyph.y_offset = base.top + base.height - mark.top;
        // A mark below is never moved up.
        if (gap > 0) == (glyph.y_offset > 0) {
            base.height -= glyph.y_offset;
            glyph.y_offset = 0;
        }
        base.height += mark.height;
    } else if above || matches!(class, class::ATTACHED_ABOVE | class::ATTACHED_ABOVE_RIGHT) {
        if above {
            base.top += gap;
            base.height -= gap;
        }
        glyph.y_offset = base.top - (mark.top + mark.height);
        // A mark above is not moved down by much.
        if (gap > 0) != (glyph.y_offset > 0) {
            let correction = -glyph.y_offset / 2;
            base.top += correction;
            base.height -= correction;
            glyph.y_offset += correction;
        }
        base.top -= mark.height;
        base.height += mark.height;
    }
}
