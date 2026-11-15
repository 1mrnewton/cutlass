//! A title's lines bent along an arc.
//!
//! The curve is the degrees of arc the title's width takes on its middle
//! line, so the circle's radius is that width over the angle. Above zero
//! the circle's middle lies below the letters and they arch up; below
//! zero it lies above and they dip. Every line bends about the same
//! middle, so lines stay apart as they were. Each letter keeps its place
//! along its line as a length along its arc, stands on the arc at its
//! baseline, and is turned to follow it.

use crate::mesh::GlyphRecord;

/// How one letter is turned on the arc: the sine and cosine of its turn,
/// clockwise, about the point on its baseline at the middle of its quad,
/// `down` of the way down the quad.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Bend {
    pub(crate) sin: f32,
    pub(crate) cos: f32,
    pub(crate) down: f32,
}

impl Bend {
    /// The point the letter on `record` turns about.
    pub(crate) fn pivot(&self, record: &GlyphRecord) -> (f32, f32) {
        (
            record.x + record.width / 2.0,
            record.y + record.height * self.down,
        )
    }

    /// `(x, y)` turned about `pivot`.
    pub(crate) fn turn(&self, (x, y): (f32, f32), pivot: (f32, f32)) -> (f32, f32) {
        let (dx, dy) = (x - pivot.0, y - pivot.1);
        (
            pivot.0 + dx * self.cos - dy * self.sin,
            pivot.1 + dx * self.sin + dy * self.cos,
        )
    }

    /// `(x, y)` turned back about `pivot`: where on the letter's own quad
    /// a point of the picture falls.
    pub(crate) fn unturn(&self, (x, y): (f32, f32), pivot: (f32, f32)) -> (f32, f32) {
        let (dx, dy) = (x - pivot.0, y - pivot.1);
        (
            pivot.0 + dx * self.cos + dy * self.sin,
            pivot.1 - dx * self.sin + dy * self.cos,
        )
    }

    /// The box around `record`'s quad, grown by `room` each way, once
    /// turned: left, top, right, bottom.
    pub(crate) fn extent(&self, record: &GlyphRecord, room: f32) -> [f32; 4] {
        let pivot = self.pivot(record);
        let (left, top) = (record.x - room, record.y - room);
        let (right, bottom) = (
            record.x + record.width + room,
            record.y + record.height + room,
        );
        let mut extent = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
        for corner in [(left, top), (right, top), (left, bottom), (right, bottom)] {
            let (x, y) = self.turn(corner, pivot);
            extent = [
                extent[0].min(x),
                extent[1].min(y),
                extent[2].max(x),
                extent[3].max(y),
            ];
        }
        extent
    }
}

/// Bend `glyphs` along an arc `degrees` wide, moving each and its
/// baseline in `baselines` onto the arc. What each glyph is turned by, in
/// their order; none when there is nothing to bend.
pub(crate) fn bend(glyphs: &mut [GlyphRecord], baselines: &mut [f32], degrees: f64) -> Vec<Bend> {
    let degrees = degrees.clamp(-cutlass_model::MAX_CURVE, cutlass_model::MAX_CURVE);
    #[allow(clippy::cast_possible_truncation)]
    let arc = degrees.to_radians() as f32;
    if !arc.is_finite() || arc.abs() < 1e-4 || glyphs.len() != baselines.len() {
        return Vec::new();
    }
    let drawn = || {
        glyphs
            .iter()
            .zip(baselines.iter())
            .filter(|(glyph, _)| glyph.width > 0.0 && glyph.height > 0.0)
    };
    let (mut left, mut right) = (f32::MAX, f32::MIN);
    let (mut high, mut low) = (f32::MAX, f32::MIN);
    for (glyph, baseline) in drawn() {
        left = left.min(glyph.x);
        right = right.max(glyph.x + glyph.width);
        high = high.min(*baseline);
        low = low.max(*baseline);
    }
    let wide = right - left;
    if !(wide.is_finite() && wide > 0.0 && high.is_finite() && low.is_finite()) {
        return Vec::new();
    }
    let radius = wide / arc;
    let middle = (left + right) / 2.0;
    let centre = (high + low) / 2.0 + radius;
    glyphs
        .iter_mut()
        .zip(baselines.iter_mut())
        .map(|(glyph, baseline)| {
            let down = if glyph.height > 0.0 {
                (*baseline - glyph.y) / glyph.height
            } else {
                1.0
            };
            // The line's own radius: a line nearer the circle's middle
            // than a pixel, or past it, stays straight.
            let own = centre - *baseline;
            if own.abs() < 1.0 || own.signum() != radius.signum() {
                return Bend {
                    sin: 0.0,
                    cos: 1.0,
                    down,
                };
            }
            let along = glyph.x + glyph.width / 2.0;
            let (sin, cos) = ((along - middle) / own).sin_cos();
            let (x, y) = (middle + own * sin, centre - own * cos);
            glyph.x += x - along;
            glyph.y += y - *baseline;
            *baseline = y;
            Bend { sin, cos, down }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glyph(x: f32) -> GlyphRecord {
        GlyphRecord {
            glyph_id: 1,
            missing: false,
            source: 0,
            line: 0,
            face: 0,
            x,
            y: 0.0,
            width: 10.0,
            height: 20.0,
        }
    }

    #[test]
    fn an_arch_lifts_the_middle_and_turns_the_ends_out() {
        let mut glyphs: Vec<GlyphRecord> = (0..5).map(|at| glyph(at as f32 * 10.0)).collect();
        let mut baselines = vec![15.0; 5];
        let bends = bend(&mut glyphs, &mut baselines, 90.0);
        assert_eq!(bends.len(), 5);
        // The middle letter stays; the ends drop and turn away from it.
        assert!((baselines[2] - 15.0).abs() < 1e-3);
        assert!(baselines[0] > 16.0 && (baselines[0] - baselines[4]).abs() < 1e-3);
        assert!(bends[0].sin < 0.0 && bends[4].sin > 0.0);
        assert!((bends[2].sin).abs() < 1e-6);
        // The last letter's middle is 20 of the line's 50 from its middle,
        // so it turns two fifths of the way to the arc's end: 36 degrees.
        let turned = 36.0_f32.to_radians().sin();
        assert!((bends[4].sin - turned).abs() < 1e-4, "{}", bends[4].sin);
        // A point turned and back is where it was.
        let pivot = bends[4].pivot(&glyphs[4]);
        let back = bends[4].unturn(bends[4].turn((3.0, 4.0), pivot), pivot);
        assert!((back.0 - 3.0).abs() < 1e-4 && (back.1 - 4.0).abs() < 1e-4);
    }

    #[test]
    fn a_dip_raises_the_ends_and_none_leaves_the_letters() {
        let mut glyphs: Vec<GlyphRecord> = (0..5).map(|at| glyph(at as f32 * 10.0)).collect();
        let mut baselines = vec![15.0; 5];
        let bends = bend(&mut glyphs, &mut baselines, -90.0);
        assert!(baselines[0] < 14.0);
        assert!(bends[0].sin > 0.0 && bends[4].sin < 0.0);

        let mut straight: Vec<GlyphRecord> = (0..5).map(|at| glyph(at as f32 * 10.0)).collect();
        let before = straight.clone();
        assert!(bend(&mut straight, &mut [15.0; 5], 0.0).is_empty());
        assert_eq!(straight, before);
    }
}
