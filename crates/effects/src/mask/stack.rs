//! A clip's masks placed on the canvas together, for a pass that keeps or
//! limits by all of them at once: each mask's shape, or for a drawn mask
//! (text, brush, pen) its picture, joined in order by its mode.
//!
//! The GPU reads the masks as one uniform block ([`STACK_WGSL`]) and every
//! drawn mask's picture from one one-byte-a-pixel atlas, the pictures laid
//! one under the other. The CPU arithmetic here is the same.

use std::sync::Arc;

use super::{CanvasMask, Mask};

/// The most masks one pass joins: a clip lists at most eight of a kind.
pub const MAX_STACK: usize = 8;

/// Floats in the uniform block: a head, then each mask.
pub const STACK_FLOATS: usize = 4 + MAX_STACK * MASK_FLOATS;

/// Floats for one mask: its [`CanvasMask`] and its art.
const MASK_FLOATS: usize = 24;

/// A drawn mask's coverage, one byte a pixel, row by row from the top. It
/// covers the mask's box `reach` times over each way, so a feather may
/// spill past the box: a picture point `(x, y)`, each -1 to 1 with y up,
/// is the box point `(x·reach, y·reach)`.
#[derive(Clone, Debug, PartialEq)]
pub struct ArtPicture {
    pub width: u32,
    pub height: u32,
    pub alpha: Arc<[u8]>,
    pub reach: f32,
}

impl ArtPicture {
    /// Whether the bytes fill the size and the reach is a reach.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.width > 0
            && self.height > 0
            && self.alpha.len() == self.width as usize * self.height as usize
            && self.reach.is_finite()
            && self.reach >= 1.0
    }

    /// The coverage at box point `(x, y)` (-1 to 1 across the mask's box,
    /// y up), read between the four nearest pixels; nothing off the
    /// picture.
    #[must_use]
    pub fn coverage(&self, x: f32, y: f32) -> f32 {
        let (x, y) = (x / self.reach, y / self.reach);
        if !(-1.0..=1.0).contains(&x) || !(-1.0..=1.0).contains(&y) || !self.is_valid() {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let (w, h) = (self.width as f32, self.height as f32);
        let px = ((x + 1.0) * 0.5 * w - 0.5).clamp(0.0, w - 1.0);
        let py = ((1.0 - y) * 0.5 * h - 0.5).clamp(0.0, h - 1.0);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let (x0, y0) = (px as usize, py as usize);
        let (x1, y1) = (
            (x0 + 1).min(self.width as usize - 1),
            (y0 + 1).min(self.height as usize - 1),
        );
        #[allow(clippy::cast_precision_loss)]
        let (fx, fy) = (px - x0 as f32, py - y0 as f32);
        let at = |x: usize, y: usize| f32::from(self.alpha[y * self.width as usize + x]) / 255.0;
        let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * fx;
        let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * fx;
        top + (bottom - top) * fy
    }
}

/// One mask of a stack: its numbers, and the picture of a drawn mask.
#[derive(Clone, Debug, PartialEq)]
pub struct DrawnMask {
    pub mask: Mask,
    /// For a text, brush or pen mask, its coverage over its box. A drawn
    /// mask with no picture yet keeps nothing.
    pub art: Option<ArtPicture>,
}

impl DrawnMask {
    /// A mask drawn from its shape alone.
    #[must_use]
    pub fn shape(mask: Mask) -> Self {
        Self { mask, art: None }
    }

    /// How much it keeps at clip point `(u, v)`, invert taken in.
    #[must_use]
    pub fn coverage(&self, u: f32, v: f32) -> f32 {
        let Some(art) = &self.art else {
            return if self.mask.shape.is_drawn() {
                if self.mask.invert { 1.0 } else { 0.0 }
            } else {
                self.mask.coverage(u, v)
            };
        };
        let m = &self.mask;
        let (sin, cos) = m.rotation.to_radians().sin_cos();
        let (du, dv) = (u - m.x, v - m.y);
        let (a, b) = (cos * du + sin * dv, -sin * du + cos * dv);
        let keep = art.coverage(a / m.width.max(1e-4), b / m.height.max(1e-4));
        if m.invert { 1.0 - keep } else { keep }
    }
}

/// How much `masks` keep at clip point `(u, v)`, joined in order by their
/// modes; everything when there are none.
#[must_use]
pub fn drawn_coverage(masks: &[DrawnMask], u: f32, v: f32) -> f32 {
    let Some(first) = masks.first() else {
        return 1.0;
    };
    masks
        .iter()
        .fold(first.mask.mode.start(), |kept, mask| {
            mask.mask
                .mode
                .combine(kept, mask.coverage(u, v).clamp(0.0, 1.0))
        })
        .clamp(0.0, 1.0)
}

/// A clip's masks placed on the canvas: `frame` takes a canvas pixel to
/// the clip's own frame, as for a [`CanvasMask`].
#[derive(Clone, Debug, PartialEq)]
pub struct CanvasMasks {
    pub masks: Vec<DrawnMask>,
    pub frame: [f32; 6],
}

impl From<CanvasMask> for CanvasMasks {
    fn from(one: CanvasMask) -> Self {
        Self {
            masks: vec![DrawnMask::shape(one.mask)],
            frame: one.frame,
        }
    }
}

impl CanvasMasks {
    /// How much the masks keep at canvas pixel centre `(x, y)`.
    #[must_use]
    pub fn coverage_at(&self, x: f32, y: f32) -> f32 {
        let [a, b, c, d, e, f] = self.frame;
        drawn_coverage(&self.masks, a * x + b * y + c, d * x + e * y + f)
    }

    /// The masks on `frame` scaled for a canvas `sx` and `sy` times the
    /// size each way.
    #[must_use]
    pub fn scaled(&self, sx: f32, sy: f32) -> Self {
        let [a, b, c, d, e, f] = self.frame;
        Self {
            masks: self.masks.clone(),
            frame: [a / sx, b / sy, c, d / sx, e / sy, f],
        }
    }

    /// The drawn masks' pictures laid one under the other: the atlas's
    /// width, height and bytes, a 1 by 1 empty one when there are none.
    #[must_use]
    pub fn atlas(&self) -> (u32, u32, Vec<u8>) {
        let arts = || {
            self.masks
                .iter()
                .take(MAX_STACK)
                .filter_map(|mask| mask.art.as_ref())
                .filter(|art| art.is_valid())
        };
        let width = arts().map(|art| art.width).max().unwrap_or(0);
        let height: u32 = arts().map(|art| art.height).sum();
        if width == 0 || height == 0 {
            return (1, 1, vec![0]);
        }
        let mut bytes = vec![0_u8; width as usize * height as usize];
        let mut top = 0_usize;
        for art in arts() {
            let w = art.width as usize;
            for (row, line) in art.alpha.chunks_exact(w).enumerate() {
                let at = (top + row) * width as usize;
                bytes[at..at + w].copy_from_slice(line);
            }
            top += art.height as usize;
        }
        (width, height, bytes)
    }

    /// The [`STACK_FLOATS`] numbers [`STACK_WGSL`]'s `MaskStack` holds, the
    /// art placed as [`CanvasMasks::atlas`] lays it.
    #[must_use]
    pub fn uniform(&self) -> Vec<f32> {
        let mut values = vec![0.0_f32; STACK_FLOATS];
        let masks = &self.masks[..self.masks.len().min(MAX_STACK)];
        #[allow(clippy::cast_precision_loss)]
        {
            values[0] = masks.len() as f32;
        }
        values[1] = masks.first().map_or(1.0, |first| first.mask.mode.start());
        let mut top = 0_u32;
        for (index, drawn) in masks.iter().enumerate() {
            let at = 4 + index * MASK_FLOATS;
            let one = CanvasMask {
                mask: drawn.mask,
                frame: self.frame,
            };
            values[at..at + 16].copy_from_slice(&one.uniform());
            let mode = super::MaskMode::ALL
                .iter()
                .position(|mode| *mode == drawn.mask.mode)
                .unwrap_or_default();
            #[allow(clippy::cast_precision_loss)]
            {
                values[at + 16] = mode as f32;
            }
            // A drawn mask: 1 with its picture, 2 with none (keeps
            // nothing), 0 for a shape.
            values[at + 17] = match &drawn.art {
                Some(art) if art.is_valid() => 1.0,
                Some(_) | None if drawn.mask.shape.is_drawn() => 2.0,
                _ => 0.0,
            };
            if let Some(art) = drawn.art.as_ref().filter(|art| art.is_valid()) {
                #[allow(clippy::cast_precision_loss)]
                {
                    values[at + 18] = art.reach;
                    values[at + 19] = top as f32;
                    values[at + 20] = art.width as f32;
                    values[at + 21] = art.height as f32;
                }
                top += art.height;
            }
        }
        values
    }
}

/// What a set of [`CanvasMasks`] keeps at a canvas pixel, for a GPU pass:
/// `mask_stack(stack, atlas, pixel)`. It needs [`super::MASK_WGSL`] beside
/// it.
pub const STACK_WGSL: &str = r"
struct StackMask {
    frame0: vec4<f32>,
    frame1: vec4<f32>,
    shape0: vec4<f32>,
    shape1: vec4<f32>,
    // mode, drawn (1 with a picture, 2 without), reach, atlas row
    art0: vec4<f32>,
    // picture width, height
    art1: vec4<f32>,
}

struct MaskStack {
    // count, what is kept before the first
    head: vec4<f32>,
    masks: array<StackMask, 8>,
}

fn stack_art_at(atlas: texture_2d<f32>, x: i32, y: i32) -> f32 {
    return textureLoad(atlas, vec2<i32>(x, y), 0).r;
}

fn stack_art(m: StackMask, atlas: texture_2d<f32>, pixel: vec2<f32>) -> f32 {
    let u = m.frame0.x * pixel.x + m.frame0.y * pixel.y + m.frame0.z;
    let v = m.frame0.w * pixel.x + m.frame1.x * pixel.y + m.frame1.y;
    let turn = radians(m.shape1.x);
    let s = sin(turn);
    let c = cos(turn);
    let du = u - m.shape0.x;
    let dv = v - m.shape0.y;
    let reach = m.art0.z;
    let x = (c * du + s * dv) / max(m.shape0.z, 1e-4) / reach;
    let y = (-s * du + c * dv) / max(m.shape0.w, 1e-4) / reach;
    var keep = 0.0;
    if (x >= -1.0 && x <= 1.0 && y >= -1.0 && y <= 1.0) {
        let w = m.art1.x;
        let h = m.art1.y;
        let px = clamp((x + 1.0) * 0.5 * w - 0.5, 0.0, w - 1.0);
        let py = clamp((1.0 - y) * 0.5 * h - 0.5, 0.0, h - 1.0);
        let x0 = i32(px);
        let y0 = i32(py);
        let x1 = min(x0 + 1, i32(w) - 1);
        let y1 = min(y0 + 1, i32(h) - 1);
        let fx = px - f32(x0);
        let fy = py - f32(y0);
        let row = i32(m.art0.w);
        let top = stack_art_at(atlas, x0, row + y0)
            + (stack_art_at(atlas, x1, row + y0) - stack_art_at(atlas, x0, row + y0)) * fx;
        let bottom = stack_art_at(atlas, x0, row + y1)
            + (stack_art_at(atlas, x1, row + y1) - stack_art_at(atlas, x0, row + y1)) * fx;
        keep = top + (bottom - top) * fy;
    }
    if (m.frame1.w > 0.5) {
        keep = 1.0 - keep;
    }
    return clamp(keep, 0.0, 1.0);
}

fn stack_one(m: StackMask, atlas: texture_2d<f32>, pixel: vec2<f32>) -> f32 {
    if (m.art0.y > 1.5) {
        return select(0.0, 1.0, m.frame1.w > 0.5);
    }
    if (m.art0.y > 0.5) {
        return stack_art(m, atlas, pixel);
    }
    return mask_coverage(CanvasMask(m.frame0, m.frame1, m.shape0, m.shape1), pixel);
}

fn mask_stack(stack: MaskStack, atlas: texture_2d<f32>, pixel: vec2<f32>) -> f32 {
    let count = u32(stack.head.x + 0.5);
    if (count == 0u) {
        return 1.0;
    }
    var kept = stack.head.y;
    for (var index = 0u; index < min(count, 8u); index = index + 1u) {
        let m = stack.masks[index];
        let cover = clamp(stack_one(m, atlas, pixel), 0.0, 1.0);
        let mode = u32(m.art0.x + 0.5);
        if (mode == 0u) {
            kept = kept + cover - kept * cover;
        } else if (mode == 1u) {
            kept = kept * (1.0 - cover);
        } else {
            kept = kept * cover;
        }
    }
    return clamp(kept, 0.0, 1.0);
}
";

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{ArtPicture, CanvasMasks, DrawnMask, MAX_STACK, STACK_FLOATS, drawn_coverage};
    use crate::mask::{CanvasMask, Mask, MaskMode, MaskShape};

    /// A 4 by 2 picture whose left half is covered.
    fn left_half() -> ArtPicture {
        ArtPicture {
            width: 4,
            height: 2,
            alpha: Arc::from([255, 255, 0, 0, 255, 255, 0, 0].as_slice()),
            reach: 1.0,
        }
    }

    #[test]
    fn a_drawn_mask_reads_its_picture_through_its_box() {
        let drawn = DrawnMask {
            mask: Mask {
                width: 0.5,
                height: 0.5,
                ..Mask::new(MaskShape::Pen)
            },
            art: Some(left_half()),
        };
        // The box is half the clip: its left half is u -0.5 to 0.
        assert!(drawn.coverage(-0.4, 0.0) > 0.99);
        assert!(drawn.coverage(0.4, 0.0) < 0.01);
        // Off the box, nothing; inverted, everything.
        assert_eq!(drawn.coverage(-0.9, 0.0), 0.0);
        let inverted = DrawnMask {
            mask: Mask {
                invert: true,
                ..drawn.mask
            },
            ..drawn.clone()
        };
        assert_eq!(inverted.coverage(-0.9, 0.0), 1.0);
        // Turned a half turn the picture's left is on the right.
        let turned = DrawnMask {
            mask: Mask {
                rotation: 180.0,
                ..drawn.mask
            },
            ..drawn.clone()
        };
        assert!(turned.coverage(0.4, 0.0) > 0.99);
        // A picture reaching twice past its box shows it at half size.
        let reaching = DrawnMask {
            art: Some(ArtPicture {
                reach: 2.0,
                ..left_half()
            }),
            ..drawn.clone()
        };
        assert!(reaching.coverage(-0.9, 0.0) > 0.99);
        // A drawn mask with no picture keeps nothing.
        let empty = DrawnMask::shape(Mask::new(MaskShape::Brush));
        assert_eq!(empty.coverage(0.0, 0.0), 0.0);
        // Odd bytes keep nothing either.
        let odd = ArtPicture {
            alpha: Arc::from([255_u8; 3].as_slice()),
            ..left_half()
        };
        assert_eq!(odd.coverage(-0.5, 0.0), 0.0);
    }

    #[test]
    fn shapes_and_pictures_join_by_their_modes() {
        let pen = DrawnMask {
            mask: Mask {
                width: 1.0,
                height: 1.0,
                ..Mask::new(MaskShape::Pen)
            },
            art: Some(left_half()),
        };
        let circle = DrawnMask::shape(Mask {
            x: -0.5,
            width: 0.2,
            height: 0.2,
            mode: MaskMode::Subtract,
            ..Mask::new(MaskShape::Circle)
        });
        let masks = [pen, circle];
        assert!(drawn_coverage(&masks, -0.9, 0.0) > 0.99);
        assert!(drawn_coverage(&masks, -0.5, 0.0) < 0.01);
        assert!(drawn_coverage(&masks, 0.5, 0.0) < 0.01);
        assert_eq!(drawn_coverage(&[], 0.5, 0.0), 1.0);
    }

    #[test]
    fn the_atlas_lays_the_pictures_one_under_the_other_and_the_uniform_finds_them() {
        let tall = ArtPicture {
            width: 2,
            height: 3,
            alpha: Arc::from([9_u8; 6].as_slice()),
            reach: 1.5,
        };
        let masks = CanvasMasks {
            masks: vec![
                DrawnMask {
                    mask: Mask::new(MaskShape::Pen),
                    art: Some(left_half()),
                },
                DrawnMask::shape(Mask {
                    mode: MaskMode::Intersect,
                    ..Mask::new(MaskShape::Star)
                }),
                DrawnMask {
                    mask: Mask::new(MaskShape::Text),
                    art: Some(tall),
                },
                DrawnMask::shape(Mask::new(MaskShape::Brush)),
            ],
            frame: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        };
        let (width, height, bytes) = masks.atlas();
        assert_eq!((width, height), (4, 5));
        assert_eq!(&bytes[..8], &[255, 255, 0, 0, 255, 255, 0, 0]);
        assert_eq!(&bytes[8..12], &[9, 9, 0, 0]);
        let uniform = masks.uniform();
        assert_eq!(uniform.len(), STACK_FLOATS);
        assert_eq!(&uniform[..2], &[4.0, 0.0]);
        let mask = |index: usize| &uniform[4 + index * 24..4 + (index + 1) * 24];
        assert_eq!(&mask(0)[16..22], &[0.0, 1.0, 1.0, 0.0, 4.0, 2.0]);
        assert_eq!(&mask(1)[16..18], &[2.0, 0.0]);
        assert_eq!(&mask(2)[16..22], &[0.0, 1.0, 1.5, 2.0, 2.0, 3.0]);
        assert_eq!(mask(3)[17], 2.0, "a drawn mask with no picture");
        // No pictures, a one pixel atlas; one mask, as one mask was.
        let one = CanvasMasks::from(CanvasMask {
            mask: Mask::new(MaskShape::Circle),
            frame: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        });
        assert_eq!(one.atlas(), (1, 1, vec![0]));
        assert_eq!(&one.uniform()[..2], &[1.0, 0.0]);
        // Past eight, the rest are left out.
        let many = CanvasMasks {
            masks: vec![DrawnMask::shape(Mask::new(MaskShape::Circle)); MAX_STACK + 3],
            frame: one.frame,
        };
        assert_eq!(many.uniform()[0], 8.0);
    }
}
