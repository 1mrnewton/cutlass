//! A mask's handles on the preview, and how a drag on one becomes the
//! mask's new place, size, turn or feather; and how a finger drawing on the
//! preview becomes a pen or brush mask's drawing.
//!
//! The shell hands over the clip's frame at the time shown: the map from a
//! canvas pixel (x right, y down) to the clip's own picture (-1 to 1 each
//! way, y up), as `cutlass_eval::clip_point` gives it. A mask is measured
//! in that frame, so its handles are worked out there and drawn back on
//! the canvas. A drag on a mask whose numbers are keyed keys them at the
//! time shown; on any other mask it changes the mask.

use cutlass_effects::{BrushStroke, COLOR_MASK, MASK, Mask, MaskArt, MaskShape, listed_masks};
use cutlass_model::{Draft, MASK_KEYS, MaterialId, ShapePoint, mask_key};
use cutlass_session::{Micros, Request, SegmentId};

/// How far past the box the turn knob sits, in clip parts.
const KNOB_GAP: f64 = 0.15;
/// The smallest a box side is dragged to, in clip parts.
const MIN_SIDE: f64 = 0.01;
/// The largest, as a mask allows.
const MAX_SIDE: f64 = 4.0;
/// Brush points closer than this many canvas pixels are one point.
const BRUSH_STEP: f64 = 1.5;

/// The map from a canvas pixel to the clip's own picture, as
/// `(a·x + b·y + c, d·x + e·y + f)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipFrame(pub [f64; 6]);

impl ClipFrame {
    /// The frame from where three clip points land on the canvas: the
    /// clip's middle, `(1, 0)` and `(0, 1)`. `None` when they are in a
    /// line, as for a clip squeezed to nothing.
    #[must_use]
    pub fn from_canvas(middle: (f64, f64), right: (f64, f64), up: (f64, f64)) -> Option<Self> {
        // canvas = middle + u·a + v·b; solve for (u, v).
        let (a, b) = (
            (right.0 - middle.0, right.1 - middle.1),
            (up.0 - middle.0, up.1 - middle.1),
        );
        let det = a.0 * b.1 - b.0 * a.1;
        if !det.is_finite() || det.abs() < 1e-12 {
            return None;
        }
        let (ia, ib, ic, id) = (b.1 / det, -b.0 / det, -a.1 / det, a.0 / det);
        Some(Self([
            ia,
            ib,
            -(ia * middle.0 + ib * middle.1),
            ic,
            id,
            -(ic * middle.0 + id * middle.1),
        ]))
    }

    /// The clip point canvas pixel `(x, y)` shows.
    #[must_use]
    pub fn clip(&self, x: f64, y: f64) -> (f64, f64) {
        let [a, b, c, d, e, f] = self.0;
        (a * x + b * y + c, d * x + e * y + f)
    }

    /// The canvas pixel clip point `(u, v)` lands on, or `None` for a frame
    /// that squeezes the clip to nothing.
    #[must_use]
    pub fn canvas(&self, u: f64, v: f64) -> Option<(f64, f64)> {
        let [a, b, c, d, e, f] = self.0;
        let det = a * e - b * d;
        if !det.is_finite() || det.abs() < 1e-12 {
            return None;
        }
        let (du, dv) = (u - c, v - f);
        Some(((e * du - b * dv) / det, (a * dv - d * du) / det))
    }
}

/// A part of a mask a finger can take hold of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskHandle {
    /// The whole mask, by its middle or inside it.
    Move,
    /// A corner of its box, sizing it about its middle: right is `+x` in
    /// the box's own turned frame, up is `+y`.
    Corner { right: bool, up: bool },
    /// The knob past its box that turns it.
    Turn,
    /// The knob below its box that softens its edge.
    Feather,
}

/// Where a mask's handles are on the canvas, in pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct MaskHandles {
    pub middle: (f64, f64),
    /// The box's corners: top left, top right, bottom right, bottom left,
    /// as the mask's own frame has them.
    pub corners: [(f64, f64); 4],
    pub turn: (f64, f64),
    pub feather: (f64, f64),
    /// For a line or a mirror, the line through the middle: two points on
    /// it either side, as far out as the box.
    pub line: Option<[(f64, f64); 2]>,
}

/// A clip point in `mask`'s own turned frame, in clip parts.
fn into_mask(mask: &Mask, (u, v): (f64, f64)) -> (f64, f64) {
    let (sin, cos) = f64::from(mask.rotation).to_radians().sin_cos();
    let (du, dv) = (u - f64::from(mask.x), v - f64::from(mask.y));
    (cos * du + sin * dv, -sin * du + cos * dv)
}

/// A point in `mask`'s own turned frame as a clip point.
fn out_of_mask(mask: &Mask, (a, b): (f64, f64)) -> (f64, f64) {
    let (sin, cos) = f64::from(mask.rotation).to_radians().sin_cos();
    (
        f64::from(mask.x) + cos * a - sin * b,
        f64::from(mask.y) + sin * a + cos * b,
    )
}

/// `mask`'s handles on the canvas through `frame`.
#[must_use]
pub fn mask_handles(mask: &Mask, frame: &ClipFrame) -> Option<MaskHandles> {
    let (w, h) = (f64::from(mask.width), f64::from(mask.height));
    let at = |point: (f64, f64)| {
        let (u, v) = out_of_mask(mask, point);
        frame.canvas(u, v)
    };
    let corners = [at((-w, h))?, at((w, h))?, at((w, -h))?, at((-w, -h))?];
    let line = matches!(mask.shape, MaskShape::Linear | MaskShape::Mirror)
        .then(|| Some([at((-w.max(0.5), 0.0))?, at((w.max(0.5), 0.0))?]))
        .flatten();
    Some(MaskHandles {
        middle: at((0.0, 0.0))?,
        corners,
        turn: at((0.0, h + KNOB_GAP))?,
        feather: at((0.0, -(h + KNOB_GAP + f64::from(mask.feather) * 0.5)))?,
        line,
    })
}

/// The handle of `mask` canvas pixel `(x, y)` lands on, within `reach`
/// pixels of a knob or a corner, or inside the box for a move.
#[must_use]
pub fn mask_pick(mask: &Mask, frame: &ClipFrame, x: f64, y: f64, reach: f64) -> Option<MaskHandle> {
    let handles = mask_handles(mask, frame)?;
    let near = |(hx, hy): (f64, f64)| (hx - x).hypot(hy - y) <= reach;
    if near(handles.turn) {
        return Some(MaskHandle::Turn);
    }
    if near(handles.feather) {
        return Some(MaskHandle::Feather);
    }
    let sides = [(false, true), (true, true), (true, false), (false, false)];
    for (corner, (right, up)) in handles.corners.iter().zip(sides) {
        if near(*corner) {
            return Some(MaskHandle::Corner { right, up });
        }
    }
    let (a, b) = into_mask(mask, frame.clip(x, y));
    let inside = a.abs() <= f64::from(mask.width) && b.abs() <= f64::from(mask.height);
    (inside || near(handles.middle)).then_some(MaskHandle::Move)
}

/// A drag on one handle of one mask, measured from where it started.
#[derive(Clone, Debug, PartialEq)]
pub struct MaskGesture {
    segment_id: SegmentId,
    index: usize,
    color: bool,
    frame: ClipFrame,
    start: Mask,
    handle: MaskHandle,
    /// Where the finger came down, as a clip point.
    grab: (f64, f64),
    /// The offset from the clip's start to key the mask at, when it has
    /// keys.
    keyed: Option<Micros>,
    /// The names its keys go under.
    key_record: Option<String>,
}

impl MaskGesture {
    /// A drag on the mask at `index` of `segment_id`'s clip masks, or with
    /// `color` its colour masks, at timeline `time`, from canvas pixel
    /// `(x, y)` on `handle`. `None` when the clip has no such mask.
    #[must_use]
    pub fn begin(
        draft: &Draft,
        segment_id: &SegmentId,
        place: (usize, bool),
        frame: ClipFrame,
        time: Micros,
        handle: MaskHandle,
        (x, y): (f64, f64),
    ) -> Option<Self> {
        let (index, color) = place;
        let segment = draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .find(|segment| segment.id == *segment_id)?;
        let kind = if color { COLOR_MASK } else { MASK };
        let listed = listed_masks(draft, segment, kind).into_iter().nth(index)?;
        let start = listed.at(segment, &draft.keyframe_graph_list, time);
        let key_record = listed.key_record().map(str::to_owned);
        let names = MASK_KEYS.map(|name| mask_key(name, key_record.as_deref()));
        let has_keys = segment
            .common_keyframes
            .iter()
            .any(|group| names.contains(&group.property_type) && !group.keyframe_list.is_empty());
        let offset = time.0.saturating_sub(segment.target.start.0);
        Some(Self {
            segment_id: segment_id.clone(),
            index,
            color,
            frame,
            start,
            handle,
            grab: frame.clip(x, y),
            keyed: has_keys.then_some(Micros(offset)),
            key_record,
        })
    }

    /// The mask as it was when the drag began.
    #[must_use]
    pub fn start(&self) -> &Mask {
        &self.start
    }

    /// The mask with the finger at canvas pixel `(x, y)`.
    #[must_use]
    pub fn step(&self, x: f64, y: f64) -> Mask {
        let now = self.frame.clip(x, y);
        let start = self.start;
        let mut mask = start;
        #[allow(clippy::cast_possible_truncation)]
        match self.handle {
            MaskHandle::Move => {
                mask.x = (f64::from(start.x) + now.0 - self.grab.0).clamp(-2.0, 2.0) as f32;
                mask.y = (f64::from(start.y) + now.1 - self.grab.1).clamp(-2.0, 2.0) as f32;
            }
            MaskHandle::Corner { .. } => {
                let (a, b) = into_mask(&start, now);
                mask.width = a.abs().clamp(MIN_SIDE, MAX_SIDE) as f32;
                mask.height = b.abs().clamp(MIN_SIDE, MAX_SIDE) as f32;
            }
            MaskHandle::Turn => {
                let middle = (f64::from(start.x), f64::from(start.y));
                let angle = |(u, v): (f64, f64)| (v - middle.1).atan2(u - middle.0);
                let turned = (angle(now) - angle(self.grab)).to_degrees();
                let rotation = f64::from(start.rotation) + turned;
                // Kept within a half turn either way.
                mask.rotation = ((rotation + 180.0).rem_euclid(360.0) - 180.0) as f32;
            }
            MaskHandle::Feather => {
                // Down and out of the box softens.
                let (_, from) = into_mask(&start, self.grab);
                let (_, to) = into_mask(&start, now);
                mask.feather =
                    (f64::from(start.feather) + 2.0 * (from - to)).clamp(0.0, 1.0) as f32;
            }
        }
        mask
    }

    /// The requests that store `mask` as the drag left it: keys at the
    /// time shown when the mask is keyed, or the mask itself, a shared
    /// record copied to `material_id`.
    #[must_use]
    pub fn requests(&self, mask: &Mask, material_id: &MaterialId) -> Vec<Request> {
        if let Some(offset) = self.keyed {
            let values = [
                mask.x,
                mask.y,
                mask.width,
                mask.height,
                mask.rotation,
                mask.feather,
                mask.round,
            ];
            return vec![Request::KeyValues {
                segment_id: self.segment_id.clone(),
                offset,
                values: MASK_KEYS
                    .iter()
                    .zip(values)
                    .map(|(name, value)| {
                        (mask_key(name, self.key_record.as_deref()), f64::from(value))
                    })
                    .collect(),
            }];
        }
        vec![Request::EditMask {
            segment_id: self.segment_id.clone(),
            index: self.index,
            material_id: material_id.clone(),
            mask: *mask,
            art: None,
            color: self.color,
        }]
    }
}

/// Canvas pixel `(x, y)` as a point in `mask`'s box, -1 to 1 each way
/// with y up, where a pen or brush mask's drawing is measured.
#[must_use]
pub fn box_point(mask: &Mask, frame: &ClipFrame, x: f64, y: f64) -> [f32; 2] {
    let (a, b) = into_mask(mask, frame.clip(x, y));
    #[allow(clippy::cast_possible_truncation)]
    [
        (a / f64::from(mask.width).max(1e-4)) as f32,
        (b / f64::from(mask.height).max(1e-4)) as f32,
    ]
}

/// A brush stroke from the canvas pixels a finger went through, `size`
/// pixels wide, in `mask`'s box. Points closer than a pixel and a half
/// are one point. `None` with no points.
#[must_use]
pub fn brush_stroke(
    mask: &Mask,
    frame: &ClipFrame,
    trace: &[(f64, f64)],
    size: f64,
    erase: bool,
) -> Option<BrushStroke> {
    let mut kept: Vec<(f64, f64)> = Vec::with_capacity(trace.len());
    for &point in trace.iter().filter(|(x, y)| x.is_finite() && y.is_finite()) {
        if kept
            .last()
            .is_none_or(|last| (last.0 - point.0).hypot(last.1 - point.1) >= BRUSH_STEP)
        {
            kept.push(point);
        }
    }
    let first = *kept.first()?;
    // The width across the box: a pixel step to the side, in box parts.
    let side = box_point(mask, frame, first.0 + size, first.1);
    let middle = box_point(mask, frame, first.0, first.1);
    let across = (f64::from(side[0] - middle[0])).hypot(f64::from(side[1] - middle[1]));
    #[allow(clippy::cast_possible_truncation)]
    let size = (across as f32).clamp(1e-3, 2.0);
    Some(BrushStroke {
        points: kept
            .iter()
            .map(|&(x, y)| box_point(mask, frame, x, y))
            .collect(),
        size,
        erase,
    })
}

/// `strokes` as a brush mask's drawing, with `stroke` added at the end.
#[must_use]
pub fn brush_art(art: Option<&MaskArt>, stroke: BrushStroke) -> MaskArt {
    let mut strokes = match art {
        Some(MaskArt::Brush { strokes }) => strokes.clone(),
        _ => Vec::new(),
    };
    strokes.push(stroke);
    MaskArt::Brush { strokes }
}

/// A pen mask's drawing through the canvas pixels of `points`, each with
/// its handles as canvas pixel offsets, as [`crate::PenPath`] gives them.
/// `None` with fewer than three points.
#[must_use]
pub fn pen_art(mask: &Mask, frame: &ClipFrame, points: &[ShapePoint]) -> Option<MaskArt> {
    if points.len() < 3 {
        return None;
    }
    let to_box = |(x, y): (f64, f64)| {
        let [a, b] = box_point(mask, frame, x, y);
        [f64::from(a), f64::from(b)]
    };
    let points = points
        .iter()
        .map(|point| {
            let at = to_box((point.at[0], point.at[1]));
            let handle = |offset: [f64; 2]| {
                let end = to_box((point.at[0] + offset[0], point.at[1] + offset[1]));
                [end[0] - at[0], end[1] - at[1]]
            };
            ShapePoint {
                at,
                handle_in: handle(point.handle_in),
                handle_out: handle(point.handle_out),
                round: 0.0,
            }
        })
        .collect();
    let art = MaskArt::Pen { points };
    art.is_valid().then_some(art)
}

/// A text mask's box for words `aspect` times as wide as they are tall on
/// a clip `clip` pixels in size: `width` of the clip across, tall enough
/// that the letters keep their shape.
#[must_use]
pub fn text_box(aspect: f32, clip: (f64, f64), width: f32) -> (f32, f32) {
    let (cw, ch) = (clip.0.max(1.0), clip.1.max(1.0));
    let pixels_across = f64::from(width) * cw;
    #[allow(clippy::cast_possible_truncation)]
    let height = (pixels_across / f64::from(aspect.max(1e-3)) / ch) as f32;
    (width, height.clamp(0.01, 4.0))
}

#[cfg(test)]
mod tests;
