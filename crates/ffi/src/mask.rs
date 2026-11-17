//! A clip's masks as a phone works on them: the list of its clip masks or
//! its colour masks, each one's handles on the preview, a drag on one, and
//! the pen outlines, brush strokes and words drawn masks are cut from.
//!
//! The geometry and how a change is stored are `cutlass-editor`'s; this
//! names masks by their place in the list and folds a drag or a slider
//! into one undo step.

use cutlass_editor::{ClipFrame, FreshIds, MaskGesture};
use cutlass_effects::{COLOR_MASK, MASK, Mask, MaskArt, MaskMode, MaskShape, listed_masks};
use cutlass_eval::Fonts;
use cutlass_model::{Draft, MaterialId, Segment, ShapePoint};
use cutlass_session::{MAX_MASKS, Micros, Request, SegmentId};

use crate::{FfiError, PhoneHost, clip};

/// How far from a knob a finger still takes it, in canvas pixels per
/// thousand pixels of the canvas's longer side.
const REACH: f64 = 28.0;
/// What a new text mask says.
const NEW_MASK_TEXT: &str = "Text";

/// One mask of a clip's list, as its keys have it at the time asked for.
#[derive(Clone, Debug, PartialEq)]
pub struct PhoneMask {
    /// The shape's key: `linear`, `mirror`, `circle`, `rectangle`,
    /// `heart`, `star`, `text`, `brush` or `pen`.
    pub shape: String,
    /// How it joins the masks before it: `add`, `subtract` or
    /// `intersect`.
    pub mode: String,
    /// Its middle in the clip, -1 to 1 each way, y up.
    pub x: f64,
    pub y: f64,
    /// Half its box across and up, in clip halves.
    pub width: f64,
    pub height: f64,
    /// Degrees, anticlockwise.
    pub rotation: f64,
    /// From 0 hard to 1 softest.
    pub feather: f64,
    /// Corner rounding for a rectangle, 0 to 1.
    pub round: f64,
    pub invert: bool,
    /// Its numbers have keys, so a change keys them at the time shown.
    pub keyed: bool,
    /// What a text mask says.
    pub text: Option<String>,
}

impl PhoneMask {
    fn new(mask: &Mask, keyed: bool, art: Option<&MaskArt>) -> Self {
        Self {
            shape: mask.shape.key().to_owned(),
            mode: mask.mode.key().to_owned(),
            x: f64::from(mask.x),
            y: f64::from(mask.y),
            width: f64::from(mask.width),
            height: f64::from(mask.height),
            rotation: f64::from(mask.rotation),
            feather: f64::from(mask.feather),
            round: f64::from(mask.round),
            invert: mask.invert,
            keyed,
            text: match art {
                Some(MaskArt::Text { text, .. }) => Some(text.clone()),
                _ => None,
            },
        }
    }

    /// This as a mask; `None` for a shape or mode no mask has.
    #[allow(clippy::cast_possible_truncation)]
    fn mask(&self) -> Option<Mask> {
        Some(Mask {
            shape: MaskShape::from_key(&self.shape)?,
            mode: MaskMode::from_key(&self.mode)?,
            x: self.x as f32,
            y: self.y as f32,
            width: self.width as f32,
            height: self.height as f32,
            rotation: self.rotation as f32,
            feather: self.feather as f32,
            round: self.round as f32,
            invert: self.invert,
        })
    }
}

/// A mask's handles on the canvas, in canvas pixels with y down.
#[derive(Clone, Debug, PartialEq)]
pub struct PhoneMaskHandles {
    pub canvas_width: f64,
    pub canvas_height: f64,
    pub middle_x: f64,
    pub middle_y: f64,
    /// The box's corners as x, y pairs: top left, top right, bottom right,
    /// bottom left in the mask's own frame.
    pub corners: Vec<f64>,
    pub turn_x: f64,
    pub turn_y: f64,
    pub feather_x: f64,
    pub feather_y: f64,
    /// For a line or a mirror, two points on its line as x, y pairs; empty
    /// for any other shape.
    pub line: Vec<f64>,
    /// How far from a knob a finger still takes it, in canvas pixels.
    pub reach: f64,
}

/// A brush stroke's width in canvas pixels, and whether it rubs out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhoneBrush {
    pub size: f64,
    pub erase: bool,
}

/// A shape a mask can take, and its name for a tile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneMaskShape {
    pub key: String,
    pub label: String,
}

/// Every mask shape in the order a panel shows them.
#[must_use]
pub fn mask_shapes() -> Vec<PhoneMaskShape> {
    MaskShape::ALL
        .into_iter()
        .chain(MaskShape::DRAWN)
        .map(|shape| PhoneMaskShape {
            key: shape.key().to_owned(),
            label: shape.label().to_owned(),
        })
        .collect()
}

/// The mask drag in flight, and the revision and mask of its last change.
pub(crate) struct LiveMask {
    gesture: MaskGesture,
    applied: Option<(u64, Mask)>,
}

fn kind(color: bool) -> &'static str {
    if color { COLOR_MASK } else { MASK }
}

fn usize_of(index: u32) -> usize {
    usize::try_from(index).unwrap_or(usize::MAX)
}

fn refused(reason: &str) -> FfiError {
    FfiError::Refused {
        reason: reason.to_owned(),
    }
}

/// `segment`'s frame at `time`: the map from a canvas pixel to its own
/// picture, where its masks are measured.
fn frame_of(draft: &Draft, segment: &Segment, time: Micros) -> Option<ClipFrame> {
    let (width, height) = cutlass_editor::canvas_size(draft);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let canvas = (width.max(1.0) as u32, height.max(1.0) as u32);
    cutlass_eval::clip_frame(draft, segment, time, canvas).map(ClipFrame)
}

fn reach(draft: &Draft) -> f64 {
    let (width, height) = cutlass_editor::canvas_size(draft);
    width.max(height) * REACH / 1000.0
}

/// Whether the keyed numbers of the mask at `index` have keys.
fn has_keys(segment: &Segment, record: Option<&str>) -> bool {
    let names = cutlass_model::MASK_KEYS.map(|name| cutlass_model::mask_key(name, record));
    segment
        .common_keyframes
        .iter()
        .any(|group| names.contains(&group.property_type) && !group.keyframe_list.is_empty())
}

impl PhoneHost {
    /// `segment_id`'s clip masks, or with `color` its colour masks, in
    /// order, each as its keys have it at `time`.
    #[must_use]
    pub fn masks(&self, segment_id: &SegmentId, color: bool, time: Micros) -> Vec<PhoneMask> {
        let Some((segment, _)) = clip::locate(self.draft(), segment_id) else {
            return Vec::new();
        };
        listed_masks(self.draft(), segment, kind(color))
            .iter()
            .map(|listed| {
                let mask = listed.at(segment, &self.draft().keyframe_graph_list, time);
                let keyed = has_keys(segment, listed.key_record());
                PhoneMask::new(&mask, keyed, listed.art.as_ref())
            })
            .collect()
    }

    /// Where the handles of the mask at `index` are on the canvas at
    /// `time`; `None` when there is no such mask or the clip is not shown.
    #[must_use]
    pub fn mask_handles(
        &self,
        segment_id: &SegmentId,
        color: bool,
        index: u32,
        time: Micros,
    ) -> Option<PhoneMaskHandles> {
        let draft = self.draft();
        let (segment, _) = clip::locate(draft, segment_id)?;
        let listed = listed_masks(draft, segment, kind(color))
            .into_iter()
            .nth(usize_of(index))?;
        let mask = listed.at(segment, &draft.keyframe_graph_list, time);
        let frame = frame_of(draft, segment, time)?;
        let handles = cutlass_editor::mask_handles(&mask, &frame)?;
        let (canvas_width, canvas_height) = cutlass_editor::canvas_size(draft);
        Some(PhoneMaskHandles {
            canvas_width,
            canvas_height,
            middle_x: handles.middle.0,
            middle_y: handles.middle.1,
            corners: handles.corners.iter().flat_map(|(x, y)| [*x, *y]).collect(),
            turn_x: handles.turn.0,
            turn_y: handles.turn.1,
            feather_x: handles.feather.0,
            feather_y: handles.feather.1,
            line: handles
                .line
                .iter()
                .flatten()
                .flat_map(|(x, y)| [*x, *y])
                .collect(),
            reach: reach(draft),
        })
    }

    /// Put a new mask of the shape keyed `shape` at the end of
    /// `segment_id`'s clip masks, or with `color` its colour masks, as one
    /// undo step, and give back its place. A text mask starts as a word in
    /// the first face; a brush or pen mask starts with nothing drawn.
    pub fn add_mask(
        &mut self,
        segment_id: &SegmentId,
        color: bool,
        shape: &str,
        time: Micros,
    ) -> Result<u32, FfiError> {
        let (segment, _) = clip::unlocked(self.draft(), segment_id)?;
        let shape = MaskShape::from_key(shape).ok_or_else(|| refused("no mask has that shape"))?;
        let count = listed_masks(self.draft(), segment, kind(color)).len();
        if count >= MAX_MASKS {
            return Err(refused("a clip takes eight masks at most"));
        }
        let mut mask = Mask::new(shape);
        let art = if shape == MaskShape::Text {
            let art = self
                .text_art(self.faces.name(), NEW_MASK_TEXT)
                .ok_or_else(|| refused("no face can set a text mask"))?;
            self.fit_text(segment, time, &mut mask, &art);
            Some(art)
        } else {
            None
        };
        let material_id = self.fresh_mask_id(color);
        self.apply(
            self.revision(),
            &[Request::AddMask {
                segment_id: segment_id.clone(),
                material_id,
                mask,
                art,
                color,
            }],
        )?;
        u32::try_from(count).map_err(|_| refused("too many masks"))
    }

    /// Make the mask at `index` into `mask` at `time`: its keyed numbers
    /// are keyed there when it has keys. Changes made while `live` join one
    /// undo step, so a slider drag is one step; its last change passes
    /// `live` false, and so does a single tap.
    pub fn edit_mask(
        &mut self,
        segment_id: &SegmentId,
        color: bool,
        index: u32,
        time: Micros,
        mask: &PhoneMask,
        live: bool,
    ) -> Result<(), FfiError> {
        clip::unlocked(self.draft(), segment_id)?;
        let mask = mask
            .mask()
            .filter(Mask::is_valid)
            .ok_or_else(|| refused("that is not a mask"))?;
        let material_id = self.fresh_mask_id(color);
        let requests = cutlass_editor::mask_edit_requests(
            self.draft(),
            segment_id,
            (usize_of(index), color),
            time,
            &mask,
            &material_id,
        )
        .ok_or_else(|| refused("the clip has no mask there"))?;
        self.restyle(segment_id, &requests, live)
    }

    /// Take the mask at `index` off, as one undo step.
    pub fn remove_mask(
        &mut self,
        segment_id: &SegmentId,
        color: bool,
        index: u32,
    ) -> Result<(), FfiError> {
        clip::unlocked(self.draft(), segment_id)?;
        self.apply(
            self.revision(),
            &[Request::RemoveMask {
                segment_id: segment_id.clone(),
                index: usize_of(index),
                color,
            }],
        )?;
        Ok(())
    }

    /// Move the mask at `from` to `to` in the list, as one undo step.
    pub fn move_mask(
        &mut self,
        segment_id: &SegmentId,
        color: bool,
        from: u32,
        to: u32,
    ) -> Result<(), FfiError> {
        clip::unlocked(self.draft(), segment_id)?;
        self.apply(
            self.revision(),
            &[Request::MoveMask {
                segment_id: segment_id.clone(),
                from: usize_of(from),
                to: usize_of(to),
                color,
            }],
        )?;
        Ok(())
    }

    /// Spell the text mask at `index` `text`, in the first face, as one
    /// undo step. Refused when the face cannot set it.
    pub fn set_mask_text(
        &mut self,
        segment_id: &SegmentId,
        color: bool,
        index: u32,
        text: &str,
    ) -> Result<(), FfiError> {
        let art = self
            .text_art(self.faces.name(), text)
            .ok_or_else(|| refused("that text cannot be set"))?;
        self.set_art(segment_id, color, index, art)
    }

    /// Paint, or rub out, a `brush` stroke through the canvas points `trace` (x, y pairs) on the brush
    /// mask at `index`, as one undo step.
    pub fn paint_mask(
        &mut self,
        segment_id: &SegmentId,
        color: bool,
        index: u32,
        time: Micros,
        trace: &[f64],
        brush: PhoneBrush,
    ) -> Result<(), FfiError> {
        let PhoneBrush { size, erase } = brush;
        let draft = self.draft();
        let (segment, _) = clip::unlocked(draft, segment_id)?;
        let listed = listed_masks(draft, segment, kind(color))
            .into_iter()
            .nth(usize_of(index))
            .ok_or_else(|| refused("the clip has no mask there"))?;
        let mask = listed.at(segment, &draft.keyframe_graph_list, time);
        let frame =
            frame_of(draft, segment, time).ok_or_else(|| refused("the clip is not shown here"))?;
        let points: Vec<(f64, f64)> = trace
            .as_chunks::<2>()
            .0
            .iter()
            .map(|[x, y]| (*x, *y))
            .collect();
        let Some(stroke) = cutlass_editor::brush_stroke(&mask, &frame, &points, size, erase) else {
            return Ok(());
        };
        let art = cutlass_editor::brush_art(listed.art.as_ref(), stroke);
        if !art.is_valid() {
            return Err(refused("the brush mask holds no more strokes"));
        }
        self.set_art(segment_id, color, index, art)
    }

    /// Make the pen mask at `index` the outline through the canvas points
    /// `corners` (x, y pairs), closed on the first, as one undo step.
    pub fn draw_mask_outline(
        &mut self,
        segment_id: &SegmentId,
        color: bool,
        index: u32,
        time: Micros,
        corners: &[f64],
    ) -> Result<(), FfiError> {
        let draft = self.draft();
        let (segment, _) = clip::unlocked(draft, segment_id)?;
        let listed = listed_masks(draft, segment, kind(color))
            .into_iter()
            .nth(usize_of(index))
            .ok_or_else(|| refused("the clip has no mask there"))?;
        let mask = listed.at(segment, &draft.keyframe_graph_list, time);
        let frame =
            frame_of(draft, segment, time).ok_or_else(|| refused("the clip is not shown here"))?;
        let points: Vec<ShapePoint> = corners
            .as_chunks::<2>()
            .0
            .iter()
            .map(|at| ShapePoint {
                at: *at,
                ..ShapePoint::default()
            })
            .collect();
        let art = cutlass_editor::pen_art(&mask, &frame, &points)
            .ok_or_else(|| refused("a pen mask needs three points or more"))?;
        self.set_art(segment_id, color, index, art)
    }

    /// The handle of the mask at `index` under canvas point `(x, y)` at
    /// `time`, taken hold of for a drag; `false` when none is there. A drag
    /// still open is ended first.
    pub fn mask_begin(
        &mut self,
        segment_id: &SegmentId,
        color: bool,
        index: u32,
        time: Micros,
        x: f64,
        y: f64,
    ) -> Result<bool, FfiError> {
        self.mask_live = None;
        let draft = self.draft();
        let (segment, _) = clip::unlocked(draft, segment_id)?;
        let place = (usize_of(index), color);
        let mask = listed_masks(draft, segment, kind(color))
            .into_iter()
            .nth(place.0)
            .ok_or_else(|| refused("the clip has no mask there"))?
            .at(segment, &draft.keyframe_graph_list, time);
        let frame =
            frame_of(draft, segment, time).ok_or_else(|| refused("the clip is not shown here"))?;
        let Some(handle) = cutlass_editor::mask_pick(&mask, &frame, x, y, reach(draft)) else {
            return Ok(false);
        };
        let gesture = MaskGesture::begin(draft, segment_id, place, frame, time, handle, (x, y))
            .ok_or_else(|| refused("the clip has no mask there"))?;
        self.mask_live = Some(LiveMask {
            gesture,
            applied: None,
        });
        Ok(true)
    }

    /// Move the handle taken hold of to canvas point `(x, y)`, and give
    /// back the mask as it now is. The first change is an undo step and
    /// later ones join it.
    pub fn mask_drag(&mut self, x: f64, y: f64) -> Result<PhoneMask, FfiError> {
        let live = self
            .mask_live
            .as_ref()
            .ok_or_else(|| refused("no mask drag is open"))?;
        let mask = live.gesture.step(x, y);
        let shown = PhoneMask::new(&mask, false, None);
        let revision = self.revision();
        let joins = match live.applied {
            Some((_, applied)) if applied == mask => return Ok(shown),
            Some((applied, _)) => applied == revision,
            None if *live.gesture.start() == mask => return Ok(shown),
            None => false,
        };
        let color = live.gesture.color();
        let requests = live.gesture.requests(&mask, &self.fresh_mask_id(color));
        let revision = if joins {
            self.commands.fold(revision, &requests)?
        } else {
            self.apply(revision, &requests)?
        };
        if let Some(live) = self.mask_live.as_mut() {
            live.applied = Some((revision, mask));
        }
        Ok(shown)
    }

    /// End the mask drag. A drag that changed nothing leaves no step.
    pub fn mask_end(&mut self) {
        self.mask_live = None;
    }

    /// Give the mask at `index` `art` to be drawn from, its shape following
    /// the art's, as one undo step.
    fn set_art(
        &mut self,
        segment_id: &SegmentId,
        color: bool,
        index: u32,
        art: MaskArt,
    ) -> Result<(), FfiError> {
        let (segment, _) = clip::unlocked(self.draft(), segment_id)?;
        let listed = listed_masks(self.draft(), segment, kind(color))
            .into_iter()
            .nth(usize_of(index))
            .ok_or_else(|| refused("the clip has no mask there"))?;
        let mask = Mask {
            shape: art.shape(),
            ..listed.mask
        };
        let material_id = self.fresh_mask_id(color);
        self.apply(
            self.revision(),
            &[Request::EditMask {
                segment_id: segment_id.clone(),
                index: usize_of(index),
                material_id,
                mask,
                art: Some(art),
                color,
            }],
        )?;
        Ok(())
    }

    fn text_art(&self, font: &str, text: &str) -> Option<MaskArt> {
        let bytes = self.faces.face(font)?;
        cutlass_eval::text_mask_art(bytes, font, text)
    }

    /// Size `mask`'s box so the letters of `art` keep their shape on
    /// `segment`'s picture, six tenths of it across.
    fn fit_text(&self, segment: &Segment, time: Micros, mask: &mut Mask, art: &MaskArt) {
        let MaskArt::Text { aspect, .. } = art else {
            return;
        };
        let size = frame_of(self.draft(), segment, time)
            .and_then(|frame| {
                let middle = frame.canvas(0.0, 0.0)?;
                let right = frame.canvas(1.0, 0.0)?;
                let up = frame.canvas(0.0, 1.0)?;
                Some((
                    2.0 * (right.0 - middle.0).hypot(right.1 - middle.1),
                    2.0 * (up.0 - middle.0).hypot(up.1 - middle.1),
                ))
            })
            .unwrap_or((1920.0, 1080.0));
        let (width, height) = cutlass_editor::text_box(*aspect, size, 0.6);
        mask.width = width;
        mask.height = height;
    }

    fn fresh_mask_id(&self, color: bool) -> MaterialId {
        MaterialId::new(FreshIds::new(self.draft(), 0).make(if color {
            "color-mask"
        } else {
            "mask"
        }))
    }
}

#[cfg(test)]
mod tests;
