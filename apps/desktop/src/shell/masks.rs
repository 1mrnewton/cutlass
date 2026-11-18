//! The selected clip's masks as the Mask page works on them: the list of
//! its clip masks or its colour masks, the one the page has in hand, and
//! the edits on that one. A mask's keys go under its own key names, so a
//! key on one mask never moves another.

use cutlass_effects::{ListedMask, Mask, MaskArt, MaskMode, MaskShape};
use cutlass_eval::Fonts;
use cutlass_model::{MaterialId, TrackType, mask_key};
use cutlass_session::{Request, SegmentId};

use super::inspector::{KeyedSetting, mask_values};
use super::{Shell, ShellError, placed};

/// Which of the selected clip's masks the Mask page has in hand: a place
/// in its clip masks, or with `color` in its colour masks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MaskFocus {
    pub color: bool,
    pub index: usize,
}

/// Drawing a pen or brush mask on the preview.
#[derive(Clone, Debug, PartialEq)]
pub enum MaskDraw {
    /// Points put down one by one, the outline closed on the first.
    Pen(cutlass_editor::PenPath),
    /// Strokes painted, or rubbed out, `size` canvas pixels wide.
    Brush { size: f64, erase: bool },
}

/// One mask of the list, as it is at the playhead.
#[derive(Clone, Debug, PartialEq)]
pub struct MaskRow {
    pub mask: Mask,
    pub art: Option<MaskArt>,
}

impl Shell {
    /// Take the mask at `index` of the selected clip's clip masks, or with
    /// `color` its colour masks, in hand.
    pub fn focus_mask(&mut self, color: bool, index: usize) {
        self.mask_focus = MaskFocus { color, index };
    }

    /// The mask in hand, its place kept inside the list.
    #[must_use]
    pub fn mask_focus(&self) -> MaskFocus {
        let count = self.mask_rows(self.mask_focus.color).len();
        MaskFocus {
            index: self.mask_focus.index.min(count.saturating_sub(1)),
            ..self.mask_focus
        }
    }

    /// The selected clip's clip masks, or with `color` its colour masks, in
    /// order, each as its keys have it at the playhead.
    #[must_use]
    pub fn mask_rows(&self, color: bool) -> Vec<MaskRow> {
        let (Some(draft), Some(segment)) = (self.draft(), self.selected_segment()) else {
            return Vec::new();
        };
        let kind = if color {
            cutlass_effects::COLOR_MASK
        } else {
            cutlass_effects::MASK
        };
        cutlass_effects::listed_masks(draft, segment, kind)
            .into_iter()
            .map(|listed| MaskRow {
                mask: listed.at(segment, &draft.keyframe_graph_list, self.playhead),
                art: listed.art,
            })
            .collect()
    }

    /// The mask in hand as the list holds it.
    fn focused_listed(&self) -> Option<ListedMask> {
        let focus = self.mask_focus();
        let draft = self.draft()?;
        let segment = self.selected_segment()?;
        let kind = if focus.color {
            cutlass_effects::COLOR_MASK
        } else {
            cutlass_effects::MASK
        };
        cutlass_effects::listed_masks(draft, segment, kind)
            .into_iter()
            .nth(focus.index)
    }

    /// The mask in hand as its keys have it at the playhead.
    #[must_use]
    pub fn focused_mask(&self) -> Option<Mask> {
        let row = self.mask_rows(self.mask_focus.color);
        row.into_iter()
            .nth(self.mask_focus().index)
            .map(|row| row.mask)
    }

    /// The key names of `setting`, a mask setting, for the mask in hand:
    /// the plain names for a clip's first clip mask, its own for any
    /// other. `None` for any other setting.
    pub(super) fn mask_setting_names(&self, setting: KeyedSetting) -> Option<Vec<String>> {
        if !is_mask_setting(setting) {
            return None;
        }
        let listed = self.focused_listed();
        let record = listed.as_ref().and_then(ListedMask::key_record);
        Some(
            setting
                .names()
                .iter()
                .map(|name| mask_key(name, record))
                .collect(),
        )
    }

    /// The mask in hand's `setting`, a mask setting, as key values.
    pub(super) fn mask_setting_values(&self, setting: KeyedSetting) -> Option<Vec<f64>> {
        let values = mask_values(&self.focused_mask()?);
        if setting == KeyedSetting::Mask {
            return Some(values.to_vec());
        }
        let (_, range) = KeyedSetting::MASK_PARTS
            .into_iter()
            .find(|(part, _)| *part == setting)?;
        Some(values[range].to_vec())
    }

    /// The clip the masks of the page go on: a picture for clip masks; for
    /// colour masks the selected effect, adjustment layer or picture.
    fn mask_segment(&self, color: bool) -> Result<SegmentId, ShellError> {
        let segment_id = if !color {
            self.picture_segment("a mask belongs on a picture")?
        } else {
            match self.selected() {
                Some(segment_id)
                    if matches!(
                        placed(self.session_draft()?, segment_id),
                        Some((_, TrackType::Effect))
                    ) =>
                {
                    segment_id.clone()
                }
                _ => self.adjust_segment()?,
            }
        };
        self.refuse_locked(&segment_id)?;
        Ok(segment_id)
    }

    /// Put `mask`, drawn from `art`, at the end of the selected clip's clip
    /// masks, or with `color` its colour masks, and take it in hand.
    pub fn add_mask(
        &mut self,
        color: bool,
        mask: Mask,
        art: Option<MaskArt>,
    ) -> Result<(), ShellError> {
        let segment_id = self.mask_segment(color)?;
        let count = self.mask_rows(color).len();
        if count >= cutlass_session::MAX_MASKS {
            return Err(ShellError::Unavailable(
                "a clip takes eight masks at most".to_owned(),
            ));
        }
        let material_id = MaterialId::new(self.fresh(if color { "color-mask" } else { "mask" }));
        self.apply(Request::AddMask {
            segment_id,
            material_id,
            mask,
            art,
            color,
        })?;
        self.focus_mask(color, count);
        Ok(())
    }

    /// Make the mask in hand into `mask`, or take it off with `None`. With
    /// keys on a part of it, a changed part is keyed at the playhead too.
    /// With no mask yet, `mask` is the first.
    pub fn set_focused_mask(&mut self, mask: Option<Mask>) -> Result<(), ShellError> {
        let focus = self.mask_focus();
        let segment_id = self.mask_segment(focus.color)?;
        let was = self.focused_mask();
        match (mask, was) {
            (None, None) => Ok(()),
            (None, Some(_)) => self.apply(Request::RemoveMask {
                segment_id,
                index: focus.index,
                color: focus.color,
            }),
            (Some(mask), None) => self.add_mask(focus.color, mask, None),
            (Some(mask), Some(was)) => {
                let material_id =
                    MaterialId::new(self.fresh(if focus.color { "color-mask" } else { "mask" }));
                let mut requests = vec![Request::EditMask {
                    segment_id,
                    index: focus.index,
                    material_id,
                    mask,
                    art: None,
                    color: focus.color,
                }];
                // Each keyed part that changed is keyed here; the others
                // are only stored.
                let (values, before) = (mask_values(&mask), mask_values(&was));
                for (part, range) in KeyedSetting::MASK_PARTS {
                    if before[range.clone()] != values[range.clone()] {
                        requests.extend(self.key_setting_here(part, values[range].to_vec()));
                    }
                }
                self.apply_requests(&requests)
            }
        }
    }

    /// Give the mask in hand, a text, brush or pen mask, `art` to be drawn
    /// from. One undo step.
    pub fn set_mask_art(&mut self, art: MaskArt) -> Result<(), ShellError> {
        let focus = self.mask_focus();
        let segment_id = self.mask_segment(focus.color)?;
        let listed = self
            .focused_listed()
            .ok_or_else(|| ShellError::Unavailable("pick a mask first".to_owned()))?;
        let mask = Mask {
            shape: art.shape(),
            ..listed.mask
        };
        let material_id =
            MaterialId::new(self.fresh(if focus.color { "color-mask" } else { "mask" }));
        self.apply(Request::EditMask {
            segment_id,
            index: focus.index,
            material_id,
            mask,
            art: Some(art),
            color: focus.color,
        })
    }

    /// Move the mask in hand one place up the list (earlier), or down, and
    /// keep it in hand.
    pub fn move_focused_mask(&mut self, up: bool) -> Result<(), ShellError> {
        let focus = self.mask_focus();
        let count = self.mask_rows(focus.color).len();
        let to = if up {
            focus.index.checked_sub(1)
        } else {
            (focus.index + 1 < count).then_some(focus.index + 1)
        };
        let Some(to) = to else {
            return Ok(());
        };
        let segment_id = self.mask_segment(focus.color)?;
        self.apply(Request::MoveMask {
            segment_id,
            from: focus.index,
            to,
            color: focus.color,
        })?;
        self.focus_mask(focus.color, to);
        Ok(())
    }

    /// How the mask in hand joins the masks before it.
    pub fn set_mask_mode(&mut self, mode: MaskMode) -> Result<(), ShellError> {
        let mask = self
            .focused_mask()
            .ok_or_else(|| ShellError::Unavailable("pick a mask first".to_owned()))?;
        self.set_focused_mask(Some(Mask { mode, ..mask }))
    }

    /// A new mask of `shape` on the selected clip's list, sized for text
    /// as wide as `aspect` says when it is a text mask.
    #[must_use]
    pub fn new_mask(&self, shape: MaskShape, aspect: Option<f32>) -> Mask {
        let mut mask = Mask::new(shape);
        if let Some(aspect) = aspect {
            let clip = self.mask_clip_size().unwrap_or((1920.0, 1080.0));
            let (width, height) = cutlass_editor::text_box(aspect, clip, 0.6);
            mask.width = width;
            mask.height = height;
        }
        mask
    }

    /// The selected clip's drawn size in canvas pixels at the playhead.
    fn mask_clip_size(&self) -> Option<(f64, f64)> {
        let frame = cutlass_editor::ClipFrame(self.mask_frame()?);
        let middle = frame.canvas(0.0, 0.0)?;
        let right = frame.canvas(1.0, 0.0)?;
        let up = frame.canvas(0.0, 1.0)?;
        Some((
            2.0 * (right.0 - middle.0).hypot(right.1 - middle.1),
            2.0 * (up.0 - middle.0).hypot(up.1 - middle.1),
        ))
    }

    /// The selected clip's frame at the playhead: the map from a canvas
    /// pixel to the clip's own picture, where its masks are measured.
    #[must_use]
    pub fn mask_frame(&self) -> Option<[f64; 6]> {
        let draft = self.draft()?;
        let segment = self.selected_segment()?;
        let (width, height) = cutlass_editor::canvas_size(draft);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let canvas = (width.max(1.0) as u32, height.max(1.0) as u32);
        cutlass_eval::clip_frame(draft, segment, self.playhead, canvas)
    }

    /// A text mask's drawing: `text` in the face named `font`.
    #[must_use]
    pub fn text_mask_art(&self, font: &str, text: &str) -> Option<MaskArt> {
        let bytes = self.face.face(font)?;
        cutlass_eval::text_mask_art(bytes, font, text)
    }
}

impl Shell {
    /// Say the Mask page is showing, so the preview shows the mask in
    /// hand's handles rather than moving the clip.
    pub fn note_mask_page(&mut self) {
        self.mask_page_seen = true;
    }

    /// Whether the Mask page showed since this was last asked, and forget
    /// it. A drawing in progress stops when the page has gone.
    pub fn take_mask_page(&mut self) -> bool {
        let seen = std::mem::take(&mut self.mask_page_seen);
        if !seen {
            self.mask_draw = None;
        }
        seen
    }

    /// Whether the Mask page is showing, without forgetting it.
    #[must_use]
    pub fn mask_page_showing(&self) -> bool {
        self.mask_page_seen
    }

    /// The drawing in progress on the preview.
    #[must_use]
    pub fn mask_draw(&self) -> Option<&MaskDraw> {
        self.mask_draw.as_ref()
    }

    pub fn mask_draw_mut(&mut self) -> Option<&mut MaskDraw> {
        self.mask_draw.as_mut()
    }

    /// Start drawing on the preview, or stop with `None`.
    pub fn set_mask_draw(&mut self, draw: Option<MaskDraw>) {
        self.mask_draw = draw;
    }

    /// Store what a drag on a mask's handle asks for. While the pointer is
    /// down the edits join one undo step.
    pub fn apply_mask_requests(&mut self, requests: &[Request]) -> Result<(), ShellError> {
        let segment_id = self.selected().cloned().ok_or(ShellError::NoProject)?;
        self.refuse_locked(&segment_id)?;
        self.apply_requests(requests)
    }

    /// The face a new text mask is set in.
    #[must_use]
    pub fn mask_font(&self) -> String {
        self.face.name().to_owned()
    }

    /// Make the mask in hand `mask` drawn from `art`, as one undo step.
    pub fn set_focused_drawn(&mut self, mask: Mask, art: MaskArt) -> Result<(), ShellError> {
        let focus = self.mask_focus();
        if self.focused_mask().is_none() {
            return self.add_mask(focus.color, mask, Some(art));
        }
        let segment_id = self.mask_segment(focus.color)?;
        let material_id =
            MaterialId::new(self.fresh(if focus.color { "color-mask" } else { "mask" }));
        self.apply(Request::EditMask {
            segment_id,
            index: focus.index,
            material_id,
            mask,
            art: Some(art),
            color: focus.color,
        })
    }

    /// Spell the mask in hand, a text mask, `text`. Refused when the face
    /// cannot set it.
    pub fn set_mask_text(&mut self, text: &str) -> Result<(), ShellError> {
        let art = self
            .text_mask_art(&self.mask_font(), text)
            .ok_or_else(|| ShellError::Unavailable("that text cannot be set".to_owned()))?;
        self.set_mask_art(art)
    }

    /// Close the pen outline drawn on the preview into the mask in hand.
    pub fn finish_mask_pen(&mut self) -> Result<(), ShellError> {
        let Some(MaskDraw::Pen(pen)) = self.mask_draw.take() else {
            return Ok(());
        };
        let (Some(mask), Some(frame)) = (self.focused_mask(), self.mask_frame()) else {
            return Ok(());
        };
        let frame = cutlass_editor::ClipFrame(frame);
        let art = cutlass_editor::pen_art(&mask, &frame, pen.points()).ok_or_else(|| {
            ShellError::Unavailable("a pen mask needs three points or more".to_owned())
        })?;
        self.set_mask_art(art)
    }

    /// Paint, or rub out, a brush stroke through canvas pixels `trace` on
    /// the mask in hand, a brush mask.
    pub fn paint_mask_stroke(&mut self, trace: &[(f64, f64)]) -> Result<(), ShellError> {
        let Some(MaskDraw::Brush { size, erase }) = self.mask_draw.clone() else {
            return Ok(());
        };
        let listed = self
            .focused_listed()
            .ok_or_else(|| ShellError::Unavailable("pick a mask first".to_owned()))?;
        let frame = cutlass_editor::ClipFrame(
            self.mask_frame()
                .ok_or_else(|| ShellError::Unavailable("the clip is not shown here".to_owned()))?,
        );
        let mask = listed.at(
            self.selected_segment().ok_or(ShellError::NoProject)?,
            &self
                .draft()
                .ok_or(ShellError::NoProject)?
                .keyframe_graph_list,
            self.playhead,
        );
        let Some(stroke) = cutlass_editor::brush_stroke(&mask, &frame, trace, size, erase) else {
            return Ok(());
        };
        let art = cutlass_editor::brush_art(listed.art.as_ref(), stroke);
        if !art.is_valid() {
            return Err(ShellError::Unavailable(
                "the brush mask holds no more strokes".to_owned(),
            ));
        }
        self.set_mask_art(art)
    }

    /// Take every mask of the selected clip's list off, as one undo step.
    pub fn clear_masks(&mut self, color: bool) -> Result<(), ShellError> {
        let count = self.mask_rows(color).len();
        if count == 0 {
            return Ok(());
        }
        let segment_id = self.mask_segment(color)?;
        let requests: Vec<Request> = (0..count)
            .map(|_| Request::RemoveMask {
                segment_id: segment_id.clone(),
                index: 0,
                color,
            })
            .collect();
        self.mask_draw = None;
        self.apply_requests(&requests)
    }
}

/// Whether `setting` is one of a mask's.
pub(super) fn is_mask_setting(setting: KeyedSetting) -> bool {
    matches!(
        setting,
        KeyedSetting::Mask
            | KeyedSetting::MaskPosition
            | KeyedSetting::MaskSize
            | KeyedSetting::MaskRotation
            | KeyedSetting::MaskFeather
            | KeyedSetting::MaskRound
    )
}

#[cfg(test)]
mod tests;
