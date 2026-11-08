//! The selected shape: what it looks like, and the edits its inspector
//! makes. Each setter is one undo step.

use cutlass_editor::PenPath;
use cutlass_model::{
    MaterialId, MaterialRef, MaterialShape, SHAPE_FILL, SHAPE_SHADOW, SHAPE_STROKE_WIDTH,
    SegmentId, TrackId, TrackType,
};
use cutlass_session::Request;

use super::{STICKER_LENGTH, Shell, ShellError, placed};

impl Shell {
    /// The shape being drawn with the pen, while the preview takes presses.
    #[must_use]
    pub fn pen(&self) -> Option<&PenPath> {
        self.pen.as_ref()
    }

    /// The pen path, for the preview's presses and drags.
    pub fn pen_mut(&mut self) -> Option<&mut PenPath> {
        self.pen.as_mut()
    }

    /// Pick up the pen: presses on the preview draw a new shape. Picking
    /// and tracking stop.
    pub fn start_pen(&mut self) -> Result<(), ShellError> {
        self.session_draft()?;
        self.picking = None;
        self.tracking = None;
        self.pen = Some(PenPath::new());
        Ok(())
    }

    /// Put the pen down, dropping a path not yet finished.
    pub fn stop_pen(&mut self) {
        self.pen = None;
    }

    /// Place the path drawn so far as a shape at the playhead, on the first
    /// sticker row with room, and select it. The pen stays up for another.
    /// A path of fewer than two points places nothing.
    pub fn finish_pen(&mut self) -> Result<(), ShellError> {
        let Some(pen) = self.pen.as_ref() else {
            return Ok(());
        };
        let draft = self.session_draft()?;
        let canvas = (draft.canvas_config.width, draft.canvas_config.height);
        let Some((points, closed)) = pen.finish(canvas) else {
            return Ok(());
        };
        let start = self.playhead;
        let track_id = cutlass_editor::free_row(draft, TrackType::Sticker, start, STICKER_LENGTH)
            .unwrap_or_else(|| TrackId::new(self.fresh("track")));
        let segment_id = SegmentId::new(self.fresh("seg"));
        let material_id = MaterialId::new(self.fresh("shape"));
        self.pen = Some(PenPath::new());
        self.apply(Request::DrawShape {
            track_id,
            segment_id: segment_id.clone(),
            material_id,
            start,
            duration: STICKER_LENGTH,
            points,
            closed,
        })?;
        self.select(Some(segment_id));
        Ok(())
    }

    /// The selected clip's shape, when it is one.
    #[must_use]
    pub fn shape(&self) -> Option<MaterialShape> {
        let segment_id = self.selected()?;
        let draft = self.draft()?;
        let (segment, _) = placed(draft, segment_id)?;
        match draft.materials.get(segment.material_id.as_str()) {
            Some(MaterialRef::Shape(shape)) => Some(shape.clone()),
            _ => None,
        }
    }

    /// Make the selected shape `width` by `height` canvas pixels.
    pub fn set_shape_size(&mut self, width: f64, height: f64) -> Result<(), ShellError> {
        let segment_id = self.shape_segment()?;
        self.apply(Request::SetShapeSize {
            segment_id,
            width,
            height,
        })
    }

    /// Fill the selected shape with `color` at `opacity`. An empty color
    /// takes the fill off; switched back on it is grey.
    pub fn set_shape_fill(&mut self, color: &str, opacity: f64) -> Result<(), ShellError> {
        let segment_id = self.shape_segment()?;
        self.apply(Request::SetShapeFill {
            segment_id,
            color: color.to_owned(),
            opacity,
        })
    }

    /// Switch the fill on, grey and opaque, or off.
    pub fn toggle_shape_fill(&mut self) -> Result<(), ShellError> {
        let shape = self.shape().ok_or_else(not_shape)?;
        if shape.color.is_empty() {
            self.set_shape_fill(SHAPE_FILL, 1.0)
        } else {
            self.set_shape_fill("", shape.global_alpha)
        }
    }

    /// Stroke the selected shape's outline. A width of 0 takes it off.
    pub fn set_shape_stroke(
        &mut self,
        color: &str,
        width: f64,
        opacity: f64,
        style: i32,
    ) -> Result<(), ShellError> {
        let segment_id = self.shape_segment()?;
        self.apply(Request::SetShapeStroke {
            segment_id,
            color: color.to_owned(),
            width,
            opacity,
            style,
        })
    }

    /// Switch the stroke on at its starting weight, or off.
    pub fn toggle_shape_stroke(&mut self) -> Result<(), ShellError> {
        let shape = self.shape().ok_or_else(not_shape)?;
        let width = if shape.border_width > 0.0 {
            0.0
        } else {
            SHAPE_STROKE_WIDTH
        };
        let opacity = if shape.border_alpha > 0.0 {
            shape.border_alpha
        } else {
            1.0
        };
        self.set_shape_stroke(&shape.border_color, width, opacity, shape.border_line_style)
    }

    /// Round the selected shape's corners: one radius for all, or one each.
    pub fn set_shape_corners(&mut self, radius: Vec<f64>) -> Result<(), ShellError> {
        let segment_id = self.shape_segment()?;
        self.apply(Request::SetShapeCorners { segment_id, radius })
    }

    /// Cast the selected shape's shadow. An opacity of 0 takes it off.
    pub fn set_shape_shadow(
        &mut self,
        color: &str,
        distance: f64,
        angle: f64,
        opacity: f64,
    ) -> Result<(), ShellError> {
        let segment_id = self.shape_segment()?;
        self.apply(Request::SetShapeShadow {
            segment_id,
            color: color.to_owned(),
            distance,
            angle,
            opacity,
        })
    }

    /// Switch the shadow on where it starts, or off.
    pub fn toggle_shape_shadow(&mut self) -> Result<(), ShellError> {
        let shape = self.shape().ok_or_else(not_shape)?;
        if shape.shadow_alpha > 0.0 {
            return self.set_shape_shadow(
                &shape.shadow_color,
                shape.shadow_distance,
                shape.shadow_angle,
                0.0,
            );
        }
        let (distance, angle, opacity) = SHAPE_SHADOW;
        let color = if shape.shadow_color.is_empty() {
            "#000000"
        } else {
            &shape.shadow_color
        };
        self.set_shape_shadow(color, distance, angle, opacity)
    }

    /// The selected clip, when it is an unlocked shape.
    fn shape_segment(&self) -> Result<SegmentId, ShellError> {
        let segment_id = self.selected().cloned().ok_or_else(|| {
            if self.draft().is_none() {
                ShellError::NoProject
            } else {
                ShellError::Unavailable("nothing is selected".to_owned())
            }
        })?;
        if self.shape().is_none() {
            return Err(not_shape());
        }
        self.refuse_locked(&segment_id)?;
        Ok(segment_id)
    }
}

fn not_shape() -> ShellError {
    ShellError::Unavailable("the selected clip is not a shape".to_owned())
}
