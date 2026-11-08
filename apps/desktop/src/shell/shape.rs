//! The selected shape: what it looks like, and the edits its inspector
//! makes. Each setter is one undo step.

use cutlass_model::{
    MaterialRef, MaterialShape, SHAPE_FILL, SHAPE_SHADOW, SHAPE_STROKE_WIDTH, SegmentId,
};
use cutlass_session::Request;

use super::{Shell, ShellError, placed};

impl Shell {
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
