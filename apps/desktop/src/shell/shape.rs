//! The selected shape: what it looks like, and the edits its inspector
//! makes. Each setter is one undo step.

use cutlass_editor::PenPath;
use cutlass_model::{
    MaterialId, MaterialRef, MaterialShape, SHAPE_FILL, SHAPE_SHADOW, SHAPE_STROKE_WIDTH,
    SegmentId, ShapeStyleKey as Key, TrackId, TrackType,
};
use cutlass_session::Request;

use super::{KeyedSetting, STICKER_LENGTH, Shell, ShellError, placed};

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

    /// The selected clip's shape, when it is one, with each setting it
    /// keys as keyed at the playhead.
    #[must_use]
    pub fn shape(&self) -> Option<MaterialShape> {
        let segment_id = self.selected()?;
        let draft = self.draft()?;
        let (segment, _) = placed(draft, segment_id)?;
        match draft.materials.get(segment.material_id.as_str()) {
            Some(MaterialRef::Shape(shape)) => Some(
                cutlass_model::shape_style_at(
                    shape,
                    segment,
                    &draft.keyframe_graph_list,
                    self.playhead,
                )
                .unwrap_or_else(|| shape.clone()),
            ),
            _ => None,
        }
    }

    /// `request` with a key at the playhead for each keyed setting in
    /// `parts` (the setting and its new values) whose value changes. A
    /// keyed setting plays its keys, so the stored value alone would not
    /// show; a setting with no keys is only stored.
    fn with_shape_keys(&self, request: Request, parts: &[(Key, Vec<f64>)]) -> Vec<Request> {
        let now = self.shape();
        let mut requests = vec![request];
        requests.extend(
            parts
                .iter()
                .filter(|(key, values)| {
                    now.as_ref().is_some_and(|now| {
                        key.get(now)
                            .iter()
                            .zip(values)
                            .any(|(was, value)| (was - value).abs() > 1e-9)
                    })
                })
                .filter_map(|(key, values)| {
                    self.key_setting_here(KeyedSetting::ShapeStyle(*key), values.clone())
                }),
        );
        requests
    }

    /// Make the selected shape `width` by `height` canvas pixels.
    pub fn set_shape_size(&mut self, width: f64, height: f64) -> Result<(), ShellError> {
        let segment_id = self.shape_segment()?;
        let size = Request::SetShapeSize {
            segment_id,
            width,
            height,
        };
        // A line keys no height of its own.
        let height = if self.shape().is_some_and(|shape| shape.is_straight()) {
            0.0
        } else {
            height
        };
        let requests = self.with_shape_keys(size, &[(Key::Size, vec![width, height])]);
        self.apply_requests(&requests)
    }

    /// Fill the selected shape with `color` at `opacity`. An empty color
    /// takes the fill off; switched back on it is grey.
    pub fn set_shape_fill(&mut self, color: &str, opacity: f64) -> Result<(), ShellError> {
        let segment_id = self.shape_segment()?;
        let fill = Request::SetShapeFill {
            segment_id,
            color: color.to_owned(),
            opacity,
        };
        let requests = self.with_shape_keys(fill, &[(Key::FillOpacity, vec![opacity])]);
        self.apply_requests(&requests)
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
        let stroke = Request::SetShapeStroke {
            segment_id,
            color: color.to_owned(),
            width,
            opacity,
            style,
        };
        let requests = self.with_shape_keys(
            stroke,
            &[
                (Key::StrokeWidth, vec![width]),
                (Key::StrokeOpacity, vec![opacity]),
            ],
        );
        self.apply_requests(&requests)
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
        // The key holds one radius for every corner.
        let most = radius.iter().copied().fold(0.0_f64, f64::max);
        let corners = Request::SetShapeCorners { segment_id, radius };
        let requests = self.with_shape_keys(corners, &[(Key::Corners, vec![most])]);
        self.apply_requests(&requests)
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
        let shadow = Request::SetShapeShadow {
            segment_id,
            color: color.to_owned(),
            distance,
            angle,
            opacity,
        };
        let requests = self.with_shape_keys(
            shadow,
            &[
                (Key::ShadowDistance, vec![distance]),
                (Key::ShadowAngle, vec![angle]),
                (Key::ShadowOpacity, vec![opacity]),
            ],
        );
        self.apply_requests(&requests)
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
