//! How a shape looks on the phone: its fill, outline, corners and shadow,
//! read from the clip and changed one request at a time, and the pen's
//! color and weight for a freehand stroke.

use cutlass_model::{
    Draft, MaterialRef, MaterialShape, SHAPE_ARROW, SHAPE_FILL, SHAPE_LINE, SHAPE_LINE_WIDTH,
    SHAPE_SHADOW, SHAPE_STROKE_WIDTH, SegmentId,
};
use cutlass_session::Request;

/// How a shape looks now.
#[derive(Clone, Debug, PartialEq)]
pub struct PhoneShapeStyle {
    /// An open line, drawn by its outline: it has no fill and no corners.
    pub line: bool,
    /// `#RRGGBB`, or empty when it has no fill.
    pub fill: String,
    pub fill_opacity: f64,
    /// `#RRGGBB`; the outline is off while its width is 0.
    pub stroke: String,
    /// Canvas pixels.
    pub stroke_width: f64,
    pub stroke_opacity: f64,
    /// Solid (0), dashed (1) or dotted (2).
    pub stroke_style: i32,
    /// The largest corner radius, canvas pixels.
    pub corners: f64,
    /// How far a corner can round: half the shorter side.
    pub corners_max: f64,
    /// `#RRGGBB`; the shadow is off while its opacity is 0.
    pub shadow: String,
    pub shadow_opacity: f64,
    pub shadow_distance: f64,
    pub shadow_angle: f64,
}

/// One change to how a shape looks.
#[derive(Clone, Debug, PartialEq)]
pub enum ShapeStyleChange {
    /// Fill with `color` at `opacity`; an empty color takes the fill off.
    Fill { color: String, opacity: f64 },
    /// Draw the outline `width` canvas pixels wide; 0 takes it off.
    Stroke {
        color: String,
        width: f64,
        opacity: f64,
        style: i32,
    },
    /// Round every corner by `radius` canvas pixels.
    Corners(f64),
    /// Cast a shadow; an opacity of 0 takes it off.
    Shadow {
        color: String,
        distance: f64,
        angle: f64,
        opacity: f64,
    },
}

/// The shape panel's slider ends and where a part starts when switched on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapeStyleRanges {
    /// The fill a shape gets back when its fill is switched on.
    pub fill_start: &'static str,
    /// Canvas pixels.
    pub stroke_max: f64,
    pub stroke_start: f64,
    pub shadow_distance_max: f64,
    /// Distance, angle in degrees, and opacity.
    pub shadow_start: (f64, f64, f64),
    /// The pen's weight, thinnest, thickest and first.
    pub pen: (f64, f64, f64),
}

/// See [`ShapeStyleRanges`]. The stroke and shadow ends sit well inside
/// what a shape can hold, so a finger can hit a value.
pub const SHAPE_STYLE_RANGES: ShapeStyleRanges = ShapeStyleRanges {
    fill_start: SHAPE_FILL,
    stroke_max: 64.0,
    stroke_start: SHAPE_STROKE_WIDTH,
    shadow_distance_max: 120.0,
    shadow_start: SHAPE_SHADOW,
    pen: (2.0, 40.0, SHAPE_LINE_WIDTH),
};

/// The shape `segment_id` shows, or `None` when it is not a shape.
fn shape_of<'a>(draft: &'a Draft, segment_id: &SegmentId) -> Option<&'a MaterialShape> {
    let (segment, _) = crate::clip::locate(draft, segment_id)?;
    match draft.materials.get(segment.material_id.as_str()) {
        Some(MaterialRef::Shape(shape)) => Some(shape),
        _ => None,
    }
}

/// How `segment_id` looks, or `None` when it is not a shape.
pub(crate) fn style_of(draft: &Draft, segment_id: &SegmentId) -> Option<PhoneShapeStyle> {
    let shape = shape_of(draft, segment_id)?;
    let (width, height) = shape.size();
    Some(PhoneShapeStyle {
        line: matches!(shape.shape_type, SHAPE_LINE | SHAPE_ARROW),
        fill: shape.color.clone(),
        fill_opacity: shape.global_alpha,
        stroke: shape.border_color.clone(),
        stroke_width: shape.border_width,
        stroke_opacity: shape.border_alpha,
        stroke_style: shape.border_line_style,
        corners: shape.roundness.iter().copied().fold(0.0, f64::max),
        corners_max: (width.min(height) / 2.0).max(0.0),
        shadow: shape.shadow_color.clone(),
        shadow_opacity: shape.shadow_alpha,
        shadow_distance: shape.shadow_distance,
        shadow_angle: shape.shadow_angle,
    })
}

/// The request that makes `change` to `segment_id`.
pub(crate) fn style_request(segment_id: &SegmentId, change: ShapeStyleChange) -> Request {
    let segment_id = segment_id.clone();
    match change {
        ShapeStyleChange::Fill { color, opacity } => Request::SetShapeFill {
            segment_id,
            color,
            opacity,
        },
        ShapeStyleChange::Stroke {
            color,
            width,
            opacity,
            style,
        } => Request::SetShapeStroke {
            segment_id,
            color,
            width,
            opacity,
            style,
        },
        ShapeStyleChange::Corners(radius) => Request::SetShapeCorners {
            segment_id,
            radius: vec![radius],
        },
        ShapeStyleChange::Shadow {
            color,
            distance,
            angle,
            opacity,
        } => Request::SetShapeShadow {
            segment_id,
            color,
            distance,
            angle,
            opacity,
        },
    }
}

/// The request that gives a fresh pen line `color` and `width`.
pub(crate) fn pen_request(segment_id: &SegmentId, color: &str, width: f64) -> Request {
    let (thinnest, thickest, _) = SHAPE_STYLE_RANGES.pen;
    Request::SetShapeStroke {
        segment_id: segment_id.clone(),
        color: color.to_owned(),
        width: width.clamp(thinnest, thickest),
        opacity: 1.0,
        style: 0,
    }
}
