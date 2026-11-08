use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::id::MaterialId;

/// `shape_type` for an outline of points with in and out handles. The
/// rectangle, circle and polygons are stored this way, so a resize rewrites
/// their points.
pub const SHAPE_CONTOUR: i32 = 4;
/// `shape_type` for an open line, drawn by its stroke.
pub const SHAPE_LINE: i32 = 5;
/// `shape_type` for a line with a filled head at its end.
pub const SHAPE_ARROW: i32 = 6;

/// A new shape's width and height, in canvas pixels.
pub const SHAPE_SIZE: f64 = 320.0;
/// A new shape's fill.
pub const SHAPE_FILL: &str = "#a6a6a6";
/// A stroke's weight when it is first switched on, in canvas pixels.
pub const SHAPE_STROKE_WIDTH: f64 = 4.0;
/// A line's weight when it is placed, in canvas pixels.
pub const SHAPE_LINE_WIDTH: f64 = 8.0;
/// A shadow's distance, angle (degrees) and opacity when it is first
/// switched on.
pub const SHAPE_SHADOW: (f64, f64, f64) = (10.0, 45.0, 0.5);

/// The shapes the library offers. Each is stored as points, so the kind is
/// kept in the material's name, not in `shape_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShapeKind {
    Rectangle,
    Circle,
    Triangle,
    Parallelogram,
    Trapezoid,
    Line,
    Arrow,
}

impl ShapeKind {
    pub const ALL: [Self; 7] = [
        Self::Rectangle,
        Self::Circle,
        Self::Triangle,
        Self::Parallelogram,
        Self::Trapezoid,
        Self::Line,
        Self::Arrow,
    ];

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Rectangle => "Rectangle",
            Self::Circle => "Circle",
            Self::Triangle => "Triangle",
            Self::Parallelogram => "Parallelogram",
            Self::Trapezoid => "Trapezoid",
            Self::Line => "Line",
            Self::Arrow => "Arrow",
        }
    }

    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }

    /// A line or an arrow: open, drawn by its stroke, with no fill.
    #[must_use]
    pub const fn is_line(self) -> bool {
        matches!(self, Self::Line | Self::Arrow)
    }
}

/// The handle length that makes four cubics a circle, as a part of the
/// radius.
const CIRCLE_HANDLE: f64 = 0.552_284_7;

/// One point of an outline: where it is, its in and out handles as offsets
/// from it, and how far its corner is rounded. Canvas pixels from the
/// shape's centre, y up.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ShapePoint {
    pub at: [f64; 2],
    pub handle_in: [f64; 2],
    pub handle_out: [f64; 2],
    pub round: f64,
}

/// A shape. Roundness, custom points, size, scale, and the in and out tangents
/// are arrays of doubles. A lone JSON number is read as a one-element array.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct MaterialShape {
    #[serde(default)]
    pub id: MaterialId,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub shape_type: i32,
    #[serde(default, deserialize_with = "crate::serde_util::one_or_many_f64")]
    pub roundness: Vec<f64>,
    #[serde(default, deserialize_with = "crate::serde_util::one_or_many_f64")]
    pub custom_points: Vec<f64>,
    #[serde(default, deserialize_with = "crate::serde_util::one_or_many_f64")]
    pub shape_size: Vec<f64>,
    #[serde(default)]
    pub global_alpha: f64,
    #[serde(default)]
    pub color: String,
    #[serde(default)]
    pub border_line_style: i32,
    #[serde(default)]
    pub border_width: f64,
    #[serde(default)]
    pub border_color: String,
    #[serde(default)]
    pub shadow_color: String,
    #[serde(default)]
    pub shadow_alpha: f64,
    #[serde(default)]
    pub shadow_distance: f64,
    #[serde(default)]
    pub shadow_angle: f64,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub combo_info: Option<Value>,
    #[serde(default, deserialize_with = "crate::serde_util::one_or_many_f64")]
    pub shape_scale: Vec<f64>,
    #[serde(default, deserialize_with = "crate::serde_util::one_or_many_f64")]
    pub custom_points_in: Vec<f64>,
    #[serde(default, deserialize_with = "crate::serde_util::one_or_many_f64")]
    pub custom_points_out: Vec<f64>,
    #[serde(default)]
    pub border_alpha: f64,
    #[serde(default)]
    pub shadow_ambiguity: f64,
    #[serde(default)]
    pub endpoint_left_style: i32,
    #[serde(default)]
    pub endpoint_right_style: i32,
    #[serde(default)]
    pub line_style: i32,
    #[serde(default)]
    pub fill_render_style: Option<Value>,
    #[serde(default)]
    pub constant_material_id: String,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl MaterialShape {
    /// A new `kind` shape, `width` by `height` canvas pixels, filled grey;
    /// a line or an arrow is stroked grey instead.
    #[must_use]
    pub fn preset(id: MaterialId, kind: ShapeKind, width: f64, height: f64) -> Self {
        let (half_w, half_h) = (width / 2.0, height / 2.0);
        let corners: &[[f64; 2]] = match kind {
            ShapeKind::Rectangle => &[[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]],
            ShapeKind::Triangle => &[[0.0, 1.0], [1.0, -1.0], [-1.0, -1.0]],
            ShapeKind::Parallelogram => &[[-0.7, 1.0], [1.0, 1.0], [0.7, -1.0], [-1.0, -1.0]],
            ShapeKind::Trapezoid => &[[-0.75, 1.0], [0.75, 1.0], [1.0, -1.0], [-1.0, -1.0]],
            ShapeKind::Line | ShapeKind::Arrow => &[[-1.0, 0.0], [1.0, 0.0]],
            ShapeKind::Circle => &[[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0]],
        };
        let mut points = Vec::new();
        let mut handle_in = Vec::new();
        let mut handle_out = Vec::new();
        for (index, [x, y]) in corners.iter().enumerate() {
            points.extend([x * half_w, y * half_h]);
            // Clockwise round the circle: each handle runs along the
            // tangent, a fixed part of the radius long.
            let (inward, outward) = if kind == ShapeKind::Circle {
                let k = CIRCLE_HANDLE;
                match index {
                    0 => ([0.0, -k], [0.0, k]),
                    1 => ([-k, 0.0], [k, 0.0]),
                    2 => ([0.0, k], [0.0, -k]),
                    _ => ([k, 0.0], [-k, 0.0]),
                }
            } else {
                ([0.0, 0.0], [0.0, 0.0])
            };
            handle_in.extend([inward[0] * half_w, inward[1] * half_h]);
            handle_out.extend([outward[0] * half_w, outward[1] * half_h]);
        }
        let count = corners.len();
        let line = kind.is_line();
        Self {
            id,
            kind: "shape".to_owned(),
            shape_type: match kind {
                ShapeKind::Line => SHAPE_LINE,
                ShapeKind::Arrow => SHAPE_ARROW,
                _ => SHAPE_CONTOUR,
            },
            roundness: vec![0.0; count],
            custom_points: points,
            custom_points_in: handle_in,
            custom_points_out: handle_out,
            shape_size: vec![width, if line { 0.0 } else { height }],
            shape_scale: vec![1.0, 1.0],
            global_alpha: 1.0,
            color: if line {
                String::new()
            } else {
                SHAPE_FILL.to_owned()
            },
            border_width: if line { SHAPE_LINE_WIDTH } else { 0.0 },
            border_color: SHAPE_FILL.to_owned(),
            border_alpha: 1.0,
            shadow_color: "#000000".to_owned(),
            name: kind.name().to_owned(),
            ..Self::default()
        }
    }

    /// Which library shape this is, read from its name.
    #[must_use]
    pub fn shape_kind(&self) -> Option<ShapeKind> {
        ShapeKind::from_name(&self.name)
    }

    /// The outline's points. Missing handles or roundness read as none.
    #[must_use]
    pub fn points(&self) -> Vec<ShapePoint> {
        let pair = |values: &[f64], index: usize| {
            [
                values.get(index * 2).copied().unwrap_or(0.0),
                values.get(index * 2 + 1).copied().unwrap_or(0.0),
            ]
        };
        (0..self.custom_points.len() / 2)
            .map(|index| ShapePoint {
                at: pair(&self.custom_points, index),
                handle_in: pair(&self.custom_points_in, index),
                handle_out: pair(&self.custom_points_out, index),
                round: self.roundness.get(index).copied().unwrap_or(0.0),
            })
            .collect()
    }

    /// The width and height in canvas pixels: the stored size, else the
    /// span of the points.
    #[must_use]
    pub fn size(&self) -> (f64, f64) {
        if let [width, height, ..] = self.shape_size[..] {
            return (width, height);
        }
        let points = self.points();
        let span = |axis: usize| {
            let (low, high) = points
                .iter()
                .fold((0.0_f64, 0.0_f64), |(low, high), point| {
                    (low.min(point.at[axis]), high.max(point.at[axis]))
                });
            high - low
        };
        (span(0), span(1))
    }

    /// Make the shape `width` by `height`: the points and their handles are
    /// stretched to it, and the rounding stays.
    pub fn resize(&mut self, width: f64, height: f64) {
        let (old_w, old_h) = self.size();
        let ratio = |new: f64, old: f64| {
            if old.abs() > f64::EPSILON {
                new / old
            } else {
                1.0
            }
        };
        let (sx, sy) = (ratio(width, old_w), ratio(height, old_h));
        for values in [
            &mut self.custom_points,
            &mut self.custom_points_in,
            &mut self.custom_points_out,
        ] {
            for (index, value) in values.iter_mut().enumerate() {
                *value *= if index % 2 == 0 { sx } else { sy };
            }
        }
        self.shape_size = vec![width, height];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contour_fields_are_arrays_and_absence_is_not_a_zero() {
        let absent: MaterialShape = serde_json::from_str("{}").unwrap();
        assert!(absent.roundness.is_empty());
        assert!(absent.shape_size.is_empty());
        assert_ne!(absent.roundness, vec![0.0]);

        let shape: MaterialShape = serde_json::from_str(
            r#"{"id":"sh","roundness":0.25,"custom_points":[0.0,1.0],"shape_size":[10.0,20.0],"shape_scale":1.0,"custom_points_in":[0.0],"custom_points_out":[1.0],"future_shape":true}"#,
        )
        .unwrap();
        assert_eq!(shape.roundness, vec![0.25]);
        assert_eq!(shape.custom_points, vec![0.0, 1.0]);
        assert_eq!(shape.shape_size, vec![10.0, 20.0]);
        assert_eq!(shape.shape_scale, vec![1.0]);
        let json = serde_json::to_value(&shape).unwrap();
        assert_eq!(json["roundness"], serde_json::json!([0.25]));
        assert_eq!(json["future_shape"], true);
    }

    #[test]
    fn a_preset_is_points_and_a_resize_stretches_them() {
        let mut square = MaterialShape::preset("sq".into(), ShapeKind::Rectangle, 320.0, 320.0);
        assert_eq!(square.shape_type, SHAPE_CONTOUR);
        assert_eq!(square.shape_kind(), Some(ShapeKind::Rectangle));
        assert_eq!(square.points()[0].at, [-160.0, 160.0]);
        assert_eq!(square.size(), (320.0, 320.0));
        assert_eq!(square.color, SHAPE_FILL);
        square.resize(640.0, 160.0);
        assert_eq!(square.points()[1].at, [320.0, 80.0]);
        assert_eq!(square.size(), (640.0, 160.0));

        let mut circle = MaterialShape::preset("c".into(), ShapeKind::Circle, 200.0, 200.0);
        assert_eq!(circle.points().len(), 4);
        let handle = circle.points()[0].handle_out[1];
        assert!((handle - 100.0 * CIRCLE_HANDLE).abs() < 1e-9);
        circle.resize(400.0, 200.0);
        assert!((circle.points()[1].handle_out[0] - 200.0 * CIRCLE_HANDLE).abs() < 1e-9);

        let line = MaterialShape::preset("l".into(), ShapeKind::Arrow, 320.0, 320.0);
        assert_eq!(line.shape_type, SHAPE_ARROW);
        assert!(line.color.is_empty());
        assert_eq!(line.border_width, SHAPE_LINE_WIDTH);
        assert_eq!(line.size(), (320.0, 0.0));
        let json = serde_json::to_value(&line).unwrap();
        let back: MaterialShape = serde_json::from_value(json).unwrap();
        assert_eq!(back, line);
    }
}
