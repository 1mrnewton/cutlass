//! A shape's settings that key on their own: its size, fill and stroke
//! opacity, stroke width, round corners and shadow. Each is keyed on the
//! segment under a Cutlass name and played onto a copy of the shape.

use crate::keyframe::{KeyframeGraph, sample};
use crate::materials::MaterialShape;
use crate::segment::Segment;
use crate::time::Micros;

/// The widest and tallest shape, in canvas pixels.
pub const MAX_SHAPE_SIZE: f64 = 8_192.0;
/// The widest stroke, in canvas pixels.
pub const MAX_SHAPE_STROKE: f64 = 256.0;
/// The farthest shadow, in canvas pixels.
pub const MAX_SHAPE_SHADOW_DISTANCE: f64 = 1_024.0;

/// The key names. Size keys its width and height, the first two.
pub static SHAPE_STYLE_KEYS: [&str; 9] = [
    "KFTypeShapeWidth",
    "KFTypeShapeHeight",
    "KFTypeShapeFillAlpha",
    "KFTypeShapeStrokeWidth",
    "KFTypeShapeStrokeAlpha",
    "KFTypeShapeCorners",
    "KFTypeShapeShadowDistance",
    "KFTypeShapeShadowAngle",
    "KFTypeShapeShadowAlpha",
];

/// One keyed shape setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShapeStyleKey {
    /// Width and height, in canvas pixels.
    Size,
    FillOpacity,
    StrokeWidth,
    StrokeOpacity,
    /// One radius for every corner.
    Corners,
    ShadowDistance,
    ShadowAngle,
    ShadowOpacity,
}

impl ShapeStyleKey {
    /// Every setting.
    pub const ALL: [Self; 8] = [
        Self::Size,
        Self::FillOpacity,
        Self::StrokeWidth,
        Self::StrokeOpacity,
        Self::Corners,
        Self::ShadowDistance,
        Self::ShadowAngle,
        Self::ShadowOpacity,
    ];

    /// The key names the setting is stored under: width then height for
    /// Size, one name for the rest.
    #[must_use]
    pub fn names(self) -> &'static [&'static str] {
        match self {
            Self::Size => &SHAPE_STYLE_KEYS[0..2],
            Self::FillOpacity => &SHAPE_STYLE_KEYS[2..3],
            Self::StrokeWidth => &SHAPE_STYLE_KEYS[3..4],
            Self::StrokeOpacity => &SHAPE_STYLE_KEYS[4..5],
            Self::Corners => &SHAPE_STYLE_KEYS[5..6],
            Self::ShadowDistance => &SHAPE_STYLE_KEYS[6..7],
            Self::ShadowAngle => &SHAPE_STYLE_KEYS[7..8],
            Self::ShadowOpacity => &SHAPE_STYLE_KEYS[8..9],
        }
    }

    /// The setting a key name stores, when it is one.
    #[must_use]
    pub fn of(property: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|key| key.names().contains(&property))
    }

    /// Whether `value` is one the setting's edit takes.
    #[must_use]
    pub fn takes(self, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        match self {
            // A line keys a height of 0.
            Self::Size => (0.0..=MAX_SHAPE_SIZE).contains(&value),
            Self::FillOpacity | Self::StrokeOpacity | Self::ShadowOpacity => {
                (0.0..=1.0).contains(&value)
            }
            Self::StrokeWidth => (0.0..=MAX_SHAPE_STROKE).contains(&value),
            Self::Corners => (0.0..=MAX_SHAPE_SIZE).contains(&value),
            Self::ShadowDistance => (0.0..=MAX_SHAPE_SHADOW_DISTANCE).contains(&value),
            Self::ShadowAngle => true,
        }
    }

    /// The setting's values on `shape`, one for each of its names.
    #[must_use]
    pub fn get(self, shape: &MaterialShape) -> Vec<f64> {
        match self {
            Self::Size => {
                let (width, height) = shape.size();
                vec![width, height]
            }
            Self::FillOpacity => vec![shape.global_alpha],
            Self::StrokeWidth => vec![shape.border_width],
            Self::StrokeOpacity => vec![shape.border_alpha],
            Self::Corners => vec![shape.roundness.first().copied().unwrap_or_default()],
            Self::ShadowDistance => vec![shape.shadow_distance],
            Self::ShadowAngle => vec![shape.shadow_angle],
            Self::ShadowOpacity => vec![shape.shadow_alpha],
        }
    }
}

/// `shape` with each setting `segment` keys set as keyed at timeline
/// `time`; `None` when it keys none. A value past its range is held to it,
/// and a corner rounds no further than half the shorter side.
#[must_use]
pub fn shape_style_at(
    shape: &MaterialShape,
    segment: &Segment,
    graphs: &[KeyframeGraph],
    time: Micros,
) -> Option<MaterialShape> {
    if !segment
        .common_keyframes
        .iter()
        .any(|group| SHAPE_STYLE_KEYS.contains(&group.property_type.as_str()))
    {
        return None;
    }
    let offset = time.0.saturating_sub(segment.target.start.0);
    let keyed = |name: &str| match sample(&segment.common_keyframes, graphs, &[name], offset) {
        Some(Some(value)) if value.is_finite() => Some(value),
        _ => None,
    };
    let unit = |value: f64| value.clamp(0.0, 1.0);
    let mut out = shape.clone();
    let (width, height) = shape.size();
    let (new_w, new_h) = (keyed(SHAPE_STYLE_KEYS[0]), keyed(SHAPE_STYLE_KEYS[1]));
    if new_w.is_some() || new_h.is_some() {
        let width = new_w.unwrap_or(width).clamp(1.0, MAX_SHAPE_SIZE);
        let height = if shape.is_straight() {
            0.0
        } else {
            new_h.unwrap_or(height).clamp(1.0, MAX_SHAPE_SIZE)
        };
        out.resize(width, height);
    }
    if let Some(value) = keyed(SHAPE_STYLE_KEYS[2]) {
        out.global_alpha = unit(value);
    }
    if let Some(value) = keyed(SHAPE_STYLE_KEYS[3]) {
        out.border_width = value.clamp(0.0, MAX_SHAPE_STROKE);
    }
    if let Some(value) = keyed(SHAPE_STYLE_KEYS[4]) {
        out.border_alpha = unit(value);
    }
    let (width, height) = out.size();
    let most = (width.min(height) / 2.0).max(0.0);
    if let Some(value) = keyed(SHAPE_STYLE_KEYS[5]) {
        let corners = out.custom_points.len() / 2;
        out.roundness = vec![value.clamp(0.0, most); corners];
    }
    if let Some(value) = keyed(SHAPE_STYLE_KEYS[6]) {
        out.shadow_distance = value.clamp(0.0, MAX_SHAPE_SHADOW_DISTANCE);
    }
    if let Some(value) = keyed(SHAPE_STYLE_KEYS[7]) {
        out.shadow_angle = value;
    }
    if let Some(value) = keyed(SHAPE_STYLE_KEYS[8]) {
        out.shadow_alpha = unit(value);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keyframe::{is_keyable, put_key};
    use crate::time::TimeRange;

    fn square() -> MaterialShape {
        let mut shape = MaterialShape {
            global_alpha: 1.0,
            ..MaterialShape::default()
        };
        shape.custom_points = vec![-50.0, -50.0, 50.0, -50.0, 50.0, 50.0, -50.0, 50.0];
        shape.shape_size = vec![100.0, 100.0];
        shape.roundness = vec![0.0; 4];
        shape
    }

    #[test]
    fn a_keyed_shape_setting_moves_along_the_clip_and_leaves_the_rest() {
        let shape = square();
        let mut segment = Segment::new("s", "m", TimeRange::new(1_000, 2_000));
        assert_eq!(shape_style_at(&shape, &segment, &[], Micros(1_000)), None);
        let fill = ShapeStyleKey::FillOpacity.names()[0];
        put_key(&mut segment, fill, Micros(0), 0.0);
        put_key(&mut segment, fill, Micros(2_000), 1.0);
        let keyed = shape_style_at(&shape, &segment, &[], Micros(2_000)).unwrap();
        assert!((keyed.global_alpha - 0.5).abs() < 1e-9);
        assert_eq!(keyed.size(), shape.size());
        assert_eq!(keyed.border_width, shape.border_width);
        // A width key resizes; corners round no further than half a side.
        put_key(
            &mut segment,
            ShapeStyleKey::Size.names()[0],
            Micros(0),
            40.0,
        );
        put_key(
            &mut segment,
            ShapeStyleKey::Corners.names()[0],
            Micros(0),
            500.0,
        );
        let keyed = shape_style_at(&shape, &segment, &[], Micros(1_000)).unwrap();
        assert_eq!(keyed.size(), (40.0, 100.0));
        assert_eq!(keyed.roundness, vec![20.0; 4]);
    }

    #[test]
    fn every_setting_has_its_own_names_and_range() {
        let mut seen = Vec::new();
        for key in ShapeStyleKey::ALL {
            for name in key.names() {
                assert_eq!(ShapeStyleKey::of(name), Some(key));
                assert!(is_keyable(name), "{name}");
                seen.push(*name);
            }
            assert!(!key.takes(f64::NAN));
            assert_eq!(key.get(&square()).len(), key.names().len());
        }
        assert_eq!(seen, SHAPE_STYLE_KEYS);
        assert!(!ShapeStyleKey::FillOpacity.takes(1.5));
        assert!(!ShapeStyleKey::StrokeWidth.takes(-1.0));
        assert!(ShapeStyleKey::ShadowAngle.takes(-720.0));
    }
}
