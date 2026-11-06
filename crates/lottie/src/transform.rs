//! A layer's or a group's transform: anchor, position, scale, rotation,
//! skew and opacity.
//!
//! A point moves by the anchor's negative, then scale (percent), skew,
//! rotation (degrees, clockwise on screen), then position. Position may be
//! split into separate x and y properties. Opacity is a percentage.

use serde_json::Value;

use crate::LottieError;
use crate::geom::{Affine, Point};
use crate::limits::Limits;
use crate::prop::{Prop, number_prop};

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Position {
    Joined(Prop<Vec<f64>>),
    Split(Prop<Vec<f64>>, Prop<Vec<f64>>),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Transform {
    anchor: Prop<Vec<f64>>,
    position: Position,
    scale: Prop<Vec<f64>>,
    rotation: Prop<Vec<f64>>,
    opacity: Prop<Vec<f64>>,
    skew: Prop<Vec<f64>>,
    skew_axis: Prop<Vec<f64>>,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            anchor: Prop::Static(vec![0.0, 0.0]),
            position: Position::Joined(Prop::Static(vec![0.0, 0.0])),
            scale: Prop::Static(vec![100.0, 100.0]),
            rotation: Prop::constant(0.0),
            opacity: Prop::constant(100.0),
            skew: Prop::constant(0.0),
            skew_axis: Prop::constant(0.0),
        }
    }
}

impl Transform {
    pub(crate) fn parse(value: &Value, limits: &mut Limits) -> Result<Self, LottieError> {
        let mut transform = Self::default();
        let Value::Object(object) = value else {
            return Ok(transform);
        };
        let mut read = |key: &str, slot: &mut Prop<Vec<f64>>| -> Result<(), LottieError> {
            if let Some(prop) = number_prop(object.get(key), limits)? {
                *slot = prop;
            }
            Ok(())
        };
        read("a", &mut transform.anchor)?;
        read("s", &mut transform.scale)?;
        read("o", &mut transform.opacity)?;
        read("sk", &mut transform.skew)?;
        read("sa", &mut transform.skew_axis)?;
        if object.contains_key("r") {
            read("r", &mut transform.rotation)?;
        } else {
            read("rz", &mut transform.rotation)?;
        }
        match object.get("p") {
            Some(Value::Object(position))
                if position.get("s").and_then(Value::as_bool) == Some(true) =>
            {
                let x = number_prop(position.get("x"), limits)?.unwrap_or(Prop::constant(0.0));
                let y = number_prop(position.get("y"), limits)?.unwrap_or(Prop::constant(0.0));
                transform.position = Position::Split(x, y);
            }
            Some(value) => {
                if let Some(prop) = number_prop(Some(value), limits)? {
                    transform.position = Position::Joined(prop);
                }
            }
            None => {}
        }
        Ok(transform)
    }

    pub(crate) fn position(&self, time: f64) -> Point {
        match &self.position {
            Position::Joined(prop) => prop.point(time, Point::default()),
            Position::Split(x, y) => Point::new(x.scalar(time, 0.0), y.scalar(time, 0.0)),
        }
    }

    pub(crate) fn anchor(&self, time: f64) -> Point {
        self.anchor.point(time, Point::default())
    }

    pub(crate) fn scale(&self, time: f64) -> Point {
        self.scale.point(time, Point::new(100.0, 100.0)).scale(0.01)
    }

    pub(crate) fn rotation(&self, time: f64) -> f64 {
        self.rotation.scalar(time, 0.0)
    }

    /// Opacity from 0 to 1.
    pub(crate) fn opacity(&self, time: f64) -> f64 {
        (self.opacity.scalar(time, 100.0) / 100.0).clamp(0.0, 1.0)
    }

    pub(crate) fn matrix(&self, time: f64) -> Affine {
        let anchor = self.anchor(time);
        let position = self.position(time);
        let scale = self.scale(time);
        let skew = self.skew.scalar(time, 0.0);
        let axis = self.skew_axis.scalar(time, 0.0);
        let skewed = if skew == 0.0 {
            Affine::IDENTITY
        } else {
            let shear = (-skew).clamp(-89.0, 89.0).to_radians().tan();
            Affine::rotate(-axis)
                .then(Affine::shear_x(shear))
                .then(Affine::rotate(axis))
        };
        let matrix = Affine::translate(position.x, position.y)
            .then(Affine::rotate(self.rotation(time)))
            .then(skewed)
            .then(Affine::scale(scale.x, scale.y))
            .then(Affine::translate(-anchor.x, -anchor.y));
        if matrix.is_finite() {
            matrix
        } else {
            Affine::scale(0.0, 0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn transform(value: &Value) -> Transform {
        Transform::parse(value, &mut Limits::default()).unwrap()
    }

    #[test]
    fn anchor_scale_rotation_then_position() {
        let t = transform(&json!({
            "a": {"k": [10, 10]},
            "p": {"k": [100, 50]},
            "s": {"k": [200, 100]},
            "r": {"k": 90},
            "o": {"k": 50}
        }));
        let m = t.matrix(0.0);
        // The anchor lands on the position.
        let anchor = m.apply(Point::new(10.0, 10.0));
        assert!((anchor.x - 100.0).abs() < 1e-9 && (anchor.y - 50.0).abs() < 1e-9);
        // One unit right of the anchor: scaled to 2, then turned to point down.
        let right = m.apply(Point::new(11.0, 10.0));
        assert!((right.x - 100.0).abs() < 1e-9 && (right.y - 52.0).abs() < 1e-9);
        assert!((t.opacity(0.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_split_position_reads_x_and_y_apart() {
        let t = transform(&json!({
            "p": {"s": true, "x": {"a": 1, "k": [{"t": 0, "s": [0]}, {"t": 10, "s": [100]}]},
                  "y": {"k": 7}}
        }));
        let p = t.position(5.0);
        assert!((p.x - 50.0).abs() < 1e-9 && (p.y - 7.0).abs() < 1e-9);
    }

    #[test]
    fn skew_leans_along_its_axis() {
        let t = transform(&json!({"sk": {"k": 45}, "sa": {"k": 0}}));
        let m = t.matrix(0.0);
        let p = m.apply(Point::new(0.0, 10.0));
        assert!(
            (p.x.abs() - 10.0).abs() < 1e-9 && (p.y - 10.0).abs() < 1e-9,
            "{p:?}"
        );
        let flat = m.apply(Point::new(10.0, 0.0));
        assert!((flat.x - 10.0).abs() < 1e-9 && flat.y.abs() < 1e-9);
    }

    #[test]
    fn a_missing_transform_is_the_identity_and_rz_is_read() {
        let t = transform(&json!({}));
        assert_eq!(t.matrix(0.0), Affine::IDENTITY);
        assert_eq!(t.opacity(0.0), 1.0);
        let t = transform(&json!({"rz": {"k": 30}}));
        assert!((t.rotation(0.0) - 30.0).abs() < 1e-12);
    }
}
