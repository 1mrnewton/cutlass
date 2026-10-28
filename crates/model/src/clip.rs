use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A pair of floats. Scale, translation, and shadow offset all use this shape.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Point {
    #[serde(default)]
    pub x: f64,
    #[serde(default)]
    pub y: f64,
    /// Keys this model does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            x,
            y,
            rest: Map::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Flip {
    #[serde(default)]
    pub horizontal: bool,
    #[serde(default)]
    pub vertical: bool,
    /// Keys this model does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// The picture placed on the canvas. Identity is scale 1, no rotation, no move, opaque.
///
/// `bounding_box` and any other key this model does not name stay in `rest`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Clip {
    pub scale: Point,
    pub rotation: f64,
    pub transform: Point,
    pub flip: Flip,
    pub alpha: f64,
    /// Keys this model does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl Default for Clip {
    fn default() -> Self {
        Self {
            scale: Point::new(1.0, 1.0),
            rotation: 0.0,
            transform: Point::new(0.0, 0.0),
            flip: Flip::default(),
            alpha: 1.0,
            rest: Map::new(),
        }
    }
}

/// When `on` is set, one scale value drives both axes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UniformScale {
    pub on: bool,
    pub value: f64,
    /// Keys this model does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl Default for UniformScale {
    fn default() -> Self {
        Self {
            on: true,
            value: 1.0,
            rest: Map::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_null_clip_is_identity() {
        #[derive(Deserialize)]
        struct Wrap {
            #[serde(default, deserialize_with = "crate::serde_util::null_as_default")]
            clip: Clip,
        }
        let wrap: Wrap = serde_json::from_str(r#"{"clip":null}"#).unwrap();
        assert_eq!(wrap.clip, Clip::default());
        assert_eq!(wrap.clip.alpha, 1.0);
        assert_eq!(wrap.clip.scale, Point::new(1.0, 1.0));
    }

    #[test]
    fn a_clip_keeps_bounding_box() {
        let clip: Clip = serde_json::from_str(
            r#"{"scale":{"x":1.0,"y":2.0},"rotation":0.0,"transform":{"x":0.0,"y":0.0},"flip":{"horizontal":false,"vertical":false},"alpha":1.0,"bounding_box":{"left":1}}"#,
        )
        .unwrap();
        let json = serde_json::to_value(&clip).unwrap();
        assert_eq!(json["bounding_box"]["left"], 1);
        assert_eq!(json["scale"]["y"], 2.0);
    }
}
