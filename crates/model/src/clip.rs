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

    /// Both zero and nothing else stored: a pivot at the middle, which is
    /// left out of the stored clip.
    #[must_use]
    pub fn is_origin(&self) -> bool {
        self.x == 0.0 && self.y == 0.0 && self.rest.is_empty()
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
    /// The point of the picture that `rotation` turns it about and `scale`
    /// sizes it about, as a share of its own half size from its middle with
    /// y up: (0, 0) is the middle, (-1, -1) the lower left corner, and a
    /// point past 1 lies outside the picture. `transform` still says where
    /// the picture's middle sits when it is neither turned nor scaled, so
    /// moving the pivot does not move such a picture. A pivot at the middle
    /// is not stored.
    #[serde(default, skip_serializing_if = "Point::is_origin")]
    pub pivot: Point,
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
            pivot: Point::default(),
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

    #[test]
    fn a_pivot_is_stored_only_when_it_is_off_the_middle() {
        let stored = r#"{"scale":{"x":1.0,"y":1.0},"rotation":0.0,"transform":{"x":0.0,"y":0.0},"flip":{"horizontal":false,"vertical":false},"alpha":1.0}"#;
        // A clip written before there was a pivot turns about its middle,
        // and is written back with no pivot.
        let clip: Clip = serde_json::from_str(stored).unwrap();
        assert!(clip.pivot.is_origin());
        let json = serde_json::to_value(&clip).unwrap();
        assert!(json.get("pivot").is_none());
        assert_eq!(json, serde_json::from_str::<Value>(stored).unwrap());
        // A pivot off the middle is kept through the store.
        let mut hinged = clip;
        hinged.pivot = Point::new(-1.0, 0.5);
        let json = serde_json::to_value(&hinged).unwrap();
        assert_eq!(json["pivot"]["x"], -1.0);
        assert_eq!(json["pivot"]["y"], 0.5);
        let back: Clip = serde_json::from_value(json).unwrap();
        assert_eq!(back, hinged);
        assert!(!back.pivot.is_origin());
    }
}
