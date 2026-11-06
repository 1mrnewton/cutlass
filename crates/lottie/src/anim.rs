//! The animation file: its size, frame rate, in and out points, layers and
//! precompositions.

use serde_json::Value;

use crate::LottieError;
use crate::limits::{Limits, MAX_BYTES, MAX_EDGE};
use crate::prop::{Prop, number_prop, shape_prop};
use crate::shape::{Item, PathShape, parse_items};
use crate::transform::Transform;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum LayerKind {
    /// A precomposition's layers, by index into the animation's comps.
    Precomp {
        comp: usize,
        width: f64,
        height: f64,
        time_remap: Option<Prop<Vec<f64>>>,
    },
    Solid {
        color: [f64; 3],
        width: f64,
        height: f64,
    },
    Shapes(Vec<Item>),
    /// Parents others and draws nothing; also every layer kind not drawn
    /// here, such as images and text.
    Null,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MaskKind {
    Add,
    Subtract,
    Intersect,
    Lighten,
    Darken,
    Difference,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MaskSpec {
    pub(crate) kind: MaskKind,
    pub(crate) inverted: bool,
    pub(crate) shape: Prop<PathShape>,
    pub(crate) opacity: Prop<Vec<f64>>,
    pub(crate) expansion: Prop<Vec<f64>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MatteKind {
    Alpha,
    AlphaInverted,
    Luma,
    LumaInverted,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Layer {
    pub(crate) kind: LayerKind,
    pub(crate) index: Option<i64>,
    pub(crate) parent: Option<i64>,
    pub(crate) transform: Transform,
    pub(crate) in_point: f64,
    pub(crate) out_point: f64,
    pub(crate) start: f64,
    pub(crate) stretch: f64,
    pub(crate) masks: Vec<MaskSpec>,
    pub(crate) matte: Option<MatteKind>,
    /// The layer whose index the matte reads; otherwise the one above.
    pub(crate) matte_parent: Option<i64>,
    /// A matte source, not drawn by itself.
    pub(crate) is_matte: bool,
    pub(crate) hidden: bool,
}

/// A parsed Lottie animation.
#[derive(Clone, Debug, PartialEq)]
pub struct Animation {
    pub(crate) width: f64,
    pub(crate) height: f64,
    pub(crate) frame_rate: f64,
    pub(crate) in_point: f64,
    pub(crate) out_point: f64,
    pub(crate) layers: Vec<Layer>,
    pub(crate) comps: Vec<Vec<Layer>>,
}

fn number(object: &serde_json::Map<String, Value>, key: &str) -> Option<f64> {
    object
        .get(key)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
}

impl Animation {
    /// Parse a Lottie file.
    ///
    /// The size, frame rate and in and out points must be present and sane.
    /// A layer or shape this crate does not draw is kept as nothing drawn.
    pub fn parse(bytes: &[u8]) -> Result<Self, LottieError> {
        if bytes.len() > MAX_BYTES {
            return Err(LottieError::Limit("file size"));
        }
        let root: Value =
            serde_json::from_slice(bytes).map_err(|error| LottieError::Json(error.to_string()))?;
        let Value::Object(object) = &root else {
            return Err(LottieError::Invalid("the file is not an object"));
        };
        let width = number(object, "w").ok_or(LottieError::Invalid("no width"))?;
        let height = number(object, "h").ok_or(LottieError::Invalid("no height"))?;
        if !(width > 0.0 && height > 0.0 && width <= MAX_EDGE && height <= MAX_EDGE) {
            return Err(LottieError::Invalid("the size is out of range"));
        }
        let frame_rate = number(object, "fr").ok_or(LottieError::Invalid("no frame rate"))?;
        if !(frame_rate > 0.0 && frame_rate <= 1_000.0) {
            return Err(LottieError::Invalid("the frame rate is out of range"));
        }
        let in_point = number(object, "ip").unwrap_or(0.0);
        let out_point = number(object, "op").ok_or(LottieError::Invalid("no out point"))?;
        if out_point <= in_point {
            return Err(LottieError::Invalid(
                "the out point is not after the in point",
            ));
        }
        let mut limits = Limits::default();
        // Precompositions first, so a layer can find its comp by id.
        let mut ids: Vec<String> = Vec::new();
        let mut raw: Vec<&Value> = Vec::new();
        if let Some(Value::Array(assets)) = object.get("assets") {
            for asset in assets {
                let (Some(id), Some(layers)) =
                    (asset.get("id").and_then(Value::as_str), asset.get("layers"))
                else {
                    continue;
                };
                ids.push(id.to_owned());
                raw.push(layers);
            }
        }
        let mut comps = Vec::with_capacity(raw.len());
        for layers in raw {
            comps.push(parse_layers(layers, &ids, &mut limits)?);
        }
        let layers = parse_layers(
            object.get("layers").unwrap_or(&Value::Null),
            &ids,
            &mut limits,
        )?;
        Ok(Self {
            width,
            height,
            frame_rate,
            in_point,
            out_point,
            layers,
            comps,
        })
    }

    /// The width the file draws at, in its own pixels.
    #[must_use]
    pub fn width(&self) -> f64 {
        self.width
    }

    #[must_use]
    pub fn height(&self) -> f64 {
        self.height
    }

    /// Frames per second.
    #[must_use]
    pub fn frame_rate(&self) -> f64 {
        self.frame_rate
    }

    /// The first frame.
    #[must_use]
    pub fn in_point(&self) -> f64 {
        self.in_point
    }

    /// The frame after the last.
    #[must_use]
    pub fn out_point(&self) -> f64 {
        self.out_point
    }

    /// One play-through, in seconds.
    #[must_use]
    pub fn duration(&self) -> f64 {
        (self.out_point - self.in_point) / self.frame_rate
    }

    /// The frame shown `seconds` into one play-through, which repeats.
    #[must_use]
    pub fn frame_at(&self, seconds: f64) -> f64 {
        let duration = self.duration();
        let within = if seconds.is_finite() {
            seconds.rem_euclid(duration)
        } else {
            0.0
        };
        self.in_point + within * self.frame_rate
    }
}

fn parse_layers(
    value: &Value,
    comp_ids: &[String],
    limits: &mut Limits,
) -> Result<Vec<Layer>, LottieError> {
    let Value::Array(list) = value else {
        return Ok(Vec::new());
    };
    limits.layers(list.len())?;
    let mut layers = Vec::with_capacity(list.len());
    for entry in list {
        let Value::Object(object) = entry else {
            continue;
        };
        let kind = match object.get("ty").and_then(Value::as_f64).map(|ty| ty as i64) {
            Some(0) => {
                let comp = object
                    .get("refId")
                    .and_then(Value::as_str)
                    .and_then(|id| comp_ids.iter().position(|known| known == id));
                match comp {
                    Some(comp) => LayerKind::Precomp {
                        comp,
                        width: number(object, "w").unwrap_or(0.0),
                        height: number(object, "h").unwrap_or(0.0),
                        time_remap: number_prop(object.get("tm"), limits)?,
                    },
                    None => LayerKind::Null,
                }
            }
            Some(1) => LayerKind::Solid {
                color: object
                    .get("sc")
                    .and_then(Value::as_str)
                    .and_then(hex_color)
                    .unwrap_or([0.0; 3]),
                width: number(object, "sw").unwrap_or(0.0),
                height: number(object, "sh").unwrap_or(0.0),
            },
            Some(4) => LayerKind::Shapes(parse_items(object.get("shapes"), 0, limits)?.0),
            _ => LayerKind::Null,
        };
        let mut masks = Vec::new();
        if let Some(Value::Array(list)) = object.get("masksProperties") {
            limits.shapes(list.len())?;
            for mask in list {
                let kind = match mask.get("mode").and_then(Value::as_str) {
                    Some("a") => MaskKind::Add,
                    Some("s") => MaskKind::Subtract,
                    Some("i") => MaskKind::Intersect,
                    Some("l") => MaskKind::Lighten,
                    Some("d") => MaskKind::Darken,
                    Some("f") => MaskKind::Difference,
                    _ => continue,
                };
                let Some(shape) = shape_prop(mask.get("pt"), limits)? else {
                    continue;
                };
                masks.push(MaskSpec {
                    kind,
                    inverted: mask.get("inv").and_then(Value::as_bool).unwrap_or(false),
                    shape,
                    opacity: number_prop(mask.get("o"), limits)?.unwrap_or(Prop::constant(100.0)),
                    expansion: number_prop(mask.get("x"), limits)?.unwrap_or(Prop::constant(0.0)),
                });
            }
        }
        let matte = match object.get("tt").and_then(Value::as_f64).map(|tt| tt as i64) {
            Some(1) => Some(MatteKind::Alpha),
            Some(2) => Some(MatteKind::AlphaInverted),
            Some(3) => Some(MatteKind::Luma),
            Some(4) => Some(MatteKind::LumaInverted),
            _ => None,
        };
        let flag = |key: &str| match object.get(key) {
            Some(Value::Bool(flag)) => *flag,
            Some(value) => value.as_f64().is_some_and(|flag| flag != 0.0),
            None => false,
        };
        let stretch = number(object, "sr").filter(|sr| *sr != 0.0).unwrap_or(1.0);
        layers.push(Layer {
            kind,
            index: number(object, "ind").map(|ind| ind as i64),
            parent: number(object, "parent").map(|parent| parent as i64),
            transform: match object.get("ks") {
                Some(value) => Transform::parse(value, limits)?,
                None => Transform::default(),
            },
            in_point: number(object, "ip").unwrap_or(f64::MIN),
            out_point: number(object, "op").unwrap_or(f64::MAX),
            start: number(object, "st").unwrap_or(0.0),
            stretch,
            masks,
            matte,
            matte_parent: number(object, "tp").map(|tp| tp as i64),
            is_matte: flag("td"),
            hidden: flag("hd"),
        });
    }
    Ok(layers)
}

/// `#rrggbb` as three numbers from 0 to 1.
fn hex_color(text: &str) -> Option<[f64; 3]> {
    let hex = text.trim().strip_prefix('#').unwrap_or(text.trim());
    if hex.len() != 6 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let channel = |at: usize| {
        u8::from_str_radix(hex.get(at..at + 2)?, 16)
            .ok()
            .map(|value| f64::from(value) / 255.0)
    };
    Some([channel(0)?, channel(2)?, channel(4)?])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hex_color_is_six_digits() {
        assert_eq!(hex_color("#ff0000"), Some([1.0, 0.0, 0.0]));
        assert_eq!(hex_color("00ff00"), Some([0.0, 1.0, 0.0]));
        assert_eq!(hex_color("#fff"), None);
        assert_eq!(hex_color("#gg0000"), None);
        assert_eq!(hex_color("#ffé000"), None);
    }

    #[test]
    fn the_frame_at_a_time_wraps_round_one_play() {
        let animation =
            Animation::parse(br#"{"w": 10, "h": 10, "fr": 25, "ip": 5, "op": 30, "layers": []}"#)
                .unwrap();
        assert!((animation.duration() - 1.0).abs() < 1e-12);
        assert!((animation.frame_at(0.0) - 5.0).abs() < 1e-12);
        assert!((animation.frame_at(0.5) - 17.5).abs() < 1e-9);
        assert!((animation.frame_at(1.5) - 17.5).abs() < 1e-9);
        assert!((animation.frame_at(-0.5) - 17.5).abs() < 1e-9);
        assert!((animation.frame_at(f64::NAN) - 5.0).abs() < 1e-12);
    }
}
