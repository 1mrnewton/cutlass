//! Masks: a shape that keeps part of a clip and lets the rest show what is
//! below.
//!
//! A clip's mask is a `mask` record in the draft's `common_mask` pool that
//! the clip lists among its extra materials. Its numbers are in the clip's
//! own frame, so the mask moves, turns and scales with the clip: `x` and
//! `y` from -1 to 1 across the clip (y up), `width` and `height` as parts
//! of the clip's size, `rotation` in degrees, `feather` from 0 to 1,
//! `round` (a rectangle's corners) from 0 to 1, and `invert`.

use cutlass_model::{Draft, MaterialRef, Segment};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

/// The material `type` of a mask.
pub const MASK: &str = "mask";

/// A mask's shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskShape {
    /// Keeps one side of a line.
    Linear,
    /// Keeps a band between two lines.
    Mirror,
    Circle,
    Rectangle,
    Heart,
    Star,
}

impl MaskShape {
    pub const ALL: [Self; 6] = [
        Self::Linear,
        Self::Mirror,
        Self::Circle,
        Self::Rectangle,
        Self::Heart,
        Self::Star,
    ];

    #[must_use]
    pub fn key(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Mirror => "mirror",
            Self::Circle => "circle",
            Self::Rectangle => "rectangle",
            Self::Heart => "heart",
            Self::Star => "star",
        }
    }

    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::Mirror => "Mirror",
            Self::Circle => "Circle",
            Self::Rectangle => "Rectangle",
            Self::Heart => "Heart",
            Self::Star => "Star",
        }
    }

    #[must_use]
    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|shape| shape.key() == key)
    }
}

/// One mask, in the clip's own frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mask {
    pub shape: MaskShape,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub rotation: f32,
    pub feather: f32,
    pub round: f32,
    pub invert: bool,
}

impl Mask {
    /// A new mask of `shape`, centred, half the clip's size.
    #[must_use]
    pub fn new(shape: MaskShape) -> Self {
        Self {
            shape,
            x: 0.0,
            y: 0.0,
            width: 0.5,
            height: 0.5,
            rotation: 0.0,
            feather: 0.0,
            round: 0.0,
            invert: false,
        }
    }

    /// Every number finite and in its range.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let numbers = [
            self.x,
            self.y,
            self.width,
            self.height,
            self.rotation,
            self.feather,
            self.round,
        ];
        numbers.iter().all(|value| value.is_finite())
            && (-2.0..=2.0).contains(&self.x)
            && (-2.0..=2.0).contains(&self.y)
            && (0.0..=4.0).contains(&self.width)
            && (0.0..=4.0).contains(&self.height)
            && (-360.0..=360.0).contains(&self.rotation)
            && (0.0..=1.0).contains(&self.feather)
            && (0.0..=1.0).contains(&self.round)
    }

    /// How much of the clip the mask keeps at clip point `u`, `v` (-1 to 1
    /// across the clip, y up), from 0 to 1.
    #[must_use]
    pub fn coverage(&self, u: f32, v: f32) -> f32 {
        let (sin, cos) = self.rotation.to_radians().sin_cos();
        let (du, dv) = (u - self.x, v - self.y);
        // Into the mask's own turned frame.
        let (a, b) = (cos * du + sin * dv, -sin * du + cos * dv);
        let soft = self.feather.max(0.0) * 0.5 + 1e-4;
        let half_w = self.width.max(1e-4);
        let half_h = self.height.max(1e-4);
        let keep = match self.shape {
            MaskShape::Linear => smoothstep(soft, -soft, b),
            MaskShape::Mirror => smoothstep(half_h + soft, half_h - soft, b.abs()),
            MaskShape::Circle => {
                let r = ((a / half_w).powi(2) + (b / half_h).powi(2)).sqrt();
                edge(r, soft / half_w.min(half_h))
            }
            MaskShape::Rectangle => {
                let corner = self.round.clamp(0.0, 1.0) * half_w.min(half_h);
                let qx = a.abs() - (half_w - corner);
                let qy = b.abs() - (half_h - corner);
                let outside = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt();
                let distance = outside + qx.max(qy).min(0.0) - corner;
                smoothstep(soft, -soft, distance)
            }
            MaskShape::Heart => {
                let (x, y) = (a / half_w * 1.2, b / half_h * 1.2 + 0.15);
                let value = (x * x + y * y - 1.0).powi(3) - x * x * y * y * y;
                smoothstep(soft * 2.0, -soft * 2.0, value)
            }
            MaskShape::Star => {
                let (x, y) = (a / half_w, b / half_h);
                let radius = (x * x + y * y).sqrt();
                let angle = y.atan2(x) - std::f32::consts::FRAC_PI_2;
                let sector = std::f32::consts::TAU / 5.0;
                let into = (angle.rem_euclid(sector) / sector - 0.5).abs() * 2.0;
                let reach = 0.45 + 0.55 * into;
                edge(radius / reach, soft / half_w.min(half_h))
            }
        };
        if self.invert { 1.0 - keep } else { keep }
    }

    /// The mask as the record's fields.
    #[must_use]
    pub fn to_fields(&self) -> Map<String, Value> {
        let mut fields = Map::new();
        fields.insert("shape".to_owned(), json!(self.shape.key()));
        for (name, value) in [
            ("x", self.x),
            ("y", self.y),
            ("width", self.width),
            ("height", self.height),
            ("rotation", self.rotation),
            ("feather", self.feather),
            ("round", self.round),
        ] {
            fields.insert(name.to_owned(), json!(value));
        }
        fields.insert("invert".to_owned(), json!(self.invert));
        fields
    }

    /// A mask read from a record's fields, when its shape is known and its
    /// numbers are whole.
    #[must_use]
    pub fn from_fields(fields: &Map<String, Value>) -> Option<Self> {
        let shape = MaskShape::from_key(fields.get("shape")?.as_str()?)?;
        let defaults = Self::new(shape);
        #[allow(clippy::cast_possible_truncation)]
        let number = |name: &str, default: f32| {
            fields
                .get(name)
                .and_then(Value::as_f64)
                .map_or(default, |value| value as f32)
        };
        let mask = Self {
            shape,
            x: number("x", defaults.x),
            y: number("y", defaults.y),
            width: number("width", defaults.width),
            height: number("height", defaults.height),
            rotation: number("rotation", defaults.rotation),
            feather: number("feather", defaults.feather),
            round: number("round", defaults.round),
            invert: fields
                .get("invert")
                .and_then(Value::as_bool)
                .unwrap_or_default(),
        };
        mask.is_valid().then_some(mask)
    }

    /// Multiply premultiplied RGBA by the mask. `to_clip` turns a canvas
    /// pixel centre into the clip point it shows.
    pub fn apply(
        &self,
        rgba: &mut [u8],
        width: u32,
        to_clip: impl Fn(f32, f32) -> Option<(f32, f32)>,
    ) {
        let width = width as usize;
        if width == 0 {
            return;
        }
        for (index, pixel) in rgba.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            if pixel[3] == 0 {
                continue;
            }
            #[allow(clippy::cast_precision_loss)]
            let (x, y) = ((index % width) as f32 + 0.5, (index / width) as f32 + 0.5);
            let keep = to_clip(x, y).map_or(0.0, |(u, v)| self.coverage(u, v));
            if keep >= 1.0 {
                continue;
            }
            for channel in pixel.iter_mut() {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let value = (f32::from(*channel) * keep.max(0.0)).round() as u8;
                *channel = value;
            }
        }
    }
}

/// The mask `segment` lists, when it has one.
#[must_use]
pub fn segment_mask(draft: &Draft, segment: &Segment) -> Option<Mask> {
    segment.extra_material_refs.iter().find_map(|id| {
        let Some(MaterialRef::Other(raw)) = draft.materials.get(id.as_str()) else {
            return None;
        };
        (raw.kind == MASK).then(|| Mask::from_fields(&raw.fields))?
    })
}

/// 1 inside a shape whose edge is at `value` 1, soft for `soft` past it.
fn edge(value: f32, soft: f32) -> f32 {
    smoothstep(1.0 + soft, 1.0 - soft, value)
}

fn smoothstep(low: f32, high: f32, value: f32) -> f32 {
    let t = ((value - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, MaterialId, RawMaterial, Segment, TimeRange};

    use super::{MASK, Mask, MaskShape, segment_mask};

    #[test]
    fn each_shape_keeps_its_middle_and_drops_its_corner() {
        for shape in MaskShape::ALL {
            let mask = Mask::new(shape);
            let middle = if shape == MaskShape::Linear {
                mask.coverage(0.0, -0.5)
            } else {
                mask.coverage(0.0, 0.0)
            };
            assert!(middle > 0.99, "{shape:?} {middle}");
            assert!(mask.coverage(0.95, 0.95) < 0.01, "{shape:?}");
            let inverted = Mask {
                invert: true,
                ..mask
            };
            assert!(inverted.coverage(0.95, 0.95) > 0.99, "{shape:?}");
        }
    }

    #[test]
    fn the_mask_moves_turns_and_feathers() {
        let circle = Mask {
            x: 0.5,
            width: 0.2,
            height: 0.2,
            ..Mask::new(MaskShape::Circle)
        };
        assert!(circle.coverage(0.5, 0.0) > 0.99);
        assert!(circle.coverage(0.0, 0.0) < 0.01);
        let line = Mask {
            rotation: 90.0,
            ..Mask::new(MaskShape::Linear)
        };
        // Turned a quarter, the line keeps the right side.
        assert!(
            line.coverage(0.5, 0.0) > 0.99,
            "{}",
            line.coverage(0.5, 0.0)
        );
        assert!(line.coverage(-0.5, 0.0) < 0.01);
        let soft = Mask {
            feather: 1.0,
            ..Mask::new(MaskShape::Rectangle)
        };
        let half = soft.coverage(0.5, 0.0);
        assert!(half > 0.3 && half < 0.7, "{half}");
        let star = Mask::new(MaskShape::Star);
        assert!(star.coverage(0.0, 0.45) > 0.99, "a point is kept");
        assert!(
            star.coverage(0.0, -0.45) < 0.01,
            "between two points is not"
        );
    }

    #[test]
    fn apply_multiplies_and_the_draft_round_trips() {
        let mask = Mask::new(MaskShape::Rectangle);
        let mut rgba = [200, 100, 50, 255].repeat(4);
        mask.apply(&mut rgba, 2, |x, _| {
            Some((if x < 1.0 { 0.0 } else { 0.9 }, 0.0))
        });
        assert_eq!(&rgba[..4], &[200, 100, 50, 255]);
        assert_eq!(&rgba[4..8], &[0, 0, 0, 0]);

        let mut draft = Draft::empty("D");
        let stored = Mask {
            feather: 0.3,
            invert: true,
            ..Mask::new(MaskShape::Heart)
        };
        draft.materials.common_mask.push(RawMaterial {
            id: MaterialId::new("m"),
            kind: MASK.to_owned(),
            fields: stored.to_fields(),
        });
        let mut segment = Segment::new("s", "x", TimeRange::new(0, 1));
        segment.extra_material_refs.push(MaterialId::new("m"));
        assert_eq!(segment_mask(&draft, &segment), Some(stored));
        assert!(
            !Mask {
                feather: 3.0,
                ..stored
            }
            .is_valid()
        );
    }
}
