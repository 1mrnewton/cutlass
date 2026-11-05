//! Colour grading on a clip: curves, colour wheels and HSL.
//!
//! They run before the basic adjust sliders, in the reference editor's
//! order: curves, then wheels, then HSL. Each is a material the clip lists
//! among its extra materials:
//!
//! - HSL is one `hsl` record per colour, with `hsl_color_type` 0 to 7 (red,
//!   orange, yellow, green, cyan, blue, purple, magenta) and `hue`,
//!   `saturation` and `lightness` from -100 to 100.
//! - Curves are one `color_curves` record with `rgb`, `red`, `green` and
//!   `blue`, each a list of `[x, y]` points from 0 to 1. A missing or empty
//!   list is the straight line.
//! - Wheels are one `primary_color_wheels` record with `lift`, `gamma` and
//!   `gain`, each `[x, y, luma]`: where the puck sits in the unit circle and
//!   how far the ring below it is turned, from -1 to 1.

use cutlass_model::{Draft, MaterialRef, Segment};
use serde_json::{Value, json};

use crate::adjust::{Adjust, adjust_rgba, segment_adjust};

/// The material `type` of a clip's curves.
pub const COLOR_CURVES: &str = "color_curves";
/// The material `type` of a clip's colour wheels.
pub const COLOR_WHEELS: &str = "primary_color_wheels";
/// The material `type` of one HSL colour.
pub const HSL: &str = "hsl";

/// The eight colours HSL works on, by `hsl_color_type`.
pub const HSL_COLORS: [&str; 8] = [
    "Red", "Orange", "Yellow", "Green", "Cyan", "Blue", "Purple", "Magenta",
];
/// Where each HSL colour sits on the hue circle, in degrees.
const HSL_CENTRES: [f32; 8] = [0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 275.0, 310.0];
/// The most an HSL hue slider turns its colour, in degrees.
const HSL_TURN: f32 = 30.0;
/// The most points one curve keeps.
pub const MAX_CURVE_POINTS: usize = 16;

/// A curve: all four, or one colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CurveChannel {
    Rgb,
    Red,
    Green,
    Blue,
}

impl CurveChannel {
    pub const ALL: [Self; 4] = [Self::Rgb, Self::Red, Self::Green, Self::Blue];

    /// The field the points are stored under.
    #[must_use]
    pub fn key(self) -> &'static str {
        match self {
            Self::Rgb => "rgb",
            Self::Red => "red",
            Self::Green => "green",
            Self::Blue => "blue",
        }
    }

    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Rgb => "RGB",
            Self::Red => "Red",
            Self::Green => "Green",
            Self::Blue => "Blue",
        }
    }

    #[must_use]
    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|channel| channel.key() == key)
    }

    fn index(self) -> usize {
        match self {
            Self::Rgb => 0,
            Self::Red => 1,
            Self::Green => 2,
            Self::Blue => 3,
        }
    }
}

/// One colour wheel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Wheel {
    /// The shadows.
    Lift,
    /// The midtones.
    Gamma,
    /// The highlights.
    Gain,
}

impl Wheel {
    pub const ALL: [Self; 3] = [Self::Lift, Self::Gamma, Self::Gain];

    #[must_use]
    pub fn key(self) -> &'static str {
        match self {
            Self::Lift => "lift",
            Self::Gamma => "gamma",
            Self::Gain => "gain",
        }
    }

    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Lift => "Lift",
            Self::Gamma => "Gamma",
            Self::Gain => "Gain",
        }
    }

    #[must_use]
    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|wheel| wheel.key() == key)
    }

    fn index(self) -> usize {
        match self {
            Self::Lift => 0,
            Self::Gamma => 1,
            Self::Gain => 2,
        }
    }
}

/// A wheel's setting: the puck in the unit circle and the ring.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WheelSetting {
    pub x: f32,
    pub y: f32,
    pub luma: f32,
}

impl WheelSetting {
    /// The setting held to its range: the puck inside the circle, the
    /// ring from -1 to 1. Values that are not numbers are zero.
    #[must_use]
    pub fn held(self) -> Self {
        let finite = |value: f32| if value.is_finite() { value } else { 0.0 };
        let (mut x, mut y) = (finite(self.x), finite(self.y));
        let length = (x * x + y * y).sqrt();
        if length > 1.0 {
            x /= length;
            y /= length;
        }
        Self {
            x,
            y,
            luma: finite(self.luma).clamp(-1.0, 1.0),
        }
    }

    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.x == 0.0 && self.y == 0.0 && self.luma == 0.0
    }

    /// The colour offset the wheel adds: the puck's hue with its own grey
    /// taken out, plus the ring as grey.
    fn offset(&self) -> [f32; 3] {
        let amount = (self.x * self.x + self.y * self.y).sqrt();
        let mut offset = [self.luma; 3];
        if amount > 0.0 {
            let hue = self.y.atan2(self.x).to_degrees().rem_euclid(360.0);
            let colour = hue_colour(hue);
            let grey = luma(colour);
            for (out, channel) in offset.iter_mut().zip(colour) {
                *out += (channel - grey) * amount;
            }
        }
        offset.map(|value| value * 0.3)
    }
}

/// A clip's curves, wheels and HSL.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Grade {
    /// Hue, saturation and lightness for each HSL colour, -1 to 1.
    pub hsl: [[f32; 3]; 8],
    /// Each curve's points, sorted by x. Empty is the straight line.
    pub curves: [Vec<(f32, f32)>; 4],
    pub wheels: [WheelSetting; 3],
}

impl Grade {
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.hsl.iter().flatten().all(|value| *value == 0.0)
            && self.curves.iter().all(|points| straight(points))
            && self.wheels.iter().all(WheelSetting::is_zero)
    }

    #[must_use]
    pub fn curve(&self, channel: CurveChannel) -> &[(f32, f32)] {
        &self.curves[channel.index()]
    }

    #[must_use]
    pub fn wheel(&self, wheel: Wheel) -> WheelSetting {
        self.wheels[wheel.index()]
    }

    /// Run the grade over premultiplied RGBA.
    pub fn apply(&self, rgba: &mut [u8]) {
        if self.is_identity() {
            return;
        }
        let tables: [Option<[f32; 256]>; 4] = std::array::from_fn(|index| {
            let points = &self.curves[index];
            (!straight(points)).then(|| curve_table(points))
        });
        let wheels = self.wheels.map(|wheel| wheel.offset());
        let hsl_on = self.hsl.iter().flatten().any(|value| *value != 0.0);
        for pixel in rgba.as_chunks_mut::<4>().0 {
            let alpha = f32::from(pixel[3]) / 255.0;
            if alpha <= 0.0 {
                continue;
            }
            let mut colour =
                [0, 1, 2].map(|index| (f32::from(pixel[index]) / 255.0 / alpha).min(1.0));
            if let Some(table) = &tables[0] {
                colour = colour.map(|value| look_up(table, value));
            }
            for (index, value) in colour.iter_mut().enumerate() {
                if let Some(table) = &tables[index + 1] {
                    *value = look_up(table, *value);
                }
            }
            let [lift, gamma, gain] = wheels;
            for (index, value) in colour.iter_mut().enumerate() {
                let lifted = *value + lift[index] * (1.0 - *value);
                let gained = (lifted * (1.0 + gain[index])).max(0.0);
                *value = gained.powf(1.0 / (1.0 + gamma[index]).max(0.1));
            }
            if hsl_on {
                colour = hsl_colour(&self.hsl, colour);
            }
            for (index, value) in colour.into_iter().enumerate() {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let byte = (value.clamp(0.0, 1.0) * alpha * 255.0).round() as u8;
                pixel[index] = byte;
            }
        }
    }
}

/// A clip's grade and basic sliders, the whole of its colour work.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Colour {
    pub grade: Grade,
    pub adjust: Adjust,
}

impl Colour {
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.grade.is_identity() && self.adjust.is_identity()
    }

    /// Run the grade, then the sliders, over premultiplied RGBA.
    pub fn apply(&self, rgba: &mut [u8], width: u32, height: u32, seed: u32) {
        self.grade.apply(rgba);
        adjust_rgba(&self.adjust, rgba, width, height, seed);
    }
}

/// The colour work `segment` lists among its extra materials.
#[must_use]
pub fn segment_colour(draft: &Draft, segment: &Segment) -> Colour {
    Colour {
        grade: segment_grade(draft, segment),
        adjust: segment_adjust(draft, segment),
    }
}

/// The curves, wheels and HSL `segment` lists among its extra materials.
#[must_use]
pub fn segment_grade(draft: &Draft, segment: &Segment) -> Grade {
    let mut grade = Grade::default();
    for id in &segment.extra_material_refs {
        match draft.materials.get(id.as_str()) {
            Some(MaterialRef::Hsl(hsl)) => {
                if let Some(slot) = usize::try_from(hsl.hsl_color_type)
                    .ok()
                    .and_then(|index| grade.hsl.get_mut(index))
                {
                    #[allow(clippy::cast_precision_loss)]
                    let unit = |value: i32| (value.clamp(-100, 100) as f32) / 100.0;
                    *slot = [unit(hsl.hue), unit(hsl.saturation), unit(hsl.lightness)];
                }
            }
            Some(MaterialRef::Other(raw)) if raw.kind == COLOR_CURVES => {
                for channel in CurveChannel::ALL {
                    grade.curves[channel.index()] = read_points(raw.fields.get(channel.key()));
                }
            }
            Some(MaterialRef::Other(raw)) if raw.kind == COLOR_WHEELS => {
                for wheel in Wheel::ALL {
                    grade.wheels[wheel.index()] = read_wheel(raw.fields.get(wheel.key()));
                }
            }
            _ => {}
        }
    }
    grade
}

/// The points of one curve, as stored: held to 0 to 1, sorted by x, one
/// point a column, at most [`MAX_CURVE_POINTS`].
#[must_use]
pub fn tidy_points(points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let mut kept: Vec<(f32, f32)> = points
        .iter()
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)))
        .collect();
    kept.sort_by(|a, b| a.0.total_cmp(&b.0));
    kept.dedup_by(|later, earlier| (later.0 - earlier.0).abs() < 1e-3);
    kept.truncate(MAX_CURVE_POINTS);
    kept
}

/// One curve's points as stored JSON.
#[must_use]
pub fn points_json(points: &[(f32, f32)]) -> Value {
    Value::Array(
        tidy_points(points)
            .into_iter()
            .map(|(x, y)| json!([x, y]))
            .collect(),
    )
}

/// A wheel's setting as stored JSON.
#[must_use]
pub fn wheel_json(setting: WheelSetting) -> Value {
    let setting = setting.held();
    json!([setting.x, setting.y, setting.luma])
}

fn read_points(value: Option<&Value>) -> Vec<(f32, f32)> {
    let Some(Value::Array(items)) = value else {
        return Vec::new();
    };
    #[allow(clippy::cast_possible_truncation)]
    let points: Vec<(f32, f32)> = items
        .iter()
        .filter_map(|item| {
            let pair = item.as_array()?;
            Some((
                pair.first()?.as_f64()? as f32,
                pair.get(1)?.as_f64()? as f32,
            ))
        })
        .collect();
    tidy_points(&points)
}

fn read_wheel(value: Option<&Value>) -> WheelSetting {
    let Some(Value::Array(items)) = value else {
        return WheelSetting::default();
    };
    #[allow(clippy::cast_possible_truncation)]
    let at = |index: usize| items.get(index).and_then(Value::as_f64).unwrap_or_default() as f32;
    WheelSetting {
        x: at(0),
        y: at(1),
        luma: at(2),
    }
    .held()
}

/// A curve with no bend: fewer than two points, or every point on the
/// diagonal from one end to the other.
fn straight(points: &[(f32, f32)]) -> bool {
    let (Some(first), Some(last)) = (points.first(), points.last()) else {
        return true;
    };
    points.len() < 2
        || (first.0 <= 1e-4
            && last.0 >= 1.0 - 1e-4
            && points.iter().all(|(x, y)| (x - y).abs() < 1e-4))
}

/// The curve through `points`, sampled at 256 steps. Before the first
/// point and after the last the curve is flat; between, a monotone cubic.
fn curve_table(points: &[(f32, f32)]) -> [f32; 256] {
    let points = if points.len() < 2 {
        vec![(0.0, 0.0), (1.0, 1.0)]
    } else {
        points.to_vec()
    };
    let count = points.len();
    let slopes: Vec<f32> = points
        .windows(2)
        .map(|pair| {
            let dx = (pair[1].0 - pair[0].0).max(1e-6);
            (pair[1].1 - pair[0].1) / dx
        })
        .collect();
    let mut tangents = vec![0.0f32; count];
    tangents[0] = slopes[0];
    tangents[count - 1] = slopes[count - 2];
    for index in 1..count - 1 {
        let (before, after) = (slopes[index - 1], slopes[index]);
        tangents[index] = if before * after <= 0.0 {
            0.0
        } else {
            (before + after) / 2.0
        };
    }
    std::array::from_fn(|step| {
        #[allow(clippy::cast_precision_loss)]
        let x = step as f32 / 255.0;
        if x <= points[0].0 {
            return points[0].1;
        }
        if x >= points[count - 1].0 {
            return points[count - 1].1;
        }
        let index = points
            .windows(2)
            .position(|pair| x <= pair[1].0)
            .unwrap_or(count - 2);
        let (x0, y0) = points[index];
        let (x1, y1) = points[index + 1];
        let width = (x1 - x0).max(1e-6);
        let t = (x - x0) / width;
        let (t2, t3) = (t * t, t * t * t);
        let value = (2.0 * t3 - 3.0 * t2 + 1.0) * y0
            + (t3 - 2.0 * t2 + t) * width * tangents[index]
            + (-2.0 * t3 + 3.0 * t2) * y1
            + (t3 - t2) * width * tangents[index + 1];
        value.clamp(0.0, 1.0)
    })
}

fn look_up(table: &[f32; 256], value: f32) -> f32 {
    let position = value.clamp(0.0, 1.0) * 255.0;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let low = position.floor() as usize;
    let high = (low + 1).min(255);
    #[allow(clippy::cast_precision_loss)]
    let t = position - low as f32;
    table[low] * (1.0 - t) + table[high] * t
}

fn luma([r, g, b]: [f32; 3]) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Full colour at `hue` degrees.
fn hue_colour(hue: f32) -> [f32; 3] {
    hsl_to_rgb(hue, 1.0, 0.5)
}

fn rgb_to_hsl([r, g, b]: [f32; 3]) -> (f32, f32, f32) {
    let high = r.max(g).max(b);
    let low = r.min(g).min(b);
    let light = (high + low) / 2.0;
    let range = high - low;
    if range <= 1e-6 {
        return (0.0, 0.0, light);
    }
    let saturation = range / (1.0 - (2.0 * light - 1.0).abs()).max(1e-6);
    let hue = if high == r {
        60.0 * ((g - b) / range).rem_euclid(6.0)
    } else if high == g {
        60.0 * ((b - r) / range + 2.0)
    } else {
        60.0 * ((r - g) / range + 4.0)
    };
    (hue, saturation.min(1.0), light)
}

fn hsl_to_rgb(hue: f32, saturation: f32, light: f32) -> [f32; 3] {
    let chroma = (1.0 - (2.0 * light - 1.0).abs()) * saturation;
    let sector = hue.rem_euclid(360.0) / 60.0;
    let second = chroma * (1.0 - (sector.rem_euclid(2.0) - 1.0).abs());
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (r, g, b) = match sector as u32 {
        0 => (chroma, second, 0.0),
        1 => (second, chroma, 0.0),
        2 => (0.0, chroma, second),
        3 => (0.0, second, chroma),
        4 => (second, 0.0, chroma),
        _ => (chroma, 0.0, second),
    };
    let base = light - chroma / 2.0;
    [r + base, g + base, b + base]
}

/// How much the colour at `hue` belongs to HSL colour `index`: 1 at its
/// centre, falling to 0 at the next colour either way.
fn hsl_weight(index: usize, hue: f32) -> f32 {
    let centre = HSL_CENTRES[index];
    let next = HSL_CENTRES[(index + 1) % 8] + if index == 7 { 360.0 } else { 0.0 };
    let previous = HSL_CENTRES[(index + 7) % 8] - if index == 0 { 360.0 } else { 0.0 };
    let mut delta = (hue - centre).rem_euclid(360.0);
    if delta > 180.0 {
        delta -= 360.0;
    }
    let reach = if delta >= 0.0 {
        next - centre
    } else {
        centre - previous
    };
    (1.0 - delta.abs() / reach).max(0.0)
}

fn hsl_colour(settings: &[[f32; 3]; 8], colour: [f32; 3]) -> [f32; 3] {
    let (hue, saturation, light) = rgb_to_hsl(colour);
    if saturation <= 1e-4 {
        return colour;
    }
    let (mut turn, mut scale, mut lift) = (0.0, 0.0, 0.0);
    for (index, [h, s, l]) in settings.iter().enumerate() {
        let weight = hsl_weight(index, hue);
        if weight > 0.0 {
            turn += h * weight;
            scale += s * weight;
            lift += l * weight;
        }
    }
    let hue = hue + turn * HSL_TURN;
    let saturation = (saturation * (1.0 + scale)).clamp(0.0, 1.0);
    let light = (light + lift * 0.3 * saturation).clamp(0.0, 1.0);
    hsl_to_rgb(hue, saturation, light)
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, MaterialHsl, MaterialId, RawMaterial, Segment, TimeRange};
    use serde_json::json;

    use super::{
        COLOR_CURVES, COLOR_WHEELS, CurveChannel, Grade, Wheel, WheelSetting, segment_grade,
        tidy_points,
    };

    fn run(grade: &Grade, pixel: [u8; 4]) -> [u8; 4] {
        let mut bytes = pixel.to_vec();
        grade.apply(&mut bytes);
        [bytes[0], bytes[1], bytes[2], bytes[3]]
    }

    #[test]
    fn nothing_set_leaves_the_picture_alone() {
        let grade = Grade::default();
        assert!(grade.is_identity());
        assert_eq!(run(&grade, [10, 200, 30, 255]), [10, 200, 30, 255]);
        let mut diagonal = Grade::default();
        diagonal.curves[0] = vec![(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)];
        assert!(diagonal.is_identity());
    }

    #[test]
    fn a_curve_bends_its_channel_only() {
        let mut grade = Grade::default();
        grade.curves[CurveChannel::Red as usize] = vec![(0.0, 0.0), (0.5, 0.75), (1.0, 1.0)];
        let out = run(&grade, [128, 128, 128, 255]);
        assert!((190..=192).contains(&out[0]), "{out:?}");
        assert_eq!(&out[1..], &[128, 128, 255]);
        let mut grade = Grade::default();
        grade.curves[0] = vec![(0.0, 1.0), (1.0, 0.0)];
        assert_eq!(run(&grade, [0, 255, 51, 255]), [255, 0, 204, 255]);
    }

    #[test]
    fn wheels_tint_their_tones() {
        let mut grade = Grade::default();
        grade.wheels[Wheel::Gain as usize] = WheelSetting {
            x: 1.0,
            y: 0.0,
            luma: 0.0,
        };
        let out = run(&grade, [200, 200, 200, 255]);
        assert!(out[0] > 200 && out[1] < 200, "{out:?}");
        let mut grade = Grade::default();
        grade.wheels[Wheel::Lift as usize] = WheelSetting {
            x: 0.0,
            y: 0.0,
            luma: 1.0,
        };
        assert!(run(&grade, [0, 0, 0, 255])[0] > 60);
        assert!(run(&grade, [255, 255, 255, 255])[0] == 255);
    }

    #[test]
    fn hsl_moves_one_colour_and_leaves_grey() {
        let mut grade = Grade::default();
        grade.hsl[0] = [0.0, -1.0, 0.0];
        let red = run(&grade, [220, 30, 30, 255]);
        assert_eq!(red[0], red[1]);
        let blue = run(&grade, [30, 30, 220, 255]);
        assert_eq!(blue, [30, 30, 220, 255]);
        assert_eq!(run(&grade, [90, 90, 90, 255]), [90, 90, 90, 255]);
        let mut grade = Grade::default();
        grade.hsl[5] = [1.0, 0.0, 0.0];
        let turned = run(&grade, [30, 30, 220, 255]);
        assert!(turned[0] > 60, "{turned:?}");
    }

    #[test]
    fn the_draft_holds_each_part() {
        let mut draft = Draft::empty("D");
        draft.materials.hsl.push(MaterialHsl {
            id: MaterialId::new("h"),
            kind: "hsl".to_owned(),
            hsl_color_type: 3,
            hue: 50,
            saturation: -20,
            lightness: 200,
            ..MaterialHsl::default()
        });
        let mut curves = RawMaterial {
            id: MaterialId::new("c"),
            kind: COLOR_CURVES.to_owned(),
            ..RawMaterial::default()
        };
        curves.fields.insert(
            "blue".to_owned(),
            json!([[1.0, 0.5], [0.0, 0.2], [2.0, 9.0]]),
        );
        draft.materials.color_curves.push(curves);
        let mut wheels = RawMaterial {
            id: MaterialId::new("w"),
            kind: COLOR_WHEELS.to_owned(),
            ..RawMaterial::default()
        };
        wheels
            .fields
            .insert("gamma".to_owned(), json!([3.0, 4.0, -0.5]));
        draft.materials.primary_color_wheels.push(wheels);
        let mut segment = Segment::new("s", "m", TimeRange::new(0, 1));
        for id in ["h", "c", "w"] {
            segment.extra_material_refs.push(MaterialId::new(id));
        }
        let grade = segment_grade(&draft, &segment);
        assert_eq!(grade.hsl[3], [0.5, -0.2, 1.0]);
        assert_eq!(grade.curve(CurveChannel::Blue), &[(0.0, 0.2), (1.0, 0.5)]);
        let gamma = grade.wheel(Wheel::Gamma);
        assert!((gamma.x - 0.6).abs() < 1e-6 && (gamma.y - 0.8).abs() < 1e-6);
        assert_eq!(gamma.luma, -0.5);
        assert!(!grade.is_identity());
        assert_eq!(tidy_points(&[(0.5, 0.5), (0.5004, 0.1)]).len(), 1);
    }
}
