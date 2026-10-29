use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::enums::SpeedMode;
use crate::id::MaterialId;
use crate::time::Micros;

fn one() -> f64 {
    1.0
}

/// Why a constant-speed duration was not produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstantSpeedError {
    NonPositiveDuration,
    NonPositiveSpeed,
    Unrepresentable,
    /// Curve mode is mapped by [`curve_target_duration`], not the constant quotient.
    Curve,
    /// Fewer than two points. The constant speed number stays in force.
    TooFewPoints,
}

/// Target duration of a constant speed: the source duration divided by `speed`,
/// rounded to the nearest microsecond. A non-positive duration or speed is refused.
pub fn constant_target_duration(source: Micros, speed: f64) -> Result<Micros, ConstantSpeedError> {
    if source.0 <= 0 {
        return Err(ConstantSpeedError::NonPositiveDuration);
    }
    if !speed.is_finite() || speed <= 0.0 {
        return Err(ConstantSpeedError::NonPositiveSpeed);
    }
    if source.0 > (1_i64 << 53) {
        return Err(ConstantSpeedError::Unrepresentable);
    }
    let target = (source.0 as f64) / speed;
    let target = target.round();
    if !(target.is_finite() && target > 0.0 && target < i64::MAX as f64) {
        return Err(ConstantSpeedError::Unrepresentable);
    }
    Ok(Micros(target as i64))
}

/// Integral from 0 to 1 of `dx / y` for a curve whose `y` is linear in `x`.
///
/// `x` is the fraction of the source duration. Ahead of the first point and
/// behind the last, `y` holds. A point outside 0 to 1 stays in the list; only
/// the span inside that interval is integrated. Two points with the same `x`
/// keep the later `y`. A `y` at or below 0 is refused.
pub fn curve_speed_factor(points: &[SpeedPoint]) -> Result<f64, ConstantSpeedError> {
    let prepared = prepared_points(points)?;
    let mut total = 0.0;
    let Some(&(first_x, first_y)) = prepared.first() else {
        return Err(ConstantSpeedError::TooFewPoints);
    };
    let Some(&(last_x, last_y)) = prepared.last() else {
        return Err(ConstantSpeedError::TooFewPoints);
    };
    if first_x > 0.0 {
        total += speed_piece(0.0, first_x.min(1.0), first_y, first_y)?;
    }
    for pair in prepared.windows(2) {
        let (left_x, left_y) = pair[0];
        let (right_x, right_y) = pair[1];
        let start = left_x.max(0.0);
        let end = right_x.min(1.0);
        if end <= start {
            continue;
        }
        let y0 = interpolate(left_x, left_y, right_x, right_y, start);
        let y1 = interpolate(left_x, left_y, right_x, right_y, end);
        total += speed_piece(start, end, y0, y1)?;
    }
    if last_x < 1.0 {
        total += speed_piece(last_x.max(0.0), 1.0, last_y, last_y)?;
    }
    if !total.is_finite() || total <= 0.0 {
        return Err(ConstantSpeedError::Unrepresentable);
    }
    Ok(total)
}

/// Target duration is the source duration times [`curve_speed_factor`].
pub fn curve_target_duration(
    source: Micros,
    points: &[SpeedPoint],
) -> Result<Micros, ConstantSpeedError> {
    if source.0 <= 0 {
        return Err(ConstantSpeedError::NonPositiveDuration);
    }
    if source.0 > (1_i64 << 53) {
        return Err(ConstantSpeedError::Unrepresentable);
    }
    let target = (source.0 as f64) * curve_speed_factor(points)?;
    let target = target.round();
    if !(target.is_finite() && target > 0.0 && target < i64::MAX as f64) {
        return Err(ConstantSpeedError::Unrepresentable);
    }
    Ok(Micros(target as i64))
}

fn prepared_points(points: &[SpeedPoint]) -> Result<Vec<(f64, f64)>, ConstantSpeedError> {
    let mut indexed = Vec::with_capacity(points.len());
    for (index, point) in points.iter().enumerate() {
        if !point.x.is_finite() {
            return Err(ConstantSpeedError::Unrepresentable);
        }
        if !point.y.is_finite() || point.y <= 0.0 {
            return Err(ConstantSpeedError::NonPositiveSpeed);
        }
        indexed.push((index, point.x, point.y));
    }
    indexed.sort_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)));
    let mut prepared: Vec<(f64, f64)> = Vec::new();
    for (_, x, y) in indexed {
        if let Some(last) = prepared.last_mut()
            && last.0 == x
        {
            last.1 = y;
            continue;
        }
        prepared.push((x, y));
    }
    if prepared.len() < 2 {
        return Err(ConstantSpeedError::TooFewPoints);
    }
    Ok(prepared)
}

fn interpolate(x0: f64, y0: f64, x1: f64, y1: f64, x: f64) -> f64 {
    let span = x1 - x0;
    if span == 0.0 {
        return y1;
    }
    let t = (x - x0) / span;
    y0 + (y1 - y0) * t
}

/// One straight piece. Constant `y` is `width / y`. A changing piece is
/// `(x1 - x0) * ln(y1 / y0) / (y1 - y0)`.
fn speed_piece(x0: f64, x1: f64, y0: f64, y1: f64) -> Result<f64, ConstantSpeedError> {
    if !(y0.is_finite() && y1.is_finite()) || y0 <= 0.0 || y1 <= 0.0 {
        return Err(ConstantSpeedError::NonPositiveSpeed);
    }
    let width = x1 - x0;
    if width <= 0.0 {
        return Ok(0.0);
    }
    let value = if y0 == y1 {
        width / y0
    } else {
        width * (y1 / y0).ln() / (y1 - y0)
    };
    if !value.is_finite() {
        return Err(ConstantSpeedError::Unrepresentable);
    }
    Ok(value)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpeedPoint {
    pub x: f64,
    pub y: f64,
    /// Keys this model does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct CurveSpeed {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub source_platform: i32,
    #[serde(default)]
    pub speed_points: Vec<SpeedPoint>,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialSpeed {
    #[serde(default)]
    pub id: MaterialId,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub mode: SpeedMode,
    #[serde(default = "one")]
    pub speed: f64,
    #[serde(default)]
    pub curve_speed: Option<CurveSpeed>,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl MaterialSpeed {
    /// Resource-mapping fallback. An absent key stays at 1.
    pub const MAPPING_SPEED: f64 = 0.0;

    pub fn constant_target(&self, source: Micros) -> Result<Micros, ConstantSpeedError> {
        if self.mode == SpeedMode::Curve {
            return Err(ConstantSpeedError::Curve);
        }
        constant_target_duration(source, self.speed)
    }
}

impl Default for MaterialSpeed {
    fn default() -> Self {
        Self {
            id: MaterialId::default(),
            kind: String::new(),
            mode: SpeedMode::Normal,
            speed: 1.0,
            curve_speed: None,
            rest: Map::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_speed_material_defaults_to_constant_one() {
        let speed: MaterialSpeed =
            serde_json::from_str(r#"{"id":"s","type":"speed","mode":0,"curve_speed":null}"#)
                .unwrap();
        assert_eq!(speed.mode, SpeedMode::Normal);
        assert_eq!(speed.speed, 1.0);
        assert_ne!(speed.speed, MaterialSpeed::MAPPING_SPEED);
        assert!(speed.curve_speed.is_none());
        let absent: MaterialSpeed = serde_json::from_str("{}").unwrap();
        assert_eq!(absent.speed, 1.0);
        let curve: MaterialSpeed = serde_json::from_str(
            r#"{"id":"c","type":"speed","mode":1,"speed":1.0,"curve_speed":{"name":"","speed_points":[{"x":0.0,"y":1.0,"z":1},{"x":1.0,"y":1.0}]}}"#,
        )
        .unwrap();
        assert_eq!(curve.mode, SpeedMode::Curve);
        let points = curve.curve_speed.as_ref().unwrap().speed_points.clone();
        assert_eq!(points.len(), 2);
        assert_eq!(points[0].rest["z"], 1);
        assert_eq!(
            curve.constant_target(Micros(1_000_000)),
            Err(ConstantSpeedError::Curve)
        );
    }

    #[test]
    fn a_constant_speed_maps_source_duration_onto_the_target() {
        assert_eq!(
            constant_target_duration(Micros(2_000_000), 2.0),
            Ok(Micros(1_000_000))
        );
        assert_eq!(
            constant_target_duration(Micros(1_000_000), 0.5),
            Ok(Micros(2_000_000))
        );
        assert_eq!(
            constant_target_duration(Micros(1_000_000), 1.5),
            Ok(Micros(666_667))
        );
        assert_eq!(
            MaterialSpeed::default().constant_target(Micros(2_000_000)),
            Ok(Micros(2_000_000))
        );
        assert_eq!(
            constant_target_duration(Micros(0), 1.0),
            Err(ConstantSpeedError::NonPositiveDuration)
        );
        assert_eq!(
            constant_target_duration(Micros(1_000), 0.0),
            Err(ConstantSpeedError::NonPositiveSpeed)
        );
        assert_eq!(
            constant_target_duration(Micros(1_000), f64::NAN),
            Err(ConstantSpeedError::NonPositiveSpeed)
        );
    }

    fn point(x: f64, y: f64) -> SpeedPoint {
        SpeedPoint {
            x,
            y,
            rest: Map::new(),
        }
    }

    #[test]
    fn a_curve_integrates_dx_over_linear_y() {
        let flat = [point(0.0, 1.0), point(1.0, 1.0)];
        assert_eq!(
            curve_target_duration(Micros(2_000_000), &flat),
            Ok(Micros(2_000_000))
        );
        let twice = [point(0.0, 2.0), point(1.0, 2.0)];
        assert_eq!(
            curve_target_duration(Micros(2_000_000), &twice),
            Ok(Micros(1_000_000))
        );
        let ramp = [point(0.0, 1.0), point(1.0, 2.0)];
        assert_eq!(
            curve_target_duration(Micros(1_000_000), &ramp),
            Ok(Micros(693_147))
        );
        let held = [point(0.25, 2.0), point(0.75, 2.0)];
        assert_eq!(
            curve_target_duration(Micros(2_000_000), &held),
            Ok(Micros(1_000_000))
        );
        let later = [point(0.0, 1.0), point(0.0, 4.0), point(1.0, 4.0)];
        assert_eq!(
            curve_target_duration(Micros(1_000_000), &later),
            Ok(Micros(250_000))
        );
        assert_eq!(
            curve_speed_factor(&[point(0.0, 0.0), point(1.0, 1.0)]),
            Err(ConstantSpeedError::NonPositiveSpeed)
        );
        assert_eq!(
            curve_speed_factor(&[point(0.0, 1.0)]),
            Err(ConstantSpeedError::TooFewPoints)
        );
        let crossing = [point(-0.2, 1.0), point(1.2, 3.0)];
        assert_eq!(
            curve_target_duration(Micros(1_000_000), &crossing),
            Ok(Micros(523_050))
        );
    }
}
