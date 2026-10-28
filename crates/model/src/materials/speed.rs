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
    /// The curve formula is still open, so a curve is not mapped.
    Curve,
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
}
