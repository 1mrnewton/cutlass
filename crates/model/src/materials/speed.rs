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

/// Cut a curve where `share` of its timeline length has played. Returns
/// the share of the source played by then, and the two halves' curves,
/// each stretched back over 0 to 1, so each half's integral is its own
/// length. `share` is strictly between 0 and 1.
pub fn split_curve(
    points: &[SpeedPoint],
    share: f64,
) -> Result<(f64, Vec<SpeedPoint>, Vec<SpeedPoint>), ConstantSpeedError> {
    if !(share.is_finite() && share > 0.0 && share < 1.0) {
        return Err(ConstantSpeedError::NonPositiveDuration);
    }
    let corners = corners(&prepared_points(points)?);
    let total = curve_speed_factor(points)?;
    let mut left = share * total;
    let mut cut = None;
    for pair in corners.windows(2) {
        let ((x0, y0), (x1, y1)) = (pair[0], pair[1]);
        let piece = speed_piece(x0, x1, y0, y1)?;
        if left > piece {
            left -= piece;
            continue;
        }
        // Invert dF = dx / y on a piece where y is linear in x.
        let width = x1 - x0;
        let slope = (y1 - y0) / width;
        let along = if slope == 0.0 {
            left * y0
        } else {
            y0 * (slope * left).exp_m1() / slope
        };
        let x = (x0 + along).clamp(x0, x1);
        cut = Some((x, interpolate(x0, y0, x1, y1, x)));
        break;
    }
    let (x, y) = cut.ok_or(ConstantSpeedError::Unrepresentable)?;
    if !(x > 0.0 && x < 1.0) {
        return Err(ConstantSpeedError::NonPositiveDuration);
    }
    let point = |x: f64, y: f64| SpeedPoint {
        x,
        y,
        rest: Map::new(),
    };
    let mut before: Vec<SpeedPoint> = corners
        .iter()
        .filter(|corner| corner.0 < x)
        .map(|&(cx, cy)| point(cx / x, cy))
        .collect();
    before.push(point(1.0, y));
    let mut after = vec![point(0.0, y)];
    after.extend(
        corners
            .iter()
            .filter(|corner| corner.0 > x)
            .map(|&(cx, cy)| point((cx - x) / (1.0 - x), cy)),
    );
    Ok((x, before, after))
}

/// The curve from 0 to 1 as straight pieces: 0, each point inside, and 1,
/// with the speed held before the first point and after the last.
fn corners(prepared: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let at = |x: f64| -> f64 {
        let (first, last) = (prepared[0], prepared[prepared.len() - 1]);
        if x <= first.0 {
            return first.1;
        }
        if x >= last.0 {
            return last.1;
        }
        prepared
            .windows(2)
            .find(|pair| x <= pair[1].0)
            .map_or(last.1, |pair| {
                interpolate(pair[0].0, pair[0].1, pair[1].0, pair[1].1, x)
            })
    };
    let mut corners = vec![(0.0, at(0.0))];
    corners.extend(
        prepared
            .iter()
            .filter(|point| point.0 > 0.0 && point.0 < 1.0)
            .copied(),
    );
    corners.push((1.0, at(1.0)));
    corners
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

impl SpeedPoint {
    /// A point at `x`, the share of the source, running at `y` times speed.
    #[must_use]
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            x,
            y,
            rest: Map::new(),
        }
    }
}

/// A ready-made speed curve. Each is a list of points: `x` the share of the
/// source, `y` the speed there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeedPreset {
    /// Fast, slow, fast, slow: cuts a long shot into beats.
    Montage,
    /// Quick in, a long slow middle, quick out.
    Hero,
    /// Fast, a near stop in the middle, fast again.
    Bullet,
    /// Normal, a sudden burst, normal.
    JumpCut,
    /// Starts fast and settles to normal.
    FlashIn,
    /// Starts normal and ends fast.
    FlashOut,
}

impl SpeedPreset {
    pub const ALL: [Self; 6] = [
        Self::Montage,
        Self::Hero,
        Self::Bullet,
        Self::JumpCut,
        Self::FlashIn,
        Self::FlashOut,
    ];

    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Montage => "Montage",
            Self::Hero => "Hero",
            Self::Bullet => "Bullet",
            Self::JumpCut => "Jump cut",
            Self::FlashIn => "Flash in",
            Self::FlashOut => "Flash out",
        }
    }

    /// The preset's points, from `x` 0 to 1.
    #[must_use]
    pub fn points(self) -> Vec<SpeedPoint> {
        let pairs: &[(f64, f64)] = match self {
            Self::Montage => &[
                (0.0, 1.0),
                (0.15, 2.5),
                (0.35, 0.6),
                (0.55, 2.5),
                (0.8, 0.6),
                (1.0, 1.0),
            ],
            Self::Hero => &[
                (0.0, 1.0),
                (0.2, 2.0),
                (0.4, 0.4),
                (0.6, 0.4),
                (0.8, 2.0),
                (1.0, 1.0),
            ],
            Self::Bullet => &[
                (0.0, 2.0),
                (0.35, 2.0),
                (0.45, 0.2),
                (0.55, 0.2),
                (0.65, 2.0),
                (1.0, 2.0),
            ],
            Self::JumpCut => &[
                (0.0, 1.0),
                (0.4, 1.0),
                (0.45, 5.0),
                (0.55, 5.0),
                (0.6, 1.0),
                (1.0, 1.0),
            ],
            Self::FlashIn => &[(0.0, 5.0), (0.3, 3.0), (0.5, 1.0), (1.0, 1.0)],
            Self::FlashOut => &[(0.0, 1.0), (0.5, 1.0), (0.7, 3.0), (1.0, 5.0)],
        };
        pairs.iter().map(|&(x, y)| SpeedPoint::new(x, y)).collect()
    }

    /// The preset `points` are, when they are one unchanged.
    #[must_use]
    pub fn of(points: &[SpeedPoint]) -> Option<Self> {
        Self::ALL.into_iter().find(|preset| {
            let own = preset.points();
            own.len() == points.len()
                && own
                    .iter()
                    .zip(points)
                    .all(|(a, b)| (a.x - b.x).abs() < 1e-6 && (a.y - b.y).abs() < 1e-6)
        })
    }
}

/// The speed a curve runs at `x`, the share of the source: linear between
/// points and held past either end. One at no points.
#[must_use]
pub fn curve_speed_at(points: &[SpeedPoint], x: f64) -> f64 {
    let Some(first) = points.first() else {
        return 1.0;
    };
    if x <= first.x {
        return first.y;
    }
    for pair in points.windows(2) {
        let (left, right) = (&pair[0], &pair[1]);
        if x <= right.x {
            let width = right.x - left.x;
            if width <= 0.0 {
                return right.y;
            }
            return left.y + (right.y - left.y) * (x - left.x) / width;
        }
    }
    points.last().map_or(1.0, |last| last.y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_split_curve_keeps_each_halfs_length_and_the_speed_at_the_cut() {
        let point = |x: f64, y: f64| SpeedPoint {
            x,
            y,
            rest: Map::new(),
        };
        let points = [point(0.0, 1.0), point(0.5, 4.0), point(1.0, 0.5)];
        let source = Micros(10_000_000);
        let whole = curve_target_duration(source, &points).unwrap().0 as f64;
        for share in [0.1, 0.3, 0.5, 0.8] {
            let (x, before, after) = split_curve(&points, share).unwrap();
            let left = curve_speed_factor(&before).unwrap() * x * source.0 as f64;
            let right = curve_speed_factor(&after).unwrap() * (1.0 - x) * source.0 as f64;
            assert!((left - share * whole).abs() < 1.0, "{share}: {left}");
            assert!(
                (right - (1.0 - share) * whole).abs() < 1.0,
                "{share}: {right}"
            );
            assert_eq!(before.last().unwrap().y, after[0].y);
        }
        let flat = [point(0.0, 2.0), point(1.0, 2.0)];
        let (x, _, _) = split_curve(&flat, 0.25).unwrap();
        assert!((x - 0.25).abs() < 1e-12);
        assert!(split_curve(&flat, 1.0).is_err());
    }

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

    #[test]
    fn each_preset_is_a_curve_that_changes_the_length_and_is_known_again() {
        for preset in SpeedPreset::ALL {
            let points = preset.points();
            let factor = curve_speed_factor(&points).unwrap();
            assert!(
                factor > 0.0 && (factor - 1.0).abs() > 0.01,
                "{preset:?} {factor}"
            );
            assert_eq!(SpeedPreset::of(&points), Some(preset));
            assert!((curve_speed_at(&points, 0.0) - points[0].y).abs() < 1e-9);
        }
        // Flash in plays its fast start in less time than a normal one.
        assert!(curve_speed_factor(&SpeedPreset::FlashIn.points()).unwrap() < 1.0);
        let mut moved = SpeedPreset::Hero.points();
        moved[2].y = 0.5;
        assert_eq!(SpeedPreset::of(&moved), None);
        assert!((curve_speed_at(&moved, 0.3) - 1.25).abs() < 1e-9);
        assert!((curve_speed_at(&[], 0.5) - 1.0).abs() < 1e-9);
    }
}
