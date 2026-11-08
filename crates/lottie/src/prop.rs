//! Keyframed properties: static or animated values, eased between keys.
//!
//! A number list stands for a scalar, a point, a color or a gradient. A key
//! holds its start value and, in the older layout, its end value; otherwise
//! the next key's start is the end. A hold key keeps its start until the
//! next key. Between keys the progress is eased by the cubic through
//! `(0, 0)`, the key's out handle `o`, the in handle `i`, and `(1, 1)`, one
//! curve per dimension when the handles list one. A point with spatial
//! tangents `to` and `ti` moves along the cubic through them, at the eased
//! fraction of that cubic's length.

use serde_json::Value;

use crate::LottieError;
use crate::geom::{Cubic, Point, point_at_length_fraction};
use crate::limits::Limits;
use crate::shape::PathShape;

/// A value that can sit between two keys.
pub(crate) trait Lerp: Clone {
    /// `self` toward `other`, dimension `d` moved by `progress(d)`.
    fn lerp(&self, other: &Self, progress: &dyn Fn(usize) -> f64) -> Self;
}

impl Lerp for Vec<f64> {
    fn lerp(&self, other: &Self, progress: &dyn Fn(usize) -> f64) -> Self {
        self.iter()
            .enumerate()
            .map(|(index, start)| match other.get(index) {
                Some(end) => start + (end - start) * progress(index),
                None => *start,
            })
            .collect()
    }
}

impl Lerp for PathShape {
    fn lerp(&self, other: &Self, progress: &dyn Fn(usize) -> f64) -> Self {
        if self.vertices.len() != other.vertices.len() {
            return self.clone();
        }
        let t = progress(0);
        let mix = |a: &[Point], b: &[Point]| -> Vec<Point> {
            a.iter().zip(b).map(|(a, b)| a.lerp(*b, t)).collect()
        };
        Self {
            closed: self.closed,
            vertices: mix(&self.vertices, &other.vertices),
            in_tangents: mix(&self.in_tangents, &other.in_tangents),
            out_tangents: mix(&self.out_tangents, &other.out_tangents),
        }
    }
}

/// One curve's handles, a list per dimension.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Handle {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Handle {
    fn at(&self, dimension: usize) -> Option<(f64, f64)> {
        let pick = |list: &[f64]| list.get(dimension).or_else(|| list.first()).copied();
        Some((pick(&self.x)?, pick(&self.y)?))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Key<T> {
    pub(crate) time: f64,
    pub(crate) start: Option<T>,
    pub(crate) end: Option<T>,
    pub(crate) hold: bool,
    out_handle: Handle,
    in_handle: Handle,
    /// Spatial tangents, out of the start and into the end.
    spatial: Option<(Point, Point)>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Prop<T> {
    Static(T),
    Animated(Vec<Key<T>>),
}

/// The eased progress for the curve through `(x1, y1)` and `(x2, y2)` at
/// time fraction `x`.
pub(crate) fn ease(x1: f64, y1: f64, x2: f64, y2: f64, x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    let (x1, x2) = (x1.clamp(0.0, 1.0), x2.clamp(0.0, 1.0));
    if !(y1.is_finite() && y2.is_finite()) {
        return x;
    }
    let bez = |a: f64, b: f64, t: f64| {
        let u = 1.0 - t;
        3.0 * u * u * t * a + 3.0 * u * t * t * b + t * t * t
    };
    // x(t) rises from 0 to 1 since both controls lie in [0, 1].
    let (mut low, mut high) = (0.0_f64, 1.0_f64);
    let mut t = x;
    for _ in 0..8 {
        let error = bez(x1, x2, t) - x;
        if error.abs() < 1e-7 {
            return bez(y1, y2, t);
        }
        let u = 1.0 - t;
        let slope = 3.0 * u * u * x1 + 6.0 * u * t * (x2 - x1) + 3.0 * t * t * (1.0 - x2);
        if slope.abs() < 1e-6 {
            break;
        }
        t = (t - error / slope).clamp(0.0, 1.0);
    }
    for _ in 0..40 {
        t = 0.5 * (low + high);
        if bez(x1, x2, t) < x {
            low = t;
        } else {
            high = t;
        }
    }
    bez(y1, y2, t)
}

impl<T: Lerp> Prop<T> {
    pub(crate) fn at(&self, time: f64) -> Option<T> {
        match self {
            Self::Static(value) => Some(value.clone()),
            Self::Animated(keys) => match locate(keys, time)? {
                Located::Exact(value) => Some(value),
                Located::Between {
                    key,
                    start,
                    end,
                    fraction,
                } => Some(start.lerp(&end, &|dimension| key.progress(dimension, fraction))),
            },
        }
    }
}

impl Prop<Vec<f64>> {
    /// The first number, or `fallback`.
    pub(crate) fn scalar(&self, time: f64, fallback: f64) -> f64 {
        self.at(time)
            .and_then(|value| value.first().copied())
            .filter(|value| value.is_finite())
            .unwrap_or(fallback)
    }

    /// The first two numbers; a lone number is used for both. A key with
    /// spatial tangents moves along its cubic.
    pub(crate) fn point(&self, time: f64, fallback: Point) -> Point {
        let value = match self {
            Self::Static(value) => Some(value.clone()),
            Self::Animated(keys) => match locate(keys, time) {
                None => None,
                Some(Located::Exact(value)) => Some(value),
                Some(Located::Between {
                    key,
                    start,
                    end,
                    fraction,
                }) => match (key.spatial, two(&start), two(&end)) {
                    (Some((out_tangent, in_tangent)), Some(from), Some(to))
                        if out_tangent != Point::default() || in_tangent != Point::default() =>
                    {
                        let cubic: Cubic = [from, from.add(out_tangent), to.add(in_tangent), to];
                        let point = point_at_length_fraction(&cubic, key.progress(0, fraction));
                        Some(vec![point.x, point.y])
                    }
                    _ => Some(start.lerp(&end, &|dimension| key.progress(dimension, fraction))),
                },
            },
        };
        let Some(value) = value else {
            return fallback;
        };
        let x = value.first().copied().unwrap_or(fallback.x);
        let y = value.get(1).copied().unwrap_or(x);
        let point = Point::new(x, y);
        if point.is_finite() { point } else { fallback }
    }

    pub(crate) fn constant(value: f64) -> Self {
        Self::Static(vec![value])
    }
}

fn two(value: &[f64]) -> Option<Point> {
    Some(Point::new(*value.first()?, *value.get(1)?))
}

impl<T> Key<T> {
    fn progress(&self, dimension: usize, fraction: f64) -> f64 {
        match (self.out_handle.at(dimension), self.in_handle.at(dimension)) {
            (Some((x1, y1)), Some((x2, y2))) => ease(x1, y1, x2, y2, fraction),
            _ => fraction,
        }
    }
}

enum Located<'a, T> {
    Exact(T),
    Between {
        key: &'a Key<T>,
        start: T,
        end: T,
        fraction: f64,
    },
}

fn locate<T: Clone>(keys: &[Key<T>], time: f64) -> Option<Located<'_, T>> {
    let first = keys.first()?;
    if time.is_nan() || time <= first.time {
        return first
            .start
            .clone()
            .or_else(|| first.end.clone())
            .map(Located::Exact);
    }
    // The last key at or before `time`.
    let index = keys
        .partition_point(|key| key.time <= time)
        .saturating_sub(1);
    let key = &keys[index];
    let Some(next) = keys.get(index + 1) else {
        // After the last key: its start, or the end the key before it had.
        return key
            .start
            .clone()
            .or_else(|| {
                index
                    .checked_sub(1)
                    .and_then(|before| keys[before].end.clone())
            })
            .map(Located::Exact);
    };
    let start = key.start.clone()?;
    if key.hold {
        return Some(Located::Exact(start));
    }
    let Some(end) = key.end.clone().or_else(|| next.start.clone()) else {
        return Some(Located::Exact(start));
    };
    let span = next.time - key.time;
    let fraction = if span > 0.0 {
        ((time - key.time) / span).clamp(0.0, 1.0)
    } else {
        1.0
    };
    Some(Located::Between {
        key,
        start,
        end,
        fraction,
    })
}

/// Numbers from a number or a list of numbers.
pub(crate) fn numbers(value: &Value, limits: &mut Limits) -> Result<Option<Vec<f64>>, LottieError> {
    match value {
        Value::Number(number) => Ok(number.as_f64().map(|value| vec![value])),
        Value::Array(items) => {
            limits.numbers(items.len())?;
            Ok(items.iter().map(Value::as_f64).collect())
        }
        _ => Ok(None),
    }
}

fn handle(value: Option<&Value>, limits: &mut Limits) -> Result<Handle, LottieError> {
    let Some(Value::Object(object)) = value else {
        return Ok(Handle::default());
    };
    let list = |key: &str, limits: &mut Limits| -> Result<Vec<f64>, LottieError> {
        Ok(match object.get(key) {
            Some(value) => numbers(value, limits)?.unwrap_or_default(),
            None => Vec::new(),
        })
    };
    Ok(Handle {
        x: list("x", limits)?,
        y: list("y", limits)?,
    })
}

fn is_keyed(k: &Value) -> bool {
    matches!(k, Value::Array(items) if items.first().is_some_and(Value::is_object))
}

/// Reads one key's value.
type ParseValue<'a, T> = &'a dyn Fn(&Value, &mut Limits) -> Result<Option<T>, LottieError>;

fn keys<T>(
    k: &Value,
    limits: &mut Limits,
    parse: ParseValue<'_, T>,
) -> Result<Vec<Key<T>>, LottieError> {
    let Value::Array(items) = k else {
        return Ok(Vec::new());
    };
    limits.keys(items.len())?;
    let mut keys = Vec::with_capacity(items.len());
    for item in items {
        let Value::Object(object) = item else {
            continue;
        };
        let Some(time) = object.get("t").and_then(Value::as_f64) else {
            continue;
        };
        let value = |key: &str, limits: &mut Limits| -> Result<Option<T>, LottieError> {
            match object.get(key) {
                Some(value) => parse(value, limits),
                None => Ok(None),
            }
        };
        let hold = match object.get("h") {
            Some(Value::Bool(flag)) => *flag,
            Some(value) => value.as_f64().is_some_and(|flag| flag != 0.0),
            None => false,
        };
        let tangent = |key: &str, limits: &mut Limits| -> Result<Option<Point>, LottieError> {
            Ok(match object.get(key) {
                Some(value) => numbers(value, limits)?.and_then(|list| two(&list)),
                None => None,
            })
        };
        let spatial = match (tangent("to", limits)?, tangent("ti", limits)?) {
            (Some(out_tangent), Some(in_tangent)) => Some((out_tangent, in_tangent)),
            _ => None,
        };
        keys.push(Key {
            time,
            start: value("s", limits)?,
            end: value("e", limits)?,
            hold,
            out_handle: handle(object.get("o"), limits)?,
            in_handle: handle(object.get("i"), limits)?,
            spatial,
        });
    }
    keys.sort_by(|a, b| a.time.total_cmp(&b.time));
    Ok(keys)
}

/// A property of numbers: `{"a": 0, "k": value}` or `{"a": 1, "k": [keys]}`.
/// Absent or unreadable is `None`.
pub(crate) fn number_prop(
    value: Option<&Value>,
    limits: &mut Limits,
) -> Result<Option<Prop<Vec<f64>>>, LottieError> {
    let Some(Value::Object(object)) = value else {
        return Ok(None);
    };
    let Some(k) = object.get("k") else {
        return Ok(None);
    };
    if is_keyed(k) {
        let keys = keys(k, limits, &|value, limits| numbers(value, limits))?;
        if keys.is_empty() {
            return Ok(None);
        }
        return Ok(Some(Prop::Animated(keys)));
    }
    Ok(numbers(k, limits)?.map(Prop::Static))
}

/// A path property. A key's value is a list holding one shape.
pub(crate) fn shape_prop(
    value: Option<&Value>,
    limits: &mut Limits,
) -> Result<Option<Prop<PathShape>>, LottieError> {
    let Some(Value::Object(object)) = value else {
        return Ok(None);
    };
    let Some(k) = object.get("k") else {
        return Ok(None);
    };
    let one = |value: &Value, limits: &mut Limits| -> Result<Option<PathShape>, LottieError> {
        match value {
            Value::Array(items) => match items.first() {
                Some(first) => PathShape::parse(first, limits),
                None => Ok(None),
            },
            other => PathShape::parse(other, limits),
        }
    };
    if is_keyed(k) && k.get(0).and_then(|key| key.get("t")).is_some() {
        let keys = keys(k, limits, &one)?;
        if keys.is_empty() {
            return Ok(None);
        }
        return Ok(Some(Prop::Animated(keys)));
    }
    Ok(one(k, limits)?.map(Prop::Static))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn prop(value: &Value) -> Prop<Vec<f64>> {
        number_prop(Some(value), &mut Limits::default())
            .unwrap()
            .unwrap()
    }

    #[test]
    fn a_static_value_is_the_same_at_every_time() {
        let value = prop(&json!({"a": 0, "k": [10, 20]}));
        assert_eq!(value.at(-5.0), Some(vec![10.0, 20.0]));
        assert_eq!(value.at(500.0), Some(vec![10.0, 20.0]));
        assert_eq!(prop(&json!({"k": 7})).scalar(3.0, 0.0), 7.0);
    }

    #[test]
    fn linear_keys_interpolate_and_hold_past_the_ends() {
        let value = prop(&json!({"a": 1, "k": [
            {"t": 0, "s": [0], "o": {"x": [0], "y": [0]}, "i": {"x": [1], "y": [1]}},
            {"t": 10, "s": [100]}
        ]}));
        assert_eq!(value.scalar(-1.0, -9.0), 0.0);
        assert!((value.scalar(5.0, -9.0) - 50.0).abs() < 1e-6);
        assert!((value.scalar(2.5, -9.0) - 25.0).abs() < 1e-6);
        assert_eq!(value.scalar(10.0, -9.0), 100.0);
        assert_eq!(value.scalar(99.0, -9.0), 100.0);
    }

    #[test]
    fn a_missing_handle_is_linear_and_the_old_end_value_is_read() {
        let value = prop(&json!({"a": 1, "k": [
            {"t": 0, "s": [0], "e": [10]},
            {"t": 4}
        ]}));
        assert!((value.scalar(1.0, -1.0) - 2.5).abs() < 1e-9);
        // After the last key, which has no start: the end before it.
        assert_eq!(value.scalar(8.0, -1.0), 10.0);
    }

    #[test]
    fn a_hold_key_jumps_at_the_next_key() {
        let value = prop(&json!({"a": 1, "k": [
            {"t": 0, "s": [1], "h": 1},
            {"t": 10, "s": [2]}
        ]}));
        assert_eq!(value.scalar(9.99, 0.0), 1.0);
        assert_eq!(value.scalar(10.0, 0.0), 2.0);
    }

    #[test]
    fn ease_in_out_is_slow_at_both_ends_and_half_way_in_the_middle() {
        let value = prop(&json!({"a": 1, "k": [
            {"t": 0, "s": [0], "o": {"x": 0.42, "y": 0}, "i": {"x": 0.58, "y": 1}},
            {"t": 100, "s": [100]}
        ]}));
        assert!((value.scalar(50.0, 0.0) - 50.0).abs() < 1e-3);
        assert!(value.scalar(10.0, 0.0) < 5.0);
        assert!(value.scalar(90.0, 0.0) > 95.0);
        // A known point of this curve: at x = 0.25, y ~ 0.1291.
        assert!((ease(0.42, 0.0, 0.58, 1.0, 0.25) - 0.1291).abs() < 1e-3);
        // Overshoot handles are kept: y may leave [0, 1].
        assert!(ease(0.3, 1.8, 0.6, 1.0, 0.5) > 1.0);
    }

    #[test]
    fn each_dimension_follows_its_own_curve() {
        let value = prop(&json!({"a": 1, "k": [
            {"t": 0, "s": [0, 0],
             "o": {"x": [0, 0.9], "y": [0, 0]}, "i": {"x": [1, 1], "y": [1, 0.1]}},
            {"t": 10, "s": [10, 10]}
        ]}));
        let mid = value.at(5.0).unwrap();
        assert!((mid[0] - 5.0).abs() < 1e-6);
        assert!(mid[1] < 2.0, "{mid:?}");
    }

    #[test]
    fn a_spatial_key_moves_along_its_curve() {
        let value = prop(&json!({"a": 1, "k": [
            {"t": 0, "s": [0, 0], "to": [0, -40], "ti": [0, -40]},
            {"t": 10, "s": [100, 0]}
        ]}));
        let mid = value.point(5.0, Point::default());
        assert!((mid.x - 50.0).abs() < 0.5, "{mid:?}");
        assert!(mid.y < -25.0, "{mid:?}");
        let straight = prop(&json!({"a": 1, "k": [
            {"t": 0, "s": [0, 0], "to": [0, 0], "ti": [0, 0]},
            {"t": 10, "s": [100, 0]}
        ]}));
        let mid = straight.point(5.0, Point::default());
        assert!((mid.x - 50.0).abs() < 1e-9 && mid.y.abs() < 1e-9);
    }

    #[test]
    fn keys_out_of_order_are_sorted_and_junk_keys_skipped() {
        let value = prop(&json!({"a": 1, "k": [
            {"t": 10, "s": [10]},
            "junk",
            {"s": [99]},
            {"t": 0, "s": [0]}
        ]}));
        assert!((value.scalar(5.0, -1.0) - 5.0).abs() < 1e-9);
        assert!(
            number_prop(
                Some(&json!({"a": 1, "k": [{"x": 1}]})),
                &mut Limits::default()
            )
            .unwrap()
            .is_none()
        );
        assert!(
            number_prop(Some(&json!("text")), &mut Limits::default())
                .unwrap()
                .is_none()
        );
    }
}
