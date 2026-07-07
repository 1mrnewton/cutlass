//! Time, color, easing, and track-kind conversions for the Python API.

use cutlass_models::Easing;
use cutlass_models::{Rational, RationalTime, TimeRange, TrackKind};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

/// Integer frame rate → exact [`Rational`] (`fps/1`), clamped to ≥ 1.
pub fn rate_from_fps(fps: u32) -> Rational {
    Rational::new(fps.max(1) as i32, 1)
}

/// Seconds → tick count at `rate`, rounded, clamped to ≥ 0.
pub fn ticks(seconds: f64, rate: Rational) -> i64 {
    if rate.den == 0 {
        return 0;
    }
    let t = seconds * f64::from(rate.num) / f64::from(rate.den);
    t.round().max(0.0) as i64
}

/// Tick count → seconds at `rate`.
pub fn seconds(tick: i64, rate: Rational) -> f64 {
    if rate.num == 0 {
        return 0.0;
    }
    tick as f64 * f64::from(rate.den) / f64::from(rate.num)
}

/// A `[start, start+duration)` timeline range from seconds (duration ≥ 1 tick).
pub fn span(start: f64, duration: f64, rate: Rational) -> TimeRange {
    let start = ticks(start, rate);
    let dur = ticks(duration, rate).max(1);
    TimeRange::at_rate(start, dur, rate)
}

/// Timeline instant from seconds.
pub fn time_at(seconds: f64, rate: Rational) -> RationalTime {
    RationalTime::new(ticks(seconds, rate), rate)
}

/// Clip-relative keyframe seconds → absolute timeline time for a clip
/// spanning `clip_start .. clip_start + clip_duration` ticks. The clip's
/// exclusive end (`rel == duration`) maps to its last animatable frame, so
/// `(duration, value)` keyframes the final frame instead of erroring.
pub fn clip_key_time(
    clip_start: i64,
    clip_duration: i64,
    rel_sec: f64,
    rate: Rational,
) -> RationalTime {
    let mut tick = ticks(rel_sec, rate);
    if tick == clip_duration && tick > 0 {
        tick -= 1;
    }
    RationalTime::new(clip_start + tick, rate)
}

/// Positive playback speed as an approximate rational.
pub fn speed_from_f64(speed: f64) -> PyResult<Rational> {
    if !speed.is_finite() || speed <= 0.0 {
        return Err(PyValueError::new_err("speed must be positive"));
    }
    let num = (speed * 1000.0).round().max(1.0) as i32;
    Ok(Rational::new(num, 1000).reduced())
}

/// Playback speed as a float multiplier.
pub fn speed_to_f64(speed: Rational) -> f64 {
    if speed.den == 0 {
        return 1.0;
    }
    f64::from(speed.num) / f64::from(speed.den)
}

/// Parse a track-kind string into a [`TrackKind`].
pub fn parse_track_kind(kind: &str) -> PyResult<TrackKind> {
    Ok(match kind.to_ascii_lowercase().as_str() {
        "video" => TrackKind::Video,
        "audio" => TrackKind::Audio,
        "text" => TrackKind::Text,
        "sticker" => TrackKind::Sticker,
        "effect" => TrackKind::Effect,
        "filter" => TrackKind::Filter,
        "adjustment" => TrackKind::Adjustment,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown track kind {other:?}"
            )));
        }
    })
}

/// Track kind → API string.
pub fn track_kind_name(kind: TrackKind) -> &'static str {
    match kind {
        TrackKind::Video => "video",
        TrackKind::Audio => "audio",
        TrackKind::Text => "text",
        TrackKind::Sticker => "sticker",
        TrackKind::Effect => "effect",
        TrackKind::Filter => "filter",
        TrackKind::Adjustment => "adjustment",
    }
}

/// [`parse_color`] with a default of opaque white when no color was given.
pub fn parse_color_opt(obj: Option<&Bound<'_, PyAny>>) -> PyResult<[u8; 4]> {
    match obj {
        Some(obj) => parse_color(obj),
        None => Ok([255, 255, 255, 255]),
    }
}

/// Parse RGB/RGBA tuple, hex string, or a small named set into `[u8; 4]`.
pub fn parse_color(obj: &Bound<'_, PyAny>) -> PyResult<[u8; 4]> {
    if let Ok(s) = obj.extract::<String>() {
        return parse_color_str(&s);
    }
    if let Ok((r, g, b)) = obj.extract::<(u8, u8, u8)>() {
        return Ok([r, g, b, 255]);
    }
    if let Ok((r, g, b, a)) = obj.extract::<(u8, u8, u8, u8)>() {
        return Ok([r, g, b, a]);
    }
    Err(PyValueError::new_err(
        "color must be (r,g,b), (r,g,b,a), a hex string, or a named color",
    ))
}

pub fn parse_color_str(s: &str) -> PyResult<[u8; 4]> {
    let lower = s.to_ascii_lowercase();
    if let Some(rgba) = named_color(&lower) {
        return Ok(rgba);
    }
    let hex = lower.strip_prefix('#').unwrap_or(&lower);
    match hex.len() {
        6 => {
            let r = u8::from_str_radix(&hex[0..2], 16)
                .map_err(|_| PyValueError::new_err(format!("invalid color {s:?}")))?;
            let g = u8::from_str_radix(&hex[2..4], 16)
                .map_err(|_| PyValueError::new_err(format!("invalid color {s:?}")))?;
            let b = u8::from_str_radix(&hex[4..6], 16)
                .map_err(|_| PyValueError::new_err(format!("invalid color {s:?}")))?;
            Ok([r, g, b, 255])
        }
        8 => {
            let r = u8::from_str_radix(&hex[0..2], 16)
                .map_err(|_| PyValueError::new_err(format!("invalid color {s:?}")))?;
            let g = u8::from_str_radix(&hex[2..4], 16)
                .map_err(|_| PyValueError::new_err(format!("invalid color {s:?}")))?;
            let b = u8::from_str_radix(&hex[4..6], 16)
                .map_err(|_| PyValueError::new_err(format!("invalid color {s:?}")))?;
            let a = u8::from_str_radix(&hex[6..8], 16)
                .map_err(|_| PyValueError::new_err(format!("invalid color {s:?}")))?;
            Ok([r, g, b, a])
        }
        _ => Err(PyValueError::new_err(format!("invalid color {s:?}"))),
    }
}

fn named_color(name: &str) -> Option<[u8; 4]> {
    Some(match name {
        "white" => [255, 255, 255, 255],
        "black" => [0, 0, 0, 255],
        "red" => [255, 0, 0, 255],
        "green" => [0, 255, 0, 255],
        "blue" => [0, 0, 255, 255],
        "transparent" => [0, 0, 0, 0],
        _ => return None,
    })
}

/// Parse easing from a string or `(x1, y1, x2, y2)` bezier tuple.
pub fn parse_easing(obj: &Bound<'_, PyAny>) -> PyResult<Easing> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(match s.to_ascii_lowercase().as_str() {
            "linear" => Easing::Linear,
            "ease_in" => Easing::EaseIn,
            "ease_out" => Easing::EaseOut,
            "ease_in_out" => Easing::EaseInOut,
            other => {
                return Err(PyValueError::new_err(format!("unknown easing {other:?}")));
            }
        });
    }
    if let Ok((x1, y1, x2, y2)) = obj.extract::<(f32, f32, f32, f32)>() {
        let easing = Easing::Bezier {
            points: [x1, y1, x2, y2],
        };
        easing
            .validate()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(easing);
    }
    Err(PyValueError::new_err(
        "easing must be 'linear', 'ease_in', 'ease_out', 'ease_in_out', or (x1,y1,x2,y2)",
    ))
}

/// Extract `(time, value[, easing])` keyframe pairs from a Python list.
pub fn parse_keyframe_pairs(
    _py: Python,
    list: &Bound<'_, PyList>,
    default_easing: Easing,
    parse_value: impl Fn(&Bound<'_, PyAny>) -> PyResult<cutlass_models::ParamValue>,
) -> PyResult<Vec<(f64, cutlass_models::ParamValue, Easing)>> {
    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let tuple = item.cast::<PyTuple>()?;
        let t: f64 = tuple.get_item(0)?.extract()?;
        let value = parse_value(&tuple.get_item(1)?)?;
        let easing = if tuple.len() >= 3 {
            parse_easing(&tuple.get_item(2)?)?
        } else {
            default_easing
        };
        out.push((t, value, easing));
    }
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    for w in out.windows(2) {
        if (w[0].0 - w[1].0).abs() < f64::EPSILON {
            return Err(PyValueError::new_err(
                "duplicate keyframe times in one call",
            ));
        }
    }
    Ok(out)
}

/// Build a catalog entry dict for IDE-friendly introspection.
pub fn param_spec_dict(
    py: Python,
    name: &str,
    label: &str,
    default: f32,
    min: f32,
    max: f32,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("name", name)?;
    dict.set_item("label", label)?;
    dict.set_item("default", default)?;
    dict.set_item("min", min)?;
    dict.set_item("max", max)?;
    Ok(dict.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_round_trip_at_30fps() {
        let rate = rate_from_fps(30);
        assert_eq!(ticks(1.0, rate), 30);
        assert!((seconds(30, rate) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ticks_rounds_and_clamps_negative() {
        let rate = rate_from_fps(30);
        assert_eq!(ticks(0.4999 / 30.0, rate), 0);
        assert_eq!(ticks(0.5001 / 30.0, rate), 1);
        assert_eq!(ticks(-5.0, rate), 0);
    }

    #[test]
    fn seconds_guards_zero_rate() {
        assert_eq!(seconds(30, Rational::new(0, 1)), 0.0);
    }

    #[test]
    fn rate_from_fps_clamps_zero() {
        assert_eq!(rate_from_fps(0), Rational::new(1, 1));
        assert_eq!(rate_from_fps(24), Rational::new(24, 1));
    }

    #[test]
    fn span_has_at_least_one_tick() {
        let rate = rate_from_fps(30);
        let range = span(1.0, 0.0, rate);
        assert_eq!(range.start.value, 30);
        assert_eq!(range.duration.value, 1);
    }

    #[test]
    fn clip_key_time_offsets_from_clip_start() {
        let rate = rate_from_fps(30);
        assert_eq!(clip_key_time(60, 60, 1.0, rate).value, 90);
    }

    #[test]
    fn clip_key_time_clamps_exclusive_end_to_last_frame() {
        let rate = rate_from_fps(30);
        // A keyframe at the clip's full duration lands on the final frame.
        assert_eq!(clip_key_time(0, 60, 2.0, rate).value, 59);
        // Past-the-end stays out of range so the model can reject it.
        assert_eq!(clip_key_time(0, 60, 2.5, rate).value, 75);
    }

    #[test]
    fn speed_round_trip() {
        let r = speed_from_f64(2.0).unwrap();
        assert!((speed_to_f64(r) - 2.0).abs() < 0.01);
        let r = speed_from_f64(0.25).unwrap();
        assert!((speed_to_f64(r) - 0.25).abs() < 0.01);
    }

    #[test]
    fn speed_rejects_non_positive_and_non_finite() {
        assert!(speed_from_f64(0.0).is_err());
        assert!(speed_from_f64(-1.0).is_err());
        assert!(speed_from_f64(f64::NAN).is_err());
        assert!(speed_from_f64(f64::INFINITY).is_err());
    }

    #[test]
    fn speed_to_f64_guards_zero_denominator() {
        assert_eq!(speed_to_f64(Rational::new(1, 0)), 1.0);
    }

    #[test]
    fn track_kind_round_trips() {
        for name in [
            "video",
            "audio",
            "text",
            "sticker",
            "effect",
            "filter",
            "adjustment",
        ] {
            let kind = parse_track_kind(name).unwrap();
            assert_eq!(track_kind_name(kind), name);
        }
        assert_eq!(
            parse_track_kind("VIDEO").unwrap(),
            TrackKind::Video,
            "kind parsing is case-insensitive"
        );
        assert!(parse_track_kind("subtitle").is_err());
    }

    #[test]
    fn color_str_hex_and_named() {
        assert_eq!(parse_color_str("#ffffff").unwrap(), [255, 255, 255, 255]);
        assert_eq!(parse_color_str("ff8000").unwrap(), [255, 128, 0, 255]);
        assert_eq!(parse_color_str("#ff8000c8").unwrap(), [255, 128, 0, 200]);
        assert_eq!(parse_color_str("WHITE").unwrap(), [255, 255, 255, 255]);
        assert_eq!(parse_color_str("transparent").unwrap(), [0, 0, 0, 0]);
        assert!(parse_color_str("#12345").is_err());
        assert!(parse_color_str("#gggggg").is_err());
        assert!(parse_color_str("chartreuse").is_err());
    }
}
