use super::*;
use crate::wire::{WireLookParam, WireShapeParam, WireStyleParam, WireTextParam};

// --- seconds → ticks ---------------------------------------------------------

pub(super) fn timeline_rate(project: &Project) -> Rational {
    project.timeline().frame_rate
}

/// Lower a wire speed multiplier to the engine's exact rational, snapped to
/// hundredths (2.0 → 2/1, 0.5 → 1/2, 0.333 → 33/100). CapCut's UI range.
pub(super) fn rational_speed(speed: f64) -> Result<Rational, Rejection> {
    if !speed.is_finite() || !(0.05..=100.0).contains(&speed) {
        return Err(Rejection::new(format!(
            "speed must be between 0.05 and 100 (got {speed})"
        )));
    }
    let num = (speed * 100.0).round() as i32;
    let g = gcd(num, 100);
    Ok(Rational::new(num / g, 100 / g))
}

fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

pub(super) fn easing(easing: Option<WireEasing>) -> Result<Easing, Rejection> {
    let easing = match easing {
        None | Some(WireEasing::Linear) => Easing::Linear,
        Some(WireEasing::EaseIn) => Easing::EaseIn,
        Some(WireEasing::EaseOut) => Easing::EaseOut,
        Some(WireEasing::EaseInOut) => Easing::EaseInOut,
        Some(WireEasing::Snappy) => Easing::from_preset_id("snappy").unwrap_or(Easing::Linear),
        Some(WireEasing::Overshoot) => {
            Easing::from_preset_id("overshoot").unwrap_or(Easing::Linear)
        }
        Some(WireEasing::Anticipate) => {
            Easing::from_preset_id("anticipate").unwrap_or(Easing::Linear)
        }
        Some(WireEasing::Hold) => Easing::Hold,
        Some(WireEasing::Bezier { points }) => Easing::Bezier { points },
    };
    easing
        .validate()
        .map_err(|e| Rejection::new(format!("invalid easing: {e}")))?;
    Ok(easing)
}

/// Build the typed parameter value from the wire's `value` / `position` /
/// `rgba` / `rect` fields, rejecting the wrong shape with a message naming
/// the right one. Effect params look up the catalog kind on `clip`.
pub(super) fn param_value(
    clip: &Clip,
    wire_clip: u64,
    param: &WireClipParam,
    value: Option<f64>,
    position: Option<[f64; 2]>,
    rgba: Option<[u8; 4]>,
    rect: Option<[f64; 4]>,
) -> Result<ParamValue, Rejection> {
    match param {
        WireClipParam::Effect { index, param: name } => {
            let (_, _, pspec) = effect_param_slot(clip, *index, name, wire_clip)?;
            match pspec.kind {
                cutlass_models::EffectParamKind::Scalar => {
                    value.map(|v| ParamValue::Scalar(v as f32)).ok_or_else(|| {
                        Rejection::new(format!(
                            "effect param '{name}' is a scalar; pass 'value' (a number)"
                        ))
                    })
                }
                cutlass_models::EffectParamKind::Vec2 => position
                    .map(|p| ParamValue::Vec2([p[0] as f32, p[1] as f32]))
                    .ok_or_else(|| {
                        Rejection::new(format!(
                            "effect param '{name}' is a vec2; pass 'position' as [x, y]"
                        ))
                    }),
                cutlass_models::EffectParamKind::Color => {
                    rgba.map(ParamValue::Color).ok_or_else(|| {
                        Rejection::new(format!(
                            "effect param '{name}' is a color; pass 'rgba' as \
                             [red, green, blue, alpha]"
                        ))
                    })
                }
            }
        }
        WireClipParam::Crop => {
            let r = rect.ok_or_else(|| {
                Rejection::new("crop param needs the 'rect' argument as [x, y, w, h]")
            })?;
            let crop = CropRect {
                x: r[0] as f32,
                y: r[1] as f32,
                w: r[2] as f32,
                h: r[3] as f32,
            };
            crop.validate().map_err(|e| Rejection::new(e.to_string()))?;
            Ok(ParamValue::Rect([crop.x, crop.y, crop.w, crop.h]))
        }
        WireClipParam::Position
        | WireClipParam::AnchorPoint
        | WireClipParam::Style {
            param: WireStyleParam::ShadowOffset,
        }
        | WireClipParam::Look {
            param: WireLookParam::MaskCenter | WireLookParam::MaskSize,
        } => position
            .map(|p| ParamValue::Vec2([p[0] as f32, p[1] as f32]))
            .ok_or_else(|| {
                Rejection::new(
                    "position, anchor_point, style shadow_offset, and look \
                     mask_center/mask_size params need the 'position' argument as [x, y]",
                )
            }),
        WireClipParam::Scale => {
            // Uniform number via `value`, or per-axis via `position: [x, y]`.
            if let Some(p) = position {
                Ok(ParamValue::Vec2([p[0] as f32, p[1] as f32]))
            } else if let Some(v) = value {
                Ok(ParamValue::Scalar(v as f32))
            } else {
                Err(Rejection::new(
                    "scale param needs 'value' (uniform number) or 'position' as [x, y]",
                ))
            }
        }
        WireClipParam::Rotation
        | WireClipParam::Opacity
        | WireClipParam::Volume
        | WireClipParam::Pan
        | WireClipParam::Speed
        | WireClipParam::Shape {
            param:
                WireShapeParam::Width
                | WireShapeParam::Height
                | WireShapeParam::CornerRadius
                | WireShapeParam::InnerRatio
                | WireShapeParam::StrokeWidth,
        }
        | WireClipParam::Text {
            param:
                WireTextParam::Size
                | WireTextParam::LetterSpacing
                | WireTextParam::LineSpacing
                | WireTextParam::StrokeWidth
                | WireTextParam::ShadowBlur
                | WireTextParam::ShadowDistance
                | WireTextParam::BackgroundRadius,
        }
        | WireClipParam::Look { .. }
        | WireClipParam::Style {
            param:
                WireStyleParam::ShadowBlur
                | WireStyleParam::GlowRadius
                | WireStyleParam::GlowIntensity
                | WireStyleParam::OutlineWidth
                | WireStyleParam::BackgroundPadding
                | WireStyleParam::BackgroundRadius,
        } => value.map(|v| ParamValue::Scalar(v as f32)).ok_or_else(|| {
            Rejection::new(
                format!("param '{param:?}' needs the 'value' argument (a number)",).to_lowercase(),
            )
        }),
        WireClipParam::Shape {
            param: WireShapeParam::Fill | WireShapeParam::StrokeColor,
        }
        | WireClipParam::Text {
            param:
                WireTextParam::Fill
                | WireTextParam::StrokeColor
                | WireTextParam::ShadowColor
                | WireTextParam::BackgroundColor,
        }
        | WireClipParam::Style {
            param:
                WireStyleParam::ShadowColor
                | WireStyleParam::GlowColor
                | WireStyleParam::OutlineColor
                | WireStyleParam::BackgroundColor,
        } => rgba.map(ParamValue::Color).ok_or_else(|| {
            Rejection::new(
                format!("param '{param:?}' needs the 'rgba' argument as [red, green, blue, alpha]",)
                    .to_lowercase(),
            )
        }),
    }
}

/// A keyframe's timeline position in seconds, pre-checked against the
/// clip's extent so the model gets a message naming where the clip sits.
pub(super) fn keyframe_position(
    project: &Project,
    clip: &Clip,
    seconds: f64,
) -> Result<RationalTime, Rejection> {
    let at = timeline_time(project, seconds, "at")?;
    let tl = clip.timeline;
    if at.value < tl.start.value || at.value >= tl.end_tick() {
        let rate = timeline_rate(project);
        return Err(Rejection::new(format!(
            "keyframe position {seconds:.3}s is outside clip {} ({:.3}s to {:.3}s)",
            clip.id.raw(),
            ticks_to_seconds(tl.start.value, rate),
            ticks_to_seconds(tl.end_tick(), rate),
        )));
    }
    Ok(at)
}

pub(super) fn ticks_to_seconds(ticks: i64, rate: Rational) -> f64 {
    ticks as f64 * rate.seconds_per_unit()
}

pub(super) fn seconds_to_ticks(seconds: f64, rate: Rational, what: &str) -> Result<i64, Rejection> {
    if !seconds.is_finite() {
        return Err(Rejection::new(format!("{what} must be a finite number")));
    }
    let ticks = seconds * f64::from(rate.num) / f64::from(rate.den);
    if !(-(2f64.powi(53))..=2f64.powi(53)).contains(&ticks) {
        return Err(Rejection::new(format!(
            "{what} of {seconds}s is out of range"
        )));
    }
    Ok(ticks.round() as i64)
}

pub(super) fn require_non_negative(seconds: f64, what: &str) -> Result<(), Rejection> {
    if seconds < 0.0 {
        return Err(Rejection::new(format!(
            "{what} must not be negative (got {seconds}s)"
        )));
    }
    Ok(())
}

/// A non-negative timeline position, frame-snapped to the project rate.
pub(super) fn timeline_time(
    project: &Project,
    seconds: f64,
    what: &str,
) -> Result<RationalTime, Rejection> {
    let ticks = seconds_to_ticks(seconds, timeline_rate(project), what)?;
    Ok(RationalTime::new(ticks, timeline_rate(project)))
}

/// A signed timeline delta, frame-snapped to the project rate.
pub(super) fn timeline_time_signed(
    project: &Project,
    seconds: f64,
    what: &str,
) -> Result<RationalTime, Rejection> {
    let ticks = seconds_to_ticks(seconds, timeline_rate(project), what)?;
    Ok(RationalTime::new(ticks, timeline_rate(project)))
}

/// A timeline range from `start`/`duration` seconds; duration must survive
/// frame snapping with at least one frame.
pub(super) fn timeline_range(
    project: &Project,
    start: f64,
    duration: f64,
) -> Result<TimeRange, Rejection> {
    require_non_negative(start, "start")?;
    if duration <= 0.0 {
        return Err(Rejection::new(format!(
            "duration must be positive (got {duration}s)"
        )));
    }
    let rate = timeline_rate(project);
    let start_ticks = seconds_to_ticks(start, rate, "start")?;
    let duration_ticks = seconds_to_ticks(duration, rate, "duration")?.max(1);
    Ok(TimeRange::at_rate(start_ticks, duration_ticks, rate))
}

/// Source range (at the media's native rate) + timeline position for
/// placing media. Pre-checks bounds so the model gets a message naming the
/// media's actual extent.
pub(super) fn media_placement(
    project: &Project,
    media: &cutlass_models::MediaSource,
    source_start: f64,
    source_duration: f64,
    timeline_seconds: f64,
    timeline_what: &str,
) -> Result<(TimeRange, RationalTime), Rejection> {
    require_non_negative(source_start, "source_start")?;
    if source_duration <= 0.0 {
        return Err(Rejection::new(format!(
            "source_duration must be positive (got {source_duration}s)"
        )));
    }
    require_non_negative(timeline_seconds, timeline_what)?;

    let rate = media.frame_rate;
    let start_ticks = seconds_to_ticks(source_start, rate, "source_start")?;
    let duration_ticks = seconds_to_ticks(source_duration, rate, "source_duration")?.max(1);
    if start_ticks + duration_ticks > media.duration.value {
        return Err(Rejection::new(format!(
            "source range {:.3}s + {:.3}s exceeds media {} which is {:.3}s long",
            source_start,
            source_duration,
            media.id.raw(),
            ticks_to_seconds(media.duration.value, rate),
        )));
    }
    let source = TimeRange::at_rate(start_ticks, duration_ticks, rate);
    let at = timeline_time(project, timeline_seconds, timeline_what)?;
    Ok((source, at))
}
