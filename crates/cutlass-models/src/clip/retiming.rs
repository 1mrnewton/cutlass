use crate::error::ModelError;
use crate::ids::MediaId;
use crate::param::{Easing, Keyframe, Param};
use crate::time::{RationalTime, TimeRange, resample, time_sub};

use super::Clip;
use super::generator::ClipSource;
use super::{is_unit_speed, is_unit_speed_curve};

/// Normalized tick span of a [`Clip::speed_curve`]: keyframe tick `0` is the
/// clip's start, [`SPEED_CURVE_SCALE`] its end. The ramp is stored over this
/// fixed domain (not absolute clip ticks) so its shape survives trims and
/// base-speed changes that re-derive the clip's timeline duration.
pub const SPEED_CURVE_SCALE: i64 = 1000;

/// Slowest instantaneous speed multiplier a ramp keyframe may hold (matches
/// the agent's `set_clip_speed` floor). A positive floor keeps the curve's
/// average — and thus the derived duration — finite.
pub const MIN_SPEED: f32 = 0.05;

/// Fastest instantaneous speed multiplier a ramp keyframe may hold.
pub const MAX_SPEED: f32 = 100.0;

impl Clip {
    /// True iff the clip plays at anything but forward 1× — the audio mixers
    /// time-stretch retimed clips (M8 Phase 3) and the UI badges them. A
    /// non-flat speed ramp counts (M2 speed curves).
    pub fn is_retimed(&self) -> bool {
        !is_unit_speed(&self.speed) || self.reversed || self.has_speed_curve()
    }

    /// True iff the clip carries a non-flat playback-rate ramp (M2 speed
    /// curves) — the constant `1.0` default does not.
    pub fn has_speed_curve(&self) -> bool {
        !is_unit_speed_curve(&self.speed_curve)
    }

    /// Frequency multiplier the varispeed renderer (M8 Phase 3) applies to a
    /// retimed clip's audio: `1.0` when pitch is locked (the CapCut default,
    /// time-stretch preserves pitch), else the clip's overall playback-speed
    /// ratio (`base speed × ramp average`) so pitch rides the speed — the
    /// optional "chipmunk" mode. Reverse does not change pitch.
    pub fn audio_pitch_factor(&self) -> f32 {
        if self.preserve_pitch {
            1.0
        } else {
            let base = f64::from(self.speed.num) / f64::from(self.speed.den);
            (base * self.speed_curve_average()) as f32
        }
    }

    /// `∫₀ᵖ speed_curve(q) dq` over the normalized clip span, `p` in `0..=1`
    /// (`0` = clip start, `1` = clip end). The speed curve is a *rate*, so
    /// this cumulative integral — not the sampled value — is what maps a
    /// timeline position to a fraction of the source window. Pure and
    /// allocation-free; `O(keyframes)` (a handful of ramp points).
    pub fn speed_curve_integral(&self, p: f64) -> f64 {
        speed_curve_integral(&self.speed_curve, p)
    }

    /// Average instantaneous multiplier of the speed ramp over the whole clip
    /// (`speed_curve_integral(1.0)`). The clip's timeline duration derives
    /// from `source_duration ÷ (base_speed × this)`.
    pub fn speed_curve_average(&self) -> f64 {
        self.speed_curve_integral(1.0)
    }

    /// Source ticks consumed by `tl_ticks` timeline ticks at this clip's
    /// speed (both in the same rate; exact rational scale, truncating).
    pub fn scale_by_speed(&self, tl_ticks: i64) -> i64 {
        tl_ticks * i64::from(self.speed.num) / i64::from(self.speed.den)
    }

    /// Timeline ticks covered by `src_ticks` source ticks at this clip's
    /// speed (the inverse of [`Self::scale_by_speed`], truncating).
    pub fn unscale_by_speed(&self, src_ticks: i64) -> i64 {
        src_ticks * i64::from(self.speed.den) / i64::from(self.speed.num)
    }

    /// Clip-relative animation tick for an absolute timeline position: the
    /// offset from the clip's start. Positions outside the clip clamp into
    /// `[0, duration)` so callers sampling at a stale playhead still get the
    /// nearest in-range value.
    pub fn animation_tick(&self, timeline_tick: i64) -> i64 {
        let offset = timeline_tick - self.timeline.start.value;
        offset.clamp(0, (self.timeline.duration.value - 1).max(0))
    }

    /// [`animation_tick`](Self::animation_tick) for a fractional timeline
    /// position — sub-frame export sampling. Clamps into the same
    /// `[0, duration - 1]` range, so the last frame's value holds through
    /// any trailing output frames.
    pub fn animation_tick_f(&self, timeline_tick: f64) -> f64 {
        let offset = timeline_tick - self.timeline.start.value as f64;
        offset.clamp(0.0, (self.timeline.duration.value - 1).max(0) as f64)
    }

    /// Timeline start position.
    pub fn start(&self) -> RationalTime {
        self.timeline.start
    }

    /// Exclusive timeline end.
    pub fn end(&self) -> Result<RationalTime, ModelError> {
        self.timeline.end().map_err(Into::into)
    }

    /// The media this clip references, or `None` for generated content.
    pub fn media(&self) -> Option<MediaId> {
        match &self.content {
            ClipSource::Media { media, .. } => Some(*media),
            ClipSource::Generated(_) => None,
        }
    }

    /// The source in/out range, or `None` for generated content.
    pub fn source_range(&self) -> Option<TimeRange> {
        match &self.content {
            ClipSource::Media { source, .. } => Some(*source),
            ClipSource::Generated(_) => None,
        }
    }

    pub fn is_generated(&self) -> bool {
        matches!(self.content, ClipSource::Generated(_))
    }

    /// Map a timeline position to the corresponding source time, for media
    /// clips. Honors the clip's retiming: without a ramp the timeline offset
    /// scales by `speed` (exact rational math); with a [`Self::speed_curve`]
    /// the source offset is the curve's cumulative integral (speed is a
    /// rate). `reversed` walks the source window backward from its end. The
    /// result clamps into the source window so duration rounding can never
    /// read past an edge.
    pub fn source_time_at(
        &self,
        timeline_pos: RationalTime,
    ) -> Result<Option<RationalTime>, ModelError> {
        if !self.timeline.contains(timeline_pos)? {
            return Ok(None);
        }
        match &self.content {
            ClipSource::Media { source, .. } => {
                if self.freeze_frame {
                    return Ok(Some(source.start));
                }
                let offset_tl = time_sub(&timeline_pos, &self.timeline.start)?;
                let first = source.start.value;
                let last = first + (source.duration.value - 1).max(0);
                let offset_src = if self.has_speed_curve() {
                    // Speed is a rate: the fraction of the source window swept
                    // by clip-relative position `p` is `∫₀ᵖ curve ÷ ∫₀¹ curve`.
                    // base_speed and the derived duration cancel in the ratio
                    // (the duration was derived to consume the window exactly),
                    // so the curve *shape* alone places the source frame.
                    let dur = self.timeline.duration.value.max(1) as f64;
                    let p = offset_tl.value as f64 / dur;
                    let total = self.speed_curve_average();
                    let ratio = if total > 0.0 {
                        self.speed_curve_integral(p) / total
                    } else {
                        p
                    };
                    (source.duration.value as f64 * ratio).round() as i64
                } else {
                    // Flat ramp: the exact rational fast path (zero f64 drift),
                    // identical to M1 constant speed.
                    let scaled =
                        RationalTime::new(self.scale_by_speed(offset_tl.value), offset_tl.rate);
                    resample(scaled, source.start.rate).value
                };
                let tick = if self.reversed {
                    last - offset_src
                } else {
                    first + offset_src
                };
                Ok(Some(RationalTime::new(
                    tick.clamp(first, last),
                    source.start.rate,
                )))
            }
            ClipSource::Generated(_) => Ok(None),
        }
    }
}

/// `∫₀ᵖ curve(q) dq` over the normalized clip span, `p` in `0..=1` (`0` = clip
/// start, `1` = clip end). Free-function core of [`Clip::speed_curve_integral`]
/// so the audio mixers can evaluate a clip's ramp without a whole [`Clip`].
/// Pure and allocation-free; `O(keyframes)`.
pub fn speed_curve_integral(curve: &Param<f32>, p: f64) -> f64 {
    let p = p.clamp(0.0, 1.0);
    match curve {
        Param::Constant(v) => f64::from(*v) * p,
        Param::Keyframed { keyframes } => {
            let scale = SPEED_CURVE_SCALE as f64;
            let pos = |kf: &Keyframe<f32>| kf.tick as f64 / scale;
            let first = &keyframes[0];
            let q0 = pos(first);
            // Leading flat region holds the first value (CapCut clamp).
            let mut acc = f64::from(first.value) * p.min(q0);
            if p <= q0 {
                return acc;
            }
            for pair in keyframes.windows(2) {
                let (k0, k1) = (&pair[0], &pair[1]);
                let (qa, qb) = (pos(k0), pos(k1));
                if p <= qa {
                    return acc;
                }
                let seg = qb - qa;
                if seg > 0.0 {
                    let upper = p.min(qb);
                    let t_hi = ((upper - qa) / seg) as f32;
                    let (va, vb) = (f64::from(k0.value), f64::from(k1.value));
                    // ∫ over [qa, upper] of lerp(va, vb, e(t)) dq, dq = seg·dt
                    //   = seg·[va·t_hi + (vb − va)·∫₀^{t_hi} e].
                    let e_int = f64::from(k0.easing.integral_to(t_hi));
                    acc += seg * (va * f64::from(t_hi) + (vb - va) * e_int);
                }
                if p <= qb {
                    return acc;
                }
            }
            // Trailing flat region holds the last value.
            let last = &keyframes[keyframes.len() - 1];
            acc + f64::from(last.value) * (p - pos(last))
        }
    }
}

/// Fraction of a clip's source window swept by clip-relative output position
/// `p` in `0..=1`: the normalized cumulative integral `∫₀ᵖ curve ÷ ∫₀¹ curve`.
/// Monotonic with `0 ↦ 0` and `1 ↦ 1`, this is exactly the source placement
/// [`Clip::source_time_at`] uses for video, so the varispeed audio renderer
/// (M8 Phase 3) warps the sound in lockstep with the picture. A degenerate
/// (zero-area) curve falls back to a linear map.
pub fn speed_curve_source_fraction(curve: &Param<f32>, p: f64) -> f64 {
    let total = speed_curve_integral(curve, 1.0);
    if total > 0.0 {
        (speed_curve_integral(curve, p) / total).clamp(0.0, 1.0)
    } else {
        p.clamp(0.0, 1.0)
    }
}

/// Divide a normalized speed curve at timeline fraction `split`, stretching
/// each restricted domain back onto `0..=`[`SPEED_CURVE_SCALE`].
///
/// The values are instantaneous rates, so preserving the original function on
/// each subdomain preserves its cumulative integral (and therefore
/// [`Clip::source_time_at`]) after the source window is divided at the same
/// point. Boundary easings are subdivided exactly; the fixed integer domain
/// can still make an original keyframe unrepresentable when it lands less than
/// one normalized tick from the cut, in which case splitting fails closed.
pub(crate) fn split_speed_curve(
    curve: &Param<f32>,
    split: f64,
) -> Result<(Param<f32>, Param<f32>), ModelError> {
    if !split.is_finite() || !(0.0..1.0).contains(&split) {
        return Err(ModelError::InvalidParam(
            "speed-ramp split must lie strictly inside the clip".into(),
        ));
    }
    match curve {
        Param::Constant(value) => Ok((Param::Constant(*value), Param::Constant(*value))),
        Param::Keyframed { .. } => Ok((
            speed_curve_subsegment(curve, 0.0, split)?,
            speed_curve_subsegment(curve, split, 1.0)?,
        )),
    }
}

fn speed_curve_subsegment(
    curve: &Param<f32>,
    start: f64,
    end: f64,
) -> Result<Param<f32>, ModelError> {
    let scale = SPEED_CURVE_SCALE as f64;
    let start_tick = start * scale;
    let end_tick = end * scale;
    let mut positions = Vec::with_capacity(curve.keyframes().len() + 2);
    positions.push(start_tick);
    positions.extend(
        curve
            .keyframes()
            .iter()
            .map(|kf| kf.tick as f64)
            .filter(|tick| *tick > start_tick && *tick < end_tick),
    );
    positions.push(end_tick);

    let mut keyframes = Vec::with_capacity(positions.len());
    for (index, &position) in positions.iter().enumerate() {
        let tick = if index == 0 {
            0
        } else if index + 1 == positions.len() {
            SPEED_CURVE_SCALE
        } else {
            (((position - start_tick) / (end_tick - start_tick)) * scale).round() as i64
        };
        if keyframes
            .last()
            .is_some_and(|previous: &Keyframe<f32>| previous.tick >= tick)
        {
            return Err(ModelError::InvalidParam(
                "speed-ramp keyframe is too close to the split boundary".into(),
            ));
        }
        let easing = positions
            .get(index + 1)
            .map_or(Ok(Easing::Linear), |&next| {
                speed_curve_interval_easing(curve, position, next)
            })?;
        keyframes.push(Keyframe {
            tick,
            value: curve.sample_at(position),
            easing,
            tangents: None,
        });
    }

    let result = Param::Keyframed { keyframes };
    validate_speed_curve(&result)?;
    Ok(result)
}

/// Easing for one interval that is wholly inside an original curve segment
/// (all original keyframe positions were inserted into the interval list).
fn speed_curve_interval_easing(
    curve: &Param<f32>,
    from: f64,
    to: f64,
) -> Result<Easing, ModelError> {
    let keyframes = curve.keyframes();
    let first = &keyframes[0];
    let last = &keyframes[keyframes.len() - 1];
    if from < first.tick as f64 || from >= last.tick as f64 {
        return Ok(Easing::Linear);
    }

    let upper = keyframes.partition_point(|kf| kf.tick as f64 <= from);
    let lower = upper.saturating_sub(1);
    let Some(next) = keyframes.get(lower + 1) else {
        return Ok(Easing::Linear);
    };
    let current = &keyframes[lower];
    if current.value == next.value {
        return Ok(Easing::Linear);
    }
    if to > next.tick as f64 + f64::EPSILON {
        return Err(ModelError::InvalidParam(
            "speed-ramp split crossed an untracked keyframe".into(),
        ));
    }

    let span = (next.tick - current.tick) as f64;
    let local_from = ((from - current.tick as f64) / span).clamp(0.0, 1.0) as f32;
    let local_to = ((to - current.tick as f64) / span).clamp(0.0, 1.0) as f32;
    current.easing.subsegment(local_from, local_to)
}

/// Validate a speed ramp (M2 speed curves) before it is stored: a structurally
/// sound `Param` (sorted, non-empty, valid easings) whose every keyframe value
/// is finite and within `[`[`MIN_SPEED`]`, `[`MAX_SPEED`]`]`, with normalized
/// ticks inside `0..=`[`SPEED_CURVE_SCALE`].
pub fn validate_speed_curve(curve: &Param<f32>) -> Result<(), ModelError> {
    curve.validate_shape()?;
    for kf in curve.keyframes() {
        if kf.tick < 0 || kf.tick > SPEED_CURVE_SCALE {
            return Err(ModelError::InvalidParam(format!(
                "speed ramp keyframe tick {} is outside 0..={SPEED_CURVE_SCALE}",
                kf.tick
            )));
        }
    }
    curve.for_each_value(|v| {
        if !v.is_finite() || !(MIN_SPEED..=MAX_SPEED).contains(v) {
            return Err(ModelError::InvalidParam(format!(
                "speed ramp value {v} must be within {MIN_SPEED}..={MAX_SPEED}"
            )));
        }
        Ok(())
    })
}

/// One speed-ramp preset catalog entry (id + display label; the curve comes
/// from [`speed_preset`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeedPresetSpec {
    pub id: &'static str,
    pub label: &'static str,
}

const SPEED_PRESETS: &[SpeedPresetSpec] = &[
    SpeedPresetSpec {
        id: "ramp_up",
        label: "Ramp up",
    },
    SpeedPresetSpec {
        id: "ramp_down",
        label: "Ramp down",
    },
    SpeedPresetSpec {
        id: "montage",
        label: "Montage",
    },
    SpeedPresetSpec {
        id: "hero",
        label: "Hero",
    },
    SpeedPresetSpec {
        id: "bullet",
        label: "Bullet",
    },
    SpeedPresetSpec {
        id: "jump_cut",
        label: "Jump cut",
    },
    SpeedPresetSpec {
        id: "flash_in",
        label: "Flash in",
    },
    SpeedPresetSpec {
        id: "flash_out",
        label: "Flash out",
    },
];

/// Every speed-ramp preset (UI browsing order). Each id resolves through
/// [`speed_preset`]; the drift is locked by a test.
pub fn speed_preset_catalog() -> &'static [SpeedPresetSpec] {
    SPEED_PRESETS
}

/// The catalog id whose curve equals `curve`, or `None` for a flat ramp or a
/// hand-edited curve. How the shells highlight the active preset tile: curves
/// are normalized over `0..=`[`SPEED_CURVE_SCALE`], so a preset's shape (and
/// this match) survives trims and base-speed changes.
pub fn speed_preset_id(curve: &Param<f32>) -> Option<&'static str> {
    SPEED_PRESETS
        .iter()
        .find(|spec| speed_preset(spec.id).as_ref() == Some(curve))
        .map(|spec| spec.id)
}

/// Built-in speed-ramp presets (M2 speed curves, "presets as data"). Each is
/// a normalized [`Param`] over `0..=`[`SPEED_CURVE_SCALE`] of multipliers on
/// the clip's base speed. Shared by the inspector buttons, the mobile speed
/// panel, and the agent's `set_speed_curve` tool. Returns `None` for an
/// unknown name; the ids are cataloged in [`speed_preset_catalog`].
pub fn speed_preset(name: &str) -> Option<Param<f32>> {
    let s = SPEED_CURVE_SCALE;
    let kf = |frac: f64, value: f32, easing: Easing| Keyframe {
        tick: (frac * s as f64).round() as i64,
        value,
        easing,
        tangents: None,
    };
    let keyframes = match name {
        // Accelerate from slow-mo into fast (CapCut "speed up").
        "ramp_up" => vec![kf(0.0, 0.4, Easing::EaseIn), kf(1.0, 2.5, Easing::Linear)],
        // Decelerate from fast into slow-mo (CapCut "slow down").
        "ramp_down" => vec![kf(0.0, 2.5, Easing::EaseOut), kf(1.0, 0.4, Easing::Linear)],
        // Fast / slow / fast cuts — montage energy.
        "montage" => vec![
            kf(0.0, 2.0, Easing::EaseInOut),
            kf(0.5, 0.5, Easing::EaseInOut),
            kf(1.0, 2.0, Easing::Linear),
        ],
        // Normal, dip to slow-mo on the action, back to normal — "hero moment".
        "hero" => vec![
            kf(0.0, 1.5, Easing::EaseInOut),
            kf(0.5, 0.3, Easing::EaseInOut),
            kf(1.0, 1.5, Easing::Linear),
        ],
        // Punchy fast / hard slow / fast — "bullet time".
        "bullet" => vec![
            kf(0.0, 3.0, Easing::EaseInOut),
            kf(0.4, 0.25, Easing::EaseInOut),
            kf(0.6, 0.25, Easing::EaseInOut),
            kf(1.0, 3.0, Easing::Linear),
        ],
        // Alternating normal / triple-speed bursts — jump-cut energy.
        "jump_cut" => vec![
            kf(0.0, 1.0, Easing::Linear),
            kf(0.25, 3.0, Easing::Linear),
            kf(0.5, 1.0, Easing::Linear),
            kf(0.75, 3.0, Easing::Linear),
            kf(1.0, 1.0, Easing::Linear),
        ],
        // Blast in fast, settle to normal.
        "flash_in" => vec![
            kf(0.0, 3.0, Easing::EaseOut),
            kf(0.3, 1.0, Easing::Linear),
            kf(1.0, 1.0, Easing::Linear),
        ],
        // Hold normal, accelerate out.
        "flash_out" => vec![
            kf(0.0, 1.0, Easing::Linear),
            kf(0.7, 1.0, Easing::EaseIn),
            kf(1.0, 3.0, Easing::Linear),
        ],
        _ => return None,
    };
    Some(Param::Keyframed { keyframes })
}
