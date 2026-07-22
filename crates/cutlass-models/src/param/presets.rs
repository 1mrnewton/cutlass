//! Multi-keyframe easing presets (bounce / elastic / back).
//!
//! These expand one segment (`from` → `to`) into a short keyframe sequence
//! that approximates classic Penner-style piecewise curves. Visual
//! approximation only — verified by shape tests, not bit-identical to the
//! closed-form Penner formulas.
//!
//! Restricted to types that safely extrapolate past `t ∈ [0,1]` via
//! [`Extrapolate`] (scalars / vec2 / scale). Colors and crop are excluded.

use serde::{Deserialize, Serialize};

use super::{Easing, Extrapolate, Keyframe, Lerp};

/// Minimum segment length (ticks) required to expand a preset. Shorter
/// segments return `from`/`to` unchanged — there isn't room for the
/// intermediate extrema with a 1-tick gap.
pub const MIN_PRESET_SEGMENT_TICKS: i64 = 8;

/// Multi-keyframe easing presets: expand one segment into a keyframe
/// sequence approximating bounce / elastic / back-overshoot.
///
/// Named [`PiecewiseEasingPreset`] (not `EasingPreset`) to avoid clashing
/// with the single-bezier catalog in [`super::easing`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PiecewiseEasingPreset {
    BounceOut,
    ElasticOut,
    BackOut,
}

impl PiecewiseEasingPreset {
    /// Stable wire / UI id (`bounce_out`, …).
    pub fn id(self) -> &'static str {
        match self {
            Self::BounceOut => "bounce_out",
            Self::ElasticOut => "elastic_out",
            Self::BackOut => "back_out",
        }
    }

    /// Parse a stable id.
    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "bounce_out" => Some(Self::BounceOut),
            "elastic_out" => Some(Self::ElasticOut),
            "back_out" => Some(Self::BackOut),
            _ => None,
        }
    }
}

/// Waypoint in normalized segment space: `(t_frac, v_frac, easing)`.
/// `t_frac`/`v_frac` are fractions of the from→to tick/value delta; `v_frac`
/// may exceed `[0,1]` for overshoot (elastic / back).
type Waypoint = (f32, f32, Easing);

/// Expand `from → to` into a keyframe sequence for `preset`.
///
/// - Segment shorter than [`MIN_PRESET_SEGMENT_TICKS`]: returns `[from, to]`
///   unchanged.
/// - Ticks are rounded; collisions are nudged to keep a min 1-tick gap.
/// - Endpoint tangents (if any) are preserved on the first/last keyframes.
pub fn expand_preset<T: Lerp + Extrapolate>(
    preset: PiecewiseEasingPreset,
    from: &Keyframe<T>,
    to: &Keyframe<T>,
) -> Vec<Keyframe<T>> {
    let span = to.tick - from.tick;
    if span < MIN_PRESET_SEGMENT_TICKS {
        return vec![*from, *to];
    }
    let waypoints = match preset {
        PiecewiseEasingPreset::BounceOut => bounce_out_waypoints(),
        PiecewiseEasingPreset::ElasticOut => elastic_out_waypoints(),
        PiecewiseEasingPreset::BackOut => back_out_waypoints(),
    };
    materialize(waypoints, from, to)
}

/// Penner bounce-out extrema (visual approximation).
///
/// Landings at t≈0.3636 / 0.7272 / 0.9090 / 1.0 (v=1); valleys at
/// 0.5454 (v=0.75), 0.8181 (v=0.9375), 0.9545 (v≈0.9844). EaseOut leaves
/// landings (falling into a valley); EaseIn arrives at the next landing.
fn bounce_out_waypoints() -> &'static [Waypoint] {
    &[
        (0.0, 0.0, Easing::EaseIn),
        (0.3636, 1.0, Easing::EaseOut),
        (0.5454, 0.75, Easing::EaseIn),
        (0.7272, 1.0, Easing::EaseOut),
        (0.8181, 0.9375, Easing::EaseIn),
        (0.9090, 1.0, Easing::EaseOut),
        (0.9545, 0.984375, Easing::EaseIn),
        (1.0, 1.0, Easing::Linear),
    ]
}

/// Decaying oscillation that overshoots the target, then converges.
fn elastic_out_waypoints() -> &'static [Waypoint] {
    &[
        (0.0, 0.0, Easing::EaseInOut),
        (0.3, 1.25, Easing::EaseInOut),
        (0.55, 0.875, Easing::EaseInOut),
        (0.75, 1.0625, Easing::EaseInOut),
        (0.9, 0.96875, Easing::EaseInOut),
        (1.0, 1.0, Easing::Linear),
    ]
}

/// 10% overshoot then settle (three keyframes).
fn back_out_waypoints() -> &'static [Waypoint] {
    &[
        (0.0, 0.0, Easing::EaseOut),
        (0.6, 1.1, Easing::EaseInOut),
        (1.0, 1.0, Easing::Linear),
    ]
}

fn materialize<T: Lerp + Extrapolate>(
    waypoints: &[Waypoint],
    from: &Keyframe<T>,
    to: &Keyframe<T>,
) -> Vec<Keyframe<T>> {
    let span = (to.tick - from.tick) as f64;
    let mut ticks: Vec<i64> = waypoints
        .iter()
        .map(|(t_frac, _, _)| from.tick + (f64::from(*t_frac) * span).round() as i64)
        .collect();
    // Pin endpoints, then enforce a strict +1 tick gap.
    if let Some(first) = ticks.first_mut() {
        *first = from.tick;
    }
    if let Some(last) = ticks.last_mut() {
        *last = to.tick;
    }
    for i in 1..ticks.len() {
        if ticks[i] <= ticks[i - 1] {
            ticks[i] = ticks[i - 1] + 1;
        }
    }
    // If nudging pushed past the end, squeeze from the back.
    if let Some(last) = ticks.last().copied()
        && last > to.tick
    {
        let overflow = last - to.tick;
        for t in ticks.iter_mut() {
            // Keep the start pinned; shift later ticks back when possible.
            if *t > from.tick {
                *t = (*t - overflow).max(from.tick + 1);
            }
        }
        if let Some(last) = ticks.last_mut() {
            *last = to.tick;
        }
        for i in 1..ticks.len() {
            if ticks[i] <= ticks[i - 1] {
                ticks[i] = ticks[i - 1] + 1;
            }
        }
        if ticks.last().copied() != Some(to.tick) || ticks[0] != from.tick {
            return vec![*from, *to];
        }
    }

    waypoints
        .iter()
        .zip(ticks.iter())
        .enumerate()
        .map(|(i, ((_, v_frac, easing), tick))| {
            let value = T::lerp(from.value, to.value, *v_frac);
            let tangents = if i == 0 {
                from.tangents
            } else if i + 1 == waypoints.len() {
                to.tangents
            } else {
                None
            };
            Keyframe {
                tick: *tick,
                value,
                easing: *easing,
                tangents,
            }
        })
        .collect()
}
