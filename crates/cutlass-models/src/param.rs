//! Animatable parameters: the M2 keystone.
//!
//! A [`Param<T>`] is either a constant value or a keyframed curve. One type
//! serves every animatable property — clip transforms today; effect
//! parameters, volume envelopes, and speed ramps as later milestones land.
//!
//! Design notes:
//! - **Ticks are clip-relative.** A keyframe's `tick` is the offset from the
//!   owning clip's timeline start, at the timeline rate. Moving a clip moves
//!   its animation for free; no fix-ups on `MoveClip`/`ShiftClips`.
//! - **Compact, forward-tolerant serialization.** A constant param
//!   serializes as the bare value (`1.0`, `[0.0, 0.5]`) — byte-identical to
//!   the pre-M2 format — and a keyframed param as `{"kf":[...]}`. Old
//!   projects load unchanged; constant-only projects stay readable by old
//!   builds.
//! - **Sampling is hot-path.** `sample` is pure and allocation-free: a
//!   binary search over the keyframe slice plus an eased lerp. It runs
//!   per-layer-per-frame in `resolve_layers`.

use serde::{Deserialize, Serialize};

use crate::error::ModelError;

/// Interpolation curve for the segment *leaving* a keyframe (toward the next
/// one). The last keyframe's easing is unused until a keyframe follows it.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Easing {
    /// Constant-velocity interpolation.
    #[default]
    Linear,
    /// Accelerate from rest (quadratic).
    EaseIn,
    /// Decelerate to rest (quadratic).
    EaseOut,
    /// Accelerate then decelerate (smoothstep).
    EaseInOut,
    /// CSS-style cubic bezier: control points `(x1, y1)`, `(x2, y2)` with
    /// `x1`/`x2` in `0..=1`. `y` outside `0..=1` overshoots, like CSS.
    Bezier { points: [f32; 4] },
}

impl Easing {
    /// Map linear progress `t` in `0..=1` to eased progress.
    pub fn apply(self, t: f32) -> f32 {
        match self {
            Easing::Linear => t,
            Easing::EaseIn => t * t,
            Easing::EaseOut => t * (2.0 - t),
            Easing::EaseInOut => t * t * (3.0 - 2.0 * t),
            Easing::Bezier {
                points: [x1, y1, x2, y2],
            } => cubic_bezier(t, x1, y1, x2, y2),
        }
    }

    /// Definite integral `∫₀ᵗ apply(τ) dτ` for `t` in `0..=1`.
    ///
    /// Speed is a *rate*, so the source position swept by a keyframed speed
    /// segment is the integral of the eased curve, not the eased value
    /// itself (M2 speed ramps). The preset easings integrate in closed form;
    /// bezier falls back to Simpson's rule (smooth and monotonic over the
    /// unit interval, so a fixed step count is accurate and allocation-free).
    pub fn integral_to(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            // ∫ τ        = t²/2
            Easing::Linear => 0.5 * t * t,
            // ∫ τ²       = t³/3
            Easing::EaseIn => t * t * t / 3.0,
            // ∫ (2τ−τ²)  = t² − t³/3
            Easing::EaseOut => t * t - t * t * t / 3.0,
            // ∫ (3τ²−2τ³)= t³ − t⁴/2
            Easing::EaseInOut => {
                let t3 = t * t * t;
                t3 - 0.5 * t3 * t
            }
            Easing::Bezier {
                points: [x1, y1, x2, y2],
            } => {
                // Simpson's rule over [0, t] with an even step count.
                const STEPS: usize = 32;
                let h = t / STEPS as f32;
                if h == 0.0 {
                    return 0.0;
                }
                let f = |s: f32| cubic_bezier(s, x1, y1, x2, y2);
                let mut sum = f(0.0) + f(t);
                for i in 1..STEPS {
                    let s = h * i as f32;
                    sum += if i % 2 == 0 { 2.0 } else { 4.0 } * f(s);
                }
                sum * h / 3.0
            }
        }
    }

    /// `Ok` iff a bezier's x control points are within `0..=1` and every
    /// component is finite (an x outside the unit range makes the curve
    /// non-monotonic in time — not a function of t).
    pub fn validate(self) -> Result<(), ModelError> {
        if let Easing::Bezier { points } = self {
            if points.iter().any(|v| !v.is_finite()) {
                return Err(ModelError::InvalidParam(
                    "bezier easing has non-finite control point".into(),
                ));
            }
            let [x1, _, x2, _] = points;
            if !(0.0..=1.0).contains(&x1) || !(0.0..=1.0).contains(&x2) {
                return Err(ModelError::InvalidParam(
                    "bezier easing x control points must be in 0..=1".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Evaluate a CSS-style cubic bezier easing at progress `t`: solve the curve
/// parameter `s` where `x(s) = t` (Newton with bisection fallback), then
/// return `y(s)`. Endpoints are fixed at (0,0) and (1,1).
fn cubic_bezier(t: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }
    // Polynomial coefficients for B(s) with P0=0, P3=1.
    let (cx, bx, ax) = poly_coefficients(x1, x2);
    let (cy, by, ay) = poly_coefficients(y1, y2);
    let eval = |c: f32, b: f32, a: f32, s: f32| ((a * s + b) * s + c) * s;

    // Newton-Raphson: x(s) is monotonic for x1,x2 in 0..=1.
    let mut s = t;
    for _ in 0..8 {
        let x = eval(cx, bx, ax, s) - t;
        if x.abs() < 1e-5 {
            return eval(cy, by, ay, s);
        }
        let dx = (3.0 * ax * s + 2.0 * bx) * s + cx;
        if dx.abs() < 1e-6 {
            break;
        }
        s -= x / dx;
    }
    // Bisection fallback for flat derivatives.
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    s = t;
    for _ in 0..20 {
        let x = eval(cx, bx, ax, s);
        if (x - t).abs() < 1e-5 {
            break;
        }
        if x < t {
            lo = s;
        } else {
            hi = s;
        }
        s = 0.5 * (lo + hi);
    }
    eval(cy, by, ay, s)
}

/// Coefficients `(c, b, a)` of `B(s) = a·s³ + b·s² + c·s` for a unit bezier
/// with inner control values `p1`, `p2`.
fn poly_coefficients(p1: f32, p2: f32) -> (f32, f32, f32) {
    let c = 3.0 * p1;
    let b = 3.0 * (p2 - p1) - c;
    let a = 1.0 - c - b;
    (c, b, a)
}

/// Values a [`Param`] can animate: lerp-able, plain-old-data.
pub trait Lerp: Copy {
    fn lerp(a: Self, b: Self, t: f32) -> Self;
}

impl Lerp for f32 {
    fn lerp(a: Self, b: Self, t: f32) -> Self {
        a + (b - a) * t
    }
}

impl Lerp for [f32; 2] {
    fn lerp(a: Self, b: Self, t: f32) -> Self {
        [f32::lerp(a[0], b[0], t), f32::lerp(a[1], b[1], t)]
    }
}

/// RGBA colors (shape fill/stroke animation): per-channel lerp in encoded
/// 8-bit space — what CapCut-class editors do; a perceptual space would be
/// overkill for UI color fades.
impl Lerp for [u8; 4] {
    fn lerp(a: Self, b: Self, t: f32) -> Self {
        let mut out = [0u8; 4];
        for (o, (&x, &y)) in out.iter_mut().zip(a.iter().zip(b.iter())) {
            *o = (f32::from(x) + (f32::from(y) - f32::from(x)) * t)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
        out
    }
}

/// One point on a keyframed curve. `tick` is clip-relative (offset from the
/// clip's timeline start) at the timeline rate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Keyframe<T> {
    /// Offset from the clip's timeline start, in timeline-rate ticks.
    #[serde(rename = "t")]
    pub tick: i64,
    /// Property value at this keyframe.
    #[serde(rename = "v")]
    pub value: T,
    /// Curve of the segment leaving this keyframe.
    #[serde(rename = "e", default, skip_serializing_if = "is_linear")]
    pub easing: Easing,
}

fn is_linear(easing: &Easing) -> bool {
    *easing == Easing::Linear
}

/// An animatable property: a constant, or a keyframed curve.
///
/// Invariants when keyframed: at least one keyframe, sorted by strictly
/// increasing `tick`. Mutators preserve this; deserialization re-validates
/// through [`Param::validate_shape`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Param<T> {
    /// Keyframed curve. Serializes as `{"kf":[{"t":..,"v":..},..]}`.
    ///
    /// Listed before `Constant` so untagged deserialization tries the map
    /// shape first — a bare value can never parse as `{"kf": ...}`.
    Keyframed {
        #[serde(rename = "kf")]
        keyframes: Vec<Keyframe<T>>,
    },
    /// Fixed value. Serializes as the bare value, matching the pre-M2 format.
    Constant(T),
}

impl<T: Lerp> Param<T> {
    /// Value at a clip-relative `tick`. Before the first keyframe the first
    /// value holds; after the last, the last (CapCut behavior). Between two
    /// keyframes the segment's easing shapes the lerp.
    ///
    /// Hot path: pure, allocation-free, O(log k).
    pub fn sample(&self, tick: i64) -> T {
        self.sample_at(tick as f64)
    }

    /// [`sample`](Self::sample) at a fractional clip-relative tick. Curves
    /// are continuous in time between keyframes, so they can be evaluated
    /// between timeline frames — what export uses when the output rate
    /// exceeds the timeline rate (a 60 fps export of a 24 fps timeline
    /// samples animation at the exact output frame times instead of
    /// repeating the 24 fps values in an uneven 3-2 cadence).
    pub fn sample_at(&self, tick: f64) -> T {
        match self {
            Param::Constant(value) => *value,
            Param::Keyframed { keyframes } => {
                // Invariant: non-empty (mutators preserve it; deserialization
                // is checked through `validate_shape`).
                let first = &keyframes[0];
                if tick <= first.tick as f64 {
                    return first.value;
                }
                let last = &keyframes[keyframes.len() - 1];
                if tick >= last.tick as f64 {
                    return last.value;
                }
                // Index of the first keyframe with kf.tick > tick; the
                // segment is [idx-1, idx]. Bounds hold: first.tick < tick <
                // last.tick.
                let idx = keyframes.partition_point(|kf| (kf.tick as f64) <= tick);
                let k0 = &keyframes[idx - 1];
                let k1 = &keyframes[idx];
                let span = (k1.tick - k0.tick) as f64;
                let t = ((tick - k0.tick as f64) / span) as f32;
                T::lerp(k0.value, k1.value, k0.easing.apply(t))
            }
        }
    }
}

impl<T: Copy> Param<T> {
    /// The constant value, or `None` when keyframed.
    pub fn constant(&self) -> Option<T> {
        match self {
            Param::Constant(value) => Some(*value),
            Param::Keyframed { .. } => None,
        }
    }

    /// Insert or replace the keyframe at `tick`. A constant param becomes a
    /// single-keyframe curve.
    pub fn set_keyframe(&mut self, tick: i64, value: T, easing: Easing) {
        match self {
            Param::Constant(_) => {
                *self = Param::Keyframed {
                    keyframes: vec![Keyframe {
                        tick,
                        value,
                        easing,
                    }],
                };
            }
            Param::Keyframed { keyframes } => {
                match keyframes.binary_search_by_key(&tick, |kf| kf.tick) {
                    Ok(i) => {
                        keyframes[i] = Keyframe {
                            tick,
                            value,
                            easing,
                        }
                    }
                    Err(i) => keyframes.insert(
                        i,
                        Keyframe {
                            tick,
                            value,
                            easing,
                        },
                    ),
                }
            }
        }
    }

    /// Remove the keyframe at exactly `tick`. Removing the last keyframe
    /// collapses the param to a constant of that keyframe's value (the
    /// property keeps its on-screen value, CapCut-style). Returns `false` if
    /// no keyframe sits at `tick`.
    pub fn remove_keyframe(&mut self, tick: i64) -> bool {
        let Param::Keyframed { keyframes } = self else {
            return false;
        };
        let Ok(i) = keyframes.binary_search_by_key(&tick, |kf| kf.tick) else {
            return false;
        };
        let removed = keyframes.remove(i);
        if keyframes.is_empty() {
            *self = Param::Constant(removed.value);
        }
        true
    }

    /// Replace the param (and any keyframes) with a constant.
    pub fn set_constant(&mut self, value: T) {
        *self = Param::Constant(value);
    }
}

impl<T: Clone> Param<T> {
    /// A copy with every keyframe tick remapped by `f` (constants pass through
    /// unchanged). Used to rebase a clip-relative envelope from timeline ticks
    /// into the audio mixer's sample-frame domain once per span, so the
    /// per-sample gain lookup stays a plain tick compare. `f` must be
    /// monotonic so the remapped keyframes stay sorted.
    pub fn map_ticks(&self, f: impl Fn(i64) -> i64) -> Param<T> {
        match self {
            Param::Constant(value) => Param::Constant(value.clone()),
            Param::Keyframed { keyframes } => Param::Keyframed {
                keyframes: keyframes
                    .iter()
                    .map(|kf| Keyframe {
                        tick: f(kf.tick),
                        value: kf.value.clone(),
                        easing: kf.easing,
                    })
                    .collect(),
            },
        }
    }
}

impl<T> Param<T> {
    pub fn is_animated(&self) -> bool {
        matches!(self, Param::Keyframed { .. })
    }

    /// Keyframes in tick order; empty for a constant.
    pub fn keyframes(&self) -> &[Keyframe<T>] {
        match self {
            Param::Constant(_) => &[],
            Param::Keyframed { keyframes } => keyframes,
        }
    }

    /// Structural invariants: keyframed params are non-empty, strictly
    /// sorted by tick, with valid easings. Call after deserializing.
    pub fn validate_shape(&self) -> Result<(), ModelError> {
        let Param::Keyframed { keyframes } = self else {
            return Ok(());
        };
        if keyframes.is_empty() {
            return Err(ModelError::InvalidParam(
                "keyframed param with no keyframes".into(),
            ));
        }
        for pair in keyframes.windows(2) {
            if pair[1].tick <= pair[0].tick {
                return Err(ModelError::InvalidParam(
                    "keyframes must be strictly sorted by tick".into(),
                ));
            }
        }
        for kf in keyframes {
            kf.easing.validate()?;
        }
        Ok(())
    }

    /// Visit every stored value (constant or per-keyframe) — the hook for
    /// per-property range validation.
    pub fn for_each_value<E>(&self, mut f: impl FnMut(&T) -> Result<(), E>) -> Result<(), E> {
        match self {
            Param::Constant(value) => f(value),
            Param::Keyframed { keyframes } => {
                for kf in keyframes {
                    f(&kf.value)?;
                }
                Ok(())
            }
        }
    }
}

impl<T> From<T> for Param<T> {
    fn from(value: T) -> Self {
        Param::Constant(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kf(tick: i64, value: f32) -> Keyframe<f32> {
        Keyframe {
            tick,
            value,
            easing: Easing::Linear,
        }
    }

    // --- sampling -----------------------------------------------------------

    #[test]
    fn constant_samples_everywhere() {
        let p = Param::Constant(2.5f32);
        assert_eq!(p.sample(-100), 2.5);
        assert_eq!(p.sample(0), 2.5);
        assert_eq!(p.sample(i64::MAX), 2.5);
    }

    #[test]
    fn keyframed_clamps_outside_range() {
        let p = Param::Keyframed {
            keyframes: vec![kf(10, 1.0), kf(20, 3.0)],
        };
        assert_eq!(p.sample(0), 1.0);
        assert_eq!(p.sample(10), 1.0);
        assert_eq!(p.sample(20), 3.0);
        assert_eq!(p.sample(1000), 3.0);
    }

    #[test]
    fn linear_interpolation_between_keyframes() {
        let p = Param::Keyframed {
            keyframes: vec![kf(0, 0.0), kf(10, 10.0)],
        };
        assert_eq!(p.sample(5), 5.0);
        assert_eq!(p.sample(1), 1.0);
        assert_eq!(p.sample(9), 9.0);
    }

    #[test]
    fn single_keyframe_acts_constant() {
        let p = Param::Keyframed {
            keyframes: vec![kf(50, 7.0)],
        };
        assert_eq!(p.sample(0), 7.0);
        assert_eq!(p.sample(50), 7.0);
        assert_eq!(p.sample(100), 7.0);
    }

    #[test]
    fn multi_segment_picks_correct_pair() {
        let p = Param::Keyframed {
            keyframes: vec![kf(0, 0.0), kf(10, 100.0), kf(30, 0.0)],
        };
        assert_eq!(p.sample(5), 50.0);
        assert_eq!(p.sample(10), 100.0);
        assert_eq!(p.sample(20), 50.0);
    }

    #[test]
    fn fractional_sampling_interpolates_between_ticks() {
        let p = Param::Keyframed {
            keyframes: vec![kf(0, 0.0), kf(10, 10.0)],
        };
        // Whole ticks agree with the integer path.
        assert_eq!(p.sample_at(5.0), p.sample(5));
        // Sub-tick positions land between frame values.
        assert_eq!(p.sample_at(5.5), 5.5);
        assert_eq!(p.sample_at(0.25), 0.25);
        // Clamping matches the integer path on both sides.
        assert_eq!(p.sample_at(-3.7), 0.0);
        assert_eq!(p.sample_at(10.4), 10.0);
        assert_eq!(Param::Constant(2.5f32).sample_at(1.5), 2.5);
    }

    #[test]
    fn vec2_lerp() {
        let p = Param::Keyframed {
            keyframes: vec![
                Keyframe {
                    tick: 0,
                    value: [0.0, 0.0],
                    easing: Easing::Linear,
                },
                Keyframe {
                    tick: 10,
                    value: [1.0, -1.0],
                    easing: Easing::Linear,
                },
            ],
        };
        assert_eq!(p.sample(5), [0.5, -0.5]);
    }

    // --- easing ---------------------------------------------------------------

    #[test]
    fn easing_endpoints_are_exact() {
        for easing in [
            Easing::Linear,
            Easing::EaseIn,
            Easing::EaseOut,
            Easing::EaseInOut,
            Easing::Bezier {
                points: [0.42, 0.0, 0.58, 1.0],
            },
        ] {
            assert_eq!(easing.apply(0.0), 0.0, "{easing:?} at 0");
            assert!((easing.apply(1.0) - 1.0).abs() < 1e-4, "{easing:?} at 1");
        }
    }

    #[test]
    fn ease_in_starts_slow_ease_out_starts_fast() {
        assert!(Easing::EaseIn.apply(0.25) < 0.25);
        assert!(Easing::EaseOut.apply(0.25) > 0.25);
        let mid = Easing::EaseInOut.apply(0.5);
        assert!((mid - 0.5).abs() < 1e-6);
    }

    #[test]
    fn bezier_matches_css_ease_in_out_shape() {
        // cubic-bezier(0.42, 0, 0.58, 1) — CSS "ease-in-out".
        let e = Easing::Bezier {
            points: [0.42, 0.0, 0.58, 1.0],
        };
        assert!(e.apply(0.1) < 0.1);
        assert!(e.apply(0.9) > 0.9);
        assert!((e.apply(0.5) - 0.5).abs() < 1e-3);
        // Monotonic over a sweep.
        let mut prev = 0.0;
        for i in 0..=100 {
            let v = e.apply(i as f32 / 100.0);
            assert!(v >= prev - 1e-4, "non-monotonic at {i}");
            prev = v;
        }
    }

    #[test]
    fn easing_integrals_match_closed_form_endpoints() {
        // ∫₀¹ of each easing over the unit interval.
        assert!((Easing::Linear.integral_to(1.0) - 0.5).abs() < 1e-6);
        assert!((Easing::EaseIn.integral_to(1.0) - 1.0 / 3.0).abs() < 1e-6);
        assert!((Easing::EaseOut.integral_to(1.0) - 2.0 / 3.0).abs() < 1e-6);
        assert!((Easing::EaseInOut.integral_to(1.0) - 0.5).abs() < 1e-6);
        // The symmetric CSS ease-in-out bezier integrates to ½ by symmetry.
        let e = Easing::Bezier {
            points: [0.42, 0.0, 0.58, 1.0],
        };
        assert!((e.integral_to(1.0) - 0.5).abs() < 1e-3);
        // Integral is 0 at t=0 and monotonic increasing.
        for easing in [
            Easing::Linear,
            Easing::EaseIn,
            Easing::EaseOut,
            Easing::EaseInOut,
        ] {
            assert_eq!(easing.integral_to(0.0), 0.0);
            let mut prev = 0.0;
            for i in 0..=20 {
                let v = easing.integral_to(i as f32 / 20.0);
                assert!(v >= prev - 1e-6, "{easing:?} non-monotonic integral");
                prev = v;
            }
        }
    }

    #[test]
    fn bezier_validation_rejects_bad_x() {
        assert!(
            Easing::Bezier {
                points: [1.5, 0.0, 0.5, 1.0]
            }
            .validate()
            .is_err()
        );
        assert!(
            Easing::Bezier {
                points: [0.5, 0.0, -0.1, 1.0]
            }
            .validate()
            .is_err()
        );
        assert!(
            Easing::Bezier {
                points: [0.5, f32::NAN, 0.5, 1.0]
            }
            .validate()
            .is_err()
        );
        // Overshooting y is allowed (CSS semantics).
        assert!(
            Easing::Bezier {
                points: [0.3, -0.5, 0.7, 1.5]
            }
            .validate()
            .is_ok()
        );
    }

    // --- mutation ---------------------------------------------------------------

    #[test]
    fn set_keyframe_on_constant_becomes_curve() {
        let mut p = Param::Constant(1.0f32);
        p.set_keyframe(10, 2.0, Easing::Linear);
        assert!(p.is_animated());
        assert_eq!(p.keyframes().len(), 1);
        assert_eq!(p.sample(10), 2.0);
    }

    #[test]
    fn set_keyframe_inserts_sorted_and_replaces() {
        let mut p = Param::Constant(0.0f32);
        p.set_keyframe(20, 2.0, Easing::Linear);
        p.set_keyframe(0, 0.0, Easing::Linear);
        p.set_keyframe(10, 1.0, Easing::Linear);
        let ticks: Vec<i64> = p.keyframes().iter().map(|k| k.tick).collect();
        assert_eq!(ticks, vec![0, 10, 20]);

        p.set_keyframe(10, 5.0, Easing::EaseIn);
        assert_eq!(p.keyframes().len(), 3);
        assert_eq!(p.sample(10), 5.0);
    }

    #[test]
    fn remove_keyframe_collapses_last_to_constant() {
        let mut p = Param::Constant(0.0f32);
        p.set_keyframe(10, 7.0, Easing::Linear);
        assert!(!p.remove_keyframe(5), "no keyframe at 5");
        assert!(p.remove_keyframe(10));
        assert!(!p.is_animated());
        assert_eq!(p.constant(), Some(7.0));
    }

    #[test]
    fn set_constant_wipes_keyframes() {
        let mut p = Param::Constant(0.0f32);
        p.set_keyframe(10, 1.0, Easing::Linear);
        p.set_keyframe(20, 2.0, Easing::Linear);
        p.set_constant(9.0);
        assert_eq!(p.constant(), Some(9.0));
        assert!(p.keyframes().is_empty());
    }

    // --- serde -----------------------------------------------------------------

    #[test]
    fn constant_serializes_as_bare_value() {
        let p = Param::Constant(1.5f32);
        assert_eq!(serde_json::to_string(&p).unwrap(), "1.5");
        let v: Param<[f32; 2]> = Param::Constant([0.0, 0.25]);
        assert_eq!(serde_json::to_string(&v).unwrap(), "[0.0,0.25]");
    }

    #[test]
    fn bare_value_deserializes_as_constant() {
        let p: Param<f32> = serde_json::from_str("2.0").unwrap();
        assert_eq!(p, Param::Constant(2.0));
        let v: Param<[f32; 2]> = serde_json::from_str("[0.1,0.2]").unwrap();
        assert_eq!(v, Param::Constant([0.1, 0.2]));
    }

    #[test]
    fn keyframed_roundtrips_compactly() {
        let p = Param::Keyframed {
            keyframes: vec![
                Keyframe {
                    tick: 0,
                    value: 1.0f32,
                    easing: Easing::Linear,
                },
                Keyframe {
                    tick: 24,
                    value: 2.0,
                    easing: Easing::EaseInOut,
                },
            ],
        };
        let json = serde_json::to_string(&p).unwrap();
        // Linear easing is elided; non-linear spelled out.
        assert_eq!(
            json,
            r#"{"kf":[{"t":0,"v":1.0},{"t":24,"v":2.0,"e":"ease_in_out"}]}"#
        );
        let back: Param<f32> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn bezier_easing_roundtrips() {
        let p = Param::Keyframed {
            keyframes: vec![
                Keyframe {
                    tick: 0,
                    value: 0.0f32,
                    easing: Easing::Bezier {
                        points: [0.42, 0.0, 0.58, 1.0],
                    },
                },
                Keyframe {
                    tick: 10,
                    value: 1.0,
                    easing: Easing::Linear,
                },
            ],
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: Param<f32> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn vec2_keyframed_roundtrips() {
        let p = Param::Keyframed {
            keyframes: vec![
                Keyframe {
                    tick: 0,
                    value: [0.0f32, 0.0],
                    easing: Easing::Linear,
                },
                Keyframe {
                    tick: 48,
                    value: [0.5, -0.5],
                    easing: Easing::EaseOut,
                },
            ],
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: Param<[f32; 2]> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
    }

    // --- validation -------------------------------------------------------------

    #[test]
    fn validate_shape_rejects_unsorted_and_empty() {
        let unsorted: Param<f32> = Param::Keyframed {
            keyframes: vec![kf(10, 1.0), kf(5, 2.0)],
        };
        assert!(unsorted.validate_shape().is_err());

        let dup: Param<f32> = Param::Keyframed {
            keyframes: vec![kf(10, 1.0), kf(10, 2.0)],
        };
        assert!(dup.validate_shape().is_err());

        let empty: Param<f32> = Param::Keyframed { keyframes: vec![] };
        assert!(empty.validate_shape().is_err());

        let ok: Param<f32> = Param::Keyframed {
            keyframes: vec![kf(0, 1.0), kf(10, 2.0)],
        };
        assert!(ok.validate_shape().is_ok());
        assert!(Param::Constant(1.0f32).validate_shape().is_ok());
    }
}
