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
//! - **Spatial tangents** (`Keyframe::tangents`) are a serde slot on every
//!   keyframe type for schema simplicity, but only `[f32; 2]` sampling reads
//!   them (cubic bezier motion paths). The project routing layer rejects
//!   tangents on non-position params.

use serde::{Deserialize, Serialize};

use crate::error::ModelError;

mod easing;

pub use easing::{EASING_PRESETS, Easing, EasingPreset, easing_preset};

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

/// Segment interpolation with access to the bounding keyframes; the default
/// forwards to plain [`Lerp`]. `[f32; 2]` overrides it to follow cubic bezier
/// motion paths when spatial tangents are present.
///
/// Temporal easing controls SPEED along the path; spatial tangents control
/// SHAPE (After Effects semantics). Both tangents zero/`None` → exact legacy
/// straight-line lerp (early-return, bit-identical).
///
/// Public because it bounds [`Param::sample`]; outside this crate, implement
/// the default (`impl SegmentSample for T {}`) for any custom [`Lerp`] type.
pub trait SegmentSample: Lerp {
    fn segment_sample(a: &Keyframe<Self>, b: &Keyframe<Self>, eased_t: f32) -> Self {
        Lerp::lerp(a.value, b.value, eased_t)
    }
}

impl SegmentSample for f32 {}
impl SegmentSample for [u8; 4] {}

impl SegmentSample for [f32; 2] {
    fn segment_sample(a: &Keyframe<Self>, b: &Keyframe<Self>, eased_t: f32) -> Self {
        let out_t = a.tangents.map(|t| t.out_t).unwrap_or([0.0, 0.0]);
        let in_t = b.tangents.map(|t| t.in_t).unwrap_or([0.0, 0.0]);
        if is_zero_vec2(&out_t) && is_zero_vec2(&in_t) {
            return Lerp::lerp(a.value, b.value, eased_t);
        }
        let p0 = a.value;
        let p1 = [p0[0] + out_t[0], p0[1] + out_t[1]];
        let p3 = b.value;
        let p2 = [p3[0] + in_t[0], p3[1] + in_t[1]];
        let u = arc_length_parameter(p0, p1, p2, p3, eased_t.clamp(0.0, 1.0));
        eval_cubic_bezier2(p0, p1, p2, p3, u)
    }
}

/// Spatial bezier handles for 2-d position segments (motion paths). Offsets
/// are relative to the keyframe's value, in the same units as the value.
/// `out_t` shapes the segment LEAVING this keyframe, `in_t` the segment
/// ARRIVING at it. `None` ⇔ straight-line (legacy) motion.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpatialTangents {
    #[serde(rename = "o", default, skip_serializing_if = "is_zero_vec2")]
    pub out_t: [f32; 2],
    #[serde(rename = "i", default, skip_serializing_if = "is_zero_vec2")]
    pub in_t: [f32; 2],
}

impl SpatialTangents {
    /// Both handles at the origin — equivalent to a missing `tangents` slot
    /// for sampling, but distinct in serde when explicitly stored.
    pub const ZERO: Self = Self {
        out_t: [0.0, 0.0],
        in_t: [0.0, 0.0],
    };

    /// `Ok` when every component is finite and within ±4 canvas fractions
    /// (generous for motion-path handles).
    pub fn validate(self) -> Result<(), ModelError> {
        for c in [self.out_t[0], self.out_t[1], self.in_t[0], self.in_t[1]] {
            if !c.is_finite() || c.abs() > 4.0 {
                return Err(ModelError::InvalidParam(format!(
                    "spatial tangent component {c} must be finite and within ±4.0"
                )));
            }
        }
        Ok(())
    }
}

fn is_zero_vec2(v: &[f32; 2]) -> bool {
    v[0] == 0.0 && v[1] == 0.0
}

/// Evaluate a 2-d cubic bezier at parameter `u`.
fn eval_cubic_bezier2(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], p3: [f32; 2], u: f32) -> [f32; 2] {
    let u2 = u * u;
    let u3 = u2 * u;
    let mu = 1.0 - u;
    let mu2 = mu * mu;
    let mu3 = mu2 * mu;
    [
        mu3 * p0[0] + 3.0 * mu2 * u * p1[0] + 3.0 * mu * u2 * p2[0] + u3 * p3[0],
        mu3 * p0[1] + 3.0 * mu2 * u * p1[1] + 3.0 * mu * u2 * p2[1] + u3 * p3[1],
    ]
}

/// Map eased progress `t` in `0..=1` to a cubic parameter `u` so that equal
/// `t` steps travel roughly equal arc length along the curve.
///
/// Builds a 17-point cumulative chord-length table (16 segments, stack array,
/// no alloc). Acceptable cost: 17 bezier evals per curved segment per sample
/// call — motion paths are rare relative to straight segments.
fn arc_length_parameter(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], p3: [f32; 2], t: f32) -> f32 {
    const LUT_SIZE: usize = 17;
    let mut cumlen = [0.0f32; LUT_SIZE];
    let mut prev = p0;
    for i in 1..LUT_SIZE {
        let u = i as f32 / (LUT_SIZE - 1) as f32;
        let pt = eval_cubic_bezier2(p0, p1, p2, p3, u);
        let dx = pt[0] - prev[0];
        let dy = pt[1] - prev[1];
        cumlen[i] = cumlen[i - 1] + (dx * dx + dy * dy).sqrt();
        prev = pt;
    }
    let total = cumlen[LUT_SIZE - 1];
    if total <= f32::EPSILON {
        return t;
    }
    let target = t * total;
    // First index with cumlen[i] >= target (linear scan; 17 entries).
    let mut hi = 1;
    while hi < LUT_SIZE - 1 && cumlen[hi] < target {
        hi += 1;
    }
    let lo = hi - 1;
    let span = cumlen[hi] - cumlen[lo];
    let local = if span > 0.0 {
        (target - cumlen[lo]) / span
    } else {
        0.0
    };
    let u0 = lo as f32 / (LUT_SIZE - 1) as f32;
    let u1 = hi as f32 / (LUT_SIZE - 1) as f32;
    u0 + (u1 - u0) * local
}

/// One point on a keyframed curve. `tick` is clip-relative (offset from the
/// clip's timeline start) at the timeline rate.
///
/// `tangents` is present on every keyframe type for serde simplicity, but
/// only `[f32; 2]` [`SegmentSample`] reads it. Routing rejects tangents on
/// non-position params.
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
    /// Spatial bezier handles (motion paths). Only meaningful for
    /// `[f32; 2]` position params; ignored by other samplers.
    #[serde(rename = "s", default, skip_serializing_if = "Option::is_none")]
    pub tangents: Option<SpatialTangents>,
}

impl<T> Keyframe<T> {
    /// Build a keyframe with no spatial tangents (straight-line segments).
    pub fn new(tick: i64, value: T, easing: Easing) -> Self {
        Self {
            tick,
            value,
            easing,
            tangents: None,
        }
    }
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

impl<T: SegmentSample> Param<T> {
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
                let eased_t = k0.easing.apply(t);
                T::segment_sample(k0, k1, eased_t)
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
    /// single-keyframe curve. Replacing an existing keyframe preserves its
    /// spatial tangents (use [`Self::set_keyframe_tangents`] to change them).
    pub fn set_keyframe(&mut self, tick: i64, value: T, easing: Easing) {
        match self {
            Param::Constant(_) => {
                *self = Param::Keyframed {
                    keyframes: vec![Keyframe::new(tick, value, easing)],
                };
            }
            Param::Keyframed { keyframes } => {
                match keyframes.binary_search_by_key(&tick, |kf| kf.tick) {
                    Ok(i) => {
                        let tangents = keyframes[i].tangents;
                        keyframes[i] = Keyframe {
                            tick,
                            value,
                            easing,
                            tangents,
                        };
                    }
                    Err(i) => keyframes.insert(i, Keyframe::new(tick, value, easing)),
                }
            }
        }
    }

    /// Set or clear spatial tangents on the keyframe at `tick`. Errors when
    /// no keyframe sits there (including on a constant param).
    pub fn set_keyframe_tangents(
        &mut self,
        tick: i64,
        tangents: Option<SpatialTangents>,
    ) -> Result<(), ModelError> {
        let Param::Keyframed { keyframes } = self else {
            return Err(ModelError::InvalidParam(format!(
                "no keyframe at {tick} to set tangents"
            )));
        };
        let Ok(i) = keyframes.binary_search_by_key(&tick, |kf| kf.tick) else {
            return Err(ModelError::InvalidParam(format!(
                "no keyframe at {tick} to set tangents"
            )));
        };
        if let Some(t) = tangents {
            t.validate()?;
        }
        keyframes[i].tangents = tangents;
        Ok(())
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
                        tangents: kf.tangents,
                    })
                    .collect(),
            },
        }
    }

    /// Shift every keyframe by `delta` ticks (constants pass through
    /// unchanged). The operation is atomic: an overflowing tick leaves the
    /// parameter untouched.
    ///
    /// Clip splitting uses this to preserve an absolute animation curve on
    /// the tail while rebasing its clip-relative origin to the split point.
    /// Spatial tangents travel with their keyframes unchanged — the full
    /// segment (including bezier handles) stays intact on both halves, so
    /// mid-segment splits need no de Casteljau subdivision.
    pub fn shift_ticks(&mut self, delta: i64) -> Result<(), ModelError> {
        let Param::Keyframed { keyframes } = self else {
            return Ok(());
        };
        let shifted = keyframes
            .iter()
            .map(|kf| {
                Ok(Keyframe {
                    tick: kf.tick.checked_add(delta).ok_or(ModelError::TimeOverflow)?,
                    value: kf.value.clone(),
                    easing: kf.easing,
                    tangents: kf.tangents,
                })
            })
            .collect::<Result<Vec<_>, ModelError>>()?;
        *keyframes = shifted;
        Ok(())
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
            if let Some(t) = kf.tangents {
                t.validate()?;
            }
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
mod tests;
