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

    /// Shift every keyframe by `delta` ticks (constants pass through
    /// unchanged). The operation is atomic: an overflowing tick leaves the
    /// parameter untouched.
    ///
    /// Clip splitting uses this to preserve an absolute animation curve on
    /// the tail while rebasing its clip-relative origin to the split point.
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
