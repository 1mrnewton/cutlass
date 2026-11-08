//! Auto reframe's plan: where a window of one shape sits over each frame so
//! it follows the subject without shaking or swinging.
//!
//! A job finds a [`Subject`] in frames sampled at a steady rate: where in
//! the upright frame the subject is, 0 to 1 across and down, and how sure
//! that is. [`plan_reframe`] smooths that track, holds the window still
//! while the subject stays near its middle, and moves it no faster than the
//! speed allows, then answers the window's middle at each sample.

/// Where the subject is in one frame: `x` across and `y` down, 0 to 1, and
/// how sure that is, 0 for a frame with nothing that stands out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Subject {
    pub x: f64,
    pub y: f64,
    pub weight: f64,
}

impl Subject {
    /// Nothing found: the middle, unsure.
    pub const NONE: Self = Self {
        x: 0.5,
        y: 0.5,
        weight: 0.0,
    };
}

/// How the window follows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Follow {
    /// Seconds over which the subject's track is smoothed.
    pub smooth: f64,
    /// How far, as a share of the window, the subject may drift from the
    /// window's middle before it moves.
    pub slack: f64,
    /// The most the window moves, as a share of the frame, each second.
    pub speed: f64,
}

impl Follow {
    /// Image stabilization 1 Flexible, 2 Default or 3 Stable, and camera
    /// moving speed 1 Fast, 2 Default or 3 Slow; anything else is Default.
    #[must_use]
    pub fn from_settings(stability: i64, speed: i64) -> Self {
        let (smooth, slack) = match stability {
            1 => (0.25, 0.06),
            3 => (1.5, 0.3),
            _ => (0.6, 0.15),
        };
        let speed = match speed {
            1 => 0.6,
            3 => 0.12,
            _ => 0.3,
        };
        Self {
            smooth,
            slack,
            speed,
        }
    }
}

/// The share of the frame a window `window` wide over high covers in a
/// frame `frame` wide over high: the whole of one side, and as much of the
/// other as the shape allows.
#[must_use]
pub fn window_size(frame: f64, window: f64) -> (f64, f64) {
    if !(frame.is_finite() && window.is_finite() && frame > 0.0 && window > 0.0) {
        return (1.0, 1.0);
    }
    if window < frame {
        (window / frame, 1.0)
    } else {
        (1.0, frame / window)
    }
}

/// The window's middle, 0 to 1 across and down, at each of `subjects`,
/// sampled `rate` times a second, for a window `size` (from
/// [`window_size`]) following as `follow` says. Always inside the frame.
#[must_use]
pub fn plan_reframe(
    subjects: &[Subject],
    rate: f64,
    size: (f64, f64),
    follow: Follow,
) -> Vec<(f64, f64)> {
    if subjects.is_empty() {
        return Vec::new();
    }
    let rate = if rate.is_finite() && rate > 0.0 {
        rate
    } else {
        1.0
    };
    let clean: Vec<Subject> = subjects
        .iter()
        .map(|s| {
            if s.x.is_finite() && s.y.is_finite() && s.weight.is_finite() {
                Subject {
                    x: s.x.clamp(0.0, 1.0),
                    y: s.y.clamp(0.0, 1.0),
                    weight: s.weight.clamp(0.0, 1.0),
                }
            } else {
                Subject::NONE
            }
        })
        .collect();
    let weights: Vec<f64> = clean.iter().map(|s| s.weight).collect();
    let axis = |values: Vec<f64>, side: f64| -> Vec<f64> {
        let low = side / 2.0;
        let high = 1.0 - side / 2.0;
        if high - low <= 1e-9 {
            return vec![0.5; values.len()];
        }
        let smoothed = smooth(&values, &weights, follow.smooth * rate);
        let held = hold(
            &smoothed,
            follow.slack * side,
            follow.speed / rate,
            low,
            high,
        );
        let eased = smooth(&held, &vec![1.0; held.len()], 0.15 * rate);
        eased.into_iter().map(|v| v.clamp(low, high)).collect()
    };
    let xs = axis(clean.iter().map(|s| s.x).collect(), size.0);
    let ys = axis(clean.iter().map(|s| s.y).collect(), size.1);
    xs.into_iter().zip(ys).collect()
}

/// `values` smoothed forward and back over about `frames` samples, each
/// sample counting as much as its weight. Where no sample counts yet, the
/// nearest that does, or the middle, holds.
fn smooth(values: &[f64], weights: &[f64], frames: f64) -> Vec<f64> {
    let take = if frames > 1e-9 {
        1.0 - (-1.0 / frames).exp()
    } else {
        1.0
    };
    // Each pass starts from the first sample it meets that counts.
    let pass = |order: Vec<usize>| -> Vec<(usize, f64)> {
        let mut at = order
            .iter()
            .find(|i| weights[**i] > 0.0)
            .map_or(0.5, |i| values[*i]);
        order
            .into_iter()
            .map(|i| {
                at += take * weights[i] * (values[i] - at);
                (i, at)
            })
            .collect()
    };
    let forward = pass((0..values.len()).collect());
    let backward = pass((0..values.len()).rev().collect());
    let mut out = vec![0.5; values.len()];
    for (i, v) in forward {
        out[i] = v;
    }
    for (i, v) in backward {
        out[i] = (out[i] + v) / 2.0;
    }
    out
}

/// A window middle that stays put while `targets` stay within `slack` of
/// it, then follows no faster than `step` a sample, between `low` and
/// `high`.
fn hold(targets: &[f64], slack: f64, step: f64, low: f64, high: f64) -> Vec<f64> {
    let mut at = targets.first().copied().unwrap_or(0.5).clamp(low, high);
    targets
        .iter()
        .map(|target| {
            let off = target - at;
            if off.abs() > slack {
                let want = off - slack.copysign(off);
                at += want.clamp(-step, step);
                at = at.clamp(low, high);
            }
            at
        })
        .collect()
}

#[cfg(test)]
mod tests;
