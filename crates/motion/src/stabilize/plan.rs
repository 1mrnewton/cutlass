//! Turns a measured [`Path`] into a steady one.
//!
//! The camera's place through the clip (slide, turn and growth, added up
//! from the moves) is smoothed with a Gaussian as wide as the [`Strength`]
//! asks. Each frame is then moved from where the camera was to where the
//! smooth path says it should have been, and the whole clip is zoomed in
//! just enough that no frame shows its edge. A strength caps that zoom: a
//! frame whose full correction would need more keeps only the share that
//! fits, and that share is eased in and out over its neighbours so the cap
//! does not jolt. This is cheap, so a new strength never measures again.

use super::{Move, Path, Sim, check_rate};
use crate::MotionError;

/// How hard to steady, against how much of the picture is cut away.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Strength {
    /// Light smoothing; keeps nearly the whole frame.
    MinimumCut,
    /// Smooths a walking hand; the default.
    #[default]
    Recommended,
    /// Smooths almost to a tripod or a dolly, and cuts more away.
    MostStable,
}

impl Strength {
    /// The Gaussian's width, in seconds.
    fn seconds(self) -> f64 {
        match self {
            Self::MinimumCut => 0.2,
            Self::Recommended => 0.5,
            Self::MostStable => 1.2,
        }
    }

    /// The most the clip may be zoomed in to hide the edges.
    #[must_use]
    pub fn max_zoom(self) -> f64 {
        match self {
            Self::MinimumCut => 1.05,
            Self::Recommended => 1.1,
            Self::MostStable => 1.2,
        }
    }
}

/// How one frame is placed to steady it, about the frame's middle: grown
/// by `scale` (which holds the clip's zoom), turned by `turn` radians
/// (clockwise on screen), then slid by `x`, `y` in units of the frame's
/// width and height.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Correction {
    pub x: f64,
    pub y: f64,
    pub turn: f64,
    pub scale: f64,
}

impl Correction {
    /// Leaves the frame where it is.
    pub const NONE: Self = Self {
        x: 0.0,
        y: 0.0,
        turn: 0.0,
        scale: 1.0,
    };
}

/// A steadied clip: one correction per frame, the first frame first.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    /// The zoom every frame holds, 1 or more.
    pub zoom: f64,
    pub frames: Vec<Correction>,
}

/// The most frames a path may hold: over four hours at 60 a second.
const MAX_FRAMES: usize = 1_000_000;

/// The steps that find how much of a correction fits.
const BISECTIONS: usize = 24;

/// Steadies `path` at `strength`.
pub fn plan(path: &Path, strength: Strength) -> Result<Plan, MotionError> {
    check_rate(path.rate)?;
    if path.width == 0 || path.height == 0 {
        return Err(MotionError::Path {
            reason: "the frame has no size".into(),
        });
    }
    if path.moves.len() >= MAX_FRAMES {
        return Err(MotionError::Path {
            reason: format!("{} frames is too many", path.moves.len() + 1),
        });
    }
    let frame = Frame::new(path.width, path.height);
    let raw = poses(path, &frame);
    let sigma = strength.seconds() * path.rate;
    let radius = (3.0 * sigma).ceil() as usize;
    let smooth = Pose::blur(&raw, sigma, radius);
    let max_zoom = strength.max_zoom();

    // The share of each correction that fits in the zoom, eased so that no
    // frame gets more than fits it: the least over each window, then
    // blurred over the same window.
    let fits: Vec<f64> = raw
        .iter()
        .zip(&smooth)
        .map(|(r, s)| frame.largest_share(r, s, 1.0, max_zoom))
        .collect();
    let eased = blur(&least(&fits, radius), sigma, radius, false);

    let mut moves = Vec::with_capacity(raw.len());
    let mut zoom = 1.0_f64;
    for (((r, s), &fit), &ease) in raw.iter().zip(&smooth).zip(&fits).zip(&eased) {
        let mut share = ease.min(fit).clamp(0.0, 1.0);
        let mut correct = frame.correction(r, &r.toward(s, share));
        if frame.needed_zoom(&correct) > max_zoom {
            share = frame.largest_share(r, s, share, max_zoom);
            correct = frame.correction(r, &r.toward(s, share));
        }
        zoom = zoom.max(frame.needed_zoom(&correct));
        moves.push(correct);
    }
    let zoom = zoom.clamp(1.0, max_zoom);
    let frames = moves.iter().map(|t| frame.place(t, zoom)).collect();
    Ok(Plan { zoom, frames })
}

/// Where the camera is in one frame against the first: how far the
/// middle slid, how far it turned, and the log of how much it grew.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Pose {
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) turn: f64,
    pub(crate) grow: f64,
}

impl Pose {
    /// `share` of the way from this pose to `other`.
    fn toward(&self, other: &Self, share: f64) -> Self {
        let mix = |a: f64, b: f64| a + (b - a) * share;
        Self {
            x: mix(self.x, other.x),
            y: mix(self.y, other.y),
            turn: mix(self.turn, other.turn),
            grow: mix(self.grow, other.grow),
        }
    }

    fn blur(poses: &[Self], sigma: f64, radius: usize) -> Vec<Self> {
        let channel = |get: fn(&Self) -> f64| {
            let values: Vec<f64> = poses.iter().map(get).collect();
            blur(&values, sigma, radius, true)
        };
        let (x, y, turn, grow) = (
            channel(|p| p.x),
            channel(|p| p.y),
            channel(|p| p.turn),
            channel(|p| p.grow),
        );
        (0..poses.len())
            .map(|i| Self {
                x: x[i],
                y: y[i],
                turn: turn[i],
                grow: grow[i],
            })
            .collect()
    }
}

/// The camera's pose in every frame, the first at rest. A move that is not
/// a number or is too wild to be a camera's is taken as still.
fn poses(path: &Path, frame: &Frame) -> Vec<Pose> {
    let mut out = Vec::with_capacity(path.moves.len() + 1);
    out.push(Pose::default());
    let (mut sim, mut turn, mut grow) = (Sim::IDENTITY, 0.0, 0.0);
    for found in &path.moves {
        let found = if frame.believable(found) {
            *found
        } else {
            Move::STILL
        };
        sim = sim.then(&Sim::about(
            frame.middle,
            found.dx,
            found.dy,
            found.turn,
            found.scale,
        ));
        turn += found.turn;
        grow += found.scale.ln();
        let at = sim.apply(frame.middle);
        out.push(Pose {
            x: at.0 - frame.middle.0,
            y: at.1 - frame.middle.1,
            turn,
            grow,
        });
    }
    out
}

/// A Gaussian blur of `values` cut off at `radius`. Past the ends, `odd`
/// mirrors the values through the end one so a pan keeps going; otherwise
/// the end value repeats.
fn blur(values: &[f64], sigma: f64, radius: usize, odd: bool) -> Vec<f64> {
    if values.len() < 2 || radius == 0 || sigma < 1e-6 {
        return values.to_vec();
    }
    let weights: Vec<f64> = (0..=radius)
        .map(|d| (-0.5 * (d as f64 / sigma).powi(2)).exp())
        .collect();
    let total: f64 = weights[0] + 2.0 * weights[1..].iter().sum::<f64>();
    let last = values.len() - 1;
    let at = |i: isize| -> f64 {
        if i < 0 {
            let mirror = values[(i.unsigned_abs()).min(last)];
            if odd {
                2.0 * values[0] - mirror
            } else {
                values[0]
            }
        } else if i as usize > last {
            let mirror = values[last.saturating_sub(i as usize - last)];
            if odd {
                2.0 * values[last] - mirror
            } else {
                values[last]
            }
        } else {
            values[i as usize]
        }
    };
    (0..values.len())
        .map(|i| {
            let i = i as isize;
            let mut sum = weights[0] * at(i);
            for (d, w) in weights.iter().enumerate().skip(1) {
                let d = d as isize;
                sum += w * (at(i - d) + at(i + d));
            }
            sum / total
        })
        .collect()
}

/// The least of `values` within `radius` of each one.
fn least(values: &[f64], radius: usize) -> Vec<f64> {
    (0..values.len())
        .map(|i| {
            let (from, to) = (i.saturating_sub(radius), (i + radius).min(values.len() - 1));
            values[from..=to]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min)
        })
        .collect()
}

/// The grey copy's size, which the path is measured in.
struct Frame {
    width: f64,
    height: f64,
    middle: (f64, f64),
}

impl Frame {
    fn new(width: u32, height: u32) -> Self {
        let (width, height) = (f64::from(width), f64::from(height));
        Self {
            width,
            height,
            middle: (width / 2.0, height / 2.0),
        }
    }

    fn believable(&self, found: &Move) -> bool {
        [found.dx, found.dy, found.turn, found.scale]
            .iter()
            .all(|v| v.is_finite())
            && found.dx.abs() <= self.width
            && found.dy.abs() <= self.height
            && found.turn.abs() <= std::f64::consts::FRAC_PI_2
            && (0.5..=2.0).contains(&found.scale)
    }

    fn sim(&self, pose: &Pose) -> Sim {
        Sim::about(self.middle, pose.x, pose.y, pose.turn, pose.grow.exp())
    }

    /// The move that takes a frame shot at `from` to where `to` would
    /// have shown it.
    fn correction(&self, from: &Pose, to: &Pose) -> Sim {
        match self.sim(from).inverse() {
            Some(back) => back.then(&self.sim(to)),
            None => Sim::IDENTITY,
        }
    }

    /// The least zoom, about the middle and after `correct`, that leaves no
    /// part of the canvas outside the frame; infinite when none does.
    fn needed_zoom(&self, correct: &Sim) -> f64 {
        let Some(back) = correct.inverse() else {
            return f64::INFINITY;
        };
        let m = self.middle;
        let home = back.apply(m);
        let shift = (home.0 - m.0, home.1 - m.1);
        let half = (self.width / 2.0, self.height / 2.0);
        if shift.0.abs() >= half.0 || shift.1.abs() >= half.1 {
            return f64::INFINITY;
        }
        let mut zoom = 0.0_f64;
        for corner in [
            (0.0, 0.0),
            (self.width, 0.0),
            (self.width, self.height),
            (0.0, self.height),
        ] {
            let v = (corner.0 - m.0, corner.1 - m.1);
            let d = (back.a * v.0 - back.b * v.1, back.b * v.0 + back.a * v.1);
            for (d, c, half) in [(d.0, shift.0, half.0), (d.1, shift.1, half.1)] {
                let need = if d > 0.0 {
                    d / (half - c)
                } else if d < 0.0 {
                    -d / (half + c)
                } else {
                    0.0
                };
                zoom = zoom.max(need);
            }
        }
        zoom
    }

    /// The largest share, up to `most`, of the way from `from` to `to` whose
    /// correction needs no more than `max_zoom`.
    fn largest_share(&self, from: &Pose, to: &Pose, most: f64, max_zoom: f64) -> f64 {
        let fits = |share: f64| {
            self.needed_zoom(&self.correction(from, &from.toward(to, share))) <= max_zoom
        };
        if fits(most) {
            return most;
        }
        let (mut low, mut high) = (0.0, most);
        for _ in 0..BISECTIONS {
            let mid = (low + high) / 2.0;
            if fits(mid) {
                low = mid;
            } else {
                high = mid;
            }
        }
        low
    }

    /// `correct`, then `zoom` about the middle, as a [`Correction`].
    fn place(&self, correct: &Sim, zoom: f64) -> Correction {
        let m = self.middle;
        let to = correct.apply(m);
        Correction {
            x: zoom * (to.0 - m.0) / self.width,
            y: zoom * (to.1 - m.1) / self.height,
            turn: correct.b.atan2(correct.a),
            scale: zoom * correct.a.hypot(correct.b),
        }
    }
}

#[cfg(test)]
mod tests;
