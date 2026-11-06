//! Follows a target through frames.
//!
//! The shell decodes each frame, turns it into a small [`Gray`] copy and
//! hands them to a [`Tracker`] one at a time, in reverse to track back.
//! The target is four corners in unit coordinates of the frame (0 to 1,
//! x right, y down). Points inside it are followed with pyramidal
//! Lucas-Kanade and a forward-backward check, then a RANSAC fit finds the
//! target's move: [`Mode::Object`] fits move, scale and turn,
//! [`Mode::Plane`] fits a homography so all four corners follow a tilt.
//! Each step answers the new corners and a confidence, or [`Step::Lost`].
//! Pure Rust; no model, no file.

mod fit;
mod flow;
mod gray;
mod points;
mod quad;

use fit::{Model, Random, ransac};
use gray::Pyramid;
use quad::Corners;

pub use gray::{GRAY_SIDE, Gray, MAX_SIDE};

/// What the target's move may be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Move, scale and turn: a face, a person, a ball.
    Object,
    /// Four corners that move apart: a screen, a sign, a number plate.
    Plane,
}

/// Four corners in unit coordinates of the frame (0 to 1, x right, y
/// down), in order around the target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quad(pub [(f64, f64); 4]);

impl Quad {
    /// The box at `x`, `y` (its top left) that is `width` by `height`.
    #[must_use]
    pub fn from_box(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self([
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height),
        ])
    }

    /// The middle of the four corners.
    #[must_use]
    pub fn middle(&self) -> (f64, f64) {
        Corners(self.0).middle()
    }

    fn to_pixels(self, width: u32, height: u32) -> Corners {
        let (w, h) = (f64::from(width), f64::from(height));
        Corners(self.0.map(|(x, y)| (x * w, y * h)))
    }

    fn from_pixels(corners: Corners, width: u32, height: u32) -> Self {
        let (w, h) = (f64::from(width), f64::from(height));
        Self(corners.0.map(|(x, y)| (x / w, y / h)))
    }
}

/// What one frame gave.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Step {
    /// The target's corners in this frame, and how sure the fit is, 0 to 1.
    Tracked { quad: Quad, confidence: f32 },
    /// Too few points agree, the target jumped, folded or left the frame.
    /// Every later step answers this too.
    Lost,
}

/// Why tracking could not start or go on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MotionError {
    /// The frame is empty, too large, or its pixels do not fill it.
    Frame { reason: String },
    /// The target is folded, too small, not a number, or off the frame.
    Target { reason: String },
    /// The target has too little texture to follow.
    Featureless,
}

impl std::fmt::Display for MotionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Frame { reason } => write!(f, "bad frame: {reason}"),
            Self::Target { reason } => write!(f, "bad target: {reason}"),
            Self::Featureless => f.write_str("the target has too little detail to follow"),
        }
    }
}

impl std::error::Error for MotionError {}

/// The smallest target, in grey-copy pixels.
const MIN_AREA: f64 = 100.0;

/// A pair agrees with the fit within this many grey-copy pixels.
const THRESHOLD: f64 = 1.5;

/// The share of followed points that must agree with the fit.
const MIN_AGREEMENT: f64 = 0.3;

/// The share of the points carried into a frame that must agree with the
/// fit. When the target vanishes, a few stray points can still agree on
/// standing still; this keeps them from passing for the target.
const MIN_KEPT: f64 = 0.25;

/// The most the target's area may grow or shrink in one frame.
const MAX_GROWTH: f64 = 1.6;

/// Fresh points are picked when fewer than this share of the first remain.
const RESEED: f64 = 0.6;

/// The fixed seed of RANSAC's picks.
const SEED: u64 = 0x5eed_c0de_cafe_f00d;

/// A followed point: where it is now, and where it was in the first
/// frame. Every fit runs from the first frame to the current one, so small
/// errors in one frame's fit do not add up over the track.
#[derive(Clone, Copy, Debug)]
struct Point {
    at: (f32, f32),
    origin: (f64, f64),
}

/// Follows one target from frame to frame.
pub struct Tracker {
    mode: Mode,
    width: u32,
    height: u32,
    prev: Pyramid,
    /// The target in the first frame, and in the last.
    origin: Corners,
    corners: Corners,
    points: Vec<Point>,
    first: usize,
    random: Random,
    lost: bool,
}

impl Tracker {
    /// Starts on `frame` with the target at `target`.
    pub fn start(frame: &Gray, target: Quad, mode: Mode) -> Result<Self, MotionError> {
        let (width, height) = (frame.width(), frame.height());
        let corners = target.to_pixels(width, height);
        if !corners.is_convex() {
            return Err(MotionError::Target {
                reason: "the corners are folded or not numbers".into(),
            });
        }
        if corners.area() < MIN_AREA {
            return Err(MotionError::Target {
                reason: "the target is too small".into(),
            });
        }
        if !inside(corners.middle(), width, height) {
            return Err(MotionError::Target {
                reason: "the target is off the frame".into(),
            });
        }
        let prev = Pyramid::new(frame);
        let points: Vec<Point> = points::pick(prev.base(), &corners, &[], points::MAX_POINTS)
            .into_iter()
            .map(|at| Point {
                at,
                origin: (f64::from(at.0), f64::from(at.1)),
            })
            .collect();
        if points.len() < min_points(mode) {
            return Err(MotionError::Featureless);
        }
        Ok(Self {
            mode,
            width,
            height,
            prev,
            origin: corners,
            corners,
            first: points.len(),
            points,
            random: Random::new(SEED),
            lost: false,
        })
    }

    /// Follows the target into `next`, the frame after the last one (or
    /// before it, tracking back).
    pub fn step(&mut self, next: &Gray) -> Result<Step, MotionError> {
        if (next.width(), next.height()) != (self.width, self.height) {
            return Err(MotionError::Frame {
                reason: "the frame's size changed".into(),
            });
        }
        if self.lost {
            return Ok(Step::Lost);
        }
        let next = Pyramid::new(next);
        let Some((model, corners, points, confidence)) = self.follow(&next) else {
            self.lost = true;
            return Ok(Step::Lost);
        };
        self.corners = corners;
        self.points = points;
        if (self.points.len() as f64) < self.first as f64 * RESEED
            && let Some(back) = model.inverse()
        {
            let taken: Vec<(f32, f32)> = self.points.iter().map(|p| p.at).collect();
            let fresh = points::pick(next.base(), &corners, &taken, points::MAX_POINTS);
            self.points.extend(fresh.into_iter().filter_map(|at| {
                let origin = back.apply((f64::from(at.0), f64::from(at.1)))?;
                Some(Point { at, origin })
            }));
        }
        self.prev = next;
        Ok(Step::Tracked {
            quad: Quad::from_pixels(corners, self.width, self.height),
            confidence,
        })
    }

    /// The target's corners in the last frame.
    #[must_use]
    pub fn quad(&self) -> Quad {
        Quad::from_pixels(self.corners, self.width, self.height)
    }

    #[must_use]
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// The move from the first frame, the new corners, the points that
    /// agreed, and the confidence; `None` when the target is lost.
    fn follow(&mut self, next: &Pyramid) -> Option<(Model, Corners, Vec<Point>, f32)> {
        let need = min_points(self.mode);
        let at: Vec<(f32, f32)> = self.points.iter().map(|p| p.at).collect();
        let followed = flow::follow_all(&self.prev, next, &at);
        if followed.len() < need {
            return None;
        }
        let pairs: Vec<fit::Anchored> = followed
            .iter()
            .map(|&(i, q)| (self.points[i].origin, q))
            .collect();
        let fit = ransac(
            &pairs,
            self.mode == Mode::Plane,
            THRESHOLD,
            &mut self.random,
        )?;
        let agreement = fit.count as f64 / pairs.len() as f64;
        let kept = fit.count as f64 / self.points.len().max(1) as f64;
        if fit.count < need || agreement < MIN_AGREEMENT || kept < MIN_KEPT {
            return None;
        }
        let moved = self.origin.0.map(|p| fit.model.apply(p));
        let corners = Corners([moved[0]?, moved[1]?, moved[2]?, moved[3]?]);
        let growth = corners.area() / self.corners.area().max(f64::EPSILON);
        if !corners.is_convex()
            || corners.area() < MIN_AREA
            || !(1.0 / MAX_GROWTH..=MAX_GROWTH).contains(&growth)
            || !inside(corners.middle(), self.width, self.height)
        {
            return None;
        }
        let points: Vec<Point> = pairs
            .iter()
            .zip(&fit.inliers)
            .filter_map(|(&(origin, at), &keep)| keep.then_some(Point { at, origin }))
            .collect();
        let enough = (fit.count as f64 / (self.first as f64 / 2.0).max(need as f64)).min(1.0);
        Some((fit.model, corners, points, (agreement * enough) as f32))
    }
}

/// The fewest points a fit needs to be trusted.
fn min_points(mode: Mode) -> usize {
    match mode {
        Mode::Object => 6,
        Mode::Plane => 10,
    }
}

fn inside(p: (f64, f64), width: u32, height: u32) -> bool {
    p.0 >= 0.0 && p.1 >= 0.0 && p.0 <= f64::from(width) && p.1 <= f64::from(height)
}

#[cfg(test)]
mod tests;
