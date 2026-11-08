//! Steadies a shaky clip.
//!
//! [`Camera`] takes the clip's small grey copies in order and measures how
//! the whole picture moved from each frame to the next: corner points spread
//! over a grid, followed with Lucas-Kanade, and a RANSAC similarity (move,
//! turn, scale) that the background agrees on, so a person walking through
//! does not count. The moves make a [`Path`]. That is the slow part, done
//! once per clip. [`plan`] turns it into a steady clip at a [`Strength`].

use crate::fit::{self, Model, Random};
use crate::gray::{Gray, Level, Pyramid};
use crate::quad::Corners;
use crate::{MotionError, SEED, flow, points};

mod plan;

pub use plan::{Correction, Plan, Strength, plan};

/// The frame is cut into this many cells a side, and points are picked in
/// each, so one busy corner of the picture does not take them all.
const GRID: usize = 4;

/// The most points picked in one cell.
const PER_CELL: usize = 12;

/// The most points followed from one frame to the next.
pub(crate) const MAX_SPREAD: usize = GRID * GRID * PER_CELL;

/// The fewest points that must agree on a move to trust it.
const MIN_POINTS: usize = 12;

/// A point agrees with the move within this many grey-copy pixels.
const THRESHOLD: f64 = 1.0;

/// The share of followed points that must agree on the move.
const MIN_AGREEMENT: f64 = 0.3;

/// The most frames a rate may be, per second.
const MAX_RATE: f64 = 1000.0;

/// How the picture moved from one frame to the next, about the frame's
/// middle: it slid `dx`, `dy` grey-copy pixels (x right, y down), turned
/// `turn` radians (clockwise on screen) and grew by `scale`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Move {
    pub dx: f64,
    pub dy: f64,
    pub turn: f64,
    pub scale: f64,
    /// The share of followed points that agree, 0 to 1. 0 when the move
    /// could not be measured (a flat or dark frame, a cut, a whip pan)
    /// and the frame is taken as still.
    pub confidence: f32,
}

impl Move {
    /// No move, measured or assumed.
    pub const STILL: Self = Self {
        dx: 0.0,
        dy: 0.0,
        turn: 0.0,
        scale: 1.0,
        confidence: 0.0,
    };
}

/// The moves measured through a clip, one between each pair of frames.
#[derive(Clone, Debug, PartialEq)]
pub struct Path {
    /// The grey copies' size, which the moves are in.
    pub width: u32,
    pub height: u32,
    /// Frames measured per second.
    pub rate: f64,
    pub moves: Vec<Move>,
}

/// Measures the camera's moves through a clip, one frame at a time.
pub struct Camera {
    prev: Pyramid,
    path: Path,
    random: Random,
}

impl Camera {
    /// Starts on the clip's first frame; `rate` is the frames handed in per
    /// second.
    pub fn start(first: &Gray, rate: f64) -> Result<Self, MotionError> {
        check_rate(rate)?;
        Ok(Self {
            prev: Pyramid::new(first),
            path: Path {
                width: first.width(),
                height: first.height(),
                rate,
                moves: Vec::new(),
            },
            random: Random::new(SEED),
        })
    }

    /// Measures the move into `next`, the frame after the last one.
    pub fn step(&mut self, next: &Gray) -> Result<Move, MotionError> {
        if (next.width(), next.height()) != (self.path.width, self.path.height) {
            return Err(MotionError::Frame {
                reason: "the frame's size changed".into(),
            });
        }
        let next = Pyramid::new(next);
        let found = measure(&self.prev, &next, &mut self.random).unwrap_or(Move::STILL);
        self.path.moves.push(found);
        self.prev = next;
        Ok(found)
    }

    /// The moves so far.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    pub fn finish(self) -> Path {
        self.path
    }
}

pub(crate) fn check_rate(rate: f64) -> Result<(), MotionError> {
    if rate.is_finite() && rate > 0.0 && rate <= MAX_RATE {
        Ok(())
    } else {
        Err(MotionError::Path {
            reason: format!("{rate} frames a second is not a frame rate"),
        })
    }
}

/// The move from `prev` to `next`, or `None` when too few points agree.
fn measure(prev: &Pyramid, next: &Pyramid, random: &mut Random) -> Option<Move> {
    let base = prev.base();
    let at = spread(base);
    if at.len() < MIN_POINTS {
        return None;
    }
    let followed = flow::follow_all(prev, next, &at);
    if followed.len() < MIN_POINTS {
        return None;
    }
    let pairs: Vec<fit::Anchored> = followed
        .iter()
        .map(|&(i, q)| ((f64::from(at[i].0), f64::from(at[i].1)), q))
        .collect();
    let found = fit::ransac(&pairs, false, THRESHOLD, random)?;
    let agreement = found.count as f64 / pairs.len() as f64;
    if found.count < MIN_POINTS || agreement < MIN_AGREEMENT {
        return None;
    }
    let Model::Similarity { a, b, tx, ty } = found.model else {
        return None;
    };
    let middle = (base.width as f64 / 2.0, base.height as f64 / 2.0);
    let to = Sim { a, b, tx, ty }.apply(middle);
    let measured = Move {
        dx: to.0 - middle.0,
        dy: to.1 - middle.1,
        turn: b.atan2(a),
        scale: a.hypot(b),
        confidence: agreement as f32,
    };
    [measured.dx, measured.dy, measured.turn, measured.scale]
        .iter()
        .all(|v| v.is_finite())
        .then_some(measured)
}

/// Corner points over the whole frame, picked cell by cell.
pub(crate) fn spread(level: &Level) -> Vec<(f32, f32)> {
    let (w, h) = (level.width as f64, level.height as f64);
    let mut out = Vec::with_capacity(MAX_SPREAD);
    for row in 0..GRID {
        for col in 0..GRID {
            let (x0, x1) = (
                w * col as f64 / GRID as f64,
                w * (col + 1) as f64 / GRID as f64,
            );
            let (y0, y1) = (
                h * row as f64 / GRID as f64,
                h * (row + 1) as f64 / GRID as f64,
            );
            let cell = Corners([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]);
            out.extend(points::pick(level, &cell, &[], PER_CELL));
        }
    }
    out
}

/// A similarity in pixels: `x' = a x - b y + tx`, `y' = b x + a y + ty`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Sim {
    pub(crate) a: f64,
    pub(crate) b: f64,
    pub(crate) tx: f64,
    pub(crate) ty: f64,
}

impl Sim {
    pub(crate) const IDENTITY: Self = Self {
        a: 1.0,
        b: 0.0,
        tx: 0.0,
        ty: 0.0,
    };

    /// About `middle`: grow by `scale`, turn by `turn`, then slide `dx`, `dy`.
    pub(crate) fn about(middle: (f64, f64), dx: f64, dy: f64, turn: f64, scale: f64) -> Self {
        let (a, b) = (scale * turn.cos(), scale * turn.sin());
        Self {
            a,
            b,
            tx: middle.0 + dx - (a * middle.0 - b * middle.1),
            ty: middle.1 + dy - (b * middle.0 + a * middle.1),
        }
    }

    pub(crate) fn apply(&self, p: (f64, f64)) -> (f64, f64) {
        (
            self.a * p.0 - self.b * p.1 + self.tx,
            self.b * p.0 + self.a * p.1 + self.ty,
        )
    }

    /// This move, then `next`.
    pub(crate) fn then(&self, next: &Self) -> Self {
        Self {
            a: next.a * self.a - next.b * self.b,
            b: next.b * self.a + next.a * self.b,
            tx: next.a * self.tx - next.b * self.ty + next.tx,
            ty: next.b * self.tx + next.a * self.ty + next.ty,
        }
    }

    /// The move back, or `None` when this one squashes the plane to a point.
    pub(crate) fn inverse(&self) -> Option<Self> {
        let norm = self.a * self.a + self.b * self.b;
        if !norm.is_finite() || norm < 1e-12 {
            return None;
        }
        let (a, b) = (self.a / norm, -self.b / norm);
        Some(Self {
            a,
            b,
            tx: -(a * self.tx - b * self.ty),
            ty: -(b * self.tx + a * self.ty),
        })
    }
}

#[cfg(test)]
mod tests;
