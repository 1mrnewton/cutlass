//! Steadies a shaky clip.
//!
//! A [`Path`] holds how the whole picture moved from each frame to the
//! next. [`plan`] turns it into a steady clip at a [`Strength`].

use crate::MotionError;

mod plan;

pub use plan::{Correction, Plan, Strength, plan};

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

pub(crate) fn check_rate(rate: f64) -> Result<(), MotionError> {
    if rate.is_finite() && rate > 0.0 && rate <= MAX_RATE {
        Ok(())
    } else {
        Err(MotionError::Path {
            reason: format!("{rate} frames a second is not a frame rate"),
        })
    }
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
