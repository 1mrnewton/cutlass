//! Drawing a shape with the pen, point by point, as in any vector editor.
//!
//! A press puts down a point. Dragging before the release pulls the point's
//! out handle to the pointer and its in handle the other way, so the curve
//! runs smoothly through it; a press and release without a drag leaves a
//! sharp corner. A press back on the first point closes the path, and a
//! drag from there bends the closing curve. An open path is finished by
//! the caller (Enter, or leaving the pen).
//!
//! Points here are canvas pixels from the top left, y down, as the preview
//! measures them. [`PenPath::finish`] turns them into the shape's own
//! frame for a [`cutlass_session::Request::DrawShape`].

use cutlass_model::ShapePoint;

/// How far, in canvas pixels, a drag has to go before it pulls handles
/// rather than leaving a corner.
pub const PEN_DRAG_MIN: f64 = 2.0;

/// What a press did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PenPress {
    /// A new point went down.
    Added,
    /// The press landed on the first point and closed the path.
    Closed,
}

/// A pen path being drawn.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PenPath {
    points: Vec<ShapePoint>,
    /// The point whose handles a drag pulls, from its press to its release.
    pulling: Option<usize>,
    closed: bool,
}

impl PenPath {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The points so far, canvas pixels from the top left, y down; each
    /// handle is an offset from its point.
    #[must_use]
    pub fn points(&self) -> &[ShapePoint] {
        &self.points
    }

    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Whether a drag is pulling a point's handles.
    #[must_use]
    pub fn is_pulling(&self) -> bool {
        self.pulling.is_some()
    }

    /// Whether `at` is within `reach` canvas pixels of the first point, so
    /// a press there would close the path.
    #[must_use]
    pub fn closes_at(&self, at: [f64; 2], reach: f64) -> bool {
        self.points.len() >= 2
            && !self.closed
            && self
                .points
                .first()
                .is_some_and(|first| distance(first.at, at) <= reach)
    }

    /// A press at `at`. Within `reach` canvas pixels of the first point, it
    /// closes the path; otherwise it adds a point. Nothing happens once the
    /// path is closed.
    pub fn press(&mut self, at: [f64; 2], reach: f64) -> Option<PenPress> {
        if self.closed {
            return None;
        }
        if self.closes_at(at, reach) {
            self.closed = true;
            self.pulling = Some(0);
            return Some(PenPress::Closed);
        }
        self.points.push(ShapePoint {
            at,
            ..ShapePoint::default()
        });
        self.pulling = Some(self.points.len() - 1);
        Some(PenPress::Added)
    }

    /// The pointer, held down since the last press, is at `to`: the point
    /// pressed takes handles out to it and back the other way. A short
    /// drag leaves a corner.
    pub fn drag(&mut self, to: [f64; 2]) {
        let Some(point) = self.pulling.and_then(|index| self.points.get_mut(index)) else {
            return;
        };
        let out = [to[0] - point.at[0], to[1] - point.at[1]];
        if distance([0.0, 0.0], out) < PEN_DRAG_MIN {
            point.handle_in = [0.0, 0.0];
            point.handle_out = [0.0, 0.0];
        } else {
            point.handle_out = out;
            point.handle_in = [-out[0], -out[1]];
        }
    }

    /// The pointer was released.
    pub fn release(&mut self) {
        self.pulling = None;
    }

    /// Take back the last point, or reopen a closed path.
    pub fn undo(&mut self) {
        self.pulling = None;
        if self.closed {
            self.closed = false;
        } else {
            self.points.pop();
        }
    }

    /// The curves drawn so far, each a start, two handles and an end in
    /// canvas pixels, y down; a closed path ends back at its start.
    #[must_use]
    pub fn cubics(&self) -> Vec<[[f64; 2]; 4]> {
        let count = self.points.len();
        let sides = if self.closed {
            count
        } else {
            count.saturating_sub(1)
        };
        (0..sides)
            .map(|side| cubic(self.points[side], self.points[(side + 1) % count]))
            .collect()
    }

    /// The curve the next press at `hover` would add, from the last point,
    /// for the preview to show ahead of the press.
    #[must_use]
    pub fn next_cubic(&self, hover: [f64; 2]) -> Option<[[f64; 2]; 4]> {
        if self.closed || self.pulling.is_some() {
            return None;
        }
        let last = *self.points.last()?;
        let end = ShapePoint {
            at: hover,
            ..ShapePoint::default()
        };
        Some(cubic(last, end))
    }

    /// The path in the shape's frame, ready for a `DrawShape`: canvas
    /// pixels from the middle of a `canvas` sized canvas, y up, and whether
    /// it is closed. None until there are two points.
    #[must_use]
    pub fn finish(&self, canvas: (i32, i32)) -> Option<(Vec<ShapePoint>, bool)> {
        if self.points.len() < 2 {
            return None;
        }
        let half = |side: i32| f64::from(side.max(1)) / 2.0;
        let (half_w, half_h) = (half(canvas.0), half(canvas.1));
        let flip = |d: [f64; 2]| [d[0], -d[1]];
        let points = self
            .points
            .iter()
            .map(|point| ShapePoint {
                at: [point.at[0] - half_w, half_h - point.at[1]],
                handle_in: flip(point.handle_in),
                handle_out: flip(point.handle_out),
                round: 0.0,
            })
            .collect();
        Some((points, self.closed))
    }
}

fn cubic(from: ShapePoint, to: ShapePoint) -> [[f64; 2]; 4] {
    let add = |p: [f64; 2], d: [f64; 2]| [p[0] + d[0], p[1] + d[1]];
    [
        from.at,
        add(from.at, from.handle_out),
        add(to.at, to.handle_in),
        to.at,
    ]
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clicks_make_corners_and_a_drag_pulls_mirrored_handles() {
        let mut pen = PenPath::new();
        assert_eq!(pen.press([100.0, 100.0], 10.0), Some(PenPress::Added));
        pen.drag([101.0, 100.0]);
        pen.release();
        assert_eq!(pen.points()[0].handle_out, [0.0, 0.0]);

        pen.press([300.0, 100.0], 10.0);
        pen.drag([340.0, 130.0]);
        assert!(pen.is_pulling());
        pen.release();
        let bent = pen.points()[1];
        assert_eq!(bent.handle_out, [40.0, 30.0]);
        assert_eq!(bent.handle_in, [-40.0, -30.0]);
        assert_eq!(pen.cubics().len(), 1);
        assert_eq!(pen.cubics()[0][2], [260.0, 70.0]);
        assert_eq!(pen.next_cubic([300.0, 300.0]).unwrap()[3], [300.0, 300.0]);

        pen.undo();
        assert_eq!(pen.points().len(), 1);
    }

    #[test]
    fn a_press_on_the_first_point_closes_and_finish_centres_y_up() {
        let mut pen = PenPath::new();
        pen.press([960.0, 540.0], 10.0);
        pen.release();
        assert!(
            !pen.closes_at([962.0, 540.0], 10.0),
            "one point cannot close"
        );
        pen.press([1160.0, 540.0], 10.0);
        pen.release();
        pen.press([1160.0, 340.0], 10.0);
        pen.drag([1180.0, 340.0]);
        pen.release();
        assert_eq!(pen.press([963.0, 544.0], 10.0), Some(PenPress::Closed));
        pen.release();
        assert!(pen.is_closed());
        assert_eq!(pen.cubics().len(), 3);
        assert_eq!(pen.press([0.0, 0.0], 10.0), None);

        let (points, closed) = pen.finish((1920, 1080)).unwrap();
        assert!(closed);
        assert_eq!(points[0].at, [0.0, 0.0]);
        assert_eq!(points[2].at, [200.0, 200.0]);
        assert_eq!(points[2].handle_out, [20.0, 0.0]);

        pen.undo();
        assert!(!pen.is_closed());
        assert_eq!(PenPath::new().finish((1920, 1080)), None);
    }
}
