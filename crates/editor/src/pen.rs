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

/// How far, in canvas pixels, a freehand stroke may stray from the line
/// that stands in for it before a point is kept.
pub const FREEHAND_TOLERANCE: f64 = 2.0;

/// A finger's freehand stroke through `trace`, canvas pixels from the top
/// left, y down, as a smooth open path in the shape's frame for a
/// `DrawShape` on a `canvas` sized canvas: the trace thinned to the points
/// that shape it, each with handles that carry the curve smoothly through.
/// None for a stroke that does not go anywhere.
#[must_use]
pub fn freehand(trace: &[[f64; 2]], canvas: (i32, i32)) -> Option<Vec<ShapePoint>> {
    let trace: Vec<[f64; 2]> = trace
        .iter()
        .copied()
        .filter(|point| point.iter().all(|value| value.is_finite()))
        .collect();
    let mut kept = vec![true; trace.len()];
    thin(&trace, 0, trace.len().checked_sub(1)?, &mut kept);
    let points: Vec<[f64; 2]> = trace
        .iter()
        .zip(&kept)
        .filter(|(_, keep)| **keep)
        .map(|(point, _)| *point)
        .collect();
    let first = *trace.first()?;
    let reach = trace
        .iter()
        .map(|point| distance(first, *point))
        .fold(0.0, f64::max);
    if points.len() < 2 || reach < PEN_DRAG_MIN {
        return None;
    }
    let mut path = PenPath::new();
    path.points = points
        .iter()
        .enumerate()
        .map(|(index, at)| {
            // A Catmull-Rom tangent: a sixth of the way from the point
            // before to the point after.
            let before = points[index.saturating_sub(1)];
            let after = points[(index + 1).min(points.len() - 1)];
            let tangent = [(after[0] - before[0]) / 6.0, (after[1] - before[1]) / 6.0];
            ShapePoint {
                at: *at,
                handle_in: [-tangent[0], -tangent[1]],
                handle_out: tangent,
                round: 0.0,
            }
        })
        .collect();
    path.finish(canvas).map(|(points, _)| points)
}

/// Mark the points between `from` and `to` to drop where they sit within
/// [`FREEHAND_TOLERANCE`] of the line between them (Ramer-Douglas-Peucker).
fn thin(trace: &[[f64; 2]], from: usize, to: usize, kept: &mut [bool]) {
    if to <= from + 1 {
        return;
    }
    let (a, b) = (trace[from], trace[to]);
    let length = distance(a, b);
    let off = |p: [f64; 2]| {
        if length < f64::EPSILON {
            distance(a, p)
        } else {
            ((b[0] - a[0]) * (a[1] - p[1]) - (a[0] - p[0]) * (b[1] - a[1])).abs() / length
        }
    };
    let (far, gap) =
        (from + 1..to)
            .map(|index| (index, off(trace[index])))
            .fold(
                (from, 0.0),
                |best, next| if next.1 > best.1 { next } else { best },
            );
    if gap > FREEHAND_TOLERANCE {
        thin(trace, from, far, kept);
        thin(trace, far, to, kept);
    } else {
        kept[from + 1..to].iter_mut().for_each(|keep| *keep = false);
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
    fn a_freehand_stroke_keeps_the_points_that_shape_it_and_runs_smoothly() {
        // A straight stroke thins to its two ends.
        let line: Vec<[f64; 2]> = (0..=50).map(|x| [f64::from(x) * 4.0, 100.0]).collect();
        let straight = freehand(&line, (400, 200)).unwrap();
        assert_eq!(straight.len(), 2);
        assert_eq!(straight[0].at, [-200.0, 0.0]);
        assert_eq!(straight[1].at, [0.0, 0.0]);

        // A V keeps its bend, and the bend's handles point along the stroke.
        let mut vee: Vec<[f64; 2]> = (0..=20)
            .map(|i| [f64::from(i) * 5.0, f64::from(i) * 5.0])
            .collect();
        vee.extend((1..=20).map(|i| [100.0 + f64::from(i) * 5.0, 100.0 - f64::from(i) * 5.0]));
        let bent = freehand(&vee, (200, 200)).unwrap();
        assert_eq!(bent.len(), 3);
        assert_eq!(bent[1].at, [0.0, 0.0]);
        assert_eq!(
            bent[1].handle_out,
            [-bent[1].handle_in[0], -bent[1].handle_in[1]]
        );
        assert!(bent[1].handle_out[0] > 0.0);

        // A tap, or nothing at all, draws nothing.
        assert!(freehand(&[[10.0, 10.0], [11.0, 10.5]], (100, 100)).is_none());
        assert!(freehand(&[], (100, 100)).is_none());
        assert!(freehand(&[[f64::NAN, 0.0]], (100, 100)).is_none());
    }

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
