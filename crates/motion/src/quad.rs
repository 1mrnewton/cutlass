//! The target's four corners in grey-copy pixels, and the checks on them.

/// Four corners in pixels, in order around the shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Corners(pub(crate) [(f64, f64); 4]);

impl Corners {
    /// The smallest box holding the corners: left, top, right, bottom.
    pub(crate) fn bounds(&self) -> (f64, f64, f64, f64) {
        self.0.iter().fold(
            (
                f64::INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            ),
            |(l, t, r, b), &(x, y)| (l.min(x), t.min(y), r.max(x), b.max(y)),
        )
    }

    /// The area inside the corners.
    pub(crate) fn area(&self) -> f64 {
        let mut twice = 0.0;
        for i in 0..4 {
            let (a, b) = (self.0[i], self.0[(i + 1) % 4]);
            twice += a.0 * b.1 - b.0 * a.1;
        }
        (twice / 2.0).abs()
    }

    pub(crate) fn middle(&self) -> (f64, f64) {
        let sum = self
            .0
            .iter()
            .fold((0.0, 0.0), |s, p| (s.0 + p.0, s.1 + p.1));
        (sum.0 / 4.0, sum.1 / 4.0)
    }

    /// Whether every corner is a number and the shape turns the same way at
    /// each corner: not folded over itself.
    pub(crate) fn is_convex(&self) -> bool {
        if self.0.iter().any(|p| !p.0.is_finite() || !p.1.is_finite()) {
            return false;
        }
        let turns: Vec<f64> = (0..4)
            .map(|i| {
                let (a, b, c) = (self.0[i], self.0[(i + 1) % 4], self.0[(i + 2) % 4]);
                (b.0 - a.0) * (c.1 - b.1) - (b.1 - a.1) * (c.0 - b.0)
            })
            .collect();
        turns.iter().all(|&t| t > 0.0) || turns.iter().all(|&t| t < 0.0)
    }

    /// Whether `x`, `y` is inside a convex shape.
    pub(crate) fn contains(&self, x: f64, y: f64) -> bool {
        let sides: Vec<f64> = (0..4)
            .map(|i| {
                let (a, b) = (self.0[i], self.0[(i + 1) % 4]);
                (b.0 - a.0) * (y - a.1) - (b.1 - a.1) * (x - a.0)
            })
            .collect();
        sides.iter().all(|&s| s >= 0.0) || sides.iter().all(|&s| s <= 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_square_is_convex_and_a_bow_tie_is_not() {
        let square = Corners([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]);
        assert!(square.is_convex());
        assert!((square.area() - 100.0).abs() < 1e-9);
        assert!(square.contains(5.0, 5.0) && !square.contains(11.0, 5.0));
        assert_eq!(square.bounds(), (0.0, 0.0, 10.0, 10.0));
        let bow = Corners([(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)]);
        assert!(!bow.is_convex());
        let nan = Corners([(f64::NAN, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]);
        assert!(!nan.is_convex());
    }
}
