//! Points, affine maps, cubic contours, flattening, length and trimming.
//!
//! Coordinates are y down, as Lottie and the canvas both are. A contour is a
//! run of cubics; a closed contour carries its closing segment explicitly,
//! so its length and its trim include it.

/// A point or a vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Point {
    pub(crate) x: f64,
    pub(crate) y: f64,
}

impl Point {
    pub(crate) const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub(crate) fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }

    pub(crate) fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }

    pub(crate) fn scale(self, factor: f64) -> Self {
        Self::new(self.x * factor, self.y * factor)
    }

    pub(crate) fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    pub(crate) fn cross(self, other: Self) -> f64 {
        self.x * other.y - self.y * other.x
    }

    pub(crate) fn length(self) -> f64 {
        self.x.hypot(self.y)
    }

    pub(crate) fn lerp(self, other: Self, t: f64) -> Self {
        Self::new(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
        )
    }

    pub(crate) fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

/// `x' = a x + b y + c`, `y' = d x + e y + f`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Affine(pub(crate) [f64; 6]);

impl Affine {
    pub(crate) const IDENTITY: Self = Self([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

    pub(crate) const fn translate(x: f64, y: f64) -> Self {
        Self([1.0, 0.0, x, 0.0, 1.0, y])
    }

    pub(crate) const fn scale(x: f64, y: f64) -> Self {
        Self([x, 0.0, 0.0, 0.0, y, 0.0])
    }

    /// Clockwise on screen, since y is down.
    pub(crate) fn rotate(degrees: f64) -> Self {
        let (sin, cos) = degrees.to_radians().sin_cos();
        Self([cos, -sin, 0.0, sin, cos, 0.0])
    }

    /// `x' = x + y * k`.
    pub(crate) const fn shear_x(k: f64) -> Self {
        Self([1.0, k, 0.0, 0.0, 1.0, 0.0])
    }

    /// `self` after `inner`: `inner` applies first.
    pub(crate) fn then(self, inner: Self) -> Self {
        let [a, b, c, d, e, f] = self.0;
        let [g, h, i, j, k, l] = inner.0;
        Self([
            a * g + b * j,
            a * h + b * k,
            a * i + b * l + c,
            d * g + e * j,
            d * h + e * k,
            d * i + e * l + f,
        ])
    }

    pub(crate) fn apply(self, point: Point) -> Point {
        let [a, b, c, d, e, f] = self.0;
        Point::new(a * point.x + b * point.y + c, d * point.x + e * point.y + f)
    }

    pub(crate) fn determinant(self) -> f64 {
        let [a, b, _, d, e, _] = self.0;
        a * e - b * d
    }

    pub(crate) fn invert(self) -> Option<Self> {
        let [a, b, c, d, e, f] = self.0;
        let det = self.determinant();
        if !det.is_finite() || det.abs() < 1e-12 {
            return None;
        }
        let inv = 1.0 / det;
        Some(Self([
            e * inv,
            -b * inv,
            (b * f - e * c) * inv,
            -d * inv,
            a * inv,
            (d * c - a * f) * inv,
        ]))
    }

    /// The largest stretch this map gives a unit length, at most.
    pub(crate) fn stretch(self) -> f64 {
        let [a, b, _, d, e, _] = self.0;
        a.hypot(d).max(b.hypot(e))
    }

    pub(crate) fn is_finite(self) -> bool {
        self.0.iter().all(|value| value.is_finite())
    }
}

/// One cubic: start, two controls, end.
pub(crate) type Cubic = [Point; 4];

/// Cubics end to end. `closed` joins the end back to the start in a stroke;
/// the closing segment is already among the cubics.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Contour {
    pub(crate) cubics: Vec<Cubic>,
    pub(crate) closed: bool,
}

/// The cubic of a straight line.
pub(crate) fn line(from: Point, to: Point) -> Cubic {
    [from, from.lerp(to, 1.0 / 3.0), from.lerp(to, 2.0 / 3.0), to]
}

pub(crate) fn cubic_at(cubic: &Cubic, t: f64) -> Point {
    let u = 1.0 - t;
    let [p0, p1, p2, p3] = *cubic;
    p0.scale(u * u * u)
        .add(p1.scale(3.0 * u * u * t))
        .add(p2.scale(3.0 * u * t * t))
        .add(p3.scale(t * t * t))
}

/// The cubic between `t0` and `t1`, by de Casteljau.
pub(crate) fn sub_cubic(cubic: &Cubic, t0: f64, t1: f64) -> Cubic {
    let split_left = |c: &Cubic, t: f64| -> Cubic {
        let [p0, p1, p2, p3] = *c;
        let a = p0.lerp(p1, t);
        let b = p1.lerp(p2, t);
        let cc = p2.lerp(p3, t);
        let d = a.lerp(b, t);
        let e = b.lerp(cc, t);
        let f = d.lerp(e, t);
        [p0, a, d, f]
    };
    let split_right = |c: &Cubic, t: f64| -> Cubic {
        let [p0, p1, p2, p3] = *c;
        let a = p0.lerp(p1, t);
        let b = p1.lerp(p2, t);
        let cc = p2.lerp(p3, t);
        let d = a.lerp(b, t);
        let e = b.lerp(cc, t);
        let f = d.lerp(e, t);
        [f, e, cc, p3]
    };
    let left = split_left(cubic, t1);
    if t1 <= 0.0 {
        return [cubic[0]; 4];
    }
    split_right(&left, (t0 / t1).clamp(0.0, 1.0))
}

pub(crate) fn transform_cubic(map: Affine, cubic: &Cubic) -> Cubic {
    cubic.map(|point| map.apply(point))
}

/// Lines through a cubic so no point strays more than `tolerance` from it.
/// The start is not pushed; the end is.
pub(crate) fn flatten_cubic(cubic: &Cubic, tolerance: f64, out: &mut Vec<Point>) {
    let [p0, p1, p2, p3] = *cubic;
    let dd1 = p0.sub(p1.scale(2.0)).add(p2).length();
    let dd2 = p1.sub(p2.scale(2.0)).add(p3).length();
    let bend = dd1.max(dd2);
    let steps = if bend <= f64::EPSILON || !bend.is_finite() {
        1
    } else {
        let wanted = (0.75 * bend / tolerance.max(1e-6)).sqrt().ceil();
        if wanted.is_finite() {
            wanted.clamp(1.0, MAX_STEPS) as usize
        } else {
            MAX_STEPS as usize
        }
    };
    for step in 1..=steps {
        out.push(cubic_at(cubic, step as f64 / steps as f64));
    }
}

/// The most lines one cubic is cut into.
const MAX_STEPS: f64 = 256.0;

/// Steps per cubic when measuring length.
const LENGTH_STEPS: usize = 24;

/// Cumulative lengths at `LENGTH_STEPS` even steps of `t`.
fn length_table(cubic: &Cubic) -> [f64; LENGTH_STEPS + 1] {
    let mut table = [0.0; LENGTH_STEPS + 1];
    let mut previous = cubic[0];
    for (index, slot) in table.iter_mut().enumerate().skip(1) {
        let point = cubic_at(cubic, index as f64 / LENGTH_STEPS as f64);
        *slot = previous.sub(point).length();
        previous = point;
    }
    for index in 1..table.len() {
        table[index] += table[index - 1];
    }
    table
}

pub(crate) fn cubic_length(cubic: &Cubic) -> f64 {
    length_table(cubic)[LENGTH_STEPS]
}

/// The `t` at which a cubic has run `distance` along its length.
fn t_at_length(cubic: &Cubic, distance: f64) -> f64 {
    let table = length_table(cubic);
    let total = table[LENGTH_STEPS];
    if total <= 0.0 || distance <= 0.0 {
        return 0.0;
    }
    if distance >= total {
        return 1.0;
    }
    for index in 1..table.len() {
        if table[index] >= distance {
            let span = table[index] - table[index - 1];
            let within = if span > 0.0 {
                (distance - table[index - 1]) / span
            } else {
                0.0
            };
            return ((index - 1) as f64 + within) / LENGTH_STEPS as f64;
        }
    }
    1.0
}

/// The point at fraction `t` of a cubic's length.
pub(crate) fn point_at_length_fraction(cubic: &Cubic, fraction: f64) -> Point {
    let total = cubic_length(cubic);
    cubic_at(cubic, t_at_length(cubic, fraction.clamp(0.0, 1.0) * total))
}

impl Contour {
    pub(crate) fn length(&self) -> f64 {
        self.cubics.iter().map(cubic_length).sum()
    }

    /// The open piece between `from` and `to`, distances along the contour.
    pub(crate) fn piece(&self, from: f64, to: f64) -> Self {
        let mut cubics = Vec::new();
        let mut start = 0.0;
        for cubic in &self.cubics {
            let length = cubic_length(cubic);
            let end = start + length;
            if end > from && start < to && length > 0.0 {
                let t0 = t_at_length(cubic, (from - start).max(0.0));
                let t1 = t_at_length(cubic, (to - start).min(length));
                if t1 > t0 {
                    cubics.push(sub_cubic(cubic, t0, t1));
                }
            }
            start = end;
        }
        Self {
            cubics,
            closed: false,
        }
    }
}

/// Trim `contours`, taken as one run end to end, to the fractions in
/// `ranges` of its whole length. A piece that runs over a closed contour's
/// start is joined into one open contour.
pub(crate) fn trim_run(contours: &[Contour], ranges: &[(f64, f64)]) -> Vec<Contour> {
    let lengths: Vec<f64> = contours.iter().map(Contour::length).collect();
    let total: f64 = lengths.iter().sum();
    if total <= 0.0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = 0.0;
    for (contour, length) in contours.iter().zip(&lengths) {
        let end = start + length;
        let mut pieces: Vec<(f64, f64)> = Vec::new();
        for &(from, to) in ranges {
            let (from, to) = (from * total, to * total);
            let a = from.max(start);
            let b = to.min(end);
            if b > a {
                pieces.push((a - start, b - start));
            }
        }
        // Over the start of a closed contour: the tail, then the head.
        if contour.closed
            && pieces.len() == 2
            && pieces[0].0 <= 1e-9
            && (pieces[1].1 - length).abs() <= 1e-9
        {
            let mut tail = contour.piece(pieces[1].0, pieces[1].1);
            let head = contour.piece(pieces[0].0, pieces[0].1);
            tail.cubics.extend(head.cubics);
            if !tail.cubics.is_empty() {
                out.push(tail);
            }
        } else {
            for (a, b) in pieces {
                let piece = contour.piece(a, b);
                if !piece.cubics.is_empty() {
                    out.push(piece);
                }
            }
        }
        start = end;
    }
    out
}

/// The ranges, fractions of the length, that trim start, end and offset
/// (all fractions) keep. `None` keeps everything; an empty list keeps
/// nothing.
pub(crate) fn trim_ranges(start: f64, end: f64, offset: f64) -> Option<Vec<(f64, f64)>> {
    let (mut s, mut e) = (start.clamp(0.0, 1.0), end.clamp(0.0, 1.0));
    if s > e {
        std::mem::swap(&mut s, &mut e);
    }
    if e - s >= 1.0 {
        return None;
    }
    if e - s <= 0.0 {
        return Some(Vec::new());
    }
    let shift = if offset.is_finite() {
        offset.rem_euclid(1.0)
    } else {
        0.0
    };
    let (s, e) = (s + shift, e + shift);
    Some(if e <= 1.0 {
        vec![(s, e)]
    } else if s >= 1.0 {
        vec![(s - 1.0, e - 1.0)]
    } else {
        vec![(0.0, e - 1.0), (s, 1.0)]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> Contour {
        let corners = [
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ];
        let cubics = (0..4)
            .map(|index| line(corners[index], corners[(index + 1) % 4]))
            .collect();
        Contour {
            cubics,
            closed: true,
        }
    }

    #[test]
    fn an_affine_inverts_and_composes() {
        let map = Affine::translate(5.0, -2.0)
            .then(Affine::rotate(30.0))
            .then(Affine::scale(2.0, 3.0));
        let back = map.invert().unwrap();
        let point = Point::new(1.5, -4.0);
        let round = back.apply(map.apply(point));
        assert!((round.x - point.x).abs() < 1e-9 && (round.y - point.y).abs() < 1e-9);
        assert!(Affine::scale(0.0, 1.0).invert().is_none());
        // A quarter turn is clockwise on a y-down screen.
        let turned = Affine::rotate(90.0).apply(Point::new(1.0, 0.0));
        assert!(turned.x.abs() < 1e-12 && (turned.y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_line_is_one_step_and_a_curve_is_cut_to_the_tolerance() {
        let mut out = Vec::new();
        flatten_cubic(
            &line(Point::new(0.0, 0.0), Point::new(9.0, 0.0)),
            0.2,
            &mut out,
        );
        assert_eq!(out.len(), 1);
        out.clear();
        let bend = [
            Point::new(0.0, 0.0),
            Point::new(0.0, 100.0),
            Point::new(100.0, 100.0),
            Point::new(100.0, 0.0),
        ];
        flatten_cubic(&bend, 0.2, &mut out);
        let many = out.len();
        assert!((10..=64).contains(&many), "{many}");
        out.clear();
        flatten_cubic(&bend, 2.0, &mut out);
        assert!(out.len() < many);
    }

    #[test]
    fn length_and_pieces_follow_the_square() {
        let square = square();
        assert!((square.length() - 40.0).abs() < 1e-6);
        let piece = square.piece(5.0, 15.0);
        assert_eq!(piece.cubics.len(), 2);
        assert!((piece.length() - 10.0).abs() < 1e-6);
        let start = piece.cubics[0][0];
        assert!((start.x - 5.0).abs() < 1e-6 && start.y.abs() < 1e-6);
        let end = piece.cubics[1][3];
        assert!((end.x - 10.0).abs() < 1e-6 && (end.y - 5.0).abs() < 1e-6);
    }

    #[test]
    fn trim_ranges_wrap_the_offset_and_swap_a_reversed_pair() {
        assert_eq!(trim_ranges(0.0, 1.0, 0.3), None);
        assert_eq!(trim_ranges(0.4, 0.4, 0.0), Some(Vec::new()));
        assert_eq!(trim_ranges(0.5, 0.25, 0.0), Some(vec![(0.25, 0.5)]));
        let wrapped = trim_ranges(0.0, 0.5, 0.75).unwrap();
        assert_eq!(wrapped.len(), 2);
        assert!((wrapped[0].1 - 0.25).abs() < 1e-12 && (wrapped[1].0 - 0.75).abs() < 1e-12);
        assert_eq!(trim_ranges(0.0, 0.5, f64::NAN), Some(vec![(0.0, 0.5)]));
    }

    #[test]
    fn a_trim_over_the_start_of_a_closed_contour_is_one_piece() {
        let square = square();
        let ranges = trim_ranges(0.0, 0.5, 0.75).unwrap();
        let pieces = trim_run(std::slice::from_ref(&square), &ranges);
        assert_eq!(pieces.len(), 1);
        assert!((pieces[0].length() - 20.0).abs() < 1e-6);
        let start = pieces[0].cubics[0][0];
        assert!(start.x.abs() < 1e-6 && (start.y - 10.0).abs() < 1e-6);
    }

    #[test]
    fn a_trim_runs_over_contours_end_to_end() {
        let a = square();
        let b = square();
        let pieces = trim_run(&[a, b], &[(0.25, 0.75)]);
        let total: f64 = pieces.iter().map(Contour::length).sum();
        assert!((total - 40.0).abs() < 1e-6);
        assert_eq!(pieces.len(), 2);
    }
}
