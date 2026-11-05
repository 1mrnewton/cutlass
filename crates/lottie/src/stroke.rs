//! Strokes as fills: each line segment, join and cap becomes a small
//! polygon, all wound the same way, so their non-zero union is the stroke.
//! Dashes cut the lines first.

use crate::geom::{Affine, Contour, Point, flatten_cubic, transform_cubic};
use crate::shape::{Cap, Join};

/// A line through points; `corners` marks the points where cubics meet,
/// which take the pen's join. The others are cuts inside a curve.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Polyline {
    pub(crate) points: Vec<Point>,
    pub(crate) corners: Vec<bool>,
    pub(crate) closed: bool,
}

impl Polyline {
    fn push(&mut self, point: Point, corner: bool) {
        if self.points.last() == Some(&point) {
            if let Some(last) = self.corners.last_mut() {
                *last |= corner;
            }
            return;
        }
        self.points.push(point);
        self.corners.push(corner);
    }
}

/// `contour` mapped by `map` and cut into lines within `tolerance`.
pub(crate) fn flatten(contour: &Contour, map: Affine, tolerance: f64) -> Polyline {
    let mut line = Polyline {
        closed: contour.closed,
        ..Polyline::default()
    };
    let mut cut = Vec::new();
    for cubic in &contour.cubics {
        let cubic = transform_cubic(map, cubic);
        if !cubic.iter().all(|point| point.is_finite()) {
            continue;
        }
        if line.points.is_empty() {
            line.push(cubic[0], true);
        }
        cut.clear();
        flatten_cubic(&cubic, tolerance, &mut cut);
        let last = cut.len().saturating_sub(1);
        for (index, point) in cut.iter().enumerate() {
            line.push(*point, index == last);
        }
    }
    if line.closed && line.points.len() > 1 && line.points.first() == line.points.last() {
        line.points.pop();
        line.corners.pop();
        if let Some(first) = line.corners.first_mut() {
            *first = true;
        }
    }
    line
}

/// The pen a stroke is drawn with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Pen {
    pub(crate) half: f64,
    pub(crate) cap: Cap,
    pub(crate) join: Join,
    pub(crate) miter: f64,
}

/// The most dashes one stroke is cut into; past it the stroke is solid.
const MAX_DASHES: f64 = 100_000.0;

/// `lines` cut by a dash pattern of lengths, dash first, from `offset`.
pub(crate) fn dash(lines: Vec<Polyline>, pattern: &[f64], offset: f64) -> Vec<Polyline> {
    let mut pattern: Vec<f64> = pattern.to_vec();
    if pattern.len() % 2 == 1 {
        pattern.extend_from_within(..);
    }
    let period: f64 = pattern.iter().sum();
    if pattern.is_empty()
        || !period.is_finite()
        || period <= 0.0
        || pattern
            .iter()
            .any(|length| *length < 0.0 || !length.is_finite())
    {
        return lines;
    }
    let total: f64 = lines.iter().map(polyline_length).sum();
    if !total.is_finite() || total / period > MAX_DASHES / pattern.len() as f64 {
        return lines;
    }
    let mut out = Vec::new();
    for line in lines {
        let mut points = line.points.clone();
        let mut corners = line.corners.clone();
        if line.closed
            && let Some(first) = points.first().copied()
        {
            points.push(first);
            corners.push(true);
        }
        // Where in the pattern the line starts.
        let mut index = 0;
        let mut remaining = pattern[0];
        let mut phase = if offset.is_finite() {
            offset.rem_euclid(period)
        } else {
            0.0
        };
        while phase > 0.0 {
            if phase >= remaining {
                phase -= remaining;
                index = (index + 1) % pattern.len();
                remaining = pattern[index];
            } else {
                remaining -= phase;
                phase = 0.0;
            }
        }
        let mut piece = Polyline::default();
        if index % 2 == 0
            && let Some(first) = points.first()
        {
            piece.push(*first, false);
        }
        for pair in 0..points.len().saturating_sub(1) {
            let (from, to) = (points[pair], points[pair + 1]);
            let length = to.sub(from).length();
            let mut at = 0.0;
            while length - at > remaining {
                at += remaining;
                let point = from.lerp(to, at / length);
                if index % 2 == 0 {
                    piece.push(point, false);
                    if piece.points.len() > 1 {
                        out.push(std::mem::take(&mut piece));
                    } else {
                        piece = Polyline::default();
                    }
                } else {
                    piece = Polyline::default();
                    piece.push(point, false);
                }
                index = (index + 1) % pattern.len();
                remaining = pattern[index];
            }
            remaining -= length - at;
            if index % 2 == 0 {
                piece.push(to, corners[pair + 1] && pair + 2 < points.len());
            }
        }
        if index % 2 == 0 && piece.points.len() > 1 {
            out.push(piece);
        }
    }
    out
}

fn polyline_length(line: &Polyline) -> f64 {
    let mut length: f64 = line
        .points
        .windows(2)
        .map(|pair| pair[1].sub(pair[0]).length())
        .sum();
    if line.closed
        && let (Some(first), Some(last)) = (line.points.first(), line.points.last())
    {
        length += first.sub(*last).length();
    }
    length
}

/// The polygons that cover `line` drawn with `pen`, each wound with a
/// positive area. Arcs keep within `tolerance`.
pub(crate) fn stroke(line: &Polyline, pen: &Pen, tolerance: f64, out: &mut Vec<Vec<Point>>) {
    let h = pen.half;
    if !(h > 0.0 && h.is_finite()) {
        return;
    }
    let points = &line.points;
    let count = points.len();
    if count == 0 {
        return;
    }
    let step = arc_step(h, tolerance);
    if count == 1 {
        // A dot: only a round or a square cap shows.
        let center = points[0];
        match pen.cap {
            Cap::Round => push(out, arc(center, h, 0.0, std::f64::consts::TAU, step)),
            Cap::Square => push(
                out,
                vec![
                    center.add(Point::new(-h, -h)),
                    center.add(Point::new(h, -h)),
                    center.add(Point::new(h, h)),
                    center.add(Point::new(-h, h)),
                ],
            ),
            Cap::Butt => {}
        }
        return;
    }
    let segments = if line.closed { count } else { count - 1 };
    let direction = |index: usize| -> Point {
        let from = points[index % count];
        let to = points[(index + 1) % count];
        let d = to.sub(from);
        let length = d.length();
        if length > 0.0 {
            d.scale(1.0 / length)
        } else {
            Point::default()
        }
    };
    for index in 0..segments {
        let from = points[index];
        let to = points[(index + 1) % count];
        let d = direction(index);
        let n = Point::new(-d.y, d.x).scale(h);
        push(out, vec![from.add(n), to.add(n), to.sub(n), from.sub(n)]);
    }
    let joints = if line.closed { 0..count } else { 1..count - 1 };
    for vertex in joints {
        let before = direction((vertex + count - 1) % count);
        let after = direction(vertex);
        let join = if line.corners.get(vertex).copied().unwrap_or(true) {
            pen.join
        } else {
            Join::Bevel
        };
        joint(out, points[vertex], before, after, join, pen, step);
    }
    if !line.closed {
        let start = direction(0);
        let end = direction(count - 2);
        cap(out, points[0], start.scale(-1.0), pen, step);
        cap(out, points[count - 1], end, pen, step);
    }
}

/// The angle between arc points so the arc keeps within `tolerance`.
fn arc_step(radius: f64, tolerance: f64) -> f64 {
    let ratio = 1.0 - (tolerance / radius).min(1.0);
    let step = 2.0 * ratio.acos();
    step.clamp(std::f64::consts::TAU / 256.0, std::f64::consts::FRAC_PI_2)
}

/// Points round `center` from angle `from` through `sweep`, both ends in.
fn arc(center: Point, radius: f64, from: f64, sweep: f64, step: f64) -> Vec<Point> {
    let pieces = (sweep.abs() / step).ceil().clamp(1.0, 256.0) as usize;
    (0..=pieces)
        .map(|index| {
            let angle = from + sweep * index as f64 / pieces as f64;
            let (sin, cos) = angle.sin_cos();
            center.add(Point::new(cos * radius, sin * radius))
        })
        .collect()
}

fn joint(
    out: &mut Vec<Vec<Point>>,
    vertex: Point,
    before: Point,
    after: Point,
    join: Join,
    pen: &Pen,
    step: f64,
) {
    let turn = before.cross(after);
    if turn.abs() < 1e-9 && before.dot(after) > 0.0 {
        return;
    }
    let side = if turn > 0.0 { -1.0 } else { 1.0 };
    let n0 = Point::new(-before.y, before.x).scale(side);
    let n1 = Point::new(-after.y, after.x).scale(side);
    let h = pen.half;
    let a = vertex.add(n0.scale(h));
    let b = vertex.add(n1.scale(h));
    match join {
        Join::Bevel => push(out, vec![vertex, a, b]),
        Join::Round => {
            let from = n0.y.atan2(n0.x);
            let sweep = n0.cross(n1).atan2(n0.dot(n1));
            let mut wedge = vec![vertex];
            wedge.extend(arc(vertex, h, from, sweep, step));
            push(out, wedge);
        }
        Join::Miter => {
            let middle = n0.add(n1);
            let length = middle.length();
            let cos_half = if length > 0.0 {
                n0.dot(middle.scale(1.0 / length))
            } else {
                0.0
            };
            if cos_half > 0.0 && 1.0 / cos_half <= pen.miter {
                let tip = vertex.add(middle.scale(h / (cos_half * length)));
                push(out, vec![vertex, a, tip, b]);
            } else {
                push(out, vec![vertex, a, b]);
            }
        }
    }
}

/// A cap at `end`, the line arriving along `outward`.
fn cap(out: &mut Vec<Vec<Point>>, end: Point, outward: Point, pen: &Pen, step: f64) {
    let h = pen.half;
    let n = Point::new(-outward.y, outward.x).scale(h);
    match pen.cap {
        Cap::Butt => {}
        Cap::Square => {
            let reach = outward.scale(h);
            push(
                out,
                vec![
                    end.add(n),
                    end.add(n).add(reach),
                    end.sub(n).add(reach),
                    end.sub(n),
                ],
            );
        }
        Cap::Round => {
            let from = n.y.atan2(n.x);
            // From one side, round the outward point, to the other side.
            let sweep = if n.cross(outward) > 0.0 {
                std::f64::consts::PI
            } else {
                -std::f64::consts::PI
            };
            push(out, arc(end, h, from, sweep, step));
        }
    }
}

fn area(polygon: &[Point]) -> f64 {
    let mut sum = 0.0;
    for index in 0..polygon.len() {
        let a = polygon[index];
        let b = polygon[(index + 1) % polygon.len()];
        sum += a.cross(b);
    }
    sum / 2.0
}

fn push(out: &mut Vec<Vec<Point>>, mut polygon: Vec<Point>) {
    let area = area(&polygon);
    if !area.is_finite() || area.abs() < 1e-12 {
        return;
    }
    if area < 0.0 {
        polygon.reverse();
    }
    out.push(polygon);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::line;

    fn open(points: &[(f64, f64)]) -> Polyline {
        Polyline {
            points: points.iter().map(|&(x, y)| Point::new(x, y)).collect(),
            corners: vec![true; points.len()],
            closed: false,
        }
    }

    fn pen(cap: Cap, join: Join) -> Pen {
        Pen {
            half: 2.0,
            cap,
            join,
            miter: 4.0,
        }
    }

    #[test]
    fn every_piece_is_wound_the_same_way() {
        let mut out = Vec::new();
        let zigzag = open(&[
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (5.0, 3.0),
        ]);
        for join in [Join::Miter, Join::Round, Join::Bevel] {
            for cap in [Cap::Butt, Cap::Round, Cap::Square] {
                out.clear();
                stroke(&zigzag, &pen(cap, join), 0.1, &mut out);
                assert!(!out.is_empty());
                assert!(out.iter().all(|polygon| area(polygon) > 0.0));
            }
        }
    }

    #[test]
    fn a_butt_line_is_one_rectangle_and_caps_add_their_area() {
        let segment = open(&[(0.0, 0.0), (10.0, 0.0)]);
        let total = |cap| {
            let mut out = Vec::new();
            stroke(&segment, &pen(cap, Join::Miter), 0.01, &mut out);
            out.iter().map(|polygon| area(polygon)).sum::<f64>()
        };
        assert!((total(Cap::Butt) - 40.0).abs() < 1e-9);
        assert!((total(Cap::Square) - 56.0).abs() < 1e-9);
        let round = total(Cap::Round);
        assert!(
            (round - (40.0 + std::f64::consts::PI * 4.0)).abs() < 0.15,
            "{round}"
        );
    }

    #[test]
    fn a_sharp_miter_past_its_limit_is_bevelled() {
        let sharp = open(&[(0.0, 0.0), (20.0, 0.0), (0.0, 1.0)]);
        let mut miter = Vec::new();
        stroke(&sharp, &pen(Cap::Butt, Join::Miter), 0.1, &mut miter);
        let mut bevel = Vec::new();
        stroke(&sharp, &pen(Cap::Butt, Join::Bevel), 0.1, &mut bevel);
        let sum = |list: &[Vec<Point>]| list.iter().map(|p| area(p)).sum::<f64>();
        assert!((sum(&miter) - sum(&bevel)).abs() < 1e-9);
        let right = open(&[(0.0, 0.0), (20.0, 0.0), (20.0, 20.0)]);
        let mut miter = Vec::new();
        stroke(&right, &pen(Cap::Butt, Join::Miter), 0.1, &mut miter);
        let mut bevel = Vec::new();
        stroke(&right, &pen(Cap::Butt, Join::Bevel), 0.1, &mut bevel);
        // A right angle's miter adds a 2 by 2 square less the bevel's half.
        assert!((sum(&miter) - sum(&bevel) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn dashes_cut_the_line_by_the_pattern_and_offset() {
        let long = open(&[(0.0, 0.0), (10.0, 0.0)]);
        let dashes = dash(vec![long.clone()], &[2.0, 1.0], 0.0);
        assert_eq!(dashes.len(), 4);
        assert_eq!(
            dashes[0].points,
            vec![Point::new(0.0, 0.0), Point::new(2.0, 0.0)]
        );
        assert_eq!(
            dashes[3].points,
            vec![Point::new(9.0, 0.0), Point::new(10.0, 0.0)]
        );
        let shifted = dash(vec![long.clone()], &[2.0, 1.0], 1.0);
        assert_eq!(
            shifted[0].points,
            vec![Point::new(0.0, 0.0), Point::new(1.0, 0.0)]
        );
        // A bad pattern leaves the line whole.
        assert_eq!(dash(vec![long.clone()], &[0.0, 0.0], 0.0).len(), 1);
        assert_eq!(dash(vec![long.clone()], &[-1.0, 2.0], 0.0).len(), 1);
        // Too many dashes leaves the line whole too.
        assert_eq!(dash(vec![long], &[1e-9, 1e-9], 0.0).len(), 1);
    }

    #[test]
    fn flattening_marks_where_cubics_meet() {
        let contour = Contour {
            cubics: vec![
                line(Point::new(0.0, 0.0), Point::new(10.0, 0.0)),
                [
                    Point::new(10.0, 0.0),
                    Point::new(20.0, 0.0),
                    Point::new(20.0, 10.0),
                    Point::new(10.0, 10.0),
                ],
                line(Point::new(10.0, 10.0), Point::new(0.0, 0.0)),
            ],
            closed: true,
        };
        let line = flatten(&contour, Affine::IDENTITY, 0.1);
        assert_eq!(line.points[0], Point::new(0.0, 0.0));
        assert_ne!(line.points.last(), Some(&Point::new(0.0, 0.0)));
        assert!(line.corners[0] && line.corners[1]);
        assert!(line.corners.iter().filter(|corner| !**corner).count() > 3);
    }
}
