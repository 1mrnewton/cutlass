//! Outlines from outside a Lottie file, cut into the edges a [`Scene`]
//! fills, with the same flattening, strokes and dashes the player uses.
//!
//! The editor's shapes are drawn this way, so a shape and a Lottie sticker
//! share one rasterizer and one GPU pass.
//!
//! [`Scene`]: crate::Scene

use crate::geom::{Affine, Contour, Point};
use crate::scene::Edge;
use crate::shape::{Cap, Join};
use crate::stroke::{Pen as StrokePen, Polyline, dash, flatten, stroke};

/// How far a flattened curve may stray from the true one, in pixels.
const TOLERANCE: f64 = 0.2;

/// How an open line ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineCap {
    Butt,
    Round,
    Square,
}

/// How a stroke turns a corner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineJoin {
    Miter,
    Round,
    Bevel,
}

/// A stroke's pen: its full width in pixels, its ends and its corners.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pen {
    pub width: f64,
    pub cap: LineCap,
    pub join: LineJoin,
    /// How far a miter may reach, as a multiple of half the width.
    pub miter_limit: f64,
}

/// Cubics end to end, in output pixels with y down. A closed outline joins
/// its end back to its start.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outline {
    /// Each cubic's start, two handles and end.
    pub cubics: Vec<[[f64; 2]; 4]>,
    pub closed: bool,
}

impl Outline {
    fn contour(&self) -> Contour {
        let mut cubics: Vec<[Point; 4]> = self
            .cubics
            .iter()
            .map(|cubic| cubic.map(|[x, y]| Point::new(x, y)))
            .collect();
        // A closed contour carries its closing segment.
        if self.closed
            && let (Some(first), Some(last)) = (cubics.first(), cubics.last())
            && first[0] != last[3]
        {
            cubics.push(crate::geom::line(last[3], first[0]));
        }
        Contour {
            cubics,
            closed: self.closed,
        }
    }
}

/// The edges that fill `outlines`. Use them with the non-zero rule when
/// the outlines wind the same way.
#[must_use]
pub fn fill_edges(outlines: &[Outline]) -> Vec<Edge> {
    let polygons = outlines
        .iter()
        .map(|outline| flatten(&outline.contour(), Affine::IDENTITY, TOLERANCE).points);
    polygon_edges(polygons)
}

/// The edges that cover `outlines` drawn with `pen`, to fill with the
/// non-zero rule. `dashes` is a pattern of lengths, dash first, and where
/// along the line it starts.
#[must_use]
pub fn stroke_edges(outlines: &[Outline], pen: &Pen, dashes: Option<(&[f64], f64)>) -> Vec<Edge> {
    let mut lines: Vec<Polyline> = outlines
        .iter()
        .map(|outline| flatten(&outline.contour(), Affine::IDENTITY, TOLERANCE))
        .collect();
    if let Some((pattern, offset)) = dashes {
        lines = dash(lines, pattern, offset);
    }
    let inner = StrokePen {
        half: pen.width / 2.0,
        cap: match pen.cap {
            LineCap::Butt => Cap::Butt,
            LineCap::Round => Cap::Round,
            LineCap::Square => Cap::Square,
        },
        join: match pen.join {
            LineJoin::Miter => Join::Miter,
            LineJoin::Round => Join::Round,
            LineJoin::Bevel => Join::Bevel,
        },
        miter: pen.miter_limit,
    };
    let mut polygons = Vec::new();
    for line in &lines {
        stroke(line, &inner, TOLERANCE, &mut polygons);
    }
    polygon_edges(polygons.into_iter())
}

/// Closed polygons as edges. Level edges cross no sample row and are left
/// out.
fn polygon_edges(polygons: impl Iterator<Item = Vec<Point>>) -> Vec<Edge> {
    let mut edges = Vec::new();
    for polygon in polygons {
        let mapped: Vec<(f32, f32)> = polygon
            .iter()
            .filter(|point| point.is_finite())
            .map(|point| (point.x as f32, point.y as f32))
            .collect();
        for index in 0..mapped.len() {
            let (x0, y0) = mapped[index];
            let (x1, y1) = mapped[(index + 1) % mapped.len()];
            if y0 != y1 && x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite() {
                edges.push(Edge { x0, y0, x1, y1 });
            }
        }
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{Bounds, Fill, FillRule, Item, Paint, Scene};

    fn square() -> Outline {
        let corner = |x: f64, y: f64| [x, y];
        let side = |a: [f64; 2], b: [f64; 2]| {
            [
                a,
                [a[0] + (b[0] - a[0]) / 3.0, a[1] + (b[1] - a[1]) / 3.0],
                b,
                b,
            ]
        };
        let (a, b, c, d) = (
            corner(2.0, 2.0),
            corner(8.0, 2.0),
            corner(8.0, 8.0),
            corner(2.0, 8.0),
        );
        Outline {
            cubics: vec![side(a, b), side(b, c), side(c, d)],
            closed: true,
        }
    }

    fn drawn(edges: Vec<Edge>) -> Vec<u8> {
        let bounds = Bounds::of(&edges, 10, 10).unwrap();
        crate::rasterize(&Scene {
            width: 10,
            height: 10,
            items: vec![Item::Fill(Fill {
                edges,
                rule: FillRule::NonZero,
                paint: Paint::Solid([1.0, 1.0, 1.0, 1.0]),
                bounds,
            })],
        })
    }

    #[test]
    fn a_closed_outline_fills_inside_and_its_stroke_rings_it() {
        let fill = drawn(fill_edges(&[square()]));
        let alpha = |bytes: &[u8], x: usize, y: usize| bytes[(y * 10 + x) * 4 + 3];
        assert_eq!(alpha(&fill, 5, 5), 255);
        assert_eq!(alpha(&fill, 0, 0), 0);

        let pen = Pen {
            width: 2.0,
            cap: LineCap::Butt,
            join: LineJoin::Miter,
            miter_limit: 4.0,
        };
        let ring = drawn(stroke_edges(&[square()], &pen, None));
        assert_eq!(alpha(&ring, 2, 5), 255);
        assert_eq!(alpha(&ring, 5, 5), 0);
    }
}
