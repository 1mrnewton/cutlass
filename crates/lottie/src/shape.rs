//! Shape items: paths, rectangles, ellipses, stars and polygons, the paints
//! that draw them, groups, trims and repeaters.
//!
//! The first item in a list is on top. A paint draws every path above it in
//! its list, those inside groups above it included. A trim or a repeater
//! changes everything above it in its list. Parsing folds a repeater around
//! the items above it, so evaluation meets one node that holds them.

use serde_json::Value;

use crate::LottieError;
use crate::geom::{Contour, Cubic, Point, line};
use crate::limits::Limits;
use crate::prop::{Prop, number_prop, numbers, shape_prop};
use crate::transform::Transform;

/// A Bezier path as Lottie stores it: vertices with tangents relative to
/// them.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct PathShape {
    pub(crate) closed: bool,
    pub(crate) vertices: Vec<Point>,
    pub(crate) in_tangents: Vec<Point>,
    pub(crate) out_tangents: Vec<Point>,
}

impl PathShape {
    pub(crate) fn parse(value: &Value, limits: &mut Limits) -> Result<Option<Self>, LottieError> {
        let Value::Object(object) = value else {
            return Ok(None);
        };
        let points = |key: &str, limits: &mut Limits| -> Result<Vec<Point>, LottieError> {
            let Some(Value::Array(items)) = object.get(key) else {
                return Ok(Vec::new());
            };
            limits.path_points(items.len())?;
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                let pair = numbers(item, limits)?.unwrap_or_default();
                let x = pair.first().copied().unwrap_or(0.0);
                let y = pair.get(1).copied().unwrap_or(0.0);
                out.push(Point::new(x, y));
            }
            Ok(out)
        };
        let vertices = points("v", limits)?;
        let mut in_tangents = points("i", limits)?;
        let mut out_tangents = points("o", limits)?;
        in_tangents.resize(vertices.len(), Point::default());
        out_tangents.resize(vertices.len(), Point::default());
        let closed = object.get("c").and_then(Value::as_bool).unwrap_or(false);
        Ok(Some(Self {
            closed,
            vertices,
            in_tangents,
            out_tangents,
        }))
    }

    pub(crate) fn contour(&self) -> Option<Contour> {
        let count = self.vertices.len();
        if count == 0 {
            return None;
        }
        let segment = |from: usize, to: usize| -> Cubic {
            let a = self.vertices[from];
            let b = self.vertices[to];
            [
                a,
                a.add(self.out_tangents[from]),
                b.add(self.in_tangents[to]),
                b,
            ]
        };
        let mut cubics: Vec<Cubic> = (1..count).map(|index| segment(index - 1, index)).collect();
        if self.closed {
            cubics.push(segment(count - 1, 0));
        }
        if cubics.is_empty() {
            // One vertex: a dot, which a round cap can still draw.
            cubics.push(line(self.vertices[0], self.vertices[0]));
        }
        Some(Contour {
            cubics,
            closed: self.closed,
        })
    }
}

/// Straight Bezier vertices into a path.
fn polygon(vertices: Vec<Point>, in_tangents: Vec<Point>, out_tangents: Vec<Point>) -> Contour {
    let shape = PathShape {
        closed: true,
        vertices,
        in_tangents,
        out_tangents,
    };
    shape.contour().unwrap_or_default()
}

/// The handle length of a quarter circle of radius 1.
const KAPPA: f64 = 0.552_284_749_830_793_4;

pub(crate) fn rectangle(center: Point, size: Point, roundness: f64, reversed: bool) -> Contour {
    let (hw, hh) = (size.x.abs() / 2.0, size.y.abs() / 2.0);
    let r = roundness.max(0.0).min(hw).min(hh);
    let (l, t, rr, b) = (center.x - hw, center.y - hh, center.x + hw, center.y + hh);
    let zero = Point::default();
    let (v, i, o) = if r <= 0.0 {
        let v = if reversed {
            vec![
                Point::new(rr, t),
                Point::new(l, t),
                Point::new(l, b),
                Point::new(rr, b),
            ]
        } else {
            vec![
                Point::new(rr, t),
                Point::new(rr, b),
                Point::new(l, b),
                Point::new(l, t),
            ]
        };
        (v, vec![zero; 4], vec![zero; 4])
    } else {
        let k = r * KAPPA;
        // Clockwise from the top of the right edge, each corner an arc.
        let mut v = vec![
            Point::new(rr, t + r),
            Point::new(rr, b - r),
            Point::new(rr - r, b),
            Point::new(l + r, b),
            Point::new(l, b - r),
            Point::new(l, t + r),
            Point::new(l + r, t),
            Point::new(rr - r, t),
        ];
        let mut i = vec![
            Point::new(0.0, -k),
            zero,
            Point::new(k, 0.0),
            zero,
            Point::new(0.0, k),
            zero,
            Point::new(-k, 0.0),
            zero,
        ];
        let mut o = vec![
            zero,
            Point::new(0.0, k),
            zero,
            Point::new(-k, 0.0),
            zero,
            Point::new(0.0, -k),
            zero,
            Point::new(k, 0.0),
        ];
        if reversed {
            v[1..].reverse();
            i[1..].reverse();
            o[1..].reverse();
            std::mem::swap(&mut i, &mut o);
        }
        (v, i, o)
    };
    polygon(v, i, o)
}

pub(crate) fn ellipse(center: Point, size: Point, reversed: bool) -> Contour {
    let (rx, ry) = (size.x / 2.0, size.y / 2.0);
    let (kx, ky) = (rx * KAPPA, ry * KAPPA);
    let (cx, cy) = (center.x, center.y);
    // From the top, clockwise: top, right, bottom, left.
    let mut v = vec![
        Point::new(cx, cy - ry),
        Point::new(cx + rx, cy),
        Point::new(cx, cy + ry),
        Point::new(cx - rx, cy),
    ];
    let mut i = vec![
        Point::new(-kx, 0.0),
        Point::new(0.0, -ky),
        Point::new(kx, 0.0),
        Point::new(0.0, ky),
    ];
    let mut o = vec![
        Point::new(kx, 0.0),
        Point::new(0.0, ky),
        Point::new(-kx, 0.0),
        Point::new(0.0, -ky),
    ];
    if reversed {
        v[1..].reverse();
        i[1..].reverse();
        o[1..].reverse();
        std::mem::swap(&mut i, &mut o);
    }
    polygon(v, i, o)
}

/// A star (`inner` set) or a polygon, its first point straight up before
/// `rotation` degrees. Roundness is a percentage of the arc between points.
pub(crate) struct StarShape {
    pub(crate) center: Point,
    pub(crate) points: f64,
    pub(crate) rotation: f64,
    pub(crate) outer: (f64, f64),
    pub(crate) inner: Option<(f64, f64)>,
    pub(crate) reversed: bool,
}

pub(crate) fn star(star: &StarShape) -> Contour {
    let count = star.points.floor();
    if !(1.0..=1_000.0).contains(&count) {
        return Contour::default();
    }
    let count = count as usize;
    let corners = if star.inner.is_some() {
        count * 2
    } else {
        count
    };
    let direction = if star.reversed { -1.0 } else { 1.0 };
    let step = std::f64::consts::TAU / corners as f64;
    let mut angle = star.rotation.to_radians() - std::f64::consts::FRAC_PI_2;
    let (mut v, mut i, mut o) = (Vec::new(), Vec::new(), Vec::new());
    for index in 0..corners {
        let outer = index % 2 == 0 || star.inner.is_none();
        let (radius, roundness) = if outer {
            star.outer
        } else {
            star.inner.unwrap_or(star.outer)
        };
        let perimeter = if star.inner.is_some() {
            std::f64::consts::TAU * radius / (corners as f64 * 2.0)
        } else {
            std::f64::consts::TAU * radius / (corners as f64 * 4.0)
        };
        let (sin, cos) = angle.sin_cos();
        let point = Point::new(radius * cos, radius * sin);
        let along = if radius == 0.0 {
            Point::default()
        } else {
            Point::new(sin, -cos)
        };
        let handle = along.scale(perimeter * roundness / 100.0 * direction);
        v.push(point.add(star.center));
        i.push(handle);
        o.push(handle.scale(-1.0));
        angle += step * direction;
    }
    polygon(v, i, o)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Cap {
    Butt,
    Round,
    Square,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Join {
    Miter,
    Round,
    Bevel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DashPart {
    Dash,
    Gap,
    Offset,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct StrokeStyle {
    pub(crate) width: Prop<Vec<f64>>,
    pub(crate) cap: Cap,
    pub(crate) join: Join,
    pub(crate) miter: Prop<Vec<f64>>,
    pub(crate) dashes: Vec<(DashPart, Prop<Vec<f64>>)>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GradientStyle {
    pub(crate) radial: bool,
    pub(crate) start: Prop<Vec<f64>>,
    pub(crate) end: Prop<Vec<f64>>,
    pub(crate) highlight: Prop<Vec<f64>>,
    pub(crate) angle: Prop<Vec<f64>>,
    /// The number of color stops; after them come opacity stops.
    pub(crate) count: usize,
    pub(crate) stops: Prop<Vec<f64>>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum PaintStyle {
    Color(Prop<Vec<f64>>),
    Gradient(GradientStyle),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PaintItem {
    pub(crate) style: PaintStyle,
    pub(crate) opacity: Prop<Vec<f64>>,
    /// `None` fills; `Some` strokes.
    pub(crate) stroke: Option<StrokeStyle>,
    pub(crate) even_odd: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Repeater {
    pub(crate) copies: Prop<Vec<f64>>,
    pub(crate) offset: Prop<Vec<f64>>,
    pub(crate) transform: Transform,
    pub(crate) start_opacity: Prop<Vec<f64>>,
    pub(crate) end_opacity: Prop<Vec<f64>>,
    /// Each copy above the one before it.
    pub(crate) above: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Item {
    Group {
        items: Vec<Item>,
        transform: Transform,
    },
    Path {
        shape: Prop<PathShape>,
    },
    Rect {
        position: Prop<Vec<f64>>,
        size: Prop<Vec<f64>>,
        roundness: Prop<Vec<f64>>,
        reversed: bool,
    },
    Ellipse {
        position: Prop<Vec<f64>>,
        size: Prop<Vec<f64>>,
        reversed: bool,
    },
    Star {
        polygon: bool,
        position: Prop<Vec<f64>>,
        points: Prop<Vec<f64>>,
        rotation: Prop<Vec<f64>>,
        outer_radius: Prop<Vec<f64>>,
        outer_roundness: Prop<Vec<f64>>,
        inner_radius: Prop<Vec<f64>>,
        inner_roundness: Prop<Vec<f64>>,
        reversed: bool,
    },
    Paint(PaintItem),
    Trim {
        start: Prop<Vec<f64>>,
        end: Prop<Vec<f64>>,
        offset: Prop<Vec<f64>>,
        /// Every path in turn as one run, rather than each path alone.
        one_run: bool,
    },
    Repeat {
        items: Vec<Item>,
        repeater: Box<Repeater>,
    },
}

fn prop_or(
    object: &serde_json::Map<String, Value>,
    key: &str,
    fallback: f64,
    limits: &mut Limits,
) -> Result<Prop<Vec<f64>>, LottieError> {
    Ok(number_prop(object.get(key), limits)?.unwrap_or_else(|| Prop::constant(fallback)))
}

fn int(object: &serde_json::Map<String, Value>, key: &str) -> Option<i64> {
    object
        .get(key)
        .and_then(Value::as_f64)
        .map(|value| value as i64)
}

fn hidden(object: &serde_json::Map<String, Value>) -> bool {
    object.get("hd").and_then(Value::as_bool).unwrap_or(false)
}

/// The items of one list, a group's `tr` taken out as its transform.
pub(crate) fn parse_items(
    value: Option<&Value>,
    depth: usize,
    limits: &mut Limits,
) -> Result<(Vec<Item>, Option<Transform>), LottieError> {
    Limits::depth(depth)?;
    let Some(Value::Array(list)) = value else {
        return Ok((Vec::new(), None));
    };
    limits.shapes(list.len())?;
    let mut items = Vec::new();
    let mut transform = None;
    for entry in list {
        let Value::Object(object) = entry else {
            continue;
        };
        if hidden(object) {
            continue;
        }
        let kind = object.get("ty").and_then(Value::as_str).unwrap_or("");
        let reversed = int(object, "d") == Some(3);
        let item = match kind {
            "tr" => {
                transform = Some(Transform::parse(entry, limits)?);
                continue;
            }
            "gr" => {
                let (inner, own) = parse_items(object.get("it"), depth + 1, limits)?;
                Item::Group {
                    items: inner,
                    transform: own.unwrap_or_default(),
                }
            }
            "sh" => match shape_prop(object.get("ks"), limits)? {
                Some(shape) => Item::Path { shape },
                None => continue,
            },
            "rc" => Item::Rect {
                position: prop_or(object, "p", 0.0, limits)?,
                size: prop_or(object, "s", 0.0, limits)?,
                roundness: prop_or(object, "r", 0.0, limits)?,
                reversed,
            },
            "el" => Item::Ellipse {
                position: prop_or(object, "p", 0.0, limits)?,
                size: prop_or(object, "s", 0.0, limits)?,
                reversed,
            },
            "sr" => Item::Star {
                polygon: int(object, "sy") == Some(2),
                position: prop_or(object, "p", 0.0, limits)?,
                points: prop_or(object, "pt", 5.0, limits)?,
                rotation: prop_or(object, "r", 0.0, limits)?,
                outer_radius: prop_or(object, "or", 0.0, limits)?,
                outer_roundness: prop_or(object, "os", 0.0, limits)?,
                inner_radius: prop_or(object, "ir", 0.0, limits)?,
                inner_roundness: prop_or(object, "is", 0.0, limits)?,
                reversed,
            },
            "fl" | "st" | "gf" | "gs" => Item::Paint(paint(object, kind, limits)?),
            "tm" => Item::Trim {
                start: prop_or(object, "s", 0.0, limits)?,
                end: prop_or(object, "e", 100.0, limits)?,
                offset: prop_or(object, "o", 0.0, limits)?,
                one_run: int(object, "m") == Some(2),
            },
            "rp" => {
                let transform = object.get("tr");
                let opacity = |key: &str, limits: &mut Limits| -> Result<_, LottieError> {
                    Ok(
                        number_prop(transform.and_then(|value| value.get(key)), limits)?
                            .unwrap_or_else(|| Prop::constant(100.0)),
                    )
                };
                let repeater = Repeater {
                    copies: prop_or(object, "c", 1.0, limits)?,
                    offset: prop_or(object, "o", 0.0, limits)?,
                    transform: match transform {
                        Some(value) => Transform::parse(value, limits)?,
                        None => Transform::default(),
                    },
                    start_opacity: opacity("so", limits)?,
                    end_opacity: opacity("eo", limits)?,
                    above: int(object, "m") != Some(2),
                };
                let above = std::mem::take(&mut items);
                Item::Repeat {
                    items: above,
                    repeater: Box::new(repeater),
                }
            }
            _ => continue,
        };
        items.push(item);
    }
    Ok((items, transform))
}

fn paint(
    object: &serde_json::Map<String, Value>,
    kind: &str,
    limits: &mut Limits,
) -> Result<PaintItem, LottieError> {
    let style = if kind == "fl" || kind == "st" {
        PaintStyle::Color(prop_or(object, "c", 0.0, limits)?)
    } else {
        let count = object
            .get("g")
            .and_then(|g| g.get("p"))
            .and_then(Value::as_f64)
            .filter(|count| *count >= 0.0 && *count <= 1_000.0)
            .unwrap_or(0.0) as usize;
        let stops = number_prop(object.get("g").and_then(|g| g.get("k")), limits)?
            .unwrap_or(Prop::Static(Vec::new()));
        PaintStyle::Gradient(GradientStyle {
            radial: int(object, "t") == Some(2),
            start: prop_or(object, "s", 0.0, limits)?,
            end: prop_or(object, "e", 0.0, limits)?,
            highlight: prop_or(object, "h", 0.0, limits)?,
            angle: prop_or(object, "a", 0.0, limits)?,
            count,
            stops,
        })
    };
    let stroke = if kind == "st" || kind == "gs" {
        let mut dashes = Vec::new();
        if let Some(Value::Array(list)) = object.get("d") {
            limits.shapes(list.len())?;
            for dash in list {
                let part = match dash.get("n").and_then(Value::as_str) {
                    Some("d") => DashPart::Dash,
                    Some("g") => DashPart::Gap,
                    Some("o") => DashPart::Offset,
                    _ => continue,
                };
                if let Some(value) = number_prop(dash.get("v"), limits)? {
                    dashes.push((part, value));
                }
            }
        }
        Some(StrokeStyle {
            width: prop_or(object, "w", 1.0, limits)?,
            cap: match int(object, "lc") {
                Some(2) => Cap::Round,
                Some(3) => Cap::Square,
                _ => Cap::Butt,
            },
            join: match int(object, "lj") {
                Some(2) => Join::Round,
                Some(3) => Join::Bevel,
                _ => Join::Miter,
            },
            miter: match object.get("ml2") {
                Some(value) => number_prop(Some(value), limits)?,
                None => None,
            }
            .unwrap_or(Prop::constant(
                object.get("ml").and_then(Value::as_f64).unwrap_or(4.0),
            )),
            dashes,
        })
    } else {
        None
    };
    Ok(PaintItem {
        style,
        opacity: prop_or(object, "o", 100.0, limits)?,
        stroke,
        even_odd: int(object, "r") == Some(2),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(contour: &Contour) -> (f64, f64, f64, f64) {
        let mut b = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for cubic in &contour.cubics {
            let p = cubic[0];
            b = (b.0.min(p.x), b.1.min(p.y), b.2.max(p.x), b.3.max(p.y));
        }
        b
    }

    #[test]
    fn a_rectangle_starts_at_the_top_right_and_runs_clockwise() {
        let rect = rectangle(Point::new(10.0, 10.0), Point::new(20.0, 10.0), 0.0, false);
        assert_eq!(rect.cubics.len(), 4);
        assert_eq!(rect.cubics[0][0], Point::new(20.0, 5.0));
        assert_eq!(rect.cubics[0][3], Point::new(20.0, 15.0));
        assert_eq!(bounds(&rect), (0.0, 5.0, 20.0, 15.0));
        let back = rectangle(Point::new(10.0, 10.0), Point::new(20.0, 10.0), 0.0, true);
        assert_eq!(back.cubics[0][3], Point::new(0.0, 5.0));
        assert!((rect.length() - 60.0).abs() < 1e-6);
    }

    #[test]
    fn a_round_rectangle_clamps_its_radius_to_half_the_short_side() {
        let round = rectangle(Point::default(), Point::new(20.0, 10.0), 50.0, false);
        assert_eq!(round.cubics.len(), 8);
        // A stadium: two half circles of radius 5 and two 10 long sides.
        let expected = 2.0 * 10.0 + std::f64::consts::TAU * 5.0;
        assert!(
            (round.length() - expected).abs() < 0.05,
            "{}",
            round.length()
        );
    }

    #[test]
    fn an_ellipse_starts_at_the_top_and_is_close_to_a_circle() {
        let circle = ellipse(Point::new(0.0, 0.0), Point::new(20.0, 20.0), false);
        assert_eq!(circle.cubics[0][0], Point::new(0.0, -10.0));
        assert_eq!(circle.cubics[0][3], Point::new(10.0, 0.0));
        assert!((circle.length() - std::f64::consts::TAU * 10.0).abs() < 0.05);
        let back = ellipse(Point::new(0.0, 0.0), Point::new(20.0, 20.0), true);
        assert_eq!(back.cubics[0][3], Point::new(-10.0, 0.0));
    }

    #[test]
    fn a_star_alternates_radii_and_a_polygon_keeps_one() {
        let five = star(&StarShape {
            center: Point::default(),
            points: 5.0,
            rotation: 0.0,
            outer: (10.0, 0.0),
            inner: Some((4.0, 0.0)),
            reversed: false,
        });
        assert_eq!(five.cubics.len(), 10);
        let top = five.cubics[0][0];
        assert!(top.x.abs() < 1e-9 && (top.y + 10.0).abs() < 1e-9);
        assert!((five.cubics[1][0].length() - 4.0).abs() < 1e-9);
        let hexagon = star(&StarShape {
            center: Point::default(),
            points: 6.0,
            rotation: 0.0,
            outer: (10.0, 0.0),
            inner: None,
            reversed: false,
        });
        assert_eq!(hexagon.cubics.len(), 6);
        assert!((hexagon.length() - 60.0).abs() < 1e-6);
        let none = star(&StarShape {
            center: Point::default(),
            points: f64::INFINITY,
            rotation: 0.0,
            outer: (10.0, 0.0),
            inner: None,
            reversed: false,
        });
        assert!(none.cubics.is_empty());
    }

    #[test]
    fn a_repeater_holds_the_items_above_it() {
        let list = serde_json::json!([
            {"ty": "rc", "p": {"k": [0, 0]}, "s": {"k": [4, 4]}},
            {"ty": "fl", "c": {"k": [1, 0, 0, 1]}},
            {"ty": "rp", "c": {"k": 3}, "tr": {"p": {"k": [10, 0]}}},
            {"ty": "st", "c": {"k": [0, 0, 1, 1]}, "w": {"k": 2}},
            {"ty": "tr", "p": {"k": [5, 5]}},
            {"ty": "zz"},
            {"ty": "el", "hd": true}
        ]);
        let (items, transform) = parse_items(Some(&list), 0, &mut Limits::default()).unwrap();
        assert!(transform.is_some());
        assert_eq!(items.len(), 2);
        let Item::Repeat { items: inner, .. } = &items[0] else {
            panic!("not a repeat: {:?}", items[0]);
        };
        assert_eq!(inner.len(), 2);
        assert!(matches!(
            &items[1],
            Item::Paint(PaintItem {
                stroke: Some(_),
                ..
            })
        ));
    }
}
