//! A shape material as a vector scene: its shadow, its fill and its stroke,
//! drawn by the Lottie rasterizer on the CPU or its pass on the GPU.
//!
//! The outline is the material's points, each joined to the next by a cubic
//! through their handles; each cubic is flattened within a fifth of a pixel.
//! A sharp corner with rounding is cut back along both sides and joined by a
//! quarter-circle cubic. A line or an arrow is open and drawn by its stroke;
//! an arrow ends in a filled head.
//!
//! The picture is `density` pixels per canvas pixel, with the shape at its
//! centre and room around it for the stroke and the shadow, so the clip's
//! position is the shape's centre.

use cutlass_lottie::{
    Bounds, Edge, Fill, FillRule, Group, Item, LineCap, LineJoin, Outline, Paint, Pen, Scene,
    fill_edges, stroke_edges,
};
use cutlass_model::{MaterialShape, SHAPE_ARROW, SHAPE_LINE};

/// The widest or tallest picture a shape is drawn into.
pub const MAX_SHAPE_EDGE: f64 = 2_048.0;
/// A quarter circle's handle length as a part of its radius.
const ARC_HANDLE: f64 = 0.552_284_7;
/// A miter reaches at most this many half widths past its corner.
const MITER_LIMIT: f64 = 4.0;

/// `material` drawn `density` picture pixels per canvas pixel, with the
/// density the picture was drawn at; it is less than asked when the
/// picture would pass [`MAX_SHAPE_EDGE`]. `None` when there is nothing to
/// draw.
#[must_use]
pub fn shape_scene(material: &MaterialShape, density: f64) -> Option<(Scene, f64)> {
    let points = material.points();
    if points.len() < 2 {
        return None;
    }
    let open = matches!(material.shape_type, SHAPE_LINE | SHAPE_ARROW);
    let arrow = material.shape_type == SHAPE_ARROW;
    let stroke = (material.border_width > 0.0 && material.border_width.is_finite())
        .then_some(material.border_width);
    let fill = (!open)
        .then(|| rgba(&material.color, material.global_alpha))
        .flatten();
    let stroke_paint = stroke.and_then(|_| rgba(&material.border_color, material.border_alpha));
    if fill.is_none() && stroke_paint.is_none() {
        return None;
    }
    let width = stroke.unwrap_or(0.0);
    let head = arrow.then(|| arrow_head(width));
    let shadow = (material.shadow_alpha > 0.0)
        .then(|| rgba(&material.shadow_color, 1.0))
        .flatten()
        .filter(|_| material.shadow_distance.is_finite() && material.shadow_angle.is_finite());

    // Room round the shape: half the stroke (a miter can reach farther),
    // the arrow head, the shadow's fall, and a pixel for smoothing.
    let reach = width / 2.0 * MITER_LIMIT;
    let reach = reach.max(head.map_or(0.0, |(_, half)| half));
    let fall = if shadow.is_some() {
        material.shadow_distance.max(0.0)
    } else {
        0.0
    };
    let margin = reach + fall + 1.0;
    let (low, high) = extent(&points);
    let half_w = low[0].abs().max(high[0].abs()) + margin;
    let half_h = low[1].abs().max(high[1].abs()) + margin;
    let longest = 2.0 * half_w.max(half_h);
    if !(longest.is_finite() && longest > 0.0) {
        return None;
    }
    let density = if density.is_finite() {
        density.max(0.01)
    } else {
        1.0
    };
    let density = density.min(MAX_SHAPE_EDGE / longest);
    let size = |half: f64| (2.0 * half * density).ceil().clamp(1.0, MAX_SHAPE_EDGE);
    let (pw, ph) = (size(half_w), size(half_h));
    // Canvas pixels from the centre, y up, to picture pixels, y down.
    let to_picture = |[x, y]: [f64; 2], shift: [f64; 2]| {
        [
            pw / 2.0 + (x + shift[0]) * density,
            ph / 2.0 - (y + shift[1]) * density,
        ]
    };

    let outline = |shift: [f64; 2]| -> Outline {
        let cubics = cubics(&points, !open, head.map(|(length, _)| length));
        Outline {
            cubics: cubics
                .iter()
                .map(|cubic| cubic.map(|point| to_picture(point, shift)))
                .collect(),
            closed: !open,
        }
    };
    let pen = Pen {
        width: width * density,
        cap: if material.border_line_style == 2 {
            LineCap::Round
        } else {
            LineCap::Butt
        },
        join: LineJoin::Miter,
        miter_limit: MITER_LIMIT,
    };
    let dashes: Option<Vec<f64>> = match material.border_line_style {
        1 => Some(vec![3.0 * pen.width, 2.0 * pen.width]),
        // A dot is a zero-length dash with round ends.
        2 => Some(vec![0.0, 2.0 * pen.width]),
        _ => None,
    };
    let head_outline = |shift: [f64; 2]| {
        head.map(|(length, half)| {
            let tip = points[points.len() - 1].at;
            let from = points[points.len() - 2].at;
            triangle(from, tip, length, half)
                .map(|corner| to_picture(corner, shift))
                .to_vec()
        })
    };
    let shape_edges = |shift: [f64; 2]| -> (Vec<Edge>, Vec<Edge>) {
        let line = outline(shift);
        let fill_part = if fill.is_some() {
            fill_edges(std::slice::from_ref(&line))
        } else {
            Vec::new()
        };
        let mut stroke_part = if stroke_paint.is_some() {
            stroke_edges(
                std::slice::from_ref(&line),
                &pen,
                dashes.as_deref().map(|pattern| (pattern, 0.0)),
            )
        } else {
            Vec::new()
        };
        if let Some(corners) = head_outline(shift) {
            stroke_part.extend(polygon(&corners));
        }
        (fill_part, stroke_part)
    };

    let (pw, ph) = (pw as u32, ph as u32);
    let mut items = Vec::new();
    if let Some(color) = shadow {
        let radians = material.shadow_angle.to_radians();
        // The angle turns clockwise from the right, so 45 falls down and
        // to the right.
        let shift = [
            radians.cos() * material.shadow_distance,
            -radians.sin() * material.shadow_distance,
        ];
        let (fill_part, stroke_part) = shape_edges(shift);
        let inner: Vec<Item> = [fill_part, stroke_part]
            .into_iter()
            .filter_map(|edges| solid(edges, color, pw, ph))
            .collect();
        if !inner.is_empty() {
            items.push(Item::Group(Group {
                opacity: material.shadow_alpha.clamp(0.0, 1.0) as f32,
                masks: Vec::new(),
                matte: None,
                items: inner,
            }));
        }
    }
    let (fill_part, stroke_part) = shape_edges([0.0, 0.0]);
    if let Some(color) = fill {
        items.extend(solid(fill_part, color, pw, ph));
    }
    if let Some(color) = stroke_paint {
        items.extend(solid(stroke_part, color, pw, ph));
    }
    if items.is_empty() {
        return None;
    }
    Some((
        Scene {
            width: pw,
            height: ph,
            items,
        },
        density,
    ))
}

/// The lowest and highest x and y of the points and their handles.
fn extent(points: &[cutlass_model::ShapePoint]) -> ([f64; 2], [f64; 2]) {
    let mut low = [0.0_f64, 0.0];
    let mut high = [0.0_f64, 0.0];
    for point in points {
        for at in [
            point.at,
            add(point.at, point.handle_in),
            add(point.at, point.handle_out),
        ] {
            for axis in 0..2 {
                low[axis] = low[axis].min(at[axis]);
                high[axis] = high[axis].max(at[axis]);
            }
        }
    }
    (low, high)
}

/// The outline as cubics, corners rounded. An open line stops `trim` short
/// of its end, where an arrow head takes over.
fn cubics(
    points: &[cutlass_model::ShapePoint],
    closed: bool,
    trim: Option<f64>,
) -> Vec<[[f64; 2]; 4]> {
    let count = points.len();
    // Each point becomes where the outline arrives and where it leaves,
    // with the arc between them for a rounded corner.
    let mut arrive = Vec::with_capacity(count);
    let mut leave = Vec::with_capacity(count);
    let mut arcs = Vec::with_capacity(count);
    for (index, point) in points.iter().enumerate() {
        let sharp = point.handle_in == [0.0, 0.0] && point.handle_out == [0.0, 0.0];
        let has_both = closed || (index > 0 && index + 1 < count);
        let rounded = sharp && has_both && point.round > 0.0 && point.round.is_finite();
        if !rounded {
            arrive.push(point.at);
            leave.push(point.at);
            arcs.push(None);
            continue;
        }
        let before = points[(index + count - 1) % count].at;
        let after = points[(index + 1) % count].at;
        let reach_in = length(sub(before, point.at)) / 2.0;
        let reach_out = length(sub(after, point.at)) / 2.0;
        let cut = point.round.min(reach_in).min(reach_out);
        let a = toward(point.at, before, cut);
        let b = toward(point.at, after, cut);
        arrive.push(a);
        leave.push(b);
        arcs.push(Some([
            a,
            lerp(a, point.at, ARC_HANDLE),
            lerp(b, point.at, ARC_HANDLE),
            b,
        ]));
    }
    let mut out = Vec::new();
    let sides = if closed { count } else { count - 1 };
    for index in 0..sides {
        let next = (index + 1) % count;
        if let Some(arc) = arcs[index] {
            out.push(arc);
        }
        let from = leave[index];
        let mut to = arrive[next];
        let mut handle_in = add(to, points[next].handle_in);
        if !closed
            && next == count - 1
            && let Some(trim) = trim
        {
            let span = length(sub(to, from));
            let back = trim.min(span);
            to = toward(to, from, back);
            handle_in = to;
        }
        let handle_out = add(from, points[index].handle_out);
        out.push([from, handle_out, handle_in, to]);
    }
    if closed && let Some(arc) = arcs[0] {
        // The first corner's arc closes the outline.
        out.push(arc);
        out.remove(0);
    }
    out
}

/// An arrow head's length and half width for a stroke `width` wide.
fn arrow_head(width: f64) -> (f64, f64) {
    let width = width.max(1.0);
    ((width * 4.0).max(12.0), (width * 2.0).max(6.0))
}

/// The head's three corners: its tip at `tip`, its base `length` back
/// toward `from`, `half` wide either side.
fn triangle(from: [f64; 2], tip: [f64; 2], length: f64, half: f64) -> [[f64; 2]; 3] {
    let along = sub(tip, from);
    let span = self::length(along);
    let unit = if span > f64::EPSILON {
        [along[0] / span, along[1] / span]
    } else {
        [1.0, 0.0]
    };
    let base = [tip[0] - unit[0] * length, tip[1] - unit[1] * length];
    let side = [-unit[1] * half, unit[0] * half];
    [tip, add(base, side), sub(base, side)]
}

/// A polygon's edges, closed back to its start, with the same winding the
/// stroke's pieces have.
fn polygon(corners: &[[f64; 2]]) -> Vec<Edge> {
    let mut corners = corners.to_vec();
    let area: f64 = (0..corners.len())
        .map(|i| {
            let [x0, y0] = corners[i];
            let [x1, y1] = corners[(i + 1) % corners.len()];
            x0 * y1 - x1 * y0
        })
        .sum();
    if area < 0.0 {
        corners.reverse();
    }
    let cubics = (0..corners.len())
        .map(|i| {
            let (a, b) = (corners[i], corners[(i + 1) % corners.len()]);
            [a, a, b, b]
        })
        .collect();
    fill_edges(&[Outline {
        cubics,
        closed: true,
    }])
}

fn solid(edges: Vec<Edge>, color: [f32; 4], width: u32, height: u32) -> Option<Item> {
    let bounds = Bounds::of(&edges, width, height)?;
    Some(Item::Fill(Fill {
        edges,
        rule: FillRule::NonZero,
        paint: Paint::Solid(color),
        bounds,
    }))
}

/// `#rrggbb` or `#rrggbbaa` as a straight color, its alpha times `opacity`.
/// Empty, or not a color, is nothing.
fn rgba(hex: &str, opacity: f64) -> Option<[f32; 4]> {
    let body = hex.trim().strip_prefix('#').unwrap_or(hex.trim());
    if !(body.len() == 6 || body.len() == 8) {
        return None;
    }
    let channel = |at: usize| {
        u8::from_str_radix(body.get(at..at + 2)?, 16)
            .ok()
            .map(|byte| f32::from(byte) / 255.0)
    };
    let alpha = if body.len() == 8 { channel(6)? } else { 1.0 };
    let opacity = if opacity.is_finite() {
        opacity.clamp(0.0, 1.0) as f32
    } else {
        1.0
    };
    let alpha = alpha * opacity;
    (alpha > 0.0).then_some([channel(0)?, channel(2)?, channel(4)?, alpha])
}

fn add(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn length(v: [f64; 2]) -> f64 {
    v[0].hypot(v[1])
}

fn lerp(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

/// The point `distance` from `from` toward `to`.
fn toward(from: [f64; 2], to: [f64; 2], distance: f64) -> [f64; 2] {
    let span = length(sub(to, from));
    if span <= f64::EPSILON {
        return from;
    }
    lerp(from, to, distance / span)
}

#[cfg(test)]
mod tests {
    use cutlass_lottie::rasterize;
    use cutlass_model::{MaterialShape, ShapeKind};

    use super::*;

    fn drawn(shape: &MaterialShape) -> (u32, u32, Vec<u8>) {
        let (scene, _) = shape_scene(shape, 1.0).expect("a scene");
        (scene.width, scene.height, rasterize(&scene))
    }

    fn pixel(picture: &(u32, u32, Vec<u8>), x: u32, y: u32) -> [u8; 4] {
        let at = ((y * picture.0 + x) * 4) as usize;
        [
            picture.2[at],
            picture.2[at + 1],
            picture.2[at + 2],
            picture.2[at + 3],
        ]
    }

    #[test]
    fn a_square_fills_grey_to_its_edge_and_no_further() {
        let square = MaterialShape::preset("s".into(), ShapeKind::Rectangle, 40.0, 40.0);
        let picture = drawn(&square);
        // Room of one pixel round a 40 pixel square.
        assert_eq!((picture.0, picture.1), (42, 42));
        assert_eq!(pixel(&picture, 21, 21), [0xa6, 0xa6, 0xa6, 255]);
        assert_eq!(pixel(&picture, 0, 0)[3], 0);
    }

    #[test]
    fn rounded_corners_leave_the_corner_clear() {
        let mut square = MaterialShape::preset("s".into(), ShapeKind::Rectangle, 40.0, 40.0);
        let sharp = drawn(&square);
        assert_eq!(pixel(&sharp, 2, 2)[3], 255);
        square.roundness = vec![15.0; 4];
        let round = drawn(&square);
        assert_eq!(pixel(&round, 2, 2)[3], 0);
        assert_eq!(pixel(&round, 21, 2)[3], 255);
    }

    #[test]
    fn a_stroke_rings_the_fill_and_a_shadow_falls_down_right() {
        let mut square = MaterialShape::preset("s".into(), ShapeKind::Rectangle, 40.0, 40.0);
        square.border_width = 4.0;
        square.border_color = "#ff0000".to_owned();
        let ringed = drawn(&square);
        let centre = ringed.0 / 2;
        // The left side's stroke, then the grey inside.
        let left = centre - 20;
        assert_eq!(pixel(&ringed, left, centre), [255, 0, 0, 255]);
        assert_eq!(pixel(&ringed, centre, centre), [0xa6, 0xa6, 0xa6, 255]);

        let mut shadowed = MaterialShape::preset("s".into(), ShapeKind::Rectangle, 40.0, 40.0);
        shadowed.shadow_alpha = 0.5;
        shadowed.shadow_distance = 10.0;
        shadowed.shadow_angle = 45.0;
        let picture = drawn(&shadowed);
        let centre = picture.0 / 2;
        // Down and right of the square, only the shadow shows.
        let below = pixel(&picture, centre + 24, centre + 24);
        assert!(below[3] > 100 && below[3] < 160, "{below:?}");
        assert_eq!(pixel(&picture, centre - 24, centre - 24)[3], 0);
    }

    #[test]
    fn a_line_is_its_stroke_and_an_arrow_has_a_head() {
        let line = MaterialShape::preset("l".into(), ShapeKind::Line, 100.0, 0.0);
        let picture = drawn(&line);
        let (cx, cy) = (picture.0 / 2, picture.1 / 2);
        assert_eq!(pixel(&picture, cx, cy)[3], 255);
        assert_eq!(pixel(&picture, cx, cy - 8)[3], 0);

        let arrow = MaterialShape::preset("a".into(), ShapeKind::Arrow, 100.0, 0.0);
        let picture = drawn(&arrow);
        let (cx, cy) = (picture.0 / 2, picture.1 / 2);
        // Across the head it is wider than the line.
        assert_eq!(pixel(&picture, cx + 22, cy - 8)[3], 255);
        assert_eq!(pixel(&picture, cx - 30, cy - 8)[3], 0);
    }

    #[test]
    fn nothing_to_draw_is_no_scene() {
        let mut square = MaterialShape::preset("s".into(), ShapeKind::Rectangle, 40.0, 40.0);
        square.color.clear();
        assert!(shape_scene(&square, 1.0).is_none());
    }
}
