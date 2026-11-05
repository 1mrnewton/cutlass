//! One frame of an animation as a [`Scene`].
//!
//! Layers are drawn last first, so the first layer is on top. A layer shows
//! from its in point up to its out point, in its composition's frames. Its
//! properties are read at the composition time less its start time. A
//! precomposition's own layers run at that time divided by the stretch, or
//! at its time remap (seconds) in frames. A child's transform follows its
//! parent's, but not its opacity. Layer opacity, masks, a matte and a
//! precomposition's bounds make the layer a [`Group`]; group opacity inside
//! a shape layer multiplies its paints.

use std::collections::HashMap;

use crate::LottieError;
use crate::anim::{Animation, Layer, LayerKind, MaskKind, MaskSpec, MatteKind};
use crate::geom::{Affine, Contour, Point, trim_ranges, trim_run};
use crate::limits::{Budget, Limits, MAX_COPIES};

use crate::scene::{
    Bounds, Edge, Expansion, Fill, FillRule, Gradient, GradientKind, Group, Item, Mask, MaskMode,
    Matte, MatteMode, Paint, Scene, Stop,
};
use crate::shape::{
    Cap, DashPart, GradientStyle, Item as ShapeItem, Join, PaintItem, PaintStyle, StarShape,
    ellipse, rectangle, star,
};
use crate::stroke::{Pen, Polyline, dash, flatten, stroke};
use crate::transform::Transform;

/// How far a drawn line may stray from its curve, in output pixels.
const TOLERANCE: f64 = 0.2;
/// The longest parent chain followed.
const MAX_PARENTS: usize = 64;
/// The most gradient stops kept.
const MAX_STOPS: usize = 256;

struct Context<'a> {
    animation: &'a Animation,
    width: u32,
    height: u32,
    budget: Budget,
}

impl Animation {
    /// The frame `frame` drawn into a `width` by `height` output, the
    /// animation stretched to fill it.
    pub fn scene(&self, frame: f64, width: u32, height: u32) -> Result<Scene, LottieError> {
        if width == 0 || height == 0 {
            return Err(LottieError::Invalid("the output is empty"));
        }
        let mut context = Context {
            animation: self,
            width,
            height,
            budget: Budget::default(),
        };
        let to_out = Affine::scale(
            f64::from(width) / self.width,
            f64::from(height) / self.height,
        );
        let frame = if frame.is_finite() {
            frame
        } else {
            self.in_point
        };
        let items = layers(&mut context, &self.layers, frame, to_out, 0)?;
        Ok(Scene {
            width,
            height,
            items,
        })
    }
}

fn visible(layer: &Layer, time: f64) -> bool {
    time >= layer.in_point && time < layer.out_point
}

/// Each layer's transform to its composition, parents included.
fn worlds(list: &[Layer], time: f64) -> Vec<Affine> {
    let by_index: HashMap<i64, usize> = list
        .iter()
        .enumerate()
        .rev()
        .filter_map(|(at, layer)| layer.index.map(|index| (index, at)))
        .collect();
    let local = |at: usize| list[at].transform.matrix(time - list[at].start);
    (0..list.len())
        .map(|at| {
            let mut world = local(at);
            let mut chain = vec![at];
            while chain.len() <= MAX_PARENTS {
                let Some(parent) = list[chain[chain.len() - 1]]
                    .parent
                    .and_then(|parent| by_index.get(&parent).copied())
                else {
                    break;
                };
                // A parent loop stops where it comes round again.
                if chain.contains(&parent) {
                    break;
                }
                world = local(parent).then(world);
                chain.push(parent);
            }
            world
        })
        .collect()
}

fn layers(
    context: &mut Context<'_>,
    list: &[Layer],
    time: f64,
    to_out: Affine,
    depth: usize,
) -> Result<Vec<Item>, LottieError> {
    Limits::depth(depth)?;
    let worlds = worlds(list, time);
    let mut out = Vec::new();
    for at in (0..list.len()).rev() {
        let layer = &list[at];
        if layer.is_matte || layer.hidden || !visible(layer, time) {
            continue;
        }
        let Some(mut drawn) = layer_items(context, layer, time, to_out.then(worlds[at]), depth)?
        else {
            continue;
        };
        if let Some(kind) = layer.matte {
            let source = match layer.matte_parent {
                Some(index) => list.iter().position(|other| other.index == Some(index)),
                None => at.checked_sub(1),
            };
            if let Some(source) = source.filter(|source| *source != at) {
                let matte_layer = &list[source];
                let items = if visible(matte_layer, time) && !matte_layer.hidden {
                    layer_items(
                        context,
                        matte_layer,
                        time,
                        to_out.then(worlds[source]),
                        depth,
                    )?
                    .unwrap_or_default()
                } else {
                    Vec::new()
                };
                let mode = match kind {
                    MatteKind::Alpha => MatteMode::Alpha,
                    MatteKind::AlphaInverted => MatteMode::AlphaInverted,
                    MatteKind::Luma => MatteMode::Luma,
                    MatteKind::LumaInverted => MatteMode::LumaInverted,
                };
                drawn = vec![Item::Group(Group {
                    opacity: 1.0,
                    masks: Vec::new(),
                    matte: Some(Matte { mode, items }),
                    items: drawn,
                })];
            }
        }
        out.extend(drawn);
    }
    Ok(out)
}

/// One layer's items, wrapped in a group when it needs one. `None` when it
/// draws nothing.
fn layer_items(
    context: &mut Context<'_>,
    layer: &Layer,
    time: f64,
    to_out: Affine,
    depth: usize,
) -> Result<Option<Vec<Item>>, LottieError> {
    context.budget.layer()?;
    let local = time - layer.start;
    let opacity = layer.transform.opacity(local);
    if opacity <= 0.0 {
        return Ok(None);
    }
    let mut masks = Vec::new();
    let mut items = match &layer.kind {
        LayerKind::Null => return Ok(None),
        LayerKind::Shapes(shapes) => shape_layer(context, shapes, local, to_out)?,
        LayerKind::Solid {
            color,
            width,
            height,
        } => {
            let rect = rectangle(
                Point::new(width / 2.0, height / 2.0),
                Point::new(*width, *height),
                0.0,
                false,
            );
            let paint = Paint::Solid([color[0] as f32, color[1] as f32, color[2] as f32, 1.0]);
            let mut out = Vec::new();
            fill_item(context, &[rect], to_out, FillRule::NonZero, paint, &mut out)?;
            out
        }
        LayerKind::Precomp {
            comp,
            width,
            height,
            time_remap,
        } => {
            let child_time = match time_remap {
                Some(remap) => remap.scalar(local, 0.0) * context.animation.frame_rate,
                None => local / layer.stretch,
            };
            let Some(child) = context.animation.comps.get(*comp) else {
                return Ok(None);
            };
            if *width > 0.0 && *height > 0.0 {
                // A precomposition shows only inside its own bounds.
                let rect = rectangle(
                    Point::new(width / 2.0, height / 2.0),
                    Point::new(*width, *height),
                    0.0,
                    false,
                );
                masks.push(Mask {
                    mode: MaskMode::Add,
                    inverted: false,
                    opacity: 1.0,
                    edges: contour_edges(context, &[rect], to_out)?,
                    expansion: None,
                });
            }
            layers(context, child, child_time, to_out, depth + 1)?
        }
    };
    for spec in &layer.masks {
        masks.push(mask(context, spec, local, to_out)?);
    }
    if items.is_empty() {
        return Ok(None);
    }
    if opacity < 1.0 || !masks.is_empty() {
        items = vec![Item::Group(Group {
            opacity: opacity as f32,
            masks,
            matte: None,
            items,
        })];
    }
    Ok(Some(items))
}

fn mask(
    context: &mut Context<'_>,
    spec: &MaskSpec,
    time: f64,
    to_out: Affine,
) -> Result<Mask, LottieError> {
    let contour = spec.shape.at(time).and_then(|shape| shape.contour());
    let contours: Vec<Contour> = contour.into_iter().collect();
    let edges = contour_edges(context, &contours, to_out)?;
    let grow = spec.expansion.scalar(time, 0.0);
    let expansion = if grow.abs() > 1e-6 && !contours.is_empty() {
        let pen = Pen {
            half: grow.abs(),
            cap: Cap::Round,
            join: Join::Round,
            miter: 4.0,
        };
        let edges = stroke_edges(context, &contours, Affine::IDENTITY, to_out, &pen, None)?;
        Some(Expansion {
            grow: grow > 0.0,
            edges,
        })
    } else {
        None
    };
    Ok(Mask {
        mode: match spec.kind {
            MaskKind::Add => MaskMode::Add,
            MaskKind::Subtract => MaskMode::Subtract,
            MaskKind::Intersect => MaskMode::Intersect,
            MaskKind::Lighten => MaskMode::Lighten,
            MaskKind::Darken => MaskMode::Darken,
            MaskKind::Difference => MaskMode::Difference,
        },
        inverted: spec.inverted,
        opacity: (spec.opacity.scalar(time, 100.0) / 100.0).clamp(0.0, 1.0) as f32,
        edges,
        expansion,
    })
}

/// A path the shape items made, and its map to the layer.
struct PathEntry {
    contours: Vec<Contour>,
    to_layer: Affine,
}

/// A paint waiting for the trims below it.
struct Pending<'a> {
    paint: &'a PaintItem,
    paths: Vec<usize>,
    to_layer: Affine,
    opacity: f64,
}

fn shape_layer(
    context: &mut Context<'_>,
    shapes: &[ShapeItem],
    time: f64,
    to_out: Affine,
) -> Result<Vec<Item>, LottieError> {
    let mut paths = Vec::new();
    let mut pending = Vec::new();
    shape_items(
        context,
        shapes,
        time,
        Affine::IDENTITY,
        1.0,
        0,
        &mut paths,
        &mut pending,
    )?;
    let mut out = Vec::new();
    // Pending paints are top first; the scene is bottom first.
    for paint in pending.iter().rev() {
        draw_paint(context, paint, &paths, time, to_out, &mut out)?;
    }
    Ok(out)
}

/// Walk one list of shape items. Returns the paths it made, for the paints
/// and trims below it in its parent's list.
#[allow(clippy::too_many_arguments)]
fn shape_items<'a>(
    context: &mut Context<'_>,
    list: &'a [ShapeItem],
    time: f64,
    to_layer: Affine,
    opacity: f64,
    depth: usize,
    paths: &mut Vec<PathEntry>,
    pending: &mut Vec<Pending<'a>>,
) -> Result<Vec<usize>, LottieError> {
    Limits::depth(depth)?;
    let mut above = Vec::new();
    let push = |paths: &mut Vec<PathEntry>, contour: Contour, above: &mut Vec<usize>| {
        if !contour.cubics.is_empty() {
            above.push(paths.len());
            paths.push(PathEntry {
                contours: vec![contour],
                to_layer,
            });
        }
    };
    for item in list {
        match item {
            ShapeItem::Group { items, transform } => {
                let inner = shape_items(
                    context,
                    items,
                    time,
                    to_layer.then(transform.matrix(time)),
                    opacity * transform.opacity(time),
                    depth + 1,
                    paths,
                    pending,
                )?;
                above.extend(inner);
            }
            ShapeItem::Path { shape } => {
                if let Some(contour) = shape.at(time).and_then(|shape| shape.contour()) {
                    push(paths, contour, &mut above);
                }
            }
            ShapeItem::Rect {
                position,
                size,
                roundness,
                reversed,
            } => push(
                paths,
                rectangle(
                    position.point(time, Point::default()),
                    size.point(time, Point::default()),
                    roundness.scalar(time, 0.0),
                    *reversed,
                ),
                &mut above,
            ),
            ShapeItem::Ellipse {
                position,
                size,
                reversed,
            } => push(
                paths,
                ellipse(
                    position.point(time, Point::default()),
                    size.point(time, Point::default()),
                    *reversed,
                ),
                &mut above,
            ),
            ShapeItem::Star {
                polygon,
                position,
                points,
                rotation,
                outer_radius,
                outer_roundness,
                inner_radius,
                inner_roundness,
                reversed,
            } => push(
                paths,
                star(&StarShape {
                    center: position.point(time, Point::default()),
                    points: points.scalar(time, 5.0),
                    rotation: rotation.scalar(time, 0.0),
                    outer: (
                        outer_radius.scalar(time, 0.0),
                        outer_roundness.scalar(time, 0.0),
                    ),
                    inner: (!polygon).then(|| {
                        (
                            inner_radius.scalar(time, 0.0),
                            inner_roundness.scalar(time, 0.0),
                        )
                    }),
                    reversed: *reversed,
                }),
                &mut above,
            ),
            ShapeItem::Paint(paint) => {
                context.budget.draw()?;
                pending.push(Pending {
                    paint,
                    paths: above.clone(),
                    to_layer,
                    opacity,
                });
            }
            ShapeItem::Trim {
                start,
                end,
                offset,
                one_run,
            } => trim(
                paths,
                &above,
                start.scalar(time, 0.0) / 100.0,
                end.scalar(time, 100.0) / 100.0,
                offset.scalar(time, 0.0) / 360.0,
                *one_run,
            ),
            ShapeItem::Repeat { items, repeater } => {
                let copies = repeater.copies.scalar(time, 1.0).ceil();
                if copies.is_nan() || copies < 1.0 {
                    continue;
                }
                let copies = copies.min(MAX_COPIES as f64) as usize;
                let offset = repeater.offset.scalar(time, 0.0);
                let first = repeater.start_opacity.scalar(time, 100.0) / 100.0;
                let last = repeater.end_opacity.scalar(time, 100.0) / 100.0;
                let order: Vec<usize> = if repeater.above {
                    (0..copies).rev().collect()
                } else {
                    (0..copies).collect()
                };
                for copy in order {
                    let along = if copies > 1 {
                        copy as f64 / (copies - 1) as f64
                    } else {
                        0.0
                    };
                    let copy_opacity = (first + (last - first) * along).clamp(0.0, 1.0);
                    let map = repeated(&repeater.transform, time, offset + copy as f64);
                    let inner = shape_items(
                        context,
                        items,
                        time,
                        to_layer.then(map),
                        opacity * copy_opacity,
                        depth + 1,
                        paths,
                        pending,
                    )?;
                    above.extend(inner);
                }
            }
        }
    }
    Ok(above)
}

/// The repeater's transform applied `times` times.
fn repeated(transform: &Transform, time: f64, times: f64) -> Affine {
    let anchor = transform.anchor(time);
    let position = transform.position(time);
    let scale = transform.scale(time);
    let map = Affine::translate(position.x * times, position.y * times)
        .then(Affine::translate(anchor.x, anchor.y))
        .then(Affine::rotate(transform.rotation(time) * times))
        .then(Affine::scale(scale.x.powf(times), scale.y.powf(times)))
        .then(Affine::translate(-anchor.x, -anchor.y));
    if map.is_finite() {
        map
    } else {
        Affine::scale(0.0, 0.0)
    }
}

fn trim(
    paths: &mut [PathEntry],
    above: &[usize],
    start: f64,
    end: f64,
    offset: f64,
    one_run: bool,
) {
    let Some(ranges) = trim_ranges(start, end, offset) else {
        return;
    };
    if !one_run {
        for &index in above {
            if let Some(entry) = paths.get_mut(index) {
                entry.contours = trim_run(&entry.contours, &ranges);
            }
        }
        return;
    }
    // One run through every path in turn.
    let lengths: Vec<f64> = above
        .iter()
        .map(|&index| {
            paths.get(index).map_or(0.0, |entry| {
                entry.contours.iter().map(Contour::length).sum()
            })
        })
        .collect();
    let total: f64 = lengths.iter().sum();
    let mut from = 0.0;
    for (&index, length) in above.iter().zip(&lengths) {
        let to = from + length;
        let Some(entry) = paths.get_mut(index) else {
            continue;
        };
        if *length <= 0.0 || total <= 0.0 {
            entry.contours.clear();
            from = to;
            continue;
        }
        let local: Vec<(f64, f64)> = ranges
            .iter()
            .filter_map(|&(a, b)| {
                let a = (a * total).max(from);
                let b = (b * total).min(to);
                (b > a).then(|| ((a - from) / length, (b - from) / length))
            })
            .collect();
        entry.contours = trim_run(&entry.contours, &local);
        from = to;
    }
}

fn draw_paint(
    context: &mut Context<'_>,
    pending: &Pending<'_>,
    paths: &[PathEntry],
    time: f64,
    to_out: Affine,
    out: &mut Vec<Item>,
) -> Result<(), LottieError> {
    let paint_item = pending.paint;
    let opacity =
        pending.opacity * (paint_item.opacity.scalar(time, 100.0) / 100.0).clamp(0.0, 1.0);
    if opacity <= 0.0 {
        return Ok(());
    }
    let paint_to_out = to_out.then(pending.to_layer);
    let Some(paint) = paint_of(&paint_item.style, time, paint_to_out, opacity) else {
        return Ok(());
    };
    match &paint_item.stroke {
        None => {
            let rule = if paint_item.even_odd {
                FillRule::EvenOdd
            } else {
                FillRule::NonZero
            };
            let mut edges = Vec::new();
            for &index in &pending.paths {
                if let Some(entry) = paths.get(index) {
                    edges.extend(contour_edges(
                        context,
                        &entry.contours,
                        to_out.then(entry.to_layer),
                    )?);
                }
            }
            push_fill(context, edges, rule, paint, out);
        }
        Some(style) => {
            let width = style.width.scalar(time, 1.0);
            if width.is_nan() || width <= 0.0 {
                return Ok(());
            }
            let Some(from_layer) = pending.to_layer.invert() else {
                return Ok(());
            };
            let pen = Pen {
                half: width / 2.0,
                cap: style.cap,
                join: style.join,
                miter: style.miter.scalar(time, 4.0).max(1.0),
            };
            let mut pattern = Vec::new();
            let mut dash_offset = 0.0;
            for (part, value) in &style.dashes {
                let value = value.scalar(time, 0.0);
                match part {
                    DashPart::Dash | DashPart::Gap => pattern.push(value),
                    DashPart::Offset => dash_offset = value,
                }
            }
            let dashes = (!pattern.is_empty()).then_some((pattern.as_slice(), dash_offset));
            let mut edges = Vec::new();
            for &index in &pending.paths {
                if let Some(entry) = paths.get(index) {
                    edges.extend(stroke_edges(
                        context,
                        &entry.contours,
                        from_layer.then(entry.to_layer),
                        paint_to_out,
                        &pen,
                        dashes,
                    )?);
                }
            }
            push_fill(context, edges, FillRule::NonZero, paint, out);
        }
    }
    Ok(())
}

fn push_fill(
    context: &Context<'_>,
    edges: Vec<Edge>,
    rule: FillRule,
    paint: Paint,
    out: &mut Vec<Item>,
) {
    if let Some(bounds) = Bounds::of(&edges, context.width, context.height) {
        out.push(Item::Fill(Fill {
            edges,
            rule,
            paint,
            bounds,
        }));
    }
}

fn fill_item(
    context: &mut Context<'_>,
    contours: &[Contour],
    to_out: Affine,
    rule: FillRule,
    paint: Paint,
    out: &mut Vec<Item>,
) -> Result<(), LottieError> {
    let edges = contour_edges(context, contours, to_out)?;
    push_fill(context, edges, rule, paint, out);
    Ok(())
}

/// Closed polygons as edges, each closed back to its start. Level edges
/// cross no sample row and are left out.
fn polygon_edges(
    context: &mut Context<'_>,
    polygons: impl Iterator<Item = Vec<Point>>,
    map: Affine,
) -> Result<Vec<Edge>, LottieError> {
    let mut edges = Vec::new();
    for polygon in polygons {
        let mapped: Vec<(f32, f32)> = polygon
            .iter()
            .map(|point| map.apply(*point))
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
    context.budget.edges(edges.len())?;
    Ok(edges)
}

fn contour_edges(
    context: &mut Context<'_>,
    contours: &[Contour],
    to_out: Affine,
) -> Result<Vec<Edge>, LottieError> {
    let polygons = contours
        .iter()
        .map(|contour| flatten(contour, to_out, TOLERANCE).points);
    let polygons: Vec<Vec<Point>> = polygons.collect();
    polygon_edges(context, polygons.into_iter(), Affine::IDENTITY)
}

/// `contours`, mapped into the paint's space by `to_paint`, stroked there
/// with `pen`, then mapped out by `paint_to_out`.
fn stroke_edges(
    context: &mut Context<'_>,
    contours: &[Contour],
    to_paint: Affine,
    paint_to_out: Affine,
    pen: &Pen,
    dashes: Option<(&[f64], f64)>,
) -> Result<Vec<Edge>, LottieError> {
    let stretch = paint_to_out.stretch();
    if !(stretch > 0.0 && stretch.is_finite()) {
        return Ok(Vec::new());
    }
    let tolerance = TOLERANCE / stretch;
    let mut lines: Vec<Polyline> = contours
        .iter()
        .map(|contour| flatten(contour, to_paint, tolerance))
        .collect();
    if let Some((pattern, offset)) = dashes {
        lines = dash(lines, pattern, offset);
    }
    let mut polygons = Vec::new();
    for line in &lines {
        stroke(line, pen, tolerance, &mut polygons);
    }
    polygon_edges(context, polygons.into_iter(), paint_to_out)
}

/// A straight color from 0 to 1; a file in 0 to 255 is scaled down.
fn color(value: &[f64]) -> [f64; 4] {
    let scale = if value.iter().take(3).any(|channel| *channel > 1.0) {
        1.0 / 255.0
    } else {
        1.0
    };
    let channel = |at: usize, fallback: f64| {
        let value = value.get(at).copied().unwrap_or(fallback) * if at < 3 { scale } else { 1.0 };
        if value.is_finite() {
            value.clamp(0.0, 1.0)
        } else {
            fallback
        }
    };
    [
        channel(0, 0.0),
        channel(1, 0.0),
        channel(2, 0.0),
        channel(3, 1.0),
    ]
}

fn paint_of(style: &PaintStyle, time: f64, paint_to_out: Affine, opacity: f64) -> Option<Paint> {
    match style {
        PaintStyle::Color(prop) => {
            let [r, g, b, a] = color(&prop.at(time)?);
            Some(Paint::Solid([
                r as f32,
                g as f32,
                b as f32,
                (a * opacity) as f32,
            ]))
        }
        PaintStyle::Gradient(gradient) => {
            gradient_of(gradient, time, paint_to_out, opacity).map(Paint::Gradient)
        }
    }
}

/// Linear interpolation through `(offset, values)` pairs, sorted.
fn ramp(stops: &[(f64, Vec<f64>)], at: f64) -> Option<Vec<f64>> {
    let first = stops.first()?;
    if at <= first.0 {
        return Some(first.1.clone());
    }
    for pair in stops.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        if at <= b.0 {
            let span = b.0 - a.0;
            let t = if span > 0.0 { (at - a.0) / span } else { 1.0 };
            return Some(a.1.iter().zip(&b.1).map(|(x, y)| x + (y - x) * t).collect());
        }
    }
    stops.last().map(|last| last.1.clone())
}

fn gradient_of(
    style: &GradientStyle,
    time: f64,
    paint_to_out: Affine,
    opacity: f64,
) -> Option<Gradient> {
    let raw = style.stops.at(time)?;
    let count = style.count.min(raw.len() / 4).min(MAX_STOPS);
    if count == 0 {
        return None;
    }
    let mut colors: Vec<(f64, Vec<f64>)> = (0..count)
        .map(|stop| {
            let at = stop * 4;
            (raw[at], raw[at + 1..at + 4].to_vec())
        })
        .collect();
    colors.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut alphas: Vec<(f64, Vec<f64>)> = raw[count * 4..]
        .as_chunks::<2>()
        .0
        .iter()
        .take(MAX_STOPS)
        .map(|pair| (pair[0], vec![pair[1]]))
        .collect();
    alphas.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut offsets: Vec<f64> = colors
        .iter()
        .chain(&alphas)
        .map(|stop| stop.0)
        .filter(|offset| offset.is_finite())
        .collect();
    offsets.sort_by(f64::total_cmp);
    offsets.dedup();
    let stops: Vec<Stop> = offsets
        .iter()
        .take(MAX_STOPS)
        .filter_map(|&offset| {
            let rgb = ramp(&colors, offset)?;
            let alpha = ramp(&alphas, offset)
                .and_then(|alpha| alpha.first().copied())
                .unwrap_or(1.0);
            let [r, g, b, a] = color(&[rgb[0], rgb[1], rgb[2], alpha]);
            Some(Stop {
                offset: offset.clamp(0.0, 1.0) as f32,
                color: [r as f32, g as f32, b as f32, a as f32],
            })
        })
        .collect();
    let to_gradient = paint_to_out.invert()?;
    let start = style.start.point(time, Point::default());
    let end = style.end.point(time, Point::default());
    let (kind, focal, radius) = if style.radial {
        let delta = end.sub(start);
        let radius = delta.length();
        let angle = delta.y.atan2(delta.x) + style.angle.scalar(time, 0.0).to_radians();
        let reach = (style.highlight.scalar(time, 0.0) / 100.0).clamp(-0.99, 0.99) * radius;
        let focal = start.add(Point::new(angle.cos() * reach, angle.sin() * reach));
        (GradientKind::Radial, focal, radius)
    } else {
        (GradientKind::Linear, start, 0.0)
    };
    Some(Gradient {
        kind,
        to_gradient: to_gradient.0.map(|value| value as f32),
        start: [start.x as f32, start.y as f32],
        end: [end.x as f32, end.y as f32],
        focal: [focal.x as f32, focal.y as f32],
        radius: radius as f32,
        stops,
        opacity: opacity as f32,
    })
}
