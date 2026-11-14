//! A layered color glyph painted into one premultiplied RGBA image.
//!
//! The font hands over a tree of paints. Each outline in it becomes
//! anti-aliased coverage by the nonzero rule, and each solid or gradient
//! is laid through that coverage onto the image. Colors are blended as
//! the sRGB values the palette stores, and a gradient changes each channel
//! and the alpha evenly between two stops before they are premultiplied.
//!
//! The image is cached for the glyph and not for a color. A glyph with a
//! paint that asks for the color of the text is painted twice, with black
//! text and with white: every blend here changes evenly with the color
//! under it, so the glyph in any color of text lies between the two.
//!
//! The work is bounded by the font's reader, which hands over at most
//! 4096 paints nested 32 deep, and by the cell: each paint touches at most
//! one image of 256 by 256 pixels.

use std::collections::HashMap;

use cutlass_font::{
    Affine, ClipBox, Color, ColorGlyph, ColorLine, CompositeMode, Contour, Extend, Font, Outline,
    Paint, Segment,
};

use super::{ATLAS_EM, Edge, GlyphImage, GlyphKind, MAX_CELL, Point, blank, bounds, flatten};
use crate::TextError;

/// Image pixels across one em. A distance field is cut at 64 and stays
/// sharp when it is enlarged; a color image does not, so it gets twice
/// that, which is near the largest size emoji strikes are stored at. A
/// glyph just under two ems across still fits the cell at this size.
pub(super) const COLOR_EM: f32 = 128.0;
/// Rows of coverage sampled inside each pixel row.
const SUBROWS: u8 = 8;
/// A clear pixel all round, so a sample at the edge blends to nothing.
const PAD: f32 = 1.0;

/// One pixel, premultiplied, each channel from 0 to 1.
type Pixel = [f32; 4];

const CLEAR: Pixel = [0.0; 4];

/// Paint `glyph` at up to [`COLOR_EM`] pixels to the em. A glyph too large
/// for the cell at that size is painted smaller, so that it fits.
pub(super) fn paint(font: &Font<'_>, glyph: &ColorGlyph) -> Result<GlyphImage, TextError> {
    let em = f32::from(font.units_per_em());
    let mut painter = Painter {
        font,
        width: 0,
        height: 0,
        outlines: HashMap::new(),
        text: TextInk::default(),
    };
    let frame = match glyph.clip {
        Some(clip) => Some(Frame::of_box(&clip, &Affine::IDENTITY)),
        None => {
            let mut frame = None;
            painter.extent(&glyph.paint, &Affine::IDENTITY, &mut frame)?;
            frame
        }
    };
    // A glyph whose paints fill no outline draws nothing.
    let Some(frame) = frame else {
        return Ok(blank(em / ATLAS_EM));
    };
    let (across, up) = (frame.max_x - frame.min_x, frame.max_y - frame.min_y);
    if !(across.is_finite() && up.is_finite()) {
        return Err(TextError::Limit);
    }
    if across <= 0.0 || up <= 0.0 {
        return Ok(blank(em / ATLAS_EM));
    }
    let room = f32::from(MAX_CELL) - 2.0 * PAD;
    let unit = (em / COLOR_EM).max(across / room).max(up / room);
    if !(unit.is_finite() && unit > 0.0) {
        return Err(TextError::Limit);
    }
    let width = cells(across / unit)?;
    let height = cells(up / unit)?;
    let left = frame.min_x - PAD * unit;
    let top = frame.max_y + PAD * unit;
    painter.width = usize::try_from(width).map_err(|_| TextError::Limit)?;
    painter.height = usize::try_from(height).map_err(|_| TextError::Limit)?;
    // Font units, y up, to image pixels, y down.
    let to_pixels = Affine {
        xx: 1.0 / unit,
        yx: 0.0,
        xy: 0.0,
        yy: -1.0 / unit,
        dx: -left / unit,
        dy: top / unit,
    };
    let clip = glyph
        .clip
        .map(|clip| painter.box_mask(&clip, &to_pixels, None));
    let mut color = Vec::new();
    // Once with black text; and once more with white, when a paint asked
    // for the color of the text.
    for level in [0.0, 1.0] {
        painter.text.level = level;
        let mut canvas = vec![CLEAR; painter.width * painter.height];
        painter.draw(&glyph.paint, &to_pixels, clip.as_deref(), &mut canvas)?;
        color.reserve(canvas.len() * 4);
        for [red, green, blue, alpha] in canvas {
            let alpha = byte(alpha);
            // A premultiplied channel is never above its alpha.
            color.extend([
                byte(red).min(alpha),
                byte(green).min(alpha),
                byte(blue).min(alpha),
                alpha,
            ]);
        }
        if !painter.text.asked.get() {
            break;
        }
    }
    Ok(GlyphImage {
        width,
        height,
        left,
        top,
        unit,
        pixels_per_em: em / unit,
        kind: GlyphKind::Color,
        sdf: Vec::new(),
        color,
    })
}

/// Pixels for a span of `pixels`, with the clear edge on both sides.
fn cells(pixels: f32) -> Result<u32, TextError> {
    let edge = (pixels.ceil() + 2.0 * PAD).min(f32::from(MAX_CELL));
    if !(edge.is_finite() && edge >= 1.0) {
        return Err(TextError::Limit);
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let edge = edge as u32;
    Ok(edge)
}

fn byte(value: f32) -> u8 {
    let scaled = (value * 255.0).round();
    if scaled >= 255.0 {
        255
    } else if scaled >= 0.0 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            scaled as u8
        }
    } else {
        // Below zero, or not a number.
        0
    }
}

/// The part of the plane a glyph's paints cover, in font units.
#[derive(Clone, Copy)]
struct Frame {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
}

impl Frame {
    fn of_box(clip: &ClipBox, transform: &Affine) -> Self {
        let mut frame: Option<Self> = None;
        for (x, y) in corners(clip) {
            let (x, y) = transform.apply(x, y);
            include(&mut frame, x, y);
        }
        frame.unwrap_or(Self {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 0.0,
            max_y: 0.0,
        })
    }
}

fn corners(clip: &ClipBox) -> [(f32, f32); 4] {
    [
        (clip.x_min, clip.y_min),
        (clip.x_max, clip.y_min),
        (clip.x_max, clip.y_max),
        (clip.x_min, clip.y_max),
    ]
}

fn include(frame: &mut Option<Frame>, x: f32, y: f32) {
    *frame = Some(match *frame {
        Some(frame) => Frame {
            min_x: frame.min_x.min(x),
            min_y: frame.min_y.min(y),
            max_x: frame.max_x.max(x),
            max_y: frame.max_y.max(y),
        },
        None => Frame {
            min_x: x,
            min_y: y,
            max_x: x,
            max_y: y,
        },
    });
}

struct Painter<'f, 'a> {
    font: &'f Font<'a>,
    width: usize,
    height: usize,
    /// The outlines the paints fill, read once each.
    outlines: HashMap<u16, Outline>,
    text: TextInk,
}

/// The color the text is taken to be while a glyph is painted: `level`
/// in each channel, 0 for black and 1 for white. `asked` is set when a
/// paint took it.
#[derive(Default)]
struct TextInk {
    level: f32,
    asked: std::cell::Cell<bool>,
}

impl Painter<'_, '_> {
    fn outline(&mut self, glyph_id: u16) -> Result<&Outline, TextError> {
        if !self.outlines.contains_key(&glyph_id) {
            let outline = self.font.outline(glyph_id).map_err(TextError::Font)?;
            self.outlines.insert(glyph_id, outline);
        }
        self.outlines.get(&glyph_id).ok_or(TextError::Limit)
    }

    /// Grow `frame` to hold every outline `paint` fills. A glyph used
    /// again counts as its clip box when it has one.
    fn extent(
        &mut self,
        paint: &Paint,
        transform: &Affine,
        frame: &mut Option<Frame>,
    ) -> Result<(), TextError> {
        match paint {
            Paint::Layers(layers) => {
                for layer in layers {
                    self.extent(layer, transform, frame)?;
                }
            }
            Paint::Glyph { glyph_id, .. } => {
                let outline = self.outline(*glyph_id)?;
                if !outline.contours.is_empty() {
                    let inked = bounds(outline)?;
                    let clip = ClipBox {
                        x_min: inked.min_x,
                        y_min: inked.min_y,
                        x_max: inked.max_x,
                        y_max: inked.max_y,
                    };
                    let inner = Frame::of_box(&clip, transform);
                    include(frame, inner.min_x, inner.min_y);
                    include(frame, inner.max_x, inner.max_y);
                }
            }
            Paint::ColorGlyph(glyph) => match glyph.clip {
                Some(clip) => {
                    let inner = Frame::of_box(&clip, transform);
                    include(frame, inner.min_x, inner.min_y);
                    include(frame, inner.max_x, inner.max_y);
                }
                None => self.extent(&glyph.paint, transform, frame)?,
            },
            Paint::Transform {
                transform: inner,
                paint,
            } => self.extent(paint, &transform.then_after(inner), frame)?,
            Paint::Composite {
                source, backdrop, ..
            } => {
                self.extent(backdrop, transform, frame)?;
                self.extent(source, transform, frame)?;
            }
            Paint::Solid(_)
            | Paint::LinearGradient { .. }
            | Paint::RadialGradient { .. }
            | Paint::SweepGradient { .. } => {}
        }
        Ok(())
    }

    /// Lay `paint` onto `canvas` through `clip`, which is coverage for each
    /// pixel, or everything when there is none.
    fn draw(
        &mut self,
        paint: &Paint,
        to_pixels: &Affine,
        clip: Option<&[f32]>,
        canvas: &mut [Pixel],
    ) -> Result<(), TextError> {
        match paint {
            Paint::Layers(layers) => {
                for layer in layers {
                    self.draw(layer, to_pixels, clip, canvas)?;
                }
            }
            Paint::Solid(color) => {
                let color = premultiplied(straight(color, &self.text));
                self.fill(canvas, clip, |_, _| Some(color));
            }
            Paint::LinearGradient { line, p0, p1, p2 } => {
                // The color changes across the direction from p0 to p2.
                let normal = [p2[1] - p0[1], p0[0] - p2[0]];
                let length = (p1[0] - p0[0]) * normal[0] + (p1[1] - p0[1]) * normal[1];
                if length != 0.0 && length.is_finite() {
                    self.shade(canvas, clip, to_pixels, line, |x, y| {
                        Some(((x - p0[0]) * normal[0] + (y - p0[1]) * normal[1]) / length)
                    });
                }
            }
            Paint::RadialGradient {
                line,
                c0,
                r0,
                c1,
                r1,
            } => {
                let circles = Circles::new(*c0, *r0, *c1, *r1);
                self.shade(canvas, clip, to_pixels, line, |x, y| circles.at(x, y));
            }
            Paint::SweepGradient {
                line,
                center,
                start,
                end,
            } => {
                let span = end - start;
                self.shade(canvas, clip, to_pixels, line, |x, y| {
                    let angle = (y - center[1])
                        .atan2(x - center[0])
                        .to_degrees()
                        .rem_euclid(360.0);
                    if span == 0.0 {
                        // No sector: the first color before the angle, the last after.
                        Some(if angle < *start {
                            f32::NEG_INFINITY
                        } else {
                            f32::INFINITY
                        })
                    } else {
                        Some((angle - start) / span)
                    }
                });
            }
            Paint::Glyph { glyph_id, paint } => {
                let mask = self.outline_mask(*glyph_id, to_pixels, clip)?;
                self.draw(paint, to_pixels, Some(&mask), canvas)?;
            }
            Paint::ColorGlyph(glyph) => match glyph.clip {
                Some(inner) => {
                    let mask = self.box_mask(&inner, to_pixels, clip);
                    self.draw(&glyph.paint, to_pixels, Some(&mask), canvas)?;
                }
                None => self.draw(&glyph.paint, to_pixels, clip, canvas)?,
            },
            Paint::Transform { transform, paint } => {
                self.draw(paint, &to_pixels.then_after(transform), clip, canvas)?;
            }
            Paint::Composite {
                source,
                mode,
                backdrop,
            } => {
                let mut under = vec![CLEAR; canvas.len()];
                self.draw(backdrop, to_pixels, clip, &mut under)?;
                let mut over = vec![CLEAR; canvas.len()];
                self.draw(source, to_pixels, clip, &mut over)?;
                for ((pixel, over), under) in canvas.iter_mut().zip(over).zip(under) {
                    *pixel = source_over(blend(*mode, over, under), *pixel);
                }
            }
        }
        Ok(())
    }

    /// Lay the color `shade` gives each pixel center onto `canvas`.
    fn fill(
        &self,
        canvas: &mut [Pixel],
        clip: Option<&[f32]>,
        mut shade: impl FnMut(f32, f32) -> Option<Pixel>,
    ) {
        let (mut x, mut y) = (0.5, 0.5);
        let mut column = 0;
        for (index, pixel) in canvas.iter_mut().enumerate() {
            let cover = match clip {
                Some(mask) => mask.get(index).copied().unwrap_or(0.0),
                None => 1.0,
            };
            if cover > 0.0
                && let Some(color) = shade(x, y)
            {
                *pixel = source_over(color.map(|channel| channel * cover), *pixel);
            }
            column += 1;
            if column == self.width {
                column = 0;
                x = 0.5;
                y += 1.0;
            } else {
                x += 1.0;
            }
        }
    }

    /// Lay a gradient onto `canvas`. `place` gives where along the color
    /// line a point of the paint's own space falls, or nothing there.
    fn shade(
        &self,
        canvas: &mut [Pixel],
        clip: Option<&[f32]>,
        to_pixels: &Affine,
        line: &ColorLine,
        place: impl Fn(f32, f32) -> Option<f32>,
    ) {
        let Some(from_pixels) = inverse(to_pixels) else {
            return;
        };
        self.fill(canvas, clip, |x, y| {
            let (x, y) = from_pixels.apply(x, y);
            let color = color_at(line, place(x, y)?, &self.text)?;
            Some(premultiplied(color))
        });
    }

    /// Coverage of an outline under `to_pixels`, inside `clip`.
    fn outline_mask(
        &mut self,
        glyph_id: u16,
        to_pixels: &Affine,
        clip: Option<&[f32]>,
    ) -> Result<Vec<f32>, TextError> {
        let (width, height) = (self.width, self.height);
        let outline = self.outline(glyph_id)?;
        // `flatten` turns y over; hand it the pixels with y turned back.
        let turned = Affine {
            yx: -to_pixels.yx,
            yy: -to_pixels.yy,
            dy: -to_pixels.dy,
            ..*to_pixels
        };
        let moved = Outline {
            contours: outline
                .contours
                .iter()
                .map(|contour| move_contour(contour, &turned))
                .collect(),
        };
        let edges = flatten(&moved, 0.0, 0.0, 1.0)?;
        let mut mask = coverage(&edges, width, height);
        narrow(&mut mask, clip);
        Ok(mask)
    }

    /// Coverage of a clip box under `to_pixels`, inside `clip`.
    fn box_mask(&self, inner: &ClipBox, to_pixels: &Affine, clip: Option<&[f32]>) -> Vec<f32> {
        let points = corners(inner).map(|(x, y)| {
            let (x, y) = to_pixels.apply(x, y);
            Point { x, y }
        });
        let edges: Vec<Edge> = (0..4)
            .map(|index| Edge {
                start: points[index],
                end: points[(index + 1) % 4],
            })
            .collect();
        let mut mask = coverage(&edges, self.width, self.height);
        narrow(&mut mask, clip);
        mask
    }
}

fn narrow(mask: &mut [f32], clip: Option<&[f32]>) {
    if let Some(clip) = clip {
        for (cover, outer) in mask.iter_mut().zip(clip) {
            *cover *= outer;
        }
    }
}

fn move_contour(contour: &Contour, transform: &Affine) -> Contour {
    let at = |x: f32, y: f32| transform.apply(x, y);
    let (x, y) = at(contour.x, contour.y);
    Contour {
        x,
        y,
        segments: contour
            .segments
            .iter()
            .map(|segment| match *segment {
                Segment::Line { x, y } => {
                    let (x, y) = at(x, y);
                    Segment::Line { x, y }
                }
                Segment::Quad { cx, cy, x, y } => {
                    let (cx, cy) = at(cx, cy);
                    let (x, y) = at(x, y);
                    Segment::Quad { cx, cy, x, y }
                }
                Segment::Cubic {
                    c1x,
                    c1y,
                    c2x,
                    c2y,
                    x,
                    y,
                } => {
                    let (c1x, c1y) = at(c1x, c1y);
                    let (c2x, c2y) = at(c2x, c2y);
                    let (x, y) = at(x, y);
                    Segment::Cubic {
                        c1x,
                        c1y,
                        c2x,
                        c2y,
                        x,
                        y,
                    }
                }
            })
            .collect(),
    }
}

fn inverse(transform: &Affine) -> Option<Affine> {
    let det = transform.xx * transform.yy - transform.xy * transform.yx;
    if !det.is_finite() || det.abs() < f32::MIN_POSITIVE {
        return None;
    }
    let (xx, yx) = (transform.yy / det, -transform.yx / det);
    let (xy, yy) = (-transform.xy / det, transform.xx / det);
    let all = Affine {
        xx,
        yx,
        xy,
        yy,
        dx: -(xx * transform.dx + xy * transform.dy),
        dy: -(yx * transform.dx + yy * transform.dy),
    };
    [all.xx, all.yx, all.xy, all.yy, all.dx, all.dy]
        .iter()
        .all(|value| value.is_finite())
        .then_some(all)
}

/// Coverage of each pixel by the area inside `edges`, by the nonzero rule.
///
/// Each pixel row is sampled on [`SUBROWS`] lines. Along a line the cover
/// is exact: the spans where the winding is not zero, cut at the pixel
/// edges.
fn coverage(edges: &[Edge], width: usize, height: usize) -> Vec<f32> {
    let mut mask = vec![0.0_f32; width * height];
    if width == 0 || height == 0 {
        return mask;
    }
    // Edges by where they start, top first, so each line looks only at the
    // ones that reach it.
    let mut order: Vec<&Edge> = edges
        .iter()
        .filter(|edge| edge.start.y != edge.end.y)
        .collect();
    order.sort_by(|left, right| top(left).total_cmp(&top(right)));
    let weight = 1.0 / f32::from(SUBROWS);
    let mut next = 0;
    let mut active: Vec<&Edge> = Vec::new();
    let mut crossings: Vec<(f32, i32)> = Vec::new();
    let mut y_pixel = 0.0_f32;
    for row in mask.chunks_exact_mut(width) {
        for sub in 0..SUBROWS {
            let y = y_pixel + (f32::from(sub) + 0.5) * weight;
            while let Some(edge) = order.get(next)
                && top(edge) <= y
            {
                active.push(edge);
                next += 1;
            }
            active.retain(|edge| edge.start.y.max(edge.end.y) > y);
            crossings.clear();
            for edge in &active {
                let run = edge.end.y - edge.start.y;
                let x = edge.start.x + (y - edge.start.y) * (edge.end.x - edge.start.x) / run;
                if x.is_finite() {
                    crossings.push((x, if run > 0.0 { 1 } else { -1 }));
                }
            }
            crossings.sort_by(|left, right| left.0.total_cmp(&right.0));
            let mut winding = 0;
            let mut from = 0.0;
            for (x, turn) in &crossings {
                if winding != 0 {
                    add_span(row, from, *x, weight);
                }
                winding += turn;
                from = *x;
            }
        }
        y_pixel += 1.0;
    }
    for cover in &mut mask {
        *cover = cover.clamp(0.0, 1.0);
    }
    mask
}

fn top(edge: &Edge) -> f32 {
    edge.start.y.min(edge.end.y)
}

/// Add `weight` to the pixels of `row` from `from` to `to`, by how much of
/// each the span covers.
fn add_span(row: &mut [f32], from: f32, to: f32, weight: f32) {
    let mut left = from.max(0.0).floor();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let first = left as usize;
    for cover in row.iter_mut().skip(first) {
        let right = left + 1.0;
        let overlap = to.min(right) - from.max(left);
        if overlap > 0.0 {
            *cover += overlap * weight;
        }
        left = right;
        if left >= to {
            break;
        }
    }
}

/// A color as straight RGBA from 0 to 1. The color of the text is what
/// `text` says it is for now.
fn straight(color: &Color, text: &TextInk) -> Pixel {
    let alpha = color.alpha.clamp(0.0, 1.0);
    let Some([red, green, blue]) = color.rgb else {
        text.asked.set(true);
        return [text.level, text.level, text.level, alpha];
    };
    [
        f32::from(red) / 255.0,
        f32::from(green) / 255.0,
        f32::from(blue) / 255.0,
        alpha,
    ]
}

fn premultiplied([red, green, blue, alpha]: Pixel) -> Pixel {
    [red * alpha, green * alpha, blue * alpha, alpha]
}

fn source_over(over: Pixel, under: Pixel) -> Pixel {
    let keep = 1.0 - over[3];
    [
        over[0] + under[0] * keep,
        over[1] + under[1] * keep,
        over[2] + under[2] * keep,
        over[3] + under[3] * keep,
    ]
}

/// The straight color at `place` along a color line. Channels and alpha
/// change evenly between two stops, before they are premultiplied.
fn color_at(line: &ColorLine, place: f32, text: &TextInk) -> Option<Pixel> {
    let first = line.stops.first()?;
    let last = line.stops.last()?;
    if place.is_nan() {
        return None;
    }
    let span = last.offset - first.offset;
    let place = if span > 0.0 && place.is_finite() {
        match line.extend {
            Extend::Pad => place,
            Extend::Repeat => first.offset + (place - first.offset).rem_euclid(span),
            Extend::Reflect => {
                let along = (place - first.offset).rem_euclid(span * 2.0);
                first.offset
                    + if along > span {
                        span * 2.0 - along
                    } else {
                        along
                    }
            }
        }
    } else {
        place
    };
    if place <= first.offset {
        return Some(straight(&first.color, text));
    }
    if place >= last.offset {
        return Some(straight(&last.color, text));
    }
    for pair in line.stops.windows(2) {
        let [before, after] = pair else {
            continue;
        };
        if place < after.offset {
            let run = after.offset - before.offset;
            let along = if run > 0.0 {
                (place - before.offset) / run
            } else {
                1.0
            };
            let (from, to) = (straight(&before.color, text), straight(&after.color, text));
            return Some([
                from[0] + (to[0] - from[0]) * along,
                from[1] + (to[1] - from[1]) * along,
                from[2] + (to[2] - from[2]) * along,
                from[3] + (to[3] - from[3]) * along,
            ]);
        }
    }
    Some(straight(&last.color, text))
}

/// The two circles of a radial gradient. A point takes its place from the
/// circle between them that passes through it, the later one when two do.
struct Circles {
    c0: [f32; 2],
    r0: f32,
    between: [f32; 2],
    grow: f32,
    a: f32,
}

impl Circles {
    fn new(c0: [f32; 2], r0: f32, c1: [f32; 2], r1: f32) -> Self {
        let between = [c1[0] - c0[0], c1[1] - c0[1]];
        let grow = r1 - r0;
        Self {
            c0,
            r0,
            between,
            grow,
            a: between[0] * between[0] + between[1] * between[1] - grow * grow,
        }
    }

    fn at(&self, x: f32, y: f32) -> Option<f32> {
        let (px, py) = (x - self.c0[0], y - self.c0[1]);
        let b = px * self.between[0] + py * self.between[1] + self.r0 * self.grow;
        let c = px * px + py * py - self.r0 * self.r0;
        // A place counts only where its circle has a radius.
        let real = |place: f32| (self.r0 + place * self.grow >= 0.0).then_some(place);
        if self.a.abs() < 1.0e-6 {
            if b == 0.0 {
                return None;
            }
            return real(c / (2.0 * b));
        }
        let under = b * b - self.a * c;
        if under < 0.0 {
            return None;
        }
        let root = under.sqrt();
        let (one, other) = ((b + root) / self.a, (b - root) / self.a);
        real(one.max(other)).or_else(|| real(one.min(other)))
    }
}

/// `over` blended onto `under` by `mode`. Both are premultiplied.
pub(super) fn blend(mode: CompositeMode, over: Pixel, under: Pixel) -> Pixel {
    let (sa, ba) = (over[3], under[3]);
    let each = |mix: &dyn Fn(f32, f32) -> f32| {
        [
            mix(over[0], under[0]),
            mix(over[1], under[1]),
            mix(over[2], under[2]),
            mix(over[3], under[3]),
        ]
    };
    match mode {
        CompositeMode::Clear => CLEAR,
        CompositeMode::Src => over,
        CompositeMode::Dest => under,
        CompositeMode::SrcOver => source_over(over, under),
        CompositeMode::DestOver => source_over(under, over),
        CompositeMode::SrcIn => each(&|s, _| s * ba),
        CompositeMode::DestIn => each(&|_, b| b * sa),
        CompositeMode::SrcOut => each(&|s, _| s * (1.0 - ba)),
        CompositeMode::DestOut => each(&|_, b| b * (1.0 - sa)),
        CompositeMode::SrcAtop => each(&|s, b| s * ba + b * (1.0 - sa)),
        CompositeMode::DestAtop => each(&|s, b| b * sa + s * (1.0 - ba)),
        CompositeMode::Xor => each(&|s, b| s * (1.0 - ba) + b * (1.0 - sa)),
        CompositeMode::Plus => each(&|s, b| (s + b).min(1.0)),
        _ => mixed(mode, over, under),
    }
}

/// The blend modes that mix the two colors where both are present.
fn mixed(mode: CompositeMode, over: Pixel, under: Pixel) -> Pixel {
    let (sa, ba) = (over[3], under[3]);
    let plain = |pixel: Pixel| {
        if pixel[3] > 0.0 {
            [
                (pixel[0] / pixel[3]).clamp(0.0, 1.0),
                (pixel[1] / pixel[3]).clamp(0.0, 1.0),
                (pixel[2] / pixel[3]).clamp(0.0, 1.0),
            ]
        } else {
            [0.0; 3]
        }
    };
    let (cs, cb) = (plain(over), plain(under));
    let channels =
        |mix: &dyn Fn(f32, f32) -> f32| [mix(cb[0], cs[0]), mix(cb[1], cs[1]), mix(cb[2], cs[2])];
    let both = match mode {
        CompositeMode::Screen => channels(&screen),
        CompositeMode::Overlay => channels(&|b, s| hard_light(s, b)),
        CompositeMode::Darken => channels(&f32::min),
        CompositeMode::Lighten => channels(&f32::max),
        CompositeMode::ColorDodge => channels(&|b, s| {
            if b <= 0.0 {
                0.0
            } else if s >= 1.0 {
                1.0
            } else {
                (b / (1.0 - s)).min(1.0)
            }
        }),
        CompositeMode::ColorBurn => channels(&|b, s| {
            if b >= 1.0 {
                1.0
            } else if s <= 0.0 {
                0.0
            } else {
                1.0 - ((1.0 - b) / s).min(1.0)
            }
        }),
        CompositeMode::HardLight => channels(&hard_light),
        CompositeMode::SoftLight => channels(&|b, s| {
            if s <= 0.5 {
                b - (1.0 - 2.0 * s) * b * (1.0 - b)
            } else {
                let lit = if b <= 0.25 {
                    ((16.0 * b - 12.0) * b + 4.0) * b
                } else {
                    b.sqrt()
                };
                b + (2.0 * s - 1.0) * (lit - b)
            }
        }),
        CompositeMode::Difference => channels(&|b, s| (b - s).abs()),
        CompositeMode::Exclusion => channels(&|b, s| b + s - 2.0 * b * s),
        CompositeMode::Multiply => channels(&|b, s| b * s),
        CompositeMode::Hue => with_light(with_color(cs, color_of(cb)), light(cb)),
        CompositeMode::Saturation => with_light(with_color(cb, color_of(cs)), light(cb)),
        CompositeMode::Color => with_light(cs, light(cb)),
        CompositeMode::Luminosity => with_light(cb, light(cs)),
        // The modes that keep or drop whole pixels are handled before this.
        _ => cs,
    };
    let channel =
        |index: usize| over[index] * (1.0 - ba) + under[index] * (1.0 - sa) + sa * ba * both[index];
    [channel(0), channel(1), channel(2), sa + ba - sa * ba]
}

fn screen(backdrop: f32, source: f32) -> f32 {
    backdrop + source - backdrop * source
}

fn hard_light(backdrop: f32, source: f32) -> f32 {
    if source <= 0.5 {
        backdrop * 2.0 * source
    } else {
        screen(backdrop, 2.0 * source - 1.0)
    }
}

/// How light a color is.
fn light(color: [f32; 3]) -> f32 {
    0.3 * color[0] + 0.59 * color[1] + 0.11 * color[2]
}

/// How far a color is from gray.
fn color_of(color: [f32; 3]) -> f32 {
    color[0].max(color[1]).max(color[2]) - color[0].min(color[1]).min(color[2])
}

/// `color` made as light as `target`, kept inside 0 to 1.
fn with_light(color: [f32; 3], target: f32) -> [f32; 3] {
    let shift = target - light(color);
    let moved = color.map(|channel| channel + shift);
    let level = light(moved);
    let low = moved[0].min(moved[1]).min(moved[2]);
    let high = moved[0].max(moved[1]).max(moved[2]);
    moved.map(|channel| {
        let mut channel = channel;
        if low < 0.0 && level - low > 0.0 {
            channel = level + (channel - level) * level / (level - low);
        }
        if high > 1.0 && high - level > 0.0 {
            channel = level + (channel - level) * (1.0 - level) / (high - level);
        }
        channel
    })
}

/// `color` with its distance from gray set to `target`.
fn with_color(color: [f32; 3], target: f32) -> [f32; 3] {
    let low = color[0].min(color[1]).min(color[2]);
    let high = color[0].max(color[1]).max(color[2]);
    if high > low {
        color.map(|channel| (channel - low) * target / (high - low))
    } else {
        [0.0; 3]
    }
}

#[cfg(test)]
mod tests;
