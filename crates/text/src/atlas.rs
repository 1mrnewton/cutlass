//! One image for each glyph of a face: a color picture, a signed-distance
//! field from the outline, or gray coverage from a bitmap strike.
//!
//! Each glyph chooses for itself. A glyph the face draws in layered color
//! is painted from those layers. Failing that, a glyph with a picture in the
//! face's color strike takes that picture, so an emoji face whose outlines
//! are empty still draws. Any other glyph takes the face's route: its outline
//! when the face has outlines, and the gray strike when it does not. A gray
//! strike is never taken over an outline, because the small strikes old fonts
//! carry are worse than their outlines at any size but their own. Each glyph
//! is painted once. A later sample reads the cached texels. A face with
//! neither outlines nor a strike is an error.

mod layers;
mod strike;

use std::collections::HashMap;

use cutlass_font::{BitmapGlyph, BitmapPixels, Contour, Font, Outline, Segment};

use crate::TextError;

/// Atlas pixels across one em.
pub(crate) const ATLAS_EM: f32 = 64.0;
/// The coarsest grid a glyph too large for the usual one is painted on:
/// at this a cell holds a glyph thirty ems across.
const COARSEST_EM: f32 = 8.0;
/// Texels of distance stored on each side of the outline.
pub(crate) const SPREAD: f32 = 8.0;
const MAX_CELL: u16 = 256;
const MAX_SEGMENTS: usize = 4_096;
const MAX_DEPTH: u8 = 8;
const FLAT: f32 = 0.35;

/// Where a face's glyphs are drawn from when they have no color image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtlasRoute {
    SignedDistance,
    Bitmap,
}

/// The route of a face: its outlines when it has them, and its bitmap
/// strike when it does not. A glyph with a color image takes that image on
/// either route.
#[must_use]
pub fn route(font: &Font<'_>) -> AtlasRoute {
    if font.has_outlines() {
        AtlasRoute::SignedDistance
    } else {
        AtlasRoute::Bitmap
    }
}

/// How the cached bytes cover a glyph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlyphKind {
    /// Signed distance. `sdf` is one byte per texel.
    Distance,
    /// Gray or mono coverage. `sdf` is one byte per pixel, 0 outside and 255 covered.
    Coverage,
    /// Premultiplied color. `color` is RGBA, and `sdf` is empty.
    Color,
}

/// One cached glyph. Row 0 is the top of the image. `left` and `top` are font units.
#[derive(Clone, Debug)]
pub struct GlyphImage {
    pub width: u32,
    pub height: u32,
    pub left: f32,
    pub top: f32,
    /// Font units across one texel.
    pub unit: f32,
    /// Pixels per em this image was built at. Outlines use [`ATLAS_EM`].
    pub pixels_per_em: f32,
    pub kind: GlyphKind,
    pub sdf: Vec<u8>,
    pub color: Vec<u8>,
}

/// Glyph images for one face. A second request for a glyph does not paint it again.
#[derive(Debug, Default)]
pub struct Atlas {
    paints: u32,
    cells: HashMap<u16, GlyphImage>,
}

impl Atlas {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn paints(&self) -> u32 {
        self.paints
    }

    #[must_use]
    pub fn image(&self, glyph_id: u16) -> Option<&GlyphImage> {
        self.cells.get(&glyph_id)
    }

    /// Paint `glyph_id` on a miss. A cached glyph is returned as stored.
    pub fn glyph(&mut self, font: &Font<'_>, glyph_id: u16) -> Result<&GlyphImage, TextError> {
        if self.cells.contains_key(&glyph_id) {
            return match self.cells.get(&glyph_id) {
                Some(image) => Ok(image),
                None => Err(TextError::Limit),
            };
        }
        let image = paint(font, glyph_id)?;
        self.paints = self.paints.saturating_add(1);
        self.cells.insert(glyph_id, image);
        match self.cells.get(&glyph_id) {
            Some(image) => Ok(image),
            None => Err(TextError::Limit),
        }
    }

    /// Paint `glyph_id` for a title. A glyph that cannot be painted, because
    /// its outline or its picture does not read or is past a limit, is
    /// left blank: one glyph does not take the title with it. A glyph the
    /// face does not have, and a face with nothing to draw from, are still
    /// errors.
    pub(crate) fn drawn(
        &mut self,
        font: &Font<'_>,
        glyph_id: u16,
    ) -> Result<&GlyphImage, TextError> {
        if !self.cells.contains_key(&glyph_id) {
            let image = match paint(font, glyph_id) {
                Ok(image) => image,
                Err(
                    error @ (TextError::Font(cutlass_font::FontError::UnknownGlyph)
                    | TextError::MissingStrike),
                ) => return Err(error),
                Err(_) => blank(f32::from(font.units_per_em()) / ATLAS_EM),
            };
            self.paints = self.paints.saturating_add(1);
            self.cells.insert(glyph_id, image);
        }
        self.cells.get(&glyph_id).ok_or(TextError::Limit)
    }

    /// Take an image of `glyph_id` that was painted before, so it is not
    /// painted again. A glyph already here keeps the image it has.
    pub(crate) fn adopt(&mut self, glyph_id: u16, image: &GlyphImage) {
        self.cells.entry(glyph_id).or_insert_with(|| image.clone());
    }
}

/// Read `bytes` and paint one glyph. The length is the cached texel count.
pub fn paint_bytes(bytes: &[u8], glyph_id: u16) -> Result<usize, TextError> {
    let font = cutlass_font::read(bytes).map_err(TextError::Font)?;
    let mut atlas = Atlas::new();
    let image = atlas.glyph(&font, glyph_id)?;
    Ok(match image.kind {
        GlyphKind::Color => image.color.len() / 4,
        GlyphKind::Distance | GlyphKind::Coverage => image.sdf.len(),
    })
}

/// The glyph's image: its color layers when the face has them for it, else
/// its picture in the color strike, and otherwise what the face's route
/// draws.
fn paint(font: &Font<'_>, glyph_id: u16) -> Result<GlyphImage, TextError> {
    if let Some(glyph) = font.color_glyph(glyph_id).map_err(TextError::Font)? {
        return layers::paint(font, &glyph);
    }
    if let Some(bitmap) = font
        .color_bitmap(glyph_id, u32::from(MAX_CELL))
        .map_err(TextError::Font)?
        && bitmap.width > 0
        && bitmap.height > 0
    {
        return bitmap_image(font, bitmap);
    }
    match route(font) {
        AtlasRoute::SignedDistance => paint_outline(font, glyph_id),
        AtlasRoute::Bitmap => paint_bitmap(font, glyph_id),
    }
}

fn paint_outline(font: &Font<'_>, glyph_id: u16) -> Result<GlyphImage, TextError> {
    let outline = font.outline(glyph_id).map_err(TextError::Font)?;
    let unit = f32::from(font.units_per_em()) / ATLAS_EM;
    if !(unit.is_finite() && unit > 0.0) {
        return Err(TextError::Limit);
    }
    if outline.contours.is_empty() {
        return Ok(blank(unit));
    }
    let bounds = bounds(&outline)?;
    // A glyph too large for a cell at the usual grid is painted on a
    // coarser one, each half as fine as the last: a phrase drawn as one
    // glyph, a long swash, a tall bracket. It has the texels to spare.
    let mut em = ATLAS_EM;
    let (unit, width, height) = loop {
        let unit = f32::from(font.units_per_em()) / em;
        let spans = (
            cell_span(bounds.min_x, bounds.max_x, unit),
            cell_span(bounds.min_y, bounds.max_y, unit),
        );
        match spans {
            (Ok(width), Ok(height)) => break (unit, width, height),
            _ if em > COARSEST_EM => em /= 2.0,
            (Err(error), _) | (_, Err(error)) => return Err(error),
        }
    };
    if width == 0 || height == 0 {
        return Ok(blank(unit));
    }
    let pad = SPREAD * unit;
    let left = bounds.min_x - pad;
    let top = bounds.max_y + pad;
    let segments = flatten(&outline, left, top, unit)?;
    let sdf = raster(width, height, &segments)?;
    Ok(GlyphImage {
        width,
        height,
        left,
        top,
        unit,
        pixels_per_em: em,
        kind: GlyphKind::Distance,
        sdf,
        color: Vec::new(),
    })
}

fn blank(unit: f32) -> GlyphImage {
    GlyphImage {
        width: 0,
        height: 0,
        left: 0.0,
        top: 0.0,
        unit,
        pixels_per_em: ATLAS_EM,
        kind: GlyphKind::Distance,
        sdf: Vec::new(),
        color: Vec::new(),
    }
}

fn paint_bitmap(font: &Font<'_>, glyph_id: u16) -> Result<GlyphImage, TextError> {
    if !font.has_bitmap_strike() {
        return Err(TextError::MissingStrike);
    }
    let Some(bitmap) = font.bitmap(glyph_id).map_err(TextError::Font)? else {
        let unit = f32::from(font.units_per_em()) / ATLAS_EM;
        return Ok(blank(unit));
    };
    bitmap_image(font, bitmap)
}

/// One strike bitmap as an atlas image. A PNG picture is decoded here; one
/// larger than the atlas cell is refused before it is.
fn bitmap_image(font: &Font<'_>, bitmap: BitmapGlyph) -> Result<GlyphImage, TextError> {
    let ppem = f32::from(bitmap.ppem);
    let unit = f32::from(font.units_per_em()) / ppem;
    if !(unit.is_finite() && unit > 0.0 && ppem.is_finite() && ppem > 0.0) {
        return Err(TextError::Limit);
    }
    let edge = u32::from(MAX_CELL);
    if bitmap.width > edge || bitmap.height > edge {
        return Err(TextError::Limit);
    }
    let pixels = bitmap
        .width
        .checked_mul(bitmap.height)
        .ok_or(TextError::Limit)?;
    let rgba = usize::try_from(pixels)
        .ok()
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(TextError::Limit)?;
    let (kind, sdf, mut color) = match bitmap.pixels {
        BitmapPixels::Coverage(bytes) => {
            if bytes.len() != rgba / 4 {
                return Err(TextError::Limit);
            }
            (GlyphKind::Coverage, bytes, Vec::new())
        }
        BitmapPixels::Color(bytes) => {
            if bytes.len() != rgba {
                return Err(TextError::Limit);
            }
            (GlyphKind::Color, Vec::new(), bytes)
        }
        BitmapPixels::Png(bytes) => {
            let decoded = strike::decode(bytes, bitmap.width, bitmap.height, edge)?;
            if decoded.len() != rgba {
                return Err(TextError::Limit);
            }
            (GlyphKind::Color, Vec::new(), decoded)
        }
    };
    if bitmap.mirrored {
        strike::mirror(&mut color, bitmap.width)?;
    }
    #[allow(clippy::cast_precision_loss)]
    let bearing_x = bitmap.bearing_x as f32;
    #[allow(clippy::cast_precision_loss)]
    let bearing_y = bitmap.bearing_y as f32;
    Ok(GlyphImage {
        width: bitmap.width,
        height: bitmap.height,
        left: bearing_x * unit,
        top: bearing_y * unit,
        unit,
        pixels_per_em: ppem,
        kind,
        sdf,
        color,
    })
}

/// Pixel span of a font-unit interval, including the distance spread.
pub(crate) fn cell_span(min: f32, max: f32, unit: f32) -> Result<u32, TextError> {
    if !min.is_finite() || !max.is_finite() || !unit.is_finite() || unit <= 0.0 || max < min {
        return Err(TextError::Limit);
    }
    let span = (max + SPREAD * unit) - (min - SPREAD * unit);
    if !span.is_finite() || span < 0.0 {
        return Err(TextError::Limit);
    }
    let pixels = span / unit;
    if !pixels.is_finite() || pixels > f32::from(MAX_CELL) {
        return Err(TextError::Limit);
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let edge = pixels.ceil() as u32;
    if edge > u32::from(MAX_CELL) {
        Err(TextError::Limit)
    } else {
        Ok(edge)
    }
}

struct Bounds {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
}

fn bounds(outline: &Outline) -> Result<Bounds, TextError> {
    let mut bounds: Option<Bounds> = None;
    for contour in &outline.contours {
        include(&mut bounds, contour.x, contour.y)?;
        for segment in &contour.segments {
            match *segment {
                Segment::Line { x, y } => include(&mut bounds, x, y)?,
                Segment::Quad { cx, cy, x, y } => {
                    include(&mut bounds, cx, cy)?;
                    include(&mut bounds, x, y)?;
                }
                Segment::Cubic {
                    c1x,
                    c1y,
                    c2x,
                    c2y,
                    x,
                    y,
                } => {
                    include(&mut bounds, c1x, c1y)?;
                    include(&mut bounds, c2x, c2y)?;
                    include(&mut bounds, x, y)?;
                }
            }
        }
    }
    bounds.ok_or(TextError::Limit)
}

fn include(bounds: &mut Option<Bounds>, x: f32, y: f32) -> Result<(), TextError> {
    if !x.is_finite() || !y.is_finite() {
        return Err(TextError::Limit);
    }
    match bounds {
        Some(bounds) => {
            bounds.min_x = bounds.min_x.min(x);
            bounds.min_y = bounds.min_y.min(y);
            bounds.max_x = bounds.max_x.max(x);
            bounds.max_y = bounds.max_y.max(y);
        }
        None => {
            *bounds = Some(Bounds {
                min_x: x,
                min_y: y,
                max_x: x,
                max_y: y,
            });
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Point {
    x: f32,
    y: f32,
}

struct Edge {
    start: Point,
    end: Point,
}

fn flatten(outline: &Outline, left: f32, top: f32, unit: f32) -> Result<Vec<Edge>, TextError> {
    let mut edges = Vec::new();
    for contour in &outline.contours {
        let points = contour_points(contour, left, top, unit)?;
        if points.len() < 2 {
            continue;
        }
        for pair in points.windows(2) {
            if edges.len() >= MAX_SEGMENTS {
                return Err(TextError::Limit);
            }
            edges.push(Edge {
                start: pair[0],
                end: pair[1],
            });
        }
    }
    Ok(edges)
}

fn contour_points(
    contour: &Contour,
    left: f32,
    top: f32,
    unit: f32,
) -> Result<Vec<Point>, TextError> {
    let mut points = vec![map_point(contour.x, contour.y, left, top, unit)?];
    let mut x = contour.x;
    let mut y = contour.y;
    for segment in &contour.segments {
        if points.len() > MAX_SEGMENTS {
            return Err(TextError::Limit);
        }
        match *segment {
            Segment::Line { x: x1, y: y1 } => {
                points.push(map_point(x1, y1, left, top, unit)?);
                x = x1;
                y = y1;
            }
            Segment::Quad {
                cx,
                cy,
                x: x1,
                y: y1,
            } => {
                let start = map_point(x, y, left, top, unit)?;
                let control = map_point(cx, cy, left, top, unit)?;
                let end = map_point(x1, y1, left, top, unit)?;
                flatten_quad(&mut points, start, control, end, MAX_DEPTH)?;
                x = x1;
                y = y1;
            }
            Segment::Cubic {
                c1x,
                c1y,
                c2x,
                c2y,
                x: x1,
                y: y1,
            } => {
                let start = map_point(x, y, left, top, unit)?;
                let control_a = map_point(c1x, c1y, left, top, unit)?;
                let control_b = map_point(c2x, c2y, left, top, unit)?;
                let end = map_point(x1, y1, left, top, unit)?;
                flatten_cubic(&mut points, start, control_a, control_b, end, MAX_DEPTH)?;
                x = x1;
                y = y1;
            }
        }
    }
    Ok(points)
}

fn map_point(x: f32, y: f32, left: f32, top: f32, unit: f32) -> Result<Point, TextError> {
    let point = Point {
        x: (x - left) / unit,
        y: (top - y) / unit,
    };
    if point.x.is_finite() && point.y.is_finite() {
        Ok(point)
    } else {
        Err(TextError::Limit)
    }
}

fn flatten_quad(
    points: &mut Vec<Point>,
    start: Point,
    control: Point,
    end: Point,
    depth: u8,
) -> Result<(), TextError> {
    if points.len() > MAX_SEGMENTS {
        return Err(TextError::Limit);
    }
    if depth == 0 || flat(start, control, end) {
        points.push(end);
        return Ok(());
    }
    let left = mid(start, control);
    let right = mid(control, end);
    let center = mid(left, right);
    flatten_quad(points, start, left, center, depth - 1)?;
    flatten_quad(points, center, right, end, depth - 1)
}

fn flatten_cubic(
    points: &mut Vec<Point>,
    start: Point,
    control_a: Point,
    control_b: Point,
    end: Point,
    depth: u8,
) -> Result<(), TextError> {
    if points.len() > MAX_SEGMENTS {
        return Err(TextError::Limit);
    }
    if depth == 0 || (flat(start, control_a, end) && flat(start, control_b, end)) {
        points.push(end);
        return Ok(());
    }
    let ab = mid(start, control_a);
    let bc = mid(control_a, control_b);
    let cd = mid(control_b, end);
    let abc = mid(ab, bc);
    let bcd = mid(bc, cd);
    let center = mid(abc, bcd);
    flatten_cubic(points, start, ab, abc, center, depth - 1)?;
    flatten_cubic(points, center, bcd, cd, end, depth - 1)
}

fn mid(left: Point, right: Point) -> Point {
    Point {
        x: (left.x + right.x) * 0.5,
        y: (left.y + right.y) * 0.5,
    }
}

fn flat(start: Point, control: Point, end: Point) -> bool {
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let len = dx.hypot(dy);
    let cross = (control.x - start.x) * dy - (control.y - start.y) * dx;
    if len <= 1.0e-4 {
        (control.x - start.x).hypot(control.y - start.y) <= FLAT
    } else {
        cross.abs() / len <= FLAT
    }
}

/// The distance field of `edges` over a cell of `width` by `height`
/// texels: for each texel, how far its center is from the outline, inside
/// or outside.
///
/// A texel farther than [`SPREAD`] from every edge stores only which side
/// it is on, so each edge is measured against the texels within that
/// reach of it and no others; and a texel is inside by the edges that
/// cross its row to its right, which are gathered once for the row. The
/// field is the same, byte for byte, as measuring every texel against
/// every edge.
fn raster(width: u32, height: u32, edges: &[Edge]) -> Result<Vec<u8>, TextError> {
    let width_px = usize::try_from(width).map_err(|_| TextError::Limit)?;
    let height_px = usize::try_from(height).map_err(|_| TextError::Limit)?;
    let pixels = width_px.checked_mul(height_px).ok_or(TextError::Limit)?;
    if pixels > usize::from(MAX_CELL) * usize::from(MAX_CELL) {
        return Err(TextError::Limit);
    }
    if pixels == 0 {
        return Ok(Vec::new());
    }
    // The first and last texel, of `count`, whose center may be within
    // reach of the stretch from `low` to `high`.
    let reach = |low: f32, high: f32, count: usize| -> Option<(usize, usize)> {
        if !(low.is_finite() && high.is_finite()) {
            return Some((0, count - 1));
        }
        let first = (low - SPREAD - 1.0).floor().max(0.0);
        let last = (high + SPREAD + 1.0).ceil();
        if last < 0.0 {
            return None;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let (first, last) = (first as usize, last as usize);
        (first < count).then(|| (first, last.min(count - 1)))
    };
    let mut nearest = vec![f32::MAX; pixels];
    for edge in edges {
        let across = reach(
            edge.start.x.min(edge.end.x),
            edge.start.x.max(edge.end.x),
            width_px,
        );
        let down = reach(
            edge.start.y.min(edge.end.y),
            edge.start.y.max(edge.end.y),
            height_px,
        );
        let (Some((left, right)), Some((top, bottom))) = (across, down) else {
            continue;
        };
        for row in top..=bottom {
            #[allow(clippy::cast_precision_loss)]
            let y = row as f32 + 0.5;
            let line = &mut nearest[row * width_px..(row + 1) * width_px];
            for (col, texel) in line.iter_mut().enumerate().take(right + 1).skip(left) {
                #[allow(clippy::cast_precision_loss)]
                let point = Point {
                    x: col as f32 + 0.5,
                    y,
                };
                let distance = segment_distance(point, edge);
                if distance < *texel {
                    *texel = distance;
                }
            }
        }
    }
    let mut sdf = vec![0_u8; pixels];
    // The edges that cross a row, each with where it crosses and which
    // way it goes.
    let mut crossings: Vec<(f32, i32)> = Vec::new();
    for row in 0..height_px {
        #[allow(clippy::cast_precision_loss)]
        let y = row as f32 + 0.5;
        crossings.clear();
        for edge in edges {
            let (y0, y1) = (edge.start.y, edge.end.y);
            if y0 <= y && y1 > y {
                crossings.push((cross_x(edge, y), 1));
            } else if y1 <= y && y0 > y {
                crossings.push((cross_x(edge, y), -1));
            }
        }
        for col in 0..width_px {
            #[allow(clippy::cast_precision_loss)]
            let x = col as f32 + 0.5;
            let wind: i32 = crossings
                .iter()
                .filter(|(cross, _)| x < *cross)
                .map(|(_, turn)| *turn)
                .sum();
            let distance = nearest[row * width_px + col];
            let signed = if wind != 0 { -distance } else { distance };
            sdf[row * width_px + col] = encode(signed);
        }
    }
    Ok(sdf)
}

/// The distance field as it was first made: every texel against every
/// edge. Kept to check the quick one against.
#[cfg(test)]
fn raster_whole(width: u32, height: u32, edges: &[Edge]) -> Vec<u8> {
    let mut sdf = Vec::new();
    for row in 0..height {
        for col in 0..width {
            let point = Point {
                x: pixel(col) + 0.5,
                y: pixel(row) + 0.5,
            };
            let distance = edges
                .iter()
                .map(|edge| segment_distance(point, edge))
                .fold(f32::MAX, f32::min);
            let inside = winding(point, edges) != 0;
            sdf.push(encode(if inside { -distance } else { distance }));
        }
    }
    sdf
}

fn segment_distance(point: Point, edge: &Edge) -> f32 {
    let dx = edge.end.x - edge.start.x;
    let dy = edge.end.y - edge.start.y;
    let len2 = dx * dx + dy * dy;
    if len2 <= f32::EPSILON {
        return (point.x - edge.start.x).hypot(point.y - edge.start.y);
    }
    let t = ((point.x - edge.start.x) * dx + (point.y - edge.start.y) * dy) / len2;
    let t = t.clamp(0.0, 1.0);
    let x = edge.start.x + t * dx;
    let y = edge.start.y + t * dy;
    (point.x - x).hypot(point.y - y)
}

#[cfg(test)]
fn winding(point: Point, edges: &[Edge]) -> i32 {
    let mut wind = 0_i32;
    for edge in edges {
        let y0 = edge.start.y;
        let y1 = edge.end.y;
        if y0 <= point.y && y1 > point.y {
            if point.x < cross_x(edge, point.y) {
                wind += 1;
            }
        } else if y1 <= point.y && y0 > point.y && point.x < cross_x(edge, point.y) {
            wind -= 1;
        }
    }
    wind
}

fn cross_x(edge: &Edge, y: f32) -> f32 {
    let y0 = edge.start.y;
    let y1 = edge.end.y;
    let span = y1 - y0;
    if span.abs() <= f32::EPSILON {
        return edge.start.x;
    }
    edge.start.x + (y - y0) * (edge.end.x - edge.start.x) / span
}

#[cfg(test)]
fn pixel(value: u32) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    {
        value as f32
    }
}

fn encode(signed: f32) -> u8 {
    if !signed.is_finite() {
        return 0;
    }
    let t = (0.5 - signed / (SPREAD * 2.0)).clamp(0.0, 1.0);
    let byte = (t * 255.0).round();
    if byte <= 0.0 {
        0
    } else if byte >= 255.0 {
        255
    } else {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            byte as u8
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Atlas, AtlasRoute, cell_span, paint_bytes, route};
    use crate::TextError;
    use cutlass_font::{FontError, read};

    const TTF: &[u8] = include_bytes!("../../font/fixtures/sample.ttf");
    const OTF: &[u8] = include_bytes!("../../font/fixtures/sample.otf");
    const NO_OUTLINE: &[u8] = include_bytes!("../../font/fixtures/no-outline.otf");
    const COLOR: &[u8] = include_bytes!("../../font/fixtures/color-strike.otf");
    const GRAY: &[u8] = include_bytes!("../../font/fixtures/gray-strike.otf");

    fn texel(image: &super::GlyphImage, x: f32, y: f32) -> u8 {
        let col = ((x - image.left) / image.unit).floor();
        let row = ((image.top - y) / image.unit).floor();
        assert!(col >= 0.0 && row >= 0.0, "point outside the image");
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let col = col as u32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let row = row as u32;
        assert!(col < image.width && row < image.height);
        let offset = usize::try_from(row * image.width + col).expect("offset");
        image.sdf[offset]
    }

    #[test]
    fn an_outlined_glyph_is_inside_the_contour_and_cached() {
        let font = read(TTF).expect("ttf");
        assert_eq!(route(&font), AtlasRoute::SignedDistance);
        let mut atlas = Atlas::new();
        let (inside, edge, outside, first, width) = {
            let image = atlas.glyph(&font, 2).expect("A");
            (
                texel(image, 300.0, 400.0),
                texel(image, 100.0, 400.0),
                texel(image, 0.0, 400.0),
                image.sdf.clone(),
                image.width,
            )
        };
        assert!(inside > edge);
        assert!(edge > outside);
        assert!(inside >= 250);
        assert!(outside <= 80);
        assert!(first.len() <= 256 * 256);
        let paints = atlas.paints();
        let again = atlas.glyph(&font, 2).expect("cached");
        let again_sdf = again.sdf.clone();
        let again_width = again.width;
        assert_eq!(atlas.paints(), paints);
        assert_eq!(again_sdf, first);
        assert_eq!(again_width, width);

        let curve_len = atlas.glyph(&font, 6).expect("quad").sdf.len();
        assert!(curve_len > 0);
        assert_eq!(atlas.paints(), paints + 1);
        let _cached = atlas.glyph(&font, 6).expect("cached curve").sdf.len();
        assert_eq!(atlas.paints(), paints + 1);

        let otf = read(OTF).expect("otf");
        assert_eq!(route(&otf), AtlasRoute::SignedDistance);
        assert!(paint_bytes(OTF, 2).expect("otf") > 0);
    }

    #[test]
    fn a_second_request_does_not_paint_an_empty_glyph() {
        let font = read(TTF).expect("ttf");
        let mut atlas = Atlas::new();
        let empty = atlas.glyph(&font, 1).expect("space").sdf.is_empty();
        assert!(empty);
        assert_eq!(atlas.paints(), 1);
        let _cached = atlas.glyph(&font, 1).expect("cached").sdf.len();
        assert_eq!(atlas.paints(), 1);
    }

    #[test]
    fn a_font_without_outlines_is_not_a_distance_field() {
        let font = read(NO_OUTLINE).expect("cff2");
        assert_eq!(route(&font), AtlasRoute::Bitmap);
        let mut missing = Atlas::new();
        let error = missing.glyph(&font, 0).expect_err("strike");
        assert!(matches!(error, TextError::MissingStrike));
        assert_eq!(error.to_string(), "font has no bitmap strike");
        assert_eq!(missing.paints(), 0);
        assert!(missing.image(0).is_none());

        let color_font = read(COLOR).expect("color");
        let gray_font = read(GRAY).expect("gray");
        let outline_font = read(TTF).expect("ttf");
        assert_eq!(route(&color_font), AtlasRoute::Bitmap);
        assert_eq!(route(&gray_font), AtlasRoute::Bitmap);
        assert_eq!(route(&outline_font), AtlasRoute::SignedDistance);

        let mut color = Atlas::new();
        let mut gray = Atlas::new();
        let mut outline = Atlas::new();
        let cached = {
            let image = color.glyph(&color_font, 1).expect("color");
            assert_eq!(image.kind, super::GlyphKind::Color);
            assert!(image.sdf.is_empty());
            assert_eq!(&image.color[..4], &[255, 0, 0, 255]);
            assert_eq!(&image.color[4..8], &[0, 128, 0, 128]);
            image.color.clone()
        };
        let paints = color.paints();
        assert_eq!(paints, 1);
        let again = color.glyph(&color_font, 1).expect("cached").color.clone();
        assert_eq!(color.paints(), paints);
        assert_eq!(again, cached);

        let coverage = gray.glyph(&gray_font, 1).expect("gray");
        assert_eq!(coverage.kind, super::GlyphKind::Coverage);
        assert!(coverage.color.is_empty());
        assert_eq!(coverage.sdf, [255, 0, 128, 64]);
        assert_eq!(gray.paints(), 1);
        let _cached = gray.glyph(&gray_font, 1).expect("cached gray").sdf.clone();
        assert_eq!(gray.paints(), 1);

        let distance = outline.glyph(&outline_font, 2).expect("outline");
        assert_eq!(distance.kind, super::GlyphKind::Distance);
        assert!(!distance.sdf.is_empty());
        assert_eq!(outline.paints(), 1);
        assert_eq!(color.paints(), 1);
        assert_eq!(gray.paints(), 1);
    }

    #[test]
    fn a_bad_glyph_or_a_truncated_font_is_an_error() {
        let font = read(TTF).expect("ttf");
        let mut atlas = Atlas::new();
        assert!(matches!(
            atlas.glyph(&font, u16::MAX),
            Err(TextError::Font(FontError::UnknownGlyph))
        ));
        assert_eq!(atlas.paints(), 0);
        assert!(matches!(
            paint_bytes(&[], 2),
            Err(TextError::Font(FontError::Truncated))
        ));
        assert!(matches!(
            paint_bytes(&[0, 1, 0, 0], 2),
            Err(TextError::Font(FontError::Truncated))
        ));
        let unit = 1_000.0_f32 / super::ATLAS_EM;
        assert!(matches!(
            cell_span(0.0, 20_000.0, unit),
            Err(TextError::Limit)
        ));
    }

    const SBIX: &[u8] = include_bytes!("../../font/fixtures/sbix-png.ttf");
    const CBDT: &[u8] = include_bytes!("../../font/fixtures/cbdt-png.ttf");

    /// The emoji face a Mac installs. A host without it skips the test.
    const SYSTEM_EMOJI: &str = "/System/Library/Fonts/Apple Color Emoji.ttc";

    #[test]
    fn the_system_emoji_face_draws_color_pictures_where_it_is_installed() {
        let Ok(bytes) = std::fs::read(SYSTEM_EMOJI) else {
            return;
        };
        let font = cutlass_font::read_collection(&bytes, 0).expect("face");
        let glyph = font
            .glyph_id('\u{1F600}')
            .expect("map")
            .expect("grinning face");
        let mut atlas = Atlas::new();
        let image = atlas.glyph(&font, glyph).expect("picture");
        assert_eq!(image.kind, super::GlyphKind::Color);
        // The largest strike that fits the cell, standing on the baseline.
        assert!((64..=256).contains(&image.width) && image.width == image.height);
        assert_eq!(image.top, super::pixel(image.height) * image.unit);
        let at = |x: u32, y: u32| {
            let offset = usize::try_from((y * image.width + x) * 4).expect("offset");
            [
                image.color[offset],
                image.color[offset + 1],
                image.color[offset + 2],
                image.color[offset + 3],
            ]
        };
        // The face is yellow a third of the way down and clear in the corner.
        let [red, green, blue, alpha] = at(image.width / 2, image.height / 3);
        assert_eq!(alpha, 255);
        assert!(
            red > 200 && green > 150 && blue < 120,
            "{red} {green} {blue}"
        );
        assert_eq!(at(0, 0), [0, 0, 0, 0]);
        // Premultiplied: no channel is above its alpha.
        assert!(
            image
                .color
                .as_chunks::<4>()
                .0
                .iter()
                .all(|[red, green, blue, alpha]| red <= alpha && green <= alpha && blue <= alpha)
        );
        // Glyphs across the face paint: pictures, flipped pictures, the
        // letters with outlines, and the blank ones.
        let mut pictures = 0;
        for glyph in (0..font.glyph_count()).step_by(41) {
            let image = atlas.glyph(&font, glyph).expect("glyph");
            if image.kind == super::GlyphKind::Color {
                pictures += 1;
            }
        }
        assert!(pictures > 40, "{pictures} pictures");
        let flipped = (0..font.glyph_count()).find(|glyph| {
            font.color_bitmap(*glyph, 256)
                .ok()
                .flatten()
                .is_some_and(|bitmap| bitmap.mirrored)
        });
        if let Some(flipped) = flipped {
            let image = atlas.glyph(&font, flipped).expect("flipped");
            assert_eq!(image.kind, super::GlyphKind::Color);
            assert!(image.color.iter().any(|byte| *byte != 0));
        }
    }

    /// The distance field of a glyph made the slow way, every texel
    /// against every edge; `None` for a glyph with no outline.
    fn whole_field(font: &cutlass_font::Font<'_>, glyph: u16) -> Option<Vec<u8>> {
        let outline = font.outline(glyph).ok()?;
        if outline.contours.is_empty() {
            return None;
        }
        let unit = f32::from(font.units_per_em()) / super::ATLAS_EM;
        let bounds = super::bounds(&outline).ok()?;
        let width = cell_span(bounds.min_x, bounds.max_x, unit).ok()?;
        let height = cell_span(bounds.min_y, bounds.max_y, unit).ok()?;
        let pad = super::SPREAD * unit;
        let edges = super::flatten(&outline, bounds.min_x - pad, bounds.max_y + pad, unit).ok()?;
        Some(super::raster_whole(width, height, &edges))
    }

    #[test]
    fn a_glyph_too_large_for_the_usual_grid_is_painted_on_a_coarser_one() {
        let font = read(include_bytes!("../../font/fixtures/wide.ttf")).expect("wide");
        let mut atlas = Atlas::new();
        // Six ems across: 32 texels to the em, and its box is inside.
        let image = atlas.glyph(&font, 1).expect("six ems").clone();
        assert_eq!(image.kind, super::GlyphKind::Distance);
        assert_eq!(image.pixels_per_em, 32.0);
        assert_eq!(image.unit, 1_000.0 / 32.0);
        assert!(image.width <= 256 && image.height <= 256);
        assert_eq!((image.width, image.height), (6 * 32 + 16, 32 + 16));
        // The field's own reach is the same number of texels, so the
        // box starts that far in from the left and the top.
        assert_eq!(image.left, -8.0 * image.unit);
        assert_eq!(image.top, 1_000.0 + 8.0 * image.unit);
        assert!(texel(&image, 3_000.0, 500.0) >= 250);
        assert!(texel(&image, -200.0, 500.0) < 100);
        assert_eq!(image.sdf[0], 0);
        // Twenty ems across: 8 to the em.
        let wider = atlas.glyph(&font, 2).expect("twenty ems");
        assert_eq!(wider.pixels_per_em, 8.0);
        assert!(wider.width <= 256);
        // Past the coarsest grid a glyph is refused, and a title leaves
        // it blank.
        assert!(matches!(atlas.glyph(&font, 3), Err(TextError::Limit)));
        assert_eq!(atlas.drawn(&font, 3).expect("blank").width, 0);
        // A title of the wide glyph draws it as wide as its advance.
        let wide = title(include_bytes!("../../font/fixtures/wide.ttf"), None, "A");
        assert!(count(&wide, |pixel| pixel == [255, 255, 255, 255]) > 6 * 64 * 64 / 2);
    }

    #[test]
    fn the_quick_distance_field_is_the_whole_one_byte_for_byte() {
        // Every glyph of the fixture faces, and a spread of the glyphs of
        // faces a Mac installs, where they are: Latin, and Han characters
        // of many strokes drawn with cubic curves.
        let mut compared = 0;
        let mut check = |bytes: &[u8], index: Option<u32>, every: usize| {
            let font = match index {
                Some(index) => cutlass_font::read_collection(bytes, index),
                None => read(bytes),
            }
            .expect("face");
            let mut atlas = Atlas::new();
            for glyph in (0..font.glyph_count()).step_by(every) {
                let Some(whole) = whole_field(&font, glyph) else {
                    continue;
                };
                let image = atlas.glyph(&font, glyph).expect("glyph");
                assert!(image.sdf == whole, "glyph {glyph} differs");
                compared += 1;
            }
        };
        check(include_bytes!("../../font/fixtures/sample.ttf"), None, 1);
        check(include_bytes!("../../font/fixtures/sample.otf"), None, 1);
        check(include_bytes!("../../font/fixtures/shaping.ttf"), None, 1);
        for (path, index, every) in [
            ("/System/Library/Fonts/Supplemental/Arial.ttf", None, 97),
            ("/System/Library/Fonts/Hiragino Sans GB.ttc", Some(0), 1_499),
            ("/System/Library/Fonts/Supplemental/Zapfino.ttf", None, 211),
        ] {
            if let Ok(bytes) = std::fs::read(path) {
                check(&bytes, index, every);
            }
        }
        assert!(compared > 30, "{compared} glyphs");
    }

    #[test]
    fn the_quick_distance_field_agrees_on_edges_of_every_kind() {
        use super::{Edge, Point};
        // Edges made from a fixed sequence: inside the cell and far
        // outside it, of no length, level, upright, and through the
        // centers of texels.
        let mut seed = 0x2545_F491_u32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for round in 0..300 {
            let (width, height) = (1 + next() % 40, 1 + next() % 40);
            let count = next() % 24;
            let mut edges = Vec::new();
            for _ in 0..count {
                #[allow(clippy::cast_precision_loss)]
                let mut place = |limit: u32| match next() % 6 {
                    0 => (next() % (limit + 1)) as f32,
                    1 => (next() % (limit + 1)) as f32 + 0.5,
                    2 => (next() % 400) as f32 - 200.0,
                    _ => (next() % (limit * 16 + 1)) as f32 / 16.0,
                };
                let start = Point {
                    x: place(width),
                    y: place(height),
                };
                let end = match round % 4 {
                    0 => Point {
                        x: start.x,
                        y: start.y,
                    },
                    1 => Point {
                        x: place(width),
                        y: start.y,
                    },
                    _ => Point {
                        x: place(width),
                        y: place(height),
                    },
                };
                edges.push(Edge { start, end });
            }
            let quick = super::raster(width, height, &edges).expect("field");
            assert!(
                quick == super::raster_whole(width, height, &edges),
                "round {round}: {width} by {height}, {count} edges"
            );
        }
        // No edges: every texel is outside.
        assert_eq!(super::raster(3, 2, &[]).expect("field"), [0; 6]);
        assert!(super::raster(0, 5, &[]).expect("field").is_empty());
        // A cell past the limit is refused before anything is measured.
        assert!(matches!(
            super::raster(4_096, 4_096, &[]),
            Err(TextError::Limit)
        ));
    }

    #[test]
    fn a_glyph_of_many_edges_is_painted_quickly() {
        // The glyph of most edges in the shaping fixture, many times; and
        // where a Mac's Han face is, a hundred characters of many strokes.
        let started = std::time::Instant::now();
        if let Ok(bytes) = std::fs::read("/System/Library/Fonts/Hiragino Sans GB.ttc") {
            let font = cutlass_font::read_collection(&bytes, 0).expect("face");
            let first = font.glyph_id('\u{9F8D}').expect("map").unwrap_or(1);
            let mut atlas = Atlas::new();
            for glyph in first..first.saturating_add(100).min(font.glyph_count()) {
                let _image = atlas.glyph(&font, glyph).expect("glyph");
            }
        }
        let font = read(include_bytes!("../../font/fixtures/shaping.ttf")).expect("face");
        for _ in 0..200 {
            let mut atlas = Atlas::new();
            for glyph in 0..font.glyph_count() {
                let _image = atlas.glyph(&font, glyph).expect("glyph");
            }
        }
        // Unoptimized, this is well under a second with each edge
        // measured against the texels near it alone.
        assert!(started.elapsed() < std::time::Duration::from_secs(20));
    }

    /// Paint every glyph of the faces named in `CUTLASS_COLOR_FACES` (paths
    /// with `:` between them), and say what each face gave. With
    /// `CUTLASS_COLOR_SHEET` set, the color pictures of each face are also
    /// written side by side to `<that>-<n>.rgba`, 12 pictures of 192 pixels
    /// to a row, for looking at.
    #[test]
    #[ignore = "reads the color faces named in CUTLASS_COLOR_FACES"]
    #[allow(clippy::print_stdout)]
    fn the_named_color_faces_paint() {
        const CELL: usize = 192;
        const ACROSS: usize = 12;
        let Ok(paths) = std::env::var("CUTLASS_COLOR_FACES") else {
            return;
        };
        let sheet = std::env::var("CUTLASS_COLOR_SHEET").ok();
        let every = std::env::var("CUTLASS_COLOR_EVERY")
            .ok()
            .and_then(|every| every.parse::<usize>().ok())
            .unwrap_or(1);
        for (number, path) in paths.split(':').enumerate() {
            let bytes = std::fs::read(path).expect("face file");
            let font = cutlass_font::read_collection(&bytes, 0)
                .or_else(|_| read(&bytes))
                .expect("face");
            let started = std::time::Instant::now();
            let mut atlas = Atlas::new();
            let (mut color, mut distance, mut coverage, mut failed) = (0, 0, 0, 0);
            let mut pictures: Vec<super::GlyphImage> = Vec::new();
            for glyph in 0..font.glyph_count() {
                match atlas.glyph(&font, glyph) {
                    Ok(image) => match image.kind {
                        super::GlyphKind::Color => {
                            assert_eq!(
                                image.color.len(),
                                usize::try_from(image.width * image.height * 4).expect("size")
                            );
                            assert!(
                                image.color.as_chunks::<4>().0.iter().all(
                                    |[red, green, blue, alpha]| red <= alpha
                                        && green <= alpha
                                        && blue <= alpha
                                ),
                                "glyph {glyph} is not premultiplied"
                            );
                            if color % every == 0 {
                                pictures.push(image.clone());
                            }
                            color += 1;
                        }
                        super::GlyphKind::Distance => distance += 1,
                        super::GlyphKind::Coverage => coverage += 1,
                    },
                    Err(error) => {
                        if failed < 8 {
                            println!("  glyph {glyph}: {error:?}");
                        }
                        failed += 1;
                    }
                }
            }
            println!(
                "{path}: {} glyphs, {color} color, {distance} outline, {coverage} gray, {failed} failed, {:?}",
                font.glyph_count(),
                started.elapsed()
            );
            let Some(sheet) = &sheet else {
                continue;
            };
            let down = pictures.len().div_ceil(ACROSS).max(1);
            let mut rgba = vec![0_u8; ACROSS * CELL * down * CELL * 4];
            for (index, image) in pictures.iter().enumerate() {
                let (left, top) = ((index % ACROSS) * CELL, (index / ACROSS) * CELL);
                let width = usize::try_from(image.width).expect("width");
                let height = usize::try_from(image.height).expect("height");
                // The em square of the picture: where the glyph sits in it.
                let origin_x = (image.left / image.unit).round();
                let origin_y = (image.pixels_per_em - image.top / image.unit).round();
                for y in 0..height {
                    for x in 0..width {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
                        let (cell_x, cell_y) = (
                            (x as f32 + origin_x + 16.0) as i64,
                            (y as f32 + origin_y + 24.0) as i64,
                        );
                        if !(0..CELL as i64).contains(&cell_x)
                            || !(0..CELL as i64).contains(&cell_y)
                        {
                            continue;
                        }
                        let to = ((top + usize::try_from(cell_y).expect("y")) * ACROSS * CELL
                            + left
                            + usize::try_from(cell_x).expect("x"))
                            * 4;
                        let from = (y * width + x) * 4;
                        rgba[to..to + 4].copy_from_slice(&image.color[from..from + 4]);
                    }
                }
            }
            let name = format!("{sheet}-{number}.rgba");
            std::fs::write(&name, &rgba).expect("sheet");
            println!("  {name}: {} by {}", ACROSS * CELL, down * CELL);
        }
    }

    /// The fixture's two by two color picture, premultiplied.
    const RGBA: [u8; 16] = [255, 0, 0, 255, 0, 128, 0, 128, 0, 0, 255, 255, 0, 0, 0, 0];

    fn color_of(atlas: &mut Atlas, font: &cutlass_font::Font<'_>, glyph: u16) -> Vec<u8> {
        let image = atlas.glyph(font, glyph).expect("glyph");
        assert_eq!(image.kind, super::GlyphKind::Color, "glyph {glyph}");
        assert_eq!((image.width, image.height), (2, 2), "glyph {glyph}");
        assert!(image.sdf.is_empty());
        image.color.clone()
    }

    #[test]
    fn a_face_with_empty_outlines_and_a_color_strike_draws_its_pictures() {
        let font = read(SBIX).expect("sbix");
        // The face has outlines, and the picture glyph's outline is empty.
        assert_eq!(route(&font), AtlasRoute::SignedDistance);
        assert!(font.outline(2).expect("outline").contours.is_empty());
        let mut atlas = Atlas::new();
        let image = atlas.glyph(&font, 2).expect("picture");
        assert_eq!(image.kind, super::GlyphKind::Color);
        assert_eq!(image.color, RGBA);
        // Two pixels to the em of 1000 units, the bottom edge on the baseline.
        assert_eq!(image.pixels_per_em, 2.0);
        assert_eq!(image.unit, 500.0);
        assert_eq!((image.left, image.top), (0.0, 1_000.0));
        assert_eq!(atlas.paints(), 1);
        let _cached = atlas.glyph(&font, 2).expect("cached");
        assert_eq!(atlas.paints(), 1);
        // A letter of the same face still draws from its outline.
        let letter = atlas.glyph(&font, 1).expect("letter");
        assert_eq!(letter.kind, super::GlyphKind::Distance);
        assert!(letter.sdf.iter().any(|texel| *texel >= 250));
        assert!(letter.color.is_empty());
    }

    #[test]
    fn a_face_of_pictures_alone_draws_them_and_leaves_the_rest_blank() {
        let font = read(include_bytes!("../../font/fixtures/cbdt-only.ttf")).expect("pictures");
        assert_eq!(route(&font), AtlasRoute::Bitmap);
        let mut atlas = Atlas::new();
        let image = atlas.glyph(&font, 1).expect("picture");
        assert_eq!(image.kind, super::GlyphKind::Color);
        assert_eq!(image.color, RGBA);
        // The missing glyph has no picture, and that is a blank and no error.
        let blank = atlas.glyph(&font, 0).expect("blank");
        assert_eq!((blank.width, blank.height), (0, 0));
    }

    #[test]
    fn every_png_color_type_becomes_premultiplied_color() {
        let font = read(SBIX).expect("sbix");
        let mut atlas = Atlas::new();
        // A palette whose alphas are full, half, none, and absent.
        assert_eq!(
            color_of(&mut atlas, &font, 3),
            [255, 0, 0, 255, 0, 128, 0, 128, 0, 0, 0, 0, 255, 255, 0, 255]
        );
        // Gray with one value that is clear.
        assert_eq!(
            color_of(&mut atlas, &font, 4),
            [
                0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255, 0, 0, 0, 0
            ]
        );
        // 16 bits a sample.
        assert_eq!(
            color_of(&mut atlas, &font, 5),
            [255, 0, 0, 255, 0, 64, 0, 128, 0, 0, 255, 255, 0, 0, 0, 0]
        );
        // Gray with alpha.
        assert_eq!(
            color_of(&mut atlas, &font, 6),
            [
                255, 255, 255, 255, 50, 50, 50, 128, 0, 0, 0, 255, 0, 0, 0, 0
            ]
        );
        // Color with no alpha is opaque.
        assert_eq!(
            color_of(&mut atlas, &font, 12),
            [
                10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 255
            ]
        );
        // One bit a pixel.
        assert_eq!(
            color_of(&mut atlas, &font, 15),
            [
                255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255
            ]
        );
    }

    #[test]
    fn a_strike_picture_is_placed_flipped_and_sized_as_the_font_says() {
        let font = read(SBIX).expect("sbix");
        let mut atlas = Atlas::new();
        // A flipped record swaps the columns of each row.
        assert_eq!(
            color_of(&mut atlas, &font, 7),
            [0, 128, 0, 128, 255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 255, 255]
        );
        assert_eq!(color_of(&mut atlas, &font, 8), RGBA);
        // Origin offsets of 3 right and 1 down, at 500 units a pixel.
        let moved = atlas.glyph(&font, 14).expect("moved");
        assert_eq!((moved.left, moved.top), (1_500.0, 500.0));
        // The 600-pixel strike does not fit the cell, so the next one is used.
        let fitting = atlas.glyph(&font, 2).expect("fits");
        assert_eq!(fitting.pixels_per_em, 2.0);

        let font = read(CBDT).expect("cbdt");
        assert_eq!(route(&font), AtlasRoute::SignedDistance);
        let mut atlas = Atlas::new();
        assert_eq!(color_of(&mut atlas, &font, 1), RGBA);
        let small = atlas.glyph(&font, 1).expect("small metrics");
        assert_eq!((small.left, small.top), (0.0, 1_000.0));
        let big = atlas.glyph(&font, 2).expect("big metrics");
        assert_eq!((big.left, big.top), (500.0, 1_500.0));
        assert_eq!(&big.color[..8], &[255, 0, 0, 255, 0, 128, 0, 128]);
        let shared = atlas.glyph(&font, 3).expect("shared metrics");
        assert_eq!((shared.left, shared.top), (-500.0, 1_000.0));
        assert_eq!(&shared.color[..8], &[255, 255, 255, 255, 50, 50, 50, 128]);
        // A glyph with no picture and an empty outline is blank, not an error.
        let empty = atlas.glyph(&font, 0).expect("no picture");
        assert_eq!((empty.width, empty.height), (0, 0));
    }

    #[test]
    fn a_hostile_png_picture_is_an_error() {
        let font = read(SBIX).expect("sbix");
        let mut atlas = Atlas::new();
        // A header that declares a hundred thousand pixels each way is
        // refused before anything is decoded.
        assert!(matches!(atlas.glyph(&font, 9), Err(TextError::Limit)));
        // A picture cut short.
        assert!(matches!(
            atlas.glyph(&font, 10),
            Err(TextError::Font(FontError::Truncated))
        ));
        // A chunk whose check value is wrong.
        assert!(matches!(
            atlas.glyph(&font, 11),
            Err(TextError::Font(FontError::BadTable))
        ));
        // A graphic that is not a PNG picture.
        assert!(matches!(
            atlas.glyph(&font, 13),
            Err(TextError::Font(FontError::Unsupported))
        ));
        assert_eq!(atlas.paints(), 0);
        for glyph in [9, 10, 11, 13] {
            assert!(atlas.image(glyph).is_none());
        }
        // A picture larger than the edge it may have is refused, and so is
        // one whose header disagrees with the size it was announced with.
        let picture = match font
            .color_bitmap(2, 256)
            .expect("glyph")
            .expect("png")
            .pixels
        {
            cutlass_font::BitmapPixels::Png(bytes) => bytes,
            other => panic!("not a png: {other:?}"),
        };
        assert!(super::strike::decode(picture.clone(), 2, 2, 2).is_ok());
        assert!(matches!(
            super::strike::decode(picture.clone(), 2, 2, 1),
            Err(TextError::Limit)
        ));
        assert!(matches!(
            super::strike::decode(picture.clone(), 2, 1, 256),
            Err(TextError::Font(FontError::BadTable))
        ));
        assert!(matches!(
            super::strike::decode(Vec::new(), 2, 2, 256),
            Err(TextError::Font(_))
        ));
    }

    /// The pixels of `content` set as a title in the face `bytes` hold.
    fn title(bytes: &[u8], face: Option<u32>, content: &str) -> Vec<u8> {
        use cutlass_model::{MaterialText, Micros, Segment, TimeRange};
        use cutlass_picture::{Picture, PictureSource, Sample};

        let mut text = MaterialText::new("text-material");
        text.content = content.to_owned();
        text.font_size = 64.0;
        text.text_color = "#FFFFFF".to_owned();
        let clip = Segment::new("segment", "text-material", TimeRange::new(1_000, 5_000));
        let source = crate::TextSource::open(bytes, face, &text, &clip).expect("title");
        match source.sample(Micros(1_000)) {
            Sample::Ready(Picture::Cpu(picture)) => picture.rgba().to_vec(),
            _ => panic!("no picture"),
        }
    }

    fn count(rgba: &[u8], wanted: impl Fn([u8; 4]) -> bool) -> usize {
        rgba.as_chunks::<4>()
            .0
            .iter()
            .filter(|pixel| wanted(**pixel))
            .count()
    }

    #[test]
    fn a_glyph_that_cannot_be_painted_is_blank_and_the_title_still_draws() {
        let font = read(SBIX).expect("sbix");
        let mut atlas = Atlas::new();
        // The four hostile pictures, each an error when painted alone.
        for glyph in [9, 10, 11, 13] {
            assert!(atlas.glyph(&font, glyph).is_err());
            let image = atlas.drawn(&font, glyph).expect("blank");
            assert_eq!((image.width, image.height), (0, 0), "glyph {glyph}");
        }
        // Each is settled once, and a good glyph is painted as before.
        assert_eq!(atlas.paints(), 4);
        let _again = atlas.drawn(&font, 9).expect("kept");
        assert_eq!(atlas.paints(), 4);
        let picture = atlas.drawn(&font, 2).expect("picture");
        assert_eq!(picture.kind, super::GlyphKind::Color);
        // A glyph the face does not have is still an error, and so is a
        // face with nothing to draw from.
        assert!(matches!(
            atlas.drawn(&font, 9_999),
            Err(TextError::Font(FontError::UnknownGlyph))
        ));
        let empty = read(include_bytes!("../../font/fixtures/no-outline.otf")).expect("cff2");
        assert!(matches!(
            Atlas::new().drawn(&empty, 0),
            Err(TextError::MissingStrike)
        ));
        // A title with a hostile picture in it shows its other glyphs: the
        // letter in the text's white and the good picture's red.
        let alone = title(SBIX, None, "A\u{1F600}");
        let beside = title(SBIX, None, "A\u{E009}\u{1F600}\u{E00B}");
        let white = |rgba: &[u8]| count(rgba, |pixel| pixel == [255, 255, 255, 255]);
        let red = |rgba: &[u8]| count(rgba, |pixel| pixel == [255, 0, 0, 255]);
        assert!(white(&beside) > 100 && red(&beside) > 100);
        assert_eq!((white(&beside), red(&beside)), (white(&alone), red(&alone)));
    }

    #[test]
    fn a_title_in_a_color_face_shows_the_glyphs_own_colors() {
        // A strike picture: red, half green, and blue, each many pixels now.
        let strike = title(SBIX, None, "\u{1F600}");
        assert!(count(&strike, |pixel| pixel == [255, 0, 0, 255]) > 100);
        assert!(count(&strike, |pixel| pixel == [0, 0, 255, 255]) > 100);
        assert!(count(&strike, |pixel| pixel == [0, 128, 0, 128]) > 100);
        // The text is white, and none of the picture is.
        assert_eq!(count(&strike, |pixel| pixel == [255, 255, 255, 255]), 0);
        // Layers: red on the left and blue on the right, in equal parts.
        let colr = include_bytes!("../../font/fixtures/colr.ttf");
        let layers = title(colr, None, "\u{E00A}");
        let red = count(&layers, |pixel| pixel == [255, 0, 0, 255]);
        let blue = count(&layers, |pixel| pixel == [0, 0, 255, 255]);
        assert!(
            red > 500 && red.abs_diff(blue) < red / 8,
            "{red} red, {blue} blue"
        );
        // A letter beside a picture keeps the text's color.
        let mixed = title(SBIX, None, "A\u{1F600}");
        assert!(count(&mixed, |pixel| pixel == [255, 255, 255, 255]) > 100);
        assert!(count(&mixed, |pixel| pixel == [255, 0, 0, 255]) > 100);
        // The system's emoji face, where it is installed: mostly yellow.
        if let Ok(bytes) = std::fs::read(SYSTEM_EMOJI) {
            let face = title(&bytes, Some(0), "\u{1F600}");
            let yellow = count(&face, |[red, green, blue, alpha]| {
                alpha == 255 && red > 200 && green > 150 && blue < 120
            });
            assert!(yellow > 500, "{yellow} yellow pixels");
        }
    }
}
