//! A signed-distance atlas for a font that has outlines, and a bitmap atlas
//! for a font that does not.
//!
//! The route picks the atlas. Each glyph is painted once. A later sample reads
//! the cached texels. A face with neither outlines nor a strike is an error.

use std::collections::HashMap;

use cutlass_font::{BitmapPixels, Contour, Font, Outline, Segment};

use crate::TextError;

/// Atlas pixels across one em.
pub(crate) const ATLAS_EM: f32 = 64.0;
/// Texels of distance stored on each side of the outline.
pub(crate) const SPREAD: f32 = 8.0;
const MAX_CELL: u16 = 256;
const MAX_SEGMENTS: usize = 4_096;
const MAX_DEPTH: u8 = 8;
const FLAT: f32 = 0.35;

/// Which atlas a face uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtlasRoute {
    SignedDistance,
    Bitmap,
}

/// The only choice between the two atlas routes.
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
        let image = match route(font) {
            AtlasRoute::SignedDistance => paint(font, glyph_id)?,
            AtlasRoute::Bitmap => paint_bitmap(font, glyph_id)?,
        };
        self.paints = self.paints.saturating_add(1);
        self.cells.insert(glyph_id, image);
        match self.cells.get(&glyph_id) {
            Some(image) => Ok(image),
            None => Err(TextError::Limit),
        }
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

fn paint(font: &Font<'_>, glyph_id: u16) -> Result<GlyphImage, TextError> {
    let outline = font.outline(glyph_id).map_err(TextError::Font)?;
    let unit = f32::from(font.units_per_em()) / ATLAS_EM;
    if !(unit.is_finite() && unit > 0.0) {
        return Err(TextError::Limit);
    }
    if outline.contours.is_empty() {
        return Ok(blank(unit));
    }
    let bounds = bounds(&outline)?;
    let width = cell_span(bounds.min_x, bounds.max_x, unit)?;
    let height = cell_span(bounds.min_y, bounds.max_y, unit)?;
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
        pixels_per_em: ATLAS_EM,
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
    let ppem = f32::from(bitmap.ppem);
    let unit = f32::from(font.units_per_em()) / ppem;
    if !(unit.is_finite() && unit > 0.0 && ppem.is_finite() && ppem > 0.0) {
        return Err(TextError::Limit);
    }
    let pixels = bitmap
        .width
        .checked_mul(bitmap.height)
        .ok_or(TextError::Limit)?;
    let (kind, sdf, color) = match bitmap.pixels {
        BitmapPixels::Coverage(bytes) => {
            if bytes.len() != usize::try_from(pixels).map_err(|_| TextError::Limit)? {
                return Err(TextError::Limit);
            }
            (GlyphKind::Coverage, bytes, Vec::new())
        }
        BitmapPixels::Color(bytes) => {
            let rgba = pixels.checked_mul(4).ok_or(TextError::Limit)?;
            if bytes.len() != usize::try_from(rgba).map_err(|_| TextError::Limit)? {
                return Err(TextError::Limit);
            }
            (GlyphKind::Color, Vec::new(), bytes)
        }
    };
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

fn raster(width: u32, height: u32, edges: &[Edge]) -> Result<Vec<u8>, TextError> {
    let width_px = usize::try_from(width).map_err(|_| TextError::Limit)?;
    let height_px = usize::try_from(height).map_err(|_| TextError::Limit)?;
    let pixels = width_px.checked_mul(height_px).ok_or(TextError::Limit)?;
    if pixels > usize::from(MAX_CELL) * usize::from(MAX_CELL) {
        return Err(TextError::Limit);
    }
    let mut sdf = vec![0_u8; pixels];
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
            let signed = if inside { -distance } else { distance };
            let offset = row
                .checked_mul(width)
                .and_then(|value| value.checked_add(col))
                .ok_or(TextError::Limit)?;
            let offset = usize::try_from(offset).map_err(|_| TextError::Limit)?;
            let Some(texel) = sdf.get_mut(offset) else {
                return Err(TextError::Limit);
            };
            *texel = encode(signed);
        }
    }
    Ok(sdf)
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
}
