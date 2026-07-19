//! Pen-tool paths: cubic-bezier outlines rasterized on the CPU.
//!
//! Arbitrary curved paths have no cheap SDF, so unlike the parametric shapes
//! they realize as bitmaps through `tiny-skia` and ride the compositor's
//! existing RGBA layer path. Paths animate through the clip *transform*
//! (position/scale/rotation/opacity on the GPU quad), so a raster is rebuilt
//! only when the path or its style is edited — [`PathRaster`] memoizes the
//! recent results exactly like `cutlass-text`'s renderer memoizes text runs.
//!
//! Fill rule is **non-zero winding** everywhere (raster and
//! [`path_hit_test`]), matching the pen-tool convention in Photoshop/AE.

use std::collections::HashMap;

use cutlass_core::RgbaImage;
use tiny_skia::{FillRule, LineCap, LineJoin, Paint, PathBuilder, Pixmap, Transform};

use crate::{BezierPath, PathPoint, ShapeStyle};

/// Flattening steps per cubic segment for bounds/hit-test queries. Editor
/// precision, not raster precision (tiny-skia flattens adaptively when
/// drawing): 16 chords keep the error well under a pixel at UI sizes.
const FLATTEN_STEPS: usize = 16;

/// Extra transparent margin around the inked bounds, covering tiny-skia's
/// anti-alias overhang.
const RASTER_PAD: f32 = 2.0;

/// Rasterizes pen paths, memoizing recent results. Owned by the renderer next
/// to the `TextRenderer`; reuse one across frames so unchanged paths cost a
/// memo lookup plus a bitmap copy-out.
pub struct PathRaster {
    memo: HashMap<PathKey, RgbaImage>,
}

impl Default for PathRaster {
    fn default() -> Self {
        Self::new()
    }
}

impl PathRaster {
    pub fn new() -> Self {
        Self {
            memo: HashMap::new(),
        }
    }

    /// Rasterize `path` (shape-local pixels) scaled by `scale` into a
    /// straight-alpha bitmap sized to the inked bounds plus stroke overhang.
    /// The bounds center lands on the image center — the point the renderer
    /// places at the layer center. Returns a zero-area image for paths with
    /// nothing to draw (fewer than 2 points, or no fill and no stroke).
    pub fn rasterize(&mut self, path: &BezierPath, style: &ShapeStyle, scale: f32) -> RgbaImage {
        let key = PathKey::new(path, style, scale);
        if let Some(hit) = self.memo.get(&key) {
            return hit.clone();
        }
        let image = rasterize_uncached(path, style, scale);
        // Bounded memory: wholesale clear at the cap (a frame loop keeps a
        // handful of live keys; see cutlass-text for the same reasoning).
        const MEMO_CAP: usize = 32;
        if self.memo.len() >= MEMO_CAP {
            self.memo.clear();
        }
        self.memo.insert(key, image.clone());
        image
    }
}

fn rasterize_uncached(path: &BezierPath, style: &ShapeStyle, scale: f32) -> RgbaImage {
    let scale = if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    };
    let stroke_w = style.stroke.map_or(0.0, |s| (s.width * scale).max(0.0));
    let Some(skia_path) = build_skia_path(path, scale) else {
        return RgbaImage::transparent(0, 0);
    };
    if style.fill.is_none() && style.stroke.is_none() {
        return RgbaImage::transparent(0, 0);
    }

    let bounds = skia_path.bounds();
    let pad = stroke_w * 0.5 + RASTER_PAD;
    let w = (bounds.width() + 2.0 * pad).ceil().max(1.0) as u32;
    let h = (bounds.height() + 2.0 * pad).ceil().max(1.0) as u32;
    let Some(mut pixmap) = Pixmap::new(w, h) else {
        return RgbaImage::transparent(0, 0);
    };
    // Center the inked bounds in the padded image.
    let translate = Transform::from_translate(
        pad - bounds.x() + (w as f32 - bounds.width() - 2.0 * pad) * 0.5,
        pad - bounds.y() + (h as f32 - bounds.height() - 2.0 * pad) * 0.5,
    );

    let mut paint = Paint {
        anti_alias: true,
        ..Paint::default()
    };

    // Fill only closes visually on closed paths; an open pen path is
    // stroke-only even if a fill color is set (matches AE/Photoshop).
    if let (Some(fill), true) = (style.fill, path.closed) {
        paint.set_color_rgba8(fill[0], fill[1], fill[2], fill[3]);
        pixmap.fill_path(&skia_path, &paint, FillRule::Winding, translate, None);
    }
    if let Some(stroke) = style.stroke
        && stroke_w > 0.0
    {
        paint.set_color_rgba8(
            stroke.rgba[0],
            stroke.rgba[1],
            stroke.rgba[2],
            stroke.rgba[3],
        );
        let sk_stroke = tiny_skia::Stroke {
            width: stroke_w,
            line_cap: LineCap::Round,
            line_join: LineJoin::Round,
            ..tiny_skia::Stroke::default()
        };
        pixmap.stroke_path(&skia_path, &paint, &sk_stroke, translate, None);
    }

    // tiny-skia stores premultiplied alpha; the compositor's RGBA path takes
    // straight alpha (it premultiplies on upload).
    let mut out = vec![0u8; (w as usize) * (h as usize) * 4];
    for (i, px) in pixmap.pixels().iter().enumerate() {
        let c = px.demultiply();
        let o = i * 4;
        out[o] = c.red();
        out[o + 1] = c.green();
        out[o + 2] = c.blue();
        out[o + 3] = c.alpha();
    }
    RgbaImage::new(w, h, out)
}

/// Build the tiny-skia path for `path` scaled by `scale`. `None` when the
/// path has no drawable segment or a coordinate is non-finite.
fn build_skia_path(path: &BezierPath, scale: f32) -> Option<tiny_skia::Path> {
    if !path.is_drawable() {
        return None;
    }
    let s = |v: [f32; 2]| (v[0] * scale, v[1] * scale);
    let mut pb = PathBuilder::new();
    let first = path.points[0];
    let (x0, y0) = s(first.anchor);
    pb.move_to(x0, y0);
    for pair in path.points.windows(2) {
        push_cubic(&mut pb, &pair[0], &pair[1], scale);
    }
    if path.closed {
        let last = path.points[path.points.len() - 1];
        push_cubic(&mut pb, &last, &first, scale);
        pb.close();
    }
    pb.finish()
}

/// Append the cubic segment `a → b` (out-handle of `a`, in-handle of `b`).
/// Two collapsed handles degrade to a straight line, which keeps corner
/// points sharp instead of numerically flat cubics.
fn push_cubic(pb: &mut PathBuilder, a: &PathPoint, b: &PathPoint, scale: f32) {
    let straight = a.handle_out == a.anchor && b.handle_in == b.anchor;
    if straight {
        pb.line_to(b.anchor[0] * scale, b.anchor[1] * scale);
    } else {
        pb.cubic_to(
            a.handle_out[0] * scale,
            a.handle_out[1] * scale,
            b.handle_in[0] * scale,
            b.handle_in[1] * scale,
            b.anchor[0] * scale,
            b.anchor[1] * scale,
        );
    }
}

/// Tight bounds of the drawn curve in shape-local pixels (`(min, max)`
/// corners), or `None` for non-drawable paths. Computed on the flattened
/// curve, so it hugs the ink (control points may lie far outside). The pen
/// tool uses this to normalize a committed path around its center; selection
/// UIs use it for the content box.
pub fn path_bounds(path: &BezierPath) -> Option<([f32; 2], [f32; 2])> {
    let pts = flatten(path)?;
    let mut min = pts[0];
    let mut max = pts[0];
    for p in &pts {
        min = [min[0].min(p[0]), min[1].min(p[1])];
        max = [max[0].max(p[0]), max[1].max(p[1])];
    }
    Some((min, max))
}

/// True when `point` (shape-local pixels) hits the path: inside the fill
/// (non-zero winding; closed paths only) or within half the stroke width of
/// the curve. Editor hit-testing — not a per-frame call.
pub fn path_hit_test(path: &BezierPath, point: [f32; 2], stroke_width: f32) -> bool {
    let Some(pts) = flatten(path) else {
        return false;
    };

    // Distance to the polyline (squared, for the stroke test).
    let tol = (stroke_width * 0.5).max(1.0);
    let tol2 = tol * tol;
    let segment_hit = |a: [f32; 2], b: [f32; 2]| -> bool {
        let e = [b[0] - a[0], b[1] - a[1]];
        let w = [point[0] - a[0], point[1] - a[1]];
        let len2 = e[0] * e[0] + e[1] * e[1];
        let t = if len2 > 0.0 {
            ((w[0] * e[0] + w[1] * e[1]) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let d = [w[0] - e[0] * t, w[1] - e[1] * t];
        d[0] * d[0] + d[1] * d[1] <= tol2
    };
    for pair in pts.windows(2) {
        if segment_hit(pair[0], pair[1]) {
            return true;
        }
    }
    if path.closed && segment_hit(pts[pts.len() - 1], pts[0]) {
        return true;
    }

    // Fill: non-zero winding number over the closed flattened loop.
    if path.closed {
        let mut winding = 0i32;
        let n = pts.len();
        for i in 0..n {
            let a = pts[i];
            let b = pts[(i + 1) % n];
            if a[1] <= point[1] {
                if b[1] > point[1] && cross(a, b, point) > 0.0 {
                    winding += 1;
                }
            } else if b[1] <= point[1] && cross(a, b, point) < 0.0 {
                winding -= 1;
            }
        }
        if winding != 0 {
            return true;
        }
    }
    false
}

/// Cross product `(b - a) × (p - a)`: positive when `p` is left of `a→b`.
fn cross(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

/// Flatten the path's cubics into a polyline (shape-local pixels), `None`
/// for non-drawable paths.
fn flatten(path: &BezierPath) -> Option<Vec<[f32; 2]>> {
    if !path.is_drawable() {
        return None;
    }
    let mut out = Vec::with_capacity(path.points.len() * FLATTEN_STEPS);
    out.push(path.points[0].anchor);
    for pair in path.points.windows(2) {
        flatten_segment(&pair[0], &pair[1], &mut out);
    }
    if path.closed {
        let last = path.points[path.points.len() - 1];
        let first = path.points[0];
        flatten_segment(&last, &first, &mut out);
        out.pop(); // the loop's last point duplicates the first
    }
    Some(out)
}

/// Append the flattened chords of segment `a → b` (excluding `a`'s anchor).
fn flatten_segment(a: &PathPoint, b: &PathPoint, out: &mut Vec<[f32; 2]>) {
    if a.handle_out == a.anchor && b.handle_in == b.anchor {
        out.push(b.anchor);
        return;
    }
    let (p0, p1, p2, p3) = (a.anchor, a.handle_out, b.handle_in, b.anchor);
    for k in 1..=FLATTEN_STEPS {
        let t = k as f32 / FLATTEN_STEPS as f32;
        let u = 1.0 - t;
        let c0 = u * u * u;
        let c1 = 3.0 * u * u * t;
        let c2 = 3.0 * u * t * t;
        let c3 = t * t * t;
        out.push([
            c0 * p0[0] + c1 * p1[0] + c2 * p2[0] + c3 * p3[0],
            c0 * p0[1] + c1 * p1[1] + c2 * p2[1] + c3 * p3[1],
        ]);
    }
}

/// Memo identity of a raster request: every coordinate/color/width by bit
/// pattern, so any edit is a new key (never a wrong hit).
#[derive(PartialEq, Eq, Hash)]
struct PathKey {
    points: Vec<[u32; 6]>,
    closed: bool,
    fill: Option<[u8; 4]>,
    stroke: Option<([u8; 4], u32)>,
    scale: u32,
}

impl PathKey {
    fn new(path: &BezierPath, style: &ShapeStyle, scale: f32) -> Self {
        Self {
            points: path
                .points
                .iter()
                .map(|p| {
                    [
                        p.anchor[0].to_bits(),
                        p.anchor[1].to_bits(),
                        p.handle_in[0].to_bits(),
                        p.handle_in[1].to_bits(),
                        p.handle_out[0].to_bits(),
                        p.handle_out[1].to_bits(),
                    ]
                })
                .collect(),
            closed: path.closed,
            fill: style.fill,
            stroke: style.stroke.map(|s| (s.rgba, s.width.to_bits())),
            scale: scale.to_bits(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Stroke;

    /// A closed unit-ish diamond around the origin.
    fn diamond(r: f32) -> BezierPath {
        BezierPath {
            points: vec![
                PathPoint::corner([0.0, -r]),
                PathPoint::corner([r, 0.0]),
                PathPoint::corner([0.0, r]),
                PathPoint::corner([-r, 0.0]),
            ],
            closed: true,
        }
    }

    /// An S-curve open path with real handles.
    fn s_curve() -> BezierPath {
        BezierPath {
            points: vec![
                PathPoint {
                    anchor: [0.0, 0.0],
                    handle_in: [0.0, 0.0],
                    handle_out: [40.0, 0.0],
                },
                PathPoint {
                    anchor: [60.0, 40.0],
                    handle_in: [20.0, 40.0],
                    handle_out: [60.0, 40.0],
                },
            ],
            closed: false,
        }
    }

    fn filled(rgba: [u8; 4]) -> ShapeStyle {
        ShapeStyle {
            fill: Some(rgba),
            stroke: None,
        }
    }

    #[test]
    fn closed_path_fills() {
        let mut r = PathRaster::new();
        let img = r.rasterize(&diamond(20.0), &filled([0, 255, 0, 255]), 1.0);
        assert!(img.is_well_formed());
        let center = img.pixel(img.width / 2, img.height / 2);
        assert_eq!(center, [0, 255, 0, 255]);
        assert_eq!(img.pixel(0, 0)[3], 0, "outside the diamond is clear");
    }

    #[test]
    fn open_path_ignores_fill_and_strokes() {
        let mut r = PathRaster::new();
        let style = ShapeStyle {
            fill: Some([255, 0, 0, 255]),
            stroke: Some(Stroke {
                rgba: [255, 255, 255, 255],
                width: 4.0,
            }),
        };
        let img = r.rasterize(&s_curve(), &style, 1.0);
        assert!(img.width > 0 && img.height > 0);
        // Some stroke ink exists, and no pixel is fill-red (open ⇒ no fill).
        let mut any_ink = false;
        for px in img.pixels.chunks_exact(4) {
            if px[3] > 0 {
                any_ink = true;
                assert!(
                    !(px[0] > 200 && px[1] < 60 && px[2] < 60),
                    "open path must not fill: {px:?}"
                );
            }
        }
        assert!(any_ink, "stroke drew nothing");
    }

    #[test]
    fn non_drawable_paths_are_empty() {
        let mut r = PathRaster::new();
        let dot = BezierPath {
            points: vec![PathPoint::corner([5.0, 5.0])],
            closed: false,
        };
        let img = r.rasterize(&dot, &filled([255, 255, 255, 255]), 1.0);
        assert_eq!((img.width, img.height), (0, 0));
    }

    #[test]
    fn scale_scales_the_bitmap() {
        let mut r = PathRaster::new();
        let s1 = r.rasterize(&diamond(20.0), &filled([255, 255, 255, 255]), 1.0);
        let s2 = r.rasterize(&diamond(20.0), &filled([255, 255, 255, 255]), 2.0);
        assert!(
            s2.width > (s1.width as f32 * 1.8) as u32,
            "2x scale should roughly double the raster: {} vs {}",
            s1.width,
            s2.width
        );
    }

    #[test]
    fn memo_hits_repeat_requests() {
        let mut r = PathRaster::new();
        let a = r.rasterize(&diamond(10.0), &filled([1, 2, 3, 255]), 1.0);
        let b = r.rasterize(&diamond(10.0), &filled([1, 2, 3, 255]), 1.0);
        assert_eq!(a, b);
        assert_eq!(r.memo.len(), 1);
        let _ = r.rasterize(&diamond(11.0), &filled([1, 2, 3, 255]), 1.0);
        assert_eq!(r.memo.len(), 2, "different geometry is a new entry");
    }

    #[test]
    fn bounds_hug_the_curve_not_the_handles() {
        // Extreme out-handle: the curve bulges far less than the handle.
        let path = BezierPath {
            points: vec![
                PathPoint {
                    anchor: [0.0, 0.0],
                    handle_in: [0.0, 0.0],
                    handle_out: [0.0, -100.0],
                },
                PathPoint {
                    anchor: [40.0, 0.0],
                    handle_in: [40.0, -100.0],
                    handle_out: [40.0, 0.0],
                },
            ],
            closed: false,
        };
        let (min, max) = path_bounds(&path).unwrap();
        // Cubic with both handles at -100 peaks at 3/4 of that: -75.
        assert!(min[1] > -80.0 && min[1] < -70.0, "min y {}", min[1]);
        assert_eq!(max[1], 0.0);
        assert_eq!((min[0], max[0]), (0.0, 40.0));
    }

    #[test]
    fn hit_test_fill_stroke_and_miss() {
        let d = diamond(20.0);
        assert!(path_hit_test(&d, [0.0, 0.0], 0.0), "center is filled");
        assert!(!path_hit_test(&d, [30.0, 30.0], 0.0), "far corner misses");
        // On the edge midpoint with a fat stroke: hit even outside the fill.
        assert!(path_hit_test(&d, [11.0, -11.0], 6.0));
        // Open path: only the stroke band hits.
        let s = s_curve();
        assert!(!path_hit_test(&s, [30.0, 0.0], 0.0));
        assert!(path_hit_test(&s, [0.0, 0.0], 4.0), "on the start anchor");
    }

    #[test]
    fn winding_fill_covers_self_overlap() {
        // A figure-eight-ish loop: winding keeps both lobes filled where
        // even-odd would punch a hole. Just assert the crossing region hits.
        let bow = BezierPath {
            points: vec![
                PathPoint::corner([-20.0, -10.0]),
                PathPoint::corner([20.0, 10.0]),
                PathPoint::corner([20.0, -10.0]),
                PathPoint::corner([-20.0, 10.0]),
            ],
            closed: true,
        };
        assert!(path_hit_test(&bow, [0.0, 0.0], 2.0));
    }
}
