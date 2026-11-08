//! Cutlass's own stickers, drawn from signed distance shapes.
//!
//! A shape is a distance in a square from -1 to 1, y down, negative inside.
//! The sticker is the shape in its color, a white edge [`EDGE`] wide round
//! it, and a soft shadow under both, each edge smoothed over one pixel.

/// How far the white edge reaches past the shape.
const EDGE: f32 = 0.07;
/// Where the shadow falls, down and to the right.
const SHADOW: (f32, f32) = (0.02, 0.045);
/// How dark the shadow is under the middle of the sticker.
const SHADOW_ALPHA: f32 = 0.32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Shape {
    Heart,
    Star,
    Sparkle,
    Arrow,
    Burst,
    Check,
    Smile,
    Bolt,
}

type Point = (f32, f32);

fn rgb(hex: u32) -> [f32; 3] {
    let channel = |shift: u32| ((hex >> shift) & 0xFF) as f32 / 255.0;
    [channel(16), channel(8), channel(0)]
}

impl Shape {
    /// The distance to the shape's outline.
    fn distance(self, p: Point) -> f32 {
        match self {
            Self::Heart => heart(p),
            Self::Star => star(p, 5, 0.86, 0.42, 0.05),
            Self::Sparkle => {
                let big = star((p.0 + 0.12, p.1 - 0.1), 4, 0.78, 0.2, 0.03);
                let small = star((p.0 - 0.58, p.1 + 0.55), 4, 0.3, 0.08, 0.02);
                big.min(small)
            }
            Self::Arrow => {
                let shaft = capsule(p, (-0.78, 0.0), (0.15, 0.0), 0.17);
                let head = polygon(p, &[(0.0, -0.52), (0.84, 0.0), (0.0, 0.52)]) - 0.05;
                shaft.min(head)
            }
            Self::Burst => star(p, 14, 0.88, 0.68, 0.02),
            Self::Check => {
                let short = capsule(p, (-0.6, 0.02), (-0.18, 0.44), 0.16);
                let long = capsule(p, (-0.18, 0.44), (0.62, -0.46), 0.16);
                short.min(long)
            }
            Self::Smile => length(p) - 0.82,
            Self::Bolt => {
                polygon(
                    p,
                    &[
                        (0.12, -0.88),
                        (-0.48, 0.12),
                        (-0.06, 0.12),
                        (-0.2, 0.88),
                        (0.5, -0.2),
                        (0.08, -0.2),
                        (0.34, -0.88),
                    ],
                ) - 0.03
            }
        }
    }

    /// The color at `p` inside the shape, details drawn over the fill with
    /// `smooth` as the width of one pixel.
    fn color(self, p: Point, smooth: f32) -> [f32; 3] {
        let fill = match self {
            Self::Heart => rgb(0xFF_4D_6D),
            Self::Star => rgb(0xFF_C8_3D),
            Self::Sparkle => rgb(0x5C_E1_E6),
            Self::Arrow => rgb(0xFF_7A_1A),
            Self::Burst => rgb(0xFF_3E_A5),
            Self::Check => rgb(0x2B_D9_7C),
            Self::Smile => rgb(0xFF_D2_3F),
            Self::Bolt => rgb(0x6C_8C_FF),
        };
        // A little lighter at the top, as if lit from above.
        let light = 1.06 - 0.12 * (p.1 + 1.0) / 2.0;
        let mut color = fill.map(|channel| (channel * light).min(1.0));
        if self == Self::Smile {
            let eyes = circle(p, (-0.28, -0.2), 0.11).min(circle(p, (0.28, -0.2), 0.11));
            let face = eyes.min(arc(p, 0.46, 0.4, 2.75, 0.07));
            let ink = coverage(face, smooth);
            let dark = rgb(0x3A_2A_1A);
            for (channel, dark) in color.iter_mut().zip(dark) {
                *channel += (dark - *channel) * ink;
            }
        }
        color
    }
}

/// The sticker drawn `side` pixels square, as straight RGBA.
#[must_use]
pub(crate) fn rgba(shape: Shape, side: u32) -> Vec<u8> {
    let side_f = side.max(1) as f32;
    let pixel = 2.0 / side_f;
    let mut out = Vec::with_capacity((side as usize) * (side as usize) * 4);
    for y in 0..side {
        for x in 0..side {
            let p = (
                (x as f32 + 0.5) * pixel - 1.0,
                (y as f32 + 0.5) * pixel - 1.0,
            );
            // Draw a touch smaller so the edge and the shadow fit.
            let q = (p.0 / 0.86, p.1 / 0.86);
            let distance = shape.distance(q) * 0.86;
            let shadow = {
                let under =
                    shape.distance(((p.0 - SHADOW.0) / 0.86, (p.1 - SHADOW.1) / 0.86)) * 0.86;
                SHADOW_ALPHA * (1.0 - smoothstep(-0.03, EDGE + 0.05, under))
            };
            let white = coverage(distance - EDGE, pixel);
            let inside = coverage(distance, pixel);
            let color = shape.color(q, pixel / 0.86);
            // Premultiplied, back to front: the black shadow, the white
            // edge over it, then the fill.
            let mut premultiplied = [white; 3];
            let mut alpha = shadow * (1.0 - white) + white;
            for (channel, value) in color.iter().enumerate() {
                premultiplied[channel] = premultiplied[channel] * (1.0 - inside) + value * inside;
            }
            alpha = alpha * (1.0 - inside) + inside;
            let straight = |value: f32| {
                if alpha <= 0.0 {
                    0
                } else {
                    to_byte(value / alpha)
                }
            };
            out.extend_from_slice(&[
                straight(premultiplied[0]),
                straight(premultiplied[1]),
                straight(premultiplied[2]),
                to_byte(alpha),
            ]);
        }
    }
    out
}

fn to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// How much of a pixel `smooth` wide is inside at `distance`.
fn coverage(distance: f32, smooth: f32) -> f32 {
    (0.5 - distance / smooth.max(f32::EPSILON)).clamp(0.0, 1.0)
}

fn smoothstep(from: f32, to: f32, value: f32) -> f32 {
    let t = ((value - from) / (to - from)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn length(p: Point) -> f32 {
    p.0.hypot(p.1)
}

fn dot(a: Point, b: Point) -> f32 {
    a.0 * b.0 + a.1 * b.1
}

fn circle(p: Point, center: Point, radius: f32) -> f32 {
    length((p.0 - center.0, p.1 - center.1)) - radius
}

/// A thick line from `a` to `b` with round ends.
fn capsule(p: Point, a: Point, b: Point, radius: f32) -> f32 {
    let pa = (p.0 - a.0, p.1 - a.1);
    let ba = (b.0 - a.0, b.1 - a.1);
    let h = (dot(pa, ba) / dot(ba, ba)).clamp(0.0, 1.0);
    length((pa.0 - ba.0 * h, pa.1 - ba.1 * h)) - radius
}

/// A thick arc of `radius` about the middle, from angle `from` to `to`
/// (radians, y down), with round ends.
fn arc(p: Point, radius: f32, from: f32, to: f32, thick: f32) -> f32 {
    let angle = p.1.atan2(p.0);
    if (from..=to).contains(&angle) {
        return (length(p) - radius).abs() - thick;
    }
    let end = |angle: f32| (radius * angle.cos(), radius * angle.sin());
    circle(p, end(from), thick).min(circle(p, end(to), thick))
}

/// A closed polygon's signed distance: negative inside.
fn polygon(p: Point, points: &[Point]) -> f32 {
    let Some(&first) = points.first() else {
        return f32::MAX;
    };
    let mut distance = dot(
        (p.0 - first.0, p.1 - first.1),
        (p.0 - first.0, p.1 - first.1),
    );
    let mut sign = 1.0_f32;
    let mut j = points.len() - 1;
    for i in 0..points.len() {
        let (vi, vj) = (points[i], points[j]);
        let e = (vj.0 - vi.0, vj.1 - vi.1);
        let w = (p.0 - vi.0, p.1 - vi.1);
        let h = (dot(w, e) / dot(e, e)).clamp(0.0, 1.0);
        let b = (w.0 - e.0 * h, w.1 - e.1 * h);
        distance = distance.min(dot(b, b));
        let c = [p.1 >= vi.1, p.1 < vj.1, e.0 * w.1 > e.1 * w.0];
        if c.iter().all(|c| *c) || c.iter().all(|c| !*c) {
            sign = -sign;
        }
        j = i;
    }
    sign * distance.sqrt()
}

/// A star of `points` tips at `outer`, valleys at `inner`, one tip up,
/// its corners rounded by `round`.
fn star(p: Point, points: u32, outer: f32, inner: f32, round: f32) -> f32 {
    let count = points * 2;
    let corners: Vec<Point> = (0..count)
        .map(|n| {
            let angle =
                std::f32::consts::TAU * n as f32 / count as f32 - std::f32::consts::FRAC_PI_2;
            let radius = if n % 2 == 0 {
                outer - round
            } else {
                inner - round
            };
            (radius * angle.cos(), radius * angle.sin())
        })
        .collect();
    polygon(p, &corners) - round
}

/// A heart about the middle, its point down.
fn heart(p: Point) -> f32 {
    let scale = 1.25;
    let q = ((p.0 / scale).abs(), -p.1 / scale + 0.55);
    let d = if q.0 + q.1 > 1.0 {
        length((q.0 - 0.25, q.1 - 0.75)) - std::f32::consts::SQRT_2 / 4.0
    } else {
        let to_top = dot((q.0, q.1 - 1.0), (q.0, q.1 - 1.0));
        let m = 0.5 * (q.0 + q.1).max(0.0);
        let to_middle = dot((q.0 - m, q.1 - m), (q.0 - m, q.1 - m));
        to_top.min(to_middle).sqrt() * (q.0 - q.1).signum()
    };
    d * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel(rgba: &[u8], side: u32, x: u32, y: u32) -> [u8; 4] {
        let at = ((y * side + x) * 4) as usize;
        [rgba[at], rgba[at + 1], rgba[at + 2], rgba[at + 3]]
    }

    #[test]
    fn each_sticker_is_its_color_in_the_middle_white_at_the_edge_and_clear_outside() {
        let side = 128;
        for shape in [
            Shape::Heart,
            Shape::Star,
            Shape::Sparkle,
            Shape::Arrow,
            Shape::Burst,
            Shape::Check,
            Shape::Smile,
            Shape::Bolt,
        ] {
            let rgba = rgba(shape, side);
            assert_eq!(rgba.len(), (side * side * 4) as usize);
            // The corners are clear.
            assert_eq!(pixel(&rgba, side, 0, 0)[3], 0, "{shape:?}");
            assert_eq!(pixel(&rgba, side, side - 1, 0)[3], 0, "{shape:?}");
            // Somewhere is solid fill that is not white, and somewhere is
            // the solid white edge.
            let pixels: Vec<[u8; 4]> = rgba.as_chunks::<4>().0.to_vec();
            let fill = rgb_bytes(shape);
            assert!(
                pixels.iter().any(|p| p[3] == 255 && near(*p, fill, 24)),
                "{shape:?} has no fill"
            );
            assert!(
                pixels
                    .iter()
                    .any(|p| p[3] == 255 && p[..3] == [255, 255, 255]),
                "{shape:?} has no white edge"
            );
            // And somewhere is the soft shadow, partly clear.
            assert!(
                pixels.iter().any(|p| p[3] > 0 && p[3] < 120 && p[0] < 40),
                "{shape:?} has no shadow"
            );
        }
    }

    fn rgb_bytes(shape: Shape) -> [u8; 3] {
        shape.color((0.0, 0.0), 0.01).map(to_byte)
    }

    fn near(pixel: [u8; 4], color: [u8; 3], within: u8) -> bool {
        pixel[..3]
            .iter()
            .zip(color)
            .all(|(a, b)| a.abs_diff(b) <= within)
    }

    #[test]
    fn shapes_are_inside_in_the_middle_and_outside_at_the_corner() {
        assert!(heart((0.0, 0.0)) < 0.0);
        assert!(heart((0.95, 0.95)) > 0.0);
        assert!(star((0.0, 0.0), 5, 0.8, 0.4, 0.0) < 0.0);
        assert!(
            star((0.0, -0.79), 5, 0.8, 0.4, 0.0) < 0.0,
            "a tip points up"
        );
        assert!(star((0.0, 0.79), 5, 0.8, 0.4, 0.0) > 0.0);
        let square = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)];
        assert!((polygon((0.0, 0.0), &square) + 0.5).abs() < 1e-5);
        assert!((polygon((1.0, 0.0), &square) - 0.5).abs() < 1e-5);
        assert!(capsule((0.0, 0.0), (-1.0, 0.0), (1.0, 0.0), 0.1) < 0.0);
        assert!(arc((0.0, 0.46), 0.46, 0.4, 2.75, 0.05) < 0.0);
        assert!(arc((0.0, -0.46), 0.46, 0.4, 2.75, 0.05) > 0.0);
    }
}
