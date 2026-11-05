//! The CPU rasterizer, the same arithmetic as the WGSL in [`crate::WGSL`].
//!
//! Each pixel holds sixteen samples, four rows of four, at `(i + 0.5) / 4`
//! of the pixel. A sample is inside a fill when the edges crossing its row
//! to its right wind round it (non-zero) or cross an odd number of times
//! (even-odd). An edge crosses a row at `y` when `y` is in `[y0, y1)`, its
//! ends taken in either order. Coverage is the share of samples inside. The
//! paint is read once, at the pixel center. Every draw lands with
//! premultiplied "over" and is rounded to bytes, as an 8-bit GPU target is.

use crate::scene::{
    Bounds, Edge, Fill, FillRule, Gradient, GradientKind, Group, Item, Paint, Scene,
};

/// Samples per pixel edge; a pixel has this squared.
const GRID: u32 = 4;
const SAMPLES: f32 = (GRID * GRID) as f32;

/// The scene as premultiplied RGBA bytes, row by row.
#[must_use]
pub fn rasterize(scene: &Scene) -> Vec<u8> {
    draw(&scene.items, scene.width, scene.height)
}

fn draw(items: &[Item], width: u32, height: u32) -> Vec<u8> {
    let mut canvas = vec![0_u8; width as usize * height as usize * 4];
    for item in items {
        match item {
            Item::Fill(fill) => draw_fill(&mut canvas, width, fill),
            Item::Group(group) => draw_group(&mut canvas, width, height, group),
        }
    }
    canvas
}

fn to_byte(value: f32) -> u8 {
    (value * 255.0).round().clamp(0.0, 255.0) as u8
}

fn over(slot: &mut [u8], source: [f32; 4]) {
    let keep = 1.0 - source[3];
    for (channel, value) in slot.iter_mut().zip(source) {
        *channel = to_byte(value + f32::from(*channel) / 255.0 * keep);
    }
}

fn draw_fill(canvas: &mut [u8], width: u32, fill: &Fill) {
    let stride = width as usize * 4;
    sample_counts(&fill.edges, fill.rule, fill.bounds, |y, x0, counts| {
        for (offset, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let x = x0 + offset as u32;
            let coverage = f32::from(count) / SAMPLES;
            let color = shade(&fill.paint, x as f32 + 0.5, y as f32 + 0.5);
            let at = y as usize * stride + x as usize * 4;
            if let Some(slot) = canvas.get_mut(at..at + 4) {
                over(slot, color.map(|channel| channel * coverage));
            }
        }
    });
}

/// Sample counts, 0 to 16, for every pixel of `edges` over the whole
/// output.
fn coverage_map(edges: &[Edge], width: u32, height: u32) -> Vec<u8> {
    let mut map = vec![0_u8; width as usize * height as usize];
    if let Some(bounds) = Bounds::of(edges, width, height) {
        sample_counts(edges, FillRule::NonZero, bounds, |y, x0, counts| {
            let at = y as usize * width as usize + x0 as usize;
            if let Some(row) = map.get_mut(at..at + counts.len()) {
                row.copy_from_slice(counts);
            }
        });
    }
    map
}

fn draw_group(canvas: &mut [u8], width: u32, height: u32, group: &Group) {
    let child = draw(&group.items, width, height);
    let matte = group
        .matte
        .as_ref()
        .map(|matte| (matte.mode, draw(&matte.items, width, height)));
    let masks: Vec<(Vec<u8>, Option<Vec<u8>>)> = group
        .masks
        .iter()
        .map(|mask| {
            (
                coverage_map(&mask.edges, width, height),
                mask.expansion
                    .as_ref()
                    .map(|expansion| coverage_map(&expansion.edges, width, height)),
            )
        })
        .collect();
    let start = group.mask_start();
    for (pixel, (slot, own)) in canvas
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(child.as_chunks::<4>().0)
        .enumerate()
    {
        let mut factor = group.opacity;
        if !group.masks.is_empty() {
            let mut value = start;
            for (mask, (map, grown)) in group.masks.iter().zip(&masks) {
                let mut coverage = f32::from(map[pixel]) / SAMPLES;
                if let (Some(expansion), Some(grown)) = (&mask.expansion, grown) {
                    let extra = f32::from(grown[pixel]) / SAMPLES;
                    coverage = if expansion.grow {
                        coverage + extra - coverage * extra
                    } else {
                        coverage * (1.0 - extra)
                    };
                }
                let shown = if mask.inverted {
                    1.0 - coverage
                } else {
                    coverage
                };
                value = mask.mode.combine(value, shown * mask.opacity);
            }
            factor *= value;
        }
        if let Some((mode, matte)) = &matte {
            let at = pixel * 4;
            let texel = [0, 1, 2, 3].map(|channel| f32::from(matte[at + channel]) / 255.0);
            factor *= mode.factor(texel);
        }
        if own[3] == 0 && own[..3].iter().all(|channel| *channel == 0) {
            continue;
        }
        let source = [0, 1, 2, 3].map(|channel| f32::from(own[channel]) / 255.0 * factor);
        over(slot, source);
    }
}

/// Visit each row of `bounds` with the sample count of each of its pixels.
fn sample_counts(
    edges: &[Edge],
    rule: FillRule,
    bounds: Bounds,
    mut visit: impl FnMut(u32, u32, &[u8]),
) {
    let Bounds { x0, y0, x1, y1 } = bounds;
    let mut counts = vec![0_u8; (x1 - x0) as usize];
    // Edges by the first pixel row they reach; a row keeps those it crosses.
    let rows = (y1 - y0) as usize;
    let mut starts: Vec<Vec<usize>> = vec![Vec::new(); rows];
    for (index, edge) in edges.iter().enumerate() {
        let top = edge.y0.min(edge.y1);
        let bottom = edge.y0.max(edge.y1);
        if !(bottom >= y0 as f32 && top < y1 as f32) {
            continue;
        }
        let row = (top.floor().max(y0 as f32) as u32).min(y1 - 1) - y0;
        starts[row as usize].push(index);
    }
    let mut active: Vec<usize> = Vec::new();
    let mut crossings: Vec<(f32, i32)> = Vec::new();
    let low = f64::from(x0 * GRID);
    let high = f64::from(x1 * GRID);
    for (row, starting) in starts.iter().enumerate() {
        let y = y0 + row as u32;
        active.extend_from_slice(starting);
        active.retain(|&index| edges[index].y0.max(edges[index].y1) > y as f32);
        counts.fill(0);
        for sub in 0..GRID {
            let ys = y as f32 + (sub as f32 + 0.5) * 0.25;
            crossings.clear();
            for &index in &active {
                let edge = &edges[index];
                if (edge.y0 <= ys && ys < edge.y1) || (edge.y1 <= ys && ys < edge.y0) {
                    let x = edge.x0 + (ys - edge.y0) * (edge.x1 - edge.x0) / (edge.y1 - edge.y0);
                    let dir = if edge.y1 > edge.y0 { 1 } else { -1 };
                    crossings.push((x, dir));
                }
            }
            if crossings.is_empty() {
                continue;
            }
            crossings.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut winding: i32 = crossings.iter().map(|crossing| crossing.1).sum();
            // Sample `n` sits at x = (n + 0.5) / 4; it is right of a
            // crossing at `c` when n >= 4c - 0.5.
            let index_of = |x: f32| (4.0 * f64::from(x) - 0.5).ceil().clamp(low, high);
            let mut left = low;
            for at in 0..=crossings.len() {
                let right = crossings
                    .get(at)
                    .map_or(high, |crossing| index_of(crossing.0));
                let inside = match rule {
                    FillRule::NonZero => winding != 0,
                    FillRule::EvenOdd => winding % 2 != 0,
                };
                if inside && right > left {
                    for sample in left as u32..right as u32 {
                        counts[(sample / GRID - x0) as usize] += 1;
                    }
                }
                if let Some(crossing) = crossings.get(at) {
                    winding -= crossing.1;
                    left = left.max(right);
                }
            }
        }
        visit(y, x0, &counts);
    }
}

/// The premultiplied color of `paint` at the output point `(x, y)`.
#[must_use]
pub fn shade(paint: &Paint, x: f32, y: f32) -> [f32; 4] {
    let [r, g, b, a] = match paint {
        Paint::Solid(color) => *color,
        Paint::Gradient(gradient) => {
            let [r, g, b, a] = ramp(gradient, gradient_t(gradient, x, y));
            [r, g, b, a * gradient.opacity]
        }
    };
    [r * a, g * a, b * a, a]
}

/// Where the point `(x, y)` falls along the gradient, 0 to 1.
fn gradient_t(gradient: &Gradient, x: f32, y: f32) -> f32 {
    let [a, b, c, d, e, f] = gradient.to_gradient;
    let gx = a * x + b * y + c;
    let gy = d * x + e * y + f;
    let t = match gradient.kind {
        GradientKind::Linear => {
            let dx = gradient.end[0] - gradient.start[0];
            let dy = gradient.end[1] - gradient.start[1];
            let length = dx * dx + dy * dy;
            if length > 0.0 {
                ((gx - gradient.start[0]) * dx + (gy - gradient.start[1]) * dy) / length
            } else {
                1.0
            }
        }
        GradientKind::Radial => {
            let r = gradient.radius;
            let cx = gradient.start[0] - gradient.focal[0];
            let cy = gradient.start[1] - gradient.focal[1];
            let px = gx - gradient.focal[0];
            let py = gy - gradient.focal[1];
            let qa = cx * cx + cy * cy - r * r;
            let qb = px * cx + py * cy;
            let qc = px * px + py * py;
            if r <= 1e-6 || qa >= -1e-6 {
                1.0
            } else {
                (qb - (qb * qb - qa * qc).max(0.0).sqrt()) / qa
            }
        }
    };
    if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

fn ramp(gradient: &Gradient, t: f32) -> [f32; 4] {
    let stops = &gradient.stops;
    let Some(first) = stops.first() else {
        return [0.0; 4];
    };
    if t <= first.offset {
        return first.color;
    }
    for pair in stops.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if t <= b.offset {
            let span = b.offset - a.offset;
            let k = if span > 0.0 {
                (t - a.offset) / span
            } else {
                1.0
            };
            return [0, 1, 2, 3]
                .map(|channel| a.color[channel] + (b.color[channel] - a.color[channel]) * k);
        }
    }
    stops.last().map_or([0.0; 4], |last| last.color)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{Mask, MaskMode, Matte, MatteMode, Stop};

    fn square(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<Edge> {
        vec![
            Edge { x0: x1, y0, x1, y1 },
            Edge {
                x0,
                y0: y1,
                x1: x0,
                y1: y0,
            },
        ]
    }

    fn fill(edges: Vec<Edge>, color: [f32; 4], size: u32) -> Item {
        let bounds = Bounds::of(&edges, size, size).unwrap();
        Item::Fill(Fill {
            edges,
            rule: FillRule::NonZero,
            paint: Paint::Solid(color),
            bounds,
        })
    }

    fn pixel(bytes: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
        let at = (y * width + x) as usize * 4;
        [bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]
    }

    #[test]
    fn a_pixel_aligned_square_is_solid_inside_and_clear_outside() {
        let scene = Scene {
            width: 8,
            height: 8,
            items: vec![fill(square(2.0, 2.0, 6.0, 6.0), [1.0, 0.0, 0.0, 1.0], 8)],
        };
        let bytes = rasterize(&scene);
        assert_eq!(pixel(&bytes, 8, 2, 2), [255, 0, 0, 255]);
        assert_eq!(pixel(&bytes, 8, 5, 5), [255, 0, 0, 255]);
        assert_eq!(pixel(&bytes, 8, 6, 5), [0, 0, 0, 0]);
        assert_eq!(pixel(&bytes, 8, 1, 2), [0, 0, 0, 0]);
    }

    #[test]
    fn a_half_pixel_edge_covers_half_the_samples() {
        let scene = Scene {
            width: 4,
            height: 1,
            items: vec![fill(square(0.0, 0.0, 1.5, 1.0), [1.0, 1.0, 1.0, 1.0], 4)],
        };
        let bytes = rasterize(&scene);
        assert_eq!(pixel(&bytes, 4, 0, 0), [255; 4]);
        assert_eq!(pixel(&bytes, 4, 1, 0), [128; 4]);
        assert_eq!(pixel(&bytes, 4, 2, 0), [0; 4]);
    }

    #[test]
    fn even_odd_leaves_the_inner_square_open_and_non_zero_fills_it() {
        let mut edges = square(0.0, 0.0, 8.0, 8.0);
        edges.extend(square(2.0, 2.0, 6.0, 6.0));
        let bounds = Bounds::of(&edges, 8, 8).unwrap();
        let mut scene = Scene {
            width: 8,
            height: 8,
            items: vec![Item::Fill(Fill {
                edges,
                rule: FillRule::EvenOdd,
                paint: Paint::Solid([0.0, 0.0, 1.0, 1.0]),
                bounds,
            })],
        };
        let even_odd = rasterize(&scene);
        assert_eq!(pixel(&even_odd, 8, 4, 4), [0; 4]);
        assert_eq!(pixel(&even_odd, 8, 0, 0), [0, 0, 255, 255]);
        if let Item::Fill(fill) = &mut scene.items[0] {
            fill.rule = FillRule::NonZero;
        }
        let non_zero = rasterize(&scene);
        assert_eq!(pixel(&non_zero, 8, 4, 4), [0, 0, 255, 255]);
    }

    #[test]
    fn a_linear_gradient_runs_from_its_first_stop_to_its_last() {
        let gradient = Gradient {
            kind: GradientKind::Linear,
            to_gradient: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            start: [0.0, 0.0],
            end: [10.0, 0.0],
            focal: [0.0, 0.0],
            radius: 0.0,
            stops: vec![
                Stop {
                    offset: 0.0,
                    color: [0.0, 0.0, 0.0, 1.0],
                },
                Stop {
                    offset: 1.0,
                    color: [1.0, 1.0, 1.0, 1.0],
                },
            ],
            opacity: 1.0,
        };
        let paint = Paint::Gradient(gradient.clone());
        assert_eq!(shade(&paint, 0.0, 3.0), [0.0, 0.0, 0.0, 1.0]);
        assert!((shade(&paint, 5.0, 3.0)[0] - 0.5).abs() < 1e-6);
        assert_eq!(shade(&paint, 20.0, 3.0), [1.0, 1.0, 1.0, 1.0]);
        let radial = Paint::Gradient(Gradient {
            kind: GradientKind::Radial,
            ..gradient
        });
        let Paint::Gradient(mut radial_gradient) = radial else {
            unreachable!()
        };
        radial_gradient.radius = 10.0;
        let radial = Paint::Gradient(radial_gradient);
        assert!((shade(&radial, 5.0, 0.0)[0] - 0.5).abs() < 1e-6);
        assert!((shade(&radial, 0.0, 5.0)[0] - 0.5).abs() < 1e-6);
        assert_eq!(shade(&radial, 0.0, 0.0)[0], 0.0);
    }

    #[test]
    fn masks_and_mattes_limit_a_group() {
        let size = 8;
        let content = fill(square(0.0, 0.0, 8.0, 8.0), [1.0, 1.0, 1.0, 1.0], size);
        let left = square(0.0, 0.0, 4.0, 8.0);
        let group = |masks: Vec<Mask>, matte: Option<Matte>| Scene {
            width: size,
            height: size,
            items: vec![Item::Group(Group {
                opacity: 1.0,
                masks,
                matte,
                items: vec![content.clone()],
            })],
        };
        let add = Mask {
            mode: MaskMode::Add,
            inverted: false,
            opacity: 1.0,
            edges: left.clone(),
            expansion: None,
        };
        let masked = rasterize(&group(vec![add.clone()], None));
        assert_eq!(pixel(&masked, size, 1, 1), [255; 4]);
        assert_eq!(pixel(&masked, size, 6, 1), [0; 4]);
        let inverted = rasterize(&group(
            vec![Mask {
                inverted: true,
                ..add.clone()
            }],
            None,
        ));
        assert_eq!(pixel(&inverted, size, 1, 1), [0; 4]);
        assert_eq!(pixel(&inverted, size, 6, 1), [255; 4]);
        let subtract = rasterize(&group(
            vec![Mask {
                mode: MaskMode::Subtract,
                ..add.clone()
            }],
            None,
        ));
        assert_eq!(pixel(&subtract, size, 1, 1), [0; 4]);
        assert_eq!(pixel(&subtract, size, 6, 1), [255; 4]);
        let half = rasterize(&group(
            vec![Mask {
                opacity: 0.5,
                ..add.clone()
            }],
            None,
        ));
        assert_eq!(pixel(&half, size, 1, 1), [128; 4]);
        let intersect = rasterize(&group(
            vec![
                add,
                Mask {
                    mode: MaskMode::Intersect,
                    inverted: false,
                    opacity: 1.0,
                    edges: square(0.0, 0.0, 8.0, 4.0),
                    expansion: None,
                },
            ],
            None,
        ));
        assert_eq!(pixel(&intersect, size, 1, 1), [255; 4]);
        assert_eq!(pixel(&intersect, size, 1, 6), [0; 4]);
        assert_eq!(pixel(&intersect, size, 6, 1), [0; 4]);
        for (mode, inside, outside) in [
            (MatteMode::Alpha, 255, 0),
            (MatteMode::AlphaInverted, 0, 255),
            (MatteMode::Luma, 255, 0),
            (MatteMode::LumaInverted, 0, 255),
        ] {
            let matte = Matte {
                mode,
                items: vec![fill(left.clone(), [1.0, 1.0, 1.0, 1.0], size)],
            };
            let drawn = rasterize(&group(Vec::new(), Some(matte)));
            assert_eq!(pixel(&drawn, size, 1, 1)[3], inside, "{mode:?}");
            assert_eq!(pixel(&drawn, size, 6, 1)[3], outside, "{mode:?}");
        }
    }
}
