//! Outline, shadow, and background drawn with a title's letters.
//!
//! Each is read from the title in units of its size, so a title drawn
//! larger keeps the same look: `border_width` is the outline's width in
//! ems, `shadow_distance` hundredths of an em, `shadow_smoothing` the blur
//! from 0 to 1 (a quarter of an em at 1), and `background_width` and
//! `background_height` the room around the letters in ems. The angle is in
//! degrees, counterclockwise from the right. `background_round_radius` from
//! 0 to 1 rounds the box's corners up to half its height. A title's
//! `bubble` shapes the box as a [`Bubble`] instead: a speech bubble's tail,
//! a thought's puffs and a burst's spikes reach out past the box, and the
//! picture grows to hold them.
//!
//! A glow is the clip's `bloom` effect: its `strength` from 0 to 1 is how
//! bright it is, and its `range` from 0 to 1 how far it spreads, up to
//! [`GLOW_REACH`] ems.
//!
//! An outline is grown from the glyph distance field, so it reaches at most
//! the field's spread beyond each letter; a wider one is drawn that wide.

use cutlass_model::{BloomParams, Bubble, MaterialText};

use crate::TextError;
use crate::atlas::{ATLAS_EM, SPREAD};

/// What is drawn with the letters, in picture pixels.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Style {
    pub(crate) outline: Option<Outline>,
    pub(crate) shadow: Option<Shadow>,
    pub(crate) glow: Option<Glow>,
    pub(crate) background: Option<Background>,
}

/// How far a full-range glow spreads, in ems.
pub const GLOW_REACH: f64 = 0.3;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Glow {
    /// Premultiplied.
    pub(crate) color: [u8; 4],
    pub(crate) blur: f32,
    pub(crate) strength: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Outline {
    /// Premultiplied.
    pub(crate) color: [u8; 4],
    pub(crate) width: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Shadow {
    /// Premultiplied.
    pub(crate) color: [u8; 4],
    pub(crate) dx: f32,
    pub(crate) dy: f32,
    pub(crate) blur: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Background {
    /// Premultiplied.
    pub(crate) color: [u8; 4],
    pub(crate) pad_x: f32,
    pub(crate) pad_y: f32,
    /// 0 square to 1 fully round ends.
    pub(crate) round: f32,
    /// The shape it takes instead of a box.
    pub(crate) shape: Option<Bubble>,
}

/// A box in picture pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Area {
    pub(crate) left: f32,
    pub(crate) top: f32,
    pub(crate) right: f32,
    pub(crate) bottom: f32,
}

impl Area {
    fn union(self, other: Self) -> Self {
        Self {
            left: self.left.min(other.left),
            top: self.top.min(other.top),
            right: self.right.max(other.right),
            bottom: self.bottom.max(other.bottom),
        }
    }

    fn grow(self, x: f32, y: f32) -> Self {
        Self {
            left: self.left - x,
            top: self.top - y,
            right: self.right + x,
            bottom: self.bottom + y,
        }
    }

    fn shift(self, x: f32, y: f32) -> Self {
        Self {
            left: self.left + x,
            top: self.top + y,
            right: self.right + x,
            bottom: self.bottom + y,
        }
    }
}

/// The room a glyph cell keeps around its letter, in pixels at `font_size`.
pub(crate) fn cell_pad(font_size: f32) -> f32 {
    font_size * SPREAD / ATLAS_EM
}

impl Style {
    /// The style `material` asks for at `font_size` pixels. A color that is
    /// not a hex color is an error; a part with no color, no width, or no
    /// opacity is left out, and so is a glow with no strength.
    pub(crate) fn of(
        material: &MaterialText,
        bloom: Option<&BloomParams>,
        font_size: f32,
    ) -> Result<Self, TextError> {
        let global = unit(material.global_alpha);
        let outline = if material.border_color.trim().is_empty() {
            None
        } else {
            let color = color(&material.border_color, material.border_alpha * global)?;
            Outline::new(color, material.border_width, font_size)
        };
        let shadow = if material.has_shadow {
            let shade = if material.shadow_color.trim().is_empty() {
                "#000000"
            } else {
                &material.shadow_color
            };
            let color = color(shade, material.shadow_alpha * global)?;
            Shadow::new(
                color,
                [
                    material.shadow_distance,
                    material.shadow_angle,
                    material.shadow_smoothing,
                ],
                font_size,
            )
        } else {
            None
        };
        let background =
            if material.background_style == 0 || material.background_color.trim().is_empty() {
                None
            } else {
                let color = color(
                    &material.background_color,
                    material.background_alpha * global,
                )?;
                #[allow(clippy::cast_possible_truncation)]
                let round = unit(material.background_round_radius) as f32;
                (color[3] > 0).then(|| Background {
                    color,
                    pad_x: em(material.background_width, font_size),
                    pad_y: em(material.background_height, font_size),
                    round,
                    shape: Bubble::of(&material.bubble),
                })
            };
        let glow = match bloom {
            Some(bloom) if !bloom.color.trim().is_empty() => {
                let color = color(&bloom.color, global)?;
                #[allow(clippy::cast_possible_truncation)]
                let strength = unit(bloom.strength) as f32;
                (color[3] > 0 && strength > 0.0).then(|| Glow {
                    color,
                    blur: em(unit(bloom.range) * GLOW_REACH, font_size).max(1.0),
                    strength,
                })
            }
            _ => None,
        };
        Ok(Self {
            outline,
            shadow,
            glow,
            background,
        })
    }

    /// The picture's box for letters whose cells cover `cells`, cast with
    /// `shadows`.
    pub(crate) fn bounds(&self, cells: Area, font_size: f32, shadows: &[Shadow]) -> Area {
        let mut area = cells;
        for shadow in shadows {
            let reach = shadow.blur * 2.0 + 1.0;
            area = area.union(cells.shift(shadow.dx, shadow.dy).grow(reach, reach));
        }
        if let Some(glow) = &self.glow {
            let reach = glow.blur * 2.0 + 1.0;
            area = area.union(cells.grow(reach, reach));
        }
        if let Some(background) = &self.background {
            area = area.union(background.extent(background.area(cells, font_size)));
        }
        area
    }
}

impl Outline {
    /// An outline in premultiplied `color`, `width` ems wide at `font_size`
    /// pixels; none when it has no width or no opacity.
    pub(crate) fn new(color: [u8; 4], width: f64, font_size: f32) -> Option<Self> {
        let width = em(width, font_size);
        (width > 0.0 && color[3] > 0).then_some(Self { color, width })
    }

    /// This outline on letters `size` times the title's, at most what their
    /// distance field reaches.
    pub(crate) fn at(&self, size: f32, font_size: f32) -> Option<Self> {
        let width = (self.width * size).min((cell_pad(font_size * size) - 1.0).max(0.0));
        (width > 0.0).then_some(Self {
            color: self.color,
            width,
        })
    }
}

impl Shadow {
    /// A shadow in premultiplied `color`. `shape` is the distance in
    /// hundredths of an em, the angle in degrees, and the blur from 0 to 1;
    /// none when it has no opacity.
    pub(crate) fn new(color: [u8; 4], shape: [f64; 3], font_size: f32) -> Option<Self> {
        let [distance, angle, blur] = shape;
        let distance = em(distance / 100.0, font_size);
        let angle = if angle.is_finite() {
            angle.to_radians()
        } else {
            0.0
        };
        #[allow(clippy::cast_possible_truncation)]
        let (dx, dy) = (
            distance * angle.cos() as f32,
            -distance * angle.sin() as f32,
        );
        let blur = em(unit(blur) * 0.25, font_size);
        (color[3] > 0).then_some(Self {
            color,
            dx,
            dy,
            blur,
        })
    }

    /// This shadow under letters `size` times the title's.
    pub(crate) fn at(&self, size: f32) -> Self {
        Self {
            color: self.color,
            dx: self.dx * size,
            dy: self.dy * size,
            blur: self.blur * size,
        }
    }
}

/// Red, green, blue, and alpha from 0 to 1, times `alpha`, premultiplied.
pub(crate) fn unit_color(rgba: [f64; 4], alpha: f64) -> [u8; 4] {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let byte = |value: f64| (unit(value) * 255.0).round() as u8;
    premultiply([
        byte(rgba[0]),
        byte(rgba[1]),
        byte(rgba[2]),
        byte(rgba[3] * unit(alpha)),
    ])
}

impl Background {
    /// The box behind letters whose cells cover `cells`.
    pub(crate) fn area(&self, cells: Area, font_size: f32) -> Area {
        let pad = cell_pad(font_size);
        cells.grow(self.pad_x - pad, self.pad_y - pad)
    }

    /// All a bubble around the box `body` covers, its tail and spikes too.
    pub(crate) fn extent(&self, body: Area) -> Area {
        let w = body.right - body.left;
        let h = body.bottom - body.top;
        match self.shape {
            None | Some(Bubble::Pill) => body,
            Some(Bubble::Speech) => Area {
                bottom: body.bottom + 0.45 * h,
                ..body
            },
            Some(Bubble::Thought) => {
                let cloud = body.grow(0.2 * w / 2.0, 0.35 * h / 2.0);
                let puffs = Area {
                    left: body.left + 0.05 * w - 0.1 * h,
                    top: body.bottom,
                    right: body.left + 0.15 * w + 0.15 * h,
                    bottom: body.bottom + 0.72 * h,
                };
                cloud.union(puffs)
            }
            Some(Bubble::Burst) => body.grow(0.55 * w / 2.0, 0.55 * h / 2.0),
            Some(Bubble::Banner) => body.grow(0.35 * h, 0.0),
        }
    }
}

/// What [`compose`] paints into a premultiplied picture `width` pixels
/// wide. `fill` is how much of each pixel the letters cover, `grown` the
/// same for their outlines, each shadow's mask what casts it, and `shape`
/// everything that glows, each from 0 to 1. `background` is the box behind
/// the letters in picture pixels.
pub(crate) struct Layers<'a> {
    pub(crate) style: &'a Style,
    /// Premultiplied.
    pub(crate) ink: [u8; 4],
    pub(crate) fill: &'a [f32],
    /// Each pixel's ink, premultiplied; `ink` where it has none.
    pub(crate) inks: &'a [[u8; 4]],
    pub(crate) grown: Option<&'a [f32]>,
    /// Each pixel's outline color, premultiplied.
    pub(crate) rings: &'a [[u8; 4]],
    pub(crate) shadows: &'a [(Shadow, Vec<f32>)],
    pub(crate) shape: &'a [f32],
    pub(crate) background: Option<Area>,
    /// Boxes behind runs of glyphs, before the origin.
    pub(crate) boxes: &'a [crate::paint::WordBox],
    pub(crate) origin: (f32, f32),
    pub(crate) width: usize,
    pub(crate) height: usize,
}

/// Paint the background, shadow, glow, outline, and fill, in that order.
pub(crate) fn compose(layers: &Layers<'_>, rgba: &mut [u8]) {
    let Layers {
        style,
        ink,
        fill,
        inks,
        grown,
        rings,
        shadows,
        shape,
        background,
        boxes,
        origin,
        width,
        height,
    } = *layers;
    if let (Some(back), Some(area)) = (&style.background, background) {
        match back.shape {
            None => paint_box(rgba, width, height, area, back),
            Some(shape) => paint_bubble(rgba, width, height, area, back.color, shape),
        }
    }
    #[allow(clippy::cast_possible_truncation)]
    let (box_w, box_h) = (width as u32, height as u32);
    for word in boxes {
        crate::paint::paint_round_box(rgba, box_w, box_h, word, origin);
    }
    for (shadow, mask) in shadows {
        let blurred = blur(mask, width, height, shadow.blur);
        #[allow(clippy::cast_possible_truncation)]
        let (dx, dy) = (shadow.dx.round() as isize, shadow.dy.round() as isize);
        // The pixels across a row that the moved mask reaches, and where
        // in the mask each row and the first of them come from.
        let from_x = usize::try_from(-dx).unwrap_or(0);
        let to_x = usize::try_from(dx).unwrap_or(0);
        let Some(across) = width
            .checked_sub(from_x.max(to_x))
            .filter(|across| *across > 0)
        else {
            continue;
        };
        for y in 0..height {
            let Some(sy) = y.checked_add_signed(-dy).filter(|sy| *sy < height) else {
                continue;
            };
            let covers = blurred
                .get(sy * width + from_x..)
                .and_then(|row| row.get(..across));
            let pixels = rgba
                .get_mut((y * width + to_x) * 4..)
                .and_then(|row| row.get_mut(..across * 4));
            if let (Some(covers), Some(pixels)) = (covers, pixels) {
                over_each(pixels, covers, |_| shadow.color, |cover| cover);
            }
        }
    }
    if let Some(glow) = &style.glow {
        // Doubled so the glow stays bright at the letters' edges, where the
        // blur would leave only half.
        let blurred = blur(shape, width, height, glow.blur);
        over_each(
            rgba,
            &blurred,
            |_| glow.color,
            |cover| (cover * 2.0).min(1.0) * glow.strength,
        );
    }
    if let Some(grown) = grown {
        let covers = grown.get(..rings.len()).unwrap_or(grown);
        over_each(
            rgba,
            covers,
            |index| rings.get(index).copied().unwrap_or([0; 4]),
            |cover| cover,
        );
    }
    over_each(
        rgba,
        fill,
        |index| inks.get(index).copied().unwrap_or(ink),
        |cover| cover,
    );
}

/// Lay a color over each pixel of `rgba` by its cover in `covers`, the
/// two side by side: `color` gives a pixel's color, premultiplied, and
/// `shown` how much of it a cover shows.
fn over_each(
    rgba: &mut [u8],
    covers: &[f32],
    color: impl Fn(usize) -> [u8; 4],
    shown: impl Fn(f32) -> f32,
) {
    for (index, (pixel, cover)) in rgba
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(covers)
        .enumerate()
    {
        // Most of a picture is not covered at all.
        if *cover > 0.0 {
            mix(pixel, color(index), shown(*cover));
        }
    }
}

/// A rounded box with one pixel of soft edge.
fn paint_box(rgba: &mut [u8], width: usize, height: usize, area: Area, back: &Background) {
    let half_w = (area.right - area.left) / 2.0;
    let half_h = (area.bottom - area.top) / 2.0;
    if !(half_w > 0.0 && half_h > 0.0) {
        return;
    }
    let center_x = area.left + half_w;
    let center_y = area.top + half_h;
    let radius = back.round * half_w.min(half_h);
    for y in 0..height {
        for x in 0..width {
            let px = pixel(x) + 0.5 - center_x;
            let py = pixel(y) + 0.5 - center_y;
            let qx = px.abs() - (half_w - radius);
            let qy = py.abs() - (half_h - radius);
            let outside = qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0) - radius;
            let cover = (0.5 - outside).clamp(0.0, 1.0);
            over(rgba, y * width + x, back.color, cover);
        }
    }
}

/// A bubble around the box `body`, with one pixel of soft edge.
fn paint_bubble(
    rgba: &mut [u8],
    width: usize,
    height: usize,
    body: Area,
    color: [u8; 4],
    shape: Bubble,
) {
    let w = body.right - body.left;
    let h = body.bottom - body.top;
    if !(w > 0.0 && h > 0.0) {
        return;
    }
    let (hw, hh) = (w / 2.0, h / 2.0);
    let (cx, cy) = (body.left + hw, body.top + hh);
    let tail = [
        (body.left + 0.18 * w, body.bottom - 1.0),
        (body.left + 0.4 * w, body.bottom - 1.0),
        (body.left + 0.08 * w, body.bottom + 0.45 * h),
    ];
    let notched = [
        (body.left - 0.35 * h, body.top),
        (body.right + 0.35 * h, body.top),
        (body.right + 0.1 * h, cy),
        (body.right + 0.35 * h, body.bottom),
        (body.left - 0.35 * h, body.bottom),
        (body.left - 0.1 * h, cy),
    ];
    let spikes: Vec<(f32, f32)> = (0..28_u8)
        .map(|k| {
            let turn = f32::from(k) * std::f32::consts::PI / 14.0;
            let reach = if k % 2 == 0 { 1.55 } else { 1.18 };
            (cx + turn.cos() * hw * reach, cy + turn.sin() * hh * reach)
        })
        .collect();
    for y in 0..height {
        for x in 0..width {
            let p = (pixel(x) + 0.5, pixel(y) + 0.5);
            let local = (p.0 - cx, p.1 - cy);
            let distance = match shape {
                Bubble::Pill => rounded_box(local, hw, hh, hw.min(hh)),
                Bubble::Speech => {
                    rounded_box(local, hw, hh, 0.35 * hw.min(hh)).min(polygon(&tail, p))
                }
                Bubble::Thought => ellipse(local, hw * 1.2, hh * 1.35)
                    .min(circle(
                        p,
                        (body.left + 0.15 * w, body.bottom + 0.3 * h),
                        0.15 * h,
                    ))
                    .min(circle(
                        p,
                        (body.left + 0.05 * w, body.bottom + 0.62 * h),
                        0.09 * h,
                    )),
                Bubble::Burst => polygon(&spikes, p),
                Bubble::Banner => polygon(&notched, p),
            };
            let cover = (0.5 - distance).clamp(0.0, 1.0);
            over(rgba, y * width + x, color, cover);
        }
    }
}

/// Signed distance from `p`, about the middle, to a box `hw` by `hh` half
/// sizes with corners rounded `radius`.
fn rounded_box(p: (f32, f32), hw: f32, hh: f32, radius: f32) -> f32 {
    let qx = p.0.abs() - (hw - radius);
    let qy = p.1.abs() - (hh - radius);
    qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0) - radius
}

/// About the signed distance from `p`, about the middle, to an ellipse.
fn ellipse(p: (f32, f32), rx: f32, ry: f32) -> f32 {
    let k0 = (p.0 / rx).hypot(p.1 / ry);
    let k1 = (p.0 / (rx * rx)).hypot(p.1 / (ry * ry));
    if k1 > 0.0 {
        k0 * (k0 - 1.0) / k1
    } else {
        -rx.min(ry)
    }
}

fn circle(p: (f32, f32), center: (f32, f32), radius: f32) -> f32 {
    (p.0 - center.0).hypot(p.1 - center.1) - radius
}

/// Signed distance from `p` to a closed polygon: negative inside.
fn polygon(points: &[(f32, f32)], p: (f32, f32)) -> f32 {
    let Some(&first) = points.first() else {
        return f32::INFINITY;
    };
    let mut nearest = (p.0 - first.0).powi(2) + (p.1 - first.1).powi(2);
    let mut sign = 1.0;
    let mut previous = points[points.len() - 1];
    for &point in points {
        let edge = (previous.0 - point.0, previous.1 - point.1);
        let w = (p.0 - point.0, p.1 - point.1);
        let length = edge.0 * edge.0 + edge.1 * edge.1;
        let along = if length > 0.0 {
            ((w.0 * edge.0 + w.1 * edge.1) / length).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let b = (w.0 - edge.0 * along, w.1 - edge.1 * along);
        nearest = nearest.min(b.0 * b.0 + b.1 * b.1);
        let above = p.1 >= point.1;
        let below = p.1 < previous.1;
        let turns = edge.0 * w.1 > edge.1 * w.0;
        if (above && below && turns) || (!above && !below && !turns) {
            sign = -sign;
        }
        previous = point;
    }
    sign * nearest.sqrt()
}

/// The most a mask is made smaller by, each way, to be blurred.
const MAX_SHRINK: usize = 8;
/// The widest box summed pixel by pixel; a wider one keeps a running sum.
const SHORT_REACH: usize = 4;

/// Three box blurs, close to a gaussian `radius` pixels wide. A wide blur
/// is made on a mask a half, a quarter or an eighth the size and drawn
/// back up: what it gives is smooth enough that the two differ by about
/// a hundredth of full cover at most, and a title whose letters move has
/// its shadow and its glow blurred again at every frame.
fn blur(mask: &[f32], width: usize, height: usize, radius: f32) -> Vec<f32> {
    let whole = width.checked_mul(height);
    if radius.is_nan()
        || radius < 0.5
        || width == 0
        || height == 0
        || whole.is_none_or(|whole| mask.len() < whole)
    {
        return mask.to_vec();
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let reach = ((radius / 2.0).round() as usize).clamp(1, width.max(height));
    // Three boxes reaching `reach` spread a pixel as a gaussian whose
    // width squared is this.
    let spread = pixel(reach) * pixel(reach + 1);
    // Drawn back up between pixels `shrink` apart, a curve this wide is
    // off by about `shrink` squared over thirty-three times `spread`.
    let mut shrink = 1;
    while shrink < MAX_SHRINK && pixel(shrink * shrink * 4) <= 0.2 * spread {
        shrink *= 2;
    }
    if shrink == 1 {
        let mut out = mask.to_vec();
        boxes(&mut out, width, height, reach, 0.0);
        return out;
    }
    let (small_w, small_h) = (width.div_ceil(shrink), height.div_ceil(shrink));
    let mut small = smaller(mask, width, height, shrink);
    // Making the mask smaller and drawing it back up each spread it a
    // little; the boxes make up the rest, in the small mask's pixels.
    let across = pixel(shrink * shrink);
    let left = ((spread - (across - 1.0) / 12.0 - across / 6.0) / across).max(0.0);
    let (reach, edge) = fitted(left / 3.0);
    boxes(&mut small, small_w, small_h, reach, edge);
    larger(&small, small_w, small_h, shrink, width, height)
}

/// The box that spreads a pixel by `spread`, a third of what three of
/// them do: how far it reaches in whole pixels, and how much it counts
/// of the pixel one further at each end, from nothing to all of it. A
/// box of whole pixels alone could only come near.
fn fitted(spread: f32) -> (usize, f32) {
    if spread.is_nan() || spread <= 0.0 {
        return (0, 0.0);
    }
    // A box reaching `reach` spreads by a third of `reach` times
    // `reach + 1`.
    let mut reach = 0_usize;
    while reach < 4_096 && pixel((reach + 1) * (reach + 2)) / 3.0 <= spread {
        reach += 1;
    }
    let whole = pixel(reach * (reach + 1)) / 3.0;
    let next = pixel((reach + 1) * (reach + 1));
    let edge = pixel(reach * 2 + 1) * (spread - whole) / (2.0 * (next - spread));
    (
        reach,
        if edge.is_finite() {
            edge.clamp(0.0, 1.0)
        } else {
            0.0
        },
    )
}

/// Blur `mask` in place with three boxes, each reaching `reach` pixels
/// each way and counting `edge` of the pixel one further, along the rows
/// and then down the columns. Past the mask's edge is empty.
fn boxes(mask: &mut [f32], width: usize, height: usize, reach: usize, edge: f32) {
    if reach == 0 && edge <= 0.0 {
        return;
    }
    let scale = 1.0 / (pixel(reach * 2 + 1) + edge * 2.0);
    let mut other = vec![0.0_f32; mask.len()];
    let mut sums = vec![0.0_f32; width];
    let row = |at: usize| at * width..(at + 1) * width;
    for _ in 0..3 {
        for (from, to) in mask
            .chunks_exact(width)
            .zip(other.chunks_exact_mut(width))
            .take(height)
        {
            across(from, to, reach, edge, scale);
        }
        // Down the columns, every column's sum kept side by side so a
        // row is added and taken away whole.
        sums.fill(0.0);
        for at in 0..reach.min(height) {
            for (sum, value) in sums.iter_mut().zip(&other[row(at)]) {
                *sum += value;
            }
        }
        for at in 0..height {
            if at + reach < height {
                for (sum, value) in sums.iter_mut().zip(&other[row(at + reach)]) {
                    *sum += value;
                }
            }
            for (slot, sum) in mask[row(at)].iter_mut().zip(&sums) {
                *slot = *sum;
            }
            if edge > 0.0 {
                // The rows one past the box, each counted in part.
                let ends = [at.checked_sub(reach + 1), Some(at + reach + 1)];
                for end in ends.into_iter().flatten().filter(|end| *end < height) {
                    for (slot, value) in mask[row(at)].iter_mut().zip(&other[row(end)]) {
                        *slot += value * edge;
                    }
                }
            }
            for slot in &mut mask[row(at)] {
                *slot = (*slot * scale).clamp(0.0, 1.0);
            }
            if let Some(leaving) = at.checked_sub(reach) {
                for (sum, value) in sums.iter_mut().zip(&other[row(leaving)]) {
                    *sum -= value;
                }
            }
        }
    }
}

/// One row through one box: each pixel of `to` the mean of the pixels of
/// `from` within `reach` of it, with `edge` of the pixel one further at
/// each end, counting what is past either end of the row as empty.
fn across(from: &[f32], to: &mut [f32], reach: usize, edge: f32, scale: f32) {
    let width = from.len();
    // How far the box reaches with its part pixels.
    let outer = reach + usize::from(edge > 0.0);
    let at = |index: Option<usize>| index.and_then(|index| from.get(index)).copied();
    let ends = |index: usize| {
        if edge > 0.0 {
            let before = at(index.checked_sub(reach + 1)).unwrap_or(0.0);
            (before + at(Some(index + reach + 1)).unwrap_or(0.0)) * edge
        } else {
            0.0
        }
    };
    if outer <= SHORT_REACH && width > outer * 2 {
        // A narrow box is summed a neighbour at a time, every pixel of
        // the row at once.
        let inner = width - outer * 2;
        let (head, rest) = to.split_at_mut(outer);
        let (middle, tail) = rest.split_at_mut(inner);
        middle.fill(0.0);
        let first = outer - reach;
        for offset in first..=first + reach * 2 {
            for (slot, value) in middle.iter_mut().zip(&from[offset..offset + inner]) {
                *slot += value;
            }
        }
        if edge > 0.0 {
            for offset in [0, outer * 2] {
                for (slot, value) in middle.iter_mut().zip(&from[offset..offset + inner]) {
                    *slot += value * edge;
                }
            }
        }
        for slot in middle.iter_mut() {
            *slot = (*slot * scale).clamp(0.0, 1.0);
        }
        let whole = |index: usize| {
            let low = index.saturating_sub(reach);
            let high = (index + reach + 1).min(width);
            from[low..high].iter().sum::<f32>() + ends(index)
        };
        for (index, slot) in head.iter_mut().enumerate() {
            *slot = (whole(index) * scale).clamp(0.0, 1.0);
        }
        for (index, slot) in tail.iter_mut().enumerate() {
            *slot = (whole(outer + inner + index) * scale).clamp(0.0, 1.0);
        }
        return;
    }
    let mut sum: f32 = from.iter().take(reach).sum();
    for (index, slot) in to.iter_mut().enumerate() {
        if let Some(entering) = from.get(index + reach) {
            sum += entering;
        }
        *slot = ((sum + ends(index)) * scale).clamp(0.0, 1.0);
        if let Some(leaving) = at(index.checked_sub(reach)) {
            sum -= leaving;
        }
    }
}

/// `mask` with each square of `shrink` by `shrink` pixels as one, their
/// mean; a square that hangs over the edge counts what is past it as
/// empty.
fn smaller(mask: &[f32], width: usize, height: usize, shrink: usize) -> Vec<f32> {
    let (small_w, small_h) = (width.div_ceil(shrink), height.div_ceil(shrink));
    let scale = 1.0 / pixel(shrink * shrink);
    let mut small = vec![0.0_f32; small_w * small_h];
    for (y, row) in mask.chunks_exact(width).take(height).enumerate() {
        let to = &mut small[(y / shrink) * small_w..][..small_w];
        for (slot, square) in to.iter_mut().zip(row.chunks(shrink)) {
            *slot += square.iter().sum::<f32>();
        }
    }
    for value in &mut small {
        *value *= scale;
    }
    small
}

/// A mask `small_w` by `small_h` drawn `shrink` times the size, into one
/// `width` by `height`, each pixel mixed from the four nearest. A pixel
/// past the middle of the last small one, at any side, carries on the
/// slope of the two nearest.
fn larger(
    small: &[f32],
    small_w: usize,
    small_h: usize,
    shrink: usize,
    width: usize,
    height: usize,
) -> Vec<f32> {
    // Where a pixel's middle falls among the small mask's: the one
    // before it, the one after, and how far it is toward the second.
    let place = |at: usize, count: usize| {
        let along = (pixel(at) + 0.5) / pixel(shrink) - 0.5;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let first = (along.max(0.0) as usize).min(count.saturating_sub(2));
        (first, (first + 1).min(count - 1), along - pixel(first))
    };
    // Each small row drawn out across the width, then the rows mixed.
    let columns: Vec<(usize, usize, f32)> = (0..width).map(|x| place(x, small_w)).collect();
    let mut wide = vec![0.0_f32; width * small_h];
    for (from, to) in small
        .chunks_exact(small_w)
        .zip(wide.chunks_exact_mut(width))
    {
        for (slot, (first, second, toward)) in to.iter_mut().zip(&columns) {
            *slot = from[*first] + (from[*second] - from[*first]) * toward;
        }
    }
    let mut out = vec![0.0_f32; width * height];
    for (y, to) in out.chunks_exact_mut(width).enumerate() {
        let (first, second, toward) = place(y, small_h);
        let (above, below) = (
            &wide[first * width..][..width],
            &wide[second * width..][..width],
        );
        for ((slot, above), below) in to.iter_mut().zip(above).zip(below) {
            *slot = (above + (below - above) * toward).clamp(0.0, 1.0);
        }
    }
    out
}

/// Lay premultiplied `color` at `cover` over the pixel at `index`.
pub(crate) fn over(rgba: &mut [u8], index: usize, color: [u8; 4], cover: f32) {
    if let Some(pixel) = index.checked_mul(4).and_then(|at| rgba.get_mut(at..at + 4)) {
        mix(pixel, color, cover);
    }
}

/// Lay premultiplied `color` at `cover` over the four bytes of `pixel`.
fn mix(pixel: &mut [u8], color: [u8; 4], cover: f32) {
    if cover.is_nan() || cover <= 0.0 {
        return;
    }
    let cover = cover.min(1.0);
    let alpha = f32::from(color[3]) * cover;
    let keep = 1.0 - alpha / 255.0;
    for (slot, channel) in pixel.iter_mut().zip(color) {
        let mixed = f32::from(channel) * cover + f32::from(*slot) * keep;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            *slot = mixed.round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// A hex color at `alpha` from 0 to 1, premultiplied.
fn color(text: &str, alpha: f64) -> Result<[u8; 4], TextError> {
    let (rgb, extra) = crate::source::parse_hex(text.trim())?;
    let mut alpha = unit(alpha);
    if let Some(channel) = extra {
        alpha *= f64::from(channel) / 255.0;
    }
    Ok(premultiply([rgb[0], rgb[1], rgb[2], byte(alpha)]))
}

/// Scale color channels by alpha.
pub(crate) fn premultiply(color: [u8; 4]) -> [u8; 4] {
    let scale =
        |channel: u8| u8::try_from(u16::from(channel) * u16::from(color[3]) / 255).unwrap_or(255);
    [scale(color[0]), scale(color[1]), scale(color[2]), color[3]]
}

fn unit(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn byte(value: f64) -> u8 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let byte = (unit(value) * 255.0).round() as u8;
    byte
}

/// `value` ems in pixels at `font_size`, from 0 to four ems.
fn em(value: f64, font_size: f32) -> f32 {
    if !value.is_finite() || value <= 0.0 {
        return 0.0;
    }
    #[allow(clippy::cast_possible_truncation)]
    let value = value.min(4.0) as f32;
    value * font_size
}

fn pixel(value: usize) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    {
        value as f32
    }
}

#[cfg(test)]
mod tests {
    use super::{blur, boxes, fitted};

    /// A mask of letter-like shapes: bars, a ring and a few dots, hard
    /// edged, with room around them.
    fn shapes(width: usize, height: usize) -> Vec<f32> {
        let mut mask = vec![0.0_f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let bar = (x / 37) % 2 == 0 && (20..height.saturating_sub(20)).contains(&y);
                let (cx, cy) = (x.abs_diff(width / 2), y.abs_diff(height / 2));
                let ring = (900..1600).contains(&(cx * cx + cy * cy));
                let dot = x % 53 < 3 && y % 29 < 3;
                if (bar && x > 30 && x + 30 < width) || ring || dot {
                    mask[y * width + x] = 1.0;
                }
            }
        }
        mask
    }

    /// The blur made the plain way: three boxes on the mask at its size.
    fn plain(mask: &[f32], width: usize, height: usize, radius: f32) -> Vec<f32> {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let reach = ((radius / 2.0).round() as usize).max(1);
        let mut out = mask.to_vec();
        boxes(&mut out, width, height, reach, 0.0);
        out
    }

    #[test]
    fn a_wide_blur_made_small_is_the_blur_made_at_full_size() {
        let (width, height) = (301, 157);
        let mask = shapes(width, height);
        let mut widest = 0.0_f32;
        for radius in [
            1.0_f32, 3.0, 6.0, 9.0, 13.0, 18.0, 21.6, 30.0, 47.0, 64.0, 120.0,
        ] {
            let quick = blur(&mask, width, height, radius);
            let slow = plain(&mask, width, height, radius);
            assert_eq!(quick.len(), slow.len());
            let off = quick
                .iter()
                .zip(&slow)
                .map(|(quick, slow)| (quick - slow).abs())
                .fold(0.0_f32, f32::max);
            // By about a hundredth of full cover at most, at the picture's
            // edge; less than that inside it.
            assert!(off < 0.012, "radius {radius}: off by {off}");
            widest = widest.max(off);
            // What was there is still there: a blur moves cover about
            // and loses only what goes past the edge.
            let (kept, had): (f32, f32) = (quick.iter().sum(), slow.iter().sum());
            assert!(
                (kept - had).abs() < 0.02 * had,
                "radius {radius}: {kept} of {had}"
            );
            assert!(quick.iter().all(|cover| (0.0..=1.0).contains(cover)));
        }
        assert!(widest > 0.0, "no blur was made small");
        // A narrow blur is made at full size, the same to the bit.
        for radius in [0.6_f32, 1.0, 2.0, 3.0] {
            assert_eq!(
                blur(&mask, width, height, radius),
                plain(&mask, width, height, radius)
            );
        }
    }

    #[test]
    fn a_narrow_box_summed_by_neighbours_is_the_running_sum() {
        // Reaches up to four are summed a neighbour at a time; the same
        // row through the running sum gives the same means.
        let row: Vec<f32> = (0..97_u8).map(|at| f32::from(at % 7) / 6.0).collect();
        for reach in 1..=6_usize {
            #[allow(clippy::cast_precision_loss)]
            let scale = 1.0 / (reach * 2 + 1) as f32;
            let mut quick = vec![0.0; row.len()];
            super::across(&row, &mut quick, reach, 0.0, scale);
            for (at, mean) in quick.iter().enumerate() {
                let from = at.saturating_sub(reach);
                let to = (at + reach + 1).min(row.len());
                let wanted = row[from..to].iter().sum::<f32>() * scale;
                assert!((mean - wanted).abs() < 1e-5, "reach {reach} at {at}");
            }
        }
        // A row shorter than the box.
        let mut short = vec![0.0; 3];
        super::across(&[1.0, 1.0, 1.0], &mut short, 4, 0.0, 1.0 / 9.0);
        assert!(short.iter().all(|mean| (mean - 3.0 / 9.0).abs() < 1e-6));
    }

    #[test]
    fn a_box_is_fitted_to_spread_as_far_as_asked() {
        // Whole boxes: reaching one spreads by two thirds, two by two.
        assert_eq!(fitted(0.0), (0, 0.0));
        let (reach, edge) = fitted(2.0 / 3.0);
        assert_eq!(reach, 1);
        assert!(edge.abs() < 1e-5);
        let (reach, edge) = fitted(2.0);
        assert_eq!(reach, 2);
        assert!(edge.abs() < 1e-5);
        // Between two, the pixel one further counts in part, and the box
        // spreads by what was asked: the mean square of how far its
        // pixels are from its middle.
        for spread in [0.1_f32, 0.5, 1.0, 1.5, 3.0, 7.7, 40.0] {
            let (reach, edge) = fitted(spread);
            assert!((0.0..=1.0).contains(&edge), "{spread}");
            let mut squares = 0.0_f32;
            let mut weight = 1.0_f32;
            for step in 1..=reach + 1 {
                #[allow(clippy::cast_precision_loss)]
                let far = step as f32;
                let counted = if step > reach { edge } else { 1.0 };
                squares += 2.0 * counted * far * far;
                weight += 2.0 * counted;
            }
            assert!(
                (squares / weight - spread).abs() < 1e-3 * spread.max(1.0),
                "{spread}"
            );
        }
        // No spread that is no number, and one no mask is wide enough for.
        assert_eq!(fitted(f32::NAN), (0, 0.0));
        assert_eq!(fitted(-1.0), (0, 0.0));
        assert!(fitted(f32::MAX).0 <= 4_096);
    }

    #[test]
    fn an_odd_mask_or_radius_is_blurred_or_left_and_a_large_one_is_quick() {
        // Nothing to blur, a mask shorter than its size says, and radii
        // that are no radius: the mask comes back as it was.
        assert!(blur(&[], 0, 0, 8.0).is_empty());
        assert_eq!(blur(&[1.0, 0.0], 4, 4, 8.0), [1.0, 0.0]);
        assert_eq!(blur(&[1.0; 4], usize::MAX, 2, 8.0), [1.0; 4]);
        for radius in [f32::NAN, -3.0, 0.0, 0.4] {
            assert_eq!(
                blur(&[1.0, 0.0, 0.0, 1.0], 2, 2, radius),
                [1.0, 0.0, 0.0, 1.0]
            );
        }
        // A radius far wider than the mask, and an endless one.
        for radius in [1.0e6_f32, f32::MAX, f32::INFINITY] {
            let blurred = blur(&[1.0; 15], 5, 3, radius);
            assert_eq!(blurred.len(), 15);
            assert!(blurred.iter().all(|cover| (0.0..=1.0).contains(cover)));
        }
        // One pixel wide, one tall, and sizes no square divides.
        for (width, height) in [(1, 9), (9, 1), (7, 5), (17, 3), (33, 31)] {
            let mask = vec![1.0; width * height];
            for radius in [2.0_f32, 12.0, 40.0] {
                assert_eq!(blur(&mask, width, height, radius).len(), width * height);
            }
        }
        // A picture as large as a title in a 4K frame, blurred wide.
        let (width, height) = (3_840, 600);
        let mask = shapes(width, height);
        let started = std::time::Instant::now();
        let blurred = blur(&mask, width, height, 60.0);
        assert_eq!(blurred.len(), mask.len());
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
    }
}
