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
        width,
        height,
    } = *layers;
    if let (Some(back), Some(area)) = (&style.background, background) {
        match back.shape {
            None => paint_box(rgba, width, height, area, back),
            Some(shape) => paint_bubble(rgba, width, height, area, back.color, shape),
        }
    }
    for (shadow, mask) in shadows {
        let blurred = blur(mask, width, height, shadow.blur);
        #[allow(clippy::cast_possible_truncation)]
        let (dx, dy) = (shadow.dx.round() as isize, shadow.dy.round() as isize);
        for y in 0..height {
            for x in 0..width {
                let (Some(sx), Some(sy)) = (x.checked_add_signed(-dx), y.checked_add_signed(-dy))
                else {
                    continue;
                };
                if sx >= width || sy >= height {
                    continue;
                }
                let cover = blurred[sy * width + sx];
                over(rgba, y * width + x, shadow.color, cover);
            }
        }
    }
    if let Some(glow) = &style.glow {
        // Doubled so the glow stays bright at the letters' edges, where the
        // blur would leave only half.
        let blurred = blur(shape, width, height, glow.blur);
        for (index, cover) in blurred.iter().enumerate() {
            over(
                rgba,
                index,
                glow.color,
                (cover * 2.0).min(1.0) * glow.strength,
            );
        }
    }
    if let Some(grown) = grown {
        for (index, cover) in grown.iter().enumerate() {
            if let Some(ring) = rings.get(index) {
                over(rgba, index, *ring, *cover);
            }
        }
    }
    for (index, cover) in fill.iter().enumerate() {
        let ink = inks.get(index).copied().unwrap_or(ink);
        over(rgba, index, ink, *cover);
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

/// Three box blurs, close to a gaussian `radius` pixels wide.
fn blur(mask: &[f32], width: usize, height: usize, radius: f32) -> Vec<f32> {
    let mut out = mask.to_vec();
    if radius.is_nan() || radius < 0.5 || width == 0 || height == 0 {
        return out;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let reach = ((radius / 2.0).round() as usize).max(1);
    let mut line = Vec::new();
    for _ in 0..3 {
        for y in 0..height {
            line.clear();
            line.extend((0..width).map(|x| out[y * width + x]));
            box_pass(&line, reach, |x, value| out[y * width + x] = value);
        }
        for x in 0..width {
            line.clear();
            line.extend((0..height).map(|y| out[y * width + x]));
            box_pass(&line, reach, |y, value| out[y * width + x] = value);
        }
    }
    out
}

fn box_pass(line: &[f32], reach: usize, mut put: impl FnMut(usize, f32)) {
    let span = pixel(reach * 2 + 1);
    let mut sum: f32 = line.iter().take(reach).sum();
    for index in 0..line.len() {
        if let Some(entering) = line.get(index + reach) {
            sum += entering;
        }
        put(index, (sum / span).clamp(0.0, 1.0));
        if let Some(leaving) = index.checked_sub(reach).and_then(|at| line.get(at)) {
            sum -= leaving;
        }
    }
}

/// Lay premultiplied `color` at `cover` over the pixel at `index`.
pub(crate) fn over(rgba: &mut [u8], index: usize, color: [u8; 4], cover: f32) {
    if cover.is_nan() || cover <= 0.0 {
        return;
    }
    let Some(pixel) = index.checked_mul(4).and_then(|at| rgba.get_mut(at..at + 4)) else {
        return;
    };
    let cover = cover.min(1.0);
    let alpha = f32::from(color[3]) * cover;
    let keep = 1.0 - alpha / 255.0;
    for channel in 0..4 {
        let mixed = f32::from(color[channel]) * cover + f32::from(pixel[channel]) * keep;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            pixel[channel] = mixed.round().clamp(0.0, 255.0) as u8;
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
