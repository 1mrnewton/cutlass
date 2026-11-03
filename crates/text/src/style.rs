//! Outline, shadow, and background drawn with a title's letters.
//!
//! Each is read from the title in units of its size, so a title drawn
//! larger keeps the same look: `border_width` is the outline's width in
//! ems, `shadow_distance` hundredths of an em, `shadow_smoothing` the blur
//! from 0 to 1 (a quarter of an em at 1), and `background_width` and
//! `background_height` the room around the letters in ems. The angle is in
//! degrees, counterclockwise from the right. `background_round_radius` from
//! 0 to 1 rounds the box's corners up to half its height.
//!
//! An outline is grown from the glyph distance field, so it reaches at most
//! the field's spread beyond each letter; a wider one is drawn that wide.

use cutlass_model::MaterialText;

use crate::TextError;
use crate::atlas::{ATLAS_EM, SPREAD};

/// What is drawn with the letters, in picture pixels.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Style {
    pub(crate) outline: Option<Outline>,
    pub(crate) shadow: Option<Shadow>,
    pub(crate) background: Option<Background>,
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
    /// opacity is left out.
    pub(crate) fn of(material: &MaterialText, font_size: f32) -> Result<Self, TextError> {
        let global = unit(material.global_alpha);
        let outline = if material.border_color.trim().is_empty() {
            None
        } else {
            let width =
                em(material.border_width, font_size).min((cell_pad(font_size) - 1.0).max(0.0));
            let color = color(&material.border_color, material.border_alpha * global)?;
            (width > 0.0 && color[3] > 0).then_some(Outline { color, width })
        };
        let shadow = if material.has_shadow {
            let shade = if material.shadow_color.trim().is_empty() {
                "#000000"
            } else {
                &material.shadow_color
            };
            let color = color(shade, material.shadow_alpha * global)?;
            let distance = em(material.shadow_distance / 100.0, font_size);
            let angle = if material.shadow_angle.is_finite() {
                material.shadow_angle.to_radians()
            } else {
                0.0
            };
            #[allow(clippy::cast_possible_truncation)]
            let (dx, dy) = (
                distance * angle.cos() as f32,
                -distance * angle.sin() as f32,
            );
            let blur = em(unit(material.shadow_smoothing) * 0.25, font_size);
            (color[3] > 0).then_some(Shadow {
                color,
                dx,
                dy,
                blur,
            })
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
                })
            };
        Ok(Self {
            outline,
            shadow,
            background,
        })
    }

    /// The picture's box for letters whose cells cover `cells`.
    pub(crate) fn bounds(&self, cells: Area, font_size: f32) -> Area {
        let mut area = cells;
        if let Some(shadow) = &self.shadow {
            let reach = shadow.blur * 2.0 + 1.0;
            area = area.union(cells.shift(shadow.dx, shadow.dy).grow(reach, reach));
        }
        if let Some(background) = &self.background {
            area = area.union(background.area(cells, font_size));
        }
        area
    }
}

impl Background {
    /// The box behind letters whose cells cover `cells`.
    pub(crate) fn area(&self, cells: Area, font_size: f32) -> Area {
        let pad = cell_pad(font_size);
        cells.grow(self.pad_x - pad, self.pad_y - pad)
    }
}

/// What [`compose`] paints into a premultiplied picture `width` pixels
/// wide. `fill` is how much of each pixel the letters cover, `grown` the
/// same for the outline, and `shape` everything that casts a shadow, each
/// from 0 to 1. `background` is the box behind the letters in picture
/// pixels.
pub(crate) struct Layers<'a> {
    pub(crate) style: &'a Style,
    /// Premultiplied.
    pub(crate) ink: [u8; 4],
    pub(crate) fill: &'a [f32],
    pub(crate) grown: Option<&'a [f32]>,
    pub(crate) shape: &'a [f32],
    pub(crate) background: Option<Area>,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

/// Paint the background, shadow, outline, and fill, in that order.
pub(crate) fn compose(layers: &Layers<'_>, rgba: &mut [u8]) {
    let Layers {
        style,
        ink,
        fill,
        grown,
        shape,
        background,
        width,
        height,
    } = *layers;
    if let (Some(back), Some(area)) = (&style.background, background) {
        paint_box(rgba, width, height, area, back);
    }
    if let Some(shadow) = &style.shadow {
        let blurred = blur(shape, width, height, shadow.blur);
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
    if let (Some(outline), Some(grown)) = (&style.outline, grown) {
        for (index, cover) in grown.iter().enumerate() {
            over(rgba, index, outline.color, *cover);
        }
    }
    for (index, cover) in fill.iter().enumerate() {
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
