//! Outline, shadow, and background drawn with a title's letters.
//!
//! Each is read from the title in units of its size, so a title drawn
//! larger keeps the same look: `border_width` is the outline's width in
//! ems, `shadow_distance` hundredths of an em, `shadow_smoothing` the blur
//! from 0 to 1 (a quarter of an em at 1), and `background_width` and
//! `background_height` the room around the letters in ems. The angle is in
//! degrees, counterclockwise from the right. `background_round_radius` from
//! 0 to 1 rounds the box's corners up to half its height. With
//! `background_lines` each line has a box of its own, as wide as its own
//! letters, and boxes that meet are one shape. `background_offset_x` and
//! `background_offset_y` move the box right and up from the letters, in
//! ems. A title's `bubble` shapes the box as a [`Bubble`] instead, one
//! around every line: a speech bubble's tail, a thought's puffs and a
//! burst's spikes reach out past the box, and the picture grows to hold
//! them.
//!
//! A glow is the clip's `bloom` effect: its `strength` from 0 to 1 is how
//! bright it is, and its `range` from 0 to 1 how far it spreads, up to
//! [`GLOW_REACH`] ems. A glow with `rays` also streams out of the letters
//! as rays of light, from a point behind their middle: the rays reach
//! from 0.15 of the way past the letters to 0.6 with the range, and
//! lean right and up by `lean_x` and `lean_y`, the point moving the
//! other way by up to half the letters' width and height at
//! [`MAX_GLOW_LEAN`] degrees.
//!
//! An outline is grown from the glyph distance field, so it reaches at most
//! the field's spread beyond each letter; a wider one is drawn that wide.

use cutlass_model::{BloomParams, Bubble, MAX_GLOW_LEAN, MaterialText};

use cutlass_picture::blur::{blur_mask as blur, larger, smaller};

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

/// How far rays reach at no range and at full range, as a share of each
/// point's distance from where they stream from.
const RAYS_SHORTEST: f64 = 0.15;
const RAYS_LONGEST: f64 = 0.6;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Glow {
    /// Premultiplied.
    pub(crate) color: [u8; 4],
    pub(crate) blur: f32,
    pub(crate) strength: f32,
    pub(crate) rays: Option<Rays>,
}

/// Rays of light streaming out of the letters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Rays {
    /// How far a ray reaches, as a share of its distance from the point
    /// they stream from: a point that far out is lit by the letters
    /// between it and that much nearer in.
    pub(crate) reach: f32,
    /// Which way they lean, right and down, each from -1 to 1.
    pub(crate) lean: (f32, f32),
}

impl Rays {
    /// The point the rays stream from, behind the letters in `area`.
    fn from(&self, area: Area) -> (f32, f32) {
        let (half_w, half_h) = (
            (area.right - area.left) / 2.0,
            (area.bottom - area.top) / 2.0,
        );
        (
            area.left + half_w - self.lean.0 * half_w,
            area.top + half_h - self.lean.1 * half_h,
        )
    }

    /// Where the rays from the letters in `area` end: the area drawn
    /// out from the point they stream from.
    fn extent(&self, area: Area) -> Area {
        let (x, y) = self.from(area);
        let out = 1.0 / (1.0 - self.reach).max(0.05);
        area.union(Area {
            left: x + (area.left - x) * out,
            top: y + (area.top - y) * out,
            right: x + (area.right - x) * out,
            bottom: y + (area.bottom - y) * out,
        })
    }
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
    /// A box for each line rather than one around them all.
    pub(crate) lines: bool,
    /// How far the box is moved from the letters: right and down.
    pub(crate) shift: (f32, f32),
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
    pub(crate) fn union(self, other: Self) -> Self {
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
                let shape = Bubble::of(&material.bubble);
                (color[3] > 0).then(|| Background {
                    color,
                    pad_x: em(material.background_width, font_size),
                    pad_y: em(material.background_height, font_size),
                    round,
                    shape,
                    lines: material.background_lines && shape.is_none(),
                    shift: (
                        signed_em(material.background_offset_x, font_size),
                        -signed_em(material.background_offset_y, font_size),
                    ),
                })
            };
        let glow = match bloom {
            Some(bloom) if !bloom.color.trim().is_empty() => {
                let color = color(&bloom.color, global)?;
                #[allow(clippy::cast_possible_truncation)]
                let strength = unit(bloom.strength) as f32;
                let range = unit(bloom.range);
                let lean = |degrees: f64| {
                    #[allow(clippy::cast_possible_truncation)]
                    let lean = if degrees.is_finite() {
                        (degrees / MAX_GLOW_LEAN).clamp(-1.0, 1.0) as f32
                    } else {
                        0.0
                    };
                    lean
                };
                // Rays carry most of the spread, so the glow about the
                // letters stays tight.
                let spread = if bloom.rays { 0.5 } else { 1.0 };
                #[allow(clippy::cast_possible_truncation)]
                let rays = bloom.rays.then(|| Rays {
                    reach: (RAYS_SHORTEST + range * (RAYS_LONGEST - RAYS_SHORTEST)) as f32,
                    lean: (lean(bloom.lean_x), -lean(bloom.lean_y)),
                });
                (color[3] > 0 && strength > 0.0).then(|| Glow {
                    color,
                    blur: em(range * GLOW_REACH * spread, font_size).max(1.0),
                    strength,
                    rays,
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
    /// `shadows`. A box for each line lies inside the one around them all,
    /// so that one holds them.
    pub(crate) fn bounds(&self, cells: Area, font_size: f32, shadows: &[Shadow]) -> Area {
        let mut area = cells;
        for shadow in shadows {
            let reach = shadow.blur * 2.0 + 1.0;
            area = area.union(cells.shift(shadow.dx, shadow.dy).grow(reach, reach));
        }
        if let Some(glow) = &self.glow {
            let reach = glow.blur * 2.0 + 1.0;
            let lit = glow.rays.map_or(cells, |rays| rays.extent(cells));
            area = area.union(lit.grow(reach, reach));
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
    /// The box behind letters whose cells cover `cells`, moved by its
    /// shift.
    pub(crate) fn area(&self, cells: Area, font_size: f32) -> Area {
        let pad = cell_pad(font_size);
        cells
            .grow(self.pad_x - pad, self.pad_y - pad)
            .shift(self.shift.0, self.shift.1)
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
/// everything that glows, each from 0 to 1. `background` is the boxes
/// behind the letters in picture pixels: one, or one for each line.
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
    pub(crate) background: &'a [Area],
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
    if let Some(back) = &style.background {
        match (back.shape, background.first()) {
            (Some(shape), Some(area)) => {
                paint_bubble(rgba, width, height, *area, back.color, shape);
            }
            (None, Some(_)) => paint_boxes(rgba, width, height, background, back),
            (_, None) => {}
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
        if let Some(rays) = &glow.rays
            && let Some(lit) = inked(shape, width, height)
        {
            let streams = streams(shape, width, height, rays.from(lit), rays.reach);
            over_each(
                rgba,
                &streams,
                |_| glow.color,
                |cover| (cover * RAYS_GAIN).min(1.0).powf(1.5) * glow.strength,
            );
        }
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

/// How much brighter rays are drawn than the share of their path the
/// letters cover, so a ray between letters still shows.
const RAYS_GAIN: f32 = 2.0;

/// The box around what `mask` covers, in its pixels; none when it covers
/// nothing.
fn inked(mask: &[f32], width: usize, height: usize) -> Option<Area> {
    let mut found: Option<Area> = None;
    for (y, row) in mask.chunks_exact(width.max(1)).take(height).enumerate() {
        let (Some(first), Some(last)) = (
            row.iter().position(|cover| *cover > 0.0),
            row.iter().rposition(|cover| *cover > 0.0),
        ) else {
            continue;
        };
        let line = Area {
            left: pixel(first),
            top: pixel(y),
            right: pixel(last + 1),
            bottom: pixel(y + 1),
        };
        found = Some(found.map_or(line, |found| found.union(line)));
    }
    found
}

/// Taps in each pass of [`streams`]; three passes, each this many times
/// further apart than the last, cover a ray with this many cubed.
const STREAM_TAPS: usize = 8;

/// `mask` streamed out from `from` as rays: each pixel the mean of the
/// mask along the line from it toward `from`, `reach` of the way. Drawn
/// at half size, and in three passes of a few taps each, each pass's
/// taps further apart, so a long ray is still smooth.
fn streams(mask: &[f32], width: usize, height: usize, from: (f32, f32), reach: f32) -> Vec<f32> {
    const SHRINK: usize = 2;
    let whole = width.checked_mul(height);
    if width == 0 || height == 0 || whole.is_none_or(|whole| mask.len() < whole) {
        return vec![0.0; mask.len()];
    }
    let (small_w, small_h) = (width.div_ceil(SHRINK), height.div_ceil(SHRINK));
    let mut small = smaller(mask, width, height, SHRINK);
    // The point in the small mask's pixels, each counted at its middle.
    let (from_x, from_y) = (from.0 / pixel(SHRINK) - 0.5, from.1 / pixel(SHRINK) - 0.5);
    let near = (1.0 - reach.clamp(0.0, 0.95)).ln();
    let taps = pixel(STREAM_TAPS);
    let mut next = vec![0.0_f32; small.len()];
    for pass in 0..3_i32 {
        // Each tap moves this much of the log of the way in, so the three
        // passes' taps reach from the pixel to `reach` of the way.
        let step = near / taps.powi(3 - pass);
        // The last pass, whose taps lie furthest apart, counts the near
        // ones most, so a ray fades as it leaves the letters.
        let fade = if pass == 2 { 0.8 / taps } else { 0.0 };
        let shares: Vec<(f32, f32)> = (0..STREAM_TAPS)
            .map(|tap| ((step * pixel(tap)).exp(), 1.0 - fade * pixel(tap)))
            .collect();
        let weight: f32 = shares.iter().map(|(_, weight)| weight).sum();
        for (y, row) in next.chunks_exact_mut(small_w).enumerate() {
            let dy = pixel(y) - from_y;
            for (x, slot) in row.iter_mut().enumerate() {
                let dx = pixel(x) - from_x;
                let sum: f32 = shares
                    .iter()
                    .map(|(share, weight)| {
                        weight
                            * sample(
                                &small,
                                small_w,
                                small_h,
                                from_x + dx * share,
                                from_y + dy * share,
                            )
                    })
                    .sum();
                *slot = sum / weight;
            }
        }
        std::mem::swap(&mut small, &mut next);
    }
    larger(&small, small_w, small_h, SHRINK, width, height)
}

/// `mask` at `(x, y)`, mixed from the four nearest pixels; past its edge
/// is empty.
fn sample(mask: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    if !(x > -1.0 && y > -1.0 && x < pixel(width) && y < pixel(height)) {
        return 0.0;
    }
    let (left, top) = (x.floor(), y.floor());
    let (toward_x, toward_y) = (x - left, y - top);
    #[allow(clippy::cast_possible_truncation)]
    let (left, top) = (left as isize, top as isize);
    let at = |x: isize, y: isize| match (usize::try_from(x), usize::try_from(y)) {
        (Ok(x), Ok(y)) if x < width && y < height => mask[y * width + x],
        _ => 0.0,
    };
    let above = at(left, top) + (at(left + 1, top) - at(left, top)) * toward_x;
    let below = at(left, top + 1) + (at(left + 1, top + 1) - at(left, top + 1)) * toward_x;
    above + (below - above) * toward_y
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

/// Rounded boxes with one pixel of soft edge, as one shape: where two
/// meet, a pixel takes the most either covers it.
fn paint_boxes(rgba: &mut [u8], width: usize, height: usize, areas: &[Area], back: &Background) {
    let Some(whole) = width.checked_mul(height) else {
        return;
    };
    let mut covers = vec![0.0_f32; whole];
    for area in areas {
        box_cover(&mut covers, width, height, *area, back.round);
    }
    over_each(rgba, &covers, |_| back.color, |cover| cover);
}

/// Raise each pixel of `covers` to how much of it a box over `area`, its
/// corners rounded by `round` of its shorter half, covers.
fn box_cover(covers: &mut [f32], width: usize, height: usize, area: Area, round: f32) {
    let half_w = (area.right - area.left) / 2.0;
    let half_h = (area.bottom - area.top) / 2.0;
    if !(half_w > 0.0 && half_h > 0.0) {
        return;
    }
    let center_x = area.left + half_w;
    let center_y = area.top + half_h;
    let radius = round * half_w.min(half_h);
    // Only the pixels the box can reach, a pixel of soft edge around it.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let reach = |low: f32, high: f32, limit: usize| {
        let low = (low - 1.0).floor().max(0.0) as usize;
        let high = ((high + 1.0).ceil().max(0.0) as usize).min(limit);
        low..high
    };
    for y in reach(area.top, area.bottom, height) {
        for x in reach(area.left, area.right, width) {
            let px = pixel(x) + 0.5 - center_x;
            let py = pixel(y) + 0.5 - center_y;
            let qx = px.abs() - (half_w - radius);
            let qy = py.abs() - (half_h - radius);
            let outside = qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0) - radius;
            let cover = (0.5 - outside).clamp(0.0, 1.0);
            if let Some(slot) = covers.get_mut(y * width + x) {
                *slot = slot.max(cover);
            }
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

/// `value` ems either way, at most four, in pixels at `font_size`.
fn signed_em(value: f64, font_size: f32) -> f32 {
    if value < 0.0 {
        -em(-value, font_size)
    } else {
        em(value, font_size)
    }
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
