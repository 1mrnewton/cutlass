//! Remove BG: a clip cut down to its subject by a matte video, its edge
//! cleaned and feathered, and an outline drawn around what is kept.
//!
//! A clip's matting is a `matting` record in the draft's `common_mask` pool
//! that the clip lists among its extra materials, beside any shape mask,
//! selection and chroma key. It holds only settings. The matte itself, a
//! grey picture a frame (white kept, black removed, soft between), is made
//! once by a model and kept in the project folder, where eval finds it by
//! the clip's file, stretch and model. Until it is there the clip draws
//! uncut.
//!
//! - `mode` is `auto`, where a model finds the subject on its own, or
//!   `custom`, where the user's picks say what to keep and a model follows
//!   them through the clip.
//! - `feather` softens the edge and `clean_up` pulls it in, each 0 to 1.
//!   They are applied when the frame is drawn, so they cost no new matte.
//! - `stroke` is an outline around what is kept, also drawn live.
//! - `picks` are Custom removal's clicks and brush strokes, each on one
//!   frame of the clip's file. They are in unit coordinates of that frame
//!   (0 to 1, x right, y down), so the frame's size never leaks out.
//!
//! Every length is a share of the picture's short side, so preview at any
//! size and export look the same.

use cutlass_model::{Draft, MaterialRef, Micros, Segment};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The material `type` of a clip's matting.
pub const MATTING: &str = "matting";

/// The most picks one clip keeps.
pub const MAX_PICKS: usize = 64;
/// The most clicks one pick keeps.
pub const MAX_PICK_CLICKS: usize = 32;
/// The most brush strokes one pick keeps.
pub const MAX_BRUSHES: usize = 256;
/// The most points one brush stroke keeps.
pub const MAX_BRUSH_POINTS: usize = 4096;

/// The widest feather, as a share of the short side.
const FEATHER_REACH: f64 = 0.02;
/// The most an edge is pulled in, as a share of the short side.
const CLEAN_UP_REACH: f64 = 0.015;
/// The widest outline, as a share of the short side.
const STROKE_REACH: f64 = 0.04;
/// The widest gap between the subject and its outline.
const SPACING_REACH: f64 = 0.03;

/// How the subject is found.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MattingMode {
    /// A model finds the person or object on its own.
    #[default]
    Auto,
    /// The user's picks say what to keep, followed through the clip.
    Custom,
}

/// How the outline looks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrokeStyle {
    /// A white band with a soft shadow, like a paper cut-out.
    #[default]
    Paper,
    /// Light that fades away from the edge.
    Glow,
    /// A band whose width wanders, as if drawn by hand.
    HandDrawn,
    /// A thin line a little away from the edge.
    Straight,
    /// A plain band.
    Solid,
    /// A flat copy of the subject's shape, moved down and right.
    Offset,
    /// Dots around the edge.
    Dotted,
}

impl StrokeStyle {
    /// Every style, in the order the page shows them.
    pub const ALL: [Self; 7] = [
        Self::Paper,
        Self::Glow,
        Self::HandDrawn,
        Self::Straight,
        Self::Solid,
        Self::Offset,
        Self::Dotted,
    ];

    /// The name the page shows.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Paper => "Paper",
            Self::Glow => "Glow",
            Self::HandDrawn => "Hand-drawn",
            Self::Straight => "Straight",
            Self::Solid => "Solid",
            Self::Offset => "Offset",
            Self::Dotted => "Dotted",
        }
    }

    /// The colour the style starts with, `#rrggbb`.
    #[must_use]
    pub const fn default_color(self) -> &'static str {
        match self {
            Self::Paper | Self::HandDrawn | Self::Straight | Self::Dotted => "#ffffff",
            Self::Glow => "#7fd3ff",
            Self::Solid => "#ffd23f",
            Self::Offset => "#ff5a8c",
        }
    }
}

/// An outline around what is kept.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stroke {
    pub style: StrokeStyle,
    /// `#rrggbb`.
    pub color: String,
    /// How wide, 0 to 1.
    pub size: f64,
    /// The gap between the subject and the outline, 0 to 1.
    pub spacing: f64,
    /// How much grain the outline has, 0 to 1.
    pub texture: f64,
}

impl Stroke {
    /// `style` at the values the page starts it with.
    #[must_use]
    pub fn new(style: StrokeStyle) -> Self {
        Self {
            style,
            color: style.default_color().to_owned(),
            size: 0.35,
            spacing: 0.0,
            texture: 0.0,
        }
    }

    fn is_valid(&self) -> bool {
        crate::parse_hex(&self.color).is_some()
            && [self.size, self.spacing, self.texture]
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
    }
}

/// A click on a pick's frame, in unit coordinates of that frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PickClick {
    pub x: f64,
    pub y: f64,
    /// Add what is under it, or take it out.
    pub keep: bool,
}

/// A brush stroke on a pick's frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Brush {
    /// The path, in unit coordinates of the frame.
    pub points: Vec<[f64; 2]>,
    /// The brush's radius, as a share of the frame's short side.
    pub radius: f64,
    /// Paint the subject in, or erase it.
    pub keep: bool,
    /// A smart brush lets the model find the edge of what it touches; a
    /// plain one paints exactly where it goes.
    pub smart: bool,
}

/// What the user said to keep on one frame of the clip's file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pick {
    /// The source time of the frame.
    pub frame: Micros,
    #[serde(default)]
    pub clicks: Vec<PickClick>,
    #[serde(default)]
    pub brushes: Vec<Brush>,
}

impl Pick {
    fn is_valid(&self) -> bool {
        let unit = |value: f64| (0.0..=1.0).contains(&value);
        self.frame.0 >= 0
            && (!self.clicks.is_empty() || !self.brushes.is_empty())
            && self.clicks.len() <= MAX_PICK_CLICKS
            && self.brushes.len() <= MAX_BRUSHES
            && self.clicks.iter().all(|c| unit(c.x) && unit(c.y))
            && self.brushes.iter().all(|brush| {
                !brush.points.is_empty()
                    && brush.points.len() <= MAX_BRUSH_POINTS
                    && brush.radius > 0.0
                    && brush.radius <= 0.5
                    && brush.points.iter().all(|[x, y]| unit(*x) && unit(*y))
            })
    }
}

/// One clip's Remove BG settings.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Matting {
    pub mode: MattingMode,
    #[serde(default)]
    pub feather: f64,
    #[serde(default)]
    pub clean_up: f64,
    #[serde(default)]
    pub stroke: Option<Stroke>,
    /// Custom removal's picks, in source time order.
    #[serde(default)]
    pub picks: Vec<Pick>,
}

impl Matting {
    /// Auto removal at its starting values.
    #[must_use]
    pub fn auto() -> Self {
        Self::default()
    }

    /// Every value in range, every pick whole, and picks only in Custom.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        (0.0..=1.0).contains(&self.feather)
            && (0.0..=1.0).contains(&self.clean_up)
            && self.stroke.as_ref().is_none_or(Stroke::is_valid)
            && self.picks.len() <= MAX_PICKS
            && self.picks.iter().all(Pick::is_valid)
            && (self.mode == MattingMode::Custom || self.picks.is_empty())
            && self
                .picks
                .windows(2)
                .all(|pair| pair[0].frame < pair[1].frame)
    }

    /// Whether a matte can be made: Auto always, Custom once it has a pick.
    #[must_use]
    pub fn ready(&self) -> bool {
        self.mode == MattingMode::Auto || !self.picks.is_empty()
    }

    /// The settings as the record's fields.
    #[must_use]
    pub fn to_fields(&self) -> Map<String, Value> {
        match serde_json::to_value(self) {
            Ok(Value::Object(fields)) => fields,
            _ => Map::new(),
        }
    }

    /// Settings read from a record's fields, when they are whole and valid.
    #[must_use]
    pub fn from_fields(fields: &Map<String, Value>) -> Option<Self> {
        let matting: Self = serde_json::from_value(Value::Object(fields.clone())).ok()?;
        matting.is_valid().then_some(matting)
    }

    /// What the picks are worth to a matte: a matte made from other picks,
    /// or in the other mode, is not this one's. Auto has no picks.
    #[must_use]
    pub fn picks_key(&self) -> String {
        match self.mode {
            MattingMode::Auto => String::new(),
            MattingMode::Custom => {
                let text = serde_json::to_string(&self.picks).unwrap_or_default();
                format!("{:016x}", fnv1a(text.as_bytes()))
            }
        }
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// The matting `segment` lists, when it has a whole one.
#[must_use]
pub fn segment_matting(draft: &Draft, segment: &Segment) -> Option<Matting> {
    segment.extra_material_refs.iter().find_map(|id| {
        let Some(MaterialRef::Other(raw)) = draft.materials.get(id.as_str()) else {
            return None;
        };
        (raw.kind == MATTING).then(|| Matting::from_fields(&raw.fields))?
    })
}

/// A length that is `share` of the short side of `width` by `height`, in
/// pixels.
fn pixels(share: f64, width: u32, height: u32) -> f64 {
    share * f64::from(width.min(height))
}

/// The matte as eval draws it: `alpha`, `width` by `height`, pulled in by
/// `clean_up` and softened by `feather`.
pub fn refine(alpha: &mut [u8], width: u32, height: u32, clean_up: f64, feather: f64) {
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 || alpha.len() != w * h {
        return;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let erode = pixels(clean_up * CLEAN_UP_REACH, width, height).round() as usize;
    if erode > 0 {
        min_filter(alpha, w, h, erode);
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let blur = pixels(feather * FEATHER_REACH, width, height).round() as usize;
    if blur > 0 {
        // Three box passes come close to a Gaussian.
        let pass = (blur / 2).max(1);
        for _ in 0..3 {
            box_blur(alpha, w, h, pass);
        }
    }
}

/// Cut premultiplied RGBA down to `alpha`, one byte a pixel at its size.
pub fn cut(rgba: &mut [u8], alpha: &[u8]) {
    for (pixel, keep) in rgba.as_chunks_mut::<4>().0.iter_mut().zip(alpha) {
        if *keep == 255 {
            continue;
        }
        let keep = u16::from(*keep);
        for channel in pixel.iter_mut() {
            #[allow(clippy::cast_possible_truncation)]
            let scaled = ((u16::from(*channel) * keep + 127) / 255) as u8;
            *channel = scaled;
        }
    }
}

/// The outline of `alpha`, `width` by `height`, as premultiplied RGBA of
/// the same size, to draw under the cut picture.
#[must_use]
pub fn stroke_layer(alpha: &[u8], width: u32, height: u32, stroke: &Stroke) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let mut out = vec![0_u8; w * h * 4];
    if w == 0 || h == 0 || alpha.len() != w * h {
        return out;
    }
    let Some(color) = crate::parse_hex(&stroke.color) else {
        return out;
    };
    let size = pixels(stroke.size * STROKE_REACH, width, height).max(1.0);
    let gap = pixels(stroke.spacing * SPACING_REACH, width, height);
    let distance = if stroke.style == StrokeStyle::Offset {
        Vec::new()
    } else {
        outside_distance(alpha, w, h)
    };
    let offset = (size * 0.7).round();
    for y in 0..h {
        for x in 0..w {
            let index = y * w + x;
            let cover = match stroke.style {
                StrokeStyle::Offset => {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let (dx, dy) = ((offset + gap) as usize, (offset + gap) as usize);
                    if x >= dx && y >= dy {
                        f64::from(alpha[(y - dy) * w + (x - dx)]) / 255.0
                    } else {
                        0.0
                    }
                }
                style => {
                    let d = distance[index] - gap;
                    if distance[index] <= 0.0 {
                        0.0
                    } else {
                        style_cover(style, d, size, x, y)
                    }
                }
            };
            if cover <= 0.0 {
                continue;
            }
            let grain = 1.0 - stroke.texture * 0.35 * noise(x, y, 2.0);
            let pixel = &mut out[index * 4..index * 4 + 4];
            for channel in 0..3 {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let byte = (f64::from(color[channel]) * grain * cover * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                pixel[channel] = byte;
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let byte = (cover * 255.0).round().clamp(0.0, 255.0) as u8;
            pixel[3] = byte;
        }
    }
    out
}

/// How much of the outline covers a pixel `d` pixels past the gap, 0 to 1.
fn style_cover(style: StrokeStyle, d: f64, size: f64, x: usize, y: usize) -> f64 {
    let band = |width: f64| (width - d + 1.0).clamp(0.0, 1.0) * (d + 0.5).clamp(0.0, 1.0);
    match style {
        StrokeStyle::Solid => band(size),
        StrokeStyle::Paper => {
            let paper = band(size);
            let shadow_reach = size * 0.6;
            let shadow = if d > size && shadow_reach > 0.0 {
                0.35 * (1.0 - (d - size) / shadow_reach).clamp(0.0, 1.0)
            } else {
                0.0
            };
            paper.max(shadow)
        }
        StrokeStyle::Glow => {
            let fall = (1.0 - d / (size * 2.0)).clamp(0.0, 1.0);
            fall * fall
        }
        StrokeStyle::HandDrawn => band(size * (1.0 + 0.35 * (noise(x, y, size * 3.0) - 0.5) * 2.0)),
        StrokeStyle::Straight => {
            let line = (size * 0.15).max(2.0);
            (1.0 - (d - size).abs() / (line / 2.0)).clamp(0.0, 1.0)
        }
        StrokeStyle::Dotted => {
            if d > size {
                return 0.0;
            }
            let period = size * 1.6;
            let (cx, cy) = (
                (x as f64 / period).round() * period,
                (y as f64 / period).round() * period,
            );
            let r = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
            (size * 0.45 - r + 0.5).clamp(0.0, 1.0)
        }
        StrokeStyle::Offset => 0.0,
    }
}

/// Smooth value noise 0 to 1 with features about `period` pixels apart.
fn noise(x: usize, y: usize, period: f64) -> f64 {
    let period = period.max(1.0);
    let (fx, fy) = (x as f64 / period, y as f64 / period);
    let (x0, y0) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - x0, fy - y0);
    let lattice = |i: f64, j: f64| {
        #[allow(clippy::cast_possible_truncation)]
        let seed = (i as i64).wrapping_mul(73_856_093) ^ (j as i64).wrapping_mul(19_349_663);
        #[allow(clippy::cast_sign_loss)]
        let hashed = fnv1a(&seed.to_le_bytes());
        (hashed % 1024) as f64 / 1023.0
    };
    let ease = |t: f64| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (ease(tx), ease(ty));
    let top = lattice(x0, y0) + (lattice(x0 + 1.0, y0) - lattice(x0, y0)) * sx;
    let bottom = lattice(x0, y0 + 1.0) + (lattice(x0 + 1.0, y0 + 1.0) - lattice(x0, y0 + 1.0)) * sx;
    top + (bottom - top) * sy
}

/// Each pixel's distance in pixels to the nearest kept pixel (alpha at
/// least half), 0 on kept pixels: an exact Euclidean transform, two passes
/// of the lower envelope of parabolas.
#[must_use]
pub fn outside_distance(alpha: &[u8], w: usize, h: usize) -> Vec<f64> {
    const FAR: f64 = 1e20;
    let mut grid: Vec<f64> = alpha
        .iter()
        .map(|a| if *a >= 128 { 0.0 } else { FAR })
        .collect();
    let longest = w.max(h);
    let mut f = vec![0.0; longest];
    let mut d = vec![0.0; longest];
    let mut v = vec![0_usize; longest];
    let mut z = vec![0.0; longest + 1];
    for x in 0..w {
        for y in 0..h {
            f[y] = grid[y * w + x];
        }
        squared_1d(&f[..h], &mut d[..h], &mut v, &mut z);
        for y in 0..h {
            grid[y * w + x] = d[y];
        }
    }
    for y in 0..h {
        f[..w].copy_from_slice(&grid[y * w..(y + 1) * w]);
        squared_1d(&f[..w], &mut d[..w], &mut v, &mut z);
        grid[y * w..(y + 1) * w].copy_from_slice(&d[..w]);
    }
    for value in &mut grid {
        *value = value.sqrt();
    }
    grid
}

/// One axis of the squared distance transform (Felzenszwalb and
/// Huttenlocher).
fn squared_1d(f: &[f64], d: &mut [f64], v: &mut [usize], z: &mut [f64]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    let mut k = 0_usize;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        let qf = q as f64;
        // `f` is finite, so `s` is too, and `z[0]` (minus infinity) stops
        // the walk back before `k` passes 0.
        let mut s;
        loop {
            let p = v[k] as f64;
            s = ((f[q] + qf * qf) - (f[v[k]] + p * p)) / (2.0 * qf - 2.0 * p);
            if s <= z[k] && k > 0 {
                k -= 1;
            } else {
                break;
            }
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    k = 0;
    for (q, out) in d.iter_mut().enumerate() {
        let qf = q as f64;
        while z[k + 1] < qf {
            k += 1;
        }
        let p = v[k] as f64;
        *out = (qf - p) * (qf - p) + f[v[k]];
    }
}

/// The smallest value within `r` pixels along each axis: a square erosion,
/// in time that does not grow with `r`.
fn min_filter(alpha: &mut [u8], w: usize, h: usize, r: usize) {
    let mut line = Vec::with_capacity(w.max(h));
    let mut out = Vec::with_capacity(w.max(h));
    for y in 0..h {
        line.clear();
        line.extend_from_slice(&alpha[y * w..(y + 1) * w]);
        running_min(&line, r, &mut out);
        alpha[y * w..(y + 1) * w].copy_from_slice(&out);
    }
    for x in 0..w {
        line.clear();
        line.extend((0..h).map(|y| alpha[y * w + x]));
        running_min(&line, r, &mut out);
        for (y, value) in out.iter().enumerate() {
            alpha[y * w + x] = *value;
        }
    }
}

/// `out[i]` is the least of `line[i - r..=i + r]`, clamped at the ends
/// (van Herk and Gil-Werman: a forward and a backward running minimum over
/// blocks of `2r + 1`).
fn running_min(line: &[u8], r: usize, out: &mut Vec<u8>) {
    let n = line.len();
    out.clear();
    if n == 0 {
        return;
    }
    let block = 2 * r + 1;
    let mut forward = vec![0_u8; n];
    let mut backward = vec![0_u8; n];
    for i in 0..n {
        forward[i] = if i % block == 0 {
            line[i]
        } else {
            forward[i - 1].min(line[i])
        };
    }
    for i in (0..n).rev() {
        backward[i] = if i == n - 1 || (i + 1) % block == 0 {
            line[i]
        } else {
            backward[i + 1].min(line[i])
        };
    }
    for i in 0..n {
        let lo = i.saturating_sub(r);
        let hi = (i + r).min(n - 1);
        out.push(backward[lo].min(forward[hi]));
    }
}

/// A box blur of radius `r` along each axis, edges held.
fn box_blur(alpha: &mut [u8], w: usize, h: usize, r: usize) {
    let mut line = Vec::with_capacity(w.max(h));
    let mut out = Vec::with_capacity(w.max(h));
    for y in 0..h {
        line.clear();
        line.extend_from_slice(&alpha[y * w..(y + 1) * w]);
        running_mean(&line, r, &mut out);
        alpha[y * w..(y + 1) * w].copy_from_slice(&out);
    }
    for x in 0..w {
        line.clear();
        line.extend((0..h).map(|y| alpha[y * w + x]));
        running_mean(&line, r, &mut out);
        for (y, value) in out.iter().enumerate() {
            alpha[y * w + x] = *value;
        }
    }
}

/// The mean of `line[i - r..=i + r]` with the ends repeated.
fn running_mean(line: &[u8], r: usize, out: &mut Vec<u8>) {
    let n = line.len();
    out.clear();
    if n == 0 {
        return;
    }
    let at = |i: isize| -> u32 {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
        let clamped = i.clamp(0, n as isize - 1) as usize;
        u32::from(line[clamped])
    };
    #[allow(clippy::cast_possible_wrap)]
    let ri = r as isize;
    let mut sum: u32 = (-ri..=ri).map(at).sum();
    #[allow(clippy::cast_possible_truncation)]
    let count = (2 * r + 1) as u32;
    for i in 0..n {
        #[allow(clippy::cast_possible_truncation)]
        out.push(((sum + count / 2) / count) as u8);
        #[allow(clippy::cast_possible_wrap)]
        let ii = i as isize;
        sum = sum + at(ii + ri + 1) - at(ii - ri);
    }
}

#[cfg(test)]
mod tests;
