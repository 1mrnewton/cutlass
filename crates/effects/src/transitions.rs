//! The transitions Cutlass draws itself, with no package: twenty ways to go
//! from the outgoing picture to the incoming one across a cut.
//!
//! A transition material names one with `effect_id` `cutlass-<key>`. A
//! material from before that, with no package id, is known by its name:
//! "Dissolve", and "Dip" for dip to black. [`mix`] draws one moment of a
//! transition on the CPU from two same-size RGBA pictures; the preview and
//! the export both show that picture.

use cutlass_model::MaterialTransition;

/// The prefix of a built-in transition's `effect_id`.
pub const BUILT_IN_PREFIX: &str = "cutlass-";

/// A way across a cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransitionKind {
    Dissolve,
    DipToBlack,
    DipToWhite,
    WipeLeft,
    WipeRight,
    WipeUp,
    WipeDown,
    SlideLeft,
    SlideRight,
    SlideUp,
    SlideDown,
    PushLeft,
    PushRight,
    PushUp,
    PushDown,
    ZoomIn,
    ZoomOut,
    Spin,
    Blur,
    Glitch,
}

/// Which way a wipe, slide or push moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Way {
    Left,
    Right,
    Up,
    Down,
}

impl TransitionKind {
    pub const ALL: [Self; 20] = [
        Self::Dissolve,
        Self::DipToBlack,
        Self::DipToWhite,
        Self::WipeLeft,
        Self::WipeRight,
        Self::WipeUp,
        Self::WipeDown,
        Self::SlideLeft,
        Self::SlideRight,
        Self::SlideUp,
        Self::SlideDown,
        Self::PushLeft,
        Self::PushRight,
        Self::PushUp,
        Self::PushDown,
        Self::ZoomIn,
        Self::ZoomOut,
        Self::Spin,
        Self::Blur,
        Self::Glitch,
    ];

    /// The stable key, as stored after [`BUILT_IN_PREFIX`].
    #[must_use]
    pub fn key(self) -> &'static str {
        match self {
            Self::Dissolve => "dissolve",
            Self::DipToBlack => "dip",
            Self::DipToWhite => "dip-white",
            Self::WipeLeft => "wipe-left",
            Self::WipeRight => "wipe-right",
            Self::WipeUp => "wipe-up",
            Self::WipeDown => "wipe-down",
            Self::SlideLeft => "slide-left",
            Self::SlideRight => "slide-right",
            Self::SlideUp => "slide-up",
            Self::SlideDown => "slide-down",
            Self::PushLeft => "push-left",
            Self::PushRight => "push-right",
            Self::PushUp => "push-up",
            Self::PushDown => "push-down",
            Self::ZoomIn => "zoom-in",
            Self::ZoomOut => "zoom-out",
            Self::Spin => "spin",
            Self::Blur => "blur",
            Self::Glitch => "glitch",
        }
    }

    /// The name on the shelf.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Dissolve => "Dissolve",
            Self::DipToBlack => "Dip to black",
            Self::DipToWhite => "Dip to white",
            Self::WipeLeft => "Wipe left",
            Self::WipeRight => "Wipe right",
            Self::WipeUp => "Wipe up",
            Self::WipeDown => "Wipe down",
            Self::SlideLeft => "Slide left",
            Self::SlideRight => "Slide right",
            Self::SlideUp => "Slide up",
            Self::SlideDown => "Slide down",
            Self::PushLeft => "Push left",
            Self::PushRight => "Push right",
            Self::PushUp => "Push up",
            Self::PushDown => "Push down",
            Self::ZoomIn => "Zoom in",
            Self::ZoomOut => "Zoom out",
            Self::Spin => "Spin",
            Self::Blur => "Blur",
            Self::Glitch => "Glitch",
        }
    }

    /// The `effect_id` a material stores for this transition.
    #[must_use]
    pub fn effect_id(self) -> String {
        format!("{BUILT_IN_PREFIX}{}", self.key())
    }

    /// The built-in transition `material` names, if any.
    #[must_use]
    pub fn of(material: &MaterialTransition) -> Option<Self> {
        if let Some(key) = material.effect_id.strip_prefix(BUILT_IN_PREFIX) {
            return Self::ALL.into_iter().find(|kind| kind.key() == key);
        }
        if !material.effect_id.is_empty() || !material.resource_id.is_empty() {
            return None;
        }
        match material.name.as_str() {
            "Dissolve" => Some(Self::Dissolve),
            "Dip" => Some(Self::DipToBlack),
            name => Self::ALL.into_iter().find(|kind| kind.label() == name),
        }
    }

    fn way(self) -> Way {
        match self {
            Self::WipeRight | Self::SlideRight | Self::PushRight => Way::Right,
            Self::WipeUp | Self::SlideUp | Self::PushUp => Way::Up,
            Self::WipeDown | Self::SlideDown | Self::PushDown => Way::Down,
            _ => Way::Left,
        }
    }
}

/// Smooth start and end for moving pictures.
fn ease(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: [u8; 4], b: [u8; 4], t: f32) -> [u8; 4] {
    let mut out = [0; 4];
    for channel in 0..4 {
        let value = f32::from(a[channel]) + (f32::from(b[channel]) - f32::from(a[channel])) * t;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            out[channel] = value.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// One picture to read from.
struct Picture<'a> {
    rgba: &'a [u8],
    width: usize,
    height: usize,
}

impl Picture<'_> {
    fn at(&self, x: usize, y: usize) -> [u8; 4] {
        let index = (y * self.width + x) * 4;
        [
            self.rgba[index],
            self.rgba[index + 1],
            self.rgba[index + 2],
            self.rgba[index + 3],
        ]
    }

    /// The pixel under `(x, y)`, in pixels, or clear outside the picture.
    fn sample(&self, x: f32, y: f32) -> [u8; 4] {
        if x < 0.0 || y < 0.0 {
            return [0; 4];
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let (column, row) = (x as usize, y as usize);
        if column >= self.width || row >= self.height {
            return [0; 4];
        }
        self.at(column, row)
    }

    /// The pixel `(x, y)` shows when the picture is scaled by `scale` and
    /// turned by `angle` radians about its middle.
    fn turned(&self, x: f32, y: f32, scale: f32, angle: f32) -> [u8; 4] {
        if scale <= 0.0 {
            return [0; 4];
        }
        #[allow(clippy::cast_precision_loss)]
        let (cx, cy) = (self.width as f32 / 2.0, self.height as f32 / 2.0);
        let (dx, dy) = ((x - cx) / scale, (y - cy) / scale);
        let (sin, cos) = (-angle).sin_cos();
        self.sample(cx + dx * cos - dy * sin, cy + dx * sin + dy * cos)
    }
}

/// A box blur of `radius` pixels, across then down.
fn blurred(picture: &Picture<'_>, radius: usize) -> Vec<u8> {
    let (width, height) = (picture.width, picture.height);
    if radius == 0 {
        return picture.rgba.to_vec();
    }
    let pass = |source: &[u8], across: bool| {
        let mut out = vec![0; source.len()];
        let (lines, length) = if across {
            (height, width)
        } else {
            (width, height)
        };
        for line in 0..lines {
            let index = |at: usize| {
                if across {
                    (line * width + at) * 4
                } else {
                    (at * width + line) * 4
                }
            };
            for at in 0..length {
                let low = at.saturating_sub(radius);
                let high = (at + radius).min(length - 1);
                let mut sum = [0_u32; 4];
                for near in low..=high {
                    let pixel = index(near);
                    for channel in 0..4 {
                        sum[channel] += u32::from(source[pixel + channel]);
                    }
                }
                #[allow(clippy::cast_possible_truncation)]
                let count = (high - low + 1) as u32;
                let pixel = index(at);
                for channel in 0..4 {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        out[pixel + channel] = ((sum[channel] + count / 2) / count) as u8;
                    }
                }
            }
        }
        out
    };
    let across = pass(picture.rgba, true);
    pass(&across, false)
}

/// A number from 0 to 1 that looks random for `seed`.
fn noise(seed: u32) -> f32 {
    let mut value = seed.wrapping_mul(0x9E37_79B9) ^ 0x85EB_CA6B;
    value ^= value >> 15;
    value = value.wrapping_mul(0x2C1B_3C6D);
    value ^= value >> 12;
    #[allow(clippy::cast_precision_loss)]
    let unit = (value & 0xFFFF) as f32 / 65535.0;
    unit
}

/// One moment `t`, 0 to 1, of `kind` from `from` to `to`, both `width` by
/// `height` RGBA. `None` when the sizes do not match.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn mix(
    kind: TransitionKind,
    from: &[u8],
    to: &[u8],
    width: u32,
    height: u32,
    t: f32,
) -> Option<Vec<u8>> {
    let (w, h) = (usize::try_from(width).ok()?, usize::try_from(height).ok()?);
    let size = w.checked_mul(h)?.checked_mul(4)?;
    if from.len() != size || to.len() != size || w == 0 || h == 0 {
        return None;
    }
    let t = if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let from = Picture {
        rgba: from,
        width: w,
        height: h,
    };
    let to = Picture {
        rgba: to,
        width: w,
        height: h,
    };
    #[allow(clippy::cast_precision_loss)]
    let (fw, fh) = (w as f32, h as f32);
    let moved = ease(t);
    let mut out = vec![0; size];
    let blur = (kind == TransitionKind::Blur).then(|| {
        let strength = 1.0 - (2.0 * t - 1.0).abs();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let radius = (strength * fw.max(fh) / 30.0).round() as usize;
        (blurred(&from, radius), blurred(&to, radius))
    });
    for y in 0..h {
        for x in 0..w {
            #[allow(clippy::cast_precision_loss)]
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let pixel = match kind {
                TransitionKind::Dissolve => lerp(from.at(x, y), to.at(x, y), t),
                TransitionKind::DipToBlack | TransitionKind::DipToWhite => {
                    let shade = if kind == TransitionKind::DipToBlack {
                        [0, 0, 0, 255]
                    } else {
                        [255, 255, 255, 255]
                    };
                    if t < 0.5 {
                        lerp(from.at(x, y), shade, t * 2.0)
                    } else {
                        lerp(shade, to.at(x, y), t * 2.0 - 1.0)
                    }
                }
                TransitionKind::WipeLeft
                | TransitionKind::WipeRight
                | TransitionKind::WipeUp
                | TransitionKind::WipeDown => {
                    // How far along the wipe's path this pixel is, 0 to 1.
                    let along = match kind.way() {
                        Way::Left => 1.0 - px / fw,
                        Way::Right => px / fw,
                        Way::Up => 1.0 - py / fh,
                        Way::Down => py / fh,
                    };
                    let edge = 0.02;
                    let reach = t * (1.0 + edge);
                    let shown = ((reach - along) / edge).clamp(0.0, 1.0);
                    lerp(from.at(x, y), to.at(x, y), shown)
                }
                TransitionKind::SlideLeft
                | TransitionKind::SlideRight
                | TransitionKind::SlideUp
                | TransitionKind::SlideDown
                | TransitionKind::PushLeft
                | TransitionKind::PushRight
                | TransitionKind::PushUp
                | TransitionKind::PushDown => {
                    let push = matches!(
                        kind,
                        TransitionKind::PushLeft
                            | TransitionKind::PushRight
                            | TransitionKind::PushUp
                            | TransitionKind::PushDown
                    );
                    // The incoming picture's offset from its place.
                    let (ox, oy) = match kind.way() {
                        Way::Left => ((1.0 - moved) * fw, 0.0),
                        Way::Right => (-(1.0 - moved) * fw, 0.0),
                        Way::Up => (0.0, (1.0 - moved) * fh),
                        Way::Down => (0.0, -(1.0 - moved) * fh),
                    };
                    let incoming = to.sample(px - ox, py - oy);
                    let (fx, fy) = match kind.way() {
                        Way::Left => (ox - fw, 0.0),
                        Way::Right => (ox + fw, 0.0),
                        Way::Up => (0.0, oy - fh),
                        Way::Down => (0.0, oy + fh),
                    };
                    let inside = (0.0..fw).contains(&(px - ox)) && (0.0..fh).contains(&(py - oy));
                    if inside {
                        incoming
                    } else if push {
                        from.sample(px - fx, py - fy)
                    } else {
                        from.at(x, y)
                    }
                }
                TransitionKind::ZoomIn => {
                    let away = from.turned(px, py, 1.0 + moved, 0.0);
                    lerp(away, to.at(x, y), t)
                }
                TransitionKind::ZoomOut => {
                    let near = to.turned(px, py, 2.0 - moved, 0.0);
                    lerp(from.at(x, y), near, t)
                }
                TransitionKind::Spin => {
                    let turn = std::f32::consts::PI;
                    if t < 0.5 {
                        let half = ease(t * 2.0);
                        from.turned(px, py, 1.0 - half, half * turn)
                    } else {
                        let half = ease(t * 2.0 - 1.0);
                        to.turned(px, py, half, (half - 1.0) * turn)
                    }
                }
                TransitionKind::Blur => {
                    let (soft_from, soft_to) = blur.as_ref()?;
                    let index = (y * w + x) * 4;
                    let a = [
                        soft_from[index],
                        soft_from[index + 1],
                        soft_from[index + 2],
                        soft_from[index + 3],
                    ];
                    let b = [
                        soft_to[index],
                        soft_to[index + 1],
                        soft_to[index + 2],
                        soft_to[index + 3],
                    ];
                    lerp(a, b, ease(((t - 0.3) / 0.4).clamp(0.0, 1.0)))
                }
                TransitionKind::Glitch => {
                    let strength = 1.0 - (2.0 * t - 1.0).abs();
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let band = (y * 16 / h) as u32;
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let beat = (t * 12.0) as u32;
                    let jitter = (noise(band * 31 + beat) - 0.5) * strength * fw * 0.2;
                    let source = if noise(band * 7 + beat * 13) < t {
                        &to
                    } else {
                        &from
                    };
                    let split = strength * fw * 0.02;
                    let red = source.sample(px - jitter - split, py)[0];
                    let mut pixel = source.sample(px - jitter, py);
                    let blue = source.sample(px - jitter + split, py)[2];
                    pixel[0] = red;
                    pixel[2] = blue;
                    if pixel[3] == 0 {
                        source.at(x, y)
                    } else {
                        pixel
                    }
                }
            };
            let index = (y * w + x) * 4;
            out[index..index + 4].copy_from_slice(&pixel);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use cutlass_model::MaterialTransition;

    use super::*;

    const W: u32 = 16;
    const H: u32 = 8;

    fn solid(color: [u8; 4]) -> Vec<u8> {
        color.repeat((W * H) as usize)
    }

    fn at(rgba: &[u8], x: u32, y: u32) -> [u8; 4] {
        let index = ((y * W + x) * 4) as usize;
        [
            rgba[index],
            rgba[index + 1],
            rgba[index + 2],
            rgba[index + 3],
        ]
    }

    const RED: [u8; 4] = [255, 0, 0, 255];
    const BLUE: [u8; 4] = [0, 0, 255, 255];

    #[test]
    fn every_transition_starts_on_the_outgoing_and_ends_on_the_incoming() {
        let (from, to) = (solid(RED), solid(BLUE));
        for kind in TransitionKind::ALL {
            let start = mix(kind, &from, &to, W, H, 0.0).unwrap();
            let end = mix(kind, &from, &to, W, H, 1.0).unwrap();
            assert_eq!(at(&start, 8, 4), RED, "{kind:?} start");
            assert_eq!(at(&end, 8, 4), BLUE, "{kind:?} end");
            let middle = mix(kind, &from, &to, W, H, 0.5).unwrap();
            assert_eq!(middle.len(), from.len());
        }
        assert!(mix(TransitionKind::Dissolve, &from, &to[4..], W, H, 0.5).is_none());
    }

    #[test]
    fn each_moves_the_way_its_name_says() {
        let (from, to) = (solid(RED), solid(BLUE));
        let quarter = |kind| mix(kind, &from, &to, W, H, 0.3).unwrap();
        // A wipe left shows the incoming at the right first.
        let wipe = quarter(TransitionKind::WipeLeft);
        assert_eq!(at(&wipe, 15, 4), BLUE);
        assert_eq!(at(&wipe, 0, 4), RED);
        let wipe = quarter(TransitionKind::WipeDown);
        assert_eq!(at(&wipe, 8, 0), BLUE);
        assert_eq!(at(&wipe, 8, 7), RED);
        // A slide right comes in from the left.
        let slide = quarter(TransitionKind::SlideRight);
        assert_eq!(at(&slide, 0, 4), BLUE);
        assert_eq!(at(&slide, 15, 4), RED);
        // A dip goes through black in the middle.
        let dip = mix(TransitionKind::DipToBlack, &from, &to, W, H, 0.5).unwrap();
        assert_eq!(at(&dip, 8, 4), [0, 0, 0, 255]);
        let dip = mix(TransitionKind::DipToWhite, &from, &to, W, H, 0.5).unwrap();
        assert_eq!(at(&dip, 8, 4), [255, 255, 255, 255]);
        // A push moves the outgoing picture out ahead of the incoming one.
        let mut left_half = solid(RED);
        for y in 0..H {
            for x in 0..W / 2 {
                let index = ((y * W + x) * 4) as usize;
                left_half[index..index + 4].copy_from_slice(&[0, 255, 0, 255]);
            }
        }
        let push = mix(TransitionKind::PushLeft, &left_half, &to, W, H, 0.5).unwrap();
        assert_eq!(at(&push, 0, 4), RED, "the outgoing right half moved left");
        assert_eq!(at(&push, 15, 4), BLUE);
        // Spin shows the canvas around a shrinking picture.
        let spin = mix(TransitionKind::Spin, &from, &to, W, H, 0.45).unwrap();
        assert_eq!(at(&spin, 0, 0), [0; 4]);
    }

    #[test]
    fn a_material_names_its_transition_by_effect_id_or_old_name() {
        let named = |effect_id: &str, name: &str| MaterialTransition {
            effect_id: effect_id.to_owned(),
            name: name.to_owned(),
            ..MaterialTransition::default()
        };
        for kind in TransitionKind::ALL {
            assert_eq!(
                TransitionKind::of(&named(&kind.effect_id(), "")),
                Some(kind)
            );
        }
        assert_eq!(
            TransitionKind::of(&named("", "Dip")),
            Some(TransitionKind::DipToBlack)
        );
        assert_eq!(
            TransitionKind::of(&named("", "Dissolve")),
            Some(TransitionKind::Dissolve)
        );
        assert_eq!(
            TransitionKind::of(&named("transition-fade", "Dissolve")),
            None
        );
        assert_eq!(TransitionKind::of(&named("cutlass-nope", "")), None);
    }
}
