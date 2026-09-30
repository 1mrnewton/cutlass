//! Preview-frame eyedropper: widget→frame mapping, pixel sampling, and the
//! arm → hover-preview → commit/cancel session used by the color picker.
//!
//! Sampling reads the composited preview RGBA buffer only (no OS screen
//! capture). Mapping reuses [`crate::preview_select::viewport_mapping`] so
//! letterbox / zoom / pan agree with hit-testing.

use slint::Image;

use crate::preview_select::viewport_mapping;

/// Odd neighborhood size for the zoom loupe (center pixel outlined in UI).
pub const LOUPE_SIZE: u32 = 11;

/// Map viewport-element coordinates to a frame pixel, or `None` when the
/// point lands in letterbox bars / outside the canvas.
#[allow(clippy::too_many_arguments)] // mirrors PreviewBackend.sample-preview
pub fn widget_to_frame_pixel(
    x: f32,
    y: f32,
    view_w: f32,
    view_h: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
    canvas_w: f32,
    canvas_h: f32,
    frame_w: u32,
    frame_h: u32,
) -> Option<(u32, u32)> {
    if frame_w == 0
        || frame_h == 0
        || canvas_w <= 0.0
        || canvas_h <= 0.0
        || view_w <= 0.0
        || view_h <= 0.0
    {
        return None;
    }
    let (scale, ox, oy) = viewport_mapping(canvas_w, canvas_h, view_w, view_h, zoom, pan_x, pan_y);
    if scale <= 0.0 {
        return None;
    }
    let cx = (x - ox) / scale;
    let cy = (y - oy) / scale;
    if cx < 0.0 || cy < 0.0 || cx >= canvas_w || cy >= canvas_h {
        return None;
    }
    let fx = ((cx / canvas_w) * frame_w as f32).floor() as i64;
    let fy = ((cy / canvas_h) * frame_h as f32).floor() as i64;
    if fx < 0 || fy < 0 || fx >= i64::from(frame_w) || fy >= i64::from(frame_h) {
        return None;
    }
    Some((fx as u32, fy as u32))
}

/// Exact RGBA of one frame pixel. Returns `None` when out of bounds.
pub fn sample_color(pixels: &[u8], width: u32, height: u32, x: u32, y: u32) -> Option<[u8; 4]> {
    if x >= width || y >= height {
        return None;
    }
    let i = ((y * width + x) * 4) as usize;
    if i + 3 >= pixels.len() {
        return None;
    }
    Some([pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]])
}

/// Build an `LOUPE_SIZE`×`LOUPE_SIZE` RGBA neighborhood around `(cx, cy)`.
/// Offsets that leave the frame clamp to the nearest edge pixel.
pub fn sample_region(pixels: &[u8], width: u32, height: u32, cx: u32, cy: u32) -> Vec<u8> {
    let half = (LOUPE_SIZE / 2) as i32;
    let mut out = vec![0u8; (LOUPE_SIZE * LOUPE_SIZE * 4) as usize];
    if width == 0 || height == 0 || pixels.len() < 4 {
        return out;
    }
    let max_x = width.saturating_sub(1) as i32;
    let max_y = height.saturating_sub(1) as i32;
    for dy in -half..=half {
        for dx in -half..=half {
            let sx = (cx as i32 + dx).clamp(0, max_x) as u32;
            let sy = (cy as i32 + dy).clamp(0, max_y) as u32;
            let src = ((sy * width + sx) * 4) as usize;
            let lx = (dx + half) as u32;
            let ly = (dy + half) as u32;
            let dst = ((ly * LOUPE_SIZE + lx) * 4) as usize;
            if src + 3 < pixels.len() && dst + 3 < out.len() {
                out[dst..dst + 4].copy_from_slice(&pixels[src..src + 4]);
            }
        }
    }
    out
}

/// Pure sample result (converted to the Slint `EyedropperSample` at the wire).
#[derive(Clone, Debug, Default)]
pub struct FrameSample {
    pub hit: bool,
    pub rgba: [u8; 4],
    pub loupe_rgba: Vec<u8>,
    pub label: String,
}

/// Sample the preview frame at viewport coordinates.
#[allow(clippy::too_many_arguments)] // mirrors PreviewBackend.sample-preview
pub fn sample_preview(
    frame: &Image,
    x: f32,
    y: f32,
    view_w: f32,
    view_h: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
    canvas_w: f32,
    canvas_h: f32,
    enable_alpha: bool,
) -> FrameSample {
    let Some(buffer) = frame.to_rgba8() else {
        return FrameSample::default();
    };
    let fw = buffer.width();
    let fh = buffer.height();
    let pixels = buffer.as_bytes();
    let Some((px, py)) = widget_to_frame_pixel(
        x, y, view_w, view_h, zoom, pan_x, pan_y, canvas_w, canvas_h, fw, fh,
    ) else {
        return FrameSample::default();
    };
    let Some([r, g, b, a]) = sample_color(pixels, fw, fh, px, py) else {
        return FrameSample::default();
    };
    let a_out = if enable_alpha { a } else { 0xFF };
    let loupe_rgba = sample_region(pixels, fw, fh, px, py);
    let label = if enable_alpha && a_out != 0xFF {
        format!("#{r:02X}{g:02X}{b:02X}{a_out:02X}")
    } else {
        format!("#{r:02X}{g:02X}{b:02X}")
    };
    FrameSample {
        hit: true,
        rgba: [r, g, b, a_out],
        loupe_rgba,
        label,
    }
}

/// Wire-level eyedropper session: arm → hover previews → one commit, or cancel.
///
/// The Slint overlay drives the live UI via `EyedropperStore` epochs; this
/// type is the pure protocol under unit test (same role as
/// `TransformGestureSession` for inspector preview/commit).
#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
pub struct EyedropperSession {
    armed: bool,
    consumer: u32,
    enable_alpha: bool,
    last: Option<[u8; 4]>,
    previewed: bool,
}

/// Action the UI/consumer should take after a session transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum EyedropperAction {
    Preview { r: u8, g: u8, b: u8, a: u8 },
    Commit { r: u8, g: u8, b: u8, a: u8 },
    Cancel,
}

#[allow(dead_code)]
impl EyedropperSession {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_armed(&self) -> bool {
        self.armed
    }

    pub fn consumer(&self) -> u32 {
        self.consumer
    }

    pub fn enable_alpha(&self) -> bool {
        self.enable_alpha
    }

    /// Arm for `consumer`. Cancels any prior arm first (returns `Cancel` if
    /// the previous session had previewed).
    pub fn arm(&mut self, consumer: u32, enable_alpha: bool) -> Option<EyedropperAction> {
        let prior = if self.armed && self.previewed {
            Some(EyedropperAction::Cancel)
        } else {
            None
        };
        self.armed = true;
        self.consumer = consumer;
        self.enable_alpha = enable_alpha;
        self.last = None;
        self.previewed = false;
        prior
    }

    /// Hover update. `None` color = letterbox / miss (no preview tick).
    pub fn hover(&mut self, color: Option<[u8; 4]>) -> Option<EyedropperAction> {
        if !self.armed {
            return None;
        }
        let mut rgba = color?;
        if !self.enable_alpha {
            rgba[3] = 0xFF;
        }
        self.last = Some(rgba);
        self.previewed = true;
        Some(EyedropperAction::Preview {
            r: rgba[0],
            g: rgba[1],
            b: rgba[2],
            a: rgba[3],
        })
    }

    /// Click commit — exactly one edit when a hit color is available.
    pub fn commit(&mut self) -> Option<EyedropperAction> {
        if !self.armed {
            return None;
        }
        let color = self.last?;
        self.armed = false;
        self.previewed = false;
        self.last = None;
        Some(EyedropperAction::Commit {
            r: color[0],
            g: color[1],
            b: color[2],
            a: color[3],
        })
    }

    /// Escape / right-click / outside — disarm and revert preview if any.
    pub fn cancel(&mut self) -> Option<EyedropperAction> {
        if !self.armed {
            return None;
        }
        let had_preview = self.previewed;
        self.armed = false;
        self.previewed = false;
        self.last = None;
        had_preview.then_some(EyedropperAction::Cancel)
    }
}

#[cfg(test)]
mod tests;
