//! CPU frame compositor: flattens a back-to-front layer stack into one RGBA8
//! image. This is the **reference** compositor — correct and dependency-light,
//! the ground truth a future WGPU path must match. It is deliberately headless
//! (no GPU, window, or engine dependency) so it can be unit-tested against exact
//! pixel values.
//!
//! Layers are drawn in order (index 0 = bottommost) using straight-alpha `over`
//! compositing. Media frames are sampled to the canvas with nearest-neighbor
//! scaling, so the canvas size is independent of any source resolution — the
//! "scale at draw, don't key the cache by size" idea from the engine notes.

mod convert;

pub use convert::yuv_to_rgba;

use cutlass_decode::{DecodedFrame, PixelFormat};

/// An RGBA8 image, row-major, 4 bytes per pixel (`len == width * height * 4`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl RgbaImage {
    /// A fully transparent canvas.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![0; (width as usize) * (height as usize) * 4],
        }
    }

    /// The RGBA value at `(x, y)`; `[0,0,0,0]` if out of bounds.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        if x >= self.width || y >= self.height {
            return [0, 0, 0, 0];
        }
        let i = self.index(x, y);
        [
            self.pixels[i],
            self.pixels[i + 1],
            self.pixels[i + 2],
            self.pixels[i + 3],
        ]
    }

    fn index(&self, x: u32, y: u32) -> usize {
        ((y as usize) * (self.width as usize) + (x as usize)) * 4
    }

    fn blend_over(&mut self, x: u32, y: u32, src: [u8; 4]) {
        let i = self.index(x, y);
        let dst = [
            self.pixels[i],
            self.pixels[i + 1],
            self.pixels[i + 2],
            self.pixels[i + 3],
        ];
        let out = over(dst, src);
        self.pixels[i] = out[0];
        self.pixels[i + 1] = out[1];
        self.pixels[i + 2] = out[2];
        self.pixels[i + 3] = out[3];
    }
}

/// One layer to composite, back-to-front.
#[derive(Debug, Clone, Copy)]
pub enum CompositeLayer<'a> {
    /// A decoded media frame, sampled to the canvas (nearest-neighbor).
    Frame(&'a DecodedFrame),
    /// A solid RGBA fill covering the whole canvas.
    Solid([u8; 4]),
}

/// Flatten `layers` (bottom-to-top) into a single `width`×`height` RGBA image.
pub fn composite(width: u32, height: u32, layers: &[CompositeLayer]) -> RgbaImage {
    let mut canvas = RgbaImage::new(width, height);
    if width == 0 || height == 0 {
        return canvas;
    }

    for layer in layers {
        match layer {
            CompositeLayer::Solid(rgba) => {
                for y in 0..height {
                    for x in 0..width {
                        canvas.blend_over(x, y, *rgba);
                    }
                }
            }
            CompositeLayer::Frame(frame) => draw_frame(&mut canvas, frame),
        }
    }
    canvas
}

/// Sample `frame` across the whole canvas with nearest-neighbor scaling.
fn draw_frame(canvas: &mut RgbaImage, frame: &DecodedFrame) {
    let (fw, fh) = (frame.width, frame.height);
    if fw == 0 || fh == 0 {
        return;
    }
    let (cw, ch) = (canvas.width, canvas.height);
    for y in 0..ch {
        // Map canvas row -> source row (nearest), clamped into the frame.
        let sy = (((y as u64) * (fh as u64)) / (ch as u64)).min((fh - 1) as u64) as u32;
        for x in 0..cw {
            let sx = (((x as u64) * (fw as u64)) / (cw as u64)).min((fw - 1) as u64) as u32;
            let rgba = sample(frame, sx, sy);
            canvas.blend_over(x, y, rgba);
        }
    }
}

/// Read source pixel `(sx, sy)` from `frame` as RGBA8, by pixel format.
fn sample(frame: &DecodedFrame, sx: u32, sy: u32) -> [u8; 4] {
    let (sx, sy) = (sx as usize, sy as usize);
    match frame.format {
        PixelFormat::Yuv420p => {
            let y = plane_byte(frame, 0, sx, sy);
            let (cx, cy) = (sx / 2, sy / 2);
            let u = plane_byte(frame, 1, cx, cy);
            let v = plane_byte(frame, 2, cx, cy);
            yuv_to_rgba(y, u, v)
        }
        PixelFormat::Nv12 => {
            let y = plane_byte(frame, 0, sx, sy);
            let (cx, cy) = (sx / 2, sy / 2);
            // Interleaved chroma: [U, V] pairs, so the U column is 2*cx.
            let u = plane_byte(frame, 1, cx * 2, cy);
            let v = plane_byte(frame, 1, cx * 2 + 1, cy);
            yuv_to_rgba(y, u, v)
        }
        PixelFormat::Rgba8 => {
            let plane = &frame.planes[0];
            let base = sy * plane.stride + sx * 4;
            match plane.data.get(base..base + 4) {
                Some(px) => [px[0], px[1], px[2], px[3]],
                None => [0, 0, 0, 0],
            }
        }
    }
}

/// One byte from plane `p` at column `col`, row `row` (0 if out of bounds).
fn plane_byte(frame: &DecodedFrame, p: usize, col: usize, row: usize) -> u8 {
    let plane = &frame.planes[p];
    plane
        .data
        .get(row * plane.stride + col)
        .copied()
        .unwrap_or(0)
}

/// Straight-alpha `over`: `src` composited atop `dst`.
fn over(dst: [u8; 4], src: [u8; 4]) -> [u8; 4] {
    let sa = src[3] as f32 / 255.0;
    if sa >= 1.0 {
        return src;
    }
    if sa <= 0.0 {
        return dst;
    }
    let da = dst[3] as f32 / 255.0;
    let out_a = sa + da * (1.0 - sa);
    if out_a <= 0.0 {
        return [0, 0, 0, 0];
    }
    let mut out = [0u8; 4];
    for i in 0..3 {
        let cs = src[i] as f32;
        let cd = dst[i] as f32;
        let c = (cs * sa + cd * da * (1.0 - sa)) / out_a;
        out[i] = c.round().clamp(0.0, 255.0) as u8;
    }
    out[3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_decode::Plane;

    fn solid_yuv420p(width: u32, height: u32, y: u8, u: u8, v: u8) -> DecodedFrame {
        let w = width as usize;
        let h = height as usize;
        DecodedFrame {
            width,
            height,
            pts_ticks: 0,
            format: PixelFormat::Yuv420p,
            planes: vec![
                Plane {
                    data: vec![y; w * h],
                    stride: w,
                },
                Plane {
                    data: vec![u; (w / 2) * (h / 2)],
                    stride: w / 2,
                },
                Plane {
                    data: vec![v; (w / 2) * (h / 2)],
                    stride: w / 2,
                },
            ],
        }
    }

    #[test]
    fn solid_fills_canvas() {
        let img = composite(4, 3, &[CompositeLayer::Solid([10, 20, 30, 255])]);
        assert_eq!(img.width, 4);
        assert_eq!(img.height, 3);
        for y in 0..3 {
            for x in 0..4 {
                assert_eq!(img.pixel(x, y), [10, 20, 30, 255]);
            }
        }
    }

    #[test]
    fn opaque_top_layer_covers_bottom() {
        let img = composite(
            2,
            2,
            &[
                CompositeLayer::Solid([255, 0, 0, 255]),
                CompositeLayer::Solid([0, 255, 0, 255]),
            ],
        );
        assert_eq!(img.pixel(0, 0), [0, 255, 0, 255]);
    }

    #[test]
    fn half_alpha_blends_with_bottom() {
        // 50% white over opaque black -> mid grey, opaque.
        let img = composite(
            1,
            1,
            &[
                CompositeLayer::Solid([0, 0, 0, 255]),
                CompositeLayer::Solid([255, 255, 255, 128]),
            ],
        );
        let p = img.pixel(0, 0);
        assert_eq!(p[3], 255);
        assert!((p[0] as i32 - 128).abs() <= 1, "got {p:?}");
        assert_eq!(p[0], p[1]);
        assert_eq!(p[1], p[2]);
    }

    #[test]
    fn yuv_frame_white_fills_canvas() {
        let frame = solid_yuv420p(4, 4, 235, 128, 128);
        let img = composite(4, 4, &[CompositeLayer::Frame(&frame)]);
        assert_eq!(img.pixel(2, 2), [255, 255, 255, 255]);
    }

    #[test]
    fn frame_is_nearest_scaled_to_canvas() {
        // 2x2 source (all black) drawn onto a 6x4 canvas: every pixel covered.
        let frame = solid_yuv420p(2, 2, 16, 128, 128);
        let img = composite(6, 4, &[CompositeLayer::Frame(&frame)]);
        for y in 0..4 {
            for x in 0..6 {
                assert_eq!(img.pixel(x, y), [0, 0, 0, 255], "at {x},{y}");
            }
        }
    }

    #[test]
    fn rgba8_frame_samples_directly() {
        let frame = DecodedFrame {
            width: 1,
            height: 1,
            pts_ticks: 0,
            format: PixelFormat::Rgba8,
            planes: vec![Plane {
                data: vec![12, 34, 56, 200],
                stride: 4,
            }],
        };
        let img = composite(1, 1, &[CompositeLayer::Frame(&frame)]);
        // Over a transparent canvas, an alpha-200 source stays alpha-200.
        assert_eq!(img.pixel(0, 0), [12, 34, 56, 200]);
    }

    #[test]
    fn empty_canvas_is_safe() {
        let img = composite(0, 0, &[CompositeLayer::Solid([1, 2, 3, 4])]);
        assert!(img.pixels.is_empty());
    }
}
