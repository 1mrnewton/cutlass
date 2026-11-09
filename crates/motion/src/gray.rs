//! Small grey copies of frames, and the image pyramids the point tracker
//! reads.

use crate::MotionError;

/// The longest the short side of a grey copy gets. Tracking reads a small
/// picture: it is fast, and the pixel noise of a large one does not help.
pub const GRAY_SIDE: u32 = 360;

/// The largest frame side accepted.
pub const MAX_SIDE: u32 = 16_384;

/// The smallest grey copy that can be tracked in.
const MIN_SIDE: u32 = 16;

/// A grey picture, values 0 to 255, rows top to bottom.
#[derive(Clone, Debug, PartialEq)]
pub struct Gray {
    width: u32,
    height: u32,
    data: Vec<f32>,
}

impl Gray {
    /// A grey copy of RGB pixels, shrunk so its short side is at most
    /// [`GRAY_SIDE`].
    pub fn from_rgb(width: u32, height: u32, rgb: &[u8]) -> Result<Self, MotionError> {
        Self::from_pixels(width, height, rgb, 3)
    }

    /// A grey copy of RGBA pixels, shrunk so its short side is at most
    /// [`GRAY_SIDE`]. Alpha is ignored.
    pub fn from_rgba(width: u32, height: u32, rgba: &[u8]) -> Result<Self, MotionError> {
        Self::from_pixels(width, height, rgba, 4)
    }

    /// A grey picture from its values, as it is.
    pub fn new(width: u32, height: u32, data: Vec<f32>) -> Result<Self, MotionError> {
        check_size(width, height)?;
        if width < MIN_SIDE || height < MIN_SIDE {
            return Err(MotionError::Frame {
                reason: "the frame is too small to track in".into(),
            });
        }
        if data.len() != width as usize * height as usize {
            return Err(MotionError::Frame {
                reason: "the values do not fill the frame".into(),
            });
        }
        if data.iter().any(|v| !v.is_finite()) {
            return Err(MotionError::Frame {
                reason: "a value is not a number".into(),
            });
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    fn from_pixels(
        width: u32,
        height: u32,
        pixels: &[u8],
        channels: usize,
    ) -> Result<Self, MotionError> {
        check_size(width, height)?;
        let (w, h) = (width as usize, height as usize);
        if pixels.len() != w * h * channels {
            return Err(MotionError::Frame {
                reason: "the pixels do not fill the frame".into(),
            });
        }
        // Shrink by a whole factor, averaging each block, so no detail is
        // invented and every source pixel counts once.
        let short = width.min(height);
        let factor = short.div_ceil(GRAY_SIDE).max(1) as usize;
        let (out_w, out_h) = (w / factor, h / factor);
        let mut data = vec![0.0_f32; out_w * out_h];
        let area = (factor * factor) as f32;
        for (oy, row) in data.chunks_exact_mut(out_w.max(1)).enumerate() {
            for (ox, out) in row.iter_mut().enumerate() {
                let mut sum = 0.0;
                for y in oy * factor..(oy + 1) * factor {
                    let start = (y * w + ox * factor) * channels;
                    let block = &pixels[start..start + factor * channels];
                    for pixel in block.chunks_exact(channels) {
                        sum += 0.299 * f32::from(pixel[0])
                            + 0.587 * f32::from(pixel[1])
                            + 0.114 * f32::from(pixel[2]);
                    }
                }
                *out = sum / area;
            }
        }
        Self::new(out_w as u32, out_h as u32, data)
    }
}

fn check_size(width: u32, height: u32) -> Result<(), MotionError> {
    if width == 0 || height == 0 || width > MAX_SIDE || height > MAX_SIDE {
        return Err(MotionError::Frame {
            reason: format!("a {width}x{height} frame is out of range"),
        });
    }
    Ok(())
}

/// One level of a pyramid: the picture and its gradients.
#[derive(Clone, Debug)]
pub(crate) struct Level {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) image: Vec<f32>,
    pub(crate) gx: Vec<f32>,
    pub(crate) gy: Vec<f32>,
}

impl Level {
    fn new(width: usize, height: usize, image: Vec<f32>) -> Self {
        let (gx, gy) = gradients(width, height, &image);
        Self {
            width,
            height,
            image,
            gx,
            gy,
        }
    }

    /// The picture at `x`, `y`, between pixels by bilinear weights, the
    /// edge pixels standing in outside.
    pub(crate) fn sample(&self, x: f32, y: f32) -> f32 {
        sample(&self.image, self.width, self.height, x, y)
    }

    pub(crate) fn sample_gx(&self, x: f32, y: f32) -> f32 {
        sample(&self.gx, self.width, self.height, x, y)
    }

    pub(crate) fn sample_gy(&self, x: f32, y: f32) -> f32 {
        sample(&self.gy, self.width, self.height, x, y)
    }
}

/// A picture at full size, then each half the size of the one before.
#[derive(Clone, Debug)]
pub(crate) struct Pyramid {
    pub(crate) levels: Vec<Level>,
}

/// The most levels, and the smallest side a level may have.
const MAX_LEVELS: usize = 4;
const MIN_LEVEL_SIDE: usize = 24;

impl Pyramid {
    pub(crate) fn new(gray: &Gray) -> Self {
        let mut levels = vec![Level::new(
            gray.width as usize,
            gray.height as usize,
            gray.data.clone(),
        )];
        while levels.len() < MAX_LEVELS {
            let Some(last) = levels.last() else { break };
            let (w, h) = (last.width / 2, last.height / 2);
            if w < MIN_LEVEL_SIDE || h < MIN_LEVEL_SIDE {
                break;
            }
            let mut image = vec![0.0; w * h];
            for (y, row) in image.chunks_exact_mut(w).enumerate() {
                for (x, out) in row.iter_mut().enumerate() {
                    let at =
                        |dx: usize, dy: usize| last.image[(2 * y + dy) * last.width + 2 * x + dx];
                    *out = (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) * 0.25;
                }
            }
            levels.push(Level::new(w, h, image));
        }
        Self { levels }
    }

    pub(crate) fn base(&self) -> &Level {
        // A pyramid always has its full-size level.
        &self.levels[0]
    }
}

/// Scharr gradients, the edge pixels repeated outside.
fn gradients(width: usize, height: usize, image: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut gx = vec![0.0; width * height];
    let mut gy = vec![0.0; width * height];
    let at = |x: usize, y: usize| image[y * width + x];
    for y in 0..height {
        let (up, down) = (y.saturating_sub(1), (y + 1).min(height - 1));
        for x in 0..width {
            let (left, right) = (x.saturating_sub(1), (x + 1).min(width - 1));
            let dx = 3.0 * (at(right, up) - at(left, up))
                + 10.0 * (at(right, y) - at(left, y))
                + 3.0 * (at(right, down) - at(left, down));
            let dy = 3.0 * (at(left, down) - at(left, up))
                + 10.0 * (at(x, down) - at(x, up))
                + 3.0 * (at(right, down) - at(right, up));
            gx[y * width + x] = dx / 32.0;
            gy[y * width + x] = dy / 32.0;
        }
    }
    (gx, gy)
}

fn sample(values: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, (width - 1) as f32);
    let y = y.clamp(0.0, (height - 1) as f32);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(height - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let top = values[y0 * width + x0] * (1.0 - fx) + values[y0 * width + x1] * fx;
    let bottom = values[y1 * width + x0] * (1.0 - fx) + values[y1 * width + x1] * fx;
    top * (1.0 - fy) + bottom * fy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_1080p_frame_becomes_a_360_line_grey_copy() {
        let (w, h) = (1920_u32, 1080_u32);
        let mut rgba = vec![0_u8; (w * h * 4) as usize];
        for pixel in rgba.as_chunks_mut::<4>().0 {
            pixel.copy_from_slice(&[255, 0, 0, 255]);
        }
        let gray = Gray::from_rgba(w, h, &rgba).unwrap();
        assert_eq!((gray.width(), gray.height()), (640, 360));
        assert!((gray.data()[0] - 0.299 * 255.0).abs() < 1e-3);
        // A small frame keeps its size.
        let rgb = vec![10_u8; 100 * 50 * 3];
        let small = Gray::from_rgb(100, 50, &rgb).unwrap();
        assert_eq!((small.width(), small.height()), (100, 50));
        assert!((small.data()[7] - 10.0).abs() < 1e-3);
    }

    #[test]
    fn a_bad_frame_is_refused() {
        assert!(Gray::from_rgb(0, 10, &[]).is_err());
        assert!(Gray::from_rgb(20, 20, &[0; 10]).is_err());
        assert!(Gray::from_rgb(MAX_SIDE + 1, 1, &[]).is_err());
        assert!(Gray::from_rgba(8, 8, &[0; 256]).is_err(), "too small");
        assert!(Gray::new(16, 16, vec![f32::NAN; 256]).is_err());
        assert!(Gray::new(16, 16, vec![0.0; 255]).is_err());
    }

    #[test]
    fn a_pyramid_halves_down_to_its_smallest_level() {
        let gray = Gray::new(200, 100, vec![1.0; 20_000]).unwrap();
        let pyramid = Pyramid::new(&gray);
        let sizes: Vec<_> = pyramid.levels.iter().map(|l| (l.width, l.height)).collect();
        assert_eq!(sizes, vec![(200, 100), (100, 50), (50, 25)]);
        assert!(pyramid.levels[2].gx.iter().all(|g| g.abs() < 1e-6));
    }
}
