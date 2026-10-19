//! [`RgbaImage`]: the canonical CPU-side RGBA8 image.
//!
//! This is the plain pixel container that crosses crate seams in the CPU
//! direction: the compositor reads a rendered canvas back into one, a text or
//! shape generator rasterizes into one, a decoder can hand back a still in one,
//! and the future Python bindings expose `get_frame(t)` as one (`width`,
//! `height`, and a flat `Vec<u8>` map straight onto a NumPy array).
//!
//! Pixels are **straight (non-premultiplied) alpha**, row-major, tightly packed
//! (row pitch is exactly `width * 4`). It is deliberately dependency-free so it
//! can live in core without dragging `wgpu` or any image library along.

/// A tightly-packed, straight-alpha RGBA8 image (`width * height * 4` bytes).
#[derive(Clone, PartialEq, Eq)]
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` bytes, row-major, no padding, straight alpha.
    pub pixels: Vec<u8>,
}

impl RgbaImage {
    /// Wrap existing pixels. `pixels.len()` should be `width * height * 4`.
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> Self {
        Self {
            width,
            height,
            pixels,
        }
    }

    /// A fully transparent `width × height` image (all zero).
    pub fn transparent(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![0; (width as usize) * (height as usize) * 4],
        }
    }

    /// True if the buffer holds exactly `width * height * 4` bytes.
    pub fn is_well_formed(&self) -> bool {
        (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|n| n.checked_mul(4))
            .is_some_and(|need| self.pixels.len() == need)
    }

    /// The RGBA quad at `(x, y)`, or `[0; 4]` if out of bounds.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        if x >= self.width || y >= self.height {
            return [0; 4];
        }
        let i = ((y * self.width + x) * 4) as usize;
        [
            self.pixels[i],
            self.pixels[i + 1],
            self.pixels[i + 2],
            self.pixels[i + 3],
        ]
    }
}

impl std::fmt::Debug for RgbaImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Summarize; never dump the pixel bytes.
        f.debug_struct("RgbaImage")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bytes", &self.pixels.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transparent_is_zeroed_and_well_formed() {
        let img = RgbaImage::transparent(3, 2);
        assert_eq!(img.pixels.len(), 3 * 2 * 4);
        assert!(img.is_well_formed());
        assert_eq!(img.pixel(0, 0), [0, 0, 0, 0]);
        assert_eq!(img.pixel(2, 1), [0, 0, 0, 0]);
    }

    #[test]
    fn pixel_reads_are_bounds_checked() {
        let mut img = RgbaImage::transparent(2, 2);
        img.pixels[..4].copy_from_slice(&[10, 20, 30, 40]);
        assert_eq!(img.pixel(0, 0), [10, 20, 30, 40]);
        // Out of bounds is transparent, not a panic.
        assert_eq!(img.pixel(2, 0), [0, 0, 0, 0]);
        assert_eq!(img.pixel(0, 2), [0, 0, 0, 0]);
    }

    #[test]
    fn well_formed_rejects_wrong_length() {
        assert!(!RgbaImage::new(2, 2, vec![0; 10]).is_well_formed());
        assert!(RgbaImage::new(2, 2, vec![0; 16]).is_well_formed());
    }
}
