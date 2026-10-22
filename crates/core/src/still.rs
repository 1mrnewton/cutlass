//! A decoded still.

/// One picture. Straight RGBA8, full range, row by row, no padding.
///
/// `docs/codecs.md` is the layout. A missing alpha channel is opaque. The
/// decoder does not turn the picture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Still {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl Still {
    /// `pixels` is `width * height * 4` bytes. A zero edge is refused.
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> Result<Self, StillError> {
        if width == 0 || height == 0 {
            return Err(StillError::ZeroSize);
        }
        let expected = usize::try_from(width)
            .ok()
            .and_then(|width| {
                usize::try_from(height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .and_then(|count| count.checked_mul(4))
            .ok_or(StillError::Length)?;
        if pixels.len() != expected {
            return Err(StillError::Length);
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Straight RGBA8, row by row.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
}

/// Why [`Still::new`] rejected a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StillError {
    /// Width or height is zero.
    ZeroSize,
    /// The buffer is not `width * height * 4` bytes.
    Length,
}

impl std::fmt::Display for StillError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ZeroSize => "a still edge is zero",
            Self::Length => "a still buffer does not match its size",
        })
    }
}

impl std::error::Error for StillError {}

#[cfg(test)]
mod tests {
    use super::{Still, StillError};

    #[test]
    fn a_pixel_is_four_bytes() {
        let still = Still::new(1, 1, vec![1, 2, 3, 4]).expect("one pixel");
        assert_eq!(still.width(), 1);
        assert_eq!(still.height(), 1);
        assert_eq!(still.pixels(), &[1, 2, 3, 4]);
    }

    #[test]
    fn a_zero_edge_is_rejected() {
        assert_eq!(Still::new(0, 1, Vec::new()), Err(StillError::ZeroSize));
        assert_eq!(
            Still::new(1, 0, vec![0, 0, 0, 0]),
            Err(StillError::ZeroSize)
        );
    }

    #[test]
    fn a_short_buffer_is_rejected() {
        assert_eq!(Still::new(1, 1, vec![0, 0, 0]), Err(StillError::Length));
    }
}
