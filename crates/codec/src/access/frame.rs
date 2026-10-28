//! An 8-bit 4:2:0 picture in the coded sample domain.
//!
//! This is the picture a video codec exchanges. Display conversion lives in
//! `color`. The compositor picture is a separate RGBA value.

use std::fmt;

/// Why a 4:2:0 picture could not be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameError {
    /// A dimension is zero.
    Empty,
    /// 4:2:0 needs even edges.
    Odd,
    /// A plane length does not match the size.
    Length,
    /// The picture would pass 8192 on an edge, or the product overflowed.
    Limit,
}

impl fmt::Display for FrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Empty => "frame size is empty",
            Self::Odd => "4:2:0 frame size is odd",
            Self::Length => "plane length does not match the frame size",
            Self::Limit => "frame size is past the limit",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for FrameError {}

/// Tightly packed 8-bit 4:2:0, visible region only.
///
/// `y` is `width * height` bytes. `u` and `v` are each `(width / 2) * (height / 2)`
/// bytes. Rows are contiguous. There is no padding and no separate stride.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Yuv420p {
    width: u32,
    height: u32,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl Yuv420p {
    /// Both edges must be even, non-zero, and at most 8192.
    pub fn try_new(
        width: u32,
        height: u32,
        y: Vec<u8>,
        u: Vec<u8>,
        v: Vec<u8>,
    ) -> Result<Self, FrameError> {
        if width == 0 || height == 0 {
            return Err(FrameError::Empty);
        }
        if width > 8192 || height > 8192 {
            return Err(FrameError::Limit);
        }
        if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(FrameError::Odd);
        }
        let y_len = (width as usize)
            .checked_mul(height as usize)
            .ok_or(FrameError::Limit)?;
        let c_len = ((width / 2) as usize)
            .checked_mul((height / 2) as usize)
            .ok_or(FrameError::Limit)?;
        if y.len() != y_len || u.len() != c_len || v.len() != c_len {
            return Err(FrameError::Length);
        }
        Ok(Self {
            width,
            height,
            y,
            u,
            v,
        })
    }

    /// A solid picture. Luma and both chroma planes are filled with one value each.
    pub fn solid(width: u32, height: u32, y: u8, u: u8, v: u8) -> Result<Self, FrameError> {
        if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Self::try_new(width, height, Vec::new(), Vec::new(), Vec::new());
        }
        let y_len = (width as usize)
            .checked_mul(height as usize)
            .ok_or(FrameError::Limit)?;
        let c_len = ((width / 2) as usize)
            .checked_mul((height / 2) as usize)
            .ok_or(FrameError::Limit)?;
        Self::try_new(
            width,
            height,
            vec![y; y_len],
            vec![u; c_len],
            vec![v; c_len],
        )
    }

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    #[must_use]
    pub fn y(&self) -> &[u8] {
        &self.y
    }

    #[must_use]
    pub fn u(&self) -> &[u8] {
        &self.u
    }

    #[must_use]
    pub fn v(&self) -> &[u8] {
        &self.v
    }

    #[must_use]
    pub fn y_mut(&mut self) -> &mut [u8] {
        &mut self.y
    }

    #[must_use]
    pub fn u_mut(&mut self) -> &mut [u8] {
        &mut self.u
    }

    #[must_use]
    pub fn v_mut(&mut self) -> &mut [u8] {
        &mut self.v
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameError, Yuv420p};

    #[test]
    fn a_solid_picture_keeps_even_planes() {
        let frame = Yuv420p::solid(4, 2, 16, 128, 128).expect("frame");
        assert_eq!(frame.width(), 4);
        assert_eq!(frame.height(), 2);
        assert_eq!(frame.y().len(), 8);
        assert_eq!(frame.u().len(), 2);
        assert_eq!(frame.v(), &[128, 128]);
    }

    #[test]
    fn an_odd_or_short_picture_is_refused() {
        assert_eq!(
            Yuv420p::try_new(1, 2, vec![0, 0], vec![], vec![]),
            Err(FrameError::Odd)
        );
        assert_eq!(
            Yuv420p::try_new(2, 2, vec![0, 0, 0], vec![0], vec![0]),
            Err(FrameError::Length)
        );
        assert_eq!(
            Yuv420p::try_new(0, 2, Vec::new(), Vec::new(), Vec::new()),
            Err(FrameError::Empty)
        );
    }
}
