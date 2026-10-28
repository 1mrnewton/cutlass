use super::error::CodecError;

/// One tightly packed sample plane. `samples.len()` is `width * height`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plane {
    width: u32,
    height: u32,
    samples: Vec<u8>,
}

impl Plane {
    /// Rejects an empty edge and a buffer whose length is not the pixel count.
    pub fn new(width: u32, height: u32, samples: Vec<u8>) -> Result<Self, CodecError> {
        if width == 0 || height == 0 {
            return Err(CodecError::InvalidApi {
                what: "plane edge is zero",
            });
        }
        let len = (width as usize)
            .checked_mul(height as usize)
            .ok_or(CodecError::Limit {
                what: "plane byte length",
            })?;
        if samples.len() != len {
            return Err(CodecError::InvalidApi {
                what: "plane byte length",
            });
        }
        Ok(Self {
            width,
            height,
            samples,
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
    pub fn samples(&self) -> &[u8] {
        &self.samples
    }

    #[must_use]
    pub fn samples_mut(&mut self) -> &mut [u8] {
        &mut self.samples
    }
}

/// 8-bit 4:2:0. Chroma edges are the luma edges rounded up to a multiple of two, then halved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Yuv420 {
    pub y: Plane,
    pub u: Plane,
    pub v: Plane,
}

impl Yuv420 {
    pub fn new(y: Plane, u: Plane, v: Plane) -> Result<Self, CodecError> {
        let chroma_w = y.width().div_ceil(2);
        let chroma_h = y.height().div_ceil(2);
        if u.width() != chroma_w
            || v.width() != chroma_w
            || u.height() != chroma_h
            || v.height() != chroma_h
        {
            return Err(CodecError::InvalidApi {
                what: "4:2:0 chroma size",
            });
        }
        Ok(Self { y, u, v })
    }
}

#[cfg(test)]
mod tests {
    use super::{Plane, Yuv420};
    use crate::access::CodecError;

    #[test]
    fn a_plane_keeps_a_matching_buffer() {
        let plane = Plane::new(2, 2, vec![1, 2, 3, 4]).expect("plane");
        assert_eq!(plane.samples(), &[1, 2, 3, 4]);
    }

    #[test]
    fn a_short_plane_is_refused() {
        let err = Plane::new(2, 2, vec![1, 2, 3]).expect_err("short");
        assert_eq!(
            err,
            CodecError::InvalidApi {
                what: "plane byte length",
            }
        );
    }

    #[test]
    fn yuv420_requires_half_resolution_chroma() {
        let y = Plane::new(4, 2, vec![0; 8]).expect("y");
        let u = Plane::new(2, 1, vec![128, 128]).expect("u");
        let v = Plane::new(2, 1, vec![128, 128]).expect("v");
        assert!(Yuv420::new(y, u, v).is_ok());
    }
}
