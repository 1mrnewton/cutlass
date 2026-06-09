use crate::error::CompositorError;

/// RGBA8 image returned from GPU readback (row-major, tightly packed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
}

impl RgbaImage {
    pub fn new(width: u32, height: u32, bytes: Vec<u8>) -> Result<Self, CompositorError> {
        let expected = usize::try_from(width)
            .ok()
            .and_then(|w| usize::try_from(height).ok().map(|h| w * h * 4))
            .ok_or(CompositorError::InvalidDimensions { width, height })?;
        if bytes.len() != expected {
            return Err(CompositorError::LayerSizeMismatch {
                got: bytes.len(),
                expected,
            });
        }
        Ok(Self {
            width,
            height,
            bytes,
        })
    }
}
