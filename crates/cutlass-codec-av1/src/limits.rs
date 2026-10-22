//! Resource caps. These are not sequence-level declarations: a header that
//! advertises a large max frame size still cannot allocate past the cap.

/// Bounds checked before allocation and before accepting input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderLimits {
    /// Maximum coded or upscaled width or height in luma samples.
    pub max_dimension: u32,
    /// Maximum luma samples in one frame (`width * height` of the upscaled image).
    pub max_pixels: u64,
    /// Maximum bytes in one submitted packet.
    pub max_packet_bytes: usize,
    /// Maximum bytes in one OBU payload.
    pub max_obu_bytes: usize,
    /// Maximum bytes retained for one metadata OBU.
    pub max_metadata_bytes: usize,
    /// Maximum OBUs buffered while a frame is incomplete.
    pub max_pending_obus: usize,
    /// Maximum presentation records waiting in `receive`.
    pub max_queued_frames: usize,
    /// Tile decode workers. Only 1 is implemented.
    pub workers: u32,
}

impl Default for DecoderLimits {
    fn default() -> Self {
        Self {
            // Level 5.1 is 4096 wide; the default allows 4K-class frames
            // including AV1's 8-pixel alignment padding. Callers raise this
            // explicitly for larger targets.
            max_dimension: 4096,
            max_pixels: 4096 * 2176,
            max_packet_bytes: 16 * 1024 * 1024,
            max_obu_bytes: 8 * 1024 * 1024,
            max_metadata_bytes: 1024 * 1024,
            max_pending_obus: 64,
            max_queued_frames: 8,
            workers: 1,
        }
    }
}

impl DecoderLimits {
    pub fn validate(&self) -> Result<(), crate::error::DecodeError> {
        if self.max_dimension == 0 || self.max_dimension > 65536 {
            return Err(crate::error::DecodeError::limit(
                "max_dimension must be in 1..=65536",
            ));
        }
        if self.max_pixels == 0 {
            return Err(crate::error::DecodeError::limit("max_pixels is 0"));
        }
        if self.workers != 1 {
            return Err(crate::error::DecodeError::unsupported(
                "only a single scalar worker is implemented",
            ));
        }
        Ok(())
    }
}
