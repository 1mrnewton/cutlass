//! Pixel formats: the plane layouts that cross the decode→compositor seam.
//!
//! The set is intentionally small — exactly what platform hardware decoders
//! emit plus packed RGB for stills and the CPU fallback. A `VideoToolbox` /
//! `MediaCodec` / `Media Foundation` decoder hands back biplanar YUV (NV12 for
//! 8-bit, P010 for 10-bit); software/Linux fallbacks and image decoders may give
//! planar YUV or packed RGBA/BGRA. The compositor samples whichever layout it
//! receives (planes bound as textures) and converts on the GPU.

/// Chroma subsampling of a YUV format.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChromaSubsampling {
    /// 4:2:0 — chroma at half width and half height (the video norm).
    Yuv420,
    /// 4:4:4 — chroma at full resolution (no subsampling; RGB-like).
    Yuv444,
}

/// A pixel format crossing the frame handoff.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// 8-bit biplanar 4:2:0: full-res Y plane, then interleaved U/V at quarter
    /// resolution. The standard hardware-decode output (VideoToolbox / NVDEC /
    /// MediaCodec / Media Foundation).
    Nv12,
    /// 10-bit biplanar 4:2:0: like NV12 but each sample is 16-bit little-endian
    /// with the 10 significant bits in the high end. The 10-bit HEVC / HDR
    /// hardware-decode output.
    P010,
    /// 8-bit planar 4:2:0: separate Y, U, V planes. Software-decode and image
    /// fallback layout.
    Yuv420p,
    /// 8-bit packed RGBA, one plane (decoded stills, generated frames).
    Rgba8,
    /// 8-bit packed BGRA, one plane. CoreVideo's common RGB output order.
    Bgra8,
}

impl PixelFormat {
    /// Number of separate planes the frame carries.
    pub fn plane_count(self) -> usize {
        match self {
            PixelFormat::Nv12 | PixelFormat::P010 => 2,
            PixelFormat::Yuv420p => 3,
            PixelFormat::Rgba8 | PixelFormat::Bgra8 => 1,
        }
    }

    /// Bits actually used per component sample (8 for SDR layouts, 10 for P010).
    pub fn bit_depth(self) -> u8 {
        match self {
            PixelFormat::P010 => 10,
            _ => 8,
        }
    }

    /// Storage bytes per component sample (P010 stores 10-bit values in 16 bits).
    pub fn bytes_per_sample(self) -> usize {
        match self {
            PixelFormat::P010 => 2,
            _ => 1,
        }
    }

    /// True for the YUV formats (need a YUV→RGB matrix); false for packed RGB.
    pub fn is_yuv(self) -> bool {
        matches!(
            self,
            PixelFormat::Nv12 | PixelFormat::P010 | PixelFormat::Yuv420p
        )
    }

    /// Chroma subsampling, or `None` for packed-RGB formats.
    pub fn chroma_subsampling(self) -> Option<ChromaSubsampling> {
        match self {
            PixelFormat::Nv12 | PixelFormat::P010 | PixelFormat::Yuv420p => {
                Some(ChromaSubsampling::Yuv420)
            }
            PixelFormat::Rgba8 | PixelFormat::Bgra8 => None,
        }
    }

    /// True if width/height must be even (any 4:2:0 layout, so chroma divides).
    pub fn requires_even_dimensions(self) -> bool {
        matches!(self.chroma_subsampling(), Some(ChromaSubsampling::Yuv420))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_counts() {
        assert_eq!(PixelFormat::Nv12.plane_count(), 2);
        assert_eq!(PixelFormat::P010.plane_count(), 2);
        assert_eq!(PixelFormat::Yuv420p.plane_count(), 3);
        assert_eq!(PixelFormat::Rgba8.plane_count(), 1);
        assert_eq!(PixelFormat::Bgra8.plane_count(), 1);
    }

    #[test]
    fn p010_is_ten_bit_two_bytes() {
        assert_eq!(PixelFormat::P010.bit_depth(), 10);
        assert_eq!(PixelFormat::P010.bytes_per_sample(), 2);
        assert_eq!(PixelFormat::Nv12.bit_depth(), 8);
        assert_eq!(PixelFormat::Nv12.bytes_per_sample(), 1);
    }

    #[test]
    fn yuv_classification() {
        assert!(PixelFormat::Nv12.is_yuv());
        assert!(PixelFormat::P010.is_yuv());
        assert!(PixelFormat::Yuv420p.is_yuv());
        assert!(!PixelFormat::Rgba8.is_yuv());
        assert!(!PixelFormat::Bgra8.is_yuv());
    }

    #[test]
    fn subsampling_and_even_requirement() {
        assert_eq!(
            PixelFormat::Nv12.chroma_subsampling(),
            Some(ChromaSubsampling::Yuv420)
        );
        assert_eq!(PixelFormat::Bgra8.chroma_subsampling(), None);
        assert!(PixelFormat::Nv12.requires_even_dimensions());
        assert!(!PixelFormat::Rgba8.requires_even_dimensions());
    }
}
