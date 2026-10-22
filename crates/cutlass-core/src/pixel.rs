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
    /// 4:2:2 — chroma at half width and full height.
    Yuv422,
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
    /// 8-bit planar 4:2:2: full-height chroma at half width.
    Yuv422p,
    /// 8-bit planar 4:4:4.
    Yuv444p,
    /// 10-bit planar 4:2:0. Each sample is a native-endian `u16` in the low
    /// 10 bits (not P010, which stores the bits in the high end of a biplanar
    /// layout).
    Yuv420p10,
    /// 10-bit planar 4:2:2, low-aligned native-endian `u16` samples.
    Yuv422p10,
    /// 10-bit planar 4:4:4, low-aligned native-endian `u16` samples.
    Yuv444p10,
    /// 12-bit planar 4:2:0, low-aligned native-endian `u16` samples.
    Yuv420p12,
    /// 12-bit planar 4:2:2, low-aligned native-endian `u16` samples.
    Yuv422p12,
    /// 12-bit planar 4:4:4, low-aligned native-endian `u16` samples.
    Yuv444p12,
    /// 8-bit monochrome (one luma plane). Chroma was not coded.
    Gray8,
    /// 10-bit monochrome, low-aligned native-endian `u16` samples.
    Gray10,
    /// 12-bit monochrome, low-aligned native-endian `u16` samples.
    Gray12,
}

impl PixelFormat {
    /// Number of separate planes the frame carries.
    pub fn plane_count(self) -> usize {
        match self {
            PixelFormat::Nv12 | PixelFormat::P010 => 2,
            PixelFormat::Yuv420p
            | PixelFormat::Yuv422p
            | PixelFormat::Yuv444p
            | PixelFormat::Yuv420p10
            | PixelFormat::Yuv422p10
            | PixelFormat::Yuv444p10
            | PixelFormat::Yuv420p12
            | PixelFormat::Yuv422p12
            | PixelFormat::Yuv444p12 => 3,
            PixelFormat::Rgba8
            | PixelFormat::Bgra8
            | PixelFormat::Gray8
            | PixelFormat::Gray10
            | PixelFormat::Gray12 => 1,
        }
    }

    /// Bits actually used per component sample.
    ///
    /// Planar 10- and 12-bit formats store each sample in a native-endian
    /// `u16` with the significant bits in the low end. [`PixelFormat::P010`]
    /// is the hardware biplanar layout with 10 bits in the high end.
    pub fn bit_depth(self) -> u8 {
        match self {
            PixelFormat::P010
            | PixelFormat::Yuv420p10
            | PixelFormat::Yuv422p10
            | PixelFormat::Yuv444p10
            | PixelFormat::Gray10 => 10,
            PixelFormat::Yuv420p12
            | PixelFormat::Yuv422p12
            | PixelFormat::Yuv444p12
            | PixelFormat::Gray12 => 12,
            _ => 8,
        }
    }

    /// Storage bytes per component sample.
    ///
    /// High-bit-depth planar formats and P010 use 2. Byte-dump serializers
    /// must state endianness: planar samples are native-endian low-aligned.
    pub fn bytes_per_sample(self) -> usize {
        match self.bit_depth() {
            8 => 1,
            _ => 2,
        }
    }

    /// True for luma/chroma layouts, including monochrome. Packed RGB is false.
    pub fn is_yuv(self) -> bool {
        !matches!(self, PixelFormat::Rgba8 | PixelFormat::Bgra8)
    }

    /// True when the format is a single luma plane and chroma was not coded.
    pub fn is_monochrome(self) -> bool {
        matches!(
            self,
            PixelFormat::Gray8 | PixelFormat::Gray10 | PixelFormat::Gray12
        )
    }

    /// Chroma subsampling, or `None` for packed RGB and monochrome.
    pub fn chroma_subsampling(self) -> Option<ChromaSubsampling> {
        match self {
            PixelFormat::Nv12
            | PixelFormat::P010
            | PixelFormat::Yuv420p
            | PixelFormat::Yuv420p10
            | PixelFormat::Yuv420p12 => Some(ChromaSubsampling::Yuv420),
            PixelFormat::Yuv422p | PixelFormat::Yuv422p10 | PixelFormat::Yuv422p12 => {
                Some(ChromaSubsampling::Yuv422)
            }
            PixelFormat::Yuv444p | PixelFormat::Yuv444p10 | PixelFormat::Yuv444p12 => {
                Some(ChromaSubsampling::Yuv444)
            }
            PixelFormat::Rgba8
            | PixelFormat::Bgra8
            | PixelFormat::Gray8
            | PixelFormat::Gray10
            | PixelFormat::Gray12 => None,
        }
    }

    /// True when a dimension must be even so chroma planes divide evenly.
    ///
    /// 4:2:0 requires even width and height. 4:2:2 requires even width.
    /// Callers still check height separately via [`Self::requires_even_height`].
    pub fn requires_even_dimensions(self) -> bool {
        matches!(self.chroma_subsampling(), Some(ChromaSubsampling::Yuv420))
    }

    /// True when the coded width must be even.
    pub fn requires_even_width(self) -> bool {
        matches!(
            self.chroma_subsampling(),
            Some(ChromaSubsampling::Yuv420 | ChromaSubsampling::Yuv422)
        )
    }

    /// True when the coded height must be even.
    pub fn requires_even_height(self) -> bool {
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
        assert_eq!(
            PixelFormat::Yuv422p10.chroma_subsampling(),
            Some(ChromaSubsampling::Yuv422)
        );
        assert!(PixelFormat::Yuv422p.requires_even_width());
        assert!(!PixelFormat::Yuv422p.requires_even_height());
        assert!(PixelFormat::Gray8.is_monochrome());
        assert_eq!(PixelFormat::Gray8.plane_count(), 1);
        assert_eq!(PixelFormat::Yuv444p12.bit_depth(), 12);
        assert_eq!(PixelFormat::Yuv420p10.bytes_per_sample(), 2);
        assert_ne!(PixelFormat::Yuv420p10, PixelFormat::P010);
    }
}
