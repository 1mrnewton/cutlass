//! A picture and the source that produces one at a microsecond.
//!
//! A picture is CPU bytes or a platform buffer. A sample is ready or missing.
//! The software device can sample the bytes. It cannot sample the platform
//! buffer.

#![forbid(unsafe_code)]

use std::fmt;

use cutlass_gpu::{Extent, TexelSource};
use cutlass_model::Micros;

/// Why CPU bytes could not be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PictureError {
    /// An edge is past the device limit.
    Extent,
    /// The byte length is not four times the pixel count.
    Length,
}

impl fmt::Display for PictureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Extent => "the picture size is past the limit",
            Self::Length => "the picture byte length does not match its size",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for PictureError {}

/// Four bytes per pixel, or nothing when an edge is past the device limit.
#[must_use]
pub fn rgba_len(width: u32, height: u32) -> Option<usize> {
    Extent { width, height }.rgba_len()
}

/// RGBA bytes, four per pixel, tightly packed, row by row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuBytes {
    extent: Extent,
    rgba: Vec<u8>,
}

impl CpuBytes {
    /// `rgba` must be exactly four bytes times `width` times `height`.
    pub fn new(width: u32, height: u32, rgba: Vec<u8>) -> Result<Self, PictureError> {
        let extent = Extent { width, height };
        let len = rgba_len(width, height).ok_or(PictureError::Extent)?;
        if rgba.len() != len {
            return Err(PictureError::Length);
        }
        Ok(Self { extent, rgba })
    }

    #[must_use]
    pub fn extent(&self) -> Extent {
        self.extent
    }

    #[must_use]
    pub fn width(&self) -> u32 {
        self.extent.width
    }

    #[must_use]
    pub fn height(&self) -> u32 {
        self.extent.height
    }

    #[must_use]
    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }
}

/// A buffer owned by a platform device.
///
/// The software device cannot sample this. An encoder may accept it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlatformBuffer {
    pub extent: Extent,
    pub token: u64,
}

/// CPU bytes or a platform buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Picture {
    Cpu(CpuBytes),
    Platform(PlatformBuffer),
}

impl Picture {
    /// The draw input for a device. A platform buffer stays opaque.
    #[must_use]
    pub fn texels(&self) -> TexelSource<'_> {
        match self {
            Self::Cpu(cpu) => TexelSource::Rgba {
                extent: cpu.extent,
                bytes: &cpu.rgba,
            },
            Self::Platform(platform) => TexelSource::Platform {
                extent: platform.extent,
                token: platform.token,
            },
        }
    }
}

/// A sample at one microsecond.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Sample {
    Ready(Picture),
    Miss,
}

/// One picture at one timeline microsecond.
pub trait PictureSource {
    fn sample(&self, time: Micros) -> Sample;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixed {
        sample: Sample,
    }

    impl PictureSource for Fixed {
        fn sample(&self, _time: Micros) -> Sample {
            self.sample.clone()
        }
    }

    #[test]
    fn cpu_bytes_keep_a_matching_buffer() {
        let bytes = CpuBytes::new(
            2,
            2,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        )
        .unwrap();
        assert_eq!(bytes.width(), 2);
        assert_eq!(bytes.height(), 2);
        assert_eq!(bytes.rgba().len(), 16);
    }

    #[test]
    fn cpu_bytes_refuse_a_short_buffer_and_a_long_edge() {
        assert_eq!(
            CpuBytes::new(1, 1, vec![1, 2, 3]),
            Err(PictureError::Length)
        );
        assert_eq!(
            CpuBytes::new(cutlass_gpu::MAX_EDGE + 1, 1, Vec::new()),
            Err(PictureError::Extent)
        );
    }

    #[test]
    fn a_sample_is_ready_or_missing() {
        let ready = Fixed {
            sample: Sample::Ready(Picture::Cpu(CpuBytes::new(1, 1, vec![9, 8, 7, 6]).unwrap())),
        };
        let miss = Fixed {
            sample: Sample::Miss,
        };
        assert!(matches!(
            ready.sample(Micros(0)),
            Sample::Ready(Picture::Cpu(_))
        ));
        assert!(matches!(miss.sample(Micros(0)), Sample::Miss));
    }

    #[test]
    fn a_platform_buffer_stays_a_platform_buffer() {
        let picture = Picture::Platform(PlatformBuffer {
            extent: Extent {
                width: 4,
                height: 2,
            },
            token: 11,
        });
        assert!(matches!(
            picture.texels(),
            TexelSource::Platform {
                token: 11,
                extent: Extent {
                    width: 4,
                    height: 2,
                },
            }
        ));
    }
}
