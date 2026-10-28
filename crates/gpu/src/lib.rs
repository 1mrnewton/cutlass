//! The device the compositor submits a frame to.
//!
//! This crate is traits and types. A device creates a framebuffer, begins a
//! pass, sets the blend, clears, draws, ends, submits, and destroys the
//! framebuffer. The software device implements this over CPU buffers. It
//! cannot sample a [`TexelSource::Platform`].
//!
//! An edge longer than [`MAX_EDGE`] is refused. A frame stores four bytes per
//! pixel, tightly packed, row by row.

#![forbid(unsafe_code)]

/// The longest edge a framebuffer or a drawn picture may have.
pub const MAX_EDGE: u32 = 8_192;

/// Width and height in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Extent {
    pub width: u32,
    pub height: u32,
}

impl Extent {
    /// Four bytes per pixel, or nothing when an edge is past [`MAX_EDGE`].
    #[must_use]
    pub fn rgba_len(self) -> Option<usize> {
        if self.width > MAX_EDGE || self.height > MAX_EDGE {
            return None;
        }
        let width = usize::try_from(self.width).ok()?;
        let height = usize::try_from(self.height).ok()?;
        width.checked_mul(height)?.checked_mul(4)
    }
}

/// The color written by a clear.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClearColor {
    pub rgba: [u8; 4],
}

impl ClearColor {
    /// Empty pixels. An empty source list leaves this color.
    pub const TRANSPARENT: Self = Self { rgba: [0, 0, 0, 0] };
}

/// How a drawn picture combines with the framebuffer.
///
/// `Over` is `output = source + destination * (1 - source.alpha)` on each
/// channel. The 8-bit alpha is divided by 255 and the remainder is truncated.
/// A channel above 255 is clamped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Blend {
    #[default]
    Over,
}

/// CPU bytes the device can sample, or a platform buffer it may not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexelSource<'a> {
    Rgba { extent: Extent, bytes: &'a [u8] },
    Platform { extent: Extent, token: u64 },
}

/// A framebuffer created by a [`Device`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Framebuffer {
    id: u64,
}

impl Framebuffer {
    #[must_use]
    pub fn new(id: u64) -> Self {
        Self { id }
    }

    #[must_use]
    pub fn id(self) -> u64 {
        self.id
    }
}

/// One pass targeting a framebuffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Pass {
    framebuffer: Framebuffer,
}

impl Pass {
    #[must_use]
    pub fn new(framebuffer: Framebuffer) -> Self {
        Self { framebuffer }
    }

    #[must_use]
    pub fn framebuffer(self) -> Framebuffer {
        self.framebuffer
    }
}

/// Framebuffers, passes, draw, and submit.
///
/// The legal sequence is `create_framebuffer`, `begin`, `blend`, `clear`,
/// `draw` for each picture, `end`, `submit`, `read_rgba`, then
/// `destroy_framebuffer`. `read_rgba` is the CPU readback.
pub trait Device {
    type Error: std::error::Error;

    fn create_framebuffer(&mut self, extent: Extent) -> Result<Framebuffer, Self::Error>;

    fn destroy_framebuffer(&mut self, framebuffer: Framebuffer) -> Result<(), Self::Error>;

    fn begin(&mut self, framebuffer: Framebuffer) -> Result<Pass, Self::Error>;

    fn blend(&mut self, pass: Pass, blend: Blend) -> Result<(), Self::Error>;

    fn clear(&mut self, pass: Pass, color: ClearColor) -> Result<(), Self::Error>;

    fn draw(&mut self, pass: Pass, source: TexelSource<'_>) -> Result<(), Self::Error>;

    fn end(&mut self, pass: Pass) -> Result<(), Self::Error>;

    fn submit(&mut self, pass: Pass) -> Result<(), Self::Error>;

    fn read_rgba(&self, framebuffer: Framebuffer) -> Result<&[u8], Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_len_stops_at_the_edge_limit() {
        assert_eq!(
            Extent {
                width: 0,
                height: 0
            }
            .rgba_len(),
            Some(0)
        );
        assert_eq!(
            Extent {
                width: 2,
                height: 3
            }
            .rgba_len(),
            Some(24)
        );
        let edge = usize::try_from(MAX_EDGE).unwrap();
        assert_eq!(
            Extent {
                width: MAX_EDGE,
                height: MAX_EDGE,
            }
            .rgba_len(),
            Some(edge * edge * 4)
        );
        assert!(
            Extent {
                width: MAX_EDGE + 1,
                height: 1,
            }
            .rgba_len()
            .is_none()
        );
        assert!(
            Extent {
                width: 1,
                height: MAX_EDGE + 1,
            }
            .rgba_len()
            .is_none()
        );
    }

    #[test]
    fn a_platform_source_is_not_cpu_bytes() {
        let extent = Extent {
            width: 1,
            height: 1,
        };
        let source = TexelSource::Platform { extent, token: 7 };
        assert!(matches!(source, TexelSource::Platform { token: 7, .. }));
        assert_ne!(
            source,
            TexelSource::Rgba {
                extent,
                bytes: &[0, 0, 0, 0],
            }
        );
    }

    #[test]
    fn a_pass_points_at_its_framebuffer() {
        let framebuffer = Framebuffer::new(12);
        let pass = Pass::new(framebuffer);
        assert_eq!(pass.framebuffer().id(), 12);
    }
}
