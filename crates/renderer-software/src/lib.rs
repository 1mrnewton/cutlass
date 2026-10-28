//! The [`cutlass_gpu::Device`] over CPU buffers.
//!
//! Draw blends RGBA bytes into the framebuffer. A platform buffer is refused
//! and the framebuffer is left unchanged. Nothing here links Metal, Direct3D,
//! or Vulkan.

#![forbid(unsafe_code)]

use std::fmt;

use cutlass_gpu::{Blend, ClearColor, Device, Extent, Framebuffer, Pass, TexelSource};

/// Why the software device refused a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoftwareError {
    /// An edge is past the limit.
    Extent,
    /// The picture size is not the framebuffer size.
    Size,
    /// The byte length is not four times the pixel count.
    Bytes,
    /// A platform buffer cannot be sampled here.
    PlatformBuffer,
    /// The id is not a live framebuffer.
    MissingFramebuffer,
    /// `begin` was called while a pass was already open.
    PassOpen,
    /// The pass is not the open one.
    NoPass,
    /// Draw, clear, or blend was called after `end`.
    Ended,
    /// `submit` was called before `end`.
    NotEnded,
    /// `read_rgba` was called before `submit`.
    NotSubmitted,
}

impl fmt::Display for SoftwareError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Extent => "the frame size is past the limit",
            Self::Size => "the picture size does not match the framebuffer",
            Self::Bytes => "the picture byte length does not match its size",
            Self::PlatformBuffer => "the software device cannot sample a platform buffer",
            Self::MissingFramebuffer => "the framebuffer is not on this device",
            Self::PassOpen => "a pass is already open",
            Self::NoPass => "this pass is not open",
            Self::Ended => "this pass has ended",
            Self::NotEnded => "this pass has not ended",
            Self::NotSubmitted => "the frame has not been submitted",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for SoftwareError {}

#[derive(Debug)]
struct Slot {
    id: u64,
    extent: Extent,
    rgba: Vec<u8>,
    submitted: bool,
}

#[derive(Clone, Copy, Debug)]
struct OpenPass {
    framebuffer: Framebuffer,
    ended: bool,
    blend: Blend,
}

/// One software device. It owns its framebuffer bytes.
#[derive(Debug)]
pub struct SoftwareDevice {
    next_id: u64,
    buffers: Vec<Slot>,
    open: Option<OpenPass>,
}

impl SoftwareDevice {
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: 1,
            buffers: Vec::new(),
            open: None,
        }
    }

    fn slot_index(&self, framebuffer: Framebuffer) -> Result<usize, SoftwareError> {
        self.buffers
            .iter()
            .position(|slot| slot.id == framebuffer.id())
            .ok_or(SoftwareError::MissingFramebuffer)
    }

    fn require_drawing(&self, pass: Pass) -> Result<(), SoftwareError> {
        match self.open {
            Some(open) if open.framebuffer == pass.framebuffer() && !open.ended => Ok(()),
            Some(open) if open.framebuffer == pass.framebuffer() => Err(SoftwareError::Ended),
            _ => Err(SoftwareError::NoPass),
        }
    }
}

impl Default for SoftwareDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl Device for SoftwareDevice {
    type Error = SoftwareError;

    fn create_framebuffer(&mut self, extent: Extent) -> Result<Framebuffer, Self::Error> {
        let len = extent.rgba_len().ok_or(SoftwareError::Extent)?;
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).ok_or(SoftwareError::Extent)?;
        let mut rgba = Vec::new();
        rgba.try_reserve_exact(len)
            .map_err(|_| SoftwareError::Extent)?;
        rgba.resize(len, 0);
        self.buffers.push(Slot {
            id,
            extent,
            rgba,
            submitted: false,
        });
        Ok(Framebuffer::new(id))
    }

    fn destroy_framebuffer(&mut self, framebuffer: Framebuffer) -> Result<(), Self::Error> {
        if self
            .open
            .is_some_and(|open| open.framebuffer == framebuffer)
        {
            self.open = None;
        }
        let index = self.slot_index(framebuffer)?;
        self.buffers.swap_remove(index);
        Ok(())
    }

    fn begin(&mut self, framebuffer: Framebuffer) -> Result<Pass, Self::Error> {
        if self.open.is_some() {
            return Err(SoftwareError::PassOpen);
        }
        let index = self.slot_index(framebuffer)?;
        self.buffers[index].submitted = false;
        self.open = Some(OpenPass {
            framebuffer,
            ended: false,
            blend: Blend::Over,
        });
        Ok(Pass::new(framebuffer))
    }

    fn blend(&mut self, pass: Pass, blend: Blend) -> Result<(), Self::Error> {
        let Some(open) = self.open.as_mut() else {
            return Err(SoftwareError::NoPass);
        };
        if open.framebuffer != pass.framebuffer() {
            return Err(SoftwareError::NoPass);
        }
        if open.ended {
            return Err(SoftwareError::Ended);
        }
        open.blend = blend;
        Ok(())
    }

    fn clear(&mut self, pass: Pass, color: ClearColor) -> Result<(), Self::Error> {
        self.require_drawing(pass)?;
        let index = self.slot_index(pass.framebuffer())?;
        fill_rgba(&mut self.buffers[index].rgba, color.rgba);
        Ok(())
    }

    fn draw(&mut self, pass: Pass, source: TexelSource<'_>) -> Result<(), Self::Error> {
        self.require_drawing(pass)?;
        let TexelSource::Rgba { extent, bytes } = source else {
            return Err(SoftwareError::PlatformBuffer);
        };
        let index = self.slot_index(pass.framebuffer())?;
        let slot = &self.buffers[index];
        if slot.extent != extent {
            return Err(SoftwareError::Size);
        }
        let expected = slot.extent.rgba_len().ok_or(SoftwareError::Extent)?;
        if bytes.len() != expected {
            return Err(SoftwareError::Bytes);
        }
        if self.open.map(|open| open.blend) != Some(Blend::Over) {
            return Err(SoftwareError::NoPass);
        }
        blend_over(&mut self.buffers[index].rgba, bytes);
        Ok(())
    }

    fn end(&mut self, pass: Pass) -> Result<(), Self::Error> {
        let Some(open) = self.open.as_mut() else {
            return Err(SoftwareError::NoPass);
        };
        if open.framebuffer != pass.framebuffer() {
            return Err(SoftwareError::NoPass);
        }
        if open.ended {
            return Err(SoftwareError::Ended);
        }
        open.ended = true;
        Ok(())
    }

    fn submit(&mut self, pass: Pass) -> Result<(), Self::Error> {
        let Some(open) = self.open else {
            return Err(SoftwareError::NoPass);
        };
        if open.framebuffer != pass.framebuffer() {
            return Err(SoftwareError::NoPass);
        }
        if !open.ended {
            return Err(SoftwareError::NotEnded);
        }
        let index = self.slot_index(pass.framebuffer())?;
        self.buffers[index].submitted = true;
        self.open = None;
        Ok(())
    }

    fn read_rgba(&self, framebuffer: Framebuffer) -> Result<&[u8], Self::Error> {
        let index = self.slot_index(framebuffer)?;
        let slot = &self.buffers[index];
        if !slot.submitted {
            return Err(SoftwareError::NotSubmitted);
        }
        Ok(&slot.rgba)
    }
}

fn fill_rgba(destination: &mut [u8], pixel: [u8; 4]) {
    for chunk in destination.chunks_exact_mut(4) {
        chunk.copy_from_slice(&pixel);
    }
}

/// `output = source + destination * (1 - source.alpha)` on each channel.
///
/// Alpha is the stored byte divided by 255, with the remainder truncated.
/// A channel past 255 is clamped. The bytes are not premultiplied first.
fn blend_over(destination: &mut [u8], source: &[u8]) {
    for (dst, src) in destination.chunks_exact_mut(4).zip(source.chunks_exact(4)) {
        let Ok(src) = <[u8; 4]>::try_from(src) else {
            continue;
        };
        let Ok(current) = <[u8; 4]>::try_from(&dst[..]) else {
            continue;
        };
        let alpha = src[3];
        let mut mixed = [0; 4];
        for channel in 0..4 {
            mixed[channel] = blend_channel(src[channel], current[channel], alpha);
        }
        dst.copy_from_slice(&mixed);
    }
}

fn blend_channel(source: u8, destination: u8, source_alpha: u8) -> u8 {
    let inverse = u16::from(255_u8.saturating_sub(source_alpha));
    let mixed = u16::from(source) + u16::from(destination) * inverse / 255;
    u8::try_from(mixed.min(255)).unwrap_or(255)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draw_pixel(onto: [u8; 4], source: [u8; 4]) -> [u8; 4] {
        let mut device = SoftwareDevice::new();
        let extent = Extent {
            width: 1,
            height: 1,
        };
        let framebuffer = device.create_framebuffer(extent).unwrap();
        let pass = device.begin(framebuffer).unwrap();
        device.blend(pass, Blend::Over).unwrap();
        device.clear(pass, ClearColor { rgba: onto }).unwrap();
        device
            .draw(
                pass,
                TexelSource::Rgba {
                    extent,
                    bytes: &source,
                },
            )
            .unwrap();
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        let mut out = [0; 4];
        out.copy_from_slice(device.read_rgba(framebuffer).unwrap());
        out
    }

    #[test]
    fn an_opaque_pixel_replaces_the_framebuffer() {
        assert_eq!(
            draw_pixel([1, 2, 3, 4], [10, 20, 30, 255]),
            [10, 20, 30, 255]
        );
    }

    #[test]
    fn over_blends_each_channel_with_truncated_alpha() {
        // source alpha 128 → inverse 127. 200 * 127 / 255 = 99.
        assert_eq!(
            draw_pixel([0, 0, 200, 255], [100, 0, 0, 128]),
            [100, 0, 99, 255]
        );
        assert_eq!(draw_pixel([1, 2, 3, 4], [10, 20, 30, 0]), [11, 22, 33, 4]);
    }

    #[test]
    fn a_platform_buffer_is_refused_and_the_clear_stays() {
        let mut device = SoftwareDevice::new();
        let extent = Extent {
            width: 1,
            height: 1,
        };
        let framebuffer = device.create_framebuffer(extent).unwrap();
        let pass = device.begin(framebuffer).unwrap();
        device
            .clear(pass, ClearColor { rgba: [4, 5, 6, 7] })
            .unwrap();
        let error = device
            .draw(pass, TexelSource::Platform { extent, token: 0 })
            .unwrap_err();
        assert_eq!(error, SoftwareError::PlatformBuffer);
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap(), &[4, 5, 6, 7]);
    }

    #[test]
    fn a_short_buffer_and_a_mismatched_size_do_not_write() {
        let mut device = SoftwareDevice::new();
        let extent = Extent {
            width: 1,
            height: 1,
        };
        let framebuffer = device.create_framebuffer(extent).unwrap();
        let pass = device.begin(framebuffer).unwrap();
        device
            .clear(pass, ClearColor { rgba: [1, 1, 1, 1] })
            .unwrap();
        assert_eq!(
            device
                .draw(
                    pass,
                    TexelSource::Rgba {
                        extent,
                        bytes: &[1, 2, 3],
                    },
                )
                .unwrap_err(),
            SoftwareError::Bytes
        );
        assert_eq!(
            device
                .draw(
                    pass,
                    TexelSource::Rgba {
                        extent: Extent {
                            width: 2,
                            height: 1,
                        },
                        bytes: &[0, 0, 0, 0, 0, 0, 0, 0],
                    },
                )
                .unwrap_err(),
            SoftwareError::Size
        );
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap(), &[1, 1, 1, 1]);
    }

    #[test]
    fn an_edge_past_the_limit_is_refused() {
        let mut device = SoftwareDevice::new();
        let error = device
            .create_framebuffer(Extent {
                width: cutlass_gpu::MAX_EDGE + 1,
                height: 1,
            })
            .unwrap_err();
        assert_eq!(error, SoftwareError::Extent);
    }

    #[test]
    fn read_before_submit_is_refused_and_a_second_begin_is_refused() {
        let mut device = SoftwareDevice::new();
        let extent = Extent {
            width: 1,
            height: 1,
        };
        let framebuffer = device.create_framebuffer(extent).unwrap();
        let pass = device.begin(framebuffer).unwrap();
        assert_eq!(
            device.begin(framebuffer).unwrap_err(),
            SoftwareError::PassOpen
        );
        device.end(pass).unwrap();
        assert_eq!(
            device.read_rgba(framebuffer).unwrap_err(),
            SoftwareError::NotSubmitted
        );
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap().len(), 4);
    }
}
