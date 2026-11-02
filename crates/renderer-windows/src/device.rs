//! The [`cutlass_gpu::Device`] over CPU buffers.
//!
//! Draw blends RGBA bytes into the framebuffer. A platform buffer is refused
//! and the framebuffer is left unchanged.

use crate::error::WindowsError;
use cutlass_gpu::{Blend, ClearColor, Device, Extent, Framebuffer, Pass, TexelSource};

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

/// One Windows device. It owns its framebuffer bytes.
#[derive(Debug)]
pub struct WindowsDevice {
    next_id: u64,
    buffers: Vec<Slot>,
    open: Option<OpenPass>,
}

impl WindowsDevice {
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: 1,
            buffers: Vec::new(),
            open: None,
        }
    }

    fn slot_index(&self, framebuffer: Framebuffer) -> Result<usize, WindowsError> {
        self.buffers
            .iter()
            .position(|slot| slot.id == framebuffer.id())
            .ok_or(WindowsError::MissingFramebuffer)
    }

    fn require_drawing(&self, pass: Pass) -> Result<(), WindowsError> {
        match self.open {
            Some(open) if open.framebuffer == pass.framebuffer() && !open.ended => Ok(()),
            Some(open) if open.framebuffer == pass.framebuffer() => Err(WindowsError::Ended),
            _ => Err(WindowsError::NoPass),
        }
    }
}

impl Default for WindowsDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl Device for WindowsDevice {
    type Error = WindowsError;

    fn create_framebuffer(&mut self, extent: Extent) -> Result<Framebuffer, Self::Error> {
        let len = extent.rgba_len().ok_or(WindowsError::Extent)?;
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).ok_or(WindowsError::Extent)?;
        let mut rgba = Vec::new();
        rgba.try_reserve_exact(len)
            .map_err(|_| WindowsError::Extent)?;
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
            return Err(WindowsError::PassOpen);
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
            return Err(WindowsError::NoPass);
        };
        if open.framebuffer != pass.framebuffer() {
            return Err(WindowsError::NoPass);
        }
        if open.ended {
            return Err(WindowsError::Ended);
        }
        open.blend = blend;
        Ok(())
    }

    fn clear(&mut self, pass: Pass, color: ClearColor) -> Result<(), Self::Error> {
        self.require_drawing(pass)?;
        let index = self.slot_index(pass.framebuffer())?;
        crate::blend::fill_rgba(&mut self.buffers[index].rgba, color.rgba);
        Ok(())
    }

    fn draw(&mut self, pass: Pass, source: TexelSource<'_>) -> Result<(), Self::Error> {
        self.require_drawing(pass)?;
        let TexelSource::Rgba { extent, bytes } = source else {
            return Err(WindowsError::PlatformBuffer);
        };
        let index = self.slot_index(pass.framebuffer())?;
        let slot = &self.buffers[index];
        if slot.extent != extent {
            return Err(WindowsError::Size);
        }
        let expected = slot.extent.rgba_len().ok_or(WindowsError::Extent)?;
        if bytes.len() != expected {
            return Err(WindowsError::Bytes);
        }
        if self.open.map(|open| open.blend) != Some(Blend::Over) {
            return Err(WindowsError::NoPass);
        }
        crate::blend::blend_over(&mut self.buffers[index].rgba, bytes);
        Ok(())
    }

    fn end(&mut self, pass: Pass) -> Result<(), Self::Error> {
        let Some(open) = self.open.as_mut() else {
            return Err(WindowsError::NoPass);
        };
        if open.framebuffer != pass.framebuffer() {
            return Err(WindowsError::NoPass);
        }
        if open.ended {
            return Err(WindowsError::Ended);
        }
        open.ended = true;
        Ok(())
    }

    fn submit(&mut self, pass: Pass) -> Result<(), Self::Error> {
        let Some(open) = self.open else {
            return Err(WindowsError::NoPass);
        };
        if open.framebuffer != pass.framebuffer() {
            return Err(WindowsError::NoPass);
        }
        if !open.ended {
            return Err(WindowsError::NotEnded);
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
            return Err(WindowsError::NotSubmitted);
        }
        Ok(&slot.rgba)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel() -> Extent {
        Extent {
            width: 1,
            height: 1,
        }
    }

    fn device_with_pixel() -> (WindowsDevice, Framebuffer) {
        let mut device = WindowsDevice::new();
        let framebuffer = device.create_framebuffer(pixel()).unwrap();
        (device, framebuffer)
    }

    #[test]
    fn an_edge_past_the_limit_is_refused() {
        let mut device = WindowsDevice::new();
        assert_eq!(
            device
                .create_framebuffer(Extent {
                    width: cutlass_gpu::MAX_EDGE + 1,
                    height: 1,
                })
                .unwrap_err(),
            WindowsError::Extent
        );
        assert_eq!(
            device
                .create_framebuffer(Extent {
                    width: 1,
                    height: cutlass_gpu::MAX_EDGE + 1,
                })
                .unwrap_err(),
            WindowsError::Extent
        );
        assert_eq!(device.create_framebuffer(pixel()).unwrap().id(), 1);
    }

    #[test]
    fn a_missing_framebuffer_is_refused() {
        let mut device = WindowsDevice::new();
        let missing = Framebuffer::new(4);
        assert_eq!(
            device.begin(missing).unwrap_err(),
            WindowsError::MissingFramebuffer
        );
        assert_eq!(
            device.destroy_framebuffer(missing).unwrap_err(),
            WindowsError::MissingFramebuffer
        );
        assert_eq!(
            device.read_rgba(missing).unwrap_err(),
            WindowsError::MissingFramebuffer
        );
        let framebuffer = device.create_framebuffer(pixel()).unwrap();
        let pass = device.begin(framebuffer).unwrap();
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
    }

    #[test]
    fn a_zero_framebuffer_id_is_missing() {
        let mut device = WindowsDevice::new();
        let missing = Framebuffer::new(0);
        assert_eq!(
            device.begin(missing).unwrap_err(),
            WindowsError::MissingFramebuffer
        );
        assert_eq!(
            device.destroy_framebuffer(missing).unwrap_err(),
            WindowsError::MissingFramebuffer
        );
        assert_eq!(
            device.read_rgba(missing).unwrap_err(),
            WindowsError::MissingFramebuffer
        );
    }

    #[test]
    fn begin_while_a_pass_is_open_is_refused() {
        let (mut device, framebuffer) = device_with_pixel();
        let pass = device.begin(framebuffer).unwrap();
        device
            .clear(pass, ClearColor { rgba: [1, 2, 3, 4] })
            .unwrap();
        assert_eq!(
            device.begin(framebuffer).unwrap_err(),
            WindowsError::PassOpen
        );
        assert_eq!(
            device.begin(Framebuffer::new(99)).unwrap_err(),
            WindowsError::PassOpen
        );
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn a_different_pass_is_not_open() {
        let mut device = WindowsDevice::new();
        let framebuffer = device.create_framebuffer(pixel()).unwrap();
        let other = device.create_framebuffer(pixel()).unwrap();
        let pass = device.begin(framebuffer).unwrap();
        device
            .clear(pass, ClearColor { rgba: [4, 3, 2, 1] })
            .unwrap();
        let wrong = Pass::new(other);
        assert_eq!(
            device.blend(wrong, Blend::Over).unwrap_err(),
            WindowsError::NoPass
        );
        assert_eq!(
            device
                .clear(wrong, ClearColor { rgba: [9, 9, 9, 9] })
                .unwrap_err(),
            WindowsError::NoPass
        );
        assert_eq!(
            device
                .draw(
                    wrong,
                    TexelSource::Platform {
                        extent: pixel(),
                        token: 1,
                    },
                )
                .unwrap_err(),
            WindowsError::NoPass
        );
        assert_eq!(device.end(wrong).unwrap_err(), WindowsError::NoPass);
        assert_eq!(device.submit(wrong).unwrap_err(), WindowsError::NoPass);
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap(), &[4, 3, 2, 1]);
    }

    #[test]
    fn drawing_after_end_is_refused() {
        let (mut device, framebuffer) = device_with_pixel();
        let pass = device.begin(framebuffer).unwrap();
        device
            .clear(pass, ClearColor { rgba: [2, 2, 2, 2] })
            .unwrap();
        device.end(pass).unwrap();
        assert_eq!(
            device
                .draw(
                    pass,
                    TexelSource::Rgba {
                        extent: pixel(),
                        bytes: &[0, 0, 0, 255],
                    },
                )
                .unwrap_err(),
            WindowsError::Ended
        );
        assert_eq!(
            device
                .clear(pass, ClearColor { rgba: [1, 1, 1, 1] })
                .unwrap_err(),
            WindowsError::Ended
        );
        assert_eq!(
            device.blend(pass, Blend::Over).unwrap_err(),
            WindowsError::Ended
        );
        assert_eq!(device.end(pass).unwrap_err(), WindowsError::Ended);
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap(), &[2, 2, 2, 2]);
    }

    #[test]
    fn submit_before_end_is_refused() {
        let (mut device, framebuffer) = device_with_pixel();
        let pass = device.begin(framebuffer).unwrap();
        device
            .clear(pass, ClearColor { rgba: [3, 3, 3, 3] })
            .unwrap();
        assert_eq!(device.submit(pass).unwrap_err(), WindowsError::NotEnded);
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap(), &[3, 3, 3, 3]);
    }

    #[test]
    fn a_second_submit_is_refused() {
        let (mut device, framebuffer) = device_with_pixel();
        let pass = device.begin(framebuffer).unwrap();
        device
            .clear(pass, ClearColor { rgba: [6, 5, 4, 3] })
            .unwrap();
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.submit(pass).unwrap_err(), WindowsError::NoPass);
        assert_eq!(device.read_rgba(framebuffer).unwrap(), &[6, 5, 4, 3]);
    }

    #[test]
    fn read_before_submit_is_refused() {
        let (mut device, framebuffer) = device_with_pixel();
        let pass = device.begin(framebuffer).unwrap();
        device
            .clear(pass, ClearColor { rgba: [7, 7, 7, 7] })
            .unwrap();
        assert_eq!(
            device.read_rgba(framebuffer).unwrap_err(),
            WindowsError::NotSubmitted
        );
        device.end(pass).unwrap();
        assert_eq!(
            device.read_rgba(framebuffer).unwrap_err(),
            WindowsError::NotSubmitted
        );
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap(), &[7, 7, 7, 7]);
    }

    #[test]
    fn begin_clears_the_submitted_flag() {
        let (mut device, framebuffer) = device_with_pixel();
        let pass = device.begin(framebuffer).unwrap();
        device.blend(pass, Blend::Over).unwrap();
        device
            .clear(pass, ClearColor { rgba: [8, 7, 6, 5] })
            .unwrap();
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap(), &[8, 7, 6, 5]);

        let pass = device.begin(framebuffer).unwrap();
        assert_eq!(
            device.read_rgba(framebuffer).unwrap_err(),
            WindowsError::NotSubmitted
        );
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap(), &[8, 7, 6, 5]);
    }

    #[test]
    fn a_platform_buffer_is_refused_and_the_clear_stays() {
        let (mut device, framebuffer) = device_with_pixel();
        let pass = device.begin(framebuffer).unwrap();
        device
            .clear(pass, ClearColor { rgba: [4, 5, 6, 7] })
            .unwrap();
        let error = device
            .draw(
                pass,
                TexelSource::Platform {
                    extent: pixel(),
                    token: 0,
                },
            )
            .unwrap_err();
        assert_eq!(error, WindowsError::PlatformBuffer);
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap(), &[4, 5, 6, 7]);
    }

    #[test]
    fn a_bad_picture_does_not_write() {
        let (mut device, framebuffer) = device_with_pixel();
        let pass = device.begin(framebuffer).unwrap();
        device
            .clear(pass, ClearColor { rgba: [1, 1, 1, 1] })
            .unwrap();
        assert_eq!(
            device
                .draw(
                    pass,
                    TexelSource::Rgba {
                        extent: pixel(),
                        bytes: &[1, 2, 3],
                    },
                )
                .unwrap_err(),
            WindowsError::Bytes
        );
        assert_eq!(
            device
                .draw(
                    pass,
                    TexelSource::Rgba {
                        extent: pixel(),
                        bytes: &[1, 2, 3, 4, 5],
                    },
                )
                .unwrap_err(),
            WindowsError::Bytes
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
            WindowsError::Size
        );
        assert_eq!(
            device
                .draw(
                    pass,
                    TexelSource::Rgba {
                        extent: Extent {
                            width: 2,
                            height: 2,
                        },
                        bytes: &[9, 9, 9, 9, 9, 9, 9],
                    },
                )
                .unwrap_err(),
            WindowsError::Size
        );
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap(), &[1, 1, 1, 1]);
    }

    #[test]
    fn destroy_of_the_open_framebuffer_closes_the_pass() {
        let (mut device, framebuffer) = device_with_pixel();
        let pass = device.begin(framebuffer).unwrap();
        device.destroy_framebuffer(framebuffer).unwrap();
        assert_eq!(device.submit(pass).unwrap_err(), WindowsError::NoPass);
    }

    #[test]
    fn destroy_of_another_framebuffer_leaves_the_open_pass() {
        let mut device = WindowsDevice::new();
        let live = device.create_framebuffer(pixel()).unwrap();
        let other = device.create_framebuffer(pixel()).unwrap();
        let pass = device.begin(live).unwrap();
        device.destroy_framebuffer(other).unwrap();
        device
            .clear(pass, ClearColor { rgba: [9, 8, 7, 6] })
            .unwrap();
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(live).unwrap(), &[9, 8, 7, 6]);
        assert_eq!(
            device.read_rgba(other).unwrap_err(),
            WindowsError::MissingFramebuffer
        );
    }

    #[test]
    fn a_zero_size_framebuffer_round_trips_empty_bytes() {
        let mut device = WindowsDevice::new();
        let extent = Extent {
            width: 0,
            height: 0,
        };
        let framebuffer = device.create_framebuffer(extent).unwrap();
        let pass = device.begin(framebuffer).unwrap();
        device.blend(pass, Blend::Over).unwrap();
        device
            .clear(pass, ClearColor { rgba: [9, 9, 9, 9] })
            .unwrap();
        device
            .draw(pass, TexelSource::Rgba { extent, bytes: &[] })
            .unwrap();
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        let bytes = device.read_rgba(framebuffer).unwrap();
        let empty: &[u8] = &[];
        assert_eq!(bytes, empty);
    }

    #[test]
    fn the_first_framebuffer_id_is_one() {
        let mut device = WindowsDevice::default();
        let first = device.create_framebuffer(pixel()).unwrap();
        let second = device
            .create_framebuffer(Extent {
                width: 0,
                height: 0,
            })
            .unwrap();
        assert_eq!(first.id(), 1);
        assert_eq!(second.id(), 2);
    }

    #[test]
    fn each_error_displays_its_sentence() {
        let sentences = [
            (WindowsError::Extent, "the frame size is past the limit"),
            (
                WindowsError::Size,
                "the picture size does not match the framebuffer",
            ),
            (
                WindowsError::Bytes,
                "the picture byte length does not match its size",
            ),
            (
                WindowsError::PlatformBuffer,
                "the windows device cannot sample a platform buffer",
            ),
            (
                WindowsError::MissingFramebuffer,
                "the framebuffer is not on this device",
            ),
            (WindowsError::PassOpen, "a pass is already open"),
            (WindowsError::NoPass, "this pass is not open"),
            (WindowsError::Ended, "this pass has ended"),
            (WindowsError::NotEnded, "this pass has not ended"),
            (
                WindowsError::NotSubmitted,
                "the frame has not been submitted",
            ),
            (
                WindowsError::GpuUnsupported,
                "Direct3D is not available on this host",
            ),
            (
                WindowsError::UnknownResource,
                "the platform token is not a Direct3D resource on this device",
            ),
        ];
        for (error, sentence) in sentences {
            assert_eq!(error.to_string(), sentence);
        }
    }
}
