//! Portable framebuffer lifecycle for [`MetalDevice`].
//!
//! One pass is open at a time. Draw samples tightly packed RGBA bytes.
//! A platform buffer is refused and the framebuffer is left unchanged.

use cutlass_gpu::{Blend, ClearColor, Device, Extent, Framebuffer, Pass, TexelSource};

use crate::error::DeviceError;

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

/// The device that owns portable CPU framebuffers.
///
/// A clip does not choose a backend. The device creates a framebuffer,
/// begins one pass, clears and draws, ends, submits, and reads the bytes
/// back. A platform buffer is refused and the stored bytes stay as they were.
#[derive(Debug)]
pub struct MetalDevice {
    next_id: u64,
    slots: Vec<Slot>,
    open: Option<OpenPass>,
}

impl MetalDevice {
    /// An empty device. The first framebuffer id is 1.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: 1,
            slots: Vec::new(),
            open: None,
        }
    }

    /// Width and height of a framebuffer on this device.
    pub fn framebuffer_extent(&self, framebuffer: Framebuffer) -> Result<Extent, DeviceError> {
        let index = self.slot_index(framebuffer)?;
        Ok(self.slots[index].extent)
    }

    fn slot_index(&self, framebuffer: Framebuffer) -> Result<usize, DeviceError> {
        self.slots
            .iter()
            .position(|slot| slot.id == framebuffer.id())
            .ok_or(DeviceError::MissingFramebuffer)
    }

    fn require_drawing(&self, pass: Pass) -> Result<(), DeviceError> {
        match self.open {
            Some(open) if open.framebuffer == pass.framebuffer() && !open.ended => Ok(()),
            Some(open) if open.framebuffer == pass.framebuffer() => Err(DeviceError::Ended),
            _ => Err(DeviceError::NoPass),
        }
    }
}

impl Default for MetalDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl Device for MetalDevice {
    type Error = DeviceError;

    fn create_framebuffer(&mut self, extent: Extent) -> Result<Framebuffer, Self::Error> {
        let len = extent.rgba_len().ok_or(DeviceError::Extent)?;
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).ok_or(DeviceError::Extent)?;
        let mut rgba = Vec::new();
        rgba.try_reserve_exact(len)
            .map_err(|_| DeviceError::Extent)?;
        rgba.resize(len, 0);
        self.slots.push(Slot {
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
        self.slots.swap_remove(index);
        Ok(())
    }

    fn begin(&mut self, framebuffer: Framebuffer) -> Result<Pass, Self::Error> {
        if self.open.is_some() {
            return Err(DeviceError::PassOpen);
        }
        let index = self.slot_index(framebuffer)?;
        self.slots[index].submitted = false;
        self.open = Some(OpenPass {
            framebuffer,
            ended: false,
            blend: Blend::Over,
        });
        Ok(Pass::new(framebuffer))
    }

    fn blend(&mut self, pass: Pass, blend: Blend) -> Result<(), Self::Error> {
        let Some(open) = self.open.as_mut() else {
            return Err(DeviceError::NoPass);
        };
        if open.framebuffer != pass.framebuffer() {
            return Err(DeviceError::NoPass);
        }
        if open.ended {
            return Err(DeviceError::Ended);
        }
        open.blend = blend;
        Ok(())
    }

    fn clear(&mut self, pass: Pass, color: ClearColor) -> Result<(), Self::Error> {
        self.require_drawing(pass)?;
        let index = self.slot_index(pass.framebuffer())?;
        crate::blend::fill_rgba(&mut self.slots[index].rgba, color.rgba);
        Ok(())
    }

    fn draw(&mut self, pass: Pass, source: TexelSource<'_>) -> Result<(), Self::Error> {
        self.require_drawing(pass)?;
        let (extent, bytes) = match source {
            TexelSource::Rgba { extent, bytes } => (extent, bytes),
            TexelSource::Platform { .. } => return Err(DeviceError::PlatformBuffer),
        };
        let index = self.slot_index(pass.framebuffer())?;
        let slot_extent = self.slots[index].extent;
        if slot_extent != extent {
            return Err(DeviceError::Size);
        }
        let expected = slot_extent.rgba_len().ok_or(DeviceError::Extent)?;
        if bytes.len() != expected {
            return Err(DeviceError::Bytes);
        }
        if self.open.map(|open| open.blend) != Some(Blend::Over) {
            return Err(DeviceError::NoPass);
        }
        crate::blend::blend_over(&mut self.slots[index].rgba, bytes);
        Ok(())
    }

    fn end(&mut self, pass: Pass) -> Result<(), Self::Error> {
        let Some(open) = self.open.as_mut() else {
            return Err(DeviceError::NoPass);
        };
        if open.framebuffer != pass.framebuffer() {
            return Err(DeviceError::NoPass);
        }
        if open.ended {
            return Err(DeviceError::Ended);
        }
        open.ended = true;
        Ok(())
    }

    fn submit(&mut self, pass: Pass) -> Result<(), Self::Error> {
        let Some(open) = self.open else {
            return Err(DeviceError::NoPass);
        };
        if open.framebuffer != pass.framebuffer() {
            return Err(DeviceError::NoPass);
        }
        if !open.ended {
            return Err(DeviceError::NotEnded);
        }
        let index = self.slot_index(pass.framebuffer())?;
        self.slots[index].submitted = true;
        self.open = None;
        Ok(())
    }

    fn read_rgba(&self, framebuffer: Framebuffer) -> Result<&[u8], Self::Error> {
        let index = self.slot_index(framebuffer)?;
        let slot = &self.slots[index];
        if !slot.submitted {
            return Err(DeviceError::NotSubmitted);
        }
        Ok(&slot.rgba)
    }
}

#[cfg(test)]
mod tests {
    use cutlass_gpu::{
        Blend, ClearColor, Device, Extent, Framebuffer, MAX_EDGE, Pass, TexelSource,
    };

    use super::MetalDevice;
    use crate::error::DeviceError;

    fn pixel() -> Extent {
        Extent {
            width: 1,
            height: 1,
        }
    }

    fn create_pixel(device: &mut MetalDevice) -> Framebuffer {
        device.create_framebuffer(pixel()).unwrap()
    }

    fn submit_clear(device: &mut MetalDevice, framebuffer: Framebuffer, rgba: [u8; 4]) {
        let pass = device.begin(framebuffer).unwrap();
        device.clear(pass, ClearColor { rgba }).unwrap();
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
    }

    fn assert_missing(device: &mut MetalDevice, framebuffer: Framebuffer) {
        assert_eq!(
            device.begin(framebuffer).unwrap_err(),
            DeviceError::MissingFramebuffer
        );
        assert_eq!(
            device.destroy_framebuffer(framebuffer).unwrap_err(),
            DeviceError::MissingFramebuffer
        );
        assert_eq!(
            device.read_rgba(framebuffer).unwrap_err(),
            DeviceError::MissingFramebuffer
        );
        assert_eq!(
            device.framebuffer_extent(framebuffer).unwrap_err(),
            DeviceError::MissingFramebuffer
        );
    }

    fn displayed(error: DeviceError) -> String {
        let as_error: &dyn std::error::Error = &error;
        as_error.to_string()
    }

    #[test]
    fn an_edge_past_the_limit_creates_no_framebuffer() {
        let mut device = MetalDevice::new();
        let refused = [
            Extent {
                width: MAX_EDGE + 1,
                height: 1,
            },
            Extent {
                width: 1,
                height: MAX_EDGE + 1,
            },
            Extent {
                width: MAX_EDGE + 1,
                height: 0,
            },
            Extent {
                width: 0,
                height: MAX_EDGE + 1,
            },
        ];
        for extent in refused {
            assert_eq!(
                device.create_framebuffer(extent).unwrap_err(),
                DeviceError::Extent
            );
        }
        let missing = Framebuffer::new(1);
        assert_eq!(
            device.framebuffer_extent(missing).unwrap_err(),
            DeviceError::MissingFramebuffer
        );
        assert_eq!(
            device.begin(missing).unwrap_err(),
            DeviceError::MissingFramebuffer
        );

        let edges = [
            Extent {
                width: MAX_EDGE,
                height: 1,
            },
            Extent {
                width: 1,
                height: MAX_EDGE,
            },
        ];
        for (offset, extent) in edges.into_iter().enumerate() {
            let framebuffer = device.create_framebuffer(extent).unwrap();
            let id = u64::try_from(offset).unwrap() + 1;
            assert_eq!(framebuffer.id(), id);
            assert_eq!(device.framebuffer_extent(framebuffer).unwrap(), extent);
            let pass = device.begin(framebuffer).unwrap();
            device.end(pass).unwrap();
            device.submit(pass).unwrap();
            let bytes = device.read_rgba(framebuffer).unwrap();
            let len = usize::try_from(extent.width).unwrap()
                * usize::try_from(extent.height).unwrap()
                * 4;
            assert_eq!(bytes.len(), len);
            assert!(bytes.iter().all(|byte| *byte == 0));
        }
    }

    #[test]
    fn an_empty_framebuffer_submits_an_empty_readback() {
        let mut device = MetalDevice::new();
        let extent = Extent {
            width: 0,
            height: 0,
        };
        let framebuffer = device.create_framebuffer(extent).unwrap();
        assert_eq!(device.framebuffer_extent(framebuffer).unwrap(), extent);
        let pass = device.begin(framebuffer).unwrap();
        device
            .clear(pass, ClearColor { rgba: [9, 8, 7, 6] })
            .unwrap();
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap().len(), 0);
    }

    #[test]
    fn a_missing_framebuffer_is_refused() {
        let mut device = MetalDevice::new();
        assert_missing(&mut device, Framebuffer::new(1));
        let live = create_pixel(&mut device);
        assert_missing(&mut device, Framebuffer::new(live.id() + 1));
        assert_eq!(device.framebuffer_extent(live).unwrap(), pixel());
    }

    #[test]
    fn destroy_removes_the_framebuffer() {
        let mut device = MetalDevice::new();
        let framebuffer = create_pixel(&mut device);
        device.destroy_framebuffer(framebuffer).unwrap();
        assert_eq!(
            device.begin(framebuffer).unwrap_err(),
            DeviceError::MissingFramebuffer
        );
        let replacement = create_pixel(&mut device);
        assert_ne!(replacement.id(), framebuffer.id());
        assert_eq!(
            device.begin(framebuffer).unwrap_err(),
            DeviceError::MissingFramebuffer
        );
        assert_eq!(
            device.begin(replacement).unwrap().framebuffer().id(),
            replacement.id()
        );
    }

    #[test]
    fn a_second_begin_is_refused_while_the_pass_is_open() {
        let mut device = MetalDevice::new();
        let first = create_pixel(&mut device);
        let second = create_pixel(&mut device);
        let pass = device.begin(first).unwrap();
        assert_eq!(device.begin(first).unwrap_err(), DeviceError::PassOpen);
        assert_eq!(device.begin(second).unwrap_err(), DeviceError::PassOpen);
        assert_eq!(
            device.begin(Framebuffer::new(u64::MAX)).unwrap_err(),
            DeviceError::PassOpen
        );
        device.end(pass).unwrap();
        assert_eq!(device.begin(first).unwrap_err(), DeviceError::PassOpen);
        assert_eq!(device.begin(second).unwrap_err(), DeviceError::PassOpen);
        assert_eq!(
            device.begin(Framebuffer::new(u64::MAX)).unwrap_err(),
            DeviceError::PassOpen
        );
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(first).unwrap().len(), 4);
    }

    #[test]
    fn begin_leaves_another_framebuffer_unchanged() {
        let mut device = MetalDevice::new();
        let kept = create_pixel(&mut device);
        let opened = create_pixel(&mut device);
        submit_clear(&mut device, kept, [9, 8, 7, 6]);
        let before = device.read_rgba(kept).unwrap().to_vec();
        let pass = device.begin(opened).unwrap();
        assert_eq!(device.read_rgba(kept).unwrap(), before.as_slice());
        assert_eq!(
            device.read_rgba(opened).unwrap_err(),
            DeviceError::NotSubmitted
        );
        device
            .clear(pass, ClearColor { rgba: [1, 2, 3, 4] })
            .unwrap();
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(opened).unwrap(), &[1, 2, 3, 4]);
        assert_eq!(device.read_rgba(kept).unwrap(), before.as_slice());
    }

    #[test]
    fn a_pass_for_another_framebuffer_is_refused() {
        let mut device = MetalDevice::new();
        let target = create_pixel(&mut device);
        let other = create_pixel(&mut device);
        submit_clear(&mut device, other, [9, 9, 9, 9]);
        let pass = device.begin(target).unwrap();
        device
            .clear(pass, ClearColor { rgba: [4, 5, 6, 7] })
            .unwrap();
        let wrong = [Pass::new(other), Pass::new(Framebuffer::new(u64::MAX))];
        for foreign in wrong {
            assert_eq!(
                device.blend(foreign, Blend::Over).unwrap_err(),
                DeviceError::NoPass
            );
            assert_eq!(
                device
                    .clear(foreign, ClearColor { rgba: [1, 2, 3, 4] })
                    .unwrap_err(),
                DeviceError::NoPass
            );
            assert_eq!(
                device
                    .draw(
                        foreign,
                        TexelSource::Rgba {
                            extent: pixel(),
                            bytes: &[8, 7, 6, 255],
                        },
                    )
                    .unwrap_err(),
                DeviceError::NoPass
            );
            assert_eq!(device.end(foreign).unwrap_err(), DeviceError::NoPass);
            assert_eq!(device.submit(foreign).unwrap_err(), DeviceError::NoPass);
        }
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(target).unwrap(), &[4, 5, 6, 7]);
        assert_eq!(device.read_rgba(other).unwrap(), &[9, 9, 9, 9]);
    }

    #[test]
    fn drawing_after_end_is_refused() {
        let mut device = MetalDevice::new();
        let framebuffer = create_pixel(&mut device);
        let pass = device.begin(framebuffer).unwrap();
        device
            .clear(pass, ClearColor { rgba: [4, 5, 6, 7] })
            .unwrap();
        device.end(pass).unwrap();
        assert_eq!(
            device.blend(pass, Blend::Over).unwrap_err(),
            DeviceError::Ended
        );
        assert_eq!(
            device
                .clear(pass, ClearColor { rgba: [1, 1, 1, 1] })
                .unwrap_err(),
            DeviceError::Ended
        );
        assert_eq!(
            device
                .draw(
                    pass,
                    TexelSource::Rgba {
                        extent: pixel(),
                        bytes: &[9, 9, 9, 9],
                    },
                )
                .unwrap_err(),
            DeviceError::Ended
        );
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap(), &[4, 5, 6, 7]);
    }

    #[test]
    fn ending_twice_is_refused() {
        let mut device = MetalDevice::new();
        let framebuffer = create_pixel(&mut device);
        let pass = device.begin(framebuffer).unwrap();
        device.end(pass).unwrap();
        assert_eq!(device.end(pass).unwrap_err(), DeviceError::Ended);
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap().len(), 4);
    }

    #[test]
    fn submit_before_end_is_refused() {
        let mut device = MetalDevice::new();
        let framebuffer = create_pixel(&mut device);
        let pass = device.begin(framebuffer).unwrap();
        assert_eq!(device.submit(pass).unwrap_err(), DeviceError::NotEnded);
        assert_eq!(
            device.read_rgba(framebuffer).unwrap_err(),
            DeviceError::NotSubmitted
        );
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap().len(), 4);
    }

    #[test]
    fn submit_twice_is_refused() {
        let mut device = MetalDevice::new();
        let framebuffer = create_pixel(&mut device);
        let pass = device.begin(framebuffer).unwrap();
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap().len(), 4);
        assert_eq!(device.submit(pass).unwrap_err(), DeviceError::NoPass);
        assert_eq!(device.read_rgba(framebuffer).unwrap().len(), 4);
    }

    #[test]
    fn read_before_submit_is_refused() {
        let mut device = MetalDevice::new();
        let framebuffer = create_pixel(&mut device);
        assert_eq!(
            device.read_rgba(framebuffer).unwrap_err(),
            DeviceError::NotSubmitted
        );
        let pass = device.begin(framebuffer).unwrap();
        assert_eq!(
            device.read_rgba(framebuffer).unwrap_err(),
            DeviceError::NotSubmitted
        );
        device.end(pass).unwrap();
        assert_eq!(
            device.read_rgba(framebuffer).unwrap_err(),
            DeviceError::NotSubmitted
        );
    }

    #[test]
    fn begin_clears_the_submitted_readback() {
        let mut device = MetalDevice::new();
        let framebuffer = create_pixel(&mut device);
        submit_clear(&mut device, framebuffer, [3, 3, 3, 3]);
        assert_eq!(device.read_rgba(framebuffer).unwrap(), &[3, 3, 3, 3]);
        let pass = device.begin(framebuffer).unwrap();
        assert_eq!(
            device.read_rgba(framebuffer).unwrap_err(),
            DeviceError::NotSubmitted
        );
        device
            .clear(pass, ClearColor { rgba: [8, 8, 8, 8] })
            .unwrap();
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap(), &[8, 8, 8, 8]);
    }

    #[test]
    fn destroying_the_open_framebuffer_allows_another_begin() {
        let mut device = MetalDevice::new();
        let open = create_pixel(&mut device);
        let other = create_pixel(&mut device);
        let pass = device.begin(open).unwrap();
        device.destroy_framebuffer(open).unwrap();
        assert_eq!(device.end(pass).unwrap_err(), DeviceError::NoPass);
        assert_eq!(
            device.begin(open).unwrap_err(),
            DeviceError::MissingFramebuffer
        );
        let next = device.begin(other).unwrap();
        device
            .clear(next, ClearColor { rgba: [2, 2, 2, 2] })
            .unwrap();
        device.end(next).unwrap();
        device.submit(next).unwrap();
        assert_eq!(device.read_rgba(other).unwrap(), &[2, 2, 2, 2]);

        let ended = create_pixel(&mut device);
        let spare = create_pixel(&mut device);
        let pass = device.begin(ended).unwrap();
        device.end(pass).unwrap();
        device.destroy_framebuffer(ended).unwrap();
        assert_eq!(device.submit(pass).unwrap_err(), DeviceError::NoPass);
        assert_eq!(
            device.begin(ended).unwrap_err(),
            DeviceError::MissingFramebuffer
        );
        let recovered = device.begin(spare).unwrap();
        device.end(recovered).unwrap();
        device.submit(recovered).unwrap();
        assert_eq!(device.read_rgba(spare).unwrap().len(), 4);
    }

    #[test]
    fn destroying_another_framebuffer_leaves_the_pass_usable() {
        let mut device = MetalDevice::new();
        let open = create_pixel(&mut device);
        let other = create_pixel(&mut device);
        let pass = device.begin(open).unwrap();
        device.destroy_framebuffer(other).unwrap();
        assert_eq!(
            device.framebuffer_extent(other).unwrap_err(),
            DeviceError::MissingFramebuffer
        );
        assert_eq!(
            device.read_rgba(other).unwrap_err(),
            DeviceError::MissingFramebuffer
        );
        device.blend(pass, Blend::Over).unwrap();
        device
            .clear(pass, ClearColor { rgba: [5, 4, 3, 2] })
            .unwrap();
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(open).unwrap(), &[5, 4, 3, 2]);
        assert_eq!(device.framebuffer_extent(open).unwrap(), pixel());
    }

    #[test]
    fn a_platform_buffer_is_refused_and_the_clear_stays() {
        let mut device = MetalDevice::new();
        let framebuffer = create_pixel(&mut device);
        let pass = device.begin(framebuffer).unwrap();
        device
            .clear(pass, ClearColor { rgba: [4, 5, 6, 7] })
            .unwrap();
        let error = device
            .draw(
                pass,
                TexelSource::Platform {
                    extent: pixel(),
                    token: 11,
                },
            )
            .unwrap_err();
        assert_eq!(error, DeviceError::PlatformBuffer);
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap(), &[4, 5, 6, 7]);
    }

    #[test]
    fn a_short_buffer_and_a_mismatched_size_do_not_write() {
        let mut device = MetalDevice::new();
        let framebuffer = create_pixel(&mut device);
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
                        bytes: &[2, 3, 4],
                    },
                )
                .unwrap_err(),
            DeviceError::Bytes
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
                        bytes: &[9, 9, 9, 255, 9, 9, 9, 255],
                    },
                )
                .unwrap_err(),
            DeviceError::Size
        );
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap(), &[1, 1, 1, 1]);
    }

    #[test]
    fn bytes_longer_than_the_framebuffer_do_not_write() {
        let mut device = MetalDevice::new();
        let framebuffer = create_pixel(&mut device);
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
                        bytes: &[8, 7, 6, 255, 8, 7, 6, 255],
                    },
                )
                .unwrap_err(),
            DeviceError::Bytes
        );
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap(), &[1, 1, 1, 1]);
    }

    #[test]
    fn an_rgba_draw_is_accepted() {
        let mut device = MetalDevice::new();
        let framebuffer = create_pixel(&mut device);
        let pass = device.begin(framebuffer).unwrap();
        device.blend(pass, Blend::Over).unwrap();
        device
            .draw(
                pass,
                TexelSource::Rgba {
                    extent: pixel(),
                    bytes: &[2, 3, 4, 5],
                },
            )
            .unwrap();
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap().len(), 4);
    }

    #[test]
    fn device_error_display_is_the_contract_string() {
        let cases = [
            (DeviceError::Extent, "the frame size is past the limit"),
            (
                DeviceError::Size,
                "the picture size does not match the framebuffer",
            ),
            (
                DeviceError::Bytes,
                "the picture byte length does not match its size",
            ),
            (
                DeviceError::PlatformBuffer,
                "the portable path cannot sample a platform buffer",
            ),
            (
                DeviceError::MissingFramebuffer,
                "the framebuffer is not on this device",
            ),
            (DeviceError::PassOpen, "a pass is already open"),
            (DeviceError::NoPass, "this pass is not open"),
            (DeviceError::Ended, "this pass has ended"),
            (DeviceError::NotEnded, "this pass has not ended"),
            (
                DeviceError::NotSubmitted,
                "the frame has not been submitted",
            ),
            (
                DeviceError::Backend,
                "the metal device could not submit the frame",
            ),
        ];
        for (error, message) in cases {
            assert_eq!(displayed(error), message);
        }
    }

    #[test]
    fn two_submits_return_independent_buffers() {
        let mut device = MetalDevice::new();
        let first = create_pixel(&mut device);
        let second = create_pixel(&mut device);
        assert_eq!(first.id(), 1);
        assert_eq!(second.id(), 2);
        submit_clear(&mut device, first, [2, 3, 4, 5]);
        submit_clear(&mut device, second, [6, 7, 8, 9]);
        assert_eq!(device.read_rgba(first).unwrap(), &[2, 3, 4, 5]);
        assert_eq!(device.read_rgba(second).unwrap(), &[6, 7, 8, 9]);
    }

    #[test]
    fn the_default_device_submits_a_zero_filled_framebuffer() {
        let mut device = MetalDevice::default();
        let framebuffer = create_pixel(&mut device);
        assert_eq!(framebuffer.id(), 1);
        let pass = device.begin(framebuffer).unwrap();
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap(), &[0, 0, 0, 0]);
    }
}
