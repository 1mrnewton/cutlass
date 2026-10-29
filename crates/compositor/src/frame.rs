use std::fmt;

use cutlass_gpu::{ClearColor, Device, Extent, Framebuffer};
use cutlass_model::Micros;
use cutlass_picture::PictureSource;

use crate::blend::pass_blend;
use crate::seek::draw_ready;

/// The pixels submitted for one microsecond.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub extent: Extent,
    pub rgba: Vec<u8>,
}

/// Why a frame was not submitted.
#[derive(Debug, PartialEq, Eq)]
pub enum ComposeError<E> {
    /// The device refused the pass.
    Device(E),
    /// The frame size is past the limit, or the readback length does not match.
    Extent,
}

impl<E: fmt::Display> fmt::Display for ComposeError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Device(error) => write!(formatter, "the device refused the frame: {error}"),
            Self::Extent => formatter.write_str("the frame size is past the limit"),
        }
    }
}

impl<E: std::error::Error> std::error::Error for ComposeError<E> {}

/// Ask each source for `time`, then clear, draw, and submit one frame.
///
/// An empty source list, and a miss, leave [`ClearColor::TRANSPARENT`].
/// Sources are drawn in order. The framebuffer is destroyed before this
/// returns.
pub fn submit<D: Device>(
    device: &mut D,
    extent: Extent,
    time: Micros,
    sources: &[&dyn PictureSource],
) -> Result<Frame, ComposeError<D::Error>> {
    let Some(len) = extent.rgba_len() else {
        return Err(ComposeError::Extent);
    };
    let framebuffer = device
        .create_framebuffer(extent)
        .map_err(ComposeError::Device)?;
    let drawn = composite(device, framebuffer, extent, len, time, sources);
    match device.destroy_framebuffer(framebuffer) {
        Ok(()) => drawn,
        Err(error) => match drawn {
            Ok(_) => Err(ComposeError::Device(error)),
            Err(draw_error) => Err(draw_error),
        },
    }
}

fn composite<D: Device>(
    device: &mut D,
    framebuffer: Framebuffer,
    extent: Extent,
    len: usize,
    time: Micros,
    sources: &[&dyn PictureSource],
) -> Result<Frame, ComposeError<D::Error>> {
    let pass = device.begin(framebuffer).map_err(ComposeError::Device)?;
    device
        .blend(pass, pass_blend())
        .map_err(ComposeError::Device)?;
    device
        .clear(pass, ClearColor::TRANSPARENT)
        .map_err(ComposeError::Device)?;
    draw_ready(device, pass, sources, time).map_err(ComposeError::Device)?;
    device.end(pass).map_err(ComposeError::Device)?;
    device.submit(pass).map_err(ComposeError::Device)?;
    let rgba = device
        .read_rgba(framebuffer)
        .map_err(ComposeError::Device)?
        .to_vec();
    if rgba.len() != len {
        return Err(ComposeError::Extent);
    }
    Ok(Frame { extent, rgba })
}

#[cfg(test)]
mod tests {
    use cutlass_gpu::Extent;
    use cutlass_model::Micros;
    use cutlass_picture::{CpuBytes, Picture, PictureSource, PlatformBuffer, Sample};
    use cutlass_renderer_software::{SoftwareDevice, SoftwareError};

    use super::*;

    struct AtTime {
        time: Micros,
        pixel: [u8; 4],
    }

    impl PictureSource for AtTime {
        fn sample(&self, time: Micros) -> Sample {
            if time != self.time {
                return Sample::Miss;
            }
            Sample::Ready(Picture::Cpu(
                CpuBytes::new(1, 1, self.pixel.to_vec()).unwrap(),
            ))
        }
    }

    struct Missing;

    impl PictureSource for Missing {
        fn sample(&self, _time: Micros) -> Sample {
            Sample::Miss
        }
    }

    struct Platform;

    impl PictureSource for Platform {
        fn sample(&self, _time: Micros) -> Sample {
            Sample::Ready(Picture::Platform(PlatformBuffer {
                extent: Extent {
                    width: 1,
                    height: 1,
                },
                token: 4,
            }))
        }
    }

    fn extent() -> Extent {
        Extent {
            width: 1,
            height: 1,
        }
    }

    #[test]
    fn a_software_device_reads_the_solid_pixels_back() {
        let mut device = SoftwareDevice::new();
        let source = AtTime {
            time: Micros(40),
            pixel: [0x33, 0x66, 0x99, 0xFF],
        };
        let frame = submit(&mut device, extent(), Micros(40), &[&source]).unwrap();
        assert_eq!(frame.extent, extent());
        assert_eq!(frame.rgba, vec![0x33, 0x66, 0x99, 0xFF]);
        let missed = submit(&mut device, extent(), Micros(41), &[&source]).unwrap();
        assert_eq!(missed.rgba, vec![0, 0, 0, 0]);
    }

    #[test]
    fn a_miss_and_an_empty_source_list_leave_a_clear_frame() {
        let mut device = SoftwareDevice::new();
        let missing = Missing;
        let missed = submit(&mut device, extent(), Micros(0), &[&missing]).unwrap();
        assert_eq!(missed.rgba, vec![0, 0, 0, 0]);
        let empty = submit(
            &mut device,
            Extent {
                width: 2,
                height: 1,
            },
            Micros(0),
            &[],
        )
        .unwrap();
        assert_eq!(empty.rgba, vec![0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn a_platform_buffer_is_handed_to_the_software_device_and_refused() {
        let mut device = SoftwareDevice::new();
        let source = Platform;
        let error = submit(&mut device, extent(), Micros(0), &[&source]).unwrap_err();
        assert_eq!(error, ComposeError::Device(SoftwareError::PlatformBuffer));
    }

    #[test]
    fn later_sources_blend_over_earlier_ones() {
        let mut device = SoftwareDevice::new();
        let under = AtTime {
            time: Micros(1),
            pixel: [0, 0, 200, 255],
        };
        let over = AtTime {
            time: Micros(1),
            pixel: [100, 0, 0, 128],
        };
        let frame = submit(&mut device, extent(), Micros(1), &[&under, &over]).unwrap();
        assert_eq!(frame.rgba, vec![100, 0, 99, 255]);
    }

    #[test]
    fn an_edge_past_the_limit_is_refused() {
        let mut device = SoftwareDevice::new();
        let error = submit(
            &mut device,
            Extent {
                width: cutlass_gpu::MAX_EDGE + 1,
                height: 1,
            },
            Micros(0),
            &[],
        )
        .unwrap_err();
        assert_eq!(error, ComposeError::Extent);
    }
}
