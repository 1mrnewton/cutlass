//! Vulkan device behind [`cutlass_gpu::Device`].
//!
//! The portable path always keeps CPU framebuffers. Each channel is
//! `output = source + destination * (1 - source.alpha)`, with 8-bit alpha
//! divided by 255, the remainder truncated, and a channel above 255 clamped.
//! A platform buffer is refused on that path.
//!
//! Android also compiles a Vulkan backend for platform image tokens. Other
//! hosts return [`AndroidError::GpuUnsupported`] when that backend is
//! requested. That refusal is not a GPU result.

mod blend;
mod error;
mod frames;

#[cfg(target_os = "android")]
#[allow(unsafe_code)]
mod vulkan;

#[cfg(not(target_os = "android"))]
mod off;

#[cfg(test)]
mod lifecycle_tests;

#[cfg(test)]
mod pixels;

pub use error::AndroidError;

use cutlass_gpu::{Blend, ClearColor, Device, Extent, Framebuffer, Pass, TexelSource};

/// One Android device. The CPU framebuffer path is always available.
#[derive(Debug)]
pub struct AndroidDevice {
    frames: frames::Frames,
    #[cfg(target_os = "android")]
    gpu: Option<vulkan::Session>,
}

impl AndroidDevice {
    /// An empty device. Framebuffer ids start at 1.
    #[must_use]
    pub fn new() -> Self {
        Self {
            frames: frames::Frames::new(),
            #[cfg(target_os = "android")]
            gpu: None,
        }
    }

    /// Open the Vulkan backend.
    ///
    /// On a host that does not compile it, this is [`AndroidError::GpuUnsupported`].
    pub fn open_vulkan(&mut self) -> Result<(), AndroidError> {
        #[cfg(target_os = "android")]
        {
            if self.gpu.is_none() {
                self.gpu = Some(vulkan::open()?);
            }
            Ok(())
        }
        #[cfg(not(target_os = "android"))]
        {
            off::open()
        }
    }

    /// Mint a Vulkan image token for `extent`.
    ///
    /// On a host that does not compile Vulkan, this is [`AndroidError::GpuUnsupported`].
    pub fn allocate_image(&mut self, extent: Extent) -> Result<u64, AndroidError> {
        #[cfg(target_os = "android")]
        {
            self.open_vulkan()?;
            let Some(session) = self.gpu.as_mut() else {
                return Err(AndroidError::Device);
            };
            session.allocate_rgba(extent)
        }
        #[cfg(not(target_os = "android"))]
        {
            off::allocate(extent)
        }
    }

    /// Write RGBA bytes into an image this device allocated.
    ///
    /// On a host that does not compile Vulkan, this is [`AndroidError::GpuUnsupported`].
    pub fn upload_image(&mut self, token: u64, bytes: &[u8]) -> Result<(), AndroidError> {
        #[cfg(target_os = "android")]
        {
            self.open_vulkan()?;
            let Some(session) = self.gpu.as_mut() else {
                return Err(AndroidError::Device);
            };
            session.upload_rgba(token, bytes)
        }
        #[cfg(not(target_os = "android"))]
        {
            off::upload(token, bytes)
        }
    }

    fn draw_rgba(&mut self, pass: Pass, extent: Extent, bytes: &[u8]) -> Result<(), AndroidError> {
        self.frames.validate_source(pass, extent, bytes.len())?;
        if self.frames.blend_of(pass)? != Blend::Over {
            return Err(AndroidError::NoPass);
        }
        let destination = self.frames.drawing_rgba(pass)?;
        blend::blend_over(destination, bytes);
        Ok(())
    }

    fn draw_platform(
        &mut self,
        pass: Pass,
        extent: Extent,
        token: u64,
    ) -> Result<(), AndroidError> {
        self.frames.require_drawing(pass)?;
        #[cfg(not(target_os = "android"))]
        {
            let _ = (extent, token);
            Err(AndroidError::PlatformBuffer)
        }
        #[cfg(target_os = "android")]
        {
            self.draw_platform_vulkan(pass, extent, token)
        }
    }

    #[cfg(target_os = "android")]
    fn draw_platform_vulkan(
        &mut self,
        pass: Pass,
        extent: Extent,
        token: u64,
    ) -> Result<(), AndroidError> {
        let expected = self.frames.check_extent(pass, extent)?;
        if self.frames.blend_of(pass)? != Blend::Over {
            return Err(AndroidError::NoPass);
        }
        if self.gpu.is_none() {
            self.gpu = Some(vulkan::open()?);
        }
        let copied = {
            let Some(session) = self.gpu.as_mut() else {
                return Err(AndroidError::Device);
            };
            vulkan::sample_platform(session, extent, token)?
        };
        if copied.len() != expected {
            return Err(AndroidError::Bytes);
        }
        let destination = self.frames.drawing_rgba(pass)?;
        blend::blend_over(destination, &copied);
        Ok(())
    }
}

impl Default for AndroidDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl Device for AndroidDevice {
    type Error = AndroidError;

    fn create_framebuffer(&mut self, extent: Extent) -> Result<Framebuffer, Self::Error> {
        self.frames.create(extent)
    }

    fn destroy_framebuffer(&mut self, framebuffer: Framebuffer) -> Result<(), Self::Error> {
        self.frames.destroy(framebuffer)
    }

    fn begin(&mut self, framebuffer: Framebuffer) -> Result<Pass, Self::Error> {
        self.frames.begin(framebuffer)
    }

    fn blend(&mut self, pass: Pass, blend: Blend) -> Result<(), Self::Error> {
        self.frames.set_blend(pass, blend)
    }

    fn clear(&mut self, pass: Pass, color: ClearColor) -> Result<(), Self::Error> {
        self.frames.clear(pass, color)
    }

    fn draw(&mut self, pass: Pass, source: TexelSource<'_>) -> Result<(), Self::Error> {
        match source {
            TexelSource::Rgba { extent, bytes } => self.draw_rgba(pass, extent, bytes),
            TexelSource::Platform { extent, token } => self.draw_platform(pass, extent, token),
        }
    }

    fn end(&mut self, pass: Pass) -> Result<(), Self::Error> {
        self.frames.end(pass)
    }

    fn submit(&mut self, pass: Pass) -> Result<(), Self::Error> {
        self.frames.submit(pass)
    }

    fn read_rgba(&self, framebuffer: Framebuffer) -> Result<&[u8], Self::Error> {
        self.frames.read(framebuffer)
    }
}

#[cfg(all(test, not(target_os = "android")))]
mod host_tests {
    use cutlass_gpu::{ClearColor, Device, Extent};

    use super::{AndroidDevice, AndroidError};

    #[test]
    fn the_gpu_backend_is_unsupported_and_the_cpu_path_still_reads() {
        let mut device = AndroidDevice::new();
        let refused = device.open_vulkan().unwrap_err();
        assert_eq!(refused, AndroidError::GpuUnsupported);
        assert_eq!(refused.to_string(), "vulkan is not compiled on this host");
        assert_eq!(
            device.open_vulkan().unwrap_err(),
            AndroidError::GpuUnsupported
        );
        assert_eq!(
            device
                .allocate_image(Extent {
                    width: 1,
                    height: 1,
                })
                .unwrap_err(),
            AndroidError::GpuUnsupported
        );
        assert_eq!(
            device.upload_image(1, &[0, 0, 0, 0]).unwrap_err(),
            AndroidError::GpuUnsupported
        );

        let extent = Extent {
            width: 1,
            height: 1,
        };
        let framebuffer = device.create_framebuffer(extent).unwrap();
        let pass = device.begin(framebuffer).unwrap();
        device
            .clear(pass, ClearColor { rgba: [9, 8, 7, 6] })
            .unwrap();
        device.end(pass).unwrap();
        device.submit(pass).unwrap();
        assert_eq!(device.read_rgba(framebuffer).unwrap(), &[9, 8, 7, 6]);
    }
}
