//! macOS-only: VideoToolbox → CVPixelBuffer → IOSurface → Metal → wgpu textures (NV12 planes).
#![cfg(target_os = "macos")]

use core::ptr::NonNull;

use ffmpeg::format::{Pixel, context::Input, input};
use ffmpeg::util::frame::video::Video as VideoFrame;
use ffmpeg_next as ffmpeg;
use objc2_core_foundation::CFRetained;
use objc2_core_video::{
    CVPixelBuffer, CVPixelBufferGetHeightOfPlane, CVPixelBufferGetIOSurface,
    CVPixelBufferGetPixelFormatType, CVPixelBufferGetPlaneCount,
    CVPixelBufferGetWidthOfPlane, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
    kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
};
use objc2_metal::{
    MTLDevice, MTLPixelFormat, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};
use std::path::Path;
use wgpu::hal::metal::Device as HalMetalDevice;

use super::attach_videotoolbox;

#[derive(Debug)]
#[allow(dead_code)]
pub enum GpuFrameImportError {
    NotVideoToolbox,
    NullPixelBuffer,
    NotIOSurfaceBacked,
    UnsupportedPixelFormat(u32),
    WrongPlaneCount(usize),
    NotMetalDevice,
    MetalTextureFailed(&'static str),
}

/// Zero-copy NV12 frame in GPU memory. Keep `_pixel_buffer` alive: it owns the IOSurface.
pub struct GpuFrame {
    pub y: wgpu::Texture,
    pub cbcr: wgpu::Texture,
    #[allow(dead_code)]
    pub width: u32,
    #[allow(dead_code)]
    pub height: u32,
    /// Retains the CVPixelBuffer so the IOSurface stays valid for the wgpu textures.
    _pixel_buffer: CFRetained<CVPixelBuffer>,
}

impl GpuFrame {
    /// Import a VideoToolbox (`AV_PIX_FMT_VIDEOTOOLBOX`) frame into wgpu.
    ///
    /// # Panics / errors
    /// `device` must be backed by Metal; fails with [`GpuFrameImportError::NotMetalDevice`] otherwise.
    pub fn from_vt_video_frame(
        device: &wgpu::Device,
        frame: &VideoFrame,
    ) -> Result<Self, GpuFrameImportError> {
        if frame.format() != Pixel::VIDEOTOOLBOX {
            return Err(GpuFrameImportError::NotVideoToolbox);
        }

        let pixel_buffer = unsafe {
            let f = frame.as_ptr();
            let ptr = (*f).data[3].cast::<CVPixelBuffer>();
            let Some(nn) = NonNull::new(ptr) else {
                return Err(GpuFrameImportError::NullPixelBuffer);
            };
            CFRetained::<CVPixelBuffer>::retain(nn)
        };

        let fmt = CVPixelBufferGetPixelFormatType(&pixel_buffer);
        if fmt != kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
            && fmt != kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        {
            return Err(GpuFrameImportError::UnsupportedPixelFormat(fmt));
        }

        let planes = CVPixelBufferGetPlaneCount(&pixel_buffer);
        if planes < 2 {
            return Err(GpuFrameImportError::WrongPlaneCount(planes));
        }

        let y_w = CVPixelBufferGetWidthOfPlane(&pixel_buffer, 0) as u32;
        let y_h = CVPixelBufferGetHeightOfPlane(&pixel_buffer, 0) as u32;
        let c_w = CVPixelBufferGetWidthOfPlane(&pixel_buffer, 1) as u32;
        let c_h = CVPixelBufferGetHeightOfPlane(&pixel_buffer, 1) as u32;

        let iosurface = CVPixelBufferGetIOSurface(Some(&pixel_buffer))
            .ok_or(GpuFrameImportError::NotIOSurfaceBacked)?;

        let hal_guard = unsafe { device.as_hal::<wgpu::hal::api::Metal>() }
            .ok_or(GpuFrameImportError::NotMetalDevice)?;
        let mtl = hal_guard.raw_device();

        let y_desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::R8Unorm,
                y_w as usize,
                y_h as usize,
                false,
            )
        };
        y_desc.setUsage(MTLTextureUsage::ShaderRead);

        let cbcr_desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::RG8Unorm,
                c_w as usize,
                c_h as usize,
                false,
            )
        };
        cbcr_desc.setUsage(MTLTextureUsage::ShaderRead);

        let y_mtl = mtl
            .newTextureWithDescriptor_iosurface_plane(&y_desc, iosurface.as_ref(), 0)
            .ok_or(GpuFrameImportError::MetalTextureFailed("Y plane"))?;

        let cbcr_mtl = mtl
            .newTextureWithDescriptor_iosurface_plane(&cbcr_desc, iosurface.as_ref(), 1)
            .ok_or(GpuFrameImportError::MetalTextureFailed("CbCr plane"))?;

        let hal_y = unsafe {
            HalMetalDevice::texture_from_raw(
                y_mtl,
                wgpu::TextureFormat::R8Unorm,
                MTLTextureType::Type2D,
                1,
                1,
                wgpu::hal::CopyExtent {
                    width: y_w,
                    height: y_h,
                    depth: 1,
                },
            )
        };

        let hal_cbcr = unsafe {
            HalMetalDevice::texture_from_raw(
                cbcr_mtl,
                wgpu::TextureFormat::Rg8Unorm,
                MTLTextureType::Type2D,
                1,
                1,
                wgpu::hal::CopyExtent {
                    width: c_w,
                    height: c_h,
                    depth: 1,
                },
            )
        };

        let y = unsafe {
            device.create_texture_from_hal::<wgpu::hal::api::Metal>(
                hal_y,
                &wgpu::TextureDescriptor {
                    label: Some("nv12-luma"),
                    size: wgpu::Extent3d {
                        width: y_w,
                        height: y_h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
            )
        };

        let cbcr = unsafe {
            device.create_texture_from_hal::<wgpu::hal::api::Metal>(
                hal_cbcr,
                &wgpu::TextureDescriptor {
                    label: Some("nv12-chroma"),
                    size: wgpu::Extent3d {
                        width: c_w,
                        height: c_h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rg8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
            )
        };

        Ok(Self {
            y,
            cbcr,
            width: y_w,
            height: y_h,
            _pixel_buffer: pixel_buffer,
        })
    }
}

/// Decoder that outputs VideoToolbox surfaces (no CPU `av_hwframe_transfer_data`).
pub struct VideoDecoderGpu {
    ictx: Input,
    decoder: ffmpeg::decoder::Video,
    stream_idx: usize,
    width: u32,
    height: u32,
}

impl VideoDecoderGpu {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ffmpeg::Error> {
        ffmpeg::init()?;
        let ictx = input(&path)?;
        let stream = ictx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or(ffmpeg::Error::StreamNotFound)?;
        let stream_idx = stream.index();

        let ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?;

        let mut dec = ctx.decoder();
        if !attach_videotoolbox(&mut dec) {
            return Err(ffmpeg::Error::InvalidData);
        }
        log::debug!("GPU path: VideoToolbox decoding (IOSurface → wgpu)");

        let decoder = dec.video()?;
        let (width, height) = (decoder.width(), decoder.height());

        Ok(Self {
            ictx,
            decoder,
            stream_idx,
            width,
            height,
        })
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Writes the next decoded frame into `out`. Returns `Ok(None)` at EOF.
    pub fn decode_into(&mut self, out: &mut VideoFrame) -> Result<Option<()>, ffmpeg::Error> {
        loop {
            if self.decoder.receive_frame(out).is_ok() {
                return Ok(Some(()));
            }

            match self.ictx.packets().next() {
                Some((stream, packet)) => {
                    if stream.index() == self.stream_idx {
                        self.decoder.send_packet(&packet)?;
                    }
                }
                None => {
                    self.decoder.send_eof()?;
                    if self.decoder.receive_frame(out).is_ok() {
                        return Ok(Some(()));
                    }
                    return Ok(None);
                }
            }
        }
    }
}
