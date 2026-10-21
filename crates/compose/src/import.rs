//! Wraps an IOSurface as two wgpu textures. The pixels stay in the surface.

#![expect(unsafe_code, reason = "Metal wraps an IOSurface the GPU already owns")]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_io_surface::IOSurfaceRef;
use objc2_metal::{
    MTLDevice, MTLPixelFormat, MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureType,
    MTLTextureUsage,
};

use cutlass_videotoolbox::PixelBuffer;

use crate::Error;

pub(crate) struct Imported {
    pub y: wgpu::Texture,
    pub chroma: wgpu::Texture,
}

pub(crate) fn import(
    device: &wgpu::Device,
    buffer: &PixelBuffer,
    y_format: wgpu::TextureFormat,
    chroma_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> Result<Imported, Error> {
    let surface = buffer.io_surface().ok_or(Error::NoSurface)?;
    let (y, chroma) = {
        let metal = {
            // SAFETY: the guard is used only to clone the Metal device. The
            // device is not destroyed through it.
            let hal = unsafe { device.as_hal::<wgpu::hal::metal::Api>() }.ok_or(Error::NotMetal)?;
            hal.raw_device().clone()
        };
        let y = plane(
            &metal,
            &surface,
            0,
            metal_format(y_format)?,
            width,
            height,
            "luma",
        )?;
        let chroma = plane(
            &metal,
            &surface,
            1,
            metal_format(chroma_format)?,
            width / 2,
            height / 2,
            "chroma",
        )?;
        (y, chroma)
    };
    Ok(Imported {
        y: wrap(device, y, y_format, width, height, "luma"),
        chroma: wrap(
            device,
            chroma,
            chroma_format,
            width / 2,
            height / 2,
            "chroma",
        ),
    })
}

fn plane(
    device: &ProtocolObject<dyn MTLDevice>,
    surface: &IOSurfaceRef,
    index: usize,
    format: MTLPixelFormat,
    width: u32,
    height: u32,
    name: &'static str,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, Error> {
    let descriptor = MTLTextureDescriptor::new();
    descriptor.setPixelFormat(format);
    descriptor.setTextureType(MTLTextureType::Type2D);
    descriptor.setUsage(MTLTextureUsage::ShaderRead);
    descriptor.setStorageMode(MTLStorageMode::Shared);
    let width = usize::try_from(width).map_err(|_| cutlass_color::Error::TooLarge)?;
    let height = usize::try_from(height).map_err(|_| cutlass_color::Error::TooLarge)?;
    // SAFETY: the plane is a 2D texture with one mip, one sample, and depth 1.
    // Width and height are that plane's pixel size, already checked nonzero.
    unsafe {
        descriptor.setWidth(width);
        descriptor.setHeight(height);
        descriptor.setDepth(1);
        descriptor.setMipmapLevelCount(1);
        descriptor.setSampleCount(1);
    }
    device
        .newTextureWithDescriptor_iosurface_plane(&descriptor, surface, index)
        .ok_or_else(|| Error::Import(format!("Metal could not wrap the {name} plane")))
}

fn wrap(
    device: &wgpu::Device,
    texture: Retained<ProtocolObject<dyn MTLTexture>>,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    name: &'static str,
) -> wgpu::Texture {
    let desc = wgpu::TextureDescriptor {
        label: Some(name),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };
    // SAFETY: `texture` was created on this device's Metal device with `desc`.
    // The IOSurface is already initialized, and the texture is only sampled,
    // which is `TextureUses::RESOURCE`.
    let hal_texture = unsafe {
        wgpu::hal::metal::Device::texture_from_raw(
            texture,
            format,
            MTLTextureType::Type2D,
            1,
            1,
            wgpu::hal::CopyExtent {
                width,
                height,
                depth: 1,
            },
            None,
        )
    };
    // SAFETY: as above. The HAL texture came from this device.
    unsafe {
        device.create_texture_from_hal::<wgpu::hal::metal::Api>(
            hal_texture,
            &desc,
            wgpu::wgt::TextureUses::RESOURCE,
        )
    }
}

fn metal_format(format: wgpu::TextureFormat) -> Result<MTLPixelFormat, Error> {
    match format {
        wgpu::TextureFormat::R8Unorm => Ok(MTLPixelFormat::R8Unorm),
        wgpu::TextureFormat::Rg8Unorm => Ok(MTLPixelFormat::RG8Unorm),
        wgpu::TextureFormat::R16Unorm => Ok(MTLPixelFormat::R16Unorm),
        wgpu::TextureFormat::Rg16Unorm => Ok(MTLPixelFormat::RG16Unorm),
        _ => Err(Error::Import(
            "the plane format has no Metal equivalent".to_owned(),
        )),
    }
}
