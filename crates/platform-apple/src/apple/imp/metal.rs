//! A [`VtSurface`] as Metal textures on a wgpu device, without a copy.
//!
//! CoreVideo's Metal texture cache wraps each plane of the IOSurface as a
//! texture: luma as `R8Unorm`, interleaved chroma as `RG8Unorm` at half size.
//! wgpu takes ownership of each through its Metal backend. The CoreVideo
//! texture and the surface stay alive until wgpu drops the texture, which it
//! does once the GPU has finished every submission that sampled it.

use std::ffi::c_void;
use std::ptr;
use std::sync::Mutex;

use cutlass_decode::Surface;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLTexture, MTLTextureType};

use super::cf::Cf;
use super::surface::VtSurface;
use super::sys;

/// `MTLPixelFormatR8Unorm`.
const R8_UNORM: usize = 10;
/// `MTLPixelFormatRG8Unorm`.
const RG8_UNORM: usize = 30;

/// Luma and chroma textures for `surface` on `device`, or `None` when the
/// surface is not a VideoToolbox buffer or the device is not Metal.
pub fn import_surface(device: &wgpu::Device, surface: &Surface) -> Option<[wgpu::Texture; 2]> {
    let picture = surface.as_any().downcast_ref::<VtSurface>()?;
    let cache = texture_cache(device)?;
    let (width, height) = picture.size();
    let luma = plane(
        device,
        &cache,
        picture,
        surface,
        Plane {
            index: 0,
            metal: R8_UNORM,
            format: wgpu::TextureFormat::R8Unorm,
            size: (width, height),
        },
    )?;
    let chroma = plane(
        device,
        &cache,
        picture,
        surface,
        Plane {
            index: 1,
            metal: RG8_UNORM,
            format: wgpu::TextureFormat::Rg8Unorm,
            size: (width / 2, height / 2),
        },
    )?;
    Some([luma, chroma])
}

struct Plane {
    index: usize,
    metal: usize,
    format: wgpu::TextureFormat,
    size: (u32, u32),
}

/// What a wgpu texture keeps alive: the CoreVideo texture and its surface.
struct Keep {
    _texture: Cf,
    _surface: Surface,
}

// SAFETY: a `CVMetalTexture` may be released from any thread, and it is
// only released here, when the drop callback runs.
unsafe impl Sync for Keep {}

fn plane(
    device: &wgpu::Device,
    cache: &Cf,
    picture: &VtSurface,
    surface: &Surface,
    plane: Plane,
) -> Option<wgpu::Texture> {
    let (width, height) = plane.size;
    let mut out = ptr::null_mut();
    let status = unsafe {
        sys::CVMetalTextureCacheCreateTextureFromImage(
            ptr::null(),
            cache.as_mut(),
            picture.buffer(),
            ptr::null(),
            plane.metal,
            width as usize,
            height as usize,
            plane.index,
            &mut out,
        )
    };
    if status != 0 {
        return None;
    }
    let texture = Cf::from_owned(out)?;
    let raw = unsafe { sys::CVMetalTextureGetTexture(texture.as_mut()) };
    // SAFETY: `raw` is the `id<MTLTexture>` the CoreVideo texture owns; this
    // takes a retain of our own.
    let raw: Retained<ProtocolObject<dyn MTLTexture>> =
        unsafe { Retained::retain(raw.cast::<ProtocolObject<dyn MTLTexture>>()) }?;
    let keep = Keep {
        _texture: texture,
        _surface: surface.clone(),
    };
    // SAFETY: `raw` is a 2D texture of `plane.format` and `plane.size` with
    // one level and one layer, made on this device's `MTLDevice`.
    let hal = unsafe {
        wgpu::hal::metal::Device::texture_from_raw(
            raw,
            plane.format,
            MTLTextureType::Type2D,
            1,
            1,
            wgpu::hal::CopyExtent {
                width,
                height,
                depth: 1,
            },
            Some(Box::new(move || drop(keep))),
        )
    };
    let descriptor = wgpu::TextureDescriptor {
        label: Some("cutlass surface plane"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: plane.format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };
    // SAFETY: `hal` was made from a texture on this device and matches
    // `descriptor`. The GPU only reads it.
    Some(unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Metal>(
            hal,
            &descriptor,
            wgpu::TextureUses::RESOURCE,
        )
    })
}

struct Cache {
    device: usize,
    cache: Cf,
}

/// The CoreVideo texture cache for `device`'s `MTLDevice`. One process has
/// one device, so one cache is kept, made again if the device changes.
fn texture_cache(device: &wgpu::Device) -> Option<Cf> {
    // SAFETY: the Metal device is only read for its pointer, and wgpu keeps
    // it alive as long as `device`.
    let metal = unsafe { device.as_hal::<wgpu::hal::api::Metal>() }?;
    let raw = Retained::as_ptr(metal.raw_device())
        .cast_mut()
        .cast::<c_void>();
    drop(metal);
    static CACHE: Mutex<Option<Cache>> = Mutex::new(None);
    let mut held = match CACHE.lock() {
        Ok(guard) => guard,
        Err(poison) => poison.into_inner(),
    };
    if held
        .as_ref()
        .is_none_or(|cache| cache.device != raw as usize)
    {
        let mut out = ptr::null_mut();
        let status = unsafe {
            sys::CVMetalTextureCacheCreate(ptr::null(), ptr::null(), raw, ptr::null(), &mut out)
        };
        if status != 0 {
            return None;
        }
        *held = Some(Cache {
            device: raw as usize,
            cache: Cf::from_owned(out)?,
        });
    }
    let cache = held.as_ref()?;
    // Recycle the textures wgpu has already dropped.
    unsafe { sys::CVMetalTextureCacheFlush(cache.cache.as_mut(), 0) };
    let retained = unsafe { sys::CFRetain(cache.cache.as_ptr()) };
    Cf::from_owned(retained.cast_mut())
}
