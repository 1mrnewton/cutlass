//! Zero-copy import of Apple `CVPixelBuffer` video frames into `wgpu` textures.
//!
//! VideoToolbox hands the decoder an IOSurface-backed `CVPixelBuffer` (NV12 on
//! the hardware path). Rather than lock + copy its planes to system memory and
//! re-`write_texture` them, we map each plane straight into a Metal texture
//! with `CVMetalTextureCache`, then adopt that `MTLTexture` as a `wgpu` texture
//! through `wgpu-hal`'s Metal interop. The compositor's existing YUV pipeline
//! then samples the planes with no CPU pixel copy on the decode→composite seam.
//!
//! ## Why the version pinning matters
//!
//! `wgpu-hal` 28 links the `metal` 0.33 crate, and this crate depends on the
//! *same* version so the [`metal::Texture`] handed to
//! [`wgpu::hal::metal::Device::texture_from_raw`] is the exact type it expects
//! (cargo unifies the two into one crate instance). A mismatched `metal`
//! version would be a distinct, incompatible type.
//!
//! ## Bindings
//!
//! The `objc2-core-video` 0.3.2 `CVMetalTextureCache` bindings take
//! `objc2-metal` `ProtocolObject<dyn MTLDevice>` handles, which would force a
//! bridge to/from the `metal` (metal-rs) device wgpu-hal exposes. To avoid that
//! cross-crate type juggling we declare the three CoreVideo entry points
//! directly — they are plain C functions whose handle arguments are just
//! pointers at the ABI level.

use std::ffi::c_void;
use std::ptr;

use metal::foreign_types::ForeignType;

use cutlass_core::GpuSurface;

use crate::error::CompositorError;
use crate::gpu::GpuContext;

// `MTLPixelFormat` raw values. `CVMetalTextureCacheCreateTextureFromImage`
// takes the Metal enum as an `NSUInteger`; we pass the discriminants directly
// so we needn't depend on the enum's exact in-memory repr.
const MTL_PIXEL_FORMAT_R8_UNORM: usize = 10;
const MTL_PIXEL_FORMAT_RG8_UNORM: usize = 30;

#[link(name = "CoreVideo", kind = "framework")]
unsafe extern "C" {
    // CoreVideo's Metal texture cache. `metal_device` is an `id<MTLDevice>`,
    // `source_image` a `CVImageBufferRef`, and the `*_out` params receive
    // retained `CVMetalTextureCacheRef` / `CVMetalTextureRef` (CFType) handles.
    fn CVMetalTextureCacheCreate(
        allocator: *const c_void,
        cache_attributes: *const c_void,
        metal_device: *mut c_void,
        texture_attributes: *const c_void,
        cache_out: *mut *mut c_void,
    ) -> i32;

    fn CVMetalTextureCacheCreateTextureFromImage(
        allocator: *const c_void,
        texture_cache: *mut c_void,
        source_image: *mut c_void,
        texture_attributes: *const c_void,
        pixel_format: usize,
        width: usize,
        height: usize,
        plane_index: usize,
        texture_out: *mut *mut c_void,
    ) -> i32;

    // Returns the `id<MTLTexture>` the CVMetalTexture wraps (+0; owned by the
    // CVMetalTexture, valid while it lives).
    fn CVMetalTextureGetTexture(image: *mut c_void) -> *mut c_void;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    // CoreFoundation release for the CFType cache / texture handles.
    fn CFRelease(cf: *const c_void);
}

#[link(name = "objc", kind = "dylib")]
unsafe extern "C" {
    // Objective-C runtime retain (+1). Used to give the `metal::Texture`
    // wrapper its own reference to the MTLTexture `CVMetalTextureGetTexture`
    // returns at +0.
    fn objc_retain(obj: *mut c_void) -> *mut c_void;
}

/// A retained `CVMetalTextureCacheRef`. CoreVideo's cache is internally
/// synchronized, so the handle is safe to share across threads.
struct CacheHandle(*mut c_void);

// SAFETY: `CVMetalTextureCache` is documented thread-safe; we only create
// textures through it and release it exactly once on drop.
unsafe impl Send for CacheHandle {}
unsafe impl Sync for CacheHandle {}

impl Drop for CacheHandle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: `self.0` is a live CFType retained at creation.
            unsafe { CFRelease(self.0) };
        }
    }
}

/// Holds the per-plane `CVMetalTexture` wrappers alive for as long as a
/// composited frame's GPU textures exist.
///
/// Apple requires the `CVMetalTexture` to outlive GPU reads of its backing
/// `MTLTexture`; the compositor parks this in the layer's GPU resources, which
/// drop only after the render pass has been submitted and waited on.
pub(crate) struct ImportKeepAlive {
    cv_textures: Vec<*mut c_void>,
}

// SAFETY: the wrapped `CVMetalTexture` objects are immutable handles with
// atomic CoreFoundation retain/release, so the bundle can travel with a frame.
unsafe impl Send for ImportKeepAlive {}
unsafe impl Sync for ImportKeepAlive {}

impl Drop for ImportKeepAlive {
    fn drop(&mut self) {
        for &t in &self.cv_textures {
            if !t.is_null() {
                // SAFETY: each entry is a live CVMetalTexture retained by
                // CVMetalTextureCacheCreateTextureFromImage.
                unsafe { CFRelease(t) };
            }
        }
    }
}

/// Imports CoreVideo pixel buffers into `wgpu` textures for one GPU device.
///
/// Built once per [`GpuContext`] and reused: the underlying texture cache is a
/// long-lived object that recycles plane textures across frames.
pub(crate) struct MetalSurfaceImporter {
    cache: CacheHandle,
    // Keep the Metal device retained so the cache's device stays valid for the
    // importer's lifetime, independent of the wgpu device's hal guard.
    _device: metal::Device,
}

impl MetalSurfaceImporter {
    /// Build an importer for `gpu`'s device, or `None` when the device isn't a
    /// Metal device (e.g. a Vulkan/GL software adapter in CI) — callers then
    /// fall back to the CPU plane-upload path.
    pub(crate) fn new(gpu: &GpuContext) -> Option<Self> {
        // SAFETY: `as_hal` hands back the backing `metal::Device` for a Metal
        // wgpu device; we clone (retain) it so it outlives the borrowed guard.
        let device = unsafe {
            gpu.device
                .as_hal::<wgpu::hal::api::Metal>()
                .map(|hal_device| hal_device.raw_device().clone())
        }?;

        let mut cache: *mut c_void = ptr::null_mut();
        // SAFETY: a valid MTLDevice pointer and an out-param for the cache.
        let status = unsafe {
            CVMetalTextureCacheCreate(
                ptr::null(),
                ptr::null(),
                device.as_ptr().cast(),
                ptr::null(),
                &mut cache,
            )
        };
        if status != 0 || cache.is_null() {
            return None;
        }
        Some(Self {
            cache: CacheHandle(cache),
            _device: device,
        })
    }

    /// Import an NV12 `CVPixelBuffer` as a `(luma, chroma)` pair of `wgpu`
    /// textures: plane 0 as `R8Unorm` at full resolution, plane 1 as `Rg8Unorm`
    /// at half resolution — the layout the compositor's NV12 YUV path samples.
    ///
    /// `coded` is the full coded buffer size in pixels.
    pub(crate) fn import_nv12(
        &self,
        gpu: &GpuContext,
        surface: &GpuSurface,
        coded: (u32, u32),
    ) -> Result<(Vec<(wgpu::Texture, wgpu::TextureView)>, ImportKeepAlive), CompositorError> {
        let pixel_buffer = surface.handle() as *mut c_void;
        if pixel_buffer.is_null() {
            return Err(CompositorError::MalformedFrame(
                "CoreVideo surface handle is null".into(),
            ));
        }

        let (cw, ch) = coded;
        let specs = [
            (
                0usize,
                MTL_PIXEL_FORMAT_R8_UNORM,
                wgpu::TextureFormat::R8Unorm,
                cw,
                ch,
            ),
            (
                1usize,
                MTL_PIXEL_FORMAT_RG8_UNORM,
                wgpu::TextureFormat::Rg8Unorm,
                cw.div_ceil(2),
                ch.div_ceil(2),
            ),
        ];

        let mut keep = ImportKeepAlive {
            cv_textures: Vec::with_capacity(2),
        };
        let mut planes = Vec::with_capacity(2);

        for (plane_index, mtl_format, wgpu_format, w, h) in specs {
            let mut cv_texture: *mut c_void = ptr::null_mut();
            // SAFETY: cache + pixel buffer are live; out-param is valid.
            let status = unsafe {
                CVMetalTextureCacheCreateTextureFromImage(
                    ptr::null(),
                    self.cache.0,
                    pixel_buffer,
                    ptr::null(),
                    mtl_format,
                    w as usize,
                    h as usize,
                    plane_index,
                    &mut cv_texture,
                )
            };
            if status != 0 || cv_texture.is_null() {
                // `keep` drops here, releasing any earlier plane's CVMetalTexture.
                return Err(CompositorError::UnsupportedFormat(format!(
                    "CVMetalTextureCacheCreateTextureFromImage failed for plane \
                     {plane_index} (status {status})"
                )));
            }
            // Own the CVMetalTexture immediately so the error paths below (and
            // the eventual frame drop) release it exactly once.
            keep.cv_textures.push(cv_texture);

            // SAFETY: `cv_texture` is a live CVMetalTexture.
            let mtl_texture_ptr = unsafe { CVMetalTextureGetTexture(cv_texture) };
            if mtl_texture_ptr.is_null() {
                return Err(CompositorError::UnsupportedFormat(
                    "CVMetalTextureGetTexture returned null".into(),
                ));
            }

            // `CVMetalTextureGetTexture` returns the MTLTexture at +0 (owned by
            // the CVMetalTexture). Retain it so the `metal::Texture` wrapper —
            // and the wgpu texture that adopts it — can release on drop without
            // over-releasing the cache's reference.
            // SAFETY: `mtl_texture_ptr` is a live `id<MTLTexture>`.
            let retained = unsafe { objc_retain(mtl_texture_ptr) };
            // SAFETY: `retained` is a live, +1 MTLTexture; `from_ptr` adopts it.
            let metal_texture = unsafe { metal::Texture::from_ptr(retained.cast()) };

            let copy_size = wgpu::hal::CopyExtent {
                width: w,
                height: h,
                depth: 1,
            };
            // SAFETY: the texture is a valid 2D MTLTexture whose format matches
            // `wgpu_format`, created from `gpu`'s Metal device.
            let hal_texture = unsafe {
                wgpu::hal::metal::Device::texture_from_raw(
                    metal_texture,
                    wgpu_format,
                    metal::MTLTextureType::D2,
                    1,
                    1,
                    copy_size,
                )
            };

            let descriptor = wgpu::TextureDescriptor {
                label: Some("cutlass.zerocopy.plane"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            };
            // SAFETY: `hal_texture` was created from this device and matches
            // `descriptor`.
            let texture = unsafe {
                gpu.device
                    .create_texture_from_hal::<wgpu::hal::api::Metal>(hal_texture, &descriptor)
            };
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            planes.push((texture, view));
        }

        Ok((planes, keep))
    }
}
