//! Zero-copy import of Windows shared D3D11 video frames into `wgpu` textures.
//!
//! The Media Foundation decoder copies each hardware-decoded frame into an
//! NV12 texture created with `D3D11_RESOURCE_MISC_SHARED_NTHANDLE` and signals
//! a shared `ID3D11Fence` (see `cutlass-decoder`'s `wmf_gpu`). The frame
//! carries both NT handles as a [`D3d11SharedSurface`]. Here — on wgpu's D3D12
//! device — we `OpenSharedHandle` the texture, adopt it via `wgpu-hal` as a
//! `wgpu::TextureFormat::NV12` texture, and expose its planes as `R8Unorm` /
//! `Rg8Unorm` views for the compositor's existing NV12 YUV path. Pixels never
//! visit system memory between decode and composite.
//!
//! ## Caching
//!
//! Pool textures and the fence are *reused* across many frames, and the
//! payload carries never-recycled `texture_id` / `fence_id` values precisely
//! so importers can cache what they've opened. We keep one opened
//! `wgpu::Texture` (plus plane views) per pool slot and one `ID3D12Fence` per
//! decoder, so the steady-state per-frame cost is a map lookup and a fence
//! check. Caches are size-capped and flushed on overflow — in-flight layers
//! hold their own references, so a flush never invalidates a frame being
//! drawn, and re-opening a handle costs microseconds.
//!
//! ## Synchronization
//!
//! v1 is a CPU-side wait: before the frame's planes are handed to the render
//! pass we block until the shared fence reaches the frame's `fence_value`.
//! The decoder signals immediately after its copy and flushes, so by the time
//! a frame reaches the compositor the fence is almost always already past the
//! value and the wait is a single `GetCompletedValue` read. A queue-level
//! `ID3D12CommandQueue::Wait` is the zero-stall follow-up if profiling ever
//! shows the CPU wait mattering.
//!
//! ## Version pinning
//!
//! `wgpu-hal` 28 links `windows` 0.62, and this crate depends on the same
//! version so the `ID3D12Resource` handed to
//! `wgpu::hal::dx12::Device::texture_from_raw` is the exact type it expects
//! (cargo unifies the two into one crate instance) — the same trick
//! `metal_import` documents for the `metal` crate.

use std::collections::HashMap;
use std::sync::Mutex;

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::{ID3D12Device, ID3D12Fence, ID3D12Resource};

use cutlass_core::{D3d11SharedSurface, GpuSurface};

use crate::error::CompositorError;
use crate::gpu::GpuContext;

/// Cache caps. A decoder pool holds at most 16 slots and a project previews a
/// handful of sources at once; past these sizes we flush and re-open on
/// demand rather than pinning dead decoders' textures in VRAM.
const MAX_CACHED_TEXTURES: usize = 32;
const MAX_CACHED_FENCES: usize = 8;

/// One opened pool slot: the adopted NV12 texture and its plane views, reused
/// for every frame that lands in that slot.
struct CachedTexture {
    texture: wgpu::Texture,
    plane0: wgpu::TextureView,
    plane1: wgpu::TextureView,
}

/// Imports shared D3D11 NV12 textures into `wgpu` textures for one GPU device.
///
/// Built once per [`GpuContext`] and reused; holds the opened-handle caches.
///
/// ## Teardown hazard (field order matters)
///
/// The `Renderer` drops its `GpuContext` *before* the `Compositor` that owns
/// this importer, so raw D3D12 COM references cached here can be the last
/// thing keeping the device alive. Releasing a bare cloned `ID3D12Device`
/// *after* wgpu (and the DXGI factory) tore down access-violates inside the
/// driver — reproduced on Intel UHD 630. The rule: raw COM children
/// (`fences`) must release while the device chain is still alive, so this
/// struct holds a `wgpu::Device` clone as its **last** field — it pins the
/// whole wgpu instance→adapter→device chain, and Rust's declaration-order
/// drop releases the raw refs first, then lets the chain tear down properly.
pub(crate) struct Dx12SurfaceImporter {
    fences: Mutex<HashMap<u64, ID3D12Fence>>,
    textures: Mutex<HashMap<u64, CachedTexture>>,
    /// Keeps the D3D12 device (and its instance chain) alive until the raw
    /// COM references above have been released. Must stay the last field.
    device: wgpu::Device,
}

// SAFETY: D3D12 interfaces are free-threaded (the API is designed for
// multithreaded recording); the caches are behind Mutexes. The importer lives
// in the Compositor, which moves across threads with the renderer.
unsafe impl Send for Dx12SurfaceImporter {}
unsafe impl Sync for Dx12SurfaceImporter {}

impl Dx12SurfaceImporter {
    /// Build an importer for `gpu`'s device, or `None` when zero-copy import
    /// can't work there: the device isn't the D3D12 backend (e.g. Vulkan or a
    /// software adapter) or lacks `TEXTURE_FORMAT_NV12`. Callers then fall
    /// back to the CPU plane-upload path.
    pub(crate) fn new(gpu: &GpuContext) -> Option<Self> {
        if !gpu
            .device
            .features()
            .contains(wgpu::Features::TEXTURE_FORMAT_NV12)
        {
            return None;
        }
        // SAFETY: read-only backend probe; the guard drops before return.
        unsafe { gpu.device.as_hal::<wgpu::hal::api::Dx12>() }.as_ref()?;
        Some(Self {
            fences: Mutex::new(HashMap::new()),
            textures: Mutex::new(HashMap::new()),
            device: gpu.device.clone(),
        })
    }

    /// Borrow the raw `ID3D12Device` for the duration of one call. The clone
    /// (COM AddRef) is released before the caller returns; `self.device`
    /// guarantees the device outlives every cached child (see struct docs).
    fn raw_device(&self) -> Result<ID3D12Device, CompositorError> {
        // SAFETY: `as_hal` hands back the backing hal device for a D3D12 wgpu
        // device; the short-lived clone is dropped within the calling scope.
        unsafe {
            self.device
                .as_hal::<wgpu::hal::api::Dx12>()
                .map(|hal_device| hal_device.raw_device().clone())
        }
        .ok_or_else(|| CompositorError::UnsupportedFormat("device is not the D3D12 backend".into()))
    }

    /// Import a shared NV12 surface as a `(luma, chroma)` pair of `wgpu`
    /// texture views: plane 0 as `R8Unorm` at full resolution, plane 1 as
    /// `Rg8Unorm` at half resolution — the layout the compositor's NV12 YUV
    /// path samples. Blocks until the frame's fence value is reached, so the
    /// decoder's copy is complete before the pass samples the texture.
    ///
    /// `coded` is the full coded buffer size in pixels.
    pub(crate) fn import_nv12(
        &self,
        gpu: &GpuContext,
        surface: &GpuSurface,
        coded: (u32, u32),
    ) -> Result<Vec<(wgpu::Texture, wgpu::TextureView)>, CompositorError> {
        let payload = surface.handle() as *const D3d11SharedSurface;
        if payload.is_null() {
            return Err(CompositorError::MalformedFrame(
                "D3D11 surface payload is null".into(),
            ));
        }
        // SAFETY: for `GpuSurfaceKind::D3D11Texture2D` the handle points at the
        // `D3d11SharedSurface` owned by the frame's keep-alive, which outlives
        // this borrow of the frame.
        let payload = unsafe { *payload };

        self.wait_for_frame(&payload)?;

        let mut textures = self.textures.lock().expect("texture cache lock");
        let entry = match textures.entry(payload.texture_id) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(self.open_texture(gpu, &payload, coded)?)
            }
        };
        let planes = vec![
            (entry.texture.clone(), entry.plane0.clone()),
            (entry.texture.clone(), entry.plane1.clone()),
        ];

        // Flush on overflow *after* cloning out this frame's planes; wgpu
        // resources are internally ref-counted, so in-flight layers keep
        // drawing from their own references.
        if textures.len() > MAX_CACHED_TEXTURES {
            textures.clear();
        }

        Ok(planes)
    }

    /// Block until the decode-side copy guarding `payload` has completed on
    /// the GPU. Almost always a no-op read of the fence's completed value.
    fn wait_for_frame(&self, payload: &D3d11SharedSurface) -> Result<(), CompositorError> {
        let fence = {
            let mut fences = self.fences.lock().expect("fence cache lock");
            if fences.len() > MAX_CACHED_FENCES {
                fences.clear();
            }
            match fences.entry(payload.fence_id) {
                std::collections::hash_map::Entry::Occupied(e) => e.get().clone(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    let mut opened: Option<ID3D12Fence> = None;
                    // SAFETY: the fence NT handle stays open for the life of
                    // the decoder pool, which the in-flight frame keeps alive.
                    unsafe {
                        self.raw_device()?
                            .OpenSharedHandle(HANDLE(payload.fence_handle as _), &mut opened)
                    }
                    .map_err(|e| {
                        CompositorError::UnsupportedFormat(format!(
                            "opening shared D3D11 fence failed: {e}"
                        ))
                    })?;
                    let opened = opened.ok_or_else(|| {
                        CompositorError::UnsupportedFormat(
                            "opening shared D3D11 fence returned nothing".into(),
                        )
                    })?;
                    e.insert(opened).clone()
                }
            }
        };

        // SAFETY: valid fence; reading the completed value is side-effect free.
        if unsafe { fence.GetCompletedValue() } < payload.fence_value {
            // SAFETY: a null event handle makes this call block until the
            // fence reaches the value — the documented CPU-wait idiom.
            unsafe { fence.SetEventOnCompletion(payload.fence_value, HANDLE::default()) }.map_err(
                |e| CompositorError::UnsupportedFormat(format!("fence wait failed: {e}")),
            )?;
        }
        Ok(())
    }

    /// Open `payload`'s texture handle on the D3D12 device and adopt it as a
    /// `wgpu` NV12 texture with one view per plane.
    fn open_texture(
        &self,
        gpu: &GpuContext,
        payload: &D3d11SharedSurface,
        coded: (u32, u32),
    ) -> Result<CachedTexture, CompositorError> {
        let (cw, ch) = coded;
        let mut resource: Option<ID3D12Resource> = None;
        // SAFETY: the texture NT handle stays open for the life of the decoder
        // pool, which the in-flight frame keeps alive.
        unsafe {
            self.raw_device()?
                .OpenSharedHandle(HANDLE(payload.texture_handle as _), &mut resource)
        }
        .map_err(|e| {
            CompositorError::UnsupportedFormat(format!("opening shared D3D11 texture failed: {e}"))
        })?;
        let resource = resource.ok_or_else(|| {
            CompositorError::UnsupportedFormat(
                "opening shared D3D11 texture returned nothing".into(),
            )
        })?;

        let size = wgpu::Extent3d {
            width: cw,
            height: ch,
            depth_or_array_layers: 1,
        };
        // SAFETY: the resource is a 2D NV12 texture matching `size`, opened on
        // this wgpu device's ID3D12Device.
        let hal_texture = unsafe {
            wgpu::hal::dx12::Device::texture_from_raw(
                resource,
                wgpu::TextureFormat::NV12,
                wgpu::TextureDimension::D2,
                size,
                1,
                1,
            )
        };

        let descriptor = wgpu::TextureDescriptor {
            label: Some("cutlass.zerocopy.nv12"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::NV12,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        // SAFETY: `hal_texture` was created from this device and matches
        // `descriptor`.
        let texture = unsafe {
            gpu.device
                .create_texture_from_hal::<wgpu::hal::api::Dx12>(hal_texture, &descriptor)
        };

        let plane_view = |aspect, format, label| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(label),
                format: Some(format),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect,
                ..Default::default()
            })
        };
        let plane0 = plane_view(
            wgpu::TextureAspect::Plane0,
            wgpu::TextureFormat::R8Unorm,
            "cutlass.zerocopy.nv12.y",
        );
        let plane1 = plane_view(
            wgpu::TextureAspect::Plane1,
            wgpu::TextureFormat::Rg8Unorm,
            "cutlass.zerocopy.nv12.uv",
        );

        Ok(CachedTexture {
            texture,
            plane0,
            plane1,
        })
    }
}
