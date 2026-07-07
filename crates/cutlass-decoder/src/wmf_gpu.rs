//! The decoder side of Windows zero-copy video: a pool of **shareable NV12
//! textures** plus a **shared fence**, bridging Media Foundation's D3D11
//! output to the renderer's D3D12 device.
//!
//! Hardware decoder MFTs output into slices of an internal D3D11 texture
//! *array* whose layout (and lifetime) belongs to the decoder — an array
//! slice can't be shared across APIs on its own. So the decoder performs one
//! **GPU-side copy** per frame: `CopySubresourceRegion` from the decoder's
//! slice into a pool texture created with
//! `D3D11_RESOURCE_MISC_SHARED_NTHANDLE`, then signals a shared
//! `ID3D11Fence` and flushes. The frame carries the texture + fence NT
//! handles (as a [`D3d11SharedSurface`]) for the renderer to open on its
//! D3D12 device and wait on — pixels never visit system memory.
//!
//! A pool slot stays **busy** until its frame drops (the keep-alive holds a
//! [`PooledFrame`]); the compositor blocks on GPU completion before frames
//! are dropped, so a freed slot is never still being sampled. Slots grow on
//! demand and are bounded by [`MAX_SLOTS`] — decode-ahead plus in-flight
//! composition holds only a few frames at once.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Foundation::{CloseHandle, GENERIC_ALL, HANDLE};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BIND_SHADER_RESOURCE, D3D11_FENCE_FLAG_SHARED, D3D11_RESOURCE_MISC_SHARED,
    D3D11_RESOURCE_MISC_SHARED_NTHANDLE, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT, ID3D11Device5,
    ID3D11DeviceContext4, ID3D11Fence, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_NV12, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::{
    DXGI_SHARED_RESOURCE_READ, DXGI_SHARED_RESOURCE_WRITE, IDXGIResource1,
};
use windows::core::Interface;

use cutlass_core::{D3d11SharedSurface, DecodeError};

use crate::wmf_d3d::D3dShared;

/// Hard cap on pool growth. Preview decode-ahead holds ~3 frames plus one in
/// flight; hitting this means frames are being leaked somewhere.
const MAX_SLOTS: usize = 16;

/// Process-unique identity source for pool textures and fences (never
/// recycled, so importer-side caches can key on it safely).
fn next_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// An owned NT handle (from `CreateSharedHandle`), closed exactly once.
struct OwnedHandle(HANDLE);

// SAFETY: an NT handle is a plain kernel object reference; duplication and
// closure are thread-safe kernel operations.
unsafe impl Send for OwnedHandle {}
unsafe impl Sync for OwnedHandle {}

impl Drop for OwnedHandle {
    fn drop(&mut self) {
        if !self.0.is_invalid() {
            // SAFETY: handle was received from CreateSharedHandle and is
            // closed exactly once (self is dropping).
            let _ = unsafe { CloseHandle(self.0) };
        }
    }
}

struct Slot {
    texture: ID3D11Texture2D,
    handle: OwnedHandle,
    id: u64,
    busy: bool,
}

/// A pool of shareable NV12 textures + the shared fence, on the decode device.
pub(crate) struct SharedNv12Pool {
    d3d: &'static D3dShared,
    /// Fence-capable view of the immediate context (`Signal`).
    context4: ID3D11DeviceContext4,
    fence: ID3D11Fence,
    fence_handle: OwnedHandle,
    fence_id: u64,
    next_fence_value: AtomicU64,
    size: (u32, u32),
    slots: Mutex<Vec<Slot>>,
}

// SAFETY: the device and fence are free-threaded; the multithread-protected
// immediate context serializes GPU calls (see `wmf_d3d`); `slots` is behind a
// Mutex. Frames carrying `Arc<SharedNv12Pool>` may drop on any thread.
unsafe impl Send for SharedNv12Pool {}
unsafe impl Sync for SharedNv12Pool {}

impl SharedNv12Pool {
    /// Create a pool for `size` coded NV12 frames on the shared decode device.
    /// Fails on pre-fence hardware/OS (needs Win10 1703+) or exotic drivers —
    /// callers degrade to CPU output.
    pub(crate) fn new(d3d: &'static D3dShared, size: (u32, u32)) -> Result<Self, DecodeError> {
        let context4: ID3D11DeviceContext4 = d3d
            .context
            .cast()
            .map_err(|e| DecodeError::unsupported(format!("no fence-capable context: {e}")))?;
        let device5: ID3D11Device5 = d3d
            .device
            .cast()
            .map_err(|e| DecodeError::unsupported(format!("no fence-capable device: {e}")))?;

        let mut fence: Option<ID3D11Fence> = None;
        // SAFETY: valid device; out-param receives the fence.
        unsafe { device5.CreateFence(0, D3D11_FENCE_FLAG_SHARED, &mut fence) }
            .map_err(|e| DecodeError::unsupported(format!("shared fence unavailable: {e}")))?;
        let fence = fence.expect("CreateFence succeeded");
        // SAFETY: valid shared fence; GENERIC_ALL so the D3D12 side can wait.
        let fence_handle = unsafe { fence.CreateSharedHandle(None, GENERIC_ALL.0, None) }
            .map_err(|e| DecodeError::unsupported(format!("fence handle unavailable: {e}")))?;

        Ok(Self {
            d3d,
            context4,
            fence,
            fence_handle: OwnedHandle(fence_handle),
            fence_id: next_id(),
            next_fence_value: AtomicU64::new(0),
            size,
            slots: Mutex::new(Vec::new()),
        })
    }

    pub(crate) fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Copy `subresource` of `source` (a decoder output slice) into a free
    /// pool slot, signal the fence, and hand back the frame payload + slot
    /// guard.
    pub(crate) fn export_frame(
        self: &Arc<Self>,
        source: &ID3D11Texture2D,
        subresource: u32,
    ) -> Result<PooledFrame, DecodeError> {
        let (slot_index, slot_id, slot_handle, texture) = {
            let mut slots = self.slots.lock().expect("pool lock");
            let index = match slots.iter().position(|s| !s.busy) {
                Some(index) => index,
                None => {
                    if slots.len() >= MAX_SLOTS {
                        return Err(DecodeError::Decode(
                            "GPU frame pool exhausted (frames not being released?)".into(),
                        ));
                    }
                    let slot = self.create_slot()?;
                    slots.push(slot);
                    slots.len() - 1
                }
            };
            let slot = &mut slots[index];
            slot.busy = true;
            (index, slot.id, slot.handle.0.0 as u64, slot.texture.clone())
        };

        // GPU copy of the whole NV12 image (in D3D11, planar formats are one
        // subresource per array slice, so this moves luma + chroma together).
        // The fence signal is enqueued behind it and flushed so the renderer's
        // wait can't stall on an unsubmitted command buffer.
        let value = self.next_fence_value.fetch_add(1, Ordering::Relaxed) + 1;
        unsafe {
            self.d3d
                .context
                .CopySubresourceRegion(&texture, 0, 0, 0, 0, source, subresource, None);
            self.context4
                .Signal(&self.fence, value)
                .map_err(|e| DecodeError::Decode(format!("fence signal failed: {e}")))?;
            self.d3d.context.Flush();
        }

        Ok(PooledFrame {
            payload: D3d11SharedSurface {
                texture_handle: slot_handle,
                texture_id: slot_id,
                fence_handle: self.fence_handle.0.0 as u64,
                fence_id: self.fence_id,
                fence_value: value,
            },
            pool: Arc::clone(self),
            slot_index,
        })
    }

    fn create_slot(&self) -> Result<Slot, DecodeError> {
        let (width, height) = self.size;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_NV12,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: (D3D11_RESOURCE_MISC_SHARED.0 | D3D11_RESOURCE_MISC_SHARED_NTHANDLE.0)
                as u32,
        };
        let mut texture: Option<ID3D11Texture2D> = None;
        // SAFETY: valid descriptor and out-param; no initial data.
        unsafe {
            self.d3d
                .device
                .CreateTexture2D(&desc, None, Some(&mut texture))
        }
        .map_err(|e| DecodeError::Decode(format!("shared NV12 texture creation failed: {e}")))?;
        let texture = texture.expect("CreateTexture2D succeeded");

        let resource: IDXGIResource1 = texture
            .cast()
            .map_err(|e| DecodeError::Decode(format!("texture is not shareable: {e}")))?;
        // SAFETY: resource created with SHARED_NTHANDLE; both accesses so the
        // D3D12 side passes its transition validation.
        let handle = unsafe {
            resource.CreateSharedHandle(
                None,
                DXGI_SHARED_RESOURCE_READ.0 | DXGI_SHARED_RESOURCE_WRITE.0,
                None,
            )
        }
        .map_err(|e| DecodeError::Decode(format!("texture handle creation failed: {e}")))?;

        Ok(Slot {
            texture,
            handle: OwnedHandle(handle),
            id: next_id(),
            busy: false,
        })
    }

    fn release(&self, slot_index: usize) {
        let mut slots = self.slots.lock().expect("pool lock");
        if let Some(slot) = slots.get_mut(slot_index) {
            slot.busy = false;
        }
    }
}

/// One exported frame: the wire payload plus the guard that returns the pool
/// slot when the frame drops. Parked in the frame's `keep_alive`; the
/// [`cutlass_core::GpuSurface::handle`] points at [`Self::payload`].
pub(crate) struct PooledFrame {
    pub(crate) payload: D3d11SharedSurface,
    pool: Arc<SharedNv12Pool>,
    slot_index: usize,
}

impl Drop for PooledFrame {
    fn drop(&mut self) {
        self.pool.release(self.slot_index);
    }
}
