//! [`VideoFrame`]: the unit that crosses the decode→compositor seam.
//!
//! A decoded frame is one of two things, and the compositor handles both behind
//! the same type:
//!
//! - **[`FrameData::Gpu`]** — the fast path. The platform decoder produced a
//!   GPU surface (an IOSurface-backed `CVPixelBuffer` on Apple, an
//!   `AHardwareBuffer` on Android, a D3D11 texture on Windows) which the
//!   renderer maps straight into a `wgpu` texture with no pixel copy. The handle
//!   is opaque to core: a raw integer plus a `kind` tag, with the platform's
//!   retained wrapper parked in `keep_alive` so the surface outlives GPU reads.
//! - **[`FrameData::Cpu`]** — the portable fallback. Plain planes in system
//!   memory (software decode, image stills, a backend without zero-copy import),
//!   uploaded with `write_texture`.
//!
//! Frames travel from a decode worker thread to the render thread, so
//! `VideoFrame` is `Send`. It is intentionally **not** `Clone`: a GPU surface is
//! a unique handle to decoder-owned memory.

use core::any::Any;
use core::fmt;

use crate::color::ColorSpace;
use crate::geometry::{Rect, Rotation};
use crate::pixel::PixelFormat;
use crate::time::RationalTime;

/// One plane of CPU pixel data (e.g. the Y plane, or interleaved chroma).
///
/// `stride` is the row pitch in **bytes** (may exceed the visible row width;
/// decoders pad rows for alignment). `rows` is the number of sample rows this
/// plane stores, which for subsampled chroma is less than the frame height.
pub struct Plane {
    pub data: Vec<u8>,
    pub stride: usize,
    pub rows: usize,
}

impl Plane {
    pub fn new(data: Vec<u8>, stride: usize, rows: usize) -> Self {
        Self { data, stride, rows }
    }

    /// True if the backing buffer is large enough for `stride × rows`.
    pub fn is_well_formed(&self) -> bool {
        self.stride
            .checked_mul(self.rows)
            .is_some_and(|need| self.data.len() >= need)
    }
}

impl fmt::Debug for Plane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Never dump pixel bytes; summarize the layout instead.
        f.debug_struct("Plane")
            .field("bytes", &self.data.len())
            .field("stride", &self.stride)
            .field("rows", &self.rows)
            .finish()
    }
}

/// CPU-resident pixel data: one plane per [`PixelFormat::plane_count`].
#[derive(Debug)]
pub struct CpuImage {
    pub planes: Vec<Plane>,
}

impl CpuImage {
    pub fn new(planes: Vec<Plane>) -> Self {
        Self { planes }
    }
}

/// A retained, platform-specific wrapper kept alive for the lifetime of a GPU
/// frame so the decoder's surface memory isn't freed while the GPU samples it.
/// Type-erased so core needn't know about `CFRetained`, `AHardwareBuffer`, etc.
pub type KeepAlive = Box<dyn Any + Send + Sync>;

/// What kind of native GPU surface a [`GpuSurface`] handle refers to, so the
/// renderer knows which import path to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuSurfaceKind {
    /// Apple `CVPixelBuffer` (IOSurface-backed) — imported via
    /// `CVMetalTextureCache`.
    CoreVideoPixelBuffer,
    /// Android `AHardwareBuffer` — imported via external-memory / EGL.
    AHardwareBuffer,
    /// Windows: a shareable Direct3D 11 texture. The [`GpuSurface::handle`]
    /// points at a [`D3d11SharedSurface`] describing it (owned by the frame's
    /// keep-alive).
    D3D11Texture2D,
}

/// Windows zero-copy payload: how a [`GpuSurfaceKind::D3D11Texture2D`] frame
/// travels from the Media Foundation decoder to the D3D12-backed renderer.
///
/// The decoder copies each decoded frame into a texture created with
/// `D3D11_RESOURCE_MISC_SHARED_NTHANDLE` and signals a shared fence; the
/// renderer opens both NT handles on its D3D12 device (`OpenSharedHandle`),
/// waits for `fence_value`, and samples the texture — pixels never leave the
/// GPU. Plain integers so `cutlass-core` stays platform-agnostic.
///
/// ## Lifetime and identity
///
/// Both handles are owned by the decoder and stay open for as long as the
/// frame's `keep_alive` lives — importers must be done with them when the
/// frame drops. Textures (pool slots) and the fence are *reused* across many
/// frames: `texture_id` / `fence_id` are process-unique, never-recycled
/// identities for exactly that reuse, letting importers cache what they've
/// opened instead of re-opening per frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct D3d11SharedSurface {
    /// NT shared handle to the NV12 texture (valid in this process).
    pub texture_handle: u64,
    /// Stable identity of the texture behind `texture_handle`.
    pub texture_id: u64,
    /// NT shared handle to the `ID3D11Fence` guarding the texture contents.
    pub fence_handle: u64,
    /// Stable identity of the fence behind `fence_handle`.
    pub fence_id: u64,
    /// Fence value signaled after this frame's copy; wait for it before
    /// sampling.
    pub fence_value: u64,
}

/// An opaque, zero-copy GPU surface produced by a native decoder.
///
/// `handle` is a raw native pointer reinterpreted as a 64-bit integer (the same
/// convention the renderer uses to receive a `CAMetalLayer`). It is only
/// meaningful to the import path selected by [`GpuSurfaceKind`]. The retained
/// `keep_alive` wrapper (if any) drops with this surface, releasing the
/// decoder's buffer once no longer needed.
pub struct GpuSurface {
    kind: GpuSurfaceKind,
    handle: u64,
    keep_alive: Option<KeepAlive>,
}

impl GpuSurface {
    /// Wrap a native surface. `keep_alive` should hold whatever retained
    /// platform object keeps `handle` valid (pass `None` only if the caller
    /// guarantees the surface's lifetime by other means).
    pub fn new(kind: GpuSurfaceKind, handle: u64, keep_alive: Option<KeepAlive>) -> Self {
        Self {
            kind,
            handle,
            keep_alive,
        }
    }

    pub fn kind(&self) -> GpuSurfaceKind {
        self.kind
    }

    /// The raw native handle as an integer. The renderer reinterprets this as
    /// the pointer type implied by [`GpuSurface::kind`].
    pub fn handle(&self) -> u64 {
        self.handle
    }
}

impl fmt::Debug for GpuSurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuSurface")
            .field("kind", &self.kind)
            .field("handle", &format_args!("{:#x}", self.handle))
            .field("retained", &self.keep_alive.is_some())
            .finish()
    }
}

/// The pixel payload of a [`VideoFrame`]: GPU-resident (zero-copy) or CPU planes.
#[derive(Debug)]
pub enum FrameData {
    Cpu(CpuImage),
    Gpu(GpuSurface),
}

/// One decoded frame plus everything needed to interpret and place it.
#[derive(Debug)]
pub struct VideoFrame {
    /// Presentation timestamp on the source's own time base.
    pub pts: RationalTime,
    /// Frame duration if the decoder reports it (variable-frame-rate sources).
    pub duration: Option<RationalTime>,
    pub format: PixelFormat,
    pub color: ColorSpace,
    /// Full decoded buffer size in pixels (may be padded past the visible area).
    pub coded_size: (u32, u32),
    /// Visible region within the coded buffer (before rotation).
    pub visible: Rect,
    /// Display rotation to apply when presenting.
    pub rotation: Rotation,
    pub data: FrameData,
}

impl VideoFrame {
    /// Construct a frame with no reported duration.
    pub fn new(
        pts: RationalTime,
        format: PixelFormat,
        color: ColorSpace,
        coded_size: (u32, u32),
        visible: Rect,
        rotation: Rotation,
        data: FrameData,
    ) -> Self {
        Self {
            pts,
            duration: None,
            format,
            color,
            coded_size,
            visible,
            rotation,
            data,
        }
    }

    pub fn with_duration(mut self, duration: RationalTime) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Visible width in pixels (before rotation).
    pub fn width(&self) -> u32 {
        self.visible.width
    }

    /// Visible height in pixels (before rotation).
    pub fn height(&self) -> u32 {
        self.visible.height
    }

    /// Size as actually displayed, accounting for a 90°/270° axis swap.
    pub fn display_size(&self) -> (u32, u32) {
        if self.rotation.swaps_axes() {
            (self.visible.height, self.visible.width)
        } else {
            (self.visible.width, self.visible.height)
        }
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self.data, FrameData::Gpu(_))
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self.data, FrameData::Cpu(_))
    }

    /// The CPU planes, if this is a CPU frame.
    pub fn cpu(&self) -> Option<&CpuImage> {
        match &self.data {
            FrameData::Cpu(image) => Some(image),
            FrameData::Gpu(_) => None,
        }
    }

    /// The GPU surface, if this is a GPU frame.
    pub fn gpu(&self) -> Option<&GpuSurface> {
        match &self.data {
            FrameData::Gpu(surface) => Some(surface),
            FrameData::Cpu(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::Rational;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn cpu_nv12_frame() -> VideoFrame {
        // 4x4 NV12: a 4x4 Y plane + a 2x2 interleaved UV plane (8 bytes).
        let y = Plane::new(vec![16u8; 16], 4, 4);
        let uv = Plane::new(vec![128u8; 8], 4, 2);
        VideoFrame::new(
            RationalTime::new(0, Rational::FPS_30),
            PixelFormat::Nv12,
            ColorSpace::BT709,
            (4, 4),
            Rect::from_size(4, 4),
            Rotation::None,
            FrameData::Cpu(CpuImage::new(vec![y, uv])),
        )
    }

    #[test]
    fn cpu_frame_accessors() {
        let f = cpu_nv12_frame();
        assert!(f.is_cpu());
        assert!(!f.is_gpu());
        assert_eq!(f.width(), 4);
        assert_eq!(f.height(), 4);
        assert_eq!(f.display_size(), (4, 4));
        let img = f.cpu().expect("cpu image");
        assert_eq!(img.planes.len(), 2);
        assert!(img.planes.iter().all(Plane::is_well_formed));
        assert!(f.gpu().is_none());
    }

    #[test]
    fn rotation_swaps_display_size() {
        let mut f = cpu_nv12_frame();
        f.visible = Rect::from_size(1920, 1080);
        f.rotation = Rotation::Cw90;
        assert_eq!(f.width(), 1920);
        assert_eq!(f.display_size(), (1080, 1920));
    }

    #[test]
    fn plane_well_formedness() {
        assert!(Plane::new(vec![0; 16], 4, 4).is_well_formed());
        // stride*rows exceeds the buffer.
        assert!(!Plane::new(vec![0; 10], 4, 4).is_well_formed());
    }

    /// A retained wrapper parked in `keep_alive` must drop with the surface,
    /// which is what releases the decoder's buffer in real backends.
    #[test]
    fn gpu_surface_keep_alive_drops_with_frame() {
        struct DropFlag(Arc<AtomicUsize>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        let surface = GpuSurface::new(
            GpuSurfaceKind::CoreVideoPixelBuffer,
            0xDEAD_BEEF,
            Some(Box::new(DropFlag(drops.clone()))),
        );
        let frame = VideoFrame::new(
            RationalTime::new(0, Rational::FPS_30),
            PixelFormat::Nv12,
            ColorSpace::BT709,
            (1920, 1080),
            Rect::from_size(1920, 1080),
            Rotation::None,
            FrameData::Gpu(surface),
        );

        assert!(frame.is_gpu());
        let s = frame.gpu().expect("gpu surface");
        assert_eq!(s.kind(), GpuSurfaceKind::CoreVideoPixelBuffer);
        assert_eq!(s.handle(), 0xDEAD_BEEF);
        assert_eq!(drops.load(Ordering::SeqCst), 0);

        drop(frame);
        assert_eq!(drops.load(Ordering::SeqCst), 1, "keep_alive must drop");
    }

    /// The frame must be `Send + Sync` so it can move from a decode worker to
    /// the render thread.
    #[test]
    fn video_frame_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VideoFrame>();
    }
}
