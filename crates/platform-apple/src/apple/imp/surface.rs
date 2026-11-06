//! A decoded picture kept in its IOSurface-backed `CVPixelBuffer`.
//!
//! The buffer is video-range NV12 with even edges, and the surface holds one
//! retain on it. Preview samples it through [`super::metal`]; everything else
//! asks for packed 4:2:0, which is copied out the first time and shared.

use std::any::Any;
use std::fmt;
use std::sync::{Arc, OnceLock};

use cutlass_decode::{CodecError, PictureSurface};

use super::cf::Cf;
use super::sys;

pub(crate) struct VtSurface {
    buffer: Cf,
    width: u32,
    height: u32,
    cpu: OnceLock<Arc<Vec<u8>>>,
}

// SAFETY: the surface only retains, releases, and read-locks its buffer, and
// CoreVideo allows those from any thread. The buffer is never written.
unsafe impl Sync for VtSurface {}

impl VtSurface {
    /// Retain `image`, an NV12 buffer of exactly `width` by `height`.
    pub(crate) fn retain(
        image: *mut std::ffi::c_void,
        width: u32,
        height: u32,
    ) -> Result<Self, CodecError> {
        let retained = unsafe { sys::CFRetain(image) };
        let buffer = Cf::from_owned(retained.cast_mut()).ok_or(CodecError::Unsupported {
            what: "pixel buffer",
        })?;
        Ok(Self {
            buffer,
            width,
            height,
            cpu: OnceLock::new(),
        })
    }

    /// The retained `CVPixelBuffer`, valid while `self` is.
    pub(crate) fn buffer(&self) -> *mut std::ffi::c_void {
        self.buffer.as_mut()
    }

    pub(crate) fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

impl fmt::Debug for VtSurface {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VtSurface")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl PictureSurface for VtSurface {
    fn yuv420p(&self) -> Result<Arc<Vec<u8>>, CodecError> {
        if let Some(bytes) = self.cpu.get() {
            return Ok(Arc::clone(bytes));
        }
        let image = self.buffer();
        let locked = unsafe { sys::CVPixelBufferLockBaseAddress(image, 1) };
        if locked != 0 {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "pixel buffer",
            });
        }
        let copied = super::video::copy_nv12(image);
        unsafe {
            let _ = sys::CVPixelBufferUnlockBaseAddress(image, 1);
        }
        let bytes = Arc::new(copied?);
        // Two threads may copy at once; both copies are the same samples.
        Ok(Arc::clone(self.cpu.get_or_init(|| bytes)))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
