//! A decoded picture that stays in a platform GPU buffer.
//!
//! A platform decoder may hand a picture over as a buffer the GPU can
//! sample, so preview draws it without copying it to the CPU. Everything
//! that needs bytes asks the buffer for packed 8-bit 4:2:0, Y then U then
//! V, as a CPU picture stores [`PixelFormat::Yuv420p`](crate::PixelFormat).
//! The buffer copies them out once and keeps that copy.
//!
//! The buffer is released when the last [`Surface`] naming it is dropped.
//! A renderer that can import it reaches the platform type through
//! [`PictureSurface::as_any`].

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use cutlass_codec::access::CodecError;

/// What a platform crate implements for its GPU buffer.
pub trait PictureSurface: Send + Sync + fmt::Debug {
    /// Packed 8-bit 4:2:0, Y then U then V. The first call copies; later
    /// calls return the same copy.
    fn yuv420p(&self) -> Result<Arc<Vec<u8>>, CodecError>;

    /// The platform type, for the renderer that imports it.
    fn as_any(&self) -> &dyn Any;
}

/// A shared handle to one platform picture. Two handles are equal when they
/// name the same buffer.
#[derive(Clone)]
pub struct Surface(Arc<dyn PictureSurface>);

impl Surface {
    #[must_use]
    pub fn new(surface: Arc<dyn PictureSurface>) -> Self {
        Self(surface)
    }

    /// See [`PictureSurface::yuv420p`].
    pub fn yuv420p(&self) -> Result<Arc<Vec<u8>>, CodecError> {
        self.0.yuv420p()
    }

    /// See [`PictureSurface::as_any`].
    #[must_use]
    pub fn as_any(&self) -> &dyn Any {
        self.0.as_any()
    }
}

impl PartialEq for Surface {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Surface {}

impl fmt::Debug for Surface {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}
