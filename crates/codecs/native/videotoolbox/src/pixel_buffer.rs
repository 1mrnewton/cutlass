//! `CVPixelBuffer` and `IOSurface` handoff to wgpu without a CPU copy.

use std::ffi::c_void;
use std::ptr::{self, NonNull};
use std::slice;

use cutlass_core::{ColorRange, PixelFormat};
use objc2_core_foundation::{
    CFArray, CFBoolean, CFDictionary, CFNumber, CFRetained, CFString, CFType,
};
use objc2_core_video::{
    CVPixelBuffer, CVPixelBufferCreate, CVPixelBufferGetBaseAddress,
    CVPixelBufferGetBaseAddressOfPlane, CVPixelBufferGetBytesPerRow,
    CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferGetHeight, CVPixelBufferGetHeightOfPlane,
    CVPixelBufferGetIOSurface, CVPixelBufferGetPixelFormatType, CVPixelBufferGetPlaneCount,
    CVPixelBufferGetWidth, CVPixelBufferGetWidthOfPlane, CVPixelBufferLockBaseAddress,
    CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress, kCVPixelBufferHeightKey,
    kCVPixelBufferIOSurfacePropertiesKey, kCVPixelBufferMetalCompatibilityKey,
    kCVPixelBufferPixelFormatTypeKey, kCVPixelBufferWidthKey,
    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
    kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
    kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
    kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
};
use objc2_io_surface::IOSurfaceRef;

use crate::cf::{dictionary, owned};
use crate::error::{Error, check};

const MAX_PLANES: usize = 3;

const FORMATS: [(PixelFormat, ColorRange); 4] = [
    (PixelFormat::Nv12, ColorRange::Limited),
    (PixelFormat::Nv12, ColorRange::Full),
    (PixelFormat::P010, ColorRange::Limited),
    (PixelFormat::P010, ColorRange::Full),
];

/// An image owned by Core Video. It is backed by an IOSurface, so the GPU can
/// read it without a copy.
#[derive(Debug)]
pub struct PixelBuffer {
    buffer: CFRetained<CVPixelBuffer>,
    writable: bool,
}

// SAFETY: Core Video buffers use atomic reference counts, and VideoToolbox
// itself passes them between threads. Pixel access goes through
// `read_planes` and `write_planes`, which hold the buffer's lock, and writing
// needs `&mut self` on a buffer nothing else is using.
unsafe impl Send for PixelBuffer {}
// SAFETY: as above. Shared access only takes read-only locks, which Core Video
// allows from several threads at once.
unsafe impl Sync for PixelBuffer {}

/// One plane of a locked buffer.
#[derive(Debug, Default)]
pub struct Plane<'a> {
    /// `height` rows of `bytes_per_row` bytes each.
    pub data: &'a [u8],
    /// Distance from one row to the next. Can be more than the visible width.
    pub bytes_per_row: usize,
    /// Size in plane pixels. In an interleaved chroma plane, one pixel is a Cb
    /// and Cr pair.
    pub width: usize,
    pub height: usize,
}

/// One plane of a buffer locked for writing.
#[derive(Debug, Default)]
pub struct PlaneMut<'a> {
    /// `height` rows of `bytes_per_row` bytes each.
    pub data: &'a mut [u8],
    pub bytes_per_row: usize,
    pub width: usize,
    pub height: usize,
}

impl PixelBuffer {
    /// Allocates an IOSurface-backed buffer the caller can fill.
    pub fn new(
        width: u32,
        height: u32,
        format: PixelFormat,
        range: ColorRange,
    ) -> Result<Self, Error> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidInput(
                "pixel buffer width and height must be nonzero",
            ));
        }
        let attributes = attributes(&[], None);
        let mut buffer = ptr::null_mut();
        // SAFETY: `attributes` maps CFString keys to the value types Core Video
        // documents for them, and `buffer` is a valid place for the result.
        let status = unsafe {
            CVPixelBufferCreate(
                None,
                width as usize,
                height as usize,
                os_type(format, range),
                Some(attributes.as_opaque()),
                NonNull::from(&mut buffer),
            )
        };
        check("CVPixelBufferCreate", status)?;
        // SAFETY: Create functions return an owned reference.
        let buffer = unsafe { owned(buffer, "CVPixelBufferCreate") }?;
        Ok(Self::from_retained(buffer, true))
    }

    /// `writable` is true only when nothing else reads or writes the pixels,
    /// such as a fresh pool buffer or a decoder output marked modifiable.
    pub(crate) fn from_retained(buffer: CFRetained<CVPixelBuffer>, writable: bool) -> Self {
        Self { buffer, writable }
    }

    pub(crate) fn as_cv(&self) -> &CVPixelBuffer {
        &self.buffer
    }

    pub fn width(&self) -> usize {
        CVPixelBufferGetWidth(&self.buffer)
    }

    pub fn height(&self) -> usize {
        CVPixelBufferGetHeight(&self.buffer)
    }

    /// The IOSurface behind this buffer. `None` when the buffer has none.
    ///
    /// The GPU samples this instead of copying the pixels. The returned value
    /// retains the surface.
    pub fn io_surface(&self) -> Option<CFRetained<IOSurfaceRef>> {
        CVPixelBufferGetIOSurface(Some(&self.buffer))
    }

    /// `None` for Core Video formats Cutlass doesn't use.
    pub fn format(&self) -> Option<(PixelFormat, ColorRange)> {
        let os_type = CVPixelBufferGetPixelFormatType(&self.buffer);
        FORMATS
            .into_iter()
            .find(|&(format, range)| self::os_type(format, range) == os_type)
    }

    /// Reads the pixels on the CPU. This is the readback path; the GPU path
    /// never locks the buffer.
    pub fn read_planes<R>(&self, read: impl FnOnce(&[Plane<'_>]) -> R) -> Result<R, Error> {
        let _lock = Lock::new(&self.buffer, CVPixelBufferLockFlags::ReadOnly)?;
        let (raw, count) = raw_planes(&self.buffer)?;
        let planes: [Plane<'_>; MAX_PLANES] = std::array::from_fn(|i| {
            let plane = raw[i];
            Plane {
                // SAFETY: the buffer is locked until `_lock` drops, after
                // `read` returns, and `read` can't keep the slice. Core Video
                // allocates each plane as `height` rows of `bytes_per_row`.
                data: unsafe { slice::from_raw_parts(plane.base.as_ptr(), plane.len) },
                bytes_per_row: plane.bytes_per_row,
                width: plane.width,
                height: plane.height,
            }
        });
        Ok(read(&planes[..count]))
    }

    /// Fills the pixels from the CPU. Fails on buffers other code may still
    /// read, such as decoder output the decoder keeps as a reference frame.
    pub fn write_planes<R>(
        &mut self,
        write: impl FnOnce(&mut [PlaneMut<'_>]) -> R,
    ) -> Result<R, Error> {
        if !self.writable {
            return Err(Error::InvalidInput(
                "this pixel buffer is shared and read-only",
            ));
        }
        let _lock = Lock::new(&self.buffer, CVPixelBufferLockFlags::empty())?;
        let (raw, count) = raw_planes(&self.buffer)?;
        let mut planes: [PlaneMut<'_>; MAX_PLANES] = std::array::from_fn(|i| {
            let plane = raw[i];
            PlaneMut {
                // SAFETY: as in `read_planes`. Planes don't overlap, and
                // `writable` with `&mut self` means no one else touches them.
                data: unsafe { slice::from_raw_parts_mut(plane.base.as_ptr(), plane.len) },
                bytes_per_row: plane.bytes_per_row,
                width: plane.width,
                height: plane.height,
            }
        });
        Ok(write(&mut planes[..count]))
    }
}

/// Buffer attributes for VideoToolbox: IOSurface-backed and usable by Metal.
/// An empty `formats` leaves the format to the other side.
pub(crate) fn attributes(
    formats: &[(PixelFormat, ColorRange)],
    size: Option<(u32, u32)>,
) -> CFRetained<CFDictionary<CFString, CFType>> {
    let format_numbers: Vec<CFRetained<CFNumber>> = formats
        .iter()
        .map(|&(format, range)| CFNumber::new_i64(os_type(format, range).into()))
        .collect();
    let format_array = CFArray::from_retained_objects(&format_numbers);
    let io_surface = CFDictionary::<CFString, CFType>::empty();
    let size_numbers = size.map(|(width, height)| {
        (
            CFNumber::new_i64(width.into()),
            CFNumber::new_i64(height.into()),
        )
    });

    // SAFETY: these are immutable CFString constants exported by Core Video.
    let (format_key, io_surface_key, metal_key, width_key, height_key) = unsafe {
        (
            kCVPixelBufferPixelFormatTypeKey,
            kCVPixelBufferIOSurfacePropertiesKey,
            kCVPixelBufferMetalCompatibilityKey,
            kCVPixelBufferWidthKey,
            kCVPixelBufferHeightKey,
        )
    };
    let mut entries: Vec<(&CFString, &CFType)> = vec![
        (io_surface_key, io_surface.as_ref()),
        (metal_key, CFBoolean::new(true).as_ref()),
    ];
    if !formats.is_empty() {
        entries.push((format_key, format_array.as_ref()));
    }
    if let Some((width, height)) = &size_numbers {
        entries.push((width_key, width.as_ref()));
        entries.push((height_key, height.as_ref()));
    }
    dictionary(&entries)
}

pub(crate) fn os_type(format: PixelFormat, range: ColorRange) -> u32 {
    match (format, range) {
        (PixelFormat::Nv12, ColorRange::Limited) => kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        (PixelFormat::Nv12, ColorRange::Full) => kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        (PixelFormat::P010, ColorRange::Limited) => {
            kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
        }
        (PixelFormat::P010, ColorRange::Full) => kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
    }
}

/// Holds a base-address lock and releases it on drop.
struct Lock<'a> {
    buffer: &'a CVPixelBuffer,
    flags: CVPixelBufferLockFlags,
}

impl<'a> Lock<'a> {
    fn new(buffer: &'a CVPixelBuffer, flags: CVPixelBufferLockFlags) -> Result<Self, Error> {
        // SAFETY: `buffer` is live, and `Drop` pairs this lock with an unlock
        // using the same flags.
        let status = unsafe { CVPixelBufferLockBaseAddress(buffer, flags) };
        check("CVPixelBufferLockBaseAddress", status)?;
        Ok(Self { buffer, flags })
    }
}

impl Drop for Lock<'_> {
    fn drop(&mut self) {
        // Unlocking fails only for a buffer that isn't locked, which this
        // guard rules out.
        // SAFETY: `new` locked the buffer with these flags.
        unsafe { CVPixelBufferUnlockBaseAddress(self.buffer, self.flags) };
    }
}

#[derive(Clone, Copy)]
struct RawPlane {
    base: NonNull<u8>,
    len: usize,
    bytes_per_row: usize,
    width: usize,
    height: usize,
}

impl RawPlane {
    const EMPTY: Self = Self {
        base: NonNull::dangling(),
        len: 0,
        bytes_per_row: 0,
        width: 0,
        height: 0,
    };

    fn new(
        base: *mut c_void,
        bytes_per_row: usize,
        width: usize,
        height: usize,
    ) -> Result<Self, Error> {
        let base = NonNull::new(base.cast::<u8>()).ok_or(Error::NullOutput {
            call: "CVPixelBufferGetBaseAddressOfPlane",
        })?;
        let len = bytes_per_row
            .checked_mul(height)
            .ok_or(Error::InvalidInput(
                "pixel buffer plane size overflows usize",
            ))?;
        Ok(Self {
            base,
            len,
            bytes_per_row,
            width,
            height,
        })
    }
}

/// Plane addresses of a locked buffer. A non-planar buffer counts as one plane.
fn raw_planes(buffer: &CVPixelBuffer) -> Result<([RawPlane; MAX_PLANES], usize), Error> {
    let mut planes = [RawPlane::EMPTY; MAX_PLANES];
    let count = CVPixelBufferGetPlaneCount(buffer);
    if count == 0 {
        planes[0] = RawPlane::new(
            CVPixelBufferGetBaseAddress(buffer),
            CVPixelBufferGetBytesPerRow(buffer),
            CVPixelBufferGetWidth(buffer),
            CVPixelBufferGetHeight(buffer),
        )?;
        return Ok((planes, 1));
    }
    if count > MAX_PLANES {
        return Err(Error::Unsupported(
            "pixel buffers with more than three planes",
        ));
    }
    for (index, plane) in planes.iter_mut().enumerate().take(count) {
        *plane = RawPlane::new(
            CVPixelBufferGetBaseAddressOfPlane(buffer, index),
            CVPixelBufferGetBytesPerRowOfPlane(buffer, index),
            CVPixelBufferGetWidthOfPlane(buffer, index),
            CVPixelBufferGetHeightOfPlane(buffer, index),
        )?;
    }
    Ok((planes, count))
}
