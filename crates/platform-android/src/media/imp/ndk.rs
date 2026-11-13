//! The NDK media calls this crate makes, and owners that free what they open.
//!
//! Every pointer the NDK returns is checked for null and owned by one value
//! here, which deletes it on drop. A codec's input and output buffers are
//! lent to a closure for the time between asking for the buffer and handing
//! it back, which is the time MediaCodec keeps them valid. A codec is used
//! from one thread at a time: it is `Send`, so a decoder can move to a
//! worker, and not `Sync`.

use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr::{self, NonNull};
use std::slice;

use crate::planes::Described;

/// `AMEDIACODEC_INFO_TRY_AGAIN_LATER`.
const TRY_AGAIN: isize = -1;
/// `AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED`.
const FORMAT_CHANGED: isize = -2;
/// `AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED`.
const BUFFERS_CHANGED: isize = -3;
/// `AMEDIACODEC_BUFFER_FLAG_KEY_FRAME`.
pub const FLAG_KEY_FRAME: u32 = 1;
/// `AMEDIACODEC_BUFFER_FLAG_CODEC_CONFIG`.
pub const FLAG_CODEC_CONFIG: u32 = 2;
/// `AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM`.
pub const FLAG_END_OF_STREAM: u32 = 4;
/// `AMEDIACODEC_CONFIGURE_FLAG_ENCODE`.
const CONFIGURE_ENCODE: u32 = 1;
/// `AMEDIA_OK`.
const OK: i32 = 0;

/// `AMediaCodecBufferInfo`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferInfo {
    pub offset: i32,
    pub size: i32,
    pub presentation_us: i64,
    pub flags: u32,
}

#[link(name = "mediandk")]
unsafe extern "C" {
    fn AMediaCodec_createDecoderByType(mime: *const c_char) -> *mut c_void;
    fn AMediaCodec_createEncoderByType(mime: *const c_char) -> *mut c_void;
    fn AMediaCodec_createCodecByName(name: *const c_char) -> *mut c_void;
    fn AMediaCodec_delete(codec: *mut c_void) -> i32;
    fn AMediaCodec_configure(
        codec: *mut c_void,
        format: *const c_void,
        surface: *mut c_void,
        crypto: *mut c_void,
        flags: u32,
    ) -> i32;
    fn AMediaCodec_start(codec: *mut c_void) -> i32;
    fn AMediaCodec_stop(codec: *mut c_void) -> i32;
    fn AMediaCodec_flush(codec: *mut c_void) -> i32;
    fn AMediaCodec_getInputBuffer(codec: *mut c_void, index: usize, size: *mut usize) -> *mut u8;
    fn AMediaCodec_getOutputBuffer(codec: *mut c_void, index: usize, size: *mut usize) -> *mut u8;
    fn AMediaCodec_dequeueInputBuffer(codec: *mut c_void, timeout_us: i64) -> isize;
    fn AMediaCodec_queueInputBuffer(
        codec: *mut c_void,
        index: usize,
        offset: isize,
        size: usize,
        time_us: u64,
        flags: u32,
    ) -> i32;
    fn AMediaCodec_dequeueOutputBuffer(
        codec: *mut c_void,
        info: *mut BufferInfo,
        timeout_us: i64,
    ) -> isize;
    fn AMediaCodec_getOutputFormat(codec: *mut c_void) -> *mut c_void;
    fn AMediaCodec_releaseOutputBuffer(codec: *mut c_void, index: usize, render: bool) -> i32;
    fn AMediaCodec_getName(codec: *mut c_void, name: *mut *mut c_char) -> i32;
    fn AMediaCodec_createInputSurface(codec: *mut c_void, surface: *mut *mut c_void) -> i32;
    fn AMediaCodec_signalEndOfInputStream(codec: *mut c_void) -> i32;
    fn AMediaCodec_releaseName(codec: *mut c_void, name: *mut c_char);

    fn AMediaFormat_new() -> *mut c_void;
    fn AMediaFormat_delete(format: *mut c_void) -> i32;
    fn AMediaFormat_setString(format: *mut c_void, name: *const c_char, value: *const c_char);
    fn AMediaFormat_setInt32(format: *mut c_void, name: *const c_char, value: i32);
    fn AMediaFormat_setBuffer(
        format: *mut c_void,
        name: *const c_char,
        data: *const c_void,
        size: usize,
    );
    fn AMediaFormat_getInt32(format: *mut c_void, name: *const c_char, out: *mut i32) -> bool;
    fn AMediaFormat_getRect(
        format: *mut c_void,
        name: *const c_char,
        left: *mut i32,
        top: *mut i32,
        right: *mut i32,
        bottom: *mut i32,
    ) -> bool;
    fn AMediaFormat_getBuffer(
        format: *mut c_void,
        name: *const c_char,
        data: *mut *mut c_void,
        size: *mut usize,
    ) -> bool;
}

/// `ANativeWindow_Buffer`.
#[repr(C)]
struct WindowBuffer {
    width: i32,
    height: i32,
    stride: i32,
    format: i32,
    bits: *mut c_void,
    reserved: [u32; 6],
}

/// `AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM`, also `WINDOW_FORMAT_RGBA_8888`.
const WINDOW_RGBA: i32 = 1;

#[link(name = "android")]
unsafe extern "C" {
    fn ANativeWindow_setBuffersGeometry(
        window: *mut c_void,
        width: i32,
        height: i32,
        format: i32,
    ) -> i32;
    fn ANativeWindow_lock(
        window: *mut c_void,
        buffer: *mut WindowBuffer,
        dirty: *mut c_void,
    ) -> i32;
    fn ANativeWindow_unlockAndPost(window: *mut c_void) -> i32;
    fn ANativeWindow_release(window: *mut c_void);
}

/// An encoder's input surface, released on drop.
pub struct Window(NonNull<c_void>);

// SAFETY: a window may be locked and posted from any thread; `Window` is
// not `Sync`, and each call takes `&mut self`.
unsafe impl Send for Window {}

impl Window {
    /// Fix the buffers at `width` by `height` RGBA.
    pub fn set_rgba(&mut self, width: u32, height: u32) -> Result<(), Status> {
        let width = i32::try_from(width).map_err(|_| Status(-1))?;
        let height = i32::try_from(height).map_err(|_| Status(-1))?;
        // SAFETY: the window is live.
        status(unsafe {
            ANativeWindow_setBuffersGeometry(self.0.as_ptr(), width, height, WINDOW_RGBA)
        })
    }

    /// Lock the next buffer, let `fill` write its rows, then post it.
    /// `fill` gets the buffer and the byte step between rows.
    pub fn post<R>(&mut self, fill: impl FnOnce(&mut [u8], usize) -> R) -> Result<R, Status> {
        let mut buffer = WindowBuffer {
            width: 0,
            height: 0,
            stride: 0,
            format: 0,
            bits: ptr::null_mut(),
            reserved: [0; 6],
        };
        // SAFETY: the window is live and `buffer` is a valid place; no dirty rectangle.
        status(unsafe { ANativeWindow_lock(self.0.as_ptr(), &raw mut buffer, ptr::null_mut()) })?;
        let shaped = buffer.format == WINDOW_RGBA && !buffer.bits.is_null();
        let rows = usize::try_from(buffer.height).unwrap_or(0);
        let step = usize::try_from(buffer.stride)
            .unwrap_or(0)
            .saturating_mul(4);
        let result = if shaped && rows > 0 && step > 0 {
            // SAFETY: a locked RGBA buffer holds `stride * height` pixels of
            // four bytes at `bits` until it is posted, after this borrow.
            let bytes = unsafe { slice::from_raw_parts_mut(buffer.bits.cast::<u8>(), step * rows) };
            Ok(fill(bytes, step))
        } else {
            Err(Status(-1))
        };
        // SAFETY: the window was locked above.
        let posted = status(unsafe { ANativeWindow_unlockAndPost(self.0.as_ptr()) });
        let result = result?;
        posted.map(|()| result)
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        // SAFETY: the window is owned here and released once.
        unsafe { ANativeWindow_release(self.0.as_ptr()) }
    }
}

/// One `AMediaFormat`.
pub struct Format(NonNull<c_void>);

impl Format {
    pub fn new() -> Option<Self> {
        // SAFETY: no arguments; a null return is checked.
        NonNull::new(unsafe { AMediaFormat_new() }).map(Self)
    }

    fn raw(&self) -> *mut c_void {
        self.0.as_ptr()
    }

    pub fn set_str(&self, key: &CStr, value: &CStr) {
        // SAFETY: the format is live and both strings are NUL-terminated.
        // The NDK copies the value.
        unsafe { AMediaFormat_setString(self.raw(), key.as_ptr(), value.as_ptr()) }
    }

    pub fn set_i32(&self, key: &CStr, value: i32) {
        // SAFETY: the format is live and the key is NUL-terminated.
        unsafe { AMediaFormat_setInt32(self.raw(), key.as_ptr(), value) }
    }

    pub fn set_buffer(&self, key: &CStr, data: &[u8]) {
        // SAFETY: the format is live, the key is NUL-terminated, and `data`
        // is valid for its length. The NDK copies the bytes.
        unsafe {
            AMediaFormat_setBuffer(
                self.raw(),
                key.as_ptr(),
                data.as_ptr().cast::<c_void>(),
                data.len(),
            );
        }
    }

    pub fn i32(&self, key: &CStr) -> Option<i32> {
        let mut out = 0_i32;
        // SAFETY: the format is live, the key is NUL-terminated, and `out`
        // is a valid place for one integer.
        unsafe { AMediaFormat_getInt32(self.raw(), key.as_ptr(), &raw mut out) }.then_some(out)
    }

    pub fn rect(&self, key: &CStr) -> Option<[i32; 4]> {
        let mut rect = [0_i32; 4];
        let [left, top, right, bottom] = &mut rect;
        // SAFETY: the format is live, the key is NUL-terminated, and each
        // pointer names its own integer.
        let found = unsafe {
            AMediaFormat_getRect(
                self.raw(),
                key.as_ptr(),
                ptr::from_mut(left),
                ptr::from_mut(top),
                ptr::from_mut(right),
                ptr::from_mut(bottom),
            )
        };
        found.then_some(rect)
    }

    pub fn buffer(&self, key: &CStr) -> Option<Vec<u8>> {
        let mut data = ptr::null_mut::<c_void>();
        let mut size = 0_usize;
        // SAFETY: the format is live and the key is NUL-terminated. The
        // returned pointer is owned by the format and valid while it lives.
        let found = unsafe {
            AMediaFormat_getBuffer(self.raw(), key.as_ptr(), &raw mut data, &raw mut size)
        };
        if !found || data.is_null() {
            return None;
        }
        // SAFETY: the NDK said `data` holds `size` bytes, and the format
        // outlives this copy.
        Some(unsafe { slice::from_raw_parts(data.cast::<u8>(), size) }.to_vec())
    }

    /// The picture fields a decoder's output or an encoder's input names.
    pub fn described(&self) -> Described {
        Described {
            width: self.i32(c"width").unwrap_or(0),
            height: self.i32(c"height").unwrap_or(0),
            color_format: self.i32(c"color-format").unwrap_or(0),
            stride: self.i32(c"stride").unwrap_or(0),
            slice_height: self.i32(c"slice-height").unwrap_or(0),
            crop: self.rect(c"crop").or_else(|| {
                Some([
                    self.i32(c"crop-left")?,
                    self.i32(c"crop-top")?,
                    self.i32(c"crop-right")?,
                    self.i32(c"crop-bottom")?,
                ])
            }),
            image: self.buffer(c"image-data"),
        }
    }
}

impl Drop for Format {
    fn drop(&mut self) {
        // SAFETY: the format is owned here and deleted once.
        let _ = unsafe { AMediaFormat_delete(self.raw()) };
    }
}

/// A MediaCodec status that was not `AMEDIA_OK`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Status(pub i32);

fn status(code: i32) -> Result<(), Status> {
    if code == OK {
        Ok(())
    } else {
        Err(Status(code))
    }
}

/// What one `dequeueOutputBuffer` returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dequeued {
    Buffer { index: usize, info: BufferInfo },
    FormatChanged,
    TryAgain,
    Failed(isize),
}

/// One `AMediaCodec`, stopped and deleted on drop.
pub struct Codec {
    raw: NonNull<c_void>,
    started: bool,
}

// SAFETY: MediaCodec may be called from any thread as long as calls are not
// concurrent. `Codec` is not `Sync`, and every call takes `&mut self` or
// happens before the value is shared.
unsafe impl Send for Codec {}

impl Codec {
    /// The platform's choice for `mime`, hardware first where it has one.
    pub fn by_type(mime: &CStr, encoder: bool) -> Option<Self> {
        // SAFETY: `mime` is NUL-terminated; a null return is checked.
        let raw = unsafe {
            if encoder {
                AMediaCodec_createEncoderByType(mime.as_ptr())
            } else {
                AMediaCodec_createDecoderByType(mime.as_ptr())
            }
        };
        NonNull::new(raw).map(|raw| Self {
            raw,
            started: false,
        })
    }

    /// One codec by its registered name.
    pub fn by_name(name: &str) -> Option<Self> {
        let name = CString::new(name).ok()?;
        // SAFETY: `name` is NUL-terminated; a null return is checked.
        NonNull::new(unsafe { AMediaCodec_createCodecByName(name.as_ptr()) }).map(|raw| Self {
            raw,
            started: false,
        })
    }

    fn raw(&self) -> *mut c_void {
        self.raw.as_ptr()
    }

    /// The codec's registered name, such as `c2.android.avc.decoder`.
    pub fn name(&self) -> Option<String> {
        let mut name = ptr::null_mut::<c_char>();
        // SAFETY: the codec is live and `name` receives a string the codec owns.
        status(unsafe { AMediaCodec_getName(self.raw(), &raw mut name) }).ok()?;
        if name.is_null() {
            return None;
        }
        // SAFETY: the codec returned a NUL-terminated string, read before
        // it is handed back.
        let owned = unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned();
        // SAFETY: the string came from `getName` on this codec.
        unsafe { AMediaCodec_releaseName(self.raw(), name) };
        Some(owned)
    }

    pub fn configure(&mut self, format: &Format, encode: bool) -> Result<(), Status> {
        let flags = if encode { CONFIGURE_ENCODE } else { 0 };
        // SAFETY: the codec and format are live; no surface and no crypto.
        status(unsafe {
            AMediaCodec_configure(
                self.raw(),
                format.raw(),
                ptr::null_mut(),
                ptr::null_mut(),
                flags,
            )
        })
    }

    /// The surface an encoder reads its pictures from. Between configure and start.
    pub fn input_surface(&mut self) -> Result<Window, Status> {
        let mut window = ptr::null_mut::<c_void>();
        // SAFETY: the codec is live and configured as an encoder.
        status(unsafe { AMediaCodec_createInputSurface(self.raw(), &raw mut window) })?;
        NonNull::new(window).map(Window).ok_or(Status(-1))
    }

    /// No more pictures will come through the input surface.
    pub fn end_input(&mut self) -> Result<(), Status> {
        // SAFETY: the codec is live and started with an input surface.
        status(unsafe { AMediaCodec_signalEndOfInputStream(self.raw()) })
    }

    pub fn start(&mut self) -> Result<(), Status> {
        // SAFETY: the codec is live and configured.
        status(unsafe { AMediaCodec_start(self.raw()) })?;
        self.started = true;
        Ok(())
    }

    /// Drop every queued buffer. The configuration stays.
    pub fn flush(&mut self) -> Result<(), Status> {
        // SAFETY: the codec is live; flushing a started codec is allowed.
        status(unsafe { AMediaCodec_flush(self.raw()) })
    }

    /// An empty input buffer, waiting up to `timeout_us`.
    pub fn input_slot(&mut self, timeout_us: i64) -> Result<Option<usize>, Status> {
        // SAFETY: the codec is live and started.
        let index = unsafe { AMediaCodec_dequeueInputBuffer(self.raw(), timeout_us) };
        if index == TRY_AGAIN {
            return Ok(None);
        }
        usize::try_from(index)
            .map(Some)
            .map_err(|_| Status(i32::try_from(index).unwrap_or(i32::MIN)))
    }

    /// Fill input buffer `index` through `fill`, which returns the bytes it
    /// wrote. The buffer is not queued.
    pub fn with_input<R>(&mut self, index: usize, fill: impl FnOnce(&mut [u8]) -> R) -> Option<R> {
        let mut size = 0_usize;
        // SAFETY: the codec is live and `index` came from `input_slot`.
        let data = unsafe { AMediaCodec_getInputBuffer(self.raw(), index, &raw mut size) };
        if data.is_null() {
            return None;
        }
        // SAFETY: the codec lends `size` bytes at `data` until the buffer is
        // queued, which only `queue_input` does, after this borrow ends.
        let buffer = unsafe { slice::from_raw_parts_mut(data, size) };
        Some(fill(buffer))
    }

    pub fn queue_input(
        &mut self,
        index: usize,
        size: usize,
        time_us: i64,
        flags: u32,
    ) -> Result<(), Status> {
        let time = u64::try_from(time_us.max(0)).unwrap_or(0);
        // SAFETY: the codec is live, `index` came from `input_slot`, and
        // `size` is at most the bytes `with_input` wrote.
        status(unsafe { AMediaCodec_queueInputBuffer(self.raw(), index, 0, size, time, flags) })
    }

    pub fn dequeue_output(&mut self, timeout_us: i64) -> Dequeued {
        let mut info = BufferInfo::default();
        loop {
            // SAFETY: the codec is live and `info` is a valid place.
            let index =
                unsafe { AMediaCodec_dequeueOutputBuffer(self.raw(), &raw mut info, timeout_us) };
            match index {
                TRY_AGAIN => return Dequeued::TryAgain,
                FORMAT_CHANGED => return Dequeued::FormatChanged,
                // The NDK hands buffers out by index; a new set needs nothing.
                BUFFERS_CHANGED => {}
                index => {
                    return match usize::try_from(index) {
                        Ok(index) => Dequeued::Buffer { index, info },
                        Err(_) => Dequeued::Failed(index),
                    };
                }
            }
        }
    }

    /// Read output buffer `index` through `read`. The buffer is not released.
    pub fn with_output<R>(
        &mut self,
        index: usize,
        info: BufferInfo,
        read: impl FnOnce(&[u8]) -> R,
    ) -> Option<R> {
        let mut size = 0_usize;
        // SAFETY: the codec is live and `index` came from `dequeue_output`.
        let data = unsafe { AMediaCodec_getOutputBuffer(self.raw(), index, &raw mut size) };
        if data.is_null() {
            return None;
        }
        let start = usize::try_from(info.offset).ok()?;
        let len = usize::try_from(info.size).ok()?;
        let end = start.checked_add(len)?;
        if end > size {
            return None;
        }
        // SAFETY: the codec lends `size` bytes at `data` until the buffer is
        // released, which only `release_output` does, after this borrow ends.
        let buffer = unsafe { slice::from_raw_parts(data, size) };
        Some(read(&buffer[start..end]))
    }

    pub fn release_output(&mut self, index: usize) {
        // SAFETY: the codec is live and `index` came from `dequeue_output`.
        let _ = unsafe { AMediaCodec_releaseOutputBuffer(self.raw(), index, false) };
    }

    pub fn output_format(&self) -> Option<Format> {
        // SAFETY: the codec is live; the caller owns the returned format.
        NonNull::new(unsafe { AMediaCodec_getOutputFormat(self.raw()) }).map(Format)
    }
}

impl Drop for Codec {
    fn drop(&mut self) {
        if self.started {
            // SAFETY: the codec is live and was started.
            let _ = unsafe { AMediaCodec_stop(self.raw()) };
        }
        // SAFETY: the codec is owned here and deleted once.
        let _ = unsafe { AMediaCodec_delete(self.raw()) };
    }
}
