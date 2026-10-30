//! Metal and libobjc entry points. Callers cast `objc_msgSend` per signature.

use std::ffi::c_void;
use std::ptr;

/// `MTLPixelFormatRGBA8Unorm`.
pub(super) const RGBA8_UNORM: usize = 70;

/// `MTLCommandBufferStatusCompleted`. Anything else, including error (5), fails.
pub(super) const COMPLETED: usize = 4;

/// `MTLStorageModeShared` and `MTLResourceStorageModeShared`.
pub(super) const STORAGE_SHARED: usize = 0;

/// `MTLOrigin`: three `NSUInteger` fields.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Origin {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

/// `MTLSize`: three `NSUInteger` fields.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Size {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

const _: () = {
    use std::mem::{align_of, offset_of, size_of};
    assert!(size_of::<Origin>() == size_of::<[usize; 3]>());
    assert!(size_of::<Size>() == size_of::<[usize; 3]>());
    assert!(align_of::<Origin>() == align_of::<usize>());
    assert!(align_of::<Size>() == align_of::<usize>());
    assert!(offset_of!(Origin, x) == 0);
    assert!(offset_of!(Origin, y) == size_of::<usize>());
    assert!(offset_of!(Origin, z) == size_of::<usize>() * 2);
    assert!(offset_of!(Size, width) == 0);
    assert!(offset_of!(Size, height) == size_of::<usize>());
    assert!(offset_of!(Size, depth) == size_of::<usize>() * 2);
};

#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> *mut c_void;
}

#[link(name = "objc", kind = "dylib")]
unsafe extern "C" {
    fn objc_getClass(name: *const u8) -> *mut c_void;
    fn sel_registerName(name: *const u8) -> *mut c_void;
    fn objc_msgSend();
    fn objc_release(obj: *mut c_void);
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(pool: *mut c_void);
}

/// One transmute per concrete `objc_msgSend` signature.
macro_rules! msg {
    ($ty:ty) => {{
        const _: () = assert!(
            ::std::mem::size_of::<$ty>() == ::std::mem::size_of::<unsafe extern "C" fn()>()
        );
        let raw: unsafe extern "C" fn() = objc_msgSend;
        unsafe { ::std::mem::transmute::<unsafe extern "C" fn(), $ty>(raw) }
    }};
}

pub(super) fn sel(name: &[u8]) -> *mut c_void {
    debug_assert!(name.last().copied() == Some(0));
    unsafe { sel_registerName(name.as_ptr()) }
}

pub(super) fn release(obj: *mut c_void) {
    if !obj.is_null() {
        unsafe { objc_release(obj) };
    }
}

/// Drains autoreleased command buffers and encoders.
pub(super) struct Pool {
    pool: *mut c_void,
}

impl Pool {
    pub(super) fn push() -> Self {
        Self {
            pool: unsafe { objc_autoreleasePoolPush() },
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        unsafe { objc_autoreleasePoolPop(self.pool) };
    }
}

/// A +1 Metal object. Drop releases it once.
pub(super) struct Release(*mut c_void);

impl Release {
    /// # Safety
    /// `ptr` is null or a +1 object this guard uniquely owns.
    pub(super) unsafe fn new(ptr: *mut c_void) -> Self {
        Self(ptr)
    }

    pub(super) fn get(&self) -> *mut c_void {
        self.0
    }

    #[must_use]
    pub(super) fn take(mut self) -> *mut c_void {
        let ptr = self.0;
        self.0 = ptr::null_mut();
        ptr
    }
}

impl Drop for Release {
    fn drop(&mut self) {
        release(self.0);
    }
}

pub(super) unsafe fn system_device() -> *mut c_void {
    unsafe { MTLCreateSystemDefaultDevice() }
}

pub(super) unsafe fn new_command_queue(device: *mut c_void) -> *mut c_void {
    type New = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    let new: New = msg!(New);
    unsafe { new(device, sel(b"newCommandQueue\0")) }
}

pub(super) unsafe fn buffer_with_bytes(device: *mut c_void, bytes: &[u8]) -> *mut c_void {
    type New =
        unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void, usize, usize) -> *mut c_void;
    let new: New = msg!(New);
    unsafe {
        new(
            device,
            sel(b"newBufferWithBytes:length:options:\0"),
            bytes.as_ptr().cast(),
            bytes.len(),
            STORAGE_SHARED,
        )
    }
}

pub(super) unsafe fn buffer_with_length(device: *mut c_void, len: usize) -> *mut c_void {
    type New = unsafe extern "C" fn(*mut c_void, *mut c_void, usize, usize) -> *mut c_void;
    let new: New = msg!(New);
    unsafe {
        new(
            device,
            sel(b"newBufferWithLength:options:\0"),
            len,
            STORAGE_SHARED,
        )
    }
}

/// `commandBuffer` and `blitCommandEncoder` are autoreleased into the current pool.
pub(super) unsafe fn begin_blit(queue: *mut c_void) -> Option<(*mut c_void, *mut c_void)> {
    type Get = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    let get: Get = msg!(Get);
    let command = unsafe { get(queue, sel(b"commandBuffer\0")) };
    if command.is_null() {
        return None;
    }
    let encoder = unsafe { get(command, sel(b"blitCommandEncoder\0")) };
    if encoder.is_null() {
        return None;
    }
    Some((command, encoder))
}

pub(super) unsafe fn finish(encoder: *mut c_void, command: *mut c_void) -> usize {
    type Call = unsafe extern "C" fn(*mut c_void, *mut c_void);
    let call: Call = msg!(Call);
    unsafe {
        call(encoder, sel(b"endEncoding\0"));
        call(command, sel(b"commit\0"));
        call(command, sel(b"waitUntilCompleted\0"));
    }
    type Status = unsafe extern "C" fn(*mut c_void, *mut c_void) -> usize;
    let status: Status = msg!(Status);
    unsafe { status(command, sel(b"status\0")) }
}

pub(super) unsafe fn contents(buffer: *mut c_void) -> *mut u8 {
    type Get = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    let get: Get = msg!(Get);
    unsafe { get(buffer, sel(b"contents\0")).cast() }
}

pub(super) unsafe fn copy_buffer_to_buffer(
    encoder: *mut c_void,
    source: *mut c_void,
    destination: *mut c_void,
    len: usize,
) {
    type Copy = unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *mut c_void,
        usize,
        *mut c_void,
        usize,
        usize,
    );
    let copy: Copy = msg!(Copy);
    unsafe {
        copy(
            encoder,
            sel(b"copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:\0"),
            source,
            0,
            destination,
            0,
            len,
        );
    }
}

pub(super) unsafe fn copy_buffer_to_texture(
    encoder: *mut c_void,
    source: *mut c_void,
    bytes_per_row: usize,
    bytes_per_image: usize,
    size: Size,
    texture: *mut c_void,
) {
    type Copy = unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *mut c_void,
        usize,
        usize,
        usize,
        Size,
        *mut c_void,
        usize,
        usize,
        Origin,
    );
    let copy: Copy = msg!(Copy);
    unsafe {
        copy(
            encoder,
            sel(
                b"copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:\0",
            ),
            source,
            0,
            bytes_per_row,
            bytes_per_image,
            size,
            texture,
            0,
            0,
            Origin { x: 0, y: 0, z: 0 },
        );
    }
}

pub(super) unsafe fn copy_texture_to_buffer(
    encoder: *mut c_void,
    texture: *mut c_void,
    size: Size,
    destination: *mut c_void,
    bytes_per_row: usize,
    bytes_per_image: usize,
) {
    type Copy = unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *mut c_void,
        usize,
        usize,
        Origin,
        Size,
        *mut c_void,
        usize,
        usize,
        usize,
    );
    let copy: Copy = msg!(Copy);
    unsafe {
        copy(
            encoder,
            sel(
                b"copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:\0",
            ),
            texture,
            0,
            0,
            Origin { x: 0, y: 0, z: 0 },
            size,
            destination,
            0,
            bytes_per_row,
            bytes_per_image,
        );
    }
}

/// Shared-storage RGBA8 texture. The descriptor is autoreleased inside this call.
///
/// The setter is `setAllowGPUOptimizedContents:` because the property is
/// `allowGPUOptimizedContents`.
pub(super) unsafe fn texture_rgba(device: *mut c_void, width: usize, height: usize) -> *mut c_void {
    let class = unsafe { objc_getClass(c"MTLTextureDescriptor".as_ptr().cast()) };
    if class.is_null() {
        return ptr::null_mut();
    }
    let pool = Pool::push();
    type Make =
        unsafe extern "C" fn(*mut c_void, *mut c_void, usize, usize, usize, i8) -> *mut c_void;
    let make: Make = msg!(Make);
    let desc = unsafe {
        make(
            class,
            sel(b"texture2DDescriptorWithPixelFormat:width:height:mipmapped:\0"),
            RGBA8_UNORM,
            width,
            height,
            0,
        )
    };
    if desc.is_null() {
        drop(pool);
        return ptr::null_mut();
    }
    type SetUsize = unsafe extern "C" fn(*mut c_void, *mut c_void, usize);
    type SetBool = unsafe extern "C" fn(*mut c_void, *mut c_void, i8);
    let set_usize: SetUsize = msg!(SetUsize);
    let set_bool: SetBool = msg!(SetBool);
    unsafe {
        set_usize(desc, sel(b"setStorageMode:\0"), STORAGE_SHARED);
        set_bool(desc, sel(b"setAllowGPUOptimizedContents:\0"), 0);
    }
    type New = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
    let new: New = msg!(New);
    let texture = unsafe { new(device, sel(b"newTextureWithDescriptor:\0"), desc) };
    drop(pool);
    texture
}

/// Bytes per row for an RGBA8 blit.
///
/// `minimumLinearTextureAlignment` does not respond on current systems. The
/// header method is `minimumLinearTextureAlignmentForPixelFormat:`.
pub(super) unsafe fn linear_alignment(device: *mut c_void) -> usize {
    unsafe {
        if responds(device, b"minimumLinearTextureAlignment\0") {
            return alignment(device);
        }
        if responds(device, b"minimumLinearTextureAlignmentForPixelFormat:\0") {
            return alignment_for_format(device, RGBA8_UNORM);
        }
    }
    1
}

unsafe fn responds(obj: *mut c_void, name: &[u8]) -> bool {
    type Ask = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> i8;
    let ask: Ask = msg!(Ask);
    unsafe { ask(obj, sel(b"respondsToSelector:\0"), sel(name)) != 0 }
}

unsafe fn alignment(device: *mut c_void) -> usize {
    type Get = unsafe extern "C" fn(*mut c_void, *mut c_void) -> usize;
    let get: Get = msg!(Get);
    unsafe { get(device, sel(b"minimumLinearTextureAlignment\0")) }
}

unsafe fn alignment_for_format(device: *mut c_void, format: usize) -> usize {
    type Get = unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> usize;
    let get: Get = msg!(Get);
    unsafe {
        get(
            device,
            sel(b"minimumLinearTextureAlignmentForPixelFormat:\0"),
            format,
        )
    }
}
