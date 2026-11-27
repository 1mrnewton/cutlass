//! The CUDA driver: one context on the first NVIDIA GPU, shared by every
//! NVDEC and NVENC session in the process.
//!
//! `libcuda.so.1` ships with the NVIDIA driver. It is opened once; a
//! machine without it, or without a device, has no [`device`], and the
//! NVIDIA openers decline.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::ptr;
use std::sync::OnceLock;

use cutlass_decode::CodecError;

use crate::dl::Library;

pub type CuResult = c_int;
pub type CuDevicePtr = u64;
pub type CuContext = *mut c_void;

pub const CUDA_SUCCESS: CuResult = 0;

/// `CU_MEMORYTYPE_HOST`.
const MEMORY_HOST: c_uint = 1;
/// `CU_MEMORYTYPE_DEVICE`.
const MEMORY_DEVICE: c_uint = 2;

/// `CUDA_MEMCPY2D`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Memcpy2D {
    pub src_x_in_bytes: usize,
    pub src_y: usize,
    pub src_memory_type: c_uint,
    pub src_host: *const c_void,
    pub src_device: CuDevicePtr,
    pub src_array: *mut c_void,
    pub src_pitch: usize,
    pub dst_x_in_bytes: usize,
    pub dst_y: usize,
    pub dst_memory_type: c_uint,
    pub dst_host: *mut c_void,
    pub dst_device: CuDevicePtr,
    pub dst_array: *mut c_void,
    pub dst_pitch: usize,
    pub width_in_bytes: usize,
    pub height: usize,
}

type Init = unsafe extern "C" fn(c_uint) -> CuResult;
type DeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CuResult;
type DeviceGet = unsafe extern "C" fn(*mut c_int, c_int) -> CuResult;
type DeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> CuResult;
type CtxCreate = unsafe extern "C" fn(*mut CuContext, c_uint, c_int) -> CuResult;
type CtxPush = unsafe extern "C" fn(CuContext) -> CuResult;
type CtxPop = unsafe extern "C" fn(*mut CuContext) -> CuResult;
type Memcpy2DFn = unsafe extern "C" fn(*const Memcpy2D) -> CuResult;
type DriverGetVersion = unsafe extern "C" fn(*mut c_int) -> CuResult;

/// The driver calls this crate makes.
struct Driver {
    init: Init,
    device_get_count: DeviceGetCount,
    device_get: DeviceGet,
    device_get_name: DeviceGetName,
    ctx_create: CtxCreate,
    ctx_push: CtxPush,
    ctx_pop: CtxPop,
    memcpy_2d: Memcpy2DFn,
    driver_get_version: DriverGetVersion,
    _library: Library,
}

impl Driver {
    fn load() -> Option<Self> {
        let library = Library::open(&[c"libcuda.so.1", c"libcuda.so"])?;
        // SAFETY: each symbol has this signature in cuda.h; the `_v2`
        // names are the ones the 64-bit API uses. The pointers live in the
        // same struct as the library.
        unsafe {
            Some(Self {
                init: library.symbol(c"cuInit")?,
                device_get_count: library.symbol(c"cuDeviceGetCount")?,
                device_get: library.symbol(c"cuDeviceGet")?,
                device_get_name: library.symbol(c"cuDeviceGetName")?,
                ctx_create: library.symbol(c"cuCtxCreate_v2")?,
                ctx_push: library.symbol(c"cuCtxPushCurrent_v2")?,
                ctx_pop: library.symbol(c"cuCtxPopCurrent_v2")?,
                memcpy_2d: library.symbol(c"cuMemcpy2D_v2")?,
                driver_get_version: library.symbol(c"cuDriverGetVersion")?,
                _library: library,
            })
        }
    }
}

/// The first NVIDIA GPU and a context on it, for the life of the process.
pub struct Device {
    driver: Driver,
    context: CuContext,
    name: String,
    driver_version: i32,
}

// SAFETY: a CUDA context may be made current on any thread, and on several
// at once; every call here pushes it first and pops it after.
unsafe impl Send for Device {}
// SAFETY: as above. The fields are not written after open.
unsafe impl Sync for Device {}

/// The process-wide device, or `None` when there is no NVIDIA driver or GPU.
pub fn device() -> Option<&'static Device> {
    static DEVICE: OnceLock<Option<Device>> = OnceLock::new();
    DEVICE.get_or_init(Device::open).as_ref()
}

impl Device {
    fn open() -> Option<Self> {
        let driver = Driver::load()?;
        // SAFETY: cuInit takes flags, which must be 0.
        if unsafe { (driver.init)(0) } != CUDA_SUCCESS {
            return None;
        }
        let mut count = 0;
        // SAFETY: the out pointer is valid.
        if unsafe { (driver.device_get_count)(&raw mut count) } != CUDA_SUCCESS || count < 1 {
            return None;
        }
        let mut device = 0;
        // SAFETY: the out pointer is valid and ordinal 0 exists.
        if unsafe { (driver.device_get)(&raw mut device, 0) } != CUDA_SUCCESS {
            return None;
        }
        let mut name: [c_char; 128] = [0; 128];
        // SAFETY: the buffer holds the length given.
        let named = unsafe { (driver.device_get_name)(name.as_mut_ptr(), 128, device) };
        let name = if named == CUDA_SUCCESS {
            // SAFETY: the driver wrote a NUL-terminated string into the
            // buffer, which ends in NUL either way since it was zeroed and
            // the driver writes at most 127 bytes and a NUL.
            unsafe { CStr::from_ptr(name.as_ptr()) }
                .to_string_lossy()
                .into_owned()
        } else {
            String::from("NVIDIA GPU")
        };
        let mut driver_version = 0;
        // SAFETY: the out pointer is valid.
        unsafe { (driver.driver_get_version)(&raw mut driver_version) };
        let mut context = ptr::null_mut();
        // SAFETY: the out pointer is valid; flags 0 is the default
        // scheduling. The new context is current on this thread.
        if unsafe { (driver.ctx_create)(&raw mut context, 0, device) } != CUDA_SUCCESS
            || context.is_null()
        {
            return None;
        }
        let mut popped = ptr::null_mut();
        // SAFETY: the context just created is current here; it leaves this
        // thread and is pushed again by each call that needs it.
        unsafe { (driver.ctx_pop)(&raw mut popped) };
        Some(Self {
            driver,
            context,
            name,
            driver_version,
        })
    }

    pub fn context(&self) -> CuContext {
        self.context
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// The CUDA version the driver supports, as 1000 × major + 10 × minor.
    pub fn driver_version(&self) -> i32 {
        self.driver_version
    }

    /// Run `f` with the context current on this thread.
    pub fn with<T>(&self, f: impl FnOnce() -> T) -> Result<T, CodecError> {
        // SAFETY: the context lives for the process.
        if unsafe { (self.driver.ctx_push)(self.context) } != CUDA_SUCCESS {
            return Err(CodecError::Unsupported {
                what: "cuda context",
            });
        }
        let _pop = Pop(self);
        Ok(f())
    }

    /// Copy `rows` rows of `width` bytes from device memory at `source`,
    /// `pitch` bytes apart, to the end of `out`, tightly packed. The
    /// context must be current.
    pub fn copy_rows(
        &self,
        source: CuDevicePtr,
        pitch: usize,
        width: usize,
        rows: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), CodecError> {
        let start = out.len();
        let len = width.checked_mul(rows).ok_or(CodecError::Limit {
            what: "picture bytes",
        })?;
        out.resize(start + len, 0);
        let copy = Memcpy2D {
            src_x_in_bytes: 0,
            src_y: 0,
            src_memory_type: MEMORY_DEVICE,
            src_host: ptr::null(),
            src_device: source,
            src_array: ptr::null_mut(),
            src_pitch: pitch,
            dst_x_in_bytes: 0,
            dst_y: 0,
            dst_memory_type: MEMORY_HOST,
            dst_host: out[start..].as_mut_ptr().cast(),
            dst_device: 0,
            dst_array: ptr::null_mut(),
            dst_pitch: width,
            width_in_bytes: width,
            height: rows,
        };
        // SAFETY: the destination holds `width * rows` bytes, the source is
        // a mapped frame of at least `pitch * rows` bytes, and the context
        // is current, as the caller promises.
        let copied = unsafe { (self.driver.memcpy_2d)(&raw const copy) };
        if copied != CUDA_SUCCESS {
            out.truncate(start);
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "cuda copy",
            });
        }
        Ok(())
    }
}

/// Pops the context on drop.
struct Pop<'a>(&'a Device);

impl Drop for Pop<'_> {
    fn drop(&mut self) {
        let mut popped = ptr::null_mut();
        // SAFETY: the context was pushed on this thread by `with`.
        unsafe { (self.0.driver.ctx_pop)(&raw mut popped) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_copy_descriptor_has_the_c_layout() {
        // From cuda.h (nv-codec-headers n12.0.16.1) on LP64.
        assert_eq!(size_of::<Memcpy2D>(), 128);
        assert_eq!(std::mem::offset_of!(Memcpy2D, src_device), 32);
        assert_eq!(std::mem::offset_of!(Memcpy2D, dst_memory_type), 72);
        assert_eq!(std::mem::offset_of!(Memcpy2D, width_in_bytes), 112);
    }
}
