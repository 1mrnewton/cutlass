//! Load NVDEC and NVENC. `cuCtxCreate` is exported as `cuCtxCreate_v2`.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr;

use cutlass_core::VideoCodec;

use crate::caps::{self, Guid};

const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x00100;
const CUDA: u32 = 1;
const CHROMA_420: i32 = 1;
const CHROMA_444: i32 = 3;
const ASK: [i32; 10] = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11];
/// Pointers in `NV_ENCODE_API_FUNCTION_LIST` before the reserved tail.
const FN_PREFIX: usize = 43;
const FN_RESERVED: usize = 275;
const FN_COUNT: usize = FN_PREFIX + FN_RESERVED;
const GUID_COUNT: usize = 1;
const GUID_LIST: usize = 4;
const DESTROY: usize = 27;
const OPEN_EX: usize = 29;

pub(crate) fn query() -> caps::Capabilities {
    caps::Capabilities::from_lists(decode_codecs(), encode_codecs())
}

fn decode_codecs() -> Vec<VideoCodec> {
    let Some(cuda) = Library::open(c"libcuda.so.1", RTLD_NOW | RTLD_GLOBAL) else {
        return Vec::new();
    };
    let Some(init) = load::<CuInit>(&cuda, c"cuInit") else {
        return Vec::new();
    };
    // SAFETY: `cuInit` takes flags. Zero selects the default initialization.
    if unsafe { init(0) } != 0 {
        return Vec::new();
    }
    let Some(nvdec) = Library::open(c"libnvcuvid.so.1", RTLD_NOW) else {
        return Vec::new();
    };
    let Some(get_caps) = load::<GetCaps>(&nvdec, c"cuvidGetDecoderCaps") else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for id in ASK {
        let Some(codec) = caps::video_codec(id) else {
            continue;
        };
        if found.contains(&codec) {
            continue;
        }
        if decoder_supports(get_caps, id) {
            found.push(codec);
        }
    }
    found
}

fn decoder_supports(get_caps: GetCaps, codec: i32) -> bool {
    for chroma in CHROMA_420..=CHROMA_444 {
        for depth in [0, 2] {
            let mut info = DecodeCaps {
                codec,
                chroma,
                bit_depth: depth,
                reserved1: [0; 3],
                supported: 0,
                decoders: 0,
                output_mask: 0,
                max_width: 0,
                max_height: 0,
                max_mb: 0,
                min_width: 0,
                min_height: 0,
                histogram: 0,
                counter_depth: 0,
                max_bins: 0,
                reserved3: [0; 10],
            };
            // SAFETY: `info` is the `CUVIDDECODECAPS` layout, zeroed apart
            // from the three input fields. A non-zero status means unsupported.
            if unsafe { get_caps(&mut info) } == 0 && info.supported != 0 {
                return true;
            }
        }
    }
    false
}

fn encode_codecs() -> Vec<VideoCodec> {
    let Some(cuda) = Library::open(c"libcuda.so.1", RTLD_NOW | RTLD_GLOBAL) else {
        return Vec::new();
    };
    let Some(context) = CudaContext::open(&cuda) else {
        return Vec::new();
    };
    let Some(nvenc) = Library::open(c"libnvidia-encode.so.1", RTLD_NOW) else {
        return Vec::new();
    };
    let Some(max_version) = load::<MaxVersion>(&nvenc, c"NvEncodeAPIGetMaxSupportedVersion") else {
        return Vec::new();
    };
    let mut packed = 0;
    // SAFETY: `packed` is a local `uint32_t`.
    if unsafe { max_version(&mut packed) } != 0 {
        return Vec::new();
    }
    let Some((major, minor)) = caps::driver_version(packed) else {
        return Vec::new();
    };
    let api = caps::api_version(major, minor);
    let Some(create) = load::<CreateInstance>(&nvenc, c"NvEncodeAPICreateInstance") else {
        return Vec::new();
    };
    let mut list = FunctionList {
        version: caps::struct_version(api, 2),
        reserved: 0,
        ptrs: [ptr::null_mut(); FN_COUNT],
    };
    // SAFETY: `list` is the encode function list. The version tells the driver
    // which prefix it may fill. The tail stays null.
    if unsafe { create(&mut list) } != 0 {
        return Vec::new();
    }
    let Some(open_ex) = fn_at::<OpenEx>(&list, OPEN_EX) else {
        return Vec::new();
    };
    let Some(get_count) = fn_at::<GetCount>(&list, GUID_COUNT) else {
        return Vec::new();
    };
    let Some(get_guids) = fn_at::<GetGuids>(&list, GUID_LIST) else {
        return Vec::new();
    };
    let Some(destroy) = fn_at::<Destroy>(&list, DESTROY) else {
        return Vec::new();
    };
    let mut session = OpenSession {
        version: caps::struct_version(api, 1),
        device_type: CUDA,
        device: context.raw,
        reserved: ptr::null_mut(),
        api_version: api,
        reserved1: [0; 253],
        reserved2: [ptr::null_mut(); 64],
    };
    let mut encoder = ptr::null_mut();
    // SAFETY: `session` is the open-session struct and `device` is a live CUDA
    // context. A null encoder means the driver refused the session.
    if unsafe { open_ex(&mut session, &mut encoder) } != 0 || encoder.is_null() {
        return Vec::new();
    }
    let guard = EncoderGuard { encoder, destroy };
    read_guids(guard.encoder, get_count, get_guids)
}

fn read_guids(encoder: *mut c_void, get_count: GetCount, get_guids: GetGuids) -> Vec<VideoCodec> {
    let mut count = 0;
    // SAFETY: `encoder` is an open session and `count` is a local `uint32_t`.
    if unsafe { get_count(encoder, &mut count) } != 0 || count == 0 {
        return Vec::new();
    }
    let slots = count.min(8);
    let mut guids = [Guid {
        data1: 0,
        data2: 0,
        data3: 0,
        data4: [0; 8],
    }; 8];
    let mut written = 0;
    // SAFETY: `guids` holds `slots` entries, which is the capacity passed in.
    if unsafe { get_guids(encoder, guids.as_mut_ptr(), slots, &mut written) } != 0 {
        return Vec::new();
    }
    let written = usize::try_from(written).unwrap_or(0).min(guids.len());
    let mut found = Vec::new();
    for guid in guids.into_iter().take(written) {
        if let Some(codec) = caps::guid_codec(&guid) {
            if !found.contains(&codec) {
                found.push(codec);
            }
        }
    }
    found
}

#[repr(C)]
struct DecodeCaps {
    codec: i32,
    chroma: i32,
    bit_depth: u32,
    reserved1: [u32; 3],
    supported: u8,
    decoders: u8,
    output_mask: u16,
    max_width: u32,
    max_height: u32,
    max_mb: u32,
    min_width: u16,
    min_height: u16,
    histogram: u8,
    counter_depth: u8,
    max_bins: u16,
    reserved3: [u32; 10],
}

#[repr(C)]
struct FunctionList {
    version: u32,
    reserved: u32,
    ptrs: [*mut c_void; FN_COUNT],
}

#[repr(C)]
struct OpenSession {
    version: u32,
    device_type: u32,
    device: *mut c_void,
    reserved: *mut c_void,
    api_version: u32,
    reserved1: [u32; 253],
    reserved2: [*mut c_void; 64],
}

struct CudaContext {
    raw: *mut c_void,
    destroy: CtxDestroy,
}

impl CudaContext {
    fn open(cuda: &Library) -> Option<Self> {
        let init = load::<CuInit>(cuda, c"cuInit")?;
        let device_get = load::<DeviceGet>(cuda, c"cuDeviceGet")?;
        let create = load::<CtxCreate>(cuda, c"cuCtxCreate_v2")?;
        let destroy = load::<CtxDestroy>(cuda, c"cuCtxDestroy_v2")?;
        // SAFETY: flags are zero, the default initialization.
        if unsafe { init(0) } != 0 {
            return None;
        }
        let mut device = 0;
        // SAFETY: `device` is a local `CUdevice`. Ordinal zero is the first GPU.
        if unsafe { device_get(&mut device, 0) } != 0 {
            return None;
        }
        let mut raw = ptr::null_mut();
        // SAFETY: `raw` receives the context. Flags zero request the default
        // scheduling. The context is destroyed by the guard.
        if unsafe { create(&mut raw, 0, device) } != 0 || raw.is_null() {
            return None;
        }
        Some(Self { raw, destroy })
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        // SAFETY: `raw` is a context from `cuCtxCreate_v2` and this guard is
        // the only owner.
        unsafe { (self.destroy)(self.raw) };
    }
}

struct EncoderGuard {
    encoder: *mut c_void,
    destroy: Destroy,
}

impl Drop for EncoderGuard {
    fn drop(&mut self) {
        if self.encoder.is_null() {
            return;
        }
        // SAFETY: `encoder` is an open session and this guard is the only owner.
        unsafe { (self.destroy)(self.encoder) };
    }
}

struct Library {
    handle: *mut c_void,
}

impl Library {
    fn open(name: &CStr, flags: c_int) -> Option<Self> {
        // SAFETY: `name` is a nul-terminated literal. A null handle means the
        // library is not installed.
        let handle = unsafe { dlopen(name.as_ptr(), flags) };
        if handle.is_null() {
            None
        } else {
            Some(Self { handle })
        }
    }

    fn symbol(&self, name: &CStr) -> Option<*mut c_void> {
        // SAFETY: `handle` came from `dlopen` and is still open.
        let symbol = unsafe { dlsym(self.handle, name.as_ptr()) };
        if symbol.is_null() { None } else { Some(symbol) }
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        // SAFETY: `handle` came from `dlopen` and has not been closed.
        unsafe { dlclose(self.handle) };
    }
}

fn load<T>(library: &Library, name: &CStr) -> Option<T> {
    const {
        assert!(
            std::mem::size_of::<T>() == std::mem::size_of::<*mut c_void>(),
            "a loaded symbol is one pointer"
        );
    }
    let symbol = library.symbol(name)?;
    // SAFETY: `T` is the C function type of `name`, and the assert makes it
    // the same size as the symbol pointer.
    Some(unsafe { std::mem::transmute_copy(&symbol) })
}

fn fn_at<T>(list: &FunctionList, index: usize) -> Option<T> {
    const {
        assert!(
            std::mem::size_of::<T>() == std::mem::size_of::<*mut c_void>(),
            "a loaded symbol is one pointer"
        );
    }
    let symbol = *list.ptrs.get(index)?;
    if symbol.is_null() {
        None
    } else {
        // SAFETY: `T` is the function type at this index in the encode list.
        Some(unsafe { std::mem::transmute_copy(&symbol) })
    }
}

type CuInit = unsafe extern "C" fn(u32) -> i32;
type DeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
type CtxCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
type CtxDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
type GetCaps = unsafe extern "C" fn(*mut DecodeCaps) -> i32;
type MaxVersion = unsafe extern "C" fn(*mut u32) -> i32;
type CreateInstance = unsafe extern "C" fn(*mut FunctionList) -> i32;
type OpenEx = unsafe extern "C" fn(*mut OpenSession, *mut *mut c_void) -> i32;
type GetCount = unsafe extern "C" fn(*mut c_void, *mut u32) -> i32;
type GetGuids = unsafe extern "C" fn(*mut c_void, *mut Guid, u32, *mut u32) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void) -> i32;

// SAFETY: these are the POSIX dynamic-loader functions.
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}
