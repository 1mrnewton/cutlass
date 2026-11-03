//! The few Media Foundation calls the transforms need.
//!
//! There is no bindings crate. Each interface is called through its vtable
//! slot, and the slot numbers follow the method order in the Windows SDK
//! headers: `IUnknown` is 0 to 2, `IMFAttributes` 3 to 32, `IMFSample` and
//! `IMFMediaType` continue from 33, `IMFMediaBuffer` and `IMFTransform` start
//! at 3. [`Com`] owns one reference and releases it on drop.

use std::ffi::c_void;
use std::mem::transmute;
use std::ptr;
use std::sync::OnceLock;

use cutlass_decode::CodecError;

use crate::format::Guid;

pub type Hresult = i32;

const COINIT_MULTITHREADED: u32 = 0;
const RPC_E_CHANGED_MODE: Hresult = 0x8001_0106_u32 as i32;
const CLSCTX_INPROC_SERVER: u32 = 1;
const MF_VERSION: u32 = 0x0002_0070;
const MFSTARTUP_FULL: u32 = 0;

pub const MF_E_NOTACCEPTING: Hresult = 0xC00D_36B5_u32 as i32;
pub const MF_E_NO_MORE_TYPES: Hresult = 0xC00D_36B9_u32 as i32;
pub const MF_E_TRANSFORM_STREAM_CHANGE: Hresult = 0xC00D_6D61_u32 as i32;
pub const MF_E_TRANSFORM_NEED_MORE_INPUT: Hresult = 0xC00D_6D72_u32 as i32;

pub const MFT_MESSAGE_COMMAND_FLUSH: u32 = 0;
pub const MFT_MESSAGE_COMMAND_DRAIN: u32 = 1;
pub const MFT_MESSAGE_NOTIFY_BEGIN_STREAMING: u32 = 0x1000_0000;
pub const MFT_MESSAGE_NOTIFY_END_OF_STREAM: u32 = 0x1000_0002;
pub const MFT_MESSAGE_NOTIFY_START_OF_STREAM: u32 = 0x1000_0003;

pub const MFT_OUTPUT_STREAM_PROVIDES_SAMPLES: u32 = 0x100;

const fn guid(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> Guid {
    Guid {
        data1,
        data2,
        data3,
        data4,
    }
}

/// A FOURCC subtype on the Media Foundation base GUID.
const fn fourcc(code: &[u8; 4]) -> Guid {
    guid(
        u32::from_le_bytes(*code),
        0x0000,
        0x0010,
        [0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71],
    )
}

pub const IID_IMF_TRANSFORM: Guid = guid(
    0xBF94_C121,
    0x5B05,
    0x4E6F,
    [0x80, 0x00, 0xBA, 0x59, 0x89, 0x61, 0x41, 0x4D],
);
pub const CLSID_H264_DECODER: Guid = guid(
    0x62CE_7E72,
    0x4C71,
    0x4D20,
    [0xB1, 0x5D, 0x45, 0x28, 0x31, 0xA8, 0x7D, 0x9D],
);
pub const CLSID_H264_ENCODER: Guid = guid(
    0x6CA5_0344,
    0x051A,
    0x4DED,
    [0x97, 0x79, 0xA4, 0x33, 0x05, 0x16, 0x5E, 0x35],
);

pub const MF_MT_MAJOR_TYPE: Guid = guid(
    0x48EB_A18E,
    0xF8C9,
    0x4687,
    [0xBF, 0x11, 0x0A, 0x74, 0xC9, 0xF9, 0x6A, 0x8F],
);
pub const MF_MT_SUBTYPE: Guid = guid(
    0xF7E3_4C9A,
    0x42E8,
    0x4714,
    [0xB7, 0x4B, 0xCB, 0x29, 0xD7, 0x2C, 0x35, 0xE5],
);
pub const MF_MT_FRAME_SIZE: Guid = guid(
    0x1652_C33D,
    0xD6B2,
    0x4012,
    [0xB8, 0x34, 0x72, 0x03, 0x08, 0x49, 0xA3, 0x7D],
);
pub const MF_MT_FRAME_RATE: Guid = guid(
    0xC459_A2E8,
    0x3D2C,
    0x4E44,
    [0xB1, 0x32, 0xFE, 0xE5, 0x15, 0x6C, 0x7B, 0xB0],
);
pub const MF_MT_PIXEL_ASPECT_RATIO: Guid = guid(
    0xC637_6A1E,
    0x8D0A,
    0x4027,
    [0xBE, 0x45, 0x6D, 0x9A, 0x0A, 0xD3, 0x9B, 0xB6],
);
pub const MF_MT_AVG_BITRATE: Guid = guid(
    0x2033_2624,
    0xFB0D,
    0x4D9E,
    [0xBD, 0x0D, 0xCB, 0xF6, 0x78, 0x6C, 0x10, 0x2E],
);
pub const MF_MT_INTERLACE_MODE: Guid = guid(
    0xE272_4BB8,
    0xE676,
    0x4806,
    [0xB4, 0xB2, 0xA8, 0xD6, 0xEF, 0xB4, 0x4C, 0xCD],
);
pub const MF_MT_MPEG2_PROFILE: Guid = guid(
    0xAD76_A80B,
    0x2D5C,
    0x4E0B,
    [0xB3, 0x75, 0x64, 0xE5, 0x20, 0x13, 0x70, 0x36],
);
pub const MF_MT_DEFAULT_STRIDE: Guid = guid(
    0x644B_4E48,
    0x1E02,
    0x4516,
    [0xB0, 0xEB, 0xC0, 0x1C, 0xA9, 0xD4, 0x9A, 0xC6],
);
pub const MF_MT_MINIMUM_DISPLAY_APERTURE: Guid = guid(
    0xD738_8766,
    0x18FE,
    0x48C6,
    [0xA1, 0x77, 0xEE, 0x89, 0x48, 0x67, 0xC8, 0xC4],
);

pub const MF_MEDIA_TYPE_VIDEO: Guid = fourcc(b"vids");
pub const MF_VIDEO_FORMAT_H264: Guid = fourcc(b"H264");
pub const MF_VIDEO_FORMAT_NV12: Guid = fourcc(b"NV12");

/// `MFVideoInterlace_Progressive`.
pub const PROGRESSIVE: u32 = 2;
/// `eAVEncH264VProfile_Base`.
pub const H264_BASELINE: u32 = 66;

#[link(name = "ole32")]
unsafe extern "system" {
    fn CoInitializeEx(reserved: *mut c_void, model: u32) -> Hresult;
    fn CoCreateInstance(
        clsid: *const Guid,
        outer: *mut c_void,
        context: u32,
        iid: *const Guid,
        out: *mut *mut c_void,
    ) -> Hresult;
}

#[link(name = "mfplat")]
unsafe extern "system" {
    fn MFStartup(version: u32, flags: u32) -> Hresult;
    fn MFCreateMediaType(out: *mut *mut c_void) -> Hresult;
    fn MFCreateSample(out: *mut *mut c_void) -> Hresult;
    fn MFCreateMemoryBuffer(max: u32, out: *mut *mut c_void) -> Hresult;
}

type Slot = unsafe extern "system" fn();

macro_rules! com {
    ($object:expr, $index:expr, fn($($arg:ty),*), $($value:expr),* $(,)?) => {{
        type Method = unsafe extern "system" fn(*mut c_void, $($arg),*) -> Hresult;
        let object: *mut c_void = $object;
        unsafe {
            let method: Method = transmute(vslot(object, $index));
            method(object, $($value),*)
        }
    }};
}

unsafe fn vslot(object: *mut c_void, index: usize) -> Slot {
    unsafe {
        let table = *object.cast::<*const Slot>();
        *table.add(index)
    }
}

/// `MFT_OUTPUT_STREAM_INFO`.
#[repr(C)]
#[derive(Default)]
pub struct OutputStreamInfo {
    pub flags: u32,
    pub size: u32,
    pub alignment: u32,
}

/// `MFT_OUTPUT_DATA_BUFFER`.
#[repr(C)]
struct OutputDataBuffer {
    stream: u32,
    sample: *mut c_void,
    status: u32,
    events: *mut c_void,
}

/// One owned COM reference.
pub struct Com(*mut c_void);

// SAFETY: the Microsoft codec transforms and the Media Foundation sample,
// buffer, and type objects are free-threaded. Each `Com` is reached through
// `&mut` of the codec that owns it, so one thread calls it at a time.
unsafe impl Send for Com {}

impl Com {
    fn from_raw(raw: *mut c_void) -> Option<Self> {
        if raw.is_null() { None } else { Some(Self(raw)) }
    }

    fn raw(&self) -> *mut c_void {
        self.0
    }
}

impl Drop for Com {
    fn drop(&mut self) {
        let _ = com!(self.0, 2, fn(),);
    }
}

/// Media Foundation is started once for the process, and COM once for each
/// thread that calls a codec. A thread that is already apartment threaded
/// keeps its apartment; the Microsoft transforms run in either.
pub fn enter() -> Result<(), CodecError> {
    thread_local! {
        static APARTMENT: Hresult = unsafe { CoInitializeEx(ptr::null_mut(), COINIT_MULTITHREADED) };
    }
    let apartment = APARTMENT.with(|code| *code);
    if apartment < 0 && apartment != RPC_E_CHANGED_MODE {
        return Err(CodecError::Unsupported { what: "com" });
    }
    static STARTED: OnceLock<Hresult> = OnceLock::new();
    let started = *STARTED.get_or_init(|| unsafe { MFStartup(MF_VERSION, MFSTARTUP_FULL) });
    if started < 0 {
        return Err(CodecError::Unsupported {
            what: "media foundation",
        });
    }
    Ok(())
}

/// Create one of the in-box codec transforms. A Windows N edition without
/// the media feature pack has none, so a failure here is a decline.
pub fn transform(clsid: &Guid) -> Result<Com, Hresult> {
    let mut out = ptr::null_mut();
    let code = unsafe {
        CoCreateInstance(
            clsid,
            ptr::null_mut(),
            CLSCTX_INPROC_SERVER,
            &IID_IMF_TRANSFORM,
            &mut out,
        )
    };
    owned(code, out)
}

fn owned(code: Hresult, raw: *mut c_void) -> Result<Com, Hresult> {
    if code < 0 {
        if let Some(object) = Com::from_raw(raw) {
            drop(object);
        }
        return Err(code);
    }
    Com::from_raw(raw).ok_or(code)
}

pub fn media_type() -> Result<Com, Hresult> {
    let mut out = ptr::null_mut();
    let code = unsafe { MFCreateMediaType(&mut out) };
    owned(code, out)
}

pub fn set_guid(attributes: &Com, key: &Guid, value: &Guid) -> Result<(), Hresult> {
    check(com!(
        attributes.raw(),
        24,
        fn(*const Guid, *const Guid),
        key,
        value
    ))
}

pub fn set_u32(attributes: &Com, key: &Guid, value: u32) -> Result<(), Hresult> {
    check(com!(attributes.raw(), 21, fn(*const Guid, u32), key, value))
}

pub fn set_u64(attributes: &Com, key: &Guid, value: u64) -> Result<(), Hresult> {
    check(com!(attributes.raw(), 22, fn(*const Guid, u64), key, value))
}

/// Two `u32` packed high then low, as `MFSetAttributeSize` writes them.
pub fn set_pair(attributes: &Com, key: &Guid, high: u32, low: u32) -> Result<(), Hresult> {
    set_u64(attributes, key, (u64::from(high) << 32) | u64::from(low))
}

pub fn get_guid(attributes: &Com, key: &Guid) -> Option<Guid> {
    let mut value = guid(0, 0, 0, [0; 8]);
    let code = com!(
        attributes.raw(),
        10,
        fn(*const Guid, *mut Guid),
        key,
        &mut value
    );
    (code >= 0).then_some(value)
}

pub fn get_u32(attributes: &Com, key: &Guid) -> Option<u32> {
    let mut value = 0_u32;
    let code = com!(
        attributes.raw(),
        7,
        fn(*const Guid, *mut u32),
        key,
        &mut value
    );
    (code >= 0).then_some(value)
}

pub fn get_pair(attributes: &Com, key: &Guid) -> Option<(u32, u32)> {
    let mut value = 0_u64;
    let code = com!(
        attributes.raw(),
        8,
        fn(*const Guid, *mut u64),
        key,
        &mut value
    );
    (code >= 0).then_some(((value >> 32) as u32, value as u32))
}

/// Read a fixed-size blob. A blob of another size is `None`.
pub fn get_blob<const N: usize>(attributes: &Com, key: &Guid) -> Option<[u8; N]> {
    let mut value = [0_u8; N];
    let mut written = 0_u32;
    let size = u32::try_from(N).ok()?;
    let code = com!(
        attributes.raw(),
        15,
        fn(*const Guid, *mut u8, u32, *mut u32),
        key,
        value.as_mut_ptr(),
        size,
        &mut written
    );
    (code >= 0 && written == size).then_some(value)
}

/// A sample holding one buffer with a copy of `bytes`.
pub fn sample_from(bytes: &[u8], time: i64, duration: i64) -> Result<Com, Hresult> {
    let sample = empty_sample(bytes.len())?;
    let buffer = contiguous(&sample)?;
    {
        let mut lock = Locked::new(&buffer)?;
        let out = lock.bytes_mut();
        if out.len() < bytes.len() {
            return Err(MF_E_NOTACCEPTING);
        }
        out[..bytes.len()].copy_from_slice(bytes);
    }
    let len = u32::try_from(bytes.len()).map_err(|_| MF_E_NOTACCEPTING)?;
    check(com!(buffer.raw(), 6, fn(u32), len))?;
    check(com!(sample.raw(), 36, fn(i64), time))?;
    check(com!(sample.raw(), 38, fn(i64), duration.max(0)))?;
    Ok(sample)
}

/// A sample with one empty buffer of `size` bytes.
pub fn empty_sample(size: usize) -> Result<Com, Hresult> {
    let size = u32::try_from(size.max(1)).map_err(|_| MF_E_NOTACCEPTING)?;
    let mut raw = ptr::null_mut();
    let sample = owned(unsafe { MFCreateSample(&mut raw) }, raw)?;
    let mut raw = ptr::null_mut();
    let buffer = owned(unsafe { MFCreateMemoryBuffer(size, &mut raw) }, raw)?;
    check(com!(sample.raw(), 42, fn(*mut c_void), buffer.raw()))?;
    Ok(sample)
}

/// Every byte in the sample, joined.
pub fn sample_bytes(sample: &Com) -> Result<Vec<u8>, Hresult> {
    let buffer = contiguous(sample)?;
    let lock = Locked::new(&buffer)?;
    Ok(lock.bytes().to_vec())
}

/// Borrow the sample's bytes without a copy.
pub fn with_sample_bytes<T>(sample: &Com, read: impl FnOnce(&[u8]) -> T) -> Result<T, Hresult> {
    let buffer = contiguous(sample)?;
    let lock = Locked::new(&buffer)?;
    Ok(read(lock.bytes()))
}

pub fn sample_time(sample: &Com) -> Option<i64> {
    let mut time = 0_i64;
    let code = com!(sample.raw(), 35, fn(*mut i64), &mut time);
    (code >= 0).then_some(time)
}

pub fn sample_duration(sample: &Com) -> Option<i64> {
    let mut duration = 0_i64;
    let code = com!(sample.raw(), 37, fn(*mut i64), &mut duration);
    (code >= 0).then_some(duration)
}

fn contiguous(sample: &Com) -> Result<Com, Hresult> {
    let mut raw = ptr::null_mut();
    let code = com!(sample.raw(), 41, fn(*mut *mut c_void), &mut raw);
    owned(code, raw)
}

/// A locked `IMFMediaBuffer`, unlocked on drop.
struct Locked<'a> {
    buffer: &'a Com,
    data: *mut u8,
    max: usize,
    current: usize,
}

impl<'a> Locked<'a> {
    fn new(buffer: &'a Com) -> Result<Self, Hresult> {
        let mut data = ptr::null_mut();
        let mut max = 0_u32;
        let mut current = 0_u32;
        let code = com!(
            buffer.raw(),
            3,
            fn(*mut *mut u8, *mut u32, *mut u32),
            &mut data,
            &mut max,
            &mut current
        );
        check(code)?;
        if data.is_null() {
            let _ = com!(buffer.raw(), 4, fn(),);
            return Err(MF_E_NOTACCEPTING);
        }
        Ok(Self {
            buffer,
            data,
            max: max as usize,
            current: (current as usize).min(max as usize),
        })
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: Lock returned `current` valid bytes at `data`, and they
        // stay valid until Unlock in drop.
        unsafe { std::slice::from_raw_parts(self.data, self.current) }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: Lock returned `max` writable bytes at `data`.
        unsafe { std::slice::from_raw_parts_mut(self.data, self.max) }
    }
}

impl Drop for Locked<'_> {
    fn drop(&mut self) {
        let _ = com!(self.buffer.raw(), 4, fn(),);
    }
}

pub fn set_input_type(transform: &Com, kind: &Com) -> Result<(), Hresult> {
    check(com!(
        transform.raw(),
        15,
        fn(u32, *mut c_void, u32),
        0,
        kind.raw(),
        0
    ))
}

pub fn set_output_type(transform: &Com, kind: &Com) -> Result<(), Hresult> {
    check(com!(
        transform.raw(),
        16,
        fn(u32, *mut c_void, u32),
        0,
        kind.raw(),
        0
    ))
}

/// The transform's `index`th output type, or `None` past the last.
pub fn output_available_type(transform: &Com, index: u32) -> Result<Option<Com>, Hresult> {
    let mut raw = ptr::null_mut();
    let code = com!(
        transform.raw(),
        14,
        fn(u32, u32, *mut *mut c_void),
        0,
        index,
        &mut raw
    );
    if code == MF_E_NO_MORE_TYPES {
        return Ok(None);
    }
    owned(code, raw).map(Some)
}

pub fn output_current_type(transform: &Com) -> Result<Com, Hresult> {
    let mut raw = ptr::null_mut();
    let code = com!(transform.raw(), 18, fn(u32, *mut *mut c_void), 0, &mut raw);
    owned(code, raw)
}

pub fn output_stream_info(transform: &Com) -> Result<OutputStreamInfo, Hresult> {
    let mut info = OutputStreamInfo::default();
    check(com!(
        transform.raw(),
        7,
        fn(u32, *mut OutputStreamInfo),
        0,
        &mut info
    ))?;
    Ok(info)
}

pub fn message(transform: &Com, message: u32) -> Result<(), Hresult> {
    check(com!(transform.raw(), 23, fn(u32, usize), message, 0))
}

/// Hand one sample in. `MF_E_NOTACCEPTING` comes back as is so the caller
/// can collect output and try again.
pub fn process_input(transform: &Com, sample: &Com) -> Hresult {
    com!(
        transform.raw(),
        24,
        fn(u32, *mut c_void, u32),
        0,
        sample.raw(),
        0
    )
}

/// Ask for one output sample. `provided` is the caller's sample, or `None`
/// when the transform allocates. The sample that holds the output comes back
/// with `Ok`; a status such as `MF_E_TRANSFORM_NEED_MORE_INPUT` is `Err`.
pub fn process_output(transform: &Com, provided: Option<Com>) -> Result<Com, Hresult> {
    let mut buffer = OutputDataBuffer {
        stream: 0,
        sample: provided.as_ref().map_or(ptr::null_mut(), Com::raw),
        status: 0,
        events: ptr::null_mut(),
    };
    let mut status = 0_u32;
    let code = com!(
        transform.raw(),
        25,
        fn(u32, u32, *mut OutputDataBuffer, *mut u32),
        0,
        1,
        &mut buffer,
        &mut status
    );
    if let Some(events) = Com::from_raw(buffer.events) {
        drop(events);
    }
    let returned = buffer.sample;
    let sample = match provided {
        Some(own) => {
            if returned != own.raw()
                && let Some(other) = Com::from_raw(returned)
            {
                drop(other);
            }
            own
        }
        None => match Com::from_raw(returned) {
            Some(sample) => sample,
            None => return Err(if code < 0 { code } else { MF_E_NOTACCEPTING }),
        },
    };
    check(code)?;
    Ok(sample)
}

fn check(code: Hresult) -> Result<(), Hresult> {
    if code < 0 { Err(code) } else { Ok(()) }
}
