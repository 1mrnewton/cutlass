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
pub const MF_E_NO_EVENTS_AVAILABLE: Hresult = 0xC00D_3E80_u32 as i32;
pub const MF_E_TRANSFORM_STREAM_CHANGE: Hresult = 0xC00D_6D61_u32 as i32;
pub const MF_E_TRANSFORM_NEED_MORE_INPUT: Hresult = 0xC00D_6D72_u32 as i32;

/// `MFT_ENUM_FLAG_SYNCMFT`, `_HARDWARE`, and `_SORTANDFILTER`.
pub const ENUM_SOFTWARE: u32 = 0x1 | 0x40;
pub const ENUM_HARDWARE: u32 = 0x4 | 0x40;

/// `METransformNeedInput`, `METransformHaveOutput`, `METransformDrainComplete`.
pub const EVENT_NEED_INPUT: u32 = 601;
pub const EVENT_HAVE_OUTPUT: u32 = 602;
pub const EVENT_DRAIN_COMPLETE: u32 = 603;
const MF_EVENT_FLAG_NO_WAIT: u32 = 1;

pub const MFT_MESSAGE_COMMAND_FLUSH: u32 = 0;
pub const MFT_MESSAGE_COMMAND_DRAIN: u32 = 1;
pub const MFT_MESSAGE_SET_D3D_MANAGER: u32 = 2;
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

pub const MF_MT_MPEG_SEQUENCE_HEADER: Guid = guid(
    0x3C03_6DE7,
    0x3AD0,
    0x4C9E,
    [0x92, 0x16, 0xEE, 0x6D, 0x6A, 0xC2, 0x1C, 0xB3],
);
pub const MF_MT_USER_DATA: Guid = guid(
    0xB6BC_765F,
    0x4C3B,
    0x40A4,
    [0xBD, 0x51, 0x25, 0x35, 0xB6, 0x6F, 0xE0, 0x9D],
);
pub const MF_MT_AUDIO_NUM_CHANNELS: Guid = guid(
    0x37E4_8BF5,
    0x645E,
    0x4C5B,
    [0x89, 0xDE, 0xAD, 0xA9, 0xE2, 0x9B, 0x69, 0x6A],
);
pub const MF_MT_AUDIO_SAMPLES_PER_SECOND: Guid = guid(
    0x5FAE_EAE7,
    0x0290,
    0x4C31,
    [0x9E, 0x8A, 0xC5, 0x34, 0xF6, 0x8D, 0x9D, 0xBA],
);
pub const MF_MT_AUDIO_BITS_PER_SAMPLE: Guid = guid(
    0xF2DE_B57F,
    0x40FA,
    0x4764,
    [0xAA, 0x33, 0xED, 0x4F, 0x2D, 0x1F, 0xF6, 0x69],
);
pub const MF_MT_AUDIO_AVG_BYTES_PER_SECOND: Guid = guid(
    0x1AAB_75C8,
    0xCFEF,
    0x451C,
    [0xAB, 0x95, 0xAC, 0x03, 0x4B, 0x8E, 0x17, 0x31],
);
pub const MF_MT_AUDIO_BLOCK_ALIGNMENT: Guid = guid(
    0x322D_E230,
    0x9EEB,
    0x43BD,
    [0xAB, 0x7A, 0xFF, 0x41, 0x22, 0x51, 0x54, 0x1D],
);
pub const MF_MT_AAC_PAYLOAD_TYPE: Guid = guid(
    0xBFBA_BE79,
    0x7434,
    0x4D1C,
    [0x94, 0xF0, 0x72, 0xA3, 0xB9, 0xE1, 0x71, 0x88],
);
pub const MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION: Guid = guid(
    0x7632_F0E6,
    0x9538,
    0x4D61,
    [0xAC, 0xDA, 0xEA, 0x29, 0xC8, 0xC1, 0x44, 0x56],
);

pub const MFT_CATEGORY_VIDEO_DECODER: Guid = guid(
    0xD6C0_2D4B,
    0x6833,
    0x45B4,
    [0x97, 0x1A, 0x05, 0xA4, 0xB0, 0x4B, 0xAB, 0x91],
);
pub const MFT_CATEGORY_VIDEO_ENCODER: Guid = guid(
    0xF79E_AC7D,
    0xE545,
    0x4387,
    [0xBD, 0xEE, 0xD6, 0x47, 0xD7, 0xBD, 0xE4, 0x2A],
);
const MF_TRANSFORM_ASYNC: Guid = guid(
    0xF81A_699A,
    0x649A,
    0x497D,
    [0x8C, 0x73, 0x29, 0xF8, 0xFE, 0xD6, 0xAD, 0x7A],
);
const MF_TRANSFORM_ASYNC_UNLOCK: Guid = guid(
    0xE566_6D6B,
    0x3422,
    0x4EB6,
    [0xA4, 0x21, 0xDA, 0x7D, 0xB1, 0xF8, 0xE2, 0x07],
);
const MF_SA_D3D11_AWARE: Guid = guid(
    0x206B_4FC8,
    0xFCF9,
    0x4C51,
    [0xAF, 0xE3, 0x97, 0x64, 0x36, 0x9E, 0x33, 0xA0],
);
const IID_IMF_MEDIA_EVENT_GENERATOR: Guid = guid(
    0x2CD0_BD52,
    0xBCD5,
    0x4B89,
    [0xB6, 0x2C, 0xEA, 0xDC, 0x0C, 0x03, 0x1E, 0x7D],
);
const IID_ID3D10_MULTITHREAD: Guid = guid(
    0x9B7E_4E00,
    0x342C,
    0x4106,
    [0xA1, 0x9F, 0x4F, 0x27, 0x04, 0xF6, 0x89, 0xF0],
);

pub const MF_MEDIA_TYPE_VIDEO: Guid = fourcc(b"vids");
pub const MF_MEDIA_TYPE_AUDIO: Guid = fourcc(b"auds");
pub const MF_VIDEO_FORMAT_H264: Guid = fourcc(b"H264");
pub const MF_VIDEO_FORMAT_HEVC: Guid = fourcc(b"HEVC");
pub const MF_VIDEO_FORMAT_NV12: Guid = fourcc(b"NV12");
/// Audio subtypes are the `WAVE_FORMAT` tag on the same base GUID.
pub const MF_AUDIO_FORMAT_AAC: Guid = wave(0x1610);
pub const MF_AUDIO_FORMAT_FLOAT: Guid = wave(0x0003);

/// `MFVideoInterlace_Progressive`.
pub const PROGRESSIVE: u32 = 2;
/// `eAVEncH264VProfile_Base`.
pub const H264_BASELINE: u32 = 66;
/// `eAVEncH265VProfile_Main_420_8`.
pub const HEVC_MAIN: u32 = 1;

const fn wave(tag: u32) -> Guid {
    guid(
        tag,
        0x0000,
        0x0010,
        [0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71],
    )
}

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
    fn CoTaskMemFree(memory: *mut c_void);
}

#[link(name = "mfplat")]
unsafe extern "system" {
    fn MFStartup(version: u32, flags: u32) -> Hresult;
    fn MFCreateMediaType(out: *mut *mut c_void) -> Hresult;
    fn MFCreateSample(out: *mut *mut c_void) -> Hresult;
    fn MFCreateMemoryBuffer(max: u32, out: *mut *mut c_void) -> Hresult;
    fn MFTEnumEx(
        category: Guid,
        flags: u32,
        input: *const TypeInfo,
        output: *const TypeInfo,
        activates: *mut *mut *mut c_void,
        count: *mut u32,
    ) -> Hresult;
    fn MFCreateDXGIDeviceManager(token: *mut u32, manager: *mut *mut c_void) -> Hresult;
}

#[link(name = "d3d11")]
unsafe extern "system" {
    fn D3D11CreateDevice(
        adapter: *mut c_void,
        driver: u32,
        software: *mut c_void,
        flags: u32,
        levels: *const u32,
        level_count: u32,
        sdk: u32,
        device: *mut *mut c_void,
        level: *mut u32,
        context: *mut *mut c_void,
    ) -> Hresult;
}

/// `MFT_REGISTER_TYPE_INFO`.
#[repr(C)]
struct TypeInfo {
    major: Guid,
    subtype: Guid,
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

pub(super) use com;

pub(super) unsafe fn vslot(object: *mut c_void, index: usize) -> Slot {
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
    pub(super) fn from_raw(raw: *mut c_void) -> Option<Self> {
        if raw.is_null() { None } else { Some(Self(raw)) }
    }

    pub(super) fn raw(&self) -> *mut c_void {
        self.0
    }

    /// Another interface on the same object.
    pub(super) fn query(&self, iid: &Guid) -> Result<Com, Hresult> {
        let mut out = ptr::null_mut();
        let code = com!(self.0, 0, fn(*const Guid, *mut *mut c_void), iid, &mut out);
        owned(code, out)
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

/// Read a blob of any size, such as a sequence header.
pub fn get_blob_vec(attributes: &Com, key: &Guid) -> Option<Vec<u8>> {
    let mut size = 0_u32;
    let code = com!(
        attributes.raw(),
        14,
        fn(*const Guid, *mut u32),
        key,
        &mut size
    );
    if code < 0 || size == 0 {
        return None;
    }
    let mut value = vec![0_u8; size as usize];
    let mut written = 0_u32;
    let code = com!(
        attributes.raw(),
        15,
        fn(*const Guid, *mut u8, u32, *mut u32),
        key,
        value.as_mut_ptr(),
        size,
        &mut written
    );
    if code < 0 {
        return None;
    }
    value.truncate(written as usize);
    Some(value)
}

/// Write a blob, such as an `AudioSpecificConfig` behind its wave header.
pub fn set_blob(attributes: &Com, key: &Guid, value: &[u8]) -> Result<(), Hresult> {
    let size = u32::try_from(value.len()).map_err(|_| MF_E_NOTACCEPTING)?;
    check(com!(
        attributes.raw(),
        26,
        fn(*const Guid, *const u8, u32),
        key,
        value.as_ptr(),
        size
    ))
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
    message_with(transform, message, 0)
}

pub fn message_with(transform: &Com, message: u32, param: usize) -> Result<(), Hresult> {
    check(com!(transform.raw(), 23, fn(u32, usize), message, param))
}

/// The registered transforms that turn `input` into `output`, best first.
/// Each is an `IMFActivate`; [`activate`] makes the transform.
pub fn enumerate(category: &Guid, flags: u32, input: &TypePair, output: &TypePair) -> Vec<Com> {
    let input = TypeInfo {
        major: input.0,
        subtype: input.1,
    };
    let output = TypeInfo {
        major: output.0,
        subtype: output.1,
    };
    let mut array = ptr::null_mut();
    let mut count = 0_u32;
    let code = unsafe { MFTEnumEx(*category, flags, &input, &output, &mut array, &mut count) };
    if code < 0 || array.is_null() {
        return Vec::new();
    }
    let mut found = Vec::with_capacity(count as usize);
    for index in 0..count as usize {
        // SAFETY: MFTEnumEx wrote `count` activate pointers at `array`, and
        // each carries a reference this Vec now owns.
        let raw = unsafe { *array.add(index) };
        if let Some(activate) = Com::from_raw(raw) {
            found.push(activate);
        }
    }
    unsafe { CoTaskMemFree(array.cast()) };
    found
}

/// A major type and a subtype, as `MFT_REGISTER_TYPE_INFO` holds them.
pub type TypePair = (Guid, Guid);

/// Make the transform an `IMFActivate` names.
pub fn activate(activate: &Com) -> Result<Com, Hresult> {
    let mut out = ptr::null_mut();
    let code = com!(
        activate.raw(),
        33,
        fn(*const Guid, *mut *mut c_void),
        &IID_IMF_TRANSFORM,
        &mut out
    );
    owned(code, out)
}

/// The transform's own attribute store.
fn transform_attributes(transform: &Com) -> Result<Com, Hresult> {
    let mut out = ptr::null_mut();
    let code = com!(transform.raw(), 8, fn(*mut *mut c_void), &mut out);
    owned(code, out)
}

/// Whether this is an asynchronous transform, unlocked for use. A
/// hardware encoder is; the in-box software codecs are not.
pub fn unlock_async(transform: &Com) -> Result<bool, Hresult> {
    let Ok(attributes) = transform_attributes(transform) else {
        return Ok(false);
    };
    if get_u32(&attributes, &MF_TRANSFORM_ASYNC) != Some(1) {
        return Ok(false);
    }
    set_u32(&attributes, &MF_TRANSFORM_ASYNC_UNLOCK, 1)?;
    Ok(true)
}

/// Ask a decoder to release each picture as soon as display order allows,
/// rather than once its whole picture buffer is full. A stream that does not
/// say how far it reorders is otherwise held sixteen pictures deep, which
/// with a Direct3D device runs the decoder out of surfaces.
pub fn set_low_latency(transform: &Com) -> Result<(), Hresult> {
    let attributes = transform_attributes(transform)?;
    set_u32(&attributes, &MF_LOW_LATENCY, 1)
}

const MF_LOW_LATENCY: Guid = guid(
    0x9C27_891A,
    0xED7A,
    0x40E1,
    [0x88, 0xE8, 0xB2, 0x27, 0x27, 0xA0, 0x24, 0xEE],
);

/// Whether the transform takes a Direct3D 11 device.
pub fn d3d11_aware(transform: &Com) -> bool {
    transform_attributes(transform)
        .ok()
        .and_then(|attributes| get_u32(&attributes, &MF_SA_D3D11_AWARE))
        .is_some_and(|aware| aware != 0)
}

pub fn event_generator(transform: &Com) -> Result<Com, Hresult> {
    transform.query(&IID_IMF_MEDIA_EVENT_GENERATOR)
}

/// The next event's type. `wait` false returns `None` when none is queued.
pub fn next_event(generator: &Com, wait: bool) -> Result<Option<u32>, Hresult> {
    let mut raw = ptr::null_mut();
    let flags = if wait { 0 } else { MF_EVENT_FLAG_NO_WAIT };
    let code = com!(
        generator.raw(),
        3,
        fn(u32, *mut *mut c_void),
        flags,
        &mut raw
    );
    if code == MF_E_NO_EVENTS_AVAILABLE {
        return Ok(None);
    }
    let event = owned(code, raw)?;
    let mut kind = 0_u32;
    check(com!(event.raw(), 33, fn(*mut u32), &mut kind))?;
    Ok(Some(kind))
}

pub const CODECAPI_GOP_SIZE: Guid = guid(
    0x95F3_1B26,
    0x95A4,
    0x41AA,
    [0x93, 0x03, 0x24, 0x6A, 0x7F, 0xC6, 0xEE, 0xF1],
);
pub const CODECAPI_B_PICTURE_COUNT: Guid = guid(
    0x8D39_0AAC,
    0xDC5C,
    0x4200,
    [0xB5, 0x7F, 0x81, 0x4D, 0x04, 0xBA, 0xBA, 0xB2],
);
pub const CODECAPI_MEAN_BIT_RATE: Guid = guid(
    0xF722_2374,
    0x2144,
    0x4815,
    [0xB5, 0x50, 0xA3, 0x7F, 0x8E, 0x12, 0xEE, 0x52],
);
const IID_ICODEC_API: Guid = guid(
    0x901D_B4C7,
    0x31CE,
    0x41A2,
    [0x85, 0xDC, 0x8F, 0xA0, 0xBF, 0x41, 0xB8, 0xDA],
);

/// `VARIANT` as it is laid out on 64-bit Windows.
#[repr(C)]
struct Variant {
    kind: u16,
    reserved: [u16; 3],
    value: u64,
    record: u64,
}

/// Set one `ICodecAPI` property to a `VT_UI4`. A transform without the
/// interface or the property refuses, and the caller keeps its default.
pub fn set_codec_u32(transform: &Com, key: &Guid, value: u32) -> Result<(), Hresult> {
    const VT_UI4: u16 = 19;
    let api = transform.query(&IID_ICODEC_API)?;
    let variant = Variant {
        kind: VT_UI4,
        reserved: [0; 3],
        value: u64::from(value),
        record: 0,
    };
    check(com!(
        api.raw(),
        9,
        fn(*const Guid, *const Variant),
        key,
        &variant
    ))
}

/// A Direct3D 11 device with video support behind a DXGI device manager,
/// for a decoder to write its pictures on the GPU.
pub struct DeviceManager {
    manager: Com,
    _device: Com,
}

impl DeviceManager {
    pub fn create() -> Result<Self, Hresult> {
        const HARDWARE: u32 = 1;
        const VIDEO_SUPPORT: u32 = 0x800;
        const SDK_VERSION: u32 = 7;
        let mut device = ptr::null_mut();
        let code = unsafe {
            D3D11CreateDevice(
                ptr::null_mut(),
                HARDWARE,
                ptr::null_mut(),
                VIDEO_SUPPORT,
                ptr::null(),
                0,
                SDK_VERSION,
                &mut device,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        let device = owned(code, device)?;
        // A decoder calls the device from its own threads.
        let multithread = device.query(&IID_ID3D10_MULTITHREAD)?;
        let _ = com!(multithread.raw(), 5, fn(i32), 1);
        let mut token = 0_u32;
        let mut manager = ptr::null_mut();
        let code = unsafe { MFCreateDXGIDeviceManager(&mut token, &mut manager) };
        let manager = owned(code, manager)?;
        check(com!(
            manager.raw(),
            7,
            fn(*mut c_void, u32),
            device.raw(),
            token
        ))?;
        Ok(Self {
            manager,
            _device: device,
        })
    }

    /// Hand the device to a transform with `MFT_MESSAGE_SET_D3D_MANAGER`.
    pub fn attach(&self, transform: &Com) -> Result<(), Hresult> {
        message_with(
            transform,
            MFT_MESSAGE_SET_D3D_MANAGER,
            self.manager.raw() as usize,
        )
    }
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
