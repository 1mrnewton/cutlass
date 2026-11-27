//! NVENC: H.264 and HEVC encode on an NVIDIA GPU.
//!
//! `libnvidia-encode.so.1` ships with the NVIDIA driver. The session runs
//! on the process-wide CUDA context and takes host NV12 buffers that NVENC
//! allocates, so pictures are copied once, from the frame into the locked
//! buffer.
//!
//! NVENC's structs are large and mostly reserved. Each one is kept as
//! zeroed bytes, a [`Blob`], and the fields this encoder sets are written at
//! the offsets in [`super::nvenc_abi`], which a C program printed from
//! nvEncodeAPI.h.
//!
//! The settings match Windows and Apple: an IDR every 30 pictures, no B
//! pictures, H.264 High and HEVC Main, and VBR at twice the target as its
//! peak unless the caller asks for a constant rate.

use std::collections::VecDeque;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use cutlass_color::nv12;
use cutlass_decode::nal;
use cutlass_decode::{CodecError, Packet};
use cutlass_decode::{CodecId, CodecParams, PixelFormat};
use cutlass_encode::{AudioInput, Encoder, FrameInput, FrameTime, PictureInput};

use super::blob::Blob;
use super::cuda::{self, Device};
use super::nvenc_abi as abi;
use crate::dl::Library;
use crate::video::backend::{note_decline, note_open};
use crate::video::kind::Kind;
use crate::video::limits::video_size;
use crate::video::packets::Packetizer;
use crate::video::rate::{self, Vui};

const BACKEND: &str = "NVENC";

/// Input and bitstream buffer pairs. Without B pictures or lookahead each
/// picture comes out of the call that took it, so one pair is in use at a
/// time; the rest cover a driver that holds a picture back anyway.
const BUFFERS: usize = 4;

/// Room for the parameter sets `nvEncGetSequenceParams` writes.
const SEQUENCE_BYTES: usize = 1024;

type Status = c_int;
type Handle = *mut c_void;

/// `GUID` as nvEncodeAPI.h declares it, so it can pass by value.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct Guid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

impl Guid {
    const fn from_bytes(b: [u8; 16]) -> Self {
        Self {
            data1: u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            data2: u16::from_le_bytes([b[4], b[5]]),
            data3: u16::from_le_bytes([b[6], b[7]]),
            data4: [b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]],
        }
    }

    const ZERO: Self = Self::from_bytes([0; 16]);
}

// The generated offsets fit their structs, and every pointer field is
// 8-byte aligned, as `Blob` writes them.
const _: () = {
    assert!(abi::functions::SIZE == 8 * 319);
    assert!(abi::functions::GET_LAST_ERROR_STRING < abi::functions::SIZE);
    assert!(abi::preset::CONFIG.is_multiple_of(8));
    assert!(abi::preset::CONFIG + abi::config::SIZE <= abi::preset::SIZE);
    assert!(abi::config::H264_VUI + abi::vui::SIZE < abi::config::SIZE);
    assert!(abi::config::HEVC_VUI + abi::vui::SIZE < abi::config::SIZE);
    assert!(abi::open_session::DEVICE.is_multiple_of(8));
    assert!(abi::init::ENCODE_CONFIG.is_multiple_of(8));
    assert!(
        abi::pic::INPUT_BUFFER.is_multiple_of(8) && abi::pic::OUTPUT_BITSTREAM.is_multiple_of(8)
    );
    assert!(abi::create_input::INPUT_BUFFER.is_multiple_of(8));
    assert!(abi::create_bitstream::BITSTREAM_BUFFER.is_multiple_of(8));
    assert!(abi::lock_bitstream::BITSTREAM_BUFFER_PTR.is_multiple_of(8));
    assert!(abi::lock_bitstream::OUTPUT_BITSTREAM.is_multiple_of(8));
    assert!(abi::lock_input::BUFFER_DATA_PTR.is_multiple_of(8));
    assert!(abi::lock_input::INPUT_BUFFER.is_multiple_of(8));
    assert!(abi::sequence::SPSPPS_BUFFER.is_multiple_of(8));
    assert!(abi::sequence::OUT_SPSPPS_PAYLOAD_SIZE.is_multiple_of(8));
};

type GetMaxSupportedVersion = unsafe extern "C" fn(*mut u32) -> Status;
type CreateInstance = unsafe extern "C" fn(*mut c_void) -> Status;
type OpenSessionEx = unsafe extern "C" fn(*mut c_void, *mut Handle) -> Status;
type GetGuidCount = unsafe extern "C" fn(Handle, *mut u32) -> Status;
type GetGuids = unsafe extern "C" fn(Handle, *mut Guid, u32, *mut u32) -> Status;
type GetCaps = unsafe extern "C" fn(Handle, Guid, *mut c_void, *mut c_int) -> Status;
type GetPresetConfigEx = unsafe extern "C" fn(Handle, Guid, Guid, u32, *mut c_void) -> Status;
type WithParams = unsafe extern "C" fn(Handle, *mut c_void) -> Status;
type WithHandle = unsafe extern "C" fn(Handle, Handle) -> Status;
type DestroyEncoder = unsafe extern "C" fn(Handle) -> Status;
type GetLastErrorString = unsafe extern "C" fn(Handle) -> *const c_char;

/// The entry points from `NvEncodeAPICreateInstance`.
pub struct Api {
    open_session_ex: OpenSessionEx,
    get_guid_count: GetGuidCount,
    get_guids: GetGuids,
    get_caps: GetCaps,
    get_preset_config_ex: GetPresetConfigEx,
    initialize: WithParams,
    create_input: WithParams,
    destroy_input: WithHandle,
    create_bitstream: WithParams,
    destroy_bitstream: WithHandle,
    encode_picture: WithParams,
    lock_bitstream: WithParams,
    unlock_bitstream: WithHandle,
    lock_input: WithParams,
    unlock_input: WithHandle,
    get_sequence_params: WithParams,
    destroy_encoder: DestroyEncoder,
    last_error: GetLastErrorString,
    /// The highest API the driver takes, as major << 4 | minor.
    driver_api: u32,
    _library: Library,
}

/// The NVENC entry points, or `None` when the library is missing or its
/// driver is older than API 12.0.
pub fn api() -> Option<&'static Api> {
    static API: OnceLock<Option<Api>> = OnceLock::new();
    API.get_or_init(Api::load).as_ref()
}

impl Api {
    fn load() -> Option<Self> {
        let library = Library::open(&[c"libnvidia-encode.so.1", c"libnvidia-encode.so"])?;
        // SAFETY: both have these signatures in nvEncodeAPI.h.
        let (max_version, create) = unsafe {
            (
                library.symbol::<GetMaxSupportedVersion>(c"NvEncodeAPIGetMaxSupportedVersion")?,
                library.symbol::<CreateInstance>(c"NvEncodeAPICreateInstance")?,
            )
        };
        let mut driver_api = 0;
        // SAFETY: the out pointer is valid.
        if unsafe { max_version(&raw mut driver_api) } != abi::SUCCESS {
            return None;
        }
        let wanted = (abi::API_MAJOR << 4) | abi::API_MINOR;
        if driver_api < wanted {
            cutlass_logger::info!(
                "{BACKEND} needs driver API {}.{}, this driver offers {}.{}",
                abi::API_MAJOR,
                abi::API_MINOR,
                driver_api >> 4,
                driver_api & 0xf
            );
            return None;
        }
        let mut list = Blob::new(abi::functions::SIZE);
        list.put_u32(abi::functions::VERSION, abi::FUNCTION_LIST_VER);
        // SAFETY: the list is as large as NV_ENCODE_API_FUNCTION_LIST and
        // carries its version; the call fills the function pointers.
        if unsafe { create(list.as_mut_ptr()) } != abi::SUCCESS {
            return None;
        }
        // SAFETY: each slot is the entry point of that name, whose
        // signature is the type it is read as; a null slot is `None`.
        unsafe {
            Some(Self {
                open_session_ex: entry(&list, abi::functions::OPEN_ENCODE_SESSION_EX)?,
                get_guid_count: entry(&list, abi::functions::GET_ENCODE_GUID_COUNT)?,
                get_guids: entry(&list, abi::functions::GET_ENCODE_GUIDS)?,
                get_caps: entry(&list, abi::functions::GET_ENCODE_CAPS)?,
                get_preset_config_ex: entry(&list, abi::functions::GET_ENCODE_PRESET_CONFIG_EX)?,
                initialize: entry(&list, abi::functions::INITIALIZE_ENCODER)?,
                create_input: entry(&list, abi::functions::CREATE_INPUT_BUFFER)?,
                destroy_input: entry(&list, abi::functions::DESTROY_INPUT_BUFFER)?,
                create_bitstream: entry(&list, abi::functions::CREATE_BITSTREAM_BUFFER)?,
                destroy_bitstream: entry(&list, abi::functions::DESTROY_BITSTREAM_BUFFER)?,
                encode_picture: entry(&list, abi::functions::ENCODE_PICTURE)?,
                lock_bitstream: entry(&list, abi::functions::LOCK_BITSTREAM)?,
                unlock_bitstream: entry(&list, abi::functions::UNLOCK_BITSTREAM)?,
                lock_input: entry(&list, abi::functions::LOCK_INPUT_BUFFER)?,
                unlock_input: entry(&list, abi::functions::UNLOCK_INPUT_BUFFER)?,
                get_sequence_params: entry(&list, abi::functions::GET_SEQUENCE_PARAMS)?,
                destroy_encoder: entry(&list, abi::functions::DESTROY_ENCODER)?,
                last_error: entry(&list, abi::functions::GET_LAST_ERROR_STRING)?,
                driver_api,
                _library: library,
            })
        }
    }

    /// The driver's highest API as `(major, minor)`.
    pub fn driver_api(&self) -> (u32, u32) {
        (self.driver_api >> 4, self.driver_api & 0xf)
    }
}

/// The function pointer in `list` at `offset`.
///
/// # Safety
///
/// `F` is the pointer type of the entry point in that slot.
unsafe fn entry<F: Copy>(list: &Blob, offset: usize) -> Option<F> {
    let pointer = list.ptr(offset);
    if pointer.is_null() {
        return None;
    }
    // SAFETY: `F` is a function pointer, the same size as a data pointer
    // on the platforms the driver ships for, as the caller promises.
    Some(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&pointer) })
}

/// What a session is opened with.
#[derive(Clone, Copy)]
struct Settings {
    kind: Kind,
    width: u32,
    height: u32,
    bitrate: u32,
    constant: bool,
    rate: (u32, u32),
    vui: Option<Vui>,
}

impl Settings {
    fn codec_guid(self) -> [u8; 16] {
        match self.kind {
            Kind::Hevc => abi::CODEC_HEVC,
            Kind::H264 | Kind::Av1 => abi::CODEC_H264,
        }
    }

    fn profile_guid(self) -> [u8; 16] {
        match self.kind {
            Kind::Hevc => abi::HEVC_PROFILE_MAIN,
            Kind::H264 | Kind::Av1 => abi::H264_PROFILE_HIGH,
        }
    }
}

/// One input buffer and the bitstream buffer its picture is written to.
struct Pair {
    input: Handle,
    output: Handle,
}

/// An initialised NVENC session and its buffers.
struct Session {
    api: &'static Api,
    device: &'static Device,
    encoder: Handle,
    pairs: Vec<Pair>,
    next: usize,
    /// Pairs whose picture went in and whose bitstream is not read yet, in
    /// the order they went in.
    pending: VecDeque<usize>,
    /// The parameter sets the session writes, for the configuration.
    header: Vec<Vec<u8>>,
    settings: Settings,
}

// SAFETY: the session and its buffers belong to this value alone, and NVENC
// takes calls on a session from any thread, one at a time, which `&mut`
// ensures.
unsafe impl Send for Session {}

/// Why a session did not open, and whether that is a decline.
fn declined(why: &'static str, settings: Settings) -> CodecError {
    note_decline(BACKEND, settings.kind.codec(), true, why);
    CodecError::Unsupported { what: why }
}

impl Session {
    fn open(settings: Settings) -> Result<Self, CodecError> {
        let unsupported = CodecError::Unsupported { what: "nvenc" };
        let device = cuda::device().ok_or(unsupported)?;
        let api = api().ok_or(unsupported)?;
        device.with(|| Self::open_on(api, device, settings))?
    }

    fn open_on(
        api: &'static Api,
        device: &'static Device,
        settings: Settings,
    ) -> Result<Self, CodecError> {
        let mut params = Blob::new(abi::open_session::SIZE);
        params.put_u32(abi::open_session::VERSION, abi::OPEN_SESSION_VER);
        params.put_u32(abi::open_session::DEVICE_TYPE, abi::DEVICE_TYPE_CUDA);
        params.put_ptr(abi::open_session::DEVICE, device.context());
        params.put_u32(abi::open_session::API_VERSION, abi::API_VERSION);
        let mut encoder = ptr::null_mut();
        // SAFETY: the params are an NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS
        // naming the CUDA context, which lives for the process.
        let opened = unsafe { (api.open_session_ex)(params.as_mut_ptr(), &raw mut encoder) };
        if opened != abi::SUCCESS || encoder.is_null() {
            if !encoder.is_null() {
                // SAFETY: a failed open may still hand back a session to
                // destroy.
                unsafe { (api.destroy_encoder)(encoder) };
            }
            return Err(declined(
                "no session; the GPU may be at its session limit",
                settings,
            ));
        }
        let mut session = Self {
            api,
            device,
            encoder,
            pairs: Vec::new(),
            next: 0,
            pending: VecDeque::new(),
            header: Vec::new(),
            settings,
        };
        session.check_codec()?;
        session.initialize()?;
        session.create_buffers()?;
        session.header = session.sequence_params().unwrap_or_default();
        Ok(session)
    }

    fn check_codec(&self) -> Result<(), CodecError> {
        let settings = self.settings;
        let mut count = 0;
        // SAFETY: the session is open and the out pointer is valid.
        if unsafe { (self.api.get_guid_count)(self.encoder, &raw mut count) } != abi::SUCCESS {
            return Err(declined("no codec list", settings));
        }
        let mut guids = vec![Guid::ZERO; usize::try_from(count).unwrap_or(0).min(64)];
        let mut written = 0;
        let capacity = u32::try_from(guids.len()).unwrap_or(0);
        // SAFETY: the array holds `capacity` GUIDs.
        let listed = unsafe {
            (self.api.get_guids)(self.encoder, guids.as_mut_ptr(), capacity, &raw mut written)
        };
        let codec = Guid::from_bytes(settings.codec_guid());
        let offered = listed == abi::SUCCESS
            && guids
                .iter()
                .take(usize::try_from(written).unwrap_or(0))
                .any(|guid| *guid == codec);
        if !offered {
            return Err(declined("this GPU does not encode the codec", settings));
        }
        let max = (
            self.caps(abi::CAPS_WIDTH_MAX),
            self.caps(abi::CAPS_HEIGHT_MAX),
        );
        let min = (
            self.caps(abi::CAPS_WIDTH_MIN),
            self.caps(abi::CAPS_HEIGHT_MIN),
        );
        let over = matches!(max, (Some(w), Some(h)) if settings.width > w || settings.height > h);
        let under = matches!(min.0, Some(w) if settings.width < w)
            || matches!(min.1, Some(h) if settings.height < h);
        if over || under {
            return Err(declined("the picture size is out of range", settings));
        }
        Ok(())
    }

    /// One capability of the codec, or `None` when the driver does not say.
    fn caps(&self, what: u32) -> Option<u32> {
        let mut param = Blob::new(abi::caps_param::SIZE);
        param.put_u32(abi::caps_param::VERSION, abi::CAPS_PARAM_VER);
        param.put_u32(abi::caps_param::CAPS_TO_QUERY, what);
        let mut value = 0;
        let codec = Guid::from_bytes(self.settings.codec_guid());
        // SAFETY: the param is an NV_ENC_CAPS_PARAM and the out pointer is
        // valid.
        let status =
            unsafe { (self.api.get_caps)(self.encoder, codec, param.as_mut_ptr(), &raw mut value) };
        if status != abi::SUCCESS {
            return None;
        }
        u32::try_from(value).ok().filter(|value| *value > 0)
    }

    fn initialize(&mut self) -> Result<(), CodecError> {
        let settings = self.settings;
        let mut preset = Blob::new(abi::preset::SIZE);
        preset.put_u32(abi::preset::VERSION, abi::PRESET_CONFIG_VER);
        preset.put_u32(abi::preset::CONFIG + abi::config::VERSION, abi::CONFIG_VER);
        // SAFETY: the preset is an NV_ENC_PRESET_CONFIG whose config carries
        // its version; GUIDs pass by value.
        let status = unsafe {
            (self.api.get_preset_config_ex)(
                self.encoder,
                Guid::from_bytes(settings.codec_guid()),
                Guid::from_bytes(abi::PRESET_P4),
                abi::TUNING_INFO_HIGH_QUALITY,
                preset.as_mut_ptr(),
            )
        };
        if status != abi::SUCCESS {
            return Err(self.refused("no preset", status));
        }
        let mut config = preset.slice(abi::preset::CONFIG, abi::config::SIZE);
        configure(&mut config, settings);

        let mut init = Blob::new(abi::init::SIZE);
        init.put_u32(abi::init::VERSION, abi::INITIALIZE_PARAMS_VER);
        init.put_bytes(abi::init::ENCODE_GUID, &settings.codec_guid());
        init.put_bytes(abi::init::PRESET_GUID, &abi::PRESET_P4);
        init.put_u32(abi::init::ENCODE_WIDTH, settings.width);
        init.put_u32(abi::init::ENCODE_HEIGHT, settings.height);
        init.put_u32(abi::init::DAR_WIDTH, settings.width);
        init.put_u32(abi::init::DAR_HEIGHT, settings.height);
        init.put_u32(abi::init::FRAME_RATE_NUM, settings.rate.0);
        init.put_u32(abi::init::FRAME_RATE_DEN, settings.rate.1);
        init.put_u32(abi::init::ENABLE_ENCODE_ASYNC, 0);
        init.put_u32(abi::init::ENABLE_PTD, 1);
        init.put_ptr(abi::init::ENCODE_CONFIG, config.as_mut_ptr());
        init.put_u32(abi::init::MAX_ENCODE_WIDTH, settings.width);
        init.put_u32(abi::init::MAX_ENCODE_HEIGHT, settings.height);
        init.put_u32(abi::init::TUNING_INFO, abi::TUNING_INFO_HIGH_QUALITY);
        init.put_u32(abi::init::BUFFER_FORMAT, abi::BUFFER_FORMAT_NV12);
        // SAFETY: the params are an NV_ENC_INITIALIZE_PARAMS whose config
        // pointer is live for the call.
        let status = unsafe { (self.api.initialize)(self.encoder, init.as_mut_ptr()) };
        if status != abi::SUCCESS {
            return Err(self.refused("the settings were refused", status));
        }
        Ok(())
    }

    fn create_buffers(&mut self) -> Result<(), CodecError> {
        for _ in 0..BUFFERS {
            let mut input = Blob::new(abi::create_input::SIZE);
            input.put_u32(abi::create_input::VERSION, abi::CREATE_INPUT_BUFFER_VER);
            input.put_u32(abi::create_input::WIDTH, self.settings.width);
            input.put_u32(abi::create_input::HEIGHT, self.settings.height);
            input.put_u32(abi::create_input::BUFFER_FMT, abi::BUFFER_FORMAT_NV12);
            // SAFETY: the params are an NV_ENC_CREATE_INPUT_BUFFER.
            let status = unsafe { (self.api.create_input)(self.encoder, input.as_mut_ptr()) };
            if status != abi::SUCCESS {
                return Err(self.refused("no input buffer", status));
            }
            let input = input.ptr(abi::create_input::INPUT_BUFFER);
            let mut output = Blob::new(abi::create_bitstream::SIZE);
            output.put_u32(
                abi::create_bitstream::VERSION,
                abi::CREATE_BITSTREAM_BUFFER_VER,
            );
            // SAFETY: the params are an NV_ENC_CREATE_BITSTREAM_BUFFER.
            let status = unsafe { (self.api.create_bitstream)(self.encoder, output.as_mut_ptr()) };
            if status != abi::SUCCESS {
                // SAFETY: the input buffer was made by this session.
                unsafe { (self.api.destroy_input)(self.encoder, input) };
                return Err(self.refused("no bitstream buffer", status));
            }
            let output = output.ptr(abi::create_bitstream::BITSTREAM_BUFFER);
            self.pairs.push(Pair { input, output });
        }
        Ok(())
    }

    /// The parameter sets the session writes, from Annex B.
    fn sequence_params(&self) -> Option<Vec<Vec<u8>>> {
        let mut bytes = vec![0_u8; SEQUENCE_BYTES];
        let mut written = 0_u32;
        let mut payload = Blob::new(abi::sequence::SIZE);
        payload.put_u32(abi::sequence::VERSION, abi::SEQUENCE_PARAM_PAYLOAD_VER);
        payload.put_u32(
            abi::sequence::IN_BUFFER_SIZE,
            u32::try_from(SEQUENCE_BYTES).ok()?,
        );
        payload.put_ptr(abi::sequence::SPSPPS_BUFFER, bytes.as_mut_ptr().cast());
        payload.put_ptr(
            abi::sequence::OUT_SPSPPS_PAYLOAD_SIZE,
            (&raw mut written).cast(),
        );
        // SAFETY: the payload names a buffer of the size it gives and a
        // valid out pointer, both live for the call.
        let status = unsafe { (self.api.get_sequence_params)(self.encoder, payload.as_mut_ptr()) };
        if status != abi::SUCCESS {
            return None;
        }
        bytes.truncate(usize::try_from(written).ok()?.min(SEQUENCE_BYTES));
        let kind = self.settings.kind;
        let sets = nal::annex_b(&bytes)
            .ok()?
            .into_iter()
            .filter(|unit| {
                kind.nal_type(&unit.bytes)
                    .is_some_and(|nal_type| kind.is_parameter_set(nal_type))
            })
            .map(|unit| unit.bytes)
            .collect();
        Some(sets)
    }

    /// An error for a failed call, with the driver's own words in the log.
    fn refused(&self, what: &'static str, status: Status) -> CodecError {
        // SAFETY: the session is open; the string is owned by the driver
        // and read before the next call.
        let detail = unsafe { (self.api.last_error)(self.encoder) };
        let detail = if detail.is_null() {
            String::new()
        } else {
            // SAFETY: the driver hands back a NUL-terminated string.
            unsafe { CStr::from_ptr(detail) }
                .to_string_lossy()
                .into_owned()
        };
        cutlass_logger::warn!("{BACKEND}: {what} (status {status}) {detail}");
        CodecError::Unsupported { what }
    }

    /// Encode one NV12 picture, packed. Returns the units that came out,
    /// in Annex B, in input order.
    fn encode(&mut self, nv12: &[u8], index: u32, flags: u32) -> Result<Vec<Vec<u8>>, CodecError> {
        let device = self.device;
        device.with(|| self.encode_now(nv12, index, flags))?
    }

    fn encode_now(
        &mut self,
        nv12: &[u8],
        index: u32,
        flags: u32,
    ) -> Result<Vec<Vec<u8>>, CodecError> {
        let slot = self.next;
        if self.pending.contains(&slot) {
            return Err(CodecError::Limit {
                what: "nvenc queue",
            });
        }
        let (input, output) = (self.pairs[slot].input, self.pairs[slot].output);
        let pitch = self.fill(input, nv12)?;
        let (width, height) = (self.settings.width, self.settings.height);
        let mut pic = Blob::new(abi::pic::SIZE);
        pic.put_u32(abi::pic::VERSION, abi::PIC_PARAMS_VER);
        pic.put_u32(abi::pic::INPUT_WIDTH, width);
        pic.put_u32(abi::pic::INPUT_HEIGHT, height);
        pic.put_u32(abi::pic::INPUT_PITCH, pitch);
        pic.put_u32(abi::pic::ENCODE_PIC_FLAGS, flags);
        pic.put_u32(abi::pic::FRAME_IDX, index);
        pic.put_u64(abi::pic::INPUT_TIME_STAMP, u64::from(index));
        pic.put_ptr(abi::pic::INPUT_BUFFER, input);
        pic.put_ptr(abi::pic::OUTPUT_BITSTREAM, output);
        pic.put_u32(abi::pic::BUFFER_FMT, abi::BUFFER_FORMAT_NV12);
        pic.put_u32(abi::pic::PICTURE_STRUCT, abi::PIC_STRUCT_FRAME);
        // SAFETY: the params are an NV_ENC_PIC_PARAMS naming buffers this
        // session made; the input is unlocked.
        let status = unsafe { (self.api.encode_picture)(self.encoder, pic.as_mut_ptr()) };
        self.pending.push_back(slot);
        self.next = (slot + 1) % self.pairs.len();
        match status {
            abi::SUCCESS => self.collect(),
            abi::ERR_NEED_MORE_INPUT => Ok(Vec::new()),
            status => {
                self.pending.clear();
                Err(self.failed("encode", status))
            }
        }
    }

    /// Copy a packed NV12 picture into an input buffer, row by row at the
    /// buffer's pitch, and return the pitch.
    fn fill(&self, input: Handle, nv12: &[u8]) -> Result<u32, CodecError> {
        let width = usize::try_from(self.settings.width).map_err(|_| too_large())?;
        let height = usize::try_from(self.settings.height).map_err(|_| too_large())?;
        let luma = width.checked_mul(height).ok_or_else(too_large)?;
        if nv12.len() != luma + luma / 2 {
            return Err(CodecError::InvalidApi {
                what: "picture length",
            });
        }
        let mut lock = Blob::new(abi::lock_input::SIZE);
        lock.put_u32(abi::lock_input::VERSION, abi::LOCK_INPUT_BUFFER_VER);
        lock.put_ptr(abi::lock_input::INPUT_BUFFER, input);
        // SAFETY: the params are an NV_ENC_LOCK_INPUT_BUFFER naming an
        // input buffer of this session that is not locked.
        let status = unsafe { (self.api.lock_input)(self.encoder, lock.as_mut_ptr()) };
        if status != abi::SUCCESS {
            return Err(self.failed("lock input", status));
        }
        let data = lock.ptr(abi::lock_input::BUFFER_DATA_PTR).cast::<u8>();
        let pitch = lock.u32(abi::lock_input::PITCH);
        let stride = usize::try_from(pitch).unwrap_or(0);
        let copied = if data.is_null() || stride < width {
            Err(self.failed("lock input", status))
        } else {
            let rows = height + height / 2;
            // SAFETY: a locked NV12 buffer of this size holds `height` luma
            // rows then `height / 2` chroma rows, `pitch` bytes apart; it
            // stays mapped until the unlock below.
            let buffer = unsafe { std::slice::from_raw_parts_mut(data, stride * rows) };
            for (row, source) in nv12.chunks_exact(width).enumerate() {
                buffer[row * stride..row * stride + width].copy_from_slice(source);
            }
            Ok(pitch)
        };
        // SAFETY: the buffer was locked above.
        unsafe { (self.api.unlock_input)(self.encoder, input) };
        copied
    }

    /// Read every pending bitstream, oldest first.
    fn collect(&mut self) -> Result<Vec<Vec<u8>>, CodecError> {
        let mut units = Vec::with_capacity(self.pending.len());
        while let Some(slot) = self.pending.pop_front() {
            let output = self.pairs[slot].output;
            let mut lock = Blob::new(abi::lock_bitstream::SIZE);
            lock.put_u32(abi::lock_bitstream::VERSION, abi::LOCK_BITSTREAM_VER);
            lock.put_ptr(abi::lock_bitstream::OUTPUT_BITSTREAM, output);
            // SAFETY: the params are an NV_ENC_LOCK_BITSTREAM naming a
            // bitstream buffer of this session with a picture in flight;
            // without doNotWait the call waits for it.
            let status = unsafe { (self.api.lock_bitstream)(self.encoder, lock.as_mut_ptr()) };
            if status != abi::SUCCESS {
                self.pending.clear();
                return Err(self.failed("lock bitstream", status));
            }
            let data = lock
                .ptr(abi::lock_bitstream::BITSTREAM_BUFFER_PTR)
                .cast::<u8>();
            let len = usize::try_from(lock.u32(abi::lock_bitstream::BITSTREAM_SIZE_IN_BYTES))
                .unwrap_or(0);
            let unit = if data.is_null() {
                Vec::new()
            } else {
                // SAFETY: the locked buffer holds `len` bytes until the
                // unlock below.
                unsafe { std::slice::from_raw_parts(data, len) }.to_vec()
            };
            // SAFETY: the buffer was locked above.
            unsafe { (self.api.unlock_bitstream)(self.encoder, output) };
            units.push(unit);
        }
        Ok(units)
    }

    /// Send end of stream and read what was held back.
    fn finish(&mut self) -> Result<Vec<Vec<u8>>, CodecError> {
        let device = self.device;
        device.with(|| {
            let mut pic = Blob::new(abi::pic::SIZE);
            pic.put_u32(abi::pic::VERSION, abi::PIC_PARAMS_VER);
            pic.put_u32(abi::pic::ENCODE_PIC_FLAGS, abi::PIC_FLAG_EOS);
            // SAFETY: an end-of-stream NV_ENC_PIC_PARAMS names no buffers.
            let status = unsafe { (self.api.encode_picture)(self.encoder, pic.as_mut_ptr()) };
            if status != abi::SUCCESS {
                self.pending.clear();
                return Err(self.failed("end of stream", status));
            }
            self.collect()
        })?
    }

    /// An error for a call that failed mid-stream: not a decline, since
    /// pictures already went in.
    fn failed(&self, what: &'static str, status: Status) -> CodecError {
        let _ = self.refused(what, status);
        CodecError::Malformed {
            at_bit: 0,
            what: "nvenc",
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let api = self.api;
        let encoder = self.encoder;
        let pairs = std::mem::take(&mut self.pairs);
        let _ = self.device.with(|| {
            for pair in pairs {
                // SAFETY: both buffers were made by this session and are
                // not locked.
                unsafe {
                    (api.destroy_input)(encoder, pair.input);
                    (api.destroy_bitstream)(encoder, pair.output);
                }
            }
            // SAFETY: the session is open and nothing uses it after this.
            unsafe { (api.destroy_encoder)(encoder) };
        });
    }
}

fn too_large() -> CodecError {
    CodecError::Limit {
        what: "picture size",
    }
}

/// Patch a preset's `NV_ENC_CONFIG` with this encoder's settings.
fn configure(config: &mut Blob, settings: Settings) {
    use abi::config as c;
    config.put_u32(c::VERSION, abi::CONFIG_VER);
    config.put_bytes(c::PROFILE_GUID, &settings.profile_guid());
    config.put_u32(c::GOP_LENGTH, rate::KEYFRAME_INTERVAL);
    config.put_u32(c::FRAME_INTERVAL_P, 1);
    let (mode, peak) = if settings.constant {
        (abi::RC_CBR, settings.bitrate)
    } else {
        (abi::RC_VBR, rate::peak_bitrate(settings.bitrate))
    };
    config.put_u32(c::RC_MODE, mode);
    config.put_u32(c::RC_AVERAGE_BIT_RATE, settings.bitrate);
    config.put_u32(c::RC_MAX_BIT_RATE, peak);
    config.set_bits(c::RC_ENABLE_LOOKAHEAD, 0);
    let vui = match settings.kind {
        Kind::Hevc => {
            config.put_u32(c::HEVC_IDR_PERIOD, rate::KEYFRAME_INTERVAL);
            config.set_bits(c::HEVC_REPEAT_SPSPPS, 1);
            config.set_bits(c::HEVC_OUTPUT_AUD, 0);
            config.set_bits(c::HEVC_CHROMA_FORMAT_IDC, 1);
            config.set_bits(c::HEVC_PIXEL_BIT_DEPTH_MINUS8, 0);
            c::HEVC_VUI
        }
        Kind::H264 | Kind::Av1 => {
            config.put_u32(c::H264_IDR_PERIOD, rate::KEYFRAME_INTERVAL);
            config.set_bits(c::H264_REPEAT_SPSPPS, 1);
            config.set_bits(c::H264_OUTPUT_AUD, 0);
            config.put_u32(c::H264_CHROMA_FORMAT_IDC, 1);
            c::H264_VUI
        }
    };
    if let Some(colour) = settings.vui {
        use abi::vui as v;
        config.put_u32(vui + v::VIDEO_SIGNAL_TYPE_PRESENT, 1);
        config.put_u32(vui + v::VIDEO_FORMAT, abi::VUI_VIDEO_FORMAT_UNSPECIFIED);
        config.put_u32(vui + v::VIDEO_FULL_RANGE, u32::from(colour.full_range));
        config.put_u32(
            vui + v::COLOUR_DESCRIPTION_PRESENT,
            u32::from(colour.described),
        );
        config.put_u32(vui + v::COLOUR_PRIMARIES, u32::from(colour.primaries));
        config.put_u32(
            vui + v::TRANSFER_CHARACTERISTICS,
            u32::from(colour.transfer),
        );
        config.put_u32(vui + v::COLOUR_MATRIX, u32::from(colour.matrix));
    }
}

/// What NVENC encodes on this GPU: a line for the driver, then one per
/// codec, from opening a 1080p session.
pub fn report() -> Vec<String> {
    let Some(api) = api() else {
        return vec![String::from(
            "NVENC: libnvidia-encode.so.1 is missing or older than API 12.0",
        )];
    };
    let (major, minor) = api.driver_api();
    let mut lines = vec![format!("NVENC: driver API {major}.{minor}")];
    for kind in [Kind::H264, Kind::Hevc] {
        let settings = Settings {
            kind,
            width: 1920,
            height: 1080,
            bitrate: 8_000_000,
            constant: false,
            rate: rate::DEFAULT_RATE,
            vui: None,
        };
        let codec = kind.codec();
        lines.push(match Session::open(settings) {
            Ok(_) => format!("NVENC {codec:?}: opens at 1920x1080"),
            Err(err) => format!("NVENC {codec:?}: no ({err})"),
        });
    }
    lines
}

/// The NVENC encoder behind [`Encoder`].
pub struct NvEncoder {
    settings: Settings,
    /// `None` after a reset or drain, until the next picture opens one.
    session: Option<Session>,
    started: bool,
    frames: u32,
    packets: Packetizer,
    cancel: Arc<AtomicBool>,
}

/// Open an NVENC encoder, or decline so the next opener is asked.
pub fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    let kind = Kind::of(params.codec)?;
    if matches!(kind, Kind::Av1) {
        return Err(CodecError::Unsupported { what: "nvenc av1" });
    }
    let (width, height) = video_size(params.width, params.height)?;
    if !matches!(
        params.pixel,
        PixelFormat::None | PixelFormat::Rgba8 | PixelFormat::Yuv420p
    ) {
        return Err(CodecError::Unsupported {
            what: "pixel format",
        });
    }
    let settings = Settings {
        kind,
        width,
        height,
        bitrate: rate::target_bitrate(params),
        constant: params.constant_rate,
        rate: rate::DEFAULT_RATE,
        vui: Vui::of(params),
    };
    let session = Session::open(settings)?;
    let mut packets = Packetizer::new(kind);
    packets.set_header(session.header.clone());
    note_open(BACKEND, params.codec, true);
    Ok(Box::new(NvEncoder {
        settings,
        session: Some(session),
        started: false,
        frames: 0,
        packets,
        cancel: Arc::new(AtomicBool::new(false)),
    }))
}

impl NvEncoder {
    /// The session for the next picture, opened again at the picture's
    /// rate when that differs from the one it has.
    fn session(&mut self, time: FrameTime) -> Result<&mut Session, CodecError> {
        if !self.started {
            let rate = rate::frame_rate(time);
            if rate != self.settings.rate {
                self.settings.rate = rate;
                self.session = None;
            }
        }
        if self.session.is_none() {
            // The old session is gone first, so a GPU at its session limit
            // has room for this one.
            let session = Session::open(self.settings)?;
            self.packets.set_header(session.header.clone());
            self.session = Some(session);
        }
        self.session
            .as_mut()
            .ok_or(CodecError::Unsupported { what: "nvenc" })
    }

    fn packets(&mut self, units: Vec<Vec<u8>>) -> Result<Vec<Packet>, CodecError> {
        let mut packets = Vec::with_capacity(units.len());
        for unit in units {
            if let Some(packet) = self.packets.packet(&unit)? {
                packets.push(packet);
            }
        }
        Ok(packets)
    }
}

impl Encoder for NvEncoder {
    fn codec(&self) -> CodecId {
        self.settings.kind.codec()
    }

    fn push(&mut self, _input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::Unsupported {
            what: "audio input",
        })
    }

    fn push_picture(&mut self, _input: &PictureInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::Unsupported {
            what: "picture input",
        })
    }

    fn push_frame(&mut self, input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let FrameInput::Cpu(frame) = input else {
            return Err(CodecError::Unsupported {
                what: "platform frame",
            });
        };
        let (width, height) = (self.settings.width, self.settings.height);
        if frame.width() != width || frame.height() != height {
            return Err(CodecError::Unsupported { what: "frame" });
        }
        let picture = match frame.pixel() {
            PixelFormat::Rgba8 => nv12::from_rgba(width, height, frame.bytes())?,
            PixelFormat::Yuv420p => nv12::from_planar(width, height, frame.bytes())?,
            _ => {
                return Err(CodecError::Unsupported {
                    what: "pixel format",
                });
            }
        };
        let time = frame.time();
        let index = self.frames;
        let flags = if self.started {
            0
        } else {
            abi::PIC_FLAG_FORCEIDR | abi::PIC_FLAG_OUTPUT_SPSPPS
        };
        let units = self.session(time)?.encode(&picture, index, flags)?;
        self.started = true;
        self.frames = self.frames.wrapping_add(1);
        self.packets.sent(time);
        self.packets(units)
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        if !self.started {
            return Ok(Vec::new());
        }
        let units = match self.session.as_mut() {
            Some(session) => session.finish(),
            None => Ok(Vec::new()),
        };
        // A drained session takes no more pictures, so the next one opens
        // a fresh session that starts with a keyframe.
        self.session = None;
        self.started = false;
        let packets = self.packets(units?);
        self.packets.reset();
        packets
    }

    /// A fresh session for the next picture, which is a keyframe that
    /// carries the configuration again.
    fn reset(&mut self) {
        self.session = None;
        self.started = false;
        self.frames = 0;
        self.packets.reset();
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_guid_has_the_c_layout_and_the_header_values() {
        assert_eq!(size_of::<Guid>(), 16);
        // NV_ENC_CODEC_H264_GUID = {6BC82762-4E63-4CA4-AA85-1E50F321F6BF}
        let guid = Guid::from_bytes(abi::CODEC_H264);
        assert_eq!(guid.data1, 0x6bc8_2762);
        assert_eq!((guid.data2, guid.data3), (0x4e63, 0x4ca4));
        assert_eq!(guid.data4[0], 0xaa);
    }

    #[test]
    fn the_config_takes_the_settings() {
        let mut config = Blob::new(abi::config::SIZE);
        let settings = Settings {
            kind: Kind::Hevc,
            width: 1920,
            height: 1080,
            bitrate: 8_000_000,
            constant: false,
            rate: (30, 1),
            vui: Some(Vui {
                primaries: 1,
                transfer: 1,
                matrix: 1,
                full_range: false,
                described: true,
            }),
        };
        configure(&mut config, settings);
        use abi::config as c;
        assert_eq!(config.u32(c::GOP_LENGTH), 30);
        assert_eq!(config.u32(c::FRAME_INTERVAL_P), 1);
        assert_eq!(config.u32(c::RC_MODE), abi::RC_VBR);
        assert_eq!(config.u32(c::RC_MAX_BIT_RATE), 16_000_000);
        assert_eq!(config.u32(c::HEVC_IDR_PERIOD), 30);
        let repeat = c::HEVC_REPEAT_SPSPPS;
        assert_eq!((config.u32(repeat.offset) >> repeat.shift) & 1, 1);
        assert_eq!(config.u32(c::HEVC_VUI + abi::vui::COLOUR_PRIMARIES), 1);
        let profile = abi::HEVC_PROFILE_MAIN;
        assert_eq!(
            config.u32(c::PROFILE_GUID),
            u32::from_le_bytes([profile[0], profile[1], profile[2], profile[3]])
        );
    }
}
