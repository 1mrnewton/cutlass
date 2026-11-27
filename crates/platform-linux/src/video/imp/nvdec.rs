//! NVDEC: H.264, HEVC, and AV1 decode on an NVIDIA GPU.
//!
//! NVIDIA's own parser in `libnvcuvid.so.1` reads the bitstream, keeps the
//! reference pictures, and calls back in display order, so this module
//! feeds it whole access units and copies each displayed picture back to
//! the CPU. Its three callbacks run inside `cuvidParseVideoData`, on the
//! calling thread, and write into the [`Session`] the parser was given.
//!
//! The sequence callback checks the stream against `cuvidGetDecoderCaps`
//! and (re)creates the decoder when the size or format changes. 8-bit
//! streams come out as NV12 and deeper ones as P016; both become 8-bit
//! planar 4:2:0, cropped to the display area. A stream the GPU does not
//! decode fails the push, and [`Fallback`] hands it to the Rust decoder.

use std::ffi::{c_int, c_short, c_uint, c_ulong, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::sync::OnceLock;

use cutlass_color::{nv12, p010};
use cutlass_decode::{CodecError, CodecId, CodecParams, Decoder, Output, Packet, PictureOutput};

use super::cuda::{self, CUDA_SUCCESS, CuDevicePtr, CuResult, Device};
use crate::dl::Library;
use crate::video::backend::{note_decline, note_open};
use crate::video::colour::{Signal, picture_info};
use crate::video::fallback::Fallback;
use crate::video::kind::{Framing, Kind};
use crate::video::limits::packet_ok;

pub const BACKEND: &str = "NVDEC";

/// `cudaVideoChromaFormat_420`.
const CHROMA_420: c_uint = 1;
/// `cudaVideoSurfaceFormat_NV12`.
const SURFACE_NV12: c_uint = 0;
/// `cudaVideoSurfaceFormat_P016`.
const SURFACE_P016: c_uint = 1;
/// `cudaVideoDeinterlaceMode_Weave`.
const DEINTERLACE_WEAVE: c_uint = 0;
/// `cudaVideoDeinterlaceMode_Adaptive`.
const DEINTERLACE_ADAPTIVE: c_uint = 2;
/// `cudaVideoCreate_PreferCUVID`: the dedicated decode engines.
const CREATE_PREFER_CUVID: c_ulong = 0x04;
/// `CUVID_PKT_ENDOFSTREAM`.
const PACKET_END_OF_STREAM: c_ulong = 0x01;
/// `CUVID_PKT_TIMESTAMP`.
const PACKET_TIMESTAMP: c_ulong = 0x02;
/// `CUVID_PKT_ENDOFPICTURE`: the packet is one whole picture.
const PACKET_END_OF_PICTURE: c_ulong = 0x08;
/// Surfaces beyond the stream's minimum, so the parser can run ahead.
const EXTRA_SURFACES: u32 = 4;

/// `CUVIDDECODECAPS`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct DecodeCaps {
    pub codec: c_uint,
    pub chroma_format: c_uint,
    pub bit_depth_minus8: c_uint,
    pub reserved1: [c_uint; 3],
    pub is_supported: u8,
    pub num_nvdecs: u8,
    pub output_format_mask: u16,
    pub max_width: c_uint,
    pub max_height: c_uint,
    pub max_mb_count: c_uint,
    pub min_width: u16,
    pub min_height: u16,
    pub is_histogram_supported: u8,
    pub counter_bit_depth: u8,
    pub max_histogram_bins: u16,
    pub reserved3: [c_uint; 10],
}

/// `struct { short left, top, right, bottom; }`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ShortRect {
    left: c_short,
    top: c_short,
    right: c_short,
    bottom: c_short,
}

/// `CUVIDDECODECREATEINFO`.
#[repr(C)]
#[derive(Clone, Copy)]
struct CreateInfo {
    width: c_ulong,
    height: c_ulong,
    num_decode_surfaces: c_ulong,
    codec: c_uint,
    chroma_format: c_uint,
    creation_flags: c_ulong,
    bit_depth_minus8: c_ulong,
    intra_decode_only: c_ulong,
    max_width: c_ulong,
    max_height: c_ulong,
    reserved1: c_ulong,
    display_area: ShortRect,
    output_format: c_uint,
    deinterlace_mode: c_uint,
    target_width: c_ulong,
    target_height: c_ulong,
    num_output_surfaces: c_ulong,
    vid_lock: *mut c_void,
    target_rect: ShortRect,
    enable_histogram: c_ulong,
    reserved2: [c_ulong; 4],
}

/// `CUVIDPROCPARAMS`.
#[repr(C)]
#[derive(Clone, Copy)]
struct ProcParams {
    progressive_frame: c_int,
    second_field: c_int,
    top_field_first: c_int,
    unpaired_field: c_int,
    reserved_flags: c_uint,
    reserved_zero: c_uint,
    raw_input_dptr: u64,
    raw_input_pitch: c_uint,
    raw_input_format: c_uint,
    raw_output_dptr: u64,
    raw_output_pitch: c_uint,
    reserved1: c_uint,
    output_stream: *mut c_void,
    reserved: [c_uint; 46],
    histogram_dptr: *mut u64,
    reserved2: [*mut c_void; 1],
}

/// `struct { int left, top, right, bottom; }`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct IntRect {
    left: c_int,
    top: c_int,
    right: c_int,
    bottom: c_int,
}

/// `CUVIDEOFORMAT`.
#[repr(C)]
#[derive(Clone, Copy)]
struct VideoFormat {
    codec: c_uint,
    frame_rate: [c_uint; 2],
    progressive_sequence: u8,
    bit_depth_luma_minus8: u8,
    bit_depth_chroma_minus8: u8,
    min_num_decode_surfaces: u8,
    coded_width: c_uint,
    coded_height: c_uint,
    display_area: IntRect,
    chroma_format: c_uint,
    bitrate: c_uint,
    display_aspect_ratio: [c_int; 2],
    /// `video_format:3, video_full_range_flag:1, reserved:4`.
    signal_flags: u8,
    color_primaries: u8,
    transfer_characteristics: u8,
    matrix_coefficients: u8,
    seqhdr_data_length: c_uint,
}

/// `CUVIDSOURCEDATAPACKET`.
#[repr(C)]
#[derive(Clone, Copy)]
struct SourcePacket {
    flags: c_ulong,
    payload_size: c_ulong,
    payload: *const u8,
    timestamp: i64,
}

/// `CUVIDPARSERDISPINFO`.
#[repr(C)]
#[derive(Clone, Copy)]
struct DisplayInfo {
    picture_index: c_int,
    progressive_frame: c_int,
    top_field_first: c_int,
    repeat_first_field: c_int,
    timestamp: i64,
}

/// `CUVIDPICPARAMS`, passed from the parser to the decoder unread.
#[repr(C)]
struct PicParams {
    _opaque: [u8; 0],
}

type SequenceCallback = unsafe extern "C" fn(*mut c_void, *mut VideoFormat) -> c_int;
type DecodeCallback = unsafe extern "C" fn(*mut c_void, *mut PicParams) -> c_int;
type DisplayCallback = unsafe extern "C" fn(*mut c_void, *mut DisplayInfo) -> c_int;

/// `CUVIDPARSERPARAMS`.
#[repr(C)]
#[derive(Clone, Copy)]
struct ParserParams {
    codec: c_uint,
    max_num_decode_surfaces: c_uint,
    clock_rate: c_uint,
    error_threshold: c_uint,
    max_display_delay: c_uint,
    /// `bAnnexb:1, uReserved:31`.
    flags: c_uint,
    reserved1: [c_uint; 4],
    user_data: *mut c_void,
    sequence: Option<SequenceCallback>,
    decode: Option<DecodeCallback>,
    display: Option<DisplayCallback>,
    operating_point: *mut c_void,
    sei: *mut c_void,
    reserved2: [*mut c_void; 5],
    ext_video_info: *mut c_void,
}

type Handle = *mut c_void;
type GetDecoderCaps = unsafe extern "C" fn(*mut DecodeCaps) -> CuResult;
type CreateDecoder = unsafe extern "C" fn(*mut Handle, *mut CreateInfo) -> CuResult;
type DestroyDecoder = unsafe extern "C" fn(Handle) -> CuResult;
type DecodePicture = unsafe extern "C" fn(Handle, *mut PicParams) -> CuResult;
type MapFrame =
    unsafe extern "C" fn(Handle, c_int, *mut u64, *mut c_uint, *mut ProcParams) -> CuResult;
type UnmapFrame = unsafe extern "C" fn(Handle, u64) -> CuResult;
type CreateParser = unsafe extern "C" fn(*mut Handle, *mut ParserParams) -> CuResult;
type ParseData = unsafe extern "C" fn(Handle, *mut SourcePacket) -> CuResult;
type DestroyParser = unsafe extern "C" fn(Handle) -> CuResult;

/// The `libnvcuvid` calls this module makes.
pub struct Nvcuvid {
    pub get_decoder_caps: GetDecoderCaps,
    create_decoder: CreateDecoder,
    destroy_decoder: DestroyDecoder,
    decode_picture: DecodePicture,
    map_frame: MapFrame,
    unmap_frame: UnmapFrame,
    create_parser: CreateParser,
    parse_data: ParseData,
    destroy_parser: DestroyParser,
    _library: Library,
}

/// The process-wide `libnvcuvid`, or `None` when the driver lacks it.
pub fn nvcuvid() -> Option<&'static Nvcuvid> {
    static NVCUVID: OnceLock<Option<Nvcuvid>> = OnceLock::new();
    NVCUVID
        .get_or_init(|| {
            let library = Library::open(&[c"libnvcuvid.so.1", c"libnvcuvid.so"])?;
            // SAFETY: each symbol has this signature in cuviddec.h and
            // nvcuvid.h; the 64-bit map calls are the ones used. The
            // pointers live in the same struct as the library.
            unsafe {
                Some(Nvcuvid {
                    get_decoder_caps: library.symbol(c"cuvidGetDecoderCaps")?,
                    create_decoder: library.symbol(c"cuvidCreateDecoder")?,
                    destroy_decoder: library.symbol(c"cuvidDestroyDecoder")?,
                    decode_picture: library.symbol(c"cuvidDecodePicture")?,
                    map_frame: library.symbol(c"cuvidMapVideoFrame64")?,
                    unmap_frame: library.symbol(c"cuvidUnmapVideoFrame64")?,
                    create_parser: library.symbol(c"cuvidCreateVideoParser")?,
                    parse_data: library.symbol(c"cuvidParseVideoData")?,
                    destroy_parser: library.symbol(c"cuvidDestroyVideoParser")?,
                    _library: library,
                })
            }
        })
        .as_ref()
}

/// `cudaVideoCodec` for a codec.
pub fn codec_of(kind: Kind) -> c_uint {
    match kind {
        Kind::H264 => 4,
        Kind::Hevc => 8,
        Kind::Av1 => 11,
    }
}

/// What the GPU decodes for `kind` at 4:2:0 and `bit_depth`, with the
/// context current. `None` when the call fails.
pub fn caps(nv: &Nvcuvid, kind: Kind, bit_depth: u8) -> Option<DecodeCaps> {
    let mut caps = DecodeCaps {
        codec: codec_of(kind),
        chroma_format: CHROMA_420,
        bit_depth_minus8: c_uint::from(bit_depth.saturating_sub(8)),
        ..DecodeCaps::default()
    };
    // SAFETY: the struct is the C layout and the context is current, as
    // the caller promises.
    (unsafe { (nv.get_decoder_caps)(&raw mut caps) } == CUDA_SUCCESS).then_some(caps)
}

/// What NVDEC decodes on this GPU, a line per codec and bit depth.
pub fn report(device: &Device) -> Vec<String> {
    let Some(nv) = nvcuvid() else {
        return vec![String::from("NVDEC: libnvcuvid.so.1 is missing")];
    };
    let mut lines = Vec::new();
    for kind in [Kind::H264, Kind::Hevc, Kind::Av1] {
        for depth in [8, 10] {
            let codec = kind.codec();
            let line = match device.with(|| caps(nv, kind, depth)).ok().flatten() {
                Some(caps) if caps.is_supported != 0 => format!(
                    "NVDEC {codec:?} {depth}-bit: up to {}x{}",
                    caps.max_width, caps.max_height
                ),
                _ => format!("NVDEC {codec:?} {depth}-bit: no"),
            };
            lines.push(line);
        }
    }
    lines
}

/// The NVDEC opener. Declines when there is no NVIDIA GPU or driver, or
/// the GPU does not decode the codec at the size named.
pub fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(open(params)?))
}

/// The NVDEC decoder, inside the fallback to the Rust decoder.
pub fn open(params: &CodecParams) -> Result<Fallback, CodecError> {
    let kind = Kind::of(params.codec)?;
    if params.width != 0 || params.height != 0 {
        crate::video::limits::video_size(params.width, params.height)?;
    }
    let decline = CodecError::Unsupported { what: "nvdec" };
    let device = cuda::device().ok_or(decline)?;
    let nv = nvcuvid().ok_or(decline)?;
    let supported = device.with(|| caps(nv, kind, 8))?;
    let Some(supported) = supported.filter(|caps| caps.is_supported != 0) else {
        note_decline(
            BACKEND,
            params.codec,
            false,
            "this GPU does not decode the codec",
        );
        return Err(decline);
    };
    if params.width > supported.max_width || params.height > supported.max_height {
        note_decline(
            BACKEND,
            params.codec,
            false,
            "the picture size is out of range",
        );
        return Err(decline);
    }
    let decoder = NvDecoder::new(device, nv, kind, params)?;
    note_open(BACKEND, params.codec, false);
    Ok(Fallback::new(params, Box::new(decoder), BACKEND))
}

/// The decoder the parser drives, and what its callbacks hand back.
struct Session {
    device: &'static Device,
    nv: &'static Nvcuvid,
    kind: Kind,
    decoder: Handle,
    /// The format the decoder was made for.
    made: Option<Made>,
    pictures: Vec<PictureOutput>,
    error: Option<CodecError>,
    timescale: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Made {
    coded: (u32, u32),
    display: (i32, i32, u32, u32),
    bit_depth_minus8: u8,
    surfaces: u32,
    progressive: bool,
    signal: (u8, u8, u8, u8),
}

impl Made {
    fn target(&self) -> (u32, u32) {
        (self.display.2, self.display.3)
    }

    fn signal(&self) -> Signal {
        let (flags, primaries, transfer, matrix) = self.signal;
        Signal {
            primaries,
            transfer,
            matrix,
            full_range: Some(flags & 0x08 != 0),
        }
    }
}

pub struct NvDecoder {
    /// Boxed so the parser's user pointer stays put.
    session: Box<Session>,
    parser: Handle,
    framing: Framing,
}

// SAFETY: the parser, decoder, and context handles may be used from any
// thread as long as calls are not concurrent, which `&mut self` ensures;
// every call pushes the context first.
unsafe impl Send for NvDecoder {}

impl NvDecoder {
    fn new(
        device: &'static Device,
        nv: &'static Nvcuvid,
        kind: Kind,
        params: &CodecParams,
    ) -> Result<Self, CodecError> {
        Ok(Self {
            session: Box::new(Session {
                device,
                nv,
                kind,
                decoder: ptr::null_mut(),
                made: None,
                pictures: Vec::new(),
                error: None,
                timescale: 30,
            }),
            parser: ptr::null_mut(),
            framing: Framing::new(kind, params)?,
        })
    }

    /// A parser for the stream, made on the first unit so its clock is
    /// the packets' timescale. The context must be current.
    fn ensure_parser(&mut self) -> Result<(), CodecError> {
        if !self.parser.is_null() {
            return Ok(());
        }
        let mut params = ParserParams {
            codec: codec_of(self.session.kind),
            max_num_decode_surfaces: 1,
            clock_rate: self.session.timescale,
            error_threshold: 0,
            max_display_delay: 0,
            flags: 0,
            reserved1: [0; 4],
            user_data: (&raw mut *self.session).cast(),
            sequence: Some(on_sequence),
            decode: Some(on_decode),
            display: Some(on_display),
            operating_point: ptr::null_mut(),
            sei: ptr::null_mut(),
            reserved2: [ptr::null_mut(); 5],
            ext_video_info: ptr::null_mut(),
        };
        let mut parser = ptr::null_mut();
        // SAFETY: the struct is the C layout; the user pointer is the boxed
        // session, which outlives the parser (destroyed first on drop).
        let made = unsafe { (self.session.nv.create_parser)(&raw mut parser, &raw mut params) };
        if made != CUDA_SUCCESS || parser.is_null() {
            return Err(CodecError::Unsupported {
                what: "nvdec parser",
            });
        }
        self.parser = parser;
        Ok(())
    }

    /// Hand `payload` to the parser and collect what its callbacks made.
    fn parse(
        &mut self,
        payload: &[u8],
        flags: c_ulong,
        pts: i64,
    ) -> Result<Vec<Output>, CodecError> {
        let session_device = self.session.device;
        session_device.with(|| -> Result<Vec<Output>, CodecError> {
            self.ensure_parser()?;
            let mut packet = SourcePacket {
                flags,
                payload_size: payload.len() as c_ulong,
                payload: if payload.is_empty() {
                    ptr::null()
                } else {
                    payload.as_ptr()
                },
                timestamp: pts,
            };
            // SAFETY: the parser is live, the payload outlives the call,
            // and the callbacks reach only the boxed session.
            let parsed = unsafe { (self.session.nv.parse_data)(self.parser, &raw mut packet) };
            if let Some(error) = self.session.error.take() {
                self.session.pictures.clear();
                return Err(error);
            }
            if parsed != CUDA_SUCCESS {
                self.session.pictures.clear();
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "nvdec could not parse the unit",
                });
            }
            Ok(std::mem::take(&mut self.session.pictures)
                .into_iter()
                .map(Output::Picture)
                .collect())
        })?
    }

    /// Drop the parser, so the next unit starts a new one. The context
    /// must be current.
    fn drop_parser(&mut self) {
        if !self.parser.is_null() {
            // SAFETY: the parser is live and dropped once.
            unsafe { (self.session.nv.destroy_parser)(self.parser) };
            self.parser = ptr::null_mut();
        }
    }
}

impl Decoder for NvDecoder {
    fn codec(&self) -> CodecId {
        self.session.kind.codec()
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        packet_ok(packet.data.len())?;
        if packet.timescale != 0 && self.parser.is_null() {
            self.session.timescale = packet.timescale;
        }
        let unit = self.framing.unit(packet)?;
        self.parse(&unit, PACKET_TIMESTAMP | PACKET_END_OF_PICTURE, packet.pts)
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        if self.parser.is_null() {
            return Ok(Vec::new());
        }
        let out = self.parse(&[], PACKET_END_OF_STREAM, 0);
        let device = self.session.device;
        let _ = device.with(|| self.drop_parser());
        self.framing.reset();
        out
    }

    fn reset(&mut self) {
        let device = self.session.device;
        let _ = device.with(|| self.drop_parser());
        self.session.pictures.clear();
        self.session.error = None;
        self.framing.reset();
    }
}

impl Drop for NvDecoder {
    fn drop(&mut self) {
        let device = self.session.device;
        let _ = device.with(|| {
            self.drop_parser();
            self.session.destroy_decoder();
        });
    }
}

impl Session {
    /// The context must be current.
    fn destroy_decoder(&mut self) {
        if !self.decoder.is_null() {
            // SAFETY: the decoder is live and destroyed once.
            unsafe { (self.nv.destroy_decoder)(self.decoder) };
            self.decoder = ptr::null_mut();
            self.made = None;
        }
    }

    /// A new sequence: check it, make the decoder, and return the surface
    /// count for the parser.
    fn sequence(&mut self, format: &VideoFormat) -> Result<u32, CodecError> {
        if format.chroma_format != CHROMA_420 {
            return Err(CodecError::Unsupported {
                what: "nvdec chroma format",
            });
        }
        let depth = format.bit_depth_luma_minus8;
        if depth != format.bit_depth_chroma_minus8 || depth > 4 {
            return Err(CodecError::Unsupported {
                what: "nvdec bit depth",
            });
        }
        let caps = caps(self.nv, self.kind, depth + 8)
            .ok_or(CodecError::Unsupported { what: "nvdec caps" })?;
        let surface_bit = if depth == 0 {
            SURFACE_NV12
        } else {
            SURFACE_P016
        };
        if caps.is_supported == 0
            || caps.output_format_mask & (1 << surface_bit) == 0
            || format.coded_width > caps.max_width
            || format.coded_height > caps.max_height
            || format.coded_width < u32::from(caps.min_width)
            || format.coded_height < u32::from(caps.min_height)
        {
            return Err(CodecError::Unsupported {
                what: "nvdec does not decode this stream",
            });
        }
        let area = format.display_area;
        let width = u32::try_from(area.right - area.left).unwrap_or(0) & !1;
        let height = u32::try_from(area.bottom - area.top).unwrap_or(0) & !1;
        if width == 0 || height == 0 {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "nvdec display area",
            });
        }
        let made = Made {
            coded: (format.coded_width, format.coded_height),
            display: (area.left, area.top, width, height),
            bit_depth_minus8: depth,
            surfaces: u32::from(format.min_num_decode_surfaces) + EXTRA_SURFACES,
            progressive: format.progressive_sequence != 0,
            signal: (
                format.signal_flags,
                format.color_primaries,
                format.transfer_characteristics,
                format.matrix_coefficients,
            ),
        };
        if self.made == Some(made) && !self.decoder.is_null() {
            return Ok(made.surfaces);
        }
        self.destroy_decoder();
        let short = |value: i32| c_short::try_from(value).unwrap_or(c_short::MAX);
        let mut info = CreateInfo {
            width: c_ulong::from(format.coded_width),
            height: c_ulong::from(format.coded_height),
            num_decode_surfaces: c_ulong::from(made.surfaces),
            codec: codec_of(self.kind),
            chroma_format: CHROMA_420,
            creation_flags: CREATE_PREFER_CUVID,
            bit_depth_minus8: c_ulong::from(depth),
            intra_decode_only: 0,
            max_width: c_ulong::from(format.coded_width),
            max_height: c_ulong::from(format.coded_height),
            reserved1: 0,
            display_area: ShortRect {
                left: short(area.left),
                top: short(area.top),
                right: short(area.left + i32::try_from(width).unwrap_or(0)),
                bottom: short(area.top + i32::try_from(height).unwrap_or(0)),
            },
            output_format: surface_bit,
            deinterlace_mode: if made.progressive {
                DEINTERLACE_WEAVE
            } else {
                DEINTERLACE_ADAPTIVE
            },
            target_width: c_ulong::from(width),
            target_height: c_ulong::from(height),
            num_output_surfaces: 2,
            vid_lock: ptr::null_mut(),
            target_rect: ShortRect::default(),
            enable_histogram: 0,
            reserved2: [0; 4],
        };
        let mut decoder = ptr::null_mut();
        // SAFETY: the struct is the C layout and the context is current
        // inside the parse call.
        let created = unsafe { (self.nv.create_decoder)(&raw mut decoder, &raw mut info) };
        if created != CUDA_SUCCESS || decoder.is_null() {
            return Err(CodecError::Unsupported {
                what: "nvdec could not make a decoder",
            });
        }
        self.decoder = decoder;
        self.made = Some(made);
        Ok(made.surfaces)
    }

    fn decode(&mut self, params: *mut PicParams) -> Result<(), CodecError> {
        if self.decoder.is_null() {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "nvdec picture before its sequence",
            });
        }
        // SAFETY: the decoder is live and the parameters come straight
        // from the parser for this call.
        if unsafe { (self.nv.decode_picture)(self.decoder, params) } != CUDA_SUCCESS {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "nvdec could not decode a picture",
            });
        }
        Ok(())
    }

    /// Copy the picture at `display` back and keep it.
    fn display(&mut self, display: &DisplayInfo) -> Result<(), CodecError> {
        let Some(made) = self.made else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "nvdec picture before its sequence",
            });
        };
        let mut proc = ProcParams {
            progressive_frame: display.progressive_frame,
            second_field: 0,
            top_field_first: display.top_field_first,
            unpaired_field: c_int::from(display.repeat_first_field < 0),
            reserved_flags: 0,
            reserved_zero: 0,
            raw_input_dptr: 0,
            raw_input_pitch: 0,
            raw_input_format: 0,
            raw_output_dptr: 0,
            raw_output_pitch: 0,
            reserved1: 0,
            output_stream: ptr::null_mut(),
            reserved: [0; 46],
            histogram_dptr: ptr::null_mut(),
            reserved2: [ptr::null_mut()],
        };
        let mut frame: CuDevicePtr = 0;
        let mut pitch: c_uint = 0;
        // SAFETY: the decoder is live, the index is the parser's, and the
        // out pointers are valid.
        let mapped = unsafe {
            (self.nv.map_frame)(
                self.decoder,
                display.picture_index,
                &raw mut frame,
                &raw mut pitch,
                &raw mut proc,
            )
        };
        if mapped != CUDA_SUCCESS || frame == 0 {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "nvdec could not map a picture",
            });
        }
        let copied = self.copy(made, frame, pitch as usize);
        // SAFETY: the frame was mapped just above and is unmapped once.
        unsafe { (self.nv.unmap_frame)(self.decoder, frame) };
        let bytes = copied?;
        let (width, height) = made.target();
        let info = picture_info(width, height, made.signal());
        self.pictures.push(PictureOutput::cpu(
            info,
            bytes,
            display.timestamp,
            self.timescale,
        )?);
        Ok(())
    }

    /// The mapped frame as 8-bit planar 4:2:0. The luma plane is
    /// `height` rows and chroma starts right after it, rounded to even.
    fn copy(&self, made: Made, frame: CuDevicePtr, pitch: usize) -> Result<Vec<u8>, CodecError> {
        let (width, height) = made.target();
        let sample = if made.bit_depth_minus8 == 0 { 1 } else { 2 };
        let row = width as usize * sample;
        let rows = height as usize;
        let mut packed = Vec::with_capacity(row * rows * 3 / 2);
        self.device
            .copy_rows(frame, pitch, row, rows, &mut packed)?;
        let chroma = frame + (pitch * rows.next_multiple_of(2)) as u64;
        self.device
            .copy_rows(chroma, pitch, row, rows / 2, &mut packed)?;
        if sample == 1 {
            nv12::to_planar(nv12::Layout::packed(width, height), &packed)
        } else {
            p010::to_planar8(p010::Layout::packed(width, height), &packed)
        }
    }
}

/// Run a callback body on the session behind `user`, turning an error or
/// a panic into the parser's failure value and keeping the error.
fn callback(
    user: *mut c_void,
    body: impl FnOnce(&mut Session) -> Result<c_int, CodecError>,
) -> c_int {
    if user.is_null() {
        return 0;
    }
    // SAFETY: the parser was made with the boxed session as its user
    // pointer, and calls back only inside `parse`, where nothing else
    // borrows the session.
    let session = unsafe { &mut *user.cast::<Session>() };
    match catch_unwind(AssertUnwindSafe(|| body(&mut *session))) {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => {
            session.error.get_or_insert(error);
            0
        }
        Err(_) => {
            session.error.get_or_insert(CodecError::Malformed {
                at_bit: 0,
                what: "nvdec callback failed",
            });
            0
        }
    }
}

unsafe extern "C" fn on_sequence(user: *mut c_void, format: *mut VideoFormat) -> c_int {
    callback(user, |session| {
        // SAFETY: the parser passes a valid format for this call.
        let Some(format) = (unsafe { format.as_ref() }) else {
            return Ok(0);
        };
        let surfaces = session.sequence(format)?;
        Ok(c_int::try_from(surfaces).unwrap_or(c_int::MAX))
    })
}

unsafe extern "C" fn on_decode(user: *mut c_void, params: *mut PicParams) -> c_int {
    callback(user, |session| {
        session.decode(params)?;
        Ok(1)
    })
}

unsafe extern "C" fn on_display(user: *mut c_void, display: *mut DisplayInfo) -> c_int {
    callback(user, |session| {
        // SAFETY: the parser passes a valid record, or null at the end.
        if let Some(display) = unsafe { display.as_ref() } {
            session.display(display)?;
        }
        Ok(1)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::offset_of;

    #[test]
    fn the_structs_have_the_c_layout() {
        // From cuviddec.h and nvcuvid.h (nv-codec-headers n12.0.16.1) on LP64.
        assert_eq!(size_of::<DecodeCaps>(), 88);
        assert_eq!(offset_of!(DecodeCaps, is_supported), 24);
        assert_eq!(offset_of!(DecodeCaps, output_format_mask), 26);
        assert_eq!(offset_of!(DecodeCaps, max_width), 28);
        assert_eq!(offset_of!(DecodeCaps, min_width), 40);
        assert_eq!(offset_of!(DecodeCaps, reserved3), 48);

        assert_eq!(size_of::<CreateInfo>(), 176);
        assert_eq!(offset_of!(CreateInfo, codec), 24);
        assert_eq!(offset_of!(CreateInfo, creation_flags), 32);
        assert_eq!(offset_of!(CreateInfo, display_area), 80);
        assert_eq!(offset_of!(CreateInfo, output_format), 88);
        assert_eq!(offset_of!(CreateInfo, target_width), 96);
        assert_eq!(offset_of!(CreateInfo, vid_lock), 120);
        assert_eq!(offset_of!(CreateInfo, target_rect), 128);
        assert_eq!(offset_of!(CreateInfo, enable_histogram), 136);
        assert_eq!(offset_of!(CreateInfo, reserved2), 144);

        assert_eq!(size_of::<ProcParams>(), 264);
        assert_eq!(offset_of!(ProcParams, raw_input_dptr), 24);
        assert_eq!(offset_of!(ProcParams, output_stream), 56);
        assert_eq!(offset_of!(ProcParams, histogram_dptr), 248);
        assert_eq!(offset_of!(ProcParams, reserved2), 256);

        assert_eq!(size_of::<VideoFormat>(), 64);
        assert_eq!(offset_of!(VideoFormat, frame_rate), 4);
        assert_eq!(offset_of!(VideoFormat, progressive_sequence), 12);
        assert_eq!(offset_of!(VideoFormat, min_num_decode_surfaces), 15);
        assert_eq!(offset_of!(VideoFormat, coded_width), 16);
        assert_eq!(offset_of!(VideoFormat, display_area), 24);
        assert_eq!(offset_of!(VideoFormat, chroma_format), 40);
        assert_eq!(offset_of!(VideoFormat, display_aspect_ratio), 48);
        assert_eq!(offset_of!(VideoFormat, signal_flags), 56);
        assert_eq!(offset_of!(VideoFormat, seqhdr_data_length), 60);

        assert_eq!(size_of::<SourcePacket>(), 32);
        assert_eq!(offset_of!(SourcePacket, payload_size), 8);
        assert_eq!(offset_of!(SourcePacket, payload), 16);
        assert_eq!(offset_of!(SourcePacket, timestamp), 24);

        assert_eq!(size_of::<DisplayInfo>(), 24);
        assert_eq!(offset_of!(DisplayInfo, repeat_first_field), 12);
        assert_eq!(offset_of!(DisplayInfo, timestamp), 16);

        assert_eq!(size_of::<ParserParams>(), 136);
        assert_eq!(offset_of!(ParserParams, max_display_delay), 16);
        assert_eq!(offset_of!(ParserParams, reserved1), 24);
        assert_eq!(offset_of!(ParserParams, user_data), 40);
        assert_eq!(offset_of!(ParserParams, sequence), 48);
        assert_eq!(offset_of!(ParserParams, sei), 80);
        assert_eq!(offset_of!(ParserParams, reserved2), 88);
        assert_eq!(offset_of!(ParserParams, ext_video_info), 128);
    }

    #[test]
    fn the_codec_ids_match_cuda_video_codec() {
        assert_eq!(codec_of(Kind::H264), 4);
        assert_eq!(codec_of(Kind::Hevc), 8);
        assert_eq!(codec_of(Kind::Av1), 11);
    }
}
