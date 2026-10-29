//! VideoToolbox decode and encode.
//!
//! Pictures that cross the crate boundary are tightly packed RGBA.
//! VideoToolbox itself is asked for BGRA, and the conversion stays here.
//! A session the hardware will not open is [`CodecError::Unsupported`], so
//! the factory can try a software codec. VP8 has no framework format id.

use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use cutlass_decode::{
    CodecError, CodecId, CodecParams, ColorPrimaries, Decoder, Output, Packet, PictureInfo,
    PictureOutput, PixelFormat, Transfer,
};
use cutlass_encode::{
    AudioInput, CpuFrame, Encoder, FrameInput, FrameTime, PictureInput, PlatformFrame,
};

use super::cf::{self, Cf};
use super::sys::{self, CMSampleTimingInfo};
use crate::limits::{self, even_picture};
use crate::nal::{self, AvcConfig, Nal};
use crate::status::{open_status, unit_status};

const H264: u32 = u32::from_be_bytes(*b"avc1");
const HEVC: u32 = u32::from_be_bytes(*b"hvc1");
const AV1: u32 = u32::from_be_bytes(*b"av01");
const VP9: u32 = u32::from_be_bytes(*b"vp09");
const MPEG4: u32 = u32::from_be_bytes(*b"mp4v");
const MPEG2: u32 = u32::from_be_bytes(*b"mp2v");
const MJPEG: u32 = u32::from_be_bytes(*b"jpeg");
const PRORES: u32 = u32::from_be_bytes(*b"apcn");

struct Decompress {
    raw: *mut std::ffi::c_void,
}

impl Drop for Decompress {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                sys::VTDecompressionSessionInvalidate(self.raw);
                sys::CFRelease(self.raw);
            }
        }
    }
}

unsafe impl Send for Decompress {}

struct Queue {
    packets: Vec<Packet>,
    attached: bool,
}

struct FrameMeta {
    pts: i64,
    duration: i64,
    timescale: u32,
}

struct DecodeSlot {
    picture: Option<Result<PictureOutput, CodecError>>,
    pts: i64,
    timescale: u32,
}

pub struct VideoDecoder {
    codec: CodecId,
    width: u32,
    height: u32,
    extra: Vec<u8>,
    format: Option<Cf>,
    session: Option<Decompress>,
}

pub struct VideoEncoder {
    codec: CodecId,
    width: u32,
    height: u32,
    bitrate: i32,
    session: *mut std::ffi::c_void,
    output: Mutex<Queue>,
    drained: bool,
    cancel: Arc<AtomicBool>,
}

unsafe impl Send for VideoEncoder {}

pub fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    let _ = codec_type(params.codec)?;
    let (width, height) = video_size(params.width, params.height)?;
    if params.pixel == PixelFormat::Yuv420p {
        return Err(CodecError::Unsupported {
            what: "yuv picture",
        });
    }
    let mut decoder = VideoDecoder {
        codec: params.codec,
        width,
        height,
        extra: params.extra.clone(),
        format: None,
        session: None,
    };
    if !params.extra.is_empty() || needs_sized_format(params.codec) {
        decoder.open_format(&params.extra)?;
    }
    Ok(Box::new(decoder))
}

pub fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    let _ = codec_type(params.codec)?;
    let (width, height) = video_size(params.width, params.height)?;
    if params.pixel == PixelFormat::Yuv420p {
        return Err(CodecError::Unsupported {
            what: "yuv picture",
        });
    }
    let bitrate = target_bitrate(params);
    let encoder = Box::new(VideoEncoder {
        codec: params.codec,
        width,
        height,
        bitrate,
        session: ptr::null_mut(),
        output: Mutex::new(Queue {
            packets: Vec::new(),
            attached: false,
        }),
        drained: false,
        cancel: Arc::new(AtomicBool::new(false)),
    });
    let raw = Box::into_raw(encoder);
    match create_compress(params.codec, width, height, bitrate, raw.cast()) {
        Ok(session) => unsafe { (*raw).session = session },
        Err(err) => {
            drop(unsafe { Box::from_raw(raw) });
            return Err(err);
        }
    }
    let boxed: Box<dyn Encoder> = unsafe { Box::from_raw(raw) };
    Ok(boxed)
}

impl Drop for VideoEncoder {
    fn drop(&mut self) {
        self.release_session();
    }
}

impl Decoder for VideoDecoder {
    fn codec(&self) -> CodecId {
        self.codec
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        limits::packet_ok(packet.data.len())?;
        if self.session.is_none() {
            if let Some(extra) = config_of(packet) {
                self.open_format(extra)?;
            } else if let Some(extra) = in_band_config(self.codec, &packet.data)? {
                self.open_format(&extra)?;
            }
        }
        let Some(session) = self.session.as_ref() else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "parameter sets",
            });
        };
        let Some(format) = self.format.as_ref() else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "parameter sets",
            });
        };
        let access = access_unit(self.codec, &packet.data)?;
        if access.is_empty() {
            return Ok(Vec::new());
        }
        let picture = decode_access(
            session.raw,
            format,
            &access,
            packet.pts,
            if packet.timescale == 0 {
                30
            } else {
                packet.timescale
            },
        )?;
        Ok(picture.into_iter().map(Output::Picture).collect())
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.session = None;
        self.format = None;
        if !self.extra.is_empty() || needs_sized_format(self.codec) {
            let extra = self.extra.clone();
            let _ = self.open_format(&extra);
        }
    }
}

impl VideoDecoder {
    fn open_format(&mut self, extra: &[u8]) -> Result<(), CodecError> {
        if self.session.is_some() {
            return Ok(());
        }
        let format = if needs_sized_format(self.codec) {
            sized_format(self.codec, self.width, self.height)?
        } else {
            parameter_format(self.codec, extra)?
        };
        let session = create_decompress(&format)?;
        self.format = Some(format);
        self.session = Some(session);
        Ok(())
    }
}

impl Encoder for VideoEncoder {
    fn codec(&self) -> CodecId {
        self.codec
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
        if self.session.is_null() {
            return Err(CodecError::Unsupported {
                what: "videotoolbox",
            });
        }
        let (width, height, bytes, time) = frame_rgba(input, self.width, self.height)?;
        let mut meta = FrameMeta {
            pts: time.pts(),
            duration: time.duration(),
            timescale: time.timescale(),
        };
        let buffer = bgra_buffer(width, height, bytes)?;
        let force = {
            let queue = lock_queue(&self.output);
            !queue.attached && queue.packets.is_empty()
        };
        let properties = if force { Some(force_keyframe()?) } else { None };
        let status = unsafe {
            sys::VTCompressionSessionEncodeFrame(
                self.session,
                buffer.as_mut(),
                cf::time(meta.pts, i32::try_from(meta.timescale).unwrap_or(30)),
                cf::time(meta.duration, i32::try_from(meta.timescale).unwrap_or(30)),
                properties.as_ref().map(Cf::as_ptr).unwrap_or(ptr::null()),
                (&raw mut meta).cast(),
                ptr::null_mut(),
            )
        };
        if status != 0 {
            clear_queue(&self.output);
            return unit_status(status, "videotoolbox").map(|()| Vec::new());
        }
        let flushed =
            unsafe { sys::VTCompressionSessionCompleteFrames(self.session, cf::time_invalid()) };
        if flushed != 0 {
            clear_queue(&self.output);
            return unit_status(flushed, "videotoolbox").map(|()| Vec::new());
        }
        Ok(take_packets(&self.output))
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        if self.drained || self.session.is_null() {
            self.drained = true;
            return Ok(Vec::new());
        }
        let status =
            unsafe { sys::VTCompressionSessionCompleteFrames(self.session, cf::time_invalid()) };
        self.drained = true;
        if status != 0 {
            clear_queue(&self.output);
            return unit_status(status, "videotoolbox").map(|()| Vec::new());
        }
        Ok(take_packets(&self.output))
    }

    fn reset(&mut self) {
        self.release_session();
        self.drained = false;
        clear_queue(&self.output);
        let raw: *mut VideoEncoder = self;
        if let Ok(session) = create_compress(
            self.codec,
            self.width,
            self.height,
            self.bitrate,
            raw.cast(),
        ) {
            self.session = session;
        }
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

impl VideoEncoder {
    fn release_session(&mut self) {
        if !self.session.is_null() {
            unsafe {
                sys::VTCompressionSessionInvalidate(self.session);
                sys::CFRelease(self.session);
            }
            self.session = ptr::null_mut();
        }
    }
}

fn video_size(width: u32, height: u32) -> Result<(u32, u32), CodecError> {
    match even_picture(width, height) {
        Ok(()) => Ok((width, height)),
        Err(CodecError::Limit { .. }) => Err(CodecError::Unsupported {
            what: "picture size",
        }),
        Err(err) => Err(err),
    }
}

fn needs_sized_format(codec: CodecId) -> bool {
    !matches!(codec, CodecId::H264 | CodecId::Hevc)
}

fn codec_type(codec: CodecId) -> Result<u32, CodecError> {
    match codec {
        CodecId::H264 => Ok(H264),
        CodecId::Hevc => Ok(HEVC),
        CodecId::Av1 => Ok(AV1),
        CodecId::Vp9 => Ok(VP9),
        CodecId::Mpeg4 => Ok(MPEG4),
        CodecId::Mpeg2 => Ok(MPEG2),
        CodecId::Mjpeg => Ok(MJPEG),
        CodecId::Prores => Ok(PRORES),
        CodecId::Vp8 => Err(CodecError::Unsupported { what: "vp8" }),
        _ => Err(CodecError::Unsupported {
            what: "videotoolbox",
        }),
    }
}

fn target_bitrate(params: &CodecParams) -> i32 {
    if params.bitrate != 0 {
        return i32::try_from(params.bitrate).unwrap_or(i32::MAX);
    }
    let pixels = u64::from(params.width).saturating_mul(u64::from(params.height));
    let guess = pixels.saturating_mul(8).max(64_000);
    i32::try_from(guess).unwrap_or(i32::MAX)
}

fn config_of(packet: &Packet) -> Option<&[u8]> {
    packet.side.iter().find_map(|side| match side {
        cutlass_decode::SideData::Config(bytes) if !bytes.is_empty() => Some(bytes.as_slice()),
        _ => None,
    })
}

fn in_band_config(codec: CodecId, data: &[u8]) -> Result<Option<Vec<u8>>, CodecError> {
    let nals = match nal::annex_b(data) {
        Ok(nals) => nals,
        Err(_) => return Ok(None),
    };
    match codec {
        CodecId::H264 => {
            let sps: Vec<Vec<u8>> = nals
                .iter()
                .filter(|nal| nal::h264_type(&nal.bytes) == Some(7))
                .map(|nal| nal.bytes.clone())
                .collect();
            let pps: Vec<Vec<u8>> = nals
                .iter()
                .filter(|nal| nal::h264_type(&nal.bytes) == Some(8))
                .map(|nal| nal.bytes.clone())
                .collect();
            if sps.is_empty() || pps.is_empty() {
                return Ok(None);
            }
            let profile = sps[0].get(1).copied().unwrap_or(0x42);
            let compatibility = sps[0].get(2).copied().unwrap_or(0);
            let level = sps[0].get(3).copied().unwrap_or(0x1f);
            Ok(Some(nal::write_avc_config(&AvcConfig {
                length_size: 4,
                profile,
                compatibility,
                level,
                sps,
                pps,
            })?))
        }
        CodecId::Hevc => {
            let sets: Vec<Vec<u8>> = nals
                .iter()
                .filter(|nal| matches!(nal::hevc_type(&nal.bytes), Some(32..=34)))
                .map(|nal| nal.bytes.clone())
                .collect();
            if sets.is_empty() {
                return Ok(None);
            }
            Ok(Some(nal::write_hevc_envelope(&sets)?))
        }
        _ => Ok(None),
    }
}

fn access_unit(codec: CodecId, data: &[u8]) -> Result<Vec<u8>, CodecError> {
    if let Ok(nals) = nal::annex_b(data) {
        let vcl: Vec<Nal> = nals
            .into_iter()
            .filter(|nal| !is_parameter_set(codec, &nal.bytes))
            .collect();
        if vcl.is_empty() {
            return Ok(Vec::new());
        }
        return nal::annex_b_to_length(&nal::to_annex_b(&vcl));
    }
    let _ = nal::split_lengths(data, 4)?;
    Ok(data.to_vec())
}

fn is_parameter_set(codec: CodecId, nal: &[u8]) -> bool {
    match codec {
        CodecId::H264 => matches!(nal::h264_type(nal), Some(7..=9)),
        CodecId::Hevc => matches!(nal::hevc_type(nal), Some(32..=34)),
        _ => false,
    }
}

fn parameter_format(codec: CodecId, extra: &[u8]) -> Result<Cf, CodecError> {
    match codec {
        CodecId::H264 => {
            let config =
                nal::avc_config(extra).map_err(|_| CodecError::Unsupported { what: "avcC" })?;
            h264_format(&config)
        }
        CodecId::Hevc => {
            let config =
                nal::read_hevc_sets(extra).map_err(|_| CodecError::Unsupported { what: "hvcC" })?;
            hevc_format(&config.nals)
        }
        _ => Err(CodecError::Unsupported {
            what: "videotoolbox",
        }),
    }
}

fn h264_format(config: &AvcConfig) -> Result<Cf, CodecError> {
    let sets: Vec<&[u8]> = config
        .sps
        .iter()
        .chain(config.pps.iter())
        .map(Vec::as_slice)
        .collect();
    parameter_sets(sets.as_slice(), true)
}

fn hevc_format(nals: &[Vec<u8>]) -> Result<Cf, CodecError> {
    let sets: Vec<&[u8]> = nals.iter().map(Vec::as_slice).collect();
    parameter_sets(sets.as_slice(), false)
}

fn parameter_sets(nals: &[&[u8]], h264: bool) -> Result<Cf, CodecError> {
    let pointers: Vec<*const u8> = nals.iter().map(|nal| nal.as_ptr()).collect();
    let sizes: Vec<usize> = nals.iter().map(|nal| nal.len()).collect();
    let mut out = ptr::null_mut();
    let status = unsafe {
        if h264 {
            sys::CMVideoFormatDescriptionCreateFromH264ParameterSets(
                ptr::null(),
                pointers.len(),
                pointers.as_ptr(),
                sizes.as_ptr(),
                4,
                &mut out,
            )
        } else {
            sys::CMVideoFormatDescriptionCreateFromHEVCParameterSets(
                ptr::null(),
                pointers.len(),
                pointers.as_ptr(),
                sizes.as_ptr(),
                4,
                ptr::null(),
                &mut out,
            )
        }
    };
    open_status(status, "videotoolbox")?;
    Cf::from_owned(out).ok_or(CodecError::Unsupported {
        what: "videotoolbox",
    })
}

fn sized_format(codec: CodecId, width: u32, height: u32) -> Result<Cf, CodecError> {
    let mut out = ptr::null_mut();
    let status = unsafe {
        sys::CMVideoFormatDescriptionCreate(
            ptr::null(),
            codec_type(codec)?,
            i32::try_from(width).map_err(|_| CodecError::Limit {
                what: "picture edge",
            })?,
            i32::try_from(height).map_err(|_| CodecError::Limit {
                what: "picture edge",
            })?,
            ptr::null(),
            &mut out,
        )
    };
    open_status(status, "videotoolbox")?;
    Cf::from_owned(out).ok_or(CodecError::Unsupported {
        what: "videotoolbox",
    })
}

fn create_decompress(format: &Cf) -> Result<Decompress, CodecError> {
    let attributes = bgra_attributes()?;
    let callback = sys::VTDecompressionOutputCallbackRecord {
        callback: decode_output,
        refcon: ptr::null_mut(),
    };
    let mut raw = ptr::null_mut();
    let status = unsafe {
        sys::VTDecompressionSessionCreate(
            ptr::null(),
            format.as_ptr(),
            ptr::null(),
            attributes.as_ptr(),
            &callback,
            &mut raw,
        )
    };
    open_status(status, "videotoolbox")?;
    if raw.is_null() {
        return Err(CodecError::Unsupported {
            what: "videotoolbox",
        });
    }
    Ok(Decompress { raw })
}

fn decode_access(
    session: *mut std::ffi::c_void,
    format: &Cf,
    bytes: &[u8],
    pts: i64,
    timescale: u32,
) -> Result<Option<PictureOutput>, CodecError> {
    let block = block_from(bytes)?;
    let scale = i32::try_from(timescale).unwrap_or(30);
    let timing = CMSampleTimingInfo {
        duration: cf::time(1, scale),
        presentation: cf::time(pts, scale),
        decode: cf::time(pts, scale),
    };
    let size = bytes.len();
    let mut sample = ptr::null_mut();
    let status = unsafe {
        sys::CMSampleBufferCreateReady(
            ptr::null(),
            block.as_ptr(),
            format.as_ptr(),
            1,
            1,
            &timing,
            1,
            &size,
            &mut sample,
        )
    };
    unit_status(status, "videotoolbox")?;
    let sample = Cf::from_owned(sample).ok_or(CodecError::Malformed {
        at_bit: 0,
        what: "sample buffer",
    })?;
    let mut slot = DecodeSlot {
        picture: None,
        pts,
        timescale,
    };
    let status = unsafe {
        sys::VTDecompressionSessionDecodeFrame(
            session,
            sample.as_ptr(),
            0,
            (&raw mut slot).cast(),
            ptr::null_mut(),
        )
    };
    if status != 0 {
        return unit_status(status, "videotoolbox").map(|()| None);
    }
    match slot.picture {
        Some(result) => result.map(Some),
        None => Ok(None),
    }
}

fn block_from(bytes: &[u8]) -> Result<Cf, CodecError> {
    let mut raw = ptr::null_mut();
    let status = unsafe {
        sys::CMBlockBufferCreateWithMemoryBlock(
            ptr::null(),
            bytes.as_ptr().cast_mut().cast(),
            bytes.len(),
            sys::kCFAllocatorNull,
            ptr::null(),
            0,
            bytes.len(),
            0,
            &mut raw,
        )
    };
    unit_status(status, "videotoolbox")?;
    Cf::from_owned(raw).ok_or(CodecError::Malformed {
        at_bit: 0,
        what: "block buffer",
    })
}

unsafe extern "C" fn decode_output(
    _refcon: *mut std::ffi::c_void,
    source: *mut std::ffi::c_void,
    status: sys::OSStatus,
    _flags: u32,
    image: *mut std::ffi::c_void,
    _pts: sys::CMTime,
    _duration: sys::CMTime,
) {
    unsafe {
        if source.is_null() {
            return;
        }
        let slot = &mut *source.cast::<DecodeSlot>();
        if status != 0 || image.is_null() {
            slot.picture = Some(Err(CodecError::Malformed {
                at_bit: 0,
                what: "videotoolbox",
            }));
            return;
        }
        slot.picture = Some(rgba_picture(image, slot.pts, slot.timescale));
    }
}

fn rgba_picture(
    image: *mut std::ffi::c_void,
    pts: i64,
    timescale: u32,
) -> Result<PictureOutput, CodecError> {
    let locked = unsafe { sys::CVPixelBufferLockBaseAddress(image, 1) };
    if locked != 0 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "pixel buffer",
        });
    }
    let result = copy_rgba(image, pts, timescale);
    unsafe {
        let _ = sys::CVPixelBufferUnlockBaseAddress(image, 1);
    }
    result
}

fn copy_rgba(
    image: *mut std::ffi::c_void,
    pts: i64,
    timescale: u32,
) -> Result<PictureOutput, CodecError> {
    let format = unsafe { sys::CVPixelBufferGetPixelFormatType(image) };
    if format != sys::BGRA {
        return Err(CodecError::Unsupported {
            what: "pixel format",
        });
    }
    let width = u32::try_from(unsafe { sys::CVPixelBufferGetWidth(image) }).map_err(|_| {
        CodecError::Limit {
            what: "picture edge",
        }
    })?;
    let height = u32::try_from(unsafe { sys::CVPixelBufferGetHeight(image) }).map_err(|_| {
        CodecError::Limit {
            what: "picture edge",
        }
    })?;
    limits::picture_ok(width, height)?;
    let stride = unsafe { sys::CVPixelBufferGetBytesPerRow(image) };
    let base = unsafe { sys::CVPixelBufferGetBaseAddress(image) };
    if base.is_null() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "pixel buffer",
        });
    }
    let row = usize::try_from(width).unwrap_or(0).saturating_mul(4);
    if stride < row {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "pixel stride",
        });
    }
    let height_usize = usize::try_from(height).unwrap_or(0);
    let bytes = unsafe { std::slice::from_raw_parts(base.cast::<u8>(), stride * height_usize) };
    let mut rgba = vec![0u8; row * height_usize];
    for y in 0..height_usize {
        let src = &bytes[y * stride..y * stride + row];
        let dst = &mut rgba[y * row..y * row + row];
        let mut x = 0;
        while x + 4 <= src.len() {
            dst[x] = src[x + 2];
            dst[x + 1] = src[x + 1];
            dst[x + 2] = src[x];
            dst[x + 3] = src[x + 3];
            x += 4;
        }
    }
    PictureOutput::cpu(
        PictureInfo {
            width,
            height,
            pixel: PixelFormat::Rgba8,
            primaries: ColorPrimaries::Bt709,
            transfer: Transfer::Unspecified,
        },
        rgba,
        pts,
        timescale,
    )
}

fn create_compress(
    codec: CodecId,
    width: u32,
    height: u32,
    bitrate: i32,
    refcon: *mut std::ffi::c_void,
) -> Result<*mut std::ffi::c_void, CodecError> {
    let attributes = bgra_attributes()?;
    let mut raw = ptr::null_mut();
    let status = unsafe {
        sys::VTCompressionSessionCreate(
            ptr::null(),
            i32::try_from(width).map_err(|_| CodecError::Limit {
                what: "picture edge",
            })?,
            i32::try_from(height).map_err(|_| CodecError::Limit {
                what: "picture edge",
            })?,
            codec_type(codec)?,
            ptr::null(),
            attributes.as_ptr(),
            ptr::null(),
            encode_output,
            refcon,
            &mut raw,
        )
    };
    open_status(status, "videotoolbox")?;
    if raw.is_null() {
        return Err(CodecError::Unsupported {
            what: "videotoolbox",
        });
    }
    apply_properties(raw, codec, bitrate);
    Ok(raw)
}

fn apply_properties(session: *mut std::ffi::c_void, codec: CodecId, bitrate: i32) {
    let Ok(rate) = cf::number_i32(bitrate) else {
        return;
    };
    let Ok(interval) = cf::number_i32(30) else {
        return;
    };
    let Ok(fps) = cf::number_i32(30) else {
        return;
    };
    unsafe {
        let _ = sys::VTSessionSetProperty(
            session,
            sys::kVTCompressionPropertyKey_AverageBitRate,
            rate.as_ptr(),
        );
        let _ = sys::VTSessionSetProperty(
            session,
            sys::kVTCompressionPropertyKey_AllowFrameReordering,
            cf::boolean(false),
        );
        let _ = sys::VTSessionSetProperty(
            session,
            sys::kVTCompressionPropertyKey_MaxKeyFrameInterval,
            interval.as_ptr(),
        );
        let _ = sys::VTSessionSetProperty(
            session,
            sys::kVTCompressionPropertyKey_ExpectedFrameRate,
            fps.as_ptr(),
        );
        let _ = sys::VTSessionSetProperty(
            session,
            sys::kVTCompressionPropertyKey_RealTime,
            cf::boolean(true),
        );
        let profile = match codec {
            CodecId::H264 => Some(sys::kVTProfileLevel_H264_Baseline_AutoLevel),
            CodecId::Hevc => Some(sys::kVTProfileLevel_HEVC_Main_AutoLevel),
            _ => None,
        };
        if let Some(profile) = profile {
            let _ = sys::VTSessionSetProperty(
                session,
                sys::kVTCompressionPropertyKey_ProfileLevel,
                profile,
            );
        }
    }
}

fn force_keyframe() -> Result<Cf, CodecError> {
    let key = unsafe { sys::kVTEncodeFrameOptionKey_ForceKeyFrame };
    let value = cf::boolean(true);
    cf::dictionary(&[key], &[value])
}

fn bgra_attributes() -> Result<Cf, CodecError> {
    let format = cf::number_i32(i32::try_from(sys::BGRA).unwrap_or(0))?;
    let surface = cf::dictionary(&[], &[])?;
    let keys = unsafe {
        [
            sys::kCVPixelBufferPixelFormatTypeKey,
            sys::kCVPixelBufferIOSurfacePropertiesKey,
        ]
    };
    cf::dictionary(&[keys[0], keys[1]], &[format.as_ptr(), surface.as_ptr()])
}

fn bgra_buffer(width: u32, height: u32, rgba: &[u8]) -> Result<Cf, CodecError> {
    let attributes = bgra_attributes()?;
    let mut raw = ptr::null_mut();
    let status = unsafe {
        sys::CVPixelBufferCreate(
            ptr::null(),
            usize::try_from(width).map_err(|_| CodecError::Limit {
                what: "picture edge",
            })?,
            usize::try_from(height).map_err(|_| CodecError::Limit {
                what: "picture edge",
            })?,
            sys::BGRA,
            attributes.as_ptr(),
            &mut raw,
        )
    };
    unit_status(status, "videotoolbox")?;
    let buffer = Cf::from_owned(raw).ok_or(CodecError::Unsupported {
        what: "pixel buffer",
    })?;
    let locked = unsafe { sys::CVPixelBufferLockBaseAddress(buffer.as_mut(), 0) };
    if locked != 0 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "pixel buffer",
        });
    }
    let copied = write_bgra(buffer.as_mut(), rgba);
    unsafe {
        let _ = sys::CVPixelBufferUnlockBaseAddress(buffer.as_mut(), 0);
    }
    copied?;
    Ok(buffer)
}

fn write_bgra(buffer: *mut std::ffi::c_void, rgba: &[u8]) -> Result<(), CodecError> {
    let stride = unsafe { sys::CVPixelBufferGetBytesPerRow(buffer) };
    let height = unsafe { sys::CVPixelBufferGetHeight(buffer) };
    let width = unsafe { sys::CVPixelBufferGetWidth(buffer) };
    let base = unsafe { sys::CVPixelBufferGetBaseAddress(buffer) };
    if base.is_null() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "pixel buffer",
        });
    }
    let row = width.saturating_mul(4);
    if stride < row || rgba.len() < row.saturating_mul(height) {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "pixel stride",
        });
    }
    let dest =
        unsafe { std::slice::from_raw_parts_mut(base.cast::<u8>(), stride.saturating_mul(height)) };
    for y in 0..height {
        let src = &rgba[y * row..y * row + row];
        let dst = &mut dest[y * stride..y * stride + row];
        let mut x = 0;
        while x + 4 <= src.len() {
            dst[x] = src[x + 2];
            dst[x + 1] = src[x + 1];
            dst[x + 2] = src[x];
            dst[x + 3] = src[x + 3];
            x += 4;
        }
    }
    Ok(())
}

fn frame_rgba(
    input: &FrameInput,
    width: u32,
    height: u32,
) -> Result<(u32, u32, &[u8], FrameTime), CodecError> {
    match input {
        FrameInput::Platform(PlatformFrame { .. }) => Err(CodecError::Unsupported {
            what: "platform frame",
        }),
        FrameInput::Cpu(frame) => cpu_rgba(frame, width, height),
    }
}

fn cpu_rgba(
    frame: &CpuFrame,
    width: u32,
    height: u32,
) -> Result<(u32, u32, &[u8], FrameTime), CodecError> {
    if frame.pixel() != PixelFormat::Rgba8 || frame.width() != width || frame.height() != height {
        return Err(CodecError::Unsupported { what: "frame" });
    }
    Ok((frame.width(), frame.height(), frame.bytes(), frame.time()))
}

unsafe extern "C" fn encode_output(
    refcon: *mut std::ffi::c_void,
    source: *mut std::ffi::c_void,
    status: sys::OSStatus,
    _flags: u32,
    sample: *mut std::ffi::c_void,
) {
    unsafe {
        if refcon.is_null() || status != 0 || sample.is_null() {
            return;
        }
        let encoder = &*refcon.cast::<VideoEncoder>();
        let meta = if source.is_null() {
            FrameMeta {
                pts: 0,
                duration: 1,
                timescale: 30,
            }
        } else {
            source.cast::<FrameMeta>().read()
        };
        let Some(bytes) = sample_bytes(sample) else {
            return;
        };
        let keyframe = sample_is_sync(sample);
        let mut packet = Packet {
            stream: 0,
            data: bytes,
            pts: meta.pts,
            duration: meta.duration,
            timescale: meta.timescale,
            keyframe,
            side: Vec::new(),
        };
        let mut queue = lock_queue(&encoder.output);
        if keyframe
            && !queue.attached
            && let Some(config) = sample_config(encoder.codec, sample)
        {
            packet.side.push(cutlass_decode::SideData::Config(config));
            queue.attached = true;
        }
        queue.packets.push(packet);
    }
}

fn sample_bytes(sample: *mut std::ffi::c_void) -> Option<Vec<u8>> {
    unsafe {
        let block = sys::CMSampleBufferGetDataBuffer(sample);
        if block.is_null() {
            return None;
        }
        let mut data = ptr::null_mut();
        let mut total = 0usize;
        let status =
            sys::CMBlockBufferGetDataPointer(block, 0, ptr::null_mut(), &mut total, &mut data);
        if status != 0 || data.is_null() || total == 0 {
            return None;
        }
        Some(std::slice::from_raw_parts(data, total).to_vec())
    }
}

fn sample_is_sync(sample: *mut std::ffi::c_void) -> bool {
    unsafe {
        let array = sys::CMSampleBufferGetSampleAttachmentsArray(sample, 0);
        if array.is_null() {
            return true;
        }
        if sys::CFArrayGetCount(array) <= 0 {
            return true;
        }
        let dict = sys::CFArrayGetValueAtIndex(array, 0);
        if dict.is_null() {
            return true;
        }
        let value = sys::CFDictionaryGetValue(dict, sys::kCMSampleAttachmentKey_NotSync);
        value.is_null() || value != sys::kCFBooleanTrue
    }
}

fn sample_config(codec: CodecId, sample: *mut std::ffi::c_void) -> Option<Vec<u8>> {
    let format = unsafe { sys::CMSampleBufferGetFormatDescription(sample) };
    if format.is_null() {
        return None;
    }
    match codec {
        CodecId::H264 => h264_config(format).ok(),
        CodecId::Hevc => hevc_config(format).ok(),
        _ => None,
    }
}

fn h264_config(format: *const std::ffi::c_void) -> Result<Vec<u8>, CodecError> {
    let mut sps = Vec::new();
    let mut pps = Vec::new();
    for index in 0..8 {
        let Some(bytes) = parameter_set(format, index, true) else {
            break;
        };
        match nal::h264_type(&bytes) {
            Some(7) => sps.push(bytes),
            Some(8) => pps.push(bytes),
            _ => {}
        }
    }
    if sps.is_empty() || pps.is_empty() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "avcC sets",
        });
    }
    nal::write_avc_config(&AvcConfig {
        length_size: 4,
        profile: sps[0].get(1).copied().unwrap_or(0x42),
        compatibility: sps[0].get(2).copied().unwrap_or(0),
        level: sps[0].get(3).copied().unwrap_or(0x1f),
        sps,
        pps,
    })
}

fn hevc_config(format: *const std::ffi::c_void) -> Result<Vec<u8>, CodecError> {
    let mut nals = Vec::new();
    for index in 0..8 {
        let Some(bytes) = parameter_set(format, index, false) else {
            break;
        };
        nals.push(bytes);
    }
    nal::write_hevc_envelope(&nals)
}

fn parameter_set(format: *const std::ffi::c_void, index: usize, h264: bool) -> Option<Vec<u8>> {
    let mut pointer = ptr::null();
    let mut size = 0usize;
    let mut count = 0usize;
    let mut nal_length = 0i32;
    let status = unsafe {
        if h264 {
            sys::CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
                format,
                index,
                &mut pointer,
                &mut size,
                &mut count,
                &mut nal_length,
            )
        } else {
            sys::CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
                format,
                index,
                &mut pointer,
                &mut size,
                &mut count,
                &mut nal_length,
            )
        }
    };
    if status != 0 || pointer.is_null() || size == 0 {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts(pointer, size) }.to_vec())
}

fn lock_queue(queue: &Mutex<Queue>) -> std::sync::MutexGuard<'_, Queue> {
    match queue.lock() {
        Ok(guard) => guard,
        Err(poison) => poison.into_inner(),
    }
}

fn clear_queue(queue: &Mutex<Queue>) {
    lock_queue(queue).packets.clear();
}

fn take_packets(queue: &Mutex<Queue>) -> Vec<Packet> {
    std::mem::take(&mut lock_queue(queue).packets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_decode::{PictureSamples, SideData};
    use cutlass_encode::{CpuFrame, FrameTime, PlatformFrame};

    fn video_params(codec: CodecId, width: u32, height: u32) -> CodecParams {
        CodecParams {
            codec,
            extra: Vec::new(),
            sample_rate: 0,
            layout: None,
            sample_format: None,
            bitrate: 500_000,
            width,
            height,
            pixel: PixelFormat::Rgba8,
            primaries: ColorPrimaries::Bt709,
            transfer: Transfer::Unspecified,
        }
    }

    fn solid(width: u32, height: u32, pixel: [u8; 4], pts: i64) -> FrameInput {
        let mut bytes = Vec::new();
        for _ in 0..width * height {
            bytes.extend(pixel);
        }
        FrameInput::Cpu(
            CpuFrame::new(
                width,
                height,
                PixelFormat::Rgba8,
                bytes,
                FrameTime::new(pts, 1, 30).expect("time"),
            )
            .expect("frame"),
        )
    }

    fn channel_sum(bytes: &[u8], channel: usize) -> u64 {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|pixel| u64::from(pixel[channel]))
            .sum()
    }

    #[test]
    fn h264_round_trips_a_solid_color() {
        let mut encoder = open_encoder(&video_params(CodecId::H264, 64, 64)).expect("encode open");
        let platform =
            PlatformFrame::new(64, 64, 1, FrameTime::new(0, 1, 30).expect("time")).expect("token");
        assert_eq!(
            encoder.push_frame(&FrameInput::Platform(platform)),
            Err(CodecError::Unsupported {
                what: "platform frame",
            })
        );
        let mut packets = encoder
            .push_frame(&solid(64, 64, [220, 10, 10, 255], 0))
            .expect("encode");
        packets.extend(
            encoder
                .push_frame(&solid(64, 64, [10, 200, 10, 255], 1))
                .expect("second"),
        );
        packets.extend(encoder.drain().expect("drain"));
        assert!(encoder.drain().expect("second drain").is_empty());
        assert!(!packets.is_empty(), "h264 produced no packets");
        assert!(packets[0].keyframe);
        let config = packets
            .iter()
            .find_map(|packet| {
                packet.side.iter().find_map(|side| match side {
                    SideData::Config(bytes) if !bytes.is_empty() => Some(bytes.clone()),
                    _ => None,
                })
            })
            .expect("avcC");
        let mut decode_params = video_params(CodecId::H264, 64, 64);
        decode_params.extra = config;
        let mut decoder = open_decoder(&decode_params).expect("decode open");
        let bad = decoder.push(&Packet {
            stream: 0,
            data: vec![0, 0, 0, 1, 0xff],
            pts: 0,
            duration: 1,
            timescale: 30,
            keyframe: false,
            side: Vec::new(),
        });
        assert!(bad.is_err(), "a broken access unit must fail");
        let mut pictures = Vec::new();
        for packet in &packets {
            for output in decoder.push(packet).expect("decode") {
                let Output::Picture(picture) = output else {
                    panic!("picture");
                };
                pictures.push(picture);
            }
        }
        assert!(
            !pictures.is_empty(),
            "no pictures from {} packets",
            packets.len()
        );
        let PictureSamples::Cpu(bytes) = pictures[0].samples() else {
            panic!("cpu");
        };
        let red = channel_sum(bytes, 0);
        let green = channel_sum(bytes, 1);
        let blue = channel_sum(bytes, 2);
        assert!(
            red > green.saturating_mul(4) && red > blue.saturating_mul(4),
            "red {red} green {green} blue {blue}"
        );
    }

    #[test]
    fn vp8_is_declined() {
        let err = match open_encoder(&video_params(CodecId::Vp8, 64, 64)) {
            Err(err) => err,
            Ok(_) => panic!("vp8"),
        };
        assert_eq!(err, CodecError::Unsupported { what: "vp8" });
    }

    #[test]
    fn an_odd_h264_size_is_declined() {
        let err = match open_encoder(&video_params(CodecId::H264, 63, 64)) {
            Err(err) => err,
            Ok(_) => panic!("odd"),
        };
        assert_eq!(
            err,
            CodecError::Unsupported {
                what: "odd video size"
            }
        );
    }

    #[test]
    fn an_empty_h264_packet_is_an_invalid_call() {
        let mut decoder = open_decoder(&video_params(CodecId::H264, 64, 64)).expect("open");
        let err = decoder.push(&Packet {
            stream: 0,
            data: Vec::new(),
            pts: 0,
            duration: 1,
            timescale: 30,
            keyframe: true,
            side: Vec::new(),
        });
        assert_eq!(
            err,
            Err(CodecError::InvalidApi {
                what: "empty packet"
            })
        );
    }
}
