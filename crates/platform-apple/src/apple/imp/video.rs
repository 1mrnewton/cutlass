//! VideoToolbox decode and encode.
//!
//! A decoded frame stays a retained pixel buffer, named by a platform token.
//! [`to_cpu`] copies that buffer to tightly packed RGBA. VideoToolbox itself
//! is asked for BGRA. A caller that asks for [`PixelFormat::Yuv420p`] gets
//! CPU 4:2:0 instead: VideoToolbox writes video-range NV12 and the planes are
//! copied out as packed Y, then U, then V.
//!
//! Decode is asynchronous. A push submits its unit and returns the pictures
//! that have finished, so the next unit is read while the hardware works.
//! VideoToolbox finishes pictures in decode order. They are held for the
//! stream's reorder depth and leave in display order, each with its own
//! timestamp. A unit the
//! hardware rejects later is an error on the push that collects it. `drain`
//! waits for every picture still held. A session the hardware will not open is
//! [`CodecError::Unsupported`], so the factory can try a software codec.
//! A session the system invalidates, as iOS does when the app leaves the
//! foreground, is opened again on the same format by the next push or reset.
//! VP8 has no framework format id.

use std::collections::{HashMap, VecDeque};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cutlass_decode::{
    CodecError, CodecId, CodecParams, ColorPrimaries, Decoder, Output, Packet, PictureInfo,
    PictureOutput, PictureSamples, PixelFormat, Transfer,
};
use cutlass_encode::{AudioInput, CpuFrame, Encoder, FrameInput, FrameTime, PictureInput};

use super::cf::{self, Cf};
use super::sys::{self, CMSampleTimingInfo};
use crate::limits::{self, even_picture};
use crate::nal::{self, AvcConfig, Nal};
use crate::reorder::{self, Reorder};
use crate::status::{open_status, session_lost, unit_status};

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

impl Decompress {
    /// Release held pictures and wait until every callback has run.
    fn flush(&self) -> sys::OSStatus {
        unsafe {
            let finished = sys::VTDecompressionSessionFinishDelayedFrames(self.raw);
            let waited = sys::VTDecompressionSessionWaitForAsynchronousFrames(self.raw);
            if finished != 0 { finished } else { waited }
        }
    }
}

impl Drop for Decompress {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = self.flush();
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

/// What the output callback writes into. The session's refcon points here.
struct Finished {
    planar: bool,
    /// The session reported it can decode nothing more.
    lost: AtomicBool,
    pictures: Mutex<VecDeque<Result<PictureOutput, CodecError>>>,
}

pub struct VideoDecoder {
    codec: CodecId,
    width: u32,
    height: u32,
    extra: Vec<u8>,
    format: Option<Cf>,
    /// Dropped before `finished`, so no callback outlives it.
    session: Option<Decompress>,
    finished: Box<Finished>,
    /// Finished pictures waiting for display order.
    reorder: Reorder<PictureOutput>,
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
    Ok(Box::new(video_decoder(params)?))
}

fn video_decoder(params: &CodecParams) -> Result<VideoDecoder, CodecError> {
    let _ = codec_type(params.codec)?;
    let (width, height) = video_size(params.width, params.height)?;
    let mut decoder = VideoDecoder {
        codec: params.codec,
        width,
        height,
        extra: params.extra.clone(),
        format: None,
        session: None,
        finished: Box::new(Finished {
            planar: params.pixel == PixelFormat::Yuv420p,
            lost: AtomicBool::new(false),
            pictures: Mutex::new(VecDeque::new()),
        }),
        reorder: Reorder::new(reorder::MAX_DEPTH),
    };
    if !params.extra.is_empty() || needs_sized_format(params.codec) {
        decoder.open_format(&params.extra)?;
    }
    Ok(decoder)
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
        if self.finished.lost.swap(false, Ordering::AcqRel) {
            self.reopen()?;
        }
        if self.session.is_none() {
            if let Some(extra) = config_of(packet) {
                self.open_format(extra)?;
            } else if let Some(extra) = in_band_config(self.codec, &packet.data)? {
                self.open_format(&extra)?;
            }
        }
        let access = access_unit(self.codec, &packet.data)?;
        let mut status = self.submit(&access, packet)?;
        if session_lost(status) {
            self.reopen()?;
            status = self.submit(&access, packet)?;
        }
        unit_status(status, "videotoolbox")?;
        self.collect()
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        if let Some(session) = self.session.as_ref() {
            unit_status(session.flush(), "videotoolbox")?;
        }
        let mut outputs = self.collect()?;
        let mut held = Vec::new();
        self.reorder.flush(&mut held);
        outputs.extend(held.into_iter().map(Output::Picture));
        Ok(outputs)
    }

    fn reset(&mut self) {
        if self.finished.lost.swap(false, Ordering::AcqRel) {
            let _ = self.reopen();
            return;
        }
        if let Some(session) = self.session.as_ref() {
            let _ = session.flush();
            self.drop_in_flight();
            return;
        }
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
        let session = create_decompress(&format, self.surface(), self.refcon())?;
        self.reorder = Reorder::new(reorder_depth(self.codec, extra));
        self.format = Some(format);
        self.session = Some(session);
        Ok(())
    }

    /// Replace a session the system invalidated with a new one on the same
    /// format. The pictures in flight go with the old one. A session that
    /// will not open is tried again on the next push.
    fn reopen(&mut self) -> Result<(), CodecError> {
        // Dropping the old session waits for its callbacks first.
        self.session = None;
        self.drop_in_flight();
        let Some(format) = self.format.as_ref() else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "parameter sets",
            });
        };
        match create_decompress(format, self.surface(), self.refcon()) {
            Ok(session) => {
                self.session = Some(session);
                Ok(())
            }
            Err(error) => {
                self.finished.lost.store(true, Ordering::Release);
                Err(error)
            }
        }
    }

    /// Hand `access` to the session. The status is the session's own, so a
    /// lost session can be told apart from a bad unit.
    fn submit(&self, access: &[u8], packet: &Packet) -> Result<sys::OSStatus, CodecError> {
        let (Some(session), Some(format)) = (self.session.as_ref(), self.format.as_ref()) else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "parameter sets",
            });
        };
        if access.is_empty() {
            return Ok(0);
        }
        let timescale = if packet.timescale == 0 {
            30
        } else {
            packet.timescale
        };
        decode_access(session.raw, format, access, packet.pts, timescale)
    }

    /// Drop the finished pictures and the ones held for display order.
    fn drop_in_flight(&mut self) {
        let finished: Vec<PictureOutput> = lock_pictures(&self.finished)
            .drain(..)
            .filter_map(Result::ok)
            .collect();
        for picture in finished.iter().chain(self.reorder.clear().iter()) {
            if let PictureSamples::Platform { token } = picture.samples() {
                let _ = release_buffer(token);
            }
        }
    }

    /// The pixel format VideoToolbox writes.
    fn surface(&self) -> Option<u32> {
        if self.finished.planar {
            Some(sys::NV12)
        } else if needs_sized_format(self.codec) {
            None
        } else {
            Some(sys::BGRA)
        }
    }

    /// The session's refcon: the [`Finished`] its callback writes into.
    fn refcon(&self) -> *mut std::ffi::c_void {
        let refcon: *const Finished = &raw const *self.finished;
        refcon.cast_mut().cast()
    }
}

impl VideoDecoder {
    /// Move finished pictures into the reorder buffer and return the ones it
    /// releases, in display order. An error stops the move. The pictures
    /// released before it are returned now and the error on the next call.
    fn collect(&mut self) -> Result<Vec<Output>, CodecError> {
        let mut released = Vec::new();
        let mut queue = lock_pictures(&self.finished);
        while let Some(next) = queue.pop_front() {
            match next {
                Ok(picture) => self.reorder.push(picture.pts(), picture, &mut released),
                Err(err) if released.is_empty() => return Err(err),
                Err(err) => {
                    queue.push_front(Err(err));
                    break;
                }
            }
        }
        Ok(released.into_iter().map(Output::Picture).collect())
    }
}

/// How many pictures the stream may finish ahead of display order.
///
/// H.264 and HEVC read their sequence parameter set; one that does not parse
/// holds [`reorder::MAX_DEPTH`]. MPEG-2 and MPEG-4 hold one anchor. Every
/// other codec here finishes in display order.
fn reorder_depth(codec: CodecId, extra: &[u8]) -> usize {
    match codec {
        CodecId::H264 => nal::avc_config(extra)
            .ok()
            .and_then(|config| config.sps.first().and_then(|sps| reorder::h264_depth(sps)))
            .unwrap_or(reorder::MAX_DEPTH),
        CodecId::Hevc => nal::read_hevc_sets(extra)
            .ok()
            .and_then(|config| {
                config
                    .nals
                    .iter()
                    .find(|set| nal::hevc_type(set) == Some(33))
                    .and_then(|sps| reorder::hevc_depth(sps))
            })
            .unwrap_or(reorder::MAX_DEPTH),
        CodecId::Mpeg2 | CodecId::Mpeg4 => 1,
        _ => 0,
    }
}

fn lock_pictures(
    finished: &Finished,
) -> std::sync::MutexGuard<'_, VecDeque<Result<PictureOutput, CodecError>>> {
    match finished.pictures.lock() {
        Ok(guard) => guard,
        Err(poison) => poison.into_inner(),
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
        match input {
            FrameInput::Cpu(frame) => {
                let (width, height, bytes, time) = cpu_rgba(frame, self.width, self.height)?;
                let buffer = bgra_buffer(width, height, bytes)?;
                self.encode_image(buffer.as_mut(), time)
            }
            FrameInput::Platform(frame) => {
                if frame.width() != self.width || frame.height() != self.height {
                    return Err(CodecError::Unsupported { what: "frame" });
                }
                let time = frame.time();
                let token = frame.token();
                self.encode_token(token, time)
            }
        }
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
    fn encode_token(&mut self, token: u64, time: FrameTime) -> Result<Vec<Packet>, CodecError> {
        with_pixel_buffer(token, |image| {
            let format = unsafe { sys::CVPixelBufferGetPixelFormatType(image) };
            if format != sys::BGRA {
                return Err(CodecError::Unsupported {
                    what: "pixel format",
                });
            }
            self.encode_image(image, time)
        })?
    }

    fn encode_image(
        &mut self,
        image: *mut std::ffi::c_void,
        time: FrameTime,
    ) -> Result<Vec<Packet>, CodecError> {
        let mut meta = FrameMeta {
            pts: time.pts(),
            duration: time.duration(),
            timescale: time.timescale(),
        };
        let force = {
            let queue = lock_queue(&self.output);
            !queue.attached && queue.packets.is_empty()
        };
        let properties = if force { Some(force_keyframe()?) } else { None };
        let status = unsafe {
            sys::VTCompressionSessionEncodeFrame(
                self.session,
                image,
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
    if length_framed(data) {
        return Ok(None);
    }
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

/// Four-byte NAL lengths that cover every byte of `data`.
///
/// A container packet is length framed. Its prefix can read as a start
/// code: a NAL of 256 to 511 bytes begins `00 00 01`. So length framing is
/// tried before Annex B.
fn length_framed(data: &[u8]) -> bool {
    nal::split_lengths(data, 4).is_ok()
}

fn access_unit(codec: CodecId, data: &[u8]) -> Result<Vec<u8>, CodecError> {
    if !matches!(codec, CodecId::H264 | CodecId::Hevc) || length_framed(data) {
        return Ok(data.to_vec());
    }
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

/// `surface` is the pixel format VideoToolbox writes. `None` keeps the
/// decoder's own format.
/// `refcon` is the [`Finished`] the callback writes into.
fn create_decompress(
    format: &Cf,
    surface: Option<u32>,
    refcon: *mut std::ffi::c_void,
) -> Result<Decompress, CodecError> {
    let attributes = match surface {
        Some(pixel) => Some(pixel_attributes(pixel)?),
        None => None,
    };
    let callback = sys::VTDecompressionOutputCallbackRecord {
        callback: decode_output,
        refcon,
    };
    let mut raw = ptr::null_mut();
    let status = unsafe {
        sys::VTDecompressionSessionCreate(
            ptr::null(),
            format.as_ptr(),
            ptr::null(),
            attributes.as_ref().map(Cf::as_ptr).unwrap_or(ptr::null()),
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

/// The status of the decode call. A unit that could not be built is the error.
fn decode_access(
    session: *mut std::ffi::c_void,
    format: &Cf,
    bytes: &[u8],
    pts: i64,
    timescale: u32,
) -> Result<sys::OSStatus, CodecError> {
    let block = block_from(bytes)?;
    let scale = i32::try_from(timescale).unwrap_or(30);
    let timing = CMSampleTimingInfo {
        duration: cf::time(1, scale),
        presentation: cf::time(pts, scale),
        decode: cf::time_invalid(),
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
    let status = unsafe {
        sys::VTDecompressionSessionDecodeFrame(
            session,
            sample.as_ptr(),
            sys::DECODE_ASYNC | sys::DECODE_TEMPORAL,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    Ok(status)
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

/// Runs on a VideoToolbox thread, once per unit, in display order.
///
/// A unit the decoder chose to drop has no image and no error. It adds nothing.
unsafe extern "C" fn decode_output(
    refcon: *mut std::ffi::c_void,
    _source: *mut std::ffi::c_void,
    status: sys::OSStatus,
    _flags: u32,
    image: *mut std::ffi::c_void,
    pts: sys::CMTime,
    _duration: sys::CMTime,
) {
    if refcon.is_null() || (status == 0 && image.is_null()) {
        return;
    }
    // SAFETY: the session's refcon is the decoder's boxed `Finished`, and the
    // session is flushed and invalidated before that box is dropped.
    let finished = unsafe { &*refcon.cast::<Finished>() };
    if session_lost(status) {
        finished.lost.store(true, Ordering::Release);
    }
    let timescale = u32::try_from(pts.timescale).unwrap_or(30);
    let picture = if status != 0 {
        Err(CodecError::Malformed {
            at_bit: 0,
            what: "videotoolbox",
        })
    } else if finished.planar {
        planar_picture(image, pts.value, timescale)
    } else {
        gpu_picture(image, pts.value, timescale)
    };
    lock_pictures(finished).push_back(picture);
}

struct HeldBuffer {
    buffer: *mut std::ffi::c_void,
}

// SAFETY: `buffer` is a retained `CVPixelBuffer`. The map is its owner, and
// a lookup retains it again before dropping the lock.
unsafe impl Send for HeldBuffer {}

fn buffers() -> &'static Mutex<HashMap<u64, HeldBuffer>> {
    static BUFFERS: std::sync::OnceLock<Mutex<HashMap<u64, HeldBuffer>>> =
        std::sync::OnceLock::new();
    BUFFERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_token(held: &HashMap<u64, HeldBuffer>) -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    loop {
        let token = NEXT.fetch_add(1, Ordering::Relaxed);
        if token != 0 && !held.contains_key(&token) {
            return token;
        }
    }
}

fn lock_buffers() -> std::sync::MutexGuard<'static, HashMap<u64, HeldBuffer>> {
    match buffers().lock() {
        Ok(guard) => guard,
        Err(poison) => poison.into_inner(),
    }
}

/// Retain `image` and return the token that names it.
fn publish_buffer(image: *mut std::ffi::c_void) -> Result<u64, CodecError> {
    let retained = unsafe { sys::CFRetain(image) };
    if retained.is_null() {
        return Err(CodecError::Unsupported {
            what: "pixel buffer",
        });
    }
    let mut held = lock_buffers();
    let token = next_token(&held);
    held.insert(
        token,
        HeldBuffer {
            buffer: retained.cast_mut(),
        },
    );
    Ok(token)
}

/// Borrow the retained pixel buffer for `body`. The pointer is a BGRA
/// `CVPixelBuffer` and is only valid for that call.
pub fn with_pixel_buffer<T>(
    token: u64,
    body: impl FnOnce(*mut std::ffi::c_void) -> T,
) -> Result<T, CodecError> {
    let retained = {
        let held = lock_buffers();
        let Some(buffer) = held.get(&token) else {
            return Err(CodecError::Unsupported {
                what: "platform frame",
            });
        };
        let retained = unsafe { sys::CFRetain(buffer.buffer) };
        if retained.is_null() {
            return Err(CodecError::Unsupported {
                what: "pixel buffer",
            });
        }
        retained.cast_mut()
    };
    let value = body(retained);
    unsafe { sys::CFRelease(retained) }
    Ok(value)
}

/// Drop one retained pixel buffer. A token this crate did not publish is
/// [`CodecError::Unsupported`].
pub fn release_buffer(token: u64) -> Result<(), CodecError> {
    let Some(held) = lock_buffers().remove(&token) else {
        return Err(CodecError::Unsupported {
            what: "platform frame",
        });
    };
    unsafe { sys::CFRelease(held.buffer) }
    Ok(())
}

/// RGBA bytes for a picture. A CPU picture is cloned. A token this crate
/// published is copied out of its BGRA buffer.
pub fn to_cpu(picture: &PictureOutput) -> Result<PictureOutput, CodecError> {
    match picture.samples() {
        PictureSamples::Cpu(_) => Ok(picture.clone()),
        PictureSamples::Platform { token } => with_pixel_buffer(token, |image| {
            rgba_picture(image, picture.pts(), picture.timescale())
        })?,
    }
}

fn gpu_picture(
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
    let token = publish_buffer(image)?;
    match PictureOutput::platform(
        PictureInfo::new(
            width,
            height,
            PixelFormat::Rgba8,
            ColorPrimaries::Bt709,
            Transfer::Unspecified,
        ),
        token,
        pts,
        timescale,
    ) {
        Ok(picture) => Ok(picture),
        Err(err) => {
            let _ = release_buffer(token);
            Err(err)
        }
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
        PictureInfo::new(
            width,
            height,
            PixelFormat::Rgba8,
            ColorPrimaries::Bt709,
            Transfer::Unspecified,
        ),
        rgba,
        pts,
        timescale,
    )
}

fn planar_picture(
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
    let result = copy_planar(image, pts, timescale);
    unsafe {
        let _ = sys::CVPixelBufferUnlockBaseAddress(image, 1);
    }
    result
}

/// One locked plane: its first byte, its row stride, and its rows.
fn plane(
    image: *mut std::ffi::c_void,
    index: usize,
    row: usize,
    rows: usize,
) -> Result<&'static [u8], CodecError> {
    let base = unsafe { sys::CVPixelBufferGetBaseAddressOfPlane(image, index) };
    let stride = unsafe { sys::CVPixelBufferGetBytesPerRowOfPlane(image, index) };
    let height = unsafe { sys::CVPixelBufferGetHeightOfPlane(image, index) };
    if base.is_null() || stride < row || height < rows {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "pixel plane",
        });
    }
    let len = stride.checked_mul(rows).ok_or(CodecError::Limit {
        what: "picture bytes",
    })?;
    // SAFETY: the buffer is locked for the caller's copy, and `rows` rows of
    // `stride` bytes lie inside the plane.
    Ok(unsafe { std::slice::from_raw_parts(base.cast::<u8>(), len) })
}

/// Copy a video-range NV12 buffer to packed Y, then U, then V.
///
/// An odd edge drops its last column or row, since 4:2:0 needs even edges.
fn copy_planar(
    image: *mut std::ffi::c_void,
    pts: i64,
    timescale: u32,
) -> Result<PictureOutput, CodecError> {
    let format = unsafe { sys::CVPixelBufferGetPixelFormatType(image) };
    if format != sys::NV12 || unsafe { sys::CVPixelBufferGetPlaneCount(image) } != 2 {
        return Err(CodecError::Unsupported {
            what: "pixel format",
        });
    }
    let width = unsafe { sys::CVPixelBufferGetWidth(image) } & !1;
    let height = unsafe { sys::CVPixelBufferGetHeight(image) } & !1;
    let edge = |value: usize| {
        u32::try_from(value).map_err(|_| CodecError::Limit {
            what: "picture edge",
        })
    };
    let (picture_width, picture_height) = (edge(width)?, edge(height)?);
    limits::picture_ok(picture_width, picture_height)?;
    let (chroma_width, chroma_height) = (width / 2, height / 2);
    if unsafe { sys::CVPixelBufferGetWidthOfPlane(image, 1) } < chroma_width {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "pixel plane",
        });
    }
    let luma = plane(image, 0, width, height)?;
    let chroma = plane(image, 1, chroma_width * 2, chroma_height)?;
    let luma_stride = unsafe { sys::CVPixelBufferGetBytesPerRowOfPlane(image, 0) };
    let chroma_stride = unsafe { sys::CVPixelBufferGetBytesPerRowOfPlane(image, 1) };
    let chroma_len = chroma_width * chroma_height;
    let mut bytes = vec![0u8; width * height + chroma_len * 2];
    let (y, rest) = bytes.split_at_mut(width * height);
    let (u, v) = rest.split_at_mut(chroma_len);
    for (row, out) in y.chunks_exact_mut(width).enumerate() {
        let start = row * luma_stride;
        out.copy_from_slice(&luma[start..start + width]);
    }
    for row in 0..chroma_height {
        let start = row * chroma_stride;
        let pairs = &chroma[start..start + chroma_width * 2];
        let out = row * chroma_width;
        for (x, pair) in pairs.as_chunks::<2>().0.iter().enumerate() {
            u[out + x] = pair[0];
            v[out + x] = pair[1];
        }
    }
    let (primaries, transfer) = buffer_color(image);
    PictureOutput::cpu(
        PictureInfo::new(
            picture_width,
            picture_height,
            PixelFormat::Yuv420p,
            primaries,
            transfer,
        ),
        bytes,
        pts,
        timescale,
    )
}

/// The primaries and transfer VideoToolbox attached to `image`.
///
/// A missing or unnamed tag is unspecified. CoreVideo has no SMPTE 170M
/// transfer, so a BT.601 stream reads back as the identical BT.709 curve.
fn buffer_color(image: *mut std::ffi::c_void) -> (ColorPrimaries, Transfer) {
    let primaries = unsafe {
        match attachment(image, sys::kCVImageBufferColorPrimariesKey) {
            Some(tag) if same(&tag, sys::kCVImageBufferColorPrimaries_ITU_R_709_2) => {
                ColorPrimaries::Bt709
            }
            Some(tag)
                if same(&tag, sys::kCVImageBufferColorPrimaries_SMPTE_C)
                    || same(&tag, sys::kCVImageBufferColorPrimaries_EBU_3213) =>
            {
                ColorPrimaries::Bt601
            }
            Some(tag) if same(&tag, sys::kCVImageBufferColorPrimaries_ITU_R_2020) => {
                ColorPrimaries::Bt2020
            }
            _ => ColorPrimaries::Unspecified,
        }
    };
    let transfer = unsafe {
        match attachment(image, sys::kCVImageBufferTransferFunctionKey) {
            Some(tag) if same(&tag, sys::kCVImageBufferTransferFunction_ITU_R_709_2) => {
                Transfer::Bt709
            }
            Some(tag) if same(&tag, sys::kCVImageBufferTransferFunction_ITU_R_2020) => {
                Transfer::Bt2020
            }
            _ => Transfer::Unspecified,
        }
    };
    (primaries, transfer)
}

fn attachment(image: *mut std::ffi::c_void, key: *const std::ffi::c_void) -> Option<Cf> {
    Cf::from_owned(unsafe { sys::CVBufferCopyAttachment(image, key, ptr::null_mut()) })
}

fn same(tag: &Cf, name: *const std::ffi::c_void) -> bool {
    !name.is_null() && unsafe { sys::CFEqual(tag.as_ptr(), name) } != 0
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
    pixel_attributes(sys::BGRA)
}

/// IOSurface-backed buffers in `pixel`.
fn pixel_attributes(pixel: u32) -> Result<Cf, CodecError> {
    let format = cf::number_i32(i32::try_from(pixel).unwrap_or(0))?;
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
        let mut params = CodecParams::with_format(codec, width, height, PixelFormat::Rgba8);
        params.bitrate = 500_000;
        params.primaries = ColorPrimaries::Bt709;
        params
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

    /// Every output of `packets`, then the pictures `drain` releases.
    fn decode_all(
        decoder: &mut dyn Decoder,
        packets: &[Packet],
    ) -> Result<Vec<Output>, CodecError> {
        let mut outputs = Vec::new();
        for packet in packets {
            outputs.extend(decoder.push(packet)?);
        }
        outputs.extend(decoder.drain()?);
        Ok(outputs)
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
    fn a_length_prefix_that_reads_as_a_start_code_stays_length_framed() {
        let mut nal = vec![0x41];
        nal.resize(300, 0x9a);
        let mut packet = u32::try_from(nal.len()).unwrap().to_be_bytes().to_vec();
        packet.extend(&nal);
        assert_eq!(&packet[..3], &[0, 0, 1]);
        assert_eq!(access_unit(CodecId::H264, &packet).unwrap(), packet);
        assert_eq!(in_band_config(CodecId::H264, &packet).unwrap(), None);
    }

    fn encoded(codec: CodecId, frames: i64) -> Option<(Vec<Packet>, Vec<u8>)> {
        let mut encoder = open_encoder(&video_params(codec, 64, 48)).ok()?;
        let mut packets = Vec::new();
        for pts in 0..frames {
            let shade = u8::try_from(pts * 20 % 200).unwrap_or(0);
            packets.extend(
                encoder
                    .push_frame(&solid(64, 48, [shade, 40, 200, 255], pts))
                    .ok()?,
            );
        }
        packets.extend(encoder.drain().ok()?);
        let config = packets.iter().find_map(|packet| {
            packet.side.iter().find_map(|side| match side {
                SideData::Config(bytes) if !bytes.is_empty() => Some(bytes.clone()),
                _ => None,
            })
        })?;
        Some((packets, config))
    }

    #[test]
    fn pictures_leave_in_display_order_once_drained() {
        for codec in [CodecId::H264, CodecId::Hevc] {
            let Some((packets, config)) = encoded(codec, 12) else {
                continue;
            };
            let mut params = CodecParams::with_format(codec, 64, 48, PixelFormat::Yuv420p);
            params.extra = config;
            let mut decoder = open_decoder(&params).expect("decode open");
            let pts: Vec<i64> = decode_all(decoder.as_mut(), &packets)
                .expect("decode")
                .iter()
                .filter_map(|output| match output {
                    Output::Picture(picture) => Some(picture.pts()),
                    Output::Audio(_) => None,
                })
                .collect();
            assert_eq!(pts, (0..12).collect::<Vec<_>>(), "{codec:?}");
            assert!(decoder.drain().expect("second drain").is_empty());
        }
    }

    #[test]
    fn reset_drops_pictures_still_in_flight() {
        let (packets, config) = encoded(CodecId::H264, 6).expect("h264");
        let mut params = CodecParams::with_format(CodecId::H264, 64, 48, PixelFormat::Yuv420p);
        params.extra = config;
        let mut decoder = open_decoder(&params).expect("decode open");
        for packet in &packets {
            let _ = decoder.push(packet).expect("decode");
        }
        decoder.reset();
        assert!(decoder.drain().expect("drain").is_empty());
        let again = decode_all(decoder.as_mut(), &packets).expect("decode after reset");
        assert_eq!(again.len(), 6);
    }

    #[test]
    fn a_session_the_system_invalidates_is_opened_again() {
        let (packets, config) = encoded(CodecId::H264, 6).expect("h264");
        let mut params = CodecParams::with_format(CodecId::H264, 64, 48, PixelFormat::Yuv420p);
        params.extra = config;
        let mut decoder = video_decoder(&params).expect("decode open");
        assert_eq!(decode_all(&mut decoder, &packets).expect("decode").len(), 6);
        // What iOS does to a hardware session when the app leaves the foreground.
        let raw = decoder.session.as_ref().expect("session").raw;
        unsafe { sys::VTDecompressionSessionInvalidate(raw) };
        decoder.reset();
        let again = decode_all(&mut decoder, &packets).expect("decode after invalidate");
        assert_eq!(again.len(), 6);
    }

    #[test]
    fn h264_decodes_to_cpu_yuv_when_asked() {
        let mut encoder = open_encoder(&video_params(CodecId::H264, 64, 48)).expect("encode open");
        let mut packets = encoder
            .push_frame(&solid(64, 48, [220, 10, 10, 255], 0))
            .expect("encode");
        packets.extend(encoder.drain().expect("drain"));
        let config = packets
            .iter()
            .find_map(|packet| {
                packet.side.iter().find_map(|side| match side {
                    SideData::Config(bytes) if !bytes.is_empty() => Some(bytes.clone()),
                    _ => None,
                })
            })
            .expect("avcC");
        let mut decode_params =
            CodecParams::with_format(CodecId::H264, 64, 48, PixelFormat::Yuv420p);
        decode_params.extra = config;
        let mut decoder = open_decoder(&decode_params).expect("decode open");
        let mut pictures = Vec::new();
        for output in decode_all(decoder.as_mut(), &packets).expect("decode") {
            let Output::Picture(picture) = output else {
                panic!("picture");
            };
            pictures.push(picture);
        }
        let picture = pictures.first().expect("a picture");
        assert_eq!(picture.info().pixel, PixelFormat::Yuv420p);
        assert_eq!((picture.info().width, picture.info().height), (64, 48));
        let PictureSamples::Cpu(bytes) = picture.samples() else {
            panic!("cpu");
        };
        assert_eq!(bytes.len(), 64 * 48 * 3 / 2);
        let mean = |plane: &[u8]| {
            plane.iter().map(|&value| u64::from(value)).sum::<u64>() / plane.len() as u64
        };
        let (y, rest) = bytes.split_at(64 * 48);
        let (u, v) = rest.split_at(32 * 24);
        let (y, u, v) = (mean(y), mean(u), mean(v));
        assert!((40..=90).contains(&y), "red luma {y}");
        assert!(u < 128 && v > 180, "red chroma u {u} v {v}");
    }

    #[test]
    fn h264_round_trips_a_solid_color() {
        let mut encoder = open_encoder(&video_params(CodecId::H264, 64, 64)).expect("encode open");
        let platform =
            PlatformFrame::new(64, 64, 0, FrameTime::new(0, 1, 30).expect("time")).expect("token");
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
        for output in decode_all(decoder.as_mut(), &packets).expect("decode") {
            let Output::Picture(picture) = output else {
                panic!("picture");
            };
            pictures.push(picture);
        }
        assert!(
            !pictures.is_empty(),
            "no pictures from {} packets",
            packets.len()
        );
        let PictureSamples::Platform { token } = pictures[0].samples() else {
            panic!("platform");
        };
        let copied = to_cpu(&pictures[0]).expect("cpu copy");
        let PictureSamples::Cpu(bytes) = copied.samples() else {
            panic!("cpu");
        };
        let red = channel_sum(bytes, 0);
        let green = channel_sum(bytes, 1);
        let blue = channel_sum(bytes, 2);
        assert!(
            red > green.saturating_mul(4) && red > blue.saturating_mul(4),
            "red {red} green {green} blue {blue}"
        );
        release_buffer(token).expect("release");
        assert_eq!(
            release_buffer(token),
            Err(CodecError::Unsupported {
                what: "platform frame",
            })
        );
        for picture in pictures.iter().skip(1) {
            if let PictureSamples::Platform { token } = picture.samples() {
                release_buffer(token).expect("release");
            }
        }
    }

    #[test]
    fn a_retained_frame_encodes_without_a_cpu_copy() {
        let mut encoder = open_encoder(&video_params(CodecId::H264, 64, 64)).expect("encode open");
        let mut packets = encoder
            .push_frame(&solid(64, 64, [220, 10, 10, 255], 0))
            .expect("encode");
        packets.extend(encoder.drain().expect("drain"));
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
        let mut token = None;
        for output in decode_all(decoder.as_mut(), &packets).expect("decode") {
            let Output::Picture(picture) = output else {
                panic!("picture");
            };
            if token.is_none() {
                let PictureSamples::Platform { token: published } = picture.samples() else {
                    panic!("platform");
                };
                token = Some((published, picture.pts(), picture.timescale()));
            } else if let PictureSamples::Platform { token } = picture.samples() {
                release_buffer(token).expect("release extra");
            }
        }
        let (token, pts, timescale) = token.expect("retained frame");
        let mut again = open_encoder(&video_params(CodecId::H264, 64, 64)).expect("second encode");
        let frame = PlatformFrame::new(
            64,
            64,
            token,
            FrameTime::new(pts, 1, timescale).expect("time"),
        )
        .expect("platform frame");
        let mut encoded = again
            .push_frame(&FrameInput::Platform(frame))
            .expect("encode token");
        encoded.extend(again.drain().expect("drain"));
        assert!(!encoded.is_empty(), "token produced no packets");
        let second_config = encoded
            .iter()
            .find_map(|packet| {
                packet.side.iter().find_map(|side| match side {
                    SideData::Config(bytes) if !bytes.is_empty() => Some(bytes.clone()),
                    _ => None,
                })
            })
            .expect("second avcC");
        let mut second_params = video_params(CodecId::H264, 64, 64);
        second_params.extra = second_config;
        let mut second = open_decoder(&second_params).expect("second decode");
        let mut red = 0u64;
        for output in decode_all(second.as_mut(), &encoded).expect("decode token") {
            let Output::Picture(picture) = output else {
                continue;
            };
            let PictureSamples::Platform { token: decoded } = picture.samples() else {
                panic!("platform");
            };
            let copied = to_cpu(&picture).expect("cpu");
            release_buffer(decoded).expect("release decoded");
            let PictureSamples::Cpu(bytes) = copied.samples() else {
                panic!("cpu");
            };
            red += channel_sum(bytes, 0);
        }
        release_buffer(token).expect("release source");
        assert!(red > 0, "retained frame encoded to black");
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

    fn color_round_trip(codec: CodecId) -> Result<(), CodecError> {
        let mut encoder = open_encoder(&video_params(codec, 64, 64))?;
        let mut packets = encoder.push_frame(&solid(64, 64, [220, 10, 10, 255], 0))?;
        packets.extend(encoder.push_frame(&solid(64, 64, [220, 10, 10, 255], 1))?);
        packets.extend(encoder.drain()?);
        if packets.is_empty() {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "no packets",
            });
        }
        let mut decode_params = video_params(codec, 64, 64);
        if let Some(config) = packets.iter().find_map(|packet| {
            packet.side.iter().find_map(|side| match side {
                SideData::Config(bytes) if !bytes.is_empty() => Some(bytes.clone()),
                _ => None,
            })
        }) {
            decode_params.extra = config;
        }
        let mut decoder = open_decoder(&decode_params)?;
        let mut red = 0u64;
        let mut green = 0u64;
        let mut blue = 0u64;
        for output in decode_all(decoder.as_mut(), &packets)? {
            let Output::Picture(picture) = output else {
                continue;
            };
            let PictureSamples::Platform { token } = picture.samples() else {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "platform",
                });
            };
            let copied = to_cpu(&picture);
            let released = release_buffer(token);
            let copied = copied?;
            released?;
            let PictureSamples::Cpu(bytes) = copied.samples() else {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "cpu",
                });
            };
            red += channel_sum(bytes, 0);
            green += channel_sum(bytes, 1);
            blue += channel_sum(bytes, 2);
        }
        if red == 0 || red <= green.saturating_mul(4) || red <= blue.saturating_mul(4) {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "color",
            });
        }
        Ok(())
    }

    #[test]
    fn hevc_round_trips_a_solid_color_when_the_hardware_can() {
        match color_round_trip(CodecId::Hevc) {
            Ok(()) => {}
            Err(CodecError::Unsupported { .. }) => {}
            Err(err) => panic!("{err}"),
        }
    }

    #[test]
    fn prores_round_trips_a_solid_color_when_the_hardware_can() {
        match color_round_trip(CodecId::Prores) {
            Ok(()) => {}
            Err(CodecError::Unsupported { .. }) => {}
            Err(err) => panic!("{err}"),
        }
    }

    #[test]
    fn mjpeg_round_trips_a_solid_color_when_the_hardware_can() {
        match color_round_trip(CodecId::Mjpeg) {
            Ok(()) => {}
            Err(CodecError::Unsupported { .. }) => {}
            Err(err) => panic!("{err}"),
        }
    }

    #[test]
    fn later_video_codecs_decline_or_open() {
        for codec in [CodecId::Av1, CodecId::Vp9, CodecId::Mpeg4, CodecId::Mpeg2] {
            match color_round_trip(codec) {
                Ok(()) | Err(CodecError::Unsupported { .. }) => {}
                Err(err) => panic!("{codec:?} {err}"),
            }
        }
    }

    #[test]
    fn random_packets_fail_closed() {
        let mut decoder = open_decoder(&video_params(CodecId::H264, 64, 64)).expect("open");
        let mut state = 0x1234_5678u32;
        for _ in 0..32 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let len = usize::try_from(state % 48).unwrap_or(0);
            let mut data = Vec::with_capacity(len);
            for index in 0..len {
                let byte = state.wrapping_add(u32::try_from(index).unwrap_or(0)) % 256;
                data.push(u8::try_from(byte).unwrap_or(0));
            }
            let result = decoder.push(&Packet {
                stream: 0,
                data,
                pts: 0,
                duration: 1,
                timescale: 30,
                keyframe: false,
                side: Vec::new(),
            });
            assert!(result.is_err());
        }
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
