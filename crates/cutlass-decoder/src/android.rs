//! Android video decoding via the NDK **MediaExtractor + MediaCodec** pipeline.
//!
//! Unlike the Apple/Windows backends — where one object does demux *and* decode —
//! Android splits the job: [`AMediaExtractor`](ndk_sys::AMediaExtractor) demuxes
//! the container and [`AMediaCodec`](ndk_sys::AMediaCodec) runs the hardware
//! decoder. MediaCodec is also **buffered and asynchronous**: you feed input
//! buffers and drain output buffers on separate cadences. We hide that behind the
//! synchronous [`cutlass_core::VideoDecoder`] pull by running a feed-input /
//! drain-output *pump* inside [`MediaCodecDecoder::next_frame`] until one frame
//! pops out (or the stream ends).
//!
//! ## What this backend does today
//!
//! - Demuxes with `AMediaExtractor`, picks the first `video/*` track, and spins
//!   up a `MediaCodec` decoder for its MIME type in **ByteBuffer mode** (no output
//!   surface), so decoded frames land in system memory.
//! - Reads each output buffer stride/slice-height-aware into [`FrameData::Cpu`],
//!   interpreting the buffer per the codec's reported `COLOR_FORMAT`
//!   (semi-planar → NV12, planar → I420).
//! - Probes size / rotation / frame-rate / colorimetry from the track format,
//!   refining size/stride/color from the codec's output format once it settles.
//! - Seeks with `AMediaExtractor_seekTo(PREVIOUS_SYNC)` + `AMediaCodec_flush`,
//!   and answers [`VideoDecoder::frame_at`] with the shared roll-forward policy
//!   ([`crate::seek`]) so sequential playback / forward scrubbing never pays a
//!   seek-plus-GOP-re-decode per frame.
//!
//! ## What it does not do yet
//!
//! The **GPU zero-copy path is stubbed** — see [`MediaCodecDecoder::next_frame`].
//! The proper path configures the codec with an `ANativeWindow` from an
//! `AImageReader`, then pulls `AHardwareBuffer`s
//! ([`cutlass_core::GpuSurfaceKind::AHardwareBuffer`]) for the renderer to import.
//! That needs an Android device/GPU to validate and is left for follow-up; the
//! CPU path is what this (macOS) build host can typecheck via
//! `cargo check --target aarch64-linux-android`.
//!
//! ## Time base
//!
//! MediaExtractor/MediaCodec timestamps are **microseconds**, so every frame's
//! PTS rate — and [`SourceInfo::time_base`] — is [`US_PER_SEC`].

use core::ffi::c_char;
use core::ptr;
use core::slice;
use std::ffi::{CStr, CString};
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;

use ndk_sys::{
    AMEDIACODEC_BUFFER_FLAG_CODEC_CONFIG, AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM,
    AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED, AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED,
    AMEDIAFORMAT_KEY_CHANNEL_COUNT, AMEDIAFORMAT_KEY_COLOR_FORMAT, AMEDIAFORMAT_KEY_COLOR_RANGE,
    AMEDIAFORMAT_KEY_COLOR_STANDARD, AMEDIAFORMAT_KEY_COLOR_TRANSFER, AMEDIAFORMAT_KEY_DURATION,
    AMEDIAFORMAT_KEY_FRAME_RATE, AMEDIAFORMAT_KEY_HEIGHT, AMEDIAFORMAT_KEY_MIME,
    AMEDIAFORMAT_KEY_PCM_ENCODING, AMEDIAFORMAT_KEY_ROTATION, AMEDIAFORMAT_KEY_SAMPLE_RATE,
    AMEDIAFORMAT_KEY_SLICE_HEIGHT, AMEDIAFORMAT_KEY_STRIDE, AMEDIAFORMAT_KEY_WIDTH, AMediaCodec,
    AMediaCodec_configure, AMediaCodec_createDecoderByType, AMediaCodec_delete,
    AMediaCodec_dequeueInputBuffer, AMediaCodec_dequeueOutputBuffer, AMediaCodec_flush,
    AMediaCodec_getInputBuffer, AMediaCodec_getOutputBuffer, AMediaCodec_getOutputFormat,
    AMediaCodec_queueInputBuffer, AMediaCodec_releaseOutputBuffer, AMediaCodec_start,
    AMediaCodec_stop, AMediaCodecBufferInfo, AMediaExtractor, AMediaExtractor_advance,
    AMediaExtractor_delete, AMediaExtractor_getSampleTime, AMediaExtractor_getTrackCount,
    AMediaExtractor_getTrackFormat, AMediaExtractor_new, AMediaExtractor_readSampleData,
    AMediaExtractor_seekTo, AMediaExtractor_selectTrack, AMediaExtractor_setDataSourceFd,
    AMediaFormat, AMediaFormat_delete, AMediaFormat_getInt32, AMediaFormat_getInt64,
    AMediaFormat_getString, SeekMode, media_status_t,
};

use cutlass_core::{
    AudioReader, ColorPrimaries, ColorRange, ColorSpace, CpuImage, DecodeError, FrameData,
    MatrixCoefficients, PixelFormat, Plane, Rational, RationalTime, Rect, Rotation, SourceInfo,
    TransferFunction, VideoDecoder, VideoFrame,
};

use crate::OutputMode;

/// MediaExtractor/MediaCodec timestamp tick rate: microseconds per second.
const US_PER_SEC: i64 = 1_000_000;

/// Safety net on the feed/drain pump so a stuck codec can't hang the worker.
const MAX_PUMP_ITERS: usize = 512;

/// Don't block waiting for a free input buffer — drain output instead.
const INPUT_TIMEOUT_US: i64 = 0;
/// Give the decoder a short window to surface an output buffer each iteration.
const OUTPUT_TIMEOUT_US: i64 = 10_000;

const GPU_UNSUPPORTED: &str = "Android MediaCodec GPU/AHardwareBuffer output is not implemented yet; open with OutputMode::Cpu";

// `android.media.MediaFormat` COLOR_STANDARD_* values (KEY_COLOR_STANDARD).
const COLOR_STANDARD_BT709: i32 = 1;
const COLOR_STANDARD_BT601_PAL: i32 = 2;
const COLOR_STANDARD_BT601_NTSC: i32 = 4;
const COLOR_STANDARD_BT2020: i32 = 6;
// COLOR_TRANSFER_* values (KEY_COLOR_TRANSFER).
const COLOR_TRANSFER_SDR_VIDEO: i32 = 3;
const COLOR_TRANSFER_ST2084: i32 = 6;
const COLOR_TRANSFER_HLG: i32 = 7;
// COLOR_RANGE_* values (KEY_COLOR_RANGE).
const COLOR_RANGE_FULL: i32 = 1;
// `MediaCodecInfo.CodecCapabilities` COLOR_Format* values (KEY_COLOR_FORMAT).
const COLOR_FORMAT_YUV420_PLANAR: i32 = 19;
const COLOR_FORMAT_YUV420_PACKED_PLANAR: i32 = 20;
const COLOR_FORMAT_YUV420_SEMIPLANAR: i32 = 21;
const COLOR_FORMAT_YUV420_PACKED_SEMIPLANAR: i32 = 39;

/// Point `extractor` at `path` via a file descriptor.
///
/// `AMediaExtractor_setDataSource(path)` parses its argument as a URL and is
/// unreliable for bare filesystem paths on many devices (it fails with no
/// detail, as seen on app-private `cache/` files). Opening the file ourselves
/// and handing MediaExtractor the fd via `AMediaExtractor_setDataSourceFd` is
/// the robust path for local files. The returned [`File`] owns the fd and must
/// outlive the extractor.
fn set_data_source_fd(extractor: *mut AMediaExtractor, path: &Path) -> Result<File, DecodeError> {
    let file = File::open(path)
        .map_err(|e| DecodeError::Open(format!("open {} failed: {e}", path.display())))?;
    let len = file
        .metadata()
        .map(|m| m.len() as i64)
        .map_err(|e| DecodeError::Open(format!("stat {} failed: {e}", path.display())))?;
    let fd = file.as_raw_fd();
    if !media_ok(unsafe { AMediaExtractor_setDataSourceFd(extractor, fd, 0, len) }) {
        return Err(DecodeError::Open(format!(
            "AMediaExtractor_setDataSourceFd failed for {}",
            path.display()
        )));
    }
    Ok(file)
}

/// A pull-based decoder over one video file, backed by MediaExtractor +
/// MediaCodec.
pub struct MediaCodecDecoder {
    extractor: *mut AMediaExtractor,
    codec: *mut AMediaCodec,
    info: SourceInfo,
    mode: OutputMode,
    /// Keeps the source fd open for the extractor's lifetime (see
    /// [`set_data_source_fd`]). Dropped after the native teardown in `Drop`.
    _source: File,
    /// Y-plane row pitch in bytes and number of luma rows in the output buffer
    /// (the codec pads both for alignment); refined from the output format.
    stride: usize,
    slice_height: usize,
    /// We've queued the end-of-stream marker into the codec's input.
    input_done: bool,
    /// We've drained the end-of-stream marker out of the codec's output.
    eos: bool,
    /// PTS of the last emitted frame — the [`crate::seek`] roll-forward anchor.
    /// Cleared on [`VideoDecoder::seek`] (decode position moved).
    last_pts: Option<RationalTime>,
    /// Observed seek/decode costs driving the adaptive roll window.
    seek_stats: crate::seek::SeekStats,
}

// SAFETY: the extractor and codec are owned and touched by a single thread at a
// time — the decode-ahead worker the engine parks this decoder on. We never
// share `&self` across threads, satisfying `VideoDecoder: Send` without `Sync`.
unsafe impl Send for MediaCodecDecoder {}

/// Cleans up the half-built native objects if `open` bails partway through.
/// On success `open` `mem::forget`s the guard and hands the pointers to the
/// [`MediaCodecDecoder`], whose own `Drop` then owns teardown.
struct PartialCleanup {
    extractor: *mut AMediaExtractor,
    codec: *mut AMediaCodec,
    format: *mut AMediaFormat,
}

impl Drop for PartialCleanup {
    fn drop(&mut self) {
        unsafe {
            if !self.codec.is_null() {
                AMediaCodec_stop(self.codec);
                AMediaCodec_delete(self.codec);
            }
            if !self.format.is_null() {
                AMediaFormat_delete(self.format);
            }
            if !self.extractor.is_null() {
                AMediaExtractor_delete(self.extractor);
            }
        }
    }
}

impl MediaCodecDecoder {
    /// Open `path` and probe its first video track.
    pub fn open(path: &Path, mode: OutputMode) -> Result<Self, DecodeError> {
        let extractor = unsafe { AMediaExtractor_new() };
        if extractor.is_null() {
            return Err(DecodeError::Open(
                "AMediaExtractor_new returned null".into(),
            ));
        }
        let mut guard = PartialCleanup {
            extractor,
            codec: ptr::null_mut(),
            format: ptr::null_mut(),
        };

        let source = set_data_source_fd(extractor, path)?;

        // Find the first video track and keep its format.
        let (track_index, track_format, mime) = find_video_track(extractor)?;
        guard.format = track_format;

        if !media_ok(unsafe { AMediaExtractor_selectTrack(extractor, track_index) }) {
            return Err(DecodeError::Open(
                "AMediaExtractor_selectTrack failed".into(),
            ));
        }

        let c_mime = CString::new(mime.clone())
            .map_err(|_| DecodeError::Unsupported(format!("MIME type has a NUL byte: {mime}")))?;
        let codec = unsafe { AMediaCodec_createDecoderByType(c_mime.as_ptr()) };
        if codec.is_null() {
            return Err(DecodeError::Unsupported(format!(
                "no MediaCodec decoder for {mime}"
            )));
        }
        guard.codec = codec;

        // ByteBuffer mode: null surface => decoded planes land in system memory.
        if !media_ok(unsafe {
            AMediaCodec_configure(codec, track_format, ptr::null_mut(), ptr::null_mut(), 0)
        }) {
            return Err(DecodeError::Decode("AMediaCodec_configure failed".into()));
        }
        if !media_ok(unsafe { AMediaCodec_start(codec) }) {
            return Err(DecodeError::Decode("AMediaCodec_start failed".into()));
        }

        let info = unsafe { probe_container(track_format) }?;
        let (stride, slice_height) = (info.coded_size.0 as usize, info.coded_size.1 as usize);

        // Track format is copied by `configure`; release our reference.
        unsafe { AMediaFormat_delete(track_format) };
        guard.format = ptr::null_mut();

        let decoder = Self {
            extractor,
            codec,
            info,
            mode,
            _source: source,
            stride,
            slice_height,
            input_done: false,
            eos: false,
            last_pts: None,
            seek_stats: Default::default(),
        };
        // Ownership transferred to `decoder`; don't let the guard tear it down.
        core::mem::forget(guard);
        Ok(decoder)
    }

    /// Feed one input buffer from the extractor into the codec, if a buffer is
    /// free and the source isn't exhausted.
    fn pump_input(&mut self) {
        if self.input_done {
            return;
        }
        let in_index = unsafe { AMediaCodec_dequeueInputBuffer(self.codec, INPUT_TIMEOUT_US) };
        if in_index < 0 {
            return; // no free input buffer right now
        }
        let index = in_index as usize;
        let mut capacity: usize = 0;
        let buffer = unsafe { AMediaCodec_getInputBuffer(self.codec, index, &mut capacity) };
        if buffer.is_null() {
            return;
        }

        let read = unsafe { AMediaExtractor_readSampleData(self.extractor, buffer, capacity) };
        if read < 0 {
            // End of stream: queue an empty buffer flagged EOS to flush the codec.
            unsafe {
                AMediaCodec_queueInputBuffer(
                    self.codec,
                    index,
                    0,
                    0,
                    0,
                    AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM,
                );
            }
            self.input_done = true;
        } else {
            let pts_us = unsafe { AMediaExtractor_getSampleTime(self.extractor) }.max(0) as u64;
            unsafe {
                AMediaCodec_queueInputBuffer(self.codec, index, 0, read as usize, pts_us, 0);
                AMediaExtractor_advance(self.extractor);
            }
        }
    }

    /// Copy a decoded output buffer into a CPU [`VideoFrame`].
    fn read_cpu_frame(
        &self,
        buffer: *const u8,
        info: &AMediaCodecBufferInfo,
        pts: RationalTime,
    ) -> Result<VideoFrame, DecodeError> {
        let base = unsafe { buffer.add(info.offset.max(0) as usize) };
        let available = info.size.max(0) as usize;
        let planes = unsafe {
            read_planes(
                base,
                available,
                self.info.pixel_format,
                self.stride,
                self.slice_height,
            )
        }?;
        let visible = Rect::from_size(self.info.display_size.0, self.info.display_size.1);
        Ok(VideoFrame::new(
            pts,
            self.info.pixel_format,
            self.info.color,
            self.info.coded_size,
            visible,
            self.info.rotation,
            FrameData::Cpu(CpuImage::new(planes)),
        ))
    }
}

impl VideoDecoder for MediaCodecDecoder {
    fn info(&self) -> &SourceInfo {
        &self.info
    }

    fn seek(&mut self, target: RationalTime) -> Result<(), DecodeError> {
        let seek_us = to_us(target);
        if !media_ok(unsafe {
            AMediaExtractor_seekTo(
                self.extractor,
                seek_us,
                SeekMode::AMEDIAEXTRACTOR_SEEK_PREVIOUS_SYNC,
            )
        }) {
            return Err(DecodeError::Seek(format!("seekTo {seek_us}us failed")));
        }
        // Drop the codec's in-flight buffers so the next `next_frame` decodes
        // forward from the new keyframe.
        if !media_ok(unsafe { AMediaCodec_flush(self.codec) }) {
            return Err(DecodeError::Seek("AMediaCodec_flush failed".into()));
        }
        self.input_done = false;
        self.eos = false;
        self.last_pts = None;
        Ok(())
    }

    fn next_frame(&mut self) -> Result<Option<VideoFrame>, DecodeError> {
        if self.eos {
            return Ok(None);
        }
        // The GPU surface path isn't wired up; fail before doing decode work
        // rather than silently downgrading the caller's chosen mode.
        if matches!(self.mode, OutputMode::Gpu) {
            return Err(DecodeError::unsupported(GPU_UNSUPPORTED));
        }

        let mut info = AMediaCodecBufferInfo {
            offset: 0,
            size: 0,
            presentationTimeUs: 0,
            flags: 0,
        };

        for _ in 0..MAX_PUMP_ITERS {
            self.pump_input();

            let out_index = unsafe {
                AMediaCodec_dequeueOutputBuffer(self.codec, &mut info, OUTPUT_TIMEOUT_US)
            };

            if out_index >= 0 {
                let index = out_index as usize;
                let is_eos = info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM != 0;
                let is_config = info.flags & AMEDIACODEC_BUFFER_FLAG_CODEC_CONFIG != 0;
                if is_eos {
                    self.eos = true;
                }
                // Codec-config (CSD) and empty buffers aren't displayable frames.
                if is_config || info.size == 0 {
                    unsafe { AMediaCodec_releaseOutputBuffer(self.codec, index, false) };
                    if self.eos {
                        return Ok(None);
                    }
                    continue;
                }

                let mut capacity: usize = 0;
                let buffer =
                    unsafe { AMediaCodec_getOutputBuffer(self.codec, index, &mut capacity) };
                if buffer.is_null() {
                    unsafe { AMediaCodec_releaseOutputBuffer(self.codec, index, false) };
                    return Err(DecodeError::Decode("null output buffer".into()));
                }
                let pts = RationalTime::new(info.presentationTimeUs, self.info.time_base);
                let frame = self.read_cpu_frame(buffer, &info, pts);
                // Release before propagating any read error so we never leak a
                // dequeued buffer back to the codec.
                unsafe { AMediaCodec_releaseOutputBuffer(self.codec, index, false) };
                let frame = frame?;
                self.last_pts = Some(pts);
                return Ok(Some(frame));
            }

            let out_index = out_index as i32;
            if out_index == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED {
                let out_format = unsafe { AMediaCodec_getOutputFormat(self.codec) };
                if !out_format.is_null() {
                    self.refine_from_output(out_format);
                    unsafe { AMediaFormat_delete(out_format) };
                }
            } else if out_index == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED {
                // Deprecated no-op on the NDK path; keep pumping.
            } else {
                // INFO_TRY_AGAIN_LATER: nothing ready yet.
                if self.input_done && self.eos {
                    return Ok(None);
                }
            }
        }

        Err(DecodeError::Decode(
            "decode pump exceeded its iteration budget without producing a frame".into(),
        ))
    }

    fn frame_at(&mut self, target: RationalTime) -> Result<Option<VideoFrame>, DecodeError> {
        let last_pts = self.last_pts;
        // Copied out and written back: `frame_at_rolling` borrows `self` for
        // the decode walk, so it can't also borrow the field.
        let mut stats = self.seek_stats;
        let result = crate::seek::frame_at_rolling(self, last_pts, &mut stats, target);
        self.seek_stats = stats;
        result
    }
}

impl MediaCodecDecoder {
    /// Update decode geometry, pixel format, and color from the codec's settled
    /// output format (emitted as `INFO_OUTPUT_FORMAT_CHANGED` before frame one).
    fn refine_from_output(&mut self, format: *mut AMediaFormat) {
        if let Some(color_format) = unsafe { format_i32(format, AMEDIAFORMAT_KEY_COLOR_FORMAT) } {
            self.info.pixel_format = color_format_to_pixel(color_format);
        }
        let width = unsafe { format_i32(format, AMEDIAFORMAT_KEY_WIDTH) }
            .filter(|w| *w > 0)
            .map(|w| w as u32)
            .unwrap_or(self.info.display_size.0);
        let height = unsafe { format_i32(format, AMEDIAFORMAT_KEY_HEIGHT) }
            .filter(|h| *h > 0)
            .map(|h| h as u32)
            .unwrap_or(self.info.display_size.1);

        let stride = unsafe { format_i32(format, AMEDIAFORMAT_KEY_STRIDE) }
            .filter(|s| *s > 0)
            .map(|s| s as usize)
            .unwrap_or(width as usize);
        let slice_height = unsafe { format_i32(format, AMEDIAFORMAT_KEY_SLICE_HEIGHT) }
            .filter(|s| *s > 0)
            .map(|s| s as usize)
            .unwrap_or(height as usize);

        self.stride = stride.max(width as usize);
        self.slice_height = slice_height.max(height as usize);
        self.info.coded_size = (self.stride as u32, self.slice_height as u32);
        self.info.display_size = (width, height);
        self.info.color = unsafe { color_from_format(format) };
    }
}

impl Drop for MediaCodecDecoder {
    fn drop(&mut self) {
        unsafe {
            if !self.codec.is_null() {
                AMediaCodec_stop(self.codec);
                AMediaCodec_delete(self.codec);
            }
            if !self.extractor.is_null() {
                AMediaExtractor_delete(self.extractor);
            }
        }
    }
}

/// Whether the file at `path` carries at least one `audio/*` track. Used by the
/// probe to set [`crate::MediaProbe::has_audio`]; `false` on any error.
pub(crate) fn has_audio_track(path: &Path) -> bool {
    let extractor = unsafe { AMediaExtractor_new() };
    if extractor.is_null() {
        return false;
    }
    // `_source` keeps the fd open while we read track formats below.
    let mut found = false;
    if let Ok(_source) = set_data_source_fd(extractor, path) {
        let count = unsafe { AMediaExtractor_getTrackCount(extractor) };
        for index in 0..count {
            let format = unsafe { AMediaExtractor_getTrackFormat(extractor, index) };
            if format.is_null() {
                continue;
            }
            let is_audio = unsafe { format_string(format, AMEDIAFORMAT_KEY_MIME) }
                .is_some_and(|mime| mime.starts_with("audio/"));
            unsafe { AMediaFormat_delete(format) };
            if is_audio {
                found = true;
                break;
            }
        }
    }
    unsafe { AMediaExtractor_delete(extractor) };
    found
}

/// Scan the extractor's tracks and return the first `video/*` one: its index,
/// its (caller-owned) format, and its MIME type.
fn find_video_track(
    extractor: *mut AMediaExtractor,
) -> Result<(usize, *mut AMediaFormat, String), DecodeError> {
    let track_count = unsafe { AMediaExtractor_getTrackCount(extractor) };
    for index in 0..track_count {
        let format = unsafe { AMediaExtractor_getTrackFormat(extractor, index) };
        if format.is_null() {
            continue;
        }
        match unsafe { format_string(format, AMEDIAFORMAT_KEY_MIME) } {
            Some(mime) if mime.starts_with("video/") => return Ok((index, format, mime)),
            _ => unsafe {
                AMediaFormat_delete(format);
            },
        }
    }
    Err(DecodeError::Unsupported("source has no video track".into()))
}

/// Probe static properties from the container track format at open time. Size,
/// stride, and exact color are refined later from the codec's output format.
///
/// # Safety
/// `format` must be a valid `AMediaFormat` pointer.
unsafe fn probe_container(format: *mut AMediaFormat) -> Result<SourceInfo, DecodeError> {
    let width = unsafe { format_i32(format, AMEDIAFORMAT_KEY_WIDTH) }
        .filter(|w| *w > 0)
        .ok_or_else(|| DecodeError::Open("track format has no width".into()))?
        as u32;
    let height = unsafe { format_i32(format, AMEDIAFORMAT_KEY_HEIGHT) }
        .filter(|h| *h > 0)
        .ok_or_else(|| DecodeError::Open("track format has no height".into()))?
        as u32;

    let rotation = unsafe { format_i32(format, AMEDIAFORMAT_KEY_ROTATION) }
        .and_then(Rotation::from_degrees)
        .unwrap_or(Rotation::None);

    // FRAME_RATE is usually an int; some containers store a float (which the
    // int getter rejects), so fall back to 30 fps.
    let frame_rate = unsafe { format_i32(format, AMEDIAFORMAT_KEY_FRAME_RATE) }
        .filter(|r| *r > 0)
        .map(|r| Rational::new(r, 1))
        .unwrap_or(Rational::FPS_30);

    let time_base = Rational::new(US_PER_SEC as i32, 1);
    let duration = unsafe { format_i64(format, AMEDIAFORMAT_KEY_DURATION) }
        .filter(|d| *d > 0)
        .map(|us| RationalTime::new(us, time_base));

    let pixel_format = unsafe { format_i32(format, AMEDIAFORMAT_KEY_COLOR_FORMAT) }
        .map(color_format_to_pixel)
        .unwrap_or(PixelFormat::Nv12);

    Ok(SourceInfo {
        coded_size: (width, height),
        display_size: (width, height),
        rotation,
        pixel_format,
        color: unsafe { color_from_format(format) },
        frame_rate,
        time_base,
        duration,
    })
}

/// Copy a biplanar (NV12/P010) or triplanar (I420) 4:2:0 frame out of a
/// MediaCodec ByteBuffer, honoring the codec's `stride` and `slice_height`.
///
/// # Safety
/// `base` must point at `available` readable bytes laid out as the given
/// `format` with the given luma `stride`/`slice_height` (the caller holds the
/// dequeued output buffer).
unsafe fn read_planes(
    base: *const u8,
    available: usize,
    format: PixelFormat,
    stride: usize,
    slice_height: usize,
) -> Result<Vec<Plane>, DecodeError> {
    let y_rows = slice_height;
    let chroma_rows = slice_height.div_ceil(2);
    let overflow = || DecodeError::Decode("plane size overflow".into());

    match format {
        // Biplanar: full-res Y, then interleaved chroma at the same pitch.
        PixelFormat::Nv12 | PixelFormat::P010 => {
            let y_len = stride.checked_mul(y_rows).ok_or_else(overflow)?;
            let chroma_len = stride.checked_mul(chroma_rows).ok_or_else(overflow)?;
            let needed = y_len.checked_add(chroma_len).ok_or_else(overflow)?;
            if available < needed {
                return Err(DecodeError::Decode(format!(
                    "output buffer too small: {available} < {needed} bytes"
                )));
            }
            let luma = unsafe { slice::from_raw_parts(base, y_len) }.to_vec();
            let chroma = unsafe { slice::from_raw_parts(base.add(y_len), chroma_len) }.to_vec();
            Ok(vec![
                Plane::new(luma, stride, y_rows),
                Plane::new(chroma, stride, chroma_rows),
            ])
        }
        // Triplanar I420: Y at full pitch, then U and V at half pitch.
        PixelFormat::Yuv420p => {
            let chroma_stride = stride / 2;
            let y_len = stride.checked_mul(y_rows).ok_or_else(overflow)?;
            let chroma_len = chroma_stride
                .checked_mul(chroma_rows)
                .ok_or_else(overflow)?;
            let needed = y_len
                .checked_add(chroma_len.checked_mul(2).ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
            if available < needed {
                return Err(DecodeError::Decode(format!(
                    "output buffer too small: {available} < {needed} bytes"
                )));
            }
            let luma = unsafe { slice::from_raw_parts(base, y_len) }.to_vec();
            let u = unsafe { slice::from_raw_parts(base.add(y_len), chroma_len) }.to_vec();
            let v =
                unsafe { slice::from_raw_parts(base.add(y_len + chroma_len), chroma_len) }.to_vec();
            Ok(vec![
                Plane::new(luma, stride, y_rows),
                Plane::new(u, chroma_stride, chroma_rows),
                Plane::new(v, chroma_stride, chroma_rows),
            ])
        }
        PixelFormat::Rgba8 | PixelFormat::Bgra8 => Err(DecodeError::unsupported(
            "MediaCodec reported a packed-RGB output color format",
        )),
    }
}

/// Read colorimetry from a format's COLOR_STANDARD/TRANSFER/RANGE keys, leaving
/// the BT.709 SDR default for anything the source doesn't signal.
///
/// # Safety
/// `format` must be a valid `AMediaFormat` pointer.
unsafe fn color_from_format(format: *mut AMediaFormat) -> ColorSpace {
    let mut color = ColorSpace::BT709;
    if let Some(standard) = unsafe { format_i32(format, AMEDIAFORMAT_KEY_COLOR_STANDARD) } {
        let (primaries, matrix) = map_color_standard(standard);
        color.primaries = primaries;
        color.matrix = matrix;
    }
    if let Some(transfer) = unsafe { format_i32(format, AMEDIAFORMAT_KEY_COLOR_TRANSFER) } {
        color.transfer = map_color_transfer(transfer);
    }
    if let Some(range) = unsafe { format_i32(format, AMEDIAFORMAT_KEY_COLOR_RANGE) } {
        color.range = map_color_range(range);
    }
    color
}

/// Map a `COLOR_STANDARD_*` value to primaries + YUV matrix (Android bundles
/// both into one "standard").
fn map_color_standard(value: i32) -> (ColorPrimaries, MatrixCoefficients) {
    match value {
        COLOR_STANDARD_BT709 => (ColorPrimaries::Bt709, MatrixCoefficients::Bt709),
        COLOR_STANDARD_BT601_PAL => (ColorPrimaries::Bt601_625, MatrixCoefficients::Bt601),
        COLOR_STANDARD_BT601_NTSC => (ColorPrimaries::Bt601_525, MatrixCoefficients::Bt601),
        COLOR_STANDARD_BT2020 => (ColorPrimaries::Bt2020, MatrixCoefficients::Bt2020Ncl),
        _ => (ColorPrimaries::Unspecified, MatrixCoefficients::Unspecified),
    }
}

/// Map a `COLOR_TRANSFER_*` value to a transfer function. `LINEAR` has no
/// counterpart in our SDR/HDR set, so it falls through to unspecified.
fn map_color_transfer(value: i32) -> TransferFunction {
    match value {
        COLOR_TRANSFER_SDR_VIDEO => TransferFunction::Bt709,
        COLOR_TRANSFER_ST2084 => TransferFunction::Pq,
        COLOR_TRANSFER_HLG => TransferFunction::Hlg,
        _ => TransferFunction::Unspecified,
    }
}

/// Map a `COLOR_RANGE_*` value; anything but the explicit full code is limited.
fn map_color_range(value: i32) -> ColorRange {
    if value == COLOR_RANGE_FULL {
        ColorRange::Full
    } else {
        ColorRange::Limited
    }
}

/// Map a MediaCodec `COLOR_Format*` value to a plane layout. Flexible / surface
/// / vendor formats can't be interpreted from a bare ByteBuffer, so we assume
/// NV12 (by far the most common hardware output).
fn color_format_to_pixel(value: i32) -> PixelFormat {
    match value {
        COLOR_FORMAT_YUV420_PLANAR | COLOR_FORMAT_YUV420_PACKED_PLANAR => PixelFormat::Yuv420p,
        COLOR_FORMAT_YUV420_SEMIPLANAR | COLOR_FORMAT_YUV420_PACKED_SEMIPLANAR => PixelFormat::Nv12,
        _ => PixelFormat::Nv12,
    }
}

/// Convert a [`RationalTime`] to MediaCodec's microsecond ticks (`i128` math to
/// avoid overflow before truncating to `i64`).
fn to_us(target: RationalTime) -> i64 {
    let num = i128::from(target.rate.num.max(1));
    let den = i128::from(target.rate.den.max(1));
    ((i128::from(target.value) * den * i128::from(US_PER_SEC)) / num) as i64
}

/// `true` when a `media_status_t` is `AMEDIA_OK` (0).
fn media_ok(status: media_status_t) -> bool {
    status.0 == 0
}

/// Read an integer key, or `None` if the format doesn't carry it.
///
/// # Safety
/// `format` must be a valid `AMediaFormat` pointer.
unsafe fn format_i32(format: *mut AMediaFormat, key: *const c_char) -> Option<i32> {
    let mut out: i32 = 0;
    unsafe { AMediaFormat_getInt32(format, key, &mut out) }.then_some(out)
}

/// Read a 64-bit integer key, or `None` if the format doesn't carry it.
///
/// # Safety
/// `format` must be a valid `AMediaFormat` pointer.
unsafe fn format_i64(format: *mut AMediaFormat, key: *const c_char) -> Option<i64> {
    let mut out: i64 = 0;
    unsafe { AMediaFormat_getInt64(format, key, &mut out) }.then_some(out)
}

/// Read a string key as an owned `String`, or `None` if absent.
///
/// # Safety
/// `format` must be a valid `AMediaFormat` pointer.
unsafe fn format_string(format: *mut AMediaFormat, key: *const c_char) -> Option<String> {
    let mut out: *const c_char = ptr::null();
    if unsafe { AMediaFormat_getString(format, key, &mut out) } && !out.is_null() {
        Some(
            unsafe { CStr::from_ptr(out) }
                .to_string_lossy()
                .into_owned(),
        )
    } else {
        None
    }
}

// `android.media.AudioFormat` ENCODING_* values (KEY_PCM_ENCODING).
const ENCODING_PCM_16BIT: i32 = 2;
const ENCODING_PCM_FLOAT: i32 = 4;

/// Safety net on the audio decode pump (whole stream, so a larger budget than
/// the per-frame video pump).
const MAX_AUDIO_PUMP_ITERS: usize = 1_000_000;

/// A pull-based audio decoder over one file: MediaExtractor + MediaCodec decode
/// the whole audio track to PCM up front, which is then converted to
/// interleaved `f32`, mixed to `channels`, and linearly resampled to `out_rate`
/// on demand.
///
/// Decoding eagerly (rather than streaming) keeps seeking trivial — random
/// access into an in-memory buffer — at the cost of holding the decoded track;
/// fine for export, the only consumer in this pass.
pub struct MediaCodecAudioReader {
    /// Whole track as interleaved `f32` at `channels`, still at the *source*
    /// sample rate (resampled to `out_rate` lazily in [`read`](Self::read)).
    samples: Vec<f32>,
    channels: usize,
    src_rate: u32,
    out_rate: u32,
    /// Source-frame count (`samples.len() / channels`).
    src_frames: i64,
    /// Output-frame position of the next [`read`](Self::read).
    position: Option<i64>,
}

// SAFETY: the decoded buffer is plain data and the reader is touched by a single
// worker thread; no native handles outlive `open`.
unsafe impl Send for MediaCodecAudioReader {}

impl MediaCodecAudioReader {
    /// Decode the first audio track of `path` to interleaved `f32`, mixed to
    /// `channels` and (lazily) resampled to `out_rate`.
    pub fn open(path: &Path, out_rate: u32, channels: u16) -> Result<Self, DecodeError> {
        if out_rate == 0 || channels == 0 {
            return Err(DecodeError::unsupported("zero audio rate or channels"));
        }
        let channels = channels as usize;

        let extractor = unsafe { AMediaExtractor_new() };
        if extractor.is_null() {
            return Err(DecodeError::Open(
                "AMediaExtractor_new returned null".into(),
            ));
        }
        let mut guard = PartialCleanup {
            extractor,
            codec: ptr::null_mut(),
            format: ptr::null_mut(),
        };

        // Holds the source fd open through decode; dropped at the end of `open`,
        // after `guard` tears the extractor down.
        let _source = set_data_source_fd(extractor, path)?;

        let (track_index, track_format, mime) = find_audio_track(extractor)?;
        guard.format = track_format;

        let container_rate = unsafe { format_i32(track_format, AMEDIAFORMAT_KEY_SAMPLE_RATE) }
            .filter(|r| *r > 0)
            .unwrap_or(out_rate as i32) as u32;
        let container_channels =
            unsafe { format_i32(track_format, AMEDIAFORMAT_KEY_CHANNEL_COUNT) }
                .filter(|c| *c > 0)
                .unwrap_or(channels as i32);

        if !media_ok(unsafe { AMediaExtractor_selectTrack(extractor, track_index) }) {
            return Err(DecodeError::Open(
                "AMediaExtractor_selectTrack failed".into(),
            ));
        }

        let c_mime = CString::new(mime.clone())
            .map_err(|_| DecodeError::Unsupported(format!("MIME type has a NUL byte: {mime}")))?;
        let codec = unsafe { AMediaCodec_createDecoderByType(c_mime.as_ptr()) };
        if codec.is_null() {
            return Err(DecodeError::Unsupported(format!(
                "no MediaCodec decoder for {mime}"
            )));
        }
        guard.codec = codec;

        if !media_ok(unsafe {
            AMediaCodec_configure(codec, track_format, ptr::null_mut(), ptr::null_mut(), 0)
        }) {
            return Err(DecodeError::Decode(
                "AMediaCodec_configure (audio) failed".into(),
            ));
        }
        if !media_ok(unsafe { AMediaCodec_start(codec) }) {
            return Err(DecodeError::Decode(
                "AMediaCodec_start (audio) failed".into(),
            ));
        }

        // Track format is copied by `configure`; release our reference.
        unsafe { AMediaFormat_delete(track_format) };
        guard.format = ptr::null_mut();

        let (interleaved, src_rate, src_channels) = decode_audio_track(
            extractor,
            codec,
            container_rate,
            container_channels,
            channels,
        )?;

        // `guard`'s Drop tears down the codec + extractor now that we're done.
        drop(guard);

        let src_frames = (interleaved.len() / channels) as i64;
        let _ = src_channels; // already mixed to `channels`
        Ok(Self {
            samples: interleaved,
            channels,
            src_rate,
            out_rate,
            src_frames,
            position: Some(0),
        })
    }

    /// Interleaved `f32` for source frame `frame` (clamped to range), linearly
    /// interpolating between the two nearest source frames.
    fn sample_frame(&self, frame: f64, out: &mut [f32]) {
        let ch = self.channels;
        if self.src_frames == 0 {
            out.iter_mut().for_each(|s| *s = 0.0);
            return;
        }
        let i0 = frame.floor();
        let frac = (frame - i0) as f32;
        let i0 = (i0 as i64).clamp(0, self.src_frames - 1) as usize;
        let i1 = (i0 + 1).min(self.src_frames as usize - 1);
        let base0 = i0 * ch;
        let base1 = i1 * ch;
        for (c, slot) in out.iter_mut().enumerate().take(ch) {
            let a = self.samples[base0 + c];
            let b = self.samples[base1 + c];
            *slot = a + (b - a) * frac;
        }
    }
}

impl AudioReader for MediaCodecAudioReader {
    fn read(&mut self, out: &mut [f32]) -> Result<usize, DecodeError> {
        let ch = self.channels;
        debug_assert_eq!(out.len() % ch, 0, "buffer not a frame multiple");
        let want = out.len() / ch;
        let pos = self.position.unwrap_or(0);
        let ratio = self.src_rate as f64 / self.out_rate as f64;

        let mut produced = 0;
        while produced < want {
            let out_frame = pos + produced as i64;
            let src_pos = out_frame as f64 * ratio;
            if src_pos >= self.src_frames as f64 {
                break; // past end of source
            }
            let dst = &mut out[produced * ch..(produced + 1) * ch];
            self.sample_frame(src_pos, dst);
            produced += 1;
        }

        self.position = Some(pos + produced as i64);
        Ok(produced)
    }

    fn seek_to_frame(&mut self, frame: i64) -> Result<(), DecodeError> {
        // Random access into the decoded buffer: just move the cursor.
        self.position = Some(frame.max(0));
        Ok(())
    }

    fn position(&self) -> Option<i64> {
        self.position
    }
}

/// Scan the extractor's tracks and return the first `audio/*` one: its index,
/// its (caller-owned) format, and its MIME type.
fn find_audio_track(
    extractor: *mut AMediaExtractor,
) -> Result<(usize, *mut AMediaFormat, String), DecodeError> {
    let track_count = unsafe { AMediaExtractor_getTrackCount(extractor) };
    for index in 0..track_count {
        let format = unsafe { AMediaExtractor_getTrackFormat(extractor, index) };
        if format.is_null() {
            continue;
        }
        match unsafe { format_string(format, AMEDIAFORMAT_KEY_MIME) } {
            Some(mime) if mime.starts_with("audio/") => return Ok((index, format, mime)),
            _ => unsafe {
                AMediaFormat_delete(format);
            },
        }
    }
    Err(DecodeError::Unsupported("source has no audio track".into()))
}

/// Pump the codec to end-of-stream, returning the whole track as interleaved
/// `f32` mixed to `out_channels`, plus the authoritative source rate/channels
/// (read from the codec's settled output format).
fn decode_audio_track(
    extractor: *mut AMediaExtractor,
    codec: *mut AMediaCodec,
    fallback_rate: u32,
    fallback_channels: i32,
    out_channels: usize,
) -> Result<(Vec<f32>, u32, usize), DecodeError> {
    let mut input_done = false;
    let mut src_rate = fallback_rate;
    let mut src_channels = fallback_channels.max(1) as usize;
    let mut encoding = ENCODING_PCM_16BIT;
    let mut mixed: Vec<f32> = Vec::new();

    let mut info = AMediaCodecBufferInfo {
        offset: 0,
        size: 0,
        presentationTimeUs: 0,
        flags: 0,
    };

    for _ in 0..MAX_AUDIO_PUMP_ITERS {
        if !input_done {
            let in_index = unsafe { AMediaCodec_dequeueInputBuffer(codec, INPUT_TIMEOUT_US) };
            if in_index >= 0 {
                let index = in_index as usize;
                let mut capacity: usize = 0;
                let buffer = unsafe { AMediaCodec_getInputBuffer(codec, index, &mut capacity) };
                if !buffer.is_null() {
                    let read =
                        unsafe { AMediaExtractor_readSampleData(extractor, buffer, capacity) };
                    if read < 0 {
                        unsafe {
                            AMediaCodec_queueInputBuffer(
                                codec,
                                index,
                                0,
                                0,
                                0,
                                AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM,
                            );
                        }
                        input_done = true;
                    } else {
                        let pts = unsafe { AMediaExtractor_getSampleTime(extractor) }.max(0) as u64;
                        unsafe {
                            AMediaCodec_queueInputBuffer(codec, index, 0, read as usize, pts, 0);
                            AMediaExtractor_advance(extractor);
                        }
                    }
                }
            }
        }

        let out_index =
            unsafe { AMediaCodec_dequeueOutputBuffer(codec, &mut info, OUTPUT_TIMEOUT_US) };

        if out_index >= 0 {
            let index = out_index as usize;
            let is_eos = info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM != 0;
            let is_config = info.flags & AMEDIACODEC_BUFFER_FLAG_CODEC_CONFIG != 0;
            let size = info.size.max(0) as usize;
            let offset = info.offset.max(0) as usize;

            if !is_config && size > 0 {
                let mut capacity: usize = 0;
                let buffer = unsafe { AMediaCodec_getOutputBuffer(codec, index, &mut capacity) };
                if !buffer.is_null() {
                    let base = unsafe { buffer.add(offset) };
                    append_pcm(base, size, encoding, src_channels, out_channels, &mut mixed);
                }
            }
            unsafe { AMediaCodec_releaseOutputBuffer(codec, index, false) };
            if is_eos {
                break;
            }
            continue;
        }

        let out_index = out_index as i32;
        if out_index == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED {
            let out_format = unsafe { AMediaCodec_getOutputFormat(codec) };
            if !out_format.is_null() {
                if let Some(r) = unsafe { format_i32(out_format, AMEDIAFORMAT_KEY_SAMPLE_RATE) }
                    .filter(|r| *r > 0)
                {
                    src_rate = r as u32;
                }
                if let Some(c) = unsafe { format_i32(out_format, AMEDIAFORMAT_KEY_CHANNEL_COUNT) }
                    .filter(|c| *c > 0)
                {
                    src_channels = c as usize;
                }
                if let Some(e) = unsafe { format_i32(out_format, AMEDIAFORMAT_KEY_PCM_ENCODING) } {
                    encoding = e;
                }
                unsafe { AMediaFormat_delete(out_format) };
            }
        } else if out_index == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED {
            // Deprecated no-op on the NDK path; keep pumping.
        } else if input_done {
            // INFO_TRY_AGAIN_LATER after EOS was queued: give the codec a few
            // more spins, then assume the stream produced nothing further.
            // (The EOS output flag breaks the loop in the `>= 0` arm.)
        }
    }

    Ok((mixed, src_rate, src_channels))
}

/// Convert one codec output buffer (`encoding` PCM, `src_channels` interleaved)
/// to interleaved `f32` mixed to `out_channels`, appending to `mixed`.
fn append_pcm(
    base: *const u8,
    size: usize,
    encoding: i32,
    src_channels: usize,
    out_channels: usize,
    mixed: &mut Vec<f32>,
) {
    let src_channels = src_channels.max(1);
    // Decode the raw PCM into interleaved f32 at the source channel count.
    let src: Vec<f32> = match encoding {
        ENCODING_PCM_FLOAT => {
            let count = size / 4;
            let bytes = unsafe { slice::from_raw_parts(base, count * 4) };
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        }
        // Default to 16-bit signed, the common MediaCodec PCM output.
        _ => {
            let count = size / 2;
            let bytes = unsafe { slice::from_raw_parts(base, count * 2) };
            bytes
                .chunks_exact(2)
                .map(|b| i16::from_ne_bytes([b[0], b[1]]) as f32 / 32768.0)
                .collect()
        }
    };

    for frame in src.chunks_exact(src_channels) {
        for c in 0..out_channels {
            // Mono source ⇒ duplicate; multichannel ⇒ take matching channels,
            // padding the high channels from channel 0 (MVP downmix).
            let sample = if src_channels == 1 {
                frame[0]
            } else if c < src_channels {
                frame[c]
            } else {
                frame[0]
            };
            mixed.push(sample);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_color_standard_to_primaries_and_matrix() {
        assert_eq!(
            map_color_standard(COLOR_STANDARD_BT709),
            (ColorPrimaries::Bt709, MatrixCoefficients::Bt709)
        );
        assert_eq!(
            map_color_standard(COLOR_STANDARD_BT601_PAL),
            (ColorPrimaries::Bt601_625, MatrixCoefficients::Bt601)
        );
        assert_eq!(
            map_color_standard(COLOR_STANDARD_BT601_NTSC),
            (ColorPrimaries::Bt601_525, MatrixCoefficients::Bt601)
        );
        assert_eq!(
            map_color_standard(COLOR_STANDARD_BT2020),
            (ColorPrimaries::Bt2020, MatrixCoefficients::Bt2020Ncl)
        );
        assert_eq!(
            map_color_standard(0),
            (ColorPrimaries::Unspecified, MatrixCoefficients::Unspecified)
        );
    }

    #[test]
    fn maps_color_transfer() {
        assert_eq!(
            map_color_transfer(COLOR_TRANSFER_SDR_VIDEO),
            TransferFunction::Bt709
        );
        assert_eq!(
            map_color_transfer(COLOR_TRANSFER_ST2084),
            TransferFunction::Pq
        );
        assert_eq!(
            map_color_transfer(COLOR_TRANSFER_HLG),
            TransferFunction::Hlg
        );
        assert_eq!(map_color_transfer(0), TransferFunction::Unspecified);
    }

    #[test]
    fn maps_color_range() {
        assert_eq!(map_color_range(COLOR_RANGE_FULL), ColorRange::Full);
        assert_eq!(map_color_range(2), ColorRange::Limited);
        assert_eq!(map_color_range(0), ColorRange::Limited);
    }

    #[test]
    fn maps_color_format_to_pixel_layout() {
        assert_eq!(
            color_format_to_pixel(COLOR_FORMAT_YUV420_SEMIPLANAR),
            PixelFormat::Nv12
        );
        assert_eq!(
            color_format_to_pixel(COLOR_FORMAT_YUV420_PLANAR),
            PixelFormat::Yuv420p
        );
        // Flexible (0x7F420888) and unknowns default to NV12.
        assert_eq!(color_format_to_pixel(0x7F42_0888), PixelFormat::Nv12);
    }

    #[test]
    fn us_conversion_matches_time_base() {
        // 3 ticks already on the microsecond base is exactly 3 us.
        let exact = RationalTime::new(3, Rational::new(US_PER_SEC as i32, 1));
        assert_eq!(to_us(exact), 3);
        // Frame 30 at 30 fps is one second = 1e6 us.
        let one_second = RationalTime::new(30, Rational::FPS_30);
        assert_eq!(to_us(one_second), US_PER_SEC);
        // Frame 5 at 30 fps: 5/30 s -> 166_666 us (truncated toward zero).
        let frame_5 = RationalTime::new(5, Rational::FPS_30);
        assert_eq!(to_us(frame_5), 166_666);
    }
}
