//! Android backend: MediaCodec (H.264 video + AAC audio encode) muxed to mp4
//! with `AMediaMuxer`, behind [`cutlass_core::VideoEncoder`].
//!
//! The mirror of the decoder's `android.rs`. The renderer hands us packed RGBA;
//! we convert to I420 ([`crate::yuv`]) and feed the video encoder, convert mixed
//! `f32` to S16 LPCM and feed the AAC encoder, and mux both elementary streams.
//!
//! MediaCodec/Muxer ordering constraint: a muxer track can only be added from an
//! encoder's *settled* output format (the `INFO_OUTPUT_FORMAT_CHANGED` that
//! carries codec-specific data), and `AMediaMuxer_start` must follow *all*
//! `addTrack`s. Encoded sample buffers that arrive before the muxer starts are
//! stashed in [`MediaCodecEncoder::pending`] and flushed on start.
//!
//! Device-untestable on the macOS host: validated via
//! `cargo check --target aarch64-linux-android` (same constraint as the
//! decoder's CPU path).

use core::ptr;
use core::slice;
use std::ffi::CString;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::Path;

use ndk_sys::{
    AMEDIACODEC_BUFFER_FLAG_CODEC_CONFIG, AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM,
    AMEDIACODEC_CONFIGURE_FLAG_ENCODE, AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED,
    AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED, AMEDIAFORMAT_KEY_AAC_PROFILE,
    AMEDIAFORMAT_KEY_BIT_RATE, AMEDIAFORMAT_KEY_CHANNEL_COUNT, AMEDIAFORMAT_KEY_COLOR_FORMAT,
    AMEDIAFORMAT_KEY_FRAME_RATE, AMEDIAFORMAT_KEY_HEIGHT, AMEDIAFORMAT_KEY_I_FRAME_INTERVAL,
    AMEDIAFORMAT_KEY_MIME, AMEDIAFORMAT_KEY_SAMPLE_RATE, AMEDIAFORMAT_KEY_WIDTH, AMediaCodec,
    AMediaCodec_configure, AMediaCodec_createEncoderByType, AMediaCodec_delete,
    AMediaCodec_dequeueInputBuffer, AMediaCodec_dequeueOutputBuffer, AMediaCodec_getInputBuffer,
    AMediaCodec_getOutputBuffer, AMediaCodec_getOutputFormat, AMediaCodec_queueInputBuffer,
    AMediaCodec_releaseOutputBuffer, AMediaCodec_start, AMediaCodec_stop, AMediaCodecBufferInfo,
    AMediaFormat, AMediaFormat_delete, AMediaFormat_new, AMediaFormat_setFloat,
    AMediaFormat_setInt32, AMediaFormat_setString, AMediaMuxer, AMediaMuxer_addTrack,
    AMediaMuxer_delete, AMediaMuxer_new, AMediaMuxer_start, AMediaMuxer_stop,
    AMediaMuxer_writeSampleData, OutputFormat, media_status_t,
};

use cutlass_core::{
    AudioEncoderConfig, EncodeError, EncoderConfig, PixelFormat, RationalTime, VideoEncoder,
    VideoFrame,
};

use crate::yuv::{YuvMatrix, matrix_for, rgba_to_i420};

/// MediaCodec/MediaExtractor timestamp tick rate.
const US_PER_SEC: i64 = 1_000_000;
/// Don't block waiting for a free input buffer — drain output instead.
const INPUT_TIMEOUT_US: i64 = 10_000;
/// Short window to surface an output buffer each drain iteration.
const OUTPUT_TIMEOUT_US: i64 = 10_000;
/// Safety net on the finish-time drain so a wedged codec can't hang the export.
const MAX_DRAIN_ITERS: usize = 100_000;

/// `MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar` (I420 input).
const COLOR_FORMAT_YUV420_PLANAR: i32 = 19;
/// `MediaCodecInfo.CodecProfileLevel.AACObjectLC`.
const AAC_OBJECT_LC: i32 = 2;
/// Keyframe interval, seconds.
const I_FRAME_INTERVAL_SECS: i32 = 1;
/// AAC bitrate (128 kbps), matching the Apple backend.
const AAC_BIT_RATE: i32 = 128_000;

const MIME_H264: &str = "video/avc";
const MIME_AAC: &str = "audio/mp4a-latm";

/// Which elementary stream an encoded buffer belongs to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stream {
    Video,
    Audio,
}

/// An encoded sample held until the muxer has all its tracks and can start.
struct PendingSample {
    stream: Stream,
    data: Vec<u8>,
    pts_us: i64,
    flags: u32,
}

/// An H.264 (+ optional AAC) → mp4 encoder backed by MediaCodec + AMediaMuxer.
pub struct MediaCodecEncoder {
    /// Owns the output fd for the muxer's lifetime.
    _file: File,
    muxer: *mut AMediaMuxer,
    video: *mut AMediaCodec,
    /// Null when `config.audio` was `None`.
    audio_codec: *mut AMediaCodec,
    audio: Option<AudioEncoderConfig>,
    size: (u32, u32),
    /// RGB→YUV coefficients for the I420 conversion, from `config.color`.
    matrix: YuvMatrix,
    /// Muxer track indices, `-1` until each codec's format is added.
    video_track: isize,
    audio_track: isize,
    muxer_started: bool,
    /// Encoded samples produced before the muxer started.
    pending: Vec<PendingSample>,
    finished: bool,
}

// SAFETY: native handles are owned and touched by a single export worker thread;
// `&self` is never shared across threads (`VideoEncoder: Send` without `Sync`).
unsafe impl Send for MediaCodecEncoder {}

/// Tears down half-built native objects if `open` bails partway through.
struct PartialCleanup {
    muxer: *mut AMediaMuxer,
    video: *mut AMediaCodec,
    audio: *mut AMediaCodec,
}

impl Drop for PartialCleanup {
    fn drop(&mut self) {
        unsafe {
            if !self.video.is_null() {
                AMediaCodec_delete(self.video);
            }
            if !self.audio.is_null() {
                AMediaCodec_delete(self.audio);
            }
            if !self.muxer.is_null() {
                AMediaMuxer_delete(self.muxer);
            }
        }
    }
}

impl MediaCodecEncoder {
    /// Create an mp4 encoder at `path` for `config`. Overwrites any existing
    /// file. Adds an AAC audio track when `config.audio` is `Some`.
    pub fn open(path: &Path, config: EncoderConfig) -> Result<Self, EncodeError> {
        let (width, height) = config.size;
        if width == 0 || height == 0 {
            return Err(EncodeError::unsupported("zero output dimensions"));
        }

        let file = File::create(path).map_err(|e| EncodeError::Start(e.to_string()))?;
        let muxer = unsafe {
            AMediaMuxer_new(
                file.as_raw_fd(),
                OutputFormat::AMEDIAMUXER_OUTPUT_FORMAT_MPEG_4,
            )
        };
        if muxer.is_null() {
            return Err(EncodeError::Start("AMediaMuxer_new returned null".into()));
        }
        let mut guard = PartialCleanup {
            muxer,
            video: ptr::null_mut(),
            audio: ptr::null_mut(),
        };

        let video = open_video_encoder(width, height, config.frame_rate)?;
        guard.video = video;

        let audio_codec = match config.audio {
            Some(audio) => {
                let codec = open_audio_encoder(audio)?;
                guard.audio = codec;
                codec
            }
            None => ptr::null_mut(),
        };

        // Ownership passes to the encoder; its Drop now owns teardown.
        core::mem::forget(guard);
        Ok(Self {
            _file: file,
            muxer,
            video,
            audio_codec,
            audio: config.audio,
            size: (width, height),
            matrix: matrix_for(&config.color),
            video_track: -1,
            audio_track: -1,
            muxer_started: false,
            pending: Vec::new(),
            finished: false,
        })
    }

    /// Feed one raw input buffer into `codec`, returning whether a buffer was
    /// accepted (so callers can retry on backpressure).
    fn queue_input(
        &self,
        codec: *mut AMediaCodec,
        bytes: &[u8],
        pts_us: i64,
        eos: bool,
    ) -> Result<bool, EncodeError> {
        let in_index = unsafe { AMediaCodec_dequeueInputBuffer(codec, INPUT_TIMEOUT_US) };
        if in_index < 0 {
            return Ok(false);
        }
        let index = in_index as usize;
        let mut capacity: usize = 0;
        let buffer = unsafe { AMediaCodec_getInputBuffer(codec, index, &mut capacity) };
        if buffer.is_null() {
            return Err(EncodeError::Encode("null encoder input buffer".into()));
        }
        if bytes.len() > capacity {
            return Err(EncodeError::Encode(format!(
                "encoded input ({} bytes) exceeds buffer capacity ({capacity})",
                bytes.len()
            )));
        }
        if !bytes.is_empty() {
            unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), buffer, bytes.len()) };
        }
        let flags = if eos {
            AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM
        } else {
            0
        };
        let status = unsafe {
            AMediaCodec_queueInputBuffer(codec, index, 0, bytes.len(), pts_us.max(0) as u64, flags)
        };
        if !media_ok(status) {
            return Err(EncodeError::Encode("queueInputBuffer failed".into()));
        }
        Ok(true)
    }

    /// Feed `bytes` to `codec`, spinning until a free input buffer accepts it
    /// (draining output between attempts to relieve backpressure).
    fn queue_input_blocking(
        &mut self,
        codec: *mut AMediaCodec,
        stream: Stream,
        bytes: &[u8],
        pts_us: i64,
    ) -> Result<(), EncodeError> {
        for _ in 0..MAX_DRAIN_ITERS {
            if self.queue_input(codec, bytes, pts_us, false)? {
                return Ok(());
            }
            self.drain(codec, stream)?;
        }
        Err(EncodeError::Encode("encoder never accepted input".into()))
    }

    /// Drain available encoded output from `codec`, muxing or stashing it.
    /// Returns `true` once the stream's end-of-stream output is seen.
    fn drain(&mut self, codec: *mut AMediaCodec, stream: Stream) -> Result<bool, EncodeError> {
        let mut info = AMediaCodecBufferInfo {
            offset: 0,
            size: 0,
            presentationTimeUs: 0,
            flags: 0,
        };
        loop {
            let out_index =
                unsafe { AMediaCodec_dequeueOutputBuffer(codec, &mut info, OUTPUT_TIMEOUT_US) };
            if out_index >= 0 {
                let index = out_index as usize;
                let is_eos = info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM != 0;
                let is_config = info.flags & AMEDIACODEC_BUFFER_FLAG_CODEC_CONFIG != 0;
                let size = info.size.max(0) as usize;

                // Codec-config (CSD) is carried in the track format; never muxed.
                if !is_config && size > 0 {
                    let mut capacity: usize = 0;
                    let buffer =
                        unsafe { AMediaCodec_getOutputBuffer(codec, index, &mut capacity) };
                    if !buffer.is_null() {
                        let offset = info.offset.max(0) as usize;
                        let src = unsafe { slice::from_raw_parts(buffer.add(offset), size) };
                        self.emit(stream, src, info.presentationTimeUs, info.flags)?;
                    }
                }
                unsafe { AMediaCodec_releaseOutputBuffer(codec, index, false) };
                if is_eos {
                    return Ok(true);
                }
                continue;
            }

            let out_index = out_index as i32;
            if out_index == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED {
                let format = unsafe { AMediaCodec_getOutputFormat(codec) };
                if !format.is_null() {
                    let track = unsafe { AMediaMuxer_addTrack(self.muxer, format) };
                    unsafe { AMediaFormat_delete(format) };
                    if track < 0 {
                        return Err(EncodeError::Encode("AMediaMuxer_addTrack failed".into()));
                    }
                    match stream {
                        Stream::Video => self.video_track = track,
                        Stream::Audio => self.audio_track = track,
                    }
                    self.maybe_start_muxer()?;
                }
            } else if out_index == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED {
                // Deprecated no-op on the NDK path.
            } else {
                // INFO_TRY_AGAIN_LATER: nothing ready right now.
                return Ok(false);
            }
        }
    }

    /// Mux an encoded sample, or stash it if the muxer hasn't started yet.
    fn emit(
        &mut self,
        stream: Stream,
        data: &[u8],
        pts_us: i64,
        flags: u32,
    ) -> Result<(), EncodeError> {
        if self.muxer_started {
            let track = match stream {
                Stream::Video => self.video_track,
                Stream::Audio => self.audio_track,
            };
            self.write_sample(track, data, pts_us, flags)
        } else {
            self.pending.push(PendingSample {
                stream,
                data: data.to_vec(),
                pts_us,
                flags,
            });
            Ok(())
        }
    }

    /// Write one encoded sample to `track`.
    fn write_sample(
        &self,
        track: isize,
        data: &[u8],
        pts_us: i64,
        flags: u32,
    ) -> Result<(), EncodeError> {
        if track < 0 {
            return Err(EncodeError::Encode(
                "sample for an unregistered track".into(),
            ));
        }
        let info = AMediaCodecBufferInfo {
            offset: 0,
            size: data.len() as i32,
            presentationTimeUs: pts_us,
            flags: flags & !AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM,
        };
        let status = unsafe {
            AMediaMuxer_writeSampleData(self.muxer, track as usize, data.as_ptr(), &info)
        };
        if !media_ok(status) {
            return Err(EncodeError::Encode(
                "AMediaMuxer_writeSampleData failed".into(),
            ));
        }
        Ok(())
    }

    /// Start the muxer and flush stashed samples once every track is registered.
    fn maybe_start_muxer(&mut self) -> Result<(), EncodeError> {
        if self.muxer_started {
            return Ok(());
        }
        let audio_ready = self.audio_codec.is_null() || self.audio_track >= 0;
        if self.video_track < 0 || !audio_ready {
            return Ok(());
        }
        if !media_ok(unsafe { AMediaMuxer_start(self.muxer) }) {
            return Err(EncodeError::Encode("AMediaMuxer_start failed".into()));
        }
        self.muxer_started = true;
        for sample in std::mem::take(&mut self.pending) {
            let track = match sample.stream {
                Stream::Video => self.video_track,
                Stream::Audio => self.audio_track,
            };
            self.write_sample(track, &sample.data, sample.pts_us, sample.flags)?;
        }
        Ok(())
    }
}

impl VideoEncoder for MediaCodecEncoder {
    fn push(&mut self, frame: &VideoFrame) -> Result<(), EncodeError> {
        let (width, height) = self.size;
        if (frame.width(), frame.height()) != (width, height) {
            return Err(EncodeError::Encode(format!(
                "frame {}x{} does not match output {width}x{height}",
                frame.width(),
                frame.height()
            )));
        }
        let cpu = frame
            .cpu()
            .ok_or_else(|| EncodeError::unsupported("encoder needs CPU frames"))?;
        let plane = cpu
            .planes
            .first()
            .ok_or_else(|| EncodeError::Encode("frame has no plane".into()))?;
        // RGBA in, BGRA gets swapped to RGBA order first so luma/chroma math is
        // correct regardless of the source channel order.
        let rgba = match frame.format {
            PixelFormat::Rgba8 => {
                rgba_to_i420(&plane.data, width, height, plane.stride, self.matrix)
            }
            PixelFormat::Bgra8 => {
                let swapped = swap_bgra_to_rgba(&plane.data, width, height, plane.stride);
                rgba_to_i420(&swapped, width, height, (width * 4) as usize, self.matrix)
            }
            other => {
                return Err(EncodeError::unsupported(format!(
                    "encoder expects Rgba8/Bgra8 frames, got {other:?}"
                )));
            }
        };

        let pts_us = to_us(frame.pts);
        let video = self.video;
        self.queue_input_blocking(video, Stream::Video, &rgba, pts_us)?;
        self.drain(video, Stream::Video)?;
        Ok(())
    }

    fn push_audio(&mut self, samples: &[f32], pts: RationalTime) -> Result<(), EncodeError> {
        let cfg = self
            .audio
            .ok_or_else(|| EncodeError::unsupported("this encoder has no audio track"))?;
        if self.audio_codec.is_null() {
            return Err(EncodeError::unsupported("this encoder has no audio track"));
        }
        if samples.is_empty() {
            return Ok(());
        }
        let channels = usize::from(cfg.channels);
        if channels == 0 || samples.len() % channels != 0 {
            return Err(EncodeError::Encode(
                "audio block length is not a frame multiple".into(),
            ));
        }

        // Interleaved f32 → S16 LE.
        let mut pcm = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
            pcm.extend_from_slice(&v.to_le_bytes());
        }

        let pts_us = to_us(pts);
        let codec = self.audio_codec;
        self.queue_input_blocking(codec, Stream::Audio, &pcm, pts_us)?;
        self.drain(codec, Stream::Audio)?;
        Ok(())
    }

    fn finish(&mut self) -> Result<(), EncodeError> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;

        // Signal end-of-stream to both encoders, then drain to their EOS output.
        let video = self.video;
        for _ in 0..MAX_DRAIN_ITERS {
            if self.queue_input(video, &[], 0, true)? {
                break;
            }
            self.drain(video, Stream::Video)?;
        }
        for _ in 0..MAX_DRAIN_ITERS {
            if self.drain(video, Stream::Video)? {
                break;
            }
        }

        if !self.audio_codec.is_null() {
            let audio = self.audio_codec;
            for _ in 0..MAX_DRAIN_ITERS {
                if self.queue_input(audio, &[], 0, true)? {
                    break;
                }
                self.drain(audio, Stream::Audio)?;
            }
            for _ in 0..MAX_DRAIN_ITERS {
                if self.drain(audio, Stream::Audio)? {
                    break;
                }
            }
        }

        // A track that never produced a format-change can't be muxed; this only
        // happens for an empty stream, which we treat as a start failure.
        self.maybe_start_muxer()?;
        if !self.muxer_started {
            return Err(EncodeError::Encode(
                "muxer never started (no encoded output)".into(),
            ));
        }
        if !media_ok(unsafe { AMediaMuxer_stop(self.muxer) }) {
            return Err(EncodeError::Encode("AMediaMuxer_stop failed".into()));
        }
        Ok(())
    }
}

impl Drop for MediaCodecEncoder {
    fn drop(&mut self) {
        unsafe {
            if !self.video.is_null() {
                AMediaCodec_stop(self.video);
                AMediaCodec_delete(self.video);
            }
            if !self.audio_codec.is_null() {
                AMediaCodec_stop(self.audio_codec);
                AMediaCodec_delete(self.audio_codec);
            }
            if !self.muxer.is_null() {
                AMediaMuxer_delete(self.muxer);
            }
        }
    }
}

/// Configure and start the H.264 video encoder.
fn open_video_encoder(
    width: u32,
    height: u32,
    frame_rate: cutlass_core::Rational,
) -> Result<*mut AMediaCodec, EncodeError> {
    let mime = CString::new(MIME_H264).unwrap();
    let codec = unsafe { AMediaCodec_createEncoderByType(mime.as_ptr()) };
    if codec.is_null() {
        return Err(EncodeError::Start(format!("no encoder for {MIME_H264}")));
    }
    let fps = (frame_rate.num.max(1) as f64 / frame_rate.den.max(1) as f64).round() as i32;
    let fps = fps.max(1);
    // Bitrate heuristic: ~0.1 bits per pixel·second, clamped to a sane floor.
    let bitrate = (((width as u64) * (height as u64) * (fps as u64)) as f64 * 0.1).round() as i32;
    let bitrate = bitrate.max(1_000_000);

    let format = unsafe { AMediaFormat_new() };
    if format.is_null() {
        unsafe { AMediaCodec_delete(codec) };
        return Err(EncodeError::Start("AMediaFormat_new failed".into()));
    }
    unsafe {
        set_string(format, AMEDIAFORMAT_KEY_MIME, MIME_H264);
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_WIDTH, width as i32);
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_HEIGHT, height as i32);
        AMediaFormat_setInt32(
            format,
            AMEDIAFORMAT_KEY_COLOR_FORMAT,
            COLOR_FORMAT_YUV420_PLANAR,
        );
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_BIT_RATE, bitrate);
        AMediaFormat_setFloat(format, AMEDIAFORMAT_KEY_FRAME_RATE, fps as f32);
        AMediaFormat_setInt32(
            format,
            AMEDIAFORMAT_KEY_I_FRAME_INTERVAL,
            I_FRAME_INTERVAL_SECS,
        );
    }

    let status = unsafe {
        AMediaCodec_configure(
            codec,
            format,
            ptr::null_mut(),
            ptr::null_mut(),
            AMEDIACODEC_CONFIGURE_FLAG_ENCODE as u32,
        )
    };
    unsafe { AMediaFormat_delete(format) };
    if !media_ok(status) {
        unsafe { AMediaCodec_delete(codec) };
        return Err(EncodeError::Start(
            "video AMediaCodec_configure failed".into(),
        ));
    }
    if !media_ok(unsafe { AMediaCodec_start(codec) }) {
        unsafe { AMediaCodec_delete(codec) };
        return Err(EncodeError::Start("video AMediaCodec_start failed".into()));
    }
    Ok(codec)
}

/// Configure and start the AAC audio encoder.
fn open_audio_encoder(audio: AudioEncoderConfig) -> Result<*mut AMediaCodec, EncodeError> {
    let mime = CString::new(MIME_AAC).unwrap();
    let codec = unsafe { AMediaCodec_createEncoderByType(mime.as_ptr()) };
    if codec.is_null() {
        return Err(EncodeError::Start(format!("no encoder for {MIME_AAC}")));
    }
    let format = unsafe { AMediaFormat_new() };
    if format.is_null() {
        unsafe { AMediaCodec_delete(codec) };
        return Err(EncodeError::Start("AMediaFormat_new failed".into()));
    }
    unsafe {
        set_string(format, AMEDIAFORMAT_KEY_MIME, MIME_AAC);
        AMediaFormat_setInt32(
            format,
            AMEDIAFORMAT_KEY_SAMPLE_RATE,
            audio.sample_rate as i32,
        );
        AMediaFormat_setInt32(
            format,
            AMEDIAFORMAT_KEY_CHANNEL_COUNT,
            i32::from(audio.channels),
        );
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_BIT_RATE, AAC_BIT_RATE);
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_AAC_PROFILE, AAC_OBJECT_LC);
    }

    let status = unsafe {
        AMediaCodec_configure(
            codec,
            format,
            ptr::null_mut(),
            ptr::null_mut(),
            AMEDIACODEC_CONFIGURE_FLAG_ENCODE as u32,
        )
    };
    unsafe { AMediaFormat_delete(format) };
    if !media_ok(status) {
        unsafe { AMediaCodec_delete(codec) };
        return Err(EncodeError::Start(
            "audio AMediaCodec_configure failed".into(),
        ));
    }
    if !media_ok(unsafe { AMediaCodec_start(codec) }) {
        unsafe { AMediaCodec_delete(codec) };
        return Err(EncodeError::Start("audio AMediaCodec_start failed".into()));
    }
    Ok(codec)
}

/// Swap BGRA bytes to RGBA order into a tightly-packed buffer.
fn swap_bgra_to_rgba(data: &[u8], width: u32, height: u32, stride: usize) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut out = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let s = y * stride + x * 4;
            let d = (y * w + x) * 4;
            out[d] = data[s + 2];
            out[d + 1] = data[s + 1];
            out[d + 2] = data[s];
            out[d + 3] = data[s + 3];
        }
    }
    out
}

/// Set a string key on a format from a Rust `&str`.
///
/// # Safety
/// `format` and `key` must be valid.
unsafe fn set_string(format: *mut AMediaFormat, key: *const core::ffi::c_char, value: &str) {
    if let Ok(c) = CString::new(value) {
        unsafe { AMediaFormat_setString(format, key, c.as_ptr()) };
    }
}

/// Convert a [`RationalTime`] to microsecond ticks.
fn to_us(t: RationalTime) -> i64 {
    let num = i128::from(t.rate.num.max(1));
    let den = i128::from(t.rate.den.max(1));
    ((i128::from(t.value) * den * i128::from(US_PER_SEC)) / num) as i64
}

/// `true` when a `media_status_t` is `AMEDIA_OK` (0).
fn media_ok(status: media_status_t) -> bool {
    status.0 == 0
}
