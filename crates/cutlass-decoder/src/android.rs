//! Android backend: `AMediaExtractor` (demux) + `AMediaCodec` (MediaCodec
//! hardware decode) behind [`cutlass_core::VideoDecoder`].
//!
//! The extractor demuxes the first video track; MediaCodec decodes its packets
//! to CPU `ByteBuffer`s, which we copy into [`cutlass_core::VideoFrame`] YUV
//! planes the compositor's `decoded_to_yuv_layer` ingests. Seeking repositions
//! the extractor at the keyframe at/before the target and flushes the codec, so
//! a forward `next_frame` walk reaches the target — the same contract the Apple
//! backend honours by rebuilding its reader.
//!
//! This is the FFmpeg-free path for Android (no libav* link, no AVC/HEVC
//! royalties). It is **device-untested** from the desktop host — it is built and
//! type-checked against the `aarch64-linux-android` target; runtime validation
//! needs a device/emulator.
//!
//! Output is CPU planes only for now: we request `COLOR_FormatYUV420Planar` and
//! handle planar (I420) / semi-planar (NV12) byte layouts. Devices that only
//! vend `COLOR_FormatYUV420Flexible` aren't readable from a raw `ByteBuffer`;
//! that needs the `AImageReader`/Surface path (future zero-copy work), and is
//! reported as [`DecodeError::Unsupported`] rather than mis-decoded.

use std::ffi::CString;
use std::path::Path;
use std::ptr::NonNull;
use std::time::Duration;

use ndk::media::media_codec::{
    DequeuedInputBufferResult, DequeuedOutputBufferInfoResult, MediaCodec, MediaCodecDirection,
};
use ndk::media::media_format::MediaFormat;
use ndk_sys as ffi;

use cutlass_core::{
    ColorPrimaries, ColorRange, ColorSpace, CpuImage, DecodeError, FrameData, MatrixCoefficients,
    PixelFormat, Plane, Rational, RationalTime, Rect, Rotation, SourceInfo, TransferFunction,
    VideoDecoder, VideoFrame,
};

use crate::OutputMode;

/// PTS time base: MediaCodec reports presentation times in microseconds.
const MICROS_PER_SEC: i32 = 1_000_000;

/// `MediaFormat` color-format we request and the two raw `ByteBuffer` layouts
/// we can read (the OMX color-format constants).
const COLOR_FORMAT_YUV420_PLANAR: i32 = 19; // I420: Y, U, V
const COLOR_FORMAT_YUV420_SEMIPLANAR: i32 = 21; // NV12: Y, interleaved UV

/// A decoded video source backed by Android MediaCodec.
pub struct AndroidDecoder {
    /// Owned demuxer; deleted on drop.
    extractor: NonNull<ffi::AMediaExtractor>,
    codec: MediaCodec,
    info: SourceInfo,
    #[allow(dead_code)]
    mode: OutputMode,
    /// EOS has been signalled into the codec input — stop feeding.
    input_done: bool,
    /// EOS has drained out of the codec output — stream is exhausted.
    output_done: bool,
}

// SAFETY: the extractor and codec are owned and touched by a single thread at a
// time — the decode-ahead worker the engine parks this decoder on. We never
// share `&self` across threads, satisfying `VideoDecoder: Send` without `Sync`.
unsafe impl Send for AndroidDecoder {}

impl AndroidDecoder {
    /// Open `path` and configure a decoder for its first video track.
    pub fn open(path: &Path, mode: OutputMode) -> Result<Self, DecodeError> {
        let path_str = path
            .to_str()
            .ok_or_else(|| DecodeError::unsupported("path is not valid UTF-8"))?;
        let c_path = CString::new(path_str)
            .map_err(|_| DecodeError::unsupported("path contains an interior NUL"))?;

        let extractor = NonNull::new(unsafe { ffi::AMediaExtractor_new() })
            .ok_or_else(|| DecodeError::Open("AMediaExtractor_new returned null".into()))?;

        // Wrap immediately so any early return deletes the extractor.
        let mut this_extractor = ExtractorGuard(extractor);

        media_ok(
            unsafe {
                ffi::AMediaExtractor_setDataSource(this_extractor.0.as_ptr(), c_path.as_ptr())
            },
            "AMediaExtractor_setDataSource",
        )?;

        let (track_index, mut format) = first_video_track(this_extractor.0)?;

        let mime = format
            .str("mime")
            .ok_or_else(|| DecodeError::unsupported("video track has no mime type"))?
            .to_owned();

        let info = probe(&format);

        media_ok(
            unsafe { ffi::AMediaExtractor_selectTrack(this_extractor.0.as_ptr(), track_index) },
            "AMediaExtractor_selectTrack",
        )?;

        let codec = MediaCodec::from_decoder_type(&mime)
            .ok_or_else(|| DecodeError::unsupported(format!("no decoder for mime {mime}")))?;

        // Ask for planar YUV so the output `ByteBuffer` is one of the layouts we
        // can read. Decoders may ignore this and vend their native format; the
        // per-frame format check then either reads it or reports unsupported.
        format.set_i32("color-format", COLOR_FORMAT_YUV420_PLANAR);

        codec
            .configure(&format, None, MediaCodecDirection::Decoder)
            .map_err(|e| DecodeError::Open(format!("MediaCodec configure: {e}")))?;
        codec
            .start()
            .map_err(|e| DecodeError::Open(format!("MediaCodec start: {e}")))?;

        // Ownership of the extractor moves into the decoder now; defuse the guard.
        let extractor = this_extractor.0;
        this_extractor.0 = NonNull::dangling();
        std::mem::forget(this_extractor);

        Ok(Self {
            extractor,
            codec,
            info,
            mode,
            input_done: false,
            output_done: false,
        })
    }

    /// Feed one demuxed packet into the codec, signalling EOS at end of stream.
    /// Returns `false` when there is no free input buffer right now (the caller
    /// should try draining output instead).
    fn pump_input(&mut self) -> Result<bool, DecodeError> {
        if self.input_done {
            return Ok(false);
        }
        let input = match self
            .codec
            .dequeue_input_buffer(Duration::ZERO)
            .map_err(|e| DecodeError::Decode(format!("dequeue_input_buffer: {e}")))?
        {
            DequeuedInputBufferResult::Buffer(input) => input,
            DequeuedInputBufferResult::TryAgainLater => return Ok(false),
        };

        let mut input = input;
        let buf = input.buffer_mut();
        let read = unsafe {
            ffi::AMediaExtractor_readSampleData(
                self.extractor.as_ptr(),
                buf.as_mut_ptr().cast::<u8>(),
                buf.len(),
            )
        };

        if read < 0 {
            self.codec
                .queue_input_buffer(input, 0, 0, 0, ffi::AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM)
                .map_err(|e| DecodeError::Decode(format!("queue EOS: {e}")))?;
            self.input_done = true;
        } else {
            let pts_us = unsafe { ffi::AMediaExtractor_getSampleTime(self.extractor.as_ptr()) };
            self.codec
                .queue_input_buffer(input, 0, read as usize, pts_us.max(0) as u64, 0)
                .map_err(|e| DecodeError::Decode(format!("queue_input_buffer: {e}")))?;
            unsafe { ffi::AMediaExtractor_advance(self.extractor.as_ptr()) };
        }
        Ok(true)
    }

    /// Try to drain one decoded frame from the codec output.
    fn drain_output(&mut self) -> Result<DrainResult, DecodeError> {
        let output = match self
            .codec
            .dequeue_output_buffer(Duration::from_millis(10))
            .map_err(|e| DecodeError::Decode(format!("dequeue_output_buffer: {e}")))?
        {
            DequeuedOutputBufferInfoResult::Buffer(output) => output,
            DequeuedOutputBufferInfoResult::TryAgainLater
            | DequeuedOutputBufferInfoResult::OutputFormatChanged
            | DequeuedOutputBufferInfoResult::OutputBuffersChanged => {
                return Ok(DrainResult::Again);
            }
        };

        let info = output.info();
        let flags = info.flags();
        let size = info.size();
        let offset = info.offset();
        let pts_us = info.presentation_time_us();

        if flags & ffi::AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM != 0 {
            self.codec
                .release_output_buffer(output, false)
                .map_err(|e| DecodeError::Decode(format!("release EOS: {e}")))?;
            self.output_done = true;
            return Ok(DrainResult::EndOfStream);
        }

        if size <= 0 {
            self.codec
                .release_output_buffer(output, false)
                .map_err(|e| DecodeError::Decode(format!("release empty: {e}")))?;
            return Ok(DrainResult::Again);
        }

        // The negotiated output format (stride / slice-height / color-format) is
        // stable once decoding starts; the codec-level getter avoids the
        // api-level-28-gated per-buffer `OutputBuffer::format`.
        let fmt = self.codec.output_format();
        let frame = {
            let buffer = output.buffer();
            let start = offset.max(0) as usize;
            let end = start
                .checked_add(size as usize)
                .filter(|&e| e <= buffer.len())
                .ok_or_else(|| DecodeError::Decode("output buffer bounds out of range".into()))?;
            self.frame_from_bytes(&fmt, &buffer[start..end], pts_us)?
        };

        self.codec
            .release_output_buffer(output, false)
            .map_err(|e| DecodeError::Decode(format!("release_output_buffer: {e}")))?;
        Ok(DrainResult::Frame(Box::new(frame)))
    }

    /// Copy one decoded `ByteBuffer` into a [`VideoFrame`].
    fn frame_from_bytes(
        &self,
        fmt: &MediaFormat,
        data: &[u8],
        pts_us: i64,
    ) -> Result<VideoFrame, DecodeError> {
        let width = fmt
            .i32("width")
            .map(|w| w as u32)
            .unwrap_or(self.info.display_size.0);
        let height = fmt
            .i32("height")
            .map(|h| h as u32)
            .unwrap_or(self.info.display_size.1);
        let stride = fmt.i32("stride").filter(|&s| s > 0).unwrap_or(width as i32) as usize;
        let slice_height = fmt
            .i32("slice-height")
            .filter(|&s| s > 0)
            .unwrap_or(height as i32) as usize;
        let color_format = fmt.i32("color-format").unwrap_or(COLOR_FORMAT_YUV420_PLANAR);

        let (format, planes) = build_planes(color_format, data, stride, slice_height)?;

        let color = ColorSpace {
            range: fmt
                .i32("color-range")
                .map(map_color_range)
                .unwrap_or(self.info.color.range),
            ..self.info.color
        };

        let pts = RationalTime::new(pts_us.max(0), Rational::new(MICROS_PER_SEC, 1));

        Ok(VideoFrame::new(
            pts,
            format,
            color,
            (width, height),
            Rect::from_size(width, height),
            self.info.rotation,
            FrameData::Cpu(CpuImage::new(planes)),
        ))
    }
}

impl VideoDecoder for AndroidDecoder {
    fn info(&self) -> &SourceInfo {
        &self.info
    }

    fn seek(&mut self, target: RationalTime) -> Result<(), DecodeError> {
        let pos_us = (target.seconds() * f64::from(MICROS_PER_SEC)).round().max(0.0) as i64;
        media_ok(
            unsafe {
                ffi::AMediaExtractor_seekTo(
                    self.extractor.as_ptr(),
                    pos_us,
                    ffi::SeekMode::AMEDIAEXTRACTOR_SEEK_PREVIOUS_SYNC,
                )
            },
            "AMediaExtractor_seekTo",
        )?;
        self.codec
            .flush()
            .map_err(|e| DecodeError::Seek(format!("MediaCodec flush: {e}")))?;
        self.input_done = false;
        self.output_done = false;
        Ok(())
    }

    fn next_frame(&mut self) -> Result<Option<VideoFrame>, DecodeError> {
        if self.output_done {
            return Ok(None);
        }
        loop {
            // Keep the codec fed; ignore "no input buffer right now" and drain.
            while self.pump_input()? {}
            match self.drain_output()? {
                DrainResult::Frame(frame) => return Ok(Some(*frame)),
                DrainResult::EndOfStream => return Ok(None),
                DrainResult::Again => {
                    if self.input_done && self.output_done {
                        return Ok(None);
                    }
                    // Loop again: more input to feed and/or output still pending.
                }
            }
        }
    }
}

impl Drop for AndroidDecoder {
    fn drop(&mut self) {
        unsafe { ffi::AMediaExtractor_delete(self.extractor.as_ptr()) };
    }
}

/// Outcome of one [`AndroidDecoder::drain_output`] attempt.
enum DrainResult {
    /// A decoded frame (boxed to keep the enum small).
    Frame(Box<VideoFrame>),
    /// Nothing ready yet, or a format/buffer-change notice — retry.
    Again,
    /// The codec drained its end-of-stream marker.
    EndOfStream,
}

/// Deletes an [`AMediaExtractor`] on drop unless its pointer is defused
/// (set to dangling) after ownership transfers into the decoder.
struct ExtractorGuard(NonNull<ffi::AMediaExtractor>);

impl Drop for ExtractorGuard {
    fn drop(&mut self) {
        if self.0 != NonNull::dangling() {
            unsafe { ffi::AMediaExtractor_delete(self.0.as_ptr()) };
        }
    }
}

/// Find the first track whose mime type is `video/*`; return its index and
/// (owned) format. The format the extractor vends must be deleted by us, which
/// [`MediaFormat`]'s `Drop` handles.
fn first_video_track(
    extractor: NonNull<ffi::AMediaExtractor>,
) -> Result<(usize, MediaFormat), DecodeError> {
    let count = unsafe { ffi::AMediaExtractor_getTrackCount(extractor.as_ptr()) };
    for index in 0..count {
        let ptr = unsafe { ffi::AMediaExtractor_getTrackFormat(extractor.as_ptr(), index) };
        let Some(ptr) = NonNull::new(ptr) else {
            continue;
        };
        let mut format = unsafe { MediaFormat::from_ptr(ptr) };
        if format
            .str("mime")
            .is_some_and(|m| m.starts_with("video/"))
        {
            return Ok((index, format));
        }
    }
    Err(DecodeError::unsupported("no video track found"))
}

/// Read static source properties from the selected track's format.
fn probe(format: &MediaFormat) -> SourceInfo {
    let width = format.i32("width").unwrap_or(0).max(0) as u32;
    let height = format.i32("height").unwrap_or(0).max(0) as u32;
    let rotation = match format.i32("rotation-degrees").unwrap_or(0).rem_euclid(360) {
        90 => Rotation::Cw90,
        180 => Rotation::Cw180,
        270 => Rotation::Cw270,
        _ => Rotation::None,
    };
    let frame_rate = format
        .i32("frame-rate")
        .filter(|&r| r > 0)
        .map(|r| Rational::new(r, 1))
        .or_else(|| {
            format
                .f32("frame-rate")
                .filter(|&r| r > 0.0)
                .map(|r| Rational::new((r * 1000.0).round() as i32, 1000))
        })
        .unwrap_or(Rational::FPS_30);
    let duration = format
        .i64("durationUs")
        .filter(|&d| d > 0)
        .map(|d| RationalTime::new(d, Rational::new(MICROS_PER_SEC, 1)));

    SourceInfo {
        coded_size: (width, height),
        display_size: (width, height),
        rotation,
        pixel_format: PixelFormat::Nv12,
        color: color_from_format(format),
        frame_rate,
        time_base: Rational::new(MICROS_PER_SEC, 1),
        duration,
    }
}

/// Map the MediaFormat color keys (API 24+) to a [`ColorSpace`], defaulting to
/// BT.709 when a key is absent or unrecognized.
fn color_from_format(format: &MediaFormat) -> ColorSpace {
    let mut color = ColorSpace::BT709;
    if let Some(range) = format.i32("color-range") {
        color.range = map_color_range(range);
    }
    // MediaFormat.COLOR_STANDARD_*: 1=BT709, 2=BT601_PAL, 4=BT601_NTSC, 6=BT2020.
    if let Some(standard) = format.i32("color-standard") {
        match standard {
            2 => {
                color.primaries = ColorPrimaries::Bt601_625;
                color.matrix = MatrixCoefficients::Bt601;
            }
            4 => {
                color.primaries = ColorPrimaries::Bt601_525;
                color.matrix = MatrixCoefficients::Bt601;
            }
            6 => {
                color.primaries = ColorPrimaries::Bt2020;
                color.matrix = MatrixCoefficients::Bt2020Ncl;
            }
            _ => {}
        }
    }
    // MediaFormat.COLOR_TRANSFER_*: 1=LINEAR, 3=SDR_VIDEO, 6=ST2084, 7=HLG.
    if let Some(transfer) = format.i32("color-transfer") {
        color.transfer = match transfer {
            6 => TransferFunction::Pq,
            7 => TransferFunction::Hlg,
            _ => TransferFunction::Bt709,
        };
    }
    color
}

/// MediaFormat.COLOR_RANGE_*: 1=FULL, 2=LIMITED.
fn map_color_range(range: i32) -> ColorRange {
    match range {
        1 => ColorRange::Full,
        _ => ColorRange::Limited,
    }
}

/// Slice a decoded `ByteBuffer` into [`Plane`]s per the codec's color format.
/// Only the raw planar (I420) and semi-planar (NV12) 4:2:0 layouts are readable
/// from a `ByteBuffer`; flexible/vendor formats need the ImageReader path.
fn build_planes(
    color_format: i32,
    data: &[u8],
    stride: usize,
    slice_height: usize,
) -> Result<(PixelFormat, Vec<Plane>), DecodeError> {
    let y_len = stride * slice_height;
    match color_format {
        COLOR_FORMAT_YUV420_PLANAR => {
            let c_stride = stride / 2;
            let c_height = slice_height / 2;
            let c_len = c_stride * c_height;
            let need = y_len + 2 * c_len;
            if data.len() < need {
                return Err(DecodeError::Decode(format!(
                    "I420 buffer short: {} < {need}",
                    data.len()
                )));
            }
            let y = data[..y_len].to_vec();
            let u = data[y_len..y_len + c_len].to_vec();
            let v = data[y_len + c_len..need].to_vec();
            Ok((
                PixelFormat::Yuv420p,
                vec![
                    Plane::new(y, stride, slice_height),
                    Plane::new(u, c_stride, c_height),
                    Plane::new(v, c_stride, c_height),
                ],
            ))
        }
        COLOR_FORMAT_YUV420_SEMIPLANAR => {
            let uv_height = slice_height / 2;
            let uv_len = stride * uv_height;
            let need = y_len + uv_len;
            if data.len() < need {
                return Err(DecodeError::Decode(format!(
                    "NV12 buffer short: {} < {need}",
                    data.len()
                )));
            }
            let y = data[..y_len].to_vec();
            let uv = data[y_len..need].to_vec();
            Ok((
                PixelFormat::Nv12,
                vec![
                    Plane::new(y, stride, slice_height),
                    Plane::new(uv, stride, uv_height),
                ],
            ))
        }
        other => Err(DecodeError::unsupported(format!(
            "MediaCodec color format {other}: only planar/semi-planar YUV420 \
             ByteBuffer layouts are supported (flexible/Surface path TBD)"
        ))),
    }
}

/// Map a `media_status_t` to a [`DecodeError`] on failure.
fn media_ok(status: ffi::media_status_t, context: &str) -> Result<(), DecodeError> {
    if status == ffi::media_status_t::AMEDIA_OK {
        Ok(())
    } else {
        Err(DecodeError::Open(format!("{context} failed: {}", status.0)))
    }
}
