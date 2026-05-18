use std::cell::Cell;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::OnceLock;

use ffmpeg_next::Error as FfmpegError;
use ffmpeg_next::Rational as FfmpegRational;
use ffmpeg_next::codec::Context;
use ffmpeg_next::error::EAGAIN;
use ffmpeg_next::format::{self, context::Input};
use ffmpeg_next::media::Type;
use ffmpeg_next::packet::Packet;
use ffmpeg_next::util::frame::video::Video as FfmpegVideo;

use crate::error::DecoderError;
use crate::frame::{CpuFrame, DecodedVideoFrame, FrameData, Plane};
use crate::outcome::DecodeOutcome;
use crate::pixel::PixelFormat;
use crate::source::SourceInfo;
use crate::time::Rational;

static FFMPEG_INIT: OnceLock<Result<(), ffmpeg_next::Error>> = OnceLock::new();

fn ensure_ffmpeg_init() -> Result<(), DecoderError> {
    match FFMPEG_INIT.get_or_init(ffmpeg_next::init) {
        Ok(()) => Ok(()),
        Err(e) => Err(DecoderError::Open(*e)),
    }
}

/// Demux + decode session for **one** video file.
///
/// # Threading
/// **Not `Sync`:** [`PhantomData<Cell<()>>`] prevents a future careless `unsafe impl Sync` on a dependency
/// from making this type shareable across threads. Use one engine decode thread (see `docs/decoder/research.md`).
///
/// # Seeking
/// - [`Decoder::seek_scrub`]: backward sync-point seek + **first** picture (fast scrub).
/// - [`Decoder::seek_exact`]: backward seek + decode until [`DecodedVideoFrame::pts`] ≥ target (accurate still).
pub struct Decoder {
    input: Input,
    decoder: ffmpeg_next::codec::decoder::Video,
    stream_index: usize,
    stream_timebase: FfmpegRational,
    info: SourceInfo,
    demuxer_done: bool,
    _not_sync: PhantomData<Cell<()>>,
}

impl Decoder {
    /// Opens the container, picks the best **video** stream, opens the codec, and validates **v1** pixel formats
    /// ([`PixelFormat::Yuv420p`], [`PixelFormat::Nv12`], [`PixelFormat::Rgba8`] only).
    ///
    /// Runs [`ffmpeg_next::init`] once per process on first success (see [`ffmpeg_version`]).
    pub fn open(path: &Path) -> Result<Self, DecoderError> {
        ensure_ffmpeg_init()?;

        let path_str = path.to_str().ok_or_else(|| {
            DecoderError::unsupported(
                "path is not valid UTF-8 (ffmpeg accepts UTF-8 paths only here)",
            )
        })?;

        let input = format::input(path_str).map_err(DecoderError::Open)?;
        let stream = input
            .streams()
            .best(Type::Video)
            .ok_or_else(|| DecoderError::unsupported("no video stream found"))?;
        let stream_index = stream.index();
        let stream_timebase = stream.time_base();

        let ctx = Context::from_parameters(stream.parameters()).map_err(DecoderError::Open)?;
        let decoder = ctx.decoder().video().map_err(DecoderError::Open)?;

        let pixel_format = PixelFormat::from_ffmpeg(decoder.format())
            .ok_or_else(|| DecoderError::unsupported("pixel format not YUV420P, NV12, or RGBA"))?;

        let width = decoder.width();
        let height = decoder.height();
        if width == 0 || height == 0 {
            return Err(DecoderError::unsupported("zero video dimensions"));
        }

        let timebase = Rational::new_raw(
            i64::from(stream_timebase.numerator()),
            stream_timebase.denominator() as u32,
        );

        let duration_ticks = stream.duration();
        let duration = Rational::from_duration_ticks(duration_ticks, stream_timebase);

        let info = SourceInfo {
            width,
            height,
            timebase: timebase.reduced(),
            duration,
            pixel_format,
        };

        Ok(Self {
            input,
            decoder,
            stream_index,
            stream_timebase,
            info,
            demuxer_done: false,
            _not_sync: PhantomData,
        })
    }

    /// Probed metadata from open (immutable for the lifetime of this session).
    pub fn info(&self) -> &SourceInfo {
        &self.info
    }

    fn media_seconds_to_timestamp(
        target: Rational,
        tb: FfmpegRational,
    ) -> Result<i64, DecoderError> {
        let tb_n = i64::from(tb.numerator());
        let tb_d = i64::from(tb.denominator());
        if tb_n == 0 {
            return Err(DecoderError::unsupported(
                "stream timebase numerator is zero",
            ));
        }
        let num = target.num as i128 * tb_d as i128;
        let den = target.den as i128 * tb_n as i128;
        if den == 0 {
            return Err(DecoderError::unsupported("invalid timebase for seek"));
        }
        let ts = (num / den) as i64;
        Ok(ts)
    }

    /// Backward seek on the container + flush codec state.
    fn seek_inner(&mut self, target: Rational) -> Result<(), DecoderError> {
        let ts = Self::media_seconds_to_timestamp(target, self.stream_timebase)?;
        // `Input::seek` always uses stream_index -1, which interprets timestamps in `AV_TIME_BASE`
        // (microseconds), not stream timebase. For editors we need stream-relative PTS units.
        const AVSEEK_FLAG_BACKWARD: i32 = 1;
        unsafe {
            let ret = ffmpeg_next::ffi::avformat_seek_file(
                self.input.as_mut_ptr(),
                self.stream_index as i32,
                i64::MIN,
                ts,
                ts,
                AVSEEK_FLAG_BACKWARD,
            );
            if ret < 0 {
                return Err(DecoderError::Seek(ffmpeg_next::Error::from(ret)));
            }
        }
        self.decoder.flush();
        self.demuxer_done = false;
        Ok(())
    }

    /// **Keyframe snap:** backward seek, flush, return the **first** decoded picture — no PTS chase.
    /// Cost is **O(GOP)**; image may jump between keyframes while scrubbing (see design doc).
    pub fn seek_scrub(&mut self, target: Rational) -> Result<DecodeOutcome, DecoderError> {
        self.seek_inner(target)?;
        match self.pump_decoded_frame()? {
            Some(frame) => Ok(DecodeOutcome::Frame(frame)),
            None => Ok(DecodeOutcome::Eof),
        }
    }

    /// Exact presentation time: backward seek, flush, decode forward until frame PTS ≥ `target`
    /// (media seconds). Past end of stream → `Ok(DecodeOutcome::Eof)` (engine may clamp).
    pub fn seek_exact(&mut self, target: Rational) -> Result<DecodeOutcome, DecoderError> {
        self.seek_inner(target)?;
        loop {
            match self.pump_decoded_frame()? {
                None => return Ok(DecodeOutcome::Eof),
                Some(frame) => {
                    if frame.pts.ge(target) {
                        return Ok(DecodeOutcome::Frame(frame));
                    }
                }
            }
        }
    }

    /// Next frame in presentation order. After `Ok(DecodeOutcome::Eof)` the stream is drained until the next seek.
    pub fn next_frame(&mut self) -> Result<DecodeOutcome, DecoderError> {
        match self.pump_decoded_frame()? {
            Some(frame) => Ok(DecodeOutcome::Frame(frame)),
            None => Ok(DecodeOutcome::Eof),
        }
    }

    fn is_eagain(e: &FfmpegError) -> bool {
        matches!(e, FfmpegError::Other { errno } if *errno == EAGAIN)
    }

    fn pump_decoded_frame(&mut self) -> Result<Option<DecodedVideoFrame>, DecoderError> {
        let fmt = self.info.pixel_format;

        loop {
            let mut frame = FfmpegVideo::empty();
            match self.decoder.receive_frame(&mut frame) {
                Ok(()) => {
                    let decoded = Self::copy_video_frame(
                        &frame,
                        self.stream_timebase,
                        fmt,
                        self.info.timebase,
                    )?;
                    return Ok(Some(decoded));
                }
                Err(FfmpegError::Eof) => return Ok(None),
                Err(e) if Self::is_eagain(&e) => {
                    if self.demuxer_done {
                        // Should not get EAGAIN after full drain; treat as done.
                        return Ok(None);
                    }
                    let mut packet = Packet::empty();
                    match packet.read(&mut self.input) {
                        Ok(()) => {
                            if packet.stream() == self.stream_index {
                                self.decoder
                                    .send_packet(&packet)
                                    .map_err(DecoderError::Decode)?;
                            }
                        }
                        Err(FfmpegError::Eof) => {
                            self.demuxer_done = true;
                            self.decoder.send_eof().map_err(DecoderError::Decode)?;
                        }
                        Err(err) => return Err(DecoderError::Io(err)),
                    }
                }
                Err(e) => return Err(DecoderError::Decode(e)),
            }
        }
    }

    fn copy_video_frame(
        frame: &FfmpegVideo,
        stream_tb: FfmpegRational,
        fmt: PixelFormat,
        info_timebase: Rational,
    ) -> Result<DecodedVideoFrame, DecoderError> {
        let width = frame.width();
        let height = frame.height();
        let pts_ticks = frame.timestamp().or_else(|| frame.pts()).ok_or_else(|| {
            DecoderError::unsupported("decoded frame has no PTS / best_effort_timestamp")
        })?;
        let pts = Rational::from_stream_ticks(pts_ticks, stream_tb).ok_or_else(|| {
            DecoderError::unsupported("PTS × timebase overflow in Rational conversion")
        })?;

        let plane_heights = fmt
            .plane_heights_full(height)
            .map_err(DecoderError::unsupported)?;
        let mut planes = Vec::with_capacity(plane_heights.len());

        for (i, &ph) in plane_heights.iter().enumerate() {
            let ph = ph as usize;
            let stride = frame.stride(i);
            let need = stride
                .checked_mul(ph)
                .ok_or_else(|| DecoderError::unsupported("frame plane size overflow"))?;
            let data_src = frame.data(i);
            if data_src.len() < need {
                return Err(DecoderError::unsupported(format!(
                    "plane {i}: buffer too small (have {} need {})",
                    data_src.len(),
                    need
                )));
            }
            planes.push(Plane {
                data: data_src[..need].to_vec(),
                stride,
            });
        }

        Ok(DecodedVideoFrame {
            width,
            height,
            pts,
            timebase: info_timebase,
            data: FrameData::Cpu(CpuFrame {
                format: fmt,
                planes,
            }),
        })
    }
}

/// FFmpeg library version string (e.g. `avutil 0x...`).
///
/// Fails with [`DecoderError::Open`] if [`ffmpeg_next::init`] failed on first use (same `OnceLock` state as [`Decoder::open`]).
pub fn ffmpeg_version() -> Result<String, DecoderError> {
    ensure_ffmpeg_init()?;
    Ok(format!("avutil {:#08x}", ffmpeg_next::util::version()))
}
