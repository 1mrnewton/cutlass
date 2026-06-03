use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;

use ffmpeg_next::codec::{Context, threading};
use ffmpeg_next::error::EAGAIN;
use ffmpeg_next::format::{self, context::Input};
use ffmpeg_next::media::Type;
use ffmpeg_next::packet::Packet;
use ffmpeg_next::util::frame::video::Video;
use ffmpeg_next::{Error as FfmpegError, Rational};
use tracing::{debug, warn};

use crate::error::DecodeError;
use crate::frame::{DecodedFrame, PixelFormat};
use crate::hwaccel::{
    self, DecodeOptions, HwAccel, is_hardware_pixel_format, release_device_ref, retain_device_ref,
    transfer_to_cpu,
};

static FFMPEG_INIT: OnceLock<Result<(), FfmpegError>> = OnceLock::new();

fn ensure_ffmpeg_init() -> Result<(), DecodeError> {
    match FFMPEG_INIT.get_or_init(ffmpeg_next::init) {
        Ok(()) => Ok(()),
        Err(e) => Err(DecodeError::Open(*e)),
    }
}

#[derive(Debug, Clone)]
pub struct SourceInfo {
    pub width: u32,
    pub height: u32,
    /// Expected CPU pixel layout after decode (software or post-transfer).
    pub pixel_format: PixelFormat,
    pub time_base: Rational,
    pub frame_rate: Rational,
    pub hw_accel: HwAccel,
}

pub struct Decoder {
    input: Input,
    decoder: ffmpeg_next::codec::decoder::Video,
    stream_index: usize,
    info: SourceInfo,
    demuxer_done: bool,
    sw_frame: Video,
    /// A packet the demuxer produced but the decoder couldn't accept yet
    /// (`send_packet` returned EAGAIN). Retried before reading a new one.
    pending_packet: Option<Packet>,
    _hw_device: Option<*mut ffmpeg_next::ffi::AVBufferRef>,
}

impl Decoder {
    pub fn open(path: &Path) -> Result<Self, DecodeError> {
        Self::open_with(path, DecodeOptions::default())
    }

    pub fn open_with(path: &Path, options: DecodeOptions) -> Result<Self, DecodeError> {
        ensure_ffmpeg_init()?;

        let path_str = path
            .to_str()
            .ok_or_else(|| DecodeError::unsupported("path is not valid UTF-8"))?;

        let input = format::input(path_str).map_err(DecodeError::Open)?;
        let stream = input
            .streams()
            .best(Type::Video)
            .ok_or_else(|| DecodeError::unsupported("no video stream found"))?;

        let stream_index = stream.index();
        let time_base = stream.time_base();
        let frame_rate = stream.avg_frame_rate();

        let mut ctx = Context::from_parameters(stream.parameters()).map_err(DecodeError::Open)?;
        ctx.set_threading(threading::Config {
            kind: threading::Type::Frame,
            count: 0,
        });

        let active_hw = hwaccel::attach(&mut ctx, options.hw_accel)?;
        let hw_device = retain_device_ref(&ctx);

        let decoder = ctx.decoder().video().map_err(DecodeError::Open)?;

        let width = decoder.width();
        let height = decoder.height();
        if width == 0 || height == 0 {
            return Err(DecodeError::unsupported("zero video dimensions"));
        }

        let pixel_format = if active_hw.uses_hardware() {
            // Transfer target is usually NV12 or YUV420P; refined per frame after transfer.
            PixelFormat::from_ffmpeg(decoder.format()).unwrap_or(PixelFormat::Nv12)
        } else {
            PixelFormat::from_ffmpeg(decoder.format()).ok_or_else(|| {
                DecodeError::unsupported("pixel format not YUV420P, NV12, or RGBA")
            })?
        };

        let info = SourceInfo {
            width,
            height,
            pixel_format,
            time_base,
            frame_rate,
            hw_accel: active_hw,
        };

        // info!(
        //     path = path_str,
        //     width,
        //     height,
        //     ?pixel_format,
        //     hw = active_hw.name(),
        //     "opened video for decode"
        // );

        Ok(Self {
            input,
            decoder,
            stream_index,
            info,
            demuxer_done: false,
            sw_frame: Video::empty(),
            pending_packet: None,
            _hw_device: hw_device,
        })
    }

    pub fn info(&self) -> &SourceInfo {
        &self.info
    }

    /// Seek to the keyframe at or before `target` and flush decoder buffers.
    ///
    /// This positions the demuxer; the next [`next_frame`](Self::next_frame) call
    /// returns the first frame decoded from that keyframe, which may precede
    /// `target`. Use [`seek_to_frame`](Self::seek_to_frame) for frame-accurate
    /// positioning.
    pub fn seek(&mut self, target: Duration) -> Result<(), DecodeError> {
        // `avformat_seek_file` with stream_index = -1 expects AV_TIME_BASE
        // (microsecond) units. Range `..ts` requests a keyframe at or before it.
        let ts = i64::try_from(target.as_micros()).unwrap_or(i64::MAX);
        self.input.seek(ts, ..ts).map_err(DecodeError::Io)?;
        self.decoder.flush();
        self.demuxer_done = false;
        self.pending_packet = None;
        debug!(target_micros = ts, "seeked to keyframe");
        Ok(())
    }

    /// Seek and decode forward to the first frame at or after `target`.
    ///
    /// Returns `Ok(None)` if the stream ends before reaching `target`.
    ///
    /// Intermediate frames between the entry keyframe and `target` are decoded
    /// but never read back to CPU memory: with a long GOP that throwaway work
    /// dominates, so the GPU→CPU transfer and the plane copy
    /// ([`DecodedFrame::from_ffmpeg`]) happen only for the frame that lands.
    pub fn seek_to_frame(&mut self, target: Duration) -> Result<Option<DecodedFrame>, DecodeError> {
        self.seek(target)?;

        let target_ticks = self.duration_to_ticks(target);
        while let Some(frame) = self.next_video_frame()? {
            let pts = frame_pts(&frame).unwrap_or(i64::MIN);
            if pts >= target_ticks {
                let cpu = if is_hardware_pixel_format(frame.format()) {
                    transfer_to_cpu(&frame, &mut self.sw_frame)?;
                    &self.sw_frame
                } else {
                    &frame
                };
                return Ok(Some(DecodedFrame::from_ffmpeg(cpu)?));
            }
        }
        Ok(None)
    }

    /// Convert a wall-clock duration into this stream's `time_base` ticks.
    fn duration_to_ticks(&self, target: Duration) -> i64 {
        let tb = self.info.time_base;
        let num = tb.numerator();
        let den = tb.denominator();
        if num <= 0 || den <= 0 {
            return 0;
        }
        // ticks = seconds * (den / num)
        (target.as_secs_f64() * (f64::from(den) / f64::from(num))) as i64
    }

    pub fn next_video_frame(&mut self) -> Result<Option<Video>, DecodeError> {
        loop {
            let mut frame = Video::empty();
            match self.decoder.receive_frame(&mut frame) {
                Ok(()) => {
                    return Ok(Some(frame));
                }
                Err(FfmpegError::Eof) => return Ok(None),
                Err(e) if is_eagain(&e) => {
                    if self.demuxer_done {
                        return Ok(None);
                    }
                    self.read_packet()?;
                }
                Err(e) => return Err(DecodeError::Decode(e)),
            }
        }
    }

    pub fn next_frame(&mut self) -> Result<Option<DecodedFrame>, DecodeError> {
        loop {
            let mut frame = Video::empty();
            match self.decoder.receive_frame(&mut frame) {
                Ok(()) => {
                    let cpu = if is_hardware_pixel_format(frame.format()) {
                        transfer_to_cpu(&frame, &mut self.sw_frame)?;
                        &self.sw_frame
                    } else {
                        &frame
                    };

                    let decoded = DecodedFrame::from_ffmpeg(cpu)?;
                    debug!(
                        width = decoded.width,
                        height = decoded.height,
                        pts = decoded.pts_ticks,
                        format = ?decoded.format,
                        hw = self.info.hw_accel.name(),
                        "decoded frame"
                    );
                    return Ok(Some(decoded));
                }
                Err(FfmpegError::Eof) => return Ok(None),
                Err(e) if is_eagain(&e) => {
                    if self.demuxer_done {
                        return Ok(None);
                    }
                    self.read_packet()?;
                }
                Err(e) => return Err(DecodeError::Decode(e)),
            }
        }
    }

    fn read_packet(&mut self) -> Result<(), DecodeError> {
        // Retry a previously un-accepted packet before pulling a new one.
        if let Some(packet) = self.pending_packet.take() {
            return self.send_packet(packet);
        }

        let mut packet = Packet::empty();
        match packet.read(&mut self.input) {
            Ok(()) => {
                if packet.stream() == self.stream_index {
                    self.send_packet(packet)
                } else {
                    Ok(())
                }
            }
            Err(FfmpegError::Eof) => {
                self.demuxer_done = true;
                self.decoder.send_eof().map_err(DecodeError::Decode)
            }
            Err(e) => Err(DecodeError::Io(e)),
        }
    }

    /// Send a packet, stashing it as pending if the decoder's input is full.
    fn send_packet(&mut self, packet: Packet) -> Result<(), DecodeError> {
        match self.decoder.send_packet(&packet) {
            Ok(()) => Ok(()),
            Err(e) if is_eagain(&e) => {
                self.pending_packet = Some(packet);
                Ok(())
            }
            Err(e) => Err(DecodeError::Decode(e)),
        }
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        release_device_ref(&mut self._hw_device);
    }
}

fn is_eagain(e: &FfmpegError) -> bool {
    matches!(e, FfmpegError::Other { errno } if *errno == EAGAIN)
}

/// Best-effort presentation timestamp of a raw decoded frame, in stream ticks.
fn frame_pts(frame: &Video) -> Option<i64> {
    frame.timestamp().or_else(|| frame.pts())
}

pub fn ffmpeg_version() -> String {
    unsafe {
        std::ffi::CStr::from_ptr(ffmpeg_next::ffi::av_version_info())
            .to_string_lossy()
            .into_owned()
    }
}

/// Parse `CUTLASS_HWACCEL` / common names into [`HwAccel`].
pub fn hw_accel_from_env(value: &str) -> HwAccel {
    match value.trim().to_ascii_lowercase().as_str() {
        "0" | "none" | "sw" | "software" => HwAccel::None,
        "auto" => HwAccel::Auto,
        "videotoolbox" | "vt" => HwAccel::VideoToolbox,
        #[cfg(any(target_os = "linux", doc))]
        "vaapi" => HwAccel::Vaapi,
        #[cfg(any(
            all(target_os = "linux", target_arch = "x86_64"),
            target_os = "windows",
            doc
        ))]
        "nvdec" | "cuda" => HwAccel::Nvdec,
        #[cfg(any(target_os = "linux", target_os = "windows", doc))]
        "qsv" => HwAccel::Qsv,
        #[cfg(any(target_os = "windows", doc))]
        "d3d11" | "d3d11va" => HwAccel::D3d11va,
        other => {
            warn!(value = other, "unknown hwaccel name, using auto");
            HwAccel::Auto
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn sibling_main_asset(name: &str) -> Option<PathBuf> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../cutlass-main/crates/decoder/tests/assets")
            .join(name);
        path.exists().then_some(path)
    }

    #[test]
    fn corrupt_fixture_fails_open() {
        let Some(path) = sibling_main_asset("corrupt_truncated.mp4") else {
            return;
        };
        assert!(Decoder::open(&path).is_err());
    }

    #[test]
    fn decode_first_frame_software() {
        let Some(path) = sibling_main_asset("testsrc_h264.mp4") else {
            return;
        };
        let mut dec = Decoder::open_with(&path, DecodeOptions::default().hw_accel(HwAccel::None))
            .expect("open");
        assert_eq!(dec.info().hw_accel, HwAccel::None);
        let frame = dec.next_frame().expect("decode").expect("first frame");
        assert!(frame.width > 0 && !frame.planes.is_empty());
    }

    #[test]
    fn seek_then_decode_lands_at_or_after_target() {
        let Some(path) = sibling_main_asset("testsrc_h264.mp4") else {
            return;
        };
        let mut dec = Decoder::open_with(&path, DecodeOptions::default().hw_accel(HwAccel::None))
            .expect("open");

        let target = Duration::from_millis(500);
        let target_ticks = dec.duration_to_ticks(target);

        let frame = dec
            .seek_to_frame(target)
            .expect("seek")
            .expect("frame after seek");
        assert!(frame.pts_ticks >= target_ticks);
        assert!(frame.width > 0 && !frame.planes.is_empty());
    }

    #[test]
    fn decode_first_frame_hw_auto() {
        let Some(path) = sibling_main_asset("testsrc_h264.mp4") else {
            return;
        };
        let mut dec = Decoder::open_with(&path, DecodeOptions::default()).expect("open");
        let frame = dec.next_frame().expect("decode").expect("first frame");
        assert!(frame.width > 0 && !frame.planes.is_empty());
        assert!(matches!(
            frame.format,
            PixelFormat::Yuv420p | PixelFormat::Nv12 | PixelFormat::Rgba8
        ));
    }
}
