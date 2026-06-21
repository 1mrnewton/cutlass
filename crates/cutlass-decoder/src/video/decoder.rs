use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;

use ffmpeg_next::codec::{Context, threading};
use ffmpeg_next::error::EAGAIN;
use ffmpeg_next::format::{self, context::Input};
use ffmpeg_next::media::Type;
use ffmpeg_next::packet::Packet;
use ffmpeg_next::util::frame::video::Video;
use ffmpeg_next::{Discard, Error as FfmpegError, Rational};
use tracing::{debug, warn};

use crate::error::DecodeError;
use crate::video::frame::{DecodedFrame, PixelFormat};
use crate::video::hwaccel::{
    self, DecodeOptions, HwAccel, is_hardware_pixel_format, release_device_ref, retain_device_ref,
    transfer_to_cpu,
};
use crate::video::keyframe_indexer::KeyframeIndex;
use crate::video::scale::{convert_to_yuv420p, is_pipeline_pixel_format};

static FFMPEG_INIT: OnceLock<Result<(), FfmpegError>> = OnceLock::new();

pub(crate) fn ensure_ffmpeg_init() -> Result<(), DecodeError> {
    match FFMPEG_INIT.get_or_init(ffmpeg_next::init) {
        Ok(()) => Ok(()),
        Err(e) => Err(DecodeError::Open(*e)),
    }
}

#[derive(Debug, Clone)]
pub struct SourceInfo {
    pub width: u32,
    pub height: u32,

    pub pixel_format: PixelFormat,
    /// Source uses full (JPEG, 0–255) YUV range rather than limited (16–235).
    /// True for `yuvj420p` media (phones, screen recordings). Decoded frames
    /// carry the same flag; this is the stable per-source value used to stamp
    /// frames served from the preview cache (which stores only raw planes).
    pub full_range: bool,
    pub time_base: Rational,
    pub frame_rate: Rational,
    pub hw_accel: HwAccel,
}

impl SourceInfo {
    /// Average frame rate as `(numerator, denominator)`.
    pub fn frame_rate_parts(&self) -> (i32, i32) {
        (self.frame_rate.numerator(), self.frame_rate.denominator())
    }
}

pub struct Decoder {
    input: Input,
    decoder: ffmpeg_next::codec::decoder::Video,
    stream_index: usize,
    info: SourceInfo,
    demuxer_done: bool,
    sw_frame: Video,
    /// Packet the demuxer produced but the decoder couldn't accept yet (`EAGAIN`).
    pending_packet: Option<Packet>,
    indexer: KeyframeIndex,
    /// PTS of the last frame this decoder emitted — the roll-forward
    /// decision point for [`frame_at`](Self::frame_at). Cleared whenever
    /// decoder buffers flush (`seek`, `set_skip_frame`).
    last_pts: Option<i64>,
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

        let active_hw = hwaccel::attach(&mut ctx, options.hw_accel)?;
        // Frame threading is a large win for software decode but most hwaccel
        // backends decode on a single async device queue; combining both tends
        // to add latency on the first receive_frame after send_packet.
        if !active_hw.uses_hardware() {
            ctx.set_threading(threading::Config {
                kind: threading::Type::Frame,
                count: 0,
            });
        }

        let hw_device = retain_device_ref(&ctx);

        let decoder = ctx.decoder().video().map_err(DecodeError::Open)?;

        let width = decoder.width();
        let height = decoder.height();
        if width == 0 || height == 0 {
            return Err(DecodeError::unsupported("zero video dimensions"));
        }

        // Formats outside {YUV420P, NV12, RGBA} are normalized to limited-range
        // YUV420P at decode time (see `finish_frame`), so report that here
        // instead of refusing to open. Hardware decode hasn't produced a frame
        // yet, so its surface format is unknown — assume the usual NV12 transfer
        // output; software decode already knows the codec's pixel format.
        let pixel_format = if active_hw.uses_hardware() {
            PixelFormat::from_ffmpeg(decoder.format()).unwrap_or(PixelFormat::Nv12)
        } else {
            PixelFormat::from_ffmpeg(decoder.format()).unwrap_or(PixelFormat::Yuv420p)
        };
        // Stable per-source range: software decode already knows the codec's
        // pixel format, so `yuvj420p` is detectable here and matches what every
        // decoded frame reports (used to re-stamp cache hits, which keep only
        // raw planes). Hardware surfaces don't expose this until the first
        // transferred frame, so they default to limited — the historical
        // assumption, unchanged.
        let full_range = matches!(decoder.format(), format::Pixel::YUVJ420P);

        let info = SourceInfo {
            width,
            height,
            pixel_format,
            full_range,
            time_base,
            frame_rate,
            hw_accel: active_hw,
        };

        let indexer = KeyframeIndex::build(path)?;

        Ok(Self {
            input,
            decoder,
            stream_index,
            info,
            demuxer_done: false,
            sw_frame: Video::empty(),
            pending_packet: None,
            indexer,
            last_pts: None,
            _hw_device: hw_device,
        })
    }

    pub fn info(&self) -> &SourceInfo {
        &self.info
    }

    /// Container duration, if the demuxer reports one (microsecond `AV_TIME_BASE` units).
    pub fn duration(&self) -> Option<Duration> {
        let micros = self.input.duration();
        (micros > 0).then(|| Duration::from_micros(micros as u64))
    }

    /// Seek to the keyframe at or before `target` and flush decoder buffers.
    pub fn seek(&mut self, target: Duration) -> Result<(), DecodeError> {
        self.seek_ticks(self.indexer.duration_to_ticks(target))
    }

    /// [`seek`](Self::seek) with the target already in stream `time_base`
    /// ticks — the exact-math entry point (no `Duration` truncation).
    pub fn seek_ticks(&mut self, target_ticks: i64) -> Result<(), DecodeError> {
        let ts = self
            .indexer
            .seek_us_at_or_before(target_ticks)
            .ok_or_else(|| DecodeError::unsupported("target is before the first keyframe"))?;
        self.input.seek(ts, ..ts).map_err(DecodeError::Io)?;
        self.decoder.flush();
        self.demuxer_done = false;
        self.pending_packet = None;
        self.last_pts = None;
        debug!(target_micros = ts, target_ticks, "seeked to keyframe");
        Ok(())
    }

    /// Seek and decode forward to the first frame at or after `target`.
    pub fn seek_to_frame(&mut self, target: Duration) -> Result<Option<DecodedFrame>, DecodeError> {
        self.seek(target)?;
        let target_ticks = self.indexer.duration_to_ticks(target);
        self.walk_to_frame(target_ticks)
    }

    /// Decode the first frame at or after `target`, reusing the decoder's
    /// position when it helps — the sequential-playback fast path.
    pub fn frame_at(&mut self, target: Duration) -> Result<Option<DecodedFrame>, DecodeError> {
        self.frame_at_ticks(self.indexer.duration_to_ticks(target))
    }

    /// [`frame_at`](Self::frame_at) with the target already in stream
    /// `time_base` ticks (see
    /// [`KeyframeIndex::rate_ticks_to_stream_ticks`]).
    ///
    /// When the target lies *ahead* of the last emitted frame and that frame
    /// sits inside the target's GOP, every frame in between has to be decoded
    /// regardless, so this rolls forward instead of re-seeking (a seek would
    /// flush and re-decode the whole GOP prefix: O(GOP²) per GOP across a
    /// playback run). Anything else — backward targets, GOP jumps, a fresh
    /// decoder — falls back to seek + walk, byte-identical to
    /// [`seek_to_frame`](Self::seek_to_frame).
    pub fn frame_at_ticks(
        &mut self,
        target_ticks: i64,
    ) -> Result<Option<DecodedFrame>, DecodeError> {
        let roll = self.last_pts.is_some_and(|last_pts| {
            target_ticks > last_pts
                && self
                    .indexer
                    .gop_containing(target_ticks)
                    .is_some_and(|gop| gop.contains(last_pts))
        });
        if !roll {
            self.seek_ticks(target_ticks)?;
        }
        self.walk_to_frame(target_ticks)
    }

    /// Decode forward from the current position to the first frame with
    /// `pts >= target_ticks`, recording it as the roll-forward anchor.
    fn walk_to_frame(&mut self, target_ticks: i64) -> Result<Option<DecodedFrame>, DecodeError> {
        while let Some(frame) = self.next_video_frame()? {
            let pts = frame_pts(&frame).unwrap_or(i64::MIN);
            if pts >= target_ticks {
                let decoded = self.finish_frame(&frame)?;
                self.last_pts = Some(decoded.pts_ticks);
                return Ok(Some(decoded));
            }
        }
        Ok(None)
    }

    /// Turn a freshly received frame into a [`DecodedFrame`]: copy it off the
    /// GPU when hardware-decoded, then normalize any pixel format the pipeline
    /// can't ingest directly (full-range `yuvj*`, 10-bit, 4:2:2 / 4:4:4, …) to
    /// limited-range YUV420P.
    fn finish_frame(&mut self, frame: &Video) -> Result<DecodedFrame, DecodeError> {
        let cpu = if is_hardware_pixel_format(frame.format()) {
            transfer_to_cpu(frame, &mut self.sw_frame)?;
            &self.sw_frame
        } else {
            frame
        };

        if is_pipeline_pixel_format(cpu.format()) {
            DecodedFrame::from_ffmpeg(cpu)
        } else {
            let converted = convert_to_yuv420p(cpu)?;
            DecodedFrame::from_ffmpeg(&converted)
        }
    }

    pub fn set_skip_frame(&mut self, discard: Discard) {
        unsafe {
            (*self.decoder.as_mut_ptr()).skip_frame = discard.into();
        }

        self.decoder.flush();
        self.last_pts = None;
    }
    pub fn seek_dirty_to_frame(
        &mut self,
        target: Duration,
    ) -> Result<Option<DecodedFrame>, DecodeError> {
        let target_ticks = self.indexer.duration_to_ticks(target);
        self.seek(target)?;
        self.set_skip_frame(Discard::NonKey);
        if let Some(frame) = self.walk_to_frame(target_ticks)? {
            self.set_skip_frame(Discard::Default);
            return Ok(Some(frame));
        }
        // No keyframe at/after target (common with sparse GOPs): fall back to
        // full decode from a fresh seek.
        self.set_skip_frame(Discard::Default);
        self.seek(target)?;
        self.walk_to_frame(target_ticks)
    }

    pub fn next_video_frame(&mut self) -> Result<Option<Video>, DecodeError> {
        let mut frame = Video::empty();
        loop {
            match self.decoder.receive_frame(&mut frame) {
                Ok(()) => return Ok(Some(frame)),
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
        let mut frame = Video::empty();
        loop {
            match self.decoder.receive_frame(&mut frame) {
                Ok(()) => {
                    let decoded = self.finish_frame(&frame)?;
                    debug!(
                        width = decoded.width,
                        height = decoded.height,
                        pts = decoded.pts_ticks,
                        format = ?decoded.format,
                        hw = self.info.hw_accel.name(),
                        "decoded frame"
                    );
                    // Keep the roll-forward anchor honest when callers mix
                    // sequential pulls with frame_at on one decoder.
                    self.last_pts = Some(decoded.pts_ticks);
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
        if let Some(packet) = self.pending_packet.take() {
            return self.send_packet(packet);
        }

        loop {
            // Fresh packet per read: non-target (e.g. audio) packets are
            // dropped at the end of the iteration, which unrefs their payload.
            // Reusing one packet would leak every skipped packet, since
            // `Packet::read` does not unref and `Packet` only unrefs on drop.
            let mut packet = Packet::empty();
            match packet.read(&mut self.input) {
                Ok(()) => {
                    if packet.stream() == self.stream_index {
                        return self.send_packet(packet);
                    }
                }
                Err(FfmpegError::Eof) => {
                    self.demuxer_done = true;
                    return self.decoder.send_eof().map_err(DecodeError::Decode);
                }
                Err(e) => return Err(DecodeError::Io(e)),
            }
        }
    }

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

    fn workspace_asset(name: &str) -> Option<PathBuf> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../local-assets/assets")
            .join(name);
        path.exists().then_some(path)
    }

    fn any_video_asset() -> Option<PathBuf> {
        workspace_asset("6137050-hd_1920_1080_24fps.mp4").or_else(|| {
            std::fs::read_dir(
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../local-assets/assets"),
            )
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|e| e == "mp4"))
        })
    }

    #[test]
    fn decode_first_frame_software() {
        let Some(path) = any_video_asset() else {
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
        let Some(path) = any_video_asset() else {
            return;
        };
        let mut dec = Decoder::open_with(&path, DecodeOptions::default().hw_accel(HwAccel::None))
            .expect("open");

        let target = Duration::from_millis(500);
        let target_ticks = dec.indexer.duration_to_ticks(target);

        let frame = dec
            .seek_to_frame(target)
            .expect("seek")
            .expect("frame after seek");
        assert!(frame.pts_ticks >= target_ticks);
        assert!(frame.width > 0 && !frame.planes.is_empty());
    }

    #[test]
    fn decode_first_frame_hw_auto() {
        let Some(path) = any_video_asset() else {
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

    #[test]
    fn hw_accel_from_env_parses_aliases() {
        assert_eq!(hw_accel_from_env("none"), HwAccel::None);
        assert_eq!(hw_accel_from_env("SW"), HwAccel::None);
        assert_eq!(hw_accel_from_env("auto"), HwAccel::Auto);
        assert_eq!(hw_accel_from_env("videotoolbox"), HwAccel::VideoToolbox);
        assert_eq!(hw_accel_from_env(" vt "), HwAccel::VideoToolbox);
        assert_eq!(hw_accel_from_env("bogus-backend"), HwAccel::Auto);
    }

    #[test]
    fn ffmpeg_version_is_non_empty() {
        let v = ffmpeg_version();
        assert!(!v.is_empty());
        assert!(v.chars().any(|c| c.is_ascii_digit()));
    }

    #[test]
    #[cfg(unix)]
    fn open_rejects_non_utf8_path() {
        use std::os::unix::ffi::OsStrExt;
        let path = std::path::Path::new(std::ffi::OsStr::from_bytes(b"\xff\xfe/video.mp4"));
        let err = match Decoder::open(path) {
            Err(e) => e,
            Ok(_) => panic!("expected non-utf8 path to be rejected"),
        };
        assert!(matches!(err, DecodeError::Unsupported { .. }));
    }

    #[test]
    fn source_info_reports_dimensions_and_time_base() {
        let Some(path) = any_video_asset() else {
            return;
        };
        let dec = Decoder::open_with(&path, DecodeOptions::default().hw_accel(HwAccel::None))
            .expect("open");
        let info = dec.info();
        assert!(info.width > 0);
        assert!(info.height > 0);
        assert!(info.time_base.denominator() > 0);
        assert!(info.frame_rate_parts().0 > 0);
    }

    #[test]
    fn sequential_decode_advances_pts() {
        let Some(path) = any_video_asset() else {
            return;
        };
        let mut dec = Decoder::open_with(&path, DecodeOptions::default().hw_accel(HwAccel::None))
            .expect("open");
        let f0 = dec.next_frame().expect("decode").expect("frame 0");
        let f1 = dec.next_frame().expect("decode").expect("frame 1");
        assert!(f1.pts_ticks >= f0.pts_ticks);
    }

    #[test]
    fn frame_at_matches_seek_to_frame_on_sequential_targets() {
        let Some(path) = any_video_asset() else {
            return;
        };
        let opts = DecodeOptions::default().hw_accel(HwAccel::None);
        let mut rolled = Decoder::open_with(&path, opts).expect("open rolled");
        let mut seeked = Decoder::open_with(&path, opts).expect("open seeked");

        // ~2s of sequential 30fps playback targets, plus fractional offsets
        // (a 24fps timeline over mismatched-rate media never lands exactly
        // on source frame times).
        for i in 0..60_u64 {
            let target = Duration::from_micros(i * 33_333 + 7);
            let a = rolled.frame_at(target).expect("frame_at").expect("frame");
            let b = seeked
                .seek_to_frame(target)
                .expect("seek_to_frame")
                .expect("frame");
            assert_eq!(a.pts_ticks, b.pts_ticks, "diverged at target {i}");
        }
    }

    #[test]
    fn frame_at_handles_backward_and_repeated_targets() {
        let Some(path) = any_video_asset() else {
            return;
        };
        let opts = DecodeOptions::default().hw_accel(HwAccel::None);
        let mut rolled = Decoder::open_with(&path, opts).expect("open rolled");
        let mut seeked = Decoder::open_with(&path, opts).expect("open seeked");

        // Forward, repeat, backward, forward jump — the scrub pattern.
        for ms in [0_u64, 500, 500, 100, 1500, 200] {
            let target = Duration::from_millis(ms);
            let a = rolled.frame_at(target).expect("frame_at").expect("frame");
            let b = seeked
                .seek_to_frame(target)
                .expect("seek_to_frame")
                .expect("frame");
            assert_eq!(a.pts_ticks, b.pts_ticks, "diverged at {ms}ms");
        }
    }

    #[test]
    fn frame_at_stays_correct_after_sequential_pulls() {
        let Some(path) = any_video_asset() else {
            return;
        };
        let opts = DecodeOptions::default().hw_accel(HwAccel::None);
        let mut dec = Decoder::open_with(&path, opts).expect("open");
        let mut reference = Decoder::open_with(&path, opts).expect("open reference");

        // Advance the decoder position outside frame_at...
        for _ in 0..5 {
            dec.next_frame().expect("decode").expect("frame");
        }
        // ...then ask for a target the position may have passed; the answer
        // must match a clean seek.
        let target = Duration::from_millis(50);
        let a = dec.frame_at(target).expect("frame_at").expect("frame");
        let b = reference
            .seek_to_frame(target)
            .expect("seek")
            .expect("frame");
        assert_eq!(a.pts_ticks, b.pts_ticks);
    }

    #[test]
    fn seek_snaps_to_keyframe_at_or_before_target() {
        let Some(path) = any_video_asset() else {
            return;
        };
        let dec = Decoder::open_with(&path, DecodeOptions::default().hw_accel(HwAccel::None))
            .expect("open");
        let target = Duration::from_millis(500);
        let target_ticks = dec.indexer.duration_to_ticks(target);
        let kf = dec
            .indexer
            .keyframe_at_or_before_ticks(target_ticks)
            .expect("keyframe");
        assert!(kf <= target_ticks);
    }

    #[test]
    fn seek_dirty_to_frame_returns_keyframe_or_later() {
        let Some(path) = any_video_asset() else {
            return;
        };
        let mut dec = Decoder::open_with(&path, DecodeOptions::default().hw_accel(HwAccel::None))
            .expect("open");
        let target = Duration::from_millis(750);
        let target_ticks = dec.indexer.duration_to_ticks(target);
        let frame = dec
            .seek_dirty_to_frame(target)
            .expect("seek dirty")
            .expect("frame");
        assert!(frame.pts_ticks >= target_ticks);
    }

    #[test]
    fn duration_reports_positive_for_asset() {
        let Some(path) = any_video_asset() else {
            return;
        };
        let dec = Decoder::open_with(&path, DecodeOptions::default().hw_accel(HwAccel::None))
            .expect("open");
        let dur = dec.duration().expect("duration");
        assert!(dur > Duration::ZERO);
    }
}
