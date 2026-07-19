//! Apple backend: `AVAssetReader` (demux + VideoToolbox decode) behind
//! [`cutlass_core::VideoDecoder`].
//!
//! `AVAssetReader` reads decoded `CVPixelBuffer`s forward from a configurable
//! start time. The output settings list every format we can ingest, so
//! VideoToolbox vends frames in its native decoded format (NV12 video/full
//! range, or 10-bit/`BGRA` for some sources); the exact format is detected per
//! frame from the pixel buffer. Sample data is *not* copied out of the decoder
//! (`alwaysCopiesSampleData = NO`): the CPU path copies planes itself and the
//! GPU path hands the renderer the decoder's own IOSurface, retained.
//!
//! ## Seeking
//!
//! Playback and forward scrubbing never re-seek: [`VideoDecoder::frame_at`]
//! rolls forward from the last emitted frame when the target is just ahead
//! (see [`crate::seek`]), which is what avoids the O(GOP²) re-decode of a
//! seek-per-frame loop. A true [`AvfDecoder::seek`] (backward, or a long jump)
//! tears down and rebuilds the reader at the new start time against the
//! already-parsed `AVURLAsset` — `AVAssetReader` is forward-only, so this is
//! the correct random-access primitive. (`resetForReadingTimeRanges:` was
//! evaluated and rejected: it only re-arms after the *current* range is fully
//! drained, so a scrub seek would decode-and-drop the rest of the in-flight
//! range first, costing more than the reader rebuild it saves.)
//!
//! One caveat other backends don't share: `AVAssetReader` trims its output to
//! the time range, decoding the GOP prefix *internally*, so
//! [`VideoDecoder::frame_at_nearest`]'s "seek + one decode" here returns the
//! frame at/after the target at the cost of the full prefix decode — there is
//! no cheap snap-to-sync-frame on this backend. On long-GOP 4K that is
//! hundreds of milliseconds per cold target (thumbnails, filmstrips); if that
//! ever matters it needs demux/decode split apart (compressed sample output +
//! our own `VTDecompressionSession`), not tuning here.
//!
//! Two pieces of lifecycle hygiene keep sustained scrubbing stable: an
//! outgoing reader is [`AVAssetReader::cancelReading`]-ed when replaced or
//! dropped (releasing a started reader without cancelling leaves its decode
//! pipeline — VideoToolbox session plus queued IOSurfaces — to wind down
//! asynchronously; under scrub churn hundreds of those pile up and exhaust
//! process-wide decode resources, after which *every* decoder in the process
//! fails with "Cannot Decode"), and each frame pull runs inside an
//! [`autoreleasepool`] because decode workers are plain Rust threads with no
//! enclosing pool — without one, AVFoundation's autoreleased bookkeeping
//! accumulates for the life of the thread.
//!
//! ## Open-GOP sources: seek-point decode recovery
//!
//! `AVAssetReader` starts decode at the container sync sample at/before the
//! time range's start and trusts it blindly. Open-GOP H.264 (x264
//! `open_gop=1`, some hardware encoders) marks **non-IDR I-frames** as sync
//! samples, and most of those are followed in decode order by leading
//! B-frames that display *before* the I-frame and reference the **previous**
//! GOP. Starting decode there, VideoToolbox hits references that don't exist
//! and — unlike FFmpeg, which drops the undecodable leading frames — fails
//! the entire reader session (`AVErrorDecodeFailed`, "Cannot Decode") before
//! vending a single frame. On a real-world 4K export with ~10 s GOPs this
//! made *every seek into 8 of the file's 11 GOPs* a hard error, while
//! sequential decode of the same file was flawless (the references exist by
//! the time the boundary is crossed in order).
//!
//! The recovery ([`AvfDecoder::recover_read_failure`]): when the reader
//! fails, rebuild it with the start pulled back from where the caller was —
//! just past the last emitted frame, or the requested seek start — doubling
//! the back-off (1 s, 2 s, 4 s, … clamped to 0, where the first sample must
//! be genuinely decodable) until decode sticks, then walk forward and trim
//! back up to that position. The failure can strike *at* the seek point
//! (start sync sample has leading B-frames; the reader dies before vending
//! anything) or *mid-walk* while crossing such a sync sample after a start
//! at an earlier non-IDR one — resuming past the last emitted frame handles
//! both. Failed attempts are cheap (VideoToolbox dies within a few frames of
//! a bad start, and pre-bound frames skip conversion); the successful
//! attempt pays one GOP-prefix walk, which is what a closed-GOP file would
//! have charged for the seek anyway. If even the start of the stream won't
//! decode (real corruption), the error surfaces — but through
//! [`AvfDecoder::reset_failed_reader`], which replaces the dead reader (a
//! `Failed` reader can never vend again) and clears the roll-forward anchor,
//! so the next `frame_at` re-seeks on a healthy reader instead of re-reading
//! the corpse forever.

use core::cmp::Ordering;
use std::path::Path;
use std::ptr::NonNull;
use std::slice;

use objc2::AllocAnyThread;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::AnyObject;
use objc2_av_foundation::{
    AVAssetReader, AVAssetReaderStatus, AVAssetReaderTrackOutput, AVAssetTrack, AVMediaTypeAudio,
    AVMediaTypeVideo, AVURLAsset,
};
use objc2_core_foundation::{CFRetained, CFString, CFType, CGAffineTransform, CGSize};
use objc2_core_media::{
    CMFormatDescription, CMTime, CMTimeFlags, CMTimeRange,
    kCMFormatDescriptionColorPrimaries_DCI_P3, kCMFormatDescriptionColorPrimaries_EBU_3213,
    kCMFormatDescriptionColorPrimaries_ITU_R_709_2, kCMFormatDescriptionColorPrimaries_ITU_R_2020,
    kCMFormatDescriptionColorPrimaries_P3_D65, kCMFormatDescriptionColorPrimaries_SMPTE_C,
    kCMFormatDescriptionExtension_ColorPrimaries, kCMFormatDescriptionExtension_TransferFunction,
    kCMFormatDescriptionExtension_YCbCrMatrix, kCMFormatDescriptionTransferFunction_ITU_R_709_2,
    kCMFormatDescriptionTransferFunction_ITU_R_2020,
    kCMFormatDescriptionTransferFunction_ITU_R_2100_HLG,
    kCMFormatDescriptionTransferFunction_SMPTE_ST_2084_PQ,
    kCMFormatDescriptionTransferFunction_sRGB, kCMFormatDescriptionYCbCrMatrix_ITU_R_601_4,
    kCMFormatDescriptionYCbCrMatrix_ITU_R_709_2, kCMFormatDescriptionYCbCrMatrix_ITU_R_2020,
    kCMTimePositiveInfinity,
};
use objc2_core_video::{
    CVImageBuffer, CVPixelBuffer, CVPixelBufferGetBaseAddressOfPlane,
    CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferGetHeight, CVPixelBufferGetHeightOfPlane,
    CVPixelBufferGetPixelFormatType, CVPixelBufferGetPlaneCount, CVPixelBufferGetWidth,
    CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
    kCVPixelBufferPixelFormatTypeKey,
};
use objc2_foundation::{
    NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL, NSUnderlyingErrorKey,
};

use cutlass_core::{
    ColorPrimaries, ColorRange, ColorSpace, CpuImage, DecodeError, FrameData, GpuSurface,
    GpuSurfaceKind, MatrixCoefficients, PixelFormat, Plane, Rational, RationalTime, Rect, Rotation,
    SourceInfo, TransferFunction, VideoDecoder, VideoFrame,
};

use crate::OutputMode;

// FourCC pixel format codes (CoreVideo OSType constants), matched against
// `CVPixelBufferGetPixelFormatType`.
const FOURCC_420V: u32 = u32::from_be_bytes(*b"420v"); // NV12, video (limited) range
const FOURCC_420F: u32 = u32::from_be_bytes(*b"420f"); // NV12, full range
const FOURCC_X420: u32 = u32::from_be_bytes(*b"x420"); // 10-bit 4:2:0, video range
const FOURCC_BGRA: u32 = u32::from_be_bytes(*b"BGRA"); // 32-bit BGRA

/// A decoded video source backed by AVFoundation / VideoToolbox.
pub struct AvfDecoder {
    asset: Retained<AVURLAsset>,
    track: Retained<AVAssetTrack>,
    reader: Retained<AVAssetReader>,
    output: Retained<AVAssetReaderTrackOutput>,
    info: SourceInfo,
    mode: OutputMode,
    started: bool,
    /// The requested start of the current reader (`None` = whole clip):
    /// where a replacement reader is rebuilt after a failure, and the trim
    /// point the failure recovery walks back up to (see the module docs).
    start: Option<RationalTime>,
    /// PTS of the last emitted frame — the [`crate::seek`] roll-forward anchor.
    /// Cleared on [`VideoDecoder::seek`] (decode position moved) and on read
    /// failure (the rebuilt reader is not at that position).
    last_pts: Option<RationalTime>,
    /// Observed seek/decode costs driving the adaptive roll window.
    seek_stats: crate::seek::SeekStats,
}

// SAFETY: AVFoundation/CoreVideo objects here are owned and touched by a single
// thread at a time — the decode-ahead worker the engine parks this decoder on.
// We never share `&self` across threads, satisfying the `VideoDecoder: Send`
// requirement without needing `Sync`.
unsafe impl Send for AvfDecoder {}

/// Parks a retained CoreVideo buffer in a [`GpuSurface`]'s `keep_alive` so the
/// decoder's IOSurface outlives the renderer's reads. The raw handle in the
/// surface points at this same object.
struct RetainedImageBuffer(#[allow(dead_code)] CFRetained<CVImageBuffer>);

// SAFETY: a decoded `CVImageBuffer` is immutable, and CoreVideo's
// retain/release is atomic, so the wrapper can travel to the render thread and
// drop there. We only ever hold (never mutate) the buffer.
unsafe impl Send for RetainedImageBuffer {}
unsafe impl Sync for RetainedImageBuffer {}

/// Lower PTS bound for frames the failure-recovery walk may emit: the
/// rebuilt reader starts earlier than the caller's position, and everything
/// below the bound is trimmed without conversion.
#[derive(Clone, Copy)]
enum EmitFrom {
    /// Emit the first frame at/after the bound (a requested seek start).
    At(RationalTime),
    /// Emit the first frame strictly after the bound (resuming past the
    /// last frame the caller already has).
    After(RationalTime),
}

impl EmitFrom {
    fn bound(self) -> RationalTime {
        match self {
            EmitFrom::At(t) | EmitFrom::After(t) => t,
        }
    }

    fn admits(self, pts: RationalTime) -> bool {
        match self {
            EmitFrom::At(t) => pts.compare(t) != Ordering::Less,
            EmitFrom::After(t) => pts.compare(t) == Ordering::Greater,
        }
    }
}

impl AvfDecoder {
    /// Open `path` and probe its first video track.
    pub fn open(path: &Path, mode: OutputMode) -> Result<Self, DecodeError> {
        let path_str = path
            .to_str()
            .ok_or_else(|| DecodeError::unsupported("path is not valid UTF-8"))?;

        let url = NSURL::fileURLWithPath(&NSString::from_str(path_str));
        let asset = unsafe { AVURLAsset::initWithURL_options(AVURLAsset::alloc(), &url, None) };

        let track = first_video_track(&asset)
            .ok_or_else(|| DecodeError::unsupported("no video track found"))?;

        let info = probe(&track);
        // No explicit time range -> AVFoundation's default (the whole clip).
        let (reader, output) = build_reader(&asset, &track, None)?;

        Ok(Self {
            asset,
            track,
            reader,
            output,
            info,
            mode,
            started: false,
            start: None,
            last_pts: None,
            seek_stats: Default::default(),
        })
    }

    /// Whether this decoder's asset also carries an audio track. Reads the
    /// already-parsed asset, so the probe doesn't pay a second open of the
    /// file.
    pub(crate) fn has_audio_track(&self) -> bool {
        let Some(media_audio) = (unsafe { AVMediaTypeAudio }) else {
            return false;
        };
        #[allow(deprecated)]
        let tracks = unsafe { self.asset.tracksWithMediaType(media_audio) };
        tracks.firstObject().is_some()
    }

    /// Pull the next decoded frame, recovering from reader decode failures
    /// (see the module docs on open-GOP sources). Non-reader errors — an
    /// unsupported pixel format, a lock failure — pass through untouched;
    /// reader failures that survive recovery surface only after
    /// [`Self::reset_failed_reader`] has put the decoder back into a usable
    /// state.
    fn next_pixel_frame(&mut self) -> Result<Option<VideoFrame>, DecodeError> {
        let result = self.ensure_started().and_then(|()| self.pull_frame(None));
        match result {
            Err(err) if unsafe { self.reader.status() } == AVAssetReaderStatus::Failed => {
                self.recover_read_failure(err)
            }
            other => other,
        }
    }

    /// Start the reader if it hasn't been started yet.
    fn ensure_started(&mut self) -> Result<(), DecodeError> {
        if !self.started {
            if !unsafe { self.reader.startReading() } {
                return Err(self.reader_error("startReading failed"));
            }
            self.started = true;
        }
        Ok(())
    }

    /// Pull the next decoded sample buffer's pixel buffer, skipping
    /// marker-only buffers — and, when `emit_from` is set, frames below that
    /// bound (the failure recovery reads from an earlier start and trims back
    /// up to where the caller was) — and convert it to a [`VideoFrame`].
    ///
    /// Runs inside an autorelease pool: decode workers are plain Rust threads,
    /// so without one AVFoundation's per-pull autoreleased objects would
    /// accumulate until the thread exits. The returned frame only holds
    /// explicitly retained references, so it safely outlives the pool.
    fn pull_frame(
        &mut self,
        emit_from: Option<EmitFrom>,
    ) -> Result<Option<VideoFrame>, DecodeError> {
        autoreleasepool(|_| {
            loop {
                let Some(sample) = (unsafe { self.output.copyNextSampleBuffer() }) else {
                    // Exhausted, or an error mid-stream — distinguish via status.
                    return match unsafe { self.reader.status() } {
                        AVAssetReaderStatus::Failed => Err(self.reader_error("read failed")),
                        _ => Ok(None),
                    };
                };

                // Marker-only buffers carry no image; skip them.
                if unsafe { sample.num_samples() } == 0 {
                    continue;
                }

                let Some(image_buffer) = (unsafe { sample.image_buffer() }) else {
                    continue;
                };

                let pts = cmtime_to_rational(unsafe { sample.presentation_time_stamp() })
                    .unwrap_or_else(|| RationalTime::zero(self.info.time_base));

                // Trim frames below the recovery bound without paying the
                // plane copy / surface wrap for them.
                if let Some(bound) = emit_from
                    && !bound.admits(pts)
                {
                    continue;
                }

                let frame = self.image_buffer_to_frame(image_buffer, pts)?;
                self.last_pts = Some(pts);
                return Ok(Some(frame));
            }
        })
    }

    /// Recover from a reader that failed mid-read. On valid files that is
    /// VideoToolbox refusing an open-GOP start point (see the module docs):
    /// the reader dies on the sync sample's leading B-frames — sometimes
    /// after vending a few decodable frames — even though every frame the
    /// caller actually wants decodes fine from an earlier, sounder start.
    ///
    /// Rebuild the reader progressively earlier (doubling back-off, clamped
    /// to the start of the stream, where the first sample must be a real
    /// sync frame) and walk forward to where the caller was: just past the
    /// last emitted frame, or the requested seek start if nothing was
    /// emitted yet. Attempts that fail at their own start point are
    /// near-free (VideoToolbox rejects them before decoding much); the
    /// successful attempt pays one GOP-prefix walk, which is what a
    /// closed-GOP file would have charged for the seek anyway.
    ///
    /// Gives up — surfacing `original` after [`Self::reset_failed_reader`] —
    /// when there is nowhere to back away from (a failure at the very start
    /// of the stream) or when even the start of the stream won't decode
    /// (real corruption; the bounded back-off makes the total give-up cost a
    /// couple of decode walks, paid once per caller-visible error).
    fn recover_read_failure(
        &mut self,
        original: DecodeError,
    ) -> Result<Option<VideoFrame>, DecodeError> {
        // Resume just past the last emitted frame (mid-walk failure), or at
        // the requested start (nothing emitted since the seek).
        let resume = match (self.last_pts, self.start) {
            (Some(pts), _) => EmitFrom::After(pts),
            (None, Some(start)) => EmitFrom::At(start),
            (None, None) => {
                // Cold open failing at the head of the stream: nothing to
                // back away from.
                self.reset_failed_reader();
                return Err(original);
            }
        };
        let anchor = resume.bound();
        let mut backoff = one_second_ticks(anchor.rate);
        loop {
            let earlier = RationalTime::new((anchor.value - backoff).max(0), anchor.rate);
            let attempt = self
                .rebuild_reader_at(Some(earlier))
                .and_then(|()| self.ensure_started())
                .and_then(|()| self.pull_frame(Some(resume)));
            match attempt {
                // Recovered (`Some`), or the resume point really is past the
                // end of the stream (`None`) — a healthy reader's answer.
                Ok(frame) => return Ok(frame),
                Err(err) if unsafe { self.reader.status() } != AVAssetReaderStatus::Failed => {
                    // Not a reader failure (unsupported format, lock error):
                    // an earlier start cannot change it.
                    return Err(err);
                }
                Err(_) => {
                    if earlier.value == 0 {
                        // Even the start of the stream won't decode.
                        self.reset_failed_reader();
                        return Err(original);
                    }
                    backoff = backoff.saturating_mul(2);
                }
            }
        }
    }

    /// Put the decoder back into a usable state after a read failure: a
    /// `Failed` reader can never vend again, so replace it (best-effort) with
    /// a fresh, un-started one at the last requested position, and clear the
    /// roll-forward anchor so the next [`VideoDecoder::frame_at`] takes the
    /// seek path on the healthy reader instead of rolling onto the dead one.
    /// The decode *position* after a failure is unspecified; callers re-drive
    /// through `frame_at`/`seek`, which both re-position.
    fn reset_failed_reader(&mut self) {
        self.last_pts = None;
        // On rebuild failure the cancelled reader stays; the next pull then
        // errors through `ensure_started` and lands back here — no loop, the
        // recovery path is bounded and `frame_at` callers see the error.
        let _ = self.rebuild_reader_at(self.start);
    }

    /// Replace the current reader with one whose time range starts at
    /// `range_start` (`None` = whole clip). Builds the replacement first so a
    /// failure leaves the decoder unchanged, then cancels the old reader
    /// before releasing it (see the module docs on reader lifecycle).
    fn rebuild_reader_at(&mut self, range_start: Option<RationalTime>) -> Result<(), DecodeError> {
        let start = range_start.map(rational_to_cmtime);
        let (reader, output) = build_reader(&self.asset, &self.track, start)?;
        self.cancel_current_reader();
        self.reader = reader;
        self.output = output;
        self.started = false;
        Ok(())
    }

    /// Cancel the current reader's in-flight decode work before it is replaced
    /// or dropped.
    ///
    /// Releasing a started reader *without* cancelling leaves its pipeline
    /// (VideoToolbox session, queued IOSurfaces) to unwind asynchronously.
    /// Sustained scrubbing rebuilds readers faster than those zombies drain,
    /// and once enough pile up the process hits its decode-resource ceiling and
    /// every subsequent read fails with "Cannot Decode".
    fn cancel_current_reader(&mut self) {
        if self.started {
            unsafe { self.reader.cancelReading() };
            self.started = false;
        }
    }

    /// Convert one decoded `CVImageBuffer` into a [`VideoFrame`]. Takes ownership
    /// of the retained buffer so the GPU path can park it as the surface's
    /// keep-alive; the CPU path copies its planes and lets it drop.
    fn image_buffer_to_frame(
        &self,
        image_buffer: CFRetained<CVImageBuffer>,
        pts: RationalTime,
    ) -> Result<VideoFrame, DecodeError> {
        // A decoded `CVImageBuffer` *is* a `CVPixelBuffer` (same CFType);
        // reinterpret to reach the pixel-buffer accessors.
        let pb: &CVPixelBuffer = unsafe {
            CFRetained::as_ptr(&image_buffer)
                .cast::<CVPixelBuffer>()
                .as_ref()
        };

        let fourcc = CVPixelBufferGetPixelFormatType(pb);
        let (format, range) = detect_format(fourcc).ok_or_else(|| {
            DecodeError::unsupported(format!("unsupported pixel format {}", fourcc_str(fourcc)))
        })?;

        let width = CVPixelBufferGetWidth(pb) as u32;
        let height = CVPixelBufferGetHeight(pb) as u32;

        let color = ColorSpace {
            range,
            ..self.info.color
        };

        let data = match self.mode {
            OutputMode::Cpu => FrameData::Cpu(CpuImage::new(read_cpu_planes(pb)?)),
            OutputMode::Gpu => {
                // Hand the renderer the IOSurface-backed buffer's address and
                // keep our retained reference alive alongside it. The pointer
                // and the `keep_alive` refer to the same object.
                let handle = CFRetained::as_ptr(&image_buffer).as_ptr() as usize as u64;
                let surface = GpuSurface::new(
                    GpuSurfaceKind::CoreVideoPixelBuffer,
                    handle,
                    Some(Box::new(RetainedImageBuffer(image_buffer))),
                );
                FrameData::Gpu(surface)
            }
        };

        Ok(VideoFrame::new(
            pts,
            format,
            color,
            (width, height),
            Rect::from_size(width, height),
            self.info.rotation,
            data,
        ))
    }

    /// Build a [`DecodeError`] from the reader's `NSError`, if any, carrying
    /// enough to diagnose a failure from a log line alone: error domain +
    /// code, the underlying error (for `AVErrorDecodeFailed` that is the
    /// VideoToolbox `OSStatus`), and where the reader was positioned.
    fn reader_error(&self, context: &str) -> DecodeError {
        let detail = unsafe { self.reader.error() }
            .map(|e| describe_ns_error(&e))
            .unwrap_or_else(|| "no NSError".to_string());
        let start = match self.start {
            Some(t) => format!("{:.3}s", rt_secs(t)),
            None => "clip start".to_string(),
        };
        let position = match self.last_pts {
            Some(t) => format!(", last pts {:.3}s", rt_secs(t)),
            None => ", no frames delivered".to_string(),
        };
        DecodeError::Decode(format!(
            "{context}: {detail} (reader start {start}{position})"
        ))
    }
}

/// `NSError` → "description [domain code]", recursing one level into the
/// underlying error, which for AVFoundation decode failures holds the
/// otherwise-invisible VideoToolbox `OSStatus`.
fn describe_ns_error(err: &NSError) -> String {
    let mut text = format!(
        "{} [{} {}]",
        err.localizedDescription(),
        err.domain(),
        err.code()
    );
    let underlying = err
        .userInfo()
        .objectForKey(unsafe { NSUnderlyingErrorKey })
        .and_then(|value| value.downcast::<NSError>().ok());
    if let Some(u) = underlying {
        text.push_str(&format!(
            " (underlying: {} [{} {}])",
            u.localizedDescription(),
            u.domain(),
            u.code()
        ));
    }
    text
}

/// [`RationalTime`] in seconds, for error messages only.
fn rt_secs(t: RationalTime) -> f64 {
    t.value as f64 * f64::from(t.rate.den.max(1)) / f64::from(t.rate.num.max(1))
}

/// One second expressed in ticks of `rate`, rounded up, at least one tick —
/// the base step of the seek-point recovery back-off.
fn one_second_ticks(rate: Rational) -> i64 {
    let num = i64::from(rate.num.max(1));
    let den = i64::from(rate.den.max(1));
    ((num + den - 1) / den).max(1)
}

impl VideoDecoder for AvfDecoder {
    fn info(&self) -> &SourceInfo {
        &self.info
    }

    fn seek(&mut self, target: RationalTime) -> Result<(), DecodeError> {
        // Rebuild first so a failed seek leaves the decoder unchanged.
        self.rebuild_reader_at(Some(target))?;
        self.start = Some(target);
        self.last_pts = None;
        Ok(())
    }

    fn next_frame(&mut self) -> Result<Option<VideoFrame>, DecodeError> {
        self.next_pixel_frame()
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

    fn frame_at_nearest(
        &mut self,
        target: RationalTime,
    ) -> Result<Option<VideoFrame>, DecodeError> {
        let last_pts = self.last_pts;
        let mut stats = self.seek_stats;
        let result = crate::seek::frame_at_nearest_rolling(self, last_pts, &mut stats, target);
        self.seek_stats = stats;
        result
    }
}

impl Drop for AvfDecoder {
    fn drop(&mut self) {
        self.cancel_current_reader();
    }
}

/// Find the first video track of an asset (synchronous probe).
fn first_video_track(asset: &AVURLAsset) -> Option<Retained<AVAssetTrack>> {
    let media_video = unsafe { AVMediaTypeVideo }?;
    #[allow(deprecated)]
    let tracks = unsafe { asset.tracksWithMediaType(media_video) };
    tracks.firstObject()
}

/// Duration of the first audio track at `path`, for probing sources with no
/// video stream (music/voiceover files). `None` when there is no audio track
/// or the container reports no duration.
pub(crate) fn audio_track_duration(path: &Path) -> Option<RationalTime> {
    let track = first_audio_track(path)?;
    cmtime_to_rational(unsafe { track.timeRange() }.duration)
}

fn first_audio_track(path: &Path) -> Option<Retained<AVAssetTrack>> {
    let path_str = path.to_str()?;
    let url = NSURL::fileURLWithPath(&NSString::from_str(path_str));
    let asset = unsafe { AVURLAsset::initWithURL_options(AVURLAsset::alloc(), &url, None) };
    let media_audio = unsafe { AVMediaTypeAudio }?;
    #[allow(deprecated)]
    let tracks = unsafe { asset.tracksWithMediaType(media_audio) };
    tracks.firstObject()
}

/// Output settings asking the reader for decoded pixel buffers in one of our
/// supported formats. Listing several lets VideoToolbox return the native depth
/// (8-bit NV12 or 10-bit) rather than forcing a conversion.
fn decode_output_settings() -> Retained<NSDictionary<NSString, AnyObject>> {
    let formats = [FOURCC_420V, FOURCC_420F, FOURCC_X420, FOURCC_BGRA];
    let numbers: Vec<Retained<NSNumber>> = formats
        .iter()
        .map(|&f| NSNumber::numberWithUnsignedInt(f))
        .collect();
    let number_refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
    let array = NSArray::from_slice(&number_refs);

    // `kCVPixelBufferPixelFormatTypeKey` is a CFString; toll-free bridged to
    // NSString for use as a dictionary key.
    let key_cf: &CFString = unsafe { kCVPixelBufferPixelFormatTypeKey };
    let key: &NSString = unsafe { &*(core::ptr::from_ref(key_cf).cast::<NSString>()) };
    let value: &AnyObject = unsafe { &*(Retained::as_ptr(&array).cast::<AnyObject>()) };

    NSDictionary::from_slices(&[key], &[value])
}

/// Build a reader + track output starting at `start`, reading to end of stream.
fn build_reader(
    asset: &AVURLAsset,
    track: &AVAssetTrack,
    start: Option<CMTime>,
) -> Result<(Retained<AVAssetReader>, Retained<AVAssetReaderTrackOutput>), DecodeError> {
    let reader = unsafe { AVAssetReader::initWithAsset_error(AVAssetReader::alloc(), asset) }
        .map_err(|e| DecodeError::Open(e.localizedDescription().to_string()))?;

    // Request decoded pixel buffers. With `nil` settings the reader hands back
    // *compressed* sample buffers; we list the formats we can ingest and let
    // VideoToolbox pick the one closest to its native decode output (no extra
    // conversion when the source is 8-bit 4:2:0 or 10-bit), detected per frame.
    let settings = decode_output_settings();
    let output = unsafe {
        AVAssetReaderTrackOutput::initWithTrack_outputSettings(
            AVAssetReaderTrackOutput::alloc(),
            track,
            Some(&settings),
        )
    };

    unsafe {
        // We never mutate sample data in place (the CPU path copies planes
        // itself; the GPU path retains the buffer), so skip AVFoundation's
        // defensive per-frame copy — a per-frame allocation+memcpy otherwise.
        output.setAlwaysCopiesSampleData(false);
        // Only constrain the range when seeking; otherwise read the whole clip.
        if let Some(start) = start {
            reader.setTimeRange(CMTimeRange {
                start,
                duration: kCMTimePositiveInfinity,
            });
        }
        if !reader.canAddOutput(&output) {
            return Err(DecodeError::Open("reader cannot add track output".into()));
        }
        reader.addOutput(&output);
    }

    Ok((reader, output))
}

/// Probe static source properties from the track.
fn probe(track: &AVAssetTrack) -> SourceInfo {
    let size: CGSize = unsafe { track.naturalSize() };
    let display_w = size.width.round().max(0.0) as u32;
    let display_h = size.height.round().max(0.0) as u32;

    let transform: CGAffineTransform = unsafe { track.preferredTransform() };
    let rotation = rotation_from_transform(transform);

    let frame_rate = frame_rate_of(track);
    let time_base = Rational::new(unsafe { track.naturalTimeScale() }.max(1), 1);
    // Track-level duration (CMTime) at the source's own timescale; consumers
    // resample to whatever rate they need (e.g. the media pool's frame count).
    let duration = cmtime_to_rational(unsafe { track.timeRange() }.duration);

    SourceInfo {
        coded_size: (display_w, display_h),
        display_size: (display_w, display_h),
        rotation,
        pixel_format: PixelFormat::Nv12,
        // Primaries/transfer/matrix come from the track's format description;
        // `range` is refined per frame from the actual pixel format.
        color: color_from_track(track),
        frame_rate,
        time_base,
        duration,
    }
}

/// Read colorimetry (primaries / transfer / matrix) from the track's first
/// `CMFormatDescription`. Missing tags keep the BT.709 SDR default; `range` is
/// left at the default and refined per frame from the pixel format.
fn color_from_track(track: &AVAssetTrack) -> ColorSpace {
    let mut color = ColorSpace::BT709;

    let descs = unsafe { track.formatDescriptions() };
    let Some(obj) = descs.firstObject() else {
        return color;
    };
    // The array holds `CMFormatDescription`s (a CoreMedia CFType) bridged in as
    // objects; reinterpret the element to reach the extension accessors.
    let desc: &CMFormatDescription =
        unsafe { &*(Retained::as_ptr(&obj).cast::<CMFormatDescription>()) };

    if let Some(p) = read_tag(
        desc,
        unsafe { kCMFormatDescriptionExtension_ColorPrimaries },
        map_primaries,
    ) {
        color.primaries = p;
    }
    if let Some(t) = read_tag(
        desc,
        unsafe { kCMFormatDescriptionExtension_TransferFunction },
        map_transfer,
    ) {
        color.transfer = t;
    }
    if let Some(m) = read_tag(
        desc,
        unsafe { kCMFormatDescriptionExtension_YCbCrMatrix },
        map_matrix,
    ) {
        color.matrix = m;
    }
    color
}

/// Look up a format-description extension expected to be a `CFString` tag and
/// run it through `map`. Returns `None` when the tag is absent or not a string.
fn read_tag<T>(
    desc: &CMFormatDescription,
    key: &CFString,
    map: impl Fn(&CFString) -> T,
) -> Option<T> {
    let value = unsafe { desc.extension(key) }?;
    let tag = value.downcast_ref::<CFString>()?;
    Some(map(tag))
}

/// CFString content equality (`CFEqual`) against a known CoreMedia constant.
fn cf_tag_eq(value: &CFString, constant: &CFString) -> bool {
    let value: &CFType = value;
    let constant: &CFType = constant;
    value == constant
}

fn map_primaries(value: &CFString) -> ColorPrimaries {
    if cf_tag_eq(value, unsafe {
        kCMFormatDescriptionColorPrimaries_ITU_R_709_2
    }) {
        ColorPrimaries::Bt709
    } else if cf_tag_eq(value, unsafe {
        kCMFormatDescriptionColorPrimaries_ITU_R_2020
    }) {
        ColorPrimaries::Bt2020
    } else if cf_tag_eq(value, unsafe { kCMFormatDescriptionColorPrimaries_P3_D65 })
        || cf_tag_eq(value, unsafe { kCMFormatDescriptionColorPrimaries_DCI_P3 })
    {
        ColorPrimaries::DisplayP3
    } else if cf_tag_eq(value, unsafe { kCMFormatDescriptionColorPrimaries_SMPTE_C }) {
        ColorPrimaries::Bt601_525
    } else if cf_tag_eq(value, unsafe {
        kCMFormatDescriptionColorPrimaries_EBU_3213
    }) {
        ColorPrimaries::Bt601_625
    } else {
        ColorPrimaries::Unspecified
    }
}

fn map_transfer(value: &CFString) -> TransferFunction {
    // Rec.2020's SDR transfer is, for our purposes, the Rec.709 curve.
    if cf_tag_eq(value, unsafe {
        kCMFormatDescriptionTransferFunction_ITU_R_709_2
    }) || cf_tag_eq(value, unsafe {
        kCMFormatDescriptionTransferFunction_ITU_R_2020
    }) {
        TransferFunction::Bt709
    } else if cf_tag_eq(value, unsafe {
        kCMFormatDescriptionTransferFunction_SMPTE_ST_2084_PQ
    }) {
        TransferFunction::Pq
    } else if cf_tag_eq(value, unsafe {
        kCMFormatDescriptionTransferFunction_ITU_R_2100_HLG
    }) {
        TransferFunction::Hlg
    } else if cf_tag_eq(value, unsafe { kCMFormatDescriptionTransferFunction_sRGB }) {
        TransferFunction::Srgb
    } else {
        TransferFunction::Unspecified
    }
}

fn map_matrix(value: &CFString) -> MatrixCoefficients {
    if cf_tag_eq(value, unsafe {
        kCMFormatDescriptionYCbCrMatrix_ITU_R_709_2
    }) {
        MatrixCoefficients::Bt709
    } else if cf_tag_eq(value, unsafe {
        kCMFormatDescriptionYCbCrMatrix_ITU_R_601_4
    }) {
        MatrixCoefficients::Bt601
    } else if cf_tag_eq(value, unsafe { kCMFormatDescriptionYCbCrMatrix_ITU_R_2020 }) {
        MatrixCoefficients::Bt2020Ncl
    } else {
        MatrixCoefficients::Unspecified
    }
}

/// Frame rate from the track's minimum frame duration (exact for constant
/// frame rates), guarded against VFR by the nominal average rate.
fn frame_rate_of(track: &AVAssetTrack) -> Rational {
    let min_dur = unsafe { track.minFrameDuration() };
    // fps = timescale / value (frames per second).
    let exact = (min_dur.value > 0 && min_dur.timescale > 0)
        .then(|| Rational::new(min_dur.timescale, min_dur.value as i32));
    resolve_frame_rate(exact, unsafe { track.nominalFrameRate() })
}

/// Pick the reported frame rate: the exact minimum-frame-duration rate when
/// it agrees with the nominal average, the average otherwise. On VFR sources
/// the smallest inter-frame gap is an artifact, not the cadence — reporting
/// it inflates every duration-derived frame count, and the pool then asks
/// for frames past EOF.
fn resolve_frame_rate(min_duration_rate: Option<Rational>, nominal_fps: f32) -> Rational {
    if let Some(exact) = min_duration_rate {
        // 5%: comfortably above timescale rounding (23.976 vs 23.98), well
        // below any real VFR spread.
        if nominal_fps <= 0.0 || exact.as_f64() <= f64::from(nominal_fps) * 1.05 {
            return exact;
        }
    }
    if nominal_fps > 0.0 {
        Rational::new((nominal_fps * 1000.0).round() as i32, 1000)
    } else {
        Rational::FPS_30
    }
}

/// Derive a 0/90/180/270 rotation from a track's preferred transform.
fn rotation_from_transform(t: CGAffineTransform) -> Rotation {
    let degrees = t.b.atan2(t.a).to_degrees().round() as i32;
    Rotation::from_degrees(degrees).unwrap_or(Rotation::None)
}

/// Map a CoreVideo FourCC pixel format to ours plus its sample range.
fn detect_format(fourcc: u32) -> Option<(PixelFormat, ColorRange)> {
    match fourcc {
        FOURCC_420V => Some((PixelFormat::Nv12, ColorRange::Limited)),
        FOURCC_420F => Some((PixelFormat::Nv12, ColorRange::Full)),
        FOURCC_X420 => Some((PixelFormat::P010, ColorRange::Limited)),
        FOURCC_BGRA => Some((PixelFormat::Bgra8, ColorRange::Full)),
        _ => None,
    }
}

/// Copy every plane of a (locked) pixel buffer into owned [`Plane`]s.
fn read_cpu_planes(pb: &CVPixelBuffer) -> Result<Vec<Plane>, DecodeError> {
    unsafe {
        if CVPixelBufferLockBaseAddress(pb, CVPixelBufferLockFlags::ReadOnly) != 0 {
            return Err(DecodeError::Decode(
                "CVPixelBufferLockBaseAddress failed".into(),
            ));
        }
        let plane_count = CVPixelBufferGetPlaneCount(pb);
        let mut planes = Vec::with_capacity(plane_count.max(1));
        for i in 0..plane_count {
            let base = CVPixelBufferGetBaseAddressOfPlane(pb, i);
            let stride = CVPixelBufferGetBytesPerRowOfPlane(pb, i);
            let rows = CVPixelBufferGetHeightOfPlane(pb, i);
            let Some(base) = NonNull::new(base) else {
                CVPixelBufferUnlockBaseAddress(pb, CVPixelBufferLockFlags::ReadOnly);
                return Err(DecodeError::Decode(format!(
                    "plane {i} base address is null"
                )));
            };
            let bytes = slice::from_raw_parts(base.as_ptr() as *const u8, stride * rows).to_vec();
            planes.push(Plane::new(bytes, stride, rows));
        }
        CVPixelBufferUnlockBaseAddress(pb, CVPixelBufferLockFlags::ReadOnly);
        Ok(planes)
    }
}

/// `CMTime` (value/timescale seconds) -> exact [`RationalTime`].
pub(crate) fn cmtime_to_rational(t: CMTime) -> Option<RationalTime> {
    (t.timescale > 0 && t.flags.contains(CMTimeFlags::Valid))
        .then(|| RationalTime::new(t.value, Rational::new(t.timescale, 1)))
}

/// [`RationalTime`] -> `CMTime` (seconds preserved exactly as value·den / num).
pub(crate) fn rational_to_cmtime(t: RationalTime) -> CMTime {
    let num = t.rate.num.max(1);
    let den = i64::from(t.rate.den.max(1));
    let (value, timescale) = match t.value.checked_mul(den) {
        Some(value) => (value, num),
        // value·den past i64 (an astronomical time on an extreme time base):
        // approximate at microsecond resolution instead of wrapping into a
        // nonsense — possibly negative — time.
        None => (
            (t.value as f64 * den as f64 / f64::from(num) * 1e6) as i64,
            1_000_000,
        ),
    };
    CMTime {
        value,
        timescale,
        flags: CMTimeFlags::Valid,
        epoch: 0,
    }
}

#[cfg(test)]
fn cm_zero() -> CMTime {
    CMTime {
        value: 0,
        timescale: 1,
        flags: CMTimeFlags::Valid,
        epoch: 0,
    }
}

/// Human-readable FourCC for error messages.
fn fourcc_str(code: u32) -> String {
    String::from_utf8_lossy(&code.to_be_bytes()).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_known_pixel_formats() {
        assert_eq!(
            detect_format(FOURCC_420V),
            Some((PixelFormat::Nv12, ColorRange::Limited))
        );
        assert_eq!(
            detect_format(FOURCC_420F),
            Some((PixelFormat::Nv12, ColorRange::Full))
        );
        assert_eq!(
            detect_format(FOURCC_X420),
            Some((PixelFormat::P010, ColorRange::Limited))
        );
        assert_eq!(
            detect_format(FOURCC_BGRA),
            Some((PixelFormat::Bgra8, ColorRange::Full))
        );
        assert_eq!(detect_format(u32::from_be_bytes(*b"av01")), None);
    }

    #[test]
    fn fourcc_roundtrips_to_text() {
        assert_eq!(fourcc_str(FOURCC_420V), "420v");
        assert_eq!(fourcc_str(FOURCC_BGRA), "BGRA");
    }

    #[test]
    fn maps_color_tags_from_constants() {
        // Primaries.
        assert_eq!(
            map_primaries(unsafe { kCMFormatDescriptionColorPrimaries_ITU_R_709_2 }),
            ColorPrimaries::Bt709
        );
        assert_eq!(
            map_primaries(unsafe { kCMFormatDescriptionColorPrimaries_ITU_R_2020 }),
            ColorPrimaries::Bt2020
        );
        assert_eq!(
            map_primaries(unsafe { kCMFormatDescriptionColorPrimaries_P3_D65 }),
            ColorPrimaries::DisplayP3
        );
        assert_eq!(
            map_primaries(unsafe { kCMFormatDescriptionColorPrimaries_SMPTE_C }),
            ColorPrimaries::Bt601_525
        );

        // Transfer (PQ / HLG are the HDR signals we care about).
        assert_eq!(
            map_transfer(unsafe { kCMFormatDescriptionTransferFunction_ITU_R_709_2 }),
            TransferFunction::Bt709
        );
        assert_eq!(
            map_transfer(unsafe { kCMFormatDescriptionTransferFunction_SMPTE_ST_2084_PQ }),
            TransferFunction::Pq
        );
        assert_eq!(
            map_transfer(unsafe { kCMFormatDescriptionTransferFunction_ITU_R_2100_HLG }),
            TransferFunction::Hlg
        );

        // Matrix.
        assert_eq!(
            map_matrix(unsafe { kCMFormatDescriptionYCbCrMatrix_ITU_R_709_2 }),
            MatrixCoefficients::Bt709
        );
        assert_eq!(
            map_matrix(unsafe { kCMFormatDescriptionYCbCrMatrix_ITU_R_601_4 }),
            MatrixCoefficients::Bt601
        );
        assert_eq!(
            map_matrix(unsafe { kCMFormatDescriptionYCbCrMatrix_ITU_R_2020 }),
            MatrixCoefficients::Bt2020Ncl
        );

        // A BT.2020 + PQ combination must read back as HDR + wide gamut.
        let hdr = ColorSpace {
            primaries: map_primaries(unsafe { kCMFormatDescriptionColorPrimaries_ITU_R_2020 }),
            transfer: map_transfer(unsafe {
                kCMFormatDescriptionTransferFunction_SMPTE_ST_2084_PQ
            }),
            matrix: map_matrix(unsafe { kCMFormatDescriptionYCbCrMatrix_ITU_R_2020 }),
            range: ColorRange::Limited,
        };
        assert!(hdr.is_hdr());
        assert!(hdr.is_wide_gamut());
        assert_eq!(hdr, ColorSpace::BT2020_PQ);
    }

    #[test]
    fn cmtime_to_rational_is_exact_and_guards_invalid() {
        let rt = cmtime_to_rational(CMTime {
            value: 1001,
            timescale: 30000,
            flags: CMTimeFlags::Valid,
            epoch: 0,
        })
        .expect("valid");
        assert_eq!(rt.value, 1001);
        assert_eq!(rt.rate, Rational::new(30000, 1));
        // Invalid (timescale 0, no Valid flag) -> None.
        assert!(
            cmtime_to_rational(CMTime {
                value: 0,
                timescale: 0,
                flags: CMTimeFlags::empty(),
                epoch: 0,
            })
            .is_none()
        );
    }

    #[test]
    fn resolve_frame_rate_prefers_exact_but_guards_vfr() {
        // CFR: min frame duration agrees with nominal — keep the exact rate.
        let ntsc = Rational::new(24_000, 1_001);
        assert_eq!(resolve_frame_rate(Some(ntsc), 23.976), ntsc);
        // Missing nominal still keeps the exact rate.
        assert_eq!(resolve_frame_rate(Some(ntsc), 0.0), ntsc);
        // VFR: the smallest gap (120 fps) is far above the 30 fps average —
        // report the average, not the artifact.
        assert_eq!(
            resolve_frame_rate(Some(Rational::new(120, 1)), 30.0),
            Rational::new(30_000, 1000)
        );
        // No min duration: nominal, then the 30 fps fallback.
        assert_eq!(resolve_frame_rate(None, 25.0), Rational::new(25_000, 1000));
        assert_eq!(resolve_frame_rate(None, 0.0), Rational::FPS_30);
    }

    #[test]
    fn one_second_ticks_rounds_up_and_never_zero() {
        assert_eq!(one_second_ticks(Rational::FPS_30), 30);
        // 24000/1001 fps: one second is 23.976… ticks -> 24.
        assert_eq!(one_second_ticks(Rational::new(24_000, 1_001)), 24);
        assert_eq!(one_second_ticks(Rational::new(90_000, 1)), 90_000);
        // Degenerate rates still make progress.
        assert_eq!(one_second_ticks(Rational::new(0, 0)), 1);
    }

    #[test]
    fn describe_ns_error_carries_domain_code_and_underlying() {
        use objc2_foundation::NSInteger;
        let underlying = unsafe {
            NSError::errorWithDomain_code_userInfo(
                &NSString::from_str("NSOSStatusErrorDomain"),
                -12909 as NSInteger, // kVTVideoDecoderBadDataErr
                None,
            )
        };
        let key: &NSString = unsafe { NSUnderlyingErrorKey };
        let value: &AnyObject = underlying.as_ref();
        let user_info = NSDictionary::from_slices(&[key], &[value]);
        let err = unsafe {
            NSError::errorWithDomain_code_userInfo(
                &NSString::from_str("AVFoundationErrorDomain"),
                -11821 as NSInteger, // AVErrorDecodeFailed
                Some(&user_info),
            )
        };
        let text = describe_ns_error(&err);
        assert!(text.contains("AVFoundationErrorDomain"), "{text}");
        assert!(text.contains("-11821"), "{text}");
        assert!(text.contains("NSOSStatusErrorDomain"), "{text}");
        assert!(text.contains("-12909"), "{text}");
    }

    #[test]
    fn rational_to_cmtime_preserves_seconds() {
        // 5 frames at 30fps -> 5/30 s. Represent as value*den / num.
        // CMTime is a packed struct, so copy fields out before comparing.
        let t = rational_to_cmtime(RationalTime::new(5, Rational::FPS_30));
        let (t_val, t_scale) = (t.value, t.timescale);
        assert_eq!(t_scale, 30);
        assert_eq!(t_val, 5);
        assert_eq!(t_val as f64 / t_scale as f64, 5.0 / 30.0);
        // NTSC: 1 frame at 30000/1001 -> value 1001, timescale 30000.
        let n = rational_to_cmtime(RationalTime::new(1, Rational::FPS_29_97));
        let (n_val, n_scale) = (n.value, n.timescale);
        assert_eq!(n_val, 1001);
        assert_eq!(n_scale, 30000);
    }

    #[test]
    fn rational_to_cmtime_survives_value_den_overflow() {
        // value·den overflows i64: falls back to microsecond resolution
        // instead of wrapping negative.
        let t = rational_to_cmtime(RationalTime::new(i64::MAX / 2, Rational::new(1, 4)));
        let (value, timescale) = (t.value, t.timescale);
        assert_eq!(timescale, 1_000_000);
        assert!(value > 0, "must not wrap negative");
    }

    #[test]
    fn rotation_from_transform_maps_quadrants() {
        let id = CGAffineTransform {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            tx: 0.0,
            ty: 0.0,
        };
        assert_eq!(rotation_from_transform(id), Rotation::None);
        // 90° CW: a=0, b=1.
        let r90 = CGAffineTransform {
            a: 0.0,
            b: 1.0,
            c: -1.0,
            d: 0.0,
            tx: 0.0,
            ty: 0.0,
        };
        assert_eq!(rotation_from_transform(r90), Rotation::Cw90);
        // 180°: a=-1, b=0.
        let r180 = CGAffineTransform {
            a: -1.0,
            b: 0.0,
            c: 0.0,
            d: -1.0,
            tx: 0.0,
            ty: 0.0,
        };
        assert_eq!(rotation_from_transform(r180), Rotation::Cw180);
        // 270° CW (a=0, b=-1) normalizes from -90°.
        let r270 = CGAffineTransform {
            a: 0.0,
            b: -1.0,
            c: 1.0,
            d: 0.0,
            tx: 0.0,
            ty: 0.0,
        };
        assert_eq!(rotation_from_transform(r270), Rotation::Cw270);
    }
}

/// Integration tests that round-trip through real AVFoundation: synthesize a
/// tiny H.264 movie with `AVAssetWriter`, then decode it back. Hermetic — no
/// external asset needed.
#[cfg(test)]
mod integration {
    use super::*;
    use core::cmp::Ordering;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use objc2::AllocAnyThread;
    use objc2::runtime::{AnyObject, ProtocolObject};
    use objc2_av_foundation::{
        AVAssetWriter, AVAssetWriterInput, AVAssetWriterInputPixelBufferAdaptor,
        AVAssetWriterStatus, AVFileTypeQuickTimeMovie, AVMediaTypeVideo, AVVideoCodecKey,
        AVVideoCodecTypeH264, AVVideoColorPrimaries_SMPTE_C, AVVideoColorPrimariesKey,
        AVVideoColorPropertiesKey, AVVideoHeightKey, AVVideoTransferFunction_ITU_R_709_2,
        AVVideoTransferFunctionKey, AVVideoWidthKey, AVVideoYCbCrMatrix_ITU_R_601_4,
        AVVideoYCbCrMatrixKey,
    };
    use objc2_core_foundation::CFRetained;
    use objc2_core_media::{CMTime, CMTimeFlags};
    use objc2_core_video::{
        CVPixelBuffer, CVPixelBufferCreate, CVPixelBufferGetBaseAddress,
        CVPixelBufferGetBytesPerRow, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
        CVPixelBufferUnlockBaseAddress,
    };
    use objc2_foundation::{NSMutableDictionary, NSNumber, NSString, NSURL};

    fn temp_movie(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("cutlass_avf_{tag}_{nanos}.mov"))
    }

    /// Synthesize `frames` solid-gray BGRA frames at `fps` into an H.264 movie.
    /// Exercises the same CoreMedia/CoreVideo stack the decoder reads back. When
    /// `tag_601` is set, the file is tagged with SD/Rec.601 color properties so
    /// the reader's color parsing can be checked against a non-default value.
    #[allow(deprecated)] // finishWriting() (sync) is fine for a test fixture.
    fn write_test_movie(
        path: &std::path::Path,
        w: u32,
        h: u32,
        frames: u32,
        fps: i32,
        tag_601: bool,
    ) {
        let _ = std::fs::remove_file(path); // AVAssetWriter refuses an existing file.

        let url = NSURL::fileURLWithPath(&NSString::from_str(path.to_str().unwrap()));
        let file_type = unsafe { AVFileTypeQuickTimeMovie }.expect("AVFileTypeQuickTimeMovie");
        let writer = unsafe {
            AVAssetWriter::initWithURL_fileType_error(AVAssetWriter::alloc(), &url, file_type)
        }
        .expect("create writer");

        // Output settings: { codec: H.264, width, height }.
        let settings = NSMutableDictionary::<NSString, AnyObject>::new();
        let codec_key = unsafe { AVVideoCodecKey }.expect("AVVideoCodecKey");
        let codec_h264 = unsafe { AVVideoCodecTypeH264 }.expect("AVVideoCodecTypeH264");
        let width_key = unsafe { AVVideoWidthKey }.expect("AVVideoWidthKey");
        let height_key = unsafe { AVVideoHeightKey }.expect("AVVideoHeightKey");
        let width_num = NSNumber::numberWithInt(w as i32);
        let height_num = NSNumber::numberWithInt(h as i32);
        unsafe {
            settings.setObject_forKey(codec_h264, ProtocolObject::from_ref(codec_key));
            settings.setObject_forKey(&width_num, ProtocolObject::from_ref(width_key));
            settings.setObject_forKey(&height_num, ProtocolObject::from_ref(height_key));
        }

        // Optional SD/Rec.601 color tagging: SMPTE-C primaries, 709 transfer,
        // 601 matrix. AVFoundation requires all three together.
        if tag_601 {
            let color_props = NSMutableDictionary::<NSString, AnyObject>::new();
            let prim_key = unsafe { AVVideoColorPrimariesKey }.expect("AVVideoColorPrimariesKey");
            let prim_val =
                unsafe { AVVideoColorPrimaries_SMPTE_C }.expect("AVVideoColorPrimaries_SMPTE_C");
            let trc_key =
                unsafe { AVVideoTransferFunctionKey }.expect("AVVideoTransferFunctionKey");
            let trc_val = unsafe { AVVideoTransferFunction_ITU_R_709_2 }
                .expect("AVVideoTransferFunction_ITU_R_709_2");
            let mat_key = unsafe { AVVideoYCbCrMatrixKey }.expect("AVVideoYCbCrMatrixKey");
            let mat_val =
                unsafe { AVVideoYCbCrMatrix_ITU_R_601_4 }.expect("AVVideoYCbCrMatrix_ITU_R_601_4");
            unsafe {
                color_props.setObject_forKey(prim_val, ProtocolObject::from_ref(prim_key));
                color_props.setObject_forKey(trc_val, ProtocolObject::from_ref(trc_key));
                color_props.setObject_forKey(mat_val, ProtocolObject::from_ref(mat_key));
            }
            let props_key =
                unsafe { AVVideoColorPropertiesKey }.expect("AVVideoColorPropertiesKey");
            unsafe {
                settings.setObject_forKey(&color_props, ProtocolObject::from_ref(props_key));
            }
        }

        let media_video = unsafe { AVMediaTypeVideo }.expect("AVMediaTypeVideo");
        let input = unsafe {
            AVAssetWriterInput::initWithMediaType_outputSettings(
                AVAssetWriterInput::alloc(),
                media_video,
                Some(&settings),
            )
        };
        unsafe { input.setExpectsMediaDataInRealTime(false) };

        let adaptor = unsafe {
            AVAssetWriterInputPixelBufferAdaptor::initWithAssetWriterInput_sourcePixelBufferAttributes(
                AVAssetWriterInputPixelBufferAdaptor::alloc(),
                &input,
                None,
            )
        };

        assert!(
            unsafe { writer.canAddInput(&input) },
            "writer rejects input"
        );
        unsafe { writer.addInput(&input) };
        assert!(unsafe { writer.startWriting() }, "startWriting failed");
        unsafe { writer.startSessionAtSourceTime(cm_zero()) };

        for i in 0..frames {
            let pb = make_bgra_buffer(w, h);
            // Few frames + non-realtime: readiness is immediate, but be safe.
            let mut spins = 0;
            while !unsafe { input.isReadyForMoreMediaData() } && spins < 1000 {
                std::thread::sleep(std::time::Duration::from_millis(1));
                spins += 1;
            }
            let pts = CMTime {
                value: i as i64,
                timescale: fps,
                flags: CMTimeFlags::Valid,
                epoch: 0,
            };
            assert!(
                unsafe { adaptor.appendPixelBuffer_withPresentationTime(&pb, pts) },
                "append frame {i} failed"
            );
        }

        unsafe { input.markAsFinished() };
        assert!(unsafe { writer.finishWriting() }, "finishWriting failed");
        assert_eq!(
            unsafe { writer.status() },
            AVAssetWriterStatus::Completed,
            "writer error: {:?}",
            unsafe { writer.error() }.map(|e| e.localizedDescription().to_string())
        );
    }

    /// A solid-gray BGRA pixel buffer.
    fn make_bgra_buffer(w: u32, h: u32) -> CFRetained<CVPixelBuffer> {
        let mut out: *mut CVPixelBuffer = std::ptr::null_mut();
        let r = unsafe {
            CVPixelBufferCreate(
                None,
                w as usize,
                h as usize,
                FOURCC_BGRA,
                None,
                NonNull::from(&mut out),
            )
        };
        assert_eq!(r, 0, "CVPixelBufferCreate failed");
        let pb = unsafe { CFRetained::from_raw(NonNull::new(out).expect("null pixel buffer")) };

        unsafe {
            CVPixelBufferLockBaseAddress(&pb, CVPixelBufferLockFlags(0));
            let base = CVPixelBufferGetBaseAddress(&pb);
            let bytes_per_row = CVPixelBufferGetBytesPerRow(&pb);
            if let Some(base) = NonNull::new(base) {
                std::ptr::write_bytes(base.as_ptr() as *mut u8, 128, bytes_per_row * h as usize);
            }
            CVPixelBufferUnlockBaseAddress(&pb, CVPixelBufferLockFlags(0));
        }
        pb
    }

    #[test]
    fn opens_probes_and_decodes_first_frame() {
        let path = temp_movie("decode");
        write_test_movie(&path, 320, 240, 6, 30, false);

        let mut dec = AvfDecoder::open(&path, OutputMode::Cpu).expect("open");
        assert_eq!(dec.info().display_size, (320, 240));
        assert!(dec.info().frame_rate.as_f64() > 0.0);

        let frame = dec.next_frame().expect("decode").expect("first frame");
        assert_eq!(frame.width(), 320);
        assert_eq!(frame.height(), 240);
        assert!(frame.is_cpu());
        // H.264 decodes to biplanar NV12 on Apple.
        assert_eq!(frame.format, PixelFormat::Nv12);
        let img = frame.cpu().expect("cpu image");
        assert_eq!(img.planes.len(), 2);
        assert!(img.planes.iter().all(Plane::is_well_formed));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn sequential_decode_advances_pts() {
        let path = temp_movie("seq");
        write_test_movie(&path, 160, 120, 8, 30, false);

        let mut dec = AvfDecoder::open(&path, OutputMode::Cpu).expect("open");
        let f0 = dec.next_frame().expect("decode").expect("frame 0");
        let f1 = dec.next_frame().expect("decode").expect("frame 1");
        assert_eq!(f0.pts.compare(f1.pts), Ordering::Less, "pts must advance");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn frame_at_lands_at_or_after_target() {
        let path = temp_movie("seek");
        write_test_movie(&path, 160, 120, 30, 30, false);

        let mut dec = AvfDecoder::open(&path, OutputMode::Cpu).expect("open");
        // ~frame 10 (10/30 s), expressed at a *different* rate to exercise the
        // rate-agnostic comparison.
        let target = RationalTime::new(10, Rational::FPS_30);
        let frame = dec.frame_at(target).expect("frame_at").expect("frame");
        assert_ne!(
            frame.pts.compare(target),
            Ordering::Less,
            "decoded frame must be at or after the seek target"
        );

        let _ = std::fs::remove_file(&path);
    }

    /// Sequential `frame_at` targets (the playback/scrub pattern) must land on
    /// the same frames as a fresh seek per target — the roll-forward fast path
    /// may change cost, never results.
    #[test]
    fn frame_at_roll_forward_matches_seek_per_target() {
        let path = temp_movie("roll");
        write_test_movie(&path, 160, 120, 30, 30, false);

        let mut rolled = AvfDecoder::open(&path, OutputMode::Cpu).expect("open rolled");
        let mut seeked = AvfDecoder::open(&path, OutputMode::Cpu).expect("open seeked");

        for value in [0i64, 2, 3, 7, 8, 15, 22, 29] {
            let target = RationalTime::new(value, Rational::FPS_30);
            let a = rolled.frame_at(target).expect("frame_at").expect("frame");
            // Byte-identical reference: explicit seek + walk on the other decoder.
            seeked.seek(target).expect("seek");
            let b = loop {
                let f = seeked.next_frame().expect("decode").expect("frame");
                if f.pts.compare(target) != Ordering::Less {
                    break f;
                }
            };
            assert_eq!(a.pts, b.pts, "target {value}");
        }

        let _ = std::fs::remove_file(&path);
    }

    /// Backward and repeated targets fall off the roll-forward path and must
    /// still land correctly (via a real seek).
    #[test]
    fn frame_at_handles_backward_and_repeated_targets() {
        let path = temp_movie("back");
        write_test_movie(&path, 160, 120, 30, 30, false);

        let mut dec = AvfDecoder::open(&path, OutputMode::Cpu).expect("open");
        for value in [20i64, 5, 5, 12, 0] {
            let target = RationalTime::new(value, Rational::FPS_30);
            let frame = dec.frame_at(target).expect("frame_at").expect("frame");
            assert_ne!(frame.pts.compare(target), Ordering::Less, "target {value}");
            // Solid-color 30fps fixture: frame_at must land exactly on `value`.
            assert_eq!(
                cutlass_core::resample(frame.pts, Rational::FPS_30).value,
                value
            );
        }

        // Rolling past the end of the stream reports end-of-stream, and the
        // decoder recovers with a later backward target.
        let last = RationalTime::new(29, Rational::FPS_30);
        dec.frame_at(last).expect("frame_at").expect("last frame");
        assert!(
            dec.frame_at(RationalTime::new(31, Rational::FPS_30))
                .expect("frame_at")
                .is_none(),
            "past-the-end target must be None"
        );
        let recovered = dec
            .frame_at(RationalTime::new(3, Rational::FPS_30))
            .expect("frame_at")
            .expect("frame");
        assert_eq!(
            cutlass_core::resample(recovered.pts, Rational::FPS_30).value,
            3
        );

        let _ = std::fs::remove_file(&path);
    }

    /// Sustained seek churn — many reader rebuilds in a tight loop — must not
    /// degrade the decoder. Guards the `cancelReading` lifecycle: without it,
    /// each rebuild leaks an in-flight decode pipeline, and enough of those
    /// exhaust process-wide decode resources ("Cannot Decode" on every
    /// subsequent read; reproduced with ~300 rebuilds against a 4K source).
    /// The fixture is too small to trip the ceiling itself, so this is a
    /// smoke test of the cancel path plus decoder health after churn.
    #[test]
    fn seek_churn_keeps_decoder_healthy() {
        let path = temp_movie("churn");
        write_test_movie(&path, 160, 120, 30, 30, false);

        let mut dec = AvfDecoder::open(&path, OutputMode::Cpu).expect("open");
        for i in 0..150 {
            let target = RationalTime::new((i * 7) % 30, Rational::FPS_30);
            dec.seek(target).expect("seek");
            dec.next_frame().expect("decode").expect("frame");
        }
        // Still healthy: an exact frame_at keeps landing correctly.
        let frame = dec
            .frame_at(RationalTime::new(11, Rational::FPS_30))
            .expect("frame_at")
            .expect("frame");
        assert_eq!(
            cutlass_core::resample(frame.pts, Rational::FPS_30).value,
            11
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn gpu_mode_hands_back_a_live_pixel_buffer() {
        let path = temp_movie("gpu");
        write_test_movie(&path, 320, 240, 6, 30, false);

        let mut dec = AvfDecoder::open(&path, OutputMode::Gpu).expect("open");
        let frame = dec.next_frame().expect("decode").expect("first frame");

        assert!(frame.is_gpu(), "gpu mode must yield a GPU frame");
        assert_eq!(frame.width(), 320);
        assert_eq!(frame.height(), 240);

        let surface = frame.gpu().expect("gpu surface");
        assert_eq!(surface.kind(), GpuSurfaceKind::CoreVideoPixelBuffer);
        assert_ne!(surface.handle(), 0, "handle must be a real pointer");

        // The handle must be a usable CVPixelBuffer kept alive by the frame's
        // retained wrapper: reinterpret it and read its dimensions back.
        let pb: &CVPixelBuffer =
            unsafe { (surface.handle() as *const CVPixelBuffer).as_ref() }.expect("live buffer");
        assert_eq!(CVPixelBufferGetWidth(pb), 320);
        assert_eq!(CVPixelBufferGetHeight(pb), 240);

        let _ = std::fs::remove_file(&path);
    }

    /// Manual benchmark, not a correctness test: per-target latency of
    /// sequential `frame_at` (roll-forward) vs a seek per target. Run with
    /// `cargo test -p cutlass-decoder --release -- --ignored bench_roll --nocapture`.
    #[test]
    #[ignore = "manual benchmark"]
    fn bench_roll_forward_vs_seek_per_target() {
        use std::time::Instant;

        let path = temp_movie("bench");
        let frames = 300; // 10s @ 30fps
        write_test_movie(&path, 320, 180, frames, 30, false);

        let targets: Vec<RationalTime> = (0..frames as i64)
            .map(|i| RationalTime::new(i, Rational::FPS_30))
            .collect();

        let mut rolled = AvfDecoder::open(&path, OutputMode::Cpu).expect("open");
        let start = Instant::now();
        for &t in &targets {
            rolled.frame_at(t).expect("frame_at").expect("frame");
        }
        let roll_ms = start.elapsed().as_secs_f64() * 1000.0 / targets.len() as f64;

        let mut seeked = AvfDecoder::open(&path, OutputMode::Cpu).expect("open");
        let start = Instant::now();
        for &t in &targets {
            seeked.seek(t).expect("seek");
            loop {
                let f = seeked.next_frame().expect("decode").expect("frame");
                if f.pts.compare(t) != Ordering::Less {
                    break;
                }
            }
        }
        let seek_ms = start.elapsed().as_secs_f64() * 1000.0 / targets.len() as f64;

        println!(
            "sequential playback, {frames} frames: frame_at {roll_ms:.3} ms/frame, \
             seek-per-frame {seek_ms:.3} ms/frame ({:.1}x)",
            seek_ms / roll_ms
        );

        let _ = std::fs::remove_file(&path);
    }

    /// Open-GOP H.264 fixture (`tests/fixtures/opengop_h264.mp4`; 320×180,
    /// 24 fps, 10 s, x264 `open_gop=1` with the reference structure of a
    /// real-world 4K export that hard-failed every seek): sync samples every
    /// 2 s, only frame 0 IDR, and the sync samples at 4 s and 8 s carry three
    /// leading B-frames referencing the previous GOP — the start points
    /// VideoToolbox refuses with `AVErrorDecodeFailed` ("Cannot Decode").
    fn opengop_fixture() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/opengop_h264.mp4")
    }

    const OPENGOP_FPS: Rational = Rational::new(24, 1);

    /// The regression this fixture exists for: a seek whose effective sync
    /// sample has leading B-frames must recover (back off, walk, trim) and
    /// land on the exact requested frame — before the recovery path existed,
    /// this exact call returned "Cannot Decode".
    #[test]
    fn open_gop_seek_point_failure_recovers_exactly() {
        let mut dec = AvfDecoder::open(&opengop_fixture(), OutputMode::Cpu).expect("open");
        // 5.0 s: inside the GOP whose sync sample (4.0 s) is undecodable as
        // a start point.
        let target = RationalTime::new(120, OPENGOP_FPS);
        let frame = dec.frame_at(target).expect("must recover").expect("frame");
        assert_eq!(cutlass_core::resample(frame.pts, OPENGOP_FPS).value, 120);

        // The decoder is healthy afterwards: sequential decode continues.
        let next = dec.next_frame().expect("decode").expect("frame");
        assert_eq!(cutlass_core::resample(next.pts, OPENGOP_FPS).value, 121);
    }

    /// Random access across good and bad GOPs, both directions, lands
    /// exactly everywhere — the whole timeline is reachable, not just the
    /// GOPs whose sync samples happen to be closed.
    #[test]
    fn open_gop_random_access_reaches_every_gop() {
        let mut dec = AvfDecoder::open(&opengop_fixture(), OutputMode::Cpu).expect("open");
        // 5.5 s and 9 s sit in bad GOPs (syncs 4 s / 8 s); the rest are good
        // or sequential rolls. Mix directions to exercise seek and roll.
        for value in [132i64, 12, 216, 60, 220, 100, 230] {
            let target = RationalTime::new(value, OPENGOP_FPS);
            let frame = dec
                .frame_at(target)
                .unwrap_or_else(|e| panic!("frame_at {value}: {e}"))
                .unwrap_or_else(|| panic!("frame_at {value}: no frame"));
            assert_eq!(
                cutlass_core::resample(frame.pts, OPENGOP_FPS).value,
                value,
                "target {value}"
            );
        }
    }

    /// The nearest-sync scrub path takes the same reader seek and must
    /// recover the same way.
    #[test]
    fn open_gop_frame_at_nearest_recovers() {
        let mut dec = AvfDecoder::open(&opengop_fixture(), OutputMode::Cpu).expect("open");
        let target = RationalTime::new(216, OPENGOP_FPS); // 9.0 s, bad GOP
        let frame = dec
            .frame_at_nearest(target)
            .expect("must recover")
            .expect("frame");
        // AVAssetReader trims to the requested range, so nearest on this
        // backend is the frame at/after the target.
        assert_eq!(cutlass_core::resample(frame.pts, OPENGOP_FPS).value, 216);
    }

    #[test]
    fn hermetic_movie_reports_no_audio_track() {
        let path = temp_movie("noaudio");
        write_test_movie(&path, 160, 120, 4, 30, false);
        let dec = AvfDecoder::open(&path, OutputMode::Cpu).expect("open");
        assert!(!dec.has_audio_track(), "video-only fixture");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn probes_color_metadata_from_format_description() {
        let path = temp_movie("color");
        // Tag the file SD/Rec.601 — distinct from the BT.709 default, so a pass
        // proves the tags were actually read (not just defaulted).
        write_test_movie(&path, 160, 120, 4, 30, true);

        let dec = AvfDecoder::open(&path, OutputMode::Cpu).expect("open");
        let color = dec.info().color;
        assert_eq!(color.matrix, MatrixCoefficients::Bt601, "matrix from tag");
        assert_eq!(
            color.primaries,
            ColorPrimaries::Bt601_525,
            "SMPTE-C primaries from tag"
        );
        assert_eq!(color.transfer, TransferFunction::Bt709, "transfer from tag");
        assert!(!color.is_hdr());

        let _ = std::fs::remove_file(&path);
    }
}
