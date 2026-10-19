//! Apple backend: `AVAssetReader` (demux + VideoToolbox decode) behind
//! [`cutlass_core::VideoDecoder`].
//!
//! `AVAssetReader` reads decoded `CVPixelBuffer`s forward from a configurable
//! start time. We don't pass output settings, so VideoToolbox vends frames in
//! its native decoded format (NV12 video/full range, or 10-bit/`BGRA` for some
//! sources); the format is detected per frame from the pixel buffer. Seeking
//! rebuilds the reader at the new start time — `AVAssetReader` is forward-only,
//! so this is the simplest correct seek (a faster random-access path via
//! `resetForReadingTimeRanges` can come later).
//!
//! This backend is additive to the FFmpeg desktop path: on macOS both compile,
//! and the engine still drives FFmpeg for desktop today. `AvfDecoder` is what
//! the mobile shells (and a future macOS hardware path) use.

use std::path::Path;
use std::ptr::NonNull;
use std::slice;

use objc2::AllocAnyThread;
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_av_foundation::{
    AVAssetReader, AVAssetReaderStatus, AVAssetReaderTrackOutput, AVAssetTrack, AVMediaTypeVideo,
    AVURLAsset,
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
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString, NSURL};

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
        })
    }

    /// Pull the next decoded sample buffer's pixel buffer, skipping any
    /// marker-only buffers, and convert it to a [`VideoFrame`].
    fn next_pixel_frame(&mut self) -> Result<Option<VideoFrame>, DecodeError> {
        if !self.started {
            if !unsafe { self.reader.startReading() } {
                return Err(self.reader_error("startReading failed"));
            }
            self.started = true;
        }

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

            let frame = self.image_buffer_to_frame(image_buffer, pts)?;
            return Ok(Some(frame));
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

    /// Build a [`DecodeError`] from the reader's `NSError`, if any.
    fn reader_error(&self, context: &str) -> DecodeError {
        let detail = unsafe { self.reader.error() }
            .map(|e| e.localizedDescription().to_string())
            .unwrap_or_else(|| context.to_string());
        DecodeError::Decode(format!("{context}: {detail}"))
    }
}

impl VideoDecoder for AvfDecoder {
    fn info(&self) -> &SourceInfo {
        &self.info
    }

    fn seek(&mut self, target: RationalTime) -> Result<(), DecodeError> {
        let start = rational_to_cmtime(target);
        let (reader, output) = build_reader(&self.asset, &self.track, Some(start))?;
        self.reader = reader;
        self.output = output;
        self.started = false;
        Ok(())
    }

    fn next_frame(&mut self) -> Result<Option<VideoFrame>, DecodeError> {
        self.next_pixel_frame()
    }
}

/// Find the first video track of an asset (synchronous probe).
fn first_video_track(asset: &AVURLAsset) -> Option<Retained<AVAssetTrack>> {
    let media_video = unsafe { AVMediaTypeVideo }?;
    #[allow(deprecated)]
    let tracks = unsafe { asset.tracksWithMediaType(media_video) };
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

/// Exact frame rate from the track's minimum frame duration, falling back to
/// the (approximate) nominal rate.
fn frame_rate_of(track: &AVAssetTrack) -> Rational {
    let min_dur = unsafe { track.minFrameDuration() };
    if min_dur.value > 0 && min_dur.timescale > 0 {
        // fps = timescale / value (frames per second).
        return Rational::new(min_dur.timescale, min_dur.value as i32);
    }
    let nominal = unsafe { track.nominalFrameRate() };
    if nominal > 0.0 {
        Rational::new((nominal * 1000.0).round() as i32, 1000)
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
    CMTime {
        value: t.value * i64::from(t.rate.den.max(1)),
        timescale: t.rate.num.max(1),
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
