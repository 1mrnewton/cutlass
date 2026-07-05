//! Windows video decoding via Media Foundation's **Source Reader**.
//!
//! [`IMFSourceReader`] is the Windows analogue of Apple's `AVAssetReader`: one
//! object that demuxes the container *and* drives the hardware (or software)
//! decoder, so we feed it a URL and pull decoded samples. Control stays in Rust
//! behind [`cutlass_core::VideoDecoder`]; only the codec call is native.
//!
//! ## What this backend does today
//!
//! - Forces an **uncompressed NV12** output type, which makes the reader insert
//!   a decoder (+ a converter when the codec is 10-bit), and hands back
//!   system-memory planes.
//! - Reads each sample's [`IMFMediaBuffer`] stride-aware (via [`IMF2DBuffer`]
//!   when available, falling back to a flat `Lock`) into [`FrameData::Cpu`].
//! - Probes size / rotation / frame-rate / colorimetry from the negotiated
//!   output [`IMFMediaType`] and duration from the presentation descriptor
//!   (`MF_PD_DURATION`), and seeks with `SetCurrentPosition` (100-ns units).
//! - Answers [`VideoDecoder::frame_at`] with the shared roll-forward policy
//!   ([`crate::seek`]) so sequential playback / forward scrubbing never pays a
//!   seek-plus-GOP-re-decode per frame.
//!
//! ## What it does not do yet
//!
//! The **GPU (DXGI) zero-copy path is intentionally stubbed** — see
//! [`WmfDecoder::sample_to_frame`] for the rationale. The CPU path is the one
//! verifiable on a non-Windows build host (`cargo check --target
//! x86_64-pc-windows-gnu`); the GPU path needs a Windows GPU to validate and a
//! small `cutlass-core` addition (a per-frame subresource index) to be correct.
//!
//! ## Time base
//!
//! Media Foundation timestamps are in **100-nanosecond units** (`hns`), so every
//! frame's PTS rate — and [`SourceInfo::time_base`] — is [`HNS_PER_SEC`].

use core::slice;
use std::path::Path;
use std::sync::Once;

use windows::Win32::Media::MediaFoundation::{
    IMF2DBuffer, IMFAttributes, IMFByteStream, IMFMediaBuffer, IMFMediaType, IMFSample,
    IMFSourceReader, MF_ACCESSMODE_READ, MF_FILEFLAGS_NONE, MF_MT_FRAME_RATE, MF_MT_FRAME_SIZE,
    MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE, MF_MT_TRANSFER_FUNCTION, MF_MT_VIDEO_NOMINAL_RANGE,
    MF_MT_VIDEO_PRIMARIES, MF_MT_VIDEO_ROTATION, MF_MT_YUV_MATRIX, MF_OPENMODE_FAIL_IF_NOT_EXIST,
    MF_PD_DURATION, MF_SOURCE_READER_ALL_STREAMS, MF_SOURCE_READER_FIRST_AUDIO_STREAM,
    MF_SOURCE_READER_FIRST_VIDEO_STREAM, MF_SOURCE_READER_MEDIASOURCE,
    MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED, MF_SOURCE_READERF_ENDOFSTREAM, MF_VERSION,
    MFCreateFile, MFCreateMediaType, MFCreateSourceReaderFromByteStream,
    MFCreateSourceReaderFromURL, MFMediaType_Video, MFNominalRange_0_255,
    MFSTARTUP_FULL, MFStartup, MFVideoFormat_NV12, MFVideoFormat_P010,
    MFVideoPrimaries_BT470_2_SysBG, MFVideoPrimaries_BT709, MFVideoPrimaries_BT2020,
    MFVideoPrimaries_DCI_P3, MFVideoPrimaries_SMPTE170M, MFVideoTransFunc_709,
    MFVideoTransFunc_2084, MFVideoTransFunc_HLG, MFVideoTransFunc_sRGB,
    MFVideoTransferMatrix_BT601, MFVideoTransferMatrix_BT709, MFVideoTransferMatrix_BT2020_10,
    MFVideoTransferMatrix_BT2020_12,
};
use windows::Win32::System::Com::StructuredStorage::PROPVARIANT;
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx};
use windows::core::{GUID, HSTRING, Interface};

use cutlass_core::{
    ColorPrimaries, ColorRange, ColorSpace, CpuImage, DecodeError, FrameData, MatrixCoefficients,
    PixelFormat, Plane, Rational, RationalTime, Rect, Rotation, SourceInfo, TransferFunction,
    VideoDecoder, VideoFrame,
};

use crate::OutputMode;

/// Media Foundation's timestamp tick rate: 100-nanosecond units per second.
const HNS_PER_SEC: i64 = 10_000_000;

/// Upper bound on consecutive empty `ReadSample` returns (stream ticks / gaps)
/// before we give up, so a pathological source can't hang the decode worker.
const MAX_EMPTY_READS: usize = 1024;

/// A pull-based decoder over one video file, backed by Media Foundation.
pub struct WmfDecoder {
    reader: IMFSourceReader,
    /// The magic "first video stream" selector, reused for every reader call.
    stream: u32,
    info: SourceInfo,
    mode: OutputMode,
    eos: bool,
    /// PTS of the last emitted frame — the [`crate::seek`] roll-forward anchor.
    /// Cleared on [`VideoDecoder::seek`] (decode position moved).
    last_pts: Option<RationalTime>,
}

// SAFETY: the Source Reader and its COM children are owned and touched by a
// single thread at a time — the decode-ahead worker the engine parks this
// decoder on. We open in the multithreaded apartment (see `open`) so the COM
// objects may legally travel between MTA worker threads, and we never share
// `&self` across threads, which satisfies `VideoDecoder: Send` without `Sync`.
unsafe impl Send for WmfDecoder {}

impl WmfDecoder {
    /// Open `path` and probe its first video track.
    pub fn open(path: &Path, mode: OutputMode) -> Result<Self, DecodeError> {
        ensure_mf_platform();
        // The source resolver instantiates COM objects on the calling thread, so
        // it must have COM initialized. Join the MTA so the decoder can live on
        // any worker thread; best-effort — if the host already put this thread in
        // an STA, MF still works and `CoInitializeEx` just reports the conflict.
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        }

        // `None: Param<IMFAttributes>` — default source-reader configuration.
        let reader = open_source_reader(path, None)?;

        let stream = MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32;

        // Decode only the video track (leave audio deselected so MF never spins
        // up an audio decoder for it).
        unsafe {
            reader
                .SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS.0 as u32, false)
                .map_err(open_err)?;
            reader.SetStreamSelection(stream, true).map_err(open_err)?;
        }

        // Request uncompressed NV12. Setting an uncompressed subtype is what
        // makes the reader insert the decoder (and a converter for 10-bit
        // sources); without it the reader would hand back compressed samples.
        let out_type = unsafe { MFCreateMediaType() }.map_err(open_err)?;
        unsafe {
            out_type
                .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)
                .map_err(open_err)?;
            out_type
                .SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_NV12)
                .map_err(open_err)?;
            reader
                .SetCurrentMediaType(stream, None, &out_type)
                .map_err(|e| {
                    DecodeError::Unsupported(format!("could not negotiate NV12 output: {e}"))
                })?;
        }

        let info = probe(&reader, stream)?;

        Ok(Self {
            reader,
            stream,
            info,
            mode,
            eos: false,
            last_pts: None,
        })
    }

    /// Turn one decoded [`IMFSample`] into a [`VideoFrame`].
    fn sample_to_frame(
        &self,
        sample: &IMFSample,
        pts: RationalTime,
    ) -> Result<VideoFrame, DecodeError> {
        match self.mode {
            // The zero-copy path is deliberately not wired up yet. A D3D11-backed
            // Source Reader hands back samples whose buffer is an `IMFDXGIBuffer`
            // wrapping an `ID3D11Texture2D`, but two problems block a correct,
            // testable implementation right now:
            //
            //  1. Hardware decoders output into a *texture array*; the live frame
            //     is one array slice identified by `IMFDXGIBuffer::GetSubresourceIndex`.
            //     `cutlass_core::GpuSurface` carries a single opaque handle with no
            //     slot for that subresource index, so the renderer couldn't select
            //     the right slice. Fixing this means a small core addition.
            //  2. wgpu uses D3D12 on Windows while MF decodes on D3D11; bridging
            //     the texture means a shared NT handle (`IDXGIResource1`), which
            //     needs a Windows GPU to validate and can't be exercised from this
            //     macOS build host.
            //
            // Until both are addressed, GPU mode is an explicit error rather than a
            // silent CPU fallback (the caller chose the mode deliberately).
            OutputMode::Gpu => Err(DecodeError::unsupported(
                "Media Foundation GPU/DXGI output is not implemented yet; open with OutputMode::Cpu",
            )),
            OutputMode::Cpu => {
                // Flatten any multi-buffer sample into one contiguous buffer.
                let buffer = unsafe { sample.ConvertToContiguousBuffer() }.map_err(decode_err)?;
                let (width, height) = self.info.coded_size;
                let planes = read_planes(&buffer, self.info.pixel_format, width, height)?;
                Ok(VideoFrame::new(
                    pts,
                    self.info.pixel_format,
                    self.info.color,
                    (width, height),
                    Rect::from_size(width, height),
                    self.info.rotation,
                    FrameData::Cpu(CpuImage::new(planes)),
                ))
            }
        }
    }
}

impl VideoDecoder for WmfDecoder {
    fn info(&self) -> &SourceInfo {
        &self.info
    }

    fn seek(&mut self, target: RationalTime) -> Result<(), DecodeError> {
        // `GUID_NULL` selects the 100-ns time format; the position rides in a
        // `VT_I8` PROPVARIANT. The reader lands at the keyframe at/before the
        // target and flushes, matching the trait contract.
        let position = PROPVARIANT::from(to_hns(target));
        let time_format = GUID::from_u128(0);
        unsafe { self.reader.SetCurrentPosition(&time_format, &position) }.map_err(seek_err)?;
        self.eos = false;
        self.last_pts = None;
        Ok(())
    }

    fn next_frame(&mut self) -> Result<Option<VideoFrame>, DecodeError> {
        if self.eos {
            return Ok(None);
        }

        for _ in 0..MAX_EMPTY_READS {
            let mut stream_flags: u32 = 0;
            let mut timestamp: i64 = 0;
            let mut sample: Option<IMFSample> = None;
            unsafe {
                self.reader.ReadSample(
                    self.stream,
                    0,
                    None,
                    Some(&mut stream_flags),
                    Some(&mut timestamp),
                    Some(&mut sample),
                )
            }
            .map_err(decode_err)?;

            // The decoder renegotiated (resolution / format / color changed
            // mid-stream); re-probe so frames after the change are described
            // correctly.
            if stream_flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED.0 as u32 != 0 {
                self.info = probe(&self.reader, self.stream)?;
            }

            if stream_flags & MF_SOURCE_READERF_ENDOFSTREAM.0 as u32 != 0 {
                self.eos = true;
            }

            match sample {
                Some(sample) => {
                    let pts = RationalTime::new(timestamp, self.info.time_base);
                    let frame = self.sample_to_frame(&sample, pts)?;
                    self.last_pts = Some(pts);
                    return Ok(Some(frame));
                }
                // A stream tick or gap can return `S_OK` with no sample; keep
                // draining unless that empty read was also the end of stream.
                None if self.eos => return Ok(None),
                None => continue,
            }
        }

        Err(DecodeError::Decode(
            "source reader produced no frame within the read budget".into(),
        ))
    }

    fn frame_at(&mut self, target: RationalTime) -> Result<Option<VideoFrame>, DecodeError> {
        let last_pts = self.last_pts;
        crate::seek::frame_at_rolling(self, last_pts, target)
    }
}

/// Start the Media Foundation platform once per process. The matching
/// `MFShutdown` is intentionally skipped: the platform lives for the process
/// (like a global logger), and decoders open/close freely against it.
fn ensure_mf_platform() {
    static MF_PLATFORM: Once = Once::new();
    MF_PLATFORM.call_once(|| unsafe {
        let _ = MFStartup(MF_VERSION, MFSTARTUP_FULL);
    });
}

/// Whether the file at `path` carries at least one audio stream. Used by the
/// probe to set [`crate::MediaProbe::has_audio`]; `false` on any error.
///
/// Opens a throwaway source reader and asks for the first audio stream's
/// native media type — present iff the container has an audio stream. No
/// decoder is spun up (no output type is set, nothing is read).
pub(crate) fn has_audio_track(path: &Path) -> bool {
    ensure_mf_platform();
    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
    }
    let Ok(reader) = open_source_reader(path, None) else {
        return false;
    };
    unsafe {
        reader
            .GetNativeMediaType(MF_SOURCE_READER_FIRST_AUDIO_STREAM.0 as u32, 0)
            .is_ok()
    }
}

/// Open a Media Foundation source reader for the local file at `path`,
/// optionally with reader-configuration `attributes`.
///
/// Prefers a **self-opened byte stream** over [`MFCreateSourceReaderFromURL`].
/// Once the process brings up a Direct3D device (wgpu, in the desktop app),
/// MF's URL *source resolver* starts failing local-file opens with
/// `ERROR_BAD_NETPATH` (`0x80070035`) — it misroutes a plain `C:\…` path as if
/// it were a UNC/network path. Opening the file ourselves with [`MFCreateFile`]
/// and resolving the source from that byte stream sidesteps the URL resolver
/// entirely (plain file I/O is unaffected by the D3D init). The URL path stays
/// as a fallback for the pre-D3D case and anything the byte-stream resolver
/// can't handle (e.g. non-`file:` URLs, should we ever pass one).
pub(crate) fn open_source_reader(
    path: &Path,
    attributes: Option<&IMFAttributes>,
) -> Result<IMFSourceReader, DecodeError> {
    let url = path
        .to_str()
        .ok_or_else(|| DecodeError::Open("path is not valid UTF-8".into()))?;
    let url = HSTRING::from(url);

    match reader_from_byte_stream(&url, attributes) {
        Ok(reader) => Ok(reader),
        // `&HSTRING: Param<PCWSTR>`.
        Err(stream_err) => {
            unsafe { MFCreateSourceReaderFromURL(&url, attributes) }.map_err(|url_err| {
                DecodeError::Open(format!(
                    "byte-stream open failed ({stream_err}); URL open failed ({url_err})"
                ))
            })
        }
    }
}

/// Open `url` as a file byte stream and build a source reader from it, so the
/// source is resolved via `CreateObjectFromByteStream` rather than the URL
/// resolver. See [`open_source_reader`] for why this is preferred.
fn reader_from_byte_stream(
    url: &HSTRING,
    attributes: Option<&IMFAttributes>,
) -> Result<IMFSourceReader, windows::core::Error> {
    let stream: IMFByteStream = unsafe {
        MFCreateFile(
            MF_ACCESSMODE_READ,
            MF_OPENMODE_FAIL_IF_NOT_EXIST,
            MF_FILEFLAGS_NONE,
            url,
        )
    }?;
    unsafe { MFCreateSourceReaderFromByteStream(&stream, attributes) }
}

/// Source duration from the presentation descriptor (`MF_PD_DURATION`, a
/// `VT_UI8` count of 100-ns units), or `None` when the container doesn't
/// report one.
fn probe_duration(reader: &IMFSourceReader) -> Option<RationalTime> {
    let value = unsafe {
        reader.GetPresentationAttribute(MF_SOURCE_READER_MEDIASOURCE.0 as u32, &MF_PD_DURATION)
    }
    .ok()?;
    let hns = u64::try_from(&value).ok()?;
    (hns > 0).then(|| RationalTime::new(hns as i64, Rational::new(HNS_PER_SEC as i32, 1)))
}

/// Read static source properties from the reader's negotiated output type.
fn probe(reader: &IMFSourceReader, stream: u32) -> Result<SourceInfo, DecodeError> {
    let media_type = unsafe { reader.GetCurrentMediaType(stream) }.map_err(open_err)?;

    // FRAME_SIZE packs width in the high 32 bits, height in the low 32.
    let frame_size = unsafe { media_type.GetUINT64(&MF_MT_FRAME_SIZE) }.map_err(open_err)?;
    let width = (frame_size >> 32) as u32;
    let height = (frame_size & 0xffff_ffff) as u32;

    // FRAME_RATE packs numerator in the high 32 bits, denominator in the low 32.
    let frame_rate = match unsafe { media_type.GetUINT64(&MF_MT_FRAME_RATE) } {
        Ok(packed) => {
            let num = (packed >> 32) as u32;
            let den = (packed & 0xffff_ffff) as u32;
            Rational::new(num.max(1) as i32, den.max(1) as i32)
        }
        Err(_) => Rational::FPS_30,
    };

    // VIDEO_ROTATION is stored as degrees clockwise (0/90/180/270).
    let rotation = unsafe { media_type.GetUINT32(&MF_MT_VIDEO_ROTATION) }
        .ok()
        .and_then(|deg| Rotation::from_degrees(deg as i32))
        .unwrap_or(Rotation::None);

    let pixel_format = subtype_to_format(unsafe { media_type.GetGUID(&MF_MT_SUBTYPE) }.ok());
    let color = color_from_type(&media_type);

    Ok(SourceInfo {
        coded_size: (width, height),
        // MF reports a single frame size; the clean-aperture crop
        // (MF_MT_MINIMUM_DISPLAY_APERTURE) isn't parsed yet, so visible == coded.
        display_size: (width, height),
        rotation,
        pixel_format,
        color,
        frame_rate,
        time_base: Rational::new(HNS_PER_SEC as i32, 1),
        duration: probe_duration(reader),
    })
}

/// Pull colorimetry from the output type's color attributes, falling back to the
/// BT.709 SDR default for any attribute the source under-signals.
fn color_from_type(media_type: &IMFMediaType) -> ColorSpace {
    let mut color = ColorSpace::BT709;
    if let Ok(v) = unsafe { media_type.GetUINT32(&MF_MT_VIDEO_PRIMARIES) } {
        color.primaries = map_primaries(v);
    }
    if let Ok(v) = unsafe { media_type.GetUINT32(&MF_MT_TRANSFER_FUNCTION) } {
        color.transfer = map_transfer(v);
    }
    if let Ok(v) = unsafe { media_type.GetUINT32(&MF_MT_YUV_MATRIX) } {
        color.matrix = map_matrix(v);
    }
    if let Ok(v) = unsafe { media_type.GetUINT32(&MF_MT_VIDEO_NOMINAL_RANGE) } {
        color.range = map_range(v);
    }
    color
}

/// Copy a biplanar 4:2:0 sample (NV12 / P010) out of its media buffer.
///
/// Prefers [`IMF2DBuffer`] so we honor the decoder's real row pitch (which it
/// pads for alignment), and falls back to a flat `Lock` assuming an unpadded
/// `width * bytes-per-sample` stride.
fn read_planes(
    buffer: &IMFMediaBuffer,
    format: PixelFormat,
    width: u32,
    height: u32,
) -> Result<Vec<Plane>, DecodeError> {
    if let Ok(buffer_2d) = buffer.cast::<IMF2DBuffer>() {
        let mut scanline0: *mut u8 = core::ptr::null_mut();
        let mut pitch: i32 = 0;
        unsafe { buffer_2d.Lock2D(&mut scanline0, &mut pitch) }.map_err(decode_err)?;
        let planes = (|| -> Result<Vec<Plane>, DecodeError> {
            if scanline0.is_null() || pitch <= 0 {
                return Err(DecodeError::Decode(
                    "2D buffer returned a null base or non-positive pitch".into(),
                ));
            }
            // SAFETY: between Lock2D/Unlock2D the buffer guarantees `pitch` bytes
            // per row across the full biplanar extent.
            unsafe { copy_biplanar(scanline0, pitch as usize, height, None) }
        })();
        unsafe { buffer_2d.Unlock2D() }.map_err(decode_err)?;
        return planes;
    }

    let mut base: *mut u8 = core::ptr::null_mut();
    let mut max_len: u32 = 0;
    let mut cur_len: u32 = 0;
    unsafe { buffer.Lock(&mut base, Some(&mut max_len), Some(&mut cur_len)) }
        .map_err(decode_err)?;
    let planes = (|| -> Result<Vec<Plane>, DecodeError> {
        if base.is_null() {
            return Err(DecodeError::Decode("locked buffer is null".into()));
        }
        let stride = width as usize * format.bytes_per_sample();
        // SAFETY: the contiguous buffer holds `cur_len` valid bytes; copy_biplanar
        // verifies the biplanar extent fits before reading.
        unsafe { copy_biplanar(base, stride, height, Some(cur_len as usize)) }
    })();
    unsafe { buffer.Unlock() }.map_err(decode_err)?;
    planes
}

/// Copy `height` luma rows plus `ceil(height/2)` interleaved-chroma rows, each
/// `stride` bytes, out of a biplanar buffer starting at `base`.
///
/// # Safety
/// `base` must point at a readable region of at least `stride * (height +
/// ceil(height/2))` bytes (the caller holds the buffer lock). When `available`
/// is `Some`, that extent is checked against it first.
unsafe fn copy_biplanar(
    base: *mut u8,
    stride: usize,
    height: u32,
    available: Option<usize>,
) -> Result<Vec<Plane>, DecodeError> {
    let y_rows = height as usize;
    let chroma_rows = (height as usize).div_ceil(2);
    let needed = stride
        .checked_mul(y_rows + chroma_rows)
        .ok_or_else(|| DecodeError::Decode("plane size overflow".into()))?;
    if let Some(available) = available
        && available < needed
    {
        return Err(DecodeError::Decode(format!(
            "buffer too small for {height}-row biplanar frame: {available} < {needed} bytes"
        )));
    }

    let luma = unsafe { slice::from_raw_parts(base, stride * y_rows) }.to_vec();
    let chroma =
        unsafe { slice::from_raw_parts(base.add(stride * y_rows), stride * chroma_rows) }.to_vec();
    Ok(vec![
        Plane::new(luma, stride, y_rows),
        Plane::new(chroma, stride, chroma_rows),
    ])
}

/// Convert a [`RationalTime`] to Media Foundation's 100-ns ticks.
///
/// `seconds = value * den / num`, so `hns = value * den * 1e7 / num`. Computed in
/// `i128` to avoid overflow before truncating to the `i64` MF expects.
fn to_hns(target: RationalTime) -> i64 {
    let num = i128::from(target.rate.num.max(1));
    let den = i128::from(target.rate.den.max(1));
    ((i128::from(target.value) * den * i128::from(HNS_PER_SEC)) / num) as i64
}

/// Map an `MF_MT_SUBTYPE` GUID to our pixel format. We only ever request NV12,
/// but P010 is recognized so a future 10-bit request maps cleanly.
fn subtype_to_format(subtype: Option<GUID>) -> PixelFormat {
    match subtype {
        Some(guid) if guid == MFVideoFormat_P010 => PixelFormat::P010,
        _ => PixelFormat::Nv12,
    }
}

/// Map an `MFVideoPrimaries` code to [`ColorPrimaries`].
fn map_primaries(value: u32) -> ColorPrimaries {
    if value == MFVideoPrimaries_BT709.0 as u32 {
        ColorPrimaries::Bt709
    } else if value == MFVideoPrimaries_SMPTE170M.0 as u32 {
        ColorPrimaries::Bt601_525
    } else if value == MFVideoPrimaries_BT470_2_SysBG.0 as u32 {
        ColorPrimaries::Bt601_625
    } else if value == MFVideoPrimaries_BT2020.0 as u32 {
        ColorPrimaries::Bt2020
    } else if value == MFVideoPrimaries_DCI_P3.0 as u32 {
        ColorPrimaries::DisplayP3
    } else {
        ColorPrimaries::Unspecified
    }
}

/// Map an `MFVideoTransferFunction` code to [`TransferFunction`].
fn map_transfer(value: u32) -> TransferFunction {
    if value == MFVideoTransFunc_709.0 as u32 {
        TransferFunction::Bt709
    } else if value == MFVideoTransFunc_sRGB.0 as u32 {
        TransferFunction::Srgb
    } else if value == MFVideoTransFunc_2084.0 as u32 {
        TransferFunction::Pq
    } else if value == MFVideoTransFunc_HLG.0 as u32 {
        TransferFunction::Hlg
    } else {
        TransferFunction::Unspecified
    }
}

/// Map an `MFVideoTransferMatrix` code to [`MatrixCoefficients`].
fn map_matrix(value: u32) -> MatrixCoefficients {
    if value == MFVideoTransferMatrix_BT709.0 as u32 {
        MatrixCoefficients::Bt709
    } else if value == MFVideoTransferMatrix_BT601.0 as u32 {
        MatrixCoefficients::Bt601
    } else if value == MFVideoTransferMatrix_BT2020_10.0 as u32
        || value == MFVideoTransferMatrix_BT2020_12.0 as u32
    {
        MatrixCoefficients::Bt2020Ncl
    } else {
        MatrixCoefficients::Unspecified
    }
}

/// Map an `MFNominalRange` code to [`ColorRange`]. Only the explicit full-range
/// code (`0_255`) is full; limited and unknown both map to limited (the video
/// norm).
fn map_range(value: u32) -> ColorRange {
    if value == MFNominalRange_0_255.0 as u32 {
        ColorRange::Full
    } else {
        ColorRange::Limited
    }
}

fn open_err(error: windows::core::Error) -> DecodeError {
    DecodeError::Open(error.to_string())
}

fn decode_err(error: windows::core::Error) -> DecodeError {
    DecodeError::Decode(error.to_string())
}

fn seek_err(error: windows::core::Error) -> DecodeError {
    DecodeError::Seek(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_primaries_from_mf_codes() {
        assert_eq!(
            map_primaries(MFVideoPrimaries_BT709.0 as u32),
            ColorPrimaries::Bt709
        );
        assert_eq!(
            map_primaries(MFVideoPrimaries_SMPTE170M.0 as u32),
            ColorPrimaries::Bt601_525
        );
        assert_eq!(
            map_primaries(MFVideoPrimaries_BT470_2_SysBG.0 as u32),
            ColorPrimaries::Bt601_625
        );
        assert_eq!(
            map_primaries(MFVideoPrimaries_BT2020.0 as u32),
            ColorPrimaries::Bt2020
        );
        assert_eq!(
            map_primaries(MFVideoPrimaries_DCI_P3.0 as u32),
            ColorPrimaries::DisplayP3
        );
        assert_eq!(map_primaries(0), ColorPrimaries::Unspecified);
    }

    #[test]
    fn maps_transfer_from_mf_codes() {
        assert_eq!(
            map_transfer(MFVideoTransFunc_709.0 as u32),
            TransferFunction::Bt709
        );
        assert_eq!(
            map_transfer(MFVideoTransFunc_sRGB.0 as u32),
            TransferFunction::Srgb
        );
        assert_eq!(
            map_transfer(MFVideoTransFunc_2084.0 as u32),
            TransferFunction::Pq
        );
        assert_eq!(
            map_transfer(MFVideoTransFunc_HLG.0 as u32),
            TransferFunction::Hlg
        );
        assert_eq!(map_transfer(0), TransferFunction::Unspecified);
    }

    #[test]
    fn maps_matrix_from_mf_codes() {
        assert_eq!(
            map_matrix(MFVideoTransferMatrix_BT709.0 as u32),
            MatrixCoefficients::Bt709
        );
        assert_eq!(
            map_matrix(MFVideoTransferMatrix_BT601.0 as u32),
            MatrixCoefficients::Bt601
        );
        assert_eq!(
            map_matrix(MFVideoTransferMatrix_BT2020_10.0 as u32),
            MatrixCoefficients::Bt2020Ncl
        );
        assert_eq!(
            map_matrix(MFVideoTransferMatrix_BT2020_12.0 as u32),
            MatrixCoefficients::Bt2020Ncl
        );
        assert_eq!(map_matrix(0), MatrixCoefficients::Unspecified);
    }

    #[test]
    fn maps_range_from_mf_codes() {
        assert_eq!(map_range(MFNominalRange_0_255.0 as u32), ColorRange::Full);
        assert_eq!(map_range(2), ColorRange::Limited);
        assert_eq!(map_range(0), ColorRange::Limited);
    }

    #[test]
    fn subtype_maps_to_pixel_format() {
        assert_eq!(
            subtype_to_format(Some(MFVideoFormat_NV12)),
            PixelFormat::Nv12
        );
        assert_eq!(
            subtype_to_format(Some(MFVideoFormat_P010)),
            PixelFormat::P010
        );
        assert_eq!(subtype_to_format(None), PixelFormat::Nv12);
    }

    #[test]
    fn hns_conversion_matches_time_base() {
        // 3 ticks already on the 100-ns base is exactly 3 hns.
        let exact = RationalTime::new(3, Rational::new(HNS_PER_SEC as i32, 1));
        assert_eq!(to_hns(exact), 3);
        // Frame 30 at 30 fps is one second = 1e7 hns.
        let one_second = RationalTime::new(30, Rational::FPS_30);
        assert_eq!(to_hns(one_second), HNS_PER_SEC);
        // Frame 5 at 30 fps: 5/30 s -> 1_666_666 hns (truncated toward zero).
        let frame_5 = RationalTime::new(5, Rational::FPS_30);
        assert_eq!(to_hns(frame_5), 1_666_666);
    }
}
